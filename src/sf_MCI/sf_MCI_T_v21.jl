# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

#=
    sf_MCI_T_v21.jl

    Monte Carlo Integration (MCI/T-approach) for all inefficiency distributions.

    This file provides a UNIFIED implementation for all inefficiency distributions:
    Exponential, HalfNormal, TruncatedNormal, Weibull, Lognormal, Lomax, Rayleigh, Gamma,
    Erlang{2,3,4}.

    The MCI approach computes the likelihood as:
        L(t) = f_noise(ε + sign·u) · f_u(u; params) · J(t)

    where:
        - f_noise: noise PDF (Normal, StudentT, or Laplace)
        - f_u: inefficiency PDF
        - J(t): Jacobian of transformation |du/dt|
        - u = trans(t): transformation from uniform t ∈(0,1) to u ∈(0,∞)

    In log-space for numerical stability:
        log L = log(f_noise) + log(f_u) + log(J)
        LL_i = logsumexp_d(log L_i,d) - log(D)

    Architecture:
        - Single unified NLL function: MCI_nll_mci_T()
        - Single unified JLMS/BC function: _jlms_bc_mci_T()
        - Distribution-specific behavior via type dispatch:
            - get_scale_param(ineff, vals): extract scale parameter
            - log_pdf_ineff!(ineff, out, u, vals, c): compute log-PDF

    Features:
        - GPU compatible (element-wise operations, no scalar indexing)
        - ForwardDiff compatible for automatic differentiation
        - Supports both chunked and non-chunked paths
=#

using SpecialFunctions: erf, erfinv

# Helper function for standard normal CDF (used by TruncatedNormal)
"""Standard normal CDF: Φ(x) = 0.5 × (1 + erf(x/√2))"""
_normcdf(x) = 0.5 * (1 + erf(x / sqrt(oftype(x, 2))))

"""
    _mci_tn_logPhi(a)

log Φ(a), the truncated-normal normaliser at a = μ/σᵤ. The direct form
`log(max(_normcdf(a), 1e-15))` fails as a falls: `1 + erf` cancels, which puts errors of either
sign, up to 0.037 per observation, into the log-likelihood just above a = -7.94, and below that
the floor takes over, so the density is mis-normalised by about a²/2 per observation (at
μ = -0.096 and ln σᵤ² = -10, a = -14.25, the direct form gives an MCI log-likelihood of -28888
on 400 observations against -672.4 for MLE and MSLE). Below a = -1 the stable form
`sf_normlogcdf_lowtail` (src/sf_numerics.jl) is used; at a ≥ -1 the direct form is used
(Φ(a) ≥ 0.158 there, so the floor never binds).

The density's own terms, -z²/2 and -log Φ(a), are each about a²/2 when u is near 0 and still
cancel, by about eps·a²/2 per draw (5e-10 at a = -2115, 0.01 nats near a = -1e7). That is far
inside the region where MCI's draws cover the density poorly (manual, Section 9).
"""
@inline _mci_tn_logPhi(a) = a < -1 ? sf_normlogcdf_lowtail(1.0 * a) : log(_normcdf(a))
# (1.0 * a: log(_normcdf(a)) returns Float64, because _normcdf's 0.5 promotes; the tail
# branch promotes the same way, so the result type does not depend on the branch)

# ============================================================================
# Section 0a: Generic Transformation Rules (Configurable)
# ============================================================================
#
# Four generic transformation rules that can be applied to any distribution.
# The user can select these via the `transformation` keyword in sfmodel_MCI_spec().
# Each rule maps t ∈(0,1) to u ∈(0,∞) using a scale parameter s.

# -----------------------------------------------------------------------------
# :expo_rule - Exponential mapping: u = s * (-log(1-t))
# -----------------------------------------------------------------------------
"""
    expo_rule_trans(t, s)

Exponential mapping transformation: u = s · (-log(1-t)), maps (0,1) ∈(0,∞).
"""
@inline expo_rule_trans(t, s) = s * (-log(1 - t))

"""
    expo_rule_jacob(t, s)

Jacobian of exponential mapping: |du/dt| = s/(1-t).
"""
@inline expo_rule_jacob(t, s) = s / (1 - t)

# -----------------------------------------------------------------------------
# :logistic_1_rule - Logistic mapping: u = s * (t / (1-t))
# -----------------------------------------------------------------------------
"""
    logistic_1_rule_trans(t, s)

Logistic mapping transformation: u = s · t/(1-t), maps (0,1) ∈(0,∞).
"""
@inline logistic_1_rule_trans(t, s) = s * t / (1 - t)

"""
    logistic_1_rule_jacob(t, s)

Jacobian of logistic mapping: |du/dt| = s/(1-t)².
"""
@inline logistic_1_rule_jacob(t, s) = s / (1 - t)^2

# -----------------------------------------------------------------------------
# :logistic_2_rule - Logistic-power mapping: u = s * (t / (1-t))^2
# -----------------------------------------------------------------------------
"""
    logistic_2_rule_trans(t, s)

Logistic-power mapping transformation: u = s · (t/(1-t))², maps (0,1) ∈(0,∞).
"""
@inline logistic_2_rule_trans(t, s) = s * (t / (1 - t))^2

"""
    logistic_2_rule_jacob(t, s)

Jacobian of logistic-power mapping: |du/dt| = 2st/(1-t)³.
"""
@inline logistic_2_rule_jacob(t, s) = 2 * s * t / (1 - t)^3

# ============================================================================
# Section 0b: Transformation Registry
# ============================================================================

"""
Lookup table mapping transformation rule symbols to (trans, jacob) function pairs.
"""
const TRANSFORMATION_RULES = Dict{Symbol, Tuple{Function, Function}}(
    :expo_rule       => (expo_rule_trans, expo_rule_jacob),
    :logistic_1_rule => (logistic_1_rule_trans, logistic_1_rule_jacob),
    :logistic_2_rule => (logistic_2_rule_trans, logistic_2_rule_jacob),
)

"""
    default_transformation_rule(ineff::Symbol)

Get the default transformation rule for a given inefficiency distribution.
"""
function default_transformation_rule(ineff::Symbol)
    if ineff in (:Exponential, :Weibull, :Gamma, :Rayleigh, :Erlang2, :Erlang3, :Erlang4)
        return :expo_rule
    elseif ineff in (:HalfNormal, :TruncatedNormal, :Lognormal)
        return :logistic_1_rule
    elseif ineff == :Lomax
        return :logistic_1_rule
    else
        return :expo_rule
    end
end

# Type-dispatch versions for MCI types (used by unified MCI_nll_mci_T)
default_transformation_rule(::Exponential_MCI) = :expo_rule
default_transformation_rule(::Weibull_MCI) = :expo_rule
default_transformation_rule(::Gamma_MCI) = :expo_rule
default_transformation_rule(::Erlang_MCI) = :expo_rule
default_transformation_rule(::Rayleigh_MCI) = :expo_rule
default_transformation_rule(::HalfNormal_MCI) = :logistic_1_rule
default_transformation_rule(::TruncatedNormal_MCI) = :logistic_1_rule
default_transformation_rule(::Lognormal_MCI) = :logistic_1_rule
default_transformation_rule(::Lomax_MCI) = :logistic_1_rule

"""
    resolve_transformation(rule::Symbol)

Resolve a transformation rule symbol to (trans, jacob) function pair.
"""
function resolve_transformation(rule::Symbol)
    haskey(TRANSFORMATION_RULES, rule) ||
        error("Unknown transformation rule: $rule. Valid: :expo_rule, :logistic_1_rule, :logistic_2_rule")
    return TRANSFORMATION_RULES[rule]
end

# ============================================================================
# Section 1: Log-PDF Functions for Each Inefficiency Distribution
# ============================================================================
#
# All PDF computations are in log-space for numerical stability.
# Arguments are clamped to prevent log(0) singularities.
# These are the core math functions used by log_pdf_ineff!() dispatch.

"""
    log_pdf_ineff_exp(u, lambda, clamp_lo)

Log-PDF of Exponential distribution at u, parameterised by the squared scale σᵤ², which
for this distribution is also Var(u); the scale is √σᵤ².
    rate = 1/√σᵤ²,  f(u; σᵤ²) = (1/√σᵤ²) · exp(-u/√σᵤ²)
    log f = -0.5·log(σᵤ²) - u/√σᵤ²
"""
@inline function log_pdf_ineff_exp(u, sigma_sq, clamp_lo)
    sigma_sq_safe = max(sigma_sq, clamp_lo)
    return -0.5 * log(sigma_sq_safe) - u / sqrt(sigma_sq_safe)
end

"""
    log_pdf_ineff_halfnorm(u, sigma, clamp_lo)

Log-PDF of HalfNormal(σ) distribution at u.
    f(u; σ) = √(2/π) · exp(-u²/(2σ²)) / σ
    log f = 0.5·log(2/π) - log(σ) - 0.5·(u/σ)²
"""
@inline function log_pdf_ineff_halfnorm(u, sigma, clamp_lo)
    sigma_safe = max(sigma, clamp_lo)
    return 0.5 * log(2/π) - log(sigma_safe) - 0.5 * (u/sigma_safe)^2
end

"""
    log_pdf_ineff_truncnorm(u, mu, sigma, log_Phi_ratio, clamp_lo)

Log-PDF of TruncatedNormal(μ, σ; lower=0) distribution at u.
    f(u; μ, σ) = φ((u-μ)/σ) / (σ · Φ(μ/σ))
    log f = -0.5·log(2π) - log(σ) - 0.5·((u-μ)/σ)² - log(Φ(μ/σ))

where log_Phi_ratio = log(Φ(μ/σ)) is precomputed.
"""
@inline function log_pdf_ineff_truncnorm(u, mu, sigma, log_Phi_ratio, clamp_lo)
    sigma_safe = max(sigma, clamp_lo)
    z = (u - mu) / sigma_safe
    return -0.5 * log(2π) - log(sigma_safe) - 0.5 * z^2 - log_Phi_ratio
end

"""
    log_pdf_ineff_weibull(u, lambda, k, clamp_lo)

Log-PDF of Weibull(λ, k) distribution at u.
    f(u; λ, k) = (k/λ) · (u/λ)^(k-1) · exp(-(u/λ)^k)
    log f = log(k) - k·log(λ) + (k-1)·log(u) - (u/λ)^k
"""
@inline function log_pdf_ineff_weibull(u, lambda, k, clamp_lo)
    u_safe = max(u, clamp_lo)
    lambda_safe = max(lambda, clamp_lo)
    return log(k) - k * log(lambda_safe) + (k - 1) * log(u_safe) - (u_safe/lambda_safe)^k
end

"""
    log_pdf_ineff_lognorm(u, mu, sigma, clamp_lo)

Log-PDF of LogNormal(μ, σ) distribution at u.
    f(u; μ, σ) = exp(-(log(u)-μ)²/(2σ²)) / (u·σ·√(2π))
    log f = -0.5·log(2π) - log(σ) - log(u) - 0.5·((log(u)-μ)/σ)²
"""
@inline function log_pdf_ineff_lognorm(u, mu, sigma, clamp_lo)
    u_safe = max(u, clamp_lo)
    sigma_safe = max(sigma, clamp_lo)
    log_u = log(u_safe)
    return -0.5 * log(2π) - log(sigma_safe) - log_u - 0.5 * ((log_u - mu)/sigma_safe)^2
end

"""
    log_pdf_ineff_lomax(u, lambda, alpha, clamp_lo, α_floor)

Log-PDF of Lomax(α, λ) distribution at u.
    f(u; α, λ) = (α/λ) · (1 + u/λ)^{-(α+1)}  for u ≥ 0
    log f = log(α) - log(λ) - (α+1)·log(1 + u/λ)

Uses α_floor for alpha bounds to prevent extreme values.
"""
@inline function log_pdf_ineff_lomax(u, lambda, alpha, clamp_lo, α_floor)
    u_safe = max(u, clamp_lo)      # Lomax: u ≥ 0
    alpha_safe = max(alpha, α_floor)
    lambda_safe = max(lambda, clamp_lo)
    return log(alpha_safe) - log(lambda_safe) - (alpha_safe + 1) * log1p(u_safe / lambda_safe)
end

"""
    log_pdf_ineff_rayleigh(u, sigma, clamp_lo)

Log-PDF of Rayleigh(σ) distribution at u.
    f(u; σ) = (u/σ²) · exp(-u²/(2σ²))
    log f = log(u) - 2·log(σ) - 0.5·(u/σ)²
"""
@inline function log_pdf_ineff_rayleigh(u, sigma, clamp_lo)
    u_safe = max(u, clamp_lo)
    sigma_safe = max(sigma, clamp_lo)
    return log(u_safe) - 2 * log(sigma_safe) - 0.5 * (u_safe/sigma_safe)^2
end

"""
    log_pdf_ineff_gamma(u, k, theta, lgk, clamp_lo)

Log-PDF of Gamma(k, θ) distribution at u.
    f(u; k, θ) = u^(k-1) · exp(-u/θ) / (θ^k · Γ(k))
    log f = (k-1)·log(u) - u/θ - k·log(θ) - loggamma(k)

where lgk = loggamma(k) is precomputed.
"""
@inline function log_pdf_ineff_gamma(u, k, theta, lgk, clamp_lo)
    u_safe = max(u, clamp_lo)
    theta_safe = max(theta, clamp_lo)
    return (k - 1) * log(u_safe) - u / theta_safe - k * log(theta_safe) - lgk
end

# ============================================================================
# Section 1a: Scale Parameter Extraction (dispatch by distribution type)
# ============================================================================
#
# Each distribution uses a different parameter as its "scale" for transformations.
# These dispatch functions extract the appropriate scale parameter.

"""
    get_scale_param(ineff, ineff_vals)

Extract the scale parameter used for transformation from ineff_vals.
Each distribution uses a different parameter as its scale.
"""
get_scale_param(::Exponential_MCI, vals) = sqrt.(vals.sigma_sq)
get_scale_param(::HalfNormal_MCI, vals) = vals.sigma
get_scale_param(::TruncatedNormal_MCI, vals) = vals.sigma_u
get_scale_param(::Weibull_MCI, vals) = vals.lambda
get_scale_param(::Lognormal_MCI, vals) = vals.sigma
# Lomax: scale by the MEDIAN λ(2^(1/α) − 1), not λ. λ alone is far too wide once α is large,
# so almost every draw lands where the integrand is negligible. The MEAN λ/(α−1) is not usable
# as a scale: α is only floored at c.α_floor = 0.1, so the mean is negative or infinite over
# part of the optimizer's admissible region, while the median is positive and finite for every
# admissible α.
get_scale_param(::Lomax_MCI, vals) = vals.lambda .* (exp2.(1 ./ vals.alpha) .- 1)
get_scale_param(::Rayleigh_MCI, vals) = vals.sigma
# Gamma: θ·√(1+k²), observation by observation, as in the panel (mypan_get_mci_scale(::Gamma_Panel)
# in src/sf_panel/sf_panel_v20.jl). The mass of Gamma(k, θ) sits near k·θ: scaling the expo_rule
# proposal by θ alone makes the importance weight proportional to u^(k−1), unbounded and increasing
# for k > 1, so a single draw decides each observation's integral. A scale of k·θ is too narrow for
# k < 1 (and often for k from 1 to 2 when the noise is small): the draws thin out faster than the
# Gamma's own tail exp(−u/θ), so an observation far in the tail gets almost no draws (at k = 0.5 on
# 500 observations, tens to hundreds of nats off in some samples). θ·√(1+k²) is close to k·θ for
# large k and to θ for small k, and smooth in k. The broadcast serves a heteroscedastic k on the CPU
# or a CuArray, and Float32 stays Float32. Gamma with k = M does not reproduce Erlang M (scaled by
# M·θ, below) bit for bit.
get_scale_param(::Gamma_MCI, vals) = vals.theta .* sqrt.(1 .+ vals.k .^ 2)
# Mass of Erlang(M, θ) sits near M·θ — scale the expo_rule draws accordingly
get_scale_param(::Erlang_MCI{M}, vals) where {M} = vals.theta .* M

# ============================================================================
# Section 1b: Log-PDF Computation (dispatch by distribution type)
# ============================================================================
#
# In-place log-PDF computation for each distribution type.
# These functions handle the N×1 parameter vectors properly.

"""
    log_pdf_ineff!(ineff, out, u, ineff_vals, c)

Compute log-PDF of inefficiency distribution in-place.
Dispatches to distribution-specific implementation.

# Arguments
- `ineff`: Distribution type (e.g., Exponential_MCI())
- `out`: Output buffer (N×D matrix)
- `u`: Transformed values (N×D matrix)
- `ineff_vals`: Named tuple of parameter vectors
- `c`: Constants struct with clamp_lo
"""
function log_pdf_ineff!(::Exponential_MCI, out, u, ineff_vals, c)
    sigma_sq = reshape(ineff_vals.sigma_sq, :, 1)
    @. out = log_pdf_ineff_exp(u, sigma_sq, c.clamp_lo)
end

function log_pdf_ineff!(::HalfNormal_MCI, out, u, ineff_vals, c)
    sigma = reshape(ineff_vals.sigma, :, 1)
    @. out = log_pdf_ineff_halfnorm(u, sigma, c.clamp_lo)
end

function log_pdf_ineff!(::TruncatedNormal_MCI, out, u, ineff_vals, c)
    mu = reshape(ineff_vals.mu, :, 1)
    sigma_u = reshape(ineff_vals.sigma_u, :, 1)
    log_Phi = _mci_tn_logPhi.(mu ./ sigma_u)       # log Φ with no floor and no cancellation
    @. out = log_pdf_ineff_truncnorm(u, mu, sigma_u, log_Phi, c.clamp_lo)
end

function log_pdf_ineff!(::Weibull_MCI, out, u, ineff_vals, c)
    lambda = reshape(ineff_vals.lambda, :, 1)
    k = reshape(ineff_vals.k, :, 1)
    @. out = log_pdf_ineff_weibull(u, lambda, k, c.clamp_lo)
end

function log_pdf_ineff!(::Lognormal_MCI, out, u, ineff_vals, c)
    mu = reshape(ineff_vals.mu, :, 1)
    sigma = reshape(ineff_vals.sigma, :, 1)
    @. out = log_pdf_ineff_lognorm(u, mu, sigma, c.clamp_lo)
end

function log_pdf_ineff!(::Lomax_MCI, out, u, ineff_vals, c)
    lambda = reshape(ineff_vals.lambda, :, 1)
    alpha = reshape(ineff_vals.alpha, :, 1)
    @. out = log_pdf_ineff_lomax(u, lambda, alpha, c.clamp_lo, c.α_floor)
end

function log_pdf_ineff!(::Rayleigh_MCI, out, u, ineff_vals, c)
    sigma = reshape(ineff_vals.sigma, :, 1)
    @. out = log_pdf_ineff_rayleigh(u, sigma, c.clamp_lo)
end

function log_pdf_ineff!(::Gamma_MCI, out, u, ineff_vals, c)
    k = reshape(ineff_vals.k, :, 1)
    theta = reshape(ineff_vals.theta, :, 1)
    lgk = reshape(ineff_vals.lgk, :, 1)
    @. out = log_pdf_ineff_gamma(u, k, theta, lgk, c.clamp_lo)
end

# Erlang ineff_vals carries (k, theta, lgk) with the shape pinned at the Erlang
# order, so every Gamma kernel below applies unchanged
log_pdf_ineff!(::Erlang_MCI, out, u, ineff_vals, c) =
    log_pdf_ineff!(Gamma_MCI(), out, u, ineff_vals, c)

# ============================================================================
# Section 1c: Chunked Log-PDF Computation (dispatch by distribution type)
# ============================================================================
#
# Chunked versions for memory-efficient GPU processing.

"""
    log_pdf_ineff_chunk!(ineff, out, u, ineff_vals, c, row_start, row_end)

Chunked version of log_pdf_ineff! for memory-efficient processing.
Extracts the relevant slice of parameter vectors for the current chunk.
"""
function log_pdf_ineff_chunk!(::Exponential_MCI, out, u, ineff_vals, c, rs, re)
    sigma_sq = reshape((@view ineff_vals.sigma_sq[rs:re]), :, 1)
    @. out = log_pdf_ineff_exp(u, sigma_sq, c.clamp_lo)
end

function log_pdf_ineff_chunk!(::HalfNormal_MCI, out, u, ineff_vals, c, rs, re)
    sigma = reshape((@view ineff_vals.sigma[rs:re]), :, 1)
    @. out = log_pdf_ineff_halfnorm(u, sigma, c.clamp_lo)
end

function log_pdf_ineff_chunk!(::TruncatedNormal_MCI, out, u, ineff_vals, c, rs, re)
    mu = reshape((@view ineff_vals.mu[rs:re]), :, 1)
    sigma_u = reshape((@view ineff_vals.sigma_u[rs:re]), :, 1)
    log_Phi = _mci_tn_logPhi.(mu ./ sigma_u)       # log Φ with no floor and no cancellation
    @. out = log_pdf_ineff_truncnorm(u, mu, sigma_u, log_Phi, c.clamp_lo)
end

function log_pdf_ineff_chunk!(::Weibull_MCI, out, u, ineff_vals, c, rs, re)
    lambda = reshape((@view ineff_vals.lambda[rs:re]), :, 1)
    k = reshape((@view ineff_vals.k[rs:re]), :, 1)
    @. out = log_pdf_ineff_weibull(u, lambda, k, c.clamp_lo)
end

function log_pdf_ineff_chunk!(::Lognormal_MCI, out, u, ineff_vals, c, rs, re)
    mu = reshape((@view ineff_vals.mu[rs:re]), :, 1)
    sigma = reshape((@view ineff_vals.sigma[rs:re]), :, 1)
    @. out = log_pdf_ineff_lognorm(u, mu, sigma, c.clamp_lo)
end

function log_pdf_ineff_chunk!(::Lomax_MCI, out, u, ineff_vals, c, rs, re)
    lambda = reshape((@view ineff_vals.lambda[rs:re]), :, 1)
    alpha = reshape((@view ineff_vals.alpha[rs:re]), :, 1)
    @. out = log_pdf_ineff_lomax(u, lambda, alpha, c.clamp_lo, c.α_floor)
end

function log_pdf_ineff_chunk!(::Rayleigh_MCI, out, u, ineff_vals, c, rs, re)
    sigma = reshape((@view ineff_vals.sigma[rs:re]), :, 1)
    @. out = log_pdf_ineff_rayleigh(u, sigma, c.clamp_lo)
end

function log_pdf_ineff_chunk!(::Gamma_MCI, out, u, ineff_vals, c, rs, re)
    k = reshape((@view ineff_vals.k[rs:re]), :, 1)
    theta = reshape((@view ineff_vals.theta[rs:re]), :, 1)
    lgk = reshape((@view ineff_vals.lgk[rs:re]), :, 1)
    @. out = log_pdf_ineff_gamma(u, k, theta, lgk, c.clamp_lo)
end

log_pdf_ineff_chunk!(::Erlang_MCI, out, u, ineff_vals, c, rs, re) =
    log_pdf_ineff_chunk!(Gamma_MCI(), out, u, ineff_vals, c, rs, re)

# ============================================================================
# Section 1d: Inefficiency CDF F_u(u) — copula models only
# ============================================================================
#
# The T-approach draws u = trans(t) via a generic importance transform, so the
# raw Halton draw t is NOT the inefficiency CDF F_u(u) (they coincide only when
# `trans` is the inverse-CDF/quantile). The copula density's second argument must
# be the true F_u(u). Because scaling u = h·u* is a monotone transform,
# F_u(u) = F_{u*}(u*), so these are evaluated on the base draw u* (pre-scaling).
# Each F_u is the exact inverse of the matching quantile function (Section 3).
#
# `_copula_true_Fu` is a trait: true → use the computed CDF F_u(u) as the copula's
# u-side argument. All 8 inefficiency distributions pass the true CDF,
# so MCI + copula estimates the same c(F_v,F_u) model the manual documents (§8)
# as MSLE, and the two agree up to simulation error. For the truncated normal this
# needs the log-space tail of `_mci_tn_cdf_tail` (below): a direct evaluation of F_u
# collapses below μ/σᵤ ≈ -7.5. Gamma uses the AD-safe regularized incomplete gamma below.
_copula_true_Fu(::MCIIneffModel) = true

using ForwardDiff: ForwardDiff, Dual

# The continued fraction stops when its factor is 1 to within tol. Under ForwardDiff the
# factor's partials must have converged too: at an integer shape a the factor is exactly 1
# while its derivative in a is not. Partials are tested at 10 times the tolerance of the
# order below, which stays above the rounding noise of the partials.
# Fits differentiate in Float64 (the coefficients are Float64, also with Float32 data);
# the 10× steps assume Float64 partials.
_cf_converged(d::Real, tol) = abs(d) < tol
_cf_converged(d::Dual, tol) = _cf_converged(ForwardDiff.value(d), tol) &&
    all(i -> _cf_converged(ForwardDiff.partials(d, i), 10 * tol), 1:ForwardDiff.npartials(d))

# AD-safe regularized lower incomplete gamma P(a,x) = γ(a,x)/Γ(a), the Gamma(a,θ)
# CDF at u=θx. Series (gser) for x < a+1, continued fraction for Q (gcf) otherwise;
# both use only ForwardDiff-differentiable primitives, and the continued fraction stops
# only when `_cf_converged` (above) holds for its factor and, under ForwardDiff, for the
# factor's partials, so P and its derivatives in a and in x converge together.
# `lga = loggamma(a)` is passed precomputed (ineff_vals.lgk).
@inline function _reg_inc_gamma_P(a, x, lga)
    x <= zero(x) && return zero(x)
    if x < a + one(a)
        ap  = a
        del = one(x) / a
        s   = del
        for _ in 1:1000
            ap  += one(ap)
            del *= x / ap
            s   += del
            abs(del) < abs(s) * 1e-15 && break
        end
        return s * exp(-x + a * log(x) - lga)
    else
        FPMIN = oftype(x, 1e-30)   # typed to the input eltype so the CF branch stays type-stable (Float32/GPU); 1e-30 is representable in Float32 and the standard Lentz guard
        b = x + one(x) - a
        c = one(x) / FPMIN
        d = one(x) / b
        h = d
        for i in 1:1000
            an = -i * (i - a)
            b += 2 * one(x)
            d  = an * d + b
            abs(d) < FPMIN && (d = oftype(d, FPMIN))
            c  = b + an / c
            abs(c) < FPMIN && (c = oftype(c, FPMIN))
            d  = one(x) / d
            del = d * c
            h  *= del
            _cf_converged(del - one(x), 1e-15) && break
        end
        return one(x) - exp(-x + a * log(x) - lga) * h
    end
end

function ineff_cdf!(::Exponential_MCI, out, u, ineff_vals, c)
    sigma_sq = reshape(ineff_vals.sigma_sq, :, 1)
    @. out = -expm1(-max(u, zero(u)) / sqrt(sigma_sq))          # F = 1 - exp(-u/√σᵤ²)
    return nothing
end
function ineff_cdf!(::HalfNormal_MCI, out, u, ineff_vals, c)
    sigma = reshape(ineff_vals.sigma, :, 1)
    @. out = 2 * _normcdf(max(u, zero(u)) / sigma) - 1          # F = erf(u/(σ√2))
    return nothing
end
# The log-erfcx pieces of `_mci_tn_cdf_tail`; `Dual` (for `_mci_tn_switch`) is imported above `_cf_converged`.
using SpecialFunctions: erfcx
using ..SFNumerics: _sf_logS_erfcx

"""
    _mci_tn_cdf_tail(F, u, mu, sigma_u)

The truncated-normal CDF F_u(u) that the copula reads, in the truncated normal's lower tail:
where μ/σᵤ < -3 (< -1 when -μ/σᵤ is a Float32; see below) it returns F_u formed in log space,
elsewhere `F`, the old expression's value, unchanged. With x0 = -μ/σᵤ and x1 = (u - μ)/σᵤ,
F_u = (Φ(x1) - Φ(x0))/(1 - Φ(x0)), which `ineff_cdf!` and `ineff_cdf_chunk!` form from
Φ = (1 + erf)/2 with the denominator floored at `clamp_lo` (1e-15). As x0 grows, the numerator
and 1 - Φ(x0) both cancel: that F_u is off by about 3e-10 at μ/σᵤ = -5 and 1e-4 at -7, takes a
few coarse values near -8 (0.111, 0.333 and 0.666 for a true 0.1, 0.5 and 0.9), and is 0 for
every draw from about -8.3. The copula log-likelihood of the Gaussian, Clayton, Clayton90 and
Gumbel models is then off by hundreds of nats: too high near -8, too low below.

In the tail this returns 1 - Q(x1)/Q(x0), Q the normal upper tail, formed in log space. With
y0 = x0/√2, t = u/(σᵤ√2) and erfc(y) = erfcx(y)·exp(-y²),

    1 - F_u = exp(-E),   E = t(2y0 + t) + log erfcx(y0) - log erfcx(y0 + t),

and F_u = -expm1(-E), with no floor. Beyond y0 = 28 (μ/σᵤ < -39.6) the log-erfcx difference is
log1p(t/y0) + log S(y0) - log S(y0 + t), from the asymptotic series of erfcx (`_sf_logS_erfcx`,
src/sf_numerics.jl, as in `sf_tn_quantile_tail`): through erfcx, ForwardDiff's rule
erfcx'(x) = 2x·erfcx(x) - 2/√π cancels as x grows.

Where the switch is (`_mci_tn_switch`). With -μ/σᵤ a Float64, or a ForwardDiff Dual of one (what
every fit uses, also with Float32 data, since the coefficients are Float64), the old expression's
error in F_u is at most about 7e-16 at μ/σᵤ ≥ -1, 4e-15 at -2 and 1e-13 near -3, then 3e-12
at -4 and 3e-10 at -5. On 300 observations the old and new MCI log-likelihoods differ by at
most 5e-9 just below -3, and their gradients by at most 8e-8, against 4e-7 at -4 (Newton's
default g_abstol is 1e-7). So the switch is at -3, and a fit whose optimizer only dips below -1
gives the same numbers as before, bit for bit. With -μ/σᵤ a Float32 (Float32 coefficients), the
old expression evaluates erf in Float32 and is off by up to 6e-7 already at -1 (2.4e-6 at -2,
5.4e-5 at -3), so there the switch stays at -1.

The kernels keep the old broadcast as it was and call this in a second pass, so that above the
switch the values and their ForwardDiff derivatives stay the same bit for bit on the GPU as
well: the GPU compiler fuses a*b + c into one fma, so the old formula compiled into a different
kernel could round differently in the last bit. The tail is formed in Float64 (the `1.0 *` and
the Float64 constant √2) and rounded to the type of `F`, so it is correctly rounded (at most
0.5 ulp) for Float32 as well. The old expression also returns a Float64 for Float32 arguments
(its 0.5 promotes), but its erf runs in Float32, so with Float32 arguments F_u can jump by up to
about 6e-7 at the switch.
"""
@inline function _mci_tn_cdf_tail(F, u, mu, sigma_u)
    r = -mu / sigma_u
    r > _mci_tn_switch(typeof(r)) || return F
    return oftype(F, _mci_tn_cdf_lowtail(1.0 * max(u, zero(u)), 1.0 * mu, 1.0 * sigma_u))
end

# -μ/σᵤ above which _mci_tn_cdf_tail replaces the old expression: 3 when -μ/σᵤ is a Float64 (or
# a Dual, nested or not, of one), 1 when it is a Float32; see the docstring of _mci_tn_cdf_tail
_mci_tn_switch(::Type{<:Dual{T, V}}) where {T, V} = _mci_tn_switch(V)
_mci_tn_switch(::Type{Float32}) = 1
_mci_tn_switch(::Type) = 3

# F_u in the lower tail (see _mci_tn_cdf_tail); u ≥ 0
@inline function _mci_tn_cdf_lowtail(u, mu, sigma_u)
    s = sigma_u * 1.4142135623730951          # σᵤ√2
    y0 = -mu / s
    t = u / s
    if y0 > 28
        E = t * (2y0 + t) + (log1p(t / y0) + (_sf_logS_erfcx(y0) - _sf_logS_erfcx(y0 + t)))
    else
        E = t * (2y0 + t) + (log(erfcx(y0)) - log(erfcx(y0 + t)))
    end
    return -expm1(-E)
end

function ineff_cdf!(::TruncatedNormal_MCI, out, u, ineff_vals, c)
    mu = reshape(ineff_vals.mu, :, 1)
    sigma_u = reshape(ineff_vals.sigma_u, :, 1)
    @. out = (_normcdf((max(u, zero(u)) - mu) / sigma_u) - _normcdf(-mu / sigma_u)) /
             max(1 - _normcdf(-mu / sigma_u), c.clamp_lo)
    @. out = _mci_tn_cdf_tail(out, u, mu, sigma_u)      # log-space tail: μ/σᵤ < -3 (Float32: < -1)
    return nothing
end
function ineff_cdf!(::Weibull_MCI, out, u, ineff_vals, c)
    lambda = reshape(ineff_vals.lambda, :, 1)
    k = reshape(ineff_vals.k, :, 1)
    @. out = -expm1(-(max(u, zero(u)) / lambda)^k)             # F = 1 - exp(-(u/λ)^k)
    return nothing
end
function ineff_cdf!(::Lognormal_MCI, out, u, ineff_vals, c)
    mu = reshape(ineff_vals.mu, :, 1)
    sigma = reshape(ineff_vals.sigma, :, 1)
    @. out = _normcdf((log(max(u, c.clamp_lo)) - mu) / sigma)   # F = Φ((ln u − μ)/σ)
    return nothing
end
function ineff_cdf!(::Lomax_MCI, out, u, ineff_vals, c)
    lambda = reshape(ineff_vals.lambda, :, 1)
    alpha = reshape(ineff_vals.alpha, :, 1)
    @. out = -expm1(-alpha * log1p(max(u, zero(u)) / lambda))   # F = 1 - (1+u/λ)^(-α)
    return nothing
end
function ineff_cdf!(::Rayleigh_MCI, out, u, ineff_vals, c)
    sigma = reshape(ineff_vals.sigma, :, 1)
    @. out = -expm1(-max(u, zero(u))^2 / (2 * sigma^2))         # F = 1 - exp(-u²/2σ²)
    return nothing
end
function ineff_cdf!(::Gamma_MCI, out, u, ineff_vals, c)
    k = reshape(ineff_vals.k, :, 1)
    theta = reshape(ineff_vals.theta, :, 1)
    lgk = reshape(ineff_vals.lgk, :, 1)
    @. out = _reg_inc_gamma_P(k, max(u, zero(u)) / max(theta, c.clamp_lo), lgk)  # F = P(k, u/θ)
    return nothing
end
# Erlang CDF is the regularized incomplete gamma at the pinned integer shape
ineff_cdf!(::Erlang_MCI, out, u, ineff_vals, c) =
    ineff_cdf!(Gamma_MCI(), out, u, ineff_vals, c)

# Chunked variants (slice parameter vectors for the current chunk).
function ineff_cdf_chunk!(::Exponential_MCI, out, u, ineff_vals, c, rs, re)
    sigma_sq = reshape((@view ineff_vals.sigma_sq[rs:re]), :, 1)
    @. out = -expm1(-max(u, zero(u)) / sqrt(sigma_sq))
    return nothing
end
function ineff_cdf_chunk!(::HalfNormal_MCI, out, u, ineff_vals, c, rs, re)
    sigma = reshape((@view ineff_vals.sigma[rs:re]), :, 1)
    @. out = 2 * _normcdf(max(u, zero(u)) / sigma) - 1
    return nothing
end
function ineff_cdf_chunk!(::TruncatedNormal_MCI, out, u, ineff_vals, c, rs, re)
    mu = reshape((@view ineff_vals.mu[rs:re]), :, 1)
    sigma_u = reshape((@view ineff_vals.sigma_u[rs:re]), :, 1)
    @. out = (_normcdf((max(u, zero(u)) - mu) / sigma_u) - _normcdf(-mu / sigma_u)) /
             max(1 - _normcdf(-mu / sigma_u), c.clamp_lo)
    @. out = _mci_tn_cdf_tail(out, u, mu, sigma_u)      # log-space tail: μ/σᵤ < -3 (Float32: < -1)
    return nothing
end
function ineff_cdf_chunk!(::Weibull_MCI, out, u, ineff_vals, c, rs, re)
    lambda = reshape((@view ineff_vals.lambda[rs:re]), :, 1)
    k = reshape((@view ineff_vals.k[rs:re]), :, 1)
    @. out = -expm1(-(max(u, zero(u)) / lambda)^k)
    return nothing
end
function ineff_cdf_chunk!(::Lognormal_MCI, out, u, ineff_vals, c, rs, re)
    mu = reshape((@view ineff_vals.mu[rs:re]), :, 1)
    sigma = reshape((@view ineff_vals.sigma[rs:re]), :, 1)
    @. out = _normcdf((log(max(u, c.clamp_lo)) - mu) / sigma)
    return nothing
end
function ineff_cdf_chunk!(::Lomax_MCI, out, u, ineff_vals, c, rs, re)
    lambda = reshape((@view ineff_vals.lambda[rs:re]), :, 1)
    alpha = reshape((@view ineff_vals.alpha[rs:re]), :, 1)
    @. out = -expm1(-alpha * log1p(max(u, zero(u)) / lambda))
    return nothing
end
function ineff_cdf_chunk!(::Rayleigh_MCI, out, u, ineff_vals, c, rs, re)
    sigma = reshape((@view ineff_vals.sigma[rs:re]), :, 1)
    @. out = -expm1(-max(u, zero(u))^2 / (2 * sigma^2))
    return nothing
end
function ineff_cdf_chunk!(::Gamma_MCI, out, u, ineff_vals, c, rs, re)
    k = reshape((@view ineff_vals.k[rs:re]), :, 1)
    theta = reshape((@view ineff_vals.theta[rs:re]), :, 1)
    lgk = reshape((@view ineff_vals.lgk[rs:re]), :, 1)
    @. out = _reg_inc_gamma_P(k, max(u, zero(u)) / max(theta, c.clamp_lo), lgk)
    return nothing
end
ineff_cdf_chunk!(::Erlang_MCI, out, u, ineff_vals, c, rs, re) =
    ineff_cdf_chunk!(Gamma_MCI(), out, u, ineff_vals, c, rs, re)

# ============================================================================
# Section 2: Unified Workspace Allocation
# ============================================================================

"""
    make_mci_workspace(ε, P, n_obs, D, chunks)

Preallocate workspace buffers for MCI T-approach (all distributions).
One function serves every inefficiency distribution.

# Returns
Named tuple with:
- `u`: Buffer for transformed values (rows × D)
- `log_w`: Buffer for log-likelihoods (rows × D)

where rows = n_obs if chunks == 1, else cld(n_obs, chunks).
"""
function make_mci_workspace(ε, ::Type{P}, n_obs, D, chunks; has_copula::Bool=false) where {P}
    rows = chunks == 1 ? n_obs : cld(n_obs, chunks)
    ws = (u = similar(ε, P, rows, D), log_w = similar(ε, P, rows, D))
    if has_copula
        return merge(ws, (copula_adj = similar(ε, P, rows, D), Fv_buf = similar(ε, P, rows, D),
                          Fu_buf = similar(ε, P, rows, D)))
    end
    return ws
end

# ============================================================================
# Section 3: Unified MCI NLL Function
# ============================================================================

# Helper to slice draws for chunked computation (handles both 1×D and N×D cases)
# When draws is 1×D (broadcast mode), returns unchanged
# When draws is N×D (multiRand mode), returns rows for current chunk
@inline function _slice_draws_chunk(draws, row_start, row_end)
    return size(draws, 1) > 1 ? (@view draws[row_start:row_end, :]) : draws
end

# A host copy of the per-observation values of one chunk (the array itself when it is on the host already).
_host_vec(x::Array) = x
_host_vec(x) = Array(x)

"""
    MCI_nll_mci_T(ε, draws_1D, model, c, n_obs, D, frontier_sign, chunks,
                  noise_vals, ineff_vals; workspace=nothing, trans, jacob)

Compute negative log-likelihood using MCI T-approach for ANY inefficiency distribution.

One function serves every inefficiency distribution, by type dispatch.

# The MCI Likelihood
The likelihood at each draw is:
    L(t) = f_noise(ε + sign·u) · f_u(u; params) · J(t)

where:
- u = trans(t, scale) transforms uniform t ∈(0,1) to inefficiency u ∈(0,∞)
- J = jacob(t, scale) is the Jacobian |du/dt|
- f_noise is the noise PDF (Normal, StudentT, or Laplace)
- f_u is the inefficiency PDF (dispatched by distribution type)

In log-space:
    log L = log(f_noise) + log(f_u) + log(J)
    LL_i = logsumexp_d(log L_{i,d}) - log(D)

# Arguments
- `ε::AbstractVector{P}`: Residuals (N observations)
- `draws_1D`: MCI draws as 1×D matrix
- `model`: Model struct with `noise` and `ineff` fields
- `c`: Constants struct with numerical stability parameters
- `n_obs`: Number of observations
- `D`: Number of MCI draws
- `frontier_sign`: +1 for production, -1 for cost frontier
- `chunks`: Number of chunks for memory management
- `noise_vals`: Named tuple of noise parameters
- `ineff_vals`: Named tuple of inefficiency parameters

# Keyword Arguments
- `workspace=nothing`: Pre-allocated buffers (from make_mci_workspace)
- `trans`: Transformation function (t, scale) → u
- `jacob`: Jacobian function (t, scale) → |du/dt|
- `per_obs=false`: return the per-observation log-likelihoods LL_i (not negated), a host vector of
  length N, instead of the negative log-likelihood. They are computed in the chunk loop for every
  value of `chunks` (with `chunks = 1` the loop has one chunk of N rows); their sum equals minus the
  negative log-likelihood up to the order of summation.

# Returns
Negative log-likelihood (scalar); with `per_obs = true`, the vector of LL_i.
"""
function MCI_nll_mci_T(ε::AbstractVector{P}, draws_1D, model, c, n_obs, D,
                        frontier_sign, chunks, noise_vals, ineff_vals;
                        workspace=nothing,
                        trans, jacob,
                        copula_vals=NamedTuple(),
                        scaling_h=nothing,
                        per_obs::Bool=false) where {P<:Real}

    # Get scale parameter for transformation (dispatch on distribution type)
    scale = get_scale_param(model.ineff, ineff_vals)

    # Handle scalar vs vector scale (Lomax has vector lambda)
    # Use scalar directly for broadcast instead of fill() to avoid CPU Array on GPU
    scale_bcast = scale isa Number ? P(scale) : reshape(scale, n_obs, 1)

    has_copula = copula_plen(model.copula) > 0

    if chunks == 1 && !per_obs
        # ----------------------------------------------------------------
        # Non-chunked path: process all observations at once
        # ----------------------------------------------------------------
        u_buffer = workspace === nothing ? similar(ε, P, n_obs, D) : workspace.u
        log_lik_buffer = workspace === nothing ? similar(ε, P, n_obs, D) : workspace.log_w

        ε_N1 = reshape(ε, n_obs, 1)

        # Component 1: Transform t ??u
        @. u_buffer = trans(draws_1D, scale_bcast)
        @. u_buffer = min(u_buffer, 1/c.clamp_lo)

        # Component 2: Log inefficiency PDF (dispatch on distribution type)
        log_pdf_ineff!(model.ineff, log_lik_buffer, u_buffer, ineff_vals, c)

        # Component 3: Log Jacobian
        @. log_lik_buffer = log_lik_buffer + log(max(jacob(draws_1D, scale_bcast), c.clamp_lo))

        # Copula: the copula density's second argument is the inefficiency CDF F_u(u*),
        # computed on the base draw before scaling and the composite error
        # (F_u(u) = F_{u*}(u*), since scaling is monotone); every inefficiency family
        # supplies it, the Gamma and Erlang through the regularized incomplete gamma
        # `_reg_inc_gamma_P`.
        Fu_src = draws_1D
        if has_copula && _copula_true_Fu(model.ineff)
            Fu_buf = workspace === nothing ? similar(ε, P, n_obs, D) : workspace.Fu_buf
            ineff_cdf!(model.ineff, Fu_buf, u_buffer, ineff_vals, c)
            Fu_src = Fu_buf
        end

        # Scaling: multiply u* by h_i AFTER log_pdf and log_Jacobian, BEFORE composite error
        if scaling_h !== nothing
            u_buffer .= u_buffer .* scaling_h
        end

        # Component 4: Composite error z = ε + sign · u
        @. u_buffer = ε_N1 + frontier_sign * u_buffer

        # Component 4b: Copula adjustment (before log_noise_pdf! overwrites u_buffer)
        if has_copula
            copula_adj = workspace === nothing ? similar(ε, P, n_obs, D) : workspace.copula_adj
            Fv_buf = workspace === nothing ? similar(ε, P, n_obs, D) : workspace.Fv_buf
            copula_log_adjustment!(model.copula, copula_adj, Fv_buf, u_buffer, Fu_src,
                                   model.noise, noise_vals, copula_vals, c)
        end

        # Component 5: Log noise PDF (dispatch on noise type)
        log_noise_pdf!(model.noise, u_buffer, u_buffer, noise_vals, c)
        @. log_lik_buffer = log_lik_buffer + u_buffer

        # Add copula log-density
        if has_copula
            @. log_lik_buffer = log_lik_buffer + copula_adj
        end

        # Aggregate: LL_i = logsumexp_d(log L_i,d) - log(D)
        log_likes = logsumexp_rows(log_lik_buffer) .- log(P(D))
        return -_sum_scalar(log_likes)
    else
        # ----------------------------------------------------------------
        # Chunked path: process observations in chunks for memory efficiency
        # ----------------------------------------------------------------
        chunk_size = cld(n_obs, chunks)
        u_buffer = workspace === nothing ? similar(ε, P, chunk_size, D) : workspace.u
        log_lik_buffer = workspace === nothing ? similar(ε, P, chunk_size, D) : workspace.log_w
        copula_adj_buf = has_copula ? (workspace === nothing ? similar(ε, P, chunk_size, D) : workspace.copula_adj) : log_lik_buffer
        Fv_adj_buf = has_copula ? (workspace === nothing ? similar(ε, P, chunk_size, D) : workspace.Fv_buf) : log_lik_buffer
        Fu_adj_buf = has_copula ? (workspace === nothing ? similar(ε, P, chunk_size, D) : workspace.Fu_buf) : log_lik_buffer
        # The log-sum-exp temporary, allocated once per evaluation (not anew for every chunk
        # inside logsumexp_rows); see _logsumexp_rows_tmp!.
        lse_tmp = similar(ε, P, chunk_size, D)
        chunk_nlls = zeros(P, chunks)  # zero-init: empty trailing chunks skip their slot (`continue`); sum must not read undef
        # With `per_obs`: the log-likelihood of each observation, copied to the host chunk by chunk.
        ll_obs = per_obs ? Vector{P}(undef, n_obs) : nothing

        for chunk_idx in 1:chunks
            row_start = (chunk_idx - 1) * chunk_size + 1
            row_end = min(chunk_idx * chunk_size, n_obs)
            row_start > row_end && continue

            chunk_N = row_end - row_start + 1
            u_buf = @view u_buffer[1:chunk_N, :]
            log_lik_buf = @view log_lik_buffer[1:chunk_N, :]

            ε_chunk = reshape((@view ε[row_start:row_end]), chunk_N, 1)
            # Use scalar directly for broadcast instead of fill() to avoid CPU Array on GPU
            scale_chunk = scale isa Number ? P(scale) :
                          reshape((@view scale[row_start:row_end]), chunk_N, 1)
            # Slice draws for this chunk (handles both 1×D and N×D cases)
            draws_chunk = _slice_draws_chunk(draws_1D, row_start, row_end)

            # Same 5 components as non-chunked path
            @. u_buf = trans(draws_chunk, scale_chunk)
            @. u_buf = min(u_buf, 1/c.clamp_lo)
            log_pdf_ineff_chunk!(model.ineff, log_lik_buf, u_buf, ineff_vals, c, row_start, row_end)
            @. log_lik_buf = log_lik_buf + log(max(jacob(draws_chunk, scale_chunk), c.clamp_lo))

            # Copula: true F_u(u*) on the base draw before scaling.
            Fu_src = draws_chunk
            if has_copula && _copula_true_Fu(model.ineff)
                Fu_buf_c = @view Fu_adj_buf[1:chunk_N, :]
                ineff_cdf_chunk!(model.ineff, Fu_buf_c, u_buf, ineff_vals, c, row_start, row_end)
                Fu_src = Fu_buf_c
            end

            # Scaling: multiply u* by h_i AFTER log_pdf and log_Jacobian, BEFORE composite error
            if scaling_h !== nothing
                u_buf .= u_buf .* (@view scaling_h[row_start:row_end])
            end

            @. u_buf = ε_chunk + frontier_sign * u_buf

            # Copula adjustment (before log_noise_pdf overwrites u_buf)
            if has_copula
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                Fv_buf = @view Fv_adj_buf[1:chunk_N, :]
                copula_log_adjustment!(model.copula, copula_adj, Fv_buf, u_buf, Fu_src,
                                       model.noise, noise_vals, copula_vals, c)
            end

            log_noise_pdf_chunk!(model.noise, u_buf, u_buf, noise_vals, c, row_start, row_end)
            @. log_lik_buf = log_lik_buf + u_buf

            # Add copula log-density
            if has_copula
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                @. log_lik_buf = log_lik_buf + copula_adj
            end

            # A shorter last chunk gets a temporary of its own exact size
            lse_buf = chunk_N == chunk_size ? lse_tmp : similar(ε, P, chunk_N, D)
            log_likes_chunk = _logsumexp_rows_tmp!(lse_buf, log_lik_buf) .- log(P(D))
            chunk_nlls[chunk_idx] = _sum_scalar(log_likes_chunk)
            per_obs && copyto!(ll_obs, row_start, _host_vec(log_likes_chunk), 1, chunk_N)
        end

        return per_obs ? ll_obs : -sum(chunk_nlls)
    end
end

# ============================================================================
# Section 4: Unified JLMS/BC Function
# ============================================================================

# The per-observation effective sample size of the jlms/bc importance weights,
# ESS_i = (Σ_d w_id)² / Σ_d w_id². With Halton draws small values also occur when
# the integrals are accurate, so the warning it triggers (below 10) flags a
# potential numerical issue and does not establish that the fit is affected; the
# integration check (the `integration_check` option of `sfmodel_fit`) tells
# whether it is. Values are never altered.
const _JLMS_ESS_WARN_MIN = 10.0

"Print the MCI warning when any observation's jlms/bc effective sample size is below 10; return the count."
function _warn_low_ess(ess::AbstractVector, D::Integer)
    n_low = Int(sum(ess .< _JLMS_ESS_WARN_MIN))
    n_low == 0 && return n_low
    n = length(ess)
    printstyled("Warning (MCI): For $n_low of the $n observations, the simulated likelihood and the JLMS " *
                "and BC indices are effectively computed from fewer than 10 of the $D draws (effective " *
                "sample size below 10; smallest $(round(Float64(minimum(ess)), sigdigits = 3))). This " *
                "warning flags a potential numerical issue; it does not establish that this fit is " *
                "affected. To check this fit, use sfmodel_fit(..., integration_check = true); if the " *
                "check reports CAUTION or FAIL, increase n_draws, try transformation = " *
                ":logistic_2_rule, or refit with method = :MSLE or :MLE where the model allows it.\n"; color = :yellow)
    return n_low
end

"""
    _jlms_bc_mci_T(ε, draws_1D, model, c, n_obs, D, sign, chunks, noise_vals, ineff_vals;
                   trans, jacob)

Compute JLMS and BC efficiency indices using MCI T-approach for ANY inefficiency distribution.

One function serves every inefficiency distribution, by type dispatch.

# Mathematical Definitions
Given the composed error ε = v - u, where v is noise and u ≥ 0 is inefficiency:

- **JLMS (Jondrow et al. 1982):** E(u|ε) = [∫ u·L(t) dt] / [∫ L(t) dt]
- **BC (Battese & Coelli 1988):** E(e^{-u}|ε) = [∫ e^{-u}·L(t) dt] / [∫ L(t) dt]

where L(t) = f_noise(ε + sign·u) · f_u(u) · J(t) is the likelihood at draw t.

# Arguments
Same as MCI_nll_mci_T.

# Returns
Named tuple with:
- `jlms::Vector`: E(u|ε) for each observation (inefficiency index)
- `bc::Vector`: E(e^{-u}|ε) for each observation (efficiency index)
- `likelihood::Vector`: f_ε(ε) for each observation (density value)
"""
function _jlms_bc_mci_T(ε::AbstractVector{P}, draws_1D, model, c, n_obs, D,
                         sign, chunks, noise_vals, ineff_vals;
                         trans, jacob,
                         copula_vals=NamedTuple(),
                         scaling_h=nothing) where {P<:Real}

    # Get scale parameter for transformation
    scale = get_scale_param(model.ineff, ineff_vals)
    # Use scalar directly for broadcast instead of fill() to avoid CPU Array on GPU
    scale_bcast = scale isa Number ? P(scale) : reshape(scale, n_obs, 1)

    if chunks == 1
        # ----------------------------------------------------------------
        # Non-chunked path
        # ----------------------------------------------------------------
        u_buffer = similar(ε, P, n_obs, D)
        log_w_buffer = similar(ε, P, n_obs, D)
        z_buffer = similar(ε, P, n_obs, D)

        ε_N1 = reshape(ε, n_obs, 1)

        # Transform and compute log weights
        @. u_buffer = trans(draws_1D, scale_bcast)
        @. u_buffer = min(u_buffer, 1/c.clamp_lo)
        log_pdf_ineff!(model.ineff, log_w_buffer, u_buffer, ineff_vals, c)
        @. log_w_buffer = log_w_buffer + log(max(jacob(draws_1D, scale_bcast), c.clamp_lo))

        # Copula: true F_u(u*) on the base draw before scaling.
        has_copula = copula_plen(model.copula) > 0
        Fu_src = draws_1D
        if has_copula && _copula_true_Fu(model.ineff)
            Fu_buf = similar(ε, P, n_obs, D)
            ineff_cdf!(model.ineff, Fu_buf, u_buffer, ineff_vals, c)
            Fu_src = Fu_buf
        end

        # Scaling: multiply u* by h_i AFTER log_pdf and log_Jacobian, BEFORE composite error
        if scaling_h !== nothing
            u_buffer .= u_buffer .* scaling_h
        end

        @. z_buffer = ε_N1 + sign * u_buffer

        # Copula adjustment (must run before log_noise_pdf! overwrites z_buffer)
        if has_copula
            copula_adj = similar(ε, P, n_obs, D)
            Fv_buf = similar(ε, P, n_obs, D)
            copula_log_adjustment!(model.copula, copula_adj, Fv_buf, z_buffer, Fu_src,
                                   model.noise, noise_vals, copula_vals, c)
        end

        log_noise_pdf!(model.noise, z_buffer, z_buffer, noise_vals, c)
        @. z_buffer = z_buffer + log_w_buffer
        if has_copula
            @. z_buffer = z_buffer + copula_adj
        end

        # Compute indices
        log_denom = logsumexp_rows(z_buffer) .- log(P(D))
        @. log_w_buffer = log(max(u_buffer, c.clamp_lo)) + z_buffer
        log_jlms_num = logsumexp_rows(log_w_buffer) .- log(P(D))
        @. log_w_buffer = -u_buffer + z_buffer
        log_bc_num = logsumexp_rows(log_w_buffer) .- log(P(D))

        # Effective-sample-size warning (skipped under AD dual numbers; no effect
        # on returned values). ESS = exp(2·logΣw − logΣw²), logΣw = log_denom + log D.
        if P <: AbstractFloat
            @. log_w_buffer = 2 * z_buffer
            ess = exp.(2 .* (log_denom .+ log(P(D))) .- logsumexp_rows(log_w_buffer))
            _warn_low_ess(ess, D)
        end

        return (jlms=exp.(log_jlms_num .- log_denom),
                bc=exp.(log_bc_num .- log_denom),
                likelihood=exp.(log_denom))
    else
        # ----------------------------------------------------------------
        # Chunked path
        # ----------------------------------------------------------------
        chunk_size = cld(n_obs, chunks)
        jlms_out = similar(ε, P, n_obs)
        bc_out = similar(ε, P, n_obs)
        likelihood_out = similar(ε, P, n_obs)
        u_buffer = similar(ε, P, chunk_size, D)
        log_w_buffer = similar(ε, P, chunk_size, D)
        z_buffer = similar(ε, P, chunk_size, D)
        # Effective-sample-size warning accumulator (diagnostics only; skipped
        # under AD dual numbers)
        ess_out = P <: AbstractFloat ? similar(ε, P, n_obs) : nothing

        # Pre-allocate copula buffers if needed
        has_copula_chunked = copula_plen(model.copula) > 0
        copula_adj_buf = has_copula_chunked ? similar(ε, P, chunk_size, D) : z_buffer
        Fv_adj_buf = has_copula_chunked ? similar(ε, P, chunk_size, D) : z_buffer
        Fu_adj_buf = has_copula_chunked ? similar(ε, P, chunk_size, D) : z_buffer

        for chunk_idx in 1:chunks
            row_start = (chunk_idx - 1) * chunk_size + 1
            row_end = min(chunk_idx * chunk_size, n_obs)
            row_start > row_end && continue

            chunk_N = row_end - row_start + 1
            u_buf = @view u_buffer[1:chunk_N, :]
            lw_buf = @view log_w_buffer[1:chunk_N, :]
            z_buf = @view z_buffer[1:chunk_N, :]

            ε_chunk = reshape((@view ε[row_start:row_end]), chunk_N, 1)
            # Use scalar directly for broadcast instead of fill() to avoid CPU Array on GPU
            scale_chunk = scale isa Number ? P(scale) :
                          reshape((@view scale[row_start:row_end]), chunk_N, 1)
            # Slice draws for this chunk (handles both 1×D and N×D cases)
            draws_chunk = _slice_draws_chunk(draws_1D, row_start, row_end)

            # Transform and compute log weights
            @. u_buf = trans(draws_chunk, scale_chunk)
            @. u_buf = min(u_buf, 1/c.clamp_lo)
            log_pdf_ineff_chunk!(model.ineff, lw_buf, u_buf, ineff_vals, c, row_start, row_end)
            @. lw_buf = lw_buf + log(max(jacob(draws_chunk, scale_chunk), c.clamp_lo))

            # Copula: true F_u(u*) on the base draw before scaling.
            Fu_src = draws_chunk
            if has_copula_chunked && _copula_true_Fu(model.ineff)
                Fu_buf_c = @view Fu_adj_buf[1:chunk_N, :]
                ineff_cdf_chunk!(model.ineff, Fu_buf_c, u_buf, ineff_vals, c, row_start, row_end)
                Fu_src = Fu_buf_c
            end

            # Scaling: multiply u* by h_i AFTER log_pdf and log_Jacobian, BEFORE composite error
            if scaling_h !== nothing
                u_buf .= u_buf .* (@view scaling_h[row_start:row_end])
            end

            @. z_buf = ε_chunk + sign * u_buf

            # Copula adjustment (must run before log_noise_pdf_chunk! overwrites z_buf)
            if has_copula_chunked
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                Fv_buf = @view Fv_adj_buf[1:chunk_N, :]
                copula_log_adjustment!(model.copula, copula_adj, Fv_buf, z_buf, Fu_src,
                                       model.noise, noise_vals, copula_vals, c)
            end

            log_noise_pdf_chunk!(model.noise, z_buf, z_buf, noise_vals, c, row_start, row_end)
            @. z_buf = z_buf + lw_buf
            if has_copula_chunked
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                @. z_buf = z_buf + copula_adj
            end

            # Compute indices for this chunk
            log_denom_chunk = logsumexp_rows(z_buf) .- log(P(D))
            @. lw_buf = log(max(u_buf, c.clamp_lo)) + z_buf
            log_jlms_chunk = logsumexp_rows(lw_buf) .- log(P(D))
            @. lw_buf = -u_buf + z_buf
            log_bc_chunk = logsumexp_rows(lw_buf) .- log(P(D))

            # Store results
            jlms_out[row_start:row_end] .= exp.(log_jlms_chunk .- log_denom_chunk)
            bc_out[row_start:row_end] .= exp.(log_bc_chunk .- log_denom_chunk)
            likelihood_out[row_start:row_end] .= exp.(log_denom_chunk)

            # Effective-sample-size warning for this chunk (values unchanged)
            if ess_out !== nothing
                @. lw_buf = 2 * z_buf
                ess_out[row_start:row_end] .= exp.(2 .* (log_denom_chunk .+ log(P(D))) .-
                                                   logsumexp_rows(lw_buf))
            end
        end
        ess_out === nothing || _warn_low_ess(ess_out, D)

        return (jlms=jlms_out, bc=bc_out, likelihood=likelihood_out)
    end
end
