# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

#=
    sf_MCI_T_v18.jl

    Monte Carlo Integration (MCI/T-approach) for all inefficiency distributions.

    This file provides a UNIFIED implementation for all inefficiency distributions:
    Exponential, HalfNormal, TruncatedNormal, Weibull, Lognormal, Lomax, Rayleigh, Gamma.

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
        - Modular design following temp_MCI_example.jl pattern
=#

using SpecialFunctions: erf, erfinv

# Helper function for standard normal CDF (used by TruncatedNormal)
"""Standard normal CDF: Φ(x) = 0.5 × (1 + erf(x/√2))"""
_normcdf(x) = 0.5 * (1 + erf(x / sqrt(oftype(x, 2))))

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
    if ineff in (:Exponential, :Weibull, :Gamma, :Rayleigh)
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

Log-PDF of Exponential distribution at u, where λ = Var(u).
    rate = 1/√λ,  f(u; λ) = (1/√λ) · exp(-u/√λ)
    log f = -0.5·log(λ) - u/√λ
"""
@inline function log_pdf_ineff_exp(u, lambda, clamp_lo)
    lambda_safe = max(lambda, clamp_lo)
    return -0.5 * log(lambda_safe) - u / sqrt(lambda_safe)
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
get_scale_param(::Exponential_MCI, vals) = sqrt.(vals.lambda)
get_scale_param(::HalfNormal_MCI, vals) = vals.sigma
get_scale_param(::TruncatedNormal_MCI, vals) = vals.sigma_u
get_scale_param(::Weibull_MCI, vals) = vals.lambda
get_scale_param(::Lognormal_MCI, vals) = vals.sigma
get_scale_param(::Lomax_MCI, vals) = vals.lambda
get_scale_param(::Rayleigh_MCI, vals) = vals.sigma
get_scale_param(::Gamma_MCI, vals) = vals.theta

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
    lambda = reshape(ineff_vals.lambda, :, 1)
    @. out = log_pdf_ineff_exp(u, lambda, c.clamp_lo)
end

function log_pdf_ineff!(::HalfNormal_MCI, out, u, ineff_vals, c)
    sigma = reshape(ineff_vals.sigma, :, 1)
    @. out = log_pdf_ineff_halfnorm(u, sigma, c.clamp_lo)
end

function log_pdf_ineff!(::TruncatedNormal_MCI, out, u, ineff_vals, c)
    mu = reshape(ineff_vals.mu, :, 1)
    sigma_u = reshape(ineff_vals.sigma_u, :, 1)
    log_Phi = log.(max.(_normcdf.(mu ./ sigma_u), c.clamp_lo))
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
    lambda = reshape((@view ineff_vals.lambda[rs:re]), :, 1)
    @. out = log_pdf_ineff_exp(u, lambda, c.clamp_lo)
end

function log_pdf_ineff_chunk!(::HalfNormal_MCI, out, u, ineff_vals, c, rs, re)
    sigma = reshape((@view ineff_vals.sigma[rs:re]), :, 1)
    @. out = log_pdf_ineff_halfnorm(u, sigma, c.clamp_lo)
end

function log_pdf_ineff_chunk!(::TruncatedNormal_MCI, out, u, ineff_vals, c, rs, re)
    mu = reshape((@view ineff_vals.mu[rs:re]), :, 1)
    sigma_u = reshape((@view ineff_vals.sigma_u[rs:re]), :, 1)
    log_Phi = log.(max.(_normcdf.(mu ./ sigma_u), c.clamp_lo))
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

# ============================================================================
# Section 2: Unified Workspace Allocation
# ============================================================================

"""
    make_mci_workspace(ε, P, n_obs, D, chunks)

Preallocate workspace buffers for MCI T-approach (all distributions).
This single function replaces the 7 distribution-specific workspace functions.

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
        return merge(ws, (copula_adj = similar(ε, P, rows, D), Fv_buf = similar(ε, P, rows, D)))
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

"""
    MCI_nll_mci_T(ε, draws_1D, model, c, n_obs, D, frontier_sign, chunks,
                  noise_vals, ineff_vals; workspace=nothing, trans, jacob)

Compute negative log-likelihood using MCI T-approach for ANY inefficiency distribution.

This single unified function replaces all 7 distribution-specific NLL functions
(MCI_nll_exp_T, MCI_nll_halfnorm_T, etc.) by using type dispatch.

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

# Returns
Negative log-likelihood (scalar).
"""
function MCI_nll_mci_T(ε::AbstractVector{P}, draws_1D, model, c, n_obs, D,
                        frontier_sign, chunks, noise_vals, ineff_vals;
                        workspace=nothing,
                        trans, jacob,
                        copula_vals=NamedTuple(),
                        scaling_h=nothing) where {P<:Real}

    # Get scale parameter for transformation (dispatch on distribution type)
    scale = get_scale_param(model.ineff, ineff_vals)

    # Handle scalar vs vector scale (Lomax has vector lambda)
    # Use scalar directly for broadcast instead of fill() to avoid CPU Array on GPU
    scale_bcast = scale isa Number ? P(scale) : reshape(scale, n_obs, 1)

    has_copula = copula_plen(model.copula) > 0

    if chunks == 1
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

        # Scaling: multiply u* by h_i AFTER log_pdf and log_Jacobian, BEFORE composite error
        if scaling_h !== nothing
            u_buffer .= u_buffer .* scaling_h
        end

        # Component 4: Composite error z = ε + sign · u
        @. u_buffer = ε_N1 + frontier_sign * u_buffer

        # Component 4b: Copula adjustment (before log_noise_pdf! overwrites u_buffer)
        # In T-approach, F_u(u) = t (draws_1D) by probability integral transform
        if has_copula
            copula_adj = workspace === nothing ? similar(ε, P, n_obs, D) : workspace.copula_adj
            Fv_buf = workspace === nothing ? similar(ε, P, n_obs, D) : workspace.Fv_buf
            copula_log_adjustment!(model.copula, copula_adj, Fv_buf, u_buffer, draws_1D,
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
        chunk_nlls = Vector{P}(undef, chunks)

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

            # Scaling: multiply u* by h_i AFTER log_pdf and log_Jacobian, BEFORE composite error
            if scaling_h !== nothing
                u_buf .= u_buf .* (@view scaling_h[row_start:row_end])
            end

            @. u_buf = ε_chunk + frontier_sign * u_buf

            # Copula adjustment (before log_noise_pdf overwrites u_buf)
            if has_copula
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                Fv_buf = @view Fv_adj_buf[1:chunk_N, :]
                copula_log_adjustment!(model.copula, copula_adj, Fv_buf, u_buf, draws_chunk,
                                       model.noise, noise_vals, copula_vals, c)
            end

            log_noise_pdf_chunk!(model.noise, u_buf, u_buf, noise_vals, c, row_start, row_end)
            @. log_lik_buf = log_lik_buf + u_buf

            # Add copula log-density
            if has_copula
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                @. log_lik_buf = log_lik_buf + copula_adj
            end

            log_likes_chunk = logsumexp_rows(log_lik_buf) .- log(P(D))
            chunk_nlls[chunk_idx] = _sum_scalar(log_likes_chunk)
        end

        return -sum(chunk_nlls)
    end
end

# ============================================================================
# Section 4: Unified JLMS/BC Function
# ============================================================================

"""
    _jlms_bc_mci_T(ε, draws_1D, model, c, n_obs, D, sign, chunks, noise_vals, ineff_vals;
                   trans, jacob)

Compute JLMS and BC efficiency indices using MCI T-approach for ANY inefficiency distribution.

This single unified function replaces all 7 distribution-specific JLMS/BC functions.

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

        # Scaling: multiply u* by h_i AFTER log_pdf and log_Jacobian, BEFORE composite error
        if scaling_h !== nothing
            u_buffer .= u_buffer .* scaling_h
        end

        @. z_buffer = ε_N1 + sign * u_buffer

        # Copula adjustment (must run before log_noise_pdf! overwrites z_buffer)
        has_copula = copula_plen(model.copula) > 0
        if has_copula
            copula_adj = similar(ε, P, n_obs, D)
            Fv_buf = similar(ε, P, n_obs, D)
            copula_log_adjustment!(model.copula, copula_adj, Fv_buf, z_buffer, draws_1D,
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

        # Pre-allocate copula buffers if needed
        has_copula_chunked = copula_plen(model.copula) > 0
        copula_adj_buf = has_copula_chunked ? similar(ε, P, chunk_size, D) : z_buffer
        Fv_adj_buf = has_copula_chunked ? similar(ε, P, chunk_size, D) : z_buffer

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

            # Scaling: multiply u* by h_i AFTER log_pdf and log_Jacobian, BEFORE composite error
            if scaling_h !== nothing
                u_buf .= u_buf .* (@view scaling_h[row_start:row_end])
            end

            @. z_buf = ε_chunk + sign * u_buf

            # Copula adjustment (must run before log_noise_pdf_chunk! overwrites z_buf)
            if has_copula_chunked
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                Fv_buf = @view Fv_adj_buf[1:chunk_N, :]
                copula_log_adjustment!(model.copula, copula_adj, Fv_buf, z_buf, draws_chunk,
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
        end

        return (jlms=jlms_out, bc=bc_out, likelihood=likelihood_out)
    end
end
