# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

#=
    sf_MSLE_marginal_v21.jl

    Marginal effects of exogenous determinants on E(u)
    for MSLE-based stochastic frontier models.

    Uses ForwardDiff automatic differentiation on analytical E(u) formulas,
    following the same pattern as SFmarginal.jl of the :MLE backend (src/sf_MLE).

    Supported Models:
    - HalfNormal, TruncatedNormal, Exponential, Lognormal
    - Rayleigh, Lomax, Weibull, Erlang{2,3,4}
=#

using ForwardDiff
using DataFrames
using Statistics: mean
using SpecialFunctions: erf, loggamma, logerfcx

# ============================================================================
# Section 1: Helper Functions
# ============================================================================

"""Standard normal PDF: φ(x) = exp(-x²/2) / √(2π)"""
_normpdf(x) = exp(-0.5 * x^2) / sqrt(2π)

"""Standard normal CDF: Φ(x) = 0.5 * (1 + erf(x/√2))"""
_normcdf(x) = 0.5 * (1 + erf(x / sqrt(2)))

"""Log of standard normal PDF: log φ(x), numerically stable for all x"""
_normlogpdf(x) = -0.5 * x^2 - 0.5 * log(2π)

"""Log of standard normal CDF: log Φ(x), numerically stable for very negative x"""
_normlogcdf(x) = logerfcx(-x / sqrt(2)) - x^2 / 2 - log(2)

# _is_constant_column is defined in sf_MSLE_v21.jl (included earlier in the module)

"""Convert array to CPU if it's on GPU (no-op for CPU arrays)."""
_to_cpu(x::AbstractArray) = Array(x)
_to_cpu(x::AbstractVector) = Array(x)

# ============================================================================
# Section 2: E(u) Functions (uncondU) - ForwardDiff Compatible
# ============================================================================

#? ----------- Half Normal: E(u) = σ × √(2/π) -----------

function uncondU_half(coef, idx, Zmarg)
    # σ = exp(0.5 × Z'δ)
    k_sigma = length(idx.ineff.sigma_sq)
    ln_sigma_u_sq = sum(coef[idx.ineff.sigma_sq[j]] * Zmarg[j] for j in 1:k_sigma)
    sigma = clamp(exp(0.5 * ln_sigma_u_sq), oftype(ln_sigma_u_sq, 1e-12), oftype(ln_sigma_u_sq, 1e12))
    return sigma * sqrt(2/π)
end

#? ----------- Truncated Normal: E(u) = σ × (Λ + φ(Λ)/Φ(Λ)), Λ = μ/σ -----------

function uncondU_trun(coef, idx, Zmarg_mu, Zmarg_sigma)
    # μ = Z'δ
    k_mu = length(idx.ineff.mu)
    mu = sum(coef[idx.ineff.mu[j]] * Zmarg_mu[j] for j in 1:k_mu)

    # σ = exp(0.5 × Z'γ)
    k_sigma = length(idx.ineff.sigma_u)
    ln_sigma_u_sq = sum(coef[idx.ineff.sigma_u[j]] * Zmarg_sigma[j] for j in 1:k_sigma)
    sigma = clamp(exp(0.5 * ln_sigma_u_sq), oftype(ln_sigma_u_sq, 1e-12), oftype(ln_sigma_u_sq, 1e12))

    Λ = mu / sigma
    # E(u) = σ·m(Λ), with m(Λ) = Λ + φ(Λ)/Φ(Λ). Formed as below, the two logarithms are
    # about -Λ²/2 each and the sum cancels as Λ falls: the value and its ForwardDiff derivative
    # carry a relative error of about eps·Λ⁴/4 (1.7e-9 at Λ = -100, 5e-5 at -1e3, 61% at -1e4,
    # the wrong sign by -1e5). Below Λ = -40 the stable m of SFNumerics.sf_tn_mean is used
    # instead; for Λ ≥ -40, the expression below, bit for bit. _base_Eu_star(::TruncatedNormal)
    # below uses the same switch.
    Λ < -40 && return sigma * sf_tn_mean(Λ)
    return sigma * (Λ + exp(_normlogpdf(Λ) - _normlogcdf(Λ)))
end

#? ----------- Exponential: E(u) = √σᵤ²  (Var(u) = σᵤ²) -----------

function uncondU_expo(coef, idx, Zmarg)
    # σᵤ² = exp(Z'δ); for the exponential the squared scale is also Var(u), and E(u) is
    # the scale itself, √σᵤ²
    k_sigma = length(idx.ineff.sigma_sq)
    ln_sigma_sq = sum(coef[idx.ineff.sigma_sq[j]] * Zmarg[j] for j in 1:k_sigma)
    # σᵤ² clamped to the likelihood's [σ_floor, σ_ceil] = [1e-12, 1e12] (make_constants — the
    # Exponential σᵤ² is clamped against that base tuple, not against make_ineff_constants) so
    # the marginal mean and its gradient describe the same effective model the NLL evaluated;
    # unclamped, a large Z'δ returns Inf and a small one returns 0, with a NaN gradient in both
    # cases, while the likelihood is saturated at the bound
    sigma_sq = clamp(exp(ln_sigma_sq), oftype(ln_sigma_sq, 1e-12), oftype(ln_sigma_sq, 1e12))
    return sqrt(sigma_sq)
end

#? ----------- Lognormal: E(u) = exp(μ + σ²/2) -----------

function uncondU_lognorm(coef, idx, Zmarg_mu, Zmarg_sigma)
    # μ = Z'δ
    k_mu = length(idx.ineff.mu)
    mu = sum(coef[idx.ineff.mu[j]] * Zmarg_mu[j] for j in 1:k_mu)

    # σ = exp(0.5 × Z'γ), so σ² = exp(Z'γ)
    k_sigma = length(idx.ineff.sigma_sq)
    ln_sigma_u_sq = sum(coef[idx.ineff.sigma_sq[j]] * Zmarg_sigma[j] for j in 1:k_sigma)
    sigma_sq = clamp(exp(ln_sigma_u_sq), oftype(ln_sigma_u_sq, 1e-24), oftype(ln_sigma_u_sq, 1e24))

    return exp(mu + sigma_sq / 2)
end

#? ----------- Rayleigh: E(u) = σ × √(π/2) -----------

function uncondU_rayleigh(coef, idx, Zmarg)
    # σ = exp(0.5 × Z'δ)
    k_sigma = length(idx.ineff.sigma_sq)
    ln_sigma_u_sq = sum(coef[idx.ineff.sigma_sq[j]] * Zmarg[j] for j in 1:k_sigma)
    sigma = clamp(exp(0.5 * ln_sigma_u_sq), oftype(ln_sigma_u_sq, 1e-12), oftype(ln_sigma_u_sq, 1e12))
    return sigma * sqrt(π/2)
end

#? ----------- Erlang: E(u) = M × θ (shape pinned at the Erlang order) -----------

function uncondU_erlang(coef, idx, Zmarg_theta, M)
    # θ = exp(Z'δ_θ)
    theta_len = length(idx.ineff.theta)
    ln_theta = sum(coef[idx.ineff.theta[j]] * Zmarg_theta[j] for j in 1:theta_len)
    # θ clamped to the likelihood's [theta_floor, theta_ceil] = [1e-12, 1e12]
    # (make_ineff_constants) so the marginal mean and its gradient describe the same
    # effective model the NLL evaluated; unclamped, a large Z'δ_θ returns a different
    # or infinite E(u) while the likelihood is saturated at the bound
    theta_val = clamp(exp(ln_theta), oftype(ln_theta, 1e-12), oftype(ln_theta, 1e12))

    return M * theta_val
end

#? ----------- Lomax: E(u) = λ/(α-1), requires α > 1 -----------

function uncondU_lomax(coef, idx, Zmarg_lambda, Zmarg_alpha)
    # λ = exp(Z'γ_λ)
    k_lambda = length(idx.ineff.ln_lambda)
    ln_lambda_val = sum(coef[idx.ineff.ln_lambda[j]] * Zmarg_lambda[j] for j in 1:k_lambda)
    # λ and α clamped to the likelihood's bounds (make_ineff_constants: λ ∈ [1e-10, 1e10],
    # α ∈ [0.1, 100]) so the marginal mean and its gradient describe the same effective model
    # the NLL evaluated. Without the α clamp, exp(Z'γ_α) = Inf passes the α ≤ 1 test below and
    # E(u) = λ/(Inf-1) underflows to 0 instead of the λ/99 the NLL used. Where the clamp sits
    # relative to that test does not matter — [0.1, 100] straddles 1 and clamp is monotone, so
    # clamp(α, 0.1, 100) ≤ 1 exactly when α ≤ 1 — but it must precede the division
    lambda = clamp(exp(ln_lambda_val), oftype(ln_lambda_val, 1e-10), oftype(ln_lambda_val, 1e10))

    # α = exp(Z'γ_α)
    k_alpha = length(idx.ineff.alpha)
    ln_alpha = sum(coef[idx.ineff.alpha[j]] * Zmarg_alpha[j] for j in 1:k_alpha)
    alpha = clamp(exp(ln_alpha), oftype(ln_alpha, 0.1), oftype(ln_alpha, 100.0))

    # E(u) = λ/(α-1), undefined for α ≤ 1. The likelihood's α floor is 0.1, which is below 1, so
    # a clamped α can still be ≤ 1 and this branch stays reachable — as in the panel backend.
    #
    # Returning a bare `oftype(alpha, Inf)` here made E(u) a CONSTANT in z, so ForwardDiff
    # reported ∂E(u)/∂z = 0 and the marginal-effect table printed exactly 0.0 for such
    # observations — a finite, actionable and false number, from a converged fit, with no
    # warning. `(alpha - alpha + one(alpha))` is the identity on the value (it is 1) and carries
    # the Inf into the partials, so the value stays +∞ and the marginal effect reads NaN, which
    # is what an undefined derivative should look like.
    if alpha <= 1
        return oftype(alpha, Inf) * (alpha - alpha + one(alpha))
    end
    return lambda / (alpha - 1)
end

#? ----------- Weibull: E(u) = λ × Γ(1 + 1/k) -----------

function uncondU_weibull(coef, idx, Zmarg_lambda, Zmarg_k)
    # λ = exp(Z'δ_λ)
    k_lambda = length(idx.ineff.lambda)
    ln_lambda = sum(coef[idx.ineff.lambda[j]] * Zmarg_lambda[j] for j in 1:k_lambda)
    # λ and k clamped to the likelihood's bounds (λ ∈ [1e-12, 1e12] from make_constants' base
    # tuple, k ∈ [0.1, 10] from make_ineff_constants) so the marginal mean and its gradient
    # describe the same effective model the NLL evaluated; unclamped, a k below the floor makes
    # loggamma(1 + 1/k) overflow to Inf and the whole marginal column NaN
    lambda = clamp(exp(ln_lambda), oftype(ln_lambda, 1e-12), oftype(ln_lambda, 1e12))

    # k = exp(Z'γ_k)
    k_len = length(idx.ineff.k)
    ln_k = sum(coef[idx.ineff.k[j]] * Zmarg_k[j] for j in 1:k_len)
    k_val = clamp(exp(ln_k), oftype(ln_k, 0.1), oftype(ln_k, 10.0))

    # E(u) = λ × Γ(1 + 1/k) = λ × exp(loggamma(1 + 1/k))
    return lambda * exp(loggamma(1 + 1/k_val))
end

# ============================================================================
# Section 3: Post-Processing Functions
# ============================================================================

"""
Remove marginal effects for constant columns from a DataFrame.
Similar to SFmarginal.jl's nonConsDataFrame.
"""
function nonConsDataFrame(df::DataFrame, Z::AbstractMatrix)
    cols_to_keep = String[]
    for (i, col_name) in enumerate(names(df))
        if i <= size(Z, 2) && !_is_constant_column(@view Z[:, i])
            push!(cols_to_keep, col_name)
        end
    end
    return isempty(cols_to_keep) ? DataFrame() : df[:, cols_to_keep]
end

"""
Add marginal effects from two DataFrames, summing columns with same names.
Similar to SFmarginal.jl's addDataFrame.
"""
function addDataFrame(df1::DataFrame, df2::DataFrame)
    isempty(df2) && return df1
    isempty(df1) && return df2

    result = copy(df1)
    for col_name in names(df2)
        if col_name in names(result)
            # Sum effects for same variable
            result[!, col_name] .+= df2[!, col_name]
        else
            # Add new column
            result[!, col_name] = df2[!, col_name]
        end
    end
    return result
end

# ============================================================================
# Section 4: get_marg Dispatched Functions
# ============================================================================

#? ----------- Half Normal -----------

function get_marg(::HalfNormal_MSLE, spec::sfmodel_MSLE_spec, coef::AbstractVector{T}) where T
    # Convert to CPU for serial ForwardDiff operations
    Z = _to_cpu(spec.zvar)
    coef = _to_cpu(coef)
    idx = spec.idx
    n_obs = spec.N
    L = spec.L

    mm_sigma = Matrix{T}(undef, L, n_obs)

    @inbounds for i in 1:n_obs
        Zi = collect(@view Z[i, :])
        marg = ForwardDiff.gradient(z -> uncondU_half(coef, idx, z), Zi)
        mm_sigma[:, i] = marg
    end

    # Create DataFrame with variable names
    varnames = spec.varnames
    # Get names for sigma_sq equation
    sigma_names = _get_eq_varnames(spec, :sigma_sq)

    margeff = DataFrame(mm_sigma', sigma_names)

    # Remove constant columns
    margeff = nonConsDataFrame(margeff, Z)

    # Prepare mean summary
    margMean = _compute_marg_mean(margeff)

    # Rename with marg_ prefix
    margeff = _add_marg_prefix(margeff)

    return margeff, margMean
end

#? ----------- Truncated Normal -----------

function get_marg(::TruncatedNormal_MSLE, spec::sfmodel_MSLE_spec, coef::AbstractVector{T}) where T
    # Convert to CPU for serial ForwardDiff operations
    Z = _to_cpu(spec.zvar)
    coef = _to_cpu(coef)
    idx = spec.idx
    n_obs = spec.N
    L = spec.L
    hetero = spec.hetero

    # Determine which parameters are heteroscedastic
    mu_hetero = :mu in hetero
    sigma_hetero = :sigma_u_sq in hetero

    n_mu = mu_hetero ? L : 1
    n_sigma = sigma_hetero ? L : 1

    mm_mu = Matrix{T}(undef, n_mu, n_obs)
    mm_sigma = Matrix{T}(undef, n_sigma, n_obs)

    @inbounds for i in 1:n_obs
        Zi = collect(@view Z[i, :])

        # Build combined Z vector for gradient. A homoscedastic parameter is pinned at the
        # constant `[one(T)]`; the spec constructor's `_validate_homoscedastic_zvar` guard
        # requires the first zvar column to be the ones intercept whenever a parameter is
        # homoscedastic, so `coef·Z[:,1]` equals `coef·1` and this pin matches the likelihood
        # and the MLE `_cons`.
        Zmarg_mu = mu_hetero ? Zi : [one(T)]
        Zmarg_sigma = sigma_hetero ? Zi : [one(T)]
        Zmarg_combined = vcat(Zmarg_mu, Zmarg_sigma)

        marg = ForwardDiff.gradient(
            z -> uncondU_trun(coef, idx, z[1:n_mu], z[n_mu+1:end]),
            Zmarg_combined
        )

        mm_mu[:, i] = marg[1:n_mu]
        mm_sigma[:, i] = marg[n_mu+1:end]
    end

    # Get variable names
    mu_names = mu_hetero ? _get_eq_varnames(spec, :mu) : ["_cons_mu"]
    sigma_names = sigma_hetero ? _get_eq_varnames(spec, :sigma_sq) : ["_cons_sigma"]

    margeff_mu = DataFrame(mm_mu', mu_names)
    margeff_sigma = DataFrame(mm_sigma', sigma_names)

    # Remove constant columns
    if mu_hetero
        margeff_mu = nonConsDataFrame(margeff_mu, Z)
    else
        margeff_mu = DataFrame()
    end

    if sigma_hetero
        margeff_sigma = nonConsDataFrame(margeff_sigma, Z)
    else
        margeff_sigma = DataFrame()
    end

    # Combine: sum effects for same variables
    margeff = addDataFrame(margeff_mu, margeff_sigma)

    # Prepare mean summary
    margMean = _compute_marg_mean(margeff)

    # Rename with marg_ prefix
    margeff = _add_marg_prefix(margeff)

    return margeff, margMean
end

#? ----------- Exponential -----------

function get_marg(::Exponential_MSLE, spec::sfmodel_MSLE_spec, coef::AbstractVector{T}) where T
    # Convert to CPU for serial ForwardDiff operations
    Z = _to_cpu(spec.zvar)
    coef = _to_cpu(coef)
    idx = spec.idx
    n_obs = spec.N
    L = spec.L

    mm_sigma_sq = Matrix{T}(undef, L, n_obs)

    @inbounds for i in 1:n_obs
        Zi = collect(@view Z[i, :])
        marg = ForwardDiff.gradient(z -> uncondU_expo(coef, idx, z), Zi)
        mm_sigma_sq[:, i] = marg
    end

    sigma_sq_names = _get_eq_varnames(spec, :sigma_u_sq)
    margeff = DataFrame(mm_sigma_sq', sigma_sq_names)

    margeff = nonConsDataFrame(margeff, Z)
    margMean = _compute_marg_mean(margeff)
    margeff = _add_marg_prefix(margeff)

    return margeff, margMean
end

#? ----------- Lognormal -----------

function get_marg(::Lognormal_MSLE, spec::sfmodel_MSLE_spec, coef::AbstractVector{T}) where T
    # Convert to CPU for serial ForwardDiff operations
    Z = _to_cpu(spec.zvar)
    coef = _to_cpu(coef)
    idx = spec.idx
    n_obs = spec.N
    L = spec.L
    hetero = spec.hetero

    mu_hetero = :mu in hetero
    sigma_hetero = :sigma_u_sq in hetero

    n_mu = mu_hetero ? L : 1
    n_sigma = sigma_hetero ? L : 1

    mm_mu = Matrix{T}(undef, n_mu, n_obs)
    mm_sigma = Matrix{T}(undef, n_sigma, n_obs)

    @inbounds for i in 1:n_obs
        Zi = collect(@view Z[i, :])

        Zmarg_mu = mu_hetero ? Zi : [one(T)]
        Zmarg_sigma = sigma_hetero ? Zi : [one(T)]
        Zmarg_combined = vcat(Zmarg_mu, Zmarg_sigma)

        marg = ForwardDiff.gradient(
            z -> uncondU_lognorm(coef, idx, z[1:n_mu], z[n_mu+1:end]),
            Zmarg_combined
        )

        mm_mu[:, i] = marg[1:n_mu]
        mm_sigma[:, i] = marg[n_mu+1:end]
    end

    mu_names = mu_hetero ? _get_eq_varnames(spec, :mu) : ["_cons_mu"]
    sigma_names = sigma_hetero ? _get_eq_varnames(spec, :sigma_sq) : ["_cons_sigma"]

    margeff_mu = DataFrame(mm_mu', mu_names)
    margeff_sigma = DataFrame(mm_sigma', sigma_names)

    if mu_hetero
        margeff_mu = nonConsDataFrame(margeff_mu, Z)
    else
        margeff_mu = DataFrame()
    end

    if sigma_hetero
        margeff_sigma = nonConsDataFrame(margeff_sigma, Z)
    else
        margeff_sigma = DataFrame()
    end

    margeff = addDataFrame(margeff_mu, margeff_sigma)
    margMean = _compute_marg_mean(margeff)
    margeff = _add_marg_prefix(margeff)

    return margeff, margMean
end

#? ----------- Rayleigh -----------

function get_marg(::Rayleigh_MSLE, spec::sfmodel_MSLE_spec, coef::AbstractVector{T}) where T
    # Convert to CPU for serial ForwardDiff operations
    Z = _to_cpu(spec.zvar)
    coef = _to_cpu(coef)
    idx = spec.idx
    n_obs = spec.N
    L = spec.L

    mm_sigma = Matrix{T}(undef, L, n_obs)

    @inbounds for i in 1:n_obs
        Zi = collect(@view Z[i, :])
        marg = ForwardDiff.gradient(z -> uncondU_rayleigh(coef, idx, z), Zi)
        mm_sigma[:, i] = marg
    end

    sigma_names = _get_eq_varnames(spec, :sigma_sq)
    margeff = DataFrame(mm_sigma', sigma_names)

    margeff = nonConsDataFrame(margeff, Z)
    margMean = _compute_marg_mean(margeff)
    margeff = _add_marg_prefix(margeff)

    return margeff, margMean
end

#? ----------- Erlang -----------

function get_marg(::Erlang_MSLE{M}, spec::sfmodel_MSLE_spec, coef::AbstractVector{T}) where {M, T}
    # Convert to CPU for serial ForwardDiff operations
    Z = _to_cpu(spec.zvar)
    coef = _to_cpu(coef)
    idx = spec.idx
    n_obs = spec.N
    L = spec.L
    hetero = spec.hetero

    theta_hetero = :theta in hetero
    n_theta = theta_hetero ? L : 1

    mm_theta = Matrix{T}(undef, n_theta, n_obs)

    @inbounds for i in 1:n_obs
        Zi = collect(@view Z[i, :])
        Zmarg_theta = theta_hetero ? Zi : [one(T)]
        marg = ForwardDiff.gradient(z -> uncondU_erlang(coef, idx, z, M), Zmarg_theta)
        mm_theta[:, i] = marg
    end

    theta_names = theta_hetero ? _get_eq_varnames(spec, :theta) : ["_cons_theta"]
    margeff = DataFrame(mm_theta', theta_names)

    if theta_hetero
        margeff = nonConsDataFrame(margeff, Z)
    else
        margeff = DataFrame()
    end

    margMean = _compute_marg_mean(margeff)
    margeff = _add_marg_prefix(margeff)

    return margeff, margMean
end

#? ----------- Lomax -----------

function get_marg(::Lomax_MSLE, spec::sfmodel_MSLE_spec, coef::AbstractVector{T}) where T
    Z = _to_cpu(spec.zvar)
    coef = _to_cpu(coef)
    idx = spec.idx
    n_obs = spec.N
    L = spec.L
    hetero = spec.hetero

    lambda_hetero = :lambda in hetero
    alpha_hetero = :alpha in hetero

    n_lambda = lambda_hetero ? L : 1
    n_alpha = alpha_hetero ? L : 1

    mm_lambda = Matrix{T}(undef, n_lambda, n_obs)
    mm_alpha = Matrix{T}(undef, n_alpha, n_obs)

    @inbounds for i in 1:n_obs
        Zi = collect(@view Z[i, :])

        Zmarg_lambda = lambda_hetero ? Zi : [one(T)]
        Zmarg_alpha = alpha_hetero ? Zi : [one(T)]
        Zmarg_combined = vcat(Zmarg_lambda, Zmarg_alpha)

        marg = ForwardDiff.gradient(
            z -> uncondU_lomax(coef, idx, z[1:n_lambda], z[n_lambda+1:end]),
            Zmarg_combined
        )

        mm_lambda[:, i] = marg[1:n_lambda]
        mm_alpha[:, i] = marg[n_lambda+1:end]
    end

    lambda_names = lambda_hetero ? _get_eq_varnames(spec, :lambda) : ["_cons_lambda"]
    alpha_names = alpha_hetero ? _get_eq_varnames(spec, :alpha) : ["_cons_alpha"]

    margeff_lambda = DataFrame(mm_lambda', lambda_names)
    margeff_alpha = DataFrame(mm_alpha', alpha_names)

    if lambda_hetero
        margeff_lambda = nonConsDataFrame(margeff_lambda, Z)
    else
        margeff_lambda = DataFrame()
    end

    if alpha_hetero
        margeff_alpha = nonConsDataFrame(margeff_alpha, Z)
    else
        margeff_alpha = DataFrame()
    end

    margeff = addDataFrame(margeff_lambda, margeff_alpha)
    margMean = _compute_marg_mean(margeff)
    margeff = _add_marg_prefix(margeff)

    return margeff, margMean
end

#? ----------- Weibull -----------

function get_marg(::Weibull_MSLE, spec::sfmodel_MSLE_spec, coef::AbstractVector{T}) where T
    # Convert to CPU for serial ForwardDiff operations
    Z = _to_cpu(spec.zvar)
    coef = _to_cpu(coef)
    idx = spec.idx
    n_obs = spec.N
    L = spec.L
    hetero = spec.hetero

    lambda_hetero = :lambda in hetero
    k_hetero = :k in hetero

    n_lambda = lambda_hetero ? L : 1
    n_k = k_hetero ? L : 1

    mm_lambda = Matrix{T}(undef, n_lambda, n_obs)
    mm_k = Matrix{T}(undef, n_k, n_obs)

    @inbounds for i in 1:n_obs
        Zi = collect(@view Z[i, :])

        Zmarg_lambda = lambda_hetero ? Zi : [one(T)]
        Zmarg_k = k_hetero ? Zi : [one(T)]
        Zmarg_combined = vcat(Zmarg_lambda, Zmarg_k)

        marg = ForwardDiff.gradient(
            z -> uncondU_weibull(coef, idx, z[1:n_lambda], z[n_lambda+1:end]),
            Zmarg_combined
        )

        mm_lambda[:, i] = marg[1:n_lambda]
        mm_k[:, i] = marg[n_lambda+1:end]
    end

    lambda_names = lambda_hetero ? _get_eq_varnames(spec, :lambda) : ["_cons_lambda"]
    k_names = k_hetero ? _get_eq_varnames(spec, :k) : ["_cons_k"]

    margeff_lambda = DataFrame(mm_lambda', lambda_names)
    margeff_k = DataFrame(mm_k', k_names)

    if lambda_hetero
        margeff_lambda = nonConsDataFrame(margeff_lambda, Z)
    else
        margeff_lambda = DataFrame()
    end

    if k_hetero
        margeff_k = nonConsDataFrame(margeff_k, Z)
    else
        margeff_k = DataFrame()
    end

    margeff = addDataFrame(margeff_lambda, margeff_k)
    margMean = _compute_marg_mean(margeff)
    margeff = _add_marg_prefix(margeff)

    return margeff, margMean
end

# ============================================================================
# Section 5: Helper Functions for get_marg
# ============================================================================

"""
Get Z column variable names for marginal effects.
Uses consistent naming across all heteroscedastic equations so that
effects through different channels can be properly summed.
"""
function _get_z_varnames(spec::sfmodel_MSLE_spec)
    L = spec.L

    # Try to extract Z variable names from varnames if they follow a pattern
    # For heteroscedastic equations, varnames typically contains the Z column names
    # Look for equation blocks that use Z

    # Find the first heteroscedastic equation and extract its variable names
    eq_name_patterns = [
        ("μ", :mu), ("mu", :mu),
        ("ln_σᵤ²", :sigma_u_sq), ("ln_σ²", :sigma_u_sq), ("ln_σ2", :sigma_u_sq), ("ln_sigma", :sigma_u_sq), ("sigma", :sigma_u_sq),
        ("λ", :lambda), ("lambda", :lambda),
        ("ln_k", :k), ("k", :k),
        ("θ", :theta), ("theta", :theta),
        ("α", :alpha), ("alpha", :alpha)
    ]

    for (i, eq_name) in enumerate(spec.eqnames)
        for (pattern, sym) in eq_name_patterns
            if occursin(lowercase(pattern), lowercase(eq_name)) && sym in spec.hetero
                # Found a heteroscedastic equation, extract its variable names
                start_idx = spec.eq_indices[i]
                end_idx = i < length(spec.eq_indices) ? spec.eq_indices[i+1] - 1 : length(spec.varnames)
                names = spec.varnames[start_idx:end_idx]
                if length(names) == L
                    return names
                end
            end
        end
    end

    # Fallback: generate generic Z column names
    return ["z$i" for i in 1:L]
end

"""Get variable names for a specific equation from spec (deprecated, use _get_z_varnames)."""
function _get_eq_varnames(spec::sfmodel_MSLE_spec, eq_symbol::Symbol)
    return _get_z_varnames(spec)
end

"""Compute mean marginal effects as a NamedTuple."""
function _compute_marg_mean(margeff::DataFrame)
    isempty(margeff) && return NamedTuple()
    col_means = [round(mean(col); digits=5) for col in eachcol(margeff)]
    return (; zip(Symbol.(names(margeff)), col_means)...)
end

"""Add 'marg_' prefix to all column names."""
function _add_marg_prefix(margeff::DataFrame)
    isempty(margeff) && return margeff
    new_names = Symbol.("marg_" .* names(margeff))
    return rename(margeff, new_names)
end

# ============================================================================
# Section 6: Main Interface
# ============================================================================

"""
    marginal_effects(spec::sfmodel_MSLE_spec, coef::AbstractVector)

Compute marginal effects of exogenous variables on E(u) for a MSLE stochastic frontier model.

# Arguments
- `spec::sfmodel_MSLE_spec`: Model specification containing data and configuration
- `coef::AbstractVector`: Estimated coefficient vector

# Returns
- `margeff::DataFrame`: Observation-specific marginal effects (N rows), columns prefixed with "marg_"
- `margMean::NamedTuple`: Mean marginal effect for each variable

# Example
```julia
spec = sfmodel_MSLE_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal, ineff=:HalfNormal, hetero=[:sigma_u_sq])
coef = [...]  # estimated coefficients

margeff, margMean = marginal_effects(spec, coef)
println("Mean marginal effects: ", margMean)
```
"""
function marginal_effects(spec::sfmodel_MSLE_spec, coef::AbstractVector)
    return get_marg(spec.model.ineff, spec, coef)
end

# ============================================================================
# Section 7: Scaling Property Model — Marginal Effects
# ============================================================================

# ---------------------------------------------------------------------------
# _base_Eu_star: E(u*) for each base distribution with scalar parameters.
# Under scaling, all ineff params are scalar (L=1, zvar=ones), so we index
# directly into the coefficient vector via idx.ineff fields.
# ---------------------------------------------------------------------------

#? ----------- Half Normal: E(u*) = σ × √(2/π) -----------
function _base_Eu_star(coef, idx, ::HalfNormal_MSLE)
    sigma = clamp(exp(0.5 * coef[idx.ineff.sigma_sq[1]]), oftype(coef[idx.ineff.sigma_sq[1]], 1e-12),
                  oftype(coef[idx.ineff.sigma_sq[1]], 1e12))
    return sigma * sqrt(2/π)
end

#? ----------- Truncated Normal: E(u*) = σ(Λ + φ(Λ)/Φ(Λ)), Λ = μ/σ -----------
function _base_Eu_star(coef, idx, ::TruncatedNormal_MSLE)
    mu = coef[idx.ineff.mu[1]]
    sigma = clamp(exp(0.5 * coef[idx.ineff.sigma_u[1]]), oftype(coef[idx.ineff.sigma_u[1]], 1e-12),
                  oftype(coef[idx.ineff.sigma_u[1]], 1e12))
    Λ = mu / sigma
    # Below Λ = -40 the stable σ·m(Λ) (see uncondU_trun); above, the direct form
    Λ < -40 && return sigma * sf_tn_mean(Λ)
    return sigma * (Λ + exp(_normlogpdf(Λ) - _normlogcdf(Λ)))
end

#? ----------- Exponential: E(u*) = √λ  (Var(u) = λ) -----------
function _base_Eu_star(coef, idx, ::Exponential_MSLE)
    # kernel-aligned clamp, as in uncondU_expo
    ls = coef[idx.ineff.sigma_sq[1]]
    return sqrt(clamp(exp(ls), oftype(ls, 1e-12), oftype(ls, 1e12)))
end

#? ----------- Weibull: E(u*) = λ × Γ(1 + 1/k) -----------
function _base_Eu_star(coef, idx, ::Weibull_MSLE)
    # kernel-aligned clamps, as in uncondU_weibull
    ll = coef[idx.ineff.lambda[1]]; lk = coef[idx.ineff.k[1]]
    lambda = clamp(exp(ll), oftype(ll, 1e-12), oftype(ll, 1e12))
    k_val = clamp(exp(lk), oftype(lk, 0.1), oftype(lk, 10.0))
    return lambda * exp(loggamma(1 + 1/k_val))
end

#? ----------- Lognormal: E(u*) = exp(μ + σ²/2) -----------
function _base_Eu_star(coef, idx, ::Lognormal_MSLE)
    mu = coef[idx.ineff.mu[1]]
    sigma_sq = clamp(exp(coef[idx.ineff.sigma_sq[1]]), oftype(coef[idx.ineff.sigma_sq[1]], 1e-24),
                     oftype(coef[idx.ineff.sigma_sq[1]], 1e24))
    return exp(mu + sigma_sq / 2)
end

#? ----------- Lomax: E(u*) = λ/(α-1), requires α > 1 -----------
function _base_Eu_star(coef, idx, ::Lomax_MSLE)
    # Kernel-aligned clamps, as in uncondU_lomax. E(u*) = λ/(α − 1) does not exist for α ≤ 1;
    # NaN makes the marginal effects of the scaling model undefined (NaN) for every z, as in the
    # parameter-level kernel. α depends on the coefficients only and is a plain number here, so
    # the comparison is decided by its value, also at α = 1.
    ll = coef[idx.ineff.ln_lambda[1]]; la = coef[idx.ineff.alpha[1]]
    lambda = clamp(exp(ll), oftype(ll, 1e-10), oftype(ll, 1e10))
    alpha = clamp(exp(la), oftype(la, 0.1), oftype(la, 100.0))
    if alpha <= 1
        return oftype(alpha, NaN)
    end
    return lambda / (alpha - 1)
end

#? ----------- Rayleigh: E(u*) = σ × √(π/2) -----------
function _base_Eu_star(coef, idx, ::Rayleigh_MSLE)
    sigma = clamp(exp(0.5 * coef[idx.ineff.sigma_sq[1]]), oftype(coef[idx.ineff.sigma_sq[1]], 1e-12),
                  oftype(coef[idx.ineff.sigma_sq[1]], 1e12))
    return sigma * sqrt(π/2)
end

function _base_Eu_star(coef, idx, ::Erlang_MSLE{M}) where {M}
    # kernel-aligned clamp, as in uncondU_erlang
    lt = coef[idx.ineff.theta[1]]
    theta_val = clamp(exp(lt), oftype(lt, 1e-12), oftype(lt, 1e12))
    return M * theta_val
end

# ---------------------------------------------------------------------------
# uncondU_scaling: E(u_i) = h(z_i) × E(u*) for the scaling property model.
# The gradient w.r.t. z_i gives the marginal effect of z on E(u).
# ---------------------------------------------------------------------------

"""
    uncondU_scaling(coef, idx, Zscaling, ineff_type)

Compute E(u_i) = h(z_i) × E(u*) for observation i under the scaling property model.
`Zscaling` is the z-vector for observation i (not the full matrix).
"""
function uncondU_scaling(coef, idx, Zscaling, ineff_type)
    L = length(Zscaling)
    hi = exp(sum(coef[idx.delta[j]] * Zscaling[j] for j in 1:L))
    Eu_star = _base_Eu_star(coef, idx, ineff_type)
    return hi * Eu_star
end

# ---------------------------------------------------------------------------
# _get_scaling_varnames: extract Z variable names for the δ equation block.
# ---------------------------------------------------------------------------

"""Get variable names for the scaling (δ) equation from spec."""
function _get_scaling_varnames(spec::sfmodel_MSLE_spec)
    # Find the δ equation in eqnames and extract its variable names
    for (i, eq_name) in enumerate(spec.eqnames)
        if eq_name == "scaling"
            start_idx = spec.eq_indices[i]
            end_idx = i < length(spec.eq_indices) ? spec.eq_indices[i+1] - 1 : length(spec.varnames)
            return spec.varnames[start_idx:end_idx]
        end
    end
    # Fallback: generic names
    return ["z$i" for i in 1:spec.L_scaling]
end

# ---------------------------------------------------------------------------
# marginal_effects_scaling: main entry point for scaling model marginal effects.
# ---------------------------------------------------------------------------

"""
    marginal_effects_scaling(spec::sfmodel_MSLE_spec, coef::AbstractVector)

Compute marginal effects of scaling variables on E(u) = h(z) × E(u*).

Under the scaling property model, E(u_i) = exp(z_i'δ) × E(u*), where E(u*) depends
only on the scalar base-distribution parameters. The marginal effect of z_j on E(u_i)
is ∂E(u_i)/∂z_j, computed via ForwardDiff.

# Returns
- `margeff::DataFrame`: Observation-specific marginal effects (N rows), columns prefixed with "marg_"
- `margMean::NamedTuple`: Mean marginal effect for each variable
"""
function marginal_effects_scaling(spec::sfmodel_MSLE_spec, coef::AbstractVector{T}) where T
    Z = _to_cpu(spec.scaling_zvar)
    coef = _to_cpu(coef)
    idx = spec.idx
    n_obs = spec.N
    L_s = spec.L_scaling
    ineff_type = spec.model.ineff

    mm = Matrix{T}(undef, L_s, n_obs)

    @inbounds for i in 1:n_obs
        Zi = collect(@view Z[i, :])
        marg = ForwardDiff.gradient(
            z -> uncondU_scaling(coef, idx, z, ineff_type),
            Zi
        )
        mm[:, i] = marg
    end

    # Get scaling Z variable names from the δ equation
    scaling_varnames = _get_scaling_varnames(spec)
    margeff = DataFrame(mm', scaling_varnames)

    # No constant column removal needed (enforced at spec time)
    # Compute mean summary
    margMean = _compute_marg_mean(margeff)

    # Rename with marg_ prefix
    margeff = _add_marg_prefix(margeff)

    return margeff, margMean
end
