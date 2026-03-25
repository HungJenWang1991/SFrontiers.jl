# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

#=
    sf_MCI_marginal_v15.jl

    Marginal effects of exogenous determinants on E(u)
    for MCI-based stochastic frontier models.

    Uses ForwardDiff automatic differentiation on analytical E(u) formulas,
    following the same pattern as SFmarginal.jl from the SFmle package.

    Supported Models:
    - HalfNormal, TruncatedNormal, Exponential, Lognormal
    - Rayleigh, Gamma, Lomax, Weibull
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

# _normcdf is defined in sf_MCI_T_v21.jl (included earlier in the module)

"""Log of standard normal PDF: log φ(x), numerically stable for all x"""
_normlogpdf(x) = -0.5 * x^2 - 0.5 * log(2π)

"""Log of standard normal CDF: log Φ(x), numerically stable for very negative x"""
_normlogcdf(x) = logerfcx(-x / sqrt(2)) - x^2 / 2 - log(2)

"""Check if a column is constant (all values equal)"""
function _is_constant_column(col::AbstractVector)
    length(col) == 0 && return true
    first_val = col[1]
    return all(x -> x ≈ first_val, col)
end

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
    ln_sigma_sq = sum(coef[idx.ineff.sigma_sq[j]] * Zmarg[j] for j in 1:k_sigma)
    sigma = clamp(exp(0.5 * ln_sigma_sq), oftype(ln_sigma_sq, 1e-12), oftype(ln_sigma_sq, 1e12))
    return sigma * sqrt(2/π)
end

#? ----------- Truncated Normal: E(u) = σ × (Λ + φ(Λ)/Φ(Λ)), Λ = μ/σ -----------

function uncondU_trun(coef, idx, Zmarg_mu, Zmarg_sigma)
    # μ = Z'δ
    k_mu = length(idx.ineff.mu)
    mu = sum(coef[idx.ineff.mu[j]] * Zmarg_mu[j] for j in 1:k_mu)

    # σ = exp(0.5 × Z'γ)
    k_sigma = length(idx.ineff.sigma_u)
    ln_sigma_sq = sum(coef[idx.ineff.sigma_u[j]] * Zmarg_sigma[j] for j in 1:k_sigma)
    sigma = clamp(exp(0.5 * ln_sigma_sq), oftype(ln_sigma_sq, 1e-12), oftype(ln_sigma_sq, 1e12))

    Λ = mu / sigma
    return sigma * (Λ + exp(_normlogpdf(Λ) - _normlogcdf(Λ)))
end

#? ----------- Exponential: E(u) = √λ  (Var(u) = λ) -----------

function uncondU_expo(coef, idx, Zmarg)
    # λ = exp(Z'δ), where λ = Var(u)
    k_lambda = length(idx.ineff.lambda)
    ln_lambda = sum(coef[idx.ineff.lambda[j]] * Zmarg[j] for j in 1:k_lambda)
    lambda = exp(ln_lambda)
    return sqrt(lambda)
end

#? ----------- Lognormal: E(u) = exp(μ + σ²/2) -----------

function uncondU_lognorm(coef, idx, Zmarg_mu, Zmarg_sigma)
    # μ = Z'δ
    k_mu = length(idx.ineff.mu)
    mu = sum(coef[idx.ineff.mu[j]] * Zmarg_mu[j] for j in 1:k_mu)

    # σ = exp(0.5 × Z'γ), so σ² = exp(Z'γ)
    k_sigma = length(idx.ineff.sigma_sq)
    ln_sigma_sq = sum(coef[idx.ineff.sigma_sq[j]] * Zmarg_sigma[j] for j in 1:k_sigma)
    sigma_sq = clamp(exp(ln_sigma_sq), oftype(ln_sigma_sq, 1e-24), oftype(ln_sigma_sq, 1e24))

    return exp(mu + sigma_sq / 2)
end

#? ----------- Rayleigh: E(u) = σ × ×(π/2) -----------

function uncondU_rayleigh(coef, idx, Zmarg)
    # σ = exp(0.5 × Z'δ)
    k_sigma = length(idx.ineff.sigma_sq)
    ln_sigma_sq = sum(coef[idx.ineff.sigma_sq[j]] * Zmarg[j] for j in 1:k_sigma)
    sigma = clamp(exp(0.5 * ln_sigma_sq), oftype(ln_sigma_sq, 1e-12), oftype(ln_sigma_sq, 1e12))
    return sigma * sqrt(π/2)
end

#? ----------- Gamma: E(u) = k × θ -----------

function uncondU_gamma(coef, idx, Zmarg_k, Zmarg_theta)
    # k = exp(Z'δ_k)
    k_len = length(idx.ineff.k)
    ln_k = sum(coef[idx.ineff.k[j]] * Zmarg_k[j] for j in 1:k_len)
    k_val = exp(ln_k)

    # θ = exp(Z'δ_θ)
    theta_len = length(idx.ineff.theta)
    ln_theta = sum(coef[idx.ineff.theta[j]] * Zmarg_theta[j] for j in 1:theta_len)
    theta_val = exp(ln_theta)

    return k_val * theta_val
end

#? ----------- Lomax: E(u) = λ/(α-1), requires α > 1 -----------

function uncondU_lomax(coef, idx, Zmarg_lambda, Zmarg_alpha)
    # λ = exp(Z'γ_λ)
    k_lambda = length(idx.ineff.ln_lambda)
    ln_lambda_val = sum(coef[idx.ineff.ln_lambda[j]] * Zmarg_lambda[j] for j in 1:k_lambda)
    lambda = exp(ln_lambda_val)

    # α = exp(Z'γ_α)
    k_alpha = length(idx.ineff.alpha)
    ln_alpha = sum(coef[idx.ineff.alpha[j]] * Zmarg_alpha[j] for j in 1:k_alpha)
    alpha = exp(ln_alpha)

    # E(u) = λ/(α-1), undefined for α ≤ 1
    if alpha <= 1
        return oftype(alpha, Inf)
    end
    return lambda / (alpha - 1)
end

#? ----------- Weibull: E(u) = λ × Γ(1 + 1/k) -----------

function uncondU_weibull(coef, idx, Zmarg_lambda, Zmarg_k)
    # λ = exp(Z'δ_λ)
    k_lambda = length(idx.ineff.lambda)
    ln_lambda = sum(coef[idx.ineff.lambda[j]] * Zmarg_lambda[j] for j in 1:k_lambda)
    lambda = exp(ln_lambda)

    # k = exp(Z'γ_k)
    k_len = length(idx.ineff.k)
    ln_k = sum(coef[idx.ineff.k[j]] * Zmarg_k[j] for j in 1:k_len)
    k_val = exp(ln_k)

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

function get_marg(::HalfNormal_MCI, spec::sfmodel_MCI_spec, coef::AbstractVector{T}) where T
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

function get_marg(::TruncatedNormal_MCI, spec::sfmodel_MCI_spec, coef::AbstractVector{T}) where T
    # Convert to CPU for serial ForwardDiff operations
    Z = _to_cpu(spec.zvar)
    coef = _to_cpu(coef)
    idx = spec.idx
    n_obs = spec.N
    L = spec.L
    hetero = spec.hetero

    # Determine which parameters are heteroscedastic
    mu_hetero = :mu in hetero
    sigma_hetero = :sigma_sq in hetero

    n_mu = mu_hetero ? L : 1
    n_sigma = sigma_hetero ? L : 1

    mm_mu = Matrix{T}(undef, n_mu, n_obs)
    mm_sigma = Matrix{T}(undef, n_sigma, n_obs)

    @inbounds for i in 1:n_obs
        Zi = collect(@view Z[i, :])

        # Build combined Z vector for gradient
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

function get_marg(::Exponential_MCI, spec::sfmodel_MCI_spec, coef::AbstractVector{T}) where T
    # Convert to CPU for serial ForwardDiff operations
    Z = _to_cpu(spec.zvar)
    coef = _to_cpu(coef)
    idx = spec.idx
    n_obs = spec.N
    L = spec.L

    mm_lambda = Matrix{T}(undef, L, n_obs)

    @inbounds for i in 1:n_obs
        Zi = collect(@view Z[i, :])
        marg = ForwardDiff.gradient(z -> uncondU_expo(coef, idx, z), Zi)
        mm_lambda[:, i] = marg
    end

    lambda_names = _get_eq_varnames(spec, :lambda)
    margeff = DataFrame(mm_lambda', lambda_names)

    margeff = nonConsDataFrame(margeff, Z)
    margMean = _compute_marg_mean(margeff)
    margeff = _add_marg_prefix(margeff)

    return margeff, margMean
end

#? ----------- Lognormal -----------

function get_marg(::Lognormal_MCI, spec::sfmodel_MCI_spec, coef::AbstractVector{T}) where T
    # Convert to CPU for serial ForwardDiff operations
    Z = _to_cpu(spec.zvar)
    coef = _to_cpu(coef)
    idx = spec.idx
    n_obs = spec.N
    L = spec.L
    hetero = spec.hetero

    mu_hetero = :mu in hetero
    sigma_hetero = :sigma_sq in hetero

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

function get_marg(::Rayleigh_MCI, spec::sfmodel_MCI_spec, coef::AbstractVector{T}) where T
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

#? ----------- Gamma -----------

function get_marg(::Gamma_MCI, spec::sfmodel_MCI_spec, coef::AbstractVector{T}) where T
    # Convert to CPU for serial ForwardDiff operations
    Z = _to_cpu(spec.zvar)
    coef = _to_cpu(coef)
    idx = spec.idx
    n_obs = spec.N
    L = spec.L
    hetero = spec.hetero

    k_hetero = :k in hetero
    theta_hetero = :theta in hetero

    n_k = k_hetero ? L : 1
    n_theta = theta_hetero ? L : 1

    mm_k = Matrix{T}(undef, n_k, n_obs)
    mm_theta = Matrix{T}(undef, n_theta, n_obs)

    @inbounds for i in 1:n_obs
        Zi = collect(@view Z[i, :])

        Zmarg_k = k_hetero ? Zi : [one(T)]
        Zmarg_theta = theta_hetero ? Zi : [one(T)]
        Zmarg_combined = vcat(Zmarg_k, Zmarg_theta)

        marg = ForwardDiff.gradient(
            z -> uncondU_gamma(coef, idx, z[1:n_k], z[n_k+1:end]),
            Zmarg_combined
        )

        mm_k[:, i] = marg[1:n_k]
        mm_theta[:, i] = marg[n_k+1:end]
    end

    k_names = k_hetero ? _get_eq_varnames(spec, :k) : ["_cons_k"]
    theta_names = theta_hetero ? _get_eq_varnames(spec, :theta) : ["_cons_theta"]

    margeff_k = DataFrame(mm_k', k_names)
    margeff_theta = DataFrame(mm_theta', theta_names)

    if k_hetero
        margeff_k = nonConsDataFrame(margeff_k, Z)
    else
        margeff_k = DataFrame()
    end

    if theta_hetero
        margeff_theta = nonConsDataFrame(margeff_theta, Z)
    else
        margeff_theta = DataFrame()
    end

    margeff = addDataFrame(margeff_k, margeff_theta)
    margMean = _compute_marg_mean(margeff)
    margeff = _add_marg_prefix(margeff)

    return margeff, margMean
end

#? ----------- Lomax -----------

function get_marg(::Lomax_MCI, spec::sfmodel_MCI_spec, coef::AbstractVector{T}) where T
    # Convert to CPU for serial ForwardDiff operations
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

function get_marg(::Weibull_MCI, spec::sfmodel_MCI_spec, coef::AbstractVector{T}) where T
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
function _get_z_varnames(spec::sfmodel_MCI_spec)
    L = spec.L

    # Try to extract Z variable names from varnames if they follow a pattern
    # For heteroscedastic equations, varnames typically contains the Z column names
    # Look for equation blocks that use Z

    # Find the first heteroscedastic equation and extract its variable names
    eq_name_patterns = [
        ("μ", :mu), ("mu", :mu),
        ("ln_σᵤ²", :sigma_sq), ("ln_sigma", :sigma_sq), ("sigma", :sigma_sq),
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
function _get_eq_varnames(spec::sfmodel_MCI_spec, eq_symbol::Symbol)
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
    marginal_effects(spec::sfmodel_MCI_spec, coef::AbstractVector)

Compute marginal effects of exogenous variables on E(u) for a MCI stochastic frontier model.

# Arguments
- `spec::sfmodel_MCI_spec`: Model specification containing data and configuration
- `coef::AbstractVector`: Estimated coefficient vector

# Returns
- `margeff::DataFrame`: Observation-specific marginal effects (N rows), columns prefixed with "marg_"
- `margMean::NamedTuple`: Mean marginal effect for each variable

# Example
```julia
spec = sfmodel_MCI_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal, ineff=:HalfNormal, hetero=[:sigma_sq])
coef = [...]  # estimated coefficients

margeff, margMean = marginal_effects(spec, coef)
println("Mean marginal effects: ", margMean)
```
"""
function marginal_effects(spec::sfmodel_MCI_spec, coef::AbstractVector)
    return get_marg(spec.model.ineff, spec, coef)
end

# ============================================================================
# Section 7: Scaling Property Model — Marginal Effects
# ============================================================================

# --- E(u*) for the base distribution (scalar parameters, no Z dependence) ---

function _base_Eu_star(coef, idx, ::HalfNormal_MCI)
    sigma = clamp(exp(0.5 * coef[idx.ineff.sigma_sq[1]]), oftype(coef[idx.ineff.sigma_sq[1]], 1e-12),
                  oftype(coef[idx.ineff.sigma_sq[1]], 1e12))
    return sigma * sqrt(2/π)
end

function _base_Eu_star(coef, idx, ::TruncatedNormal_MCI)
    mu = coef[idx.ineff.mu[1]]
    sigma = clamp(exp(0.5 * coef[idx.ineff.sigma_u[1]]), oftype(coef[idx.ineff.sigma_u[1]], 1e-12),
                  oftype(coef[idx.ineff.sigma_u[1]], 1e12))
    Λ = mu / sigma
    return sigma * (Λ + exp(_normlogpdf(Λ) - _normlogcdf(Λ)))
end

function _base_Eu_star(coef, idx, ::Exponential_MCI)
    return sqrt(exp(coef[idx.ineff.lambda[1]]))
end

function _base_Eu_star(coef, idx, ::Weibull_MCI)
    lambda = exp(coef[idx.ineff.lambda[1]])
    k_val = exp(coef[idx.ineff.k[1]])
    return lambda * exp(loggamma(1 + 1/k_val))
end

function _base_Eu_star(coef, idx, ::Lognormal_MCI)
    mu = coef[idx.ineff.mu[1]]
    sigma_sq = clamp(exp(coef[idx.ineff.sigma_sq[1]]), oftype(coef[idx.ineff.sigma_sq[1]], 1e-24),
                     oftype(coef[idx.ineff.sigma_sq[1]], 1e24))
    return exp(mu + sigma_sq / 2)
end

function _base_Eu_star(coef, idx, ::Lomax_MCI)
    lambda = exp(coef[idx.ineff.ln_lambda[1]])
    alpha = exp(coef[idx.ineff.alpha[1]])
    if alpha <= 1
        return oftype(alpha, Inf)
    end
    return lambda / (alpha - 1)
end

function _base_Eu_star(coef, idx, ::Rayleigh_MCI)
    sigma = clamp(exp(0.5 * coef[idx.ineff.sigma_sq[1]]), oftype(coef[idx.ineff.sigma_sq[1]], 1e-12),
                  oftype(coef[idx.ineff.sigma_sq[1]], 1e12))
    return sigma * sqrt(π/2)
end

function _base_Eu_star(coef, idx, ::Gamma_MCI)
    k_val = exp(coef[idx.ineff.k[1]])
    theta_val = exp(coef[idx.ineff.theta[1]])
    return k_val * theta_val
end

# --- Scaling unconditional E(u) ---

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

# --- Variable name extraction ---

"""Get variable names for the scaling (δ) equation from spec."""
function _get_scaling_varnames(spec::sfmodel_MCI_spec)
    for (i, eq_name) in enumerate(spec.eqnames)
        if eq_name == "scaling"
            start_idx = spec.eq_indices[i]
            end_idx = i < length(spec.eq_indices) ? spec.eq_indices[i+1] - 1 : length(spec.varnames)
            return spec.varnames[start_idx:end_idx]
        end
    end
    return ["z$i" for i in 1:spec.L_scaling]
end

# --- Main scaling marginal effects function ---

"""
    marginal_effects_scaling(spec::sfmodel_MCI_spec, coef::AbstractVector)

Compute marginal effects of scaling variables on E(u) = h(z) × E(u*).

Under the scaling property model, E(u_i) = exp(z_i'δ) × E(u*), where E(u*) depends
only on the scalar base-distribution parameters. The marginal effect of z_j on E(u_i)
is ∂E(u_i)/∂z_j, computed via ForwardDiff.

# Returns
- `margeff::DataFrame`: Observation-specific marginal effects (N rows), columns prefixed with "marg_"
- `margMean::NamedTuple`: Mean marginal effect for each variable
"""
function marginal_effects_scaling(spec::sfmodel_MCI_spec, coef::AbstractVector{T}) where T
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

    scaling_varnames = _get_scaling_varnames(spec)
    margeff = DataFrame(mm', scaling_varnames)

    margMean = _compute_marg_mean(margeff)
    margeff = _add_marg_prefix(margeff)

    return margeff, margMean
end
