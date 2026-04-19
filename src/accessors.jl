# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later
#
# StatsAPI-conformant accessor layer for SFrontiers.jl.
#
# All four backends (MCI, MSLE, MLE, Panel) build results as NamedTuples with
# common field names (`coeff`, `var_cov_mat`, `std_err`, `loglikelihood`,
# `n_observations`, etc.). We wrap each NamedTuple in a thin `SFResult` struct
# and forward field access via `getproperty`, so `result.coeff` and friends
# continue to work unchanged while the standard Julia statistical API
# (`coef`, `vcov`, `stderror`, `loglikelihood`, `nobs`, `dof`, `aic`, `bic`,
# `confint`, `coeftable`, `summary`, `show`) dispatches uniformly across all
# backends.

using StatsAPI
using StatsBase: CoefTable
using StatsFuns: normccdf
using LinearAlgebra: diag
using Printf

"""
    SFResult

Thin wrapper around the NamedTuple returned by each backend. Forwards
property access so existing field-based code (`result.coeff`,
`result.loglikelihood`, etc.) works unchanged. Tagged with a `:backend`
symbol (`:MCI`, `:MSLE`, `:MLE`, `:Panel`) for backend-aware dispatch
when needed.
"""
struct SFResult
    nt::NamedTuple
    backend::Symbol
    spec::Any       # UnifiedSpec or backend-specific spec; used by predict().
end

SFResult(nt::NamedTuple, backend::Symbol) = SFResult(nt, backend, nothing)

# --- Property forwarding (backward compatibility) ---
function Base.getproperty(r::SFResult, k::Symbol)
    if k === :nt || k === :backend || k === :spec
        return getfield(r, k)
    end
    return getproperty(getfield(r, :nt), k)
end

Base.propertynames(r::SFResult) = propertynames(getfield(r, :nt))

Base.hasproperty(r::SFResult, k::Symbol) =
    k === :nt || k === :backend || k === :spec ||
    hasproperty(getfield(r, :nt), k)

# ------------------------------------------------------------------
# StatsAPI conformance — thin field forwarders
# ------------------------------------------------------------------

StatsAPI.coef(r::SFResult)          = r.coeff
StatsAPI.vcov(r::SFResult)          = r.var_cov_mat
StatsAPI.stderror(r::SFResult)      = r.std_err
StatsAPI.loglikelihood(r::SFResult) = r.loglikelihood
StatsAPI.nobs(r::SFResult)          = r.n_observations
StatsAPI.isfitted(r::SFResult)      = true
StatsAPI.islinear(r::SFResult)      = false

# ------------------------------------------------------------------
# Degrees of freedom and information criteria
# ------------------------------------------------------------------

StatsAPI.dof(r::SFResult)          = length(r.coeff)
StatsAPI.dof_residual(r::SFResult) = StatsAPI.nobs(r) - StatsAPI.dof(r)
StatsAPI.aic(r::SFResult) = -2 * StatsAPI.loglikelihood(r) + 2 * StatsAPI.dof(r)
StatsAPI.bic(r::SFResult) =
    -2 * StatsAPI.loglikelihood(r) + log(StatsAPI.nobs(r)) * StatsAPI.dof(r)

# ------------------------------------------------------------------
# Confidence intervals
# ------------------------------------------------------------------

"""
    confint(r::SFResult; level=0.95)

Return an `n × 2` matrix of Wald confidence intervals for the parameter
vector at the requested `level`. Uses the normal approximation
`coef ± z * stderror`.
"""
function StatsAPI.confint(r::SFResult; level::Real = 0.95)
    α = 1 - level
    z = _normal_quantile(1 - α / 2)
    β  = StatsAPI.coef(r)
    se = StatsAPI.stderror(r)
    return hcat(β .- z .* se, β .+ z .* se)
end

# Avoid adding Distributions as a hard dep for one quantile; invert normccdf.
# Bisection on the standard normal CDF — adequate precision for CI levels.
function _normal_quantile(p::Real)
    # For common levels we can hardcode; otherwise bisect.
    p ≈ 0.975 && return 1.9599639845400545
    p ≈ 0.95  && return 1.6448536269514722
    p ≈ 0.995 && return 2.5758293035489004
    p ≈ 0.99  && return 2.3263478740408408
    p ≈ 0.9   && return 1.2815515655446004
    lo, hi = -8.0, 8.0
    for _ in 1:80
        mid = (lo + hi) / 2
        if (1 - normccdf(mid)) < p
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

# ------------------------------------------------------------------
# Coefficient table
# ------------------------------------------------------------------

"""
    coeftable(r::SFResult; level=0.95)

Return a `StatsBase.CoefTable` with Estimate, Std. Error, z, P(>|z|),
and confidence interval columns. Row names are taken from the spec's
`varnames` when available, otherwise synthesized.
"""
function StatsAPI.coeftable(r::SFResult; level::Real = 0.95)
    β  = StatsAPI.coef(r)
    se = StatsAPI.stderror(r)
    z  = β ./ se
    p  = 2 .* (1 .- _normal_cdf.(abs.(z)))
    ci = StatsAPI.confint(r; level = level)
    pct = Int(round(level * 100))
    cols = Any[β, se, z, p, ci[:, 1], ci[:, 2]]
    colnms = ["Estimate", "Std.Error", "z", "Pr(>|z|)",
              "Lower $(pct)%", "Upper $(pct)%"]
    rownms = _row_names(r, length(β))
    return CoefTable(cols, colnms, rownms, 4, 3)
end

_normal_cdf(x::Real) = 1 - normccdf(x)

function _row_names(r::SFResult, n::Int)
    # Path 1: MSLE/MCI/Panel — varnames inside the model struct
    try
        m = getproperty(getfield(r, :nt), :model)
        vn = getproperty(m, :varnames)
        if vn isa AbstractVector{<:AbstractString} && length(vn) >= n
            return String[string(vn[i]) for i in 1:n]
        end
    catch
    end
    # Path 2: MLE — varnames stored directly on the result NamedTuple
    try
        vn = getproperty(getfield(r, :nt), :varnames)
        if vn isa AbstractVector{<:AbstractString} && length(vn) >= n
            return String[string(vn[i]) for i in 1:n]
        end
    catch
    end
    return String["param_$i" for i in 1:n]
end

# ------------------------------------------------------------------
# Summary and display
# ------------------------------------------------------------------

"""
    summary(r::SFResult)

Return a short one-line status string for the fitted model.
"""
function Base.summary(r::SFResult)
    conv = hasproperty(r, :converged) ? r.converged : true
    N    = StatsAPI.nobs(r)
    k    = StatsAPI.dof(r)
    ll   = StatsAPI.loglikelihood(r)
    return @sprintf("SFrontiers %s fit: N=%d, dof=%d, logL=%.4f, converged=%s",
                    getfield(r, :backend), N, k, ll, conv)
end

function Base.show(io::IO, r::SFResult)
    print(io, "SFResult(", getfield(r, :backend),
          ", N=", StatsAPI.nobs(r),
          ", logL=", round(StatsAPI.loglikelihood(r); digits = 4), ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::SFResult)
    println(io, "SFrontiers.jl fitted model")
    println(io, "  backend       : ", getfield(r, :backend))
    println(io, "  observations  : ", StatsAPI.nobs(r))
    println(io, "  parameters    : ", StatsAPI.dof(r))
    println(io, "  log-likelihood: ", round(StatsAPI.loglikelihood(r); digits = 6))
    println(io, "  AIC / BIC     : ",
            round(StatsAPI.aic(r); digits = 4), " / ",
            round(StatsAPI.bic(r); digits = 4))
    if hasproperty(r, :converged)
        println(io, "  converged     : ", r.converged)
    end
    if hasproperty(r, :gradient_norm) && r.gradient_norm !== nothing
        println(io, "  gradient norm : ", round(r.gradient_norm; digits = 6))
    end
    print(io, "\n")
    print(io, StatsAPI.coeftable(r))
end

# ------------------------------------------------------------------
# Convenience wrap constructor used at each return site
# ------------------------------------------------------------------

"""
    _wrap_result(nt, backend)

Internal helper: wrap a backend-built NamedTuple. If the input is already
an `SFResult` (nested calls), return it unchanged.
"""
_wrap_result(r::SFResult, ::Symbol, spec = nothing) = r
_wrap_result(nt::NamedTuple, backend::Symbol, spec = nothing) =
    SFResult(nt, backend, spec)
