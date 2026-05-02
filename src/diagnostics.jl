# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Post-estimation numeric diagnostics: residuals, fitted values, LR tests,
# efficiency-distribution summaries, residual normality/skewness checks.
#
# All methods dispatch on `SFResult` and therefore apply uniformly to the
# MLE, MSLE, MCI, and Panel backends.

using Statistics: mean, std, var, median, quantile
using StatsFuns: normccdf, chisqccdf

# ------------------------------------------------------------------
# Residuals and fitted values
# ------------------------------------------------------------------

"""
    residuals(r::SFResult; type=:composed)

Observation-level residuals from a fitted stochastic frontier model.

- `:composed` (default): composed error ε̂ᵢ = yᵢ − xᵢ'β̂.
- `:u`: JLMS point estimate of inefficiency, E(uᵢ|ε̂ᵢ). Alias of `r.jlms`.
- `:v`: noise estimate, v̂ᵢ = ε̂ᵢ + sign·û, where sign = +1 for
  production (ε = v − u) and −1 for cost (ε = v + u).
- `:ols`: residuals from the frontier-only OLS regression, yᵢ − xᵢ'β̂ₒₗₛ.

For panel TFE models, `:composed`/`:ols` residuals are computed on the
within-demeaned data (the scale at which the likelihood is evaluated).
"""
function StatsAPI.residuals(r::SFResult; type::Symbol = :composed)
    y = _get_depvar(r)
    X = _get_frontier_matrix(r)
    β = _get_frontier_coef(r, size(X, 2))

    if type === :composed
        return y .- X * β
    elseif type === :u
        return collect(r.jlms)
    elseif type === :v
        sgn = _frontier_sign(r)
        eps = y .- X * β
        return eps .+ sgn .* collect(r.jlms)
    elseif type === :ols
        β_ols = X \ y
        return y .- X * β_ols
    else
        error("residuals: unknown `type=:$type`. " *
              "Valid options are :composed, :u, :v, :ols.")
    end
end

"""
    fitted(r::SFResult; type=:frontier)

Fitted values from the stochastic frontier model.

- `:frontier` (default): xᵢ'β̂.
- `:response`: xᵢ'β̂ − û for production, xᵢ'β̂ + û for cost.
"""
function StatsAPI.fitted(r::SFResult; type::Symbol = :frontier)
    X = _get_frontier_matrix(r)
    β = _get_frontier_coef(r, size(X, 2))
    yhat_frontier = X * β
    if type === :frontier
        return yhat_frontier
    elseif type === :response
        sgn = _frontier_sign(r)
        return yhat_frontier .- sgn .* collect(r.jlms)
    else
        error("fitted: unknown `type=:$type`. Valid options are :frontier, :response.")
    end
end

# Internal helpers ------------------------------------------------------

function _get_depvar(r::SFResult)
    nt = getfield(r, :nt)
    hasproperty(nt, :_sf_depvar) && return Array(getproperty(nt, :_sf_depvar))
    # MCI/MSLE/Panel nest data in the spec struct stored as `model`.
    try
        m = getproperty(nt, :model)
        return Array(getproperty(m, :depvar))
    catch
        error("diagnostics: depvar not recoverable from this result " *
              "(backend=$(getfield(r, :backend))).")
    end
end

function _get_frontier_matrix(r::SFResult)
    nt = getfield(r, :nt)
    hasproperty(nt, :_sf_frontier) && return Array(getproperty(nt, :_sf_frontier))
    try
        m = getproperty(nt, :model)
        return Array(getproperty(m, :frontier))
    catch
        error("diagnostics: frontier matrix not recoverable from this result " *
              "(backend=$(getfield(r, :backend))).")
    end
end

function _get_frontier_coef(r::SFResult, K::Integer)
    nt = getfield(r, :nt)
    if getfield(r, :backend) === :MLE
        hasproperty(nt, :coeff_frontier) && return collect(getproperty(nt, :coeff_frontier))
        return r.coeff[1:K]
    end
    # Simulation backends: `r.frontier` holds the β vector.
    f = getproperty(nt, :frontier)
    f isa AbstractVector{<:Real} && return collect(f)
    return r.coeff[1:K]
end

function _frontier_sign(r::SFResult)
    nt = getfield(r, :nt)
    hasproperty(nt, :_sf_sign) &&
        return getproperty(nt, :_sf_sign) == -1 ? -1.0 : 1.0
    if getfield(r, :backend) === :MLE
        if hasproperty(nt, :PorC)
            p = getproperty(nt, :PorC)
            return (p == -1) ? -1.0 : 1.0
        end
    end
    try
        m = getproperty(nt, :model)
        s = getproperty(m, :sign)
        return s == -1 ? -1.0 : 1.0
    catch
        return 1.0  # default to production
    end
end

# ------------------------------------------------------------------
# Likelihood-ratio testing
# ------------------------------------------------------------------

"""
    LRTestResult

Result of an LR test produced by [`lrtest`](@ref) or [`sf_vs_ols`](@ref).

Fields: `LR`, `dof`, `pvalue`, `mixed` (Bool), `critical_values` (NamedTuple
with 10%, 5%, 2.5%, 1% levels), `ll_restricted`, `ll_unrestricted`.
"""
struct LRTestResult
    LR::Float64
    dof::Int
    pvalue::Float64
    mixed::Bool
    critical_values::NamedTuple
    ll_restricted::Float64
    ll_unrestricted::Float64
end

function Base.show(io::IO, ::MIME"text/plain", r::LRTestResult)
    kind = r.mixed ? "mixed χ̄²" : "χ²"
    println(io, "Likelihood-ratio test ($kind reference distribution)")
    println(io, "  LR statistic   : ", round(r.LR; digits = 6))
    println(io, "  degrees of freedom: ", r.dof)
    println(io, "  p-value        : ", round(r.pvalue; digits = 6))
    println(io, "  log-lik (R, U) : (",
            round(r.ll_restricted; digits = 4), ", ",
            round(r.ll_unrestricted; digits = 4), ")")
    println(io, "  critical values: ",
            "10% = ", round(r.critical_values.p10;  digits = 4),
            ", 5% = ",  round(r.critical_values.p05;  digits = 4),
            ", 2.5% = ", round(r.critical_values.p025; digits = 4),
            ", 1% = ",   round(r.critical_values.p01;  digits = 4))
end

"""
    lrtest(restricted::SFResult, unrestricted::SFResult;
           mixed=false, dof=nothing) -> LRTestResult

Likelihood-ratio test comparing nested SF models.

- `mixed=false`: reference distribution is χ²(dof).
- `mixed=true`: reference is the 50:50 mixture χ̄² used for boundary
  tests (e.g., σ_u² = 0). Critical values follow Kodde and Palm (1986).
- `dof`: defaults to `dof(unrestricted) - dof(restricted)`; override when
  the boundary structure specifies a different value.
"""
function lrtest(restricted::SFResult, unrestricted::SFResult;
                mixed::Bool = false, dof::Union{Nothing, Integer} = nothing)
    ll_R = StatsAPI.loglikelihood(restricted)
    ll_U = StatsAPI.loglikelihood(unrestricted)
    LR   = -2 * (ll_R - ll_U)
    k    = dof === nothing ? (StatsAPI.dof(unrestricted) - StatsAPI.dof(restricted)) : Int(dof)
    k > 0 || error("lrtest: effective dof must be positive; got $k.")
    pval = mixed ? _mixed_chisq_pvalue(LR, k) : _chisq_pvalue(LR, k)
    cv   = mixed ? _mixed_crits(k) : _chisq_crits(k)
    return LRTestResult(LR, k, pval, mixed, cv, ll_R, ll_U)
end

"""
    sf_vs_ols(r::SFResult; dof=nothing) -> LRTestResult

Boundary likelihood-ratio test of the stochastic frontier against the
OLS baseline (H₀: σ_u² = 0). Uses the mixed χ̄² distribution and the
`OLS_loglikelihood` stored in `r`. If `dof` is omitted, it defaults to
the number of parameters associated with the inefficiency distribution.
"""
function sf_vs_ols(r::SFResult; dof::Union{Nothing, Integer} = nothing)
    ll_U = StatsAPI.loglikelihood(r)
    hasproperty(r, :OLS_loglikelihood) ||
        error("sf_vs_ols: `OLS_loglikelihood` not available on this result.")
    ll_R = r.OLS_loglikelihood
    LR = -2 * (ll_R - ll_U)
    k  = dof === nothing ? _default_sf_vs_ols_dof(r) :
                           (Int(dof) > 0 ? Int(dof) : error("sf_vs_ols: dof must be positive."))
    pval = _mixed_chisq_pvalue(LR, k)
    cv   = _mixed_crits(k)
    return LRTestResult(LR, k, pval, true, cv, ll_R, ll_U)
end

# Crude default: total parameters minus number of frontier coefficients.
# Users are encouraged to pass `dof` explicitly to match the literature.
function _default_sf_vs_ols_dof(r::SFResult)
    K = length(r.frontier)
    return max(StatsAPI.dof(r) - K, 1)
end

_chisq_pvalue(x::Real, k::Integer)       = chisqccdf(k, max(x, 0.0))
function _mixed_chisq_pvalue(x::Real, k::Integer)
    x <= 0 && return 1.0
    return 0.5 * chisqccdf(k - 1, x) + 0.5 * chisqccdf(k, x)
end

_chisq_crits(k::Integer) = (
    p10  = _chisq_quantile(k, 0.90),
    p05  = _chisq_quantile(k, 0.95),
    p025 = _chisq_quantile(k, 0.975),
    p01  = _chisq_quantile(k, 0.99))

_mixed_crits(k::Integer) = (
    p10  = _mixed_quantile(k, 0.90),
    p05  = _mixed_quantile(k, 0.95),
    p025 = _mixed_quantile(k, 0.975),
    p01  = _mixed_quantile(k, 0.99))

# Bisection quantiles (avoid a Distributions dep).
function _chisq_quantile(k::Integer, p::Real)
    lo, hi = 0.0, 200.0
    for _ in 1:80
        mid = 0.5 * (lo + hi)
        if (1 - chisqccdf(k, mid)) < p; lo = mid; else; hi = mid; end
    end
    return 0.5 * (lo + hi)
end

function _mixed_quantile(k::Integer, p::Real)
    lo, hi = 0.0, 200.0
    for _ in 1:80
        mid = 0.5 * (lo + hi)
        cdf = 1 - _mixed_chisq_pvalue(mid, k)
        if cdf < p; lo = mid; else; hi = mid; end
    end
    return 0.5 * (lo + hi)
end

# ------------------------------------------------------------------
# Efficiency-distribution summary
# ------------------------------------------------------------------

"""
    EfficiencySummary

Result of [`efficiency_summary`](@ref). Holds location/spread summaries and
quantiles (deciles + quartiles) of the observation-level JLMS inefficiency
index and the Battese–Coelli efficiency index.

Fields: `n`, `mean_jlms`, `std_jlms`, `min_jlms`, `max_jlms`, `median_jlms`,
`quantiles_jlms`, `mean_bc`, `std_bc`, `min_bc`, `max_bc`, `median_bc`,
`quantiles_bc`. The two `quantiles_*` fields are `NamedTuple`s with
`q10, q20, q25, q30, q40, q50, q60, q70, q75, q80, q90`.
"""
struct EfficiencySummary
    n::Int
    mean_jlms::Float64
    std_jlms::Float64
    min_jlms::Float64
    max_jlms::Float64
    median_jlms::Float64
    quantiles_jlms::NamedTuple
    mean_bc::Float64
    std_bc::Float64
    min_bc::Float64
    max_bc::Float64
    median_bc::Float64
    quantiles_bc::NamedTuple
end

function Base.show(io::IO, ::MIME"text/plain", es::EfficiencySummary)
    println(io, "EfficiencySummary  (n = ", es.n, ")")
    println(io, "  JLMS:  mean = ", round(es.mean_jlms;   digits = 4),
                "  std = ",         round(es.std_jlms;    digits = 4),
                "  min = ",         round(es.min_jlms;    digits = 4),
                "  max = ",         round(es.max_jlms;    digits = 4),
                "  median = ",      round(es.median_jlms; digits = 4))
    qj = es.quantiles_jlms
    println(io, "         q10 = ", round(qj.q10; digits = 4),
                "  q25 = ",         round(qj.q25; digits = 4),
                "  q50 = ",         round(qj.q50; digits = 4),
                "  q75 = ",         round(qj.q75; digits = 4),
                "  q90 = ",         round(qj.q90; digits = 4))
    println(io, "  BC:    mean = ", round(es.mean_bc;   digits = 4),
                "  std = ",         round(es.std_bc;    digits = 4),
                "  min = ",         round(es.min_bc;    digits = 4),
                "  max = ",         round(es.max_bc;    digits = 4),
                "  median = ",      round(es.median_bc; digits = 4))
    qb = es.quantiles_bc
    println(io, "         q10 = ", round(qb.q10; digits = 4),
                "  q25 = ",         round(qb.q25; digits = 4),
                "  q50 = ",         round(qb.q50; digits = 4),
                "  q75 = ",         round(qb.q75; digits = 4),
                "  q90 = ",         round(qb.q90; digits = 4))
    println(io, "  Fields: e.g., <name>.mean_jlms, <name>.quantiles_bc.q25; ",
                "full list via propertynames(<name>).")
end

"""
    efficiency_summary(r::SFResult) -> EfficiencySummary

Location/spread summaries and quantiles (deciles + quartiles) of the
observation-level JLMS inefficiency index and the Battese–Coelli
efficiency index. The result is an [`EfficiencySummary`](@ref) struct
that pretty-prints a two-block (JLMS / BC) summary at four-decimal
precision and exposes its fields by dot access; the full eleven-quantile
vectors are accessible as `result.quantiles_jlms` and `result.quantiles_bc`.
"""
function efficiency_summary(r::SFResult)
    j = collect(r.jlms)
    b = collect(r.bc)
    probs = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    qkeys = Tuple(Symbol.("q", Int.(100 .* probs)))
    qj    = NamedTuple{qkeys}(Tuple(quantile(j, probs)))
    qb    = NamedTuple{qkeys}(Tuple(quantile(b, probs)))
    return EfficiencySummary(
        length(j),
        mean(j), std(j), minimum(j), maximum(j), median(j), qj,
        mean(b), std(b), minimum(b), maximum(b), median(b), qb,
    )
end

# ------------------------------------------------------------------
# Residual normality / skewness diagnostics
# ------------------------------------------------------------------

"""
    residual_diagnostics(r::SFResult) -> NamedTuple

Numeric checks on the composed and OLS residuals.

Fields:
- `skew_ols`: skewness of OLS residuals (matches `r.OLS_resid_skew`).
- `skew_composed`: skewness of the composed SF residuals ε̂.
- `expected_skew_sign`: `-1` for production, `+1` for cost.
- `skew_ols_sign_ok`, `skew_composed_sign_ok`: whether the observed
  skewness has the expected sign (negative for production, positive
  for cost).
- `JarqueBera_stat`, `JarqueBera_pvalue`: Jarque–Bera statistic and
  p-value on OLS residuals as a normality check of the noise component.
"""
function residual_diagnostics(r::SFResult)
    eps_sf  = StatsAPI.residuals(r; type = :composed)
    eps_ols = StatsAPI.residuals(r; type = :ols)
    sk_sf   = _skewness(eps_sf)
    sk_ols  = _skewness(eps_ols)
    sgn     = _frontier_sign(r)
    expected = sgn == 1.0 ? -1 : 1   # production => neg skew; cost => pos
    jb, jb_p = _jarque_bera(eps_ols)
    return (
        skew_ols              = sk_ols,
        skew_composed         = sk_sf,
        expected_skew_sign    = expected,
        skew_ols_sign_ok      = sign(sk_ols) == expected,
        skew_composed_sign_ok = sign(sk_sf)  == expected,
        JarqueBera_stat       = jb,
        JarqueBera_pvalue     = jb_p,
    )
end

function _skewness(x::AbstractVector{<:Real})
    n = length(x)
    μ = mean(x)
    σ = std(x; corrected = false)
    σ == 0 && return 0.0
    return mean(((x .- μ) ./ σ) .^ 3)
end

function _kurtosis(x::AbstractVector{<:Real})
    n = length(x)
    μ = mean(x)
    σ = std(x; corrected = false)
    σ == 0 && return 0.0
    return mean(((x .- μ) ./ σ) .^ 4)
end

function _jarque_bera(x::AbstractVector{<:Real})
    n = length(x)
    s = _skewness(x)
    k = _kurtosis(x)
    jb = (n / 6) * (s^2 + 0.25 * (k - 3)^2)
    return jb, chisqccdf(2, max(jb, 0.0))
end
