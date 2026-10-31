# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Post-estimation numeric diagnostics: residuals, fitted values, LR tests,
# efficiency-distribution summaries, residual normality/skewness checks.
#
# All methods dispatch on `SFResult` and therefore apply uniformly to the
# MLE, MSLE, MCI, and Panel backends; sf_vs_ols and sf_skewness_test accept
# cross-sectional fits only.

using Statistics: mean, std, var, median, quantile
using StatsFuns: normcdf, normccdf, chisqccdf

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
within-demeaned data (the scale at which the likelihood is evaluated). For
these Wang–Ho fits `:v` uses the within-firm demeaned inefficiency ũ (see
[`_inefficiency_for_noise`](@ref)), so `ε̃ = ṽ − sign·ũ` stays consistent and
each firm's v̂ averages to ~0.
"""
function StatsAPI.residuals(r::SFResult; type::Symbol = :composed)
    y = _get_depvar(r)
    X = _get_frontier_matrix(r)
    β = _get_frontier_coef(r, size(X, 2))

    if type === :composed
        return y .- X * β
    elseif type === :u
        return collect(_require_eff(r, :jlms, "residuals(r; type = :u)"))
    elseif type === :v
        sgn = _frontier_sign(r)
        eps = y .- X * β
        return eps .+ sgn .* _inefficiency_for_noise(r)
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
- `:response`: xᵢ'β̂ − û for production, xᵢ'β̂ + û for cost. For Wang–Ho
  panel fits, û is the within-firm demeaned inefficiency ũ, matching the
  demeaned frontier (see [`_inefficiency_for_noise`](@ref)).
"""
function StatsAPI.fitted(r::SFResult; type::Symbol = :frontier)
    X = _get_frontier_matrix(r)
    β = _get_frontier_coef(r, size(X, 2))
    yhat_frontier = X * β
    if type === :frontier
        return yhat_frontier
    elseif type === :response
        sgn = _frontier_sign(r)
        return yhat_frontier .- sgn .* _inefficiency_for_noise(r)
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
        # Every current fit path stores the sign via _sf_sign, PorC, or model.sign,
        # so this branch is unreachable today. Fail loudly rather than silently
        # assuming production: a future result shape that loses its sign metadata
        # would otherwise return production-signed numbers for a cost model.
        error("diagnostics: frontier sign not recoverable from this result " *
              "(backend=$(getfield(r, :backend))). Expected `_sf_sign`, `PorC`, " *
              "or `model.sign`.")
    end
end

# Panel fits: the exported sfmodel_panel_fit (backend :Panel, no spec), and sfmodel_fit
# results whose unified spec has a panel datatype (also :MLE panel_TFE, panel_TFE_CSW, panel_TRE).
_is_panel_result(r::SFResult) =
    getfield(r, :backend) === :Panel ||
    (getfield(r, :spec) isa UnifiedSpec && getfield(r, :spec).datatype !== :cross_sectional)

# Firm boundary offsets (length N+1) attached for Wang–Ho panel_TFE fits; `nothing`
# for every other fit (cross-sectional, CSW, TRE).
function _panel_offsets(r::SFResult)
    nt = getfield(r, :nt)
    return hasproperty(nt, :_sf_panel_offsets) ? getproperty(nt, :_sf_panel_offsets) : nothing
end

# Within-group demeaning by firm-boundary offsets (mirrors Panel_Backend.sf_panel_demean).
function _within_demean(x::AbstractVector, offsets::AbstractVector{<:Integer})
    out = similar(x, float(eltype(x)))
    for i in 1:(length(offsets) - 1)
        s, e = offsets[i] + 1, offsets[i + 1]
        seg = @view x[s:e]
        out[s:e] .= seg .- mean(seg)
    end
    return out
end

# A fit made with `jlms_bc_index = false` carries `jlms === nothing` and `bc === nothing`.
# Every accessor below that reads them checks here first: the raw
# `MethodError: no method matching length(::Nothing)` would name neither the field nor
# the argument that switched it off.
function _require_eff(r::SFResult, field::Symbol, who::AbstractString)
    v = getproperty(r, field)
    v === nothing && error(
        "$(who) needs `result.$(field)`, and this fit does not carry it: it was made with " *
        "`jlms_bc_index = false`, which skips the inefficiency and efficiency indices. " *
        "Refit with `jlms_bc_index = true` (the default).")
    return v
end

"""
    _inefficiency_for_noise(r) -> Vector

The inefficiency term to subtract from ε̂/frontier when forming `:v`/`:response`.

For a **Wang–Ho panel_TFE** fit (any backend) the composed residual is within-firm
demeaned (ε̃), while `r.jlms = h_it·E[u*ᵢ|ε̃]` is built from the *non-demeaned*
scaling h_it. The demeaned inefficiency that actually enters ε̃ is
`ũ_it = h̃_it·E[u*ᵢ|ε̃]`; because `E[u*ᵢ|ε̃]` is constant within a firm, this equals
the within-firm demeaning of `r.jlms` (`jlmsᵢₜ − mean_firm(jlms)`) — no backend
quantity beyond the firm offsets is needed. Using the raw jlms would leave
`:v`/`:response` off by the firm-mean term `h̄ᵢ·E[u*|·]`.

For every non-panel fit this returns `collect(r.jlms)` unchanged.
"""
function _inefficiency_for_noise(r::SFResult)
    jl = collect(_require_eff(r, :jlms, "residuals(r; type = :v) / fitted(r; type = :response)"))
    off = _panel_offsets(r)
    off === nothing && return jl
    return _within_demean(vec(jl), off)
end

# ------------------------------------------------------------------
# Likelihood-ratio testing
# ------------------------------------------------------------------

"""
    LRTestResult

Result of an LR test produced by [`lrtest`](@ref) or [`sf_vs_ols`](@ref).

Fields: `LR`, `dof`, `pvalue`, `mixed` (Bool), `critical_values` (NamedTuple
with 10%, 5%, 2.5%, 1% levels), `ll_restricted`, `ll_unrestricted`, `simulated`
(Symbol: `:none`, `:restricted`, `:unrestricted` or `:both`, which log-likelihoods
are simulated), `simulation_note` (String: the line printed under "simulation",
empty when nothing is printed). The seven-argument constructor of versions before
2.0.0 still works and sets `simulated = :none` and `simulation_note = ""`.
"""
struct LRTestResult
    LR::Float64
    dof::Int
    pvalue::Float64
    mixed::Bool
    critical_values::NamedTuple
    ll_restricted::Float64
    ll_unrestricted::Float64
    simulated::Symbol
    simulation_note::String
end

# The seven-argument form of versions before 2.0.0: no simulated log-likelihood, no note.
LRTestResult(LR, dof, pvalue, mixed, critical_values, ll_restricted, ll_unrestricted) =
    LRTestResult(LR, dof, pvalue, mixed, critical_values, ll_restricted, ll_unrestricted, :none, "")

function Base.show(io::IO, ::MIME"text/plain", r::LRTestResult)
    kind = r.mixed ? "mixed χ̄²" : "χ²"
    println(io, "Likelihood-ratio test ($kind reference distribution)")
    println(io, "  LR statistic   : ", round(r.LR; digits = 6))
    println(io, "  degrees of freedom: ", r.dof)
    println(io, "  p-value        : ", round(r.pvalue; digits = 6))
    println(io, "  log-lik (R, U) : (",
            round(r.ll_restricted; digits = 4), ", ",
            round(r.ll_unrestricted; digits = 4), ")")
    isempty(r.simulation_note) || println(io, "  simulation     : ", r.simulation_note)
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
- `mixed=true`: reference ½χ²(dof − 1) + ½χ²(dof), with critical values as
  tabulated by Kodde and Palm (1986). It applies when one of the dof
  restrictions places a parameter on the boundary of the parameter space and
  every other parameter is identified under H₀ (Self and Liang 1987). It does
  not apply when parameters vanish from the likelihood under H₀, as for σᵤ² = 0
  in a truncated-normal or heteroscedastic model: counting such parameters does
  not determine the null distribution.
- `dof`: defaults to `dof(unrestricted) - dof(restricted)`; override when
  the boundary structure specifies a different value.

Simulated log-likelihoods (MSLE and MCI fits) carry simulation error. `lrtest`
compares the draw settings stored in the two results (method, `n_draws`,
`multiRand`, `draw_scheme` and the length of the Halton sequence, the MCI
transformation rule, user-supplied draws) and warns when they differ, or when one
log-likelihood is simulated and the other is computed without simulation; the
printed result then has a `simulation` line, and the fields `simulated` and
`simulation_note` record it. Common draws reduce the simulation error of the
difference but do not remove it. `lrtest` does not compare the inefficiency
distributions of two MCI fits: under MCI different distributions transform the
same draws differently (for example the Gamma and Erlang scales), so their
simulation errors need not cancel. It cannot check user-supplied draws. To see how
large the simulation error is, fit both models with
`sfmodel_fit(..., integration_check = true)`. The usual χ² reference for a simulated
LR statistic is justified asymptotically only when the number of draws grows faster
than the square root of the sample size (Lee 1999), a result derived for random draws.
Whatever the draws, also refit both models with more draws and check that the
statistic has stopped moving, or compare with `method = :MLE` where the model has a
closed form.
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
    simulated, note = _simulation_check(restricted, unrestricted)
    return LRTestResult(LR, k, pval, mixed, cv, ll_R, ll_U, simulated, note)
end

# Draw settings of simulated fits ---------------------------------------

_is_simulated(r::SFResult) = getfield(r, :backend) in (:MCI, :MSLE, :Panel)
_getor(r::SFResult, k::Symbol) = hasproperty(r, k) ? getproperty(r, k) : missing

# The MCI transformation rule a fit used; `nothing` for MSLE fits and fits without simulation.
function _mci_rule(r::SFResult)
    (hasproperty(r, :estimation_method) && r.estimation_method === :MCI) || return nothing
    if getfield(r, :backend) === :Panel
        return r.transformation === nothing ?
               Panel_Backend.default_panel_transformation(r.model.model.ineff) : r.transformation
    end
    return r.model.transformation
end

# Differences in the stored draw settings, as "name a vs b" (restricted first).
function _draw_differences(R::SFResult, U::SFResult)
    d = String[]
    for k in (:estimation_method, :n_draws, :multiRand)
        a, b = _getor(R, k), _getor(U, k)
        isequal(a, b) || push!(d, "$k $a vs $b")
    end
    sR, sU = _getor(R, :draw_scheme), _getor(U, :draw_scheme)
    gen = (:halton, :halton_recycled)
    if !ismissing(sR) && !ismissing(sU) && !(sR in gen && sU in gen)
        sR === sU || push!(d, "draw_scheme $sR vs $sU")
    else
        # both generated by the Halton rule (equal lengths give identical draws for the
        # same N and n_draws, recycled or not), or a result without draw_scheme
        a, b = _getor(R, :distinct_Halton_length), _getor(U, :distinct_Halton_length)
        isequal(a, b) || push!(d, "distinct_Halton_length $a vs $b")
    end
    rR, rU = _mci_rule(R), _mci_rule(U)
    (rR !== nothing && rU !== nothing && rR !== rU) && push!(d, "transformation $rR vs $rU")
    a, b = _getor(R, :user_draws_supplied), _getor(U, :user_draws_supplied)
    isequal(a, b) || push!(d, "user-supplied draws $a vs $b")
    return d
end

# Which log-likelihoods of the test are simulated, and the line `show` prints for it. Warns, once at
# most, when the simulation error of a fit does not cancel in the LR statistic, or may not.
function _simulation_check(R::SFResult, U::SFResult)
    simR, simU = _is_simulated(R), _is_simulated(U)
    (simR || simU) || return (:none, "")
    if simR != simU
        r, who, other = simU ? (U, "unrestricted", "restricted") : (R, "restricted", "unrestricted")
        desc = "$(r.estimation_method), $(r.n_draws) draws"
        msg = "lrtest: the $who log-likelihood is simulated ($desc) and the $other one is computed " *
              "without simulation, so the simulation error of the $who fit enters the LR statistic in full."
        # the integration check is not available for user-supplied draws
        _getor(r, :user_draws_supplied) === true || (msg *= " To see how large it is, refit that model " *
                                                     "with sfmodel_fit(..., integration_check = true).")
        @warn msg
        return (Symbol(who),
                "the $who log-likelihood is simulated ($desc); the $other one is computed without simulation")
    end
    d = _draw_differences(R, U)
    userR = _getor(R, :user_draws_supplied) === true
    userU = _getor(U, :user_draws_supplied) === true
    if !isempty(d)
        dd = join(d, "; ")
        msg = "lrtest: the two simulated log-likelihoods were not computed with the same draws ($dd). " *
              "Their simulation errors do not cancel in the LR statistic."
        (userR || userU) || (msg *= " To see how large they are, refit both models with " *
                                    "sfmodel_fit(..., integration_check = true).")
        @warn msg
        return (:both, "both log-likelihoods are simulated, with different draw settings ($dd)")
    end
    if userR && userU
        msg = "lrtest: both log-likelihoods are simulated with user-supplied draws. lrtest cannot check " *
              "that the two fits used the same draws; their simulation errors partly cancel in the LR statistic " *
              "only if they did."
        @warn msg
        return (:both, "both log-likelihoods are simulated with user-supplied draws (not checked)")
    end
    return (:both, "both log-likelihoods are simulated ($(U.estimation_method), $(U.n_draws) draws, " *
                   "same draw settings); sfmodel_fit(..., integration_check = true) on both fits shows " *
                   "their accuracy")
end

"""
    sf_vs_ols(r::SFResult) -> LRTestResult

Likelihood-ratio test of the stochastic frontier against OLS. H₀ is no inefficiency
(u ≡ 0); under H₀ the model is the OLS regression with normal errors, whose
log-likelihood the fit stores (`r.OLS_loglikelihood`). LR = −2(ℓ\\_OLS − ℓ\\_SF) is
referred to ½χ²(0) + ½χ²(1) (Lee 1993).

This null distribution is established for the cross-sectional model with normal
noise, no copula and homoscedastic half-normal inefficiency, and `sf_vs_ols` accepts
only fits of that model, by any method (also when it is written with `hetero` and a
`zvar` whose columns are all constant, not all zero). For other models it raises an error
that names the reason: the truncated-normal location, the shape parameters of the Weibull, Gamma,
Lomax and Lognormal distributions, heteroscedasticity and scaling coefficients, and
copula parameters are not identified under H₀, and counting the parameters that vanish
under H₀ does not determine the null distribution; for the exponential, Rayleigh and
Erlang distributions no published derivation of the null distribution is known; with
Student-t or Laplace noise the model without inefficiency is not OLS; panel models are
not supported.

The `dof` keyword of versions before 2.0.0 was removed.

For an MSLE or MCI fit the unrestricted log-likelihood is simulated; the result's
`simulated` field is then `:unrestricted`, and nothing is printed.
"""
function sf_vs_ols(r::SFResult; dof = nothing)
    dof === nothing || error(
        "sf_vs_ols: the `dof` keyword was removed in version 2.0.0. The test now applies only " *
        "to the cross-sectional model with normal noise, no copula and homoscedastic " *
        "half-normal inefficiency, whose reference distribution ½χ²(0) + ½χ²(1) has one " *
        "degree of freedom; for other models, counting the parameters that vanish under H₀ " *
        "does not determine the null distribution. Call sf_vs_ols(result) without `dof`.")
    kind, reasons = _sf_vs_ols_refusal(r)
    kind === :ok || error(_sf_vs_ols_message(kind, reasons))
    hasproperty(r, :OLS_loglikelihood) ||
        error("sf_vs_ols: `OLS_loglikelihood` not available on this result.")
    ll_U = StatsAPI.loglikelihood(r)
    ll_R = r.OLS_loglikelihood
    LR   = -2 * (ll_R - ll_U)
    return LRTestResult(LR, 1, _mixed_chisq_pvalue(LR, 1), true, _mixed_crits(1), ll_R, ll_U,
                        getfield(r, :backend) in (:MCI, :MSLE) ? :unrestricted : :none, "")
end

# Whether sf_vs_ols accepts a result: (:ok, []) for the cross-sectional model with normal noise,
# no copula and homoscedastic half-normal inefficiency; otherwise :panel, :nospec, or :model with
# the reasons. It reads the unified spec: the backend specs of scaling models carry an empty `hetero`.
# A `hetero` block whose `zvar` columns are all constant, not all zero, is homoscedastic.
function _sf_vs_ols_refusal(r::SFResult)
    _is_panel_result(r) && return (:panel, String[])
    spec = getfield(r, :spec)
    spec isa UnifiedSpec || return (:nospec, String[])
    reasons = String[]
    if spec.noise !== :Normal
        nm = spec.noise === :StudentT ? "Student-t" : string(spec.noise)
        push!(reasons, "its noise is $nm, so the model without inefficiency is not the normal OLS regression")
    end
    spec.copula === :None || push!(reasons, "its copula parameter is not identified under H₀")
    f = spec.ineff
    if f === :TruncatedNormal
        push!(reasons, "the truncated-normal location μ is not identified under H₀")
    elseif f in (:Weibull, :Gamma, :Lomax, :Lognormal)
        push!(reasons, "the shape (or location) parameter of the $f distribution is not identified under H₀")
    elseif f in (:Exponential, :Rayleigh, :Erlang2, :Erlang3, :Erlang4)
        push!(reasons, "no published derivation of the null distribution is known for $f inefficiency")
    elseif f !== :HalfNormal
        push!(reasons, "the null distribution has not been established for $f inefficiency")
    end
    h = spec.hetero
    if h === :scaling
        push!(reasons, "the scaling coefficients are not identified under H₀")
    elseif h isa AbstractVector && !isempty(h) && !_sf_constant_zvar(spec)
        push!(reasons, "the heteroscedasticity coefficients of $(join(string.(h), ", ")) are not identified under H₀")
    end
    return (isempty(reasons) ? :ok : :model, reasons)
end

# Whether every column of the model's `zvar` is constant and at least one is not zero, so that each `hetero`
# equation is a free constant (an all-zero `zvar` would fix the parameter instead).
function _sf_constant_zvar(spec)
    spec.mci_spec === nothing && return false
    Z = spec.mci_spec.zvar
    return size(Z, 1) > 0 && size(Z, 2) > 0 && all(j -> allequal(view(Z, :, j)), axes(Z, 2)) &&
           any(j -> !iszero(Z[1, j]), axes(Z, 2))
end

# The error message of sf_vs_ols for a refused result of kind :model, :panel or :nospec.
function _sf_vs_ols_message(kind::Symbol, reasons::Vector{String})
    s = "sf_vs_ols: the null distribution of the likelihood-ratio statistic for H₀: no inefficiency " *
        "(u ≡ 0) is established for the cross-sectional model with normal noise, no copula and " *
        "homoscedastic half-normal inefficiency (Lee 1993), where it is ½χ²(0) + ½χ²(1)."
    if kind === :model
        return s * " This model is outside that case: " * join(reasons, "; ") * ". Counting the " *
               "parameters that vanish under H₀ does not determine the null distribution."
    elseif kind === :panel
        return s * " This result is a panel model, whose model without inefficiency is not the OLS " *
               "regression that cross-sectional fits store. sf_vs_ols and sf_skewness_test do not " *
               "support panel models."
    else  # :nospec
        return s * " This result does not carry the model specification that sf_vs_ols checks; " *
               "fit the model with sfmodel_fit."
    end
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
    j = collect(_require_eff(r, :jlms, "efficiency_summary(r)"))
    b = collect(_require_eff(r, :bc,   "efficiency_summary(r)"))
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

Numeric checks on the composed and OLS residuals. Under the frontier model the OLS
residuals estimate v − u up to a constant, so their statistics describe the composed
error, not the noise.

Fields:
- `skew_ols`: skewness of the OLS residuals (matches `r.OLS_resid_skew` when the frontier has a
  constant).
- `skew_composed`: skewness of the composed SF residuals ε̂.
- `expected_skew_sign`: −1 for production, +1 for cost.
- `skew_ols_sign_ok`, `skew_composed_sign_ok`: whether the observed skewness has the
  expected sign.
- `JarqueBera_stat`, `JarqueBera_pvalue`: the Jarque–Bera statistic of the OLS residuals
  and its χ²(2) p-value, a description of their shape. Under the frontier model the
  composed error is not normal, so the statistic tends to reject normality whenever
  inefficiency is present; it also rejects without inefficiency when the noise is
  symmetric but heavy-tailed or heteroscedastic, so a rejection is not evidence of
  inefficiency; for `:StudentT` and `:Laplace` fits it says nothing about whether the
  chosen noise distribution fits.
- `M3T_stat`, `M3T_pvalue`, `M3T_robust_stat`, `M3T_robust_pvalue`: the one-sided
  third-moment test of the OLS residuals with the normal-theory and the kurtosis-robust
  variance (see [`sf_skewness_test`](@ref), which states their conditions); `NaN` for
  panel fits.
"""
function residual_diagnostics(r::SFResult)
    eps_sf  = StatsAPI.residuals(r; type = :composed)
    eps_ols = StatsAPI.residuals(r; type = :ols)
    sk_sf   = _skewness(eps_sf)
    sk_ols  = _skewness(eps_ols)
    sgn     = _frontier_sign(r)
    expected = sgn == 1.0 ? -1 : 1   # production => neg skew; cost => pos
    jb, jb_p = _jarque_bera(eps_ols)
    zN, pN, zR, pR = _is_panel_result(r) ? (NaN, NaN, NaN, NaN) : _third_moment_test(vec(eps_ols), expected)
    return (
        skew_ols              = sk_ols,
        skew_composed         = sk_sf,
        expected_skew_sign    = expected,
        skew_ols_sign_ok      = sign(sk_ols) == expected,
        skew_composed_sign_ok = sign(sk_sf)  == expected,
        JarqueBera_stat       = jb,
        JarqueBera_pvalue     = jb_p,
        M3T_stat              = zN,
        M3T_pvalue            = pN,
        M3T_robust_stat       = zR,
        M3T_robust_pvalue     = pR,
    )
end

"""
    sf_skewness_test(r::SFResult) -> NamedTuple

One-sided third-moment test of the OLS residuals (Schmidt and Lin 1984; Coelli 1995).
H₀: the OLS residuals are not skewed; the alternative is skewness in the direction a
one-sided inefficiency term implies (negative for a production frontier, positive for a
cost frontier).

With e the residuals of the OLS regression of y on the frontier variables stored with the
fit, c = e − ē, m₂ = mean(c²), m₃ = mean(c³) and N observations:
- `M3T_stat` = m₃/√(6m₂³/N), the normal-theory statistic (Coelli 1995; the test Stata
  reports), which assumes normal, homoscedastic noise under H₀;
- `M3T_robust_stat` = √N·m₃/√(mean(g²)), gᵢ = cᵢ³ − 3m₂cᵢ − m₃, with the kurtosis-robust
  variance of the i.i.d. case of Bai and Ng (2005), valid under symmetric noise with a
  finite sixth moment (for example Laplace, or Student-t with ν > 6) and conservative under
  heteroscedastic noise;
- `M3T_pvalue`, `M3T_robust_pvalue`: one-sided p-values, Φ(z) for a production frontier
  and 1 − Φ(z) for a cost frontier;
- also `skew_ols` (m₃/m₂^(3/2)), `expected_skew_sign` (−1 production, +1 cost) and `n`.

The results of both tests are at most supporting evidence of inefficiency under the
maintained frontier model, not a test of the model's specification: a misspecified
frontier or omitted skewed regressors can also skew the OLS residuals. The p-values are
asymptotic. Every fit with the same y and frontier variables gives the same result.
Cross-sectional fits only: for panel fits the function raises an error, because the OLS
residuals of a panel model are correlated within each firm.
"""
function sf_skewness_test(r::SFResult)
    _is_panel_result(r) && error(
        "sf_skewness_test: the test is derived for independent cross-sectional observations. " *
        "The OLS residuals of a panel model are correlated within each firm, so the test " *
        "is not provided for panel models.")
    e = vec(StatsAPI.residuals(r; type = :ols))
    expected = _frontier_sign(r) == 1.0 ? -1 : 1
    zN, pN, zR, pR = _third_moment_test(e, expected)
    return (skew_ols = _skewness(e), expected_skew_sign = expected,
            M3T_stat = zN, M3T_pvalue = pN,
            M3T_robust_stat = zR, M3T_robust_pvalue = pR, n = length(e))
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

# One-sided third-moment test of a residual vector; expected = -1 (production) or +1 (cost).
# Returns (normal-theory z, its p-value, kurtosis-robust z, its p-value), in Float64.
function _third_moment_test(e::AbstractVector{<:Real}, expected::Integer)
    x  = Float64.(e)
    n  = length(x)
    c  = x .- mean(x)
    m2 = mean(abs2, c)
    m3 = mean(t -> t^3, c)
    m2 > 0 || return (NaN, NaN, NaN, NaN)
    zN = m3 / sqrt(6 * m2^3 / n)                 # normal-theory variance (Coelli 1995)
    g  = c .^ 3 .- 3 * m2 .* c .- m3             # i.i.d. influence function (Bai and Ng 2005)
    vg = mean(abs2, g)
    zR = vg > 0 ? sqrt(n) * m3 / sqrt(vg) : NaN  # kurtosis-robust variance
    p(z) = isnan(z) ? NaN : (expected < 0 ? normcdf(z) : normccdf(z))
    return (zN, p(zN), zR, p(zR))
end
