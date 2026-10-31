# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

# sf_accuracy.jl
# Accuracy diagnostics of simulated likelihoods: the optional integration check. After an MCI or MSLE fit
# with `sfmodel_fit(..., integration_check = true)`, the simulated log-likelihood is evaluated again at
# the estimate with four times the draws, and the one-step parameter shift −V·(g_4S − g_S) that the larger
# draw set implies is expressed in standard errors. Each part reads OK, CAUTION or FAIL against thresholds that
# are heuristics. The check runs once after the fit and does not re-estimate. The shape-range check compares
# the estimated shape of MCI Gamma, Weibull and Lomax fits under the default transformation rule, and of MSLE
# Lomax fits, with the thresholds of SF_SHAPE_RANGES and prints a warning below them. sf_warn_lomax_mean
# prints the warning for Lomax marginal effects that are NaN where the mean does not exist. A fit with
# `shape_warning = false` prints neither warning; the verdict is stored either way. Loaded before the
# backends.

module SFAccuracy

using ForwardDiff: ForwardDiff
using ..SFRedflag: sf_fatal_error, sf_out_of_memory
using ..SFDraws: sf_halton_period
using Printf: @sprintf

export SF_INTEGRATION_DEFAULTS, sf_integration_settings, sf_integration_unavailable, sf_integration_verdict,
       sf_run_integration_check, sf_integration_line
export SF_SHAPE_RANGES, sf_shape_entry, sf_values_below, sf_print_warning, sf_shape_check
export sf_warn_lomax_mean

"""
The default thresholds of the integration check: CAUTION from a change of 0.1 and FAIL from a change of 1 in
the sample log-likelihood (`ll_caution`, `ll_fail`), CAUTION from 0.1 and FAIL from 0.5 standard errors for
the largest one-step parameter shift (`shift_caution`, `shift_fail`). They are heuristics.
"""
const SF_INTEGRATION_DEFAULTS = (ll_caution = 0.1, ll_fail = 1.0, shift_caution = 0.1, shift_fail = 0.5)

"""
    sf_integration_settings(x) -> Union{Nothing, NamedTuple}

The thresholds of the integration check from the `integration_check` option: `false` gives `nothing` (the
check is off), `true` gives `SF_INTEGRATION_DEFAULTS`, and a NamedTuple with any of `ll_caution`, `ll_fail`,
`shift_caution`, `shift_fail` replaces those defaults (as Float64). Each value must be finite and positive,
and each CAUTION threshold below its FAIL threshold. Anything else is an `ArgumentError`.
"""
function sf_integration_settings(x)
    x === false && return nothing
    x === true && return SF_INTEGRATION_DEFAULTS
    x isa NamedTuple || throw(ArgumentError("`integration_check` must be false, true, or a NamedTuple with " *
                                            "any of ll_caution, ll_fail, shift_caution, shift_fail."))
    for (k, v) in pairs(x)
        haskey(SF_INTEGRATION_DEFAULTS, k) ||
            throw(ArgumentError("`integration_check` has the unknown key $k; the keys are ll_caution, " *
                                "ll_fail, shift_caution, shift_fail."))
        (v isa Real && isfinite(v) && v > 0) ||
            throw(ArgumentError("the `integration_check` thresholds must be finite and positive; got $k = $v."))
    end
    s = map(Float64, merge(SF_INTEGRATION_DEFAULTS, x))
    s.ll_caution < s.ll_fail ||
        throw(ArgumentError("`integration_check`: ll_caution must be below ll_fail; got " *
                            "$(s.ll_caution) and $(s.ll_fail)."))
    s.shift_caution < s.shift_fail ||
        throw(ArgumentError("`integration_check`: shift_caution must be below shift_fail; got " *
                            "$(s.shift_caution) and $(s.shift_fail)."))
    return s
end

"""
    sf_integration_unavailable(method, N) -> Union{Nothing, String}

Why the integration check cannot run for a fit of `N` observations with the method settings `method` (the
fields `draws`, `multiRand`, `n_draws`, `recycling` and `halton_legacy` of an MCI, MSLE or panel method), or
`nothing` when it can. The check's 4 × `n_draws` draws are those a fit with `n_draws = 4 × n_draws` and the
same settings would use. They do not exist when the fit used user-supplied draws, or when its recycled Halton
sequence (the `recycling` option, or the deprecated `distinct_Halton_length`) has fewer than 4 × `n_draws`
points. Shared draws (`multiRand = false`) always have them.
"""
function sf_integration_unavailable(method, N::Integer)
    method.draws !== nothing && return "the fit used user-supplied draws"
    method.multiRand || return nothing
    try
        sf_halton_period(N, 4 * method.n_draws; recycling = method.recycling, legacy = method.halton_legacy)
    catch err
        err isa ArgumentError || rethrow()
        return "the fit's recycled Halton sequence has fewer than 4 × n_draws points"
    end
    return nothing
end

# The order of the verdicts: the status of the check is the worse of its two parts.
_sf_verdict_rank(v::Symbol) = v === :FAIL ? 2 : v === :CAUTION ? 1 : 0

"""
    sf_integration_verdict(; ll, ll_check, n_obs, S, S4, V, se, hessian_flag, g_diff, theta, names,
                           thresholds, seconds) -> NamedTuple

The result of the integration check from its inputs; no likelihood is evaluated here. `ll` and `ll_check` are
the sample log-likelihoods at the estimate with the fit's `S` draws and with `S4` draws; `V`, `se` and
`hessian_flag` are the fit's variance matrix, standard errors and Hessian status (`sf_classify_hessian`);
`g_diff` is the gradient of the negative log-likelihood with `S4` draws minus that with `S` draws, or
`nothing`; `theta` is the estimate, `names` are the parameter names and `thresholds` the settings of
`sf_integration_settings`.

The change Δ = `ll_check − ll` reads FAIL when it is not finite or |Δ| ≥ `ll_fail`, CAUTION when
|Δ| ≥ `ll_caution`, and OK otherwise. The one-step shift −V·`g_diff`, divided by the standard errors, reads
FAIL when its largest absolute entry is not finite or at least `shift_fail`, CAUTION when it is at least
`shift_caution`, and OK otherwise; `max_shift_param` names the parameter of that entry,
`max_shift_estimate` is its value in `theta` and `max_shift_onestep` that value plus its shift. The shift
is not computed when the variance matrix is not usable (`hessian_flag` 1 or 2) or when the log-likelihood
or the gradient with `S4` draws is not finite; the status then rests on Δ alone, `shift_note` says why,
and `max_shift_param` is "" and the two values NaN. Otherwise the status is the worse of the two parts.
"""
function sf_integration_verdict(; ll, ll_check, n_obs, S, S4, V, se, hessian_flag, g_diff, theta, names,
                                thresholds, seconds)
    Δ = Float64(ll_check) - Float64(ll)
    ll_verdict = (!isfinite(Δ) || abs(Δ) >= thresholds.ll_fail) ? :FAIL :
                 abs(Δ) >= thresholds.ll_caution ? :CAUTION : :OK
    note = if hessian_flag == 1 && !all(isfinite, V)
        "the variance matrix could not be computed or inverted (see redflag)"
    elseif hessian_flag == 2
        "some standard errors are NaN (see redflag)"
    elseif hessian_flag == 1
        "the variance matrix is not positive definite (see redflag)"
    elseif !isfinite(ll_check)
        "the log-likelihood with $S4 draws is not finite"
    elseif g_diff === nothing || !all(isfinite, g_diff)
        "the gradient with $S4 draws is not finite"
    else
        ""
    end
    shift_available = isempty(note)
    if shift_available
        shift = Vector{Float64}(-(V * g_diff))
        shift_se = Vector{Float64}(shift ./ se)
        m = argmax(abs.(shift_se))
        max_shift_se = abs(shift_se[m])
        max_shift_param = String(names[m])
        max_shift_estimate = Float64(theta[m])
        max_shift_onestep = max_shift_estimate + shift[m]
        shift_verdict = (!isfinite(max_shift_se) || max_shift_se >= thresholds.shift_fail) ? :FAIL :
                        max_shift_se >= thresholds.shift_caution ? :CAUTION : :OK
        status = _sf_verdict_rank(shift_verdict) > _sf_verdict_rank(ll_verdict) ? shift_verdict : ll_verdict
    else
        shift = nothing
        shift_se = nothing
        max_shift_se = NaN
        max_shift_param = ""
        max_shift_estimate = NaN
        max_shift_onestep = NaN
        shift_verdict = nothing
        status = ll_verdict
    end
    return (status = status, reason = "", n_draws = Int(S), n_draws_check = Int(S4), loglik = Float64(ll),
            loglik_check = Float64(ll_check), delta_loglik = Δ, delta_loglik_per_obs = Δ / n_obs,
            ll_verdict = ll_verdict, shift_available = shift_available, shift_note = note, shift = shift,
            shift_se = shift_se, max_shift_se = max_shift_se, max_shift_param = max_shift_param,
            max_shift_estimate = max_shift_estimate, max_shift_onestep = max_shift_onestep,
            shift_verdict = shift_verdict, thresholds = thresholds, seconds = Float64(seconds))
end

# The result when the check cannot run (`:not_available`) or stops on an error (`:error`): the fields of a
# verdict, with the reason in `reason` and in `shift_note`.
function _sf_integration_missing(status::Symbol, reason::AbstractString; ll, S, thresholds, seconds)
    return (status = status, reason = String(reason), n_draws = Int(S), n_draws_check = 4 * Int(S),
            loglik = Float64(ll), loglik_check = NaN, delta_loglik = NaN, delta_loglik_per_obs = NaN,
            ll_verdict = nothing, shift_available = false, shift_note = String(reason), shift = nothing,
            shift_se = nothing, max_shift_se = NaN, max_shift_param = "", max_shift_estimate = NaN,
            max_shift_onestep = NaN, shift_verdict = nothing, thresholds = thresholds,
            seconds = Float64(seconds))
end

"""
    sf_run_integration_check(settings; S, n_obs, ll, theta, nll_S, build_nll_4S, V, se, hessian_flag, names,
                             unavailable) -> NamedTuple

The integration check of a fit, run once after its variance step. `settings` are the thresholds
(`sf_integration_settings`); `S` is the fit's number of draws per observation and `n_obs` its number of
observations; `ll` is the log-likelihood at the estimate `theta` and `nll_S` the fit's negative
log-likelihood. `build_nll_4S()` returns the negative log-likelihood with 4S draws, the draws a fit with
`n_draws = 4S` and the same settings would use; it is called once, and the draws it generates live only in
this call. `V`, `se` and `hessian_flag` come from the fit's variance step, `names` are the parameter names,
and `unavailable` is `nothing` or the reason the check cannot run (`sf_integration_unavailable`).

The gradients behind the shift are computed only when the variance matrix and the standard errors are usable
and the log-likelihood with 4S draws is finite; `sf_integration_verdict` gives the result. When the check
cannot run, or the log-likelihood at the estimate is not finite, the status is `:not_available`. An
out-of-memory error inside the check prints one red line and also gives `:not_available`; any other error
prints one red line and gives `:error`, with the error as the reason. Ctrl-C and a stack overflow stop the
work.
"""
function sf_run_integration_check(settings::NamedTuple; S::Integer, n_obs::Integer, ll::Real,
                                  theta::AbstractVector, nll_S, build_nll_4S, V::AbstractMatrix,
                                  se::AbstractVector, hessian_flag::Integer, names::AbstractVector,
                                  unavailable)
    t0 = time()
    unavailable === nothing ||
        return _sf_integration_missing(:not_available, unavailable; ll = ll, S = S, thresholds = settings,
                                       seconds = time() - t0)
    isfinite(ll) ||
        return _sf_integration_missing(:not_available, "the log-likelihood at the estimate is not finite";
                                       ll = ll, S = S, thresholds = settings, seconds = time() - t0)
    try
        nll4 = build_nll_4S()
        ll4 = -Float64(nll4(theta))
        usable = hessian_flag == 0 && all(isfinite, V) && all(s -> isfinite(s) && s > 0, se) && isfinite(ll4)
        g_diff = usable ? ForwardDiff.gradient(nll4, theta) .- ForwardDiff.gradient(nll_S, theta) : nothing
        return sf_integration_verdict(; ll = ll, ll_check = ll4, n_obs = n_obs, S = S, S4 = 4 * S, V = V,
                                      se = se, hessian_flag = hessian_flag, g_diff = g_diff, theta = theta,
                                      names = names, thresholds = settings, seconds = time() - t0)
    catch err
        sf_fatal_error(err) && !sf_out_of_memory(err) && rethrow()
        if sf_out_of_memory(err)
            reason = "out of memory: lower n_draws or raise chunks"
            printstyled("The integration check could not be completed ($(reason)).\n"; color = :red)
            return _sf_integration_missing(:not_available, reason; ll = ll, S = S, thresholds = settings,
                                           seconds = time() - t0)
        end
        txt = try                             # the error on one line; its type if it cannot be printed
            first(replace(sprint(showerror, err), r"\s+" => " "), 300)
        catch e
            sf_fatal_error(e) && rethrow()
            string(typeof(err))
        end
        printstyled("The integration check could not be completed ($(txt)).\n"; color = :red)
        return _sf_integration_missing(:error, txt; ll = ll, S = S, thresholds = settings,
                                       seconds = time() - t0)
    end
end

"""
    sf_integration_line(ic) -> String

The text that the results table prints after "Integration check: " for the result `ic` of the integration
check. When the shift is computed, the text ends with the largest shift in standard errors and the
parameter it belongs to, with that parameter's estimate and its value after the shift in the four decimals
of the table's Coef. column, for example "largest shift: 0.18 SE, x1 from 0.9120 to 0.9053".
"""
function sf_integration_line(ic)
    ic.status === :not_available && return "not available ($(ic.reason))"
    ic.status === :error && return "error ($(ic.reason))"
    S, S4 = ic.n_draws, ic.n_draws_check
    Δ = round(ic.delta_loglik; digits = 4)
    ic.shift_available ||
        return "$(ic.status), log-likelihood only (log-likelihood change from $S to $S4 draws: $Δ; " *
               "shift not computed: $(ic.shift_note))"
    return "$(ic.status) (log-likelihood change from $S to $S4 draws: $Δ; largest shift: " *
           "$(round(ic.max_shift_se; digits = 2)) SE, $(ic.max_shift_param) from " *
           "$(@sprintf("%.4f", ic.max_shift_estimate)) to $(@sprintf("%.4f", ic.max_shift_onestep)))"
end

# Shape ranges where the default MCI transformation rule (and MSLE at the default draws, for the
# Lomax) is unreliable: below the threshold, the simulated log-likelihood of a sample of 500
# observations at the default 4,096 draws can lie more than 1 nat from the exact value (manual:
# "Shape ranges where MCI is unreliable"). The Gamma and Weibull thresholds leave out the samples with an
# observation beyond the reach of the draws. For MSLE and the Lomax no shape up to 100 kept every sample
# within 1 nat at 4,096 draws; its threshold is the one found at 8,192 draws.
const SF_SHAPE_RANGES = (
    (method = :MCI,  family = :Gamma,   parameter = :k,     rule = :expo_rule,       threshold = 0.6),
    (method = :MCI,  family = :Weibull, parameter = :k,     rule = :expo_rule,       threshold = 1.0),
    (method = :MCI,  family = :Lomax,   parameter = :alpha, rule = :logistic_1_rule, threshold = 3.0),
    (method = :MSLE, family = :Lomax,   parameter = :alpha, rule = nothing,          threshold = 30.0),
)

"""
    sf_shape_entry(method::Symbol, family::Symbol, rule) -> Union{Nothing, NamedTuple}

The row of `SF_SHAPE_RANGES` for `method`, `family` and the transformation rule `rule` (`nothing` for MSLE), or
`nothing` when the table has none. The MCI rows name the default rules, so a fit under another rule gets
`nothing` and an explicitly passed default rule matches.
"""
function sf_shape_entry(method::Symbol, family::Symbol, rule)
    for row in SF_SHAPE_RANGES
        row.method === method && row.family === family && row.rule === rule && return row
    end
    return nothing
end

"""
    sf_values_below(values, threshold) -> NamedTuple

`(n_below, n, share, min_value)` for a number or a vector of `values`: how many lie below `threshold`, out of
how many, their share, and the smallest value (as Float64).
"""
sf_values_below(v::Real, threshold) =
    (n_below = Int(v < threshold), n = 1, share = Float64(v < threshold), min_value = Float64(v))
function sf_values_below(values::AbstractVector, threshold)
    n_below = count(<(threshold), values)
    n = length(values)
    return (n_below = n_below, n = n, share = n_below / n, min_value = Float64(minimum(values)))
end

"""
    sf_print_warning(text::AbstractString)

Print the warning `text`, which begins with "Warning", in yellow on a line of its own, whatever `show_table`
and `verbose` are.
"""
sf_print_warning(text::AbstractString) = printstyled(text, "\n"; color = :yellow)

# A value below the threshold `t`, with three significant digits, or with as many more as it takes to print
# it below `t` (0.5996 prints as 0.5996, not 0.6, under a threshold of 0.6).
function _sf_below_text(v::Real, t)
    d = 3
    while d < 17 && round(v; sigdigits = d) >= t
        d += 1
    end
    return string(round(v; sigdigits = d))
end

# The share of the observations below the threshold, in percent with one decimal; a share that would print
# as 0.0% or 100.0% while some but not all observations lie below prints as "<0.1%" or ">99.9%".
function _sf_share_text(vb)
    p = round(100 * vb.share; digits = 1)
    if 0 < vb.n_below < vb.n
        p == 0.0 && return "<0.1%"
        p == 100.0 && return ">99.9%"
    end
    return "$(p)%"
end

# The value clause of a shape-range warning, after the parameter's name: "= 0.42 is below the threshold of
# 0.6" when there is one value (or all are equal), otherwise the count below the threshold, its share and
# the smallest value.
function _sf_shape_clause(vb, all_equal::Bool, t)
    (vb.n == 1 || all_equal) && return "= $(_sf_below_text(vb.min_value, t)) is below the threshold of $(t)"
    return "is below the threshold of $(t) for $(vb.n_below) of $(vb.n) observations " *
           "($(_sf_share_text(vb)); smallest $(_sf_below_text(vb.min_value, t)))"
end

# The two sentences that every shape-range warning carries: after the value, that it is about the method at such
# shapes; at the end, how to turn it off.
const _SF_SHAPE_QUALIFIER = "This warning flags a potential numerical issue; it does not establish that this " *
    "fit is affected. "
const _SF_SHAPE_OFF = "This warning may be turned off with sfmodel_fit(..., shape_warning = false); the " *
    "result still records the shape check in result.shape_check."

# The warning of each row of SF_SHAPE_RANGES, with its value clause.
function _sf_shape_message(entry, clause::AbstractString)
    m, f = entry.method, entry.family
    m === :MCI && f === :Gamma && return "Warning (MCI): The estimated Gamma shape parameter k $(clause). " *
        _SF_SHAPE_QUALIFIER * "For shapes in this range the default transformation rule (:expo_rule) puts " *
        "too few draws where the Gamma density spikes near zero, so the simulated log-likelihood, jlms and bc " *
        "can be far from their exact values even when the fit reports convergence, and k tends to be " *
        "estimated too high. To check this fit, use sfmodel_fit(..., integration_check = true); if the check " *
        "fails, refit with transformation = :logistic_2_rule (the Gamma model has no method = :MSLE). " *
        _SF_SHAPE_OFF
    m === :MCI && f === :Weibull && return "Warning (MCI): The estimated Weibull shape parameter k $(clause). " *
        _SF_SHAPE_QUALIFIER * "In this range the Weibull tail is heavier than the exponential tail of the " *
        "default rule's draws (:expo_rule), so the simulated log-likelihood, jlms and bc can be far from " *
        "their exact values even when the fit reports convergence. To check this fit, use sfmodel_fit(..., " *
        "integration_check = true); if the check fails, refit with transformation = :logistic_2_rule or with " *
        "method = :MSLE. " * _SF_SHAPE_OFF
    m === :MCI && f === :Lomax && return "Warning (MCI): The estimated Lomax shape parameter α $(clause). " *
        _SF_SHAPE_QUALIFIER * "With a tail this heavy the default transformation rule (:logistic_1_rule) " *
        "puts few draws where observations far in the tail need them, so the simulated log-likelihood, jlms " *
        "and bc can be far from their exact values even when the fit reports convergence. To check this fit, " *
        "use sfmodel_fit(..., integration_check = true); if the check fails, refit with transformation = " *
        ":logistic_2_rule. " * _SF_SHAPE_OFF
    m === :MSLE && f === :Lomax && return "Warning (MSLE): The estimated Lomax shape parameter α $(clause). " *
        _SF_SHAPE_QUALIFIER * "With a tail this heavy, MSLE may need substantially more draws than the " *
        "default to approximate the exact log-likelihood accurately, even when the fit reports convergence. " *
        "To check this fit, use sfmodel_fit(..., integration_check = true); if the check fails, refit with " *
        "method = :MCI and transformation = :logistic_2_rule, which is generally more accurate for this " *
        "model, or with substantially more draws. " * _SF_SHAPE_OFF
    throw(ArgumentError("SF_SHAPE_RANGES has no warning for $m $f"))
end

"""
    sf_shape_check(entry, values; shape_warning::Bool = true) -> NamedTuple

The shape-range verdict for the row `entry` of `SF_SHAPE_RANGES` and the estimated shape `values` (a number, or
one value per observation). Prints the row's warning when any value lies below `entry.threshold` and
`shape_warning` (the fit's option) is `true`, and returns
`(method, family, parameter, rule, threshold, in_range, n_below, n, share, min_value)` either way.
"""
function sf_shape_check(entry, values; shape_warning::Bool = true)
    vb = sf_values_below(values, entry.threshold)
    in_range = vb.n_below > 0
    if in_range && shape_warning
        all_equal = vb.n == 1 || allequal(values)
        sf_print_warning(_sf_shape_message(entry, _sf_shape_clause(vb, all_equal, entry.threshold)))
    end
    return (method = entry.method, family = entry.family, parameter = entry.parameter, rule = entry.rule,
            threshold = entry.threshold, in_range = in_range, n_below = vb.n_below, n = vb.n, share = vb.share,
            min_value = vb.min_value)
end

"""
    sf_warn_lomax_mean(n_undefined::Int, n::Int; shape_warning::Bool = true, after_fit::Bool = false)

Print one warning through `sf_print_warning` when `n_undefined`, the number of the `n` observations whose
marginal effects on E(u) in a Lomax fit are NaN, is positive: the Lomax mean E(u) = λ/(α − 1) exists only
for α > 1, so the marginal effects are undefined where α ≤ 1 and are reported as NaN. The counts appear as
"(37 of 500 observations)", as "(all 500 observations)" when `n_undefined == n`, and as "(1 observation)"
when `n` is 1. The warning prints whatever `show_table` and `verbose` are and says nothing about the
estimates; nothing prints when `n_undefined` is 0 or `shape_warning` (the fit's option) is `false`. After a
fit (`after_fit = true`) the warning adds how to turn it off; `predict` calls it with the defaults.
"""
function sf_warn_lomax_mean(n_undefined::Int, n::Int; shape_warning::Bool = true, after_fit::Bool = false)
    (n_undefined > 0 && shape_warning) || return nothing
    obs = n == 1 ? "1 observation" :
          n_undefined == n ? "all $n observations" : "$n_undefined of $n observations"
    off = after_fit ? " This warning may be turned off with sfmodel_fit(..., shape_warning = false); the " *
                      "marginal effects are NaN there either way." : ""
    sf_print_warning("Warning: The Lomax mean E(u) = λ/(α − 1) exists only for α > 1, so the marginal " *
                     "effects on E(u) are undefined where α ≤ 1 and are reported as NaN ($obs)." * off)
    return nothing
end

end # module SFAccuracy
