# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

# sf_redflag.jl
# The Hessian part of `redflag`, shared by the four backends. One copy of the Hessian
# classifier and of the clamp test serves MLE, MCI, MSLE and the panel backend, so the same
# fit gets the same verdict from each. Separate copies need not agree: a test that needs an
# exact-zero pivot for "not invertible" gives 1 for the simulation backends (whose sigma clamp
# makes a Hessian row exactly zero) where :MLE classifies the same point from tiny non-zero
# entries (0, or its non-positive-diagonal case), and its verdict on a numerically singular
# Hessian depends on the parameter order.
# The parts of `redflag` that come from Optim's result are shared too: the objective and
# gradient norm at the estimate (sf_final_fg) and the convergence part (sf_convergence_flag).
# For fits with Laplace noise the simulation backends classify, by the same rules, the outer product of
# gradients of the per-observation log-likelihoods in place of the Hessian (sf_classify_opg).

module SFRedflag

using LinearAlgebra: diag, inv, opnorm
using Optim: Optim
using ForwardDiff: ForwardDiff
using Printf: @sprintf

export sf_classify_hessian, sf_scale_at_clamp, SF_SCALE_FLOOR, SF_SCALE_CEIL, sf_final_fg,
       sf_ls_failed, sf_gradient_free, sf_convergence_flag, sf_fatal_error, sf_out_of_memory,
       SF_BOUND_RTOL, sf_bound_side, SFBoundEntry, sf_bound_entries!, sf_bound_message, sf_bound_move,
       sf_start_check, sf_classify_opg

"""
The bounds the simulation backends apply to a variance or scale parameter: σᵤ, σᵥ and the
other scales (Weibull λ, Gamma and Erlang θ, Laplace b) lie in [1e-12, 1e12]; for the
Exponential the bounded quantity is σᵤ² itself. The Lomax λ has its own [1e-10, 1e10], and
shape and copula parameters have their own bounds in the backends' constants. The `:MLE`
likelihoods of `panel_TFE_CSW` and `panel_TRE` bound their variances at [1e-12, 1e12].
Beyond a bound the likelihood is flat in the clamped value; for a homoscedastic parameter its
Hessian row is then exactly zero.
"""
const SF_SCALE_FLOOR = 1e-12
const SF_SCALE_CEIL  = 1e12

"""
The relative tolerance of the boundary verdict: a value counts as at or beyond a bound when it
lies within a relative `SF_BOUND_RTOL` of the bound on the natural scale,
`x <= lo*(1 + SF_BOUND_RTOL)` or `x >= hi*(1 - SF_BOUND_RTOL)`, for every bound.
"""
const SF_BOUND_RTOL = 1e-6

"""
    sf_scale_at_clamp(x, lo = SF_SCALE_FLOOR, hi = SF_SCALE_CEIL) -> Bool

`true` when the value `x` the likelihood clamps is at or beyond a bound within the relative
tolerance `SF_BOUND_RTOL` (1e-6): `x <= lo*(1 + SF_BOUND_RTOL)` or
`x >= hi*(1 - SF_BOUND_RTOL)`; a NaN counts as well.
"""
sf_scale_at_clamp(x::Real, lo::Real = SF_SCALE_FLOOR, hi::Real = SF_SCALE_CEIL) =
    !(lo * (1 + SF_BOUND_RTOL) < x < hi * (1 - SF_BOUND_RTOL))

"""
    sf_bound_side(x, lo, hi) -> Union{Symbol, Nothing}

The side of the bounds `lo` and `hi` at or beyond which the value `x` lies, within the
relative tolerance `SF_BOUND_RTOL`: `:lower` when `x <= lo*(1 + SF_BOUND_RTOL)`, `:upper`
when `x >= hi*(1 - SF_BOUND_RTOL)`, otherwise `nothing`; `nothing` for a NaN. `lo = -Inf`
means that no lower bound is tested.
"""
function sf_bound_side(x::Real, lo::Real, hi::Real)
    isnan(x) && return nothing
    x <= lo * (1 + SF_BOUND_RTOL) && return :lower
    x >= hi * (1 - SF_BOUND_RTOL) && return :upper
    return nothing
end

"""
    SFBoundEntry

One entry of a fit's boundary verdict, `bound_check`: a NamedTuple with
- `parameter::Symbol`: the bounded quantity, for example `:sigma_u`, `:k`, `:alpha` or `:rho`;
- `label::String`: its printed name, for example "σᵤ" or "α";
- `coefficient::String`: the name of the coefficient it is formed from, for example "ln_σᵤ²";
- `side::Symbol`: `:lower` or `:upper`;
- `bound::Float64`: the bound on that side, on the natural scale;
- `scalar::Bool`: `true` for a parameter with one value for all observations, `false` for one
  that varies over the observations (a heteroscedastic parameter);
- `n_at::Int`, `n::Int` and `share::Float64`: the number of observations at or beyond the
  bound, the number of observations, and `n_at / n`;
- `moved::Bool` and `coef_before::Float64`: whether the reported coefficient was moved to the
  bound, and its value before the move (`false` and `NaN` when it was not moved).
"""
const SFBoundEntry = NamedTuple{(:parameter, :label, :coefficient, :side, :bound, :scalar, :n_at, :n, :share,
                                 :moved, :coef_before),
                                Tuple{Symbol, String, String, Symbol, Float64, Bool, Int, Int, Float64, Bool, Float64}}

"""
    sf_bound_entries!(out, values, lo, hi, nobs; parameter, label, coefficient, scalar) -> out

Pushes onto `out::Vector{SFBoundEntry}` the entries of one bounded quantity. `values` are the
clamped natural-scale values the likelihood uses, a host scalar or a host vector, and `lo` and
`hi` its bounds (`lo = -Inf`: no lower bound tested). For a scalar, `n = nobs` and `n_at` is
`nobs` or 0; for a vector, `n = length(values)` and `n_at` counts the values on each side
(`sf_bound_side`; a NaN counts on neither). One entry per side with `n_at > 0`, the lower side
first, with `bound` the bound of that side, `share = n_at / n`, `moved = false` and
`coef_before = NaN`; `scalar` is stored as given.
"""
function sf_bound_entries!(out::Vector{SFBoundEntry}, values, lo::Real, hi::Real, nobs::Integer;
                           parameter::Symbol, label::AbstractString, coefficient::AbstractString, scalar::Bool)
    if values isa AbstractArray
        n = length(values)
        n_lo = count(x -> sf_bound_side(x, lo, hi) === :lower, values)
        n_hi = count(x -> sf_bound_side(x, lo, hi) === :upper, values)
    else
        n = Int(nobs)
        side = sf_bound_side(values, lo, hi)
        n_lo = side === :lower ? n : 0
        n_hi = side === :upper ? n : 0
    end
    n_lo > 0 && push!(out, SFBoundEntry((parameter, String(label), String(coefficient), :lower, Float64(lo),
                                         scalar, n_lo, n, n_lo / n, false, NaN)))
    n_hi > 0 && push!(out, SFBoundEntry((parameter, String(label), String(coefficient), :upper, Float64(hi),
                                         scalar, n_hi, n, n_hi / n, false, NaN)))
    return out
end

# The share of the observations at or beyond the bound, in percent with one decimal; a share that would print as
# 0.0% or 100.0% while some but not all observations are at or beyond the bound prints as "<0.1%" or ">99.9%".
function _sf_bound_share_text(e)
    p = @sprintf("%.1f", 100 * e.share)
    if 0 < e.n_at < e.n
        p == "0.0" && return "<0.1%"
        p == "100.0" && return ">99.9%"
    end
    return p * "%"
end

"""
    sf_bound_message(bc::Vector{SFBoundEntry}) -> String

The boundary message that names each entry of `bc`: "A parameter is at or beyond the numerical
bound SFrontiers applies to it, a boundary solution: " followed by one part per entry, joined by
"; ", each reading "<label> (coefficient <coefficient>) at its <lower|upper> bound <bound>",
then ". See Parameter Bounds in the manual." and a newline. A parameter that varies over the
observations adds " for <n_at> of <n> observations (<share in percent>%)", the share with one
decimal, or "<0.1%" and ">99.9%" where it would print as 0.0% or 100.0% while some but not all
observations are at or beyond the bound; one whose coefficient was moved to the bound adds
" (coefficient reported at the bound)".
"""
function sf_bound_message(bc::Vector{SFBoundEntry})::String
    parts = map(bc) do e
        s = "$(e.label) (coefficient $(e.coefficient)) at its $(e.side) bound $(@sprintf("%g", e.bound))"
        e.scalar || (s *= " for $(e.n_at) of $(e.n) observations ($(_sf_bound_share_text(e)))")
        e.moved && (s *= " (coefficient reported at the bound)")
        s
    end
    return "A parameter is at or beyond the numerical bound SFrontiers applies to it, a boundary solution: " *
           join(parts, "; ") * ". See Parameter Bounds in the manual.\n"
end

# The natural value of a coefficient θ (`:exp`: exp(θ) + offset; `:exp_half`: exp(θ/2)), and the coefficient whose
# natural value is the bound b.
_sf_bound_natural(form::Symbol, offset::Float64, θ::Float64) = form === :exp_half ? exp(θ / 2) : exp(θ) + offset
_sf_bound_coef(form::Symbol, offset::Float64, b::Float64) = form === :exp_half ? 2 * log(b) : log(b - offset)

"""
    sf_bound_move(list, bc, p) -> (bc2, p2)

Moves to its bound every homoscedastic coefficient of `p`, the optimizer's point, that lies beyond the
numerical bound at which its likelihood clamps it. For each item of the bound list `list` that is
`scalar` and `movable`, the unclamped natural value g(θ) of θ = `p[block[1]]` is formed in Float64
(form `:exp`: exp(θ) + `offset`; `:exp_half`: exp(θ/2)). When g(θ) > `hi`, θ is replaced by
g⁻¹(`hi`) + m, and when g(θ) < `lo`, by g⁻¹(`lo`) − m, where g⁻¹(b) is log(b − `offset`) for `:exp`
and 2·log(b) for `:exp_half`, and m is 1e-12 for `:exp` and 2e-12 for `:exp_half`; the new value is
used only when g of it lies beyond the bound as well. Its natural value lies beyond the bound by a
relative 1e-12, so the likelihood clamps it on every device and uses the bound itself: the log-likelihood,
the gradient and the Hessian there equal those at θ. The entry of a moved quantity in `bc` (matched by
`parameter`) gets `moved = true` and `coef_before = θ`. Heteroscedastic items, items within their
bounds and items the likelihood does not clamp are left as they are. `bc2` and `p2` are copies when
a coefficient moves, and `bc` and `p` themselves otherwise; `p` is not modified.
"""
function sf_bound_move(list, bc::Vector{SFBoundEntry}, p::AbstractVector)
    bc2, p2 = bc, p
    for it in list
        (it.scalar && it.movable) || continue
        i = it.block[1]
        θ = Float64(p[i])
        m = it.form === :exp_half ? 2e-12 : 1e-12
        v = _sf_bound_natural(it.form, it.offset, θ)
        if v > it.hi
            θnew = _sf_bound_coef(it.form, it.offset, it.hi) + m
            _sf_bound_natural(it.form, it.offset, θnew) > it.hi || continue
        elseif v < it.lo
            θnew = _sf_bound_coef(it.form, it.offset, it.lo) - m
            _sf_bound_natural(it.form, it.offset, θnew) < it.lo || continue
        else
            continue
        end
        p2 === p && (p2 = copy(p))
        p2[i] = θnew
        bc2 === bc && (bc2 = copy(bc))
        for j in eachindex(bc2)
            bc2[j].parameter === it.parameter && (bc2[j] = merge(bc2[j], (moved = true, coef_before = θ)))
        end
    end
    return bc2, p2
end

"""
    sf_start_check(list, bc, p) -> (error_text, warnings)

The check of the starting values `p` before the optimizer. `bc` is the boundary verdict at `p` and
`list` the bound list it was formed from; each entry is matched to its item by `parameter`. Each
homoscedastic entry adds one paragraph to `error_text` (one paragraph per line): "The starting
value of <label> is <value> (<keyword> = <θ>), at or beyond its <lower|upper> bound <bound>. …",
with θ = `p[block[1]]` and <value> its natural value g(θ) before the clamp (form `:exp`:
g(θ) = exp(θ) + `offset`; `:exp_half`: g(θ) = exp(θ/2)), ending with the interval of the
coefficient inside the bounds, from g⁻¹(`lo`) to g⁻¹(`hi`) ("<keyword> below g⁻¹(`hi`)" when
`lo = -Inf`); when g(θ) lies inside the bound but within the relative tolerance `SF_BOUND_RTOL`,
the paragraph adds that such a start counts as at the bound. Each heteroscedastic entry adds one
text to `warnings`: "at the starting values, <label> is at or beyond its <lower|upper> bound
<bound> for <n\\_at> of <n> observations (<share in percent>%); …", the share written as in the
boundary message (`sf_bound_message`: one decimal, or "<0.1%" and ">99.9%"). Both are empty when
nothing is at a bound. The fits print each warning and then, when `error_text` is not empty, throw
it as an `ArgumentError`; a fit held at its starting values is not checked.
"""
function sf_start_check(list, bc::Vector{SFBoundEntry}, p::AbstractVector)
    paragraphs = String[]
    warnings = String[]
    for e in bc
        it = list[findfirst(x -> x.parameter === e.parameter, list)]
        bound = @sprintf("%g", e.bound)
        if e.scalar
            θ = Float64(p[it.block[1]])
            v = _sf_bound_natural(it.form, it.offset, θ)
            coef_at(b) = @sprintf("%.4f", _sf_bound_coef(it.form, it.offset, b))
            interval = it.lo == -Inf ? "$(it.keyword) below $(coef_at(it.hi))" :
                       "$(it.keyword) strictly between $(coef_at(it.lo)) and $(coef_at(it.hi))"
            s = "The starting value of $(e.label) is $(@sprintf("%g", v)) ($(it.keyword) = $(@sprintf("%.4f", θ))), " *
                "at or beyond its $(e.side) bound $bound. Beyond a bound the log-likelihood does not depend on this " *
                "coefficient, so a gradient-based optimizer cannot move it. Give a starting value inside the bounds: " *
                interval * "."
            # a natural value inside the bound that the tolerance counts as at the bound
            (e.side === :upper ? v < it.hi : v > it.lo) &&
                (s *= " A starting value within a relative 1e-6 of a bound counts as at the bound.")
            push!(paragraphs, s)
        else
            push!(warnings, "at the starting values, $(e.label) is at or beyond its $(e.side) bound $bound for " *
                            "$(e.n_at) of $(e.n) observations ($(_sf_bound_share_text(e))); at those " *
                            "observations the log-likelihood does not depend on the coefficients of $(it.keyword). " *
                            "Consider other starting values.")
        end
    end
    return join(paragraphs, "\n"), warnings
end

# The boundary message of sf_classify_hessian when the fit passes no text that names its parameters.
const _SF_BOUND_GENERIC = "A parameter is at or beyond the numerical bound SFrontiers applies to it: " *
                          "a boundary solution. See Parameter Bounds in the manual.\n"

# The boundary message of sf_classify_hessian and sf_classify_opg, in red: `bound_text` when it is not empty (the
# fits pass the text that names each parameter), otherwise the generic text.
_sf_boundary_message(bound_text::AbstractString) =
    printstyled(isempty(bound_text) ? _SF_BOUND_GENERIC : bound_text; color = :red)

# The positive-definiteness test of sf_classify_hessian.
using LinearAlgebra: Symmetric, cholesky, issuccess

"""
    sf_classify_hessian(H; at_clamp = false, message = true, bound_text = "")
        -> (var_cov_matrix, redflag, hessian_flag)

The Hessian part of `redflag`. `H` is the Hessian of the objective
the fit minimizes, the negative log-likelihood, at the estimate. At a local maximum of
the log-likelihood it is positive semidefinite, and it is positive definite where the
log-likelihood curves down in every direction (the second-order condition for a strict
local maximum). The rules, in this order:

- 1 when `inv(H)` fails, or when the 1-norm reciprocal condition number
  1/(‖H‖₁ ‖H⁻¹‖₁) is below machine epsilon, so that the inverse may be inaccurate.
  The var-cov matrix is then all NaN, as for a Hessian that cannot be inverted at all.
  The number is not scale-free: besides collinearity or a flat likelihood, variables on
  very different scales can make it that small.
- otherwise 2 when some diagonal element of H⁻¹ is not positive (the affected
  standard errors become NaN downstream);
- otherwise 1 when H is not positive definite, that is, when the Cholesky factorization
  of its symmetric part (H + H')/2 fails. The estimate is then not a strict local maximum
  of the log-likelihood: a saddle point, or a point on a ridge along which the likelihood
  does not fall. The test is strict: an eigenvalue that is negative only by rounding fails
  it too. The var-cov matrix is kept as computed, so the standard errors are finite, but
  they are not reliable;
- otherwise 0.
With `message = true` the case is printed (one message about the Hessian at most).

`at_clamp = true` means that a parameter of the fit is at or beyond the numerical bound its
likelihood applies (a variance, scale, shape or copula dependence parameter): a boundary
solution. It overrides the result to 1, also over 2, and leaves the matrix as computed.
`hessian_flag` is the value before that override. With `message = true` the boundary message
comes first: `bound_text` when it is not empty (the fits pass the text that names each
parameter), otherwise a generic one.

The reciprocal condition number and positive definiteness do not depend on the order of
the parameters, so a permuted Hessian gets the same verdict.
"""
function sf_classify_hessian(H::AbstractMatrix{T}; at_clamp::Bool = false,
                             message::Bool = true, bound_text::AbstractString = "") where {T<:Real}
    F = float(T)
    if at_clamp && message
        _sf_boundary_message(bound_text)
    end

    V = try
        inv(H)
    catch
        nothing
    end
    if V === nothing
        message && printstyled("The Hessian matrix is not invertible, indicating the model does not converge properly. Standard errors are set to NaN.\n"; color = :red)
        return (var_cov_matrix = fill!(similar(H, F), F(NaN)), redflag = 1, hessian_flag = 1)
    end

    rcond = inv(opnorm(H, 1) * opnorm(V, 1))
    if !(rcond >= eps(F))    # also catches NaN and Inf entries
        message && printstyled("The Hessian matrix is numerically singular (reciprocal condition number $(rcond), below machine epsilon): the model may not be identified or may not converge properly, or its variables may be on very different scales. Standard errors are set to NaN.\n"; color = :red)
        return (var_cov_matrix = fill!(similar(H, F), F(NaN)), redflag = 1, hessian_flag = 1)
    end

    hflag = 0
    if !all(diag(V) .> 0)
        hflag = 2
        message && printstyled("Some diagonal elements of the var-cov matrix are non-positive, indicating problems in the convergence. The affected standard errors are set to NaN.\n"; color = :red)
    elseif !issuccess(cholesky(Symmetric((H + H') / 2); check = false))
        # A positive-definite Hessian of the negative log-likelihood is the second-order
        # condition for a strict local maximum of the log-likelihood. A Hessian that passes
        # the tests above but not this one (a saddle point, or a ridge along which the
        # likelihood does not fall) gets 1. The var-cov matrix is kept.
        hflag = 1
        message && printstyled("The Hessian matrix is not positive definite: the estimate is not a strict local maximum of the log-likelihood (a saddle point, or a ridge along which the likelihood does not fall). The standard errors are not reliable.\n"; color = :red)
    end
    return (var_cov_matrix = V, redflag = at_clamp ? 1 : hflag, hessian_flag = hflag)
end

"""
    sf_ls_failed(optres) -> Bool

`true` when Optim stopped because the line search failed (`stopped_by.ls_failed`, Optim 2);
`false` for a result that does not carry the field.
"""
function sf_ls_failed(optres)
    sb = hasproperty(optres, :stopped_by) ? optres.stopped_by : nothing
    return sb isa NamedTuple && get(sb, :ls_failed, false) === true
end

"""
    sf_gradient_free(optres) -> Bool

`true` when the solver that produced `optres` does not use gradients: Optim's
`ZerothOrderOptimizer`s, i.e. `NelderMead`, `ParticleSwarm` and `SimulatedAnnealing`. Their
`g_residual` is not a gradient norm: for NelderMead it is the spread of the objective over the
simplex, the quantity its stopping rule tests against `g_abstol`; for ParticleSwarm and
SimulatedAnnealing it is NaN.
"""
sf_gradient_free(optres) =
    hasproperty(optres, :method) && getproperty(optres, :method) isa Optim.ZerothOrderOptimizer

"""
    sf_final_fg(optres, f) -> (fmin, gnorm)

The objective `f` and the sup-norm of its gradient at `Optim.minimizer(optres)`, the point
every backend takes the coefficients, the Hessian and the standard errors from. The backends
use them for `loglikelihood`, `gradient_norm`, the gradient part of `redflag` and the printed
log-likelihood. Optim's own `minimum` and `g_residual` are returned unchanged except in the
following cases.

Optim (2.0, the version SFrontiers is tested with) can return a minimizer that its own
`minimum` and `g_residual` do not describe:
- the line search failed (`stopped_by.ls_failed`): Optim has already moved `x` by the step
  the line search returned and stops before evaluating the objective and gradient there, so
  `minimum` and `g_residual` describe the previous iterate;
- the objective increased with `allow_f_increases = false` (`stopped_by.f_increased`): Optim
  returns the previous point and its objective, but `g_residual` of the rejected point.
In these two cases `f` and its ForwardDiff gradient are evaluated at the minimizer, once (for
some solvers, and for a line search that returned a zero step, Optim's values were already
fresh and the re-evaluation only repeats them).

A gradient-free solver (`sf_gradient_free`: NelderMead, ParticleSwarm, SimulatedAnnealing)
has no gradient to report; its `g_residual` is NelderMead's simplex spread or NaN. Its
`minimum` is the objective at the point it returns, and is kept; the ForwardDiff gradient of
`f` is evaluated at the minimizer.

So the second value is always the sup-norm of the gradient at the minimizer, the measure
Optim reports as `g_residual` for gradient-based solvers.

When `f` or its gradient throws an error at the minimizer, the fit is not ended. The second value is then
`NaN`, so the gradient part of `redflag` flags the fit; the first is `f` at the minimizer when
`f` itself could be evaluated there, and otherwise Optim's own `minimum` (after a failed line
search under Optim 2.0 that is the objective of the iterate before the minimizer; the line
searches of LineSearches.jl normally evaluate `f` at the step they report, at the minimizer or
at a point that differs from it in the last bits, since Optim 2.0 forms `x + alpha * s` and
LineSearches.jl `muladd(alpha, s, x)`, so `f` fails at the minimizer only by an error that did
not occur in that evaluation). Optim's
`g_residual` is not used in place of the gradient norm: in these cases it belongs to another
point, or is not a gradient norm. A red message gives the error. Errors for which
`sf_fatal_error` is true (Ctrl-C, out of memory, a stack overflow, CUDA's out-of-GPU-memory
error) are rethrown.
"""
function sf_final_fg(optres, f)
    sb = hasproperty(optres, :stopped_by) ? optres.stopped_by : nothing
    stale = sb isa NamedTuple && (get(sb, :ls_failed, false) === true || get(sb, :f_increased, false) === true)
    if stale || sf_gradient_free(optres)
        x = Optim.minimizer(optres)
        fmin = Optim.minimum(optres)   # a gradient-free solver's value at x; replaced by f(x) when stale
        fdone = !stale
        try
            if stale
                fmin = f(x)
                fdone = true
            end
            return fmin, maximum(abs, ForwardDiff.gradient(f, x))
        catch err
            sf_fatal_error(err) && rethrow()
            txt = try                         # the error on one line; its type if it cannot be printed
                first(replace(sprint(showerror, err), r"\s+" => " "), 300)
            catch e
                sf_fatal_error(e) && rethrow()
                string(typeof(err))
            end
            if fdone
                printstyled("The gradient of the log-likelihood could not be evaluated at the estimate ($(txt)). The gradient norm is set to NaN.\n"; color = :red)
            else
                printstyled("The log-likelihood could not be evaluated at the estimate ($(txt)). The log-likelihood reported is Optim's own value, which after a failed line search under Optim 2.0 is that of the previous iterate, and the gradient norm is set to NaN.\n"; color = :red)
            end
            return fmin, oftype(float(fmin), NaN)
        end
    end
    return Optim.minimum(optres), Optim.g_residual(optres)
end

"""
    sf_fatal_error(err) -> Bool

`true` for an error that must stop the work instead of being reported as a Hessian that could
not be computed, a failed evaluation at the estimate or a failed bootstrap resample: Ctrl-C
(`InterruptException`), `OutOfMemoryError`, `StackOverflowError`, and CUDA's out-of-GPU-memory
error. The last is recognised without loading CUDA: by its type name, which contains `OutOfGPUMemory`
(`CUDA.OutOfGPUMemoryError`), or by its message, which contains "out of gpu memory" in any case
(a script may stop its GPU job on that text). An error raised in a spawned task arrives wrapped
(`TaskFailedException`, `CompositeException`, `CapturedException`); the wrapped errors are
tested. The message is formatted (`showerror`) only for an error type defined outside Julia's
`Core` and `Base`; for those two only a text `msg` field is read, because formatting, for
example, a `MethodError` that names ForwardDiff's dual types takes tens of milliseconds and the
MLE bootstrap applies this test to every failed resample.
"""
function sf_fatal_error(err)
    (err isa InterruptException || err isa OutOfMemoryError || err isa StackOverflowError) && return true
    err isa TaskFailedException && return sf_fatal_error(err.task.exception)
    err isa CompositeException && return any(sf_fatal_error, err.exceptions)
    err isa CapturedException && return sf_fatal_error(err.ex)
    T = typeof(err)
    occursin("OutOfGPUMemory", string(nameof(T))) && return true
    txt = if err isa AbstractString
        err
    elseif parentmodule(T) === Core || parentmodule(T) === Base
        (hasfield(T, :msg) && getfield(err, :msg) isa AbstractString) ? getfield(err, :msg) : ""
    else
        try
            sprint(showerror, err)
        catch e
            (e isa InterruptException || e isa OutOfMemoryError || e isa StackOverflowError) && rethrow()
            ""
        end
    end
    return occursin("out of gpu memory", lowercase(txt))
end

"""
    sf_out_of_memory(err) -> Bool

`true` for an out-of-memory error: `OutOfMemoryError`, or CUDA's out-of-GPU-memory error,
recognised as `sf_fatal_error` recognises it (by its type name, which contains `OutOfGPUMemory`,
or by its message, which contains "out of gpu memory" in any case). An error raised in a spawned
task arrives wrapped (`TaskFailedException`, `CompositeException`, `CapturedException`); the
wrapped errors are tested. The integration check reports such an error as "not available"
instead of stopping the work.
"""
function sf_out_of_memory(err)
    err isa OutOfMemoryError && return true
    err isa TaskFailedException && return sf_out_of_memory(err.task.exception)
    err isa CompositeException && return any(sf_out_of_memory, err.exceptions)
    err isa CapturedException && return sf_out_of_memory(err.ex)
    T = typeof(err)
    occursin("OutOfGPUMemory", string(nameof(T))) && return true
    txt = if err isa AbstractString
        err
    elseif parentmodule(T) === Core || parentmodule(T) === Base
        (hasfield(T, :msg) && getfield(err, :msg) isa AbstractString) ? getfield(err, :msg) : ""
    else
        try
            sprint(showerror, err)
        catch e
            (e isa InterruptException || e isa OutOfMemoryError || e isa StackOverflowError) && rethrow()
            ""
        end
    end
    return occursin("out of gpu memory", lowercase(txt))
end

"""
    sf_classify_hessian(f, x; at_clamp = false, message = true, bound_text = "")
        -> (var_cov_matrix, redflag, hessian_flag, hessian)

The Hessian of `f` at `x` (`ForwardDiff.hessian`), classified as `sf_classify_hessian(H)`
classifies it, together with the Hessian itself. The four backends compute the final Hessian
of a fit this way. When computing it throws an error, the fit keeps its estimate and the result
is the one for a Hessian that cannot be inverted: `hessian` and `var_cov_matrix` all `NaN`,
`redflag = 1` and `hessian_flag = 1`; a red message gives the error. That message is printed
whatever `message` is, since it reports an error that would otherwise stop the fit; with
`at_clamp` and `message` the boundary message comes first, as from the matrix method. Errors for which
`sf_fatal_error` is true (Ctrl-C, out of memory, a stack overflow, CUDA's out-of-GPU-memory
error) are rethrown. Without an error the result is that of the matrix method applied to
`ForwardDiff.hessian(f, x)`, plus that matrix. With `Newton()` or `NewtonTrustRegion()` as the
main solver, Optim computes the Hessian at every point it accepts, the estimate included (not
after a failed line search under Optim 2.0), so an error that recurs there ends the fit inside
`optimize` before this method runs. `KrylovTrustRegion()` computes Hessian-vector products only
at the points from which it tries a step, not at a point it has just accepted (where a fit that
converges stops): an error at every point ends its fit inside `optimize` at the first
iteration, an error only at such an estimate reaches this method.

`at_clamp = true` means that a parameter of the fit is at or beyond the numerical bound its
likelihood applies (a variance, scale, shape or copula dependence parameter): a boundary
solution. It overrides the result to 1, also over 2, and leaves the matrix as computed.
`hessian_flag` is the value before that override. With `message = true` the boundary message
comes first: `bound_text` when it is not empty (the fits pass the text that names each
parameter), otherwise a generic one.
"""
function sf_classify_hessian(f, x::AbstractVector{<:Real}; at_clamp::Bool = false,
                             message::Bool = true, bound_text::AbstractString = "")
    H = try
        ForwardDiff.hessian(f, x)
    catch err
        sf_fatal_error(err) && rethrow()
        F = float(eltype(x))
        n = length(x)
        txt = try                             # the error on one line; its type if it cannot be printed
            first(replace(sprint(showerror, err), r"\s+" => " "), 300)
        catch e
            sf_fatal_error(e) && rethrow()
            string(typeof(err))
        end
        if at_clamp && message    # the boundary message of the matrix method, same text
            _sf_boundary_message(bound_text)
        end
        printstyled("The Hessian matrix could not be computed at the estimate ($(txt)). Standard errors are set to NaN.\n"; color = :red)
        return (var_cov_matrix = fill(F(NaN), n, n), redflag = 1, hessian_flag = 1,
                hessian = fill(F(NaN), n, n))
    end
    return merge(sf_classify_hessian(H; at_clamp = at_clamp, message = message, bound_text = bound_text),
                 (hessian = H,))
end

"""
    sf_classify_opg(f_obs, x; at_clamp = false, message = true, bound_text = "")
        -> (var_cov_matrix, redflag, hessian_flag, hessian, opg)

The variance step of a fit from the outer product of gradients (OPG; Berndt, Hall, Hall and
Hausman 1974). `f_obs(x)` returns the N per-observation log-likelihoods at `x` (not negated);
J = `ForwardDiff.jacobian(f_obs, x)` is their N × p Jacobian and G = J'J the OPG. The simulation
backends use it in place of the Hessian for fits with Laplace noise.

G is classified by `sf_classify_hessian(G)` with its rules unchanged (inverse, reciprocal
condition number, diagonal, Cholesky), and its inverse is the var-cov matrix. G is positive
semidefinite, so a failure means a direction without information: a parameter at a bound or not
identified. `at_clamp` overrides `redflag` to 1, as in `sf_classify_hessian`, and `hessian_flag`
is the value before that override. With `message = true` the boundary message comes first when
`at_clamp` (`bound_text` when it is not empty, otherwise a generic one), then at most one message
about G: that it cannot be inverted reliably, that some diagonal elements of the var-cov matrix are
non-positive, or that it is not positive definite.

The result is that of `sf_classify_hessian(G)` with `hessian = nothing` and `opg = G`. When
computing J throws an error, `var_cov_matrix` and `opg` are all `NaN`, `redflag = 1`,
`hessian_flag = 1` and `hessian = nothing`, and a red message gives the error, printed whatever
`message` is (after the boundary message when `at_clamp` and `message`). Errors for which
`sf_fatal_error` is true (Ctrl-C, out of memory, a stack overflow, CUDA's out-of-GPU-memory error)
are rethrown.
"""
function sf_classify_opg(f_obs, x::AbstractVector{<:Real}; at_clamp::Bool = false,
                         message::Bool = true, bound_text::AbstractString = "")
    J = try
        ForwardDiff.jacobian(f_obs, x)
    catch err
        sf_fatal_error(err) && rethrow()
        F = float(eltype(x))
        n = length(x)
        txt = try                             # the error on one line; its type if it cannot be printed
            first(replace(sprint(showerror, err), r"\s+" => " "), 300)
        catch e
            sf_fatal_error(e) && rethrow()
            string(typeof(err))
        end
        if at_clamp && message
            _sf_boundary_message(bound_text)
        end
        printstyled("The outer product of gradients could not be computed at the estimate ($(txt)). " *
                    "Standard errors are set to NaN.\n"; color = :red)
        return (var_cov_matrix = fill(F(NaN), n, n), redflag = 1, hessian_flag = 1, hessian = nothing,
                opg = fill(F(NaN), n, n))
    end
    G = J' * J
    r = sf_classify_hessian(G; at_clamp = at_clamp, message = false)
    if message
        if at_clamp
            _sf_boundary_message(bound_text)
        end
        if all(isnan, r.var_cov_matrix)
            printstyled("The outer product of gradients cannot be inverted reliably (singular, or reciprocal " *
                        "condition number below machine epsilon): a parameter may be at a bound or not identified, " *
                        "or the variables may be on very different scales. Standard errors are set to NaN.\n";
                        color = :red)
        elseif r.hessian_flag == 2
            printstyled("Some diagonal elements of the var-cov matrix are non-positive. The affected standard " *
                        "errors are set to NaN.\n"; color = :red)
        elseif r.hessian_flag == 1
            printstyled("The outer product of gradients is not positive definite: the standard errors are not " *
                        "reliable.\n"; color = :red)
        end
    end
    return merge(r, (hessian = nothing, opg = G))
end

"""
    sf_convergence_flag(optres; message = true) -> Int

The part of `redflag` that comes from how Optim ended the main stage, shared by the four
backends:
- 1 when Optim does not report convergence. This includes a run stopped by the
  iteration limit, but a fit that converges on exactly its last allowed iteration is not
  flagged. Optim never reports convergence for ParticleSwarm and SimulatedAnnealing, which
  run to the iteration limit.
- 1 when the line search failed (`sf_ls_failed`) although Optim reports convergence.
  Optim 2.0 leaves its loop on a failed line search before it tests convergence, so
  `converged` then repeats the verdict of the iteration before, which is true when the f
  test was met there (with a user-set f tolerance, or an objective that did not change at
  all).
- 0 otherwise.
With `message = true` it prints which case it was.
"""
function sf_convergence_flag(optres; message::Bool = true)
    if !Optim.converged(optres)
        if message
            if Optim.iteration_limit_reached(optres)
                printstyled("Caution: The number of iterations reached the limit.\n\n"; color = :red)
            else
                printstyled("Caution: Optim did not report convergence.\n\n"; color = :red)
            end
        end
        return 1
    elseif sf_ls_failed(optres)
        message && printstyled("Caution: The line search failed at the last iteration; Optim's report of convergence refers to the iteration before it.\n\n"; color = :red)
        return 1
    end
    return 0
end

end # module SFRedflag
