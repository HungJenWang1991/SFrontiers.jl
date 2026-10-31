# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

# y x         σᵤ²  σᵥ² -- expo
# y x         σᵤ²  σᵥ² -- half
# y x μ       σᵤ²  σᵥ² -- trun
# y x μ  h    σᵤ²  σᵥ² -- scal
# y x    h    σᵤ²  σᵥ² -- TFE_WH2010, half
# y x μ  h    σᵤ²  σᵥ² -- TFE_WH2010, truncated
# y x μ  g    σᵤ²  σᵥ² -- decay
# y x         σᵤ²  σᵥ² -- panel half (2014 JoE)
# y x    σₐ²  σᵤ²  σᵥ² -- TRE
# ------------------------------------------
# y x z  q    w    v   -- generic varname
#   β δ1 τ    δ2   γ   -- coeff

# Clamp constants for numerical stability (aligned with SFrontiers pattern)
const _MLE_CLAMP = (
    exp_lo  = 1e-12,   # floor for exp() results (σ², σᵤ², etc.)
    exp_hi  = 1e12,    # ceiling for exp() results
    log_lo  = 1e-15,   # floor for log() arguments
)

    # reduce the overhead of normpdf(a,b,c); safe in how it is used in the current module
sfnormpdf(μ::Real, σ::Real, x::Number) =
  exp(-0.5 * StatsFuns.abs2( StatsFuns.zval(μ, σ, x) )) * invsqrt2π / σ

    # log Φ(z) for the likelihoods below, which Newton differentiates twice.
    # StatsFuns.normlogcdf(z) = log(erfcx(-z/√2)/2) - z²/2
    # below z = -1 is accurate as a VALUE, but ForwardDiff differentiates it through
    # DiffRules' rule erfcx'(x) = 2(x·erfcx(x) - 1/√π), whose two terms cancel to O(1/x²),
    # so the second derivative of log Φ carries a relative error of about eps·z²: 6e-7 at
    # z = -6.65e4, where the true value is -1 + 1/z². In LL_T(Trun) at a σᵤ → 0 corner that
    # error is multiplied by (∂z/∂θ)² ≈ 1e10, and the two log Φ terms, which cancel exactly,
    # stop cancelling in the error. At such a point in one simulated data set (n = 1000),
    # Hessian entries were off by up to 1.6e4 and the smallest eigenvalue was as low as
    # -2.2e4, where the truth is 0.15. Newton stalled at its iteration limit; with an
    # accurate Hessian it reaches the maximum from the stall points in 1-3 steps (6-16 from
    # the starting values of that fit). The gradient was unaffected by this
    # defect. Its remaining error at that optimum (about 1.2e-6, where |g| is about 7e-6)
    # comes from the value-level cancellation between the two log Φ terms, which the tail
    # branch of LL_T(Trun) and LL_T(Trun_Scale) and _tn_panel_core below remove.
    #
    # Below z = -40 the asymptotic series
    #   log Φ(z) = -z²/2 - log(-z) - log√(2π) + log(1 - 1/z² + 3/z⁴ - 15/z⁶ + 105/z⁸ - 945/z¹⁰ + …)
    # is used instead. Every piece has derivatives that ForwardDiff forms without
    # cancellation, and the first omitted term, 10395/z¹², is below 1e-15 at the switch, so
    # the value agrees with StatsFuns to rounding. Above -40, this is StatsFuns.normlogcdf
    # itself; its second derivative there is off by at most about 6.5e-13.
function _stable_normlogcdf(z)
    if z < -40
        x = inv(z*z)
        return -z*z/2 - log(-z) - log(sqrt2π) + log1p(-x*(1 - 3x*(1 - 5x*(1 - 7x*(1 - 9x)))))
    else
        return normlogcdf(z)
    end
end

    # The truncated-normal likelihoods below difference two log Φ whose arguments,
    # a₀ = μ/σᵤ and a₁ = μ*/σ*, both run off to -∞ together as σᵤ → 0 with μ < 0. Each
    # log Φ is then about -a²/2, and their O(1) difference, formed from two Float64s,
    # carries a value error of eps·a²/2 (1.1e-6 at a = -1e5, 2.45 nats on one fit at
    # ln σᵤ² = -28.9, +2.5e28 for a true -3.4e14 at another) and a gradient floor of
    # eps·|a|·|∂a/∂θ|, which reached 1.2e-6 against a g_tol of 1e-5. The panel likelihoods
    # also carry ½(a₁² - a₀²) explicitly, so there the bulk cancels as well when both
    # arguments run off to +∞ (σᵤ → 0 with μ > 0).
    # The three helpers below (plus the predicate _tn_panel_extreme) take the Gaussian bulk
    # out analytically. The panel likelihoods use them only for an observation whose argument
    # is beyond ±40 and evaluate the direct expression for every other observation. The
    # cross-sectional likelihoods use h (_normlogcdf_halfsq) for an observation with an
    # argument below -40 (the note after _tn_shift), and _tn_shift also at moderate arguments
    # when μ and σᵤ are both tiny next to σᵥ (the note before _tn_invmills).

    # h(a) = log Φ(a) + a²/2 = log(erfcx(-a/√2)/2): log Φ with its Gaussian bulk removed,
    # O(log|a|) as a → -∞. Below -40 this is the series of _stable_normlogcdf without its
    # -a²/2. Between -40 and 0 it is erfcx, whose value carries no bulk; at a = -40
    # ForwardDiff's h' through the erfcx rule is off by about 3.5e-13 relative and h'' by
    # about 3.5e-13 absolute (5.5e-10 relative). Above 0, log Φ is O(1) and a²/2 is added
    # directly.
function _normlogcdf_halfsq(a)
    if a < -40
        x = inv(a*a)
        return -log(-a) - log(sqrt2π) + log1p(-x*(1 - 3x*(1 - 5x*(1 - 7x*(1 - 9x)))))
    elseif a < 0
        return log(erfcx(-a*invsqrt2)/2)
    else
        return normlogcdf(a) + a*a/2
    end
end

    # δ = a₁ - a₀, with a₀ = μ/σᵤ and a₁ = μ*/σ*, where μ* = (σᵥ²μ - σᵤ²E)/D,
    # σ*² = σᵥ²σᵤ²/D and D = σᵥ² + σᵤ²S; in the panels S = Σh² and E = Σhϵ; in the cross
    # section S = 1 and E = ϵ (the cross-sectional likelihoods need δ only where μ and σᵤ are
    # both tiny). It is formed without subtracting the two ratios:
    #   δ = -(σᵤ/√D)·(μS/(σᵥ + √D) + E/σᵥ),
    # which uses σᵥ/√D - 1 = -σᵤ²S/(√D(σᵥ + √D)). As σᵤ → 0 both ratios run off while δ → 0,
    # so a₁ - a₀ taken as a difference would carry an error of eps·|a₀|.
function _tn_shift(μ, σᵤ², σᵥ², S, E)
    σᵥ = sqrt(σᵥ²)
    sD = sqrt(σᵥ² + σᵤ²*S)
    return -(sqrt(σᵤ²)/sD)*(μ*S/(σᵥ + sD) + E/σᵥ)
end

    # The cross-sectional likelihoods, LL_T(Trun) and LL_T(Trun_Scale). With σ² = σᵤ² + σᵥ²,
    # μ* = (σᵥ²μ - σᵤ²ϵ)/σ² and σ*² = σᵥ²σᵤ²/σ², exactly
    #   (μ + ϵ)²/σ² = ϵ²/σᵥ² + a₀² - a₁²,
    # so an observation's log-likelihood is
    #   -½log σ² + log φ((μ+ϵ)/σ) + log Φ(a₁) - log Φ(a₀)
    #     = -½log σ² - log√(2π) - ½ϵ²/σᵥ² + h(a₁) - h(a₀),   h = _normlogcdf_halfsq,
    # which has no term of order a² for an argument below -40 (h is O(log|a|) there). The
    # likelihoods use this form when min(a₁, a₀) < -40 and a₁ <= 0. When a₁ > 0 (then
    # a₀ < -40, so μ < 0 and ϵ < 0: an observation below the frontier), h(a₁) holds +½a₁²,
    # which cancels against -½ϵ²/σᵥ² as σᵥ → 0 while the observation's log-likelihood
    # stays O(1). The two are therefore combined exactly: with w = μ + σᵤϵ/(σ + σᵤ) and
    # d = σᵥw/(σσᵤ) = a₁ + ϵ/σᵥ (σ - σᵤ = σᵥ²/(σ + σᵤ)),
    #   ½a₁² - ½ϵ²/σᵥ² = d·a₁ - ½d² = wμ*/σᵤ² - ½d²,
    # two terms that are both <= 0 (w < 0 < μ*), so nothing cancels, and σᵥ enters only
    # through d, so σᵥ² = 0 gives the finite limit. (Using the h form here as well would give
    # 0.22 nats of error at ln σᵥ² = -30 and a log-likelihood of +1.25e21 at -80 on 200
    # observations below the frontier.)
    # A branch that fires only when both arguments are below -40 and keeps log φ((μ+ϵ)/σ)
    # beside h(a₁) - h(a₀) - δ(a₀ + δ/2), δ = a₁ - a₀, is not enough:
    # δ·a₀ = -μ²/(σ(σᵥ+σ)) - μϵ/(σσᵥ) is O((μ/σ)²) and cancels against log φ((μ+ϵ)/σ) as
    # |μ/σ| grows (45 nats on 200 observations at μ = -1e8, σᵤ² = σᵥ² = 1); and with only
    # a₀ below -40, as near the exponential limit μ = -K, σᵤ² = K, such a branch
    # differences the two log Φ whole (284 nats at K = 1e16).

    # ½(a₁² - a₀²) + log Φ(a₁) - log Φ(a₀) (panel likelihoods), when an argument is below
    # -40 or both are above +40. With one argument below -40 it equals h(a₁) - h(a₀). With
    # both above +40, log Φ is about 0 and the two quadratics are what cancel, so their
    # difference is taken as δ(a₀ + δ/2).
function _tn_panel_core(a₁, a₀, δ)
    if a₁ < -40 || a₀ < -40
        return _normlogcdf_halfsq(a₁) - _normlogcdf_halfsq(a₀)
    else
        return δ*(a₀ + δ/2) + _stable_normlogcdf(a₁) - _stable_normlogcdf(a₀)
    end
end

    # when a panel likelihood needs _tn_panel_core: an argument below -40, or both above +40.
    # The cross-sectional likelihoods switch when either argument is below -40; with both
    # above +40 they need nothing, having no explicit ½(a₁² - a₀²).
_tn_panel_extreme(a₁, a₀) = min(a₁, a₀) < -40 || min(a₁, a₀) > 40

#* ------ the panel_TRE orthant probability, in log space ---------------------------

# Both true-random-effect likelihoods and both of their jlms/bc indices need
#
#     P = ∫ φ(u) ∏ₜ Φ((Bₜ - ρu)/γ) du,
#
# a T-dimensional orthant probability written as one-dimensional quadrature over the common
# random factor.  Formed as a probability and then taken through log(max(P, 1e-15)) it is
# right only while P stays above the floor.  Φ underflows to exactly 0 below about -38 and P
# reaches 1e-15 much sooner, after which the floor supplies log(1e-15) = -34.539 whatever the
# truth is.  Against a 256-bit reference at T = 3 that is 49.5 nats out at B/γ = -8, 449 at
# -20 and 1.18e10 at -1e5, and the threshold moves with the panel length, so a long panel, or
# a firm on the wrong side of the frontier, reaches it.  Worse, on the flat side of the floor
# d log P/dB is exactly 0, so the optimiser is told the objective does not depend on the
# parameters at all: an N = 4, T = 10 truncated-normal fit reported ll = 11167 where a
# direct 256-bit integral over the random effect says -133.5, every firm's contribution
# having been replaced by the floor.
#
# So take the logarithm inside.  With
#
#     h(u) = log φ(u) + ∑ₜ log Φ(aₜ(u)),   aₜ(u) = (Bₜ - ρu)/γ,
#
# h is strictly concave (log Φ is concave and aₜ is affine in u), so it has one maximum, at
# u* ≤ 0 because h'(0) ≤ 0.  Integrating each side of the peak in its own units,
#
#   log P = h(u*) + log( sₗ∫₀^∞ e^{h(u* - sₗt) - h(u*)} dt + sᵣ∫₀^∞ e^{h(u* + sᵣt) - h(u*)} dt ),
#
# an identity for ANY constants u*, sₗ, sᵣ > 0, which is why all three are taken from the
# VALUE part under ForwardDiff and held fixed: they cancel out of
# d(log P)/dθ = ∫e^{h-h(u*)}h_θ / ∫e^{h-h(u*)}.  Checked against a central difference and a
# nested-Dual Hessian in every regime, to 1e-10.
#
# Four things about the integral are not optional.
#
# CENTRING.  quadgk over (-Inf, Inf) samples |u| ≤ 59 on its first pass, and at σᵥ² near its
# clamp the peak sits at u* = -8e3, so an uncentred integral returns 0 and log P returns -Inf.
#
# TWO WIDTHS.  The peak is routinely one-sided: at the N = 4, T = 10 drift point h falls 112
# nats one curvature unit to the RIGHT of u* and 0.63 nats forty units to the LEFT, because to
# the left every Φ tends to 1 and only the Gaussian factor is left.  Scaling both sides by the
# curvature at the maximum put log P 0.61 nats out there; each side carries its own one-nat
# width instead.
#
# DIFFERENCING h, NOT SUBTRACTING TWO EVALUATIONS OF IT.  h itself reaches -3e11 on parameters
# an optimiser visits, so h(u) - h(u*) formed as a difference of two evaluations is quantised
# at eps·3e11 = 6e-5: the integrand becomes a staircase, quadgk cannot reach its tolerance on
# a discontinuous integrand, and it spends its whole 1e7-evaluation budget -- 1.2 s for one
# firm, where an ordinary firm takes 20 µs -- to return the answer it already had.  The
# difference is therefore taken term by term in a form where nothing cancels; see
# _tre_dlogcdf.
#
# A FINITE RANGE, for the reason given at _tre_logorthant below: quadgk's own (0,∞) change
# of variable multiplies an underflowed integrand by an overflowing Jacobian and returns
# NaN.

    # value part of a possibly nested ForwardDiff.Dual, as Float64
_mlevalue(x::Real) = Float64(x)
_mlevalue(x::ForwardDiff.Dual) = _mlevalue(ForwardDiff.value(x))

    # how many of its own widths of each side of the peak the quadrature covers
const _TRE_SPAN = 60.0

    # φ(a)/Φ(a), the inverse Mills ratio.  Φ(a) = erfcx(-a/√2)e^{-a²/2}/2 for every a, so the
    # ratio is √(2/π)/erfcx(-a/√2) with the huge exponential cancelled ANALYTICALLY.  Written
    # instead as exp(normlogpdf(a) - normlogcdf(a)) it differences two logarithms that both
    # go as -a²/2: at a = -1.2e6 those are -7.2e11 apiece and their difference carries 1.2e-4
    # of noise, which puts ±88 of noise on h' and leaves the peak search up to 80 away from
    # the maximum -- far enough that h rises by 144 nats to one side of the `peak'.  Above
    # zero erfcx(-a/√2) would overflow, and there Φ(a) ≥ 1/2 so nothing cancels anyway.
_tre_invmills(a) = a <= 0 ? 2/(sqrt2π*erfcx(-a*invsqrt2)) : exp(normlogpdf(a) - normlogcdf(a))

    # A Hessian differentiates the line above on inner Duals (the Dual rule of _tre_dlogcdf
    # below calls it on av + dv), through DiffRules' erfcx rule, whose two terms cancel: dλ/da
    # then carries a relative error of about eps·a², 1.5e-10 at a = -1e3, 4.6e-5 at -1e6 and
    # 100% at -1e8, and truncated-normal TRE Hessians at μ/σᵤ of -3e6 were 5% off. Below
    # a = -1e3 this method takes λ(a) = m(a) - a with m = _tre_millsm (its series), whose
    # derivative has nothing to cancel; above, the line above, bit for bit. Values and
    # gradients never come here (they call _tre_invmills on Float64). This method moves a
    # Hessian only where an argument below -1e3 carries weight in the integral: at a
    # truncated-normal firm with μ/σᵤ below -1e3, or at a firm with an observation far on the
    # wrong side of the frontier and a small σₐ², also when σᵤ/σᵥ ≤ 100 (in checks by 1.6e-17
    # to 6.4e-9 of the Hessian's largest entry, the truncated-normal ones towards the exact
    # value). At 1150 ordinary points of the TRE and CSW likelihoods it changed no Hessian bit.
_tre_invmills(a::ForwardDiff.Dual) = a < -1e3 ? _tre_millsm(a) - a : invoke(_tre_invmills, Tuple{Any}, a)

    # a + φ(a)/Φ(a) = (φ(a) + aΦ(a))/Φ(a) > 0, the factor by which the JLMS numerator's
    # integrand differs from the orthant integrand.  Below a = -35 the two terms cancel to
    # about eps·|a|³ and the asymptotic series in t = -a is the accurate branch; the two agree
    # to about 5e-12 where they meet.
function _tre_millsm(a)
    if a > -35
        return a + _tre_invmills(a)
    else
        t = -a
        x = 1/(t*t)
        return (1 + x*(-2 + x*(10 + x*(-74 + x*706))))/t
    end
end

    # log Φ(a₀ + δ) - log Φ(a₀), without ever forming the difference of two large logarithms.
    # For a ≤ 0, log Φ(a) = log(erfcx(-a/√2)/2) - a²/2 exactly (erfcx(x) = e^{x²}erfc(x), which
    # neither underflows nor loses the tail), and in that form the two pieces difference
    # cleanly: log erfcx is O(log|a|), and the quadratic part is δ(a₀ + δ/2), taken from δ
    # itself rather than from a - a₀.  At a = -4.7e5 that is the difference between a smooth
    # integrand and one quantised at 1.5e-5.  Where either argument is positive both
    # logarithms are O(1) and there is nothing to cancel, so the direct difference is used.
function _tre_dlogcdf(a₀, δ)
    a = a₀ + δ
    if a <= 0 && a₀ <= 0
        return log(erfcx(-a*invsqrt2)) - log(erfcx(-a₀*invsqrt2)) - δ*(a₀ + δ/2)
    else
        return normlogcdf(a) - normlogcdf(a₀)
    end
end

    # The same cancellation one level down, in the DERIVATIVE.  ∂/∂a₀ of the difference
    # above is λ(a₀+δ) - λ(a₀) with λ the inverse Mills ratio, and ForwardDiff's generic
    # rule forms the two λ's apart: at a = -1.2e7 they are 1.2e7 apiece, their O(δ)
    # difference is quantised at eps·λ·|ȧ₀|, and the gradient of LL_T(PTRET) came out 0.42
    # wrong (per central difference) on the ln σᵤ² coordinate at the σᵤ² clamp -- with the
    # Hessian, a difference of such gradients, unusable.  λ(a) - λ(a₀) = m(a) - m(a₀) - δ
    # EXACTLY with m(x) = x + λ(x) = _tre_millsm(x), and both m's are O(1/|a|) there:
    # nothing cancels.  Only the same-tag Dual pair needs the rule -- every argument in
    # the likelihood descends from the one ρ vector, so tags cannot mix; anything else
    # falls back to the generic method above, which is correct, only noisier.  The value
    # part is the generic method's own, so plain evaluation is unchanged, and nested Duals
    # (the Hessian) recurse through this same rule one tag at a time.
function _tre_dlogcdf(a₀::ForwardDiff.Dual{Tg}, δ::ForwardDiff.Dual{Tg}) where {Tg}
    av = ForwardDiff.value(a₀)
    dv = ForwardDiff.value(δ)
    λa = _tre_invmills(av + dv)
    dλ = _tre_millsm(av + dv) - _tre_millsm(av) - dv
    return ForwardDiff.Dual{Tg}(_tre_dlogcdf(av, dv),
                                dλ*ForwardDiff.partials(a₀) + λa*ForwardDiff.partials(δ))
end

    # h(u* + du) - h(u*), differenced term by term.  du, u* and the widths are Float64
    # constants, so the Gaussian part -du(u* + du/2) is exact; δ = -(ρ/γ)du is the shift every
    # standardised argument takes.
function _tre_dh(a₀, r, u0, du)
    δ = -r*du
    s = -du*(u0 + du/2)
    @inbounds for k in eachindex(a₀)
        s += _tre_dlogcdf(a₀[k], δ)
    end
    return s
end

    # u* and the two one-nat half-widths, all from the value part.
    # h'(u) = -u - r ∑ λ(aₜ) and -h''(u) = 1 + r² ∑ λ(aₜ)m(aₜ) ≥ 1 with r = ρ/γ, so h' is
    # strictly decreasing: bracket by doubling from 0, then take safeguarded Newton steps
    # inside the bracket.  h(u*) - h(u* ± d) is monotone in d on each side, so the geometric
    # search for the width cannot land in the wrong place either -- but it must be TIGHTENED
    # rather than stopped at the first bracket, because on a steep side one doubling can take
    # the drop from below 1 nat to 459 of them, and a width 20 times too large leaves the
    # whole peak inside the first quadrature node.  Bisecting in log d to within 10% costs
    # about five more evaluations of h.  Neither u* nor the widths has to be accurate beyond
    # that -- the identity holds for any constants -- only good enough to centre and scale.
function _tre_orthant_peak(B, ρ, γ)
    ρv = _mlevalue(ρ)
    γv = _mlevalue(γ)
    r  = ρv/γv
    bv = Vector{Float64}(undef, length(B))
    @inbounds for k in eachindex(B)
        bv[k] = _mlevalue(B[k])
    end
    dh = function (u)
        s = 0.0
        @inbounds for k in eachindex(bv)
            s += _tre_invmills((bv[k] - ρv*u)/γv)
        end
        return -u - r*s
    end
    nd2h = function (u)
        s = 0.0
        @inbounds for k in eachindex(bv)
            a = (bv[k] - ρv*u)/γv
            s += _tre_invmills(a)*_tre_millsm(a)
        end
        return 1.0 + r*r*s
    end
    hi, lo = 0.0, 0.0
    if dh(0.0) < 0
        lo = -1.0
        for _ in 1:200
            dh(lo) > 0 && break
            lo *= 2
        end
    end
    u0 = 0.0
    if lo < hi
        u0 = (lo + hi)/2
        for _ in 1:100
            d = dh(u0)
            d > 0 ? (lo = u0) : (hi = u0)
            (abs(d) < 1e-10*(1 + abs(u0)) || (hi - lo) < 1e-13*(1 + abs(u0))) && break
            un = u0 + d/nd2h(u0)
            u0 = (lo < un < hi) ? un : (lo + hi)/2
        end
    end
    a0v = (bv .- ρv*u0)./γv
    d0  = 1/sqrt(nd2h(u0))
    width = function (sgn)
        drop = d -> -_tre_dh(a0v, r, u0, sgn*d)   # differenced, so a drop of 1 is resolvable
        wlo, whi = 0.0, d0                        # even where h itself is -3e12
        if drop(d0) < 1
            wlo = d0
            for _ in 1:400
                whi = 2*wlo
                drop(whi) >= 1 && break
                wlo = whi
            end
        else
            for _ in 1:400
                wlo = whi/2
                (wlo <= 0 || drop(wlo) < 1) && break
                whi = wlo
            end
        end
        for _ in 1:100                    # tighten in log d until whi is within 10% of wlo
            (wlo <= 0 || !(whi > wlo) || whi <= 1.1*wlo) && break
            mid = sqrt(wlo)*sqrt(whi)
            (mid <= wlo || mid >= whi) && break
            drop(mid) < 1 ? (wlo = mid) : (whi = mid)
        end
        return (isfinite(whi) && whi > 1e-300) ? whi : 1.0
    end
    return u0, width(-1.0), width(1.0)
end

    # Firms whose integrand turns much more sharply than the widths above. Where σᵥ is very small
    # next to σᵤ, each factor Φ(aₜ(u)) turns from 1 to its Gaussian tail within a few units of its
    # argument, that is within a few 1/r in u (r = ρ/γ), while a side's one-nat width is up to
    # about 1.5. One adaptive rule per side over [0, 60] widths (the layout of the helpers below)
    # can then step over the turn, which lies either at the peak, making it one-sided (narrow/wide
    # width q = 0.01 and less), or about one width from a peak where every Φ is 1 (q up to 1, so
    # the switch on q in _csw_logorthant misses it). At σᵤ/σᵥ = 1000 to 1e4 on simulated
    # T = 5 panels the firms' Hessians came out wrong by up to twice their largest entry, in
    # LL_T(PTREH) and LL_T(PTRET) alike, gradients by up to 1.6e-3, values by up to 6e-7.
    #
    # z = r·max(sL, sR) measures that sharpness. Where z > _TRE_SHARP, _tre_logorthant below
    # (LL_T(PTREH)) integrates piece by piece over [-60sL, 60sR], with breakpoints at the peak,
    # at the ends and at (aₜ(u*) + j)/r, j = 0, ±1, ±3, ±10, ±30, around every turn whose centre
    # lies within the range and where h has dropped less than 40 nats, so that every turn is
    # resolved inside pieces of its own scale; each quadgk call stops after 1e5 evaluations, as
    # in _csw_logorthant. Since max(sL, sR) ≤ 1.1√2 and r < (σᵤ/σᵥ)/√T, z stays below 156 for
    # every firm with σᵤ/σᵥ ≤ 100, at any T, σₐ² and μ: below the switch the one-rule-per-side
    # layout runs. In a survey of 4926 firms that layout's Hessian below z = 160 was within
    # 2.1e-6 of its largest entry, apart from two short-panel firms (5.2e-5 at z = 149, 3.1e-5
    # at 159), which stay on it; from z = 190 it was off by 3.8 to 85 times that entry.
    #
    # In the new branch the differences of h are taken by _tre_dh_x, whose Dual rule forms
    # λ(a) - λ(a₀) directly where either argument is positive: the rule of _tre_dlogcdf,
    # m(a) - m(a₀) - δ, cancels there to eps·a, which d a₀/dθ, of order r, multiplies up to
    # gradient errors of 7e-7 at σᵤ/σᵥ = 1e5 and 6e-5 at 1e6. _tre_dh and _tre_dlogcdf, which
    # _csw_logorthant also uses, keep the m form everywhere.
const _TRE_SHARP = 160.0

    # B = (-K*G)' is formed in the likelihoods and in jlmsbc from K = ε'Σ⁻¹ - μ'Σᵤ⁻¹,
    # whose entries are of order 1/σᵥ², and G's ρ²11' part needs their sum, which is of order
    # one: the cancellation shifts every aₜ by about eps·r²T·σᵤ/σᵥ (up to 2.4e-9 at
    # σᵤ/σᵥ = 100, 2e-5 at 3162 and 265 at 1e6). In LL_T(PTRET) the constant in front comes
    # from the exact D while the integrand comes from B, so the two parts no longer cancel in
    # the derivatives: its Hessian was off by up to 1.6e-6 at σᵤ/σᵥ = 100, 5.2e-5 at 150, 2e-3
    # at 316 and 2.6 times its largest entry at 3162, and at 1e6 its value by 3e4 nats
    # (LL_T(PTREH), which forms all of h from the same B, is hurt far less: 1.4e-7 at 3162,
    # 1e-3 at 1e6). _tre_logorthant_ratio therefore takes the new branch, with every argument
    # formed from D and m, wherever r√T > _TRE_CANCEL (_tre_past_switch); r√T < σᵤ/σᵥ, so no
    # firm with σᵤ/σᵥ ≤ 100 takes it, and z > 160 implies r√T > 100. LL_T(PTREH) passes the B
    # of _tre_Bx below to _tre_logorthant, which uses it only past the layout switch. jlmsbc
    # (SFindex.jl) takes the same switch per firm: a firm with r√T > _TRE_CANCEL gets jlms
    # and bc from the new branches below, from that B; on the one-rule-per-side layout the jlms
    # numerator (the integrand times m(a_j)) can be mis-integrated well below z = 160 (a T = 20
    # firm's jlms was 1.2e-5 off at σᵤ/σᵥ = 3000 with z = 92).
const _TRE_CANCEL = 100.0

_tre_sharp(ρ, γ, sL, sR) = (_mlevalue(ρ)/_mlevalue(γ))*max(sL, sR) > _TRE_SHARP
_tre_past_switch(ρ, γ, T) = (_mlevalue(ρ)/_mlevalue(γ))*sqrt(T) > _TRE_CANCEL

    # B without that cancellation (Sherman-Morrison, as D in LL_T(PTRET)):
    #   B = -(γ²/σᵥ²)(ε - (Tσₐ²/(σᵥ²+Tσₐ²))ε̄) - ρ²Tε̄/(σᵥ²+Tσₐ²) + (γ²/σᵤ²)μ + (ρ²/σᵤ²)∑μ,
    # every term of order ε or μ. LL_T(PTREH) and jlmsbc form it and pass it on.
function _tre_Bx(ε, μ, σₐ², σᵤ², σᵥ², γ², ρ², T)
    εbar = sum(ε)/T
    return (-(γ²/σᵥ²)) .* (ε .- (T*σₐ²/(σᵥ² + T*σₐ²))*εbar) .- ρ²*T*εbar/(σᵥ² + T*σₐ²) .+
           (γ²/σᵤ²) .* μ .+ (ρ²/σᵤ²)*sum(μ)
end

    # _tre_dlogcdf and _tre_dh for the new branch: the same values; in the Dual rule
    # λ(a) - λ(a₀) is formed directly where either argument is positive.
_tre_dlogcdf_x(a₀, δ) = _tre_dlogcdf(a₀, δ)
function _tre_dlogcdf_x(a₀::ForwardDiff.Dual{Tg}, δ::ForwardDiff.Dual{Tg}) where {Tg}
    av = ForwardDiff.value(a₀)
    dv = ForwardDiff.value(δ)
    λa = _tre_invmills(av + dv)
    dλ = (av + dv > 0 || av > 0) ? λa - _tre_invmills(av) : _tre_millsm(av + dv) - _tre_millsm(av) - dv
    return ForwardDiff.Dual{Tg}(_tre_dlogcdf_x(av, dv),
                                dλ*ForwardDiff.partials(a₀) + λa*ForwardDiff.partials(δ))
end
function _tre_dh_x(a₀, r, u0, du)
    δ = -r*du
    s = -du*(u0 + du/2)
    @inbounds for k in eachindex(a₀)
        s += _tre_dlogcdf_x(a₀[k], δ)
    end
    return s
end

    # the breakpoints, in du = u - u*, from the value parts
function _tre_cliffs(a₀, r, u0, sL, sR)
    av = [_mlevalue(a) for a in a₀]
    rv = _mlevalue(r)
    lo, hi = -_TRE_SPAN*sL, _TRE_SPAN*sR
    pts = [lo, 0.0, hi]
    for a in av
        c = a/rv
        (lo < c < hi && -_tre_dh(av, rv, u0, c) < 40) || continue
        for j in (0.0, 1.0, -1.0, 3.0, -3.0, 10.0, -10.0, 30.0, -30.0)
            p = (a + j)/rv
            lo < p < hi && push!(pts, p)
        end
    end
    return sort!(unique!(pts))
end

    # ∫ f(du) over the pieces
function _tre_cliffint(f, pts)
    I = quadgk(f, pts[1], pts[2], rtol = 1e-10, maxevals = 10^5)[1]
    for k in 2:length(pts)-1
        I += quadgk(f, pts[k], pts[k+1], rtol = 1e-10, maxevals = 10^5)[1]
    end
    return I
end

    # the new branches: of _tre_logorthant and _tre_logorthant_ratio below, which switch to
    # them themselves, and the jlms and bc of a firm past _tre_past_switch, which jlmsbc calls
    # directly. _tre_orthant_x is a firm's integral ∫ exp(h(u* + du) - h(u*)) du with its peak,
    # arguments and breakpoints; jlmsbc computes it once per firm and passes it to
    # _tre_orthant_jlms_x and to every _tre_logorthant_diff_x (the same numbers as computing it
    # in each call).
function _tre_logorthant_x(B, ρ, γ)
    u0, sL, sR = _tre_orthant_peak(B, ρ, γ)
    r   = ρ/γ
    a₀  = (B .- ρ*u0)./γ
    hu0 = normlogpdf(u0) + sum(_stable_normlogcdf, a₀)
    isfinite(_mlevalue(hu0)) || return hu0
    return hu0 + log(_tre_cliffint(du -> exp(_tre_dh_x(a₀, r, u0, du)), _tre_cliffs(a₀, r, u0, sL, sR)))
end

function _tre_logorthant_ratio_x(ρ, γ, m, D)
    u0, sL, sR = _tre_orthant_peak(D .+ γ .* m, ρ, γ)
    r   = ρ/γ
    a₀  = m .+ (D .- ρ*u0)./γ             # the arguments of hr0 below, term for term
    hr0 = normlogpdf(u0) + sum(_tre_dlogcdf_x(m[k], (D[k] - ρ*u0)/γ) for k in eachindex(m))
    isfinite(_mlevalue(hr0)) || return hr0
    return hr0 + log(_tre_cliffint(du -> exp(_tre_dh_x(a₀, r, u0, du)), _tre_cliffs(a₀, r, u0, sL, sR)))
end

function _tre_orthant_x(B, ρ, γ)
    u0, sL, sR = _tre_orthant_peak(B, ρ, γ)
    r   = ρ/γ
    a₀  = (B .- ρ*u0)./γ
    pts = _tre_cliffs(a₀, r, u0, sL, sR)
    return (u0 = u0, a₀ = a₀, pts = pts, I = _tre_cliffint(du -> exp(_tre_dh_x(a₀, r, u0, du)), pts))
end

function _tre_logorthant_diff_x(B, ρ, γ, S, P1 = _tre_orthant_x(B, ρ, γ))
    u01 = P1.u0
    a₀1 = P1.a₀
    B2  = B .+ S
    u02, sL2, sR2 = _tre_orthant_peak(B2, ρ, γ)
    r   = ρ/γ
    a₀2 = (B2 .- ρ*u02)./γ
    hd  = -(u02 - u01)*(u02 + u01)/2 +
          sum(_tre_dlogcdf_x(a₀1[k], (S[k] - ρ*(u02 - u01))/γ) for k in eachindex(S))
    isfinite(_mlevalue(hd)) || return hd
    I2 = _tre_cliffint(du -> exp(_tre_dh_x(a₀2, r, u02, du)), _tre_cliffs(a₀2, r, u02, sL2, sR2))
    return hd + log(I2/P1.I)
end

    # The T numerators share the factor exp(h(u* + du) - h(u*)), which costs T terms at every
    # node and is computed once per node (cache); every number is the same as without it.
function _tre_orthant_jlms_x(B, ρ, γ, P = _tre_orthant_x(B, ρ, γ))
    r     = ρ/γ
    a₀    = P.a₀
    u0    = P.u0
    cache = Dict{typeof(u0), typeof(P.I)}()
    eh(du) = get!(() -> exp(_tre_dh_x(a₀, r, u0, du)), cache, du)
    return [γ*_tre_cliffint(du -> eh(du)*_tre_millsm(a₀[j] - r*du), P.pts)/P.I for j in eachindex(B)]
end

    # Each side is integrated over sixty of its own widths, not to infinity.  h is concave
    # and has fallen one nat at t = 1, so it has fallen at least t nats at every t ≥ 1 and
    # what lies past t = 60 is bounded by e^-60 = 9e-27 of the peak, far below the tolerance.
    # A semi-infinite quadgk range is not merely unnecessary but harmful: its (0,∞) change of
    # variable multiplies the integrand by a Jacobian that overflows near the far end,
    # and once the integrand has underflowed to 0 there the product is 0·Inf = NaN, which
    # quadgk reports as a DomainError.  That is a NaN objective.
    #
    # rtol is 1e-10: the integrand is normalised to 1 at its peak, so the tolerance translates
    # straight into an absolute error in log P.
function _tre_logorthant(B, ρ, γ, Bx = B)
    u0, sL, sR = _tre_orthant_peak(B, ρ, γ)
    _tre_sharp(ρ, γ, sL, sR) && return _tre_logorthant_x(Bx, ρ, γ)
    r   = ρ/γ
    a₀  = (B .- ρ*u0)./γ
    hu0 = normlogpdf(u0) + sum(_stable_normlogcdf, a₀)
    # log P below about -9e307 is not a Float64; -Inf is then the honest answer, and it is
    # what the firm's contribution means -- these parameters make its data impossible.
    isfinite(_mlevalue(hu0)) || return hu0
    I = sL*quadgk(t -> exp(_tre_dh(a₀, r, u0, -sL*t)), 0, _TRE_SPAN, rtol = 1e-10)[1] +
        sR*quadgk(t -> exp(_tre_dh(a₀, r, u0,  sR*t)), 0, _TRE_SPAN, rtol = 1e-10)[1]
    return hu0 + log(I)
end

    # log ∫ φ(u) ∏ₜ [Φ(aₜ(u))/Φ(mₜ)] du -- the orthant probability divided through by the
    # truncated normal's normalising constant, the two combined BEFORE either logarithm is
    # formed.  LL_T(PTRET) needs log P - ∑ₜ log Φ(μₜ/σᵤ); as σᵤ → 0 with μ < 0, B → μ and
    # γ → σᵤ elementwise, so every aₜ → μₜ/σᵤ and the two terms cancel EXACTLY in the
    # limit.  At the σᵤ² clamp an N = 4, T = 10 truncated-normal fit walks to, each is
    # about -5.9e14, and their O(1) difference as two Float64s is quantised at 0.06: the
    # likelihood sat 2.0 nats above a direct 256-bit integral over the random effect
    # (which at ordinary parameters it matches to 5e-15), and bc, whose exponent carries
    # the same bulk twice, reported 1.101 for a quantity that cannot exceed 1.
    #
    # The combined log-integrand is h(u) - ∑ₜlog Φ(mₜ): the same h shifted by a constant,
    # so u* and the one-nat widths are UNCHANGED and the normalised ∫e^{h-h(u*)} is the
    # same integral _tre_logorthant computes.  Only the constant in front changes, from
    # h(u*) to log φ(u*) + ∑ₜ[log Φ(aₜ(u*)) - log Φ(mₜ)], each bracket taken through
    # _tre_dlogcdf so the two large logarithms are never formed.  The shift
    # δₜ = aₜ(u*) - mₜ cannot be taken as a difference either: aₜ(u*) and mₜ are both
    # O(μ/σᵤ), and one ulp of -1.2e7 is 1.9e-9, which log Φ's slope multiplies back to
    # 0.02.  The caller supplies D = B - γm formed analytically -- for LL_T(PTRET), with
    # g = γ/σᵤ, q = σᵤ²/σᵥ², ε̄ = ∑ε/T,
    #
    #   D = (g²-g)μ + (ρ²/σᵤ²)(∑μ)1 - (γ²/σᵥ²)(ε - (Tσₐ²/(σᵥ²+Tσₐ²))ε̄1) - (ρ²Tε̄/(σᵥ²+Tσₐ²))1,
    #   g²-g = -q/((1+q)(1+√(1+q))),
    #
    # exact by Sherman-Morrison on B = G(Σᵤ⁻¹μ - Σ⁻¹ε), with every term of order q or
    # smaller: nothing of order μ survives, so nothing cancels -- and δₜ = (Dₜ - ρu*)/γ.
function _tre_logorthant_ratio(B, ρ, γ, m, D)
    _tre_past_switch(ρ, γ, length(m)) && return _tre_logorthant_ratio_x(ρ, γ, m, D)
    u0, sL, sR = _tre_orthant_peak(B, ρ, γ)
    r   = ρ/γ
    a₀  = (B .- ρ*u0)./γ
    hr0 = normlogpdf(u0) + sum(_tre_dlogcdf(m[k], (D[k] - ρ*u0)/γ) for k in eachindex(m))
    isfinite(_mlevalue(hr0)) || return hr0
    I = sL*quadgk(t -> exp(_tre_dh(a₀, r, u0, -sL*t)), 0, _TRE_SPAN, rtol = 1e-10)[1] +
        sR*quadgk(t -> exp(_tre_dh(a₀, r, u0,  sR*t)), 0, _TRE_SPAN, rtol = 1e-10)[1]
    return hr0 + log(I)
end

    # log P(B + S) - log P(B) at the same ρ, γ, for bc's ratio of orthant probabilities:
    # with B₂ = B - Ge_j the shift S = -Ge_j is exact and a₂ₜ - a₁ₜ = Sₜ/γ exactly, while
    # each log P alone carries the ∑ₜlog Φ(μₜ/σᵤ)-sized bulk that cancels in the
    # difference (see _tre_logorthant_ratio above; at the σᵤ² clamp forming the two logs
    # apart quantises the difference at 0.06 and bc came out 1.101).  Each side keeps its
    # OWN peak and widths -- the two normalised integrals are O(1) and divide cleanly --
    # and the peak values are differenced term by term:
    #   Δₜ = a₂ₜ(u₂*) - a₁ₜ(u₁*) = (Sₜ - ρ(u₂* - u₁*))/γ,
    #   log φ(u₂*) - log φ(u₁*) = -(u₂* - u₁*)(u₂* + u₁*)/2,
    # both from S and the Float64 peak constants, so nothing large is ever formed.
function _tre_logorthant_diff(B, ρ, γ, S)
    u01, sL1, sR1 = _tre_orthant_peak(B, ρ, γ)
    B2 = B .+ S
    u02, sL2, sR2 = _tre_orthant_peak(B2, ρ, γ)
    r   = ρ/γ
    a₀1 = (B  .- ρ*u01)./γ
    a₀2 = (B2 .- ρ*u02)./γ
    hd  = -(u02 - u01)*(u02 + u01)/2 +
          sum(_tre_dlogcdf(a₀1[k], (S[k] - ρ*(u02 - u01))/γ) for k in eachindex(S))
    isfinite(_mlevalue(hd)) || return hd
    I1 = sL1*quadgk(t -> exp(_tre_dh(a₀1, r, u01, -sL1*t)), 0, _TRE_SPAN, rtol = 1e-10)[1] +
         sR1*quadgk(t -> exp(_tre_dh(a₀1, r, u01,  sR1*t)), 0, _TRE_SPAN, rtol = 1e-10)[1]
    I2 = sL2*quadgk(t -> exp(_tre_dh(a₀2, r, u02, -sL2*t)), 0, _TRE_SPAN, rtol = 1e-10)[1] +
         sR2*quadgk(t -> exp(_tre_dh(a₀2, r, u02,  sR2*t)), 0, _TRE_SPAN, rtol = 1e-10)[1]
    return hd + log(I2/I1)
end

    # E[u_j | ε] for j = 1..T, the JLMS index.  The numerator integrand jlmsbc used,
    # φ(u)γ(a_jΦ(a_j) + φ(a_j))∏_{k≠j}Φ(a_k), is φ(u)γ·m(a_j)·∏ₖΦ(aₖ), so numerator and
    # denominator share the same integrand and the ratio needs neither a logarithm nor a floor.
function _tre_orthant_jlms(B, ρ, γ)
    u0, sL, sR = _tre_orthant_peak(B, ρ, γ)
    r  = ρ/γ
    a₀ = (B .- ρ*u0)./γ
    two = function (f)
        sL*quadgk(t -> f(-sL*t), 0, _TRE_SPAN, rtol = 1e-10)[1] +
        sR*quadgk(t -> f( sR*t), 0, _TRE_SPAN, rtol = 1e-10)[1]
    end
    D = two(du -> exp(_tre_dh(a₀, r, u0, du)))
    return [γ*two(du -> exp(_tre_dh(a₀, r, u0, du))*_tre_millsm(a₀[j] - r*du))/D
            for j in eachindex(B)]
end



 #* ----  Normal Truncated-Normal -------------

    # When μ and σᵤ are both tiny next to σᵥ, a₀ = μ/σᵤ
    # can stay moderate while a₁ = μ*/σ* = a₀ + δ with δ ≈ -(σᵤ/σᵥ)(ϵ/σᵥ) → 0, and the two log Φ
    # of LL_T(Trun) and LL_T(Trun_Scale) nearly coincide. Their difference is accurate as a value,
    # but ForwardDiff forms the μ-derivatives of the two terms separately: each second-derivative
    # piece L''(a)(∂a/∂μ)² is about |L''|/σᵤ² (h''/σᵤ² ≈ 1/μ² in the tail form), while their
    # difference is O(1/σᵤ). One observation's H[μ,μ] therefore carries an error of about
    # eps·max(1, a₀²)·|L''(a₀)|·σᵥ²/σᵤ² in units of 1/σᵥ² (the a₀² is the ForwardDiff error of
    # the second derivative of log Φ, or of h, through the erfcx rule near -40), or eps·σᵥ²/μ²
    # when both arguments are below -40 (the tail form with h from its series at both): relative
    # to max(1, max|H|), 3e-5 at σᵤ/σᵥ = 1e-4, 2e-2 at 1e-6, 20 at 1e-9 and 5e5 at 1e-13 near
    # a₀ = -40 with σᵥ = 1. For such an observation the difference is written as δ·D(a₀, δ),
    # with δ from _tn_shift (formed without subtracting the two ratios) and D the divided
    # difference, D(a, δ) = (f(a + δ) - f(a))/δ = ∫₀¹ f'(a + tδ) dt, taken by 4-point
    # Gauss–Legendre (for |δ| ≤ 0.05 within 2e-15 of D where a ≤ 5, and within 7e-21 in δ·D up to
    # a = 40); f = log Φ (f' = φ/Φ, _tn_invmills) when a₀ ≥ -40, and f = h (f' = sf_tn_mean) in
    # the tail form. Nothing large is then differenced: the μ-derivatives are products such as
    # δ·D_a/σᵤ = O(1) and δ·D_aa/σᵤ² = O(1/σᵤ).

    # φ(t)/Φ(t), the derivative of log Φ, without cancellation: below -5 as m(t) + |t| with the
    # continued fraction of sf_tn_mean, on [-5, 0] from erfcx, above 0 as φ/Φ.
    # ForwardDiff's second derivative is within about 5e-12 relative on [-5, 0] (the erfcx rule),
    # and within 1e-15 below -5.
@inline function _tn_invmills(t)
    if t < -5
        return sf_tn_mean(t) - t
    elseif t <= 0
        return 0.7978845608028654 / erfcx(-t * invsqrt2)
    else
        e = exp(-t * t / 2)
        return 0.3989422804014327 * e / (1 - erfcx(t * invsqrt2) * e / 2)
    end
end

    # f(a + δ) - f(a) = δ·∫₀¹ fp(a + tδ) dt, fp = f', by 4-point Gauss–Legendre (|δ| ≤ 0.05)
@inline function _tn_step(fp, a, δ)
    c = a + δ / 2
    g1 = 0.16999052179242813 * δ
    g2 = 0.4305681557970263 * δ
    return δ * (0.32607257743127305 * (fp(c - g1) + fp(c + g1)) +
                0.17392742256872692 * (fp(c - g2) + fp(c + g2)))
end

    # The switch for μ and σᵤ both tiny next to σᵥ: true where the observation's H[μ,μ] error
    # above is estimated to exceed 1e-6/σᵥ² (eps/1e-6 = 2.2e-10), with σᵤ|ϵ| < 0.04σᵥ² so that
    # |δ| < 0.05. The estimate
    # eps·σᵥ²/μ² is used only when both arguments are below -40 (a₁ as the caller forms it): with
    # only one of them below -40 the tail form takes h at the other through erfcx, and the
    # first estimate applies. Every other observation evaluates the tail form or the direct
    # expression. (σᵤ² < 1e-4σᵥ² excludes no observation the estimate would select.)
@inline function _tn_tiny(μ, σᵤ², σᵥ², ϵ, a₀, a₁)
    (0 < σᵤ² < 1e-4 * σᵥ² && isfinite(σᵥ²) && σᵤ² * ϵ^2 < 1.6e-3 * σᵥ²^2) || return false
    (a₀ < -40 && a₁ < -40) && return μ^2 < 2.2e-10 * σᵥ²
    λ = _tn_invmills(a₀)
    return σᵤ² < 2.2e-10 * max(1, a₀^2) * (λ * (a₀ + λ)) * σᵥ²
end

function LL_T(::Type{Trun}, Y::Matrix, X::Matrix, Z::Matrix, q, W::Matrix, V::Matrix,
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, dum2, ::Nothing )

    β  = rho[po.begx : po.endx]
    δ1 = rho[po.begz : po.endz]
    δ2 = rho[po.begw : po.endw]
    γ  = rho[po.begv : po.endv]

    μ   = Z * δ1
    σᵤ² = exp.(W * δ2)
    σᵤ  = sqrt.(σᵤ²)
    σᵥ² = exp.(V * γ)
    σ²  = σᵤ² + σᵥ²
    ϵ   = PorC*(Y - X * β)
    μₛ  = (σᵥ² .* μ - σᵤ² .* ϵ) ./ σ²
    σₛ² = (σᵥ² .* σᵤ²) ./ σ²

ll = Vector{eltype(ϵ)}(undef, nobs)
@floop begin
@inbounds for i in 1:nobs
       a₁ = (μₛ[i]) / sqrt(σₛ²[i])
       a₀ = (μ[i]) / σᵤ[i]
       if _tn_tiny(μ[i], σᵤ²[i], σᵥ²[i], ϵ[i], a₀, a₁)    # μ and σᵤ both tiny next to σᵥ
           δᵢ = _tn_shift(μ[i], σᵤ²[i], σᵥ²[i], 1, ϵ[i])
           if a₀ < -40
               ll[i] = ( - 0.5 * log(σ²[i]) - log(sqrt2π) - 0.5 * ϵ[i]^2 / σᵥ²[i]
                         + _tn_step(sf_tn_mean, a₀, δᵢ) )
           else
               ll[i] = ( - 0.5 * log(σ²[i]) + normlogpdf((μ[i] + ϵ[i]) / sqrt(σ²[i]))
                         + _tn_step(_tn_invmills, a₀, δᵢ) )
           end
       elseif min(a₁, a₀) < -40    # no Gaussian bulk, as (μ+ϵ)²/σ² = ϵ²/σᵥ² + a₀² - a₁²
           if a₁ <= 0
               ll[i] = ( - 0.5 * log(σ²[i]) - log(sqrt2π) - 0.5 * ϵ[i]^2 / σᵥ²[i]
                         + _normlogcdf_halfsq(a₁) - _normlogcdf_halfsq(a₀) )
           else    # ½a₁² - ½ϵ²/σᵥ² = wμ*/σᵤ² - ½d², without cancellation (see the note above)
               sᵢ = sqrt(σ²[i])
               wᵢ = μ[i] + σᵤ[i] * ϵ[i] / (sᵢ + σᵤ[i])
               dᵢ = sqrt(σᵥ²[i]) * wᵢ / (sᵢ * σᵤ[i])
               ll[i] = ( - 0.5 * log(σ²[i]) - log(sqrt2π) + normlogcdf(a₁)
                         + wᵢ * μₛ[i] / σᵤ²[i] - 0.5 * dᵢ^2 - _normlogcdf_halfsq(a₀) )
           end
       else
           @views ll[i] = (- 0.5 * log(σ²[i])
                        + normlogpdf((μ[i] + ϵ[i]) / sqrt(σ²[i]))
                        + _stable_normlogcdf(a₁)
                        - _stable_normlogcdf(a₀) )
       end
          end
       end

return -sum(ll)
end



#* -------   Normal Half-Normal -----------------#

function LL_T(::Type{Half}, Y::Matrix, X::Matrix, z, q, W::Matrix, V::Matrix,
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, dum2, ::Nothing  )

    β  = rho[po.begx : po.endx]
    δ2 = rho[po.begw : po.endw]
    γ  = rho[po.begv : po.endv]

    σᵤ² = exp.(W * δ2)
    σᵥ² = exp.(V * γ)
    σ²  = σᵤ² + σᵥ²
    ϵ   = PorC*(Y - X * β)
    μₛ  = ( - σᵤ² .* ϵ) ./ σ²
    σₛ² = (σᵥ² .* σᵤ²) ./ σ²

ll = Vector{eltype(ϵ)}(undef, nobs)
@floop begin
@inbounds for i in (1:nobs)
       @views ll[i] = ( - 0.5 * log(σ²[i])
                         + normlogpdf( (ϵ[i]) / sqrt(σ²[i]))
                         + _stable_normlogcdf((μₛ[i]) / sqrt(σₛ²[i]))
                         - normlogcdf(0)  )
  end
end

    return -sum(ll)
end




#* ------ Exponential --------------------#


function LL_T(::Type{Expo}, Y::Matrix, X::Matrix, z, q, W::Matrix, V::Matrix,
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, dum2, ::Nothing  )

    β  = rho[po.begx : po.endx]
    δ2 = rho[po.begw : po.endw]
    γ  = rho[po.begv : po.endv]

    # σᵤ² is the exponential's squared scale, which for this distribution is also Var(u);
    # σᵤ = sqrt(σᵤ²) is the scale and equals E(u).
    σᵤ² = exp.(W * δ2)
    σᵤ  = sqrt.(σᵤ²)
    σᵥ² = exp.(V * γ)
    σᵥ  = sqrt.(σᵥ²)

    ϵ   = PorC*(Y - X * β)

    # With a = -ϵ/σᵥ - σᵥ/σᵤ, the terms
    # log Φ(a) ≈ -a²/2 and ϵ/σᵤ + σᵥ²/(2σᵤ²) are each O((σᵥ/σᵤ)²) and cancel as σᵤ → 0:
    # a value error of up to about eps·a²/2 per observation (1e-7 at ln σᵤ² = -24, 2 at -40,
    # 1e9 at -60 with σᵥ = 0.3), which makes the log-likelihood RISE spuriously toward the
    # σᵤ → 0 boundary. Since -a²/2 = -ϵ²/(2σᵥ²) - ϵ/σᵤ - σᵥ²/(2σᵤ²) exactly,
    #   ll = -log σᵤ - ϵ²/(2σᵥ²) + h(a),   h(a) = log Φ(a) + a²/2 = _normlogcdf_halfsq(a),
    # with no large term left. As in the truncated-normal likelihoods, the rewritten form is used
    # only where it is needed, a < -40; every other observation evaluates the direct expression.
ll = Vector{eltype(ϵ)}(undef, nobs)
@floop begin
@inbounds for i in (1:nobs)
       a = -(ϵ[i]/σᵥ[i])  - (σᵥ[i]/σᵤ[i])
       if a < -40
           ll[i] = ( -log(σᵤ[i])
                     - ϵ[i]^2/(2*σᵥ²[i])
                     + _normlogcdf_halfsq(a) )
       else
           @views ll[i] = ( -log(σᵤ[i])
                             + _stable_normlogcdf(a)
                             + ϵ[i]/σᵤ[i]
                             + σᵥ²[i]/(2*σᵤ²[i]) )
       end
          end
       end
return -sum(ll)
end

#* ----- scaling property -------------

function LL_T(::Type{Trun_Scale}, Y::Matrix, X::Matrix, Z::Matrix, Q::Matrix, W::Matrix,  V::Matrix,
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, dum2, ::Nothing )

    β  = rho[po.begx : po.endx]
    δ1 = rho[po.begz : po.endz]
    τ  = rho[po.begq : po.endq]
    δ2 = rho[po.begw : po.endw]
    γ  = rho[po.begv : po.endv]

 #  h   = exp(Z*δ1)
    hscale = exp.(Q*τ)
    μ   = (Z*δ1) .* hscale              # Q is _cons; μ is the after-mutiplied-by-scaling function
    σᵤ² = exp.(W * δ2) .* hscale.^2     # The notation of σᵤ² is different from the paper. W is _cons
    σᵤ  = sqrt.(σᵤ²)
    σᵥ² = exp.(V * γ)
    σ²  = σᵤ² + σᵥ²
    ϵ   = PorC*(Y - X * β)
    μₛ  = (σᵥ² .* μ - σᵤ² .* ϵ) ./ σ²
    σₛ² = (σᵥ² .* σᵤ²) ./ σ²

    ll = Vector{eltype(ϵ)}(undef, nobs)
    @floop begin
    @inbounds for i in (1:nobs)
           a₁ = (μₛ[i]) / sqrt(σₛ²[i])
           a₀ = (μ[i]) / σᵤ[i]
           if _tn_tiny(μ[i], σᵤ²[i], σᵥ²[i], ϵ[i], a₀, a₁)    # see LL_T(Trun)
               δᵢ = _tn_shift(μ[i], σᵤ²[i], σᵥ²[i], 1, ϵ[i])
               if a₀ < -40
                   ll[i] = ( - 0.5 * log(σ²[i]) - log(sqrt2π) - 0.5 * ϵ[i]^2 / σᵥ²[i]
                             + _tn_step(sf_tn_mean, a₀, δᵢ) )
               else
                   ll[i] = ( - 0.5 * log(σ²[i]) + normlogpdf((μ[i] + ϵ[i]) / sqrt(σ²[i]))
                             + _tn_step(_tn_invmills, a₀, δᵢ) )
               end
           elseif min(a₁, a₀) < -40    # no Gaussian bulk (see LL_T(Trun) and the note above)
               if a₁ <= 0
                   ll[i] = ( - 0.5 * log(σ²[i]) - log(sqrt2π) - 0.5 * ϵ[i]^2 / σᵥ²[i]
                             + _normlogcdf_halfsq(a₁) - _normlogcdf_halfsq(a₀) )
               else    # ½a₁² - ½ϵ²/σᵥ² = wμ*/σᵤ² - ½d², without cancellation
                   sᵢ = sqrt(σ²[i])
                   wᵢ = μ[i] + σᵤ[i] * ϵ[i] / (sᵢ + σᵤ[i])
                   dᵢ = sqrt(σᵥ²[i]) * wᵢ / (sᵢ * σᵤ[i])
                   ll[i] = ( - 0.5 * log(σ²[i]) - log(sqrt2π) + normlogcdf(a₁)
                             + wᵢ * μₛ[i] / σᵤ²[i] - 0.5 * dᵢ^2 - _normlogcdf_halfsq(a₀) )
               end
           else
               @views ll[i] = ( - 0.5 * log(σ²[i])
                      + normlogpdf((μ[i] + ϵ[i]) / sqrt(σ²[i]))
                      + _stable_normlogcdf(a₁)
                      - _stable_normlogcdf(a₀)  )
           end
              end  # for
    end  # begin
    return -sum(ll)

end


#* -------- panel fixed effect, Half normal, Wang and Ho 2010 --------


function LL_T(::Type{PFEWHH}, Ỹ::Union{Vector,Matrix}, X̃::Matrix, z, Q::Matrix, w, v,
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, idt::Matrix{Any}, ::Nothing)

    β  = rho[1:po.endx]
    τ  = rho[po.begq : po.endq]
    δ2 = rho[po.begw]
    γ  = rho[po.begv]


    μ   = 0.0
    σᵤ² = exp(δ2)
    σᵥ² = exp(γ)
    h   = exp.(Q*τ)
    ϵ̃   = PorC*(Ỹ - X̃ * β)

    nofid = size(idt,1)

ll = Vector{eltype(ϵ̃)}(undef, nofid)
@floop begin
 @inbounds for i in 1:nofid
        @views ind = idt[i,1]
        @views   h̃ = sf_demean(h[ind])
              σₛₛ² = 1.0/(h̃'*h̃*(1/σᵥ²) + 1/σᵤ²)
        @views μₛₛ = (μ/σᵤ² - ϵ̃[ind]'*h̃*(1/σᵥ²)) * σₛₛ²
        @views es2 = -0.5*sum(sf_demean(ϵ̃[ind]).^2)*(1/σᵥ²)
        @views KK  = -0.5*(idt[i,2] - 1)*log(2π) - 0.5*(idt[i,2] - 1)*γ

              a₁ = μₛₛ/sqrt(σₛₛ²)
              a₀ = μ/sqrt(σᵤ²)
              if _tn_panel_extreme(a₁, a₀)    # quadratics and log Φ without their bulk
                  @views δ = _tn_shift(μ, σᵤ², σᵥ², h̃'*h̃, ϵ̃[ind]'*h̃)
                  ll[i] = (KK + es2
                          + 0.5*log(σₛₛ²) - 0.5*δ2
                          + _tn_panel_core(a₁, a₀, δ) )
              else
              ll[i] = (KK + es2
                      + 0.5*((μₛₛ^2)/σₛₛ² - μ^2/σᵤ²)
                      + 0.5*log(σₛₛ²) + _stable_normlogcdf(a₁)
                      - 0.5*δ2 - _stable_normlogcdf(a₀) )
              end
           end # for i=1:N
      end
return -sum(ll)
end


#* -------- panel fixed effect, truncated normal, Wang and Ho 2010 --------


function LL_T(::Type{PFEWHT}, Ỹ::Union{Vector,Matrix}, X̃::Matrix, z, Q::Matrix, w, v,
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, idt::Matrix{Any}, ::Nothing)

    β  = rho[1:po.endx]
    δ1 = rho[po.begz]
    τ  = rho[po.begq:po.endq]
    δ2 = rho[po.begw]
    γ  = rho[po.begv]

    μ   = δ1
    σᵤ² = exp(δ2)
    σᵥ² = exp(γ)
    h   = exp.(Q*τ)
    ϵ̃   = PorC*(Ỹ - X̃ * β)

    nofid = size(idt,1)

    ll = Vector{eltype(ϵ̃)}(undef, nofid)
    @floop begin
      @inbounds for i in 1:nofid
            @views ind = idt[i,1]
            @views h̃    = sf_demean(h[ind])
                  σₛₛ² = 1.0/(h̃'*h̃*(1/σᵥ²) + 1/σᵤ²)
            @views μₛₛ  = (μ/σᵤ² - ϵ̃[ind]'*h̃*(1/σᵥ²)) * σₛₛ²
            @views es2  = -0.5*sum(sf_demean(ϵ̃[ind]).^2)*(1/σᵥ²)

            @views KK   = -0.5*(idt[i,2] - 1)*log(2π) - 0.5*(idt[i,2] - 1)*γ

                  a₁ = μₛₛ/sqrt(σₛₛ²)
                  a₀ = μ/sqrt(σᵤ²)
                  if _tn_panel_extreme(a₁, a₀)    # quadratics and log Φ without their bulk
                      @views δ = _tn_shift(μ, σᵤ², σᵥ², h̃'*h̃, ϵ̃[ind]'*h̃)
                      ll[i] = (KK + es2
                              + 0.5*log(σₛₛ²) - 0.5*δ2
                              + _tn_panel_core(a₁, a₀, δ) )
                  else
                  ll[i] = (KK + es2
                          + 0.5*((μₛₛ^2)/σₛₛ² - μ^2/σᵤ²)
                          + 0.5*log(σₛₛ²) + _stable_normlogcdf(a₁)
                          - 0.5*δ2 - _stable_normlogcdf(a₀) )
                  end
                  end # for i=1:N
       end
     return -sum(ll)
end


#* -------- panel BC1992, Time Decay Model --------------

function LL_T(::Type{PanDecay}, Y::Union{Vector,Matrix}, X::Matrix, Z::Matrix, Q::Matrix, W::Matrix, V::Matrix,
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, idt::Matrix{Any}, ::Nothing)


    β  = rho[1:po.endx]
    δ1 = rho[po.begz:po.endz]
    τ  = rho[po.begq:po.endq]
    δ2 = rho[po.begw]
    γ  = rho[po.begv]

    nofid = size(idt,1)

    σᵤ² = exp(δ2)
    σᵤ  = sqrt(σᵤ²)
    σᵥ² = exp(γ)
    σᵥ  = sqrt(σᵥ²)
    σₛ² = σᵤ² + σᵥ²

    ll = Vector{eltype(rho)}(undef, nofid)
    @floop begin
      @inbounds for i = 1:nofid
            @views ind = idt[i,1]
            @views Tᵢ  = idt[i,2]
            @views ε   = PorC*(Y[ind] - X[ind, :] * β)
            @views μ   = (Z[ind, :] * δ1)[1]  # ok becuase time-invariant
            @views Gₜ  = exp.(Q[ind, :] * τ)
                   Gε  = Gₜ .* ε

                  Σε²  = sum(ε.^2)  #*  sensitive in this part, need precision
                  ΣGε  = sum(Gε)
                  ΣGₜ² = sum(Gₜ.^2)

                  μₛ      = (μ * σᵥ² - ΣGε * σᵤ²) / (σᵥ² + σᵤ² * ΣGₜ²)
                  σₛ²     =  σᵥ² * σᵤ² / (σᵥ² + σᵤ² * ΣGₜ²)
                  comp1   = (μ * σᵥ² - σᵤ² * ΣGε)^2 / (σᵥ² * σᵤ² * (σᵥ² + σᵤ² * ΣGₜ²))
                  bigterm =  Σε²/σᵥ² + (μ)^2 / σᵤ² - comp1

                  a₁ = μₛ/sqrt(σₛ²)
                  a₀ = μ/σᵤ
                  if _tn_panel_extreme(a₁, a₀)    # quadratics and log Φ without their bulk
                      ll[i] = ( - 0.5*(Tᵢ)*log(2π) + 0.5*log(σₛ²) - 0.5*Σε²/σᵥ² - Tᵢ*0.5*γ
                              - 0.5*δ2 + _tn_panel_core(a₁, a₀, _tn_shift(μ, σᵤ², σᵥ², ΣGₜ², ΣGε)) )
                  else
                  ll[i] = ( - 0.5*(Tᵢ)*log(2π) + _stable_normlogcdf(a₁)
                          + 0.5*log(σₛ²) - 0.5*bigterm - Tᵢ*0.5*γ
                          - 0.5*δ2 - _stable_normlogcdf(a₀)  )
                  end
              end
    end
    return -sum(ll)
end



#* -------- panel Kumbhakar Model --------------

function LL_T(::Type{PanKumb90}, Y::Union{Vector,Matrix}, X::Matrix, Z::Matrix, Q::Matrix, W::Matrix, V::Matrix,
  PorC::Int64, nobs::Int64, po::NamedTuple, rho, idt::Matrix{Any}, ::Nothing)


  β  = rho[1:po.endx]
  δ1 = rho[po.begz:po.endz]
  τ  = rho[po.begq:po.endq]
  δ2 = rho[po.begw]
  γ  = rho[po.begv]

  nofid = size(idt,1)

  σᵤ² = exp(δ2)
  σᵤ  = sqrt(σᵤ²)
  σᵥ² = exp(γ)
  σᵥ  = sqrt(σᵥ²)
  σₛ² = σᵤ² + σᵥ²

  ll = Vector{eltype(rho)}(undef, nofid)
  @floop begin
  @inbounds for i = 1:nofid
         @views ind = idt[i,1]
         @views Tᵢ  = idt[i,2]
         @views ε   = PorC*(Y[ind] - X[ind, :] * β)
         @views μ   = (Z[ind, :] * δ1)[1]  # ok becuase time-invariant
         @views Gₜ  = 2 ./ (1 .+ exp.(Q[ind, :] * τ))
                Gε  = Gₜ .* ε

               Σε²  = sum(ε.^2)  #*  sensitive in this part, need precision
               ΣGε  = sum(Gε)
               ΣGₜ² = sum(Gₜ.^2)

               μₛ  = (μ * σᵥ² - ΣGε * σᵤ²) / (σᵥ² + σᵤ² * ΣGₜ²)
               σₛ²  =  σᵥ² * σᵤ² / (σᵥ² + σᵤ² * ΣGₜ²)
               comp1 = (μ * σᵥ² - σᵤ² * ΣGε)^2 / (σᵥ² * σᵤ² * (σᵥ² + σᵤ² * ΣGₜ²))
               bigterm =  Σε²/σᵥ² + (μ)^2 / σᵤ² - comp1

              a₁ = μₛ/sqrt(σₛ²)
              a₀ = μ/σᵤ
              if _tn_panel_extreme(a₁, a₀)    # quadratics and log Φ without their bulk
                  ll[i] = ( - 0.5*(Tᵢ)*log(2π) + 0.5*log(σₛ²) - 0.5*Σε²/σᵥ² - Tᵢ*0.5*γ
                          - 0.5*δ2 + _tn_panel_core(a₁, a₀, _tn_shift(μ, σᵤ², σᵥ², ΣGₜ², ΣGε)) )
              else
              ll[i] = ( - 0.5*(Tᵢ)*log(2π) + _stable_normlogcdf(a₁)
                     + 0.5*log(σₛ²) - 0.5*bigterm - Tᵢ*0.5*γ
                     - 0.5*δ2 - _stable_normlogcdf(a₀)  )
              end
          end # for i
       end  # @floop
  return -sum(ll)
end



#* -------- panel fixed effect, Half normal, CSW JoE 2014 (CSN) --------


function joe2014lnmvnpdf(x, s2)

    k = length(x)

    # lnf = -k/2*log(2π)-0.5*log(det(σ))-0.5*x'*inv(σ)*x

    invSig = Matrix(I, k, k).*(1/s2) .+ (1/s2) # the analytic form of inv(σ)
    detSig = (s2^k)/(k+1) # analytic form of det(σ)
    # With k = Tᵢ - 1, det(σ) = s2^k/(k+1)
    # leaves the Float64 range on long panels -- at Tᵢ = 200 unless 0.024 < s2 < 36 --
    # and log(detSig) is then ±Inf. Its Hessian fails earlier: ForwardDiff's second
    # derivative of log(x) is Inf below x = 7.5e-155 and loses digits above about
    # 1.6e154 (at Tᵢ = 200: s2 outside [0.17, 6.1]). Outside [1e-154, 1e154] the log is
    # therefore taken in closed form, k*log(s2) - log(k+1); inside, where the value,
    # gradient and Hessian of log(detSig) are right, log(detSig) is used.
    ldet = (1e-154 <= _mlevalue(detSig) <= 1e154) ? log(detSig) : k*log(s2) - log(oftype(s2, k + 1))
    res = -k/2*log(2π) - 0.5*ldet - 0.5*x'*invSig*x

    return res
end



    # log B for LL_T(PFECSWH) below, where
    #   B = ∫ φ(u) ∏ₜ Φ(bₜ - ρu) du,   bₜ = -(λ/σ)ϵₜ,  ρ = λ/√Tᵢ,
    # for a firm whose probability-form B (quadgk) is at or below 1e-15. B shrinks roughly
    # geometrically with Tᵢ (as λ → 0 every Φ → 1/2 and B → 2^-Tᵢ, below 1e-15 from
    # Tᵢ = 50), and log(max(B, 1e-15)) is then too high: on simulated 20-firm panels by up
    # to 7.3 nats at T = 50 and 355 to 636 at T = 100, with no dependence on λ on the
    # floor. This is _tre_logorthant (γ = 1) with three changes, all
    # local to this model (panel_TRE keeps its helper as it is):
    #  (i) where the peak cannot be placed within a small part of its width in Float64 --
    #      a one-nat width at or below 1e-10(1 + |u*|), while the peak search stops at
    #      1e-13(1 + |u*|) -- the integral is not attempted and the firm keeps the floored
    #      value logB_s25 = log(max(B, 1e-15)). Such peaks (|u*| of 1e11 and more) come from
    #      trial points with σᵥ² near underflow, such as BFGS's first step; integrating
    #      there threw a DomainError or ran for seconds to minutes;
    #  (ii) each quadgk call stops after 1e5 integrand evaluations (quadgk's default 1e7
    #      took minutes at such points; ordinary firms need a few hundred);
    #  (iii) at large λ the peak is sharply one-sided, and one adaptive rule over [0, 60]
    #      misses the structure of the wide side near the peak (up to 3e-6 in log B above
    #      λ = 1e4, Hessian diagonals of the wrong sign). When the narrow width is below 0.2 of the wide
    #      one (q), the wide side is integrated piecewise between q, q√10, 10q, ... and 1
    #      (in its own widths) and 60; otherwise the layout of _tre_logorthant is kept.
function _csw_logorthant(B, ρ, γ, logB_s25)
    _csw_past_switch(ρ, γ, length(B)) && return _csw_logorthant_x(B, ρ, γ, logB_s25)
    u0, sL, sR = _tre_orthant_peak(B, ρ, γ)
    r   = ρ/γ
    a₀  = (B .- ρ*u0)./γ
    hu0 = normlogpdf(u0) + sum(_stable_normlogcdf, a₀)
    isfinite(_mlevalue(hu0)) || return hu0
    min(sL, sR) > 1e-10*(1 + abs(u0)) || return logB_s25
    q = min(sL, sR)/max(sL, sR)
    if q >= 0.2
        I = sL*quadgk(t -> exp(_tre_dh(a₀, r, u0, -sL*t)), 0, _TRE_SPAN, rtol = 1e-10, maxevals = 10^5)[1] +
            sR*quadgk(t -> exp(_tre_dh(a₀, r, u0,  sR*t)), 0, _TRE_SPAN, rtol = 1e-10, maxevals = 10^5)[1]
    else
        sw, sn = sL > sR ? (-sL, sR) : (sR, -sL)        # signed widths: wide side, narrow side
        fw = t -> exp(_tre_dh(a₀, r, u0, sw*t))
        Iw = quadgk(fw, 0.0, q, rtol = 1e-10, maxevals = 10^5)[1]
        lo = q
        while lo < 1.0
            hi = min(lo*sqrt(10.0), 1.0)
            Iw += quadgk(fw, lo, hi, rtol = 1e-10, maxevals = 10^5)[1]
            lo = hi
        end
        Iw += quadgk(fw, 1.0, _TRE_SPAN, rtol = 1e-10, maxevals = 10^5)[1]
        In  = quadgk(t -> exp(_tre_dh(a₀, r, u0, sn*t)), 0, _TRE_SPAN, rtol = 1e-10, maxevals = 10^5)[1]
        I   = abs(sw)*Iw + abs(sn)*In
    end
    return hu0 + log(I)
end

    # The branch of _csw_logorthant for σᵤ/σᵥ = (ρ/γ)√Tᵢ above 300, where LL_T(PFECSWH) sends
    # every firm to _csw_logorthant.
    # There each Φ factor turns from 1 to its Gaussian tail within a few 1/ρ of its turn
    # bₜ/ρ, and the peak sits on the shoulder of the first turn. The value above is right
    # there, but its derivatives were not. Every argument aₜ = (bₜ - ρu*)/γ has partials of
    # order ρ (those of bₜ and of ρu*, each about ρ times a residual), so in the Hessian
    # terms of order ρ² from h(u*) and from the integral had to cancel to entries of order
    # one; and the Dual rule of _tre_dlogcdf, m(a) - m(a₀) - δ, cancels to eps·a for the
    # large positive arguments of the factors that have not turned. On 60 simulated firms
    # near the bound of σᵥ² (σᵤ/σᵥ from 8.4e4 to 1.3e6) the gradient was off by up to 3.7e-5
    # and the Hessian by up to 1.6e-4 of their largest entries, and the ln σᵥ² row of the
    # Hessian, whose true entries are of order 1e-10 there, by up to 6.6e-3: fits that run
    # toward σᵥ → 0 mostly ended with converged = false, many at Newton's iteration limit.
    #
    # _csw_logorthant_x measures the integration variable from the first turn: with
    # s = b_k/ρ (k the factor whose turn comes first) and u = s + v, exactly
    #     B = ∫ φ(s + v) ∏ₜ Φ((bₜ - b_k)/γ - (ρ/γ)v) dv.
    # The first turn's own argument is then -(ρ/γ)v, whose partials have no part of order
    # ρ; s enters only through φ, which is smooth; the other factors enter through
    # bₜ - b_k and matter only within a few 1/ρ of the first turn. Nothing of order ρ is
    # left to cancel. The integral is taken piecewise with breakpoints at the turns
    # (_tre_cliffs and _tre_cliffint) and h is differenced through _tre_dlogcdf_x, whose Dual
    # rule forms the difference of the two Mills ratios directly where an argument is positive.
    # Against 256-bit references on 206 simulated firms from σᵤ/σᵥ = 300 to the bound, the
    # gradient and Hessian are within 1.3e-15 and 6.8e-15 of their largest entries, and the
    # value is as accurate as with the lines above. Where two residuals
    # of a firm lie within a few turn widths (σ/λ in the residual) of each other, the
    # rounding of the residuals limits the gradient and Hessian to a few times 1e-10 of
    # their largest entries: a change of one residual by one rounding moves the true
    # values by as much.
    #
    # The switch compares ρ/γ with 300/√Tᵢ, from the value parts, so a Float64 pass and a
    # ForwardDiff pass take the same branch. LL_T passes ρ = λ/√Tᵢ and γ = 1, and rounding
    # is monotone, so no σᵤ/σᵥ ≤ 300 takes the new branch: at σᵤ/σᵥ ≤ 300 (a long-panel
    # firm on the 1e-15 floor) the lines above run. (One rounding above 300, some Tᵢ still
    # take the lines above, which are accurate there.)
const _CSW_SWITCH = 300.0
_csw_past_switch(ρ, γ, T) = _mlevalue(ρ)/_mlevalue(γ) > _CSW_SWITCH/sqrt(T)

    # h(u* + w) - h(u*), term by term; us = s + v₀ is u* carrying the partials of s
function _csw_dh_x(a₀, r, us, w)
    δ = -r*w
    d = -w*(us + w/2)
    @inbounds for k in eachindex(a₀)
        d += _tre_dlogcdf_x(a₀[k], δ)
    end
    return d
end

function _csw_logorthant_x(B, ρ, γ, logB_s25)
    u0, sL, sR = _tre_orthant_peak(B, ρ, γ)
    k   = argmin(map(_mlevalue, B))          # the first turn is at u = B[k]/ρ
    r   = ρ/γ
    s   = B[k]/ρ
    v0  = u0 - _mlevalue(s)                  # the peak, measured from s
    us  = s + v0                             # u* with the partials of s
    a₀  = (B .- B[k])./γ .- r*v0             # the arguments at the peak; a₀[k] = -r*v0
    hv0 = normlogpdf(us) + sum(_stable_normlogcdf, a₀)
    isfinite(_mlevalue(hv0)) || return hv0
    min(sL, sR) > 1e-10*(1 + abs(u0)) || return logB_s25      # guard (i) above
    return hv0 + log(_tre_cliffint(w -> exp(_csw_dh_x(a₀, r, us, w)), _tre_cliffs(a₀, r, u0, sL, sR)))
end

function LL_T(::Type{PFECSWH}, Ỹ::Union{Vector,Matrix}, X̃::Matrix, z, q, w, v,
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, idt::Matrix{Any}, ::Nothing)

    β   = rho[1:po.endx]
    # σᵤ² and σᵥ² are clamped to [1e-12, 1e12], as in jlmsbc(PFECSWH) and
    # LL_T(PTREH); inside those bounds nothing changes. Without the clamp a trial step that
    # underflows σᵥ² to 0 makes λ = Inf, and the probability-form quadgk below throws a
    # DomainError on its NaN integrand. The log-space branch below changes the path of some
    # fits, and without the clamp 4 of 9 BFGS test fits reached such a step and stopped with
    # that error.
    σᵤ² = clamp(exp(rho[po.begw]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵥ² = clamp(exp(rho[po.begv]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)

    σᵤ = sqrt(σᵤ²)
    σᵥ = sqrt(σᵥ²)
    λ  =  σᵤ / σᵥ
    σ² =  σᵤ² + σᵥ²
    ϵ  = PorC*(Ỹ - X̃ * β)

    nofid = size(idt,1)

    ll = Vector{eltype(ϵ)}(undef, nofid)
    @floop begin
    @inbounds for i in 1:nofid
           @views Tᵢ    = idt[i, 2]
           @views ind_a = idt[i, 1][1: Tᵢ-1]
           @views ind   = idt[i, 1]
           @views A     = joe2014lnmvnpdf(ϵ[ind_a], σ²)
                  B     = quadgk(u -> normpdf(u)*prod(normcdf.(-λ/sqrt(σ²)*ϵ[ind].-λ/sqrt(Tᵢ)*u)), -Inf, Inf, rtol=1e-8)
           # The probability form is used only where the floor does not bind AND λ = σᵤ/σᵥ ≤ 300.
           # As λ grows, each Φ factor turns into a step of width about √Tᵢ/λ in u, and quadgk
           # over (-∞, ∞) can stop on a wrong answer for a firm above the floor: in two narrow
           # windows of B (about 2.4e-15 to 5e-15, just above the floor, and about 7e-5 to 1e-4),
           # most for firms with few periods, and by more the larger λ is. In checks it missed
           # there by up to 0.2 nats per firm between λ = 300 and 1000 and 0.19 above 1000,
           # with gradients off by up to about their own size; below 300 by up to 0.07 near 300,
           # 7e-3 at 200 and 2.4e-5 at 100 (these misses stay). The log-space branch was within
           # 4e-14 on every firm checked above 300. λ is one number per evaluation, so at a trial
           # point with λ > 300 no firm takes the probability form, and with λ ≤ 300 every firm
           # above the floor does. B is still computed for every firm: it decides the floor, and
           # the width guard of _csw_logorthant falls back to log(B).
           if B[1] > _MLE_CLAMP.log_lo && _mlevalue(λ) <= 300.0
               @views ll[i] = A + log(B[1])  + log(2)*Tᵢ
           else                              # at or below the floor, or λ > 300: log B in log space
               @views ll[i] = A + _csw_logorthant(-λ/sqrt(σ²)*ϵ[ind], λ/sqrt(Tᵢ), one(λ),
                                                  log(max(B[1], _MLE_CLAMP.log_lo))) + log(2)*Tᵢ
           end
             end
          end
    return -sum(ll)
end



#* ------------------ true random effect model, Half Normal --------------

function LL_T(::Type{PTREH}, Y::Matrix, X::Matrix, z, Q::Matrix, w, v,
       PorC::Int64, nobs::Int64, po::NamedTuple, rho, idt::Matrix{Any}, ::Nothing)

        β   = rho[1:po.endx]
        # Same parameter map as jlmsbc(PTREH) in SFindex.jl: without the clamp a
        # subnormal σᵤ² makes 1/σᵤ² overflow to Inf, and μ'invΣᵤ is then 0*Inf = NaN.
        σₐ² = clamp(exp(rho[po.begq]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
        σᵤ² = clamp(exp(rho[po.begw]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
        σᵥ² = clamp(exp(rho[po.begv]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
        σᵤ  = sqrt(σᵤ²)

        ε1  = PorC*(Y - X * β)

       nofid = size(idt,1)

   ll = Vector{eltype(ε1)}(undef, nofid)
   @floop begin
          for i = 1:nofid
              @views ind  = idt[i, 1]
              @views T    = idt[i, 2]
              @views ε    = ε1[ind]
                     μ    = zeros(eltype(ε1), T)
                   # Σ = σᵥ²I + σₐ²11' has eigenvalues σᵥ²+Tσₐ² once and σᵥ² (T-1) times,
                   # so log|Σ| is the sum below.  Forming the determinant first and then
                   # flooring it underflows for an ordinary σᵥ² once T is moderate: at
                   # T = 10, σₐ² = 1, σᵥ² = 0.01 the floored form returns -34.54 for a true
                   # -39.14, and the error reaches 202 nats over an ordinary parameter range.
                  logdetΣ = (T-1)*log(σᵥ²) + log(σᵥ² + T*σₐ²)
                     invΣ = (1/σᵥ²) * (Matrix(I,T,T) - σₐ²/(σᵥ² + T*σₐ²)*ones(T,T))
                    invΣᵤ = (1/σᵤ²) * Matrix(I,T,T)
                     K    = ε'*invΣ - μ'*invΣᵤ

                   # G = inv(inv(Σ)+inv(Σᵤ)) in closed form.  inv(Σ)+inv(Σᵤ) = c*I + d*11'
                   # with c = 1/σᵤ²+1/σᵥ² and d = -σₐ²/(σᵥ²(σᵥ²+Tσₐ²)), so by Sherman-Morrison
                   # G = γ²I + ρ²11' with γ² = 1/c and ρ² = -d/(c(c+Td)).  c+Td is written as
                   # 1/σᵤ² + 1/(σᵥ²+Tσₐ²): algebraically identical, but a sum of two positive
                   # terms, so it cannot cancel.  Written as c*(1+g) it loses every digit once
                   # σₐ²/σᵥ² is large and rounds to exactly 0 (ρ² = Inf, then a NaN integrand)
                   # on parameters an optimiser reaches.  Both γ² and ρ² are strictly positive
                   # for positive variances, so neither needs a floor, and flooring γ² broke
                   # the identity above and with it the cancellation of the two quadratic
                   # forms in ll below, which is what let the objective run to -1e35.
                     c    = 1/σᵤ² + 1/σᵥ²
                     cTd  = 1/σᵤ² + 1/(σᵥ² + T*σₐ²)
                     γ²   = 1/c
                     ρ²   = σₐ²/(σᵥ²*(σᵥ² + T*σₐ²)) / (c*cTd)
                     G    = γ² * Matrix(I,T,T) + ρ² * ones(T,T)
                     logdetG = (T-1)*log(γ²) + log(γ² + T*ρ²)

                     B    = (-K*G)'
                     γ    = sqrt(γ²)
                     ρ    = sqrt(ρ²)
                   # log of the orthant probability, formed in log space: the probability
                   # itself underflows to zero well before a floor such as 1e-15 could be a
                   # sensible substitute for it, and on the flat side of such a floor the
                   # objective's slope is exactly 0.  See _tre_logorthant above.  No clamp
                   # on the standardised argument is needed either (a clamp at ±500 would
                   # keep a·Φ(a) from becoming Inf·0 and exp(-a²/2) from being asked for
                   # absurd arguments): normlogcdf is finite and smooth at every finite
                   # argument, and normlogcdf(-8e5) is -3.2e11.
                   # Bx is B formed again without the cancellation of its 1/σᵥ²-sized terms
                   # (_tre_Bx above). _tre_logorthant uses it only for a firm past its switch
                   # (z > 160), whose integral it takes piecewise; for every other firm Bx is
                   # not used.
                     Bx   = _tre_Bx(ε, μ, σₐ², σᵤ², σᵥ², γ², ρ², T)
                     logcdfBG = _tre_logorthant(B, ρ, γ, Bx)

                   # The three quadratic forms of this likelihood are ONE.  With
                   # G = (Σ⁻¹+Σᵤ⁻¹)⁻¹ and B = -G(Σ⁻¹ε - Σᵤ⁻¹μ),
                   #
                   #   -½ε'Σ⁻¹ε - ½μ'Σᵤ⁻¹μ + ½B'(Σ⁻¹+Σᵤ⁻¹)B = -½(ε+μ)'(Σ+Σᵤ)⁻¹(ε+μ),
                   #
                   # since Σ⁻¹GΣ⁻¹ = Σ⁻¹ - Σ⁻¹GΣᵤ⁻¹ and Σ⁻¹GΣᵤ⁻¹ = (Σ+Σᵤ)⁻¹ -- which is
                   # only the statement that ε + μ is N(0, Σ+Σᵤ) before the truncation.  As
                   # three terms each is of order 1/σᵥ² and they cancel to O(1): on the
                   # N = 4, T = 10 panel at σᵥ² = 2.5e-12 the third is 7.0e9 and carries 3704
                   # of error, so each firm's contribution came out 3704 nats too high and
                   # the panel's 10016 -- against a direct 256-bit integral over the random
                   # effect, which the closed form has to match and does at ordinary
                   # parameters to 2e-15.  As one term, split into its within and between
                   # parts (both non-negative), nothing cancels and the same combination is
                   # right to 1e-16.  (Σ+Σᵤ)⁻¹ = (1/s²)(I - σₐ²/(s²+Tσₐ²)11'), s² = σᵤ²+σᵥ².
                     zz   = ε .+ μ
                     zbar = sum(zz)/T
                     s²   = σᵤ² + σᵥ²
                     quad = -sum(abs2, zz .- zbar)/(2*s²) - T*zbar^2/(2*(s² + T*σₐ²))

                   # μ is identically zero here, so the term below is log(σᵤ/2), and its floor
                   # would bind only for σᵤ < 2e-15, which the clamp of σᵤ² at 1e-12 rules out.
                   # In LL_T(PTRET), where μ = Zδ, such a floor could bind, so that likelihood
                   # forms the term without one (see the note there).
             @views ll[i] = (logcdfBG + 0.5*logdetG - 0.5*logdetΣ
                     - sum(log.(max.(σᵤ*normcdf.(μ/σᵤ), _MLE_CLAMP.log_lo)))
                     + quad - T*0.5*log(2*π) )

         end
       end
       return -sum(ll)
   end


#* ------ true random effect model, truncated normal ---------


function LL_T(::Type{PTRET}, Y::Matrix, X::Matrix, Z::Matrix, Q::Matrix, w, v,
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, idt::Matrix{Any}, ::Nothing)

    β   = rho[1:po.endx]
    δ   = rho[po.begz : po.endz]

    # Same parameter map as jlmsbc(PTRET) in SFindex.jl; see LL_T(PTREH) above.
    σₐ² = clamp(exp(rho[po.begq]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵤ² = clamp(exp(rho[po.begw]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵥ² = clamp(exp(rho[po.begv]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)

    σᵤ  = sqrt(σᵤ²)

    nofid = size(idt, 1)

ll = Vector{eltype(rho)}(undef, nofid)
@floop begin
       for i in 1:nofid
          @views ind = idt[i, 1]
          @views T   = idt[i, 2]
          @views ε   = PorC*(Y[ind] - X[ind, :] * β)
          @views μ   = (Z[ind, :] * δ)
        # log|Σ| from the eigenvalues of σᵥ²I + σₐ²11'; see LL_T(PTREH) above.
          logdetΣ   = (T-1)*log(σᵥ²) + log(σᵥ² + T*σₐ²)

          invΣ = (1/σᵥ²) * (Matrix(I,T,T) - σₐ²/(σᵥ² + T*σₐ²)*ones(T,T))
          invΣᵤ= (1/σᵤ²) * Matrix(I,T,T)
          K    = ε'*invΣ - μ'*invΣᵤ

        # G = inv(inv(Σ)+inv(Σᵤ)) by Sherman-Morrison, cancellation-free; see LL_T(PTREH).
          c    = 1/σᵤ² + 1/σᵥ²
          cTd  = 1/σᵤ² + 1/(σᵥ² + T*σₐ²)
          γ²   = 1/c
          ρ²   = σₐ²/(σᵥ²*(σᵥ² + T*σₐ²)) / (c*cTd)
          G    = γ² * Matrix(I,T,T) + ρ² * ones(T,T)
          logdetG = (T-1)*log(γ²) + log(γ² + T*ρ²)

          B    = (-K*G)'
          γ    = sqrt(γ²)
          ρ    = sqrt(ρ²)
        # log of the orthant probability COMBINED with the truncation constant
        # ∑ₜlog Φ(μₜ/σᵤ) before either is formed: apart, the two cancel to O(1) from
        # O(5.9e14) at the σᵤ² clamp and the difference is quantised at 0.06 -- 2.0 nats
        # against a direct 256-bit integral over the random effect, moving in steps.  See
        # _tre_logorthant_ratio above; D = B - γμ/σᵤ is its analytic, cancellation-free
        # form, every term of order q = σᵤ²/σᵥ² or smaller.
          m    = μ ./ σᵤ
          q    = σᵤ²/σᵥ²
          εbar = sum(ε)/T
          D    = (-q/((1+q)*(1+sqrt(1+q)))) .* μ .+ (ρ²/σᵤ²)*sum(μ) .-
                 (γ²/σᵥ²).*(ε .- (T*σₐ²/(σᵥ² + T*σₐ²))*εbar) .- ρ²*T*εbar/(σᵥ² + T*σₐ²)
          logratio = _tre_logorthant_ratio(B, ρ, γ, m, D)

        # The three quadratic forms as one; see LL_T(PTREH) above.
          zz   = ε .+ μ
          zbar = sum(zz)/T
          s²   = σᵤ² + σᵥ²
          quad = -sum(abs2, zz .- zbar)/(2*s²) - T*zbar^2/(2*(s² + T*σₐ²))

        # The truncated normal's normalising constant, -T·log(σᵤ) - ∑ₜnormlogcdf(μ/σᵤ).
        # A σᵤΦ(μ/σᵤ) form with a 1e-15 floor would hit the floor one step from a fitted
        # σᵤ = 0.152 (7.0 invented nats per period at μ/σᵤ = -8.5). The -∑ₜnormlogcdf(μ/σᵤ)
        # part is taken INSIDE logratio above, where it cancels against log P term by term
        # instead of as two formed logarithms.  Only -T·log(σᵤ) remains here.
          @views ll[i] = (logratio + 0.5*logdetG - 0.5*logdetΣ
                        - T*log(σᵤ)
                        + quad - T*0.5*log(2*π) )
       end
    end

    return -sum(ll)
end
