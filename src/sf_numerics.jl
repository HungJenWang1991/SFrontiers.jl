# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

# sf_numerics.jl
# Small numerical kernels shared by all four backends and predict. Each is a
# scalar function, so it broadcasts on CPU arrays and CuArrays alike, and ForwardDiff
# differentiates it without cancellation. Loaded before the backends, which import it.

module SFNumerics

using SpecialFunctions: erfcx, erfc

export sf_normlogcdf_lowtail, sf_tn_quantile_tail

"""
    sf_normlogcdf_lowtail(a)

log Φ(a) for a < -1.

`log(Φ(a))` formed from `Φ(a) = (1 + erf(a/√2))/2` cancels as a falls: its error, of either
sign, grows as Φ falls (at most about 1.5e-10 near a = -5, bounded by eps/Φ(a)) and reaches
0.037 just above a = -7.94, where a floor of 1e-15 on Φ starts to bind (3.7% below the true
Φ). Below that the floor holds log Φ at -34.5 while the truth falls like -a²/2 (-105.8 at
a = -14.3, -804.6 at a = -40). Here the value is formed from erfcx,
log Φ(a) = log(erfcx(-a/√2)/2) - a²/2, which carries no cancellation; below -40 from the
asymptotic series

    log Φ(a) = -a²/2 - log(-a) - log√(2π) + log(1 - 1/a² + 3/a⁴ - 15/a⁶ + 105/a⁸ - 945/a¹⁰ + …),

the one the analytic backend uses (`_stable_normlogcdf`), whose derivatives
ForwardDiff forms without the cancellation of the erfcx rule. The first omitted term,
10395/a¹², is below 1e-15 at the switch. erfcx and log1p run on the GPU (CUDA.jl's
SpecialFunctions extension). Callers keep their own expression for a ≥ -1, where it is
exact to rounding.
"""
@inline function sf_normlogcdf_lowtail(a)
    if a < -40
        x = inv(a * a)
        return -a * a / 2 - log(-a) - oftype(x, 0.9189385332046728) +
               log1p(-x * (1 - 3x * (1 - 5x * (1 - 7x * (1 - 9x)))))
    else
        return log(erfcx(-a / sqrt(oftype(a, 2))) / 2) - a * a / 2
    end
end

"""
    sf_tn_quantile_tail(y, r)

The truncated-normal quantile in its far tail. For u ~ N(μ, σ²) truncated below at 0 and
y = -μ/(σ√2), the draw at probability r is u = σ√2·t, where t ≥ 0 solves
erfc(y + t) = (1 - r)·erfc(y). The closed form u = μ + σ√2·erfcinv(erfc(y)·(1 - r)) breaks
down as y grows: erfc(y) turns subnormal at y ≈ 26.5 (μ/σ ≈ -37.5) and is 0 from y ≈ 27.2
(μ/σ ≈ -38.5) in Float64 (9.2 and 10.05 in Float32). The draws' derivatives are then NaN, the
draws themselves lose accuracy, and once erfc(y) is 0, erfcinv(0) = Inf makes every draw
infinite. Before that, μ and σ√2·erfcinv(...) cancel. With erfc(x) = erfcx(x)·exp(-x²) the
equation is g(t) = 0, where

    g(t) = t(2y + t) + log erfcx(y) - log erfcx(y + t) + log(1 - r),
    g'(t) = 2/(√π·erfcx(y + t)) > 0,

and g is convex. Newton starts from the root of the quadratic (the log-erfcx term dropped),
which lies to the right of the root, so the iterates fall monotonically onto it; four
steps reach machine precision for y > 8.

The term log erfcx(y) - log erfcx(y + t) is small (about t/y) while neither log is, and
ForwardDiff's rule for erfcx, erfcx'(x) = 2x·erfcx(x) - 2/√π, itself cancels as x grows
(relative error about eps·2x²): formed from erfcx, the draw's derivative with respect to μ
or σ had relative errors of about 1e-2 near μ/σ = -3e5 and 0.1 or more below -1e6, and a
Newton step taken through erfcx still left its second derivative (which MSLE's Newton uses)
wrong beyond about -3e6. Above y = 28 the term
and the Newton step are therefore formed from the asymptotic series erfcx(x) = S(x)/(x√π),

    S(x) = 1 - z + 3z² - 15z³ + 105z⁴ - 945z⁵ + …,   z = 1/(2x²),

as log1p(t/y) + log S(y) - log S(y + t), in which nothing large is differenced. It is the
series of `sf_normlogcdf_lowtail` at a = -x√2; the first omitted term, 10395z⁶, is below
1e-15 at the switch. Above y = 28 nothing underflows or cancels, so the draw and its
first and second ForwardDiff derivatives stay accurate however negative μ/σ is. On (8, 28]
the log-erfcx difference is taken from two logarithms of order 3, which costs t about
3·eps/E relative, E = -log(1 - r) (1.5e-13 at r = 1e-3; absolute error below 1e-15), and
its derivatives about 4e-10 relative for r ≥ 1e-3. The package calls this with a Float64
(or ForwardDiff Dual{Float64}) y; in pure Float32 the (8, 28] branch keeps only about three
digits in the derivatives. Returns t.
"""
@inline function sf_tn_quantile_tail(y, r)
    E = -log1p(-r)
    t = E / (y + sqrt(y * y + E))
    if y > 28
        # With erfcx(x) = S(x)/(x√π): log erfcx(y) - log erfcx(y + t) = log1p(t/y) + log S(y)
        # - log S(y + t), and g' = 2x/S(x) at x = y + t, so the Newton step g/g' = g·S(x)/(2x)
        # uses no erfcx either (through erfcx, ForwardDiff's second derivative of t went
        # wrong for y beyond about 2e6 whenever the last residual g was not exactly 0).
        ly = _sf_logS_erfcx(y)
        for _ in 1:4
            x = y + t
            lx = _sf_logS_erfcx(x)
            g = t * (2y + t) + (log1p(t / y) + (ly - lx)) - E
            t -= g * exp(lx) / (2x)
        end
    else
        ly = log(erfcx(y))
        for _ in 1:4
            ex = erfcx(y + t)
            g = t * (2y + t) + (ly - log(ex)) - E
            t -= g * ex * oftype(t, 0.886226925452758)    # g/g'; √π/2 = 0.8862269254527580
        end
    end
    return t
end

# log S(x) = log(√π·x·erfcx(x)) for x > 28, from the asymptotic series (see sf_tn_quantile_tail)
@inline function _sf_logS_erfcx(x)
    z = inv(2 * x * x)
    return log1p(-z * (1 - 3z * (1 - 5z * (1 - 7z * (1 - 9z)))))
end

# ---------------------------------------------------------------------------------------------
# The truncated normal's mean factor and log moment generating function, behind every
# closed-form JLMS/BC and truncated-normal E(u). Unlike the kernels above, these also serve
# the analytic backend (SFindex.jl, SFmtest.jl) and predict().
# ---------------------------------------------------------------------------------------------
export sf_tn_mean, sf_tn_logmgf

"""
    sf_tn_mean(a)

m(a) = a + φ(a)/Φ(a) = E(Z | Z > -a) for a standard normal Z. For u ~ N⁺(μ, σ²), the normal
N(μ, σ²) truncated below at 0, E(u) = σ·m(μ/σ): the closed-form JLMS is σₛ·m(μₛ/σₛ), and the
truncated-normal E(u) of the marginal effects is σᵤ·m(μ/σᵤ). `a` must be a floating-point
number or a ForwardDiff Dual (an Integer `a` ≥ -5 throws an InexactError).

Formed by adding a to the inverse Mills ratio λ = φ/Φ, m cancels as a → -∞, since λ ≈ -a
while m ≈ -1/a: with λ = exp(log φ - log Φ), two logarithms of about -a²/2 each, m has a
relative error of about eps·a⁴/4 (1.5e-10 near a = -40, 5e-5 at -1e3, 13% at -1e4, of any
sign and size beyond); with Φ = (1 + erf)/2 floored at 1e-15, m has the wrong sign from
a ≈ -7.9; and ForwardDiff's derivative of a + φ(a)/Φ(a) squares Φ, which underflows from
a ≈ -27. Here nothing large is subtracted:
- a < -5: the continued fraction m(a) = 1/(t + 2/(t + 3/(t + ⋯ + 40/t))), t = -a. It is the
  Mills ratio's continued fraction, λ(a) = t + 1/(t + 2/(t + 3/(t + ⋯))), with its leading t
  taken off analytically, so only positive terms are added; 40 levels reach full precision
  for t ≥ 5, and ForwardDiff's first and second derivatives are formed without cancellation.
- -5 ≤ a ≤ 0: a + √(2/π)/erfcx(-a/√2), in which Φ's factor e^(-a²/2) cancels analytically;
  the sum a + λ amplifies rounding by at most λ/m ≤ 28 there (about 5 bits).
- a > 0: a + φ(a)/Φ(a) with Φ(a) = 1 - erfcx(a/√2)·e^(-a²/2)/2; nothing cancels.
Relative errors against 1024-bit references on dense grids (eps = 2.2e-16):
the value is within 64 eps (1.4e-14) on [-5, 0], 1 eps below -5 and 3 eps above 0; ForwardDiff's
m' and m'' are within 4e-13 and 7e-12 on [-5, 0] and within 1e-15 below -5, and they stay
finite down to a = -1e300. At the switch a = -5 the value jumps by 3e-15 relative, m' by 4e-14
and m'' by 5e-13. Scalar and generic: Float32 (within 36 Float32 eps) and ForwardDiff Duals
work.
"""
@inline function sf_tn_mean(a)
    if a < -5
        t = -a
        p = t
        for k in 40:-1:2
            p = t + k / p
        end
        return inv(p)
    elseif a <= 0
        return a + oftype(a, 0.7978845608028654) / erfcx(-a / sqrt(oftype(a, 2)))
    else
        e = exp(-a * a / 2)
        return a + oftype(a, 0.3989422804014327) * e / (1 - erfcx(a / sqrt(oftype(a, 2))) * e / 2)
    end
end

"""
    sf_tn_logmgf(a, s)

log E(e^(-s·v)) for v ~ N⁺(a, 1) and s ≥ 0:

    log E(e^(-s·v)) = -s·a + s²/2 + log Φ(a - s) - log Φ(a) = h(a - s) - h(a),

with h(x) = log Φ(x) + x²/2. For u ~ N⁺(μ, σ²), E(e^(-c·u)) = exp(sf_tn_logmgf(μ/σ, c·σ)).
The closed-form BC is therefore exp(sf_tn_logmgf(μₛ/σₛ, σₛ)), and exp(sf_tn_logmgf(μₛ/σₛ,
h·σₛ)) in the panel models that scale u by h (-s·a is the -μₛ term of the usual closed form,
since a·σₛ = μₛ).

Added up directly, the four terms on the left are each of order a²/2 as a → -∞ while the sum
is small, so the sum carries an absolute error of about eps·a² (4e-13 near a = -40, 6e-5 at
-1e6, 8e-3 at -1e7) and BC can exceed 1. On the right the a²/2 parts cancel analytically.
When a and a - s are both below -40, h(x) = -log(-x) - log√(2π) + log S(x)
with S(x) = 1 - 1/x² + 3/x⁴ - 15/x⁶ + 105/x⁸ - 945/x¹⁰ + 10395/x¹² (the series of
`sf_normlogcdf_lowtail` with one more term; the first omitted one is about 5e-18 at -40), so

    h(a - s) - h(a) = -log1p(s/(-a)) + log S(a - s) - log S(a),

in which nothing large is differenced; for s ≥ 0 it is never positive (20,000 random points
checked), so BC = e^(...) lies in (0, 1]. When a > 0 and a - s ≥ 0, the four terms are added
directly, as -s·(a - s/2) + (log Φ(a - s) - log Φ(a)): both parts are ≤ 0 there, so nothing
cancels, while h(a - s) - h(a) would difference two numbers of size a²/2. Elsewhere h is taken
from erfcx. Against 1024-bit references BC = e^(...) is within 1.3e-16 absolute below -40 and
1.5e-15 absolute on (-40, 5); for a from 5 to 1e7 with a - s ≥ 0 the value (log BC) is within
1 eps relative, so BC is within 2e-16 relative where BC ≥ e^-1.

Domain: real (floating-point or ForwardDiff Dual) a and s ≥ 0. The package calls it only
with a < -40; the vectorised call sites evaluate it at every observation but use its value
only there.
"""
@inline function sf_tn_logmgf(a, s)
    b = a - s
    if a < -40 && b < -40
        return -log1p(s / -a) + (_sf_logS_normcdf(b) - _sf_logS_normcdf(a))
    elseif a > 0 && b >= 0
        return -s * (a - s / 2) + (_sf_normlogcdf_upper(b) - _sf_normlogcdf_upper(a))
    else
        return _sf_normlogcdf_halfsq(b) - _sf_normlogcdf_halfsq(a)
    end
end

# log Φ(x) for x ≥ 0 (see sf_tn_logmgf)
@inline _sf_normlogcdf_upper(x) = log1p(-erfcx(x / sqrt(oftype(x, 2))) * exp(-x * x / 2) / 2)

# log S(x) for x < -40, where Φ(x) = φ(x)·S(x)/(-x) (see sf_tn_logmgf)
@inline function _sf_logS_normcdf(x)
    y = inv(x * x)
    return log1p(-y * (1 - 3y * (1 - 5y * (1 - 7y * (1 - 9y * (1 - 11y))))))
end

# h(x) = log Φ(x) + x²/2 (see sf_tn_logmgf)
@inline function _sf_normlogcdf_halfsq(x)
    if x < -40
        return -log(-x) - oftype(x, 0.9189385332046728) + _sf_logS_normcdf(x)
    elseif x <= 0
        return log(erfcx(-x / sqrt(oftype(x, 2))) / 2)
    else
        return log1p(-erfcx(x / sqrt(oftype(x, 2))) * exp(-x * x / 2) / 2) + x * x / 2
    end
end

# ---------------------------------------------------------------------------------------------
# The Gumbel copula's log density in logarithms (MCI and MSLE): log(-log F) of the noise CDF,
# taken from the complementary CDF where F is near 1, and log S as a log-sum-exp.
# ---------------------------------------------------------------------------------------------
export sf_logaddexp, sf_log_neglogcdf_normal, sf_log_neglogcdf_laplace

"""
    sf_logaddexp(a, b)

log(exp(a) + exp(b)) without overflow: the larger argument plus log1p(exp(smaller - larger)).
Written as a conditional rather than as max(a, b) + log1p(exp(-|a - b|)), so that it is -Inf
(not NaN) when both are -Inf and its ForwardDiff derivative at a == b is 1/2 in each argument;
it broadcasts on the GPU.
"""
@inline sf_logaddexp(a, b) = a > b ? a + log1p(exp(b - a)) : (b == -Inf ? b : b + log1p(exp(a - b)))   # ternary: GPU-safe, and ForwardDiff-safe also at a == b

# log(-log1p(-q)) from lq = log q, 0 < q < 1; for q < 1e-8 it is log q + q/2 (dropped term about 2e-17)
"""
    _sf_log_neglog1m(lq)

log(-log(1 - q)) from lq = log q, for 0 < q < 1, without forming 1 - q: log(-log1p(-q)), and
log q + q/2 for q < 1e-8, where the dropped term is about 2e-17. With q = 1 - F it gives
log(-log F) also where F is too close to 1 to be stored.
"""
@inline function _sf_log_neglog1m(lq)
    q = exp(lq)
    return q < 1e-8 ? lq + q / 2 : log(-log1p(-q))
end

# log(-log Φ(x)), Φ the standard normal CDF, with no cancellation in either tail
"""
    sf_log_neglogcdf_normal(x)

log(-log Φ(x)), Φ the standard normal CDF, finite for every finite x. For x < -1 it is formed
from `sf_normlogcdf_lowtail(x)`; otherwise from log(1 - Φ(x)) (`sf_normlogcdf_lowtail(-x)` for
x > 1, log(erfc(x/√2)/2) in between) by `_sf_log_neglog1m`, so it needs no clamp where Φ(x)
rounds to 1: at x = 50 the value is log(1 - Φ(50)) ≈ -1254.8. The Gumbel copula's noise side
under normal noise, log(-log F_v), at the standardized residual.
"""
@inline function sf_log_neglogcdf_normal(x)
    x < -1 && return log(-sf_normlogcdf_lowtail(x))
    lq = x > 1 ? sf_normlogcdf_lowtail(-x) : log(erfc(x / sqrt(oftype(x, 2))) / 2)   # log(1 - Φ(x))
    return _sf_log_neglog1m(lq)
end

# log(-log F(x)), F the standard Laplace CDF: e^x/2 for x < 0, 1 - e^(-x)/2 for x ≥ 0
"""
    sf_log_neglogcdf_laplace(x)

log(-log F(x)), F the standard Laplace CDF, finite for every finite x: log(log 2 - x) for x < 0,
and for x ≥ 0 log(-log1p(-e^(-x)/2)), from log(1 - F(x)) = -log 2 - x by `_sf_log_neglog1m`.
The Gumbel copula's noise side under Laplace noise, at v/b.
"""
@inline sf_log_neglogcdf_laplace(x) =
    x < 0 ? log(log(oftype(x, 2)) - x) : _sf_log_neglog1m(-log(oftype(x, 2)) - x)

end # module SFNumerics
