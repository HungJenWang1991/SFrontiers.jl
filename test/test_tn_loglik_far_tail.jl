# Regression tests: the cross-sectional truncated-normal likelihood without its Gaussian bulk.
#
# The cross-sectional truncated-normal likelihoods LL_T(Trun) and LL_T(Trun_Scale) contain
# log φ((μ+ϵ)/σ) + log Φ(a₁) - log Φ(a₀), a₀ = μ/σᵤ, a₁ = μ*/σ*. An earlier form took the Gaussian
# bulk out only when BOTH arguments were below -40, and even then an O((μ/σ)²) term cancelled
# against log φ. Near the exponential limit (μ = -K, σᵤ² = K: a₀ = -√K, a₁ moderate) the two
# log Φ were differenced whole. On the 200 observations below, the log-likelihood was off by
# 6.6e-7 / 5.8e-3 / 284 nats at K = 1e8 / 1e12 / 1e16, and by 8.1e-7 / 3.3e-3 / 45 nats at
# μ = -1e4 / -1e6 / -1e8 with σᵤ² = σᵥ² = 1; gradients and Hessians were off in proportion.
# With (μ+ϵ)²/σ² = ϵ²/σᵥ² + a₀² - a₁² the likelihoods are now written, whenever
# min(a₁, a₀) < -40, as -½log σ² - log√(2π) - ½ϵ²/σᵥ² + h(a₁) - h(a₀) with h(a) = log Φ(a) + a²/2
# if a₁ <= 0, and with ½a₁² - ½ϵ²/σᵥ² written as wμ*/σᵤ² - ½d² (w = μ + σᵤϵ/(σ + σᵤ),
# d = σᵥw/(σσᵤ)) if a₁ > 0, where the h form would cancel as σᵥ → 0 (observations below the
# frontier). Every other observation evaluates the original expression, bit for bit.
# The value, derivative and switch-position tests fail on the code before the fix.

using Test
using SFrontiers
using ForwardDiff
using StatsFuns
using SpecialFunctions
using Random
using LinearAlgebra

const _SF26 = SFrontiers.MLE_Backend.SFmle

# 512-bit log Φ: erfc down to -20, the Mills-ratio continued fraction below (300 terms are
# enough for 512 bits from x = 20 on)
function _b26_lPhi(a::BigFloat)
    a >= -20 && return log(erfc(-a / sqrt(BigFloat(2))) / 2)
    x = -a; r = x
    for k in 300:-1:1
        r = x + k / r
    end
    return -a^2 / 2 - log(sqrt(2 * BigFloat(pi))) - log(r)
end
# one observation's log-likelihood in its original form, in BigFloat
function _b26_llobs(μ, su2, sv2, ϵ)
    s2 = su2 + sv2
    μs = (sv2 * μ - su2 * ϵ) / s2
    ss = sqrt(sv2 * su2 / s2)
    return -log(s2) / 2 - (μ + ϵ)^2 / (2s2) - log(sqrt(2 * BigFloat(pi))) + _b26_lPhi(μs / ss) - _b26_lPhi(μ / sqrt(su2))
end
# 512-bit central differences (gradient, Hessian) of a BigFloat function
function _b26_grad(f, θ)
    θb = BigFloat.(θ); k = length(θb); g = zeros(BigFloat, k)
    for j in 1:k
        h = big"1e-35" * max(1, abs(θb[j])); e = zeros(BigFloat, k); e[j] = h
        g[j] = (f(θb .+ e) - f(θb .- e)) / (2h)
    end
    return g
end
function _b26_hess(f, θ)
    θb = BigFloat.(θ); k = length(θb); H = zeros(BigFloat, k, k); fc = f(θb)
    hs = [big"1e-30" * max(1, abs(θb[j])) for j in 1:k]
    E(j) = (e = zeros(BigFloat, k); e[j] = hs[j]; e)
    for j in 1:k
        H[j, j] = (f(θb .+ E(j)) - 2 * fc + f(θb .- E(j))) / hs[j]^2
        for l in j+1:k
            H[j, l] = H[l, j] = (f(θb .+ E(j) .+ E(l)) - f(θb .+ E(j) .- E(l)) -
                                 f(θb .- E(j) .+ E(l)) + f(θb .- E(j) .- E(l))) / (4 * hs[j] * hs[l])
        end
    end
    return H
end

# One observation of LL_T(Trun) with θ = [β0, μ, ln σᵤ², ln σᵥ²], y = 0, x = 1, so ϵ = -β0;
# the package's negative log-likelihood, and the three per-observation forms written with
# the operations LL_T uses (original expression, the h form, and the rewrite for a₁ > 0).
const _po26_1 = (begx = 1, endx = 1, begz = 2, endz = 2, begw = 3, endw = 3, begv = 4, endv = 4)
_nll26_1(θ) = _SF26.LL_T(_SF26.Trun, zeros(1, 1), ones(1, 1), ones(1, 1), (), ones(1, 1), ones(1, 1),
                         1, 1, _po26_1, θ, nothing, nothing)
function _forms26_1(θ)
    μ = θ[2]; σᵤ² = exp(θ[3]); σᵤ = sqrt(σᵤ²); σᵥ² = exp(θ[4]); σ² = σᵤ² + σᵥ²; ϵ = -θ[1]
    μₛ = (σᵥ² * μ - σᵤ² * ϵ) / σ²; σₛ² = (σᵥ² * σᵤ²) / σ²
    a₁ = μₛ / sqrt(σₛ²); a₀ = μ / σᵤ
    h = _SF26._normlogcdf_halfsq
    orig = (- 0.5 * log(σ²) + normlogpdf((μ + ϵ) / sqrt(σ²))
            + _SF26._stable_normlogcdf(a₁) - _SF26._stable_normlogcdf(a₀))
    P = - 0.5 * log(σ²) - log(StatsFuns.sqrt2π) - 0.5 * ϵ^2 / σᵥ² + h(a₁) - h(a₀)
    s = sqrt(σ²); w = μ + σᵤ * ϵ / (s + σᵤ); d = sqrt(σᵥ²) * w / (s * σᵤ)
    R = (- 0.5 * log(σ²) - log(StatsFuns.sqrt2π) + normlogcdf(a₁)
         + w * μₛ / σᵤ² - 0.5 * d^2 - h(a₀))
    return (orig = orig, P = P, R = R, a₁ = a₁, a₀ = a₀)
end
_ref26_1(θ) = _b26_llobs(θ[2], exp(θ[3]), exp(θ[4]), -θ[1])      # θ::Vector{BigFloat}
# β0 that puts a₁ at a target: a₁ = (σᵥ²μ - σᵤ²ϵ)/(σσᵥσᵤ)
_b0for26(a1, μ, su2, sv2) = -(sv2 * μ - a1 * sqrt(su2 + sv2) * sqrt(sv2) * sqrt(su2)) / su2
# bisection to two adjacent Float64 values, f(lo) true and f(hi) false
function _bis26(f, lo, hi)
    while nextfloat(lo) < hi
        m = (lo + hi) / 2
        (m == lo || m == hi) && break
        f(m) ? (lo = m) : (hi = m)
    end
    return lo, hi
end

@testset "cross-sectional truncated-normal likelihood without its Gaussian bulk" begin
    setprecision(BigFloat, 512) do

    rng = MersenneTwister(20260929)
    n = 200
    x = randn(rng, n); z = randn(rng, n)
    u = abs.(randn(rng, n)) .* exp.(0.5 .* z)
    y = 0.5 .+ 0.5 .* x .+ randn(rng, n) .- u
    Y = reshape(y, n, 1); X = hcat(ones(n), x); Z = hcat(ones(n), z); V = ones(n, 1)
    O = ones(n, 1); Q = reshape(copy(z), n, 1)

    # Trun: θ = [β0, β1, a, b, c, d, e]; μ = a + b z, ln σᵤ² = c + d z, ln σᵥ² = e
    # Trun_Scale: θ = [β0, β1, μ*, τ, c*, e]; h = exp(τ z), μ = μ*·h, σᵤ² = exp(c*)·h²
    # Each on the first m observations: the package's negative log-likelihood and the
    # 512-bit log-likelihood.
    poT = (begx = 1, endx = 2, begz = 3, endz = 4, begw = 5, endw = 6, begv = 7, endv = 7)
    poS = (begx = 1, endx = 2, begz = 3, endz = 3, begq = 4, endq = 4, begw = 5, endw = 5, begv = 6, endv = 6)
    function models(m)
        Ym, Xm, Zm, Vm, Om, Qm = Y[1:m, :], X[1:m, :], Z[1:m, :], V[1:m, :], O[1:m, :], Q[1:m, :]
        nT(p) = _SF26.LL_T(_SF26.Trun, Ym, Xm, Zm, (), Zm, Vm, 1, m, poT, p, nothing, nothing)
        nS(p) = _SF26.LL_T(_SF26.Trun_Scale, Ym, Xm, Om, Qm, Om, Vm, 1, m, poS, p, nothing, nothing)
        function lT(θ)
            β0, β1, a, b, c, d, e = BigFloat.(θ); s = BigFloat(0)
            for i in 1:m
                s += _b26_llobs(a + b * z[i], exp(c + d * z[i]), exp(e), y[i] - β0 - β1 * x[i])
            end
            return s
        end
        function lS(θ)
            β0, β1, μs, τ, c, e = BigFloat.(θ); s = BigFloat(0)
            for i in 1:m
                h = exp(τ * z[i])
                s += _b26_llobs(μs * h, exp(c) * h^2, exp(e), y[i] - β0 - β1 * x[i])
            end
            return s
        end
        return nT, lT, nS, lS
    end
    nllT, llT, nllS, llS = models(n)
    ridgeT(K) = [0.5, 0.5, -K, 0.0, log(K), 0.0, 0.0]
    ridgeS(K) = [0.5, 0.5, -K, 0.3, log(K), 0.0]
    muT(m) = [0.5, 0.5, m, 0.0, 0.0, 0.0, 0.0]
    muS(m) = [0.5, 0.5, m, 0.3, 0.0, 0.0]

    @testset "values against 512 bits" begin
        # pre-fix errors (Trun): ridge 6.6e-7 / 5.8e-3 / 284; μ sweep 8.1e-7 / 3.3e-3 / 45.4;
        # arguments between -1000 and -40 (ridge K = 9e5, a₀ = -949; μ = -900): 1.9e-8 / 8.9e-9
        for θ in (ridgeT(9e5), ridgeT(1e8), ridgeT(1e12), ridgeT(1e16), muT(-900.0), muT(-1e4), muT(-1e6), muT(-1e8))
            @test abs(-nllT(θ) - llT(θ)) <= 1e-11
        end
        # pre-fix errors (Trun_Scale): ridge 7.4e-8 / 1.2e-3 / 64; μ* sweep 1.4e-3 / 16
        for θ in (ridgeS(1e8), ridgeS(1e12), ridgeS(1e16), muS(-1e6), muS(-1e8))
            @test abs(-nllS(θ) - llS(θ)) <= 1e-11
        end
    end

    @testset "below the frontier as σᵥ → 0 (a₀ < -40 with a₁ > 0)" begin
        # Intercept raised so that every ϵ <= -1e-3; on the ridge μ = -K, σᵤ² = K every
        # observation then has a₀ = -√K and a₁ > 0, and its log-likelihood stays O(1) as σᵥ → 0.
        # The h form cancels ½a₁² against ½ϵ²/σᵥ² there (the first version of this fix:
        # 0.22 nats at ln σᵥ² = -30, +1.25e21 at -80, NaN at σᵥ² = 0); absolute tolerances.
        b0c = maximum(y .- 0.5 .* x) + 1e-3
        for K in (1e4, 1e12), e in (-30.0, -80.0)
            θ = [b0c, 0.5, -K, 0.0, log(K), 0.0, e]
            @test abs(-nllT(θ) - llT(θ)) <= 1e-10
            gr = _b26_grad(llT, θ)
            @test maximum(abs.(big.(-ForwardDiff.gradient(nllT, θ)) .- gr)) <= 1e-10 * max(1.0, Float64(maximum(abs.(gr))))
        end
        θ = [b0c, 0.5, -1e12, 0.3, log(1e12), -30.0]
        @test abs(-nllS(θ) - llS(θ)) <= 1e-10
        gr = _b26_grad(llS, θ)
        @test maximum(abs.(big.(-ForwardDiff.gradient(nllS, θ)) .- gr)) <= 1e-10 * max(1.0, Float64(maximum(abs.(gr))))
        # σᵥ² = exp(-745.5) = 0 in Float64: the finite limit (the reference uses exp(-745.5) in
        # 512 bits, which differs from the limit by far less than the tolerance)
        for K in (1e4, 1e12)
            θ = [b0c, 0.5, -K, 0.0, log(K), 0.0, -745.5]
            @test abs(-nllT(θ) - llT(θ)) <= 1e-10
            θs = [b0c, 0.5, -K, 0.3, log(K), -745.5]
            @test abs(-nllS(θs) - llS(θs)) <= 1e-10
        end
        # one observation, μ = -1e12, σᵤ² = 0.8e12, ln σᵥ² = -40, a₁ = 20 (the first version's
        # Hessian was off by 64 here, where max|H| is about 1.2)
        θ1 = [_b0for26(20.0, -1e12, 0.8e12, exp(-40.0)), -1e12, log(0.8e12), -40.0]
        @test _forms26_1(θ1).a₀ < -40 && 19 < _forms26_1(θ1).a₁ < 21
        @test abs(-_nll26_1(θ1) - _ref26_1(big.(θ1))) <= 1e-12
        gr = _b26_grad(_ref26_1, θ1); Hr = _b26_hess(_ref26_1, θ1)
        @test maximum(abs.(big.(-ForwardDiff.gradient(_nll26_1, θ1)) .- gr)) <= 1e-10 * max(1.0, Float64(maximum(abs.(gr))))
        @test maximum(abs.(big.(-ForwardDiff.hessian(_nll26_1, θ1)) .- Hr)) <= 1e-10 * max(1.0, Float64(maximum(abs.(Hr))))
    end

    @testset "gradient and Hessian against 512-bit differences" begin
        # on the first 60 observations (to keep the 512-bit differences quick); pre-fix
        # gradient errors 1.3e-4 (Trun, K = 1e12; |g| = 5.0), 3.4e-3 (Trun_Scale, K = 1e12)
        # and 2.2e-3 (Trun, μ = -1e6), Hessian errors 1.0e-2 and 1.9e-2 (|H| = 35); after the
        # fix at most about 1e-13
        nT, lT, nS, lS = models(60)
        for (nll, ll, θ, hess) in ((nT, lT, ridgeT(1e12), true), (nS, lS, ridgeS(1e12), true),
                                   (nT, lT, muT(-1e6), false))
            g = -ForwardDiff.gradient(nll, θ); gr = _b26_grad(ll, θ)
            @test maximum(abs.(big.(g) .- gr)) <= 1e-12 * max(1.0, Float64(maximum(abs.(gr))))
            if hess
                H = -ForwardDiff.hessian(nll, θ); Hr = _b26_hess(ll, θ)
                @test maximum(abs.(big.(H) .- Hr)) <= 1e-11 * max(1.0, Float64(maximum(abs.(Hr))))
            end
        end
    end

    @testset "the switch sits at min(a₁, a₀) = -40 and at a₁ = 0" begin
        # Exactness on single observations: the package value is the form the documented
        # condition selects, bit for bit. A switch at -50 or on a₀ alone would be numerically
        # almost harmless, so only these asserts guard the documented condition.
        bis = _bis26
        # a₀ crosses -40 with a₁ ≈ -0.2 (ϵ = -0.8, μ = -1600): h form below, original above
        θa(t) = [0.8, -1600.0, t, 0.0]
        lo, hi = bis(t -> _forms26_1(θa(t)).a₀ < -40, log(1600.0) - 0.01, log(1600.0) + 0.01)
        Fl, Fh = _forms26_1(θa(lo)), _forms26_1(θa(hi))
        @test Fl.a₀ < -40 && Fh.a₀ >= -40 && nextfloat(lo) == hi && -1 < Fl.a₁ < 0
        @test _nll26_1(θa(lo)) === -Fl.P
        @test _nll26_1(θa(hi)) === -Fh.orig
        # a₁ < -40 <= a₀ (μ = 0.5, σᵤ² = 1, σᵥ² = 1e-3, ϵ = 4): h form, not the original
        θb = [-4.0, 0.5, 0.0, log(1e-3)]
        Fb = _forms26_1(θb)
        @test Fb.a₁ < -40 && Fb.a₀ >= -40
        @test _nll26_1(θb) === -Fb.P
        @test ForwardDiff.gradient(_nll26_1, θb) == ForwardDiff.gradient(p -> -_forms26_1(p).P, θb)
        # a₀ < -40 with a₁ > 0 (μ = -1e4, σᵤ² = 1e4, ln σᵥ² = -30, ϵ = -0.5): the rewrite
        θc = [0.5, -1e4, log(1e4), -30.0]
        Fc = _forms26_1(θc)
        @test Fc.a₀ < -40 && Fc.a₁ > 0
        @test _nll26_1(θc) === -Fc.R
        @test ForwardDiff.gradient(_nll26_1, θc) == ForwardDiff.gradient(p -> -_forms26_1(p).R, θc)
        # a₁ crosses 0 with a₀ = -100 (μ = -1e4, σᵤ² = 1e4, σᵥ² = 1; a₁ = 0 at ϵ = -1): h form
        # at a₁ <= 0, rewrite above; the two agree to rounding
        θd(b) = [b, -1e4, log(1e4), 0.0]
        lo, hi = bis(b -> _forms26_1(θd(b)).a₁ <= 0, 0.99, 1.01)
        Fl, Fh = _forms26_1(θd(lo)), _forms26_1(θd(hi))
        @test Fl.a₁ <= 0 && Fh.a₁ > 0 && nextfloat(lo) == hi
        @test _nll26_1(θd(lo)) === -Fl.P
        @test _nll26_1(θd(hi)) === -Fh.R
        @test abs(_nll26_1(θd(lo)) - _nll26_1(θd(hi))) <= 1e-14
        @test maximum(abs.(ForwardDiff.gradient(_nll26_1, θd(lo)) .- ForwardDiff.gradient(_nll26_1, θd(hi)))) <= 1e-13
        @test maximum(abs.(ForwardDiff.hessian(_nll26_1, θd(lo)) .- ForwardDiff.hessian(_nll26_1, θd(hi)))) <= 1e-12
        # next to the split, where the two forms differ in their last bits (μ = -1e8, σᵤ² = 1e8,
        # σᵥ² = 1): the h form at a₁ = -0.1, the rewrite at a₁ = +0.1
        for (a1t, form) in ((-0.1, :P), (0.1, :R))
            θe = [_b0for26(a1t, -1e8, 1e8, 1.0), -1e8, log(1e8), 0.0]
            Fe = _forms26_1(θe)
            @test Fe.a₀ < -40 && abs(Fe.a₁ - a1t) < 1e-6 && Fe.P !== Fe.R
            @test _nll26_1(θe) === -getfield(Fe, form)
            @test ForwardDiff.gradient(_nll26_1, θe) == ForwardDiff.gradient(p -> -getfield(_forms26_1(p), form), θe)
        end
    end

    @testset "no jump where a₀ crosses -40" begin
        # with a₁ moderate: one observation of the 7-parameter model (ϵ = -0.8), μ = -1600:
        # a₀ = -40 at ln σᵤ² = log(1600), a₁ ≈ -0.2
        Y1 = fill(-0.15, 1, 1); X1 = [1.0 0.3]; Z1 = [1.0 0.0]; V1 = ones(1, 1)
        f(p) = _SF26.LL_T(_SF26.Trun, Y1, X1, Z1, (), Z1, V1, 1, 1, poT, p, nothing, nothing)
        θt(t) = [0.5, 0.5, -1600.0, 0.0, t, 0.0, 0.0]
        a0(t) = -1600.0 / sqrt(exp(t))
        lo, hi = log(1600.0) - 0.01, log(1600.0) + 0.01     # a₀ < -40 at lo, > -40 at hi
        while nextfloat(lo) < hi
            m = (lo + hi) / 2
            (m == lo || m == hi) && break
            a0(m) < -40 ? (lo = m) : (hi = m)
        end
        @test a0(lo) < -40 && a0(hi) >= -40 && nextfloat(lo) == hi
        @test abs(f(θt(lo)) - f(θt(hi))) <= 1e-12
        @test maximum(abs.(ForwardDiff.gradient(f, θt(lo)) .- ForwardDiff.gradient(f, θt(hi)))) <= 1e-11
        @test maximum(abs.(ForwardDiff.hessian(f, θt(lo)) .- ForwardDiff.hessian(f, θt(hi)))) <= 1e-8
        # with a₁ ≈ 5e3 (σᵥ² = 1e-6, ϵ = -5): the rewrite below -40, the original above (the
        # first version of this fix jumped by 2.6e-9 in value here)
        θu(t) = [5.0, -1600.0, t, log(1e-6)]
        lo, hi = log(1600.0) - 0.01, log(1600.0) + 0.01
        while nextfloat(lo) < hi
            m = (lo + hi) / 2
            (m == lo || m == hi) && break
            _forms26_1(θu(m)).a₀ < -40 ? (lo = m) : (hi = m)
        end
        Fl, Fh = _forms26_1(θu(lo)), _forms26_1(θu(hi))
        @test Fl.a₀ < -40 && Fh.a₀ >= -40 && nextfloat(lo) == hi && 4e3 < Fl.a₁ < 6e3
        @test abs(_nll26_1(θu(lo)) - _nll26_1(θu(hi))) <= 1e-12
        @test maximum(abs.(ForwardDiff.gradient(_nll26_1, θu(lo)) .- ForwardDiff.gradient(_nll26_1, θu(hi)))) <= 1e-11
    end

    @testset "unchanged, bit for bit, where min(a₁, a₀) >= -40 at every observation" begin
        # the original expression, in the same order of operations as LL_T
        function oldT(θ)
            μ = Z * θ[3:4]; σᵤ² = exp.(Z * θ[5:6]); σᵤ = sqrt.(σᵤ²); σᵥ² = exp.(V * θ[7:7])
            σ² = σᵤ² + σᵥ²; ϵ = 1 * (Y - X * θ[1:2])
            μₛ = (σᵥ² .* μ - σᵤ² .* ϵ) ./ σ²; σₛ² = (σᵥ² .* σᵤ²) ./ σ²
            ll = [(-0.5 * log(σ²[i]) + normlogpdf((μ[i] + ϵ[i]) / sqrt(σ²[i]))
                   + _SF26._stable_normlogcdf((μₛ[i]) / sqrt(σₛ²[i]))
                   - _SF26._stable_normlogcdf((μ[i]) / σᵤ[i])) for i in 1:n]
            return -sum(ll), minimum(min.(μₛ ./ sqrt.(σₛ²), μ ./ σᵤ))
        end
        function oldS(θ)
            hscale = exp.(Q * θ[4:4]); μ = (O * θ[3:3]) .* hscale; σᵤ² = exp.(O * θ[5:5]) .* hscale .^ 2
            σᵤ = sqrt.(σᵤ²); σᵥ² = exp.(V * θ[6:6]); σ² = σᵤ² + σᵥ²; ϵ = 1 * (Y - X * θ[1:2])
            μₛ = (σᵥ² .* μ - σᵤ² .* ϵ) ./ σ²; σₛ² = (σᵥ² .* σᵤ²) ./ σ²
            ll = [(-0.5 * log(σ²[i]) + normlogpdf((μ[i] + ϵ[i]) / sqrt(σ²[i]))
                   + _SF26._stable_normlogcdf((μₛ[i]) / sqrt(σₛ²[i]))
                   - _SF26._stable_normlogcdf((μ[i]) / σᵤ[i])) for i in 1:n]
            return -sum(ll), minimum(min.(μₛ ./ sqrt.(σₛ²), μ ./ σᵤ))
        end
        for θ in ([0.4, 0.6, 1.0, 1.5, -1.0, 0.5, -0.3],
                  [0.5, 0.5, -400.0, 0.0, 2log(400 / 39.9), 0.0, 0.0],   # ridge with a₀ = -39.9
                  [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, log(1e-2)])             # σᵥ = 0.1: a₁ above +40 at some observations
            fo = p -> oldT(p)[1]
            @test oldT(θ)[2] >= -40
            @test nllT(θ) === fo(θ)
            @test ForwardDiff.gradient(nllT, θ) == ForwardDiff.gradient(fo, θ)
            @test ForwardDiff.hessian(nllT, θ) == ForwardDiff.hessian(fo, θ)
        end
        for θ in ([0.4, 0.6, -1.0, 0.5, -1.0, -0.3], [0.5, 0.5, -400.0, 0.0, 2log(400 / 39.9), 0.0])
            fo = p -> oldS(p)[1]
            @test oldS(θ)[2] >= -40
            @test nllS(θ) === fo(θ)
            @test ForwardDiff.gradient(nllS, θ) == ForwardDiff.gradient(fo, θ)
            @test ForwardDiff.hessian(nllS, θ) == ForwardDiff.hessian(fo, θ)
        end
    end

    end   # setprecision
end
