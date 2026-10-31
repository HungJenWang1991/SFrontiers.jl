# Regression tests: the truncated-normal quantile in the lower tail.
#
# MSLE draws the truncated-normal inefficiency through its quantile,
# u = μ + σ√2·erfcinv(erfc(y)·(1 - r)) with y = -μ/(σ√2). As μ/σ falls, erfc(y) heads for
# underflow (subnormal from μ/σ ≈ -37.5, 0 from -38.5 in Float64): the draws' derivatives were
# NaN from -37.5, and below -38.5 erfcinv(0) = Inf made every draw infinite, so the MSLE
# log-likelihood, gradient, jlms and bc were NaN; before that, μ cancelled against the erfcinv
# term. The same formula sat in the panel backend's MSLE draw and in MCI's low-level
# matrix-form jlms_bc_indices. Beyond y = 8 the draw is now σ√2·t with t from SFNumerics.sf_tn_quantile_tail;
# up to y = 8 it is the old closed form, bit for bit. These tests fail on the code before
# the fix.

using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions
using Random
using Optim

# 512-bit log erfcx: erfc up to 30, the asymptotic series beyond (MPFR's erfc underflows)
function _b22_logerfcx(x::BigFloat)
    x < 30 && return log(erfc(x)) + x^2
    s = big(1); term = big(1); z = 1 / (2 * x^2)
    for n in 1:80
        term *= -(2n - 1) * z
        s += term
    end
    return -log(x * sqrt(big(pi))) + log(s)
end
# the root t of erfc(y + t) = (1 - r)·erfc(y), and dt/dy by the implicit-function theorem
function _b22_t(y::BigFloat, r::BigFloat)
    E = -log1p(-r)
    ly = _b22_logerfcx(y)
    t = E / (y + sqrt(y^2 + E))
    for _ in 1:14                                    # Newton from the right, as in the kernel
        lyt = _b22_logerfcx(y + t)
        t -= (t * (2y + t) + ly - lyt - E) * exp(lyt) * sqrt(big(pi)) / 2
    end
    return t
end
function _b22_dtdy(y::BigFloat, r::BigFloat)
    t = _b22_t(y, r)
    L(x) = 2x - 2 / (sqrt(big(pi)) * exp(_b22_logerfcx(x)))     # erfcx'(x)/erfcx(x)
    gy = 2t + L(y) - L(y + t)
    gt = 2 / (sqrt(big(pi)) * exp(_b22_logerfcx(y + t)))
    return -gy / gt
end

@testset "the truncated-normal quantile far in the lower tail" begin
    setprecision(BigFloat, 512) do

    @testset "sf_tn_quantile_tail against a 512-bit reference" begin
        Q = SFrontiers.SFNumerics.sf_tn_quantile_tail
        # relative errors, for y below and above 28
        elo = Float64[]; ehi = Float64[]
        for y in (8.0001, 10.0, 26.5, 27.9, 28.1, 45.0, 707.1, 7.07e4, 7.07e7),
            r in (1e-3, 0.3, 0.9, 1 - 1e-12)
            tr = _b22_t(BigFloat(y), BigFloat(r))
            push!(y > 28 ? ehi : elo, Float64(abs(Q(y, r) - tr) / tr))
        end
        # below y = 28, log erfcx(y) - log erfcx(y + t) (about t/y) is a difference of two
        # logs of order 3, which costs about eps·3/E relative for small r (E = -log(1 - r))
        @test maximum(elo) <= 1e-12
        @test maximum(ehi) <= 1e-14
        @test Q(30.0, 0.0) == 0.0
        # derivatives: ForwardDiff through the four Newton steps against the implicit derivative.
        # Above y = 28 the log-erfcx difference comes from the asymptotic series; formed from
        # erfcx there, ForwardDiff's erfcx rule cancelled and dt/dy lost up to three digits.
        e1 = Float64[]
        for y in (10.0, 27.9, 28.1, 707.1, 7.07e4, 7.07e5), r in (1e-3, 0.3, 0.9)
            d = ForwardDiff.derivative(yy -> Q(yy, r), y)
            dr = _b22_dtdy(BigFloat(y), BigFloat(r))
            push!(e1, Float64(abs(d - dr)) / abs(Float64(dr)))
        end
        @test maximum(e1) <= 1e-9
        # second derivatives far out (MSLE's Newton uses ForwardDiff Hessians). Formed through
        # erfcx, the Newton step gave d²t/dy² wrong by up to 130% for y in (1e7, 1e8] and by
        # orders of magnitude beyond whenever the last residual was not exactly 0; the
        # reference is the central difference of the 512-bit implicit derivative.
        e2 = Float64[]
        for y in (2.660139e7, 4.370843e7, 3e7, 1e9), r in (1e-12, 1e-4, 0.3, 0.9)
            d2 = ForwardDiff.derivative(yy -> ForwardDiff.derivative(z -> Q(z, r), yy), y)
            yb = BigFloat(y); h = yb * big(10)^-40
            d2r = (_b22_dtdy(yb + h, BigFloat(r)) - _b22_dtdy(yb - h, BigFloat(r))) / (2h)
            push!(e2, Float64(abs(d2 - d2r)) / abs(Float64(d2r)))
        end
        @test maximum(e2) <= 1e-9
        # the switch at y = 28 is continuous to rounding
        @test abs(Q(prevfloat(28.0), 0.3) - Q(nextfloat(28.0), 0.3)) <= 1e-15
        # single precision
        is32 = true; e32 = Float64[]
        for y in (9f0, 30f0, 700f0)
            q = Q(y, 0.3f0); tr = _b22_t(BigFloat(y), BigFloat(0.3f0))
            is32 &= q isa Float32
            push!(e32, Float64(abs(q - tr) / tr))
        end
        @test is32
        @test maximum(e32) <= 1e-5
    end

    s2 = sqrt(2); is2 = inv(sqrt(2))
    closed(μ, σ, r) = max(μ + σ * s2 * erfcinv(erfc(-μ * is2 / σ) * (1 - r)), zero(μ * σ * r))
    μo = [0.3, -0.5, -2.0, -11.0]; σo = [0.5, 0.5, 0.2, 1.0]    # y = -0.42, 0.71, 7.07, 7.78
    μf = [-0.096, -0.096, -1.0]; σf = [exp(-6.0), exp(-10.0), 1e-4]   # μ/σ = -38.7, -2115, -1e4
    R = [0.001 0.3 0.9 0.999]

    @testset "MSLE and MCI draws: the closed form up to y = 8, finite and accurate beyond" begin
        for B in (SFrontiers.MSLE_Backend, SFrontiers.MCI_Backend)
            q = B.myTruncatedNormalQuantile(μ = μo, σ = σo, r = R)
            @test isequal(q, closed.(μo, σo, R))
            out = similar(q)
            B.myTruncatedNormalQuantile!(out; μ = μo, σ = σo, r = R)
            @test isequal(out, q)
            qf = B.myTruncatedNormalQuantile(μ = μf, σ = σf, r = R)
            @test all(isfinite, qf)
            ef = Float64[]
            for i in eachindex(μf), j in eachindex(R)
                y = -BigFloat(μf[i]) / (BigFloat(σf[i]) * sqrt(BigFloat(2)))
                ur = BigFloat(σf[i]) * sqrt(BigFloat(2)) * _b22_t(y, BigFloat(R[j]))
                push!(ef, Float64(abs(qf[i, j] - ur) / ur))
            end
            @test maximum(ef) <= 1e-13
        end
    end

    @testset "panel MSLE draw: the closed form up to y = 8, finite and accurate beyond" begin
        P = SFrontiers.Panel_Backend
        oldp(mu, s, r) = max(mu + s * s2 * erfcinv(erfc(-mu / (s * s2)) * (1 - r)), zero(r))
        @test all(P._panel_tn_udraw(μo[i], σo[i], s2, r) === oldp(μo[i], σo[i], r) for i in eachindex(μo), r in R)
        ef = Float64[]
        for i in eachindex(μf), r in R
            u = P._panel_tn_udraw(μf[i], σf[i], s2, r)
            y = -BigFloat(μf[i]) / (BigFloat(σf[i]) * sqrt(BigFloat(2)))
            ur = BigFloat(σf[i]) * sqrt(BigFloat(2)) * _b22_t(y, BigFloat(r))
            push!(ef, Float64(abs(u - ur) / ur))
        end
        @test maximum(ef) <= 1e-13
    end

    end   # setprecision
end
