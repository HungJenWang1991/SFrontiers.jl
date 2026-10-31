# Regression tests: the MCI and panel truncated-normal normaliser log Φ(μ/σᵤ).
#
# MCI's truncated-normal density (MCI_Backend.log_pdf_ineff! and its chunked twin) and the
# panel backend's (Panel_Backend.log_pdf_truncnormal_panel) divide by Φ(μ/σᵤ). Both formed
# log Φ as log(max(Φ, 1e-15)) with Φ = (1 + erf(a/√2))/2: `1 + erf` cancels as a = μ/σᵤ
# falls (errors of either sign, up to 0.037 per observation just above a = -7.94) and below
# a ≈ -7.94 the floor took over, so log Φ stayed at -34.5 while the truth falls like -a²/2. The
# density was then mis-normalised by about a²/2 per observation: at a = -14.25 the MCI
# log-likelihood of a 400-observation fit was -28888 against -672.4 under MLE. Below a = -1 the normaliser now comes from
# SFNumerics.sf_normlogcdf_lowtail; at a ≥ -1 it is the old expression, bit for bit.
# These tests fail on the code before the fix.

using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions
using Random
using Optim

# 512-bit log Φ: erfc down to -30, the Mills-ratio continued fraction below
function _b21_lPhi(a::BigFloat)
    a >= -30 && return log(erfc(-a / sqrt(BigFloat(2))) / 2)
    x = -a; r = x
    for k in 2000:-1:1
        r = x + k / r
    end
    return -a^2 / 2 - log(sqrt(2 * BigFloat(pi))) - log(r)
end
_b21_lphi(a::BigFloat) = -a^2 / 2 - log(sqrt(2 * BigFloat(pi)))

@testset "the MCI and panel truncated-normal normaliser log Φ(μ/σᵤ)" begin
    setprecision(BigFloat, 512) do

    @testset "sf_normlogcdf_lowtail against a 512-bit reference" begin
        f = SFrontiers.SFNumerics.sf_normlogcdf_lowtail
        for a in (-1e8, -6.65e4, -1e3, -300.0, -41.0, prevfloat(-40.0), -40.0, nextfloat(-40.0),
                  -39.9, -30.0, -14.3, -7.9, -5.0, -2.0, -1.5, prevfloat(-1.0))
            ref = _b21_lPhi(BigFloat(a))
            @test abs(f(a) - ref) <= 4e-15 * max(1.0, abs(Float64(ref)))
        end
        # the switch at -40 is continuous to rounding
        @test abs(f(prevfloat(-40.0)) - f(-40.0)) <= 1e-12
        # first and second derivatives: λ(a) = φ(a)/Φ(a) and -λ(a)(a + λ(a))
        for a in (-1e5, -1e3, -41.0, -39.0, -14.3, -3.0, -1.0001)
            b = BigFloat(a)
            λ = exp(_b21_lphi(b) - _b21_lPhi(b))
            d2ref = -λ * (b + λ)
            d1 = ForwardDiff.derivative(f, a)
            d2 = ForwardDiff.derivative(x -> ForwardDiff.derivative(f, x), a)
            @test abs(d1 - λ) <= 1e-12 * abs(Float64(λ))
            @test abs(d2 - d2ref) <= 1e-9 * abs(Float64(d2ref))
        end
        # single precision
        for a in (-60f0, -14.3f0, -3f0)
            ref = _b21_lPhi(BigFloat(a))
            @test f(a) isa Float32
            @test abs(f(a) - ref) <= 1e-6 * abs(Float64(ref))
        end
    end

    @testset "MCI: the old expression at μ/σᵤ ≥ -1, bit for bit" begin
        g = SFrontiers.MCI_Backend._mci_tn_logPhi
        Φ = SFrontiers.MCI_Backend._normcdf
        for a in (-1.0, -0.999, -0.5, 0.0, 0.7, 3.0, 9.0, 40.0)
            @test g(a) === log(max(Φ(a), 1e-15))
            @test ForwardDiff.derivative(g, a) === ForwardDiff.derivative(x -> log(max(Φ(x), 1e-15)), a)
        end
        @test g(-0.5f0) === log(max(Φ(-0.5f0), 1f-15))
        # the result type does not depend on the branch: Float64 for a Float32 argument, as the
        # old expression gave (a Union return type made CuArray broadcasts fail)
        @test Base.return_types(g, (Float32,)) == [Float64]
        @test g(-3f0) isa Float64 && g(-3f0) == SFrontiers.SFNumerics.sf_normlogcdf_lowtail(-3.0)
        for a in (-1.0001, -14.3, -1e3)
            @test g(a) === SFrontiers.SFNumerics.sf_normlogcdf_lowtail(a)
        end
    end

    @testset "panel: the truncated-normal log density" begin
        lp = SFrontiers.Panel_Backend.log_pdf_truncnormal_panel
        # the pre-fix kernel, in the same order of operations
        function lp_old(u, mu, sigma, clamp_lo)
            s = max(sigma, clamp_lo)
            z = (u - mu) / s
            lΦ = log(max(oftype(z, 0.5) * (1 + erf(mu / (s * sqrt(oftype(z, 2))))), clamp_lo))
            return oftype(z, -0.5) * log(oftype(z, 2π)) - log(s) - oftype(z, 0.5) * z^2 - lΦ
        end
        for (u, mu, s) in ((0.3, 0.2, 0.5), (0.01, -0.5, 0.5), (1.0, 2.0, 0.1), (0.2, -0.3, 0.3))
            @test lp(u, mu, s, 1e-15) === lp_old(u, mu, s, 1e-15)
        end
        # far in the lower tail, against 512 bits
        # (the density's own terms -z²/2 and -log Φ(μ/σ) are each about a²/2 and cancel; the
        # tolerance is relative to their size, 2.5e6 at μ/σ = -2222)
        for (u, mu, s) in ((1e-3, -0.1, 0.007), (1e-4, -0.1, 1e-3), (1e-6, -0.1, 4.5e-5), (0.05, -2.0, 0.5))
            B = BigFloat
            zb = (B(u) - B(mu)) / B(s); lΦ = _b21_lPhi(B(mu) / B(s))
            ref = -log(sqrt(2 * B(pi))) - log(B(s)) - zb^2 / 2 - lΦ
            @test abs(lp(u, mu, s, 1e-15) - ref) <= 4eps() * (Float64(zb^2 / 2) + abs(Float64(lΦ)) + 1)
        end
    end

    end   # setprecision
end
