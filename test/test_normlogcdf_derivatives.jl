# Regression tests: ForwardDiff derivatives of log Φ in the MLE likelihoods.
#
# The MLE backend fits by Newton with ForwardDiff Hessians. StatsFuns.normlogcdf(z) is
# log(erfcx(-z/√2)/2) - z²/2 below z = -1, accurate as a value, but ForwardDiff
# differentiates it through DiffRules' rule erfcx'(x) = 2(x·erfcx(x) - 1/√π), which cancels:
# the second derivative of log Φ came out with relative error about eps·z² (6e-7 at
# z = -6.65e4, of order 1 at z = -1e10). In LL_T(Trun) at a σᵤ → 0 corner that error is
# multiplied by (∂z/∂θ)² and the Hessian was off by thousands; Newton stalled at its
# iteration limit next to higher-likelihood maxima (universality MC, approximation-theory
# paper, n = 1000, rep 231: all 16 attempts failed; after the fix every start converges in
# 6-16 steps). The likelihoods now call _stable_normlogcdf, which uses the asymptotic
# series below z = -40. These tests fail on the pre-fix code.

using Test
using SFrontiers
using ForwardDiff
using StatsFuns
using Random
using LinearAlgebra

const _SF18 = SFrontiers.MLE_Backend.SFmle

# m(z) = z + φ(z)/Φ(z) for z <= -45 by its asymptotic series (no cancellation; the omitted
# term is below 1e-18 relative there), so λ = m - z and d²/dz² log Φ = -λ m in Float64.
_b18_m(z) = (x = inv(z * z); (-1 + x * (2 + x * (-10 + x * (74 + x * (-706 + x * 8162))))) / z)

@testset "ForwardDiff derivatives of log Φ in the MLE likelihoods" begin

    @testset "value and first two derivatives of _stable_normlogcdf" begin
        f = _SF18._stable_normlogcdf
        for z in (-45.0, -100.0, -1e3, -1e4, -6.65e4, -1e6, -1e10)
            m = _b18_m(z); λ = m - z
            d1 = ForwardDiff.derivative(f, z)
            d2 = ForwardDiff.derivative(t -> ForwardDiff.derivative(f, t), z)
            @test abs(d1 - λ) <= 4eps(λ)
            @test abs(d2 - (-λ * m)) <= 1e-12          # pre-fix: 6.1e-7 at -6.65e4, 1.0 at -1e10
            @test abs(f(z) - normlogcdf(z)) <= 8eps(abs(normlogcdf(z)))
        end
        # above the switch it IS StatsFuns.normlogcdf
        for z in (-40.0, -39.5, -7.0, -1.0, 0.0, 3.0, 40.0)
            @test f(z) === normlogcdf(z)
        end
        # no jump at the switch beyond one ulp of the value
        @test abs(f(prevfloat(-40.0)) - normlogcdf(prevfloat(-40.0))) <= 2eps(800.0)
        @test isnan(f(NaN))
        @test f(-Inf) == -Inf
    end
end
