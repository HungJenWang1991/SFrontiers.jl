# Regression tests: the Gaussian bulk taken out of the truncated-normal likelihoods.
#
# The truncated-normal likelihoods difference log Φ(a₁) - log Φ(a₀), a₀ = μ/σᵤ, a₁ = μ*/σ*.
# As σᵤ → 0 with μ < 0 both arguments run off to -∞, each log Φ is about -a²/2, and their
# O(1) difference was formed from two Float64s: the value carried eps·a²/2 of error and the
# gradient a floor of eps·|a|·|∂a/∂θ|. On a ridge a Newton fit walks down this was 0.04
# nats at ln σᵤ² = -28.9 (2.45 on a real fit), 2e11 at -60 and 1e72 at -200; one fit's
# start reported +2.5e28 for a true -3.4e14. The panel likelihoods also carry ½(a₁² - a₀²)
# explicitly, so there the bulk cancels on the +∞ side too. The likelihoods now take the
# Gaussian bulk out analytically beyond ±40 and are unchanged, bit for bit, elsewhere.
# These tests fail on the code before the fix.

using Test
using SFrontiers
using ForwardDiff
using StatsFuns
using SpecialFunctions
using Random
using LinearAlgebra

const _SF19 = SFrontiers.MLE_Backend.SFmle

# 512-bit log Φ: erfc down to -30, the Mills-ratio continued fraction below
function _b19_lPhi(a::BigFloat)
    a >= -30 && return log(erfc(-a / sqrt(BigFloat(2))) / 2)
    x = -a; r = x
    for k in 2000:-1:1
        r = x + k / r
    end
    return -a^2 / 2 - log(sqrt(2 * BigFloat(pi))) - log(r)
end

@testset "Gaussian bulk taken out of the truncated-normal likelihoods" begin
    setprecision(BigFloat, 512) do

    @testset "helpers" begin
        h = _SF19._normlogcdf_halfsq
        for a in (-1e8, -6.65e4, -1e3, -45.0, -40.0, -39.9, -10.0, -1.0, 0.0, 2.0, 30.0)
            ref = _b19_lPhi(BigFloat(a)) + BigFloat(a)^2 / 2
            @test abs(h(a) - ref) <= 4e-15 * max(1.0, abs(Float64(ref)))
        end
        @test abs(h(prevfloat(-40.0)) - h(-40.0)) <= 1e-14
        # the shift equals a₁ - a₀ (512-bit) without being formed as a difference
        for (μ, su2, sv2, S, E) in ((1.0, 0.3, 1.0, 1.0, 0.5),)
            B = BigFloat
            D = B(sv2) + B(su2) * B(S)
            a0 = B(μ) / sqrt(B(su2))
            a1 = ((B(sv2) * B(μ) - B(su2) * B(E)) / D) / sqrt(B(sv2) * B(su2) / D)
            δ = _SF19._tn_shift(μ, su2, sv2, S, E)
            @test abs(δ - (a1 - a0)) <= 1e-14 * max(1.0, abs(Float64(a1 - a0)))
        end
    end

    # the panel data of the PanDecay set are drawn from this stream after these cross-sectional draws
    rng = MersenneTwister(20260929)
    n = 200
    x = randn(rng, n); z = randn(rng, n)
    u = abs.(randn(rng, n)) .* exp.(0.5 .* z)
    y = 0.5 .+ 0.5 .* x .+ randn(rng, n) .- u

    @testset "LL_T(PanDecay) against a 512-bit reference" begin
        N = 30; T = 5
        ivar = repeat(1:N, inner = T); nn = N * T
        idt = _SF19.get_rowIDT(ivar)
        xp = randn(rng, nn); Xp = hcat(ones(nn), xp); Zp = ones(nn, 1)
        Qp = reshape(collect(repeat(0:T-1, N)) .- 2.0, nn, 1)
        yp = 0.5 .+ 0.5 .* xp .+ 0.3 .* randn(rng, nn) .- repeat(abs.(randn(rng, N)), inner = T)
        pop = (begx = 1, endx = 2, begz = 3, endz = 3, begq = 4, endq = 4, begw = 5, endw = 5, begv = 6, endv = 6)
        nllP(p) = _SF19.LL_T(_SF19.PanDecay, yp, Xp, Zp, Qp, ones(nn, 1), ones(nn, 1), 1, nn, pop, p, idt, nothing)
        function llPref(θ)
            β0, β1, μ, τ, c, g = BigFloat.(θ)
            su2 = exp(c); sv2 = exp(g); s = BigFloat(0)
            for i in 1:N
                ind = idt[i, 1]; Ti = idt[i, 2]
                ε = [BigFloat(yp[k]) - β0 - β1 * xp[k] for k in ind]
                G = [exp(τ * Qp[k]) for k in ind]
                D = sv2 + su2 * sum(G .^ 2)
                μs = (μ * sv2 - sum(G .* ε) * su2) / D; ss2 = sv2 * su2 / D
                quad = sum(ε .^ 2) / sv2 + μ^2 / su2 - μs^2 / ss2
                s += (-Ti * log(2 * BigFloat(pi)) / 2 + _b19_lPhi(μs / sqrt(ss2)) + log(ss2) / 2 - quad / 2
                      - Ti * g / 2 - c / 2 - _b19_lPhi(μ / sqrt(su2)))
            end
            return s
        end
        for θ in ([0.5, 0.5, 0.3, 0.1, -1.0, -2.0],)
            @test abs(-nllP(θ) - llPref(θ)) <= 1e-9
        end
    end

    end   # setprecision
end
