# Regression tests: the analytic Exponential log-likelihood as σᵤ → 0.
#
# The analytic Normal-Exponential log density is
#     log f(ϵ) = -log σᵤ + log Φ(a) + ϵ/σᵤ + σᵥ²/(2σᵤ²),   a = -ϵ/σᵥ - σᵥ/σᵤ.
# As σᵤ → 0, log Φ(a) ≈ -a²/2 and ϵ/σᵤ + σᵥ²/(2σᵤ²) are each O((σᵥ/σᵤ)²) and cancel, so
# an observation carried an error of up to about eps·a²/2: a few nats at ln σᵤ² = -40 and 1e9
# at -60 with σᵥ = 0.3. On a 400-observation sample the log-likelihood read +6e11 at ln σᵤ² = -60
# for a true -1988, and it rose toward the σᵤ → 0 boundary instead of levelling off. For
# a < -40 the density is now -log σᵤ - ϵ²/(2σᵥ²) + h(a) with h(a) = log Φ(a) + a²/2
# (`_normlogcdf_halfsq`); elsewhere it is the original expression, bit for bit.

using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions
using Random
using Optim

const _SF23 = SFrontiers.MLE_Backend.SFmle

# 512-bit log Φ: erfc down to -30, the Mills-ratio continued fraction below
function _b23_lPhi(a::BigFloat)
    a >= -30 && return log(erfc(-a / sqrt(BigFloat(2))) / 2)
    x = -a; r = x
    for k in 2000:-1:1
        r = x + k / r
    end
    return -a^2 / 2 - log(sqrt(2 * BigFloat(pi))) - log(r)
end

@testset "the analytic Exponential log-likelihood as σᵤ → 0" begin
    setprecision(BigFloat, 512) do

    rng = MersenneTwister(20260923)
    n = 200
    x = randn(rng, n); w = randn(rng, n)
    y = 1.0 .+ 0.5 .* x .+ 0.3 .* randn(rng, n) .+ 0.5 .* log.(rand(rng, n))     # v - u, u ~ Exp(0.5)
    Y = reshape(y, n, 1); X = hcat(ones(n), x); W = hcat(ones(n), w); V = ones(n, 1)
    po = (begx = 1, endx = 2, begw = 3, endw = 4, begv = 5, endv = 5)
    nll(p) = _SF23.LL_T(_SF23.Expo, Y, X, (), (), W, V, 1, n, po, p, nothing, nothing)
    function llref(θ)       # ln σᵤ² = c + d w, ln σᵥ² = e, in 512 bits
        β0, β1, c, d, e = BigFloat.(θ); s = BigFloat(0)
        for i in 1:n
            su = sqrt(exp(c + d * w[i])); sv = sqrt(exp(e)); ϵ = y[i] - β0 - β1 * x[i]
            s += -log(su) + _b23_lPhi(-ϵ / sv - sv / su) + ϵ / su + sv^2 / (2 * su^2)
        end
        return s
    end

    @testset "unchanged at ordinary points" begin
        # the pre-fix expression, in the same order of operations
        function nll_old(θ)
            σᵤ² = exp.(W * θ[3:4]); σᵤ = sqrt.(σᵤ²); σᵥ² = exp.(V * θ[5:5]); σᵥ = sqrt.(σᵥ²)
            ϵ = 1 * (Y - X * θ[1:2])
            ll = [(-log(σᵤ[i]) + _SF23._stable_normlogcdf(-(ϵ[i] / σᵥ[i]) - (σᵥ[i] / σᵤ[i]))
                   + ϵ[i] / σᵤ[i] + σᵥ²[i] / (2 * σᵤ²[i])) for i in 1:n]
            return -sum(ll)
        end
        for θ in ([1.0, 0.5, log(0.25), 0.2, log(0.09)],)
            a = [-(y[i] - θ[1] - θ[2] * x[i]) / exp(θ[5] / 2) - exp(θ[5] / 2) / exp((θ[3] + θ[4] * w[i]) / 2)
                 for i in 1:n]
            @test minimum(a) > -40                 # no observation takes the new branch
            @test nll(θ) === nll_old(θ)
            @test ForwardDiff.gradient(nll, θ) == ForwardDiff.gradient(nll_old, θ)
        end
    end

    @testset "against a 512-bit reference" begin
        for c in (-2.0,), d in (0.0, 0.5)
            θ = [1.0, 0.5, c, d, log(0.09)]
            @test abs(-nll(θ) - llref(θ)) <= 1e-9 * max(1.0, abs(Float64(llref(θ))))
        end
    end

    end   # setprecision
end
