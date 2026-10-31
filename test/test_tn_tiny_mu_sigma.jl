# Regression tests: the truncated normal with μ and σᵤ both tiny.
#
# LL_T(Trun) and LL_T(Trun_Scale), the analytic truncated-normal likelihoods of cross-sectional
# models, contain log Φ(a₁) - log Φ(a₀) with a₀ = μ/σᵤ and a₁ = μ*/σ* = a₀ + δ (h(a₁) - h(a₀),
# h = log Φ + a²/2, where an argument is below -40). When μ and σᵤ are both tiny next to σᵥ,
# δ ≈ -(σᵤ/σᵥ)(ϵ/σᵥ) → 0 while a₀ stays moderate. The difference is accurate as a value, but
# ForwardDiff took its derivatives from the two terms separately, each about σᵥ/σᵤ times larger
# than the difference: on one observation with σᵥ = 1 the Hessian was off by 1.75e-2 of its largest
# entry at σᵤ = 1e-6, 22.9 at 1e-9 and 4.57e5 at 1e-13, and the curvature in μ was exactly 0 against
# -2.62e11 at σᵤ = 9.4e-14, μ/σᵤ = -1. Where that error is estimated above 1e-6/σᵥ², the difference
# is now δ·D(a₀, δ), δ from _tn_shift and D a 4-point Gauss-Legendre divided difference of log Φ
# (or of h); every other observation evaluates the earlier expression, bit for bit. In the band
# a₀ < -40 ≤ a₁ the old form takes h(a₁) through erfcx, so the switch estimates the error there as
# for a₀ ≥ -40.
# The bit-identity tests compare with in-test copies of the earlier likelihoods.

using Test
using SFrontiers
using ForwardDiff
using StatsFuns
using SpecialFunctions
using Random

const _SF44 = SFrontiers.MLE_Backend.SFmle

# ---- the package's likelihoods and in-test copies of the old ones ----------------------------
# As callable objects of one type each, so that ForwardDiff compiles each likelihood once.
const _t44_poT = (begx = 1, endx = 1, begz = 2, endz = 2, begw = 3, endw = 3, begv = 4, endv = 4)
const _t44_poS = (begx = 1, endx = 1, begz = 2, endz = 2, begq = 3, endq = 3, begw = 4, endw = 4, begv = 5, endv = 5)
struct _T44F{K, TP}
    Y::Matrix{Float64}; X::Matrix{Float64}; Z::Matrix{Float64}; Q::Matrix{Float64}
    W::Matrix{Float64}; V::Matrix{Float64}; n::Int; po::TP
end
# K = :new / :old (LL_T(Trun) / its copy), :newS / :oldS (LL_T(Trun_Scale) / its copy); data y, x, z, w, q
function _T44F{K}(y, x, z, w, q) where {K}
    n = length(y); col(v) = reshape(Float64.(v), n, 1)
    po = K in (:new, :old) ? _t44_poT : _t44_poS
    return _T44F{K, typeof(po)}(col(y), col(x), col(z), col(q), col(w), ones(n, 1), n, po)
end

# The same operations in the same order as the earlier form of LL_T(Trun) and LL_T(Trun_Scale) (a plain
# loop instead of @floop, which only distributes the observations), so that a point where no
# observation takes the new form must give the same bits.
function _t44_oldTrun(Y, X, Z, W, V, PorC, nobs, po, rho)
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
    for i in 1:nobs
        a₁ = (μₛ[i]) / sqrt(σₛ²[i])
        a₀ = (μ[i]) / σᵤ[i]
        if min(a₁, a₀) < -40
            if a₁ <= 0
                ll[i] = ( - 0.5 * log(σ²[i]) - log(StatsFuns.sqrt2π) - 0.5 * ϵ[i]^2 / σᵥ²[i]
                          + _SF44._normlogcdf_halfsq(a₁) - _SF44._normlogcdf_halfsq(a₀) )
            else
                sᵢ = sqrt(σ²[i])
                wᵢ = μ[i] + σᵤ[i] * ϵ[i] / (sᵢ + σᵤ[i])
                dᵢ = sqrt(σᵥ²[i]) * wᵢ / (sᵢ * σᵤ[i])
                ll[i] = ( - 0.5 * log(σ²[i]) - log(StatsFuns.sqrt2π) + StatsFuns.normlogcdf(a₁)
                          + wᵢ * μₛ[i] / σᵤ²[i] - 0.5 * dᵢ^2 - _SF44._normlogcdf_halfsq(a₀) )
            end
        else
            ll[i] = (- 0.5 * log(σ²[i])
                     + StatsFuns.normlogpdf((μ[i] + ϵ[i]) / sqrt(σ²[i]))
                     + _SF44._stable_normlogcdf(a₁)
                     - _SF44._stable_normlogcdf(a₀) )
        end
    end
    return -sum(ll)
end
function _t44_oldScale(Y, X, Z, Q, W, V, PorC, nobs, po, rho)
    β  = rho[po.begx : po.endx]
    δ1 = rho[po.begz : po.endz]
    τ  = rho[po.begq : po.endq]
    δ2 = rho[po.begw : po.endw]
    γ  = rho[po.begv : po.endv]
    hscale = exp.(Q*τ)
    μ   = (Z*δ1) .* hscale
    σᵤ² = exp.(W * δ2) .* hscale.^2
    σᵤ  = sqrt.(σᵤ²)
    σᵥ² = exp.(V * γ)
    σ²  = σᵤ² + σᵥ²
    ϵ   = PorC*(Y - X * β)
    μₛ  = (σᵥ² .* μ - σᵤ² .* ϵ) ./ σ²
    σₛ² = (σᵥ² .* σᵤ²) ./ σ²
    ll = Vector{eltype(ϵ)}(undef, nobs)
    for i in (1:nobs)
        a₁ = (μₛ[i]) / sqrt(σₛ²[i])
        a₀ = (μ[i]) / σᵤ[i]
        if min(a₁, a₀) < -40
            if a₁ <= 0
                ll[i] = ( - 0.5 * log(σ²[i]) - log(StatsFuns.sqrt2π) - 0.5 * ϵ[i]^2 / σᵥ²[i]
                          + _SF44._normlogcdf_halfsq(a₁) - _SF44._normlogcdf_halfsq(a₀) )
            else
                sᵢ = sqrt(σ²[i])
                wᵢ = μ[i] + σᵤ[i] * ϵ[i] / (sᵢ + σᵤ[i])
                dᵢ = sqrt(σᵥ²[i]) * wᵢ / (sᵢ * σᵤ[i])
                ll[i] = ( - 0.5 * log(σ²[i]) - log(StatsFuns.sqrt2π) + StatsFuns.normlogcdf(a₁)
                          + wᵢ * μₛ[i] / σᵤ²[i] - 0.5 * dᵢ^2 - _SF44._normlogcdf_halfsq(a₀) )
            end
        else
            ll[i] = ( - 0.5 * log(σ²[i])
                   + StatsFuns.normlogpdf((μ[i] + ϵ[i]) / sqrt(σ²[i]))
                   + _SF44._stable_normlogcdf(a₁)
                   - _SF44._stable_normlogcdf(a₀)  )
        end
    end
    return -sum(ll)
end
(f::_T44F{:new})(θ)  = _SF44.LL_T(_SF44.Trun, f.Y, f.X, f.Z, (), f.W, f.V, 1, f.n, f.po, θ, nothing, nothing)
(f::_T44F{:old})(θ)  = _t44_oldTrun(f.Y, f.X, f.Z, f.W, f.V, 1, f.n, f.po, θ)
(f::_T44F{:newS})(θ) = _SF44.LL_T(_SF44.Trun_Scale, f.Y, f.X, f.Z, f.Q, f.W, f.V, 1, f.n, f.po, θ, nothing, nothing)
(f::_T44F{:oldS})(θ) = _t44_oldScale(f.Y, f.X, f.Z, f.Q, f.W, f.V, 1, f.n, f.po, θ)

# value, gradient and Hessian of two functions agree in every bit
function _t44_same(f1, f2, θ)
    return isequal(f1(θ), f2(θ)) && isequal(ForwardDiff.gradient(f1, θ), ForwardDiff.gradient(f2, θ)) &&
           isequal(ForwardDiff.hessian(f1, θ), ForwardDiff.hessian(f2, θ))
end

@testset "truncated normal with μ and σᵤ both tiny (LL_T(Trun), LL_T(Trun_Scale))" begin

    # 1. ordinary points on 60 observations, including the paused Monte Carlo study's extreme regions
    #    (the truncated normal near its exponential limit, μ/σᵤ = -360 with σᵤ = 659 and -764 with
    #    σᵤ = 1431; the scaling model at the σᵥ → 0 corner): the earlier expression, bit for bit.
    @testset "ordinary points: unchanged bit for bit" begin
        rng = MersenneTwister(44)
        n = 60
        x = randn(rng, n); z = randn(rng, n); v = randn(rng, n); u = abs.(2 .* randn(rng, n))
        y = 0.5 .+ 0.5 .* x .+ v .- u
        for (zc, wc, θs) in ((ones(n), ones(n), ([0.5, 2.42, 1.9, -0.3], [0.5, -2.373e5, 12.98, 0.128],
                                                [0.5, -1.094e6, 14.53, -0.02816])),
                             (1 .+ 0.6 .* z, 1 .- 0.3 .* z, ([0.5, 2.43, 1.9, -0.3], [0.5, -0.55, 2.39, -0.02])))
            fn = _T44F{:new}(y, x, zc, wc, ones(n)); fo = _T44F{:old}(y, x, zc, wc, ones(n))
            for θ in θs
                @test _t44_same(fn, fo, θ)
            end
        end
        fn = _T44F{:newS}(y, ones(n), ones(n), ones(n), z); fo = _T44F{:oldS}(y, ones(n), ones(n), ones(n), z)
        for θ in ([0.45, 1.70, 0.33, 1.54, -0.09], [0.30, -1.10, 0.40, 1.96, 0.10],
                  [12.1, 16.13, 0.05074, 1.338, -29.62], [12.1, 16.13, 0.05074, 1.338, -29.29])
            @test _t44_same(fn, fo, θ)
        end
    end

    # 2. the names the paused Monte Carlo study's startup guard tests
    @testset "names kept" begin
        @test isdefined(_SF44, :_tn_shift)
        @test isdefined(_SF44, :_stable_normlogcdf)
        @test !isdefined(_SF44, :_normlogcdf_diff_tail)
    end
end
