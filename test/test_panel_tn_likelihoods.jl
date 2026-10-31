# The analytic truncated-normal likelihoods of the panel models against 512-bit references.
#
# LL_T(PanKumb90) (Kumbhakar's time-varying model, with the package's G_t = 2/(1 + exp(τQ_t))) and
# LL_T(PFEWHT) (Wang and Ho's true fixed-effect model with a truncated normal, on rows demeaned per
# firm) are checked in value at an ordinary point, LL_T(PFEWHT) also where μ/σᵤ lies far below -40,
# where the likelihoods take the Gaussian bulk of ½(a₁² - a₀²) + log Φ(a₁) - log Φ(a₀) out analytically. The
# gradients and Hessians of LL_T(PFEWHT), LL_T(PanDecay) and LL_T(PanKumb90) are checked at the same
# points. The references evaluate the textbook expressions in 512-bit BigFloat; their log Φ comes from
# erfc and, below -20, from the Mills-ratio continued fraction, and its derivative φ/Φ, formed in
# BigFloat, is given to ForwardDiff as a rule, so the derivative references are exact automatic
# differentiation at 512 bits (no finite differences).

using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions
using Random

@testset "panel truncated-normal likelihoods against 512-bit references" begin
    _PTN = SFrontiers.MLE_Backend.SFmle

    # log Φ(a) in BigFloat: log1p(-erfc(a/√2)/2) above 0, erfc on [-20, 0], below -20 the
    # Mills-ratio continued fraction (400 levels); for a Dual, the derivative φ/Φ in BigFloat
    function _ptn_lPhi(a::BigFloat)
        a > 0 && return log1p(-erfc(a / sqrt(BigFloat(2))) / 2)
        a >= -20 && return log(erfc(-a / sqrt(BigFloat(2))) / 2)
        x = -a; r = x
        for k in 400:-1:1
            r = x + k / r
        end
        return -a^2 / 2 - log(sqrt(2 * BigFloat(pi))) - log(r)
    end
    function _ptn_lPhi(d::ForwardDiff.Dual{Tg}) where {Tg}
        a = ForwardDiff.value(d)
        v = _ptn_lPhi(a)
        λ = exp(-a * a / 2 - log(sqrt(2 * BigFloat(pi))) - v)
        return ForwardDiff.Dual{Tg}(v, λ * ForwardDiff.partials(d))
    end

    # LL_T(PanDecay) and LL_T(PanKumb90), θ = (β₀, β₁, μ, τ, ln σᵤ², ln σᵥ²): per firm
    # ε = y - β₀ - β₁x, G_t = exp(τQ_t) (PanDecay) or 2/(1 + exp(τQ_t)) (PanKumb90),
    # D = σᵥ² + σᵤ²ΣG², μ* = (μσᵥ² - σᵤ²ΣGε)/D, σ*² = σᵥ²σᵤ²/D, quad = Σε²/σᵥ² + μ²/σᵤ² - μ*²/σ*², and
    # ll = -(T/2)log 2π + log Φ(μ*/σ*) + ½log σ*² - ½quad - (T/2)ln σᵥ² - ½ln σᵤ² - log Φ(μ/σᵤ)
    function _ptn_refG(θ, y, x, Q, idt, kumb)
        β0, β1, μ, τ, c, g = θ
        su2 = exp(c); sv2 = exp(g); s = zero(β0)
        for i in axes(idt, 1)
            ind = idt[i, 1]; Ti = idt[i, 2]
            ε = [BigFloat(y[k]) - β0 - β1 * x[k] for k in ind]
            G = [kumb ? 2 / (1 + exp(τ * Q[k])) : exp(τ * Q[k]) for k in ind]
            D = sv2 + su2 * sum(G .^ 2)
            μs = (μ * sv2 - su2 * sum(G .* ε)) / D; ss2 = sv2 * su2 / D
            quad = sum(ε .^ 2) / sv2 + μ^2 / su2 - μs^2 / ss2
            s += (-Ti * log(2 * BigFloat(pi)) / 2 + _ptn_lPhi(μs / sqrt(ss2)) + log(ss2) / 2 - quad / 2
                  - Ti * g / 2 - c / 2 - _ptn_lPhi(μ / sqrt(su2)))
        end
        return s
    end

    # LL_T(PFEWHT), θ = (β₁, β₂, μ, τ, ln σᵤ², ln σᵥ²), on rows demeaned per firm: per firm
    # ε̃ = ỹ - x̃β, h = exp(τq), h̃ = h - mean(h), σ**² = 1/(h̃'h̃/σᵥ² + 1/σᵤ²),
    # μ** = (μ/σᵤ² - ε̃'h̃/σᵥ²)σ**², and
    # ll = -½(T - 1)log 2π - ½(T - 1)ln σᵥ² - ½Σ(ε̃_t - mean ε̃)²/σᵥ² + ½(μ**²/σ**² - μ²/σᵤ²)
    #      + ½log σ**² - ½ln σᵤ² + log Φ(μ**/σ**) - log Φ(μ/σᵤ)
    function _ptn_refW(θ, y, x, q, idt)
        b1, b2, μ, τ, c, g = θ
        su2 = exp(c); sv2 = exp(g); s = zero(b1)
        for i in axes(idt, 1)
            ind = idt[i, 1]; Ti = idt[i, 2]
            ε = [BigFloat(y[k]) - b1 * x[k, 1] - b2 * x[k, 2] for k in ind]
            ct = ε .- sum(ε) / Ti
            h = [exp(τ * q[k]) for k in ind]; ht = h .- sum(h) / Ti
            ss2 = 1 / (sum(ht .^ 2) / sv2 + 1 / su2)
            mss = (μ / su2 - sum(ε .* ht) / sv2) * ss2
            s += (-(Ti - 1) * log(2 * BigFloat(pi)) / 2 - (Ti - 1) * g / 2 - sum(ct .^ 2) / (2 * sv2)
                  + (mss^2 / ss2 - μ^2 / su2) / 2 + log(ss2) / 2 - c / 2
                  + _ptn_lPhi(mss / sqrt(ss2)) - _ptn_lPhi(μ / sqrt(su2)))
        end
        return s
    end

    # errors of the gradient and the Hessian of the log-likelihood against the reference, normalised
    # by max(1, max|g|) and max(1, max|H|) of the reference
    _ptn_mx(v) = maximum(abs, v)
    function _ptn_err(nll, ref, θ)
        g = -ForwardDiff.gradient(nll, θ); H = -ForwardDiff.hessian(nll, θ)
        tb = BigFloat.(θ)
        gr = ForwardDiff.gradient(ref, tb); Hr = ForwardDiff.hessian(ref, tb)
        return (Float64(_ptn_mx(big.(g) .- gr) / max(1, _ptn_mx(gr))),
                Float64(_ptn_mx(big.(H) .- Hr) / max(1, _ptn_mx(Hr))))
    end

    # an ordinary point
    _ptn_ptsG = ([0.5, 0.5, 0.3, 0.1, -1.0, -2.0],)
    # the same for PFEWHT, and μ/σᵤ = -1e3 and -1e6
    _ptn_ptsW = ([0.6, -0.4, 0.3, 0.3, log(0.25), log(0.04)], [0.6, -0.4, -1e3, 0.3, 0.0, log(0.04)],
                 [0.6, -0.4, -1e6, 0.3, 0.0, log(0.04)])

    setprecision(BigFloat, 512) do

    @testset "LL_T(PanKumb90) against a 512-bit reference" begin
        rng = MersenneTwister(20261007)
        N = 30; T = 5; nn = N * T
        idt = _PTN.get_rowIDT(repeat(1:N, inner = T))
        xp = randn(rng, nn); Xp = hcat(ones(nn), xp); Zp = ones(nn, 1)
        Qp = reshape(collect(repeat(0:T-1, N)) .- 2.0, nn, 1)
        yp = 0.5 .+ 0.5 .* xp .+ 0.3 .* randn(rng, nn) .- repeat(abs.(randn(rng, N)), inner = T)
        pop = (begx = 1, endx = 2, begz = 3, endz = 3, begq = 4, endq = 4, begw = 5, endw = 5, begv = 6, endv = 6)
        nll(θ) = _PTN.LL_T(_PTN.PanKumb90, yp, Xp, Zp, Qp, ones(nn, 1), ones(nn, 1), 1, nn, pop, θ, idt, nothing)
        for θ in _ptn_ptsG
            ref = Float64(_ptn_refG(BigFloat.(θ), yp, xp, Qp, idt, true))
            @test abs(-nll(θ) - ref) <= 1e-9 * max(1, abs(ref))
        end
    end

    @testset "LL_T(PFEWHT) against a 512-bit reference" begin
        rng = MersenneTwister(20261008)
        Np = 30; Tp = 5; np = Np * Tp
        idt = _PTN.get_rowIDT(repeat(1:Np, inner = Tp))
        xp = randn(rng, np, 2); qp = randn(rng, np)
        yp = xp * [0.6, -0.4] .+ 0.2 .* randn(rng, np) .- 0.5 .* abs.(randn(rng, np))
        ydm = similar(yp); xdm = similar(xp)
        for i in 1:Np
            ind = idt[i, 1]; ydm[ind] = _PTN.sf_demean(yp[ind]); xdm[ind, :] = _PTN.sf_demean(xp[ind, :])
        end
        posT = (begx = 1, endx = 2, begz = 3, endz = 3, begq = 4, endq = 4, begw = 5, endw = 5, begv = 6, endv = 6)
        nll(θ) = _PTN.LL_T(_PTN.PFEWHT, ydm, xdm, ones(np, 1), reshape(qp, np, 1), ones(np, 1), ones(np, 1), 1, np,
                           posT, θ, idt, nothing)
        for θ in _ptn_ptsW
            ref = Float64(_ptn_refW(BigFloat.(θ), ydm, xdm, qp, idt))
            @test abs(-nll(θ) - ref) <= 1e-9 * max(1, abs(ref))
        end
    end

    @testset "derivatives of the panel truncated-normal likelihoods" begin
        rng = MersenneTwister(20261009)
        # 10 firms, T = 5: data drawn as in the PanKumb90 test set, then as in the PFEWHT test set
        N = 10; T = 5; nn = N * T
        idt = _PTN.get_rowIDT(repeat(1:N, inner = T))
        xp = randn(rng, nn); Xp = hcat(ones(nn), xp); Zp = ones(nn, 1)
        Qp = reshape(collect(repeat(0:T-1, N)) .- 2.0, nn, 1)
        yp = 0.5 .+ 0.5 .* xp .+ 0.3 .* randn(rng, nn) .- repeat(abs.(randn(rng, N)), inner = T)
        pop = (begx = 1, endx = 2, begz = 3, endz = 3, begq = 4, endq = 4, begw = 5, endw = 5, begv = 6, endv = 6)
        xw = randn(rng, nn, 2); qw = randn(rng, nn)
        yw = xw * [0.6, -0.4] .+ 0.2 .* randn(rng, nn) .- 0.5 .* abs.(randn(rng, nn))
        ydm = similar(yw); xdm = similar(xw)
        for i in 1:N
            ind = idt[i, 1]; ydm[ind] = _PTN.sf_demean(yw[ind]); xdm[ind, :] = _PTN.sf_demean(xw[ind, :])
        end
        # one layout serves the three likelihoods: β in 1–2, μ in 3, τ in 4, ln σᵤ² in 5, ln σᵥ² in 6
        nllD(θ) = _PTN.LL_T(_PTN.PanDecay, yp, Xp, Zp, Qp, ones(nn, 1), ones(nn, 1), 1, nn, pop, θ, idt, nothing)
        nllK(θ) = _PTN.LL_T(_PTN.PanKumb90, yp, Xp, Zp, Qp, ones(nn, 1), ones(nn, 1), 1, nn, pop, θ, idt, nothing)
        nllW(θ) = _PTN.LL_T(_PTN.PFEWHT, ydm, xdm, ones(nn, 1), reshape(qw, nn, 1), ones(nn, 1), ones(nn, 1), 1, nn,
                            pop, θ, idt, nothing)
        refD(θ) = _ptn_refG(θ, yp, xp, Qp, idt, false)
        refK(θ) = _ptn_refG(θ, yp, xp, Qp, idt, true)
        refW(θ) = _ptn_refW(θ, ydm, xdm, qw, idt)
        for (nll, ref, pts) in ((nllD, refD, _ptn_ptsG), (nllK, refK, _ptn_ptsG), (nllW, refW, _ptn_ptsW)), θ in pts
            eg, eH = _ptn_err(nll, ref, θ)
            @test eg <= 1e-11
            @test eH <= 1e-9
        end
    end

    end   # setprecision
end
