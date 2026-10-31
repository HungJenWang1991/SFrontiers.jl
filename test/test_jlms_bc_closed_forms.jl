# Regression tests: stable JLMS/BC closed forms and truncated-normal E(u).
#
# With r = μₛ/σₛ, the closed-form JLMS is σₛ·m(r), m(r) = r + φ(r)/Φ(r), and log BC is
# -μₛ + σₛ²/2 + log Φ(r - σₛ) - log Φ(r) = h(r - σₛ) - h(r), h(x) = log Φ(x) + x²/2. The package
# formed m as exp(normlogpdf - normlogcdf) + r and log BC as that sum of four terms; both cancel
# as r → -∞ (μ < 0 with σᵤ → 0, or ϵ ≫ 0 with σᵥ → 0), so JLMS came out negative and BC above 1
# (all eight truncated-normal-family jlmsbc methods, jlmsbc(Expo), their predict and
# sfmodel_MoMTest twins). The panel_TFE truncated-normal marginal kernel took Φ(Λ) from 1 + erf
# floored at 1e-15 (wrong sign from Λ ≈ -8), and predict(:marginal) differentiated μ + σφ/Φ
# (wrong from a ≈ -27, NaN from -38). SFNumerics.sf_tn_mean (m) and sf_tn_logmgf (log BC) now
# take over below r = -40 (closed forms), Λ = -4 (panel kernel) and a = -10 (predict); above,
# the old expressions are evaluated, bit for bit. These tests fail on the code before the fix.

using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions
using StatsFuns: normlogpdf, normlogcdf, normpdf, normcdf
using Random
using Optim
using DataFrames
using Statistics

const _SF27N = SFrontiers.SFNumerics
const _SF27M = SFrontiers.MLE_Backend
const _SF27P = SFrontiers.Panel_Backend

# ---- 512-bit references (λ from the Mills-ratio continued fraction below -30) -------------------
function _b27_lam(a::BigFloat)                       # φ(a)/Φ(a)
    if a < -30
        x = -a; r = x
        for k in 300:-1:1                            # converged to 512 bits for -a ≥ 30 by 200 levels
            r = x + k / r
        end
        return r
    end
    return exp(-a^2 / 2 - log(sqrt(2 * BigFloat(pi))) - log(erfc(-a / sqrt(BigFloat(2))) / 2))
end
_b27_m(a) = (A = BigFloat(a); A + _b27_lam(A))
_b27_mp(a) = (A = BigFloat(a); 1 - _b27_lam(A) * _b27_m(A))                       # m'
_b27_mpp(a) = (A = BigFloat(a); l = _b27_lam(A); m = A + l; l * (m^2 + l * m - 1))  # m''
function _b27_h(a)                                   # log Φ(a) + a²/2
    A = BigFloat(a)
    A < -30 && return -log(_b27_lam(A)) - log(sqrt(2 * BigFloat(pi)))
    return log(erfc(-A / sqrt(BigFloat(2))) / 2) + A^2 / 2
end
_b27_jlms(mus, ss) = (M = BigFloat(mus); S = BigFloat(ss); S * _b27_m(M / S))
_b27_bc(mus, ss, s) = (M = BigFloat(mus); S = BigFloat(ss); r = M / S; exp(_b27_h(r - BigFloat(s)) - _b27_h(r)))
_b27_rel(x, r) = Float64(abs((BigFloat(x) - r) / r))
_b27_cl(x) = clamp(x, 1e-12, 1e12)
_b27_quiet(f) = redirect_stdout(f, devnull)
_b27_held(spec, θ; marginal = false) = _b27_quiet() do
    sfmodel_fit(spec = spec, method = sfmodel_method(method = :MLE), init = θ,
                optim_options = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,)),
                marginal = marginal, show_table = false, verbose = false)
end
# JLMS ≥ 0, 0 < BC ≤ 1, JLMS relative error and BC absolute error against the references
function _b27_check(j, b, jr, br; rtol = 1e-12, atol = 1e-14)
    j = vec(j); b = vec(b)
    @test all(>=(0), j)
    @test all(x -> 0 < x <= 1, b)
    @test maximum(_b27_rel(j[i], jr[i]) for i in eachindex(j)) <= rtol
    @test maximum(Float64(abs(BigFloat(b[i]) - br[i])) for i in eachindex(b)) <= atol
end

@testset "stable JLMS/BC closed forms and truncated-normal E(u)" begin
    setprecision(BigFloat, 512) do

    @testset "the kernels sf_tn_mean and sf_tn_logmgf" begin
        @test isdefined(_SF27N, :sf_tn_mean) && isdefined(_SF27N, :sf_tn_logmgf)
        if isdefined(_SF27N, :sf_tn_mean)
            m = _SF27N.sf_tn_mean
            d1(x) = ForwardDiff.derivative(m, x)
            d2(x) = ForwardDiff.derivative(y -> ForwardDiff.derivative(m, y), x)
            ev = 0.0; em = 0.0; e1 = 0.0; e2 = 0.0
            for a in vcat(-(10.0 .^ range(-2, 8; length = 60)), [-5.0, prevfloat(-5.0), -4.9, -4.8, -40.0, 0.0, 0.5, 3.0, 12.0])
                if -5 <= a <= 0        # the erfcx branch: up to about 60 eps (1.3e-14) on dense grids
                    em = max(em, _b27_rel(m(a), _b27_m(a)))
                else
                    ev = max(ev, _b27_rel(m(a), _b27_m(a)))
                end
                e1 = max(e1, _b27_rel(d1(a), _b27_mp(a)))
                e2 = max(e2, _b27_rel(d2(a), _b27_mpp(a)))
            end
            @test em <= 3e-14
            @test ev <= 1e-14          # before: the old forms lose all digits by r = -1e5
            @test e1 <= 5e-13
            @test e2 <= 1e-11
            @test all(isfinite, (m(-1e150), d1(-1e150), d2(-1e150)))
            @test m(-3.0f0) isa Float32 && abs(m(-3.0f0) - Float64(_b27_m(-3.0))) <= 1e-6
            lg = _SF27N.sf_tn_logmgf
            rng = Xoshiro(27)
            eb = 0.0; npos = 0
            for _ in 1:300
                a = -10.0^(log10(40.0) + 7 * rand(rng)); s = 10.0^(-15 + 20 * rand(rng))
                v = lg(a, s); npos += v > 0
                eb = max(eb, Float64(abs(exp(BigFloat(v)) - exp(_b27_h(BigFloat(a) - BigFloat(s)) - _b27_h(a)))))
            end
            @test npos == 0
            @test eb <= 1e-15
            @test abs(exp(lg(-3.0, 0.7)) - Float64(exp(_b27_h(BigFloat(-3.0) - BigFloat(0.7)) - _b27_h(-3.0)))) <= 1e-15
            # a > 0 (not reached by the package): the four terms directly, not h(a - s) - h(a) (9.9e-9 off at 1e4)
            for (a, s) in ((1e4, 1e-4), (1e7, 1e-7), (4.7, 1e-6))
                bref = exp(_b27_h(BigFloat(a) - BigFloat(s)) - _b27_h(a))
                @test _b27_rel(exp(lg(a, s)), bref) <= 1e-15
            end
        end
    end

    rng = Xoshiro(2701); n = 12
    x = randn(rng, n); z = randn(rng, n); X = hcat(ones(n), x); Z = hcat(ones(n), z)
    y = X * [1.0, 0.5] .+ 0.3 .* randn(rng, n) .- abs.(randn(rng, n)); Y = reshape(y, n, 1)
    e = y .- X * [1.0, 0.5]

    @testset "jlmsbc: cross-sectional methods (Trun, Half, Expo, Trun_Scale)" begin
        # Trun: μ = -5 + 0.5z with σᵤ² at its clamp (r ≈ -5e6)
        pos = (begx = 1, endx = 2, begz = 3, endz = 4, begw = 5, endw = 5, begv = 6, endv = 6)
        c = [1.0, 0.5, -5.0, 0.5, -33.0, 0.99]
        j, b = _SF27M.jlmsbc(_SF27M.Trun, 1, pos, c, Y, X, Z, nothing, ones(n, 1), ones(n, 1), nothing)
        mu = BigFloat.(Z * c[3:4]); su2 = BigFloat(1e-12); sv2 = BigFloat(exp(0.99)); s2 = su2 + sv2
        mus = (sv2 .* mu .- su2 .* BigFloat.(e)) ./ s2; ss = sqrt(sv2 * su2 / s2)
        _b27_check(j, b, [_b27_jlms(m, ss) for m in mus], [_b27_bc(m, ss, ss) for m in mus])
        # Trun at an ordinary point: the old expression, bit for bit
        c0 = [1.0, 0.5, 0.3, 0.2, log(0.5), log(0.3)]
        j0, b0 = _SF27M.jlmsbc(_SF27M.Trun, 1, pos, c0, Y, X, Z, nothing, ones(n, 1), ones(n, 1), nothing)
        μ = Z * c0[3:4]; w_pre = ones(n, 1) * c0[5:5]; v_pre = ones(n, 1) * c0[6:6]
        σᵤ² = @. clamp(exp(w_pre), 1e-12, 1e12); σᵥ² = @. clamp(exp(v_pre), 1e-12, 1e12)
        σ² = σᵤ² + σᵥ²; ϵ = 1 * (Y - X * c0[1:2])
        μₛ = @. (σᵥ² * μ - σᵤ² * ϵ) / σ²; σₛ = @. sqrt((σᵥ² * σᵤ²) / σ²); r = @. μₛ / σₛ
        @test j0 == (@. σₛ * exp(normlogpdf(r) - normlogcdf(r)) + μₛ)
        @test b0 == (@. exp(clamp(-μₛ + 0.5 * σₛ^2 + normlogcdf(r - σₛ) - normlogcdf(r), -500.0, 500.0)))

        # Half: σᵥ² at its clamp, observations above the frontier (ϵ > 0)
        posH = (begx = 1, endx = 2, begw = 3, endw = 3, begv = 4, endv = 4)
        yy = X * [1.0, 0.5] .+ 0.5 .+ abs.(randn(rng, n)); ee = yy .- X * [1.0, 0.5]
        j, b = _SF27M.jlmsbc(_SF27M.Half, 1, posH, [1.0, 0.5, log(0.25), -40.0], reshape(yy, n, 1), X,
                             nothing, nothing, ones(n, 1), ones(n, 1), nothing)
        su2 = BigFloat(0.25); sv2 = BigFloat(1e-12); s2 = su2 + sv2
        mus = (-su2 .* BigFloat.(ee)) ./ s2; ss = sqrt(sv2 * su2 / s2)
        _b27_check(j, b, [_b27_jlms(m, ss) for m in mus], [_b27_bc(m, ss, ss) for m in mus])

        # Expo at an ordinary point: the old expression, bit for bit
        c0 = [1.0, 0.5, log(0.25), log(0.09)]
        j0, b0 = _SF27M.jlmsbc(_SF27M.Expo, 1, posH, c0, Y, X, nothing, nothing, ones(n, 1), ones(n, 1), nothing)
        w_pre = ones(n, 1) * c0[3:3]; v_pre = ones(n, 1) * c0[4:4]
        σᵤ = @. sqrt(clamp(exp(w_pre), 1e-12, 1e12)); σᵥ² = @. clamp(exp(v_pre), 1e-12, 1e12)
        σᵥ = @. sqrt(σᵥ²); ϵ = 1 * (Y - X * c0[1:2]); μₛ = (-ϵ) - (σᵥ² ./ σᵤ); r = @. μₛ / σᵥ
        @test j0 == (@. σᵥ * exp(normlogpdf(r) - normlogcdf(r)) + μₛ)
        @test b0 == (@. exp(clamp(-μₛ + 0.5 * σᵥ² + normlogcdf(r - σᵥ) - normlogcdf(r), -500.0, 500.0)))

        # The switch r = -40 itself, through jlmsbc(Trun) and jlmsbc(Expo) at designed r (β = 0, μ = 0,
        # unit variances): the stable forms at r = -45 and -100, where the old expressions are off by
        # 4.5e-12 and 1.7e-9 in JLMS and 3e-14 to 7e-13 in BC; the old expressions, bit for bit, at
        # r = -39.9 and -20. Together these pin the switch to (-45, -39.9].
        rs = [-20.0, -39.9, -45.0, -100.0]; nr = length(rs); onr = ones(nr, 1); X0 = zeros(nr, 1)
        YT = reshape(-sqrt(2) .* rs, nr, 1)                       # Trun: μₛ = -ϵ/2, σₛ = √(1/2), r = -ϵ/√2
        jT, bT = _SF27M.jlmsbc(_SF27M.Trun, 1, (begx = 1, endx = 1, begz = 2, endz = 2, begw = 3, endw = 3, begv = 4, endv = 4),
                               [0.0, 0.0, 0.0, 0.0], YT, X0, onr, nothing, onr, onr, nothing)
        μ = onr * [0.0]; σᵤ² = @. clamp(exp(onr * [0.0]), 1e-12, 1e12); σᵥ² = @. clamp(exp(onr * [0.0]), 1e-12, 1e12)
        σ² = σᵤ² + σᵥ²; ϵ = 1 * (YT - X0 * [0.0])
        μₛ = @. (σᵥ² * μ - σᵤ² * ϵ) / σ²; σₛ = @. sqrt((σᵥ² * σᵤ²) / σ²); r = @. μₛ / σₛ
        jTo = @. σₛ * exp(normlogpdf(r) - normlogcdf(r)) + μₛ
        bTo = @. exp(clamp(-μₛ + 0.5 * σₛ^2 + normlogcdf(r - σₛ) - normlogcdf(r), -500.0, 500.0))
        YE = reshape(-(rs .+ 1), nr, 1)                           # Expo: μₛ = -ϵ - 1 = r
        jE, bE = _SF27M.jlmsbc(_SF27M.Expo, 1, (begx = 1, endx = 1, begw = 2, endw = 2, begv = 3, endv = 3),
                               [0.0, 0.0, 0.0], YE, X0, nothing, nothing, onr, onr, nothing)
        σᵤ = @. sqrt(clamp(exp(onr * [0.0]), 1e-12, 1e12)); σᵥ² = @. clamp(exp(onr * [0.0]), 1e-12, 1e12); σᵥ = @. sqrt(σᵥ²)
        ϵ = 1 * (YE - X0 * [0.0]); μₑ = (-ϵ) - (σᵥ² ./ σᵤ); rE = @. μₑ / σᵥ
        jEo = @. σᵥ * exp(normlogpdf(rE) - normlogcdf(rE)) + μₑ
        bEo = @. exp(clamp(-μₑ + 0.5 * σᵥ² + normlogcdf(rE - σᵥ) - normlogcdf(rE), -500.0, 500.0))
        sT = sqrt(BigFloat(1) / 2)
        for i in 1:nr
            mT = -BigFloat(YT[i]) / 2; mE = -BigFloat(YE[i]) - 1
            if rs[i] < -40
                @test _b27_rel(jT[i], _b27_jlms(mT, sT)) <= 1e-13
                @test abs(BigFloat(bT[i]) - _b27_bc(mT, sT, sT)) <= 1e-15
                @test _b27_rel(jE[i], _b27_jlms(mE, 1)) <= 1e-13
                @test abs(BigFloat(bE[i]) - _b27_bc(mE, 1, 1)) <= 1e-15
            else
                @test jT[i] === jTo[i] && bT[i] === bTo[i]
                @test jE[i] === jEo[i] && bE[i] === bEo[i]
            end
        end

        # Trun_Scale: u = h(q)u*, u* ~ N⁺(-5, e^-30)
        posS = (begx = 1, endx = 2, begz = 3, endz = 3, begq = 4, endq = 4, begw = 5, endw = 5, begv = 6, endv = 6)
        cS = [1.0, 0.5, -5.0, 0.4, -30.0, 0.99]
        j, b = _SF27M.jlmsbc(_SF27M.Trun_Scale, 1, posS, cS, Y, X, ones(n, 1), reshape(z, n, 1), ones(n, 1), ones(n, 1), nothing)
        h = _b27_cl.(exp.(0.4 .* z)); su2 = BigFloat.(_b27_cl.(exp(-30.0) .* h .^ 2)); sv2 = BigFloat(exp(0.99))
        s2 = su2 .+ sv2
        mus = (sv2 .* (-5 .* BigFloat.(h)) .- su2 .* BigFloat.(e)) ./ s2; ss = sqrt.(sv2 .* su2 ./ s2)
        _b27_check(j, b, [_b27_jlms(mus[i], ss[i]) for i in 1:n], [_b27_bc(mus[i], ss[i], ss[i]) for i in 1:n])
    end

    @testset "jlmsbc: panel methods (PFEWHH, PFEWHT, PanDecay, PanKumb90, PFECSWH)" begin
        Np, Tp = 4, 5; np_ = Np * Tp; idt = _SF27M.get_rowIDT(repeat(1:Np, inner = Tp))
        xp = randn(rng, np_, 2); qp = randn(rng, np_); tp = repeat(collect(1.0:Tp), Np)
        yp = xp * [0.6, -0.4] .+ 0.2 .* randn(rng, np_) .- 0.5 .* abs.(randn(rng, np_))
        ydm = similar(yp); xdm = similar(xp)
        for i in 1:Np
            ind = idt[i, 1]; ydm[ind] = _SF27M.sf_demean(yp[ind]); xdm[ind, :] = _SF27M.sf_demean(xp[ind, :])
        end
        edm = ydm .- xdm * [0.6, -0.4]
        function refWH(mu, su2, sv2, hv)
            jr = BigFloat[]; br = BigFloat[]
            for i in 1:Np
                ind = idt[i, 1]; hb = BigFloat.(hv[ind]); ht = hb .- sum(hb) / length(hb)
                et = BigFloat.(edm[ind]); et = et .- sum(et) / length(et)
                s2 = 1 / (sum(ht .^ 2) / sv2 + 1 / su2); ss = sqrt(s2); mss = (mu / su2 - sum(et .* ht) / sv2) * s2
                append!(jr, hb .* _b27_jlms(mss, ss)); append!(br, [_b27_bc(mss, ss, hh * ss) for hh in hb])
            end
            return jr, br
        end
        hv = _b27_cl.(exp.(0.3 .* qp))
        j, b = _SF27M.jlmsbc(_SF27M.PFEWHH, 1, (begx = 1, endx = 2, begq = 3, endq = 3, begw = 4, endw = 4, begv = 5, endv = 5),
                             [0.6, -0.4, 0.3, 0.0, -40.0], ydm, xdm, nothing, reshape(qp, np_, 1), ones(np_, 1), ones(np_, 1), idt)
        _b27_check(j, b, refWH(BigFloat(0), BigFloat(1), BigFloat(1e-12), hv)...)
        posT = (begx = 1, endx = 2, begz = 3, endz = 3, begq = 4, endq = 4, begw = 5, endw = 5, begv = 6, endv = 6)
        j, b = _SF27M.jlmsbc(_SF27M.PFEWHT, 1, posT, [0.6, -0.4, -5.0, 0.3, -33.0, log(0.04)], ydm, xdm, ones(np_, 1),
                             reshape(qp, np_, 1), ones(np_, 1), ones(np_, 1), idt)
        _b27_check(j, b, refWH(BigFloat(-5), BigFloat(1e-12), BigFloat(exp(log(0.04))), hv)...)
        function refG(G)
            jr = BigFloat[]; br = BigFloat[]
            su2 = BigFloat(1e-12); sv2 = BigFloat(exp(log(0.04)))
            for i in 1:Np
                ind = idt[i, 1]; g = BigFloat.(G[ind]); ee = BigFloat.(yp[ind] .- xp[ind, :] * [0.6, -0.4])
                D = sv2 + su2 * sum(g .^ 2); ms = (-5 * sv2 - sum(g .* ee) * su2) / D; ss = sqrt(sv2 * su2 / D)
                append!(jr, g .* _b27_jlms(ms, ss)); append!(br, [_b27_bc(ms, ss, gg * ss) for gg in g])
            end
            return jr, br
        end
        cG = [0.6, -0.4, -5.0, 0.1, -33.0, log(0.04)]
        j, b = _SF27M.jlmsbc(_SF27M.PanDecay, 1, posT, cG, reshape(yp, np_, 1), xp, ones(np_, 1), reshape(tp, np_, 1),
                             ones(np_, 1), ones(np_, 1), idt)
        _b27_check(j, b, refG(_b27_cl.(exp.(0.1 .* tp)))...)
        j, b = _SF27M.jlmsbc(_SF27M.PanKumb90, 1, posT, cG, reshape(yp, np_, 1), xp, ones(np_, 1), reshape(tp, np_, 1),
                             ones(np_, 1), ones(np_, 1), idt)
        _b27_check(j, b, refG(2 ./ (1 .+ _b27_cl.(exp.(0.1 .* tp))))...)
        # PanDecay at an ordinary point: the old expression, bit for bit
        c0 = [0.6, -0.4, 0.3, 0.1, log(0.25), log(0.04)]
        j0, b0 = _SF27M.jlmsbc(_SF27M.PanDecay, 1, posT, c0, reshape(yp, np_, 1), xp, ones(np_, 1), reshape(tp, np_, 1),
                               ones(np_, 1), ones(np_, 1), idt)
        σᵤ² = clamp(exp(c0[5]), 1e-12, 1e12); σᵥ² = clamp(exp(c0[6]), 1e-12, 1e12)
        jo = zeros(np_, 1); bo = zeros(np_, 1)
        for i in 1:Np
            ind = idt[i, 1]; ε = 1 * (reshape(yp, np_, 1)[ind] - (xp * c0[1:2])[ind])
            Gₜ = clamp.(exp.((reshape(tp, np_, 1) * c0[4:4])[ind]), 1e-12, 1e12); Gε = Gₜ .* ε
            ΣGε = sum(Gε); ΣGₜ² = sum(Gₜ .^ 2)
            μₛ = (0.3 * σᵥ² - ΣGε * σᵤ²) / (σᵥ² + σᵤ² * ΣGₜ²); σₛ² = σᵥ² * σᵤ² / (σᵥ² + σᵤ² * ΣGₜ²); σₛ = sqrt(σₛ²)
            r = μₛ / σₛ; mills = exp(normlogpdf(r) - normlogcdf(r))
            jo[ind] = Gₜ .* (μₛ + σₛ * mills)
            bo[ind] = exp.(clamp.(-Gₜ .* μₛ .+ 0.5 * ((Gₜ) .^ 2) * σₛ² .+ normlogcdf.(r .- Gₜ * σₛ) .- normlogcdf(r), -500.0, 500.0))
        end
        @test j0 == jo && b0 == bo
        # PFECSWH: σᵥ² at its clamp
        j, b = _SF27M.jlmsbc(_SF27M.PFECSWH, 1, (begx = 1, endx = 2, begw = 3, endw = 3, begv = 4, endv = 4),
                             [0.6, -0.4, log(0.01), -40.0], yp, xp, nothing, nothing, nothing, nothing, idt)
        su2 = BigFloat(exp(log(0.01))); sv2 = BigFloat(1e-12); ae = BigFloat.(yp .- xp * [0.6, -0.4])
        jr = BigFloat[]; br = BigFloat[]
        for i in 1:Np
            ind = idt[i, 1]; am = sum(ae[ind]) / length(ind) + sqrt(2 / BigFloat(pi)) * sqrt(su2)
            ms = -(su2 .* (ae[ind] .- am)) ./ (su2 + sv2); ss = sqrt(su2 * sv2 / (su2 + sv2))
            append!(jr, [_b27_jlms(m, ss) for m in ms]); append!(br, [_b27_bc(m, ss, ss) for m in ms])
        end
        _b27_check(j, b, jr, br)
    end

    rng = Xoshiro(2702); n = 200
    x = randn(rng, n); z = randn(rng, n); X = hcat(ones(n), x); Z1 = hcat(ones(n), z)
    y = X * [1.0, 0.5] .+ randn(rng, n) .- 2 .* abs.(randn(rng, n))

    @testset "a fit and its predict twins: TN with μ = a + bz at the σᵤ² clamp" begin
        spec = sfmodel_spec(depvar = y, frontier = X, zvar = Z1, noise = :Normal, ineff = :TruncatedNormal,
                            hetero = [:mu], type = :prod)
        θ = [0.54, 0.5, 0.54, 1.7, -33.4, 1.01]          # an arm-B ridge fit of the universality example
        r = _b27_held(spec, θ)
        @test r.coeff == θ
        mu = BigFloat.(Z1 * θ[3:4]); su2 = BigFloat(1e-12); sv2 = BigFloat(exp(θ[6])); s2 = su2 + sv2
        mus = (sv2 .* mu .- su2 .* BigFloat.(y .- X * θ[1:2])) ./ s2; ss = sqrt(sv2 * su2 / s2)
        jr = [_b27_jlms(m, ss) for m in mus]; br = [_b27_bc(m, ss, ss) for m in mus]
        _b27_check(r.jlms, r.bc, jr, br)            # before: 46 of 200 JLMS < 0, 20 BC > 1
        pj = predict(r; what = :jlms, frontier = X, zvar = Z1, depvar = y)
        pb = predict(r; what = :bc, frontier = X, zvar = Z1, depvar = y)
        @test vec(pj) == vec(r.jlms) && vec(pb) == vec(r.bc)
    end

    @testset "sfmodel_MoMTest twins" begin
        # σᵤ² and σᵥ² are taken from the result (res.coeff ends with σᵥ², σᵤ²): with a third moment
        # that is rounding noise, σᵤ² depends on the order of every floating-point operation.
        # exponential: residuals symmetric by construction, so σᵤ² ≈ 0
        rg = Xoshiro(11); hh = 100
        xm = randn(rg, hh); em = 0.3 .* randn(rg, hh)
        Xm = vcat(hcat(ones(hh), xm), hcat(ones(hh), xm)); ym = Xm * [1.0, 0.5] .+ vcat(em, -em)
        resE = _b27_quiet() do
            _SF27M.sfmodel_MoMTest((:type, [:prod]), (:dist, [:e]), (:depvar, [:y]), (:frontier, [:cons, :x]);
                                   data = DataFrame(y = ym, cons = Xm[:, 1], x = Xm[:, 2]), verbose = false)
        end
        Ym = reshape(ym, :, 1); Rs = vec(Ym - Xm * (Xm \ Ym))
        sv2, su2 = resE.coeff[end-1], resE.coeff[end]
        @test sqrt(sv2 / su2) > 1e4
        must = -(Rs .- sqrt(su2)) .- sv2 / sqrt(su2); sv = sqrt(BigFloat(sv2))
        _b27_check(resE.jlms, resE.bc, [_b27_jlms(BigFloat(m), sv) for m in must],
                   [_b27_bc(BigFloat(m), sv, sv) for m in must]; rtol = 1e-10)
        # half normal: residuals shaped so that σᵥ² = m₂ - (1 - 2/π)σᵤ² ≈ 3.6e-10. The new branch needs
        # residuals above E(u) (rat1 < -40), which this construction gives only by chance: 16 of 400
        # observations at nh = 400 (none at nh = 200); the guard below keeps the case on the branch.
        rg = Xoshiro(12); nh = 400; g = abs.(randn(rg, nh)); xh0 = randn(rg, nh)
        λ2 = 1 - 2 / pi; λ3 = (1 - 4 / pi) * sqrt(2 / pi)
        function momH(t)
            et = -(g .^ t .- mean(g .^ t)); et = et ./ std(et) .* 0.6
            xh = xh0 .- mean(xh0); xh = xh .- (xh' * et / (et' * et)) .* et
            Xh = hcat(ones(nh), xh); yh = Xh * [1.0, 0.5] .+ et
            R = yh .- Xh * (Xh \ yh); C = R .- mean(R); s2u = ((mean(C .^ 3) / λ3)^2)^(1 / 3)
            return mean(C .^ 2) - λ2 * s2u, s2u, yh, Xh, R
        end
        lo, hi = 0.5, 2.0
        for _ in 1:200
            mid = (lo + hi) / 2
            momH(mid)[1] / 0.36 > 1e-9 ? (lo = mid) : (hi = mid)
        end
        _, _, yh, Xh, _ = momH(lo)
        resH = _b27_quiet() do
            _SF27M.sfmodel_MoMTest((:type, [:prod]), (:dist, [:h]), (:depvar, [:y]), (:frontier, [:cons, :x]);
                                   data = DataFrame(y = yh, cons = Xh[:, 1], x = Xh[:, 2]), verbose = false)
        end
        sv2h, su2h = resH.coeff[end-1], resH.coeff[end]
        @test 0 < sv2h < 1e-8 * su2h
        Yh = reshape(yh, :, 1); Rh = vec(Yh - Xh * (Xh \ Yh))
        mst = BigFloat.(-su2h .* (Rh .- sqrt(su2h) * sqrt(2 / pi)) ./ (su2h + sv2h))
        sst = sqrt(BigFloat(su2h) * BigFloat(sv2h) / (BigFloat(su2h) + BigFloat(sv2h)))
        rat1h = (-su2h .* (Rh .- sqrt(su2h) * sqrt(2 / pi)) ./ (su2h + sv2h)) ./ sqrt(su2h * sv2h / (su2h + sv2h))
        # Julia 1.10 reaches none of these observations with this construction: the guard applies from 1.11 on.
        @test count(<(-40), rat1h) > 0 skip = VERSION < v"1.11"   # before (s25): JLMS 3.8e-4 off at those observations
        _b27_check(resH.jlms, resH.bc, [_b27_jlms(m, sst) for m in mst], [_b27_bc(m, sst, sst) for m in mst]; rtol = 1e-10)
    end

    @testset "panel_TFE truncated normal: the marginal kernel; JLMS and BC at the σᵤ² clamp" begin
        idx = (delta = [1], mu = 2, ln_sigma_u_sq = 3)
        k(L) = _SF27P._panel_uncondU_trun([1.0, L * 0.1, log(0.01)], idx, [0.0])
        for L in (-4.5, -7.0, -8.0, -10.0, -14.8, -100.0, -1e6)
            @test _b27_rel(k(L), 0.1 * _b27_m(L)) <= 1e-14        # before: wrong sign from -8
        end
        # above the switch the old expression, bit for bit (values and the z-gradient)
        function k_old(coef, z_obs)
            z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:1); h_val = exp(z_delta); mu = coef[idx.mu]
            sigma_u = clamp(exp(oftype(z_delta, 0.5) * coef[idx.ln_sigma_u_sq]), oftype(z_delta, 1e-12), oftype(z_delta, 1e12))
            Lambda = mu / sigma_u
            phi_Lambda = exp(oftype(z_delta, -0.5) * Lambda^2) / sqrt(oftype(z_delta, 2π))
            Phi_Lambda = max(oftype(z_delta, 0.5) * (1 + erf(Lambda / sqrt(oftype(z_delta, 2)))), oftype(z_delta, 1e-15))
            return h_val * sigma_u * (Lambda + phi_Lambda / Phi_Lambda)
        end
        for L in (-3.99, -2.0, -0.5, 1.5), zz in (-0.3, 0.8)
            cf = [0.3, L * 0.2, log(0.04)]
            @test _SF27P._panel_uncondU_trun(cf, idx, [zz]) === k_old(cf, [zz])
            @test ForwardDiff.gradient(zv -> _SF27P._panel_uncondU_trun(cf, idx, zv), [zz]) ==
                  ForwardDiff.gradient(zv -> k_old(cf, zv), [zz])
        end
        # a panel_TFE truncated-normal model on 40 firms over 5 periods
        rp = Xoshiro(2021); nf, T = 40, 5; npp = nf * T
        idp = repeat(1:nf, inner = T); XP = hcat(randn(rp, npp), randn(rp, npp)); ZP = hcat(randn(rp, npp))
        yP = repeat(0.5 .* randn(rp, nf), inner = T) .+ XP * [0.6, -0.4] .+ 0.3 .* randn(rp, npp) .-
             exp.(0.3 .* ZP[:, 1]) .* repeat(0.2 .+ 0.5 .* abs.(randn(rp, nf)), inner = T)
        specP = sfmodel_spec(datatype = :panel_TFE, depvar = yP, frontier = XP, zvar = ZP, id = idp,
                             noise = :Normal, ineff = :TruncatedNormal)
        # the MLE panel_TFE truncated normal (PFEWHT) at the σᵤ² clamp: JLMS ≥ 0, BC ≤ 1
        rP = _b27_held(specP, sfmodel_init(spec = specP, frontier = [0.6, -0.4], scaling = [0.3], mu = -0.1,
                                           ln_sigma_u_sq = -33.0, ln_sigma_v_sq = log(0.09)))
        @test all(>=(0), rP.jlms) && all(x -> 0 < x <= 1, rP.bc)   # before: all 200 JLMS < 0
    end

    @testset "predict(:marginal) for the truncated normal" begin
        rg = Xoshiro(7); nt = 150
        zt = randn(rg, nt); Xt = hcat(ones(nt), randn(rg, nt)); Zt = hcat(ones(nt), zt)
        yt = Xt * [1.0, 0.5] .+ 0.2 .* randn(rg, nt) .- 0.1 .* abs.(randn(rg, nt))
        spec = sfmodel_spec(depvar = yt, frontier = Xt, zvar = Zt, noise = :Normal, ineff = :TruncatedNormal,
                            hetero = [:mu], type = :prod)
        θ = [1.0, 0.5, -1.0, 1.0, log(0.01), log(0.04)]
        r = _b27_held(spec, θ)
        zs = [-1.2, -2.0, -2.7, -2.8, -3.0, -5.0, -100.0]              # a = -22 .. -1010
        pm = predict(r; what = :marginal, zvar = hcat(ones(length(zs)), zs))
        for (i, zz) in enumerate(zs)
            @test _b27_rel(pm[i, 2], _b27_mp((-1.0 + zz) / 0.1)) <= 1e-12   # before: -Inf / NaN from a = -28
        end
        @test all(isfinite, predict(r; what = :marginal, zvar = hcat(ones(nt), 3 .* zt)))
        # above a = -10 the old expression, bit for bit
        Eu_old(zv) = (μ = zv[1] * θ[3] + zv[2] * θ[4]; σu = sqrt(exp(θ[5])); a = μ / σu;
                      μ + σu * (normpdf(a) / normcdf(a)))
        zo = [-0.08, 0.0, 0.5, 2.0]                                     # a = -10.8 .. +10
        po = predict(r; what = :marginal, zvar = hcat(ones(length(zo)), zo))
        @test po[1, :] != ForwardDiff.gradient(Eu_old, [1.0, zo[1]])  # a = -10.8: the new branch
        for i in 2:length(zo)
            @test po[i, :] == ForwardDiff.gradient(Eu_old, [1.0, zo[i]])
        end
        # the scaling property model (E(u*) at a = μ*/σᵤ* = -45; before NaN below -38.5)
        specS = sfmodel_spec(depvar = yt, frontier = Xt, zvar = reshape(zt, nt, 1), noise = :Normal,
                             ineff = :TruncatedNormal, hetero = :scaling, type = :prod)
        rS = _b27_held(specS, sfmodel_init(spec = specS, frontier = [1.0, 0.5], scaling = [0.3], mu = -4.5,
                                           ln_sigma_u_sq = log(0.01), ln_sigma_v_sq = log(0.04)))
        pS = predict(rS; what = :marginal, zvar = reshape([0.0, 1.0], 2, 1))
        for (i, zz) in enumerate((0.0, 1.0))
            @test _b27_rel(pS[i], 0.3 * exp(0.3 * zz) * sqrt(BigFloat(exp(log(0.01)))) * _b27_m(-4.5 / sqrt(exp(log(0.01))))) <= 1e-12
        end
    end

    end   # setprecision
end
