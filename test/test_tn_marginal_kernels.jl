# Regression tests: the truncated-normal E(u) kernels behind result.marginal.
#
# For u ~ N⁺(μ, σᵤ²), E(u) = σᵤ·m(Λ), m(Λ) = Λ + φ(Λ)/Φ(Λ), Λ = μ/σᵤ. The nine truncated-normal E(u)
# kernels behind result.marginal (marg_trun, marg_scal, marg_fewht, marg_pdecay, marg_pkumb90 in
# SFmarginal.jl; uncondU_trun and _base_Eu_star(::TruncatedNormal) in sf_MCI_marginal_v21.jl and
# sf_MSLE_marginal_v21.jl) formed σᵤ·(Λ + exp(log φ(Λ) - log Φ(Λ))), whose two logarithms of about
# -Λ²/2 cancel as Λ → -∞.
# Relative errors: E(u) 2e-9 at Λ = -100, 5e-5 at -1e3, 13% to 61% at -1e4, negative at -1e5; the
# marginal effects (ForwardDiff derivatives), maxima over random designs, about 5e-8 near -100, 5e-4
# near -1e3 and 6 near -1e4 (6e-9, 7e-5 and 0.6 in the scaling and panel models, which carry only
# the value error). Below Λ = -40 they now return σᵤ·SFNumerics.sf_tn_mean(Λ); above,
# the old expression, bit for bit. These tests fail on the code before the fix.

using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions: erfc, logerfcx
using StatsFuns: normlogpdf, normlogcdf
using Random
using Optim

const _B33M = SFrontiers.MLE_Backend.SFmle
const _B33C = SFrontiers.MCI_Backend
const _B33S = SFrontiers.MSLE_Backend

# ---- 512-bit references: λ = φ/Φ from the Mills-ratio continued fraction below -30 ------------
function _b33_lam(a::BigFloat)
    if a < -30
        t = -a; r = t
        for k in 300:-1:1                            # converged to 512 bits for -a ≥ 30
            r = t + k / r
        end
        return r
    end
    return exp(-a^2 / 2 - log(sqrt(2 * BigFloat(pi))) - log(erfc(-a / sqrt(BigFloat(2))) / 2))
end
_b33_m(a) = (A = BigFloat(a); A + _b33_lam(A))                      # m
_b33_mp(a) = (A = BigFloat(a); 1 - _b33_lam(A) * _b33_m(A))         # m'
_b33_rel(x, r) = isfinite(x) ? Float64(abs((BigFloat(x) - r) / r)) : Inf
_b33_relv(x, r) = maximum(iszero(r[i]) ? Float64(abs(BigFloat(x[i]))) : _b33_rel(x[i], r[i]) for i in eachindex(r))
_b33_quiet(f) = redirect_stdout(f, devnull)

# ---- in-test copies of the nine kernels as they were before the fix ---------------------------
function _b33o_mle_trun(pos, coef, Zmarg, Wmarg)                      # SFmarginal.jl, marg_trun
    z_pre = Zmarg'*coef[pos.begz : pos.endz]
    w_pre = Wmarg'*coef[pos.begw : pos.endw]
    μ   = z_pre
    σᵤ  = sqrt(clamp(exp(w_pre), 1e-12, 1e12))
    Λ = μ/σᵤ
    uncondU = σᵤ* (Λ + exp(normlogpdf(Λ) - normlogcdf(Λ)))
end
function _b33o_mle_scal(pos, coef, Zmarg, Qmarg, Wmarg)               # marg_scal
    z_pre = Zmarg'*coef[pos.begz : pos.endz]
    q_pre = Qmarg'*coef[pos.begq : pos.endq]
    w_pre = Wmarg'*coef[pos.begw : pos.endw]
    hscale = clamp(exp(q_pre), 1e-12, 1e12)
    μ   = z_pre * hscale
    σᵤ  = sqrt(clamp(exp(w_pre), 1e-12, 1e12)) * hscale
    Λ   = μ/σᵤ
    uncondU = σᵤ* (Λ + exp(normlogpdf(Λ) - normlogcdf(Λ)))
end
function _b33o_mle_fewht(pos, coef, Zmarg, Qmarg, Wmarg)              # marg_fewht and marg_pdecay
    z_pre = Zmarg'*coef[pos.begz : pos.endz]
    q_pre = Qmarg'*coef[pos.begq : pos.endq]
    w_pre = Wmarg'*coef[pos.begw : pos.endw]
    hscale = clamp(exp(q_pre), 1e-12, 1e12)
    μ  = hscale*z_pre
    σᵤ = sqrt(clamp(exp(w_pre), 1e-12, 1e12)) * hscale
    Λ  = μ/σᵤ
    uncondU = σᵤ* (Λ + exp(normlogpdf(Λ) - normlogcdf(Λ)))
end
function _b33o_mle_pkumb90(pos, coef, Zmarg, Qmarg, Wmarg)            # marg_pkumb90
    z_pre = Zmarg'*coef[pos.begz : pos.endz]
    q_pre = Qmarg'*coef[pos.begq : pos.endq]
    w_pre = Wmarg'*coef[pos.begw : pos.endw]
    Gₜ  = 2 / (1 + clamp(exp(q_pre), 1e-12, 1e12))
    μ  = Gₜ * z_pre
    σᵤ = Gₜ * sqrt(clamp(exp(w_pre), 1e-12, 1e12))
    Λ  = μ/σᵤ
    uncondU = σᵤ* (Λ + exp(normlogpdf(Λ) - normlogcdf(Λ)))
end
_b33o_nlpdf(x) = -0.5 * x^2 - 0.5 * log(2π)                           # MCI/MSLE _normlogpdf
_b33o_nlcdf(x) = logerfcx(-x / sqrt(2)) - x^2 / 2 - log(2)            # MCI/MSLE _normlogcdf
function _b33o_uncondU_trun(coef, idx, Zmarg_mu, Zmarg_sigma)         # MCI and MSLE uncondU_trun
    k_mu = length(idx.ineff.mu)
    mu = sum(coef[idx.ineff.mu[j]] * Zmarg_mu[j] for j in 1:k_mu)
    k_sigma = length(idx.ineff.sigma_u)
    ln_sigma_u_sq = sum(coef[idx.ineff.sigma_u[j]] * Zmarg_sigma[j] for j in 1:k_sigma)
    sigma = clamp(exp(0.5 * ln_sigma_u_sq), oftype(ln_sigma_u_sq, 1e-12), oftype(ln_sigma_u_sq, 1e12))
    Λ = mu / sigma
    return sigma * (Λ + exp(_b33o_nlpdf(Λ) - _b33o_nlcdf(Λ)))
end
function _b33o_base_Eu_star(coef, idx)                                # MCI and MSLE _base_Eu_star(TN)
    mu = coef[idx.ineff.mu[1]]
    sigma = clamp(exp(0.5 * coef[idx.ineff.sigma_u[1]]), oftype(coef[idx.ineff.sigma_u[1]], 1e-12),
                  oftype(coef[idx.ineff.sigma_u[1]], 1e12))
    Λ = mu / sigma
    return sigma * (Λ + exp(_b33o_nlpdf(Λ) - _b33o_nlcdf(Λ)))
end
_b33o_uncondU_scaling(coef, idx, z) = exp(sum(coef[idx.delta[j]] * z[j] for j in 1:length(z))) * _b33o_base_Eu_star(coef, idx)

# ---- the kernels as their callers use them, at a designed Λ ------------------------------------
# kind: (backend, kernel); L: the target Λ = μ/σᵤ; g = ln σᵤ² (coefficient on a z of 1); q, dl: the
# scaling variable and its coefficient. Returns the package kernel and its old copy as functions of
# the z-row that get_marg / marginal_effects differentiate, the row, and the 512-bit E and ∂E/∂z.
const _B33_KINDS = (:mle_trun, :mle_scal, :mle_fewht, :mle_pdecay, :mle_pkumb90,
                    :mci_trun, :mci_scal, :msle_trun, :msle_scal)
function _b33_case(kind, L, g; q = 0.4, dl = 0.3, T = Float64)
    mle = kind in (:mle_trun, :mle_scal, :mle_fewht, :mle_pdecay, :mle_pkumb90)
    s = mle ? sqrt(clamp(exp(g), 1e-12, 1e12)) : clamp(exp(0.5 * g), 1e-12, 1e12)
    d = T(L * s); g = T(g); q = T(q); dl = T(dl)
    sB = mle ? sqrt(exp(BigFloat(g))) : exp(BigFloat(g) / 2)
    LB = BigFloat(d) / sB; m = _b33_m(LB); mp = _b33_mp(LB); gB = BigFloat(g)
    if kind in (:mle_trun, :mci_trun, :msle_trun)
        c = T[d, g]; z0 = T[1, 1]
        if kind === :mle_trun
            pos = (begz = 1, endz = 1, begw = 2, endw = 2)
            f = z -> _B33M.marg_trun(pos, c, z[1:1], z[2:2])
            fo = z -> _b33o_mle_trun(pos, c, z[1:1], z[2:2])
        else
            B = kind === :mci_trun ? _B33C : _B33S
            idx = (ineff = (mu = [1], sigma_u = [2]),)
            f = z -> B.uncondU_trun(c, idx, z[1:1], z[2:2])
            fo = z -> _b33o_uncondU_trun(c, idx, z[1:1], z[2:2])
        end
        return f, fo, z0, sB * m, [BigFloat(d) * mp, gB / 2 * sB * (m - LB * mp)]
    elseif mle
        pos = (begz = 1, endz = 1, begq = 2, endq = 2, begw = 3, endw = 3)
        c = T[d, dl, g]; z0 = T[1, q, 1]
        K, Ko = kind === :mle_scal ? (_B33M.marg_scal, _b33o_mle_scal) :
                kind === :mle_fewht ? (_B33M.marg_fewht, _b33o_mle_fewht) :
                kind === :mle_pdecay ? (_B33M.marg_pdecay, _b33o_mle_fewht) : (_B33M.marg_pkumb90, _b33o_mle_pkumb90)
        f = z -> K(pos, c, z[1:1], z[2:2], z[3:3]); fo = z -> Ko(pos, c, z[1:1], z[2:2], z[3:3])
        e = exp(BigFloat(dl) * BigFloat(q))
        h, dh = kind === :mle_pkumb90 ? (2 / (1 + e), -2 * BigFloat(dl) * e / (1 + e)^2) : (e, BigFloat(dl) * e)
        return f, fo, z0, h * sB * m, [h * BigFloat(d) * mp, dh * sB * m, h * gB / 2 * sB * (m - LB * mp)]
    else
        B = kind === :mci_scal ? _B33C : _B33S
        TN = kind === :mci_scal ? _B33C.TruncatedNormal_MCI() : _B33S.TruncatedNormal_MSLE()
        idx = (delta = [1], ineff = (mu = [2], sigma_u = [3]))
        c = T[dl, d, g]; z0 = T[q]
        f = z -> B.uncondU_scaling(c, idx, z, TN); fo = z -> _b33o_uncondU_scaling(c, idx, z)
        E = exp(BigFloat(dl) * BigFloat(q)) * sB * m
        return f, fo, z0, E, [BigFloat(dl) * E]
    end
end

@testset "truncated-normal E(u) kernels behind result.marginal" begin
    setprecision(BigFloat, 512) do

    @testset "the nine kernels below Λ = -40: value and z-gradient against 512 bits" begin
        # before: relative errors of 5e-12 to 2.3e-10 (value) and 2e-10 to 1.2e-9 (gradient) at
        # Λ = -45, 5e-5 and up to 3e-4 at -1e3, and E(u) < 0 at -1e5 in all nine
        for kind in _B33_KINDS
            for L in (-45.0, -1e3, -1e5)
                ev = 0.0; eg = 0.0; pos = true
                for g in (-2.0, -20.0)                         # σᵤ = 0.37 and 4.5e-5
                    f, _, z0, E, G = _b33_case(kind, L, g)
                    v = f(z0)
                    pos &= v > 0
                    ev = max(ev, _b33_rel(v, E)); eg = max(eg, _b33_relv(ForwardDiff.gradient(f, z0), G))
                end
                @test pos && ev <= 1e-13
                @test eg <= 1e-13
            end
        end
    end

    @testset "the switch at Λ = -40" begin
        # σᵤ = 1 and h = 1, so Λ is exactly the designed number. Just below -40 the stable form
        # (the old one is off by 3.7e-11 to 2.2e-10 in the value there); from -40 up, the old
        # expression, bit for bit; the jump at the switch is the old form's own error.
        for kind in _B33_KINDS
            fb, _, zb, Eb, Gb = _b33_case(kind, prevfloat(-40.0), 0.0; q = 0.0)
            fa, fao, za, Ea, Ga = _b33_case(kind, nextfloat(-40.0), 0.0; q = 0.0)
            vb = fb(zb); gb = ForwardDiff.gradient(fb, zb); va = fa(za); ga = ForwardDiff.gradient(fa, za)
            @test _b33_rel(vb, Eb) <= 1e-14 && _b33_relv(gb, Gb) <= 1e-14
            @test va === fao(za) && isequal(ga, ForwardDiff.gradient(fao, za))
            @test abs(vb - va) <= 1e-9 * abs(va) && maximum(abs.(gb .- ga)) <= 2e-9 * maximum(abs.(ga))
        end
    end

    @testset "Λ ≥ -40: the old expression, bit for bit (value and z-gradient)" begin
        # (Λ = μ/σᵤ is formed in floating point, so the grid stays an ulp-safe distance above -40;
        # the switch itself is pinned above with Λ exact)
        for kind in _B33_KINDS
            nv = 0; ng = 0; npt = 0
            for g in (-2.0, 0.0, -20.0), L in (-39.9999, -39.9, -25.0, -10.0, -5.0, -0.5, 0.0, 2.0)
                f, fo, z0 = _b33_case(kind, L, g)
                npt += 1
                nv += !(f(z0) === fo(z0))
                ng += !isequal(ForwardDiff.gradient(f, z0), ForwardDiff.gradient(fo, z0))
            end
            @test npt == 24 && nv == 0
            @test ng == 0
        end
    end

    @testset "Float32 coefficients (MCI and MSLE kernels)" begin
        for kind in (:mci_trun, :msle_trun, :mci_scal, :msle_scal)
            for L in (-1e3, -1e5)                              # before: 6e-5 off at -1e3, < 0 at -1e5
                f, _, z0, E, G = _b33_case(kind, L, -2.0; T = Float32)
                @test _b33_rel(f(z0), E) <= 1e-6 && _b33_relv(ForwardDiff.gradient(f, z0), G) <= 1e-5
            end
            for L in (-39.9, -5.0)
                f, fo, z0 = _b33_case(kind, L, -2.0; T = Float32)
                @test f(z0) === fo(z0) && isequal(ForwardDiff.gradient(f, z0), ForwardDiff.gradient(fo, z0))
            end
        end
    end

    # ---- the path a user sees: result.marginal of a held fit -----------------------------------
    held(spec, meth, θ) = _b33_quiet() do
        sfmodel_fit(spec = spec, method = meth, init = θ,
                    optim_options = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,)),
                    marginal = true, show_table = false, verbose = false)
    end

    @testset "panel_TFE (MLE): one Λ for the whole sample" begin
        # panel_TFE with a truncated normal under MLE at the σᵤ² clamp: μ = -0.1, σᵤ = 1e-6, Λ = -1e5
        rp = Xoshiro(2021); nf, nt = 20, 5; npp = nf * nt
        idp = repeat(1:nf, inner = nt); XP = hcat(randn(rp, npp), randn(rp, npp)); ZP = hcat(randn(rp, npp))
        yP = repeat(0.5 .* randn(rp, nf), inner = nt) .+ XP * [0.6, -0.4] .+ 0.3 .* randn(rp, npp) .-
             exp.(0.3 .* ZP[:, 1]) .* repeat(0.2 .+ 0.5 .* abs.(randn(rp, nf)), inner = nt)
        specP = sfmodel_spec(datatype = :panel_TFE, depvar = yP, frontier = XP, zvar = ZP, id = idp,
                             noise = :Normal, ineff = :TruncatedNormal)
        rP = held(specP, sfmodel_method(method = :MLE),
                  sfmodel_init(spec = specP, frontier = [0.6, -0.4], scaling = [0.3], mu = -0.1,
                               ln_sigma_u_sq = -33.0, ln_sigma_v_sq = log(0.09)))
        refP = [BigFloat(0.3) * exp(BigFloat(0.3) * BigFloat(ZP[i, 1])) * BigFloat(1e-6) * _b33_m(BigFloat(-0.1) / BigFloat(1e-6))
                for i in 1:npp]
        meP = vec(Matrix(rP.marginal))
        @test all(>(0), meP)                                                # before: all 100 negative
        @test maximum(_b33_rel(meP[i], refP[i]) for i in 1:npp) <= 1e-13
    end

    end   # setprecision
end
