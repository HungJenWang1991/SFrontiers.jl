# test_marginal_clamps.jl — the marginal-effect helpers clamp their parameters as the likelihood does.
# The Gamma, Erlang, HalfNormal, TruncatedNormal, Rayleigh and Lognormal marginal-effect helpers
# already clamped their parameters exactly as the likelihood does; the Exponential, Weibull and
# Lomax helpers used a raw exp() against clamped likelihoods, so at a saturated optimum the
# marginal mean and its ForwardDiff gradient described a different effective model from the one
# the NLL maximised — E(u) = Inf or 0, and a NaN or sign-flipped marginal effect. These testsets
# pin the six uncondU_* and six _base_Eu_star helpers (three distributions x two backends x two
# helper families) to the likelihood's own bounds, and pin the bounds themselves so a kernel
# change trips here rather than silently desynchronising the marginal table.
using Test
using SFrontiers
using Optim: NelderMead, Newton
using Random
using ForwardDiff: gradient
using SpecialFunctions: loggamma

@testset "Marginal-effect clamp alignment" begin
    _MCIB = SFrontiers.MCI_Backend
    _MSLEB = SFrontiers.MSLE_Backend

    # Calling convention: each uncondU_* sums only the first length(idx.ineff.<param>) entries of
    # its own Zmarg, so with length-1 index blocks EVERY channel takes the same ones basis [1.0].
    idx_e = (ineff=(sigma_sq=[1],),)
    idx_w = (ineff=(lambda=[1], k=[2]),)
    idx_l = (ineff=(ln_lambda=[1], alpha=[2]),)
    ONE = [1.0]

    # ---- (0) the bounds the helpers hard-code are still the bounds the kernels use --------------
    # The helpers take no constants NamedTuple (unlike the panel backend, which threads `c`), so
    # the numbers are literals in two places. If a kernel bound ever moves, fail here.
    let mM = _MCIB.MCIModel(_MCIB.NormalNoise_MCI(), _MCIB.Exponential_MCI(), _MCIB.NoCopula_MCI()),
        mS = _MSLEB.MSLEModel(_MSLEB.NormalNoise_MSLE(), _MSLEB.Exponential_MSLE(), _MSLEB.NoCopula_MSLE())
        # The Exponential's sigma_u_sq and Weibull's lambda come from make_constants' BASE tuple, not make_ineff_constants
        @test _MCIB.make_constants(mM, Float64).σ_floor == 1e-12
        @test _MCIB.make_constants(mM, Float64).σ_ceil == 1e12
        @test _MSLEB.make_constants(mS, Float64).σ_floor == 1e-12
        @test _MSLEB.make_constants(mS, Float64).σ_ceil == 1e12
    end
    @test _MCIB.make_ineff_constants(_MCIB.Weibull_MCI(), Float64).k_floor == 0.1
    @test _MCIB.make_ineff_constants(_MCIB.Weibull_MCI(), Float64).k_ceil == 10.0
    @test _MSLEB.make_ineff_constants(_MSLEB.Weibull_MSLE(), Float64).k_floor == 0.1
    @test _MSLEB.make_ineff_constants(_MSLEB.Weibull_MSLE(), Float64).k_ceil == 10.0
    let cl = _MCIB.make_ineff_constants(_MCIB.Lomax_MCI(), Float64),
        sl = _MSLEB.make_ineff_constants(_MSLEB.Lomax_MSLE(), Float64)
        @test (cl.lambda_floor, cl.lambda_ceil, cl.α_floor, cl.α_ceil) == (1e-10, 1e10, 0.1, 100.0)
        @test (sl.lambda_floor, sl.lambda_ceil, sl.α_floor, sl.α_ceil) == (1e-10, 1e10, 0.1, 100.0)
    end
    # The copula, gamma and panel constants that the boundary verdict also reads.
    for (B, cops) in ((_MCIB, (_MCIB.GaussianCopula_MCI(), _MCIB.ClaytonCopula_MCI(),
                               _MCIB.Clayton90Copula_MCI(), _MCIB.GumbelCopula_MCI())),
                      (_MSLEB, (_MSLEB.GaussianCopula_MSLE(), _MSLEB.ClaytonCopula_MSLE(),
                                _MSLEB.Clayton90Copula_MSLE(), _MSLEB.GumbelCopula_MSLE())))
        cG, cC, c9, cU = (B.make_copula_constants(cop, Float64) for cop in cops)
        @test cG.rho_max == 0.999
        @test (cC.clayton_rho_floor, cC.clayton_rho_max) == (1e-6, 50.0)
        @test (c9.clayton_rho_floor, c9.clayton_rho_max) == (1e-6, 50.0)
        @test cU.gumbel_rho_max == 50.0
    end
    let cg = _MCIB.make_ineff_constants(_MCIB.Gamma_MCI(), Float64)
        @test (cg.k_floor, cg.k_ceil, cg.theta_floor, cg.theta_ceil) == (0.1, 100.0, 1e-12, 1e12)
    end
    let panB = SFrontiers.Panel_Backend
        pc = panB.make_panel_constants(panB.PanelModel(panB.NormalNoise_Panel(), panB.Lomax_Panel()), Float64, 4)
        @test (pc.k_floor, pc.weibull_k_ceil, pc.gamma_k_ceil) == (0.1, 10.0, 100.0)
        @test (pc.alpha_floor, pc.alpha_ceil, pc.xm_floor, pc.xm_ceil) == (0.1, 100.0, 1e-10, 1e10)
    end

    for B in (_MCIB, _MSLEB)
        # ---- (1) Exponential, E(u) = sqrt(sigma_u_sq), sigma_u_sq in [1e-12, 1e12] -------------
        @test B.uncondU_expo([800.0], idx_e, ONE) == 1.0e6        # was Inf
        @test B.uncondU_expo([-800.0], idx_e, ONE) == 1.0e-6      # was exactly 0.0
        @test B.uncondU_expo([0.5], idx_e, ONE) ≈ exp(0.25)       # interior: unchanged

        # ---- (2) Weibull, E(u) = lambda*Gamma(1 + 1/k), lambda in [1e-12,1e12], k in [0.1,10] --
        @test B.uncondU_weibull([800.0, 0.0], idx_w, ONE, ONE) == 1.0e12      # was Inf
        @test B.uncondU_weibull([-800.0, 0.0], idx_w, ONE, ONE) == 1.0e-12    # was 0.0
        # k ceiling: a k of 50 must give exactly what k = 10 gives
        @test B.uncondU_weibull([log(2.0), log(50.0)], idx_w, ONE, ONE) ≈ 2 * exp(loggamma(1.1))
        @test B.uncondU_weibull([log(2.0), log(50.0)], idx_w, ONE, ONE) ==
              B.uncondU_weibull([log(2.0), log(10.0)], idx_w, ONE, ONE)
        # k floor: the worse half of the defect — 1/k overflowed loggamma and the column went NaN
        @test B.uncondU_weibull([log(2.0), -800.0], idx_w, ONE, ONE) ≈ 2 * exp(loggamma(11.0))
        @test B.uncondU_weibull([log(2.0), log(0.05)], idx_w, ONE, ONE) ≈ 2 * exp(loggamma(11.0))
        # interior: unchanged, and genuinely different from the clamped value
        @test B.uncondU_weibull([log(2.0), log(5.0)], idx_w, ONE, ONE) ≈ 2 * exp(loggamma(1.2))
        @test B.uncondU_weibull([log(2.0), log(3.0)], idx_w, ONE, ONE) ≈ 2 * exp(loggamma(1 + 1/3))
        @test B.uncondU_weibull([log(2.0), log(5.0)], idx_w, ONE, ONE) !=
              B.uncondU_weibull([log(2.0), log(50.0)], idx_w, ONE, ONE)

        # ---- (3) Lomax, E(u) = lambda/(alpha-1), lambda in [1e-10,1e10], alpha in [0.1,100] ----
        @test B.uncondU_lomax([log(2.0), log(500.0)], idx_l, ONE, ONE) == 2 / 99   # was 2/499
        @test B.uncondU_lomax([log(2.0), 800.0], idx_l, ONE, ONE) == 2 / 99        # was 0.0
        @test B.uncondU_lomax([log(2.0), log(500.0)], idx_l, ONE, ONE) ==
              B.uncondU_lomax([log(2.0), 800.0], idx_l, ONE, ONE)
        @test B.uncondU_lomax([log(1e12), log(2.0)], idx_l, ONE, ONE) == 1.0e10    # was 9.99999e11
        @test B.uncondU_lomax([-800.0, log(2.0)], idx_l, ONE, ONE) == 1.0e-10      # was 0.0
        @test B.uncondU_lomax([log(2.0), log(3.0)], idx_l, ONE, ONE) ≈ 1.0         # interior
        # the alpha <= 1 branch SURVIVES the clamp: the kernel's alpha floor is 0.1, below 1, so
        # the Lomax mean genuinely does not exist there — the panel backend reaches this case too
        @test B.uncondU_lomax([log(2.0), log(0.5)], idx_l, ONE, ONE) == Inf
        @test B.uncondU_lomax([log(2.0), -800.0], idx_l, ONE, ONE) == Inf

        # ---- (4) the scaling model's _base_Eu_star twins carry the same clamps -----------------
        TE = B === _MCIB ? _MCIB.Exponential_MCI() : _MSLEB.Exponential_MSLE()
        TW = B === _MCIB ? _MCIB.Weibull_MCI() : _MSLEB.Weibull_MSLE()
        TL = B === _MCIB ? _MCIB.Lomax_MCI() : _MSLEB.Lomax_MSLE()
        @test B._base_Eu_star([800.0], idx_e, TE) == 1.0e6
        @test B._base_Eu_star([-800.0], idx_e, TE) == 1.0e-6
        @test B._base_Eu_star([0.5], idx_e, TE) ≈ exp(0.25)
        @test B._base_Eu_star([800.0, 0.0], idx_w, TW) == 1.0e12
        @test B._base_Eu_star([-800.0, 0.0], idx_w, TW) == 1.0e-12    # lambda FLOOR, not only the ceiling
        @test B._base_Eu_star([log(2.0), log(50.0)], idx_w, TW) ≈ 2 * exp(loggamma(1.1))
        @test B._base_Eu_star([log(2.0), -800.0], idx_w, TW) ≈ 2 * exp(loggamma(11.0))
        @test B._base_Eu_star([log(2.0), log(3.0)], idx_w, TW) ≈ 2 * exp(loggamma(1 + 1/3))
        @test B._base_Eu_star([log(2.0), log(500.0)], idx_l, TL) == 2 / 99
        @test B._base_Eu_star([log(1e12), log(2.0)], idx_l, TL) == 1.0e10
        @test B._base_Eu_star([-800.0, log(2.0)], idx_l, TL) == 1.0e-10   # lambda FLOOR
        @test isnan(B._base_Eu_star([log(2.0), log(0.5)], idx_l, TL))
        @test isnan(B._base_Eu_star([log(2.0), 0.0], idx_l, TL))          # α = 1 exactly
        # every uncondU_* / _base_Eu_star pair must agree where E(u) exists, since both feed marginal effects
        @test B._base_Eu_star([800.0], idx_e, TE) == B.uncondU_expo([800.0], idx_e, ONE)
        @test B._base_Eu_star([log(2.0), -800.0], idx_w, TW) ==
              B.uncondU_weibull([log(2.0), -800.0], idx_w, ONE, ONE)
        @test B._base_Eu_star([log(2.0), 800.0], idx_l, TL) ==
              B.uncondU_lomax([log(2.0), 800.0], idx_l, ONE, ONE)

        # ---- (5) the derivative path: marginal effects are ForwardDiff of these very functions --
        # A saturated channel must give exactly 0.0, not NaN/Inf, and an unsaturated channel must
        # keep its analytic value; interior gradients must not move.
        idx2_e = (ineff=(sigma_sq=[1, 2],),)
        idx2_w = (ineff=(lambda=[1, 2], k=[3, 4]),)
        idx2_l = (ineff=(ln_lambda=[1, 2], alpha=[3, 4]),)
        ge = gradient(z -> B.uncondU_expo([800.0, 0.3], idx2_e, z), [1.0, 1.0])
        @test all(isfinite, ge) && ge == [0.0, 0.0]                       # was [NaN, NaN]
        ge2 = gradient(z -> B.uncondU_expo([-800.0, 0.3], idx2_e, z), [1.0, 1.0])
        @test all(isfinite, ge2) && ge2 == [0.0, 0.0]                     # was [NaN, NaN]
        gw = gradient(z -> B.uncondU_weibull([800.0, 0.3, log(3.0), 0.2], idx2_w, z, z), [1.0, 1.0])
        @test all(isfinite, gw)                                           # was [Inf, Inf]
        # k below its floor: the k channel goes to exactly 0 while the interior lambda channel
        # keeps its analytic slope, which is the whole point of clamping only what saturates
        gwf = gradient(z -> B.uncondU_weibull([log(2.0), 0.0, -800.0, 0.0], idx2_w, z, z), [1.0, 1.0])
        @test all(isfinite, gwf)                                          # was [NaN, NaN]
        @test gwf ≈ [log(2.0) * 2 * exp(loggamma(11.0)), 0.0]
        # alpha saturated at its ceiling: that channel contributes exactly zero while the interior
        # lambda channel keeps its analytic slope. Pre-fix this gradient had the OPPOSITE SIGN on
        # its first component, so `all(isfinite, gl)` alone would not have caught the defect.
        gl = gradient(z -> B.uncondU_lomax([log(2.0), 0.3, log(500.0), 0.2], idx2_l, z, z), [1.0, 1.0])
        @test all(isfinite, gl)
        @test gl ≈ (2 * exp(0.3) / 99) .* [log(2.0), 0.3]
        # interior gradients keep the unclamped analytic value
        gi = gradient(z -> B.uncondU_expo([log(0.7), 0.3], idx2_e, z), [1.0, 1.0])
        @test gi ≈ 0.5 * sqrt(0.7 * exp(0.3)) .* [log(0.7), 0.3]

        # ---- (6) Lomax with α ≤ 1: E(u) is +∞, and the marginal effect must not read 0.0 ----
        # The kernel's α floor is 0.1, so a converged fit can sit below 1. Returning a bare
        # `oftype(alpha, Inf)` made E(u) a CONSTANT in z, so ForwardDiff reported ∂E(u)/∂z = 0
        # and the table printed exactly 0.0 — a finite, actionable and false number from a
        # converged fit. The value is still +∞; only the derivative changed, to NaN.
        for la in (log(0.3), log(0.5), log(0.9), 0.0)
            f = z -> B.uncondU_lomax([log(2.0), 0.0, la, 0.0], idx2_l, z, z)
            @test f([1.0, 1.0]) == Inf                       # the value is unchanged
            g = gradient(f, [1.0, 1.0])
            @test all(isnan, g)                              # was [0.0, 0.0]
            @test !any(iszero, g)
        end
        # strictly above 1 is untouched
        let gok = gradient(z -> B.uncondU_lomax([log(2.0), 0.0, log(3.0), 0.0], idx2_l, z, z),
                           [1.0, 1.0])
            @test all(isfinite, gok)
        end
        # The scaling path reports NaN as well: E(u) does not exist for α ≤ 1, so its derivative
        # has no value, in every column, also where δ_j = 0.
        let idxs = (delta=[1, 2], ineff=(ln_lambda=[3], alpha=[4])),
            TLs = B === _MCIB ? _MCIB.Lomax_MCI() : _MSLEB.Lomax_MSLE()
            gs = gradient(z -> B.uncondU_scaling([0.4, -0.2, log(2.0), log(0.5)], idxs, z, TLs),
                          [1.0, 1.0])
            @test isnan(B.uncondU_scaling([0.4, -0.2, log(2.0), log(0.5)], idxs, [1.0, 1.0], TLs))
            @test all(isnan, gs)
            g0 = gradient(z -> B.uncondU_scaling([0.0, -0.2, log(2.0), log(0.5)], idxs, z, TLs),
                          [1.0, 1.0])
            @test all(isnan, g0)                                  # δ_1 = 0
            # α > 1 is unchanged: E(u) = h(z)·λ/(α − 1) and its derivative in z, in closed form
            idx1 = (delta = [1], ineff = (ln_lambda = [2], alpha = [3]))
            c1 = [0.5, -1.0, log(1.5)]
            @test B.uncondU_scaling(c1, idx1, [0.3], TLs) ≈ exp(0.15) * exp(-1.0) / 0.5 rtol = 1e-12
            @test gradient(z -> B.uncondU_scaling(c1, idx1, z, TLs), [0.3])[1] ≈
                  0.5 * exp(0.15) * exp(-1.0) / 0.5 rtol = 1e-12
        end
    end

    # ---- (6b) the panel Lomax helper -------------------------------------------------------
    # The panel Lomax helper returns NaN for α ≤ 1, like the cross-sectional kernels: E(u_it) does
    # not exist there.
    let BP = SFrontiers.Panel_Backend,
        cp = BP.make_panel_constants(BP.PanelModel(BP.NormalNoise_Panel(), BP.Lomax_Panel()),
                                     Float64, 4),
        idxp = (delta=[1, 2], ln_lambda=3, ln_alpha=4)
        for la in (log(0.5), log(0.9), 0.0)
            f = z -> BP._panel_uncondU_lomax([0.4, -0.2, log(2.0), la], idxp, z, cp)
            @test isnan(f([1.0, 1.0]))
            gp = gradient(f, [1.0, 1.0])
            @test all(isnan, gp)
        end
        # α > 1 is untouched, value and gradient
        let f = z -> BP._panel_uncondU_lomax([0.4, -0.2, log(2.0), log(1.5)], idxp, z, cp)
            @test f([1.0, 1.0]) ≈ exp(0.2) * 2.0 / 0.5
            @test all(isfinite, gradient(f, [1.0, 1.0]))
        end
    end

    # ---- (6) end to end: a saturated fit's marginal table now matches the flat likelihood -------
    # The fit path (result.marginal) and predict(:marginal) share these helpers, so one
    # forced-coefficient result object exercises both routes.
    Random.seed!(515)
    N = 60
    X = hcat(ones(N), randn(N))
    z = rand(N) .+ 1.0                                   # z in [1, 2), so a large gamma saturates
    y = X * [0.8, 0.5] .- abs.(0.3 .* randn(N)) .+ 0.25 .* randn(N)
    Zm = reshape(z, :, 1)
    op = sfmodel_opt(warmstart_solver=NelderMead(), warmstart_opt=(iterations=20, g_abstol=1e-3),
                     main_solver=Newton(), main_opt=(iterations=20, g_abstol=1e-6))
    sp = sfmodel_spec(depvar=y, frontier=X, zvar=Zm, noise=:Normal, ineff=:Exponential,
                      hetero=[:sigma_u_sq])
    r = sfmodel_fit(spec=sp, method=sfmodel_method(method=:MCI, n_draws=64, multiRand=false),
                    init=sfmodel_init(spec=sp, frontier=X \ y, ln_sigma_u_sq=[-1.0],
                                      ln_sigma_v_sq=log(0.0625), message=false),
                    optim_options=op, marginal=true, show_table=false, verbose=false)
    idxf = r.spec.mci_spec.idx
    cf = collect(float.(r.coeff))
    # interior fit: predict(:marginal) still reproduces the fit's own table
    pm_int = predict(r; zvar=Zm, what=:marginal)
    @test all(isfinite, pm_int)
    @test maximum(abs.(vec(pm_int) .- collect(float.(r.marginal[!, 1])))) < 1e-8

    # force the sigma_u_sq channel above the ceiling: z'gamma in [30, 60) > log(1e12) = 27.63
    cS = copy(cf); cS[idxf.ineff.sigma_sq[1]] = 30.0
    cB = copy(cS); cB[idxf.ineff.sigma_sq[1]] = 60.0
    msp = _MCIB._assemble_MCI_spec(r.spec.mci_spec,
                                   _MCIB.sfmodel_method(method=:MCI, n_draws=64, multiRand=false))
    nll = p -> _MCIB.MCI_nll(msp, p)
    @test nll(cS) == nll(cB)                              # the likelihood really is flat there
    rS = SFrontiers.SFResult(merge(r.nt, (coeff=cS,)), r.backend, r.spec)
    pmS = predict(rS; zvar=Zm, what=:marginal)
    @test all(isfinite, pmS)
    @test all(iszero, pmS)                                # flat likelihood => zero marginal effect
    @test _MCIB.uncondU_expo(cS, idxf, [z[1]]) == 1.0e6   # was sqrt(exp(30*z[1])) = 3.6e10

    # force the sigma_u_sq channel below the floor: E(u) pinned at 1e-6, marginal exactly 0
    cF = copy(cf); cF[idxf.ineff.sigma_sq[1]] = -30.0
    rF = SFrontiers.SFResult(merge(r.nt, (coeff=cF,)), r.backend, r.spec)
    pmF = predict(rF; zvar=Zm, what=:marginal)
    @test all(isfinite, pmF) && all(iszero, pmF)
    @test _MCIB.uncondU_expo(cF, idxf, [z[1]]) == 1.0e-6  # was sqrt(exp(-30*z[1])) = 2.8e-11, and
                                                          # exactly 0.0 further below the floor

    # ---- (7) a saturated Weibull k no longer turns the whole marginal table into NaN -----------
    # Reuses section 6's X, y and z deliberately, so the two end-to-end blocks share one seeded DGP.
    # This is the only place proving the k floor — the worse half of the defect — reaches the user
    # through predict(:marginal).
    Zw = hcat(ones(N), z)                                 # lambda homoscedastic, k heteroscedastic
    spw = sfmodel_spec(depvar=y, frontier=X, zvar=Zw, noise=:Normal, ineff=:Weibull, hetero=[:k])
    rw = sfmodel_fit(spec=spw, method=sfmodel_method(method=:MCI, n_draws=64, multiRand=false),
                     init=sfmodel_init(spec=spw, frontier=X \ y, ln_lambda=log(0.3),
                                       ln_k=[-1.0, 0.0], ln_sigma_v_sq=log(0.0625), message=false),
                     optim_options=op, marginal=true, show_table=false, verbose=false)
    idxw = rw.spec.mci_spec.idx
    cw = collect(float.(rw.coeff))
    cwS = copy(cw); cwS[idxw.ineff.k[1]] = -20.0; cwS[idxw.ineff.k[2]] = 0.0
    rw2 = SFrontiers.SFResult(merge(rw.nt, (coeff=cwS,)), rw.backend, rw.spec)
    pmw = predict(rw2; zvar=Zw, what=:marginal)
    @test all(isfinite, pmw)                              # was an all-NaN matrix
    @test _MCIB.uncondU_weibull(cwS, idxw, [1.0], Zw[1, :]) ≈
          exp(cw[idxw.ineff.lambda[1]]) * exp(loggamma(11.0))
end
