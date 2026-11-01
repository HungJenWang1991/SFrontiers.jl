# Parameters at the numerical bounds of the likelihoods. A parameter at or beyond the bound its likelihood applies
# (a variance or scale, the shape of the Weibull, gamma and Lomax, the ceiling of the Clayton, Clayton 90° and Gumbel
# copulas), within a relative 1e-6 of the bound on the natural scale, is listed in `bound_check`, the fit reads
# redflag 1, and `hessian_flag` keeps the Hessian part of redflag; fits inside every bound are unchanged. Most fits
# are held at their start values (no warmstart, no Newton iteration), so the data only define the model.

using Test
using SFrontiers
using Random
using Optim: Newton, NelderMead
using SpecialFunctions: erfinv
using Statistics: var

@testset "parameter bounds: verdict and reporting" begin
    hold = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,))
    fitat(sp, m, p; nd = 64, verbose = false) =
        sfmodel_fit(spec = sp,
                    method = m === :MLE ? sfmodel_method(method = :MLE) : sfmodel_method(method = m, n_draws = nd),
                    init = p, optim_options = hold, marginal = false, show_table = false, verbose = verbose)
    # Weibull data
    Random.seed!(20261004); N = 500; x = randn(N); X = hcat(ones(N), x); u = (-log.(1 .- rand(N))) .^ (1/1.5)
    y = X * [1.0, 0.5] .+ 0.5 .* randn(N) .- u
    spW = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :Weibull, type = :prod)
    pW(lk) = sfmodel_init(spec = spW, frontier = X \ y, ln_lambda = 0.0, ln_k = lk, ln_sigma_v_sq = log(0.25))
    # Half-normal data for the other families
    rng3 = Xoshiro(3101); N3 = 300; x3 = randn(rng3, N3); z3 = randn(rng3, N3); X3 = hcat(ones(N3), x3)
    y3 = X3 * [1.0, 0.5] .+ 0.3 .* randn(rng3, N3) .- abs.(0.8 .* randn(rng3, N3))
    # The Weibull with a heteroscedastic shape k_i = 5·exp(z_i), which reaches 10 where z_i ≥ log 2
    zW = randn(Xoshiro(3102), 500)
    spH = sfmodel_spec(depvar = y, frontier = X, zvar = hcat(ones(500), zW), noise = :Normal, ineff = :Weibull,
                       hetero = [:k], type = :prod)
    pH = sfmodel_init(spec = spH, frontier = X \ y, ln_lambda = 0.0, ln_k = [log(5.0), 1.0], ln_sigma_v_sq = log(0.25))
    # The two result fields of the verdict, with their types.
    _bnd_fields(r) = r.hessian_flag isa Int && r.bound_check isa Vector{SFrontiers.SFRedflag.SFBoundEntry}

    @testset "the bounds come from the likelihood's constants" begin
        mciB = SFrontiers.MCI_Backend
        msleB = SFrontiers.MSLE_Backend
        panB = SFrontiers.Panel_Backend
        trip(list) = [(it.parameter, it.lo, it.hi) for it in list]
        sp3(noise, ineff; copula = :None) =
            sfmodel_spec(depvar = y3, frontier = X3, noise = noise, ineff = ineff, copula = copula, type = :prod)
        mci(sp) = trip(mciB._bound_list(mciB._assemble_MCI_spec(sp.mci_spec,
                                                                 sfmodel_method(method = :MCI, n_draws = 8))))
        msle(sp) = trip(msleB._bound_list(msleB._assemble_MSLE_spec(sp.msle_spec,
                                                                     sfmodel_method(method = :MSLE, n_draws = 8))))
        sv = (:sigma_v, 1e-12, 1e12)
        cases = [(:Normal, :HalfNormal, :None) => [sv, (:sigma_u, 1e-12, 1e12)],
                 (:StudentT, :HalfNormal, :None) => [sv, (:sigma_u, 1e-12, 1e12)],
                 (:Laplace, :HalfNormal, :None) => [(:b, 1e-12, 1e12), (:sigma_u, 1e-12, 1e12)],
                 (:Normal, :TruncatedNormal, :None) => [sv, (:sigma_u, 1e-12, 1e12)],
                 (:Normal, :Exponential, :None) => [sv, (:sigma_u_sq, 1e-12, 1e12)],
                 (:Normal, :Lognormal, :None) => [sv, (:sigma, 1e-12, 1e12)],
                 (:Normal, :Rayleigh, :None) => [sv, (:sigma, 1e-12, 1e12)],
                 (:Normal, :Weibull, :None) => [sv, (:lambda, 1e-12, 1e12), (:k, 0.1, 10.0)],
                 (:Normal, :Lomax, :None) => [sv, (:lambda, 1e-10, 1e10), (:alpha, 0.1, 100.0)],
                 (:Normal, :Erlang2, :None) => [sv, (:theta, 1e-12, 1e12)],
                 (:Normal, :HalfNormal, :Clayton) => [sv, (:sigma_u, 1e-12, 1e12), (:rho, -Inf, 50.0)],
                 (:Normal, :HalfNormal, :Clayton90) => [sv, (:sigma_u, 1e-12, 1e12), (:rho, -Inf, 50.0)],
                 (:Normal, :HalfNormal, :Gumbel) => [sv, (:sigma_u, 1e-12, 1e12), (:rho, -Inf, 50.0)]]
        for ((noise, ineff, copula), want) in cases
            sp = sp3(noise, ineff; copula = copula)
            @test mci(sp) == want
            @test msle(sp) == want
        end
        @test mci(sp3(:Normal, :Gamma)) == [sv, (:k, 0.1, 100.0), (:theta, 1e-12, 1e12)]
        # Float32 data: the bounds are the Float32 constants of the likelihood
        sp32 = sfmodel_spec(depvar = Float32.(y3), frontier = Float32.(X3), noise = :Normal, ineff = :HalfNormal,
                            type = :prod)
        b32 = [(:sigma_v, Float64(1f-12), Float64(1f12)), (:sigma_u, Float64(1f-12), Float64(1f12))]
        @test mci(sp32) == b32
        @test msle(sp32) == b32
        # panel_TFE
        rngp = Xoshiro(3802); nf = 50; T = 4; NT = nf * T; id = repeat(1:nf, inner = T)
        xp = randn(rngp, NT); zp = randn(rngp, NT)
        yp = 0.5 .* xp .+ 0.3 .* randn(rngp, NT) .- abs.(0.5 .* randn(rngp, NT)) .* exp.(0.3 .* zp)
        pan(ineff) = trip(panB._panel_bound_list(panB._assemble_panel_spec(
            sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = reshape(xp, :, 1), zvar = reshape(zp, :, 1),
                         id = id, noise = :Normal, ineff = ineff).panel_spec,
            panB.sfmodel_panel_method(method = :MCI, n_draws = 8))))
        sv2 = (:sigma_v_sq, (1e-12)^2, (1e12)^2)
        for ineff in (:HalfNormal, :TruncatedNormal, :Lognormal, :Rayleigh)
            @test pan(ineff) == [sv2, (:sigma_u_sq, (1e-12)^2, (1e12)^2)]
        end
        @test pan(:Exponential) == [sv2, (:sigma_u_sq, 1e-12, 1e12)]
        @test pan(:Weibull) == [sv2, (:lambda, 1e-12, 1e12), (:k, 0.1, 10.0)]
        @test pan(:Lomax) == [sv2, (:lambda, 1e-10, 1e10), (:alpha, 0.1, 100.0)]
        @test pan(:Gamma) == [sv2, (:k, 0.1, 100.0), (:theta, 1e-12, 1e12)]
        @test pan(:Erlang2) == [sv2, (:theta, 1e-12, 1e12)]
    end

    @testset "a verdict for every bound, $m" for m in (:MCI, :MSLE)
        # the Weibull shape above its ceiling: a singular Hessian, so every standard error is NaN
        r = redirect_stdout(() -> fitat(spW, m, pW(log(15.0))), devnull)
        @test _bnd_fields(r)
        @test length(r.bound_check) == 1
        e = r.bound_check[1]
        @test (e.parameter, e.label, e.coefficient, e.side, e.bound, e.scalar) == (:k, "k", "ln_k", :upper, 10.0, true)
        @test e.n_at == e.n == 500 && e.share == 1.0 && e.moved && e.coef_before == log(15.0)
        @test r.redflag == 1 && r.hessian_flag == 1
        @test all(isnan, r.std_err)
        # below its floor
        r = redirect_stdout(() -> fitat(spW, m, pW(log(0.05))), devnull)
        @test _bnd_fields(r)
        @test [(e.parameter, e.side, e.bound) for e in r.bound_check] == [(:k, :lower, 0.1)]
        # inside its bounds
        r = redirect_stdout(() -> fitat(spW, m, pW(log(1.5))), devnull)
        @test _bnd_fields(r)
        @test isempty(r.bound_check)
        # the Lomax shape above 100
        spL = sfmodel_spec(depvar = y3, frontier = X3, noise = :Normal, ineff = :Lomax, type = :prod)
        pL = sfmodel_init(spec = spL, frontier = X3 \ y3, ln_lambda = 0.0, ln_alpha = log(150.0),
                          ln_sigma_v_sq = log(0.09))
        r = redirect_stdout(() -> fitat(spL, m, pL), devnull)
        @test _bnd_fields(r)
        @test [(e.parameter, e.side, e.bound) for e in r.bound_check] == [(:alpha, :upper, 100.0)]
        # the gamma shape above 100 (MCI only)
        if m === :MCI
            spG = sfmodel_spec(depvar = y3, frontier = X3, noise = :Normal, ineff = :Gamma, type = :prod)
            pG = sfmodel_init(spec = spG, frontier = X3 \ y3, ln_k = log(200.0), ln_theta = log(0.008),
                              ln_sigma_v_sq = log(0.09))
            r = redirect_stdout(() -> fitat(spG, m, pG), devnull)
            @test _bnd_fields(r)
            @test [(e.parameter, e.side, e.bound) for e in r.bound_check] == [(:k, :upper, 100.0)]
        end
        # the copula dependence above its ceiling 50
        for copula in (:Clayton, :Clayton90, :Gumbel)
            spC = sfmodel_spec(depvar = y3, frontier = X3, noise = :Normal, ineff = :HalfNormal, copula = copula,
                               type = :prod)
            pC = sfmodel_init(spec = spC, frontier = X3 \ y3, ln_sigma_u_sq = log(0.64), ln_sigma_v_sq = log(0.09),
                              theta_rho = log(60.0))
            r = redirect_stdout(() -> fitat(spC, m, pC), devnull)
            @test _bnd_fields(r)
            @test [(e.parameter, e.side, e.bound) for e in r.bound_check] == [(:rho, :upper, 50.0)]
        end
        # σᵤ = e^(−30) ≈ 9.4e-14, below its floor 1e-12
        spN = sfmodel_spec(depvar = y3, frontier = X3, noise = :Normal, ineff = :HalfNormal, type = :prod)
        pN = sfmodel_init(spec = spN, frontier = X3 \ y3, ln_sigma_u_sq = -60.0, ln_sigma_v_sq = log(0.09))
        r = redirect_stdout(() -> fitat(spN, m, pN), devnull)
        @test _bnd_fields(r)
        @test [(e.parameter, e.label, e.side, e.bound) for e in r.bound_check] == [(:sigma_u, "σᵤ", :lower, 1e-12)]
    end

    @testset "a heteroscedastic share, $m" for m in (:MCI, :MSLE)
        ki = exp.(log(5.0) .+ 1.0 .* zW)
        nhi = count(>=(10 * (1 - 1e-6)), ki)
        nlo = count(<=(0.1 * (1 + 1e-6)), ki)
        want = [(:k, :upper, false, nhi, 500)]
        nlo > 0 && pushfirst!(want, (:k, :lower, false, nlo, 500))
        r = redirect_stdout(() -> fitat(spH, m, pH), devnull)
        @test _bnd_fields(r)
        @test [(e.parameter, e.side, e.scalar, e.n_at, e.n) for e in r.bound_check] == want
        @test r.bound_check[end].share == nhi / 500
        @test r.redflag == 1
        # hessian_flag keeps the Hessian's own verdict (a variance that is not positive), not the boundary's 1
        hv = SFrontiers.SFRedflag.sf_classify_hessian(r.Hessian; message = false).hessian_flag
        @test r.hessian_flag == hv == 2
        # the variance matrix is kept as computed
        @test isequal(r.var_cov_mat,
                      SFrontiers.SFRedflag.sf_classify_hessian(r.Hessian; message = false).var_cov_matrix)
    end

    @testset "the tolerance" begin
        sbs = SFrontiers.SFRedflag.sf_bound_side
        @test sbs(10 * (1 - 5e-7), 0.1, 10.0) === :upper
        @test sbs(10 * (1 - 2e-6), 0.1, 10.0) === nothing
        @test sbs(0.1, 0.1, 10.0) === :lower
        @test sbs(NaN, 0.1, 10.0) === nothing
        @test sbs(60.0, -Inf, 50.0) === :upper
        # the Lomax shape just inside its ceiling 100
        spL = sfmodel_spec(depvar = y3, frontier = X3, noise = :Normal, ineff = :Lomax, type = :prod)
        pL(la) = sfmodel_init(spec = spL, frontier = X3 \ y3, ln_lambda = 0.0, ln_alpha = la, ln_sigma_v_sq = log(0.09))
        for m in (:MCI, :MSLE)
            r = redirect_stdout(() -> fitat(spL, m, pL(log(100 * (1 - 5e-7)))), devnull)
            @test _bnd_fields(r)
            @test [(e.parameter, e.side) for e in r.bound_check] == [(:alpha, :upper)]
            r = redirect_stdout(() -> fitat(spL, m, pL(log(100 * (1 - 1e-5)))), devnull)
            @test _bnd_fields(r)
            @test isempty(r.bound_check)
        end
    end

    @testset "the message names the parameter" begin
        # The value of f() and the text it printed.
        function _bnd_capture(f)
            path, io = mktemp()
            ret = redirect_stdout(f, io)
            close(io)
            out = read(path, String)
            rm(path)
            return ret, out
        end
        spL = sfmodel_spec(depvar = y3, frontier = X3, noise = :Normal, ineff = :Lomax, type = :prod)
        pL = sfmodel_init(spec = spL, frontier = X3 \ y3, ln_lambda = 0.0, ln_alpha = log(150.0),
                          ln_sigma_v_sq = log(0.09))
        r, out = _bnd_capture(() -> fitat(spL, :MCI, pL; verbose = true))
        @test _bnd_fields(r)
        @test occursin("at or beyond the numerical bound", out)
        @test occursin("boundary solution", out)
        @test occursin("α (coefficient ln_α) at its upper bound 100", out)
        @test occursin("See Parameter Bounds in the manual", out)
        rq, outq = _bnd_capture(() -> fitat(spL, :MCI, pL))
        @test _bnd_fields(rq)
        @test !occursin("boundary solution", outq)
        @test isequal(rq.bound_check, r.bound_check)
        BE = SFrontiers.SFRedflag.SFBoundEntry
        bc2 = BE[BE((:alpha, "α", "ln_α", :upper, 100.0, true, 500, 500, 1.0, false, NaN)),
                 BE((:sigma_u, "σᵤ", "ln_σᵤ²", :lower, 1e-12, false, 37, 500, 0.074, false, NaN))]
        @test SFrontiers.SFRedflag.sf_bound_message(bc2) ==
              "A parameter is at or beyond the numerical bound SFrontiers applies to it, a boundary solution: " *
              "α (coefficient ln_α) at its upper bound 100; σᵤ (coefficient ln_σᵤ²) at its lower bound 1e-12 " *
              "for 37 of 500 observations (7.4%). See Parameter Bounds in the manual.\n"
        # a share that would print as 0.0% or 100.0% while some but not all observations are at the bound
        bc3 = BE[BE((:sigma_u, "σᵤ", "ln_σᵤ²", :lower, 1e-12, false, 1, 3000, 1 / 3000, false, NaN)),
                 BE((:sigma_u, "σᵤ", "ln_σᵤ²", :upper, 1e12, false, 2999, 3000, 2999 / 3000, false, NaN))]
        @test SFrontiers.SFRedflag.sf_bound_message(bc3) ==
              "A parameter is at or beyond the numerical bound SFrontiers applies to it, a boundary solution: " *
              "σᵤ (coefficient ln_σᵤ²) at its lower bound 1e-12 for 1 of 3000 observations (<0.1%); " *
              "σᵤ (coefficient ln_σᵤ²) at its upper bound 1e+12 for 2999 of 3000 observations (>99.9%). " *
              "See Parameter Bounds in the manual.\n"
        # every observation at the bound, and a share that prints as 0.1%, keep the one-decimal share
        bc4 = BE[BE((:sigma_u, "σᵤ", "ln_σᵤ²", :lower, 1e-12, false, 3000, 3000, 1.0, false, NaN))]
        @test occursin("for 3000 of 3000 observations (100.0%). ", SFrontiers.SFRedflag.sf_bound_message(bc4))
        bc5 = BE[BE((:sigma_u, "σᵤ", "ln_σᵤ²", :lower, 1e-12, false, 3, 3000, 3 / 3000, false, NaN))]
        @test occursin("for 3 of 3000 observations (0.1%). ", SFrontiers.SFRedflag.sf_bound_message(bc5))
    end

    @testset "panel and :MLE" begin
        # panel_TFE Lomax, the shape above 100
        rngp = Xoshiro(3802); nf = 50; T = 4; NT = nf * T; id = repeat(1:nf, inner = T)
        xp = randn(rngp, NT); zp = randn(rngp, NT)
        yp = 0.5 .* xp .+ 0.3 .* randn(rngp, NT) .- abs.(0.5 .* randn(rngp, NT)) .* exp.(0.3 .* zp)
        spp = sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = reshape(xp, :, 1), zvar = reshape(zp, :, 1),
                           id = id, noise = :Normal, ineff = :Lomax)
        pp = sfmodel_init(spec = spp, frontier = [0.5], scaling = [0.3], ln_lambda = -1.0, ln_alpha = log(150.0),
                          ln_sigma_v_sq = log(0.09))
        r = redirect_stdout(() -> fitat(spp, :MCI, pp), devnull)
        @test _bnd_fields(r)
        @test [(e.parameter, e.side, e.bound, e.scalar, e.n_at, e.n) for e in r.bound_check] ==
              [(:alpha, :upper, 100.0, true, 200, 200)]
        # :MLE panel_TFE_CSW, σᵥ² = e^(−40) below its floor
        rngc = MersenneTwister(3605)
        Nc, Tc = 10, 5; NTc = Nc * Tc; idc = repeat(1:Nc, inner = Tc)
        Xc = randn(rngc, NTc, 1); αc = repeat(randn(rngc, Nc), inner = Tc)
        yc = αc .+ 0.5 .* Xc[:, 1] .+ 0.01 .* randn(rngc, NTc) .- abs.(randn(rngc, NTc))
        spc = sfmodel_spec(depvar = yc, frontier = Xc, noise = :Normal, ineff = :HalfNormal, type = :prod,
                           datatype = :panel_TFE_CSW, id = idc)
        r = redirect_stdout(() -> fitat(spc, :MLE, [0.5, 0.0, -40.0]), devnull)
        @test _bnd_fields(r)
        @test [(e.parameter, e.label, e.coefficient, e.side, e.bound, e.scalar, e.n) for e in r.bound_check] ==
              [(:sigma_v_sq, "σᵥ²", "log_σᵥ²", :lower, 1e-12, true, 50)]
        # :MLE panel_TRE half normal, σᵤ² = e^(−30) below its floor
        rngT = Xoshiro(11); nfT, TT = 30, 3; NTT = nfT * TT
        idT = repeat(1:nfT, inner = TT)
        XT = hcat(ones(NTT), randn(rngT, NTT))
        yT = XT * [1.0, 0.5] .+ repeat(0.3 .* randn(rngT, nfT), inner = TT) .+ 0.2 .* randn(rngT, NTT) .-
             abs.(0.4 .* randn(rngT, NTT))
        spT = sfmodel_spec(datatype = :panel_TRE, depvar = yT, frontier = XT, id = idT, noise = :Normal,
                           ineff = :HalfNormal)
        pT = sfmodel_init(spec = spT, frontier = XT \ yT, sigma_a_sq = log(0.09), ln_sigma_u_sq = -30.0,
                          ln_sigma_v_sq = log(0.04))
        r = redirect_stdout(() -> fitat(spT, :MLE, pT), devnull)
        @test _bnd_fields(r)
        @test [(e.parameter, e.side, e.bound) for e in r.bound_check] == [(:sigma_u_sq, :lower, 1e-12)]
    end

    @testset "the :MLE order" begin
        # a truncated-normal scaling model, held at σᵤ = e^(−30); the coefficients in the canonical order
        spS = sfmodel_spec(depvar = y3, frontier = X3, zvar = reshape(z3, :, 1), noise = :Normal,
                           ineff = :TruncatedNormal, hetero = :scaling, type = :prod)
        r = redirect_stdout(() -> fitat(spS, :MLE, [X3 \ y3; 0.3; 0.1; -60.0; log(0.09)]), devnull)
        @test _bnd_fields(r)
        @test [(e.parameter, e.side) for e in r.bound_check] == [(:sigma_u, :lower)]
        @test r.coeff[end - 1] == -60.0
    end

    @testset "interior fits are unchanged" begin
        # Data from fixed sequences (no random stream), so that they are the same on every Julia version.
        # The fits stop at a gradient norm below 1e-6, and where exactly can move with rounding and between
        # versions of the optimization and special-function packages: so the coefficients are compared in
        # units of their standard errors (1e-4), the log-likelihood (second order in the stopping point) to
        # 1e-6, and the standard errors and jlms to 1e-5 of their values. Bound bookkeeping that touched an
        # interior fit would move them by far more.
        Nd = 500; iD = 1:Nd
        xd = sqrt(2) .* erfinv.(2 .* mod.(iD .* 0.6180339887498949, 1.0) .- 1)
        vd = 0.5 .* sqrt(2) .* erfinv.(2 .* mod.(iD .* 0.4142135623730951, 1.0) .- 1)
        ud = (-log.(1 .- mod.(iD .* 0.7320508075688772, 1.0))) .^ (1/1.5)
        Xd = hcat(ones(Nd), xd); yd = Xd * [1.0, 0.5] .+ vd .- ud
        spD = sfmodel_spec(depvar = yd, frontier = Xd, noise = :Normal, ineff = :Weibull, type = :prod)
        pD = sfmodel_init(spec = spD, frontier = Xd \ yd, ln_lambda = 0.0, ln_k = log(2.0), ln_sigma_v_sq = log(0.25))
        optD = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 50, g_abstol = 1e-6))
        want = Dict(
            :MCI => (coeff = [1.0699011165420564, 0.4860969958819673, 0.07994288344681963, 0.45997738645682057,
                              -1.4466182846931106],
                     loglikelihood = -584.3273528568618,
                     std_err = [0.33897462229747066, 0.0339390550132678, 0.373819850149669, 0.26478193104168357,
                                0.4103346114197987],
                     jlms_sum = 486.0707192775913,
                     jlms3 = [1.1666995013660786, 0.4608446040002559, 0.7935273064223668]),
            :MSLE => (coeff = [0.9752700768358262, 0.48885522361658934, -0.026183046764910747, 0.37670504290679496,
                               -1.3380487641950782],
                      loglikelihood = -585.0958499596923,
                      std_err = [0.2970452534683434, 0.03374812950907836, 0.37224814143537044, 0.2552597520411635,
                                 0.34130831991809035],
                      jlms_sum = 438.7525749291359,
                      jlms3 = [1.0440593136503569, 0.4150974885608545, 0.7001083693756671]))
        for m in (:MCI, :MSLE)
            r = redirect_stdout(() -> sfmodel_fit(spec = spD, method = sfmodel_method(method = m, n_draws = 64),
                                                  init = pD, optim_options = optD, marginal = false, show_table = false,
                                                  verbose = false), devnull)
            w = want[m]
            @test _bnd_fields(r)
            @test r.converged && r.redflag == 0 && r.hessian_flag == 0 && isempty(r.bound_check)
            @test all(abs.(r.coeff .- w.coeff) .<= 1e-4 .* w.std_err)
            @test isapprox(r.loglikelihood, w.loglikelihood; atol = 1e-6)
            @test all(isapprox.(r.std_err, w.std_err; rtol = 1e-5))
            @test isapprox(sum(r.jlms), w.jlms_sum; rtol = 1e-5)
            @test all(isapprox.(r.jlms[[1, 250, 500]], w.jlms3; rtol = 1e-5))
        end
    end

    @testset "a coefficient beyond its bound is reported at the bound" begin
        mciB = SFrontiers.MCI_Backend
        msleB = SFrontiers.MSLE_Backend
        # The simulated negative log-likelihood of the specification `sp` under `m`, assembled with 64 draws.
        function _bnd_nll(sp, m)
            if m === :MCI
                msp = mciB._assemble_MCI_spec(sp.mci_spec, sfmodel_method(method = :MCI, n_draws = 64))
                return p -> mciB.MCI_nll(msp, p)
            end
            msp = msleB._assemble_MSLE_spec(sp.msle_spec, sfmodel_method(method = :MSLE, n_draws = 64))
            return p -> msleB.qmc_nll(msp, p)
        end
        # The one entry of the boundary verdict of `r` for the parameter `par`.
        _bnd_entry(r, par) = only([e for e in r.bound_check if e.parameter === par])
        # The value of f() and the text it printed.
        function _bnd_printed(f)
            path, io = mktemp()
            ret = redirect_stdout(f, io)
            close(io)
            out = read(path, String)
            rm(path)
            return ret, out
        end
        _bnd_held(sp, m, p; verbose = false) = redirect_stdout(() -> fitat(sp, m, p; verbose = verbose), devnull)

        # The Weibull shape above its ceiling 10 and below its floor 0.1: the likelihood, its Hessian and the
        # other coefficients are those at the start; the coefficient is reported just beyond the bound.
        for m in (:MCI, :MSLE)
            nll = _bnd_nll(spW, m)
            p15 = pW(log(15.0))
            r = _bnd_held(spW, m, p15)
            r50 = _bnd_held(spW, m, pW(log(50.0)))
            @test 10 < exp(r.coeff[4]) < 10 * (1 + 1e-9)
            @test r.coeff[[1, 2, 3, 5]] == p15[[1, 2, 3, 5]]
            @test nll(collect(r.coeff)) == nll(p15)
            @test isapprox(r.loglikelihood, -nll(p15); rtol = 1e-12)
            # both starts use k = 10
            @test r50.coeff == r.coeff
            @test r50.loglikelihood == r.loglikelihood
            @test r50.jlms == r.jlms
            # the Hessian is that at the start, whose row of ln k is zero
            @test all(iszero, r.Hessian[4, :])
            @test _bnd_entry(r50, :k).coef_before == log(50.0)
            p005 = pW(log(0.05))
            r = _bnd_held(spW, m, p005)
            @test 0.1 * (1 - 1e-9) < exp(r.coeff[4]) < 0.1
            @test nll(collect(r.coeff)) == nll(p005)
            @test _bnd_entry(r, :k).moved
        end

        # σᵤ = e^(−30) below its floor 1e-12; the coefficient is ln σᵤ², so σᵤ = exp(θ/2)
        spHN = sfmodel_spec(depvar = y3, frontier = X3, noise = :Normal, ineff = :HalfNormal, type = :prod)
        pHN = sfmodel_init(spec = spHN, frontier = X3 \ y3, ln_sigma_u_sq = -60.0, ln_sigma_v_sq = log(0.09))
        for m in (:MCI, :MSLE)
            nll = _bnd_nll(spHN, m)
            r = _bnd_held(spHN, m, pHN)
            @test 2 * log(1e-12) - 1e-9 < r.coeff[3] < 2 * log(1e-12)
            @test r.coeff[[1, 2, 4]] == pHN[[1, 2, 4]]
            @test nll(collect(r.coeff)) == nll(pHN)
            e = _bnd_entry(r, :sigma_u)
            @test e.moved && e.coef_before == -60.0
        end

        # The copula dependence above its ceiling 50 (ρ = exp(θ) + 1e-6 for the Clayton copulas, exp(θ) + 1 for the
        # Gumbel)
        for cop in (:Clayton, :Clayton90, :Gumbel), m in (:MCI, :MSLE)
            spC = sfmodel_spec(depvar = y3, frontier = X3, noise = :Normal, ineff = :HalfNormal, copula = cop,
                               type = :prod)
            pC = sfmodel_init(spec = spC, frontier = X3 \ y3, ln_sigma_u_sq = log(0.64), ln_sigma_v_sq = log(0.09),
                              theta_rho = log(60.0))
            off = cop === :Gumbel ? 1.0 : 1e-6
            nll = _bnd_nll(spC, m)
            r = _bnd_held(spC, m, pC)
            @test 50 < exp(r.coeff[end]) + off < 50 * (1 + 1e-9)
            @test r.rho == 50.0
            @test nll(collect(r.coeff)) == nll(pC)
            @test _bnd_entry(r, :rho).moved
        end
        # The copula table reports the ρ of the likelihood
        for B in (mciB, msleB)
            idxc = (copula = (theta_rho = 1,),)
            for cop in (:Clayton, :Clayton90, :Gumbel), θ in (log(0.5), log(10.0), log(60.0))
                mdl = B._build_model(:Normal, :HalfNormal; copula = cop)
                @test B._detect_copula_params(mdl, idxc, [θ], [0.1]).rho ==
                      B.get_copula_vals(mdl.copula, [θ], idxc, B.make_copula_constants(mdl.copula, Float64)).rho
            end
            mdl = B._build_model(:Normal, :HalfNormal; copula = :Gaussian)
            @test B._detect_copula_params(mdl, idxc, [2.0], [0.1]).rho ==
                  B.get_copula_vals(mdl.copula, [2.0], idxc, B.make_copula_constants(mdl.copula, Float64)).rho
        end

        # :MLE panel_TFE_CSW, σᵥ² below its floor (coefficients: frontier, ln σᵤ², ln σᵥ²)
        rngc = MersenneTwister(3605)
        Nc, Tc = 50, 5; NTc = Nc * Tc; idc = repeat(1:Nc, inner = Tc)
        Xc = randn(rngc, NTc, 1); αc = repeat(randn(rngc, Nc), inner = Tc)
        yc = αc .+ 0.5 .* Xc[:, 1] .+ 0.01 .* randn(rngc, NTc) .- abs.(randn(rngc, NTc))
        spc = sfmodel_spec(depvar = yc, frontier = Xc, noise = :Normal, ineff = :HalfNormal, type = :prod,
                           datatype = :panel_TFE_CSW, id = idc)
        r = _bnd_held(spc, :MLE, [0.5, 0.0, -40.0])
        r50 = _bnd_held(spc, :MLE, [0.5, 0.0, -50.0])
        @test log(1e-12) - 1e-9 < r.coeff[3] < log(1e-12)
        @test r.coeff[1:2] == [0.5, 0.0]
        @test r50.coeff == r.coeff
        @test r50.loglikelihood == r.loglikelihood
        @test all(iszero, r.Hessian[end, :])
        @test all(isnan, r.std_err)
        e = _bnd_entry(r, :sigma_v_sq)
        @test e.moved && e.coef_before == -40.0
        @test _bnd_held(spc, :MLE, collect(r.coeff)).loglikelihood == r.loglikelihood

        # :MLE panel_TRE half normal, σᵤ² below its floor
        rngT = Xoshiro(11); nfT, TT = 50, 5; NTT = nfT * TT
        idT = repeat(1:nfT, inner = TT)
        XT = hcat(ones(NTT), randn(rngT, NTT))
        yT = XT * [1.0, 0.5] .+ repeat(0.3 .* randn(rngT, nfT), inner = TT) .+ 0.2 .* randn(rngT, NTT) .-
             abs.(0.4 .* randn(rngT, NTT))
        spT = sfmodel_spec(datatype = :panel_TRE, depvar = yT, frontier = XT, id = idT, noise = :Normal,
                           ineff = :HalfNormal)
        pT(lu) = sfmodel_init(spec = spT, frontier = XT \ yT, sigma_a_sq = log(0.09), ln_sigma_u_sq = lu,
                              ln_sigma_v_sq = log(0.04))
        r = _bnd_held(spT, :MLE, pT(-30.0))
        r40 = _bnd_held(spT, :MLE, pT(-40.0))
        @test log(1e-12) - 1e-9 < r.coeff[4] < log(1e-12)
        @test r40.coeff == r.coeff
        @test r40.loglikelihood == r.loglikelihood
        e = _bnd_entry(r, :sigma_u_sq)
        @test e.moved && e.coef_before == -30.0
        @test _bnd_held(spT, :MLE, collect(r.coeff)).loglikelihood == r.loglikelihood

        # panel_TFE Lomax, the shape above its ceiling 100
        rngp = Xoshiro(3802); nf = 50; T = 5; NT = nf * T; id = repeat(1:nf, inner = T)
        xp = randn(rngp, NT); zp = randn(rngp, NT)
        yp = 0.5 .* xp .+ 0.3 .* randn(rngp, NT) .- abs.(0.5 .* randn(rngp, NT)) .* exp.(0.3 .* zp)
        spp = sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = reshape(xp, :, 1), zvar = reshape(zp, :, 1),
                           id = id, noise = :Normal, ineff = :Lomax)
        pp(la) = sfmodel_init(spec = spp, frontier = [0.5], scaling = [0.3], ln_lambda = -1.0, ln_alpha = la,
                              ln_sigma_v_sq = log(0.09))
        r = _bnd_held(spp, :MCI, pp(log(150.0)))
        r300 = _bnd_held(spp, :MCI, pp(log(300.0)))
        @test 100 < exp(r.coeff[4]) < 100 * (1 + 1e-9)
        @test r300.coeff == r.coeff
        @test r300.loglikelihood == r.loglikelihood
        @test r300.jlms == r.jlms
        e = _bnd_entry(r, :alpha)
        @test e.moved && e.coef_before == log(150.0)
        @test _bnd_held(spp, :MCI, collect(r.coeff)).loglikelihood == r.loglikelihood

        # The message says that the coefficient is reported at the bound; a heteroscedastic shape is not moved
        spL = sfmodel_spec(depvar = y3, frontier = X3, noise = :Normal, ineff = :Lomax, type = :prod)
        pL(la) = sfmodel_init(spec = spL, frontier = X3 \ y3, ln_lambda = 0.0, ln_alpha = la,
                              ln_sigma_v_sq = log(0.09))
        r, out = _bnd_printed(() -> fitat(spL, :MCI, pL(log(150.0)); verbose = true))
        @test occursin("α (coefficient ln_α) at its upper bound 100 (coefficient reported at the bound)", out)
        r, out = _bnd_printed(() -> fitat(spH, :MCI, pH; verbose = true))
        @test occursin("boundary solution", out)
        @test !occursin("reported at the bound", out)

        # Not moved: a heteroscedastic shape, a shape within the tolerance inside its bound, and :MLE parameters
        # whose likelihood has no clamp
        pin = pL(log(100 * (1 - 5e-7)))
        for m in (:MCI, :MSLE)
            r = _bnd_held(spH, m, pH)
            @test r.coeff == pH
            @test !isempty(r.bound_check) && all(e -> !e.moved, r.bound_check)
            r = _bnd_held(spL, m, pin)
            @test r.coeff == pin
            @test !_bnd_entry(r, :alpha).moved
        end
        r = _bnd_held(spHN, :MLE, pHN)
        @test r.coeff == pHN
        @test !_bnd_entry(r, :sigma_u).moved
    end

    @testset "starting values outside the bounds" begin
        # The exception that f() throws, or `nothing` when it returns; what it prints goes to devnull.
        function _bnd_thrown(f)
            try
                redirect_stdout(f, devnull)
            catch err
                return err
            end
            return nothing
        end
        # The value of f() and the text it printed.
        function _bnd_output(f)
            path, io = mktemp()
            ret = redirect_stdout(f, io)
            close(io)
            out = read(path, String)
            rm(path)
            return ret, out
        end
        opt2 = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 2,))
        fit2(sp, m, p) = sfmodel_fit(spec = sp, method = sfmodel_method(method = m, n_draws = 64), init = p,
                                     optim_options = opt2, marginal = false, show_table = false, verbose = false)

        # A Weibull shape started above its ceiling 10: the log-likelihood does not depend on ln k there, so Newton
        # cannot move it, and the fit is refused before the optimizer. Started inside, the fit moves k.
        Random.seed!(13); N8 = 300; x8 = randn(N8); X8 = hcat(ones(N8), x8); u8 = (-log.(1 .- rand(N8))) .^ (1/1.5)
        y8 = X8 * [1.0, 0.5] .+ 0.5 .* randn(N8) .- u8
        sp8 = sfmodel_spec(depvar = y8, frontier = X8, noise = :Normal, ineff = :Weibull, type = :prod)
        res8 = y8 .- X8 * (X8 \ y8)
        p8(k) = sfmodel_init(spec = sp8, frontier = X8 \ y8, ln_lambda = 0.0, ln_k = log(k),
                             ln_sigma_v_sq = log(var(res8) / 2))
        opt8 = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 40, g_abstol = 1e-5))
        fit8(m, p) = sfmodel_fit(spec = sp8, method = sfmodel_method(method = m, n_draws = 256), init = p,
                                 optim_options = opt8, marginal = false, show_table = false, verbose = false)
        for m in (:MCI, :MSLE)
            err = _bnd_thrown(() -> fit8(m, p8(20.0)))
            @test err isa ArgumentError
            txt = sprint(showerror, err)
            @test occursin("k", txt) && occursin("10", txt) && occursin("ln_k", txt)
            @test txt == "ArgumentError: The starting value of k is 20 (ln_k = 2.9957), at or beyond its upper bound " *
                         "10. Beyond a bound the log-likelihood does not depend on this coefficient, so a " *
                         "gradient-based optimizer cannot move it. Give a starting value inside the bounds: ln_k " *
                         "strictly between -2.3026 and 2.3026."
            r = redirect_stdout(() -> fit8(m, p8(2.0)), devnull)
            @test 0.1 < exp(r.coeff[4]) < 10
        end

        # A fit held at its starting values (no warmstart stage, no main iteration) is not checked; a warmstart of
        # five iterations is a fit that is not held.
        p15 = pW(log(15.0))
        @test _bnd_fields(redirect_stdout(() -> fitat(spW, :MCI, p15), devnull))
        fitW(o) = sfmodel_fit(spec = spW, method = sfmodel_method(method = :MCI, n_draws = 64), init = p15,
                              optim_options = o, marginal = false, show_table = false, verbose = false)
        optw0 = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 0,), main_solver = Newton(),
                            main_opt = (iterations = 0,))
        optw5 = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 5,), main_solver = Newton(),
                            main_opt = (iterations = 0,))
        @test _bnd_fields(redirect_stdout(() -> fitW(optw0), devnull))
        @test _bnd_thrown(() -> fitW(optw5)) isa ArgumentError

        # The Clayton dependence started above its ceiling 50, and inside it
        spc = sfmodel_spec(depvar = y3, frontier = X3, noise = :Normal, ineff = :HalfNormal, copula = :Clayton,
                           type = :prod)
        pc(tr) = sfmodel_init(spec = spc, frontier = X3 \ y3, ln_sigma_u_sq = log(0.64), ln_sigma_v_sq = log(0.09),
                              theta_rho = tr)
        err = _bnd_thrown(() -> fit2(spc, :MCI, pc(log(60.0))))
        @test err isa ArgumentError && occursin("theta_rho", sprint(showerror, err))
        # ρ has no lower bound, so the advice gives the upper end of theta_rho only
        @test occursin("Give a starting value inside the bounds: theta_rho below ", sprint(showerror, err))
        @test _bnd_fields(redirect_stdout(() -> fit2(spc, :MCI, pc(log(10.0))), devnull))

        # :MLE panel_TFE_CSW with σᵥ² = e^(−40) below its floor, and panel_TRE with σᵤ² = e^(−30) below its floor
        opt5 = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 5,))
        fit5(sp, p) = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE), init = p, optim_options = opt5,
                                  marginal = false, show_table = false, verbose = false)
        rngc = MersenneTwister(3605)
        Nc, Tc = 50, 5; NTc = Nc * Tc; idc = repeat(1:Nc, inner = Tc)
        Xc = randn(rngc, NTc, 1); αc = repeat(randn(rngc, Nc), inner = Tc)
        yc = αc .+ 0.5 .* Xc[:, 1] .+ 0.01 .* randn(rngc, NTc) .- abs.(randn(rngc, NTc))
        spcw = sfmodel_spec(depvar = yc, frontier = Xc, noise = :Normal, ineff = :HalfNormal, type = :prod,
                            datatype = :panel_TFE_CSW, id = idc)
        err = _bnd_thrown(() -> fit5(spcw, [0.5, 0.0, -40.0]))
        @test err isa ArgumentError && occursin("ln_sigma_v_sq", sprint(showerror, err))
        rngT = Xoshiro(11); nfT, TT = 50, 5; NTT = nfT * TT
        idT = repeat(1:nfT, inner = TT)
        XT = hcat(ones(NTT), randn(rngT, NTT))
        yT = XT * [1.0, 0.5] .+ repeat(0.3 .* randn(rngT, nfT), inner = TT) .+ 0.2 .* randn(rngT, NTT) .-
             abs.(0.4 .* randn(rngT, NTT))
        spT = sfmodel_spec(datatype = :panel_TRE, depvar = yT, frontier = XT, id = idT, noise = :Normal,
                           ineff = :HalfNormal)
        pT = sfmodel_init(spec = spT, frontier = XT \ yT, sigma_a_sq = log(0.09), ln_sigma_u_sq = -30.0,
                          ln_sigma_v_sq = log(0.04))
        err = _bnd_thrown(() -> fit5(spT, pT))
        @test err isa ArgumentError && occursin("ln_sigma_u_sq", sprint(showerror, err))
        # The other :MLE likelihoods have no clamp, and their starts are not checked: a half normal started with
        # σᵤ = e^30, above the ceiling 1e12, runs
        spHN = sfmodel_spec(depvar = y3, frontier = X3, noise = :Normal, ineff = :HalfNormal, type = :prod)
        pHN = sfmodel_init(spec = spHN, frontier = X3 \ y3, ln_sigma_u_sq = 60.0, ln_sigma_v_sq = log(0.09))
        @test _bnd_fields(redirect_stdout(() -> fit5(spHN, pHN), devnull))

        # panel_TFE Lomax, the shape started above its ceiling 100
        rngp = Xoshiro(3802); nf = 50; T = 5; NT = nf * T; id = repeat(1:nf, inner = T)
        xp = randn(rngp, NT); zp = randn(rngp, NT)
        yp = 0.5 .* xp .+ 0.3 .* randn(rngp, NT) .- abs.(0.5 .* randn(rngp, NT)) .* exp.(0.3 .* zp)
        spp = sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = reshape(xp, :, 1), zvar = reshape(zp, :, 1),
                           id = id, noise = :Normal, ineff = :Lomax)
        pp = sfmodel_init(spec = spp, frontier = [0.5], scaling = [0.3], ln_lambda = -1.0, ln_alpha = log(150.0),
                          ln_sigma_v_sq = log(0.09))
        err = _bnd_thrown(() -> fit2(spp, :MCI, pp))
        @test err isa ArgumentError && occursin("ln_alpha", sprint(showerror, err))

        # A start whose natural value lies inside the bound but within the relative tolerance counts as at the
        # bound, and the error says so; a start beyond the bound does not need the sentence.
        spL = sfmodel_spec(depvar = y3, frontier = X3, noise = :Normal, ineff = :Lomax, type = :prod)
        pL(la) = sfmodel_init(spec = spL, frontier = X3 \ y3, ln_lambda = 0.0, ln_alpha = la, ln_sigma_v_sq = log(0.09))
        err = _bnd_thrown(() -> fit2(spL, :MCI, pL(log(100 * (1 - 5e-7)))))
        @test err isa ArgumentError && occursin("counts as at the bound", sprint(showerror, err))
        err = _bnd_thrown(() -> fit2(spL, :MCI, pL(log(150.0))))
        @test err isa ArgumentError && !occursin("counts as at the bound", sprint(showerror, err))

        # A heteroscedastic shape at its bound for some observations at the start: a warning, and the fit goes on
        opt1 = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 1,))
        r, out = _bnd_output(() -> sfmodel_fit(spec = spH, method = sfmodel_method(method = :MCI, n_draws = 64),
                                               init = pH, optim_options = opt1, marginal = false, show_table = false,
                                               verbose = false))
        @test occursin("at the starting values", out) && occursin("of 500 observations", out)
        # the warning counts the observations with k_i = 5·exp(z_i) at or above the ceiling 10
        nhi = count(>=(10 * (1 - 1e-6)), exp.(log(5.0) .+ 1.0 .* zW))
        @test occursin("Warning: at the starting values, k is at or beyond its upper bound 10 for $nhi of 500 " *
                       "observations", out)
        @test _bnd_fields(r)
        # The share is written as in the boundary message: "<0.1%" and ">99.9%" where one decimal would show 0.0% or
        # 100.0% while some but not all observations are at the bound
        BE = SFrontiers.SFRedflag.SFBoundEntry
        lk = [(parameter = :k, keyword = :ln_k, block = [4], lo = 0.1, hi = 10.0, form = :exp, offset = 0.0)]
        w = SFrontiers.SFRedflag.sf_start_check(lk, BE[BE((:k, "k", "ln_k", :lower, 0.1, false, 1, 3000, 1 / 3000,
                                                           false, NaN))], zeros(5))[2]
        @test w == ["at the starting values, k is at or beyond its lower bound 0.1 for 1 of 3000 observations " *
                    "(<0.1%); at those observations the log-likelihood does not depend on the coefficients of ln_k. " *
                    "Consider other starting values."]
        w = SFrontiers.SFRedflag.sf_start_check(lk, BE[BE((:k, "k", "ln_k", :upper, 10.0, false, 2999, 3000,
                                                           2999 / 3000, false, NaN))], zeros(5))[2]
        @test occursin("for 2999 of 3000 observations (>99.9%);", only(w))

        # The default starts lie inside every bound
        optd = sfmodel_opt(main_solver = NelderMead(), main_opt = (iterations = 1,))
        fitd(sp, m) = sfmodel_fit(spec = sp, method = m === :MLE ? sfmodel_method(method = :MLE) :
                                                      sfmodel_method(method = m, n_draws = 64),
                                  init = nothing, optim_options = optd, marginal = false, show_table = false,
                                  verbose = false)
        for fam in (:Weibull, :Lomax, :Exponential, :Gamma), m in (:MCI, :MSLE)
            fam === :Gamma && m === :MSLE && continue
            sp = sfmodel_spec(depvar = y3, frontier = X3, noise = :Normal, ineff = fam, type = :prod)
            @test _bnd_fields(redirect_stdout(() -> fitd(sp, m), devnull))
        end
        for cop in (:Clayton, :Gumbel), m in (:MCI, :MSLE)
            sp = sfmodel_spec(depvar = y3, frontier = X3, noise = :Normal, ineff = :HalfNormal, copula = cop,
                              type = :prod)
            @test _bnd_fields(redirect_stdout(() -> fitd(sp, m), devnull))
        end
        @test _bnd_fields(redirect_stdout(() -> fitd(spp, :MCI), devnull))
        @test _bnd_fields(redirect_stdout(() -> fitd(spcw, :MLE), devnull))
        @test _bnd_fields(redirect_stdout(() -> fitd(spT, :MLE), devnull))
    end
end
