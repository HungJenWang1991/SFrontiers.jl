# The one-sided third-moment test of the OLS residuals: sf_skewness_test and the four fields it adds to
# residual_diagnostics. Data, fits and helpers stay inside the outer test set, because runtests.jl includes every
# test file into one module whose constants include `X` and `y`.

using SFrontiers, Test, Random, Statistics, Optim, StatsAPI

@testset "one-sided third-moment test of the OLS residuals" begin
    # Rice farm data, 271 observations. The header names `yr` twice, so the file is parsed by hand.
    lines = readlines(joinpath(@__DIR__, "data", "sampledata.csv"))
    M = reduce(vcat, [permutedims(parse.(Float64, split(l, ','))) for l in lines[2:end]])
    y = M[:, 1]
    X = hcat(ones(size(M, 1)), M[:, 2:7])   # constant, Lland, PIland, Llabor, Lbull, Lcost, yr
    _skt_opt = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 200, g_abstol = 1e-5),
                           main_solver = Newton(), main_opt = (iterations = 100, g_abstol = 1e-8))
    r_hn = sfmodel_fit(spec = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal,
                                           type = :prod),
                       method = sfmodel_method(method = :MLE), optim_options = _skt_opt,
                       marginal = false, show_table = false, verbose = false)

    @testset "rice data values" begin
        t = sf_skewness_test(r_hn)
        @test t.M3T_stat ≈ -4.7280420600 atol = 1e-6
        @test t.M3T_pvalue ≈ 1.133476e-6 rtol = 1e-4
        @test t.M3T_robust_stat ≈ -3.3318900921 atol = 1e-6
        @test t.M3T_robust_pvalue ≈ 4.312916e-4 rtol = 1e-4
        @test t.skew_ols ≈ -0.703513298114 atol = 1e-9
        @test t.expected_skew_sign == -1
        @test t.n == 271
        @test t.M3T_stat ≈ sqrt(271 / 6) * t.skew_ols rtol = 1e-12
        rd = residual_diagnostics(r_hn)
        @test rd.M3T_stat == t.M3T_stat
        @test rd.M3T_pvalue == t.M3T_pvalue
        @test rd.M3T_robust_stat == t.M3T_robust_stat
        @test rd.M3T_robust_pvalue == t.M3T_robust_pvalue
        @test rd.JarqueBera_stat ≈ 32.6099471723 rtol = 1e-9
        @test rd.JarqueBera_pvalue ≈ 8.295450e-8 rtol = 1e-5
        # The statistics depend only on y and the frontier matrix, so an MSLE fit of the same model gives the same result.
        r_ms = sfmodel_fit(spec = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal,
                                               type = :prod),
                           method = sfmodel_method(method = :MSLE, n_draws = 64),
                           optim_options = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40,),
                                                       main_solver = Newton(), main_opt = (iterations = 30,)),
                           marginal = false, show_table = false, verbose = false)
        @test sf_skewness_test(r_ms) == t
    end

    @testset "a cost frontier looks in the other tail" begin
        t = sf_skewness_test(r_hn)
        r_c = sfmodel_fit(spec = sfmodel_spec(depvar = -y, frontier = X, noise = :Normal, ineff = :HalfNormal,
                                              type = :cost),
                          method = sfmodel_method(method = :MLE), optim_options = _skt_opt,
                          marginal = false, show_table = false, verbose = false)
        tc = sf_skewness_test(r_c)
        @test tc.expected_skew_sign == 1
        @test tc.M3T_stat ≈ 4.7280420600 atol = 1e-6
        @test tc.M3T_robust_stat ≈ 3.3318900921 atol = 1e-6
        @test tc.M3T_pvalue ≈ t.M3T_pvalue rtol = 1e-10
        @test tc.M3T_robust_pvalue ≈ t.M3T_robust_pvalue rtol = 1e-10
        # A production fit of -y, held at its starting values: its residuals are skewed the wrong way.
        r_w = sfmodel_fit(spec = sfmodel_spec(depvar = -y, frontier = X, noise = :Normal, ineff = :HalfNormal,
                                              type = :prod),
                          method = sfmodel_method(method = :MLE),
                          optim_options = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,)),
                          marginal = false, show_table = false, verbose = false)
        @test sf_skewness_test(r_w).M3T_pvalue ≈ 1 - 1.133476e-6 atol = 1e-10
        # The two one-sided p-values of the same residuals add up to one.
        e = residuals(r_hn; type = :ols)
        @test SFrontiers._third_moment_test(e, -1)[2] + SFrontiers._third_moment_test(e, 1)[2] ≈ 1 atol = 1e-12
        @test SFrontiers._third_moment_test(e, -1)[4] + SFrontiers._third_moment_test(e, 1)[4] ≈ 1 atol = 1e-12
    end

    @testset "size under normal and Laplace noise" begin
        # OLS residuals of y = v on the rice design (they do not depend on the frontier coefficients); 10,000 samples
        # of normal noise, then, with the same generator, 10,000 samples of Laplace(0, 1) noise.
        rng = Xoshiro(20261006)
        R = 10_000
        rej_normal = 0
        rej_normal_robust = 0
        for _ in 1:R
            v = randn(rng, 271)
            e = v .- X * (X \ v)
            zN, pN, zR, pR = SFrontiers._third_moment_test(e, -1)
            rej_normal += pN < 0.05
            rej_normal_robust += pR < 0.05
        end
        rej_laplace_robust = 0
        for _ in 1:R
            v = randexp(rng, 271) .- randexp(rng, 271)
            e = v .- X * (X \ v)
            zN, pN, zR, pR = SFrontiers._third_moment_test(e, -1)
            rej_laplace_robust += pR < 0.05
        end
        @test 0.035 <= rej_normal / R <= 0.065
        @test 0.035 <= rej_normal_robust / R <= 0.065
        @test 0.035 <= rej_laplace_robust / R <= 0.065
    end

    @testset "the new fields come after the existing seven" begin
        rd = residual_diagnostics(r_hn)
        @test keys(rd)[1:7] == (:skew_ols, :skew_composed, :expected_skew_sign, :skew_ols_sign_ok,
                                :skew_composed_sign_ok, :JarqueBera_stat, :JarqueBera_pvalue)
        @test keys(rd)[8:11] == (:M3T_stat, :M3T_pvalue, :M3T_robust_stat, :M3T_robust_pvalue)
        a1, a2, a3, a4, a5, a6, a7 = rd
        @test a1 == rd.skew_ols
        @test a7 == rd.JarqueBera_pvalue
    end

    @testset "panel fits" begin
        Random.seed!(303)
        Nf = 30; Tp = 5; NT = Nf * Tp; id = repeat(1:Nf, inner = Tp)
        Xp = hcat(randn(NT), randn(NT))                # no intercept (the fixed effects absorb it)
        αi = repeat(2 .* randn(Nf), inner = Tp)
        zp = randn(NT)
        yp = αi .+ Xp * [0.6, -0.5] .+ 0.3 .* randn(NT) .- abs.(0.5 .* randn(NT))
        sp = sfmodel_spec(depvar = yp, frontier = Xp, zvar = zp, noise = :Normal, ineff = :HalfNormal,
                          type = :prod, datatype = :panel_TFE, id = id)
        popt = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40,),
                           main_solver = NelderMead(), main_opt = (iterations = 200,))
        r_pm = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE), optim_options = popt,
                           marginal = false, show_table = false, verbose = false)
        r_ps = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MSLE, n_draws = 32), optim_options = popt,
                           marginal = false, show_table = false, verbose = false)
        @test getfield(r_pm, :backend) === :MLE
        @test getfield(r_ps, :backend) === :Panel
        for r in (r_pm, r_ps)
            @test_throws "not provided for panel models" sf_skewness_test(r)
            rd = residual_diagnostics(r)
            @test all(isnan, (rd.M3T_stat, rd.M3T_pvalue, rd.M3T_robust_stat, rd.M3T_robust_pvalue))
            @test isfinite(rd.skew_ols)
        end
        # A result wrapper without a spec, as the exported sfmodel_panel_fit returns.
        @test_throws "not provided for panel models" sf_skewness_test(SFrontiers.SFResult((loglikelihood = -1.0,), :Panel))
    end
end
