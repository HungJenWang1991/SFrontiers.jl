# The likelihood-ratio test of the stochastic frontier against OLS: the one model for which sf_vs_ols refers its
# statistic to ½χ²(0) + ½χ²(1), the refusals of every other model, the removed `dof` keyword, the OLS
# log-likelihood that cross-sectional results carry and panel results do not, and the table of sfmodel_MixTable
# against the exact quantiles of the mixture. Data, fits and helpers stay inside the outer test set, because
# runtests.jl includes every test file into one module whose constants include `X` and `y`.

using SFrontiers, Test, Random, Statistics, Optim, StatsAPI
using StatsFuns: chisqccdf
using DataFrames: DataFrame

@testset "sf_vs_ols and the mixed χ² table" begin
    _svo_msg(r) = try sf_vs_ols(r); "" catch err; sprint(showerror, err) end
    # A result that carries only the backend tag and the spec, which is all the refusal reads.
    _svo_dummy(sp) = SFrontiers.SFResult((loglikelihood = -1.0, OLS_loglikelihood = -2.0), :MSLE, sp)
    # Runs f() with stdout sent to a file; returns the value of f() and the printed text.
    function _svo_captured_stdout(f)
        old = stdout
        path, io = mktemp()
        local ret
        redirect_stdout(io)
        try
            ret = f()
        finally
            flush(io); redirect_stdout(old); close(io)
        end
        (ret, read(path, String))
    end

    @testset "normal noise and homoscedastic half-normal inefficiency on the rice data" begin
        # Rice farm data, 271 observations. The header names `yr` twice, so the file is parsed by hand.
        lines = readlines(joinpath(@__DIR__, "data", "sampledata.csv"))
        M = reduce(vcat, [permutedims(parse.(Float64, split(l, ','))) for l in lines[2:end]])
        y = M[:, 1]
        X = hcat(ones(size(M, 1)), M[:, 2:7])   # constant, Lland, PIland, Llabor, Lbull, Lcost, yr
        _svo_rice_opt = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 200, g_abstol = 1e-5),
                                    main_solver = Newton(), main_opt = (iterations = 100, g_abstol = 1e-8))
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal, type = :prod)
        r_hn = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE), optim_options = _svo_rice_opt,
                           marginal = false, show_table = false, verbose = false)
        t = sf_vs_ols(r_hn)
        @test t.dof == 1
        @test t.mixed
        @test t.pvalue ≈ 0.5 * chisqccdf(1, t.LR) rtol = 1e-12
        @test t.LR ≈ 23.207786 atol = 1e-4
        @test t.pvalue ≈ 7.270374e-7 rtol = 1e-3
        @test t.critical_values.p10 ≈ 1.642374 atol = 1e-5
        @test t.critical_values.p05 ≈ 2.705543 atol = 1e-5
        @test t.critical_values.p025 ≈ 3.841459 atol = 1e-5
        @test t.critical_values.p01 ≈ 5.411894 atol = 1e-5
        @test r_hn.loglikelihood ≈ -93.366038 atol = 1e-5
        @test r_hn.OLS_loglikelihood ≈ -104.969931 atol = 1e-6
        @test hasproperty(r_hn, :OLS_loglikelihood)
        # A cross-sectional fit still prints the OLS log-likelihood.
        _, out = _svo_captured_stdout(() -> sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE),
                                                         optim_options = _svo_rice_opt, marginal = false,
                                                         show_table = true, verbose = false))
        @test occursin("OLS (frontier-only) log-likelihood", out)

        # The same model fitted by simulation is accepted too, without any log record.
        sim_opt = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40,),
                              main_solver = Newton(), main_opt = (iterations = 30,))
        for m in (:MSLE, :MCI)
            r = sfmodel_fit(spec = sp, method = sfmodel_method(method = m, n_draws = 64), optim_options = sim_opt,
                            marginal = false, show_table = false, verbose = false)
            @test_logs sf_vs_ols(r)
            @test sf_vs_ols(r).dof == 1
        end

        # The `dof` keyword is gone: any value other than `nothing` raises an error.
        @test_throws "keyword was removed" sf_vs_ols(r_hn; dof = 1)
        @test_throws "keyword was removed" sf_vs_ols(r_hn; dof = 8)
        @test sf_vs_ols(r_hn; dof = nothing).dof == 1

        # The same model written with `hetero` and a `zvar` whose columns are all constant is accepted, with
        # the statistic of the homoscedastic fit.
        n = length(y)
        _svo_hspec(zv) = sfmodel_spec(depvar = y, frontier = X, zvar = zv, noise = :Normal, ineff = :HalfNormal,
                                      hetero = [:sigma_u_sq], type = :prod)
        r_c = sfmodel_fit(spec = _svo_hspec(ones(n, 1)), method = sfmodel_method(method = :MLE),
                          optim_options = _svo_rice_opt, marginal = false, show_table = false, verbose = false)
        t_c = sf_vs_ols(r_c)
        @test t_c.dof == 1 && t_c.mixed
        @test t_c.LR ≈ t.LR atol = 1e-4
        for zv in (fill(2.0, n, 1), hcat(ones(n), fill(3.0, n)))
            @test _svo_msg(_svo_dummy(_svo_hspec(zv))) == ""
        end
        # the macro form, `@zvar(cons)`
        df = DataFrame(hcat(y, X), [:y, :cons, :x1, :x2, :x3, :x4, :x5, :x6])
        sp_m = sfmodel_spec(@useData(df), @depvar(y), @frontier(cons, x1, x2, x3, x4, x5, x6), @zvar(cons);
                            noise = :Normal, ineff = :HalfNormal, hetero = [:sigma_u_sq], type = :prod)
        @test _svo_msg(_svo_dummy(sp_m)) == ""
        # a `zvar` column that varies is still refused, and so is an all-zero `zvar`, which fixes σᵤ² at 1
        @test occursin("heteroscedasticity coefficients of sigma_u_sq",
                       _svo_msg(_svo_dummy(_svo_hspec(hcat(ones(n), X[:, 2])))))
        @test occursin("heteroscedasticity coefficients of sigma_u_sq", _svo_msg(_svo_dummy(_svo_hspec(zeros(n, 1)))))
    end

    @testset "refused cross-sectional models" begin
        Random.seed!(304)
        n = 200
        X = hcat(ones(n), randn(n), randn(n))
        z1 = randn(n); z2 = randn(n)
        y = X * [1.0, 0.5, -0.3] .+ 0.3 .* randn(n) .- abs.(0.4 .* randn(n))
        _svo_spec(; kw...) = sfmodel_spec(; depvar = y, frontier = X, type = :prod, kw...)
        cases = [
            (_svo_spec(noise = :Normal, ineff = :TruncatedNormal), ["truncated-normal location"]),
            (_svo_spec(noise = :Normal, ineff = :HalfNormal, zvar = hcat(ones(n), z1), hetero = [:sigma_u_sq]),
             ["heteroscedasticity coefficients of sigma_u_sq"]),
            (_svo_spec(noise = :Normal, ineff = :TruncatedNormal, zvar = reshape(z1, n, 1), hetero = :scaling),
             ["scaling coefficients", "truncated-normal location"]),
            [(_svo_spec(noise = :Normal, ineff = f), ["no published derivation", string(f)])
             for f in (:Exponential, :Rayleigh, :Erlang2)]...,
            [(_svo_spec(noise = :Normal, ineff = f), ["is not identified under H₀", string(f)])
             for f in (:Weibull, :Gamma, :Lomax, :Lognormal)]...,
            (_svo_spec(noise = :Normal, ineff = :HalfNormal, copula = :Gaussian), ["copula parameter"]),
            (_svo_spec(noise = :StudentT, ineff = :HalfNormal), ["Student-t"]),
            (_svo_spec(noise = :Laplace, ineff = :HalfNormal), ["Laplace"]),
        ]
        for (sp, words) in cases
            m = _svo_msg(_svo_dummy(sp))
            for w in ("homoscedastic half-normal", "Counting the parameters", words...)
                @test occursin(w, m)
            end
            for w in ("Alternatives", "sf_skewness_test", "same data")
                @test !occursin(w, m)
            end
        end

        # A real fit of a refused model.
        nm_opt = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40,),
                             main_solver = NelderMead(), main_opt = (iterations = 200,))
        r_tn = sfmodel_fit(spec = _svo_spec(noise = :Normal, ineff = :TruncatedNormal, zvar = hcat(z1, z2),
                                            hetero = [:mu, :sigma_u_sq]),
                           method = sfmodel_method(method = :MCI, n_draws = 100), optim_options = nm_opt,
                           marginal = false, show_table = false, verbose = false)
        m = _svo_msg(r_tn)
        @test occursin("truncated-normal location", m)
        @test occursin("heteroscedasticity coefficients of mu, sigma_u_sq", m)
    end

    @testset "panel models" begin
        Random.seed!(303)
        Nf = 30; Tp = 5; NT = Nf * Tp; id = repeat(1:Nf, inner = Tp)
        Xp = hcat(randn(NT), randn(NT))                # no intercept (the fixed effects absorb it)
        αi = repeat(2 .* randn(Nf), inner = Tp)
        zp = randn(NT)
        yp = αi .+ Xp * [0.6, -0.5] .+ 0.3 .* randn(NT) .- abs.(0.5 .* randn(NT))
        popt = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40,),
                           main_solver = NelderMead(), main_opt = (iterations = 200,))
        _svo_pspec(dt; kw...) = sfmodel_spec(; depvar = yp, frontier = Xp, noise = :Normal, ineff = :HalfNormal,
                                             type = :prod, datatype = dt, id = id, kw...)
        _svo_pfit(sp, meth) = sfmodel_fit(spec = sp, method = meth, optim_options = popt,
                                          marginal = false, show_table = false, verbose = false)
        r_pt_s = _svo_pfit(_svo_pspec(:panel_TFE; zvar = zp), sfmodel_method(method = :MSLE, n_draws = 32))
        r_pt_m = _svo_pfit(_svo_pspec(:panel_TFE; zvar = zp), sfmodel_method(method = :MLE))
        r_csw = _svo_pfit(_svo_pspec(:panel_TFE_CSW), sfmodel_method(method = :MLE))
        r_tre = _svo_pfit(_svo_pspec(:panel_TRE), sfmodel_method(method = :MLE))
        @test getfield(r_pt_s, :backend) === :Panel
        @test getfield(r_pt_m, :backend) === :MLE
        # The last one is a result wrapper without a spec, as the exported sfmodel_panel_fit returns.
        for r in (r_pt_s, r_pt_m, r_csw, r_tre,
                  SFrontiers.SFResult((loglikelihood = -1.0, OLS_loglikelihood = -2.0), :Panel))
            m = _svo_msg(r)
            @test occursin("panel model", m)
            @test occursin("do not support panel models", m)
            @test !occursin("Alternatives", m)
        end

        # Panel results neither store nor print the OLS log-likelihood; the skewness of the OLS residuals stays.
        for r in (r_pt_s, r_pt_m, r_csw, r_tre)
            @test !hasproperty(r, :OLS_loglikelihood)
        end
        # the panel backend does not import the normal log-density, which it does not use
        @test !isdefined(SFrontiers.Panel_Backend, :normlogpdf)
        for meth in (sfmodel_method(method = :MSLE, n_draws = 32), sfmodel_method(method = :MLE))
            _, out = _svo_captured_stdout(() -> sfmodel_fit(spec = _svo_pspec(:panel_TFE; zvar = zp), method = meth,
                                                             optim_options = popt, marginal = false,
                                                             show_table = true, verbose = false))
            @test !occursin("OLS (frontier-only) log-likelihood", out)
            @test occursin("Skewness of OLS residuals", out)
        end
    end

    @testset "a result without the model specification" begin
        m = _svo_msg(SFrontiers.SFResult((loglikelihood = -1.0, OLS_loglikelihood = -2.0), :MLE))
        @test occursin("fit the model with sfmodel_fit", m)
    end

    @testset "sfmodel_MixTable against the exact quantiles" begin
        tab = redirect_stdout(devnull) do; sfmodel_MixTable(); end
        @test tab[:, 1] == 1:40
        @test tab[6, 5] ≈ 16.074 atol = 5e-4
        @test tab[12, 3] ≈ 20.401 atol = 5e-4
        # Every entry within 0.003 of the quantile of ½χ²(dof − 1) + ½χ²(dof).
        errs = [abs(tab[k, col] - SFrontiers._mixed_quantile(k, 1 - level))
                for k in 1:40 for (col, level) in ((2, 0.10), (3, 0.05), (4, 0.025), (5, 0.01))]
        @test maximum(errs) <= 0.003
    end
end
