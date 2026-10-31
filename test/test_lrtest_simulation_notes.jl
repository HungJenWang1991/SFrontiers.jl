# lrtest and the draw settings of simulated fits: which log-likelihoods are simulated, the warning when two simulated
# fits were not computed with the same draws or when a simulated log-likelihood meets one computed without
# simulation, the printed simulation line, and the seven-argument constructor of LRTestResult. Data, fits and helpers
# stay inside the outer test set, because runtests.jl includes every test file into one module whose constants
# include `X` and `y`.

using SFrontiers, Test, Random, Optim

@testset "lrtest and the draw settings of simulated fits" begin
    # The design of runtests.jl's shared data: X = [1, x1, x2], β = [1.0, 0.5, −0.3], normal noise, half-normal u.
    rng = MersenneTwister(4242)
    n = 200
    Xd = hcat(ones(n), randn(rng, n), randn(rng, n))
    vd = 0.3 .* randn(rng, n)
    ud = abs.(0.5 .* randn(rng, n))
    yd = Xd * [1.0, 0.5, -0.3] .+ vd .- ud
    sp = sfmodel_spec(depvar = yd, frontier = Xd, noise = :Normal, ineff = :HalfNormal, type = :prod)
    opt = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40,),
                      main_solver = Newton(), main_opt = (iterations = 30,))
    _lrs_fit(meth) = sfmodel_fit(spec = sp, method = meth, optim_options = opt,
                                 marginal = false, show_table = false, verbose = false)
    _lrs_shown(lr) = sprint(show, MIME"text/plain"(), lr)

    r_mle     = _lrs_fit(sfmodel_method(method = :MLE))
    r_s64     = _lrs_fit(sfmodel_method(method = :MSLE, n_draws = 64))
    r_s128    = _lrs_fit(sfmodel_method(method = :MSLE, n_draws = 128))
    r_c64     = _lrs_fit(sfmodel_method(method = :MCI, n_draws = 64))
    r_c64e    = _lrs_fit(sfmodel_method(method = :MCI, n_draws = 64, transformation = :expo_rule))
    r_sh      = _lrs_fit(sfmodel_method(method = :MSLE, n_draws = 64, multiRand = false))
    # N·S = 12,800 is below the cap, so these draws equal the default's.
    r_rec     = _lrs_fit(sfmodel_method(method = :MSLE, n_draws = 64, recycling = true))
    r_rec1023 = _lrs_fit(sfmodel_method(method = :MSLE, n_draws = 64, recycling = 1023))
    U = 0.001 .+ 0.998 .* rand(MersenneTwister(7), 200, 64)
    r_u       = _lrs_fit(sfmodel_method(method = :MSLE, draws = U))

    @testset "which log-likelihoods are simulated" begin
        lr = @test_logs lrtest(r_mle, r_mle; dof = 1)
        @test lr.simulated === :none
        @test lr.simulation_note == ""
        @test !occursin("simulation", _lrs_shown(lr))

        lr = @test_logs lrtest(r_s64, r_s64; dof = 1)
        @test lr.simulated === :both
        @test occursin("same draw settings", lr.simulation_note)
        @test occursin("same draw settings", _lrs_shown(lr))
        @test occursin("sfmodel_fit(..., integration_check = true)", lr.simulation_note)
        @test occursin("sfmodel_fit(..., integration_check = true)", _lrs_shown(lr))

        lr = @test_logs (:warn, r"not computed with the same draws") lrtest(r_s64, r_s128; dof = 1)
        @test occursin("n_draws 64 vs 128", lr.simulation_note)

        lr = @test_logs (:warn, r"not computed with the same draws") lrtest(r_s64, r_c64; dof = 1)
        @test occursin("estimation_method", lr.simulation_note)

        lr = @test_logs (:warn, r"not computed with the same draws") lrtest(r_c64, r_c64e; dof = 1)
        @test occursin("transformation logistic_1_rule vs expo_rule", lr.simulation_note)

        lr = @test_logs (:warn, r"not computed with the same draws") lrtest(r_s64, r_sh; dof = 1)
        @test occursin("multiRand", lr.simulation_note)

        @test r_rec.draw_scheme === :halton_recycled
        @test_logs lrtest(r_s64, r_rec; dof = 1)
        lr = @test_logs (:warn, r"not computed with the same draws") lrtest(r_s64, r_rec1023; dof = 1)
        @test occursin("distinct_Halton_length", lr.simulation_note)

        lr = @test_logs (:warn, r"user-supplied draws.*partly cancel") lrtest(r_u, r_u; dof = 1)
        @test lr.simulated === :both

        lr = @test_logs (:warn, r"computed without simulation") lrtest(r_mle, r_s64; dof = 1)
        @test lr.simulated === :unrestricted
        lr = @test_logs (:warn, r"computed without simulation") lrtest(r_s64, r_mle; dof = 1)
        @test lr.simulated === :restricted
    end

    @testset "the seven-argument constructor" begin
        lr = LRTestResult(1.0, 1, 0.5, false, (p10 = 1.0, p05 = 2.0, p025 = 3.0, p01 = 4.0), -2.0, -1.5)
        @test lr.simulated === :none
        @test lr.simulation_note == ""
        @test occursin("Likelihood-ratio test", _lrs_shown(lr))
    end

    @testset "sf_vs_ols prints no simulation line" begin
        t = @test_logs sf_vs_ols(r_s64)
        @test t.simulated === :unrestricted
        @test !occursin("simulation", _lrs_shown(t))
    end

    @testset "panel_TFE transformation rules" begin
        # The panel data of the diagnostics tests for panel-sim, CSW and TRE fits.
        Random.seed!(303)
        Nf = 30; Tp = 5; NT = Nf * Tp; id = repeat(1:Nf, inner = Tp)
        Xp = hcat(randn(NT), randn(NT))
        αi = repeat(2 .* randn(Nf), inner = Tp)
        zp = randn(NT)
        yp = αi .+ Xp * [0.6, -0.5] .+ 0.3 .* randn(NT) .- abs.(0.5 .* randn(NT))
        psp = sfmodel_spec(depvar = yp, frontier = Xp, zvar = zp, noise = :Normal, ineff = :HalfNormal,
                           type = :prod, datatype = :panel_TFE, id = id)
        popt = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40,),
                           main_solver = NelderMead(), main_opt = (iterations = 200,))
        _lrs_pfit(meth) = sfmodel_fit(spec = psp, method = meth, optim_options = popt,
                                      marginal = false, show_table = false, verbose = false)
        r_pd = _lrs_pfit(sfmodel_method(method = :MCI, n_draws = 32))
        r_p1 = _lrs_pfit(sfmodel_method(method = :MCI, n_draws = 32, transformation = :logistic_1_rule))
        r_p2 = _lrs_pfit(sfmodel_method(method = :MCI, n_draws = 32, transformation = :logistic_2_rule))
        @test getfield(r_pd, :backend) === :Panel
        @test_logs lrtest(r_pd, r_p1; dof = 1)
        lr = @test_logs (:warn, r"not computed with the same draws") lrtest(r_pd, r_p2; dof = 1)
        @test occursin("transformation logistic_1_rule vs logistic_2_rule", lr.simulation_note)
    end
end
