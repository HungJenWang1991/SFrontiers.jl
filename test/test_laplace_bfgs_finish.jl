# Laplace noise: an MCI or MSLE fit whose main solver stops without convergence continues from its endpoint with
# BFGS. Checked here: the finish after an unconverged Newton stage; no finish for normal noise, for the BFGS and
# LBFGS main solvers, or for a fit held at its starting values; the Newton cap of the default settings; the printed
# lines; and the integration check at the finish's point.

using Test, SFrontiers, Random
using Optim: BFGS, LBFGS, NelderMead, Newton

@testset "Laplace noise: the BFGS finish after an unconverged main solver" begin
    # Data from Xoshiro(21), drawn in this order: x, the two uniforms of the Laplace draw (b = 0.3), the half-normal
    # draw (σᵤ = 0.6); production frontier y = 1 + 0.5x + v - u. The normal noise of the control fits, 0.3·randn, is
    # drawn from the same generator after them, so x and u are those of the Laplace data.
    function _lbf_data(n)
        rng = Random.Xoshiro(21)
        x = randn(rng, n); U1 = rand(rng, n); U2 = rand(rng, n); z = randn(rng, n)
        X = hcat(ones(n), x)
        u = abs.(0.6 .* z)
        y = X * [1.0, 0.5] .+ 0.3 .* (log.(U1) .- log.(U2)) .- u
        yN = X * [1.0, 0.5] .+ 0.3 .* randn(rng, n) .- u
        return X, y, yN
    end
    _lbf_fit(sp, m, init, opt; show = false, verb = false, ic = false) =
        sfmodel_fit(spec = sp, method = sfmodel_method(method = m, n_draws = 128), init = init,
                    optim_options = opt, marginal = false, integration_check = ic, show_table = show,
                    verbose = verb)
    _lbf_newton() = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 100, g_abstol = 1e-6))
    # NelderMead warmstart (200 iterations, g_abstol = 1e-3), then Newton with the main settings `main_opt`
    _lbf_twostage(main_opt) = sfmodel_opt(warmstart_solver = NelderMead(),
                                          warmstart_opt = (iterations = 200, g_abstol = 1e-3),
                                          main_solver = Newton(), main_opt = main_opt)
    # The value of f() and what it printed.
    function _lbf_capture(f)
        path, io = mktemp()
        r = redirect_stdout(f, io)
        close(io)
        txt = read(path, String)
        rm(path)
        return r, txt
    end

    n1 = 300
    X1, y1, y1N = _lbf_data(n1)
    sp = sfmodel_spec(depvar = y1, frontier = X1, noise = :Laplace, ineff = :HalfNormal, type = :prod)
    init = sfmodel_init(spec = sp, frontier = X1 \ y1, ln_sigma_u_sq = log(0.3), ln_b = log(0.2), message = false)
    spN = sfmodel_spec(depvar = y1N, frontier = X1, noise = :Normal, ineff = :HalfNormal, type = :prod)
    initN = sfmodel_init(spec = spN, frontier = X1 \ y1N, ln_sigma_u_sq = log(0.3), ln_sigma_v_sq = log(0.05),
                         message = false)

    # The first fits: Newton with at most 100 iterations, no warmstart. The MCI fit also prints its progress and
    # table and runs the integration check (the sets on the printed lines and on the check below use it).
    rM, outM = _lbf_capture(() -> _lbf_fit(sp, :MCI, init, _lbf_newton();
                                           show = true, verb = true, ic = true))
    rS = _lbf_fit(sp, :MSLE, init, _lbf_newton())

    @testset "an unconverged Newton stage continues with BFGS, $m" for (m, r) in ((:MCI, rM), (:MSLE, rS))
        if r.finish !== nothing
            @test r.finish.solver == "BFGS"
            @test r.finish.main_converged == false
            @test r.finish.main_iterations <= 100
            @test r.finish.main_solver == "Newton"
            @test r.finish.newton_cap === nothing
            @test r.converged == true
            # The finish may stop where the objective no longer changes, with the gradient norm above
            # g_abstol; how close it gets depends on the versions of the optimization packages.
            @test r.gradient_norm <= 1e-4
            @test r.actual_iterations == r.finish.main_iterations + r.finish.iterations
            @test r.main_solver isa Newton
            @test r.redflag == 0
            @test r.vcov_type === :OPG
        else
            # Newton converged, which the kinks of the Laplace likelihood make rare
            @test r.converged
        end
    end

    @testset "no finish for normal noise, $m" for m in (:MCI, :MSLE)
        r, _ = _lbf_capture(() -> _lbf_fit(spN, m, initN, sfmodel_opt(main_solver = Newton(),
                                                                      main_opt = (iterations = 2,))))
        @test r.finish === nothing
        @test r.converged == false
    end

    @testset "no finish after BFGS or LBFGS, $m" for m in (:MCI, :MSLE)
        for solver in (BFGS(), LBFGS())
            r, _ = _lbf_capture(() -> _lbf_fit(sp, m, init, sfmodel_opt(main_solver = solver,
                                                                         main_opt = (iterations = 300,))))
            @test r.finish === nothing
        end
    end

    @testset "no finish for a fit held at its start, $m" for m in (:MCI, :MSLE)
        r, _ = _lbf_capture(() -> _lbf_fit(sp, m, init, sfmodel_opt(main_solver = Newton(),
                                                                     main_opt = (iterations = 0,))))
        @test r.finish === nothing
        @test r.coeff == init
    end

    @testset "the Newton cap of the default settings, $m" for m in (:MCI, :MSLE)
        # the default settings written out
        r = _lbf_fit(sp, m, init, _lbf_twostage((iterations = 200, g_abstol = 1e-7)))
        if r.finish !== nothing
            @test r.finish.newton_cap == 30
            @test r.finish.main_iterations <= 30
        else
            @test r.converged      # Newton converged within its 30 iterations
        end
        # other main settings are the user's: no cap
        r, _ = _lbf_capture(() -> _lbf_fit(sp, m, init, _lbf_twostage((iterations = 50, g_abstol = 1e-6))))
        @test r.finish === nothing || r.finish.newton_cap === nothing
        r, _ = _lbf_capture(() -> _lbf_fit(sp, m, init,
                                           _lbf_twostage((iterations = 200, g_abstol = 1e-7, time_limit = 1.0e6))))
        @test r.finish === nothing || r.finish.newton_cap === nothing
        # normal noise under the default settings: no finish, so the cap leaves no trace
        r = _lbf_fit(spN, m, initN, nothing)
        @test r.finish === nothing
    end

    @testset "the printed lines of the finish" begin
        if rM.finish !== nothing
            @test occursin("(BFGS finish)", outM)
            @test occursin("BFGS finish:", outM)
            @test occursin("continuing with BFGS from its endpoint", outM)
            @test occursin("finish", sprint(show, MIME"text/plain"(), rM))
        end
    end

    @testset "the integration check after the finish" begin
        @test rM.integration_check.n_draws == 128
        @test isfinite(rM.integration_check.delta_loglik)
        @test rM.integration_check.loglik == rM.loglikelihood
    end
end
