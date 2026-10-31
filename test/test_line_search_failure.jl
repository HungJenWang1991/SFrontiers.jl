# Regression tests: the result after a failed line search describes one point.
#
# When the line search fails, Optim 2.0 has already moved x by the step the line search
# returned and stops before evaluating the objective and gradient there: its minimizer is the
# new point, its minimum and g_residual those of the previous iterate. Every backend took
# `coeff`, the Hessian and the standard errors from the minimizer but `loglikelihood`,
# `gradient_norm` and the gradient part of `redflag` from the previous iterate, so one result
# described two points. They now all describe `coeff`. The check below is independent of the
# code under test: the same model evaluated at the reported `coeff` by a fit of zero
# iterations. Under Optim 2.0 (the package's Manifest: Optim 2.0.1, LineSearches 7.6.0) it
# fails on the code before the fix: 14 of the 34 asserts, at least one in every testset.
#
# In Optim 2.2.1, 2.2.2 and 2.3.2 (the versions read; 2.1.0, 2.2.0, 2.3.0 and 2.3.1 were not
# checked), a failed line search no longer moves x (GradientDescent and Newton propose a trial
# point and commit it only after the line search succeeds), so Optim's own values already
# describe the minimizer and the line-search checks pass on the code before the fix as well.
# Under Optim 2.3.2 only the stop on an objective increase still tells the two apart (2 of the
# 34 asserts fail on the code before the fix): Optim still returns the previous point with the
# gradient of the rejected one. The fixed code passes under every version tried (Optim 2.0.1,
# 2.2.1, 2.2.2, 2.3.2).
#
# The line-search failure is forced by the test, as in test_fit_report_names_flags.jl, which is
# valid under LineSearches 7.6 and 7.8. LS.BackTracking(iterations = 0) gives the same fits under
# Optim 2.0.1, bit for bit, but LineSearches 7.8 refuses it with an ArgumentError (Optim 2.3
# requires 7.8).

using Test
using SFrontiers
using Random
using Optim

@testset "after a line-search failure the result describes one point" begin
    LS = Optim.LineSearches
    # gradient descent whose line search fails on its first call, with the step of 1e-2 that
    # InitialStatic proposes: the run ends on iteration 1 with ls_failed (under Optim 2.0, x has
    # then moved by that step). BackTracking(iterations = 0), used by an earlier form of this
    # test, ended every fit here the same way: its first step failed the Armijo test.
    fail_at_once = (d, x, s, alpha, x_ls, phi0, dphi0) ->
        throw(LS.LineSearchException("line search failure forced by the test", alpha))
    gd_fail = sfmodel_opt(main_solver = GradientDescent(alphaguess = LS.InitialStatic(alpha = 1e-2),
                                                        linesearch = fail_at_once),
                          main_opt = (iterations = 100,))
    # Newton with a HagerZhang line search allowed a single step: it fails with the best
    # point it tried (Newton's default line search, cut short). On this data that point is the
    # start (step 0) under Optim 2.0 and 2.3, so x does not move and these two cases pass on the
    # code before the fix as well
    newton_hz1 = sfmodel_opt(main_solver = Newton(linesearch = LS.HagerZhang(linesearchmax = 1)),
                             main_opt = (iterations = 100,))
    # a static step that overshoots, with allow_f_increases = false: Optim returns the previous
    # point but the gradient of the rejected one
    gd_up = sfmodel_opt(main_solver = GradientDescent(alphaguess = LS.InitialStatic(alpha = 0.05),
                                                      linesearch = LS.Static()),
                        main_opt = (iterations = 100, allow_f_increases = false))
    at_point = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,))
    meth(m) = m === :MLE ? sfmodel_method(method = :MLE) : sfmodel_method(method = m, n_draws = 256)
    fit(spec, m, init, opt) = redirect_stdout(devnull) do
        sfmodel_fit(spec = spec, method = meth(m), init = init, optim_options = opt,
                    marginal = false, show_table = false, verbose = false)
    end

    rng = Xoshiro(25); n = 300
    x = randn(rng, n); X = hcat(ones(n), x)
    y = X * [1.0, 0.5] .+ 0.3 .* randn(rng, n) .- abs.(0.6 .* randn(rng, n))
    spec = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal, type = :prod)
    good = [1.0, 0.5, log(0.36), log(0.09)]

    nf, T = 60, 5; np_ = nf * T
    idp = repeat(1:nf, inner = T)
    XP = hcat(randn(rng, np_), randn(rng, np_)); ZP = hcat(randn(rng, np_))
    yp = repeat(0.5 .* randn(rng, nf), inner = T) .+ XP * [0.6, -0.4] .+ 0.3 .* randn(rng, np_) .-
         exp.(0.3 .* ZP[:, 1]) .* repeat(abs.(0.5 .* randn(rng, nf)), inner = T)
    spp = sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = XP, zvar = ZP, id = idp,
                       noise = :Normal, ineff = :HalfNormal)
    goodp = [0.6, -0.4, 0.3, log(0.25), log(0.09)]

    for (label, sp, m, ini) in (("MLE", spec, :MLE, good), ("MCI", spec, :MCI, good),
                                ("MSLE", spec, :MSLE, good), ("panel MCI", spp, :MCI, goodp),
                                ("panel MSLE", spp, :MSLE, goodp))
        @testset "$label" begin
            r = fit(sp, m, ini, gd_fail)
            @test !r.converged && r.redflag >= 1          # the run stopped on the failed line search
            @test r.actual_iterations < 100
            e = fit(sp, m, collect(r.coeff), at_point)    # the same model at the reported coeff
            @test isapprox(r.loglikelihood, e.loglikelihood; rtol = 1e-12)
            @test isapprox(r.gradient_norm, e.gradient_norm; rtol = 1e-10)
        end
    end

    @testset "other stops: HagerZhang cut short, and an increase with allow_f_increases = false" begin
        for (label, m, opt) in (("MLE Newton+HZ", :MLE, newton_hz1), ("MCI Newton+HZ", :MCI, newton_hz1),
                                ("MLE f increased", :MLE, gd_up), ("MSLE f increased", :MSLE, gd_up))
            r = fit(spec, m, good, opt)
            e = fit(spec, m, collect(r.coeff), at_point)
            @test isapprox(r.loglikelihood, e.loglikelihood; rtol = 1e-12)
            @test isapprox(r.gradient_norm, e.gradient_norm; rtol = 1e-10)
        end
    end

    @testset "the printed log-likelihood is the reported one" begin
        for m in (:MLE, :MCI)
            path, io = mktemp()
            r = redirect_stdout(io) do
                sfmodel_fit(spec = spec, method = meth(m), init = good, optim_options = gd_fail,
                            marginal = false, show_table = true, verbose = false)
            end
            close(io)
            txt = read(path, String)
            line = match(r"Log-likelihood: *(-?[0-9.]+)", txt)
            @test line !== nothing
            @test parse(Float64, line.captures[1]) == round(r.loglikelihood; digits = 5)
            # and it is the log-likelihood at the reported coefficients (before the fix, both the
            # printed and the reported value were the previous iterate's)
            e = fit(spec, m, collect(r.coeff), at_point)
            @test isapprox(parse(Float64, line.captures[1]), e.loglikelihood; atol = 1e-5)
        end
    end
end
