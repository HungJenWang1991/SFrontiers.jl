# Regression tests: MLE table names, gradient_norm of gradient-free solvers, failed line searches flagged.
#
# (g) Under :MLE, for the truncated normal with hetero = :scaling and for panel_TFE with a
#     truncated normal, user-supplied varnames (canonical order: frontier, scaling, mu, ...)
#     were put by position on the analytic backend's rows (frontier, mu, scaling, ...): the
#     table printed during the fit, result.table and result.list[:varnames] mislabelled the mu
#     and scaling rows. Every row now carries its own name.
# (h) With a gradient-free main solver, gradient_norm was Optim's g_residual: NelderMead's
#     simplex spread, NaN for ParticleSwarm and SimulatedAnnealing. It is now the sup-norm of
#     the ForwardDiff gradient at coeff (SFRedflag.sf_final_fg); the check below is a fit of
#     zero iterations at the reported coeff, whose gradient norm Optim computes itself. A
#     NelderMead run that converges on a loose g_abstol far from the optimum is now flagged.
# (i) A fit stopped by a failed line search that Optim reports as converged (the f test met
#     on the previous iteration, here through a user f_reltol) could read redflag = 0. It now
#     reads 1 (SFRedflag.sf_convergence_flag). The failure is forced by a line search that
#     accepts a fixed step once and then throws (valid under Optim 2.0 and 2.3); a fit whose
#     line search fails while Optim reports it as not converged keeps its old message.
# The fit-level checks fail on the code before the fix; so do the unit checks of the shared
# functions, which do not exist there. Every solver run has an iteration and a time limit.

using Test
using SFrontiers
using Random
using Optim
using Printf
using ForwardDiff

@testset "MLE table names, gradient-free gradient_norm, failed line searches flagged" begin
    LS = Optim.LineSearches
    NLB = Optim.NLSolversBase
    RF = SFrontiers.SFRedflag
    TL = 60.0                                           # time limit (s) of every solver run
    at_point = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0, time_limit = TL))
    newton = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 100, time_limit = TL))
    meth(m) = m === :MLE ? sfmodel_method(method = :MLE) : sfmodel_method(method = m, n_draws = 256)
    function capture(f)
        path, io = mktemp()
        r = redirect_stdout(f, io)
        close(io)
        txt = read(path, String); rm(path; force = true)
        return r, txt
    end
    fitc(sp, m, init, opt; table = false) = capture(() -> sfmodel_fit(spec = sp, method = meth(m), init = init,
                                                                     optim_options = opt, marginal = false,
                                                                     show_table = table, verbose = false))
    fitq(sp, m, init, opt) = fitc(sp, m, init, opt)[1]
    # name => printed coefficient, from the rows of the printed estimation table
    function printed_rows(txt)
        rows = Dict{String, String}()
        for l in split(txt, '\n')
            occursin('│', l) || continue
            f = strip.(split(l, '│'))
            length(f) >= 5 && (rows[f[3]] = f[4])
        end
        return rows
    end

    @testset "(g) user varnames under :MLE" begin
        rng = Xoshiro(7); n = 400
        x = randn(rng, n); X = hcat(ones(n), x); z1 = randn(rng, n); z2 = randn(rng, n)
        y = X * [1.0, 0.5] .+ 0.3 .* randn(rng, n) .- exp.(0.4 .* z1 .- 0.2 .* z2) .* abs.(0.2 .+ 0.5 .* randn(rng, n))
        vn = ["CONST", "XX", "ZA", "ZB", "MU", "LNSU", "LNSV"]
        sp = sfmodel_spec(depvar = y, frontier = X, zvar = hcat(z1, z2), noise = :Normal,
                          ineff = :TruncatedNormal, hetero = :scaling, varnames = vn)
        r, txt = fitc(sp, :MLE, nothing, newton; table = true)
        mle = [1, 2, 5, 3, 4, 6, 7]                       # canonical positions in the MLE order
        @test r.converged
        @test r.varnames == vn && coeftable(r).rownms == vn            # published: unchanged
        @test collect(r.coeff) == collect(r.list[:coeff])[[1, 2, 4, 5, 3, 6, 7]]
        @test String.(r.table[2:end, 2]) == vn[mle]                    # stored table
        @test r.list[:varnames] == vn[mle]
        rows = printed_rows(txt)                                       # printed table
        @test all(rows[vn[i]] == @sprintf("%.4f", r.coeff[i]) for i in eachindex(vn))
        # auto names: the labels the backend has always produced
        sp0 = sfmodel_spec(depvar = y, frontier = X, zvar = hcat(z1, z2), noise = :Normal,
                           ineff = :TruncatedNormal, hetero = :scaling)
        r0 = fitq(sp0, :MLE, nothing, newton)
        @test r0.list[:varnames] == ["frontier_var1", "frontier_var2", "_cons", "hscale_var1",
                                     "hscale_var2", "_cons", "_cons"]
        @test r0.varnames == r0.list[:varnames][[1, 2, 4, 5, 3, 6, 7]]
        @test collect(r0.coeff) == collect(r.coeff)

        rng = Xoshiro(2021)
        nf, T = 80, 5; np_ = nf * T
        idp = repeat(1:nf, inner = T)
        XP = hcat(randn(rng, np_), randn(rng, np_)); ZP = hcat(randn(rng, np_))
        tn = Float64[]
        while length(tn) < nf
            d = 0.2 + 0.5 * randn(rng); d > 0 && push!(tn, d)
        end
        yp = repeat(0.5 .* randn(rng, nf), inner = T) .+ XP * [0.6, -0.4] .+ 0.3 .* randn(rng, np_) .-
             exp.(0.3 .* ZP[:, 1]) .* repeat(tn, inner = T)
        vnp = ["XA", "XB", "ZA", "MU", "LNSU", "LNSV"]
        spp = sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = XP, zvar = ZP, id = idp,
                           noise = :Normal, ineff = :TruncatedNormal, varnames = vnp)
        i0 = sfmodel_init(spec = spp, frontier = [0.6, -0.4], scaling = [0.3], mu = 0.2,
                          ln_sigma_u_sq = log(0.25), ln_sigma_v_sq = log(0.09))
        rp, txtp = fitc(spp, :MLE, i0, newton; table = true)
        mlep = [1, 2, 4, 3, 5, 6]
        @test rp.converged
        @test rp.varnames == vnp && coeftable(rp).rownms == vnp
        @test String.(rp.table[2:end, 2]) == vnp[mlep]
        @test rp.list[:varnames] == vnp[mlep]
        rowsp = printed_rows(txtp)
        @test all(rowsp[vnp[i]] == @sprintf("%.4f", rp.coeff[i]) for i in eachindex(vnp))
    end

    # the data of test_line_search_failure.jl
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
    optimum = Dict(lab => fitq(sp, m, ini, newton)
                   for (lab, sp, m, ini) in (("MLE", spec, :MLE, good), ("MCI", spec, :MCI, good),
                                             ("MSLE", spec, :MSLE, good), ("panel MCI", spp, :MCI, goodp)))

    @testset "(h) gradient-free main solver: gradient_norm is the gradient at coeff" begin
        nm60 = sfmodel_opt(main_solver = NelderMead(), main_opt = (iterations = 60, time_limit = TL))
        for (lab, sp, m, ini) in (("MLE", spec, :MLE, good), ("MCI", spec, :MCI, good),
                                  ("panel MCI", spp, :MCI, goodp))
            r = fitq(sp, m, ini, nm60)
            e = fitq(sp, m, collect(r.coeff), at_point)
            @test isapprox(r.gradient_norm, e.gradient_norm; rtol = 1e-10)   # was the simplex spread
            @test r.gradient_norm > 0.1 && r.redflag == 1
            @test isapprox(r.loglikelihood, e.loglikelihood; rtol = 1e-12)   # unchanged by the fix
        end
        # NelderMead converged on a loose g_abstol (simplex spread < 1e-2) at a point with a
        # large gradient: flagged, with the gradient message (it read redflag 0 before)
        nmloose = sfmodel_opt(main_solver = NelderMead(), main_opt = (iterations = 200, g_abstol = 1e-2, time_limit = TL))
        for m in (:MLE, :MCI)
            r, txt = fitc(spec, m, good, nmloose)
            @test r.converged && r.gradient_norm > 0.1
            @test r.redflag == 1 && occursin("gradients are problematic", txt)
        end
        # ParticleSwarm and SimulatedAnnealing started at the optimum: a finite gradient norm
        # below 0.1, and no gradient message; Optim never reports them converged, so the
        # iteration limit still gives redflag 1
        for (lab, m, solver, its) in (("MLE PS", :MLE, ParticleSwarm(n_particles = 6), 15),
                                      ("MCI PS", :MCI, ParticleSwarm(n_particles = 6), 15),
                                      ("MLE SA", :MLE, SimulatedAnnealing(), 50))
            Random.seed!(31)
            lab0 = split(lab)[1]
            r, txt = fitc(spec, m, collect(optimum[lab0].coeff),
                          sfmodel_opt(main_solver = solver, main_opt = (iterations = its, time_limit = TL)))
            e = fitq(spec, m, collect(r.coeff), at_point)
            @test isfinite(r.gradient_norm) && r.gradient_norm < 0.1
            @test isapprox(r.gradient_norm, e.gradient_norm; rtol = 1e-8, atol = 1e-10)
            @test !occursin("gradients are problematic", txt)
            @test !r.converged && r.redflag == 1 && occursin("reached the limit", txt)
        end
    end

    @testset "(i) a failed line search is flagged even when Optim reports convergence" begin
        # a line search that accepts the fixed step `a` on its first `nok` calls and then throws
        # (valid under LineSearches 7.6 and 7.8, Optim 2.0 and 2.3)
        function forced_ls(nok, a)
            k = Ref(0)
            return (d, x, s, alpha, x_ls, phi0, dphi0) -> begin
                k[] += 1
                k[] > nok && throw(LS.LineSearchException("line search failure forced by the test", a))
                x_ls .= x .+ a .* s
                (a, NLB.value!(d, x_ls))
            end
        end
        # converged + failed: f_reltol = 1e-2 is met on iteration 1; iteration 2 fails
        for (lab, sp, m) in (("MLE", spec, :MLE), ("MCI", spec, :MCI), ("MSLE", spec, :MSLE), ("panel MCI", spp, :MCI))
            rb = optimum[lab]
            dir = length(rb.coeff) == 4 ? [0.02, -0.02, 0.05, -0.05] : [0.02, -0.02, 0.03, 0.05, -0.05]
            calls = Ref(0)
            opt = sfmodel_opt(main_solver = GradientDescent(linesearch = forced_ls(1, 2e-4)),
                              main_opt = (iterations = 50, f_reltol = 1e-2, time_limit = TL,
                                          callback = st -> (calls[] += 1; false)))
            r, txt = fitc(sp, m, collect(rb.coeff) .+ 0.002 .* dir, opt)
            @test r.converged && r.actual_iterations == 2 && calls[] == 2   # converged, stopped by the line search
            @test r.gradient_norm < 0.1                                     # the gradient check passes
            @test r.redflag == 1
            @test occursin("The line search failed", txt)
        end
        # not converged + failed (the failure on iteration 1): the old message, not the new one
        for (lab, sp, m, ini) in (("MLE", spec, :MLE, good), ("MCI", spec, :MCI, good))
            opt = sfmodel_opt(main_solver = GradientDescent(linesearch = forced_ls(0, 1e-3)),
                              main_opt = (iterations = 50, time_limit = TL))
            r, txt = fitc(sp, m, ini, opt)
            @test !r.converged && r.redflag >= 1
            @test occursin("did not report convergence", txt) && !occursin("The line search failed", txt)
        end
    end

    @testset "shared functions (src/sf_redflag.jl)" begin
        f(x) = (x[1] - 1)^2 + 10 * (x[2] + 0.5)^2 + 0.1 * x[1]^4
        x0 = [0.3, 0.4]
        # gradient-based solver: Optim's own values, bit for bit, and no flag
        rn = optimize(f, x0, Newton(), Optim.Options(g_abstol = 1e-10, iterations = 50, time_limit = 30.0);
                      autodiff = Optim.ADTypes.AutoForwardDiff())
        @test Optim.converged(rn) && !RF.sf_gradient_free(rn) && !RF.sf_ls_failed(rn)
        @test RF.sf_final_fg(rn, f) === (Optim.minimum(rn), Optim.g_residual(rn))
        @test RF.sf_convergence_flag(rn; message = false) == 0
        # NelderMead (converged): its minimum is f at its minimizer; the gradient is ForwardDiff's there
        rm_ = optimize(f, x0, NelderMead(), Optim.Options(iterations = 40, time_limit = 30.0))
        xm = Optim.minimizer(rm_)
        @test Optim.converged(rm_)
        @test RF.sf_gradient_free(rm_) && Optim.minimum(rm_) == f(xm)
        @test RF.sf_final_fg(rm_, f) === (f(xm), maximum(abs, ForwardDiff.gradient(f, xm)))
        @test Optim.g_residual(rm_) != RF.sf_final_fg(rm_, f)[2]
        # ParticleSwarm: g_residual is NaN, the gradient is not; never converged
        Random.seed!(3)
        rp = optimize(f, x0, ParticleSwarm(n_particles = 4), Optim.Options(iterations = 30, time_limit = 30.0))
        @test isnan(Optim.g_residual(rp)) && isfinite(RF.sf_final_fg(rp, f)[2])
        @test RF.sf_convergence_flag(rp; message = false) == 1
    end
end
