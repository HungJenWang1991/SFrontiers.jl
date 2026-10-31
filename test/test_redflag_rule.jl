# Regression tests:
# `redflag` behaves as the manual describes, and the same way under every method.
#
# The manual's "Convergence Warnings" section defines redflag as an Int: 0 = no issue;
# 1 = gradient norm > 0.1 or NaN, Optim did not report convergence (which includes
# stopping at the iteration limit), a Hessian that is not invertible or is numerically
# singular (1-norm reciprocal condition number below machine epsilon), or a boundary
# solution (a variance or scale parameter at the simulation clamp, 1e-12 / 1e12);
# 2 = a non-positive var-cov diagonal. The highest applicable value is reported, except
# that a boundary solution is always 1.
#
# Before the fix: under :MLE redflag was a Bool (2 could not be stored) and "not
# converged" was set only inside `if show_table`; each backend carried its own Hessian
# test, so the same fit could read 0 (or its non-positive-diagonal `true`) under :MLE and 1
# under :MCI/:MSLE, with finite standard errors under one and NaN under the other; a fit that
# converged on its last allowed iteration was flagged; :MLE ran a different default
# optimizer and ignored every Optim option but the iteration caps and the main g_abstol;
# a warmstart of 0 iterations still ran under :MLE. Each testset below fails on that code.

using Test
using SFrontiers
using Random
using Optim
using LinearAlgebra

const _RF = SFrontiers.SFRedflag

@testset "redflag is the manual's 0/1/2, the same under every method" begin
    Random.seed!(20)
    n = 300
    x = randn(n); X = hcat(ones(n), x)
    y = X * [1.0, 0.5] .+ 0.3 .* randn(n) .- abs.(0.6 .* randn(n))
    spec = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal, type = :prod)
    meth(m) = m === :MLE ? sfmodel_method(method = :MLE) : sfmodel_method(method = m, n_draws = 512)
    fit(m, init, opt; st = false, sp = spec) = redirect_stdout(devnull) do
        sfmodel_fit(spec = sp, method = meth(m), init = init, optim_options = opt,
                    marginal = false, show_table = st, verbose = false)
    end
    newton(k) = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = k,))

    good = sfmodel_init(spec = spec, frontier = X \ y, ln_sigma_u_sq = log(0.36), ln_sigma_v_sq = log(0.09))
    far  = sfmodel_init(spec = spec, frontier = [0.0, 0.0], ln_sigma_u_sq = 3.0, ln_sigma_v_sq = -3.0)

    @testset "MLE: an Int, whatever show_table is" begin
        for st in (false, true)
            # 0: a clean fit
            r = fit(:MLE, good, newton(100); st = st)
            @test r.redflag === 0
            @test r.converged

            # 2: the far start itself (no iteration). Optim did not converge (1) and the
            # var-cov diagonal is not positive (2); the higher value is reported.
            r = fit(:MLE, far, newton(0); st = st)
            @test r.redflag === 2
            @test any(isnan, stderror(r)) && !all(isnan, stderror(r))

            # 1: one Newton step from the far start lands at ln σᵤ² ≈ -168, a boundary
            # solution with a numerically singular Hessian: inv succeeds there, but the
            # reciprocal condition number is about 1e-38, so every standard error is NaN.
            # Before the fix :MLE read `true` with one standard error NaN (the non-positive-
            # diagonal branch), where MCI and MSLE read 1 with all of them NaN.
            r = fit(:MLE, far, newton(1); st = st)
            @test r.coeff[3] < -100
            @test r.redflag === 1
            @test r.iter_limit_reached
            @test all(isnan, stderror(r))
        end

        # 1: an exactly singular Hessian (a zero frontier column), with or without the table,
        # where the collinearity diagnostic runs as well
        Xs = hcat(X, zeros(n))
        specs = sfmodel_spec(depvar = y, frontier = Xs, noise = :Normal, ineff = :HalfNormal, type = :prod)
        inits = sfmodel_init(spec = specs, frontier = [X \ y; 0.0], ln_sigma_u_sq = log(0.36), ln_sigma_v_sq = log(0.09))
        optnm = sfmodel_opt(main_solver = NelderMead(), main_opt = (iterations = 2000,))
        flags = [fit(:MLE, inits, optnm; st = st, sp = specs).redflag for st in (false, true)]
        @test flags == [1, 1]
    end

    @testset "the shared Hessian rule" begin
        cl(H; kw...) = _RF.sf_classify_hessian(H; message = false, kw...)
        @test cl([2.0 0.5; 0.5 1.0]).redflag == 0
        r = cl([1.0 1.0; 1.0 1.0])                     # inv throws
        @test r.redflag == 1 && all(isnan, r.var_cov_matrix)
        @test cl([1.0 0.0; 0.0 -1.0]).redflag == 2
        B = [1.0 2.0; 3.0 4.0; 5.0 7.0]
        r = cl(B * B')                                 # rank 2 (inv throws on this one)
        @test r.redflag == 1 && all(isnan, r.var_cov_matrix)
        @test cl([1.0 0.0; 0.0 NaN]).redflag == 1      # a NaN Hessian read 2 before
        @test cl([1.0 0.0; 0.0 1e-17]).redflag == 1    # cond 1e17, below eps: read 0 before
        @test cl([1.0 0.0; 0.0 1e-15]).redflag == 0    # cond 1e15 is still invertible
        # a boundary solution is 1, over a clean Hessian or over 2; the matrix is kept
        r = cl([2.0 0.5; 0.5 1.0]; at_clamp = true)
        @test r.redflag == 1 && r.hessian_flag == 0 && all(isfinite, r.var_cov_matrix)
        @test cl([1.0 0.0; 0.0 -1.0]; at_clamp = true).redflag == 1
        # the verdict does not depend on the parameter order (the TN-scaling and panel-TFE
        # results are classified in the analytic order, then permuted)
        rng = Xoshiro(20)
        agree = 0
        for _ in 1:300
            Q = Matrix(qr(randn(rng, 4, 4)).Q)
            H = Q * Diagonal([1.0, 1.0, 1.0, 10.0^(-rand(rng, 13:19))]) * Q'
            H = (H + H') / 2
            P = [1, 2, 4, 3]
            agree += cl(H).redflag == cl(H[P, P]).redflag
        end
        @test agree == 300
        @test _RF.sf_scale_at_clamp(1e-12) && _RF.sf_scale_at_clamp(1e12) && _RF.sf_scale_at_clamp(NaN)
        @test !_RF.sf_scale_at_clamp(1.0001e-12) && !_RF.sf_scale_at_clamp(0.9999e12)
        # the backends' own entry points apply the same rule
        H = [1.0 0.0; 0.0 1e-17]
        @test SFrontiers.MCI_Backend._invert_hessian(H; message = false).redflag == 1
        @test SFrontiers.MSLE_Backend._invert_hessian(H; message = false).redflag == 1
        @test SFrontiers.Panel_Backend._panel_invert_hessian(H; message = false).redflag == 1
        # an exactly zero var-cov diagonal (redflag 2) gives a NaN standard error, not 0
        @test isnan(SFrontiers.MCI_Backend._stderr_from_vcov([1.0 0.0; 0.0 0.0])[2])
        @test isnan(SFrontiers.MLE_Backend.SFmle._stderr_from_vcov([1.0 0.0; 0.0 0.0])[2])
    end

    @testset "the same point gets the same flag under every method" begin
        at(m, p) = fit(m, p, newton(0))
        # the MLE one-step landing point: all standard errors NaN under every method now;
        # before, :MLE kept three of its four
        p1 = collect(fit(:MLE, far, newton(1)).coeff)
        @test [at(m, p1).redflag for m in (:MLE, :MCI, :MSLE)] == [1, 1, 1]
        @test all(m -> all(isnan, stderror(at(m, p1))), (:MLE, :MCI, :MSLE))
        # the clean optimum with ln σᵤ² moved past the clamp (about -55.26). Under
        # :MLE the Hessian is still invertible there, so the boundary rule decides, and the
        # standard errors are kept as computed.
        pc = collect(fit(:MLE, good, newton(100)).coeff)
        pb = copy(pc); pb[3] = -60.0
        rb = at(:MLE, pb)
        @test [rb.redflag; [at(m, pb).redflag for m in (:MCI, :MSLE)]] == [1, 1, 1]
        @test !all(isnan, stderror(rb))

        # the Exponential: the simulation clamp is on σᵤ² itself, ln σᵤ² ∉ (-27.63, 27.63)
        spe = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :Exponential, type = :prod)
        pe = [pc[1], pc[2], -30.0, pc[4]]
        @test [fit(m, pe, newton(0); sp = spe).redflag for m in (:MLE, :MCI, :MSLE)] == [1, 1, 1]

        # a duplicated regressor. On these data inv throws under all three methods (whether
        # it does is rounding luck), so this block pins only the parity; the panel case below
        # is one where :MLE's inv succeeds and the reciprocal condition number alone decides.
        Xd = hcat(X, X[:, 2])
        spd = sfmodel_spec(depvar = y, frontier = Xd, noise = :Normal, ineff = :HalfNormal, type = :prod)
        pd = sfmodel_init(spec = spd, frontier = [X \ y; 0.0], ln_sigma_u_sq = log(0.36), ln_sigma_v_sq = log(0.09))
        optnm = sfmodel_opt(main_solver = NelderMead(), main_opt = (iterations = 2000,))
        for m in (:MLE, :MCI, :MSLE)
            r = fit(m, pd, optnm; sp = spd)
            @test r.redflag == 1
            @test all(isnan, stderror(r))
        end

        # a converged fit whose Hessian inv accepts (cond 2.6e18) but whose reciprocal
        # condition number (3e-17) is below eps: before the fix :MLE read `false` with
        # standard errors up to 1.5e6, where MCI read 1 (its inv throws). The data are a
        # panel_TFE duplicated-frontier case (MersenneTwister(201)).
        rng2 = MersenneTwister(201); N2, T2 = 100, 5; NT2 = N2 * T2
        id2 = repeat(1:N2, inner = T2); x2 = randn(rng2, NT2); z2 = randn(rng2, NT2)
        al2 = repeat(0.5 .* randn(rng2, N2), inner = T2); v2 = 0.3 .* randn(rng2, NT2)
        u2 = exp.(0.5 .* z2) .* repeat(abs.(0.6 .* randn(rng2, N2)), inner = T2)
        sp2 = sfmodel_spec(depvar = al2 .+ 0.5 .* x2 .+ v2 .- u2, frontier = hcat(x2, x2),
                           zvar = reshape(z2, :, 1), id = id2, noise = :Normal, ineff = :HalfNormal,
                           type = :prod, datatype = :panel_TFE)
        in2 = sfmodel_init(spec = sp2, frontier = [0.25, 0.25], scaling = [0.3],
                           ln_sigma_u_sq = log(0.36), ln_sigma_v_sq = log(0.09))
        for m in (:MLE, :MCI)
            r = fit(m, in2, newton(100); sp = sp2)
            @test r.converged
            @test r.redflag === 1
            @test all(isnan, stderror(r))
        end

        # the panel backend: panel_TFE half-normal, ln σᵤ² past the clamp, MLE and MCI
        rngp = Xoshiro(2020)
        nf, T = 60, 5
        np_ = nf * T
        idp = repeat(1:nf, inner = T)
        XP = hcat(randn(rngp, np_), randn(rngp, np_))
        ZP = hcat(randn(rngp, np_))
        yp = XP * [0.6, -0.4] .- abs.(0.4 .* randn(rngp, np_)) .+ 0.3 .* randn(rngp, np_)
        spp = sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = XP, zvar = ZP, id = idp,
                           noise = :Normal, ineff = :HalfNormal)
        pp = [0.6, -0.4, 0.1, -60.0, log(0.09)]     # [frontier, δ, ln σᵤ², ln σᵥ²], one order for both
        @test [fit(m, pp, newton(0); sp = spp).redflag for m in (:MLE, :MCI)] == [1, 1]
        pp[4] = log(0.16)
        @test fit(:MCI, pp, sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 100,)); sp = spp).redflag == 0
    end

    @testset "the iteration limit counts only when the fit did not converge" begin
        for m in (:MLE, :MCI, :MSLE)
            need = fit(m, good, newton(100)).actual_iterations
            r = fit(m, good, newton(need))              # converges on its last allowed iteration
            @test r.converged && r.iter_limit_reached
            @test r.redflag == 0                        # was 1
            r = fit(m, good, newton(need - 1))
            @test !r.converged && r.redflag >= 1
        end
    end

    @testset "one default optimizer for every method" begin
        for m in (:MLE, :MCI, :MSLE)
            r = fit(m, nothing, nothing)
            @test r.warmstart_solver isa NelderMead
            @test r.warmstart_maxIT == 200
            @test r.main_solver isa Newton
            @test r.main_maxIT == 200
            @test r.main_tolerance == 1e-7
            @test r.redflag == 0
        end
    end

    @testset "MLE honours every Optim option" begin
        # a callback that stops the run: ignored under :MLE before, which then converged
        cb = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 100, callback = st -> true))
        for m in (:MLE, :MCI)
            r = fit(m, good, cb)
            @test r.actual_iterations == 0
            @test !r.converged && r.redflag >= 1
        end
        # the warmstart's own tolerance: a huge one stops NelderMead at once. Before, the
        # warmstart ran at the main tolerance and used all 50 iterations.
        opt_ws = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 50, g_abstol = 1e10),
                             main_solver = Newton(), main_opt = (iterations = 100,))
        r = fit(:MLE, good, opt_ws)
        r2 = fit(:MLE, r.main_ini, newton(100))
        @test r.actual_iterations == r2.actual_iterations     # 0 warmstart iterations
        # actual_iterations counts the warmstart too (it counted the main stage only)
        opt5 = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 5, g_abstol = 1e-12),
                           main_solver = Newton(), main_opt = (iterations = 100,))
        r = fit(:MLE, good, opt5)
        @test r.warmstart_ini == collect(good)
        @test r.actual_iterations == 5 + fit(:MLE, r.main_ini, newton(100)).actual_iterations
    end

    @testset "a warmstart of 0 iterations is skipped, and the result fields match" begin
        opt0 = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 0,),
                           main_solver = Newton(), main_opt = (iterations = 100,))
        for m in (:MLE, :MCI, :MSLE)
            r = fit(m, good, opt0)
            @test r.warmstart_solver === nothing
            @test r.warmstart_ini === nothing           # :MLE stored 0
            @test r.main_ini == collect(good)           # :MLE started from the simplex centroid
            @test r.main_tolerance == 1e-8              # absent under :MLE before
        end
        @test fit(:MLE, good, opt0).tolerance == 1e-8   # kept: the bootstrap reads it
    end

    @testset "the MLE bootstrap no longer warns in every replicate" begin
        Random.seed!(2024)
        Nb = 200
        Xb = hcat(ones(Nb), randn(Nb)); zb = randn(Nb)
        yb = Xb * [1.0, 0.5] .+ 0.3 .* randn(Nb) .- abs.(0.5 .* randn(Nb) .* exp.(0.3 .* zb))
        spb = sfmodel_spec(depvar = yb, frontier = Xb, zvar = hcat(ones(Nb), zb), noise = :Normal,
                           ineff = :HalfNormal, hetero = [:sigma_u_sq], type = :prod)
        ib = sfmodel_init(spec = spb, frontier = Xb \ yb, ln_sigma_u_sq = [log(0.25), 0.0], ln_sigma_v_sq = log(0.09))
        rb = redirect_stdout(devnull) do
            sfmodel_fit(spec = spb, method = sfmodel_method(method = :MLE), init = ib,
                        optim_options = newton(100), marginal = true, show_table = false, verbose = false)
        end
        logs, out = Test.collect_test_logs() do
            redirect_stdout(devnull) do
                SFrontiers.MLE_Backend.sfmodel_boot_marginal(result = rb, R = 5, seed = 1, every = 1000)
            end
        end
        @test !any(l -> occursin("deprecated", string(l.message)), logs)
        @test all(x -> x isa Real && isfinite(x), out[:, 1])
    end
end
