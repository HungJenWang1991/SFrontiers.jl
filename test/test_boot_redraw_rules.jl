# Regression tests: the MLE bootstrap's redraw loop (cap, Ctrl-C, report).
#
# MLE_Backend.sfmodel_boot_marginal replaces a resample whose re-estimation fails (optimizer error,
# iteration cap, gradient above the tolerance, Hessian not invertible, a variance not positive,
# NaN or missing marginal effects) by a new draw. Before the fix it did so without limit and
# without a word: an `iter` below what every resample needs made it run forever, 40% of the
# resamples could be dropped at the default settings with nobody told, Ctrl-C was taken for a
# failed resample (the bootstrap could not be interrupted), an exception that escaped the loop
# left the _dicM equation names overridden, a resample in which every determinant was constant
# ended the run with MethodError(zero, Any), and with ParticleSwarm or SimulatedAnnealing as main
# solver no resample could ever be kept, so it never ended. Now it stops after 10R failures,
# rethrows InterruptException (also inside a TaskFailedException), restores the names in a
# `finally`, redraws the all-constant resample, stops a ParticleSwarm/SimulatedAnnealing fit
# before any draw, prints the failures by kind (verbose fits) and warns when more than 10% of the
# resamples drawn failed. The draws and results are unchanged: the testsets compare the bootstrap
# bit for bit with a copy of the loop as it was before the fix.
#
# The fix also repairs the verbose report: with
# `verbose = true` the final table threw a MethodError under PrettyTables 3. Four asserts depend on
# it: in "no failure ...", `!(ov isa Exception)`, `isequal(ov, o)` and the table text; in
# "failures below the cap ...", `isequal(ov, o)` at iter = 5.
#
# Every testset also terminates on the code before the fix (the endless cases are guarded).

using Test
using SFrontiers
using Random
using Optim
using Statistics
using LinearAlgebra
using StatsBase: sample
using ADTypes: AutoForwardDiff
import ForwardDiff

# A line search wrapping the solver's own: it counts its calls and, on the calls in `at`, throws
# `exc` (or calls it, when it is a function). Optim rethrows anything that is not a
# LineSearchException, so the exception leaves `optimize` inside a bootstrap replicate, where the
# redraw loop's catch sees it.
mutable struct _BF30InjectLS{L}
    inner::L
    calls::Int
    at::Set{Int}
    exc::Any
end
_BF30InjectLS(inner; at = Int[], exc = DomainError(-1.0, "injected by the test")) =
    _BF30InjectLS(inner, 0, Set{Int}(at), exc)
function (t::_BF30InjectLS)(args...)
    t.calls += 1
    t.calls in t.at && (t.exc isa Function ? t.exc() : throw(t.exc))
    return t.inner(args...)
end

# an exception whose message cannot be printed (its showerror throws)
struct _BF30BadShow <: Exception end
Base.showerror(io::IO, ::_BF30BadShow) = error("showerror of _BF30BadShow throws (test)")

# An in-test copy of the earlier cross-sectional redraw loop (SFmainfun.jl), recording the kind
# of each failure. `maxdraw` only protects this copy.
function _bf30_ref_boot(result; R, seed, iter = -1, level = 0.05, maxdraw = 5000)
    M = SFrontiers.MLE_Backend.SFmle
    data = M._dicM[:sdf]
    sf_init = copy(result.list[:coeff])
    sf_algo = result.main_solver
    sf_maxit = iter > 0 ? iter : result.main_maxIT
    sf_tol = result.tolerance
    (minfo1, minfo2, pos, num, eqvec, eqvec2, yvar, xvar, zvar, qvar, wvar, vvar, rowIDT, varlist) =
        M.getvar(result.modelid, data)
    yvar0 = deepcopy(yvar); xvar0 = deepcopy(xvar); zvar0 = deepcopy(zvar)
    qvar0 = deepcopy(qvar); wvar0 = deepcopy(wvar); vvar0 = deepcopy(vvar)
    sim_res = Array{Real}(undef, size(result.marginal, 2), R)
    rng = Xoshiro(seed)
    kinds = Symbol[]; ndrawn = 0
    saved = M._boot_common_marg_names!()
    try
        for i in 1:R
            @label again
            if ndrawn >= maxdraw
                return nothing, nothing, kinds, ndrawn
            end
            select_row = sample(rng, 1:size(data, 1), size(data, 1); replace = true); ndrawn += 1
            yvar = yvar0[select_row, :]; xvar = xvar0[select_row, :]
            (zvar0 == ()) || (zvar = zvar0[select_row, :]); (qvar0 == ()) || (qvar = qvar0[select_row, :])
            (wvar0 == ()) || (wvar = wvar0[select_row, :]); (vvar0 == ()) || (vvar = vvar0[select_row, :])
            n1 = size(yvar, 1)
            lik = rho -> M.LL_T(result.modelid, yvar, xvar, zvar, qvar, wvar, vvar, result.PorC, n1, pos, rho,
                                rowIDT, nothing)
            opt = try
                optimize(lik, sf_init, sf_algo,
                         Optim.Options(g_abstol = sf_tol, f_reltol = 0.0, f_abstol = 0.0, x_abstol = 0.0,
                                       x_reltol = 0.0, iterations = sf_maxit, store_trace = false,
                                       show_trace = false); autodiff = AutoForwardDiff())
            catch err
                push!(kinds, :optimizer); @goto again
            end
            if Optim.iteration_limit_reached(opt) || isnan(Optim.g_residual(opt)) || Optim.g_residual(opt) > sf_tol
                push!(kinds, Optim.iteration_limit_reached(opt) ? :itercap : :gradient); @goto again
            end
            b = Optim.minimizer(opt)
            H = ForwardDiff.hessian(lik, b)
            V = try
                inv(H)
            catch err
                push!(kinds, :inverse); @goto again
            end
            if !all(diag(V) .> 0)
                push!(kinds, :variance); @goto again
            end
            margeff, _ = M.get_marg(result.modelid, pos, num, b, zvar, qvar, wvar)
            theM = mean.(eachcol(margeff))
            if sum(isnan.(theM)) != 0
                push!(kinds, :margnan); @goto again
            end
            if length(theM) != size(sim_res, 1)
                push!(kinds, :margdim); @goto again
            end
            sim_res[:, i] = theM
        end
    finally
        M._boot_restore_marg_names!(saved)
    end
    sim_res = sim_res'
    theMean = mean.(eachcol(result.marginal))
    theBootMean = mean(sim_res, dims = 1)
    theSTD = sqrt.(sum((sim_res .- theBootMean) .^ 2, dims = 1) ./ (R - 1))
    ci_mat = sfmodel_CI(bootdata = sim_res, observed = theMean, level = level, verbose = false)
    return hcat(theSTD', ci_mat), sim_res, kinds, ndrawn
end

@testset "the MLE bootstrap's redraw loop (cap, Ctrl-C, report)" begin
    MB = SFrontiers.MLE_Backend
    M = MB.SFmle
    boot = MB.sfmodel_boot_marginal
    fixed = isdefined(M, :_BootFailures)      # guards the cases that never end on the old code
    quiet(f) = redirect_stdout(f, devnull)
    # run f with stdout captured and log records collected; an exception is returned, not thrown
    function logged(f)
        path, io = mktemp()
        val = nothing; logs = Any[]
        try
            logs, val = Test.collect_test_logs() do
                redirect_stdout(io) do
                    try
                        f()
                    catch e
                        e
                    end
                end
            end
        finally
            close(io)
        end
        txt = read(path, String); rm(path; force = true)
        return val, txt, logs
    end
    bootwarn(logs) = filter(l -> occursin("resamples drawn", string(l.message)), logs)
    verbose(r) = merge(getfield(r, :nt), (verbose = true,))
    function with_ls(r, ls)
        s = r.main_solver
        return merge(getfield(r, :nt), (main_solver = Newton(alphaguess = s.alphaguess!, linesearch = ls),))
    end
    errmsg(e) = e isa Exception ? sprint(showerror, e) : ""

    # a half-normal fit with hetero = [:sigma_u_sq] (its σᵤ² names are overridden while the
    # bootstrap runs); Newton, 100 iterations, g_abstol 1e-8
    rng = Xoshiro(2024); n = 200
    X = hcat(ones(n), randn(rng, n)); z = randn(rng, n)
    y = X * [1.0, 0.5] .+ 0.3 .* randn(rng, n) .- abs.(0.5 .* randn(rng, n) .* exp.(0.3 .* z))
    sp = sfmodel_spec(depvar = y, frontier = X, zvar = hcat(ones(n), z), noise = :Normal,
                      ineff = :HalfNormal, hetero = [:sigma_u_sq], type = :prod)
    ini = sfmodel_init(spec = sp, frontier = X \ y, ln_sigma_u_sq = [log(0.25), 0.0], ln_sigma_v_sq = log(0.09))
    rh = quiet(() -> sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE), init = ini,
                                 optim_options = sfmodel_opt(warmstart_solver = nothing, main_solver = Newton(),
                                                             main_opt = (iterations = 100,)),
                                 marginal = true, show_table = false, verbose = false))
    @test rh.converged && rh.redflag == 0
    names0 = copy(M._dicM[:σᵤ²])

    @testset "no failure: same draws as before; the report says none failed" begin
        (o, bs), _, logs = logged(() -> boot(result = rh, R = 10, seed = 5, getBootData = true, every = 1000))
        ro, rbs, kinds, nd = _bf30_ref_boot(rh; R = 10, seed = 5)
        @test isempty(kinds) && nd == 10
        @test isequal(o, ro) && isequal(bs, rbs)                 # bit for bit
        @test isempty(bootwarn(logs))
        ov, txt, _ = logged(() -> boot(result = verbose(rh), R = 10, seed = 5, every = 1000))
        @test !(ov isa Exception)                               # the table threw a MethodError before
        @test isequal(ov, o)
        @test occursin("Resamples drawn: 10; none failed.", txt)
        @test occursin("mean of the marginal", txt) && occursin("bias-corrected", txt)
        @test M._dicM[:σᵤ²] == names0
    end

    @testset "failures below the cap: same redraws as before, counted by kind" begin
        # How many resamples fail at a given `iter` depends on the dependency versions: on the
        # same resamples Newton takes other numbers of iterations with LineSearches 7.8.2 than
        # with LineSearches 7.6.0 (Newton's default line search, HagerZhang, comes from
        # LineSearches). So the expected counts are those of the copy of the old loop, which
        # records the kind of each failure (nd resamples drawn, nf = length(kinds) of them
        # failed), and the report line and warning the bootstrap must print are built from them,
        # with the share rounded as _boot_report rounds it.
        pct(nf, nd) = round(100 * nf / nd; digits = 1)
        reportline(nf, nd, it) = "Resamples drawn: $(nd); $(nf) failed and were drawn again ($(pct(nf, nd))%): " *
                                 "reached the iteration cap ($(it)): $(nf)."

        # iter = 5: most resamples reach the cap (with LineSearches 7.6.0, 15 of 25, 60.0%; with
        # 7.8.2, 26 of 36, 72.2%): bit for bit, reported, and warned when more than 10% failed
        (o, bs), _, logs = logged(() -> boot(result = rh, R = 10, seed = 5, iter = 5, getBootData = true, every = 1000))
        ro, rbs, kinds, nd = _bf30_ref_boot(rh; R = 10, seed = 5, iter = 5)
        nf = length(kinds)
        @test nd == 10 + nf && nf >= 1 && all(==(:itercap), kinds)
        @test isequal(o, ro) && isequal(bs, rbs)
        w = bootwarn(logs)
        if 10 * nf > nd                # more than 10% of the resamples drawn failed: one warning
            @test length(w) == 1 && w[1].level == Base.CoreLogging.Warn
            @test length(w) == 1 &&
                  occursin("sfmodel_boot_marginal: $(nf) of the $(nd) resamples drawn ($(pct(nf, nd))%) failed",
                           string(w[1].message)) &&
                  occursin("(reached the iteration cap (5): $(nf))", string(w[1].message))
        else                           # at most 10%: none
            @test isempty(w)
        end
        ov, txt, _ = logged(() -> boot(result = verbose(rh), R = 10, seed = 5, iter = 5, every = 1000))
        @test isequal(ov, o)
        @test occursin(reportline(nf, nd, 5), txt)

        # at least one failure but at most 10% of the resamples drawn (with R = 10: 1 of 11, 9.1%):
        # reported, no warning. Which `iter` gives this depends on the versions (seed 5: iter = 6
        # with LineSearches 7.6.0, iter = 8 with 7.8.2), and for many seeds no `iter` does (the
        # count can fall from 2 or more straight to 0), so the copy of the old loop searches for
        # it: seed 5, then 6 to 30, each with iter = 6 to 20; the first pair that qualifies is
        # used. A pair that qualifies draws exactly 11 resamples, so each call of the search stops
        # after 11 draws (maxdraw = 11).
        found = nothing
        for s in 5:30, it in 6:20
            _, _, ks, ndr = _bf30_ref_boot(rh; R = 10, seed = s, iter = it, maxdraw = 11)
            if !isempty(ks) && all(==(:itercap), ks) && 10 * length(ks) <= ndr
                found = (s, it)
                break
            end
        end
        if found === nothing
            @error("test: the copy of the old loop finds no seed in 5:30 and iter in 6:20 with at " *
                   "least one failed resample, all at the iteration cap, and at most 10% failed; widen the search")
            @test found !== nothing
        else
            s, it = found
            (o, bs), _, logs = logged(() -> boot(result = rh, R = 10, seed = s, iter = it, getBootData = true,
                                                 every = 1000))
            ro, rbs, kinds, nd = _bf30_ref_boot(rh; R = 10, seed = s, iter = it)
            nf = length(kinds)
            @test nd == 10 + nf && nf >= 1 && 10 * nf <= nd && all(==(:itercap), kinds)
            @test isequal(o, ro) && isequal(bs, rbs)
            @test isempty(bootwarn(logs))
            ov, txt, _ = logged(() -> boot(result = verbose(rh), R = 10, seed = s, iter = it, every = 1000))
            @test occursin(reportline(nf, nd, it), txt)
        end

        # two optimizer errors (injected in the line search): 2 of 12, warned; same draws as before.
        # Each resample calls the line search at least once, so call 12 comes whatever the
        # number of Newton iterations the dependency versions give.
        mk() = _BF30InjectLS(rh.main_solver.linesearch!; at = [3, 12])
        (o, bs), _, logs = logged(() -> boot(result = with_ls(rh, mk()), R = 10, seed = 5, getBootData = true, every = 1000))
        ro, rbs, kinds, nd = _bf30_ref_boot(with_ls(rh, mk()); R = 10, seed = 5)
        @test nd == 12 && kinds == [:optimizer, :optimizer]
        @test isequal(o, ro) && isequal(bs, rbs)
        w = bootwarn(logs)
        @test length(w) == 1 && occursin("the optimizer threw an error: 2", string(w[1].message)) &&
              occursin("injected by the test", string(w[1].message))
        @test M._dicM[:σᵤ²] == names0
    end

    @testset "10R failed resamples stop the bootstrap (optimizer errors)" begin
        # every line search throws until call 1000: after 10R = 20 failures the bootstrap stops
        # (before the fix it drew 1000 resamples and then carried on)
        ls = _BF30InjectLS(rh.main_solver.linesearch!; at = 1:1000)
        e, _, _ = logged(() -> boot(result = with_ls(rh, ls), R = 2, seed = 1, every = 1000))
        @test e isa ErrorException
        @test ls.calls == 20
        msg = errmsg(e)
        @test occursin("20 resamples failed (the limit is 10R = 20)", msg) &&
              occursin("the optimizer threw an error: 20", msg) && occursin("0 of the R = 2 replicates", msg)
        @test M._dicM[:σᵤ²] == names0                            # restored in the `finally`
    end

    @testset "iter below what every resample needs: stops after 10R failures" begin
        if fixed
            # iter = 1: no resample converges in one Newton step from the fit's estimate
            e, _, _ = logged(() -> boot(result = rh, R = 2, seed = 1, iter = 1, every = 1000))
            @test e isa ErrorException
            msg = errmsg(e)
            @test occursin("20 resamples failed", msg) && occursin("reached the iteration cap (1): 20", msg) &&
                  occursin("raise `iter`", msg)
            @test M._dicM[:σᵤ²] == names0
        else
            @test fixed        # before the fix this bootstrap never ends, so it is not run
        end
    end

    @testset "Ctrl-C (InterruptException) in a replicate stops the bootstrap" begin
        ls = _BF30InjectLS(rh.main_solver.linesearch!; at = [3], exc = InterruptException())
        e, _, _ = logged(() -> boot(result = with_ls(rh, ls), R = 10, seed = 5, every = 1000))
        @test e isa InterruptException                          # it was swallowed and redrawn
        @test ls.calls == 3
        @test M._dicM[:σᵤ²] == names0
    end

    @testset "Ctrl-C inside a spawned task; an error whose message cannot be printed" begin
        # an InterruptException that lands in a task spawned by the likelihood's threaded loop
        # reaches the catch wrapped in a TaskFailedException; it stops the bootstrap too
        spawned_interrupt() = fetch(Threads.@spawn throw(InterruptException()))
        ls = _BF30InjectLS(rh.main_solver.linesearch!; at = [3], exc = spawned_interrupt)
        e, _, _ = logged(() -> boot(result = with_ls(rh, ls), R = 10, seed = 5, every = 1000))
        @test e isa TaskFailedException                         # it was swallowed and redrawn
        @test ls.calls == 3
        @test M._dicM[:σᵤ²] == names0
        # an optimizer error whose `showerror` throws is still a failed resample: drawn again,
        # the same draws as before; the warning names its type
        mk() = _BF30InjectLS(rh.main_solver.linesearch!; at = [4], exc = _BF30BadShow())
        v, _, logs = logged(() -> boot(result = with_ls(rh, mk()), R = 6, seed = 3, getBootData = true, every = 1000))
        ro, rbs, kinds, nd = _bf30_ref_boot(with_ls(rh, mk()); R = 6, seed = 3)
        @test kinds == [:optimizer] && nd == 7
        @test v isa Tuple && isequal(v, (ro, rbs))
        w = bootwarn(logs)
        @test length(w) == 1 && occursin("the last optimizer error: ", string(w[1].message)) &&
              occursin("_BF30BadShow", string(w[1].message))
        @test M._dicM[:σᵤ²] == names0
    end

    @testset "ParticleSwarm or SimulatedAnnealing as main solver: stops before any draw" begin
        if fixed
            for s in (ParticleSwarm(), SimulatedAnnealing())
                Random.seed!(30); x0 = rand()
                Random.seed!(30)
                e, _, _ = logged(() -> boot(result = merge(getfield(rh, :nt), (main_solver = s,)), R = 5, every = 1000))
                @test e isa ErrorException
                msg = errmsg(e)
                @test occursin("its main solver is $(nameof(typeof(s)))", msg) &&
                      occursin("Refit with a gradient-based main solver", msg)
                @test rand() == x0                      # no resample was drawn from the global RNG
                @test M._dicM[:σᵤ²] == names0
            end
        else
            @test fixed        # before the fix these bootstraps never end, so they are not run
        end
        # NelderMead has its own convergence measure and is not stopped
        v, _, _ = logged(() -> boot(result = merge(getfield(rh, :nt), (main_solver = NelderMead(),)), R = 2,
                                    seed = 5, iter = 1000, every = 1000))
        @test v isa Matrix
    end

    @testset "the failure bookkeeping (no fit)" begin
        if fixed
            labels = (optimizer = "the optimizer threw an error: 1",
                      itercap   = "reached the iteration cap (60): 1",
                      gradient  = "stopped with the gradient norm above the tolerance (1.0e-5), or NaN: 1",
                      hesserror = "computing the Hessian threw an error: 1",
                      inverse   = "Hessian not invertible: 1",
                      variance  = "inverse Hessian with a non-positive variance: 1",
                      margerror = "computing the marginal effects threw an error: 1",
                      margnan   = "NaN in the mean marginal effects: 1",
                      margdim   = "an inefficiency determinant constant in the resample: 1")
            @test collect(keys(M._BOOT_FAIL_KINDS)) == collect(keys(labels))
            for k in keys(labels)
                f = M._BootFailures(60, 1e-5)
                M._boot_fail!(f, k, 10, 0)
                @test M._boot_fail_summary(f) == labels[k]
            end
            # the warning starts strictly above 10% of the resamples drawn
            f = M._BootFailures(60, 1e-5)
            M._boot_fail!(f, :itercap, 9, 0)
            l1, _ = Test.collect_test_logs(() -> M._boot_report(f, 9, false))     # 1 of 10 drawn
            M._boot_fail!(f, :itercap, 9, 0)
            l2, _ = Test.collect_test_logs(() -> M._boot_report(f, 9, false))     # 2 of 11 drawn
            @test isempty(l1) && length(l2) == 1
            # the limit: the 10R-th failure stops the bootstrap, the one before does not
            f = M._BootFailures(60, 1e-5)
            for _ in 1:19
                M._boot_fail!(f, :inverse, 2, 1)
            end
            @test sum(f.n) == 19
            @test_throws ErrorException M._boot_fail!(f, :inverse, 2, 1)
            # the optimizer error's message: one line; its type when it cannot be printed
            @test M._boot_errtext(DomainError(-1.0, "a b\n  c")) == "DomainError with -1.0: a b c"
            @test occursin("_BF30BadShow", M._boot_errtext(_BF30BadShow()))
        else
            @test fixed
        end
        # every catch block of the loop (the optimizer's and the one around the Hessian's
        # inverse; also those around the Hessian and get_marg) rethrows what
        # _boot_fatal flags; an interrupt cannot be injected into inv() through the public
        # API, so this is checked in the source
        src = read(joinpath(pkgdir(SFrontiers), "src", "sf_MLE", "SFmainfun.jl"), String)
        @test count("_boot_fatal(err) && rethrow()", src) == 4
    end

    # The fits below replace the state of the last fit, so they come after every use of `rh`.

    @testset "a resample in which every determinant is constant is drawn again" begin
        # zvar = d alone, with d = 1 except one of 200 observations: about 37% of the resamples
        # miss that observation, so d is constant in them, get_marg drops its column and the mean
        # marginal effects are empty (before the fix: MethodError(zero, Any), and the names stayed
        # overridden). Without an intercept column such a resample's model is still identified
        # (σᵤ² = exp(δ)), so it converges and passes the Hessian checks whatever the rounding, and
        # every seed counts resamples of this kind. With zvar = [1 d] a constant d would be
        # collinear with the intercept, and whether such a resample got past the Hessian checks
        # would depend on the machine.
        rng = Xoshiro(6); n = 200
        X = hcat(ones(n), randn(rng, n))
        d = ones(n); d[1] = 2.0
        u = abs.(randn(rng, n)) .* sqrt.(exp.(log(0.64) .* d))
        y = X * [0.8, 0.4] .+ 0.2 .* randn(rng, n) .- u
        sp = sfmodel_spec(depvar = y, frontier = X, zvar = reshape(d, n, 1), noise = :Normal,
                          ineff = :HalfNormal, hetero = [:sigma_u_sq], type = :prod)
        ini = sfmodel_init(spec = sp, frontier = X \ y, ln_sigma_u_sq = [log(0.64)], ln_sigma_v_sq = log(0.04))
        rc = quiet(() -> sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE), init = ini,
                                     optim_options = sfmodel_opt(warmstart_solver = nothing, main_solver = Newton(),
                                                                 main_opt = (iterations = 100,)),
                                     marginal = true, show_table = false, verbose = false))
        @test rc.converged
        nm = copy(M._dicM[:σᵤ²])
        nconst(txt) = (m = match(r"an inefficiency determinant constant in the resample: (\d+)", txt);
                       m === nothing ? 0 : parse(Int, m[1]))
        for s in (1, 2)
            v, _, _ = logged(() -> boot(result = rc, R = 50, seed = s, every = 1000))
            @test v isa Matrix                                   # no MethodError
            _, txt, _ = logged(() -> boot(result = verbose(rc), R = 50, seed = s, every = 1000))
            @test nconst(txt) >= 1                               # counted by its kind, for each seed
            @test M._dicM[:σᵤ²] == nm
        end
    end

    @testset "every resample's Hessian singular: counted as such, stops after 10R" begin
        if fixed
            # a determinant column of zeros makes every resample's Hessian exactly singular
            rng = Xoshiro(31); n = 150
            X = hcat(ones(n), randn(rng, n)); z = randn(rng, n)
            y = X * [0.8, 0.4] .+ 0.25 .* randn(rng, n) .- abs.(0.5 .* randn(rng, n) .* exp.(0.3 .* z))
            sp = sfmodel_spec(depvar = y, frontier = X, zvar = hcat(ones(n), z, zeros(n)), noise = :Normal,
                              ineff = :HalfNormal, hetero = [:sigma_u_sq], type = :prod)
            ini = sfmodel_init(spec = sp, frontier = X \ y, ln_sigma_u_sq = [log(0.25), 0.0, 0.0],
                               ln_sigma_v_sq = log(0.06))
            rz, _, _ = logged(() -> sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE), init = ini,
                                                optim_options = sfmodel_opt(warmstart_solver = nothing,
                                                                            main_solver = Newton(),
                                                                            main_opt = (iterations = 100,)),
                                                marginal = true, show_table = false, verbose = false))
            e, _, _ = logged(() -> boot(result = rz, R = 2, seed = 1, every = 1000))
            @test e isa ErrorException
            @test occursin("Failed resamples by kind: Hessian not invertible: 20.", errmsg(e))
        else
            @test fixed        # before the fix this bootstrap never ends, so it is not run
        end
    end
end
