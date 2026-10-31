# Regression tests: the MLE bootstrap refuses a result that is not the last :MLE fit.
#
# MLE_Backend.sfmodel_boot_marginal takes the data and the model's equations from the module-level
# state that the last :MLE fit leaves behind (`_dicM`), and everything else (starting values,
# solver settings, labels, the point estimates) from `result`. Its only guard checked that the
# state was still the last fit's, not that `result` was the last fit: after fitting a model A and
# then a model B of the same type with the same number of marginal-effect columns, the call
# boot(result = A) bootstrapped B's data with A's settings and returned wrong standard errors
# without a word, also with `data` given (getvar reads the equations from `_dicM` there too). It
# now refuses a result that is not the result of the last completed :MLE fit (after one
# deprecated Vararg spec, whose fits share one state, an earlier result whose start coefficients
# equal the last fit's in every bit still passes, and it is bootstrapped on its own data and
# equations, so its numbers are right). Second, under :MLE a fit with
# marginal = false stores the empty tuple () in marginal_mean, and every panel_TFE_CSW and
# panel_TRE fit does too; () passed the "No statistics to bootstrap" test and the bootstrap ended
# in a MethodError at ncol(result.marginal). It now stops with that message.
#
# Bootstraps called in a correct order are unchanged: the testsets compare them, bit for bit, with
# the bootstrap of the same fit made right after it, before any other fit.

using Test
using SFrontiers
using Random
using Optim

@testset "the MLE bootstrap refuses a result that is not the last :MLE fit" begin
    MB = SFrontiers.MLE_Backend
    M = MB.SFmle
    boot = MB.sfmodel_boot_marginal
    quiet(f) = redirect_stdout(f, devnull)
    # run f with stdout discarded and log records collected; return its value, or what it threw
    # (the function throws plain strings for some errors)
    caught(f) = try
        last(Test.collect_test_logs(() -> quiet(f)))
    catch e
        e
    end
    msg(e) = e isa Exception ? sprint(showerror, e) : string(e)
    refused(e) = e isa ErrorException &&
                 occursin("`result` is not the result of the last `sfmodel_fit` with `method = :MLE`", msg(e)) &&
                 occursin("immediately after the fit it refers to", msg(e))
    nostat = "No statistics to bootstrap."

    # the half normal with hetero = [:sigma_u_sq] of test_boot_redraw_rules.jl; B is the same
    # specification on the first 150 of A's 200 observations: same model type, same number of
    # marginal-effect columns, the case the bootstrap used to run on without a word
    rng = Xoshiro(2024); n = 200
    X = hcat(ones(n), randn(rng, n)); z = randn(rng, n)
    y = X * [1.0, 0.5] .+ 0.3 .* randn(rng, n) .- abs.(0.5 .* randn(rng, n) .* exp.(0.3 .* z))
    spec(rows) = sfmodel_spec(depvar = y[rows], frontier = X[rows, :], zvar = hcat(ones(length(rows)), z[rows]),
                              noise = :Normal, ineff = :HalfNormal, hetero = [:sigma_u_sq], type = :prod)
    opt = sfmodel_opt(warmstart_solver = nothing, main_solver = Newton(), main_opt = (iterations = 100,))
    function fit(sp, rows; marginal = true)
        ini = sfmodel_init(spec = sp, frontier = X[rows, :] \ y[rows], ln_sigma_u_sq = [log(0.25), 0.0],
                           ln_sigma_v_sq = log(0.09))
        return quiet(() -> sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE), init = ini,
                                       optim_options = opt, marginal = marginal, show_table = false,
                                       verbose = false))
    end
    bootA(r; kw...) = boot(; result = r, R = 10, seed = 5, getBootData = true, every = 1000, kw...)

    spA = spec(1:n)
    rA = fit(spA, 1:n)
    @test rA.converged && rA.redflag == 0
    oA = caught(() -> bootA(rA))                 # right after the fit: the reference
    @test oA isa Tuple && all(isfinite, oA[1][:, 1])

    @testset "after another :MLE fit, the earlier result is refused" begin
        rB = fit(spec(1:150), 1:150)
        @test rB.converged && rB.redflag == 0
        @test rB.modelid === rA.modelid && size(rB.marginal, 2) == size(rA.marginal, 2)
        @test M._mle_state_matches_last_fit()    # the older guard is satisfied, so it let this through
        @test refused(caught(() -> bootA(rA)))   # before the fix: B's data with A's settings
        # with `data` given, getvar still reads the model's equations from the module state
        @test refused(caught(() -> bootA(rA; data = M._dicM[:sdf])))
        oB = caught(() -> bootA(rB))             # the last fit runs
        @test oB isa Tuple && all(isfinite, oB[1][:, 1]) && !isequal(oB[1][:, 1], oA[1][:, 1])
    end

    @testset "a rerun of the fit: its result runs, with the numbers of the first run" begin
        rA2 = fit(spA, 1:n)
        @test isequal(rA2.list[:coeff], rA.list[:coeff])
        @test isequal(caught(() -> bootA(rA2)), oA)                       # bit for bit
        @test refused(caught(() -> bootA(rA)))   # the result of the earlier run is refused
        @test isequal(caught(() -> bootA(rA2; data = M._dicM[:sdf])), oA) # data given: same numbers

        # each link of the check on its own, on copies of the last fit's result
        nt = getfield(rA2, :nt)
        # the same coefficients, but equation names that are not the module state's own vector
        @test refused(caught(() -> bootA(merge(nt, (frontier = copy(nt.frontier),)))))
        # the module state's names, but a coefficient one ulp away from the last fit's
        l2 = copy(nt.list); c2 = copy(nt.list[:coeff]); c2[end] = nextfloat(c2[end]); l2[:coeff] = c2
        @test refused(caught(() -> bootA(merge(nt, (list = l2, coeff = c2)))))
        # without `list` the start is `coeff`; in the analytic order it passes
        r24 = merge(Base.structdiff(nt, NamedTuple{(:list,)}), (coeff = collect(nt.list[:coeff]),))
        @test isequal(caught(() -> bootA(r24)), oA)
        # a deep copy holds equal names, not the module state's own vector: refused
        @test refused(caught(() -> bootA(deepcopy(rA2))))

        # a spec built after the fit: still the older guard and its message
        spec(1:100)
        e = caught(() -> bootA(rA2))
        @test e isa ErrorException && occursin("replaced since the last `sfmodel_fit`", msg(e))
    end

    @testset "two fits after one deprecated Vararg spec share the state: only the last runs" begin
        nV = 150
        M.sfmodel_spec(M.sftype(M.production), M.sfdist(M.half), M.depvar(y[1:nV]), M.frontier(X[1:nV, :]),
                       M.σᵤ²(hcat(ones(nV), z[1:nV])), M.σᵥ²(ones(nV, 1)))
        vopt() = MB.sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 100, g_abstol = 1e-8))
        vfit(ini) = quiet(() -> MB.sfmodel_fit(init = ini, optim_options = vopt(), marginal = true,
                                               show_table = false, verbose = false))
        b0 = X[1:nV, :] \ y[1:nV]
        r1 = vfit(vcat(b0, [log(0.25), 0.0], [log(0.09)]))
        r2 = vfit(vcat(b0, [log(0.20), 0.1], [log(0.10)]))
        @test r1.converged && r2.converged
        @test r1.frontier === r2.frontier && !isequal(r1.list[:coeff], r2.list[:coeff])
        @test r1.transfer == false && caught(() -> boot(result = r2, R = 5, seed = 1)) ==
                                      "Need the `data=<DataFrame name>` option."
        @test refused(caught(() -> boot(result = r1, data = M._dicM[:sdf], R = 5, seed = 1, every = 1000)))
        o2 = caught(() -> boot(result = r2, data = M._dicM[:sdf], R = 5, seed = 1, every = 1000))
        @test o2 isa Matrix && all(isfinite, o2[:, 1])
    end

    @testset "nothing to bootstrap: the intended message, not a MethodError" begin
        rN = fit(spA, 1:n; marginal = false)
        @test rN.marginal === nothing && rN.marginal_mean === ()   # what :MLE stores
        e = caught(() -> boot(result = rN, R = 5, seed = 1))
        @test !(e isa MethodError)               # before the fix: ncol(::Nothing)
        @test e == nostat

        # no inefficiency determinant, marginal = true: an empty NamedTuple, as before
        sp0 = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal, type = :prod)
        r0 = quiet(() -> sfmodel_fit(spec = sp0, method = sfmodel_method(method = :MLE), optim_options = opt,
                                     marginal = true, show_table = false, verbose = false))
        @test r0.marginal_mean === NamedTuple()
        @test caught(() -> boot(result = r0, R = 5, seed = 1)) == nostat

        # panel_TRE has no inefficiency determinants: () even with marginal = true
        rngT = Xoshiro(11); nf, T = 30, 3; N = nf * T
        idT = repeat(1:nf, inner = T)
        XT = hcat(ones(N), randn(rngT, N))
        yT = XT * [1.0, 0.5] .+ repeat(0.3 .* randn(rngT, nf), inner = T) .+ 0.2 .* randn(rngT, N) .-
             abs.(0.4 .* randn(rngT, N))
        spT = sfmodel_spec(datatype = :panel_TRE, depvar = yT, frontier = XT, id = idT, noise = :Normal,
                           ineff = :HalfNormal)
        rT = quiet(() -> sfmodel_fit(spec = spT, method = sfmodel_method(method = :MLE),
                                     optim_options = sfmodel_opt(warmstart_solver = nothing, main_solver = Newton(),
                                                                 main_opt = (iterations = 50,)),
                                     marginal = true, show_table = false, verbose = false))
        @test rT.marginal_mean === ()
        e = caught(() -> boot(result = rT, R = 3, seed = 1))
        @test !(e isa MethodError)               # before the fix: ncol(::Tuple{})
        @test e == nostat
    end
end
