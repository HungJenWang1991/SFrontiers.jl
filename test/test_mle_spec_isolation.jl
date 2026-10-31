# test_mle_spec_isolation.jl — the `method=:MLE` global-state defect.
#
# `MLE_Backend.sfmodel_fit` took the estimation frame, the model id, the frontier sign, the
# panel flag and the whole equation layout from the module-level `_dicM`/`tagD`, which
# `sfmodel_spec` writes when a spec is CONSTRUCTED. The globals therefore described the last
# spec constructed rather than the spec handed to `sfmodel_fit`, so constructing a second spec
# before fitting the first silently estimated the second model — converged, no warning, and
# labelled with the first model's variable names. Only `:MLE` was affected; the MCI, MSLE and
# panel backends already derive everything from their spec argument. `sfmodel_fit` now rebuilds
# `_dicM`/`tagD` from the spec it is handed, so the MLE backend behaves like the other three.
#
# Every case below uses TWO DIFFERENT datasets or two genuinely different models. That is
# deliberate: `test/dsl_equiv_lib.jl` runs this exact construct-construct-fit sequence and
# cannot detect the defect, because both of its specs wrap the same DataFrame, which makes the
# poisoned and the clean fit bit-identical. Any test that reuses one dataset across the two
# specs is worthless here, so each testset also asserts that the two answers actually differ.
using Test
using SFrontiers
using Random
using Optim: NelderMead, Newton

@testset "MLE fits the spec it is handed" begin

    _opt = sfmodel_opt(warmstart_solver=NelderMead(), warmstart_opt=(iterations=30, g_abstol=1e-3),
                       main_solver=Newton(), main_opt=(iterations=200, g_abstol=1e-6))
    _meth(m) = m === :MLE ? sfmodel_method(method = m) : sfmodel_method(method = m, n_draws = 1024)

    Random.seed!(90210)
    n = 120
    XA = hcat(ones(n), randn(n), randn(n))
    yA = XA * [1.0, 0.5, -0.3] .- abs.(0.4 .* randn(n)) .+ 0.3 .* randn(n)
    XB = hcat(ones(n), randn(n), randn(n))
    yB = XB * [-2.0, 3.0, 4.0] .- abs.(0.6 .* randn(n)) .+ 0.4 .* randn(n)
    ZA = hcat(ones(n), randn(n))
    ZB = hcat(ones(n), randn(n))

    _fit(sp, m=:MLE) = sfmodel_fit(spec=sp, method = _meth(m), init=nothing,
                                   optim_options=_opt, marginal=false, show_table=false,
                                   verbose=false)
    _co(r) = collect(float.(r.coeff))

    # `build` and `fit` are thunks so each case can be run in isolation (nothing constructed
    # between building the spec and fitting it) and then again with `other` constructed in
    # between. `isolated` is the answer the user asked for.
    function _case(buildA, buildB, m=:MLE)
        a  = _fit(buildA(), m)                      # isolated: nothing built in between
        b  = _fit(buildB(), m)                      # the other model, on its own
        sa = buildA(); buildB()                     # build A, then build B ...
        ab = _fit(sa, m)                            # ... then fit A
        return _co(a), _co(b), _co(ab)
    end

    # ------------------------------------------------------------------ same shape, other data
    @testset "same model, different data" begin
        mkA = () -> sfmodel_spec(depvar=yA, frontier=XA, noise=:Normal, ineff=:HalfNormal)
        mkB = () -> sfmodel_spec(depvar=yB, frontier=XB, noise=:Normal, ineff=:HalfNormal)
        a, b, ab = _case(mkA, mkB)
        @test length(a) == length(b) == 5           # identical parameter counts: no guard can see this
        @test !isapprox(a, b; rtol=1e-3)            # the test is not vacuous
        @test ab == a                               # bit-for-bit, not merely close
        @test !isapprox(ab, b; rtol=1e-3)
    end

    # ------------------------------------------------- different parameter counts, same data
    # The length check in sfmodel_fit cannot catch this: with the default init the starting
    # vector is assembled against whichever model the globals describe, so its length always
    # matches. Before the repair this returned six coefficients for a five-parameter spec.
    @testset "different parameter counts" begin
        mkA = () -> sfmodel_spec(depvar=yA, frontier=XA, noise=:Normal, ineff=:HalfNormal)
        mkB = () -> sfmodel_spec(depvar=yB, frontier=XB, noise=:Normal, ineff=:TruncatedNormal)
        a, b, ab = _case(mkA, mkB)
        @test length(a) == 5
        @test length(b) == 6
        @test length(ab) == 5                       # A's shape, not B's
        @test ab == a
    end

    # ------------------------------------------------------------------ production vs cost
    # Same data, same distribution, same parameter count: only the frontier sign differs, which
    # the backend read from the globals. The two log-likelihoods are close, so nothing about the
    # output would tell a user which model was estimated.
    @testset "production displaced by cost" begin
        mkA = () -> sfmodel_spec(depvar=yA, frontier=XA, noise=:Normal, ineff=:HalfNormal, type=:prod)
        mkB = () -> sfmodel_spec(depvar=yA, frontier=XA, noise=:Normal, ineff=:HalfNormal, type=:cost)
        a, b, ab = _case(mkA, mkB)
        @test !isapprox(a, b; rtol=1e-3)
        @test ab == a
    end

    # ----------------------------------------------------- heteroscedastic z-column swap
    # Same y, same X, same hetero option, same parameter count; only which column drives σᵤ²
    # differs. Before the repair the returned fit used the other spec's z column, with an output
    # identical in shape, labels, observation count and OLS diagnostics.
    @testset "heteroscedastic z columns" begin
        mkA = () -> sfmodel_spec(depvar=yA, frontier=XA, zvar=ZA, noise=:Normal,
                                 ineff=:HalfNormal, hetero=[:sigma_u_sq])
        mkB = () -> sfmodel_spec(depvar=yA, frontier=XA, zvar=ZB, noise=:Normal,
                                 ineff=:HalfNormal, hetero=[:sigma_u_sq])
        a, b, ab = _case(mkA, mkB)
        @test length(a) == length(b)
        @test !isapprox(a, b; rtol=1e-3)
        @test ab == a
    end

    # ------------------------------------------------- equation assignment at equal count
    # hetero=[:mu] and hetero=[:sigma_u_sq] on the same TruncatedNormal and the same zvar have the
    # same parameter count but assign those parameters to different equations.
    @testset "equation assignment at equal parameter count" begin
        mkA = () -> sfmodel_spec(depvar=yA, frontier=XA, zvar=ZA, noise=:Normal,
                                 ineff=:TruncatedNormal, hetero=[:mu])
        mkB = () -> sfmodel_spec(depvar=yA, frontier=XA, zvar=ZA, noise=:Normal,
                                 ineff=:TruncatedNormal, hetero=[:sigma_u_sq])
        a, b, ab = _case(mkA, mkB)
        @test length(a) == length(b)
        @test !isapprox(a, b; rtol=1e-3)
        @test ab == a
    end

    # ------------------------------------------------------------------ panel models
    # A second panel_TRE spec used to make the first one throw `DomainError with 0.0` from inside
    # the differentiated likelihood; the stale key was the estimation frame itself.
    @testset "panel TRE and a cross-sectional displacer" begin
        # the DGP and the starting values are set here because the TRE likelihood does not
        # converge from the OLS defaults on this design
        Random.seed!(303)
        nf, T = 20, 5
        np  = nf * T
        idp = repeat(1:nf, inner=T)
        XP  = hcat(randn(np), randn(np))
        zp  = randn(np)
        yP1 = repeat(randn(nf), inner=T) .+ XP * [0.6, -0.4] .+ 0.3 .* randn(np) .-
              exp.(0.3 .* zp) .* repeat(abs.(0.4 .* randn(nf)), inner=T)
        Random.seed!(404)
        yP2 = repeat(randn(nf), inner=T) .+ XP * [-3.0, 2.0] .+ 0.4 .* randn(np) .-
              exp.(0.5 .* zp) .* repeat(abs.(1.2 .* randn(nf)), inner=T)

        mkP(y) = sfmodel_spec(datatype=:panel_TRE, depvar=y, frontier=XP, id=idp,
                              noise=:Normal, ineff=:HalfNormal)
        _iniP(sp, y, s2, sv) = sfmodel_init(spec=sp, frontier=XP \ y,
                                            ln_sigma_u_sq=log(s2), ln_sigma_v_sq=log(sv),
                                            sigma_a_sq=[0.1], message=false)
        _fitP(sp, ini) = _co(sfmodel_fit(spec=sp, method=sfmodel_method(method=:MLE), init=ini,
                                         optim_options=_opt, marginal=false, show_table=false,
                                         verbose=false))

        s1 = mkP(yP1); a = _fitP(s1, _iniP(s1, yP1, 0.16, 0.09))
        s2 = mkP(yP2); b = _fitP(s2, _iniP(s2, yP2, 1.44, 0.16))
        @test !isapprox(a, b; rtol=1e-3)

        # before the repair this threw `DomainError with 0.0` from inside the differentiated
        # likelihood, because the estimation frame belonged to the second spec
        sa = mkP(yP1); ini = _iniP(sa, yP1, 0.16, 0.09); mkP(yP2)
        @test _fitP(sa, ini) == a

        # a cross-sectional spec displacing a panel one changes the estimator, not just the data
        sb = mkP(yP1); ini2 = _iniP(sb, yP1, 0.16, 0.09)
        sfmodel_spec(depvar=yA, frontier=XA, noise=:Normal, ineff=:HalfNormal)
        @test _fitP(sb, ini2) == a
    end

    # ------------------------------------------------ agreement with the other backends
    # The reference behaviour: MCI and MSLE already return the spec they were handed. They run
    # the sequence of "same model, different data" (which runs it under :MLE) on the same two
    # models.
    @testset "agrees with the MCI and MSLE backends" begin
        mkA = () -> sfmodel_spec(depvar=yA, frontier=XA, noise=:Normal, ineff=:HalfNormal)
        mkB = () -> sfmodel_spec(depvar=yB, frontier=XB, noise=:Normal, ineff=:HalfNormal)
        for m in (:MCI, :MSLE)
            a, b, ab = _case(mkA, mkB, m)
            @test !isapprox(a, b; rtol=1e-3)
            @test ab == a                            # every backend returns the spec it was given
        end
    end

    # ------------------------------------------------------------------ the reusable spec
    # One spec, several methods, several sets of starting values, fits in any order and repeated:
    # each must equal the answer that combination gives on its own, whether or not an unrelated
    # spec has been constructed in the meantime.
    @testset "one spec reused across methods, inits and orderings" begin
        b0 = XA \ yA
        mkA = () -> sfmodel_spec(depvar=yA, frontier=XA, noise=:Normal, ineff=:HalfNormal)
        mkOther = () -> sfmodel_spec(depvar=yB, frontier=XB, noise=:Normal, ineff=:TruncatedNormal)
        # One starting-value object for every method: `sfmodel_init` takes no `method`
        # argument and returns the same vector whatever will fit it.
        mkinit(sp, m, sc) =
            sfmodel_init(spec=sp, frontier=b0 .* sc,
                         ln_sigma_u_sq=log(0.16sc), ln_sigma_v_sq=log(0.09sc), message=false)
        run1(sp, m, ini) = _co(sfmodel_fit(spec=sp, method = _meth(m), init=ini,
                                           optim_options=_opt, marginal=false, show_table=false,
                                           verbose=false))

        combos = [(m, sc) for m in (:MLE, :MCI, :MSLE), sc in (1.0, 1.6)]
        ref = Dict(c => (sp = mkA(); run1(sp, c[1], mkinit(sp, c[1], c[2]))) for c in combos)

        myspec = mkA()
        inits  = Dict(c => mkinit(myspec, c[1], c[2]) for c in combos)
        mkOther()                                    # an unrelated spec, constructed in between
        for c in combos
            @test run1(myspec, c[1], inits[c]) == ref[c]
        end
        # scrambled order and repeats on that same spec
        for c in reverse(combos)
            @test run1(myspec, c[1], inits[c]) == ref[c]
        end
        @test run1(myspec, :MLE, inits[(:MLE, 1.0)]) == ref[(:MLE, 1.0)]
        # the two sets of starting values must actually be distinguishable somewhere, or the
        # ordering assertions above prove nothing
        @test any(ref[(m, 1.0)] != ref[(m, 1.6)] for m in (:MLE, :MCI, :MSLE))
    end

    # ------------------------------------------------------------------ unit-level pins
    @testset "module state tracks the spec that was fitted" begin
        _S = SFrontiers.MLE_Backend.SFmle
        mkA = () -> sfmodel_spec(depvar=yA, frontier=XA, noise=:Normal, ineff=:HalfNormal)
        mkB = () -> sfmodel_spec(depvar=yB, frontier=XB, noise=:Normal, ineff=:TruncatedNormal)

        # the marker that tells sfmodel_fit the dictionary came from a spec object
        mkA()
        @test _S._dicM[:_spec_built] === true

        # after constructing B the state describes B; after fitting A it describes A, so the
        # post-fit consumers that read it see the model that was actually estimated
        sp = mkA(); mkB()
        @test _S.tagD[:modelid] === _S.Trun
        _fit(sp)
        @test _S.tagD[:modelid] === _S.Half
        @test size(_S._dicM[:sdf], 1) == n
        @test _S._dicM[:type] == [:production]
        @test Vector(_S._dicM[:sdf][:, :depvar]) == yA

        # repeated fits of one spec do not drift
        r1 = _co(_fit(sp)); r2 = _co(_fit(sp)); r3 = _co(_fit(sp))
        @test r1 == r2 == r3
    end

end

# ---------------------------------------------------------------------------------------
# The two POST-estimation isolation holes.
#
# `sfmodel_fit` rebuilds `_dicM` from the spec it is handed. Two
# legacy entry points do not: `sfmodel_predict` and `sfmodel_boot_marginal` take their DATA
# from the module-level `_dicM` at CALL time while taking their COEFFICIENTS from the last fit,
# so a spec built between the fit and the call paired one model's data with another model's
# estimates. Measured before the guard: `sfmodel_predict([:frontier])` returned `X_new * beta_old`
# exactly. Every spec route assigns a fresh `_dicM` Dict, so the fit now records which object its
# coefficients belong to and both entry points refuse when the current state is not it.
#
# Neither function is exported — not from `SFrontiers` and not from `MLE_Backend`. The manual documents
# `sfmodel_boot_marginal`, called as `SFrontiers.MLE_Backend.sfmodel_boot_marginal(result =
# result, ...)` (section "Bootstrap standard errors"), so this testset guards a documented
# call. `predict`, which is exported, reads the spec attached to its result object and was
# never affected.
# ---------------------------------------------------------------------------------------
@testset "post-estimation reads refuse a displaced model state" begin
    _sfp = SFrontiers.MLE_Backend.sfmodel_predict
    _sfb = SFrontiers.MLE_Backend.sfmodel_boot_marginal

    Random.seed!(11)
    m  = 60
    XP = hcat(ones(m), randn(m)); zP = randn(m)
    yP = XP*[1.0, 0.5] .+ 0.2 .* randn(m) .- abs.(0.3 .* randn(m))
    XQ = hcat(ones(m), 10 .+ randn(m)); zQ = 5 .+ randn(m)      # different data, same shapes
    yQ = XQ*[3.0, -2.0] .+ 0.2 .* randn(m) .- abs.(0.3 .* randn(m))

    _mk(y, X, z) = sfmodel_spec(depvar = y, frontier = X, type = :production,
                                noise = :Normal, ineff = :HalfNormal,
                                zvar = hcat(ones(m), z), hetero = [:sigma_u_sq])
    spP = _mk(yP, XP, zP)
    iniP = sfmodel_init(spec = spP, frontier = [0.9, 0.4],
                        ln_sigma_u_sq = [-1.0, 0.0], ln_sigma_v_sq = -3.0)
    optP = sfmodel_opt(warmstart_solver = nothing, main_solver = NelderMead(),
                       main_opt = (iterations = 150,))
    rP = last(Test.collect_test_logs(() -> sfmodel_fit(spec = spP,
              method = sfmodel_method(method = :MLE), init = iniP, optim_options = optP,
              jlms_bc_index = false, marginal = true, show_table = false, verbose = false)))

    # right after the fit it answers, and answers from the fit's own data
    p_ok = _sfp([:frontier])
    @test isapprox(Float64.(vec(p_ok)), XP*Float64.(rP.coeff[1:2]); rtol = 1e-10)
    b_ok = last(Test.collect_test_logs(() -> _sfb(result = rP, R = 3, seed = 1, every = 1000)))
    @test b_ok !== nothing

    # a spec built in between displaces the state; both entry points now refuse by name
    _mk(yQ, XQ, zQ)
    err = try (_sfp([:frontier]); nothing) catch e; e end
    @test err !== nothing
    msg = sprint(showerror, err)
    @test occursin("sfmodel_predict", msg)
    @test occursin("replaced since the last", msg)
    # the wrong pairing is what it used to return
    @test !isapprox(Float64.(vec(p_ok)), XQ*Float64.(rP.coeff[1:2]); rtol = 1e-6)

    errb = try (_sfb(result = rP, R = 3, seed = 1, every = 1000); nothing) catch e; e end
    @test errb !== nothing
    @test occursin("sfmodel_boot_marginal", sprint(showerror, errb))

    # re-running the fit restores the pairing
    rP2 = last(Test.collect_test_logs(() -> sfmodel_fit(spec = spP,
               method = sfmodel_method(method = :MLE), init = iniP, optim_options = optP,
               jlms_bc_index = false, marginal = true, show_table = false, verbose = false)))
    p_again = _sfp([:frontier])
    @test isapprox(Float64.(vec(p_again)), XP*Float64.(rP2.coeff[1:2]); rtol = 1e-10)
end
