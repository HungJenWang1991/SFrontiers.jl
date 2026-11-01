# Regression tests: input forms, refused arguments and error paths.
#
#  - `sfmodel_spec` normalizes one-column inputs (a length-N `frontier`/`zvar` vector to an
#    N x 1 matrix, an N x 1 `depvar` matrix to a length-N vector), but `predict` demanded
#    the stored shape. The very arrays a fit had accepted were refused when handed back:
#    "predict: `frontier` must be a matrix." on the estimation frame itself.
#  - The unified DSL took `eqnames`/`eq_indices` and dropped them on every method- and
#    datatype-specific route: DSL panel returned the panel defaults, and DSL MLE — the only
#    route an MLE-only model has — returned `String[]` / `Int64[]`. The keyword sibling
#    honoured the same arguments on the same models.
#  - `sfmodel_MoMTest` documents `ω` and `omega` as one argument. Given both with different
#    values it took `ω` and discarded `omega` without a word: `ω=2, omega=3` returned the
#    `ω=2` statistics exactly. (Covered in test_accessors_recipes_misc.jl's MoM testset neighbourhood;
#    pinned here on the exported entry point.)
#  - A cross-sectional `sfmodel_spec` accepted `id`, the removed `T_periods`, and a `zvar`
#    no equation can read. At fixed parameters the NLL was identical to the last bit with
#    `id=101:114`, with `T_periods=777`, and with `zvar` scaled by 1000.
#  - `method=:MLE` is analytic and reads no simulation setting, but warned about only four
#    of seven; `multiRand`, `chunks` and `distinct_Halton_length` produced no log at all.
#    `method=:MSLE` stored a `transformation` its likelihood cannot read (base and
#    `:expo_rule` gave the identical NLL).
#  - `warmstart_opt` without `warmstart_solver` was replaced by `nothing`, so an explicit
#    `(iterations = 999,)` vanished and the fit ran with no warmstart stage.
#  - `predict(level=...)` was in the docstring and the signature and was never read: 0.01,
#    0.999999 and a string all returned the same point predictions.
#  - A GPU fit accepts a `CuArray` as `draws=`; predict uses the stored draws when none are
#    passed and accepts the same object when it is passed. Passing it threw "Scalar indexing
#    is disallowed" while the plain CPU copy returned the cached jlms exactly. (GPU-gated below.)
#  - The fit constructors restrict `draws` to `AbstractFloat` — the house convention for a
#    typed keyword — but predict validated only shape and range, so `predict(draws=[0])` was
#    accepted while `sfmodel_method(draws=[0])` was a `TypeError`. Closed at predict's
#    end: predict now applies the fit's rule, and the fit's restriction stays.
#
# Left as they are: the Float32 predict/cache gaps (max 1.4e-7) and the
# cross-sectional GPU Float64 gaps (max 1.3e-15) are below the 1e-3 threshold.
using Test
using SFrontiers
using Random
using DataFrames
using Optim
using StatsAPI: predict

@testset "input forms and refused arguments" begin

    Random.seed!(20260824)
    n  = 14
    x1 = randn(n)
    X  = reshape(x1, :, 1)
    y  = X * [1.0] .- abs.(0.4 .* randn(n)) .+ 0.3 .* randn(n)
    z1 = randn(n)

    # Swallow the fits' @info/@warn records without letting them reach the test output. Uses
    # Test's own collector rather than the Logging stdlib: `Pkg.test()` runs in a sandbox whose
    # only test dependencies are CSV and Test (Project.toml [targets]), so `using Logging` loads
    # standalone but fails under Pkg.test with "Package Logging not found in current path" — the
    # same reason test_predict_draw_alignment.jl uses this collector.
    _quiet(f) = last(Test.collect_test_logs(f))
    _opt1() = sfmodel_opt(main_solver = NelderMead(), main_opt = (iterations = 3,))

    # ---- predict takes the one-column forms the spec takes
    @testset "predict takes the one-column forms the spec takes" begin
        # The spec normalizes all three, so predict must too — otherwise handing the
        # estimation data straight back is refused.
        sp = sfmodel_spec(depvar = reshape(y, :, 1), frontier = x1,
                          noise = :Normal, ineff = :HalfNormal)
        @test size(sp.mci_spec.frontier) == (n, 1)
        @test sp.mci_spec.depvar isa Vector{Float64}

        r = _quiet(() -> sfmodel_fit(spec = sp,
                method = sfmodel_method(method = :MCI, n_draws = 12, multiRand = false),
                init = sfmodel_init(spec = sp, message = false), optim_options = _opt1(),
                marginal = false, show_table = false, verbose = false))

        # frontier: a length-N vector is the N x 1 the spec stored.
        @test predict(r; frontier = x1, what = :frontier) ==
              predict(r; frontier = X,  what = :frontier)
        # depvar: an N x 1 matrix is the length-N vector the spec stored.
        @test predict(r; frontier = X, depvar = reshape(y, :, 1), what = :residuals) ==
              predict(r; frontier = X, depvar = y,                what = :residuals)
        # the vector-of-columns form the spec also accepts
        @test predict(r; frontier = [x1], what = :frontier) ==
              predict(r; frontier = X,    what = :frontier)

        # zvar: same rule, on the target that needs it.
        sph = sfmodel_spec(depvar = y, frontier = X, zvar = z1, noise = :Normal,
                           ineff = :HalfNormal, hetero = [:sigma_u_sq])
        @test size(sph.mci_spec.zvar) == (n, 1)
        rh = _quiet(() -> sfmodel_fit(spec = sph,
                method = sfmodel_method(method = :MCI, n_draws = 12, multiRand = false),
                init = sfmodel_init(spec = sph, message = false), optim_options = _opt1(),
                marginal = false, show_table = false, verbose = false))
        @test predict(rh; zvar = z1,                what = :marginal) ==
              predict(rh; zvar = reshape(z1, :, 1), what = :marginal)

        # A wider matrix is still refused, and by name — the shape rule did not go away.
        @test_throws "single-column matrix" predict(r; frontier = X,
                                                    depvar = hcat(y, y), what = :residuals)
    end

    # ---- the DSL forwards eqnames/eq_indices on every route
    @testset "the DSL forwards eqnames/eq_indices on every route" begin
        df = DataFrame(yv = y, x1 = x1, z1 = z1)
        eqn3 = ["SENT_frontier", "SENT_u", "SENT_v"]
        eqi3 = [91, 92, 93]

        # cross-sectional MLE — the route an MLE-only model has, and the one that returned [].
        s = sfmodel_spec(@useData(df), @depvar(yv), @frontier(x1); noise = :Normal,
                         ineff = :HalfNormal, eqnames = eqn3, eq_indices = eqi3)
        @test s.mci_spec.eqnames == eqn3
        @test s.mle_spec.eqnames == eqn3
        @test s.mle_spec.eq_indices == eqi3

        # panel_TFE — both the panel spec and its co-built MLE spec.
        nf, T = 6, 3
        np  = nf * T
        idp = repeat(1:nf, inner = T)
        xp  = randn(np); zp = randn(np)
        yp  = xp .* 1.0 .- abs.(0.3 .* randn(np)) .+ 0.2 .* randn(np)
        dfp = DataFrame(yv = yp, x1 = xp, z1 = zp, firm = idp)
        eqn4 = ["SENT_frontier", "SENT_scaling", "SENT_u", "SENT_v"]
        eqi4 = [81, 82, 83, 84]

        sp = sfmodel_spec(@useData(dfp), @depvar(yv), @frontier(x1), @zvar(z1), @id(firm);
                          datatype = :panel_TFE, noise = :Normal, ineff = :HalfNormal,
                          eqnames = eqn4, eq_indices = eqi4)
        @test sp.panel_spec.eqnames == eqn4
        @test sp.panel_spec.eq_indices == eqi4
        @test sp.mle_spec.eqnames == eqn4

        # and it agrees with the keyword sibling, which always honoured them
        skw = sfmodel_spec(depvar = yp, frontier = reshape(xp, :, 1),
                           zvar = reshape(zp, :, 1), id = idp, datatype = :panel_TFE,
                           noise = :Normal, ineff = :HalfNormal,
                           eqnames = eqn4, eq_indices = eqi4)
        @test sp.panel_spec.eqnames == skw.panel_spec.eqnames
        @test sp.panel_spec.eq_indices == skw.panel_spec.eq_indices

        # the MLE-only panels
        xc = hcat(randn(np), randn(np))
        yc = xc * [0.6, -0.4] .- abs.(0.4 .* randn(np)) .+ 0.3 .* randn(np)
        dfc = DataFrame(yv = yc, x1 = xc[:, 1], x2 = xc[:, 2], firm = idp)
        for dt in (:panel_TFE_CSW, :panel_TRE)
            sc = sfmodel_spec(@useData(dfc), @depvar(yv), @frontier(x1, x2), @id(firm);
                              datatype = dt, noise = :Normal, ineff = :HalfNormal,
                              eqnames = eqn3, eq_indices = eqi3)
            @test sc.mle_spec.eqnames == eqn3
            @test sc.mle_spec.eq_indices == eqi3
        end
    end

    # ---- conflicting omega spellings are named, not silently resolved
    @testset "conflicting omega spellings are named, not silently resolved" begin
        Random.seed!(7301)
        nm = 180
        xm = collect(range(-1.0, 1.0; length = nm))
        Xm = hcat(ones(nm), xm)
        ym = Xm * [1.1, 0.35] .+ 0.25 .* randn(nm) .- abs.(0.45 .* randn(nm))
        dfm = DataFrame(y = ym, c = ones(nm), x = xm)
        margs = ((:type, [:prod]), (:dist, [:h]), (:depvar, [:y]), (:frontier, [:c, :x]))
        _mom(; kw...) = _quiet(() -> sfmodel_MoMTest(margs...; data = dfm,
                                                     verbose = false, testonly = true, kw...))

        # Each spelling on its own is untouched, and the two pick out different tests.
        r2 = _mom(ω = 2.0)
        r3 = _mom(omega = 3.0)
        @test r2.MoMtest != r3.MoMtest

        # Both, in conflict: `omega` used to be discarded and the ω=2 statistics returned.
        @test_throws "two spellings of the same argument" _mom(ω = 2.0, omega = 3.0)
        # The same value under both spellings is not a conflict, and is the ω-only result.
        @test _mom(ω = 2.0, omega = 2.0).MoMtest == r2.MoMtest
        # Neither supplied: the default is unchanged.
        @test _mom() !== nothing
    end

    # ---- a cross-sectional spec refuses what it cannot read
    @testset "a cross-sectional spec refuses what it cannot read" begin
        @test_throws "only valid for panel models" sfmodel_spec(
            depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal,
            id = collect(1:n))
        @test_throws "no longer supported" sfmodel_spec(
            depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal,
            T_periods = 777)
        # both messages say that the argument is not accepted in a cross-sectional model
        @test_throws "`id` is not accepted in cross-sectional models" sfmodel_spec(
            depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal,
            id = collect(1:n))
        @test_throws "`T_periods` is not accepted in cross-sectional models" sfmodel_spec(
            depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal,
            T_periods = 777)
        @test_throws "cannot be used by this model" sfmodel_spec(
            depvar = y, frontier = X, zvar = ones(n, 1),
            noise = :Normal, ineff = :HalfNormal)

        # T_periods is refused on the panel datatypes and on the DSL too
        df = DataFrame(yv = y, x1 = x1, z1 = z1)
        @test_throws "no longer supported" sfmodel_spec(
            @useData(df), @depvar(yv), @frontier(x1); noise = :Normal,
            ineff = :HalfNormal, T_periods = 777)
        @test_throws "no longer supported" sfmodel_spec(
            depvar = y, frontier = X, zvar = reshape(z1, :, 1), id = collect(1:n),
            datatype = :panel_TFE, noise = :Normal, ineff = :HalfNormal, T_periods = 777)

        # a zvar an equation CAN read is still accepted, on both forms
        @test sfmodel_spec(depvar = y, frontier = X, zvar = z1, noise = :Normal,
                           ineff = :HalfNormal, hetero = [:sigma_u_sq]) !== nothing
        @test sfmodel_spec(depvar = y, frontier = X, zvar = z1, noise = :Normal,
                           ineff = :TruncatedNormal, hetero = :scaling) !== nothing
        @test sfmodel_spec(@useData(df), @depvar(yv), @frontier(x1), @zvar(z1);
                           noise = :Normal, ineff = :HalfNormal,
                           hetero = [:sigma_u_sq]) !== nothing
        @test_throws "cannot be used by this model" sfmodel_spec(
            @useData(df), @depvar(yv), @frontier(x1), @zvar(z1);
            noise = :Normal, ineff = :HalfNormal)
    end

    # ---- a method refuses settings it cannot read
    @testset "a method refuses settings it cannot read" begin
        # all eight simulation settings, including the three the old check never saw
        @test_throws "transformation"          sfmodel_method(method = :MLE, transformation = :expo_rule)
        @test_throws "draws"                   sfmodel_method(method = :MLE, draws = [0.5])
        @test_throws "n_draws"                 sfmodel_method(method = :MLE, n_draws = 64)
        @test_throws "multiRand"               sfmodel_method(method = :MLE, multiRand = false)
        @test_throws "GPU"                     sfmodel_method(method = :MLE, GPU = true)
        @test_throws "chunks"                  sfmodel_method(method = :MLE, chunks = 777)
        @test_throws "distinct_Halton_length"  sfmodel_method(method = :MLE, distinct_Halton_length = 3)
        @test_throws "distinct_Halton_length"  sfmodel_method(method = :MLE, distinct_Halton_length = 2^15 - 1)
        @test_throws "recycling"               sfmodel_method(method = :MLE, recycling = true)
        # the plain call, and defaults written out explicitly, still build
        @test sfmodel_method(method = :MLE) !== nothing
        @test sfmodel_method(method = :MLE, multiRand = true, GPU = false, chunks = 10) !== nothing

        # transformation is MCI-only, on the unified and the direct panel constructor
        @test_throws "only effective with `method=:MCI`" sfmodel_method(method = :MSLE,
                                                                        transformation = :expo_rule)
        @test_throws "only effective with `method=:MCI`" sfmodel_panel_method(method = :MSLE,
                                                                              transformation = :expo_rule)
        @test sfmodel_method(method = :MCI, transformation = :expo_rule, n_draws = 8) !== nothing
        @test sfmodel_panel_method(method = :MCI, transformation = :expo_rule, n_draws = 8) !== nothing
        @test sfmodel_method(method = :MSLE, n_draws = 8) !== nothing
    end

    # ---- warmstart_opt without a solver is refused, not discarded
    @testset "warmstart_opt without a solver is refused, not discarded" begin
        for build in (kw -> sfmodel_opt(; kw..., main_solver = Newton(),
                                          main_opt = (iterations = 10,)),
                      kw -> sfmodel_panel_opt(; kw..., main_solver = Newton(),
                                                main_opt = (iterations = 10,)),
                      kw -> SFrontiers.MSLE_Backend.sfmodel_MSLE_opt(; kw..., main_solver = Newton(),
                                                                     main_opt = (iterations = 10,)))
            @test_throws "needs a `warmstart_solver`" build((warmstart_opt = (iterations = 999,),))
            # paired: the caller's own options survive, they are not replaced by the default
            @test build((warmstart_solver = NelderMead(),
                         warmstart_opt = (iterations = 999,))).warmstart_opt.iterations == 999
            # neither supplied: unchanged
            @test build(NamedTuple()).warmstart_opt === nothing
        end
        # the `:MLE` backend's own builder refuses it too, before it stores any option
        smle = SFrontiers.MLE_Backend.SFmle
        dic0 = isdefined(smle, :_dicOPT) ? copy(smle._dicOPT) : nothing
        @test_throws "needs a `warmstart_solver`" smle.sfmodel_opt(warmstart_opt = (iterations = 999,))
        @test (isdefined(smle, :_dicOPT) ? smle._dicOPT : nothing) == dic0
    end

    # ---- predict refuses the dead `level` keyword
    @testset "predict refuses the dead `level` keyword" begin
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal)
        r  = _quiet(() -> sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE),
                init = sfmodel_init(spec = sp, message = false), optim_options = _opt1(),
                marginal = false, show_table = false, verbose = false))
        for v in (0.01, 0.95, 0.999999, "nonsense")
            @test_throws "`level` is not accepted" predict(r; frontier = X,
                                                            what = :frontier, level = v)
        end
        @test predict(r; frontier = X, what = :frontier, level = nothing) ==
              predict(r; frontier = X, what = :frontier)
    end

    # ---- fit and predict take the same numeric types for custom draws
    @testset "fit and predict take the same numeric types for custom draws" begin
        # The fit side is unchanged: `AbstractVecOrMat{<:AbstractFloat}`, refused by the
        # keyword's own type, the house convention for a typed keyword.
        for m in (:MCI, :MSLE)
            @test_throws TypeError sfmodel_method(method = m, draws = [0])
            @test_throws TypeError sfmodel_method(method = m, draws = [0//1, 1//2])
            @test_throws TypeError sfmodel_panel_method(method = m, draws = [0])
        end

        # predict now reads the same rule. Before the repair it accepted `[0]` and returned a
        # number, so one contract had two answers at the two ends of one workflow.
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal)
        d  = [0.0, 0.25, 0.5]
        r  = _quiet(() -> sfmodel_fit(spec = sp,
                method = sfmodel_method(method = :MCI, draws = d, multiRand = false),
                init = sfmodel_init(spec = sp, message = false), optim_options = _opt1(),
                marginal = false, show_table = false, verbose = false))
        for bad in ([0, 0, 0], [0//1, 1//4, 1//2], Real[0.0, 0.25, 0.5])
            @test_throws "floating-point element type" predict(r; frontier = X, depvar = y,
                                                               draws = bad, what = :jlms)
        end
        # the float object the fit was actually given still reproduces the cached index
        @test predict(r; frontier = X, depvar = y, draws = d, what = :jlms) == r.jlms
        # Float32 draws are AbstractFloat and stay acceptable at both ends
        @test sfmodel_method(method = :MCI, draws = Float32[0.25, 0.5]) !== nothing

        # panel predict reads the same rule
        nf, T = 6, 3
        np  = nf * T
        idp = repeat(1:nf, inner = T)
        xp  = randn(np); zp = randn(np)
        yp  = xp .* 1.0 .- abs.(0.3 .* randn(np)) .+ 0.2 .* randn(np)
        spp = sfmodel_spec(depvar = yp, frontier = reshape(xp, :, 1),
                           zvar = reshape(zp, :, 1), id = idp, datatype = :panel_TFE,
                           noise = :Normal, ineff = :HalfNormal)
        rp = _quiet(() -> sfmodel_fit(spec = spp,
                method = sfmodel_panel_method(method = :MCI, draws = d, multiRand = false),
                init = sfmodel_init(spec = spp), optim_options = _opt1(),
                marginal = false, show_table = false, verbose = false))
        pargs = (frontier = reshape(xp, :, 1), zvar = reshape(zp, :, 1), depvar = yp,
                 id = idp, what = :jlms)
        @test_throws "floating-point element type" predict(rp; pargs..., draws = [0, 0, 0])
        @test predict(rp; pargs..., draws = d) == rp.jlms

        # A Float32 panel fit is the case that gets the guard's ORDER wrong if it is placed in
        # the per-backend resolver: that route narrows with `Tfit_p.(user_draws)` first, so
        # `[0]` arrives as `Float32[0.0]` and passes. The check therefore sits at
        # `_dispatch_jlms_bc`, ahead of every narrowing.
        xf = Float32.(xp); zf = Float32.(zp)
        yf = xf .* 1.0f0 .- abs.(0.3f0 .* Float32.(randn(np))) .+ 0.2f0 .* Float32.(randn(np))
        sp32 = sfmodel_spec(depvar = yf, frontier = reshape(xf, :, 1),
                            zvar = reshape(zf, :, 1), id = idp, datatype = :panel_TFE,
                            noise = :Normal, ineff = :HalfNormal)
        d32 = Float32[0.25, 0.5, 0.75]
        r32 = _quiet(() -> sfmodel_fit(spec = sp32,
                method = sfmodel_panel_method(method = :MCI, draws = d32, multiRand = false),
                init = sfmodel_init(spec = sp32), optim_options = _opt1(),
                marginal = false, show_table = false, verbose = false))
        @test eltype(sp32.panel_spec.depvar) === Float32
        args32 = (frontier = reshape(xf, :, 1), zvar = reshape(zf, :, 1), depvar = yf,
                  id = idp, what = :jlms)
        for bad in ([0, 0, 0], [0//1, 1//4, 1//2])
            @test_throws "floating-point element type" predict(r32; args32..., draws = bad)
        end
        @test predict(r32; args32..., draws = d32) isa Vector   # the valid object still works
    end
end

# ---------------------------------------------------------------------------
# The GPU draw object at predict — GPU-gated, and run in a SEPARATE process: `using CUDA` must precede
# `using SFrontiers` for the GPU overloads to register, and this file is included after
# SFrontiers is already loaded. Same mechanism as `test_gpu_equivalence.jl`, including the
# load path — `Pkg.test()` sandboxes it, and CUDA is deliberately not a test dependency, so
# the subprocess gets the test environment followed by the user's default environment.
# Skipped as a no-op testset when CUDA is unavailable, and when its version is newer than
# the versions SFrontiers supports (6.4 or earlier).
# ---------------------------------------------------------------------------
@testset "predict takes the GPU draw object the fit took (gated)" begin
    julia   = joinpath(Sys.BINDIR, Base.julia_exename())
    _lp = join([Base.load_path(); "@v#.#"; "@stdlib"], Sys.iswindows() ? ";" : ":")
    probe_code = "try; using CUDA; println(CUDA.functional()); print(pkgversion(CUDA)); catch; print(false); end"
    probe_cmd = addenv(`$julia --startup-file=no -e $probe_code`, "JULIA_LOAD_PATH" => _lp)
    probe = try
        split(read(probe_cmd, String))
    catch
        String[]
    end
    gpu_ok = !isempty(probe) && probe[1] == "true"
    cuda_v = length(probe) > 1 ? tryparse(VersionNumber, probe[2]) : nothing

    if !gpu_ok
        @test_skip "CUDA not available/functional - GPU custom-draw roundtrip skipped"
    elseif cuda_v !== nothing && cuda_v >= v"6.5"
        @warn("GPU custom-draw roundtrip skipped: CUDA.jl $cuda_v in the default environment is " *
              "newer than the versions SFrontiers supports (6.4 or earlier)")
        @test_skip "CUDA.jl newer than 6.4 - GPU custom-draw roundtrip skipped"
    else
        runner = joinpath(@__DIR__, "gpu_custom_draw_runner.jl")
        cmd = addenv(`$julia --startup-file=no $runner`,
                     "JULIA_LOAD_PATH" => _lp)
        log = IOBuffer()
        ok  = success(pipeline(cmd; stdout = log, stderr = log))
        out = String(take!(log))
        @test ok
        # every cell ran, and none of them differed from its CPU-copy control
        m = match(r"CUSTOM-DRAW ROUNDTRIP COMPLETE: (\d+) cases, (\d+) mismatches", out)
        @test m !== nothing
        if m !== nothing
            @test parse(Int, m.captures[1]) == 16
            @test parse(Int, m.captures[2]) == 0
        end
        # the scalar-indexing failure must not reappear
        @test !occursin("Scalar indexing is disallowed", out)
    end
end

# ---------------------------------------------------------------------------
# Error paths and their messages.
#
#  - `_init_kwargs` tested `haskey`, not the value, so an explicit `nothing` on one of
#    the `scaling`/`hscale` aliases was read as a supplied value: `hscale=0.3,
#    scaling=nothing` was refused as a conflict, and so was
#    `scaling=nothing, hscale=nothing`, which supplies nothing at all.
#  - The Float32 panel route converted the draw object before the vector/matrix guard ran,
#    so a string threw a raw `MethodError: no method matching Float32(::String)`.
#  - The simulation normalizers unwrapped a list `depvar` with `only`, whose failure is
#    Base's "Collection has multiple elements". Since the unified constructor builds a
#    simulation spec first, the analytic backend's own clear message was unreachable from
#    every public route.
#  - `hscale=` on the two MLE-only panel models gave a clear "no scaling function" error;
#    the canonical `scaling=` fell through to a raw `MethodError`, the analytic
#    initializer having no such keyword.
#  - The removed-`method` error claimed the vector is accepted under `:MCI`, `:MSLE` and
#    `:MLE` alike. A Weibull model gets that message and then its MLE fit errors
#    "not available"; an MLE-only CSW model is the mirror image under `:MCI`.
#  - The draw-of-one message said a draw of 1 maps to an infinite inefficiency (NaN
#    likelihood) on all three backends. MSLE's quantile kernels clamp instead — measured,
#    `myErlangStdQuantile(1.0, 2, 0.0)` is 40.461567483087464, finite, and identical to
#    the value at `prevfloat(1.0)`.
#  - An empty `draws` at predict reported "0 draws per observation; the fit used N" —
#    a downstream count mismatch rather than the dedicated emptiness error the fit
#    boundary and panel prediction both give.
#  - Fits made with a D x 1 column are accepted, and prediction with the same column
#    works, but the missing-draws advice listed only the vector, 1 x D row and N x D
#    block.
#  - On a homoscedastic model the alias-conflict message explained the two names as
#    belonging to "the same scaling function". That model has no scaling function.
#
# Four guards that no public route reaches are kept as internal invariants and labelled
# as such in the source, rather than deleted — a named error is worth
# more than the line it costs if the invariant ever breaks.
# ---------------------------------------------------------------------------
@testset "error paths and their messages" begin

    Random.seed!(20260825)
    n7 = 12
    X7  = hcat(ones(n7), randn(n7), randn(n7))
    Zs7 = reshape(randn(n7), :, 1)
    y7  = X7 * [1.0, 0.5, -0.3] .- abs.(0.4 .* randn(n7)) .+ 0.3 .* randn(n7)

    _q7(f) = last(Test.collect_test_logs(f))
    _opt7() = sfmodel_opt(main_solver = NelderMead(), main_opt = (iterations = 2,))
    _msg(f) = try; f(); "" catch ex; sprint(showerror, ex) end

    sp_scal = sfmodel_spec(depvar = y7, frontier = X7, zvar = Zs7, noise = :Normal,
                           ineff = :HalfNormal, hetero = :scaling)
    sp_homo = sfmodel_spec(depvar = y7, frontier = X7, noise = :Normal, ineff = :HalfNormal)

    # ---- an explicit `nothing` alias behaves as an omitted component
    @testset "an explicit `nothing` alias behaves as an omitted component" begin
        one_value = _q7(() -> sfmodel_init(spec = sp_scal, scaling = 0.3))
        no_value  = _q7(() -> sfmodel_init(spec = sp_scal))
        @test one_value != no_value      # the sentinel is distinguishable, so nothing below is vacuous
        @test _q7(() -> sfmodel_init(spec = sp_scal, hscale = 0.3, scaling = nothing)) == one_value
        @test _q7(() -> sfmodel_init(spec = sp_scal, scaling = 0.3, hscale = nothing)) == one_value
        @test _q7(() -> sfmodel_init(spec = sp_scal, scaling = nothing, hscale = nothing)) == no_value
        # two genuine values are still a conflict, equal or not
        @test_throws "not both" sfmodel_init(spec = sp_scal, scaling = 0.3, hscale = 0.4)
        @test_throws "not both" sfmodel_init(spec = sp_scal, scaling = 0.3, hscale = 0.3)
    end

    # ---- the alias-conflict message fits the model it is raised on
    @testset "the alias-conflict message fits the model it is raised on" begin
        # a model that HAS a scaling function keeps the two-names explanation
        @test_throws "same scaling function" sfmodel_init(spec = sp_scal, scaling = 0.3,
                                                          hscale = 0.4)
        # one that does not must not be told about a function it does not have
        e = _msg(() -> sfmodel_init(spec = sp_homo, scaling = 0.3, hscale = 0.3))
        @test occursin("no scaling function", e)
        @test occursin("Neither `scaling` nor `hscale` is part of the current model", e)
        # and it must NOT describe a scaling function this model does not have
        @test !occursin("two names for the coefficients of the same scaling function", e)
        # each alias on its own was always right here, and still is
        @test_throws "Not part of the current model" sfmodel_init(spec = sp_homo, scaling = 0.3)
        @test_throws "Not part of the current model" sfmodel_init(spec = sp_homo, hscale = 0.3)
    end

    # ---- a list `depvar` reaches the package's own message
    @testset "a list `depvar` reaches the package's own message" begin
        for build in (() -> sfmodel_spec(depvar = [y7, y7], frontier = X7,
                                         noise = :Normal, ineff = :HalfNormal),
                      () -> SFrontiers.MCI_Backend.sfmodel_spec(depvar = [y7, y7], frontier = X7,
                                         noise = :Normal, ineff = :HalfNormal),
                      () -> SFrontiers.MSLE_Backend.sfmodel_spec(depvar = [y7, y7], frontier = X7,
                                         noise = :Normal, ineff = :HalfNormal))
            @test_throws "not a list of 2" build()
        end
        # the one-element list form still works and still gives the right N
        @test sfmodel_spec(depvar = [y7], frontier = X7, noise = :Normal,
                           ineff = :HalfNormal).mci_spec.N == n7
    end

    # ---- `scaling` gets the same refusal as its `hscale` synonym
    @testset "`scaling` gets the same refusal as its `hscale` synonym" begin
        nf7, T7 = 4, 3
        np7  = nf7 * T7
        idp7 = repeat(1:nf7, inner = T7)
        Xp7  = hcat(randn(np7), randn(np7))
        Xpc7 = hcat(ones(np7), randn(np7))
        yp7  = Xp7  * [0.6, -0.4] .- abs.(0.4 .* randn(np7)) .+ 0.3 .* randn(np7)
        ypc7 = Xpc7 * [0.6, -0.4] .- abs.(0.4 .* randn(np7)) .+ 0.3 .* randn(np7)
        csw = sfmodel_spec(datatype = :panel_TFE_CSW, depvar = yp7, frontier = Xp7,
                           id = idp7, noise = :Normal, ineff = :HalfNormal)
        tre = sfmodel_spec(datatype = :panel_TRE, depvar = ypc7, frontier = Xpc7,
                           id = idp7, noise = :Normal, ineff = :HalfNormal)
        for s in (csw, tre)
            @test_throws "no scaling function" sfmodel_init(spec = s, hscale = 0.77)
            @test_throws "no scaling function" sfmodel_init(spec = s, scaling = 0.77)
            # and the message names the spelling the caller actually used
            @test occursin("`hscale` is not a parameter",
                           _msg(() -> sfmodel_init(spec = s, hscale = 0.77)))
            @test occursin("`scaling` is not a parameter",
                           _msg(() -> sfmodel_init(spec = s, scaling = 0.77)))
            @test_throws "not both" sfmodel_init(spec = s, hscale = 0.77, scaling = 0.77)
        end
    end

    # ---- the removed-`method` error claims only what is true
    @testset "the removed-`method` error claims only what is true" begin
        e = _msg(() -> sfmodel_init(spec = sp_homo, method = :MLE))
        @test occursin("no longer takes a `method` argument", e)
        @test occursin("every estimation method this model supports", e)
        # the claim the old text made, which the control below falsifies
        @test !occursin("`:MCI`, `:MSLE` and `:MLE` alike", e)
        spw = sfmodel_spec(depvar = y7, frontier = X7, noise = :Normal, ineff = :Weibull)
        @test_throws "no longer takes a `method` argument" sfmodel_init(spec = spw, method = :MLE)
        @test_throws "is not available" _q7(() -> sfmodel_fit(spec = spw,
                method = sfmodel_method(method = :MLE), init = nothing,
                optim_options = _opt7(), marginal = false, show_table = false, verbose = false))
    end

    # ---- the draw-of-one message names each backend's own consequence
    @testset "the draw-of-one message names each backend's own consequence" begin
        mci   = _msg(() -> SFrontiers.MCI_Backend._validate_user_draws([1.0]))
        msle  = _msg(() -> SFrontiers.MSLE_Backend._validate_user_draws([1.0]))
        panel = _msg(() -> SFrontiers.Panel_Backend._validate_panel_user_draws([1.0]))
        for m in (mci, msle, panel)
            @test occursin("must lie in [0, 1)", m)
            @test occursin("half-open", m)          # the contract itself, stated first
        end
        @test occursin("the likelihood is NaN", mci)     # MCI really does produce a NaN
        @test occursin("clamp", msle)                    # MSLE clamps instead
        @test occursin("do not return a NaN", msle)      # and says so explicitly
        @test occursin("NaN", panel) && occursin("clamp", panel)   # panel runs both methods
        # the old text, which claimed the NaN for all three backends alike
        for m in (mci, msle, panel)
            @test !occursin("A draw of exactly 1 maps to an infinite inefficiency (NaN likelihood)", m)
        end
        # the measurement behind the correction: MSLE's Erlang quantile is finite at 1
        q1 = SFrontiers.MSLE_Backend.myErlangStdQuantile(1.0, 2, 0.0)
        @test isfinite(q1)
        @test q1 == SFrontiers.MSLE_Backend.myErlangStdQuantile(prevfloat(1.0), 2, 0.0)
        # zero is still accepted and one still refused: the contract itself did not move
        @test SFrontiers.MCI_Backend._validate_user_draws([0.0, 0.5]) === nothing
    end

    # ---- an empty `draws` is reported as empty
    @testset "an empty `draws` is reported as empty" begin
        @test_throws "`draws` is empty" SFrontiers._validate_user_draws(Float64[], 5, 8)
        # the two boundaries it now agrees with
        @test_throws "`draws` is empty" SFrontiers.MCI_Backend._validate_user_draws(Float64[])
        @test_throws "`draws` is empty" SFrontiers.Panel_Backend._validate_panel_user_draws(Float64[])
        # a non-empty count mismatch still reports the count, so the reorder cost nothing
        @test_throws "the fit used 5" SFrontiers._validate_user_draws([0.1, 0.2], 5, 8)
    end

    # ---- a D × 1 column of draws is accepted by the fit and by predict, and predict uses it when draws are omitted
    @testset "a D × 1 column of draws is accepted by the fit and by predict, and predict uses it when draws are omitted" begin
        D7 = 6
        dcol = reshape(collect(range(0.05, 0.85; length = D7)), D7, 1)
        for meth in (:MCI, :MSLE)
            r = _q7(() -> sfmodel_fit(spec = sp_homo,
                    method = sfmodel_method(method = meth, draws = dcol, multiRand = false),
                    init = nothing, optim_options = _opt7(), marginal = false,
                    show_table = false, verbose = false))
            # the column really is accepted at both boundaries
            @test _q7(() -> predict(r; frontier = X7, depvar = y7, draws = dcol,
                                    what = :jlms)) isa Vector
            # predict uses the stored D × 1 column when draws are omitted
            @test _q7(() -> predict(r; frontier = X7, depvar = y7, what = :jlms)) ==
                  _q7(() -> predict(r; frontier = X7, depvar = y7, draws = dcol, what = :jlms))
        end
        # the same for a panel fit
        nf7, T7 = 4, 3
        np7  = nf7 * T7
        idp7 = repeat(1:nf7, inner = T7)
        Xp7  = hcat(randn(np7), randn(np7))
        zp7  = randn(np7)
        yp7  = Xp7 * [0.6, -0.4] .- abs.(0.4 .* randn(np7)) .+ 0.3 .* randn(np7)
        spp = sfmodel_spec(datatype = :panel_TFE, depvar = yp7, frontier = Xp7,
                           zvar = reshape(zp7, :, 1), id = idp7,
                           noise = :Normal, ineff = :HalfNormal)
        rp = _q7(() -> sfmodel_fit(spec = spp,
                method = sfmodel_panel_method(method = :MCI, draws = dcol, multiRand = false),
                init = nothing, optim_options = _opt7(), marginal = false,
                show_table = false, verbose = false))
        pa = (frontier = Xp7, zvar = reshape(zp7, :, 1), depvar = yp7, id = idp7, what = :jlms)
        @test _q7(() -> predict(rp; pa..., draws = dcol)) isa Vector
        @test _q7(() -> predict(rp; pa...)) == _q7(() -> predict(rp; pa..., draws = dcol))
    end

    # ---- a non-array `draws` is refused before any element-type conversion
    @testset "a non-array `draws` is refused before any element-type conversion" begin
        # The Float32 panel route is the one that converted first; the guard now sits ahead
        # of it, so these three reach the package error instead of a raw MethodError or a
        # raw broadcasting ArgumentError.
        nf7, T7 = 4, 3
        np7  = nf7 * T7
        idp7 = repeat(1:nf7, inner = T7)
        xf = Float32.(randn(np7)); zf = Float32.(randn(np7))
        yf = xf .* 1.0f0 .- abs.(0.3f0 .* Float32.(randn(np7))) .+ 0.2f0 .* Float32.(randn(np7))
        sp32 = sfmodel_spec(datatype = :panel_TFE, depvar = yf, frontier = reshape(xf, :, 1),
                            zvar = reshape(zf, :, 1), id = idp7,
                            noise = :Normal, ineff = :HalfNormal)
        r32 = _q7(() -> sfmodel_fit(spec = sp32,
                method = sfmodel_panel_method(method = :MCI, draws = Float32[0.25, 0.5, 0.75],
                                              multiRand = false),
                init = nothing, optim_options = _opt7(), marginal = false,
                show_table = false, verbose = false))
        a32 = (frontier = reshape(xf, :, 1), zvar = reshape(zf, :, 1), depvar = yf,
               id = idp7, what = :jlms)
        for bad in (0.5, "not draws", Dict(:x => 0.5))
            @test_throws "must be a vector or a matrix" predict(r32; a32..., draws = bad)
        end
    end
end
