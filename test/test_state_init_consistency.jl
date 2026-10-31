# Regression tests: state isolation, starting values and predict constants.
#
#  - A deprecated Vararg `sfmodel_spec` marked `_dicM[:_spec_built] = false`, and the fit read
#    that marker instead of asking whether IT had been handed a spec. Every later
#    `sfmodel_fit(spec = A)` then silently estimated whatever the Vararg call wrote:
#    ll -40.181666 became -127.146739, with B's coefficient vector exactly.
#  - `optim_options = nothing` is documented as "fresh defaults", but the unified fit passed
#    that `nothing` straight through, and the MLE backend cannot tell it apart from a bare
#    legacy fit. A standing deprecated `sfmodel_opt(...)` therefore governed a modern fit:
#    ll -61.386553 became -147.380654, not converged, on three Nelder-Mead iterations.
#  - `predict()` rebuilt the backend constants at Float64. Several are type-dependent
#    numerical GUARDS, not scale factors — the inverse-erf clamp is 0.999996 in Float32 and
#    1-7.1e-15 in Float64 — so predict evaluated a different integrand from the one the fit
#    maximised. Measured 0.155 absolute on a converged Float32 MSLE fit's jlms.
#  - The panel starting-value builders concatenated component blocks without checking their
#    widths. Offsetting mistakes kept the total right, so the fit's own length check passed:
#    `frontier=[0.41], ln_sigma_u_sq=[-0.71,-0.72]` returned [0.41,-0.71,-0.72,-0.91], making
#    a variance start the second frontier coefficient. `:panel_TFE` broadcast a short block
#    and truncated a long one, and refused a documented scalar with `MethodError: vec`.
#  - The frontier OLS default was solved in the specification's element type by the builder
#    and in Float64 by the analytic fit, so `sfmodel_init(spec)` did not return the vector the
#    fit builds for itself — by Float32 eps, or 9.03 absolute on an ill-conditioned design.
#  - panel_TFE truncated normal starts mu at 0.0 on its own builder and fit, but the analytic
#    backend gave it the flat 0.1, so one model had two starting points depending on `method=`.
#  - The MCI/MSLE `print_table` core still coupled coef's element type to vcov's through one
#    type parameter, which had been relaxed only on the forwarding method.
#  - The rename to `:sigma_u_sq` missed the marginal-effect name lookup (which still keyed on
#    `:sigma_sq`, so a DSL determinant name fell through to "z2"), the panel Lognormal and
#    Rayleigh labels, and the analytic Exponential's converted row.
#  - `sfmodel_init` on an MLE-only panel model left `_dicINI` written, against the
#    guarantee that it writes no module state.
using Test
using SFrontiers
using Random
using DataFrames
using LinearAlgebra
using Optim

@testset "state isolation, starting values and predict constants" begin

    Random.seed!(20260822)
    n = 120
    X = hcat(ones(n), randn(n), randn(n))
    y = X * [1.0, 0.5, -0.3] .- abs.(0.4 .* randn(n)) .+ 0.3 .* randn(n)
    XB = hcat(ones(n), randn(n), randn(n))
    yB = XB * [-2.0, 2.5, 3.5] .- abs.(0.8 .* randn(n)) .+ 0.5 .* randn(n)

    nf, T = 20, 5
    np  = nf * T
    idp = repeat(1:nf, inner = T)
    XPn = hcat(randn(np), randn(np))
    ZPn = hcat(randn(np))
    yp  = XPn * [0.6, -0.4] .- abs.(0.4 .* randn(np)) .+ 0.3 .* randn(np)

    opt1 = sfmodel_opt(main_solver = NelderMead(), main_opt = (iterations = 1, g_abstol = 1e-8))
    MLE  = sfmodel_method(method = :MLE)
    MCI  = sfmodel_method(method = :MCI,  n_draws = 64)
    MSLE = sfmodel_method(method = :MSLE, n_draws = 64)

    start_of(sp, m; init = nothing) =
        sfmodel_fit(spec = sp, method = m, init = init, optim_options = opt1,
                    jlms_bc_index = false, marginal = false,
                    show_table = false, verbose = false).main_ini

    co(r) = collect(float.(r.coeff))
    threw(f) = try (f(); false) catch; true end

    # ---- a deprecated Vararg spec does not displace an explicit spec
    @testset "a deprecated Vararg spec does not displace an explicit spec" begin
        S = SFrontiers.MLE_Backend.SFmle
        optN = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 300, g_abstol = 1e-7))
        fit(sp) = sfmodel_fit(spec = sp, method = MLE, init = nothing, optim_options = optN,
                              jlms_bc_index = false, marginal = false,
                              show_table = false, verbose = false)

        spA = sfmodel_spec(depvar = y,  frontier = X,  noise = :Normal, ineff = :HalfNormal)
        a   = co(fit(spA))

        vararg_B() = S.sfmodel_spec(S.sftype(S.production), S.sfdist(S.half),
                                    S.depvar(yB), S.frontier(XB),
                                    S.sigma_u_2(ones(n)), S.sigma_v_2(ones(n)))

        vararg_B()
        @test S._dicM[:_spec_built] === false     # the marker really is off
        @test co(fit(spA)) == a                   # and the fit still returns A, bit for bit

        # the spec-less legacy flow the marker protects is unchanged: a bare fit after the
        # Vararg builder keeps reading the state that builder wrote, not `_MLE_spec`
        sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal)
        vararg_B()
        rL = S.sfmodel_fit(show_table = false, verbose = false,
                           jlms_bc_index = false, marginal = false)
        @test Vector(S._dicM[:sdf][:, :depvar]) == yB
        @test !isapprox(co(rL), a; rtol = 1e-3)

        # a Vararg call that THROWS leaves `_dicM` half-built with no `:sdf`; an
        # explicit-spec fit must still work (this used to raise `KeyError: key :sdf not found`)
        try
            S.sfmodel_spec(S.sftype(S.production), S.sfdist(S.half), S.depvar(yB))
        catch
        end
        @test co(fit(spA)) == a
    end

    # ---- a deprecated sfmodel_opt does not govern a modern default fit
    @testset "a deprecated sfmodel_opt does not govern a modern default fit" begin
        S = SFrontiers.MLE_Backend.SFmle
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal)
        run_(m, opt) = sfmodel_fit(spec = sp, method = m, init = nothing, optim_options = opt,
                                   marginal = false, show_table = false, verbose = false)

        first_default = run_(MLE, nothing)
        S.sfmodel_opt(main_solver = NelderMead(), main_opt = (iterations = 3, g_abstol = 0.1))
        @test S._dicOPT[:_user_set] == true
        again = run_(MLE, nothing)
        @test S._dicOPT[:main_maxIT] == 200                # the unified default
        @test again.converged == first_default.converged == true
        @test co(again) == co(first_default)               # bit for bit

        # the same boundary serves the MLE-only panel models
        spp = sfmodel_spec(datatype = :panel_TFE_CSW, depvar = yp, frontier = XPn, id = idp,
                           noise = :Normal, ineff = :HalfNormal)
        runp(opt) = sfmodel_fit(spec = spp, method = MLE, init = nothing, optim_options = opt,
                                marginal = false, show_table = false, verbose = false)
        p1 = runp(nothing)
        S.sfmodel_opt(main_solver = NelderMead(), main_opt = (iterations = 3, g_abstol = 0.1))
        p2 = runp(nothing)
        @test S._dicOPT[:main_maxIT] == 200                # the unified default
        @test co(p2) == co(p1)
    end

    # ---- predict reproduces a Float32 fit's cached indices
    @testset "predict reproduces a Float32 fit's cached indices" begin
        Random.seed!(2026082104)
        N32 = 60
        X32 = Float32.(hcat(ones(N32), randn(N32), randn(N32)))
        y32 = Float32.(X32 * Float32[1.0, 0.45, -0.2] .- abs.(0.45f0 .* randn(Float32, N32)) .+
                       0.25f0 .* randn(Float32, N32))
        D   = 64
        d32 = SFrontiers.MSLE_Backend.make_halton_p(D; T = Float32)
        d64 = Float64.(d32)
        # four legal Float64 draws that all narrow to prevfloat(1f0): the fit and predict must
        # agree on which value they use, or erfinv is evaluated at two different points
        pf, ulp = Float64(prevfloat(1.0f0)), Float64(eps(1.0f0))
        d64[end-3:end] .= pf .+ ulp .* (0.05, 0.10, 0.20, 0.24)
        @test all(Float32.(d64[end-3:end]) .== prevfloat(1.0f0))

        sp32 = sfmodel_spec(depvar = y32, frontier = X32, noise = :Normal, ineff = :HalfNormal)
        p0   = sfmodel_init(spec = sp32, ln_sigma_u_sq = log(0.45^2),
                            ln_sigma_v_sq = log(0.25^2), message = false)
        op   = sfmodel_opt(warmstart_solver = NelderMead(),
                           warmstart_opt = (iterations = 100, g_abstol = 1e-4),
                           main_solver = Newton(), main_opt = (iterations = 100, g_abstol = 1e-7))
        r32 = sfmodel_fit(spec = sp32, method = sfmodel_method(method = :MSLE, draws = d64),
                          init = p0, optim_options = op,
                          marginal = false, show_table = false, verbose = false)
        pj = predict(r32; frontier = X32, depvar = y32, what = :jlms, draws = d64)
        pb = predict(r32; frontier = X32, depvar = y32, what = :bc,   draws = d64)
        # one Float32 rounding unit of the largest index, which is the documented bound
        tol = 4 * eps(Float32) * max(maximum(abs, r32.jlms), 1.0)
        @test maximum(abs, pj .- r32.jlms) < tol
        @test maximum(abs, pb .- r32.bc)   < tol
        # narrowing the draws at the call site must make no difference any more
        @test predict(r32; frontier = X32, depvar = y32, what = :jlms,
                      draws = Float32.(d64)) == pj

        # a Float64 fit still round-trips bit for bit
        sp64 = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal)
        r64  = sfmodel_fit(spec = sp64, method = MSLE, init = nothing, optim_options = opt1,
                           marginal = false, show_table = false, verbose = false)
        @test predict(r64; frontier = X, depvar = y, what = :jlms) == r64.jlms
    end

    # ---- a starting-value block of the wrong width is refused by name
    @testset "a starting-value block of the wrong width is refused by name" begin
        spc = sfmodel_spec(datatype = :panel_TFE_CSW, depvar = yp, frontier = XPn, id = idp,
                           noise = :Normal, ineff = :HalfNormal)
        # offsetting widths: the total is right, so only a per-block check can catch it
        @test threw(() -> sfmodel_init(spec = spc, frontier = [0.41],
                                       ln_sigma_u_sq = [-0.71, -0.72], ln_sigma_v_sq = -0.91))
        @test threw(() -> sfmodel_init(spec = spc, ln_sigma_u_sq = [-0.71, -0.72]))
        # a full vector of the wrong length is refused by the builder, not left to the fit
        @test threw(() -> sfmodel_init(spec = spc, init = [1.0, 2.0]))
        # the right widths still work
        v = sfmodel_init(spec = spc, frontier = [0.41, 0.42],
                         ln_sigma_u_sq = -0.71, ln_sigma_v_sq = -0.91)
        @test v == [0.41, 0.42, -0.71, -0.91]

        spt = sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = XPn, zvar = ZPn,
                           id = idp, noise = :Normal, ineff = :HalfNormal)
        @test threw(() -> sfmodel_init(spec = spt, frontier = [0.41]))       # was broadcast
        @test threw(() -> sfmodel_init(spec = spt, scaling = [0.21, 0.22]))  # was truncated
        @test threw(() -> sfmodel_init(spec = spt, ln_sigma_u_sq = [-0.7, -0.8]))
        # a documented scalar for a one-wide block is accepted (was `MethodError: vec`)
        vt = sfmodel_init(spec = spt, scaling = 0.7)
        @test length(vt) == 5 && vt[3] == 0.7
    end

    # ---- the default frontier start does not depend on the element type
    @testset "the default frontier start does not depend on the element type" begin
        X32 = Float32.(X); y32 = Float32.(y)
        sp32 = sfmodel_spec(depvar = y32, frontier = X32, noise = :Normal, ineff = :HalfNormal)
        v32  = sfmodel_init(spec = sp32, message = false)
        @test v32 == start_of(sp32, MLE)
        @test v32 == start_of(sp32, MCI)
        @test v32 == start_of(sp32, MSLE)
        @test v32[1:3] == Float64.(X32) \ Float64.(y32)

        sp32p = sfmodel_spec(datatype = :panel_TFE, depvar = Float32.(yp),
                             frontier = Float32.(XPn), zvar = Float32.(ZPn), id = idp,
                             noise = :Normal, ineff = :HalfNormal)
        @test sfmodel_init(spec = sp32p) == start_of(sp32p, MLE)
        @test sfmodel_init(spec = sp32p) == start_of(sp32p, MCI)
    end

    # ---- panel_TFE truncated normal starts mu at 0.0 under every method
    @testset "panel_TFE truncated normal starts mu at 0.0 under every method" begin
        sptfe = sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = XPn, zvar = ZPn,
                             id = idp, noise = :Normal, ineff = :TruncatedNormal)
        v = sfmodel_init(spec = sptfe)
        @test v[sptfe.panel_spec.idx.mu] == 0.0
        # the analytic backend's `main_ini` is published in the panel order, like `coeff`
        # (not in the backend's own order, which has μ before δ)
        @test v == start_of(sptfe, MLE)
        @test start_of(sptfe, MLE) == start_of(sptfe, MLE, init = v)

        # panel_TRE keeps the flat 0.1, which is what its builder and its fit both use
        XPc = hcat(ones(np), randn(np))
        ypc = XPc * [0.6, -0.4] .- abs.(0.4 .* randn(np)) .+ 0.3 .* randn(np)
        sptre = sfmodel_spec(datatype = :panel_TRE, depvar = ypc, frontier = XPc, id = idp,
                             noise = :Normal, ineff = :TruncatedNormal)
        vtre = sfmodel_init(spec = sptre)
        @test all(vtre[3:end] .== 0.1)
        @test vtre == start_of(sptre, MLE)
    end

    # ---- print_table does not couple coef's element type to vcov's
    @testset "print_table does not couple coef's element type to vcov's" begin
        varn = ["_cons", "x1", "_cons", "_cons"]
        eqn  = ["frontier", "ln_σᵤ²", "ln_σᵥ²"]
        eqi  = [1, 3, 4]
        for B in (SFrontiers.MCI_Backend, SFrontiers.MSLE_Backend)
            for (ct, vt) in ((Float64, Float64), (Float32, Float64),
                             (Float64, Float32), (Float32, Float32))
                out = B.print_table(zeros(ct, 4), Matrix{vt}(I(4)), varn, eqn, eqi;
                                    nobs = 50, noise = :Normal, ineff = :HalfNormal,
                                    K = 2, L = 1, table_format = :text)
                @test out.table isa AbstractMatrix
            end
        end
    end

    # ---- the rename reaches the displayed names too
    @testset "the rename reaches the displayed names too" begin
        # (a) a DSL determinant name must survive into the marginal-effect columns under
        #     every method, not only under the analytic one
        df = DataFrame(c = ones(n), x = X[:, 2], named_determinant = X[:, 3])
        df.y = y
        opm = sfmodel_opt(warmstart_solver = NelderMead(),
                          warmstart_opt = (iterations = 20, g_abstol = 1e-3),
                          main_solver = Newton(), main_opt = (iterations = 40, g_abstol = 1e-5))
        for (dist, het) in ((:HalfNormal, [:sigma_u_sq]), (:Exponential, [:sigma_u_sq]))
            spd = sfmodel_spec(@useData(df), @depvar(y), @frontier(c, x),
                               @zvar(c, named_determinant);
                               noise = :Normal, ineff = dist, hetero = het)
            for m in (MCI, MSLE, MLE)
                r = sfmodel_fit(spec = spd, method = m, init = nothing, optim_options = opm,
                                marginal = true, show_table = false, verbose = false)
                @test names(r.marginal) == ["marg_named_determinant"]
                @test haskey(r.marginal_mean, :named_determinant)
            end
        end

        # (b) panel Lognormal and Rayleigh own a sigma_u variance like HalfNormal does
        for d in (:HalfNormal, :TruncatedNormal, :Lognormal, :Rayleigh)
            spp = sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = XPn, zvar = ZPn,
                               id = idp, noise = :Normal, ineff = d)
            @test "ln_σᵤ²" in spp.panel_spec.varnames
            @test !("ln_σ²" in spp.panel_spec.varnames)
        end
    end

    # ---- sfmodel_init writes no module starting-value state
    @testset "sfmodel_init writes no module starting-value state" begin
        S = SFrontiers.MLE_Backend.SFmle
        sentinel = Dict{Symbol, Any}(:sentinel => [42.0])
        sptre = sfmodel_spec(datatype = :panel_TRE, depvar = yp, frontier = XPn, id = idp,
                             noise = :Normal, ineff = :TruncatedNormal)

        S._dicINI = deepcopy(sentinel)
        sfmodel_init(spec = sptre, ln_sigma_u_sq = -7.777)
        @test S._dicINI == sentinel

        # and on the throwing paths too: `sfmodel_init` clears the global before it validates
        S._dicINI = deepcopy(sentinel)
        @test threw(() -> sfmodel_init(spec = sptre, sigma_a_sq = 0.5, hscale = 0.5))
        @test S._dicINI == sentinel
    end
end
