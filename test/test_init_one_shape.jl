# test_init_one_shape.jl — `sfmodel_init`: one shape, optional components.
#
# (1) Every starting-value component is optional. A component that is not supplied takes
# exactly the value the fit itself would have used had no starting values been supplied at
# all — the OLS estimate for the frontier and 0.1 for every other parameter — so that
# omitting one component and omitting the whole starting-value object agree component by
# component. Before this change the two cross-sectional simulation builders refused a
# partial set with "Missing required init(s)".
#
# (2) `sfmodel_init` returns a `Vector{Float64}` for every datatype and every estimation
# method, and its `method` argument is gone. Before this change `method=:MLE` and the two
# MLE-only panel datatypes returned a component-keyed `Dict`, and on a scaling model the
# same typed vector meant two different things depending on `method=`.
#
# Also pinned here: `zvar` is refused by the two MLE-only panel models, which used to accept
# it and drop it silently; `hscale` is refused by them, having previously set σₐ² on
# panel_TRE; and the guards that must keep firing — wrong length, an inapplicable component,
# and the panel rule that refuses a keyword the distribution does not own.
using Test
using SFrontiers
using Random
using Optim

@testset "sfmodel_init: one shape, optional components" begin

    Random.seed!(20260822)
    n  = 140
    X  = hcat(ones(n), randn(n), randn(n))
    Z  = hcat(ones(n), randn(n))
    Zs = hcat(randn(n))                       # scaling z: no constant
    y  = X * [1.0, 0.5, -0.3] .- abs.(0.4 .* randn(n)) .+ 0.3 .* randn(n)

    nf, T = 24, 5
    np  = nf * T
    idp = repeat(1:nf, inner = T)
    XPn = hcat(randn(np), randn(np))          # no constant: TFE / TFE_CSW demean it away
    XPc = hcat(ones(np), randn(np))           # constant allowed for TRE
    ZPn = hcat(randn(np))
    yp  = XPn * [0.6, -0.4] .- abs.(0.4 .* randn(np)) .+ 0.3 .* randn(np)
    ypc = XPc * [0.6, -0.4] .- abs.(0.4 .* randn(np)) .+ 0.3 .* randn(np)

    threw(f) = try (f(); false) catch; true end

    # One iteration of a derivative-free solver, so `main_ini` is the untouched starting
    # vector and each fit is cheap. Nothing here asserts on an estimate.
    opt1 = sfmodel_opt(main_solver = NelderMead(), main_opt = (iterations = 1, g_abstol = 1e-8))
    MLE  = sfmodel_method(method = :MLE)
    MCI  = sfmodel_method(method = :MCI,  n_draws = 64)
    MSLE = sfmodel_method(method = :MSLE, n_draws = 64)

    start_of(sp, m; init = nothing) =
        sfmodel_fit(spec = sp, method = m, init = init, optim_options = opt1,
                    jlms_bc_index = false, marginal = false,
                    show_table = false, verbose = false).main_ini

    # ------------------------------------------------------------------ (2) the shape
    @testset "every route returns a Vector" begin
        specs = [
            ("cross HalfNormal",      sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                                                   ineff = :HalfNormal)),
            ("cross TruncatedNormal", sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                                                   ineff = :TruncatedNormal)),
            ("cross Exponential",     sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                                                   ineff = :Exponential)),
            ("cross TN scaling",      sfmodel_spec(depvar = y, frontier = X, zvar = Zs,
                                                   noise = :Normal, ineff = :TruncatedNormal,
                                                   hetero = :scaling)),
            ("cross hetero sigma",    sfmodel_spec(depvar = y, frontier = X, zvar = Z,
                                                   noise = :Normal, ineff = :HalfNormal,
                                                   hetero = [:sigma_u_sq])),
            ("panel_TFE HN",          sfmodel_spec(datatype = :panel_TFE, depvar = yp,
                                                   frontier = XPn, zvar = ZPn, id = idp,
                                                   noise = :Normal, ineff = :HalfNormal)),
            ("panel_TFE_CSW HN",      sfmodel_spec(datatype = :panel_TFE_CSW, depvar = yp,
                                                   frontier = XPn, id = idp,
                                                   noise = :Normal, ineff = :HalfNormal)),
            ("panel_TRE HN",          sfmodel_spec(datatype = :panel_TRE, depvar = ypc,
                                                   frontier = XPc, id = idp,
                                                   noise = :Normal, ineff = :HalfNormal)),
            ("panel_TRE TN",          sfmodel_spec(datatype = :panel_TRE, depvar = ypc,
                                                   frontier = XPc, id = idp,
                                                   noise = :Normal, ineff = :TruncatedNormal)),
        ]
        for (tag, sp) in specs
            v = sfmodel_init(spec = sp)
            @test v isa Vector{Float64}
            @test !isempty(v)
        end
    end

    @testset "`method` is gone and says so" begin
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal)
        for m in (:MCI, :MSLE, :MLE, nothing)
            @test_throws "no longer takes a `method` argument" sfmodel_init(spec = sp, method = m)
        end
        # the method OBJECT is refused too, not only the Symbol
        @test_throws "no longer takes a `method` argument" sfmodel_init(spec = sp, method = MLE)
    end

    # -------------------------------------------------- (1) omitting equals omitting all
    @testset "an omitted component equals the fit's own default" begin
        # Cross-sectional: the builder's all-omitted vector IS the fit's `init=nothing` start.
        cross = [
            ("HalfNormal",      :HalfNormal,      NamedTuple()),
            ("TruncatedNormal", :TruncatedNormal, NamedTuple()),
            ("Lognormal",       :Lognormal,       NamedTuple()),
            ("Exponential",     :Exponential,     NamedTuple()),
            ("Weibull",         :Weibull,         NamedTuple()),
            ("Lomax",           :Lomax,           NamedTuple()),
        ]
        for (tag, ineff, _) in cross
            sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = ineff)
            v  = sfmodel_init(spec = sp, message = false)
            @test v == start_of(sp, MCI)
            @test v == start_of(sp, MSLE)
            # the non-frontier tail is exactly 0.1, and the head is the OLS estimate
            @test all(v[4:end] .== 0.1)
            @test v[1:3] ≈ X \ y
        end

        # Supplying one component moves that component and nothing else.
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :TruncatedNormal)
        base = sfmodel_init(spec = sp, message = false)
        for (kw, slot) in ((:mu, 4), (:ln_sigma_u_sq, 5), (:ln_sigma_v_sq, 6))
            v = sfmodel_init(; spec = sp, message = false, (kw => 7.25,)...)
            @test v[slot] == 7.25
            @test v[setdiff(1:length(v), slot)] == base[setdiff(1:length(base), slot)]
        end
        vf = sfmodel_init(spec = sp, frontier = [11.0, 12.0, 13.0], message = false)
        @test vf == vcat([11.0, 12.0, 13.0], base[4:end])

        # Heteroscedastic components take one default per z column, not one in total.
        sph = sfmodel_spec(depvar = y, frontier = X, zvar = Z, noise = :Normal,
                           ineff = :TruncatedNormal, hetero = [:mu, :sigma_u_sq])
        vh = sfmodel_init(spec = sph, message = false)
        @test length(vh) == 3 + 2 + 2 + 1
        @test vh == start_of(sph, MCI)

        # The two MLE-only panel models, and panel_TFE, agree with their own fits too.
        for sp in (sfmodel_spec(datatype = :panel_TFE_CSW, depvar = yp, frontier = XPn,
                                id = idp, noise = :Normal, ineff = :HalfNormal),
                   sfmodel_spec(datatype = :panel_TRE, depvar = ypc, frontier = XPc,
                                id = idp, noise = :Normal, ineff = :HalfNormal),
                   sfmodel_spec(datatype = :panel_TRE, depvar = ypc, frontier = XPc,
                                id = idp, noise = :Normal, ineff = :TruncatedNormal))
            @test sfmodel_init(spec = sp) == start_of(sp, MLE)
        end
        sptfe = sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = XPn, zvar = ZPn,
                             id = idp, noise = :Normal, ineff = :TruncatedNormal)
        @test sfmodel_init(spec = sptfe) == start_of(sptfe, MCI)
    end

    # ------------------------------------------- the MLE-only panel layouts, slot by slot
    @testset "MLE-only panel layouts" begin
        csw = sfmodel_spec(datatype = :panel_TFE_CSW, depvar = yp, frontier = XPn, id = idp,
                           noise = :Normal, ineff = :HalfNormal)
        tre = sfmodel_spec(datatype = :panel_TRE, depvar = ypc, frontier = XPc, id = idp,
                           noise = :Normal, ineff = :HalfNormal)
        tret = sfmodel_spec(datatype = :panel_TRE, depvar = ypc, frontier = XPc, id = idp,
                            noise = :Normal, ineff = :TruncatedNormal)

        v1 = sfmodel_init(spec = csw, frontier = [1.1, 1.2],
                          ln_sigma_u_sq = -3.0, ln_sigma_v_sq = -4.0)
        @test v1 == [1.1, 1.2, -3.0, -4.0]
        @test v1 == start_of(csw, MLE, init = v1)

        v2 = sfmodel_init(spec = tre, frontier = [1.1, 1.2], sigma_a_sq = 0.2,
                          ln_sigma_u_sq = -3.0, ln_sigma_v_sq = -4.0)
        @test v2 == [1.1, 1.2, 0.2, -3.0, -4.0]
        @test v2 == start_of(tre, MLE, init = v2)

        v3 = sfmodel_init(spec = tret, frontier = [1.1, 1.2], mu = 0.15, sigma_a_sq = 0.2,
                          ln_sigma_u_sq = -3.0, ln_sigma_v_sq = -4.0)
        @test v3 == [1.1, 1.2, 0.15, 0.2, -3.0, -4.0]
        @test v3 == start_of(tret, MLE, init = v3)

        # full-vector mode passes straight through
        @test sfmodel_init(spec = tre, init = [9.0, 8.0, 7.0, 6.0, 5.0]) == [9.0, 8.0, 7.0, 6.0, 5.0]

        # components these models do not own are refused by name
        @test_throws "not a parameter of this model" sfmodel_init(spec = csw, mu = 0.3)
        @test_throws "no scaling function" sfmodel_init(spec = tre, hscale = [0.3])
        @test_throws "no scaling function" sfmodel_init(spec = csw, hscale = [0.3])
        @test threw(() -> sfmodel_init(spec = csw, sigma_a_sq = 0.3))
    end

    # --------------------------------------------- one vector, one meaning, on the scaling model
    @testset "one order for the scaling model, whatever fits it" begin
        sp = sfmodel_spec(depvar = y, frontier = X, zvar = Zs, noise = :Normal,
                          ineff = :TruncatedNormal, hetero = :scaling)
        # [frontier(3), scaling(1), mu, ln_sigma_u_sq, ln_sigma_v_sq]
        v = sfmodel_init(spec = sp, frontier = [11.0, 12.0, 13.0], scaling = [21.0],
                         mu = 31.0, ln_sigma_u_sq = 41.0, ln_sigma_v_sq = 51.0,
                         message = false)
        @test v == [11.0, 12.0, 13.0, 21.0, 31.0, 41.0, 51.0]
        # `hscale` is the analytic backend's name for the same coefficients
        @test sfmodel_init(spec = sp, frontier = [11.0, 12.0, 13.0], hscale = [21.0],
                           mu = 31.0, ln_sigma_u_sq = 41.0, ln_sigma_v_sq = 51.0,
                           message = false) == v
        @test_throws "not both" sfmodel_init(spec = sp, scaling = [1.0], hscale = [1.0],
                                             message = false)
        # the analytic fit puts μ before the scaling block; the conversion happens at the
        # fit boundary, so this one vector means the same thing under all three methods,
        # and the result's `main_ini` comes back in the published order, like `coeff`
        # (an earlier form left it in the analytic order [11, 12, 13, 31, 21, 41, 51]).
        @test start_of(sp, MLE,  init = v) == v
        @test start_of(sp, MCI,  init = v) == v
        @test start_of(sp, MSLE, init = v) == v
    end

    # ------------------------------------------------------------- guards that must still fire
    @testset "the guards that must not be lost" begin
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal)
        @test_throws "Length mismatch(es)" sfmodel_init(spec = sp, frontier = [1.0, 2.0],
                                                        message = false)
        @test_throws "expected 5 parameters" sfmodel_init(spec = sp, init = [1.0, 2.0],
                                                          message = false)
        @test_throws "Unused init component(s)" sfmodel_init(spec = sp, mu = 0.0,
                                                             message = false)
        # the message names the keyword the user typed, not an internal struct
        e = try (sfmodel_init(spec = sp, mu = 0.0, message = false); "") catch err
            sprint(showerror, err) end
        @test occursin("mu", e)
        @test !occursin("MuInit", e)

        # the panel rule: a keyword the distribution does not own
        spp = sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = XPn, zvar = ZPn,
                           id = idp, noise = :Normal, ineff = :Exponential)
        @test_throws "ln_k" sfmodel_init(spec = spp, ln_k = -1.0)
        # and the one it does own still works (renamed from ln_lambda)
        @test sfmodel_init(spec = spp, ln_sigma_u_sq = -1.0) isa Vector
        # its name of versions before 2.0.0 warns and reaches the same slot
        v_old = @test_logs (:warn, r"`ln_lambda` is deprecated") match_mode = :any sfmodel_init(
                    spec = spp, ln_lambda = -1.0)
        @test v_old == sfmodel_init(spec = spp, ln_sigma_u_sq = -1.0)
    end

    # ------------------------------------------------- zvar refused by the MLE-only panels
    @testset "the MLE-only panels refuse zvar" begin
        Zp = hcat(ones(np), randn(np))
        for dt in (:panel_TFE_CSW, :panel_TRE)
            fr = dt === :panel_TRE ? XPc : XPn
            dv = dt === :panel_TRE ? ypc : yp
            @test_throws "not a parameter of the" sfmodel_spec(
                datatype = dt, depvar = dv, frontier = fr, zvar = Zp, id = idp,
                noise = :Normal, ineff = :HalfNormal)
        end
        @test_throws "not a parameter of the" sfmodel_spec(
            datatype = :panel_TRE, depvar = ypc, frontier = XPc, zvar = Zp, id = idp,
            noise = :Normal, ineff = :TruncatedNormal)
        # panel_TFE, which does have a scaling function, still takes zvar
        @test sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = XPn, zvar = ZPn,
                           id = idp, noise = :Normal, ineff = :HalfNormal) !== nothing
    end

    # ---------------------------------------------------- the fit accepts what the builder gives
    @testset "the vector fits, under every method the model has" begin
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal)
        v  = sfmodel_init(spec = sp, frontier = X \ y, ln_sigma_u_sq = log(0.25),
                          ln_sigma_v_sq = log(0.09), message = false)
        optN = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 500, g_abstol = 1e-8))
        lls = Float64[]
        for m in (MCI, MSLE, MLE)
            r = sfmodel_fit(spec = sp, method = m, init = v, optim_options = optN,
                            marginal = false, show_table = false, verbose = false)
            @test r.main_ini == v
            push!(lls, r.loglikelihood)
        end
        # the three are different estimators, so they must not be asserted equal; what is
        # asserted is that each accepted the same object and started where it was told.
        @test all(isfinite, lls)
    end
end
