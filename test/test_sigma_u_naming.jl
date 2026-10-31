# test_sigma_u_naming.jl — one name for the inefficiency-distribution variance.
#
# The parameter is `ln_sigma_u_sq` under every estimation method and datatype, and the matching
# heteroscedasticity option is `hetero = [:sigma_u_sq]`. The names of versions before 2.0.0
# (`ln_sigma_sq` and `:sigma_sq`; for the Exponential also `ln_lambda` and `:lambda`) are accepted
# with a deprecation warning and reach the same slot. The subscript marks the parameter as
# belonging to the inefficiency distribution rather than the noise distribution; it is not a claim
# that the parameter equals Var(u), which it does not for the half-normal, truncated-normal,
# lognormal or Rayleigh. For the Exponential the squared scale IS Var(u); its old name, `lambda`,
# collided with the Weibull's and Lomax's `lambda` (a scale) in the same backend.
#
# The internal index and result-object keys keep their own spelling and are pinned here so a
# future sweep does not quietly take them along.
using Test
using SFrontiers
using Random
using DataFrames

@testset "one name for the inefficiency variance, and the deprecated names" begin

    Random.seed!(20260822)
    n = 90
    X  = hcat(ones(n), randn(n), randn(n))
    Z  = hcat(ones(n), randn(n))
    Zn = hcat(randn(n))
    y  = X * [1.0, 0.5, -0.3] .- abs.(0.4 .* randn(n)) .+ 0.3 .* randn(n)
    nf, T = 18, 5
    np  = nf * T
    idp = repeat(1:nf, inner=T)
    XPn = hcat(randn(np), randn(np))
    XPc = hcat(ones(np), randn(np))
    ZPn = hcat(randn(np))
    yp  = XPn * [0.6, -0.4] .- abs.(0.4 .* randn(np)) .+ 0.3 .* randn(np)

    threw(f) = try (f(); false) catch; true end

    # ---------------------------------------------------------------- the new name works
    @testset "ln_sigma_u_sq is accepted everywhere" begin
        specs = Dict(
            "cross HalfNormal"      => sfmodel_spec(depvar=y, frontier=X, noise=:Normal, ineff=:HalfNormal),
            "cross TruncatedNormal" => sfmodel_spec(depvar=y, frontier=X, noise=:Normal, ineff=:TruncatedNormal),
            "cross Lognormal"       => sfmodel_spec(depvar=y, frontier=X, noise=:Normal, ineff=:Lognormal),
            "cross Rayleigh"        => sfmodel_spec(depvar=y, frontier=X, noise=:Normal, ineff=:Rayleigh),
            "panel_TFE"             => sfmodel_spec(datatype=:panel_TFE, depvar=yp, frontier=XPn, zvar=ZPn,
                                                    id=idp, noise=:Normal, ineff=:HalfNormal, hetero=:scaling),
            "panel_TFE_CSW"         => sfmodel_spec(datatype=:panel_TFE_CSW, depvar=yp, frontier=XPn,
                                                    id=idp, noise=:Normal, ineff=:HalfNormal),
            "panel_TRE"             => sfmodel_spec(datatype=:panel_TRE, depvar=yp, frontier=XPc,
                                                    id=idp, noise=:Normal, ineff=:HalfNormal),
        )
        # TruncatedNormal and Lognormal also own a `mu`, which the simulation builders require
        needs_mu = ("cross TruncatedNormal", "cross Lognormal")
        for (tag, sp) in specs
            extra = tag in needs_mu ? (mu = 0.0,) : NamedTuple()
            v = nothing
            @test !threw(() -> (v = sfmodel_init(; spec=sp, ln_sigma_u_sq=log(0.25),
                                                 ln_sigma_v_sq=log(0.09), extra...)))
            @test v !== nothing
            # the name of versions before 2.0.0 warns and reaches the same slot
            v_old = @test_logs (:warn, r"`ln_sigma_sq` is deprecated") match_mode=:any sfmodel_init(;
                        spec=sp, ln_sigma_sq=log(0.25), ln_sigma_v_sq=log(0.09), extra...)
            @test v_old == v
            @test_throws "not both" sfmodel_init(; spec=sp, ln_sigma_sq=log(0.25), ln_sigma_u_sq=log(0.25),
                                                 ln_sigma_v_sq=log(0.09), extra...)
        end
    end

    # ------------------------------------------------------- the value lands where it was put
    # Distinguishable values per component, so a slot can be identified by its number.
    @testset "the supplied value reaches the variance slot" begin
        sp = sfmodel_spec(depvar=y, frontier=X, noise=:Normal, ineff=:HalfNormal)
        v  = sfmodel_init(spec=sp, frontier=[101.0, 102.0, 103.0],
                          ln_sigma_u_sq=301.0, ln_sigma_v_sq=401.0)
        @test v isa Vector
        @test v == [101.0, 102.0, 103.0, 301.0, 401.0]
    end

    # ------------------------------------------------------------- the hetero option follows
    @testset "the hetero option is :sigma_u_sq" begin
        sp = sfmodel_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal,
                          ineff=:HalfNormal, hetero=[:sigma_u_sq])
        @test sp.mle_spec !== nothing
        # heteroscedastic: the variance equation now has one coefficient per z column
        v = sfmodel_init(spec=sp, ln_sigma_u_sq=[0.1, 0.2], ln_sigma_v_sq=log(0.09))
        @test length(v) == size(X, 2) + size(Z, 2) + 1

        # the option name of versions before 2.0.0 warns and builds the same model
        sp_old = @test_logs (:warn, r"`hetero = \[:sigma_sq\]` is deprecated") match_mode=:any sfmodel_spec(
                     depvar=y, frontier=X, zvar=Z, noise=:Normal, ineff=:HalfNormal, hetero=[:sigma_sq])
        @test sp_old.hetero == [:sigma_u_sq]
        @test sfmodel_init(spec=sp_old, ln_sigma_u_sq=[0.1, 0.2], ln_sigma_v_sq=log(0.09)) ==
              sfmodel_init(spec=sp, ln_sigma_u_sq=[0.1, 0.2], ln_sigma_v_sq=log(0.09))
        @test_throws "not both" sfmodel_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal,
                                             ineff=:HalfNormal, hetero=[:sigma_sq, :sigma_u_sq])

        # :mu is untouched, and so are the names of the other distributions' parameters
        sp2 = sfmodel_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal,
                           ineff=:TruncatedNormal, hetero=[:mu, :sigma_u_sq])
        @test sp2.mle_spec !== nothing
        # The Exponential uses the same name: its parameter is the squared scale, which for that
        # distribution is also Var(u). `ln_lambda` and `:lambda` are accepted for it as deprecated
        # names, and `lambda` stays the Weibull's and Lomax's name for their SCALE.
        sp3 = sfmodel_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal,
                           ineff=:Exponential, hetero=[:sigma_u_sq])
        @test sp3 !== nothing
        @test !threw(() -> sfmodel_init(spec=sp3, ln_sigma_u_sq=[0.1, 0.2], ln_sigma_v_sq=log(0.09)))
        v3_old = @test_logs (:warn, r"`ln_lambda` is deprecated") match_mode=:any sfmodel_init(
                     spec=sp3, ln_lambda=[0.1, 0.2], ln_sigma_v_sq=log(0.09))
        @test v3_old == sfmodel_init(spec=sp3, ln_sigma_u_sq=[0.1, 0.2], ln_sigma_v_sq=log(0.09))
        sp3_old = @test_logs (:warn, r"`hetero = \[:lambda\]` is deprecated") match_mode=:any sfmodel_spec(
                      depvar=y, frontier=X, zvar=Z, noise=:Normal, ineff=:Exponential, hetero=[:lambda])
        @test sp3_old.hetero == [:sigma_u_sq]
        # the Weibull keeps lambda as the name of its own scale
        spw = sfmodel_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal,
                           ineff=:Weibull, hetero=[:lambda])
        @test spw !== nothing
        @test_logs min_level=Base.CoreLogging.Warn sfmodel_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal,
                                                                ineff=:Weibull, hetero=[:lambda])
    end

    # ------------------------------------------------- the panel rejection rule is preserved
    # The name is rejected for the distributions that own no such parameter, and accepted
    # for the five that do. Only the spelling in that rule has changed.
    @testset "panel rejects the name where the parameter does not exist" begin
        pspec(ineff) = sfmodel_spec(datatype=:panel_TFE, depvar=yp, frontier=XPn, zvar=ZPn,
                                    id=idp, noise=:Normal, ineff=ineff, type=:prod)
        for ineff in (:Weibull, :Lomax, :Gamma)
            @test_throws "ln_sigma_u_sq" sfmodel_init(spec=pspec(ineff), ln_sigma_u_sq=0.5)
        end
        # the Exponential's own parameter is `ln_sigma_u_sq`, so it is in this list
        for ineff in (:HalfNormal, :TruncatedNormal, :Lognormal, :Rayleigh, :Exponential)
            v = sfmodel_init(spec=pspec(ineff), ln_sigma_u_sq=log(0.25))
            @test v isa Vector && all(isfinite, v)
        end
        # a sibling keyword still errors, so the guard is otherwise intact
        @test_throws "ln_lambda" sfmodel_init(spec=pspec(:HalfNormal), ln_lambda=0.5)
        # the Exponential's name of versions before 2.0.0 warns and reaches its slot
        v_pe = @test_logs (:warn, r"`ln_lambda` is deprecated") match_mode=:any sfmodel_init(
                   spec=pspec(:Exponential), ln_lambda=0.5)
        @test v_pe == sfmodel_init(spec=pspec(:Exponential), ln_sigma_u_sq=0.5)
    end

    # ------------------------------------------------------------------ what did NOT change
    # The result object and the internal index keep their own spelling; this pins that so a
    # later sweep does not quietly take them along.
    @testset "the result field and index keys are unchanged" begin
        sp = sfmodel_spec(depvar=y, frontier=X, noise=:Normal, ineff=:HalfNormal)
        r  = sfmodel_fit(spec=sp, method=sfmodel_method(method=:MCI, n_draws=2^7, multiRand=false),
                         init=sfmodel_init(spec=sp, ln_sigma_u_sq=log(0.25), ln_sigma_v_sq=log(0.09)),
                         marginal=false, show_table=false, verbose=false)
        @test hasproperty(r, :sigma_sq)          # the MCI/MSLE result field, deliberately kept
        @test !hasproperty(r, :sigma_u_sq)
    end

    # ------------------------------------------------------ the names of versions before 2.0.0
    # Each old name warns once per session under its own log id, reaches the slot of the new name
    # with the value unchanged, and is refused together with the new name. Data of the sizes the
    # tests use: 200 observations, and 50 firms over 5 periods.
    @testset "deprecated names of versions before 2.0.0" begin
        rng = MersenneTwister(20261010)
        n2  = 200
        X2  = hcat(ones(n2), randn(rng, n2), randn(rng, n2))
        Z2  = hcat(ones(n2), randn(rng, n2))
        y2  = X2 * [1.0, 0.5, -0.3] .- abs.(0.4 .* randn(rng, n2)) .+ 0.3 .* randn(rng, n2)
        nf2, tp2 = 50, 5
        np2  = nf2 * tp2
        idp2 = repeat(1:nf2, inner=tp2)
        XP2  = hcat(randn(rng, np2), randn(rng, np2))
        ZP2  = hcat(randn(rng, np2))
        yp2  = XP2 * [0.6, -0.4] .- abs.(0.4 .* randn(rng, np2)) .+ 0.3 .* randn(rng, np2)
        sp_hn = sfmodel_spec(depvar=y2, frontier=X2, noise=:Normal, ineff=:HalfNormal)
        sp_ex = sfmodel_spec(depvar=y2, frontier=X2, noise=:Normal, ineff=:Exponential)
        sp_w  = sfmodel_spec(depvar=y2, frontier=X2, noise=:Normal, ineff=:Weibull)

        # the exported panel route
        psp = sfmodel_panel_spec(depvar=yp2, frontier=XP2, zvar=ZP2, id=idp2, noise=:Normal,
                                 ineff=:Exponential)
        v_old = @test_logs (:warn, r"`ln_lambda` is deprecated") match_mode=:any sfmodel_panel_init(
                    spec=psp, ln_lambda=log(0.25))
        @test v_old == sfmodel_panel_init(spec=psp, ln_sigma_u_sq=log(0.25))
        psh = sfmodel_panel_spec(depvar=yp2, frontier=XP2, zvar=ZP2, id=idp2, noise=:Normal,
                                 ineff=:HalfNormal)
        v_old = @test_logs (:warn, r"`ln_sigma_sq` is deprecated") match_mode=:any sfmodel_panel_init(
                    spec=psh, ln_sigma_sq=log(0.25))
        @test v_old == sfmodel_panel_init(spec=psh, ln_sigma_u_sq=log(0.25))
        @test_throws "not both" sfmodel_panel_init(spec=psp, ln_lambda=log(0.25), ln_sigma_u_sq=log(0.25))

        # the DSL form
        df = DataFrame(y=y2, cons=ones(n2), x1=X2[:, 2], x2=X2[:, 3], z1=Z2[:, 2])
        sp_dsl = @test_logs (:warn, r"`hetero = \[:sigma_sq\]` is deprecated") match_mode=:any sfmodel_spec(
                     @useData(df), @depvar(y), @frontier(cons, x1, x2), @zvar(cons, z1);
                     noise=:Normal, ineff=:HalfNormal, hetero=[:sigma_sq])
        @test sp_dsl.hetero == [:sigma_u_sq]

        # a value of `nothing` for an old name counts as absent
        v_n = @test_logs min_level=Base.CoreLogging.Warn sfmodel_init(spec=sp_hn, ln_sigma_sq=nothing,
                                                                      ln_sigma_u_sq=log(0.25), ln_sigma_v_sq=log(0.09))
        @test v_n == sfmodel_init(spec=sp_hn, ln_sigma_u_sq=log(0.25), ln_sigma_v_sq=log(0.09))

        # a distribution that does not own the parameter keeps the error that names the old name
        @test_throws "ln_sigma_sq" sfmodel_init(spec=sp_w, ln_sigma_sq=0.1)
        # the Weibull keeps `ln_lambda`, the log of its own scale, without a warning
        @test_logs min_level=Base.CoreLogging.Warn sfmodel_init(spec=sp_w, ln_lambda=log(0.5), ln_k=log(1.5),
                                                                ln_sigma_v_sq=log(0.09))

        # the Exponential's `hetero` names together
        @test_throws "not both" sfmodel_spec(depvar=y2, frontier=X2, zvar=Z2, noise=:Normal,
                                             ineff=:Exponential, hetero=[:lambda, :sigma_u_sq])

        # one log id per old name, so that one warning does not silence another
        logs, _ = Test.collect_test_logs() do
            sfmodel_init(spec=sp_hn, ln_sigma_sq=log(0.25), ln_sigma_v_sq=log(0.09))
            sfmodel_init(spec=sp_hn, ln_sigma_sq=log(0.25), ln_sigma_v_sq=log(0.09))
            sfmodel_init(spec=sp_ex, ln_lambda=log(0.36), ln_sigma_v_sq=log(0.09))
            sfmodel_spec(depvar=y2, frontier=X2, zvar=Z2, noise=:Normal, ineff=:HalfNormal,
                         hetero=[:sigma_sq])
            sfmodel_spec(depvar=y2, frontier=X2, zvar=Z2, noise=:Normal, ineff=:Exponential,
                         hetero=[:lambda])
        end
        for id in (:sf_deprecated_ln_sigma_sq, :sf_deprecated_ln_lambda,
                   :sf_deprecated_hetero_sigma_sq, :sf_deprecated_hetero_lambda)
            @test count(l -> l.id === id, logs) == 1
        end
    end
end
