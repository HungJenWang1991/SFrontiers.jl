# test_intake_guards.jl — user-input shape/type intake guards.
# The panel `multiRand=true` + 1-D `draws` MethodError (test_erlang.jl) turned out to be one
# instance of a class: a public keyword whose shape is accepted and then either flattened into
# something else or dropped into a raw Julia error deep in the likelihood. These testsets pin
# the guards added for the other instances found by sweeping every backend.
using Test
using SFrontiers
using Optim: NelderMead, Newton
using Random, Statistics

@testset "Input intake guards" begin
    _MCIB = SFrontiers.MCI_Backend
    _MSLEB = SFrontiers.MSLE_Backend
    _PB = SFrontiers.Panel_Backend

    Random.seed!(9)
    Ng = 60; Dg = 12
    Xg = hcat(ones(Ng), randn(Ng)); Zg = hcat(ones(Ng), randn(Ng))
    yg = Xg * [1.0, 0.5] .+ 0.3 .* randn(Ng) .- abs.(0.4 .* randn(Ng))
    Vg = collect(range(0.01, 0.99, length=Dg))          # shared draws
    Mg = rand(Ng, Dg)                                   # per-observation block

    # --- ONE shape rule for user-supplied draws, in both backends and whatever `multiRand`
    # was passed (supplying `draws` forces multiRand=false, since the generator is unused):
    # a vector / 1×D row / D×1 column is one sequence shared by every observation; an N×D
    # matrix is one row of draws per observation; any other row count cannot be split.
    for (B, meth, asm) in ((_MCIB, :MCI, _MCIB._assemble_MCI_spec),
                           (_MSLEB, :MSLE, _MSLEB._assemble_MSLE_spec))
        sp = B.sfmodel_spec(depvar=yg, frontier=Xg, noise=:Normal, ineff=:Weibull, hetero=Symbol[])
        mk(dr, mr) = asm(sp, B.sfmodel_method(method=meth, draws=dr, multiRand=mr, n_draws=Dg))
        for mr in (true, false)
            ref = mk(Vg, mr).draws_2D
            @test size(ref) == (1, Dg)
            @test mk(reshape(Vg, 1, :), mr).draws_2D == ref
            @test mk(reshape(Vg, :, 1), mr).draws_2D == ref
            @test size(mk(Mg, mr).draws_2D) == (Ng, Dg)
            @test_throws ErrorException mk(rand(3, Dg), mr)
            @test_throws ErrorException mk(rand(Ng + 1, Dg), mr)
            # supplying draws forces multiRand=false; the flag still governs generation
            @test B.sfmodel_method(method=meth, draws=Vg, multiRand=mr).multiRand == false
        end
        @test B.sfmodel_method(method=meth, n_draws=Dg, multiRand=true).multiRand == true
    end

    Random.seed!(44)
    Nf, Tp = 12, 4; NT = Nf * Tp; idp = repeat(1:Nf, inner=Tp)
    Xp = hcat(randn(NT), randn(NT)); zp = randn(NT)
    yp = repeat(randn(Nf), inner=Tp) .+ Xp * [0.6, -0.4] .+ 0.3 .* randn(NT) .-
         exp.(0.3 .* zp) .* repeat(abs.(0.5 .* randn(Nf)) .+ 0.1, inner=Tp)
    spp = sfmodel_spec(depvar=yp, frontier=Xp, zvar=zp, id=idp, noise=:Normal, ineff=:HalfNormal,
                       type=:prod, datatype=:panel_TFE)
    pmk(dr, mr) = _PB._assemble_panel_spec(spp.panel_spec,
        SFrontiers.sfmodel_panel_method(method=:MCI, draws=dr, multiRand=mr, n_draws=Dg))
    for mr in (true, false)
        @test pmk(Vg, mr).draws == Vg
        @test pmk(reshape(Vg, 1, :), mr).draws == Vg
        @test pmk(reshape(Vg, :, 1), mr).draws == Vg
        @test size(pmk(rand(Nf, Dg), mr).draws) == (Nf, Dg)     # N = FIRMS
        @test_throws ErrorException pmk(rand(NT, Dg), mr)        # NT rows is not N firms
        @test_throws ErrorException pmk(rand(3, Dg), mr)
    end

    # --- the [0, 1) check must also see the draws AFTER they are narrowed to the data's element
    # type: a Float64 draw just below 1 becomes exactly 1 in Float32, and the MCI change of variables
    # then has an infinite Jacobian and a NaN likelihood (MSLE's quantile clamp hid it).
    badf = collect(range(0.05, 0.9, length=8)); badf[end] = prevfloat(1.0)
    goodf = collect(range(0.05, 0.9, length=8))
    @test Float32(prevfloat(1.0)) == 1f0            # the narrowing that defeats a pre-conversion check
    for (B, meth, asm) in ((_MCIB, :MCI, _MCIB._assemble_MCI_spec),
                           (_MSLEB, :MSLE, _MSLEB._assemble_MSLE_spec))
        sp32 = B.sfmodel_spec(depvar=Float32.(yg), frontier=Float32.(Xg), noise=:Normal,
                              ineff=:HalfNormal, hetero=Symbol[])
        sp64 = B.sfmodel_spec(depvar=yg, frontier=Xg, noise=:Normal, ineff=:HalfNormal, hetero=Symbol[])
        @test_throws ErrorException asm(sp32, B.sfmodel_method(method=meth, draws=badf))
        @test size(asm(sp32, B.sfmodel_method(method=meth, draws=goodf)).draws_2D) == (1, 8)
        @test size(asm(sp64, B.sfmodel_method(method=meth, draws=badf)).draws_2D) == (1, 8)   # fine in Float64
    end
    let sp32p = sfmodel_spec(depvar=Float32.(yp), frontier=Float32.(Xp), zvar=Float32.(zp), id=idp,
                             noise=:Normal, ineff=:HalfNormal, type=:prod, datatype=:panel_TFE)
        mkp(d) = _PB._assemble_panel_spec(sp32p.panel_spec,
                     SFrontiers.sfmodel_panel_method(method=:MCI, draws=d))
        @test_throws ErrorException mkp(badf)
        @test length(mkp(goodf).draws) == 8
    end

    # --- Float32 data: the Float64-only MLE sub-spec no longer forces UnifiedSpec{Float64}, so the
    # three MLE-capable distributions build like the other eight instead of a raw convert MethodError
    y32 = Float32.(yg); X32 = Float32.(Xg)
    for d in (:HalfNormal, :TruncatedNormal, :Exponential, :Weibull, :Erlang2)
        sp32 = sfmodel_spec(depvar=y32, frontier=X32, noise=:Normal, ineff=d)
        @test sp32 isa SFrontiers.UnifiedSpec{Float32}
    end
    op = sfmodel_opt(warmstart_solver=NelderMead(), warmstart_opt=(iterations=30, g_abstol=1e-3),
                     main_solver=Newton(), main_opt=(iterations=50, g_abstol=1e-6))
    ll = Dict{String,Float64}()
    for (lab, dv, fr) in (("f32", y32, X32), ("f64", yg, Xg))
        sp = sfmodel_spec(depvar=dv, frontier=fr, noise=:Normal, ineff=:HalfNormal)
        ini = sfmodel_init(spec=sp, frontier=Float64.(Xg \ yg), ln_sigma_u_sq=log(0.16),
                           ln_sigma_v_sq=log(0.09), message=false)
        r = sfmodel_fit(spec=sp, method=sfmodel_method(method=:MLE), init=ini, optim_options=op,
                        marginal=false, show_table=false, verbose=false)
        @test r.converged == true
        ll[lab] = loglikelihood(r)
    end
    # MLE runs in Float64 either way (as the manual promises); the two fits differ only by the
    # Float32 rounding of the DATA, so they agree to Float32 data precision, not bitwise
    @test isapprox(ll["f32"], ll["f64"]; rtol=1e-6)

    # --- Float32 data with show_table=true (the DEFAULT) used to die in print_table AFTER the
    # optimisation had finished: its 3-argument forwarding method tied the spec's element type to
    # the coefficient vector's and the variance matrix's, but the optimizer always returns Float64
    # coefficients (sfmodel_MCI_init / sfmodel_MSLE_init and both fitters cast with Float64.(init)),
    # so a Float32 spec raised a raw MethodError and every Float32 MCI or MSLE fit failed by
    # default, for all eleven distributions and at any iteration budget. The panel backend's
    # panel_print_table never tied them and both docstrings already advertised the untied form.
    # Display only: the fitted numbers are unchanged. The sweep calls print_table directly, since
    # the defect was in dispatch and `ineff` only sets a spec field.
    let np_of = isp -> length(isp.varnames),
        eye = (np) -> [i == j ? 0.01 : 0.0 for i in 1:np, j in 1:np],
        is32 = Bool[]                                             # one entry per display call
        for d in (:HalfNormal, :TruncatedNormal, :Exponential, :Weibull, :Lognormal, :Lomax,
                  :Rayleigh, :Gamma, :Erlang2, :Erlang3, :Erlang4)
            for (B, asm, meth) in ((SFrontiers.MCI_Backend, SFrontiers.MCI_Backend._assemble_MCI_spec, :MCI),
                                   (SFrontiers.MSLE_Backend, SFrontiers.MSLE_Backend._assemble_MSLE_spec, :MSLE))
                d === :Gamma && meth === :MSLE && continue        # MCI-only by design
                bsp = B.sfmodel_spec(depvar=y32, frontier=X32, noise=:Normal, ineff=d, hetero=Symbol[])
                isp = asm(bsp, B.sfmodel_method(method=meth, n_draws=32, multiRand=false))
                push!(is32, eltype(isp.depvar) == Float32)
                np = np_of(isp)
                out = redirect_stdout(devnull) do
                    B.print_table(isp, zeros(Float64, np), eye(np))   # was a raw MethodError
                end
                @test out !== nothing
            end
        end
        @test all(is32)                                           # a genuine mixed-type display call
    end
    # end to end on the default path: show_table=true must no longer break a Float32 fit, and the
    # estimation still runs in Float64 (only the data and the draws are single precision)
    op32 = sfmodel_opt(warmstart_solver=NelderMead(), warmstart_opt=(iterations=8, g_abstol=1e-2),
                       main_solver=Newton(), main_opt=(iterations=8, g_abstol=1e-4))
    for (dv, fr, meth, d) in ((y32, X32, :MCI, :HalfNormal), (y32, X32, :MSLE, :Weibull),
                              (yg, Xg, :MCI, :HalfNormal))
        r = redirect_stdout(devnull) do
            sfmodel_fit(spec=sfmodel_spec(depvar=dv, frontier=fr, noise=:Normal, ineff=d),
                        method=sfmodel_method(method=meth, n_draws=64, multiRand=false),
                        optim_options=op32, marginal=false, show_table=true, verbose=false)
        end
        @test isfinite(loglikelihood(r))
        @test eltype(r.coeff) == Float64
    end
    # --- panel predict must check zvar's ROW count: h(z) is built from zvar, so an oversized zvar
    # returned jlms/bc of zvar's length computed from the wrong firms' rows, silently
    rp = sfmodel_fit(spec=spp, method=sfmodel_method(method=:MCI, n_draws=200),
                     init=sfmodel_init(spec=spp, ln_sigma_u_sq=log(0.16), ln_sigma_v_sq=log(0.09)),
                     optim_options=op, marginal=false, show_table=false, verbose=false)
    k = 25:48
    pj = predict(rp; frontier=Xp[k, :], zvar=reshape(zp[k], :, 1), depvar=yp[k], id=idp[k], what=:jlms)
    @test length(pj) == length(k)
    @test maximum(abs.(pj .- rp.jlms[k])) < 1e-10
    @test_throws ErrorException predict(rp; frontier=Xp[k, :], zvar=reshape(zp, :, 1),
                                        depvar=yp[k], id=idp[k], what=:jlms)      # was length 48, wrong rows
    @test_throws ErrorException predict(rp; frontier=Xp, zvar=reshape(zp[1:NT-1], :, 1),
                                        depvar=yp, id=idp, what=:jlms)            # was a raw BoundsError

    # --- a predict row-count error must name the keyword at fault (zvar was reported as depvar)
    sph = sfmodel_spec(depvar=yg, frontier=Xg, zvar=Zg, noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])
    rh = sfmodel_fit(spec=sph, method=sfmodel_method(method=:MCI, n_draws=200),
                     init=sfmodel_init(spec=sph, frontier=Xg \ yg, mu=[0.1, 0.0], ln_sigma_u_sq=log(0.16),
                                       ln_sigma_v_sq=log(0.09), message=false),
                     optim_options=op, marginal=false, show_table=false, verbose=false)
    e_z = try; predict(rh; frontier=Xg, depvar=yg, zvar=Zg[1:30, :], what=:jlms); catch e; sprint(showerror, e); end
    e_y = try; predict(rh; frontier=Xg, depvar=yg[1:30], zvar=Zg, what=:jlms); catch e; sprint(showerror, e); end
    @test occursin("length(zvar)=30", e_z)
    @test occursin("length(depvar)=30", e_y)
    @test length(predict(rh; frontier=Xg, depvar=yg, zvar=Zg, what=:jlms)) == Ng

    # --- sfmodel_spec row/column guards: the co-built MLE spec caught these for 3 of the 11
    # distributions, so the behaviour used to flip with `ineff` and the other 8 died later with a
    # raw DimensionMismatch out of the OLS baseline
    for d in (:HalfNormal, :Weibull, :Erlang2)
        het = d === :Weibull ? [:lambda] : (d === :Erlang2 ? [:theta] : [:sigma_u_sq])
        @test_throws ErrorException sfmodel_spec(depvar=yg, frontier=Xg[1:55, :], noise=:Normal, ineff=d)
        @test_throws ErrorException sfmodel_spec(depvar=yg, frontier=Xg, zvar=Zg[1:55, :],
                                                 noise=:Normal, ineff=d, hetero=het)
        @test_throws ErrorException sfmodel_spec(depvar=hcat(yg, randn(Ng)), frontier=Xg, noise=:Normal, ineff=d)
        @test sfmodel_spec(depvar=reshape(yg, :, 1), frontier=Xg, noise=:Normal, ineff=d).mci_spec.N == Ng
        @test sfmodel_spec(depvar=yg, frontier=Xg, zvar=Zg, noise=:Normal, ineff=d, hetero=het).mci_spec.N == Ng
    end
    @test_throws ErrorException sfmodel_spec(depvar=yg, frontier=Xg, zvar=reshape(randn(55), :, 1),
                                             noise=:Normal, ineff=:HalfNormal, hetero=:scaling)
    @test sfmodel_spec(depvar=yg, frontier=Xg, zvar=reshape(randn(Ng), :, 1), noise=:Normal,
                       ineff=:HalfNormal, hetero=:scaling).mci_spec.N == Ng
end
