# test_predict_draw_alignment.jl — predict's draw alignment by position, the draws of new panel firms, and the summation order of the scaling multiplier.
#
# Alignment by position. `predict` pairs fit-time draws with predict-time rows BY POSITION. That is exact for the
# estimation frame in its original order and wrong for anything else — a subset, a reordering, or a
# frame with an extra row — and it is reachable with no `draws=` keyword at all, because multiRand
# defaults to true and therefore gives every observation its own row of draws. The behaviour is
# kept and documented, and predict prints no message about it. These testsets pin the behaviour
# and the silence.
#
# Scaling sum order. `predict` built the scaling multiplier h(z) = exp(z'δ) as one matrix-vector product while
# the fit builds it as a column-by-column sum. Floating-point addition is not associative and BLAS
# sums in a different order, so with four or more scaling regressors predict no longer reproduced
# the fitted jlms/bc exactly. predict now sums in the fit's order.
#
# New panel firms. A panel firm that was not in the fit has no draws of its own, so predict gives the
# new firms the estimation's draws in order of first appearance: the k-th new firm uses the draws of
# estimation firm mod1(k, N_fit), N_fit being the number of estimation firms, cycling through them
# when there are more new firms. A new firm's values therefore change, beyond rounding, only when new
# firms before it are added or removed; predict prints no message about it. Fit firms are matched by
# id and use their own draws.
using Test
using SFrontiers
using Optim: NelderMead, Newton
using Random

@testset "predict draw alignment and scaling reduction order" begin

    # Capture log records emitted by `f` without letting them reach the test output. Uses Test's
    # own collector rather than the Logging stdlib: Pkg.test() runs in a sandbox whose only test
    # dependencies are CSV and Test (Project.toml [targets]), so `using Logging` loads standalone
    # but fails under Pkg.test with "Package Logging not found in current path".
    _logs(f) = first(Test.collect_test_logs(f))
    _silent(f) = isempty(_logs(f))

    op = sfmodel_opt(warmstart_solver=NelderMead(), warmstart_opt=(iterations=15, g_abstol=1e-3),
                     main_solver=Newton(), main_opt=(iterations=20, g_abstol=1e-6))

    # ---- alignment by position -----------------------------------------------------------------------------
    Random.seed!(7)
    N = 60
    X = hcat(ones(N), randn(N))
    y = X * [1.0, 0.5] .- abs.(0.5 .* randn(N)) .+ 0.2 .* randn(N)
    sp = sfmodel_spec(depvar=y, frontier=X, noise=:Normal, ineff=:HalfNormal)
    mkinit() = sfmodel_init(spec=sp, frontier=X \ y, ln_sigma_u_sq=log(0.25),
                            ln_sigma_v_sq=log(0.04), message=false)

    # multiRand=true is the DEFAULT, so this fit has one row of draws per observation and no
    # `draws=` keyword appears anywhere in the call
    r = sfmodel_fit(spec=sp, method=sfmodel_method(method=:MCI, n_draws=256), init=mkinit(),
                    optim_options=op, marginal=false, show_table=false, verbose=false)
    @test r.multiRand == true
    @test r.user_draws_supplied == false

    # the full estimation frame, in its original order, reproduces the fit exactly (checked in
    # test_predict_sim_identity.jl) and predict says nothing — this is the case the package guarantees
    pf = predict(r; frontier=X, depvar=y, what=:jlms)
    @test _silent(() -> predict(r; frontier=X, depvar=y, what=:jlms))

    # a subset is matched by position, so rows 31:60 receive training rows 1:30's draws. The values
    # differ from the full-frame answer for the same rows, and predict prints no message.
    ps = predict(r; frontier=X[31:60, :], depvar=y[31:60], what=:jlms)
    @test length(ps) == 30
    @test maximum(abs.(ps .- pf[31:60])) > 1e-4          # measurably different, not roundoff
    @test _silent(() -> predict(r; frontier=X[31:60, :], depvar=y[31:60], what=:jlms))

    # and the position rule is exactly what happened: the subset equals what a frame consisting of
    # rows 31:60 followed by rows 1:30 returns in its first 30 positions
    perm = vcat(31:60, 1:30)
    pp = predict(r; frontier=X[perm, :], depvar=y[perm], what=:jlms)
    @test maximum(abs.(ps .- pp[1:30])) == 0.0

    # a LEADING prefix is exact, which is what hides the trap from anyone who spot-checks with one
    @test maximum(abs.(predict(r; frontier=X[1:30, :], depvar=y[1:30], what=:jlms) .- pf[1:30])) == 0.0

    # brand-new rows are the same situation, and predict prints no message either
    Xn = hcat(ones(20), randn(20))
    yn = Xn * [1.0, 0.5] .- abs.(0.5 .* randn(20))
    pn = predict(r; frontier=Xn, depvar=yn, what=:jlms)
    @test length(pn) == 20 && all(isfinite, pn)
    @test _silent(() -> predict(r; frontier=Xn, depvar=yn, what=:jlms))

    # with SHARED draws there is nothing to align: every row uses the same sequence, so a subset is
    # exact and no message is emitted
    rs = sfmodel_fit(spec=sp, method=sfmodel_method(method=:MCI, n_draws=256, multiRand=false),
                     init=mkinit(), optim_options=op, marginal=false, show_table=false, verbose=false)
    psf = predict(rs; frontier=X, depvar=y, what=:jlms)
    pss = predict(rs; frontier=X[31:60, :], depvar=y[31:60], what=:jlms)
    @test maximum(abs.(pss .- psf[31:60])) == 0.0
    @test _silent(() -> predict(rs; frontier=X[31:60, :], depvar=y[31:60], what=:jlms))

    # the same holds for MSLE, and for what=:bc
    rm = sfmodel_fit(spec=sp, method=sfmodel_method(method=:MSLE, n_draws=256), init=mkinit(),
                     optim_options=op, marginal=false, show_table=false, verbose=false)
    @test _silent(() -> predict(rm; frontier=X[31:60, :], depvar=y[31:60], what=:bc))

    # ---- new panel firms -----------------------------------------------------------------------------
    # A panel firm that was in the fit is matched by id and uses its own draws; it reproduces its
    # fitted values exactly here, where the fit firms keep the groups of `chunks` they had in the
    # fit. A brand-new firm has no draws of its own: the k-th new firm, in order of first appearance,
    # uses the draws of estimation firm mod1(k, N_fit), and predict prints no message about it.
    Random.seed!(44)
    Nf, Tp = 12, 4
    NT = Nf * Tp
    idp = repeat(1:Nf, inner=Tp)
    Xp = hcat(randn(NT), randn(NT))
    zp = randn(NT)
    yp = repeat(randn(Nf), inner=Tp) .+ Xp * [0.6, -0.4] .+ 0.3 .* randn(NT) .-
         exp.(0.3 .* zp) .* repeat(abs.(0.4 .* randn(Nf)), inner=Tp)
    spp = sfmodel_spec(depvar=yp, frontier=Xp, zvar=zp, id=idp, noise=:Normal, ineff=:HalfNormal,
                       type=:prod, datatype=:panel_TFE)
    rp = sfmodel_fit(spec=spp, method=sfmodel_method(method=:MCI, n_draws=64),
                     init=sfmodel_init(spec=spp, ln_sigma_u_sq=log(0.16), ln_sigma_v_sq=log(0.09)),
                     optim_options=op, marginal=false, show_table=false, verbose=false)

    # fit firms only: exact, and silent
    ppf = predict(rp; frontier=Xp, zvar=reshape(zp, :, 1), depvar=yp, id=idp, what=:jlms)
    @test maximum(abs.(ppf .- rp.jlms)) == 0.0
    @test _silent(() -> predict(rp; frontier=Xp, zvar=reshape(zp, :, 1), depvar=yp,
                                id=idp, what=:jlms))

    # add one brand-new firm, then a second: the fit firms never move, the first new firm keeps the
    # draws of estimation firm 1 when the second joins, and predict prints no message about them
    # Firm 101 must have IDENTICAL data in both frames, so that the comparison below is about the
    # draws, not the data. Drawing the new firms' data inside a helper called twice does NOT achieve
    # that: the two calls consume different numbers of RNG values.
    Random.seed!(99)
    Xn1 = randn(Tp, 2); zn1 = randn(Tp); yn1 = randn(Tp)     # firm 101, used by BOTH frames
    Xn2 = randn(Tp, 2); zn2 = randn(Tp); yn2 = randn(Tp)     # firm 102, frame B only
    XA = vcat(Xp, Xn1);      zA = vcat(zp, zn1);      yA = vcat(yp, yn1)
    idA = vcat(idp, fill(101, Tp))
    XB = vcat(Xp, Xn1, Xn2); zB = vcat(zp, zn1, zn2); yB = vcat(yp, yn1, yn2)
    idB = vcat(idp, fill(101, Tp), fill(102, Tp))
    @test XA[NT+1:NT+Tp, :] == XB[NT+1:NT+Tp, :]             # same data in both frames
    @test zA[NT+1:NT+Tp] == zB[NT+1:NT+Tp]
    @test yA[NT+1:NT+Tp] == yB[NT+1:NT+Tp]
    pA = predict(rp; frontier=XA, zvar=reshape(zA, :, 1), depvar=yA, id=idA, what=:jlms)
    pB = predict(rp; frontier=XB, zvar=reshape(zB, :, 1), depvar=yB, id=idB, what=:jlms)
    @test _silent(() -> predict(rp; frontier=XA, zvar=reshape(zA, :, 1), depvar=yA,
                                id=idA, what=:jlms))
    @test maximum(abs.(pA[1:NT] .- rp.jlms)) == 0.0        # fit firms exact in frame A
    @test maximum(abs.(pB[1:NT] .- rp.jlms)) == 0.0        # and in frame B
    # Firm 101 is the first new firm in both frames, so it uses the draws of estimation firm 1 in
    # both; the two frames form different groups of `chunks` (firm 101 is alone in its group in
    # frame A), so the values agree to rounding.
    @test isapprox(pA[NT+1:NT+Tp], pB[NT+1:NT+Tp]; rtol = 1e-13)
    @test all(isfinite, pA) && all(isfinite, pB)

    # Unit-level pins, no fit needed. They show the row rule directly: fit firms keep their own rows
    # by id, and the k-th new firm, in order of first appearance, gets estimation row mod1(k, N_fit).
    let dfit = Float64[i + j/100 for i in 1:3, j in 1:4]
        o1 = SFrontiers._build_predict_panel_draws(dfit, [1, 2, 3], [1, 2, 3, 99])
        o2 = SFrontiers._build_predict_panel_draws(dfit, [1, 2, 3], [1, 2, 3, 99, 100])
        @test o1[1:3, :] == dfit                     # fit firms are matched BY ID, not by position
        @test o2[1:3, :] == dfit
        @test o1[4, :] == dfit[1, :]                 # one new firm: the draws of estimation firm 1
        @test o2[4, :] == dfit[1, :]                 # and it keeps them when a second new firm joins
        @test o2[5, :] == dfit[2, :]                 # the second new firm: estimation firm 2's draws
        # ids need not be integers, and order among fit firms is irrelevant
        o3 = SFrontiers._build_predict_panel_draws(dfit, ["a", "b", "c"], ["c", "a", "zz"])
        @test o3[1, :] == dfit[3, :] && o3[2, :] == dfit[1, :]
        @test o3[3, :] == dfit[1, :] == o1[4, :]     # the one new firm gets estimation firm 1's draws
        # more new firms than estimation firms: the fourth new firm gets row mod1(4, 3) = 1
        o4 = SFrontiers._build_predict_panel_draws(dfit, [1, 2, 3], [91, 92, 93, 94])
        @test o4[4, :] == dfit[1, :]
    end

    # ---- _align_draws directly, including the recycle branch no fit above reaches ------------
    let dtr = Float64[10i + j for i in 1:5, j in 1:3]
        @test SFrontiers._align_draws(dtr, 5) === dtr                       # equal: passed through
        @test SFrontiers._align_draws(dtr, 3)[:, 1] == [11.0, 21.0, 31.0]   # fewer: leading rows
        @test SFrontiers._align_draws(dtr, 7)[:, 1] ==
              [11.0, 21.0, 31.0, 41.0, 51.0, 11.0, 21.0]                    # more: recycled by mod1
        @test size(SFrontiers._align_draws(dtr, 7)) == (7, 3)
        @test _silent(() -> SFrontiers._align_draws(dtr, 7))                # no message for any row count
    end

    # ---- scaling sum order ----------------------------------------------------------------------------
    # predict must reproduce the fitted jlms/bc bit for bit at EVERY number of scaling regressors.
    # Before the fix, L_scaling ≤ 3 agreed and 4 and above did not (8.9e-16 at 4, 2.2e-15 at 6).
    Random.seed!(20)
    Ns = 150
    for L_s in (2, 3, 4, 6, 12)
        Xs = hcat(ones(Ns), randn(Ns), randn(Ns))
        Zs = randn(Ns, L_s)                       # a scaling zvar must not contain a constant column
        h = exp.(Zs * (0.15 .* randn(L_s)))
        ys = Xs * [1.0, 0.5, -0.3] .- h .* abs.(0.4 .* randn(Ns)) .+ 0.3 .* randn(Ns)
        sps = sfmodel_spec(depvar=ys, frontier=Xs, zvar=Zs, noise=:Normal, ineff=:TruncatedNormal,
                           hetero=:scaling)
        rsc = sfmodel_fit(spec=sps, method=sfmodel_method(method=:MCI, n_draws=64, multiRand=false),
                          init=sfmodel_init(spec=sps, frontier=Xs \ ys, scaling=zeros(L_s), mu=0.0,
                                            ln_sigma_u_sq=log(0.16), ln_sigma_v_sq=log(0.09),
                                            message=false),
                          optim_options=op, marginal=false, show_table=false, verbose=false)
        @test maximum(abs.(predict(rsc; frontier=Xs, zvar=Zs, depvar=ys, what=:jlms) .- rsc.jlms)) == 0.0
        @test maximum(abs.(predict(rsc; frontier=Xs, zvar=Zs, depvar=ys, what=:bc) .- rsc.bc)) == 0.0
    end
end
