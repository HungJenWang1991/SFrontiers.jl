# Regression tests: the predict layer: input checks and capability.
# Kept self-contained (own data) so it can run independently of runtests.jl.

using SFrontiers
using Test
using Random
using Statistics
using Optim
using StatsAPI
using StatsAPI: predict

# Loose NelderMead options — these assertions are about the predict surface (input
# validation, draws round-trip, marginal dispatch), not the quality of the optimum.
_g4opt() = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40,),
                       main_solver = NelderMead(), main_opt = (iterations = 200,))

# ---------------------------------------------------------------------------
# The simulation-based hetero predict branches (`_predict_eff_mci` /
# `_predict_eff_msle`) validated only the zvar row count, then fed the raw matrix
# into `get_ineff_vals`, which reads only the leading `L` columns. An EXTRA column
# was silently accepted (wrong OOS jlms/bc, no error); a MISSING column raised a
# raw `BoundsError`. Both now hit an explicit `size(Zn,2) == L` check.
# ---------------------------------------------------------------------------
@testset "predict rejects a wrong-width hetero zvar (MCI + MSLE)" begin
    Random.seed!(418)
    n  = 150
    Xg = hcat(ones(n), randn(n), randn(n))          # K = 3
    Zg = hcat(ones(n), randn(n))                     # L = 2 (cons + z1)
    yg = Xg * [1.0, 0.5, -0.3] .+ 0.3 .* randn(n) .- abs.(0.5 .* randn(n))
    βo = Xg \ yg

    spec = sfmodel_spec(depvar = yg, frontier = Xg, zvar = Zg, noise = :Normal,
                        ineff = :HalfNormal, hetero = [:sigma_u_sq], type = :prod)
    init = sfmodel_init(spec = spec, frontier = βo,
                        ln_sigma_u_sq = [log(0.25), 0.0], ln_sigma_v_sq = log(0.09))

    for m in (:MCI, :MSLE)
        r = sfmodel_fit(spec = spec,
                        method = sfmodel_method(method = m, n_draws = 100),
                        init = init, optim_options = _g4opt(),
                        marginal = false, show_table = false, verbose = false)

        # Correct-width zvar (L = 2) predicts fine — positive control.
        jl = predict(r; frontier = Xg, depvar = yg, zvar = Zg, what = :jlms)
        @test length(jl) == n && all(isfinite, jl)

        # An EXTRA column was silently accepted before the fix → wrong OOS jlms with
        # NO error. This assertion therefore FAILS pre-fix (nothing is thrown).
        Zwide = hcat(Zg, randn(n))                   # 3 columns
        ew = try
            predict(r; frontier = Xg, depvar = yg, zvar = Zwide, what = :jlms); nothing
        catch e; e end
        @test ew isa ErrorException && occursin("columns", ew.msg)

        # A MISSING column raised a raw BoundsError before; now an informative error.
        Znarrow = reshape(Zg[:, 1], n, 1)            # 1 column
        en = try
            predict(r; frontier = Xg, depvar = yg, zvar = Znarrow, what = :jlms); nothing
        catch e; e end
        @test en isa ErrorException && occursin("columns", en.msg)
    end
end

# ---------------------------------------------------------------------------
# The user-supplied-draws round-trip was asymmetric.
#  (a) multiRand=false + a D×1 matrix: the fit ran (draws are `vec`-flattened) but
#      stored n_draws = size(draws)[end] = 1, so predict could never validate the
#      draws back — the fit was un-predictable. Now n_draws = size(model.draws_2D,2)
#      = D, and predict's `_validate_user_draws` flattens any 1-row-equivalent shape.
#  (b) multiRand=true + a 1-D draws vector: this used to warn and auto-substitute Halton,
#      discarding the user's draws. Now supplying `draws`
#      forces multiRand=false (the generator is not used) and the vector IS consumed, as
#      one sequence shared by every observation; predict uses the stored draws when none
#      are passed and accepts the same draws when they are passed.
# ---------------------------------------------------------------------------
@testset "user-draws metadata round-trips symmetrically (MCI + MSLE)" begin
    Random.seed!(419)
    n  = 120
    Xg = hcat(ones(n), randn(n), randn(n))
    yg = Xg * [1.0, 0.5, -0.3] .+ 0.3 .* randn(n) .- abs.(0.5 .* randn(n))
    # Lognormal → the *simulation* predict path (`_predict_eff_mci`/`_msle`), which
    # actually consumes the draws; a HalfNormal fit would use the closed form and
    # ignore them, so it cannot exercise the draws round-trip. This is the input that
    # failed (Normal + Lognormal).
    spec = sfmodel_spec(depvar = yg, frontier = Xg, noise = :Normal,
                        ineff = :Lognormal, type = :prod)

    D = 200
    draws_col = reshape(rand(D), D, 1)               # D×1 matrix, multiRand=false
    other_col = reshape(rand(D), D, 1)               # a different draw set

    for m in (:MCI, :MSLE)
        # (a) multiRand=false, D×1 custom draws.
        r = sfmodel_fit(spec = spec,
                        method = sfmodel_method(method = m, draws = draws_col,
                                                multiRand = false),
                        optim_options = _g4opt(),
                        marginal = false, show_table = false, verbose = false)
        @test r.n_draws == D                          # was 1 (size(D×1)[end]) pre-fix
        @test r.user_draws_supplied == true
        # The same D×1 object round-trips BIT-for-bit — proof predict consumed exactly
        # these draws (not auto-Halton), reproducing r.jlms. Pre-fix predict errored
        # because the stored n_draws was 1.
        jl = predict(r; frontier = Xg, depvar = yg, draws = draws_col, what = :jlms)
        @test length(jl) == n && all(isfinite, jl)
        @test maximum(abs.(vec(jl) .- vec(r.jlms))) < 1e-8
        # The equivalent 1×D and length-D shapes are accepted identically.
        @test predict(r; frontier = Xg, depvar = yg,
                      draws = permutedims(draws_col), what = :jlms) ≈ vec(jl)
        @test predict(r; frontier = Xg, depvar = yg,
                      draws = vec(draws_col), what = :jlms) ≈ vec(jl)
        # A different draw set yields a different prediction — the draws are genuinely
        # consumed, so a silent draws mix-up cannot pass unnoticed.
        jl_other = predict(r; frontier = Xg, depvar = yg, draws = other_col, what = :jlms)
        @test vec(jl_other) != vec(jl)

        # (b) a 1-D vector handed with multiRand=true: supplying `draws` forces
        # multiRand=false, so the vector is used as one shared sequence (not discarded).
        dvec = rand(150)
        r2 = sfmodel_fit(spec = spec,
                         method = sfmodel_method(method = m, n_draws = 150,
                                                 draws = dvec, multiRand = true),
                         optim_options = _g4opt(),
                         marginal = false, show_table = false, verbose = false)
        @test r2.user_draws_supplied == true
        @test r2.multiRand == false                   # forced: the generator was not used
        @test r2.n_draws == 150
        # predict uses the stored draws when none are passed and accepts the same draws when
        # they are passed; with them it reproduces the fit exactly.
        jl2 = predict(r2; frontier = Xg, depvar = yg, draws = dvec, what = :jlms)
        @test length(jl2) == n && all(isfinite, jl2)
        @test maximum(abs.(vec(jl2) .- vec(r2.jlms))) < 1e-8
        # the same sequence as a 1×D row or a D×1 column is the same object to predict
        @test predict(r2; frontier = Xg, depvar = yg, draws = reshape(dvec, 1, :), what = :jlms) == jl2
        @test predict(r2; frontier = Xg, depvar = yg, draws = reshape(dvec, :, 1), what = :jlms) == jl2
    end
end

# ---------------------------------------------------------------------------
# predict(:marginal) and predict(:all) hard-errored for the 6 non-HN/TN
# inefficiency distributions (guard in `_predict_marginal`); `result.marginal`
# was already computed for all of them at fit time. The predict surface now
# routes those distributions through the same fit-time `uncondU_*` kernels,
# returning the raw n×L gradient (so `size == size(zvar)`), and `:all` no longer
# crashes on its `:marginal` sub-call.
# ---------------------------------------------------------------------------
@testset "predict(:marginal)/:all works for non-HN/TN distributions" begin
    Random.seed!(408)
    n  = 150
    Xg = hcat(ones(n), randn(n), randn(n))            # K = 3
    Zg = hcat(ones(n), randn(n))                       # L = 2 (cons + z1)
    yg = Xg * [1.0, 0.5, -0.3] .+ 0.3 .* randn(n) .- abs.(0.5 .* randn(n))

    # Exponential (hetero λ), MCI — the input that failed. Pre-fix both
    # predict(:marginal) and predict(:all) threw from the ineff guard.
    spec = sfmodel_spec(depvar = yg, frontier = Xg, zvar = Zg, noise = :Normal,
                        ineff = :Exponential, hetero = [:sigma_u_sq], type = :prod)
    r = sfmodel_fit(spec = spec, method = sfmodel_method(method = :MCI, n_draws = 100),
                    optim_options = _g4opt(), marginal = true,
                    show_table = false, verbose = false)
    m = predict(r; zvar = Zg, what = :marginal)
    @test size(m) == size(Zg)                     # raw n×L gradient contract
    @test all(isfinite, m)
    # predict(:marginal) reuses the fit-time kernel, so its non-constant column
    # equals result.marginal (which drops the constant column Zg[:,1]) bit-for-bit.
    ref = Matrix(r.marginal)
    @test size(ref, 2) == 1
    @test maximum(abs.(m[:, 2] .- ref[:, 1])) < 1e-8
    # predict(:all) previously crashed via its :marginal sub-call.
    allout = predict(r; frontier = Xg, depvar = yg, zvar = Zg, what = :all)
    @test haskey(allout, :marginal) && size(allout.marginal) == size(Zg)
    @test all(isfinite, allout.marginal)

    # Every remaining non-HN/TN distribution now returns a finite n×L matrix
    # (MCI covers all 6; MSLE covers every one except Gamma, which is MCI-only).
    for (m_sym, dists) in ((:MCI,  ((:Rayleigh, [:sigma_u_sq]), (:Lognormal, [:mu]),
                                    (:Weibull, [:lambda]), (:Lomax, [:lambda]),
                                    (:Gamma, [:k]))),
                           (:MSLE, ((:Rayleigh, [:sigma_u_sq]), (:Lognormal, [:mu]),
                                    (:Weibull, [:lambda]), (:Lomax, [:lambda]))))
        for (ineff, het) in dists
            sp = sfmodel_spec(depvar = yg, frontier = Xg, zvar = Zg, noise = :Normal,
                              ineff = ineff, hetero = het, type = :prod)
            rr = sfmodel_fit(spec = sp,
                             method = sfmodel_method(method = m_sym, n_draws = 80),
                             optim_options = _g4opt(), marginal = false,
                             show_table = false, verbose = false)
            mm = predict(rr; zvar = Zg, what = :marginal)
            @test size(mm) == size(Zg) && all(isfinite, mm)
        end
    end

    # Scaling-property form (no-constant scaling Z) also routes through the kernel
    # (via uncondU_scaling), returning an n × L_scaling matrix.
    Zs  = randn(n, 2)
    sps = sfmodel_spec(depvar = yg, frontier = Xg, zvar = Zs, noise = :Normal,
                       ineff = :Exponential, hetero = :scaling, type = :prod)
    rs  = sfmodel_fit(spec = sps, method = sfmodel_method(method = :MCI, n_draws = 80),
                      optim_options = _g4opt(), marginal = false,
                      show_table = false, verbose = false)
    ms  = predict(rs; zvar = Zs, what = :marginal)
    @test size(ms) == (n, 2) && all(isfinite, ms)
end

# ---------------------------------------------------------------------------
# The predict docstring/header claimed only Normal + HN/TN and "Panel
# datatypes are not yet supported", but `_dispatch_jlms_bc` routes all eight
# distributions (+ copulas) and implements the three panel datatypes. Pure doc
# fix: the docstring must no longer carry the stale false claim and must document
# the real capability.
# ---------------------------------------------------------------------------
@testset "predict docstring reflects the actual capability" begin
    # The package's own docstrings of `predict`, read from its documentation metadata (from Julia 1.13 on, an
    # @doc lookup outside an interactive session returns only the first docstring of the function).
    pb = Base.Docs.Binding(StatsAPI, :predict)
    md = Base.Docs.meta(SFrontiers)
    doctext = haskey(md, pb) ? join([join(string.(ds.text)) for ds in values(md[pb].docs)], "\n") : ""
    # The stale false claim is gone (this @test failed before the doc fix).
    @test !occursin("not yet supported", doctext)
    # The real capability is now documented: all distributions and the panels.
    # (Phrasing changed from "all eight" when the Erlang orders joined the roster.)
    @test occursin("all inefficiency distributions", doctext)
    @test occursin(":Erlang2", doctext)
    @test occursin(":panel_TFE", doctext)
    @test occursin(":Exponential", doctext)
end
