# Regression tests for the 2026-07 predict re-review.
#
# Change: cross-sectional MCI / MSLE predict(:jlms/:bc/:response/:all) now reuses the
# fit-time SIMULATION kernel for EVERY inefficiency distribution — including HalfNormal
# and TruncatedNormal, which previously short-circuited to an analytic closed form. The
# fit populates r.jlms / r.bc by simulating E(u|ε) over Halton draws, so predict on the
# estimation data must reproduce those values BIT-FOR-BIT by reconstructing the identical
# draws (or accepting them via `draws=`). The analytic MLE backend keeps its closed form.
#
# These would FAIL before the change: HN/TN MCI/MSLE predict used the closed form, which
# disagreed with the simulated r.jlms by the Monte-Carlo simulation error (~1e-2 here).

using SFrontiers
using Test
using Random
using Optim
using StatsAPI: predict

# Loose NelderMead — bit-identity compares predict against r.jlms at the SAME fitted
# coefficients, so the quality of the optimum is irrelevant; we only need a finite fit.
_psi_opt() = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 60,),
                         main_solver = NelderMead(), main_opt = (iterations = 400,))

@testset "predict sim-kernel identity (2026-07 predict re-review)" begin

    Random.seed!(70714)
    n  = 140
    X  = hcat(ones(n), randn(n), randn(n))
    Z  = hcat(ones(n), randn(n))                      # [_cons, z1]
    βo = X \ (X * [1.0, 0.5, -0.3])
    y  = X * [1.0, 0.5, -0.3] .+ 0.3 .* randn(n) .- abs.(0.5 .* randn(n))

    # ----------------------------------------------------------------------
    # (1) HN / TN via MCI & MSLE, multiRand true & false: predict on the
    #     estimation data reproduces r.jlms / r.bc EXACTLY (== , max|Δ| = 0).
    # ----------------------------------------------------------------------
    @testset "HN/TN MCI/MSLE predict==fit on training data (bit-identical)" begin
        # HalfNormal, homoscedastic (no zvar).
        sp_hn = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                             ineff = :HalfNormal, type = :prod)
        ini_hn = sfmodel_init(spec = sp_hn, frontier = βo,
                              ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
        # TruncatedNormal, μ heteroscedastic over [_cons, z1].
        sp_tn = sfmodel_spec(depvar = y, frontier = X, zvar = Z, noise = :Normal,
                             ineff = :TruncatedNormal, hetero = [:mu], type = :prod)
        ini_tn = sfmodel_init(spec = sp_tn, frontier = βo, mu = [0.0, 0.0],
                              ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))

        for backend in (:MCI, :MSLE), mr in (true, false)
            for (tag, sp, ini, zk) in (("HalfNormal", sp_hn, ini_hn, nothing),
                                       ("TruncNormal", sp_tn, ini_tn, Z))
                r = sfmodel_fit(spec = sp,
                                method = sfmodel_method(method = backend, n_draws = 150,
                                                        multiRand = mr),
                                init = ini, optim_options = _psi_opt(),
                                marginal = false, show_table = false, verbose = false)

                # Sanity: fit really produced simulated indices.
                @test all(isfinite, vec(r.jlms)) && all(isfinite, vec(r.bc))

                jl = zk === nothing ? predict(r; frontier = X, depvar = y, what = :jlms) :
                                      predict(r; frontier = X, depvar = y, zvar = zk, what = :jlms)
                bc = zk === nothing ? predict(r; frontier = X, depvar = y, what = :bc) :
                                      predict(r; frontier = X, depvar = y, zvar = zk, what = :bc)

                # BIT-IDENTICAL — the whole point of the change. (`==`, not `≈`.)
                @test vec(jl) == vec(r.jlms)
                @test vec(bc) == vec(r.bc)
                # Belt-and-suspenders: exact zero max deviation, tagged for a readable failure.
                @test maximum(abs.(vec(jl) .- vec(r.jlms))) == 0.0  # $backend $tag multiRand=$mr jlms
                @test maximum(abs.(vec(bc) .- vec(r.bc)))   == 0.0  # $backend $tag multiRand=$mr bc
            end
        end
    end

    # ----------------------------------------------------------------------
    # (2) `:response` and `:all` inherit the same identity (they call :jlms/:bc
    #     internally), for an HN sim-path fit.
    # ----------------------------------------------------------------------
    @testset "HN MCI :response/:all reuse the sim-path jlms/bc" begin
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                          ineff = :HalfNormal, type = :prod)
        ini = sfmodel_init(spec = sp, frontier = βo,
                           ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
        r = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MCI, n_draws = 150),
                        init = ini, optim_options = _psi_opt(),
                        marginal = false, show_table = false, verbose = false)
        ŷ  = predict(r; frontier = X, what = :frontier)
        û  = predict(r; frontier = X, depvar = y, what = :jlms)
        resp = predict(r; frontier = X, depvar = y, what = :response)
        sgn = SFrontiers._frontier_sign(r)
        @test resp ≈ ŷ .- sgn .* û
        allout = predict(r; frontier = X, depvar = y, what = :all)
        @test vec(allout.jlms) == vec(û) && vec(allout.jlms) == vec(r.jlms)
        @test vec(allout.bc)   == vec(r.bc)
    end

    # ----------------------------------------------------------------------
    # (3) `draws=` round-trips for an HN fit (caveat #3). Because HN now uses the
    #     simulation path, a HN fit run with user-supplied `draws=` (a) reports
    #     user_draws_supplied=true, (b) reproduces r.jlms bit-for-bit ONLY when the
    #     same draws are passed back, and (c) errors when they are omitted (they are
    #     not stored on the result). Pre-change, HN ignored draws entirely.
    # ----------------------------------------------------------------------
    @testset "draws= round-trips for a HalfNormal fit" begin
        D = 200
        draws_col  = reshape(rand(D), D, 1)              # D×1, multiRand=false
        other_col  = reshape(rand(D), D, 1)
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                          ineff = :HalfNormal, type = :prod)
        ini = sfmodel_init(spec = sp, frontier = βo,
                           ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
        for backend in (:MCI, :MSLE)
            r = sfmodel_fit(spec = sp,
                            method = sfmodel_method(method = backend, draws = draws_col,
                                                    multiRand = false),
                            init = ini, optim_options = _psi_opt(),
                            marginal = false, show_table = false, verbose = false)
            @test r.n_draws == D
            @test r.user_draws_supplied == true
            # Same draws back ⇒ bit-identical to the fit's own jlms.
            jl = predict(r; frontier = X, depvar = y, draws = draws_col, what = :jlms)
            @test vec(jl) == vec(r.jlms)
            # Equivalent flat shapes accepted identically.
            @test predict(r; frontier = X, depvar = y, draws = vec(draws_col), what = :jlms) == vec(jl)
            # A different draw set ⇒ a different prediction (draws are genuinely consumed).
            @test predict(r; frontier = X, depvar = y, draws = other_col, what = :jlms) != vec(jl)
            # Omitting draws= for a user-draws fit must error (draws not stored).
            @test_throws Exception predict(r; frontier = X, depvar = y, what = :jlms)
        end
    end

    # ----------------------------------------------------------------------
    # (4) The analytic MLE backend is UNCHANGED: Normal + HN/TN still use the
    #     closed form, which reproduces the fit's own closed-form r.jlms/r.bc.
    # ----------------------------------------------------------------------
    @testset "MLE HN/TN still closed-form (unchanged)" begin
        for (ineff, hetero, zk, ini_extra) in (
                (:HalfNormal, Symbol[], nothing, (ln_sigma_sq = log(0.25),)),
                (:TruncatedNormal, [:mu], Z, (mu = [0.0, 0.0], ln_sigma_sq = log(0.25))))
            sp = zk === nothing ?
                 sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = ineff, type = :prod) :
                 sfmodel_spec(depvar = y, frontier = X, zvar = zk, noise = :Normal,
                              ineff = ineff, hetero = hetero, type = :prod)
            ini = sfmodel_init(; spec = sp, frontier = βo, ln_sigma_v_sq = log(0.09), ini_extra...)
            r = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE),
                            init = ini, optim_options = _psi_opt(),
                            marginal = false, show_table = false, verbose = false)
            @test r.backend === :MLE
            jl = zk === nothing ? predict(r; frontier = X, depvar = y, what = :jlms) :
                                  predict(r; frontier = X, depvar = y, zvar = zk, what = :jlms)
            # Closed form on both sides ⇒ machine-precision agreement (no draws involved).
            @test maximum(abs.(vec(jl) .- vec(r.jlms))) < 1e-8
        end
    end
end
