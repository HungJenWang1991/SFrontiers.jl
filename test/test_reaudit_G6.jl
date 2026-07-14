# Regression tests for the 03_reaudit_findings.md Group 6 (G6) fix — the FIX-06 residual and
# R-05 (marginal effects wrong for a homoscedastic parameter over a non-constant Z[:,1]).
#
# Resolution: 04 "Decisions to make first" #1, option (i) — validate at spec time, chosen to
# keep MCI/MSLE consistent with the MLE backend. In MLE a homoscedastic inefficiency parameter
# is a true `_cons` constant (SFmainfun.jl synthesises ones(N,1) and feeds the user `zvar` only
# to parameters named in `hetero`). MCI/MSLE instead evaluate a homoscedastic parameter as
# coef·Z[:,1] (get_ineff_vals), so the first `zvar` column must be the ones intercept whenever
# any inefficiency parameter is homoscedastic — otherwise that parameter silently becomes
# heteroscedastic in the first z-variable and `result.marginal` / `predict` disagree with the
# likelihood. Each `@test_throws` below would have BUILT successfully (no exception) before the
# fix. Kept self-contained (own data) so it can run independently of runtests.jl.

using SFrontiers
using Test
using Random
using Statistics
using Optim
using StatsAPI
using StatsAPI: predict
using DataFrames

const MCIB6 = SFrontiers.MCI_Backend
const MSB6  = SFrontiers.MSLE_Backend

# Loose NelderMead options — G6 assertions are about the spec-time guard and the predict/
# marginal surface, not the quality of the optimum.
_g6opt() = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40,),
                       main_solver = NelderMead(), main_opt = (iterations = 200,))

@testset "G6 R-05/FIX-06: homoscedastic param requires a constant first zvar column" begin
    Random.seed!(606)
    n  = 150
    X  = hcat(ones(n), randn(n), randn(n))
    z1 = randn(n); z2 = randn(n)
    Znc = hcat(z1, z2)          # NON-constant first column
    Zc  = hcat(ones(n), z1)     # constant `_cons` first column
    y   = X * [1.0, 0.5, -0.3] .+ 0.3 .* randn(n) .- abs.(0.5 .* randn(n))

    @testset "mixed hetero + non-constant first column is rejected (unified, MCI, MSLE)" begin
        # TN hetero=[:mu]: σᵤ² is homoscedastic ⇒ keys off Z[:,1]. A non-constant first column
        # must error at spec time (pre-fix it built and produced a silently heteroscedastic
        # σᵤ² with wrong marginal effects). Confirm the error is OUR guard, then the backends.
        err = nothing
        try
            sfmodel_spec(depvar = y, frontier = X, zvar = Znc, noise = :Normal,
                         ineff = :TruncatedNormal, hetero = [:mu], type = :prod)
        catch e
            err = e
        end
        @test err !== nothing
        @test occursin("constant intercept", sprint(showerror, err))

        @test_throws Exception MCIB6.sfmodel_spec(depvar = y, frontier = X, zvar = Znc,
            noise = :Normal, ineff = :TruncatedNormal, hetero = [:mu], type = :prod)
        @test_throws Exception MSB6.sfmodel_spec(depvar = y, frontier = X, zvar = Znc,
            noise = :Normal, ineff = :TruncatedNormal, hetero = [:mu], type = :prod)
        # Symmetric: hetero=[:sigma_sq] leaves μ homoscedastic.
        @test_throws Exception sfmodel_spec(depvar = y, frontier = X, zvar = Znc,
            noise = :Normal, ineff = :TruncatedNormal, hetero = [:sigma_sq], type = :prod)
    end

    @testset "guard fires across distributions with a homoscedastic 2nd parameter" begin
        # For each two-parameter distribution, heteroscedasticising only one parameter leaves
        # the other homoscedastic, so a non-constant Z[:,1] must be rejected.
        for (ineff, het) in ((:Gamma, [:k]), (:Gamma, [:theta]),        # Gamma is MCI-only
                             (:Weibull, [:lambda]), (:Weibull, [:k]),
                             (:Lomax, [:lambda]), (:Lomax, [:alpha]),
                             (:Lognormal, [:mu]), (:Lognormal, [:sigma_sq]))
            @test_throws Exception sfmodel_spec(depvar = y, frontier = X, zvar = Znc,
                noise = :Normal, ineff = ineff, hetero = het, type = :prod)
        end
    end

    @testset "FIX-06(A) fully homoscedastic + non-constant zvar is rejected" begin
        # hetero=Symbol[] with a supplied non-constant zvar: BOTH params key off Z[:,1].
        @test_throws Exception sfmodel_spec(depvar = y, frontier = X, zvar = Znc,
            noise = :Normal, ineff = :TruncatedNormal, type = :prod)
        @test_throws Exception sfmodel_spec(depvar = y, frontier = X, zvar = z1,
            noise = :Normal, ineff = :HalfNormal, type = :prod)   # single non-const column
    end

    @testset "no false positives: fully-hetero, constant-col and no-zvar all build" begin
        # Fully heteroscedastic ⇒ no homoscedastic parameter ⇒ any zvar (even non-constant
        # first column) is fine.
        @test sfmodel_spec(depvar = y, frontier = X, zvar = Znc, noise = :Normal,
            ineff = :TruncatedNormal, hetero = [:mu, :sigma_sq], type = :prod) isa SFrontiers.UnifiedSpec
        @test sfmodel_spec(depvar = y, frontier = X, zvar = z1, noise = :Normal,
            ineff = :HalfNormal, hetero = [:sigma_sq], type = :prod) isa SFrontiers.UnifiedSpec
        # Mixed hetero with a proper `_cons` first column builds.
        @test sfmodel_spec(depvar = y, frontier = X, zvar = Zc, noise = :Normal,
            ineff = :TruncatedNormal, hetero = [:mu], type = :prod) isa SFrontiers.UnifiedSpec
        # Fully homoscedastic with no zvar (auto ones) builds.
        @test sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
            ineff = :TruncatedNormal, type = :prod) isa SFrontiers.UnifiedSpec
        # Scaling (main-model zvar is ones internally) builds with a non-constant scaling Z.
        @test sfmodel_spec(depvar = y, frontier = X, zvar = Znc, noise = :Normal,
            ineff = :TruncatedNormal, hetero = :scaling, type = :prod) isa SFrontiers.UnifiedSpec
    end

    @testset "DSL @zvar path is guarded too (funnels through the keyword ctor)" begin
        df = DataFrame(yv = y, cons = ones(n), z1 = z1, z2 = z2, x1 = X[:, 2], x2 = X[:, 3])
        # Non-constant first @zvar column + homoscedastic σᵤ² ⇒ error.
        @test_throws Exception sfmodel_spec(@useData(df), @depvar(yv),
            @frontier(cons, x1, x2), @zvar(z1, z2);
            noise = :Normal, ineff = :TruncatedNormal, hetero = [:mu], type = :prod)
        # Constant `cons` first @zvar column ⇒ builds.
        @test sfmodel_spec(@useData(df), @depvar(yv),
            @frontier(cons, x1, x2), @zvar(cons, z1);
            noise = :Normal, ineff = :TruncatedNormal, hetero = [:mu], type = :prod) isa SFrontiers.UnifiedSpec
    end

    @testset "corrected form fits; result.marginal, predict(:marginal)/(:jlms) all finite" begin
        # The valid (constant-first-column) form of the R-05 failing input: μ hetero over
        # [_cons, z1], σᵤ² homoscedastic = coef·1. result.marginal, predict(:marginal) and
        # predict(:jlms) must be finite and mutually consistent (the intra-package
        # contradiction R-05 flagged is gone because σᵤ² is a genuine constant).
        sp  = sfmodel_spec(depvar = y, frontier = X, zvar = Zc, noise = :Normal,
                           ineff = :TruncatedNormal, hetero = [:mu], type = :prod)
        ini = sfmodel_init(spec = sp, frontier = X \ y, mu = [0.0, 0.0],
                           ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
        r = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MCI, n_draws = 200),
                        init = ini, optim_options = _g6opt(),
                        marginal = true, show_table = false, verbose = false)
        @test all(isfinite, Matrix(r.marginal))
        @test size(Matrix(r.marginal), 2) == 1                 # μ over [cons,z1] ⇒ z1 only
        pm = predict(r; zvar = Zc, what = :marginal)
        @test size(pm) == size(Zc) && all(isfinite, pm)
        # predict(:jlms) reproduces r.jlms: MCI/MSLE predict reuses the fit-time simulation
        # kernel for HN/TN, so it is bit-identical on the estimation data (σᵤ² = coef·1 everywhere).
        pj = predict(r; frontier = X, depvar = y, zvar = Zc, what = :jlms)
        @test maximum(abs.(vec(pj) .- vec(r.jlms))) < 0.02

        # MSLE corrected-form smoke: builds and fits with finite jlms.
        rM = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MSLE, n_draws = 200),
                         init = ini, optim_options = _g6opt(),
                         marginal = false, show_table = false, verbose = false)
        @test all(isfinite, vec(rM.jlms))
    end
end

@testset "G6 predict: a homoscedastic MCI/MSLE param is a true constant (MLE `_cons` parity)" begin
    # `predict`'s `_evaluate_param` used to evaluate a homoscedastic MCI/MSLE parameter as
    # coef·Z[:,1] (matching the old likelihood), so a non-constant first column in the
    # *new-data* zvar leaked into it and made predict(:jlms) disagree with predict(:marginal).
    # It now returns the constant coef — matching the MLE `_cons` semantics and the spec-time
    # guard's invariant (Z[:,1] ≡ 1 for homoscedastic params). This unit test would fail under
    # the old coef·Z[:,1] branch (a random first column ⇒ per-observation σᵤ²).
    Random.seed!(626)
    n  = 150
    X  = hcat(ones(n), randn(n), randn(n))
    z1 = randn(n)
    Zc = hcat(ones(n), z1)                      # [_cons, z1]: μ hetero, σᵤ² homoscedastic
    y  = X * [1.0, 0.5, -0.3] .+ 0.3 .* randn(n) .- abs.(0.5 .* randn(n))
    sp  = sfmodel_spec(depvar = y, frontier = X, zvar = Zc, noise = :Normal,
                       ineff = :TruncatedNormal, hetero = [:mu], type = :prod)
    ini = sfmodel_init(spec = sp, frontier = X \ y, mu = [0.0, 0.0],
                       ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
    r = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MCI, n_draws = 200),
                    init = ini, optim_options = _g6opt(),
                    marginal = false, show_table = false, verbose = false)

    spec   = SFrontiers._predict_get_unified_spec(r)
    σfield = SFrontiers._variance_field(spec.ineff, spec.mci_spec.idx.ineff)  # :sigma_u for TN
    # A NON-constant first column in the new-data zvar must NOT leak into the homoscedastic σᵤ².
    Zbad = hcat(randn(20), randn(20))
    σu2  = SFrontiers._evaluate_param(r, spec, Zbad, σfield, false; exponentiate = true)
    @test length(σu2) == 20
    @test all(x -> x ≈ σu2[1], σu2)                                    # constant across obs
    @test σu2[1] ≈ SFrontiers._scalar_from(r, spec, σfield)            # == exp(fitted coef)
end
