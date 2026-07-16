using SFrontiers
using Test
using Random
using Statistics
using DataFrames
using Optim
using StatsAPI
using StatsAPI: residuals, fitted, predict
using StatsBase: CoefTable
using RecipesBase

# ---------------------------------------------------------------------
# Shared synthetic data for every testset — a single Normal + HalfNormal
# dataset with a zvar column. Keeping this at module scope lets downstream
# testsets reuse the shared MLE + HN fit (`res_mle`) without re-fitting.
# ---------------------------------------------------------------------
Random.seed!(12345)
const N = 200

const cons = ones(Float64, N)
const x1   = randn(N)
const x2   = randn(N)
const X    = hcat(cons, x1, x2)

const z1 = randn(N)
const Z  = hcat(cons, z1)

const β_true = [1.0, 0.5, -0.3]
v = 0.3 .* randn(N)
u = abs.(0.5 .* randn(N))
const y = X * β_true .+ v .- u

const β_ols = X \ y
const df    = DataFrame(yvar = y, cons = cons, x1 = x1, x2 = x2, z1 = z1)

# ---------------------------------------------------------------------
# Thorough predict-path coverage (bit-identity on every backend and every
# model combination, subset / recycling / user-draws / brand-new firms)
# lives in E:/ajen/myJulia/SFrontiers_myowntests/. runtests.jl is kept
# lightweight — it verifies that every public code path is wired and
# functional, not that every numerical value is correct to 1e-10.
# ---------------------------------------------------------------------

@testset "SFrontiers.jl" begin

    # Shared MLE + HalfNormal fit — reused by the accessor, diagnostics,
    # plot-recipe, and predict-smoke testsets below.
    _spec = sfmodel_spec(depvar = y, frontier = X,
                         noise = :Normal, ineff = :HalfNormal, type = :prod)
    _init = sfmodel_init(spec = _spec, frontier = β_ols,
                         ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
    _opt  = sfmodel_opt(
        warmstart_solver = NelderMead(),
        warmstart_opt    = (iterations = 40, g_abstol = 1e-3),
        main_solver      = Newton(),
        main_opt         = (iterations = 80, g_abstol = 1e-5, show_trace = false))
    res_mle = sfmodel_fit(spec = _spec, method = sfmodel_method(method = :MLE),
                          init = _init, optim_options = _opt,
                          marginal = false, show_table = false, verbose = false)

    @testset "Backend fits (smoke)" begin
        # Verify each backend converges on the same Normal + HN dataset.
        # Tight assertions (converged, loglik finite, backend tag, #coeffs)
        # are what catches a backend-specific regression cheaply.
        for (method_sym, n_draws, tag) in ((:MLE,  nothing, :MLE),
                                            (:MCI,  100,     :MCI),
                                            (:MSLE, 100,     :MSLE))
            meth = n_draws === nothing ?
                   sfmodel_method(method = method_sym) :
                   sfmodel_method(method = method_sym, n_draws = n_draws)
            res = sfmodel_fit(spec = _spec, method = meth, init = _init,
                              optim_options = _opt, marginal = false,
                              show_table = false, verbose = false)
            @test res isa SFResult
            @test res.backend === tag
            @test res.converged == true
            @test length(res.coeff) == 5
            @test isfinite(res.loglikelihood)
        end
    end

    @testset "StatsAPI accessors" begin
        @test StatsAPI.coef(res_mle)      === res_mle.coeff
        @test StatsAPI.vcov(res_mle)      === res_mle.var_cov_mat
        @test StatsAPI.stderror(res_mle)  === res_mle.std_err
        @test StatsAPI.nobs(res_mle)       == length(y)
        @test StatsAPI.dof(res_mle)        == length(res_mle.coeff)
        @test StatsAPI.aic(res_mle) ≈ -2 * res_mle.loglikelihood + 2 * length(res_mle.coeff)
        @test StatsAPI.bic(res_mle) ≈ -2 * res_mle.loglikelihood + log(length(y)) * length(res_mle.coeff)
        ci = StatsAPI.confint(res_mle; level = 0.95)
        @test size(ci) == (length(res_mle.coeff), 2)
        @test all(ci[:, 1] .<= res_mle.coeff .<= ci[:, 2])
        @test StatsAPI.coeftable(res_mle) isa CoefTable
        @test !isempty(sprint(show, MIME"text/plain"(), res_mle))
    end

    @testset "Diagnostics" begin
        ε  = residuals(res_mle; type = :composed)
        ŷ  = fitted(res_mle;    type = :frontier)
        @test length(ε) == length(y)
        @test ŷ .+ ε ≈ y
        û  = residuals(res_mle; type = :u)
        @test û == collect(res_mle.jlms)
        v̂  = residuals(res_mle; type = :v)
        @test v̂ ≈ ε .+ û
        @test length(residuals(res_mle; type = :ols)) == length(y)
        @test fitted(res_mle; type = :response) ≈ ŷ .- û

        lr = sf_vs_ols(res_mle)
        @test lr isa SFrontiers.LRTestResult
        @test lr.LR ≈ -2 * (res_mle.OLS_loglikelihood - res_mle.loglikelihood)
        @test lr.mixed == true
        @test 0 <= lr.pvalue <= 1

        lr0 = lrtest(res_mle, res_mle; dof = 1)
        @test lr0.LR ≈ 0 atol = 1e-8
        @test lr0.pvalue ≈ 1.0

        es = efficiency_summary(res_mle)
        @test es.mean_bc   ≈ mean(res_mle.bc)
        @test es.mean_jlms ≈ mean(res_mle.jlms)
        @test es.n == length(res_mle.bc)

        rd = residual_diagnostics(res_mle)
        @test sign(rd.skew_ols) == sign(res_mle.OLS_resid_skew)
        @test rd.expected_skew_sign == -1
        @test isfinite(rd.JarqueBera_stat) && 0 <= rd.JarqueBera_pvalue <= 1
    end

    @testset "Plot recipes" begin
        @test !isempty(RecipesBase.apply_recipe(Dict{Symbol, Any}(), res_mle))
        @test !isempty(RecipesBase.apply_recipe(Dict{Symbol, Any}(), res_mle, :efficiency))
        @test !isempty(RecipesBase.apply_recipe(Dict{Symbol, Any}(), res_mle, :residuals))
    end

    @testset "predict smoke" begin
        # Cross-sectional predict surface using the shared MLE + HN fit.
        # Bit-identity (rtol = 1e-10) for every backend/ineff/panel
        # combination lives in SFrontiers_myowntests/ — here we only check
        # that each target returns the right shape on new inputs.
        @test predict(res_mle; frontier = X, what = :frontier) ≈ X * res_mle.coeff_frontier
        @test predict(res_mle; frontier = X, depvar = y, what = :residuals) ≈
              y .- X * res_mle.coeff_frontier

        jl = predict(res_mle; frontier = X, depvar = y, what = :jlms)
        bc = predict(res_mle; frontier = X, depvar = y, what = :bc)
        @test length(jl) == N && all(isfinite, jl)
        @test length(bc) == N && all(0 .<= bc .<= 1)
        @test maximum(abs.(jl .- vec(res_mle.jlms))) < 1e-4
        @test maximum(abs.(bc .- vec(res_mle.bc)))   < 1e-4

        @test length(predict(res_mle; frontier = X, depvar = y, what = :response)) == N
        allout = predict(res_mle; frontier = X, depvar = y, what = :all)
        @test haskey(allout, :jlms) && haskey(allout, :bc)

        # Marginal-effect predict shape — the only :marginal-path assertion
        # that the personal suite doesn't already cover.
        spec_tn = sfmodel_spec(depvar = y, frontier = X, zvar = Z,
                               noise = :Normal, ineff = :TruncatedNormal,
                               hetero = [:mu, :sigma_sq], type = :prod)
        init_tn = sfmodel_init(spec = spec_tn, frontier = β_ols,
                               mu = [0.0, 0.0], ln_sigma_sq = [0.0, 0.0],
                               ln_sigma_v_sq = log(0.09))
        res_tn  = sfmodel_fit(spec = spec_tn,
                              method = sfmodel_method(method = :MSLE, n_draws = 100),
                              init = init_tn, optim_options = _opt,
                              marginal = false, show_table = false, verbose = false)
        @test size(predict(res_tn; zvar = Z, what = :marginal)) == size(Z)

        # Input validation: missing required kwargs must error.
        @test_throws Exception predict(res_mle; what = :jlms)
        @test_throws Exception predict(res_mle; frontier = X, what = :jlms)
    end

    @testset "DSL macros: DataFrame specification" begin
        spec = sfmodel_spec(
            @useData(df),
            @depvar(yvar),
            @frontier(cons, x1, x2),
            @zvar(cons, z1);
            noise = :Normal, ineff = :TruncatedNormal,
            hetero = [:mu], type = :prod)
        @test spec isa SFrontiers.UnifiedSpec

        init = sfmodel_init(spec = spec, frontier = β_ols,
                            mu = [0.0, 0.0],
                            ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
        res = sfmodel_fit(spec = spec,
                          method = sfmodel_method(method = :MSLE, n_draws = 100),
                          init = init, optim_options = _opt,
                          marginal = false, show_table = false, verbose = false)
        @test res.converged == true
        @test isfinite(res.loglikelihood)
    end

end

# Regression tests for the correctness fixes in 00to_fix.md.
include("test_fixes.jl")

# Regression tests for the prod/cost sign fixes in 00Direction2_findings.md.
include("test_direction2_fixes.jl")

# Regression tests for the 03_reaudit_findings.md Group 1 fixes (R-06, R-23).
include("test_reaudit_G1.jl")

# Regression tests for the 03_reaudit_findings.md Group 2 fixes (R-02, R-13, R-15, R-16, R-17).
include("test_reaudit_G2.jl")

# Regression tests for the 03_reaudit_findings.md Group 3 fixes (R-04, R-03, R-26).
include("test_reaudit_G3.jl")

# Regression tests for the 03_reaudit_findings.md Group 4 fixes (R-18, R-19, R-20, R-08).
include("test_reaudit_G4.jl")

# Regression tests for the 03_reaudit_findings.md Group 5 fixes (R-12, R-11, R-14, R-24, R-21).
include("test_reaudit_G5.jl")

# Regression tests for the 03_reaudit_findings.md Group 6 fix (R-05, FIX-06(A)/(B)).
include("test_reaudit_G6.jl")

# Regression tests for the 03_reaudit_findings.md Group 7 fixes
# (R-01, R-22, R-28, FIX-09, R-27, R-07, R-09, R-10).
include("test_reaudit_G7.jl")

# Regression tests for the 2026-07 predict re-review: MCI/MSLE predict reuses the
# fit-time simulation kernel for HN/TN so predict==fit bit-for-bit on the estimation data.
include("test_predict_sim_identity.jl")

# GPU/CUDA CPU-vs-GPU equivalence smoke (2026-07 GPU audit, 07_gpu_audit_findings.md).
# No-op unless CUDA.jl is installed and functional; runs in a subprocess because
# `using CUDA` must precede `using SFrontiers`. Full sweep: test/gpu_sweep_runner.jl.
include("test_gpu_equivalence.jl")

# Regression test for the gap-closure fix GC-01 (09_gap_closure_findings.md):
# bare-Symbol hetero errors cleanly on all DSL ctor paths (R-13 parity).
include("test_gap_closure_GC01.jl")

# Regression test for the gap-closure diagnostic OBS-1 (09_gap_closure_findings.md):
# MCI jlms/bc warn when the importance sample is degenerate (low ESS).
include("test_gap_closure_OBS1.jl")
