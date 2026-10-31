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

@testset "SFrontiers.jl" begin

    # Shared MLE + HalfNormal fit — reused by the accessor, diagnostics,
    # plot-recipe, and predict-smoke testsets below.
    _spec = sfmodel_spec(depvar = y, frontier = X,
                         noise = :Normal, ineff = :HalfNormal, type = :prod)
    _init = sfmodel_init(spec = _spec, frontier = β_ols,
                         ln_sigma_u_sq = log(0.25), ln_sigma_v_sq = log(0.09))
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
        @test lr.dof == 1

        lr0 = lrtest(res_mle, res_mle; dof = 1)
        @test lr0.LR ≈ 0 atol = 1e-8
        @test lr0.pvalue ≈ 1.0
        @test lr0.simulated === :none
        @test lr0.simulation_note == ""

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
        # Cross-sectional predict surface on new inputs, using the shared MLE + HN fit: each
        # target returns the right shape. predict's agreement with the fit is tested in other
        # files, for example test_predict_sim_identity.jl and test_predict_draw_alignment.jl.
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

        # Marginal-effect predict shape on new inputs.
        spec_tn = sfmodel_spec(depvar = y, frontier = X, zvar = Z,
                               noise = :Normal, ineff = :TruncatedNormal,
                               hetero = [:mu, :sigma_u_sq], type = :prod)
        init_tn = sfmodel_init(spec = spec_tn, frontier = β_ols,
                               mu = [0.0, 0.0], ln_sigma_u_sq = [0.0, 0.0],
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
                            ln_sigma_u_sq = log(0.25), ln_sigma_v_sq = log(0.09))
        res = sfmodel_fit(spec = spec,
                          method = sfmodel_method(method = :MSLE, n_draws = 100),
                          init = init, optim_options = _opt,
                          marginal = false, show_table = false, verbose = false)
        @test res.converged == true
        @test isfinite(res.loglikelihood)
    end

end

# Correctness checks across the backends: panel draw floors, CSW cost-frontier ranking,
# copula predict, panel zvar checks, stderror.
include("test_fixes.jl")

# The prod/cost sign of the matrix-form jlms_bc_indices (MCI, MSLE).
include("test_prod_cost_sign.jl")

# Standard errors when the Hessian is not positive definite or not invertible; redflag levels.
include("test_stderr_redflag_levels.jl")

# The unified spec layer: user varnames, bare-Symbol hetero, DSL scaling specs, panel_TFE
# component inits and supported methods.
include("test_unified_spec_dispatch.jl")

# Diagnostics across backends: panel-simulation, CSW and TRE fits, Wang-Ho panel residuals.
include("test_diagnostics_backends.jl")

# The predict layer: zvar widths, user-draw metadata, :marginal and :all for every
# distribution, the predict docstring.
include("test_predict_layer.jl")

# Auxiliary tables (Weibull k, Lomax α, Student-t ν), panel marginal clamps,
# sfmodel_panel_init validation, result.sigma_u.
include("test_aux_tables_panel_inputs.jl")

# A homoscedastic inefficiency parameter needs a constant first zvar column (MCI, MSLE).
include("test_homoscedastic_zvar.jl")

# MoMTest cost recentring, custom draws with multiRand = false, plot recipes, store_trace,
# legacy accessors, the MLE bootstrap, the Gamma copula F_u.
include("test_accessors_recipes_misc.jl")

# MCI/MSLE predict reuses the fit-time simulation kernel for every distribution, so predict
# equals the fit bit for bit on the estimation data.
include("test_predict_sim_identity.jl")

# GPU (CUDA) against CPU equivalence smoke. No-op unless CUDA.jl is installed and functional;
# runs in a subprocess because `using CUDA` must precede `using SFrontiers`. Full sweep:
# test/gpu_sweep_runner.jl.
include("test_gpu_equivalence.jl")

# Bare-Symbol hetero errors cleanly on every DSL constructor path.
include("test_dsl_hetero_symbol.jl")

# The MCI jlms/bc effective-sample-size warning.
include("test_mci_ess_note.jl")

# Erlang{2,3,4} inefficiency: spec construction (MCI + MSLE; Gamma stays MCI-only), the MSLE
# quantile kernel, MCI and MSLE likelihoods against numerical integration, recovery fits,
# predict identity, and the Wang-Ho panel_TFE models.
include("test_erlang.jl")

# Input shape and type guards: multiRand = false draw shapes, Float32 specs, panel predict zvar
# rows, error-message attribution, sfmodel_spec row and column checks.
include("test_intake_guards.jl")

# The marginal-effect helpers clamp their parameters as the likelihood does.
include("test_marginal_clamps.jl")

# Lomax marginal effects where the mean does not exist: NaN and one warning.
include("test_lomax_mean_existence.jl")

# Optimizer options do not leak from one fit to the next.
include("test_optim_options_isolation.jl")

# One name for the inefficiency-distribution variance, `ln_sigma_u_sq` with `hetero = [:sigma_u_sq]`, under
# every method; the names of versions before 2.0.0 map to the same slot with a deprecation warning. The
# internal index and result-object keys keep their own spelling.
include("test_sigma_u_naming.jl")

# method = :MLE fits the spec it is handed, not the last spec constructed.
include("test_mle_spec_isolation.jl")

# predict pairs fit-time draws with rows by position, the draws of new panel firms, and the
# summation order of the scaling multiplier.
include("test_predict_draw_alignment.jl")

# sfmodel_init: every component optional, one Vector{Float64} for every method; the MLE-only
# panel models refuse zvar and hscale.
include("test_init_one_shape.jl")

# Behaviour the manual documents: sfmodel_ChiSquareTable, the table utilities, the list form of
# frontier, AbstractVector inputs, the indices of a jlms_bc_index = false fit.
include("test_documented_api.jl")

# State isolation and starting values: deprecated Vararg specs and sfmodel_opt, Float32 predict
# constants, starting-value widths, the frontier OLS start, the panel_TFE TN start, print_table
# types, displayed names, sfmodel_init state.
include("test_state_init_consistency.jl")

# Input forms and refused arguments: one-column inputs to predict, DSL eqnames, omega
# spellings, settings a spec or method cannot read, warmstart options, predict's level,
# custom-draw types, GPU draw objects, error paths.
include("test_input_forms_and_refusals.jl")

# The panel_TRE (Greene 2005 true random-effect) likelihood: clamped variances, a
# cancellation-free G, log-determinants from the eigenvalues; agreement with quadrature over
# the random effect.
include("test_tre_likelihood.jl")

# The truncated-normal inverse CDF of the simulation backends, and the admitted ranges of the
# copula parameters.
include("test_quantile_and_ranges.jl")

# One name for the Exponential's parameter (sigma_u_sq, ln_sigma_u_sq, σᵤ²); lambda stays the
# scale of Weibull and Lomax.
include("test_expo_naming.jl")

# The panel_TRE orthant probability in log space, combined with the truncation constant; bc
# within (0, 1].
include("test_tre_orthant_floor.jl")

# The Student-t log normalising constant at large ν.
include("test_t_logconst_stability.jl")

# ForwardDiff derivatives of log Φ in the MLE likelihoods (asymptotic series below z = -40).
include("test_normlogcdf_derivatives.jl")

# The Gaussian bulk taken out of the truncated-normal likelihoods as σᵤ → 0.
include("test_tn_gaussian_bulk.jl")

# redflag is the manual's Int 0/1/2 under every method: one Hessian rule, one default
# optimizer, every Optim option honoured under :MLE.
include("test_redflag_rule.jl")

# The MCI and panel truncated-normal normaliser log Φ(μ/σᵤ) in the lower tail.
include("test_tn_normaliser.jl")

# The truncated-normal quantile in the lower tail (MSLE and panel draws, matrix-form
# jlms_bc_indices).
include("test_tn_quantile_tail.jl")

# The analytic Exponential log-likelihood as σᵤ → 0.
include("test_expo_small_sigma.jl")

# The MLE bootstrap starts each replicate from the estimate in the analytic backend's order.
include("test_boot_start_order.jl")

# After a failed line search the result describes one point.
include("test_line_search_failure.jl")

# The cross-sectional truncated-normal likelihood without its Gaussian bulk (μ/σᵤ or μ*/σ* below -40).
include("test_tn_loglik_far_tail.jl")

# Stable JLMS/BC closed forms and truncated-normal E(u).
include("test_jlms_bc_closed_forms.jl")

# The MCI truncated-normal CDF read by the copulas, in the lower tail.
include("test_copula_tn_cdf.jl")

# The panel_TFE_CSW orthant probability and determinant in log space; variance clamps.
include("test_csw_orthant_logspace.jl")

# The MLE bootstrap's redraw loop: cap, Ctrl-C, restored names, failure report.
include("test_boot_redraw_rules.jl")

# MLE table names for TN scaling and panel_TFE models, gradient_norm of gradient-free solvers,
# failed line searches flagged.
include("test_fit_report_names_flags.jl")

# The truncated-normal E(u) kernels behind result.marginal in the lower tail.
include("test_tn_marginal_kernels.jl")

# The MLE bootstrap refuses a result that is not the last :MLE fit, and fits without
# statistics to bootstrap.
include("test_boot_guards.jl")

# panel_TFE_CSW at large σᵤ/σᵥ in log space; the boundary test on the variances.
include("test_csw_lambda_boundary.jl")

# The printed header of scaling models and the DSL coefficient names under :MLE.
include("test_header_and_names.jl")

# predict(:marginal) and (:all) for cross-sectional :MLE Exponential fits.
include("test_mle_expo_predict_marginal.jl")

# An error in the final Hessian, or in the re-evaluation at the estimate, keeps the fit; the
# bootstrap redraws such replicates.
include("test_hessian_error_guard.jl")

# One log-sum-exp temporary per evaluation in the chunked MCI and MSLE likelihoods.
include("test_logsumexp_temporary.jl")

# The copula clamps follow the precision in which the probabilities are computed.
include("test_copula_clamp_precision.jl")

# panel_TRE with σᵥ very small next to σᵤ.
include("test_tre_onesided_orthant.jl")

# The truncated normal with μ and σᵤ both tiny next to σᵥ (LL_T(Trun), LL_T(Trun_Scale)).
include("test_tn_tiny_mu_sigma.jl")

# redflag requires a positive-definite Hessian.
include("test_redflag_posdef.jl")

# predict for :MLE Exponential fits with a one-column zvar without a constant.
include("test_mle_expo_onecol_predict.jl")

# panel_TFE MCI scales: Gamma θ·√(1+k²) and the Lomax median.
include("test_panel_mci_scale.jl")

# The cross-sectional MCI Gamma scale θ·√(1+k²).
include("test_gamma_mci_scale.jl")

# The panel_TFE_CSW log-space orthant: gradient and Hessian up to the variance bound.
include("test_csw_logorthant_derivatives.jl")

# The panel likelihood transforms its draws per group of firms: bit identity and memory.
include("test_panel_chunk_memory.jl")

# The panel likelihood per group of firms on the GPU (gated).
include("test_gpu_draw_memory.jl")

# The efficiency indices computed per group of observations or firms: equal to the indices computed at once.
include("test_jlms_bc_chunks.jl")

# predict reads the fit's stored draws in groups of chunks; new panel firms use the estimation's draws.
include("test_predict_draw_reuse.jl")

# A fit stores its draws once and uses a user's draw array without a copy when type and device match.
include("test_draw_storage.jl")

# Halton draws: one base-2 sequence of N × n_draws points per fit, single precision kept below 1, and the rule of versions before 2.0.0 behind distinct_Halton_length.
include("test_halton_draws.jl")

# The default number of draws per observation of :MCI and :MSLE.
include("test_default_draws.jl")

# Memory of one Hessian evaluation and the warning a GPU fit prints when it exceeds the free GPU memory.
include("test_gpu_memory_warning.jl")

# The optional integration check of MCI and MSLE fits.
include("test_integration_check.jl")

# The integration check flags a spurious maximum of the simulated MCI objective that shared draws create.
include("test_integration_check_shared_draws.jl")

# Warning and stored verdict when an estimated shape lies where the default MCI rule (or MSLE for the Lomax) is unreliable.
include("test_shape_range_warning.jl")

# one-sided third-moment test of the OLS residuals
include("test_sf_skewness_test.jl")

# sf_vs_ols: the established case, the refusals, the mixed χ² table
include("test_sf_vs_ols_reference.jl")

# lrtest: draw settings of simulated fits
include("test_lrtest_simulation_notes.jl")

# panel truncated-normal likelihoods against 512-bit references
include("test_panel_tn_likelihoods.jl")

# Boundary verdicts, reported coefficients and starting values at the numerical bounds.
include("test_bounds_reporting.jl")

# Laplace noise: the MSLE density uses the smoothed absolute value of MCI.
include("test_laplace_msle_kernel.jl")

# Laplace noise: standard errors from the outer product of gradients, checked against the closed-form likelihood.
include("test_laplace_opg_standard_errors.jl")

# Laplace noise: the BFGS finish after an unconverged main solver.
include("test_laplace_bfgs_finish.jl")

# The incomplete-gamma CDF of the copula Gamma model: derivatives in the shape.
include("test_gamma_copula_cdf_derivative.jl")

# Gumbel copula: the log density in logarithms, without floors.
include("test_gumbel_copula_logspace.jl")

# The verdict rules of the CPU-GPU equivalence sweep (no GPU needed).
include("test_gpu_sweep_verdict.jl")

# Reference tests: cross-sectional likelihoods against numerical integration of textbook densities.
include("test_reference_integration_crosssection.jl")

# Reference test: the Wang-Ho fixed-effects panel likelihood against two-dimensional numerical integration.
include("test_reference_wang_ho_panel.jl")

# Reference test: the Chen-Schmidt-Wang fixed-effects panel likelihood against one-dimensional numerical integration.
include("test_reference_csw_panel.jl")
