# SFrontiers.jl release notes

## Version 2.0.1

Version 2.0.1 changes only the test suite. Its GPU checks now also run when the package is installed from the registry, and they are skipped, with a message, when the CUDA.jl version found is newer than 6.4. Tests whose outcome could depend on the platform's rounding, on the path an optimizer takes or on tolerances tighter than their purpose requires (one of them failed on Linux) now check what they guard without that dependence.

## Version 2.0.0

Version 2.0.0 is a major release. It corrects results that earlier versions computed wrongly in several models, makes simulation-based estimation (`method = :MSLE` and `:MCI`) more accurate by default, and makes the interface the same under every estimation method. Code written for version 1.4 may need changes. Models fitted with version 1.4 or earlier should be re-estimated: the results of `:MSLE` and `:MCI` fits change, and so do those of the models listed under "Corrected results".

### Corrected results

Earlier versions returned the following results without an error or a warning.

#### Standard errors

- **Laplace noise (`:MCI` and `:MSLE`).** Standard errors were computed from the Hessian of the simulated likelihood, which has kinks under Laplace noise. Those of the frontier and inefficiency coefficients were far too small (by factors of 50 to 900 in test fits) or `NaN`. They now come from the outer product of gradients (OPG), as the table header and `result.vcov_type` state.
- **Bootstrap of marginal effects (`:MLE`).** For truncated-normal models with `:scaling` and `panel_TFE` truncated-normal models, the bootstrap resamples started from the estimates in the wrong parameter order. Some resamples ended at an inferior local optimum, and the bootstrap standard errors were off by several percent.
- **Singular Hessians (all methods).** A numerically singular Hessian was still inverted, which gave meaningless standard errors. They are now `NaN`, with `redflag = 1`.

#### Log-likelihood and estimates

- **`panel_TRE`.** The log-likelihood could be `NaN`, or wrong by up to about 200 nats once T is moderate, even at ordinary parameter values.
- **`panel_TFE_CSW`.** For panels longer than about 50 periods, the log-likelihood was too high (by hundreds of nats at T = 100) and its derivatives pointed the wrong way, so fits could end at wrong estimates. It was also inaccurate when σᵤ/σᵥ exceeds about 300.
- **Two specifications under `:MLE`.** `sfmodel_fit(spec = A, ...)` estimated the most recently built specification, so building a second specification before fitting the first gave the second model's estimates under the first model's names.
- **MCI with Gamma or Lomax inefficiency.** The draw transformation did not follow the mass of the distribution, which biased the simulated log-likelihood. In cross-sectional fits the bias was about half a nat. In `panel_TFE` fits it reached about 100 nats: the Gamma shape was underestimated (k̂ ≈ 3.5 when k = 5), and the standard error of ln k̂ was less than half its correct size. The transformation now uses revised scales.
- **Gumbel copula (`:MCI` and `:MSLE`).** A floor inside the copula density lowered the log-likelihood as the dependence grows. The effect is negligible for ρ ≤ 1.5; under normal noise it is about 0.003 nats per observation at ρ = 5 and 0.06 at ρ = 10, and larger under Laplace noise.
- **Copulas with `Float32` data.** The Gaussian-copula log-likelihood was off by about 1 nat at moderate ρ, and by hundreds to thousands of nats at |ρ| ≥ 0.96.
- **Truncated-normal inefficiency concentrated near zero (μ/σᵤ far below 0).**
  - `:MCI`: below μ/σᵤ ≈ −5 (≈ −3 with a copula), the log-likelihood lost accuracy, with errors of several nats near −7 and of hundreds to thousands further out, which could move a fit to another optimum.
  - `:MSLE`: below μ/σᵤ ≈ −37, the log-likelihood lost accuracy and then became `NaN`.
  - `:MLE` (truncated-normal and exponential models, cross-sectional and panel): as σᵤ → 0, or with wrong-skew data, the log-likelihood and the Hessian lost accuracy; standard errors could be up to 5.6 times too small.
- **Student-t noise with very large ν.** When the estimated degrees of freedom became extremely large (above about 10¹⁴, i.e., data close to normal noise), the log-likelihood lost accuracy, and at still larger ν it could be wrong by about 200 nats, or `NaN`.
- **Log-likelihood after a failed line search.** `loglikelihood` and `gradient_norm` described the optimizer's previous point, not the reported estimates (for example, −207.6 reported against −244.8 at the estimates). Both are now computed at the reported estimates.
- **Parameters beyond their bounds.** A homoscedastic variance, scale, shape or copula coefficient that ended beyond its numerical bound was reported at a value the likelihood never used. It is now reported at the bound, and the optimizer's value is kept in `result.bound_check`.

#### Efficiency scores, marginal effects and predictions

- **Closed-form JLMS and BC (`:MLE`, all models).** At observations where σᵤ is very small relative to σᵥ, which are common in heteroscedastic models, JLMS could be negative and BC slightly above 1; in extreme cases the scores were off by a factor of up to 10.
- **`predict` for heteroscedastic `:MLE` Exponential models with a one-column `zvar`.** `zvar` was ignored, so the predicted `jlms`, `bc` and `response` were wrong wherever that column is not 1 (by about 100% in a test case).
- **Marginal effects of truncated-normal models** (heteroscedastic, `:scaling` and `panel_TFE`, every method, and `predict(what = :marginal)`). They were inaccurate where μ/σᵤ is very negative, with the wrong sign in extreme cases.
- **Lomax marginal effects.** Where the shape α ≤ 1, the Lomax mean does not exist. Marginal effects on E(u) could be ±Inf there; they are now `NaN`, with a warning.

#### Tables

- **Row labels under `:MLE`.** With user-supplied `varnames`, the rows of μ and of the scaling coefficients were mislabelled in truncated-normal `:scaling` models and `panel_TFE` truncated-normal models (the estimates were right).
- **`sfmodel_MixTable`.** Two critical values were wrong: 6 degrees of freedom at the 1% level (now 16.074; was 16.704) and 12 degrees of freedom at the 5% level (now 20.401; was 20.410).

### Simulation-based estimation (`:MSLE` and `:MCI`)

These are deliberate changes; results differ from version 1.4 by about the simulation error.

- **More draws by default.** `n_draws` defaults to 4,096 (it was 1,024). Fits take about four times as long and need about four times as much memory at the same `chunks`.
- **No recycled Halton points.** Each observation (each firm in panel models) gets its own block of `n_draws` points from one Halton sequence; version 1.4 recycled a sequence of at most 32,767 points. To reproduce a version 1.4 fit, pass its `n_draws` and `recycling = 2^15 - 1` to `sfmodel_method`. The keyword `distinct_Halton_length` is deprecated in favour of `recycling`.
- **Draws in `predict`.** `predict` uses the draws stored with the fit. In panel models, the k-th new firm uses the draws of the k-th estimation firm.
- **User-supplied `draws`** are read by their number of rows: one row per observation (or firm), or one sequence shared by all. The fit may keep the array without copying it, so do not modify it while the result is in use.
- **Laplace noise.** When the main solver stops without converging, the fit continues with BFGS and reports it in `result.finish`; by default Newton runs at most 30 iterations before that. The `:MSLE` likelihood is smoothed as the `:MCI` one already was, so default `:MSLE` fits converge where they used to fail.

### Interface changes

- **One name for the inefficiency variance**, σᵤ², under every method and model: `ln_sigma_u_sq` in `sfmodel_init` (it was `ln_sigma_sq`, and `ln_lambda` for the Exponential) and `hetero = [:sigma_u_sq]` (it was `:sigma_sq` or `:lambda`). The old input names still work, with a deprecation warning; the names in tables and result fields change.
- **`sfmodel_init()`** no longer takes `method`. It returns one vector, in one parameter order, for every method; for cross-sectional models every component is optional.
- **Arguments that have no effect** are refused with an error that names them, instead of being silently ignored. Starting values at or beyond a parameter's numerical bound are refused too.
- **`redflag`** is an `Int` (0, 1 or 2) under every method and follows one rule, which now includes parameters at their numerical bounds (`result.bound_check`, `result.hessian_flag`). Code that tested `if result.redflag` must test `result.redflag > 0`. With a gradient-free main solver, `gradient_norm` is now the actual gradient at the estimates, so such fits can receive `redflag = 1`.
- **`:MLE` optimizer.** Without `optim_options`, `:MLE` fits run NelderMead (200 iterations) and then Newton (200 iterations), as the other methods do; every setting in `optim_options` is now respected.
- **`:MLE` with the macro syntax.** A scaling or heteroscedastic equation with one `@zvar` column names its coefficient after the variable, as `:MCI` and `:MSLE` do.
- **`sf_vs_ols`** applies only to the model for which the null distribution of its statistic is known (cross-sectional, normal noise, homoscedastic half-normal, no copula); the `dof` keyword was removed.
- **Panel results** no longer include `OLS_loglikelihood`.

### New features

- `sf_skewness_test(result)`: the one-sided third-moment test of the OLS residuals, with a kurtosis-robust variance; `residual_diagnostics` reports the same statistics.
- `sfmodel_fit(..., integration_check = true)`: checks whether the number of draws is enough (OK, CAUTION or FAIL).
- Erlang inefficiency distributions (`:Erlang2`, `:Erlang3`, `:Erlang4`) under `:MCI` and `:MSLE`.
- Warnings when an estimated shape parameter may reduce numerical accuracy, and when a GPU fit may run out of memory.
- A `recycling` option for very large N × `n_draws`.
- Single-precision (`Float32`) data.
- `lrtest` warns when simulated fits used different draws, or when a simulated fit is compared with a closed-form one.
- `predict(what = :marginal)` for cross-sectional `:MLE` Exponential models.

### Other improvements

- Improved numerical accuracy and stability of the likelihoods, their derivatives, the efficiency scores and the marginal effects, beyond the corrections above.
- When the Hessian or the log-likelihood cannot be evaluated at the estimates, the fit returns the estimates and efficiency scores, with `NaN` diagnostics and `redflag = 1`, instead of stopping with an error (except under `Newton()` and `NewtonTrustRegion()`, which need the Hessian to run).
- Calls that failed in version 1.4 now work: data columns of any vector type (for example from CSV.jl), the list form `frontier = [v1, v2, …]` for every model that `:MLE` can fit, and `sfmodel_ChiSquareTable`.
- Lower memory use for panel models and for the efficiency scores.
- Clearer errors and warnings, and a more robust bootstrap of `:MLE` marginal effects.
- An expanded manual, with new sections on parameter bounds, statistical tests, copulas and Laplace noise.
- New reference tests check the likelihoods against independent numerical integration. CPU–GPU agreement is documented in `test/GPU_VALIDATION.md`.

### Requirements

- Julia 1.10 or later.
- PrettyTables 3.
- GPU estimation: CUDA.jl 6.4 or earlier. CUDA.jl 6.5 fails to compile the GPU sums that MCI and MSLE fits use for their derivatives; SFrontiers declares this bound, so the package manager installs CUDA.jl 6.4 alongside it.
