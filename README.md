# SFrontiers.jl

[![CI](https://github.com/HungJenWang1991/SFrontiers.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HungJenWang1991/SFrontiers.jl/actions/workflows/CI.yml)
[![User manual](https://img.shields.io/badge/docs-user%20manual-blue.svg)](https://hungjenwang1991.github.io/SFrontiers.jl/#user-manual-for-julia-implementation)

**Stochastic frontier analysis in Julia, from the classic normal–half-normal model to Student-t noise, Weibull or Lomax inefficiency, and copula dependence, in one workflow.**

SFrontiers.jl estimates the classic models by analytic maximum likelihood. Every other model is estimated by simulated maximum likelihood, made accurate by automatic differentiation and fast by GPU computing. Where both approaches apply, their estimates agree closely, and switching between them takes one argument.

## Features

- **Three estimation methods, one interface**
  - **MLE**: analytic maximum likelihood, with no simulation, for the classic models: normal noise with half-normal, truncated-normal or exponential inefficiency.
  - **MSLE**: maximum simulated likelihood, drawing inefficiency through its inverse CDF.
  - **MCI**: Monte Carlo integration with change-of-variable transformations.
  - The simulation methods use Halton quasi-random draws, or draws you supply.

- A wide menu of models
  - 3 noise distributions and 11 inefficiency distributions, supporting all combinations for production and cost frontiers.
  - Cross-sectional models allow copula dependence (except with Student-t noise) and inefficiency determinants through parameter-specific or scaling-property specifications.
  - 3 panel models: Wang and Ho (2010), Chen, Schmidt and Wang (2014), and the true random-effect model.

- GPU computing
  - Simulation-based estimation requires many draws per observation, making computation expensive.
  - Up to 39× faster than CPU in our benchmarks (NVIDIA RTX 5090) for larger problems.

- Automatic differentiation
  - Uses ForwardDiff.jl to compute gradients and Hessians automatically.
  - Provides fast, highly accurate derivatives essential for maximum likelihood estimation.

- **A complete post-estimation toolkit**
  - **Efficiency:** firm-level JLMS inefficiency and Battese–Coelli efficiency indices, with summary statistics.
  - **Marginal effects** of the inefficiency determinants on expected inefficiency.
  - **Tests:** likelihood-ratio tests, including the test of no inefficiency against OLS and boundary (mixed χ²) references; third-moment skewness tests of the OLS residuals; and residual diagnostics.
  - **Reliability checks:** convergence and boundary warnings, and an integration check that tells you whether a simulated fit used enough draws.
  - **Prediction** of frontier values, efficiency indices and marginal effects for new data.
  - **Diagnostic plots** through Plots.jl recipes.


## Installation

```julia
using Pkg
Pkg.add("SFrontiers")
```

SFrontiers.jl requires Julia 1.10 or later. For GPU computing, also install [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) and load it **before** SFrontiers.jl (`using CUDA, SFrontiers`).

## Quick Start

The example simulates a production frontier with one inefficiency determinant, estimates it by analytic MLE, and then analyzes the fit.

```julia
using SFrontiers, Optim, Random, Statistics

# Simulated data: y = x'β + v - u, with σᵤ² = exp(-1 + 0.8 z1)
Random.seed!(2026)
N = 1_000
x1, x2, z1 = randn(N), randn(N), randn(N)
u = abs.(randn(N)) .* exp.(0.5 .* (-1.0 .+ 0.8 .* z1))
v = 0.3 .* randn(N)
y = 1.0 .+ 0.5 .* x1 .+ 0.3 .* x2 .+ v .- u

X = hcat(ones(N), x1, x2)    # frontier variables, with a constant
Z = hcat(ones(N), z1)        # inefficiency determinants, with a constant

# Step 1: specify the model
spec = sfmodel_spec(
    type     = :production,
    depvar   = y,
    frontier = X,
    zvar     = Z,
    noise    = :Normal,
    ineff    = :HalfNormal,
    hetero   = [:sigma_u_sq],    # σᵤ² = exp(Z'γ)
)

# Step 2: choose the estimation method
meth = sfmodel_method(method = :MLE)

# Step 3 (optional): starting values; the default is OLS-based
init = sfmodel_init(spec = spec,
    frontier      = X \ y,
    ln_sigma_u_sq = zeros(2),
    ln_sigma_v_sq = 0.0)

# Step 4 (optional): the optimizer; the default is a Nelder–Mead warm start, then Newton
opt = sfmodel_opt(
    main_solver = Newton(),
    main_opt    = (iterations = 200, g_abstol = 1e-7))

# Step 5: estimate; prints the coefficient table and a summary of the fit
result = sfmodel_fit(spec = spec, method = meth, init = init, optim_options = opt)
```

### Post-estimation analysis

```julia
# Inference, through the StatsAPI interface
coeftable(result)                 # estimates, standard errors, z statistics, p-values
loglikelihood(result), aic(result), bic(result)

# Efficiency of each firm
result.jlms                       # JLMS inefficiency index E(u|ε)
result.bc                         # Battese–Coelli efficiency index E(exp(-u)|ε)
efficiency_summary(result)        # summary statistics of both indices

# How the determinant shifts expected inefficiency E(u)
result.marginal_mean              # sample-mean marginal effect
result.marginal                   # one row per firm (a DataFrame)

# Diagnostics
sf_skewness_test(result)          # are the OLS residuals skewed as inefficiency implies?
residual_diagnostics(result)      # residual skewness checks and the Jarque–Bera statistic

# Likelihood-ratio tests
spec0   = sfmodel_spec(type = :production, depvar = y, frontier = X,
                       noise = :Normal, ineff = :HalfNormal)
result0 = sfmodel_fit(spec = spec0, method = meth)
sf_vs_ols(result0)                # is there any inefficiency? (mixed χ² reference)
lrtest(result0, result)           # does z1 affect inefficiency?

# Prediction for new firms (here the first ten firms stand in for new data)
new = 1:10
predict(result; frontier = X[new, :], zvar = Z[new, :], depvar = y[new], what = :bc)

# Diagnostic plots
using Plots
plot(result)                      # efficiency histograms, residuals, marginal effects
```

### Beyond closed forms: simulation on the GPU

A model with Student-t noise has no closed-form likelihood, so it is estimated by MSLE. The calls are the same; only the specification and the method change.

```julia
using CUDA                        # load CUDA.jl before SFrontiers.jl
using SFrontiers, Optim

spec_t = sfmodel_spec(type = :production, depvar = y, frontier = X, zvar = Z,
                      noise = :StudentT, ineff = :HalfNormal, hetero = [:sigma_u_sq])

meth_gpu = sfmodel_method(method = :MSLE, n_draws = 2^12, GPU = true)

result_t = sfmodel_fit(spec = spec_t, method = meth_gpu,
                       integration_check = true)   # were 4,096 draws enough?
```

Without a GPU, drop `GPU = true`; the fit runs on the CPU. Setting `method = :MSLE` for the Quick Start model reproduces the MLE estimates closely, a quick way to see the accuracy of the simulation.

### Specifying the model with DataFrame columns

```julia
using DataFrames

df = DataFrame(y = y, x1 = x1, x2 = x2, z1 = z1)
df._cons = ones(nrow(df))         # a constant column

spec_df = sfmodel_spec(
    @useData(df),
    @depvar(y),
    @frontier(_cons, x1, x2),
    @zvar(_cons, z1);
    type   = :production,
    noise  = :Normal,
    ineff  = :TruncatedNormal,
    hetero = [:mu, :sigma_u_sq],   # μ and σᵤ² both depend on z1
)
```

The output tables then use the column names.

## Documentation

The [user manual](https://hungjenwang1991.github.io/SFrontiers.jl/#user-manual-for-julia-implementation) covers every function and option, the supported models, a detailed empirical example, panel models, and guidance on choosing a method, the number of draws and the GPU settings.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) for details.
