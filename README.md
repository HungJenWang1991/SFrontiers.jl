# SFrontiers.jl

[![CI](https://github.com/HungJenWang/SFrontiers.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HungJenWang/SFrontiers.jl/actions/workflows/CI.yml)

A Julia package for **stochastic frontier analysis**, supporting multiple estimation methods and a wide range of distributional assumptions.

## Features

- **Multiple estimation backends**:
  - **MLE** -- Analytic maximum likelihood (closed-form, fastest)
  - **MCI** -- Monte Carlo integration with change-of-variable transforms
  - **MSLE** -- Maximum simulated likelihood via inverse CDF
  - **Panel** -- Wang and Ho (2010) true fixed-effect panel models

- **Distributional flexibility**:
  - Noise: Normal, Student-t, Laplace
  - Inefficiency: Half-Normal, Truncated-Normal, Exponential, Weibull, Lognormal, Lomax, Rayleigh, Gamma

- **Heteroscedasticity**: Scaling function and component-specific heterogeneity
- **Efficiency indices**: JLMS and BC firm-level technical efficiency
- **Marginal effects** computation
- **DSL macros** for DataFrame-based model specification

## Installation

```julia
using Pkg
Pkg.add("SFrontiers")
```

## Quick Start

```julia
using SFrontiers, DataFrames, Optim

# Specify the model
spec = sfmodel_spec(
    depvar = y,
    frontier = X,
    noise = :Normal,
    ineff = :HalfNormal,
    type = :prod
)

# Choose estimation method
meth = sfmodel_method(method = :MLE)

# Set initial values
init = sfmodel_init(spec = spec,
    frontier = beta_ols,
    ln_sigma_sq = log(0.25),
    ln_sigma_v_sq = log(0.09))

# Configure optimizer
opt = sfmodel_opt(
    main_solver = Newton(),
    main_opt = (iterations = 200, g_abstol = 1e-7))

# Estimate
result = sfmodel_fit(spec = spec, method = meth,
    init = init, optim_options = opt)
```

### Using DataFrame DSL

```julia
spec = sfmodel_spec(
    @useData(df),
    @depvar(y),
    @frontier(cons, x1, x2),
    @zvar(cons, z1);
    noise = :Normal,
    ineff = :TruncatedNormal,
    type = :prod
)
```

## License

GPL-3.0-or-later License. See [LICENSE](LICENSE) for details.
