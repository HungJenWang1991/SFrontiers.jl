---
marp: false
---
# SFrontiers.jl: Stochastic Frontier Model Estimation: Simulation-Based and Analytic Methods


## User Manual for Julia Implementation

**Version 2**<br>
**&copy; Hung-Jen Wang**<br>
**wangh@ntu.edu.tw**

---

## Table of Contents

0. [Theoretical Presentation](#0-theoretical-presentation)
1. [Introduction](#1-introduction)
2. [Software and Hardware Requirements](#2-software-and-hardware-requirements)
3. [Installation and Dependencies](#3-installation-and-dependencies)
4. [Quick Start Example](#4-quick-start-example)
5. [Function Reference](#5-function-reference)
   - 5.1 [sfmodel_spec()](#51-sfmodel_spec)
   - 5.2 [sfmodel_method()](#52-sfmodel_method)
   - 5.3 [sfmodel_init()](#53-sfmodel_init)
   - 5.4 [sfmodel_opt()](#54-sfmodel_opt)
   - 5.5 [sfmodel_fit()](#55-sfmodel_fit)
   - 5.6 [sfmodel_MixTable() and sfmodel_ChiSquareTable()](#56-sfmodel_mixtable-and-sfmodel_chisquaretable)
6. [Supported Models](#6-supported-models)
7. [Distributions Reference](#7-distributions-reference)
   - 7.1 [Distribution Selection Guidance](#71-distribution-selection-guidance)
8. [Working with Results](#8-working-with-results)
   - 8.1 [Interpreting Estimation Results](#81-interpreting-estimation-results)
9. [Advanced Topics](#9-advanced-topics)
   - 9.1 [Copula Interpretation Guide](#91-copula-interpretation-guide)
   - 9.2 [Cost Frontier Interpretation](#92-cost-frontier-interpretation)
10. [Panel Data Models](#10-panel-data-models)

---

## 0. Theoretical Presentation

The stochastic frontier (SF) model was first proposed by Aigner et al. (1977) in which the inefficiency is assumed to have a half-normal distribution. The MLE was suggested as the estimation method, which has since then been the dominating method in the parametric SF literature.

Greene (2003) is the first well-known study that used the simulation-based method to estimate stochastic frontier models. He considered models with complex joint densities that are difficult to derive analytically, and therefore he resorted to simulation-based approaches. Since then, simulation-based MLE has been adopted widely in the econometric literature, particularly for estimating discrete choice models and panel data models.

Consider the standard stochastic frontier specification

$$
\begin{align}
y_i &= x_i'\beta + \epsilon_i, \\
\epsilon_i &= v_i - u_i,
\end{align}
$$

where $v_i$ is the noise term and $u_i \ge 0$ is the inefficiency term. The density of the composed error $\epsilon_i$ is

$$
\begin{align}
   f(\varepsilon_i \mid \theta)
     & = \int_0^\infty f_{v}(\varepsilon_i + u_i \mid \theta)\, f_u(u_i \mid \theta)\, d u_i \notag \\
     & = \mathbb{E}_{f_u(u \mid \theta)} \left[ f_{v}(\varepsilon_i + u_i \mid \theta)  \right].
\end{align}
$$

The log-likelihood of the model is

$$
\begin{align}
   \ln L & = \sum_{i=1}^N \ln \mathbb{E}_{f_u(u \mid \theta)} \left[ f_{v}(\varepsilon_i + u_i \mid \theta)  \right].
\end{align}
$$

In estimation, the expected value is approximated by its empirical counterpart using quasi-random draws $u_i^{(s)}$, $s = 1, \ldots, S$:

$$
\mathbb{E}_{f_u(u \mid \theta)} \left[ f_{v}(\varepsilon_i + u_i \mid \theta)  \right] \approx
\frac{1}{S}\sum_{s=1}^S  f_{v}(\varepsilon_i + u_i^{(s)} \mid \theta).
$$

This module provides two approaches for generating the draws $u_i^{(s)}$:

### MSLE (Maximum Simulated Likelihood Estimation)

The MSLE approach uses the **inverse CDF (quantile function)** of $f_u$ to generate draws. Given a set of Halton draws $t^{(s)} \in (0,1)$, the inefficiency draws are obtained as

$$
u_i^{(s)} = F_u^{-1}(t^{(s)} \mid \theta),
$$

where $F_u^{-1}$ is the quantile function of the inefficiency distribution.

### MCI (Monte Carlo Integration)

The MCI approach uses a **change-of-variable transformation** to map uniform draws to the support of $u$. Given a transformation function $g$ and Halton draws $t^{(s)} \in (0,1)$, the integral is rewritten as

$$
\int_0^\infty f_{v}(\varepsilon_i + u)\, f_u(u)\, du
= \int_0^1 f_{v}(\varepsilon_i + g(t, s_i))\, f_u(g(t, s_i))\, |g'(t, s_i)|\, dt,
$$

where $s_i$ is a scale parameter derived from the inefficiency distribution's parameters for observation $i$, and $g'$ is the Jacobian of the transformation. The MCI approach supports several transformation rules (see [Section 5.2](#52-sfmodel_method)).

As shown by Wang (2025), MSLE is a special case of MCI. When the quantile function is used as the transformation rule, the Jacobian reduces to $\frac{1}{f_u(g(t,s_i))}$, which cancels the density term and yields the familiar MSLE equation.

---

## 1. Introduction

This package provides a unified framework for estimating stochastic frontier (SF) models via simulation-based likelihood evaluation and, for a limited subset of models, analytic maximum likelihood estimation (MLE). The simulation-based methods -- Maximum Simulated Likelihood Estimation (MSLE) and Monte Carlo Integration (MCI) -- support a broad class of distributional and dependence specifications. For the classical Normal–HalfNormal, Normal–TruncatedNormal, and Normal–Exponential models, closed-form analytic MLE is also available.

### Key features include:

* **Three estimation methods:** Maximum Simulated Likelihood Estimation (MSLE), Monte Carlo Integration (MCI), and Analytic Maximum Likelihood Estimation (MLE).

  - **MSLE** and **MCI** are simulation-based, using quasi Monte Carlo (QMC) draws with Halton sequences. As shown in Wang (2025), MSLE is a special case of MCI. The two names are retained here following conventions in the literature. They support all 8 inefficiency distributions, up to 3 noise distributions, and copula dependence.
  - **MLE** uses closed-form log-likelihoods — no simulation draws are needed, making it fast and exact. It is available for a limited subset of models: Normal noise with HalfNormal, TruncatedNormal, or Exponential inefficiency (no copula).
* **Cross-sectional and panel data:** In addition to cross-sectional models, the module supports several panel stochastic frontier models:

  - `datatype=:panel_TFE` — Wang and Ho (2010) true fixed-effect model. Supports MCI, MSLE, and MLE (HalfNormal/TruncatedNormal only for MLE).
  - `datatype=:panel_TFE_CSW` — Chen, Schmidt, and Wang (2014) fixed-effect model. MLE only, HalfNormal only.
  - `datatype=:panel_TRE` — True random-effect model. MLE only, HalfNormal or TruncatedNormal.

  See [Section 10](#10-panel-data-models) for details.
* **A large set of noise ($v$) and inefficiency ($u$) combinations:** For cross-sectional models (`datatype=:cross_sectional`), any combinations between the following sets are supported:

  - $v$: Normal, Student T, and Laplace (cross-sectional); Normal only (panel)
  - $u$: Half Normal, Truncated Normal, Exponential, Weibull, Lognormal, Lomax, Rayleigh, and Gamma distributions.

  MCI supports all 8 inefficiency distributions (including Gamma, which is MCI-only). MSLE supports all except Gamma. MLE is limited to HalfNormal, TruncatedNormal, and Exponential with Normal noise.

  For panel data models, the noise ($v$) distribution is fixed at the Normal distribution. The simulation-based panel model (`datatype=:panel_TFE`) accommodates all 8 $u$ distributions via MCI/MSLE. The MLE-only panel models (`datatype=:panel_TFE_CSW`, `:panel_TRE`) support a subset (see [Section 10](#10-panel-data-models)).
* **Copula dependence (cross-sectional only):** Gaussian, Gumbel, Clayton, and 90°-rotated Clayton copula support for modeling dependence between noise $v$ and inefficiency $u$, with automatic computation of the dependence parameter and Kendall's $\tau$.

  - Exception: When $v$ is Student T, the copula is not supported.
  - Note: Copula models are not available for panel data.
* **Heteroscedastic inefficiency specifications:** For cross-sectional models, the parameters of the inefficiency distribution can be modeled as functions of covariates, consistent with common practice in modern SF applications. Cross-sectional models also support the **scaling property model** $u_i = h(\mathbf{z}_i) \cdot u_i^*$, where $h(\mathbf{z}_i) = \exp(\mathbf{z}_i'\boldsymbol{\delta})$ and $u_i^*$ follows a homoscedastic base distribution. For panel data models, heterogeneity enters via the same scaling function $h(z_{it}) = \exp(z_{it}'\delta)$.
* **Compute important statistics and quantities:** For every model estimation, it automatically computes important statistics and quantities for post-estimation analysis. For instance, the inefficiency (JLMS) and efficiency (BC) index, the marginal effects of the determinants of inefficiency, the corresponding OLS loglikelihood values and the skewness of OLS residuals.
* **CPU or GPU execution:** Numerical/simulation methods typically require a large number of draws for accuracy, making computation costly on the CPU. GPU execution substantially reduces runtime in such settings and makes applications practical.
* **Automatic differentiation (AD):** The module uses AD to compute derivatives for gradient-based optimization. For the differentiable computations, AD is algebraically equivalent to analytic differentiation and typically attains near machine-precision accuracy in floating-point arithmetic. AD is significantly more reliable than finite-difference methods. The improved accuracy is especially important for maintaining numerical stability in challenging optimization problems.

### Estimation proceeds in five steps:

1. **Specify** the model using `sfmodel_spec()`.
2. **Choose** the estimation method using `sfmodel_method()`.
3. **Initialize** parameters using `sfmodel_init()`.
4. **Configure** optimization using `sfmodel_opt()`.
5. **Estimate** the model using `sfmodel_fit()`.

---

## 2. Software and Hardware Requirements

### 2.1 Software

* **Julia,** which is a free and open-source programming language. A recent stable release is recommended. Julia 1.10 or later is required for GPU execution via CUDA.jl. The `Optim` package (used for maximum likelihood estimation) requires version 2.0 or above. ([julialang.org](https://julialang.org/))
* **NVIDIA driver and CUDA.jl (optional).** GPU execution requires a compatible NVIDIA driver to be installed on the system. The Julia package CUDA.jl provides the programming interface and can be installed on any machine, but it relies on the NVIDIA driver at run time. The package also supports CPU-only execution, which does not require the driver or CUDA.jl. ([cuda.juliagpu.org](https://cuda.juliagpu.org/stable/installation/overview/))

### 2.2 Hardware

* **CPU execution.** Any modern personal computer supported by Julia is sufficient for CPU-only estimation. Runtime will scale with the number of observations, the number of parameters, and the number of QMC draws.
* **GPU execution (optional).** Requires NVIDIA GPU. The GPU memory requirements increase with the number of draws and the degree of batching. In practice, larger VRAM allows larger draw blocks and reduces the need for chunking. In estimation, the number of batches/chunks is set by the `chunks` option in `sfmodel_method()`.

---

## 3. Installation and Dependencies

SFrontiers.jl is a registered Julia package. Install it from the Julia package manager:

```julia
using Pkg
Pkg.add("SFrontiers")
```


To load the package:

```julia
using SFrontiers
```

### GPU Support (Optional)

CUDA.jl is **not** a required dependency. It is only needed if you want GPU-accelerated estimation via `sfmodel_method(method=:MCI, GPU=true)` or `sfmodel_method(method=:MSLE, GPU=true)`. MLE estimation does not use GPU.

> **Important:** If you plan to use GPU features, CUDA.jl must be loaded **before** SFrontiers.jl. This is because SFrontiers conditionally detects CUDA at load time and registers GPU function overloads only if CUDA is already available.

```julia
# Correct order for GPU usage:
using CUDA          # Load CUDA first
using SFrontiers    # Then load SFrontiers

# CPU-only usage (no CUDA needed):
using SFrontiers
```

If CUDA is loaded after SFrontiers, GPU features will not be available and you will need to restart Julia and load them in the correct order.

---

## 4. Quick Start Example

```julia
using CSV, DataFrames
using CUDA
using SFrontiers

# Load data
df = CSV.read("sampledata.csv", DataFrame)
y = df.y
X = hcat(ones(length(y)), df.x1, df.x2)  # Frontier variables (include constant)
Z = hcat(ones(length(y)), df.z1)          # Inefficiency determinants

# Step 1: Specify the model
myspec = sfmodel_spec(
    depvar = y,                # dependent variable; a vector
    frontier = X,              # matrix of X vars in frontier
    zvar = Z,                  # matrix of Z vars for inefficiency
    noise = :Normal,           # dist of v
    ineff = :TruncatedNormal,  # dist of u
    hetero = [:mu, :sigma_sq], # heteroscedastic of u's μ & σ²
    type = :production         # for production frontier; alt. :cost
)

# Step 2: Choose the estimation method
mymeth = sfmodel_method(
    method = :MSLE,           # :MLE, :MSLE, or :MCI
    n_draws = 2^12 - 1,       # number of Halton draws
    GPU = true,               # use GPU; default is `false` thus CPU
    chunks = 10               # for GPU memory management; the default
)

# Step 3: Set initial values
myinit = sfmodel_init(
    spec = myspec,              # from sfmodel_spec()
    frontier = X \ y,           # OLS estimates
    mu = zeros(size(Z, 2)),     # μ coefficients
    ln_sigma_sq = zeros(size(Z, 2)),  # ln(σ²) coefficients
    ln_sigma_v_sq = [0.0]       # ln(σᵥ²)
)

# Step 4: Configure optimization for Julia's Optim
myopt = sfmodel_opt(
    warmstart_solver = NelderMead(),  # first stage "warm-up" estimation
    warmstart_opt = (iterations = 200, g_abstol = 1e-5),
    main_solver = Newton(),           # 2nd stage "main" estimation
    main_opt = (iterations = 100, g_abstol = 1e-8)
)

# Step 5: Estimate the model
result = sfmodel_fit(
    spec = myspec,         # from sfmodel_spec()
    method = mymeth,       # from sfmodel_method()
    init = myinit,         # from sfmodel_init()
    optim_options = myopt, # from sfmodel_opt()
    marginal = true,       # marginal effects of Z; the default
    show_table = true      # print estimation table; the default
)

# Access results
println("coefficients: ", result.coeff)
println("BC efficiency index: ", result.bc)
println("Marginal Effects of Z: ", result.marginal)
```

> **MLE alternative:** For models with Normal noise and HalfNormal, TruncatedNormal, or Exponential inefficiency, analytic MLE is available and requires no simulation settings:
> ```julia
> mymeth_mle = sfmodel_method(method = :MLE)  # no draws, GPU, or chunks needed
> result_mle = sfmodel_fit(spec = myspec, method = mymeth_mle, init = myinit, optim_options = myopt)
> ```

---

## 5. Function Reference

### 5.1 sfmodel_spec()

The `sfmodel_spec()` function creates a model specification object that encapsulates all data, distributional assumptions, and metadata required for estimation. It is independent of the choice of estimation method.

#### Syntax

`sfmodel_spec()` — define the model specification.

**Required:**

- `depvar` — dependent variable vector
- `frontier` — covariate matrix for the frontier equation
- `noise` — distribution of noise term $v$
- `ineff` — distribution of inefficiency term $u$

**Optional:**

- `datatype` — `:cross_sectional` (default), `:panel_TFE`, `:panel_TFE_CSW`, or `:panel_TRE`
- `type` — `:production` (default) or `:cost`
- `zvar` — covariate matrix for heteroscedasticity / scaling function
- `copula` — copula for $v$–$u$ dependence (default: none)
- `hetero` — parameters of $u$ allowed to be heteroscedastic (default: none)
- `T_periods` — number of time periods per firm (panel, balanced)
- `id` — unit identifier column (panel, unbalanced)
- `varnames` — variable names for output tables (default: auto-generated)
- `eqnames` — equation block names (default: auto-generated)
- `eq_indices` — equation boundary indices (default: auto-generated)

#### Arguments


| Argument     | Type           | Description                                                                                                                                                                                                                                                                                                        |
| ------------ | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `datatype`   | Symbol         | Data type. `:cross_sectional` (default), `:panel_TFE` (Wang and Ho 2010 true fixed-effect), `:panel_TFE_CSW` (Chen, Schmidt, and Wang 2014, MLE only), or `:panel_TRE` (true random-effect, MLE only). See [Section 10](#10-panel-data-models).                                                                     |
| `type`       | Symbol         | Frontier type. Use`:production` or `:prod` for production frontier ($\varepsilon_i = v_i - u_i$), and `:cost` for cost frontier ($\varepsilon_i = v_i + u_i$).                                                                                                                                                     |
| `depvar`     | Vector         | Dependent variable. Cross-sectional: $N$ observations. Panel: $N \times T$ stacked by firm.                                                                                                                                                                                                                         |
| `frontier`   | Matrix         | Covariate matrix for the frontier equation, dimension$N\times K$. Accepts a `Matrix` or a list form `[v1, v2, ...]` that is internally assembled into a matrix. Cross-sectional: include a column of ones (`1`) for intercept. **Panel: do NOT include a constant column** (within-demeaning eliminates it).       |
| `zvar`       | Matrix         | Covariate matrix. Cross-sectional: for heteroscedasticity equations, dimension $N\times L$; include a column of ones if an intercept is required. When `hetero=:scaling`, the `zvar` matrix supplies the $\mathbf{z}_i$ variables for the scaling function $h(\mathbf{z}_i)=\exp(\mathbf{z}_i'\boldsymbol{\delta})$; **do NOT include a constant column** (for identification). Panel: for scaling function $h(z)=\exp(z'\delta)$, dimension $NT \times L$; **do NOT include a constant column**. Optional for both. |
| `noise`      | Symbol         | Distribution of the noise term $v$. Cross-sectional: `:Normal`, `:StudentT`, `:Laplace`. Panel: `:Normal` only. See [Section 6](#6-supported-models).                                                                                                                                                               |
| `ineff`      | Symbol         | Distribution of the inefficiency term$u$. Supported options: `:HalfNormal`, `:TruncatedNormal`, `:Exponential`, `:Weibull`, `:Lognormal`, `:Lomax`, `:Rayleigh`, and `:Gamma`. See [Section 6](#6-supported-models). Note: `:Gamma` is MCI only; `method=:MLE` supports only `:HalfNormal`, `:TruncatedNormal`, `:Exponential`. |
| `copula`     | Symbol         | *Cross-sectional only.* Copula for dependence between $v$ and $u$. Options: `:None` (default), `:Gaussian`, `:Clayton`, `:Clayton90`, `:Gumbel`. Not available with panel datatypes.                                                                                                                               |
| `hetero`     | Vector{Symbol} or Symbol | *Cross-sectional only.* Parameters of the distributional specification that are allowed to be heteroscedastic (e.g., `[:mu, :sigma_sq]`), **or** the symbol `:scaling` to activate the scaling property model. Not available with panel datatypes. See [Section 7](#7-distributions-reference) and [Section 9.5](#scaling-property-model-cross-sectional). |
| `T_periods`  | Int/Nothing    | *Panel only.* Number of time periods per firm (balanced panel). Required when using a panel `datatype` and `id` is not provided; use only for balanced panel.                                                                                                                                                      |
| `id`         | Vector/Nothing | *Panel only.* Unit identifier column (unbalanced panel). Required when using a panel `datatype` and `T_periods` is not provided. Data must be grouped by unit.                                                                                                                                                     |
| `varnames`   | Vector{String} | Variable names used in output tables. If`nothing` (default), names are generated automatically.                                                                                                                                                                                                                    |
| `eqnames`    | Vector{String} | Equation block names (e.g.,`["frontier", "mu", "ln_sigma_u_sq"]`). If `nothing` (default), names are generated from `ineff`.                                                                                                                                                                                       |
| `eq_indices` | Vector{Int}    | Equation boundary indices. If`nothing` (default), auto-generated based on the model structure.                                                                                                                                                                                                                     |

---

#### Return Value

Returns a `UnifiedSpec{T}` struct containing the model specifications internally. For cross-sectional models, it holds MCI, MSLE, and MLE backend specs (as applicable). For `panel_TFE`, it holds both Panel (MCI/MSLE) and MLE specs. For `panel_TFE_CSW` and `panel_TRE`, it holds MLE spec only.

#### Example: Basic Specification

```julia
spec1 = sfmodel_spec(  # homoscedastic, no Z
    depvar = y,
    frontier = X,
    noise = :Normal,
    ineff = :TruncatedNormal,
)

spec2 = sfmodel_spec(  # heteroscedastic
    depvar = y,
    frontier = X,
    zvar = Z,
    noise = :Normal,
    ineff = :TruncatedNormal,
    hetero = [:mu]
)
```

#### Example: Full Specification with Custom Names

```julia
spec = sfmodel_spec(
    depvar = y,
    frontier = X,
    zvar = Z,
    noise = :Normal,
    ineff = :TruncatedNormal,
    hetero = [:mu, :sigma_sq],
    varnames = ["_cons", "output", "capital", "_cons", "age", "size", "_cons", "age", "size", "_cons"],
    eqnames = ["frontier", "mu", "ln_sigma_u_sq", "ln_sigma_v_sq"]
)
```

#### Example: Scaling Property Model

The scaling property model uses `hetero = :scaling` (a `Symbol`, not a `Vector{Symbol}`). In this specification, `zvar` provides the environmental variables $\mathbf{z}_i$ for the scaling function $h(\mathbf{z}_i) = \exp(\mathbf{z}_i'\boldsymbol{\delta})$, and the inefficiency distribution parameters remain scalar (homoscedastic). The `zvar` matrix must **not** contain a constant column (for identification; see [Section 9.5](#scaling-property-model-cross-sectional)).

```julia
# Scaling property model (keyword form)
# Z_nocons must NOT contain a constant column
spec = sfmodel_spec(
    depvar = y,
    frontier = X,           # include constant as usual
    zvar = Z_nocons,        # environmental variables (no constant!)
    noise = :Normal,
    ineff = :HalfNormal,
    hetero = :scaling,      # scaling property model
    type = :production
)

# Scaling + copula (allowed)
spec = sfmodel_spec(
    depvar = y,
    frontier = X,
    zvar = Z_nocons,
    noise = :Normal,
    ineff = :Exponential,
    hetero = :scaling,
    copula = :Clayton,
    type = :production
)
```

> **Notes on scaling property model:**
> - `hetero = :scaling` cannot be combined with heteroscedastic parameters (`:mu`, `:sigma_sq`, etc.).
> - `zvar` is required and must not contain a constant column.
> - All 8 inefficiency distributions are supported. `:Gamma` requires `method = :MCI`.
> - Copula models are compatible with `hetero = :scaling`.
> - Both `:MSLE` and `:MCI` estimation methods are supported (`:Gamma` is MCI only).

#### DSL-Style Specification with DataFrames

The function also supports a macro-based interface using DataFrames. Instead of passing data vectors and matrices directly, you specify **column names** of the DataFrame in the macros (`@depvar`, `@frontier`, `@zvar`), and the data is extracted automatically. The macros can appear in **any order** — they are identified by type, not by position.

```julia
# Add constant column
df._cons = ones(nrow(df))

# DSL-style specification — arguments are column names, not data
spec = sfmodel_spec(
    @useData(df),   # specify the DataFrame
    @depvar(yvar),
    @frontier(_cons, Lland, PIland, Llabor),
    @zvar(_cons, age, school),
    noise = :Normal,
    ineff = :TruncatedNormal,
    hetero = [:mu, :sigma_sq]
)
```

#### Example: Scaling Property Model (DSL)

```julia
# DSL-style scaling property model — zvar columns must NOT include a constant
spec = sfmodel_spec(
    @useData(df),
    @depvar(yvar),
    @frontier(_cons, Lland, PIland, Llabor),
    @zvar(age, school),           # no constant column!
    noise = :Normal,
    ineff = :HalfNormal,
    hetero = :scaling
)
```

#### Example: Panel Data Specification

```julia
# Panel TFE — Wang and Ho (2010) true fixed-effect (balanced, keyword form)
# Supports method=:MCI, :MSLE, and :MLE (HalfNormal/TruncatedNormal for MLE)
spec = sfmodel_spec(
    datatype = :panel_TFE,  # Wang and Ho 2010 panel model
    type = :production,
    depvar = y,             # N*T stacked by firm
    frontier = X,           # NT x K (no constant)
    zvar = Z,               # NT x L (no constant)
    noise = :Normal,
    ineff = :TruncatedNormal,
    T_periods = 10,         # balanced panel with 10 periods
)

# Panel TFE — unbalanced (keyword form)
spec = sfmodel_spec(
    depvar = y,
    frontier = X,
    zvar = Z,
    noise = :Normal,
    ineff = :HalfNormal,
    datatype = :panel_TFE,
    id = firm_ids           # unit identifier column
)

# Panel TFE_CSW — Chen, Schmidt, and Wang (2014) fixed-effect (MLE only, HalfNormal only)
spec = sfmodel_spec(
    depvar = y,
    frontier = X,
    noise = :Normal,
    ineff = :HalfNormal,
    datatype = :panel_TFE_CSW,
    id = firm_ids
)

# Panel TRE — True random-effect (MLE only, HalfNormal or TruncatedNormal)
spec = sfmodel_spec(
    depvar = y,
    frontier = X,
    zvar = Z,
    noise = :Normal,
    ineff = :HalfNormal,
    datatype = :panel_TRE,
    id = firm_ids
)

# Balanced panel (DSL form)
spec = sfmodel_spec(
    @useData(df),
    @depvar(yvar),
    @frontier(Lland, PIland, Llabor),
    @zvar(age, school),
    noise = :Normal,
    ineff = :HalfNormal,
    datatype = :panel_TFE,
    T_periods = 10
)

# Unbalanced panel (DSL form with @id)
spec = sfmodel_spec(
    @useData(df),
    @depvar(yvar),
    @frontier(Lland, PIland, Llabor),
    @zvar(age, school),
    @id(firm),
    noise = :Normal,
    ineff = :HalfNormal,
    datatype = :panel_TFE
)
```

> **Important:** Panel models do not support `copula` or `hetero` arguments. In the panel model, heterogeneity enters through the scaling function $h(z_{it}) = \exp(z_{it}'\delta)$, and the `frontier` and `zvar` matrices must NOT include a constant column. See [Section 10](#10-panel-data-models) for details.
>
> For cross-sectional models, the scaling property model is activated by `hetero = :scaling`. See [Section 9.5](#scaling-property-model-cross-sectional) for the mathematical formulation and usage details.

---

### 5.2 sfmodel_method()

The `sfmodel_method()` function specifies the estimation method and its computational settings (number of draws, GPU usage, etc.). This is separate from `sfmodel_spec()` so that the same model specification can be estimated under different methods.

#### Syntax

`sfmodel_method()` — choose the estimation method and simulation settings.

**Required:**

- `method` — `:MSLE`, `:MCI`, or `:MLE`

**Optional (MCI/MSLE only — ignored with a warning for MLE):**

- `transformation` — only when `method=:MCI`; transformation rule (default: distribution-specific)
- `draws` — user-supplied draws as a 1 × D row matrix (default: auto-generated Halton)
- `n_draws` — number of Halton draws per observation (default: 1024)
- `multiRand` — observation-specific Halton draws (default: true)
- `GPU` — use GPU acceleration (default: false, i.e., CPU)
- `chunks` — number of data chunks for memory management (default: 10)
- `distinct_Halton_length` — max length of distinct Halton sequence (default: 2¹⁵ − 1) used in the entire NxD matrix.

#### Arguments


| Argument                 | Type           | Description                                                                                                                                                                                                                                                                           |
| ------------------------ | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `method`                 | Symbol         | Estimation method. Use `:MSLE` for Maximum Simulated Likelihood Estimation, `:MCI` for Monte Carlo Integration, or `:MLE` for analytic Maximum Likelihood Estimation (available for a limited subset of models). Required. MLE does not use any simulation arguments below.                                                       |
| `transformation`         | Symbol/Nothing | **MCI only.** Transformation rule for mapping uniform draws to inefficiency values. Options: `:expo_rule`, `:logistic_1_rule`, `:logistic_2_rule`, or `nothing` for distribution-specific defaults. Ignored with a warning if `method=:MSLE`.                         |
| `draws`                  | Matrix         | Draws for Monte Carlo integration as a**1 x D row matrix**. Use `reshape(your_draws, 1, length(your_draws))` to convert a vector. If `nothing` (default), Halton draws are auto-generated with correct shape.                                                                         |
| `n_draws`                | Int            | Number of Halton draws per observation. Default: 1024 for both MCI and MSLE. Used only when`draws` is not provided. When `multiRand=true`, must be $\leq$ `distinct_Halton_length`.                                                                                                   |
| `multiRand`              | Bool           | Whether each observation gets different Halton draws.`true` (default) generates an N x D wrapped Halton matrix where each observation uses different consecutive draws. `false` uses the original 1 x D shared draws. When `true`, `n_draws` must be $\leq$ `distinct_Halton_length`. |
| `GPU`                    | Bool           | Whether to use GPU computing.`false` (default) uses CPU, and `true` uses GPU (requires `using CUDA`).                                                                                                                                                                                 |
| `chunks`                 | Int            | Number of chunks for processing data, most useful for GPU memory management when`GPU = true` (default: `10`).                                                                                                                                                                         |
| `distinct_Halton_length` | Int            | Maximum length of the distinct Halton sequence generated for`multiRand=true` mode (default: `2^15-1 = 32767`). Increase this if you need `n_draws` larger than the default limit. See [Observation-Specific Halton Draws](#observation-specific-halton-draws-multirand).              |

#### About `GPU` and `chunks` options

The `chunks` option controls how the computation is split for memory management. It works for both CPU and GPU computation, but is particularly essential for GPU computing (`GPU=true`). When `chunks=1`, all N observations are processed simultaneously, creating an N x D matrix (where D is the number of Halton draws). For large datasets (big N) or a large `n_draws` (big D), this matrix may exceed available GPU memory, creating an estimation bottleneck.

Setting `chunks` to a value greater than 1 (e.g., `chunks=10`) splits the observations into smaller batches, creating matrices of size (N/chunks) x D. Each batch is processed sequentially while accumulating the log-likelihood. This reduces peak memory usage, allowing larger datasets and `n_draws` to fit in GPU memory at the expense of slightly increased computation overhead due to the splitting and looping.

In Windows, users may use Task Manager to monitor the memory usage and adjust `chunks` to avoid bottlenecks.

#### Transformation Rules (MCI Only)

When `method=:MCI`, the `transformation` option controls the change-of-variable mapping from uniform draws $t \in (0,1)$ to inefficiency values $u \geq 0$. If `nothing`, a distribution-specific default is used.


| Rule               | Formula                   | Jacobian      | Default for                            |
| ------------------ | ------------------------- | ------------- | -------------------------------------- |
| `:expo_rule`       | $u = s \cdot (-\ln(1-t))$ | $s/(1-t)$     | Exponential, Weibull, Gamma, Rayleigh  |
| `:logistic_1_rule` | $u = s \cdot t/(1-t)$     | $s/(1-t)^2$   | HalfNormal, TruncatedNormal, Lognormal, Lomax |
| `:logistic_2_rule` | $u = s \cdot (t/(1-t))^2$ | $2st/(1-t)^3$ | --                                     |

Here $s$ is a scale parameter derived from the inefficiency distribution's parameters. When heteroscedasticity is specified, $s$ becomes observation-specific ($s_i$). The scale parameter for each distribution is:


| Distribution    | Scale$s$   | Meaning                                                          |
| --------------- | ---------- | ---------------------------------------------------------------- |
| HalfNormal      | $\sigma$   | Standard deviation of$N^+(0, \sigma^2)$                          |
| TruncatedNormal | $\sigma_u$ | Standard deviation of$N^+(\mu, \sigma_u^2)$                      |
| Exponential     | $\sqrt{\lambda}$  | $\sqrt{\lambda}$, where $\lambda = \text{Var}(u)$              |
| Weibull         | $\lambda$  | Scale parameter of$\text{Weibull}(\lambda, k)$                   |
| Lognormal       | $\sigma$   | Log-scale standard deviation of$\text{LogNormal}(\mu, \sigma^2)$ |
| Lomax           | $\lambda$  | Scale parameter of$\text{Lomax}(\alpha, \lambda)$                |
| Rayleigh        | $\sigma$   | Scale parameter of$\text{Rayleigh}(\sigma)$                      |
| Gamma           | $\theta$   | Scale parameter of$\text{Gamma}(k, \theta)$                      |

#### Return Value

Returns a method specification struct that encodes both the estimation method and computational settings.

#### Example

```julia
# MSLE with default settings
meth1 = sfmodel_method(method = :MSLE)

# MCI with custom draws and GPU
meth2 = sfmodel_method(
    method = :MCI,
    n_draws = 2^12 - 1,
    GPU = true,
    chunks = 10,
    transformation = :logistic_1_rule
)

# Larger Halton pool for multiRand mode
meth3 = sfmodel_method(
    method = :MSLE,
    n_draws = 50000,
    distinct_Halton_length = 2^16 - 1  # 65535
)

# MLE — analytic, no simulation parameters needed (limited model support)
meth0 = sfmodel_method(method = :MLE)
```

> **Note:** If simulation arguments (`transformation`, `draws`, `n_draws`, `GPU`) are passed with `method=:MLE`, a warning is issued and they are ignored.

---

### 5.3 sfmodel_init()

The `sfmodel_init()` function creates an initial value vector for optimization. The initial-value vector depends only on the model specification (distribution choice), not on the estimation method. It supports two usage modes: full vector mode and component mode.

#### Syntax

`sfmodel_init()` — set initial values for optimization.

**Required:**

- `spec` — model specification from `sfmodel_spec()`

**Optional (full vector mode):**

- `init` — a single vector of all initial values

**Optional (component mode):**

- `frontier` — frontier equation coefficients
- `scaling` — scaling function coefficients δ (when `hetero = :scaling`)
- `mu` — μ coefficients
- `ln_sigma_sq` — ln(σ²) coefficients
- `ln_sigma_v_sq` — ln(σᵥ²)
- `ln_nu_minus_2` — ln(ν − 2), for Student T noise
- `ln_b` — ln(b), for Laplace noise
- `ln_lambda` — ln(λ), for Exponential
- `ln_k` — ln(k), for Weibull shape
- `ln_lambda` — ln(λ), for Lomax
- `ln_alpha` — ln(α), for Lomax
- `ln_theta` — ln(θ), for Gamma
- `theta_rho` — copula dependence parameter
- `message` — print summary of initial values (default: true)

#### Arguments


| Argument        | Type         | Description                                                                                                                           |
| --------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `spec`          | UnifiedSpec  | Model specification returned by`sfmodel_spec()`. Required.                                                                            |
| `init`          | Vector/Tuple | Complete initial-value vector (or tuple). If`init` is provided, all other component initial-value arguments are ignored.              |
| `frontier`      | Vector/Tuple | Initial values for the frontier coefficients ($K$ elements).                                                                          |
| `scaling`       | Vector/Tuple | Initial values for the scaling function coefficients $\boldsymbol{\delta}$ ($L$ elements, one per column of `zvar`). Used when `hetero = :scaling`. |
| `mu`            | Vector/Tuple | Initial values for$\mu$ (used by `TruncatedNormal` and `Lognormal` inefficiency specifications).                                      |
| `ln_sigma_sq`   | Vector/Tuple | Initial values for$\ln(\sigma^2)$ (used by `TruncatedNormal`, `HalfNormal`, `Lognormal`, and `Rayleigh` inefficiency specifications). |
| `ln_sigma_v_sq` | Vector/Tuple | Initial values for$\ln(\sigma_v^2)$ (used when the noise distribution is `Normal` or `StudentT`).                                     |
| `ln_nu_minus_2` | Vector/Tuple | Initial values for$\ln(\nu-2)$ (used when the noise distribution is `StudentT`).                                                      |
| `ln_b`          | Vector/Tuple | Initial values for$\ln(b)$ (used when the noise distribution is `Laplace`).                                                           |
| `ln_lambda`     | Vector/Tuple | Initial values for$\ln(\lambda)$ (used by `Exponential` and `Weibull` inefficiency specifications).                                   |
| `ln_k`          | Vector/Tuple | Initial values for$\ln(k)$ (used by `Weibull` and `Gamma` inefficiency specifications).                                               |
| `ln_lambda`     | Vector/Tuple | Initial values for$\ln(\lambda)$ (used by `Lomax` inefficiency specifications).                                                        |
| `ln_alpha`      | Vector/Tuple | Initial values for$\ln(\alpha)$ (used by `Lomax` inefficiency specifications).                                                        |
| `ln_theta`      | Vector/Tuple | Initial values for$\ln(\theta)$ (used by `Gamma` inefficiency specifications; MCI only).                                              |
| `theta_rho`     | Vector/Tuple | Initial value for the copula parameter$\theta_\rho$ (used when `copula` $\neq$ `:None`).                                              |
| `message`       | Bool         | If`true`, print a warning when `init` overrides the component initial-value arguments.                                                |

#### Return Value

Returns a `Vector{Float64}` containing the initial parameter values in the correct order for optimization.

#### Mode 1: Full Vector Mode

Provide the complete parameter vector directly:

```julia
myinit = sfmodel_init(
    spec = myspec,
    init = [0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0]
)
```

#### Mode 2: Component Mode

Specify initial values by parameter group. The required groups depend on the model specification. The ordering of equation blocks is irrelevant; the program maps each block to the correct position in the coefficient vector automatically.

```julia
# Normal + TruncatedNormal with hetero = [:mu]
myinit = sfmodel_init(
    spec = myspec,
    frontier = [0.5, 0.3, 0.2],     # frontier coefficients
    mu = [0.1, 0.1, 0.1],           # mu of truncated normal
    ln_sigma_sq = [0.0],            # log(sigma_u^2) of truncated normal
    ln_sigma_v_sq = [0.0]           # log(sigma_v^2) of normal
)

# alternatively,
myinit2 = sfmodel_init(
    spec = myspec,
    frontier = X \ y,               # OLS coefficients
    mu = 0.1*ones(size(Z, 2)),      # mu of truncated normal
    ln_sigma_sq = [0.0],            # log(sigma_u^2) of truncated normal
    ln_sigma_v_sq = (0.0)           # log(sigma_v^2) of normal
)
```

#### Scaling Property Initial Values

When `hetero = :scaling`, provide the `scaling` keyword with initial values for the $\boldsymbol{\delta}$ coefficients (one per column of `zvar`). All inefficiency distribution parameters are scalar:

```julia
# Normal + HalfNormal with scaling property
myinit = sfmodel_init(
    spec = myspec,
    frontier = X \ y,                   # OLS estimates
    scaling = zeros(size(Z_nocons, 2)), # δ coefficients (one per z-variable)
    ln_sigma_sq = [0.0],               # scalar base parameter
    ln_sigma_v_sq = [0.0]              # noise variance
)
```

#### Panel Initial Values

When `spec.datatype == :panel_TFE`, the function dispatches to the Panel backend. When `spec.datatype` is `:panel_TFE_CSW` or `:panel_TRE`, it dispatches to the MLE backend. The keyword arguments differ from cross-sectional:


| Argument        | Description                                                           |
| --------------- | --------------------------------------------------------------------- |
| `frontier`      | Initial values for frontier coefficients ($K$ elements)               |
| `delta`         | Initial values for scaling function$h(z)$ coefficients ($L$ elements) |
| `mu`            | TruncatedNormal, Lognormal: $\mu$ (scalar)                             |
| `ln_sigma_u_sq` | HalfNormal, TruncatedNormal: $\ln(\sigma_u^2)$ (scalar)                |
| `ln_sigma_v_sq` | All distributions: $\ln(\sigma_v^2)$ (scalar)                          |
| `ln_lambda`     | Exponential, Weibull: $\ln(\lambda)$ (scalar)                          |
| `ln_k`          | Weibull, Gamma: $\ln(k)$ (scalar)                                      |
| `ln_sigma_sq`   | Lognormal, Rayleigh: $\ln(\sigma^2)$ (scalar)                          |
| `ln_lambda`     | Lomax: $\ln(\lambda)$ (scalar)                                         |
| `ln_alpha`      | Lomax: $\ln(\alpha)$ (scalar)                                          |
| `ln_theta`      | Gamma: $\ln(\theta)$ (scalar)                                          |

```julia
# Panel TFE: Normal + HalfNormal
myinit = sfmodel_init(
    spec = myspec,                # spec with datatype=:panel_TFE
    frontier = X_tilde \ y_tilde, # OLS on demeaned data (auto-computed if omitted)
    delta = [0.1, 0.1],          # scaling function coefficients
    ln_sigma_u_sq = 0.1,         # scalar
    ln_sigma_v_sq = 0.1          # scalar
)
```

**Note:** In panel models, all inefficiency distribution parameters are **scalar** (not heteroscedastic). If `frontier` and `delta` are omitted, OLS-based defaults are used automatically.

> **Panel TFE + MLE:** When `datatype=:panel_TFE` and `method=:MLE`, `sfmodel_fit` automatically uses MLE's own OLS-based defaults for initial values (the Panel-backend init is not passed to MLE). You may still call `sfmodel_init` for MCI/MSLE estimation of the same spec.

#### Input Format Flexibility

All parameter arguments accept vectors, row vectors, or tuples:

```julia
# All equivalent
frontier = [0.5, 0.3, 0.2]    # Vector
frontier = [0.5 0.3 0.2]      # Row vector (1x3 matrix)
frontier = (0.5, 0.3, 0.2)    # Tuple
```

---

### 5.4 sfmodel_opt()

The `sfmodel_opt()` function specifies the optimization options. Solvers and options are passed directly to Julia's `Optim.jl` interface, so any solver (e.g., `NelderMead()`, `BFGS()`, `Newton()`) and any `Optim.Options` keyword (e.g., `iterations`, `g_abstol`, `show_trace`) are permissible. All three estimation methods (MCI, MSLE, and MLE) use the same optimizer interface, so no `method` argument is needed. The function supports a two-stage approach: a derivative-free solver in the first stage (*warmstart*) to refine the initial values (which is often very useful for highly nonlinear models), followed by a gradient-based solver in the second stage (*main*) for accurate convergence.

#### Syntax

`sfmodel_opt()` — configure optimization solvers and options.

**Required:**

- `main_solver` — main optimizer, e.g., `Newton()`, `BFGS()`
- `main_opt` — main optimizer options as a NamedTuple

**Optional:**

- `warmstart_solver` — warmstart optimizer, e.g., `NelderMead()`
- `warmstart_opt` — warmstart options as a NamedTuple

#### Arguments


| Argument           | Description                                                                   | Required |
| ------------------ | ----------------------------------------------------------------------------- | -------- |
| `warmstart_solver` | Warmstart optimizer, e.g.,`NelderMead()`, `BFGS()`                            | No       |
| `warmstart_opt`    | Warmstart options as a NamedTuple, e.g.,`(iterations = 400, g_abstol = 1e-5)` | No       |
| `main_solver`      | Main optimizer, e.g.,`Newton()`, `BFGS()`                                     | Yes      |
| `main_opt`         | Main options as a NamedTuple, e.g.,`(iterations = 2000, g_abstol = 1e-8)`     | Yes      |

**Common options for `warmstart_opt` and `main_opt`:**


| Parameter            | Description                  | Typical Value |
| -------------------- | ---------------------------- | ------------- |
| `iterations`         | Maximum iterations           | 200--2000     |
| `g_abstol`<br> `g_reltol` | Gradient absolute and relative tolerance  | 1e-5 to 1e-8  |
| `f_abstol`<br> `f_reltol` | Function absolute and relative tolerance  | 1e-32         |
| `x_abstol`<br> `x_reltol` | Parameter absolute and relative tolerance | 1e-32         |
| `show_trace`         | Display iteration progress   | `false`       |

**Notes:**

- If `warmstart_solver` is omitted, the warmstart stage is skipped.

#### Return Value

Returns an optimization specification struct containing the solver and option specifications.

#### Example: Two-Stage Optimization

```julia
myopt = sfmodel_opt(
    warmstart_solver = NelderMead(),
    warmstart_opt = (iterations = 100, g_reltol = 1e-4),
    main_solver = Newton(),
    main_opt = (iterations = 200, g_reltol = 1e-8)
)
```

#### Example: Skip Warmstart

If the warmstart solver is not provided, the warmstart stage is skipped:

```julia
myopt = sfmodel_opt(
    main_solver = Newton(),
    main_opt = (iterations = 200, g_abstol = 1e-8)
)
```

#### Notes

- If `warmstart_solver` is provided without `warmstart_opt`, default options are used: `(iterations = 100, g_abstol = 1e-3)`.

#### Common Pitfall: Trailing Comma for Single-Element Options

When specifying options with only **one** element, you must include a **trailing comma** to create a NamedTuple:

```julia
# CORRECT - trailing comma creates a NamedTuple
main_opt = (iterations = 200,)

# WRONG - without comma, this is just the integer 200, not a NamedTuple
main_opt = (iterations = 200)
```

For **two or more** elements, the trailing comma is not needed:

```julia
# Both work fine
main_opt = (iterations = 200, g_abstol = 1e-8)
main_opt = (iterations = 200, g_abstol = 1e-8,)  # trailing comma optional
```

If you forget the trailing comma, the function will display a helpful error message:

```
ERROR: Invalid `main_opt`: expected a NamedTuple, got Int64.
Hint: For single-element options, use a trailing comma:
`main_opt = (iterations = 200,)` not `main_opt = (iterations = 200)`.
```

---

### 5.5 sfmodel_fit()

The `sfmodel_fit()` function is the main estimation routine. It organizes the entire workflow: optimization, variance-covariance computation, efficiency index calculation, marginal effects, and results presentation.

#### Syntax

`sfmodel_fit()` — run estimation and produce results.

**Required:**

- `spec` — model specification from `sfmodel_spec()`
- `method` — estimation method from `sfmodel_method()`

**Optional:**

- `init` — initial values from `sfmodel_init()`
- `optim_options` — optimization settings from `sfmodel_opt()`
- `jlms_bc_index` — compute JLMS/BC efficiency index (default: true)
- `marginal` — compute marginal effects (default: true)
- `show_table` — display results table (default: true)
- `verbose` — print progress messages (default: true)

#### Arguments


| Argument        | Type          | Description                                                                                                      |
| --------------- | ------------- | ---------------------------------------------------------------------------------------------------------------- |
| `spec`          | UnifiedSpec   | Model specification from`sfmodel_spec()` (required).                                                             |
| `method`        | UnifiedMethod | Method specification from`sfmodel_method()` (required).                                                          |
| `init`          | Vector        | Initial parameter vector from`sfmodel_init()`. If `nothing`, uses OLS for frontier and 0.1 for other parameters. |
| `optim_options` | --            | Optimization options from`sfmodel_opt()`. If `nothing`, uses defaults.                                           |
| `jlms_bc_index` | Bool          | Compute JLMS and BC efficiency indices (default:`true`).                                                         |
| `marginal`      | Bool          | Compute marginal effects of$Z$ on E(u) (default: `true`).                                                        |
| `show_table`    | Bool          | Print formatted estimation table (default:`true`).                                                               |
| `verbose`       | Bool          | Print detailed progress information (default:`true`).                                                            |

#### Return Value

Returns a `NamedTuple` with comprehensive results:

**Convergence Information**


| Field                | Type | Description                                |
| -------------------- | ---- | ------------------------------------------ |
| `converged`          | Bool | Whether optimization converged             |
| `iter_limit_reached` | Bool | Whether iteration limit was reached        |
| `redflag`            | Int  | Warning flag: 0 = OK, 1 = potential issues |

**Method Information**

| Field                    | Type   | Description                                            |
| ------------------------ | ------ | ------------------------------------------------------ |
| `GPU`                    | Bool   | Whether GPU acceleration was used                      |
| `n_draws`                | Int    | Actual number of draws per observation (or per firm for panel) |
| `multiRand`              | Bool   | Whether per-observation/per-firm Halton draws were used |
| `chunks`                 | Int    | Number of chunks for memory management                 |
| `distinct_Halton_length` | Int    | Maximum Halton sequence length for multiRand           |
| `estimation_method`      | Symbol | Estimation method used (`:MCI`, `:MSLE`, `:MLE`)       |

**Model Results**


| Field            | Type    | Description                    |
| ---------------- | ------- | ------------------------------ |
| `n_observations` | Int     | Number of observations (N)     |
| `loglikelihood`  | Float64 | Maximized log-likelihood value |
| `coeff`          | Vector  | Estimated coefficient vector   |
| `std_err`        | Vector  | Standard errors                |
| `var_cov_mat`    | Matrix  | Variance-covariance matrix     |
| `table`          | Matrix  | Formatted coefficient table    |

**Efficiency Indices**


| Field  | Type   | Description                                                                             |
| ------ | ------ | --------------------------------------------------------------------------------------- |
| `jlms` | Vector | JLMS inefficiency index$\text{E}(u \mid \varepsilon)$ for each observation              |
| `bc`   | Vector | Battese-Coelli efficiency index$\text{E}(e^{-u} \mid \varepsilon)$ for each observation |

**Marginal Effects**


| Field           | Type       | Description                                |
| --------------- | ---------- | ------------------------------------------ |
| `marginal`      | DataFrame  | Observation-level marginal effects on E(u) |
| `marginal_mean` | NamedTuple | Mean marginal effects                      |

> For the scaling property model (`hetero = :scaling`), marginal effects are computed via $\partial E(u_i) / \partial z_{ij} = \delta_j \cdot h(\mathbf{z}_i) \cdot E(u_i^*)$, using automatic differentiation. The coefficient $\delta_j$ directly gives the semi-elasticity $\partial \ln E(u_i) / \partial z_{ij} = \delta_j$. See [Section 9.5](#scaling-property-model-cross-sectional).

**OLS Diagnostics**


| Field               | Type    | Description                      |
| ------------------- | ------- | -------------------------------- |
| `OLS_loglikelihood` | Float64 | Log-likelihood from OLS frontier |
| `OLS_resid_skew`    | Float64 | Skewness of OLS residuals        |

**Technical Details**


| Field               | Type    | Description                        |
| ------------------- | ------- | ---------------------------------- |
| `model`             | --      | The internal model specification   |
| `Hessian`           | Matrix  | Numerical Hessian at optimum       |
| `gradient_norm`     | Float64 | Gradient norm at convergence       |
| `actual_iterations` | Int     | Total iterations across all stages |
| `warmstart_solver`  | Solver  | Warmstart algorithm used           |
| `warmstart_ini`     | Vector  | Initial values for warmstart       |
| `warmstart_maxIT`   | Int     | Maximum warmstart iterations       |
| `main_solver`       | Solver  | Main algorithm used                |
| `main_ini`          | Vector  | Initial values for main stage      |
| `main_maxIT`        | Int     | Maximum main iterations            |
| `main_tolerance`    | Float64 | Convergence tolerance              |

**Parameter Subsets**

Individual coefficient vectors are also available:

- `frontier` -- Frontier coefficients ($\beta$)
- `delta` -- Scaling function coefficients ($\boldsymbol{\delta}$), when `hetero = :scaling`
- `mu` -- $\mu$ coefficients (if applicable)
- `sigma_sq` / `sigma_u` -- $\sigma^2$ coefficients
- `lambda` -- $\lambda$ coefficients (Exponential, Weibull)
- `k` -- Shape parameter $k$ (Weibull, Gamma)
- `theta` -- Scale parameter $\theta$ (Gamma, MCI only)
- `alpha` -- $\alpha$ parameter (Lomax)
- `ln_sigma_v_sq` -- Noise variance (Normal, StudentT)
- `ln_b` -- Scale parameter (Laplace)
- `ln_nu_minus_2` -- Degrees of freedom (StudentT)

**Dictionary Access**

All results are also available through `result.list`, an `OrderedDict`:

```julia
keys(result.list)    # View all available keys
result.list[:coeff]  # Access specific result
result.coeff         # alternative 
```

#### Example: Full Estimation

```julia
result = sfmodel_fit(
    spec = myspec,
    method = mymeth,
    init = myinit,
    optim_options = myopt,
    jlms_bc_index = true,
    marginal = true,
    show_table = true,
    verbose = true
)

# Access results
println("Converged: ", result.converged)
println("Log-likelihood: ", result.loglikelihood)
println("Frontier coefficients: ", result.frontier)
println("Mean efficiency: ", mean(result.bc))
println("Marginal effects: ", result.marginal_mean)
```

#### Example: Minimal Call with Defaults

```julia
# Uses OLS-based initial values and default optimization
result = sfmodel_fit(spec = myspec, method = mymeth)
```

#### Output Display

When `show_table = true`, the function prints:

1. Model specification summary (distributions, sample size, etc.)
2. Warmstart progress (if enabled)
3. Main optimization progress
4. Formatted coefficient table with standard errors, z-statistics, p-values, and confidence intervals
5. Auxiliary table converting log-transformed parameters to original scale
6. Additional statistics (OLS log-likelihood, skewness, mean efficiency indices, marginal effects)

#### Convergence Warnings

The function monitors convergence and sets `redflag = 1` if:

- Gradient norm exceeds 0.1 or is NaN
- Iteration limit was reached
- Variance-covariance matrix has non-positive diagonal elements

### 5.6 `sfmodel_MixTable()` and `sfmodel_ChiSquareTable()`

These utility functions print critical values for hypothesis testing. They are available from the MLE backend.

#### `sfmodel_MixTable(dof)`

Prints the critical values of the **mixed chi-squared distribution** $\bar{\chi}^2$ for a given number of degrees of freedom (1 to 40). The mixed chi-squared distribution is a 50:50 mixture of $\chi^2(p-1)$ and $\chi^2(p)$, where $p$ is the number of restrictions.

**When to use:** The mixed chi-squared test is required for likelihood ratio (LR) tests where the null hypothesis places a parameter on the **boundary** of the parameter space. The classic example in stochastic frontier analysis is testing for the absence of inefficiency:

$$
H_0: \sigma_u^2 = 0 \quad \text{vs.} \quad H_1: \sigma_u^2 > 0.
$$

Because $\sigma_u^2 \ge 0$, the null value is on the boundary, and the standard $\chi^2$ distribution does not apply. The LR statistic $-2(\ln L_R - \ln L_U)$ should be compared against the mixed $\bar{\chi}^2$ critical values instead.

**Syntax:**

```julia
sfmodel_MixTable()       # Print the full table (dof 1 to 40)
sfmodel_MixTable(3)      # Print critical values for dof = 3
```

**Return:** A row (or matrix) of critical values at significance levels 0.10, 0.05, 0.025, and 0.01.

**Source:** Table 1, Kodde and Palm (1986, *Econometrica*).

#### `sfmodel_ChiSquareTable(dof)`

Prints the critical values of the **standard chi-squared distribution** $\chi^2$ for a given number of degrees of freedom.

**When to use:** For standard LR tests where the null hypothesis does **not** place parameters on the boundary. For example, testing a subset of frontier coefficients:

$$
H_0: \beta_2 = \beta_3 = 0 \quad \text{(interior restriction)}.
$$

**Syntax:**

```julia
sfmodel_ChiSquareTable(2)   # Critical values for dof = 2
```

**Return:** A row of critical values at significance levels 0.10, 0.05, 0.025, and 0.01.

#### Example: Testing for Inefficiency

```julia
# Estimate unrestricted model (stochastic frontier)
result_sf = sfmodel_fit(spec = myspec, method = mymeth)

# Estimate restricted model (OLS, no inefficiency)
# The OLS log-likelihood is available from the SF estimation output:
ll_ols = result_sf.OLS_loglikelihood
ll_sf  = result_sf.loglikelihood

# Compute LR statistic
LR = -2 * (ll_ols - ll_sf)

# Compare against mixed chi-squared critical values
# dof = 1 (one restriction: sigma_u^2 = 0)
sfmodel_MixTable(1)
# At 5% level, critical value is 2.705
# If LR > 2.705, reject H0 (inefficiency is statistically significant)
```

---

## 6. Supported Models

### Noise Distributions


| Symbol      | Distribution                            | Init Parameters                                                         | Models       | Methods        |
| ----------- | --------------------------------------- | ----------------------------------------------------------------------- | ------------ | -------------- |
| `:Normal`   | $v \sim N(0, \sigma_v^2)$               | `ln_sigma_v_sq` $= \log(\sigma_v^2)$                                    | cross, panel | MCI, MSLE, MLE |
| `:StudentT` | $v \sim t(0, \sigma_v, \nu)$, $\nu > 2$ | `ln_sigma_v_sq` $= \log(\sigma_v^2)$, `ln_nu_minus_2` $= \log(\nu - 2)$ | cross        | MCI, MSLE      |
| `:Laplace`  | $v \sim \text{Laplace}(0, b)$           | `ln_b` $= \log(b)$                                                      | cross        | MCI, MSLE      |

### Inefficiency Distributions


| Symbol             | Distribution                             | Init Parameters                                                 | Models       | Methods        |
| ------------------ | ---------------------------------------- | --------------------------------------------------------------- | ------------ | -------------- |
| `:HalfNormal`      | $u \sim N^+(0, \sigma^2)$                | `ln_sigma_sq` $= \log(\sigma^2)$                                | cross, panel | MCI, MSLE, MLE |
| `:TruncatedNormal` | $u \sim N^+(\mu, \sigma_u^2)$            | `mu` $= \mu$, `ln_sigma_sq` $= \log(\sigma_u^2)$                | cross, panel | MCI, MSLE, MLE |
| `:Exponential`     | $u \sim \text{Exp}(\lambda)$, $\lambda = \text{Var}(u)$ | `ln_lambda` $= \log(\lambda)$                                   | cross, panel | MCI, MSLE, MLE |
| `:Weibull`         | $u \sim \text{Weibull}(\lambda, k)$      | `ln_lambda` $= \log(\lambda)$, `ln_k` $= \log(k)$               | cross, panel | MCI, MSLE      |
| `:Lognormal`       | $u \sim \text{LogNormal}(\mu, \sigma^2)$ | `mu` $= \mu$, `ln_sigma_sq` $= \log(\sigma^2)$                  | cross, panel | MCI, MSLE      |
| `:Lomax`           | $u \sim \text{Lomax}(\alpha, \lambda)$   | `ln_lambda` $= \log(\lambda)$, `ln_alpha` $= \log(\alpha)$       | cross, panel | MCI, MSLE      |
| `:Rayleigh`        | $u \sim \text{Rayleigh}(\sigma)$         | `ln_sigma_sq` $= \log(\sigma^2)$                                | cross, panel | MCI, MSLE      |
| `:Gamma`           | $u \sim \text{Gamma}(k, \theta)$         | `ln_k` $= \log(k)$ (shape), `ln_theta` $= \log(\theta)$ (scale) | cross, panel | MCI            |

### Copula Models

_Cross-sectional models only._ A copula models the dependence between the noise term $v$ and the inefficiency term $u$. When `copula=:None` (default), $v$ and $u$ are assumed independent. With a copula, the joint density becomes $f(v,u) = f_v(v) \cdot f_u(u) \cdot c(F_v(v), F_u(u); \rho)$, where $c$ is the copula density and $\rho$ is the dependence parameter.


| Symbol       | Copula                | Domain            | Kendall's$\tau$        | Tail Dependence               | Init Parameter                           |
| ------------ | --------------------- | ----------------- | ---------------------- | ----------------------------- | ---------------------------------------- |
| `:Gaussian`  | Gaussian              | $\rho \in (-1,1)$ | $(2/\pi)\arcsin(\rho)$ | None                          | `theta_rho` $= \text{atanh}(\rho/0.999)$ |
| `:Clayton`   | Clayton               | $\rho > 0$        | $\rho/(2+\rho)$        | Lower: $2^{-1/\rho}$           | `theta_rho` $= \log(\rho - 10^{-6})$     |
| `:Clayton90` | Clayton 90° rotation | $\rho > 0$        | $-\rho/(2+\rho)$       | Upper-lower: $2^{-1/\rho}$ | `theta_rho` $= \log(\rho - 10^{-6})$     |
| `:Gumbel`    | Gumbel                | $\rho \geq 1$     | $1 - 1/\rho$           | Upper: $2 - 2^{1/\rho}$        | `theta_rho` $= \log(\rho - 1)$           |

**Notes:**

- Copula models are not available with `noise=:StudentT` (the Student-t CDF is not yet implemented for copula use).
- The `theta_rho` initial value is on the transformed (unconstrained) scale. A value of `0.0` is a reasonable default for all copula types.
- Clayton captures lower tail dependence (co-movement in the lower tail of distributions).
- Clayton 90° (rotated) captures upper-lower tail dependence. It uses $F_v(-v)$ instead of $1 - F_v(v)$ internally for numerical precision.
- Gumbel captures upper tail dependence (co-movement in the upper tail of distributions).
- Gaussian has no tail dependence but allows flexible symmetric dependence.

---

## 7. Distributions Reference

##### Noise Distributions


| Distribution | Specification                               | Required Init Parameters                                              |
| ------------ | ------------------------------------------- | --------------------------------------------------------------------- |
| Normal       | $v \sim N(0, \sigma_v^2)$                   | `ln_sigma_v_sq` = $\log(\sigma_v^2)$                                  |
| StudentT     | $v \sim t(0, \sigma_v, \nu)$ with $\nu > 2$ | `ln_sigma_v_sq` = $\log(\sigma_v^2)$,<br> `ln_nu_minus_2` = $\log(\nu-2)$ |
| Laplace      | $v \sim \text{Laplace}(0, b)$               | `ln_b` = $\log(b)$                                                    |

##### Inefficiency Distributions


| Distribution    | Specification                               | Required Init Parameters                         | Hetero Options                                 |
| --------------- | ------------------------------------------- | ------------------------------------------------ | ---------------------------------------------- |
| TruncatedNormal | $u \sim N^+(\mu, \sigma_u^2)$               | `mu` = $\mu$,<br> `ln_sigma_sq` = $\log(\sigma_u^2)$ | `:mu` for $\mu$,<br> `:sigma_sq` for $\sigma_u^2$  |
| HalfNormal      | $u \sim N^+(0, \sigma^2)$                   | `ln_sigma_sq` = $\log(\sigma^2)$                 | `:sigma_sq` for $\sigma^2$                     |
| Exponential     | $u \sim \text{Exp}(\lambda)$, $\lambda = \text{Var}(u)$ | `lambda` = $\lambda$                             | `:lambda` for $\lambda$                        |
| Weibull         | $u \sim \text{Weibull}(\lambda, k)$         | `lambda` = $\lambda$, <br>`k` = $k$                  | `:lambda` for $\lambda$,<br> `:k` for $k$          |
| Lognormal       | $u \sim \text{LogNormal}(\mu, \sigma^2)$    | `mu` = $\mu$,<br> `ln_sigma_sq` = $\log(\sigma^2)$   | `:mu` for $\mu$,<br> `:sigma_sq` for $\sigma^2$    |
| Lomax           | $u \sim \text{Lomax}(\alpha, \lambda)$      | `ln_lambda` = $\log(\lambda)$, <br>`ln_alpha` = $\log(\alpha)$ | `:lambda` for $\lambda$,<br> `:alpha` for $\alpha$ |
| Rayleigh        | $u \sim \text{Rayleigh}(\sigma)$            | `ln_sigma_sq` = $\log(\sigma^2)$               | `:sigma_sq` for $\sigma^2$                     |
| Gamma           | $u \sim \text{Gamma}(k, \theta)$ (MCI only) | `k` = $k$, <br>`theta` = $\theta$                    | `:k` for $k$,<br> `:theta` for $\theta$            |

---

**Notes:**

- The "Required Init Parameters" column shows the argument names used in `sfmodel_init()` component mode.
- The "Hetero Options" column shows valid symbols for the `hetero` argument in `sfmodel_spec()`.
- **Scaling property model alternative**: Instead of making individual distribution parameters heteroscedastic (via `hetero = [:mu]`, etc.), you can use `hetero = :scaling` to model heterogeneity through a single multiplicative scaling function $u_i = h(\mathbf{z}_i) \cdot u_i^*$. Under scaling, all distribution parameters remain scalar and a separate set of $\boldsymbol{\delta}$ coefficients is estimated. All 8 inefficiency distributions support the scaling property model. See [Section 9.5](#scaling-property-model-cross-sectional).

### Parameterization

When a parameter is modeled as heteroscedastic (observation-specific), we use a link function to ensure it stays in the correct domain:

* **Linear link** for parameters on $(-\infty,\infty)$.
  Example (TruncatedNormal): $\mu_i = Z_i'\delta$.
* **Exponential link** for parameters on $(0,\infty)$.
  Example (TruncatedNormal): $\sigma_{u,i}^2 = \exp(Z_i'\gamma)$, equivalently $\log(\sigma_{u,i}^2) = Z_i'\gamma$.

### Parameter Vector Length

The total number of parameters depends on heteroscedasticity settings:

```julia
# Homoscedastic: scalar parameters
hetero = Symbol[]
# -> Each inefficiency parameter contributes 1 to parameter count

# Heteroscedastic mu: L parameters
hetero = [:mu]
# -> mu contributes L parameters (one for each Z column)

# Fully heteroscedastic TruncatedNormal
hetero = [:mu, :sigma_sq]
# -> mu contributes L, sigma_u^2 contributes L

# Scaling property model
hetero = :scaling
# -> Adds L_scaling delta coefficients (one per column of zvar)
# -> All inefficiency parameters are scalar (1 each)
```

### Example: Heteroscedastic Model

```julia
# Z has 4 columns (including constant)
# hetero = [:mu, :sigma_sq] means both mu and sigma_u^2 depend on Z

spec = sfmodel_spec(
    depvar = y,
    frontier = X,          # K = 3 variables
    zvar = Z,              # L = 4 variables
    noise = :Normal,
    ineff = :TruncatedNormal,
    hetero = [:mu, :sigma_sq]
)

# Parameter vector structure:
# [beta_1, beta_2, beta_3,           <- frontier (K = 3)
#  delta_1, delta_2, delta_3, delta_4,     <- mu (L = 4, heteroscedastic)
#  gamma_1, gamma_2, gamma_3, gamma_4,     <- ln_sigma_u^2 (L = 4, heteroscedastic)
#  ln_sigma_v^2]                           <- noise variance (1)
# Total: 3 + 4 + 4 + 1 = 12 parameters
```

### Example: Scaling Property Model

```julia
# Z_nocons has 2 columns (no constant!)
# hetero = :scaling activates the scaling property model

spec = sfmodel_spec(
    depvar = y,
    frontier = X,              # K = 3 variables
    zvar = Z_nocons,           # L = 2 variables (no constant)
    noise = :Normal,
    ineff = :HalfNormal,
    hetero = :scaling
)

# Parameter vector structure:
# [beta_1, beta_2, beta_3,    <- frontier (K = 3)
#  delta_1, delta_2,           <- scaling function (L_scaling = 2)
#  ln_sigma^2,                 <- inefficiency (scalar, 1)
#  ln_sigma_v^2]               <- noise variance (1)
# Total: 3 + 2 + 1 + 1 = 7 parameters
```

### 7.1 Distribution Selection Guidance {#71-distribution-selection-guidance}

#### Inefficiency Distribution

The choice of inefficiency distribution affects the shape of the estimated inefficiency, particularly its mode and tail behavior. The following table summarizes the key properties:

| Distribution | Parameters | Mode at Zero? | Tail | Recommended Use |
| ------------ | ---------- | ------------- | ---- | --------------- |
| HalfNormal | 1 ($\sigma$) | Yes | Light | Default starting point; most common in the literature |
| Exponential | 1 ($\lambda$) | Yes | Light | Simple alternative to HalfNormal; monotone decreasing density |
| TruncatedNormal | 2 ($\mu$, $\sigma$) | Not necessarily | Light | When the mode of inefficiency may be at a positive value |
| Rayleigh | 1 ($\sigma$) | No (mode > 0) | Light | One-parameter distribution with mode away from zero |
| Weibull | 2 ($\lambda$, $k$) | Flexible | Moderate | Flexible shape: mode at zero ($k \le 1$) or positive ($k > 1$) |
| Lognormal | 2 ($\mu$, $\sigma$) | No (mode > 0) | Heavy | Right-skewed inefficiency with a heavy right tail |
| Lomax | 2 ($\alpha$, $\lambda$) | Yes | Heavy | Heavy-tailed; useful when a few firms are highly inefficient |
| Gamma | 2 ($k$, $\theta$) | Flexible | Moderate | Maximum flexibility; requires MCI method |

**Practical workflow:**

1. **Start simple.** Begin with HalfNormal or Exponential. These one-parameter distributions are easy to estimate (MLE available) and serve as a baseline.
2. **Allow non-zero mode.** If theory suggests that most firms have some positive level of inefficiency (rather than clustering near zero), try TruncatedNormal or Rayleigh.
3. **Add flexibility.** If one- or two-parameter distributions seem restrictive, try Weibull, Lognormal, or Gamma. These can capture a wider variety of shapes but require simulation-based estimation (MCI or MSLE).
4. **Compare models.** Compare log-likelihood values across specifications. Higher log-likelihood (less negative) indicates better fit. For nested models, use a likelihood ratio test (see [Section 5.6](#56-sfmodel_mixtable-and-sfmodel_chisquaretable) for critical values).

#### Noise Distribution

| Distribution | When to Use |
| ------------ | ----------- |
| Normal | Default and standard assumption. Use unless there is specific evidence otherwise. |
| Student-t | When residuals exhibit excess kurtosis (heavier tails than Normal). The degrees-of-freedom parameter $\nu$ is estimated. |
| Laplace | When the noise distribution is believed to be double-exponential (sharper peak, heavier tails than Normal). |

> **Note:** Student-t and Laplace noise require simulation-based estimation (MCI or MSLE) and are available for cross-sectional models only.

---

## 8. Working with Results

### Accessing Efficiency Indices

```julia
result = sfmodel_fit(spec = myspec, method = mymeth, jlms_bc_index = true)

# JLMS inefficiency index: E(u|epsilon)
jlms = result.jlms
mean_inefficiency = mean(jlms)

# Battese-Coelli efficiency: E(exp(-u)|epsilon)
bc = result.bc
mean_efficiency = mean(bc)

# Efficiency scores are observation-specific
println("Firm 1 efficiency: ", bc[1])
println("Firm 1 inefficiency: ", jlms[1])
```

### Working with Marginal Effects

Marginal effects measure how changes in Z variables affect expected inefficiency E(u):

```julia
result = sfmodel_fit(spec = myspec, method = mymeth, marginal = true)

# Mean marginal effects
println(result.marginal_mean)
# Output: (age = 0.023, school = -0.015, ...)

# Observation-level marginal effects (DataFrame)
marginal_df = result.marginal
println(names(marginal_df))  # ["marg_age", "marg_school", ...]

# Access specific firm's marginal effects
println("Firm 1 marginal effect of age: ", marginal_df[1, :marg_age])
```

### Variance-Covariance Matrix

```julia
# Full variance-covariance matrix
vcov = result.var_cov_mat

# Standard errors (square root of diagonal)
se = result.std_err

# Compute confidence interval for coefficient i
coef_i = result.coeff[i]
se_i = result.std_err[i]
ci_lower = coef_i - 1.96 * se_i
ci_upper = coef_i + 1.96 * se_i
```

### Extracting Specific Parameter Groups

```julia
# Frontier coefficients
beta = result.frontier

# Inefficiency parameters (varies by model)
if haskey(result.list, :mu)
    mu_coef = result.mu
end

# Noise variance
if haskey(result.list, :ln_sigma_v_sq)
    sigma_v_sq = exp(result.ln_sigma_v_sq)
end
```

### 8.1 Interpreting Estimation Results {#81-interpreting-estimation-results}

#### Efficiency Indices

- **JLMS inefficiency index** $E(u_i \mid \varepsilon_i)$: The conditional expectation of inefficiency for each observation. Higher values indicate greater inefficiency. Typical summary statistics to report: mean, median, min, max, and standard deviation across firms.

- **Battese-Coelli (BC) efficiency index** $E(e^{-u_i} \mid \varepsilon_i)$: The efficiency ratio, bounded between 0 and 1. A BC value of 0.85 means the firm produces 85% of its potential output. For cost frontiers, the same value means the firm's costs are $1/0.85 \approx 117.6\%$ of the efficient cost level.

#### Marginal Effects

When `zvar` is specified and `marginal = true` in `sfmodel_fit()`:

- **Observation-level marginal effects** (`result.marginal`): A DataFrame with one row per observation, showing $\partial E(u_i) / \partial z_{ij}$ for each Z variable.
- **Mean marginal effects** (`result.marginal_mean`): Sample averages of the observation-level effects.
- **Interpretation:** A positive marginal effect means that increasing $z_j$ increases expected inefficiency.
- **Scaling property model:** The estimated $\delta_j$ coefficients are semi-elasticities: $\partial \ln E(u_i) / \partial z_{ij} = \delta_j$. A coefficient of 0.10 means a one-unit increase in $z_j$ increases expected inefficiency by approximately 10%.

#### Log-Transformed Parameters

Many parameters are estimated on a log-transformed scale (e.g., `ln_sigma_sq`, `ln_sigma_v_sq`, `ln_lambda`). To recover the original-scale value, take the exponential:

```julia
sigma_u_sq = exp(result.ln_sigma_sq)    # σ_u²
sigma_v_sq = exp(result.ln_sigma_v_sq)  # σ_v²
```

The auxiliary table printed by `sfmodel_fit()` (when `show_table = true`) already reports the original-scale values alongside the log-transformed estimates.

#### Red Flags

Watch for these warning signs in the estimation output:

| Indicator | What It Means |
| --------- | ------------- |
| `redflag = 1` | Convergence issues detected (large gradient, iteration limit, or non-positive Hessian diagonal) |
| Wrong-sign `OLS_resid_skew` | Data may not support the presence of inefficiency in the assumed direction (see [Cost Frontier Interpretation](#92-cost-frontier-interpretation)) |
| BC values near 1.0 for all firms | Inefficiency may be negligible; consider whether a frontier model is appropriate |
| Very large standard errors | Possible multicollinearity in covariates or near-boundary parameter estimates |
| Gradient norm > 0.1 | Optimization did not reach a stationary point; results may be unreliable |

---

## 9. Advanced Topics

### Choosing Between MLE, MCI, and MSLE

The package offers three estimation methods. MLE uses analytic (closed-form) log-likelihoods, while MCI and MSLE are simulation-based and use Halton quasi-random draws.


| Feature              | MLE                            | MCI                           | MSLE                            |
| -------------------- | ------------------------------ | ----------------------------- | ------------------------------- |
| Likelihood           | Analytic (exact)               | Simulated                     | Simulated                       |
| Simulation draws     | Not needed                     | Halton QMC                    | Halton QMC                      |
| GPU support          | No                             | Yes                           | Yes                             |
| Noise distributions  | Normal only                    | Normal, StudentT, Laplace     | Normal, StudentT, Laplace       |
| Ineff distributions  | HalfNormal, TruncNormal, Expo | All 8                         | All except Gamma                |
| Copula support       | None                           | Gaussian, Clayton, Clayton90, Gumbel | Gaussian, Clayton, Clayton90, Gumbel |
| Scaling property     | Yes (TruncNormal only)         | All 8 distributions           | All except Gamma                |
| Heteroscedastic      | Yes                            | Yes                           | Yes                             |
| Panel models         | TFE, TFE_CSW, TRE             | TFE only                      | TFE only                        |
| Default`n_draws`     | N/A                            | 1024                          | 1024                            |

**When to use MLE:** If your model uses Normal noise with HalfNormal, TruncatedNormal, or Exponential inefficiency (and no copula), MLE is the natural first choice — it is exact (no simulation error), fast, and does not require tuning the number of draws.

**When to use MCI/MSLE:** For models that MLE cannot handle (non-Normal noise, copula dependence, Weibull/Lognormal/Lomax/Rayleigh/Gamma inefficiency), the simulation-based methods are required. For models supported by all three methods, MLE and simulation estimates typically agree closely.

> For a given model supported by both MCI and MSLE, the two simulation methods typically produce similar estimates. Differences may arise because MCI and MSLE use different likelihood constructions, so they can respond differently to the choice of distribution, sample size, and starting values. In particular, when the data do not conform well to the assumed distributional shape, the finite set of simulation draws may cover the tails unevenly, causing the two methods to weight those observations differently.

### GPU Computation

For large datasets, GPU computation can significantly speed up estimation:

```julia
using CUDA

meth = sfmodel_method(
    method = :MSLE,
    n_draws = 2^12 - 1,
    GPU = true,        # Enable GPU
    chunks = 10        # Split computation into 10 chunks for memory management
)
```

Requirements:

- CUDA.jl must be loaded **before** SFrontiers.jl (see [Section 3](#3-installation-and-dependencies))
- The `chunks` parameter controls memory usage by processing data in chunks

### Copula Models

To model dependence between the noise and inefficiency terms, specify a copula:

```julia
# Clayton copula (lower tail dependence)
spec = sfmodel_spec(
    depvar = y,
    frontier = X,
    noise = :Normal,
    ineff = :HalfNormal,
    copula = :Clayton,
    type = :production
)

# Initial values must include theta_rho for the copula parameter
myinit = sfmodel_init(
    spec = spec,
    frontier = X \ y,
    ln_sigma_sq = [0.0],
    ln_sigma_v_sq = [0.0],
    theta_rho = [0.0]       # copula dependence parameter (transformed scale)
)

# Clayton 90° rotated copula (upper-lower tail dependence)
spec_c90 = sfmodel_spec(
    depvar = y,
    frontier = X,
    noise = :Normal,
    ineff = :HalfNormal,
    copula = :Clayton90,
    type = :production
)

# Gumbel copula (upper tail dependence)
spec_g = sfmodel_spec(
    depvar = y,
    frontier = X,
    noise = :Normal,
    ineff = :Exponential,
    copula = :Gumbel,
    type = :production
)
```

The estimation output includes a copula auxiliary table showing:

- $\rho$: the dependence parameter (on the original scale) with standard error
- Kendall's $\tau$: rank correlation measure
- Tail dependence coefficient

#### Copula Interpretation Guide {#91-copula-interpretation-guide}

**Interpreting Kendall's $\tau$:**

Kendall's $\tau$ measures the degree of concordance (rank correlation) between the noise term $v$ and the inefficiency term $u$. A positive $\tau$ means that firms with larger noise shocks tend to have larger inefficiency (and vice versa). A negative $\tau$ indicates the opposite pattern. The magnitude of $\tau$ reflects the strength of the dependence.

**Tail dependence by copula type:**

| Copula | Tail Dependence | Interpretation |
| ------ | --------------- | -------------- |
| Gaussian | None (symmetric, no tail dependence) | Dependence is moderate and evenly spread; no extreme co-movement in the tails |
| Clayton | Lower tail | Firms at the low end of both $v$ and $u$ tend to co-move; captures "floor effects" where small disturbances coincide with low inefficiency |
| Clayton 90° | Upper-lower (rotated) | Mixed tail dependence pattern; useful when dependence is asymmetric in a non-standard direction |
| Gumbel | Upper tail | Firms at the high end of both $v$ and $u$ tend to co-move; captures "ceiling effects" where large shocks coincide with high inefficiency |

**When to choose which copula:**

1. **Start with `copula = :None`** (independence). Estimate the model without copula dependence as a baseline.
2. **Try Gaussian** if you suspect general dependence but have no strong prior on tail behavior. It is the most flexible symmetric option.
3. **Try Clayton or Gumbel** if you expect that dependence is stronger for extreme values (e.g., very efficient or very inefficient firms behave differently from the rest).
4. **Compare models** using the log-likelihood values across specifications. Also check whether the estimated $\hat{\rho}$ (copula parameter) is statistically significant — its standard error and z-statistic are reported in the estimation output.

> **Note:** Copula models are supported only for cross-sectional data. Student-t noise is not compatible with copulas.

### Scaling Property Model (Cross-Sectional)

The scaling property model introduces observation-specific heterogeneity through a single multiplicative scaling function, rather than making individual distributional parameters heteroscedastic:

$$
u_i = h(\mathbf{z}_i) \cdot u_i^*, \qquad h(\mathbf{z}_i) = \exp(\mathbf{z}_i'\boldsymbol{\delta}),
$$

where $u_i^*$ follows a homoscedastic (scalar-parameter) base distribution and $\mathbf{z}_i$ is a vector of environmental/exogenous variables. The scaling function $h(\mathbf{z}_i)$ is always positive and rescales the base inefficiency without changing the distributional shape.

**Key mathematical properties:**

1. **Jacobian cancellation**: The change of variable $u_i = h_i u_i^*$ produces a Jacobian $h_i$ that cancels with the $1/h_i$ from the density transformation $f_u(u_i) = \frac{1}{h_i} f_{u^*}(u_i/h_i)$, leaving $f_{u^*}(u_i^*)$ directly in all integrals.
2. **JLMS factorization**: $E(u_i \mid \varepsilon_i) = h_i \cdot E(u_i^* \mid \varepsilon_i)$.
3. **BC non-factorization**: $E(e^{-u_i} \mid \varepsilon_i) = E(e^{-h_i u_i^*} \mid \varepsilon_i)$; the $h_i$ must remain inside the exponential.
4. **Semi-elasticity**: $\partial \ln E(u_i) / \partial z_{ij} = \delta_j$, providing direct interpretation of the scaling coefficients.

**Unconditional mean** $E(u_i) = h(\mathbf{z}_i) \cdot E(u_i^*)$, where $E(u_i^*)$ depends on the base distribution:

| Distribution    | $E(u_i^*)$                                                                    |
| --------------- | ----------------------------------------------------------------------------- |
| HalfNormal      | $\sigma\sqrt{2/\pi}$                                                          |
| TruncatedNormal | $\sigma_u(\Lambda + \phi(\Lambda)/\Phi(\Lambda))$, $\Lambda = \mu/\sigma_u$   |
| Exponential     | $\sqrt{\lambda}$                                                               |
| Weibull         | $\lambda\,\Gamma(1 + 1/k)$                                                    |
| Lognormal       | $\exp(\mu + \sigma^2/2)$                                                       |
| Lomax           | $\lambda / (\alpha - 1)$, $\alpha > 1$                                         |
| Rayleigh        | $\sigma\sqrt{\pi/2}$                                                           |
| Gamma           | $k\theta$                                                                      |

**Identification constraint**: The `zvar` matrix must **not** contain a constant column. A constant in $\mathbf{z}_i$ would create an intercept $\delta_0$ that is not separately identified from the scale parameter of the base distribution (e.g., $\sigma$ in HalfNormal or $\lambda$ in Exponential).

**Supported distributions**: All 8 inefficiency distributions are supported. The Gamma distribution requires `method = :MCI`. All other distributions work with both `:MSLE` and `:MCI`.

#### Usage Example

```julia
# Step 1: Specify a scaling property model
spec = sfmodel_spec(
    depvar = y,
    frontier = X,                 # include constant
    zvar = Z_nocons,              # environmental variables (NO constant!)
    noise = :Normal,
    ineff = :HalfNormal,
    hetero = :scaling,            # activates scaling property model
    type = :production
)

# Step 2: Choose estimation method (both MSLE and MCI work)
meth = sfmodel_method(
    method = :MSLE,
    n_draws = 2^12 - 1,
    GPU = true,
    chunks = 10
)

# Step 3: Set initial values — note the "scaling" keyword for δ
init = sfmodel_init(
    spec = spec,
    frontier = X \ y,
    scaling = zeros(size(Z_nocons, 2)),   # δ initial values
    ln_sigma_sq = [0.0],                  # scalar base parameter
    ln_sigma_v_sq = [0.0]
)

# Step 4: Configure optimization
opt = sfmodel_opt(
    warmstart_solver = NelderMead(),
    warmstart_opt = (iterations = 200,),
    main_solver = Newton(),
    main_opt = (iterations = 100, g_abstol = 1e-8)
)

# Step 5: Estimate
result = sfmodel_fit(
    spec = spec, method = meth, init = init,
    optim_options = opt, marginal = true, show_table = true
)

# Access scaling coefficients
println("δ coefficients: ", result.delta)

# Marginal effects (computed via ForwardDiff)
println("Mean marginal effects: ", result.marginal_mean)
println("Observation-level marginal effects: ", result.marginal)
```

#### Scaling with Copula

The scaling property model can be combined with copula dependence:

```julia
spec = sfmodel_spec(
    depvar = y,
    frontier = X,
    zvar = Z_nocons,
    noise = :Normal,
    ineff = :HalfNormal,
    hetero = :scaling,
    copula = :Clayton,
    type = :production
)

init = sfmodel_init(
    spec = spec,
    frontier = X \ y,
    scaling = zeros(size(Z_nocons, 2)),
    ln_sigma_sq = [0.0],
    ln_sigma_v_sq = [0.0],
    theta_rho = [0.0]
)
```

### Cost Frontier Models {#92-cost-frontier-interpretation}

For cost frontiers, where inefficiency increases costs:

```julia
spec = sfmodel_spec(
    depvar = totalC,
    frontier = X,
    zvar = Z,
    noise = :Normal,
    ineff = :Exponential,
    type = :cost    # Cost frontier
)
```

#### Mathematical Difference

The key difference between production and cost frontiers lies in the sign of the inefficiency term:

- **Production frontier:** $\varepsilon_i = v_i - u_i$ (inefficiency reduces output)
- **Cost frontier:** $\varepsilon_i = v_i + u_i$ (inefficiency increases cost)

The software handles the sign convention internally when `type = :cost` is set. All post-estimation quantities (JLMS, BC, marginal effects) are computed correctly for cost frontiers without additional user adjustments.

#### Interpretation Differences

| Aspect | Production Frontier | Cost Frontier |
| ------ | ------------------- | ------------- |
| Composed error | $\varepsilon = v - u$ | $\varepsilon = v + u$ |
| OLS residual skewness | Should be **negative** | Should be **positive** |
| JLMS $E(u \mid \varepsilon)$ | Higher = more inefficient | Same interpretation |
| BC $E(e^{-u} \mid \varepsilon)$ | Closer to 1 = more efficient | Same interpretation |
| BC interpretation | Firm produces $\text{BC} \times 100\%$ of potential output | Firm's cost is $\frac{1}{\text{BC}} \times 100\%$ of the efficient cost level |
| Frontier coefficients | Output elasticities | Cost elasticities |

#### Common Pitfall: Wrong-Sign Skewness

Before estimating a stochastic frontier model, check the OLS residual skewness (reported in the `sfmodel_fit()` output as `OLS_resid_skew`):

- For a **production** frontier, the OLS residuals should be **negatively** skewed. This indicates the presence of one-sided inefficiency pulling output below the frontier.
- For a **cost** frontier, the OLS residuals should be **positively** skewed, indicating inefficiency pushing costs above the efficient level.

If the skewness has the wrong sign, the data may not support the presence of inefficiency in the assumed direction, and the model may have difficulty converging or produce unreliable estimates.

### Choosing the Number of Halton Draws

The number of QMC draws affects accuracy and computation time:


| n_draws           | Typical Use Case              |
| ----------------- | ----------------------------- |
| 127               | Quick testing                 |
| 1024              | Standard estimation (default) |
| 4095 ($2^{12}-1$) | Publication-quality results   |
| 8191              | High precision                |

```julia
meth = sfmodel_method(
    method = :MSLE,
    n_draws = 2^12 - 1  # 4095 draws
)
```

### Observation-Specific Halton Draws (multiRand)

By default (`multiRand=true`), each observation receives a different consecutive segment of the Halton sequence, providing better quasi-Monte Carlo coverage for the model:

- Observation 1 gets draws [1, 2, ..., D]
- Observation 2 gets draws [D+1, D+2, ..., 2D]
- And so on, recycling the sequence when it runs out

The mechanism is designed so that a complete, equidistributed Halton sequence of length $L$ is generated and then assigned to $N$ observations in consecutive blocks of $D$ draws, and it is recycled, if necessary, to fill in the $N\times D$ elements.

We impose a default upper bound of $L=2^{15}-1=32767$ (controlled by `distinct_Halton_length`) to avoid generating points that lie extremely close to 0 or 1, which can trigger numerical instability in floating-point arithmetic (e.g., through $\log(t)$, $\log(1-t)$, or divisions by $t$ or $(1-t)$). This bound can be increased via the `distinct_Halton_length` option in `sfmodel_method()`.

Ideally $L \leq N \times D$ so that the full sequence is consumed and recycled if necessary. When $N \times D < L$, the length-$L$ sequence cannot be fully utilized. In this case, the program automatically selects the longest $L'$ satisfying $L' \leq N \times D$ and uses it as described.

**Constraint**: When `multiRand=true`, `n_draws` (the number of draws per observation) must be $\leq$ `distinct_Halton_length` (default $2^{15} - 1 = 32767$). To use more draws, either increase `distinct_Halton_length` or set `multiRand=false`.

```julia
# Default: observation-specific draws (recommended)
meth = sfmodel_method(
    method = :MSLE,
    n_draws = 1024,    # Must be <= distinct_Halton_length (default 32767)
    multiRand = true   # Default
)

# Use more draws by increasing distinct_Halton_length
meth = sfmodel_method(
    method = :MSLE,
    n_draws = 50000,
    multiRand = true,
    distinct_Halton_length = 2^16 - 1  # 65535
)

# Legacy mode: all observations share the same draws
meth = sfmodel_method(
    method = :MSLE,
    n_draws = 8191,    # No constraint from distinct_Halton_length
    multiRand = false
)
```

### Custom Halton Sequences

For reproducibility or special requirements when using `multiRand=false`:

```julia
using HaltonSequences

# Generate custom Halton sequence as 1xD matrix (required format for multiRand=false)
halton_vec = make_halton_p(1024; T = Float64)
halton = reshape(halton_vec, 1, length(halton_vec))  # Convert to 1xD

meth = sfmodel_method(
    method = :MSLE,
    draws = halton,     # Pre-reshaped 1xD matrix
    multiRand = false   # Required when providing custom 1xD draws
)
```

### Handling Convergence Issues

If estimation fails to converge:

1. **Try different initial values**

   ```julia
   # Use grid search for starting values
   for sigma_init in [-2.0, -1.0, 0.0, 1.0]
       init = sfmodel_init(spec=spec, ..., ln_sigma_sq=[sigma_init])
       result = sfmodel_fit(spec=spec, method=meth, init=init)
       if result.converged
           break
       end
   end
   ```
2. **Increase warmstart iterations**

   ```julia
   myopt = sfmodel_opt(
       warmstart_solver = NelderMead(),
       warmstart_opt = (iterations = 400,),  # More warmstart iterations
       main_solver = Newton(),
       main_opt = (iterations = 200,)
   )
   ```
3. **Use different optimizers**

   ```julia
   myopt = sfmodel_opt(
       warmstart_solver = ParticleSwarm(),  # Global search, slow
       warmstart_opt = (iterations = 500,),
       main_solver = BFGS(),  # Alternative to Newton
       main_opt = (iterations = 2000,)
   )
   ```
4. **Check the gradient norm**

   ```julia
   if result.gradient_norm > 0.1
       println("Warning: Large gradient norm indicates poor convergence")
   end
   ```
5. **Try a different estimation method**

   If one method has convergence difficulties, try another:

   ```julia
   # If MSLE fails, try MCI (or vice versa)
   meth_alt = sfmodel_method(method = :MCI, n_draws = 1024)
   result = sfmodel_fit(spec = spec, method = meth_alt)

   # Or try MLE (if the model supports it: Normal noise, no copula,
   # HalfNormal/TruncatedNormal/Exponential inefficiency)
   meth_mle = sfmodel_method(method = :MLE)
   result = sfmodel_fit(spec = spec, method = meth_mle)
   ```
6. **Check OLS residual skewness**

   Before investing effort in convergence tuning, verify that the data supports the presence of inefficiency. If `result.OLS_resid_skew` has the wrong sign (positive for production, negative for cost), the data may not exhibit the one-sided pattern that stochastic frontier models require. In such cases, even a perfectly converged model may produce meaningless estimates.

7. **Reduce model complexity first**

   If a heteroscedastic or copula model fails to converge, first estimate a simpler version:
   - Drop the copula (`copula = :None`) to establish a baseline.
   - Remove heteroscedasticity (`hetero = Symbol[]`) and use a homoscedastic specification.
   - Use a simpler inefficiency distribution (e.g., HalfNormal instead of Lognormal).

   Once the simple model converges, use its estimates as initial values for the more complex specification.

8. **Diagnostic decision tree**

   | Symptom | Likely Cause | Action |
   | ------- | ------------ | ------ |
   | Warmstart did not converge | Poor starting region | Increase warmstart iterations (e.g., 500+) or switch to `ParticleSwarm()` for global search |
   | Main stage did not converge | Starting values still far from optimum | Use warmstart estimates as init, increase main iterations, or try `BFGS()` instead of `Newton()` |
   | Hessian non-invertible | Model overparameterized or data insufficient | Reduce model complexity; check for multicollinearity in X or Z |
   | Very large standard errors | Near-boundary parameters or collinear covariates | Check correlation among Z variables; consider dropping redundant covariates |
   | Gradient norm between 0.01 and 0.1 | Near convergence but not quite | Increase `iterations` or loosen `g_abstol` slightly |
   | Gradient norm > 1 | Far from optimum | Completely different initial values needed; try grid search |

---

## 10. Panel Data Models

The module supports several panel stochastic frontier models for estimating **persistent firm-level inefficiency** from balanced or unbalanced panel data. Panel estimation is accessed through the same unified API by setting the `datatype` argument in `sfmodel_spec()`.

### Panel Model Types

| `datatype`       | Model                                      | Methods            | Ineff (MLE)                   |
| ---------------- | ------------------------------------------ | ------------------ | ----------------------------- |
| `:panel_TFE`     | Wang and Ho (2010) true fixed-effect       | MCI, MSLE, MLE     | HalfNormal, TruncatedNormal   |
| `:panel_TFE_CSW` | Chen, Schmidt, and Wang (2014) fixed-effect | MLE only           | HalfNormal only               |
| `:panel_TRE`     | True random-effect                         | MLE only           | HalfNormal, TruncatedNormal   |

For `panel_TFE`, the simulation-based methods (MCI, MSLE) support all 8 inefficiency distributions. MLE support is limited to the distributions listed above. For `panel_TFE_CSW` and `panel_TRE`, only MLE is available.

### 10.1 Theoretical Background

#### Wang and Ho (2010) True Fixed-Effect (`panel_TFE`)

The Wang and Ho (2010) model starts from:

$$
y_{it} = \alpha_i + x_{it}'\beta + v_{it} - h(z_{it}) \cdot u_i^*
$$

where $\alpha_i$ is a firm-specific fixed effect, $v_{it} \sim N(0, \sigma_v^2)$ is noise, $u_i^* \ge 0$ is persistent inefficiency, and $h(z_{it}) = \exp(z_{it}'\delta)$ is a scaling function.

To eliminate $\alpha_i$, a within-group transformation (demeaning) is applied:

$$
\tilde{y}_{it} = \tilde{x}_{it}'\beta + \tilde{v}_{it} - \tilde{h}_{it} \cdot u_i^*
$$

where tildes denote demeaned values. **Key point:** $y$ and $X$ are demeaned, but $Z$ is NOT demeaned. Instead, $h(z_{it})$ is computed first, then demeaned to obtain $\tilde{h}_{it}$.

Because the demeaned noise has a singular covariance $\Sigma = \sigma_v^2(I - \frac{1}{T}\mathbf{1}\mathbf{1}')$, the quadratic form simplifies to:

$$
\text{pert}' \Sigma^+ \text{pert} = \|\text{pert}\|^2 / \sigma_v^2
$$

This eliminates the need for the pseudo-inverse, keeping computation efficient.

#### Chen, Schmidt, and Wang (2014) Fixed-Effect (`panel_TFE_CSW`)

The CSW model (Chen, Schmidt, and Wang 2014, *Journal of Econometrics*) provides an alternative approach to estimating true fixed-effect stochastic frontier models. After the within-group transformation eliminates the fixed effects $\alpha_i$, the CSW approach exploits properties of the **closed skew-normal (CSN) distribution** to derive a closed-form likelihood for the demeaned composed error. This avoids the need for the scaling-property structure required by Wang and Ho (2010).

**Key characteristics:**

- **No scaling function needed.** Unlike the Wang-Ho TFE model, CSW does not require the $h(z_{it}) = \exp(z_{it}'\delta)$ structure. Consequently, no `zvar` argument is needed.
- **No exogenous inefficiency determinants.** Because there is no scaling function, this model cannot incorporate Z variables that explain heterogeneity in inefficiency.
- **MLE only.** The closed-form likelihood means no simulation is required, but it also means only the MLE method is available.
- **HalfNormal inefficiency only.** The CSN derivation relies on the half-normal distribution.

**When to use CSW vs. Wang-Ho TFE:**

- Use **CSW** when you have panel data with persistent inefficiency but do not need exogenous determinants of inefficiency, and prefer a simpler model with a closed-form likelihood and no simulation error.
- Use **Wang-Ho TFE** when you need flexible distributional assumptions (any of the 8 inefficiency distributions via MCI/MSLE) or when you want to model how Z variables affect the level of inefficiency through the scaling function.

#### True Random-Effect (`panel_TRE`)

The TRE model, attributed to Greene (2005), treats the individual effect $\alpha_i$ as a random draw from $N(0, \sigma_\alpha^2)$ rather than as a fixed parameter to be eliminated. Because $\alpha_i$ is random and integrated out of the likelihood, a **constant term may be included** in the frontier equation (unlike the fixed-effect models where demeaning eliminates constants).

**Key characteristics:**

- **Random individual effects.** Assumes $\alpha_i$ is uncorrelated with the regressors $x_{it}$. If this assumption is violated (e.g., firm size is correlated with both input choices and the firm effect), the TRE estimates may be inconsistent.
- **MLE only.** Available with HalfNormal or TruncatedNormal inefficiency.
- **Constant term allowed.** Unlike TFE and CSW, you may include a column of ones in `frontier`.
- **Supports `zvar`.** Inefficiency determinants can be modeled through $\mu_i = z_i'\delta$ (for TruncatedNormal) or through the overall variance.

**When to use TRE vs. TFE:**

The choice between TRE and TFE mirrors the classic random-effects vs. fixed-effects tradeoff in panel econometrics:

- Use **TRE** when firm-specific effects are believed to be uncorrelated with the regressors. TRE is more efficient (uses both within- and between-variation) but inconsistent if the uncorrelatedness assumption fails.
- Use **TFE** (Wang-Ho or CSW) when correlation between firm effects and regressors is likely. TFE is consistent regardless of this correlation but less efficient.

### 10.2 Panel Quick Start

#### Example: Panel TFE with MSLE (simulation-based)

```julia
using CSV, DataFrames, Optim
using CUDA
using SFrontiers

# Load panel data (stacked: firm 1 all T periods, firm 2 all T periods, ...)
df = CSV.read("panel_data.csv", DataFrame)
y = df.y
X = hcat(df.x1, df.x2)    # NT x K — no constant column!
Z = hcat(df.z1)            # NT x L — no constant column!

# Step 1: Specify panel model
myspec = sfmodel_spec(
    depvar = y,
    frontier = X,
    zvar = Z,
    noise = :Normal,
    ineff = :HalfNormal,
    datatype = :panel_TFE,    # Wang and Ho 2010 true fixed-effect
    T_periods = 10,           # balanced panel, 10 periods
    type = :production
)

# Step 2: Choose estimation method
mymeth = sfmodel_method(method = :MSLE, n_draws = 1024)

# Step 3: Set initial values (panel-specific keywords)
myinit = sfmodel_init(
    spec = myspec,
    delta = [0.1],            # scaling function coefficient
    ln_sigma_u_sq = 0.1,     # scalar
    ln_sigma_v_sq = 0.1      # scalar
)

# Step 4: Configure optimization
myopt = sfmodel_opt(
    warmstart_solver = NelderMead(),
    warmstart_opt = (iterations = 200, g_abstol = 1e-3),
    main_solver = Newton(),
    main_opt = (iterations = 200, g_abstol = 1e-8)
)

# Step 5: Estimate
result = sfmodel_fit(
    spec = myspec,
    method = mymeth,
    init = myinit,
    optim_options = myopt,
    show_table = true
)

# Access results
println("Log-likelihood: ", result.loglikelihood)
println("Mean JLMS: ", mean(result.jlms))      # firm-level, N-vector
println("Mean BC: ", mean(result.bc))            # firm-level, N-vector
```

#### Example: Panel TFE with MLE (analytic)

```julia
# Same spec as above — MLE is available when ineff is HalfNormal or TruncatedNormal
mymeth_mle = sfmodel_method(method = :MLE)  # no draws needed
result_mle = sfmodel_fit(spec = myspec, method = mymeth_mle, show_table = true)
```

#### Example: Panel TFE_CSW (MLE only)

```julia
spec_csw = sfmodel_spec(
    depvar = y,
    frontier = X,
    noise = :Normal,
    ineff = :HalfNormal,        # CSW requires HalfNormal
    datatype = :panel_TFE_CSW,
    id = firm_ids               # unbalanced panel
)
meth = sfmodel_method(method = :MLE)
result = sfmodel_fit(spec = spec_csw, method = meth)
```

#### Example: Panel TRE (MLE only)

```julia
spec_tre = sfmodel_spec(
    depvar = y,
    frontier = X,
    zvar = Z,
    noise = :Normal,
    ineff = :HalfNormal,        # or :TruncatedNormal
    datatype = :panel_TRE,
    id = firm_ids
)
meth = sfmodel_method(method = :MLE)
result = sfmodel_fit(spec = spec_tre, method = meth)
```

### 10.3 Panel vs. Cross-Sectional Differences


| Feature                      | Cross-Sectional                                                        | Panel (TFE / TFE_CSW / TRE)                  |
| ---------------------------- | ---------------------------------------------------------------------- | --------------------------------------------- |
| `datatype`                   | `:cross_sectional` (default)                                           | `:panel_TFE`, `:panel_TFE_CSW`, `:panel_TRE` |
| Noise distributions          | Normal, Student T, Laplace                                             | Normal only                                   |
| Inefficiency distributions   | All 8                                                                  | TFE: all 8 (MCI/MSLE), 2 (MLE); CSW: HalfNormal; TRE: 2 |
| Methods                      | MCI, MSLE, MLE                                                         | TFE: all 3; CSW/TRE: MLE only                |
| Copula                       | Gaussian, Clayton, Clayton90, Gumbel                                   | Not supported                                 |
| Heteroscedastic`hetero`      | Yes (via Z), or `hetero=:scaling` for scaling property model           | Not supported; use$h(z)$ scaling              |
| Scaling property model       | `hetero=:scaling`; `zvar` has no constant; init keyword: `scaling`     | Always active; `zvar` has no constant; init keyword: `delta` |
| Constant in`frontier`/`zvar` | Required in `frontier`; required in `zvar` unless `hetero=:scaling`    | **Not allowed** (within-demeaning eliminates) |
| JLMS/BC indices              | Observation-level ($N$ vector)                                         | Firm-level ($N_{\text{firms}}$ vector)        |
| Panel structure              | N/A                                                                    | `T_periods` (balanced) or `id` (unbalanced)   |
| Init: scaling coefficients   | `scaling` keyword (when `hetero=:scaling`)                             | `delta` keyword                               |
| Marginal effects             | Yes (heteroscedastic and scaling)                                      | N/A                                           |

### 10.4 Supported Panel Inefficiency Distributions

#### Panel TFE (Wang and Ho 2010)

All eight inefficiency distributions are supported via MCI/MSLE. MLE supports a subset:


| Distribution       | Parameters (scalar)   | MLE | MSLE | MCI |
| ------------------ | --------------------- | --- | ---- | --- |
| `:HalfNormal`      | `ln_sigma_u_sq`       | Yes | Yes  | Yes |
| `:TruncatedNormal` | `mu`, `ln_sigma_u_sq` | Yes | Yes  | Yes |
| `:Exponential`     | `ln_lambda`           | No  | Yes  | Yes |
| `:Weibull`         | `ln_lambda`, `ln_k`   | No  | Yes  | Yes |
| `:Lognormal`       | `mu`, `ln_sigma_sq`   | No  | Yes  | Yes |
| `:Lomax`           | `ln_lambda`, `ln_alpha` | No  | Yes  | Yes |
| `:Rayleigh`        | `ln_sigma_sq`         | No  | Yes  | Yes |
| `:Gamma`           | `ln_k`, `ln_theta`    | No  | No   | Yes |

#### Panel TFE_CSW (Chen, Schmidt, and Wang 2014)

| Distribution       | MLE |
| ------------------ | --- |
| `:HalfNormal`      | Yes |

#### Panel TRE (True Random-Effect)

| Distribution       | MLE |
| ------------------ | --- |
| `:HalfNormal`      | Yes |
| `:TruncatedNormal` | Yes |

**Important:** In all panel models, inefficiency distribution parameters are **scalar** (not heteroscedastic via Z). Heterogeneity enters only through the scaling function $h(z_{it}) = \exp(z_{it}'\delta)$.

### 10.5 Balanced vs. Unbalanced Panels

**Balanced panels** have the same number of time periods for all firms. Use the `T_periods` keyword:

```julia
spec = sfmodel_spec(depvar = y, frontier = X, zvar = Z,
    noise = :Normal, ineff = :HalfNormal,
    datatype = :panel_TFE, T_periods = 10)
```

Data must be stacked by firm: all $T$ observations for firm 1, then all $T$ for firm 2, etc.

**Unbalanced panels** have varying numbers of time periods per firm. Use the `id` keyword:

```julia
spec = sfmodel_spec(depvar = y, frontier = X, zvar = Z,
    noise = :Normal, ineff = :HalfNormal,
    datatype = :panel_TFE, id = firm_ids)
```

Data must be grouped by firm (contiguous rows for each unit). The number of periods per firm is inferred from the `id` column.

### 10.6 No Constant Columns

In the Wang and Ho (2010) panel model (`panel_TFE`), within-group demeaning eliminates any constant terms. Therefore:

- **Do NOT include a column of ones** in `frontier` or `zvar`.
- The model will raise an error if a constant column is detected.
- This differs from cross-sectional models, where constants are typically required.
- The same applies to `panel_TFE_CSW`. For `panel_TRE`, a constant term **is** allowed in `frontier` (see [Section 10.1](#101-theoretical-background)).

### 10.7 Panel DSL Macros

In addition to `@useData`, `@depvar`, `@frontier`, and `@zvar`, panel models support the `@id` macro for unbalanced panels:

```julia
# Panel TFE — balanced (no @id, requires T_periods keyword)
spec = sfmodel_spec(@useData(df), @depvar(y), @frontier(x1, x2), @zvar(z1),
    noise = :Normal, ineff = :HalfNormal, datatype = :panel_TFE, T_periods = 10)

# Panel TFE — unbalanced (with @id)
spec = sfmodel_spec(@useData(df), @depvar(y), @frontier(x1, x2), @zvar(z1), @id(firm),
    noise = :Normal, ineff = :HalfNormal, datatype = :panel_TFE)

# Panel TRE — unbalanced (with @id)
spec = sfmodel_spec(@useData(df), @depvar(y), @frontier(x1, x2), @id(firm),
    noise = :Normal, ineff = :HalfNormal, datatype = :panel_TRE)
```

### 10.8 Error Messages for Unsupported Configurations

When `method=:MLE` is used with a configuration that MLE does not support, `sfmodel_fit` issues an informative error listing the reason and the supported alternatives. For example:

- `method=:MLE` + `datatype=:panel_TFE` + `ineff=:Exponential` → error explaining MLE supports only HalfNormal/TruncatedNormal for panel TFE, with `:MCI` and `:MSLE` listed as alternatives.
- `method=:MCI` + `datatype=:panel_TRE` → error explaining panel TRE supports only `:MLE`.

### 10.9 Standalone Panel API (Backward Compatible)

For users who prefer the standalone panel API (without the unified `sfmodel_spec` interface), the following functions are available and delegate directly to the Panel backend (Wang and Ho 2010 model only):

```julia
sfmodel_panel_spec(; kwargs...)      # Panel model specification
sfmodel_panel_method(; kwargs...)    # Panel method specification
sfmodel_panel_init(; kwargs...)      # Panel initial values
sfmodel_panel_opt(; kwargs...)       # Panel optimization options
sfmodel_panel_fit(; kwargs...)       # Panel estimation
```

These are retained for backward compatibility with existing panel scripts.

---

## Citation

- Wang, H.J. (2025) NSC Project.
- Wang, H.-J. and Ho, C.-W. (2010). Estimating fixed-effect panel stochastic frontier models by model transformation. *Journal of Econometrics*, 157(2), 286-296.
- Chen, Y.-Y., Schmidt, P. and Wang, H.-J. (2014). Consistent estimation of the fixed effects stochastic frontier model. *Journal of Econometrics*, 181(2), 65-76.

---
