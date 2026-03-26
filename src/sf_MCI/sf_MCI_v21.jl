# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

#=
    sf_MCI_v21.jl

    Negative log-likelihood for stochastic frontier models using MCI (Halton) draws.
    Supports multiple noise and inefficiency distribution combinations via multiple dispatch.

    Supported Models:
    - Noise: Normal, StudentT, Laplace
    - Inefficiency: TruncatedNormal, Exponential, HalfNormal, Weibull, Lognormal, Lomax, Rayleigh, Gamma

    Heteroscedasticity Options (via hetero keyword):
    - TruncatedNormal: [:mu, :sigma_sq]
    - Exponential: [:lambda]
    - HalfNormal: [:sigma_sq]
    - Weibull: [:lambda, :k]
    - Lognormal: [:mu, :sigma_sq]
    - Lomax: [:lambda, :alpha]
    - Rayleigh: [:sigma_sq]
    - Gamma: [:k, :theta] (T-approach: GPU + ForwardDiff compatible)

    Features:
    - Multiple dispatch for model combinations (no if-else logic)
    - CPU and GPU support via array type dispatch
    - ForwardDiff compatible via type parameter P
    - Heteroscedastic parameterization via Z matrix
=#

using HaltonSequences
using SpecialFunctions: erf, erfinv, loggamma
using Statistics: mean
using LinearAlgebra: diag, inv
using ForwardDiff: hessian
using Distributions: Normal, TDist, Gamma, quantile, cquantile, ccdf, normlogpdf
using PrettyTables: pretty_table, fmt__printf
using Optim
using ADTypes: AutoForwardDiff, AutoFiniteDiff
using OrderedCollections: OrderedDict

# ============================================================================
# Section 1: Type Hierarchy
# ============================================================================

"""Abstract type for noise models in MCI-based stochastic frontier estimation."""
abstract type MCINoiseModel end

"""Abstract type for inefficiency models in MCI-based stochastic frontier estimation."""
abstract type MCIIneffModel end

"""Normal noise: v ~ N(0, σ_v2)"""
struct NormalNoise_MCI <: MCINoiseModel end

"""Student T noise: v ~ t(0, σ_v, ν) with scale σ_v and degrees of freedom ν > 2"""
struct StudentTNoise_MCI <: MCINoiseModel end

"""Laplace noise: v ~ Laplace(0, b) with scale b"""
struct LaplaceNoise_MCI <: MCINoiseModel end

"""Truncated Normal inefficiency: u ~ TN(μ, σ_u; lower=0)"""
struct TruncatedNormal_MCI <: MCIIneffModel end

"""Exponential inefficiency: u ~ Exp(λ), where λ = Var(u)"""
struct Exponential_MCI <: MCIIneffModel end

"""Half Normal inefficiency: u ~ HalfNormal(σ), i.e., |N(0, σ2)|"""
struct HalfNormal_MCI <: MCIIneffModel end

"""Weibull inefficiency: u ~ Weibull(λ, k) with scale λ and shape k"""
struct Weibull_MCI <: MCIIneffModel end

"""Lognormal inefficiency: u ~ LogNormal(μ, σ)"""
struct Lognormal_MCI <: MCIIneffModel end

"""Lomax inefficiency: u ~ Lomax(α, λ) with shape α and scale λ"""
struct Lomax_MCI <: MCIIneffModel end

"""Rayleigh inefficiency: u ~ Rayleigh(σ)"""
struct Rayleigh_MCI <: MCIIneffModel end

"""Gamma inefficiency: u ~ Gamma(k, θ) with shape k and scale θ"""
struct Gamma_MCI <: MCIIneffModel end

"""Abstract type for copula models in MCI-based stochastic frontier estimation."""
abstract type MCICopulaModel end

"""No copula: independent noise and inefficiency (default)."""
struct NoCopula_MCI <: MCICopulaModel end

"""Gaussian copula: dependence between noise v and inefficiency u via Gaussian copula."""
struct GaussianCopula_MCI <: MCICopulaModel end

"""Clayton copula: models lower tail dependence between v and u via parameter rho > 0."""
struct ClaytonCopula_MCI <: MCICopulaModel end

"""Gumbel copula: models upper tail dependence between v and u via parameter rho >= 1."""
struct GumbelCopula_MCI <: MCICopulaModel end

"""Clayton 90° rotated copula: models upper-lower tail dependence via parameter ρ > 0."""
struct Clayton90Copula_MCI <: MCICopulaModel end

"""
    MCIModel{N<:MCINoiseModel, U<:MCIIneffModel, C<:MCICopulaModel}

Composite type representing a stochastic frontier model with specific noise, inefficiency,
and copula distributions. Enables multiple dispatch on model combinations.
"""
struct MCIModel{N<:MCINoiseModel, U<:MCIIneffModel, C<:MCICopulaModel}
    noise::N
    ineff::U
    copula::C
end

# Registry for symbol-based lookup (used by MCI_nll interface)
# Store Types (not instances) to avoid Julia 1.12+ world age issues
const NOISE_MODELS = Dict{Symbol, Type{<:MCINoiseModel}}(
    :Normal   => NormalNoise_MCI,
    :StudentT => StudentTNoise_MCI,
    :Laplace  => LaplaceNoise_MCI
)

const INEFF_MODELS = Dict{Symbol, Type{<:MCIIneffModel}}(
    :TruncatedNormal => TruncatedNormal_MCI,
    :Exponential     => Exponential_MCI,
    :HalfNormal      => HalfNormal_MCI,
    :Weibull         => Weibull_MCI,
    :Lognormal       => Lognormal_MCI,
    :Lomax           => Lomax_MCI,
    :Rayleigh        => Rayleigh_MCI,
    :Gamma           => Gamma_MCI
)

const COPULA_MODELS = Dict{Symbol, Type{<:MCICopulaModel}}(
    :None     => NoCopula_MCI,
    :Gaussian => GaussianCopula_MCI,
    :Clayton  => ClaytonCopula_MCI,
    :Clayton90 => Clayton90Copula_MCI,
    :Gumbel   => GumbelCopula_MCI,
)

"""
    _build_model(noise::Symbol, ineff::Symbol; copula::Symbol=:None) -> MCIModel

Build a MCIModel from noise, inefficiency, and copula symbols.
Validates that symbols are recognized and provides helpful error messages.
"""
function _build_model(noise::Symbol, ineff::Symbol; copula::Symbol=:None)
    if !haskey(NOISE_MODELS, noise)
        error("Unknown noise model: :$noise. Valid options: $(collect(keys(NOISE_MODELS)))")
    end
    if !haskey(INEFF_MODELS, ineff)
        error("Unknown inefficiency model: :$ineff. Valid options: $(collect(keys(INEFF_MODELS)))")
    end
    if !haskey(COPULA_MODELS, copula)
        error("Unknown copula model: :$copula. Valid options: $(collect(keys(COPULA_MODELS)))")
    end
    if copula != :None && noise == :StudentT
        error("StudentT noise is not yet supported with copula models. " *
              "Use Normal or Laplace noise with copula=:$copula.")
    end
    return MCIModel(NOISE_MODELS[noise](), INEFF_MODELS[ineff](), COPULA_MODELS[copula]())
end

# ============================================================================
# Section 2: Trait Functions (Parameter Counts via Dispatch)
# ============================================================================

"""Number of extra noise parameters beyond ln_sigma_v_sq (or ln_b for Laplace)."""
noise_extras(::NormalNoise_MCI) = 0
noise_extras(::StudentTNoise_MCI) = 1  # ln(ν - 2)
noise_extras(::LaplaceNoise_MCI) = 0   # Just ln_b (uses ln_sigma_v_sq slot)

"""Number of extra inefficiency parameters (beyond main heteroscedastic params)."""
ineff_extras(::TruncatedNormal_MCI) = 1  # ln_sigma_sq (may be heteroscedastic)
ineff_extras(::Exponential_MCI) = 0      # λ params handled via hetero
ineff_extras(::HalfNormal_MCI) = 0       # σ2 params handled via hetero
ineff_extras(::Weibull_MCI) = 0          # λ, k params handled via hetero
ineff_extras(::Lognormal_MCI) = 0        # μ, σ2 params handled via hetero
ineff_extras(::Lomax_MCI) = 0            # λ, α params handled via hetero
ineff_extras(::Rayleigh_MCI) = 0         # σ2 params handled via hetero
ineff_extras(::Gamma_MCI) = 0            # k, θ handled via hetero

"""Does this inefficiency model have a μ parameter?"""
has_mu(::TruncatedNormal_MCI) = true
has_mu(::Exponential_MCI) = false
has_mu(::HalfNormal_MCI) = false
has_mu(::Weibull_MCI) = false
has_mu(::Lognormal_MCI) = true
has_mu(::Lomax_MCI) = false
has_mu(::Rayleigh_MCI) = false
has_mu(::Gamma_MCI) = false

"""Number of copula parameters."""
copula_plen(::NoCopula_MCI) = 0
copula_plen(::GaussianCopula_MCI) = 1
copula_plen(::ClaytonCopula_MCI) = 1  # theta_rho (transformed via exp)
copula_plen(::GumbelCopula_MCI) = 1   # theta_rho (transformed via exp + 1)
copula_plen(::Clayton90Copula_MCI) = 1  # theta_rho (transformed via exp, same as Clayton)

"""Valid heteroscedasticity options for each inefficiency model."""
valid_hetero(::TruncatedNormal_MCI) = [:mu, :sigma_sq]
valid_hetero(::Exponential_MCI) = [:lambda]
valid_hetero(::HalfNormal_MCI) = [:sigma_sq]
valid_hetero(::Weibull_MCI) = [:lambda, :k]
valid_hetero(::Lognormal_MCI) = [:mu, :sigma_sq]
valid_hetero(::Lomax_MCI) = [:lambda, :alpha]
valid_hetero(::Rayleigh_MCI) = [:sigma_sq]
valid_hetero(::Gamma_MCI) = [:k, :theta]

"""
    _validate_hetero(ineff::MCIIneffModel, hetero::Vector{Symbol})

Validate that hetero options are valid for the given inefficiency model.
Raises an error with helpful message if invalid options are provided.
"""
function _validate_hetero(ineff::MCIIneffModel, hetero::Vector{Symbol})
    valid = valid_hetero(ineff)
    for h in hetero
        if h ∉ valid
            error("Invalid hetero option :$h for $(typeof(ineff)). " *
                  "Valid options: $(valid)")
        end
    end
end

"""
    plen(model::MCIModel, K::Int, L::Int, hetero::Vector{Symbol})

Calculate total parameter vector length for a given model and heteroscedasticity settings.
"""
function plen(model::MCIModel, K::Int, L::Int, hetero::Vector{Symbol}; L_scaling::Int=0)
    n_beta = K
    n_delta = L_scaling
    n_noise = 1 + noise_extras(model.noise)  # ln_sigma_v_sq + extras (e.g., ln_nu_minus_2)
    n_ineff = ineff_plen(model.ineff, L, hetero)
    n_copula = copula_plen(model.copula)
    return n_beta + n_delta + n_noise + n_ineff + n_copula
end

function ineff_plen(::TruncatedNormal_MCI, L::Int, hetero::Vector{Symbol})
    n_mu = :mu in hetero ? L : 1
    n_sigma_u = :sigma_sq in hetero ? L : 1
    return n_mu + n_sigma_u
end

function ineff_plen(::Exponential_MCI, L::Int, hetero::Vector{Symbol})
    n_lambda = :lambda in hetero ? L : 1
    return n_lambda
end

function ineff_plen(::HalfNormal_MCI, L::Int, hetero::Vector{Symbol})
    return :sigma_sq in hetero ? L : 1
end

function ineff_plen(::Weibull_MCI, L::Int, hetero::Vector{Symbol})
    n_lambda = :lambda in hetero ? L : 1
    n_k = :k in hetero ? L : 1
    return n_lambda + n_k
end

function ineff_plen(::Lognormal_MCI, L::Int, hetero::Vector{Symbol})
    n_mu = :mu in hetero ? L : 1
    n_sigma = :sigma_sq in hetero ? L : 1
    return n_mu + n_sigma
end

function ineff_plen(::Lomax_MCI, L::Int, hetero::Vector{Symbol})
    n_lambda = :lambda in hetero ? L : 1
    n_alpha = :alpha in hetero ? L : 1
    return n_lambda + n_alpha
end

function ineff_plen(::Rayleigh_MCI, L::Int, hetero::Vector{Symbol})
    return :sigma_sq in hetero ? L : 1
end

function ineff_plen(::Gamma_MCI, L::Int, hetero::Vector{Symbol})
    n_k = :k in hetero ? L : 1
    n_theta = :theta in hetero ? L : 1
    return n_k + n_theta
end

# ============================================================================
# Section 2b: Parameter Index System (for SFGPU-style parameter access)
# ============================================================================

"""
    _param_ind(model::MCIModel, K::Int, L::Int, hetero::Vector{Symbol})

Compute parameter indices (ranges) for a given model and heteroscedasticity settings.
Returns a NamedTuple with index ranges for beta, noise, and inefficiency parameters.
This enables direct indexing into CPU parameter vectors (no views into CuArray needed).
"""
function _param_ind(model::MCIModel, K::Int, L::Int, hetero::Vector{Symbol}; L_scaling::Int=0)
    idx = 1

    # Beta indices (K elements)
    beta = idx:(idx+K-1)
    idx += K

    # Scaling δ indices (L_scaling elements, empty range if no scaling)
    delta = L_scaling > 0 ? (idx:(idx+L_scaling-1)) : (1:0)
    idx += L_scaling

    # Inefficiency parameter indices via dispatch (before noise, to match eq_names/eq_indices convention)
    ineff_idx, idx = _ineff_ind(model.ineff, idx, L, hetero)

    # Noise parameter indices via dispatch
    noise_idx, idx = _noise_ind(model.noise, idx)

    # Copula parameter indices via dispatch
    copula_idx, idx = _copula_ind(model.copula, idx)

    return (beta=beta, delta=delta, noise=noise_idx, ineff=ineff_idx, copula=copula_idx)
end

# --- Noise parameter indices ---

function _noise_ind(::NormalNoise_MCI, idx)
    ln_sigma_v_sq = idx
    return (ln_sigma_v_sq=ln_sigma_v_sq,), idx + 1
end

function _noise_ind(::StudentTNoise_MCI, idx)
    ln_sigma_v_sq = idx
    ln_nu_minus_2 = idx + 1
    return (ln_sigma_v_sq=ln_sigma_v_sq, ln_nu_minus_2=ln_nu_minus_2), idx + 2
end

function _noise_ind(::LaplaceNoise_MCI, idx)
    ln_b = idx
    return (ln_b=ln_b,), idx + 1
end

# --- Inefficiency parameter indices ---

function _ineff_ind(::TruncatedNormal_MCI, idx, L, hetero)
    mu_hetero = :mu in hetero
    sigma_u_hetero = :sigma_sq in hetero

    n_mu = mu_hetero ? L : 1
    mu = idx:(idx+n_mu-1)
    idx += n_mu

    n_sigma_u = sigma_u_hetero ? L : 1
    sigma_u = idx:(idx+n_sigma_u-1)
    idx += n_sigma_u

    return (mu=mu, sigma_u=sigma_u, mu_hetero=mu_hetero, sigma_u_hetero=sigma_u_hetero), idx
end

function _ineff_ind(::Exponential_MCI, idx, L, hetero)
    lambda_hetero = :lambda in hetero
    n_lambda = lambda_hetero ? L : 1
    lambda = idx:(idx+n_lambda-1)
    idx += n_lambda

    return (lambda=lambda, lambda_hetero=lambda_hetero), idx
end

function _ineff_ind(::HalfNormal_MCI, idx, L, hetero)
    sigma_sq_hetero = :sigma_sq in hetero
    n_sigma = sigma_sq_hetero ? L : 1
    sigma_sq = idx:(idx+n_sigma-1)
    idx += n_sigma

    return (sigma_sq=sigma_sq, sigma_sq_hetero=sigma_sq_hetero), idx
end

function _ineff_ind(::Weibull_MCI, idx, L, hetero)
    lambda_hetero = :lambda in hetero
    k_hetero = :k in hetero

    n_lambda = lambda_hetero ? L : 1
    lambda = idx:(idx+n_lambda-1)
    idx += n_lambda

    n_k = k_hetero ? L : 1
    k = idx:(idx+n_k-1)
    idx += n_k

    return (lambda=lambda, k=k, lambda_hetero=lambda_hetero, k_hetero=k_hetero), idx
end

function _ineff_ind(::Lognormal_MCI, idx, L, hetero)
    mu_hetero = :mu in hetero
    sigma_sq_hetero = :sigma_sq in hetero

    n_mu = mu_hetero ? L : 1
    mu = idx:(idx+n_mu-1)
    idx += n_mu

    n_sigma = sigma_sq_hetero ? L : 1
    sigma_sq = idx:(idx+n_sigma-1)
    idx += n_sigma

    return (mu=mu, sigma_sq=sigma_sq, mu_hetero=mu_hetero, sigma_sq_hetero=sigma_sq_hetero), idx
end

function _ineff_ind(::Lomax_MCI, idx, L, hetero)
    lambda_hetero = :lambda in hetero
    n_lambda = lambda_hetero ? L : 1
    ln_lambda = idx:(idx+n_lambda-1)
    idx += n_lambda

    alpha_hetero = :alpha in hetero
    n_alpha = alpha_hetero ? L : 1
    alpha = idx:(idx+n_alpha-1)
    idx += n_alpha

    return (ln_lambda=ln_lambda, alpha=alpha, lambda_hetero=lambda_hetero, alpha_hetero=alpha_hetero), idx
end

function _ineff_ind(::Rayleigh_MCI, idx, L, hetero)
    sigma_sq_hetero = :sigma_sq in hetero
    n_sigma = sigma_sq_hetero ? L : 1
    sigma_sq = idx:(idx+n_sigma-1)
    idx += n_sigma

    return (sigma_sq=sigma_sq, sigma_sq_hetero=sigma_sq_hetero), idx
end

function _ineff_ind(::Gamma_MCI, idx, L, hetero)
    k_hetero = :k in hetero
    theta_hetero = :theta in hetero

    n_k = k_hetero ? L : 1
    k = idx:(idx+n_k-1)
    idx += n_k

    n_theta = theta_hetero ? L : 1
    theta = idx:(idx+n_theta-1)
    idx += n_theta

    return (k=k, theta=theta, k_hetero=k_hetero, theta_hetero=theta_hetero), idx
end

# --- Copula parameter indices ---

_copula_ind(::NoCopula_MCI, idx) = (NamedTuple(), idx)

function _copula_ind(::GaussianCopula_MCI, idx)
    return (theta_rho=idx,), idx + 1
end

function _copula_ind(::ClaytonCopula_MCI, idx)
    theta_rho = idx
    return (theta_rho=theta_rho,), idx + 1
end

function _copula_ind(::GumbelCopula_MCI, idx)
    theta_rho = idx
    return (theta_rho=theta_rho,), idx + 1
end

function _copula_ind(::Clayton90Copula_MCI, idx)
    theta_rho = idx
    return (theta_rho=theta_rho,), idx + 1
end

# ============================================================================
# Section 2c: Model Specification (sfmodel_MCI_spec)
# ============================================================================

"""
    sfmodel_MCI_spec{T<:AbstractFloat}

Stochastic Frontier Model Specification.
Holds all model configuration and data for convenient function calls.

# Fields
## Data
- `Y::AbstractVector{T}`: Response vector (N observations)
- `X::AbstractMatrix{T}`: Frontier design matrix (N x K)
- `Z::AbstractMatrix{T}`: Heteroscedasticity design matrix (N x L)

## Model specification
- `noise::Symbol`: Noise distribution (:Normal, :StudentT, :Laplace)
- `ineff::Symbol`: Inefficiency distribution
- `hetero::Vector{Symbol}`: Heteroscedasticity options

## Pre-computed
- `draws::AbstractVector{T}`: Halton probabilities (from make_halton_p)
- `constants::NamedTuple`: Precomputed constants

## Metadata (for print_table)
- `varnames::Vector{String}`: Variable names for each coefficient
- `eqnames::Vector{String}`: Equation block names
- `eq_indices::Vector{Int}`: Starting index for each equation block

## Derived (computed on construction)
- `N::Int`: Number of observations
- `K::Int`: Number of frontier regressors
- `L::Int`: Number of Z columns
- `model::MCIModel`: Built model object
- `idx::NamedTuple`: Parameter indices

# Example
```julia
spec = sfmodel_MCI_spec(
    depvar = y, frontier = X, zvar = Z,
    noise = :Normal,
    ineff = :TruncatedNormal,
    hetero = [:mu],
    varnames = ["_cons", "x1", "x2", "_cons", "_cons"],
    eqnames = ["frontier", "μ", "ln_σᵤ²", "ln_σᵥ²"],
    eq_indices = [1, 4, 5, 6]
)

# Use with simplified function calls
nll = MCI_nll(spec, p)
eff = jlms_bc_indices(spec, p)
print_table(spec, coef, vcov)
```
"""
mutable struct sfmodel_MCI_spec{T<:AbstractFloat}
    # Data
    depvar::AbstractVector{T}
    frontier::AbstractMatrix{T}
    zvar::AbstractMatrix{T}

    # Model specification
    noise::Symbol
    ineff::Symbol
    copula::Symbol
    hetero::Vector{Symbol}

    # Pre-computed
    draws::AbstractVector{T}
    draws_2D::AbstractMatrix{T}  # Pre-reshaped draws: 1 x D (multiRand=false) or N x D (multiRand=true)
    multiRand::Bool              # true = per-observation draws (N x D), false = shared draws (1 x D)
    constants::NamedTuple

    # Metadata (for print_table)
    varnames::Vector{String}
    eqnames::Vector{String}
    eq_indices::Vector{Int}

    # Derived dimensions (computed)
    N::Int
    K::Int
    L::Int
    model::MCIModel
    idx::NamedTuple
    sign::Int  # 1 for production frontier, -1 for cost frontier
    chunks::Int  # Number of chunks for GPU memory management
    transformation::Symbol  # :expo_rule, :logistic_1_rule, :logistic_2_rule
    scaling::Bool                                    # true if scaling property model
    scaling_zvar::Union{Nothing, AbstractMatrix{T}}  # Z matrix for h(z)=exp(z'δ)
    L_scaling::Int                                   # number of scaling Z columns (0 if no scaling)
end

# ============================================================================
# New API Structs: Separated Model Specification and Method Specification
# ============================================================================

"""
    SFModelSpec{T<:AbstractFloat}

Model specification struct returned by `sfmodel_spec()`. Contains data, distributional
assumptions, variable names, and derived dimensions — but NOT numerical method settings
(draws, GPU, chunks, transformation), which are specified separately via `sfmodel_method()`.

# Fields
- `depvar`: Dependent variable vector (normalized, CPU)
- `frontier`: Frontier regressor matrix (normalized, CPU)
- `zvar`: Z variable matrix (normalized, CPU; auto-generated as ones if not provided)
- `noise`: Noise distribution symbol (e.g., `:Normal`, `:StudentT`, `:Laplace`)
- `ineff`: Inefficiency distribution symbol (e.g., `:HalfNormal`, `:TruncatedNormal`)
- `hetero`: Heteroscedastic parameters (e.g., `[:mu, :sigma_sq]`)
- `varnames`: Parameter variable names for output tables
- `eqnames`: Equation names for output tables
- `eq_indices`: Indices mapping variables to equations
- `N`: Number of observations
- `K`: Number of frontier regressors
- `L`: Number of Z columns
- `model`: `MCIModel` built from noise + ineff
- `idx`: Parameter indices (NamedTuple)
- `sign`: Frontier sign (1 for production, -1 for cost)
"""
struct SFModelSpec{T<:AbstractFloat}
    depvar::AbstractVector{T}
    frontier::AbstractMatrix{T}
    zvar::AbstractMatrix{T}
    noise::Symbol
    ineff::Symbol
    copula::Symbol
    hetero::Vector{Symbol}
    varnames::Vector{String}
    eqnames::Vector{String}
    eq_indices::Vector{Int}
    N::Int
    K::Int
    L::Int
    model::MCIModel
    idx::NamedTuple
    sign::Int
    scaling::Bool                                    # true if scaling property model
    scaling_zvar::Union{Nothing, AbstractMatrix{T}}  # Z matrix for h(z)=exp(z'δ)
    L_scaling::Int                                   # number of scaling Z columns (0 if no scaling)
end

"""
    SFMethodSpec

Numerical method specification struct returned by `sfmodel_method()`. Contains settings
for the estimation method (MCI), draw generation, GPU usage, and transformation rules.

# Fields
- `method`: Estimation method symbol (default `:MCI`)
- `transformation`: Transformation rule (`:expo_rule`, `:logistic_1_rule`, etc.), or `nothing` for default
- `draws`: User-provided draws, or `nothing` to auto-generate Halton sequences
- `n_draws`: Number of draws (default 1024)
- `multiRand`: Per-observation draws (`true`, N×D) or shared draws (`false`, 1×D)
- `GPU`: Whether to use GPU acceleration
- `chunks`: Number of chunks for GPU memory management
- `distinct_Halton_length`: Maximum Halton sequence length for multiRand mode (default 2^15-1 = 32767)
"""
struct SFMethodSpec
    method::Symbol
    transformation::Union{Symbol, Nothing}
    draws::Union{Nothing, AbstractVecOrMat{<:AbstractFloat}}
    n_draws::Int
    multiRand::Bool
    GPU::Bool
    chunks::Int
    distinct_Halton_length::Int
end

# ============================================================================
# Initial Value Specification Types and Functions
# ============================================================================

"""
Abstract parent type for all initial value specifications used by `sfmodel_MCI_init()`.
"""
abstract type InitSpec end

struct FrontierInit <: InitSpec
    values::Vector{Float64}
end

struct MuInit <: InitSpec
    values::Vector{Float64}
end

struct LnSigmaSqInit <: InitSpec      # For inefficiency sigma_sq (ln_sigma_sq)
    values::Vector{Float64}
end

struct LnSigmaVSqInit <: InitSpec     # For noise sigma_v_sq (ln_sigma_v_sq)
    values::Vector{Float64}
end

struct LnNuMinus2Init <: InitSpec     # For StudentT nu parameter
    values::Vector{Float64}
end

struct LnBInit <: InitSpec            # For Laplace b parameter
    values::Vector{Float64}
end

struct LambdaInit <: InitSpec         # For Exponential/Weibull lambda
    values::Vector{Float64}
end

struct KInit <: InitSpec              # For Weibull k
    values::Vector{Float64}
end

struct AlphaInit <: InitSpec          # For Lomax alpha
    values::Vector{Float64}
end

struct ThetaInit <: InitSpec          # For Gamma θ
    values::Vector{Float64}
end

struct ThetaRhoInit <: InitSpec      # For copula θ_ρ
    values::Vector{Float64}
end

struct ScalingInit <: InitSpec       # For scaling function δ coefficients
    values::Vector{Float64}
end

# Helper to normalize any input to Vector{Float64}
_to_vec(x::Real) = [Float64(x)]
_to_vec(x::AbstractVector) = Float64.(x)
_to_vec(x::Real, xs::Real...) = Float64[x, xs...]

# Helper to convert various user inputs to Vector{Float64} for init values
# Handles: [0.0, 0.0] (vector), [0.0 0.0] (row vector/matrix), (0.0, 0.0) (tuple)
_to_init_vec(x::AbstractVector) = Float64.(vec(x))
_to_init_vec(x::AbstractMatrix) = Float64.(vec(x))
_to_init_vec(x::Tuple) = Float64.(collect(x))
_to_init_vec(x::Real) = [Float64(x)]
_to_init_vec(::Nothing) = nothing

# ============================================================================
# Input Normalization Helpers (GPU-compatible)
# ============================================================================

# For depvar: unwrap [yvar] × yvar using only()
_to_vector(x::AbstractVector{<:Real}) = x
_to_vector(x::AbstractVector{<:AbstractVector}) = only(x)
_to_vector(x::AbstractMatrix{<:Real}) = vec(x)  # handle N×1 matrix input

# For frontier/zvar: convert [v1, v2, v3] × matrix using reduce(hcat, ...)
# Note: use reduce(hcat, x) instead of stack(x) to preserve CuArray type for GPU
_to_matrix(x::AbstractMatrix) = x
_to_matrix(x::AbstractVector{<:AbstractVector}) = reduce(hcat, x)
_to_matrix(x::AbstractVector{<:Real}) = reshape(x, :, 1)  # handle plain vector as N×1 matrix

# Convert array to same device (CPU/GPU) as target array
# Uses similar() and copyto!() for generic AbstractArray compatibility
function _to_device_array(target::AbstractArray, source::AbstractVector)
    dest = similar(target, eltype(source), size(source))
    copyto!(dest, source)
    return dest
end

function _to_device_array(target::AbstractArray, source::AbstractMatrix)
    dest = similar(target, eltype(source), size(source))
    copyto!(dest, source)
    return dest
end

# GPU conversion helper: optionally move data arrays to GPU
function _maybe_gpu_convert(depvar::AbstractVector, frontier::AbstractMatrix,
                            zvar::AbstractMatrix, gpu::Bool)
    if gpu
        if !isdefined(Main, :CUDA)
            error("GPU=true requires CUDA.jl to be loaded. Please run `using CUDA` before calling this function.")
        end
        return Main.CUDA.CuArray(depvar), Main.CUDA.CuArray(frontier), Main.CUDA.CuArray(zvar)
    end
    return depvar, frontier, zvar
end

# Draw preparation helper: generate or validate draws for MCI integration
function _prepare_draws(depvar_ref::AbstractVector{T}, N::Int, ::Type{T},
                        user_draws, n_draws::Int, multiRand::Bool,
                        distinct_Halton_length::Int=2^15-1) where {T}
    if isnothing(user_draws)
        if multiRand
            halton_cpu = make_halton_wrap(N, n_draws; T=T, distinct_Halton_length=distinct_Halton_length)
            draws_2D = _to_device_array(depvar_ref, halton_cpu)
            draws_vec = vec(draws_2D)
        else
            halton_cpu = make_halton_p(n_draws; T=T)
            draws_vec = _to_device_array(depvar_ref, halton_cpu)
            draws_2D = reshape(draws_vec, 1, length(draws_vec))
        end
    else
        # User-provided draws
        depvar_is_gpu = string(typeof(depvar_ref).name.wrapper) == "CuArray"
        draws_is_gpu = string(typeof(user_draws).name.wrapper) == "CuArray"
        if depvar_is_gpu != draws_is_gpu
            @warn "Type inconsistency: `depvar` is on $(depvar_is_gpu ? "GPU" : "CPU") but `draws` is on $(draws_is_gpu ? "GPU" : "CPU"). " *
                  "Consider letting the program auto-generate Halton sequences with the correct device type by specifying `n_draws` instead of providing `draws`."
        end

        if multiRand
            if ndims(user_draws) == 1
                @warn "multiRand=true but `draws` is a 1D vector. Generating wrapped N x D matrix from `n_draws` instead."
                halton_cpu = make_halton_wrap(N, n_draws; T=T, distinct_Halton_length=distinct_Halton_length)
                draws_2D = _to_device_array(depvar_ref, halton_cpu)
                draws_vec = vec(draws_2D)
            elseif size(user_draws, 1) == N
                draws_2D = _to_device_array(depvar_ref, T.(user_draws))
                draws_vec = vec(draws_2D)
            else
                error("When multiRand=true, `draws` must be an N x D matrix with N=$N rows, got size $(size(user_draws))")
            end
        else
            draws_vec = T.(vec(user_draws))
            draws_2D = reshape(draws_vec, 1, length(draws_vec))
        end
    end

    return draws_vec, draws_2D
end

# Assemble the internal sfmodel_MCI_spec from SFModelSpec + SFMethodSpec
# Performs deferred computations: GPU conversion, draw generation, constants, transformation
function _assemble_MCI_spec(spec::SFModelSpec{T}, method::SFMethodSpec) where {T}
    # 1. GPU conversion
    depvar, frontier, zvar = _maybe_gpu_convert(spec.depvar, spec.frontier, spec.zvar, method.GPU)

    # 2. Prepare draws
    draws_vec, draws_2D = _prepare_draws(depvar, spec.N, T,
        method.draws, method.n_draws, method.multiRand, method.distinct_Halton_length)

    # 3. Constants
    constants = make_constants(spec.model, T)

    # 4. Resolve transformation (use default for distribution if not specified)
    trans_rule = isnothing(method.transformation) ?
        default_transformation_rule(spec.ineff) : method.transformation

    # Validate transformation
    trans_rule in (:expo_rule, :logistic_1_rule, :logistic_2_rule) ||
        error("Invalid `transformation`: $trans_rule. " *
              "Use :expo_rule, :logistic_1_rule, or :logistic_2_rule.")

    # 5. GPU-convert scaling_zvar if applicable
    scaling_zvar_dev = if spec.scaling && method.GPU
        _to_device_array(depvar, spec.scaling_zvar)
    else
        spec.scaling_zvar
    end

    # 6. Build internal sfmodel_MCI_spec
    return sfmodel_MCI_spec{T}(
        depvar, frontier, zvar,
        spec.noise, spec.ineff, spec.copula, spec.hetero,
        draws_vec, draws_2D, method.multiRand, constants,
        spec.varnames, spec.eqnames, spec.eq_indices,
        spec.N, spec.K, spec.L, spec.model, spec.idx,
        spec.sign, method.chunks, trans_rule,
        spec.scaling, scaling_zvar_dev, spec.L_scaling
    )
end

# Constructor functions - accept scalar(s) or vector
frontier(x::Real, xs::Real...)    = FrontierInit(_to_vec(x, xs...))
frontier(v::AbstractVector)       = FrontierInit(_to_vec(v))
frontier(x::Real)                 = FrontierInit(_to_vec(x))

mu(x::Real, xs::Real...)          = MuInit(_to_vec(x, xs...))
mu(v::AbstractVector)             = MuInit(_to_vec(v))
mu(x::Real)                       = MuInit(_to_vec(x))

ln_sigma_sq(x::Real, xs::Real...)     = LnSigmaSqInit(_to_vec(x, xs...))
ln_sigma_sq(v::AbstractVector)        = LnSigmaSqInit(_to_vec(v))
ln_sigma_sq(x::Real)                  = LnSigmaSqInit(_to_vec(x))

ln_sigma_v_sq(x::Real, xs::Real...)   = LnSigmaVSqInit(_to_vec(x, xs...))
ln_sigma_v_sq(v::AbstractVector)      = LnSigmaVSqInit(_to_vec(v))
ln_sigma_v_sq(x::Real)                = LnSigmaVSqInit(_to_vec(x))

ln_nu_minus_2(x::Real, xs::Real...)   = LnNuMinus2Init(_to_vec(x, xs...))
ln_nu_minus_2(v::AbstractVector)      = LnNuMinus2Init(_to_vec(v))
ln_nu_minus_2(x::Real)                = LnNuMinus2Init(_to_vec(x))

ln_b(x::Real, xs::Real...)            = LnBInit(_to_vec(x, xs...))
ln_b(v::AbstractVector)               = LnBInit(_to_vec(v))
ln_b(x::Real)                         = LnBInit(_to_vec(x))

lambda(x::Real, xs::Real...)          = LambdaInit(_to_vec(x, xs...))
lambda(v::AbstractVector)             = LambdaInit(_to_vec(v))
lambda(x::Real)                       = LambdaInit(_to_vec(x))

k(x::Real, xs::Real...)               = KInit(_to_vec(x, xs...))
k(v::AbstractVector)                  = KInit(_to_vec(v))
k(x::Real)                            = KInit(_to_vec(x))

alpha(x::Real, xs::Real...)           = AlphaInit(_to_vec(x, xs...))
alpha(v::AbstractVector)              = AlphaInit(_to_vec(v))
alpha(x::Real)                        = AlphaInit(_to_vec(x))

theta(x::Real, xs::Real...)           = ThetaInit(_to_vec(x, xs...))
theta(v::AbstractVector)              = ThetaInit(_to_vec(v))
theta(x::Real)                        = ThetaInit(_to_vec(x))

theta_rho(x::Real, xs::Real...)      = ThetaRhoInit(_to_vec(x, xs...))
theta_rho(v::AbstractVector)         = ThetaRhoInit(_to_vec(v))
theta_rho(x::Real)                   = ThetaRhoInit(_to_vec(x))

scaling(x::Real, xs::Real...)        = ScalingInit(_to_vec(x, xs...))
scaling(v::AbstractVector)           = ScalingInit(_to_vec(v))
scaling(x::Real)                     = ScalingInit(_to_vec(x))

"""
    _get_init(spec::SFModelSpec)

Determine required init specifications based on the model specification.
Returns a vector of tuples: (Type, name_string, expected_length).
"""
function _get_init(spec::SFModelSpec)
    required = Vector{Tuple{Type, String, Int}}()

    # Helper: returns spec.L if symbol is heteroscedastic, otherwise 1
    hetero_len(sym::Symbol) = sym in spec.hetero ? spec.L : 1

    # Always need frontier (K parameters)
    push!(required, (FrontierInit, "frontier", spec.K))

    # Scaling δ parameters (after frontier, before inefficiency)
    if spec.scaling
        push!(required, (ScalingInit, "scaling", spec.L_scaling))
    end

    # Inefficiency parameters (before noise, to match _param_ind / eq_names / eq_indices convention)
    # Grouped by required parameters (not by mathematical similarity)
    if spec.ineff in (:TruncatedNormal, :Lognormal)
        push!(required, (MuInit, "mu", hetero_len(:mu)))
        push!(required, (LnSigmaSqInit, "ln_sigma_sq", hetero_len(:sigma_sq)))

    elseif spec.ineff in (:HalfNormal, :Rayleigh)
        push!(required, (LnSigmaSqInit, "ln_sigma_sq", hetero_len(:sigma_sq)))

    elseif spec.ineff == :Exponential
        push!(required, (LambdaInit, "ln_lambda", hetero_len(:lambda)))

    elseif spec.ineff == :Weibull
        push!(required, (LambdaInit, "ln_lambda", hetero_len(:lambda)))
        push!(required, (KInit, "ln_k", hetero_len(:k)))

    elseif spec.ineff == :Lomax
        push!(required, (LambdaInit, "ln_lambda", hetero_len(:lambda)))
        push!(required, (AlphaInit, "ln_alpha", hetero_len(:alpha)))

    elseif spec.ineff == :Gamma
        push!(required, (KInit, "ln_k", hetero_len(:k)))
        push!(required, (ThetaInit, "ln_theta", hetero_len(:theta)))
    end

    # Noise parameters (after inefficiency)
    if spec.noise in (:Normal, :StudentT)
        push!(required, (LnSigmaVSqInit, "ln_sigma_v_sq", 1))
        if spec.noise == :StudentT
            push!(required, (LnNuMinus2Init, "ln_nu_minus_2", 1))
        end
    elseif spec.noise == :Laplace
        push!(required, (LnBInit, "ln_b", 1))
    end

    # Copula parameters (after noise)
    if spec.copula in (:Gaussian, :Clayton, :Clayton90, :Gumbel)
        push!(required, (ThetaRhoInit, "theta_rho", 1))
    end

    return required
end

"""
    _build_init(spec::SFModelSpec, init_dict, required)

Build the parameter vector in the correct order from the init specifications.
"""
function _build_init(spec::SFModelSpec, init_dict, required)
    p = Float64[]
    for (req_type, _, _) in required
        append!(p, init_dict[req_type].values)
    end
    return p
end

"""
    sfmodel_MCI_init(; model, init=nothing, frontier=nothing, mu=nothing, ...)

Create an initial parameter vector for optimization based on user-specified initial values.
All arguments are keyword arguments. Initial values can be specified as vectors, row vectors, or tuples.

Two usage modes are supported:

1. **Full vector mode**: Supply the entire initial value vector directly via `init`.
2. **Component mode**: Supply individual components (frontier, mu, ln_sigma_sq, etc.).

# Arguments
- `spec::SFModelSpec`: The model specification (required)
- `init`: Complete initial value vector (optional). If provided, other parameters are ignored.
         Can be vector, row vector, or tuple. Length must match the number of model parameters.
- `frontier`: Frontier coefficients (K values), can be vector, row vector, or tuple
- `mu`: Mu parameter for TruncatedNormal, Lognormal (if applicable)
- `ln_sigma_sq`: Log sigma squared for TruncatedNormal, HalfNormal, Lognormal, Rayleigh
- `ln_sigma_v_sq`: Log noise variance for Normal, StudentT
- `ln_nu_minus_2`: Log(nu-2) for StudentT
- `ln_b`: Log scale for Laplace
- `ln_lambda`: Log lambda for Exponential/Weibull
- `ln_k`: Log shape k for Weibull/Gamma
- `ln_lambda`: Log lambda (scale) for Lomax/Exponential/Weibull
- `ln_alpha`: Log alpha for Lomax
- `ln_theta`: Log theta for Gamma
- `message::Bool=true`: If true, issue a warning when `init` is provided along with other parameters.

# Examples
```julia
# Full vector mode: supply complete initial values
myinit = sfmodel_MCI_init(
    spec = myspec,
    init = [0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0]
)

# Component mode: supply individual parameters
myinit = sfmodel_MCI_init(
    spec = myspec,
    frontier = [0.5, 0.3, 0.2],   # K=3 coefficients (vector)
    mu = [0.1, 0.1, 0.1],         # L=3 with :mu hetero
    ln_sigma_sq = [0.0],          # scalar as single-element vector
    ln_sigma_v_sq = (0.0,)        # can also use tuple
)
# Row vectors are also supported: frontier = [0.5 0.3 0.2]
```
"""
function sfmodel_MCI_init(;
    spec::SFModelSpec,
    init = nothing,
    frontier = nothing,
    scaling = nothing,
    mu = nothing,
    ln_sigma_sq = nothing,
    ln_sigma_v_sq = nothing,
    ln_nu_minus_2 = nothing,
    ln_b = nothing,
    ln_lambda = nothing,
    ln_k = nothing,
    ln_alpha = nothing,
    ln_theta = nothing,
    theta_rho = nothing,
    message::Bool = true
)
    # Internal aliases: user-facing names use ln_ prefix, internal code uses bare names
    lambda = ln_lambda
    k = ln_k
    alpha = ln_alpha
    theta = ln_theta

    # Mode 1: Full vector mode - user supplies complete initial value vector
    if init !== nothing
        # Check if any component-specific parameters were also provided
        if message && any(x -> x !== nothing, (frontier, scaling, mu, ln_sigma_sq, ln_sigma_v_sq,
                                                ln_nu_minus_2, ln_b, lambda, k, alpha, theta, theta_rho))
            @warn "Using `init` instead of function-specific init."
        end
        init_vec = _to_init_vec(init)
        expected_len = plen(spec.model, spec.K, spec.L, spec.hetero; L_scaling=spec.L_scaling)
        if length(init_vec) != expected_len
            error("Length mismatch in sfmodel_MCI_init(): expected $expected_len parameters, got $(length(init_vec)).")
        end
        return init_vec
    end

    # Mode 2: Component mode - user supplies individual parameters
    if frontier === nothing
        error("Either `init` (full vector) or `frontier` (with other components) must be provided.")
    end

    # Convert all inputs to Vector{Float64} (or nothing)
    frontier_vec = _to_init_vec(frontier)
    scaling_vec = _to_init_vec(scaling)
    mu_vec = _to_init_vec(mu)
    ln_sigma_sq_vec = _to_init_vec(ln_sigma_sq)
    ln_sigma_v_sq_vec = _to_init_vec(ln_sigma_v_sq)
    ln_nu_minus_2_vec = _to_init_vec(ln_nu_minus_2)
    ln_b_vec = _to_init_vec(ln_b)
    lambda_vec = _to_init_vec(lambda)
    k_vec = _to_init_vec(k)
    alpha_vec = _to_init_vec(alpha)
    theta_vec = _to_init_vec(theta)
    theta_rho_vec = _to_init_vec(theta_rho)

    # Build init_dict mapping Type -> InitSpec
    init_dict = Dict{Type, InitSpec}()
    if frontier_vec !== nothing
        init_dict[FrontierInit] = FrontierInit(frontier_vec)
    end
    if scaling_vec !== nothing
        init_dict[ScalingInit] = ScalingInit(scaling_vec)
    end
    if mu_vec !== nothing
        init_dict[MuInit] = MuInit(mu_vec)
    end
    if ln_sigma_sq_vec !== nothing
        init_dict[LnSigmaSqInit] = LnSigmaSqInit(ln_sigma_sq_vec)
    end
    if ln_sigma_v_sq_vec !== nothing
        init_dict[LnSigmaVSqInit] = LnSigmaVSqInit(ln_sigma_v_sq_vec)
    end
    if ln_nu_minus_2_vec !== nothing
        init_dict[LnNuMinus2Init] = LnNuMinus2Init(ln_nu_minus_2_vec)
    end
    if ln_b_vec !== nothing
        init_dict[LnBInit] = LnBInit(ln_b_vec)
    end
    if lambda_vec !== nothing
        init_dict[LambdaInit] = LambdaInit(lambda_vec)
    end
    if k_vec !== nothing
        init_dict[KInit] = KInit(k_vec)
    end
    if alpha_vec !== nothing
        init_dict[AlphaInit] = AlphaInit(alpha_vec)
    end
    if theta_vec !== nothing
        init_dict[ThetaInit] = ThetaInit(theta_vec)
    end
    if theta_rho_vec !== nothing
        init_dict[ThetaRhoInit] = ThetaRhoInit(theta_rho_vec)
    end

    # Determine required equations based on model spec
    required = _get_init(spec)

    # Validate: check all required are provided
    missing_inits = String[]
    length_mismatches = String[]

    for (req_type, req_name, expected_len) in required
        if !haskey(init_dict, req_type)
            push!(missing_inits, "$req_name = [...]")
        else
            # Check length
            provided_len = length(init_dict[req_type].values)
            if provided_len != expected_len
                push!(length_mismatches, "$req_name: expected $expected_len, got $provided_len")
            end
        end
    end

    # Report all missing inits at once
    if !isempty(missing_inits)
        missing_list = join(missing_inits, ", ")
        error("Missing required init(s) in sfmodel_MCI_init(): $missing_list.")
    end

    # Report all length mismatches at once
    if !isempty(length_mismatches)
        mismatch_list = join(length_mismatches, "; ")
        error("Length mismatch(es) in sfmodel_MCI_init(): $mismatch_list.")
    end

    # Check for extra (unused) specifications
    provided_types = Set(keys(init_dict))
    required_types = Set(r[1] for r in required)
    extras = setdiff(provided_types, required_types)
    if !isempty(extras)
        extra_names = join([string(e) for e in extras], ", ")
        error("Unused specification(s) in sfmodel_MCI_init(): $extra_names. These are not required for the current model.")
    end

    # Build parameter vector in correct order
    return _build_init(spec, init_dict, required)
end

# ============================================================================
# End of Initial Value Specification
# ============================================================================

# ============================================================================
# Optimization Options Specification (sfmodel_MCI_opt)
# ============================================================================

"""
    sfmodel_MCI_optim

Struct holding all optimization settings for `sfmodel_MCI_fit()`.

# Fields
- `warmstart_solver`: Warmstart solver (e.g., NelderMead()). If nothing, warmstart is skipped.
- `warmstart_opt::Union{Nothing, Optim.Options}`: Warmstart Optim.Options
- `main_solver`: Main solver (e.g., Newton())
- `main_opt::Optim.Options`: Main Optim.Options
"""
struct sfmodel_MCI_optim
    warmstart_solver::Any
    warmstart_opt::Union{Nothing, Optim.Options}
    main_solver::Any
    main_opt::Optim.Options
end

"""
    sfmodel_MCI_opt(; warmstart_solver=nothing, warmstart_opt=nothing,
                      main_solver, main_opt)

Construct optimization options for `sfmodel_MCI_fit()`.

# Arguments
- `warmstart_solver=nothing`: Warmstart optimizer, e.g., `NelderMead()`, `BFGS()`. Optional.
- `warmstart_opt=nothing`: Warmstart options as a NamedTuple, e.g., `(iterations = 400, g_abstol = 1e-5)`. Optional.
- `main_solver`: Main optimizer, e.g., `Newton()`, `BFGS()`. Required.
- `main_opt`: Main options as a NamedTuple, e.g., `(iterations = 2000, g_abstol = 1e-8)`. Required.

If `warmstart_solver` is not provided, the warmstart stage will be skipped.

# Example
```julia
# With warmstart
myopt = sfmodel_MCI_opt(
    warmstart_solver = NelderMead(),
    warmstart_opt = (iterations = 400, g_abstol = 1e-5),
    main_solver = Newton(),
    main_opt = (iterations = 2000, g_abstol = 1e-8)
)

# Without warmstart (skip directly to main optimization)
myopt = sfmodel_MCI_opt(
    main_solver = Newton(),
    main_opt = (iterations = 2000, g_abstol = 1e-8)
)
```
"""
function sfmodel_MCI_opt(;
    warmstart_solver = nothing,
    warmstart_opt = nothing,
    main_solver,
    main_opt
)
    # Validate that main_opt is a NamedTuple (common mistake: missing trailing comma)
    if !(main_opt isa NamedTuple)
        error("Invalid `main_opt`: expected a NamedTuple, got $(typeof(main_opt)). " *
              "Hint: For single-element options, use a trailing comma: " *
              "`main_opt = (iterations = 200,)` not `main_opt = (iterations = 200)`.")
    end

    # Validate warmstart_opt if provided
    if warmstart_opt !== nothing && !(warmstart_opt isa NamedTuple)
        error("Invalid `warmstart_opt`: expected a NamedTuple, got $(typeof(warmstart_opt)). " *
              "Hint: For single-element options, use a trailing comma: " *
              "`warmstart_opt = (iterations = 200,)` not `warmstart_opt = (iterations = 200)`.")
    end

    # Convert NamedTuple to Optim.Options for main_opt
    m_opt = Optim.Options(; main_opt...)

    # Convert NamedTuple to Optim.Options for warmstart_opt if provided
    if warmstart_solver !== nothing
        if warmstart_opt === nothing
            # Default warmstart options if solver provided but no options
            ws_opt = Optim.Options(iterations = 100, g_abstol = 1e-3)
        else
            ws_opt = Optim.Options(; warmstart_opt...)
        end
    else
        ws_opt = nothing
    end

    return sfmodel_MCI_optim(warmstart_solver, ws_opt, main_solver, m_opt)
end

# ============================================================================
# End of Optimization Options Specification
# ============================================================================

"""
    _default_eq_names(model::MCIModel, hetero::Vector{Symbol})

Generate default equation names based on model type and heteroscedasticity settings.
"""
function _default_eq_names(model::MCIModel, hetero::Vector{Symbol}; scaling::Bool=false)
    eqnames = ["frontier"]

    # Add scaling equation name (after frontier, before inefficiency)
    if scaling
        push!(eqnames, "scaling")
    end

    # Add inefficiency equation names based on model type
    ineff = model.ineff
    if ineff isa TruncatedNormal_MCI
        push!(eqnames, :mu in hetero ? "μ" : "μ")
        push!(eqnames, :sigma_sq in hetero ? "ln_σᵤ²" : "ln_σᵤ²")
    elseif ineff isa Exponential_MCI
        push!(eqnames, :lambda in hetero ? "ln_λ" : "ln_λ")
    elseif ineff isa HalfNormal_MCI
        push!(eqnames, :sigma_sq in hetero ? "ln_σᵤ²" : "ln_σᵤ²")
    elseif ineff isa Weibull_MCI
        push!(eqnames, :lambda in hetero ? "ln_λ" : "ln_λ")
        push!(eqnames, :k in hetero ? "ln_k" : "ln_k")
    elseif ineff isa Lognormal_MCI
        push!(eqnames, :mu in hetero ? "μ" : "μ")
        push!(eqnames, :sigma_sq in hetero ? "ln_σ2" : "ln_σ2")
    elseif ineff isa Lomax_MCI
        push!(eqnames, :lambda in hetero ? "ln_λ" : "ln_λ")
        push!(eqnames, :alpha in hetero ? "ln_α" : "ln_α")
    elseif ineff isa Rayleigh_MCI
        push!(eqnames, :sigma_sq in hetero ? "ln_σ2" : "ln_σ2")
    elseif ineff isa Gamma_MCI
        push!(eqnames, :k in hetero ? "ln_k" : "ln_k")
        push!(eqnames, :theta in hetero ? "ln_θ" : "ln_θ")
    end

    # Add noise equation names
    noise = model.noise
    if noise isa NormalNoise_MCI
        push!(eqnames, "ln_σᵥ²")
    elseif noise isa StudentTNoise_MCI
        push!(eqnames, "ln_σᵥ²")
        push!(eqnames, "ln_(ν-2)")
    elseif noise isa LaplaceNoise_MCI
        push!(eqnames, "ln_b")
    end

    # Add copula equation names
    if copula_plen(model.copula) > 0
        push!(eqnames, "theta_rho")
    end

    return eqnames
end

"""
    _default_eq_ind(model::MCIModel, K::Int, L::Int, hetero::Vector{Symbol})

Generate default equation indices based on model type and heteroscedasticity settings.
"""
function _default_eq_ind(model::MCIModel, K::Int, L::Int, hetero::Vector{Symbol}; scaling::Bool=false, L_scaling::Int=0)
    eq_indices = [1]  # frontier starts at 1
    idx = K + 1       # after beta

    # Add scaling equation index (after frontier, before inefficiency)
    if scaling
        push!(eq_indices, idx)
        idx += L_scaling
    end

    # Add inefficiency equation indices based on model type
    ineff = model.ineff
    if ineff isa TruncatedNormal_MCI
        push!(eq_indices, idx)  # μ
        n_mu = :mu in hetero ? L : 1
        idx += n_mu
        push!(eq_indices, idx)  # ln_σᵤ²
        n_sigma = :sigma_sq in hetero ? L : 1
        idx += n_sigma
    elseif ineff isa Exponential_MCI
        push!(eq_indices, idx)  # ln_λ
        n_lambda = :lambda in hetero ? L : 1
        idx += n_lambda
    elseif ineff isa HalfNormal_MCI
        push!(eq_indices, idx)  # ln_σᵤ²
        n_sigma = :sigma_sq in hetero ? L : 1
        idx += n_sigma
    elseif ineff isa Weibull_MCI
        push!(eq_indices, idx)  # ln_λ
        n_lambda = :lambda in hetero ? L : 1
        idx += n_lambda
        push!(eq_indices, idx)  # ln_k
        n_k = :k in hetero ? L : 1
        idx += n_k
    elseif ineff isa Lognormal_MCI
        push!(eq_indices, idx)  # μ
        n_mu = :mu in hetero ? L : 1
        idx += n_mu
        push!(eq_indices, idx)  # ln_σ2
        n_sigma = :sigma_sq in hetero ? L : 1
        idx += n_sigma
    elseif ineff isa Lomax_MCI
        push!(eq_indices, idx)  # ln_λ
        n_lambda = :lambda in hetero ? L : 1
        idx += n_lambda
        push!(eq_indices, idx)  # ln_α
        n_alpha = :alpha in hetero ? L : 1
        idx += n_alpha
    elseif ineff isa Rayleigh_MCI
        push!(eq_indices, idx)  # ln_σ2
        n_sigma = :sigma_sq in hetero ? L : 1
        idx += n_sigma
    elseif ineff isa Gamma_MCI
        push!(eq_indices, idx)  # ln_k
        n_k = :k in hetero ? L : 1
        idx += n_k
        push!(eq_indices, idx)  # ln_θ
        n_theta = :theta in hetero ? L : 1
        idx += n_theta
    end

    # Add noise equation indices
    noise = model.noise
    push!(eq_indices, idx)  # ln_σᵥ² or ln_b
    idx += 1
    if noise isa StudentTNoise_MCI
        push!(eq_indices, idx)  # ln_(ν-2)
        idx += 1
    end

    # Add copula equation indices
    if copula_plen(model.copula) > 0
        push!(eq_indices, idx)  # theta_rho
        idx += copula_plen(model.copula)
    end

    return eq_indices
end

# ============================================================================
# Section 2b: DSL Macros for DataFrame-based Specification
# ============================================================================

"""
Marker types for DSL-style model specification.
These types hold the parsed macro arguments until they're processed by sfmodel_MCI_spec.
"""
struct UseDataSpec
    df::DataFrame
end

struct DepvarSpec
    name::Symbol
end

struct FrontierSpec
    names::Vector{Symbol}
end

struct ZvarSpec
    names::Vector{Symbol}
end

"""
    @useData(df)

Specify the DataFrame to extract variables from.

# Example
```julia
@useData(mydata)  # mydata must be a DataFrame
```
"""
macro useData(df)
    :(UseDataSpec($(esc(df))))
end

"""
    @depvar(varname)

Specify the dependent variable by column name.

# Example
```julia
@depvar(yvar)  # extracts df.yvar as the dependent variable
```
"""
macro depvar(var)
    :(DepvarSpec($(QuoteNode(var))))
end

"""
    @frontier(var1, var2, ...)

Specify frontier variables by column names.

# Example
```julia
@frontier(_cons, Lland, PIland, Llabor)  # extracts these columns for the frontier
```
"""
macro frontier(vars...)
    names = [QuoteNode(v) for v in vars]
    :(FrontierSpec(Symbol[$(names...)]))
end

"""
    @zvar(var1, var2, ...)

Specify Z variables (for heteroscedasticity) by column names.

# Example
```julia
@zvar(_cons, age, school, yr)  # extracts these columns for zvar
```
"""
macro zvar(vars...)
    names = [QuoteNode(v) for v in vars]
    :(ZvarSpec(Symbol[$(names...)]))
end

"""
    _gen_names(model::MCIModel, K::Int, L::Int, hetero::Vector{Symbol},
                         idx::NamedTuple, varnames, eqnames, eq_indices)

Auto-generate variable names, equation names, and equation indices if not provided.
"""
function _gen_names(model::MCIModel, K::Int, L::Int, hetero::Vector{Symbol},
                               param_idx::NamedTuple, varnames, eqnames, eq_indices;
                               scaling::Bool=false, L_scaling::Int=0)
    # If all provided, return as-is
    if !isnothing(varnames) && !isnothing(eqnames) && !isnothing(eq_indices)
        return varnames, eqnames, eq_indices
    end

    # Generate default variable names
    n_params = plen(model, K, L, hetero; L_scaling=L_scaling)
    default_varnames = ["x$i" for i in 1:n_params]

    # Generate equation names and indices based on model type
    default_eqnames = _default_eq_names(model, hetero; scaling=scaling)
    default_eq_indices = _default_eq_ind(model, K, L, hetero; scaling=scaling, L_scaling=L_scaling)

    return (
        isnothing(varnames) ? default_varnames : varnames,
        isnothing(eqnames) ? default_eqnames : eqnames,
        isnothing(eq_indices) ? default_eq_indices : eq_indices
    )
end

# ============================================================================
# New API: sfmodel_spec() and sfmodel_method()
# ============================================================================

"""
    sfmodel_spec(; depvar, frontier, zvar=nothing, noise, ineff, hetero=Symbol[],
                 varnames=nothing, eqnames=nothing, eq_indices=nothing, type=:prod)

Construct a model specification containing data, distributional assumptions, and variable names.
Numerical method settings (draws, GPU, chunks, transformation) are specified separately
via `sfmodel_method()`.

# Arguments
- `depvar`: Response vector (N observations). Accepts Vector, N×1 Matrix, or [Vector].
- `frontier`: Frontier design matrix (N × K). Accepts Matrix, Vector, or [v1, v2, ...].
- `zvar=nothing`: Z variable matrix (N × L). Optional: auto-generates `ones(N)` if not provided.
- `noise::Symbol`: Noise distribution (`:Normal`, `:StudentT`, `:Laplace`)
- `ineff::Symbol`: Inefficiency distribution (`:HalfNormal`, `:TruncatedNormal`, `:Exponential`, etc.)
- `hetero::Vector{Symbol}=Symbol[]`: Heteroscedasticity options
- `varnames=nothing`: Variable names (auto-generated as x1, x2, ... if not provided)
- `eqnames=nothing`: Equation names (auto-generated based on model if not provided)
- `eq_indices=nothing`: Equation indices (auto-generated based on model if not provided)
- `type::Symbol=:prod`: Frontier type (`:prod`, `:production`, or `:cost`)

# Returns
- `SFModelSpec{T}`: Model specification struct

# Example
```julia
spec = sfmodel_spec(depvar=y, frontier=X, zvar=Z,
    noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu, :sigma_sq], type=:prod)
```
"""
function sfmodel_spec(; depvar, frontier, zvar=nothing,
                noise::Symbol, ineff::Symbol,
                copula::Symbol=:None,
                hetero::Union{Vector{Symbol}, Symbol}=Symbol[],
                varnames::Union{Nothing, Vector{String}}=nothing,
                eqnames::Union{Nothing, Vector{String}}=nothing,
                eq_indices::Union{Nothing, Vector{Int}}=nothing,
                type::Symbol=:prod)

    # Normalize inputs (handle common user errors like [yvar] or [v1, v2, v3])
    depvar_norm = _to_vector(depvar)
    frontier_norm = _to_matrix(frontier)

    # Infer type T from normalized depvar
    T = eltype(depvar_norm)

    # --- Handle scaling property model ---
    _scaling = false
    _scaling_zvar = nothing
    _L_scaling = 0
    hetero_vec = hetero isa Symbol ? Symbol[] : hetero

    if hetero === :scaling
        # Scaling requires zvar (the Z matrix for h(z) = exp(z'δ))
        isnothing(zvar) && error("Scaling property model (hetero=:scaling) requires `zvar` to be provided.")

        zvar_norm = _to_matrix(zvar)
        zvar_norm = T.(zvar_norm)

        # Validate: no constant columns in Z for scaling
        N_tmp = length(depvar_norm)
        for col in 1:size(zvar_norm, 2)
            zcol = @view zvar_norm[:, col]
            if all(x -> x == zcol[1], zcol)
                error("Scaling Z variable column $col is constant (all values = $(zcol[1])). " *
                      "The scaling function h(z)=exp(z'δ) should NOT include a constant column.")
            end
        end

        _scaling = true
        _scaling_zvar = zvar_norm
        _L_scaling = size(zvar_norm, 2)

        # For the main model, use ones(N) as zvar (homoscedastic ineff params)
        zvar_norm = ones(T, N_tmp, 1)
        hetero_vec = Symbol[]
    else
        # Auto-generate zvar as ones(N) for homoscedastic models when not provided
        if isnothing(zvar)
            zvar = ones(T, length(depvar_norm))
        end
        zvar_norm = _to_matrix(zvar)
        zvar_norm = T.(zvar_norm)
    end

    # Convert frontier to match T if needed
    frontier_norm = T.(frontier_norm)

    # No GPU conversion here — deferred to _assemble_MCI_spec via sfmodel_MCI_fit

    # Validate and compute sign for frontier type
    frontier_sign = if type in (:prod, :production)
        1
    elseif type == :cost
        -1
    else
        error("Invalid `type` in sfmodel_spec(): $type. Use :prod, :production, or :cost.")
    end

    N, K, L = length(depvar_norm), size(frontier_norm, 2), size(zvar_norm, 2)

    # Build model and validate
    model = _build_model(noise, ineff; copula=copula)
    _validate_hetero(model.ineff, hetero_vec)

    # Validate varnames length if provided
    if !isnothing(varnames)
        expected_len = plen(model, K, L, hetero_vec; L_scaling=_L_scaling)
        if length(varnames) != expected_len
            error("Length of `varnames` ($(length(varnames))) does not match " *
                  "the number of parameters ($expected_len). " *
                  "Expected $expected_len names for: frontier ($K) + inefficiency + noise parameters.")
        end
    end

    # Compute parameter indices
    idx = _param_ind(model, K, L, hetero_vec; L_scaling=_L_scaling)

    # Auto-generate varnames/eqnames/eq_indices if not provided
    varnames_vec, eqnames_vec, eq_indices_vec = _gen_names(
        model, K, L, hetero_vec, idx, varnames, eqnames, eq_indices;
        scaling=_scaling, L_scaling=_L_scaling
    )

    return SFModelSpec{T}(depvar_norm, frontier_norm, zvar_norm, noise, ineff, copula, hetero_vec,
                     varnames_vec, eqnames_vec, eq_indices_vec,
                     N, K, L, model, idx, frontier_sign,
                     _scaling, _scaling_zvar, _L_scaling)
end

"""
    sfmodel_method(; method=:MCI, transformation=nothing, draws=nothing,
                   n_draws=1024, multiRand=true, GPU=false, chunks=10, distinct_Halton_length=2^15-1)

Specify numerical method settings for stochastic frontier estimation.

# Arguments
- `method::Symbol=:MCI`: Estimation method (currently only `:MCI` is supported)
- `transformation=nothing`: Transformation rule (`:expo_rule`, `:logistic_1_rule`,
  `:logistic_2_rule`), or `nothing` (use the default for the inefficiency distribution)
- `draws=nothing`: User-provided draws, or `nothing` to auto-generate Halton sequences
- `n_draws::Int=1024`: Number of Halton draws (used if `draws` not provided)
- `multiRand::Bool=true`: Per-observation draws (`true`, N×D) or shared draws (`false`, 1×D)
- `GPU::Bool=false`: Whether to use GPU acceleration (requires `using CUDA`)
- `chunks::Int=10`: Number of chunks for GPU memory management
- `distinct_Halton_length::Int=2^15-1`: Maximum Halton sequence length for multiRand mode (default 32767)

# Returns
- `SFMethodSpec`: Method specification struct

# Example
```julia
meth = sfmodel_method(method=:MCI, n_draws=2^12-1, GPU=true, chunks=10)

# Larger distinct_Halton_length pool for multiRand mode
meth = sfmodel_method(method=:MCI, n_draws=50000, distinct_Halton_length=2^16-1)
```
"""
function sfmodel_method(;
                method::Symbol = :MCI,
                transformation::Union{Symbol,Nothing}=nothing,
                draws::Union{Nothing, AbstractVecOrMat{<:AbstractFloat}}=nothing,
                n_draws::Int=1024,
                multiRand::Bool=true,
                GPU::Bool=false,
                chunks::Int=10,
                distinct_Halton_length::Int=2^15-1)

    # Validate method
    method == :MCI || error("Currently only method=:MCI is supported. Got: :$method")

    # Validate transformation if provided
    if !isnothing(transformation)
        transformation in (:expo_rule, :logistic_1_rule, :logistic_2_rule) ||
            error("Invalid `transformation`: $transformation. " *
                  "Use :expo_rule, :logistic_1_rule, or :logistic_2_rule.")
    end

    return SFMethodSpec(method, transformation, draws, n_draws, multiRand, GPU, chunks, distinct_Halton_length)
end

"""
    sfmodel_spec(data_spec, depvar_spec, frontier_spec, zvar_spec; ...)

DSL-style model specification using macros. Automatically extracts data from DataFrame
and generates variable names from column names.

# Arguments
- `data_spec::UseDataSpec`: DataFrame wrapped by `@useData(df)`
- `depvar_spec::DepvarSpec`: Dependent variable name from `@depvar(varname)`
- `frontier_spec::FrontierSpec`: Frontier variable names from `@frontier(var1, var2, ...)`
- `zvar_spec::ZvarSpec`: Z variable names from `@zvar(var1, var2, ...)`
- Other keyword arguments: same as the matrix-input version of `sfmodel_spec()`

# Example
```julia
df._cons = ones(nrow(df))
spec = sfmodel_spec(
    @useData(df), @depvar(yvar),
    @frontier(_cons, Lland, PIland, Llabor, Lbull, Lcost, yr),
    @zvar(_cons, age, school, yr),
    noise = :Normal, ineff = :TruncatedNormal,
    hetero = [:mu, :sigma_sq]
)
```
"""
function sfmodel_spec(data_spec::UseDataSpec, depvar_spec::DepvarSpec,
                          frontier_spec::FrontierSpec, zvar_spec::ZvarSpec;
                          noise::Symbol, ineff::Symbol,
                          copula::Symbol=:None,
                          hetero::Union{Vector{Symbol}, Symbol}=Symbol[],
                          eqnames::Union{Nothing, Vector{String}}=nothing,
                          eq_indices::Union{Nothing, Vector{Int}}=nothing,
                          type::Symbol=:prod)

    df = data_spec.df

    # Extract depvar from DataFrame
    depvar = Vector{Float64}(df[!, depvar_spec.name])

    # Build frontier matrix - all columns must exist in DataFrame
    frontier_names = [String(name) for name in frontier_spec.names]
    frontier = hcat([Vector{Float64}(df[!, name]) for name in frontier_spec.names]...)

    # Build zvar matrix - all columns must exist in DataFrame
    zvar_names = [String(name) for name in zvar_spec.names]
    zvar = hcat([Vector{Float64}(df[!, name]) for name in zvar_spec.names]...)

    # Handle scaling property model
    if hetero === :scaling
        # For DSL version: Z columns become scaling varnames (δ eq)
        # All ineff params become scalar (no hetero)
        K = length(frontier_spec.names)
        L_scaling = length(zvar_spec.names)
        model = _build_model(noise, ineff; copula=copula)
        n_params = plen(model, K, 1, Symbol[]; L_scaling=L_scaling)

        # Build varnames: frontier names + scaling Z names + scalar ineff + noise + copula
        varnames = Vector{String}(undef, n_params)
        varnames[1:K] = frontier_names
        vi = K + 1

        # Scaling equation (δ coefficients)
        for name in zvar_names
            varnames[vi] = name
            vi += 1
        end

        # Inefficiency parameters (all scalar since scaling uses homoscedastic ineff)
        ineff_type = model.ineff
        if ineff_type isa TruncatedNormal_MCI
            varnames[vi] = "μ";  vi += 1
            varnames[vi] = "ln_σᵤ²";  vi += 1
        elseif ineff_type isa HalfNormal_MCI
            varnames[vi] = "ln_σᵤ²";  vi += 1
        elseif ineff_type isa Exponential_MCI
            varnames[vi] = "ln_λ";  vi += 1
        elseif ineff_type isa Gamma_MCI
            varnames[vi] = "ln_k";  vi += 1
            varnames[vi] = "ln_θ";  vi += 1
        elseif ineff_type isa Lognormal_MCI
            varnames[vi] = "μ";  vi += 1
            varnames[vi] = "ln_σ2";  vi += 1
        elseif ineff_type isa Weibull_MCI
            varnames[vi] = "ln_λ";  vi += 1
            varnames[vi] = "ln_k";  vi += 1
        elseif ineff_type isa Lomax_MCI
            varnames[vi] = "ln_λ";  vi += 1
            varnames[vi] = "ln_α";  vi += 1
        elseif ineff_type isa Rayleigh_MCI
            varnames[vi] = "ln_σ2";  vi += 1
        end

        # Noise parameters
        if noise == :Laplace
            varnames[vi] = "ln_b";  vi += 1
        else
            varnames[vi] = "ln_σᵥ²";  vi += 1
        end
        if noise == :StudentT
            varnames[vi] = "ln_(ν-2)";  vi += 1
        end

        # Copula parameter
        if copula != :None
            varnames[vi] = "θ_ρ";  vi += 1
        end

        # Pass :scaling to keyword sfmodel_spec (which handles the rest)
        return sfmodel_spec(; depvar=depvar, frontier=frontier, zvar=zvar,
                                 noise=noise, ineff=ineff, copula=copula, hetero=:scaling,
                                 varnames=varnames,
                                 eqnames=eqnames, eq_indices=eq_indices,
                                 type=type)
    end

    # --- Standard (non-scaling) path ---
    # Compute total number of parameters to build full varnames
    K = length(frontier_spec.names)
    L = length(zvar_spec.names)

    # Build model to determine parameter count
    model = _build_model(noise, ineff; copula=copula)
    _validate_hetero(model.ineff, hetero)
    n_params = plen(model, K, L, hetero)

    # Build varnames based on equation structure
    # Structure follows _default_eq_ind order: frontier, ineff, noise
    varnames = Vector{String}(undef, n_params)

    # Frontier equation (K params)
    varnames[1:K] = frontier_names
    idx = K + 1

    # Inefficiency distribution parameters (before noise, to match _default_eq_ind)
    ineff_type = model.ineff
    if ineff_type isa TruncatedNormal_MCI
        # μ equation
        if :mu in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "μ"
            idx += 1
        end
        # ln_σᵤ² equation
        if :sigma_sq in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_σᵤ²"
            idx += 1
        end
    elseif ineff_type isa HalfNormal_MCI
        # ln_σᵤ² equation only
        if :sigma_sq in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_σᵤ²"
            idx += 1
        end
    elseif ineff_type isa Exponential_MCI
        # ln_λ equation
        if :lambda in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_λ"
            idx += 1
        end
    elseif ineff_type isa Gamma_MCI
        # ln_k (shape) and ln_θ (scale) equations
        if :k in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_k"
            idx += 1
        end
        if :theta in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_θ"
            idx += 1
        end
    elseif ineff_type isa Lognormal_MCI
        # μ equation
        if :mu in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "μ"
            idx += 1
        end
        # ln_σ2 equation
        if :sigma_sq in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_σ2"
            idx += 1
        end
    elseif ineff_type isa Weibull_MCI
        # ln_λ equation
        if :lambda in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_λ"
            idx += 1
        end
        # ln_k equation
        if :k in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_k"
            idx += 1
        end
    elseif ineff_type isa Lomax_MCI
        # ln_λ equation
        if :lambda in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_λ"
            idx += 1
        end
        # ln_α equation
        if :alpha in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_α"
            idx += 1
        end
    elseif ineff_type isa Rayleigh_MCI
        # ln_σ2 equation
        if :sigma_sq in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_σ2"
            idx += 1
        end
    end

    # Noise equation (at the end, to match _default_eq_ind)
    if noise == :Laplace
        varnames[idx] = "ln_b"
    else
        varnames[idx] = "ln_σᵥ²"
    end
    idx += 1

    # StudentT has extra param
    if noise == :StudentT
        varnames[idx] = "ln_(ν-2)"
        idx += 1
    end

    # Copula parameter
    if copula != :None
        varnames[idx] = "θ_ρ"
        idx += 1
    end

    # Call the matrix-input version with auto-generated varnames
    return sfmodel_spec(; depvar=depvar, frontier=frontier, zvar=zvar,
                             noise=noise, ineff=ineff, copula=copula, hetero=hetero,
                             varnames=varnames,
                             eqnames=eqnames, eq_indices=eq_indices,
                             type=type)
end

"""
    sfmodel_spec(data_spec, depvar_spec, frontier_spec; ...)

DSL-style model specification **without** `@zvar`. Automatically creates a column of ones
as the Z matrix (homoscedastic model). Use the 4-argument form with `@zvar(...)` if you
need heteroscedastic parameters.

# Example
```julia
spec = sfmodel_spec(
    @useData(df),
    @depvar(yvar),
    @frontier(_cons, x1, x2),
    noise = :Normal,
    ineff = :HalfNormal
)
```
"""
function sfmodel_spec(data_spec::UseDataSpec, depvar_spec::DepvarSpec,
                          frontier_spec::FrontierSpec;
                          noise::Symbol, ineff::Symbol,
                          copula::Symbol=:None,
                          hetero::Union{Vector{Symbol}, Symbol}=Symbol[],
                          eqnames::Union{Nothing, Vector{String}}=nothing,
                          eq_indices::Union{Nothing, Vector{Int}}=nothing,
                          type::Symbol=:prod)

    if hetero === :scaling
        error("Scaling property model (hetero=:scaling) requires @zvar(...) to specify the scaling variables.")
    end

    df = data_spec.df

    # Extract depvar from DataFrame
    depvar = Vector{Float64}(df[!, depvar_spec.name])

    # Build frontier matrix
    frontier_names = [String(name) for name in frontier_spec.names]
    frontier = hcat([Vector{Float64}(df[!, name]) for name in frontier_spec.names]...)

    # No zvar provided → call keyword version with zvar=nothing (auto-generates ones(N))
    # Pass frontier varnames so the output table has meaningful column names
    K = length(frontier_spec.names)
    model = _build_model(noise, ineff; copula=copula)
    _validate_hetero(model.ineff, hetero)
    n_params = plen(model, K, 1, hetero)  # L=1 for homoscedastic

    # Build varnames: frontier names + scalar param names for non-frontier equations
    varnames = Vector{String}(undef, n_params)
    varnames[1:K] = frontier_names
    vi = K + 1

    # Inefficiency parameters (scalar for each since no hetero zvar)
    ineff_type = model.ineff
    if ineff_type isa TruncatedNormal_MCI
        varnames[vi] = :mu in hetero ? "_cons" : "μ";  vi += 1
        varnames[vi] = :sigma_sq in hetero ? "_cons" : "ln_σᵤ²";  vi += 1
    elseif ineff_type isa HalfNormal_MCI
        varnames[vi] = :sigma_sq in hetero ? "_cons" : "ln_σᵤ²";  vi += 1
    elseif ineff_type isa Exponential_MCI
        varnames[vi] = :lambda in hetero ? "_cons" : "ln_λ";  vi += 1
    elseif ineff_type isa Gamma_MCI
        varnames[vi] = :k in hetero ? "_cons" : "ln_k";  vi += 1
        varnames[vi] = :theta in hetero ? "_cons" : "ln_θ";  vi += 1
    elseif ineff_type isa Lognormal_MCI
        varnames[vi] = :mu in hetero ? "_cons" : "μ";  vi += 1
        varnames[vi] = :sigma_sq in hetero ? "_cons" : "ln_σ2";  vi += 1
    elseif ineff_type isa Weibull_MCI
        varnames[vi] = :lambda in hetero ? "_cons" : "ln_λ";  vi += 1
        varnames[vi] = :k in hetero ? "_cons" : "ln_k";  vi += 1
    elseif ineff_type isa Lomax_MCI
        varnames[vi] = :lambda in hetero ? "_cons" : "ln_λ";  vi += 1
        varnames[vi] = :alpha in hetero ? "_cons" : "ln_α";  vi += 1
    elseif ineff_type isa Rayleigh_MCI
        varnames[vi] = :sigma_sq in hetero ? "_cons" : "ln_σ2";  vi += 1
    end

    # Noise parameters
    if noise == :Laplace
        varnames[vi] = "ln_b";  vi += 1
    else
        varnames[vi] = "ln_σᵥ²";  vi += 1
    end
    if noise == :StudentT
        varnames[vi] = "ln_(ν-2)";  vi += 1
    end

    # Copula parameter
    if copula != :None
        varnames[vi] = "θ_ρ";  vi += 1
    end

    return sfmodel_spec(; depvar=depvar, frontier=frontier, zvar=nothing,
                             noise=noise, ineff=ineff, copula=copula, hetero=hetero,
                             varnames=varnames,
                             eqnames=eqnames, eq_indices=eq_indices,
                             type=type)
end

# NOTE: The legacy sfmodel_MCI_spec() constructor functions have been removed.
# Use sfmodel_spec() + sfmodel_method() instead.
# The sfmodel_MCI_spec struct is still used internally by MCI_nll, jlms_bc_indices,
# print_table, get_marg, and marginal_effects.

#= REMOVED: Legacy sfmodel_MCI_spec() constructor functions

"""
    sfmodel_MCI_spec(; depvar, frontier, zvar=nothing, noise, ineff, hetero=Symbol[], draws=nothing,
           varnames=nothing, eqnames=nothing, eq_indices=nothing,
           n_draws=1023, GPU=false)

Construct a model specification with automatic defaults.

# Arguments
- `depvar`: Response vector (N observations). Accepts Vector, N×1 Matrix (auto-flattened via `vec()`),
   or [Vector] (a vector wrapped in a 1-element array, auto-unwrapped via `only()`).
- `frontier`: Frontier design matrix (N x K). Accepts Matrix, Vector (auto-reshaped to N×1 via `reshape(x, :, 1)`),
   or [v1, v2, ...] (vector of vectors, auto-converted to matrix via `reduce(hcat, ...)`).
- `zvar=nothing`: Heteroscedasticity design matrix (N x L). Accepts Matrix, Vector (auto-reshaped to N×1 via `reshape(x, :, 1)`),
   or [v1, v2, ...] (vector of vectors, auto-converted to matrix via `reduce(hcat, ...)`).
   Optional: if not provided, auto-generates `ones(N)` for homoscedastic models.
- `noise::Symbol`: Noise distribution (:Normal, :StudentT, :Laplace)
- `ineff::Symbol`: Inefficiency distribution
- `hetero::Vector{Symbol}=Symbol[]`: Heteroscedasticity options
- `draws=nothing`: Halton sequence (auto-generated if not provided)
- `varnames=nothing`: Variable names (auto-generated as x1, x2, ... if not provided)
- `eqnames=nothing`: Equation names (auto-generated based on model if not provided)
- `eq_indices=nothing`: Equation indices (auto-generated based on model if not provided)
- `n_draws::Int=1023`: Number of Halton draws (used if draws not provided)
- `GPU::Bool=false`: If true, convert data to GPU arrays for GPU computation. Requires `using CUDA` before calling.

# Example
```julia
# Homoscedastic model (zvar and hetero auto-default)
spec = sfmodel_MCI_spec(depvar=y, frontier=X, noise=:Normal, ineff=:HalfNormal)

# Heteroscedastic model with zvar
spec = sfmodel_MCI_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])

# GPU computation (requires `using CUDA` first)
spec = sfmodel_MCI_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu], GPU=true)

# Full specification with custom names
spec = sfmodel_MCI_spec(
    depvar = y, frontier = X, zvar = Z,
    noise = :Normal, ineff = :TruncatedNormal, hetero = [:mu],
    varnames = ["_cons", "x1", "x2", "_cons", "_cons"],
    eqnames = ["frontier", "μ", "ln_σᵤ²", "ln_σᵥ²"],
    eq_indices = [1, 4, 5, 6]
)
```
"""
function sfmodel_MCI_spec(; depvar, frontier, zvar=nothing,
                noise::Symbol, ineff::Symbol,
                hetero::Vector{Symbol}=Symbol[],
                transformation::Union{Symbol,Nothing}=nothing,
                draws::Union{Nothing, AbstractVecOrMat{<:AbstractFloat}}=nothing,
                varnames::Union{Nothing, Vector{String}}=nothing,
                eqnames::Union{Nothing, Vector{String}}=nothing,
                eq_indices::Union{Nothing, Vector{Int}}=nothing,
                n_draws::Int=1023,
                multiRand::Bool=true,
                GPU::Bool=false,
                type::Symbol=:prod,
                chunks::Int=4)

    # Normalize inputs (handle common user errors like [yvar] or [v1, v2, v3])
    depvar_norm = _to_vector(depvar)
    frontier_norm = _to_matrix(frontier)

    # Auto-generate zvar as ones(N) for homoscedastic models when not provided
    if isnothing(zvar)
        zvar = ones(eltype(depvar_norm), length(depvar_norm))
    end
    zvar_norm = _to_matrix(zvar)

    # Infer type T from normalized depvar
    T = eltype(depvar_norm)

    # Convert frontier and zvar to match T if needed
    # Note: T.(x) broadcasting preserves array type (CuArray stays CuArray)
    frontier_norm = T.(frontier_norm)
    zvar_norm = T.(zvar_norm)

    # Convert to GPU arrays if requested
    if GPU
        # Note: Gamma now uses T-approach which is GPU-compatible
        # Check if CUDA.jl has been loaded
        if !isdefined(Main, :CUDA)
            error("GPU=true requires CUDA.jl to be loaded. Please run `using CUDA` before calling this function.")
        end
        depvar_norm = Main.CUDA.CuArray(depvar_norm)
        frontier_norm = Main.CUDA.CuArray(frontier_norm)
        zvar_norm = Main.CUDA.CuArray(zvar_norm)
    end

    # Validate and compute sign for frontier type
    frontier_sign = if type in (:prod, :production)
        1
    elseif type == :cost
        -1
    else
        error("Invalid `type` in sfmodel_MCI_spec(): $type. Use :prod, :production, or :cost.")
    end

    N, K, L = length(depvar_norm), size(frontier_norm, 2), size(zvar_norm, 2)

    # Build model and validate
    model = _build_model(noise, ineff)
    _validate_hetero(model.ineff, hetero)

    # Validate varnames length if provided
    if !isnothing(varnames)
        expected_len = plen(model, K, L, hetero)
        if length(varnames) != expected_len
            error("Length of `varnames` ($(length(varnames))) does not match " *
                  "the number of parameters ($expected_len). " *
                  "Expected $expected_len names for: frontier ($K) + inefficiency + noise parameters.")
        end
    end

    # Auto-generate draws if not provided
    # If draws not provided, generate Halton draws and convert to same device as depvar
    if isnothing(draws)
        if multiRand
            # Generate N x D wrapped Halton matrix (each observation gets different draws)
            halton_cpu = make_halton_wrap(N, n_draws; T=T)
            draws_2D = _to_device_array(depvar_norm, halton_cpu)
            draws_vec = vec(draws_2D)
        else
            # Generate 1 x D shared Halton draws (all observations share same draws)
            halton_cpu = make_halton_p(n_draws; T=T)
            draws_vec = _to_device_array(depvar_norm, halton_cpu)
            draws_2D = reshape(draws_vec, 1, length(draws_vec))
        end
    else
        # User-provided draws
        depvar_is_gpu = string(typeof(depvar_norm).name.wrapper) == "CuArray"
        draws_is_gpu = string(typeof(draws).name.wrapper) == "CuArray"
        if depvar_is_gpu != draws_is_gpu
            @warn "Type inconsistency: `depvar` is on $(depvar_is_gpu ? "GPU" : "CPU") but `draws` is on $(draws_is_gpu ? "GPU" : "CPU"). " *
                  "Consider letting the program auto-generate Halton sequences with the correct device type by specifying `n_draws` instead of providing `draws`."
        end

        if multiRand
            if ndims(draws) == 1
                # User passed 1D vector but multiRand=true - generate wrapped matrix
                @warn "multiRand=true but `draws` is a 1D vector. Generating wrapped N x D matrix from `n_draws` instead."
                halton_cpu = make_halton_wrap(N, n_draws; T=T)
                draws_2D = _to_device_array(depvar_norm, halton_cpu)
                draws_vec = vec(draws_2D)
            elseif size(draws, 1) == N
                draws_2D = T.(draws)
                draws_vec = vec(draws_2D)
            else
                error("When multiRand=true, `draws` must be an N x D matrix with N=$N rows, got size $(size(draws))")
            end
        else
            draws_vec = T.(vec(draws))
            draws_2D = reshape(draws_vec, 1, length(draws_vec))
        end
    end

    # Auto-generate constants
    constants = make_constants(model, T)

    # Compute parameter indices
    idx = _param_ind(model, K, L, hetero)

    # Auto-generate varnames/eqnames/eq_indices if not provided
    varnames_vec, eqnames_vec, eq_indices_vec = _gen_names(
        model, K, L, hetero, idx, varnames, eqnames, eq_indices
    )

    # Resolve transformation rule (use default if not specified)
    trans_rule = isnothing(transformation) ? default_transformation_rule(ineff) : transformation

    # Validate transformation
    trans_rule in (:expo_rule, :logistic_1_rule, :logistic_2_rule) ||
        error("Invalid `transformation`: $trans_rule. " *
              "Use :expo_rule, :logistic_1_rule, or :logistic_2_rule.")

    return sfmodel_MCI_spec{T}(depvar_norm, frontier_norm, zvar_norm, noise, ineff, hetero,
                     draws_vec, draws_2D, multiRand, constants,
                     varnames_vec, eqnames_vec, eq_indices_vec,
                     N, K, L, model, idx, frontier_sign, chunks, trans_rule)
end

"""
    sfmodel_MCI_spec(data_spec, depvar_spec, frontier_spec, zvar_spec; ...)

DSL-style model specification using macros. Automatically extracts data from DataFrame
and generates variable names from column names.

# Arguments
- `data_spec::UseDataSpec`: DataFrame wrapped by `@useData(df)`
- `depvar_spec::DepvarSpec`: Dependent variable name from `@depvar(varname)`
- `frontier_spec::FrontierSpec`: Frontier variable names from `@frontier(var1, var2, ...)`
- `zvar_spec::ZvarSpec`: Z variable names from `@zvar(var1, var2, ...)`
- Other keyword arguments: same as the matrix-input version

# Example
```julia
# Prepare DataFrame with a constant column
df._cons = ones(nrow(df))

# DSL-style specification
spec = sfmodel_MCI_spec(
    @useData(df),
    @depvar(yvar),
    @frontier(_cons, Lland, PIland, Llabor, Lbull, Lcost, yr),
    @zvar(_cons, age, school, yr),
    noise = :Normal,
    ineff = :TruncatedNormal,
    hetero = [:mu, :sigma_sq],
    n_draws = 2^12 - 1,
    GPU = true
)
# Variable names auto-extracted: ["_cons", "Lland", "PIland", ..., "_cons", "age", "school", "yr"]
```
"""
function sfmodel_MCI_spec(data_spec::UseDataSpec, depvar_spec::DepvarSpec,
                          frontier_spec::FrontierSpec, zvar_spec::ZvarSpec;
                          noise::Symbol, ineff::Symbol,
                          hetero::Vector{Symbol}=Symbol[],
                          transformation::Union{Symbol,Nothing}=nothing,
                          draws::Union{Nothing, AbstractVecOrMat{<:AbstractFloat}}=nothing,
                          eqnames::Union{Nothing, Vector{String}}=nothing,
                          eq_indices::Union{Nothing, Vector{Int}}=nothing,
                          n_draws::Int=4095,
                          multiRand::Bool=true,
                          GPU::Bool=false,
                          type::Symbol=:prod,
                          chunks::Int=4)

    df = data_spec.df

    # Extract depvar from DataFrame
    depvar = Vector{Float64}(df[!, depvar_spec.name])

    # Build frontier matrix - all columns must exist in DataFrame
    frontier_names = [String(name) for name in frontier_spec.names]
    frontier = hcat([Vector{Float64}(df[!, name]) for name in frontier_spec.names]...)

    # Build zvar matrix - all columns must exist in DataFrame
    zvar_names = [String(name) for name in zvar_spec.names]
    zvar = hcat([Vector{Float64}(df[!, name]) for name in zvar_spec.names]...)

    # Compute total number of parameters to build full varnames
    # varnames order: frontier (K) + zvar for various equations depending on hetero
    K = length(frontier_spec.names)
    L = length(zvar_spec.names)

    # Build model to determine parameter count
    model = _build_model(noise, ineff)
    _validate_hetero(model.ineff, hetero)
    n_params = plen(model, K, L, hetero)

    # Build varnames based on equation structure
    # Structure follows _default_eq_ind order: frontier, ineff, noise
    varnames = Vector{String}(undef, n_params)

    # Frontier equation (K params)
    varnames[1:K] = frontier_names
    idx = K + 1

    # Inefficiency distribution parameters (before noise, to match _default_eq_ind)
    ineff_type = model.ineff
    if ineff_type isa TruncatedNormal_MCI
        # μ equation
        if :mu in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "μ"
            idx += 1
        end
        # ln_σᵤ² equation
        if :sigma_sq in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_σᵤ²"
            idx += 1
        end
    elseif ineff_type isa HalfNormal_MCI
        # ln_σᵤ² equation only
        if :sigma_sq in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_σᵤ²"
            idx += 1
        end
    elseif ineff_type isa Exponential_MCI
        # ln_λ equation
        if :lambda in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_λ"
            idx += 1
        end
    elseif ineff_type isa Gamma_MCI
        # ln_k (shape) and ln_θ (scale) equations
        if :k in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_k"
            idx += 1
        end
        if :theta in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_θ"
            idx += 1
        end
    elseif ineff_type isa Lognormal_MCI
        # μ equation
        if :mu in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "μ"
            idx += 1
        end
        # ln_σ2 equation
        if :sigma_sq in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_σ2"
            idx += 1
        end
    elseif ineff_type isa Weibull_MCI
        # ln_λ equation
        if :lambda in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_λ"
            idx += 1
        end
        # ln_k equation
        if :k in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_k"
            idx += 1
        end
    elseif ineff_type isa Lomax_MCI
        # ln_λ equation
        if :lambda in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_λ"
            idx += 1
        end
        # ln_α equation
        if :alpha in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_α"
            idx += 1
        end
    elseif ineff_type isa Rayleigh_MCI
        # ln_σ2 equation
        if :sigma_sq in hetero
            for name in zvar_names
                varnames[idx] = name
                idx += 1
            end
        else
            varnames[idx] = "ln_σ2"
            idx += 1
        end
    end

    # Noise equation (at the end, to match _default_eq_ind)
    if noise == :Laplace
        varnames[idx] = "ln_b"
    else
        varnames[idx] = "ln_σᵥ²"
    end
    idx += 1

    # StudentT has extra param
    if noise == :StudentT
        varnames[idx] = "ln_(ν-2)"
        idx += 1
    end

    # Call the matrix-input version with auto-generated varnames
    return sfmodel_MCI_spec(; depvar=depvar, frontier=frontier, zvar=zvar,
                             noise=noise, ineff=ineff, hetero=hetero,
                             transformation=transformation,
                             draws=draws, varnames=varnames,
                             eqnames=eqnames, eq_indices=eq_indices,
                             n_draws=n_draws, multiRand=multiRand,
                             GPU=GPU, type=type, chunks=chunks)
end
=# # END REMOVED legacy sfmodel_MCI_spec() constructor functions


# ============================================================================
# Section 3: Quantile Functions
# ============================================================================

"""
    myTruncatedNormalQuantile(; μ, σ, r, sqrt2=sqrt(2), inv_sqrt2=inv(sqrt(2)), clamp_lo=nothing, clamp_hi=nothing)

Compute quantile of TruncatedNormal(μ, σ; lower=0) at probability r.
Works on both CPU and GPU arrays via broadcasting.
"""
function myTruncatedNormalQuantile(; μ, σ, r,
                                    sqrt2=sqrt(2), inv_sqrt2=inv(sqrt(2)),
                                    clamp_lo=nothing, clamp_hi=nothing)
    erf_arg = @. r + (r - 1) * erf(μ * inv_sqrt2 / σ)
    if clamp_lo !== nothing && clamp_hi !== nothing
        erf_arg = @. clamp(erf_arg, clamp_lo, clamp_hi)
    end
    return @. μ + σ * sqrt2 * erfinv(erf_arg)
end

"""
    myTruncatedNormalQuantile!(out; μ, σ, r, sqrt2=sqrt(2), inv_sqrt2=inv(sqrt(2)), clamp_lo=nothing, clamp_hi=nothing)

In-place version of `myTruncatedNormalQuantile`. Stores result in `out`.
"""
function myTruncatedNormalQuantile!(out; μ, σ, r,
                                     sqrt2=sqrt(2), inv_sqrt2=inv(sqrt(2)),
                                     clamp_lo=nothing, clamp_hi=nothing)
    if clamp_lo !== nothing && clamp_hi !== nothing
        @. out = μ + σ * sqrt2 * erfinv(clamp(r + (r - 1) * erf(μ * inv_sqrt2 / σ), clamp_lo, clamp_hi))
    else
        @. out = μ + σ * sqrt2 * erfinv(r + (r - 1) * erf(μ * inv_sqrt2 / σ))
    end
    return out
end

"""
    myHalfNormalQuantile(; σ, r)

Compute quantile of HalfNormal(σ) at probability r.
HalfNormal is TruncatedNormal(0, σ; lower=0).
"""
function myHalfNormalQuantile(; σ, r)
    return @. sqrt(2) * σ * erfinv(r)
end

"""
    myStandardNormalQuantile(; r)

Compute quantile of Standard Normal N(0,1) at probability r.
"""
function myStandardNormalQuantile(; r)
    return @. sqrt(2) * erfinv(2 * r - 1)
end

"""
    myExponentialQuantile(; λ, r)

Compute quantile (inverse CDF) of Exponential at probability r,
where λ = Var(u) (variance parameterization).
F⁻¹(p; λ) = -ln(1-p) * √λ

# Arguments
- `λ`: Variance parameter (Var(u) = λ, positive scalar or array)
- `r`: Probability values in (0,1)
"""
function myExponentialQuantile(; λ, r)
    return @. -log(max(1 - r, eps(eltype(r)))) * sqrt(λ)
end

"""
    myExponentialQuantile!(out; λ, r, clamp_lo=nothing)

In-place version of `myExponentialQuantile`. Stores result in `out`.
Variance parameterization: Var(u) = λ.

# Arguments
- `out`: Pre-allocated output array (modified in-place)
- `λ`: Variance parameter (Var(u) = λ, positive scalar or array)
- `r`: Probability values in (0,1)
- `clamp_lo`: Optional lower clamp for (1-r) to avoid log(0)
"""
function myExponentialQuantile!(out; λ, r, clamp_lo=nothing)
    if clamp_lo !== nothing
        @. out = -log(max(1 - r, clamp_lo)) * sqrt(λ)
    else
        @. out = -log(max(1 - r, eps(eltype(r)))) * sqrt(λ)
    end
    return out
end

"""
    myHalfNormalQuantile!(out; σ, r, sqrt2, clamp_lo, clamp_hi)

Compute quantile of HalfNormal(σ) at probability r.
Q(p) = σ×2 �P erfinv(p)

Note: For HalfNormal, p ? [0, 1) maps to u ? [0, ×).
"""
function myHalfNormalQuantile!(out; σ, r, sqrt2, clamp_lo, clamp_hi)
    @. out = σ * sqrt2 * erfinv(clamp(r, clamp_lo, clamp_hi))
    return out
end

"""
    myWeibullQuantile!(out; λ, k, r, clamp_lo)

Compute quantile of Weibull(λ, k) at probability r.
Q(p; λ, k) = λ �P (-ln(1-p))^(1/k)

# Arguments
- `λ`: Scale parameter
- `k`: Shape parameter
"""
function myWeibullQuantile!(out; λ, k, r, clamp_lo)
    @. out = λ * (-log(max(1 - r, clamp_lo)))^(1 / k)
    return out
end

"""
    myLognormalQuantile!(out; μ, σ, r, sqrt2, clamp_lo, clamp_hi)

Compute quantile of LogNormal(μ, σ) at probability r.
Q(p; μ, σ) = exp(μ + σ×2 �P erfinv(2p-1))

# Arguments
- `μ`: Location parameter (mean of underlying normal)
- `σ`: Scale parameter (std of underlying normal)
"""
function myLognormalQuantile!(out; μ, σ, r, sqrt2, clamp_lo, clamp_hi)
    @. out = exp(μ + σ * sqrt2 * erfinv(clamp(2 * r - 1, clamp_lo, clamp_hi)))
    return out
end

"""
    myLomaxQuantile!(out; λ, α, r, clamp_lo, out_ceil)

Compute quantile of Lomax(α, λ) at probability r.
Q(p; α, λ) = λ * ((1-p)^(-1/α) - 1) = λ * expm1((-1/α) * log(1-p))
Uses expm1 for numerical stability when α is large.
"""
function myLomaxQuantile!(out; λ, α, r, clamp_lo, out_ceil)
    one_minus_r = max.(1 .- r, clamp_lo)
    @. out = clamp(λ * expm1((-1 / α) * log(one_minus_r)), 0, out_ceil)
    return out
end

"""
    myRayleighQuantile!(out; σ, r, clamp_lo)

Compute quantile of Rayleigh(σ) at probability r.
Q(p; σ) = σ �P ×(-2�Pln(1-p))

# Arguments
- `σ`: Scale parameter
"""
function myRayleighQuantile!(out; σ, r, clamp_lo)
    @. out = σ * sqrt(-2 * log(max(1 - r, clamp_lo)))
    return out
end

"""
    myGammaQuantile!(out; k, θ, r, clamp_lo)

Compute quantile of Gamma(k, θ) at probability r.
CPU-only implementation using Distributions.jl.

DEPRECATED: This function is no longer used for likelihood computation.
The T-approach in sf_MCI_Gamma_v17.jl is now used instead, which provides
GPU compatibility and ForwardDiff support.

This function is kept for backward compatibility and testing purposes.
"""
function myGammaQuantile!(out; k, θ, r, clamp_lo)
    # r is 1×D, k and θ are N×1
    # out is N×D
    N, D = size(out)
    for d in 1:D
        rd = clamp(r[1, d], clamp_lo, 1 - clamp_lo)
        for i in 1:N
            ki = k[i, 1]
            θi = θ[i, 1]
            out[i, d] = quantile(Gamma(ki, θi), rd)
        end
    end
    return out
end

# ============================================================================
# Section 4: PDF Functions
# ============================================================================

"""
    myNormalPDF(; z, σ, sqrt2=sqrt(2), sqrt_pi=sqrt(π))

Compute PDF of Normal(0, σ) at points z.
Returns: 1/(σ×(2π)) × exp(-0.5 × (z/σ)2)
"""
function myNormalPDF(; z, σ, sqrt2=sqrt(2), sqrt_pi=sqrt(π))
    return @. inv(σ * sqrt2 * sqrt_pi) * exp(-0.5 * (z / σ)^2)
end

"""
    myNormalPDF!(out; z, σ=nothing, sqrt2=sqrt(2), sqrt_pi=sqrt(π), pdf_const=nothing, inv_σ=nothing)

In-place version of `myNormalPDF`. When `pdf_const` and `inv_σ` are provided, uses fast path.
"""
function myNormalPDF!(out; z, σ=nothing, sqrt2=sqrt(2), sqrt_pi=sqrt(π),
                       pdf_const=nothing, inv_σ=nothing)
    if pdf_const !== nothing && inv_σ !== nothing
        @. out = pdf_const * exp(-0.5 * (z * inv_σ)^2)
    else
        @. out = inv(σ * sqrt2 * sqrt_pi) * exp(-0.5 * (z / σ)^2)
    end
    return out
end

"""
    myStudentTPDF(; z, σ_v, ν)

Compute PDF of zero-mean Student T with scale σ_v and degrees of freedom ν.
f(x; σ_v, ν) = �F((ν+1)/2) / (σ_v×(νπ) �F(ν/2)) × (1 + (x/σ_v)2/ν)^(-(ν+1)/2)

# Arguments
- `z`: Points at which to evaluate PDF
- `σ_v`: Scale parameter (positive)
- `ν`: Degrees of freedom (must be > 2 for finite variance)
"""
function myStudentTPDF(; z, σ_v, ν)
    log_c = loggamma((ν + 1) / 2) - loggamma(ν / 2) - 0.5 * log(ν * π) - log(σ_v)
    return @. exp(log_c - ((ν + 1) / 2) * log(1 + (z / σ_v)^2 / ν))
end

"""
    myStudentTPDF!(out; z, σ_v=nothing, ν=nothing, log_const=nothing, inv_σ_v=nothing, half_nu_plus_one=nothing)

In-place version of `myStudentTPDF`.
When `log_const`, `inv_σ_v`, and `half_nu_plus_one` are provided, uses fast path.

Fast path parameters:
- `log_const`: Precomputed log(�F((ν+1)/2)) - log(�F(ν/2)) - 0.5*log(νπ) - log(σ_v)
- `inv_σ_v`: Precomputed 1/σ_v
- `half_nu_plus_one`: Precomputed (ν+1)/2
"""
function myStudentTPDF!(out; z, σ_v=nothing, ν=nothing,
                         log_const=nothing, inv_σ_v=nothing, half_nu_plus_one=nothing)
    if log_const !== nothing && inv_σ_v !== nothing && half_nu_plus_one !== nothing
        # Fast path with precomputed constants
        # NOTE: ν computation inlined to ensure proper broadcasting with CuArray + ForwardDiff
        @. out = exp(log_const - half_nu_plus_one * log(1 + (z * inv_σ_v)^2 / (2 * half_nu_plus_one - 1)))
    else
        log_c = loggamma((ν + 1) / 2) - loggamma(ν / 2) - 0.5 * log(ν * π) - log(σ_v)
        @. out = exp(log_c - ((ν + 1) / 2) * log(1 + (z / σ_v)^2 / ν))
    end
    return out
end


"""
    myLaplacePDF(; z, b)

Compute PDF of Laplace(0, b) at points z.
f(x; b) = exp(-|x|/b) / (2b)

# Arguments
- `z`: Points at which to evaluate PDF
- `b`: Scale parameter (positive)
"""
function myLaplacePDF(; z, b)
    # return @. exp(-abs(z) / b) / (2 * b)
    return @. exp(-sqrt(z^2 + (1e-8)^2) / b) / (2 * b)
end

"""
    myLaplacePDF!(out; z, b=nothing, inv_b=nothing, inv_2b=nothing)

In-place version of `myLaplacePDF`.
When `inv_b` and `inv_2b` are provided, uses fast path.

Fast path parameters:
- `inv_b`: Precomputed 1/b
- `inv_2b`: Precomputed 1/(2b)
"""
function myLaplacePDF!(out; z, b=nothing, inv_b=nothing, inv_2b=nothing)
    if inv_b !== nothing && inv_2b !== nothing
       # @. out = inv_2b * exp(-abs(z) * inv_b)
       @. out = inv_2b * exp(-sqrt(z^2 + (1e-8)^2) * inv_b)
    else
       # @. out = exp(-abs(z) / b) / (2 * b)
       @. out = exp(-sqrt(z^2 + (1e-8)^2) / b) / (2 * b)
    end
    return out
end

# ============================================================================
# Section 5: Constants and Utilities
# ============================================================================

"""
    make_halton_p(draws::Int; base::Int=2, T::Type{<:AbstractFloat}=Float64)

Generate Halton probabilities in (0,1) for MCI integration.
Call this ONCE outside the likelihood function and reuse.
"""
function make_halton_p(draws::Int; base::Int=2, T::Type{<:AbstractFloat}=Float64)
    draws > 0 || throw(ArgumentError("draws must be positive, got $draws"))
    return T.(collect(Halton(base, length=draws)))
end

"""
    make_halton_wrap(N::Int, D::Int; base::Int=2, T::Type{<:AbstractFloat}=Float64)

Generate a wrapped Halton quasi-random sequence matrix for multiRand mode.

Returns an N × D matrix where each observation gets different consecutive
Halton sequence elements, providing greater variation across observations.

# Arguments
- `N::Int`: Number of observations
- `D::Int`: Number of draws per observation (must be ≤ `distinct_Halton_length`)
- `base::Int=2`: Base for Halton sequence generation
- `T::Type{<:AbstractFloat}=Float64`: Element type (Float64 or Float32)
- `distinct_Halton_length::Int=2^15-1`: Maximum Halton sequence length (controls the cap on sequence generation)

# Returns
- `Matrix{T}` of size (N, D) with Halton values in (0,1)

# Throws
- `ArgumentError` if D > `distinct_Halton_length` (use multiRand=false for larger draws)
"""
function make_halton_wrap(N::Int, D::Int; base::Int=2, T::Type{<:AbstractFloat}=Float64, distinct_Halton_length::Int=2^15-1)
    N > 0 || throw(ArgumentError("N must be positive, got $N"))
    D > 0 || throw(ArgumentError("D (draws) must be positive, got $D"))

    # Validate D is within limits for multiRand mode
    if D > distinct_Halton_length
        throw(ArgumentError("`n_draws` (=$D, which is per observation draws) is too large for mltiRand=true. " *
                           "Three options. (1) Reduce it to ≤ $distinct_Halton_length . " *
                           "(2) Set `multiRand=false` and each observation will use the same Halton draws. " *
                           "(3) Increase `distinct_Halton_length` to be larger than your `n_draws`."))
    end

    # Compute sequence length: find largest n where 2^n - 1 <= total_needed, capped by distinct_Halton_length
    total_needed = D * N
    n = floor(Int, log2(total_needed + 1))
    n_cap = floor(Int, log2(distinct_Halton_length + 1))
    n = min(n, n_cap)
    seq_len = 2^n - 1

    # Generate base Halton sequence
    myH = T.(collect(Halton(base, length=seq_len)))

    # Recycle sequence to fill exactly D*N elements
    if seq_len >= total_needed
        wrapped = myH[1:total_needed]
    else
        # Repeat sequence as many times as needed, then take the remainder
        n_full_repeats = total_needed ÷ seq_len
        remainder = total_needed % seq_len
        wrapped = vcat(repeat(myH, n_full_repeats), myH[1:remainder])
    end

    # Reshape to N x D matrix
    # Julia is column-major, so reshape(wrapped, D, N)' gives us row i = draws[(i-1)*D+1 : i*D]
    return copy(reshape(wrapped, D, N)')
end

"""
    make_constants(model::MCIModel, T::Type{<:AbstractFloat}=Float64)

Precompute invariant constants for the given model. Call this ONCE before optimization.
"""
function make_constants(model::MCIModel, T::Type{<:AbstractFloat}=Float64)
    sqrt2 = sqrt(T(2))
    base = (
        clamp_lo  = T(1e-15),     # Universal lower clamp for log safety
        σ_floor   = T(1e-12),
        σ_ceil    = T(1e12),
        sqrt2     = sqrt2,
        inv_sqrt2 = inv(sqrt2),
        sqrt_pi   = sqrt(T(π)),
    )
    noise_c = make_noise_constants(model.noise, T)
    ineff_c = make_ineff_constants(model.ineff, T)
    copula_c = make_copula_constants(model.copula, T)
    return merge(base, noise_c, ineff_c, copula_c)
end

make_noise_constants(::NormalNoise_MCI, T) = NamedTuple()
make_noise_constants(::StudentTNoise_MCI, T) = NamedTuple()
make_noise_constants(::LaplaceNoise_MCI, T) = NamedTuple()

make_ineff_constants(::TruncatedNormal_MCI, T) = (
    lo_erf = -one(T) + T(32) * eps(T),
    hi_erf =  one(T) - T(32) * eps(T),
)
make_ineff_constants(::Exponential_MCI, T) = (
    exp_clamp = T(1e-15),
)
make_ineff_constants(::HalfNormal_MCI, T) = (
    lo_erf = T(32) * eps(T),           # erfinv domain: [0, 1)
    hi_erf = one(T) - T(32) * eps(T),
)
make_ineff_constants(::Weibull_MCI, T) = (
    clamp_lo = T(1e-15),  # For log(1-r)
    k_floor = T(0.1),     # Min shape parameter (prevents 1/k explosion)
    k_ceil = T(10.0),     # Max shape parameter (reasonable upper bound)
)
make_ineff_constants(::Lognormal_MCI, T) = (
    lo_erf = -one(T) + T(32) * eps(T),  # erfinv(2r-1) domain: (-1, 1)
    hi_erf = one(T) - T(32) * eps(T),
)
make_ineff_constants(::Lomax_MCI, T) = (
    clamp_lo = T(1e-15),
    α_floor = T(0.1),
    α_ceil = T(100.0),
    lambda_floor = T(1e-10),
    lambda_ceil = T(1e10),
    out_ceil = T(1e15),
)
make_ineff_constants(::Rayleigh_MCI, T) = (
    clamp_lo = T(1e-15),  # For log(1-r)
)
make_ineff_constants(::Gamma_MCI, T) = (
    clamp_lo = T(1e-15),     # For quantile edge cases
    k_floor = T(0.1),        # Min shape parameter
    k_ceil = T(100.0),       # Max shape parameter
    theta_floor = T(1e-12),
    theta_ceil = T(1e12),
)

make_copula_constants(::NoCopula_MCI, T) = NamedTuple()
make_copula_constants(::GaussianCopula_MCI, T) = (
    copula_clamp_lo = T(1e-15),
    copula_clamp_hi = one(T) - T(1e-15),
    lo_erfinv = -one(T) + T(32) * eps(T),
    hi_erfinv =  one(T) - T(32) * eps(T),
    rho_max = T(0.999),
)
make_copula_constants(::ClaytonCopula_MCI, T) = (
    copula_clamp_lo = T(1e-6),                  # Clamp F_v, F_u away from 0 (u^(-rho) blows up)
    copula_clamp_hi = one(T) - T(1e-6),         # Clamp F_v, F_u away from 1
    clayton_rho_floor = T(1e-6),                 # Floor rho away from 0 to prevent -1/rho -> -Inf
    clayton_rho_max = T(50.0),                   # Ceiling rho to prevent u^(-rho) overflow
)
make_copula_constants(::GumbelCopula_MCI, T) = (
    copula_clamp_lo = T(1e-16),                  # Clamp F_v, F_u away from 0 (-log(u) -> Inf)
    copula_clamp_hi = one(T) - T(1e-16),         # Clamp F_v, F_u away from 1
    gumbel_sum_floor = T(1e-16),                 # Floor for w1^rho + w2^rho to avoid 0^(negative)
    gumbel_rho_max = T(50.0),                    # Ceiling rho to prevent w^rho overflow
)
make_copula_constants(::Clayton90Copula_MCI, T) = (
    copula_clamp_lo = T(1e-6),                   # Clamp F_v, F_u away from 0 (u^(-rho) blows up)
    copula_clamp_hi = one(T) - T(1e-6),          # Clamp F_v, F_u away from 1
    clayton_rho_floor = T(1e-6),                 # Floor rho away from 0 to prevent -1/rho -> -Inf
    clayton_rho_max = T(50.0),                   # Ceiling rho to prevent u^(-rho) overflow
)

# ============================================================================
# Section 6: (Removed - parameter access now uses _param_ind in Section 2b)
# ============================================================================

# ============================================================================
# Section 7: Computation Helpers (SFGPU-style with broadcasting)
# ============================================================================

# --- Noise parameter computation ---
# New signature: (noise_type, p, idx, c) where p is CPU AbstractVector{P}
# P (element type of p) can be Float64 or Dual for ForwardDiff

function get_noise_vals(::NormalNoise_MCI, p, idx, c)
    P = eltype(p)
    sigma_v = clamp(exp(P(0.5) * p[idx.noise.ln_sigma_v_sq]), c.σ_floor, c.σ_ceil)
    inv_sigma_v = inv(sigma_v)
    pdf_const = inv(sigma_v * c.sqrt2 * c.sqrt_pi)
    return (sigma_v=sigma_v, inv_sigma_v=inv_sigma_v, pdf_const=pdf_const)
end

function get_noise_vals(::StudentTNoise_MCI, p, idx, c)
    P = eltype(p)
    # Direct indexing from CPU parameter vector - no _scalar needed!
    # Works with ForwardDiff.Dual because p stays on CPU

    sigma_v = clamp(exp(P(0.5) * p[idx.noise.ln_sigma_v_sq]), c.σ_floor, c.σ_ceil)
    inv_sigma_v = inv(sigma_v)

    # ν = exp(ln_nu_minus_2) + 2, ensures ν > 2
    nu = exp(p[idx.noise.ln_nu_minus_2]) + P(2)
    half_nu_plus_one = (nu + 1) / P(2)

    # loggamma on CPU scalar - ForwardDiff's digamma works on CPU!
    log_const = loggamma((nu + 1) / P(2)) - loggamma(nu / P(2)) -
                P(0.5) * log(nu * P(π)) - log(sigma_v)

    return (sigma_v=sigma_v, inv_sigma_v=inv_sigma_v, nu=nu,
            half_nu_plus_one=half_nu_plus_one, log_const=log_const)
end

function get_noise_vals(::LaplaceNoise_MCI, p, idx, c)
    P = eltype(p)
    b = clamp(exp(p[idx.noise.ln_b]), c.σ_floor, c.σ_ceil)
    inv_b = inv(b)
    inv_2b = inv(P(2) * b)
    return (b=b, inv_b=inv_b, inv_2b=inv_2b)
end

# --- Inefficiency parameter computation ---
# New signature: (ineff_type, p, idx, Z, c) using broadcasting instead of mul!
# Broadcasting pattern: sum(P(p[idx[j]]) .* (@view Z[:, j]) for j in 1:k)

function get_ineff_vals(::TruncatedNormal_MCI, p, idx, Z, c)
    P = eltype(p)
    ineff = idx.ineff

    # μ = Z * delta (broadcasting loop)
    k_mu = length(ineff.mu)
    mu = sum(P(p[ineff.mu[j]]) .* (@view Z[:, j]) for j in 1:k_mu)

    # σ_u = exp(0.5 * Z * gamma)
    k_sigma = length(ineff.sigma_u)
    sigma_u = exp.(P(0.5) .* sum(P(p[ineff.sigma_u[j]]) .* (@view Z[:, j]) for j in 1:k_sigma))
    sigma_u = clamp.(sigma_u, c.σ_floor, c.σ_ceil)

    return (mu=mu, sigma_u=sigma_u)
end

function get_ineff_vals(::Exponential_MCI, p, idx, Z, c)
    P = eltype(p)
    ineff = idx.ineff

    # λ = exp(Z * gamma), where λ = Var(u)
    k_lambda = length(ineff.lambda)
    lambda = exp.(sum(P(p[ineff.lambda[j]]) .* (@view Z[:, j]) for j in 1:k_lambda))
    lambda = clamp.(lambda, c.σ_floor, c.σ_ceil)

    return (lambda=lambda,)
end

function get_ineff_vals(::HalfNormal_MCI, p, idx, Z, c)
    P = eltype(p)
    ineff = idx.ineff

    # σ = exp(0.5 * Z * gamma)
    k_sigma = length(ineff.sigma_sq)
    sigma = exp.(P(0.5) .* sum(P(p[ineff.sigma_sq[j]]) .* (@view Z[:, j]) for j in 1:k_sigma))
    sigma = clamp.(sigma, c.σ_floor, c.σ_ceil)

    return (sigma=sigma,)
end

function get_ineff_vals(::Weibull_MCI, p, idx, Z, c)
    P = eltype(p)
    ineff = idx.ineff

    # λ = exp(Z * gamma_lambda)
    k_lambda = length(ineff.lambda)
    lambda = exp.(sum(P(p[ineff.lambda[j]]) .* (@view Z[:, j]) for j in 1:k_lambda))
    lambda = clamp.(lambda, c.σ_floor, c.σ_ceil)

    # k = exp(Z * gamma_k)
    k_k = length(ineff.k)
    k_val = exp.(sum(P(p[ineff.k[j]]) .* (@view Z[:, j]) for j in 1:k_k))
    k_val = clamp.(k_val, c.k_floor, c.k_ceil)

    return (lambda=lambda, k=k_val)
end

function get_ineff_vals(::Lognormal_MCI, p, idx, Z, c)
    P = eltype(p)
    ineff = idx.ineff

    # μ = Z * delta
    k_mu = length(ineff.mu)
    mu = sum(P(p[ineff.mu[j]]) .* (@view Z[:, j]) for j in 1:k_mu)

    # σ = exp(0.5 * Z * gamma)
    k_sigma = length(ineff.sigma_sq)
    sigma = exp.(P(0.5) .* sum(P(p[ineff.sigma_sq[j]]) .* (@view Z[:, j]) for j in 1:k_sigma))
    sigma = clamp.(sigma, c.σ_floor, c.σ_ceil)

    return (mu=mu, sigma=sigma)
end

function get_ineff_vals(::Lomax_MCI, p, idx, Z, c)
    P = eltype(p)
    ineff = idx.ineff

    # λ = exp(Z * γ_λ) — can be heteroscedastic
    k_lambda = length(ineff.ln_lambda)
    lambda = exp.(sum(P(p[ineff.ln_lambda[j]]) .* (@view Z[:, j]) for j in 1:k_lambda))
    lambda = clamp.(lambda, c.lambda_floor, c.lambda_ceil)

    # α = exp(Z * γ_α) — can be heteroscedastic
    k_alpha = length(ineff.alpha)
    alpha = exp.(sum(P(p[ineff.alpha[j]]) .* (@view Z[:, j]) for j in 1:k_alpha))
    alpha = clamp.(alpha, c.α_floor, c.α_ceil)

    return (lambda=lambda, alpha=alpha)
end

function get_ineff_vals(::Rayleigh_MCI, p, idx, Z, c)
    P = eltype(p)
    ineff = idx.ineff

    # σ = exp(0.5 * Z * gamma)
    k_sigma = length(ineff.sigma_sq)
    sigma = exp.(P(0.5) .* sum(P(p[ineff.sigma_sq[j]]) .* (@view Z[:, j]) for j in 1:k_sigma))
    sigma = clamp.(sigma, c.σ_floor, c.σ_ceil)

    return (sigma=sigma,)
end

function get_ineff_vals(::Gamma_MCI, p, idx, Z, c)
    P = eltype(p)
    ineff = idx.ineff

    # k = exp(Z * gamma_k)
    k_len = length(ineff.k)
    k_val = exp.(sum(P(p[ineff.k[j]]) .* (@view Z[:, j]) for j in 1:k_len))
    k_val = clamp.(k_val, c.k_floor, c.k_ceil)

    # θ = exp(Z * gamma_theta)
    theta_len = length(ineff.theta)
    theta_val = exp.(sum(P(p[ineff.theta[j]]) .* (@view Z[:, j]) for j in 1:theta_len))
    theta_val = clamp.(theta_val, c.theta_floor, c.theta_ceil)

    # lgk = loggamma(k) for log-PDF computation in T-approach
    lgk_val = loggamma.(k_val)

    return (k=k_val, theta=theta_val, lgk=lgk_val)
end

# --- Quantile computation (generates inefficiency draws) ---

function get_u_quantile!(::TruncatedNormal_MCI, buffer, ineff_vals, draws_1D, c, N)
    mu_N1 = reshape(ineff_vals.mu, N, 1)
    sigma_N1 = reshape(ineff_vals.sigma_u, N, 1)

    myTruncatedNormalQuantile!(buffer; μ=mu_N1, σ=sigma_N1, r=draws_1D,
                               sqrt2=c.sqrt2, inv_sqrt2=c.inv_sqrt2,
                               clamp_lo=c.lo_erf, clamp_hi=c.hi_erf)
end

function get_u_quantile!(::Exponential_MCI, buffer, ineff_vals, draws_1D, c, N)
    lambda_N1 = reshape(ineff_vals.lambda, N, 1)

    myExponentialQuantile!(buffer; λ=lambda_N1, r=draws_1D, clamp_lo=c.exp_clamp)
end

function get_u_quantile!(::HalfNormal_MCI, buffer, ineff_vals, draws_1D, c, N)
    sigma_N1 = reshape(ineff_vals.sigma, N, 1)

    myHalfNormalQuantile!(buffer; σ=sigma_N1, r=draws_1D,
                          sqrt2=c.sqrt2, clamp_lo=c.lo_erf, clamp_hi=c.hi_erf)
end

function get_u_quantile!(::Weibull_MCI, buffer, ineff_vals, draws_1D, c, N)
    lambda_N1 = reshape(ineff_vals.lambda, N, 1)
    k_N1 = reshape(ineff_vals.k, N, 1)

    myWeibullQuantile!(buffer; λ=lambda_N1, k=k_N1, r=draws_1D, clamp_lo=c.clamp_lo)
end

function get_u_quantile!(::Lognormal_MCI, buffer, ineff_vals, draws_1D, c, N)
    mu_N1 = reshape(ineff_vals.mu, N, 1)
    sigma_N1 = reshape(ineff_vals.sigma, N, 1)

    myLognormalQuantile!(buffer; μ=mu_N1, σ=sigma_N1, r=draws_1D,
                         sqrt2=c.sqrt2, clamp_lo=c.lo_erf, clamp_hi=c.hi_erf)
end

function get_u_quantile!(::Lomax_MCI, buffer, ineff_vals, draws_1D, c, N)
    lambda_N1 = reshape(ineff_vals.lambda, N, 1)
    alpha_N1 = reshape(ineff_vals.alpha, N, 1)

    myLomaxQuantile!(buffer; λ=lambda_N1, α=alpha_N1, r=draws_1D,
                      clamp_lo=c.clamp_lo, out_ceil=c.out_ceil)
end

function get_u_quantile!(::Rayleigh_MCI, buffer, ineff_vals, draws_1D, c, N)
    sigma_N1 = reshape(ineff_vals.sigma, N, 1)

    myRayleighQuantile!(buffer; σ=sigma_N1, r=draws_1D, clamp_lo=c.clamp_lo)
end

function get_u_quantile!(::Gamma_MCI, buffer, ineff_vals, draws_1D, c, N)
    k_N1 = reshape(ineff_vals.k, N, 1)
    theta_N1 = reshape(ineff_vals.theta, N, 1)

    myGammaQuantile!(buffer; k=k_N1, θ=theta_N1, r=draws_1D, clamp_lo=c.clamp_lo)
end

# --- Chunked quantile computation (for GPU memory management when chunks > 1) ---

function get_u_quantile_chunk!(::TruncatedNormal_MCI, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N)
    mu_chunk = reshape((@view ineff_vals.mu[row_start:row_end]), chunk_N, 1)
    sig_chunk = reshape((@view ineff_vals.sigma_u[row_start:row_end]), chunk_N, 1)

    myTruncatedNormalQuantile!(buf; μ=mu_chunk, σ=sig_chunk, r=draws_1D,
                               sqrt2=c.sqrt2, inv_sqrt2=c.inv_sqrt2,
                               clamp_lo=c.lo_erf, clamp_hi=c.hi_erf)
end

function get_u_quantile_chunk!(::Exponential_MCI, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N)
    lambda_chunk = reshape((@view ineff_vals.lambda[row_start:row_end]), chunk_N, 1)

    myExponentialQuantile!(buf; λ=lambda_chunk, r=draws_1D, clamp_lo=c.exp_clamp)
end

function get_u_quantile_chunk!(::HalfNormal_MCI, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N)
    sigma_chunk = reshape((@view ineff_vals.sigma[row_start:row_end]), chunk_N, 1)

    myHalfNormalQuantile!(buf; σ=sigma_chunk, r=draws_1D,
                          sqrt2=c.sqrt2, clamp_lo=c.lo_erf, clamp_hi=c.hi_erf)
end

function get_u_quantile_chunk!(::Weibull_MCI, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N)
    lambda_chunk = reshape((@view ineff_vals.lambda[row_start:row_end]), chunk_N, 1)
    k_chunk = reshape((@view ineff_vals.k[row_start:row_end]), chunk_N, 1)

    myWeibullQuantile!(buf; λ=lambda_chunk, k=k_chunk, r=draws_1D, clamp_lo=c.clamp_lo)
end

function get_u_quantile_chunk!(::Lognormal_MCI, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N)
    mu_chunk = reshape((@view ineff_vals.mu[row_start:row_end]), chunk_N, 1)
    sigma_chunk = reshape((@view ineff_vals.sigma[row_start:row_end]), chunk_N, 1)

    myLognormalQuantile!(buf; μ=mu_chunk, σ=sigma_chunk, r=draws_1D,
                         sqrt2=c.sqrt2, clamp_lo=c.lo_erf, clamp_hi=c.hi_erf)
end

function get_u_quantile_chunk!(::Lomax_MCI, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N)
    lambda_chunk = reshape((@view ineff_vals.lambda[row_start:row_end]), chunk_N, 1)
    alpha_chunk = reshape((@view ineff_vals.alpha[row_start:row_end]), chunk_N, 1)

    myLomaxQuantile!(buf; λ=lambda_chunk, α=alpha_chunk, r=draws_1D,
                      clamp_lo=c.clamp_lo, out_ceil=c.out_ceil)
end

function get_u_quantile_chunk!(::Rayleigh_MCI, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N)
    sigma_chunk = reshape((@view ineff_vals.sigma[row_start:row_end]), chunk_N, 1)

    myRayleighQuantile!(buf; σ=sigma_chunk, r=draws_1D, clamp_lo=c.clamp_lo)
end

function get_u_quantile_chunk!(::Gamma_MCI, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N)
    k_chunk = reshape((@view ineff_vals.k[row_start:row_end]), chunk_N, 1)
    theta_chunk = reshape((@view ineff_vals.theta[row_start:row_end]), chunk_N, 1)

    myGammaQuantile!(buf; k=k_chunk, θ=theta_chunk, r=draws_1D, clamp_lo=c.clamp_lo)
end

# --- PDF computation (evaluates noise density) ---

function get_noise_pdf!(::NormalNoise_MCI, out, z, noise_vals, c)
    myNormalPDF!(out; z=z, pdf_const=noise_vals.pdf_const, inv_σ=noise_vals.inv_sigma_v)
end

function get_noise_pdf!(::StudentTNoise_MCI, out, z, noise_vals, c)
    myStudentTPDF!(out; z=z, log_const=noise_vals.log_const,
                   inv_σ_v=noise_vals.inv_sigma_v,
                   half_nu_plus_one=noise_vals.half_nu_plus_one)
end

function get_noise_pdf!(::LaplaceNoise_MCI, out, z, noise_vals, c)
    myLaplacePDF!(out; z=z, inv_b=noise_vals.inv_b, inv_2b=noise_vals.inv_2b)
end

# --- Log-space noise PDF computation (for T-approach) ---

function log_noise_pdf!(::NormalNoise_MCI, out, z, noise_vals, c)
    # log f(z) = -0.5*log(2π) - log(σ) - 0.5*(z/σ)²
    inv_σ = noise_vals.inv_sigma_v
    @. out = -0.5 * log(2π) - log(max(inv(inv_σ), c.clamp_lo)) - 0.5 * (z * inv_σ)^2
end

function log_noise_pdf!(::StudentTNoise_MCI, out, z, noise_vals, c)
    # log f(z) = log_const - (ν+1)/2 * log(1 + (z/σ)²/ν)
    inv_σ = noise_vals.inv_sigma_v
    half_nu_plus_one = noise_vals.half_nu_plus_one
    nu = 2 * half_nu_plus_one - 1  # Recover ν from (ν+1)/2
    @. out = noise_vals.log_const - half_nu_plus_one * log(1 + (z * inv_σ)^2 / nu)
end

function log_noise_pdf!(::LaplaceNoise_MCI, out, z, noise_vals, c)
    # log f(z) = log(inv_2b) - |z|*inv_b
    # Use smoothed abs for AD: sqrt(z² + ε²)
    inv_b = noise_vals.inv_b
    inv_2b = noise_vals.inv_2b
    @. out = log(max(inv_2b, c.clamp_lo)) - sqrt(z^2 + 1e-16) * inv_b
end

# Chunked versions (same logic, different signature for chunk processing)
function log_noise_pdf_chunk!(::NormalNoise_MCI, out, z, noise_vals, c, row_start, row_end)
    inv_σ = noise_vals.inv_sigma_v
    @. out = -0.5 * log(2π) - log(max(inv(inv_σ), c.clamp_lo)) - 0.5 * (z * inv_σ)^2
end

function log_noise_pdf_chunk!(::StudentTNoise_MCI, out, z, noise_vals, c, row_start, row_end)
    inv_σ = noise_vals.inv_sigma_v
    half_nu_plus_one = noise_vals.half_nu_plus_one
    nu = 2 * half_nu_plus_one - 1
    @. out = noise_vals.log_const - half_nu_plus_one * log(1 + (z * inv_σ)^2 / nu)
end

function log_noise_pdf_chunk!(::LaplaceNoise_MCI, out, z, noise_vals, c, row_start, row_end)
    inv_b = noise_vals.inv_b
    inv_2b = noise_vals.inv_2b
    @. out = log(max(inv_2b, c.clamp_lo)) - sqrt(z^2 + 1e-16) * inv_b
end

# ============================================================================
# GPU-safe reduction helpers (avoid CPU-fallback on CuArray{Dual})
# ============================================================================
# These helpers are required by sf_MCI_T_v18.jl for logsumexp_rows computation.
# They provide transparent dispatch to GPU-specific functions when CUDA is available.

_maximum(A; dims) = maximum(A; dims=dims)
_sum(A; dims)     = sum(A; dims=dims)
_sum_scalar(v::AbstractArray) = sum(v)

# Conditional GPU overloads (if CUDA is loaded in Main)
if isdefined(Main, :CUDA)
    _maximum(A::Main.CUDA.AnyCuArray; dims) = Main.CUDA.maximum(A; dims=dims)
    _sum(A::Main.CUDA.AnyCuArray; dims)     = Main.CUDA.sum(A; dims=dims)
    _sum_scalar(v::Main.CUDA.AnyCuArray)    = sum(v)
end

"""
    logsumexp_rows(A::AbstractMatrix)

Compute log(sum(exp(A[i,:]))) for each row i, using the log-sum-exp trick
for numerical stability.

For each row: logsumexp(x) = max(x) + log(sum(exp(x - max(x))))

# Arguments
- `A`: N×D matrix of log-values

# Returns
- N-vector of log-sum-exp values
"""
function logsumexp_rows(A::AbstractMatrix, clamp_lo=1e-300)
    max_vals = _maximum(A; dims=2)
    sum_exp  = _sum(exp.(A .- max_vals); dims=2)
    # Use oftype to convert clamp_lo to match element type (works with ForwardDiff.Dual)
    fmin = oftype(zero(eltype(A)), clamp_lo)
    @. sum_exp = max(sum_exp, fmin)
    return vec(max_vals .+ log.(sum_exp))
end

# ============================================================================
# Section 7b: Copula Helper Functions
# ============================================================================

# --- Noise CDF functions (in-place, needed for copula: computes F_v(v)) ---

"""Compute CDF of Normal noise in-place: out .= F_v(v)"""
function noise_cdf!(::NormalNoise_MCI, out, z, noise_vals, c)
    @. out = 0.5 * (1 + erf(z * noise_vals.inv_sigma_v * c.inv_sqrt2))
    return nothing
end

"""Compute CDF of Laplace noise in-place: out .= F_v(v)"""
function noise_cdf!(::LaplaceNoise_MCI, out, z, noise_vals, c)
    inv_b = noise_vals.inv_b
    @. out = ifelse(z < 0, 0.5 * exp(z * inv_b), 1 - 0.5 * exp(-z * inv_b))
    return nothing
end

"""StudentT noise CDF is not yet supported for copula (unreachable due to _build_model validation)."""
function noise_cdf!(::StudentTNoise_MCI, out, z, noise_vals, c)
    error("StudentT noise CDF is not yet implemented for copula models. " *
          "Use Normal or Laplace noise with copula.")
end

# --- Copula parameter extraction ---

"""No copula: return empty NamedTuple."""
get_copula_vals(::NoCopula_MCI, p, idx, c) = NamedTuple()

"""Gaussian copula: extract ρ = ρ_max · tanh(θ_rho) from parameter vector."""
function get_copula_vals(::GaussianCopula_MCI, p, idx, c)
    theta_rho = p[idx.copula.theta_rho]
    rho = c.rho_max * tanh(theta_rho)
    return (rho=rho, theta_rho=theta_rho)
end

"""Clayton copula: rho = clamp(exp(theta_rho) + floor, floor, max), ensuring rho > 0 and bounded."""
function get_copula_vals(::ClaytonCopula_MCI, p, idx, c)
    theta_rho = p[idx.copula.theta_rho]
    rho = clamp(exp(theta_rho) + c.clayton_rho_floor, c.clayton_rho_floor, c.clayton_rho_max)
    return (rho=rho, theta_rho=theta_rho)
end

"""Gumbel copula: rho = clamp(exp(theta_rho) + 1, 1, max), ensuring rho >= 1 and bounded."""
function get_copula_vals(::GumbelCopula_MCI, p, idx, c)
    theta_rho = p[idx.copula.theta_rho]
    rho = clamp(exp(theta_rho) + one(typeof(theta_rho)), one(typeof(theta_rho)), c.gumbel_rho_max)
    return (rho=rho, theta_rho=theta_rho)
end

"""Clayton 90° copula: same parameterization as Clayton (ρ = exp(θ) + floor)."""
function get_copula_vals(::Clayton90Copula_MCI, p, idx, c)
    theta_rho = p[idx.copula.theta_rho]
    rho = clamp(exp(theta_rho) + c.clayton_rho_floor, c.clayton_rho_floor, c.clayton_rho_max)
    return (rho=rho, theta_rho=theta_rho)
end

# --- Copula log-adjustment dispatch ---

"""No copula: no-op."""
copula_log_adjustment!(::NoCopula_MCI, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c) = nothing

"""
    copula_log_adjustment!(::GaussianCopula_MCI, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c)

Compute the copula log-density adjustment in-place into `adj`.
Uses `Fv_buf` as workspace for the noise CDF. Zero intermediate allocations —
all operations are fused into a single broadcast kernel via `@.`.

- `adj`: output buffer (N×D), receives log copula density
- `Fv_buf`: workspace buffer (N×D), same size as adj
- `z_buffer`: composite error v = ε + sign*u (before log_noise_pdf! overwrites it)
- `draws`: the Halton draws (= F_u(u) by the T-approach probability integral transform)
"""
function copula_log_adjustment!(::GaussianCopula_MCI, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c)
    # Step 1: Compute F_v into Fv_buf (in-place, no allocation)
    noise_cdf!(noise, Fv_buf, z_buffer, noise_vals, c)

    # Step 2: Compute log copula density directly into adj
    # Single fused broadcast — erfinv(clamp(...)) subexpressions are recomputed
    # (trades compute for zero allocations; erfinv is cheap vs memory/GC pressure)
    rho = copula_vals.rho
    rho_sq = rho * rho
    omr2 = 1 - rho_sq
    @. adj = -0.5 * log(omr2) - (
        rho_sq * (
            (c.sqrt2 * erfinv(clamp(2 * clamp(Fv_buf, c.copula_clamp_lo, c.copula_clamp_hi) - 1, c.lo_erfinv, c.hi_erfinv)))^2 +
            (c.sqrt2 * erfinv(clamp(2 * clamp(draws,  c.copula_clamp_lo, c.copula_clamp_hi) - 1, c.lo_erfinv, c.hi_erfinv)))^2
        ) - 2 * rho *
            (c.sqrt2 * erfinv(clamp(2 * clamp(Fv_buf, c.copula_clamp_lo, c.copula_clamp_hi) - 1, c.lo_erfinv, c.hi_erfinv))) *
            (c.sqrt2 * erfinv(clamp(2 * clamp(draws,  c.copula_clamp_lo, c.copula_clamp_hi) - 1, c.lo_erfinv, c.hi_erfinv)))
    ) / (2 * omr2)
    return nothing
end

"""
    copula_log_adjustment!(::ClaytonCopula_MCI, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c)

Compute Clayton copula log-density adjustment in-place into `adj`.
Clayton density: c(u1,u2) = (rho+1)(u1*u2)^{-(rho+1)} [u1^{-rho} + u2^{-rho} - 1]^{-1/rho-2}
Log form: log(rho+1) - (rho+1)(log u1 + log u2) + (-1/rho-2) log[u1^{-rho} + u2^{-rho} - 1]
"""
function copula_log_adjustment!(::ClaytonCopula_MCI, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c)
    # Step 1: Compute F_v into Fv_buf (in-place, no allocation)
    noise_cdf!(noise, Fv_buf, z_buffer, noise_vals, c)

    # Step 2: Compute log copula density directly into adj
    rho = copula_vals.rho
    rho_p1 = rho + 1
    neg_inv_rho_m2 = -1 / rho - 2
    lo = c.copula_clamp_lo
    hi = c.copula_clamp_hi

    @. adj = log(rho_p1) - rho_p1 * (
                log(clamp(Fv_buf, lo, hi)) + log(clamp(draws, lo, hi))
             ) + neg_inv_rho_m2 * log(max(
                clamp(Fv_buf, lo, hi)^(-rho) + clamp(draws, lo, hi)^(-rho) - 1, lo
             ))
    return nothing
end

"""
    copula_log_adjustment!(::GumbelCopula_MCI, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c)

Compute Gumbel copula log-density adjustment in-place into `adj`.
Let w1=-log(u1), w2=-log(u2), S=w1^rho+w2^rho.
Density: exp(-S^{1/rho}) w1^{rho-1} [(rho-1)S^{-2+1/rho} + S^{-2+2/rho}] w2^{rho-1} / (u1*u2)
Log form: -S^{1/rho} + (rho-1)(log w1+log w2) + log[(rho-1)S^{e1}+S^{e2}] - log u1 - log u2
where e1=-2+1/rho, e2=-2+2/rho.
Uses a two-pass approach: first S into adj, then final log-density.
"""
function copula_log_adjustment!(::GumbelCopula_MCI, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c)
    # Step 1: Compute F_v into Fv_buf (in-place, no allocation)
    noise_cdf!(noise, Fv_buf, z_buffer, noise_vals, c)

    # Step 2: Compute log copula density via two-pass approach
    rho = copula_vals.rho
    rho_m1 = rho - 1
    inv_rho = 1 / rho
    e1 = -2 + inv_rho       # exponent for term1
    e2 = -2 + 2 * inv_rho   # exponent for term2
    lo = c.copula_clamp_lo
    hi = c.copula_clamp_hi
    sf = c.gumbel_sum_floor

    # Clamp CDFs in-place (Fv_buf already holds F_v)
    @. Fv_buf = clamp(Fv_buf, lo, hi)

    # Pass 1: Compute S = w1^rho + w2^rho into adj (workspace)
    @. adj = max((-log(Fv_buf))^rho + (-log(clamp(draws, lo, hi)))^rho, sf)

    # Pass 2: Compute log copula density. adj holds S, Fv_buf holds clamped F_v.
    # Element-wise broadcast reads adj (=S) before overwriting -- safe with @.
    @. adj = -adj^inv_rho +
             rho_m1 * (log(-log(Fv_buf)) + log(-log(clamp(draws, lo, hi)))) +
             log(rho_m1 * adj^e1 + adj^e2) -
             log(Fv_buf) - log(clamp(draws, lo, hi))
    return nothing
end

"""
    copula_log_adjustment!(::Clayton90Copula_MCI, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c)

90° rotated Clayton copula: c^{90°}(z_v, z_u) = c^{Clayton}(1 - F_v(v), F_u(u)).
Uses F_v(-v) instead of 1 - F_v(v) for numerical precision.
"""
function copula_log_adjustment!(::Clayton90Copula_MCI, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c)
    # Step 1: Compute F_v(-v) = 1 - F_v(v) via negation (better precision)
    @. adj = -z_buffer
    noise_cdf!(noise, Fv_buf, adj, noise_vals, c)

    # Step 2: Clayton density with rotated first argument
    rho = copula_vals.rho
    rho_p1 = rho + 1
    neg_inv_rho_m2 = -1 / rho - 2
    lo = c.copula_clamp_lo
    hi = c.copula_clamp_hi

    @. adj = log(rho_p1) - rho_p1 * (
                log(clamp(Fv_buf, lo, hi)) + log(clamp(draws, lo, hi))
             ) + neg_inv_rho_m2 * log(max(
                clamp(Fv_buf, lo, hi)^(-rho) + clamp(draws, lo, hi)^(-rho) - 1, lo
             ))
    return nothing
end

# ============================================================================
# Include unified MCI T-approach for ALL distributions (GPU + ForwardDiff compatible)
# ============================================================================
include("sf_MCI_T_v21.jl")

# ============================================================================
# Section 8: CPU Likelihood Function
# ============================================================================

"""
    MCI_nll(Y, X, Z, p, draws; noise, ineff, hetero=Symbol[], constants=nothing)

Compute negative log-likelihood for stochastic frontier model using MCI integration.
CPU version - processes all draws at once without chunking.

# Positional Arguments
- `Y`: Response vector (N observations)
- `X`: Design matrix (N x K) - include a column of ones for intercept
- `Z`: Covariate matrix (N x L) for heteroscedastic parameters
- `p`: Parameter vector (structure depends on model, see `plen`)
- `draws`: Pre-generated Halton probabilities as a **1×D matrix** (row vector).
  Use `draws = reshape(make_halton_p(n), 1, n)` to convert from vector.

# Keyword Arguments
- `noise::Symbol`: Noise distribution
  - `:Normal` - Normal noise v ~ N(0, σ_v2)
  - `:StudentT` - Student-t noise with scale σ_v and degrees of freedom ν > 2
  - `:Laplace` - Laplace noise v ~ Laplace(0, b)

- `ineff::Symbol`: Inefficiency distribution
  - `:TruncatedNormal` - Truncated Normal u ~ TN(μ, σ_u; lower=0)
  - `:Exponential` - Exponential u ~ Exp(λ), λ = Var(u)
  - `:HalfNormal` - Half Normal u ~ |N(0, σ2)|
  - `:Weibull` - Weibull u ~ Weibull(λ, k)
  - `:Lognormal` - Lognormal u ~ LogNormal(μ, σ)
  - `:Lomax` - Lomax u ~ Lomax(α, λ)
  - `:Rayleigh` - Rayleigh u ~ Rayleigh(σ)

- `hetero`: Vector of symbols controlling heterogeneity:
  - `:TruncatedNormal` × `[:mu]`, `[:sigma_sq]`, or both
  - `:Exponential` × `[:lambda]`
  - `:HalfNormal` × `[:sigma_sq]`
  - `:Weibull` × `[:lambda]`, `[:k]`, or both
  - `:Lognormal` × `[:mu]`, `[:sigma_sq]`, or both
  - `:Lomax` × `[:lambda]`, `[:alpha]`, or both
  - `:Rayleigh` × `[:sigma_sq]`

- `constants`: Precomputed constants from `make_constants` (default: computed fresh)

# Returns
- Scalar negative log-likelihood value

# Examples
```julia
# Create draws as 1×D matrix (required format)
halton = reshape(make_halton_p(1023), 1, 1023)

# Normal noise + Truncated Normal inefficiency
nll = MCI_nll(Y, X, Z, p, halton; noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])

# Laplace noise + Weibull inefficiency
nll = MCI_nll(Y, X, Z, p, halton; noise=:Laplace, ineff=:Weibull, hetero=[:lambda, :k])

# StudentT noise + Lognormal inefficiency
nll = MCI_nll(Y, X, Z, p, halton; noise=:StudentT, ineff=:Lognormal, hetero=[:mu, :sigma_sq])
```
"""
function MCI_nll(Y::AbstractVector{T}, X::AbstractMatrix{T}, Z::AbstractMatrix{T},
                 p::AbstractVector{P}, draws::AbstractMatrix{T};
                 noise::Symbol, ineff::Symbol,
                 copula::Symbol=:None,
                 hetero::Vector{Symbol}=Symbol[],
                 chunks::Int=4,
                 constants=nothing,
                 type::Symbol=:prod) where {T<:AbstractFloat, P<:Real}

    # Dimensions
    n_obs = length(Y)

    # Validate draws shape: 1×D (shared) or N×D (multiRand)
    if size(draws, 1) != 1 && size(draws, 1) != n_obs
        error("Invalid `draws` shape: expected 1×D (shared) or N×D (multiRand) matrix, got $(size(draws)). " *
              "For shared draws use reshape(draws, 1, :), for multiRand use N=$n_obs rows.")
    end

    # Build model from symbols (validates inputs)
    model = _build_model(noise, ineff; copula=copula)

    # Validate hetero options
    _validate_hetero(model.ineff, hetero)

    # Compute sign for frontier type
    frontier_sign = if type in (:prod, :production)
        1
    elseif type == :cost
        -1
    else
        error("Invalid `type`: $type. Use :prod, :production, or :cost.")
    end

    # Dimensions (n_obs already computed above for validation)
    D = size(draws, 2)
    K = size(X, 2)
    L = size(Z, 2)

    # Constants
    c = isnothing(constants) ? make_constants(model, T) : constants

    # Get parameter indices (not views - enables CPU parameter access)
    idx = _param_ind(model, K, L, hetero)

    # Compute residuals using broadcasting loop (SFGPU pattern)
    # P(p[j]) extracts CPU scalar, broadcasts against data array
    ε = Y .- sum(P(p[idx.beta[j]]) .* (@view X[:, j]) for j in 1:K)

    # Compute noise values via dispatch (new signature: p, idx, c)
    noise_vals = get_noise_vals(model.noise, p, idx, c)

    # Compute inefficiency values via dispatch (new signature: p, idx, Z, c)
    ineff_vals = get_ineff_vals(model.ineff, p, idx, Z, c)

    # Compute copula values via dispatch
    copula_vals = get_copula_vals(model.copula, p, idx, c)

    # draws is already a 1×D matrix
    draws_1D = draws

    # ========================================================================
    # MCI T-approach for ALL inefficiency distributions (GPU + ForwardDiff compatible)
    # ========================================================================
    # All distributions now use the T-approach instead of inverse CDF quantiles.
    # This provides GPU compatibility and ForwardDiff support via element-wise operations.

    # Get transformation functions for this distribution (type dispatch)
    trans_rule = default_transformation_rule(model.ineff)
    trans, jacob = resolve_transformation(trans_rule)

    # Use unified MCI T-approach for ALL distributions
    return MCI_nll_mci_T(ε, draws_1D, model, c, n_obs, D,
                         frontier_sign, chunks, noise_vals, ineff_vals;
                         trans=trans, jacob=jacob,
                         copula_vals=copula_vals)
end

# REMOVED_START: Old per-distribution dispatch
#= elseif model.ineff isa Exponential_MCI
        return MCI_nll_exp_T(ε, draws_1D, model, c, n_obs, D,
                              frontier_sign, chunks, noise_vals, ineff_vals)
    elseif model.ineff isa HalfNormal_MCI
        return MCI_nll_halfnorm_T(ε, draws_1D, model, c, n_obs, D,
                                   frontier_sign, chunks, noise_vals, ineff_vals)
    elseif model.ineff isa TruncatedNormal_MCI
        return MCI_nll_truncnorm_T(ε, draws_1D, model, c, n_obs, D,
                                    frontier_sign, chunks, noise_vals, ineff_vals)
    elseif model.ineff isa Weibull_MCI
        return MCI_nll_weibull_T(ε, draws_1D, model, c, n_obs, D,
                                  frontier_sign, chunks, noise_vals, ineff_vals)
    elseif model.ineff isa Lognormal_MCI
        return MCI_nll_lognorm_T(ε, draws_1D, model, c, n_obs, D,
                                  frontier_sign, chunks, noise_vals, ineff_vals)
    elseif model.ineff isa Lomax_MCI
        return MCI_nll_pareto_T(ε, draws_1D, model, c, n_obs, D,
                                 frontier_sign, chunks, noise_vals, ineff_vals)
    elseif model.ineff isa Rayleigh_MCI
        return MCI_nll_rayleigh_T(ε, draws_1D, model, c, n_obs, D,
                                   frontier_sign, chunks, noise_vals, ineff_vals)
    else
        error("Unknown inefficiency model: $(typeof(model.ineff))")
    end
=#

"""
    MCI_nll(spec::sfmodel_MCI_spec, p::AbstractVector; chunks::Int=4)

Compute negative log-likelihood using a model specification.
This is the simplified interface that extracts all configuration from the spec.

# Arguments
- `spec::sfmodel_MCI_spec`: Model specification containing data and model config
- `p::AbstractVector`: Parameter vector

# Keyword Arguments
- `chunks::Int=4`: Number of chunks for memory management (default: 1)

# Example
```julia
spec = sfmodel_MCI_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])

# Simple call - no need to repeat model configuration
nll_value = MCI_nll(spec, p)

# For optimization
nll = p -> MCI_nll(spec, p)
result = optimize(nll, p0, Newton(); autodiff=AutoForwardDiff())
```
"""
function MCI_nll(spec::sfmodel_MCI_spec{T}, p::AbstractVector{P};
                 chunks::Int=4) where {T<:AbstractFloat, P<:Real}

    # Extract from spec
    Y, X, Z = spec.depvar, spec.frontier, spec.zvar
    c = spec.constants
    idx = spec.idx
    model = spec.model
    n_obs, K = spec.N, spec.K
    D = size(spec.draws_2D, 2)

    # Compute residuals using broadcasting loop
    ε = Y .- sum(P(p[idx.beta[j]]) .* (@view X[:, j]) for j in 1:K)

    # Compute noise values via dispatch
    noise_vals = get_noise_vals(model.noise, p, idx, c)

    # Compute inefficiency values via dispatch
    ineff_vals = get_ineff_vals(model.ineff, p, idx, Z, c)

    # Compute copula values via dispatch
    copula_vals = get_copula_vals(model.copula, p, idx, c)

    # Compute scaling h(z) = exp(z'δ) if scaling property model
    scaling_h = if spec.scaling
        Z_s = spec.scaling_zvar; L_s = spec.L_scaling
        exp.(sum(P(p[idx.delta[j]]) .* (@view Z_s[:, j]) for j in 1:L_s))
    else
        nothing
    end

    # Use pre-reshaped draws from spec
    draws_1D = spec.draws_2D

    # Resolve transformation rule from spec
    trans_func, jacob_func = resolve_transformation(spec.transformation)

    # ========================================================================
    # MCI T-approach for ALL inefficiency distributions (GPU + ForwardDiff compatible)
    # ========================================================================
    # Single unified call handles all distributions via type dispatch.
    # The MCI_nll_mci_T function dispatches on model.ineff type for:
    #   - get_scale_param(): extract appropriate scale parameter
    #   - log_pdf_ineff!(): compute distribution-specific log-PDF

    return MCI_nll_mci_T(ε, draws_1D, model, c, n_obs, D,
                          spec.sign, chunks, noise_vals, ineff_vals;
                          trans=trans_func, jacob=jacob_func,
                          copula_vals=copula_vals,
                          scaling_h=scaling_h)
end

# ============================================================================
# Section 9: GPU Support (CUDA.jl integration)
# ============================================================================
#=
GPU Support Notes:

The unified MCI_nll function now works with both CPU Arrays and GPU CuArrays
via the SFGPU-style broadcasting pattern. Key features:

1. Parameters stay on CPU: Pass p as a regular Vector (not CuArray)
2. Data can be on GPU: Y, X, Z, draws can be CuArray
3. Broadcasting handles CPU scalar × CuArray automatically
4. ForwardDiff works: Type P flows from eltype(p), loggamma on CPU scalars

Example with GPU data:
```julia
using CUDA

Y_gpu = CuArray(Y)
X_gpu = CuArray(X)
Z_gpu = CuArray(Z)
draws_gpu = CuArray(halton)

# Note: p stays as CPU Vector, NOT converted to CuArray
nll = MCI_nll(Y_gpu, X_gpu, Z_gpu, p, draws_gpu;
              noise=:StudentT, ineff=:TruncatedNormal, hetero=[:mu])

# With ForwardDiff optimization:
using Optim, ForwardDiff

result = optimize(
    theta -> MCI_nll(Y_gpu, X_gpu, Z_gpu, theta, draws_gpu;
                     noise=:StudentT, ineff=:TruncatedNormal, hetero=[:mu]),
    p0, Newton(); autodiff=AutoForwardDiff()
)
```

The old pattern `theta |> CuArray` is no longer needed.
=#

# ============================================================================
# Section 10: Efficiency Indices (JLMS and BC)
# ============================================================================

"""
    jlms_bc_indices(Y, X, Z, p, draws; noise, ineff, hetero, chunks, constants)

Compute JLMS (inefficiency index) and BC (efficiency index) for each observation
using MCI integration.

# Mathematical Definitions
Given the composed error ε = v - u, where v is noise and u ? 0 is inefficiency:

- **JLMS (Jondrow et al. 1982):** E(u|ε) = [× u�Pf_v(ε+u)�Pf_u(u) du] / [× f_v(ε+u)�Pf_u(u) du]
- **BC (Battese & Coelli 1988):** E(e^{-u}|ε) = [× e^{-u}�Pf_v(ε+u)�Pf_u(u) du] / [× f_v(ε+u)�Pf_u(u) du]

# Arguments
Same as `MCI_nll`:
- `Y`: Response vector (N observations)
- `X`: Design matrix (N x K)
- `Z`: Covariate matrix (N x L) for heteroscedastic parameters
- `p`: Parameter vector
- `draws`: Pre-generated Halton probabilities as a **1×D matrix** (row vector).
  Use `draws = reshape(make_halton_p(n), 1, n)` to convert from vector.
- `noise::Symbol`: Noise distribution (:Normal, :StudentT, :Laplace)
- `ineff::Symbol`: Inefficiency distribution
- `hetero::Vector{Symbol}`: Heteroscedasticity options (default: empty)
- `chunks::Int`: Number of chunks for memory management (default: 1)
- `constants`: Precomputed constants from `make_constants` (default: computed fresh)

# Returns
NamedTuple with fields:
- `jlms::Vector`: E(u|ε) for each observation (inefficiency index)
- `bc::Vector`: E(e^{-u}|ε) for each observation (efficiency index)
- `likelihood::Vector`: f_ε(ε) for each observation (density value)

# Examples
```julia
# Create draws as 1×D matrix (required format)
halton = reshape(make_halton_p(1023), 1, 1023)

# After estimation, compute efficiency indices
result = jlms_bc_indices(Y, X, Z, p_hat, halton;
    noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])

# Technical efficiency = result.bc
# Technical inefficiency = result.jlms
# Log-likelihood = sum(log.(result.likelihood))
```
"""
function jlms_bc_indices(Y::AbstractVector{T}, X::AbstractMatrix{T}, Z::AbstractMatrix{T},
                                    p::AbstractVector{P}, draws::AbstractMatrix{T};
                                    noise::Symbol, ineff::Symbol,
                                    copula::Symbol=:None,
                                    hetero::Vector{Symbol}=Symbol[],
                                    chunks::Int=1,
                                    constants=nothing,
                                    type::Symbol=:prod) where {T<:AbstractFloat, P<:Real}

    # Dimensions
    n_obs = length(Y)

    # Validate draws shape: 1×D (shared) or N×D (multiRand)
    if size(draws, 1) != 1 && size(draws, 1) != n_obs
        error("Invalid `draws` shape: expected 1×D (shared) or N×D (multiRand) matrix, got $(size(draws)). " *
              "For shared draws use reshape(draws, 1, :), for multiRand use N=$n_obs rows.")
    end

    # Build model from symbols (validates inputs)
    model = _build_model(noise, ineff; copula=copula)

    # Validate hetero options
    _validate_hetero(model.ineff, hetero)

    # Compute sign for frontier type
    frontier_sign = if type in (:prod, :production)
        1
    elseif type == :cost
        -1
    else
        error("Invalid `type`: $type. Use :prod, :production, or :cost.")
    end

    # Dimensions (n_obs already computed above for validation)
    D = size(draws, 2)
    K = size(X, 2)
    L = size(Z, 2)

    # Constants
    c = isnothing(constants) ? make_constants(model, T) : constants

    # Get parameter indices
    idx = _param_ind(model, K, L, hetero)

    # Compute residuals: ε = Y - Xβ
    ε = Y .- sum(P(p[idx.beta[j]]) .* (@view X[:, j]) for j in 1:K)

    # Compute noise parameters via dispatch
    noise_vals = get_noise_vals(model.noise, p, idx, c)

    # Compute inefficiency parameters via dispatch
    ineff_vals = get_ineff_vals(model.ineff, p, idx, Z, c)

    # Compute copula values via dispatch
    copula_vals = get_copula_vals(model.copula, p, idx, c)
    has_copula = copula_plen(model.copula) > 0

    # draws is already a 1×D matrix
    draws_1D = draws

    if has_copula
        # ================================================================
        # Copula active: use T-approach (F_u(u) = t naturally available)
        # ================================================================
        trans_rule = default_transformation_rule(model.ineff)
        trans_func, jacob_func = resolve_transformation(trans_rule)
        return _jlms_bc_mci_T(ε, draws_1D, model, c, n_obs, D,
                               frontier_sign, chunks, noise_vals, ineff_vals;
                               trans=trans_func, jacob=jacob_func,
                               copula_vals=copula_vals)
    elseif chunks == 1
        # ================================================================
        # Non-chunked path: process all observations at once
        # ================================================================

        # Step 1: Generate u samples × N × D matrix
        u_buffer = similar(ε, P, n_obs, D)
        get_u_quantile!(model.ineff, u_buffer, ineff_vals, draws_1D, c, n_obs)

        # Step 2: Compute z = ε + sign*u
        ε_N1 = reshape(ε, n_obs, 1)
        pdf_buffer = similar(u_buffer)
        @. pdf_buffer = ε_N1 + frontier_sign * u_buffer

        # Step 3: Compute f_v(z) in-place
        get_noise_pdf!(model.noise, pdf_buffer, pdf_buffer, noise_vals, c)

        # Step 4: Compute indices via weighted averages
        # Use sum/D instead of mean to avoid scalar indexing issues on GPU
        # Denominator: E[f_v(ε+u)] = f_ε(ε)
        likelihood = vec(sum(pdf_buffer, dims=2)) ./ P(D)

        # JLMS numerator: E[u �P f_v(ε+u)]
        jlms_num = vec(sum(u_buffer .* pdf_buffer, dims=2)) ./ P(D)

        # BC numerator: E[e^{-u} �P f_v(ε+u)]
        bc_num = vec(sum(exp.(-u_buffer) .* pdf_buffer, dims=2)) ./ P(D)

        # Avoid division by zero
        safe_denom = max.(likelihood, floatmin(T))

        jlms = jlms_num ./ safe_denom
        bc = bc_num ./ safe_denom

        return (jlms=jlms, bc=bc, likelihood=likelihood)
    else
        # ================================================================
        # Chunked path: split observations for memory management
        # ================================================================
        chunk_size = cld(n_obs, chunks)

        # Pre-allocate output vectors
        jlms_out = similar(ε, P, n_obs)
        bc_out = similar(ε, P, n_obs)
        likelihood_out = similar(ε, P, n_obs)

        # Pre-allocate chunk buffers
        u_buffer = similar(ε, P, chunk_size, D)
        pdf_buffer = similar(ε, P, chunk_size, D)

        for chunk_idx in 1:chunks
            row_start = (chunk_idx - 1) * chunk_size + 1
            row_end = min(chunk_idx * chunk_size, n_obs)

            # Skip empty chunks
            row_start > row_end && continue

            chunk_N = row_end - row_start + 1
            u_buf = @view u_buffer[1:chunk_N, :]
            pdf_buf = @view pdf_buffer[1:chunk_N, :]

            # Step 1: Generate u samples for this chunk
            get_u_quantile_chunk!(model.ineff, u_buf, ineff_vals, draws_1D, c,
                                      row_start, row_end, chunk_N)

            # Step 2: Compute z = ε + sign*u
            ε_chunk = reshape((@view ε[row_start:row_end]), chunk_N, 1)
            @. pdf_buf = ε_chunk + frontier_sign * u_buf

            # Step 3: Compute f_v(z) in-place
            get_noise_pdf!(model.noise, pdf_buf, pdf_buf, noise_vals, c)

            # Step 4: Compute indices for this chunk
            # Use sum/D instead of mean to avoid scalar indexing issues
            chunk_likes = vec(sum(pdf_buf, dims=2)) ./ P(D)
            chunk_jlms_num = vec(sum(u_buf .* pdf_buf, dims=2)) ./ P(D)
            chunk_bc_num = vec(sum(exp.(-u_buf) .* pdf_buf, dims=2)) ./ P(D)

            safe_denom = max.(chunk_likes, floatmin(T))

            # Store results
            likelihood_out[row_start:row_end] .= chunk_likes
            jlms_out[row_start:row_end] .= chunk_jlms_num ./ safe_denom
            bc_out[row_start:row_end] .= chunk_bc_num ./ safe_denom
        end

        return (jlms=jlms_out, bc=bc_out, likelihood=likelihood_out)
    end
end

"""
    jlms_bc_indices(spec::sfmodel_MCI_spec, p::AbstractVector; chunks::Int=1)

Compute JLMS and BC efficiency indices using a model specification.
This is the simplified interface that extracts all configuration from the spec.

# Arguments
- `spec::sfmodel_MCI_spec`: Model specification containing data and model config
- `p::AbstractVector`: Parameter vector

# Keyword Arguments
- `chunks::Int=1`: Number of chunks for memory management (default: 1)

# Returns
NamedTuple with fields:
- `jlms::Vector`: E(u|ε) for each observation (inefficiency index)
- `bc::Vector`: E(e^{-u}|ε) for each observation (efficiency index)
- `likelihood::Vector`: f_ε(ε) for each observation (density value)

# Example
```julia
spec = sfmodel_MCI_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])
result = jlms_bc_indices(spec, p_hat)

# Access results
println("Mean efficiency: ", mean(result.bc))
println("Mean inefficiency: ", mean(result.jlms))
```
"""
function jlms_bc_indices(spec::sfmodel_MCI_spec{T}, p::AbstractVector{P};
                         chunks::Int=1) where {T<:AbstractFloat, P<:Real}

    # Extract from spec
    Y, X, Z = spec.depvar, spec.frontier, spec.zvar
    c = spec.constants
    idx = spec.idx
    model = spec.model
    n_obs, K = spec.N, spec.K
    D = size(spec.draws_2D, 2)

    # Compute residuals: ε = Y - Xβ
    ε = Y .- sum(P(p[idx.beta[j]]) .* (@view X[:, j]) for j in 1:K)

    # Compute noise parameters via dispatch
    noise_vals = get_noise_vals(model.noise, p, idx, c)

    # Compute inefficiency parameters via dispatch
    ineff_vals = get_ineff_vals(model.ineff, p, idx, Z, c)

    # Compute copula values via dispatch
    copula_vals = get_copula_vals(model.copula, p, idx, c)

    # Compute scaling h(z) = exp(z'δ) if scaling property model
    scaling_h = if spec.scaling
        Z_s = spec.scaling_zvar; L_s = spec.L_scaling
        exp.(sum(P(p[idx.delta[j]]) .* (@view Z_s[:, j]) for j in 1:L_s))
    else
        nothing
    end

    # Use pre-reshaped draws from spec
    draws_1D = spec.draws_2D

    # Resolve transformation rule from spec
    trans_func, jacob_func = resolve_transformation(spec.transformation)

    # ========================================================================
    # MCI T-approach for ALL inefficiency distributions (GPU + ForwardDiff compatible)
    # ========================================================================
    # Single unified call handles all distributions via type dispatch.
    # The _jlms_bc_mci_T function dispatches on model.ineff type for:
    #   - get_scale_param(): extract appropriate scale parameter
    #   - log_pdf_ineff!(): compute distribution-specific log-PDF

    return _jlms_bc_mci_T(ε, draws_1D, model, c, n_obs, D,
                           spec.sign, chunks, noise_vals, ineff_vals;
                           trans=trans_func, jacob=jacob_func,
                           copula_vals=copula_vals,
                           scaling_h=scaling_h)
end

# ============================================================================
# Section 11: Post-Estimation Utilities
# ============================================================================

"""
    _invert_hessian(H::AbstractMatrix{T}; message::Bool=true) where {T<:Real}

Internal helper for Hessian inversion with error handling.

# Returns
NamedTuple: `(var_cov_matrix=Matrix, redflag=Int)`
- `redflag = 0`: Success
- `redflag = 1`: Hessian not invertible
- `redflag = 2`: Some diagonal elements are non-positive
"""
function _invert_hessian(H::AbstractMatrix{T};
                         message::Bool=true) where {T<:Real}
    redflag = 0
    var_cov_matrix = similar(H)

    # Attempt to invert Hessian
    try
        var_cov_matrix = inv(H)
    catch err
        redflag = 1
        if message
            printstyled("The Hessian matrix is not invertible, indicating the model does not converge properly.\n"; color=:red)
        end
        return (var_cov_matrix=var_cov_matrix, redflag=redflag)
    end

    # Check for non-positive diagonal elements
    if !all(diag(var_cov_matrix) .> 0)
        redflag = 2
        if message
            printstyled("Some diagonal elements of the var-cov matrix are non-positive, indicating convergence problems.\n"; color=:red)
        end
    end

    return (var_cov_matrix=var_cov_matrix, redflag=redflag)
end

"""
    var_cov_mat(nll_func::Function, coef::AbstractVector{T}; message::Bool=true) where {T<:Real}

Compute variance-covariance matrix from negative log-likelihood function.

# Arguments
- `nll_func`: A closure of the negative log-likelihood function that takes only `p` (coefficient vector)
- `coef`: The optimized coefficient vector from `Optim.minimizer(result)`
- `message`: Whether to print error messages (default: true)

# Returns
NamedTuple: `(var_cov_matrix=Matrix, redflag=Int)`
- `redflag = 0`: Success
- `redflag = 1`: Hessian not invertible
- `redflag = 2`: Some diagonal elements are non-positive

# Example
```julia
nll = p -> MCI_nll(Y, X, Z, p, halton; noise=:Normal, ineff=:TruncatedNormal)
result = optimize(nll, p0, Newton(); autodiff=AutoForwardDiff())
vcov = var_cov_mat(nll, Optim.minimizer(result))
```
"""
function var_cov_mat(nll_func::Function, coef::AbstractVector{T};
                     message::Bool=true) where {T<:Real}
    H = hessian(nll_func, coef)
    return _invert_hessian(H; message=message)
end

"""
    var_cov_mat(H::AbstractMatrix{T}; message::Bool=true) where {T<:Real}

Compute variance-covariance matrix from pre-computed Hessian matrix.
Useful for GPU workflows where Hessian is computed separately.

# Arguments
- `H`: Pre-computed Hessian matrix
- `message`: Whether to print error messages (default: true)

# Returns
NamedTuple: `(var_cov_matrix=Matrix, redflag=Int)`

# Example
```julia
H = ForwardDiff.hessian(nll, coef)
vcov = var_cov_mat(H)
```
"""
function var_cov_mat(H::AbstractMatrix{T};
                     message::Bool=true) where {T<:Real}
    return _invert_hessian(H; message=message)
end

"""
    _detect_log_params(model::MCIModel, idx, L::Int, hetero::Vector{Symbol})

Auto-detect log-transformed parameters based on model type for auxiliary table.
Only returns parameters that are scalar (not heteroscedastic).

# Returns
Vector of tuples: `[(display_name, param_index), ...]`
"""
function _detect_log_params(model::MCIModel, idx, L::Int, hetero::Vector{Symbol})
    entries = Tuple{String, Int}[]

    # Noise parameters (always scalar)
    if hasfield(typeof(idx.noise), :ln_sigma_v_sq)
        push!(entries, ("σᵥ²", idx.noise.ln_sigma_v_sq))
    elseif hasfield(typeof(idx.noise), :ln_b)
        push!(entries, ("b", idx.noise.ln_b))
    end

    # Inefficiency parameters (only if scalar - not heteroscedastic)
    ineff = idx.ineff
    if hasfield(typeof(ineff), :sigma_u) && length(ineff.sigma_u) == 1
        push!(entries, ("σᵤ²", first(ineff.sigma_u)))
    elseif hasfield(typeof(ineff), :sigma_sq) && length(ineff.sigma_sq) == 1
        push!(entries, ("σᵤ²", first(ineff.sigma_sq)))
    elseif hasfield(typeof(ineff), :lambda) && length(ineff.lambda) == 1
        push!(entries, ("λ", first(ineff.lambda)))
    elseif hasfield(typeof(ineff), :theta)  # Gamma-specific (Weibull has :k but not :theta)
        if length(ineff.k) == 1
            push!(entries, ("k", first(ineff.k)))
        end
        if length(ineff.theta) == 1
            push!(entries, ("θ", first(ineff.theta)))
        end
    end

    return entries
end

"""
    _detect_copula_params(model::MCIModel, idx, coef, stddev)

Auto-detect copula parameters for auxiliary table.
Returns a NamedTuple with copula-specific statistics.
"""
function _detect_copula_params(model::MCIModel, idx, coef, stddev)
    if model.copula isa NoCopula_MCI
        return NamedTuple()
    elseif model.copula isa GaussianCopula_MCI
        rho_max = 0.999
        theta_rho = coef[idx.copula.theta_rho]
        se_theta = stddev[idx.copula.theta_rho]
        rho = rho_max * tanh(theta_rho)
        se_rho = rho_max * (1 - tanh(theta_rho)^2) * se_theta  # Delta method
        kendalls_tau = (2/π) * asin(rho)
        return (rho=rho, se_rho=se_rho, kendalls_tau=kendalls_tau,
                tail_dep=0.0, theta_rho=theta_rho, se_theta=se_theta)
    elseif model.copula isa ClaytonCopula_MCI
        theta_rho = coef[idx.copula.theta_rho]
        se_theta = stddev[idx.copula.theta_rho]
        rho = clamp(exp(theta_rho) + 1e-6, 1e-6, 50.0)  # Match get_copula_vals clamp
        se_rho = exp(theta_rho) * se_theta    # Delta method: d/d_theta (exp(theta)+c) = exp(theta)
        kendalls_tau = rho / (2 + rho)
        tail_dep = 2^(-1/rho)                # Lower tail dependence
        return (rho=rho, se_rho=se_rho, kendalls_tau=kendalls_tau,
                tail_dep=tail_dep, theta_rho=theta_rho, se_theta=se_theta)
    elseif model.copula isa Clayton90Copula_MCI
        theta_rho = coef[idx.copula.theta_rho]
        se_theta = stddev[idx.copula.theta_rho]
        rho = clamp(exp(theta_rho) + 1e-6, 1e-6, 50.0)  # Match get_copula_vals clamp
        se_rho = exp(theta_rho) * se_theta
        kendalls_tau = -rho / (2 + rho)       # Negated for 90° rotation
        tail_dep = 2^(-1/rho)                # Upper-lower tail dependence λ_UL
        return (rho=rho, se_rho=se_rho, kendalls_tau=kendalls_tau,
                tail_dep=tail_dep, theta_rho=theta_rho, se_theta=se_theta)
    elseif model.copula isa GumbelCopula_MCI
        theta_rho = coef[idx.copula.theta_rho]
        se_theta = stddev[idx.copula.theta_rho]
        rho = clamp(exp(theta_rho) + 1, 1.0, 50.0)  # Match get_copula_vals clamp
        se_rho = exp(theta_rho) * se_theta    # Delta method: d/d_theta (exp(theta)+1) = exp(theta)
        kendalls_tau = 1 - 1/rho
        tail_dep = 2 - 2^(1/rho)             # Upper tail dependence
        return (rho=rho, se_rho=se_rho, kendalls_tau=kendalls_tau,
                tail_dep=tail_dep, theta_rho=theta_rho, se_theta=se_theta)
    else
        return NamedTuple()
    end
end

"""
    print_table(coef, var_cov_matrix, varnames, eqnames, eq_indices; kwargs...)

Print formatted estimation results table with statistics.

# Positional Arguments
- `coef::AbstractVector{T}`: Coefficient vector
- `var_cov_matrix::AbstractMatrix{T}`: Variance-covariance matrix from `var_cov_mat`
- `varnames::Vector{String}`: Variable names for each coefficient
- `eqnames::Vector{String}`: Equation/block names (e.g., `["frontier", "μ", "ln_σᵤ²", "ln_σᵥ²"]`)
- `eq_indices::Vector{Int}`: Starting index for each equation block

# Keyword Arguments
- `nobs::Int`: Number of observations
- `noise::Symbol`: Noise distribution symbol
- `ineff::Symbol`: Inefficiency distribution symbol
- `K::Int`: Number of frontier regressors (size(X, 2))
- `L::Int`: Number of Z columns (size(Z, 2))
- `hetero::Vector{Symbol}=Symbol[]`: Heteroscedasticity options
- `n_draws::Int=0`: Number of simulation draws per observation
- `sign::Int=1`: Frontier type (1=production, otherwise=cost)
- `optim_result=nothing`: Optional Optim.jl result for convergence info
- `table_format::Symbol=:text`: Output format (`:text`, `:html`, or `:latex`)

# Returns
NamedTuple: `(table=Matrix, aux_table=Matrix)`

# Example
```julia
print_table(coef, vcov.var_cov_matrix,
            ["_cons", "x1", "x2", "_cons", "_cons", "_cons"],
            ["frontier", "μ", "ln_σᵤ²", "ln_σᵥ²"],
            [1, 4, 5, 6];
            nobs=N, noise=:Normal, ineff=:TruncatedNormal,
            K=3, L=1, optim_result=result)
```
"""
function print_table(coef::AbstractVector{T}, var_cov_matrix::AbstractMatrix{T},
                     varnames::Vector{String}, eqnames::Vector{String},
                     eq_indices::Vector{Int};
                     nobs::Int,
                     noise::Symbol,
                     ineff::Symbol,
                     copula::Symbol=:None,
                     K::Int,
                     L::Int,
                     hetero::Vector{Symbol}=Symbol[],
                     L_scaling::Int=0,
                     n_draws::Int=0,
                     sign::Int=1,
                     GPU::Bool=false,
                     optim_result=nothing,
                     table_format::Symbol=:text) where {T<:Real}

    nofpara = length(coef)
    model = _build_model(noise, ineff; copula=copula)
    idx = _param_ind(model, K, L, hetero; L_scaling=L_scaling)

    # Compute statistics
    stddev = sqrt.(diag(var_cov_matrix))
    t_stats = coef ./ stddev
    tt = cquantile(Normal(0, 1), 0.025)  # ~1.96

    p_values = [2 * ccdf(TDist(nobs - nofpara), abs(t_stats[i])) for i in 1:nofpara]
    ci_low = coef .- tt .* stddev
    ci_upp = coef .+ tt .* stddev

    # Build equation column
    eq_col = fill("", nofpara)
    for (i, idx_val) in enumerate(eq_indices)
        if idx_val <= nofpara
            eq_col[idx_val] = eqnames[i]
        end
    end

    # Print header
    printstyled("*********************************\n"; color=:cyan)
    printstyled("      Estimation Results\n"; color=:cyan)
    printstyled("*********************************\n"; color=:cyan)

    print("Method: "); printstyled("MCI"; color=:yellow); println()
    if copula == :None
        print("Model type: "); printstyled("noise=$noise, ineff=$ineff"; color=:yellow); println()
    else
        print("Model type: "); printstyled("noise=$noise, ineff=$ineff, copula=$copula"; color=:yellow); println()
    end
    if !isempty(hetero)
        print("Heteroscedastic parameters: "); printstyled(hetero; color=:yellow); println()
    else
        println("Homoscedastic model (no heteroscedasticity)")
    end
    print("Number of observations: "); printstyled(nobs; color=:yellow); println()
    print("Number of frontier regressors (K): "); printstyled(K; color=:yellow); println()
    print("Number of Z columns (L): "); printstyled(L; color=:yellow); println()
    if n_draws > 0
        print("Number of draws: "); printstyled(n_draws; color=:yellow); println()
    end
    print("Frontier type: "); printstyled(sign == 1 ? "production" : "cost"; color=:yellow); println()
    print("GPU computing: "); printstyled(GPU; color=:yellow); println()

    if optim_result !== nothing
        print("Number of iterations: "); printstyled(optim_result.iterations; color=:yellow); println()
        converged = Optim.converged(optim_result)
        print("Converged: "); printstyled(converged; color=converged ? :yellow : :red); println()
        print("Log-likelihood: "); printstyled(round(-optim_result.minimum; digits=5); color=:yellow); println()
    end
    println()

    # Build table data
    table_data = hcat(eq_col, varnames, coef, stddev, t_stats, p_values, ci_low, ci_upp)

    # Main table with PrettyTables
    pretty_table(table_data;
                 column_labels=["", "Var.", "Coef.", "Std.Err.", "z", "P>|z|", "95%CI_l", "95%CI_u"],
                 formatters=[fmt__printf("%.4f", collect(3:8))],
                 compact_printing=true,
                 backend=table_format)
    println()

    # Auto-detect log-transformed parameters for auxiliary table
    aux_entries = _detect_log_params(model, idx, L, hetero)
    aux_table = Matrix{Any}(undef, 0, 3)

    if !isempty(aux_entries)
        println("Log-parameters converted to original scale (σ2 = exp(log_σ2)):")

        aux_data = Matrix{Any}(undef, length(aux_entries), 3)
        for (row, (name, param_idx)) in enumerate(aux_entries)
            aux_data[row, 1] = name
            aux_data[row, 2] = exp(coef[param_idx])
            aux_data[row, 3] = exp(coef[param_idx]) * stddev[param_idx]  # Delta method
        end

        pretty_table(aux_data;
                     column_labels=["", "Coef.", "Std.Err."],
                     formatters=[fmt__printf("%.4f", [2, 3])],
                     compact_printing=true,
                     backend=table_format)
        println()
        aux_table = aux_data
    end

    # Copula auxiliary table
    copula_info = _detect_copula_params(model, idx, coef, stddev)
    copula_aux_table = Matrix{Any}(undef, 0, 3)

    if length(copula_info) > 0
        println("Copula parameters (transformed to original scale):")

        copula_aux_data = Matrix{Any}(undef, 3, 3)
        copula_aux_data[1, 1] = "ρ"
        copula_aux_data[1, 2] = copula_info.rho
        copula_aux_data[1, 3] = copula_info.se_rho
        copula_aux_data[2, 1] = "Kendall's τ"
        copula_aux_data[2, 2] = copula_info.kendalls_tau
        copula_aux_data[2, 3] = ""
        copula_aux_data[3, 1] = "Tail dep."
        copula_aux_data[3, 2] = copula_info.tail_dep
        copula_aux_data[3, 3] = ""

        pretty_table(copula_aux_data;
                     column_labels=["", "Value", "Std.Err."],
                     formatters=[fmt__printf("%.4f", [2])],
                     compact_printing=true,
                     backend=table_format)
        println()
        copula_aux_table = copula_aux_data
    end

    print("Table format: "); printstyled(table_format; color=:yellow); println()

    return (table=table_data, aux_table=aux_table, copula_table=copula_aux_table, copula_info=copula_info)
end

"""
    print_table(spec::sfmodel_MCI_spec, coef::AbstractVector, var_cov_matrix::AbstractMatrix;
                optim_result=nothing, table_format::Symbol=:text)

Print formatted estimation results table using a model specification.
This is the simplified interface that extracts configuration from the spec.

# Arguments
- `spec::sfmodel_MCI_spec`: Model specification containing metadata for table formatting
- `coef::AbstractVector`: Coefficient vector from optimization
- `var_cov_matrix::AbstractMatrix`: Variance-covariance matrix from `var_cov_mat`

# Keyword Arguments
- `optim_result=nothing`: Optional Optim.jl result for convergence info
- `table_format::Symbol=:text`: Output format (`:text`, `:html`, or `:latex`)

# Returns
NamedTuple: `(table=Matrix, aux_table=Matrix)`

# Example
```julia
spec = sfmodel_MCI_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu],
              varnames=["_cons", "x1", "x2", "_cons", "_cons"],
              eqnames=["frontier", "μ", "ln_σᵤ²", "ln_σᵥ²"],
              eq_indices=[1, 4, 5, 6])

nll = p -> MCI_nll(spec, p)
result = optimize(nll, p0, Newton(); autodiff=AutoForwardDiff())
vcov = var_cov_mat(nll, Optim.minimizer(result))

# Simple call - uses varnames/eqnames/eq_indices from spec
print_table(spec, Optim.minimizer(result), vcov.var_cov_matrix; optim_result=result)
```
"""
function print_table(spec::sfmodel_MCI_spec{T}, coef::AbstractVector{T},
                     var_cov_matrix::AbstractMatrix{T};
                     GPU::Bool=false,
                     optim_result=nothing,
                     table_format::Symbol=:text) where {T<:Real}
    # Extract from spec
    return print_table(coef, var_cov_matrix,
                       spec.varnames, spec.eqnames, spec.eq_indices;
                       nobs=spec.N, noise=spec.noise, ineff=spec.ineff,
                       copula=spec.copula,
                       K=spec.K, L=spec.L, hetero=spec.hetero,
                       L_scaling=spec.L_scaling,
                       n_draws=size(spec.draws_2D, 2), sign=spec.sign,
                       GPU=GPU,
                       optim_result=optim_result, table_format=table_format)
end

# ============================================================================
# Section 11b: Model Fitting (sfmodel_MCI_fit)
# ============================================================================

"""
    sfmodel_MCI_fit(; model, init=nothing, optim_options=nothing, jlms_bc_index=true,
                     marginal=true, show_table=true, verbose=true)

Estimate a stochastic frontier model using MCI integration with optional two-stage optimization.

This function is a comprehensive wrapper that:
1. Prepares initial values (OLS-based if not provided)
2. Runs two-stage optimization (warmstart + main) if configured
3. Computes variance-covariance matrix
4. Calculates JLMS and BC efficiency indices
5. Computes marginal effects of inefficiency determinants
6. Prints formatted results tables
7. Returns a comprehensive NamedTuple with all results

# Arguments
- `model::sfmodel_MCI_spec`: Model specification from `sfmodel_MCI_spec()`

# Keyword Arguments
- `init=nothing`: Initial parameter vector. If nothing, OLS estimates are used for frontier
  coefficients and 0.1 for other parameters. Can be from `sfmodel_MCI_init()` or a plain vector.
- `optim_options=nothing`: Optimization options from `sfmodel_MCI_opt()`. If nothing, uses defaults:
  NelderMead warmstart (400 iter) + Newton main (2000 iter).
- `jlms_bc_index::Bool=true`: Compute JLMS and BC efficiency indices
- `marginal::Bool=true`: Compute marginal effects of exogenous determinants on E(u)
- `show_table::Bool=true`: Print estimation table
- `verbose::Bool=true`: Print detailed progress and results

# Returns
A NamedTuple containing:
- `converged`: Whether optimization converged
- `iter_limit_reached`: Whether iteration limit was reached
- `n_observations`: Number of observations
- `loglikelihood`: Log-likelihood value
- `table`: Coefficient table matrix
- `coeff`: Coefficient vector
- `std_err`: Standard errors
- `var_cov_mat`: Variance-covariance matrix
- `jlms`: JLMS inefficiency indices (if computed)
- `bc`: BC efficiency indices (if computed)
- `OLS_loglikelihood`: OLS log-likelihood
- `OLS_resid_skew`: Skewness of OLS residuals
- `marginal`: DataFrame of marginal effects (if computed)
- `marginal_mean`: Mean marginal effects
- `model`: The model specification
- `Hessian`: Hessian matrix
- `gradient_norm`: Gradient residual norm
- `actual_iterations`: Total iterations across all stages
- `warmstart_solver`, `warmstart_ini`, `warmstart_maxIT`: Warmstart info
- `main_solver`, `main_ini`, `main_maxIT`: Main optimization info
- `redflag`: Convergence warning flag (0=OK, 1=problems)
- `list`: OrderedDict containing all results

# Example
```julia
include("sf_MCI_v21.jl")

df = CSV.read("sampledata.csv", DataFrame)
y = df.y
X = hcat(ones(length(y)), df.x1, df.x2)
Z = hcat(ones(length(y)), df.z1)

# Step 1: Model specification (what to estimate)
myspec = sfmodel_spec(
    depvar = y,
    frontier = X,
    zvar = Z,
    noise = :Normal,
    ineff = :TruncatedNormal,
    hetero = Symbol[]
)

# Step 2: Method specification (how to estimate)
mymeth = sfmodel_method(
    method = :MCI,
    n_draws = 2^12 - 1
)

# Step 3: Initial values
p0 = sfmodel_MCI_init(
    model = myspec,
    frontier = X \\ y,  # OLS
    mu = [0.0],
    ln_sigma_sq = (0.0),
    ln_sigma_v_sq = (0.0)
)

# Step 4: Optimization options
myopt = sfmodel_MCI_opt(
    warmstart_solver = NelderMead(),
    warmstart_opt = (iterations = 400, g_abstol = 1e-5),
    main_solver = Newton(),
    main_opt = (iterations = 2000, g_abstol = 1e-8)
)

# Step 5: Estimate
result = sfmodel_MCI_fit(
    spec = myspec,
    method = mymeth,
    init = p0,
    optim_options = myopt,
    marginal = true,
    show_table = true,
    verbose = true
)

# Access results
println("Log-likelihood: ", result.loglikelihood)
println("Coefficients: ", result.coeff)
println("Mean BC efficiency: ", mean(result.bc))
```
"""
function sfmodel_MCI_fit(;
    spec::SFModelSpec,
    method::SFMethodSpec = sfmodel_method(),
    init=nothing,
    optim_options=nothing,
    jlms_bc_index::Bool=true,
    marginal::Bool=true,
    show_table::Bool=true,
    verbose::Bool=true
)
    # Assemble the internal sfmodel_MCI_spec from spec + method
    model = _assemble_MCI_spec(spec, method)

    # For simulation tracking
    redflag::Int = 0

    # ========== Banner ==========
    if show_table
        printstyled("\n###------------------------------------  ----------------###\n"; color=:yellow)
        printstyled("###  Estimating SF models using Quasi Monte Carlo (MCI)  ###\n"; color=:yellow)
        printstyled("###  integration in Julia                                ###\n"; color=:yellow)
        printstyled("###------------------------------------------------------###\n\n"; color=:yellow)

        printstyled("*********************************\n"; color=:cyan)
        printstyled("      The estimated model:\n"; color=:cyan)
        printstyled("*********************************\n"; color=:cyan)

        _hetero_label = model.scaling ? "Scaling Property" : (isempty(model.hetero) ? "Homoscedastic" : "Heteroscedastic")
        if model.copula != :None
            printstyled("  $(model.noise), $(model.ineff), $(_hetero_label), $(model.copula)\n"; color=:yellow)
        else
            printstyled("  $(model.noise), $(model.ineff), $(_hetero_label)\n"; color=:yellow)
        end
        printstyled("  N=$(model.N), K=$(model.K), L=$(model.L), n_draws=$(size(model.draws_2D, 2))\n"; color=:yellow)
        printstyled("  Type: $(model.sign == 1 ? "production" : "cost")\n"; color=:yellow)
        println()
    end

    # ========== Prepare Initial Values ==========
    Y = model.depvar
    X = model.frontier
    K = model.K
    N = model.N

    # OLS estimate for frontier coefficients
    β_ols_raw = X \ Y          # stays on same device as X, Y (CuArray when GPU=true)
    β_ols = Array(β_ols_raw)   # CPU copy for optimizer (sf_init must be CPU)

    if init === nothing
        # User does not provide init vector; use OLS + 0.1 for rest
        n_total = plen(model.model, model.K, model.L, model.hetero; L_scaling=model.L_scaling)
        sf_init = vcat(β_ols, fill(0.1, n_total - K))
        if verbose
            println("Using OLS-based initial values for frontier coefficients.")
            println("Other parameters initialized to 0.1")
            println()
        end
    else
        sf_init = init isa AbstractVector ? Float64.(init) : Float64.(vec(init))
    end

    # ========== OLS Statistics ==========
    resid = Y - X * β_ols_raw
    sse = sum(resid .^ 2)
    ssd = sqrt(sse / N)  # sample standard deviation
    ll_ols = sum(normlogpdf.(0, ssd, resid))  # OLS log-likelihood
    sk_ols = sum(resid .^ 3) / ((ssd^3) * N)  # skewness of OLS residuals

    # ========== Prepare Optimization Options ==========
    if optim_options === nothing
        # Use default optimization settings
        myopt = sfmodel_MCI_opt(
            warmstart_solver = NelderMead(),
            warmstart_opt = (iterations = 200, g_abstol = 1e-3),
            main_solver = Newton(),
            main_opt = (iterations = 200, g_abstol = 1e-7)
        )
    else
        myopt = optim_options
    end

    # Determine if we do warmstart
    # Skip warmstart if: no solver, no options, or iterations=0
    do_warmstart = myopt.warmstart_solver !== nothing &&
                   myopt.warmstart_opt !== nothing &&
                   myopt.warmstart_opt.iterations > 0

    # ========== Start Estimation ==========
    _lik = p -> MCI_nll(model, p; chunks=model.chunks)

    # Placeholders for recording
    sf_init_1st = do_warmstart ? copy(sf_init) : nothing
    sf_init_2nd = nothing
    sf_warmstart_algo = do_warmstart ? myopt.warmstart_solver : nothing
    sf_warmstart_maxit = do_warmstart ? myopt.warmstart_opt.iterations : 0
    sf_main_algo = myopt.main_solver
    sf_main_maxit = myopt.main_opt.iterations
    sf_total_iter = 0

    _optres = nothing
    _run = 1

    # ========== Stage 1: Warmstart ==========
    if do_warmstart && _run == 1
        if verbose
            printstyled("The warmstart run...\n\n"; color=:green)
        end

        _optres = Optim.optimize(_lik, sf_init, myopt.warmstart_solver,
                                  myopt.warmstart_opt)

        sf_total_iter += Optim.iterations(_optres)
        sf_init = Optim.minimizer(_optres)  # update for next stage
        _run = 2

        if verbose
            println()
            println(_optres)
            print("The warmstart results are:\n")
            printstyled(Optim.minimizer(_optres); color=:yellow)
            println("\n")
        end
    end

    # ========== Stage 2: Main Optimization ==========
    if !do_warmstart || _run == 2
        sf_init_2nd = copy(sf_init)

        if verbose
            println()
            printstyled("Starting the main optimization run...\n\n"; color=:green)
        end

        # Forward-mode AD for all distributions (including Gamma)
        _optres = Optim.optimize(_lik, sf_init, myopt.main_solver,
                                  myopt.main_opt; autodiff=AutoForwardDiff())

        sf_total_iter += Optim.iterations(_optres)

        if verbose
            println()
            println(_optres)
            print("The resulting coefficient vector is:\n")
            printstyled(Optim.minimizer(_optres); color=:yellow)
            println("\n")
        end

        # Check convergence
        if isnan(Optim.g_residual(_optres)) || (Optim.g_residual(_optres) > 0.1)
            redflag = 1
            printstyled("Note that the estimation may not have converged properly. The gradients are problematic (too large, > 0.1, or NaN).\n\n"; color=:red)
        end

        if Optim.iteration_limit_reached(_optres)
            redflag = 1
            printstyled("Caution: The number of iterations reached the limit.\n\n"; color=:red)
        end
    end

    # ========== Post-Estimation ==========
    _coevec = Optim.minimizer(_optres)

    # Variance-covariance matrix
    vcov_result = var_cov_mat(_lik, _coevec; message=verbose)
    var_cov_matrix = vcov_result.var_cov_matrix
    if vcov_result.redflag > 0
        redflag = max(redflag, vcov_result.redflag)
    end

    stddev = sqrt.(abs.(diag(var_cov_matrix)))  # abs for safety if negative

    # Compute Hessian for storage
    numerical_hessian = hessian(_lik, _coevec)

    # ========== JLMS and BC Indices ==========
    if jlms_bc_index
        eff_result = jlms_bc_indices(model, _coevec)
        _jlms = eff_result.jlms
        _bc = eff_result.bc
        _jlmsM = mean(_jlms)
        _bcM = mean(_bc)
    else
        _jlms = nothing
        _bc = nothing
        _jlmsM = nothing
        _bcM = nothing
    end

    # ========== Marginal Effects ==========
    if marginal && model.scaling
        margeff, margMinfo = marginal_effects_scaling(model, _coevec)
    elseif marginal && !isempty(model.hetero)
        margeff, margMinfo = marginal_effects(model, _coevec)
    else
        margeff = nothing
        margMinfo = NamedTuple()
    end

    # ========== Print Results Table ==========
    table_result = nothing
    if show_table
        table_result = print_table(model, _coevec, var_cov_matrix;
                                    GPU=method.GPU, optim_result=_optres, table_format=:text)

        printstyled("***** Additional Information *********\n"; color=:cyan)

        print("* OLS (frontier-only) log-likelihood: ")
        printstyled(round(ll_ols; digits=5); color=:yellow)
        println("")

        print("* Skewness of OLS residuals: ")
        printstyled(round(sk_ols; digits=5); color=:yellow)
        println("")

        if jlms_bc_index
            print("* The sample mean of the JLMS inefficiency index: ")
            printstyled(round(_jlmsM; digits=5); color=:yellow)
            println("")
            print("* The sample mean of the BC efficiency index: ")
            printstyled(round(_bcM; digits=5); color=:yellow)
            println("\n")
        end

        if marginal && (model.scaling || !isempty(model.hetero)) && length(margMinfo) >= 1
            print("* The sample mean of inefficiency determinants' marginal effects on E(u): ")
            printstyled(margMinfo; color=:yellow)
            println("")
            println("* Marginal effects of the inefficiency determinants at the observational level are saved in the return. See the follows.\n")
        end

        println("* Use `name.list` to see saved results (keys and values) where `name` is the return specified in `name = sfmodel_MCI_fit(..)`. Values may be retrieved using the keys. For instance:")
        println("   ** `name.loglikelihood`: the log-likelihood value of the model;")
        println("   ** `name.jlms`: Jondrow et al. (1982) inefficiency index;")
        println("   ** `name.bc`: Battese and Coelli (1988) efficiency index;")
        println("   ** `name.marginal`: a DataFrame with variables' (if any) marginal effects on E(u).")
        println("* Use `keys(name.list)` to see available keys.")

        printstyled("**************************************\n\n\n"; color=:cyan)
    end

    # ========== Build Return Dictionary ==========
    _dicRES = OrderedDict{Symbol, Any}()
    _dicRES[:converged] = Optim.converged(_optres)
    _dicRES[:iter_limit_reached] = Optim.iteration_limit_reached(_optres)
    _dicRES[:_______________] = "___________________"
    _dicRES[:n_observations] = N
    _dicRES[:loglikelihood] = -Optim.minimum(_optres)
    _dicRES[:table] = table_result !== nothing ? table_result.table : nothing
    _dicRES[:coeff] = _coevec
    _dicRES[:std_err] = stddev
    _dicRES[:var_cov_mat] = var_cov_matrix
    _dicRES[:jlms] = _jlms
    _dicRES[:bc] = _bc
    _dicRES[:OLS_loglikelihood] = ll_ols
    _dicRES[:OLS_resid_skew] = sk_ols
    _dicRES[:marginal] = margeff
    _dicRES[:marginal_mean] = margMinfo
    _dicRES[:_____________] = "___________________"
    _dicRES[:model] = model
    _dicRES[:________________] = "___________________"
    _dicRES[:Hessian] = numerical_hessian
    _dicRES[:gradient_norm] = Optim.g_residual(_optres)
    _dicRES[:actual_iterations] = sf_total_iter
    _dicRES[:______________] = "______________________"
    _dicRES[:warmstart_solver] = sf_warmstart_algo
    _dicRES[:warmstart_ini] = sf_init_1st
    _dicRES[:warmstart_maxIT] = sf_warmstart_maxit
    _dicRES[:main_solver] = sf_main_algo
    _dicRES[:main_ini] = sf_init_2nd
    _dicRES[:main_maxIT] = sf_main_maxit
    _dicRES[:main_tolerance] = myopt.main_opt.g_abstol
    _dicRES[:redflag] = redflag
    _dicRES[:______________________] = "___________________"
    _dicRES[:GPU] = method.GPU
    _dicRES[:n_draws] = method.draws !== nothing ? size(method.draws)[end] : method.n_draws
    _dicRES[:multiRand] = method.multiRand
    _dicRES[:chunks] = method.chunks
    _dicRES[:distinct_Halton_length] = method.distinct_Halton_length
    _dicRES[:estimation_method] = method.method

    # Add individual coefficient vectors based on model equations
    idx = model.idx
    _dicRES[:frontier] = _coevec[idx.beta]

    # Add scaling δ coefficients
    if model.scaling
        _dicRES[:delta] = _coevec[idx.delta]
    end

    # Add inefficiency-specific coefficients
    if hasfield(typeof(idx.ineff), :mu)
        _dicRES[:mu] = _coevec[idx.ineff.mu]
    end
    if hasfield(typeof(idx.ineff), :sigma_sq)
        _dicRES[:sigma_sq] = _coevec[idx.ineff.sigma_sq]
    end
    if hasfield(typeof(idx.ineff), :sigma_u)
        _dicRES[:sigma_u] = _coevec[idx.ineff.sigma_u]
    end
    if hasfield(typeof(idx.ineff), :lambda)
        _dicRES[:lambda] = _coevec[idx.ineff.lambda]
    end
    if hasfield(typeof(idx.ineff), :k)
        _dicRES[:k] = _coevec[idx.ineff.k]
    end
    if hasfield(typeof(idx.ineff), :theta)
        _dicRES[:theta] = _coevec[idx.ineff.theta]
    end
    if hasfield(typeof(idx.ineff), :alpha)
        _dicRES[:alpha] = _coevec[idx.ineff.alpha]
    end
    if hasfield(typeof(idx.ineff), :ln_lambda)
        _dicRES[:ln_lambda] = _coevec[idx.ineff.ln_lambda]
    end

    # Add noise-specific coefficients
    if hasfield(typeof(idx.noise), :ln_sigma_v_sq)
        _dicRES[:ln_sigma_v_sq] = _coevec[idx.noise.ln_sigma_v_sq]
    end
    if hasfield(typeof(idx.noise), :ln_b)
        _dicRES[:ln_b] = _coevec[idx.noise.ln_b]
    end
    if hasfield(typeof(idx.noise), :ln_nu_minus_2)
        _dicRES[:ln_nu_minus_2] = _coevec[idx.noise.ln_nu_minus_2]
    end

    # Add copula-specific coefficients
    if hasfield(typeof(idx.copula), :theta_rho)
        _dicRES[:theta_rho] = _coevec[idx.copula.theta_rho]
        # Also store transformed values
        copula_info = table_result !== nothing ? table_result.copula_info : nothing
        if copula_info !== nothing && length(copula_info) > 0
            _dicRES[:rho] = copula_info.rho
            _dicRES[:kendalls_tau] = copula_info.kendalls_tau
        end
    end

    # Create NamedTuple from dictionary
    _ntRES = NamedTuple{Tuple(keys(_dicRES))}(values(_dicRES))
    _ntRES = (; _ntRES..., list=_dicRES)

    return _ntRES
end

# ============================================================================
# Section 12: Exports and Usage Examples
# ============================================================================

# Export main types and functions
export MCINoiseModel, MCIIneffModel
export NormalNoise_MCI, StudentTNoise_MCI, LaplaceNoise_MCI
export TruncatedNormal_MCI, Exponential_MCI
export HalfNormal_MCI, Weibull_MCI, Lognormal_MCI, Lomax_MCI, Rayleigh_MCI, Gamma_MCI
export MCICopulaModel, NoCopula_MCI, GaussianCopula_MCI, ClaytonCopula_MCI, Clayton90Copula_MCI, GumbelCopula_MCI, COPULA_MODELS
export MCIModel, NOISE_MODELS, INEFF_MODELS
export MCI_nll, make_halton_p, make_halton_wrap, make_constants, plen
export valid_hetero
export jlms_bc_indices
export var_cov_mat, print_table
export sfmodel_MCI_init
export sfmodel_MCI_opt, sfmodel_MCI_optim, sfmodel_MCI_fit
export SFModelSpec, SFMethodSpec, sfmodel_spec, sfmodel_method

# Include marginal effects module
include("sf_MCI_marginal_v21.jl")
export marginal_effects

#=
# ============================================================================
# Usage Examples
# ============================================================================

using Random
Random.seed!(123)

N = 100
halton = make_halton_p(1023)

Y = randn(N)
X = hcat(ones(N), randn(N, 2))  # N x 3
Z = hcat(ones(N), randn(N, 2))  # N x 3
K, L = size(X, 2), size(Z, 2)

# ============================================================================
# Example 1: Normal + Truncated Normal (hetero mu)
# ============================================================================
# p = [beta(3), ln_sigma_v_sq(1), delta(3), ln_sigma_sq(1)] -> length = 8
p_ntn = [1.0, 0.5, -0.3, -1.0, 0.5, 0.1, -0.2, -0.5]
nll_ntn = MCI_nll(Y, X, Z, p_ntn, halton; noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])
println("Normal + TN NLL: $nll_ntn")

# ============================================================================
# Example 2: Normal + Exponential (hetero lambda)
# ============================================================================
# p = [beta(3), ln_sigma_v_sq(1), gamma(3)] -> length = 7
p_ne = [1.0, 0.5, -0.3, -1.0, 0.5, 0.1, -0.2]
nll_ne = MCI_nll(Y, X, Z, p_ne, halton; noise=:Normal, ineff=:Exponential, hetero=[:lambda])
println("Normal + Exponential NLL: $nll_ne")

# ============================================================================
# Example 3: Student T + Truncated Normal (hetero mu)
# ============================================================================
# p = [beta(3), ln_sigma_v_sq(1), ln_nu_minus_2(1), delta(3), ln_sigma_sq(1)] -> length = 9
p_ttn = [1.0, 0.5, -0.3, -1.0, 1.0, 0.5, 0.1, -0.2, -0.5]  # ln_nu_minus_2=1.0 -> nu?4.7
nll_ttn = MCI_nll(Y, X, Z, p_ttn, halton; noise=:StudentT, ineff=:TruncatedNormal, hetero=[:mu])
println("StudentT + TN NLL: $nll_ttn")

# ============================================================================
# Example 4: Student T + Exponential (hetero lambda)
# ============================================================================
# p = [beta(3), ln_sigma_v_sq(1), ln_nu_minus_2(1), gamma(3)] -> length = 8
p_te = [1.0, 0.5, -0.3, -1.0, 1.0, 0.5, 0.1, -0.2]
nll_te = MCI_nll(Y, X, Z, p_te, halton; noise=:StudentT, ineff=:Exponential, hetero=[:lambda])
println("StudentT + Exponential NLL: $nll_te")

# ============================================================================
# Example 5: Hetero validation error
# ============================================================================
# This will error with a helpful message:
# MCI_nll(Y, X, Z, p_ntn, halton; noise=:Normal, ineff=:TruncatedNormal, hetero=[:lambda])
# ERROR: Invalid hetero option :lambda for TruncatedNormal_MCI. Valid options: [:mu, :sigma_sq]

# ============================================================================
# Example 6: Efficiency Indices (JLMS and BC)
# ============================================================================
# Using the same model as Example 1: Normal + TruncatedNormal (hetero mu)
result = jlms_bc_indices(Y, X, Z, p_ntn, halton;
    noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])

# Access individual components
jlms = result.jlms        # E(u|ε) - inefficiency index
bc = result.bc            # E(e^{-u}|ε) - technical efficiency
likelihood = result.likelihood  # f_ε(ε) - observation-level likelihood

println("Mean JLMS (inefficiency): $(mean(jlms))")
println("Mean BC (efficiency): $(mean(bc))")
println("Log-likelihood: $(sum(log.(likelihood)))")

# Sanity checks:
# - JLMS should be non-negative (u ? 0)
# - BC should be in (0, 1] (e^{-u} ? (0, 1] for u ? 0)
@assert all(jlms .>= 0) "JLMS should be non-negative"
@assert all(0 .< bc .<= 1) "BC should be in (0, 1]"

# ============================================================================
# Example 7: Efficiency Indices with StudentT noise
# ============================================================================
# Using model from Example 3: StudentT + TruncatedNormal (hetero mu)
result_t = jlms_bc_indices(Y, X, Z, p_ttn, halton;
    noise=:StudentT, ineff=:TruncatedNormal, hetero=[:mu])

println("StudentT model - Mean JLMS: $(mean(result_t.jlms))")
println("StudentT model - Mean BC: $(mean(result_t.bc))")

# ============================================================================
# Example 8: Efficiency Indices with chunked computation
# ============================================================================
# Use chunks=4 for memory-constrained GPU scenarios
result_chunked = jlms_bc_indices(Y, X, Z, p_ntn, halton;
    noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu], chunks=4)

# Results should be identical to non-chunked version
@assert isapprox(result.jlms, result_chunked.jlms, rtol=1e-10)
@assert isapprox(result.bc, result_chunked.bc, rtol=1e-10)
println("Chunked and non-chunked results match!")

# ============================================================================
# Example 9: Using sfmodel_MCI_spec for simplified workflow
# ============================================================================
# sfmodel_MCI_spec centralizes all model configuration, so you only specify it once

# Create specification with auto-generated defaults
spec = sfmodel_MCI_spec(
    depvar = y,
    frontier = X,
    zvar = Z,
    noise = :Normal,
    ineff = :TruncatedNormal,
    hetero = [:mu],
    n_draws = 1023   # Auto-generates Halton draws
)

# Now all functions use the spec - no need to repeat arguments!
nll_from_spec = MCI_nll(spec, p_ntn)
println("NLL from sfmodel_MCI_spec: $nll_from_spec")

# Compare with original approach - should be identical
nll_original = MCI_nll(Y, X, Z, p_ntn, halton;
    noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])
@assert isapprox(nll_from_spec, nll_original, rtol=1e-10)
println("sfmodel_MCI_spec and original NLL match!")

# Efficiency indices also work with spec
result_spec = jlms_bc_indices(spec, p_ntn)
println("sfmodel_MCI_spec Mean BC: $(mean(result_spec.bc))")

# ============================================================================
# Example 10: sfmodel_MCI_spec with custom variable names
# ============================================================================
# Provide custom names for print_table
spec_named = sfmodel_MCI_spec(
    depvar = y,
    frontier = X,
    zvar = Z,
    noise = :Normal,
    ineff = :TruncatedNormal,
    hetero = [:mu],
    varnames = ["_cons", "output", "capital", "_cons", "age", "size", "_cons", "_cons"],
    eqnames = ["frontier", "μ", "ln_σᵤ²", "ln_σᵥ²"],
    eq_indices = [1, 4, 7, 8]
)

# These names will be used automatically in print_table
# print_table(spec_named, coef, vcov; optim_result=result)

# ============================================================================
# Example 11: sfmodel_MCI_spec simplifies optimization workflow
# ============================================================================
# Full workflow using sfmodel_MCI_spec

using Optim

# 1. Create spec once
spec_optim = sfmodel_MCI_spec(
    depvar = y, frontier = X, zvar = Z,
    noise = :Normal, ineff = :TruncatedNormal, hetero = [:mu]
)

# 2. Define NLL closure using spec
nll_closure = p -> MCI_nll(spec_optim, p)

# 3. Optimize
p0 = zeros(plen(spec_optim.model, spec_optim.K, spec_optim.L, spec_optim.hetero))
# result_optim = optimize(nll_closure, p0, Newton(); autodiff=AutoForwardDiff())

# 4. Get variance-covariance matrix
# vcov = var_cov_mat(nll_closure, Optim.minimizer(result_optim))

# 5. Compute efficiency indices
# eff = jlms_bc_indices(spec_optim, Optim.minimizer(result_optim))

# 6. Print table - all formatting info comes from spec
# print_table(spec_optim, Optim.minimizer(result_optim), vcov.var_cov_matrix;
#             optim_result=result_optim)

println("sfmodel_MCI_spec workflow examples completed!")
=#
