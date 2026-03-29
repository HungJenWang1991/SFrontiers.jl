# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

#=
    sf_MSLE_v21.jl

    #! Added sfmodel_spec(), sfmodel_method(), sfmodel_MSLE_init(), print_table(),
    #!       _invert_hessian(), var_cov_mat()

    Negative log-likelihood for stochastic frontier models using MSLE (Halton) draws.
    Supports multiple noise and inefficiency distribution combinations via multiple dispatch.

    Supported Models:
    - Noise: Normal, StudentT, Laplace
    - Inefficiency: TruncatedNormal, Exponential, HalfNormal, Weibull, Lognormal, Lomax, Rayleigh

    Heteroscedasticity Options (via hetero keyword):
    - TruncatedNormal: [:mu, :sigma_sq]
    - Exponential: [:lambda]
    - HalfNormal: [:sigma_sq]
    - Weibull: [:lambda, :k]
    - Lognormal: [:mu, :sigma_sq]
    - Lomax: [:lambda, :alpha]
    - Rayleigh: [:sigma_sq]

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
using Distributions: Normal, TDist, quantile, cquantile, ccdf, normlogpdf
using PrettyTables: pretty_table, fmt__printf
using Optim
using ADTypes: AutoForwardDiff
using OrderedCollections: OrderedDict


# ============================================================================
# Section 1: Type Hierarchy
# ============================================================================

"""Abstract type for noise models in MSLE-based stochastic frontier estimation."""
abstract type MSLENoiseModel end

"""Abstract type for inefficiency models in MSLE-based stochastic frontier estimation."""
abstract type MSLEIneffModel end

"""Normal noise: v ~ N(0, σ_v²)"""
struct NormalNoise_MSLE <: MSLENoiseModel end

"""Student T noise: v ~ t(0, σ_v, ν) with scale σ_v and degrees of freedom ν > 2"""
struct StudentTNoise_MSLE <: MSLENoiseModel end

"""Laplace noise: v ~ Laplace(0, b) with scale b"""
struct LaplaceNoise_MSLE <: MSLENoiseModel end

"""Truncated Normal inefficiency: u ~ TN(μ, σ_u; lower=0)"""
struct TruncatedNormal_MSLE <: MSLEIneffModel end

"""Exponential inefficiency: u ~ Exp(λ), where λ = Var(u)"""
struct Exponential_MSLE <: MSLEIneffModel end

"""Half Normal inefficiency: u ~ HalfNormal(σ), i.e., |N(0, σ²)|"""
struct HalfNormal_MSLE <: MSLEIneffModel end

"""Weibull inefficiency: u ~ Weibull(λ, k) with scale λ and shape k"""
struct Weibull_MSLE <: MSLEIneffModel end

"""Lognormal inefficiency: u ~ LogNormal(μ, σ)"""
struct Lognormal_MSLE <: MSLEIneffModel end

"""Lomax inefficiency: u ~ Lomax(α, λ)"""
struct Lomax_MSLE <: MSLEIneffModel end

"""Rayleigh inefficiency: u ~ Rayleigh(σ)"""
struct Rayleigh_MSLE <: MSLEIneffModel end

# --- Copula models ---

"""Abstract type for copula models in MSLE-based stochastic frontier estimation."""
abstract type MSLECopulaModel end

"""No copula (independence between v and u). This is the default."""
struct NoCopula_MSLE <: MSLECopulaModel end

"""Gaussian copula: models dependence between v and u via parameter ρ ∈ (-1, 1)."""
struct GaussianCopula_MSLE <: MSLECopulaModel end

"""Clayton copula: models lower tail dependence between v and u via parameter ρ > 0."""
struct ClaytonCopula_MSLE <: MSLECopulaModel end

"""Gumbel copula: models upper tail dependence between v and u via parameter ρ ≥ 1."""
struct GumbelCopula_MSLE <: MSLECopulaModel end

"""Clayton 90° rotated copula: models upper-lower tail dependence via parameter ρ > 0."""
struct Clayton90Copula_MSLE <: MSLECopulaModel end


"""
    MSLEModel{N<:MSLENoiseModel, U<:MSLEIneffModel, C<:MSLECopulaModel}

Composite type representing a stochastic frontier model with specific noise, inefficiency,
and copula distributions. Enables multiple dispatch on model combinations.
"""
struct MSLEModel{N<:MSLENoiseModel, U<:MSLEIneffModel, C<:MSLECopulaModel}
    noise::N
    ineff::U
    copula::C
end

# Registry for symbol-based lookup (used by qmc_nll interface)
# Store Types (not instances) to avoid Julia 1.12+ world age issues
const NOISE_MODELS = Dict{Symbol, Type{<:MSLENoiseModel}}(
    :Normal   => NormalNoise_MSLE,
    :StudentT => StudentTNoise_MSLE,
    :Laplace  => LaplaceNoise_MSLE
)

const INEFF_MODELS = Dict{Symbol, Type{<:MSLEIneffModel}}(
    :TruncatedNormal => TruncatedNormal_MSLE,
    :Exponential     => Exponential_MSLE,
    :HalfNormal      => HalfNormal_MSLE,
    :Weibull         => Weibull_MSLE,
    :Lognormal       => Lognormal_MSLE,
    :Lomax           => Lomax_MSLE,
    :Rayleigh        => Rayleigh_MSLE
)

const COPULA_MODELS = Dict{Symbol, Type{<:MSLECopulaModel}}(
    :None     => NoCopula_MSLE,
    :Gaussian => GaussianCopula_MSLE,
    :Clayton  => ClaytonCopula_MSLE,
    :Gumbel     => GumbelCopula_MSLE,
    :Clayton90  => Clayton90Copula_MSLE,
)

"""
    _build_model(noise::Symbol, ineff::Symbol; copula::Symbol=:None) -> MSLEModel

Build a MSLEModel from noise, inefficiency, and copula symbols.
Validates that symbols are recognized and provides helpful error messages.
"""
function _build_model(noise::Symbol, ineff::Symbol; copula::Symbol=:None)
    if !haskey(NOISE_MODELS, noise)
        error("Unknown noise model: :$noise. Valid options: $(collect(keys(NOISE_MODELS)))")
    end
    if ineff == :Gamma
        error("`ineff=:Gamma` is not supported by `method=:MSLE`. Use `method=:MCI` instead.")
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
    return MSLEModel(NOISE_MODELS[noise](), INEFF_MODELS[ineff](), COPULA_MODELS[copula]())
end

# ============================================================================
# Section 2: Trait Functions (Parameter Counts via Dispatch)
# ============================================================================

"""Number of extra noise parameters beyond ln_sigma_v_sq (or ln_b for Laplace)."""
noise_extras(::NormalNoise_MSLE) = 0
noise_extras(::StudentTNoise_MSLE) = 1  # ln(ν - 2)
noise_extras(::LaplaceNoise_MSLE) = 0   # Just ln_b (uses ln_sigma_v_sq slot)

"""Number of extra inefficiency parameters (beyond main heteroscedastic params)."""
ineff_extras(::TruncatedNormal_MSLE) = 1  # ln_sigma_sq (may be heteroscedastic)
ineff_extras(::Exponential_MSLE) = 0      # λ params handled via hetero
ineff_extras(::HalfNormal_MSLE) = 0       # σ² params handled via hetero
ineff_extras(::Weibull_MSLE) = 0          # λ, k params handled via hetero
ineff_extras(::Lognormal_MSLE) = 0        # μ, σ² params handled via hetero
ineff_extras(::Lomax_MSLE) = 0            # λ, α params handled via hetero
ineff_extras(::Rayleigh_MSLE) = 0         # σ² params handled via hetero

"""Does this inefficiency model have a μ parameter?"""
has_mu(::TruncatedNormal_MSLE) = true
has_mu(::Exponential_MSLE) = false
has_mu(::HalfNormal_MSLE) = false
has_mu(::Weibull_MSLE) = false
has_mu(::Lognormal_MSLE) = true
has_mu(::Lomax_MSLE) = false
has_mu(::Rayleigh_MSLE) = false

"""Valid heteroscedasticity options for each inefficiency model."""
valid_hetero(::TruncatedNormal_MSLE) = [:mu, :sigma_sq]
valid_hetero(::Exponential_MSLE) = [:lambda]
valid_hetero(::HalfNormal_MSLE) = [:sigma_sq]
valid_hetero(::Weibull_MSLE) = [:lambda, :k]
valid_hetero(::Lognormal_MSLE) = [:mu, :sigma_sq]
valid_hetero(::Lomax_MSLE) = [:lambda, :alpha]
valid_hetero(::Rayleigh_MSLE) = [:sigma_sq]

"""Number of copula parameters."""
copula_plen(::NoCopula_MSLE) = 0
copula_plen(::GaussianCopula_MSLE) = 1  # theta_rho (transformed via tanh)
copula_plen(::ClaytonCopula_MSLE) = 1  # theta_rho (transformed via exp)
copula_plen(::GumbelCopula_MSLE) = 1   # theta_rho (transformed via exp + 1)
copula_plen(::Clayton90Copula_MSLE) = 1  # theta_rho (transformed via exp, same as Clayton)

"""
    _validate_hetero(ineff::MSLEIneffModel, hetero::Vector{Symbol})

Validate that hetero options are valid for the given inefficiency model.
Raises an error with helpful message if invalid options are provided.
"""
function _validate_hetero(ineff::MSLEIneffModel, hetero::Vector{Symbol})
    valid = valid_hetero(ineff)
    for h in hetero
        if h ∉ valid
            error("Invalid hetero option :$h for $(typeof(ineff)). " *
                  "Valid options: $(valid)")
        end
    end
end

"""
    plen(model::MSLEModel, K::Int, L::Int, hetero::Vector{Symbol})

Calculate total parameter vector length for a given model and heteroscedasticity settings.
"""
function plen(model::MSLEModel, K::Int, L::Int, hetero::Vector{Symbol}; L_scaling::Int=0)
    n_beta = K
    n_delta = L_scaling  # scaling function coefficients (0 if no scaling)
    n_noise = 1 + noise_extras(model.noise)  # ln_sigma_v_sq + extras (e.g., ln_nu_minus_2)
    n_ineff = ineff_plen(model.ineff, L, hetero)
    n_copula = copula_plen(model.copula)
    return n_beta + n_delta + n_noise + n_ineff + n_copula
end

function ineff_plen(::TruncatedNormal_MSLE, L::Int, hetero::Vector{Symbol})
    n_mu = :mu in hetero ? L : 1
    n_sigma_u = :sigma_sq in hetero ? L : 1
    return n_mu + n_sigma_u
end

function ineff_plen(::Exponential_MSLE, L::Int, hetero::Vector{Symbol})
    n_lambda = :lambda in hetero ? L : 1
    return n_lambda
end

function ineff_plen(::HalfNormal_MSLE, L::Int, hetero::Vector{Symbol})
    return :sigma_sq in hetero ? L : 1
end

function ineff_plen(::Weibull_MSLE, L::Int, hetero::Vector{Symbol})
    n_lambda = :lambda in hetero ? L : 1
    n_k = :k in hetero ? L : 1
    return n_lambda + n_k
end

function ineff_plen(::Lognormal_MSLE, L::Int, hetero::Vector{Symbol})
    n_mu = :mu in hetero ? L : 1
    n_sigma = :sigma_sq in hetero ? L : 1
    return n_mu + n_sigma
end

function ineff_plen(::Lomax_MSLE, L::Int, hetero::Vector{Symbol})
    n_lambda = :lambda in hetero ? L : 1
    n_alpha = :alpha in hetero ? L : 1
    return n_lambda + n_alpha
end

function ineff_plen(::Rayleigh_MSLE, L::Int, hetero::Vector{Symbol})
    return :sigma_sq in hetero ? L : 1
end


# ============================================================================
# Section 2b: Parameter Index System (for SFGPU-style parameter access)
# ============================================================================

"""
    _param_ind(model::MSLEModel, K::Int, L::Int, hetero::Vector{Symbol})

Compute parameter indices (ranges) for a given model and heteroscedasticity settings.
Returns a NamedTuple with index ranges for beta, noise, and inefficiency parameters.
This enables direct indexing into CPU parameter vectors (no views into CuArray needed).
"""
function _param_ind(model::MSLEModel, K::Int, L::Int, hetero::Vector{Symbol}; L_scaling::Int=0)
    idx = 1

    # Beta indices (K elements)
    beta = idx:(idx+K-1)
    idx += K

    # Scaling delta indices (empty range if no scaling)
    delta = L_scaling > 0 ? (idx:(idx+L_scaling-1)) : (1:0)
    idx += L_scaling

    # Inefficiency parameter indices via dispatch (before noise, to match eq_names/eq_indices convention)
    ineff_idx, idx = _ineff_ind(model.ineff, idx, L, hetero)

    # Noise parameter indices via dispatch
    noise_idx, idx = _noise_ind(model.noise, idx)

    # Copula parameter indices via dispatch (at the end of parameter vector)
    copula_idx, idx = _copula_ind(model.copula, idx)

    return (beta=beta, delta=delta, noise=noise_idx, ineff=ineff_idx, copula=copula_idx)
end

# --- Noise parameter indices ---

function _noise_ind(::NormalNoise_MSLE, idx)
    ln_sigma_v_sq = idx
    return (ln_sigma_v_sq=ln_sigma_v_sq,), idx + 1
end

function _noise_ind(::StudentTNoise_MSLE, idx)
    ln_sigma_v_sq = idx
    ln_nu_minus_2 = idx + 1
    return (ln_sigma_v_sq=ln_sigma_v_sq, ln_nu_minus_2=ln_nu_minus_2), idx + 2
end

function _noise_ind(::LaplaceNoise_MSLE, idx)
    ln_b = idx
    return (ln_b=ln_b,), idx + 1
end

# --- Inefficiency parameter indices ---

function _ineff_ind(::TruncatedNormal_MSLE, idx, L, hetero)
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

function _ineff_ind(::Exponential_MSLE, idx, L, hetero)
    lambda_hetero = :lambda in hetero
    n_lambda = lambda_hetero ? L : 1
    lambda = idx:(idx+n_lambda-1)
    idx += n_lambda

    return (lambda=lambda, lambda_hetero=lambda_hetero), idx
end

function _ineff_ind(::HalfNormal_MSLE, idx, L, hetero)
    sigma_sq_hetero = :sigma_sq in hetero
    n_sigma = sigma_sq_hetero ? L : 1
    sigma_sq = idx:(idx+n_sigma-1)
    idx += n_sigma

    return (sigma_sq=sigma_sq, sigma_sq_hetero=sigma_sq_hetero), idx
end

function _ineff_ind(::Weibull_MSLE, idx, L, hetero)
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

function _ineff_ind(::Lognormal_MSLE, idx, L, hetero)
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

function _ineff_ind(::Lomax_MSLE, idx, L, hetero)
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

function _ineff_ind(::Rayleigh_MSLE, idx, L, hetero)
    sigma_sq_hetero = :sigma_sq in hetero
    n_sigma = sigma_sq_hetero ? L : 1
    sigma_sq = idx:(idx+n_sigma-1)
    idx += n_sigma

    return (sigma_sq=sigma_sq, sigma_sq_hetero=sigma_sq_hetero), idx
end

# --- Copula parameter indices ---

function _copula_ind(::NoCopula_MSLE, idx)
    return NamedTuple(), idx   # No copula parameters
end

function _copula_ind(::GaussianCopula_MSLE, idx)
    theta_rho = idx
    return (theta_rho=theta_rho,), idx + 1
end

function _copula_ind(::ClaytonCopula_MSLE, idx)
    theta_rho = idx
    return (theta_rho=theta_rho,), idx + 1
end

function _copula_ind(::GumbelCopula_MSLE, idx)
    theta_rho = idx
    return (theta_rho=theta_rho,), idx + 1
end

function _copula_ind(::Clayton90Copula_MSLE, idx)
    theta_rho = idx
    return (theta_rho=theta_rho,), idx + 1
end


"""Check if a column is constant (all values equal). Used to validate scaling zvar."""
function _is_constant_column(col::AbstractVector)
    length(col) == 0 && return true
    first_val = col[1]
    return all(x -> x ≈ first_val, col)
end


# ============================================================================
# Section 2c: Model Specification (sfmodel_MSLE_spec)
# ============================================================================

"""
    sfmodel_MSLE_spec{T<:AbstractFloat}

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
- `model::MSLEModel`: Built model object
- `idx::NamedTuple`: Parameter indices

# Example
```julia
spec = sfmodel_MSLE_spec(
    depvar = y, frontier = X, zvar = Z,
    noise = :Normal,
    ineff = :TruncatedNormal,
    hetero = [:mu],
    varnames = ["_cons", "x1", "x2", "_cons", "_cons"],
    eqnames = ["frontier", "μ", "ln_σᵤ²", "ln_σᵥ²"],
    eq_indices = [1, 4, 5, 6]
)

# Use with simplified function calls
nll = qmc_nll(spec, p)
eff = jlms_bc_indices(spec, p)
print_table(spec, coef, vcov)
```
"""
mutable struct sfmodel_MSLE_spec{T<:AbstractFloat}
    # Data
    depvar::AbstractVector{T}
    frontier::AbstractMatrix{T}
    zvar::AbstractMatrix{T}

    # Model specification
    noise::Symbol
    ineff::Symbol
    copula::Symbol              # :None, :Gaussian, :Clayton, :Clayton90, :Gumbel
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
    model::MSLEModel
    idx::NamedTuple
    sign::Int  # 1 for production frontier, -1 for cost frontier
    chunks::Int  # Number of chunks for GPU memory management

    # Scaling property model fields
    scaling::Bool                                    # true if scaling property model (u = h(z)*u*)
    scaling_zvar::Union{Nothing, AbstractMatrix{T}}  # Z matrix for scaling function h(z)=exp(z'δ)
    L_scaling::Int                                   # number of scaling Z columns (0 if no scaling)
end

# ============================================================================
# Public API Structs (User-Facing)
# ============================================================================

"""
    SFModelSpec_MSLE{T}

Model specification struct returned by `sfmodel_spec()`. Contains data arrays,
distribution choices, variable names, and dimensions — everything about
"what to estimate."

# Fields
- `depvar`: Dependent variable vector (N)
- `frontier`: Frontier variable matrix (N × K)
- `zvar`: Z variable matrix (N × L) for heteroscedasticity
- `noise`: Noise distribution symbol (:Normal, :StudentT, :Laplace)
- `ineff`: Inefficiency distribution symbol
- `hetero`: Heteroscedasticity options
- `varnames`: Variable names for print_table
- `eqnames`: Equation names
- `eq_indices`: Equation indices
- `N`, `K`, `L`: Dimensions
- `model`: MSLEModel (noise + ineff dispatch pair)
- `idx`: Parameter index NamedTuple
- `sign`: Frontier sign (1 for production, -1 for cost)
"""
struct SFModelSpec_MSLE{T<:AbstractFloat}
    depvar::AbstractVector{T}
    frontier::AbstractMatrix{T}
    zvar::AbstractMatrix{T}
    noise::Symbol
    ineff::Symbol
    copula::Symbol              # :None, :Gaussian, :Clayton, :Clayton90, :Gumbel
    hetero::Vector{Symbol}
    varnames::Vector{String}
    eqnames::Vector{String}
    eq_indices::Vector{Int}
    N::Int
    K::Int
    L::Int
    model::MSLEModel
    idx::NamedTuple
    sign::Int

    # Scaling property model fields
    scaling::Bool                                    # true if scaling property model (u = h(z)*u*)
    scaling_zvar::Union{Nothing, AbstractMatrix{T}}  # Z matrix for scaling function h(z)=exp(z'δ)
    L_scaling::Int                                   # number of scaling Z columns (0 if no scaling)
end

"""
    SFMethodSpec_MSLE

Numerical method specification struct returned by `sfmodel_method()`. Contains settings
for draw generation, GPU usage, and chunking — everything about "how to estimate."

# Fields
- `method`: Estimation method symbol (default `:MSLE`)
- `draws`: User-provided draws, or `nothing` to auto-generate Halton sequences
- `n_draws`: Number of draws (default 1024)
- `multiRand`: Per-observation draws (`true`, N×D) or shared draws (`false`, 1×D)
- `GPU`: Whether to use GPU acceleration
- `chunks`: Number of chunks for GPU memory management
- `distinct_Halton_length`: Maximum Halton sequence length for multiRand mode (default 2^15-1 = 32767)
"""
struct SFMethodSpec_MSLE
    method::Symbol
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
Abstract parent type for all initial value specifications used by `sfmodel_MSLE_init()`.
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

struct ThetaRhoInit <: InitSpec       # For copula theta_rho parameter (all copula types)
    values::Vector{Float64}
end

struct ScalingInit <: InitSpec        # For scaling function δ coefficients
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

# For depvar: unwrap [yvar] → yvar using only()
_to_vector(x::AbstractVector{<:Real}) = x
_to_vector(x::AbstractVector{<:AbstractVector}) = only(x)
_to_vector(x::AbstractMatrix{<:Real}) = vec(x)  # handle N×1 matrix input

# For frontier/zvar: convert [v1, v2, v3] → matrix using reduce(hcat, ...)
# Note: use reduce(hcat, x) instead of stack(x) to preserve CuArray type for GPU
_to_matrix(x::AbstractMatrix) = x
_to_matrix(x::AbstractVector{<:AbstractVector}) = reduce(hcat, x)
_to_matrix(x::AbstractVector{<:Real}) = reshape(x, :, 1)  # handle plain vector as N×1 matrix

# Convert array to same device (CPU/GPU) as target array
# Works without importing CUDA.jl - detects CuArray by type name
function _to_device_array(target::AbstractArray, source::AbstractVector)
    target_type_str = string(typeof(target).name.wrapper)
    if target_type_str == "CuArray"
        # Get CuArray constructor from the parent module of target
        CuArrayConstructor = typeof(target).name.wrapper
        return CuArrayConstructor(source)
    else
        return source
    end
end

function _to_device_array(target::AbstractArray, source::AbstractMatrix)
    target_type_str = string(typeof(target).name.wrapper)
    if target_type_str == "CuArray"
        CuArrayConstructor = typeof(target).name.wrapper
        return CuArrayConstructor(source)
    else
        return source
    end
end

# ============================================================================
# Deferred Computation Helpers (for _assemble_MSLE_spec)
# ============================================================================

"""
    _maybe_gpu_convert(depvar, frontier, zvar, gpu)

Convert data arrays to GPU (CuArray) if `gpu=true`. Returns the arrays unchanged
if `gpu=false`. Requires `using CUDA` to have been called before use with GPU.
"""
function _maybe_gpu_convert(depvar, frontier, zvar, gpu::Bool)
    if gpu
        if !isdefined(Main, :CUDA)
            error("GPU=true requires CUDA.jl to be loaded. Please run `using CUDA` before calling this function.")
        end
        return Main.CUDA.CuArray(depvar), Main.CUDA.CuArray(frontier), Main.CUDA.CuArray(zvar)
    end
    return depvar, frontier, zvar
end

"""
    _prepare_draws(depvar_ref, N, T, user_draws, n_draws, multiRand)

Prepare Halton draws for MSLE estimation. If `user_draws` is `nothing`, auto-generates
Halton sequences. Returns `(draws_vec, draws_2D)`.

- `multiRand=true`: N×D matrix (different draws per observation)
- `multiRand=false`: 1×D matrix (shared draws across observations)
"""
function _prepare_draws(depvar_ref::AbstractVector{T}, N::Int, ::Type{T},
                        user_draws, n_draws::Int, multiRand::Bool,
                        distinct_Halton_length::Int=2^15-1) where {T}
    if isnothing(user_draws)
        if multiRand
            # Generate N x D wrapped Halton matrix (each observation gets different draws)
            halton_cpu = make_halton_wrap(N, n_draws; T=T, distinct_Halton_length=distinct_Halton_length)
            draws_2D = _to_device_array(depvar_ref, halton_cpu)
            draws_vec = vec(draws_2D)  # Store flattened for consistency
        else
            # Original behavior: 1D vector reshaped to 1 x D (all observations share draws)
            halton_cpu = make_halton_p(n_draws; T=T)
            draws_vec = _to_device_array(depvar_ref, halton_cpu)
            draws_2D = reshape(draws_vec, 1, length(draws_vec))
        end
    else
        # User provided draws - validate shape and convert
        depvar_is_gpu = string(typeof(depvar_ref).name.wrapper) == "CuArray"
        draws_is_gpu = string(typeof(user_draws).name.wrapper) == "CuArray"
        if depvar_is_gpu != draws_is_gpu
            @warn "Type inconsistency: `depvar` is on $(depvar_is_gpu ? "GPU" : "CPU") but `draws` is on $(draws_is_gpu ? "GPU" : "CPU"). " *
                  "Consider letting the program auto-generate Halton sequences with the correct device type by specifying `n_draws` instead of providing `draws`."
        end

        if multiRand
            # Expect N x D matrix
            if ndims(user_draws) == 2 && size(user_draws, 1) == N
                draws_2D = _to_device_array(depvar_ref, T.(user_draws))
                draws_vec = vec(draws_2D)
            elseif ndims(user_draws) == 1
                # User passed 1D vector but multiRand=true - generate wrapped matrix
                @warn "multiRand=true but `draws` is a 1D vector. Generating wrapped N x D matrix from `n_draws` instead."
                halton_cpu = make_halton_wrap(N, n_draws; T=T, distinct_Halton_length=distinct_Halton_length)
                draws_2D = _to_device_array(depvar_ref, halton_cpu)
                draws_vec = vec(draws_2D)
            else
                error("When multiRand=true, `draws` must be an N x D matrix with N=$N rows, got size $(size(user_draws))")
            end
        else
            # Original behavior: expect 1D vector
            draws_vec = T.(vec(user_draws))
            draws_2D = reshape(_to_device_array(depvar_ref, draws_vec), 1, length(draws_vec))
        end
    end

    return draws_vec, draws_2D
end

"""
    _assemble_MSLE_spec(spec::SFModelSpec_MSLE{T}, method::SFMethodSpec_MSLE)

Merge a model specification and method specification into the internal
`sfmodel_MSLE_spec{T}` struct. Performs deferred computations:
GPU conversion, draw generation, and constant computation.
"""
function _assemble_MSLE_spec(spec::SFModelSpec_MSLE{T}, method::SFMethodSpec_MSLE) where {T}
    # 1. GPU conversion (deferred from spec)
    depvar, frontier, zvar = _maybe_gpu_convert(spec.depvar, spec.frontier, spec.zvar, method.GPU)

    # 1b. GPU conversion for scaling_zvar (if scaling)
    scaling_zvar = if spec.scaling && method.GPU
        _to_device_array(depvar, spec.scaling_zvar)
    else
        spec.scaling_zvar
    end

    # 2. Prepare draws (depends on both spec.N and method.n_draws)
    draws_vec, draws_2D = _prepare_draws(depvar, spec.N, T,
        method.draws, method.n_draws, method.multiRand, method.distinct_Halton_length)

    # 3. Constants
    constants = make_constants(spec.model, T)

    # 4. Build internal struct
    return sfmodel_MSLE_spec{T}(
        depvar, frontier, zvar,
        spec.noise, spec.ineff, spec.copula, spec.hetero,
        draws_vec, draws_2D, method.multiRand, constants,
        spec.varnames, spec.eqnames, spec.eq_indices,
        spec.N, spec.K, spec.L, spec.model, spec.idx,
        spec.sign, method.chunks,
        spec.scaling, scaling_zvar, spec.L_scaling
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

theta_rho(x::Real, xs::Real...)       = ThetaRhoInit(_to_vec(x, xs...))
theta_rho(v::AbstractVector)          = ThetaRhoInit(_to_vec(v))
theta_rho(x::Real)                    = ThetaRhoInit(_to_vec(x))

scaling(x::Real, xs::Real...)         = ScalingInit(_to_vec(x, xs...))
scaling(v::AbstractVector)            = ScalingInit(_to_vec(v))
scaling(x::Real)                      = ScalingInit(_to_vec(x))


"""
    _get_init(spec::sfmodel_MSLE_spec)

Determine required init specifications based on the model specification.
Returns a vector of tuples: (Type, name_string, expected_length).
"""
function _get_init(spec::Union{sfmodel_MSLE_spec, SFModelSpec_MSLE})
    required = Vector{Tuple{Type, String, Int}}()

    # Helper: returns spec.L if symbol is heteroscedastic, otherwise 1
    hetero_len(sym::Symbol) = sym in spec.hetero ? spec.L : 1

    # Always need frontier (K parameters)
    push!(required, (FrontierInit, "frontier", spec.K))

    # Scaling delta coefficients (after frontier, before ineff)
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

    # Copula parameters (after noise, at the end)
    if spec.copula in (:Gaussian, :Clayton, :Clayton90, :Gumbel)
        push!(required, (ThetaRhoInit, "theta_rho", 1))
    end

    return required
end

"""
    _build_init(spec::sfmodel_MSLE_spec, init_dict, required)

Build the parameter vector in the correct order from the init specifications.
"""
function _build_init(spec::Union{sfmodel_MSLE_spec, SFModelSpec_MSLE}, init_dict, required)
    p = Float64[]
    for (req_type, _, _) in required
        append!(p, init_dict[req_type].values)
    end
    return p
end

"""
    sfmodel_MSLE_init(; spec, init=nothing, frontier=nothing, mu=nothing, ...)

Create an initial parameter vector for optimization based on user-specified initial values.
All arguments are keyword arguments. Initial values can be specified as vectors, row vectors, or tuples.

Two usage modes are supported:

1. **Full vector mode**: Supply the entire initial value vector directly via `init`.
2. **Component mode**: Supply individual components (frontier, mu, ln_sigma_sq, etc.).

# Arguments
- `spec::SFModelSpec_MSLE`: The model specification (required)
- `init`: Complete initial value vector (optional). If provided, other parameters are ignored.
         Can be vector, row vector, or tuple. Length must match the number of model parameters.
- `frontier`: Frontier coefficients (K values), can be vector, row vector, or tuple
- `mu`: Mu parameter for TruncatedNormal, Lognormal (if applicable)
- `ln_sigma_sq`: Log sigma squared for TruncatedNormal, HalfNormal, Lognormal, Rayleigh
- `ln_sigma_v_sq`: Log noise variance for Normal, StudentT
- `ln_nu_minus_2`: Log(nu-2) for StudentT
- `ln_b`: Log scale for Laplace
- `ln_lambda`: Log lambda for Exponential/Weibull
- `ln_k`: Log shape k for Weibull
- `ln_lambda`: Log lambda (scale) for Lomax/Exponential/Weibull
- `ln_alpha`: Log alpha for Lomax
- `message::Bool=true`: If true, issue a warning when `init` is provided along with other parameters.

# Examples
```julia
# Full vector mode: supply complete initial values
myinit = sfmodel_MSLE_init(
    spec = myspec,
    init = [0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0]
)

# Component mode: supply individual parameters
myinit = sfmodel_MSLE_init(
    spec = myspec,
    frontier = [0.5, 0.3, 0.2],   # K=3 coefficients (vector)
    mu = [0.1, 0.1, 0.1],         # L=3 with :mu hetero
    ln_sigma_sq = [0.0],          # scalar as single-element vector
    ln_sigma_v_sq = (0.0,)        # can also use tuple
)
# Row vectors are also supported: frontier = [0.5 0.3 0.2]
```
"""
function sfmodel_MSLE_init(;
    spec::SFModelSpec_MSLE,
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
    theta_rho = nothing,
    message::Bool = true
)
    # Internal aliases: user-facing names use ln_ prefix, internal code uses bare names
    lambda = ln_lambda
    k = ln_k
    alpha = ln_alpha

    # Mode 1: Full vector mode - user supplies complete initial value vector
    if init !== nothing
        # Check if any component-specific parameters were also provided
        if message && any(x -> x !== nothing, (frontier, scaling, mu, ln_sigma_sq, ln_sigma_v_sq,
                                                ln_nu_minus_2, ln_b, lambda, k, alpha, theta_rho))
            @warn "Using `init` instead of function-specific init."
        end
        init_vec = _to_init_vec(init)
        expected_len = plen(spec.model, spec.K, spec.L, spec.hetero; L_scaling=spec.L_scaling)
        if length(init_vec) != expected_len
            error("Length mismatch in sfmodel_MSLE_init(): expected $expected_len parameters, got $(length(init_vec)).")
        end
        return init_vec
    end

    # Mode 2: Component mode - user supplies individual parameters
    if frontier === nothing
        # Default to OLS estimates for frontier coefficients
        frontier = spec.frontier \ spec.depvar
        if message
            @info "Using OLS estimates as default initial values for frontier coefficients."
        end
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
    if theta_rho_vec !== nothing
        init_dict[ThetaRhoInit] = ThetaRhoInit(theta_rho_vec)
    end

    # Determine required equations based on spec
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
        error("Missing required init(s) in sfmodel_MSLE_init(): $missing_list.")
    end

    # Report all length mismatches at once
    if !isempty(length_mismatches)
        mismatch_list = join(length_mismatches, "; ")
        error("Length mismatch(es) in sfmodel_MSLE_init(): $mismatch_list.")
    end

    # Check for extra (unused) specifications
    provided_types = Set(keys(init_dict))
    required_types = Set(r[1] for r in required)
    extras = setdiff(provided_types, required_types)
    if !isempty(extras)
        extra_names = join([string(e) for e in extras], ", ")
        error("Unused specification(s) in sfmodel_MSLE_init(): $extra_names. These are not required for the current model.")
    end

    # Build parameter vector in correct order
    return _build_init(spec, init_dict, required)
end

# ============================================================================
# End of Initial Value Specification
# ============================================================================

# ============================================================================
# Optimization Options Specification (sfmodel_MSLE_opt)
# ============================================================================

"""
    sfmodel_MSLE_optim

Struct holding all optimization settings for `sfmodel_MSLE_fit()`.

# Fields
- `warmstart_solver`: Warmstart solver (e.g., NelderMead()). If nothing, warmstart is skipped.
- `warmstart_opt::Union{Nothing, Optim.Options}`: Warmstart Optim.Options
- `main_solver`: Main solver (e.g., Newton())
- `main_opt::Optim.Options`: Main Optim.Options
"""
struct sfmodel_MSLE_optim
    warmstart_solver::Any
    warmstart_opt::Union{Nothing, Optim.Options}
    main_solver::Any
    main_opt::Optim.Options
end

"""
    sfmodel_MSLE_opt(; warmstart_solver=nothing, warmstart_opt=nothing,
                      main_solver, main_opt)

Construct optimization options for `sfmodel_MSLE_fit()`.

# Arguments
- `warmstart_solver=nothing`: Warmstart optimizer, e.g., `NelderMead()`, `BFGS()`. Optional.
- `warmstart_opt=nothing`: Warmstart options as a NamedTuple, e.g., `(iterations = 400, g_abstol = 1e-5)`. Optional.
- `main_solver`: Main optimizer, e.g., `Newton()`, `BFGS()`. Required.
- `main_opt`: Main options as a NamedTuple, e.g., `(iterations = 2000, g_abstol = 1e-8)`. Required.

If `warmstart_solver` is not provided, the warmstart stage will be skipped.

# Example
```julia
# With warmstart
myopt = sfmodel_MSLE_opt(
    warmstart_solver = NelderMead(),
    warmstart_opt = (iterations = 400, g_abstol = 1e-5),
    main_solver = Newton(),
    main_opt = (iterations = 2000, g_abstol = 1e-8)
)

# Without warmstart (skip directly to main optimization)
myopt = sfmodel_MSLE_opt(
    main_solver = Newton(),
    main_opt = (iterations = 2000, g_abstol = 1e-8)
)
```
"""
function sfmodel_MSLE_opt(;
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

    return sfmodel_MSLE_optim(warmstart_solver, ws_opt, main_solver, m_opt)
end

# ============================================================================
# End of Optimization Options Specification
# ============================================================================

"""
    _default_eq_names(model::MSLEModel, hetero::Vector{Symbol})

Generate default equation names based on model type and heteroscedasticity settings.
"""
function _default_eq_names(model::MSLEModel, hetero::Vector{Symbol}; scaling::Bool=false)
    eqnames = ["frontier"]

    # Add scaling equation name (before ineff equations)
    if scaling
        push!(eqnames, "scaling")
    end

    # Add inefficiency equation names based on model type
    ineff = model.ineff
    if ineff isa TruncatedNormal_MSLE
        push!(eqnames, :mu in hetero ? "μ" : "μ")
        push!(eqnames, :sigma_sq in hetero ? "ln_σᵤ²" : "ln_σᵤ²")
    elseif ineff isa Exponential_MSLE
        push!(eqnames, :lambda in hetero ? "ln_λ" : "ln_λ")
    elseif ineff isa HalfNormal_MSLE
        push!(eqnames, :sigma_sq in hetero ? "ln_σᵤ²" : "ln_σᵤ²")
    elseif ineff isa Weibull_MSLE
        push!(eqnames, :lambda in hetero ? "ln_λ" : "ln_λ")
        push!(eqnames, :k in hetero ? "ln_k" : "ln_k")
    elseif ineff isa Lognormal_MSLE
        push!(eqnames, :mu in hetero ? "μ" : "μ")
        push!(eqnames, :sigma_sq in hetero ? "ln_σ²" : "ln_σ²")
    elseif ineff isa Lomax_MSLE
        push!(eqnames, :lambda in hetero ? "ln_λ" : "ln_λ")
        push!(eqnames, :alpha in hetero ? "ln_α" : "ln_α")
    elseif ineff isa Rayleigh_MSLE
        push!(eqnames, :sigma_sq in hetero ? "ln_σ²" : "ln_σ²")
    end

    # Add noise equation names
    noise = model.noise
    if noise isa NormalNoise_MSLE
        push!(eqnames, "ln_σᵥ²")
    elseif noise isa StudentTNoise_MSLE
        push!(eqnames, "ln_σᵥ²")
        push!(eqnames, "ln_ν₋₂")
    elseif noise isa LaplaceNoise_MSLE
        push!(eqnames, "ln_b")
    end

    # Add copula equation names
    if copula_plen(model.copula) > 0
        push!(eqnames, "θ_ρ")
    end

    return eqnames
end

"""
    _default_eq_ind(model::MSLEModel, K::Int, L::Int, hetero::Vector{Symbol})

Generate default equation indices based on model type and heteroscedasticity settings.
"""
function _default_eq_ind(model::MSLEModel, K::Int, L::Int, hetero::Vector{Symbol}; scaling::Bool=false, L_scaling::Int=0)
    eq_indices = [1]  # frontier starts at 1
    idx = K + 1       # after beta

    # Add scaling equation index (before ineff equations)
    if scaling
        push!(eq_indices, idx)  # δ
        idx += L_scaling
    end

    # Add inefficiency equation indices based on model type
    ineff = model.ineff
    if ineff isa TruncatedNormal_MSLE
        push!(eq_indices, idx)  # μ
        n_mu = :mu in hetero ? L : 1
        idx += n_mu
        push!(eq_indices, idx)  # ln_σᵤ²
        n_sigma = :sigma_sq in hetero ? L : 1
        idx += n_sigma
    elseif ineff isa Exponential_MSLE
        push!(eq_indices, idx)  # ln_λ
        n_lambda = :lambda in hetero ? L : 1
        idx += n_lambda
    elseif ineff isa HalfNormal_MSLE
        push!(eq_indices, idx)  # ln_σᵤ²
        n_sigma = :sigma_sq in hetero ? L : 1
        idx += n_sigma
    elseif ineff isa Weibull_MSLE
        push!(eq_indices, idx)  # ln_λ
        n_lambda = :lambda in hetero ? L : 1
        idx += n_lambda
        push!(eq_indices, idx)  # ln_k
        n_k = :k in hetero ? L : 1
        idx += n_k
    elseif ineff isa Lognormal_MSLE
        push!(eq_indices, idx)  # μ
        n_mu = :mu in hetero ? L : 1
        idx += n_mu
        push!(eq_indices, idx)  # ln_σ²
        n_sigma = :sigma_sq in hetero ? L : 1
        idx += n_sigma
    elseif ineff isa Lomax_MSLE
        push!(eq_indices, idx)  # ln_λ
        n_lambda = :lambda in hetero ? L : 1
        idx += n_lambda
        push!(eq_indices, idx)  # ln_α
        n_alpha = :alpha in hetero ? L : 1
        idx += n_alpha
    elseif ineff isa Rayleigh_MSLE
        push!(eq_indices, idx)  # ln_σ²
        n_sigma = :sigma_sq in hetero ? L : 1
        idx += n_sigma
    end

    # Add noise equation indices
    noise = model.noise
    push!(eq_indices, idx)  # ln_σᵥ² or ln_b
    idx += 1
    if noise isa StudentTNoise_MSLE
        push!(eq_indices, idx)  # ln_ν₋₂
        idx += 1
    end

    # Add copula equation indices
    if copula_plen(model.copula) > 0
        push!(eq_indices, idx)  # θ_ρ
        idx += copula_plen(model.copula)
    end

    return eq_indices
end

# ============================================================================
# Section 2b: DSL Macros for DataFrame-based Specification
# ============================================================================

"""
Marker types for DSL-style model specification.
These types hold the parsed macro arguments until they're processed by sfmodel_spec.
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
    _gen_names(model::MSLEModel, K::Int, L::Int, hetero::Vector{Symbol},
                         idx::NamedTuple, varnames, eqnames, eq_indices)

Auto-generate variable names, equation names, and equation indices if not provided.
"""
function _gen_names(model::MSLEModel, K::Int, L::Int, hetero::Vector{Symbol},
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

"""
    sfmodel_spec(; depvar, frontier, zvar=nothing, noise, ineff, hetero=Symbol[],
                 varnames=nothing, eqnames=nothing, eq_indices=nothing, type=:prod)

Construct a model specification (what to estimate). Returns `SFModelSpec_MSLE{T}`.

Numerical method settings (draws, GPU, chunks) are specified separately via `sfmodel_method()`.

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
- `varnames=nothing`: Variable names (auto-generated as x1, x2, ... if not provided)
- `eqnames=nothing`: Equation names (auto-generated based on model if not provided)
- `eq_indices=nothing`: Equation indices (auto-generated based on model if not provided)
- `type::Symbol=:prod`: Frontier type (:prod, :production, or :cost)

# Example
```julia
# Homoscedastic model (zvar and hetero auto-default)
spec = sfmodel_spec(depvar=y, frontier=X, noise=:Normal, ineff=:HalfNormal)

# Heteroscedastic model with zvar
spec = sfmodel_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])

# Full specification with custom names
spec = sfmodel_spec(
    depvar = y, frontier = X, zvar = Z,
    noise = :Normal, ineff = :TruncatedNormal, hetero = [:mu],
    varnames = ["_cons", "x1", "x2", "_cons", "_cons"],
    eqnames = ["frontier", "μ", "ln_σᵤ²", "ln_σᵥ²"],
    eq_indices = [1, 4, 5, 6]
)
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
    N = length(depvar_norm)

    # Handle scaling property model
    if hetero === :scaling
        scaling = true
        hetero_vec = Symbol[]

        # zvar is required for scaling
        isnothing(zvar) && error("Scaling property model (`hetero=:scaling`) requires `zvar` to be provided.")
        zvar_norm = _to_matrix(zvar)
        zvar_norm = T.(zvar_norm)

        # Validate no constant column in scaling zvar
        for j in 1:size(zvar_norm, 2)
            if _is_constant_column(@view zvar_norm[:, j])
                error("Scaling function `zvar` must NOT contain a constant column. " *
                      "Column $j is constant. Remove it for parameter identification. " *
                      "See '00 Scaling Property.md' Section 6.")
            end
        end

        scaling_zvar = zvar_norm
        L_scaling = size(scaling_zvar, 2)
        zvar_norm = ones(T, N, 1)  # ones(N,1) for homoscedastic ineff params
    elseif hetero isa Symbol
        error("Invalid `hetero`: :$hetero. Use a Vector{Symbol} like [:mu], or use :scaling for the scaling property model.")
    else
        scaling = false
        hetero_vec = hetero
        scaling_zvar = nothing
        L_scaling = 0

        # Auto-generate zvar as ones(N) for homoscedastic models when not provided
        if isnothing(zvar)
            zvar = ones(T, N)
        end
        zvar_norm = _to_matrix(zvar)
        zvar_norm = T.(zvar_norm)
    end

    # Convert frontier to match T if needed
    frontier_norm = T.(frontier_norm)

    # No GPU conversion here — deferred to _assemble_MSLE_spec

    # Validate and compute sign for frontier type
    frontier_sign = if type in (:prod, :production)
        1
    elseif type == :cost
        -1
    else
        error("Invalid `type` in sfmodel_spec(): $type. Use :prod, :production, or :cost.")
    end

    K, L = size(frontier_norm, 2), size(zvar_norm, 2)

    # Build model and validate
    model = _build_model(noise, ineff; copula=copula)
    _validate_hetero(model.ineff, hetero_vec)

    # Validate varnames length if provided
    if !isnothing(varnames)
        expected_len = plen(model, K, L, hetero_vec; L_scaling=L_scaling)
        if length(varnames) != expected_len
            error("Length of `varnames` ($(length(varnames))) does not match " *
                  "the number of parameters ($expected_len). " *
                  "Expected $expected_len names for: frontier ($K) + " *
                  (scaling ? "scaling ($L_scaling) + " : "") *
                  "inefficiency + noise + copula parameters.")
        end
    end

    # Compute parameter indices
    idx = _param_ind(model, K, L, hetero_vec; L_scaling=L_scaling)

    # Auto-generate varnames/eqnames/eq_indices if not provided
    varnames_vec, eqnames_vec, eq_indices_vec = _gen_names(
        model, K, L, hetero_vec, idx, varnames, eqnames, eq_indices;
        scaling=scaling, L_scaling=L_scaling
    )

    return SFModelSpec_MSLE{T}(depvar_norm, frontier_norm, zvar_norm, noise, ineff, copula, hetero_vec,
                     varnames_vec, eqnames_vec, eq_indices_vec,
                     N, K, L, model, idx, frontier_sign,
                     scaling, scaling_zvar, L_scaling)
end

"""
    sfmodel_spec(data_spec, depvar_spec, frontier_spec, zvar_spec; ...)

DSL-style model specification using macros. Automatically extracts data from DataFrame
and generates variable names from column names. Returns `SFModelSpec_MSLE{T}`.

# Arguments
- `data_spec::UseDataSpec`: DataFrame wrapped by `@useData(df)`
- `depvar_spec::DepvarSpec`: Dependent variable name from `@depvar(varname)`
- `frontier_spec::FrontierSpec`: Frontier variable names from `@frontier(var1, var2, ...)`
- `zvar_spec::ZvarSpec`: Z variable names from `@zvar(var1, var2, ...)`
- Other keyword arguments: same as the matrix-input version (excluding method-level kwargs)

# Example
```julia
# Prepare DataFrame with a constant column
df._cons = ones(nrow(df))

# DSL-style specification
spec = sfmodel_spec(
    @useData(df),
    @depvar(yvar),
    @frontier(_cons, Lland, PIland, Llabor, Lbull, Lcost, yr),
    @zvar(_cons, age, school, yr),
    noise = :Normal,
    ineff = :TruncatedNormal,
    hetero = [:mu, :sigma_sq]
)
# Variable names auto-extracted: ["_cons", "Lland", "PIland", ..., "_cons", "age", "school", "yr"]
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

    # Compute total number of parameters to build full varnames
    K = length(frontier_spec.names)
    L = length(zvar_spec.names)

    # Handle scaling property model
    is_scaling = (hetero === :scaling)
    hetero_vec = is_scaling ? Symbol[] : (hetero isa Symbol ? error("Invalid `hetero`: :$hetero.") : hetero)

    # Build model to determine parameter count
    model = _build_model(noise, ineff; copula=copula)
    if !is_scaling
        _validate_hetero(model.ineff, hetero_vec)
    end
    L_scaling_local = is_scaling ? L : 0
    L_ineff = is_scaling ? 1 : L  # L=1 for homoscedastic ineff params when scaling
    n_params = plen(model, K, L_ineff, hetero_vec; L_scaling=L_scaling_local)

    # Build varnames based on equation structure
    varnames = Vector{String}(undef, n_params)

    # Frontier equation (K params)
    varnames[1:K] = frontier_names
    idx = K + 1

    # Scaling delta equation (if scaling)
    if is_scaling
        for name in zvar_names
            varnames[idx] = name
            idx += 1
        end
    end

    # Inefficiency distribution parameters (before noise, to match _default_eq_ind)
    ineff_type = model.ineff
    if ineff_type isa TruncatedNormal_MSLE
        if !is_scaling && :mu in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "μ"; idx += 1
        end
        if !is_scaling && :sigma_sq in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "ln_σᵤ²"; idx += 1
        end
    elseif ineff_type isa HalfNormal_MSLE
        if !is_scaling && :sigma_sq in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "ln_σᵤ²"; idx += 1
        end
    elseif ineff_type isa Exponential_MSLE
        if !is_scaling && :lambda in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "ln_λ"; idx += 1
        end
    elseif ineff_type isa Lognormal_MSLE
        if !is_scaling && :mu in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "μ"; idx += 1
        end
        if !is_scaling && :sigma_sq in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "ln_σ²"; idx += 1
        end
    elseif ineff_type isa Weibull_MSLE
        if !is_scaling && :lambda in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "ln_λ"; idx += 1
        end
        if !is_scaling && :k in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "ln_k"; idx += 1
        end
    elseif ineff_type isa Lomax_MSLE
        if !is_scaling && :lambda in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "ln_λ"; idx += 1
        end
        if !is_scaling && :alpha in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "ln_α"; idx += 1
        end
    elseif ineff_type isa Rayleigh_MSLE
        if !is_scaling && :sigma_sq in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "ln_σ²"; idx += 1
        end
    end

    # Noise equation
    if noise == :Laplace
        varnames[idx] = "ln_b"
    else
        varnames[idx] = "ln_σᵥ²"
    end
    idx += 1

    if noise == :StudentT
        varnames[idx] = "ln_ν₋₂"; idx += 1
    end

    # Copula parameter
    if copula != :None
        varnames[idx] = "θ_ρ"; idx += 1
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
    ineff = :HalfNormal,
    copula = :Gaussian
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
    df = data_spec.df

    # Extract depvar from DataFrame
    depvar = Vector{Float64}(df[!, depvar_spec.name])

    # Build frontier matrix
    frontier_names = [String(name) for name in frontier_spec.names]
    frontier = hcat([Vector{Float64}(df[!, name]) for name in frontier_spec.names]...)

    # Scaling requires zvar
    hetero === :scaling && error("Scaling property model (`hetero=:scaling`) requires `@zvar(...)`. " *
                                 "Use the 4-argument form with @zvar.")

    # No zvar provided → call keyword version with zvar=nothing (auto-generates ones(N))
    # Pass frontier varnames so the output table has meaningful column names
    K = length(frontier_spec.names)
    model = _build_model(noise, ineff; copula=copula)
    hetero_vec = hetero isa Symbol ? Symbol[] : hetero
    _validate_hetero(model.ineff, hetero_vec)
    n_params = plen(model, K, 1, hetero_vec)  # L=1 for homoscedastic

    # Build varnames: frontier names + scalar param names for non-frontier equations
    varnames = Vector{String}(undef, n_params)
    varnames[1:K] = frontier_names
    vi = K + 1

    # Inefficiency parameters (scalar for each since no hetero zvar)
    ineff_type = model.ineff
    if ineff_type isa TruncatedNormal_MSLE
        varnames[vi] = :mu in hetero ? "_cons" : "μ";  vi += 1
        varnames[vi] = :sigma_sq in hetero ? "_cons" : "ln_σᵤ²";  vi += 1
    elseif ineff_type isa HalfNormal_MSLE
        varnames[vi] = :sigma_sq in hetero ? "_cons" : "ln_σᵤ²";  vi += 1
    elseif ineff_type isa Exponential_MSLE
        varnames[vi] = :lambda in hetero ? "_cons" : "ln_λ";  vi += 1
    elseif ineff_type isa Lognormal_MSLE
        varnames[vi] = :mu in hetero ? "_cons" : "μ";  vi += 1
        varnames[vi] = :sigma_sq in hetero ? "_cons" : "ln_σ²";  vi += 1
    elseif ineff_type isa Weibull_MSLE
        varnames[vi] = :lambda in hetero ? "_cons" : "ln_λ";  vi += 1
        varnames[vi] = :k in hetero ? "_cons" : "ln_k";  vi += 1
    elseif ineff_type isa Lomax_MSLE
        varnames[vi] = :lambda in hetero ? "_cons" : "ln_λ";  vi += 1
        varnames[vi] = :alpha in hetero ? "_cons" : "ln_α";  vi += 1
    elseif ineff_type isa Rayleigh_MSLE
        varnames[vi] = :sigma_sq in hetero ? "_cons" : "ln_σ²";  vi += 1
    end

    # Noise parameters
    if noise == :Laplace
        varnames[vi] = "ln_b";  vi += 1
    else
        varnames[vi] = "ln_σᵥ²";  vi += 1
    end
    if noise == :StudentT
        varnames[vi] = "ln_ν₋₂";  vi += 1
    end

    # Copula parameters
    if copula != :None
        varnames[vi] = "θ_ρ";  vi += 1
    end

    return sfmodel_spec(; depvar=depvar, frontier=frontier, zvar=nothing,
                             noise=noise, ineff=ineff, copula=copula, hetero=hetero,
                             varnames=varnames,
                             eqnames=eqnames, eq_indices=eq_indices,
                             type=type)
end

"""
    sfmodel_method(; method=:MSLE, draws=nothing, n_draws=1024, multiRand=true, GPU=false, chunks=10, distinct_Halton_length=2^15-1)

Construct a method specification (how to estimate). Returns `SFMethodSpec_MSLE`.

# Arguments
- `method::Symbol=:MSLE`: Estimation method (currently only `:MSLE` supported)
- `draws=nothing`: User-provided Halton draws (auto-generated if `nothing`)
- `n_draws::Int=1024`: Number of Halton draws (used if `draws` not provided)
- `multiRand::Bool=true`: Per-observation draws (N×D) if `true`, shared draws (1×D) if `false`
- `GPU::Bool=false`: If `true`, convert data to GPU arrays. Requires `using CUDA` before calling.
- `chunks::Int=10`: Number of chunks for GPU memory management
- `distinct_Halton_length::Int=2^15-1`: Maximum Halton sequence length for multiRand mode (default 32767)

# Example
```julia
# Default method (MSLE with 1024 Halton draws, CPU)
method = sfmodel_method()

# Custom method (more draws, GPU enabled)
method = sfmodel_method(n_draws=4095, GPU=true)

# Larger distinct_Halton_length pool for multiRand mode
method = sfmodel_method(n_draws=50000, distinct_Halton_length=2^16-1)
```
"""
function sfmodel_method(;
    method::Symbol = :MSLE,
    draws::Union{Nothing, AbstractVecOrMat{<:AbstractFloat}} = nothing,
    n_draws::Int = 1024,
    multiRand::Bool = true,
    GPU::Bool = false,
    chunks::Int = 10,
    distinct_Halton_length::Int = 2^15-1)

    method == :MSLE || error("Currently only method=:MSLE is supported. Got: :$method")
    return SFMethodSpec_MSLE(method, draws, n_draws, multiRand, GPU, chunks, distinct_Halton_length)
end


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
Q(p) = σ√2 · erfinv(p)

Note: For HalfNormal, p ∈ [0, 1) maps to u ∈ [0, ∞).
"""
function myHalfNormalQuantile!(out; σ, r, sqrt2, clamp_lo, clamp_hi)
    @. out = σ * sqrt2 * erfinv(clamp(r, clamp_lo, clamp_hi))
    return out
end

"""
    myWeibullQuantile!(out; λ, k, r, clamp_lo)

Compute quantile of Weibull(λ, k) at probability r.
Q(p; λ, k) = λ · (-ln(1-p))^(1/k)

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
Q(p; μ, σ) = exp(μ + σ√2 · erfinv(2p-1))

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

Q(p; α, λ) = λ * ((1-p)^(-1/α) - 1)

# Arguments
- `λ`: Scale parameter
- `α`: Shape parameter
- `r`: Probability values (MSLE draws)
- `clamp_lo`: Lower bound for (1-r) to prevent log(0)
- `out_ceil`: Upper bound for quantile output
"""
function myLomaxQuantile!(out; λ, α, r, clamp_lo, out_ceil)
    one_minus_r = max.(1 .- r, clamp_lo)
    @. out = clamp(λ * expm1((-1 / α) * log(one_minus_r)), 0, out_ceil)
    return out
end

"""
    myRayleighQuantile!(out; σ, r, clamp_lo)

Compute quantile of Rayleigh(σ) at probability r.
Q(p; σ) = σ · √(-2·ln(1-p))

# Arguments
- `σ`: Scale parameter
"""
function myRayleighQuantile!(out; σ, r, clamp_lo)
    @. out = σ * sqrt(-2 * log(max(1 - r, clamp_lo)))
    return out
end


# ============================================================================
# Section 4: PDF Functions
# ============================================================================

"""
    myNormalPDF(; z, σ, sqrt2=sqrt(2), sqrt_pi=sqrt(π))

Compute PDF of Normal(0, σ) at points z.
Returns: 1/(σ√(2π)) × exp(-0.5 × (z/σ)²)
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
f(x; σ_v, ν) = Γ((ν+1)/2) / (σ_v√(νπ) Γ(ν/2)) × (1 + (x/σ_v)²/ν)^(-(ν+1)/2)

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
- `log_const`: Precomputed log(Γ((ν+1)/2)) - log(Γ(ν/2)) - 0.5*log(νπ) - log(σ_v)
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
    return @. exp(-abs(z) / b) / (2 * b)
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
       @. out = inv_2b * exp(-abs(z) * inv_b)
    else
       @. out = exp(-abs(z) / b) / (2 * b)
    end
    return out
end

# ============================================================================
# Section 5: Constants and Utilities
# ============================================================================

"""
    make_halton_p(draws::Int; base::Int=2, T::Type{<:AbstractFloat}=Float64)

Generate Halton probabilities in (0,1) for MSLE integration.
Call this ONCE outside the likelihood function and reuse.
"""
function make_halton_p(draws::Int; base::Int=2, T::Type{<:AbstractFloat}=Float64)
    draws > 0 || throw(ArgumentError("draws must be positive, got $draws"))
    return T.(collect(Halton(base, length=draws)))
end

"""
    make_halton_wrap(N::Int, D::Int; base::Int=2, T::Type{<:AbstractFloat}=Float64)

Generate wrapped Halton matrix where each observation gets different consecutive draws.
Returns an N x D matrix where row i contains consecutive Halton elements, wrapping around
when the sequence runs out.

This implements observation-specific draws: each observation uses a DIFFERENT
consecutive segment of the Halton sequence, providing greater variation across
observations compared to the standard approach where all observations share the same draws.

# Algorithm
1. Validate D ≤ `distinct_Halton_length` (default 2^15 - 1 = 32767)
2. Compute n = floor(log2(D*N + 1)), capped at floor(log2(distinct_Halton_length + 1))
3. Generate Halton sequence of length 2^n - 1
4. Recycle sequence to fill D*N elements if needed
5. Reshape to N x D matrix (each row = one observation's draws)

# Arguments
- `N::Int`: Number of observations
- `D::Int`: Number of draws per observation (must be ≤ `distinct_Halton_length`)
- `base::Int=2`: Base for Halton sequence (default: 2)
- `T::Type{<:AbstractFloat}=Float64`: Element type
- `distinct_Halton_length::Int=2^15-1`: Maximum Halton sequence length (controls the cap on sequence generation)

# Returns
- `Matrix{T}` of size (N, D) where each row contains different consecutive Halton values

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
    make_constants(model::MSLEModel, T::Type{<:AbstractFloat}=Float64)

Precompute invariant constants for the given model. Call this ONCE before optimization.
"""
function make_constants(model::MSLEModel, T::Type{<:AbstractFloat}=Float64)
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

make_noise_constants(::NormalNoise_MSLE, T) = NamedTuple()
make_noise_constants(::StudentTNoise_MSLE, T) = NamedTuple()
make_noise_constants(::LaplaceNoise_MSLE, T) = NamedTuple()

make_ineff_constants(::TruncatedNormal_MSLE, T) = (
    lo_erf = -one(T) + T(32) * eps(T),
    hi_erf =  one(T) - T(32) * eps(T),
)
make_ineff_constants(::Exponential_MSLE, T) = (
    exp_clamp = T(1e-15),
)
make_ineff_constants(::HalfNormal_MSLE, T) = (
    lo_erf = T(32) * eps(T),           # erfinv domain: [0, 1)
    hi_erf = one(T) - T(32) * eps(T),
)
make_ineff_constants(::Weibull_MSLE, T) = (
    clamp_lo = T(1e-15),  # For log(1-r)
    k_floor = T(0.1),     # Min shape parameter (prevents 1/k explosion)
    k_ceil = T(10.0),     # Max shape parameter (reasonable upper bound)
)
make_ineff_constants(::Lognormal_MSLE, T) = (
    lo_erf = -one(T) + T(32) * eps(T),  # erfinv(2r-1) domain: (-1, 1)
    hi_erf = one(T) - T(32) * eps(T),
)
make_ineff_constants(::Lomax_MSLE, T) = (
    clamp_lo = T(1e-15),
    α_floor = T(0.1),
    α_ceil = T(100.0),
    lambda_floor = T(1e-10),
    lambda_ceil = T(1e10),
    out_ceil = T(1e15),
)
make_ineff_constants(::Rayleigh_MSLE, T) = (
    clamp_lo = T(1e-15),  # For log(1-r)
)

make_copula_constants(::NoCopula_MSLE, T) = NamedTuple()
make_copula_constants(::GaussianCopula_MSLE, T) = (
    copula_clamp_lo = T(1e-15),                  #! hjw: maybe 1e-6? Clamp F_v and F_u away from 0
    copula_clamp_hi = one(T) - T(1e-15),         #! hjw: maybe 1e-6? Clamp F_v and F_u away from 1
    lo_erfinv = -one(T) + T(32) * eps(T),       # erfinv domain: (-1, 1)
    hi_erfinv =  one(T) - T(32) * eps(T),
    rho_max = T(0.999),                         # Clamp |ρ| < 0.999 to prevent 1-ρ² underflow
)
make_copula_constants(::ClaytonCopula_MSLE, T) = (
    copula_clamp_lo = T(1e-6),                  # Clamp F_v, F_u away from 0 (u^(-ρ) blows up)
    copula_clamp_hi = one(T) - T(1e-6),         # Clamp F_v, F_u away from 1
    clayton_rho_floor = T(1e-6),                 # Floor ρ away from 0 to prevent -1/ρ → -Inf
    clayton_rho_max = T(50.0),                   # Ceiling ρ to prevent u^(-ρ) overflow
)
make_copula_constants(::GumbelCopula_MSLE, T) = (
    copula_clamp_lo = T(1e-16),                  # Clamp F_v, F_u away from 0 (-log(u) → ∞)
    copula_clamp_hi = one(T) - T(1e-16),         # Clamp F_v, F_u away from 1
    gumbel_sum_floor = T(1e-16),                 # Floor for w₁^ρ + w₂^ρ to avoid 0^(negative)
    gumbel_rho_max = T(50.0),                    # Ceiling ρ to prevent w^ρ overflow
)
make_copula_constants(::Clayton90Copula_MSLE, T) = (
    copula_clamp_lo = T(1e-6),                   # Same constants as standard Clayton
    copula_clamp_hi = one(T) - T(1e-6),
    clayton_rho_floor = T(1e-6),
    clayton_rho_max = T(50.0),
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

function get_noise_vals(::NormalNoise_MSLE, p, idx, c)
    P = eltype(p)
    sigma_v = clamp(exp(P(0.5) * p[idx.noise.ln_sigma_v_sq]), c.σ_floor, c.σ_ceil)
    inv_sigma_v = inv(sigma_v)
    pdf_const = inv(sigma_v * c.sqrt2 * c.sqrt_pi)
    return (sigma_v=sigma_v, inv_sigma_v=inv_sigma_v, pdf_const=pdf_const)
end

function get_noise_vals(::StudentTNoise_MSLE, p, idx, c)
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

function get_noise_vals(::LaplaceNoise_MSLE, p, idx, c)
    P = eltype(p)
    b = clamp(exp(p[idx.noise.ln_b]), c.σ_floor, c.σ_ceil)
    inv_b = inv(b)
    inv_2b = inv(P(2) * b)
    return (b=b, inv_b=inv_b, inv_2b=inv_2b)
end

# --- Inefficiency parameter computation ---
# New signature: (ineff_type, p, idx, Z, c) using broadcasting instead of mul!
# Broadcasting pattern: sum(P(p[idx[j]]) .* (@view Z[:, j]) for j in 1:k)

function get_ineff_vals(::TruncatedNormal_MSLE, p, idx, Z, c)
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

function get_ineff_vals(::Exponential_MSLE, p, idx, Z, c)
    P = eltype(p)
    ineff = idx.ineff

    # λ = exp(Z * gamma), where λ = Var(u)
    k_lambda = length(ineff.lambda)
    lambda = exp.(sum(P(p[ineff.lambda[j]]) .* (@view Z[:, j]) for j in 1:k_lambda))
    lambda = clamp.(lambda, c.σ_floor, c.σ_ceil)

    return (lambda=lambda,)
end

function get_ineff_vals(::HalfNormal_MSLE, p, idx, Z, c)
    P = eltype(p)
    ineff = idx.ineff

    # σ = exp(0.5 * Z * gamma)
    k_sigma = length(ineff.sigma_sq)
    sigma = exp.(P(0.5) .* sum(P(p[ineff.sigma_sq[j]]) .* (@view Z[:, j]) for j in 1:k_sigma))
    sigma = clamp.(sigma, c.σ_floor, c.σ_ceil)

    return (sigma=sigma,)
end

function get_ineff_vals(::Weibull_MSLE, p, idx, Z, c)
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

function get_ineff_vals(::Lognormal_MSLE, p, idx, Z, c)
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

function get_ineff_vals(::Lomax_MSLE, p, idx, Z, c)
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

function get_ineff_vals(::Rayleigh_MSLE, p, idx, Z, c)
    P = eltype(p)
    ineff = idx.ineff

    # σ = exp(0.5 * Z * gamma)
    k_sigma = length(ineff.sigma_sq)
    sigma = exp.(P(0.5) .* sum(P(p[ineff.sigma_sq[j]]) .* (@view Z[:, j]) for j in 1:k_sigma))
    sigma = clamp.(sigma, c.σ_floor, c.σ_ceil)

    return (sigma=sigma,)
end

# --- Quantile computation (generates inefficiency draws) ---

function get_u_quantile!(::TruncatedNormal_MSLE, buffer, ineff_vals, draws_1D, c, N)
    mu_N1 = reshape(ineff_vals.mu, N, 1)
    sigma_N1 = reshape(ineff_vals.sigma_u, N, 1)

    myTruncatedNormalQuantile!(buffer; μ=mu_N1, σ=sigma_N1, r=draws_1D,
                               sqrt2=c.sqrt2, inv_sqrt2=c.inv_sqrt2,
                               clamp_lo=c.lo_erf, clamp_hi=c.hi_erf)
end

function get_u_quantile!(::Exponential_MSLE, buffer, ineff_vals, draws_1D, c, N)
    lambda_N1 = reshape(ineff_vals.lambda, N, 1)

    myExponentialQuantile!(buffer; λ=lambda_N1, r=draws_1D, clamp_lo=c.exp_clamp)
end

function get_u_quantile!(::HalfNormal_MSLE, buffer, ineff_vals, draws_1D, c, N)
    sigma_N1 = reshape(ineff_vals.sigma, N, 1)

    myHalfNormalQuantile!(buffer; σ=sigma_N1, r=draws_1D,
                          sqrt2=c.sqrt2, clamp_lo=c.lo_erf, clamp_hi=c.hi_erf)
end

function get_u_quantile!(::Weibull_MSLE, buffer, ineff_vals, draws_1D, c, N)
    lambda_N1 = reshape(ineff_vals.lambda, N, 1)
    k_N1 = reshape(ineff_vals.k, N, 1)

    myWeibullQuantile!(buffer; λ=lambda_N1, k=k_N1, r=draws_1D, clamp_lo=c.clamp_lo)
end

function get_u_quantile!(::Lognormal_MSLE, buffer, ineff_vals, draws_1D, c, N)
    mu_N1 = reshape(ineff_vals.mu, N, 1)
    sigma_N1 = reshape(ineff_vals.sigma, N, 1)

    myLognormalQuantile!(buffer; μ=mu_N1, σ=sigma_N1, r=draws_1D,
                         sqrt2=c.sqrt2, clamp_lo=c.lo_erf, clamp_hi=c.hi_erf)
end

function get_u_quantile!(::Lomax_MSLE, buffer, ineff_vals, draws_1D, c, N)
    lambda_N1 = reshape(ineff_vals.lambda, N, 1)
    alpha_N1 = reshape(ineff_vals.alpha, N, 1)

    myLomaxQuantile!(buffer; λ=lambda_N1, α=alpha_N1, r=draws_1D,
                      clamp_lo=c.clamp_lo, out_ceil=c.out_ceil)
end

function get_u_quantile!(::Rayleigh_MSLE, buffer, ineff_vals, draws_1D, c, N)
    sigma_N1 = reshape(ineff_vals.sigma, N, 1)

    myRayleighQuantile!(buffer; σ=sigma_N1, r=draws_1D, clamp_lo=c.clamp_lo)
end

# --- Chunked quantile computation (for GPU memory management when chunks > 1) ---
# Note: draws_1D can be either 1 x D (broadcast mode) or N x D (multiRand mode).
#       When N x D, we slice the appropriate rows for this chunk.

# Helper to slice draws for chunked computation (handles both 1×D and N×D cases)
@inline function _slice_draws_chunk(draws, row_start, row_end)
    return size(draws, 1) > 1 ? (@view draws[row_start:row_end, :]) : draws
end

function get_u_quantile_chunk!(::TruncatedNormal_MSLE, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N)
    mu_chunk = reshape((@view ineff_vals.mu[row_start:row_end]), chunk_N, 1)
    sig_chunk = reshape((@view ineff_vals.sigma_u[row_start:row_end]), chunk_N, 1)
    draws_chunk = _slice_draws_chunk(draws_1D, row_start, row_end)

    myTruncatedNormalQuantile!(buf; μ=mu_chunk, σ=sig_chunk, r=draws_chunk,
                               sqrt2=c.sqrt2, inv_sqrt2=c.inv_sqrt2,
                               clamp_lo=c.lo_erf, clamp_hi=c.hi_erf)
end

function get_u_quantile_chunk!(::Exponential_MSLE, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N)
    lambda_chunk = reshape((@view ineff_vals.lambda[row_start:row_end]), chunk_N, 1)
    draws_chunk = _slice_draws_chunk(draws_1D, row_start, row_end)

    myExponentialQuantile!(buf; λ=lambda_chunk, r=draws_chunk, clamp_lo=c.exp_clamp)
end

function get_u_quantile_chunk!(::HalfNormal_MSLE, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N)
    sigma_chunk = reshape((@view ineff_vals.sigma[row_start:row_end]), chunk_N, 1)
    draws_chunk = _slice_draws_chunk(draws_1D, row_start, row_end)

    myHalfNormalQuantile!(buf; σ=sigma_chunk, r=draws_chunk,
                          sqrt2=c.sqrt2, clamp_lo=c.lo_erf, clamp_hi=c.hi_erf)
end

function get_u_quantile_chunk!(::Weibull_MSLE, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N)
    lambda_chunk = reshape((@view ineff_vals.lambda[row_start:row_end]), chunk_N, 1)
    k_chunk = reshape((@view ineff_vals.k[row_start:row_end]), chunk_N, 1)
    draws_chunk = _slice_draws_chunk(draws_1D, row_start, row_end)

    myWeibullQuantile!(buf; λ=lambda_chunk, k=k_chunk, r=draws_chunk, clamp_lo=c.clamp_lo)
end

function get_u_quantile_chunk!(::Lognormal_MSLE, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N)
    mu_chunk = reshape((@view ineff_vals.mu[row_start:row_end]), chunk_N, 1)
    sigma_chunk = reshape((@view ineff_vals.sigma[row_start:row_end]), chunk_N, 1)
    draws_chunk = _slice_draws_chunk(draws_1D, row_start, row_end)

    myLognormalQuantile!(buf; μ=mu_chunk, σ=sigma_chunk, r=draws_chunk,
                         sqrt2=c.sqrt2, clamp_lo=c.lo_erf, clamp_hi=c.hi_erf)
end

function get_u_quantile_chunk!(::Lomax_MSLE, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N)
    lambda_chunk = reshape((@view ineff_vals.lambda[row_start:row_end]), chunk_N, 1)
    alpha_chunk = reshape((@view ineff_vals.alpha[row_start:row_end]), chunk_N, 1)
    draws_chunk = _slice_draws_chunk(draws_1D, row_start, row_end)

    myLomaxQuantile!(buf; λ=lambda_chunk, α=alpha_chunk, r=draws_chunk,
                      clamp_lo=c.clamp_lo, out_ceil=c.out_ceil)
end

function get_u_quantile_chunk!(::Rayleigh_MSLE, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N)
    sigma_chunk = reshape((@view ineff_vals.sigma[row_start:row_end]), chunk_N, 1)
    draws_chunk = _slice_draws_chunk(draws_1D, row_start, row_end)

    myRayleighQuantile!(buf; σ=sigma_chunk, r=draws_chunk, clamp_lo=c.clamp_lo)
end

# --- PDF computation (evaluates noise density) ---

function get_noise_pdf!(::NormalNoise_MSLE, out, z, noise_vals, c)
    myNormalPDF!(out; z=z, pdf_const=noise_vals.pdf_const, inv_σ=noise_vals.inv_sigma_v)
end

function get_noise_pdf!(::StudentTNoise_MSLE, out, z, noise_vals, c)
    myStudentTPDF!(out; z=z, log_const=noise_vals.log_const,
                   inv_σ_v=noise_vals.inv_sigma_v,
                   half_nu_plus_one=noise_vals.half_nu_plus_one)
end

function get_noise_pdf!(::LaplaceNoise_MSLE, out, z, noise_vals, c)
    myLaplacePDF!(out; z=z, inv_b=noise_vals.inv_b, inv_2b=noise_vals.inv_2b)
end

# --- Log-PDF computation (evaluates log noise density for numerical stability) ---

function log_noise_pdf!(::NormalNoise_MSLE, out, z, noise_vals, c)
    # log f(z) = -0.5*log(2π) - log(σ) - 0.5*(z/σ)²
    inv_σ = noise_vals.inv_sigma_v
    @. out = -0.5 * log(2π) - log(max(inv(inv_σ), c.clamp_lo)) - 0.5 * (z * inv_σ)^2
end

function log_noise_pdf!(::StudentTNoise_MSLE, out, z, noise_vals, c)
    # log f(z) = log_const - (ν+1)/2 * log(1 + (z/σ)²/ν)
    inv_σ = noise_vals.inv_sigma_v
    half_nu_plus_one = noise_vals.half_nu_plus_one
    nu = 2 * half_nu_plus_one - 1
    @. out = noise_vals.log_const - half_nu_plus_one * log(1 + (z * inv_σ)^2 / nu)
end

function log_noise_pdf!(::LaplaceNoise_MSLE, out, z, noise_vals, c)
    # log f(z) = log(inv_2b) - |z|*inv_b
    inv_b = noise_vals.inv_b
    inv_2b = noise_vals.inv_2b
    @. out = log(max(inv_2b, c.clamp_lo)) - abs(z) * inv_b
end

# --- Noise CDF functions (in-place, needed for copula: computes F_v(v)) ---

"""Compute CDF of Normal noise in-place: out .= F_v(v)"""
function noise_cdf!(::NormalNoise_MSLE, out, z, noise_vals, c)
    @. out = 0.5 * (1 + erf(z * noise_vals.inv_sigma_v * c.inv_sqrt2))
    return nothing
end

"""Compute CDF of Laplace noise in-place: out .= F_v(v)"""
function noise_cdf!(::LaplaceNoise_MSLE, out, z, noise_vals, c)
    inv_b = noise_vals.inv_b
    @. out = ifelse(z < 0, 0.5 * exp(z * inv_b), 1 - 0.5 * exp(-z * inv_b))
    return nothing
end

"""StudentT noise CDF is not yet supported for copula (unreachable due to _build_model validation)."""
function noise_cdf!(::StudentTNoise_MSLE, out, z, noise_vals, c)
    error("StudentT noise CDF is not yet implemented for copula models. " *
          "Use Normal or Laplace noise with copula.")
end

# --- Copula parameter extraction ---

"""No copula: return empty NamedTuple."""
get_copula_vals(::NoCopula_MSLE, p, idx, c) = NamedTuple()

"""Gaussian copula: extract ρ = ρ_max · tanh(θ_rho) from parameter vector."""
function get_copula_vals(::GaussianCopula_MSLE, p, idx, c)
    theta_rho = p[idx.copula.theta_rho]
    rho = c.rho_max * tanh(theta_rho)
    return (rho=rho, theta_rho=theta_rho)
end

"""Clayton copula: ρ = clamp(exp(θ_rho) + floor, floor, max), ensuring ρ > 0 and bounded."""
function get_copula_vals(::ClaytonCopula_MSLE, p, idx, c)
    theta_rho = p[idx.copula.theta_rho]
    rho = clamp(exp(theta_rho) + c.clayton_rho_floor, c.clayton_rho_floor, c.clayton_rho_max)
    return (rho=rho, theta_rho=theta_rho)
end

"""Gumbel copula: ρ = clamp(exp(θ_rho) + 1, 1, max), ensuring ρ ≥ 1 and bounded."""
function get_copula_vals(::GumbelCopula_MSLE, p, idx, c)
    theta_rho = p[idx.copula.theta_rho]
    rho = clamp(exp(theta_rho) + one(typeof(theta_rho)), one(typeof(theta_rho)), c.gumbel_rho_max)
    return (rho=rho, theta_rho=theta_rho)
end

"""Clayton 90° copula: same parameterization as Clayton (ρ = exp(θ) + floor)."""
function get_copula_vals(::Clayton90Copula_MSLE, p, idx, c)
    theta_rho = p[idx.copula.theta_rho]
    rho = clamp(exp(theta_rho) + c.clayton_rho_floor, c.clayton_rho_floor, c.clayton_rho_max)
    return (rho=rho, theta_rho=theta_rho)
end

# --- Copula log-adjustment dispatch ---

"""No copula: no-op (never called in the hot path)."""
copula_log_adjustment!(::NoCopula_MSLE, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c) = nothing

"""
    copula_log_adjustment!(::GaussianCopula_MSLE, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c)

Compute the copula log-density adjustment in-place into `adj`.
Uses `Fv_buf` as workspace for the noise CDF. Zero intermediate allocations —
all operations are fused into a single broadcast kernel via `@.`.

- `adj`: output buffer (N×D), receives log copula density
- `Fv_buf`: workspace buffer (N×D), same size as adj
- `z_buffer`: composite error v = ε + sign*u (before log_noise_pdf! overwrites it)
- `draws`: the Halton draws (= F_u(u) by the quantile inversion)
"""
function copula_log_adjustment!(::GaussianCopula_MSLE, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c)
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
    copula_log_adjustment!(::ClaytonCopula_MSLE, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c)

Compute Clayton copula log-density adjustment in-place into `adj`.
Clayton density: c(u₁,u₂) = (ρ+1)(u₁u₂)^{-(ρ+1)} [u₁^{-ρ} + u₂^{-ρ} - 1]^{-1/ρ-2}
Log form: log(ρ+1) - (ρ+1)(log u₁ + log u₂) + (-1/ρ-2) log[u₁^{-ρ} + u₂^{-ρ} - 1]
"""
function copula_log_adjustment!(::ClaytonCopula_MSLE, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c)
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
    copula_log_adjustment!(::GumbelCopula_MSLE, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c)

Compute Gumbel copula log-density adjustment in-place into `adj`.
Let w₁=-log(u₁), w₂=-log(u₂), S=w₁^ρ+w₂^ρ.
Density: exp(-S^{1/ρ}) w₁^{ρ-1} [(ρ-1)S^{-2+1/ρ} + S^{-2+2/ρ}] w₂^{ρ-1} / (u₁u₂)
Log form: -S^{1/ρ} + (ρ-1)(log w₁+log w₂) + log[(ρ-1)S^{e₁}+S^{e₂}] - log u₁ - log u₂
where e₁=-2+1/ρ, e₂=-2+2/ρ.
Uses a two-pass approach: first S into adj, then final log-density.
"""
function copula_log_adjustment!(::GumbelCopula_MSLE, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c)
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

    # Pass 1: Compute S = w₁^ρ + w₂^ρ into adj (workspace)
    @. adj = max((-log(Fv_buf))^rho + (-log(clamp(draws, lo, hi)))^rho, sf)

    # Pass 2: Compute log copula density. adj holds S, Fv_buf holds clamped F_v.
    # Element-wise broadcast reads adj (=S) before overwriting — safe with @.
    @. adj = -adj^inv_rho +
             rho_m1 * (log(-log(Fv_buf)) + log(-log(clamp(draws, lo, hi)))) +
             log(rho_m1 * adj^e1 + adj^e2) -
             log(Fv_buf) - log(clamp(draws, lo, hi))
    return nothing
end

"""
    copula_log_adjustment!(::Clayton90Copula_MSLE, adj, Fv_buf, z_buffer, draws, ...)

90° rotated Clayton copula: c^{90°}(z_v, z_u) = c^{Clayton}(1 - F_v(v), F_u(u)).
Uses F_v(-v) instead of 1 - F_v(v) for numerical precision (avoids catastrophic
cancellation when F_v(v) ≈ 1).
"""
function copula_log_adjustment!(::Clayton90Copula_MSLE, adj, Fv_buf, z_buffer, draws,
                                noise, noise_vals, copula_vals, c)
    # Step 1: Compute F_v(-v) = 1 - F_v(v) via negation (better precision)
    @. adj = -z_buffer                              # use adj as temp workspace
    noise_cdf!(noise, Fv_buf, adj, noise_vals, c)   # Fv_buf = F_v(-v)

    # Step 2: Clayton density with rotated first argument (identical formula)
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

# --- Log-sum-exp utility (for numerically stable likelihood aggregation) ---

_maximum_msle(A; dims) = maximum(A; dims=dims)
_sum_msle(A; dims)     = sum(A; dims=dims)
_sum_scalar_msle(v::AbstractArray) = sum(v)

# Conditional GPU overloads (if CUDA is loaded in Main)
if isdefined(Main, :CUDA)
    _maximum_msle(A::Main.CUDA.AnyCuArray; dims) = Main.CUDA.maximum(A; dims=dims)
    _sum_msle(A::Main.CUDA.AnyCuArray; dims)     = Main.CUDA.sum(A; dims=dims)
    _sum_scalar_msle(v::Main.CUDA.AnyCuArray)    = sum(v)
end

"""
    logsumexp_rows(A::AbstractMatrix)

Compute log(sum(exp(A[i,:]))) for each row i, using the log-sum-exp trick
for numerical stability.

For each row: logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
"""
function logsumexp_rows(A::AbstractMatrix, clamp_lo=1e-300)
    max_vals = _maximum_msle(A; dims=2)
    sum_exp  = _sum_msle(exp.(A .- max_vals); dims=2)
    # Use oftype to convert clamp_lo to match element type (works with ForwardDiff.Dual)
    fmin = oftype(zero(eltype(A)), clamp_lo)
    @. sum_exp = max(sum_exp, fmin)
    return vec(max_vals .+ log.(sum_exp))
end

# ============================================================================
# Section 8: CPU Likelihood Function
# ============================================================================

"""
    qmc_nll(Y, X, Z, p, draws; noise, ineff, hetero=Symbol[], constants=nothing)

Compute negative log-likelihood for stochastic frontier model using MSLE integration.
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
  - `:Normal` - Normal noise v ~ N(0, σ_v²)
  - `:StudentT` - Student-t noise with scale σ_v and degrees of freedom ν > 2
  - `:Laplace` - Laplace noise v ~ Laplace(0, b)

- `ineff::Symbol`: Inefficiency distribution
  - `:TruncatedNormal` - Truncated Normal u ~ TN(μ, σ_u; lower=0)
  - `:Exponential` - Exponential u ~ Exp(λ), λ = Var(u)
  - `:HalfNormal` - Half Normal u ~ |N(0, σ²)|
  - `:Weibull` - Weibull u ~ Weibull(λ, k)
  - `:Lognormal` - Lognormal u ~ LogNormal(μ, σ)
  - `:Lomax` - Lomax u ~ Lomax(α, λ)
  - `:Rayleigh` - Rayleigh u ~ Rayleigh(σ)

- `hetero`: Vector of symbols controlling heterogeneity:
  - `:TruncatedNormal` → `[:mu]`, `[:sigma_sq]`, or both
  - `:Exponential` → `[:lambda]`
  - `:HalfNormal` → `[:sigma_sq]`
  - `:Weibull` → `[:lambda]`, `[:k]`, or both
  - `:Lognormal` → `[:mu]`, `[:sigma_sq]`, or both
  - `:Lomax` → `[:lambda]`, `[:alpha]`, or both
  - `:Rayleigh` → `[:sigma_sq]`

- `constants`: Precomputed constants from `make_constants` (default: computed fresh)

# Returns
- Scalar negative log-likelihood value

# Examples
```julia
# Create draws as 1×D matrix (required format)
halton = reshape(make_halton_p(1023), 1, 1023)

# Normal noise + Truncated Normal inefficiency
nll = qmc_nll(Y, X, Z, p, halton; noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])

# Laplace noise + Weibull inefficiency
nll = qmc_nll(Y, X, Z, p, halton; noise=:Laplace, ineff=:Weibull, hetero=[:lambda, :k])

# StudentT noise + Lognormal inefficiency
nll = qmc_nll(Y, X, Z, p, halton; noise=:StudentT, ineff=:Lognormal, hetero=[:mu, :sigma_sq])
```
"""
function qmc_nll(Y::AbstractVector{T}, X::AbstractMatrix{T}, Z::AbstractMatrix{T},
                 p::AbstractVector{P}, draws::AbstractMatrix{T};
                 noise::Symbol, ineff::Symbol,
                 copula::Symbol=:None,
                 hetero::Vector{Symbol}=Symbol[],
                 chunks::Int=4,
                 constants=nothing,
                 type::Symbol=:prod) where {T<:AbstractFloat, P<:Real}

    # Validate draws is 1×D (shared) or N×D (unique)
    n_obs = length(Y)
    if size(draws, 1) != 1 && size(draws, 1) != n_obs
        error("Invalid `draws` shape: expected 1×D (shared) or $(n_obs)×D (unique). Got $(size(draws)).")
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

    # Dimensions
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

    # Copula values via dispatch
    copula_vals = get_copula_vals(model.copula, p, idx, c)
    has_copula = copula_plen(model.copula) > 0

    # draws is already a 1×D matrix
    draws_1D = draws

    if chunks == 1
        # ================================================================
        # Non-chunked path: process all observations at once
        # ================================================================
        buffer = similar(ε, P, n_obs, D)
        ε_N1 = reshape(ε, n_obs, 1)

        # Step A: Compute inefficiency quantiles via dispatch
        get_u_quantile!(model.ineff, buffer, ineff_vals, draws_1D, c, n_obs)

        # Step B: Add residuals in-place to get composite error z = ε + sign*u
        @. buffer = ε_N1 + frontier_sign * buffer

        # Step C': Copula adjustment (must happen before log_noise_pdf! overwrites buffer)
        if has_copula
            copula_adj = similar(buffer)
            Fv_buf = similar(buffer)
            copula_log_adjustment!(model.copula, copula_adj, Fv_buf, buffer, draws_1D,
                                   model.noise, noise_vals, copula_vals, c)
        end

        # Step C: Compute log noise PDF via dispatch (log-space for numerical stability)
        log_noise_pdf!(model.noise, buffer, buffer, noise_vals, c)

        # Step C+: Add copula log-density to log noise PDF
        if has_copula
            @. buffer = buffer + copula_adj
        end

        # Step D: Log-sum-exp aggregation (replaces linear mean + log)
        log_likes = logsumexp_rows(buffer) .- log(P(D))
        return -_sum_scalar_msle(log_likes)
    else
        # ================================================================
        # Chunked path: split observations for GPU memory management
        # ================================================================
        chunk_size = cld(n_obs, chunks)  # ceiling division
        buffer = similar(ε, P, chunk_size, D)
        copula_adj_buf = has_copula ? similar(buffer) : buffer
        Fv_adj_buf = has_copula ? similar(buffer) : buffer
        chunk_nlls = Vector{P}(undef, chunks)

        for chunk_idx in 1:chunks
            row_start = (chunk_idx - 1) * chunk_size + 1
            row_end = min(chunk_idx * chunk_size, n_obs)

            # Skip empty chunks (can happen if n_obs not divisible by chunks)
            row_start > row_end && continue

            chunk_N = row_end - row_start + 1
            buf = @view buffer[1:chunk_N, :]
            res_chunk = reshape((@view ε[row_start:row_end]), chunk_N, 1)

            # Step A: Compute inefficiency quantiles for this chunk
            get_u_quantile_chunk!(model.ineff, buf, ineff_vals, draws_1D, c,
                                      row_start, row_end, chunk_N)

            # Step B: Add residuals
            @. buf = res_chunk + frontier_sign * buf

            # Step C': Copula adjustment (before log_noise_pdf! overwrites buf)
            if has_copula
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                Fv_buf = @view Fv_adj_buf[1:chunk_N, :]
                draws_chunk = size(draws_1D, 1) == 1 ? draws_1D : @view draws_1D[row_start:row_end, :]
                copula_log_adjustment!(model.copula, copula_adj, Fv_buf, buf, draws_chunk,
                                       model.noise, noise_vals, copula_vals, c)
            end

            # Step C: Compute log noise PDF (log-space for numerical stability)
            log_noise_pdf!(model.noise, buf, buf, noise_vals, c)

            # Step C+: Add copula log-density
            if has_copula
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                @. buf = buf + copula_adj
            end

            # Step D: Log-sum-exp aggregation
            log_likes = logsumexp_rows(buf) .- log(P(D))
            chunk_nlls[chunk_idx] = _sum_scalar_msle(log_likes)
        end

        return -sum(chunk_nlls)
    end
end

"""
    qmc_nll(spec::sfmodel_MSLE_spec, p::AbstractVector; chunks::Int=4)

Compute negative log-likelihood using a model specification.
This is the simplified interface that extracts all configuration from the spec.

# Arguments
- `spec::sfmodel_MSLE_spec`: Model specification containing data and model config
- `p::AbstractVector`: Parameter vector

# Keyword Arguments
- `chunks::Int=4`: Number of chunks for memory management (default: 1)

# Example
```julia
spec = sfmodel_MSLE_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])

# Simple call - no need to repeat model configuration
nll_value = qmc_nll(spec, p)

# For optimization
nll = p -> qmc_nll(spec, p)
result = optimize(nll, p0, Newton(); autodiff=:forward)
```
"""
function qmc_nll(spec::sfmodel_MSLE_spec{T}, p::AbstractVector{P};
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

    # Copula values via dispatch
    copula_vals = get_copula_vals(model.copula, p, idx, c)
    has_copula = copula_plen(model.copula) > 0

    # Use pre-reshaped draws from spec
    draws_1D = spec.draws_2D

    if chunks == 1
        # Non-chunked path
        buffer = similar(ε, P, n_obs, D)
        ε_N1 = reshape(ε, n_obs, 1)

        get_u_quantile!(model.ineff, buffer, ineff_vals, draws_1D, c, n_obs)

        # Apply scaling function: buffer = h_i * u*
        if spec.scaling
            Z_s = spec.scaling_zvar
            L_s = spec.L_scaling
            h = exp.(sum(P(p[idx.delta[j]]) .* (@view Z_s[:, j]) for j in 1:L_s))
            buffer .= buffer .* h
        end

        @. buffer = ε_N1 + spec.sign * buffer

        # Copula adjustment (before log_noise_pdf! overwrites buffer)
        if has_copula
            copula_adj = similar(buffer)
            Fv_buf = similar(buffer)
            copula_log_adjustment!(model.copula, copula_adj, Fv_buf, buffer, draws_1D,
                                   model.noise, noise_vals, copula_vals, c)
        end

        log_noise_pdf!(model.noise, buffer, buffer, noise_vals, c)

        if has_copula
            @. buffer = buffer + copula_adj
        end

        log_likes = logsumexp_rows(buffer) .- log(P(D))
        return -_sum_scalar_msle(log_likes)
    else
        # Chunked path
        chunk_size = cld(n_obs, chunks)
        buffer = similar(ε, P, chunk_size, D)
        copula_adj_buf = has_copula ? similar(buffer) : buffer
        Fv_adj_buf = has_copula ? similar(buffer) : buffer
        chunk_nlls = Vector{P}(undef, chunks)

        for chunk_idx in 1:chunks
            row_start = (chunk_idx - 1) * chunk_size + 1
            row_end = min(chunk_idx * chunk_size, n_obs)
            row_start > row_end && continue

            chunk_N = row_end - row_start + 1
            buf = @view buffer[1:chunk_N, :]
            res_chunk = reshape((@view ε[row_start:row_end]), chunk_N, 1)

            get_u_quantile_chunk!(model.ineff, buf, ineff_vals, draws_1D, c,
                                      row_start, row_end, chunk_N)

            # Apply scaling function: buf = h_i * u* (chunked)
            if spec.scaling
                Z_s_chunk = @view spec.scaling_zvar[row_start:row_end, :]
                L_s = spec.L_scaling
                h_chunk = exp.(sum(P(p[idx.delta[j]]) .* (@view Z_s_chunk[:, j]) for j in 1:L_s))
                buf .= buf .* h_chunk
            end

            @. buf = res_chunk + spec.sign * buf

            # Copula adjustment (before log_noise_pdf! overwrites buf)
            if has_copula
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                Fv_buf = @view Fv_adj_buf[1:chunk_N, :]
                draws_chunk = size(draws_1D, 1) == 1 ? draws_1D : @view draws_1D[row_start:row_end, :]
                copula_log_adjustment!(model.copula, copula_adj, Fv_buf, buf, draws_chunk,
                                       model.noise, noise_vals, copula_vals, c)
            end

            log_noise_pdf!(model.noise, buf, buf, noise_vals, c)

            if has_copula
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                @. buf = buf + copula_adj
            end

            log_likes = logsumexp_rows(buf) .- log(P(D))
            chunk_nlls[chunk_idx] = _sum_scalar_msle(log_likes)
        end

        return -sum(chunk_nlls)
    end
end

# ============================================================================
# Section 9: GPU Support (CUDA.jl integration)
# ============================================================================
#=
GPU Support Notes:

The unified qmc_nll function now works with both CPU Arrays and GPU CuArrays
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
nll = qmc_nll(Y_gpu, X_gpu, Z_gpu, p, draws_gpu;
              noise=:StudentT, ineff=:TruncatedNormal, hetero=[:mu])

# With ForwardDiff optimization:
using Optim, ForwardDiff

result = optimize(
    theta -> qmc_nll(Y_gpu, X_gpu, Z_gpu, theta, draws_gpu;
                     noise=:StudentT, ineff=:TruncatedNormal, hetero=[:mu]),
    p0, Newton(); autodiff=:forward
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
using MSLE integration.

# Mathematical Definitions
Given the composed error ε = v - u, where v is noise and u ≥ 0 is inefficiency:

- **JLMS (Jondrow et al. 1982):** E(u|ε) = [∫ u·f_v(ε+u)·f_u(u) du] / [∫ f_v(ε+u)·f_u(u) du]
- **BC (Battese & Coelli 1988):** E(e^{-u}|ε) = [∫ e^{-u}·f_v(ε+u)·f_u(u) du] / [∫ f_v(ε+u)·f_u(u) du]

# Arguments
Same as `qmc_nll`:
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

    # Validate draws is 1×D (shared) or N×D (unique)
    n_obs = length(Y)
    if size(draws, 1) != 1 && size(draws, 1) != n_obs
        error("Invalid `draws` shape: expected 1×D (shared) or $(n_obs)×D (unique). Got $(size(draws)).")
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

    # Dimensions
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

    # Copula values via dispatch
    copula_vals = get_copula_vals(model.copula, p, idx, c)
    has_copula = copula_plen(model.copula) > 0

    # draws is already a 1×D matrix
    draws_1D = draws

    if chunks == 1
        # ================================================================
        # Non-chunked path: process all observations at once
        # ================================================================

        # Step 1: Generate u samples → N × D matrix
        u_buffer = similar(ε, P, n_obs, D)
        get_u_quantile!(model.ineff, u_buffer, ineff_vals, draws_1D, c, n_obs)

        # Step 2: Compute z = ε + sign*u
        ε_N1 = reshape(ε, n_obs, 1)
        z_buffer = similar(u_buffer)
        @. z_buffer = ε_N1 + frontier_sign * u_buffer

        # Step 2b: Copula adjustment (before log_noise_pdf! overwrites z_buffer)
        if has_copula
            copula_adj = similar(z_buffer)
            Fv_buf = similar(z_buffer)
            copula_log_adjustment!(model.copula, copula_adj, Fv_buf, z_buffer, draws_1D,
                                   model.noise, noise_vals, copula_vals, c)
        end

        # Step 3: Compute log f_v(z) in-place (log-space for numerical stability)
        log_noise_pdf!(model.noise, z_buffer, z_buffer, noise_vals, c)

        # Step 3b: Add copula log-density to log noise PDF
        if has_copula
            @. z_buffer = z_buffer + copula_adj
        end

        # Step 4: Log-space indices
        # Denominator: log f_ε(ε) = logsumexp(log f_v) - log(D)
        log_denom = logsumexp_rows(z_buffer) .- log(P(D))

        # JLMS numerator: logsumexp(log(u) + log f_v) - log(D)
        log_w_buffer = similar(u_buffer)
        @. log_w_buffer = log(max(u_buffer, c.clamp_lo)) + z_buffer
        log_jlms_num = logsumexp_rows(log_w_buffer) .- log(P(D))

        # BC numerator: logsumexp(-u + log f_v) - log(D)
        @. log_w_buffer = -u_buffer + z_buffer
        log_bc_num = logsumexp_rows(log_w_buffer) .- log(P(D))

        return (jlms=exp.(log_jlms_num .- log_denom),
                bc=exp.(log_bc_num .- log_denom),
                likelihood=exp.(log_denom))
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
        z_buffer = similar(ε, P, chunk_size, D)
        log_w_buffer = similar(ε, P, chunk_size, D)
        copula_adj_buf = has_copula ? similar(ε, P, chunk_size, D) : z_buffer
        Fv_adj_buf = has_copula ? similar(ε, P, chunk_size, D) : z_buffer

        for chunk_idx in 1:chunks
            row_start = (chunk_idx - 1) * chunk_size + 1
            row_end = min(chunk_idx * chunk_size, n_obs)

            # Skip empty chunks
            row_start > row_end && continue

            chunk_N = row_end - row_start + 1
            u_buf = @view u_buffer[1:chunk_N, :]
            z_buf = @view z_buffer[1:chunk_N, :]
            lw_buf = @view log_w_buffer[1:chunk_N, :]

            # Step 1: Generate u samples for this chunk
            get_u_quantile_chunk!(model.ineff, u_buf, ineff_vals, draws_1D, c,
                                      row_start, row_end, chunk_N)

            # Step 2: Compute z = ε + sign*u
            ε_chunk = reshape((@view ε[row_start:row_end]), chunk_N, 1)
            @. z_buf = ε_chunk + frontier_sign * u_buf

            # Step 2b: Copula adjustment (before log_noise_pdf! overwrites z_buf)
            if has_copula
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                Fv_buf = @view Fv_adj_buf[1:chunk_N, :]
                draws_chunk = size(draws_1D, 1) == 1 ? draws_1D : @view draws_1D[row_start:row_end, :]
                copula_log_adjustment!(model.copula, copula_adj, Fv_buf, z_buf, draws_chunk,
                                       model.noise, noise_vals, copula_vals, c)
            end

            # Step 3: Compute log f_v(z) in-place (log-space for numerical stability)
            log_noise_pdf!(model.noise, z_buf, z_buf, noise_vals, c)

            # Step 3b: Add copula log-density
            if has_copula
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                @. z_buf = z_buf + copula_adj
            end

            # Step 4: Log-space indices for this chunk
            log_denom_chunk = logsumexp_rows(z_buf) .- log(P(D))

            @. lw_buf = log(max(u_buf, c.clamp_lo)) + z_buf
            log_jlms_chunk = logsumexp_rows(lw_buf) .- log(P(D))

            @. lw_buf = -u_buf + z_buf
            log_bc_chunk = logsumexp_rows(lw_buf) .- log(P(D))

            # Store results
            jlms_out[row_start:row_end] .= exp.(log_jlms_chunk .- log_denom_chunk)
            bc_out[row_start:row_end] .= exp.(log_bc_chunk .- log_denom_chunk)
            likelihood_out[row_start:row_end] .= exp.(log_denom_chunk)
        end

        return (jlms=jlms_out, bc=bc_out, likelihood=likelihood_out)
    end
end

"""
    jlms_bc_indices(spec::sfmodel_MSLE_spec, p::AbstractVector; chunks::Int=1)

Compute JLMS and BC efficiency indices using a model specification.
This is the simplified interface that extracts all configuration from the spec.

# Arguments
- `spec::sfmodel_MSLE_spec`: Model specification containing data and model config
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
spec = sfmodel_MSLE_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])
result = jlms_bc_indices(spec, p_hat)

# Access results
println("Mean efficiency: ", mean(result.bc))
println("Mean inefficiency: ", mean(result.jlms))
```
"""
function jlms_bc_indices(spec::sfmodel_MSLE_spec{T}, p::AbstractVector{P};
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

    # Copula values via dispatch
    copula_vals = get_copula_vals(model.copula, p, idx, c)
    has_copula = copula_plen(model.copula) > 0

    # Use pre-reshaped draws from spec
    draws_1D = spec.draws_2D

    # ========== Standard path (all ICDF distributions) ==========
    if chunks == 1
        # Non-chunked path
        u_buffer = similar(ε, P, n_obs, D)
        get_u_quantile!(model.ineff, u_buffer, ineff_vals, draws_1D, c, n_obs)

        # Apply scaling function: u = h(z) * u*
        if spec.scaling
            Z_s = spec.scaling_zvar
            L_s = spec.L_scaling
            h = exp.(sum(P(p[idx.delta[j]]) .* (@view Z_s[:, j]) for j in 1:L_s))
            u_buffer .= u_buffer .* h
        end

        ε_N1 = reshape(ε, n_obs, 1)
        z_buffer = similar(u_buffer)
        @. z_buffer = ε_N1 + spec.sign * u_buffer

        # Copula adjustment (before log_noise_pdf! overwrites z_buffer)
        if has_copula
            copula_adj = similar(z_buffer)
            Fv_buf = similar(z_buffer)
            copula_log_adjustment!(model.copula, copula_adj, Fv_buf, z_buffer, draws_1D,
                                   model.noise, noise_vals, copula_vals, c)
        end

        # Log noise PDF (log-space for numerical stability)
        log_noise_pdf!(model.noise, z_buffer, z_buffer, noise_vals, c)

        # Add copula log-density
        if has_copula
            @. z_buffer = z_buffer + copula_adj
        end

        # Log-space indices
        log_denom = logsumexp_rows(z_buffer) .- log(P(D))

        log_w_buffer = similar(u_buffer)
        @. log_w_buffer = log(max(u_buffer, c.clamp_lo)) + z_buffer
        log_jlms_num = logsumexp_rows(log_w_buffer) .- log(P(D))

        @. log_w_buffer = -u_buffer + z_buffer
        log_bc_num = logsumexp_rows(log_w_buffer) .- log(P(D))

        return (jlms=exp.(log_jlms_num .- log_denom),
                bc=exp.(log_bc_num .- log_denom),
                likelihood=exp.(log_denom))
    else
        # Chunked path
        chunk_size = cld(n_obs, chunks)
        jlms_out = similar(ε, P, n_obs)
        bc_out = similar(ε, P, n_obs)
        likelihood_out = similar(ε, P, n_obs)

        u_buffer = similar(ε, P, chunk_size, D)
        z_buffer = similar(ε, P, chunk_size, D)
        log_w_buffer = similar(ε, P, chunk_size, D)
        copula_adj_buf = has_copula ? similar(ε, P, chunk_size, D) : z_buffer
        Fv_adj_buf = has_copula ? similar(ε, P, chunk_size, D) : z_buffer

        for chunk_idx in 1:chunks
            row_start = (chunk_idx - 1) * chunk_size + 1
            row_end = min(chunk_idx * chunk_size, n_obs)
            row_start > row_end && continue

            chunk_N = row_end - row_start + 1
            u_buf = @view u_buffer[1:chunk_N, :]
            z_buf = @view z_buffer[1:chunk_N, :]
            lw_buf = @view log_w_buffer[1:chunk_N, :]

            get_u_quantile_chunk!(model.ineff, u_buf, ineff_vals, draws_1D, c,
                                      row_start, row_end, chunk_N)

            # Apply scaling function: u = h(z) * u*
            if spec.scaling
                Z_s_chunk = @view spec.scaling_zvar[row_start:row_end, :]
                L_s = spec.L_scaling
                h_chunk = exp.(sum(P(p[idx.delta[j]]) .* (@view Z_s_chunk[:, j]) for j in 1:L_s))
                u_buf .= u_buf .* h_chunk
            end

            ε_chunk = reshape((@view ε[row_start:row_end]), chunk_N, 1)
            @. z_buf = ε_chunk + spec.sign * u_buf

            # Copula adjustment (before log_noise_pdf! overwrites z_buf)
            if has_copula
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                Fv_buf = @view Fv_adj_buf[1:chunk_N, :]
                draws_chunk = size(draws_1D, 1) == 1 ? draws_1D : @view draws_1D[row_start:row_end, :]
                copula_log_adjustment!(model.copula, copula_adj, Fv_buf, z_buf, draws_chunk,
                                       model.noise, noise_vals, copula_vals, c)
            end

            log_noise_pdf!(model.noise, z_buf, z_buf, noise_vals, c)

            # Add copula log-density
            if has_copula
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                @. z_buf = z_buf + copula_adj
            end

            log_denom_chunk = logsumexp_rows(z_buf) .- log(P(D))

            @. lw_buf = log(max(u_buf, c.clamp_lo)) + z_buf
            log_jlms_chunk = logsumexp_rows(lw_buf) .- log(P(D))

            @. lw_buf = -u_buf + z_buf
            log_bc_chunk = logsumexp_rows(lw_buf) .- log(P(D))

            jlms_out[row_start:row_end] .= exp.(log_jlms_chunk .- log_denom_chunk)
            bc_out[row_start:row_end] .= exp.(log_bc_chunk .- log_denom_chunk)
            likelihood_out[row_start:row_end] .= exp.(log_denom_chunk)
        end

        return (jlms=jlms_out, bc=bc_out, likelihood=likelihood_out)
    end
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
nll = p -> qmc_nll(Y, X, Z, p, halton; noise=:Normal, ineff=:TruncatedNormal)
result = optimize(nll, p0, Newton(); autodiff=:forward)
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
    _detect_log_params(model::MSLEModel, idx, L::Int, hetero::Vector{Symbol})

Auto-detect log-transformed parameters based on model type for auxiliary table.
Only returns parameters that are scalar (not heteroscedastic).

# Returns
Vector of tuples: `[(display_name, param_index), ...]`
"""
function _detect_log_params(model::MSLEModel, idx, L::Int, hetero::Vector{Symbol})
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
    end

    return entries
end

"""
    _detect_copula_params(model::MSLEModel, idx, coef, stddev)

Auto-detect copula parameters for auxiliary table.
Returns a NamedTuple with copula-specific statistics.
"""
function _detect_copula_params(model::MSLEModel, idx, coef, stddev)
    if model.copula isa NoCopula_MSLE
        return NamedTuple()
    elseif model.copula isa GaussianCopula_MSLE
        rho_max = 0.999
        theta_rho = coef[idx.copula.theta_rho]
        se_theta = stddev[idx.copula.theta_rho]
        rho = rho_max * tanh(theta_rho)
        se_rho = rho_max * (1 - tanh(theta_rho)^2) * se_theta  # Delta method
        kendalls_tau = (2/π) * asin(rho)
        return (rho=rho, se_rho=se_rho, kendalls_tau=kendalls_tau,
                tail_dep=0.0, theta_rho=theta_rho, se_theta=se_theta)
    elseif model.copula isa ClaytonCopula_MSLE
        theta_rho = coef[idx.copula.theta_rho]
        se_theta = stddev[idx.copula.theta_rho]
        rho = clamp(exp(theta_rho) + 1e-6, 1e-6, 50.0)  # Match get_copula_vals clamp
        se_rho = exp(theta_rho) * se_theta  # Delta method: d/dθ (exp(θ)+c) = exp(θ)
        kendalls_tau = rho / (2 + rho)
        tail_dep = 2^(-1/rho)              # Lower tail dependence
        return (rho=rho, se_rho=se_rho, kendalls_tau=kendalls_tau,
                tail_dep=tail_dep, theta_rho=theta_rho, se_theta=se_theta)
    elseif model.copula isa GumbelCopula_MSLE
        theta_rho = coef[idx.copula.theta_rho]
        se_theta = stddev[idx.copula.theta_rho]
        rho = clamp(exp(theta_rho) + 1, 1.0, 50.0)  # Match get_copula_vals clamp
        se_rho = exp(theta_rho) * se_theta  # Delta method: d/dθ (exp(θ)+1) = exp(θ)
        kendalls_tau = 1 - 1/rho
        tail_dep = 2 - 2^(1/rho)           # Upper tail dependence
        return (rho=rho, se_rho=se_rho, kendalls_tau=kendalls_tau,
                tail_dep=tail_dep, theta_rho=theta_rho, se_theta=se_theta)
    elseif model.copula isa Clayton90Copula_MSLE
        theta_rho = coef[idx.copula.theta_rho]
        se_theta = stddev[idx.copula.theta_rho]
        rho = clamp(exp(theta_rho) + 1e-6, 1e-6, 50.0)  # Match get_copula_vals clamp
        se_rho = exp(theta_rho) * se_theta  # Delta method: d/dθ (exp(θ)+c) = exp(θ)
        kendalls_tau = -rho / (2 + rho)      # Negated for 90° rotation
        tail_dep = 2^(-1/rho)               # Upper-lower tail dependence λ_UL
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

    print("Method: "); printstyled("MSLE"; color=:yellow); println()
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
        println("Log-parameters converted to original scale (σ² = exp(log_σ²)):")

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
    print_table(spec::sfmodel_MSLE_spec, coef::AbstractVector, var_cov_matrix::AbstractMatrix;
                optim_result=nothing, table_format::Symbol=:text)

Print formatted estimation results table using a model specification.
This is the simplified interface that extracts configuration from the spec.

# Arguments
- `spec::sfmodel_MSLE_spec`: Model specification containing metadata for table formatting
- `coef::AbstractVector`: Coefficient vector from optimization
- `var_cov_matrix::AbstractMatrix`: Variance-covariance matrix from `var_cov_mat`

# Keyword Arguments
- `optim_result=nothing`: Optional Optim.jl result for convergence info
- `table_format::Symbol=:text`: Output format (`:text`, `:html`, or `:latex`)

# Returns
NamedTuple: `(table=Matrix, aux_table=Matrix)`

# Example
```julia
spec = sfmodel_MSLE_spec(depvar=y, frontier=X, zvar=Z, noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu],
              varnames=["_cons", "x1", "x2", "_cons", "_cons"],
              eqnames=["frontier", "μ", "ln_σᵤ²", "ln_σᵥ²"],
              eq_indices=[1, 4, 5, 6])

nll = p -> qmc_nll(spec, p)
result = optimize(nll, p0, Newton(); autodiff=:forward)
vcov = var_cov_mat(nll, Optim.minimizer(result))

# Simple call - uses varnames/eqnames/eq_indices from spec
print_table(spec, Optim.minimizer(result), vcov.var_cov_matrix; optim_result=result)
```
"""
function print_table(spec::sfmodel_MSLE_spec{T}, coef::AbstractVector{T},
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
# Section 11b: Model Fitting (sfmodel_MSLE_fit)
# ============================================================================

"""
    sfmodel_MSLE_fit(; spec, method=sfmodel_method(), init=nothing, optim_options=nothing,
                     jlms_bc_index=true, marginal=true, show_table=true, verbose=true)

Estimate a stochastic frontier model using MSLE integration with optional two-stage optimization.

This function is a comprehensive wrapper that:
1. Prepares initial values (OLS-based if not provided)
2. Runs two-stage optimization (warmstart + main) if configured
3. Computes variance-covariance matrix
4. Calculates JLMS and BC efficiency indices
5. Computes marginal effects of inefficiency determinants
6. Prints formatted results tables
7. Returns a comprehensive NamedTuple with all results

# Arguments
- `model::sfmodel_MSLE_spec`: Model specification from `sfmodel_MSLE_spec()`

# Keyword Arguments
- `init=nothing`: Initial parameter vector. If nothing, OLS estimates are used for frontier
  coefficients and 0.1 for other parameters. Can be from `sfmodel_MSLE_init()` or a plain vector.
- `optim_options=nothing`: Optimization options from `sfmodel_MSLE_opt()`. If nothing, uses defaults:
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
include("sf_MSLE_v13.jl")

df = CSV.read("sampledata.csv", DataFrame)
y = df.y
X = hcat(ones(length(y)), df.x1, df.x2)
Z = hcat(ones(length(y)), df.z1)

# Construct model specification
m = sfmodel_MSLE_spec(
    depvar = y,
    frontier = X,
    zvar = Z,
    noise = :Normal,
    ineff = :TruncatedNormal,
    hetero = Symbol[],
    n_draws = 2^12 - 1
)



# Construct initial values
p0 = sfmodel_MSLE_init(
    spec = m,
    frontier = X \\ y,  # OLS
    mu = [0.0],
    ln_sigma_sq = (0.0),
    ln_sigma_v_sq = (0.0)
)

# Prepare optimization options
myopt = sfmodel_MSLE_opt(
    warmstart_solver = NelderMead(),
    warmstart_opt = (iterations = 400, g_abstol = 1e-5),
    main_solver = Newton(),
    main_opt = (iterations = 2000, g_abstol = 1e-8)
)

# Estimate the model
result = sfmodel_MSLE_fit(
    spec = m,
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
function sfmodel_MSLE_fit(;
    spec::SFModelSpec_MSLE,
    method::SFMethodSpec_MSLE = sfmodel_method(),
    init=nothing,
    optim_options=nothing,
    jlms_bc_index::Bool=true,
    marginal::Bool=true,
    show_table::Bool=true,
    verbose::Bool=true
)
    # Assemble internal struct from spec + method
    model = _assemble_MSLE_spec(spec, method)

    # For simulation tracking
    redflag::Int = 0

    # ========== Banner ==========
    if show_table
        printstyled("\n###------------------------------------  ----------------###\n"; color=:yellow)
        printstyled("###  Estimating SF models using Quasi Monte Carlo (MSLE)  ###\n"; color=:yellow)
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
        myopt = sfmodel_MSLE_opt(
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
    _lik = p -> qmc_nll(model, p; chunks=model.chunks)

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

        # All distributions support ForwardDiff
        autodiff_mode = AutoForwardDiff()

        _optres = Optim.optimize(_lik, sf_init, myopt.main_solver,
                                  myopt.main_opt; autodiff=autodiff_mode)

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

        println("* Use `name.list` to see saved results (keys and values) where `name` is the return specified in `name = sfmodel_MSLE_fit(..)`. Values may be retrieved using the keys. For instance:")
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

    # Add scaling delta coefficients
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
        copula_info = _detect_copula_params(model.model, idx, _coevec, stddev)
        _dicRES[:rho] = copula_info.rho
        _dicRES[:kendalls_tau] = copula_info.kendalls_tau
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
export MSLENoiseModel, MSLEIneffModel
export NormalNoise_MSLE, StudentTNoise_MSLE, LaplaceNoise_MSLE
export TruncatedNormal_MSLE, Exponential_MSLE
export HalfNormal_MSLE, Weibull_MSLE, Lognormal_MSLE, Lomax_MSLE, Rayleigh_MSLE
export MSLECopulaModel, NoCopula_MSLE, GaussianCopula_MSLE, ClaytonCopula_MSLE, Clayton90Copula_MSLE, GumbelCopula_MSLE
export MSLEModel, NOISE_MODELS, INEFF_MODELS, COPULA_MODELS
export qmc_nll, make_halton_p, make_halton_wrap, make_constants, plen
export valid_hetero
export jlms_bc_indices
export var_cov_mat, print_table
export SFModelSpec_MSLE, SFMethodSpec_MSLE
export sfmodel_spec, sfmodel_method, _assemble_MSLE_spec
export sfmodel_MSLE_init
export sfmodel_MSLE_opt, sfmodel_MSLE_optim, sfmodel_MSLE_fit


# Include marginal effects module

include("sf_MSLE_marginal_v21.jl")
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
nll_ntn = qmc_nll(Y, X, Z, p_ntn, halton; noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])
println("Normal + TN NLL: $nll_ntn")

# ============================================================================
# Example 2: Normal + Exponential (hetero lambda)
# ============================================================================
# p = [beta(3), ln_sigma_v_sq(1), gamma(3)] -> length = 7
p_ne = [1.0, 0.5, -0.3, -1.0, 0.5, 0.1, -0.2]
nll_ne = qmc_nll(Y, X, Z, p_ne, halton; noise=:Normal, ineff=:Exponential, hetero=[:lambda])
println("Normal + Exponential NLL: $nll_ne")

# ============================================================================
# Example 3: Student T + Truncated Normal (hetero mu)
# ============================================================================
# p = [beta(3), ln_sigma_v_sq(1), ln_nu_minus_2(1), delta(3), ln_sigma_sq(1)] -> length = 9
p_ttn = [1.0, 0.5, -0.3, -1.0, 1.0, 0.5, 0.1, -0.2, -0.5]  # ln_nu_minus_2=1.0 -> nu≈4.7
nll_ttn = qmc_nll(Y, X, Z, p_ttn, halton; noise=:StudentT, ineff=:TruncatedNormal, hetero=[:mu])
println("StudentT + TN NLL: $nll_ttn")

# ============================================================================
# Example 4: Student T + Exponential (hetero lambda)
# ============================================================================
# p = [beta(3), ln_sigma_v_sq(1), ln_nu_minus_2(1), gamma(3)] -> length = 8
p_te = [1.0, 0.5, -0.3, -1.0, 1.0, 0.5, 0.1, -0.2]
nll_te = qmc_nll(Y, X, Z, p_te, halton; noise=:StudentT, ineff=:Exponential, hetero=[:lambda])
println("StudentT + Exponential NLL: $nll_te")

# ============================================================================
# Example 5: Hetero validation error
# ============================================================================
# This will error with a helpful message:
# qmc_nll(Y, X, Z, p_ntn, halton; noise=:Normal, ineff=:TruncatedNormal, hetero=[:lambda])
# ERROR: Invalid hetero option :lambda for TruncatedNormal_MSLE. Valid options: [:mu, :sigma_sq]

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
# - JLMS should be non-negative (u ≥ 0)
# - BC should be in (0, 1] (e^{-u} ∈ (0, 1] for u ≥ 0)
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
# Example 9: Using sfmodel_MSLE_spec for simplified workflow
# ============================================================================
# sfmodel_MSLE_spec centralizes all model configuration, so you only specify it once

# Create specification with auto-generated defaults
spec = sfmodel_MSLE_spec(
    depvar = y,
    frontier = X,
    zvar = Z,
    noise = :Normal,
    ineff = :TruncatedNormal,
    hetero = [:mu],
    n_draws = 1023   # Auto-generates Halton draws
)

# Now all functions use the spec - no need to repeat arguments!
nll_from_spec = qmc_nll(spec, p_ntn)
println("NLL from sfmodel_MSLE_spec: $nll_from_spec")

# Compare with original approach - should be identical
nll_original = qmc_nll(Y, X, Z, p_ntn, halton;
    noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])
@assert isapprox(nll_from_spec, nll_original, rtol=1e-10)
println("sfmodel_MSLE_spec and original NLL match!")

# Efficiency indices also work with spec
result_spec = jlms_bc_indices(spec, p_ntn)
println("sfmodel_MSLE_spec Mean BC: $(mean(result_spec.bc))")

# ============================================================================
# Example 10: sfmodel_MSLE_spec with custom variable names
# ============================================================================
# Provide custom names for print_table
spec_named = sfmodel_MSLE_spec(
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
# Example 11: sfmodel_MSLE_spec simplifies optimization workflow
# ============================================================================
# Full workflow using sfmodel_MSLE_spec

using Optim

# 1. Create spec once
spec_optim = sfmodel_MSLE_spec(
    depvar = y, frontier = X, zvar = Z,
    noise = :Normal, ineff = :TruncatedNormal, hetero = [:mu]
)

# 2. Define NLL closure using spec
nll_closure = p -> qmc_nll(spec_optim, p)

# 3. Optimize
p0 = zeros(plen(spec_optim.model, spec_optim.K, spec_optim.L, spec_optim.hetero))
# result_optim = optimize(nll_closure, p0, Newton(); autodiff=:forward)

# 4. Get variance-covariance matrix
# vcov = var_cov_mat(nll_closure, Optim.minimizer(result_optim))

# 5. Compute efficiency indices
# eff = jlms_bc_indices(spec_optim, Optim.minimizer(result_optim))

# 6. Print table - all formatting info comes from spec
# print_table(spec_optim, Optim.minimizer(result_optim), vcov.var_cov_matrix;
#             optim_result=result_optim)

println("sfmodel_MSLE_spec workflow examples completed!")
=#
