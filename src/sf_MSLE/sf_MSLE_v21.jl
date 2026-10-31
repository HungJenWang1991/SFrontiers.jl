# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

#=
    sf_MSLE_v21.jl

    Negative log-likelihood for stochastic frontier models using MSLE (Halton) draws.
    Supports multiple noise and inefficiency distribution combinations via multiple dispatch.

    Supported Models:
    - Noise: Normal, StudentT, Laplace
    - Inefficiency: TruncatedNormal, Exponential, HalfNormal, Weibull, Lognormal, Lomax, Rayleigh, Erlang{2,3,4}

    Heteroscedasticity Options (via hetero keyword):
    - TruncatedNormal: [:mu, :sigma_u_sq]
    - Exponential: [:sigma_u_sq]
    - HalfNormal: [:sigma_u_sq]
    - Weibull: [:lambda, :k]
    - Lognormal: [:mu, :sigma_u_sq]
    - Lomax: [:lambda, :alpha]
    - Rayleigh: [:sigma_u_sq]
    - Erlang2/Erlang3/Erlang4: [:theta] (shape pinned at the integer order; quantile via the parameter-free standardized Newton kernel)

    Features:
    - Multiple dispatch for model combinations (no if-else logic)
    - CPU and GPU support via array type dispatch
    - ForwardDiff compatible via type parameter P
    - Heteroscedastic parameterization via Z matrix
=#

using HaltonSequences
using SpecialFunctions: erf, erfc, erfinv, erfcinv, loggamma
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

"""Exponential inefficiency: u ~ Exp(σᵤ) with scale σᵤ; the estimated parameter is σᵤ² = Var(u)"""
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

"""Erlang inefficiency: u ~ Erlang(M, θ) = Gamma(shape=M, scale=θ) with the integer shape M pinned by the type parameter (registered orders M = 2, 3, 4); only θ is estimated. Because the shape is pinned and θ is a pure scale, the quantile factorizes as Q(p) = θ·q_M(p) with q_M parameter-free, which is what makes MSLE feasible here but not for Gamma (whose estimated shape k sits inside the quantile)."""
struct Erlang_MSLE{M} <: MSLEIneffModel end

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
    :Rayleigh        => Rayleigh_MSLE,
    :Erlang2         => Erlang_MSLE{2},
    :Erlang3         => Erlang_MSLE{3},
    :Erlang4         => Erlang_MSLE{4}
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
ineff_extras(::TruncatedNormal_MSLE) = 1  # ln_sigma_u_sq (may be heteroscedastic)
ineff_extras(::Exponential_MSLE) = 0      # σᵤ² params handled via hetero
ineff_extras(::HalfNormal_MSLE) = 0       # σ² params handled via hetero
ineff_extras(::Weibull_MSLE) = 0          # λ, k params handled via hetero
ineff_extras(::Lognormal_MSLE) = 0        # μ, σ² params handled via hetero
ineff_extras(::Lomax_MSLE) = 0            # λ, α params handled via hetero
ineff_extras(::Rayleigh_MSLE) = 0         # σ² params handled via hetero
ineff_extras(::Erlang_MSLE) = 0           # θ handled via hetero (shape pinned at the Erlang order)

"""Does this inefficiency model have a μ parameter?"""
has_mu(::TruncatedNormal_MSLE) = true
has_mu(::Exponential_MSLE) = false
has_mu(::HalfNormal_MSLE) = false
has_mu(::Weibull_MSLE) = false
has_mu(::Lognormal_MSLE) = true
has_mu(::Lomax_MSLE) = false
has_mu(::Rayleigh_MSLE) = false
has_mu(::Erlang_MSLE) = false

"""Valid heteroscedasticity options for each inefficiency model."""
valid_hetero(::TruncatedNormal_MSLE) = [:mu, :sigma_u_sq]
valid_hetero(::Exponential_MSLE) = [:sigma_u_sq]
valid_hetero(::HalfNormal_MSLE) = [:sigma_u_sq]
valid_hetero(::Weibull_MSLE) = [:lambda, :k]
valid_hetero(::Lognormal_MSLE) = [:mu, :sigma_u_sq]
valid_hetero(::Lomax_MSLE) = [:lambda, :alpha]
valid_hetero(::Rayleigh_MSLE) = [:sigma_u_sq]
valid_hetero(::Erlang_MSLE) = [:theta]

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
    _validate_homoscedastic_zvar(ineff::MSLEIneffModel, hetero::Vector{Symbol}, zvar_norm)

Twin of the MCI backend's guard. A homoscedastic inefficiency parameter (not listed in
`hetero`) is evaluated as `coef·Z[:,1]`, so — to match the MLE backend, which keeps such
parameters as true `_cons` constants — the first `zvar` column must be the constant intercept
of ones whenever any inefficiency parameter is homoscedastic. See the MCI twin for the full
rationale.
"""
function _validate_homoscedastic_zvar(ineff::MSLEIneffModel, hetero::Vector{Symbol},
                                      zvar_norm::AbstractMatrix)
    homo = setdiff(valid_hetero(ineff), hetero)
    if !isempty(homo) && !all(isone, @view zvar_norm[:, 1])
        error("The first `zvar` column must be a constant intercept of ones (the `_cons` " *
              "column) because inefficiency parameter(s) $(homo) are homoscedastic and are " *
              "evaluated at `coef·zvar[:,1]`. The supplied `zvar` has a non-constant first " *
              "column, so $(homo) would silently become heteroscedastic in the first " *
              "z-variable (the marginal effects and `predict` would then disagree with the " *
              "likelihood, and with the MLE backend). Prepend a column of ones to `zvar`, " *
              "or add $(homo) to `hetero`.")
    end
    return nothing
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
    n_sigma_u = :sigma_u_sq in hetero ? L : 1
    return n_mu + n_sigma_u
end

function ineff_plen(::Exponential_MSLE, L::Int, hetero::Vector{Symbol})
    return :sigma_u_sq in hetero ? L : 1
end

function ineff_plen(::HalfNormal_MSLE, L::Int, hetero::Vector{Symbol})
    return :sigma_u_sq in hetero ? L : 1
end

function ineff_plen(::Weibull_MSLE, L::Int, hetero::Vector{Symbol})
    n_lambda = :lambda in hetero ? L : 1
    n_k = :k in hetero ? L : 1
    return n_lambda + n_k
end

function ineff_plen(::Lognormal_MSLE, L::Int, hetero::Vector{Symbol})
    n_mu = :mu in hetero ? L : 1
    n_sigma = :sigma_u_sq in hetero ? L : 1
    return n_mu + n_sigma
end

function ineff_plen(::Lomax_MSLE, L::Int, hetero::Vector{Symbol})
    n_lambda = :lambda in hetero ? L : 1
    n_alpha = :alpha in hetero ? L : 1
    return n_lambda + n_alpha
end

function ineff_plen(::Rayleigh_MSLE, L::Int, hetero::Vector{Symbol})
    return :sigma_u_sq in hetero ? L : 1
end

function ineff_plen(::Erlang_MSLE, L::Int, hetero::Vector{Symbol})
    return :theta in hetero ? L : 1
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
    sigma_u_hetero = :sigma_u_sq in hetero

    n_mu = mu_hetero ? L : 1
    mu = idx:(idx+n_mu-1)
    idx += n_mu

    n_sigma_u = sigma_u_hetero ? L : 1
    sigma_u = idx:(idx+n_sigma_u-1)
    idx += n_sigma_u

    return (mu=mu, sigma_u=sigma_u, mu_hetero=mu_hetero, sigma_u_hetero=sigma_u_hetero), idx
end

function _ineff_ind(::Exponential_MSLE, idx, L, hetero)
    sigma_sq_hetero = :sigma_u_sq in hetero
    n_sigma = sigma_sq_hetero ? L : 1
    sigma_sq = idx:(idx+n_sigma-1)
    idx += n_sigma

    return (sigma_sq=sigma_sq, sigma_sq_hetero=sigma_sq_hetero), idx
end

function _ineff_ind(::HalfNormal_MSLE, idx, L, hetero)
    sigma_sq_hetero = :sigma_u_sq in hetero
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
    sigma_sq_hetero = :sigma_u_sq in hetero

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
    sigma_sq_hetero = :sigma_u_sq in hetero
    n_sigma = sigma_sq_hetero ? L : 1
    sigma_sq = idx:(idx+n_sigma-1)
    idx += n_sigma

    return (sigma_sq=sigma_sq, sigma_sq_hetero=sigma_sq_hetero), idx
end

function _ineff_ind(::Erlang_MSLE, idx, L, hetero)
    theta_hetero = :theta in hetero
    n_theta = theta_hetero ? L : 1
    theta = idx:(idx+n_theta-1)
    idx += n_theta

    return (theta=theta, theta_hetero=theta_hetero), idx
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
    chunks::Int  # Number of groups of observations processed at a time; bounds the memory of the likelihood, its derivatives and the efficiency indices

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
- `n_draws`: Number of draws per observation (default 4096 = 2^12)
- `multiRand`: Per-observation draws (`true`, N×D) or shared draws (`false`, 1×D)
- `GPU`: Whether to use GPU acceleration
- `chunks`: Number of groups of observations processed at a time; memory is proportional to
  N × `n_draws` / `chunks`
- `recycling`: 0 (the default) means no point is reused: observation i receives points (i − 1)·n_draws + 1, …,
  i·n_draws of one base-2 Halton sequence of N × n_draws points. A positive value is the length at which the
  sequence restarts from its first point.
- `halton_legacy`: `true` when that length came from `distinct_Halton_length`; the sequence is then shortened
  to the largest length of the form 2^m − 1 not above N × n_draws or the value given, as in versions before
  2.0.0.
"""
struct SFMethodSpec_MSLE
    method::Symbol
    draws::Union{Nothing, AbstractVecOrMat{<:AbstractFloat}}
    n_draws::Int
    multiRand::Bool
    GPU::Bool
    chunks::Int
    recycling::Int
    halton_legacy::Bool
end

# The layout of versions before 2.0.0: an explicit length keeps the rule of those versions.
SFMethodSpec_MSLE(method, draws, n_draws, multiRand, GPU, chunks, distinct_Halton_length::Integer) =
    SFMethodSpec_MSLE(method, draws, n_draws, multiRand, GPU, chunks, Int(distinct_Halton_length), true)

# A copy of the method `m` with `n_draws = D` and `chunks = chunks`, every other field unchanged: the method of
# the integration check, whose draws are those of a fit with `n_draws = D` and the same settings.
_with_draws(m::SFMethodSpec_MSLE, D::Integer, chunks::Integer) =
    SFMethodSpec_MSLE(m.method, m.draws, D, m.multiRand, m.GPU, chunks, m.recycling, m.halton_legacy)

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

struct LnSigmaSqInit <: InitSpec      # For inefficiency sigma_sq (ln_sigma_u_sq)
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

struct LambdaInit <: InitSpec         # For Weibull/Lomax lambda (a SCALE; the Exponential uses LnSigmaSqInit)
    values::Vector{Float64}
end

struct KInit <: InitSpec              # For Weibull k
    values::Vector{Float64}
end

struct AlphaInit <: InitSpec          # For Lomax alpha
    values::Vector{Float64}
end

struct ThetaInit <: InitSpec          # For Erlang θ
    values::Vector{Float64}
end

struct ThetaRhoInit <: InitSpec       # For copula theta_rho parameter (all copula types)
    values::Vector{Float64}
end

struct ScalingInit <: InitSpec        # For scaling function δ coefficients
    values::Vector{Float64}
end


# The value an omitted starting-value component takes. It is the same
# 0.1 the fit uses for every non-frontier parameter when it is handed no starting values
# at all, so that omitting one component and omitting the whole starting-value object
# agree component by component.
const _INIT_COMPONENT_DEFAULT = 0.1

# User-facing keyword name for each init component. Used in error messages so a reader is
# told the name they typed rather than the internal struct name.
_init_kwarg_name(::Type{FrontierInit})   = "frontier"
_init_kwarg_name(::Type{ScalingInit})    = "scaling"
_init_kwarg_name(::Type{MuInit})         = "mu"
_init_kwarg_name(::Type{LnSigmaSqInit})  = "ln_sigma_u_sq"
_init_kwarg_name(::Type{LnSigmaVSqInit}) = "ln_sigma_v_sq"
_init_kwarg_name(::Type{LnNuMinus2Init}) = "ln_nu_minus_2"
_init_kwarg_name(::Type{LnBInit})        = "ln_b"
_init_kwarg_name(::Type{LambdaInit})     = "ln_lambda"
_init_kwarg_name(::Type{KInit})          = "ln_k"
_init_kwarg_name(::Type{AlphaInit})      = "ln_alpha"
_init_kwarg_name(::Type{ThetaInit})      = "ln_theta"
_init_kwarg_name(::Type{ThetaRhoInit})   = "theta_rho"
_init_kwarg_name(T::Type)                = string(nameof(T))

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

# For depvar: unwrap [yvar] → yvar.
_to_vector(x::AbstractVector{<:Real}) = x
# Twin of the MCI normalizer's guard. A list of several vectors gets its own message:
# `only(x)` alone fails with Base's "Collection has multiple elements, must contain exactly 1
# element" — it names neither the argument nor the list form nor the count, and it would preempt
# the analytic backend's own clear message on every public route, since the unified constructor
# builds a simulation spec first.
function _to_vector(x::AbstractVector{<:AbstractVector})
    length(x) == 1 ||
        error("`depvar` must be a single vector, not a list of $(length(x)). The list form " *
              "`depvar = [y]` exists to accept one column; for several columns there is no " *
              "single dependent variable to fit.")
    return only(x)
end
# N×1 matrix input is the intended case; anything wider is refused, since flattening an N×2
# depvar would give a 2N-long vector and a spec with N=2N (panel and MLE reject it too).
_to_vector(x::AbstractMatrix{<:Real}) =
    size(x, 2) == 1 ? vec(x) :
    error("`depvar` must be a vector or a single-column matrix, got size $(size(x)).")

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
        # Host target: a device-resident source (e.g. user-supplied CuArray draws
        # in a GPU=false fit) must come back to host or it device-mixes in the NLL
        # broadcasts. No-op for host arrays.
        return source isa Array ? source : Array(source)
    end
end

function _to_device_array(target::AbstractArray, source::AbstractMatrix)
    target_type_str = string(typeof(target).name.wrapper)
    if target_type_str == "CuArray"
        CuArrayConstructor = typeof(target).name.wrapper
        return CuArrayConstructor(source)
    else
        # Host target: pull device-resident sources back to host.
        return source isa Array ? source : Array(source)
    end
end

# True when both arrays are of the same kind: both `Array`s, or both `CuArray`s.
_same_array_kind(a, b) = typeof(a).name.wrapper === typeof(b).name.wrapper

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

# User-supplied draws must be probabilities in [0, 1). A draw of exactly 1 maps to an
# infinite inefficiency under every inverse-CDF or change-of-variable rule: in the MCI
# T-approach u is capped at 1/clamp_lo while the Jacobian s/(1-t) is infinite, so the
# likelihood is NaN; in MSLE the quantile clamp silently moves the draw to the upper
# bound. Zero is the support lower bound and harmless. `minimum`/`maximum` are
# reductions, so the check also runs on CuArray draws without scalar indexing.
function _validate_user_draws(user_draws)
    isempty(user_draws) && error("`draws` is empty.")
    dmin = minimum(user_draws)
    dmax = maximum(user_draws)
    (dmin >= 0 && dmax < 1) ||
        error("`draws` must lie in [0, 1): found min = $dmin, max = $dmax. " *
              "The contract is half-open: 0 is the lower bound of the support and is " *
              "accepted, 1 is not. The MSLE quantile kernels do not return a NaN at 1 — " *
              "they clamp it to their upper bound, so the draw silently becomes a different " *
              "draw and the simulated likelihood is not the one the draws describe. " *
              "Note that a draw just below 1 can round to exactly 1 when converted to the " *
              "data's element type — supply draws already in that type, or move them off 1.")
    return nothing
end

# One rule for user-supplied draws, whatever `multiRand` was passed: supplying `draws` forces
# multiRand=false at `sfmodel_method` (the Halton generator is not used, so there is no generation
# mode left to choose), and the ROW COUNT says how the draws are read.
#   * a vector, a 1×D row, or a D×1 column -> one sequence of D draws shared by every observation
#     (a single-column matrix is that same sequence written vertically);
#   * an N×D matrix                        -> one row of D draws per observation;
#   * anything else                        -> error: with a row count other than 1 or N there is no
#     way to split the block at the observation level.
# Returns a 1×D (shared) or N×D (per-observation) matrix.
function _shape_user_draws(user_draws, N::Int)
    ndims(user_draws) == 1 && return reshape(user_draws, 1, length(user_draws))
    nr, nc = size(user_draws)
    nc == 1 && return reshape(user_draws, 1, nr)     # D×1 is the same sequence as 1×D
    nr == 1 && return user_draws                     # 1×D: shared
    nr == N && return user_draws                     # N×D: one row per observation
    error("`draws` has size $(size(user_draws)): the number of rows must be 1 (one sequence of " *
          "draws shared by every observation) or N=$N (one row of draws per observation). " *
          "A vector or a single-column matrix is read as a shared sequence.")
end

"""
    _prepare_draws(depvar_ref, N, T, user_draws, n_draws, multiRand, recycling, halton_legacy;
                   warn_repeated = true)

Prepare the draws for MSLE estimation. If `user_draws` is `nothing`, generates base-2 Halton draws with
`sf_halton_draws`. Returns `(draws_vec, draws_2D)`.

- `multiRand=true`: N×D matrix: observation i takes the points at positions (i − 1)·D + 1, …, i·D of one
  Halton sequence, which restarts after the period `sf_halton_period` gives for `recycling` and
  `halton_legacy` (no restart when `recycling` is 0)
- `multiRand=false`: 1×D matrix: the first D points, shared by every observation
- `warn_repeated=false`: no warning when the generated draws give two observations identical draws (the
  integration check builds its draws this way)
"""
function _prepare_draws(depvar_ref::AbstractVector{T}, N::Int, ::Type{T},
                        user_draws, n_draws::Int, multiRand::Bool,
                        recycling::Int, halton_legacy::Bool; warn_repeated::Bool = true) where {T}
    if isnothing(user_draws)
        if multiRand
            period = sf_halton_period(N, n_draws; recycling = recycling, legacy = halton_legacy)
            warn_repeated && sf_warn_repeated_rows(N, n_draws, period)
            halton_cpu = sf_halton_draws(N, n_draws; T = T, period = period)
        else
            halton_cpu = sf_halton_draws(1, n_draws; T = T)        # the first n_draws points, shared
        end
        draws_2D = _to_device_array(depvar_ref, halton_cpu)   # CPU: the same matrix, not a copy; GPU: one upload
        draws_vec = vec(draws_2D)                             # shares draws_2D's memory
    else
        # User provided draws - validate shape and convert
        depvar_is_gpu = string(typeof(depvar_ref).name.wrapper) == "CuArray"
        draws_is_gpu = string(typeof(user_draws).name.wrapper) == "CuArray"
        if depvar_is_gpu != draws_is_gpu
            @warn "Type inconsistency: `depvar` is on $(depvar_is_gpu ? "GPU" : "CPU") but `draws` is on $(draws_is_gpu ? "GPU" : "CPU"). " *
                  "Consider letting the program auto-generate Halton sequences with the correct device type by specifying `n_draws` instead of providing `draws`."
        end

        _validate_user_draws(user_draws)          # in the user's own element type
        # `multiRand` is not consulted: supplying `draws` forces it false, and the row count decides
        # shared (1×D) versus per-observation (N×D). Device-convert like the auto-generated branches
        # — user draws must live where the data lives.
        shaped = _shape_user_draws(user_draws, N)
        if eltype(shaped) === T && _same_array_kind(shaped, depvar_ref)
            draws_2D = shaped                         # the user's array: already the data's element type and device
        else
            shaped_T = T.(shaped)
            _validate_user_draws(shaped_T)            # and again after narrowing to T
            draws_2D = _to_device_array(depvar_ref, shaped_T)
        end
        draws_vec = vec(draws_2D)
    end

    return draws_vec, draws_2D
end

"""
    _assemble_MSLE_spec(spec::SFModelSpec_MSLE{T}, method::SFMethodSpec_MSLE; warn_repeated = true)

Merge a model specification and method specification into the internal
`sfmodel_MSLE_spec{T}` struct. Performs deferred computations:
GPU conversion, draw generation, and constant computation. `warn_repeated` is passed to
`_prepare_draws`.
"""
function _assemble_MSLE_spec(spec::SFModelSpec_MSLE{T}, method::SFMethodSpec_MSLE;
                             warn_repeated::Bool = true) where {T}
    # 1. GPU conversion (deferred from spec)
    depvar, frontier, zvar = _maybe_gpu_convert(spec.depvar, spec.frontier, spec.zvar, method.GPU)

    # 1b. GPU conversion for scaling_zvar (if scaling)
    scaling_zvar = if spec.scaling && method.GPU
        _to_device_array(depvar, spec.scaling_zvar)
    else
        spec.scaling_zvar
    end

    # 2. Prepare draws
    draws_vec, draws_2D = _prepare_draws(depvar, spec.N, T,
        method.draws, method.n_draws, method.multiRand, method.recycling, method.halton_legacy;
        warn_repeated = warn_repeated)

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

ln_sigma_u_sq(x::Real, xs::Real...)     = LnSigmaSqInit(_to_vec(x, xs...))
ln_sigma_u_sq(v::AbstractVector)        = LnSigmaSqInit(_to_vec(v))
ln_sigma_u_sq(x::Real)                  = LnSigmaSqInit(_to_vec(x))

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
        push!(required, (LnSigmaSqInit, "ln_sigma_u_sq", hetero_len(:sigma_u_sq)))

    elseif spec.ineff in (:HalfNormal, :Rayleigh)
        push!(required, (LnSigmaSqInit, "ln_sigma_u_sq", hetero_len(:sigma_u_sq)))

    elseif spec.ineff == :Exponential
        push!(required, (LnSigmaSqInit, "ln_sigma_u_sq", hetero_len(:sigma_u_sq)))

    elseif spec.ineff == :Weibull
        push!(required, (LambdaInit, "ln_lambda", hetero_len(:lambda)))
        push!(required, (KInit, "ln_k", hetero_len(:k)))

    elseif spec.ineff == :Lomax
        push!(required, (LambdaInit, "ln_lambda", hetero_len(:lambda)))
        push!(required, (AlphaInit, "ln_alpha", hetero_len(:alpha)))

    elseif spec.ineff in (:Erlang2, :Erlang3, :Erlang4)
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
2. **Component mode**: Supply individual components (frontier, mu, ln_sigma_u_sq, etc.).

# Arguments
- `spec::SFModelSpec_MSLE`: The model specification (required)
- `init`: Complete initial value vector (optional). If provided, other parameters are ignored.
         Can be vector, row vector, or tuple. Length must match the number of model parameters.
- `frontier`: Frontier coefficients (K values), can be vector, row vector, or tuple
- `mu`: Mu parameter for TruncatedNormal, Lognormal (if applicable)
- `ln_sigma_u_sq`: Log sigma squared for TruncatedNormal, HalfNormal, Lognormal, Rayleigh
- `ln_sigma_v_sq`: log σ\\_v², the noise variance for Normal and the squared scale of
  t\\_ν(0, σ\\_v) for StudentT (Var(v) = σ\\_v²·ν/(ν − 2))
- `ln_nu_minus_2`: Log(nu-2) for StudentT
- `ln_b`: Log scale for Laplace
- `ln_lambda`: Log lambda (scale) for Weibull
- `ln_k`: Log shape k for Weibull
- `ln_lambda`: Log lambda (scale) for Lomax/Weibull
- `ln_alpha`: Log alpha for Lomax
- `ln_theta`: Log theta for Erlang
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
    ln_sigma_u_sq = [0.0],          # scalar as single-element vector
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
    ln_sigma_u_sq = nothing,
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
        if message && any(x -> x !== nothing, (frontier, scaling, mu, ln_sigma_u_sq, ln_sigma_v_sq,
                                                ln_nu_minus_2, ln_b, lambda, k, alpha, theta, theta_rho))
            @warn "Using `init` instead of function-specific init."
        end
        init_vec = _to_init_vec(init)
        expected_len = plen(spec.model, spec.K, spec.L, spec.hetero; L_scaling=spec.L_scaling)
        if length(init_vec) != expected_len
            error("Length mismatch in sfmodel_init(): expected $expected_len parameters, got $(length(init_vec)).")
        end
        return init_vec
    end

    # Mode 2: Component mode - user supplies individual parameters
    if frontier === nothing
        # Default to OLS estimates for frontier coefficients. Solved on the host in
        # Float64 whatever the specification's element type, so that this default is
        # the same number the analytic (Float64-only) fit computes for itself, and so
        # that it does not depend on whether the arrays live on a device.
        frontier = Float64.(Array(spec.frontier)) \ Float64.(Array(spec.depvar))
        if message
            @info "Using OLS estimates as default initial values for frontier coefficients."
        end
    end

    # Convert all inputs to Vector{Float64} (or nothing)
    frontier_vec = _to_init_vec(frontier)
    scaling_vec = _to_init_vec(scaling)
    mu_vec = _to_init_vec(mu)
    ln_sigma_u_sq_vec = _to_init_vec(ln_sigma_u_sq)
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
    if ln_sigma_u_sq_vec !== nothing
        init_dict[LnSigmaSqInit] = LnSigmaSqInit(ln_sigma_u_sq_vec)
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

    # Determine required equations based on spec
    required = _get_init(spec)

    # Every component is optional. A component that is NOT supplied
    # takes exactly the value the fit itself would have used had no starting values been
    # supplied at all: the OLS estimate for the frontier, filled in above, and
    # _INIT_COMPONENT_DEFAULT for every other parameter, matching
    # `vcat(β_ols, fill(0.1, n_total - K))` in sfmodel_MSLE_fit(). So omitting one
    # component gives that component the same starting value as omitting the whole
    # starting-value object. Only what IS supplied is checked, for length and for
    # applicability to this model.
    length_mismatches = String[]

    for (req_type, req_name, expected_len) in required
        if !haskey(init_dict, req_type)
            init_dict[req_type] = req_type(fill(_INIT_COMPONENT_DEFAULT, expected_len))
        else
            # Check length
            provided_len = length(init_dict[req_type].values)
            if provided_len != expected_len
                push!(length_mismatches, "$req_name: expected $expected_len, got $provided_len")
            end
        end
    end

    # Report all length mismatches at once
    if !isempty(length_mismatches)
        mismatch_list = join(length_mismatches, "; ")
        error("Length mismatch(es) in sfmodel_init(): $mismatch_list.")
    end

    # Check for extra (unused) specifications
    provided_types = Set(keys(init_dict))
    required_types = Set(r[1] for r in required)
    extras = setdiff(provided_types, required_types)
    if !isempty(extras)
        extra_names = join(sort([_init_kwarg_name(e) for e in extras]), ", ")
        error("Unused init component(s) in sfmodel_init(): $extra_names. " *
              "Not part of the current model specification.")
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
                      main_solver, main_opt, store_trace=false)

Construct optimization options for `sfmodel_MSLE_fit()`.

# Arguments
- `warmstart_solver=nothing`: Warmstart optimizer, e.g., `NelderMead()`, `BFGS()`. Optional.
- `warmstart_opt=nothing`: Warmstart options as a NamedTuple, e.g., `(iterations = 400, g_abstol = 1e-5)`. Optional.
- `main_solver`: Main optimizer, e.g., `Newton()`, `BFGS()`. Required.
- `main_opt`: Main options as a NamedTuple, e.g., `(iterations = 2000, g_abstol = 1e-8)`. Required.
- `store_trace=false`: When `true`, or when `main_opt` itself sets `store_trace = true`, the main optimizer's
  iteration trace (iterations, objective values and gradient norms) is stored in the result as `trace`;
  otherwise `trace` is `nothing`. `plot(result, :convergence)` draws it from the result of `sfmodel_fit`.

If `warmstart_solver` is not provided, the warmstart stage will be skipped, and a `warmstart_opt` is
refused: it needs a `warmstart_solver`.

With `noise = :Laplace`, a fit whose main stage stops without convergence is continued with `BFGS()` from its
endpoint with the same `main_opt`; not when the main solver is `BFGS()` or `LBFGS()`, or when `main_opt` allows no
iterations. With the default optimizer settings, whether omitted or written out, the Newton stage of a Laplace fit
runs at most 30 iterations before this finish; pass other main settings in `optim_options` to choose other budgets.

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
    main_opt,
    store_trace::Bool = false
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

    # Convert NamedTuple to Optim.Options for main_opt. When store_trace is requested, Optim's
    # `store_trace=true` is forced so the main-solver trace is kept (a later `store_trace` key wins over
    # one in main_opt); the fit reads this flag off `main_opt.store_trace` and copies the trace onto the
    # result.
    m_opt = store_trace ? Optim.Options(; main_opt..., store_trace = true) :
                          Optim.Options(; main_opt...)

    # Convert NamedTuple to Optim.Options for warmstart_opt if provided. Without a solver there is no
    # warmstart stage, so a `warmstart_opt` given without one is refused.
    if warmstart_solver !== nothing
        if warmstart_opt === nothing
            # Default warmstart options if solver provided but no options
            ws_opt = Optim.Options(iterations = 100, g_abstol = 1e-3)
        else
            ws_opt = Optim.Options(; warmstart_opt...)
        end
    else
        warmstart_opt === nothing ||
            error("`warmstart_opt` needs a `warmstart_solver`: without a solver there is no " *
              "warmstart stage to configure. Add a solver (e.g. `warmstart_solver = NelderMead()`), " *
              "or remove `warmstart_opt`.")
        ws_opt = nothing
    end

    return sfmodel_MSLE_optim(warmstart_solver, ws_opt, main_solver, m_opt)
end

# Capture the main optimizer's iteration trace as a NamedTuple for
# plot(:convergence). Returns `nothing` unless the fit's Optim.Options carried
# store_trace=true (set by `store_trace=true` in `sfmodel_MSLE_opt` or in the unified `sfmodel_opt`).
function _sf_optim_trace(optres, want::Bool)
    want || return nothing
    fv = Optim.f_trace(optres)
    return (iteration = collect(0:(length(fv) - 1)),
            value     = fv,
            g_norm    = Optim.g_norm_trace(optres))
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
        push!(eqnames, :sigma_u_sq in hetero ? "ln_σᵤ²" : "ln_σᵤ²")
    elseif ineff isa Exponential_MSLE
        push!(eqnames, :sigma_u_sq in hetero ? "ln_σᵤ²" : "ln_σᵤ²")
    elseif ineff isa HalfNormal_MSLE
        push!(eqnames, :sigma_u_sq in hetero ? "ln_σᵤ²" : "ln_σᵤ²")
    elseif ineff isa Weibull_MSLE
        push!(eqnames, :lambda in hetero ? "ln_λ" : "ln_λ")
        push!(eqnames, :k in hetero ? "ln_k" : "ln_k")
    elseif ineff isa Lognormal_MSLE
        push!(eqnames, :mu in hetero ? "μ" : "μ")
        push!(eqnames, :sigma_u_sq in hetero ? "ln_σᵤ²" : "ln_σᵤ²")
    elseif ineff isa Lomax_MSLE
        push!(eqnames, :lambda in hetero ? "ln_λ" : "ln_λ")
        push!(eqnames, :alpha in hetero ? "ln_α" : "ln_α")
    elseif ineff isa Rayleigh_MSLE
        push!(eqnames, :sigma_u_sq in hetero ? "ln_σᵤ²" : "ln_σᵤ²")
    elseif ineff isa Erlang_MSLE
        push!(eqnames, :theta in hetero ? "ln_θ" : "ln_θ")
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
        n_sigma = :sigma_u_sq in hetero ? L : 1
        idx += n_sigma
    elseif ineff isa Exponential_MSLE
        push!(eq_indices, idx)  # ln_σᵤ²
        n_sigma = :sigma_u_sq in hetero ? L : 1
        idx += n_sigma
    elseif ineff isa HalfNormal_MSLE
        push!(eq_indices, idx)  # ln_σᵤ²
        n_sigma = :sigma_u_sq in hetero ? L : 1
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
        push!(eq_indices, idx)  # ln_σᵤ²
        n_sigma = :sigma_u_sq in hetero ? L : 1
        idx += n_sigma
    elseif ineff isa Lomax_MSLE
        push!(eq_indices, idx)  # ln_λ
        n_lambda = :lambda in hetero ? L : 1
        idx += n_lambda
        push!(eq_indices, idx)  # ln_α
        n_alpha = :alpha in hetero ? L : 1
        idx += n_alpha
    elseif ineff isa Rayleigh_MSLE
        push!(eq_indices, idx)  # ln_σᵤ²
        n_sigma = :sigma_u_sq in hetero ? L : 1
        idx += n_sigma
    elseif ineff isa Erlang_MSLE
        push!(eq_indices, idx)  # ln_θ
        n_theta = :theta in hetero ? L : 1
        idx += n_theta
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
                      "Column $j is constant. Remove it for parameter identification.")
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
    # Row counts must match `depvar`: without this, MCI/MSLE accepted a mismatched frontier/zvar and
    # died later with a raw DimensionMismatch from the OLS baseline or a reshape — and only for the
    # eight distributions with no co-built MLE spec, so the behaviour flipped with `ineff`. The panel
    # (sf_panel_v20.jl) and MLE (SFmainfun.jl) backends have always checked this.
    size(frontier_norm, 1) == N ||
        error("`frontier` has $(size(frontier_norm,1)) rows, expected $N (the `depvar` length).")
    size(zvar_norm, 1) == N ||
        error("`zvar` has $(size(zvar_norm,1)) rows, expected $N (the `depvar` length).")
    if scaling && scaling_zvar !== nothing
        size(scaling_zvar, 1) == N ||
            error("scaling `zvar` has $(size(scaling_zvar,1)) rows, expected $N (the `depvar` length).")
    end

    # Build model and validate
    model = _build_model(noise, ineff; copula=copula)
    _validate_hetero(model.ineff, hetero_vec)
    _validate_homoscedastic_zvar(model.ineff, hetero_vec, zvar_norm)

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
    hetero = [:mu, :sigma_u_sq]
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
    hetero_vec = is_scaling ? Symbol[] :
        (hetero isa Symbol ?
            error("Invalid `hetero`: :$hetero. Use a Vector{Symbol} like [:$hetero], " *
                  "or use :scaling for the scaling property model.") : hetero)

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
        if !is_scaling && :sigma_u_sq in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "ln_σᵤ²"; idx += 1
        end
    elseif ineff_type isa HalfNormal_MSLE
        if !is_scaling && :sigma_u_sq in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "ln_σᵤ²"; idx += 1
        end
    elseif ineff_type isa Exponential_MSLE
        if !is_scaling && :sigma_u_sq in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "ln_σᵤ²"; idx += 1
        end
    elseif ineff_type isa Lognormal_MSLE
        if !is_scaling && :mu in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "μ"; idx += 1
        end
        if !is_scaling && :sigma_u_sq in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "ln_σᵤ²"; idx += 1
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
        if !is_scaling && :sigma_u_sq in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "ln_σᵤ²"; idx += 1
        end
    elseif ineff_type isa Erlang_MSLE
        if !is_scaling && :theta in hetero_vec
            for name in zvar_names; varnames[idx] = name; idx += 1; end
        else
            varnames[idx] = "ln_θ"; idx += 1
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
    # Bare non-:scaling Symbol: fail with the keyword ctor's clean message —
    # the varnames ternaries below (`:sigma_u_sq in hetero` etc.) otherwise throw a
    # cryptic `MethodError: iterate(::Symbol)` first.
    if hetero isa Symbol
        error("Invalid `hetero`: :$hetero. Use a Vector{Symbol} like [:$hetero], " *
              "or use :scaling for the scaling property model.")
    end

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
        varnames[vi] = :sigma_u_sq in hetero ? "_cons" : "ln_σᵤ²";  vi += 1
    elseif ineff_type isa HalfNormal_MSLE
        varnames[vi] = :sigma_u_sq in hetero ? "_cons" : "ln_σᵤ²";  vi += 1
    elseif ineff_type isa Exponential_MSLE
        varnames[vi] = :sigma_u_sq in hetero ? "_cons" : "ln_σᵤ²";  vi += 1
    elseif ineff_type isa Lognormal_MSLE
        varnames[vi] = :mu in hetero ? "_cons" : "μ";  vi += 1
        varnames[vi] = :sigma_u_sq in hetero ? "_cons" : "ln_σᵤ²";  vi += 1
    elseif ineff_type isa Weibull_MSLE
        varnames[vi] = :lambda in hetero ? "_cons" : "ln_λ";  vi += 1
        varnames[vi] = :k in hetero ? "_cons" : "ln_k";  vi += 1
    elseif ineff_type isa Lomax_MSLE
        varnames[vi] = :lambda in hetero ? "_cons" : "ln_λ";  vi += 1
        varnames[vi] = :alpha in hetero ? "_cons" : "ln_α";  vi += 1
    elseif ineff_type isa Rayleigh_MSLE
        varnames[vi] = :sigma_u_sq in hetero ? "_cons" : "ln_σᵤ²";  vi += 1
    elseif ineff_type isa Erlang_MSLE
        varnames[vi] = :theta in hetero ? "_cons" : "ln_θ";  vi += 1
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
    sfmodel_method(; method=:MSLE, draws=nothing, n_draws=4096, multiRand=true, GPU=false, chunks=10,
                   recycling=false, distinct_Halton_length=nothing)

Construct a method specification (how to estimate). Returns `SFMethodSpec_MSLE`.

# Arguments
- `method::Symbol=:MSLE`: Estimation method (currently only `:MSLE` supported)
- `draws=nothing`: User-supplied draws in [0, 1); auto-generated Halton if `nothing`. The ROW COUNT
  decides how they are read, and `multiRand` is not part of it: a length-D vector, a 1×D row matrix or
  a D×1 column matrix is one sequence of D draws shared by every observation; an N×D matrix is one row of D
  draws per observation. Any other row count is an error. Supplying `draws` forces `multiRand=false` and makes
  `n_draws` irrelevant (D comes from the draws themselves). The fit keeps the draws for `predict()`. When
  the array already has the data's element type and lives where the data live (a host `Array`, or a
  `CuArray` with `GPU = true`), the fit uses the array itself, not a copy, so changing the array afterwards
  changes what `predict()` computes.
- `n_draws::Int=4096`: Number of Halton draws per observation (2^12; used only when `draws` is not provided).
  Use powers of two.
- `multiRand::Bool=true`: generation mode, used only when `draws` is not supplied. `true` gives observation i
  (firm i in panel models) the `n_draws` consecutive points (i − 1)·n_draws + 1, …, i·n_draws of one base-2
  Halton sequence of N × `n_draws` points, so no point is used twice unless `recycling` or the deprecated
  `distinct_Halton_length` is given; `false` gives every observation the same first `n_draws` points.
- `GPU::Bool=false`: If `true`, convert data to GPU arrays. Requires `using CUDA` before calling.
- `chunks::Int=10`: Number of groups of observations (firms, for panel models) processed at a time (MCI/MSLE
  only). The memory of the likelihood, its derivatives and the efficiency indices is proportional to
  N × `n_draws` / `chunks`, on the CPU and on the GPU.
- `recycling=false`: reuse of Halton points across observations, for generated per-observation draws only
  (`multiRand = true`, no `draws`). `false` (default): no point is reused. `true`: the sequence is capped at
  L = 2^24 − 1 = 16,777,215 points and restarts from its first point when N × `n_draws` exceeds L; observation
  i receives the points at positions mod1((i − 1)·n_draws + j, L), j = 1, …, n_draws. An integer L with
  `n_draws` ≤ L ≤ 2^24 − 1 sets another cap. When N × `n_draws` exceeds L, observations i and i + L/g receive
  identical draws, where g is the greatest common divisor of `n_draws` and L; the fit warns when the sample
  has more than L/g observations. With `recycling = true` (L = 2^24 − 1, an odd number) any power of two for
  `n_draws` gives g = 1. With `draws` or `multiRand = false`, `recycling` is an error.
- `distinct_Halton_length=nothing`: deprecated; it gives a warning. Earlier `:MCI` and `:MSLE` scripts that set
  it still run and, with the same `n_draws`, get the draws they got before 2.0.0. To reproduce the draws of a
  fit of those versions that used the default length, give `recycling = 2^15 - 1` and that fit's `n_draws`
  (1,024 by default): the draws are identical whenever N × `n_draws` is at least 32,767. Use `recycling` for
  new work; giving both is an error.

# Example
```julia
# Default method (MSLE with 4096 Halton draws per observation, CPU)
method = sfmodel_method()

# Custom method (more draws, GPU enabled)
method = sfmodel_method(n_draws=2^14, GPU=true)

# More draws per observation: no other setting is needed
method = sfmodel_method(n_draws=2^16)

# Recycling: at most 2^24 - 1 points, reused when N × n_draws is larger
method = sfmodel_method(recycling=true)
```
"""
function sfmodel_method(;
    method::Symbol = :MSLE,
    draws::Union{Nothing, AbstractVecOrMat{<:AbstractFloat}} = nothing,
    n_draws::Int = 2^12,
    multiRand::Bool = true,
    GPU::Bool = false,
    chunks::Int = 10,
    recycling::Union{Bool,Integer} = false,
    distinct_Halton_length::Union{Int,Nothing} = nothing)

    method == :MSLE || error("Currently only method=:MSLE is supported. Got: :$method")
    # Supplying `draws` means the Halton generator is not used, so there is no generation mode left
    # to choose: `multiRand` is forced false and the draws' own row count decides whether they are
    # shared by every observation (1 row) or per observation (N rows).
    _multiRand = isnothing(draws) ? multiRand : false

    _rec, _legacy = sf_recycling_request(recycling, distinct_Halton_length, n_draws, isnothing(draws) && multiRand)
    return SFMethodSpec_MSLE(method, draws, n_draws, _multiRand, GPU, chunks, _rec, _legacy)
end


# ============================================================================
# Section 3: Quantile Functions
# ============================================================================

# One draw of the truncated-normal quantile. For
# y = -μ/(σ√2) up to 8 this is the closed form below, bit for bit. Beyond, erfc(y) heads for
# underflow (NaN derivatives from μ/σ ≈ -37.5, infinite draws below -38.5 in Float64) and μ
# cancels against σ√2·erfcinv(...), so the draw is σ√2·t with t from `sf_tn_quantile_tail`
# (src/sf_numerics.jl).
@inline function _tn_quantile(μ, σ, r, sqrt2, inv_sqrt2)
    y = -μ * inv_sqrt2 / σ
    # r + zero(y): with Float32 draws, log1p(-r) would otherwise run in single precision (and
    # differ between CPU and GPU by an ulp); exact for plain floats, zero partials for Duals
    y > 8 && return σ * sqrt2 * sf_tn_quantile_tail(y, r + zero(y))
    return max(μ + σ * sqrt2 * erfcinv(erfc(y) * (1 - r)), zero(μ * σ * r))
end

"""
    myTruncatedNormalQuantile(; μ, σ, r, sqrt2=sqrt(2), inv_sqrt2=inv(sqrt(2)), clamp_lo=nothing, clamp_hi=nothing)

Compute quantile of TruncatedNormal(μ, σ; lower=0) at probability r.
Works on both CPU and GPU arrays via broadcasting.
"""
function myTruncatedNormalQuantile(; μ, σ, r,
                                    sqrt2=sqrt(2), inv_sqrt2=inv(sqrt(2)),
                                    clamp_lo=nothing, clamp_hi=nothing)
    # F(u) = r for u ~ N(μ, σ²) truncated below at 0 gives
    #     u = μ + σ·√2·erfcinv( erfc(-μ/(σ√2)) · (1-r) ).
    # The inverse-error-function form, erfinv(r + (r-1)·erf(μ/(σ√2))), sends its argument to 1
    # for EVERY draw as μ/σ falls, because r + (r-1)(-1+δ) = 1 + (r-1)δ, and the
    # difference from 1 then falls below the spacing of the element type.  Clamping
    # that argument caps u at μ + 4.62σ in Float32 and μ + 7.78σ in Float64; not
    # clamping it returns Inf.  Against a 256-bit reference over μ/σ ∈ [-8, -0.5] the
    # clamped form is wrong by up to 4.44 absolute in Float32 and 1.28 in Float64,
    # while the form below is accurate to 3.9e-5 and 6.4e-15.  erfc and erfcinv are
    # both AD-differentiable and both run on the GPU.  clamp_lo / clamp_hi are still
    # accepted so callers need not change, but the argument of erfcinv is a product of
    # two non-negative factors and cannot leave its domain, so they are not applied.
    # The max(., 0) is the support constraint, not a numerical fudge: u is truncated below
    # at 0, so no quantile is negative.  It binds only at the one corner where erfc
    # saturates at its own ceiling of 2 (mu/sigma above about 8.5 in Float64, 5.4 in
    # Float32) AND the draw is exactly 0, which the [0,1) draw contract admits; there the
    # true quantile is 0 and the unguarded expression returns -Inf.
    # Beyond μ/σ = -8√2 the draw comes from the far-tail solver (see _tn_quantile).
    return @. _tn_quantile(μ, σ, r, sqrt2, inv_sqrt2)
end

"""
    myTruncatedNormalQuantile!(out; μ, σ, r, sqrt2=sqrt(2), inv_sqrt2=inv(sqrt(2)), clamp_lo=nothing, clamp_hi=nothing)

In-place version of `myTruncatedNormalQuantile`. Stores result in `out`.
"""
function myTruncatedNormalQuantile!(out; μ, σ, r,
                                     sqrt2=sqrt(2), inv_sqrt2=inv(sqrt(2)),
                                     clamp_lo=nothing, clamp_hi=nothing)
    # inverse CDF with no argument near 1; see myTruncatedNormalQuantile and _tn_quantile above
    @. out = _tn_quantile(μ, σ, r, sqrt2, inv_sqrt2)
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
    myExponentialQuantile(; σᵤ², r)

Compute quantile (inverse CDF) of Exponential at probability r, parameterised by the
squared scale σᵤ², which for the exponential is also Var(u).  The scale itself is
σᵤ = √σᵤ² = E(u).  (Weibull and Lomax in this file take λ, which is their SCALE; the
exponential's parameter is a squared scale and is named accordingly, so that one
symbol does not mean two things.)
F⁻¹(p; σᵤ²) = -ln(1-p) * √σᵤ²

# Arguments
- `σᵤ²`: Squared-scale parameter (= Var(u), positive scalar or array)
- `r`: Probability values in (0,1)
"""
function myExponentialQuantile(; σᵤ², r)
    return @. -log(max(1 - r, eps(eltype(r)))) * sqrt(σᵤ²)
end

"""
    myExponentialQuantile!(out; σᵤ², r, clamp_lo=nothing)

In-place version of `myExponentialQuantile`. Stores result in `out`.
Squared-scale parameterisation: Var(u) = σᵤ², scale = √σᵤ².

# Arguments
- `out`: Pre-allocated output array (modified in-place)
- `σᵤ²`: Squared-scale parameter (= Var(u), positive scalar or array)
- `r`: Probability values in (0,1)
- `clamp_lo`: Optional lower clamp for (1-r) to avoid log(0)
"""
function myExponentialQuantile!(out; σᵤ², r, clamp_lo=nothing)
    if clamp_lo !== nothing
        @. out = -log(max(1 - r, clamp_lo)) * sqrt(σᵤ²)
    else
        @. out = -log(max(1 - r, eps(eltype(r)))) * sqrt(σᵤ²)
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

"""
    myErlangStdQuantile(p, m, lgm)

Standardized Erlang quantile q_m(p): the quantile of Gamma(shape=m, scale=1) at
probability p, for the integer order m pinned by the Erlang distribution.
`lgm = loggamma(m) = log((m-1)!)` is passed precomputed.

Erlang has no closed-form inverse CDF for m ≥ 2, but the obstacle that rules out
Gamma-MSLE does not apply: the Erlang shape is pinned and θ is a pure scale, so
Q(p; m, θ) = θ·q_m(p) and this function runs ONLY on the fixed Halton draws. It
never sees an estimated parameter, hence no dual numbers flow through the
iteration and the likelihood's derivatives with respect to θ are exact by
construction. (For Gamma the estimated shape k sits inside the quantile, which
would force a slow, non-AD, non-GPU inversion at every likelihood evaluation.)

Safeguarded Newton on the closed-form CDF F(x) = 1 − e^(−x)·Σ_{n<m} xⁿ/n!,
seeded in the lower tail by the leading-order series F ≈ x^m/m! and in the upper
tail by the exponential-tail start. The lower branch (p < 1/2) evaluates F through
the tail series F(x) = e^(−x)·Σ_{n≥m} xⁿ/n! = f(x)·(x/m)·Σ_{j≥0} xʲ·m!/(m+j)!
(f the density), which has no cancellation: the literal 1 − e^(−x)·S loses about
eps/p relative accuracy when F is tiny (in Float32 the quantile at p = 1e-7 was
15–31% off for m = 2, 3, 4; in Float64 about 1e-5 at p = 1e-12). The upper branch
iterates on the log-survival function to avoid the vanishing density. Eight
iterations agree with Distributions.quantile(Gamma(m,1), p) to working precision
over p ∈ [1e-15, 1−1e-15] (Float64) and p ∈ [1e-15, prevfloat(1f0)] (Float32) for
m = 2, 3, 4 (test/test_erlang.jl). The input is pinned to [floatmin(T),
prevfloat(one(T))]: F⁻¹(1) = ∞, and 1f0 − 1f-15 == 1f0, so a caller's `1 - clamp_lo`
bound cannot keep a Float32 draw of exactly one out. Elementary ops only, so the
kernel broadcasts on GPU.
"""
@inline function myErlangStdQuantile(p::T, m::Int, lgm) where {T<:AbstractFloat}
    p = clamp(p, floatmin(T), prevfloat(one(T)))
    x = if p < T(0.5)
        exp((log(p) + T(lgm) + log(T(m))) / m)       # (p·m!)^(1/m): F ≈ x^m/m!
    else
        s = -log1p(-p)
        s + (m - 1) * log(max(s, one(T)))            # exponential-tail start
    end
    x = clamp(x, floatmin(T), T(700))
    for _ in 1:8
        S = one(T); term = one(T)
        for n in 1:(m - 1)
            term *= x / n                             # ends at x^(m-1)/(m-1)!
            S += term
        end
        xn = if p >= T(0.5)
            # Newton on the log-survival G(x) = -x + log(S); dG/dx = -term/S
            x + ((-x + log(S)) - log1p(-p)) * S / term
        else
            # Newton on F(x) − p with F from the tail series (no cancellation):
            # F = f·(x/m)·Σ_{j≥0} cⱼ with c₀ = 1, cⱼ = cⱼ₋₁·x/(m+j), and the density
            # f = e^(−x)·term. On this branch (p < 1/2, seeded below the root) the
            # iterate stays below about 4.2 and the series needs at most about 30
            # terms; the loop exits once the (eventually geometric) tail drops below
            # working precision. p == 1/2 exactly (the first Halton draw) takes the
            # upper branch, whose seed it already uses.
            Σ = one(T); cj = one(T)
            for j in 1:64
                cj *= x / (m + j)
                Σ += cj
                cj < eps(T) * Σ && break
            end
            f = max(term * exp(-x), floatmin(T))
            x - (f * (x / m) * Σ - p) / f
        end
        # multiplicative safeguard keeps the iterate positive and tame
        x = xn <= zero(T) ? x / 2 : clamp(xn, x / 16, 16 * x)
    end
    return x
end

# Upper clamp for probability draws fed to an inverse CDF: 1 - clamp_lo when that is
# representable below one (Float64), else the largest float below one. In Float32
# 1f0 - 1f-15 == 1f0, so the literal `1 - clamp_lo` let a draw of exactly one through
# and the Erlang quantile overflowed (about 3e12 instead of a finite upper-tail value).
@inline _draw_clamp_hi(::Type{T}, clamp_lo) where {T<:AbstractFloat} =
    min(one(T) - T(clamp_lo), prevfloat(one(T)))

"""
    myErlangQuantile!(out; θ, r, m, lgm, clamp_lo)

Compute quantile of Erlang(m, θ) at probability r: Q(p; m, θ) = θ · q_m(p), with
`θ` N×1, `out` N×D and the draws `r` either 1×D (shared across observations,
multiRand=false) or N×D (observation-specific, multiRand=true). The standardized
part q_m depends on the draws only: for shared draws it is evaluated once per draw
(D kernel evaluations) and then scaled by θ, instead of once per (observation, draw)
cell; for N×D draws every cell needs its own evaluation anyway, so the fused per-cell
broadcast is kept and no N×D temporary is allocated. Draws are clamped to
[clamp_lo, _draw_clamp_hi(eltype(r), clamp_lo)].
"""
function myErlangQuantile!(out; θ, r, m::Int, lgm, clamp_lo)
    Tr = eltype(r)
    lo = Tr(clamp_lo); hi = _draw_clamp_hi(Tr, clamp_lo); lg = Tr(lgm)
    if size(r, 1) == 1
        q = myErlangStdQuantile.(clamp.(r, lo, hi), m, lg)
        @. out = θ * q
    else
        @. out = θ * myErlangStdQuantile(clamp(r, lo, hi), m, lg)
    end
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
    return @. exp(log_c - ((ν + 1) / 2) * log1p((z / σ_v)^2 / ν))
end

"""
    myStudentTPDF!(out; z, σ_v=nothing, ν=nothing, log_const=nothing, inv_σ_v=nothing, half_nu_plus_one=nothing)

In-place version of `myStudentTPDF`.
When `log_const`, `inv_σ_v`, and `half_nu_plus_one` are provided, uses fast path; at ν = ∞
(`half_nu_plus_one` not finite) the fast path gives the normal density, the limit of the
Student-t density.

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
        if isfinite(half_nu_plus_one)
            @. out = exp(log_const - half_nu_plus_one * log1p((z * inv_σ_v)^2 / (2 * half_nu_plus_one - 1)))
        else
            # ν = ∞: the normal density, the limit of the Student-t density
            @. out = exp(log_const - 0.5 * (z * inv_σ_v)^2)
        end
    else
        log_c = loggamma((ν + 1) / 2) - loggamma(ν / 2) - 0.5 * log(ν * π) - log(σ_v)
        @. out = exp(log_c - ((ν + 1) / 2) * log1p((z / σ_v)^2 / ν))
    end
    return out
end


"""
    myLaplacePDF(; z, b)

Compute PDF of Laplace(0, b) at points z, with |x| evaluated as sqrt(x² + 1e-16), as in MCI.
f(x; b) = exp(-sqrt(x² + 1e-16)/b) / (2b)
Each log density differs from the exact Laplace log density by at most 1e-8/b.

# Arguments
- `z`: Points at which to evaluate PDF
- `b`: Scale parameter (positive)
"""
function myLaplacePDF(; z, b)
    return @. exp(-sqrt(z^2 + 1e-16) / b) / (2 * b)
end

"""
    myLaplacePDF!(out; z, b=nothing, inv_b=nothing, inv_2b=nothing)

In-place version of `myLaplacePDF`, with the same smoothed |z|, sqrt(z² + 1e-16): each log density
differs from the exact Laplace log density by at most 1e-8/b.
When `inv_b` and `inv_2b` are provided, uses fast path.

Fast path parameters:
- `inv_b`: Precomputed 1/b
- `inv_2b`: Precomputed 1/(2b)
"""
function myLaplacePDF!(out; z, b=nothing, inv_b=nothing, inv_2b=nothing)
    if inv_b !== nothing && inv_2b !== nothing
       @. out = inv_2b * exp(-sqrt(z^2 + 1e-16) * inv_b)
    else
       @. out = exp(-sqrt(z^2 + 1e-16) / b) / (2 * b)
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

The N × D per-observation Halton draws of versions before 2.0.0: one base-2 Halton sequence shortened to
2^m − 1 points, the largest such length not above N × D or `distinct_Halton_length`; row i takes the D points
that follow row i − 1, and the sequence restarts from its first point when it runs out. Another prime `base`
gives the Halton sequence in that base, built the same way. The estimation no longer uses this function; it is
kept so that earlier scripts run. A fit reproduces these draws with the deprecated `distinct_Halton_length`.
Throws `ArgumentError` when D exceeds that length.

# Arguments
- `N::Int`: Number of observations
- `D::Int`: Number of draws per observation
- `base::Int=2`: Base for Halton sequence (default: 2)
- `T::Type{<:AbstractFloat}=Float64`: Element type
- `distinct_Halton_length::Int=2^15-1`: Maximum Halton sequence length (controls the cap on sequence generation)

# Returns
- `Matrix{T}` of size (N, D) where each row contains different consecutive Halton values
"""
function make_halton_wrap(N::Int, D::Int; base::Int=2, T::Type{<:AbstractFloat}=Float64, distinct_Halton_length::Int=2^15-1)
    N > 0 || throw(ArgumentError("N must be positive, got $N"))
    D > 0 || throw(ArgumentError("D (draws) must be positive, got $D"))
    D <= distinct_Halton_length ||
        throw(ArgumentError("D = $D exceeds distinct_Halton_length = $distinct_Halton_length"))
    L = sf_halton_period(N, D; recycling = distinct_Halton_length, legacy = true)
    base == 2 && return sf_halton_draws(N, D; T = T, period = L)
    h = T.(collect(Halton(base, length = L)))      # other prime bases, as in versions before 2.0.0
    return [h[mod1((i - 1) * D + j, L)] for i in 1:N, j in 1:D]
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
make_ineff_constants(::Erlang_MSLE{M}, T) where {M} = (
    clamp_lo = T(1e-15),      # Clamp draws away from 0 and 1
    theta_floor = T(1e-12),
    theta_ceil = T(1e12),
    lgm = T(loggamma(M)),     # log((M-1)!) for the standardized quantile kernel
)

make_copula_constants(::NoCopula_MSLE, T) = NamedTuple()
make_copula_constants(::GaussianCopula_MSLE, T) = (
    copula_clamp_lo = T(1e-15),                  # Clamp F_v and F_u away from 0
    copula_clamp_hi = min(one(T) - T(1e-15), prevfloat(one(T))),   # Clamp F_v and F_u away from 1 (Float32-safe: 1f0 - 1f-15 == 1f0)
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
    # The clamps act on F_u only; the noise side is computed in logarithms, with no clamp.
    copula_clamp_lo = T(1e-16),                  # Clamp F_u away from 0 (-log(u) → ∞)
    copula_clamp_hi = min(one(T) - T(1e-16), prevfloat(one(T))),   # Clamp F_u away from 1; Float32: 1f0 - 1f-16 == 1f0 would give log(-log 1) = -Inf
    gumbel_rho_max = T(50.0),                    # Ceiling ρ: ρ = clamp(exp(θ_rho) + 1, 1, gumbel_rho_max)
)
make_copula_constants(::Clayton90Copula_MSLE, T) = (
    copula_clamp_lo = T(1e-6),                   # Same constants as standard Clayton
    copula_clamp_hi = one(T) - T(1e-6),
    clayton_rho_floor = T(1e-6),
    clayton_rho_max = T(50.0),
)

# The clamps a copula log-density applies to a probability held in the array A: make_copula_constants(copula, R),
# with R the wider of the floating-point type A's elements are computed in (for a ForwardDiff Dual, nested or not,
# its value type: ForwardDiff defines eps of a Dual type as eps of its value type) and the type of the
# specification's constants `c`. A fit's coefficients are Float64 (Duals of Float64 under AD), so with Float32 data
# F_v, and MCI's F_u, are computed in Float64. Clamped with the Float32 constants, the Gaussian copula's normal
# scores would be capped at ±4.62 instead of ±7.78 (the Float32 log-likelihood up to 1.3 nats off on 300
# observations at ρ = 0.46). Where the arithmetic is not wider than `c` (Float64 data at any coefficients; Float32
# data at Float32 coefficients) these are the values `c` holds, of c's type.
_copula_clamps(copula, A, c) =
    make_copula_constants(copula, promote_type(typeof(eps(eltype(A))), typeof(c.copula_clamp_lo)))

# The clamps of the Gaussian copula's normal scores of MSLE's draws. The draws are stored in the
# specification's element type and their scores are evaluated in that type (for Float32 data a Float32 erfinv,
# so no double-precision work is added on the GPU). They are clamped with `cv` (`_copula_clamps`),
# rounded to the draws' type Td and kept inside (-1, 1) in it. For Float32 draws in a fit (Float64 arithmetic) the
# inverse-erf bounds are therefore ±prevfloat(1f0) instead of ±(1 - 32 eps(Float32)): the base-2 Halton draws,
# exact in Float32 and at least 2^-24 from 0 and 1 for sequences of up to 2^24 - 1 points, are not capped at
# ±4.62. When Td is the type of `cv` these are the values `cv` holds.
function _gaussian_draw_clamps(cv, draws)
    Td = typeof(eps(eltype(draws)))
    h = prevfloat(one(Td))
    return (copula_clamp_lo = Td(cv.copula_clamp_lo), copula_clamp_hi = min(Td(cv.copula_clamp_hi), h),
            lo_erfinv = max(Td(cv.lo_erfinv), -h), hi_erfinv = min(Td(cv.hi_erfinv), h))
end

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

    # Stable log normalizing constant: the naive loggamma difference cancels
    # catastrophically for large ν (O(1) log-density error at ν ≈ 1e14, which an
    # optimizer climbs — see MCI_Backend._t_log_const for the derivation). At ν = ∞ the
    # limit is written without ν, so that its derivatives stay finite.
    log_const = if !isfinite(nu)
        -P(0.5) * log(P(2) * P(π)) - log(sigma_v)
    elseif nu > P(1e6)
        -P(0.5) * log(P(2) * P(π)) - P(1) / (P(4) * nu) - log(sigma_v)
    else
        loggamma((nu + 1) / P(2)) - loggamma(nu / P(2)) -
            P(0.5) * log(nu * P(π)) - log(sigma_v)
    end

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

    # σᵤ² = exp(Z * gamma).  For the exponential the squared scale IS Var(u), so this
    # is both; the quantile below forms the scale as its square root.
    k_sigma = length(ineff.sigma_sq)
    sigma_sq = exp.(sum(P(p[ineff.sigma_sq[j]]) .* (@view Z[:, j]) for j in 1:k_sigma))
    sigma_sq = clamp.(sigma_sq, c.σ_floor, c.σ_ceil)

    return (sigma_sq=sigma_sq,)
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

function get_ineff_vals(::Erlang_MSLE, p, idx, Z, c)
    P = eltype(p)
    ineff = idx.ineff

    # θ = exp(Z * gamma_theta); the shape is pinned at the Erlang order
    theta_len = length(ineff.theta)
    theta_val = exp.(sum(P(p[ineff.theta[j]]) .* (@view Z[:, j]) for j in 1:theta_len))
    theta_val = clamp.(theta_val, c.theta_floor, c.theta_ceil)

    return (theta=theta_val,)
end

# ---------------------------------------------------------------------------
# The boundary verdict. The get_*_vals functions return CLAMPED values, and `clamp` returns
# the bound itself when it binds, so the verdict tests the values the likelihood uses against
# the bounds (within SF_BOUND_RTOL, src/sf_redflag.jl). Beyond a bound the likelihood is flat
# in that parameter; for a homoscedastic parameter its Hessian row is then exactly zero. The
# bounds come from the model's own constants. Not listed, because the likelihood applies no
# bound to them: the Student-t ν, the μ of the truncated normal and the lognormal, the scaling
# function and the Gaussian copula's ρ (a smooth map); the copula floors are not tested.

# One bounded quantity: its name and label, the coefficient block it is formed from, its bounds,
# whether it has one value for all observations, its form (`:exp`: exp(θ) + offset; `:exp_half`:
# exp(θ/2)), whether the likelihood clamps it, and where the value functions return its clamped
# value (`vals`: :noise, :ineff or :copula, and its `field`).
_bound_item(parameter, label, coefficient, keyword, block, lo, hi, scalar, form, vals, field; offset = 0.0) =
    (parameter = parameter, label = label, coefficient = coefficient, keyword = keyword,
     block = Vector{Int}(vcat(block)), lo = Float64(lo), hi = Float64(hi), scalar = scalar, form = form,
     offset = Float64(offset), movable = true, vals = vals, field = field)

# An inefficiency quantity has one value for all observations when its block has one coefficient
# and it is not heteroscedastic (`hsym`, its symbol in `hetero`).
_ineff_bound_item(parameter, label, coefficient, keyword, block, lo, hi, form, field, hsym, hetero) =
    _bound_item(parameter, label, coefficient, keyword, block, lo, hi, length(block) == 1 && !(hsym in hetero),
                form, :ineff, field)

_noise_bound_items(::Union{NormalNoise_MSLE, StudentTNoise_MSLE}, idx, c) =
    (_bound_item(:sigma_v, "σᵥ", "ln_σᵥ²", :ln_sigma_v_sq, idx.noise.ln_sigma_v_sq, c.σ_floor, c.σ_ceil, true,
                 :exp_half, :noise, :sigma_v),)
_noise_bound_items(::LaplaceNoise_MSLE, idx, c) =
    (_bound_item(:b, "b", "ln_b", :ln_b, idx.noise.ln_b, c.σ_floor, c.σ_ceil, true, :exp, :noise, :b),)

_ineff_bound_items(::TruncatedNormal_MSLE, idx, c, het) =
    (_ineff_bound_item(:sigma_u, "σᵤ", "ln_σᵤ²", :ln_sigma_u_sq, idx.ineff.sigma_u, c.σ_floor, c.σ_ceil,
                       :exp_half, :sigma_u, :sigma_u_sq, het),)
_ineff_bound_items(::Exponential_MSLE, idx, c, het) =
    (_ineff_bound_item(:sigma_u_sq, "σᵤ²", "ln_σᵤ²", :ln_sigma_u_sq, idx.ineff.sigma_sq, c.σ_floor, c.σ_ceil,
                       :exp, :sigma_sq, :sigma_u_sq, het),)
_ineff_bound_items(::HalfNormal_MSLE, idx, c, het) =
    (_ineff_bound_item(:sigma_u, "σᵤ", "ln_σᵤ²", :ln_sigma_u_sq, idx.ineff.sigma_sq, c.σ_floor, c.σ_ceil,
                       :exp_half, :sigma, :sigma_u_sq, het),)
_ineff_bound_items(::Union{Lognormal_MSLE, Rayleigh_MSLE}, idx, c, het) =
    (_ineff_bound_item(:sigma, "σ", "ln_σᵤ²", :ln_sigma_u_sq, idx.ineff.sigma_sq, c.σ_floor, c.σ_ceil,
                       :exp_half, :sigma, :sigma_u_sq, het),)
_ineff_bound_items(::Weibull_MSLE, idx, c, het) =
    (_ineff_bound_item(:lambda, "λ", "ln_λ", :ln_lambda, idx.ineff.lambda, c.σ_floor, c.σ_ceil, :exp, :lambda,
                       :lambda, het),
     _ineff_bound_item(:k, "k", "ln_k", :ln_k, idx.ineff.k, c.k_floor, c.k_ceil, :exp, :k, :k, het))
_ineff_bound_items(::Lomax_MSLE, idx, c, het) =
    (_ineff_bound_item(:lambda, "λ", "ln_λ", :ln_lambda, idx.ineff.ln_lambda, c.lambda_floor, c.lambda_ceil, :exp,
                       :lambda, :lambda, het),
     _ineff_bound_item(:alpha, "α", "ln_α", :ln_alpha, idx.ineff.alpha, c.α_floor, c.α_ceil, :exp, :alpha, :alpha,
                       het))
_ineff_bound_items(::Erlang_MSLE, idx, c, het) =
    (_ineff_bound_item(:theta, "θ", "ln_θ", :ln_theta, idx.ineff.theta, c.theta_floor, c.theta_ceil, :exp, :theta,
                       :theta, het),)

_copula_bound_items(::Union{NoCopula_MSLE, GaussianCopula_MSLE}, idx, c) = ()
_copula_bound_items(::Union{ClaytonCopula_MSLE, Clayton90Copula_MSLE}, idx, c) =
    (_bound_item(:rho, "ρ", "theta_rho", :theta_rho, idx.copula.theta_rho, -Inf, c.clayton_rho_max, true, :exp,
                 :copula, :rho; offset = c.clayton_rho_floor),)
_copula_bound_items(::GumbelCopula_MSLE, idx, c) =
    (_bound_item(:rho, "ρ", "theta_rho", :theta_rho, idx.copula.theta_rho, -Inf, c.gumbel_rho_max, true, :exp,
                 :copula, :rho; offset = 1.0),)

"""
    _bound_list(model::sfmodel_MSLE_spec) -> Vector

The quantities the likelihood of `model` clamps to numerical bounds, in the order noise,
inefficiency, copula: one NamedTuple per quantity with `parameter`, `label`, `coefficient`,
`keyword` (the `sfmodel_init` keyword of its coefficients), `block` (the coefficient indices),
`lo` and `hi` (from the model's constants; `lo = -Inf` where no lower bound is tested), `scalar`,
`form`, `offset`, `movable`, and `vals` and `field` (where the value functions return its
clamped value).
"""
function _bound_list(model::sfmodel_MSLE_spec)
    m, c, idx = model.model, model.constants, model.idx
    return [_noise_bound_items(m.noise, idx, c)..., _ineff_bound_items(m.ineff, idx, c, model.hetero)...,
            _copula_bound_items(m.copula, idx, c)...]
end

"""
    _bound_verdict(model::sfmodel_MSLE_spec, p, Z) -> Vector{SFBoundEntry}

The boundary verdict at the coefficient vector `p`: the entries of every quantity of
`_bound_list(model)` at or beyond its bound, from the clamped values of the likelihood's own value
functions; empty when there is none. `Z` is the host copy of the inefficiency design (the fit
passes `spec.zvar`; `model.zvar` is a device array in GPU fits).
"""
function _bound_verdict(model::sfmodel_MSLE_spec, p::AbstractVector, Z::AbstractMatrix)
    m, c, idx = model.model, model.constants, model.idx
    vals = (noise = get_noise_vals(m.noise, p, idx, c), ineff = get_ineff_vals(m.ineff, p, idx, Z, c),
            copula = get_copula_vals(m.copula, p, idx, c))
    bc = SFBoundEntry[]
    for it in _bound_list(model)
        sf_bound_entries!(bc, getproperty(getproperty(vals, it.vals), it.field), it.lo, it.hi, model.N;
                          parameter = it.parameter, label = it.label, coefficient = it.coefficient,
                          scalar = it.scalar)
    end
    return bc
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
    sigma_sq_N1 = reshape(ineff_vals.sigma_sq, N, 1)

    myExponentialQuantile!(buffer; σᵤ²=sigma_sq_N1, r=draws_1D, clamp_lo=c.exp_clamp)
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

function get_u_quantile!(::Erlang_MSLE{M}, buffer, ineff_vals, draws_1D, c, N) where {M}
    theta_N1 = reshape(ineff_vals.theta, N, 1)

    myErlangQuantile!(buffer; θ=theta_N1, r=draws_1D, m=M, lgm=c.lgm, clamp_lo=c.clamp_lo)
end

# --- Chunked quantile computation (for GPU memory management when chunks > 1) ---
# Note: draws_1D can be either 1 x D (broadcast mode) or N x D (multiRand mode).
#       When N x D, we slice the appropriate rows for this chunk.

# Helper to slice draws for chunked computation (handles both 1×D and N×D cases)
@inline function _slice_draws_chunk(draws, row_start, row_end)
    return size(draws, 1) > 1 ? (@view draws[row_start:row_end, :]) : draws
end

# A host copy of the per-observation values of one chunk (the array itself when it is on the host already).
_host_vec(x::Array) = x
_host_vec(x) = Array(x)

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
    sigma_sq_chunk = reshape((@view ineff_vals.sigma_sq[row_start:row_end]), chunk_N, 1)
    draws_chunk = _slice_draws_chunk(draws_1D, row_start, row_end)

    myExponentialQuantile!(buf; σᵤ²=sigma_sq_chunk, r=draws_chunk, clamp_lo=c.exp_clamp)
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

function get_u_quantile_chunk!(::Erlang_MSLE{M}, buf, ineff_vals, draws_1D, c,
                                    row_start, row_end, chunk_N) where {M}
    theta_chunk = reshape((@view ineff_vals.theta[row_start:row_end]), chunk_N, 1)
    draws_chunk = _slice_draws_chunk(draws_1D, row_start, row_end)

    myErlangQuantile!(buf; θ=theta_chunk, r=draws_chunk, m=M, lgm=c.lgm, clamp_lo=c.clamp_lo)
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
    # log f(z) = log_const - (ν+1)/2 * log1p((z/σ)²/ν); at ν = ∞ its limit, log_const - (z/σ)²/2
    inv_σ = noise_vals.inv_sigma_v
    half_nu_plus_one = noise_vals.half_nu_plus_one
    if isfinite(half_nu_plus_one)
        nu = 2 * half_nu_plus_one - 1
        @. out = noise_vals.log_const - half_nu_plus_one * log1p((z * inv_σ)^2 / nu)
    else
        @. out = noise_vals.log_const - 0.5 * (z * inv_σ)^2
    end
end

# log f(z) = log(1/(2b)) − |z|/b, with |z| evaluated as sqrt(z² + 1e-16), as in MCI, so that both methods implement
# the same objective; each log density differs from the exact Laplace log density by at most 1e-8/b.
function log_noise_pdf!(::LaplaceNoise_MSLE, out, z, noise_vals, c)
    inv_b = noise_vals.inv_b
    inv_2b = noise_vals.inv_2b
    @. out = log(max(inv_2b, c.clamp_lo)) - sqrt(z^2 + 1e-16) * inv_b
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

# --- log(-log F_v) for the Gumbel copula (in logarithms, with no clamp) ---

"""
    noise_log_neglogcdf!(::NormalNoise_MSLE, out, z, noise_vals, c)

Normal noise, in place: out .= log(-log F_v(v)) at v = z, by `sf_log_neglogcdf_normal(z/σ_v)`,
which takes it from the complementary CDF where F_v is near 1; no clamp. The Gumbel copula's
noise side.
"""
function noise_log_neglogcdf!(::NormalNoise_MSLE, out, z, noise_vals, c)
    @. out = sf_log_neglogcdf_normal(z * noise_vals.inv_sigma_v)
    return nothing
end

"""
    noise_log_neglogcdf!(::LaplaceNoise_MSLE, out, z, noise_vals, c)

Laplace noise, in place: out .= log(-log F_v(v)) at v = z, by `sf_log_neglogcdf_laplace(z/b)`;
no clamp. The Gumbel copula's noise side.
"""
function noise_log_neglogcdf!(::LaplaceNoise_MSLE, out, z, noise_vals, c)
    @. out = sf_log_neglogcdf_laplace(z * noise_vals.inv_b)
    return nothing
end

"""StudentT noise is not supported for copula models (unreachable due to _build_model validation)."""
function noise_log_neglogcdf!(::StudentTNoise_MSLE, out, z, noise_vals, c)
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
    # F_v's normal score is clamped with `_copula_clamps` (after make_copula_constants): the
    # constants of the type F_v is computed in, or c's if those are wider. The draws' scores keep the draws'
    # own type and are clamped with the same clamps rounded to it (`_gaussian_draw_clamps`). For a Float64
    # specification, and for Float32 arithmetic, all of these are c's.
    cv = _copula_clamps(GaussianCopula_MSLE(), Fv_buf, c)
    cd = _gaussian_draw_clamps(cv, draws)
    @. adj = -0.5 * log(omr2) - (
        rho_sq * (
            (c.sqrt2 * erfinv(clamp(2 * clamp(Fv_buf, cv.copula_clamp_lo, cv.copula_clamp_hi) - 1, cv.lo_erfinv, cv.hi_erfinv)))^2 +
            (c.sqrt2 * erfinv(clamp(2 * clamp(draws,  cd.copula_clamp_lo, cd.copula_clamp_hi) - 1, cd.lo_erfinv, cd.hi_erfinv)))^2
        ) - 2 * rho *
            (c.sqrt2 * erfinv(clamp(2 * clamp(Fv_buf, cv.copula_clamp_lo, cv.copula_clamp_hi) - 1, cv.lo_erfinv, cv.hi_erfinv))) *
            (c.sqrt2 * erfinv(clamp(2 * clamp(draws,  cd.copula_clamp_lo, cd.copula_clamp_hi) - 1, cd.lo_erfinv, cd.hi_erfinv)))
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

Compute the Gumbel copula log-density adjustment in-place into `adj`, in logarithms.
Let w₁ = -log F_v(v), w₂ = -log F_u(u), S = w₁^ρ + w₂^ρ. Then
log c = -S^{1/ρ} + (ρ-1)(log w₁ + log w₂) + (-2 + 2/ρ) log S + log1p((ρ-1) S^{-1/ρ}) + w₁ + w₂,
computed from L₁ = log w₁, L₂ = log w₂ and log S = logaddexp(ρL₁, ρL₂); the log1p form keeps the derivative
finite at ρ = 1.
The noise side has no floor: `noise_log_neglogcdf!` writes L₁ into `Fv_buf`, from the complementary CDF where
F_v is near 1, so L₁ is finite for every finite residual. The draws (F_u, by the quantile inversion) keep the
clamps of the specification's constants: [1e-16, 1 - 1.1e-16] with Float64 data, [1f-16, prevfloat(1f0)] with
Float32 data. Halton draws do not cross them; a user-supplied draw of exactly 0 is raised to 1e-16.
Two passes in the two buffers: log S into `adj`, then the log density.
"""
function copula_log_adjustment!(::GumbelCopula_MSLE, adj, Fv_buf, z_buffer, draws, noise, noise_vals, copula_vals, c)
    noise_log_neglogcdf!(noise, Fv_buf, z_buffer, noise_vals, c)          # Fv_buf ← L1 = log(-log F_v), no clamp
    rho = copula_vals.rho; rho_m1 = rho - 1; inv_rho = 1 / rho; e2 = -2 + 2 * inv_rho
    lou = c.copula_clamp_lo; hiu = c.copula_clamp_hi                     # the draws keep the specification's clamps
    @. adj = sf_logaddexp(rho * Fv_buf, rho * log(-log(clamp(draws, lou, hiu))))          # log S
    # adj holds log S; the broadcast reads each element before overwriting it (safe with @.)
    @. adj = -exp(adj * inv_rho) + rho_m1 * (Fv_buf + log(-log(clamp(draws, lou, hiu)))) +
             e2 * adj + log1p(rho_m1 * exp(-adj * inv_rho)) + exp(Fv_buf) - log(clamp(draws, lou, hiu))
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

# `logsumexp_rows(A, clamp_lo)` with its temporary `exp.(A .- max)` written into `tmp` instead of a
# new array. The chunked likelihood (both `qmc_nll` methods) allocates `tmp` once per evaluation and
# passes it for every chunk of that size, instead of a new temporary in every chunk (about N x D
# elements per evaluation in all; device memory on the GPU). The expression is that of
# `logsumexp_rows`, and `tmp` is a fresh array of the type, element type and size
# `logsumexp_rows` allocates (`similar(ε, P, rows, D)`, never a view), so the row maximum, the
# broadcast and the row sum run the same CPU or GPU kernels on the same input and give the same
# result, bit for bit. Every element of `tmp` is written before it is read, so what an earlier
# chunk left in it does not matter.
function _logsumexp_rows_tmp!(tmp::AbstractMatrix, A::AbstractMatrix, clamp_lo=1e-300)
    size(tmp) == size(A) ||
        throw(DimensionMismatch("_logsumexp_rows_tmp!: temporary of size $(size(tmp)) for a matrix of size $(size(A))"))
    max_vals = _maximum_msle(A; dims=2)
    tmp .= exp.(A .- max_vals)
    sum_exp  = _sum_msle(tmp; dims=2)
    # Use oftype to convert clamp_lo to match element type (works with ForwardDiff.Dual)
    fmin = oftype(zero(eltype(A)), clamp_lo)
    @. sum_exp = max(sum_exp, fmin)
    return vec(max_vals .+ log.(sum_exp))
end

# ============================================================================
# Section 8: CPU Likelihood Function
# ============================================================================

"""
    qmc_nll(Y, X, Z, p, draws; noise, ineff, copula=:None, hetero=Symbol[], chunks=4, type=:prod, constants=nothing)

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
  - `:Exponential` - Exponential u ~ Exp(σᵤ), σᵤ² = Var(u)
  - `:HalfNormal` - Half Normal u ~ |N(0, σ²)|
  - `:Weibull` - Weibull u ~ Weibull(λ, k)
  - `:Lognormal` - Lognormal u ~ LogNormal(μ, σ)
  - `:Lomax` - Lomax u ~ Lomax(α, λ)
  - `:Rayleigh` - Rayleigh u ~ Rayleigh(σ)

- `hetero`: Vector of symbols controlling heterogeneity:
  - `:TruncatedNormal` → `[:mu]`, `[:sigma_u_sq]`, or both
  - `:Exponential` → `[:sigma_u_sq]`
  - `:HalfNormal` → `[:sigma_u_sq]`
  - `:Weibull` → `[:lambda]`, `[:k]`, or both
  - `:Lognormal` → `[:mu]`, `[:sigma_u_sq]`, or both
  - `:Lomax` → `[:lambda]`, `[:alpha]`, or both
  - `:Rayleigh` → `[:sigma_u_sq]`

- `copula::Symbol=:None`: Copula family coupling the noise and inefficiency draws.
  One of `:None`, `:Gaussian`, `:Clayton`, `:Clayton90`, `:Gumbel`.
- `chunks::Int=4`: Number of observation chunks for memory management.
- `type::Symbol=:prod`: Frontier type. `:prod`/`:production` gives ε = v − u (the
  composed error is reconstructed as `z = ε + u`); `:cost` gives ε = v + u
  (`z = ε − u`). **Must match the `type` used to fit the model.** Passing the wrong
  `type` silently inverts the efficiency ranking — the indices stay in a plausible
  range, so the error is invisible.
- `constants`: Precomputed constants from `make_constants` (default: computed fresh)

# Returns
- Scalar negative log-likelihood value

# Examples
```julia
# Create draws as 1×D matrix (required format)
halton = reshape(make_halton_p(1024), 1, 1024)

# Normal noise + Truncated Normal inefficiency
nll = qmc_nll(Y, X, Z, p, halton; noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])

# Laplace noise + Weibull inefficiency
nll = qmc_nll(Y, X, Z, p, halton; noise=:Laplace, ineff=:Weibull, hetero=[:lambda, :k])

# StudentT noise + Lognormal inefficiency
nll = qmc_nll(Y, X, Z, p, halton; noise=:StudentT, ineff=:Lognormal, hetero=[:mu, :sigma_u_sq])

# Cost frontier: pass type=:cost so the sign matches the fitted model
nll = qmc_nll(Y, X, Z, p, halton; noise=:Normal, ineff=:HalfNormal, type=:cost)
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
    _validate_user_draws(draws)
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
        # The log-sum-exp temporary, allocated once per evaluation (not anew for every chunk
        # inside logsumexp_rows); see _logsumexp_rows_tmp!.
        lse_tmp = similar(ε, P, chunk_size, D)
        chunk_nlls = zeros(P, chunks)  # zero-init: empty trailing chunks skip their slot (`continue`); sum must not read undef

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
                draws_chunk = _slice_draws_chunk(draws_1D, row_start, row_end)
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
            # A shorter last chunk gets a temporary of its own exact size
            lse_buf = chunk_N == chunk_size ? lse_tmp : similar(ε, P, chunk_N, D)
            log_likes = _logsumexp_rows_tmp!(lse_buf, buf) .- log(P(D))
            chunk_nlls[chunk_idx] = _sum_scalar_msle(log_likes)
        end

        return -sum(chunk_nlls)
    end
end

"""
    qmc_nll(spec::sfmodel_MSLE_spec, p::AbstractVector; chunks::Int=4, per_obs::Bool=false)

Compute negative log-likelihood using a model specification.
This is the simplified interface that extracts all configuration from the spec.

# Arguments
- `spec::sfmodel_MSLE_spec`: Model specification containing data and model config
- `p::AbstractVector`: Parameter vector

# Keyword Arguments
- `chunks::Int=4`: Number of chunks for memory management (default: 4)
- `per_obs::Bool=false`: return the per-observation log-likelihoods (not negated), a host vector of
  length N, instead of the negative log-likelihood; computed in the chunk loop for every value of
  `chunks`

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
                 chunks::Int=4, per_obs::Bool=false) where {T<:AbstractFloat, P<:Real}

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

    if chunks == 1 && !per_obs
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
        # The log-sum-exp temporary, allocated once per evaluation (not anew for every chunk
        # inside logsumexp_rows); see _logsumexp_rows_tmp!.
        lse_tmp = similar(ε, P, chunk_size, D)
        chunk_nlls = zeros(P, chunks)  # zero-init: empty trailing chunks skip their slot (`continue`); sum must not read undef
        # With `per_obs`: the log-likelihood of each observation, copied to the host chunk by chunk.
        ll_obs = per_obs ? Vector{P}(undef, n_obs) : nothing

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
                draws_chunk = _slice_draws_chunk(draws_1D, row_start, row_end)
                copula_log_adjustment!(model.copula, copula_adj, Fv_buf, buf, draws_chunk,
                                       model.noise, noise_vals, copula_vals, c)
            end

            log_noise_pdf!(model.noise, buf, buf, noise_vals, c)

            if has_copula
                copula_adj = @view copula_adj_buf[1:chunk_N, :]
                @. buf = buf + copula_adj
            end

            # A shorter last chunk gets a temporary of its own exact size
            lse_buf = chunk_N == chunk_size ? lse_tmp : similar(ε, P, chunk_N, D)
            log_likes = _logsumexp_rows_tmp!(lse_buf, buf) .- log(P(D))
            chunk_nlls[chunk_idx] = _sum_scalar_msle(log_likes)
            per_obs && copyto!(ll_obs, row_start, _host_vec(log_likes), 1, chunk_N)
        end

        return per_obs ? ll_obs : -sum(chunk_nlls)
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
    jlms_bc_indices(Y, X, Z, p, draws; noise, ineff, copula=:None, hetero, chunks, type=:prod, constants)

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
- `copula::Symbol=:None`: Copula family coupling the noise and inefficiency draws.
  One of `:None`, `:Gaussian`, `:Clayton`, `:Clayton90`, `:Gumbel`.
- `chunks::Int`: Number of chunks for memory management (default: 1)
- `type::Symbol=:prod`: Frontier type — `:prod`/`:production` or `:cost`. **Must match
  the fitted model's `type`.** Indices are computed against `z = ε + sign·u`
  (`sign = +1` prod, `−1` cost); a mismatched `type` silently inverts the ranking.
- `constants`: Precomputed constants from `make_constants` (default: computed fresh)

# Returns
NamedTuple with fields:
- `jlms::Vector`: E(u|ε) for each observation (inefficiency index)
- `bc::Vector`: E(e^{-u}|ε) for each observation (efficiency index)
- `likelihood::Vector`: f_ε(ε) for each observation (density value)

# Examples
```julia
# Create draws as 1×D matrix (required format)
halton = reshape(make_halton_p(1024), 1, 1024)

# After estimation, compute efficiency indices
result = jlms_bc_indices(Y, X, Z, p_hat, halton;
    noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])

# Cost frontier: pass type=:cost (defaults to :prod otherwise)
result_cost = jlms_bc_indices(Y, X, Z, p_hat, halton;
    noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu], type=:cost)

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
    _validate_user_draws(draws)
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
                draws_chunk = _slice_draws_chunk(draws_1D, row_start, row_end)
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
- `chunks::Int=1`: Number of groups of observations processed at a time; memory is proportional to N × D / chunks. A fit passes its own `chunks`.

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
                draws_chunk = _slice_draws_chunk(draws_1D, row_start, row_end)
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
    _invert_hessian(H::AbstractMatrix{T}; at_clamp::Bool=false, message::Bool=true) where {T<:Real}

Internal helper for Hessian inversion with error handling. It applies the rule shared by
all four backends, `sf_classify_hessian` (src/sf_redflag.jl).

# Returns
NamedTuple: `(var_cov_matrix=Matrix, redflag=Int, hessian_flag=Int)`
- `redflag = 0`: Success
- `redflag = 1`: Hessian not invertible, or numerically singular (its 1-norm reciprocal
  condition number is below machine epsilon); the matrix is then all `NaN`. Also 1 when
  the Hessian passes those tests and the diagonal test below but is not positive definite
  (a saddle point or a ridge, not a strict local maximum); the matrix is then
  kept as computed. Also 1 when `at_clamp` (a parameter at or beyond its numerical bound:
  a variance, scale, shape or copula dependence parameter), whatever the Hessian says.
- `redflag = 2`: Some diagonal elements are non-positive
`hessian_flag` is the value before the `at_clamp` override.
"""
_invert_hessian(H::AbstractMatrix{<:Real}; at_clamp::Bool=false, message::Bool=true) =
    sf_classify_hessian(H; at_clamp=at_clamp, message=message)

# NaN-safe standard errors from a var-cov matrix. A non-positive-diagonal (redflag>0) vcov
# would make `sqrt` throw a DomainError, so those entries become NaN instead — a degenerate
# fit still returns its coefficients/loglik/jlms/bc, and `stderror(r)` stays NaN-consistent
# with `sqrt.(diag(vcov(r)))`. The test is `<= 0`, so an exactly zero diagonal (redflag 2)
# also gives NaN, not 0.
_stderr_from_vcov(V::AbstractMatrix) = map(d -> d <= 0 ? oftype(d, NaN) : sqrt(d), diag(V))

"""
    var_cov_mat(nll_func::Function, coef::AbstractVector{T}; message::Bool=true) where {T<:Real}

Compute variance-covariance matrix from negative log-likelihood function.

# Arguments
- `nll_func`: A closure of the negative log-likelihood function that takes only `p` (coefficient vector)
- `coef`: The optimized coefficient vector from `Optim.minimizer(result)`
- `message`: Whether to print error messages (default: true)

# Returns
NamedTuple: `(var_cov_matrix=Matrix, redflag=Int, hessian_flag=Int)`, from `_invert_hessian`
(`hessian_flag` equals `redflag` here)
- `redflag = 0`: Success
- `redflag = 1`: Hessian not invertible or numerically singular (all-`NaN` matrix), or not
  positive definite (the matrix kept as computed)
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
NamedTuple: `(var_cov_matrix=Matrix, redflag=Int, hessian_flag=Int)`, with `redflag` as for
the method above

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
Vector of tuples: `[(display_name, param_index, offset), ...]`, where the auxiliary
table reports the original-scale value `exp(coef[param_index]) + offset` (offset is 0
for the log-variance/scale/shape params and 2 for StudentT's ν = exp(ln(ν−2)) + 2).
"""
function _detect_log_params(model::MSLEModel, idx, L::Int, hetero::Vector{Symbol})
    # Entries are (display_name, param_index, offset): the auxiliary table reports
    # the original-scale value `exp(coef[param_index]) + offset`. `offset` is 0 for
    # every log-variance/scale/shape parameter and 2 for StudentT's degrees of
    # freedom ν = exp(ln(ν−2)) + 2.
    entries = Tuple{String, Int, Float64}[]

    # Noise parameters (always scalar). σᵥ² is the noise variance under Normal noise
    # and the squared scale of t_ν(0, σᵥ) under StudentT noise.
    if hasfield(typeof(idx.noise), :ln_sigma_v_sq)
        if model.noise isa StudentTNoise_MSLE
            push!(entries, ("σᵥ² (squared scale)", idx.noise.ln_sigma_v_sq, 0.0))
        else
            push!(entries, ("σᵥ²", idx.noise.ln_sigma_v_sq, 0.0))
        end
    elseif hasfield(typeof(idx.noise), :ln_b)
        push!(entries, ("b", idx.noise.ln_b, 0.0))
    end
    # StudentT degrees of freedom: ν = exp(ln_nu_minus_2) + 2
    if hasfield(typeof(idx.noise), :ln_nu_minus_2)
        push!(entries, ("ν", idx.noise.ln_nu_minus_2, 2.0))
    end

    # Inefficiency parameters (only if scalar - not heteroscedastic). The elseif
    # chain reports the scale/first parameter; the shape/second parameter (Weibull
    # k, Lomax α) is pushed independently below.
    ineff = idx.ineff
    if hasfield(typeof(ineff), :sigma_u) && length(ineff.sigma_u) == 1
        push!(entries, ("σᵤ²", first(ineff.sigma_u), 0.0))
    elseif hasfield(typeof(ineff), :sigma_sq) && length(ineff.sigma_sq) == 1
        # every inefficiency distribution that owns this parameter prints it as σᵤ²
        push!(entries, ("σᵤ²", first(ineff.sigma_sq), 0.0))
    elseif hasfield(typeof(ineff), :lambda) && length(ineff.lambda) == 1
        push!(entries, ("λ", first(ineff.lambda), 0.0))
    elseif hasfield(typeof(ineff), :ln_lambda) && length(ineff.ln_lambda) == 1
        push!(entries, ("λ", first(ineff.ln_lambda), 0.0))   # Lomax: field is :ln_lambda, not :lambda
    end

    # Second (shape) inefficiency parameter — Weibull k, Lomax α.
    if hasfield(typeof(ineff), :k) && length(ineff.k) == 1
        push!(entries, ("k", first(ineff.k), 0.0))
    end
    if hasfield(typeof(ineff), :alpha) && length(ineff.alpha) == 1
        push!(entries, ("α", first(ineff.alpha), 0.0))
    end
    if hasfield(typeof(ineff), :theta) && length(ineff.theta) == 1
        push!(entries, ("θ", first(ineff.theta), 0.0))   # Erlang scale
    end

    return entries
end

"""
    _detect_copula_params(model::MSLEModel, idx, coef, stddev)

Auto-detect copula parameters for auxiliary table.
Returns a NamedTuple with copula-specific statistics.
"""
function _detect_copula_params(model::MSLEModel, idx, coef, stddev)
    cc = make_copula_constants(model.copula, Float64)
    if model.copula isa NoCopula_MSLE
        return NamedTuple()
    elseif model.copula isa GaussianCopula_MSLE
        rho_max = cc.rho_max
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
        # Match get_copula_vals clamp
        rho = clamp(exp(theta_rho) + cc.clayton_rho_floor, cc.clayton_rho_floor, cc.clayton_rho_max)
        se_rho = exp(theta_rho) * se_theta  # Delta method: d/dθ (exp(θ)+c) = exp(θ)
        kendalls_tau = rho / (2 + rho)
        tail_dep = 2^(-1/rho)              # Lower tail dependence
        return (rho=rho, se_rho=se_rho, kendalls_tau=kendalls_tau,
                tail_dep=tail_dep, theta_rho=theta_rho, se_theta=se_theta)
    elseif model.copula isa GumbelCopula_MSLE
        theta_rho = coef[idx.copula.theta_rho]
        se_theta = stddev[idx.copula.theta_rho]
        rho = clamp(exp(theta_rho) + 1, 1.0, cc.gumbel_rho_max)  # Match get_copula_vals clamp
        se_rho = exp(theta_rho) * se_theta  # Delta method: d/dθ (exp(θ)+1) = exp(θ)
        kendalls_tau = 1 - 1/rho
        tail_dep = 2 - 2^(1/rho)           # Upper tail dependence
        return (rho=rho, se_rho=se_rho, kendalls_tau=kendalls_tau,
                tail_dep=tail_dep, theta_rho=theta_rho, se_theta=se_theta)
    elseif model.copula isa Clayton90Copula_MSLE
        theta_rho = coef[idx.copula.theta_rho]
        se_theta = stddev[idx.copula.theta_rho]
        # Match get_copula_vals clamp
        rho = clamp(exp(theta_rho) + cc.clayton_rho_floor, cc.clayton_rho_floor, cc.clayton_rho_max)
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
- `coef::AbstractVector{<:Real}`: Coefficient vector
- `var_cov_matrix::AbstractMatrix{<:Real}`: Variance-covariance matrix from `var_cov_mat`
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
- `L_scaling::Int=0`: Number of scaling variables of a scaling-property model, 0 for other
  models. When it is positive, the header prints `:scaling` as the heteroscedastic parameter
  and this number as L.
- `n_draws::Int=0`: Number of simulation draws per observation
- `sign::Int=1`: Frontier type (1=production, otherwise=cost)
- `optim_result=nothing`: Optional Optim.jl result for convergence info
- `loglik=nothing`: the log-likelihood to print; `-optim_result.minimum` when not given. The
  fit passes the value at the reported coefficients, which after a failed line
  search differs from `-optim_result.minimum`.
- `integration=nothing`: the result of the integration check (`result.integration_check`);
  when it is given, the line "Integration check: …" is printed after "Converged:".
- `finish=nothing`: the BFGS finish of a Laplace fit (`result.finish`); when it is given, "Number of
  iterations:" counts the main solver's and the finish's iterations, and the line "BFGS finish: …" is
  printed directly after "Converged:".
- `vcov_type::Symbol=:Hessian`: what `var_cov_matrix` was computed from; with `:OPG` (the outer
  product of gradients, used for Laplace noise) the line "Standard errors: outer product of
  gradients (OPG)" is printed after "Log-likelihood:".
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
function print_table(coef::AbstractVector{<:Real}, var_cov_matrix::AbstractMatrix{<:Real},
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
                     finish=nothing,
                     loglik=nothing,
                     integration=nothing,
                     vcov_type::Symbol=:Hessian,
                     table_format::Symbol=:text)

    nofpara = length(coef)
    model = _build_model(noise, ineff; copula=copula)
    idx = _param_ind(model, K, L, hetero; L_scaling=L_scaling)

    # Compute statistics
    stddev = _stderr_from_vcov(var_cov_matrix)  # NaN-safe: non-PD vcov ⇒ NaN SEs, no DomainError
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
    # A scaling-property model (any inefficiency distribution) carries an empty `hetero` and L = 1,
    # its inefficiency parameters being constants, so on those alone these lines would read
    # "Homoscedastic model" and L = 1 whatever the number of scaling variables. They name the
    # scaling function and print `L_scaling` instead, as :MLE does.
    if L_scaling > 0
        print("Heteroscedastic parameters: "); printstyled(":scaling"; color=:yellow); println()
    elseif !isempty(hetero)
        print("Heteroscedastic parameters: "); printstyled(hetero; color=:yellow); println()
    else
        println("Homoscedastic model (no heteroscedasticity)")
    end
    print("Number of observations: "); printstyled(nobs; color=:yellow); println()
    print("Number of frontier regressors (K): "); printstyled(K; color=:yellow); println()
    print("Number of Z columns (L): "); printstyled(L_scaling > 0 ? L_scaling : L; color=:yellow); println()
    if n_draws > 0
        print("Number of draws: "); printstyled(n_draws; color=:yellow); println()
    end
    print("Frontier type: "); printstyled(sign == 1 ? "production" : "cost"; color=:yellow); println()
    print("GPU computing: "); printstyled(GPU; color=:yellow); println()

    if optim_result !== nothing
        if finish === nothing
            print("Number of iterations: "); printstyled(optim_result.iterations; color=:yellow); println()
        else
            # a Laplace fit continued by the BFGS finish: the main solver's iterations and the finish's
            print("Number of iterations: ")
            printstyled("$(finish.main_iterations) ($(finish.main_solver)) + $(finish.iterations) (BFGS finish)";
                        color=:yellow)
            println()
        end
        converged = Optim.converged(optim_result)
        print("Converged: "); printstyled(converged; color=converged ? :yellow : :red); println()
        if finish !== nothing
            print("BFGS finish: ")
            printstyled("the main solver stopped without convergence; BFGS continued from its endpoint.";
                        color=:yellow)
            println()
        end
        if integration !== nothing
            print("Integration check: ")
            printstyled(sf_integration_line(integration);
                        color = integration.status in (:OK, :not_available) ? :yellow : :red)
            println()
        end
        # The fit passes the log-likelihood at the reported coefficients
        ll_show = loglik === nothing ? -optim_result.minimum : loglik
        print("Log-likelihood: "); printstyled(round(ll_show; digits=5); color=:yellow); println()
    end
    if vcov_type === :OPG
        print("Standard errors: "); printstyled("outer product of gradients (OPG)"; color=:yellow); println()
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
        println("Log-parameters converted to original scale:")

        aux_data = Matrix{Any}(undef, length(aux_entries), 3)
        for (row, (name, param_idx, offset)) in enumerate(aux_entries)
            aux_data[row, 1] = name
            aux_data[row, 2] = exp(coef[param_idx]) + offset
            aux_data[row, 3] = exp(coef[param_idx]) * stddev[param_idx]  # Delta method (offset is a constant)
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

# `coef` and `var_cov_matrix` are deliberately NOT tied to the spec's element type. The
# optimizer returns a Float64 coefficient vector whatever the data's element type is (the init
# plumbing casts with Float64.(init)), so tying them would raise a MethodError for a Float32
# spec under show_table=true, which is the default. The panel backend's panel_print_table has the
# same untied form, and the docstring below advertises it. Display only: nothing
# here computes, so mixed element types cost nothing.

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
- `loglik=nothing`: the log-likelihood to print; `-optim_result.minimum` when not given. The
  fit passes the value at the reported coefficients, which after a failed line
  search differs from `-optim_result.minimum`.
- `integration=nothing`: the result of the integration check (`result.integration_check`);
  when it is given, the line "Integration check: …" is printed after "Converged:".
- `finish=nothing`: the BFGS finish of a Laplace fit (`result.finish`); when it is given, "Number of
  iterations:" counts the main solver's and the finish's iterations, and the line "BFGS finish: …" is
  printed directly after "Converged:".
- `vcov_type::Symbol=:Hessian`: what `var_cov_matrix` was computed from; with `:OPG` (the outer
  product of gradients, used for Laplace noise) the line "Standard errors: outer product of
  gradients (OPG)" is printed after "Log-likelihood:".
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
function print_table(spec::sfmodel_MSLE_spec, coef::AbstractVector,
                     var_cov_matrix::AbstractMatrix;
                     GPU::Bool=false,
                     optim_result=nothing,
                     finish=nothing,
                     loglik=nothing,
                     integration=nothing,
                     vcov_type::Symbol=:Hessian,
                     table_format::Symbol=:text)
    # Extract from spec
    return print_table(coef, var_cov_matrix,
                       spec.varnames, spec.eqnames, spec.eq_indices;
                       nobs=spec.N, noise=spec.noise, ineff=spec.ineff,
                       copula=spec.copula,
                       K=spec.K, L=spec.L, hetero=spec.hetero,
                       L_scaling=spec.L_scaling,
                       n_draws=size(spec.draws_2D, 2), sign=spec.sign,
                       GPU=GPU,
                       optim_result=optim_result, finish=finish, loglik=loglik, integration=integration,
                       vcov_type=vcov_type, table_format=table_format)
end

# ============================================================================
# Section 11b: Model Fitting (sfmodel_MSLE_fit)
# ============================================================================

"""
    sfmodel_MSLE_fit(; spec, method=sfmodel_method(), init=nothing, optim_options=nothing,
                     jlms_bc_index=true, marginal=true, integration_check=false, show_table=true,
                     shape_warning=true, verbose=true)

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
  Starting values at or beyond a parameter's numerical bounds are refused (an error) for a
  homoscedastic parameter and give a warning for a heteroscedastic one; a fit held at its starting
  values (no warmstart stage and `iterations = 0` in the main stage) is not checked.
- `optim_options=nothing`: Optimization options from `sfmodel_MSLE_opt()`. If nothing, uses defaults:
  NelderMead warmstart (200 iterations, `g_abstol = 1e-3`) + Newton main (200 iterations,
  `g_abstol = 1e-7`), the same default `sfmodel_fit` uses for every method. For `noise = :Laplace` the
  Newton stage of this default runs at most 30 iterations, and a fit it leaves unconverged continues with
  BFGS (see `sfmodel_MSLE_opt`).
- `jlms_bc_index::Bool=true`: Compute JLMS and BC efficiency indices
- `marginal::Bool=true`: Compute marginal effects of exogenous determinants on E(u)
- `integration_check=false`: Run the integration check after the fit: at the estimate, the simulated
  log-likelihood is evaluated again with four times the draws, and the one-step shift −V·(g_4S − g_S)
  is reported in standard errors; each part reads OK, CAUTION or FAIL and the result is stored in
  `result.integration_check`, separately from `converged` and `redflag`. `true` uses the default
  thresholds (0.1 and 1 for the change in the log-likelihood, 0.1 and 0.5 standard errors for the
  shift); a NamedTuple with any of `ll_caution`, `ll_fail`, `shift_caution`, `shift_fail` changes
  them. The thresholds are heuristics.
- `show_table::Bool=true`: Print estimation table
- `shape_warning::Bool=true`: Print the warning of the shape-range check and the warning that a Lomax
  fit's marginal effects are undefined where α ≤ 1; `false` prints neither (`shape_check` is stored
  either way)
- `verbose::Bool=true`: Print detailed progress and results

# Returns
A NamedTuple containing:
- `converged`: Whether optimization converged; after a BFGS finish (see `finish`), whether the finish converged
- `iter_limit_reached`: Whether iteration limit was reached; after a BFGS finish, by the finish
- `n_observations`: Number of observations
- `loglikelihood`: Log-likelihood value
- `table`: Coefficient table matrix
- `coeff`: Coefficient vector. A homoscedastic coefficient that ended beyond its numerical bound is
  reported at the bound, where the likelihood was evaluated (see `bound_check`, whose entry keeps the
  optimizer's value in `coef_before`).
- `std_err`: Standard errors (for Laplace noise from the outer product of gradients; see `vcov_type`)
- `var_cov_mat`: Variance-covariance matrix: the inverse Hessian of the negative log-likelihood, or for
  Laplace noise the inverse outer product of gradients (OPG) of the per-observation log-likelihoods
- `vcov_type`: `:Hessian`, or `:OPG` for Laplace noise: what `var_cov_mat` and `std_err` come from
- `jlms`: JLMS inefficiency indices (if computed)
- `bc`: BC efficiency indices (if computed)
- `OLS_loglikelihood`: OLS log-likelihood
- `OLS_resid_skew`: Skewness of OLS residuals
- `marginal`: DataFrame of marginal effects (if computed); NaN where E(u) does not exist (Lomax α ≤ 1),
  with a warning (unless `shape_warning = false`)
- `marginal_mean`: Mean marginal effects
- `model`: The model specification
- `n_draws`, `multiRand`, `chunks`, `draw_scheme`, `distinct_Halton_length`, `user_draws_supplied`,
  `estimation_method`: the draw settings of the fit. `draw_scheme` is `:halton` (one Halton sequence of
  N × n_draws points, no point reused), `:halton_recycled` (a sequence recycled across observations),
  `:halton_shared` (`multiRand = false`) or `:user` (user-supplied draws); `distinct_Halton_length` is the
  length of the Halton sequence the draws came from (N × n_draws, the recycled length, n_draws for shared
  draws, 0 for user draws).
- `Hessian`: Hessian matrix of the negative log-likelihood at the estimate; `nothing` for Laplace noise
- `OPG`: for Laplace noise the outer product of gradients J'J, with J the N × p Jacobian of the
  per-observation log-likelihoods at the estimate; `nothing` otherwise
- `gradient_norm`: Sup-norm of the log-likelihood gradient at the estimate (for a
  gradient-free main solver computed there with ForwardDiff)
- `actual_iterations`: Total iterations across all stages: warmstart, main and the BFGS finish
- `warmstart_solver`, `warmstart_ini`, `warmstart_maxIT`: Warmstart info
- `main_solver`, `main_ini`, `main_maxIT`: Main optimization info
- `finish`: for Laplace noise, when the main solver stopped without convergence and BFGS continued from its
  endpoint with the same `main_opt` (see `sfmodel_MSLE_opt`), a NamedTuple with `solver` (`"BFGS"`),
  `main_solver` (the name of the main solver), `main_iterations`, `main_converged` (`false`), `iterations` and
  `converged` (those of the finish), `newton_cap` (30 when the Newton stage of the default settings was capped
  at 30 iterations, otherwise `nothing`) and `trace` (the finish's iteration trace when `store_trace` is set,
  otherwise `nothing`); `nothing` when no finish ran. `converged`, `iter_limit_reached`, `loglikelihood`,
  `gradient_norm`, `redflag` and the estimates then refer to the point where the finish stopped.
- `redflag`: Convergence/quality flag (the highest applicable value is reported, except
  that a boundary solution is always 1): 0 = OK; 1 = convergence problems (gradient
  norm > 0.1 or `NaN`; Optim did not report convergence, which includes stopping at
  the iteration limit; the line search failed, even if Optim reports convergence; or
  the Hessian (for Laplace noise, the outer product of gradients) could not be computed, is not
  invertible, is numerically singular, with a
  reciprocal condition number below machine epsilon, or is not positive definite), or a
  boundary solution (a parameter at or beyond its numerical bound: a variance, scale,
  shape or copula dependence parameter; see `bound_check`); 2 = the variance-covariance
  matrix has non-positive diagonal elements, so the affected standard errors are `NaN`.
  The rule is the same for every method. After a BFGS finish (see `finish`), the convergence part
  refers to the finish.
- `hessian_flag`: the Hessian part of `redflag` (0, 1 or 2), before a boundary solution
  sets `redflag` to 1; for Laplace noise it is read from the outer product of gradients.
- `bound_check`: one entry per parameter and side at or beyond its numerical bound (empty
  when none), a NamedTuple
  `(parameter, label, coefficient, side, bound, scalar, n_at, n, share, moved, coef_before)`;
  filled whatever `verbose` and `show_table` are.
- `integration_check`: the result of the integration check (the option `integration_check`), or
  `nothing` when the check is off: a NamedTuple with `status` (`:OK`, `:CAUTION`, `:FAIL`,
  `:not_available` or `:error`), `reason`, `n_draws` and `n_draws_check` (the fit's draws and
  four times as many), `loglik`, `loglik_check`, `delta_loglik`, `delta_loglik_per_obs`,
  `ll_verdict`, `shift_available`, `shift_note`, `shift`, `shift_se`, `max_shift_se`,
  `max_shift_param`, `max_shift_estimate`, `max_shift_onestep` (the largest shift in standard
  errors, the parameter it belongs to, its estimate and its value after the shift),
  `shift_verdict`, `thresholds` and `seconds`. It is separate from `converged` and `redflag`.
- `shape_check`: the verdict of the shape-range check for Lomax fits, `nothing` for the other
  families: a NamedTuple with `method`, `family`, `parameter`, `rule` (`nothing`), `threshold`,
  `in_range`, `n_below`, `n`, `share` and `min_value`. A Lomax fit whose estimated shape α lies below
  `threshold` for some observations prints a warning (unless `shape_warning = false`); `in_range` is then
  `true`, `n_below` of the `n` observations (the share `share`) lie below it, and `min_value` is the
  smallest value. It is separate from `converged`, `redflag` and `integration_check`.
- `list`: OrderedDict containing all results

# Example
```julia
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
    n_draws = 2^12
)



# Construct initial values
p0 = sfmodel_MSLE_init(
    spec = m,
    frontier = X \\ y,  # OLS
    mu = [0.0],
    ln_sigma_u_sq = (0.0),
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
    integration_check=false,
    show_table::Bool=true,
    shape_warning::Bool=true,
    verbose::Bool=true
)
    # The thresholds of the integration check after the fit, or `nothing` when it is off; an invalid
    # value is an error before anything is computed.
    _ic_settings = sf_integration_settings(integration_check)

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

    if init === nothing
        # Starting value: solved again on the host in Float64 from the specification's
        # own arrays rather than copied from β_ols_raw, so that the default start
        # depends on neither the element type nor the device and is exactly what
        # `sfmodel_init` returns. β_ols_raw stays element-type-native and on-device for
        # the OLS residual statistics below.
        β_ols = Float64.(Array(spec.frontier)) \ Float64.(Array(spec.depvar))
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
        expected_len = plen(model.model, model.K, model.L, model.hetero; L_scaling=model.L_scaling)
        length(sf_init) == expected_len ||
            error("Length mismatch in sfmodel_fit(): init has $(length(sf_init)) elements, expected $expected_len.")
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

    # Starting values at or beyond a bound (`sf_start_check`, src/sf_redflag.jl): a warning for each heteroscedastic
    # parameter there, then an error when a homoscedastic coefficient is there. A fit held at its starting values (no
    # warmstart stage, no main iteration) is not checked.
    held = !do_warmstart && myopt.main_opt.iterations == 0
    if !held
        start_error, start_warnings = sf_start_check(_bound_list(model),
                                                     _bound_verdict(model, sf_init, spec.zvar), sf_init)
        foreach(w -> sf_print_warning("Warning: " * w), start_warnings)
        isempty(start_error) || throw(ArgumentError(start_error))
    end

    # GPU fits: warn before the optimizer when one Hessian evaluation is estimated to need
    # more device memory than is free.
    if method.GPU
        sf_check_gpu_memory(model.copula == :None ? SF_WORK_ARRAYS.msle : SF_WORK_ARRAYS.msle_copula,
                            model.N, size(model.draws_2D, 2), model.chunks, length(sf_init);
                            unit = "observations")
    end

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

    # Laplace noise under the default main settings (Newton, 200 iterations, g_abstol = 1e-7, with or without
    # `store_trace`; `==` compares every field of the options, so any other setting makes them the user's): the
    # Newton stage runs at most 30 iterations, and a fit it leaves unconverged continues with BFGS (the finish below).
    _laplace_cap = model.noise === :Laplace && myopt.main_solver isa Optim.Newton &&
                   myopt.main_opt == Optim.Options(iterations = 200, g_abstol = 1e-7,
                                                   store_trace = myopt.main_opt.store_trace)
    _main_opt = _laplace_cap ?
        Optim.Options(iterations = 30, g_abstol = 1e-7, store_trace = myopt.main_opt.store_trace) : myopt.main_opt

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
                                  _main_opt; autodiff=autodiff_mode)

        sf_total_iter += Optim.iterations(_optres)

        # The BFGS finish for Laplace noise: at the kinks of the Laplace density Newton tends to stop at its iteration
        # limit, so a main stage that stops without convergence (or whose last line search failed) continues from its
        # endpoint with BFGS and the main options; everything below refers to the finish. Not when the main solver is
        # BFGS or LBFGS, or when `main_opt` allows no iterations (a fit held at its starting values).
        _main_res = _optres
        _finish = nothing
        if model.noise === :Laplace && myopt.main_opt.iterations > 0 &&
           !(myopt.main_solver isa Union{Optim.BFGS, Optim.LBFGS}) && sf_convergence_flag(_optres; message = false) == 1
            verbose && printstyled("Laplace noise: the main solver stopped without convergence after " *
                                   "$(Optim.iterations(_main_res)) iterations; continuing with BFGS from its " *
                                   "endpoint.\n"; color = :green)
            _optres = Optim.optimize(_lik, Optim.minimizer(_main_res), BFGS(), myopt.main_opt;
                                     autodiff = autodiff_mode)
            sf_total_iter += Optim.iterations(_optres)
            _finish = (solver = "BFGS", main_solver = string(nameof(typeof(myopt.main_solver))),
                       main_iterations = Optim.iterations(_main_res), main_converged = false,
                       iterations = Optim.iterations(_optres), converged = Optim.converged(_optres),
                       newton_cap = _laplace_cap ? 30 : nothing,
                       trace = _sf_optim_trace(_optres, myopt.main_opt.store_trace))
        end

        # The objective and gradient norm at the point the result reports: after a
        # line-search failure Optim's own values describe the previous iterate (sf_final_fg,
        # src/sf_redflag.jl). For a gradient-free main solver (NelderMead,
        # ParticleSwarm, SimulatedAnnealing) the gradient norm is the ForwardDiff gradient there.
        _fmin, _gnorm = sf_final_fg(_optres, _lik)

        if verbose
            println()
            println(_optres)
            print("The resulting coefficient vector is:\n")
            printstyled(Optim.minimizer(_optres); color=:yellow)
            println("\n")
        end

        # Check convergence
        if isnan(_gnorm) || (_gnorm > 0.1)
            redflag = 1
            printstyled("Note that the estimation may not have converged properly. The gradients are problematic (too large, > 0.1, or NaN).\n\n"; color=:red)
        end

        # "Optim did not report convergence" counts, whatever stopped the run.
        # The iteration limit counts only through it: a fit that converges on exactly its
        # last allowed iteration is not flagged. A failed line search counts too,
        # even when Optim reports convergence. The rule and its messages are shared by the
        # four backends (sf_convergence_flag, src/sf_redflag.jl).
        redflag = max(redflag, sf_convergence_flag(_optres))
    end

    # ========== Post-Estimation ==========
    _coevec = Optim.minimizer(_optres)

    # Hessian, computed once: the matrix classified below is the one stored on the result.
    # Variance-covariance matrix and the Hessian part of redflag, by the rule shared with
    # the other backends. A parameter at or beyond its numerical bound (a variance, scale,
    # shape or copula dependence parameter) makes the fit a boundary solution: redflag 1,
    # whatever the Hessian says, and the message names each such parameter. The verdict is taken
    # at the optimizer's point from the host copy of Z (`spec.zvar`) and stored as `bound_check`.
    # The Hessian is computed and classified by sf_classify_hessian(f, x) (src/sf_redflag.jl),
    # so an error while computing it does not end the fit and discard the estimate: the fit
    # reads as one whose Hessian cannot be inverted (redflag 1, all-NaN Hessian and var-cov
    # matrix), and a message, printed whatever `verbose` is, gives the error. Ctrl-C, out of
    # memory, a stack overflow and CUDA's out-of-GPU-memory error still end the fit.
    bc = _bound_verdict(model, _coevec, spec.zvar)
    # A homoscedastic coefficient beyond its bound is moved to the bound (`sf_bound_move`, src/sf_redflag.jl), where
    # the likelihood and its derivatives equal those at the optimizer's point; the moved vector is reported from
    # JLMS/BC on, and the entry in `bound_check` keeps the optimizer's value.
    bc, coef_moved = sf_bound_move(_bound_list(model), bc, _coevec)
    # Laplace noise: the Laplace density has a kink at zero, and the Hessian of the simulated log-likelihood is not a
    # reliable measure of its curvature, so the var-cov matrix is the inverse of the outer product of gradients of the
    # per-observation log-likelihoods (sf_classify_opg, src/sf_redflag.jl), computed in the chunk loop and classified
    # by the same rules; `Hessian` is then `nothing` and the outer product is stored as `OPG`.
    _btext = isempty(bc) ? "" : sf_bound_message(bc)
    if model.noise === :Laplace
        vcov_result = sf_classify_opg(p -> qmc_nll(model, p; chunks = model.chunks, per_obs = true), _coevec;
                                      at_clamp = !isempty(bc), message = verbose, bound_text = _btext)
        vcov_type = :OPG
    else
        vcov_result = sf_classify_hessian(_lik, _coevec; at_clamp = !isempty(bc), message = verbose,
                                          bound_text = _btext)
        vcov_type = :Hessian
    end
    numerical_hessian = vcov_result.hessian           # nothing for Laplace fits
    var_cov_matrix = vcov_result.var_cov_matrix
    redflag = max(redflag, vcov_result.redflag)

    stddev = _stderr_from_vcov(var_cov_matrix)  # non-PD vcov ⇒ NaN SEs (no DomainError); keeps stderror(r) NaN-consistent with sqrt.(diag(vcov(r))).

    # ========== Shape-Range Check ==========
    # For the Lomax, a warning when the estimated shape lies where the default number of draws is unreliable
    # (SF_SHAPE_RANGES, src/sf_accuracy.jl); the verdict is stored and no number changes. `spec.zvar` is the
    # host copy of Z.
    _sc_entry = sf_shape_entry(:MSLE, model.ineff, nothing)
    _sc = _sc_entry === nothing ? nothing :
        sf_shape_check(_sc_entry, getproperty(get_ineff_vals(model.model.ineff, _coevec, model.idx, spec.zvar,
                                                             model.constants), _sc_entry.parameter);
                       shape_warning = shape_warning)

    # ========== Integration Check (optional) ==========
    # At the estimate: the simulated log-likelihood again with four times the draws, and the one-step
    # shift that they imply, in standard errors (src/sf_accuracy.jl). The draws with four times
    # `n_draws` are built without the warning about identical draws, which would name the check's
    # number of draws rather than the fit's.
    _ic = _ic_settings === nothing ? nothing :
        sf_run_integration_check(_ic_settings;
            S = size(model.draws_2D, 2), n_obs = N, ll = -_fmin, theta = _coevec, nll_S = _lik,
            build_nll_4S = () -> begin
                m4 = _assemble_MSLE_spec(spec, _with_draws(method, 4 * method.n_draws, min(N, 4 * method.chunks));
                                         warn_repeated = false)
                p -> qmc_nll(m4, p; chunks = m4.chunks)
            end,
            V = var_cov_matrix, se = stddev, hessian_flag = vcov_result.hessian_flag,
            names = model.varnames, unavailable = sf_integration_unavailable(method, N))

    # The reported coefficients from here on: a homoscedastic coefficient beyond its bound at the bound. The Hessian,
    # the standard errors and the checks above are those at the optimizer's point.
    _coevec = coef_moved

    # ========== JLMS and BC Indices ==========
    if jlms_bc_index
        eff_result = jlms_bc_indices(model, _coevec; chunks = model.chunks)
        # Store jlms/bc on host: post-estimation analysis reads them repeatedly,
        # so one bulk device→host copy here beats a transfer per use
        # (no-op for CPU fits).
        _jlms = eff_result.jlms isa Array ? eff_result.jlms : Array(eff_result.jlms)
        _bc   = eff_result.bc   isa Array ? eff_result.bc   : Array(eff_result.bc)
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

    # A Lomax fit's marginal effects are NaN where α ≤ 1, since E(u) = λ/(α − 1) does not exist there
    # (every row with a NaN counts): one warning per fit, whatever `show_table` and `verbose` are, unless
    # `shape_warning = false`.
    if margeff !== nothing && model.model.ineff isa Lomax_MSLE
        M = Matrix(margeff)
        n_nan = count(i -> any(isnan, view(M, i, :)), axes(M, 1))
        sf_warn_lomax_mean(n_nan, size(M, 1); shape_warning = shape_warning, after_fit = true)
    end

    # ========== Print Results Table ==========
    table_result = nothing
    if show_table
        table_result = print_table(model, _coevec, var_cov_matrix;
                                    GPU=method.GPU, optim_result=_optres, finish=_finish, loglik=-_fmin,
                                    integration=_ic, vcov_type=vcov_type, table_format=:text)

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
            println("* Marginal effects of the inefficiency determinants at the observational level are saved in the return. See below.\n")
        end

        println("* Use `name.list` to see saved results (keys and values) where `name` is the return specified in `name = sfmodel_fit(..)`. Values may be retrieved using the keys. For instance:")
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
    _dicRES[:loglikelihood] = -_fmin          # at the reported coefficients
    _dicRES[:table] = table_result !== nothing ? table_result.table : nothing
    _dicRES[:coeff] = _coevec
    _dicRES[:std_err] = stddev
    _dicRES[:var_cov_mat] = var_cov_matrix
    _dicRES[:vcov_type] = vcov_type
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
    _dicRES[:OPG] = vcov_type === :OPG ? vcov_result.opg : nothing
    _dicRES[:gradient_norm] = _gnorm
    _dicRES[:actual_iterations] = sf_total_iter
    _dicRES[:______________] = "______________________"
    _dicRES[:warmstart_solver] = sf_warmstart_algo
    _dicRES[:warmstart_ini] = sf_init_1st
    _dicRES[:warmstart_maxIT] = sf_warmstart_maxit
    _dicRES[:main_solver] = sf_main_algo
    _dicRES[:main_ini] = sf_init_2nd
    _dicRES[:main_maxIT] = sf_main_maxit
    _dicRES[:main_tolerance] = myopt.main_opt.g_abstol
    _dicRES[:finish] = _finish                # the BFGS finish of a Laplace fit; nothing when none ran
    # the main stage's trace, for plot(:convergence); the finish's trace is in `finish`
    _dicRES[:trace] = _sf_optim_trace(_main_res, myopt.main_opt.store_trace)
    _dicRES[:redflag] = redflag
    _dicRES[:hessian_flag] = vcov_result.hessian_flag   # the Hessian part of redflag
    _dicRES[:bound_check] = bc                # each parameter at or beyond its numerical bound
    _dicRES[:integration_check] = _ic         # nothing when the option is off
    _dicRES[:shape_check] = _sc               # nothing for the families and rules SF_SHAPE_RANGES does not cover
    _dicRES[:______________________] = "___________________"
    _dicRES[:GPU] = method.GPU
    # Report the number of draws actually used and whether user draws were
    # supplied, so predict() can round-trip. `model.draws_2D` is the (1×D or
    # N×D) matrix `_prepare_draws` built, so `size(...,2)` is the real D for every
    # input shape — `size(method.draws)[end]` would be 1 for a multiRand=false
    # D×1 matrix, making that fit impossible to predict. Draws given with `draws` are
    # always the ones the fit uses (`_prepare_draws` shapes them by their row count), so
    # `user_draws_supplied` records whether `draws` was given.
    _dicRES[:n_draws] = size(model.draws_2D, 2)
    _dicRES[:multiRand] = method.multiRand
    _dicRES[:chunks] = method.chunks
    _scheme, _hlen = sf_draw_scheme(model.N, size(model.draws_2D, 2); user = method.draws !== nothing,
                                    multiRand = method.multiRand, recycling = method.recycling,
                                    legacy = method.halton_legacy)
    _dicRES[:draw_scheme] = _scheme                # :halton, :halton_recycled, :halton_shared or :user
    _dicRES[:distinct_Halton_length] = _hlen       # length of the Halton sequence the draws came from
    _dicRES[:user_draws_supplied] = method.draws !== nothing
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
        # HalfNormal stores its ln σᵤ² under :sigma_sq; also expose it under
        # :sigma_u for parity with TruncatedNormal (the manual reserves :sigma_u for
        # HalfNormal/TN). Lognormal/Rayleigh keep only :sigma_sq — their parameter is
        # Var(ln u) / a squared scale, not σᵤ².
        if model.model.ineff isa HalfNormal_MSLE
            _dicRES[:sigma_u] = _coevec[idx.ineff.sigma_sq]
        end
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
        _dicRES[:theta] = _coevec[idx.ineff.theta]   # ln θ coefficient block (parity with the MCI backend)
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
export HalfNormal_MSLE, Weibull_MSLE, Lognormal_MSLE, Lomax_MSLE, Rayleigh_MSLE, Erlang_MSLE
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
# p = [beta(3), ln_sigma_v_sq(1), delta(3), ln_sigma_u_sq(1)] -> length = 8
p_ntn = [1.0, 0.5, -0.3, -1.0, 0.5, 0.1, -0.2, -0.5]
nll_ntn = qmc_nll(Y, X, Z, p_ntn, halton; noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])
println("Normal + TN NLL: $nll_ntn")

# ============================================================================
# Example 2: Normal + Exponential (hetero sigma_u_sq)
# ============================================================================
# p = [beta(3), ln_sigma_v_sq(1), gamma(3)] -> length = 7
p_ne = [1.0, 0.5, -0.3, -1.0, 0.5, 0.1, -0.2]
nll_ne = qmc_nll(Y, X, Z, p_ne, halton; noise=:Normal, ineff=:Exponential, hetero=[:sigma_u_sq])
println("Normal + Exponential NLL: $nll_ne")

# ============================================================================
# Example 3: Student T + Truncated Normal (hetero mu)
# ============================================================================
# p = [beta(3), ln_sigma_v_sq(1), ln_nu_minus_2(1), delta(3), ln_sigma_u_sq(1)] -> length = 9
p_ttn = [1.0, 0.5, -0.3, -1.0, 1.0, 0.5, 0.1, -0.2, -0.5]  # ln_nu_minus_2=1.0 -> nu≈4.7
nll_ttn = qmc_nll(Y, X, Z, p_ttn, halton; noise=:StudentT, ineff=:TruncatedNormal, hetero=[:mu])
println("StudentT + TN NLL: $nll_ttn")

# ============================================================================
# Example 4: Student T + Exponential (hetero sigma_u_sq)
# ============================================================================
# p = [beta(3), ln_sigma_v_sq(1), ln_nu_minus_2(1), gamma(3)] -> length = 8
p_te = [1.0, 0.5, -0.3, -1.0, 1.0, 0.5, 0.1, -0.2]
nll_te = qmc_nll(Y, X, Z, p_te, halton; noise=:StudentT, ineff=:Exponential, hetero=[:sigma_u_sq])
println("StudentT + Exponential NLL: $nll_te")

# ============================================================================
# Example 5: Hetero validation error
# ============================================================================
# This will error with a helpful message:
# qmc_nll(Y, X, Z, p_ntn, halton; noise=:Normal, ineff=:TruncatedNormal, hetero=[:lambda])
# ERROR: Invalid hetero option :lambda for TruncatedNormal_MSLE. Valid options: [:mu, :sigma_u_sq]

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
