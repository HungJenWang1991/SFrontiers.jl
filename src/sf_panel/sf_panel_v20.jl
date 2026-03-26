# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

#=
    sf_panel_v20.jl

    Panel stochastic frontier model using Wang and Ho (2010) framework.
    Supports both MSLE (inverse CDF) and MCI (change-of-variable) estimation methods.

    Model (after within-transformation to eliminate individual effects αᵢ):
        ỹ_it = x̃_it'β + ṽ_it − h̃_it · u_i*
    where:
        v_it ~ N(0, σ_v²)           noise
        u_i* ~ N⁺(0, σ_u²)          firm-specific persistent inefficiency
        h(z_it) = exp(z_it'δ)        scaling function (h̃ is demeaned h)

    Supported distributions (extensible via dispatch):
        Noise: Normal
        Inefficiency: HalfNormal, TruncatedNormal, Exponential, Weibull,
                      Lognormal, Lomax, Rayleigh, Gamma (MCI only)

    Features:
        - Multiple dispatch for model combinations (no if-else logic)
        - CPU and GPU support via array type dispatch
        - ForwardDiff compatible via type parameter P
        - Balanced and unbalanced panels (per-firm loop with offsets for unbalanced)
        - Chunking for GPU memory management
=#

using HaltonSequences
using SpecialFunctions: erfinv, erf, loggamma
using Statistics: mean
using LinearAlgebra: diag, inv, I
using ForwardDiff: hessian, gradient
using Distributions: Normal, cquantile, ccdf, normlogpdf
using PrettyTables: pretty_table, fmt__printf
using Optim
using ADTypes: AutoForwardDiff
using OrderedCollections: OrderedDict
using DataFrames: DataFrame, rename, names as df_names


# ============================================================================
# Section 1: Type Hierarchy
# ============================================================================

"""Abstract type for noise models in panel stochastic frontier estimation."""
abstract type PanelNoiseModel end

"""Abstract type for inefficiency models in panel stochastic frontier estimation."""
abstract type PanelIneffModel end

"""Normal noise: v ~ N(0, σ_v²)"""
struct NormalNoise_Panel <: PanelNoiseModel end

"""Half Normal inefficiency: u* ~ N⁺(0, σ_u²)"""
struct HalfNormal_Panel <: PanelIneffModel end

"""Truncated Normal inefficiency: u* ~ TN(μ, σ_u²; lower=0)"""
struct TruncatedNormal_Panel <: PanelIneffModel end

"""Exponential inefficiency: u* ~ Exp(λ), where λ = Var(u*)"""
struct Exponential_Panel <: PanelIneffModel end

"""Weibull inefficiency: u* ~ Weibull(λ, k) with scale λ and shape k"""
struct Weibull_Panel <: PanelIneffModel end

"""Lognormal inefficiency: u* ~ LogNormal(μ, σ)"""
struct Lognormal_Panel <: PanelIneffModel end

"""Lomax inefficiency: u ~ Lomax(α, λ)"""
struct Lomax_Panel <: PanelIneffModel end

"""Rayleigh inefficiency: u* ~ Rayleigh(σ)"""
struct Rayleigh_Panel <: PanelIneffModel end

"""Gamma inefficiency: u* ~ Gamma(k, θ) with shape k and scale θ. MCI only."""
struct Gamma_Panel <: PanelIneffModel end

"""
    PanelModel{N<:PanelNoiseModel, U<:PanelIneffModel}

Composite type representing a panel SF model with specific noise and inefficiency
distributions. Enables multiple dispatch on model combinations.
"""
struct PanelModel{N<:PanelNoiseModel, U<:PanelIneffModel}
    noise::N
    ineff::U
end

# Registry for symbol-based lookup
const PANEL_NOISE_MODELS = Dict{Symbol, Type{<:PanelNoiseModel}}(
    :Normal => NormalNoise_Panel,
)

const PANEL_INEFF_MODELS = Dict{Symbol, Type{<:PanelIneffModel}}(
    :HalfNormal      => HalfNormal_Panel,
    :TruncatedNormal => TruncatedNormal_Panel,
    :Exponential     => Exponential_Panel,
    :Weibull         => Weibull_Panel,
    :Lognormal       => Lognormal_Panel,
    :Lomax           => Lomax_Panel,
    :Rayleigh        => Rayleigh_Panel,
    :Gamma           => Gamma_Panel,
)

"""
    _build_panel_model(noise::Symbol, ineff::Symbol) -> PanelModel

Build a PanelModel from noise and inefficiency symbols.
"""
function _build_panel_model(noise::Symbol, ineff::Symbol)
    if !haskey(PANEL_NOISE_MODELS, noise)
        error("Unknown panel noise model: :$noise. Valid options: $(collect(keys(PANEL_NOISE_MODELS)))")
    end
    if !haskey(PANEL_INEFF_MODELS, ineff)
        error("Unknown panel inefficiency model: :$ineff. Valid options: $(collect(keys(PANEL_INEFF_MODELS)))")
    end
    return PanelModel(PANEL_NOISE_MODELS[noise](), PANEL_INEFF_MODELS[ineff]())
end


# ============================================================================
# Section 2: Parameter Indexing
# ============================================================================

"""
    panel_plen(::PanelIneffModel, K::Int, L::Int) -> Int

Total number of parameters for the panel model, dispatched by inefficiency distribution.
"""
panel_plen(::HalfNormal_Panel, K::Int, L::Int)       = K + L + 2
panel_plen(::TruncatedNormal_Panel, K::Int, L::Int)  = K + L + 3
panel_plen(::Exponential_Panel, K::Int, L::Int)      = K + L + 2
panel_plen(::Weibull_Panel, K::Int, L::Int)          = K + L + 3
panel_plen(::Lognormal_Panel, K::Int, L::Int)        = K + L + 3
panel_plen(::Lomax_Panel, K::Int, L::Int)            = K + L + 3
panel_plen(::Rayleigh_Panel, K::Int, L::Int)         = K + L + 2
panel_plen(::Gamma_Panel, K::Int, L::Int)            = K + L + 3

# Backward-compatible convenience (defaults to HalfNormal)
panel_plen(K::Int, L::Int) = panel_plen(HalfNormal_Panel(), K, L)

"""
    _panel_param_ind(::PanelIneffModel, K::Int, L::Int) -> NamedTuple

Compute parameter indices for the panel model, dispatched by inefficiency distribution.
Returns NamedTuple with distribution-specific fields. `ln_sigma_v_sq` is always last.
"""
function _panel_param_ind(::HalfNormal_Panel, K::Int, L::Int)
    idx = 1
    beta = idx:(idx + K - 1);   idx += K
    delta = idx:(idx + L - 1);  idx += L
    ln_sigma_u_sq = idx;        idx += 1
    ln_sigma_v_sq = idx
    return (beta=beta, delta=delta, ln_sigma_u_sq=ln_sigma_u_sq, ln_sigma_v_sq=ln_sigma_v_sq)
end

function _panel_param_ind(::TruncatedNormal_Panel, K::Int, L::Int)
    idx = 1
    beta = idx:(idx + K - 1);   idx += K
    delta = idx:(idx + L - 1);  idx += L
    mu = idx;                    idx += 1
    ln_sigma_u_sq = idx;        idx += 1
    ln_sigma_v_sq = idx
    return (beta=beta, delta=delta, mu=mu, ln_sigma_u_sq=ln_sigma_u_sq, ln_sigma_v_sq=ln_sigma_v_sq)
end

function _panel_param_ind(::Exponential_Panel, K::Int, L::Int)
    idx = 1
    beta = idx:(idx + K - 1);   idx += K
    delta = idx:(idx + L - 1);  idx += L
    ln_lambda = idx;             idx += 1
    ln_sigma_v_sq = idx
    return (beta=beta, delta=delta, ln_lambda=ln_lambda, ln_sigma_v_sq=ln_sigma_v_sq)
end

function _panel_param_ind(::Weibull_Panel, K::Int, L::Int)
    idx = 1
    beta = idx:(idx + K - 1);   idx += K
    delta = idx:(idx + L - 1);  idx += L
    ln_lambda = idx;             idx += 1
    ln_k = idx;                  idx += 1
    ln_sigma_v_sq = idx
    return (beta=beta, delta=delta, ln_lambda=ln_lambda, ln_k=ln_k, ln_sigma_v_sq=ln_sigma_v_sq)
end

function _panel_param_ind(::Lognormal_Panel, K::Int, L::Int)
    idx = 1
    beta = idx:(idx + K - 1);   idx += K
    delta = idx:(idx + L - 1);  idx += L
    mu = idx;                    idx += 1
    ln_sigma_sq = idx;           idx += 1
    ln_sigma_v_sq = idx
    return (beta=beta, delta=delta, mu=mu, ln_sigma_sq=ln_sigma_sq, ln_sigma_v_sq=ln_sigma_v_sq)
end

function _panel_param_ind(::Lomax_Panel, K::Int, L::Int)
    idx = 1
    beta = idx:(idx + K - 1);   idx += K
    delta = idx:(idx + L - 1);  idx += L
    ln_lambda = idx;             idx += 1
    ln_alpha = idx;              idx += 1
    ln_sigma_v_sq = idx
    return (beta=beta, delta=delta, ln_lambda=ln_lambda, ln_alpha=ln_alpha, ln_sigma_v_sq=ln_sigma_v_sq)
end

function _panel_param_ind(::Rayleigh_Panel, K::Int, L::Int)
    idx = 1
    beta = idx:(idx + K - 1);   idx += K
    delta = idx:(idx + L - 1);  idx += L
    ln_sigma_sq = idx;           idx += 1
    ln_sigma_v_sq = idx
    return (beta=beta, delta=delta, ln_sigma_sq=ln_sigma_sq, ln_sigma_v_sq=ln_sigma_v_sq)
end

function _panel_param_ind(::Gamma_Panel, K::Int, L::Int)
    idx = 1
    beta = idx:(idx + K - 1);   idx += K
    delta = idx:(idx + L - 1);  idx += L
    ln_k = idx;                  idx += 1
    ln_theta = idx;              idx += 1
    ln_sigma_v_sq = idx
    return (beta=beta, delta=delta, ln_k=ln_k, ln_theta=ln_theta, ln_sigma_v_sq=ln_sigma_v_sq)
end

# Backward-compatible convenience (defaults to HalfNormal)
_panel_param_ind(K::Int, L::Int) = _panel_param_ind(HalfNormal_Panel(), K, L)


# ============================================================================
# Section 3: Specification Structs
# ============================================================================

"""
    PanelModelSpec{T<:AbstractFloat}

Panel SF model specification. Holds raw (un-demeaned) data plus panel structure.
Demeaning is performed during assembly into the internal spec.
"""
struct PanelModelSpec{T<:AbstractFloat}
    # Raw data (stacked NT format: firm 1 all T₁ periods, firm 2 all T₂ periods, ...)
    depvar::AbstractVector{T}       # y, length NT_total
    frontier::AbstractMatrix{T}     # X, size NT_total × K
    zvar::AbstractMatrix{T}         # Z, size NT_total × L (for scaling function)

    # Panel structure
    N::Int                           # number of firms
    T_periods::Union{Int, Vector{Int}}  # time periods: scalar (balanced) or per-firm vector (unbalanced)
    T_max::Int                       # max time periods (= T_periods for balanced)
    offsets::Vector{Int}             # firm boundaries in stacked data, length N+1

    # Model
    noise::Symbol
    ineff::Symbol
    model::PanelModel

    # Parameter indices
    K::Int
    L::Int
    idx::NamedTuple
    sign::Int                        # +1 production, -1 cost

    # Metadata for output
    varnames::Vector{String}
    eqnames::Vector{String}
    eq_indices::Vector{Int}
end

"""
    PanelMethodSpec

Numerical method specification for panel estimation.
"""
struct PanelMethodSpec
    method::Symbol                   # :MSLE or :MCI
    transformation::Union{Symbol,Nothing}  # MCI only
    draws::Union{Nothing, AbstractVecOrMat{<:AbstractFloat}}
    n_draws::Int
    multiRand::Bool                  # per-firm draws (N×D) vs shared draws (1D)
    GPU::Bool
    chunks::Int
    distinct_Halton_length::Int      # max Halton sequence length for multiRand
end

"""
    PanelOptSpec

Optimization options for panel estimation.
"""
struct PanelOptSpec
    warmstart_solver::Any
    warmstart_opt::Union{Nothing, Optim.Options}
    main_solver::Any
    main_opt::Optim.Options
end

"""
    _PanelInternalSpec{T<:AbstractFloat}

Internal assembled spec with pre-processed data ready for NLL evaluation.
Created by `_assemble_panel_spec()` from PanelModelSpec + PanelMethodSpec.
"""
struct _PanelInternalSpec{T<:AbstractFloat}
    # Demeaned data (stacked NT format)
    y_tilde::AbstractVector{T}      # demeaned y
    x_tilde::AbstractMatrix{T}      # demeaned X
    z_raw::AbstractMatrix{T}        # raw Z (NOT demeaned; h(z) is demeaned inside NLL)

    # Panel structure
    N::Int                           # number of firms
    T_periods::Union{Int, Vector{Int}}  # time periods: scalar (balanced) or per-firm vector
    T_max::Int                       # max time periods (= T_periods for balanced)
    offsets::Vector{Int}             # firm boundaries in stacked data, length N+1
    Tm1::Union{Int, AbstractVector}  # T_i - 1: scalar (balanced) or N-vector (unbalanced; CuArray on GPU)

    # Pre-computed draws
    draws::AbstractVecOrMat{T}      # Halton draws: D-vector (shared) or N×D matrix (multiRand)

    # Model & parameters
    model::PanelModel
    noise::Symbol
    ineff::Symbol
    K::Int
    L::Int
    idx::NamedTuple
    sign::Int
    chunks::Int

    # Pre-computed constants
    constants::NamedTuple

    # Metadata
    varnames::Vector{String}
    eqnames::Vector{String}
    eq_indices::Vector{Int}

    # Method
    method::Symbol                   # :MSLE or :MCI
    transformation::Union{Symbol,Nothing}
    GPU::Bool
end


# ============================================================================
# Section 4: Panel Data Helpers
# ============================================================================

"""
    sf_panel_demean(x::AbstractVector, T::Int) -> AbstractVector

Within-group demeaning for balanced panel data.
Input: stacked vector of length N*T (firm 1 periods 1..T, firm 2 periods 1..T, ...).
Output: demeaned vector of same length.
"""
function sf_panel_demean(x::AbstractVector, T::Int)
    n = length(x)
    @assert n % T == 0 "Length ($n) must be divisible by T ($T)"
    reshaped = reshape(x, T, :)         # T × N_firms
    means = mean(reshaped, dims=1)      # 1 × N_firms
    demeaned = reshaped .- means        # T × N_firms
    return vec(demeaned)                 # back to N*T stacked vector
end

"""
    sf_panel_demean(X::AbstractMatrix, T::Int) -> AbstractMatrix

Within-group demeaning for each column of a matrix.
"""
function sf_panel_demean(X::AbstractMatrix, T::Int)
    NT, K = size(X)
    @assert NT % T == 0 "Number of rows ($NT) must be divisible by T ($T)"
    N = NT ÷ T
    # Reshape to 3D: T × N × K, demean along dim 1, reshape back
    X3 = reshape(X, T, N, K)
    means3 = mean(X3, dims=1)           # 1 × N × K
    demeaned3 = X3 .- means3            # T × N × K
    return reshape(demeaned3, NT, K)
end

"""
    compute_h_tilde(Z, p, idx, T_periods) -> AbstractVector

Compute the demeaned scaling function: h̃ = demean(exp(Z · δ)).
Z is raw (not demeaned). h(z_it) = exp(z_it'δ) is computed first, then demeaned.

Uses the GPU+ForwardDiff-safe broadcasting pattern: P(p[j]) .* (@view Z[:,j]).
"""
function compute_h_tilde(Z::AbstractMatrix{T}, p::AbstractVector{P},
                          idx, T_periods::Int) where {T<:AbstractFloat, P<:Real}
    L = length(idx.delta)
    # h = exp(Z · δ) using broadcasting pattern
    h = exp.(sum(P(p[idx.delta[j]]) .* (@view Z[:, j]) for j in 1:L))
    # Demean h within firms
    h_tilde = sf_panel_demean(h, T_periods)
    return h_tilde
end

"""
    sf_panel_demean(x::AbstractVector, offsets::Vector{Int}) -> AbstractVector

Within-group demeaning for unbalanced panel data using firm boundary offsets.
`offsets` has length N+1: firm i occupies positions `offsets[i]+1 : offsets[i+1]`.
"""
function sf_panel_demean(x::AbstractVector, offsets::Vector{Int})
    N = length(offsets) - 1
    result = similar(x)
    for i in 1:N
        s, e = offsets[i] + 1, offsets[i + 1]
        seg = @view x[s:e]
        result[s:e] .= seg .- mean(seg)
    end
    return result
end

"""
    sf_panel_demean(X::AbstractMatrix, offsets::Vector{Int}) -> AbstractMatrix

Within-group demeaning for each column of a matrix, using firm boundary offsets.
"""
function sf_panel_demean(X::AbstractMatrix, offsets::Vector{Int})
    result = similar(X)
    N = length(offsets) - 1
    for i in 1:N
        s, e = offsets[i] + 1, offsets[i + 1]
        for k in axes(X, 2)
            seg = @view X[s:e, k]
            result[s:e, k] .= seg .- mean(seg)
        end
    end
    return result
end

"""
    compute_h_tilde(Z, p, idx, offsets::Vector{Int}) -> AbstractVector

Compute demeaned scaling function for unbalanced panels using firm boundary offsets.
"""
function compute_h_tilde(Z::AbstractMatrix{T}, p::AbstractVector{P},
                          idx, offsets::Vector{Int}) where {T<:AbstractFloat, P<:Real}
    L = length(idx.delta)
    h = exp.(sum(P(p[idx.delta[j]]) .* (@view Z[:, j]) for j in 1:L))
    h_tilde = sf_panel_demean(h, offsets)
    return h_tilde
end

# Return both h_raw and h_tilde (needed by JLMS/BC which uses h_raw for obs-level expansion)
function compute_h_raw_and_tilde(Z::AbstractMatrix{T}, p::AbstractVector{P},
                                  idx, T_periods::Int) where {T<:AbstractFloat, P<:Real}
    L = length(idx.delta)
    h_raw = exp.(sum(P(p[idx.delta[j]]) .* (@view Z[:, j]) for j in 1:L))
    h_tilde = sf_panel_demean(h_raw, T_periods)
    return h_raw, h_tilde
end

function compute_h_raw_and_tilde(Z::AbstractMatrix{T}, p::AbstractVector{P},
                                  idx, offsets::Vector{Int}) where {T<:AbstractFloat, P<:Real}
    L = length(idx.delta)
    h_raw = exp.(sum(P(p[idx.delta[j]]) .* (@view Z[:, j]) for j in 1:L))
    h_tilde = sf_panel_demean(h_raw, offsets)
    return h_raw, h_tilde
end

"""
    _compute_panel_ABC(epsilon_tilde, h_tilde, offsets, N) -> (A, B, C)

Compute per-firm quadratic expansion sums using a per-firm loop with offsets.
Returns three N-vectors: A_i = Σ_t E_ti², B_i = Σ_t E_ti·H_ti, C_i = Σ_t H_ti².
Works on both CPU arrays and CuArray views (with per-firm kernel launches on GPU).
"""
function _compute_panel_ABC(epsilon_tilde::AbstractVector{P},
                             h_tilde::AbstractVector{P},
                             offsets::Vector{Int}, N::Int) where P
    # Collect to CPU if inputs are on GPU (avoid per-element scalar indexing)
    eps_cpu = epsilon_tilde isa Array ? epsilon_tilde : Array(epsilon_tilde)
    h_cpu   = h_tilde isa Array ? h_tilde : Array(h_tilde)

    A = Vector{P}(undef, N)
    B = Vector{P}(undef, N)
    C = Vector{P}(undef, N)
    for i in 1:N
        s, e = offsets[i] + 1, offsets[i + 1]
        E_i = @view eps_cpu[s:e]
        H_i = @view h_cpu[s:e]
        A[i] = sum(E_i .^ 2)
        B[i] = sum(E_i .* H_i)
        C[i] = sum(H_i .^ 2)
    end
    return A, B, C
end

"""
    _check_no_constant_column(mat, name)

Check that no column of the matrix is constant (all identical values).
A constant column is unidentifiable after within-group demeaning.
"""
function _check_no_constant_column(mat::AbstractMatrix, name::String)
    for j in axes(mat, 2)
        col = @view mat[:, j]
        if all(x -> x == col[1], col)
            error("`$name` column $j is constant (e.g., a vector of ones); " *
                  "it is unidentifiable in the Wang-Ho panel model " *
                  "because within-group demeaning eliminates it.")
        end
    end
end

"""
    _compute_panel_structure(id::AbstractVector) -> (N, T_vec, offsets)

Compute panel structure from an id column. Data must be grouped by id
(all rows for the same unit are contiguous). Returns number of units N,
per-unit period counts T_vec, and boundary offsets (length N+1).
"""
function _compute_panel_structure(id::AbstractVector)
    NT = length(id)
    NT > 0 || error("id column is empty.")

    # Find contiguous group boundaries
    group_starts = [1]
    for i in 2:NT
        if id[i] != id[i - 1]
            push!(group_starts, i)
        end
    end
    N = length(group_starts)
    T_vec = Vector{Int}(undef, N)
    for i in 1:N - 1
        T_vec[i] = group_starts[i + 1] - group_starts[i]
    end
    T_vec[N] = NT - group_starts[N] + 1
    offsets = vcat(0, cumsum(T_vec))

    # Validate: no non-contiguous groups (same id appearing in separate blocks)
    seen = Set{eltype(id)}()
    prev = id[1]
    push!(seen, prev)
    for i in 2:NT
        if id[i] != prev
            if id[i] in seen
                error("id column is not grouped: '$(id[i])' appears in non-contiguous rows. " *
                      "Sort your data by id before calling this function.")
            end
            push!(seen, id[i])
            prev = id[i]
        end
    end

    return N, T_vec, offsets
end

"""
    _to_vector_panel(x) -> Vector

Normalize input to a Vector (handles Vector, [Vector], N×1 Matrix).
"""
function _to_vector_panel(x)
    if x isa AbstractVector
        return x
    elseif x isa AbstractMatrix
        size(x, 2) == 1 || error("Matrix input for depvar must have exactly 1 column, got $(size(x, 2)).")
        return vec(x)
    elseif x isa AbstractVector{<:AbstractVector}
        length(x) == 1 || error("List input for depvar must have exactly 1 element.")
        return x[1]
    else
        error("Unsupported depvar input type: $(typeof(x))")
    end
end

"""
    _to_matrix_panel(x) -> Matrix

Normalize input to a Matrix (handles Matrix, Vector → N×1, [v1,v2,...] → hcat).
"""
function _to_matrix_panel(x)
    if x isa AbstractMatrix
        return x
    elseif x isa AbstractVector{<:Real}
        return reshape(x, :, 1)
    elseif x isa AbstractVector{<:AbstractVector}
        return hcat(x...)
    else
        error("Unsupported matrix input type: $(typeof(x))")
    end
end


# ============================================================================
# Section 5: Pre-computed Constants
# ============================================================================

"""
    make_panel_constants(model::PanelModel, ::Type{TT}, T_max::Int)

Create pre-computed constants for numerical stability and efficiency.
"""
function make_panel_constants(model::PanelModel, ::Type{TT}, T_max::Int) where {TT<:AbstractFloat}
    sqrt2 = sqrt(TT(2))
    return (
        clamp_lo      = TT(1e-15),
        sigma_floor   = TT(1e-12),
        sigma_ceil    = TT(1e12),
        k_floor       = TT(1e-6),
        k_ceil        = TT(1e6),
        theta_floor   = TT(1e-12),
        theta_ceil    = TT(1e12),
        xm_floor      = TT(1e-12),
        xm_ceil       = TT(1e6),
        alpha_floor   = TT(1e-6),
        alpha_ceil    = TT(1e6),
        sqrt2         = sqrt2,
        inv_sqrt2     = inv(sqrt2),
        sqrt_pi       = sqrt(TT(π)),
        lo_erf        = TT(32) * eps(TT),
        hi_erf        = one(TT) - TT(32) * eps(TT),
        exp_clamp     = TT(1e-15),
    )
end


# ============================================================================
# Section 6: Parameter Value Extraction
# ============================================================================

"""
    get_panel_noise_vals(::NormalNoise_Panel, p, idx, c) -> NamedTuple

Extract and transform noise parameters from the coefficient vector.
"""
function get_panel_noise_vals(::NormalNoise_Panel, p, idx, c)
    P = eltype(p)
    sigma_v_sq = clamp(exp(p[idx.ln_sigma_v_sq]), P(c.sigma_floor)^2, P(c.sigma_ceil)^2)
    sigma_v = sqrt(sigma_v_sq)
    inv_sigma_v_sq = inv(sigma_v_sq)
    log_sigma_v_sq = log(sigma_v_sq)
    return (sigma_v=sigma_v, sigma_v_sq=sigma_v_sq, inv_sigma_v_sq=inv_sigma_v_sq,
            log_sigma_v_sq=log_sigma_v_sq)
end

"""
    get_panel_ineff_vals(::HalfNormal_Panel, p, idx, c) -> NamedTuple

Extract and transform inefficiency parameters from the coefficient vector.
"""
function get_panel_ineff_vals(::HalfNormal_Panel, p, idx, c)
    P = eltype(p)
    sigma_u_sq = clamp(exp(p[idx.ln_sigma_u_sq]), P(c.sigma_floor)^2, P(c.sigma_ceil)^2)
    sigma_u = sqrt(sigma_u_sq)
    return (sigma_u=sigma_u, sigma_u_sq=sigma_u_sq)
end

function get_panel_ineff_vals(::TruncatedNormal_Panel, p, idx, c)
    P = eltype(p)
    mu = p[idx.mu]
    sigma_u_sq = clamp(exp(p[idx.ln_sigma_u_sq]), P(c.sigma_floor)^2, P(c.sigma_ceil)^2)
    sigma_u = sqrt(sigma_u_sq)
    return (mu=mu, sigma_u=sigma_u, sigma_u_sq=sigma_u_sq)
end

function get_panel_ineff_vals(::Exponential_Panel, p, idx, c)
    P = eltype(p)
    lambda = clamp(exp(p[idx.ln_lambda]), P(c.sigma_floor), P(c.sigma_ceil))
    return (lambda=lambda,)
end

function get_panel_ineff_vals(::Weibull_Panel, p, idx, c)
    P = eltype(p)
    lambda = clamp(exp(p[idx.ln_lambda]), P(c.sigma_floor), P(c.sigma_ceil))
    k = clamp(exp(p[idx.ln_k]), P(c.k_floor), P(c.k_ceil))
    return (lambda=lambda, k=k)
end

function get_panel_ineff_vals(::Lognormal_Panel, p, idx, c)
    P = eltype(p)
    mu = p[idx.mu]
    sigma_sq = clamp(exp(p[idx.ln_sigma_sq]), P(c.sigma_floor)^2, P(c.sigma_ceil)^2)
    sigma = sqrt(sigma_sq)
    return (mu=mu, sigma=sigma, sigma_sq=sigma_sq)
end

function get_panel_ineff_vals(::Lomax_Panel, p, idx, c)
    P = eltype(p)
    lambda = clamp(exp(p[idx.ln_lambda]), P(c.xm_floor), P(c.xm_ceil))
    alpha = clamp(exp(p[idx.ln_alpha]), P(c.alpha_floor), P(c.alpha_ceil))
    return (lambda=lambda, alpha=alpha)
end

function get_panel_ineff_vals(::Rayleigh_Panel, p, idx, c)
    P = eltype(p)
    sigma_sq = clamp(exp(p[idx.ln_sigma_sq]), P(c.sigma_floor)^2, P(c.sigma_ceil)^2)
    sigma = sqrt(sigma_sq)
    return (sigma=sigma, sigma_sq=sigma_sq)
end

function get_panel_ineff_vals(::Gamma_Panel, p, idx, c)
    P = eltype(p)
    k = clamp(exp(p[idx.ln_k]), P(c.k_floor), P(c.k_ceil))
    theta = clamp(exp(p[idx.ln_theta]), P(c.theta_floor), P(c.theta_ceil))
    lgk = loggamma(k)
    return (k=k, theta=theta, lgk=lgk)
end


# ============================================================================
# Section 7: MCI Transformation Rules
# ============================================================================

# --- Transformation functions ---
# g(t, σ_u) maps Halton draws t ∈ (0,1) to u ∈ (0,∞).
# Each rule defines: u = g(t, s) and J = |g'(t, s)| where s = σ_u.

# logistic_1_rule: u = σ_u · t/(1-t). Like simple but scaled by σ_u.
@inline logistic_1_panel_trans(t, s) = s * t / (1 - t)
@inline logistic_1_panel_jacob(t, s) = s / (1 - t)^2

# logistic_2_rule: u = σ_u · (t/(1-t))². Heavier tail, concentrates draws near 0.
@inline logistic_2_panel_trans(t, s) = s * (t / (1 - t))^2
@inline logistic_2_panel_jacob(t, s) = 2 * s * t / (1 - t)^3

# expo_rule: u = σ_u · (-log(1-t)). Exponential-like mapping.
@inline expo_panel_trans(t, s) = s * (-log(1 - t))
@inline expo_panel_jacob(t, s) = s / (1 - t)

const PANEL_TRANSFORMATIONS = Dict{Symbol, Tuple{Function, Function}}(
    :logistic_1_rule => (logistic_1_panel_trans, logistic_1_panel_jacob),
    :logistic_2_rule => (logistic_2_panel_trans, logistic_2_panel_jacob),
    :expo_rule       => (expo_panel_trans, expo_panel_jacob),
)

"""Default transformation for each inefficiency distribution."""
default_panel_transformation(::HalfNormal_Panel)      = :logistic_1_rule
default_panel_transformation(::TruncatedNormal_Panel)  = :logistic_1_rule
default_panel_transformation(::Exponential_Panel)      = :expo_rule
default_panel_transformation(::Weibull_Panel)          = :expo_rule
default_panel_transformation(::Lognormal_Panel)        = :logistic_1_rule
default_panel_transformation(::Lomax_Panel)            = :logistic_1_rule
default_panel_transformation(::Rayleigh_Panel)         = :expo_rule
default_panel_transformation(::Gamma_Panel)            = :expo_rule

"""
    resolve_panel_transformation(rule, model) -> (trans_func, jacob_func)

Resolve transformation rule to concrete functions. Uses distribution default if nothing.
"""
function resolve_panel_transformation(rule::Union{Symbol,Nothing}, model::PanelModel)
    r = isnothing(rule) ? default_panel_transformation(model.ineff) : rule
    if !haskey(PANEL_TRANSFORMATIONS, r)
        error("Unknown panel transformation: :$r. Valid options: $(collect(keys(PANEL_TRANSFORMATIONS)))")
    end
    return PANEL_TRANSFORMATIONS[r]
end

"""
    log_pdf_halfnormal_panel(u, sigma, clamp_lo)

Element-wise log-PDF of the half-normal distribution N⁺(0, σ²).
f(u) = (2/(σ√(2π))) exp(-u²/(2σ²)) for u ≥ 0.
"""
@inline function log_pdf_halfnormal_panel(u, sigma, clamp_lo)
    sigma_safe = max(sigma, clamp_lo)
    return log(2 / (sigma_safe * sqrt(2π))) - (u / sigma_safe)^2 / 2
end

"""Log-PDF of truncated normal distribution TN(μ, σ²; lower=0)."""
@inline function log_pdf_truncnormal_panel(u, mu, sigma, clamp_lo)
    sigma_safe = max(sigma, clamp_lo)
    z = (u - mu) / sigma_safe
    log_Phi = log(max(oftype(z, 0.5) * (1 + erf(mu / (sigma_safe * sqrt(oftype(z, 2))))), clamp_lo))
    return oftype(z, -0.5) * log(oftype(z, 2π)) - log(sigma_safe) - oftype(z, 0.5) * z^2 - log_Phi
end

"""Log-PDF of Exponential distribution, λ = Var(u). f(u) = (1/√λ)·exp(-u/√λ)."""
@inline function log_pdf_exponential_panel(u, lambda, clamp_lo)
    lambda_safe = max(lambda, clamp_lo)
    return -0.5 * log(lambda_safe) - u / sqrt(lambda_safe)
end

"""Log-PDF of Weibull distribution Weibull(λ, k). f(u) = (k/λ)(u/λ)^{k-1} exp(-(u/λ)^k)."""
@inline function log_pdf_weibull_panel(u, lambda, k, clamp_lo)
    u_safe = max(u, clamp_lo)
    lambda_safe = max(lambda, clamp_lo)
    return log(k) - k * log(lambda_safe) + (k - 1) * log(u_safe) - (u_safe / lambda_safe)^k
end

"""Log-PDF of lognormal distribution LogNormal(μ, σ)."""
@inline function log_pdf_lognormal_panel(u, mu, sigma, clamp_lo)
    u_safe = max(u, clamp_lo)
    sigma_safe = max(sigma, clamp_lo)
    log_u = log(u_safe)
    return oftype(mu, -0.5) * log(oftype(mu, 2π)) - log(sigma_safe) - log_u -
           oftype(mu, 0.5) * ((log_u - mu) / sigma_safe)^2
end

"""Log-PDF of Lomax distribution Lomax(α, λ). f(u) = α/λ · (1 + u/λ)^{-(α+1)} for u ≥ 0."""
@inline function log_pdf_lomax_panel(u, lambda, alpha, clamp_lo)
    u_safe = max(u, clamp_lo)
    alpha_safe = max(alpha, clamp_lo)
    lambda_safe = max(lambda, clamp_lo)
    return log(alpha_safe) - log(lambda_safe) - (alpha_safe + 1) * log1p(u_safe / lambda_safe)
end

"""Log-PDF of Rayleigh distribution Rayleigh(σ). f(u) = (u/σ²)exp(-u²/(2σ²))."""
@inline function log_pdf_rayleigh_panel(u, sigma, clamp_lo)
    u_safe = max(u, clamp_lo)
    sigma_safe = max(sigma, clamp_lo)
    return log(u_safe) - 2 * log(sigma_safe) - oftype(sigma, 0.5) * (u_safe / sigma_safe)^2
end

"""Log-PDF of Gamma distribution Gamma(k, θ). f(u) = u^{k-1} exp(-u/θ) / (θ^k Γ(k))."""
@inline function log_pdf_gamma_panel(u, k, theta, lgk, clamp_lo)
    u_safe = max(u, clamp_lo)
    theta_safe = max(theta, clamp_lo)
    return (k - 1) * log(u_safe) - u_safe / theta_safe - k * log(theta_safe) - lgk
end


# ============================================================================
# Section 7b: Distribution Dispatch Helpers for NLL
# ============================================================================

# --- MSLE quantile generation (inverse CDF) ---

"""Generate u draws via inverse CDF for MSLE. Dispatched by inefficiency distribution."""
function mypan_gen_udraws_msle(::HalfNormal_Panel, ineff_vals, halton_P, c, ::Type{P}) where P
    sigma_u = ineff_vals.sigma_u
    return sigma_u .* P(c.sqrt2) .* erfinv.(clamp.(halton_P, P(c.lo_erf), P(c.hi_erf)))
end

function mypan_gen_udraws_msle(::TruncatedNormal_Panel, ineff_vals, halton_P, c, ::Type{P}) where P
    mu = ineff_vals.mu
    sigma_u = ineff_vals.sigma_u
    s2 = P(c.sqrt2)
    erf_arg = erf(mu / (sigma_u * s2))
    inner = halton_P .+ (halton_P .- 1) .* erf_arg
    inner_clamped = clamp.(inner, P(c.lo_erf), P(c.hi_erf))
    return mu .+ sigma_u .* s2 .* erfinv.(inner_clamped)
end

function mypan_gen_udraws_msle(::Exponential_Panel, ineff_vals, halton_P, c, ::Type{P}) where P
    lambda = ineff_vals.lambda
    return .-log.(max.(1 .- halton_P, P(c.exp_clamp))) .* sqrt(lambda)
end

function mypan_gen_udraws_msle(::Weibull_Panel, ineff_vals, halton_P, c, ::Type{P}) where P
    lambda = ineff_vals.lambda
    k = ineff_vals.k
    return lambda .* (.-log.(max.(1 .- halton_P, P(c.exp_clamp)))) .^ (1 / k)
end

function mypan_gen_udraws_msle(::Lognormal_Panel, ineff_vals, halton_P, c, ::Type{P}) where P
    mu = ineff_vals.mu
    sigma = ineff_vals.sigma
    inner = clamp.(2 .* halton_P .- 1, P(c.lo_erf), P(c.hi_erf))
    return exp.(mu .+ sigma .* P(c.sqrt2) .* erfinv.(inner))
end

function mypan_gen_udraws_msle(::Lomax_Panel, ineff_vals, halton_P, c, ::Type{P}) where P
    lambda = ineff_vals.lambda
    alpha = ineff_vals.alpha
    return lambda .* expm1.((-1 / alpha) .* log.(max.(1 .- halton_P, P(c.exp_clamp))))
end

function mypan_gen_udraws_msle(::Rayleigh_Panel, ineff_vals, halton_P, c, ::Type{P}) where P
    sigma = ineff_vals.sigma
    return sigma .* sqrt.(-2 .* log.(max.(1 .- halton_P, P(c.exp_clamp))))
end

function mypan_gen_udraws_msle(::Gamma_Panel, args...)
    error("`ineff=:Gamma` does not have a closed-form inverse CDF. Use `method=:MCI` instead.")
end

# --- MCI scale parameter extraction ---

"""Extract the scale parameter for MCI transformation. Dispatched by inefficiency distribution."""
mypan_get_mci_scale(::HalfNormal_Panel, ineff_vals)      = ineff_vals.sigma_u
mypan_get_mci_scale(::TruncatedNormal_Panel, ineff_vals)  = ineff_vals.sigma_u
mypan_get_mci_scale(::Exponential_Panel, ineff_vals)      = sqrt(ineff_vals.lambda)
mypan_get_mci_scale(::Weibull_Panel, ineff_vals)          = ineff_vals.lambda
mypan_get_mci_scale(::Lognormal_Panel, ineff_vals)        = ineff_vals.sigma
mypan_get_mci_scale(::Lomax_Panel, ineff_vals)            = ineff_vals.lambda
mypan_get_mci_scale(::Rayleigh_Panel, ineff_vals)         = ineff_vals.sigma
mypan_get_mci_scale(::Gamma_Panel, ineff_vals)            = ineff_vals.theta

# --- MCI log-PDF evaluation ---

"""Evaluate log-PDF of inefficiency distribution over D-vector of u_draws. Dispatched by distribution."""
function mypan_logpdf_ineff(::HalfNormal_Panel, u_draws, ineff_vals, clamp_lo)
    return log_pdf_halfnormal_panel.(u_draws, ineff_vals.sigma_u, clamp_lo)
end

function mypan_logpdf_ineff(::TruncatedNormal_Panel, u_draws, ineff_vals, clamp_lo)
    return log_pdf_truncnormal_panel.(u_draws, ineff_vals.mu, ineff_vals.sigma_u, clamp_lo)
end

function mypan_logpdf_ineff(::Exponential_Panel, u_draws, ineff_vals, clamp_lo)
    return log_pdf_exponential_panel.(u_draws, ineff_vals.lambda, clamp_lo)
end

function mypan_logpdf_ineff(::Weibull_Panel, u_draws, ineff_vals, clamp_lo)
    return log_pdf_weibull_panel.(u_draws, ineff_vals.lambda, ineff_vals.k, clamp_lo)
end

function mypan_logpdf_ineff(::Lognormal_Panel, u_draws, ineff_vals, clamp_lo)
    return log_pdf_lognormal_panel.(u_draws, ineff_vals.mu, ineff_vals.sigma, clamp_lo)
end

function mypan_logpdf_ineff(::Lomax_Panel, u_draws, ineff_vals, clamp_lo)
    return log_pdf_lomax_panel.(u_draws, ineff_vals.lambda, ineff_vals.alpha, clamp_lo)
end

function mypan_logpdf_ineff(::Rayleigh_Panel, u_draws, ineff_vals, clamp_lo)
    return log_pdf_rayleigh_panel.(u_draws, ineff_vals.sigma, clamp_lo)
end

function mypan_logpdf_ineff(::Gamma_Panel, u_draws, ineff_vals, clamp_lo)
    return log_pdf_gamma_panel.(u_draws, ineff_vals.k, ineff_vals.theta, ineff_vals.lgk, clamp_lo)
end


# ============================================================================
# Section 8: Halton Sequence Generation
# ============================================================================

"""
    make_panel_halton(D::Int; base::Int=2, T::Type=Float64) -> Vector{T}

Generate Halton quasi-random sequence of length D for panel MC integration.
"""
function make_panel_halton(D::Int; base::Int=2, T::Type{<:AbstractFloat}=Float64)
    D > 0 || throw(ArgumentError("Number of draws must be positive, got $D"))
    return T.(collect(Halton(base, length=D)))
end

"""
    make_panel_halton_wrap(N::Int, D::Int; base::Int=2, T::Type=Float64,
                           distinct_Halton_length::Int=2^15-1) -> Matrix{T}

Generate a wrapped Halton quasi-random sequence matrix for multiRand mode.

Returns an N × D matrix where each firm gets different consecutive
Halton sequence elements, providing greater variation across firms.

# Arguments
- `N::Int`: Number of firms
- `D::Int`: Number of draws per firm (must be ≤ `distinct_Halton_length`)
- `base::Int=2`: Base for Halton sequence generation
- `T::Type{<:AbstractFloat}=Float64`: Element type
- `distinct_Halton_length::Int=2^15-1`: Maximum Halton sequence length

# Returns
- `Matrix{T}` of size (N, D) with Halton values in (0,1)
"""
function make_panel_halton_wrap(N::Int, D::Int; base::Int=2, T::Type{<:AbstractFloat}=Float64, distinct_Halton_length::Int=2^15-1)
    N > 0 || throw(ArgumentError("N must be positive, got $N"))
    D > 0 || throw(ArgumentError("D (draws) must be positive, got $D"))

    if D > distinct_Halton_length
        throw(ArgumentError("`n_draws` (=$D, which is per firm draws) is too large for multiRand=true. " *
                           "Three options. (1) Reduce it to ≤ $distinct_Halton_length . " *
                           "(2) Set `multiRand=false` and each firm will use the same Halton draws. " *
                           "(3) Increase `distinct_Halton_length` to be larger than your `n_draws`."))
    end

    # Compute sequence length: find largest n where 2^n - 1 <= total_needed, capped
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
        n_full_repeats = total_needed ÷ seq_len
        remainder = total_needed % seq_len
        wrapped = vcat(repeat(myH, n_full_repeats), myH[1:remainder])
    end

    # Reshape to N × D matrix (Julia is column-major)
    return copy(reshape(wrapped, D, N)')
end


# ============================================================================
# Section 9: Numerical Stability Helpers
# ============================================================================

"""
    _maximum_panel(A; dims) -> result

Maximum operation with GPU dispatch.
"""
_maximum_panel(A::AbstractArray; dims) = maximum(A; dims=dims)

"""
    _sum_panel(A; dims) -> result

Sum operation with GPU dispatch.
"""
_sum_panel(A::AbstractArray; dims) = sum(A; dims=dims)

"""
    _sum_scalar_panel(v) -> scalar

Scalar sum with GPU dispatch.
"""
_sum_scalar_panel(v::AbstractArray) = sum(v)

# GPU overloads (conditional on CUDA availability)
if isdefined(Main, :CUDA)
    _maximum_panel(A::Main.CUDA.AnyCuArray; dims) = Main.CUDA.maximum(A; dims=dims)
    _sum_panel(A::Main.CUDA.AnyCuArray; dims)     = Main.CUDA.sum(A; dims=dims)
    _sum_scalar_panel(v::Main.CUDA.AnyCuArray)     = sum(v)
    _to_device_panel(x::AbstractArray, ::Main.CUDA.AnyCuArray) = Main.CUDA.CuArray(x)
    _to_device_panel(x::Main.CUDA.AnyCuArray, ::Main.CUDA.AnyCuArray) = x  # already on GPU
end
_to_device_panel(x::AbstractArray, ::AbstractArray) = x  # CPU ref: no-op

# Pull to CPU (no-op when already CPU)
_to_cpu(x::Array) = x
_to_cpu(x::AbstractArray) = Array(x)

"""
    logsumexp_rows_panel(A::AbstractMatrix, clamp_lo=1e-300) -> Vector

Log-sum-exp across columns for each row. Numerically stable.
Returns a vector of length size(A, 1).
"""
function logsumexp_rows_panel(A::AbstractMatrix, clamp_lo=1e-300)
    max_vals = _maximum_panel(A; dims=2)
    sum_exp  = _sum_panel(exp.(A .- max_vals); dims=2)
    fmin = oftype(zero(eltype(A)), clamp_lo)
    @. sum_exp = max(sum_exp, fmin)
    return vec(max_vals .+ log.(sum_exp))
end


# ============================================================================
# Section 10: Negative Log-Likelihood Functions
# ============================================================================

"""
    panel_nll(spec::_PanelInternalSpec, p; chunks) -> scalar

Main NLL dispatcher. Calls MSLE or MCI path based on spec.method.
"""
function panel_nll(spec::_PanelInternalSpec{T}, p::AbstractVector{P};
                   chunks::Int=4) where {T<:AbstractFloat, P<:Real}
    if spec.method == :MSLE
        return _panel_nll_msle(spec, p; chunks=chunks)
    else
        return _panel_nll_mci(spec, p; chunks=chunks)
    end
end

"""
    _panel_nll_msle(spec, p; chunks) -> scalar

Negative log-likelihood for the panel model using MSLE (inverse CDF) draws.
Draw generation is dispatched by inefficiency distribution via `mypan_gen_udraws_msle`.

Log-likelihood per firm: log L_i = logsumexp_d[log f_v(ε_i + h̃_i · u_d)] - log(D)
"""
function _panel_nll_msle(spec::_PanelInternalSpec{T}, p::AbstractVector{P};
                          chunks::Int=4) where {T<:AbstractFloat, P<:Real}

    N, K = spec.N, spec.K
    c = spec.constants
    idx = spec.idx
    model = spec.model
    halton = spec.draws
    D = ndims(halton) == 1 ? length(halton) : size(halton, 2)

    # 1. Compute residuals: ε̃ = ỹ - X̃β
    epsilon_tilde = spec.y_tilde .- sum(P(p[idx.beta[j]]) .* (@view spec.x_tilde[:, j]) for j in 1:K)

    # 2. Compute h̃ = demean(exp(Z · δ))  and  A, B, C per-firm sums
    if spec.T_periods isa Int
        # Balanced: vectorized reshape path
        h_tilde = compute_h_tilde(spec.z_raw, p, idx, spec.T_max)
        E_all = reshape(epsilon_tilde, spec.T_max, N)
        H_all = reshape(h_tilde, spec.T_max, N)
        A = vec(_sum_panel(E_all .^ 2; dims=1))
        B = vec(_sum_panel(E_all .* H_all; dims=1))
        C = vec(_sum_panel(H_all .^ 2; dims=1))
    else
        # Unbalanced: per-firm loop with offsets
        h_tilde = compute_h_tilde(spec.z_raw, p, idx, spec.offsets)
        A, B, C = _compute_panel_ABC(epsilon_tilde, h_tilde, spec.offsets, N)
        # Move A, B, C to GPU if data lives there (halton is CuArray when GPU=true)
        A = _to_device_panel(A, halton)
        B = _to_device_panel(B, halton)
        C = _to_device_panel(C, halton)
    end

    # 3. Extract noise/inefficiency parameters
    noise_vals = get_panel_noise_vals(model.noise, p, idx, c)
    ineff_vals = get_panel_ineff_vals(model.ineff, p, idx, c)

    # 4. Generate u draws via inverse CDF (dispatched by inefficiency distribution)
    halton_P = P.(halton)
    u_draws = mypan_gen_udraws_msle(model.ineff, ineff_vals, halton_P, c, P)

    # 5. Pre-compute log-density constants
    inv_sigma_v_sq = noise_vals.inv_sigma_v_sq
    log_sigma_v_sq = noise_vals.log_sigma_v_sq
    log_const = P(-0.5) .* spec.Tm1 .* (log(P(2π)) .+ log_sigma_v_sq)

    frontier_sign = P(spec.sign)

    # Quadratic expansion: sum_t (E_ti + s*H_ti*u_d)^2 = A_i + 2s*B_i*u_d + C_i*u_d^2
    u_row = ndims(u_draws) == 1 ? reshape(u_draws, 1, D) : u_draws   # 1×D or N×D
    two_s = P(2) * frontier_sign

    if chunks <= 1
        # --- Non-chunked: all firms at once ---
        quad_2D = A .+ two_s .* B .* u_row .+ C .* u_row .^ 2   # N × D

        # Log-density per firm per draw
        log_fv = P(-0.5) .* quad_2D .* inv_sigma_v_sq .+ log_const

        # Aggregate: log L_i = logsumexp_d(log_fv) - log(D)
        log_likes = logsumexp_rows_panel(log_fv) .- log(P(D))
        return -_sum_scalar_panel(log_likes)
    else
        # --- Chunked: process firms in chunks ---
        firm_chunk_size = cld(N, chunks)
        total_nll = P(0)

        for ci in 1:chunks
            f_start = (ci - 1) * firm_chunk_size + 1
            f_end = min(ci * firm_chunk_size, N)
            f_start > f_end && continue

            A_c = @view A[f_start:f_end]
            B_c = @view B[f_start:f_end]
            C_c = @view C[f_start:f_end]
            u_c = ndims(u_row) == 2 && size(u_row, 1) > 1 ? (@view u_row[f_start:f_end, :]) : u_row

            quad_2D = A_c .+ two_s .* B_c .* u_c .+ C_c .* u_c .^ 2

            lc = log_const isa AbstractVector ? (@view log_const[f_start:f_end]) : log_const
            log_fv = P(-0.5) .* quad_2D .* inv_sigma_v_sq .+ lc

            log_likes = logsumexp_rows_panel(log_fv) .- log(P(D))
            total_nll -= _sum_scalar_panel(log_likes)
        end
        return total_nll
    end
end

"""
    _panel_nll_mci(spec, p; chunks) -> scalar

Negative log-likelihood for the panel model using MCI (change-of-variable) draws.

Draw generation, log-PDF, and scale parameter dispatched by inefficiency distribution.

Log-likelihood per firm:
  log L_i = logsumexp_d[log f_v(ε_i + h̃_i·g(t_d)) + log f_u(g(t_d)) + log|J(t_d)|] - log(D)
"""
function _panel_nll_mci(spec::_PanelInternalSpec{T}, p::AbstractVector{P};
                         chunks::Int=4) where {T<:AbstractFloat, P<:Real}

    N, K = spec.N, spec.K
    c = spec.constants
    idx = spec.idx
    model = spec.model
    halton = spec.draws
    D = ndims(halton) == 1 ? length(halton) : size(halton, 2)

    # 1. Compute residuals: ε̃ = ỹ - X̃β
    epsilon_tilde = spec.y_tilde .- sum(P(p[idx.beta[j]]) .* (@view spec.x_tilde[:, j]) for j in 1:K)

    # 2. Compute h̃ = demean(exp(Z · δ))  and  A, B, C per-firm sums
    if spec.T_periods isa Int
        # Balanced: vectorized reshape path
        h_tilde = compute_h_tilde(spec.z_raw, p, idx, spec.T_max)
        E_all = reshape(epsilon_tilde, spec.T_max, N)
        H_all = reshape(h_tilde, spec.T_max, N)
        A = vec(_sum_panel(E_all .^ 2; dims=1))
        B = vec(_sum_panel(E_all .* H_all; dims=1))
        C = vec(_sum_panel(H_all .^ 2; dims=1))
    else
        # Unbalanced: per-firm loop with offsets
        h_tilde = compute_h_tilde(spec.z_raw, p, idx, spec.offsets)
        A, B, C = _compute_panel_ABC(epsilon_tilde, h_tilde, spec.offsets, N)
        # Move A, B, C to GPU if data lives there (halton is CuArray when GPU=true)
        A = _to_device_panel(A, halton)
        B = _to_device_panel(B, halton)
        C = _to_device_panel(C, halton)
    end

    # 3. Extract noise/inefficiency parameters
    noise_vals = get_panel_noise_vals(model.noise, p, idx, c)
    ineff_vals = get_panel_ineff_vals(model.ineff, p, idx, c)

    # 4. MCI transformation: u = g(t, scale), Jacobian |g'(t, scale)|
    scale = mypan_get_mci_scale(model.ineff, ineff_vals)
    trans, jacob = resolve_panel_transformation(spec.transformation, model)

    halton_P = P.(halton)
    u_draws = trans.(halton_P, scale)
    u_draws = min.(u_draws, inv(P(c.clamp_lo)))                  # clamp upper

    # 5. Log-PDF of inefficiency + log-Jacobian
    log_fu = mypan_logpdf_ineff(model.ineff, u_draws, ineff_vals, P(c.clamp_lo))
    log_J  = log.(max.(jacob.(halton_P, scale), P(c.clamp_lo)))
    log_fu_J = log_fu .+ log_J

    # 6. Pre-compute log-density constants
    inv_sigma_v_sq = noise_vals.inv_sigma_v_sq
    log_sigma_v_sq = noise_vals.log_sigma_v_sq
    log_const = P(-0.5) .* spec.Tm1 .* (log(P(2π)) .+ log_sigma_v_sq)

    frontier_sign = P(spec.sign)

    # Reshape for broadcasting: 1×D (shared) or N×D (multiRand)
    log_fu_J_row = ndims(log_fu_J) == 1 ? reshape(log_fu_J, 1, D) : log_fu_J
    u_row = ndims(u_draws) == 1 ? reshape(u_draws, 1, D) : u_draws
    two_s = P(2) * frontier_sign

    if chunks <= 1
        # --- Non-chunked: all firms at once ---
        quad_2D = A .+ two_s .* B .* u_row .+ C .* u_row .^ 2   # N × D

        # Log-density per firm per draw
        log_fv = P(-0.5) .* quad_2D .* inv_sigma_v_sq .+ log_const

        # Add log f_u + log |J|
        log_lik = log_fv .+ log_fu_J_row

        # Aggregate: log L_i = logsumexp_d(log_lik) - log(D)
        log_likes = logsumexp_rows_panel(log_lik) .- log(P(D))
        return -_sum_scalar_panel(log_likes)
    else
        # --- Chunked: process firms in chunks ---
        firm_chunk_size = cld(N, chunks)
        total_nll = P(0)

        for ci in 1:chunks
            f_start = (ci - 1) * firm_chunk_size + 1
            f_end = min(ci * firm_chunk_size, N)
            f_start > f_end && continue

            A_c = @view A[f_start:f_end]
            B_c = @view B[f_start:f_end]
            C_c = @view C[f_start:f_end]
            u_c = ndims(u_row) == 2 && size(u_row, 1) > 1 ? (@view u_row[f_start:f_end, :]) : u_row
            lfj_c = ndims(log_fu_J_row) == 2 && size(log_fu_J_row, 1) > 1 ? (@view log_fu_J_row[f_start:f_end, :]) : log_fu_J_row

            quad_2D = A_c .+ two_s .* B_c .* u_c .+ C_c .* u_c .^ 2

            lc = log_const isa AbstractVector ? (@view log_const[f_start:f_end]) : log_const
            log_fv = P(-0.5) .* quad_2D .* inv_sigma_v_sq .+ lc

            log_lik = log_fv .+ lfj_c

            log_likes = logsumexp_rows_panel(log_lik) .- log(P(D))
            total_nll -= _sum_scalar_panel(log_likes)
        end
        return total_nll
    end
end


# ============================================================================
# Section 11: JLMS/BC Efficiency Indices
# ============================================================================

# Expand firm-level E[u_i*|data_i] to observation-level JLMS = h_raw_it * E[u_i*].
function _expand_jlms_obs(E_u_firm::AbstractVector{P}, h_raw::AbstractVector,
                           spec::_PanelInternalSpec) where P
    NT = length(h_raw)
    N  = spec.N
    jlms = Vector{P}(undef, NT)

    if spec.T_periods isa Int
        Tp = spec.T_max
        E_u_2d   = repeat(reshape(E_u_firm, 1, N), Tp, 1)       # T × N
        h_raw_2d = reshape(h_raw, Tp, N)                         # T × N
        jlms .= vec(h_raw_2d .* E_u_2d)
    else
        for i in 1:N
            s, e = spec.offsets[i] + 1, spec.offsets[i + 1]
            jlms[s:e] .= h_raw[s:e] .* E_u_firm[i]
        end
    end
    return jlms
end

# Compute observation-level BC = E[exp(-h_it * u_i*) | data_i].
function _compute_bc_obs(log_w::AbstractMatrix{P}, log_denom::AbstractVector{P},
                          h_raw::AbstractVector, u_draws::AbstractVecOrMat,
                          spec::_PanelInternalSpec) where P
    NT = length(h_raw)
    N  = spec.N
    D  = ndims(u_draws) == 1 ? length(u_draws) : size(u_draws, 2)
    bc = Vector{P}(undef, NT)

    if spec.T_periods isa Int
        Tp = spec.T_max
        h_raw_2d = reshape(h_raw, Tp, N)                         # T × N
        h_3d     = reshape(h_raw_2d, Tp, N, 1)

        if ndims(u_draws) == 1
            u_3d = reshape(u_draws, 1, 1, D)                     # 1 × 1 × D (shared)
        else
            # multiRand: u_draws is N × D → 1 × N × D (firm-specific draws)
            u_3d = reshape(u_draws, 1, N, D)
        end
        log_w_3d = reshape(log_w, 1, N, D)

        log_bc_w = .-h_3d .* u_3d .+ log_w_3d                    # T × N × D

        log_bc_num_flat = logsumexp_rows_panel(reshape(log_bc_w, Tp * N, D))
        log_bc_num = reshape(log_bc_num_flat, Tp, N)              # T × N

        bc_TN = exp.(log_bc_num .- reshape(log_denom, 1, N))     # T × N
        bc .= vec(bc_TN)
    else
        for i in 1:N
            s, e = spec.offsets[i] + 1, spec.offsets[i + 1]
            Ti = e - s + 1
            h_firm = reshape(h_raw[s:e], Ti, 1)                  # Ti × 1

            if ndims(u_draws) == 1
                u_firm = reshape(u_draws, 1, D)                   # 1 × D (shared)
            else
                u_firm = reshape(u_draws[i, :], 1, D)            # 1 × D (firm i's draws)
            end

            log_w_firm = reshape(log_w[i, :], 1, D)              # 1 × D
            log_bc_w_firm = .-h_firm .* u_firm .+ log_w_firm     # Ti × D
            log_bc_num = logsumexp_rows_panel(log_bc_w_firm)      # Ti
            bc[s:e] .= exp.(log_bc_num .- log_denom[i])
        end
    end
    return bc
end

"""
    panel_jlms_bc(spec::_PanelInternalSpec, p; chunks) -> NamedTuple

Compute observation-level JLMS inefficiency and BC efficiency indices.

Since u_it = h_it * u_i*, where h_it = exp(z_it'δ) is deterministic:
  JLMS: E[u_it | data_i] = h_it * E[u_i* | data_i]
  BC:   E[exp(-u_it) | data_i] = E[exp(-h_it * u_i*) | data_i]

Both are NT-length vectors (one value per observation).
h_it is the raw (un-demeaned) scaling function.

Operates entirely on CPU (one-shot post-estimation call).
The `chunks` keyword is accepted for API compatibility but unused.
"""
function panel_jlms_bc(spec::_PanelInternalSpec{T}, p::AbstractVector{P};
                        chunks::Int=1) where {T<:AbstractFloat, P<:Real}

    N, K = spec.N, spec.K
    c    = spec.constants
    idx  = spec.idx
    model = spec.model

    # Pull inputs to CPU (no-op when already CPU)
    y_tilde = _to_cpu(spec.y_tilde)
    x_tilde = _to_cpu(spec.x_tilde)
    z_raw   = _to_cpu(spec.z_raw)
    halton  = _to_cpu(spec.draws)
    D       = ndims(halton) == 1 ? length(halton) : size(halton, 2)

    # 1. Residuals
    epsilon_tilde = y_tilde .- sum(P(p[idx.beta[j]]) .* (@view x_tilde[:, j]) for j in 1:K)

    # 2. h_raw and h_tilde (need both: h_tilde for quadratic form, h_raw for obs expansion)
    h_raw, h_tilde = compute_h_raw_and_tilde(z_raw, p, idx,
                         spec.T_periods isa Int ? spec.T_max : spec.offsets)

    # 3. Per-firm quadratic sums A, B, C
    if spec.T_periods isa Int
        E_all = reshape(epsilon_tilde, spec.T_max, N)
        H_all = reshape(h_tilde, spec.T_max, N)
        A = vec(sum(E_all .^ 2; dims=1))
        B = vec(sum(E_all .* H_all; dims=1))
        C = vec(sum(H_all .^ 2; dims=1))
    else
        A, B, C = _compute_panel_ABC(epsilon_tilde, h_tilde, spec.offsets, N)
    end

    # 4. Noise/inefficiency parameters and u draws (dispatched by distribution)
    noise_vals = get_panel_noise_vals(model.noise, p, idx, c)
    ineff_vals = get_panel_ineff_vals(model.ineff, p, idx, c)
    halton_P   = P.(halton)

    if spec.method == :MSLE
        u_draws   = mypan_gen_udraws_msle(model.ineff, ineff_vals, halton_P, c, P)
        log_extra = nothing
    else
        scale = mypan_get_mci_scale(model.ineff, ineff_vals)
        trans, jacob = resolve_panel_transformation(spec.transformation, model)
        u_draws   = min.(trans.(halton_P, scale), inv(P(c.clamp_lo)))
        log_fu    = mypan_logpdf_ineff(model.ineff, u_draws, ineff_vals, P(c.clamp_lo))
        log_J     = log.(max.(jacob.(halton_P, scale), P(c.clamp_lo)))
        log_fu_J  = log_fu .+ log_J
        log_extra = ndims(log_fu_J) == 1 ? reshape(log_fu_J, 1, D) : log_fu_J
    end

    # 5. Log-weights (N × D) and denominator
    inv_sigma_v_sq = noise_vals.inv_sigma_v_sq
    log_sigma_v_sq = noise_vals.log_sigma_v_sq
    Tm1_cpu        = spec.Tm1 isa AbstractArray ? _to_cpu(spec.Tm1) : spec.Tm1
    log_const      = P(-0.5) .* Tm1_cpu .* (log(P(2π)) .+ log_sigma_v_sq)
    frontier_sign  = P(spec.sign)

    u_row   = ndims(u_draws) == 1 ? reshape(u_draws, 1, D) : u_draws  # 1×D or N×D
    two_s   = P(2) * frontier_sign
    quad_2D = A .+ two_s .* B .* u_row .+ C .* u_row .^ 2       # N × D

    log_w = P(-0.5) .* quad_2D .* inv_sigma_v_sq .+ log_const
    if log_extra !== nothing
        log_w = log_w .+ log_extra
    end

    log_denom = logsumexp_rows_panel(log_w)                       # N

    # 6. JLMS: firm-level E[u_i*], then expand to obs-level via h_raw
    log_u_row    = log.(max.(u_row, P(c.clamp_lo)))               # 1 × D (precomputed)
    log_u_w      = log_u_row .+ log_w                             # N × D
    log_jlms_num = logsumexp_rows_panel(log_u_w)                  # N
    E_u_firm     = exp.(log_jlms_num .- log_denom)                # N

    jlms = _expand_jlms_obs(E_u_firm, h_raw, spec)

    # 7. BC: obs-level E[exp(-h_it * u_i*)]
    bc = _compute_bc_obs(log_w, log_denom, h_raw, u_draws, spec)

    return (jlms=jlms, bc=bc)
end


# ============================================================================
# Section 11a: Marginal Effects of Inefficiency Determinants
# ============================================================================

"""
    _panel_uncondU_half(coef, idx, z_obs)

Unconditional mean E(u_it) = h(z_it) * σ_u * √(2/π) for a single observation
under HalfNormal inefficiency. ForwardDiff-compatible: `z_obs` carries dual
numbers; `coef` and `idx` are fixed.
"""
function _panel_uncondU_half(coef, idx, z_obs)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    ln_sigma_u_sq = coef[idx.ln_sigma_u_sq]
    sigma_u = clamp(exp(oftype(z_delta, 0.5) * ln_sigma_u_sq),
                    oftype(z_delta, 1e-12), oftype(z_delta, 1e12))
    return h_val * sigma_u * sqrt(oftype(z_delta, 2) / oftype(z_delta, π))
end

"""E(u_it) for TruncatedNormal: h(z) * σ_u * (Λ + φ(Λ)/Φ(Λ)), Λ = μ/σ_u"""
function _panel_uncondU_trun(coef, idx, z_obs)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    mu = coef[idx.mu]
    sigma_u = clamp(exp(oftype(z_delta, 0.5) * coef[idx.ln_sigma_u_sq]),
                    oftype(z_delta, 1e-12), oftype(z_delta, 1e12))
    Lambda = mu / sigma_u
    phi_Lambda = exp(oftype(z_delta, -0.5) * Lambda^2) / sqrt(oftype(z_delta, 2π))
    Phi_Lambda = max(oftype(z_delta, 0.5) * (1 + erf(Lambda / sqrt(oftype(z_delta, 2)))),
                     oftype(z_delta, 1e-15))
    return h_val * sigma_u * (Lambda + phi_Lambda / Phi_Lambda)
end

"""E(u_it) for Exponential (λ=Var(u*)): h(z) * √λ"""
function _panel_uncondU_expo(coef, idx, z_obs)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    lambda = clamp(exp(coef[idx.ln_lambda]),
                   oftype(z_delta, 1e-12), oftype(z_delta, 1e12))
    return h_val * sqrt(lambda)
end

"""E(u_it) for Weibull(λ, k): h(z) * λ * Γ(1 + 1/k)"""
function _panel_uncondU_weibull(coef, idx, z_obs)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    lambda = clamp(exp(coef[idx.ln_lambda]),
                   oftype(z_delta, 1e-12), oftype(z_delta, 1e12))
    k_val = clamp(exp(coef[idx.ln_k]),
                  oftype(z_delta, 1e-6), oftype(z_delta, 1e6))
    return h_val * lambda * exp(loggamma(1 + 1 / k_val))
end

"""E(u_it) for Lognormal(μ, σ): h(z) * exp(μ + σ²/2)"""
function _panel_uncondU_lognorm(coef, idx, z_obs)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    mu = coef[idx.mu]
    sigma_sq = clamp(exp(coef[idx.ln_sigma_sq]),
                     oftype(z_delta, 1e-24), oftype(z_delta, 1e24))
    return h_val * exp(mu + sigma_sq / 2)
end

"""E(u_it) for Lomax(α, λ): h(z) * λ / (α - 1). Requires α > 1."""
function _panel_uncondU_lomax(coef, idx, z_obs)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    lambda = clamp(exp(coef[idx.ln_lambda]),
                   oftype(z_delta, 1e-12), oftype(z_delta, 1e6))
    alpha = clamp(exp(coef[idx.ln_alpha]),
                  oftype(z_delta, 1e-6), oftype(z_delta, 1e6))
    if alpha <= 1
        return oftype(z_delta, Inf)
    end
    return h_val * lambda / (alpha - 1)
end

"""E(u_it) for Rayleigh(σ): h(z) * σ * √(π/2)"""
function _panel_uncondU_rayleigh(coef, idx, z_obs)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    sigma = clamp(exp(oftype(z_delta, 0.5) * coef[idx.ln_sigma_sq]),
                  oftype(z_delta, 1e-12), oftype(z_delta, 1e12))
    return h_val * sigma * sqrt(oftype(z_delta, π) / 2)
end

"""E(u_it) for Gamma(k, θ): h(z) * k * θ"""
function _panel_uncondU_gamma(coef, idx, z_obs)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    k_val = clamp(exp(coef[idx.ln_k]),
                  oftype(z_delta, 1e-6), oftype(z_delta, 1e6))
    theta = clamp(exp(coef[idx.ln_theta]),
                  oftype(z_delta, 1e-12), oftype(z_delta, 1e12))
    return h_val * k_val * theta
end

# --- Post-processing helpers for marginal effects ---

_panel_is_constant_column(col::AbstractVector) =
    length(col) == 0 || all(x -> x ≈ first(col), col)

function _panel_nonConsDataFrame(df::DataFrame, Z::AbstractMatrix)
    cols_to_keep = String[]
    for (i, col_name) in enumerate(df_names(df))
        if i <= size(Z, 2) && !_panel_is_constant_column(@view Z[:, i])
            push!(cols_to_keep, col_name)
        end
    end
    return isempty(cols_to_keep) ? DataFrame() : df[:, cols_to_keep]
end

function _panel_compute_marg_mean(margeff::DataFrame)
    isempty(margeff) && return NamedTuple()
    col_means = [round(mean(col); digits=5) for col in eachcol(margeff)]
    return (; zip(Symbol.(df_names(margeff)), col_means)...)
end

function _panel_add_marg_prefix(margeff::DataFrame)
    isempty(margeff) && return margeff
    new_names = Symbol.("marg_" .* df_names(margeff))
    return rename(margeff, new_names)
end

function _get_panel_z_varnames(spec::_PanelInternalSpec)
    z_names = spec.varnames[spec.idx.delta]
    return length(z_names) == spec.L ? z_names : ["z$j" for j in 1:spec.L]
end

# --- Dispatched marginal effects computation ---

"""
    _panel_marg_generic(uncondU_func, spec, coef)

Generic marginal effects computation via ForwardDiff.gradient of `uncondU_func`.
"""
function _panel_marg_generic(uncondU_func::Function, spec::_PanelInternalSpec{T},
                               coef::AbstractVector) where T
    Z = _to_cpu(spec.z_raw)
    coef_cpu = _to_cpu(coef)
    idx = spec.idx
    NT = length(spec.y_tilde)
    L = spec.L

    mm = Matrix{Float64}(undef, L, NT)
    @inbounds for i in 1:NT
        Zi = collect(@view Z[i, :])
        mm[:, i] = gradient(z -> uncondU_func(coef_cpu, idx, z), Zi)
    end

    z_names = _get_panel_z_varnames(spec)
    margeff = DataFrame(mm', z_names)
    margeff = _panel_nonConsDataFrame(margeff, Z)
    margMean = _panel_compute_marg_mean(margeff)
    margeff = _panel_add_marg_prefix(margeff)
    return margeff, margMean
end

get_panel_marg(::HalfNormal_Panel, spec::_PanelInternalSpec, coef::AbstractVector) =
    _panel_marg_generic(_panel_uncondU_half, spec, coef)
get_panel_marg(::TruncatedNormal_Panel, spec::_PanelInternalSpec, coef::AbstractVector) =
    _panel_marg_generic(_panel_uncondU_trun, spec, coef)
get_panel_marg(::Exponential_Panel, spec::_PanelInternalSpec, coef::AbstractVector) =
    _panel_marg_generic(_panel_uncondU_expo, spec, coef)
get_panel_marg(::Weibull_Panel, spec::_PanelInternalSpec, coef::AbstractVector) =
    _panel_marg_generic(_panel_uncondU_weibull, spec, coef)
get_panel_marg(::Lognormal_Panel, spec::_PanelInternalSpec, coef::AbstractVector) =
    _panel_marg_generic(_panel_uncondU_lognorm, spec, coef)
get_panel_marg(::Lomax_Panel, spec::_PanelInternalSpec, coef::AbstractVector) =
    _panel_marg_generic(_panel_uncondU_lomax, spec, coef)
get_panel_marg(::Rayleigh_Panel, spec::_PanelInternalSpec, coef::AbstractVector) =
    _panel_marg_generic(_panel_uncondU_rayleigh, spec, coef)
get_panel_marg(::Gamma_Panel, spec::_PanelInternalSpec, coef::AbstractVector) =
    _panel_marg_generic(_panel_uncondU_gamma, spec, coef)

"""
    panel_marginal_effects(spec::_PanelInternalSpec, coef) -> (DataFrame, NamedTuple)

Compute marginal effects of Z variables on E(u_it). Dispatches on the model's
inefficiency distribution type.

Returns `(margeff, margMean)`:
- `margeff::DataFrame`: per-observation marginal effects, columns prefixed "marg_*"
- `margMean::NamedTuple`: Average Marginal Effects (AME)
"""
function panel_marginal_effects(spec::_PanelInternalSpec, coef::AbstractVector)
    return get_panel_marg(spec.model.ineff, spec, coef)
end


# ============================================================================
# Section 12: Variance-Covariance Matrix & Hessian
# ============================================================================

"""
    _panel_invert_hessian(H; message=true) -> NamedTuple

Invert Hessian matrix for variance-covariance computation.
Returns (var_cov_matrix=..., redflag=Int).
"""
function _panel_invert_hessian(H::AbstractMatrix{TT};
                                message::Bool=true) where {TT<:Real}
    redflag = 0
    var_cov_matrix = similar(H)

    try
        var_cov_matrix = inv(H)
    catch err
        redflag = 1
        if message
            printstyled("The Hessian matrix is not invertible, indicating the model does not converge properly.\n"; color=:red)
        end
        return (var_cov_matrix=var_cov_matrix, redflag=redflag)
    end

    if !all(diag(var_cov_matrix) .> 0)
        redflag = 2
        if message
            printstyled("Some diagonal elements of the var-cov matrix are non-positive, indicating convergence problems.\n"; color=:red)
        end
    end

    return (var_cov_matrix=var_cov_matrix, redflag=redflag)
end

"""
    panel_var_cov_mat(nll_func, coef; message=true) -> NamedTuple

Compute variance-covariance matrix from NLL function using ForwardDiff Hessian.
"""
function panel_var_cov_mat(nll_func::Function, coef::AbstractVector{TT};
                            message::Bool=true) where {TT<:Real}
    H = hessian(nll_func, coef)
    return _panel_invert_hessian(H; message=message)
end


# ============================================================================
# Section 13: Print Table
# ============================================================================

"""Build auxiliary table rows showing transformed parameters for each distribution."""
function _panel_aux_table(::HalfNormal_Panel, coef, idx, stddev)
    aux = Matrix{Any}(undef, 2, 3)
    aux[1, :] = ["σᵤ²", exp(coef[idx.ln_sigma_u_sq]), exp(coef[idx.ln_sigma_u_sq]) * stddev[idx.ln_sigma_u_sq]]
    aux[2, :] = ["σᵥ²", exp(coef[idx.ln_sigma_v_sq]), exp(coef[idx.ln_sigma_v_sq]) * stddev[idx.ln_sigma_v_sq]]
    return aux
end

function _panel_aux_table(::TruncatedNormal_Panel, coef, idx, stddev)
    aux = Matrix{Any}(undef, 2, 3)
    aux[1, :] = ["σᵤ²", exp(coef[idx.ln_sigma_u_sq]), exp(coef[idx.ln_sigma_u_sq]) * stddev[idx.ln_sigma_u_sq]]
    aux[2, :] = ["σᵥ²", exp(coef[idx.ln_sigma_v_sq]), exp(coef[idx.ln_sigma_v_sq]) * stddev[idx.ln_sigma_v_sq]]
    return aux
end

function _panel_aux_table(::Exponential_Panel, coef, idx, stddev)
    aux = Matrix{Any}(undef, 2, 3)
    aux[1, :] = ["λ", exp(coef[idx.ln_lambda]), exp(coef[idx.ln_lambda]) * stddev[idx.ln_lambda]]
    aux[2, :] = ["σᵥ²", exp(coef[idx.ln_sigma_v_sq]), exp(coef[idx.ln_sigma_v_sq]) * stddev[idx.ln_sigma_v_sq]]
    return aux
end

function _panel_aux_table(::Weibull_Panel, coef, idx, stddev)
    aux = Matrix{Any}(undef, 3, 3)
    aux[1, :] = ["λ", exp(coef[idx.ln_lambda]), exp(coef[idx.ln_lambda]) * stddev[idx.ln_lambda]]
    aux[2, :] = ["k", exp(coef[idx.ln_k]), exp(coef[idx.ln_k]) * stddev[idx.ln_k]]
    aux[3, :] = ["σᵥ²", exp(coef[idx.ln_sigma_v_sq]), exp(coef[idx.ln_sigma_v_sq]) * stddev[idx.ln_sigma_v_sq]]
    return aux
end

function _panel_aux_table(::Lognormal_Panel, coef, idx, stddev)
    aux = Matrix{Any}(undef, 2, 3)
    aux[1, :] = ["σ²", exp(coef[idx.ln_sigma_sq]), exp(coef[idx.ln_sigma_sq]) * stddev[idx.ln_sigma_sq]]
    aux[2, :] = ["σᵥ²", exp(coef[idx.ln_sigma_v_sq]), exp(coef[idx.ln_sigma_v_sq]) * stddev[idx.ln_sigma_v_sq]]
    return aux
end

function _panel_aux_table(::Lomax_Panel, coef, idx, stddev)
    aux = Matrix{Any}(undef, 3, 3)
    aux[1, :] = ["λ", exp(coef[idx.ln_lambda]), exp(coef[idx.ln_lambda]) * stddev[idx.ln_lambda]]
    aux[2, :] = ["α", exp(coef[idx.ln_alpha]), exp(coef[idx.ln_alpha]) * stddev[idx.ln_alpha]]
    aux[3, :] = ["σᵥ²", exp(coef[idx.ln_sigma_v_sq]), exp(coef[idx.ln_sigma_v_sq]) * stddev[idx.ln_sigma_v_sq]]
    return aux
end

function _panel_aux_table(::Rayleigh_Panel, coef, idx, stddev)
    aux = Matrix{Any}(undef, 2, 3)
    aux[1, :] = ["σ²", exp(coef[idx.ln_sigma_sq]), exp(coef[idx.ln_sigma_sq]) * stddev[idx.ln_sigma_sq]]
    aux[2, :] = ["σᵥ²", exp(coef[idx.ln_sigma_v_sq]), exp(coef[idx.ln_sigma_v_sq]) * stddev[idx.ln_sigma_v_sq]]
    return aux
end

function _panel_aux_table(::Gamma_Panel, coef, idx, stddev)
    aux = Matrix{Any}(undef, 3, 3)
    aux[1, :] = ["k", exp(coef[idx.ln_k]), exp(coef[idx.ln_k]) * stddev[idx.ln_k]]
    aux[2, :] = ["θ", exp(coef[idx.ln_theta]), exp(coef[idx.ln_theta]) * stddev[idx.ln_theta]]
    aux[3, :] = ["σᵥ²", exp(coef[idx.ln_sigma_v_sq]), exp(coef[idx.ln_sigma_v_sq]) * stddev[idx.ln_sigma_v_sq]]
    return aux
end

"""
    panel_print_table(spec, coef, var_cov_matrix; optim_result=nothing, table_format=:text)

Print formatted estimation results table for panel SF model.
"""
function panel_print_table(spec::_PanelInternalSpec{T}, coef::AbstractVector,
                            var_cov_matrix::AbstractMatrix;
                            optim_result=nothing,
                            table_format::Symbol=:text) where {T}

    nofpara = length(coef)
    N_total = spec.T_periods isa Int ? spec.N * spec.T_periods : sum(spec.T_periods)

    # Compute statistics (asymptotic Normal, standard for MLE)
    stddev = sqrt.(abs.(diag(var_cov_matrix)))
    t_stats = coef ./ stddev
    tt = cquantile(Normal(0, 1), 0.025)  # ~1.96

    p_values = [2 * ccdf(Normal(0, 1), abs(t_stats[i])) for i in 1:nofpara]
    ci_low = coef .- tt .* stddev
    ci_upp = coef .+ tt .* stddev

    # Build equation column
    eq_col = fill("", nofpara)
    for (i, idx_val) in enumerate(spec.eq_indices)
        if idx_val <= nofpara
            eq_col[idx_val] = spec.eqnames[i]
        end
    end

    # Print header
    printstyled("\n*********************************\n"; color=:cyan)
    printstyled("  Panel SF: Wang and Ho (2010)\n"; color=:cyan)
    printstyled("*********************************\n"; color=:cyan)

    print("Method: "); printstyled(spec.method; color=:yellow); println()
    print("Model type: "); printstyled("noise=$(spec.noise), ineff=$(spec.ineff)"; color=:yellow); println()
    print("Number of firms (N): "); printstyled(spec.N; color=:yellow); println()
    if spec.T_periods isa Int
        print("Time periods (T): "); printstyled(spec.T_periods; color=:yellow); println()
    else
        T_min, T_max_val = extrema(spec.T_periods)
        print("Time periods (T): "); printstyled("$T_min-$T_max_val (unbalanced)"; color=:yellow); println()
    end
    print("Number of observations: "); printstyled(N_total; color=:yellow); println()
    print("Number of frontier regressors (K): "); printstyled(spec.K; color=:yellow); println()
    print("Number of Z columns (L): "); printstyled(spec.L; color=:yellow); println()
    _ndraws = ndims(spec.draws) == 1 ? length(spec.draws) : size(spec.draws, 2)
    print("Number of draws: "); printstyled(_ndraws; color=:yellow); println()
    print("Frontier type: "); printstyled(spec.sign == 1 ? "production" : "cost"; color=:yellow); println()
    print("GPU computing: "); printstyled(spec.GPU; color=:yellow); println()

    if optim_result !== nothing
        print("Number of iterations: "); printstyled(optim_result.iterations; color=:yellow); println()
        converged = Optim.converged(optim_result)
        print("Converged: "); printstyled(converged; color=converged ? :yellow : :red); println()
        print("Log-likelihood: "); printstyled(round(-optim_result.minimum; digits=5); color=:yellow); println()
    end
    println()

    # Main coefficient table
    table_data = hcat(eq_col, spec.varnames, coef, stddev, t_stats, p_values, ci_low, ci_upp)

    pretty_table(table_data;
                 column_labels=["", "Var.", "Coef.", "Std.Err.", "z", "P>|z|", "95%CI_l", "95%CI_u"],
                 formatters=[fmt__printf("%.4f", collect(3:8))],
                 compact_printing=true,
                 backend=table_format)
    println()

    # Auxiliary table: log-parameters converted to original scale
    idx = spec.idx
    println("Log-parameters converted to original scale:")

    aux_data = _panel_aux_table(spec.model.ineff, coef, idx, stddev)

    pretty_table(aux_data;
                 column_labels=["", "Coef.", "Std.Err."],
                 formatters=[fmt__printf("%.4f", [2, 3])],
                 compact_printing=true,
                 backend=table_format)
    println()

    return (table=table_data, aux_table=aux_data)
end


# ============================================================================
# Section 14: Name Generation
# ============================================================================

# Return distribution-specific variable name suffixes (after frontier and h(z) names).
_panel_ineff_varnames(::HalfNormal_Panel)      = ["ln_σᵤ²", "ln_σᵥ²"]
_panel_ineff_varnames(::TruncatedNormal_Panel)  = ["μ", "ln_σᵤ²", "ln_σᵥ²"]
_panel_ineff_varnames(::Exponential_Panel)      = ["ln_λ", "ln_σᵥ²"]
_panel_ineff_varnames(::Weibull_Panel)          = ["ln_λ", "ln_k", "ln_σᵥ²"]
_panel_ineff_varnames(::Lognormal_Panel)        = ["μ", "ln_σ²", "ln_σᵥ²"]
_panel_ineff_varnames(::Lomax_Panel)            = ["ln_λ", "ln_α", "ln_σᵥ²"]
_panel_ineff_varnames(::Rayleigh_Panel)         = ["ln_σ²", "ln_σᵥ²"]
_panel_ineff_varnames(::Gamma_Panel)            = ["ln_k", "ln_θ", "ln_σᵥ²"]

"""Return distribution-specific equation names and indices."""
function _panel_ineff_eqinfo(::HalfNormal_Panel, K, L)
    return ["ln_σᵤ²", "ln_σᵥ²"], [K + L + 1, K + L + 2]
end
function _panel_ineff_eqinfo(::TruncatedNormal_Panel, K, L)
    return ["μ", "ln_σᵤ²", "ln_σᵥ²"], [K + L + 1, K + L + 2, K + L + 3]
end
function _panel_ineff_eqinfo(::Exponential_Panel, K, L)
    return ["ln_λ", "ln_σᵥ²"], [K + L + 1, K + L + 2]
end
function _panel_ineff_eqinfo(::Weibull_Panel, K, L)
    return ["ln_λ", "ln_k", "ln_σᵥ²"], [K + L + 1, K + L + 2, K + L + 3]
end
function _panel_ineff_eqinfo(::Lognormal_Panel, K, L)
    return ["μ", "ln_σ²", "ln_σᵥ²"], [K + L + 1, K + L + 2, K + L + 3]
end
function _panel_ineff_eqinfo(::Lomax_Panel, K, L)
    return ["ln_λ", "ln_α", "ln_σᵥ²"], [K + L + 1, K + L + 2, K + L + 3]
end
function _panel_ineff_eqinfo(::Rayleigh_Panel, K, L)
    return ["ln_σ²", "ln_σᵥ²"], [K + L + 1, K + L + 2]
end
function _panel_ineff_eqinfo(::Gamma_Panel, K, L)
    return ["ln_k", "ln_θ", "ln_σᵥ²"], [K + L + 1, K + L + 2, K + L + 3]
end

"""
    _panel_gen_names(ineff_model, K, L, idx, varnames, eqnames, eq_indices)

Generate or validate variable names, equation names, and equation indices.
"""
function _panel_gen_names(ineff_model::PanelIneffModel, K::Int, L::Int, idx,
                           varnames::Union{Nothing, Vector{String}},
                           eqnames::Union{Nothing, Vector{String}},
                           eq_indices::Union{Nothing, Vector{Int}})

    n_params = panel_plen(ineff_model, K, L)

    ineff_vn = _panel_ineff_varnames(ineff_model)
    default_varnames = vcat(
        ["x$i" for i in 1:K],
        ["z$j" for j in 1:L],
        ineff_vn
    )

    ineff_en, ineff_ei = _panel_ineff_eqinfo(ineff_model, K, L)
    default_eqnames = vcat(["frontier", "h(z)"], ineff_en)
    default_eq_indices = vcat([1, K + 1], ineff_ei)

    vn = if isnothing(varnames)
        default_varnames
    else
        length(varnames) == n_params ||
            error("Length of varnames ($(length(varnames))) != number of parameters ($n_params)")
        varnames
    end

    en = isnothing(eqnames) ? default_eqnames : eqnames
    ei = isnothing(eq_indices) ? default_eq_indices : eq_indices

    return vn, en, ei
end

# Backward-compatible wrapper
_panel_gen_names(K::Int, L::Int, idx, varnames, eqnames, eq_indices) =
    _panel_gen_names(HalfNormal_Panel(), K, L, idx, varnames, eqnames, eq_indices)


# ============================================================================
# Section 15a: DSL Marker Types
# ============================================================================

"""Marker types for DSL-style model specification using DataFrames."""

if !@isdefined(UseDataSpec)
    struct UseDataSpec
        df::DataFrame
    end
end

if !@isdefined(DepvarSpec)
    struct DepvarSpec
        name::Symbol
    end
end

if !@isdefined(FrontierSpec)
    struct FrontierSpec
        names::Vector{Symbol}
    end
end

if !@isdefined(ZvarSpec)
    struct ZvarSpec
        names::Vector{Symbol}
    end
end

if !@isdefined(IdSpec)
    struct IdSpec
        name::Symbol
    end
end


# ============================================================================
# Section 15b: DSL Macros
# ============================================================================

"""
    @useData(df)

Specify the DataFrame to extract variables from.
"""
macro useData(df)
    :(UseDataSpec($(esc(df))))
end

"""
    @depvar(varname)

Specify the dependent variable by column name.
"""
macro depvar(var)
    :(DepvarSpec($(QuoteNode(var))))
end

"""
    @frontier(var1, var2, ...)

Specify frontier variables by column names.
"""
macro frontier(vars...)
    names = [QuoteNode(v) for v in vars]
    :(FrontierSpec(Symbol[$(names...)]))
end

"""
    @zvar(var1, var2, ...)

Specify scaling function h(z) variables by column names.
"""
macro zvar(vars...)
    names = [QuoteNode(v) for v in vars]
    :(ZvarSpec(Symbol[$(names...)]))
end

"""
    @id(varname)

Specify the panel unit identifier column by name (for unbalanced panels).
"""
macro id(var)
    :(IdSpec($(QuoteNode(var))))
end


# ============================================================================
# Section 15c: DSL Overloads of sfmodel_panel_spec
# ============================================================================

# --- Helper: extract DataFrame columns into arrays and varnames ---

function _panel_dsl_extract(df, depvar_spec::DepvarSpec, frontier_spec::FrontierSpec)
    depvar = Vector{Float64}(df[!, depvar_spec.name])
    frontier_names = [String(name) for name in frontier_spec.names]
    frontier = hcat([Vector{Float64}(df[!, name]) for name in frontier_spec.names]...)
    return depvar, frontier, frontier_names
end

function _panel_dsl_extract_zvar(df, zvar_spec::ZvarSpec)
    zvar_names = [String(name) for name in zvar_spec.names]
    zvar = hcat([Vector{Float64}(df[!, name]) for name in zvar_spec.names]...)
    return zvar, zvar_names
end


"""
    sfmodel_panel_spec(@useData(df), @depvar(y), @frontier(x1,x2), @zvar(z1,z2), @id(id); ...)

DSL form — unbalanced panel with @zvar and @id.
"""
function sfmodel_panel_spec(data_spec::UseDataSpec, depvar_spec::DepvarSpec,
                              frontier_spec::FrontierSpec, zvar_spec::ZvarSpec,
                              id_spec::IdSpec;
                              noise::Symbol=:Normal,
                              ineff::Symbol=:HalfNormal,
                              type::Symbol=:prod)
    df = data_spec.df
    depvar, frontier, frontier_names = _panel_dsl_extract(df, depvar_spec, frontier_spec)
    zvar, zvar_names = _panel_dsl_extract_zvar(df, zvar_spec)
    id_col = df[!, id_spec.name]
    model_temp = _build_panel_model(noise, ineff)
    varnames = vcat(frontier_names, zvar_names, _panel_ineff_varnames(model_temp.ineff))

    return sfmodel_panel_spec(; depvar=depvar, frontier=frontier, zvar=zvar,
                                 id=id_col, noise=noise, ineff=ineff, type=type,
                                 varnames=varnames)
end

"""
    sfmodel_panel_spec(@useData(df), @depvar(y), @frontier(x1,x2), @zvar(z1,z2); T_periods, ...)

DSL form — balanced panel with @zvar (no @id, requires T_periods keyword).
"""
function sfmodel_panel_spec(data_spec::UseDataSpec, depvar_spec::DepvarSpec,
                              frontier_spec::FrontierSpec, zvar_spec::ZvarSpec;
                              T_periods::Int,
                              noise::Symbol=:Normal,
                              ineff::Symbol=:HalfNormal,
                              type::Symbol=:prod)
    df = data_spec.df
    depvar, frontier, frontier_names = _panel_dsl_extract(df, depvar_spec, frontier_spec)
    zvar, zvar_names = _panel_dsl_extract_zvar(df, zvar_spec)
    model_temp = _build_panel_model(noise, ineff)
    varnames = vcat(frontier_names, zvar_names, _panel_ineff_varnames(model_temp.ineff))

    return sfmodel_panel_spec(; depvar=depvar, frontier=frontier, zvar=zvar,
                                 T_periods=T_periods, noise=noise, ineff=ineff,
                                 type=type, varnames=varnames)
end

"""
    sfmodel_panel_spec(@useData(df), @depvar(y), @frontier(x1,x2), @id(id); ...)

DSL form — unbalanced panel without @zvar (constant scaling h(z)=exp(δ₀)).
"""
function sfmodel_panel_spec(data_spec::UseDataSpec, depvar_spec::DepvarSpec,
                              frontier_spec::FrontierSpec, id_spec::IdSpec;
                              noise::Symbol=:Normal,
                              ineff::Symbol=:HalfNormal,
                              type::Symbol=:prod)
    df = data_spec.df
    depvar, frontier, frontier_names = _panel_dsl_extract(df, depvar_spec, frontier_spec)
    NT = length(depvar)
    zvar = ones(Float64, NT, 1)
    id_col = df[!, id_spec.name]
    model_temp = _build_panel_model(noise, ineff)
    varnames = vcat(frontier_names, ["_cons"], _panel_ineff_varnames(model_temp.ineff))

    return sfmodel_panel_spec(; depvar=depvar, frontier=frontier, zvar=zvar,
                                 id=id_col, noise=noise, ineff=ineff, type=type,
                                 varnames=varnames)
end

"""
    sfmodel_panel_spec(@useData(df), @depvar(y), @frontier(x1,x2); T_periods, ...)

DSL form — balanced panel without @zvar (constant scaling h(z)=exp(δ₀)).
"""
function sfmodel_panel_spec(data_spec::UseDataSpec, depvar_spec::DepvarSpec,
                              frontier_spec::FrontierSpec;
                              T_periods::Int,
                              noise::Symbol=:Normal,
                              ineff::Symbol=:HalfNormal,
                              type::Symbol=:prod)
    df = data_spec.df
    depvar, frontier, frontier_names = _panel_dsl_extract(df, depvar_spec, frontier_spec)
    NT = length(depvar)
    zvar = ones(Float64, NT, 1)
    model_temp = _build_panel_model(noise, ineff)
    varnames = vcat(frontier_names, ["_cons"], _panel_ineff_varnames(model_temp.ineff))

    return sfmodel_panel_spec(; depvar=depvar, frontier=frontier, zvar=zvar,
                                 T_periods=T_periods, noise=noise, ineff=ineff,
                                 type=type, varnames=varnames)
end


# ============================================================================
# Section 15: sfmodel_panel_spec()
# ============================================================================

"""
    sfmodel_panel_spec(; depvar, frontier, zvar, T_periods=nothing, id=nothing, ...)

Construct panel model specification. Supports both balanced and unbalanced panels.

# Panel structure — specify exactly one of:
- `T_periods::Int`: Balanced panel — all units have the same number of periods.
  Data must be stacked: unit 1 all T periods, unit 2 all T periods, ...
- `id`: Unbalanced panel — a column identifying the panel unit for each observation.
  Can be any type (integers, strings, symbols, etc.). Data must be grouped by id
  (all rows for the same unit are contiguous). The number of periods per unit is
  inferred from contiguous group lengths.

# Arguments
- `depvar`: Response vector, length NT_total
- `frontier`: Design matrix, NT_total × K (do NOT include a constant/intercept — it is unidentifiable after within-demeaning)
- `zvar`: Scaling function variables, NT_total × L
- `T_periods::Union{Int,Nothing}=nothing`: Periods per unit (balanced panel)
- `id=nothing`: Unit identifier column (unbalanced panel)
- `noise::Symbol=:Normal`: Noise distribution
- `ineff::Symbol=:HalfNormal`: Inefficiency distribution (:HalfNormal, :TruncatedNormal, :Exponential, :Weibull, :Lognormal, :Lomax, :Rayleigh, :Gamma)
- `type::Symbol=:prod`: Frontier type (`:prod`, `:production`, or `:cost`)
- `varnames=nothing`: Variable names (auto-generated if not provided)
- `eqnames=nothing`: Equation names (auto-generated if not provided)
- `eq_indices=nothing`: Equation boundary indices (auto-generated if not provided)

# Returns
`PanelModelSpec{T}` model specification object.

# Examples
```julia
# Balanced panel:
spec = sfmodel_panel_spec(
    depvar = y, frontier = X, zvar = Z,
    T_periods = 10
)

# Unbalanced panel:
spec = sfmodel_panel_spec(
    depvar = df.y, frontier = hcat(df.x1, df.x2),
    zvar = hcat(df.z1, df.z2),
    id = df.firm_name
)
```
"""
function sfmodel_panel_spec(; depvar, frontier, zvar,
                              T_periods::Union{Int,Nothing}=nothing,
                              id=nothing,
                              noise::Symbol=:Normal,
                              ineff::Symbol=:HalfNormal,
                              type::Symbol=:prod,
                              varnames::Union{Nothing, Vector{String}}=nothing,
                              eqnames::Union{Nothing, Vector{String}}=nothing,
                              eq_indices::Union{Nothing, Vector{Int}}=nothing)

    # Normalize inputs
    depvar_norm = _to_vector_panel(depvar)
    frontier_norm = _to_matrix_panel(frontier)
    zvar_norm = _to_matrix_panel(zvar)

    T_el = eltype(depvar_norm)
    frontier_norm = T_el.(frontier_norm)
    zvar_norm = T_el.(zvar_norm)

    NT = length(depvar_norm)

    # --- Determine panel structure ---
    if !isnothing(id) && !isnothing(T_periods)
        error("Specify exactly one of `T_periods` (balanced) or `id` (unbalanced), not both.")
    elseif isnothing(id) && isnothing(T_periods)
        error("Must specify either `T_periods::Int` (balanced panel) or `id` column (unbalanced panel).")
    end

    if !isnothing(T_periods)
        # --- Balanced panel ---
        @assert NT % T_periods == 0 "Length of depvar ($NT) must be divisible by T_periods ($T_periods)."
        @assert T_periods >= 2 "Wang-Ho within-transformation requires T_periods >= 2, got $T_periods."
        N = NT ÷ T_periods
        T_max = T_periods
        offsets = collect(0:T_periods:NT)
    else
        # --- Unbalanced panel (id column provided) ---
        id_vec = id isa AbstractVector ? id : error("`id` must be a vector, got $(typeof(id)).")
        length(id_vec) == NT || error("id column has $(length(id_vec)) elements, expected $NT (same as depvar).")
        N, T_vec, offsets = _compute_panel_structure(id_vec)
        @assert all(t -> t >= 2, T_vec) "Wang-Ho within-transformation requires T_i >= 2 for all units. " *
            "Got minimum T_i = $(minimum(T_vec))."
        T_periods = T_vec
        T_max = maximum(T_vec)
    end

    K = size(frontier_norm, 2)
    L = size(zvar_norm, 2)

    # Validate dimensions
    size(frontier_norm, 1) == NT || error("frontier has $(size(frontier_norm, 1)) rows, expected $NT.")
    size(zvar_norm, 1) == NT || error("zvar has $(size(zvar_norm, 1)) rows, expected $NT.")

    # Reject constant columns (unidentifiable after within-demeaning)
    _check_no_constant_column(frontier_norm, "frontier")
    if L > 1 || !all(==(1.0), zvar_norm)   # exempt DSL-generated constant-only zvar
        _check_no_constant_column(zvar_norm, "zvar")
    end

    # Build model
    model = _build_panel_model(noise, ineff)

    # Frontier sign
    frontier_sign = if type in (:prod, :production)
        1
    elseif type == :cost
        -1
    else
        error("Invalid type: :$type. Use :prod, :production, or :cost.")
    end

    # Parameter indices (dispatched by inefficiency distribution)
    idx = _panel_param_ind(model.ineff, K, L)

    # Generate names (dispatched by inefficiency distribution)
    varnames_vec, eqnames_vec, eq_indices_vec = _panel_gen_names(model.ineff, K, L, idx, varnames, eqnames, eq_indices)

    return PanelModelSpec{T_el}(depvar_norm, frontier_norm, zvar_norm,
                                 N, T_periods, T_max, offsets,
                                 noise, ineff, model,
                                 K, L, idx, frontier_sign,
                                 varnames_vec, eqnames_vec, eq_indices_vec)
end


# ============================================================================
# Section 16: sfmodel_panel_method()
# ============================================================================

"""
    sfmodel_panel_method(; method=:MSLE, transformation=nothing,
                           draws=nothing, n_draws=1024, multiRand=true,
                           GPU=false, chunks=10, distinct_Halton_length=2^15-1)

Specify the numerical estimation method for panel SF model.

# Arguments
- `method::Symbol=:MSLE`: Estimation method (`:MSLE` or `:MCI`)
- `transformation=nothing`: MCI transformation rule. Ignored for MSLE. Options:
  - `:logistic_1_rule` — `u = s · t/(1-t)` (default for HalfNormal, TruncatedNormal, Lognormal, Lomax)
  - `:logistic_2_rule` — `u = σ_u · (t/(1-t))²`
  - `:expo_rule` — `u = s · (-log(1-t))` (default for Exponential, Weibull, Rayleigh, Gamma)
  - `nothing` — use distribution-specific default
- `draws=nothing`: User-supplied Halton draws. Auto-generated if `nothing`.
- `n_draws::Int=1024`: Number of quasi-random draws
- `multiRand::Bool=true`: Per-firm draws (`true`: N×D matrix) or shared draws (`false`: 1D vector).
  When `true`, `n_draws` must be ≤ `distinct_Halton_length`.
- `GPU::Bool=false`: Use GPU acceleration
- `chunks::Int=10`: Number of chunks for memory management
- `distinct_Halton_length::Int=2^15-1`: Maximum Halton sequence length for multiRand mode.
  Increase if `n_draws` exceeds this limit.
"""
function sfmodel_panel_method(;
    method::Symbol=:MSLE,
    transformation::Union{Symbol,Nothing}=nothing,
    draws::Union{Nothing, AbstractVecOrMat{<:AbstractFloat}}=nothing,
    n_draws::Int=1024,
    multiRand::Bool=true,
    GPU::Bool=false,
    chunks::Int=10,
    distinct_Halton_length::Int=2^15-1)

    method in (:MSLE, :MCI) || error("method must be :MSLE or :MCI, got :$method")
    if method == :MSLE && !isnothing(transformation)
        @warn "`transformation` is only used with method=:MCI. Ignored for :MSLE."
    end

    return PanelMethodSpec(method, transformation, draws, n_draws, multiRand, GPU, chunks, distinct_Halton_length)
end


# ============================================================================
# Section 17: Assembly (_assemble_panel_spec)
# ============================================================================

"""
    _assemble_panel_spec(spec::PanelModelSpec, method::PanelMethodSpec) -> _PanelInternalSpec

Assemble internal spec from model specification and method specification.
Performs within-transformation (demeaning), GPU conversion, and Halton draw generation.
"""
function _assemble_panel_spec(spec::PanelModelSpec{T}, method::PanelMethodSpec) where {T}

    # Validate: Gamma is MCI-only
    if spec.ineff == :Gamma && method.method == :MSLE
        error("`ineff=:Gamma` is not supported with `method=:MSLE` (no closed-form inverse CDF). " *
              "Use `method=:MCI` instead.")
    end

    # 1. Demean y and X (within-transformation)
    if spec.T_periods isa Int
        y_tilde = sf_panel_demean(spec.depvar, spec.T_periods)
        x_tilde = sf_panel_demean(spec.frontier, spec.T_periods)
    else
        y_tilde = sf_panel_demean(spec.depvar, spec.offsets)
        x_tilde = sf_panel_demean(spec.frontier, spec.offsets)
    end
    # Z is NOT demeaned — h(z)=exp(z'δ) is computed then demeaned inside the NLL

    # 2. GPU conversion (optional)
    if method.GPU
        if !isdefined(Main, :CUDA)
            error("GPU=true requires CUDA.jl to be loaded. Please run `using CUDA` before calling this function.")
        end
        y_tilde = Main.CUDA.CuArray(y_tilde)
        x_tilde = Main.CUDA.CuArray(x_tilde)
        z_raw = Main.CUDA.CuArray(spec.zvar)
    else
        z_raw = spec.zvar
    end

    # 3. Generate or use provided Halton draws
    if method.draws !== nothing
        halton_raw = method.multiRand ? Matrix{T}(method.draws) : vec(T.(method.draws))
    elseif method.multiRand
        halton_raw = make_panel_halton_wrap(spec.N, method.n_draws;
                         T=T, distinct_Halton_length=method.distinct_Halton_length)
    else
        halton_raw = make_panel_halton(method.n_draws; T=T)
    end
    draws = _to_device_panel(halton_raw, y_tilde)

    # 4. Constants
    constants = make_panel_constants(spec.model, T, spec.T_max)

    # 5. Compute Tm1 (T_i - 1): scalar for balanced, N-vector for unbalanced
    if spec.T_periods isa Int
        Tm1 = spec.T_max - 1
    else
        Tm1_cpu = T.(spec.T_periods .- 1)
        Tm1 = method.GPU ? Main.CUDA.CuArray(Tm1_cpu) : Tm1_cpu
    end

    return _PanelInternalSpec{T}(
        y_tilde, x_tilde, z_raw,
        spec.N, spec.T_periods, spec.T_max, spec.offsets, Tm1,
        draws,
        spec.model, spec.noise, spec.ineff,
        spec.K, spec.L, spec.idx, spec.sign,
        method.chunks, constants,
        spec.varnames, spec.eqnames, spec.eq_indices,
        method.method, method.transformation, method.GPU
    )
end


# ============================================================================
# Section 18: sfmodel_panel_init()
# ============================================================================

"""
    sfmodel_panel_init(; spec::PanelModelSpec, init=nothing, frontier=nothing, delta=nothing,
                         ln_sigma_u_sq=nothing, ln_sigma_v_sq=nothing,
                         mu=nothing, ln_lambda=nothing, ln_k=nothing,
                         ln_sigma_sq=nothing, ln_alpha=nothing, ln_theta=nothing)

Create initial parameter vector for panel estimation.

# Arguments
- `spec::PanelModelSpec`: Model specification from `sfmodel_panel_spec()`
- `init=nothing`: Complete initial-value vector (overrides all component args)
- `frontier=nothing`: Initial values for frontier coefficients (default: OLS on demeaned data)
- `delta=nothing`: Initial values for h(z) coefficients (default: 0.1)
- Distribution-specific keyword arguments (used when matching the distribution):
  - `ln_sigma_u_sq`: HalfNormal, TruncatedNormal
  - `mu`: TruncatedNormal, Lognormal
  - `ln_lambda`: Exponential, Weibull, Lomax
  - `ln_k`: Weibull, Gamma
  - `ln_sigma_sq`: Lognormal, Rayleigh
  - `ln_alpha`: Lomax
  - `ln_theta`: Gamma
  - `ln_sigma_v_sq`: all distributions

# Returns
`Vector{Float64}` of initial parameter values.
"""
function sfmodel_panel_init(; spec::PanelModelSpec,
                              init=nothing,
                              frontier=nothing,
                              delta=nothing,
                              ln_sigma_u_sq=nothing,
                              ln_sigma_v_sq=nothing,
                              mu=nothing,
                              ln_lambda=nothing,
                              ln_k=nothing,
                              ln_sigma_sq=nothing,
                              ln_alpha=nothing,
                              ln_theta=nothing)

    K, L = spec.K, spec.L
    n_total = panel_plen(spec.model.ineff, K, L)
    idx = spec.idx

    # Complete vector mode
    if init !== nothing
        init_vec = Float64.(vec(init))
        length(init_vec) == n_total ||
            error("init has $(length(init_vec)) elements, expected $n_total")
        return init_vec
    end

    # Component mode: start with OLS + defaults
    if spec.T_periods isa Int
        y_t = sf_panel_demean(spec.depvar, spec.T_periods)
        x_t = sf_panel_demean(spec.frontier, spec.T_periods)
    else
        y_t = sf_panel_demean(spec.depvar, spec.offsets)
        x_t = sf_panel_demean(spec.frontier, spec.offsets)
    end
    beta_ols = x_t \ y_t
    init_vec = vcat(beta_ols, fill(0.1, n_total - K))

    # Distribution-specific default overrides
    if haskey(idx, :mu) && mu === nothing
        init_vec[idx.mu] = 0.0
    end
    if spec.ineff == :Lomax && haskey(idx, :ln_lambda) && ln_lambda === nothing
        init_vec[idx.ln_lambda] = -1.0
    end
    if spec.ineff == :Lomax && haskey(idx, :ln_alpha) && ln_alpha === nothing
        init_vec[idx.ln_alpha] = 0.5
    end

    # Apply component overrides
    if frontier !== nothing
        init_vec[idx.beta] .= Float64.(vec(frontier))
    end
    if delta !== nothing
        init_vec[idx.delta] .= Float64.(vec(delta))
    end
    if ln_sigma_v_sq !== nothing
        val = ln_sigma_v_sq isa Number ? ln_sigma_v_sq : first(ln_sigma_v_sq)
        init_vec[idx.ln_sigma_v_sq] = Float64(val)
    end
    if ln_sigma_u_sq !== nothing && haskey(idx, :ln_sigma_u_sq)
        val = ln_sigma_u_sq isa Number ? ln_sigma_u_sq : first(ln_sigma_u_sq)
        init_vec[idx.ln_sigma_u_sq] = Float64(val)
    end
    if mu !== nothing && haskey(idx, :mu)
        val = mu isa Number ? mu : first(mu)
        init_vec[idx.mu] = Float64(val)
    end
    if ln_lambda !== nothing && haskey(idx, :ln_lambda)
        val = ln_lambda isa Number ? ln_lambda : first(ln_lambda)
        init_vec[idx.ln_lambda] = Float64(val)
    end
    if ln_k !== nothing && haskey(idx, :ln_k)
        val = ln_k isa Number ? ln_k : first(ln_k)
        init_vec[idx.ln_k] = Float64(val)
    end
    if ln_sigma_sq !== nothing && haskey(idx, :ln_sigma_sq)
        val = ln_sigma_sq isa Number ? ln_sigma_sq : first(ln_sigma_sq)
        init_vec[idx.ln_sigma_sq] = Float64(val)
    end
    if ln_alpha !== nothing && haskey(idx, :ln_alpha)
        val = ln_alpha isa Number ? ln_alpha : first(ln_alpha)
        init_vec[idx.ln_alpha] = Float64(val)
    end
    if ln_theta !== nothing && haskey(idx, :ln_theta)
        val = ln_theta isa Number ? ln_theta : first(ln_theta)
        init_vec[idx.ln_theta] = Float64(val)
    end

    return init_vec
end


# ============================================================================
# Section 19: sfmodel_panel_opt()
# ============================================================================

"""
    sfmodel_panel_opt(; warmstart_solver=nothing, warmstart_opt=nothing,
                        main_solver, main_opt)

Specify optimization options for panel estimation.

# Arguments
- `warmstart_solver=nothing`: Warmstart optimizer (e.g., `NelderMead()`)
- `warmstart_opt=nothing`: Warmstart options as NamedTuple (e.g., `(iterations=200, g_abstol=1e-3)`)
- `main_solver`: Main optimizer (e.g., `Newton()`). Required.
- `main_opt`: Main options as NamedTuple. Required.
"""
function sfmodel_panel_opt(; warmstart_solver=nothing,
                             warmstart_opt=nothing,
                             main_solver,
                             main_opt)

    # Validate NamedTuple types
    if !(main_opt isa NamedTuple)
        error("Invalid `main_opt`: expected a NamedTuple, got $(typeof(main_opt)).\n" *
              "Hint: For single-element options, use a trailing comma: " *
              "`main_opt = (iterations = 200,)` not `main_opt = (iterations = 200)`.")
    end
    if warmstart_opt !== nothing && !(warmstart_opt isa NamedTuple)
        error("Invalid `warmstart_opt`: expected a NamedTuple, got $(typeof(warmstart_opt)).\n" *
              "Hint: For single-element options, use a trailing comma.")
    end

    # Convert NamedTuple to Optim.Options for main_opt
    m_opt = Optim.Options(; main_opt...)

    # Convert NamedTuple to Optim.Options for warmstart_opt if provided
    if warmstart_solver !== nothing
        if warmstart_opt === nothing
            ws_opt = Optim.Options(iterations = 100, g_abstol = 1e-3)
        else
            ws_opt = Optim.Options(; warmstart_opt...)
        end
    else
        ws_opt = nothing
    end

    return PanelOptSpec(warmstart_solver, ws_opt, main_solver, m_opt)
end


# ============================================================================
# Section 20: sfmodel_panel_fit()
# ============================================================================

"""Extract distribution-specific coefficient entries into the result OrderedDict."""
function _mypan_extract_result_coefs!(::HalfNormal_Panel, d, coef, idx)
    d[:ln_sigma_u_sq] = coef[idx.ln_sigma_u_sq]
end
function _mypan_extract_result_coefs!(::TruncatedNormal_Panel, d, coef, idx)
    d[:mu] = coef[idx.mu]
    d[:ln_sigma_u_sq] = coef[idx.ln_sigma_u_sq]
end
function _mypan_extract_result_coefs!(::Exponential_Panel, d, coef, idx)
    d[:ln_lambda] = coef[idx.ln_lambda]
end
function _mypan_extract_result_coefs!(::Weibull_Panel, d, coef, idx)
    d[:ln_lambda] = coef[idx.ln_lambda]
    d[:ln_k] = coef[idx.ln_k]
end
function _mypan_extract_result_coefs!(::Lognormal_Panel, d, coef, idx)
    d[:mu] = coef[idx.mu]
    d[:ln_sigma_sq] = coef[idx.ln_sigma_sq]
end
function _mypan_extract_result_coefs!(::Lomax_Panel, d, coef, idx)
    d[:ln_lambda] = coef[idx.ln_lambda]
    d[:ln_alpha] = coef[idx.ln_alpha]
end
function _mypan_extract_result_coefs!(::Rayleigh_Panel, d, coef, idx)
    d[:ln_sigma_sq] = coef[idx.ln_sigma_sq]
end
function _mypan_extract_result_coefs!(::Gamma_Panel, d, coef, idx)
    d[:ln_k] = coef[idx.ln_k]
    d[:ln_theta] = coef[idx.ln_theta]
end

"""
    sfmodel_panel_fit(; spec, method, init=nothing, optim_options=nothing,
                       jlms_bc_index=true, show_table=true, verbose=true)

Estimate the Wang and Ho (2010) panel stochastic frontier model.

# Arguments
- `spec::PanelModelSpec`: Model specification from `sfmodel_panel_spec()`
- `method::PanelMethodSpec`: Method specification from `sfmodel_panel_method()`
- `init=nothing`: Initial parameter vector from `sfmodel_panel_init()`
- `optim_options=nothing`: Optimization options from `sfmodel_panel_opt()`
- `jlms_bc_index::Bool=true`: Compute JLMS and BC efficiency indices
- `show_table::Bool=true`: Print formatted estimation table
- `verbose::Bool=true`: Print progress information

# Returns
NamedTuple with fields including `converged`, `loglikelihood`, `coeff`, `std_err`,
`jlms` (firm-level), `bc` (firm-level), and more. All results also accessible via `result.list`.
"""
function sfmodel_panel_fit(;
    spec::PanelModelSpec,
    method::PanelMethodSpec = sfmodel_panel_method(),
    init=nothing,
    optim_options=nothing,
    jlms_bc_index::Bool=true,
    marginal::Bool=true,
    show_table::Bool=true,
    verbose::Bool=true)

    # 1. Assemble internal spec (demean, generate draws, GPU convert)
    ispec = _assemble_panel_spec(spec, method)

    redflag::Int = 0

    # 2. Banner
    if show_table
        printstyled("\n###------------------------------------------------------###\n"; color=:yellow)
        printstyled("###  Panel SF: Wang and Ho (2010) — $(method.method)                  ###\n"; color=:yellow)
        printstyled("###  Quasi Monte Carlo integration in Julia              ###\n"; color=:yellow)
        printstyled("###------------------------------------------------------###\n\n"; color=:yellow)

        printstyled("*********************************\n"; color=:cyan)
        printstyled("      The estimated model:\n"; color=:cyan)
        printstyled("*********************************\n"; color=:cyan)

        printstyled("  $(spec.noise), $(spec.ineff)\n"; color=:yellow)
        T_info = spec.T_periods isa Int ? "T=$(spec.T_periods)" :
                 "T=$(minimum(spec.T_periods))-$(maximum(spec.T_periods)) (unbalanced)"
        _ndraws = ndims(ispec.draws) == 1 ? length(ispec.draws) : size(ispec.draws, 2)
        printstyled("  N_firms=$(spec.N), $(T_info), K=$(spec.K), L=$(spec.L), n_draws=$(_ndraws)\n"; color=:yellow)
        printstyled("  Type: $(spec.sign == 1 ? "production" : "cost")\n"; color=:yellow)
        println()
    end

    # 3. Prepare initial values
    K = ispec.K
    N = spec.N
    NT = spec.T_periods isa Int ? N * spec.T_periods : sum(spec.T_periods)

    # OLS on demeaned data
    if spec.T_periods isa Int
        y_t_cpu = sf_panel_demean(spec.depvar, spec.T_periods)
        x_t_cpu = sf_panel_demean(spec.frontier, spec.T_periods)
    else
        y_t_cpu = sf_panel_demean(spec.depvar, spec.offsets)
        x_t_cpu = sf_panel_demean(spec.frontier, spec.offsets)
    end
    beta_ols = x_t_cpu \ y_t_cpu

    if init === nothing
        n_total = panel_plen(spec.model.ineff, spec.K, spec.L)
        sf_init = vcat(beta_ols, fill(0.1, n_total - K))
        # Distribution-specific default overrides
        _idx = spec.idx
        if haskey(_idx, :mu);       sf_init[_idx.mu] = 0.0;       end
        if spec.ineff == :Lomax && haskey(_idx, :ln_lambda); sf_init[_idx.ln_lambda] = -1.0; end
        if spec.ineff == :Lomax && haskey(_idx, :ln_alpha); sf_init[_idx.ln_alpha] = 0.5;  end
        if verbose
            println("Using OLS-based initial values for frontier coefficients.")
            println("Other parameters initialized to 0.1")
            println()
        end
    else
        sf_init = init isa AbstractVector ? Float64.(init) : Float64.(vec(init))
    end

    # 4. OLS statistics on demeaned data
    resid = y_t_cpu - x_t_cpu * beta_ols
    sse = sum(resid .^ 2)
    ssd = sqrt(sse / NT)
    ll_ols = sum(normlogpdf.(0, ssd, resid))
    sk_ols = sum(resid .^ 3) / ((ssd^3) * NT)

    # 5. Prepare optimization options
    if optim_options === nothing
        myopt = sfmodel_panel_opt(
            warmstart_solver = NelderMead(),
            warmstart_opt    = (iterations = 200, g_abstol = 1e-3),
            main_solver      = Newton(),
            main_opt         = (iterations = 200, g_abstol = 1e-7)
        )
    else
        myopt = optim_options
    end

    do_warmstart = myopt.warmstart_solver !== nothing &&
                   myopt.warmstart_opt !== nothing &&
                   myopt.warmstart_opt.iterations > 0

    # 6. NLL closure
    _lik = p -> panel_nll(ispec, p; chunks=ispec.chunks)

    # Recording
    sf_init_1st = do_warmstart ? copy(sf_init) : nothing
    sf_init_2nd = nothing
    sf_warmstart_algo = do_warmstart ? myopt.warmstart_solver : nothing
    sf_warmstart_maxit = do_warmstart ? myopt.warmstart_opt.iterations : 0
    sf_main_algo = myopt.main_solver
    sf_main_maxit = myopt.main_opt.iterations
    sf_total_iter = 0

    _optres = nothing
    _run = 1

    # 7. Stage 1: Warmstart
    if do_warmstart && _run == 1
        if verbose
            printstyled("The warmstart run...\n\n"; color=:green)
        end

        _optres = Optim.optimize(_lik, sf_init, myopt.warmstart_solver,
                                  myopt.warmstart_opt)

        sf_total_iter += Optim.iterations(_optres)
        sf_init = Optim.minimizer(_optres)
        _run = 2

        if verbose
            println()
            println(_optres)
            print("The warmstart results are:\n")
            printstyled(Optim.minimizer(_optres); color=:yellow)
            println("\n")
        end
    end

    # 8. Stage 2: Main optimization
    if !do_warmstart || _run == 2
        sf_init_2nd = copy(sf_init)

        if verbose
            println()
            printstyled("Starting the main optimization run...\n\n"; color=:green)
        end

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

    # 9. Post-estimation
    _coevec = Optim.minimizer(_optres)

    # Variance-covariance matrix
    vcov_result = panel_var_cov_mat(_lik, _coevec; message=verbose)
    var_cov_matrix = vcov_result.var_cov_matrix
    if vcov_result.redflag > 0
        redflag = max(redflag, vcov_result.redflag)
    end

    stddev = sqrt.(abs.(diag(var_cov_matrix)))

    # Hessian
    numerical_hessian = hessian(_lik, _coevec)

    # 10. JLMS/BC indices (observation-level)
    if jlms_bc_index
        eff_result = panel_jlms_bc(ispec, _coevec; chunks=ispec.chunks)
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

    # 10b. Marginal effects
    if marginal
        margeff, margMinfo = panel_marginal_effects(ispec, _coevec)
    else
        margeff = nothing
        margMinfo = NamedTuple()
    end

    # 11. Print results table
    table_result = nothing
    if show_table
        table_result = panel_print_table(ispec, _coevec, var_cov_matrix;
                                          optim_result=_optres, table_format=:text)

        printstyled("***** Additional Information *********\n"; color=:cyan)

        print("* OLS (frontier-only) log-likelihood: ")
        printstyled(round(ll_ols; digits=5); color=:yellow)
        println("")

        print("* Skewness of OLS residuals: ")
        printstyled(round(sk_ols; digits=5); color=:yellow)
        println("")

        if jlms_bc_index
            print("* Mean JLMS inefficiency (observation-level): ")
            printstyled(round(_jlmsM; digits=5); color=:yellow)
            println("")
            print("* Mean BC efficiency (observation-level): ")
            printstyled(round(_bcM; digits=5); color=:yellow)
            println("\n")
        end

        if marginal && !isempty(margMinfo)
            print("* The sample mean of inefficiency determinants' marginal effects on E(u): ")
            printstyled(margMinfo; color=:yellow)
            println("")
            println("* Marginal effects of the inefficiency determinants at the observational")
            println("  level are saved in the return. See the follows.\n")
        end

        println("* Use `name.list` to see saved results (keys and values).")
        println("* Use `keys(name.list)` to see available keys.")
        println("  ** `name.loglikelihood`: the log-likelihood value;")
        println("  ** `name.jlms`: observation-level JLMS inefficiency index;")
        println("  ** `name.bc`: observation-level BC efficiency index.")
        println("  ** `name.marginal`: a DataFrame of variables' marginal effects on E(u).")

        printstyled("**************************************\n\n"; color=:cyan)
    end

    # 12. Build return dictionary
    _dicRES = OrderedDict{Symbol, Any}()
    _dicRES[:converged] = Optim.converged(_optres)
    _dicRES[:iter_limit_reached] = Optim.iteration_limit_reached(_optres)
    _dicRES[:_______________] = "___________________"
    _dicRES[:n_firms] = spec.N
    _dicRES[:T_periods] = spec.T_periods
    _dicRES[:n_observations] = NT
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
    _dicRES[:model] = ispec
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
    _dicRES[:n_draws] = method.draws !== nothing ? (ndims(method.draws) == 1 ? length(method.draws) : size(method.draws, 2)) : method.n_draws
    _dicRES[:multiRand] = method.multiRand
    _dicRES[:chunks] = method.chunks
    _dicRES[:distinct_Halton_length] = method.distinct_Halton_length
    _dicRES[:estimation_method] = method.method

    # Individual coefficient vectors (distribution-specific)
    idx = spec.idx
    _dicRES[:frontier] = _coevec[idx.beta]
    _dicRES[:delta] = _coevec[idx.delta]
    _dicRES[:ln_sigma_v_sq] = _coevec[idx.ln_sigma_v_sq]
    _mypan_extract_result_coefs!(spec.model.ineff, _dicRES, _coevec, idx)

    # Return as NamedTuple with dict access
    return (;
        converged = _dicRES[:converged],
        iter_limit_reached = _dicRES[:iter_limit_reached],
        n_firms = _dicRES[:n_firms],
        T_periods = _dicRES[:T_periods],
        n_observations = _dicRES[:n_observations],
        loglikelihood = _dicRES[:loglikelihood],
        table = _dicRES[:table],
        coeff = _dicRES[:coeff],
        std_err = _dicRES[:std_err],
        var_cov_mat = _dicRES[:var_cov_mat],
        jlms = _dicRES[:jlms],
        bc = _dicRES[:bc],
        marginal = _dicRES[:marginal],
        marginal_mean = _dicRES[:marginal_mean],
        OLS_loglikelihood = _dicRES[:OLS_loglikelihood],
        OLS_resid_skew = _dicRES[:OLS_resid_skew],
        model = _dicRES[:model],
        Hessian = _dicRES[:Hessian],
        gradient_norm = _dicRES[:gradient_norm],
        actual_iterations = _dicRES[:actual_iterations],
        warmstart_solver = _dicRES[:warmstart_solver],
        warmstart_ini = _dicRES[:warmstart_ini],
        warmstart_maxIT = _dicRES[:warmstart_maxIT],
        main_solver = _dicRES[:main_solver],
        main_ini = _dicRES[:main_ini],
        main_maxIT = _dicRES[:main_maxIT],
        main_tolerance = _dicRES[:main_tolerance],
        redflag = _dicRES[:redflag],
        GPU = _dicRES[:GPU],
        n_draws = _dicRES[:n_draws],
        multiRand = _dicRES[:multiRand],
        chunks = _dicRES[:chunks],
        distinct_Halton_length = _dicRES[:distinct_Halton_length],
        estimation_method = _dicRES[:estimation_method],
        frontier = _dicRES[:frontier],
        delta = _dicRES[:delta],
        ln_sigma_v_sq = _dicRES[:ln_sigma_v_sq],
        list = _dicRES,
    )
end


# ============================================================================
# Section 21: Usage Examples
# ============================================================================

#=
## Example 1: Balanced Panel — MSLE Method

using Random
Random.seed!(12345)

N, T = 100, 10                # 100 firms, 10 periods each
NT = N * T

# Simulate data
X = randn(NT, 2)              # 2 frontier regressors
Z = randn(NT, 1)              # 1 scaling variable
α = repeat(randn(N), inner=T) # firm fixed effects
v = 0.5 * randn(NT)           # noise:  v ~ N(0, 0.25)
u_star = repeat(abs.(randn(N)), inner=T)  # inefficiency: u* ~ N⁺(0,1)
h = exp.(Z * [0.3])
u = h .* u_star

y = α .+ X * [1.0, -0.5] .+ v .- u

# --- Step 1: Model specification (balanced) ---
spec = sfmodel_panel_spec(
    depvar   = y,
    frontier = X,
    zvar     = Z,
    T_periods = T,
    type     = :prod,
    varnames = ["capital", "labor", "z1", "ln_σᵤ²", "ln_σᵥ²"]
)

# --- Step 2: Method specification ---
method = sfmodel_panel_method(
    method  = :MSLE,
    n_draws = 1024,
    GPU     = false,
    chunks  = 4
)

# --- Step 3 (optional): Custom initial values ---
init = sfmodel_panel_init(spec=spec)   # OLS-based defaults

# --- Step 4 (optional): Optimization options ---
opt = sfmodel_panel_opt(
    warmstart_solver = NelderMead(),
    warmstart_opt    = (iterations=200, g_abstol=1e-3),
    main_solver      = Newton(),
    main_opt         = (iterations=200, g_abstol=1e-7)
)

# --- Step 5: Fit the model ---
result = sfmodel_panel_fit(
    spec           = spec,
    method         = method,
    init           = init,
    optim_options  = opt,
    jlms_bc_index  = true,
    show_table     = true,
    verbose        = true
)

# --- Access results ---
result.converged          # true/false
result.loglikelihood      # log-likelihood at optimum
result.coeff              # coefficient vector
result.std_err            # standard errors
result.jlms               # N-vector: E[u_i* | data_i]
result.bc                 # N-vector: E[exp(-u_i*) | data_i]
result.frontier           # estimated β
result.delta              # estimated δ
result.var_cov_mat        # variance-covariance matrix
keys(result.list)         # all available result keys


## Example 2: Balanced Panel — MCI Method with different transformation rules

# Using default transformation (logistic_1_rule for HalfNormal):
method_mci = sfmodel_panel_method(
    method  = :MCI,
    n_draws = 1024
)

# Using logistic_2_rule:
method_l2 = sfmodel_panel_method(
    method         = :MCI,
    transformation = :logistic_2_rule,
    n_draws        = 1024
)

# Using expo_rule:
method_expo = sfmodel_panel_method(
    method         = :MCI,
    transformation = :expo_rule,
    n_draws        = 1024
)

result_mci = sfmodel_panel_fit(spec=spec, method=method_mci)


## Example 3: Unbalanced Panel — using `id` column

using Random
Random.seed!(54321)

# Firms with different numbers of periods
firm_ids = vcat(
    fill("Firm_A", 8),
    fill("Firm_B", 12),
    fill("Firm_C", 5),
    fill("Firm_D", 10),
    fill("Firm_E", 7)
)
NT = length(firm_ids)  # = 42

X = randn(NT, 1)
Z = randn(NT, 1)
y = randn(NT)          # (simplified; use proper DGP in practice)

# --- Specify with id column (strings work fine) ---
spec_ub = sfmodel_panel_spec(
    depvar   = y,
    frontier = X,
    zvar     = Z,
    id       = firm_ids,     # any type: strings, integers, symbols, ...
    type     = :prod
)

method_ub = sfmodel_panel_method(method=:MSLE, n_draws=512)
result_ub = sfmodel_panel_fit(spec=spec_ub, method=method_ub)


## Example 4: Minimal usage with all defaults

# Balanced panel, MSLE, default optimizer, default draws:
result = sfmodel_panel_fit(
    spec = sfmodel_panel_spec(depvar=y, frontier=X, zvar=Z, T_periods=10)
)


## Example 5: Using sfmodel_panel_init for custom starting values

init = sfmodel_panel_init(
    spec          = spec,
    frontier      = [0.8, -0.3],   # override OLS-based β
    delta         = [0.2],         # override default δ
    ln_sigma_u_sq = 0.5,           # override default ln(σ_u²)
    ln_sigma_v_sq = -0.5           # override default ln(σ_v²)
)

# Or provide a complete vector directly:
init_full = sfmodel_panel_init(spec=spec, init=[0.8, -0.3, 0.2, 0.5, -0.5])


## Example 6: DSL — Unbalanced panel with @zvar and @id

using CSV, DataFrames
df = DataFrame(CSV.File("WH2010T_unbalanced.csv"))

spec = sfmodel_panel_spec(
    @useData(df),
    @depvar(yit),
    @frontier(xit),
    @zvar(zit),
    @id(id);
    type = :prod
)
method = sfmodel_panel_method(method=:MSLE, n_draws=2^12-1)
result = sfmodel_panel_fit(spec=spec, method=method)


## Example 7: DSL — Balanced panel (no @id, use T_periods keyword)

spec_bal = sfmodel_panel_spec(
    @useData(df),
    @depvar(yit),
    @frontier(xit),
    @zvar(zit);
    T_periods = 10,
    type = :prod
)


## Example 8: DSL — Without @zvar (constant scaling h(z)=exp(δ₀))

spec_no_z = sfmodel_panel_spec(
    @useData(df),
    @depvar(yit),
    @frontier(xit),
    @id(id);
    type = :prod
)
=#
