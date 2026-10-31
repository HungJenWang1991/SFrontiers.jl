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
                      Lognormal, Lomax, Rayleigh, Gamma (MCI only),
                      Erlang2, Erlang3, Erlang4

    Features:
        - Multiple dispatch for model combinations (no if-else logic)
        - CPU and GPU support via array type dispatch
        - ForwardDiff compatible via type parameter P
        - Balanced and unbalanced panels (per-firm loop with offsets for unbalanced)
        - Chunking for GPU memory management
=#

using HaltonSequences
using SpecialFunctions: erfinv, erfcinv, erf, erfc, loggamma
using Statistics: mean
using LinearAlgebra: diag, inv, I
using ForwardDiff: ForwardDiff, hessian, gradient
using Distributions: Normal, cquantile, ccdf
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

"""Exponential inefficiency: u* ~ Exp(σᵤ) with scale σᵤ; the estimated parameter is σᵤ² = Var(u*)"""
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

"""Erlang inefficiency: u* ~ Erlang(M, θ) = Gamma(shape=M, scale=θ) with the integer shape M pinned. Registered orders M ∈ {2, 3, 4}."""
struct Erlang_Panel{M} <: PanelIneffModel end

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
    :Erlang2         => Erlang_Panel{2},
    :Erlang3         => Erlang_Panel{3},
    :Erlang4         => Erlang_Panel{4},
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
panel_plen(::Erlang_Panel, K::Int, L::Int)           = K + L + 2

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
    ln_sigma_u_sq = idx;         idx += 1
    ln_sigma_v_sq = idx
    return (beta=beta, delta=delta, ln_sigma_u_sq=ln_sigma_u_sq, ln_sigma_v_sq=ln_sigma_v_sq)
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
    ln_sigma_u_sq = idx;           idx += 1
    ln_sigma_v_sq = idx
    return (beta=beta, delta=delta, mu=mu, ln_sigma_u_sq=ln_sigma_u_sq, ln_sigma_v_sq=ln_sigma_v_sq)
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
    ln_sigma_u_sq = idx;           idx += 1
    ln_sigma_v_sq = idx
    return (beta=beta, delta=delta, ln_sigma_u_sq=ln_sigma_u_sq, ln_sigma_v_sq=ln_sigma_v_sq)
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

function _panel_param_ind(::Erlang_Panel, K::Int, L::Int)
    idx = 1
    beta = idx:(idx + K - 1);   idx += K
    delta = idx:(idx + L - 1);  idx += L
    ln_theta = idx;              idx += 1
    ln_sigma_v_sq = idx
    return (beta=beta, delta=delta, ln_theta=ln_theta, ln_sigma_v_sq=ln_sigma_v_sq)
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
    T_periods::Vector{Int}           # per-firm period counts
    T_max::Int                       # max time periods
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

    # Firm identifiers (unique, in first-appearance order; used by predict()
    # to align fit-time draws rows to predict-time firms by id).
    firm_ids::AbstractVector
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
    recycling::Int  # 0 = no recycling; otherwise the length at which the Halton sequence restarts
    halton_legacy::Bool  # true when that length came from distinct_Halton_length (the rule of versions before 2.0.0)
end

# The layout of versions before 2.0.0: an explicit length keeps the rule of those versions.
PanelMethodSpec(method, transformation, draws, n_draws, multiRand, GPU, chunks, distinct_Halton_length::Integer) =
    PanelMethodSpec(method, transformation, draws, n_draws, multiRand, GPU, chunks, Int(distinct_Halton_length), true)

# A copy of the method `m` with `n_draws = D` and `chunks = chunks`, every other field unchanged: the method of
# the integration check, whose draws are those of a fit with `n_draws = D` and the same settings.
_with_draws(m::PanelMethodSpec, D, chunks) =
    PanelMethodSpec(m.method, m.transformation, m.draws, D, m.multiRand, m.GPU, chunks, m.recycling, m.halton_legacy)

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
    T_periods::Vector{Int}           # per-firm period counts
    T_max::Int                       # max time periods
    offsets::Vector{Int}             # firm boundaries in stacked data, length N+1
    Tm1::AbstractVector              # T_i - 1: N-vector (CuArray on GPU)

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
        k_floor        = TT(0.1),       # Weibull/Gamma shape floor (kernel-aligned)
        weibull_k_ceil = TT(10.0),      # Weibull shape ceiling (kernel: [0.1, 10])
        gamma_k_ceil   = TT(100.0),     # Gamma shape ceiling   (kernel: [0.1, 100])
        theta_floor   = TT(1e-12),
        theta_ceil    = TT(1e12),
        xm_floor      = TT(1e-10),      # Lomax scale floor  (kernel: [1e-10, 1e10])
        xm_ceil       = TT(1e10),       # Lomax scale ceiling
        alpha_floor   = TT(0.1),        # Lomax shape floor  (kernel: [0.1, 100])
        alpha_ceil    = TT(100.0),      # Lomax shape ceiling
        sqrt2         = sqrt2,
        inv_sqrt2     = inv(sqrt2),
        sqrt_pi       = sqrt(TT(π)),
        lo_erf        = TT(32) * eps(TT),               # HalfNormal: erfinv arg ∈ [0,1)
        lo_erf_signed = -one(TT) + TT(32) * eps(TT),    # TN/Lognormal: signed erfinv arg ∈ (−1,1)
        hi_erf        = one(TT) - TT(32) * eps(TT),
        exp_clamp     = TT(1e-15),
        out_ceil      = TT(1e15),       # Lomax draw output ceiling (kernel-aligned)
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
    sigma_sq = clamp(exp(p[idx.ln_sigma_u_sq]), P(c.sigma_floor), P(c.sigma_ceil))
    return (sigma_sq=sigma_sq,)
end

function get_panel_ineff_vals(::Weibull_Panel, p, idx, c)
    P = eltype(p)
    lambda = clamp(exp(p[idx.ln_lambda]), P(c.sigma_floor), P(c.sigma_ceil))
    k = clamp(exp(p[idx.ln_k]), P(c.k_floor), P(c.weibull_k_ceil))
    return (lambda=lambda, k=k)
end

function get_panel_ineff_vals(::Lognormal_Panel, p, idx, c)
    P = eltype(p)
    mu = p[idx.mu]
    sigma_sq = clamp(exp(p[idx.ln_sigma_u_sq]), P(c.sigma_floor)^2, P(c.sigma_ceil)^2)
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
    sigma_sq = clamp(exp(p[idx.ln_sigma_u_sq]), P(c.sigma_floor)^2, P(c.sigma_ceil)^2)
    sigma = sqrt(sigma_sq)
    return (sigma=sigma, sigma_sq=sigma_sq)
end

function get_panel_ineff_vals(::Gamma_Panel, p, idx, c)
    P = eltype(p)
    k = clamp(exp(p[idx.ln_k]), P(c.k_floor), P(c.gamma_k_ceil))
    theta = clamp(exp(p[idx.ln_theta]), P(c.theta_floor), P(c.theta_ceil))
    lgk = loggamma(k)
    return (k=k, theta=theta, lgk=lgk)
end

function get_panel_ineff_vals(::Erlang_Panel{M}, p, idx, c) where {M}
    P = eltype(p)
    theta = clamp(exp(p[idx.ln_theta]), P(c.theta_floor), P(c.theta_ceil))
    # The shape is pinned at the Erlang order; carrying (k, lgk) as constants lets
    # every Gamma kernel below apply unchanged
    return (k=P(M), theta=theta, lgk=P(loggamma(M)))
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
default_panel_transformation(::Erlang_Panel)           = :expo_rule

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

"""Log-PDF of truncated normal distribution TN(μ, σ²; lower=0).

The normaliser log Φ(μ/σ): below μ/σ = -1 it is the stable `sf_normlogcdf_lowtail`
(src/sf_numerics.jl), since `log(max(Φ, clamp_lo))` with Φ = (1 + erf)/2 cancels as μ/σ falls
(errors of either sign, up to 0.037 just above -7.94) and is floored at 1e-15 below
μ/σ ≈ -7.94. At μ/σ ≥ -1 it is `log(max(Φ, clamp_lo))`."""
@inline function log_pdf_truncnormal_panel(u, mu, sigma, clamp_lo)
    sigma_safe = max(sigma, clamp_lo)
    z = (u - mu) / sigma_safe
    a = mu / sigma_safe
    log_Phi = a < -1 ? sf_normlogcdf_lowtail(a) :
              log(max(oftype(z, 0.5) * (1 + erf(mu / (sigma_safe * sqrt(oftype(z, 2))))), clamp_lo))
    return oftype(z, -0.5) * log(oftype(z, 2π)) - log(sigma_safe) - oftype(z, 0.5) * z^2 - log_Phi
end

"""Log-PDF of Exponential distribution with squared scale σᵤ² (= Var(u)); the scale is
√σᵤ², so f(u) = (1/√σᵤ²)·exp(-u/√σᵤ²)."""
@inline function log_pdf_exponential_panel(u, sigma_sq, clamp_lo)
    sigma_sq_safe = max(sigma_sq, clamp_lo)
    return -0.5 * log(sigma_sq_safe) - u / sqrt(sigma_sq_safe)
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

# One truncated-normal draw. Up to
# y = -μ/(σ√2) = 8 this is the closed form, bit for bit; beyond, erfc(y) heads for underflow
# (NaN derivatives from μ/σ ≈ -37.5, infinite draws below -38.5 in Float64) and μ cancels, so
# the draw is σ√2·t with t from `sf_tn_quantile_tail` (src/sf_numerics.jl).
@inline function _panel_tn_udraw(mu, sigma_u, s2, r)
    y = -mu / (sigma_u * s2)
    y > 8 && return sigma_u * s2 * sf_tn_quantile_tail(y, r + zero(y))   # r widened, as in _tn_quantile
    return max(mu + sigma_u * s2 * erfcinv(erfc(y) * (1 - r)), zero(r))
end

function mypan_gen_udraws_msle(::TruncatedNormal_Panel, ineff_vals, halton_P, c, ::Type{P}) where P
    mu = ineff_vals.mu
    sigma_u = ineff_vals.sigma_u
    s2 = P(c.sqrt2)
    # inverse CDF with no argument near 1; see myTruncatedNormalQuantile in sf_MCI_v21.jl.
    # The clamped erfinv form would cap u at mu + 4.62*sigma_u in Float32.
    return _panel_tn_udraw.(mu, sigma_u, s2, halton_P)
end

function mypan_gen_udraws_msle(::Exponential_Panel, ineff_vals, halton_P, c, ::Type{P}) where P
    sigma_sq = ineff_vals.sigma_sq
    return .-log.(max.(1 .- halton_P, P(c.exp_clamp))) .* sqrt(sigma_sq)
end

function mypan_gen_udraws_msle(::Weibull_Panel, ineff_vals, halton_P, c, ::Type{P}) where P
    lambda = ineff_vals.lambda
    k = ineff_vals.k
    return lambda .* (.-log.(max.(1 .- halton_P, P(c.exp_clamp)))) .^ (1 / k)
end

function mypan_gen_udraws_msle(::Lognormal_Panel, ineff_vals, halton_P, c, ::Type{P}) where P
    mu = ineff_vals.mu
    sigma = ineff_vals.sigma
    inner = clamp.(2 .* halton_P .- 1, P(c.lo_erf_signed), P(c.hi_erf))  # signed arg: floor at −1+32eps
    return exp.(mu .+ sigma .* P(c.sqrt2) .* erfinv.(inner))
end

function mypan_gen_udraws_msle(::Lomax_Panel, ineff_vals, halton_P, c, ::Type{P}) where P
    lambda = ineff_vals.lambda
    alpha = ineff_vals.alpha
    return clamp.(lambda .* expm1.((-1 / alpha) .* log.(max.(1 .- halton_P, P(c.exp_clamp)))), zero(P), P(c.out_ceil))
end

function mypan_gen_udraws_msle(::Rayleigh_Panel, ineff_vals, halton_P, c, ::Type{P}) where P
    sigma = ineff_vals.sigma
    return sigma .* sqrt.(-2 .* log.(max.(1 .- halton_P, P(c.exp_clamp))))
end

function mypan_gen_udraws_msle(::Gamma_Panel, args...)
    error("`ineff=:Gamma` does not have a closed-form inverse CDF. Use `method=:MCI` instead.")
end

# Strip (possibly nested, for Hessians) ForwardDiff dual parts. The Halton draws are
# data with zero partials, so this is exact; the standardized Erlang quantile is
# parameter-free and must run on plain floats so no dual numbers enter its Newton
# iteration (same design as the cross-sectional Erlang_MSLE backend).
@inline _pan_primal(x::AbstractFloat) = x
@inline _pan_primal(x::Real) = _pan_primal(ForwardDiff.value(x))

function mypan_gen_udraws_msle(::Erlang_Panel{M}, ineff_vals, halton_P, c, ::Type{P}) where {M, P}
    r = _pan_primal.(halton_P)
    Tr = eltype(r)
    # Q(p; M, θ) = θ·q_M(p): q_M depends only on the fixed draws, so dQ/dθ = q_M(p)
    # is exact by construction (myErlangStdQuantile is the kernel imported
    # from MSLE_Backend in load_Panel.jl)
    # upper bound via _draw_clamp_hi (imported with the kernel): `1 - clamp_lo` is not
    # representable below one in Float32, where a draw of exactly one would overflow
    q = myErlangStdQuantile.(clamp.(r, Tr(c.clamp_lo), _draw_clamp_hi(Tr, c.clamp_lo)), M, Tr(loggamma(M)))
    return ineff_vals.theta .* q
end

# In-place mypan_gen_udraws_msle: writes the draws u for the draw rows h into out.
function mypan_gen_udraws_msle!(out, ::HalfNormal_Panel, ineff_vals, h, c, ::Type{P}) where P
    out .= ineff_vals.sigma_u .* P(c.sqrt2) .* erfinv.(clamp.(P.(h), P(c.lo_erf), P(c.hi_erf)))
end
function mypan_gen_udraws_msle!(out, ::TruncatedNormal_Panel, ineff_vals, h, c, ::Type{P}) where P
    out .= _panel_tn_udraw.(ineff_vals.mu, ineff_vals.sigma_u, P(c.sqrt2), P.(h))
end
function mypan_gen_udraws_msle!(out, ::Exponential_Panel, ineff_vals, h, c, ::Type{P}) where P
    out .= .-log.(max.(1 .- P.(h), P(c.exp_clamp))) .* sqrt(ineff_vals.sigma_sq)
end
function mypan_gen_udraws_msle!(out, ::Weibull_Panel, ineff_vals, h, c, ::Type{P}) where P
    out .= ineff_vals.lambda .* (.-log.(max.(1 .- P.(h), P(c.exp_clamp)))) .^ (1 / ineff_vals.k)
end
function mypan_gen_udraws_msle!(out, ::Lognormal_Panel, ineff_vals, h, c, ::Type{P}) where P
    out .= clamp.(2 .* P.(h) .- 1, P(c.lo_erf_signed), P(c.hi_erf))        # the argument of erfinv
    out .= exp.(ineff_vals.mu .+ ineff_vals.sigma .* P(c.sqrt2) .* erfinv.(out))
end
function mypan_gen_udraws_msle!(out, ::Lomax_Panel, ineff_vals, h, c, ::Type{P}) where P
    out .= clamp.(ineff_vals.lambda .* expm1.((-1 / ineff_vals.alpha) .* log.(max.(1 .- P.(h), P(c.exp_clamp)))),
                  zero(P), P(c.out_ceil))
end
function mypan_gen_udraws_msle!(out, ::Rayleigh_Panel, ineff_vals, h, c, ::Type{P}) where P
    out .= ineff_vals.sigma .* sqrt.(-2 .* log.(max.(1 .- P.(h), P(c.exp_clamp))))
end
mypan_gen_udraws_msle!(out, ::Gamma_Panel, args...) = mypan_gen_udraws_msle(Gamma_Panel(), args...)   # the same error
function mypan_gen_udraws_msle!(out, ::Erlang_Panel{M}, ineff_vals, h, c, ::Type{P}) where {M, P}
    Tr = typeof(_pan_primal(one(P)))          # the plain-float type under P's dual layers
    out .= ineff_vals.theta .* myErlangStdQuantile.(clamp.(_pan_primal.(P.(h)), Tr(c.clamp_lo),
                                                           _draw_clamp_hi(Tr, c.clamp_lo)), M, Tr(loggamma(M)))
end

# --- MCI scale parameter extraction ---

"""Extract the scale parameter for MCI transformation. Dispatched by inefficiency distribution."""
mypan_get_mci_scale(::HalfNormal_Panel, ineff_vals)      = ineff_vals.sigma_u
mypan_get_mci_scale(::TruncatedNormal_Panel, ineff_vals)  = ineff_vals.sigma_u
mypan_get_mci_scale(::Exponential_Panel, ineff_vals)      = sqrt(ineff_vals.sigma_sq)
mypan_get_mci_scale(::Weibull_Panel, ineff_vals)          = ineff_vals.lambda
mypan_get_mci_scale(::Lognormal_Panel, ineff_vals)        = ineff_vals.sigma
# Lomax: the MEDIAN λ(2^(1/α) − 1), the cross-section's rule (get_scale_param(::Lomax_MCI) in
# src/sf_MCI/sf_MCI_T_v21.jl). λ itself is far wider than the distribution once α is large (the median
# is about λ·0.69/α), so most draws would land where the integrand is negligible.
mypan_get_mci_scale(::Lomax_Panel, ineff_vals)            = ineff_vals.lambda * (exp2(1 / ineff_vals.alpha) - 1)
mypan_get_mci_scale(::Rayleigh_Panel, ineff_vals)         = ineff_vals.sigma
# Gamma: θ·√(1+k²). The mass of Gamma(k, θ) sits near kθ: with scale θ, for k > 1 the importance weight
# of the expo_rule draws is proportional to u^(k−1), unbounded and increasing, so a few draws decide each
# firm's integral (at k = 5 the log-likelihood of 200-firm panels comes out 8.3 to 55.0 nats too low).
# For large k, θ·√(1+k²) is close to kθ; for k < 1 it lies between θ and √2·θ, where kθ is too narrow
# for firms in the tail; and it is smooth in k (max(k, 1)·θ, the other candidate, has a kink at k = 1).
mypan_get_mci_scale(::Gamma_Panel, ineff_vals)            = sqrt(1 + ineff_vals.k^2) * ineff_vals.theta
# Mass of Erlang(M, θ) sits near M·θ — scale the expo_rule draws accordingly
mypan_get_mci_scale(::Erlang_Panel{M}, ineff_vals) where {M} = M * ineff_vals.theta

# --- MCI log-PDF evaluation ---

"""Evaluate log-PDF of inefficiency distribution over D-vector of u_draws. Dispatched by distribution."""
function mypan_logpdf_ineff(::HalfNormal_Panel, u_draws, ineff_vals, clamp_lo)
    return log_pdf_halfnormal_panel.(u_draws, ineff_vals.sigma_u, clamp_lo)
end

function mypan_logpdf_ineff(::TruncatedNormal_Panel, u_draws, ineff_vals, clamp_lo)
    return log_pdf_truncnormal_panel.(u_draws, ineff_vals.mu, ineff_vals.sigma_u, clamp_lo)
end

function mypan_logpdf_ineff(::Exponential_Panel, u_draws, ineff_vals, clamp_lo)
    return log_pdf_exponential_panel.(u_draws, ineff_vals.sigma_sq, clamp_lo)
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

# Erlang ineff_vals carries (k, theta, lgk) with the shape pinned at the Erlang
# order, so the Gamma kernel applies unchanged
mypan_logpdf_ineff(::Erlang_Panel, u_draws, ineff_vals, clamp_lo) =
    mypan_logpdf_ineff(Gamma_Panel(), u_draws, ineff_vals, clamp_lo)

# In-place mypan_logpdf_ineff: writes log f_u(u) into out.
mypan_logpdf_ineff!(out, ::HalfNormal_Panel, u, iv, clamp_lo)      = (out .= log_pdf_halfnormal_panel.(u, iv.sigma_u, clamp_lo))
mypan_logpdf_ineff!(out, ::TruncatedNormal_Panel, u, iv, clamp_lo) = (out .= log_pdf_truncnormal_panel.(u, iv.mu, iv.sigma_u, clamp_lo))
mypan_logpdf_ineff!(out, ::Exponential_Panel, u, iv, clamp_lo)     = (out .= log_pdf_exponential_panel.(u, iv.sigma_sq, clamp_lo))
mypan_logpdf_ineff!(out, ::Weibull_Panel, u, iv, clamp_lo)         = (out .= log_pdf_weibull_panel.(u, iv.lambda, iv.k, clamp_lo))
mypan_logpdf_ineff!(out, ::Lognormal_Panel, u, iv, clamp_lo)       = (out .= log_pdf_lognormal_panel.(u, iv.mu, iv.sigma, clamp_lo))
mypan_logpdf_ineff!(out, ::Lomax_Panel, u, iv, clamp_lo)           = (out .= log_pdf_lomax_panel.(u, iv.lambda, iv.alpha, clamp_lo))
mypan_logpdf_ineff!(out, ::Rayleigh_Panel, u, iv, clamp_lo)        = (out .= log_pdf_rayleigh_panel.(u, iv.sigma, clamp_lo))
mypan_logpdf_ineff!(out, ::Gamma_Panel, u, iv, clamp_lo)           = (out .= log_pdf_gamma_panel.(u, iv.k, iv.theta, iv.lgk, clamp_lo))
mypan_logpdf_ineff!(out, ::Erlang_Panel, u, iv, clamp_lo)          = mypan_logpdf_ineff!(out, Gamma_Panel(), u, iv, clamp_lo)


# ============================================================================
# Section 8: Halton Sequence Generation
# ============================================================================

# User-supplied draws must be probabilities in [0, 1): a draw of exactly 1 maps to an
# infinite inefficiency (MCI: u is capped at 1/clamp_lo while the Jacobian s/(1-t) is
# infinite, so the firm's likelihood is NaN; MSLE: the quantile clamp silently moves
# the draw to the upper bound). Zero is the support lower bound and harmless.
# `minimum`/`maximum` are reductions, so the check also runs on CuArray draws.
function _validate_panel_user_draws(user_draws)
    isempty(user_draws) && error("`draws` is empty.")
    dmin = minimum(user_draws)
    dmax = maximum(user_draws)
    (dmin >= 0 && dmax < 1) ||
        error("`draws` must lie in [0, 1): found min = $dmin, max = $dmax. " *
              "The contract is half-open: 0 is the lower bound of the support and is " *
              "accepted, 1 is not. Under `method=:MCI` a draw of exactly 1 maps to an " *
              "infinite inefficiency with an infinite Jacobian, so the likelihood is NaN; " *
              "under `method=:MSLE` the quantile kernels clamp it to their upper bound " *
              "instead, so the draw silently becomes a different draw. " *
              "Note that a draw just below 1 can round to exactly 1 when converted to the " *
              "data's element type — supply draws already in that type, or move them off 1.")
    return nothing
end

# One rule for user-supplied draws, whatever `multiRand` was passed: supplying `draws` forces
# multiRand=false at `sfmodel_panel_method`, and the ROW COUNT says how the draws are read.
#   * a vector, a 1×D row, or a D×1 column -> one sequence of D draws shared by every firm;
#   * an N×D matrix (N = number of FIRMS)  -> one row of D draws per firm;
#   * anything else                        -> error (no way to split it at the firm level).
# Returns a length-D vector (shared, the panel's storage for that case) or an N×D matrix.
function _shape_user_panel_draws(user_draws, N::Int)
    ndims(user_draws) == 1 && return vec(user_draws)
    nr, nc = size(user_draws)
    nc == 1 && return vec(user_draws)                # D×1 is the same sequence as 1×D
    nr == 1 && return vec(user_draws)                # 1×D: shared
    nr == N && return user_draws                     # N×D: one row per firm
    error("`draws` has size $(size(user_draws)): the number of rows must be 1 (one sequence of " *
          "draws shared by every firm) or N=$N (the number of firms, one row of draws per firm). " *
          "A vector or a single-column matrix is read as a shared sequence.")
end

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

The N × D per-firm Halton draws of versions before 2.0.0: one base-2 Halton sequence shortened to 2^m − 1
points, the largest such length not above N × D or `distinct_Halton_length`; row i takes the D points that
follow row i − 1, and the sequence restarts from its first point when it runs out. Another prime `base` gives
the Halton sequence in that base, built the same way. The estimation no longer uses this function; it is kept
so that earlier scripts run. A fit reproduces these draws with the deprecated `distinct_Halton_length`. Throws
`ArgumentError` when D exceeds that length.

# Arguments
- `N::Int`: Number of firms
- `D::Int`: Number of draws per firm
- `base::Int=2`: Base for Halton sequence generation
- `T::Type{<:AbstractFloat}=Float64`: Element type
- `distinct_Halton_length::Int=2^15-1`: Maximum Halton sequence length

# Returns
- `Matrix{T}` of size (N, D) with Halton values in (0,1)
"""
function make_panel_halton_wrap(N::Int, D::Int; base::Int=2, T::Type{<:AbstractFloat}=Float64, distinct_Halton_length::Int=2^15-1)
    N > 0 || throw(ArgumentError("N must be positive, got $N"))
    D > 0 || throw(ArgumentError("D (draws) must be positive, got $D"))
    D <= distinct_Halton_length ||
        throw(ArgumentError("D = $D exceeds distinct_Halton_length = $distinct_Halton_length"))
    L = sf_halton_period(N, D; recycling = distinct_Halton_length, legacy = true)
    base == 2 && return sf_halton_draws(N, D; T = T, period = L)
    h = T.(collect(Halton(base, length = L)))      # other prime bases, as in versions before 2.0.0
    return [h[mod1((i - 1) * D + j, L)] for i in 1:N, j in 1:D]
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
    _panel_draw_rows(draws::Main.CUDA.AnyCuArray, a::Int, b::Int) = Array(draws[a:b, :])
end
_to_device_panel(x::AbstractArray, ::AbstractArray) = x  # CPU ref: no-op

# True when both arrays are of the same kind: both `Array`s, or both `CuArray`s.
_same_array_kind(a, b) = typeof(a).name.wrapper === typeof(b).name.wrapper

# Pull to CPU (no-op when already CPU)
_to_cpu(x::Array) = x
_to_cpu(x::AbstractArray) = Array(x)

# Rows a:b of a per-firm draw block on the host: a view of a host array, a copy of those rows
# (and only those) for a device array.
_panel_draw_rows(draws::AbstractMatrix, a::Int, b::Int) = view(draws, a:b, :)

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

"""In-place `logsumexp_rows_panel`: overwrites `A` with `exp.(A .- rowmax)` and returns the row log-sum-exp.
`A` must be a dense array of its own, not a view, so that its row sums reduce the same input as
`logsumexp_rows_panel` does."""
function _logsumexp_rows_panel!(A::AbstractMatrix, clamp_lo=1e-300)
    max_vals = _maximum_panel(A; dims=2)
    A .= exp.(A .- max_vals)
    sum_exp  = _sum_panel(A; dims=2)
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

With `chunks >= 2` the firms are processed one group at a time in work arrays of one group
(per-firm draws are converted and transformed group by group, shared draws once), so `chunks`
bounds the memory of the likelihood and its derivatives; `chunks <= 1` processes all firms at once.
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
    h_tilde = compute_h_tilde(spec.z_raw, p, idx, spec.offsets)
    A, B, C = _compute_panel_ABC(epsilon_tilde, h_tilde, spec.offsets, N)
    # Move A, B, C to GPU if data lives there (halton is CuArray when GPU=true)
    A = _to_device_panel(A, halton)
    B = _to_device_panel(B, halton)
    C = _to_device_panel(C, halton)

    # 3. Extract noise/inefficiency parameters
    noise_vals = get_panel_noise_vals(model.noise, p, idx, c)
    ineff_vals = get_panel_ineff_vals(model.ineff, p, idx, c)

    # 5. Pre-compute log-density constants
    inv_sigma_v_sq = noise_vals.inv_sigma_v_sq
    log_sigma_v_sq = noise_vals.log_sigma_v_sq
    log_const = P(-0.5) .* spec.Tm1 .* (log(P(2π)) .+ log_sigma_v_sq)

    frontier_sign = P(spec.sign)

    # Quadratic expansion: sum_t (E_ti + s*H_ti*u_d)^2 = A_i + 2s*B_i*u_d + C_i*u_d^2
    two_s = P(2) * frontier_sign

    if chunks <= 1
        # 4. Generate u draws via inverse CDF (dispatched by inefficiency distribution)
        halton_P = P.(halton)
        u_draws = mypan_gen_udraws_msle(model.ineff, ineff_vals, halton_P, c, P)
        u_row = ndims(u_draws) == 1 ? reshape(u_draws, 1, D) : u_draws   # 1×D or N×D

        # --- Non-chunked: all firms at once ---
        quad_2D = A .+ two_s .* B .* u_row .+ C .* u_row .^ 2   # N × D

        # Log-density per firm per draw
        log_fv = P(-0.5) .* quad_2D .* inv_sigma_v_sq .+ log_const

        # Aggregate: log L_i = logsumexp_d(log_fv) - log(D)
        log_likes = logsumexp_rows_panel(log_fv) .- log(P(D))
        return -_sum_scalar_panel(log_likes)
    else
        # Each group of firms converts and transforms its own rows of the draws in a buffer of one
        # group, so `chunks` bounds the memory of the likelihood and its derivatives. With per-firm
        # draws the buffer holds u, then the log-weights, then their exponentials; a shorter last
        # group gets an array of its own size, so every row reduction sees a dense array of the
        # group's size.
        per_firm = ndims(halton) == 2
        u_row = per_firm ? nothing :      # shared draws: one row of D, transformed once for all groups
                reshape(mypan_gen_udraws_msle(model.ineff, ineff_vals, P.(halton), c, P), 1, D)
        firm_chunk_size = cld(N, chunks)
        w_buf = similar(halton, P, firm_chunk_size, D)
        total_nll = P(0)
        for ci in 1:chunks
            f_start = (ci - 1) * firm_chunk_size + 1
            f_end = min(ci * firm_chunk_size, N)
            f_start > f_end && continue
            cN = f_end - f_start + 1
            A_c = @view A[f_start:f_end]
            B_c = @view B[f_start:f_end]
            C_c = @view C[f_start:f_end]
            lc = log_const isa AbstractVector ? (@view log_const[f_start:f_end]) : log_const
            w_c = cN == firm_chunk_size ? w_buf : similar(halton, P, cN, D)
            if per_firm
                mypan_gen_udraws_msle!(w_c, model.ineff, ineff_vals, (@view halton[f_start:f_end, :]), c, P)
                u_c = w_c
            else
                u_c = u_row
            end
            # log f_v for every firm and draw of the group (the within-firm quadratic form in u)
            w_c .= P(-0.5) .* (A_c .+ two_s .* B_c .* u_c .+ C_c .* u_c .^ 2) .* inv_sigma_v_sq .+ lc
            log_likes = _logsumexp_rows_panel!(w_c) .- log(P(D))
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

With `chunks >= 2` the firms are processed one group at a time in work arrays of one group
(per-firm draws are converted and transformed group by group, shared draws once), so `chunks`
bounds the memory of the likelihood and its derivatives; `chunks <= 1` processes all firms at once.
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
    h_tilde = compute_h_tilde(spec.z_raw, p, idx, spec.offsets)
    A, B, C = _compute_panel_ABC(epsilon_tilde, h_tilde, spec.offsets, N)
    # Move A, B, C to GPU if data lives there (halton is CuArray when GPU=true)
    A = _to_device_panel(A, halton)
    B = _to_device_panel(B, halton)
    C = _to_device_panel(C, halton)

    # 3. Extract noise/inefficiency parameters
    noise_vals = get_panel_noise_vals(model.noise, p, idx, c)
    ineff_vals = get_panel_ineff_vals(model.ineff, p, idx, c)

    # 4. MCI transformation: u = g(t, scale), Jacobian |g'(t, scale)|
    scale = mypan_get_mci_scale(model.ineff, ineff_vals)
    trans, jacob = resolve_panel_transformation(spec.transformation, model)

    # 6. Pre-compute log-density constants
    inv_sigma_v_sq = noise_vals.inv_sigma_v_sq
    log_sigma_v_sq = noise_vals.log_sigma_v_sq
    log_const = P(-0.5) .* spec.Tm1 .* (log(P(2π)) .+ log_sigma_v_sq)

    frontier_sign = P(spec.sign)
    two_s = P(2) * frontier_sign

    if chunks <= 1
        halton_P = P.(halton)
        u_draws = trans.(halton_P, scale)
        u_draws = min.(u_draws, inv(P(c.clamp_lo)))                  # clamp upper

        # 5. Log-PDF of inefficiency + log-Jacobian
        log_fu = mypan_logpdf_ineff(model.ineff, u_draws, ineff_vals, P(c.clamp_lo))
        log_J  = log.(max.(jacob.(halton_P, scale), P(c.clamp_lo)))
        log_fu_J = log_fu .+ log_J

        # Reshape for broadcasting: 1×D (shared) or N×D (multiRand)
        log_fu_J_row = ndims(log_fu_J) == 1 ? reshape(log_fu_J, 1, D) : log_fu_J
        u_row = ndims(u_draws) == 1 ? reshape(u_draws, 1, D) : u_draws

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
        # As in _panel_nll_msle. With per-firm draws a second buffer holds log f_u, then
        # log f_u + log|J|; the first holds u, then log|J|, then u again, so that log|J| is a
        # finished array before it is added, as in the chunks <= 1 branch.
        per_firm = ndims(halton) == 2
        if !per_firm                     # shared draws: one row of D, transformed once
            halton_P = P.(halton)
            u_draws = trans.(halton_P, scale)
            u_draws = min.(u_draws, inv(P(c.clamp_lo)))
            log_fu = mypan_logpdf_ineff(model.ineff, u_draws, ineff_vals, P(c.clamp_lo))
            log_J  = log.(max.(jacob.(halton_P, scale), P(c.clamp_lo)))
            lfj_row = reshape(log_fu .+ log_J, 1, D)
            u_row   = reshape(u_draws, 1, D)
        end
        firm_chunk_size = cld(N, chunks)
        w_buf = similar(halton, P, firm_chunk_size, D)
        f_buf = per_firm ? similar(halton, P, firm_chunk_size, D) : nothing
        total_nll = P(0)
        for ci in 1:chunks
            f_start = (ci - 1) * firm_chunk_size + 1
            f_end = min(ci * firm_chunk_size, N)
            f_start > f_end && continue
            cN = f_end - f_start + 1
            A_c = @view A[f_start:f_end]
            B_c = @view B[f_start:f_end]
            C_c = @view C[f_start:f_end]
            lc = log_const isa AbstractVector ? (@view log_const[f_start:f_end]) : log_const
            w_c = cN == firm_chunk_size ? w_buf : similar(halton, P, cN, D)
            if per_firm
                hc  = @view halton[f_start:f_end, :]
                f_c = @view f_buf[1:cN, :]
                w_c .= trans.(P.(hc), scale)
                w_c .= min.(w_c, inv(P(c.clamp_lo)))                                    # u
                mypan_logpdf_ineff!(f_c, model.ineff, w_c, ineff_vals, P(c.clamp_lo))   # log f_u
                w_c .= log.(max.(jacob.(P.(hc), scale), P(c.clamp_lo)))                # log|J|
                f_c .= f_c .+ w_c                                                       # log f_u + log|J|
                w_c .= trans.(P.(hc), scale)                                            # u again
                w_c .= min.(w_c, inv(P(c.clamp_lo)))
                u_c = w_c
            else
                u_c = u_row
                f_c = lfj_row
            end
            # log f_v + log f_u + log|J| for every firm and draw of the group
            w_c .= P(-0.5) .* (A_c .+ two_s .* B_c .* u_c .+ C_c .* u_c .^ 2) .* inv_sigma_v_sq .+ lc .+ f_c
            log_likes = _logsumexp_rows_panel!(w_c) .- log(P(D))
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

    for i in 1:N
        s, e = spec.offsets[i] + 1, spec.offsets[i + 1]
        jlms[s:e] .= h_raw[s:e] .* E_u_firm[i]
    end
    return jlms
end

# Observation-level BC = E[exp(-h_it u_i*) | data_i] for firms f_start:f_end, from the group's arrays.
function _compute_bc_obs!(bc, log_w, log_denom, h_raw, u_c, shared::Bool, spec, f_start, f_end)
    D = size(log_w, 2)
    for i in f_start:f_end
        r = i - f_start + 1
        s, e = spec.offsets[i] + 1, spec.offsets[i + 1]
        Ti = e - s + 1
        h_firm = reshape(h_raw[s:e], Ti, 1)
        u_firm = shared ? u_c : reshape(u_c[r, :], 1, D)
        log_w_firm = reshape(log_w[r, :], 1, D)
        log_bc_w_firm = .-h_firm .* u_firm .+ log_w_firm
        log_bc_num = logsumexp_rows_panel(log_bc_w_firm)
        bc[s:e] .= exp.(log_bc_num .- log_denom[r])
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

Firms are processed in `chunks` groups, so the memory of the computation is bounded by `chunks`;
for a GPU fit each group's draws are copied to the host, where the indices are computed.
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

    # 1. Residuals
    epsilon_tilde = y_tilde .- sum(P(p[idx.beta[j]]) .* (@view x_tilde[:, j]) for j in 1:K)

    # 2. h_raw and h_tilde (need both: h_tilde for quadratic form, h_raw for obs expansion)
    h_raw, h_tilde = compute_h_raw_and_tilde(z_raw, p, idx, spec.offsets)

    # 3. Per-firm quadratic sums A, B, C
    A, B, C = _compute_panel_ABC(epsilon_tilde, h_tilde, spec.offsets, N)

    # 4. Noise/inefficiency parameters and u draws (dispatched by distribution)
    noise_vals = get_panel_noise_vals(model.noise, p, idx, c)
    ineff_vals = get_panel_ineff_vals(model.ineff, p, idx, c)
    inv_sigma_v_sq = noise_vals.inv_sigma_v_sq
    log_sigma_v_sq = noise_vals.log_sigma_v_sq
    Tm1_cpu        = _to_cpu(spec.Tm1)
    log_const      = P(-0.5) .* Tm1_cpu .* (log(P(2π)) .+ log_sigma_v_sq)
    frontier_sign  = P(spec.sign)

    two_s   = P(2) * frontier_sign
    shared = ndims(spec.draws) == 1
    D = shared ? length(spec.draws) : size(spec.draws, 2)
    if spec.method != :MSLE
        scale = mypan_get_mci_scale(model.ineff, ineff_vals)
        trans, jacob = resolve_panel_transformation(spec.transformation, model)
    end
    # u, the MCI weight term log f_u + log|J|, and log u for a set of draw rows
    function draw_terms(h)
        halton_P = P.(h)
        if spec.method == :MSLE
            u = mypan_gen_udraws_msle(model.ineff, ineff_vals, halton_P, c, P)
            lx = nothing
        else
            u = min.(trans.(halton_P, scale), inv(P(c.clamp_lo)))
            log_fu = mypan_logpdf_ineff(model.ineff, u, ineff_vals, P(c.clamp_lo))
            log_J  = log.(max.(jacob.(halton_P, scale), P(c.clamp_lo)))
            lx = log_fu .+ log_J
            lx = ndims(lx) == 1 ? reshape(lx, 1, D) : lx
        end
        u_row = ndims(u) == 1 ? reshape(u, 1, D) : u
        return u_row, lx, log.(max.(u_row, P(c.clamp_lo)))
    end
    shared_terms = shared ? draw_terms(_to_cpu(spec.draws)) : nothing
    n_groups = max(chunks, 1)
    firm_chunk_size = cld(N, n_groups)
    E_u_firm = nothing
    bc = nothing
    for ci in 1:n_groups
        f_start = (ci - 1) * firm_chunk_size + 1
        f_end = min(ci * firm_chunk_size, N)
        f_start > f_end && continue
        u_c, lx_c, lu_c = shared ? shared_terms : draw_terms(_panel_draw_rows(spec.draws, f_start, f_end))
        A_c = @view A[f_start:f_end]
        B_c = @view B[f_start:f_end]
        C_c = @view C[f_start:f_end]
        quad  = A_c .+ two_s .* B_c .* u_c .+ C_c .* u_c .^ 2
        log_w = P(-0.5) .* quad .* inv_sigma_v_sq .+ (@view log_const[f_start:f_end])
        lx_c === nothing || (log_w = log_w .+ lx_c)
        log_denom    = logsumexp_rows_panel(log_w)
        log_jlms_num = logsumexp_rows_panel(lu_c .+ log_w)
        if E_u_firm === nothing                     # outputs in the element type of the log-weights
            E_u_firm = Vector{eltype(log_w)}(undef, N)
            bc = Vector{eltype(log_w)}(undef, length(h_raw))
        end
        E_u_firm[f_start:f_end] .= exp.(log_jlms_num .- log_denom)
        _compute_bc_obs!(bc, log_w, log_denom, h_raw, u_c, shared, spec, f_start, f_end)
    end
    jlms = _expand_jlms_obs(E_u_firm, h_raw, spec)
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

"""E(u_it) for TruncatedNormal: h(z) * σ_u * m(Λ), m(Λ) = Λ + φ(Λ)/Φ(Λ), Λ = μ/σ_u. Below
Λ = -4, m is `SFNumerics.sf_tn_mean`."""
function _panel_uncondU_trun(coef, idx, z_obs)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    mu = coef[idx.mu]
    sigma_u = clamp(exp(oftype(z_delta, 0.5) * coef[idx.ln_sigma_u_sq]),
                    oftype(z_delta, 1e-12), oftype(z_delta, 1e12))
    Lambda = mu / sigma_u
    # Φ(Λ) formed from 1 + erf loses digits as Λ falls
    # (E(u) off by up to 2e-10 relative on [-4.5, -4], 4e-9 near Λ = -5 and 1.2e-3 near -7)
    # and, floored at 1e-15, gives E(u) of the wrong sign from Λ ≈ -7.93 (E(u) → μ below -10),
    # for every observation, since Λ is a scalar in this model. Below Λ = -4, E(u*) = σ_u·m(Λ)
    # with the stable m of SFNumerics.sf_tn_mean; above, the expression below, bit for bit.
    Lambda < -4 && return h_val * sigma_u * sf_tn_mean(Lambda)
    phi_Lambda = exp(oftype(z_delta, -0.5) * Lambda^2) / sqrt(oftype(z_delta, 2π))
    Phi_Lambda = max(oftype(z_delta, 0.5) * (1 + erf(Lambda / sqrt(oftype(z_delta, 2)))),
                     oftype(z_delta, 1e-15))
    return h_val * sigma_u * (Lambda + phi_Lambda / Phi_Lambda)
end

"""E(u_it) for Exponential (σᵤ²=Var(u*)): h(z) * √σᵤ²"""
function _panel_uncondU_expo(coef, idx, z_obs)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    sigma_sq = clamp(exp(coef[idx.ln_sigma_u_sq]),
                     oftype(z_delta, 1e-12), oftype(z_delta, 1e12))
    return h_val * sqrt(sigma_sq)
end

"""E(u_it) for Weibull(λ, k): h(z) * λ * Γ(1 + 1/k). Clamps use the kernel
constants `c` so the marginal effect uses the same effective parameters as the NLL."""
function _panel_uncondU_weibull(coef, idx, z_obs, c)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    lambda = clamp(exp(coef[idx.ln_lambda]),
                   oftype(z_delta, c.sigma_floor), oftype(z_delta, c.sigma_ceil))
    k_val = clamp(exp(coef[idx.ln_k]),
                  oftype(z_delta, c.k_floor), oftype(z_delta, c.weibull_k_ceil))
    return h_val * lambda * exp(loggamma(1 + 1 / k_val))
end

"""E(u_it) for Lognormal(μ, σ): h(z) * exp(μ + σ²/2)"""
function _panel_uncondU_lognorm(coef, idx, z_obs)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    mu = coef[idx.mu]
    sigma_sq = clamp(exp(coef[idx.ln_sigma_u_sq]),
                     oftype(z_delta, 1e-24), oftype(z_delta, 1e24))
    return h_val * exp(mu + sigma_sq / 2)
end

"""E(u_it) for Lomax(α, λ): h(z) * λ / (α - 1); undefined (NaN) for α ≤ 1. Clamps use the kernel
constants `c`, so the marginal effect uses the same effective parameters as the likelihood."""
function _panel_uncondU_lomax(coef, idx, z_obs, c)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    lambda = clamp(exp(coef[idx.ln_lambda]),
                   oftype(z_delta, c.xm_floor), oftype(z_delta, c.xm_ceil))
    alpha = clamp(exp(coef[idx.ln_alpha]),
                  oftype(z_delta, c.alpha_floor), oftype(z_delta, c.alpha_ceil))
    # E(u_it) = h(z)·λ/(α − 1) does not exist for α ≤ 1 (α's floor of 0.1 lies below 1, so this
    # branch is reachable). The h(z) factor carries NaN into the value and every partial, so the
    # marginal effect reads NaN, never a finite number. α is built from the coefficients and
    # constants only, so its partials are zero and the comparison is decided by its value, also
    # at α = 1.
    if alpha <= 1
        return h_val * oftype(z_delta, NaN)
    end
    return h_val * lambda / (alpha - 1)
end

"""E(u_it) for Rayleigh(σ): h(z) * σ * √(π/2)"""
function _panel_uncondU_rayleigh(coef, idx, z_obs)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    sigma = clamp(exp(oftype(z_delta, 0.5) * coef[idx.ln_sigma_u_sq]),
                  oftype(z_delta, 1e-12), oftype(z_delta, 1e12))
    return h_val * sigma * sqrt(oftype(z_delta, π) / 2)
end

"""E(u_it) for Gamma(k, θ): h(z) * k * θ. Clamps use the kernel constants `c`
so the marginal effect uses the same effective parameters as the NLL."""
function _panel_uncondU_gamma(coef, idx, z_obs, c)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    k_val = clamp(exp(coef[idx.ln_k]),
                  oftype(z_delta, c.k_floor), oftype(z_delta, c.gamma_k_ceil))
    theta = clamp(exp(coef[idx.ln_theta]),
                  oftype(z_delta, c.theta_floor), oftype(z_delta, c.theta_ceil))
    return h_val * k_val * theta
end

"""E(u_it) for Erlang(M, θ): h(z) * M * θ. Clamps use the kernel constants `c`
so the marginal effect uses the same effective parameters as the NLL."""
function _panel_uncondU_erlang(coef, idx, z_obs, c, M)
    L = length(idx.delta)
    z_delta = sum(coef[idx.delta[j]] * z_obs[j] for j in 1:L)
    h_val = exp(z_delta)
    theta = clamp(exp(coef[idx.ln_theta]),
                  oftype(z_delta, c.theta_floor), oftype(z_delta, c.theta_ceil))
    return h_val * M * theta
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
# Weibull/Lomax/Gamma/Erlang marginal helpers take the kernel constants `c` so their
# clamps stay aligned with the NLL. Build `c` once here and close over it.
function get_panel_marg(::Weibull_Panel, spec::_PanelInternalSpec, coef::AbstractVector)
    c = make_panel_constants(spec.model, Float64, spec.T_max)
    _panel_marg_generic((cf, ix, z) -> _panel_uncondU_weibull(cf, ix, z, c), spec, coef)
end
get_panel_marg(::Lognormal_Panel, spec::_PanelInternalSpec, coef::AbstractVector) =
    _panel_marg_generic(_panel_uncondU_lognorm, spec, coef)
function get_panel_marg(::Lomax_Panel, spec::_PanelInternalSpec, coef::AbstractVector)
    c = make_panel_constants(spec.model, Float64, spec.T_max)
    _panel_marg_generic((cf, ix, z) -> _panel_uncondU_lomax(cf, ix, z, c), spec, coef)
end
get_panel_marg(::Rayleigh_Panel, spec::_PanelInternalSpec, coef::AbstractVector) =
    _panel_marg_generic(_panel_uncondU_rayleigh, spec, coef)
function get_panel_marg(::Gamma_Panel, spec::_PanelInternalSpec, coef::AbstractVector)
    c = make_panel_constants(spec.model, Float64, spec.T_max)
    _panel_marg_generic((cf, ix, z) -> _panel_uncondU_gamma(cf, ix, z, c), spec, coef)
end
function get_panel_marg(::Erlang_Panel{M}, spec::_PanelInternalSpec, coef::AbstractVector) where {M}
    c = make_panel_constants(spec.model, Float64, spec.T_max)
    _panel_marg_generic((cf, ix, z) -> _panel_uncondU_erlang(cf, ix, z, c, M), spec, coef)
end

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


# ---------------------------------------------------------------------------
# The boundary verdict. The get_panel_*_vals functions return CLAMPED scalars, and `clamp`
# returns the bound itself when it binds, so the verdict tests the values the likelihood uses
# against the bounds (within SF_BOUND_RTOL, src/sf_redflag.jl), formed as the value functions
# form them (P = Float64 at the estimate). Beyond a bound the likelihood is flat in that
# parameter, and its Hessian row is then exactly zero. The μ of the truncated normal and the
# lognormal and the scaling function have no bound and are not listed.

# One bounded quantity (each is one value for all observations, and the likelihood clamps it):
# its name and label, the coefficient it is formed from (exp(θ)), its bounds, and where the
# value functions return its clamped value (`vals`: :noise or :ineff, and its `field`).
_panel_bound_item(parameter, label, coefficient, keyword, block, lo, hi, vals, field) =
    (parameter = parameter, label = label, coefficient = coefficient, keyword = keyword,
     block = Vector{Int}(vcat(block)), lo = Float64(lo), hi = Float64(hi), scalar = true, form = :exp,
     offset = 0.0, movable = true, vals = vals, field = field)

_panel_noise_bound_items(::NormalNoise_Panel, idx, c) =
    (_panel_bound_item(:sigma_v_sq, "σᵥ²", "ln_σᵥ²", :ln_sigma_v_sq, idx.ln_sigma_v_sq,
                       Float64(c.sigma_floor)^2, Float64(c.sigma_ceil)^2, :noise, :sigma_v_sq),)

_panel_ineff_bound_items(::Union{HalfNormal_Panel, TruncatedNormal_Panel}, idx, c) =
    (_panel_bound_item(:sigma_u_sq, "σᵤ²", "ln_σᵤ²", :ln_sigma_u_sq, idx.ln_sigma_u_sq,
                       Float64(c.sigma_floor)^2, Float64(c.sigma_ceil)^2, :ineff, :sigma_u_sq),)
_panel_ineff_bound_items(::Union{Lognormal_Panel, Rayleigh_Panel}, idx, c) =
    (_panel_bound_item(:sigma_u_sq, "σᵤ²", "ln_σᵤ²", :ln_sigma_u_sq, idx.ln_sigma_u_sq,
                       Float64(c.sigma_floor)^2, Float64(c.sigma_ceil)^2, :ineff, :sigma_sq),)
_panel_ineff_bound_items(::Exponential_Panel, idx, c) =
    (_panel_bound_item(:sigma_u_sq, "σᵤ²", "ln_σᵤ²", :ln_sigma_u_sq, idx.ln_sigma_u_sq, c.sigma_floor,
                       c.sigma_ceil, :ineff, :sigma_sq),)
_panel_ineff_bound_items(::Weibull_Panel, idx, c) =
    (_panel_bound_item(:lambda, "λ", "ln_λ", :ln_lambda, idx.ln_lambda, c.sigma_floor, c.sigma_ceil, :ineff,
                       :lambda),
     _panel_bound_item(:k, "k", "ln_k", :ln_k, idx.ln_k, c.k_floor, c.weibull_k_ceil, :ineff, :k))
_panel_ineff_bound_items(::Lomax_Panel, idx, c) =
    (_panel_bound_item(:lambda, "λ", "ln_λ", :ln_lambda, idx.ln_lambda, c.xm_floor, c.xm_ceil, :ineff, :lambda),
     _panel_bound_item(:alpha, "α", "ln_α", :ln_alpha, idx.ln_alpha, c.alpha_floor, c.alpha_ceil, :ineff, :alpha))
_panel_ineff_bound_items(::Gamma_Panel, idx, c) =
    (_panel_bound_item(:k, "k", "ln_k", :ln_k, idx.ln_k, c.k_floor, c.gamma_k_ceil, :ineff, :k),
     _panel_bound_item(:theta, "θ", "ln_θ", :ln_theta, idx.ln_theta, c.theta_floor, c.theta_ceil, :ineff, :theta))
_panel_ineff_bound_items(::Erlang_Panel, idx, c) =
    (_panel_bound_item(:theta, "θ", "ln_θ", :ln_theta, idx.ln_theta, c.theta_floor, c.theta_ceil, :ineff, :theta),)

"""
    _panel_bound_list(ispec::_PanelInternalSpec) -> Vector

The quantities the panel likelihood clamps to numerical bounds, σᵥ² first and then the
inefficiency's: one NamedTuple per quantity with `parameter`, `label`, `coefficient`, `keyword`
(the `sfmodel_init` keyword of its coefficient), `block` (its index), `lo` and `hi` (formed from
the model's constants as the value functions form them), `scalar`, `form`, `offset`, `movable`,
and `vals` and `field` (where the value functions return its clamped value).
"""
function _panel_bound_list(ispec::_PanelInternalSpec)
    c = ispec.constants; idx = ispec.idx; m = ispec.model
    return [_panel_noise_bound_items(m.noise, idx, c)..., _panel_ineff_bound_items(m.ineff, idx, c)...]
end

"""
    _panel_bound_verdict(ispec::_PanelInternalSpec, p, nobs) -> Vector{SFBoundEntry}

The boundary verdict at the coefficient vector `p`: the entries of every quantity of
`_panel_bound_list(ispec)` at or beyond its bound, from the clamped values of
`get_panel_noise_vals` and `get_panel_ineff_vals`, each counted over `nobs` observations; empty
when there is none.
"""
function _panel_bound_verdict(ispec::_PanelInternalSpec, p::AbstractVector, nobs::Integer)
    c = ispec.constants; idx = ispec.idx; m = ispec.model
    vals = (noise = get_panel_noise_vals(m.noise, p, idx, c), ineff = get_panel_ineff_vals(m.ineff, p, idx, c))
    bc = SFBoundEntry[]
    for it in _panel_bound_list(ispec)
        sf_bound_entries!(bc, getproperty(getproperty(vals, it.vals), it.field), it.lo, it.hi, nobs;
                          parameter = it.parameter, label = it.label, coefficient = it.coefficient,
                          scalar = it.scalar)
    end
    return bc
end

# ============================================================================
# Section 12: Variance-Covariance Matrix & Hessian
# ============================================================================

"""
    _panel_invert_hessian(H; at_clamp=false, message=true) -> NamedTuple

Invert Hessian matrix for variance-covariance computation.
Returns (var_cov_matrix=..., redflag=Int, hessian_flag=Int). It applies the rule shared by
all four backends, `sf_classify_hessian` (src/sf_redflag.jl): 1 when the
Hessian is not invertible or is numerically singular (all-NaN matrix); 2 for a
non-positive diagonal; otherwise 1 when the Hessian is not positive definite (the
matrix is kept as computed); 1 when `at_clamp`, whatever the Hessian says (also over
2); else 0.
"""
_panel_invert_hessian(H::AbstractMatrix{<:Real}; at_clamp::Bool=false, message::Bool=true) =
    sf_classify_hessian(H; at_clamp=at_clamp, message=message)

# NaN-safe standard errors from a var-cov matrix. A non-positive-diagonal (redflag>0) vcov
# would make `sqrt` throw a DomainError, so those entries become NaN instead — a degenerate
# fit still returns its coefficients/loglik/jlms/bc, and `stderror(r)` stays NaN-consistent
# with `sqrt.(diag(vcov(r)))`. The test is `<= 0`, so an exactly zero diagonal (redflag 2)
# also gives NaN, not 0.
_stderr_from_vcov(V::AbstractMatrix) = map(d -> d <= 0 ? oftype(d, NaN) : sqrt(d), diag(V))

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
    aux[1, :] = ["σᵤ²", exp(coef[idx.ln_sigma_u_sq]), exp(coef[idx.ln_sigma_u_sq]) * stddev[idx.ln_sigma_u_sq]]
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
    aux[1, :] = ["σᵤ²", exp(coef[idx.ln_sigma_u_sq]), exp(coef[idx.ln_sigma_u_sq]) * stddev[idx.ln_sigma_u_sq]]
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
    aux[1, :] = ["σᵤ²", exp(coef[idx.ln_sigma_u_sq]), exp(coef[idx.ln_sigma_u_sq]) * stddev[idx.ln_sigma_u_sq]]
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

function _panel_aux_table(::Erlang_Panel, coef, idx, stddev)
    aux = Matrix{Any}(undef, 2, 3)
    aux[1, :] = ["θ", exp(coef[idx.ln_theta]), exp(coef[idx.ln_theta]) * stddev[idx.ln_theta]]
    aux[2, :] = ["σᵥ²", exp(coef[idx.ln_sigma_v_sq]), exp(coef[idx.ln_sigma_v_sq]) * stddev[idx.ln_sigma_v_sq]]
    return aux
end

"""
    panel_print_table(spec, coef, var_cov_matrix; optim_result=nothing, loglik=nothing, integration=nothing,
                      table_format=:text)

Print formatted estimation results table for panel SF model. `loglik` is the log-likelihood
to print, `-optim_result.minimum` when not given; the fit passes the value at the reported
coefficients.

`integration` is the result of the integration check (`result.integration_check`); when it is
given, the line "Integration check: …" is printed after "Converged:".
"""
function panel_print_table(spec::_PanelInternalSpec{T}, coef::AbstractVector,
                            var_cov_matrix::AbstractMatrix;
                            optim_result=nothing,
                            loglik=nothing,
                            integration=nothing,
                            table_format::Symbol=:text) where {T}

    nofpara = length(coef)
    N_total = sum(spec.T_periods)

    # Compute statistics (asymptotic Normal, standard for MLE)
    stddev = _stderr_from_vcov(var_cov_matrix)  # non-PD vcov ⇒ NaN SEs (no DomainError); keeps stderror(r) NaN-consistent with sqrt.(diag(vcov(r))).
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
    T_min, T_max_val = extrema(spec.T_periods)
    if T_min == T_max_val
        print("Time periods (T): "); printstyled(T_min; color=:yellow); println()
    else
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
        if integration !== nothing
            print("Integration check: ")
            printstyled(sf_integration_line(integration);
                        color = integration.status in (:OK, :not_available) ? :yellow : :red)
            println()
        end
        # The fit passes the log-likelihood at the reported coefficients.
        ll_show = loglik === nothing ? -optim_result.minimum : loglik
        print("Log-likelihood: "); printstyled(round(ll_show; digits=5); color=:yellow); println()
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
_panel_ineff_varnames(::Exponential_Panel)      = ["ln_σᵤ²", "ln_σᵥ²"]
_panel_ineff_varnames(::Weibull_Panel)          = ["ln_λ", "ln_k", "ln_σᵥ²"]
_panel_ineff_varnames(::Lognormal_Panel)        = ["μ", "ln_σᵤ²", "ln_σᵥ²"]
_panel_ineff_varnames(::Lomax_Panel)            = ["ln_λ", "ln_α", "ln_σᵥ²"]
_panel_ineff_varnames(::Rayleigh_Panel)         = ["ln_σᵤ²", "ln_σᵥ²"]
_panel_ineff_varnames(::Gamma_Panel)            = ["ln_k", "ln_θ", "ln_σᵥ²"]
_panel_ineff_varnames(::Erlang_Panel)           = ["ln_θ", "ln_σᵥ²"]

"""Return distribution-specific equation names and indices."""
function _panel_ineff_eqinfo(::HalfNormal_Panel, K, L)
    return ["ln_σᵤ²", "ln_σᵥ²"], [K + L + 1, K + L + 2]
end
function _panel_ineff_eqinfo(::TruncatedNormal_Panel, K, L)
    return ["μ", "ln_σᵤ²", "ln_σᵥ²"], [K + L + 1, K + L + 2, K + L + 3]
end
function _panel_ineff_eqinfo(::Exponential_Panel, K, L)
    return ["ln_σᵤ²", "ln_σᵥ²"], [K + L + 1, K + L + 2]
end
function _panel_ineff_eqinfo(::Weibull_Panel, K, L)
    return ["ln_λ", "ln_k", "ln_σᵥ²"], [K + L + 1, K + L + 2, K + L + 3]
end
function _panel_ineff_eqinfo(::Lognormal_Panel, K, L)
    return ["μ", "ln_σᵤ²", "ln_σᵥ²"], [K + L + 1, K + L + 2, K + L + 3]
end
function _panel_ineff_eqinfo(::Lomax_Panel, K, L)
    return ["ln_λ", "ln_α", "ln_σᵥ²"], [K + L + 1, K + L + 2, K + L + 3]
end
function _panel_ineff_eqinfo(::Rayleigh_Panel, K, L)
    return ["ln_σᵤ²", "ln_σᵥ²"], [K + L + 1, K + L + 2]
end
function _panel_ineff_eqinfo(::Gamma_Panel, K, L)
    return ["ln_k", "ln_θ", "ln_σᵥ²"], [K + L + 1, K + L + 2, K + L + 3]
end
function _panel_ineff_eqinfo(::Erlang_Panel, K, L)
    return ["ln_θ", "ln_σᵥ²"], [K + L + 1, K + L + 2]
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

# Marker types for DSL-style model specification using DataFrames.

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
                              type::Symbol=:prod,
                              eqnames::Union{Nothing, Vector{String}}=nothing,
                              eq_indices::Union{Nothing, Vector{Int}}=nothing)
    df = data_spec.df
    depvar, frontier, frontier_names = _panel_dsl_extract(df, depvar_spec, frontier_spec)
    zvar, zvar_names = _panel_dsl_extract_zvar(df, zvar_spec)
    id_col = df[!, id_spec.name]
    model_temp = _build_panel_model(noise, ineff)
    varnames = vcat(frontier_names, zvar_names, _panel_ineff_varnames(model_temp.ineff))

    # `varnames` comes from the DataFrame column names, but `eqnames`/`eq_indices` are the
    # caller's and have no DSL macro — the unified DSL takes them as keywords and forwards
    # them here, so the DSL honours them as the keyword form does (without them it would
    # fall back to the panel defaults).
    return sfmodel_panel_spec(; depvar=depvar, frontier=frontier, zvar=zvar,
                                 id=id_col, noise=noise, ineff=ineff, type=type,
                                 varnames=varnames,
                                 eqnames=eqnames, eq_indices=eq_indices)
end

"""
    sfmodel_panel_spec(@useData(df), @depvar(y), @frontier(x1,x2), @id(id); ...)

DSL form without @zvar. NOT SUPPORTED: a constant scaling h(z)=exp(δ₀) is
unidentified in the Wang-Ho panel model (within-group demeaning eliminates it),
so this variant errors at spec time. Provide `@zvar(...)` with a
regressor that varies within firms.
"""
function sfmodel_panel_spec(data_spec::UseDataSpec, depvar_spec::DepvarSpec,
                              frontier_spec::FrontierSpec, id_spec::IdSpec;
                              noise::Symbol=:Normal,
                              ineff::Symbol=:HalfNormal,
                              type::Symbol=:prod)
    error("The Wang-Ho panel model requires `@zvar(...)` with within-firm variation " *
          "for the scaling function h(z)=exp(z'δ). A spec without @zvar uses a constant " *
          "scaling h(z)=exp(δ₀), which is unidentified: within-group demeaning eliminates " *
          "it, the likelihood is flat in δ₀ and the inefficiency parameters, and jlms/bc " *
          "carry no information from the data. Add `@zvar(...)` with a regressor " *
          "that varies within firms.")
end

# ============================================================================
# Section 15: sfmodel_panel_spec()
# ============================================================================

"""
    sfmodel_panel_spec(; depvar, frontier, zvar, id, ...)

Construct panel model specification. Panel structure is specified via the `id` column.

# Panel structure
- `id`: A column identifying the panel unit for each observation.
  Can be any type (integers, strings, symbols, etc.). Data must be grouped by id
  (all rows for the same unit are contiguous). The number of periods per unit is
  inferred from contiguous group lengths.
  For balanced panels: `id = repeat(1:N, inner=T)`.

# Arguments
- `depvar`: Response vector, length NT_total
- `frontier`: Design matrix, NT_total × K (do NOT include a constant/intercept — it is unidentifiable after within-demeaning)
- `zvar`: Scaling function variables, NT_total × L
- `T_periods=nothing`: Deprecated; raises an error if provided. Use `id` instead.
- `id`: Unit identifier column (required)
- `noise::Symbol=:Normal`: Noise distribution
- `ineff::Symbol=:HalfNormal`: Inefficiency distribution (:HalfNormal, :TruncatedNormal, :Exponential, :Weibull, :Lognormal, :Lomax, :Rayleigh, :Gamma, :Erlang2, :Erlang3, :Erlang4)
- `type::Symbol=:prod`: Frontier type (`:prod`, `:production`, or `:cost`)
- `varnames=nothing`: Variable names (auto-generated if not provided)
- `eqnames=nothing`: Equation names (auto-generated if not provided)
- `eq_indices=nothing`: Equation boundary indices (auto-generated if not provided)

# Returns
`PanelModelSpec{T}` model specification object.

# Examples
```julia
# Balanced panel (use id = repeat(1:N, inner=T)):
spec = sfmodel_panel_spec(
    depvar = y, frontier = X, zvar = Z,
    id = repeat(1:N, inner=T)
)

# Unbalanced panel:
spec = sfmodel_panel_spec(
    depvar = df.y, frontier = hcat(df.x1, df.x2),
    zvar = hcat(df.z1, df.z2),
    id = df.firm_name
)
```
"""
function sfmodel_panel_spec(; depvar, frontier, zvar=nothing,
                              T_periods::Union{Int,Nothing}=nothing,
                              id=nothing,
                              noise::Symbol=:Normal,
                              ineff::Symbol=:HalfNormal,
                              type::Symbol=:prod,
                              varnames::Union{Nothing, Vector{String}}=nothing,
                              eqnames::Union{Nothing, Vector{String}}=nothing,
                              eq_indices::Union{Nothing, Vector{Int}}=nothing)

    # A panel spec without `zvar` implies a constant scaling h(z)=exp(δ₀),
    # which is unidentified in the Wang-Ho model (within-group demeaning eliminates
    # it). Give the informative identification error instead of the generic
    # "Unsupported matrix input type: Nothing" that `_to_matrix_panel(nothing)`
    # would otherwise raise (this also covers the unified `sfmodel_spec` keyword
    # path, which forwards `zvar=nothing`).
    isnothing(zvar) && error(
        "The Wang-Ho panel model requires `zvar` with within-firm variation for the " *
        "scaling function h(z)=exp(z'δ). A panel spec without `zvar` uses a constant " *
        "scaling h(z)=exp(δ₀), which is unidentified: within-group demeaning eliminates " *
        "it, the likelihood is flat in δ₀ and the inefficiency parameters, and jlms/bc " *
        "carry no information from the data. Provide `zvar=...` with a regressor " *
        "that varies within firms.")

    # Normalize inputs
    depvar_norm = _to_vector_panel(depvar)
    frontier_norm = _to_matrix_panel(frontier)
    zvar_norm = _to_matrix_panel(zvar)

    T_el = eltype(depvar_norm)
    frontier_norm = T_el.(frontier_norm)
    zvar_norm = T_el.(zvar_norm)

    NT = length(depvar_norm)

    # --- Determine panel structure ---
    if !isnothing(T_periods)
        error("T_periods is no longer supported. Use id instead. For balanced panels: id = repeat(1:N, inner=T).")
    end
    if isnothing(id)
        error("Must specify id column for panel structure.")
    end

    id_vec = id isa AbstractVector ? id : error("`id` must be a vector, got $(typeof(id)).")
    length(id_vec) == NT || error("id column has $(length(id_vec)) elements, expected $NT (same as depvar).")
    N, T_vec, offsets = _compute_panel_structure(id_vec)
    firm_ids = unique(id_vec)  # first-appearance order (matches offsets)
    @assert all(t -> t >= 2, T_vec) "Wang-Ho within-transformation requires T_i >= 2 for all units. " *
        "Got minimum T_i = $(minimum(T_vec))."
    T_periods = T_vec
    T_max = maximum(T_vec)

    K = size(frontier_norm, 2)
    L = size(zvar_norm, 2)

    # Validate dimensions
    size(frontier_norm, 1) == NT || error("frontier has $(size(frontier_norm, 1)) rows, expected $NT.")
    size(zvar_norm, 1) == NT || error("zvar has $(size(zvar_norm, 1)) rows, expected $NT.")

    # Reject constant columns (unidentifiable after within-demeaning).
    _check_no_constant_column(frontier_norm, "frontier")
    _check_no_constant_column(zvar_norm, "zvar")   # incl. the DSL all-ones column
    # Scaling h(z)=exp(z'δ) is identified only if z varies WITHIN firms. If every
    # zvar column is constant within every firm (e.g. a time-invariant industry
    # dummy, or the no-@zvar ones column), the within-demeaned h̃ ≡ 0 for all δ, the
    # likelihood is flat in δ and the inefficiency parameters, and jlms/bc reduce to
    # uninformative prior means. Error at spec time rather than silently
    # return arbitrary scores.
    let z_has_within_variation = false
        for j in axes(zvar_norm, 2), i in 1:N
            s, e = offsets[i] + 1, offsets[i + 1]
            seg = @view zvar_norm[s:e, j]
            if any(x -> x != seg[1], seg)
                z_has_within_variation = true
                break
            end
        end
        z_has_within_variation || error(
            "`zvar` has no within-firm variation: every column is constant within each " *
            "firm, so the Wang-Ho scaling h(z)=exp(z'δ) is unidentified (within-group " *
            "demeaning eliminates it; the likelihood is flat and jlms/bc carry no " *
            "information from the data). Provide a zvar that varies within firms.")
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
                                 varnames_vec, eqnames_vec, eq_indices_vec,
                                 firm_ids)
end


# ============================================================================
# Section 16: sfmodel_panel_method()
# ============================================================================

"""
    sfmodel_panel_method(; method=:MSLE, transformation=nothing,
                           draws=nothing, n_draws=4096, multiRand=true,
                           GPU=false, chunks=10, recycling=false, distinct_Halton_length=nothing)

Specify the numerical estimation method for panel SF model.

# Arguments
- `method::Symbol=:MSLE`: Estimation method (`:MSLE` or `:MCI`)
- `transformation=nothing`: MCI transformation rule. MCI only; `method = :MSLE` refuses it. Under the
  default rule, MCI fits whose estimated shape lies in a range where that rule is unreliable (Gamma k,
  Weibull k, Lomax α below a threshold) print a warning (unless the fit passes `shape_warning = false`)
  and store the verdict in `result.shape_check`. Options:
  - `:logistic_1_rule` — `u = s · t/(1-t)` (default for HalfNormal, TruncatedNormal, Lognormal, Lomax)
  - `:logistic_2_rule` — `u = σ_u · (t/(1-t))²`
  - `:expo_rule` — `u = s · (-log(1-t))` (default for Exponential, Weibull, Rayleigh, Gamma, Erlang2/3/4)
  - `nothing` — use distribution-specific default
- `draws=nothing`: User-supplied draws in [0, 1); auto-generated Halton if `nothing`. The ROW COUNT
  decides how they are read, and `multiRand` is not part of it: a length-D vector, a 1×D row matrix or
  a D×1 column matrix is one sequence of D draws shared by every firm; an N×D matrix is one row of D
  draws per firm. Any other row count is an error. Supplying `draws` forces `multiRand=false` and makes
  `n_draws` irrelevant (D comes from the draws themselves). The fit keeps the draws for `predict()`. When
  the array already has the data's element type and lives where the data live (a host `Array`, or a
  `CuArray` with `GPU = true`), the fit uses the array itself, not a copy, so changing the array afterwards
  changes what `predict()` computes.
- `n_draws::Int=4096`: Number of Halton draws per firm (2^12; used only when `draws` is not provided). Use
  powers of two.
- `multiRand::Bool=true`: generation mode, used only when `draws` is not supplied. `true` gives observation i
  (firm i in panel models) the `n_draws` consecutive points (i − 1)·n_draws + 1, …, i·n_draws of one base-2
  Halton sequence of N × `n_draws` points, so no point is used twice unless `recycling` or the deprecated
  `distinct_Halton_length` is given; `false` gives every observation the same first `n_draws` points.
- `GPU::Bool=false`: Use GPU acceleration
- `chunks::Int=10`: Number of groups of observations (firms, for panel models) processed at a time (MCI/MSLE
  only). The memory of the likelihood, its derivatives and the efficiency indices is proportional to
  N × `n_draws` / `chunks`, on the CPU and on the GPU.
- `recycling=false`: reuse of Halton points across firms, for generated per-firm draws only
  (`multiRand = true`, no `draws`). `false` (default): no point is reused. `true`: the sequence is capped at
  L = 2^24 − 1 = 16,777,215 points and restarts from its first point when N × `n_draws` exceeds L; firm i
  receives the points at positions mod1((i − 1)·n_draws + j, L), j = 1, …, n_draws. An integer L with
  `n_draws` ≤ L ≤ 2^24 − 1 sets another cap. When N × `n_draws` exceeds L, firms i and i + L/g receive
  identical draws, where g is the greatest common divisor of `n_draws` and L; the fit warns when the sample
  has more than L/g firms. With `recycling = true` (L = 2^24 − 1, an odd number) any power of two for
  `n_draws` gives g = 1. With `draws` or `multiRand = false`, `recycling` is an error.
- `distinct_Halton_length=nothing`: deprecated; it gives a warning. Earlier `:MCI` and `:MSLE` scripts that set
  it still run and, with the same `n_draws`, get the draws they got before 2.0.0. To reproduce the draws of a
  fit of those versions that used the default length, give `recycling = 2^15 - 1` and that fit's `n_draws`
  (1,024 by default): the draws are identical whenever N × `n_draws` is at least 32,767. Use `recycling` for
  new work; giving both is an error.
"""
function sfmodel_panel_method(;
    method::Symbol=:MSLE,
    transformation::Union{Symbol,Nothing}=nothing,
    draws::Union{Nothing, AbstractVecOrMat{<:AbstractFloat}}=nothing,
    n_draws::Int=2^12,
    multiRand::Bool=true,
    GPU::Bool=false,
    chunks::Int=10,
    recycling::Union{Bool,Integer}=false,
    distinct_Halton_length::Union{Int,Nothing}=nothing)

    method in (:MSLE, :MCI) || error("method must be :MSLE or :MCI, got :$method")
    # Twin of the unified constructor's guard. `transformation` reparameterizes the MCI
    # inefficiency integral; the panel MSLE likelihood has no such integral, and at fixed
    # parameters base and `:expo_rule` give the identical NLL. The value would be stored on
    # the method spec and never read, so it is refused, not warned about.
    if method == :MSLE && !isnothing(transformation)
        error("`transformation` is only effective with `method=:MCI`: it reparameterizes the MCI " *
              "inefficiency integral, and the panel `:MSLE` likelihood has no such integral. " *
              "Remove `transformation`, or use `method=:MCI`.")
    end

    # Supplying `draws` means the Halton generator is not used, so there is no generation mode left
    # to choose: `multiRand` is forced false and the draws' own row count decides whether they are
    # shared by every firm (1 row) or per firm (N rows).
    _multiRand = isnothing(draws) ? multiRand : false

    _rec, _legacy = sf_recycling_request(recycling, distinct_Halton_length, n_draws, isnothing(draws) && multiRand)
    return PanelMethodSpec(method, transformation, draws, n_draws, _multiRand, GPU, chunks, _rec, _legacy)
end


# ============================================================================
# Section 17: Assembly (_assemble_panel_spec)
# ============================================================================

"""
    _assemble_panel_spec(spec::PanelModelSpec, method::PanelMethodSpec; warn_repeated = true)
        -> _PanelInternalSpec

Assemble internal spec from model specification and method specification.
Performs within-transformation (demeaning), GPU conversion, and Halton draw generation. With
`warn_repeated = false`, generated draws bring no warning when two firms receive identical draws
(the integration check builds its draws this way).
"""
function _assemble_panel_spec(spec::PanelModelSpec{T}, method::PanelMethodSpec;
                              warn_repeated::Bool = true) where {T}

    # Validate: Gamma is MCI-only
    if spec.ineff == :Gamma && method.method == :MSLE
        error("`ineff=:Gamma` is not supported with `method=:MSLE` (no closed-form inverse CDF). " *
              "Use `method=:MCI` instead.")
    end

    # 1. Demean y and X (within-transformation)
    y_tilde = sf_panel_demean(spec.depvar, spec.offsets)
    x_tilde = sf_panel_demean(spec.frontier, spec.offsets)
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
        _validate_panel_user_draws(method.draws)
        # `multiRand` is not consulted: supplying `draws` forces it false, and the row count decides
        # shared (1 row) versus per-firm (N_firms rows)
        shaped = _shape_user_panel_draws(method.draws, spec.N)
        if eltype(shaped) === T && _same_array_kind(shaped, y_tilde)
            halton_raw = shaped                  # the user's array: already the data's element type and device
        else
            halton_raw = ndims(shaped) == 1 ? vec(T.(shaped)) : Matrix{T}(shaped)
            _validate_panel_user_draws(halton_raw)   # again after narrowing to T (see the note there)
        end
    elseif method.multiRand
        period = sf_halton_period(spec.N, method.n_draws; recycling = method.recycling,
                                  legacy = method.halton_legacy)
        warn_repeated && sf_warn_repeated_rows(spec.N, method.n_draws, period)
        halton_raw = sf_halton_draws(spec.N, method.n_draws; T = T, period = period)
    else
        halton_raw = vec(sf_halton_draws(1, method.n_draws; T = T))   # D-vector, as before
    end
    draws = _to_device_panel(halton_raw, y_tilde)

    # 4. Constants
    constants = make_panel_constants(spec.model, T, spec.T_max)

    # 5. Compute Tm1 (T_i - 1): N-vector
    Tm1_cpu = T.(spec.T_periods .- 1)
    Tm1 = method.GPU ? Main.CUDA.CuArray(Tm1_cpu) : Tm1_cpu

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

# Every starting-value block has a width fixed by the specification: K for the frontier,
# L for the scaling function, one for each distribution parameter. A block of the wrong
# width is refused by name -- matching `sfmodel_MCI_init` / `sfmodel_MSLE_init` -- instead
# of being broadcast over the block (a short one) or truncated to its first element (a
# long one), either of which silently starts the optimizer somewhere the user did not ask
# for. A bare scalar is a one-element block, as the manual documents.
function _panel_init_block(name::AbstractString, x, n::Int)
    v = x isa Real ? [Float64(x)] : Float64.(vec(x))
    length(v) == n || error("Length mismatch in sfmodel_panel_init(): " *
                            "$name: expected $n, got $(length(v)).")
    return v
end

"""
    sfmodel_panel_init(; spec::PanelModelSpec, init=nothing, frontier=nothing, scaling=nothing,
                         ln_sigma_v_sq=nothing, mu=nothing, ln_lambda=nothing, ln_k=nothing,
                         ln_sigma_u_sq=nothing, ln_alpha=nothing, ln_theta=nothing)

Create initial parameter vector for panel estimation.

# Arguments
- `spec::PanelModelSpec`: Model specification from `sfmodel_panel_spec()`
- `init=nothing`: Complete initial-value vector (overrides all component args)
- `frontier=nothing`: Initial values for frontier coefficients (default: OLS on demeaned data)
- `scaling=nothing`: Initial values for h(z) coefficients (default: 0.1)
- Distribution-specific keyword arguments (used when matching the distribution):
  - `ln_sigma_u_sq`: HalfNormal, TruncatedNormal, Lognormal, Rayleigh, Exponential
  - `mu`: TruncatedNormal, Lognormal
  - `ln_lambda`: Weibull, Lomax
  - `ln_k`: Weibull, Gamma
  - `ln_alpha`: Lomax
  - `ln_theta`: Gamma, Erlang2/3/4
  - `ln_sigma_v_sq`: all distributions

# Returns
`Vector{Float64}` of initial parameter values.
"""
function sfmodel_panel_init(; spec::PanelModelSpec,
                              init=nothing,
                              frontier=nothing,
                              scaling=nothing,
                              ln_sigma_v_sq=nothing,
                              mu=nothing,
                              ln_lambda=nothing,
                              ln_k=nothing,
                              ln_sigma_u_sq=nothing,
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

    # Component mode: reject kwargs that do not apply to the current distribution,
    # matching the strict "error on extras" behavior of sfmodel_MCI_init /
    # sfmodel_MSLE_init. Without this check, e.g. `mu=0.5` passed to a HalfNormal
    # spec (no μ parameter) would be silently dropped.
    _unused = String[]
    mu        !== nothing && !haskey(idx, :mu)        && push!(_unused, "mu")
    ln_lambda !== nothing && !haskey(idx, :ln_lambda) && push!(_unused, "ln_lambda")
    ln_k      !== nothing && !haskey(idx, :ln_k)      && push!(_unused, "ln_k")
    # `ln_sigma_u_sq` sets the inefficiency-distribution squared-scale parameter, which
    # HalfNormal, TruncatedNormal, Lognormal, Rayleigh and Exponential all own; reject it
    # only for the distributions that have no such parameter (Weibull/Lomax/Gamma/Erlang),
    # where it would otherwise be dropped silently.
    ln_sigma_u_sq !== nothing && !haskey(idx, :ln_sigma_u_sq) &&
        push!(_unused, "ln_sigma_u_sq")
    ln_alpha  !== nothing && !haskey(idx, :ln_alpha)  && push!(_unused, "ln_alpha")
    ln_theta  !== nothing && !haskey(idx, :ln_theta)  && push!(_unused, "ln_theta")
    isempty(_unused) ||
        error("Unused init kwarg(s) in sfmodel_panel_init(): " *
              join(_unused, ", ") *
              ". Not part of the current distribution (:$(spec.ineff)).")

    # Component mode: start with OLS + defaults. The within-transform is done in
    # Float64 whatever the specification's element type, matching
    # getvar(::Type{PFEWHH}) in SFgetvars.jl, which converts to Float64 and then
    # demeans -- so that this default is the number the analytic fit computes for itself.
    y_t = sf_panel_demean(Float64.(spec.depvar), spec.offsets)
    x_t = sf_panel_demean(Float64.(spec.frontier), spec.offsets)
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
        init_vec[idx.beta] .= _panel_init_block("frontier", frontier, length(idx.beta))
    end
    if scaling !== nothing
        init_vec[idx.delta] .= _panel_init_block("scaling", scaling, length(idx.delta))
    end
    if ln_sigma_v_sq !== nothing
        init_vec[idx.ln_sigma_v_sq] = _panel_init_block("ln_sigma_v_sq", ln_sigma_v_sq, 1)[1]
    end
    if mu !== nothing && haskey(idx, :mu)
        init_vec[idx.mu] = _panel_init_block("mu", mu, 1)[1]
    end
    if ln_lambda !== nothing && haskey(idx, :ln_lambda)
        init_vec[idx.ln_lambda] = _panel_init_block("ln_lambda", ln_lambda, 1)[1]
    end
    if ln_k !== nothing && haskey(idx, :ln_k)
        init_vec[idx.ln_k] = _panel_init_block("ln_k", ln_k, 1)[1]
    end
    if ln_sigma_u_sq !== nothing && haskey(idx, :ln_sigma_u_sq)
        init_vec[idx.ln_sigma_u_sq] = _panel_init_block("ln_sigma_u_sq", ln_sigma_u_sq, 1)[1]
    end
    if ln_alpha !== nothing && haskey(idx, :ln_alpha)
        init_vec[idx.ln_alpha] = _panel_init_block("ln_alpha", ln_alpha, 1)[1]
    end
    if ln_theta !== nothing && haskey(idx, :ln_theta)
        init_vec[idx.ln_theta] = _panel_init_block("ln_theta", ln_theta, 1)[1]
    end

    return init_vec
end


# ============================================================================
# Section 19: sfmodel_panel_opt()
# ============================================================================

"""
    sfmodel_panel_opt(; warmstart_solver=nothing, warmstart_opt=nothing,
                        main_solver, main_opt, store_trace=false)

Specify optimization options for panel estimation.

# Arguments
- `warmstart_solver=nothing`: Warmstart optimizer (e.g., `NelderMead()`)
- `warmstart_opt=nothing`: Warmstart options as NamedTuple (e.g., `(iterations=200, g_abstol=1e-3)`).
  Requires `warmstart_solver`; an orphan `warmstart_opt` is refused, not discarded.
- `main_solver`: Main optimizer (e.g., `Newton()`). Required.
- `main_opt`: Main options as NamedTuple. Required.
- `store_trace=false`: When `true`, or when `main_opt` itself sets `store_trace = true`, the main optimizer's
  iteration trace (iterations, objective values and gradient norms) is stored in the result as `trace`;
  otherwise `trace` is `nothing`. `plot(result, :convergence)` draws it from the result of `sfmodel_fit` or
  of the exported `sfmodel_panel_fit`.
"""
function sfmodel_panel_opt(; warmstart_solver=nothing,
                             warmstart_opt=nothing,
                             main_solver,
                             main_opt,
                             store_trace::Bool=false)

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

    # Convert NamedTuple to Optim.Options for main_opt. When store_trace is requested, Optim's
    # `store_trace=true` is forced so the main-solver trace is kept (a later `store_trace` key wins over
    # one in main_opt); the fit reads this flag off `main_opt.store_trace` and copies the trace onto the
    # result.
    m_opt = store_trace ? Optim.Options(; main_opt..., store_trace = true) :
                          Optim.Options(; main_opt...)

    # Convert NamedTuple to Optim.Options for warmstart_opt if provided.
    # Twin of the MCI builder's guard: an orphan `warmstart_opt` is refused, since it would
    # otherwise be replaced by `nothing` and the fit would run with no warmstart stage at all.
    if warmstart_solver !== nothing
        if warmstart_opt === nothing
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

    return PanelOptSpec(warmstart_solver, ws_opt, main_solver, m_opt)
end

# Capture the main optimizer's iteration trace as a NamedTuple for
# plot(:convergence). Returns `nothing` unless the fit's Optim.Options carried
# store_trace=true (set by `store_trace=true` in `sfmodel_panel_opt` or in the unified `sfmodel_opt`).
function _sf_optim_trace(optres, want::Bool)
    want || return nothing
    fv = Optim.f_trace(optres)
    return (iteration = collect(0:(length(fv) - 1)),
            value     = fv,
            g_norm    = Optim.g_norm_trace(optres))
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
    d[:ln_sigma_u_sq] = coef[idx.ln_sigma_u_sq]
end
function _mypan_extract_result_coefs!(::Weibull_Panel, d, coef, idx)
    d[:ln_lambda] = coef[idx.ln_lambda]
    d[:ln_k] = coef[idx.ln_k]
end
function _mypan_extract_result_coefs!(::Lognormal_Panel, d, coef, idx)
    d[:mu] = coef[idx.mu]
    d[:ln_sigma_u_sq] = coef[idx.ln_sigma_u_sq]
end
function _mypan_extract_result_coefs!(::Lomax_Panel, d, coef, idx)
    d[:ln_lambda] = coef[idx.ln_lambda]
    d[:ln_alpha] = coef[idx.ln_alpha]
end
function _mypan_extract_result_coefs!(::Rayleigh_Panel, d, coef, idx)
    d[:ln_sigma_u_sq] = coef[idx.ln_sigma_u_sq]
end
function _mypan_extract_result_coefs!(::Gamma_Panel, d, coef, idx)
    d[:ln_k] = coef[idx.ln_k]
    d[:ln_theta] = coef[idx.ln_theta]
end
function _mypan_extract_result_coefs!(::Erlang_Panel, d, coef, idx)
    d[:ln_theta] = coef[idx.ln_theta]
end

"""
    sfmodel_panel_fit(; spec, method, init=nothing, optim_options=nothing,
                       jlms_bc_index=true, integration_check=false, show_table=true,
                       shape_warning=true, verbose=true)

Estimate the Wang and Ho (2010) panel stochastic frontier model.

# Arguments
- `spec::PanelModelSpec`: Model specification from `sfmodel_panel_spec()`
- `method::PanelMethodSpec`: Method specification from `sfmodel_panel_method()`
- `init=nothing`: Initial parameter vector from `sfmodel_panel_init()`. Starting values at or
  beyond a parameter's numerical bounds are refused (an error) for a homoscedastic parameter and
  give a warning for a heteroscedastic one; a fit held at its starting values (no warmstart stage
  and `iterations = 0` in the main stage) is not checked.
- `optim_options=nothing`: Optimization options from `sfmodel_panel_opt()`
- `jlms_bc_index::Bool=true`: Compute JLMS and BC efficiency indices
- `integration_check=false`: Run the integration check after the fit: at the estimate, the simulated
  log-likelihood is evaluated again with four times the draws, and the one-step shift −V·(g_4S − g_S)
  is reported in standard errors; each part reads OK, CAUTION or FAIL and the result is stored in
  `result.integration_check`, separately from `converged` and `redflag`. `true` uses the default
  thresholds (0.1 and 1 for the change in the log-likelihood, 0.1 and 0.5 standard errors for the
  shift); a NamedTuple with any of `ll_caution`, `ll_fail`, `shift_caution`, `shift_fail` changes
  them. The thresholds are heuristics.
- `show_table::Bool=true`: Print formatted estimation table
- `shape_warning::Bool=true`: Print the warning of the shape-range check and the warning that a Lomax
  fit's marginal effects are undefined where α ≤ 1; `false` prints neither (`shape_check` is stored
  either way)
- `verbose::Bool=true`: Print progress information

# Returns
NamedTuple with fields including `converged`, `loglikelihood`, `coeff` (a homoscedastic coefficient that
ended beyond its numerical bound is reported at the bound; see `bound_check`), `std_err`,
`jlms` (observation-level, length Σ T_i), `bc` (observation-level, length Σ T_i),
`redflag` (0, 1 or 2; 1 also for a boundary solution), `hessian_flag` (the Hessian part of
`redflag`), `bound_check` (each parameter at or beyond its numerical bound; see Parameter
Bounds in the manual), and more. All results also accessible via `result.list`.

`integration_check` is the result of the integration check (the option `integration_check`), or
`nothing` when the check is off: a NamedTuple with `status` (`:OK`, `:CAUTION`, `:FAIL`,
`:not_available` or `:error`), `reason`, `n_draws` and `n_draws_check` (the fit's draws and
four times as many), `loglik`, `loglik_check`, `delta_loglik`, `delta_loglik_per_obs`,
`ll_verdict`, `shift_available`, `shift_note`, `shift`, `shift_se`, `max_shift_se`,
`max_shift_param`, `max_shift_estimate`, `max_shift_onestep` (the largest shift in standard
errors, the parameter it belongs to, its estimate and its value after the shift),
`shift_verdict`, `thresholds` and `seconds`. It is separate from `converged` and `redflag`.

`shape_check` is the verdict of the shape-range check, or `nothing` for the models that it does not
cover: a NamedTuple with `method`, `family`, `parameter`, `rule`, `threshold`, `in_range`, `n_below`,
`n` (1: one shape per model), `share` and `min_value`. MCI Gamma, Weibull and Lomax fits under the
default transformation rule, and MSLE Lomax fits, print a warning when the estimated shape lies below
`threshold` (`in_range` is then `true`), unless the fit passes `shape_warning = false`. It is separate
from `converged`, `redflag` and `integration_check`.
"""
function sfmodel_panel_fit(;
    spec::PanelModelSpec,
    method::PanelMethodSpec = sfmodel_panel_method(),
    init=nothing,
    optim_options=nothing,
    jlms_bc_index::Bool=true,
    marginal::Bool=true,
    integration_check=false,
    show_table::Bool=true,
    shape_warning::Bool=true,
    verbose::Bool=true)

    # The thresholds of the integration check after the fit, or `nothing` when it is off; an invalid
    # value is an error before anything is computed.
    _ic_settings = sf_integration_settings(integration_check)

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
        T_min_d, T_max_d = extrema(spec.T_periods)
        T_info = T_min_d == T_max_d ? "T=$(T_min_d)" :
                 "T=$(T_min_d)-$(T_max_d) (unbalanced)"
        _ndraws = ndims(ispec.draws) == 1 ? length(ispec.draws) : size(ispec.draws, 2)
        printstyled("  N_firms=$(spec.N), $(T_info), K=$(spec.K), L=$(spec.L), n_draws=$(_ndraws)\n"; color=:yellow)
        printstyled("  Type: $(spec.sign == 1 ? "production" : "cost")\n"; color=:yellow)
        println()
    end

    # 3. Prepare initial values
    K = ispec.K
    N = spec.N
    NT = sum(spec.T_periods)

    # OLS on demeaned data
    y_t_cpu = sf_panel_demean(spec.depvar, spec.offsets)
    x_t_cpu = sf_panel_demean(spec.frontier, spec.offsets)
    beta_ols = x_t_cpu \ y_t_cpu

    if init === nothing
        # Starting value only: within-transformed in Float64, matching
        # getvar(::Type{PFEWHH}) in SFgetvars.jl, which converts to Float64 and then
        # demeans. The OLS residual statistics below keep `beta_ols`, which stays
        # element-type-native.
        beta_ols_init = sf_panel_demean(Float64.(spec.frontier), spec.offsets) \
                        sf_panel_demean(Float64.(spec.depvar), spec.offsets)
        n_total = panel_plen(spec.model.ineff, spec.K, spec.L)
        sf_init = vcat(beta_ols_init, fill(0.1, n_total - K))
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
        expected_len = panel_plen(spec.model.ineff, spec.K, spec.L)
        length(sf_init) == expected_len ||
            error("Length mismatch in sfmodel_panel_fit(): init has $(length(sf_init)) elements, expected $expected_len.")
    end

    # 4. OLS statistics on demeaned data
    resid = y_t_cpu - x_t_cpu * beta_ols
    sse = sum(resid .^ 2)
    ssd = sqrt(sse / NT)
    sk_ols = sum(resid .^ 3) / ((ssd^3) * NT)

    # 5. Prepare optimization options (the default is the one sfmodel_fit uses for every
    #    method)
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

    # Starting values at or beyond a bound (`sf_start_check`, src/sf_redflag.jl): a warning for each heteroscedastic
    # parameter there, then an error when a homoscedastic coefficient is there. A fit held at its starting values (no
    # warmstart stage, no main iteration) is not checked.
    held = !do_warmstart && myopt.main_opt.iterations == 0
    if !held
        start_error, start_warnings = sf_start_check(_panel_bound_list(ispec),
                                                     _panel_bound_verdict(ispec, sf_init, NT), sf_init)
        foreach(w -> sf_print_warning("Warning: " * w), start_warnings)
        isempty(start_error) || throw(ArgumentError(start_error))
    end

    # GPU fits: warn before the optimizer when one Hessian evaluation is estimated to need
    # more device memory than is free.
    if ispec.GPU
        sf_check_gpu_memory(ispec.method === :MCI ? SF_WORK_ARRAYS.panel_mci : SF_WORK_ARRAYS.panel_msle,
                            ispec.N, ndims(ispec.draws) == 1 ? length(ispec.draws) : size(ispec.draws, 2),
                            ispec.chunks, length(sf_init); unit = "firms")
    end

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

    # 9. Post-estimation
    _coevec = Optim.minimizer(_optres)

    # Hessian, computed once: the matrix classified below is the one stored on the result.
    # Variance-covariance matrix and the Hessian part of redflag, by the rule shared with
    # the other backends. A parameter at or beyond its numerical bound (a variance, scale or
    # shape parameter) makes the fit a boundary solution: redflag 1, whatever the Hessian says,
    # and the message names each such parameter. The verdict is taken at the optimizer's point
    # and stored as `bound_check`.
    # The Hessian is computed and classified by sf_classify_hessian(f, x) (src/sf_redflag.jl),
    # so an error while computing it does not end the fit and discard the estimate: the fit
    # reads as one whose Hessian cannot be inverted (redflag 1, all-NaN Hessian and var-cov
    # matrix), and a message, printed whatever `verbose` is, gives the error. Ctrl-C, out of
    # memory, a stack overflow and CUDA's out-of-GPU-memory error still end the fit.
    bc = _panel_bound_verdict(ispec, _coevec, NT)
    # A coefficient beyond its bound is moved to the bound (`sf_bound_move`, src/sf_redflag.jl), where the likelihood
    # and its derivatives equal those at the optimizer's point; the moved vector is reported from JLMS/BC on, and
    # the entry in `bound_check` keeps the optimizer's value.
    bc, coef_moved = sf_bound_move(_panel_bound_list(ispec), bc, _coevec)
    vcov_result = sf_classify_hessian(_lik, _coevec; at_clamp = !isempty(bc), message = verbose,
                                      bound_text = isempty(bc) ? "" : sf_bound_message(bc))
    numerical_hessian = vcov_result.hessian
    var_cov_matrix = vcov_result.var_cov_matrix
    redflag = max(redflag, vcov_result.redflag)

    stddev = _stderr_from_vcov(var_cov_matrix)  # non-PD vcov ⇒ NaN SEs (no DomainError); keeps stderror(r) NaN-consistent with sqrt.(diag(vcov(r))).

    # 9a. Shape-range check: under the default MCI transformation rule (an `ispec.transformation` of
    #     `nothing` stands for it), and for MSLE Lomax fits, a warning when the estimated shape lies where
    #     the simulated likelihood is unreliable (SF_SHAPE_RANGES, src/sf_accuracy.jl); the verdict is
    #     stored and no number changes.
    _sc_rule = ispec.method === :MCI ?
        something(ispec.transformation, default_panel_transformation(ispec.model.ineff)) : nothing
    _sc_entry = sf_shape_entry(ispec.method, ispec.ineff, _sc_rule)
    _sc = _sc_entry === nothing ? nothing :
        sf_shape_check(_sc_entry, getproperty(get_panel_ineff_vals(ispec.model.ineff, _coevec, ispec.idx,
                                                                   ispec.constants), _sc_entry.parameter);
                       shape_warning = shape_warning)

    # 9b. Integration check (optional): at the estimate, the simulated log-likelihood again with four times
    #     the draws, and the one-step shift that they imply, in standard errors (src/sf_accuracy.jl). The
    #     4S draws are those of a fit with `n_draws = 4S` and the same settings, in min(firms, 4 × `chunks`)
    #     groups of firms. The 4S draws are built without the warning about identical draws, which would
    #     name the check's number of draws rather than the fit's.
    _ic = _ic_settings === nothing ? nothing :
        sf_run_integration_check(_ic_settings;
            S = ndims(ispec.draws) == 1 ? length(ispec.draws) : size(ispec.draws, 2),
            n_obs = sum(ispec.T_periods), ll = -_fmin, theta = _coevec, nll_S = _lik,
            build_nll_4S = () -> begin
                isp4 = _assemble_panel_spec(spec, _with_draws(method, 4 * method.n_draws, min(ispec.N, 4 * method.chunks));
                                            warn_repeated = false)
                p -> panel_nll(isp4, p; chunks = isp4.chunks)
            end,
            V = var_cov_matrix, se = stddev, hessian_flag = vcov_result.hessian_flag,
            names = ispec.varnames, unavailable = sf_integration_unavailable(method, ispec.N))

    # The reported coefficients from here on: a coefficient beyond its bound at the bound. The Hessian, the
    # standard errors and the checks above are those at the optimizer's point.
    _coevec = coef_moved

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
    # A Lomax fit's marginal effects are NaN where α ≤ 1, since E(u_it) = h(z)·λ/(α − 1) does not exist
    # there (every row with a NaN counts): one warning per fit, whatever `show_table` and `verbose` are,
    # unless `shape_warning = false`.
    if margeff !== nothing && ispec.model.ineff isa Lomax_Panel
        M = Matrix(margeff)
        n_nan = count(i -> any(isnan, view(M, i, :)), axes(M, 1))
        sf_warn_lomax_mean(n_nan, size(M, 1); shape_warning = shape_warning, after_fit = true)
    end

    # 11. Print results table
    table_result = nothing
    if show_table
        table_result = panel_print_table(ispec, _coevec, var_cov_matrix;
                                          optim_result=_optres, loglik=-_fmin,
                                          integration=_ic, table_format=:text)

        printstyled("***** Additional Information *********\n"; color=:cyan)

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
            println("  level are saved in the return. See below.\n")
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
    _dicRES[:loglikelihood] = -_fmin          # at the reported coefficients
    _dicRES[:table] = table_result !== nothing ? table_result.table : nothing
    _dicRES[:coeff] = _coevec
    _dicRES[:std_err] = stddev
    _dicRES[:var_cov_mat] = var_cov_matrix
    _dicRES[:jlms] = _jlms
    _dicRES[:bc] = _bc
    _dicRES[:OLS_resid_skew] = sk_ols
    _dicRES[:marginal] = margeff
    _dicRES[:marginal_mean] = margMinfo
    _dicRES[:_____________] = "___________________"
    _dicRES[:model] = ispec
    _dicRES[:________________] = "___________________"
    _dicRES[:Hessian] = numerical_hessian
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
    _dicRES[:trace] = _sf_optim_trace(_optres, myopt.main_opt.store_trace)   # main-solver trace for plot(:convergence)
    _dicRES[:redflag] = redflag
    _dicRES[:hessian_flag] = vcov_result.hessian_flag   # the Hessian part of redflag
    _dicRES[:bound_check] = bc                # each parameter at or beyond its numerical bound
    _dicRES[:integration_check] = _ic         # nothing when the option is off
    _dicRES[:shape_check] = _sc               # nothing for the families and rules SF_SHAPE_RANGES does not cover
    _dicRES[:______________________] = "___________________"
    _dicRES[:GPU] = method.GPU
    # the draws actually used (ispec.draws): a 1-D `draws` with multiRand=true is replaced by
    # the wrapped Halton matrix at assembly, so method.draws is not what the fit consumed
    _dicRES[:n_draws] = ndims(ispec.draws) == 1 ? length(ispec.draws) : size(ispec.draws, 2)
    _dicRES[:multiRand] = method.multiRand
    _dicRES[:chunks] = method.chunks
    _scheme, _hlen = sf_draw_scheme(spec.N, _dicRES[:n_draws]; user = method.draws !== nothing,
                                    multiRand = method.multiRand, recycling = method.recycling,
                                    legacy = method.halton_legacy)
    _dicRES[:draw_scheme] = _scheme                # :halton, :halton_recycled, :halton_shared or :user
    _dicRES[:distinct_Halton_length] = _hlen       # length of the Halton sequence the draws came from
    _dicRES[:estimation_method] = method.method
    _dicRES[:user_draws_supplied] = method.draws !== nothing
    _dicRES[:firm_ids] = spec.firm_ids
    _dicRES[:transformation] = method.transformation

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
        trace = _dicRES[:trace],
        redflag = _dicRES[:redflag],
        hessian_flag = _dicRES[:hessian_flag],
        bound_check = _dicRES[:bound_check],
        integration_check = _dicRES[:integration_check],
        shape_check = _dicRES[:shape_check],
        GPU = _dicRES[:GPU],
        n_draws = _dicRES[:n_draws],
        multiRand = _dicRES[:multiRand],
        chunks = _dicRES[:chunks],
        draw_scheme = _dicRES[:draw_scheme],
        distinct_Halton_length = _dicRES[:distinct_Halton_length],
        estimation_method = _dicRES[:estimation_method],
        user_draws_supplied = _dicRES[:user_draws_supplied],
        firm_ids = _dicRES[:firm_ids],
        transformation = _dicRES[:transformation],
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
    id       = repeat(1:N, inner=T),
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
result.jlms               # NT-vector (obs-level): jlms_it = h_it * E[u_i* | data_i]
result.bc                 # NT-vector (obs-level): E[exp(-h_it * u_i*) | data_i]
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
    spec = sfmodel_panel_spec(depvar=y, frontier=X, zvar=Z, id=firm_ids)
)


## Example 5: Using sfmodel_panel_init for custom starting values

init = sfmodel_panel_init(
    spec          = spec,
    frontier      = [0.8, -0.3],   # override OLS-based β
    scaling       = [0.2],         # override default δ
    ln_sigma_u_sq   = 0.5,           # override default ln(σ²)
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


## Example 7: DSL — Balanced panel (with @id)

spec_bal = sfmodel_panel_spec(
    @useData(df),
    @depvar(yit),
    @frontier(xit),
    @zvar(zit),
    @id(id);
    type = :prod
)


## Example 8: DSL — Without @zvar → REJECTED at spec time
##
## A panel spec without @zvar implies a constant scaling h(z)=exp(δ₀), which is
## unidentified in the Wang-Ho model (within-group demeaning eliminates it). Both
## the DSL and keyword paths raise an explanatory identification error instead of
## manufacturing an all-ones zvar and publishing prior-mean jlms/bc. The call below
## documents that it is rejected — copying it yields an instructive error, not a
## silent (mis)fit. Use Example 6/7 with @zvar.

spec_no_z = sfmodel_panel_spec(   # <-- raises: requires @zvar with within-firm variation
    @useData(df),
    @depvar(yit),
    @frontier(xit),
    @id(id);
    type = :prod
)
=#
