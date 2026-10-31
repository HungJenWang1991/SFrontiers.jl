# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

# sf_draws.jl
# The draw blocks of the simulation backends: the base-2 Halton generator that writes a fit's block, and the
# row source through which predict reads a stored block one group of rows at a time. Loaded before the
# backends.

module SFDraws

export sf_halton_draws, sf_halton_point, sf_halton_period, sf_draw_scheme
export SF_RECYCLING_CAP, sf_recycling_request, sf_warn_repeated_rows
export SF_WORK_ARRAYS, sf_hessian_pass_bytes, sf_gpu_free_bytes, sf_warn_gpu_memory, sf_check_gpu_memory

# Rows `rows` of a fit's stored draw block when it is not a host Float64 matrix (a Float32 block,
# or a block on the GPU). The simulation kernels read it one group of rows at a time, as dense
# host Float64 matrices, through `host_rows`; predict computes in Float64, and widening each draw
# before use gives the values the kernels computed before, without a converted copy of the block.
struct DrawRows{B<:AbstractMatrix} <: AbstractMatrix{Float64}
    block::B
    rows::Vector{Int}
end
Base.size(d::DrawRows) = (length(d.rows), size(d.block, 2))
# Element access for host blocks; the kernels read groups through `host_rows`.
Base.getindex(d::DrawRows, i::Int, j::Int) = Float64(d.block[d.rows[i], j])

"Rows a:b of `d` as a dense host Float64 matrix; consecutive rows are read as one range."
function host_rows(d::DrawRows, a::Int, b::Int)
    a > b && return Matrix{Float64}(undef, 0, size(d.block, 2))
    r = d.rows[a:b]
    idx = all(==(1), diff(r)) ? (r[1]:r[end]) : r
    return Float64.(d.block isa Array ? view(d.block, idx, :) : Array(d.block[idx, :]))
end

# ---- The base-2 Halton generator of the simulation backends ----

# Base-2 radical inverse (van der Corput) of k >= 1; exact for k < 2^53.
_radinv2(k::UInt64) = Float64(bitreverse(k)) * 0x1p-64

# φ(p) computed in Float64, converted to T and kept below 1.
"The base-2 Halton point at position `p` of the sequence, in element type `T` and below 1."
sf_halton_point(::Type{T}, p::Integer) where {T<:AbstractFloat} =
    min(T(_radinv2(UInt64(p))), prevfloat(one(T)))

"N × S base-2 Halton draws; row i holds points (i − 1)·S + 1, …, i·S (restarting after `period` if given)."
function sf_halton_draws(N::Integer, S::Integer; T::Type{<:AbstractFloat} = Float64,
                         period::Union{Nothing,Integer} = nothing)
    N >= 1 || throw(ArgumentError("the number of rows must be positive, got $N"))
    S >= 1 || throw(ArgumentError("the number of draws must be positive, got $S"))
    (period === nothing || period >= 1) ||
        throw(ArgumentError("the sequence length must be positive, got $period"))
    M = Matrix{T}(undef, N, S)
    _fill_halton!(M, Int64(S), period === nothing ? nothing : Int64(period))
    _check_open_unit(M)
    return M
end

# Function barrier: `period` is Nothing or Int64, so the loop is type-stable.
function _fill_halton!(M::AbstractMatrix{T}, S::Int64, period) where {T}
    N = size(M, 1)
    @inbounds for j in 1:S, i in 1:N          # inner index i: column-major writes
        p = Int64(i - 1) * S + j              # position in the sequence, Int64 on every platform
        period === nothing || (p = mod1(p, period))
        M[i, j] = sf_halton_point(T, p)
    end
    return M
end

function _check_open_unit(M)
    lo, hi = extrema(M)
    (lo > 0 && hi < 1) || error("internal error: a generated Halton draw lies outside (0, 1) " *
                                "(min = $lo, max = $hi); please report it")
    return nothing
end

"Restart length of the Halton sequence (`nothing`: none); `legacy` uses the rule of versions before 2.0.0."
function sf_halton_period(N::Integer, S::Integer; recycling::Integer = 0, legacy::Bool = false)
    recycling == 0 && return nothing                    # no recycling
    recycling > 0 || throw(ArgumentError("the recycling length must be positive, got $recycling"))
    NS = Int64(N) * Int64(S)
    if legacy   # the rule of versions before 2.0.0, written exactly as their generator wrote it
        n = min(floor(Int, log2(NS + 1)), floor(Int, log2(recycling + 1)))
        L = 2^n - 1
    else        # the `recycling` option: the first min(N·S, cap) points
        L = Int(min(NS, recycling))
    end
    S <= L || throw(ArgumentError(legacy ?
        "n_draws = $S exceeds the $L points of the recycled Halton sequence (the rule of versions before " *
        "2.0.0 shortens it to the largest length of the form 2^m - 1 not above N × n_draws = $NS or " *
        "distinct_Halton_length = $recycling), so each observation's draws would repeat within the " *
        "observation. Raise distinct_Halton_length, lower n_draws, or drop distinct_Halton_length." :
        "n_draws = $S exceeds the recycled Halton sequence of $L points, so each observation's draws " *
        "would repeat within the observation."))
    return L
end

"The draw scheme of a fit and the length of the Halton sequence its draws came from."
function sf_draw_scheme(N::Integer, S::Integer; user::Bool, multiRand::Bool,
                        recycling::Integer, legacy::Bool)
    user && return (:user, 0)
    multiRand || return (:halton_shared, Int(S))
    recycling == 0 && return (:halton, Int(Int64(N) * Int64(S)))
    return (:halton_recycled, sf_halton_period(N, S; recycling = recycling, legacy = legacy))
end

# ---- Recycling on request, and the guard against identical rows ----

const SF_RECYCLING_CAP = 2^24 - 1     # recycling = true: the pool is exactly {j/2^24}

# (cap, legacy) for the method struct, from the two keywords; `generated` = no draws and multiRand.
"(recycling length, legacy flag) of a method from `recycling` and the deprecated `distinct_Halton_length`."
function sf_recycling_request(recycling, distinct_Halton_length, n_draws::Integer, generated::Bool)
    if distinct_Halton_length !== nothing
        recycling === false || throw(ArgumentError("`recycling` and `distinct_Halton_length` were both " *
            "given. `distinct_Halton_length` is deprecated; give only `recycling`."))
        @warn "`distinct_Halton_length` is deprecated. It still gives the per-observation draws of " *
              "versions before 2.0.0: one Halton sequence shortened to 2^m - 1 points (the largest such " *
              "length not above N × n_draws or the value given), recycled across observations. To reuse " *
              "points on purpose, use `recycling`; without either keyword each observation gets its own " *
              "n_draws points of one Halton sequence of N × n_draws points." _id=:sf_deprecated_distinct_Halton_length maxlog=1
        generated || return (0, false)                 # not read, as before
        n_draws <= distinct_Halton_length || throw(ArgumentError("`n_draws` = $n_draws exceeds " *
            "`distinct_Halton_length` = $distinct_Halton_length."))
        return (Int(distinct_Halton_length), true)
    end
    recycling === false && return (0, false)
    generated || throw(ArgumentError("`recycling` applies only to the per-observation Halton draws the " *
        "package generates (`multiRand = true` and no `draws`); here it would not be read."))
    cap = recycling === true ? SF_RECYCLING_CAP : Int(recycling)    # Bool <: Integer: test Bool first
    1 <= cap <= SF_RECYCLING_CAP || throw(ArgumentError("`recycling` must be `true`, `false` or an integer " *
        "from n_draws to 2^24 - 1 = $(SF_RECYCLING_CAP) (the length of the recycled sequence); got $recycling."))
    n_draws <= cap || throw(ArgumentError("`recycling = $recycling` recycles a sequence of $cap " *
        "point(s), fewer than `n_draws = $n_draws`, so each observation's draws would repeat within the " *
        "observation. Use `recycling = true` (a sequence of $(SF_RECYCLING_CAP) points), a larger " *
        "integer, or no recycling."))
    return (cap, false)
end

# Rows i and i' receive identical draws exactly when (i' - i)·S ≡ 0 (mod period).
"Warn when recycling gives two observations identical draws; return their distance, or `nothing`."
function sf_warn_repeated_rows(N::Integer, S::Integer, period)
    period === nothing && return nothing
    Int64(N) * Int64(S) > period || return nothing
    g = gcd(Int64(S), Int64(period)); P = Int64(period) ÷ g
    N > P || return nothing
    remedies = ["no recycling"]
    period < SF_RECYCLING_CAP && push!(remedies, "a longer sequence")
    N <= period && push!(remedies, "an n_draws that shares no factor with $period" *
                                   (isodd(period) ? " (any power of two)" : ""))
    @warn "With n_draws = $S and a recycled Halton sequence of $period points, observations (firms in panel models) i and " *
          "i + $P receive identical draws. Use $(join(remedies, ", ", " or "))."
    return P
end

# ---- The GPU memory check before a fit ----

"""
Work arrays of one chunk's size that one likelihood evaluation holds. Cross-sectional MCI: u,
log-likelihood and log-sum-exp temporary (+3 copula buffers); MSLE: buffer and temporary
(+2 copula buffers); panel MCI: one array for u, then the log-weights and their exponentials,
and one for log f_u + log|J|; panel MSLE: one array for u, then the log-weights and their
exponentials.
"""
const SF_WORK_ARRAYS = (mci = 3, mci_copula = 6, msle = 2, msle_copula = 4,
                        panel_mci = 2, panel_msle = 1)

"ForwardDiff's default chunk size for n parameters (its threshold is 12)."
_fd_default_chunk(n::Integer) = n <= 12 ? Int(n) : cld(Int(n), cld(Int(n), 12))

"""
Bytes of the work arrays of one likelihood evaluation inside a ForwardDiff Hessian:
narrays arrays of cld(nrows, chunks) × n_draws nested duals of 8(C + 1)^2 bytes each, plus
one array of the last group's size when that group is shorter (the likelihood gives it an
array of its own).
"""
function sf_hessian_pass_bytes(narrays::Integer, nrows::Integer, n_draws::Integer,
                               chunks::Integer, nparams::Integer)
    C = _fd_default_chunk(nparams)
    g = cld(Int(nrows), max(Int(chunks), 1))   # rows of a full group
    r = rem(Int(nrows), max(g, 1))             # rows of a shorter last group (0 when there is none)
    return (Int(narrays) * g + r) * Int(n_draws) * 8 * (C + 1)^2
end

"""
GPU memory this process can still allocate: the device's free memory plus memory that
CUDA.jl's pool holds unused; `nothing` when CUDA.jl is not loaded or cannot report it.
"""
function sf_gpu_free_bytes()
    isdefined(Main, :CUDA) || return nothing
    CU = getfield(Main, :CUDA)
    try
        free = isdefined(CU, :free_memory) ? CU.free_memory() : CU.available_memory()
        pooled = (isdefined(CU, :cached_memory) && isdefined(CU, :used_memory)) ?
                 max(0, Int(coalesce(CU.cached_memory(), 0)) - Int(coalesce(CU.used_memory(), 0))) : 0
        return Int(free) + pooled
    catch err
        err isa InterruptException && rethrow()
        return nothing
    end
end

"Warn when `est` bytes (one Hessian evaluation) exceed `free` bytes; return whether it warned."
function sf_warn_gpu_memory(est::Integer, free::Union{Nothing,Integer}; nrows::Integer,
                            unit::AbstractString, n_draws::Integer, chunks::Integer,
                            nparams::Integer)
    (free === nothing || est <= free) && return false
    gb(x)  = round(x / 1e9; digits = 1)
    @warn "One Hessian evaluation of this GPU fit is estimated to need about $(gb(est)) GB of GPU " *
          "memory, more than the $(gb(free)) GB that are free ($nrows $unit, n_draws = $n_draws, " *
          "chunks = $chunks, $nparams parameters). The fit may stop with an out-of-memory error " *
          "or slow down sharply. Memory falls in proportion to `chunks`. A smaller `n_draws` (a " *
          "smaller power of two) also reduces it; confirm the final estimates at the default " *
          "number of draws. `GC.gc(); CUDA.reclaim()` releases GPU memory that earlier fits in " *
          "this session still hold."
    return true
end

"""
Before the optimizer of a GPU fit: estimate one Hessian evaluation and warn if it exceeds
the free GPU memory.
"""
function sf_check_gpu_memory(narrays, nrows, n_draws, chunks, nparams; unit::AbstractString)
    est  = sf_hessian_pass_bytes(narrays, nrows, n_draws, chunks, nparams)
    free = sf_gpu_free_bytes()
    @debug "GPU memory check" est free
    return sf_warn_gpu_memory(est, free; nrows, unit, n_draws, chunks, nparams)
end

end # module SFDraws
