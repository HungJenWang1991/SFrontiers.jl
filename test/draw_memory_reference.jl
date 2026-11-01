# Copies of panel functions as they were before the per-chunk rewrite; the tests compare the
# package's functions with them bit for bit.
#
# Included once by the test files that use it (guarded by `isdefined(@__MODULE__, :_ref_panel_nll)`)
# and by gpu_draw_memory_runner.jl; not included by runtests.jl itself.

using SFrontiers
using Random

const _PBR = SFrontiers.Panel_Backend

function _ref_panel_nll_msle(spec::_PBR._PanelInternalSpec{T}, p::AbstractVector{P};
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
    h_tilde = _PBR.compute_h_tilde(spec.z_raw, p, idx, spec.offsets)
    A, B, C = _PBR._compute_panel_ABC(epsilon_tilde, h_tilde, spec.offsets, N)
    # Move A, B, C to GPU if data lives there (halton is CuArray when GPU=true)
    A = _PBR._to_device_panel(A, halton)
    B = _PBR._to_device_panel(B, halton)
    C = _PBR._to_device_panel(C, halton)

    # 3. Extract noise/inefficiency parameters
    noise_vals = _PBR.get_panel_noise_vals(model.noise, p, idx, c)
    ineff_vals = _PBR.get_panel_ineff_vals(model.ineff, p, idx, c)

    # 4. Generate u draws via inverse CDF (dispatched by inefficiency distribution)
    halton_P = P.(halton)
    u_draws = _PBR.mypan_gen_udraws_msle(model.ineff, ineff_vals, halton_P, c, P)

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
        log_likes = _PBR.logsumexp_rows_panel(log_fv) .- log(P(D))
        return -_PBR._sum_scalar_panel(log_likes)
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

            log_likes = _PBR.logsumexp_rows_panel(log_fv) .- log(P(D))
            total_nll -= _PBR._sum_scalar_panel(log_likes)
        end
        return total_nll
    end
end

function _ref_panel_nll_mci(spec::_PBR._PanelInternalSpec{T}, p::AbstractVector{P};
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
    h_tilde = _PBR.compute_h_tilde(spec.z_raw, p, idx, spec.offsets)
    A, B, C = _PBR._compute_panel_ABC(epsilon_tilde, h_tilde, spec.offsets, N)
    # Move A, B, C to GPU if data lives there (halton is CuArray when GPU=true)
    A = _PBR._to_device_panel(A, halton)
    B = _PBR._to_device_panel(B, halton)
    C = _PBR._to_device_panel(C, halton)

    # 3. Extract noise/inefficiency parameters
    noise_vals = _PBR.get_panel_noise_vals(model.noise, p, idx, c)
    ineff_vals = _PBR.get_panel_ineff_vals(model.ineff, p, idx, c)

    # 4. MCI transformation: u = g(t, scale), Jacobian |g'(t, scale)|
    scale = _PBR.mypan_get_mci_scale(model.ineff, ineff_vals)
    trans, jacob = _PBR.resolve_panel_transformation(spec.transformation, model)

    halton_P = P.(halton)
    u_draws = trans.(halton_P, scale)
    u_draws = min.(u_draws, inv(P(c.clamp_lo)))                  # clamp upper

    # 5. Log-PDF of inefficiency + log-Jacobian
    log_fu = _PBR.mypan_logpdf_ineff(model.ineff, u_draws, ineff_vals, P(c.clamp_lo))
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
        log_likes = _PBR.logsumexp_rows_panel(log_lik) .- log(P(D))
        return -_PBR._sum_scalar_panel(log_likes)
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

            log_likes = _PBR.logsumexp_rows_panel(log_lik) .- log(P(D))
            total_nll -= _PBR._sum_scalar_panel(log_likes)
        end
        return total_nll
    end
end

function _ref_panel_nll(spec::_PBR._PanelInternalSpec{T}, p::AbstractVector{P};
                        chunks::Int=4) where {T<:AbstractFloat, P<:Real}
    if spec.method == :MSLE
        return _ref_panel_nll_msle(spec, p; chunks=chunks)
    else
        return _ref_panel_nll_mci(spec, p; chunks=chunks)
    end
end

# Bit-for-bit equality of two numbers or arrays (of floats or ForwardDiff duals).
_same_bits(x, y) = size(x) == size(y) && reinterpret(UInt8, vec(collect(x))) == reinterpret(UInt8, vec(collect(y)))

# Bytes allocated by one call of f, after two warm-up calls: the smallest of three readings. @allocated
# reads a process-wide counter, so bytes allocated meanwhile by something else (a finalizer run by a
# collection, another thread) can add to a single reading; what f itself allocates is in every reading.
_alloc(f) = (f(); f(); minimum(@allocated(f()) for _ in 1:3))

# A Wang-Ho panel_TFE data set: firm i has 3 + mod(i, 3) periods, its observations consecutive.
function _ref_panel_data(F, seed)
    Ti = [3 + mod(i, 3) for i in 1:F]
    id = reduce(vcat, [fill(i, Ti[i]) for i in 1:F])
    NT = sum(Ti)
    rng = Xoshiro(seed)
    X = randn(rng, NT, 2)
    z = randn(rng, NT)
    a = randn(rng, F)
    us = abs.(0.4 .* randn(rng, F))
    y = a[id] .+ X * [0.6, -0.4] .+ 0.3 .* randn(rng, NT) .- exp.(0.5 .* z) .* us[id]
    return (y = y, X = X, z = z, id = id)
end

# The user-level specification of the data, with element type T.
_ref_unified_spec(d, ineff; T = Float64) =
    sfmodel_spec(depvar = T.(d.y), frontier = T.(d.X), zvar = T.(d.z), id = d.id, noise = :Normal,
                 ineff = ineff, type = :prod, datatype = :panel_TFE)

# The internal panel specification that the likelihood takes.
_ref_panel_spec(d, ineff; method, D, multiRand = true, chunks = 10, T = Float64, transformation = nothing,
                GPU = false) =
    _PBR._assemble_panel_spec(_ref_unified_spec(d, ineff; T = T).panel_spec,
                              SFrontiers.sfmodel_panel_method(method = method, n_draws = D, multiRand = multiRand,
                                                              chunks = chunks, transformation = transformation,
                                                              GPU = GPU))

# An interior evaluation point: the default start values moved by 0.05.
_ref_theta(d, ineff; T = Float64) = sfmodel_init(spec = _ref_unified_spec(d, ineff; T = T)) .+ 0.05

# Copies of `panel_jlms_bc` and `_compute_bc_obs` as they were before the per-group computation of the
# efficiency indices.
function _ref_compute_bc_obs(log_w::AbstractMatrix{P}, log_denom::AbstractVector{P},
                          h_raw::AbstractVector, u_draws::AbstractVecOrMat,
                          spec::_PBR._PanelInternalSpec) where P
    NT = length(h_raw)
    N  = spec.N
    D  = ndims(u_draws) == 1 ? length(u_draws) : size(u_draws, 2)
    bc = Vector{P}(undef, NT)

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
        log_bc_num = _PBR.logsumexp_rows_panel(log_bc_w_firm)      # Ti
        bc[s:e] .= exp.(log_bc_num .- log_denom[i])
    end
    return bc
end

function _ref_panel_jlms_bc(spec::_PBR._PanelInternalSpec{T}, p::AbstractVector{P};
                        chunks::Int=1) where {T<:AbstractFloat, P<:Real}

    N, K = spec.N, spec.K
    c    = spec.constants
    idx  = spec.idx
    model = spec.model

    # Pull inputs to CPU (no-op when already CPU)
    y_tilde = _PBR._to_cpu(spec.y_tilde)
    x_tilde = _PBR._to_cpu(spec.x_tilde)
    z_raw   = _PBR._to_cpu(spec.z_raw)
    halton  = _PBR._to_cpu(spec.draws)
    D       = ndims(halton) == 1 ? length(halton) : size(halton, 2)

    # 1. Residuals
    epsilon_tilde = y_tilde .- sum(P(p[idx.beta[j]]) .* (@view x_tilde[:, j]) for j in 1:K)

    # 2. h_raw and h_tilde (need both: h_tilde for quadratic form, h_raw for obs expansion)
    h_raw, h_tilde = _PBR.compute_h_raw_and_tilde(z_raw, p, idx, spec.offsets)

    # 3. Per-firm quadratic sums A, B, C
    A, B, C = _PBR._compute_panel_ABC(epsilon_tilde, h_tilde, spec.offsets, N)

    # 4. Noise/inefficiency parameters and u draws (dispatched by distribution)
    noise_vals = _PBR.get_panel_noise_vals(model.noise, p, idx, c)
    ineff_vals = _PBR.get_panel_ineff_vals(model.ineff, p, idx, c)
    halton_P   = P.(halton)

    if spec.method == :MSLE
        u_draws   = _PBR.mypan_gen_udraws_msle(model.ineff, ineff_vals, halton_P, c, P)
        log_extra = nothing
    else
        scale = _PBR.mypan_get_mci_scale(model.ineff, ineff_vals)
        trans, jacob = _PBR.resolve_panel_transformation(spec.transformation, model)
        u_draws   = min.(trans.(halton_P, scale), inv(P(c.clamp_lo)))
        log_fu    = _PBR.mypan_logpdf_ineff(model.ineff, u_draws, ineff_vals, P(c.clamp_lo))
        log_J     = log.(max.(jacob.(halton_P, scale), P(c.clamp_lo)))
        log_fu_J  = log_fu .+ log_J
        log_extra = ndims(log_fu_J) == 1 ? reshape(log_fu_J, 1, D) : log_fu_J
    end

    # 5. Log-weights (N × D) and denominator
    inv_sigma_v_sq = noise_vals.inv_sigma_v_sq
    log_sigma_v_sq = noise_vals.log_sigma_v_sq
    Tm1_cpu        = _PBR._to_cpu(spec.Tm1)
    log_const      = P(-0.5) .* Tm1_cpu .* (log(P(2π)) .+ log_sigma_v_sq)
    frontier_sign  = P(spec.sign)

    u_row   = ndims(u_draws) == 1 ? reshape(u_draws, 1, D) : u_draws  # 1×D or N×D
    two_s   = P(2) * frontier_sign
    quad_2D = A .+ two_s .* B .* u_row .+ C .* u_row .^ 2       # N × D

    log_w = P(-0.5) .* quad_2D .* inv_sigma_v_sq .+ log_const
    if log_extra !== nothing
        log_w = log_w .+ log_extra
    end

    log_denom = _PBR.logsumexp_rows_panel(log_w)                       # N

    # 6. JLMS: firm-level E[u_i*], then expand to obs-level via h_raw
    log_u_row    = log.(max.(u_row, P(c.clamp_lo)))               # 1 × D (precomputed)
    log_u_w      = log_u_row .+ log_w                             # N × D
    log_jlms_num = _PBR.logsumexp_rows_panel(log_u_w)                  # N
    E_u_firm     = exp.(log_jlms_num .- log_denom)                # N

    jlms = _PBR._expand_jlms_obs(E_u_firm, h_raw, spec)

    # 7. BC: obs-level E[exp(-h_it * u_i*)]
    bc = _ref_compute_bc_obs(log_w, log_denom, h_raw, u_draws, spec)

    return (jlms=jlms, bc=bc)
end

# A per-firm draw block that refuses access to the whole block and logs the rows the package reads through
# `_panel_draw_rows` (test_jlms_bc_chunks.jl: the panel efficiency indices never read the whole block at once).
# Defined in this shared file so that the test file's top level holds only its using lines and this file's include.
struct _RowLog <: AbstractMatrix{Float64}
    block::Matrix{Float64}
    spans::Vector{Int}
end
Base.size(d::_RowLog) = size(d.block)
Base.getindex(d::_RowLog, i::Int, j::Int) = error("whole-block access")
SFrontiers.Panel_Backend._panel_draw_rows(d::_RowLog, a::Int, b::Int) = (push!(d.spans, b - a + 1); view(d.block, a:b, :))
