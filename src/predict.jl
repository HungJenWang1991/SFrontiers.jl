# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later
#
# `predict(::SFResult; ...)` — application of a fitted stochastic frontier
# model to new observations. Implemented for cross-sectional models
# (MCI / MSLE / MLE) with Normal noise and HalfNormal or TruncatedNormal
# inefficiency, which collectively cover the paper's and the manual's
# worked examples (rice-farmer Wang 2002). Other distributions and
# panel datatypes raise an informative error.
#
# Dispatch targets (see Table in §3.12):
#   :frontier   -> x'β̂                              (needs frontier)
#   :residuals  -> y - x'β̂                          (needs frontier, depvar)
#   :response   -> x'β̂ - sgn·û                      (needs frontier, depvar, zvar if in model)
#   :jlms       -> E(u|ε)                            (needs frontier, depvar, zvar if in model)
#   :bc         -> E(exp(-u)|ε)                      (needs frontier, depvar, zvar if in model)
#   :marginal   -> ∂E(u)/∂z                          (needs zvar)
#   :all        -> NamedTuple packaging the above

using StatsFuns: normpdf, normcdf, normccdf, normlogpdf, normlogcdf
using ForwardDiff: gradient

"""
    predict(r::SFResult;
            frontier = nothing,
            zvar     = nothing,
            depvar   = nothing,
            id       = nothing,
            what     = :frontier,
            draws    = nothing,
            level    = nothing)

Apply a fitted stochastic frontier model to new observations.

The set of required inputs depends on `what`:

| `what`       | frontier | zvar¹ | depvar |
| ------------ | :------: | :---: | :----: |
| `:frontier`  |    ✓     |       |        |
| `:residuals` |    ✓     |       |   ✓    |
| `:response`  |    ✓     |  ✓¹  |   ✓    |
| `:jlms`      |    ✓     |  ✓¹  |   ✓    |
| `:bc`        |    ✓     |  ✓¹  |   ✓    |
| `:marginal`  |          |   ✓   |        |
| `:all`       |    ✓     |  ✓¹  |   ✓    |

¹ `zvar` is required only when the fitted model uses exogenous
determinants (`hetero` or scaling).

`draws` (optional) supplies the fit-time Halton matrix for MCI-based
`:jlms` / `:bc` / `:response` / `:all` targets. Shape: `N_train × n_draws`
for `multiRand=true`, or `1 × n_draws` (or a length-`n_draws` vector) for
`multiRand=false`. When omitted, predict reconstructs the fit-time Halton
matrix deterministically. Required only when the fit itself was run with
user-supplied `draws=` in `sfmodel_method` — in which case the draws are
not stored on the result and must be passed back in.

Supported distributions: Normal noise with `:HalfNormal` or
`:TruncatedNormal` inefficiency (homoscedastic, heteroscedastic, or
scaling-property forms). For other combinations, `predict` raises
an informative error. Panel datatypes are not yet supported.
"""
function StatsAPI.predict(r::SFResult;
        frontier = nothing,
        zvar     = nothing,
        depvar   = nothing,
        id       = nothing,
        what::Symbol = :frontier,
        draws    = nothing,
        level    = nothing)

    _predict_check_panel(r)
    spec = _predict_get_unified_spec(r)
    spec === nothing &&
        error("predict: no cross-sectional spec is attached to this result. " *
              "predict() currently supports cross-sectional models only.")

    if what === :frontier
        Xn = _require_matrix(frontier, "frontier")
        _check_K(Xn, spec)
        β = _get_frontier_coef(r, size(Xn, 2))
        return Xn * β

    elseif what === :residuals
        Xn = _require_matrix(frontier, "frontier")
        yn = _require_vector(depvar, "depvar")
        _check_K(Xn, spec)
        _check_len(yn, Xn)
        β = _get_frontier_coef(r, size(Xn, 2))
        return yn .- Xn * β

    elseif what === :response
        û = StatsAPI.predict(r; frontier = frontier, zvar = zvar, depvar = depvar,
                             id = id, what = :jlms, draws = draws)
        Xn = _require_matrix(frontier, "frontier")
        β  = _get_frontier_coef(r, size(Xn, 2))
        sgn = _frontier_sign(r)
        return Xn * β .- sgn .* û

    elseif what === :jlms
        return _dispatch_jlms_bc(r, spec, frontier, zvar, depvar;
                                 bc = false, user_draws = draws, id = id)

    elseif what === :bc
        return _dispatch_jlms_bc(r, spec, frontier, zvar, depvar;
                                 bc = true, user_draws = draws, id = id)

    elseif what === :marginal
        Zn = _require_matrix(zvar, "zvar")
        return _predict_marginal(r, spec, Zn)

    elseif what === :all
        ε   = StatsAPI.predict(r; frontier = frontier, depvar = depvar, what = :residuals)
        ŷ   = StatsAPI.predict(r; frontier = frontier, what = :frontier)
        û   = StatsAPI.predict(r; frontier = frontier, zvar = zvar, depvar = depvar,
                               id = id, what = :jlms, draws = draws)
        b   = StatsAPI.predict(r; frontier = frontier, zvar = zvar, depvar = depvar,
                               id = id, what = :bc, draws = draws)
        sgn = _frontier_sign(r)
        yrs = ŷ .- sgn .* û
        if zvar !== nothing && _hetero_needs_zvar(spec)
            m = StatsAPI.predict(r; zvar = zvar, what = :marginal)
            return (frontier = ŷ, residuals = ε, jlms = û, bc = b,
                    response = yrs, marginal = m)
        end
        return (frontier = ŷ, residuals = ε, jlms = û, bc = b, response = yrs)

    else
        error("predict: unknown `what=:$what`. Valid: :frontier, :residuals, " *
              ":response, :jlms, :bc, :marginal, :all.")
    end
end

# ==============================================================
# Spec access and validation
# ==============================================================

function _predict_get_unified_spec(r::SFResult)
    s = getfield(r, :spec)
    return s
end

function _predict_check_panel(r::SFResult)
    spec = _predict_get_unified_spec(r)
    spec === nothing && return
    hasproperty(spec, :datatype) || return
    dt = spec.datatype
    dt === :cross_sectional && return
    dt === :panel_TFE && return       # Panel_Backend, MCI/MSLE
    dt === :panel_TFE_CSW && return   # MLE (Chen-Schmidt-Wang 2014)
    dt === :panel_TRE && return       # MLE (Greene 2005 true random-effect)
    error("predict: datatype `:$dt` is not yet supported. Currently supported: " *
          "`:cross_sectional`, `:panel_TFE`, `:panel_TFE_CSW`, `:panel_TRE`.")
end

function _require_matrix(X, name::String)
    X === nothing && error("predict: keyword `$name` is required for this target.")
    X isa AbstractMatrix || error("predict: `$name` must be a matrix.")
    return X
end

function _require_vector(v, name::String)
    v === nothing && error("predict: keyword `$name` is required for this target.")
    v isa AbstractVector || error("predict: `$name` must be a vector.")
    return v
end

function _check_K(Xn, spec)
    Ktrain = if spec.mci_spec !== nothing
        size(spec.mci_spec.frontier, 2)
    elseif hasproperty(spec, :panel_spec) && spec.panel_spec !== nothing
        size(spec.panel_spec.frontier, 2)
    elseif hasproperty(spec, :mle_spec) && spec.mle_spec !== nothing
        size(spec.mle_spec.frontier, 2)
    else
        nothing
    end
    Ktrain === nothing && return
    size(Xn, 2) == Ktrain ||
        error("predict: `frontier` has $(size(Xn,2)) columns but model was fitted " *
              "with $Ktrain. Include/exclude the constant column as at fit time.")
end

_check_len(y, X) =
    length(y) == size(X, 1) ||
    error("predict: length(depvar)=$(length(y)) does not match size(frontier,1)=$(size(X,1)).")

function _hetero_needs_zvar(spec)
    spec === nothing && return false
    h = spec.hetero
    return !(h isa AbstractVector && isempty(h))
end

# ==============================================================
# Closed-form inefficiency / efficiency indices
# ==============================================================

function _predict_eff(r::SFResult, spec, Xn, Zn_in, yn_in; bc::Bool)
    spec.noise === :Normal ||
        error("predict(:jlms/:bc) currently supports `noise=:Normal` only.")
    ineff = spec.ineff
    ineff ∈ (:HalfNormal, :TruncatedNormal) ||
        error("predict(:jlms/:bc) currently supports `ineff=:HalfNormal` or " *
              "`:TruncatedNormal`; got `:$ineff`.")

    Xn = _require_matrix(Xn, "frontier")
    yn = _require_vector(yn_in, "depvar")
    _check_K(Xn, spec)
    _check_len(yn, Xn)

    Zn = _hetero_needs_zvar(spec) ? _require_matrix(Zn_in, "zvar") : zeros(size(Xn, 1), 0)
    _hetero_needs_zvar(spec) && _check_len(view(Zn, :, 1), Xn)

    β    = _get_frontier_coef(r, size(Xn, 2))
    σv²_raw  = _get_sigma_v_sq(r, spec)
    σv²  = clamp.(σv²_raw, 1e-12, 1e12)
    μvec, σu²vec = _get_mu_sigma_u_sq(r, spec, Zn, ineff)
    σu² = clamp.(σu²vec, 1e-12, 1e12)

    sgn = _frontier_sign(r)
    ε   = yn .- Xn * β            # raw residual (sign-neutral)
    ε_p = sgn .* ε                # production-convention ε' = v - u
    s²  = σu² .+ σv²
    μs  = (μvec .* σv² .- ε_p .* σu²) ./ s²
    σs  = sqrt.(σu² .* σv² ./ s²)
    a   = μs ./ σs

    # Log-space JLMS/BC to match MLE SFindex.jl (see src/sf_MLE/SFindex.jl:20-31 / 57-59).
    # This parallels the MLE fit-time routine bit-for-bit; for MCI/MSLE fits,
    # r.jlms came from simulation (Part 3 closed-form regression only checks
    # finiteness), so there is no bit-identity regression.
    if !bc
        return σs .* exp.(normlogpdf.(a) .- normlogcdf.(a)) .+ μs
    else
        expo = -μs .+ 0.5 .* σs .^ 2 .+ normlogcdf.(a .- σs) .- normlogcdf.(a)
        return exp.(clamp.(expo, -500.0, 500.0))
    end
end

# ==============================================================
# Normal + Exponential closed form for MLE fits
# ==============================================================
#
# Ported from src/sf_MLE/SFindex.jl:70-92 (function
# `jlmsbc(::Type{Expo}, ...)`). Uses the MLE parameter scale:
# `r.coeff[idx.ineff.lambda[1]]` is ln(λ²), so λ = sqrt(exp(w_pre)).
# This helper is therefore valid ONLY for r.backend === :MLE fits;
# the dispatcher guards it. Clamp constants match _MLE_CLAMP in
# src/sf_MLE/SFloglikefun.jl:18-22 (exp_lo=1e-12, exp_hi=1e12).

function _predict_eff_mle_exp(r::SFResult, spec, Xn_in, yn_in; bc::Bool)
    mci = spec.mci_spec
    mci === nothing &&
        error("predict: MLE+Exponential path requires an attached cross-sectional " *
              "spec; none found.")

    Xn = _require_matrix(Xn_in, "frontier")
    yn = _require_vector(yn_in, "depvar")
    _check_K(Xn, spec)
    _check_len(yn, Xn)

    β   = _get_frontier_coef(r, size(Xn, 2))
    sgn = _frontier_sign(r)
    ε   = sgn .* (yn .- Xn * β)   # production convention ε = v − u

    lambda_rng = _as_range(mci.idx.ineff.lambda)
    length(lambda_rng) == 1 ||
        error("predict: MLE+Exp helper expects homoscedastic λ (length(idx)==1); " *
              "got length $(length(lambda_rng)). Hetero-λ is not fittable by MLE.")

    w_pre = r.coeff[lambda_rng][1]
    v_pre = r.coeff[_as_range(mci.idx.noise.ln_sigma_v_sq)][1]

    λ   = sqrt(clamp(exp(w_pre), 1e-12, 1e12))
    σᵥ² = clamp(exp(v_pre), 1e-12, 1e12)
    σᵥ  = sqrt(σᵥ²)

    μs = -ε .- σᵥ² / λ
    rv = μs ./ σᵥ

    if !bc
        return σᵥ .* exp.(normlogpdf.(rv) .- normlogcdf.(rv)) .+ μs
    else
        expo = -μs .+ 0.5 * σᵥ² .+ normlogcdf.(rv .- σᵥ) .- normlogcdf.(rv)
        return exp.(clamp.(expo, -500.0, 500.0))
    end
end

# ==============================================================
# Simulation-based inefficiency / efficiency indices (MCI / MSLE)
# ==============================================================
#
# Reuses the fit-time simulation routine that populates r.jlms /
# r.bc at fit time. Covers every cross-sectional noise × ineff
# combination outside the Normal + HalfNormal / TruncatedNormal
# closed form, including scaling-property and copula models.
#
# Halton draws are reconstructed deterministically at predict time
# (Halton base 2) using the training N, then row-aligned to the
# predict-time N via `_align_draws`:
#   N_new == N_train → use matrix as-is
#   N_new  < N_train → take rows 1:N_new (view, no copy)
#   N_new  > N_train → recycle row i ← fit row mod1(i, N_train)
# Consequence: for any predict observation whose index maps to fit
# row k via mod1, the computed jlms/bc match r.jlms[k] / r.bc[k]
# bit-for-bit when ε also matches. The same alignment applies when
# the user forwards a custom `draws=` matrix to predict().
#
# Scaling: u_i = h(z_i) · u*_i where h(z) = exp(z · δ). At predict
# time the user passes `zvar=Z_s` (no-constant, L_scaling columns);
# we compute `h = exp.(Z_s · δ)` and pass as `scaling_h` to the
# simulation routine. The base ineff distribution is homoscedastic,
# so its zvar is the fit-time convention `ones(N_new, 1)`.
#
# Copula: `copula_vals` is a NamedTuple rebuilt from fitted
# `theta_rho` via `get_copula_vals(model.copula, p, idx, c)`. No
# separate draws are needed — the same Halton stream serves both.

_eff_use_closed_form(spec) =
    spec.noise === :Normal && spec.ineff in (:HalfNormal, :TruncatedNormal)

_frontier_type_from_sign(sgn::Real) = (sgn == -1 ? :cost : :prod)

# Route :jlms / :bc by backend. Panel_TFE goes to the panel helper (any ineff);
# cross-section HN/TN go to the shared closed form; everything else dispatches
# on r.backend.
function _dispatch_jlms_bc(r::SFResult, spec, Xn, Zn, yn; bc::Bool, user_draws, id = nothing)
    if hasproperty(spec, :datatype) && spec.datatype === :panel_TFE
        if r.backend === :Panel
            return _predict_eff_panel_tfe(r, spec, Xn, Zn, yn, id;
                                          bc = bc, user_draws = user_draws)
        end
        error("predict: panel_TFE fits are supported via the Panel backend " *
              "(`r.backend === :Panel`). Got `r.backend === :$(r.backend)`; " *
              "MLE-based panel predict is deferred to a later round.")
    end
    if hasproperty(spec, :datatype) &&
       (spec.datatype === :panel_TFE_CSW || spec.datatype === :panel_TRE)
        r.backend === :MLE ||
            error("predict: `:$(spec.datatype)` fits are MLE-only " *
                  "(got backend `:$(r.backend)`).")
        return _predict_eff_mle_panel(r, spec, Xn, Zn, yn, id; bc = bc)
    end
    if _eff_use_closed_form(spec)
        return _predict_eff(r, spec, Xn, Zn, yn; bc = bc)
    end
    if r.backend === :MCI
        return _predict_eff_mci(r, spec, Xn, Zn, yn; bc = bc, user_draws = user_draws)
    end
    if r.backend === :MSLE
        return _predict_eff_msle(r, spec, Xn, Zn, yn; bc = bc, user_draws = user_draws)
    end
    if r.backend === :MLE
        if spec.noise === :Normal && spec.ineff === :Exponential
            return _predict_eff_mle_exp(r, spec, Xn, yn; bc = bc)
        end
        error("predict: MLE fit with noise=$(spec.noise), ineff=$(spec.ineff) is " *
              "not supported. MLE supports Normal + {HalfNormal, TruncatedNormal, " *
              "Exponential} only.")
    end
    error("predict: unsupported backend $(r.backend).")
end

# Align fit-time N_train × D Halton draws to predict-time N_new rows.
function _align_draws(draws_train::AbstractMatrix{T}, N_new::Int) where {T}
    N_train = size(draws_train, 1)
    N_new == N_train && return draws_train
    N_new  < N_train && return view(draws_train, 1:N_new, :)
    D = size(draws_train, 2)
    out = Matrix{T}(undef, N_new, D)
    for i in 1:N_new
        out[i, :] .= view(draws_train, mod1(i, N_train), :)
    end
    return out
end

# Validate a user-supplied `draws` matrix passed to predict().
function _validate_user_draws(draws, n_draws::Int, multiRand::Bool, N_train::Int)
    M = draws isa AbstractVector ? reshape(draws, 1, length(draws)) :
        (draws isa AbstractMatrix ? draws :
         error("predict: `draws` must be a matrix (or a vector for multiRand=false)."))
    size(M, 2) == n_draws ||
        error("predict: `draws` has $(size(M,2)) columns; fit used $n_draws.")
    if multiRand
        size(M, 1) == N_train ||
            error("predict: `draws` has $(size(M,1)) rows; expected $N_train " *
                  "(training N). Pass the same matrix used at fit time.")
    else
        size(M, 1) == 1 ||
            error("predict: `draws` for a multiRand=false fit must have 1 row, got $(size(M,1)).")
    end
    return M
end

function _predict_eff_mci(r::SFResult, spec, Xn_in, Zn_in, yn_in;
                          bc::Bool, user_draws = nothing)
    mci = spec.mci_spec
    mci === nothing &&
        error("predict: MCI-based efficiency path requires a cross-sectional " *
              "MCI spec; none attached.")

    Xn = _require_matrix(Xn_in, "frontier")
    yn = _require_vector(yn_in, "depvar")
    _check_K(Xn, spec)
    _check_len(yn, Xn)

    T    = Float64
    Xn_T = Xn isa AbstractMatrix{T} ? Xn : convert(Matrix{T}, Xn)
    yn_T = yn isa AbstractVector{T} ? yn : convert(Vector{T}, yn)
    N_new = length(yn_T)

    # Resolve predict-time zvar:
    #   - scaling fit: user's zvar IS the scaling_zvar (no-constant, L_scaling cols);
    #     the base ineff is homoscedastic so its Z is ones(N_new, 1).
    #   - hetero fit: user's zvar feeds get_ineff_vals directly.
    #   - neither:    ones(N_new, 1).
    if mci.scaling
        Zn_scaling = _require_matrix(Zn_in, "zvar")
        _check_len(view(Zn_scaling, :, 1), Xn)
        size(Zn_scaling, 2) == mci.L_scaling ||
            error("predict: scaling `zvar` has $(size(Zn_scaling,2)) columns but " *
                  "model δ expects $(mci.L_scaling). Pass the same no-constant Z " *
                  "used at fit time.")
        Zn_scaling_T = Zn_scaling isa AbstractMatrix{T} ? Zn_scaling :
                       convert(Matrix{T}, Zn_scaling)
        Zn_T = ones(T, N_new, 1)
    elseif _hetero_needs_zvar(spec)
        Zn_raw = _require_matrix(Zn_in, "zvar")
        _check_len(view(Zn_raw, :, 1), Xn)
        Zn_T = Zn_raw isa AbstractMatrix{T} ? Zn_raw : convert(Matrix{T}, Zn_raw)
        Zn_scaling_T = nothing
    else
        Zn_T = ones(T, N_new, 1)
        Zn_scaling_T = nothing
    end

    hetero_vec = spec.hetero isa Vector{Symbol} ? spec.hetero : Symbol[]

    n_draws   = Int(r.n_draws)
    multiRand = r.multiRand
    dhl       = hasproperty(r, :distinct_Halton_length) ?
                Int(r.distinct_Halton_length) : (2^15 - 1)
    N_train   = mci.N
    user_fit  = hasproperty(r, :user_draws_supplied) && r.user_draws_supplied

    if user_draws !== nothing
        M = _validate_user_draws(user_draws, n_draws, multiRand, N_train)
        M_T = M isa AbstractMatrix{T} ? M : convert(Matrix{T}, M)
        draws_1D = multiRand ? _align_draws(M_T, N_new) : M_T
    elseif user_fit
        error("predict: fit used user-supplied custom draws, which are not stored " *
              "on the result. Pass the same draws to predict() via the `draws=` " *
              "keyword (shape $N_train × $n_draws for multiRand=true, or 1 × " *
              "$n_draws for multiRand=false).")
    elseif multiRand
        draws_train = MCI_Backend.make_halton_wrap(N_train, n_draws;
                                                   T = T, distinct_Halton_length = dhl)
        draws_1D = _align_draws(draws_train, N_new)
    else
        draws_1D = reshape(MCI_Backend.make_halton_p(n_draws; T = T), 1, n_draws)
    end

    p = r.coeff isa AbstractVector{T} ? r.coeff : convert(Vector{T}, r.coeff)

    # Reuse the training-time model / idx (they encode the parameter layout
    # that matches r.coeff). Rebuilding would work too but risks drift.
    model = mci.model
    idx   = mci.idx
    K     = size(Xn_T, 2)

    c          = MCI_Backend.make_constants(model, T)
    noise_vals = MCI_Backend.get_noise_vals(model.noise, p, idx, c)
    ineff_vals = MCI_Backend.get_ineff_vals(model.ineff, p, idx, Zn_T, c)

    # Residuals ε = y - Xβ, matching the formula in MCI_nll / jlms_bc_indices.
    P_ = eltype(p)
    ε  = yn_T .- sum(P_(p[idx.beta[j]]) .* view(Xn_T, :, j) for j in 1:K)

    # Use the same default transformation rule as the fit path when the
    # user did not override `transformation` in sfmodel_method(). This is
    # what produces bit-identical predict-on-training for ineff distributions
    # whose T-approach rule differs from the direct quantile (e.g. Lognormal,
    # Lomax, HalfNormal, TruncatedNormal under :logistic_1_rule).
    trans_rule = MCI_Backend.default_transformation_rule(spec.ineff)
    trans_func, jacob_func = MCI_Backend.resolve_transformation(trans_rule)

    sgn_int = Int(_frontier_sign(r))
    D       = size(draws_1D, 2)

    # Scaling: h_i = exp(Z_s · δ) — matches fit at sf_MCI_v21.jl:3779-3784.
    scaling_h = mci.scaling ?
        exp.(Zn_scaling_T * view(p, idx.delta)) :
        nothing

    # Copula: rebuild NamedTuple from fitted theta_rho.
    copula_vals = spec.copula === :None ? NamedTuple() :
        MCI_Backend.get_copula_vals(model.copula, p, idx, c)

    out = MCI_Backend._jlms_bc_mci_T(
        ε, draws_1D, model, c, N_new, D,
        sgn_int, 1, noise_vals, ineff_vals;
        trans = trans_func, jacob = jacob_func,
        copula_vals = copula_vals,
        scaling_h   = scaling_h,
    )

    return bc ? collect(out.bc) : collect(out.jlms)
end

# ==============================================================
# MSLE-based inefficiency / efficiency indices
# ==============================================================
#
# Parallel to `_predict_eff_mci`. Uses the MSLE backend's spec-form
# `jlms_bc_indices(spec, p; chunks=1)`, which natively handles
# scaling and copula. We construct a predict-time internal
# `sfmodel_MSLE_spec` by copying the fitted model / idx / copula
# type / etc. from `spec.msle_spec` and substituting predict-time
# data + reconstructed draws.

function _predict_eff_msle(r::SFResult, spec, Xn_in, Zn_in, yn_in;
                           bc::Bool, user_draws = nothing)
    msle = spec.msle_spec
    msle === nothing &&
        error("predict: MSLE-based efficiency path requires a cross-sectional " *
              "MSLE spec; none attached.")

    Xn = _require_matrix(Xn_in, "frontier")
    yn = _require_vector(yn_in, "depvar")
    _check_K(Xn, spec)
    _check_len(yn, Xn)

    T    = Float64
    Xn_T = Xn isa AbstractMatrix{T} ? Xn : convert(Matrix{T}, Xn)
    yn_T = yn isa AbstractVector{T} ? yn : convert(Vector{T}, yn)
    N_new = length(yn_T)

    # Same zvar resolution as MCI path.
    if msle.scaling
        Zn_scaling = _require_matrix(Zn_in, "zvar")
        _check_len(view(Zn_scaling, :, 1), Xn)
        size(Zn_scaling, 2) == msle.L_scaling ||
            error("predict: scaling `zvar` has $(size(Zn_scaling,2)) columns but " *
                  "model δ expects $(msle.L_scaling). Pass the same no-constant Z " *
                  "used at fit time.")
        Zn_scaling_T = Zn_scaling isa AbstractMatrix{T} ? Zn_scaling :
                       convert(Matrix{T}, Zn_scaling)
        Zn_T = ones(T, N_new, 1)
    elseif _hetero_needs_zvar(spec)
        Zn_raw = _require_matrix(Zn_in, "zvar")
        _check_len(view(Zn_raw, :, 1), Xn)
        Zn_T = Zn_raw isa AbstractMatrix{T} ? Zn_raw : convert(Matrix{T}, Zn_raw)
        Zn_scaling_T = nothing
    else
        Zn_T = ones(T, N_new, 1)
        Zn_scaling_T = nothing
    end

    n_draws   = Int(r.n_draws)
    multiRand = r.multiRand
    dhl       = hasproperty(r, :distinct_Halton_length) ?
                Int(r.distinct_Halton_length) : (2^15 - 1)
    N_train   = msle.N
    user_fit  = hasproperty(r, :user_draws_supplied) && r.user_draws_supplied

    if user_draws !== nothing
        M = _validate_user_draws(user_draws, n_draws, multiRand, N_train)
        M_T = M isa AbstractMatrix{T} ? M : convert(Matrix{T}, M)
        draws_2D = multiRand ? _align_draws(M_T, N_new) : M_T
    elseif user_fit
        error("predict: fit used user-supplied custom draws, which are not stored " *
              "on the result. Pass the same draws to predict() via the `draws=` " *
              "keyword (shape $N_train × $n_draws for multiRand=true, or 1 × " *
              "$n_draws for multiRand=false).")
    elseif multiRand
        draws_train = MSLE_Backend.make_halton_wrap(N_train, n_draws;
                                                    T = T, distinct_Halton_length = dhl)
        draws_2D = _align_draws(draws_train, N_new)
    else
        draws_2D = reshape(MSLE_Backend.make_halton_p(n_draws; T = T), 1, n_draws)
    end

    # Ensure draws_2D is a concrete Matrix{T} (the MSLE spec-form reads it as
    # AbstractMatrix, but some of its internal allocations derive from it).
    draws_2D_T = draws_2D isa Matrix{T} ? draws_2D : collect(draws_2D)
    # `draws` field on the internal spec is a flat 1D view of the matrix.
    draws_1D_flat = vec(draws_2D_T)

    p = r.coeff isa AbstractVector{T} ? r.coeff : convert(Vector{T}, r.coeff)

    model = msle.model
    idx   = msle.idx
    K     = size(Xn_T, 2)
    L     = size(Zn_T, 2)
    sgn   = Int(_frontier_sign(r))

    c = MSLE_Backend.make_constants(model, T)

    # Hetero as Vector{Symbol} (internal spec expects concrete vector).
    hetero_vec_msle = msle.hetero isa Vector{Symbol} ? msle.hetero : Symbol[]

    # Build predict-time internal spec and delegate to the spec-form.
    internal_spec = MSLE_Backend.sfmodel_MSLE_spec{T}(
        yn_T, Xn_T, Zn_T,
        msle.noise, msle.ineff, msle.copula, hetero_vec_msle,
        draws_1D_flat, draws_2D_T, multiRand, c,
        msle.varnames, msle.eqnames, msle.eq_indices,
        N_new, K, L, model, idx, sgn, 1,
        msle.scaling, Zn_scaling_T, msle.L_scaling,
    )

    out = MSLE_Backend.jlms_bc_indices(internal_spec, p; chunks = 1)

    return bc ? collect(out.bc) : collect(out.jlms)
end

# ==============================================================
# Panel TFE (Wang and Ho 2010) — MCI / MSLE
# ==============================================================
#
# Mirrors `_predict_eff_msle`: reconstruct fit-time Halton draws, build a
# predict-time `_PanelInternalSpec`, and delegate to `panel_jlms_bc`.
#
# Rule A firm alignment: predict firms that share an id with a fit firm
# reuse that fit firm's draws row; brand-new firms get independent Halton
# rows via `make_panel_halton_wrap`. This guarantees bit-identity on the
# training intersection regardless of row ordering in the predict data.
#
# `id=` is required (each observation's firm identifier). Rows must be
# grouped contiguously by id (same convention the fit path enforces via
# `_compute_panel_structure`).
#
# `multiRand=false`: shared 1-D draws are firm-independent by construction;
# Rule A is a no-op (all firms use the same draws).
#
# Transformation rule: `r.transformation` is the user-specified rule from
# fit (`nothing` = distribution default). Forwarded verbatim to the panel
# internal spec so `resolve_panel_transformation` reproduces fit-time
# behavior.
#
# User-supplied `draws=` at fit time: the fit stores nothing but a flag
# `r.user_draws_supplied`. predict() requires the user to pass the same
# `draws=` matrix (shape `N_firms_fit × n_draws` for multiRand=true).

function _predict_eff_panel_tfe(r::SFResult, spec, Xn_in, Zn_in, yn_in, id_in;
                                bc::Bool, user_draws = nothing)
    pspec = spec.panel_spec
    pspec === nothing &&
        error("predict: panel_TFE fit requires an attached panel spec; none found.")

    id_in === nothing &&
        error("predict: panel fits require the `id=` keyword (firm identifier " *
              "for each observation).")

    Xn = _require_matrix(Xn_in, "frontier")
    yn = _require_vector(yn_in, "depvar")
    Zn = _require_matrix(Zn_in, "zvar")
    id_vec = _require_vector(id_in, "id")

    size(Xn, 2) == pspec.K ||
        error("predict: `frontier` has $(size(Xn,2)) columns but panel model " *
              "was fitted with $(pspec.K).")
    size(Zn, 2) == pspec.L ||
        error("predict: `zvar` has $(size(Zn,2)) columns but panel model " *
              "was fitted with $(pspec.L).")
    _check_len(yn, Xn)
    length(id_vec) == size(Xn, 1) ||
        error("predict: `id` length $(length(id_vec)) ≠ rows of frontier " *
              "$(size(Xn,1)).")

    T      = Float64
    Xn_T   = Xn isa AbstractMatrix{T} ? Xn : convert(Matrix{T}, Xn)
    yn_T   = yn isa AbstractVector{T} ? yn : convert(Vector{T}, yn)
    Zn_T   = Zn isa AbstractMatrix{T} ? Zn : convert(Matrix{T}, Zn)
    NT_new = length(yn_T)

    # Predict-time panel structure (enforces contiguous id-grouping).
    N_predict, T_vec_predict, offsets_predict =
        Panel_Backend._compute_panel_structure(id_vec)
    firm_ids_predict = unique(id_vec)   # same first-appearance order as offsets
    T_max_predict    = maximum(T_vec_predict)

    # Reconstruct fit-time draws (N_firms_fit × D for multiRand=true; length-D
    # for multiRand=false).
    n_draws   = Int(r.n_draws)
    multiRand = r.multiRand
    dhl       = hasproperty(r, :distinct_Halton_length) ?
                Int(r.distinct_Halton_length) : (2^15 - 1)
    N_fit     = pspec.N
    user_fit  = hasproperty(r, :user_draws_supplied) && r.user_draws_supplied

    draws_fit = _resolve_panel_fit_draws(r, user_draws, user_fit, n_draws,
                                         multiRand, dhl, N_fit, T)

    # Rule A: match predict firms by id; brand-new firms get independent Halton
    # rows. For multiRand=false, draws are shared across firms — pass through.
    draws_predict = if multiRand
        _build_predict_panel_draws(draws_fit, pspec.firm_ids, firm_ids_predict,
                                   n_draws, dhl, T)
    else
        draws_fit
    end

    # Build predict-time internal spec mirroring `_assemble_panel_spec`:
    # demean y/X within firm, keep z raw (h(z) is demeaned inside the kernel).
    y_tilde = Panel_Backend.sf_panel_demean(yn_T, offsets_predict)
    x_tilde = Panel_Backend.sf_panel_demean(Xn_T, offsets_predict)
    Tm1     = T.(T_vec_predict .- 1)

    constants = Panel_Backend.make_panel_constants(pspec.model, T, T_max_predict)
    K = pspec.K
    L = pspec.L

    method_sym = hasproperty(r, :estimation_method) ? Symbol(r.estimation_method) :
                 Symbol(pspec.ineff == :Gamma ? :MCI : :MSLE)
    trans_sym  = hasproperty(r, :transformation) ? r.transformation : nothing

    internal_spec = Panel_Backend._PanelInternalSpec{T}(
        y_tilde, x_tilde, Zn_T,
        N_predict, T_vec_predict, T_max_predict, offsets_predict, Tm1,
        draws_predict,
        pspec.model, pspec.noise, pspec.ineff,
        K, L, pspec.idx, pspec.sign,
        1,                                # chunks (spec-form accepts this only for API parity)
        constants,
        pspec.varnames, pspec.eqnames, pspec.eq_indices,
        method_sym, trans_sym, false,     # GPU=false for predict
    )

    p = r.coeff isa AbstractVector{T} ? r.coeff : convert(Vector{T}, r.coeff)
    out = Panel_Backend.panel_jlms_bc(internal_spec, p; chunks = 1)
    return bc ? collect(out.bc) : collect(out.jlms)
end

# Reconstruct the fit-time panel Halton draws matrix bit-for-bit.
# Returns `N_fit × D` (multiRand=true) or length-`D` vector (multiRand=false),
# matching the shape `_assemble_panel_spec` produced at fit time.
function _resolve_panel_fit_draws(r::SFResult, user_draws, user_fit::Bool,
                                  n_draws::Int, multiRand::Bool, dhl::Int,
                                  N_fit::Int, ::Type{T}) where {T}
    if user_draws !== nothing
        if multiRand
            U = user_draws isa AbstractMatrix ? user_draws :
                error("predict: `draws` for multiRand=true must be a matrix " *
                      "(expected $N_fit × $n_draws).")
            size(U) == (N_fit, n_draws) ||
                error("predict: `draws` has shape $(size(U)); expected " *
                      "($N_fit, $n_draws) — the fit-time per-firm Halton matrix.")
            return U isa AbstractMatrix{T} ? copy(U) : convert(Matrix{T}, U)
        else
            if user_draws isa AbstractVector
                length(user_draws) == n_draws ||
                    error("predict: `draws` vector has length $(length(user_draws)); " *
                          "expected $n_draws.")
                return user_draws isa AbstractVector{T} ? copy(user_draws) :
                       convert(Vector{T}, user_draws)
            elseif user_draws isa AbstractMatrix
                size(user_draws) == (1, n_draws) || size(user_draws) == (n_draws, 1) ||
                    error("predict: `draws` for multiRand=false must be a " *
                          "length-$n_draws vector or a 1×$n_draws / $n_draws×1 matrix.")
                return vec(user_draws isa AbstractMatrix{T} ? user_draws :
                           convert(Matrix{T}, user_draws))
            else
                error("predict: `draws` for multiRand=false must be a vector or matrix.")
            end
        end
    end

    user_fit &&
        error("predict: fit used user-supplied custom draws, which are not stored " *
              "on the result. Pass the same draws to predict() via the `draws=` " *
              "keyword (shape $N_fit × $n_draws for multiRand=true, or a " *
              "length-$n_draws vector for multiRand=false).")

    if multiRand
        return Panel_Backend.make_panel_halton_wrap(N_fit, n_draws;
                                                    T = T,
                                                    distinct_Halton_length = dhl)
    else
        return Panel_Backend.make_panel_halton(n_draws; T = T)
    end
end

# Build the predict-time `N_predict × D` draws matrix under Rule A: each
# predict firm that shares an id with a fit firm reuses that firm's row;
# brand-new firms receive independent Halton rows.
function _build_predict_panel_draws(draws_fit::AbstractMatrix{T},
                                    fit_ids::AbstractVector,
                                    predict_ids::AbstractVector,
                                    n_draws::Int, dhl::Int, ::Type{T}) where {T}
    fit_id_to_row = Dict{eltype(fit_ids), Int}()
    for (i, id) in enumerate(fit_ids)
        fit_id_to_row[id] = i
    end

    N_predict = length(predict_ids)
    out = Matrix{T}(undef, N_predict, n_draws)
    new_firm_idxs = Int[]
    for (i, id) in enumerate(predict_ids)
        j = get(fit_id_to_row, id, 0)
        if j > 0
            out[i, :] .= view(draws_fit, j, :)
        else
            push!(new_firm_idxs, i)
        end
    end
    if !isempty(new_firm_idxs)
        draws_new = Panel_Backend.make_panel_halton_wrap(length(new_firm_idxs), n_draws;
                                                          T = T,
                                                          distinct_Halton_length = dhl)
        for (k, i) in enumerate(new_firm_idxs)
            out[i, :] .= view(draws_new, k, :)
        end
    end
    return out
end

# ==============================================================
# Panel TFE_CSW (Chen-Schmidt-Wang 2014) and Panel TRE (Greene 2005) — MLE
# ==============================================================
#
# Both datatypes are fit through MLE_Backend and have closed-form fit-time
# JLMS/BC helpers in `src/sf_MLE/SFindex.jl`:
#   * `jlmsbc(::Type{PFECSWH}, ...)` — CSW + HalfNormal (within-firm residual
#     mean adjustment plus standard JLMS/BC expressions on the demeaned ε).
#   * `jlmsbc(::Type{PTREH}, ...)` / `jlmsbc(::Type{PTRET}, ...)` — TRE with
#     HalfNormal / TruncatedNormal inefficiency; per-firm 1-D `quadgk`
#     integration over the random effect.
#
# Both fit paths run entirely on the caller's data; no Halton draws are
# involved. Re-running the same helper at predict time with the same
# coefficients and a rebuilt `rowIDT` reproduces `r.jlms` / `r.bc` up to
# floating-point roundoff — bit-identity, not simulation agreement.
#
# `id=` is required (each observation's firm identifier); the helper uses
# `MLE_Backend.get_rowIDT` to build the Nx2 per-firm index matrix the fit-
# time code expects. For CSW, Y and X are within-firm demeaned before the
# call (mirroring `SFgetvars.jl:1003-1012`); for TRE, Y and X are raw.
#
# The UnifiedSpec wrapper enforces homoscedasticity for all three variants
# (`SFrontiers.jl:237-239`), so μ (for PTRET), σᵤ², σᵥ², σₐ² are scalar
# intercepts. Concretely: `_dicM[:μ] = [:_cons]` for PTRET
# (`SFmainfun.jl:333`), so the fit-time Z matrix passed to jlmsbc is
# `ones(n, 1)` — not the user's `zvar=` from sfmodel_spec. We mirror that
# exactly and ignore `zvar` at predict time for all three. The `q, w, v`
# positional matrices are vestigial (the helpers read `coef[pos.beg*]`
# directly); pass `ones(n, 1)` placeholders to match the fit-time shape.

function _predict_eff_mle_panel(r::SFResult, spec, Xn_in, Zn_in, yn_in, id_in;
                                bc::Bool)
    mid = r.modelid
    mid in (MLE_Backend.PFECSWH, MLE_Backend.PTREH, MLE_Backend.PTRET) ||
        error("predict: unsupported MLE panel modelid $mid. Supported: " *
              "PFECSWH (CSW + HN), PTREH (TRE + HN), PTRET (TRE + TN).")

    Xn = _require_matrix(Xn_in, "frontier")
    yn = _require_vector(yn_in, "depvar")
    id_in === nothing &&
        error("predict: `id=` is required for panel_TFE_CSW / panel_TRE fits " *
              "(firm identifier for each observation).")
    id_vec = _require_vector(id_in, "id")
    _check_K(Xn, spec)
    _check_len(yn, Xn)
    length(id_vec) == size(Xn, 1) ||
        error("predict: `id` length $(length(id_vec)) ≠ rows of frontier " *
              "$(size(Xn, 1)).")

    T    = Float64
    Xn_T = Xn isa AbstractMatrix{T} ? Xn : convert(Matrix{T}, Xn)
    yn_T = yn isa AbstractVector{T} ? yn : convert(Vector{T}, yn)

    # Rebuild `pos::NamedTuple` from `r.eqpo` (the ranges stored at fit time).
    # Unused fields default to 0, mirroring the fit-time initialization in
    # SFgetvars.jl:939 / 1074 / 1197.
    ep = r.eqpo
    hasproperty(ep, :coeff_frontier) && hasproperty(ep, :coeff_log_σᵤ²) &&
        hasproperty(ep, :coeff_log_σᵥ²) ||
        error("predict: `r.eqpo` is missing expected fields (coeff_frontier / " *
              "coeff_log_σᵤ² / coeff_log_σᵥ²). Got keys $(keys(ep)).")
    begx = first(ep.coeff_frontier);  endx = last(ep.coeff_frontier)
    begw = first(ep.coeff_log_σᵤ²);   endw = last(ep.coeff_log_σᵤ²)
    begv = first(ep.coeff_log_σᵥ²);   endv = last(ep.coeff_log_σᵥ²)
    begz = endz = 0
    begq = endq = 0
    if mid in (MLE_Backend.PTREH, MLE_Backend.PTRET)
        hasproperty(ep, :coeff_log_σₐ²) ||
            error("predict: TRE fit is missing `coeff_log_σₐ²` on r.eqpo.")
        begq = first(ep.coeff_log_σₐ²); endq = last(ep.coeff_log_σₐ²)
    end
    if mid === MLE_Backend.PTRET
        hasproperty(ep, :coeff_μ) ||
            error("predict: PTRET fit is missing `coeff_μ` on r.eqpo.")
        begz = first(ep.coeff_μ); endz = last(ep.coeff_μ)
    end
    pos = (begx = begx, endx = endx, begz = begz, endz = endz,
           begq = begq, endq = endq, begw = begw, endw = endw,
           begv = begv, endv = endv)

    rowIDT = MLE_Backend.get_rowIDT(id_vec)
    n      = length(yn_T)

    # CSW demeans Y and X within firm (per-firm residual mean is part of the
    # jlmsbc formula). TRE uses raw values.
    Yc, Xc = if mid === MLE_Backend.PFECSWH
        D  = hcat(yn_T, Xn_T)
        Dd = similar(D)
        for i in 1:size(rowIDT, 1)
            idx_i = rowIDT[i, 1]
            @views Dd[idx_i, :] = MLE_Backend.sf_demean(D[idx_i, :])
        end
        (Dd[:, 1], Dd[:, 2:end])
    else
        (yn_T, Xn_T)
    end

    # Dummy placeholders — q/w/v ignored by helpers; Zn is the μ regressor
    # for PTRET, which is always ones(n, 1) under the UnifiedSpec-enforced
    # homoscedasticity.
    q = ones(T, n, 1)
    w = ones(T, n, 1)
    v = ones(T, n, 1)
    Zn_T = ones(T, n, 1)

    p = r.coeff isa AbstractVector{T} ? r.coeff : convert(Vector{T}, r.coeff)

    out_jlms, out_bc = if mid === MLE_Backend.PFECSWH
        MLE_Backend.jlmsbc(MLE_Backend.PFECSWH, Int(r.PorC), pos, p,
                           Yc, Xc, Zn_T, q, w, v, rowIDT)
    elseif mid === MLE_Backend.PTREH
        MLE_Backend.jlmsbc(MLE_Backend.PTREH, Int(r.PorC), pos, p,
                           reshape(Yc, :, 1), Xc, Zn_T, q, w, v, rowIDT)
    else # PTRET
        MLE_Backend.jlmsbc(MLE_Backend.PTRET, Int(r.PorC), pos, p,
                           reshape(Yc, :, 1), Xc, Zn_T, q, w, v, rowIDT)
    end
    return bc ? collect(vec(out_bc)) : collect(vec(out_jlms))
end

# ==============================================================
# Marginal effects at new z
# ==============================================================

function _predict_marginal(r::SFResult, spec, Zn::AbstractMatrix)
    ineff = spec.ineff
    ineff ∈ (:HalfNormal, :TruncatedNormal) ||
        error("predict(:marginal) currently supports `ineff=:HalfNormal` or " *
              "`:TruncatedNormal`; got `:$ineff`.")

    scaling = spec.hetero === :scaling
    # γ coefficients on z: for scaling, the δ vector; for hetero params,
    # coefs for ln(σ_u²) and (for TN) for μ.
    n = size(Zn, 1)
    L = size(Zn, 2)
    out = Matrix{Float64}(undef, n, L)
    for i in 1:n
        zi = Vector{Float64}(Zn[i, :])
        g = gradient(z -> _Eu_at_z(r, spec, z, ineff, scaling), zi)
        out[i, :] .= g
    end
    return out
end

function _Eu_at_z(r::SFResult, spec, z::AbstractVector, ineff::Symbol, scaling::Bool)
    if scaling
        # u_i = h(z_i)·u*, where δ lives on zvar without constant.
        δ = _get_scaling_delta(r, spec, length(z))
        h = exp(dot(z, δ))
        # E(u*) for the base distribution (homoscedastic).
        Eu_star = _Eu_star_base(r, spec, ineff)
        return h * Eu_star
    end

    if ineff === :HalfNormal
        σu² = _eval_lnσu²(r, spec, z)
        σu  = sqrt(σu²)
        return σu * sqrt(2 / π)
    elseif ineff === :TruncatedNormal
        μ   = _eval_μ(r, spec, z)
        σu² = _eval_lnσu²(r, spec, z)
        σu  = sqrt(σu²)
        a   = μ / σu
        Φa  = normcdf(a); φa = normpdf(a)
        return μ + σu * (φa / Φa)
    end
    return 0.0
end

# Base-distribution mean E(u*) for scaling-property models (homoscedastic).
function _Eu_star_base(r::SFResult, spec, ineff::Symbol)
    if ineff === :HalfNormal
        σu² = _scalar_from(r, spec, :sigma_u)
        return sqrt(σu²) * sqrt(2 / π)
    elseif ineff === :TruncatedNormal
        μ   = _scalar_from(r, spec, :mu)
        σu² = _scalar_from(r, spec, :sigma_u)
        σu  = sqrt(σu²)
        a   = μ / σu
        return μ + σu * normpdf(a) / normcdf(a)
    end
    return 0.0
end

# ==============================================================
# Parameter extraction from r.coeff via spec.mci_spec.idx
# ==============================================================

_as_range(x) = x isa AbstractRange ? x : (x:x)

function _get_sigma_v_sq(r::SFResult, spec)
    rng = _as_range(spec.mci_spec.idx.noise.ln_sigma_v_sq)
    return exp(r.coeff[rng][1])
end

# Returns (μvec, σu²vec) evaluated at supplied Zn (N × L matrix).
# For homoscedastic parameters, the vector is filled with the scalar value.
function _get_mu_sigma_u_sq(r::SFResult, spec, Zn::AbstractMatrix, ineff::Symbol)
    n = size(Zn, 1)
    scaling = spec.hetero === :scaling

    if scaling
        # u_i = h(z_i)·u*. For predict(:jlms/:bc), we need σ_u,i² and μ_i for
        # the *full* inefficiency. Under scaling with base N+(μ*, σu*²):
        #   u has mean h·μ* and standard deviation h·σu*, i.e. σu,i² = h² σu*²
        #   and (for TruncatedNormal) shifted μ_i = h·μ*.
        L = length(spec.mci_spec.idx.delta)
        size(Zn, 2) == L ||
            error("predict: scaling zvar has $(size(Zn,2)) columns but model δ has $L.")
        δ  = r.coeff[spec.mci_spec.idx.delta]
        # Clamp hscale to match MLE SFindex.jl:107.
        hi = clamp.(exp.(Zn * δ), 1e-12, 1e12)
        σu²_star = _scalar_from(r, spec, :sigma_u)  # exp(raw), outer clamp applied later
        if ineff === :HalfNormal
            μvec = zeros(n)
            σu²  = σu²_star .* (hi .^ 2)
        else # TruncatedNormal
            μ_star = _scalar_from(r, spec, :mu)
            μvec = μ_star .* hi                    # matches SFindex.jl:108 μ = z_pre * hscale
            σu²  = σu²_star .* (hi .^ 2)           # outer clamp in _predict_eff caps at 1e12
        end
        return μvec, σu²
    end

    # Non-scaling heteroscedastic or homoscedastic.
    hetero = spec.hetero isa AbstractVector ? spec.hetero : Symbol[]
    σfield = _variance_field(ineff, spec.mci_spec.idx.ineff)
    if ineff === :HalfNormal
        μvec = zeros(n)
        σu²  = _evaluate_param(r, spec, Zn, σfield, :sigma_sq in hetero, exponentiate = true)
    else # TruncatedNormal
        μvec = _evaluate_param(r, spec, Zn, :mu,    :mu       in hetero, exponentiate = false)
        σu²  = _evaluate_param(r, spec, Zn, σfield, :sigma_sq in hetero, exponentiate = true)
    end
    return μvec, σu²
end

# `idx.ineff` uses `:sigma_sq` for HalfNormal and `:sigma_u` for TruncatedNormal
# (the package's internal naming). Probe the NamedTuple to pick the right key.
function _variance_field(ineff::Symbol, ineff_idx)
    props = propertynames(ineff_idx)
    :sigma_u  in props && return :sigma_u
    :sigma_sq in props && return :sigma_sq
    error("predict: could not locate the variance-parameter field on spec.idx.ineff " *
          "(have keys $(props)).")
end

# Evaluate a parameter equation: heteroscedastic => Z·γ, homoscedastic => constant.
function _evaluate_param(r::SFResult, spec, Zn::AbstractMatrix, group::Symbol,
                         is_hetero::Bool; exponentiate::Bool)
    n = size(Zn, 1)
    γ = r.coeff[getproperty(spec.mci_spec.idx.ineff, group)]
    if is_hetero
        length(γ) == size(Zn, 2) ||
            error("predict: hetero `$group` expects $(length(γ)) z-columns, got $(size(Zn,2)).")
        lin = Zn * γ
    else
        # Homoscedastic: γ is scalar; same value for every obs.
        v = length(γ) == 1 ? γ[1] : γ[1]
        lin = fill(v, n)
    end
    return exponentiate ? exp.(lin) : lin
end

# Extract a scalar (homoscedastic) parameter value on its natural scale.
# Variance fields (:sigma_u / :sigma_sq) are stored on the log scale.
function _scalar_from(r::SFResult, spec, group::Symbol)
    if group === :sigma_u || group === :sigma_sq
        σfield = _variance_field(spec.ineff, spec.mci_spec.idx.ineff)
        rng = getproperty(spec.mci_spec.idx.ineff, σfield)
        raw = r.coeff[rng][1]
        return exp(raw)
    end
    rng = getproperty(spec.mci_spec.idx.ineff, group)
    return r.coeff[rng][1]
end

function _get_scaling_delta(r::SFResult, spec, L::Int)
    δ = r.coeff[spec.mci_spec.idx.delta]
    length(δ) == L ||
        error("predict: scaling δ has length $(length(δ)) but z has $L columns.")
    return δ
end

# Single-observation evaluators used by ForwardDiff in _predict_marginal.
function _eval_μ(r::SFResult, spec, z::AbstractVector)
    hetero = spec.hetero isa AbstractVector ? spec.hetero : Symbol[]
    γ = r.coeff[spec.mci_spec.idx.ineff.mu]
    if :mu in hetero
        length(γ) == length(z) || error("predict: μ hetero length mismatch.")
        return dot(z, γ)
    else
        return (length(γ) == 1 ? γ[1] : γ[1])
    end
end

function _eval_lnσu²(r::SFResult, spec, z::AbstractVector)
    hetero = spec.hetero isa AbstractVector ? spec.hetero : Symbol[]
    σfield = _variance_field(spec.ineff, spec.mci_spec.idx.ineff)
    γ = r.coeff[getproperty(spec.mci_spec.idx.ineff, σfield)]
    if :sigma_sq in hetero
        length(γ) == length(z) || error("predict: σ_u² hetero length mismatch.")
        return exp(dot(z, γ))
    else
        return exp(length(γ) == 1 ? γ[1] : γ[1])
    end
end

using LinearAlgebra: dot
