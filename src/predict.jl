# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later
#
# `predict(::SFResult; ...)` — application of a fitted stochastic frontier
# model to new observations. Implemented for cross-sectional models
# (MCI / MSLE / MLE) across all inefficiency distributions (HalfNormal,
# TruncatedNormal, Exponential, Lognormal, Rayleigh, Gamma, Lomax, Weibull,
# Erlang{2,3,4}) in
# homoscedastic, heteroscedastic, and scaling-property forms, with copulas; and
# for the three panel datatypes (panel_TFE via MCI/MSLE, panel_TFE_CSW and
# panel_TRE via MLE, which take `id=`). MCI / MSLE fits reuse the fit-time
# simulation kernels for EVERY inefficiency distribution and read the draw block
# stored on the result (`result.model`), in groups of the fit's `chunks`, so predict on
# the estimation data reproduces the fitted jlms/bc: bit for bit for a Float64 fit run
# on the CPU, to rounding for a Float32 or a GPU fit. The MLE backend uses a closed
# form, and for panel_TRE one-dimensional quadrature. MLE cross-sectional predict
# covers Normal + {HalfNormal, TruncatedNormal, Exponential}. Unsupported combinations
# raise an informative error.
#
# Dispatch targets (see Table in §3.12):
#   :frontier   -> x'β̂                              (needs frontier)
#   :residuals  -> y - x'β̂                          (needs frontier, depvar)
#   :response   -> x'β̂ - sgn·û                      (needs frontier, depvar, zvar if in model)
#   :jlms       -> E(u|ε)                            (needs frontier, depvar, zvar if in model)
#   :bc         -> E(exp(-u)|ε)                      (needs frontier, depvar, zvar if in model)
#   :marginal   -> ∂E(u)/∂z                          (needs zvar)
#   :all        -> NamedTuple of the targets above; marginal only for a cross-sectional
#                  model with determinants (hetero or :scaling), never for a panel

using StatsFuns: normpdf, normcdf, normccdf, normlogpdf, normlogcdf
using ForwardDiff: gradient
# The stable truncated-normal kernels (src/sf_numerics.jl).
using .SFNumerics: sf_tn_mean, sf_tn_logmgf
# The warning for Lomax marginal effects where the mean does not exist (src/sf_accuracy.jl).
using .SFAccuracy: sf_warn_lomax_mean
using .SFDraws: DrawRows, host_rows
# The simulation kernels read a DrawRows source through these functions, one group of rows at a time.
MCI_Backend._slice_draws_chunk(d::DrawRows, a, b)  = host_rows(d, a, b)
MSLE_Backend._slice_draws_chunk(d::DrawRows, a, b) = host_rows(d, a, b)
# Int bounds, as in the generic method: untyped bounds would make the call ambiguous.
Panel_Backend._panel_draw_rows(d::DrawRows, a::Int, b::Int) = host_rows(d, a, b)

"""
    predict(r::SFResult;
            frontier = nothing,
            zvar     = nothing,
            depvar   = nothing,
            id       = nothing,
            what     = :frontier,
            draws    = nothing)

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

`draws` (optional) replaces the draws of the simulation-based targets `:jlms`, `:bc`,
`:response` and `:all` of MCI, MSLE and panel_TFE fits. By default predict uses the draws
stored on the result (`result.model`), the block the fit used, in the fit's element type,
also for a fit that was given its own `draws`; it processes the observations in groups of
the fit's `chunks`, and for a Float32 or GPU fit converts each group's draws to Float64 on
the host. Predict on the estimation data therefore reproduces the fitted jlms/bc bit for
bit for a Float64 fit run on the CPU (a Float32 fit only to single-precision rounding, a
GPU fit only to double-precision rounding). Rows are matched to draws by position (row i
uses draw row i, recycled by mod1 beyond the estimation N); panel firms of the estimation
sample are matched by `id`, and the k-th new firm, in order of first appearance, uses the
draws of the k-th estimation firm, cycling through the estimation firms when there are
more new firms. Shape of `draws`: either `1 × n_draws` (a length-`n_draws` vector, or a
`n_draws × 1` column) for draws shared by every observation, or `N_train × n_draws` for
one row of draws per observation — the same object the fit was given. The element type
must be an `AbstractFloat`, as at `sfmodel_method`. A GPU fit's own draw object is
accepted here: predict copies it to the host and runs its CPU path on it.

`predict` returns point predictions only, so it takes no `level`: an explicit `level=`
is refused rather than ignored. Use `confint(result; level = ...)` or `sfmodel_CI` for
intervals on the estimated coefficients.

Supported models: all inefficiency distributions (`:HalfNormal`,
`:TruncatedNormal`, `:Exponential`, `:Lognormal`, `:Rayleigh`, `:Gamma`,
`:Lomax`, `:Weibull`, `:Erlang2`/`:Erlang3`/`:Erlang4`) in homoscedastic, heteroscedastic, and scaling-property
forms, with copulas — MCI / MSLE fits reuse the fit-time simulation kernels for
every distribution, while the MLE backend computes `jlms` and `bc` without simulation: in
closed form, and for `:panel_TRE` by one-dimensional quadrature. The three
panel datatypes are supported too:
`:panel_TFE` (via MCI/MSLE), `:panel_TFE_CSW` and `:panel_TRE` (via MLE); panel
targets require the `id=` keyword. MLE cross-sectional predict covers Normal +
`:HalfNormal`/`:TruncatedNormal`/`:Exponential`. Unsupported combinations raise
an informative error.

For a Lomax fit, `what = :marginal` returns NaN in the rows where α ≤ 1, where E(u) does
not exist, and prints one warning.
"""
function StatsAPI.predict(r::SFResult;
        frontier = nothing,
        zvar     = nothing,
        depvar   = nothing,
        id       = nothing,
        what::Symbol = :frontier,
        draws    = nothing,
        level    = nothing)

    # predict returns point predictions only; there is no interval target for a level to
    # apply to. The keyword `level` is kept solely so an explicit value is refused by name
    # rather than by a bare "unsupported keyword" MethodError (same treatment as the removed
    # `T_periods`).
    level === nothing ||
        error("predict: `level` is not accepted. predict returns point predictions, not " *
              "intervals, so there is no coverage level to set. Remove it. For a confidence " *
              "interval on the *estimated coefficients*, use `sfmodel_CI(...)` or " *
              "`confint(result; level = ...)`.")

    _predict_check_panel(r)
    spec = _predict_get_unified_spec(r)
    spec === nothing &&
        error("predict: no model specification is attached to this result.")

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
    error("predict: datatype `:$dt` is not supported. Supported: " *
          "`:cross_sectional`, `:panel_TFE`, `:panel_TFE_CSW`, `:panel_TRE`.")
end

# `sfmodel_spec` normalizes one-column inputs before storing them: a length-N
# `frontier`/`zvar` vector becomes an N×1 matrix, and an N×1 `depvar` matrix becomes a
# length-N vector (`_to_matrix` / `_to_vector` in sf_MCI_v21.jl). These two helpers apply the
# spec's normalization, rule for rule, so the two boundaries accept exactly the same objects:
# the very arrays a fit accepted are accepted when handed back, and predicting on the
# estimation frame reproduces the fitted jlms/bc.

function _require_matrix(X, name::String)
    X === nothing && error("predict: keyword `$name` is required for this target.")
    X isa AbstractMatrix && return X
    X isa AbstractVector{<:AbstractVector} && return reduce(hcat, X)
    X isa AbstractVector && return reshape(X, :, 1)   # N-vector is the N×1 the spec stores
    error("predict: `$name` must be a matrix, a vector (read as a single column), or a " *
          "vector of column vectors; got $(typeof(X)).")
end

function _require_vector(v, name::String)
    v === nothing && error("predict: keyword `$name` is required for this target.")
    if v isa AbstractVector
        # `[yvar]` — the one-element vector-of-vectors the spec unwraps. Same explicit count
        # check the spec normalizers use, so a list of several columns is named here too
        # instead of raising Base's bare `only` error.
        if v isa AbstractVector{<:AbstractVector}
            length(v) == 1 ||
                error("predict: `$name` must be a single vector, not a list of $(length(v)).")
            return only(v)
        end
        return v
    end
    if v isa AbstractMatrix
        size(v, 2) == 1 && return vec(v)              # N×1 is the N-vector the spec stores
        error("predict: `$name` must be a vector or a single-column matrix, got size $(size(v)).")
    end
    error("predict: `$name` must be a vector; got $(typeof(v)).")
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

# `name` is the keyword being checked: the helper is also called on a zvar column, where the old
# hard-coded "depvar" sent users to inspect an argument that was fine.
_check_len(y, X, name::AbstractString="depvar") =
    length(y) == size(X, 1) ||
    error("predict: length($name)=$(length(y)) does not match size(frontier,1)=$(size(X,1)).")

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
    _hetero_needs_zvar(spec) && _check_len(view(Zn, :, 1), Xn, "zvar")

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

    # Log-space JLMS/BC to match MLE SFindex.jl (jlmsbc for Trun, Half and Trun_Scale).
    # This parallels the MLE fit-time routine bit-for-bit. Only MLE fits reach it
    # (_dispatch_jlms_bc); MCI/MSLE fits simulate jlms/bc for every distribution,
    # HN and TN included, at the fit and in predict.
    # Below a = -40 the same stable forms as SFindex.jl, σs·m(a) and
    # exp(h(a - σs) - h(a)) (SFNumerics.sf_tn_mean, sf_tn_logmgf); above, the
    # expressions below, bit for bit.
    if !bc
        return ifelse.(a .< -40, σs .* sf_tn_mean.(a),
                       σs .* exp.(normlogpdf.(a) .- normlogcdf.(a)) .+ μs)
    else
        expo = -μs .+ 0.5 .* σs .^ 2 .+ normlogcdf.(a .- σs) .- normlogcdf.(a)
        return ifelse.(a .< -40, exp.(sf_tn_logmgf.(a, σs)), exp.(clamp.(expo, -500.0, 500.0)))
    end
end

# ==============================================================
# Normal + Exponential closed form for MLE fits
# ==============================================================
#
# Ported from src/sf_MLE/SFindex.jl:70-92 (function
# `jlmsbc(::Type{Expo}, ...)`). Uses the MLE parameter scale:
# `r.coeff[idx.ineff.sigma_sq[1]]` is ln(σᵤ²), so the scale σᵤ = sqrt(exp(w_pre)).
# This helper is therefore valid ONLY for r.backend === :MLE fits;
# the dispatcher guards it. Clamp constants match _MLE_CLAMP in
# src/sf_MLE/SFloglikefun.jl:18-22 (exp_lo=1e-12, exp_hi=1e12).

function _predict_eff_mle_exp(r::SFResult, spec, Xn_in, Zn_in, yn_in; bc::Bool)
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

    sigma_sq_rng = _as_range(mci.idx.ineff.sigma_sq)
    δ_σ   = r.coeff[sigma_sq_rng]
    v_pre = r.coeff[_as_range(mci.idx.noise.ln_sigma_v_sq)][1]

    # w_pre = ln(σᵤ²) per observation. Homoscedastic σᵤ² (`:sigma_u_sq` not in `hetero`) ⇒
    # one coefficient, the constant, for every row. Hetero-σᵤ² ⇒ w_pre = W·δ where W is the
    # full fit-time zvar (with or without a constant column), mirroring SFindex.jl
    # jlmsbc(Expo) `w_pre = W*coef`. σᵤ = sqrt(exp(w_pre)).
    # The branch is chosen by the model, as `_get_mu_sigma_u_sq` does for the half normal and
    # truncated normal, not by the number of ln σᵤ² coefficients: that count would take a
    # heteroscedastic σᵤ² with a single zvar column for homoscedastic, ignore z (σᵤ² = exp(δ)
    # in every row, wrong wherever that column is not 1) and accept :jlms without zvar.
    σᵤ²_hetero = spec.hetero isa AbstractVector && :sigma_u_sq in spec.hetero
    if !σᵤ²_hetero
        w_pre = fill(δ_σ[1], length(yn))
    else
        Zn = _require_matrix(Zn_in, "zvar")
        _check_len(view(Zn, :, 1), Xn, "zvar")
        size(Zn, 2) == length(sigma_sq_rng) ||
            error("predict: hetero-σᵤ² `zvar` has $(size(Zn,2)) columns but model expects " *
                  "$(length(sigma_sq_rng)). Pass the same zvar used at fit time (same columns, " *
                  "same order; a constant column only if the fit's zvar had one).")
        w_pre = Zn * δ_σ
    end

    σᵤ  = sqrt.(clamp.(exp.(w_pre), 1e-12, 1e12))
    σᵥ² = clamp(exp(v_pre), 1e-12, 1e12)
    σᵥ  = sqrt(σᵥ²)

    μs = -ε .- σᵥ² ./ σᵤ
    rv = μs ./ σᵥ

    # Below rv = -40 the same stable forms as SFindex.jl, σᵥ·m(rv) and
    # exp(h(rv - σᵥ) - h(rv)); above, the expressions below, bit for bit.
    if !bc
        return ifelse.(rv .< -40, σᵥ .* sf_tn_mean.(rv),
                       σᵥ .* exp.(normlogpdf.(rv) .- normlogcdf.(rv)) .+ μs)
    else
        expo = -μs .+ 0.5 * σᵥ² .+ normlogcdf.(rv .- σᵥ) .- normlogcdf.(rv)
        return ifelse.(rv .< -40, exp.(sf_tn_logmgf.(rv, σᵥ)), exp.(clamp.(expo, -500.0, 500.0)))
    end
end

# ==============================================================
# Simulation-based inefficiency / efficiency indices (MCI / MSLE)
# ==============================================================
#
# Reuses the fit-time simulation routine that populates r.jlms /
# r.bc at fit time. Covers every cross-sectional noise × ineff
# combination of an MCI/MSLE fit, HalfNormal and TruncatedNormal
# included, and scaling-property and copula models.
#
# Draws: predict reads the block the fit used, stored on the result
# (`_fit_draw_block`), and maps predict rows to its rows by position
# (`_predict_rows`):
#   N_new == N_train → the block as it is
#   N_new  < N_train → rows 1:N_new
#   N_new  > N_train → row i ← block row mod1(i, N_train)
# A host Float64 block is read through a view (`_align_draws`); a Float32
# or device block through a `DrawRows` source, whose rows the kernels
# widen to Float64 on the host one group of `chunks` at a time
# (`_predict_draw_source`). Consequence: for any predict observation
# whose index maps to fit row k via mod1, the computed jlms/bc match
# r.jlms[k] / r.bc[k] when ε also matches. For a Float64 fit run on the
# CPU they match bit for bit when the observation is in the same group
# as at fit time (always on the estimation frame), to rounding
# otherwise; for a Float32 or a GPU fit, to rounding. The same row map
# applies when the user passes `draws=` to predict().
#
# Scaling: u_i = h(z_i) · u*_i where h(z) = exp(z · δ). At predict
# time the user passes `zvar=Z_s` (no-constant, L_scaling columns);
# we compute `h = exp.(Z_s · δ)` and pass as `scaling_h` to the
# simulation routine. The base ineff distribution is homoscedastic,
# so its zvar is the fit-time convention `ones(N_new, 1)`.
#
# Copula: `copula_vals` is a NamedTuple rebuilt from fitted
# `theta_rho` via `get_copula_vals(model.copula, p, idx, c)`. No
# separate draws are needed — the same draws serve both.

# True only for the analytic MLE backend: Normal + HN/TN JLMS/BC are computed in
# closed form at fit time, so predict mirrors that same closed form (no Halton draws).
_eff_use_closed_form(spec) =
    spec.noise === :Normal && spec.ineff in (:HalfNormal, :TruncatedNormal) &&
    spec.copula === :None

# Route :jlms / :bc by backend so that predict on the estimation data reproduces the
# fitted r.jlms / r.bc *by the same construction the fit used*:
#   * MCI / MSLE (and panel_TFE) fits populate r.jlms / r.bc by simulating E(u|ε) over
#     their draws — so predict reuses that simulation kernel for EVERY inefficiency
#     distribution, including HalfNormal / TruncatedNormal. It reads the draws stored on
#     the result (or takes the draws passed with `draws=`) and returns the fitted values
#     on the training data.
#   * MLE fits compute JLMS/BC analytically at fit time, so predict mirrors that closed
#     form exactly (Normal + HN/TN, plus the Normal + Exponential closed form). No draws.
function _dispatch_jlms_bc(r::SFResult, spec, Xn, Zn, yn; bc::Bool, user_draws, id = nothing)
    user_draws = _draws_to_host(user_draws)
    # Shape and element type are checked HERE, at the single funnel for the simulation targets,
    # and not inside the per-backend resolvers: the panel route narrows with `Tfit_p.(user_draws)`
    # before its own guards run, so a check placed after that narrowing sees the fit's element
    # type whatever the caller passed — on a Float32 panel fit `draws = [0]` became `Float32[0.0]`
    # and sailed through. Each backend still re-checks the shape it needs.
    if user_draws !== nothing
        user_draws isa AbstractVector || user_draws isa AbstractMatrix ||
            error("predict: `draws` must be a vector or a matrix.")
        _check_draws_eltype(user_draws)
    end
    if hasproperty(spec, :datatype) && spec.datatype === :panel_TFE
        if r.backend === :Panel
            return _predict_eff_panel_tfe(r, spec, Xn, Zn, yn, id;
                                          bc = bc, user_draws = user_draws)
        end
        error("predict: panel_TFE fits are supported via the Panel backend " *
              "(`r.backend === :Panel`). Got `r.backend === :$(r.backend)`; " *
              "MLE-based panel predict is not supported.")
    end
    if hasproperty(spec, :datatype) &&
       (spec.datatype === :panel_TFE_CSW || spec.datatype === :panel_TRE)
        r.backend === :MLE ||
            error("predict: `:$(spec.datatype)` fits are MLE-only " *
                  "(got backend `:$(r.backend)`).")
        return _predict_eff_mle_panel(r, spec, Xn, Zn, yn, id; bc = bc)
    end
    # Cross-sectional MCI / MSLE: always reuse the fit-time simulation kernel (every
    # ineff, incl. HalfNormal / TruncatedNormal). `draws=` therefore applies to HN/TN too.
    if r.backend === :MCI
        return _predict_eff_mci(r, spec, Xn, Zn, yn; bc = bc, user_draws = user_draws)
    end
    if r.backend === :MSLE
        return _predict_eff_msle(r, spec, Xn, Zn, yn; bc = bc, user_draws = user_draws)
    end
    # Cross-sectional MLE: mirror the fit-time analytic closed form.
    if r.backend === :MLE
        if _eff_use_closed_form(spec)
            return _predict_eff(r, spec, Xn, Zn, yn; bc = bc)
        end
        if spec.noise === :Normal && spec.ineff === :Exponential
            return _predict_eff_mle_exp(r, spec, Xn, Zn, yn; bc = bc)
        end
        error("predict: MLE fit with noise=$(spec.noise), ineff=$(spec.ineff) is " *
              "not supported. MLE supports Normal + {HalfNormal, TruncatedNormal, " *
              "Exponential} only.")
    end
    error("predict: unsupported backend $(r.backend).")
end

# A GPU fit accepts a `CuArray` as `draws=`, and predict accepts the same object. Every predict
# path runs on the host: it validates the draw count, aligns rows to observations, and calls the
# CPU kernel. Left on the device, that alignment would index a GPU array element by element,
# which under the usual `CUDA.allowscalar(false)` throws "Scalar indexing is disallowed".
# Copying draws passed with `draws=` to host memory here, at the single funnel for the simulation
# targets, avoids that; a host-backed array is returned as it is, so no CPU path gains a copy.
# (The draws stored on a GPU fit's result are not copied here: predict reads them group by
# group through `DrawRows`.)
#
# The device test mirrors the backends' own (`_prepare_draws` in sf_MCI_v21.jl): the array's
# root parent is an ordinary `Array` iff it lives in host memory. This keeps predict free of
# a CUDA dependency — SFrontiers only ever reaches CUDA through `Main` — and works for any
# device array type, not just `CuArray`.
_draws_to_host(::Nothing) = nothing
_draws_to_host(x) = x
function _draws_to_host(A::AbstractArray)
    _host_backed(A) && return A
    return Array(A)
end

function _host_backed(A::AbstractArray)
    p = A
    while true
        q = applicable(parent, p) ? parent(p) : p
        q === p && break
        p = q
    end
    return p isa Array
end

# The fit's own draw block, stored on the result: N_train × D (1 × D for shared draws) for
# cross-sectional fits, N_firms × D or a D-vector for panel fits; fit's element type and device.
function _fit_draw_block(r::SFResult)
    m = hasproperty(r, :model) ? r.model : nothing
    blk = m === nothing ? nothing :
          r.backend === :Panel ? (hasproperty(m, :draws) ? m.draws : nothing) :
                                 (hasproperty(m, :draws_2D) ? m.draws_2D : nothing)
    blk === nothing &&
        error("predict: this result does not carry the draws of its fit (`result.model`); " *
              "pass them with `draws=`.")
    return blk
end

# Draws are paired with predict rows BY POSITION: predict row i receives the draws of training row i
# (recycled by mod1 beyond N_train). That is exact when the predict frame is the estimation frame in
# its original order. For any other frame — a subset, a reordering, or new data — a row receives draws
# that belong to a different training observation, so its jlms/bc differ from the fitted value by
# simulation error. The rule is documented (the docstring of `predict` and the manual); predict
# prints no message about it.
# The training rows that predict rows 1:N_new use: by position, recycled beyond N_train.
function _predict_rows(N_train::Int, N_new::Int)
    N_new == N_train && return 1:N_train
    return N_new < N_train ? (1:N_new) : mod1.(1:N_new, N_train)
end

function _align_draws(draws_train::AbstractMatrix, N_new::Int)
    N_new == size(draws_train, 1) && return draws_train
    return view(draws_train, _predict_rows(size(draws_train, 1), N_new), :)
end

# The estimation row each predict firm uses: estimation firms by id; the k-th new firm, in order
# of first appearance, uses estimation row mod1(k, N_fit).
function _predict_panel_rows(fit_ids::AbstractVector, predict_ids::AbstractVector)
    fit_id_to_row = Dict{eltype(fit_ids), Int}()
    for (i, id) in enumerate(fit_ids)
        fit_id_to_row[id] = i
    end
    N_fit = length(fit_ids)
    rows = Vector{Int}(undef, length(predict_ids))
    k = 0
    for (i, id) in enumerate(predict_ids)
        j = get(fit_id_to_row, id, 0)
        if j == 0
            k += 1
            j = mod1(k, N_fit)
        end
        rows[i] = j
    end
    return rows
end

_build_predict_panel_draws(draws_fit::AbstractMatrix, fit_ids::AbstractVector,
                           predict_ids::AbstractVector) =
    view(draws_fit, _predict_panel_rows(fit_ids, predict_ids), :)

# What the kernels read for a cross-sectional fit's stored block.
function _predict_draw_source(block::AbstractMatrix, N_new::Int, chunks::Int)
    size(block, 1) == 1 && return block isa Matrix{Float64} ? block : Float64.(Array(block))   # shared
    block isa Matrix{Float64} && return _align_draws(block, N_new)          # the default: block or view
    src = DrawRows(block, collect(_predict_rows(size(block, 1), N_new)))
    return chunks >= 2 ? src : host_rows(src, 1, N_new)    # unchunked kernels broadcast over the whole matrix
end

# `sfmodel_method(draws=)` is typed `AbstractVecOrMat{<:AbstractFloat}`, so an integer, rational
# or `Vector{Real}` array is refused there by the keyword's own type — the house convention for a
# typed keyword, and a restriction kept on purpose. `predict`'s `draws` cannot be typed that way
# (it also takes `nothing`), so this check applies the same contract at predict's end: predict's
# `draws` is documented as "the same object the fit was given", and no fit can ever have been
# given an integer array, so accepting one here could not serve a real roundtrip. The message
# names the fit-side rule rather than reproducing a bare `TypeError`, because predict has no
# typed signature to raise one from. It is called once, at `_dispatch_jlms_bc`, ahead of every
# per-backend narrowing.
function _check_draws_eltype(draws)
    eltype(draws) <: AbstractFloat && return nothing
    error("predict: `draws` must have a floating-point element type " *
          "(`Float64`, `Float32`, ...); got $(typeof(draws)) with eltype $(eltype(draws)). " *
          "`sfmodel_method(draws = ...)` accepts `AbstractVecOrMat{<:AbstractFloat}` only, so " *
          "no fit was ever given draws of this type and predict cannot be reproducing one. " *
          "Convert them first, e.g. `float.(draws)`.")
end

# Validate a user-supplied `draws` object passed to predict(), by the same rule the fit uses
# (`_shape_user_draws` in the backends): a vector, a 1×D row or a D×1 column is one sequence of
# draws shared by every observation; an N_train×D matrix is one row of draws per training
# observation; any other row count cannot be split at the observation level. `multiRand` is not
# consulted — it is false for every user-draws fit — so predict reads the object exactly as the
# fit did. Returns a 1×D or N_train×D matrix.
function _validate_user_draws(draws, n_draws::Int, N_train::Int)
    draws isa AbstractVector || draws isa AbstractMatrix ||
        error("predict: `draws` must be a vector or a matrix.")
    # The count comparison below would report "0 draws per observation; the fit used N",
    # which is true but describes a downstream mismatch rather than the cause. The fit
    # boundary and panel prediction both say "`draws` is empty" for this input; say the
    # same thing here, and before the count is looked at.
    isempty(draws) && error("predict: `draws` is empty.")
    M = if draws isa AbstractVector
        reshape(draws, 1, length(draws))
    elseif size(draws, 2) == 1
        reshape(draws, 1, size(draws, 1))
    elseif size(draws, 1) == 1 || size(draws, 1) == N_train
        draws
    else
        error("predict: `draws` has size $(size(draws)); the number of rows must be 1 (draws " *
              "shared by every observation) or $N_train (the training N, one row per observation).")
    end
    size(M, 2) == n_draws ||
        error("predict: `draws` has $(size(M,2)) draws per observation; the fit used $n_draws.")
    MCI_Backend._validate_user_draws(M)   # [0, 1) contract, as at fit time
    return M
end

# Rebuild a backend constants NamedTuple that was built in the FIT's element type into
# predict's working type `T`. Several of these constants are type-dependent numerical
# guards, not scale factors — notably `hi_erf = one(T) - T(32)*eps(T)`, which is
# 0.99999618530273438 in Float32 but 0.99999999999999289 in Float64. Rebuilt at Float64 for
# a Float32 fit they leave a quantile argument in (1 - 3.8e-6, 1) unclamped that the fit had
# clamped, so erfinv is evaluated at a different point and predict returns a different
# integral, not a differently rounded one (measured jlms 0.051 cached vs 0.297 predicted on
# a Float32 TruncatedNormal fit with ordinary auto-generated Halton draws, where
# r + (r-1)*erf(mu/(sigma*sqrt2)) -> 1 for every draw). Widening the fit-type values keeps
# predict's arithmetic in double precision, as the manual promises, while reproducing the
# fit's clamps. For a Float64 fit this is the identity, so those fits stay bit-for-bit.
_fit_type_constants(c::NamedTuple, ::Type{T}) where {T<:AbstractFloat} =
    map(x -> x isa AbstractFloat ? T(x) : x, c)

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
        _check_len(view(Zn_scaling, :, 1), Xn, "zvar")
        size(Zn_scaling, 2) == mci.L_scaling ||
            error("predict: scaling `zvar` has $(size(Zn_scaling,2)) columns but " *
                  "model δ expects $(mci.L_scaling). Pass the same no-constant Z " *
                  "used at fit time.")
        Zn_scaling_T = Zn_scaling isa AbstractMatrix{T} ? Zn_scaling :
                       convert(Matrix{T}, Zn_scaling)
        Zn_T = ones(T, N_new, 1)
    elseif _hetero_needs_zvar(spec)
        Zn_raw = _require_matrix(Zn_in, "zvar")
        _check_len(view(Zn_raw, :, 1), Xn, "zvar")
        size(Zn_raw, 2) == mci.L ||
            error("predict: hetero `zvar` has $(size(Zn_raw,2)) columns but model was " *
                  "fitted with $(mci.L). Pass the same zvar (same columns, same order) " *
                  "used at fit time — get_ineff_vals reads only the leading columns, so " *
                  "an extra/missing column silently shifts the hetero parameters.")
        Zn_T = Zn_raw isa AbstractMatrix{T} ? Zn_raw : convert(Matrix{T}, Zn_raw)
        Zn_scaling_T = nothing
    else
        Zn_T = ones(T, N_new, 1)
        Zn_scaling_T = nothing
    end

    hetero_vec = spec.hetero isa Vector{Symbol} ? spec.hetero : Symbol[]

    n_draws = Int(r.n_draws)
    N_train = mci.N
    chunks  = Int(r.chunks)

    if user_draws !== nothing
        M = _validate_user_draws(user_draws, n_draws, N_train)
        # The fit narrowed user draws to its own element type (`_prepare_draws`:
        # `T.(...)`, sf_MCI_v21.jl). Reproduce that narrowing before widening to predict's
        # working type, or the quantile is evaluated at a different draw than the fit used.
        Tfit_m = eltype(mci.depvar)
        M_T = Tfit_m === T ? (M isa AbstractMatrix{T} ? M : convert(Matrix{T}, M)) :
                             convert(Matrix{T}, Tfit_m.(M))
        MCI_Backend._validate_user_draws(M_T)   # a draw below 1 can round to 1 when narrowed to T
        # shared draws (1 row) pass through; per-observation draws are matched to the predict rows
        draws_1D = size(M_T, 1) == 1 ? M_T : _align_draws(M_T, N_new)
    else
        draws_1D = _predict_draw_source(_fit_draw_block(r), N_new, chunks)
    end

    p = r.coeff isa AbstractVector{T} ? r.coeff : convert(Vector{T}, r.coeff)

    # Reuse the training-time model / idx (they encode the parameter layout
    # that matches r.coeff). Rebuilding would work too but risks drift.
    model = mci.model
    idx   = mci.idx
    K     = size(Xn_T, 2)

    c          = _fit_type_constants(MCI_Backend.make_constants(model, eltype(mci.depvar)), T)
    noise_vals = MCI_Backend.get_noise_vals(model.noise, p, idx, c)
    ineff_vals = MCI_Backend.get_ineff_vals(model.ineff, p, idx, Zn_T, c)

    # Residuals ε = y - Xβ, matching the formula in MCI_nll / jlms_bc_indices.
    P_ = eltype(p)
    ε  = yn_T .- sum(P_(p[idx.beta[j]]) .* view(Xn_T, :, j) for j in 1:K)

    # Use the fit-time transformation rule resolved at fit and stored on the
    # internal assembled spec (`r.model`, a sfmodel_MCI_spec). It reflects the
    # user's sfmodel_method(transformation=…) override, or the distribution
    # default when none was given — reproducing bit-identical predict-on-training
    # for ineff distributions whose T-approach rule differs from the direct
    # quantile (e.g. Lognormal, Lomax, HalfNormal, TruncatedNormal under
    # :logistic_1_rule). Fall back to the default only when unavailable.
    trans_rule = (hasproperty(r, :model) && hasproperty(r.model, :transformation)) ?
        r.model.transformation : MCI_Backend.default_transformation_rule(spec.ineff)
    trans_func, jacob_func = MCI_Backend.resolve_transformation(trans_rule)

    sgn_int = Int(_frontier_sign(r))
    D       = size(draws_1D, 2)

    # Scaling: h_i = exp(Z_s · δ). Summed column by column, in the SAME ORDER as the fit
    # (sf_MCI_v21.jl, the `scaling_h` block of MCI_nll), not as a single matrix-vector product:
    # floating-point addition is not associative, and BLAS sums in a blocked, vectorised order,
    # so with four or more scaling regressors the two disagreed in the last bit and predict no
    # longer reproduced the fitted jlms/bc exactly (measured 8.9e-16 at L_scaling = 4 and
    # 2.2e-15 at 6; L_scaling ≤ 3 happened to agree). The MSLE path is unaffected because it
    # delegates h(z) to the fit's own spec form.
    scaling_h = mci.scaling ?
        exp.(sum(p[idx.delta[j]] .* (@view Zn_scaling_T[:, j]) for j in 1:length(idx.delta))) :
        nothing

    # Copula: rebuild NamedTuple from fitted theta_rho.
    copula_vals = spec.copula === :None ? NamedTuple() :
        MCI_Backend.get_copula_vals(model.copula, p, idx, c)

    out = MCI_Backend._jlms_bc_mci_T(
        ε, draws_1D, model, c, N_new, D,
        sgn_int, chunks, noise_vals, ineff_vals;
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
# `jlms_bc_indices(spec, p; chunks)` with the fit's `chunks`, which natively handles
# scaling and copula. We construct a predict-time internal
# `sfmodel_MSLE_spec` by copying the fitted model / idx / copula
# type / etc. from `spec.msle_spec` and substituting predict-time
# data + the draws stored on the result (or those passed with `draws=`).

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
        _check_len(view(Zn_scaling, :, 1), Xn, "zvar")
        size(Zn_scaling, 2) == msle.L_scaling ||
            error("predict: scaling `zvar` has $(size(Zn_scaling,2)) columns but " *
                  "model δ expects $(msle.L_scaling). Pass the same no-constant Z " *
                  "used at fit time.")
        Zn_scaling_T = Zn_scaling isa AbstractMatrix{T} ? Zn_scaling :
                       convert(Matrix{T}, Zn_scaling)
        Zn_T = ones(T, N_new, 1)
    elseif _hetero_needs_zvar(spec)
        Zn_raw = _require_matrix(Zn_in, "zvar")
        _check_len(view(Zn_raw, :, 1), Xn, "zvar")
        size(Zn_raw, 2) == msle.L ||
            error("predict: hetero `zvar` has $(size(Zn_raw,2)) columns but model was " *
                  "fitted with $(msle.L). Pass the same zvar (same columns, same order) " *
                  "used at fit time — get_ineff_vals reads only the leading columns, so " *
                  "an extra/missing column silently shifts the hetero parameters.")
        Zn_T = Zn_raw isa AbstractMatrix{T} ? Zn_raw : convert(Matrix{T}, Zn_raw)
        Zn_scaling_T = nothing
    else
        Zn_T = ones(T, N_new, 1)
        Zn_scaling_T = nothing
    end

    n_draws   = Int(r.n_draws)
    multiRand = r.multiRand
    N_train   = msle.N
    chunks    = Int(r.chunks)

    if user_draws !== nothing
        M = _validate_user_draws(user_draws, n_draws, N_train)
        # The fit narrowed user draws to its own element type (`_prepare_draws`:
        # `T.(...)`, sf_MSLE_v21.jl). Reproduce that narrowing before widening to predict's
        # working type, or the quantile is evaluated at a different draw than the fit used.
        Tfit = eltype(msle.depvar)
        M_T = Tfit === T ? (M isa AbstractMatrix{T} ? M : convert(Matrix{T}, M)) :
                           convert(Matrix{T}, Tfit.(M))
        MCI_Backend._validate_user_draws(M_T)   # a draw below 1 can round to 1 when narrowed to T
        # shared draws (1 row) pass through; per-observation draws are matched to the predict rows
        draws_2D = size(M_T, 1) == 1 ? M_T : _align_draws(M_T, N_new)
    else
        draws_2D = _predict_draw_source(_fit_draw_block(r), N_new, chunks)
    end

    # `draws` field on the internal spec is a flat 1D view of the matrix.
    draws_1D_flat = vec(draws_2D)

    p = r.coeff isa AbstractVector{T} ? r.coeff : convert(Vector{T}, r.coeff)

    model = msle.model
    idx   = msle.idx
    K     = size(Xn_T, 2)
    L     = size(Zn_T, 2)
    sgn   = Int(_frontier_sign(r))

    c = _fit_type_constants(MSLE_Backend.make_constants(model, eltype(msle.depvar)), T)

    # Hetero as Vector{Symbol} (internal spec expects concrete vector).
    hetero_vec_msle = msle.hetero isa Vector{Symbol} ? msle.hetero : Symbol[]

    # Build predict-time internal spec and delegate to the spec-form.
    internal_spec = MSLE_Backend.sfmodel_MSLE_spec{T}(
        yn_T, Xn_T, Zn_T,
        msle.noise, msle.ineff, msle.copula, hetero_vec_msle,
        draws_1D_flat, draws_2D, multiRand, c,
        msle.varnames, msle.eqnames, msle.eq_indices,
        N_new, K, L, model, idx, sgn, chunks,
        msle.scaling, Zn_scaling_T, msle.L_scaling,
    )

    out = MSLE_Backend.jlms_bc_indices(internal_spec, p; chunks = chunks)

    return bc ? collect(out.bc) : collect(out.jlms)
end

# ==============================================================
# Panel TFE (Wang and Ho 2010) — MCI / MSLE
# ==============================================================
#
# Mirrors `_predict_eff_msle`: read the fit's stored draws, build a
# predict-time `_PanelInternalSpec`, and delegate to `panel_jlms_bc`.
#
# Firms and draws: a predict firm that shares an id with an estimation firm
# uses that firm's row of the stored block; the k-th new firm, in order of
# first appearance, uses the row of estimation firm mod1(k, N_fit)
# (`_predict_panel_rows`). An estimation firm therefore reproduces its fitted
# jlms/bc whatever the order of the firms in the predict frame: for a Float64
# fit run on the CPU bit for bit when it is in the same group of `chunks` as
# at fit time, to rounding otherwise; for a Float32 or a GPU fit, to rounding.
#
# `id=` is required (each observation's firm identifier). Rows must be
# grouped contiguously by id (same convention the fit path enforces via
# `_compute_panel_structure`).
#
# Shared draws (a length-D vector): every firm uses the same draws, so the
# firm map does not apply.
#
# Transformation rule: `r.transformation` is the user-specified rule from
# fit (`nothing` = distribution default). Forwarded verbatim to the panel
# internal spec so `resolve_panel_transformation` reproduces fit-time
# behavior.
#
# User-supplied `draws=` at fit time: the fit keeps them as its block, so
# predict uses them when `draws=` is omitted; `draws=` replaces the stored
# draws for every fit.

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
    # Row count too: h(z) is built from zvar, so an oversized zvar silently produced jlms/bc of
    # zvar's length computed from the wrong rows, and an undersized one a raw BoundsError. The
    # cross-sectional paths (below) and the panel FIT (sf_panel_v20.jl) already check this.
    size(Zn, 1) == size(Xn, 1) ||
        error("predict: `zvar` has $(size(Zn,1)) rows but `frontier` has $(size(Xn,1)).")
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

    n_draws = Int(r.n_draws)
    N_fit   = pspec.N
    chunks  = Int(r.chunks)
    Tfit_p  = eltype(pspec.depvar)
    draws_fit = user_draws === nothing ? _fit_draw_block(r) :
        _resolve_panel_fit_draws(Tfit_p === T ? user_draws : Tfit_p.(user_draws), n_draws, N_fit, T)
    draws_predict = if ndims(draws_fit) == 1                               # shared draws: D numbers
        draws_fit isa Vector{T} ? draws_fit : T.(Array(draws_fit))
    elseif draws_fit isa Matrix{T}                                         # host Float64: a row view
        _build_predict_panel_draws(draws_fit, pspec.firm_ids, firm_ids_predict)
    else                                                                   # Float32 or GPU block
        src = DrawRows(draws_fit, _predict_panel_rows(pspec.firm_ids, firm_ids_predict))
        chunks >= 2 ? src : host_rows(src, 1, length(src.rows))
    end

    # Build predict-time internal spec mirroring `_assemble_panel_spec`:
    # demean y/X within firm, keep z raw (h(z) is demeaned inside the kernel).
    y_tilde = Panel_Backend.sf_panel_demean(yn_T, offsets_predict)
    x_tilde = Panel_Backend.sf_panel_demean(Xn_T, offsets_predict)
    Tm1     = T.(T_vec_predict .- 1)

    constants = _fit_type_constants(
        Panel_Backend.make_panel_constants(pspec.model, eltype(pspec.depvar), T_max_predict), T)
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
        chunks,                           # the fit's chunks: predict on the estimation frame uses the fit's groups
        constants,
        pspec.varnames, pspec.eqnames, pspec.eq_indices,
        method_sym, trans_sym, false,     # GPU=false for predict
    )

    p = r.coeff isa AbstractVector{T} ? r.coeff : convert(Vector{T}, r.coeff)
    out = Panel_Backend.panel_jlms_bc(internal_spec, p; chunks = chunks)
    return bc ? collect(out.bc) : collect(out.jlms)
end

# The draws passed with `draws=` for a panel fit, checked and shaped as the fit shapes its own:
# an `N_fit × D` matrix (one row per firm) or a length-`D` vector (shared by every firm).
function _resolve_panel_fit_draws(user_draws, n_draws::Int, N_fit::Int, ::Type{T}) where {T}
    user_draws isa AbstractVector || user_draws isa AbstractMatrix ||
        error("predict: `draws` must be a vector or a matrix.")
    Panel_Backend._validate_panel_user_draws(user_draws)   # [0, 1) contract, as at fit time
    # Same rule as the panel fit: a vector / 1×D / D×1 is one sequence shared by every firm,
    # an N_fit×D matrix is one row per firm. `multiRand` is false for every user-draws fit.
    shaped = Panel_Backend._shape_user_panel_draws(user_draws, N_fit)
    nd = ndims(shaped) == 1 ? length(shaped) : size(shaped, 2)
    nd == n_draws ||
        error("predict: `draws` has $nd draws per firm; the fit used $n_draws.")
    out = ndims(shaped) == 1 ?
        (shaped isa AbstractVector{T} ? copy(shaped) : convert(Vector{T}, shaped)) :
        (shaped isa AbstractMatrix{T} ? copy(shaped) : convert(Matrix{T}, shaped))
    Panel_Backend._validate_panel_user_draws(out)   # again after narrowing to T
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
    # predict(:marginal) is currently cross-sectional only. The helpers
    # _Eu_at_z / _scalar_from / _eval_μ / _eval_lnσu² / _get_scaling_delta
    # below all dereference `spec.mci_spec.idx.*`, which is `nothing` for
    # panel_TFE, panel_TFE_CSW, and panel_TRE specs. Mirrors the same
    # cross-sectional-only restriction in _dispatch_jlms_bc for MLE-based
    # panel jlms/bc.
    if hasproperty(spec, :datatype) && spec.datatype !== :cross_sectional
        error("predict(:marginal) supports cross-sectional fits only. " *
              "Got `spec.datatype === :$(spec.datatype)`. " *
              "Panel marginal effects via predict() are not supported.")
    end

    ineff = spec.ineff
    scaling = spec.hetero === :scaling

    # Distributions outside the Normal + HN/TN closed form reuse the fit-time
    # marginal E(u) kernels (`uncondU_*`, the very closures `get_marg`
    # differentiates; for an :MLE Exponential fit, the MLE's own `marg_expo`),
    # so predict(:marginal) stays consistent with the fitted object's own
    # `result.marginal`. HN/TN keep the ForwardDiff path below — its raw n×L
    # contract is pinned by runtests.jl.
    if ineff ∉ (:HalfNormal, :TruncatedNormal)
        return _predict_marginal_kernel(r, spec, Zn, ineff, scaling)
    end
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

# --------------------------------------------------------------
# Marginal effects for the inefficiency distributions other than HN/TN
# (Exponential, Lognormal, Rayleigh, Gamma, Erlang, Lomax, Weibull) and their
# scaling-property forms. Reuses the fit-time E(u) kernels that populate
# `result.marginal` (`MCI_Backend`/`MSLE_Backend` `uncondU_*` /
# `uncondU_scaling`), so predict(:marginal) is consistent with the fitted
# object's own marginal table. It also serves a cross-sectional :MLE Exponential
# fit, with that fit's own E(u) (`_marginal_Eu_mle_expo`).
# The per-observation ForwardDiff gradient of total
# E(u) w.r.t. the supplied z-row yields the same raw n×L contract as the HN/TN
# path in `_predict_marginal` (so `size(predict(:marginal)) == size(zvar)`); its
# non-constant columns equal `result.marginal` evaluated at the fit zvar.
#
# NB: the `hetero ? z : ones-basis` channel selector in
# `_marginal_Eu_closure` mirrors `get_marg`'s own `Zmarg = [one(T)]` construction.
# Because the E(u) math is delegated to `uncondU_*`, a change to how a homoscedastic
# parameter enters the marginal *inside* the kernels propagates here automatically;
# only this selector must be revisited alongside `_eval_μ`/`_eval_lnσu²`.
function _predict_marginal_kernel(r::SFResult, spec, Zn::AbstractMatrix,
                                  ineff::Symbol, scaling::Bool)
    T = Float64
    n = size(Zn, 1)
    L = size(Zn, 2)
    coef = r.coeff isa AbstractVector{T} ? r.coeff : convert(Vector{T}, r.coeff)

    if r.backend === :MCI
        sub = spec.mci_spec
        BE  = MCI_Backend
    elseif r.backend === :MSLE
        sub = spec.msle_spec
        BE  = MSLE_Backend
    elseif r.backend === :MLE && ineff === :Exponential && !scaling
        # A cross-sectional :MLE Exponential fit. Its zvar is checked against the
        # cross-sectional spec attached to the result (built from the fit's own zvar), and
        # its E(u) is the MLE fit's own function (`_marginal_Eu_mle_expo`, below).
        sub = spec.mci_spec
        BE  = MLE_Backend
    else
        error("predict(:marginal): the kernel path for `ineff=:$ineff` supports the " *
              "MCI and MSLE backends, and :MLE for `ineff=:Exponential`; got backend " *
              "`:$(r.backend)`.")
    end
    sub === nothing &&
        error("predict(:marginal): no $(r.backend) spec is attached for ineff=:$ineff.")

    idx        = sub.idx
    ineff_type = sub.model.ineff
    hetero     = sub.hetero isa AbstractVector ? sub.hetero : Symbol[]

    # Column-count validation, mirroring `_predict_eff_mci`.
    if scaling
        L == sub.L_scaling ||
            error("predict(:marginal): scaling `zvar` has $L columns but model δ " *
                  "expects $(sub.L_scaling). Pass the same no-constant Z used at fit time.")
    else
        L == sub.L ||
            error("predict(:marginal): hetero `zvar` has $L columns but model was " *
                  "fitted with $(sub.L). Pass the same zvar (same columns, same order) " *
                  "used at fit time.")
    end

    f = BE === MLE_Backend ? _marginal_Eu_mle_expo(coef, idx) :
        _marginal_Eu_closure(BE, coef, idx, ineff, hetero, ineff_type, scaling)

    out = Matrix{T}(undef, n, L)
    for i in 1:n
        zi = Vector{T}(view(Zn, i, :))
        out[i, :] .= gradient(f, zi)
    end
    # A Lomax fit's marginal effects are NaN where α ≤ 1, since E(u) does not exist there (every row with
    # a NaN counts): one warning per call.
    if ineff === :Lomax
        sf_warn_lomax_mean(count(i -> any(isnan, view(out, i, :)), 1:n), n)
    end
    return out
end

# Total-E(u) closure differentiated per observation by `_predict_marginal_kernel`.
# `BE` is the backend module (MCI_Backend / MSLE_Backend). A homoscedastic channel
# of a two-parameter distribution receives a length-1 ones basis (⇒ zero gradient
# contribution), exactly mirroring `get_marg`'s `Zmarg = [one(T)]`; a single-
# parameter distribution passes the full z (its kernel sums only the leading
# `length(idx.ineff.<param>)` columns). Differentiating `uncondU(coef, idx, z, z)`
# w.r.t. one shared z reproduces `get_marg`'s channel-summed (`addDataFrame`) result.
function _marginal_Eu_closure(BE::Module, coef, idx, ineff::Symbol,
                              hetero::AbstractVector, ineff_type, scaling::Bool)
    scaling && return z -> BE.uncondU_scaling(coef, idx, z, ineff_type)
    if ineff === :Exponential
        return z -> BE.uncondU_expo(coef, idx, z)
    elseif ineff === :Rayleigh
        return z -> BE.uncondU_rayleigh(coef, idx, z)
    elseif ineff === :Lognormal
        mh = :mu in hetero; sh = :sigma_u_sq in hetero
        return z -> BE.uncondU_lognorm(coef, idx,
                                       mh ? z : [one(eltype(z))],
                                       sh ? z : [one(eltype(z))])
    elseif ineff === :Gamma
        kh = :k in hetero; th = :theta in hetero
        return z -> BE.uncondU_gamma(coef, idx,
                                     kh ? z : [one(eltype(z))],
                                     th ? z : [one(eltype(z))])
    elseif ineff in (:Erlang2, :Erlang3, :Erlang4)
        M = parse(Int, string(ineff)[end])
        th = :theta in hetero
        return z -> BE.uncondU_erlang(coef, idx, th ? z : [one(eltype(z))], M)
    elseif ineff === :Lomax
        lh = :lambda in hetero; ah = :alpha in hetero
        return z -> BE.uncondU_lomax(coef, idx,
                                     lh ? z : [one(eltype(z))],
                                     ah ? z : [one(eltype(z))])
    elseif ineff === :Weibull
        lh = :lambda in hetero; kh = :k in hetero
        return z -> BE.uncondU_weibull(coef, idx,
                                       lh ? z : [one(eltype(z))],
                                       kh ? z : [one(eltype(z))])
    else
        error("predict(:marginal): kernel path does not support ineff=:$ineff.")
    end
end

# E(u) of a cross-sectional :MLE Exponential fit, as the fit itself differentiates it for
# `result.marginal`: `marg_expo` (src/sf_MLE/SFmarginal.jl, called by
# `get_marg(::Type{Expo}, ...)`), σᵤ = √σᵤ² with σᵤ² = exp(z'δ) clamped to [1e-12, 1e12].
# δ sits where `spec.mci_spec.idx` puts ln σᵤ², which is the order of the MLE coefficients
# of this model (as in `_predict_eff_mle_exp`);
# `marg_expo` reads no module state. On the fit's zvar, the non-constant columns of
# predict(:marginal) therefore equal `result.marginal` bit for bit.
function _marginal_Eu_mle_expo(coef, idx)
    rng = _as_range(idx.ineff.sigma_sq)
    pos = (begw = first(rng), endw = last(rng))
    cv  = Vector{Float64}(coef)
    return z -> MLE_Backend.SFmle.marg_expo(pos, cv, z)
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
        # E(u) = σu·m(a). The form below cancels (μ ≈ -σu·φ/Φ) and
        # ForwardDiff's derivative of φ(a)/Φ(a) squares Φ(a): the gradient is off by about
        # 4e-11 relative just above a = -10 and 1.6e-10 just below, of the wrong sign from about
        # a = -27, -Inf from -28 and NaN from -38. Below a = -10 the stable m of
        # SFNumerics.sf_tn_mean is used instead; above, the form below, bit for bit.
        a < -10 && return σu * sf_tn_mean(a)
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
        # The form below cancels as a falls and is NaN below about a = -38.5 (Φ underflows);
        # below a = -10, σu·m(a) with SFNumerics.sf_tn_mean.
        a < -10 && return σu * sf_tn_mean(a)
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
# Homoscedastic parameters are a constant for MLE (`_cons`), or coef·Z[:,1] for
# MCI/MSLE (matching get_ineff_vals) — see `_evaluate_param`.
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
        σu²  = _evaluate_param(r, spec, Zn, σfield, :sigma_u_sq in hetero, exponentiate = true)
    else # TruncatedNormal
        μvec = _evaluate_param(r, spec, Zn, :mu,    :mu       in hetero, exponentiate = false)
        σu²  = _evaluate_param(r, spec, Zn, σfield, :sigma_u_sq in hetero, exponentiate = true)
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
        # Homoscedastic — a true constant on every backend, matching the MLE `_cons`
        # semantics. The MCI/MSLE spec constructors require the first zvar column to be the
        # ones intercept whenever a parameter is homoscedastic, so the likelihood's
        # `coef·Z[:,1]` ≡ `coef·1`. Evaluate the constant directly, so predict is independent
        # of the new-data zvar's first column and predict(:jlms) stays consistent
        # with predict(:marginal)'s `_eval_μ`/`_eval_lnσu²`.
        lin = fill(γ[1], n)
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
    if :sigma_u_sq in hetero
        length(γ) == length(z) || error("predict: σ_u² hetero length mismatch.")
        return exp(dot(z, γ))
    else
        return exp(length(γ) == 1 ? γ[1] : γ[1])
    end
end

using LinearAlgebra: dot
