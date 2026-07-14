# Regression tests for the 03_reaudit_findings.md Group 5 (G5) fixes — kernel
# labels & panel input validation: R-12, R-11, R-14, R-24, R-21. Each testset
# targets a specific finding and would have failed before its fix. Kept
# self-contained (own data) so it can run independently of runtests.jl.

using SFrontiers
using Test
using Random
using Statistics
using Optim
using LinearAlgebra: diag

const _MCI   = SFrontiers.MCI_Backend
const _MSLE  = SFrontiers.MSLE_Backend
const _PANEL = SFrontiers.Panel_Backend

# Loose NelderMead options — G5 assertions are about table labels / clamp
# alignment / init validation / result keys, not the quality of the optimum.
_g5opt() = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40,),
                       main_solver = NelderMead(), main_opt = (iterations = 200,))

# Return only the (label, offset) pairs from a `_detect_log_params` entry list,
# so a testset can assert the labels/offsets independent of parameter indices.
_labels(entries)  = [e[1] for e in entries]
_offset(entries, name) = only(e[3] for e in entries if e[1] == name)

# ---------------------------------------------------------------------------
# R-12 — the MSLE auxiliary "Log-parameters converted to original scale" table
# enumerated at most one inefficiency entry via an elseif chain, so it silently
# omitted the second (shape) parameter — Weibull k and Lomax α — and StudentT's
# ν. `_detect_log_params` now pushes the shape parameter independently and adds
# ν = exp(ln(ν−2)) + 2 (entries carry an additive offset: 0 normally, 2 for ν).
# ---------------------------------------------------------------------------
@testset "R-12 MSLE aux table shows Weibull k, Lomax α and StudentT ν" begin
    dlp(noise, ineff) = _MSLE._detect_log_params(
        _MSLE._build_model(noise, ineff),
        _MSLE._param_ind(_MSLE._build_model(noise, ineff), 2, 1, Symbol[]),
        1, Symbol[])

    # Weibull (homoscedastic): both the scale λ and the shape k must appear.
    ew = dlp(:Normal, :Weibull)
    @test "λ" in _labels(ew)
    @test "k" in _labels(ew)                 # was silently omitted before R-12
    @test _offset(ew, "k") == 0.0

    # Lomax: both λ and α.
    el = dlp(:Normal, :Lomax)
    @test "λ" in _labels(el)
    @test "α" in _labels(el)                 # was silently omitted before R-12

    # StudentT noise: σᵥ² still shown, plus ν with the +2 offset (ν = exp(·)+2).
    es = dlp(:StudentT, :HalfNormal)
    @test "σᵥ²" in _labels(es)
    @test "ν"  in _labels(es)                # was silently omitted before R-12
    @test _offset(es, "ν")   == 2.0          # ν = exp(ln(ν−2)) + 2, not a plain exp
    @test _offset(es, "σᵥ²") == 0.0
    @test _offset(es, "σᵤ²") == 0.0

    # Integration: the (name, idx, offset) 3-tuple aux loop runs end-to-end in
    # print_table without error (guards the loop-unpacking change).
    Random.seed!(512)
    n = 150
    X = hcat(ones(n), randn(n), randn(n))
    y = X * [1.0, 0.5, -0.3] .+ 0.3 .* randn(n) .- abs.(0.5 .* randn(n))
    sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                      ineff = :HalfNormal, type = :prod)
    init = sfmodel_init(spec = sp, frontier = X \ y,
                        ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
    local res
    redirect_stdout(devnull) do
        res = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MSLE, n_draws = 100),
                          init = init, optim_options = _g5opt(),
                          marginal = false, show_table = true, verbose = false)
    end
    @test res isa SFResult
end

# ---------------------------------------------------------------------------
# R-11 — the MCI auxiliary table (`_detect_log_params`) was the pre-FIX-18 twin
# of MSLE's: it mislabeled Lognormal/Rayleigh `σ²` as `σᵤ²`, had no `:ln_lambda`
# branch (so Lomax's λ was omitted), and — like MSLE before R-12 — dropped the
# shape parameter (Weibull k) and StudentT ν. It now mirrors the completed MSLE
# twin, so identical models print identical aux tables per backend.
# ---------------------------------------------------------------------------
@testset "R-11 MCI aux table mirrors the MSLE twin" begin
    dlp_mci(noise, ineff) = _MCI._detect_log_params(
        _MCI._build_model(noise, ineff),
        _MCI._param_ind(_MCI._build_model(noise, ineff), 2, 1, Symbol[]),
        1, Symbol[])

    # Lognormal/Rayleigh: the σ variance is labeled σ² (Var(ln u)/squared scale),
    # NOT σᵤ² (the old MCI mislabel).
    @test "σ²"  in _labels(dlp_mci(:Normal, :Lognormal))
    @test "σᵤ²" ∉  _labels(dlp_mci(:Normal, :Lognormal))
    @test "σ²"  in _labels(dlp_mci(:Normal, :Rayleigh))
    # Lomax: λ now detected via the :ln_lambda field, plus the shape α.
    @test "λ" in _labels(dlp_mci(:Normal, :Lomax))     # was omitted (no :ln_lambda branch)
    @test "α" in _labels(dlp_mci(:Normal, :Lomax))
    # Weibull: λ and k (k was dropped before R-11).
    @test "λ" in _labels(dlp_mci(:Normal, :Weibull))
    @test "k" in _labels(dlp_mci(:Normal, :Weibull))
    # StudentT noise: ν with the +2 offset.
    @test _offset(dlp_mci(:StudentT, :HalfNormal), "ν") == 2.0
    # Gamma is MCI-only (no MSLE twin) and must still report both k and θ.
    eg = dlp_mci(:Normal, :Gamma)
    @test "k" in _labels(eg) && "θ" in _labels(eg)

    # Core parity: for every distribution both backends support, the MCI and MSLE
    # aux-table entries — labels, indices AND offsets — are now identical.
    for (noise, ineff) in [(:Normal, :HalfNormal), (:Normal, :TruncatedNormal),
                           (:Normal, :Lognormal), (:Normal, :Rayleigh),
                           (:Normal, :Exponential), (:Normal, :Weibull),
                           (:Normal, :Lomax), (:StudentT, :HalfNormal),
                           (:StudentT, :Weibull), (:Laplace, :Lomax)]
        mmci  = _MCI._build_model(noise, ineff)
        mmsle = _MSLE._build_model(noise, ineff)
        emci  = _MCI._detect_log_params(mmci,  _MCI._param_ind(mmci, 2, 1, Symbol[]),   1, Symbol[])
        emsle = _MSLE._detect_log_params(mmsle, _MSLE._param_ind(mmsle, 2, 1, Symbol[]), 1, Symbol[])
        @test emci == emsle
    end

    # Integration: the MCI 3-tuple aux loop runs end-to-end in print_table.
    Random.seed!(511)
    n = 150
    X = hcat(ones(n), randn(n), randn(n))
    y = X * [1.0, 0.5, -0.3] .+ 0.3 .* randn(n) .- abs.(0.5 .* randn(n))
    sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                      ineff = :HalfNormal, type = :prod)
    init = sfmodel_init(spec = sp, frontier = X \ y,
                        ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
    local res
    redirect_stdout(devnull) do
        res = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MCI, n_draws = 100),
                          init = init, optim_options = _g5opt(),
                          marginal = false, show_table = true, verbose = false)
    end
    @test res isa SFResult
end

# ---------------------------------------------------------------------------
# R-14 — FIX-15 kernel-aligned the panel NLL clamps but the marginal-effect E(u)
# helpers `_panel_uncondU_{weibull,gamma,lomax}` kept pre-FIX-15 literal clamps
# ([1e-6,1e6] shape, [1e-12,1e6] Lomax scale), so a clamp-active fit computed its
# marginal at a different effective parameter than its jlms/bc. The helpers now
# take the constants NamedTuple `c` and clamp with `c.weibull_k_ceil`,
# `c.gamma_k_ceil`, `c.xm_*`, `c.alpha_*` — the exact bounds `get_panel_ineff_vals`
# uses (Weibull k∈[0.1,10], Gamma k∈[0.1,100], Lomax λ∈[1e-10,1e10]).
# ---------------------------------------------------------------------------
@testset "R-14 panel marginal helpers use the FIX-15 kernel clamps" begin
    z = [1.0]
    # Weibull: shape k clamps at c.weibull_k_ceil = 10 (was 1e6). Above the ceiling
    # the effective k is pinned, so ln_k=log(50) and ln_k=log(10) give the same E(u),
    # while ln_k=log(5) (below the ceiling) differs.
    cw   = _PANEL.make_panel_constants(_PANEL._build_panel_model(:Normal, :Weibull), Float64, 1)
    idxw = (delta = 1:1, ln_lambda = 2, ln_k = 3)
    euw(lnk) = _PANEL._panel_uncondU_weibull([0.0, log(2.0), lnk], idxw, z, cw)
    @test euw(log(50.0)) ≈ euw(log(10.0))        # clamped at k = 10
    @test !(euw(log(50.0)) ≈ euw(log(5.0)))      # k = 5 < 10 ⇒ not clamped, differs

    # Gamma: shape k clamps at c.gamma_k_ceil = 100 (was 1e6). E(u)=h·k·θ (no Γ),
    # so with θ=2, ln_k=log(500) ⇒ k=100 ⇒ E(u)=200 (was 1000 with the loose clamp).
    cg   = _PANEL.make_panel_constants(_PANEL._build_panel_model(:Normal, :Gamma), Float64, 1)
    idxg = (delta = 1:1, ln_k = 2, ln_theta = 3)
    @test _PANEL._panel_uncondU_gamma([0.0, log(500.0), log(2.0)], idxg, z, cg) ≈ 200.0

    # Lomax: scale λ ceiling was *tighter* in the marginal (1e6) than the NLL (1e10).
    # E(u)=h·λ/(α−1). With α=2, ln_lambda=log(1e8) is below the kernel ceiling 1e10,
    # so λ passes through as 1e8 (was clamped to 1e6 by the old marginal literal).
    cl   = _PANEL.make_panel_constants(_PANEL._build_panel_model(:Normal, :Lomax), Float64, 1)
    idxl = (delta = 1:1, ln_lambda = 2, ln_alpha = 3)
    eul(lnλ) = _PANEL._panel_uncondU_lomax([0.0, lnλ, log(2.0)], idxl, z, cl)
    @test eul(log(1e8)) ≈ 1e8                    # not clamped to 1e6 anymore
    @test eul(log(1e12)) ≈ eul(log(1e10))        # both clamp λ at c.xm_ceil = 1e10
    @test !(eul(log(1e12)) ≈ eul(log(1e8)))      # 1e8 < 1e10 ⇒ not clamped, differs

    # Integration: a panel_TFE Weibull fit with marginal=true exercises the closure
    # wiring (get_panel_marg builds `c` and passes it to the helper); before the fix
    # the helper took 3 args, so this dispatch would not even compile once threaded.
    Random.seed!(514)
    Nf = 25; Tp = 5; NT = Nf * Tp; id = repeat(1:Nf, inner = Tp)
    Xp = hcat(randn(NT), randn(NT))
    αi = repeat(2 .* randn(Nf), inner = Tp)
    zp = randn(NT)
    yp = αi .+ Xp * [0.6, -0.5] .+ 0.3 .* randn(NT) .- abs.(0.5 .* randn(NT))
    local rw
    redirect_stdout(devnull) do
        rw = sfmodel_fit(spec = sfmodel_spec(depvar = yp, frontier = Xp, zvar = zp,
                             noise = :Normal, ineff = :Weibull, type = :prod,
                             datatype = :panel_TFE, id = id),
                         method = sfmodel_method(method = :MSLE, n_draws = 100),
                         optim_options = _g5opt(), marginal = true,
                         show_table = false, verbose = false)
    end
    @test rw isa SFResult
    @test rw.marginal !== nothing
end

# ---------------------------------------------------------------------------
# R-24 — `sfmodel_panel_init`'s strict "reject kwargs that don't apply" block
# checked mu/ln_lambda/ln_k/ln_alpha/ln_theta but omitted `ln_sigma_sq`, so for
# the four distributions with neither `:ln_sigma_sq` nor `:ln_sigma_u_sq` in
# their index (Exp/Weibull/Lomax/Gamma) a supplied `ln_sigma_sq` was silently
# dropped (parameter stayed at the 0.1 default), contradicting the strict
# contract that every sibling kwarg errors. `ln_sigma_sq` is now rejected for
# exactly those distributions, while still accepted for HalfNormal/TN (maps to
# :ln_sigma_u_sq) and Lognormal/Rayleigh (:ln_sigma_sq).
# ---------------------------------------------------------------------------
@testset "R-24 sfmodel_panel_init rejects ln_sigma_sq where it does not apply" begin
    Random.seed!(524)
    Nf = 20; Tp = 5; NT = Nf * Tp; id = repeat(1:Nf, inner = Tp)
    Xp = hcat(randn(NT), randn(NT))                 # no intercept (FE absorbs it)
    αi = repeat(2 .* randn(Nf), inner = Tp)
    zp = randn(NT)
    yp = αi .+ Xp * [0.6, -0.5] .+ 0.3 .* randn(NT) .- abs.(0.5 .* randn(NT))

    panel_spec(ineff) = sfmodel_spec(depvar = yp, frontier = Xp, zvar = zp,
                                     noise = :Normal, ineff = ineff, type = :prod,
                                     datatype = :panel_TFE, id = id)

    # The four distributions with no σ² / σᵤ² index parameter must now error
    # (before R-24 this returned defaults silently).
    for ineff in (:Exponential, :Weibull, :Lomax, :Gamma)
        @test_throws "ln_sigma_sq" sfmodel_init(spec = panel_spec(ineff), ln_sigma_sq = 0.5)
    end

    # Distributions that DO own the variance parameter still accept ln_sigma_sq:
    # HalfNormal/TN store it under :ln_sigma_u_sq, Lognormal/Rayleigh under
    # :ln_sigma_sq — the init must return a full vector, not throw.
    for ineff in (:HalfNormal, :TruncatedNormal, :Lognormal, :Rayleigh)
        v = sfmodel_init(spec = panel_spec(ineff), ln_sigma_sq = log(0.25))
        @test v isa Vector && all(isfinite, v)
    end

    # A sibling mismatched kwarg still errors (guards the block is otherwise intact).
    @test_throws "ln_lambda" sfmodel_init(spec = panel_spec(:HalfNormal), ln_lambda = 0.5)
end

# ---------------------------------------------------------------------------
# R-21 — HalfNormal stores its ln σᵤ² under the idx field `:sigma_sq` (only
# TruncatedNormal has `:sigma_u`), so `result.sigma_u` — which the manual
# reserves for HalfNormal/TN — errored for a HalfNormal fit. Decision: add the
# alias. The result now also stores `:sigma_u` = `:sigma_sq` for HalfNormal (MCI
# and MSLE), giving HN/TN parity; Lognormal/Rayleigh (whose `:sigma_sq` is a
# different quantity) are deliberately NOT aliased.
# ---------------------------------------------------------------------------
@testset "R-21 result.sigma_u aliases sigma_sq for HalfNormal (MCI + MSLE)" begin
    Random.seed!(521)
    n = 150
    X = hcat(ones(n), randn(n), randn(n))
    βo = X \ (X * [1.0, 0.5, -0.3])
    y = X * [1.0, 0.5, -0.3] .+ 0.3 .* randn(n) .- abs.(0.5 .* randn(n))

    sp_hn = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                         ineff = :HalfNormal, type = :prod)
    init_hn = sfmodel_init(spec = sp_hn, frontier = βo,
                           ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
    for m in (:MSLE, :MCI)
        r = sfmodel_fit(spec = sp_hn, method = sfmodel_method(method = m, n_draws = 100),
                        init = init_hn, optim_options = _g5opt(),
                        marginal = false, show_table = false, verbose = false)
        @test hasproperty(r, :sigma_u)               # errored (FieldError) before R-21
        @test r.sigma_u == r.sigma_sq                # alias, same stored ln σᵤ²
    end

    # Negative guard: Rayleigh also stores its parameter under :sigma_sq, but it is
    # NOT σᵤ² — so the alias must be HalfNormal-specific, not `:sigma_sq`-generic.
    r_ray = sfmodel_fit(spec = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                            ineff = :Rayleigh, type = :prod),
                        method = sfmodel_method(method = :MSLE, n_draws = 100),
                        optim_options = _g5opt(), marginal = false,
                        show_table = false, verbose = false)
    @test hasproperty(r_ray, :sigma_sq)
    @test !hasproperty(r_ray, :sigma_u)

    # TruncatedNormal keeps its own :sigma_u (unchanged by the alias).
    r_tn = sfmodel_fit(spec = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                            ineff = :TruncatedNormal, type = :prod),
                        method = sfmodel_method(method = :MSLE, n_draws = 100),
                        optim_options = _g5opt(), marginal = false,
                        show_table = false, verbose = false)
    @test hasproperty(r_tn, :sigma_u)
end
