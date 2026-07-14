# Regression tests for the 03_reaudit_findings.md Group 1 (G1) fixes.
# Each testset targets a specific finding and would have failed before its fix.
# Kept self-contained (own data) so it can run independently of runtests.jl.

using SFrontiers
using Test
using Random
using Statistics
using Optim
using LinearAlgebra: diag
using StatsAPI: stderror, vcov

const _MCI   = SFrontiers.MCI_Backend
const _MSLE  = SFrontiers.MSLE_Backend
const _PANEL = SFrontiers.Panel_Backend
const _MLESF = SFrontiers.MLE_Backend.SFmle   # MLE internals live in a nested SFmle module

# ---------------------------------------------------------------------------
# R-06 — REGRESSION (FIX-20): `sqrt.(diag(vcov))` threw a DomainError instead of
# producing NaN, so every redflag fit crashed at the standard-error line and lost
# its coefficients/loglik/jlms/bc. Also, the `redflag=1` branch returned an
# uninitialized `similar(H)`, whose garbage diagonal could trigger the same throw.
# ---------------------------------------------------------------------------
@testset "R-06 non-PD / non-invertible Hessian no longer crashes the SE line" begin

    # ---- Unit level: `_invert_hessian` + the NaN-safe `_stderr_from_vcov` helper.
    for backend in (_MCI, _MSLE)
        # (a) Singular Hessian: inv(H) throws -> redflag == 1, and the returned
        #     var_cov_matrix must be a *defined* all-NaN matrix (before the fix it
        #     was an uninitialized `similar(H)` — almost never all-NaN).
        r1 = backend._invert_hessian([1.0 1.0; 1.0 1.0]; message = false)
        @test r1.redflag == 1
        @test all(isnan, r1.var_cov_matrix)

        # (b) Indefinite but invertible Hessian: inv(H) has a negative diagonal ->
        #     redflag == 2 (NOT collapsed to 1), with a finite valid inverse.
        r2 = backend._invert_hessian([1.0 0.0; 0.0 -1.0]; message = false)
        @test r2.redflag == 2
        @test all(isfinite, r2.var_cov_matrix)
        @test any(<(0), diag(r2.var_cov_matrix))

        # (c) The SE helper maps a negative variance to NaN without throwing
        #     (the old inline `sqrt.(diag(...))` threw DomainError here).
        se = backend._stderr_from_vcov([1.0 0.0; 0.0 -1.0])
        @test se[1] == 1.0
        @test isnan(se[2])
    end

    # Panel backend twin (`_panel_invert_hessian` + `_stderr_from_vcov`).
    rp1 = _PANEL._panel_invert_hessian([1.0 1.0; 1.0 1.0]; message = false)
    @test rp1.redflag == 1
    @test all(isnan, rp1.var_cov_matrix)
    rp2 = _PANEL._panel_invert_hessian([1.0 0.0; 0.0 -1.0]; message = false)
    @test rp2.redflag == 2
    sep = _PANEL._stderr_from_vcov([1.0 0.0; 0.0 -1.0])
    @test sep[1] == 1.0 && isnan(sep[2])

    # ---- End-to-end: a perfectly collinear frontier makes the post-hoc Hessian
    #      exactly singular. Before the fix the fit crashed at the stddev line and
    #      lost the result; now it must return a full result with NaN SEs and
    #      redflag > 0, no exception.
    Random.seed!(2024)
    n = 150
    cons = ones(n); x1 = randn(n)
    Xsing = hcat(cons, x1, x1)                         # duplicate column => rank deficient
    β = [1.0, 0.5, 0.0]
    y = Xsing * β .+ 0.3 .* randn(n) .- abs.(0.4 .* randn(n))
    opt = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40,),
                      main_solver = NelderMead(), main_opt = (iterations = 200,))

    for m in (:MCI, :MSLE)
        spec = sfmodel_spec(depvar = y, frontier = Xsing,
                            noise = :Normal, ineff = :HalfNormal, type = :prod)
        init = sfmodel_init(spec = spec, frontier = Xsing \ y,
                            ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
        # If the SE line still threw, this call would raise and error the testset.
        res = sfmodel_fit(spec = spec,
                          method = sfmodel_method(method = m, n_draws = 50),
                          init = init, optim_options = opt,
                          marginal = false, show_table = false, verbose = false)
        @test res isa SFResult
        @test res.redflag > 0                          # degenerate fit is flagged
        @test length(res.coeff) == 5                   # coefficients preserved
        @test isfinite(res.loglikelihood)              # loglik preserved
        @test all(isfinite, res.jlms) && all(isfinite, res.bc)   # indices preserved
        @test all(isnan, stderror(res))                # NaN SEs, not a crash
        @test all(isnan, diag(vcov(res)))              # stderror NaN-consistent with vcov
    end
end

# ---------------------------------------------------------------------------
# R-23 — DOCSTRING FIX: `redflag=2` (non-positive var-cov diagonal) is a sensible
# 3-valued flag but the docstrings/manual only documented {0,1}. Decision: document
# {0,1,2}; do NOT collapse 2 -> 1. So the code must keep returning a distinct 2, and
# the public fit-function docstrings must document all three values.
# ---------------------------------------------------------------------------
@testset "R-23 redflag {0,1,2} documented and not collapsed to 1" begin

    # The three-valued flag is preserved by the code: a non-invertible Hessian is
    # redflag 1, a non-positive var-cov diagonal is the *distinct* redflag 2.
    @test _MCI._invert_hessian([1.0 1.0; 1.0 1.0]; message = false).redflag == 1
    @test _MCI._invert_hessian([1.0 0.0; 0.0 -1.0]; message = false).redflag == 2
    @test _MSLE._invert_hessian([1.0 0.0; 0.0 -1.0]; message = false).redflag == 2
    @test _PANEL._panel_invert_hessian([1.0 0.0; 0.0 -1.0]; message = false).redflag == 2

    # The public fit-function docstrings must now document redflag = 2 (the string
    # "non-positive" appears only via the redflag=2 description; the old docstrings
    # said only "(0=OK, 1=problems)").
    doc_mci  = lowercase(string(@doc SFrontiers.MCI_Backend.sfmodel_MCI_fit))
    doc_msle = lowercase(string(@doc SFrontiers.MSLE_Backend.sfmodel_MSLE_fit))
    @test occursin("redflag", doc_mci)  && occursin("non-positive", doc_mci)
    @test occursin("redflag", doc_msle) && occursin("non-positive", doc_msle)
end

# ---------------------------------------------------------------------------
# R-06 (MLE backend) — full parity with the MCI/MSLE/panel backends. A degenerate
# var-cov (a non-invertible Hessian, or an invertible one with a non-positive
# diagonal) now yields NaN standard errors and redflag instead of throwing:
# previously the non-positive-diagonal branch fell through to a plain
# `sqrt.(diag(...))` (DomainError) and the non-invertible branch threw an error.
# ---------------------------------------------------------------------------
@testset "R-06 (MLE backend) NaN-safe standard errors" begin
    # Helper: a non-positive diagonal maps to NaN without throwing (the old inline
    # `sqrt.(diag(...))` threw a DomainError); a PD vcov still gives correct SEs.
    # (The MLE backend wraps its internals in a nested `SFmle` module.)
    se = _MLESF._stderr_from_vcov([1.0 0.0; 0.0 -1.0])
    @test se[1] == 1.0 && isnan(se[2])
    @test _MLESF._stderr_from_vcov([4.0 0.0; 0.0 9.0]) == [2.0, 3.0]

    # End-to-end: a zero frontier column makes the post-hoc Hessian exactly
    # singular. Before, the MLE fit threw ("...not invertible..." / the
    # checkCollinear "Multi-collinearity" abort); now it returns a full result
    # with NaN SEs and redflag set, no exception — parity with the other backends.
    Random.seed!(11)
    n = 150
    cons = ones(n); x1 = randn(n)
    Xsing = hcat(cons, x1, zeros(n))                 # zero column => exactly singular Hessian
    y = Xsing * [1.0, 0.5, 0.0] .+ 0.3 .* randn(n) .- abs.(0.4 .* randn(n))
    spec = sfmodel_spec(depvar = y, frontier = Xsing,
                        noise = :Normal, ineff = :HalfNormal, type = :prod)
    init = sfmodel_init(spec = spec, frontier = Xsing \ y,
                        ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
    opt = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40,),
                      main_solver = NelderMead(), main_opt = (iterations = 200,))
    res = sfmodel_fit(spec = spec, method = sfmodel_method(method = :MLE),
                      init = init, optim_options = opt,
                      marginal = false, show_table = false, verbose = false)
    @test res isa SFResult
    @test res.backend === :MLE
    @test res.redflag == true                        # degenerate fit is flagged
    @test length(res.coeff) == 5                     # coefficients preserved
    @test isfinite(res.loglikelihood)                # loglik preserved
    @test all(isfinite, res.jlms) && all(isfinite, res.bc)   # indices preserved
    @test all(isnan, stderror(res))                  # NaN SEs, not a throw/crash
end
