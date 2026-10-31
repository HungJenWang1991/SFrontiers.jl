# Regression tests: diagnostics across backends.
# Kept self-contained (own data) so it can run independently of runtests.jl.

using SFrontiers
using Test
using Random
using Statistics
using Optim
using StatsAPI
using StatsAPI: residuals, fitted
using RecipesBase

# Loose NelderMead options: the assertions are about diagnostic wiring and within-firm algebra, not the quality of the optimum.
_g3opt() = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40,),
                       main_solver = NelderMead(), main_opt = (iterations = 200,))

# ---------------------------------------------------------------------------
# In an earlier form `residuals`/`fitted`/`residual_diagnostics`/`plot` (a documented,
# backend-uniform surface) errored for every panel-sim (:MSLE/:MCI), CSW, and TRE
# fit, because the data-recovery fallback only knew the MLE/MCI/panel_spec shapes.
# `_attach_diagnostic_data` now recovers the demeaned data for panel-sim (from the
# panel_spec) and the raw stacked data for CSW/TRE (from the MLE spec). The whole
# surface must return finite vectors instead of throwing "depvar not recoverable".
# ---------------------------------------------------------------------------
@testset "diagnostics work for panel-sim, CSW and TRE fits" begin
    Random.seed!(303)
    Nf = 30; Tp = 5; NT = Nf * Tp; id = repeat(1:Nf, inner = Tp)
    Xp = hcat(randn(NT), randn(NT))                # no intercept (FE absorbs it)
    αi = repeat(2 .* randn(Nf), inner = Tp)
    zp = randn(NT)                                 # within-firm-varying scaling z
    yp = αi .+ Xp * [0.6, -0.5] .+ 0.3 .* randn(NT) .- abs.(0.5 .* randn(NT))

    # panel_TFE via a simulation method (result.model is a _PanelInternalSpec).
    r_sim = sfmodel_fit(spec = sfmodel_spec(depvar = yp, frontier = Xp, zvar = zp,
                            noise = :Normal, ineff = :HalfNormal, type = :prod,
                            datatype = :panel_TFE, id = id),
                        method = sfmodel_method(method = :MSLE, n_draws = 100),
                        optim_options = _g3opt(), marginal = false,
                        show_table = false, verbose = false)
    # panel_TFE_CSW (Chen–Schmidt–Wang 2014); result.model is a display String.
    r_csw = sfmodel_fit(spec = sfmodel_spec(depvar = yp, frontier = Xp, noise = :Normal,
                            ineff = :HalfNormal, type = :prod,
                            datatype = :panel_TFE_CSW, id = id),
                        method = sfmodel_method(method = :MLE),
                        optim_options = _g3opt(), marginal = false,
                        show_table = false, verbose = false)
    # panel_TRE (true random effects); MLE only, also String result.model.
    r_tre = sfmodel_fit(spec = sfmodel_spec(depvar = yp, frontier = Xp, noise = :Normal,
                            ineff = :HalfNormal, type = :prod,
                            datatype = :panel_TRE, id = id),
                        method = sfmodel_method(method = :MLE),
                        optim_options = _g3opt(), marginal = false,
                        show_table = false, verbose = false)

    for r in (r_sim, r_csw, r_tre)
        # Every documented residual/fitted target returns finite values (before the
        # fix these threw "diagnostics: depvar not recoverable").
        for t in (:composed, :u, :v, :ols)
            resid = residuals(r; type = t)
            @test length(resid) == NT && all(isfinite, resid)
        end
        @test all(isfinite, fitted(r; type = :frontier))
        @test all(isfinite, fitted(r; type = :response))
        rd = residual_diagnostics(r)
        @test isfinite(rd.skew_composed) && isfinite(rd.skew_ols)
        # Plot recipe (residual panel) no longer errors.
        @test !isempty(RecipesBase.apply_recipe(Dict{Symbol, Any}(), r, :residuals))
    end

    # The panel-sim fit must recover the *demeaned* data (not the raw stacked data):
    # each firm's composed residual then averages to ~0.
    ε_sim = vec(residuals(r_sim; type = :composed))
    wmean = [abs(mean(ε_sim[(f-1)*Tp+1:f*Tp])) for f in 1:Nf]
    @test maximum(wmean) < 1e-8
end

# ---------------------------------------------------------------------------
# For Wang–Ho panel fits the composed residual is within-firm demeaned (ε̃),
# but r.jlms = h_it·E[u*|ε̃] uses the *non-demeaned* h_it, so the old
# `v̂ = ε̃ + sign·jlms` was off by the firm-mean term h̄ᵢ·E[u*|·]. Since E[u*|ε̃] is
# firm-constant, the demeaned inefficiency ũ_it = h̃_it·E[u*|·] equals the
# within-firm demeaning of jlms, so `:v`/`:response` now subtract that instead.
# ---------------------------------------------------------------------------
@testset "panel Wang-Ho :v/:response use within-firm demeaned inefficiency" begin
    Random.seed!(326)
    Nf = 40; Tp = 4; NT = Nf * Tp; id = repeat(1:Nf, inner = Tp)
    Xp = hcat(randn(NT), randn(NT))
    αi = repeat(2 .* randn(Nf), inner = Tp)
    zp = randn(NT)                                   # within-firm-varying scaling z
    yp = αi .+ Xp * [0.6, -0.5] .+ 0.3 .* randn(NT) .- abs.(0.5 .* randn(NT))

    # Balanced-panel within-firm demean and per-firm means.
    dm(v) = (o = copy(v); for f in 1:Nf; ix = (f-1)*Tp+1:f*Tp; o[ix] .-= mean(v[ix]); end; o)
    wmax(v) = maximum(abs, [mean(v[(f-1)*Tp+1:f*Tp]) for f in 1:Nf])

    # panel_TFE via MLE (Wang–Ho half-normal) and via MSLE — both carry firm offsets.
    for meth in (sfmodel_method(method = :MLE),
                 sfmodel_method(method = :MSLE, n_draws = 100))
        r = sfmodel_fit(spec = sfmodel_spec(depvar = yp, frontier = Xp, zvar = zp,
                            noise = :Normal, ineff = :HalfNormal, type = :prod,
                            datatype = :panel_TFE, id = id),
                        method = meth, optim_options = _g3opt(),
                        marginal = false, show_table = false, verbose = false)
        ε   = vec(residuals(r; type = :composed))
        jl  = vec(collect(r.jlms))
        v̂   = vec(residuals(r; type = :v))
        sgn = 1.0                                     # :prod ⇒ +1

        # v̂ = ε̃ + sign·demean(jlms), and each firm's v̂ averages to ~0.
        @test v̂ ≈ ε .+ sgn .* dm(jl)
        @test wmax(v̂) < 1e-8
        # fitted(:response) subtracts the same demeaned inefficiency.
        @test vec(fitted(r; type = :response)) ≈ vec(fitted(r; type = :frontier)) .- sgn .* dm(jl)

        # Discriminating contrast: the OLD formula (raw jlms) had a non-zero
        # within-firm mean — the firm-mean term h̄ᵢ·E[u*|·] this fix removes.
        @test wmax(ε .+ sgn .* jl) > 1e-3
    end

    # Cross-sectional fits carry no offsets ⇒ :v is unchanged (raw jlms).
    Xc = hcat(ones(NT), randn(NT))
    yc = Xc * [1.0, 0.5] .+ 0.3 .* randn(NT) .- abs.(0.4 .* randn(NT))
    rc = sfmodel_fit(spec = sfmodel_spec(depvar = yc, frontier = Xc, noise = :Normal,
                        ineff = :HalfNormal, type = :prod),
                     method = sfmodel_method(method = :MCI, n_draws = 100),
                     optim_options = _g3opt(), marginal = false,
                     show_table = false, verbose = false)
    @test SFrontiers._panel_offsets(rc) === nothing
    @test vec(residuals(rc; type = :v)) ≈ vec(residuals(rc; type = :composed)) .+
                                          vec(collect(rc.jlms))
end
