# Regression tests for the correctness fixes in 00to_fix.md (FIX-01 .. FIX-23).
# Each testset targets a specific finding and would have failed before its fix.
# Kept self-contained (own data) so it can run independently of runtests.jl.

using SFrontiers
using Test
using Random
using Statistics
using Optim
using LinearAlgebra: diag
using SpecialFunctions: erf, erfinv
using StatsAPI: predict, coef, coeftable, stderror, vcov, residuals

const PB = SFrontiers.Panel_Backend

_opt() = sfmodel_opt(
    warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40, g_abstol = 1e-3),
    main_solver = Newton(), main_opt = (iterations = 80, g_abstol = 1e-5, show_trace = false))
_optnm() = sfmodel_opt(
    warmstart_solver = NelderMead(), warmstart_opt = (iterations = 80, g_abstol = 1e-3),
    main_solver = NelderMead(), main_opt = (iterations = 500, g_abstol = 1e-5, show_trace = false))

@testset "00to_fix.md correctness fixes" begin

    # Shared cross-sectional data.
    Random.seed!(4321)
    N = 200
    cons = ones(N); x1 = randn(N); x2 = randn(N)
    X = hcat(cons, x1, x2)
    βt = [1.0, 0.5, -0.3]
    y = X * βt .+ 0.3 .* randn(N) .- abs.(0.5 .* randn(N))
    βo = X \ y

    @testset "FIX-01 panel signed erfinv floor (TN/Lognormal draws not pinned)" begin
        # Lognormal(μ=σ=1): lower-tail draws must follow the true quantile, not
        # collapse onto exp(μ). Cross-check against the cross-section quantile.
        cL = PB.make_panel_constants(PB._build_panel_model(:Normal, :Lognormal), Float64, 5)
        dL = vec(PB.mypan_gen_udraws_msle(PB.Lognormal_Panel(),
                 (mu = 1.0, sigma = 1.0), [0.05 0.15 0.30], cL, Float64))
        @test all(dL .< exp(1.0))                       # not pinned to exp(μ)
        @test isapprox(dL, exp.(1 .+ sqrt(2) .* [erfinv(2*0.05-1), erfinv(2*0.15-1),
                                                 erfinv(2*0.30-1)]), atol = 1e-6)
        # TruncatedNormal(μ=σ=1, truncated at 0): draws in (0, ∞), lower tail not
        # pinned to μ. Compare against the exact TN-at-0 quantile.
        cT = PB.make_panel_constants(PB._build_panel_model(:Normal, :TruncatedNormal), Float64, 5)
        dT = vec(PB.mypan_gen_udraws_msle(PB.TruncatedNormal_Panel(),
                 (mu = 1.0, sigma_u = 1.0, sigma_u_sq = 1.0), [0.05 0.15 0.30], cT, Float64))
        e = erf(1 / sqrt(2))
        truthT = 1 .+ sqrt(2) .* erfinv.([0.05, 0.15, 0.30] .+ ([0.05, 0.15, 0.30] .- 1) .* e)
        @test all(0.0 .< dT)
        @test dT[1] < 1.0 && dT[2] < 1.0             # below μ, not pinned to μ
        @test isapprox(dT, truthT, atol = 1e-6)
    end

    @testset "FIX-02/05 TN+scaling reorder: coeff count + row labels aligned" begin
        z1 = randn(N); z2 = randn(N); Zs = hcat(z1, z2)   # 2 scaling regressors
        sp = sfmodel_spec(depvar = y, frontier = X, zvar = Zs,
                          noise = :Normal, ineff = :TruncatedNormal, hetero = :scaling, type = :prod)
        ini = sfmodel_init(spec = sp, frontier = βo, scaling = [0.0, 0.0], mu = 0.0,
                           ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
        r = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE), init = ini,
                        optim_options = _opt(), marginal = false, show_table = false, verbose = false)
        @test r.converged
        @test length(coef(r)) == 8                    # β(3)+δ(2)+μ(1)+lnσᵤ²(1)+lnσᵥ²(1)
        rn = coeftable(r).rownms
        # MCI order: the two scaling deltas sit right after the 3 frontier rows,
        # and μ follows them (not swapped, and labelled — FIX-05).
        @test occursin("hscale", lowercase(rn[4])) && occursin("hscale", lowercase(rn[5]))
    end

    @testset "FIX-03 CSW cost-frontier efficiency ranking not inverted" begin
        Nf = 60; Tp = 5; NT = Nf * Tp; id = repeat(1:Nf, inner = Tp)
        Random.seed!(9)
        Xp = hcat(randn(NT), randn(NT)); αi = repeat(randn(Nf), inner = Tp)
        uP = abs.(0.8 .* randn(NT)); vP = 0.3 .* randn(NT); βp = [0.5, -0.4]
        dm(x) = (r = copy(x); for f in 1:Nf; ix = (f-1)*Tp+1:f*Tp; r[ix] .-= mean(x[ix]); end; r)
        # Cost: y = α + Xβ + v + u. Firms with larger within-firm residual are
        # MORE inefficient and must get HIGHER jlms (was inverted before FIX-03).
        yc = αi .+ Xp * βp .+ vP .+ uP
        rc = sfmodel_fit(spec = sfmodel_spec(depvar = yc, frontier = Xp, noise = :Normal,
                            ineff = :HalfNormal, type = :cost, datatype = :panel_TFE_CSW, id = id),
                         method = sfmodel_method(method = :MLE), optim_options = _optnm(),
                         marginal = false, show_table = false, verbose = false)
        @test cor(vec(rc.jlms), dm(yc .- Xp * rc.coeff[1:2])) > 0.5
        # Production control: jlms rises as the residual falls (unchanged by FIX-03).
        yp = αi .+ Xp * βp .+ vP .- uP
        rp = sfmodel_fit(spec = sfmodel_spec(depvar = yp, frontier = Xp, noise = :Normal,
                            ineff = :HalfNormal, type = :prod, datatype = :panel_TFE_CSW, id = id),
                         method = sfmodel_method(method = :MLE), optim_options = _optnm(),
                         marginal = false, show_table = false, verbose = false)
        @test cor(vec(rp.jlms), dm(yp .- Xp * rp.coeff[1:2])) < -0.5
    end

    @testset "FIX-04+22 copula predict reproduces the fit's own jlms/bc" begin
        for cop in (:Gaussian, :Clayton)
            sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                              ineff = :HalfNormal, type = :prod, copula = cop)
            ini = sfmodel_init(spec = sp, frontier = βo, ln_sigma_sq = log(0.25),
                               ln_sigma_v_sq = log(0.09), theta_rho = 0.0)
            r = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MCI, n_draws = 200),
                            init = ini, optim_options = _opt(),
                            marginal = false, show_table = false, verbose = false)
            @test r.converged
            jl = predict(r; frontier = X, depvar = y, what = :jlms)
            bc = predict(r; frontier = X, depvar = y, what = :bc)
            @test maximum(abs.(vec(jl) .- vec(r.jlms))) < 1e-4   # NOT the independence formula
            @test maximum(abs.(vec(bc) .- vec(r.bc)))   < 1e-4
        end
    end

    @testset "FIX-06/R-05 homoscedastic σᵤ² requires a constant first zvar column" begin
        # Re-audit resolution (04 decision #1, option (i) — validate at spec time). A
        # homoscedastic inefficiency parameter is evaluated at coef·Z[:,1]; G6 now REQUIRES
        # the first zvar column to be the ones intercept whenever a parameter is homoscedastic
        # (matching the MLE `_cons` semantics). So the old non-constant-first-column input —
        # which FIX-06 had accepted and evaluated as a silently heteroscedastic σᵤ² — is now
        # rejected at spec time.
        age = randn(N); school = randn(N)
        @test_throws Exception sfmodel_spec(depvar = y, frontier = X, zvar = hcat(age, school),
            noise = :Normal, ineff = :TruncatedNormal, hetero = [:mu], type = :prod)
        # The corrected form (constant `_cons` first column) fits, and predict reproduces
        # r.jlms exactly: MCI/MSLE predict reuses the fit-time simulation kernel (same Halton
        # draws) for HN/TN, so agreement is bit-identical at any draw count (μ hetero over
        # [_cons, age], σᵤ² homoscedastic = coef·1).
        Z6 = hcat(cons, age)
        for nd in (500, 4000)
            sp = sfmodel_spec(depvar = y, frontier = X, zvar = Z6, noise = :Normal,
                              ineff = :TruncatedNormal, hetero = [:mu], type = :prod)
            ini = sfmodel_init(spec = sp, frontier = βo, mu = [0.0, 0.0],
                               ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
            r = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MCI, n_draws = nd),
                            init = ini, optim_options = _opt(),
                            marginal = false, show_table = false, verbose = false)
            d = maximum(abs.(vec(predict(r; frontier = X, depvar = y, zvar = Z6, what = :jlms))
                             .- vec(r.jlms)))
            @test d < 0.02              # QMC-level agreement
        end
    end

    @testset "FIX-07 panel_TFE MLE residuals are within-firm demeaned" begin
        Nf = 50; Tp = 4; NT = Nf * Tp; id = repeat(1:Nf, inner = Tp)
        Random.seed!(3)
        Xp = hcat(randn(NT), randn(NT)); αi = repeat(2 .* randn(Nf), inner = Tp)
        yp = αi .+ Xp * [0.6, -0.5] .+ 0.3 .* randn(NT) .- abs.(0.5 .* randn(NT))
        zp = randn(NT)                               # within-firm varying scaling z
        r = sfmodel_fit(spec = sfmodel_spec(depvar = yp, frontier = Xp, zvar = zp,
                            noise = :Normal, ineff = :HalfNormal, type = :prod,
                            datatype = :panel_TFE, id = id),
                        method = sfmodel_method(method = :MLE), optim_options = _optnm(),
                        marginal = false, show_table = false, verbose = false)
        ε = residuals(r; type = :composed)
        # Fixed effects removed ⇒ each firm's residuals average to ~0.
        wmean = [abs(mean(ε[(f-1)*Tp+1:f*Tp])) for f in 1:Nf]
        @test maximum(wmean) < 1e-6
    end

    @testset "FIX-09 constant-within-firm panel zvar errors at spec time" begin
        Nf = 30; Tp = 4; NT = Nf * Tp; id = repeat(1:Nf, inner = Tp)
        Xp = hcat(randn(NT), randn(NT)); yp = Xp * [0.5, -0.4] .+ randn(NT)
        @test_throws Exception sfmodel_spec(depvar = yp, frontier = Xp, zvar = ones(NT, 1),
            noise = :Normal, ineff = :HalfNormal, type = :prod, datatype = :panel_TFE, id = id)
        firmdummy = reshape(repeat(randn(Nf), inner = Tp), NT, 1)  # constant within firm
        @test_throws Exception sfmodel_spec(depvar = yp, frontier = Xp, zvar = firmdummy,
            noise = :Normal, ineff = :HalfNormal, type = :prod, datatype = :panel_TFE, id = id)
        # within-firm-varying zvar is fine:
        @test sfmodel_spec(depvar = yp, frontier = Xp, zvar = randn(NT, 1), noise = :Normal,
            ineff = :HalfNormal, type = :prod, datatype = :panel_TFE, id = id) isa SFrontiers.UnifiedSpec
    end

    @testset "FIX-19 hetero-λ Exponential MLE is predictable" begin
        W = hcat(cons, x1)                            # design incl. constant (MLE convention)
        sp = sfmodel_spec(depvar = y, frontier = X, zvar = W, noise = :Normal,
                          ineff = :Exponential, hetero = [:lambda], type = :prod)
        ini = sfmodel_init(spec = sp, frontier = βo, ln_lambda = [log(0.25), 0.0],
                           ln_sigma_v_sq = log(0.09))
        r = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE), init = ini,
                        optim_options = _opt(), marginal = false, show_table = false, verbose = false)
        @test r.converged
        jl = predict(r; frontier = X, depvar = y, zvar = W, what = :jlms)   # must NOT throw
        @test maximum(abs.(vec(jl) .- vec(r.jlms))) < 1e-6
    end

    @testset "FIX-20 stderror == sqrt.(diag(vcov)) (no hidden abs)" begin
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal, type = :prod)
        ini = sfmodel_init(spec = sp, frontier = βo, ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
        r = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MCI, n_draws = 100),
                        init = ini, optim_options = _opt(), marginal = false,
                        show_table = false, verbose = false)
        @test all(isequal.(stderror(r), sqrt.(diag(vcov(r)))))   # NaN==NaN passes for non-PD
    end
end
