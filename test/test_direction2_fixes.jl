# Regression tests for the prod/cost sign fixes in 00Direction2_findings.md
# (D2-01 MCI, D2-02 MSLE) and the shared sign contract.
#
# The bug: the exported matrix-form `jlms_bc_indices(Y,X,Z,p,draws; ...)` defaults
# `type=:prod`. A cost model whose indices are recomputed through this API without
# `type=:cost` silently conditions on `z = ε + u` (production) instead of `z = ε − u`
# (cost) — inverting the efficiency ranking while every value stays in a plausible
# range. These tests would fail if `type` stopped being honored (a future regression
# of the default), and document the correct usage. Self-contained (own data/seed).

using SFrontiers
using Test
using Random
using Statistics
using Optim

const MCIB = SFrontiers.MCI_Backend
const MSB  = SFrontiers.MSLE_Backend

_o_d2() = sfmodel_opt(
    warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40, g_abstol = 1e-3),
    main_solver = Newton(), main_opt = (iterations = 80, g_abstol = 1e-5, show_trace = false))

# Matrix-form arguments pulled from a fitted internal spec (`r.model`), so the
# reconstructed `idx`/`L` match the fit exactly. Returns (Y, X, Z, p, draws, kw...).
_asmat(Z) = Z isa AbstractMatrix ? Z : reshape(Z, :, 1)

@testset "00Direction2 prod/cost sign fixes" begin

    Random.seed!(2024)
    N = 200
    cons = ones(N); x1 = randn(N); x2 = randn(N)
    X = hcat(cons, x1, x2)
    βt = [1.0, 0.5, -0.3]
    # Cost frontier: y = Xβ + v + u  ⇒  composed residual ε = v + u  (positively skewed).
    # Firms with larger ε are MORE inefficient and must get HIGHER jlms.
    yc = X * βt .+ 0.3 .* randn(N) .+ abs.(0.6 .* randn(N))
    βo = X \ yc

    @testset "D2-01 MCI matrix jlms_bc_indices honors type=:cost" begin
        sp  = sfmodel_spec(depvar = yc, frontier = X, noise = :Normal,
                           ineff = :HalfNormal, type = :cost)
        ini = sfmodel_init(spec = sp, frontier = βo,
                           ln_sigma_sq = log(0.36), ln_sigma_v_sq = log(0.09))
        r = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MCI, n_draws = 200),
                        init = ini, optim_options = _o_d2(),
                        marginal = false, show_table = false, verbose = false)
        @test r.converged
        m  = r.model
        Zm = _asmat(m.zvar)
        ε  = m.depvar .- m.frontier * r.coeff[1:m.K]        # = v + u for the cost fit

        # Correct: pass type=:cost.  Bug: default type=:prod.
        cost = MCIB.jlms_bc_indices(m.depvar, m.frontier, Zm, r.coeff, m.draws_2D;
                    noise = m.noise, ineff = m.ineff, hetero = m.hetero, type = :cost)
        prod = MCIB.jlms_bc_indices(m.depvar, m.frontier, Zm, r.coeff, m.draws_2D;
                    noise = m.noise, ineff = m.ineff, hetero = m.hetero)   # default :prod

        # type=:cost ranks inefficiency correctly (jlms ↑ with ε); default :prod inverts it.
        @test cor(vec(cost.jlms), ε) > 0.5
        @test cor(vec(prod.jlms), ε) < 0.0
        # type=:cost reproduces the fit's own ranking; the two calls genuinely differ.
        @test cor(vec(cost.jlms), vec(r.jlms)) > 0.9
        @test maximum(abs.(vec(cost.jlms) .- vec(prod.jlms))) > 1e-3
        # The wrong (default) result still looks plausible — why the bug is silent.
        @test all(0 .<= prod.bc .<= 1)
    end

    @testset "D2-02 MSLE matrix jlms_bc_indices honors type=:cost" begin
        sp  = sfmodel_spec(depvar = yc, frontier = X, noise = :Normal,
                           ineff = :HalfNormal, type = :cost)
        ini = sfmodel_init(spec = sp, frontier = βo,
                           ln_sigma_sq = log(0.36), ln_sigma_v_sq = log(0.09))
        r = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MSLE, n_draws = 200),
                        init = ini, optim_options = _o_d2(),
                        marginal = false, show_table = false, verbose = false)
        @test r.converged
        m  = r.model
        Zm = _asmat(m.zvar)
        ε  = m.depvar .- m.frontier * r.coeff[1:m.K]

        cost = MSB.jlms_bc_indices(m.depvar, m.frontier, Zm, r.coeff, m.draws_2D;
                    noise = m.noise, ineff = m.ineff, hetero = m.hetero, type = :cost)
        prod = MSB.jlms_bc_indices(m.depvar, m.frontier, Zm, r.coeff, m.draws_2D;
                    noise = m.noise, ineff = m.ineff, hetero = m.hetero)   # default :prod

        # MSLE matrix + spec share the raw-quantile path ⇒ type=:cost is near bit-identical
        # to the fit's own jlms; the default :prod inverts the ranking.
        @test maximum(abs.(vec(cost.jlms) .- vec(r.jlms))) < 1e-6
        @test cor(vec(prod.jlms), ε) < 0.0
        @test maximum(abs.(vec(cost.jlms) .- vec(prod.jlms))) > 1e-3
        @test all(0 .<= prod.bc .<= 1)
    end

    @testset "prod/cost mirror (spec-form sign contract)" begin
        # Cost: jlms rises with the composed residual ε = v + u.
        sp_c  = sfmodel_spec(depvar = yc, frontier = X, noise = :Normal,
                             ineff = :HalfNormal, type = :cost)
        ini_c = sfmodel_init(spec = sp_c, frontier = βo,
                             ln_sigma_sq = log(0.36), ln_sigma_v_sq = log(0.09))
        r_c = sfmodel_fit(spec = sp_c, method = sfmodel_method(method = :MCI, n_draws = 200),
                          init = ini_c, optim_options = _o_d2(),
                          marginal = false, show_table = false, verbose = false)
        ε_c = yc .- X * r_c.coeff[1:3]
        @test cor(vec(r_c.jlms), ε_c) > 0.5

        # Production mirror: y = Xβ + v − u ⇒ jlms rises as the residual falls.
        yp  = X * βt .+ 0.3 .* randn(N) .- abs.(0.6 .* randn(N))
        βp  = X \ yp
        sp_p  = sfmodel_spec(depvar = yp, frontier = X, noise = :Normal,
                             ineff = :HalfNormal, type = :prod)
        ini_p = sfmodel_init(spec = sp_p, frontier = βp,
                             ln_sigma_sq = log(0.36), ln_sigma_v_sq = log(0.09))
        r_p = sfmodel_fit(spec = sp_p, method = sfmodel_method(method = :MCI, n_draws = 200),
                          init = ini_p, optim_options = _o_d2(),
                          marginal = false, show_table = false, verbose = false)
        ε_p = yp .- X * r_p.coeff[1:3]
        @test cor(vec(r_p.jlms), ε_p) < -0.5
    end
end
