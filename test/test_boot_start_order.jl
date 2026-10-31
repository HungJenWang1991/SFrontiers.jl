# Regression tests: the MLE bootstrap starts from the estimate in the analytic order.
#
# MLE_Backend.sfmodel_boot_marginal started each replicate from `result.coeff`. The unified
# sfmodel_fit publishes `coeff` in the canonical order, which for the truncated normal with
# hetero = :scaling, and for panel_TFE with a truncated normal, puts the scaling block before
# μ, while LL_T reads μ first: every replicate started with the two swapped. Replicates that
# would have converged were redrawn or kept at an inferior optimum, so the bootstrap standard
# errors moved (by about 5% on the cross-sectional example below), and a degenerate panel fit
# could make the redraw loop run forever. The bootstrap now starts from `result.list[:coeff]`,
# the analytic backend's own order. Both testsets fail on the code before the fix.

using Test
using SFrontiers
using Random
using Optim

@testset "the MLE bootstrap starts from the estimate in the analytic order" begin
    quiet(f) = redirect_stdout(f, devnull)
    boot(res; R = 20) = quiet(() -> SFrontiers.MLE_Backend.sfmodel_boot_marginal(result = res, R = R,
                                                                                seed = 11, every = 1000))
    # The reference bootstrap: the same result with only the analytic-order vector as `coeff`
    # and no `list`, so the start cannot come from anywhere else.
    analytic_only(r) = merge(Base.structdiff(getfield(r, :nt), NamedTuple{(:list,)}),
                             (coeff = collect(r.list[:coeff]),))

    @testset "truncated normal with hetero = :scaling" begin
        rng = Xoshiro(7); n = 400
        x = randn(rng, n); X = hcat(ones(n), x); z = randn(rng, n)
        y = X * [1.0, 0.5] .+ 0.3 .* randn(rng, n) .- exp.(0.4 .* z) .* abs.(0.2 .+ 0.5 .* randn(rng, n))
        sp = sfmodel_spec(depvar = y, frontier = X, zvar = hcat(z), noise = :Normal,
                          ineff = :TruncatedNormal, hetero = :scaling)
        r = quiet(() -> sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE),
                                    optim_options = sfmodel_opt(main_solver = Newton(),
                                                                main_opt = (iterations = 200, g_abstol = 1e-6)),
                                    marginal = true, show_table = false, verbose = false))
        @test r.converged && r.redflag == 0
        cc = collect(r.coeff); cm = collect(r.list[:coeff])
        @test cc == cm[[1, 2, 4, 3, 5, 6]] && cc != cm     # published vs analytic order
        @test coef(r) == cc
        o1 = boot(r)
        o2 = boot(analytic_only(r))
        @test all(isfinite, o1[:, 1])
        @test isequal(o1, o2)
    end

    @testset "panel_TFE with a truncated normal" begin
        rng = Xoshiro(2021)
        nf, T = 80, 5; np_ = nf * T
        idp = repeat(1:nf, inner = T)
        XP = hcat(randn(rng, np_), randn(rng, np_))
        ZP = hcat(randn(rng, np_))
        tn = Float64[]
        while length(tn) < nf
            d = 0.2 + 0.5 * randn(rng); d > 0 && push!(tn, d)
        end
        yp = repeat(0.5 .* randn(rng, nf), inner = T) .+ XP * [0.6, -0.4] .+ 0.3 .* randn(rng, np_) .-
             exp.(0.3 .* ZP[:, 1]) .* repeat(tn, inner = T)
        spp = sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = XP, zvar = ZP, id = idp,
                           noise = :Normal, ineff = :TruncatedNormal)
        i0 = sfmodel_init(spec = spp, frontier = [0.6, -0.4], scaling = [0.3], mu = 0.2,
                          ln_sigma_u_sq = log(0.25), ln_sigma_v_sq = log(0.09))
        r = quiet(() -> sfmodel_fit(spec = spp, method = sfmodel_method(method = :MLE), init = i0,
                                    optim_options = sfmodel_opt(main_solver = Newton(),
                                                                main_opt = (iterations = 100,)),
                                    marginal = true, show_table = false, verbose = false))
        @test r.converged && r.redflag == 0
        cc = collect(r.coeff); cm = collect(r.list[:coeff])
        @test cc == cm[[1, 2, 4, 3, 5, 6]] && cc != cm
        o1 = boot(r; R = 10)
        o2 = boot(analytic_only(r); R = 10)
        @test isequal(o1, o2)
    end
end
