# The default number of Halton draws per observation (per firm in panel models) of `method = :MCI` and
# `:MSLE`: 4,096 (2^12) at every method constructor, an explicit `n_draws` kept as given, and fits that rely
# on the default, cross-sectional and panel, run with 4,096 draws.

using Test, SFrontiers, Random
using Optim: Newton

@testset "the default number of draws" begin

    @testset "the default is 4096 at every constructor" begin
        @test sfmodel_method(method = :MCI).n_draws == 4096
        @test sfmodel_method(method = :MSLE).n_draws == 4096
        @test SFrontiers.MCI_Backend.sfmodel_method().n_draws == 4096
        @test SFrontiers.MSLE_Backend.sfmodel_method().n_draws == 4096
        @test SFrontiers.Panel_Backend.sfmodel_panel_method().n_draws == 4096
        @test sfmodel_panel_method(method = :MCI).n_draws == 4096
        @test sfmodel_panel_method(method = :MSLE).n_draws == 4096
        @test SFrontiers._to_panel_method(sfmodel_method(method = :MCI)).n_draws == 4096
        @test SFrontiers._to_panel_method(sfmodel_method(method = :MSLE)).n_draws == 4096
    end

    @testset "an explicit n_draws is kept" begin
        @test sfmodel_method(method = :MCI, n_draws = 1024).n_draws == 1024
        @test sfmodel_method(method = :MSLE, n_draws = 1024).n_draws == 1024
        @test sfmodel_panel_method(method = :MSLE, n_draws = 1024).n_draws == 1024
        @test SFrontiers._to_panel_method(sfmodel_method(method = :MCI, n_draws = 2048)).n_draws == 2048
    end

    @testset "fits that rely on the default use 4096 draws" begin
        # Fits held at their starting values (no optimizer iteration), without n_draws.
        # Cross-section: 16 observations, normal–half-normal.
        rng = Xoshiro(62)
        X = hcat(ones(16), randn(rng, 16))
        y = X * [1.0, 0.5] .+ 0.3 .* randn(rng, 16) .- abs.(0.4 .* randn(rng, 16))
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal)
        ini = sfmodel_init(spec = sp, frontier = X \ y, ln_sigma_u_sq = log(0.16), ln_sigma_v_sq = log(0.09),
                           message = false)
        held = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,))
        for m in (:MCI, :MSLE)
            r = sfmodel_fit(spec = sp, method = sfmodel_method(method = m), init = ini, optim_options = held,
                            marginal = false, show_table = false, verbose = false)
            @test r.n_draws == 4096
            @test size(r.model.draws_2D) == (16, 4096)
            @test r.draw_scheme == :halton
            @test r.distinct_Halton_length == 16 * 4096
        end

        # Panel (panel_TFE): 12 firms observed over 4 periods, normal–half-normal.
        rng = Xoshiro(44)
        Nf, Tp = 12, 4
        NT = Nf * Tp
        idp = repeat(1:Nf, inner=Tp)
        Xp = hcat(randn(rng, NT), randn(rng, NT))
        zp = randn(rng, NT)
        yp = repeat(randn(rng, Nf), inner=Tp) .+ Xp * [0.6, -0.4] .+ 0.3 .* randn(rng, NT) .-
             exp.(0.3 .* zp) .* repeat(abs.(0.4 .* randn(rng, Nf)), inner=Tp)
        spp = sfmodel_spec(depvar=yp, frontier=Xp, zvar=zp, id=idp, noise=:Normal, ineff=:HalfNormal,
                           type=:prod, datatype=:panel_TFE)
        r = sfmodel_fit(spec = spp, method = sfmodel_method(method = :MCI),
                        init = sfmodel_init(spec = spp, ln_sigma_u_sq = log(0.16), ln_sigma_v_sq = log(0.09)),
                        optim_options = held, marginal = false, show_table = false, verbose = false)
        @test r.n_draws == 4096
        @test size(r.model.draws) == (12, 4096)
    end
end
