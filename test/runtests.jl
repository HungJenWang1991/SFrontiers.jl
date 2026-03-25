using SFrontiers
using Test
using Random
using Statistics
using DataFrames
using Optim

# Generate synthetic data for all tests
Random.seed!(12345)
N = 200

cons = ones(Float64, N)
x1 = randn(N)
x2 = randn(N)
X = hcat(cons, x1, x2)

z1 = randn(N)
Z = hcat(cons, z1)

beta_true = [1.0, 0.5, -0.3]
sigma_v = 0.3
sigma_u = 0.5

v = sigma_v .* randn(N)
u = abs.(sigma_u .* randn(N))
y = X * beta_true .+ v .- u

beta_ols = X \ y

df = DataFrame(yvar=y, cons=cons, x1=x1, x2=x2, z1=z1)

@testset "SFrontiers.jl" begin

    @testset "MLE: Normal + HalfNormal" begin
        spec = sfmodel_spec(
            depvar = y,
            frontier = X,
            noise = :Normal,
            ineff = :HalfNormal,
            type = :prod
        )
        @test spec isa SFrontiers.UnifiedSpec

        meth = sfmodel_method(method = :MLE)
        @test meth isa SFrontiers.MLEMethodSpec

        init = sfmodel_init(spec = spec,
            frontier = beta_ols,
            ln_sigma_sq = log(0.25),
            ln_sigma_v_sq = log(0.09))

        opt = sfmodel_opt(
            warmstart_solver = NelderMead(),
            warmstart_opt = (iterations = 50, g_abstol = 1e-3),
            main_solver = Newton(),
            main_opt = (iterations = 100, g_abstol = 1e-5, show_trace = false))

        res = sfmodel_fit(spec = spec, method = meth,
            init = init, optim_options = opt,
            marginal = false, show_table = false, verbose = false)

        @test res.converged == true
        @test length(res.coeff) == 5
        @test isfinite(res.loglikelihood)
    end

    @testset "MCI: Normal + HalfNormal" begin
        spec = sfmodel_spec(
            depvar = y,
            frontier = X,
            noise = :Normal,
            ineff = :HalfNormal,
            type = :prod
        )

        meth = sfmodel_method(method = :MCI, n_draws = 255)

        init = sfmodel_init(spec = spec,
            frontier = beta_ols,
            ln_sigma_sq = log(0.25),
            ln_sigma_v_sq = log(0.09))

        opt = sfmodel_opt(
            warmstart_solver = NelderMead(),
            warmstart_opt = (iterations = 50, g_abstol = 1e-3),
            main_solver = Newton(),
            main_opt = (iterations = 100, g_abstol = 1e-5, show_trace = false))

        res = sfmodel_fit(spec = spec, method = meth,
            init = init, optim_options = opt,
            marginal = false, show_table = false, verbose = false)

        @test res.converged == true
        @test length(res.coeff) == 5
        @test isfinite(res.loglikelihood)
    end

    @testset "MSLE: Normal + HalfNormal" begin
        spec = sfmodel_spec(
            depvar = y,
            frontier = X,
            noise = :Normal,
            ineff = :HalfNormal,
            type = :prod
        )

        meth = sfmodel_method(method = :MSLE, n_draws = 255)

        init = sfmodel_init(spec = spec,
            frontier = beta_ols,
            ln_sigma_sq = log(0.25),
            ln_sigma_v_sq = log(0.09))

        opt = sfmodel_opt(
            warmstart_solver = NelderMead(),
            warmstart_opt = (iterations = 50, g_abstol = 1e-3),
            main_solver = Newton(),
            main_opt = (iterations = 100, g_abstol = 1e-5, show_trace = false))

        res = sfmodel_fit(spec = spec, method = meth,
            init = init, optim_options = opt,
            marginal = false, show_table = false, verbose = false)

        @test res.converged == true
        @test length(res.coeff) == 5
        @test isfinite(res.loglikelihood)
    end

    @testset "DSL macros: DataFrame specification" begin
        spec = sfmodel_spec(
            @useData(df),
            @depvar(yvar),
            @frontier(cons, x1, x2),
            @zvar(cons, z1);
            noise = :Normal,
            ineff = :TruncatedNormal,
            hetero = [:mu],
            type = :prod
        )
        @test spec isa SFrontiers.UnifiedSpec

        meth = sfmodel_method(method = :MSLE, n_draws = 255)

        init = sfmodel_init(spec = spec,
            frontier = beta_ols,
            mu = [0.0, 0.0],
            ln_sigma_sq = log(0.25),
            ln_sigma_v_sq = log(0.09))

        opt = sfmodel_opt(
            warmstart_solver = NelderMead(),
            warmstart_opt = (iterations = 50, g_abstol = 1e-3),
            main_solver = Newton(),
            main_opt = (iterations = 100, g_abstol = 1e-5, show_trace = false))

        res = sfmodel_fit(spec = spec, method = meth,
            init = init, optim_options = opt,
            marginal = false, show_table = false, verbose = false)

        @test res.converged == true
        @test isfinite(res.loglikelihood)
    end

end
