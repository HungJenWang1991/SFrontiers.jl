# The panel likelihood converts and transforms its draws per group of firms: values, gradients and
# Hessians are unchanged, and memory falls with `chunks`. The reference functions are copies of the
# functions before the per-group rewrite (draw_memory_reference.jl).

using Test
using SFrontiers
using ForwardDiff
using Random

isdefined(@__MODULE__, :_ref_panel_nll) || include(joinpath(@__DIR__, "draw_memory_reference.jl"))

@testset "panel likelihood per group of firms" begin
    PB = SFrontiers.Panel_Backend
    _pcm_grad(f, p) = ForwardDiff.gradient(f, p, ForwardDiff.GradientConfig(nothing, p), Val{false}())
    _pcm_hess(f, p) = ForwardDiff.hessian(f, p, ForwardDiff.HessianConfig(nothing, p), Val{false}())

    @testset "in-place log-sum-exp equals logsumexp_rows_panel bit for bit" begin
        rng = Xoshiro(5801)
        duals = ForwardDiff.Dual{Nothing}.(randn(rng, 5, 300), randn(rng, 5, 300), randn(rng, 5, 300))
        for A in (randn(rng, 7, 300), randn(rng, 1, 300), randn(rng, 1, 10), duals)
            @test _same_bits(PB._logsumexp_rows_panel!(copy(A)), PB.logsumexp_rows_panel(A))
        end
    end

    @testset "value, gradient and Hessian equal the functions before the rewrite" begin
        d = _ref_panel_data(120, 2026)
        families = (:HalfNormal, :TruncatedNormal, :Exponential, :Weibull, :Lognormal, :Lomax, :Rayleigh, :Erlang2)
        cases = vcat([(:MSLE, f, nothing) for f in families],
                     [(:MCI, f, nothing) for f in (families..., :Gamma)],
                     [(:MCI, :HalfNormal, :expo_rule), (:MCI, :Weibull, :logistic_1_rule),
                      (:MCI, :Lomax, :logistic_2_rule)])
        for (meth, f, rule) in cases, mr in (true, false)
            isp = _ref_panel_spec(d, f; method = meth, D = 128, multiRand = mr, transformation = rule)
            θ = _ref_theta(d, f)
            @testset "$meth $f$(rule === nothing ? "" : " $rule") multiRand=$mr chunks=$k" for k in (1, 7, 10, 100, 120)
                fnew = p -> PB.panel_nll(isp, p; chunks = k)
                fold = p -> _ref_panel_nll(isp, p; chunks = k)
                v = fnew(θ)
                @test isequal(v, fold(θ))
                @test isfinite(v)
                @test isequal(_pcm_grad(fnew, θ), _pcm_grad(fold, θ))
                @test isequal(_pcm_hess(fnew, θ), _pcm_hess(fold, θ))
            end
        end
        for (meth, f) in ((:MCI, :HalfNormal), (:MSLE, :Erlang2)), mr in (true, false)
            isp = _ref_panel_spec(d, f; method = meth, D = 128, multiRand = mr, T = Float32)
            θ = _ref_theta(d, f; T = Float32)
            @testset "$meth $f Float32 data multiRand=$mr chunks=7" begin
                fnew = p -> PB.panel_nll(isp, p; chunks = 7)
                fold = p -> _ref_panel_nll(isp, p; chunks = 7)
                v = fnew(θ)
                @test isequal(v, fold(θ))
                @test isfinite(v)
                @test isequal(_pcm_grad(fnew, θ), _pcm_grad(fold, θ))
                @test isequal(_pcm_hess(fnew, θ), _pcm_hess(fold, θ))
            end
        end
    end

    @testset "bytes allocated fall with chunks (per-firm draws, $meth)" for meth in (:MCI, :MSLE)
        d = _ref_panel_data(400, 7)
        isp = _ref_panel_spec(d, :HalfNormal; method = meth, D = 1024, multiRand = true)
        θ = _ref_theta(d, :HalfNormal)
        a(k) = _alloc(() -> PB.panel_nll(isp, θ; chunks = k))
        aold = _alloc(() -> _ref_panel_nll(isp, θ; chunks = 10))
        a10 = a(10)
        @test a10 < 0.1 * aold
        @test a(20) < a10                  # falls with chunks; how much depends on Julia's allocator
        g(k) = _alloc(() -> _pcm_grad(p -> PB.panel_nll(isp, p; chunks = k), θ))
        gold = _alloc(() -> _pcm_grad(p -> _ref_panel_nll(isp, p; chunks = 10), θ))
        g10 = g(10)
        @test g10 < 0.1 * gold
        @test g(20) < g10
    end
end
