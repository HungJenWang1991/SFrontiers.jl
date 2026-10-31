# Regression tests for
#   (a) the TruncatedNormal inverse CDF in the simulation backends, and
#   (b) the copula parameter ranges the manual overstated.
#
# (a) The quantile was written erfinv(clamp(r + (r-1)*erf(mu/(sigma*sqrt2)), -1+32eps, 1-32eps)).
#     As mu/sigma falls, erf(mu/(sigma*sqrt2)) -> -1 and the argument tends to 1 for EVERY draw:
#     r + (r-1)(-1+d) = 1 + (r-1)d, so the distance from 1 falls below the element type's
#     spacing there and every draw saturates the clamp.  The clamp then caps u at mu + 4.62*sigma
#     in Float32 and mu + 7.78*sigma in Float64; removing the clamp gives Inf instead.  Measured
#     against a 256-bit reference over mu/sigma in [-8, -0.5], the clamped form is wrong by up to
#     4.44 absolute in Float32 and 1.28 in Float64 -- so this was never a Float32-only defect.
#     The repair computes the same quantile in a form that never approaches 1:
#         u = mu + sigma*sqrt2*erfcinv( erfc(-mu/(sigma*sqrt2)) * (1-r) ),
#     accurate to 3.9e-5 (Float32) and 6.4e-15 (Float64), differentiable, and GPU-capable.
#
# (b) The copula table gave textbook parameter spaces; the estimator admits bounded ones.
#
# Only CSV and Test are test dependencies; log records are swallowed with
# Test.collect_test_logs rather than by loading Logging.

using SFrontiers
using Test
using Optim
using Random
using Printf
using SpecialFunctions
using StatsFuns
using Distributions

const _MCIB = SFrontiers.MCI_Backend

# 256-bit reference for the truncated-normal quantile, from the same identity
function _tn_ref(μ, σ, r)
    setprecision(BigFloat, 256) do
        bμ, bσ, br = BigFloat(μ), BigFloat(σ), BigFloat(r)
        s2 = sqrt(BigFloat(2))
        Float64(bμ + bσ*s2*erfcinv(erfc(-bμ/(bσ*s2)) * (1 - br)))
    end
end

@testset "TN quantile and parameter ranges" begin

    @testset "TruncatedNormal quantile matches a 256-bit reference" begin
        # Float64: the old clamped form was wrong by up to 1.283 absolute here.
        wa64 = 0.0
        for ms in (-0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -8.0),
            r in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999)
            got  = _MCIB.myTruncatedNormalQuantile(; μ = ms, σ = 1.0, r = r,
                        sqrt2 = sqrt(2.0), inv_sqrt2 = inv(sqrt(2.0)))
            want = _tn_ref(ms, 1.0, r)
            wa64 = max(wa64, abs(got - want))
        end
        @test wa64 < 1e-12

        # Float32: the old clamped form was wrong by up to 4.444 absolute here.
        wa32 = 0.0
        for ms in (-0.5f0, -1f0, -2f0, -3f0, -4f0, -5f0, -6f0, -8f0),
            r in (0.01f0, 0.1f0, 0.3f0, 0.5f0, 0.7f0, 0.9f0, 0.99f0, 0.9999f0)
            got  = _MCIB.myTruncatedNormalQuantile(; μ = ms, σ = 1f0, r = r,
                        sqrt2 = sqrt(2f0), inv_sqrt2 = inv(sqrt(2f0)))
            want = _tn_ref(Float64(ms), 1.0, Float64(r))
            wa32 = max(wa32, abs(Float64(got) - want))
        end
        @test wa32 < 1e-3

        # the quantile is no longer capped: at mu/sigma = -6 it must reach well past
        # mu + 4.62*sigma (the Float32 cap) and mu + 7.78*sigma (the Float64 cap)
        top = _MCIB.myTruncatedNormalQuantile(; μ = -6.0, σ = 1.0, r = 1 - 1e-12,
                    sqrt2 = sqrt(2.0), inv_sqrt2 = inv(sqrt(2.0)))
        @test top > -6.0 + 8.0            # beyond both old caps
        @test isapprox(top, _tn_ref(-6.0, 1.0, 1 - 1e-12); rtol = 1e-10)
    end

    @testset "support guard: the r = 0 quantile is 0, not -Inf" begin
        # The draw contract is [0, 1), so a draw of exactly 0 is legal.  F(0) = 0 for a
        # distribution truncated below at 0, so its 0th quantile is 0.  Unguarded, erfc
        # saturates at its ceiling of 2 for mu/sigma above about 8.5 (Float64) and the
        # expression returns -Inf.
        for T in (Float64, Float32), ms in (0, 4, 9, 20)
            q = _MCIB.myTruncatedNormalQuantile(; μ = T(ms), σ = one(T), r = zero(T),
                        sqrt2 = sqrt(T(2)), inv_sqrt2 = inv(sqrt(T(2))))
            @test isfinite(q)
            @test q == zero(T)
        end
        # and it does not touch an ordinary draw
        for T in (Float64, Float32), ms in (0, 4, 9), r in (T(1e-6), T(0.5), T(0.99))
            q = _MCIB.myTruncatedNormalQuantile(; μ = T(ms), σ = one(T), r = r,
                        sqrt2 = sqrt(T(2)), inv_sqrt2 = inv(sqrt(T(2))))
            @test q > 0
            # Float32 in the far lower tail keeps only a few digits: at mu/sigma = 4 and
            # r = 1e-6 the quantile is 0.00731 against a true 0.00736.  That is the element
            # type's own resolution, not the formula's.
            @test isapprox(Float64(q), _tn_ref(Float64(ms), 1.0, Float64(r));
                           rtol = T === Float64 ? 1e-10 : 2e-2)
        end
    end

    @testset "mu is identified again where every draw used to saturate" begin
        # With a true mu of -6 and sigma_u of 0.5 the clamped form returned mu_hat near -1.5
        # (Float32) and -1.8 (Float64), because past the cap the draws stop responding to mu.
        Random.seed!(424242)
        n  = 100
        xr = randn(n); vr = randn(n)
        σt = 0.5; μt = -6.0
        Random.seed!(9090)
        u = rand(truncated(Normal(μt, σt), 0, Inf), n)
        for TT in (Float64, Float32)
            X = TT.(hcat(ones(n), xr))
            y = TT.(hcat(ones(n), xr)*[1.0, 0.5] .+ 0.2 .* vr .- u)
            d = TT.(((1:32) .- 0.5) ./ 32)
            sp = sfmodel_spec(depvar = y, frontier = X, type = :production,
                              noise = :Normal, ineff = :TruncatedNormal)
            ini = sfmodel_init(spec = sp, frontier = TT[0.9, 0.45], mu = TT(μt),
                               ln_sigma_u_sq = TT(log(σt^2)), ln_sigma_v_sq = TT(log(0.04)))
            o = sfmodel_opt(warmstart_solver = nothing, main_solver = NelderMead(),
                            main_opt = (iterations = 40,))
            r = last(Test.collect_test_logs(() -> sfmodel_fit(spec = sp,
                    method = sfmodel_method(method = :MSLE, draws = d, n_draws = 32,
                                            multiRand = false),
                    init = ini, optim_options = o, jlms_bc_index = false, marginal = false,
                    show_table = false, verbose = false)))
            μ̂ = Float64(r.coeff[3])
            @test μ̂ < -5.0             # the clamped form returned about -1.5 / -1.8
            @test μ̂ > -12.0
        end
    end

    @testset "copula dependence parameters: the admitted range, not the textbook one" begin
        idx = (copula = (theta_rho = 1,),)
        for (ctor, lo, hi) in ((_MCIB.GaussianCopula_MCI(),  -0.999, 0.999),
                               (_MCIB.ClaytonCopula_MCI(),    1e-6,  50.0),
                               (_MCIB.Clayton90Copula_MCI(),  1e-6,  50.0),
                               (_MCIB.GumbelCopula_MCI(),     1.0,   50.0))
            c = _MCIB.make_copula_constants(ctor, Float64)
            got_lo = _MCIB.get_copula_vals(ctor, [-1e8], idx, c).rho
            got_hi = _MCIB.get_copula_vals(ctor, [ 1e8], idx, c).rho
            @test isapprox(got_lo, lo; rtol = 1e-12)
            @test isapprox(got_hi, hi; rtol = 1e-12)
            # nothing outside the interval is reachable
            ρs = [_MCIB.get_copula_vals(ctor, [θ], idx, c).rho
                  for θ in (-1e3, -50.0, -5.0, -1.0, 0.0, 1.0, 5.0, 50.0, 1e3)]
            @test all(ρ -> lo - 1e-12 <= ρ <= hi + 1e-12, ρs)
        end
    end
end
