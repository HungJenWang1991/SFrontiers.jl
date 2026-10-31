# Regression test: Student-t log normalizing constant must be stable for large ν.
#
# At ν ≈ 1e14 the naive loggamma((ν+1)/2) − loggamma(ν/2) difference loses all precision
# (float64 spacing at 2.5e15 is 0.5); on a sample of 271 observations it raised a fit's
# log-likelihood by 136, and the optimizer climbed the error, reporting
# ln_nu_minus_2 ≈ 32.7 as a "converged" fit.
# The t density must converge to the Normal density as ν → ∞, uniformly in ν; its kernel,
# written with log1p, keeps it accurate at every finite ν, and at ν = ∞ (ln_nu_minus_2
# above about 709.8) the likelihood is the Normal one.

using Test
using SFrontiers
using Distributions
using ForwardDiff

const MCIB = SFrontiers.MCI_Backend

@testset "StudentT log_const stability for large ν" begin
    σv = 0.18
    # _t_log_const(ν→∞) must approach the Normal constant −½log(2π) − log σ_v
    normal_const = -0.5 * log(2π) - log(σv)
    for ν in (1e5, 9.9e5, 1.1e6, 1e8, 1e10, 1e14, 1e16)
        lc = MCIB._t_log_const(ν, σv, Float64)
        @test isfinite(lc)
        @test abs(lc - (normal_const - 1 / (4ν))) < 2e-6   # both branches near the limit
    end
    # continuity across the branch switch at ν = 1e6
    below = MCIB._t_log_const(1e6 * (1 - 1e-9), σv, Float64)
    above = MCIB._t_log_const(1e6 * (1 + 1e-9), σv, Float64)
    # 1e-8: rounding in the naive branch alone reaches 1.7e-9 at ν just below the switch
    @test abs(below - above) < 1e-8
    # moderate ν: matches Distributions.jl location-scale t log-density at 0
    for ν in (2.1, 3.0, 7.0, 50.0, 1e4)
        lc = MCIB._t_log_const(ν, σv, Float64)
        @test isapprox(lc, logpdf(TDist(ν), 0.0) - log(σv); atol = 1e-10)
    end
end

@testset "StudentT log density at very large ν equals the Normal one" begin
    # The kernel (ν+1)/2 · log1p((z/σ_v)²/ν) keeps the log density within O(1/ν) of the
    # Normal log density; checked for both backends through the functions the fits call.
    msleB = SFrontiers.MSLE_Backend
    σv = 0.3
    zs = σv .* [0.5, 1.0, 2.0, 3.0]
    ref = -0.5 * log(2π) - log(σv) .- 0.5 .* (zs ./ σv) .^ 2
    for B in (MCIB, msleB)
        model = B._build_model(:StudentT, :HalfNormal)
        c = B.make_constants(model, Float64)
        idx = B._param_ind(model, 1, 1, Symbol[])
        for ν in (1e12, 1e14, 1e16, 1e18)
            p = zeros(max(idx.noise.ln_sigma_v_sq, idx.noise.ln_nu_minus_2))
            p[idx.noise.ln_sigma_v_sq] = log(σv^2)
            p[idx.noise.ln_nu_minus_2] = log(ν - 2)
            nv = B.get_noise_vals(model.noise, p, idx, c)
            out = similar(zs)
            B.log_noise_pdf!(model.noise, out, zs, nv, c)
            @test maximum(abs.(out .- ref)) < 1e-9
            pdf = similar(zs)
            B.myStudentTPDF!(pdf; z = zs, log_const = nv.log_const, inv_σ_v = nv.inv_sigma_v,
                             half_nu_plus_one = nv.half_nu_plus_one)
            @test maximum(abs.(log.(pdf) .- ref)) < 1e-9
            if B === MCIB
                outc = similar(zs)
                B.log_noise_pdf_chunk!(model.noise, outc, zs, nv, c, 1, length(zs))
                @test maximum(abs.(outc .- ref)) < 1e-9
            end
        end
    end
end

@testset "StudentT likelihood equals the Normal one at very large and infinite ν" begin
    # ln_nu_minus_2 = 40 (ν ≈ 2.4e17) and 710 (ν overflows to Inf); 200 observations,
    # σ_u/σ_v = 2.5, the same draws for both noise distributions.
    msleB = SFrontiers.MSLE_Backend
    n = 200
    xs = collect(range(0, 1; length = n))
    Xs = hcat(ones(n), xs)
    Zs = ones(n, 1)
    ys = 0.5 .+ 0.5 .* xs .- 0.4 .* abs.(sin.(13 .* xs)) .+ 0.1 .* cos.(29 .* xs)
    draws = reshape(MCIB.make_halton_p(4096), 1, 4096)
    pN = [0.5, 0.5, log(0.25), log(0.04)]
    for nllfun in (MCIB.MCI_nll, msleB.qmc_nll), chunks in (1, 4)
        f = (p, noise) -> nllfun(ys, Xs, Zs, p, draws; noise = noise, ineff = :HalfNormal, chunks = chunks)
        nllN = f(pN, :Normal)
        for θ in (40.0, 710.0)
            @test abs(f(vcat(pN, θ), :StudentT) - nllN) < 1e-6
        end
        # at ν = ∞ the derivatives are those of the Normal likelihood, and zero in ln_nu_minus_2
        gT = ForwardDiff.gradient(p -> f(p, :StudentT), vcat(pN, 710.0))
        gN = ForwardDiff.gradient(p -> f(p, :Normal), pN)
        @test all(isfinite, gT)
        @test gT[5] == 0
        @test isapprox(gT[1:4], gN; rtol = 1e-8)
        @test all(isfinite, ForwardDiff.hessian(p -> f(p, :StudentT), vcat(pN, 710.0)))
    end
end
