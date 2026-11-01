# Regression tests: copula clamps follow the precision the probabilities are computed in.
#
# The Gaussian and Gumbel copula log-densities of MCI and MSLE (copula_log_adjustment!) clamped the noise CDF
# F_v and the inefficiency CDF F_u with the constants of the specification's element type. A fit's coefficient
# vector is Float64, so with Float32 data F_v (and MCI's F_u) are computed in Float64 but were clamped with the
# Float32 constants: the Gaussian copula's normal scores were capped at ±4.62 instead of ±7.78 and the Gumbel
# copula's F at 1 - 6.0e-8 instead of 1 - 1.1e-16; under MSLE the Gaussian copula also capped the scores of the
# Float32 draws at ±4.62, which long Halton sequences (2^20 points or more) and user-supplied draws reach. On 300
# observations the Float32 log-likelihood was 1.34 nats above the Float64 one at ρ = 0.46 (Gaussian); the Gumbel gap
# reached 4.05e-3 (MCI). The clamps now come from the wider of the type each probability is computed in and the type
# of the specification's constants (`_copula_clamps` in MCI_Backend and MSLE_Backend); MSLE's draws keep their own
# type, with those clamps rounded to it (`_gaussian_draw_clamps`). The Gumbel testset compares with 512-bit values
# at the exact F_v (the Gumbel copula's noise side is computed in logarithms, with no clamp). For the Gaussian
# copula, with Float64 data at any coefficients and with Float32 data at Float32 coefficients, nothing changes in
# any digit; only this Gaussian arm is compared bit for bit with the earlier code (an in-test copy of its old
# expressions). These tests fail on the code before the fix.

using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions
using Random

# 512-bit normal quantile Φ⁻¹(F) (Newton on erfc) and the two copula log-densities
function _b42_qB(F::BigFloat)
    q = BigFloat(sqrt(2) * erfinv(2 * Float64(F) - 1))
    for _ in 1:30
        dq = (erfc(-q / sqrt(BigFloat(2))) / 2 - F) / (exp(-q^2 / 2) / sqrt(2 * BigFloat(pi)))
        q -= dq
        abs(dq) < BigFloat(10)^-140 && break
    end
    return q
end
_b42_gaussB(qv, qu, ρ) = -log(1 - ρ^2) / 2 - (ρ^2 * (qv^2 + qu^2) - 2ρ * qv * qu) / (2 * (1 - ρ^2))
function _b42_gumbelB(u1, u2, ρ)
    w1 = -log(u1); w2 = -log(u2); S = w1^ρ + w2^ρ
    return -S^(1 / ρ) + (ρ - 1) * (log(w1) + log(w2)) + log((ρ - 1) * S^(-2 + 1 / ρ) + S^(-2 + 2 / ρ)) - log(u1) - log(u2)
end

# in-test copy of the Gaussian log density as the earlier code computed it, with the clamps of F_v (cv) and of the
# draws / F_u (cu) as separate arguments: the old code is (c, c)
function _b42_gauss!(adj, Fv, draws, rho, s2, cv, cu)
    rho_sq = rho * rho; omr2 = 1 - rho_sq
    @. adj = -0.5 * log(omr2) - (
        rho_sq * (
            (s2 * erfinv(clamp(2 * clamp(Fv, cv.copula_clamp_lo, cv.copula_clamp_hi) - 1, cv.lo_erfinv, cv.hi_erfinv)))^2 +
            (s2 * erfinv(clamp(2 * clamp(draws, cu.copula_clamp_lo, cu.copula_clamp_hi) - 1, cu.lo_erfinv, cu.hi_erfinv)))^2
        ) - 2 * rho *
            (s2 * erfinv(clamp(2 * clamp(Fv, cv.copula_clamp_lo, cv.copula_clamp_hi) - 1, cv.lo_erfinv, cv.hi_erfinv))) *
            (s2 * erfinv(clamp(2 * clamp(draws, cu.copula_clamp_lo, cu.copula_clamp_hi) - 1, cu.lo_erfinv, cu.hi_erfinv)))
    ) / (2 * omr2)
    return adj
end

@testset "copula clamps follow the precision the probabilities are computed in" begin
    setprecision(BigFloat, 512) do
    MB = SFrontiers.MCI_Backend; MS = SFrontiers.MSLE_Backend
    D64 = ForwardDiff.Dual{Nothing, Float64, 2}; D32 = ForwardDiff.Dual{Nothing, Float32, 2}
    BK = ((MB, "MCI", MB.GaussianCopula_MCI(), MB.GumbelCopula_MCI(), MB.NormalNoise_MCI(), MB.MCIModel, MB.HalfNormal_MCI()),
          (MS, "MSLE", MS.GaussianCopula_MSLE(), MS.GumbelCopula_MSLE(), MS.NormalNoise_MSLE(), MS.MSLEModel, MS.HalfNormal_MSLE()))
    consts(B, model, cop, T) = B.make_constants(model(B === MB ? MB.NormalNoise_MCI() : MS.NormalNoise_MSLE(),
                                                      B === MB ? MB.HalfNormal_MCI() : MS.HalfNormal_MSLE(), cop), T)
    h32 = prevfloat(1f0)

    @testset "the clamps are those of the wider of the arithmetic type and the constants' type" begin
        # (element type of the probability array, type of the specification's constants, expected constants' type);
        # a Dual counts as its value type
        for (B, _, gau, gum) in BK, cop in (gau, gum)
            for (E, Tc, R) in ((Float64, Float64, Float64), (Float32, Float32, Float32), (Float64, Float32, Float64),
                               (Float32, Float64, Float64), (D64, Float32, Float64), (ForwardDiff.Dual{Nothing, D64, 2}, Float32, Float64),
                               (D32, Float32, Float32), (D32, Float64, Float64))
                @test isdefined(B, :_copula_clamps) &&
                      B._copula_clamps(cop, zeros(E, 2, 2), B.make_copula_constants(cop, Tc)) === B.make_copula_constants(cop, R)
            end
        end
    end

    @testset "MSLE's draws: the clamps rounded to the draws' own type and kept inside (-1, 1)" begin
        g64 = MS.make_copula_constants(MS.GaussianCopula_MSLE(), Float64); g32 = MS.make_copula_constants(MS.GaussianCopula_MSLE(), Float32)
        four(g) = (copula_clamp_lo = g.copula_clamp_lo, copula_clamp_hi = g.copula_clamp_hi, lo_erfinv = g.lo_erfinv, hi_erfinv = g.hi_erfinv)
        has = isdefined(MS, :_gaussian_draw_clamps)
        # Float32 draws in Float64 arithmetic (a fit on Float32 data): the inverse-erf bounds become ±prevfloat(1f0)
        @test has && MS._gaussian_draw_clamps(g64, zeros(Float32, 2, 2)) ===
                     (copula_clamp_lo = g32.copula_clamp_lo, copula_clamp_hi = h32, lo_erfinv = -h32, hi_erfinv = h32)
        # draws of the arithmetic's own type: the clamps themselves
        @test has && MS._gaussian_draw_clamps(g32, zeros(Float32, 2, 2)) === four(g32)
        @test has && MS._gaussian_draw_clamps(g64, zeros(Float64, 2, 2)) === four(g64)
    end

    # composite errors z (σ_v = 1) and F_u values where the old Float32 clamps bound (|score| from 4.62 to 7.3) and
    # where they did not. Below F = 1e-9 the Float64 form 2F - 1 itself loses digits (in either precision), so the
    # lower tail stops there; F near 1 is exact in 2F - 1.
    zs = [-6.0, -5.6, -5.2, -4.8, -2.0, 0.3, 4.8, 5.2, 5.6, 6.4, 7.3]
    Fus = [1e-9, 1e-7, 1e-5, 0.3, 0.8, 1 - 1e-5, 1 - 1e-7, 1 - 1e-9, 1 - 1e-12]
    ρ = 0.999 * tanh(0.5)

    @testset "Gaussian normal scores of a Float32 specification under Float64 arithmetic, against 512 bits" begin
        for (B, bname, gau, _, nz, model) in BK
            c32 = consts(B, model, gau, Float32)
            zb = repeat(zs, 1, length(Fus)); Fu = repeat(Fus', length(zs), 1)
            dr = bname == "MCI" ? Fu : fill(0.3, size(Fu))           # MSLE: its draws are tested below
            adj = zeros(size(zb)); Fv = zeros(size(zb))
            B.copula_log_adjustment!(gau, adj, Fv, zb, dr, nz, (inv_sigma_v = 1.0,), (rho = ρ,), c32)
            # reference: the exact normal scores of the package's own F_v (noise_cdf!, unchanged by the fix) and of
            # F_u; the specification's Float32 √2 shifts each score by 1.7e-8 relative (1.3e-6 nats measured), so
            # the tolerance is 1e-5; the old cap at ±4.62 was off by up to 5 nats here
            Fv0 = zeros(size(zb)); B.noise_cdf!(nz, Fv0, zb, (inv_sigma_v = 1.0,), c32)
            err = 0.0
            for i in eachindex(adj)
                qv = _b42_qB(BigFloat(Fv0[i])); qu = _b42_qB(BigFloat(dr[i]))
                err = max(err, Float64(abs(adj[i] - _b42_gaussB(qv, qu, BigFloat(ρ)))))
            end
            @test err <= 1e-5
            # ForwardDiff gradient in (shift of z, θ_ρ) where the old Float32 clamps bound and F_v is still accurate
            # in Float64 (|z| ≤ 5.6): against 512-bit central differences of the exact density
            z2 = [-5.6, -5.2, -4.8, 4.8, 5.2, 5.6]; Fu2 = bname == "MCI" ? [1e-7, 1e-5, 0.3, 1 - 1e-5, 1 - 1e-7] : [0.3]
            ZZ = repeat(z2, 1, length(Fu2)); UU = repeat(Fu2', length(z2), 1)
            function f(θ)
                E = eltype(θ); a = zeros(E, size(ZZ)); v = zeros(E, size(ZZ))
                B.copula_log_adjustment!(gau, a, v, ZZ .+ θ[1], UU, nz, (inv_sigma_v = one(E),),
                                         (rho = Float64(c32.rho_max) * tanh(θ[2]),), c32)
                return sum(a)
            end
            qus = [_b42_qB(BigFloat(u)) for u in UU]
            fB(θ) = sum(_b42_gaussB(BigFloat(ZZ[i]) + θ[1], qus[i], BigFloat(Float64(c32.rho_max)) * tanh(θ[2])) for i in eachindex(ZZ))
            h = BigFloat(10)^-40; θ0 = [0.0, 0.5]
            GR = [(fB(BigFloat.(θ0) .+ e) - fB(BigFloat.(θ0) .- e)) / (2h) for e in ([h, 0], [0, h])]
            G = ForwardDiff.gradient(f, θ0)
            @test maximum(abs.(G .- GR)) <= 1e-6 * maximum(abs.(GR))
        end
    end

    @testset "MSLE: Float32 draws at the Halton tails under Float64 arithmetic, against 512 bits" begin
        model = MS.MSLEModel(MS.NormalNoise_MSLE(), MS.HalfNormal_MSLE(), MS.GaussianCopula_MSLE())
        c32 = MS.make_constants(model, Float32); nz = MS.NormalNoise_MSLE()
        # base-2 Halton points of sequences of 2^18 to 2^24 points (exact in Float32); the old bounds ±(1 - 2^-18)
        # capped the scores of the outer ones at ±4.62 (5.29 at 2^-24), off by up to 3.1 nats here
        tails = Float32[2f0^-24, 2f0^-22, 2f0^-20, 2f0^-18, 0.3f0, 1 - 2f0^-18, 1 - 2f0^-20, 1 - 2f0^-22, 1 - 2f0^-24]
        ZZ = repeat([-2.0, 0.3, 2.5, 4.8, 5.6], 1, length(tails)); UU = repeat(tails', 5, 1)
        adj = zeros(size(ZZ)); Fv = zeros(size(ZZ))
        MS.copula_log_adjustment!(MS.GaussianCopula_MSLE(), adj, Fv, ZZ, UU, nz, (inv_sigma_v = 1.0,), (rho = ρ,), c32)
        Fv0 = zeros(size(ZZ)); MS.noise_cdf!(nz, Fv0, ZZ, (inv_sigma_v = 1.0,), c32)
        qus = [_b42_qB(BigFloat(u)) for u in UU]
        err = maximum(Float64(abs(adj[i] - _b42_gaussB(_b42_qB(BigFloat(Fv0[i])), qus[i], BigFloat(ρ)))) for i in eachindex(adj))
        @test err <= 2e-5          # the draws' scores are Float32 (erfinv in single precision): 4.3e-6 measured
        function f(θ)
            E = eltype(θ); a = zeros(E, size(ZZ)); v = zeros(E, size(ZZ))
            MS.copula_log_adjustment!(MS.GaussianCopula_MSLE(), a, v, ZZ .+ θ[1], UU, nz, (inv_sigma_v = one(E),),
                                      (rho = Float64(c32.rho_max) * tanh(θ[2]),), c32)
            return sum(a)
        end
        fB(θ) = sum(_b42_gaussB(BigFloat(ZZ[i]) + θ[1], qus[i], BigFloat(Float64(c32.rho_max)) * tanh(θ[2])) for i in eachindex(ZZ))
        h = BigFloat(10)^-40; θ0 = [0.0, 0.5]
        GR = [(fB(BigFloat.(θ0) .+ e) - fB(BigFloat.(θ0) .- e)) / (2h) for e in ([h, 0], [0, h])]
        G = ForwardDiff.gradient(f, θ0)
        @test maximum(abs.(G .- GR)) <= 1e-6 * maximum(abs.(GR))     # 1.2e-7 measured; the old bounds 7.9e-2
    end

    @testset "Gumbel: F near 1 under Float64 arithmetic with the Float32 constants, against 512 bits" begin
        for (B, bname, _, gum, nz, model) in BK
            c32 = consts(B, model, gum, Float32)
            rg = 1 + exp(0.25)
            if bname == "MCI"     # F_u near 1 (the old upper clamp prevfloat(1f0) = 1 - 6.0e-8 bound)
                zb = fill(0.3, 1, 5); Fu = [1 - 1e-12 1 - 1e-9 1 - 1e-7 0.5 1e-9]
            else                  # MSLE: F_v near 1 (z from 5.6 to 6.9); the draws keep the Float32 clamps
                zb = [5.6 5.9 6.4 6.9 0.3]; Fu = fill(0.4, 1, 5)
            end
            adj = zeros(size(zb)); Fv = zeros(size(zb))
            B.copula_log_adjustment!(gum, adj, Fv, zb, Fu, nz, (inv_sigma_v = 1.0,), (rho = rg,), c32)
            # reference: the exact F_v, computed in 512 bits from z, and F_u, unclamped, in 512 bits
            err = 0.0
            for i in eachindex(adj)
                FvB = erfc(-BigFloat(zb[i]) / sqrt(BigFloat(2))) / 2
                err = max(err, Float64(abs(adj[i] - _b42_gumbelB(FvB, BigFloat(Fu[i]), BigFloat(rg)))))
            end
            @test err <= 1e-9                         # the old upper clamp 1 - 6.0e-8 was off by nats here
        end
    end

    @testset "Float64 specifications and Float32 arithmetic: the old expressions, bit for bit" begin
        rng = Xoshiro(42)
        zb = 3 .* randn(rng, 7, 9); zb[1, :] .= [-40, -9, -8.3, -6, 0, 6, 8.3, 9, 40]
        Fu = rand(rng, 7, 9); Fu[2, :] .= [0, 1e-300, 1e-16, 3e-15, 1e-9, 1 - 1e-9, 1 - 1e-15, prevfloat(1.0), 1]
        for (B, bname, gau, gum, nz, model) in BK
            c64 = consts(B, model, gau, Float64)
            c32 = consts(B, model, gau, Float32)
            for (cop, c, ref!) in ((gau, c64, (a, v, d, r) -> _b42_gauss!(a, v, d, r, c64.sqrt2, c64, c64)),)
                rr = ρ
                # values through z, ρ and (MCI) F_u
                function newf(θ)
                    E = eltype(θ); a = zeros(E, size(zb)); v = zeros(E, size(zb))
                    d = bname == "MCI" ? Fu .^ exp(θ[3]) .+ zero(E) : Fu
                    B.copula_log_adjustment!(cop, a, v, zb .+ θ[1], d, nz, (inv_sigma_v = one(E),), (rho = rr * exp(θ[2] / 50),), c)
                    return a
                end
                function oldf(θ)
                    E = eltype(θ); a = zeros(E, size(zb)); v = zeros(E, size(zb))
                    d = bname == "MCI" ? Fu .^ exp(θ[3]) .+ zero(E) : Fu
                    B.noise_cdf!(nz, v, zb .+ θ[1], (inv_sigma_v = one(E),), c)
                    return ref!(a, v, d, rr * exp(θ[2] / 50))
                end
                θ0 = [0.01, 0.02, 0.03]
                @test isequal(newf(θ0), oldf(θ0))
                # (Dual and nested-Dual elements, i.e. gradients and Hessians, are not compared with the
                # old code here, to keep this test short)
            end
            # Float32 arithmetic (Float32 coefficients) with the Float32 constants: the old expressions
            z32 = Float32.(zb); u32 = Float32.(min.(Fu, prevfloat(1.0)))
            for (cop, c) in ((gau, c32),)
                a = zeros(Float32, size(z32)); v = zeros(Float32, size(z32)); a0 = zeros(Float32, size(z32)); v0 = zeros(Float32, size(z32))
                rr = Float32(ρ)
                B.copula_log_adjustment!(cop, a, v, z32, u32, nz, (inv_sigma_v = 1f0,), (rho = rr,), c)
                B.noise_cdf!(nz, v0, z32, (inv_sigma_v = 1f0,), c)
                _b42_gauss!(a0, v0, u32, rr, c.sqrt2, c, c)
                @test eltype(a) === Float32 && isequal(a, a0)
            end
            # Float32 arithmetic with the Float64 constants and Float64 draws (a Float64 specification evaluated at
            # Float32 coefficients, which no fit does): the old expressions with the Float64 constants
            for (cop, c) in ((gau, c64),)
                a = zeros(Float32, size(z32)); v = zeros(Float32, size(z32)); a0 = zeros(Float32, size(z32)); v0 = zeros(Float32, size(z32))
                rr = Float32(ρ)
                B.copula_log_adjustment!(cop, a, v, z32, Fu, nz, (inv_sigma_v = 1f0,), (rho = rr,), c)
                B.noise_cdf!(nz, v0, z32, (inv_sigma_v = 1f0,), c)
                _b42_gauss!(a0, v0, Fu, rr, c.sqrt2, c, c)
                @test eltype(a) === Float32 && isequal(a, a0)
            end
            # Float64 arithmetic with the Float32 constants (a fit on Float32 data): the clamps of F_v are the Float64
            # ones; F_u's too under MCI; MSLE's Float32 draws get the Float32 clamps with the inverse-erf bounds
            # ±prevfloat(1f0) (Gaussian)
            for (cop, c, c6) in ((gau, c32, c64),)
                d = bname == "MCI" ? Fu : Float32.(min.(Fu, prevfloat(1.0)))
                cu = bname == "MCI" ? c6 :
                     (copula_clamp_lo = c.copula_clamp_lo, copula_clamp_hi = c.copula_clamp_hi, lo_erfinv = -h32, hi_erfinv = h32)
                a = zeros(size(zb)); v = zeros(size(zb)); a0 = zeros(size(zb)); v0 = zeros(size(zb))
                rr = ρ
                B.copula_log_adjustment!(cop, a, v, zb, d, nz, (inv_sigma_v = 1.0,), (rho = rr,), c)
                B.noise_cdf!(nz, v0, zb, (inv_sigma_v = 1.0,), c)
                _b42_gauss!(a0, v0, d, rr, c.sqrt2, c6, cu)
                @test isequal(a, a0)
            end
        end
    end

    @testset "log-likelihoods: Float32 data against Float64 data at Float64 coefficients" begin
        function tnpos(rng, mu, s, n)
            out = Float64[]
            while length(out) < n
                d = mu + s * randn(rng); d > 0 && push!(out, d)
            end
            return out
        end
        rng = Xoshiro(77); n = 300                       # the fixed 300-observation sample of test_copula_tn_cdf.jl
        x = randn(rng, n); X = hcat(ones(n), x)
        v = 0.25 .* randn(rng, n)
        y = X * [1.0, 0.5] .+ v .- tnpos(rng, 0.2, 0.4, n)
        sp(T) = sfmodel_spec(depvar = T.(y), frontier = T.(X), noise = :Normal, ineff = :TruncatedNormal, copula = :Gaussian, type = :prod)
        specs = Dict(T => MB._assemble_MCI_spec(sp(T).mci_spec, sfmodel_method(method = :MCI, n_draws = 1024)) for T in (Float64, Float32))
        # Gaussian copula, θ_ρ = 0.5 (ρ = 0.46), MCI (both scores computed in the evaluation): the gap was 1.34 nats at
        # μ/σᵤ = -0.5 and 0.90 at -5 (MSLE 1.32 and 0.49); now within 2.1e-7 relative, about as without a copula
        for a in (-0.5, -5.0)
            θ = [1.0, 0.5, a * 0.2, log(0.04), log(0.0625), 0.5]
            l64 = -MB.MCI_nll(specs[Float64], θ; chunks = 10); l32 = -MB.MCI_nll(specs[Float32], θ; chunks = 10)
            @test abs(l32 - l64) <= 1e-6 * abs(l64)
        end
        # MSLE with user-supplied draws (one row shared by every observation) that hold base-2 Halton tail points:
        # the old bounds capped their scores at ±4.62 (the gap was 12 to 29 nats); now within 2.0e-7 relative
        tails = [2.0^-24, 2.0^-22, 2.0^-20, 2.0^-18, 1 - 2.0^-18, 1 - 2.0^-20, 1 - 2.0^-22, 1 - 2.0^-24]
        row = vcat(tails, [(k - 0.5) / 56 for k in 1:56])'
        ms = Dict(T => MS._assemble_MSLE_spec(sp(T).msle_spec, sfmodel_method(method = :MSLE, n_draws = 64, multiRand = false, draws = row))
                  for T in (Float64, Float32))
        for (a, th) in ((-0.5, 0.5), (-5.0, 2.0))
            θ = [1.0, 0.5, a * 0.2, log(0.04), log(0.0625), th]
            l64 = -MS.qmc_nll(ms[Float64], θ; chunks = 10); l32 = -MS.qmc_nll(ms[Float32], θ; chunks = 10)
            @test abs(l32 - l64) <= 1e-6 * abs(l64)
        end
        # (the Gumbel copula's log-likelihood gap, 4.05e-3 under MCI on a half-normal Clayton-copula sample,
        # now 5e-6, and MSLE with long Halton sequences are not measured in this file; the kernels are tested above)
    end

    end   # setprecision
end
