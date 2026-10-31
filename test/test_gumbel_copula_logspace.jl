# The Gumbel copula log density of MCI and MSLE, computed in logarithms: the density of one observation by
# quadrature over its draw against a reference value; the MSLE log density at single draws near 1 and in the far
# tails of normal noise against 256-bit values; the helpers of SFNumerics and Laplace noise; ForwardDiff gradients
# against central differences, also at the independence boundary ρ = 1.
using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions
using Random
using QuadGK

@testset "Gumbel copula: the log density in logarithms, without floors" begin
    MB = SFrontiers.MCI_Backend
    MS = SFrontiers.MSLE_Backend
    NM = SFrontiers.SFNumerics

    # The parameter vector by name (frontier constant 0, σᵤ = 0.5, σᵥ = 0.3 or b = 0.3, ρ = exp(θ_ρ) + 1), from a
    # spec that only fixes the parameter layout.
    function _gum_theta(ρ; noise = :Normal)
        sp = sfmodel_spec(depvar = randn(Xoshiro(1), 200), frontier = ones(200, 1), noise = noise,
                          ineff = :HalfNormal, copula = :Gumbel, type = :prod)
        noise === :Laplace &&
            return sfmodel_init(spec = sp, frontier = [0.0], ln_sigma_u_sq = log(0.25), ln_b = log(0.3),
                                theta_rho = [log(ρ - 1)], message = false)
        return sfmodel_init(spec = sp, frontier = [0.0], ln_sigma_u_sq = log(0.25), ln_sigma_v_sq = log(0.09),
                            theta_rho = [log(ρ - 1)], message = false)
    end
    # The negative log-likelihood of one observation ε at one draw t.
    _gum_nll(nll, ε, θ, t; noise = :Normal) =
        nll([ε], ones(1, 1), ones(1, 1), θ, reshape([t], 1, 1); noise = noise, ineff = :HalfNormal,
            copula = :Gumbel, chunks = 1)

    # 256-bit references. The parameters as the likelihood receives them: σᵤ, the noise scale (σᵥ, or b for
    # Laplace noise) and ρ. The half-normal quantile q(t), erfc(q/√2) = 1 - t, by Newton from the Float64 erfinv
    # start. The standard normal CDF. The Gumbel log density from w1 = -log F_v and w2 = -log F_u, and the log
    # density of one draw under normal noise, log φ(z) - log σᵥ + log c.
    _gum_parB(θ; noise = :Normal) = (sqrt(exp(BigFloat(θ[2]))),
                                     noise === :Laplace ? exp(BigFloat(θ[3])) : sqrt(exp(BigFloat(θ[3]))),
                                     exp(BigFloat(θ[4])) + 1)
    function _gum_qB(t)
        tB = BigFloat(t)
        q = BigFloat(sqrt(2) * erfinv(t))
        for _ in 1:60
            dq = (erfc(q / sqrt(BigFloat(2))) - (1 - tB)) / (-sqrt(2 / BigFloat(pi)) * exp(-q^2 / 2))
            q -= dq
            abs(dq) < BigFloat(10)^-70 && break
        end
        return q
    end
    _gum_PhiB(z) = erfc(-z / sqrt(BigFloat(2))) / 2
    function _gum_logcB(w1, w2, ρ)
        S = w1^ρ + w2^ρ
        return -S^(1 / ρ) + (ρ - 1) * (log(w1) + log(w2)) + log((ρ - 1) * S^(-2 + 1 / ρ) + S^(-2 + 2 / ρ)) +
               w1 + w2
    end
    _gum_llB(z, σv, w1, w2, ρ) = -z^2 / 2 - log(sqrt(2 * BigFloat(pi))) - log(σv) + _gum_logcB(w1, w2, ρ)

    setprecision(BigFloat, 256) do

    @testset "the density of one observation at a reference point, MCI and MSLE" begin
        # log ∫₀¹ exp(-nll(ε; t)) dt by adaptive Gauss-Kronrod quadrature (rtol 1e-12) against the textbook density
        # integrated at 45 significant digits, within 1e-9 (half-normal σᵤ = 0.5, σᵥ = 0.3, production frontier),
        # at ρ = 5; ρ = 2.5 is cell C11 of test_reference_integration_crosssection.jl
        bps = (0.0, 1e-12, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 0.9, 0.99, 1 - 1e-4, 1 - 1e-6, 1 - 1e-8, 1 - 1e-10,
               1 - 1e-12, 1.0)
        ρ, ε, ref = 5.0, -0.9, -3.9736848009494805
        θ = _gum_theta(ρ)
        for nll in (MB.MCI_nll, MS.qmc_nll)
            I, _ = quadgk(t -> exp(-_gum_nll(nll, ε, θ, t)), bps...; rtol = 1e-12, maxevals = 10^7)
            @test abs(log(I) - ref) <= 1e-9
        end
    end

    @testset "MSLE: the log density at single draws near 1, against 256 bits" begin
        # -nll at one draw t (ε = -1.2, ρ = 2.5) equals log φ(z) - log σᵥ + log c(Φ(z), t), z = (ε + σᵤ q(t))/σᵥ,
        # for the Float64 draw t itself, in 256 bits, within 1e-9
        θ = _gum_theta(2.5)
        σuB, σvB, ρB = _gum_parB(θ)
        ε = -1.2
        for t in (0.5, 0.9, 0.99, 0.999, 1 - 1e-4, 1 - 1e-5, 1 - 1e-6, 1 - 1e-7, 1 - 1e-8)
            ll = -_gum_nll(MS.qmc_nll, ε, θ, t)
            zB = (BigFloat(ε) + σuB * _gum_qB(t)) / σvB
            ref = _gum_llB(zB, σvB, -log(_gum_PhiB(zB)), -log(BigFloat(t)), ρB)
            @test abs(ll - Float64(ref)) <= 1e-9
        end
    end

    @testset "MSLE: the far tails of the noise, against 256 bits" begin
        # one draw t = 0.5 (ρ = 2.5) and ε such that z = (ε + σᵤ q(t))/σᵥ is ±10, ±30, ±50: the same expression in
        # 256 bits, with w1 = -log Φ(z) for z ≤ 0 and w1 = -log1p(-erfc(z/√2)/2) for z > 0, within 1e-8
        # relative; the gradient in θ is finite
        θ = _gum_theta(2.5)
        σuB, σvB, ρB = _gum_parB(θ)
        t = 0.5
        qB = _gum_qB(t)
        for z in (-50, -30, -10, 10, 30, 50)
            ε = Float64(z * σvB - σuB * qB)
            ll = -_gum_nll(MS.qmc_nll, ε, θ, t)
            zB = (BigFloat(ε) + σuB * qB) / σvB
            w1 = zB <= 0 ? -log(_gum_PhiB(zB)) : -log1p(-erfc(zB / sqrt(BigFloat(2))) / 2)
            ref = _gum_llB(zB, σvB, w1, -log(BigFloat(t)), ρB)
            @test isfinite(ll) && abs(ll - Float64(ref)) <= 1e-8 * abs(Float64(ref))
            @test all(isfinite, ForwardDiff.gradient(p -> _gum_nll(MS.qmc_nll, ε, p, t), θ))
        end
    end

    @testset "the helpers and Laplace noise" begin
        # log(-log Φ(x)) against 256 bits, with -log1p(-erfc(x/√2)/2) for -log Φ(x) when x > 0, within 1e-14
        # relative
        for x in (-60.0, -40.5, -1.0000001, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0000001, 8.3, 38.0, 41.0, 60.0)
            xB = BigFloat(x)
            ref = x > 0 ? log(-log1p(-erfc(xB / sqrt(BigFloat(2))) / 2)) : log(-log(_gum_PhiB(xB)))
            @test abs(NM.sf_log_neglogcdf_normal(x) - Float64(ref)) <= 1e-14 * abs(Float64(ref))
        end
        # log(-log F(x)), F the standard Laplace CDF, against 256 bits, within 1e-14 relative; its derivative on
        # either side of 0 is -1/log 2
        for x in (-60.0, -1.0, 0.0, 1.0, 60.0, 800.0)
            xB = BigFloat(x)
            ref = x < 0 ? log(log(BigFloat(2)) - xB) : log(-log1p(-exp(-xB) / 2))
            @test abs(NM.sf_log_neglogcdf_laplace(x) - Float64(ref)) <= 1e-14 * abs(Float64(ref))
        end
        for x in (-1e-300, 1e-300)
            @test isapprox(ForwardDiff.derivative(NM.sf_log_neglogcdf_laplace, x), -1 / log(2); rtol = 1e-14)
        end
        # log(exp(a) + exp(b))
        @test NM.sf_logaddexp(1.0, 2.0) == 2.0 + log1p(exp(-1.0))
        @test NM.sf_logaddexp(-Inf, -Inf) == -Inf
        @test NM.sf_logaddexp(-Inf, 2.0) == 2.0
        @test ForwardDiff.gradient(a -> NM.sf_logaddexp(a[1], a[2]), [0.3, 0.3]) == [0.5, 0.5]
        # Laplace noise (b = 0.3) under MSLE, one draw t = 0.5 (ρ = 2.5), z = (ε + σᵤ q(t))/b at -50, -1, 0, 1,
        # 50, 700: -nll equals -sqrt(v² + 1e-16)/b - log(2b) + log c in 256 bits (v = ε + σᵤ q(t); the MSLE Laplace
        # density), with w1 = log 2 - z for z < 0 and w1 = -log1p(-e^(-z)/2) for z ≥ 0, within 1e-12 relative; the
        # gradient in θ is finite
        θ = _gum_theta(2.5; noise = :Laplace)
        σuB, bB, ρB = _gum_parB(θ; noise = :Laplace)
        t = 0.5
        qB = _gum_qB(t)
        for z in (-50, -1, 0, 1, 50, 700)
            ε = Float64(z * bB - σuB * qB)
            ll = -_gum_nll(MS.qmc_nll, ε, θ, t; noise = :Laplace)
            vB = BigFloat(ε) + σuB * qB
            zB = vB / bB
            w1 = zB < 0 ? log(BigFloat(2)) - zB : -log1p(-exp(-zB) / 2)
            ref = -sqrt(vB^2 + BigFloat(1e-16)) / bB - log(2 * bB) + _gum_logcB(w1, -log(BigFloat(t)), ρB)
            @test abs(ll - Float64(ref)) <= 1e-12 * abs(Float64(ref))
            @test all(isfinite, ForwardDiff.gradient(p -> _gum_nll(MS.qmc_nll, ε, p, t; noise = :Laplace), θ))
        end
    end

    end   # setprecision

    @testset "gradients against central differences, and at ρ = 1" begin
        # 200 observations, half-normal σᵤ = 0.5, σᵥ = 0.3, 64 draws: the ForwardDiff gradient of the likelihood at
        # θ = (β₀, β₁, ln σᵤ², ln σᵥ², θ_ρ) = (1, 0.5, ln 0.25, ln 0.09, ln 1.5) against central differences
        # (h = 1e-6) within 1e-6 of its largest component; at θ_ρ = -40 (ρ = 1 in Float64) it is finite
        rng = Xoshiro(53)
        n = 200
        xa = randn(rng, n)
        va = 0.3 .* randn(rng, n)
        ua = abs.(0.5 .* randn(rng, n))
        Xa = [ones(n) xa]
        ya = Xa * [1.0, 0.5] .+ va .- ua
        sp = sfmodel_spec(depvar = ya, frontier = Xa, noise = :Normal, ineff = :HalfNormal, copula = :Gumbel,
                          type = :prod)
        imci = MB._assemble_MCI_spec(sp.mci_spec, sfmodel_method(method = :MCI, n_draws = 64))
        imsle = MS._assemble_MSLE_spec(sp.msle_spec, sfmodel_method(method = :MSLE, n_draws = 64))
        θ0 = [1.0, 0.5, log(0.25), log(0.09), log(1.5)]
        h = 1e-6
        for f in (p -> MB.MCI_nll(imci, p; chunks = 1), p -> MS.qmc_nll(imsle, p; chunks = 1))
            g = ForwardDiff.gradient(f, θ0)
            gfd = [(f(θ0 .+ h .* (1:5 .== k)) - f(θ0 .- h .* (1:5 .== k))) / (2h) for k in 1:5]
            @test maximum(abs.(g .- gfd)) <= 1e-6 * maximum(abs.(g))
            @test all(isfinite, ForwardDiff.gradient(f, [θ0[1:4]; -40.0]))
        end
    end
end
