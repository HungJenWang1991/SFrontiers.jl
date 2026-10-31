# The regularized lower incomplete gamma P(a, x), which gives the copula Gamma and Erlang models the CDF of the
# inefficiency: its derivatives in the shape a and in x against a 256-bit reference, its values against the
# stopping rule that tests the value only, and the derivative of a copula Gamma likelihood in ln k against a
# central difference.
using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions
using Random
using LinearAlgebra

@testset "The incomplete-gamma CDF of the copula Gamma model: derivatives in the shape" begin
    MB = SFrontiers.MCI_Backend
    P = MB._reg_inc_gamma_P

    # P(a, x) = x^a e^(-x) Σ_{n=0}^{400} x^n / Γ(a + n + 1), evaluated in BigFloat.
    _gcd_pref(a, x) = x^a * exp(-x) * sum(x^n / SpecialFunctions.gamma(a + n + 1) for n in 0:400)

    @testset "the derivatives in the shape and in x, at and next to integer shapes" begin
        # Continued-fraction branch (x ≥ a + 1), then series branch (x < a + 1).
        pts = ((2.0, 3.001), (2.0, 3.05), (3.0, 4.5), (2.0 + 1e-7, 3.5), (2.5, 4.0), (2.5, 1.0), (2.0, 2.5))
        # References in 256-bit arithmetic: dP/da by a central difference with step 1e-30, and
        # dP/dx = x^(a-1) e^(-x) / Γ(a).
        refs = setprecision(BigFloat, 256) do
            h = BigFloat(10)^-30
            map(pts) do (a0, x0)
                A, Xb = BigFloat(a0), BigFloat(x0)
                (Float64((_gcd_pref(A + h, Xb) - _gcd_pref(A - h, Xb)) / (2 * h)),
                 Float64(Xb^(A - 1) * exp(-Xb) / SpecialFunctions.gamma(A)))
            end
        end
        for ((a0, x0), (dref_a, dref_x)) in zip(pts, refs)
            d_a = ForwardDiff.derivative(a -> P(a, x0, SpecialFunctions.loggamma(a)), a0)
            d_x = ForwardDiff.derivative(x -> P(a0, x, SpecialFunctions.loggamma(a0)), x0)
            @test isapprox(d_a, dref_a; rtol = 1e-10)
            @test isapprox(d_x, dref_x; rtol = 1e-12)
        end
    end

    @testset "values equal to those of the stopping rule that tests the value only" begin
        # The function with its continued fraction stopped when the value of its factor is 1 to within 1e-15.
        function _gcd_P_valueonly(a, x, lga)
            x <= zero(x) && return zero(x)
            if x < a + one(a)
                ap  = a
                del = one(x) / a
                s   = del
                for _ in 1:1000
                    ap  += one(ap)
                    del *= x / ap
                    s   += del
                    abs(del) < abs(s) * 1e-15 && break
                end
                return s * exp(-x + a * log(x) - lga)
            else
                FPMIN = oftype(x, 1e-30)
                b = x + one(x) - a
                c = one(x) / FPMIN
                d = one(x) / b
                h = d
                for i in 1:1000
                    an = -i * (i - a)
                    b += 2 * one(x)
                    d  = an * d + b
                    abs(d) < FPMIN && (d = oftype(d, FPMIN))
                    c  = b + an / c
                    abs(c) < FPMIN && (c = oftype(c, FPMIN))
                    d  = one(x) / d
                    del = d * c
                    h  *= del
                    abs(del - one(x)) < 1e-15 && break
                end
                return one(x) - exp(-x + a * log(x) - lga) * h
            end
        end
        same = Bool[]
        for T in (Float64, Float32), a in (0.3, 1.0, 2.0, 2.5, 3.0, 7.0), x in (0.01, 0.5, 1.9, 3.001, 3.05, 8.0, 30.0)
            aT, xT = T(a), T(x)
            lga = SpecialFunctions.loggamma(aT)
            push!(same, isequal(P(aT, xT, lga), _gcd_P_valueonly(aT, xT, lga)))
        end
        @test all(same)
    end

    @testset "the derivative in ln k of a copula Gamma likelihood at and next to k = 2" begin
        rng = Xoshiro(67)
        n = 300
        Xd = [ones(n) randn(rng, n) randn(rng, n)]
        ud = 0.4 .* (randexp(rng, n) .+ randexp(rng, n))
        yd = Xd * [1.0, 0.5, -0.3] .+ 0.3 .* randn(rng, n) .- ud
        sp = sfmodel_spec(depvar = yd, frontier = Xd, noise = :Normal, ineff = :Gamma, copula = :Gaussian,
                          type = :prod)
        isp = MB._assemble_MCI_spec(sp.mci_spec, sfmodel_method(method = :MCI, n_draws = 256))
        _gcd_theta(l) = sfmodel_init(spec = sp, frontier = [1.0, 0.5, -0.3], ln_k = l, ln_theta = log(0.4),
                                     ln_sigma_v_sq = log(0.09), theta_rho = [0.3], message = false)
        h = 1e-5
        for l in (log(2.0), log(2.0 + 1e-7)), ch in (1, 10)
            nll = p -> MB.MCI_nll(isp, p; chunks = ch)
            # the central difference in ln k, and the gradient along the unit vector of the ln k coordinate
            dfd = (nll(_gcd_theta(l + h)) - nll(_gcd_theta(l - h))) / (2h)
            e_k = (_gcd_theta(l + h) .- _gcd_theta(l - h)) ./ (2h)
            dad = dot(ForwardDiff.gradient(nll, _gcd_theta(l)), e_k)
            @test isapprox(dad, dfd; rtol = 1e-6)
        end
    end
end
