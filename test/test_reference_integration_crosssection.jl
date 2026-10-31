# Reference tests of the cross-sectional likelihoods. Twelve cells (Weibull, Lomax, lognormal and Rayleigh
# inefficiency; Student-t and Laplace noise; the Gaussian, Clayton, Clayton 90° and Gumbel copulas with half-normal
# inefficiency) give the density of one observation at seven residuals ε. The reference is the textbook density,
# written in this file without package code, log f(ε) = log ∫₀^∞ f_v(ε + u) f_u(u) c(F_v(ε + u), F_u(u)) du (c = 1
# without a copula), integrated by adaptive Gauss-Kronrod quadrature. Two checks per cell, under MCI and under MSLE:
# (a) the package's integrand of one draw t, integrated over t in (0, 1) by quadrature, against the reference;
# (b) the simulated log density at 2^10, 2^12 and 2^14 shared Halton draws against the reference.
# With the environment variable SFRONTIERS_REFERENCE_REPORT=1 the file also prints a table of the errors.
using Test
using SFrontiers
using SpecialFunctions
using Random
using QuadGK
using Printf

@testset "Reference tests: cross-sectional likelihoods against numerical integration of textbook densities" begin
    MB = SFrontiers.MCI_Backend
    MS = SFrontiers.MSLE_Backend

    # The parameter vector by name (frontier constant 0), from a spec that only fixes the parameter layout. Every
    # cell names all its values: sfmodel_init sets an omitted component to 0.1.
    function _refx_theta(noise, ineff, copula, init)
        sp = sfmodel_spec(depvar = randn(Xoshiro(1), 200), frontier = ones(200, 1), noise = noise, ineff = ineff,
                          copula = copula, type = :prod)
        return sfmodel_init(; spec = sp, frontier = [0.0], init..., message = false)
    end
    # The negative log-likelihood of the single observation ε with the draws in the 1 × D matrix `draws`.
    _refx_nll(nll, c, θ, ε, draws) = nll([ε], ones(1, 1), ones(1, 1), θ, draws; noise = c.noise, ineff = c.ineff,
                                         copula = c.copula, chunks = 1)

    # Textbook densities. Noise: log f_v. Inefficiency: log f_u. CDFs as pairs (F, 1 - F), each computed directly
    # where it is the smaller one. The standard normal quantile and log F from such a pair, from the smaller tail.
    _refx_lfv_normal(v, σ) = -v^2 / (2 * σ^2) - log(σ) - log(2π) / 2
    _refx_lfv_t(v, σ, ν) = loggamma((ν + 1) / 2) - loggamma(ν / 2) - log(ν * π) / 2 - log(σ) -
                           (ν + 1) / 2 * log1p((v / σ)^2 / ν)
    _refx_lfv_laplace(v, b) = -abs(v) / b - log(2 * b)
    _refx_Fv_normal(v, σ) = (erfc(-v / (σ * sqrt(2))) / 2, erfc(v / (σ * sqrt(2))) / 2)
    _refx_Fv_laplace(v, b) = v < 0 ? (exp(v / b) / 2, 1 - exp(v / b) / 2) : (1 - exp(-v / b) / 2, exp(-v / b) / 2)
    _refx_lfu_weibull(u, λ, k) = log(k / λ) + (k - 1) * log(u / λ) - (u / λ)^k
    _refx_lfu_lomax(u, λ, α) = log(α / λ) - (α + 1) * log1p(u / λ)
    _refx_lfu_lognormal(u, μ, σ) = -(log(u) - μ)^2 / (2 * σ^2) - log(u) - log(σ) - log(2π) / 2
    _refx_lfu_rayleigh(u, σ) = log(u) - 2 * log(σ) - u^2 / (2 * σ^2)
    _refx_lfu_halfnormal(u, σ) = log(2) - u^2 / (2 * σ^2) - log(σ) - log(2π) / 2
    _refx_Fu_halfnormal(u, σ) = (erf(u / (σ * sqrt(2))), erfc(u / (σ * sqrt(2))))
    _refx_qnorm(F) = F[1] < 0.5 ? -sqrt(2) * erfcinv(2 * F[1]) : sqrt(2) * erfcinv(2 * F[2])
    _refx_logF(F) = F[1] < 0.5 ? log(F[1]) : log1p(-F[2])

    # Copula log densities: Gaussian from the normal scores x and y; Clayton from log a and log b; Gumbel from
    # w1 = -log a and w2 = -log b.
    _refx_lc_gaussian(x, y, ρ) = (2 * ρ * x * y - ρ^2 * (x^2 + y^2)) / (2 * (1 - ρ^2)) - log(1 - ρ^2) / 2
    function _refx_lc_clayton(la, lb, ρ)
        xa, xb = -ρ * la, -ρ * lb
        m = max(xa, xb)
        ls = m + log(exp(xa - m) + exp(xb - m) - exp(-m))     # log(a^-ρ + b^-ρ - 1)
        return log(ρ + 1) + (-ρ - 1) * (la + lb) + (-1 / ρ - 2) * ls
    end
    function _refx_lc_gumbel(w1, w2, ρ)
        S = w1^ρ + w2^ρ
        return -S^(1 / ρ) + (ρ - 1) * (log(w1) + log(w2)) + (-2 + 2 / ρ) * log(S) + log1p((ρ - 1) * S^(-1 / ρ)) +
               w1 + w2
    end

    # The log-integrands of the copula cells (half-normal σ_u = 0.5; normal σ_v = 0.3 or Laplace b = 0.3). Where a
    # probability that the copula needs, or its complement, or the Gumbel sum w1^ρ + w2^ρ underflows to 0 in Float64,
    # the log-integrand is -Inf (integrand 0): the copula formula has no Float64 value there (a quantile of ±Inf, or
    # the logarithm of 0), and the true integrand is below 1e-300 of its largest value.
    function _refx_li_gaussian(ε, u, ρ, noise)
        v = ε + u
        Fu = _refx_Fu_halfnormal(u, 0.5)
        0.0 in Fu && return -Inf
        if noise === :Normal
            lfv, x = _refx_lfv_normal(v, 0.3), v / 0.3
        else
            Fv = _refx_Fv_laplace(v, 0.3)
            0.0 in Fv && return -Inf
            lfv, x = _refx_lfv_laplace(v, 0.3), _refx_qnorm(Fv)
        end
        return lfv + _refx_lfu_halfnormal(u, 0.5) + _refx_lc_gaussian(x, _refx_qnorm(Fu), ρ)
    end
    function _refx_li_clayton(ε, u, ρ, rotated)
        v = ε + u
        Fv = _refx_Fv_normal(v, 0.3)
        Fv = rotated ? reverse(Fv) : Fv                       # Clayton 90°: the Clayton density at (1 - F_v, F_u)
        Fu = _refx_Fu_halfnormal(u, 0.5)
        (Fv[1] == 0 || Fu[1] == 0) && return -Inf
        return _refx_lfv_normal(v, 0.3) + _refx_lfu_halfnormal(u, 0.5) +
               _refx_lc_clayton(_refx_logF(Fv), _refx_logF(Fu), ρ)
    end
    function _refx_li_gumbel(ε, u, ρ)
        v = ε + u
        w1 = -_refx_logF(_refx_Fv_normal(v, 0.3))
        w2 = -_refx_logF(_refx_Fu_halfnormal(u, 0.5))
        (w1 == 0 || w2 == 0 || w1^ρ + w2^ρ == 0) && return -Inf
        return _refx_lfv_normal(v, 0.3) + _refx_lfu_halfnormal(u, 0.5) + _refx_lc_gumbel(w1, w2, ρ)
    end

    # The reference log f(ε): quadrature over u with breakpoints, rtol 1e-13, of exp(log-integrand - M), with M the
    # largest log-integrand on a grid; M is added back.
    _refx_grid = vcat(exp10.(range(-12, 1, length = 105)), collect(range(0.05, 10, length = 400)))
    function _refx_logref(li, ε, extra)
        bps = sort(unique(vcat([0.0, 0.05, 0.25, 1.0, 3.0, 10.0], extra)))
        M = maximum(li(ε, u) for u in _refx_grid)
        I, _ = quadgk(u -> exp(li(ε, u) - M), bps..., Inf; rtol = 1e-13)
        return log(I) + M
    end
    _refx_kink(ε) = ε < 0 ? [-ε] : Float64[]                  # Laplace noise: the kink of |ε + u| at u = -ε

    # Check (a): log ∫₀¹ exp(-nll(ε; one draw t)) dt by quadrature, rtol 1e-12. Any importance scale gives the same
    # exact integral, so this check catches errors of the formulas and their guards, not of the scale.
    _refx_bps_a = (0.0, 1e-12, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 0.9, 0.99, 1 - 1e-4, 1 - 1e-6, 1 - 1e-8, 1 - 1e-10,
                   1 - 1e-12, 1.0)
    function _refx_exact(nll, c, θ, ε)
        I, _ = quadgk(t -> exp(-_refx_nll(nll, c, θ, ε, reshape([t], 1, 1))), _refx_bps_a...; rtol = 1e-12,
                      maxevals = 10^7)
        return log(I)
    end
    # Check (b): e(D), the largest |log f_sim - log f_ref| over the seven ε at D shared Halton draws.
    _refx_Ds = (2^10, 2^12, 2^14)
    _refx_ladder(nll, halton, c, θ, εs, refs) =
        [maximum(abs(-_refx_nll(nll, c, θ, εs[i], reshape(halton(D), 1, D)) - refs[i]) for i in eachindex(εs))
         for D in _refx_Ds]

    εs = [-1.2, -0.6, -0.3, -0.1, 0.0, 0.15, 0.4]
    ln_sv2 = log(0.09)                                        # σ_v = 0.3
    ln_su2 = log(0.25)                                        # σ_u = 0.5; the lognormal σ = 0.5
    tol_exact = 1e-9
    tol_laplace = 1e-8 / 0.3 + 1e-9    # the smoothed |z| of the Laplace density moves a log density by ≤ 1e-8/b
    tol_clayton = 1e-5                 # covers the clamp of F_v and F_u to [1e-6, 1 - 1e-6] in the Clayton densities

    # Each cell: the model; its init values (natural scale → transformed); the log-integrand of the reference;
    # extra quadrature breakpoints; the tolerance of check (a); `ref`, log f(ε) at the seven ε by quadrature of the
    # textbook densities at 45 significant digits; `measured`, the errors e(2^10), e(2^12), e(2^14) of check (b)
    # for MCI and for MSLE, measured when this test was written (rounded up to three significant digits). Each bound
    # of check (b) is twice the measured error, and at least 1e-12.
    cells = [
        (label = "C1", name = "normal noise, Weibull inefficiency with k = 1.5",
         noise = :Normal, ineff = :Weibull, copula = :None,
         init = (ln_sigma_v_sq = ln_sv2, ln_lambda = log(0.5), ln_k = log(1.5)),
         li = (ε, u) -> _refx_lfv_normal(ε + u, 0.3) + _refx_lfu_weibull(u, 0.5, 1.5),
         extra = ε -> Float64[], tol = tol_exact,
         ref = [-1.6421532889307773, -0.18612876795942692, -0.050771091926676062, -0.28311485361485649,
                -0.5127223179616806, -1.011976286013077, -2.2874958487965608],
         measured = (MCI = (1.23e-4, 1.54e-5, 1.93e-6), MSLE = (2.58e-3, 6.52e-4, 1.64e-4))),
        (label = "C2", name = "normal noise, Weibull inefficiency with k = 0.7",
         noise = :Normal, ineff = :Weibull, copula = :None,
         init = (ln_sigma_v_sq = ln_sv2, ln_lambda = log(0.5), ln_k = log(0.7)),
         li = (ε, u) -> _refx_lfv_normal(ε + u, 0.3) + _refx_lfu_weibull(u, 0.5, 0.7),
         extra = ε -> [1e-12, 1e-9, 1e-6, 1e-3], tol = tol_exact,
         ref = [-1.6588878191935852, -0.63874069096766274, -0.24109828235575091, -0.24080457499336995,
                -0.35436486943814814, -0.68757649947029124, -1.7154944953175852],
         measured = (MCI = (5.34e-3, 2.05e-3, 7.78e-4), MSLE = (1.49e-3, 3.71e-4, 9.28e-5))),
        (label = "C3", name = "normal noise, Lomax inefficiency",
         noise = :Normal, ineff = :Lomax, copula = :None,
         init = (ln_sigma_v_sq = ln_sv2, ln_lambda = 0.0, ln_alpha = log(3)),
         li = (ε, u) -> _refx_lfv_normal(ε + u, 0.3) + _refx_lfu_lomax(u, 1.0, 3.0),
         extra = ε -> Float64[], tol = tol_exact,
         ref = [-1.8463379423475165, -0.52454670956999609, -0.11260447615384368, -0.14306448698158879,
                -0.27710663127726349, -0.64338541705937146, -1.7266798421147464],
         measured = (MCI = (1.17e-3, 2.93e-4, 7.32e-5), MSLE = (1.50e-3, 3.76e-4, 9.38e-5))),
        (label = "C4", name = "normal noise, lognormal inefficiency",
         noise = :Normal, ineff = :Lognormal, copula = :None,
         init = (ln_sigma_v_sq = ln_sv2, mu = -1.0, ln_sigma_u_sq = ln_su2),
         li = (ε, u) -> _refx_lfv_normal(ε + u, 0.3) + _refx_lfu_lognormal(u, -1.0, 0.5),
         extra = ε -> [0.01], tol = tol_exact,
         ref = [-2.1436877396034099, -0.0902055366036604, 0.086584843701105953, -0.21386490371951996,
                -0.49946062362865892, -1.1036761543266772, -2.5942070758825086],
         measured = (MCI = (1.12e-15, 1.34e-15, 1.56e-15), MSLE = (2.41e-3, 6.49e-4, 1.72e-4))),
        (label = "C5", name = "normal noise, Rayleigh inefficiency",
         noise = :Normal, ineff = :Rayleigh, copula = :None,
         init = (ln_sigma_v_sq = ln_sv2, ln_sigma_u_sq = ln_su2),
         li = (ε, u) -> _refx_lfv_normal(ε + u, 0.3) + _refx_lfu_rayleigh(u, 0.5),
         extra = ε -> Float64[], tol = tol_exact,
         ref = [-1.0102357135127598, -0.10494235135156697, -0.29211646636935735, -0.72318581552688931,
                -1.0441016761586788, -1.6692703091687573, -3.125980816987525],
         measured = (MCI = (4.95e-6, 3.10e-7, 1.94e-8), MSLE = (5.49e-3, 1.45e-3, 3.71e-4))),
        (label = "C6", name = "Student-t noise, lognormal inefficiency",
         noise = :StudentT, ineff = :Lognormal, copula = :None,
         init = (ln_sigma_v_sq = ln_sv2, ln_nu_minus_2 = log(3), mu = -1.0, ln_sigma_u_sq = ln_su2),
         li = (ε, u) -> _refx_lfv_t(ε + u, 0.3, 5.0) + _refx_lfu_lognormal(u, -1.0, 0.5),
         extra = ε -> [0.01], tol = tol_exact,
         ref = [-1.8960254951385151, -0.16344063098169072, 0.026345625810859572, -0.2786645087020543,
                -0.55852137798463506, -1.0983953826451636, -2.1637059235426428],
         measured = (MCI = (5.56e-16, 1.12e-15, 1.78e-15), MSLE = (1.55e-3, 4.12e-4, 1.08e-4))),
        (label = "C7", name = "Laplace noise, Lomax inefficiency",
         noise = :Laplace, ineff = :Lomax, copula = :None,
         init = (ln_b = log(0.3), ln_lambda = 0.0, ln_alpha = log(3)),
         li = (ε, u) -> _refx_lfv_laplace(ε + u, 0.3) + _refx_lfu_lomax(u, 1.0, 3.0),
         extra = _refx_kink, tol = tol_laplace,
         ref = [-1.6874473772354003, -0.62084701458199878, -0.2058028438972798, -0.13914823158513446,
                -0.31504412988851988, -0.81504412988851988, -1.6483774632218532],
         measured = (MCI = (8.70e-4, 2.18e-4, 5.44e-5), MSLE = (1.12e-3, 2.79e-4, 6.97e-5))),
        (label = "C8", name = "normal noise, half-normal inefficiency, Gaussian copula",
         noise = :Normal, ineff = :HalfNormal, copula = :Gaussian,
         init = (ln_sigma_v_sq = ln_sv2, ln_sigma_u_sq = ln_su2, theta_rho = [atanh(0.6 / 0.999)]),
         li = (ε, u) -> _refx_li_gaussian(ε, u, 0.6, :Normal),
         extra = ε -> Float64[], tol = tol_exact,
         ref = [-3.3498419060013356, 0.045523704496312568, 0.3492543618770652, -0.16975916611475587,
                -0.66368059569691174, -1.7072652830204716, -4.2714775132330906],
         measured = (MCI = (6.42e-4, 1.68e-4, 9.40e-5), MSLE = (2.25e-2, 4.64e-3, 7.76e-4))),
        (label = "C9", name = "normal noise, half-normal inefficiency, Clayton copula",
         noise = :Normal, ineff = :HalfNormal, copula = :Clayton,
         init = (ln_sigma_v_sq = ln_sv2, ln_sigma_u_sq = ln_su2, theta_rho = [log(1.5 - 1e-6)]),
         li = (ε, u) -> _refx_li_clayton(ε, u, 1.5, false),
         extra = ε -> Float64[], tol = tol_clayton,
         ref = [-2.8800912256700722, 0.027314400214303667, 0.36281635195446999, -0.31429896951243899,
                -0.78095262124063059, -1.5944974032480145, -3.261339478802443],
         measured = (MCI = (6.90e-3, 1.74e-3, 3.29e-3), MSLE = (9.46e-3, 3.18e-3, 3.40e-3))),
        (label = "C10", name = "normal noise, half-normal inefficiency, Clayton 90° copula",
         noise = :Normal, ineff = :HalfNormal, copula = :Clayton90,
         init = (ln_sigma_v_sq = ln_sv2, ln_sigma_u_sq = ln_su2, theta_rho = [log(2 - 1e-6)]),
         li = (ε, u) -> _refx_li_clayton(ε, u, 2.0, true),
         extra = ε -> Float64[], tol = tol_clayton,
         ref = [-1.3440179074635453, -0.44312410704475155, -0.38966931005815653, -0.4566306441208771,
                -0.52224872523503189, -0.67511592214031664, -1.1769994179304291],
         measured = (MCI = (8.35e-7, 1.30e-8, 2.03e-10), MSLE = (1.64e-6, 2.56e-8, 3.99e-10))),
        (label = "C11", name = "normal noise, half-normal inefficiency, Gumbel copula",
         noise = :Normal, ineff = :HalfNormal, copula = :Gumbel,
         init = (ln_sigma_v_sq = ln_sv2, ln_sigma_u_sq = ln_su2, theta_rho = [log(1.5)]),
         li = (ε, u) -> _refx_li_gumbel(ε, u, 2.5),
         extra = ε -> Float64[], tol = tol_exact,
         ref = [-5.4311038604606545, 0.078061127216086548, 0.66739573285536639, -0.58413648017495355,
                -1.563995348486445, -3.3274740494123265, -7.0211783682192305],
         measured = (MCI = (1.58e-3, 1.65e-3, 9.23e-4), MSLE = (1.81e-3, 3.21e-4, 2.73e-4))),
        (label = "C12", name = "Laplace noise, half-normal inefficiency, Gaussian copula",
         noise = :Laplace, ineff = :HalfNormal, copula = :Gaussian,
         init = (ln_b = log(0.3), ln_sigma_u_sq = ln_su2, theta_rho = [atanh(0.6 / 0.999)]),
         li = (ε, u) -> _refx_li_gaussian(ε, u, 0.6, :Laplace),
         extra = _refx_kink, tol = tol_laplace,
         ref = [-2.4591102553260983, -0.087028640863773861, 0.25235249769212746, -0.16471759182620561,
                -0.65719091064983519, -1.4535290550076609, -2.6786401554096126],
         measured = (MCI = (2.41e-3, 6.93e-4, 1.83e-4), MSLE = (8.26e-4, 2.18e-4, 5.96e-5))),
    ]

    refs = Dict{String, Vector{Float64}}()
    @testset "the references reproduce the pinned values" begin
        for c in cells
            refs[c.label] = [_refx_logref(c.li, ε, c.extra(ε)) for ε in εs]
            @test maximum(abs.(refs[c.label] .- c.ref)) <= 1e-11
        end
    end

    report = String[]
    for c in cells
        @testset "$(c.label): $(c.name)" begin
            θ = _refx_theta(c.noise, c.ineff, c.copula, c.init)
            r = refs[c.label]
            for (backend, nll, halton) in (("MCI", MB.MCI_nll, MB.make_halton_p),
                                           ("MSLE", MS.qmc_nll, MS.make_halton_p))
                err_a = [abs(_refx_exact(nll, c, θ, εs[i]) - r[i]) for i in eachindex(εs)]
                @test maximum(err_a) <= c.tol
                e = _refx_ladder(nll, halton, c, θ, εs, r)
                m = getproperty(c.measured, Symbol(backend))
                for k in eachindex(_refx_Ds)
                    @test e[k] <= max(2 * m[k], 1e-12)
                end
                @test e[1] <= 1e-12 || e[3] < e[1]
                push!(report, @sprintf("%-4s %-7s %.2e  %.2e  %.2e  %.2e", c.label, backend, maximum(err_a),
                                       e[1], e[2], e[3]))
            end
        end
    end

    if get(ENV, "SFRONTIERS_REFERENCE_REPORT", "") == "1"
        @printf("%-4s %-7s %-9s %-9s %-9s %s\n", "cell", "backend", "check (a)", "e(2^10)", "e(2^12)", "e(2^14)")
        foreach(println, report)
    end
end
