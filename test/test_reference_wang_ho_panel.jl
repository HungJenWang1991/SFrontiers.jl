# Reference test of the Wang–Ho fixed-effects panel likelihood with half-normal and truncated-normal inefficiency.
# The package's Wang–Ho log-likelihood is the density of the within deviations in orthonormal coordinates of the
# subspace they span; it has no log T term. Model: y_it = α_i + x_it'β + ε_it, ε_it = v_it − s·h_it·u_i (s = 1
# production, s = −1 cost), h_it = exp(z_it'δ), v_it ~ N(0, σ_v²), u_i ~ N⁺(μ, σ_u²), α_i a fixed effect. With ε̃_i
# the within deviations of y_i − x_i'β, that density is
# f(ε̃_i) = √T_i · ∫₀^∞ f_u(u) ∫ Π_t φ_σv(ε̃_it + c + s·h_it·u) dc du.
# The reference computes this two-dimensional integral by nested adaptive Gauss-Kronrod quadrature, written in this
# file without package code, and is compared with the closed form of the :MLE backend firm by firm, with the
# log-likelihood of held :MLE fits through the public API, and with the simulated log-likelihood of the panel
# backend (MSLE and MCI) at 2^10, 2^12 and 2^14 shared Halton draws.
# With the environment variable SFRONTIERS_REFERENCE_REPORT=1 the file also prints a table of the differences.
using Test
using SFrontiers
using SpecialFunctions
using Random
using QuadGK
using Printf
using Optim: Newton

@testset "Reference test: the Wang-Ho fixed-effects panel likelihood against two-dimensional quadrature" begin
    WHM = SFrontiers.MLE_Backend
    WHP = SFrontiers.Panel_Backend

    # One dataset: the generator restarts for each dataset, so the datasets of a panel share x, z, α and v. Rows are
    # firm-major; u_i is half normal, or truncated normal drawn by rejection.
    function _wh_data(seed, firm, ineff, μ, s)
        rng = Xoshiro(seed)
        nf, nt = maximum(firm), length(firm)
        xs = randn(rng, nt)
        zs = randn(rng, nt)
        αs = randn(rng, nf)
        vs = 0.3 .* randn(rng, nt)
        us = zeros(nf)
        for i in 1:nf
            if ineff === :HalfNormal
                us[i] = abs(0.5 * randn(rng))
            else
                w = μ + 0.5 * randn(rng)
                while w <= 0
                    w = μ + 0.5 * randn(rng)
                end
                us[i] = w
            end
        end
        hs = exp.(0.3 .* zs)
        return (y = αs[firm] .+ 0.5 .* xs .+ vs .- s .* hs .* us[firm], x = xs, z = zs, firm = firm)
    end

    # The parameters of θ, in the panel order [β; δ; (μ); ln σ_u²; ln σ_v²].
    _wh_par(θ) = (β = θ[1], δ = θ[2], μ = length(θ) == 5 ? θ[3] : 0.0, σu = exp(θ[end - 1] / 2),
                  σv = exp(θ[end] / 2))
    # Firm i's within deviations ε̃ and its h_it at θ.
    function _wh_firm(d, θ, i)
        p = _wh_par(θ)
        rows = findall(==(i), d.firm)
        r = d.y[rows] .- p.β .* d.x[rows]
        return r .- sum(r) / length(r), exp.(p.δ .* d.z[rows])
    end

    _wh_lphi(x, σ) = -x^2 / (2 * σ^2) - log(σ) - log(2π) / 2                       # log φ_σ(x)
    _wh_lfu(u, μ, σu) = _wh_lphi(u - μ, σu) - log(erfc(-μ / (σu * sqrt(2))) / 2)   # log f_u(u), u ~ N⁺(μ, σ_u²)

    # The log of the inner integral over c at u, after the substitution c = −s·h̄·u + (σ_v/√T)·w: the integral is
    # (σ_v/√T)·exp(g(0))·∫ exp(g(w) − g(0)) dw, with g(w) = Σ_t log φ_σv(ε̃_t + c + s·h_t·u).
    function _wh_loginner(e, h, s, σv, u)
        nT = length(e)
        hbar = sum(h) / nT
        g = w -> sum(_wh_lphi(e[t] + (-s * hbar * u + σv / sqrt(nT) * w) + s * h[t] * u, σv) for t in 1:nT)
        g0 = g(0.0)
        J, _ = quadgk(w -> exp(g(w) - g0), -Inf, -8.0, 0.0, 8.0, Inf; rtol = 1e-12)
        return log(σv / sqrt(nT)) + g0 + log(J)
    end
    # The same in closed form: log of T^(−1/2)·(2πσ_v²)^(−(T−1)/2)·exp(−‖a − ā‖²/(2σ_v²)), a_t = ε̃_t + s·h_t·u.
    function _wh_loginner_closed(e, h, s, σv, u)
        nT = length(e)
        a = e .+ s .* h .* u
        a = a .- sum(a) / nT
        return -log(nT) / 2 - (nT - 1) / 2 * log(2π * σv^2) - sum(abs2, a) / (2 * σv^2)
    end

    # The reference log f(ε̃_i): the outer integral over u in [0, u_max], u_max = max(μ, 0) + 40σ_u, of
    # exp(L(u) − M), L(u) = log f_u(u) + log(inner integral), M the largest L on a coarse grid; rtol 1e-11.
    function _wh_logref(e, h, s, μ, σu, σv)
        umax = max(μ, 0.0) + 40 * σu
        L = u -> _wh_lfu(u, μ, σu) + _wh_loginner(e, h, s, σv, u)
        M = maximum(L(u) for u in range(0.0, 8 * σu; length = 65))
        f = u -> exp(L(u) - M)
        I, _ = μ > 0 ? quadgk(f, 0.0, μ, 0.5 * σu, σu, 2 * σu, 4 * σu, 8 * σu, umax; rtol = 1e-11) :
                       quadgk(f, 0.0, 0.5 * σu, σu, 2 * σu, 4 * σu, 8 * σu, umax; rtol = 1e-11)
        return log(length(e)) / 2 + M + log(I)
    end
    function _wh_refs(d, θ, s)
        p = _wh_par(θ)
        return [_wh_logref(_wh_firm(d, θ, i)..., s, p.μ, p.σu, p.σv) for i in 1:maximum(d.firm)]
    end

    # The :MLE closed form, −LL_T, of each firm alone (θ in the panel order; LL_T takes μ before δ).
    function _wh_mle(d, θ, s)
        tn = length(θ) == 5
        po = tn ? (begx = 1, endx = 1, begz = 2, endz = 2, begq = 3, endq = 3, begw = 4, endw = 4, begv = 5, endv = 5) :
                  (begx = 1, endx = 1, begq = 2, endq = 2, begw = 3, endw = 3, begv = 4, endv = 4)
        rho = tn ? θ[[1, 3, 2, 4, 5]] : θ
        out = zeros(maximum(d.firm))
        for i in eachindex(out)
            rows = findall(==(i), d.firm)
            nT = length(rows)
            out[i] = -WHM.LL_T(tn ? WHM.PFEWHT : WHM.PFEWHH, WHM.sf_demean(d.y[rows]),
                               WHM.sf_demean(reshape(d.x[rows], :, 1)), nothing, reshape(d.z[rows], :, 1), nothing,
                               nothing, s, nT, po, rho, WHM.get_rowIDT(fill(1, nT)), nothing)
        end
        return out
    end

    # The panel backend with D shared Halton draws.
    _wh_Ds = (2^10, 2^12, 2^14)
    _wh_isp(sp, m, D) = WHP._assemble_panel_spec(sp.panel_spec,
                                                 SFrontiers.sfmodel_panel_method(method = m, n_draws = D,
                                                                                multiRand = false, chunks = 1))

    # Panel (a) is balanced, 50 firms × 6 periods; panel (b) is unbalanced, 50 firms with 5 to 12 periods.
    panels = [(label = "a", seed = 52, firm = repeat(1:50, inner = 6)),
              (label = "b", seed = 53, firm = vcat(fill.(1:50, repeat([5, 6, 7, 9, 12], 10))...))]
    ineffs = [(label = "HN", name = "half normal", ineff = :HalfNormal, μ = 0.0),
              (label = "TN(0.2)", name = "truncated normal with μ = 0.2", ineff = :TruncatedNormal, μ = 0.2),
              (label = "TN(-0.5)", name = "truncated normal with μ = −0.5", ineff = :TruncatedNormal, μ = -0.5)]
    types = [(label = "prod", name = "production", s = 1, type = :prod),
             (label = "cost", name = "cost", s = -1, type = :cost)]

    # e(2^10), e(2^12), e(2^14) of the panel backend at the true θ, for MSLE and for MCI, measured when this test was
    # written (rounded up to three significant digits). Each bound is twice the measured error, and at least 1e-12.
    measured = Dict(
        "a HN prod" => (MSLE = (1.69e-2, 4.72e-3, 1.27e-3), MCI = (1.92e-2, 4.78e-3, 1.20e-3)),
        "a HN cost" => (MSLE = (3.26e-3, 6.85e-4, 5.20e-4), MCI = (1.82e-2, 4.54e-3, 1.14e-3)),
        "a TN(0.2) prod" => (MSLE = (1.48e-2, 4.37e-3, 1.23e-3), MCI = (1.38e-2, 3.44e-3, 8.59e-4)),
        "a TN(0.2) cost" => (MSLE = (1.66e-2, 3.04e-3, 4.20e-4), MCI = (1.25e-2, 3.13e-3, 7.80e-4)),
        "a TN(-0.5) prod" => (MSLE = (1.42e-2, 4.26e-3, 1.22e-3), MCI = (3.90e-2, 9.73e-3, 2.44e-3)),
        "a TN(-0.5) cost" => (MSLE = (1.95e-3, 1.64e-3, 6.84e-4), MCI = (3.54e-2, 8.83e-3, 2.21e-3)),
        "b HN prod" => (MSLE = (1.55e-2, 4.32e-3, 1.16e-3), MCI = (1.74e-2, 4.34e-3, 1.09e-3)),
        "b HN cost" => (MSLE = (1.31e-3, 1.62e-3, 7.48e-4), MCI = (2.10e-2, 5.23e-3, 1.31e-3)),
        "b TN(0.2) prod" => (MSLE = (1.24e-2, 3.59e-3, 9.95e-4), MCI = (1.22e-2, 3.05e-3, 7.61e-4)),
        "b TN(0.2) cost" => (MSLE = (6.32e-3, 2.01e-3, 6.11e-4), MCI = (1.26e-2, 3.13e-3, 7.82e-4)),
        "b TN(-0.5) prod" => (MSLE = (1.09e-2, 3.26e-3, 9.37e-4), MCI = (3.41e-2, 8.52e-3, 2.13e-3)),
        "b TN(-0.5) cost" => (MSLE = (9.88e-3, 9.94e-4, 2.34e-4), MCI = (3.52e-2, 8.80e-3, 2.20e-3)),
    )

    @testset "the inner integral against its Gaussian closed form" begin
        for pn in panels
            d = _wh_data(pn.seed, pn.firm, :HalfNormal, 0.0, 1)
            e, h = _wh_firm(d, [0.5, 0.3, log(0.25), log(0.09)], 1)
            for ut in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)                  # u = 0, σ_u/2, σ_u, 2σ_u, 4σ_u, 8σ_u
                @test abs(_wh_loginner(e, h, 1, 0.3, ut) - _wh_loginner_closed(e, h, 1, 0.3, ut)) <= 1e-12
            end
        end
    end

    hold = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,))
    report = String[]
    for pn in panels, ie in ineffs, ty in types
        @testset "panel ($(pn.label)), $(ie.name), $(ty.name)" begin
            d = _wh_data(pn.seed, pn.firm, ie.ineff, ie.μ, ty.s)
            sp = sfmodel_spec(depvar = d.y, frontier = reshape(d.x, :, 1), zvar = reshape(d.z, :, 1), id = d.firm,
                              noise = :Normal, ineff = ie.ineff, type = ty.type, datatype = :panel_TFE)
            θ0 = ie.ineff === :HalfNormal ? [0.5, 0.3, log(0.25), log(0.09)] : [0.5, 0.3, ie.μ, log(0.25), log(0.09)]
            public = pn.label == "a" && ty.s == 1 && ie.μ >= 0
            ref0 = Float64[]
            mle_max, pub_max = 0.0, 0.0
            for θ in (θ0, θ0 .+ 0.05)
                ref = _wh_refs(d, θ, ty.s)
                θ == θ0 && (ref0 = ref)
                diffs = abs.(_wh_mle(d, θ, ty.s) .- ref)
                @test maximum(diffs) <= 1e-9
                mle_max = max(mle_max, maximum(diffs))
                if public
                    r = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE), init = θ, optim_options = hold,
                                    show_table = false, verbose = false, marginal = false)
                    @test r.coeff == θ
                    dp = abs(loglikelihood(r) - sum(ref))
                    @test dp <= 1e-8
                    pub_max = max(pub_max, dp)
                end
            end
            key = "$(pn.label) $(ie.label) $(ty.label)"
            for (backend, m) in (("MSLE", :MSLE), ("MCI", :MCI))
                e = [abs(-WHP.panel_nll(_wh_isp(sp, m, D), θ0; chunks = 1) - sum(ref0)) for D in _wh_Ds]
                mm = getproperty(measured[key], Symbol(backend))
                for k in eachindex(_wh_Ds)
                    @test e[k] <= max(2 * mm[k], 1e-12)
                end
                @test e[1] <= 1e-12 || e[3] < e[1]
                push!(report, @sprintf("%-5s %-8s %-4s %-7s %.2e  %.2e  %.2e  %.2e  %s", pn.label, ie.label,
                                       ty.label, backend, e[1], e[2], e[3], mle_max,
                                       public ? @sprintf("%.2e", pub_max) : "-"))
            end
        end
    end

    if get(ENV, "SFRONTIERS_REFERENCE_REPORT", "") == "1"
        @printf("%-5s %-8s %-4s %-7s %-9s %-9s %-9s %-9s %s\n", "panel", "ineff", "type", "backend", "e(2^10)",
                "e(2^12)", "e(2^14)", "MLE", "public")
        foreach(println, report)
    end
end
