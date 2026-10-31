# Reference test of the Chen–Schmidt–Wang fixed-effects panel likelihood with half-normal inefficiency
# (datatype = :panel_TFE_CSW). Model: y_it = α_i + x_it'β + ε_it, ε_it = v_it − s·u_it (s = 1 production, s = −1
# cost), v_it ~ N(0, σ_v²), u_it ~ N⁺(0, σ_u²), independent over i and t, α_i a fixed effect. The ε_it of a firm are
# then independent with the cross-sectional composed-error density
# f_ε(e) = (2/σ)·φ(e/σ)·Φ(−s·λ·e/σ), σ² = σ_u² + σ_v², λ = σ_u/σ_v.
# With r_it = y_it − x_it'β and the within deviations ẽ_it = r_it − r̄_i, the joint density of the first T_i − 1
# deviations is
# L_i = T_i · ∫ Π_t f_ε(ẽ_it + c) dc,
# where c is the common shift that demeaning removes and T_i is the absolute value of the Jacobian determinant of
# the map from the first T_i − 1 deviations and the mean back to (ε_i1, …, ε_iT_i). The package's log-likelihood of
# a firm is the density of the first T_i − 1 deviations, so the factor is T_i; the Wang–Ho likelihood, the density
# of the deviations in orthonormal coordinates, has √T_i instead. The reference computes the integral over c by
# adaptive Gauss-Kronrod quadrature, written in this file without package code, and is compared with the closed form
# of the :MLE backend firm by firm and with the log-likelihood of held :MLE fits through the public API. On three
# firms the double-precision integral is checked against the same integral in 256-bit arithmetic.
# With the environment variable SFRONTIERS_REFERENCE_REPORT=1 the file also prints a table of the differences.
using Test
using SFrontiers
using SpecialFunctions
using Random
using QuadGK
using Printf
using Optim: Newton

@testset "Reference test: the Chen-Schmidt-Wang fixed-effects panel likelihood against one-dimensional quadrature" begin
    CSM = SFrontiers.MLE_Backend

    # One dataset: the generator restarts for each dataset, so the datasets of a panel share x, α and the standard
    # normal draws behind v and u. Rows are firm-major; u_it is half normal, independent over i and t.
    function _csr_data(seed, firm, σu, σv, s)
        rng = Xoshiro(seed)
        nf, nt = maximum(firm), length(firm)
        xs = randn(rng, nt)
        αs = randn(rng, nf)
        vs = σv .* randn(rng, nt)
        us = σu .* abs.(randn(rng, nt))
        return (y = αs[firm] .+ 0.5 .* xs .+ vs .- s .* us, x = xs, firm = firm)
    end

    # The parameters of θ = [β; ln σ_u²; ln σ_v²].
    _csr_par(θ) = (β = θ[1], σu = exp(θ[2] / 2), σv = exp(θ[3] / 2))
    # Firm i's within deviations ẽ at θ.
    function _csr_firm(d, θ, i)
        rows = findall(==(i), d.firm)
        r = d.y[rows] .- _csr_par(θ).β .* d.x[rows]
        return r .- sum(r) / length(r)
    end

    # log φ(z), and log Φ(z) from the logarithm of erfc, accurate far in the lower tail (BigFloat: MPFR's erfc).
    _csr_lphi(z) = -z^2 / 2 - log(2 * oftype(z, π)) / 2
    _csr_lPhi(z) = z isa BigFloat ? log(erfc(-z / sqrt(big(2))) / 2) : logerfc(-z / sqrt(2)) - log(2)
    # log f_ε(e) of the composed error ε = v − s·u.
    _csr_lf(e, s, σ, λ) = log(oftype(e, 2)) - log(σ) + _csr_lphi(e / σ) + _csr_lPhi(-s * λ * e / σ)

    # g(c) = Σ_t log f_ε(ẽ_t + c) is concave; its peak c★ by golden-section search, M the largest g on a grid around
    # c★, and the breakpoints: around c★ on the scale σ/√T of the normal factors, and at c = −ẽ_t, where the Φ factor
    # of period t turns (sharply when λ is large).
    function _csr_nodes(e, s, σu, σv)
        nT = length(e)
        σ = sqrt(σu^2 + σv^2)
        λ = σu / σv
        g = c -> sum(_csr_lf(e[t] + c, s, σ, λ) for t in 1:nT)
        a, b = -maximum(abs, e) - 20σ, maximum(abs, e) + 20σ
        q = (sqrt(5) - 1) / 2
        c1, c2 = b - q * (b - a), a + q * (b - a)
        g1, g2 = g(c1), g(c2)
        for _ in 1:80
            if g1 < g2
                a, c1, g1 = c1, c2, g2
                c2 = a + q * (b - a)
                g2 = g(c2)
            else
                b, c2, g2 = c2, c1, g1
                c1 = b - q * (b - a)
                g1 = g(c1)
            end
        end
        cstar = (a + b) / 2
        w = σ / sqrt(nT)
        M = maximum(g(cstar + k * min(w, σ / λ) / 4) for k in -16:16)
        bps = sort(unique(vcat([cstar + k * w for k in (-8, -4, -2, -1, 0, 1, 2, 4, 8)], -e)))
        return g, M, bps
    end
    # The reference log L_i = log T_i + M + log ∫ exp(g(c) − M) dc over (−∞, ∞), rtol 1e-12.
    function _csr_logref(e, s, σu, σv)
        g, M, bps = _csr_nodes(e, s, σu, σv)
        I, _ = quadgk(c -> exp(g(c) - M), -Inf, bps..., Inf; rtol = 1e-12)
        return log(length(e)) + M + log(I)
    end
    # The same integral in BigFloat arithmetic at the working precision, with the nodes of the double-precision search.
    function _csr_logref_big(e, s, σu, σv)
        _, M, bps = _csr_nodes(e, s, σu, σv)
        E = big.(e)
        S = sqrt(big(σu)^2 + big(σv)^2)
        Λ = big(σu) / big(σv)
        g = c -> sum(_csr_lf(E[t] + c, s, S, Λ) for t in eachindex(E))
        I, _ = quadgk(c -> exp(g(c) - big(M)), big(-Inf), big.(bps)..., big(Inf); rtol = 1e-20)
        return log(big(length(e))) + big(M) + log(I)
    end
    function _csr_refs(d, θ, s)
        p = _csr_par(θ)
        return [_csr_logref(_csr_firm(d, θ, i), s, p.σu, p.σv) for i in 1:maximum(d.firm)]
    end

    # The :MLE closed form, −LL_T, of each firm alone, on the firm's demeaned data.
    _csr_po = (begx = 1, endx = 1, begw = 2, endw = 2, begv = 3, endv = 3)
    function _csr_mle(d, θ, s)
        out = zeros(maximum(d.firm))
        for i in eachindex(out)
            rows = findall(==(i), d.firm)
            nT = length(rows)
            out[i] = -CSM.LL_T(CSM.PFECSWH, CSM.sf_demean(d.y[rows]), CSM.sf_demean(reshape(d.x[rows], :, 1)),
                               nothing, nothing, nothing, nothing, s, nT, _csr_po, θ,
                               CSM.get_rowIDT(fill(1, nT)), nothing)
        end
        return out
    end

    # Panel (a) is balanced, 50 firms × 6 periods; panel (b) is unbalanced, 50 firms with 5 to 12 periods; panel (c)
    # is long, 50 firms × 50 periods and 50 firms × 100 periods, where the closed form's orthant integral is so small
    # that it is taken in log space.
    panels = [(label = "a", seed = 61, firm = repeat(1:50, inner = 6)),
              (label = "b", seed = 62, firm = vcat(fill.(1:50, repeat([5, 6, 7, 9, 12], 10))...)),
              (label = "c", seed = 63, firm = vcat(repeat(1:50, inner = 50), repeat(51:100, inner = 100)))]
    # (σ_u, σ_v): λ = 1.67, 0.5, 10 and 375; the last lies above λ = 300, where the closed form takes every firm's
    # orthant integral in log space (panels (a) and (c)).
    vpairs = [(σu = 0.5, σv = 0.3), (σu = 0.3, σv = 0.6), (σu = 1.0, σv = 0.1), (σu = 1.5, σv = 0.004)]
    types = [(label = "prod", name = "production", s = 1, type = :prod),
             (label = "cost", name = "cost", s = -1, type = :cost)]

    report, quad = String[], String[]
    @testset "the double-precision integral against 256-bit arithmetic" begin
        # firms with T_i = 5 (panel (b), firm 1), 6 (panel (a), firm 1) and 100 (panel (c), firm 51), production,
        # at the true θ of the first variance pair
        vp = vpairs[1]
        θ = [0.5, log(vp.σu^2), log(vp.σv^2)]
        p = _csr_par(θ)
        for (pn, i) in ((panels[2], 1), (panels[1], 1), (panels[3], 51))
            e = _csr_firm(_csr_data(pn.seed, pn.firm, vp.σu, vp.σv, 1), θ, i)
            v64 = _csr_logref(e, 1, p.σu, p.σv)
            v256 = setprecision(() -> _csr_logref_big(e, 1, p.σu, p.σv), BigFloat, 256)
            dq = Float64(abs(v64 - v256))
            @test dq <= 1e-12
            push!(quad, @sprintf("%-5s %-4d %.2e", pn.label, length(e), dq))
        end
    end

    hold = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,))
    for pn in panels, vp in vpairs, ty in types
        pn.label == "b" && vp == vpairs[4] && continue
        pn.label == "c" && ty.s == -1 && continue
        @testset "panel ($(pn.label)), (σu, σv) = ($(vp.σu), $(vp.σv)), $(ty.name)" begin
            d = _csr_data(pn.seed, pn.firm, vp.σu, vp.σv, ty.s)
            θ0 = [0.5, log(vp.σu^2), log(vp.σv^2)]
            public = pn.label == "a" && vp == vpairs[1]
            mle_max, pub = 0.0, NaN
            for θ in (θ0, θ0 .+ 0.05)
                ref = _csr_refs(d, θ, ty.s)
                diffs = abs.(_csr_mle(d, θ, ty.s) .- ref)
                @test maximum(diffs) <= 1e-9
                mle_max = max(mle_max, maximum(diffs))
                if public && θ == θ0
                    sp = sfmodel_spec(depvar = d.y, frontier = reshape(d.x, :, 1), id = d.firm, noise = :Normal,
                                      ineff = :HalfNormal, type = ty.type, datatype = :panel_TFE_CSW)
                    r = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE), init = θ, optim_options = hold,
                                    show_table = false, verbose = false, marginal = false)
                    @test r.coeff == θ
                    pub = abs(loglikelihood(r) - sum(ref))
                    @test pub <= 1e-8
                end
            end
            push!(report, @sprintf("%-5s %-4s %-13s %.2e     %s", pn.label, ty.label, "($(vp.σu), $(vp.σv))",
                                   mle_max, public ? @sprintf("%.2e", pub) : "-"))
        end
    end

    if get(ENV, "SFRONTIERS_REFERENCE_REPORT", "") == "1"
        @printf("%-5s %-4s %-13s %-12s %s\n", "panel", "type", "(σu, σv)", "closed form", "public")
        foreach(println, report)
        @printf("%-5s %-4s %s\n", "panel", "T_i", "double - 256-bit")
        foreach(println, quad)
    end
end
