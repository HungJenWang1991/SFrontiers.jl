# Regression tests: the CSW log-space orthant: gradient and Hessian up to the variance bound.
#
# LL_T(PFECSWH), the panel_TFE_CSW (Chen, Schmidt and Wang 2014) likelihood, adds for each firm
#     A + log B + T log 2,   B = ∫ φ(u) ∏ₜ Φ(bₜ - ρu) du,   bₜ = -(λ/σ)ϵₜ,  ρ = λ/√T,  λ = σᵤ/σᵥ.
# Above λ = 300 every firm takes log B from _csw_logorthant (log space). There each Φ factor
# turns from 1 to its tail within a few 1/ρ, and the peak of the integrand sits on the shoulder of the
# first turn. The value was right, but its ForwardDiff gradient and Hessian were not: the arguments
# bₜ - ρu* have partials of order ρ, whose contributions had to cancel in the derivatives, and the Dual
# rule of _tre_dlogcdf cancels for large positive arguments. On the four firms below (λ = 1.5e3, 4.0e4,
# 6.0e5 and 1.1e6) the gradient was off by 5.5e-12, 1.2e-8, 3.0e-7 and 1.3e-5 and the Hessian by
# 3.2e-11, 3.2e-8, 8.0e-6 and 3.6e-5 of their largest entries, and at the last three the Hessian's
# ln σᵥ² row by 7.8, 4.3e5 and 3.0e6 times its own largest true entry (2e-8 to 6e-12 there). The
# integral above 300 is now taken with its variable measured from the first turn, and
# piecewise at the turns; the four firms are then within about 1e-15.
#
# Reference: 512 bits, in log space, independent of the package's helpers. log B = log of the integral of
# exp(h(u)), h(u) = log φ(u) + Σₜ log Φ(bₜ - ρu), by a composite 20-node Gauss-Legendre rule on panels
# with breakpoints at u* ± w·2^j (w: the one-nat width of each side; j = -14..6), at u* ± 90w (h is
# concave, so it has fallen at least 90 nats there), and around every turn uₜ = bₜ/ρ inside that range
# where h has fallen less than 400 nats, at uₜ + k/ρ for k = 0, ±1/8, ..., ±64. log Φ is MPFR's erfc
# (its asymptotic series below -1000), with the Mills ratio as its derivative rule. The nodes are fixed
# from the values, so ForwardDiff over BigFloat gives the exact derivatives of the discrete sum. The rule
# was checked on these four firms against itself with 30 nodes on halved panels and against an
# independent 256-bit reference.
#
# The switch is one-sided: it compares ρ/γ with 300/√Tᵢ, so at σᵤ/σᵥ ≤ 300 the old lines run for every
# Tᵢ (a test of (ρ/γ)√Tᵢ > 300 instead would be true at σᵤ/σᵥ = 300.0 exactly for
# Tᵢ = 313 and 43 other Tᵢ ≤ 5000).
#
# Before the fix 15 of the 36 asserts fail: the four on the switch (the function did not exist), the
# gradient and Hessian asserts of all four firms, and the ln σᵥ² row asserts of the last three.

using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions: erfc
using StatsFuns: normlogpdf
using QuadGK: quadgk, gauss
using Printf


const _SF50 = SFrontiers.MLE_Backend.SFmle

# ---- the reference ----------------------------------------------------------------------------
_b50_val(x::Real) = x
_b50_val(x::ForwardDiff.Dual) = _b50_val(ForwardDiff.value(x))
const _B50C = Ref{Tuple{BigFloat,BigFloat}}()        # (1/sqrt(2), log sqrt(2 pi)) at the working precision
function _b50_lPhi(a::BigFloat)
    if a > 40
        return zero(a)                       # Phi(-40) < 4e-350
    elseif a >= 0
        return log1p(-erfc(a * _B50C[][1]) / 2)
    elseif a > -1000
        return log(erfc(-a * _B50C[][1]) / 2)
    else
        z = 1 / (a * a); s = big(1); term = big(1)
        for n in 1:40
            term = -term * (2n - 1) * z
            s += term
        end
        return -a * a / 2 - log(-a) - _B50C[][2] + log(s)
    end
end
# the Mills ratio phi(a)/Phi(a), the derivative of log Phi; above a = 40 it is below 1e-347 and taken as 0
_b50_mills(a) = _b50_val(a) > 40 ? zero(a) : exp(-a * a / 2 - _B50C[][2] - _b50_lPhi(a))
_b50_lPhi(d::ForwardDiff.Dual{T}) where {T} =
    ForwardDiff.Dual{T}(_b50_lPhi(ForwardDiff.value(d)), _b50_mills(ForwardDiff.value(d)) * ForwardDiff.partials(d))
_b50_h(u, b, rho) = -u * u / 2 - _B50C[][2] + sum(_b50_lPhi(x - rho * u) for x in b)

function _b50_breaks(b::Vector{BigFloat}, ρ::BigFloat)
    dh(u) = -u - ρ * sum(_b50_mills(x - ρ * u) for x in b)        # strictly decreasing
    lo = big(-1.0); hi = big(1.0)
    while dh(lo) <= 0; lo *= 2; end
    while dh(hi) >= 0; hi *= 2; end
    for _ in 1:2000
        m = (lo + hi) / 2
        (m == lo || m == hi) && break
        dh(m) > 0 ? (lo = m) : (hi = m)
        hi - lo <= big(10)^-60 * (1 + abs(m)) && break
    end
    u0 = (lo + hi) / 2; h0 = _b50_h(u0, b, ρ)
    function width(sg)
        drop(d) = h0 - _b50_h(u0 + sg * d, b, ρ)
        a, c = big(0.0), big(1.0)
        if drop(c) < 1
            while drop(c) < 1; a = c; c *= 2; end
        else
            while drop(c / 2) >= 1 && c > big(10)^-80; c /= 2; end
            a = c / 2
        end
        while c > big(1.005) * a
            m = sqrt(a * c); drop(m) < 1 ? (a = m) : (c = m)
        end
        return c
    end
    wL = width(-1); wR = width(1)
    L = u0 - 90wL; R = u0 + 90wR
    pts = BigFloat[L, u0, R]
    for j in -14:6
        p = u0 - wL * big(2.0)^j; L < p && push!(pts, p)
        p = u0 + wR * big(2.0)^j; p < R && push!(pts, p)
    end
    for x in b
        ut = x / ρ
        (L < ut < R && h0 - _b50_h(ut, b, ρ) < 400) || continue
        for k in (0, 0.125, -0.125, 0.25, -0.25, 0.5, -0.5, 1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64)
            p = ut + big(k) / ρ
            L < p < R && push!(pts, p)
        end
    end
    return (h0 = h0, pts = sort!(unique!(pts)))
end
function _b50_logB(b, ρ, s, x, w)
    acc = zero(_b50_h(s.pts[1], b, ρ))
    for j in 1:length(s.pts)-1
        A = s.pts[j]; C = s.pts[j+1]; half = (C - A) / 2; mid = (C + A) / 2
        for i in eachindex(x)
            acc += w[i] * half * exp(_b50_h(mid + half * x[i], b, ρ) - s.h0)
        end
    end
    return s.h0 + log(acc)
end
# the firm's log-likelihood at BigFloat θ = (β, ln σᵤ², ln σᵥ²), the variances clamped as in LL_T
function _b50_parts(θ, y, x)
    T = length(y); k = T - 1
    su2 = clamp(exp(θ[2]), big(1e-12), big(1e12)); sv2 = clamp(exp(θ[3]), big(1e-12), big(1e12))
    s2 = su2 + sv2; λ = sqrt(su2) / sqrt(sv2)
    e = big.(y) .- big.(x) .* θ[1]
    ea = e[1:k]
    A = -k / 2 * log(2 * big(pi)) - (k * log(s2) - log(big(k + 1))) / 2 - (sum(abs2, ea) + sum(ea)^2) / s2 / 2
    return A, -(λ / sqrt(s2)) .* e, λ / sqrt(big(T))
end
function _b50_ref(θ64, y, x)
    _B50C[] = (1 / sqrt(big(2)), log(sqrt(2 * big(pi))))
    θ = big.(θ64)
    A, b, ρ = _b50_parts(θ, y, x)
    s = _b50_breaks(b, ρ)
    xg, wg = gauss(BigFloat, 20)
    F = t -> (p = _b50_parts(t, y, x); p[1] + _b50_logB(p[2], p[3], s, xg, wg) + length(y) * log(big(2)))
    return F(θ), ForwardDiff.gradient(F, θ), ForwardDiff.hessian(F, θ)
end

# ---- the package --------------------------------------------------------------------------------
function _b50_idt(T)
    m = Matrix{Any}(undef, 1, 2); m[1, 1] = collect(1:T); m[1, 2] = T; m
end
_b50_F(θ, y, x) = -_SF50.LL_T(_SF50.PFECSWH, y, reshape(x, :, 1), nothing, nothing, nothing, nothing, 1, length(y),
                              (begx = 1, endx = 1, begw = 2, endw = 2, begv = 3, endv = 3), θ, _b50_idt(length(y)), nothing)

# an in-test copy of the earlier form of _csw_logorthant, statement for statement (the branch the package keeps at λ ≤ 300)
function _b50_old_logorthant(B, ρ, γ, logB_s25)
    u0, sL, sR = _SF50._tre_orthant_peak(B, ρ, γ)
    r   = ρ/γ
    a₀  = (B .- ρ*u0)./γ
    hu0 = normlogpdf(u0) + sum(_SF50._stable_normlogcdf, a₀)
    isfinite(_SF50._mlevalue(hu0)) || return hu0
    min(sL, sR) > 1e-10*(1 + abs(u0)) || return logB_s25
    q = min(sL, sR)/max(sL, sR)
    if q >= 0.2
        I = sL*quadgk(t -> exp(_SF50._tre_dh(a₀, r, u0, -sL*t)), 0, _SF50._TRE_SPAN, rtol = 1e-10, maxevals = 10^5)[1] +
            sR*quadgk(t -> exp(_SF50._tre_dh(a₀, r, u0,  sR*t)), 0, _SF50._TRE_SPAN, rtol = 1e-10, maxevals = 10^5)[1]
    else
        sw, sn = sL > sR ? (-sL, sR) : (sR, -sL)
        fw = t -> exp(_SF50._tre_dh(a₀, r, u0, sw*t))
        Iw = quadgk(fw, 0.0, q, rtol = 1e-10, maxevals = 10^5)[1]
        lo = q
        while lo < 1.0
            hi = min(lo*sqrt(10.0), 1.0)
            Iw += quadgk(fw, lo, hi, rtol = 1e-10, maxevals = 10^5)[1]
            lo = hi
        end
        Iw += quadgk(fw, 1.0, _SF50._TRE_SPAN, rtol = 1e-10, maxevals = 10^5)[1]
        In  = quadgk(t -> exp(_SF50._tre_dh(a₀, r, u0, sn*t)), 0, _SF50._TRE_SPAN, rtol = 1e-10, maxevals = 10^5)[1]
        I   = abs(sw)*Iw + abs(sn)*In
    end
    return hu0 + log(I)
end
# log B of one firm as LL_T forms its arguments, through `lo`
function _b50_logB_via(lo, θ, y, x)
    T = length(y)
    σᵤ² = clamp(exp(θ[2]), 1e-12, 1e12); σᵥ² = clamp(exp(θ[3]), 1e-12, 1e12)
    λ = sqrt(σᵤ²)/sqrt(σᵥ²); σ² = σᵤ² + σᵥ²
    ϵ = y .- x .* θ[1]
    return lo(-λ/sqrt(σ²)*ϵ, λ/sqrt(T), one(λ), log(1e-15))
end

@testset "CSW log-space orthant, gradient and Hessian up to the variance bound" begin
    # The measured differences against the 512-bit reference, printed under SFRONTIERS_REFERENCE_REPORT=1.
    b50_report = String[]

    # the four firms: (y, x, θ = (β, ln σᵤ², ln σᵥ²)); λ = σᵤ/σᵥ about 1.5e3, 4.0e4, 6.0e5 and 1.1e6 (the last
    # at ln σᵥ² = -27.5, inside the bound -27.63); residuals demeaned, as sfmodel_fit's within transformation
    # makes them, then rounded
    firms = [
        ([-0.659234, -0.831238, -0.09722, 1.450534, 0.137158], [0.307718, -1.796337, -1.108666, 1.650082, 0.947203], [0.48, 0.0, -14.6]),
        ([0.714452, -0.422033, 1.212058, -1.504477], [0.057719, -0.311536, 0.070684, 0.183134], [0.48, 0.0, -21.2]),
        ([-0.988762, -0.781846, -0.06537, 1.303856, -0.380566, 0.912689], [0.236845, -1.091345, 0.096868, 1.397959, -1.438861, 0.798534], [0.48, 0.0, -26.6]),
        ([-0.522928, 0.603573, -0.080645], [-0.603444, -0.395756, 0.9992], [0.49, 0.3, -27.5])]

    @testset "the switch at σᵤ/σᵥ = 300" begin
        @test (try _SF50._csw_past_switch(300.01 / sqrt(5), 1.0, 5) catch; false end)
        @test (try !_SF50._csw_past_switch(299.99 / sqrt(5), 1.0, 5) catch; false end)
        # one-sided: at σᵤ/σᵥ = 300.0 exactly, with ρ = λ/√Tᵢ and γ = 1 as LL_T forms them, never the new branch
        @test (try !_SF50._csw_past_switch(300.0 / sqrt(313), 1.0, 313) catch; false end)
        @test (try all(T -> !_SF50._csw_past_switch(300.0 / sqrt(T), 1.0, T) &&
                            !_SF50._csw_past_switch(prevfloat(300.0) / sqrt(T), 1.0, T), 1:5000) catch; false end)
    end

    @testset "_csw_logorthant unchanged at σᵤ/σᵥ ≤ 300 (T = 60; T = 313 at 300.0)" begin
        # LL_T calls _csw_logorthant at λ ≤ 300 only for a firm on the probability form's floor (B ≤ 1e-15); here the
        # function itself is called on a T = 60 firm (B from 5.6e-15 to 3.7e-13 at these λ)
        y = [0.31 * sin(1.7t) - 0.25 * cos(0.9t) for t in 1:60]; y .-= sum(y) / 60
        x = [cos(1.3t) for t in 1:60]; x .-= sum(x) / 60
        for lam in (1.0, 100.0, 299.9)
            θ = [0.5, log(lam^2 / (1 + lam^2)), log(1 / (1 + lam^2))]
            fn = t -> _b50_logB_via(_SF50._csw_logorthant, t, y, x)
            fo = t -> _b50_logB_via(_b50_old_logorthant, t, y, x)
            @test fn(θ) === fo(θ)
            @test ForwardDiff.gradient(fn, θ) == ForwardDiff.gradient(fo, θ)
            @test ForwardDiff.hessian(fn, θ) == ForwardDiff.hessian(fo, θ)
        end
        # σᵤ/σᵥ = 300.0 exactly as LL_T forms it, on a T = 313 firm, where (ρ/γ)√Tᵢ rounds above 300:
        # ln σᵤ² near log(300²) and ln σᵥ² at or just below 0, searched over adjacent floats
        y = [0.31 * sin(1.7t) - 0.25 * cos(0.9t) for t in 1:313]; y .-= sum(y) / 313
        x = [cos(1.3t) for t in 1:313]; x .-= sum(x) / 313
        is300(lu, lv) = sqrt(clamp(exp(lu), 1e-12, 1e12)) / sqrt(clamp(exp(lv), 1e-12, 1e12)) == 300.0
        l0 = log(300.0^2)
        θ = first([0.5, lu, lv] for k in 0:400 for lu in (l0 + k * eps(l0), l0 - k * eps(l0))
                  for lv in (-1e-17 * j for j in 0:200) if is300(lu, lv))
        fn = t -> _b50_logB_via(_SF50._csw_logorthant, t, y, x)
        fo = t -> _b50_logB_via(_b50_old_logorthant, t, y, x)
        @test fn(θ) === fo(θ)
        @test ForwardDiff.gradient(fn, θ) == ForwardDiff.gradient(fo, θ)
        @test ForwardDiff.hessian(fn, θ) == ForwardDiff.hessian(fo, θ)
    end

    @testset "value, gradient and Hessian against a 512-bit reference" begin
        setprecision(BigFloat, 512) do
            for (y, x, θ) in firms
                F = t -> _b50_F(t, y, x)
                v = F(θ); g = ForwardDiff.gradient(F, θ); H = ForwardDiff.hessian(F, θ)
                vr, gr, Hr = _b50_ref(θ, y, x)
                @test abs(v - vr) <= 1e-12 * max(1, abs(vr))
                @test maximum(abs.(g .- gr)) <= 1e-12 * maximum(abs.(gr))
                @test maximum(abs.(H .- Hr)) <= 1e-12 * maximum(abs.(Hr))
                # the ln σᵥ² row on its own scale (its largest true entry is 1.6e-5, 2.1e-8, 1.1e-10 and
                # 5.8e-12 at the four firms; after the fix the row's error is at most 2.4e-5 times that,
                # before the fix it was 7.8, 4.3e5 and 3.0e6 times it at the last three)
                @test maximum(abs.(H[3, :] .- Hr[3, :])) <= 1e-3 * maximum(abs.(Hr[3, :]))
                push!(b50_report, @sprintf("%-4d %-3d %-10.3g %.2e   %.2e   %.2e   %.2e", length(b50_report) + 1,
                    length(y), exp((θ[2] - θ[3]) / 2), Float64(abs(v - vr) / max(1, abs(vr))),
                    Float64(maximum(abs.(g .- gr)) / maximum(abs.(gr))),
                    Float64(maximum(abs.(H .- Hr)) / maximum(abs.(Hr))),
                    Float64(maximum(abs.(H[3, :] .- Hr[3, :])) / maximum(abs.(Hr[3, :])))))
            end
        end
    end

    @testset "beyond the bound of σᵥ²: the ln σᵥ² entries stay exactly zero" begin
        y, x, _ = firms[4]
        F = t -> _b50_F(t, y, x)
        θ = [0.49, 0.3, -30.0]
        g = ForwardDiff.gradient(F, θ); H = ForwardDiff.hessian(F, θ)
        @test isfinite(F(θ)) && F(θ) === F([0.49, 0.3, -40.0])
        @test all(isfinite, g) && g[3] == 0
        @test all(isfinite, H) && all(iszero, H[3, :]) && all(iszero, H[:, 3])
    end

    @testset "the width guard still returns the value passed in" begin
        # λ = 1e15: the peak's one-nat width is too narrow to place in Float64
        e = [-0.523, 0.604, -0.081, 0.212, -0.212]
        @test (try _SF50._csw_logorthant(-1e15 .* e, 1e15 / sqrt(5), 1.0, -34.5) === -34.5 catch; false end)
    end

    if get(ENV, "SFRONTIERS_REFERENCE_REPORT", "") == "1"
        println("value, gradient and Hessian against a 512-bit reference, each relative (tolerance 1e-12), and the "
                * "Hessian's ln σᵥ² row relative to its own largest entry (tolerance 1e-3)")
        @printf("%-4s %-3s %-10s %-10s %-10s %-10s %s\n", "firm", "T", "σᵤ/σᵥ", "value", "gradient", "Hessian",
                "ln σᵥ² row")
        foreach(println, b50_report)
    end
end
