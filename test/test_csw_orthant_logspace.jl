# Regression tests: the CSW orthant probability and determinant in log space.
#
# LL_T(PFECSWH), the panel_TFE_CSW (Chen, Schmidt and Wang 2014) likelihood, adds for each firm
#     A = joe2014lnmvnpdf(ϵ[1:T-1], σ²)   and   log B,   B = ∫ φ(u) ∏ₜ Φ(bₜ - ρu) du,
# bₜ = -(λ/σ)ϵₜ, ρ = λ/√T. Both were formed as numbers before their logarithms were taken:
#   * B was floored, log(max(B, 1e-15)). B falls roughly geometrically in T (B → 2^-T as λ → 0),
#     so from T ≈ 45-60 the floor bound for some or all firms: +355 to +636 nats at T = 100 on
#     20-firm panels, and a likelihood that no longer depends on λ. At or below the floor log B
#     now comes from _csw_logorthant (log space, peak centred, with two guards and a breakpoint
#     layout for sharply one-sided peaks); above it the old expression is evaluated unchanged.
#   * A formed det = σ²^(T-1)/T, which leaves the Float64 range on long panels (log-likelihood
#     ±Inf), and whose ForwardDiff Hessian fails once det leaves about [7.5e-155, 1.6e154].
#     Outside [1e-154, 1e154] log det is now taken in closed form; inside, unchanged.
#   * σᵤ² and σᵥ² are now clamped to [1e-12, 1e12] inside LL_T(PFECSWH), as in jlmsbc(PFECSWH)
#     and LL_T(PTREH), so that a trial step that underflows σᵥ² no longer throws.
#
# Reference: 512 bits, in log space. log B is a trapezoid of exp(h(u) - h(u*)),
# h(u) = log φ(u) + ∑ₜ log Φ(bₜ - ρu), over u* ± 14 with step 0.5/√(1 + Tρ²); h is concave with
# -h'' ≥ 1 and the integrand is entire, so against half the step, a span of ±20 and no node
# skipped it differs by at most 1e-34 on firms like these. For a sharply one-sided peak (large λ)
# a trapezoid would need 1e5 nodes, so there log B is a composite 20-node Gauss-Legendre rule on
# panels graded geometrically away from u* (refined to 32 nodes on panels half as wide in log
# distance, it moves by at most 4e-24 on five firms with T = 5-100 and λ = 1 to 4e4). Both grids
# are fixed from Float64 constants, so ForwardDiff over BigFloat gives exact derivatives of the
# discrete sums. A uses the closed-form log determinant.
# Before the fix 17 of the 27 asserts fail, in every testset. In the far-off trial points (which
# guard against the exceptions of a first version of this fix) only the assert on _csw_logorthant
# fails, as that function does not exist before the fix.

using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions
using StatsFuns
using QuadGK
using Random
using Statistics
using LinearAlgebra
using Optim
using Printf

const _SF29 = SFrontiers.MLE_Backend.SFmle

_b29_val(x::Real) = Float64(x)
_b29_val(x::ForwardDiff.Dual) = _b29_val(ForwardDiff.value(x))

# peak of h by bisection on the analytic h'(u) = -u - ρ ∑ₜ φ(aₜ)/Φ(aₜ) (Float64)
function _b29_peak(b, ρ)
    lam(a) = exp(normlogpdf(a) - normlogcdf(a))
    dh(u) = -u - ρ * sum(lam(x - ρ * u) for x in b)
    lo, hi = -1.0, 0.0
    while dh(lo) <= 0; lo *= 2; end
    for _ in 1:200
        m = (lo + hi) / 2
        (m == lo || m == hi) && break
        dh(m) > 0 ? (lo = m) : (hi = m)
    end
    return (lo + hi) / 2
end

# 512-bit log Φ
_b29_logPhi(a) = log(erfc(-a / sqrt(big(2))) / 2)

# 512-bit log B (trapezoid); b and ρ are BigFloat or Duals over BigFloat. Nodes where a Float64 h
# is more than 90 nats below its peak are skipped (together they carry less than 1e-37 of the sum).
function _b29_logB(b, ρ)
    bv = _b29_val.(b); ρv = _b29_val(ρ)
    u0 = _b29_peak(bv, ρv)
    s = 0.5 / sqrt(1 + length(b) * ρv^2)
    n = ceil(Int, 14 / s)
    h64(u) = normlogpdf(u) + sum(normlogcdf(x - ρv * u) for x in bv)
    ks = [k for k in -n:n if h64(u0 + k * s) > h64(u0) - 90]
    U0 = big(u0); S = big(s)
    h(u) = -u^2 / 2 - log(sqrt(2 * big(pi))) + sum(_b29_logPhi(x - ρ * u) for x in b)
    hs = [h(U0 + k * S) for k in ks]
    m = maximum(_b29_val.(hs))
    return m + log(sum(exp(x - m) for x in hs) * S)
end

# 512-bit log B for a sharply one-sided peak: 20-node Gauss-Legendre on panels [d, 2d] graded away
# from u* on each side, from 1/32 of the curvature scale 1/√(-h''(u*)) until h has dropped 110 nats
function _b29_logB_graded(b, ρ)
    bv = _b29_val.(b); ρv = _b29_val(ρ)
    u0 = _b29_peak(bv, ρv)
    lam(a) = exp(normlogpdf(a) - normlogcdf(a))
    c = 1 / sqrt(1 + ρv^2 * sum(lam(x - ρv * u0) * (x - ρv * u0 + lam(x - ρv * u0)) for x in bv))
    h64(u) = normlogpdf(u) + sum(normlogcdf(x - ρv * u) for x in bv)
    xg, wg = QuadGK.gauss(BigFloat, 20)
    h(u) = -u^2 / 2 - log(sqrt(2 * big(pi))) + sum(_b29_logPhi(x - ρ * u) for x in b)
    hs = []; ws = BigFloat[]
    for sg in (-1.0, 1.0)
        d, dn = 0.0, c / 32
        while true
            A = big(u0 + sg * d); C = big(u0 + sg * dn); half = (C - A) / 2; mid = (C + A) / 2
            append!(hs, [h(mid + half * xi) for xi in xg]); append!(ws, abs(half) .* wg)
            (h64(u0) - h64(u0 + sg * dn) > 110 || dn > 40) && break
            d, dn = dn, 2 * dn
        end
    end
    m = maximum(_b29_val.(hs))
    return m + log(sum(ws[i] * exp(hs[i] - m) for i in eachindex(hs)))
end

# the CSW log-likelihood (+, not LL_T's -) at par = [β; ln σᵤ²; ln σᵥ²], 512 bits
function _b29_ll(par, y, X, idt; logB = _b29_logB)
    K = size(X, 2)
    β = par[1:K]; su2 = exp(par[K+1]); sv2 = exp(par[K+2])
    σ2 = su2 + sv2; λ = sqrt(su2) / sqrt(sv2); σ = sqrt(σ2)
    ϵ = y .- X * β
    tot = zero(eltype(ϵ))
    for i in 1:size(idt, 1)
        ind = idt[i, 1]; T = idt[i, 2]; k = T - 1
        e = ϵ[ind]; ea = e[1:k]
        # joe2014lnmvnpdf: Σ⁻¹ = (I + 11')/σ², det Σ = σ²ᵏ/(k+1)
        A = -k / 2 * log(2 * big(pi)) - (k * log(σ2) - log(big(k + 1))) / 2 - (sum(abs2, ea) + sum(ea)^2) / σ2 / 2
        tot += A + logB(-(λ / σ) .* e, λ / sqrt(big(T))) + T * log(big(2))
    end
    return tot
end

# the pre-fix joe2014lnmvnpdf, statement for statement (bit-identity asserts)
function _b29_old_joe(x, s2)
    k = length(x)
    invSig = Matrix(I, k, k).*(1/s2) .+ (1/s2)
    detSig = (s2^k)/(k+1)
    res = -k/2*log(2π) - 0.5*log(detSig) - 0.5*x'*invSig*x
    return res
end

# within-transformed CSW panel: y = x β + (v - u) demeaned by firm, x demeaned by firm
function _b29_panel(rng, Ts, su2, sv2; β = 0.5)
    y = Float64[]; x = Float64[]; idt = Matrix{Any}(undef, length(Ts), 2)
    for (i, T) in enumerate(Ts)
        xi = randn(rng, T); xi .-= mean(xi)
        e = sqrt(sv2) .* randn(rng, T) .- sqrt(su2) .* abs.(randn(rng, T)); e .-= mean(e)
        idt[i, 1] = collect(length(y)+1 : length(y)+T); idt[i, 2] = T
        append!(y, β .* xi .+ e); append!(x, xi)
    end
    return y, reshape(x, :, 1), idt
end

const _b29_po = (begx = 1, endx = 1, begw = 2, endw = 2, begv = 3, endv = 3)

# firm i's B in the pre-fix probability form: the quantity the switch compares with 1e-15
function _b29_oldB(y, X, idt, p, i)
    λ = exp((p[2] - p[3]) / 2); σ = sqrt(exp(p[2]) + exp(p[3])); ϵ = y .- X * p[1:1]
    return quadgk(u -> normpdf(u) * prod(normcdf.(-(λ / σ) .* ϵ[idt[i, 1]] .- λ / sqrt(idt[i, 2]) * u)),
                  -Inf, Inf, rtol = 1e-8)[1]
end
_b29_nll(y, X, idt, p) = _SF29.LL_T(_SF29.PFECSWH, y, X, nothing, nothing, nothing, nothing,
                                    1, length(y), _b29_po, p, idt, nothing)

@testset "CSW orthant probability and determinant in log space" begin
    # The measured differences of the comparisons with the 512-bit reference, printed under
    # SFRONTIERS_REFERENCE_REPORT=1 (gradient and Hessian relative to the largest entry of the reference).
    b29_report = String[]
    _b29_row(case, what, d, tol) = @sprintf("%-56s %-19s %.2e   %s", case, what, Float64(d), tol)
    _b29_theta(p) = "θ = [" * join((@sprintf("%.4g", v) for v in p), ", ") * "]"
    setprecision(BigFloat, 512) do

    @testset "unbalanced panel straddling the floor" begin
        rng = MersenneTwister(2903)
        y, X, idt = _b29_panel(rng, [4, 30, 45, 55, 80, 120], 0.25, 0.25)
        p = [0.5, log(0.25), log(0.25)]
        # the three short firms keep the old form, the three long ones take the new one
        @test [_b29_oldB(y, X, idt, p, i) <= 1e-15 for i in 1:6] == [false, false, false, true, true, true]
        ref = _b29_ll(big.(p), y, X, idt)
        @test abs(-_b29_nll(y, X, idt, p) - ref) <= 1e-9
        g = ForwardDiff.gradient(q -> _b29_nll(y, X, idt, q), p)
        gref = -ForwardDiff.gradient(q -> _b29_ll(q, y, X, idt), big.(p))
        @test maximum(abs.(g .- gref)) <= 1e-8 * maximum(abs.(gref))
        push!(b29_report, _b29_row("unbalanced panel, 6 firms, T = 4 to 120, (0.25, 0.25)", "value",
                                   abs(-_b29_nll(y, X, idt, p) - ref), "1e-9"))
        push!(b29_report, _b29_row("unbalanced panel, 6 firms, T = 4 to 120, (0.25, 0.25)", "gradient",
                                   maximum(abs.(g .- gref)) / maximum(abs.(gref)), "1e-8"))
    end

    @testset "gradient and Hessian at T = 100" begin
        rng = MersenneTwister(2904)
        y, X, idt = _b29_panel(rng, [100], 1.0, 0.1)
        f = q -> _b29_nll(y, X, idt, q)
        for p in ([0.5, 0.0, log(0.1)], [0.4, -1.0, -1.5])
            @test _b29_oldB(y, X, idt, p, 1) <= 1e-15
            g = ForwardDiff.gradient(f, p)
            gref = -ForwardDiff.gradient(q -> _b29_ll(q, y, X, idt), big.(p))
            @test maximum(abs.(g .- gref)) <= 1e-10 * maximum(abs.(gref))
            push!(b29_report, _b29_row("one firm, T = 100, " * _b29_theta(p), "gradient",
                                       maximum(abs.(g .- gref)) / maximum(abs.(gref)), "1e-10"))
        end
        p = [0.4, -1.0, -1.5]
        H = ForwardDiff.hessian(f, p)
        Href = -ForwardDiff.hessian(q -> _b29_ll(q, y, X, idt), big.(p))
        @test maximum(abs.(H .- Href)) <= 1e-10 * maximum(abs.(Href))
        push!(b29_report, _b29_row("one firm, T = 100, " * _b29_theta(p), "Hessian",
                                   maximum(abs.(H .- Href)) / maximum(abs.(Href)), "1e-10"))
    end

    @testset "large λ: a sharply one-sided peak" begin
        # λ = σᵤ/σᵥ = 1e4, T = 20, slope one unit off: log B = -54.65, and the peak's narrow
        # width is 0.016 of its wide one. One adaptive rule over [0, 60] (the layout of the first
        # version of this fix) is 1.8e-6 off here.
        rng = MersenneTwister(2908)
        λ = 1e4; su2 = λ^2 / (1 + λ^2); sv2 = 1 / (1 + λ^2)
        y, X, idt = _b29_panel(rng, [20], su2, sv2)
        p = [-0.5, log(su2), log(sv2)]
        # MersenneTwister gives other numbers for the same seed on Julia 1.10 (its seeding changed in 1.11):
        # the data then do not put B on the floor, so this guard of the case applies from Julia 1.11 on.
        @test _b29_oldB(y, X, idt, p, 1) <= 1e-15 skip = VERSION < v"1.11"
        ref = _b29_ll(big.(p), y, X, idt; logB = _b29_logB_graded)
        @test abs(-_b29_nll(y, X, idt, p) - ref) <= 1e-10          # before the fix: B on the floor
        g = ForwardDiff.gradient(q -> _b29_nll(y, X, idt, q), p)
        gref = -ForwardDiff.gradient(q -> _b29_ll(q, y, X, idt; logB = _b29_logB_graded), big.(p))
        @test maximum(abs.(g .- gref)) <= 1e-8 * maximum(abs.(gref))
        push!(b29_report, _b29_row("one firm, T = 20, σᵤ/σᵥ = 1e4", "value",
                                   abs(-_b29_nll(y, X, idt, p) - ref), "1e-10"))
        push!(b29_report, _b29_row("one firm, T = 20, σᵤ/σᵥ = 1e4", "gradient",
                                   maximum(abs.(g .- gref)) / maximum(abs.(gref)), "1e-8"))
    end

    @testset "the determinant on long panels" begin
        # A alone, in θ = (ln σᵤ², ln σᵥ²), against the closed form -k/2 log 2π - (k log s2 - log(k+1))/2 - x'Σ⁻¹x/2
        Aref(x, s2) = (k = length(x); -k/2*log(2*big(pi)) - (k*log(s2) - log(big(k+1)))/2 - (sum(abs2, x) + sum(x)^2)/s2/2)
        relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
        for (T, su2, sv2, what) in ((200, 0.01, 0.01, :value),   # det = 1e-340: before, A = +Inf
                                    (200, 0.06, 0.04, :hessian), # det = 5e-202: before, Hessian Inf/NaN
                                    (300, 2.0, 2.0, :hessian))   # det = 3e177: before, H[1,1] 300 times too large
            s2 = su2 + sv2
            x = sqrt(s2) .* randn(MersenneTwister(T), T - 1)
            f = q -> _SF29.joe2014lnmvnpdf(x, exp(q[1]) + exp(q[2])); fr = q -> Aref(big.(x), exp(q[1]) + exp(q[2]))
            p = [log(su2), log(sv2)]
            if what === :value
                @test abs(f(p) - fr(big.(p))) <= 1e-10
                @test relerr(ForwardDiff.gradient(f, p), ForwardDiff.gradient(fr, big.(p))) <= 1e-12
            else
                @test relerr(ForwardDiff.hessian(f, p), ForwardDiff.hessian(fr, big.(p))) <= 1e-12
            end
        end
        # just inside the band (det = 1.2e-154 and 0.9e154, T = 100) the old expression is still
        # right and is kept bit for bit
        for ldet in (log(1.2e-154), log(0.9e154))
            k = 99; s2 = exp((ldet + log(k + 1)) / k)
            x = sqrt(s2) .* randn(MersenneTwister(k), k)
            f = q -> _SF29.joe2014lnmvnpdf(x, exp(q[1]) + exp(q[2])); fo = q -> _b29_old_joe(x, exp(q[1]) + exp(q[2]))
            p = [log(0.7s2), log(0.3s2)]
            @test f(p) === fo(p) && ForwardDiff.gradient(f, p) == ForwardDiff.gradient(fo, p) &&
                  ForwardDiff.hessian(f, p) == ForwardDiff.hessian(fo, p)
        end
        # the whole likelihood of a T = 200 firm with σᵤ² + σᵥ² = 0.02: before, -LL = -Inf
        rng = MersenneTwister(2906)
        y, X, idt = _b29_panel(rng, [200], 0.01, 0.01)
        p = [0.5, log(0.01), log(0.01)]
        ref = _b29_ll(big.(p), y, X, idt)
        @test abs(-_b29_nll(y, X, idt, p) - ref) <= 1e-9
        push!(b29_report, _b29_row("one firm, T = 200, (σᵤ², σᵥ²) = (0.01, 0.01)", "value",
                                   abs(-_b29_nll(y, X, idt, p) - ref), "1e-9"))
    end

    @testset "far-off trial points" begin
        rng = MersenneTwister(3171)
        y, X, idt = _b29_panel(rng, fill(5, 3), 1.0, 0.1)
        # the width guard: at λ = 1e15 firm 1's peak has a one-nat width of 3e-13 at |u*| = 2,
        # too narrow to place in Float64, so the integral is not attempted and the pre-fix value
        # passed in comes back (the first version of the fix integrated such peaks and threw a
        # DomainError, or ran for seconds)
        e = y[idt[1, 1]] .- 0.5 .* X[idt[1, 1], 1]
        b = -1e15 .* e; ρ = 1e15 / sqrt(5)
        u0, sL, sR = _SF29._tre_orthant_peak(b, ρ, 1.0)
        @test !(min(sL, sR) > 1e-10 * (1 + abs(u0)))
        @test (try _SF29._csw_logorthant(b, ρ, 1.0, -34.5) === -34.5 catch; false end)
        # both variances far below the data's scale: finite value and gradient, promptly
        p = [0.5, -40.0, -40.0]
        @test isfinite(_b29_nll(y, X, idt, p)) && all(isfinite, ForwardDiff.gradient(q -> _b29_nll(y, X, idt, q), p))
    end

    @testset "variances beyond [1e-12, 1e12]" begin
        # σᵤ² and σᵥ² are clamped to [1e-12, 1e12], as in jlmsbc(PFECSWH) and LL_T(PTREH). At
        # ln σᵥ² = -800, σᵥ² underflows to 0 and λ = σᵤ/σᵥ to Inf, and the probability-form quadgk
        # threw a DomainError (before the fix and in the fix without the clamp). Both points below
        # clamp σᵥ² to 1e-12, so they give the same value, with zero slope in ln σᵥ².
        rng = MersenneTwister(3171)
        y, X, idt = _b29_panel(rng, fill(5, 3), 1.0, 0.1)
        f = q -> _b29_nll(y, X, idt, q)
        v = try f([0.5, 0.0, -800.0]) catch; NaN end
        @test isfinite(v) && v === f([0.5, 0.0, -100.0])
        g = try ForwardDiff.gradient(f, [0.5, 0.0, -800.0]) catch; fill(NaN, 3) end
        @test all(isfinite, g) && g[3] == 0
        # a BFGS fit (30 firms, T = 45) that the code before the fix finishes with redflag 0 near the
        # truth; its path under the corrected likelihood reaches σᵥ² = 0, where the fix without the
        # clamp stopped with that DomainError
        rng = MersenneTwister(79)
        NT = 30 * 45; id = repeat(1:30, inner = 45)
        Xf = randn(rng, NT, 1); α = repeat(randn(rng, 30), inner = 45)
        yf = α .+ 0.5 .* Xf[:, 1] .+ sqrt(0.2) .* randn(rng, NT) .- sqrt(0.5) .* abs.(randn(rng, NT))
        sp = sfmodel_spec(depvar = yf, frontier = Xf, noise = :Normal, ineff = :HalfNormal, type = :prod,
                          datatype = :panel_TFE_CSW, id = id)
        r = try
            sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE), marginal = false, show_table = false,
                        verbose = false, optim_options = sfmodel_opt(main_solver = BFGS(),
                                                                     main_opt = (iterations = 500, g_abstol = 1e-6)))
        catch
            nothing
        end
        @test r !== nothing && r.converged && r.redflag == 0 &&
              abs(r.coeff[2] - log(0.5)) < 0.3 && abs(r.coeff[3] - log(0.2)) < 0.3   # (-0.72, -1.66)
    end

    @testset "a fit on a T = 150 panel recovers the truth" begin
        # 6 firms, σᵤ² = 1, σᵥ² = 0.1: at the truth every firm's B is 6e-37 to 4e-30.
        # Before the fix: ln σᵤ² = -1.97 with a standard error of 1.1e6, redflag 0.
        # Newton is capped at 30 iterations (it needs fewer than 20 here), so a broken gradient
        # fails in minutes rather than running to the default 200.
        rng = MersenneTwister(2993)
        Nf, T = 6, 150
        NT = Nf * T; id = repeat(1:Nf, inner = T)
        X = randn(rng, NT, 1); α = repeat(randn(rng, Nf), inner = T)
        y = α .+ 0.5 .* X[:, 1] .+ sqrt(0.1) .* randn(rng, NT) .- abs.(randn(rng, NT))
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal, type = :prod,
                          datatype = :panel_TFE_CSW, id = id)
        opt = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 200, g_abstol = 1e-3),
                          main_solver = Newton(), main_opt = (iterations = 30, g_abstol = 1e-7))
        r = sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE), optim_options = opt, marginal = false,
                        show_table = false, verbose = false)
        @test r.converged && r.redflag == 0
        @test abs(r.coeff[2] - 0.0) < 0.3 && abs(r.coeff[3] - log(0.1)) < 0.5   # (-0.10, -2.07)
        @test all(r.std_err[2:3] .< 0.3)                                         # (0.10, 0.16)
    end

    end # setprecision

    if get(ENV, "SFRONTIERS_REFERENCE_REPORT", "") == "1"
        println("the closed form against a 512-bit reference")
        @printf("%-56s %-19s %-10s %s\n", "case", "compared", "measured", "tolerance")
        foreach(println, b29_report)
    end
end
