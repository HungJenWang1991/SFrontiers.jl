# Regression tests: CSW firms with σᵤ/σᵥ > 300 in log space; the boundary test on the variances.
#
# LL_T(PFECSWH), the panel_TFE_CSW (Chen, Schmidt and Wang 2014) likelihood, adds for each firm
#     A + log B + T log 2,   B = ∫ φ(u) ∏ₜ Φ(bₜ - ρu) du,   bₜ = -(λ/σ)ϵₜ,  ρ = λ/√T,  λ = σᵤ/σᵥ.
# (a) A firm whose probability-form B (quadgk over (-∞, ∞), rtol 1e-8) is at or below
#     1e-15 takes log B from _csw_logorthant (log space); in an earlier form a firm above that floor kept the
#     probability form whatever λ was. At large λ every Φ factor is a step of width about √T/λ in u,
#     and that quadgk can stop on a wrong answer: on the eight firms below (λ = 363 to 9.0e4) it is
#     off by 5.1e-7 to 0.19 nats. It fails in two narrow windows of B (about 2.4e-15 to 5e-15, just
#     above the floor, and about 7e-5 to 1e-4), most for firms with few periods, and by more the
#     larger λ is: in checks, by up to 0.2 nats per firm between
#     λ = 300 and 1000 and 0.19 above 1000, and below 300 by up to 0.07 near 300, 7e-3 at 200 and
#     2.4e-5 at 100. Now every firm takes the log-space branch once λ > 300; every evaluation
#     with λ ≤ 300 is unchanged, bit for bit, misses included: the two
#     firms of (100, 300] below, on which the probability form is off by 3.3e-8 and 9.3e-9 nats, are
#     asserted unchanged.
# (b) The likelihood clamps σᵤ² and σᵥ² to [1e-12, 1e12], but the earlier boundary test
#     for this model tested σ = exp(ln σ² / 2) against [1e-12, 1e12], i.e. |ln σ²| ≥ 55.26: a fit
#     that ended with |ln σ²| between 27.63 and 55.26, where the likelihood is flat in that variance,
#     got redflag 1 from its singular Hessian but no boundary message. The test is now on each
#     variance (|ln σ²| ≥ 27.63), as for panel TRE.
#
# Reference: 512 bits, in log space, independent of the package's helpers. log B = h(u*) + log of
# the integral of exp(h(u) - h(u*)), h(u) = log φ(u) + Σₜ log Φ(bₜ - ρu), by a composite 20-node
# Gauss-Legendre rule on panels with breakpoints at u* ± w·2^j (w: a one-nat width of each side, to
# within a factor 2; j = -10..6), at u* ± 80w (h is concave, so it has fallen at least 80 nats
# there), and around every step uₜ = bₜ/ρ inside that range, uₜ + k/ρ for k = 0, ±1/4, ..., ±32.
# The nodes are fixed from Float64 constants, so ForwardDiff over BigFloat gives the exact
# derivatives of the discrete sum. On the ten firms of the two lists below it agrees in value with an
# independent 256-bit adaptive Gauss-Kronrod reference to within 1.3e-27, and, against that
# reference's 128-bit fixed-rule ForwardDiff derivatives, to within 6e-27 relative in the gradient
# and 5e-23 in the Hessian.
# Each likelihood is differentiated as a callable object of its own type (_B36F below): ForwardDiff
# tags its Dual numbers with the type of the function, and a closure at every call site, a new type
# each, made Julia compile the derivatives of the likelihood again and again (most of this test's
# time before). The reference's value comes with its gradient, from one evaluation with Duals.
# Before the fix 25 of the 60 asserts fail: the 19 value, gradient and Hessian asserts of the eight
# firms above λ = 300 (the values are off by 5.1e-7 to 0.19 nats), the 2 at σᵤ/σᵥ = 300.01, the 3
# boundary-test asserts of PFECSWH beyond |ln σ²| = 27.63, and the boundary message.

using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions: erfc
using StatsFuns: normpdf, normcdf, normlogpdf, normlogcdf
using QuadGK: quadgk, gauss
using Random
using Statistics
using LinearAlgebra
using Optim
using Printf

const _SF36 = SFrontiers.MLE_Backend.SFmle

_b36_val(x::Real) = Float64(x)
_b36_val(x::ForwardDiff.Dual) = _b36_val(ForwardDiff.value(x))
_b36_logPhi(a) = _b36_val(a) > 0 ? log1p(-erfc(a / sqrt(big(2))) / 2) : log(erfc(-a / sqrt(big(2))) / 2)
_b36_h(u, b, ρ) = -u^2 / 2 - log(sqrt(2 * big(pi))) + sum(_b36_logPhi(x - ρ * u) for x in b)

# 512-bit log B; b and ρ are BigFloat or Duals over BigFloat (the grid comes from their values)
function _b36_logB(b, ρ)
    bv = _b36_val.(b); ρv = _b36_val(ρ)
    mills(a) = exp(normlogpdf(a) - normlogcdf(a))
    dh(u) = -u - ρv * sum(mills(x - ρv * u) for x in bv)      # decreasing; dh(0) < 0
    lo, hi = -1.0, 0.0
    while dh(lo) <= 0
        hi = lo; lo *= 2
    end
    for _ in 1:200
        m = (lo + hi) / 2
        (m == lo || m == hi) && break
        dh(m) > 0 ? (lo = m) : (hi = m)
    end
    u0 = (lo + hi) / 2
    h64(u) = normlogpdf(u) + sum(normlogcdf(x - ρv * u) for x in bv)
    function width(sg)                                        # within a factor 2 of the one-nat width
        d = 1.0
        if h64(u0) - h64(u0 + sg * d) < 1
            while h64(u0) - h64(u0 + sg * d) < 1
                d *= 2
            end
        else
            while d > 1e-300 && h64(u0) - h64(u0 + sg * d / 2) >= 1
                d /= 2
            end
        end
        return d
    end
    wL, wR = width(-1), width(1)
    pts = [u0 - 80wL, u0, u0 + 80wR]
    for j in -10:6
        push!(pts, u0 - wL * 2.0^j, u0 + wR * 2.0^j)
    end
    for x in bv, k in (0, 0.25, -0.25, 0.5, -0.5, 1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32)
        p = x / ρv + k / ρv
        (pts[1] < p < pts[3]) && push!(pts, p)
    end
    sort!(unique!(pts))
    xg, wg = gauss(BigFloat, 20)
    U0 = big(u0)
    h0 = _b36_h(U0, big.(bv), big(ρv))                        # a constant shift
    acc = zero(_b36_h(U0, b, ρ))
    for j in 1:length(pts)-1
        A = big(pts[j]); C = big(pts[j+1]); half = (C - A) / 2; mid = (C + A) / 2
        for i in eachindex(xg)
            acc += wg[i] * half * exp(_b36_h(mid + half * xg[i], b, ρ) - h0)
        end
    end
    return h0 + log(acc)
end

# one firm's contribution A + log B + T log 2 (+, not LL_T's -) at par = [β; ln σᵤ²; ln σᵥ²], 512 bits
function _b36_firm(par, y, X)
    K = size(X, 2); T = length(y); k = T - 1
    β = par[1:K]; su2 = exp(par[K+1]); sv2 = exp(par[K+2])
    s2 = su2 + sv2; λ = sqrt(su2) / sqrt(sv2); σ = sqrt(s2)
    e = big.(y) .- big.(X) * β; ea = e[1:k]
    A = -k / 2 * log(2 * big(pi)) - (k * log(s2) - log(big(k + 1))) / 2 - (sum(abs2, ea) + sum(ea)^2) / s2 / 2
    return A + _b36_logB(-(λ / σ) .* e, λ / sqrt(big(T))) + T * log(big(2))
end

# an in-test copy of the earlier form of the likelihood, statement for statement, for the bit-identity asserts
function _b36_s35_nll(Ỹ, X̃, po, rho, idt)
    β   = rho[1:po.endx]
    σᵤ² = clamp(exp(rho[po.begw]), 1e-12, 1e12)
    σᵥ² = clamp(exp(rho[po.begv]), 1e-12, 1e12)
    σᵤ = sqrt(σᵤ²)
    σᵥ = sqrt(σᵥ²)
    λ  =  σᵤ / σᵥ
    σ² =  σᵤ² + σᵥ²
    ϵ  = 1*(Ỹ - X̃ * β)
    nofid = size(idt,1)
    ll = Vector{eltype(ϵ)}(undef, nofid)
    for i in 1:nofid
           @views Tᵢ    = idt[i, 2]
           @views ind_a = idt[i, 1][1: Tᵢ-1]
           @views ind   = idt[i, 1]
           @views A     = _SF36.joe2014lnmvnpdf(ϵ[ind_a], σ²)
                  B     = quadgk(u -> normpdf(u)*prod(normcdf.(-λ/sqrt(σ²)*ϵ[ind].-λ/sqrt(Tᵢ)*u)), -Inf, Inf, rtol=1e-8)
           if B[1] > 1e-15
               @views ll[i] = A + log(B[1])  + log(2)*Tᵢ
           else
               @views ll[i] = A + _SF36._csw_logorthant(-λ/sqrt(σ²)*ϵ[ind], λ/sqrt(Tᵢ), one(λ),
                                                        log(max(B[1], 1e-15))) + log(2)*Tᵢ
           end
    end
    return -sum(ll)
end
# the same with every firm in the log-space branch (what the package does once σᵤ/σᵥ > 300)
function _b36_logspace_nll(Ỹ, X̃, po, rho, idt)
    β   = rho[1:po.endx]
    σᵤ² = clamp(exp(rho[po.begw]), 1e-12, 1e12)
    σᵥ² = clamp(exp(rho[po.begv]), 1e-12, 1e12)
    σᵤ = sqrt(σᵤ²)
    σᵥ = sqrt(σᵥ²)
    λ  =  σᵤ / σᵥ
    σ² =  σᵤ² + σᵥ²
    ϵ  = 1*(Ỹ - X̃ * β)
    nofid = size(idt,1)
    ll = Vector{eltype(ϵ)}(undef, nofid)
    for i in 1:nofid
           @views Tᵢ    = idt[i, 2]
           @views ind_a = idt[i, 1][1: Tᵢ-1]
           @views ind   = idt[i, 1]
           @views A     = _SF36.joe2014lnmvnpdf(ϵ[ind_a], σ²)
                  B     = quadgk(u -> normpdf(u)*prod(normcdf.(-λ/sqrt(σ²)*ϵ[ind].-λ/sqrt(Tᵢ)*u)), -Inf, Inf, rtol=1e-8)
               @views ll[i] = A + _SF36._csw_logorthant(-λ/sqrt(σ²)*ϵ[ind], λ/sqrt(Tᵢ), one(λ),
                                                        log(max(B[1], 1e-15))) + log(2)*Tᵢ
    end
    return -sum(ll)
end

const _b36_po = (begx = 1, endx = 1, begw = 2, endw = 2, begv = 3, endv = 3)
function _b36_idt(Ts)
    m = Matrix{Any}(undef, length(Ts), 2); s = 0
    for (i, T) in enumerate(Ts)
        m[i, 1] = collect(s+1:s+T); m[i, 2] = T; s += T
    end
    return m
end
_b36_nll(y, X, idt, p) = _SF36.LL_T(_SF36.PFECSWH, y, X, nothing, nothing, nothing, nothing,
                                    1, length(y), _b36_po, p, idt, nothing)
_b36_old(y, X, idt, p) = _b36_s35_nll(y, X, _b36_po, p, idt)
_b36_ls(y, X, idt, p)  = _b36_logspace_nll(y, X, _b36_po, p, idt)
_b36_ref(y, X, idt, p) = -_b36_firm(p, y, X)                  # the reference, as LL_T's sign
# a likelihood on one data set as a callable object: one type per likelihood function f, so ForwardDiff's
# Dual numbers have one type per likelihood and its derivatives are compiled once
struct _B36F{F}
    f::F
    y::Vector{Float64}
    X::Matrix{Float64}
    idt::Matrix{Any}
end
(L::_B36F)(p) = L.f(L.y, L.X, L.idt, p)
# the value and the gradient of f at x from one evaluation with Dual numbers
function _b36_valgrad(f, x::AbstractVector)
    n = length(x); V = eltype(x)
    D = ForwardDiff.Dual{typeof(ForwardDiff.Tag(f, V))}
    r = f([D(x[i], ntuple(j -> j == i ? one(V) : zero(V), n)) for i in 1:n])
    return ForwardDiff.value(r), [ForwardDiff.partials(r, j) for j in 1:n]
end
# parameters with σᵤ² + σᵥ² = 1 and σᵤ/σᵥ = λ
_b36_par(β, λ) = [β, log(λ^2 / (1 + λ^2)), log(1 / (1 + λ^2))]
# a within-transformed panel: x and v - u demeaned by firm; u half normal with σᵤ/σᵥ = λ0
function _b36_panel(rng, Ts, λ0; β = 0.5)
    sv = 1 / sqrt(1 + λ0^2); su = λ0 * sv
    y = Float64[]; x = Float64[]
    for T in Ts
        xi = randn(rng, T); xi .-= mean(xi)
        e = sv .* randn(rng, T) .- su .* abs.(randn(rng, T)); e .-= mean(e)
        append!(y, β .* xi .+ e); append!(x, xi)
    end
    return y, reshape(x, :, 1), _b36_idt(Ts)
end
# firm i's probability-form B (what the floor switch compares with 1e-15)
function _b36_oldB(y, X, idt, p, i)
    λ = sqrt(exp(p[2])) / sqrt(exp(p[3])); σ = sqrt(exp(p[2]) + exp(p[3])); ϵ = y .- X * p[1:1]
    return quadgk(u -> normpdf(u) * prod(normcdf.(-(λ / σ) .* ϵ[idt[i, 1]] .- λ / sqrt(idt[i, 2]) * u)),
                  -Inf, Inf, rtol = 1e-8)[1]
end

# Firms above the floor on which the probability form misses (inputs printed exactly with repr):
# (y, x, θ = [β; ln σᵤ²; ln σᵥ²]), with σᵤ² + σᵥ² = 1. Old-form error against the reference in the
# comment. Eight with σᵤ/σᵥ > 300: the first two have β = 0, so ϵ = y exactly; the other six come from
# a survey of simulated firms (the third repeats a B that the old quadgk returned at 16 surveyed firms,
# the fifth lies just above the floor, the sixth has the largest miss between 300 and 1000 among the
# survey's firms with B ≥ 1e-13, the eighth the lowest σᵤ/σᵥ at which the survey found a miss above 1e-6).
const _b36_firms = [
    # σᵤ/σᵥ = 90043: +1.0117e-3
    ([1.0684437593824079, -1.7620992301981544, 0.6936554708157467], [0.5, -0.3, -0.2],
     [0.0, -1.2333845057149855e-10, -22.8160885860522]),
    # σᵤ/σᵥ = 5206.9: -4.4414e-6
    ([-0.3088979403192471, 0.3088979403192474], [0.4, -0.4],
     [0.0, -3.688493012434108e-8, -17.115462786456767]),
    # σᵤ/σᵥ = 2955.8 (B = 9.4481e-5): -4.5179e-2
    ([0.08518058055944995, 1.7444324830057965, 1.1132459648100221, -2.9428590283752687],
     [-0.2069524635268273, -0.23307948078948054, 0.5183546889030848, -0.0783227445867768],
     [0.5, -1.144590935365935e-7, -15.983048396483278]),
    # σᵤ/σᵥ = 1879.2: -1.3612e-4
    ([0.19706215516860848, -0.19706215516860848], [-0.5494299879543363, 0.5494299879543363],
     [0.5, -2.8318989555275673e-7, -15.077148296940782]),
    # σᵤ/σᵥ = 1562.3 (B = 4.8957e-15, just above the floor): -1.9072e-1
    ([-3.143166682240734, 1.9038949090925774, -1.3012749880279844, -1.4354916122521186, 3.976038373428259],
     [0.06606943901192563, -0.027595217128876248, -0.4749984159112436, -0.6126646566810181, 1.0491888507092124],
     [0.5, -4.0972804349903634e-7, -14.707772410709294]),
    # σᵤ/σᵥ = 609.98 (B = 9.3910e-5): -3.2342e-3
    ([-2.042415551006977, -0.6633987349906899, 2.7058142859976666],
     [-0.6140059325167088, -0.4858995917593369, 1.0999055242760456],
     [0.5, -2.6876660810185884e-6, -12.82683871264644]),
    # σᵤ/σᵥ = 467.06: -5.0806e-7
    ([-0.4368178537265639, 1.2040780981161077, -0.2960555519107364, -0.4712046924788075],
     [-0.25940531883471596, 0.6787995977591943, 0.21082948793434814, -0.6302237668588266],
     [0.5, -4.584202668202442e-6, -12.292896659554103]),
    # σᵤ/σᵥ = 362.58: -1.9501e-6
    ([0.884634212176674, 0.4870088108987344, -0.12876514670074268, -0.7550989699492868, -0.7514924984300564,
      0.3620762574743604, 0.6846506720061074, -0.7830133374757903],
     [1.7131763207073443, -1.6852114388539021, 0.2699913723846453, -0.9622664705451389, -0.6236032356940028,
      0.23875569507987618, 0.826296238256818, 0.22286151866436024],
     [0.5, -7.6066515337405e-6, -11.786491294954965]),
]
# Two with σᵤ/σᵥ in (100, 300], where the probability form is kept, both from the survey: the one
# closest below 300 among its firms with B ≥ 1e-12, T ≤ 8 and a miss above 3e-9, and
# the largest miss below 300 among its firms with B ≥ 1e-13.
const _b36_firms_kept = [
    # σᵤ/σᵥ = 275.46: -9.3319e-9
    ([0.2118280748664374, -0.2118280748664374], [-0.5505816155760543, 0.5505816155760543],
     [0.5, -1.3178611742311141e-5, -11.236921954362511]),
    # σᵤ/σᵥ = 157.68: -3.2699e-8
    ([1.095174241258537, -1.095174241258537], [-0.2886566851394149, 0.288656685139415],
     [0.5, -4.022194982749335e-5, -10.121117806650565]),
]

@testset "CSW firms with σᵤ/σᵥ > 300 in log space; boundary test on the variances" begin
    # The measured differences of the firms above σᵤ/σᵥ = 300 against the 512-bit reference, printed under
    # SFRONTIERS_REFERENCE_REPORT=1.
    b36_report = String[]
    setprecision(BigFloat, 512) do

    @testset "unchanged wherever σᵤ/σᵥ ≤ 300" begin
        # the likelihood before the fix, statement for statement: value ===, gradient and Hessian ==,
        # at points up to σᵤ/σᵥ = 299.9, with firms on both sides of the 1e-15 floor
        rng = MersenneTwister(3601)
        for (Ts, λ0, pts, hess) in (([3, 5, 6, 8], 3.0, [(0.5, 3.0), (0.5, 99.9), (0.5, 299.9), (-1.5, 299.9)], true),)
            y, X, idt = _b36_panel(rng, Ts, λ0)
            f_new = _B36F(_b36_nll, y, X, idt)
            f_old = _B36F(_b36_old, y, X, idt)
            for (j, (β, λ)) in enumerate(pts)
                p = _b36_par(β, λ)
                @test f_new(p) === f_old(p)
                @test ForwardDiff.gradient(f_new, p) == ForwardDiff.gradient(f_old, p)
                hess && j in (1, length(pts)) && @test ForwardDiff.hessian(f_new, p) == ForwardDiff.hessian(f_old, p)
            end
            # the last point of each panel has firms on both sides of the floor
            p = _b36_par(pts[end]...)
            nfl = count(i -> _b36_oldB(y, X, idt, p, i) <= 1e-15, 1:length(Ts))
            # MersenneTwister gives other numbers for the same seed on Julia 1.10 (its seeding changed in 1.11):
            # the firms then fall on one side of the floor, so this guard applies from Julia 1.11 on.
            @test 0 < nfl < length(Ts) skip = VERSION < v"1.11"
        end
        # firms above the floor with σᵤ/σᵥ in (100, 300]: the probability form is kept, with its error
        for (k, (y, x, p)) in enumerate(_b36_firms_kept)
            T = length(y); X = reshape(x, :, 1); idt = _b36_idt([T])
            λ = sqrt(exp(p[2])) / sqrt(exp(p[3]))
            @test 100 < λ <= 300 && _b36_oldB(y, X, idt, p, 1) > 1e-15
            f_new = _B36F(_b36_nll, y, X, idt)
            f_old = _B36F(_b36_old, y, X, idt)
            @test f_new(p) === f_old(p)
            @test ForwardDiff.gradient(f_new, p) == ForwardDiff.gradient(f_old, p)
            @test ForwardDiff.hessian(f_new, p) == ForwardDiff.hessian(f_old, p)
        end
    end

    @testset "σᵤ/σᵥ > 300: firms above the floor in log space" begin
        # each of these firms took the probability form before the fix and was off by 5.1e-7 to
        # 0.19 nats, with gradients and Hessians off by more (see the list above)
        for (k, (y, x, p)) in enumerate(_b36_firms)
            T = length(y); X = reshape(x, :, 1); idt = _b36_idt([T])
            λ = sqrt(exp(p[2])) / sqrt(exp(p[3]))
            @test λ > 300 && _b36_oldB(y, X, idt, p, 1) > 1e-15
            f = _B36F(_b36_nll, y, X, idt)
            fref = _B36F(_b36_ref, y, X, idt)
            ref, gref = _b36_valgrad(fref, big.(p))
            @test abs(f(p) - ref) <= 1e-10
            g = ForwardDiff.gradient(f, p)
            @test maximum(abs.(g .- gref)) <= 1e-6 * maximum(abs.(gref))
            if k in (2, 4, 6)
                H = ForwardDiff.hessian(f, p)
                Href = ForwardDiff.hessian(fref, big.(p))
                @test maximum(abs.(H .- Href)) <= 1e-6 * maximum(abs.(Href))
            end
            dH = k in (2, 4, 6) ? @sprintf("%.2e", Float64(maximum(abs.(H .- Href)) / maximum(abs.(Href)))) : "-"
            push!(b36_report, @sprintf("%-4d %-3d %-10.4g %.2e   %.2e   %s", k, T, λ, Float64(abs(f(p) - ref)),
                                       Float64(maximum(abs.(g .- gref)) / maximum(abs.(gref))), dH))
        end
    end

    @testset "the switch is at σᵤ/σᵥ = 300" begin
        # just below, every firm above the floor keeps the probability form; just above, every firm
        # takes the log-space branch (before the fix: the probability form on both sides)
        rng = MersenneTwister(3603)
        y, X, idt = _b36_panel(rng, [3, 4, 5, 6, 8], 300.0)
        f = _B36F(_b36_nll, y, X, idt)
        for (λ, f_ref) in ((299.99, _B36F(_b36_old, y, X, idt)), (300.01, _B36F(_b36_ls, y, X, idt)))
            p = _b36_par(0.5, λ)
            @test all(i -> _b36_oldB(y, X, idt, p, i) > 1e-15, 1:5)
            @test f(p) === f_ref(p)
            @test ForwardDiff.gradient(f, p) == ForwardDiff.gradient(f_ref, p)
        end
    end

    @testset "the boundary test is on each variance, as the clamp" begin
        pos = (begx = 1, endx = 1, begw = 2, endw = 2, begv = 3, endv = 3)
        at(m, c) = _SF36._mle_scale_at_clamp(m, pos, c, nothing, nothing)
        # ln(1e-12) = -27.631: before the fix PFECSWH tested σ, true only from |ln σ²| ≥ 55.26
        @test !at(_SF36.PFECSWH, [0.5, 0.0, -27.6]) && !at(_SF36.PFECSWH, [0.5, 27.6, -27.6])
        @test at(_SF36.PFECSWH, [0.5, 0.0, -27.7])
        @test at(_SF36.PFECSWH, [0.5, 0.0, -40.0])
        @test at(_SF36.PFECSWH, [0.5, 27.7, 0.0])
        # the Wang-Ho panel model, which has no clamp, keeps the σ test; panel TRE tests its variances
        @test !at(_SF36.PFEWHH, [0.5, 0.0, -40.0]) && at(_SF36.PFEWHH, [0.5, 0.0, -55.3])
        posT = (begx = 1, endx = 1, begq = 2, endq = 2, begw = 3, endw = 3, begv = 4, endv = 4)
        @test _SF36._mle_scale_at_clamp(_SF36.PTREH, posT, [0.5, 0.0, 0.0, -27.7], nothing, nothing)
    end

    @testset "a CSW fit beyond the clamp prints the boundary message" begin
        # a fit held at a point (no warmstart, no Newton step); at ln σᵥ² = -40 the likelihood is flat
        # in σᵥ², its Hessian has a row of zeros, redflag is 1 and every standard error NaN before and
        # after the fix; only the boundary message is new there
        rng = MersenneTwister(3605)
        N, T = 10, 5; NT = N * T; id = repeat(1:N, inner = T)
        X = randn(rng, NT, 1); α = repeat(randn(rng, N), inner = T)
        y = α .+ 0.5 .* X[:, 1] .+ 0.01 .* randn(rng, NT) .- abs.(randn(rng, NT))
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal, type = :prod,
                          datatype = :panel_TFE_CSW, id = id)
        hold = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 0,),
                           main_solver = Newton(), main_opt = (iterations = 0,))
        path, io = mktemp()
        r = redirect_stdout(io) do
            sfmodel_fit(spec = sp, method = sfmodel_method(method = :MLE), init = [0.5, 0.0, -40.0],
                        optim_options = hold, marginal = false, show_table = false, verbose = false)
        end
        close(io)
        txt = read(path, String); rm(path)
        @test occursin("at or beyond the numerical bound", txt)
        @test r.redflag == 1 && all(isnan, r.std_err) && all(iszero, r.Hessian[end, :])
    end

    end # setprecision

    if get(ENV, "SFRONTIERS_REFERENCE_REPORT", "") == "1"
        println("firms with σᵤ/σᵥ > 300 against a 512-bit reference: the value absolute (tolerance 1e-10), the "
                * "gradient and the Hessian relative to their largest entry (tolerance 1e-6)")
        @printf("%-4s %-3s %-10s %-10s %-10s %s\n", "firm", "T", "σᵤ/σᵥ", "value", "gradient", "Hessian")
        foreach(println, b36_report)
    end
end
