# Regression tests: the cross-sectional MCI Gamma scale θ·√(1+k²).
#
# Cross-sectional models fitted by method = :MCI map each Halton draw t to an inefficiency
# u = s·(−log(1 − t)) (the Gamma default :expo_rule) with a per-observation scale s
# (MCI_Backend.get_scale_param). In an earlier form the Gamma scale was kθ: right
# where a Gamma with k > 1 puts its mass, but too narrow for k < 1, where the draws thin out faster than
# the Gamma's own tail exp(−u/θ) and an observation far in the tail gets almost no draws. Now the scale
# is θ·√(1+k²) (as in the panel), close to kθ for large k and to θ for small k, broadcast
# over a heteroscedastic k (CPU or CuArray; Float32 stays Float32).
#
# References: each observation's likelihood is a one-dimensional integral over u, computed by adaptive
# Gauss-Kronrod (QuadGK, rtol 1e-12) in s = log u, normalised at the peak (log space throughout), and
# E(u | e_i) as a ratio of two such integrals. On the four compared samples the sample log-likelihood
# agrees with a 256-bit computation to 5.7e-13 nats or better, and E(u | e_i) to 2.3e-13 relative,
# far inside the asserted tolerances (0.05 to 1 nat on 500
# observations). At k = 0.5 the test passes 8,192 draws, so that the bias every scale has at small k
# (the density spike at zero) stays well inside the tolerance. Derivatives are checked against central
# finite differences of the package's own likelihood evaluated in 256-bit BigFloat.
# Before the fix 13 of the 39 asserts fail: the 8 scale-value asserts (six homoscedastic values, the
# heteroscedastic vector and the derivative), the two k = 0.5 samples (kθ 13.9 and 34.1 nats too
# low), the heteroscedastic-k sample's scale vector and log-likelihood (14.9 nats too low), and the
# Float32 scale value.

using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions: loggamma
using Random
using Statistics
import Distributions
import QuadGK

const _MB49 = SFrontiers.MCI_Backend

# y = X b + v - u, u ~ Gamma(k_i, theta_i), k_i = exp(gk' z_i), theta_i = exp(gt' z_i), z_i = [1, z]
function _b49_data(gk, gt, sv, n, seed)
    rng = Xoshiro(seed)
    X = hcat(ones(n), randn(rng, n), randn(rng, n)); z = randn(rng, n); Z = hcat(ones(n), z)
    k = length(gk) == 1 ? fill(exp(gk[1]), n) : exp.(Z * gk)
    th = length(gt) == 1 ? fill(exp(gt[1]), n) : exp.(Z * gt)
    u = [rand(rng, Distributions.Gamma(k[i], th[i])) for i in 1:n]
    y = X * [2.0, 0.8, -0.3] .+ sv .* randn(rng, n) .- u
    return (y = y, X = X, Z = Z, k = k, th = th, sv = sv, n = n)
end
function _b49_isp(d, het; D = 1024, T = Float64)
    sp = het ? sfmodel_spec(depvar = T.(d.y), frontier = T.(d.X), zvar = T.(d.Z), noise = :Normal, ineff = :Gamma,
                            hetero = [:k, :theta], type = :prod) :
               sfmodel_spec(depvar = T.(d.y), frontier = T.(d.X), noise = :Normal, ineff = :Gamma, type = :prod)
    meth = _MB49.sfmodel_method(method = :MCI, n_draws = D, chunks = 10)
    return _MB49._assemble_MCI_spec(sp.mci_spec, meth)
end
function _b49_p(isp, gk, gt, sv)
    idx = isp.idx; p = zeros(length(isp.varnames))
    p[idx.beta] .= [2.0, 0.8, -0.3]; p[idx.ineff.k] .= gk; p[idx.ineff.theta] .= gt
    p[idx.noise.ln_sigma_v_sq] = log(sv^2)
    return p
end
# per-observation reference: log of the integral over s = log u, normalised at the peak; and E(u | e_i)
function _b49_ref(d)
    e = d.y .- d.X * [2.0, 0.8, -0.3]
    ll = 0.0; jl = zeros(d.n)
    grid = range(-60.0, 12.0; length = 14401)
    for i in 1:d.n
        k = d.k[i]; th = d.th[i]; c0 = -k * log(th) - loggamma(k) - log(d.sv) - 0.5 * log(2π)
        psi(s) = s > 12 ? -Inf : (u = exp(s); -(e[i] + u)^2 / (2 * d.sv^2) + k * s - u / th + c0)
        v = psi.(grid); vm, im = findmax(v)
        lo = grid[max(im - 1, 1)]; hi = grid[min(im + 1, length(grid))]; g = (sqrt(5) - 1) / 2
        for _ in 1:100
            a1 = hi - g * (hi - lo); a2 = lo + g * (hi - lo); psi(a1) < psi(a2) ? (lo = a1) : (hi = a2)
        end
        sp = (lo + hi) / 2; pm = max(psi(sp), vm)
        I, _ = QuadGK.quadgk(s -> exp(psi(s) - pm), -Inf, sp, Inf; rtol = 1e-12, atol = 0.0, maxevals = 10^7)
        J, _ = QuadGK.quadgk(s -> exp(psi(s) - pm + s), -Inf, sp, Inf; rtol = 1e-12, atol = 0.0, maxevals = 10^7)
        ll += pm + log(I); jl[i] = J / I
    end
    return (ll = ll, jlms = jl)
end
_b49_ll(isp, p) = -_MB49.MCI_nll(isp, p; chunks = 10)

# in-test copies of the earlier scale expressions of the families other than Gamma (unchanged)
const _b49_old = Dict(
    _MB49.Exponential_MCI => v -> sqrt.(v.sigma_sq), _MB49.HalfNormal_MCI => v -> v.sigma,
    _MB49.TruncatedNormal_MCI => v -> v.sigma_u, _MB49.Weibull_MCI => v -> v.lambda,
    _MB49.Lognormal_MCI => v -> v.sigma, _MB49.Lomax_MCI => v -> v.lambda .* (exp2.(1 ./ v.alpha) .- 1),
    _MB49.Rayleigh_MCI => v -> v.sigma)

@testset "cross-section MCI Gamma scale θ·√(1+k²)" begin

    @testset "scale values" begin
        for (k, th) in (([0.1], [2.0]), ([0.5], [1.3]), ([1.0], [0.7]), ([2.0], [0.25]), ([5.0], [0.1]), ([100.0], [1e-3]))
            s = _MB49.get_scale_param(_MB49.Gamma_MCI(), (k = k, theta = th, lgk = loggamma.(k)))
            @test s == th .* sqrt.(1 .+ k .^ 2) && eltype(s) == Float64
        end
        # heteroscedastic k and theta: element by element
        kv = [0.3, 0.8, 1.0, 2.5, 7.0]; tv = [1.5, 0.9, 0.4, 0.2, 0.05]
        s = _MB49.get_scale_param(_MB49.Gamma_MCI(), (k = kv, theta = tv, lgk = loggamma.(kv)))
        @test length(s) == 5 && all(s[i] === tv[i] * sqrt(1 + kv[i]^2) for i in 1:5)
        # ForwardDiff derivative of the scale
        @test ForwardDiff.derivative(k -> _MB49.get_scale_param(_MB49.Gamma_MCI(), (k = [k], theta = [0.3]))[1], 2.5) ≈
              0.3 * 2.5 / sqrt(1 + 2.5^2) rtol = 1e-14
    end

    @testset "k = 0.5 (500 observations): the old scale fails on these samples" begin
        # 8,192 draws. k*theta was 13.9 (seed 5) and 34.1 (seed 6) nats too low; theta*sqrt(1+k^2)
        # is within 0.21 on both
        for seed in (5, 6)
            d = _b49_data([log(0.5)], [0.0], 0.3, 500, seed)
            isp = _b49_isp(d, false; D = 8192)
            @test abs(_b49_ll(isp, _b49_p(isp, [log(0.5)], [0.0], 0.3)) - _b49_ref(d).ll) < 1.0
        end
    end

    @testset "k = 5 (500 observations): both scales close" begin
        d = _b49_data([log(5.0)], [log(0.2)], 0.3, 500, 1)
        isp = _b49_isp(d, false)
        p = _b49_p(isp, [log(5.0)], [log(0.2)], 0.3)
        ref = _b49_ref(d)
        @test abs(_b49_ll(isp, p) - ref.ll) < 0.05
        jb = _MB49.jlms_bc_indices(isp, p; chunks = 10)
        jl = jb isa NamedTuple ? jb.jlms : jb[1]
        @test maximum(abs.(jl ./ ref.jlms .- 1)) < 0.01
    end

    @testset "heteroscedastic k (0.3 to 1.6)" begin
        gk = [log(0.7), 0.3]; gt = [0.0, 0.2]
        d = _b49_data(gk, gt, 0.3, 500, 6)
        isp = _b49_isp(d, true; D = 8192)
        p = _b49_p(isp, gk, gt, 0.3)
        iv = _MB49.get_ineff_vals(isp.model.ineff, p, isp.idx, isp.zvar, isp.constants)
        s = _MB49.get_scale_param(isp.model.ineff, iv)
        @test length(s) == 500 && s == iv.theta .* sqrt.(1 .+ iv.k .^ 2)
        @test abs(_b49_ll(isp, p) - _b49_ref(d).ll) < 0.75      # k*theta: 14.9 nats too low
    end

    @testset "other families: scale unchanged bit for bit" begin
        r(n) = collect(range(0.3, 2.0; length = n))
        vals = Dict(_MB49.Exponential_MCI => (sigma_sq = r(4),), _MB49.HalfNormal_MCI => (sigma = r(4),),
                    _MB49.TruncatedNormal_MCI => (mu = r(4), sigma_u = r(4)), _MB49.Weibull_MCI => (lambda = r(4), k = r(4)),
                    _MB49.Lognormal_MCI => (mu = r(4), sigma = r(4)), _MB49.Lomax_MCI => (lambda = r(4), alpha = r(4) .* 20),
                    _MB49.Rayleigh_MCI => (sigma = r(4),))
        for (F, v) in vals, T in (Float64, Float32)
            vT = map(x -> T.(x), v)
            s = _MB49.get_scale_param(F(), vT)
            @test eltype(s) == T && s == _b49_old[F](vT)
        end
        for M in (2, 3, 4), T in (Float64, Float32)
            v = (k = fill(T(M), 3), theta = T.([0.1, 0.5, 2.0]), lgk = fill(T(0), 3))
            @test _MB49.get_scale_param(_MB49.Erlang_MCI{M}(), v) == v.theta .* M
        end
    end

    @testset "Float32" begin
        s32 = _MB49.get_scale_param(_MB49.Gamma_MCI(), (k = Float32[0.5, 3.0], theta = Float32[1.0, 0.2], lgk = Float32[0, 0]))
        @test eltype(s32) == Float32 && s32 == Float32[1.0, 0.2] .* sqrt.(1.0f0 .+ Float32[0.5, 3.0] .^ 2)
        d = _b49_data([log(5.0)], [log(0.2)], 0.3, 500, 1)
        i64 = _b49_isp(d, false); i32 = _b49_isp(d, false; T = Float32)
        p = _b49_p(i64, [log(5.0)], [log(0.2)], 0.3)
        l32 = _b49_ll(i32, Float32.(p)); l64 = _b49_ll(i64, p)
        @test l32 isa Float32 && isfinite(l32) && abs(l32 - l64) <= 1e-4 * abs(l64)
        dh = _b49_data([log(0.7), 0.3], [0.0, 0.2], 0.3, 200, 2)
        ih = _b49_isp(dh, true; T = Float32)
        ph = Float32.(_b49_p(ih, [log(0.7), 0.3], [0.0, 0.2], 0.3))
        ivh = _MB49.get_ineff_vals(ih.model.ineff, ph, ih.idx, ih.zvar, ih.constants)
        sh = _MB49.get_scale_param(ih.model.ineff, ivh)
        @test eltype(sh) == Float32 && isfinite(_b49_ll(ih, ph))
    end

    @testset "ForwardDiff gradient and Hessian against 256-bit finite differences" begin
        gk = [log(2.5), 0.3]; gt = [log(0.2), 0.1]
        d = _b49_data(gk, gt, 0.3, 20, 3)
        isp = _b49_isp(d, true; D = 64)
        p = _b49_p(isp, gk, gt, 0.3) .+ [0.02, -0.01, 0.03, 0.1, -0.2, 0.05, 0.1, 0.2]
        f(q) = _MB49.MCI_nll(isp, q; chunks = 1)
        g = ForwardDiff.gradient(f, p)
        H = ForwardDiff.hessian(f, p)
        gfd, Hfd = setprecision(BigFloat, 256) do
            pb = BigFloat.(p); n = length(p)
            e(j) = (v = zeros(BigFloat, n); v[j] = 1; v)
            h1 = BigFloat(10)^-20; h2 = BigFloat(10)^-15
            gg = [(f(pb .+ h1 .* e(j)) - f(pb .- h1 .* e(j))) / (2h1) for j in 1:n]
            HH = zeros(BigFloat, n, n)
            for i in 1:n, j in i:n
                HH[i, j] = (f(pb .+ h2 .* e(i) .+ h2 .* e(j)) - f(pb .+ h2 .* e(i) .- h2 .* e(j)) -
                            f(pb .- h2 .* e(i) .+ h2 .* e(j)) + f(pb .- h2 .* e(i) .- h2 .* e(j))) / (4 * h2^2)
                HH[j, i] = HH[i, j]
            end
            Float64.(gg), Float64.(HH)
        end
        @test maximum(abs.(g .- gfd)) <= 1e-9 * maximum(abs.(gfd))
        @test maximum(abs.(H .- Hfd)) <= 1e-8 * maximum(abs.(Hfd))
    end
end
