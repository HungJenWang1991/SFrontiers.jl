# Regression tests: panel MCI scales: Gamma θ·√(1+k²) and the Lomax median.
#
# panel_TFE models fitted by method = :MCI map each Halton draw t to an inefficiency u = g(t, s) whose
# width is set by a per-family scale s (Panel_Backend.mypan_get_mci_scale), and reweight by the
# density. An earlier form of the panel scaled the Gamma draws by θ and the Lomax draws by λ, the scales
# that the cross-section had already dropped: the mass of Gamma(k, θ) sits near kθ, so
# for k > 1 a few draws decided each firm's integral and the log-likelihood came out far too low (at
# k = 5, by 1.6 to 13.9 nats on ten 60-firm panels of this file's design); λ is far wider than a Lomax
# with large α. Now the Gamma scale is θ·√(1+k²) and the Lomax scale the median λ(2^(1/α) − 1).
#
# References: each firm's likelihood is a one-dimensional integral over u*; the test computes it from
# its own within-firm sums by adaptive Gauss-Kronrod (QuadGK, rtol 1e-12) in s = log u, normalised at
# the peak (log space throughout), and jlms and bc as ratios of such integrals. At the asserted
# tolerances (0.05 and 0.25 nats on 60 firms) its error is negligible: on the four compared panels the
# panel log-likelihood agrees with a 256-bit computation to 1.7e-13 nats or better.
# Derivatives are checked against central finite differences of the package's own
# likelihood evaluated in 256-bit BigFloat.
# Before the fix 23 of the 59 asserts fail: 16 of the 17 scale values (all but the Lomax value at
# alpha = 1, where the median equals lambda); at k = 5 the log-likelihood (3.1 nats too low), jlms (up
# to 0.31 relative) and bc (up to 0.13); the k = 0.7 panel of seed 9 (3.3 nats); and the Lomax
# alpha = 100 log-likelihood (0.65 nats), jlms (0.38) and bc (0.064).

using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions: loggamma
using Random
using Statistics
import Distributions
import QuadGK

const _PB48 = SFrontiers.Panel_Backend

# Wang-Ho TFE: y_it = a_i + x_it'b + v_it - h_it u_i*, h_it = exp(delta z_it); sigma_v^2 = 0.09
function _b48_panel(fam::Symbol, shape::Real, scl::Real, N::Int, T::Int, seed::Int)
    rng = Xoshiro(seed)
    NT = N * T
    id = repeat(1:N, inner = T)
    X = randn(rng, NT, 2)
    z = randn(rng, NT)
    a = repeat(randn(rng, N), inner = T)
    us = fam === :Gamma ? rand(rng, Distributions.Gamma(shape, scl), N) :
         scl .* expm1.((-1 / shape) .* log1p.(-rand(rng, N)))       # Lomax(alpha, lambda) by inversion
    u = exp.(0.5 .* z) .* repeat(us, inner = T)
    y = a .+ X * [0.6, -0.4] .+ 0.3 .* randn(rng, NT) .- u
    return (y = y, X = X, z = z, id = id, N = N, T = T)
end
# parameter order of the panel backend: beta (2), delta (1), the two ineff parameters, ln sigma_v^2
_b48_p(fam, shape, scl) = fam === :Gamma ? [0.6, -0.4, 0.5, log(shape), log(scl), log(0.09)] :
                                           [0.6, -0.4, 0.5, log(scl), log(shape), log(0.09)]
_b48_spec(d, fam; T = Float64) = sfmodel_spec(depvar = T.(d.y), frontier = T.(d.X), zvar = T.(d.z), id = d.id,
                                             noise = :Normal, ineff = fam, type = :prod, datatype = :panel_TFE)
_b48_isp(sp; D = 1024) = _PB48._assemble_panel_spec(sp.panel_spec,
    SFrontiers.sfmodel_panel_method(method = :MCI, n_draws = D, chunks = 1))
_b48_ll(isp, p) = -_PB48.panel_nll(isp, p; chunks = 1)
_b48_iv(isp, p) = _PB48.get_panel_ineff_vals(isp.model.ineff, p, isp.idx, isp.constants)

# per-firm reference: own within sums, then log of the integral over s = log u, normalised at the peak.
# Returns the panel log-likelihood and, per observation, jlms = h_it E(u*_i | firm i) and
# bc = E(exp(-h_it u*_i) | firm i).
function _b48_ref(d, p, fam)
    A = zeros(d.N); B = zeros(d.N); C = zeros(d.N)
    for i in 1:d.N
        r = ((i - 1) * d.T + 1):(i * d.T)
        e = d.y[r] .- d.X[r, :] * p[1:2]; e = e .- mean(e)
        h = exp.(p[3] .* d.z[r]); h = h .- mean(h)
        A[i] = sum(abs2, e); B[i] = sum(e .* h); C[i] = sum(abs2, h)
    end
    sv2 = exp(p[6])
    # log f(u) + log u at u = exp(s) (the Jacobian of u = exp(s) included), written in s
    lfs = if fam === :Gamma
        k = exp(p[4]); th = exp(p[5]); c0 = -k * log(th) - loggamma(k)
        s -> k * s - exp(s) / th + c0
    else
        lam = exp(p[4]); al = exp(p[5]); c0 = log(al) - log(lam)
        s -> c0 - (al + 1) * log1p(exp(s) / lam) + s
    end
    lc = -0.5 * (d.T - 1) * (log(2π) + log(sv2))
    grid = range(-60.0, 12.0; length = 14401)
    tot = 0.0
    jl = zeros(d.N * d.T); bc = zeros(d.N * d.T)
    for i in 1:d.N
        psi(s) = s > 12 ? -Inf : (u = exp(s); -(A[i] + 2 * B[i] * u + C[i] * u^2) / (2 * sv2) + lfs(s))
        v = psi.(grid); vm, im = findmax(v)
        lo = grid[max(im - 1, 1)]; hi = grid[min(im + 1, length(grid))]
        g = (sqrt(5) - 1) / 2
        for _ in 1:100                                 # golden section between the grid neighbours
            a1 = hi - g * (hi - lo); a2 = lo + g * (hi - lo)
            psi(a1) < psi(a2) ? (lo = a1) : (hi = a2)
        end
        sp = (lo + hi) / 2; pm = max(psi(sp), vm)
        I, _ = QuadGK.quadgk(s -> exp(psi(s) - pm), -Inf, sp, Inf; rtol = 1e-12, atol = 0.0, maxevals = 10^7)
        tot += lc + pm + log(I)
        Eu, _ = QuadGK.quadgk(s -> exp(psi(s) - pm + s), -Inf, sp, Inf; rtol = 1e-12, atol = 0.0, maxevals = 10^7)
        for t in ((i - 1) * d.T + 1):(i * d.T)
            ht = exp(p[3] * d.z[t])
            jl[t] = ht * Eu / I
            Eb, _ = QuadGK.quadgk(s -> exp(psi(s) - pm - ht * exp(s)), -Inf, sp, Inf; rtol = 1e-12, atol = 0.0, maxevals = 10^7)
            bc[t] = Eb / I
        end
    end
    return (ll = tot, jlms = jl, bc = bc)
end

# in-test copies of the earlier scale expressions of the families other than Gamma and Lomax (unchanged)
_b48_old_scale(::_PB48.HalfNormal_Panel, iv) = iv.sigma_u
_b48_old_scale(::_PB48.TruncatedNormal_Panel, iv) = iv.sigma_u
_b48_old_scale(::_PB48.Exponential_Panel, iv) = sqrt(iv.sigma_sq)
_b48_old_scale(::_PB48.Weibull_Panel, iv) = iv.lambda
_b48_old_scale(::_PB48.Lognormal_Panel, iv) = iv.sigma
_b48_old_scale(::_PB48.Rayleigh_Panel, iv) = iv.sigma
_b48_old_scale(::_PB48.Erlang_Panel{M}, iv) where {M} = M * iv.theta

@testset "panel MCI scales, Gamma θ·√(1+k²) and Lomax median" begin

    @testset "scale values" begin
        for (k, th) in ((0.1, 2.0), (0.5, 1.3), (1.0, 0.7), (2.0, 0.25), (5.0, 0.1), (100.0, 1e-3))
            iv = (k = k, theta = th, lgk = loggamma(k))
            @test _PB48.mypan_get_mci_scale(_PB48.Gamma_Panel(), iv) === sqrt(1 + k^2) * th
        end
        for (al, lam) in ((0.1, 0.5), (1.0, 2.0), (2.0, 1.0), (5.0, 3.0), (100.0, 50.0))
            iv = (lambda = lam, alpha = al)
            @test _PB48.mypan_get_mci_scale(_PB48.Lomax_Panel(), iv) === lam * (exp2(1 / al) - 1)
        end
        # through the parameter vector, as the likelihood and panel_jlms_bc call it
        dg = _b48_panel(:Gamma, 2.0, 0.25, 8, 3, 1); ig = _b48_isp(_b48_spec(dg, :Gamma); D = 16)
        pg = [0.6, -0.4, 0.5, log(3.7), log(0.15), log(0.09)]
        ivg = _b48_iv(ig, pg)
        @test _PB48.mypan_get_mci_scale(ig.model.ineff, ivg) === sqrt(1 + ivg.k^2) * ivg.theta
        dl = _b48_panel(:Lomax, 4.0, 1.5, 8, 3, 1); il = _b48_isp(_b48_spec(dl, :Lomax); D = 16)
        pl = [0.6, -0.4, 0.5, log(1.5), log(7.5), log(0.09)]
        ivl = _b48_iv(il, pl)
        @test _PB48.mypan_get_mci_scale(il.model.ineff, ivl) === ivl.lambda * (exp2(1 / ivl.alpha) - 1)
        # ForwardDiff derivative of the scale (the parameters enter only through k, theta, lambda, alpha)
        @test ForwardDiff.derivative(k -> _PB48.mypan_get_mci_scale(_PB48.Gamma_Panel(), (k = k, theta = 0.3, lgk = 0.0)), 2.5) ≈
              0.3 * 2.5 / sqrt(1 + 2.5^2) rtol = 1e-14
        @test ForwardDiff.derivative(a -> _PB48.mypan_get_mci_scale(_PB48.Lomax_Panel(), (lambda = 2.0, alpha = a)), 4.0) ≈
              -2.0 * exp2(1 / 4) * log(2) / 16 rtol = 1e-14
        # Float32 in, Float32 out
        s32g = _PB48.mypan_get_mci_scale(_PB48.Gamma_Panel(), (k = 3.0f0, theta = 0.2f0, lgk = 0.0f0))
        s32l = _PB48.mypan_get_mci_scale(_PB48.Lomax_Panel(), (lambda = 2.0f0, alpha = 30.0f0))
        @test s32g isa Float32 && s32g == sqrt(1.0f0 + 3.0f0^2) * 0.2f0
        @test s32l isa Float32 && s32l == 2.0f0 * (exp2(1.0f0 / 30.0f0) - 1.0f0)
    end

    @testset "Gamma k = 5 against QuadGK (60 firms, T = 6, 1,024 draws)" begin
        d = _b48_panel(:Gamma, 5.0, 0.1, 60, 6, 1)
        p = _b48_p(:Gamma, 5.0, 0.1)
        isp = _b48_isp(_b48_spec(d, :Gamma))
        ll = _b48_ll(isp, p)
        ref = _b48_ref(d, p, :Gamma)
        @test isfinite(ll)
        @test abs(ll - ref.ll) < 0.05                 # before the fix: 3.1 nats too low
        # panel_jlms_bc, which the fit and predict use, goes through the same scale
        jb = _PB48.panel_jlms_bc(isp, p; chunks = 1)
        @test maximum(abs.(jb.jlms ./ ref.jlms .- 1)) < 0.01      # before: up to 0.31 relative
        @test maximum(abs.(jb.bc .- ref.bc)) < 0.005               # before: up to 0.13
    end

    @testset "k < 1 guard and Lomax alpha = 100 against QuadGK" begin
        # k = 0.7, T = 8: on seed 9 the old scale theta is 3.3 nats off and k*theta (the cross-section's
        # earlier rule) 11.0; on seed 6 k*theta is 7.0 nats off while theta is close
        for seed in (6, 9)
            d = _b48_panel(:Gamma, 0.7, 0.5 / 0.7, 60, 8, seed)
            p = _b48_p(:Gamma, 0.7, 0.5 / 0.7)
            ll = _b48_ll(_b48_isp(_b48_spec(d, :Gamma)), p)
            @test abs(ll - _b48_ref(d, p, :Gamma).ll) < 0.25
        end
        # Lomax alpha = 100 (median 0.35), T = 24: the old scale lambda is 0.65 nats off on this panel
        lam = 0.35 / (exp2(1 / 100) - 1)
        d = _b48_panel(:Lomax, 100.0, lam, 60, 24, 7)
        p = _b48_p(:Lomax, 100.0, lam)
        isp = _b48_isp(_b48_spec(d, :Lomax))
        ll = _b48_ll(isp, p)
        ref = _b48_ref(d, p, :Lomax)
        @test isfinite(ll)
        @test abs(ll - ref.ll) < 0.05
        jb = _PB48.panel_jlms_bc(isp, p; chunks = 1)
        @test maximum(abs.(jb.jlms ./ ref.jlms .- 1)) < 0.01      # before: up to 0.38 relative
        @test maximum(abs.(jb.bc .- ref.bc)) < 0.005               # before: up to 0.064
    end

    @testset "other families: scale unchanged bit for bit" begin
        d = _b48_panel(:Gamma, 2.0, 0.25, 8, 3, 2)
        pars = Dict(:HalfNormal => [0.6, -0.4, 0.5, log(0.3)], :TruncatedNormal => [0.6, -0.4, 0.5, 0.2, log(0.3)],
                    :Exponential => [0.6, -0.4, 0.5, log(0.3)], :Weibull => [0.6, -0.4, 0.5, log(0.4), log(1.7)],
                    :Lognormal => [0.6, -0.4, 0.5, -1.0, log(0.5)], :Rayleigh => [0.6, -0.4, 0.5, log(0.3)],
                    :Erlang2 => [0.6, -0.4, 0.5, log(0.2)], :Erlang3 => [0.6, -0.4, 0.5, log(0.15)],
                    :Erlang4 => [0.6, -0.4, 0.5, log(0.1)])
        for fam in (:HalfNormal, :TruncatedNormal, :Exponential, :Weibull, :Lognormal, :Rayleigh,
                    :Erlang2, :Erlang3, :Erlang4)
            for T in (Float64, Float32)
                isp = _b48_isp(_b48_spec(d, fam; T = T); D = 16)
                p = T.(vcat(pars[fam], log(0.09)))
                iv = _b48_iv(isp, p)
                s = _PB48.mypan_get_mci_scale(isp.model.ineff, iv)
                @test s isa T && s === _b48_old_scale(isp.model.ineff, iv)
            end
        end
    end

    @testset "ForwardDiff gradient and Hessian against 256-bit finite differences" begin
        for (fam, shape, scl) in ((:Gamma, 2.5, 0.2), (:Lomax, 4.0, 0.35 / (exp2(1 / 4) - 1)))
            d = _b48_panel(fam, shape, scl, 12, 4, 3)
            isp = _b48_isp(_b48_spec(d, fam); D = 64)
            p = _b48_p(fam, shape, scl) .+ [0.03, -0.02, 0.05, 0.1, -0.1, 0.2]   # off the truth
            f(q) = _PB48.panel_nll(isp, q; chunks = 1)
            g = ForwardDiff.gradient(f, p)
            H = ForwardDiff.hessian(f, p)
            gfd, Hfd = setprecision(BigFloat, 256) do
                pb = BigFloat.(p); n = length(p)
                fb(q) = f(q)
                e(j) = (v = zeros(BigFloat, n); v[j] = 1; v)
                h1 = BigFloat(10)^-20; h2 = BigFloat(10)^-15
                gg = [(fb(pb .+ h1 .* e(j)) - fb(pb .- h1 .* e(j))) / (2h1) for j in 1:n]
                HH = zeros(BigFloat, n, n)
                for i in 1:n, j in i:n
                    HH[i, j] = (fb(pb .+ h2 .* e(i) .+ h2 .* e(j)) - fb(pb .+ h2 .* e(i) .- h2 .* e(j)) -
                                fb(pb .- h2 .* e(i) .+ h2 .* e(j)) + fb(pb .- h2 .* e(i) .- h2 .* e(j))) / (4 * h2^2)
                    HH[j, i] = HH[i, j]
                end
                Float64.(gg), Float64.(HH)
            end
            @test maximum(abs.(g .- gfd)) <= 1e-9 * maximum(abs.(gfd))
            @test maximum(abs.(H .- Hfd)) <= 1e-8 * maximum(abs.(Hfd))
        end
    end

    @testset "Float32 data: Float32 and finite" begin
        for (fam, shape, scl) in ((:Gamma, 5.0, 0.1), (:Lomax, 30.0, 0.35 / (exp2(1 / 30) - 1)))
            d = _b48_panel(fam, shape, scl, 40, 5, 4)
            p = _b48_p(fam, shape, scl)
            i64 = _b48_isp(_b48_spec(d, fam))
            i32 = _b48_isp(_b48_spec(d, fam; T = Float32))
            l32 = _b48_ll(i32, Float32.(p))
            l64 = _b48_ll(i64, p)
            @test l32 isa Float32 && isfinite(l32)
            @test abs(l32 - l64) <= 1e-4 * abs(l64)
            s32 = _PB48.mypan_get_mci_scale(i32.model.ineff, _b48_iv(i32, Float32.(p)))
            @test s32 isa Float32
            jb = _PB48.panel_jlms_bc(i32, Float32.(p); chunks = 1)
            @test eltype(jb.jlms) == Float32 && all(isfinite, jb.jlms) && all(>=(0), jb.jlms)
            @test eltype(jb.bc) == Float32 && all(x -> 0 < x <= 1, jb.bc)
        end
    end
end
