# Regression tests: the MCI truncated-normal CDF read by the copulas.
#
# MCI copula models read the inefficiency CDF F_u(u) (MCI_Backend.ineff_cdf! and its chunked twin
# ineff_cdf_chunk!). For the truncated normal it was formed as (Φ(x1) - Φ(x0))/max(1 - Φ(x0), 1e-15),
# x0 = -μ/σᵤ, x1 = (u - μ)/σᵤ, Φ = (1 + erf)/2. As μ/σᵤ falls, numerator and denominator cancel: F_u
# was off by 1e-4 at μ/σᵤ = -7, took coarse values near -8 and was 0 for every draw from about -8.3,
# so the Gaussian, Clayton, Clayton90 and Gumbel log-likelihoods were hundreds of nats too high near
# -8 and too low below (Gaussian, n = 300: +460 at -8.2, -478 at -9 against MSLE). For μ/σᵤ < -3 F_u
# is now 1 - Q(x1)/Q(x0) in log space (MCI_Backend._mci_tn_cdf_tail, a second pass after the old
# expression); at μ/σᵤ ≥ -3 the old expression is kept, bit for bit. When -μ/σᵤ is a Float32 (Float32
# coefficients; fits use Float64 ones) the switch is at -1, because the old expression's erf then runs
# in Float32. The second pass exists so that the GPU compiles the old broadcast into the same kernel as
# before (bit identity on the GPU); that can be checked only on a GPU, not here. These tests fail on the
# code before the fix.

using Test
using SFrontiers
using ForwardDiff
using SpecialFunctions
using Random

# 512-bit log upper tail log Q(x): MPFR erfc, the asymptotic series beyond x = 1e4
function _b28_lQ(x::BigFloat)
    x < 1e4 && return log(erfc(x / sqrt(BigFloat(2))) / 2)
    z = 1 / (x * x); s = BigFloat(1); t = BigFloat(1)
    for k in 1:60
        t *= -(2k - 1) * z; s += t
    end
    return -x^2 / 2 - log(x) - log(sqrt(2 * BigFloat(pi))) + log(s)
end
_b28_lphi(x::BigFloat) = -x^2 / 2 - log(sqrt(2 * BigFloat(pi)))
# F_u(u) = 1 - Q(x1)/Q(x0) in 512 bits
function _b28_F(u, mu, s)
    U = max(BigFloat(u), BigFloat(0)); M = BigFloat(mu); S = BigFloat(s)
    return -expm1(_b28_lQ((U - M) / S) - _b28_lQ(-M / S))
end
# the u at which F_u = r for μ = a·s (Newton on log Q in 512 bits), rounded to Float64
function _b28_tnq(a, r, s)
    x0 = -BigFloat(a); target = log(1 - BigFloat(r)) + _b28_lQ(x0)
    x1 = x0 + BigFloat(r) / max(abs(x0), BigFloat(1))
    for _ in 1:400
        x1n = x1 + (_b28_lQ(x1) - target) / exp(_b28_lphi(x1) - _b28_lQ(x1))
        abs(x1n - x1) < BigFloat(10)^-100 * max(BigFloat(1), abs(x1)) && return Float64((x1n - x0) * s)
        x1 = x1n
    end
    return Float64((x1 - x0) * s)
end
# Φ⁻¹ in 512 bits (Newton on erfc)
function _b28_qB(F::BigFloat)
    q = BigFloat(sqrt(2) * erfinv(2 * Float64(F) - 1))
    for _ in 1:20
        dq = (erfc(-q / sqrt(BigFloat(2))) / 2 - F) / exp(_b28_lphi(q))
        q -= dq
        abs(dq) < BigFloat(10)^-140 && break
    end
    return q
end

@testset "the MCI truncated-normal CDF read by the copulas" begin
    setprecision(BigFloat, 512) do
    MB = SFrontiers.MCI_Backend
    TN = MB.TruncatedNormal_MCI()
    cdf(u, mu, s) = (o = zeros(typeof(u), 1, 1);
                     MB.ineff_cdf!(TN, o, fill(u, 1, 1), (mu = [mu], sigma_u = [s]), (clamp_lo = oftype(u, 1e-15),)); o[1])
    cdfc(u, mu, s) = (o = zeros(typeof(u), 1, 1);          # chunked twin, second row of the parameter vectors
                      MB.ineff_cdf_chunk!(TN, o, fill(u, 1, 1), (mu = [zero(mu), mu], sigma_u = [one(s), s]),
                                          (clamp_lo = oftype(u, 1e-15),), 2, 2); o[1])

    @testset "F_u against 512-bit references, μ/σᵤ from -3.1 to -1e4" begin
        # at every point: the two kernels agree bit for bit, and the error |F_u - ref| in units of its tolerance
        # 2e-15 + 1e-11·ref
        same = Bool[]; err = BigFloat[]
        for a in (-3.1, -3.5, -5.0, -7.0, -7.9, -8.0, -8.2, -8.5, -12.0, -39.5, -40.0, -100.0, -1e3, -1e4),
            s in (1.0, 0.2), r in (1e-4, 0.1, 0.5, 0.9, 0.9999)
            u = _b28_tnq(a, r, s); ref = _b28_F(u, a * s, s)
            f1 = cdf(u, a * s, s); f2 = cdfc(u, a * s, s)
            push!(same, f1 === f2)
            push!(err, abs(f1 - ref) / (2e-15 + 1e-11 * ref))
        end
        @test all(same)
        @test maximum(err) <= 1                                         # was 1e-13 at -3.1, 0.9 (F = 0) from -8.3
        # no floor: at μ/σᵤ = -1e3 the smallest draws still get F_u > 0, relative to 512 bits
        u = _b28_tnq(-1e3, 1e-8, 1.0)
        @test abs(cdf(u, -1e3, 1.0) / _b28_F(u, -1e3, 1.0) - 1) <= 1e-12
    end

    @testset "Float32 arguments" begin
        # one concrete result type (a Union would not compile in a CuArray broadcast)
        for E in (Float32, Float64, ForwardDiff.Dual{Nothing, Float64, 2}, ForwardDiff.Dual{Nothing, Float32, 2})
            @test isdefined(MB, :_mci_tn_cdf_tail) && Base.return_types(MB._mci_tn_cdf_tail, NTuple{4, E}) == [E]
        end
        for a in (-2.0, -5.0, -8.5, -45.0), r in (0.1, 0.5, 0.9)
            u = Float32(_b28_tnq(a, r, 1.0)); mu = Float32(a)
            f32 = cdf(u, mu, 1f0)
            @test f32 isa Float32
            @test abs(f32 - _b28_F(u, mu, 1f0)) <= 4 * eps(1f0)       # was 0.15 at -5 and 1 (F = 0) from -6
        end
    end

    Φ = MB._normcdf
    Fold(u, mu, s, lo) = (Φ((max(u, zero(u)) - mu) / s) - Φ(-mu / s)) / max(1 - Φ(-mu / s), lo)

    @testset "where the switch is: μ/σᵤ = -3 (Float64), -1 (Float32)" begin
        # by the type of -μ/σᵤ; Duals (nested or not) follow their value type
        D64 = ForwardDiff.Dual{Nothing, Float64, 2}; D32 = ForwardDiff.Dual{Nothing, Float32, 2}
        for (E, sw) in ((Float64, 3), (D64, 3), (ForwardDiff.Dual{Nothing, D64, 2}, 3),
                        (Float32, 1), (D32, 1), (ForwardDiff.Dual{Nothing, D32, 2}, 1))
            @test isdefined(MB, :_mci_tn_switch) && MB._mci_tn_switch(E) == sw
        end
        # Float64: the old expression at -2.9 (bit for bit, in the next testset), the tail form from -3.1 on,
        # where the old expression is off by 1e-13 (and 1 at -100: F = 0)
        for a in (-3.1, -100.0)
            eo = 0.0; en = 0.0
            for s in (1.0, 0.2), r in (1e-4, 0.1, 0.5, 0.9, 0.9999)
                u = _b28_tnq(a, r, s); ref = _b28_F(u, a * s, s)
                en = max(en, Float64(abs(cdf(u, a * s, s) - ref))); eo = max(eo, Float64(abs(Fold(u, a * s, s, 1e-15) - ref)))
            end
            @test eo > 5e-14                                   # the point tells the two apart
            @test en <= 1e-14                                  # a few ulps; room for erfcx's last bits
        end
        # Float32: the old expression at -0.9 and -1 (next testset), the tail form, correctly rounded, from -1.1
        # on, where the old Float32 expression is off by 3.7 eps(1f0) (-1.1) to 310 eps(1f0) (-3.1)
        for a in (-1.1, -2.9, -3.1, -100.0)
            eo = 0.0; en = 0.0
            for r in (0.1, 0.5, 0.9)
                u = Float32(_b28_tnq(a, r, 1.0)); mu = Float32(a); ref = _b28_F(u, mu, 1f0)
                en = max(en, Float64(abs(cdf(u, mu, 1f0) - ref))); eo = max(eo, Float64(abs(Float32(Fold(u, mu, 1f0, 1f-15)) - ref)))
            end
            @test eo > 2 * eps(1f0)
            @test en <= eps(1f0)                               # about 1 ulp; room for the last bit
        end
    end

    @testset "the old expression is kept at μ/σᵤ ≥ -3 (Float32: ≥ -1), bit for bit" begin
        # per check, whether each point agrees bit for bit
        cdf_same = Bool[]; cdfc_same = Bool[]; grad_same = Bool[]; hess_same = Bool[]; f32_same = Bool[]
        for a in (3.0, 0.5, 0.0, -0.5, -0.999, nextfloat(-1.0), -1.0, -1.5, -2.0, -2.5, -2.9, -2.99),
            s in (0.2, 1.0), g in (1e-3, 0.3, 1.0, 4.0)
            u = s * g; mu = a * s
            push!(cdf_same, cdf(u, mu, s) === Fold(u, mu, s, 1e-15))
            push!(cdfc_same, cdfc(u, mu, s) === Fold(u, mu, s, 1e-15))
            f(p) = (σ = exp(p[2] / 2); o = zeros(eltype(p), 1, 1);
                    MB.ineff_cdf_chunk!(TN, o, fill(σ * g, 1, 1), (mu = [p[1]], sigma_u = [σ]), (clamp_lo = 1e-15,), 1, 1); o[1])
            fo(p) = (σ = exp(p[2] / 2); Fold(σ * g, p[1], σ, 1e-15))
            p0 = [mu, log(s^2)]
            push!(grad_same, ForwardDiff.gradient(f, p0) == ForwardDiff.gradient(fo, p0))
            push!(hess_same, ForwardDiff.hessian(f, p0) == ForwardDiff.hessian(fo, p0))
            if a >= -1
                u32 = Float32(u); mu32 = Float32(mu); s32 = Float32(s)
                push!(f32_same, cdf(u32, mu32, s32) === Float32(Fold(u32, mu32, s32, 1f-15)))
            end
        end
        @test all(cdf_same)
        @test all(cdfc_same)
        @test all(grad_same)
        @test all(hess_same)
        @test all(f32_same)
    end

    @testset "ForwardDiff derivatives through F_u and the Gaussian copula density, against 512 bits" begin
        # θ = (μ, ln σᵤ²), u = σᵤ·g (the default transformation is linear in σᵤ), F_v = 0.3, ρ = 0.999 tanh(0.5)
        c = MB.make_constants(MB.MCIModel(MB.NormalNoise_MCI(), TN, MB.GaussianCopula_MCI()), Float64)
        ρ = c.rho_max * tanh(0.5); Fv = 0.3
        function lc(Fu)                                    # the package's Gaussian copula kernel, one draw
            adj = zeros(typeof(Fu), 1, 1); Fvb = zeros(typeof(Fu), 1, 1)
            MB.copula_log_adjustment!(MB.GaussianCopula_MCI(), adj, Fvb, fill(sqrt(2) * erfinv(2Fv - 1), 1, 1),
                                      fill(Fu, 1, 1), MB.NormalNoise_MCI(), (inv_sigma_v = 1.0,), (rho = ρ,), c)
            return adj[1]
        end
        qvB = _b28_qB(BigFloat(Fv))
        function lcB(Fu::BigFloat)
            qv = qvB; qu = _b28_qB(Fu); P = BigFloat(ρ)
            return -log(1 - P^2) / 2 - (P^2 * (qv^2 + qu^2) - 2P * qv * qu) / (2 * (1 - P^2))
        end
        h = BigFloat(10)^-40
        # the largest error of each gradient and Hessian relative to the largest reference entry; the Hessians at
        # -45 and below (the series branch) apart, for their tighter tolerance
        gerr = BigFloat[]; herr = BigFloat[]; herr_series = BigFloat[]
        # -2.5: the old expression (kept above -3) is accurate there too; -45 and below: the series branch
        for a in (-2.5, -3.5, -8.0, -12.0, -45.0, -100.0, -1e3), r in (0.1, 0.5, 0.9)
            s0 = 0.2; g = _b28_tnq(a, r, s0) / s0; θ0 = [a * s0, log(s0^2)]
            F64(θ) = (σ = exp(θ[2] / 2); cdfc(σ * g, θ[1], σ))
            FB(θ) = (σ = exp(θ[2] / 2); _b28_F(σ * BigFloat(g), θ[1], σ))
            for (f, fb) in ((F64, FB), (θ -> lc(F64(θ)), θ -> lcB(FB(θ))))
                θ = BigFloat.(θ0); e1 = [h, zero(h)]; e2 = [zero(h), h]
                GR = [(fb(θ .+ e) - fb(θ .- e)) / (2h) for e in (e1, e2)]
                HR = [(fb(θ .+ ei .+ ej) - fb(θ .+ ei .- ej) - fb(θ .- ei .+ ej) + fb(θ .- ei .- ej)) / (4h^2)
                      for ei in (e1, e2), ej in (e1, e2)]
                G = ForwardDiff.gradient(f, θ0); H = ForwardDiff.hessian(f, θ0)
                push!(gerr, maximum(abs.(G .- GR)) / maximum(abs.(GR)))
                push!(a <= -45 ? herr_series : herr, maximum(abs.(H .- HR)) / maximum(abs.(HR)))
            end
        end
        @test maximum(gerr) <= 1e-11
        @test maximum(herr) <= 1e-8
        @test maximum(herr_series) <= 1e-11
    end

    @testset "fit level: the copula log-likelihood along μ/σᵤ" begin
        function tnpos(rng, mu, s, n)
            out = Float64[]
            while length(out) < n
                d = mu + s * randn(rng); d > 0 && push!(out, d)
            end
            return out
        end
        rng = Xoshiro(77); n = 300                       # a fixed 300-observation sample
        x = randn(rng, n); X = hcat(ones(n), x)
        v = 0.25 .* randn(rng, n)
        y = X * [1.0, 0.5] .+ v .- tnpos(rng, 0.2, 0.4, n)
        spec(cop) = sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :TruncatedNormal, copula = cop, type = :prod)
        θat(a, th) = [1.0, 0.5, a * 0.2, log(0.04), log(0.0625), th]
        # the log-likelihoods an MCI or MSLE fit evaluates (MCI_nll, qmc_nll, the objectives of sfmodel_fit),
        # at held coefficients, 1024 draws, default chunks (10: ineff_cdf_chunk!)
        function lls(cop, as, th; chunks = 10)
            sp = spec(cop)
            mc = MB._assemble_MCI_spec(sp.mci_spec, sfmodel_method(method = :MCI, n_draws = 1024, chunks = chunks))
            ms = SFrontiers.MSLE_Backend._assemble_MSLE_spec(sp.msle_spec, sfmodel_method(method = :MSLE, n_draws = 1024, chunks = chunks))
            return [-MB.MCI_nll(mc, θat(a, th); chunks = chunks) for a in as],
                   [-SFrontiers.MSLE_Backend.qmc_nll(ms, θat(a, th); chunks = chunks) for a in as]
        end
        as = (-5.0, -7.5, -8.2, -9.0, -12.0)
        # Gaussian copula along the path: the MCI and MSLE log-likelihoods agree within 3 nats at every
        # point, the MCI log-likelihood is smooth and falls at every step as μ/σᵤ goes from -5 to -12, and
        # the chunked kernel (chunks = 10) matches the unchunked one (chunks = 1) to 1e-9 nats at μ/σᵤ = -9
        lc, ls = lls(:Gaussian, as, 0.5)
        @test all(abs.(lc .- ls) .<= 3)
        @test all(diff(collect(lc)) .< 0)                  # smooth and falling along this path, as MSLE is
        lc1, _ = lls(:Gaussian, (-9.0,), 0.5; chunks = 1)   # the unchunked kernel ineff_cdf!
        @test abs(lc1[1] - lc[4]) <= 1e-9
        # Gumbel and Clayton90 at θρ = 0: MCI minus MSLE was +446 and -3746 at -8.2 and -8.5
        lc, ls = lls(:Gumbel, (-8.2, -8.5), 0.0)
        @test all(abs.(lc .- ls) .<= 3)
        lc, ls = lls(:Clayton90, (-8.2, -8.5), 0.0)
        @test all(abs.(lc .- ls) .<= 0.5)
        # Clayton at θρ = 0 (ρ ≈ 1; both simulators are far from draw-converged there, so no MSLE comparison):
        # the MCI log-likelihood no longer jumps (it was -319 at -8.2 and -2303 at -8.5)
        lc, _ = lls(:Clayton, (-8.2, -8.5), 0.0)
        @test abs(lc[2] - lc[1]) <= 5
    end

    end   # setprecision
end
