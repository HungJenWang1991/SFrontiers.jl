# test_erlang.jl — regression tests for the Erlang{2,3,4} inefficiency distributions.
# Erlang(M, θ) = Gamma(shape=M, scale=θ) with the shape pinned
# at the integer order; θ = exp(Z'γ), hetero :theta, init keyword ln_theta. MCI uses
# :expo_rule with importance scale M·θ; MSLE draws via the parameter-free standardized
# Newton quantile θ·q_M(p) (Gamma stays MCI-only: its estimated shape k sits inside
# the quantile). All quantities are deterministic (seeded data + Halton draws), so the
# asserts are stable.
using Test
using SFrontiers
using Optim: NelderMead, Newton
import Distributions
import QuadGK
using Random, Statistics

const _MCIB_ERL = SFrontiers.MCI_Backend
const _MSLEB_ERL = SFrontiers.MSLE_Backend

@testset "Erlang{2,3,4} (MCI + MSLE)" begin

    # --- Spec construction: MCI and MSLE built, MLE skipped, labels correct ---
    Random.seed!(11)
    N0 = 80
    X0 = hcat(ones(N0), randn(N0))
    y0 = X0 * [1.0, 0.5] .- rand(Distributions.Gamma(2, 0.3), N0) .+ 0.2 .* randn(N0)
    for M in (2, 3, 4)
        sym = Symbol("Erlang", M)
        sp = sfmodel_spec(depvar=y0, frontier=X0, noise=:Normal, ineff=sym)
        @test sp.mci_spec !== nothing
        @test sp.msle_spec !== nothing
        @test sp.mle_spec === nothing
        @test sp.mci_spec.eqnames == ["frontier", "ln_θ", "ln_σᵥ²"]
        @test sp.msle_spec.eqnames == ["frontier", "ln_θ", "ln_σᵥ²"]
        @test length(sp.mci_spec.varnames) == 4      # K=2 frontier + ln_θ + ln_σᵥ²
    end

    # Gamma remains MCI-only (estimated shape k inside the quantile)
    sp_g = sfmodel_spec(depvar=y0, frontier=X0, noise=:Normal, ineff=:Gamma)
    @test sp_g.msle_spec === nothing

    # invalid hetero symbol errors cleanly (only :theta is valid for Erlang)
    @test_throws ErrorException sfmodel_spec(depvar=y0, frontier=X0,
        noise=:Normal, ineff=:Erlang2, hetero=[:sigma_u_sq])

    # --- Standardized MSLE quantile kernel vs Distributions ---
    for M in (2, 3, 4)
        lgm = _MSLEB_ERL.loggamma(Float64(M))
        ps = vcat(1e-8, 1e-6, collect(0.001:0.001:0.999), 1 - 1e-6, 1 - 1e-8)
        err = maximum(abs(_MSLEB_ERL.myErlangStdQuantile(p, M, lgm) -
                          Distributions.quantile(Distributions.Gamma(M, 1.0), p)) /
                      Distributions.quantile(Distributions.Gamma(M, 1.0), p) for p in ps)
        @test err < 1e-7
    end

    # --- Likelihood identity: MCI and MSLE (Halton) vs QuadGK integration ---
    # MCI's change-of-variables converges faster in draws than MSLE's inverse-CDF
    # construction, hence the different tolerances (both are pure QMC error).
    Random.seed!(22)
    N1 = 300
    X1 = hcat(ones(N1), randn(N1))
    beta1 = [1.0, 0.5]; theta1 = 0.25; sv21 = 0.09
    for M in (2, 3, 4)
        sym = Symbol("Erlang", M)
        u1 = rand(Distributions.Gamma(M, theta1), N1)
        y1 = X1 * beta1 .+ sqrt(sv21) .* randn(N1) .- u1
        p1 = vcat(beta1, log(theta1), log(sv21))

        nd = Distributions.Normal(0.0, sqrt(sv21))
        gd = Distributions.Gamma(M, theta1)
        ll_quad = sum(y1 .- X1 * beta1) do e
            val, _ = QuadGK.quadgk(uu -> Distributions.pdf(nd, e + uu) *
                                         Distributions.pdf(gd, uu), 0.0, Inf; rtol=1e-10)
            log(val)
        end

        sp = _MCIB_ERL.sfmodel_spec(depvar=y1, frontier=X1, noise=:Normal,
                                    ineff=sym, hetero=Symbol[])
        me = _MCIB_ERL.sfmodel_method(method=:MCI, n_draws=2^13, multiRand=false)
        internal = _MCIB_ERL._assemble_MCI_spec(sp, me)
        nll_mci = _MCIB_ERL.MCI_nll(internal, p1; chunks=internal.chunks)
        @test isfinite(nll_mci)
        @test abs(nll_mci - (-ll_quad)) / abs(ll_quad) < 1e-4

        us = sfmodel_spec(depvar=y1, frontier=X1, noise=:Normal, ineff=sym)
        ms = _MSLEB_ERL.sfmodel_method(method=:MSLE, n_draws=2^13, multiRand=false)
        internal_m = _MSLEB_ERL._assemble_MSLE_spec(us.msle_spec, ms)
        nll_msle = _MSLEB_ERL.qmc_nll(internal_m, p1; chunks=internal_m.chunks)
        @test isfinite(nll_msle)
        @test abs(nll_msle - (-ll_quad)) / abs(ll_quad) < 5e-4
        # chunk invariance of the MSLE Erlang path
        @test abs(_MSLEB_ERL.qmc_nll(internal_m, p1; chunks=1) - nll_msle) < 1e-10
    end

    # --- Recovery fit + predict identity + marginal effects (hetero θ, Erlang2) ---
    Random.seed!(33)
    N2 = 600
    X2 = hcat(ones(N2), randn(N2))
    Z2 = hcat(ones(N2), randn(N2))
    beta2 = [1.0, 0.5]; gam2 = [log(0.25), 0.3]; sv22 = 0.09
    theta_i = exp.(Z2 * gam2)
    u2 = [rand(Distributions.Gamma(2, t)) for t in theta_i]
    y2 = X2 * beta2 .+ sqrt(sv22) .* randn(N2) .- u2

    sp2 = sfmodel_spec(depvar=y2, frontier=X2, zvar=Z2, noise=:Normal,
                       ineff=:Erlang2, hetero=[:theta])
    me2 = sfmodel_method(method=:MCI, n_draws=1500)
    p02 = sfmodel_init(spec=sp2, frontier=X2 \ y2, ln_theta=gam2,
                       ln_sigma_v_sq=log(sv22), message=false)
    op2 = sfmodel_opt(warmstart_solver=NelderMead(),
                      warmstart_opt=(iterations=60, g_abstol=1e-3),
                      main_solver=Newton(),
                      main_opt=(iterations=200, g_abstol=1e-6))
    r2 = sfmodel_fit(spec=sp2, method=me2, init=p02, optim_options=op2,
                     marginal=true, show_table=false, verbose=false)
    @test r2.converged == true
    @test isfinite(loglikelihood(r2))
    ch = coef(r2)
    @test maximum(abs.(ch[1:2] .- beta2)) < 0.15      # frontier recovered
    @test abs(ch[4] - gam2[2]) < 0.15                 # hetero slope on ln θ recovered
    @test all(r2.jlms .>= 0)
    @test all(0 .< r2.bc .<= 1)
    @test abs(mean(r2.jlms) - mean(u2)) < 0.15        # jlms tracks E[u]
    @test r2.marginal_mean !== nothing                # ∂E[u]/∂z through the θ channel

    # predict on the estimation data reproduces the fitted indices
    pj = predict(r2; frontier=X2, zvar=Z2, depvar=y2, what=:jlms)
    @test maximum(abs.(pj .- r2.jlms)) < 1e-12
    # predict(:marginal) works on new rows
    pm = predict(r2; frontier=X2[1:5, :], zvar=Z2[1:5, :], what=:marginal)
    @test size(pm, 1) == 5

    # --- MSLE fit on the same hetero-theta model: converges, agrees with MCI ---
    rm2 = sfmodel_fit(spec=sp2, method=sfmodel_method(method=:MSLE, n_draws=1500),
                      init=p02, optim_options=op2,
                      marginal=true, show_table=false, verbose=false)
    @test rm2.converged == true
    @test isfinite(loglikelihood(rm2))
    @test maximum(abs.(coef(rm2) .- coef(r2))) < 0.05   # same model, two QMC constructions
    @test all(rm2.jlms .>= 0)
    @test all(0 .< rm2.bc .<= 1)
    @test rm2.marginal_mean !== nothing
    # MSLE predict identity on the estimation data
    pjm = predict(rm2; frontier=X2, zvar=Z2, depvar=y2, what=:jlms)
    @test maximum(abs.(pjm .- rm2.jlms)) < 1e-12

end

# --- Panel: Erlang{2,3,4} in the Wang-Ho (2010) true fixed-effect backend.
# Scalar ln_θ plus the scaling delta block; MCI uses
# :expo_rule with importance scale M·θ, MSLE the standardized Newton quantile
# shared with the cross-sectional backend. The Wang-Ho likelihood is a
# one-dimensional integral per FIRM over the base draw u*, so a per-firm QuadGK
# cross-check applies. Seeded data + Halton draws keep every assert deterministic.
@testset "Erlang panel_TFE (Wang-Ho 2010)" begin
    _PB_ERL = SFrontiers.Panel_Backend

    Random.seed!(44)
    Nf, Tp = 150, 6
    NT = Nf * Tp
    idp = repeat(1:Nf, inner=Tp)
    Xp = hcat(randn(NT), randn(NT))
    zp = randn(NT)                                   # within-firm varying
    betap = [0.6, -0.4]; deltap = 0.3; thetap = 0.4; sv2p = 0.09
    alphap = repeat(randn(Nf), inner=Tp)
    hp = exp.(deltap .* zp)
    up = repeat(rand(Distributions.Gamma(2, thetap), Nf), inner=Tp)
    yp = alphap .+ Xp * betap .+ sqrt(sv2p) .* randn(NT) .- hp .* up

    # Spec construction: panel spec built, MLE skipped, labels correct, MCI+MSLE advertised
    for M in (2, 3, 4)
        spM = sfmodel_spec(depvar=yp, frontier=Xp, zvar=zp, id=idp, noise=:Normal,
                           ineff=Symbol("Erlang", M), type=:prod, datatype=:panel_TFE)
        @test spM.panel_spec !== nothing
        @test spM.mle_spec === nothing               # MLE stays HalfNormal/TN only
        @test spM.panel_spec.varnames[end-1:end] == ["ln_θ", "ln_σᵥ²"]
        @test length(spM.panel_spec.varnames) == 5   # K=2 + L=1 + ln_θ + ln_σᵥ²
        @test SFrontiers._get_supported_methods(spM) == [:MCI, :MSLE]
    end

    spp = sfmodel_spec(depvar=yp, frontier=Xp, zvar=zp, id=idp, noise=:Normal,
                       ineff=:Erlang2, type=:prod, datatype=:panel_TFE)

    # Init plumbing: ln_theta lands in its slot; ln_sigma_u_sq is rejected for Erlang
    p0p = sfmodel_init(spec=spp, ln_theta=log(thetap), ln_sigma_v_sq=log(sv2p))
    @test p0p[end-1] == log(thetap)
    @test_throws ErrorException sfmodel_init(spec=spp, ln_sigma_u_sq=0.1)

    # --- Likelihood identity vs per-firm QuadGK, plus chunk invariance ---
    pp = vcat(betap, deltap, log(thetap), log(sv2p))
    psp = spp.panel_spec
    y_t = _PB_ERL.sf_panel_demean(psp.depvar, psp.offsets)
    x_t = _PB_ERL.sf_panel_demean(psp.frontier, psp.offsets)
    eps_t = y_t .- x_t * betap
    h_t = _PB_ERL.compute_h_tilde(psp.zvar, pp, psp.idx, psp.offsets)
    A, B, C = _PB_ERL._compute_panel_ABC(eps_t, h_t, psp.offsets, Nf)
    gd2 = Distributions.Gamma(2, thetap)
    ll_quad_p = sum(1:Nf) do i
        val, _ = QuadGK.quadgk(u -> exp(-0.5 * (A[i] + 2 * B[i] * u + C[i] * u^2) / sv2p) *
                                     Distributions.pdf(gd2, u), 0.0, Inf; rtol=1e-10)
        -0.5 * (Tp - 1) * (log(2π) + log(sv2p)) + log(val)
    end
    for (meth, tol) in ((:MCI, 5e-4), (:MSLE, 5e-4))
        mep = SFrontiers.sfmodel_panel_method(method=meth, n_draws=2^13, multiRand=false)
        ip = _PB_ERL._assemble_panel_spec(psp, mep)
        nllp = _PB_ERL.panel_nll(ip, pp; chunks=ip.chunks)
        @test isfinite(nllp)
        @test abs(nllp - (-ll_quad_p)) / abs(ll_quad_p) < tol
        @test abs(_PB_ERL.panel_nll(ip, pp; chunks=1) - nllp) < 1e-8   # chunk invariance
    end

    # --- MCI fit: converges and recovers the DGP (seeded; loose finite-sample tols) ---
    op_p = sfmodel_opt(warmstart_solver=NelderMead(),
                       warmstart_opt=(iterations=80, g_abstol=1e-3),
                       main_solver=Newton(),
                       main_opt=(iterations=200, g_abstol=1e-6))
    rp = sfmodel_fit(spec=spp, method=sfmodel_method(method=:MCI, n_draws=500),
                     init=p0p, optim_options=op_p, marginal=true,
                     show_table=false, verbose=false)
    @test rp.converged == true
    @test isfinite(loglikelihood(rp))
    @test maximum(abs.(rp.frontier .- betap)) < 0.1
    @test abs(rp.delta[1] - deltap) < 0.15
    @test abs(exp(rp.list[:ln_theta]) - thetap) < 0.2
    @test all(rp.jlms .>= 0)
    @test all(0 .< rp.bc .<= 1)
    @test rp.marginal_mean !== nothing               # ∂E[u_it]/∂z via h(z)·M·θ

    # --- MSLE fit on the same data: converges, agrees with MCI ---
    rpm = sfmodel_fit(spec=spp, method=sfmodel_method(method=:MSLE, n_draws=500),
                      init=p0p, optim_options=op_p, marginal=true,
                      show_table=false, verbose=false)
    @test rpm.converged == true
    @test maximum(abs.(coef(rpm) .- coef(rp))) < 0.05   # two QMC constructions, one model
    @test all(rpm.jlms .>= 0)
    @test all(0 .< rpm.bc .<= 1)
    @test rpm.marginal_mean !== nothing

    # Panel predict identity on the estimation data (id= keyword; zvar as a matrix)
    pjp = predict(rp; frontier=Xp, zvar=reshape(zp, :, 1), depvar=yp, id=idp, what=:jlms)
    @test maximum(abs.(pjp .- rp.jlms)) < 1e-10

    # Gamma panel stays MCI-only (the estimated shape k sits inside the quantile)
    spg = sfmodel_spec(depvar=yp, frontier=Xp, zvar=zp, id=idp, noise=:Normal,
                       ineff=:Gamma, type=:prod, datatype=:panel_TFE)
    @test_throws ErrorException sfmodel_fit(spec=spg,
        method=sfmodel_method(method=:MSLE, n_draws=100), show_table=false, verbose=false)
end

# --- (1) the MSLE result exposes :theta, (2) MCI indices with N×D draws do not depend on `chunks`,
# (3) the marginal helpers clamp θ, (4) the standardized quantile kernel; and user draws equal to 1.
@testset "Erlang kernels, chunk-invariant indices, marginal clamps and user draws" begin
    _MCIB = SFrontiers.MCI_Backend
    _MSLEB = SFrontiers.MSLE_Backend
    _PB = SFrontiers.Panel_Backend
    _G(M) = Distributions.Gamma(M, 1.0)

    # (4) standardized quantile kernel at working precision over the full draw range in
    # Float64 AND Float32 (the old lower branch lost eps/p: Float32 was 15-31% off at
    # p = 1e-7, Float64 1e-5 at p = 1e-12), and a draw of exactly one is pinned to
    # prevfloat(1) instead of overflowing (1f0 - 1f-15 == 1f0 defeated the old clamp)
    for M in (2, 3, 4)
        lgm = _MSLEB.loggamma(M)
        for (T, tol) in ((Float64, 1e-13), (Float32, 2e-6))
            ps64 = vcat(10.0 .^ (-15:0.25:-1), collect(0.1:0.01:0.9), 1 .- 10.0 .^ (-1:-0.25:-15))
            ps = unique(filter(p -> 0 < p < 1, T.(ps64)))
            push!(ps, prevfloat(one(T)))
            err = maximum(abs(Float64(_MSLEB.myErlangStdQuantile(p, M, T(lgm))) -
                              Distributions.quantile(_G(M), Float64(p))) /
                          Distributions.quantile(_G(M), Float64(p)) for p in ps)
            @test err < tol
            q1 = _MSLEB.myErlangStdQuantile(one(T), M, T(lgm))
            @test isfinite(q1)
            @test q1 == _MSLEB.myErlangStdQuantile(prevfloat(one(T)), M, T(lgm))
        end
    end
    out32 = zeros(Float32, 2, 3)
    _MSLEB.myErlangQuantile!(out32; θ=reshape(Float32[0.5, 2.0], 2, 1), r=Float32[0.1 0.5 1.0],
                             m=2, lgm=Float32(_MSLEB.loggamma(2)), clamp_lo=1f-15)
    @test all(isfinite, out32) && out32[2, 3] < 100          # was about 3e12
    Random.seed!(7)                                           # N×D draws: each row its own draws
    rN = rand(30, 16); θN = rand(30, 1) .+ 0.1; outN = zeros(30, 16)
    _MSLEB.myErlangQuantile!(outN; θ=θN, r=rN, m=3, lgm=_MSLEB.loggamma(3), clamp_lo=1e-15)
    @test maximum(abs.(outN .- θN .* Distributions.quantile.(_G(3), rN))) < 1e-12

    # (2) matrix-form MCI jlms_bc_indices with observation-specific (N×D) draws: the
    # Gamma/Erlang quantile used row 1 of every chunk, so the indices changed with
    # `chunks`; now chunk-invariant and equal to the per-observation reference
    Random.seed!(1)
    N = 6; D = 32
    Y = randn(N); X = hcat(ones(N), randn(N)); Z = ones(N, 1); draws = rand(N, D)
    for (ineff, pp) in ((:Erlang2, [0.5, 0.3, log(0.4), log(0.09)]),
                        (:Gamma,   [0.5, 0.3, log(2.0), log(0.4), log(0.09)]))
        r1 = _MCIB.jlms_bc_indices(Y, X, Z, pp, draws; noise=:Normal, ineff=ineff, chunks=1)
        r4 = _MCIB.jlms_bc_indices(Y, X, Z, pp, draws; noise=:Normal, ineff=ineff, chunks=4)
        ε = Y .- X * pp[1:2]
        ref = [begin
                   u = Distributions.quantile.(Distributions.Gamma(2.0, 0.4), draws[i, :])
                   fv = Distributions.pdf.(Distributions.Normal(0.0, 0.3), ε[i] .+ u)
                   sum(u .* fv) / sum(fv)
               end for i in 1:N]
        @test maximum(abs.(r1.jlms .- r4.jlms)) < 1e-12
        @test maximum(abs.(r1.jlms .- ref)) < 1e-10
        rN = _MCIB.jlms_bc_indices(Y, X, Z, pp, draws; noise=:Normal, ineff=ineff, chunks=N)   # one-row chunks
        @test maximum(abs.(r1.jlms .- rN.jlms)) < 1e-12
        s1 = _MCIB.jlms_bc_indices(Y, X, Z, pp, draws[1:1, :]; noise=:Normal, ineff=ineff, chunks=1)
        s3 = _MCIB.jlms_bc_indices(Y, X, Z, pp, draws[1:1, :]; noise=:Normal, ineff=ineff, chunks=3)
        # shared draws unchanged (the two take different code paths: equal to rounding, as above)
        @test maximum(abs.(s1.jlms .- s3.jlms)) < 1e-12
    end

    # (3) marginal helpers apply the likelihood's θ clamp [1e-12, 1e12] (and k ∈ [0.1, 100]
    # for Gamma) in both backends; interior values unchanged
    idx_e = (ineff=(theta=[1],),)
    for B in (_MCIB, _MSLEB)
        @test B.uncondU_erlang([800.0], idx_e, [1.0], 2) == 2e12      # was Inf
        @test B.uncondU_erlang([-800.0], idx_e, [1.0], 2) == 2e-12
        @test B.uncondU_erlang([0.5], idx_e, [1.0], 3) ≈ 3 * exp(0.5)
    end
    @test _MCIB.uncondU_gamma([100.0, 100.0], (ineff=(k=[1], theta=[2]),), [1.0], [1.0]) == 1e14
    @test _MCIB._base_Eu_star([800.0], idx_e, _MCIB.Erlang_MCI{2}()) == 2e12
    @test _MSLEB._base_Eu_star([800.0], idx_e, _MSLEB.Erlang_MSLE{2}()) == 2e12

    # (1) the MSLE result exposes :theta like MCI (result.theta and result.list[:theta])
    Random.seed!(33)
    N1 = 150
    X1 = hcat(ones(N1), randn(N1)); Z1 = hcat(ones(N1), randn(N1))
    g1 = [log(0.25), 0.3]
    y1 = X1 * [1.0, 0.5] .+ 0.3 .* randn(N1) .- [rand(Distributions.Gamma(2, t)) for t in exp.(Z1 * g1)]
    sp1 = sfmodel_spec(depvar=y1, frontier=X1, zvar=Z1, noise=:Normal, ineff=:Erlang2, hetero=[:theta])
    p01 = sfmodel_init(spec=sp1, frontier=X1 \ y1, ln_theta=g1, ln_sigma_v_sq=log(0.09), message=false)
    op1 = sfmodel_opt(warmstart_solver=NelderMead(), warmstart_opt=(iterations=20, g_abstol=1e-3),
                      main_solver=Newton(), main_opt=(iterations=20, g_abstol=1e-6))
    for meth in (:MCI, :MSLE)
        r = sfmodel_fit(spec=sp1, method=sfmodel_method(method=meth, n_draws=100), init=p01,
                        optim_options=op1, marginal=false, show_table=false, verbose=false)
        @test haskey(r.list, :theta)
        @test r.theta == coef(r)[3:4]
    end

    # user draws: a draw of exactly 1 is rejected at spec assembly in every backend, with
    # the same ErrorException (the
    # MCI change of variables gave NaN: capped u, infinite Jacobian); 0 is accepted
    dr = collect(range(0.01, 0.99, length=40)); dr1 = copy(dr); dr1[end] = 1.0
    dr0 = copy(dr); dr0[1] = 0.0
    Random.seed!(5)
    N5 = 40; X5 = hcat(ones(N5), randn(N5))
    y5 = X5 * [1.0, 0.5] .- rand(Distributions.Gamma(2, 0.3), N5) .+ 0.3 .* randn(N5)
    spm = _MCIB.sfmodel_spec(depvar=y5, frontier=X5, noise=:Normal, ineff=:Erlang2, hetero=Symbol[])
    sps = _MSLEB.sfmodel_spec(depvar=y5, frontier=X5, noise=:Normal, ineff=:Erlang2, hetero=Symbol[])
    @test_throws ErrorException _MCIB._assemble_MCI_spec(spm, _MCIB.sfmodel_method(method=:MCI, draws=dr1, multiRand=false))
    @test_throws ErrorException _MSLEB._assemble_MSLE_spec(sps, _MSLEB.sfmodel_method(method=:MSLE, draws=dr1, multiRand=false))
    @test _MCIB._assemble_MCI_spec(spm, _MCIB.sfmodel_method(method=:MCI, draws=dr0, multiRand=false)).N == N5
    @test _MSLEB._assemble_MSLE_spec(sps, _MSLEB.sfmodel_method(method=:MSLE, draws=dr0, multiRand=false)).N == N5
    @test_throws ErrorException _MCIB.jlms_bc_indices(y5, X5, ones(N5, 1), [1.0, 0.5, log(0.3), log(0.09)],
                                                      reshape(dr1, 1, :); noise=:Normal, ineff=:Erlang2)
    Random.seed!(44)
    Nf, Tp = 12, 4; NT = Nf * Tp; idp = repeat(1:Nf, inner=Tp)
    Xp = hcat(randn(NT), randn(NT)); zp = randn(NT)
    yp = repeat(randn(Nf), inner=Tp) .+ Xp * [0.6, -0.4] .+ 0.3 .* randn(NT) .-
         exp.(0.3 .* zp) .* repeat(rand(Distributions.Gamma(2, 0.4), Nf), inner=Tp)
    spp = sfmodel_spec(depvar=yp, frontier=Xp, zvar=zp, id=idp, noise=:Normal, ineff=:Erlang2,
                       type=:prod, datatype=:panel_TFE)
    for meth in (:MCI, :MSLE)
        @test_throws ErrorException _PB._assemble_panel_spec(spp.panel_spec,
            SFrontiers.sfmodel_panel_method(method=meth, draws=dr1, multiRand=false))
        @test _PB._assemble_panel_spec(spp.panel_spec,
            SFrontiers.sfmodel_panel_method(method=meth, draws=dr0, multiRand=false)).N == Nf
    end
end

# --- The same defect classes outside Erlang: Float32 copula clamps, the [0,1) draw contract in
# predict and the MSLE matrix forms, and the copula fields of the MCI result.
@testset "Float32 copula clamps, the draw contract in predict, copula result fields" begin
    _MCIB = SFrontiers.MCI_Backend
    _MSLEB = SFrontiers.MSLE_Backend
    _PB = SFrontiers.Panel_Backend

    # Float32 copula clamps: 1 - 1e-15 (Gaussian) and 1 - 1e-16 (Gumbel) collapsed to 1f0, so a
    # saturated noise CDF reached log(-log 1) = -Inf in the Gumbel adjustment and the NLL was NaN
    for (B, Gau, Gum) in ((_MCIB, _MCIB.GaussianCopula_MCI(), _MCIB.GumbelCopula_MCI()),
                          (_MSLEB, _MSLEB.GaussianCopula_MSLE(), _MSLEB.GumbelCopula_MSLE()))
        @test B.make_copula_constants(Gau, Float32).copula_clamp_hi < 1f0
        @test B.make_copula_constants(Gum, Float32).copula_clamp_hi < 1f0
        @test B.make_copula_constants(Gau, Float64).copula_clamp_hi == 1 - 1e-15   # Float64 unchanged
        @test B.make_copula_constants(Gum, Float64).copula_clamp_hi == 1 - 1e-16
    end
    Random.seed!(3)
    N32 = 30
    X32 = Float32.(hcat(ones(N32), randn(N32)))
    y32 = Float32.(X32 * [1.0, 0.5] .+ 0.1 .* randn(N32) .- abs.(0.3 .* randn(N32)))
    y32[1] += 5f0                                            # residual of ~50 σ_v: Φ saturates to 1f0
    p32 = Float32[1.0, 0.5, log(0.09), log(0.01), 0.5]
    spm = _MCIB.sfmodel_spec(depvar=y32, frontier=X32, noise=:Normal, ineff=:HalfNormal, copula=:Gumbel, hetero=Symbol[])
    im = _MCIB._assemble_MCI_spec(spm, _MCIB.sfmodel_method(method=:MCI, n_draws=64, multiRand=false))
    @test isfinite(_MCIB.MCI_nll(im, p32; chunks=1))                           # was NaN
    sps = _MSLEB.sfmodel_spec(depvar=y32, frontier=X32, noise=:Normal, ineff=:HalfNormal, copula=:Gumbel, hetero=Symbol[])
    is = _MSLEB._assemble_MSLE_spec(sps, _MSLEB.sfmodel_method(method=:MSLE, n_draws=64, multiRand=false))
    @test isfinite(_MSLEB.qmc_nll(is, p32; chunks=1))                           # was NaN

    # the [0,1) draw contract reaches predict(draws=...) and the MSLE matrix forms; empty draws error clearly
    Random.seed!(5)
    N5 = 40; X5 = hcat(ones(N5), randn(N5))
    y5 = X5 * [1.0, 0.5] .- abs.(0.3 .* randn(N5)) .+ 0.3 .* randn(N5)
    dr = collect(range(0.01, 0.99, length=30)); dr1 = copy(dr); dr1[end] = 1.0
    op5 = sfmodel_opt(warmstart_solver=NelderMead(), warmstart_opt=(iterations=20, g_abstol=1e-3),
                      main_solver=Newton(), main_opt=(iterations=20, g_abstol=1e-6))
    sp5 = sfmodel_spec(depvar=y5, frontier=X5, noise=:Normal, ineff=:HalfNormal)
    in5 = sfmodel_init(spec=sp5, frontier=X5 \ y5, ln_sigma_u_sq=log(0.09), ln_sigma_v_sq=log(0.09), message=false)
    r5 = sfmodel_fit(spec=sp5, method=sfmodel_method(method=:MCI, draws=dr, multiRand=false), init=in5,
                     optim_options=op5, marginal=false, show_table=false, verbose=false)
    @test isfinite(loglikelihood(r5))
    @test all(isfinite, predict(r5; frontier=X5, depvar=y5, what=:jlms, draws=dr))
    @test_throws ErrorException predict(r5; frontier=X5, depvar=y5, what=:jlms, draws=dr1)   # was all-NaN
    pe = [1.0, 0.5, log(0.09), log(0.09)]
    @test_throws ErrorException _MSLEB.qmc_nll(y5, X5, ones(N5, 1), pe, reshape(dr1, 1, :); noise=:Normal, ineff=:HalfNormal)
    @test_throws ErrorException _MSLEB.jlms_bc_indices(y5, X5, ones(N5, 1), pe, reshape(dr1, 1, :); noise=:Normal, ineff=:HalfNormal)
    @test isfinite(_MSLEB.qmc_nll(y5, X5, ones(N5, 1), pe, reshape(dr, 1, :); noise=:Normal, ineff=:HalfNormal))
    for f in (_MCIB._validate_user_draws, _MSLEB._validate_user_draws, _PB._validate_panel_user_draws)
        @test_throws ErrorException f(Float64[])
        @test f([0.0, 0.5, prevfloat(1.0)]) === nothing
    end
    Random.seed!(44)                                          # panel predict with a draw of exactly 1
    Nf, Tp = 12, 4; NT = Nf * Tp; idp = repeat(1:Nf, inner=Tp)
    Xp = hcat(randn(NT), randn(NT)); zp = randn(NT)
    yp = repeat(randn(Nf), inner=Tp) .+ Xp * [0.6, -0.4] .+ 0.3 .* randn(NT) .-
         exp.(0.3 .* zp) .* repeat(abs.(0.4 .* randn(Nf)), inner=Tp)
    spp = sfmodel_spec(depvar=yp, frontier=Xp, zvar=zp, id=idp, noise=:Normal, ineff=:HalfNormal,
                       type=:prod, datatype=:panel_TFE)
    rp = sfmodel_fit(spec=spp, method=sfmodel_method(method=:MCI, draws=dr, multiRand=false),
                     init=sfmodel_init(spec=spp, ln_sigma_u_sq=log(0.16), ln_sigma_v_sq=log(0.09)),
                     optim_options=op5, marginal=false, show_table=false, verbose=false)
    @test isfinite(loglikelihood(rp))
    @test_throws ErrorException predict(rp; frontier=Xp, zvar=reshape(zp, :, 1), depvar=yp, id=idp,
                                        what=:jlms, draws=dr1)

    # MCI result exposes :rho/:kendalls_tau for copula models regardless of show_table (parity with MSLE)
    spc = sfmodel_spec(depvar=y5, frontier=X5, noise=:Normal, ineff=:HalfNormal, copula=:Gaussian)
    inc = sfmodel_init(spec=spc, frontier=X5 \ y5, ln_sigma_u_sq=log(0.09), ln_sigma_v_sq=log(0.09), theta_rho=[0.1], message=false)
    for meth in (:MCI, :MSLE)
        rc = sfmodel_fit(spec=spc, method=sfmodel_method(method=meth, n_draws=50), init=inc,
                         optim_options=op5, marginal=false, show_table=false, verbose=false)
        @test haskey(rc.list, :rho) && haskey(rc.list, :kendalls_tau)
        @test isfinite(rc.rho)
    end
end

# --- Panel user draws and the one rule for their shape. `multiRand=true` with a
# one-dimensional draw vector used to throw a raw MethodError at the `Matrix{T}(method.draws)`
# intake, and a wrong-row matrix failed later with a raw DimensionMismatch. There is now ONE rule
# for user-supplied draws in every backend, and `multiRand` is not part of it: supplying `draws`
# forces multiRand=false (the Halton generator is unused), and the ROW COUNT decides — a vector,
# a 1×D row or a D×1 column is one sequence shared by every firm; an N×D matrix (N = FIRMS) is one
# row of draws per firm; any other row count errors. The intake is shared by every panel
# distribution, so this was never Erlang-specific: HalfNormal, Weibull and Erlang2 are checked.
@testset "Panel user draws: one shape rule" begin
    _PB = SFrontiers.Panel_Backend
    Random.seed!(44)
    Nf, Tp = 12, 4; NT = Nf * Tp; idp = repeat(1:Nf, inner=Tp)
    Xp = hcat(randn(NT), randn(NT)); zp = randn(NT)
    yp = repeat(randn(Nf), inner=Tp) .+ Xp * [0.6, -0.4] .+ 0.3 .* randn(NT) .-
         exp.(0.3 .* zp) .* repeat(rand(Distributions.Gamma(2, 0.4), Nf), inner=Tp)
    spp = sfmodel_spec(depvar=yp, frontier=Xp, zvar=zp, id=idp, noise=:Normal, ineff=:Erlang2,
                       type=:prod, datatype=:panel_TFE)
    dr = collect(range(0.01, 0.99, length=40))
    for meth in (:MCI, :MSLE), dist in (:HalfNormal, :Weibull, :Erlang2)
        sd = sfmodel_spec(depvar=yp, frontier=Xp, zvar=zp, id=idp, noise=:Normal, ineff=dist,
                          type=:prod, datatype=:panel_TFE)
        pz = zeros(length(sd.panel_spec.varnames))       # unit scales: finite NLL for every dist
        mk(d, mr) = _PB._assemble_panel_spec(sd.panel_spec,
            SFrontiers.sfmodel_panel_method(method=meth, draws=d, multiRand=mr, n_draws=40))
        for mr in (true, false)                          # the flag must not change the reading
            ip = mk(dr, mr)                              # 1-D vector: was a raw MethodError
            @test ip.draws == dr
            @test isfinite(_PB.panel_nll(ip, pz; chunks=1))
            @test mk(reshape(dr, 1, :), mr).draws == dr  # 1×D row: the same sequence
            @test mk(reshape(dr, :, 1), mr).draws == dr  # D×1 column: the same sequence
            ok = mk(rand(Nf, 40), mr)                    # N_firms×D: one row per firm
            @test size(ok.draws) == (Nf, 40)
            @test isfinite(_PB.panel_nll(ok, pz; chunks=1))
            @test_throws ErrorException mk(rand(Nf + 1, 40), mr)   # rows ∉ {1, N_firms}
            @test_throws ErrorException mk(rand(NT, 40), mr)       # NT rows: N is firms, not obs
        end
    end
    # Full fit through the unified API, then predict with the same draws: shared and per-firm.
    op = sfmodel_opt(warmstart_solver=NelderMead(), warmstart_opt=(iterations=20, g_abstol=1e-3),
                     main_solver=Newton(), main_opt=(iterations=30, g_abstol=1e-6))
    p0 = sfmodel_init(spec=spp, ln_theta=log(0.4), ln_sigma_v_sq=log(0.09))
    for d in (dr, rand(Nf, 40))
        r = sfmodel_fit(spec=spp, method=sfmodel_method(method=:MCI, draws=d, multiRand=true, n_draws=40),
                        init=p0, optim_options=op, marginal=false, show_table=false, verbose=false)
        @test isfinite(loglikelihood(r))
        @test r.user_draws_supplied == true
        @test r.multiRand == false                       # forced: the generator was not used
        @test r.n_draws == 40
        pj = predict(r; frontier=Xp, zvar=reshape(zp, :, 1), depvar=yp, id=idp, what=:jlms, draws=d)
        @test maximum(abs.(pj .- r.jlms)) < 1e-10        # predict consumed exactly the fit's draws
    end
    # An auto-Halton fit still predicts with no draws= at all.
    ra = sfmodel_fit(spec=spp, method=sfmodel_method(method=:MCI, n_draws=40, multiRand=true),
                     init=p0, optim_options=op, marginal=false, show_table=false, verbose=false)
    @test ra.multiRand == true
    @test ra.user_draws_supplied == false
    @test maximum(abs.(predict(ra; frontier=Xp, zvar=reshape(zp, :, 1), depvar=yp, id=idp,
                               what=:jlms) .- ra.jlms)) < 1e-10
end
