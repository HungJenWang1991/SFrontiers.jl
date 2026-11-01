# Regression tests for the repair of the panel_TRE (Greene 2005 true random-effect)
# likelihood.
#
# Four faults shared by LL_T(PTREH), LL_T(PTRET) and their two jlmsbc counterparts:
#
#   1. The three variances were exp(rho) with no clamp, while jlmsbc clamped the same three
#      to [1e-12, 1e12].  A subnormal sigma_u^2 made 1/sigma_u^2 overflow to Inf and
#      mu'invSigma_u become 0*Inf = NaN -- a NaN with no automatic differentiation involved.
#   2. rho^2 was floored with max(., zero(eltype(rho))); sqrt of an exactly-zero Dual gives
#      NaN partials, so the quadrature integrand went NaN under ForwardDiff.
#   3. gamma^2 was floored at 1e-12.  G is supposed to BE inv(inv(Sigma)+inv(Sigma_u)); once
#      the floor bound, it was not, the two quadratic forms in ll stopped cancelling, and the
#      objective ran to -1.6e35.  That is what the reported DomainError was chasing.
#   4. rho^2 was computed as 1/(1+g)*... where 1+g cancels catastrophically; it rounds to
#      exactly 0 (rho^2 = Inf) on reachable parameters, and log|Sigma| was formed as
#      log(max(det, 1e-15)), which underflows the floor for an ordinary sigma_v^2 once T is
#      moderate -- an error of up to 202 nats.
#
# The repair clamps the variances as jlmsbc already did, writes G by a cancellation-free
# Sherman-Morrison form, and takes both log-determinants from the eigenvalues.
#
# Only CSV and Test are test dependencies, so nothing here loads Logging; log records are
# swallowed with Test.collect_test_logs, per test_predict_draw_alignment.jl.

using SFrontiers
using Test
using Optim
using LinearAlgebra
using QuadGK
using StatsFuns
using Printf

const _SFM_TRE = SFrontiers.MLE_Backend.SFmle

_tre_po(::Val{:H}) = (begx=1, endx=2, begz=0, endz=0, begq=3, endq=3, begw=4, endw=4, begv=5, endv=5)
_tre_po(::Val{:T}) = (begx=1, endx=2, begz=3, endz=3, begq=4, endq=4, begw=5, endw=5, begv=6, endv=6)

function _tre_data(nf, nt)
    n  = nf * nt
    id = repeat(1:nf, inner = nt)
    q  = collect(range(-1, 1; length = n))
    X  = hcat(ones(n), q)
    y  = X * [0.62, -0.24] .- abs.(0.18 .* sin.((1:n) .* 0.81)) .+ 0.035 .* cos.((1:n) .* 1.09)
    idt = Matrix{Any}(undef, nf, 2)
    for i in 1:nf
        idt[i, 1] = collect(((i-1)*nt+1):(i*nt))
        idt[i, 2] = nt
    end
    (y, X, id, idt, n)
end

# Firm-by-firm likelihood by one-dimensional quadrature over the random effect, using the
# ordinary cross-sectional composed-error densities.  Shares no algebra with the closed form.
function _tre_ll_quadrature(y, X, id, β, σₐ², σᵤ², σᵥ²; μ = nothing)
    σ² = σᵤ² + σᵥ²; σ = sqrt(σ²); σᵤ = sqrt(σᵤ²); σᵥ = sqrt(σᵥ²)
    λ = σᵤ/σᵥ; σₐ = sqrt(σₐ²)
    ε = y - X*β
    tot = 0.0
    for f in unique(id)
        ind = findall(==(f), id)
        Li, _ = quadgk(a -> begin
            p = normpdf(a/σₐ)/σₐ
            for k in ind
                w = ε[k] - a
                p *= μ === nothing ?
                     (2/σ)*normpdf(w/σ)*normcdf(-λ*w/σ) :
                     (1/σ)*normpdf((w+μ)/σ)*normcdf(μ/(λ*σ) - λ*w/σ)/normcdf(μ/σᵤ)
            end
            p
        end, -Inf, Inf, rtol = 1e-12, atol = 1e-300)
        tot += log(Li)
    end
    tot
end

@testset "panel_TRE likelihood" begin

    # ------------------------------------------------------------------
    # 1. The reported DomainError.  On a seven-firm, three-period panel (_tre_data(7, 3)),
    #    datatype=:panel_TRE with ineff=:HalfNormal threw
    #    "DomainError with 0.0: integrand produced Dual{...}(NaN,NaN,...)" under Newton at
    #    300 iterations, from the explicit start below and with init=nothing alike.
    # ------------------------------------------------------------------
    @testset "Newton no longer throws (up to 1,000 iterations)" begin
        y, X, id, _, _ = _tre_data(7, 3)
        sp = sfmodel_spec(datatype = :panel_TRE, depvar = y, frontier = X, id = id,
                          type = :production, noise = :Normal, ineff = :HalfNormal)
        ini = sfmodel_init(spec = sp, frontier = [0.44, -0.16], sigma_a_sq = 0.12,
                           ln_sigma_u_sq = -1.2, ln_sigma_v_sq = -2.2)
        # Newton needs about 160 iterations on this panel, a count that can move with rounding and
        # package versions: the cap of 1,000 leaves room (the path up to 300 is the one that threw)
        for start in (ini, nothing)
            o = sfmodel_opt(warmstart_solver = nothing, main_solver = Newton(),
                            main_opt = (iterations = 1000, g_abstol = 1e-8))
            r = last(Test.collect_test_logs(() -> sfmodel_fit(spec = sp,
                    method = sfmodel_method(method = :MLE), init = start, optim_options = o,
                    jlms_bc_index = false, marginal = false, show_table = false,
                    verbose = false)))
            @test isfinite(r.loglikelihood)
            @test r.converged
            # both starts reach the same optimum
            @test isapprox(Float64(r.loglikelihood), 31.3795024, atol = 1e-4)
        end
        # NelderMead at 300 threw as well, and now reaches the same place
        o = sfmodel_opt(warmstart_solver = nothing, main_solver = NelderMead(),
                        main_opt = (iterations = 300, g_abstol = 1e-8))
        r = last(Test.collect_test_logs(() -> sfmodel_fit(spec = sp,
                method = sfmodel_method(method = :MLE), init = ini, optim_options = o,
                jlms_bc_index = false, marginal = false, show_table = false, verbose = false)))
        @test isfinite(r.loglikelihood)
        @test isapprox(Float64(r.loglikelihood), 31.3795024, atol = 1e-3)
    end

    # ------------------------------------------------------------------
    # 2. The closed form agrees with an independent quadrature over the random effect.
    #    Before the repair the T = 10 and T = 20 rows were wrong by 6.7 to 15 nats.
    # ------------------------------------------------------------------
    # The measured differences of the closed form against the quadrature, printed under
    # SFRONTIERS_REFERENCE_REPORT=1: |closed form − quadrature|, and that difference relative to the larger of the
    # two, which isapprox compares with rtol.
    tre_report = String[]
    _tre_row(u, nf, nt, par, got, want) = @sprintf("%-3s %-16s %-26s %.2e   %.2e   rtol 1e-8", u,
        "$nf firms, T = $nt", par, abs(got - want), abs(got - want) / max(abs(got), abs(want)))
    @testset "closed form matches an independent quadrature" begin
        for (nf, nt) in ((7, 3), (4, 10), (3, 20))
            y, X, id, idt, n = _tre_data(nf, nt)
            Y = reshape(y, n, 1); one1 = ones(n, 1)
            for (la, lu, lv) in ((0.0, -1.0, -2.3), (0.0, -1.0, -4.6), (-1.0, -2.0, -6.9),
                                 (0.0, 0.0, 0.0))
                rho = [0.62, -0.24, la, lu, lv]
                got = -_SFM_TRE.LL_T(_SFM_TRE.PTREH, Y, X, (), one1, one1, one1, 1, n,
                                     _tre_po(Val(:H)), rho, idt, nothing)
                want = _tre_ll_quadrature(y, X, id, rho[1:2], exp(la), exp(lu), exp(lv))
                @test isapprox(got, want; rtol = 1e-8)
                push!(tre_report, _tre_row("HN", nf, nt, "($la, $lu, $lv)", got, want))
            end
            # truncated-normal form, same data
            Z = ones(n, 1)
            for (mu, la, lu, lv) in ((0.08, 0.12, -1.2, -2.2), (0.0, 0.0, -1.0, -4.6),
                                     (0.3, -1.0, -2.0, -6.9))
                rho = [0.62, -0.24, mu, la, lu, lv]
                got = -_SFM_TRE.LL_T(_SFM_TRE.PTRET, Y, X, Z, one1, one1, one1, 1, n,
                                     _tre_po(Val(:T)), rho, idt, nothing)
                want = _tre_ll_quadrature(y, X, id, rho[1:2], exp(la), exp(lu), exp(lv); μ = mu)
                @test isapprox(got, want; rtol = 1e-8)
                push!(tre_report, _tre_row("TN", nf, nt, "($mu; $la, $lu, $lv)", got, want))
            end
        end
    end

    # ------------------------------------------------------------------
    # 3. The likelihood and the index are built from the same parameter map.  LL_T took the
    #    raw exp(rho) while jlmsbc clamped it, so at a fitted variance outside [1e-12, 1e12]
    #    they described different models.
    # ------------------------------------------------------------------
    @testset "likelihood and index share one parameter map" begin
        src = read(joinpath(dirname(dirname(pathof(SFrontiers))), "src", "sf_MLE",
                            "SFloglikefun.jl"), String)
        idx = read(joinpath(dirname(dirname(pathof(SFrontiers))), "src", "sf_MLE",
                            "SFindex.jl"), String)
        # both TRE likelihoods clamp all three variances, as both jlmsbc already did;
        # LL_T(PFECSWH) clamps sigma_u^2 and sigma_v^2 with the same text (hence 3)
        @test count("clamp(exp(rho[po.begq]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)", src) == 2
        @test count("clamp(exp(rho[po.begw]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)", src) == 3
        @test count("clamp(exp(rho[po.begv]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)", src) == 3
        # the cancelling form is gone from all four copies
        @test !occursin("1/(1+g)", src)
        @test !occursin("1/(1+g)", idx)
        @test count("cTd", src) == 4 && count("cTd", idx) == 4
    end

    # ------------------------------------------------------------------
    # 4. A TRE fit still produces a finite jlms/bc index, and predict reproduces it.
    # ------------------------------------------------------------------
    @testset "index and predict stay finite" begin
        y, X, id, _, _ = _tre_data(4, 10)
        sp = sfmodel_spec(datatype = :panel_TRE, depvar = y, frontier = X, id = id,
                          type = :production, noise = :Normal, ineff = :HalfNormal)
        ini = sfmodel_init(spec = sp, frontier = [0.44, -0.16], sigma_a_sq = 0.12,
                           ln_sigma_u_sq = -1.2, ln_sigma_v_sq = -2.2)
        o = sfmodel_opt(warmstart_solver = nothing, main_solver = NelderMead(),
                        main_opt = (iterations = 40,))
        r = last(Test.collect_test_logs(() -> sfmodel_fit(spec = sp,
                method = sfmodel_method(method = :MLE), init = ini, optim_options = o,
                jlms_bc_index = true, marginal = false, show_table = false, verbose = false)))
        @test all(isfinite, r.jlms)
        @test all(isfinite, r.bc)
        @test all(>=(0), r.jlms)
        # predict on the estimation data reproduces the fit's own cached jlms/bc
        p = predict(r; depvar = y, frontier = X, id = id, what = :jlms)
        pj = p isa NamedTuple ? p.jlms : p
        pb = p isa NamedTuple ? p.bc   : nothing
        @test all(isfinite, pj)
        @test isapprox(Float64.(vec(pj)), Float64.(vec(r.jlms)); rtol = 1e-8)
        pb === nothing || @test isapprox(Float64.(vec(pb)), Float64.(vec(r.bc)); rtol = 1e-8)
    end

    if get(ENV, "SFRONTIERS_REFERENCE_REPORT", "") == "1"
        println("closed form against the quadrature over the random effect")
        @printf("%-3s %-16s %-26s %-10s %-10s %s\n", "u", "panel", "(μ;) ln σₐ², ln σᵤ², ln σᵥ²", "abs diff",
                "rel diff", "tolerance")
        foreach(println, tre_report)
    end
end
