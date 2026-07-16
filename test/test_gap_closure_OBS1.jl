# ---------------------------------------------------------------------------
# OBS-1 (09_gap_closure_findings.md) — MCI importance-sample degeneracy
# diagnostic. At parameter corners where the TruncatedNormal inefficiency
# concentrates outside the draw transformation's reach (μ/σᵤ ≫ n_draws), MCI's
# jlms/bc are dominated by a single draw and silently biased (jlms = σᵤ·(D−1),
# bc ≈ 1.0 exactly). The fix adds a per-observation effective-sample-size
# check to _jlms_bc_mci_T (both chunked and non-chunked paths) that WARNS
# (values unchanged) when min ESS < 10, recommending more draws or :MSLE.
# ---------------------------------------------------------------------------
@testset "OBS-1 MCI jlms/bc low-ESS warning at degenerate TN-scaling corners" begin
    Random.seed!(2032)
    n = 80
    x1 = randn(n); x2 = randn(n)
    X  = hcat(ones(n), x1, x2)
    z1 = abs.(randn(n)) .+ 0.5
    z2 = abs.(randn(n)) .+ 0.5
    Zs = hcat(z1, z2)                       # scaling zvar: no constant column
    y  = X * [1.0, 0.5, -0.3] .+ 0.3 .* randn(n) .- 0.5 .* abs.(randn(n))

    uspec = sfmodel_spec(depvar=y, frontier=X, zvar=Zs, hetero=:scaling,
                         noise=:Normal, ineff=:TruncatedNormal, type=:prod)
    B = SFrontiers.MCI_Backend
    ispec = B._assemble_MCI_spec(uspec.mci_spec,
        sfmodel_method(method=:MCI, n_draws=512, multiRand=false, chunks=1))

    _asvec(x::Integer) = Int[x]
    _asvec(x) = collect(Int, x)
    function build_θ(; β, δ, μ0, lnσu2, lnσv2)
        idx = ispec.idx
        inds = vcat(_asvec(idx.beta), _asvec(idx.delta), _asvec(idx.ineff.mu),
                    _asvec(idx.ineff.sigma_u), _asvec(idx.noise.ln_sigma_v_sq))
        θ = zeros(maximum(inds))
        θ[_asvec(idx.beta)]                 .= β
        θ[_asvec(idx.delta)]                .= δ
        θ[_asvec(idx.ineff.mu)]             .= μ0
        θ[_asvec(idx.ineff.sigma_u)]        .= lnσu2
        θ[_asvec(idx.noise.ln_sigma_v_sq)]  .= lnσv2
        θ
    end
    # The G2-TN-scaling-prod fit optimum from the gap-closure probe: TN base is
    # a near-point-mass at μ0 = 23.744 with σu0 = 2.24e-5 — far outside the
    # logistic proposal's reach [0, σu0·(D−1)] ⇒ ESS ≈ 1 for every observation.
    θ_bad  = build_θ(β=[1.0, 0.5, -0.3], δ=[-1.0162, -6.8898], μ0=23.744,
                     lnσu2=2 * log(2.2446e-5), lnσv2=2 * log(0.38398))
    θ_good = build_θ(β=[1.0, 0.5, -0.3], δ=[0.2, -0.15], μ0=0.2,
                     lnσu2=2 * log(0.5), lnσv2=2 * log(0.3))

    function captured_stdout(f)
        old = stdout
        path, io = mktemp()
        local ret
        redirect_stdout(io)
        try
            ret = f()
        finally
            flush(io); redirect_stdout(old); close(io)
        end
        (ret, read(path, String))
    end

    # Degenerate corner: warning fires on the non-chunked AND chunked paths;
    # returned values are unchanged by the diagnostic (finite, bc ≈ 1 exactly
    # is the known symptom).
    r1, out1 = captured_stdout(() -> B.jlms_bc_indices(ispec, θ_bad; chunks=1))
    @test occursin("effective sample size", out1)
    @test occursin("MSLE", out1)
    @test all(isfinite, r1.jlms) && all(isfinite, r1.bc)
    r3, out3 = captured_stdout(() -> B.jlms_bc_indices(ispec, θ_bad; chunks=3))
    @test occursin("effective sample size", out3)
    @test isapprox(collect(r3.jlms), collect(r1.jlms); rtol=1e-12)

    # Healthy parameters: no warning, on either path.
    r2, out2 = captured_stdout(() -> B.jlms_bc_indices(ispec, θ_good; chunks=1))
    @test !occursin("effective sample size", out2)
    @test all(isfinite, r2.jlms)
    _, out4 = captured_stdout(() -> B.jlms_bc_indices(ispec, θ_good; chunks=3))
    @test !occursin("effective sample size", out4)

    # Helper unit check: counts observations below the threshold and returns
    # the count; zero-count is silent.
    n_low, outh = captured_stdout(() -> B._warn_low_ess([1.0, 5.0, 50.0], 512))
    @test n_low == 2 && occursin("2 observation", outh)
    n_ok, outq = captured_stdout(() -> B._warn_low_ess([50.0, 400.0], 512))
    @test n_ok == 0 && outq == ""
end
