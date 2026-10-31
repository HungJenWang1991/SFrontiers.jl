# ---------------------------------------------------------------------------
# The MCI jlms/bc warning on a small effective sample size. At parameter
# points where the truncated-normal inefficiency concentrates far beyond the
# reach of the draw transformation (μ/σᵤ far above n_draws), the jlms/bc
# weights collapse onto one draw for every observation. The per-observation
# effective sample size detects it: the warning prints when it is below 10,
# and the returned values are unchanged.
# ---------------------------------------------------------------------------
@testset "MCI jlms/bc: the warning on a small effective sample size of the importance weights" begin
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
    # θ_bad is not an MCI optimum: it combines the analytic-MLE values of the
    # truncated-normal scaling model, fitted to another 80-observation sample,
    # with the true β. At θ_bad the MCI log-likelihood lies far below the exact
    # one; the test uses it only to trigger the warning.
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

    # The warning for k of n observations, D draws and the smallest effective sample size s.
    _ess_text(k, n, D, s) = "Warning (MCI): For $k of the $n observations, the simulated likelihood and " *
        "the JLMS and BC indices are effectively computed from fewer than 10 of the $D draws (effective " *
        "sample size below 10; smallest $s). This warning flags a potential numerical issue; it does not " *
        "establish that this fit is affected. To check this fit, use sfmodel_fit(..., integration_check " *
        "= true); if the check reports CAUTION or FAIL, increase n_draws, try transformation = " *
        ":logistic_2_rule, or refit with method = :MSLE or :MLE where the model allows it.\n"

    # Degenerate corner: the warning prints on the unchunked and on the chunked
    # path, the same text on both; the returned values are unchanged by the
    # diagnostic (finite, bc ≈ 1 exactly is the known symptom).
    r1, out1 = captured_stdout(() -> B.jlms_bc_indices(ispec, θ_bad; chunks=1))
    @test occursin("effective sample size", out1)
    @test occursin("sfmodel_fit(..., integration_check = true)", out1)
    @test occursin("method = :MSLE", out1)
    @test out1 == _ess_text(80, 80, 512, "1.0")
    @test all(isfinite, r1.jlms) && all(isfinite, r1.bc)
    r3, out3 = captured_stdout(() -> B.jlms_bc_indices(ispec, θ_bad; chunks=3))
    @test out3 == out1
    @test isapprox(collect(r3.jlms), collect(r1.jlms); rtol=1e-12)
    @test isapprox(collect(r3.bc), collect(r1.bc); rtol=1e-12)

    # Healthy parameters: nothing is printed, on either path.
    r2, out2 = captured_stdout(() -> B.jlms_bc_indices(ispec, θ_good; chunks=1))
    @test out2 == ""
    @test all(isfinite, r2.jlms)
    _, out4 = captured_stdout(() -> B.jlms_bc_indices(ispec, θ_good; chunks=3))
    @test out4 == ""

    # Helper unit check: counts the observations below 10 and returns the
    # count; nothing is printed when none is below 10.
    n_low, outh = captured_stdout(() -> B._warn_low_ess([1.0, 5.0, 50.0], 512))
    @test n_low == 2 && occursin("For 2 of the 3 observations", outh)
    @test outh == _ess_text(2, 3, 512, "1.0")
    n_ok, outq = captured_stdout(() -> B._warn_low_ess([50.0, 400.0], 512))
    @test n_ok == 0 && outq == ""
end
