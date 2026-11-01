# Laplace noise: the variance step of MCI and MSLE fits uses the outer product of gradients (OPG) of the
# per-observation log-likelihoods. Checked here: the per-observation keyword of the likelihoods; the fields, the table
# line and the display line of Laplace fits, and normal-noise fits unchanged; the classifier's error, rank and
# boundary cases; the integration check on a Laplace fit; and the standard errors against the OPG and the Hessian of
# the closed-form likelihood of the Laplace-half-normal model.

using Test, SFrontiers, Random, LinearAlgebra
using ForwardDiff: ForwardDiff
using Optim: Optim, BFGS, Newton
using QuadGK: quadgk
using SpecialFunctions: logerfc
using Printf

@testset "Laplace noise: standard errors from the outer product of gradients" begin
    MCB = SFrontiers.MCI_Backend
    MSB = SFrontiers.MSLE_Backend
    RF = SFrontiers.SFRedflag

    # Data from Xoshiro(seed), drawn in this order: x, the two uniforms of the Laplace draw, the half-normal draw;
    # y = 1 + 0.5x + v - sgn·u (sgn = 1 production, -1 cost). The generator is returned for the draws that follow.
    function _lopg_data(seed, n, b, σu, sgn)
        rng = Random.Xoshiro(seed)
        x = randn(rng, n); U1 = rand(rng, n); U2 = rand(rng, n); z = randn(rng, n)
        X = hcat(ones(n), x)
        u = abs.(σu .* z)
        y = X * [1.0, 0.5] .+ b .* (log.(U1) .- log.(U2)) .- sgn .* u
        return X, y, u, rng
    end
    _lopg_nll(m) = m === :MCI ? MCB.MCI_nll : MSB.qmc_nll
    _lopg_isp(sp, m, S) = m === :MCI ?
        MCB._assemble_MCI_spec(sp.mci_spec, sfmodel_method(method = :MCI, n_draws = S)) :
        MSB._assemble_MSLE_spec(sp.msle_spec, sfmodel_method(method = :MSLE, n_draws = S))
    _lopg_opt() = sfmodel_opt(main_solver = BFGS(), main_opt = (iterations = 300, g_abstol = 1e-6))
    # The value of f() and what it printed.
    function _lopg_capture(f)
        path, io = mktemp()
        r = redirect_stdout(f, io)
        close(io)
        txt = read(path, String)
        rm(path)
        return r, txt
    end

    # The closed-form log density of ε = v - u (production frontier), v ~ Laplace(0, b), u ~ N⁺(0, σ²), r = σ/b,
    # in log space; Q(t) = 1 - Φ(t). For a cost frontier (ε = v + u) the density is f(-ε).
    _lopg_logQ(t) = logerfc(t / sqrt(2)) - log(2)
    function _lopg_logf(e, σ, b)
        r = σ / b
        if e >= 0
            return -log(b) + r^2 / 2 - e / b + _lopg_logQ(r)
        end
        # Φ(-e/σ - r) - Φ(-r) = Q(-r) - Q(-e/σ - r)
        la, lb = _lopg_logQ(-r), _lopg_logQ(-e / σ - r)
        t1 = e / b + la + log(-expm1(lb - la))
        t2 = -e / b + _lopg_logQ(-e / σ + r)
        mx = max(t1, t2)
        return -log(b) + r^2 / 2 + mx + log(exp(t1 - mx) + exp(t2 - mx))
    end

    # Cell 1 at N = 300 (b = 0.3, σᵤ = 0.6, production), and the normal noise of the normal-noise fit, drawn after it.
    N1 = 300
    X1, y1, u1, rng1 = _lopg_data(21, N1, 0.3, 0.6, 1)
    yN = X1 * [1.0, 0.5] .+ 0.3 .* randn(rng1, N1) .- u1

    @testset "per-observation log-likelihoods from the chunk loop" begin
        for m in (:MCI, :MSLE), cop in (:None, :Gaussian), T in (Float64, Float32)
            sp = sfmodel_spec(depvar = T.(y1), frontier = T.(X1), noise = :Laplace, ineff = :HalfNormal,
                              copula = cop, type = :prod)
            θ = cop === :None ?
                collect(Float64, sfmodel_init(spec = sp, frontier = [1.0, 0.5], ln_sigma_u_sq = log(0.36),
                                              ln_b = log(0.3), message = false)) :
                collect(Float64, sfmodel_init(spec = sp, frontier = [1.0, 0.5], ln_sigma_u_sq = log(0.36),
                                              ln_b = log(0.3), theta_rho = [0.3], message = false))
            isp = _lopg_isp(sp, m, 64)
            f = _lopg_nll(m)
            ll = Dict{Int, Vector{Float64}}()
            for k in (1, 10)
                v = f(isp, θ; chunks = k, per_obs = true)
                @test eltype(v) == Float64 && length(v) == N1 && all(isfinite, v)
                @test isapprox(sum(v), -f(isp, θ; chunks = k); rtol = 1e-12)
                J = ForwardDiff.jacobian(p -> f(isp, p; chunks = k, per_obs = true), θ)
                @test size(J) == (N1, length(θ)) && all(isfinite, J)
                ll[k] = v
            end
            @test maximum(abs.(ll[1] .- ll[10]) ./ abs.(ll[10])) <= 1e-13
        end
    end

    # The fits on cell 1's data: normal noise (MCI), and Laplace noise by MCI and MSLE, without and with a Gaussian
    # copula; the MCI Laplace fit without a copula runs the integration check and prints its table.
    spN = sfmodel_spec(depvar = yN, frontier = X1, noise = :Normal, ineff = :HalfNormal, type = :prod)
    rN, outN = _lopg_capture(() -> sfmodel_fit(spec = spN, method = sfmodel_method(method = :MCI, n_draws = 64),
        init = sfmodel_init(spec = spN, frontier = X1 \ yN, ln_sigma_u_sq = log(0.3), ln_sigma_v_sq = log(0.05),
                            message = false),
        optim_options = _lopg_opt(), marginal = false, show_table = true, verbose = false))
    sp1 = Dict(cop => sfmodel_spec(depvar = y1, frontier = X1, noise = :Laplace, ineff = :HalfNormal, copula = cop,
                                   type = :prod) for cop in (:None, :Gaussian))
    fits = Dict{Tuple{Symbol, Symbol}, Any}()
    out1 = ""
    for m in (:MCI, :MSLE), cop in (:None, :Gaussian)
        sp = sp1[cop]
        init = cop === :None ?
            sfmodel_init(spec = sp, frontier = X1 \ y1, ln_sigma_u_sq = log(0.3), ln_b = log(0.2), message = false) :
            sfmodel_init(spec = sp, frontier = X1 \ y1, ln_sigma_u_sq = log(0.3), ln_b = log(0.2), theta_rho = [0.0],
                         message = false)
        meth = sfmodel_method(method = m, n_draws = cop === :None ? 64 : 256)
        if (m, cop) == (:MCI, :None)
            fits[(m, cop)], out1 = _lopg_capture(() -> sfmodel_fit(spec = sp, method = meth, init = init,
                optim_options = _lopg_opt(), marginal = false, integration_check = true, show_table = true,
                verbose = false))
        else
            fits[(m, cop)] = sfmodel_fit(spec = sp, method = meth, init = init, optim_options = _lopg_opt(),
                                         marginal = false, show_table = false, verbose = false)
        end
    end

    @testset "normal noise keeps the Hessian" begin
        @test rN.vcov_type === :Hessian
        @test rN.OPG === nothing
        @test rN.Hessian isa AbstractMatrix && size(rN.Hessian) == (4, 4)
        @test rN.var_cov_mat == RF.sf_classify_hessian(p -> MCB.MCI_nll(rN.model, p; chunks = rN.model.chunks),
                                                       rN.coeff; message = false).var_cov_matrix
    end

    @testset "OPG standard errors of a Laplace fit ($m, copula $cop)" for m in (:MCI, :MSLE), cop in (:None, :Gaussian)
        r = fits[(m, cop)]
        p = length(r.coeff)
        f = _lopg_nll(m)
        @test r.vcov_type === :OPG
        @test r.Hessian === nothing
        @test size(r.OPG) == (p, p)
        J = ForwardDiff.jacobian(q -> f(r.model, q; chunks = r.model.chunks, per_obs = true), r.coeff)
        @test isapprox(r.OPG, J' * J; rtol = 1e-12)
        @test r.var_cov_mat ≈ inv(r.OPG)
        @test all(s -> isfinite(s) && s > 0, r.std_err)
    end

    @testset "the table and the display name the OPG" begin
        @test occursin("outer product of gradients", out1)
        @test !occursin("outer product of gradients", outN)
        @test occursin("OPG", sprint(show, MIME"text/plain"(), fits[(:MCI, :None)]))
    end

    @testset "sf_classify_opg: errors, rank and the boundary message" begin
        x0 = [0.1, 0.2]
        re, oute = _lopg_capture(() -> RF.sf_classify_opg(p -> error("no scores"), x0; message = false))
        @test all(isnan, re.var_cov_matrix) && all(isnan, re.opg) && size(re.opg) == (2, 2)
        @test re.redflag == 1 && re.hessian_flag == 1 && re.hessian === nothing
        @test occursin("The outer product of gradients could not be computed at the estimate", oute)
        @test_throws InterruptException RF.sf_classify_opg(p -> throw(InterruptException()), x0)
        # two identical columns of J
        rk = RF.sf_classify_opg(p -> [p[1] + p[2], 2 * (p[1] + p[2]), 3 * (p[1] + p[2])], x0; message = false)
        @test all(isnan, sqrt.(diag(rk.var_cov_matrix))) && rk.redflag == 1
        # full rank: the boundary message comes first
        fr = p -> [p[1], p[2], p[1] + 2 * p[2]]
        txt = "σᵤ (coefficient ln_σᵤ²) at its lower bound 1e-12"
        rb, outb = _lopg_capture(() -> RF.sf_classify_opg(fr, x0; at_clamp = true, message = true, bound_text = txt))
        @test startswith(outb, txt) && rb.redflag == 1
        _, outg = _lopg_capture(() -> RF.sf_classify_opg(fr, x0; at_clamp = true, message = true, bound_text = ""))
        @test occursin("A parameter is at or beyond the numerical bound SFrontiers applies to it", outg)
    end

    @testset "the integration check on a Laplace fit uses the inverse OPG" begin
        r = fits[(:MCI, :None)]
        ic = r.integration_check
        @test ic.shift_available
        @test isfinite(ic.max_shift_se)
        # the draws of a fit with n_draws = 4 × 64 and the same settings, in 4 × chunks = 40 groups
        isp4 = MCB._assemble_MCI_spec(sp1[:None].mci_spec, sfmodel_method(method = :MCI, n_draws = 4 * 64, chunks = 40))
        gS = ForwardDiff.gradient(p -> MCB.MCI_nll(r.model, p; chunks = r.model.chunks), r.coeff)
        g4 = ForwardDiff.gradient(p -> MCB.MCI_nll(isp4, p; chunks = 40), r.coeff)
        @test ic.shift ≈ -r.var_cov_mat * (g4 - gS) rtol = 1e-10
    end

    @testset "the closed form against quadrature" begin
        σ, b = 0.6, 0.3
        for (e, ref) in ((-1.2, 0.257886123161938), (-0.1, 0.707623850091802), (0.4, 0.1477040214465139))
            g = u -> exp(-abs(e + u) / b) / (2b) * 2 / σ * exp(-u^2 / (2σ^2)) / sqrt(2π)
            q = e < 0 ? quadgk(g, 0, -e, Inf; rtol = 1e-13)[1] : quadgk(g, 0, Inf; rtol = 1e-13)[1]
            fe = exp(_lopg_logf(e, σ, b))
            @test isapprox(fe, q; rtol = 1e-12)
            @test isapprox(fe, ref; rtol = 1e-12)       # the references carry 15 significant digits
        end
    end

    # Three cells at N = 1,000: (seed, b, σᵤ, sgn, frontier type).
    cells = ((21, 0.3, 0.6, 1, :prod), (22, 0.3, 0.6, 1, :prod), (23, 0.5, 0.4, -1, :cost))
    # The measured values of the three comparisons, one row per fit, printed under SFRONTIERS_REFERENCE_REPORT=1.
    lopg_report = String[]
    @testset "the closed-form likelihood, cell $ci, $m" for ci in 1:3, m in (:MCI, :MSLE)
        seed, b, σu, sgn, ftype = cells[ci]
        X, y, _, _ = _lopg_data(seed, 1000, b, σu, sgn)
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Laplace, ineff = :HalfNormal, type = ftype)
        r = sfmodel_fit(spec = sp, method = sfmodel_method(method = m, n_draws = m === :MSLE ? 2048 : 512),
                        init = sfmodel_init(spec = sp, frontier = X \ y, ln_sigma_u_sq = log(0.3), ln_b = log(0.2),
                                            message = false),
                        optim_options = _lopg_opt(), marginal = false, show_table = false, verbose = false)
        # near a stationary point (the gradient norms reached are 1e-7 to 1e-6); whether BFGS reports
        # convergence on the kinks of the Laplace likelihood can depend on rounding and package versions,
        # and the exact maximum-likelihood estimate below is compared with this estimate in any case
        @test r.gradient_norm <= 1e-4
        @test r.vcov_type === :OPG
        # the closed-form log-likelihood of each observation at θ = (β0, β1, ln σᵤ², ln b)
        llobs = θ -> [_lopg_logf(sgn * (y[i] - θ[1] - θ[2] * X[i, 2]), exp(θ[3] / 2), exp(θ[4])) for i in eachindex(y)]
        nll = θ -> -sum(llobs(θ))
        Jx = ForwardDiff.jacobian(llobs, r.coeff)
        se_opg = sqrt.(diag(inv(Jx' * Jx)))
        @test all(abs.(r.std_err ./ se_opg .- 1) .<= 0.02)
        se_h = sqrt.(diag(inv(ForwardDiff.hessian(nll, r.coeff))))
        @test all(abs.(r.std_err ./ se_h .- 1) .<= 0.15)
        # the exact maximum-likelihood estimate lies within 0.1 standard errors of the package's
        ex = Optim.optimize(nll, (G, θ) -> ForwardDiff.gradient!(G, nll, θ), (H, θ) -> ForwardDiff.hessian!(H, nll, θ),
                            collect(r.coeff), Newton(), Optim.Options(iterations = 100, g_abstol = 1e-9))
        @test Optim.converged(ex)
        @test all(abs.(Optim.minimizer(ex) .- r.coeff) .<= 0.1 .* r.std_err)
        push!(lopg_report, @sprintf("%-4d %-5s %-5d %.2e   %.2e   %.2e", seed, m, m === :MSLE ? 2048 : 512,
                                    maximum(abs.(r.std_err ./ se_opg .- 1)), maximum(abs.(r.std_err ./ se_h .- 1)),
                                    maximum(abs.(Optim.minimizer(ex) .- r.coeff) ./ r.std_err)))
    end

    @testset "the boundary message through the OPG branch" begin
        sp = sp1[:None]
        init = sfmodel_init(spec = sp, frontier = X1 \ y1, ln_sigma_u_sq = log(0.3), ln_b = log(1e-13), message = false)
        r, out = _lopg_capture(() -> sfmodel_fit(spec = sp, method = sfmodel_method(method = :MCI, n_draws = 64),
            init = init, optim_options = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,)),
            verbose = true, show_table = false, marginal = false))
        @test occursin("b (coefficient ln_b) at its lower bound", out)
        @test r.vcov_type === :OPG
        @test r.redflag == 1
        @test length(r.bound_check) == 1 && r.bound_check[1].parameter === :b
    end

    if get(ENV, "SFRONTIERS_REFERENCE_REPORT", "") == "1"
        println("standard errors against the closed form, N = 1,000: the largest |se/se_OPG - 1| (tolerance 0.02) "
                * "and |se/se_Hessian - 1| (0.15), and the largest |exact MLE - estimate|/se (0.1)")
        @printf("%-4s %-5s %-5s %-10s %-10s %s\n", "seed", "fit", "draws", "vs OPG", "vs Hessian", "MLE shift")
        foreach(println, lopg_report)
    end
end
