# With 512 shared draws (multiRand = false) the top draw t = 511/512 maps, under the
# truncated normal's default rule u = s·t/(1 − t), to 511·σᵤ0·hᵢ. Where μ0 = 511·σᵤ0 this is
# the mode of a near-point-mass inefficiency for every observation, so the simulated
# log-likelihood exceeds the exact one by n·log(512·φ(0)), and an MCI fit started there stops
# with converged = true. The 4 × 512 shared draws contain the 512, the same draw carries all
# the weight divided by 4, so the log-likelihood falls by n·log 4 and the check reads FAIL.
import Distributions

@testset "integration check: a spurious maximum under shared draws" begin
    rng = MersenneTwister(2027); n = 80
    x1 = randn(rng, n); x2 = randn(rng, n)
    Xg = hcat(ones(n), x1, x2)
    z1 = abs.(randn(rng, n)) .+ 0.5
    z2 = abs.(randn(rng, n)) .+ 0.5
    Zg = hcat(z1, z2)                      # scaling zvar, no constant column
    vg = 0.3 .* randn(rng, n)
    ug = 0.5 .* abs.(randn(rng, n))
    yg = Xg * [1.0, 0.5, -0.3] .+ vg .- ug
    spec = sfmodel_spec(depvar = yg, frontier = Xg, zvar = Zg, hetero = :scaling,
                        noise = :Normal, ineff = :TruncatedNormal, type = :prod)
    idx = SFrontiers.MCI_Backend._assemble_MCI_spec(spec.mci_spec,
              sfmodel_method(method = :MCI, n_draws = 512, multiRand = false)).idx
    _icsd_iv(x) = x isa Integer ? [x] : collect(Int, x)
    ib = _icsd_iv(idx.beta); idl = _icsd_iv(idx.delta)
    imu = only(_icsd_iv(idx.ineff.mu)); isu = only(_icsd_iv(idx.ineff.sigma_u)); isv = only(_icsd_iv(idx.noise.ln_sigma_v_sq))
    N01 = Distributions.Normal()
    function _icsd_exact_ll(θ)        # closed form, production: u_i ~ TN(μ0·h_i, (σᵤ0·h_i)²)
        ε  = yg .- Xg * θ[ib]
        h  = exp.(Zg * θ[idl])
        μ  = θ[imu] .* h
        σu = exp(θ[isu] / 2) .* h
        σv = exp(θ[isv] / 2)
        σ  = sqrt.(σu .^ 2 .+ σv^2)
        μs = (σv^2 .* μ .- σu .^ 2 .* ε) ./ σ .^ 2
        σs = σu .* σv ./ σ
        sum(-log.(σ) .+ Distributions.logpdf.(N01, (ε .+ μ) ./ σ) .+
            Distributions.logcdf.(N01, μs ./ σs) .- Distributions.logcdf.(N01, μ ./ σu))
    end
    θ_mle = [0.6175680886700637, 0.4199479179599756, -0.3085157593344751, -1.0161811317394707,
             -6.88982575506864, 23.744297256904915, -21.408756208138808, -1.914330497691085]
    # MersenneTwister gives other numbers for the same seed on Julia 1.10 (its seeding changed in 1.11):
    # the data are then not the case built here, so the checks that need it apply from Julia 1.11 on.
    @test _icsd_exact_ll(θ_mle) ≈ -36.94186274873071 atol = 1e-6 skip = VERSION < v"1.11"   # guards the stream
    θ_spur = copy(θ_mle)
    θ_spur[isu] = 2 * log(θ_mle[imu] / 511)                 # the top draw maps to the mode μ0
    m   = sfmodel_method(method = :MCI, n_draws = 512, multiRand = false)
    opt = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 300, g_abstol = 1e-8))
    r = redirect_stdout(devnull) do
        sfmodel_fit(spec = spec, method = m, init = θ_spur, optim_options = opt,
                    jlms_bc_index = false, marginal = false, integration_check = true, show_table = false,
                    verbose = false)
    end
    ic = r.integration_check
    @test r.converged skip = VERSION < v"1.11"
    @test r.redflag == 0 skip = VERSION < v"1.11"
    @test r.loglikelihood - _icsd_exact_ll(collect(r.coeff)) ≈ n * log(512 * Distributions.pdf(N01, 0)) atol = 0.05
    @test ic.status === :FAIL
    @test ic.ll_verdict === :FAIL
    @test ic.n_draws_check == 2048
    @test ic.delta_loglik ≈ n * log(1 / 4) atol = 0.01
    @test ic.shift_available && ic.max_shift_se < 0.1
end
