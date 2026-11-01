# The memory estimate of one Hessian evaluation of the simulated likelihoods, and the warning a GPU fit
# prints before the optimizer when the estimate exceeds the free GPU memory. Runs on the CPU: the warning
# is called with given byte counts, and the estimate is compared with the bytes that one ForwardDiff
# Hessian evaluation of each likelihood allocates (cross-sectional MCI and MSLE with and without a
# Gaussian copula, panel MCI and MSLE). The check inside GPU fits is tested by the GPU-gated probe
# gpu_memory_probe.jl.

using Test
using SFrontiers
using ForwardDiff
using Random

@testset "memory estimate and GPU warning" begin
    _GD = SFrontiers.SFDraws

    @testset "ForwardDiff's chunk size" begin
        @test all(q -> _GD._fd_default_chunk(q) == ForwardDiff.pickchunksize(q), 1:40)
    end

    @testset "the estimate" begin
        @test _GD.sf_hessian_pass_bytes(3, 5000, 8192, 10, 8) == 7_962_624_000
        @test _GD.sf_hessian_pass_bytes(6, 5000, 8192, 10, 12) == 33_226_752_000
        @test _GD.sf_hessian_pass_bytes(2, 1000, 8192, 10, 8) == 1_061_683_200
        @test _GD.sf_hessian_pass_bytes(1, 1000, 8192, 10, 8) == 530_841_600
        @test _GD.sf_hessian_pass_bytes(2, 5000, 8192, 1, 5) == 23_592_960_000
        @test _GD.sf_hessian_pass_bytes(3, 271, 8192, 10, 16) == 546_766_848   # q = 16: C = 8; groups of 28, the last of 19
        @test _GD.sf_hessian_pass_bytes(2, 1009, 8192, 10, 8) == 1_603_141_632 # groups of 101, the last of 100
        @test _GD.sf_hessian_pass_bytes(1, 3, 1, 5, 5) == 288                  # chunks above nrows: groups of one row
        @test _GD.sf_hessian_pass_bytes(1, 1, 1, 1, 13) == 512                  # C = 7
        @test _GD.sf_hessian_pass_bytes(1, 1, 1, 1, 20) == 968                  # C = 10
        @test _GD.sf_hessian_pass_bytes(1, 1, 1, 0, 5) == 288                   # chunks below 1 counts as 1
    end

    @testset "the warning" begin
        # The warning says that memory falls as `chunks` grows and with a smaller `n_draws`, and names no value
        # of `chunks`, also when no value of `chunks` would bring the estimate below the free memory.
        hint = r"sharply\. Memory falls in proportion to `chunks`\. A smaller `n_draws` \(a smaller power of two\) also"
        est, low, high = 7_962_624_000, 5_000_000_000, 8_000_000_000
        kw = (nrows = 5000, unit = "observations", n_draws = 8192, chunks = 10, nparams = 8)
        @test (@test_logs (:warn, hint) _GD.sf_warn_gpu_memory(est, low; kw...)) == true
        @test (@test_logs (:warn, r"n_draws = 8192, chunks = 10") _GD.sf_warn_gpu_memory(est, low; kw...)) == true
        @test (@test_logs _GD.sf_warn_gpu_memory(est, high; kw...)) == false
        @test (@test_logs _GD.sf_warn_gpu_memory(est, nothing; kw...)) == false
        @test (@test_logs (:warn, hint) _GD.sf_warn_gpu_memory(10^15, 10^6; nrows = 100,
                   unit = "firms", n_draws = 2^15, chunks = 10, nparams = 12)) == true
    end

    @testset "no CUDA, no check" begin
        if !isdefined(Main, :CUDA)
            @test _GD.sf_gpu_free_bytes() === nothing
            @test (@test_logs _GD.sf_check_gpu_memory(6, 5000, 8192, 10, 12; unit = "observations")) == false
        end
    end

    @testset "the estimate matches what an evaluation allocates" begin
        # Bytes that one ForwardDiff Hessian evaluation of f allocates, after two warm-up calls: the
        # smallest of three readings (@allocated reads a process-wide counter, to which something else,
        # such as a finalizer run by a collection, can add during a single reading).
        function _gmw_hb(f, θ)
            cfg = ForwardDiff.HessianConfig(nothing, θ)
            h() = ForwardDiff.hessian(f, θ, cfg, Val{false}())
            h(); h()
            return minimum(@allocated(h()) for _ in 1:3)
        end
        # Allocated / estimated bytes when this test was written: 1.10, 1.14, 1.05 and 1.07 for the
        # cross-sectional cases, 1.06 and 1.12 for the panel cases; with a shorter last group (below)
        # 1.07, 1.09, 1.04 and 1.05, and 1.04 and 1.06. The rest of the bytes are dual vectors of one
        # element per observation or firm, which do not grow with the number of draws. What Julia and
        # ForwardDiff allocate around the work arrays can change between their versions, so the band is
        # [L, U] = [0.5, 1.5]: wide enough for that, and narrow enough for the warning, which compares the
        # estimate with the free GPU memory.
        L = 0.5
        U = 1.5
        W = _GD.SF_WORK_ARRAYS
        n = 400                    # observations or firms; chunks = 10 gives groups of 40
        cases = []                 # (label, work arrays, n_draws, q, likelihood, θ)

        # Cross-section: normal-truncated-normal with μ = Zδ (q = 6), and with a Gaussian copula (q = 7).
        rng = Xoshiro(4062)
        x = randn(rng, n); z = randn(rng, n)
        y = 1 .+ 0.5 .* x .+ 0.3 .* randn(rng, n) .- abs.(0.5 .* randn(rng, n))
        X = hcat(ones(n), x); Z = hcat(ones(n), z)
        for (cop, tag, kmci, kmsle) in ((:None, "", W.mci, W.msle),
                                        (:Gaussian, " with a Gaussian copula", W.mci_copula, W.msle_copula))
            us = sfmodel_spec(depvar = y, frontier = X, zvar = Z, noise = :Normal, ineff = :TruncatedNormal,
                              hetero = [:mu], copula = cop)
            a = SFrontiers.MCI_Backend._assemble_MCI_spec(us.mci_spec,
                    sfmodel_method(method = :MCI, n_draws = 512, chunks = 10))
            b = SFrontiers.MSLE_Backend._assemble_MSLE_spec(us.msle_spec,
                    sfmodel_method(method = :MSLE, n_draws = 512, chunks = 10))
            qa = SFrontiers.MCI_Backend.plen(a.model, a.K, a.L, a.hetero; L_scaling = a.L_scaling)
            qb = SFrontiers.MSLE_Backend.plen(b.model, b.K, b.L, b.hetero; L_scaling = b.L_scaling)
            push!(cases, ("MCI" * tag, kmci, 512, qa, (θ; chunks = 10) -> SFrontiers.MCI_Backend.MCI_nll(a, θ; chunks),
                          vcat(X \ y, fill(0.1, qa - 2))))
            push!(cases, ("MSLE" * tag, kmsle, 512, qb, (θ; chunks = 10) -> SFrontiers.MSLE_Backend.qmc_nll(b, θ; chunks),
                          vcat(X \ y, fill(0.1, qb - 2))))
        end

        # panel_TFE, half-normal: 400 firms × 4 periods, K = 2, L = 1 (q = 5). 4,096 draws: the panel
        # likelihood also allocates dual vectors per observation and per firm (about 5.8 MB per
        # evaluation here); at 512 draws one group buffer is 5.9 MB and the ratios would be about 1.5 and 2.
        rng = Xoshiro(44)
        Nf, Tp = 400, 4
        NT = Nf * Tp
        idp = repeat(1:Nf, inner = Tp)
        Xp = hcat(randn(rng, NT), randn(rng, NT))
        zp = randn(rng, NT)
        yp = repeat(randn(rng, Nf), inner = Tp) .+ Xp * [0.6, -0.4] .+ 0.3 .* randn(rng, NT) .-
             exp.(0.3 .* zp) .* repeat(abs.(0.4 .* randn(rng, Nf)), inner = Tp)
        spp = sfmodel_spec(depvar = yp, frontier = Xp, zvar = zp, id = idp, noise = :Normal, ineff = :HalfNormal,
                           type = :prod, datatype = :panel_TFE)
        θp = [0.6, -0.4, 0.3, log(0.16), log(0.09)]
        for (m, k) in ((:MCI, W.panel_mci), (:MSLE, W.panel_msle))
            isp = SFrontiers.Panel_Backend._assemble_panel_spec(spp.panel_spec,
                      sfmodel_panel_method(method = m, n_draws = 4096, chunks = 10))
            qp = SFrontiers.Panel_Backend.panel_plen(isp.model.ineff, isp.K, isp.L)
            push!(cases, ("panel $m", k, 4096, qp, (θ; chunks = 10) -> SFrontiers.Panel_Backend.panel_nll(isp, θ; chunks),
                          θp))
        end

        @testset "$label" for (label, narrays, S, q, f, θ) in cases
            est = _GD.sf_hessian_pass_bytes(narrays, n, S, 10, q)
            @test L * est <= _gmw_hb(f, θ) <= U * est
        end

        # chunks = 9 splits the 400 observations or firms into 8 groups of 45 and a last group of 40, for which
        # the likelihood allocates one more array of its own size; the estimate counts it.
        @testset "$label, shorter last group" for (label, narrays, S, q, f, θ) in cases
            est = _GD.sf_hessian_pass_bytes(narrays, n, S, 9, q)
            @test L * est <= _gmw_hb(θ -> f(θ; chunks = 9), θ) <= U * est
        end
    end
end
