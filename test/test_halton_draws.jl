# Halton draws of the simulation backends (MCI, MSLE, and panel_TFE fits by MCI or MSLE). By default
# observation i (firm i in panel models) receives points (i − 1)·n_draws + 1, …, i·n_draws of one base-2
# Halton sequence of N × n_draws points, so no point is used twice; single-precision draws are kept below 1;
# `distinct_Halton_length` reproduces the draws of versions before 2.0.0; every result reports its draw
# scheme and the length of the Halton sequence its draws came from.

using SFrontiers, Test, Random, HaltonSequences, Optim, QuadGK, SpecialFunctions

@testset "Halton draws" begin
    MCI_Backend = SFrontiers.MCI_Backend
    MSLE_Backend = SFrontiers.MSLE_Backend
    Panel_Backend = SFrontiers.Panel_Backend
    SFDraws = SFrontiers.SFDraws

    # Cross-sectional data with n observations, y = 1 + 0.5x + v − u, in element type T.
    function _hd_xdata(n; T = Float64)
        rng = Xoshiro(n)
        X = hcat(ones(n), randn(rng, n))
        y = X * [1.0, 0.5] .+ 0.3 .* randn(rng, n) .- abs.(0.4 .* randn(rng, n))
        return T.(X), T.(y)
    end
    # The normal–half-normal production frontier on those data.
    _hd_xspec(X, y; copula = :None) = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                                                  ineff = :HalfNormal, copula = copula, type = :prod)
    # A half-normal panel_TFE spec with Nf firms observed over Tp periods, in element type T.
    function _hd_pspec(Nf, Tp; T = Float64)
        rng = Xoshiro(Nf)
        id = repeat(1:Nf, inner = Tp)
        Xp = randn(rng, Nf * Tp, 2)
        z = randn(rng, Nf * Tp)
        yp = repeat(randn(rng, Nf), inner = Tp) .+ Xp * [0.6, -0.4] .+ 0.3 .* randn(rng, Nf * Tp) .-
             exp.(0.5 .* z) .* repeat(abs.(0.5 .* randn(rng, Nf)), inner = Tp)
        return sfmodel_spec(depvar = T.(yp), frontier = T.(Xp), zvar = T.(z), id = id, noise = :Normal,
                            ineff = :HalfNormal, type = :prod, datatype = :panel_TFE)
    end
    # A fit held at its starting values (no optimizer iteration), without marginal effects or table.
    _hd_fit0(sp, meth, init) = sfmodel_fit(spec = sp, method = meth, init = init,
                                           optim_options = sfmodel_opt(main_solver = Newton(),
                                                                       main_opt = (iterations = 0,)),
                                           marginal = false, show_table = false, verbose = false)
    # The value of f(); the log messages f emits (here the deprecation warning of `distinct_Halton_length`)
    # stay out of the test output.
    _hd_quiet(f) = last(Test.collect_test_logs(f))

    @testset "radical inverse" begin
        @test all(k -> SFDraws._radinv2(UInt64(k)) == HaltonSequences.haltonvalue(k, 2), 1:2^16)
        for k in (2^24 - 1, 2^24, 2^25 - 1, 2^25, 2^40 + 12345, 2^53 - 1)
            @test SFDraws._radinv2(UInt64(k)) == HaltonSequences.haltonvalue(k, 2)
        end
    end

    @testset "default: one sequence, no point reused" begin
        # observation i takes points (i − 1)·S + 1, …, i·S of one base-2 Halton sequence
        _hd_refm(N, S, T) = [T(haltonvalue((i - 1) * S + j, 2)) for i in 1:N, j in 1:S]
        for T in (Float64, Float32)
            sp = _hd_xspec(_hd_xdata(37; T = T)...)
            @test MCI_Backend._assemble_MCI_spec(sp.mci_spec,
                      sfmodel_method(method = :MCI, n_draws = 64)).draws_2D == _hd_refm(37, 64, T)
            @test MSLE_Backend._assemble_MSLE_spec(sp.msle_spec,
                      sfmodel_method(method = :MSLE, n_draws = 64)).draws_2D == _hd_refm(37, 64, T)
        end
        psp = _hd_pspec(37, 3)
        @test Panel_Backend._assemble_panel_spec(psp.panel_spec,
                  sfmodel_panel_method(method = :MSLE, n_draws = 64)).draws == _hd_refm(37, 64, Float64)
        # distinct points, checked in Float64: beyond 2^24 points Float32 merges values legitimately
        @test length(unique(_hd_refm(37, 64, Float64))) == 37 * 64
        # shared draws (multiRand = false): every observation takes the first n_draws points
        sp = _hd_xspec(_hd_xdata(37)...)
        @test MCI_Backend._assemble_MCI_spec(sp.mci_spec, sfmodel_method(method = :MCI, n_draws = 64,
                  multiRand = false)).draws_2D == reshape(MCI_Backend.make_halton_p(64), 1, 64)
        # no cap on n_draws, with and without a copula
        for cop in (:None, :Gaussian)
            sp4 = _hd_xspec(_hd_xdata(4)...; copula = cop)
            @test size(MCI_Backend._assemble_MCI_spec(sp4.mci_spec,
                      sfmodel_method(method = :MCI, n_draws = 40_000)).draws_2D) == (4, 40_000)
        end
    end

    @testset "one point per interval" begin
        # the 64 draws of every observation fall one into each interval [k/64, (k + 1)/64), k = 0, …, 63
        sp = _hd_xspec(_hd_xdata(50)...)
        M = MCI_Backend._assemble_MCI_spec(sp.mci_spec, sfmodel_method(method = :MCI, n_draws = 64)).draws_2D
        @test all(i -> sort(floor.(Int, M[i, :] .* 64)) == 0:63, 1:50)
    end

    @testset "single precision near 2^25 and the range check" begin
        for p in (2^25 - 2, 2^25 - 1, 2^25, 2^25 + 1)
            @test SFDraws.sf_halton_point(Float32, p) < 1f0
        end
        @test SFDraws.sf_halton_point(Float32, 2^25 - 1) == prevfloat(1f0)
        @test SFDraws.sf_halton_point(Float64, 2^25 - 1) == 1 - 2.0^-25
        @test Float32(1 - 2.0^-25) == 1f0          # why the clamp is needed: this point rounds to 1 in Float32
        @test_throws ErrorException SFDraws._check_open_unit(Float32[0.5 1.0])
        @test_throws ErrorException SFDraws._check_open_unit([0.0 0.5])
        @test SFDraws._check_open_unit([0.25 0.75]) === nothing
    end

    @testset "earlier draws through distinct_Halton_length" begin
        # The construction of versions before 2.0.0: one base-2 Halton sequence shortened to 2^n − 1 points,
        # the largest such length not above N·S or C, recycled across the rows.
        function _hd_ref(N, S, C; T = Float64)
            n = min(floor(Int, log2(N * S + 1)), floor(Int, log2(C + 1)))
            L = 2^n - 1
            h = T.(collect(HaltonSequences.Halton(2, length = L)))
            return [h[mod1((i - 1) * S + j, L)] for i in 1:N, j in 1:S]
        end
        # (N, S, C): N·S above C; N·S below C (rounded down to 2^14 − 1 points); a long C; the published
        # Monte Carlo design (500 observations, 4,096 draws, C = 2^24 − 1, which gave 2^20 − 1 points).
        # Panels have N firms observed over two periods.
        settings = ((80, 1024, 2^15 - 1), (20, 1024, 2^15 - 1), (300, 1024, 2^18 - 1), (500, 4096, 2^24 - 1))
        for (T, cases) in ((Float64, settings), (Float32, settings[1:3])), (N, S, C) in cases
            ref = _hd_ref(N, S, C; T = T)
            sp = _hd_xspec(_hd_xdata(N; T = T)...)
            psp = _hd_pspec(N, 2; T = T)
            @test MCI_Backend._assemble_MCI_spec(sp.mci_spec, _hd_quiet(() -> sfmodel_method(method = :MCI,
                      n_draws = S, distinct_Halton_length = C))).draws_2D == ref
            @test MSLE_Backend._assemble_MSLE_spec(sp.msle_spec, _hd_quiet(() -> sfmodel_method(method = :MSLE,
                      n_draws = S, distinct_Halton_length = C))).draws_2D == ref
            @test Panel_Backend._assemble_panel_spec(psp.panel_spec, _hd_quiet(() -> sfmodel_panel_method(
                      method = :MSLE, n_draws = S, distinct_Halton_length = C))).draws == ref
            @test MCI_Backend.make_halton_wrap(N, S; T = T, distinct_Halton_length = C) == ref
            @test MSLE_Backend.make_halton_wrap(N, S; T = T, distinct_Halton_length = C) == ref
            @test Panel_Backend.make_panel_halton_wrap(N, S; T = T, distinct_Halton_length = C) == ref
        end
        # the argument layout of versions before 2.0.0, which ends in the length, keeps their rule
        for m in (MCI_Backend.SFMethodSpec(:MCI, nothing, nothing, 64, true, false, 10, 32767),
                  MSLE_Backend.SFMethodSpec_MSLE(:MSLE, nothing, 64, true, false, 10, 32767),
                  Panel_Backend.PanelMethodSpec(:MCI, nothing, nothing, 64, true, false, 10, 32767))
            @test m.recycling == 32767
            @test m.halton_legacy == true
        end
        # one observation and 64 draws: the sequence is shortened to 63 points, fewer than the draws
        @test_throws ArgumentError MCI_Backend.make_halton_wrap(1, 64)
    end

    @testset "draw scheme and sequence length on the result" begin
        X, y = _hd_xdata(80)
        sp = _hd_xspec(X, y)
        psp = _hd_pspec(20, 3)
        xinit = vcat(X \ y, log(0.16), log(0.09))          # frontier, ln σᵤ², ln σᵥ²
        pinit = [0.6, -0.4, 0.5, log(0.16), log(0.09)]     # frontier, scaling, ln σᵤ², ln σᵥ²
        # N = observations (firms for the panel); distinct_Halton_length = 1023: N·S = 5,120 and 1,280 both
        # give 2^10 − 1 points
        for (b, s, N, init) in ((:MCI, sp, 80, xinit), (:MSLE, sp, 80, xinit), (:MCI, psp, 20, pinit))
            r = _hd_fit0(s, sfmodel_method(method = b, n_draws = 64), init)
            @test r.draw_scheme == :halton
            @test r.distinct_Halton_length == N * 64
            r = _hd_fit0(s, _hd_quiet(() -> sfmodel_method(method = b, n_draws = 64,
                                                           distinct_Halton_length = 1023)), init)
            @test r.draw_scheme == :halton_recycled
            @test r.distinct_Halton_length == 1023
            if s === sp
                @test predict(r; frontier = X, depvar = y, what = :jlms) == r.jlms
            end
            r = _hd_fit0(s, sfmodel_method(method = b, n_draws = 64, multiRand = false), init)
            @test r.draw_scheme == :halton_shared
            @test r.distinct_Halton_length == 64
            r = _hd_fit0(s, sfmodel_method(method = b, n_draws = 64, draws = rand(Xoshiro(1), N, 64)), init)
            @test r.draw_scheme == :user
            @test r.distinct_Halton_length == 0
        end
    end

    @testset "draw preparation allocates one block" begin
        _hd_alloc(f) = (f(); @allocated f())
        for T in (Float64, Float32)
            y = T.(randn(Xoshiro(3), 400))
            block = 400 * 1024 * sizeof(T)
            @test _hd_alloc(() -> MCI_Backend._prepare_draws(y, 400, T, nothing, 1024, true, 0, false)) <=
                  1.05 * block + 4096
            @test _hd_alloc(() -> MSLE_Backend._prepare_draws(y, 400, T, nothing, 1024, true, 0, false)) <=
                  1.05 * block + 4096
            # shared draws: one row of 1,024 points
            @test _hd_alloc(() -> MCI_Backend._prepare_draws(y, 400, T, nothing, 1024, false, 0, false)) <=
                  1.05 * 1024 * sizeof(T) + 4096
            @test _hd_alloc(() -> MSLE_Backend._prepare_draws(y, 400, T, nothing, 1024, false, 0, false)) <=
                  1.05 * 1024 * sizeof(T) + 4096
            # the generator call of the panel fits
            @test _hd_alloc(() -> SFDraws.sf_halton_draws(400, 1024; T = T)) <= 1.05 * block + 4096
        end
    end

    # Gaussian copula with truncated-normal inefficiency: the simulated log-likelihood approaches the exact
    # value (adaptive quadrature) as draws are added.
    @testset "copula: more draws bring MCI closer to the exact log-likelihood" begin
        # the copula test's 300-observation sample, Gaussian copula, held at mu/sigma_u = -8.5 (sigma_u = 0.2), theta_rho = 0.5.
        rng = Xoshiro(77); n2 = 300
        x2 = randn(rng, n2); X2 = hcat(ones(n2), x2)
        v2 = 0.25 .* randn(rng, n2)
        u2 = Float64[]
        while length(u2) < n2
            d = 0.2 + 0.4 * randn(rng); d > 0 && push!(u2, d)
        end
        y2 = X2 * [1.0, 0.5] .+ v2 .- u2
        sp2 = sfmodel_spec(depvar = y2, frontier = X2, noise = :Normal, ineff = :TruncatedNormal,
                           copula = :Gaussian, type = :prod)
        θ2 = [1.0, 0.5, -8.5 * 0.2, log(0.04), log(0.0625), 0.5]
        # the exact log-likelihood by quadrature over the inefficiency's probability r
        function _hd_exact(θ)
            mu = θ[3]; su = exp(θ[4] / 2); sv = exp(θ[5] / 2); ρ = 0.999 * tanh(θ[6])
            ε = y2 .- X2 * θ[1:2]
            brk = (0.0, 1e-9, 1e-6, 1e-3, 0.5, 1 - 1e-3, 1 - 1e-6, 1 - 1e-9, 1.0)
            ll = 0.0
            for i in 1:n2
                f(r) = begin
                    z = (ε[i] + MSLE_Backend._tn_quantile(mu, su, r, sqrt(2), inv(sqrt(2)))) / sv
                    qu = -sqrt(2) * erfcinv(2r)
                    exp(-z^2 / 2 - log(sqrt(2π)) - log(sv) - 0.5 * log(1 - ρ^2) -
                        (ρ^2 * (z^2 + qu^2) - 2ρ * z * qu) / (2 * (1 - ρ^2)))
                end
                ll += log(quadgk(f, brk...; rtol = 1e-12)[1])
            end
            return ll
        end
        ex = _hd_exact(θ2)
        _hd_err(Dd) = -MCI_Backend.MCI_nll(MCI_Backend._assemble_MCI_spec(sp2.mci_spec, sfmodel_method(method = :MCI, n_draws = Dd)),
                                           θ2; chunks = 10) - ex
        e1, e16 = _hd_err(1024), _hd_err(16384)
        @test abs(e16) < abs(e1)
        @test abs(e16) < 0.15
        @test abs(e1) < 1.0
    end

    @testset "recycling reuses points only beyond the cap" begin
        # 10 observations, 8 draws, a cap of 31 points: observation i takes the points at positions
        # mod1((i − 1)·8 + j, 31), so the 80 draws use each of the 31 points
        M = SFDraws.sf_halton_draws(10, 8; period = SFDraws.sf_halton_period(10, 8; recycling = 31))
        @test M == [haltonvalue(mod1((i - 1) * 8 + j, 31), 2) for i in 1:10, j in 1:8]
        @test length(unique(M)) == 31
        # a cap of N × n_draws = 80 points reuses none
        @test SFDraws.sf_halton_draws(10, 8; period = SFDraws.sf_halton_period(10, 8; recycling = 80)) ==
              SFDraws.sf_halton_draws(10, 8)
        # the same draws through the constructors
        sp10 = _hd_xspec(_hd_xdata(10)...)
        @test MCI_Backend._assemble_MCI_spec(sp10.mci_spec,
                  sfmodel_method(method = :MCI, n_draws = 8, recycling = 31)).draws_2D == M
        @test MSLE_Backend._assemble_MSLE_spec(sp10.msle_spec,
                  sfmodel_method(method = :MSLE, n_draws = 8, recycling = 31)).draws_2D == M
        # recycling = true caps the sequence at 2^24 − 1 points, more than N × n_draws here
        m = sfmodel_method(method = :MCI, n_draws = 8, recycling = true)
        @test m.recycling == 2^24 - 1
        @test m.halton_legacy == false
        @test MCI_Backend._assemble_MCI_spec(sp10.mci_spec, m).draws_2D == SFDraws.sf_halton_draws(10, 8)
        # on the result, for fits held at their starting values (80 observations, 64 draws: N × n_draws = 5,120)
        X, y = _hd_xdata(80)
        sp = _hd_xspec(X, y)
        xinit = vcat(X \ y, log(0.16), log(0.09))          # frontier, ln σᵤ², ln σᵥ²
        for b in (:MCI, :MSLE)
            r = _hd_fit0(sp, sfmodel_method(method = b, n_draws = 64, recycling = 1000), xinit)
            @test r.draw_scheme == :halton_recycled
            @test r.distinct_Halton_length == 1000
            @test predict(r; frontier = X, depvar = y, what = :jlms) == r.jlms
            r = _hd_fit0(sp, sfmodel_method(method = b, n_draws = 64, recycling = true), xinit)
            @test r.draw_scheme == :halton_recycled
            @test r.distinct_Halton_length == 5120         # min(N × n_draws, 2^24 − 1)
        end
    end

    @testset "identical rows are reported" begin
        # 6 observations, 3 draws, a cap of 15 points: 3 divides 15, so observations i and i + 15/3 = i + 5
        # receive identical draws, and the assembly warns
        sp6 = _hd_xspec(_hd_xdata(6)...)
        M = (@test_logs (:warn, r"identical draws") MCI_Backend._assemble_MCI_spec(sp6.mci_spec,
                 sfmodel_method(method = :MCI, n_draws = 3, recycling = 15))).draws_2D
        @test M[1, :] == M[6, :]
        # 4 draws share no factor with 15: rows would repeat only every 15 observations, so 6 observations
        # bring no warning
        logs, _ = Test.collect_test_logs() do
            MCI_Backend._assemble_MCI_spec(sp6.mci_spec, sfmodel_method(method = :MCI, n_draws = 4, recycling = 15))
        end
        @test !any(l -> occursin("identical draws", string(l.message)), logs)
        # panel models count firms: 6 firms observed over 2 periods
        psp6 = _hd_pspec(6, 2)
        draws = (@test_logs (:warn, r"identical draws") Panel_Backend._assemble_panel_spec(psp6.panel_spec,
                     sfmodel_panel_method(method = :MSLE, n_draws = 3, recycling = 15))).draws
        @test draws[1, :] == draws[6, :]
        # With a sequence of L points, identical rows lie L/gcd(n_draws, L) rows apart. The warning offers a
        # longer sequence only below 2^24 − 1 points, and an n_draws with no factor in common with L only when
        # the sample has no more observations than L (adding "any power of two" when L is odd).
        at_cap = r"Use no recycling or an n_draws that shares no factor with 16777215 \(any power of two\)\.$"
        even_cap = r"Use no recycling, a longer sequence or an n_draws that shares no factor with 1000\.$"
        many_obs = r"Use no recycling or a longer sequence\.$"
        @test (@test_logs (:warn, at_cap) SFDraws.sf_warn_repeated_rows(5000, 4095, 2^24 - 1)) == 4097
        @test (@test_logs (:warn, r"identical draws") SFDraws.sf_warn_repeated_rows(5000, 4095, 2^15 - 1)) == 4681
        @test (@test_logs (:warn, r"identical draws") SFDraws.sf_warn_repeated_rows(2000, 1023, 2^15 - 1)) == 1057
        @test (@test_logs (:warn, even_cap) SFDraws.sf_warn_repeated_rows(200, 64, 1000)) == 125
        @test (@test_logs (:warn, many_obs) SFDraws.sf_warn_repeated_rows(2000, 64, 1023)) == 1023
        # 4,096 draws share no factor with 2^24 − 1, so identical rows would lie 2^24 − 1 rows apart: no warning
        @test (@test_logs SFDraws.sf_warn_repeated_rows(5000, 4096, 2^24 - 1)) === nothing
    end

    @testset "length checks" begin
        # an integer cap below n_draws
        @test_throws ArgumentError sfmodel_method(method = :MCI, n_draws = 128, recycling = 100)
        # with the deprecated keyword the sequence is shortened to 2^15 − 1 = 32,767 points (the largest
        # 2^m − 1 not above N × n_draws = 120,000 or 50,000), fewer than the 40,000 draws
        sp3 = _hd_xspec(_hd_xdata(3)...)
        @test_throws ArgumentError MCI_Backend._assemble_MCI_spec(sp3.mci_spec,
                  _hd_quiet(() -> sfmodel_method(method = :MCI, n_draws = 40_000, distinct_Halton_length = 50_000)))
    end

    @testset "keywords" begin
        # the deprecation warning of distinct_Halton_length appears once, under its own log id
        logs, _ = Test.collect_test_logs() do
            sfmodel_method(method = :MCI, n_draws = 8, distinct_Halton_length = 32767)
            sfmodel_method(method = :MSLE, n_draws = 8, distinct_Halton_length = 32767)
        end
        @test count(l -> l.id === :sf_deprecated_distinct_Halton_length, logs) == 1
        # giving both keywords is an error
        @test_throws ArgumentError _hd_quiet(() -> sfmodel_method(method = :MCI, n_draws = 8, recycling = true,
                                                                   distinct_Halton_length = 32767))
        # recycling applies only to the per-observation draws the package generates
        @test_throws ArgumentError sfmodel_method(method = :MCI, n_draws = 8, recycling = true, draws = rand(8))
        @test_throws ArgumentError sfmodel_method(method = :MCI, n_draws = 8, recycling = true, multiRand = false)
        # an integer cap runs from n_draws to 2^24 − 1
        @test_throws ArgumentError sfmodel_method(method = :MCI, n_draws = 8, recycling = 0)
        @test_throws ArgumentError sfmodel_method(method = :MCI, n_draws = 8, recycling = -5)
        @test_throws ArgumentError sfmodel_method(method = :MCI, n_draws = 8, recycling = 2^24)
        @test_throws "recycling = true" sfmodel_method(method = :MCI, n_draws = 8, recycling = 1)
        # the panel constructor, and a cross-sectional method converted for a panel fit
        m = sfmodel_panel_method(method = :MCI, n_draws = 8, recycling = 31)
        @test m.recycling == 31
        @test m.halton_legacy == false
        m = _hd_quiet(() -> sfmodel_panel_method(method = :MCI, n_draws = 8, distinct_Halton_length = 1023))
        @test (m.recycling, m.halton_legacy) == (1023, true)
        @test SFrontiers._to_panel_method(sfmodel_method(method = :MSLE, n_draws = 8, recycling = true)).recycling ==
              2^24 - 1
    end
end
