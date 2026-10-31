# The optional integration check of MCI and MSLE fits, `sfmodel_fit(..., integration_check = ...)`:
# its settings, the verdict computed from its inputs, the handling of errors inside the check, the
# out-of-memory test, and fits with the check on. In the fits, the check's log-likelihood and shift are
# recomputed independently from a fit's draws with four times `n_draws` and four times `chunks`; the check
# leaves the fit unchanged; its line is printed with the results table; it is not available with
# user-supplied draws or a recycled Halton sequence shorter than 4 × `n_draws`; `:MLE` ignores it; and a
# variance matrix that cannot be inverted leaves only the log-likelihood part. The test set "panel_TFE" checks
# panel MCI and MSLE fits in the same way. With a recycled sequence, the check's own draws bring no warning
# about identical draws.

using Test
using SFrontiers
using Random
using Optim: Newton
using ForwardDiff
import Distributions

@testset "the integration check" begin
    SFA = SFrontiers.SFAccuracy
    DEF = SFA.SF_INTEGRATION_DEFAULTS

    # Half-normal production data, 200 observations.
    function _ict_spec()
        rng = MersenneTwister(31)
        n = 200
        xh = randn(rng, n)
        Xh = hcat(ones(n), xh)
        yh = Xh * [1.0, 0.5] .+ 0.3 .* randn(rng, n) .- abs.(0.6 .* randn(rng, n))
        return sfmodel_spec(depvar = yh, frontier = Xh, noise = :Normal, ineff = :HalfNormal, type = :prod)
    end
    _ict_opt() = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 100, g_abstol = 1e-6))
    _ict_fit(spec, meth, opt; show_table = false, integration_check = false) =
        sfmodel_fit(spec = spec, method = meth, optim_options = opt, jlms_bc_index = true, marginal = false,
                    integration_check = integration_check, show_table = show_table, verbose = false)
    # The value of f() and the text it printed.
    function _ict_capture(f)
        path, io = mktemp()
        r = redirect_stdout(f, io)
        close(io)
        out = read(path, String)
        rm(path; force = true)
        return r, out
    end

    @testset "settings" begin
        @test SFA.sf_integration_settings(false) === nothing
        @test SFA.sf_integration_settings(true) == DEF
        @test SFA.sf_integration_settings((ll_caution = 0.2,)) ==
              (ll_caution = 0.2, ll_fail = 1.0, shift_caution = 0.1, shift_fail = 0.5)
        @test_throws ArgumentError SFA.sf_integration_settings((foo = 1,))
        msg = try
            SFA.sf_integration_settings((foo = 1,))
            ""
        catch err
            sprint(showerror, err)
        end
        @test all(k -> occursin(k, msg), ("ll_caution", "ll_fail", "shift_caution", "shift_fail"))
        @test_throws ArgumentError SFA.sf_integration_settings((ll_caution = 1.0, ll_fail = 0.5))
        @test_throws ArgumentError SFA.sf_integration_settings((shift_fail = -1.0,))
        @test_throws ArgumentError SFA.sf_integration_settings((ll_fail = Inf,))
        @test_throws ArgumentError SFA.sf_integration_settings(1)
        # the option belongs to sfmodel_fit; the options builders refuse it
        @test_throws MethodError sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 1,),
                                             integration_check = true)
        @test_throws MethodError SFrontiers.MSLE_Backend.sfmodel_MSLE_opt(main_solver = Newton(),
                                                                          main_opt = (iterations = 1,),
                                                                          integration_check = true)
        o = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 1,))
        @test !hasproperty(o, :integration_check)
        @test !hasproperty(SFrontiers.MSLE_Backend.sfmodel_MSLE_optim(nothing, nothing, Newton(), o.main_opt),
                           :integration_check)
    end

    @testset "an invalid setting is an error before the fit" begin
        spec = _ict_spec()
        for meth in (sfmodel_method(method = :MCI, n_draws = 64), sfmodel_method(method = :MLE))
            path, io = mktemp()
            err = try
                redirect_stdout(io) do
                    sfmodel_fit(spec = spec, method = meth, optim_options = _ict_opt(),
                                integration_check = (ll_caution = 2.0, ll_fail = 1.0), show_table = true)
                end
                nothing
            catch e
                e
            end
            close(io)
            out = read(path, String)
            rm(path; force = true)
            @test err isa ArgumentError
            @test occursin("ll_caution must be below ll_fail", sprint(showerror, err))
            @test isempty(out)              # nothing printed: no fit started
        end
    end

    @testset "the verdict" begin
        I2 = [1.0 0.0; 0.0 1.0]
        _ict_verdict(; ll = 0.0, ll_check = 0.0, V = I2, se = [1.0, 1.0], hessian_flag = 0, g_diff = [0.0, 0.0]) =
            SFA.sf_integration_verdict(; ll, ll_check, n_obs = 200, S = 256, S4 = 1024, V, se, hessian_flag, g_diff,
                                       theta = [0.5, 1.5], names = ["a", "b"], thresholds = DEF, seconds = 0.0)
        # the change in the log-likelihood; with ll = 0 it carries no rounding
        @test _ict_verdict(ll_check = -0.0999).ll_verdict === :OK
        @test _ict_verdict(ll_check = -0.1).ll_verdict === :CAUTION
        @test _ict_verdict(ll_check = 0.999).ll_verdict === :CAUTION
        @test _ict_verdict(ll_check = -1.0).ll_verdict === :FAIL
        @test _ict_verdict(ll_check = NaN).ll_verdict === :FAIL
        # the one-step shift in standard errors
        v = _ict_verdict(g_diff = [-0.0999, 0.0])
        @test v.shift_available && v.shift_verdict === :OK && v.max_shift_param == "a"
        # the parameter's estimate and its value after the shift; the line gives them in four decimals
        @test v.max_shift_estimate == 0.5 && v.max_shift_onestep == 0.5 + 0.0999
        @test endswith(SFA.sf_integration_line(v), "largest shift: 0.1 SE, a from 0.5000 to 0.5999)")
        v2 = _ict_verdict(g_diff = [0.0, 0.25])
        @test v2.max_shift_param == "b" && v2.max_shift_estimate == 1.5 && v2.max_shift_onestep == 1.25
        @test endswith(SFA.sf_integration_line(v2), "largest shift: 0.25 SE, b from 1.5000 to 1.2500)")
        @test _ict_verdict(g_diff = [-0.1, 0.0]).shift_verdict === :CAUTION
        @test _ict_verdict(g_diff = [-0.5, 0.0]).shift_verdict === :FAIL
        # the status is the worse part
        @test _ict_verdict().status === :OK
        @test _ict_verdict(ll_check = -0.1, g_diff = [-0.5, 0.0]).status === :FAIL
        @test _ict_verdict(ll_check = -1.0, g_diff = [-0.1, 0.0]).status === :FAIL
        @test _ict_verdict(ll_check = -0.1).status === :CAUTION
        @test _ict_verdict(g_diff = [0.0, -0.1]).status === :CAUTION
        # the shift is not computed: the status is the log-likelihood's
        nc = (_ict_verdict(hessian_flag = 1, V = fill(NaN, 2, 2)),
              _ict_verdict(hessian_flag = 2, se = [1.0, NaN]),
              _ict_verdict(hessian_flag = 1),
              _ict_verdict(g_diff = [NaN, 0.0]))
        @test occursin("could not be computed or inverted", nc[1].shift_note)
        @test occursin("some standard errors are NaN", nc[2].shift_note)
        @test occursin("not positive definite", nc[3].shift_note)
        @test occursin("gradient", nc[4].shift_note)
        for w in nc
            @test !w.shift_available
            @test w.status === w.ll_verdict
            @test w.max_shift_param == "" && isnan(w.max_shift_estimate) && isnan(w.max_shift_onestep)
            @test occursin("log-likelihood only", SFA.sf_integration_line(w))
        end
    end

    @testset "errors inside the check" begin
        args = (S = 256, n_obs = 10, ll = -1.0, theta = [0.0, 0.0], nll_S = p -> sum(abs2, p),
                V = [1.0 0.0; 0.0 1.0], se = [1.0, 1.0], hessian_flag = 0, names = ["a", "b"], unavailable = nothing)
        _ict_run(build; kw...) = SFA.sf_run_integration_check(DEF; merge(args, (; kw...))..., build_nll_4S = build)
        r, out = _ict_capture(() -> _ict_run(() -> error("boom")))
        @test r.status === :error && occursin("boom", r.reason)
        @test occursin("The integration check could not be completed (boom).", out)
        r, out = _ict_capture(() -> _ict_run(() -> throw(OutOfMemoryError())))
        @test r.status === :not_available && occursin("out of memory", r.reason)
        @test occursin("The integration check could not be completed (out of memory: lower n_draws or raise chunks).",
                       out)
        @test_throws InterruptException _ict_run(() -> throw(InterruptException()))
        r = _ict_run(() -> error("must not run"); ll = NaN)
        @test r.status === :not_available && occursin("not finite", r.reason)
        # a check that runs has the same fields, with the estimate in `theta` and its value after the shift
        ok = _ict_run(() -> (p -> sum(abs2, p .- [0.1, 0.0])); theta = [0.0, 0.0])
        @test ok.shift_available && ok.max_shift_param == "a"
        @test ok.max_shift_estimate == 0.0 && ok.max_shift_onestep ≈ 0.2
        @test keys(r) == keys(ok)
        ki = findfirst(==(:max_shift_param), keys(ok))
        @test keys(ok)[ki + 1] === :max_shift_estimate && keys(ok)[ki + 2] === :max_shift_onestep
        @test isnan(r.max_shift_estimate) && isnan(r.max_shift_onestep)
    end

    @testset "out-of-memory errors" begin
        oom = SFrontiers.SFRedflag.sf_out_of_memory
        @test oom(OutOfMemoryError())
        @test oom(ErrorException("Out of GPU memory trying to allocate 1 GiB"))
        @test oom(CompositeException(Any[OutOfMemoryError()]))
        @test !oom(InterruptException())
        @test !oom(StackOverflowError())
        @test !oom(ErrorException("boom"))
    end

    @testset "cross-sectional $m fit" for m in (:MCI, :MSLE)
        spec = _ict_spec()
        meth = sfmodel_method(method = m, n_draws = 256)
        r = _ict_fit(spec, meth, _ict_opt(); integration_check = true)
        ic = r.integration_check
        @test ic.status in (:OK, :CAUTION)
        @test ic.n_draws == 256
        @test ic.n_draws_check == 1024
        @test ic.shift_available
        # the draws of a fit with n_draws = 1024 and the same settings, in 4 × chunks = 40 groups
        nll = m === :MCI ? SFrontiers.MCI_Backend.MCI_nll : SFrontiers.MSLE_Backend.qmc_nll
        meth4 = sfmodel_method(method = m, n_draws = 1024, chunks = 40)
        isp4 = m === :MCI ? SFrontiers.MCI_Backend._assemble_MCI_spec(spec.mci_spec, meth4) :
                            SFrontiers.MSLE_Backend._assemble_MSLE_spec(spec.msle_spec, meth4)
        @test ic.loglik_check == -nll(isp4, r.coeff; chunks = 40)
        gS = ForwardDiff.gradient(p -> nll(r.model, p; chunks = r.model.chunks), r.coeff)
        g4 = ForwardDiff.gradient(p -> nll(isp4, p; chunks = 40), r.coeff)
        @test ic.shift ≈ -r.var_cov_mat * (g4 - gS) rtol = 1e-10
        @test ic.shift_se ≈ ic.shift ./ r.std_err
        # the parameter of the largest shift, its estimate and its value after the shift
        k = argmax(abs.(ic.shift_se))
        @test ic.max_shift_param == r.model.varnames[k]
        @test ic.max_shift_estimate == r.coeff[k] && ic.max_shift_onestep == r.coeff[k] + ic.shift[k]
        # the check leaves the fit unchanged
        r0 = _ict_fit(spec, meth, _ict_opt())
        @test r0.integration_check === nothing
        for f in (:coeff, :loglikelihood, :std_err, :var_cov_mat, :redflag, :jlms, :bc)
            @test isequal(getproperty(r, f), getproperty(r0, f))
        end
        # the printed line, after "Converged:"; the log-likelihood line keeps its form
        rp, out = _ict_capture(() -> _ict_fit(spec, meth, _ict_opt(); show_table = true, integration_check = true))
        @test occursin("Integration check: ", out)
        @test occursin("Integration check: " * SFA.sf_integration_line(rp.integration_check), out)
        # the estimate on the line is printed as in the table: the same string stands in the table's row
        ml = match(r"largest shift: [0-9.]+ SE, (\S+) from (-?[0-9.]+) to (-?[0-9.]+)\)", out)
        @test ml !== nothing && ml.captures[1] == rp.integration_check.max_shift_param
        @test abs(parse(Float64, ml.captures[2]) - rp.integration_check.max_shift_estimate) <= 5.0001e-5
        @test occursin(ml.captures[2], replace(out, ml.match => ""))
        mm = match(r"Log-likelihood: *(-?[0-9.]+)", out)
        @test mm !== nothing && parse(Float64, mm.captures[1]) == round(rp.loglikelihood; digits = 5)
        _, out0 = _ict_capture(() -> _ict_fit(spec, meth, _ict_opt(); show_table = true))
        @test !occursin("Integration check", out0)
        @test occursin("integration", sprint(show, MIME"text/plain"(), r))
    end

    @testset "not available" begin
        spec = _ict_spec()
        opt = _ict_opt()
        # user-supplied draws (the draws set the count)
        mu = sfmodel_method(method = :MCI, draws = rand(MersenneTwister(1), 1, 256))
        r = _ict_fit(spec, mu, opt; integration_check = true)
        @test r.integration_check.status === :not_available
        @test occursin("user-supplied", r.integration_check.reason)
        @test isequal(r.coeff, _ict_fit(spec, mu, opt).coeff)
        # a recycled sequence of 1,023 points, one fewer than 4 × 256
        mr = sfmodel_method(method = :MCI, n_draws = 256, recycling = 1023)
        r = _ict_fit(spec, mr, opt; integration_check = true)
        @test r.integration_check.status === :not_available
        @test occursin("recycled", r.integration_check.reason)
        @test isequal(r.coeff, _ict_fit(spec, mr, opt).coeff)
        # the deprecated keyword: the rule of versions before 2.0.0 also gives 1,023 points
        _, r = Test.collect_test_logs() do
            _ict_fit(spec, sfmodel_method(method = :MCI, n_draws = 256, distinct_Halton_length = 1023), opt;
                     integration_check = true)
        end
        @test r.integration_check.status === :not_available
        @test occursin("recycled", r.integration_check.reason)
    end

    @testset ":MLE ignores the option" begin
        r = _ict_fit(_ict_spec(), sfmodel_method(method = :MLE), _ict_opt(); integration_check = true)
        @test !hasproperty(r, :integration_check)
    end

    @testset "a variance matrix that cannot be inverted" begin
        rng = MersenneTwister(32)
        n = 200
        uw = rand(rng, Distributions.Weibull(2.0, 1.0), n)
        xw = randn(rng, n)
        Xw = hcat(ones(n), xw)
        yw = Xw * [1.0, 0.5] .+ 0.3 .* randn(rng, n) .- uw
        spec = sfmodel_spec(depvar = yw, frontier = Xw, noise = :Normal, ineff = :Weibull, type = :prod)
        meth = sfmodel_method(method = :MCI, n_draws = 256)
        idx = SFrontiers.MCI_Backend._assemble_MCI_spec(spec.mci_spec, meth).idx
        θ = zeros(idx.noise.ln_sigma_v_sq)            # the noise parameter comes last
        θ[idx.beta] = Xw \ yw
        θ[idx.ineff.lambda] .= 0.0
        θ[idx.ineff.k] .= log(20.0)                   # beyond the Weibull ceiling 10: the Hessian has a zero row
        θ[idx.noise.ln_sigma_v_sq] = log(0.09)
        # no warmstart and no main iteration: the fit stays at θ
        r = sfmodel_fit(spec = spec, method = meth, init = θ,
                        optim_options = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,)),
                        jlms_bc_index = true, marginal = false, integration_check = true, show_table = false,
                        verbose = false)
        ic = r.integration_check
        @test r.redflag == 1
        @test all(isnan, r.std_err)
        @test !ic.shift_available
        @test occursin("could not be computed or inverted", ic.shift_note)
        @test ic.status === ic.ll_verdict
        @test isfinite(ic.delta_loglik)
    end

    @testset "panel_TFE" begin
        # 50 firms over 4 periods; half-normal inefficiency scaled by exp(0.5 z)
        rng = Xoshiro(41)
        id = repeat(1:50, inner = 4)
        Xp = randn(rng, 200, 2)
        zp = randn(rng, 200)
        ap = repeat(randn(rng, 50), inner = 4)
        us = abs.(randn(rng, 50))
        yp = ap .+ Xp * [0.6, -0.4] .+ 0.3 .* randn(rng, 200) .- exp.(0.5 .* zp) .* repeat(us, inner = 4)
        spp = sfmodel_spec(depvar = yp, frontier = Xp, zvar = zp, id = id, noise = :Normal, ineff = :HalfNormal,
                           type = :prod, datatype = :panel_TFE)
        _ict_popt() = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 50,))
        @test_throws MethodError SFrontiers.sfmodel_panel_opt(main_solver = Newton(), main_opt = (iterations = 50,),
                                                              integration_check = true)
        @test !hasproperty(SFrontiers._to_panel_opt(_ict_popt()), :integration_check)
        for m in (:MCI, :MSLE)
            meth = sfmodel_method(method = m, n_draws = 64)
            r = _ict_fit(spp, meth, _ict_popt(); integration_check = true)
            @test hasproperty(r, :integration_check)
            ic = r.integration_check
            @test ic.n_draws_check == 256
            @test isfinite(ic.delta_loglik)
            @test ic.status in (:OK, :CAUTION)
            @test ic.shift_available
            # the draws of a fit with n_draws = 256 and the same settings, in min(50, 4 × 10) = 40 groups of firms
            isp4 = SFrontiers.Panel_Backend._assemble_panel_spec(spp.panel_spec,
                       SFrontiers.sfmodel_panel_method(method = m, n_draws = 256, chunks = 40))
            @test ic.loglik_check == -SFrontiers.Panel_Backend.panel_nll(isp4, r.coeff; chunks = 40)
            gS = ForwardDiff.gradient(p -> SFrontiers.Panel_Backend.panel_nll(r.model, p; chunks = r.model.chunks),
                                      r.coeff)
            g4 = ForwardDiff.gradient(p -> SFrontiers.Panel_Backend.panel_nll(isp4, p; chunks = 40), r.coeff)
            @test ic.shift ≈ -r.var_cov_mat * (g4 - gS) rtol = 1e-10
            @test ic.shift_se ≈ ic.shift ./ r.std_err
            @test ic.max_shift_param in r.model.varnames
            k = argmax(abs.(ic.shift_se))
            @test ic.max_shift_estimate == r.coeff[k] && ic.max_shift_onestep == r.coeff[k] + ic.shift[k]
            # the option through the exported panel fit
            rpo = SFrontiers.sfmodel_panel_fit(spec = spp.panel_spec,
                      method = SFrontiers.sfmodel_panel_method(method = m, n_draws = 64),
                      optim_options = SFrontiers.sfmodel_panel_opt(main_solver = Newton(),
                                                                   main_opt = (iterations = 50,)),
                      marginal = false, integration_check = true, show_table = false, verbose = false)
            @test rpo.integration_check !== nothing && rpo.integration_check.thresholds == DEF
            # the check leaves the fit unchanged
            r0 = _ict_fit(spp, meth, _ict_popt())
            @test r0.integration_check === nothing
            for f in (:coeff, :loglikelihood, :std_err, :var_cov_mat, :redflag, :jlms, :bc)
                @test isequal(getproperty(r, f), getproperty(r0, f))
            end
            # the printed line
            rpp, out = _ict_capture(() -> _ict_fit(spp, meth, _ict_popt(); show_table = true, integration_check = true))
            @test occursin("Integration check: ", out)
            @test occursin("Integration check: " * SFA.sf_integration_line(rpp.integration_check), out)
        end
        # user-supplied draws (the draws set the count)
        mu = sfmodel_method(method = :MCI, draws = rand(MersenneTwister(2), 1, 64))
        r = _ict_fit(spp, mu, _ict_popt(); integration_check = true)
        @test r.integration_check.status === :not_available
        @test occursin("user-supplied", r.integration_check.reason)
    end

    @testset "no warning about the check's own draws" begin
        _ict_identical(logs) = [string(l.message) for l in logs if occursin("identical draws", string(l.message))]
        # 200 observations, 64 draws from a recycled sequence of 13,824 = 2^9 · 27 points: the fit's 12,800 draws
        # use no point twice, while the check's 256 draws per observation give observations i and
        # i + 13,824 / gcd(256, 13,824) = i + 54 identical draws. The fits' printed output is captured.
        spec = _ict_spec()
        for m in (:MCI, :MSLE)
            logs, (r, _) = Test.collect_test_logs() do
                _ict_capture(() -> _ict_fit(spec, sfmodel_method(method = m, n_draws = 64, recycling = 13824),
                                            _ict_opt(); integration_check = true))
            end
            @test isempty(_ict_identical(logs))
            ic = r.integration_check
            @test ic.n_draws_check == 256
            @test isfinite(ic.delta_loglik)
            # the check used these draws, which bring the warning when they are assembled on their own
            meth4 = sfmodel_method(method = m, n_draws = 256, chunks = 40, recycling = 13824)
            isp4 = @test_logs (:warn, r"n_draws = 256 .* i \+ 54 receive identical draws") (m === :MCI ?
                       SFrontiers.MCI_Backend._assemble_MCI_spec(spec.mci_spec, meth4) :
                       SFrontiers.MSLE_Backend._assemble_MSLE_spec(spec.msle_spec, meth4))
            nll = m === :MCI ? SFrontiers.MCI_Backend.MCI_nll : SFrontiers.MSLE_Backend.qmc_nll
            @test ic.loglik_check == -nll(isp4, r.coeff; chunks = 40)
            # a fit whose own draws repeat still warns, once, with its own number of draws: a sequence of
            # 6,912 = 2^8 · 27 points gives observations i and i + 108 identical draws at 64 draws
            logs, (r, _) = Test.collect_test_logs() do
                _ict_capture(() -> _ict_fit(spec, sfmodel_method(method = m, n_draws = 64, recycling = 6912),
                                            _ict_opt(); integration_check = true))
            end
            w = _ict_identical(logs)
            @test length(w) == 1
            @test occursin("n_draws = 64 ", w[1]) && occursin("i + 108 ", w[1])
            @test r.integration_check.n_draws_check == 256
        end
        # panel_TFE, 50 firms over 5 periods, 64 draws from 3,456 = 2^7 · 27 points: the fit's 3,200 draws use no
        # point twice, while the check's 256 draws per firm give firms i and i + 27 identical draws
        rng = Xoshiro(42)
        idq = repeat(1:50, inner = 5)
        Xq = randn(rng, 250, 2)
        zq = randn(rng, 250)
        aq = repeat(randn(rng, 50), inner = 5)
        uq = abs.(randn(rng, 50))
        yq = aq .+ Xq * [0.6, -0.4] .+ 0.3 .* randn(rng, 250) .- exp.(0.5 .* zq) .* repeat(uq, inner = 5)
        spq = sfmodel_spec(depvar = yq, frontier = Xq, zvar = zq, id = idq, noise = :Normal, ineff = :HalfNormal,
                           type = :prod, datatype = :panel_TFE)
        for m in (:MCI, :MSLE)
            logs, (r, _) = Test.collect_test_logs() do
                _ict_capture(() -> _ict_fit(spq, sfmodel_method(method = m, n_draws = 64, recycling = 3456),
                    sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 50,)); integration_check = true))
            end
            @test isempty(_ict_identical(logs))
            ic = r.integration_check
            @test ic.n_draws_check == 256
            @test isfinite(ic.delta_loglik)
            isp4 = @test_logs (:warn, r"n_draws = 256 .* i \+ 27 receive identical draws") (
                       SFrontiers.Panel_Backend._assemble_panel_spec(spq.panel_spec,
                           SFrontiers.sfmodel_panel_method(method = m, n_draws = 256, chunks = 40, recycling = 3456)))
            @test ic.loglik_check == -SFrontiers.Panel_Backend.panel_nll(isp4, r.coeff; chunks = 40)
        end
    end
end
