# The warning after a fit whose estimated shape lies where the default MCI transformation rule (or MSLE at the
# default number of draws, for the Lomax) is unreliable, and the verdict stored in `result.shape_check`. The
# thresholds are read from SF_SHAPE_RANGES: "in" is 0.6 times a threshold, "out" 3 times. Every fit is held at
# its start values (no warmstart, no Newton iteration), so the data only define the model.
# `shape_warning = false` turns the warning off; the verdict is stored either way.

using Test
using SFrontiers
using Random
using Optim: Newton

@testset "the shape-range warning" begin
    SFA = SFrontiers.SFAccuracy
    PBK = SFrontiers.Panel_Backend
    _srw_row(m, f) = first(e for e in SFA.SF_SHAPE_RANGES if e.method === m && e.family === f)

    # The value of f() and the text it printed.
    function _srw_capture(f)
        old = stdout
        path, io = mktemp()
        local ret
        redirect_stdout(io)
        try
            ret = f()
        finally
            flush(io); redirect_stdout(old); close(io)
        end
        out = read(path, String)
        rm(path; force = true)
        (ret, out)
    end
    _srw_fit(sp, meth, θ; kw...) = _srw_capture(() -> sfmodel_fit(; spec = sp, method = meth, init = θ,
        optim_options = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,)),
        jlms_bc_index = false, marginal = false, show_table = false, verbose = false, kw...))

    # Cross-sectional data, 200 observations; θ through the assembled spec's idx: the scale parameter 0 on
    # the log scale, the log of the shape `lnshape` (one value, or one coefficient per column of Z).
    rng = MersenneTwister(50)
    Xc = hcat(ones(200), randn(rng, 200))
    yc = Xc * [1.0, 0.5] .+ 0.3 .* randn(rng, 200) .- abs.(randn(rng, 200))
    function _srw_theta(sp, m, fam, lnshape)
        meth = sfmodel_method(method = m, n_draws = 128)
        idx = m === :MCI ? SFrontiers.MCI_Backend._assemble_MCI_spec(sp.mci_spec, meth).idx :
                           SFrontiers.MSLE_Backend._assemble_MSLE_spec(sp.msle_spec, meth).idx
        θ = zeros(idx.noise.ln_sigma_v_sq)            # the noise parameter comes last
        θ[idx.beta] = Xc \ yc
        shape = fam === :Lomax ? idx.ineff.alpha : idx.ineff.k
        θ[shape] .= lnshape
        θ[idx.noise.ln_sigma_v_sq] = log(0.09)
        return θ
    end
    _srw_spec(fam; kw...) = sfmodel_spec(; depvar = yc, frontier = Xc, noise = :Normal, ineff = fam,
                                         type = :prod, kw...)

    @testset "helpers" begin
        @test SFA.sf_values_below(0.3, 0.5) == (n_below = 1, n = 1, share = 1.0, min_value = 0.3)
        @test SFA.sf_values_below(0.7, 0.5) == (n_below = 0, n = 1, share = 0.0, min_value = 0.7)
        @test SFA.sf_values_below([0.2, 0.9, 0.4, 0.6], 0.5) == (n_below = 2, n = 4, share = 0.5, min_value = 0.2)
        @test SFA.sf_shape_entry(:MCI, :HalfNormal, :logistic_1_rule) === nothing
        @test SFA.sf_shape_entry(:MCI, :Gamma, :logistic_2_rule) === nothing
        @test SFA.sf_shape_entry(:MSLE, :Weibull, nothing) === nothing
        @test SFA.sf_shape_entry(:MSLE, :Lomax, nothing) === _srw_row(:MSLE, :Lomax)
    end

    @testset "the value and the share in the warning" begin
        g, l = _srw_row(:MCI, :Gamma), _srw_row(:MSLE, :Lomax)
        _srw_text(e, v) = _srw_capture(() -> SFA.sf_shape_check(e, v))[2]
        # three significant digits, or as many more as it takes to print the value below the threshold
        @test occursin("parameter k = 0.3 is below the threshold of 0.6. ", _srw_text(g, 0.3))
        @test occursin("parameter k = 0.123 is below the threshold of 0.6. ", _srw_text(g, 0.12345))
        @test occursin("parameter k = 0.5996 is below the threshold of 0.6. ", _srw_text(g, 0.5996))
        @test occursin("parameter α = 29.96 is below the threshold of 30.0. ", _srw_text(l, 29.96))
        @test occursin("parameter k = 0.5996 is below the threshold of 0.6. ", _srw_text(g, fill(0.5996, 200)))
        # the share in percent with one decimal; "<0.1%" and ">99.9%" when some but not all values lie below
        @test occursin("parameter k is below the threshold of 0.6 for 67 of 200 observations " *
                       "(33.5%; smallest 0.5996). ", _srw_text(g, vcat(0.5996, fill(0.5998, 66), fill(0.9, 133))))
        @test occursin("for 1 of 3000 observations (<0.1%; smallest 0.2). ", _srw_text(g, vcat(0.2, fill(0.9, 2999))))
        @test occursin("for 1999 of 2000 observations (>99.9%; smallest 0.2). ",
                       _srw_text(g, vcat(fill(0.2, 1999), 0.9)))
        @test occursin("for 200 of 200 observations (100.0%; smallest 0.2). ",
                       _srw_text(g, vcat(0.2, fill(0.3, 199))))
        # the verdict's numbers are unchanged
        @test _srw_capture(() -> SFA.sf_shape_check(g, vcat(0.2, fill(0.9, 2999))))[1].share == 1 / 3000
    end

    @testset "the table names the default rules" begin
        for fam in (:Gamma, :Weibull, :Lomax)
            @test _srw_row(:MCI, fam).rule === SFrontiers.MCI_Backend.default_transformation_rule(fam)
        end
        @test _srw_row(:MCI, :Gamma).rule === PBK.default_panel_transformation(PBK.Gamma_Panel())
        @test _srw_row(:MCI, :Weibull).rule === PBK.default_panel_transformation(PBK.Weibull_Panel())
        @test _srw_row(:MCI, :Lomax).rule === PBK.default_panel_transformation(PBK.Lomax_Panel())
        @test _srw_row(:MSLE, :Lomax).rule === nothing
    end

    @testset "cross-sectional MCI $fam" for fam in (:Gamma, :Weibull, :Lomax)
        t = _srw_row(:MCI, fam).threshold
        sp = _srw_spec(fam)
        meth = sfmodel_method(method = :MCI, n_draws = 128)
        r, out = _srw_fit(sp, meth, _srw_theta(sp, :MCI, fam, log(0.6 * t)))
        @test occursin("Warning (MCI)", out)
        @test occursin(":logistic_2_rule", out)
        @test occursin("sfmodel_fit(..., integration_check = true)", out)
        @test r.shape_check.in_range
        @test r.shape_check.n_below == 200
        @test r.shape_check.threshold == t
        if fam === :Gamma
            @test occursin("no method = :MSLE", out)
        elseif fam === :Weibull
            @test occursin("method = :MSLE", out)
        end
        r, out = _srw_fit(sp, meth, _srw_theta(sp, :MCI, fam, log(3 * t)))
        @test !occursin("Warning (MCI)", out)
        @test !r.shape_check.in_range
        @test r.shape_check.n_below == 0
    end

    @testset "heteroscedastic Gamma shape" begin
        t = _srw_row(:MCI, :Gamma).threshold
        z = randn(MersenneTwister(51), 200)
        sp = _srw_spec(:Gamma; zvar = hcat(ones(200), z), hetero = [:k])
        r, out = _srw_fit(sp, sfmodel_method(method = :MCI, n_draws = 128), _srw_theta(sp, :MCI, :Gamma, [log(t), 1.0]))
        @test r.shape_check.n_below == count(z .< 0)
        @test r.shape_check.n == 200
        @test occursin("of 200 observations", out)
    end

    @testset "transformation rules" begin
        t = _srw_row(:MCI, :Gamma).threshold
        sp = _srw_spec(:Gamma)
        θ = _srw_theta(sp, :MCI, :Gamma, log(0.6 * t))
        # another rule: no warning, no verdict
        r, out = _srw_fit(sp, sfmodel_method(method = :MCI, n_draws = 128, transformation = :logistic_2_rule), θ)
        @test !occursin("Warning (MCI)", out)
        @test r.shape_check === nothing
        # the default rule passed explicitly counts as the default rule
        r, out = _srw_fit(sp, sfmodel_method(method = :MCI, n_draws = 128, transformation = :expo_rule), θ)
        @test occursin("Warning (MCI)", out)
        @test r.shape_check.in_range
    end

    # Panel data: 50 firms over 4 periods, half-normal inefficiency scaled by exp(0.5 z), for every family
    # (the fits are held at θ, so the family of the data does not matter).
    rngp = Xoshiro(41)
    id = repeat(1:50, inner = 4)
    Xp = randn(rngp, 200, 2)
    zp = randn(rngp, 200)
    ap = repeat(randn(rngp, 50), inner = 4)
    us = abs.(randn(rngp, 50))
    yp = ap .+ Xp * [0.6, -0.4] .+ 0.3 .* randn(rngp, 200) .- exp.(0.5 .* zp) .* repeat(us, inner = 4)
    _srw_pspec(fam) = sfmodel_spec(depvar = yp, frontier = Xp, zvar = zp, id = id, noise = :Normal, ineff = fam,
                                   type = :prod, datatype = :panel_TFE)
    function _srw_ptheta(spp, fam, shape)
        idx = spp.panel_spec.idx
        θ = zeros(idx.ln_sigma_v_sq)                  # the noise parameter comes last
        θ[idx.beta] = [0.6, -0.4]
        θ[idx.delta] .= 0.5
        θ[idx.ln_sigma_v_sq] = log(0.09)
        θ[fam === :Gamma ? idx.ln_theta : idx.ln_lambda] = 0.0
        θ[fam === :Lomax ? idx.ln_alpha : idx.ln_k] = log(shape)
        return θ
    end

    @testset "panel_TFE $m $fam" for (m, fam) in ((:MCI, :Gamma), (:MCI, :Weibull), (:MCI, :Lomax))
        t = _srw_row(m, fam).threshold
        spp = _srw_pspec(fam)
        meth = sfmodel_method(method = m, n_draws = 64)
        r, out = _srw_fit(spp, meth, _srw_ptheta(spp, fam, 0.6 * t))
        @test hasproperty(r, :shape_check)
        @test occursin("Warning ($m)", out)
        @test r.shape_check.in_range
        @test r.shape_check.n == 1
        @test r.shape_check.threshold == t
        r, out = _srw_fit(spp, meth, _srw_ptheta(spp, fam, 3 * t))
        @test !occursin("Warning ($m)", out)
        @test !r.shape_check.in_range
        @test r.shape_check.n == 1
    end

    @testset "a family that the table does not cover" begin
        sp = _srw_spec(:HalfNormal)
        meth = sfmodel_method(method = :MCI, n_draws = 128)
        idx = SFrontiers.MCI_Backend._assemble_MCI_spec(sp.mci_spec, meth).idx
        θ = zeros(idx.noise.ln_sigma_v_sq)
        θ[idx.beta] = Xc \ yc
        θ[idx.noise.ln_sigma_v_sq] = log(0.09)
        r, _ = _srw_fit(sp, meth, θ)
        @test r.shape_check === nothing
        spp = _srw_pspec(:HalfNormal)
        pidx = spp.panel_spec.idx
        θp = zeros(pidx.ln_sigma_v_sq)
        θp[pidx.beta] = [0.6, -0.4]
        θp[pidx.delta] .= 0.5
        θp[pidx.ln_sigma_v_sq] = log(0.09)
        rp, _ = _srw_fit(spp, sfmodel_method(method = :MCI, n_draws = 64), θp)
        @test hasproperty(rp, :shape_check)
        @test rp.shape_check === nothing
    end

    @testset "the full texts" begin
        g, w = _srw_row(:MCI, :Gamma), _srw_row(:MCI, :Weibull)
        l, ml = _srw_row(:MCI, :Lomax), _srw_row(:MSLE, :Lomax)
        _srw_text(e, v) = _srw_capture(() -> SFA.sf_shape_check(e, v))[2]
        q = "This warning flags a potential numerical issue; it does not establish that this fit is affected. "
        off = "This warning may be turned off with sfmodel_fit(..., shape_warning = false); the result still " *
              "records the shape check in result.shape_check.\n"
        ic = "To check this fit, use sfmodel_fit(..., integration_check = true); if the check fails, refit with "
        @test _srw_text(g, 0.42) == "Warning (MCI): The estimated Gamma shape parameter k = 0.42 is below the " *
            "threshold of 0.6. " * q * "For shapes in this range the default transformation rule (:expo_rule) " *
            "puts too few draws where the Gamma density spikes near zero, so the simulated log-likelihood, jlms " *
            "and bc can be far from their exact values even when the fit reports convergence, and k tends to be " *
            "estimated too high. " * ic * "transformation = :logistic_2_rule (the Gamma model has no method = " *
            ":MSLE). " * off
        @test _srw_text(w, 0.83) == "Warning (MCI): The estimated Weibull shape parameter k = 0.83 is below the " *
            "threshold of 1.0. " * q * "In this range the Weibull tail is heavier than the exponential tail of " *
            "the default rule's draws (:expo_rule), so the simulated log-likelihood, jlms and bc can be far from " *
            "their exact values even when the fit reports convergence. " * ic * "transformation = " *
            ":logistic_2_rule or with method = :MSLE. " * off
        @test _srw_text(l, 2.1) == "Warning (MCI): The estimated Lomax shape parameter α = 2.1 is below the " *
            "threshold of 3.0. " * q * "With a tail this heavy the default transformation rule " *
            "(:logistic_1_rule) puts few draws where observations far in the tail need them, so the simulated " *
            "log-likelihood, jlms and bc can be far from their exact values even when the fit reports " *
            "convergence. " * ic * "transformation = :logistic_2_rule. " * off
        @test _srw_text(ml, 1.87) == "Warning (MSLE): The estimated Lomax shape parameter α = 1.87 is below the " *
            "threshold of 30.0. " * q * "With a tail this heavy, MSLE may need substantially more draws than the " *
            "default to approximate the exact log-likelihood accurately, even when the fit reports convergence. " *
            ic * "method = :MCI and transformation = :logistic_2_rule, which is generally more accurate for this " *
            "model, or with substantially more draws. " * off
        # the verdict is the same with shape_warning = false, and nothing is printed
        v1, o1 = _srw_capture(() -> SFA.sf_shape_check(g, 0.42))
        v0, o0 = _srw_capture(() -> SFA.sf_shape_check(g, 0.42; shape_warning = false))
        @test o0 == "" && v0 == v1 && v0.in_range
        # the Lomax-mean warning: as predict prints it, after a fit, and with shape_warning = false
        _, a = _srw_capture(() -> SFA.sf_warn_lomax_mean(3, 200))
        @test a == "Warning: The Lomax mean E(u) = λ/(α − 1) exists only for α > 1, so the marginal effects on " *
                   "E(u) are undefined where α ≤ 1 and are reported as NaN (3 of 200 observations).\n"
        _, b = _srw_capture(() -> SFA.sf_warn_lomax_mean(3, 200; after_fit = true))
        @test b == chop(a) * " This warning may be turned off with sfmodel_fit(..., shape_warning = false); " *
                   "the marginal effects are NaN there either way.\n"
        _, c = _srw_capture(() -> SFA.sf_warn_lomax_mean(3, 200; shape_warning = false, after_fit = true))
        @test c == ""
    end

    @testset "shape_warning = false" begin
        # cross-sectional MCI Gamma (k = 0.36 with scale 1, σᵥ² = 0.09) and MSLE Lomax (α = 3, λ = 1) below their
        # thresholds: the warning by default, nothing with shape_warning = false, the same verdict and estimates
        for (m, fam, lnshape) in ((:MCI, :Gamma, log(0.6 * _srw_row(:MCI, :Gamma).threshold)),
                                  (:MSLE, :Lomax, log(3.0)))
            sp = _srw_spec(fam)
            meth = sfmodel_method(method = m, n_draws = 128)
            θ = _srw_theta(sp, m, fam, lnshape)
            r1, out1 = _srw_fit(sp, meth, θ)
            r0, out0 = _srw_fit(sp, meth, θ; shape_warning = false)
            @test occursin("Warning ($m)", out1)
            @test occursin("sfmodel_fit(..., shape_warning = false)", out1)
            @test !occursin("Warning", out0)
            @test r0.shape_check.in_range && r0.shape_check == r1.shape_check
            @test r0.coeff == r1.coeff
        end
        # panel_TFE MCI Lomax (50 firms over 5 periods; α = 1.8, λ = 1)
        rngq = Xoshiro(42)
        idq = repeat(1:50, inner = 5)
        Xq = randn(rngq, 250, 2)
        zq = randn(rngq, 250)
        yq = repeat(randn(rngq, 50), inner = 5) .+ Xq * [0.6, -0.4] .+ 0.3 .* randn(rngq, 250) .-
             exp.(0.5 .* zq) .* repeat(abs.(randn(rngq, 50)), inner = 5)
        spq = sfmodel_spec(depvar = yq, frontier = Xq, zvar = zq, id = idq, noise = :Normal, ineff = :Lomax,
                           type = :prod, datatype = :panel_TFE)
        θq = _srw_ptheta(spq, :Lomax, 0.6 * _srw_row(:MCI, :Lomax).threshold)
        mq = sfmodel_method(method = :MCI, n_draws = 64)
        r1, out1 = _srw_fit(spq, mq, θq)
        r0, out0 = _srw_fit(spq, mq, θq; shape_warning = false)
        @test occursin("Warning (MCI)", out1)
        @test !occursin("Warning", out0)
        @test r0.shape_check.in_range && r0.shape_check == r1.shape_check
        # the Lomax-mean warning of a panel fit (α = 0.8, marginal effects): none with shape_warning = false
        hold = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,))
        r8, out8 = _srw_capture(() -> sfmodel_fit(; spec = spq, method = mq, init = _srw_ptheta(spq, :Lomax, 0.8),
            optim_options = hold, jlms_bc_index = false, marginal = true, shape_warning = false,
            show_table = false, verbose = false))
        @test all(isnan, Matrix(r8.marginal))
        @test !occursin("Warning", out8)
        # :MLE takes the option and ignores it
        rm1, _ = _srw_capture(() -> sfmodel_fit(spec = _srw_spec(:HalfNormal), method = sfmodel_method(method = :MLE),
                                                shape_warning = false, show_table = false, verbose = false))
        @test isfinite(rm1.loglikelihood)
    end
end
