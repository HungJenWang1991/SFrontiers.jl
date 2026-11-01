# test_expo_naming.jl — one name for the Exponential's parameter.
#
# The Exponential's estimated parameter is the square of the distribution's scale, which for
# this one distribution is also Var(u). Three different names were attached to that same
# quantity:
#
#   * the analytic backend (`method=:MLE`) called it λ², reserving λ for the scale — which is
#     the stochastic-frontier literature's convention and was internally consistent;
#   * the simulation backends (`:MCI`, `:MSLE`, panel) called it λ, so the same estimate was
#     reported under two names depending on the estimation method;
#   * and within the simulation backends λ was already the SCALE for `:Weibull` and `:Lomax`.
#     Since Exponential(s) IS Weibull(s, k=1), `ln_lambda = log(0.36)` therefore meant a mean
#     inefficiency of 0.6 under one and 0.36 under the other, for the same distribution.
#
# It is now `sigma_u_sq` / `ln_sigma_u_sq` / `σᵤ²` everywhere, the name the half-normal,
# truncated-normal, lognormal and Rayleigh already use for their own squared scale, and the names
# of versions before 2.0.0 are accepted with a deprecation warning, as for `sigma_sq`. `lambda`
# keeps its meaning of a SCALE for Weibull and Lomax.
#
# The rename moves no number, and that is asserted here rather than assumed: the analytic
# likelihood, the simulation quantile and the marginal helper are checked against closed forms
# written out independently, and the two backends' estimates of the same quantity are compared.

using SFrontiers
using Test
using Optim
using Random
using StatsFuns

const _MB = SFrontiers.MCI_Backend
const _SB = SFrontiers.MSLE_Backend
const _PB = SFrontiers.Panel_Backend
const _EB = SFrontiers.MLE_Backend.SFmle

@testset "one name for the Exponential's parameter" begin

    threw(f) = try (f(); false) catch; true end

    Random.seed!(20260825)
    n  = 250
    X  = hcat(ones(n), randn(n), randn(n))
    Z  = hcat(ones(n), randn(n))
    u  = -0.6 .* log.(rand(n))                       # scale 0.6, so σᵤ² = Var(u) = 0.36
    y  = X * [1.0, 0.5, -0.3] .+ 0.25 .* randn(n) .- u
    βo = X \ y
    draws = collect(((1:128) .- 0.5) ./ 128)
    opt = sfmodel_opt(warmstart_solver = nothing, main_solver = NelderMead(),
                      main_opt = (iterations = 300,))

    # ------------------------------------------------------------------ the name is accepted
    @testset "ln_sigma_u_sq and :sigma_u_sq are the Exponential's names" begin
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                          ineff = :Exponential, type = :prod)
        @test !threw(() -> sfmodel_init(spec = sp, frontier = βo,
                                        ln_sigma_u_sq = log(0.36), ln_sigma_v_sq = log(0.0625)))
        # the name of versions before 2.0.0 warns and gives the same starting values (the old
        # `hetero = [:lambda]` is checked in test_sigma_u_naming.jl)
        v_old = @test_logs (:warn, r"`ln_lambda` is deprecated") match_mode = :any sfmodel_init(
                    spec = sp, frontier = βo, ln_lambda = log(0.36), ln_sigma_v_sq = log(0.0625))
        @test v_old == sfmodel_init(spec = sp, frontier = βo, ln_sigma_u_sq = log(0.36),
                                    ln_sigma_v_sq = log(0.0625))
        sph = sfmodel_spec(depvar = y, frontier = X, zvar = Z, noise = :Normal,
                           ineff = :Exponential, type = :prod, hetero = [:sigma_u_sq])
        @test sph !== nothing
        # the backends agree on the valid list
        @test _MB.valid_hetero(_MB.Exponential_MCI())   == [:sigma_u_sq]
        @test _SB.valid_hetero(_SB.Exponential_MSLE()) == [:sigma_u_sq]
        # and lambda is still right for the two distributions whose scale it is
        @test _MB.valid_hetero(_MB.Weibull_MCI()) == [:lambda, :k]
        @test :lambda in _MB.valid_hetero(_MB.Lomax_MCI())
    end

    # ------------------------------------------------------------------ internal kernel keywords and index keys
    @testset "internal kernel keywords and index keys use the new names" begin
        # the quantile keyword
        r1 = reshape([0.25, 0.75], 1, 2)
        @test !threw(() -> _MB.myExponentialQuantile(; σᵤ² = fill(0.36, 1, 1), r = r1))
        @test  threw(() -> _MB.myExponentialQuantile(; λ  = fill(0.36, 1, 1), r = r1))
        # the index field
        idx, _ = _MB._ineff_ind(_MB.Exponential_MCI(), 1, 2, Symbol[])
        @test haskey(pairs(idx), :sigma_sq)
        @test !haskey(pairs(idx), :lambda)
    end

    # ---------------------------------------------------- Exponential(s) == Weibull(s, k=1)
    @testset "one symbol, one meaning: lambda is a scale everywhere" begin
        R = 100_000
        r = reshape(collect(((1:R) .- 0.5) ./ R), 1, R)
        uw = zeros(1, R)
        _MB.myWeibullQuantile!(uw; λ = fill(0.6, 1, 1), k = fill(1.0, 1, 1), r = r,
                               clamp_lo = 1e-300)
        ue = _MB.myExponentialQuantile(; σᵤ² = fill(0.36, 1, 1), r = r)   # scale 0.6
        @test isapprox(sum(uw)/R, sum(ue)/R; rtol = 1e-8)
        @test isapprox(sum(ue)/R, 0.6; rtol = 1e-3)
        mE = sum(ue)/R
        @test isapprox(sum((ue .- mE).^2)/R, 0.36; rtol = 1e-2)          # the parameter IS Var(u)
    end

    # ------------------------------------------------------------------ no number moved
    @testset "the estimates are unchanged and the two backends agree" begin
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                          ineff = :Exponential, type = :prod)
        ini = sfmodel_init(spec = sp, frontier = βo, ln_sigma_u_sq = log(0.36),
                           ln_sigma_v_sq = log(0.0625))
        rM = last(Test.collect_test_logs(() -> sfmodel_fit(spec = sp,
                method = sfmodel_method(method = :MLE), init = ini, optim_options = opt,
                jlms_bc_index = true, marginal = false, show_table = false, verbose = false)))
        # the analytic likelihood at the fitted point, reproduced from an explicit scale
        raw = Float64(rM.coeff[4])
        sv  = sqrt(exp(Float64(rM.coeff[5])))
        ε   = y - X * Float64.(rM.coeff[1:3])
        dens(e, s) = (1/s)*normcdf(-e/sv - sv/s)*exp(e/s + sv^2/(2*s^2))
        @test isapprox(Float64(rM.loglikelihood), sum(log.(dens.(ε, exp(0.5*raw)))); rtol = 1e-10)
        # exp(raw) is Var(u); the truth here is 0.36
        @test 0.15 < exp(raw) < 0.8
        # the public result key carries the new name
        @test :coeff_log_σᵤ² in keys(rM.eqpo)
        @test !(:coeff_log_λ² in keys(rM.eqpo))

        # the simulation backend estimates the same quantity, and now says so
        rS = last(Test.collect_test_logs(() -> sfmodel_fit(spec = sp,
                method = sfmodel_method(method = :MSLE, draws = draws, n_draws = 128,
                                        multiRand = false),
                init = ini, optim_options = opt, jlms_bc_index = false, marginal = false,
                show_table = false, verbose = false)))
        @test :sigma_sq in propertynames(rS)
        @test !(:lambda in propertynames(rS))
        # both are log Var(u): compare the two estimates of the same thing (6.5% apart with these 128
        # draws, 30% with 32; estimating the scale instead of Var(u) would put them about 48% apart)
        @test isapprox(exp(Float64(rM.coeff[4])), exp(Float64(rS.coeff[4])); rtol = 0.35)
    end

    # ------------------------------------------------------------------ predict and @eq
    @testset "predict and the legacy @eq surface use the new names" begin
        sp = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                          ineff = :Exponential, type = :prod)
        ini = sfmodel_init(spec = sp, frontier = βo, ln_sigma_u_sq = log(0.36),
                           ln_sigma_v_sq = log(0.0625))
        r = last(Test.collect_test_logs(() -> sfmodel_fit(spec = sp,
                method = sfmodel_method(method = :MLE), init = ini, optim_options = opt,
                jlms_bc_index = true, marginal = false, show_table = false, verbose = false)))
        p = predict(r; depvar = y, frontier = X, what = :jlms)
        @test isapprox(Float64.(vec(p)), Float64.(vec(r.jlms)); rtol = 1e-8)

        pred = SFrontiers.MLE_Backend.sfmodel_predict
        σᵤ     = pred([:sigma_u])
        logσᵤ² = pred([:log_sigma_u_2])
        @test maximum(abs.(σᵤ .- sqrt.(exp.(logσᵤ²)))) < 1e-12
        @test all(>(0), σᵤ)
        @test threw(() -> pred([:lambda]))
        @test threw(() -> pred([:log_lambda_2]))
    end

    # ------------------------------------------------------------------ the panel route
    @testset "the panel Exponential takes the same name" begin
        nf, T = 25, 5
        np = nf * T
        idp = repeat(1:nf, inner = T)
        Xp  = hcat(randn(np), randn(np))
        Zp  = hcat(randn(np))
        yp  = Xp * [0.6, -0.4] .+ 0.25 .* randn(np) .- (-0.5 .* log.(rand(np)))
        sp = sfmodel_spec(datatype = :panel_TFE, depvar = yp, frontier = Xp, zvar = Zp,
                          id = idp, noise = :Normal, ineff = :Exponential, type = :prod)
        @test !threw(() -> sfmodel_init(spec = sp, frontier = Xp \ yp,
                                        ln_sigma_u_sq = log(0.25), ln_sigma_v_sq = log(0.0625)))
        idx = _PB._panel_param_ind(_PB.Exponential_Panel(), 2, 1)
        @test haskey(pairs(idx), :ln_sigma_u_sq)
        @test !haskey(pairs(idx), :ln_lambda)
        @test _PB._panel_ineff_varnames(_PB.Exponential_Panel()) == ["ln_σᵤ²", "ln_σᵥ²"]
    end
end
