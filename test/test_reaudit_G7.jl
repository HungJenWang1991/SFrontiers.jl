# Regression tests for the 03_reaudit_findings.md Group 7 fixes.
# Group 7 = docs, plot recipes, legacy accessors & remaining judgment decisions:
#   R-01 (MoMTest cost E[u] recentring), R-22 (custom-draws multiRand=false),
#   R-28 (QQ panel title), FIX-09 (panel zvar-missing error), R-27 (store_trace
#   / plot(:convergence)), R-07 (sfmodel_boot_marginal), R-09/R-10 (sfmodel_predict).
#
# Each testset would have failed before its fix (asserts the number/label/behavior).

using SFrontiers
using Test
using Random
using Statistics
using DataFrames
using Optim
using RecipesBase

# Legacy MLE-backend entry points reached through their (unexported) module path.
const _MLE = SFrontiers.MLE_Backend

# A small Normal + HalfNormal dataset reused by several G7 testsets.
function _g7_hn_data(; N = 150, seed = 321)
    Random.seed!(seed)
    x1 = randn(N); x2 = randn(N); cons = ones(N)
    X  = hcat(cons, x1, x2)
    β  = [1.0, 0.5, -0.3]
    y  = X * β .+ 0.3 .* randn(N) .- abs.(0.5 .* randn(N))
    return X, y, X \ y
end

_g7_opt() = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40,),
                        main_solver = Newton(), main_opt = (iterations = 60, g_abstol = 1e-5))

# ---------------------------------------------------------------------
# R-01 — sfmodel_MoMTest cost-frontier JLMS/BC/llike recentring.
#
# After the L153 `Resid = _porc*resid` flip, the E[u] recentring must
# SUBTRACT E[u] unconditionally. The pre-fix `myeps = Resid - _porc*E[u]`
# ADDED E[u] for :cost (_porc=-1), shifting every conditioning residual by
# +2E[u]. A :cost run must therefore reproduce a sign-mirrored :prod run
# (cost on y ≡ production on -y) obs-by-obs.
# ---------------------------------------------------------------------
@testset "R-01 MoMTest cost recentring" begin
    Random.seed!(2718)
    Nm = 400
    xm1 = randn(Nm); xm2 = randn(Nm); consm = ones(Nm)
    Xm = hcat(consm, xm1, xm2)
    βm = [1.0, 0.5, -0.3]
    vm = 0.4 .* randn(Nm)
    um = abs.(0.8 .* randn(Nm))
    ym = Xm * βm .+ vm .+ um           # cost DGP: y = xβ + v + u
    df_cost = DataFrame(y = ym,   cons = consm, x1 = xm1, x2 = xm2)
    df_prod = DataFrame(negy = -ym, cons = consm, x1 = xm1, x2 = xm2)

    for dist in ([:h], [:e])
        rc = _MLE.sfmodel_MoMTest((:type,[:cost]), (:dist,dist), (:depvar,[:y]),
                                  (:frontier,[:cons,:x1,:x2]); data=df_cost, verbose=false)
        rp = _MLE.sfmodel_MoMTest((:type,[:prod]), (:dist,dist), (:depvar,[:negy]),
                                  (:frontier,[:cons,:x1,:x2]); data=df_prod, verbose=false)

        # cost ≡ sign-mirrored prod, bit-identical (fails pre-fix by ~2E[u]).
        @test maximum(abs.(vec(rc.jlms) .- vec(rp.jlms))) < 1e-10
        @test maximum(abs.(vec(rc.bc)   .- vec(rp.bc)))   < 1e-10
        @test abs(rc.MoM_loglikelihood - rp.MoM_loglikelihood) < 1e-8

        # Cost inefficiency index recovers the truth in mean (pre-fix it was
        # badly biased) and rises with the composed residual.
        resid = ym .- Xm * (Xm \ ym)
        @test argmax(resid) == argmax(vec(rc.jlms))
        @test rc.PorC == -1.0
    end

    # Half-normal cost jlms mean tracks the true mean inefficiency.
    rc = _MLE.sfmodel_MoMTest((:type,[:cost]), (:dist,[:h]), (:depvar,[:y]),
                              (:frontier,[:cons,:x1,:x2]); data=df_cost, verbose=false)
    @test isapprox(mean(vec(rc.jlms)), mean(um); atol = 0.1)
end

# ---------------------------------------------------------------------
# R-22 — custom-draws `meth4` pattern requires `multiRand=false`.
#
# Manual §6.2's `meth4` example (a 1×S row matrix of user draws) omitted the
# `multiRand=false` that §9.8 requires. The code is correct: with
# `multiRand=false` a 1×S draws matrix fits and `result.n_draws` reports the
# supplied length. Pin the documented (corrected) behavior for MSLE and MCI.
# ---------------------------------------------------------------------
@testset "R-22 custom draws multiRand=false" begin
    X, y, β_ols = _g7_hn_data()
    spec = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                        ineff = :HalfNormal, type = :prod)
    init = sfmodel_init(spec = spec, frontier = β_ols,
                        ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
    opt  = _g7_opt()

    S = 512
    my_draws = reshape(rand(S), 1, :)           # 1×S row matrix, the manual's meth4 shape
    for m in (:MSLE, :MCI)
        meth = sfmodel_method(method = m, draws = my_draws, multiRand = false)
        res  = sfmodel_fit(spec = spec, method = meth, init = init, optim_options = opt,
                           marginal = false, show_table = false, verbose = false)
        @test res.converged == true
        @test isfinite(res.loglikelihood)
        @test res.n_draws == S                  # reports the supplied length, not 1
    end
end

# ---------------------------------------------------------------------
# R-28 — QQ panel title reflects the self-standardized shape check.
#
# The QQ sub-panel standardizes v̂ by its own sample mean/std and plots against
# N(0,1) — so the old "N(0, σ̂_v²)" title over-claimed a scale-aware comparison.
# The title must now read as a standardized-normal shape check.
# ---------------------------------------------------------------------
@testset "R-28 QQ panel title" begin
    X, y, β_ols = _g7_hn_data()
    spec = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                        ineff = :HalfNormal, type = :prod)
    init = sfmodel_init(spec = spec, frontier = β_ols,
                        ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
    res  = sfmodel_fit(spec = spec, method = sfmodel_method(method = :MLE), init = init,
                       optim_options = _g7_opt(), marginal = false,
                       show_table = false, verbose = false)

    _titlerow(rd) = rd[findfirst(r -> haskey(r.plotattributes, :title), rd)].plotattributes[:title]

    # Default 2×2 panel (no marginal → QQ occupies panel 4).
    trow = _titlerow(RecipesBase.apply_recipe(Dict{Symbol,Any}(), res))
    @test trow[4] == "QQ: standardized v̂ vs N(0,1)"
    @test !occursin("σ̂_v²", trow[4])           # no scale-aware over-claim
    @test !occursin("N(0, σ", trow[4])

    # :residuals three-panel (QQ is panel 3).
    trow_r = _titlerow(RecipesBase.apply_recipe(Dict{Symbol,Any}(), res, Val(:residuals)))
    @test trow_r[3] == "QQ: standardized v̂ vs N(0,1)"
    @test !occursin("σ̂_v²", trow_r[3])
end

# ---------------------------------------------------------------------
# FIX-09 — a panel spec missing `zvar` raises an informative identification
# error, not the generic "Unsupported matrix input type: Nothing".
# ---------------------------------------------------------------------
@testset "FIX-09 panel zvar-missing error" begin
    Nf, T = 8, 4
    idv = repeat(1:Nf, inner = T)
    NT  = Nf * T
    Xp  = randn(NT, 2)
    yp  = randn(NT)

    _msg(f) = try; f(); ""; catch e; sprint(showerror, e); end

    # Direct panel keyword ctor with zvar=nothing.
    m1 = _msg(() -> SFrontiers.Panel_Backend.sfmodel_panel_spec(
                        depvar = yp, frontier = Xp, zvar = nothing, id = idv, type = :prod))
    @test occursin("FIX-09", m1)
    @test occursin("within-firm", m1)
    @test !occursin("Unsupported matrix input type", m1)

    # Unified keyword path (datatype=:panel_TFE) forwards zvar=nothing → same error.
    m2 = _msg(() -> sfmodel_spec(depvar = yp, frontier = Xp, id = idv,
                                 noise = :Normal, ineff = :HalfNormal,
                                 datatype = :panel_TFE, type = :prod))
    @test occursin("FIX-09", m2)
    @test !occursin("Unsupported matrix input type", m2)

    # A valid within-varying zvar still constructs (guard is not over-broad).
    zv = randn(NT)                                  # varies within firms
    spec = sfmodel_spec(depvar = yp, frontier = Xp, zvar = zv, id = idv,
                        noise = :Normal, ineff = :HalfNormal,
                        datatype = :panel_TFE, type = :prod)
    @test spec isa SFrontiers.UnifiedSpec
end

# ---------------------------------------------------------------------
# R-27 — `store_trace` captures the optimizer trace onto the result and
# powers `plot(:convergence)`; without it, the recipe errors.
# ---------------------------------------------------------------------
@testset "R-27 store_trace / plot(:convergence)" begin
    X, y, β_ols = _g7_hn_data()
    spec = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                        ineff = :HalfNormal, type = :prod)
    init = sfmodel_init(spec = spec, frontier = β_ols,
                        ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))

    _opt(store) = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 40,),
                              main_solver = Newton(), main_opt = (iterations = 60, g_abstol = 1e-5),
                              store_trace = store)
    _mk(m) = m === :MLE ? sfmodel_method(method = :MLE) :
                          sfmodel_method(method = m, n_draws = 100)

    # Cross-sectional MLE and MCI, both directions.
    for m in (:MLE, :MCI)
        res_on = sfmodel_fit(spec = spec, method = _mk(m), init = init,
                             optim_options = _opt(true), marginal = false,
                             show_table = false, verbose = false)
        @test res_on.trace !== nothing
        tr = res_on.trace
        @test length(tr.iteration) == length(tr.value) == length(tr.g_norm) >= 1
        @test tr.iteration[1] == 0
        @test !isempty(RecipesBase.apply_recipe(Dict{Symbol,Any}(), res_on, Val(:convergence)))

        res_off = sfmodel_fit(spec = spec, method = _mk(m), init = init,
                              optim_options = _opt(false), marginal = false,
                              show_table = false, verbose = false)
        @test res_off.trace === nothing                    # default: no trace stored
        @test_throws Exception RecipesBase.apply_recipe(Dict{Symbol,Any}(), res_off, Val(:convergence))
    end

    # Panel backend (explicit result NamedTuple → `trace` must be threaded through).
    Random.seed!(77)
    Nf, T = 12, 5; NTp = Nf * T
    idp = repeat(1:Nf, inner = T)
    Xp  = randn(NTp, 2)                                    # no intercept (panel_TFE)
    zp  = randn(NTp)                                       # within-varying zvar
    yp  = Xp * [0.5, -0.3] .+ 0.3 .* randn(NTp) .- abs.(0.4 .* randn(NTp))
    pspec = sfmodel_spec(depvar = yp, frontier = Xp, zvar = zp, id = idp,
                         noise = :Normal, ineff = :HalfNormal, datatype = :panel_TFE, type = :prod)
    popt = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 60,),
                       main_solver = NelderMead(), main_opt = (iterations = 200, g_abstol = 1e-4),
                       store_trace = true)
    pres = sfmodel_fit(spec = pspec, method = sfmodel_method(method = :MSLE, n_draws = 100),
                       optim_options = popt, marginal = false, show_table = false, verbose = false)
    @test pres.trace !== nothing
    @test keys(pres.trace) == (:iteration, :value, :g_norm)
end

# ---------------------------------------------------------------------
# R-09 — legacy `sfmodel_predict(@eq(lambda))` for Exponential MLE.
#
# The `@eq()` dispatch had no branch for the Exponential inefficiency scale, so
# `@eq(lambda)`/`@eq(λ)` fell through to `throw("@eq() is not specified correctly")`.
# Add it: λ = exp(0.5·(W·δ̂)) = sqrt(exp(W·δ̂)); `@eq(log_λ²)` is the raw predictor.
# ---------------------------------------------------------------------
@testset "R-09 sfmodel_predict Exponential lambda" begin
    Random.seed!(4242)
    Ne = 300
    xe1 = randn(Ne); xe2 = randn(Ne); conse = ones(Ne)
    Xe  = hcat(conse, xe1, xe2)
    ye  = Xe * [1.0, 0.5, -0.3] .+ 0.3 .* randn(Ne) .- randexp(Ne) .* 0.4
    β_ols = Xe \ ye

    spec = sfmodel_spec(depvar = ye, frontier = Xe, noise = :Normal,
                        ineff = :Exponential, type = :prod)
    init = sfmodel_init(spec = spec, frontier = β_ols,
                        ln_lambda = log(0.3), ln_sigma_v_sq = log(0.09))
    sfmodel_fit(spec = spec, method = sfmodel_method(method = :MLE), init = init,
                optim_options = _g7_opt(), marginal = false, show_table = false, verbose = false)

    pred = SFrontiers.MLE_Backend.sfmodel_predict
    λ     = pred([:lambda])                       # no longer throws (fell through pre-fix)
    logλ² = pred([:log_lambda_2])
    @test pred([:λ]) == λ                          # unicode alias
    @test pred([:log_λ²]) == logλ²
    @test maximum(abs.(λ .- sqrt.(exp.(logλ²)))) < 1e-12   # λ = sqrt(exp(W·δ̂))
    @test all(isfinite, λ) && all(>(0), λ)
    # The old code threw for :lambda — an unrecognized eqname still does (the
    # dispatch throws a bare String, so catch generically).
    @test (try; pred([:not_a_real_equation]); false; catch; true; end)
end

# ---------------------------------------------------------------------
# R-10 — legacy `sfmodel_predict(@eq(log_gamma))` for Kumbhakar-1990.
#
# The τ coefficient block is the pre-link predictor Q·τ̂. For Kumbhakar-1990
# Gₜ = 2/(1+exp(Q·τ̂)), so log Gₜ = log(2) − log1p(exp(Q·τ̂)) ≠ Q·τ̂. The pre-fix
# `:log_gamma` branch returned the raw Q·τ̂, breaking `exp(@eq(log_gamma)) ==
# @eq(gamma)`. Verified directly by driving the `sfmodel_predict` dispatch with a
# minimal Kumb90 / TimeDecay global state (the legacy fit pipeline is unexported).
# ---------------------------------------------------------------------
@testset "R-10 sfmodel_predict log_gamma Kumbhakar-1990" begin
    M = SFrontiers.MLE_Backend.SFmle
    Qmat = hcat(ones(4), [0.5, -0.3, 0.8, 1.2])
    coef = [0.2, 0.7]
    Qτ   = Qmat * coef
    sdf  = DataFrame(gamma_var1 = Qmat[:, 1], gamma_var2 = Qmat[:, 2])
    setg(panel) = begin
        Core.eval(M, :(_dicM = $(Dict{Symbol,Any}(:panel => panel,
                       :gamma => [:gamma_var1, :gamma_var2], :sdf => sdf))))
        Core.eval(M, :(_eqncoe = $(Dict{Symbol,Vector}(:log_gamma => coef))))
    end

    # Kumbhakar-1990: log Gₜ is the proper log-link, and exp(log_gamma)==gamma.
    setg([:Kumbhakar1990])
    g_k, lg_k = M.sfmodel_predict([:gamma]), M.sfmodel_predict([:log_gamma])
    @test maximum(abs.(lg_k .- (log(2.0) .- log1p.(exp.(Qτ))))) < 1e-12
    @test maximum(abs.(exp.(lg_k) .- g_k)) < 1e-12           # broken before the fix
    @test maximum(abs.(lg_k .- Qτ)) > 1e-6                    # pre-fix returned raw Q·τ̂

    # TimeDecay: Gₜ = exp(Q·τ̂), so log Gₜ = Q·τ̂ — must stay unchanged.
    setg([:TimeDecay])
    g_t, lg_t = M.sfmodel_predict([:gamma]), M.sfmodel_predict([:log_gamma])
    @test maximum(abs.(lg_t .- Qτ)) < 1e-12
    @test maximum(abs.(exp.(lg_t) .- g_t)) < 1e-12
end

# ---------------------------------------------------------------------
# R-07 — legacy `sfmodel_boot_marginal` runs end-to-end.
#
# Three defects blocked it: (a) `transfer=false` at spec time forced a `data=` arg,
# but passing a frame whose columns don't match the internal auto-names crashed
# `getvar`; (b) the replicate `get_marg` skipped the fit's common-name override, so
# for a multi-channel (μ + σᵤ²) fit it produced more columns than `result.marginal`
# → `sim_res[:, i] = theM` DimensionMismatch. Fix: mark transfer truthfully (the
# fit-time frame IS in `_dicM[:sdf]`) and mirror the common-name merge.
# ---------------------------------------------------------------------
@testset "R-07 sfmodel_boot_marginal" begin
    Random.seed!(2024)
    Nb = 250
    Xb = hcat(ones(Nb), randn(Nb), randn(Nb))
    zb1 = randn(Nb); zb2 = randn(Nb)
    Zb = hcat(ones(Nb), zb1, zb2)                 # 2 non-constant channels (merge is non-trivial)
    ub = abs.(0.5 .* randn(Nb) .+ 0.2 .* zb1)
    yb = Xb * [1.0, 0.5, -0.3] .+ 0.3 .* randn(Nb) .- ub

    spec = sfmodel_spec(depvar = yb, frontier = Xb, zvar = Zb, noise = :Normal,
                        ineff = :TruncatedNormal, hetero = [:mu, :sigma_sq], type = :prod)
    init = sfmodel_init(spec = spec, frontier = Xb \ yb, mu = zeros(3),
                        ln_sigma_sq = zeros(3), ln_sigma_v_sq = log(0.09))
    res = sfmodel_fit(spec = spec, method = sfmodel_method(method = :MLE), init = init,
                      optim_options = _g7_opt(), marginal = true, show_table = false, verbose = false)

    @test res.transfer == true                        # transfer flag now truthful
    K = ncol(res.marginal)

    boot = SFrontiers.MLE_Backend.sfmodel_boot_marginal
    out = boot(result = res, R = 30, seed = 123)       # no `data=` arg; no DimensionMismatch (merge fixed)
    @test size(out, 1) == K                            # replicate columns match result.marginal
    @test all(x -> x isa Real && isfinite(x), out[:, 1])   # SE column finite
    out2 = boot(result = res, R = 30, seed = 123)
    @test out == out2                                  # reproducible under a fixed seed

    # A rare-dummy determinant can go constant in a bootstrap resample, so
    # nonConsDataFrame purges its marginal column and the replicate is shorter than
    # result.marginal — the loop must resample such a draw, not crash with an
    # uncaught DimensionMismatch.
    Random.seed!(7)
    Nd = 100
    Xd = hcat(ones(Nd), randn(Nd), randn(Nd))
    dd = zeros(Nd); dd[[10, 55]] .= 1.0                # dummy: 1 for exactly 2 obs
    Zd = hcat(ones(Nd), dd, randn(Nd))
    yd = Xd * [1.0, 0.5, -0.3] .+ 0.3 .* randn(Nd) .- abs.(0.5 .* randn(Nd) .+ 0.4 .* dd)
    spec_d = sfmodel_spec(depvar = yd, frontier = Xd, zvar = Zd, noise = :Normal,
                          ineff = :TruncatedNormal, hetero = [:mu, :sigma_sq], type = :prod)
    init_d = sfmodel_init(spec = spec_d, frontier = Xd \ yd, mu = zeros(3),
                          ln_sigma_sq = zeros(3), ln_sigma_v_sq = log(0.09))
    res_d = sfmodel_fit(spec = spec_d, method = sfmodel_method(method = :MLE), init = init_d,
                        optim_options = _g7_opt(), marginal = true, show_table = false, verbose = false)
    Kd = ncol(res_d.marginal)
    out_d = boot(result = res_d, R = 200, seed = 42)   # ~13% of resamples drop the dummy → guard resamples
    @test size(out_d, 1) == Kd                          # completes without DimensionMismatch
end

# ---------------------------------------------------------------------
# FIX-22 — Gamma + copula (MCI) passes the true F_Gamma(u) to the copula.
#
# The copula contract (manual §8) is f = f_v·f_u·c(F_v(v), F_u(u)); Gamma was the
# lone distribution still passing the raw expo-rule pre-image t = 1−exp(−u/θ)
# (≠ F_Gamma for shape k≠1), distorting the noise marginal. Now Gamma uses an
# AD-safe regularized incomplete gamma F_Gamma(u)=P(k,u/θ). Verify the CDF against
# closed forms for integer shapes (dependency-free), the trait, and an end-to-end fit.
# ---------------------------------------------------------------------
@testset "FIX-22 Gamma copula true F_u" begin
    P = SFrontiers.MCI_Backend._reg_inc_gamma_P
    # Integer-shape closed forms: P(1,x)=1−e^{-x}; P(2,x)=1−e^{-x}(1+x);
    # P(3,x)=1−e^{-x}(1+x+x²/2). loggamma(1)=loggamma(2)=0, loggamma(3)=log 2.
    for x in (0.1, 0.5, 1.0, 3.0, 8.0)
        @test isapprox(P(1.0, x, 0.0),      1 - exp(-x);                atol = 1e-12)
        @test isapprox(P(2.0, x, 0.0),      1 - exp(-x)*(1 + x);        atol = 1e-12)
        @test isapprox(P(3.0, x, log(2.0)), 1 - exp(-x)*(1 + x + x^2/2); atol = 1e-12)
    end
    @test P(2.0, 0.0, 0.0) == 0.0
    @test isapprox(P(2.0, 1.0e3, 0.0), 1.0; atol = 1e-8)      # → 1 in the tail
    @test isfinite(P(0.4, 2.0, SFrontiers.MCI_Backend.loggamma(0.4)))  # non-integer shape finite
    # Type stability across both branches (Float32/GPU contract): the CF branch
    # (x ≥ a+1) must not promote to Float64 via the FPMIN literal.
    @test typeof(P(2.0f0, 1.0f0, 0.0f0)) == Float32          # series branch
    @test typeof(P(2.0f0, 5.0f0, 0.0f0)) == Float32          # continued-fraction branch
    @test typeof(P(2.0, 5.0, 0.0)) == Float64                # Float64 unchanged

    # Gamma now advertises the true F_u to the copula machinery.
    @test SFrontiers.MCI_Backend._copula_true_Fu(SFrontiers.MCI_Backend.Gamma_MCI()) == true

    # End-to-end: a Gamma + Gaussian-copula MCI fit runs (AD stable through F_Gamma).
    Random.seed!(55)
    Ng = 300
    Xg = hcat(ones(Ng), randn(Ng), randn(Ng))
    ug = 0.4 .* (randexp(Ng) .+ randexp(Ng))              # Gamma(k=2, θ=0.4) draws (no Distributions dep)
    yg = Xg * [1.0, 0.5, -0.3] .+ 0.3 .* randn(Ng) .- ug
    spec = sfmodel_spec(depvar = yg, frontier = Xg, noise = :Normal,
                        ineff = :Gamma, copula = :Gaussian, type = :prod)
    init = sfmodel_init(spec = spec, frontier = Xg \ yg, ln_k = log(2.0),
                        ln_theta = log(0.4), ln_sigma_v_sq = log(0.09), theta_rho = [0.0])
    opt  = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 100, g_abstol = 1e-3),
                       main_solver = Newton(), main_opt = (iterations = 200, g_abstol = 1e-6))
    res = sfmodel_fit(spec = spec, method = sfmodel_method(method = :MCI, n_draws = 400),
                      init = init, optim_options = opt, marginal = false,
                      show_table = false, verbose = false)
    @test res.converged == true
    @test isfinite(res.loglikelihood)
    @test res.redflag == 0
end
