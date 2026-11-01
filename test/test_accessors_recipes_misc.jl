# Regression tests: MoMTest cost recentring, custom draws, plot recipes, legacy accessors, the MLE bootstrap.
# Docs, plot recipes and legacy accessors:
#   MoMTest cost E[u] recentring, custom-draws multiRand=false,
#   QQ panel title, panel zvar-missing error, store_trace
#   / plot(:convergence), sfmodel_boot_marginal, sfmodel_predict.

using SFrontiers
using Test
using Random
using Statistics
using DataFrames
using Optim
using RecipesBase

# Legacy MLE-backend entry points reached through their (unexported) module path.
const _MLE = SFrontiers.MLE_Backend

# A small Normal + HalfNormal dataset reused by several testsets.
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
# sfmodel_MoMTest cost-frontier JLMS/BC/llike recentring.
#
# After the `Resid = _porc*resid` flip in sfmodel_MoMTest, the E[u] recentring must
# SUBTRACT E[u] unconditionally. The pre-fix `myeps = Resid - _porc*E[u]`
# ADDED E[u] for :cost (_porc=-1), shifting every conditioning residual by
# +2E[u]. A :cost run must therefore reproduce a sign-mirrored :prod run
# (cost on y ≡ production on -y) obs-by-obs.
# ---------------------------------------------------------------------
@testset "MoMTest cost recentring" begin
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

    # Half-normal cost jlms mean tracks the true mean inefficiency (here off by 0.08, the sampling
    # error of one sample of 400; the bias before the fix was about 2E[u] = 1.3).
    rc = _MLE.sfmodel_MoMTest((:type,[:cost]), (:dist,[:h]), (:depvar,[:y]),
                              (:frontier,[:cons,:x1,:x2]); data=df_cost, verbose=false)
    @test isapprox(mean(vec(rc.jlms)), mean(um); atol = 0.3)
end

# ---------------------------------------------------------------------
# The custom-draws `meth4` pattern requires `multiRand=false`.
#
# Manual §6.2's `meth4` example (a 1×S row matrix of user draws) omitted the
# `multiRand=false` that §9.8 requires. The code is correct: with
# `multiRand=false` a 1×S draws matrix fits and `result.n_draws` reports the
# supplied length. Pin the documented (corrected) behavior for MSLE and MCI.
# ---------------------------------------------------------------------
@testset "custom draws multiRand=false" begin
    X, y, β_ols = _g7_hn_data()
    spec = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                        ineff = :HalfNormal, type = :prod)
    init = sfmodel_init(spec = spec, frontier = β_ols,
                        ln_sigma_u_sq = log(0.25), ln_sigma_v_sq = log(0.09))
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
# The QQ panel title reflects the self-standardized shape check.
#
# The QQ sub-panel standardizes v̂ by its own sample mean/std and plots against
# N(0,1) — so the old "N(0, σ̂_v²)" title over-claimed a scale-aware comparison.
# The title must now read as a standardized-normal shape check.
# ---------------------------------------------------------------------
@testset "QQ panel title" begin
    X, y, β_ols = _g7_hn_data()
    spec = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                        ineff = :HalfNormal, type = :prod)
    init = sfmodel_init(spec = spec, frontier = β_ols,
                        ln_sigma_u_sq = log(0.25), ln_sigma_v_sq = log(0.09))
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
# A panel spec missing `zvar` raises an informative identification
# error, not the generic "Unsupported matrix input type: Nothing".
# ---------------------------------------------------------------------
@testset "panel zvar-missing error" begin
    Nf, T = 8, 4
    idv = repeat(1:Nf, inner = T)
    NT  = Nf * T
    Xp  = randn(NT, 2)
    yp  = randn(NT)

    _msg(f) = try; f(); ""; catch e; sprint(showerror, e); end

    # Direct panel keyword ctor with zvar=nothing.
    m1 = _msg(() -> SFrontiers.Panel_Backend.sfmodel_panel_spec(
                        depvar = yp, frontier = Xp, zvar = nothing, id = idv, type = :prod))
    @test occursin("unidentified", m1)
    @test occursin("within-firm", m1)
    @test !occursin("Unsupported matrix input type", m1)

    # Unified keyword path (datatype=:panel_TFE) forwards zvar=nothing → same error.
    m2 = _msg(() -> sfmodel_spec(depvar = yp, frontier = Xp, id = idv,
                                 noise = :Normal, ineff = :HalfNormal,
                                 datatype = :panel_TFE, type = :prod))
    @test occursin("within-firm variation", m2)
    @test !occursin("Unsupported matrix input type", m2)

    # A valid within-varying zvar still constructs (guard is not over-broad).
    zv = randn(NT)                                  # varies within firms
    spec = sfmodel_spec(depvar = yp, frontier = Xp, zvar = zv, id = idv,
                        noise = :Normal, ineff = :HalfNormal,
                        datatype = :panel_TFE, type = :prod)
    @test spec isa SFrontiers.UnifiedSpec
end

# ---------------------------------------------------------------------
# `store_trace` captures the optimizer trace onto the result and
# powers `plot(:convergence)`; without it, the recipe errors.
# ---------------------------------------------------------------------
@testset "store_trace / plot(:convergence)" begin
    X, y, β_ols = _g7_hn_data()
    spec = sfmodel_spec(depvar = y, frontier = X, noise = :Normal,
                        ineff = :HalfNormal, type = :prod)
    init = sfmodel_init(spec = spec, frontier = β_ols,
                        ln_sigma_u_sq = log(0.25), ln_sigma_v_sq = log(0.09))

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

    # The MSLE backend's own option builder takes `store_trace` as well: its options give the fit and the
    # trace of the unified builder's.
    _mopt(store) = SFrontiers.MSLE_Backend.sfmodel_MSLE_opt(warmstart_solver = NelderMead(),
                       warmstart_opt = (iterations = 40,), main_solver = Newton(),
                       main_opt = (iterations = 60, g_abstol = 1e-5), store_trace = store)
    @test !SFrontiers.MSLE_Backend.sfmodel_MSLE_opt(main_solver = Newton(),
                                                    main_opt = (iterations = 1,)).main_opt.store_trace
    @test _mopt(true).main_opt.store_trace && !_mopt(false).main_opt.store_trace
    msle_on = sfmodel_fit(spec = spec, method = _mk(:MSLE), init = init, optim_options = _mopt(true),
                          marginal = false, show_table = false, verbose = false)
    @test msle_on.trace !== nothing && keys(msle_on.trace) == (:iteration, :value, :g_norm)
    msle_ref = sfmodel_fit(spec = spec, method = _mk(:MSLE), init = init, optim_options = _opt(true),
                           marginal = false, show_table = false, verbose = false)
    @test isequal(msle_on.trace, msle_ref.trace) && isequal(msle_on.coeff, msle_ref.coeff)
    msle_off = sfmodel_fit(spec = spec, method = _mk(:MSLE), init = init, optim_options = _mopt(false),
                           marginal = false, show_table = false, verbose = false)
    @test msle_off.trace === nothing

    # The panel option builders take `store_trace` as well (the exported one passes it to the panel
    # backend's): their options give the fit and the trace of the unified builder's. 50 firms, 5 periods.
    Random.seed!(78)
    nfq, tq = 50, 5; ntq = nfq * tq
    idq = repeat(1:nfq, inner = tq)
    Xq  = randn(ntq, 2)                                    # no intercept (panel_TFE)
    zq  = randn(ntq)                                       # within-varying zvar
    yq  = Xq * [0.5, -0.3] .+ 0.3 .* randn(ntq) .- abs.(0.4 .* randn(ntq))
    qspec = sfmodel_spec(depvar = yq, frontier = Xq, zvar = zq, id = idq,
                         noise = :Normal, ineff = :HalfNormal, datatype = :panel_TFE, type = :prod)
    _popt(store) = sfmodel_panel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 60,),
                                     main_solver = NelderMead(), main_opt = (iterations = 200, g_abstol = 1e-4),
                                     store_trace = store)
    @test !sfmodel_panel_opt(main_solver = Newton(), main_opt = (iterations = 1,)).main_opt.store_trace
    @test _popt(true).main_opt.store_trace && !_popt(false).main_opt.store_trace
    @test SFrontiers.Panel_Backend.sfmodel_panel_opt(main_solver = Newton(), main_opt = (iterations = 1,),
                                                     store_trace = true).main_opt.store_trace
    @test sfmodel_panel_opt(main_solver = Newton(),
                            main_opt = (iterations = 1, store_trace = true)).main_opt.store_trace
    qmeth = sfmodel_method(method = :MSLE, n_draws = 100)
    q_on = sfmodel_fit(spec = qspec, method = qmeth, optim_options = _popt(true),
                       marginal = false, show_table = false, verbose = false)
    @test q_on.trace !== nothing && keys(q_on.trace) == (:iteration, :value, :g_norm)
    @test !isempty(RecipesBase.apply_recipe(Dict{Symbol,Any}(), q_on, Val(:convergence)))
    q_ref = sfmodel_fit(spec = qspec, method = qmeth,
                        optim_options = sfmodel_opt(warmstart_solver = NelderMead(),
                                                    warmstart_opt = (iterations = 60,), main_solver = NelderMead(),
                                                    main_opt = (iterations = 200, g_abstol = 1e-4), store_trace = true),
                        marginal = false, show_table = false, verbose = false)
    @test isequal(q_on.trace, q_ref.trace) && isequal(q_on.coeff, q_ref.coeff)
    q_off = sfmodel_fit(spec = qspec, method = qmeth, optim_options = _popt(false),
                        marginal = false, show_table = false, verbose = false)
    @test q_off.trace === nothing
end

# ---------------------------------------------------------------------
# Legacy `sfmodel_predict(@eq(sigma_u))` for Exponential MLE.
#
# The `@eq()` dispatch had no branch for the Exponential inefficiency scale, so it fell
# through to `throw("@eq() is not specified correctly")`. Added: the scale is
# σᵤ = exp(0.5·(W·δ̂)) = sqrt(exp(W·δ̂)), and `@eq(log_σᵤ²)` is the raw predictor. The
# branch was spelled `@eq(λ)`/`@eq(log_λ²)` in an earlier form; the Exponential's
# parameter is named σᵤ² so that `lambda` does not mean a scale here and a variance in
# the simulation backends. test_expo_naming.jl checks σᵤ = sqrt(exp(@eq(log_σᵤ²))) and the
# refusal of the old spellings; this set checks the Unicode spellings and an unknown name.
# ---------------------------------------------------------------------
@testset "sfmodel_predict Exponential sigma_u" begin
    Random.seed!(4242)
    Ne = 300
    xe1 = randn(Ne); xe2 = randn(Ne); conse = ones(Ne)
    Xe  = hcat(conse, xe1, xe2)
    ye  = Xe * [1.0, 0.5, -0.3] .+ 0.3 .* randn(Ne) .- randexp(Ne) .* 0.4
    β_ols = Xe \ ye

    spec = sfmodel_spec(depvar = ye, frontier = Xe, noise = :Normal,
                        ineff = :Exponential, type = :prod)
    init = sfmodel_init(spec = spec, frontier = β_ols,
                        ln_sigma_u_sq = log(0.3), ln_sigma_v_sq = log(0.09))
    sfmodel_fit(spec = spec, method = sfmodel_method(method = :MLE), init = init,
                optim_options = _g7_opt(), marginal = false, show_table = false, verbose = false)

    pred = SFrontiers.MLE_Backend.sfmodel_predict
    σᵤ     = pred([:sigma_u])                     # no longer throws (fell through pre-fix)
    logσᵤ² = pred([:log_sigma_u_2])
    @test pred([:σᵤ]) == σᵤ                        # unicode alias
    @test pred([:log_σᵤ²]) == logσᵤ²
    @test all(isfinite, σᵤ) && all(>(0), σᵤ)
    # An unrecognized eqname throws (the dispatch throws a bare String, so catch generically).
    @test (try; pred([:not_a_real_equation]); false; catch; true; end)
end

# ---------------------------------------------------------------------
# Legacy `sfmodel_predict(@eq(log_gamma))` for Kumbhakar-1990.
#
# The τ coefficient block is the pre-link predictor Q·τ̂. For Kumbhakar-1990
# Gₜ = 2/(1+exp(Q·τ̂)), so log Gₜ = log(2) − log1p(exp(Q·τ̂)) ≠ Q·τ̂. The pre-fix
# `:log_gamma` branch returned the raw Q·τ̂, breaking `exp(@eq(log_gamma)) ==
# @eq(gamma)`. Verified directly by driving the `sfmodel_predict` dispatch with a
# minimal Kumb90 / TimeDecay global state (the legacy fit pipeline is unexported).
# ---------------------------------------------------------------------
@testset "sfmodel_predict log_gamma Kumbhakar-1990" begin
    M = SFrontiers.MLE_Backend.SFmle
    Qmat = hcat(ones(4), [0.5, -0.3, 0.8, 1.2])
    coef = [0.2, 0.7]
    Qτ   = Qmat * coef
    sdf  = DataFrame(gamma_var1 = Qmat[:, 1], gamma_var2 = Qmat[:, 2])
    setg(panel) = begin
        Core.eval(M, :(_dicM = $(Dict{Symbol,Any}(:panel => panel,
                       :gamma => [:gamma_var1, :gamma_var2], :sdf => sdf))))
        Core.eval(M, :(_eqncoe = $(Dict{Symbol,Vector}(:log_gamma => coef))))
        # `sfmodel_predict` refuses a model state that has been replaced since the last
        # fit: it takes its data from `_dicM` at call time and its
        # coefficients from the fit, so a spec built in between used to pair one model's
        # data with another's estimates.  This fixture fabricates both halves directly,
        # so it fabricates the stamp that ties them together as well.
        Core.eval(M, :(_fitted_dicM = _dicM))
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
# Legacy `sfmodel_boot_marginal` runs end-to-end.
#
# Three defects blocked it: (a) `transfer=false` at spec time forced a `data=` arg,
# but passing a frame whose columns don't match the internal auto-names crashed
# `getvar`; (b) the replicate `get_marg` skipped the fit's common-name override, so
# for a multi-channel (μ + σᵤ²) fit it produced more columns than `result.marginal`
# → `sim_res[:, i] = theM` DimensionMismatch. Fix: mark transfer truthfully (the
# fit-time frame IS in `_dicM[:sdf]`) and mirror the common-name merge.
# ---------------------------------------------------------------------
@testset "sfmodel_boot_marginal" begin
    Random.seed!(2024)
    Nb = 250
    Xb = hcat(ones(Nb), randn(Nb), randn(Nb))
    zb1 = randn(Nb); zb2 = randn(Nb)
    Zb = hcat(ones(Nb), zb1, zb2)                 # 2 non-constant channels (merge is non-trivial)
    ub = abs.(0.5 .* randn(Nb) .+ 0.2 .* zb1)
    yb = Xb * [1.0, 0.5, -0.3] .+ 0.3 .* randn(Nb) .- ub

    spec = sfmodel_spec(depvar = yb, frontier = Xb, zvar = Zb, noise = :Normal,
                        ineff = :TruncatedNormal, hetero = [:mu, :sigma_u_sq], type = :prod)
    init = sfmodel_init(spec = spec, frontier = Xb \ yb, mu = zeros(3),
                        ln_sigma_u_sq = zeros(3), ln_sigma_v_sq = log(0.09))
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
end

# ---------------------------------------------------------------------
# Gamma + copula (MCI) passes the true F_Gamma(u) to the copula.
#
# The copula contract (manual §8) is f = f_v·f_u·c(F_v(v), F_u(u)); Gamma was the
# lone distribution still passing the raw expo-rule pre-image t = 1−exp(−u/θ)
# (≠ F_Gamma for shape k≠1), distorting the noise marginal. Now Gamma uses an
# AD-safe regularized incomplete gamma F_Gamma(u)=P(k,u/θ). Verify the CDF against
# closed forms for integer shapes (dependency-free), the trait, and an end-to-end fit.
# ---------------------------------------------------------------------
@testset "Gamma copula true F_u" begin
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
    # It converges here, after about 120 Newton iterations; whether it does within the cap can depend
    # on rounding and package versions, so the checks are that it runs and ends near a stationary point.
    @test isfinite(res.loglikelihood)
    @test all(isfinite, res.coeff)
    @test res.gradient_norm < 1e-3
end
