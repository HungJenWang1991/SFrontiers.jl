# Regression tests for the 03_reaudit_findings.md Group 2 (G2) fixes — the unified
# glue layer (spec / dispatch / reorder): R-02, R-13, R-15, R-16, R-17.
# Each testset targets a specific finding and would have failed before its fix.
# Kept self-contained (own data) so it can run independently of runtests.jl.

using SFrontiers
using Test
using Random
using Statistics
using Optim
using StatsAPI
using DataFrames

# Shared NelderMead options (degenerate/edge fits don't need a good optimum).
_g2opt() = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 20,),
                       main_solver = NelderMead(), main_opt = (iterations = 60,))

# ---------------------------------------------------------------------------
# R-02 — REGRESSION (FIX-05): user-supplied `varnames` were double-permuted for a
# TruncatedNormal + scaling MLE fit, so `coeftable` mislabeled the μ/δ rows. The
# reorder must permute only the MLE-order auto names, not the already-canonical
# user names.
# ---------------------------------------------------------------------------
@testset "R-02 user varnames not double-permuted (TN+scaling MLE)" begin
    Random.seed!(202)
    n = 200
    X = hcat(ones(n), randn(n), randn(n)); Z = hcat(randn(n), randn(n))
    y = X * [1.0, 0.5, -0.3] .+ 0.3 .* randn(n) .- abs.(0.4 .* randn(n))
    vn = ["x1", "x2", "x3", "z1", "z2", "mu", "lnsu2", "lnsv2"]   # user MCI-order (K=3, L=2)
    spec = sfmodel_spec(depvar = y, frontier = X, zvar = Z, noise = :Normal,
                        ineff = :TruncatedNormal, hetero = :scaling, type = :prod, varnames = vn)
    init = sfmodel_init(spec = spec, frontier = [1.0, 0.5, -0.3], scaling = [0.0, 0.0],
                        mu = 0.0, ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
    res = sfmodel_fit(spec = spec, method = sfmodel_method(method = :MLE, n_draws = 30),
                      init = init, optim_options = _g2opt(),
                      marginal = false, show_table = false, verbose = false)
    # Each published label must pair with its own estimate, in MCI order. Before the
    # fix these were scrambled to ["x1","x2","x3","z2","mu","z1","lnsu2","lnsv2"].
    @test res.varnames == vn
    @test StatsAPI.coeftable(res).rownms == vn

    # Auto-generated names must still be permuted correctly (not regressed): the two
    # scaling-δ labels come *before* the μ label in MCI order.
    spec_a = sfmodel_spec(depvar = y, frontier = X, zvar = Z, noise = :Normal,
                          ineff = :TruncatedNormal, hetero = :scaling, type = :prod)
    init_a = sfmodel_init(spec = spec_a, frontier = [1.0, 0.5, -0.3], scaling = [0.0, 0.0],
                          mu = 0.0, ln_sigma_sq = log(0.25), ln_sigma_v_sq = log(0.09))
    res_a = sfmodel_fit(spec = spec_a, method = sfmodel_method(method = :MLE, n_draws = 30),
                        init = init_a, optim_options = _g2opt(),
                        marginal = false, show_table = false, verbose = false)
    @test res_a.varnames[4] == "hscale_var1" && res_a.varnames[5] == "hscale_var2"
end

# ---------------------------------------------------------------------------
# R-13 — a bare-Symbol `hetero` (e.g. hetero=:k) silently collapsed to
# homoscedastic in the MCI backend, reachable via ineff=:Gamma (whose co-built
# MSLE spec — which already guards this — is skipped). It now errors, mirroring MSLE.
# ---------------------------------------------------------------------------
@testset "R-13 bare-Symbol hetero rejected (not silently homoscedastic)" begin
    Random.seed!(213)
    n = 120
    X = hcat(ones(n), randn(n)); Z = hcat(randn(n), randn(n))
    y = X * [1.0, 0.5] .+ 0.3 .* randn(n) .- abs.(0.4 .* randn(n))
    # Failing input: ineff=:Gamma + bare hetero=:k. Previously returned a spec with
    # mci_spec.hetero == Symbol[] (silently homoscedastic) and no error; now it errors.
    @test_throws Exception sfmodel_spec(depvar = y, frontier = X, zvar = Z, noise = :Normal,
                                        ineff = :Gamma, hetero = :k, type = :prod)
    @test_throws Exception sfmodel_spec(depvar = y, frontier = X, zvar = Z, noise = :Normal,
                                        ineff = :Gamma, hetero = :theta, type = :prod)
    # Valid forms still build: the vector form (with a constant `_cons` first zvar column so
    # the homoscedastic θ keys off the intercept — re-audit R-05/G6) and the :scaling model
    # (whose scaling Z must NOT contain a constant column, so it keeps the plain `Z`).
    Zc = hcat(ones(n), randn(n))
    @test sfmodel_spec(depvar = y, frontier = X, zvar = Zc, noise = :Normal,
                       ineff = :Gamma, hetero = [:k], type = :prod) isa SFrontiers.UnifiedSpec
    @test sfmodel_spec(depvar = y, frontier = X, zvar = Z, noise = :Normal,
                       ineff = :TruncatedNormal, hetero = :scaling, type = :prod) isa SFrontiers.UnifiedSpec
end

# ---------------------------------------------------------------------------
# R-15 — the DSL spec builder `_build_mle_cross_dsl` lacked the non-TN-scaling
# short-circuit its keyword sibling has, so a DSL HalfNormal/Exponential + scaling
# spec crashed at spec time ("scaling requires :TruncatedNormal"). It now builds
# (mle_spec skipped), fittable via :MCI/:MSLE.
# ---------------------------------------------------------------------------
@testset "R-15 DSL HN/Exp + scaling spec builds (mle_spec skipped)" begin
    Random.seed!(215)
    n = 150
    df = DataFrame(x1 = randn(n), x2 = randn(n), z1 = randn(n))
    df.y = 1.0 .+ 0.5 .* df.x1 .- 0.3 .* df.x2 .+ 0.3 .* randn(n) .- abs.(0.4 .* randn(n))

    # HN + scaling and Exp + scaling used to throw at spec time; now they build with
    # no MLE spec (MLE scaling is TruncatedNormal-only).
    sp_hn = sfmodel_spec(@useData(df), @depvar(y), @frontier(x1, x2), @zvar(z1),
                         noise = :Normal, ineff = :HalfNormal, hetero = :scaling)
    @test sp_hn isa SFrontiers.UnifiedSpec
    @test sp_hn.mle_spec === nothing
    @test sp_hn.mci_spec !== nothing

    sp_exp = sfmodel_spec(@useData(df), @depvar(y), @frontier(x1, x2), @zvar(z1),
                          noise = :Normal, ineff = :Exponential, hetero = :scaling)
    @test sp_exp isa SFrontiers.UnifiedSpec
    @test sp_exp.mle_spec === nothing

    # TN + scaling must still build its MLE spec (the short-circuit only skips the
    # unsupported distributions).
    sp_tn = sfmodel_spec(@useData(df), @depvar(y), @frontier(x1, x2), @zvar(z1),
                         noise = :Normal, ineff = :TruncatedNormal, hetero = :scaling)
    @test sp_tn.mle_spec !== nothing
end

# ---------------------------------------------------------------------------
# R-16 — FIX-14 incomplete: `_reorder_init_panel_to_mle` didn't short-circuit a
# Dict init (only the cross-sectional sibling did), so `collect(::Dict)` produced a
# Vector{Pair} in hash order and the panel_TFE MLE fit crashed with a misleading
# "number of initial values does not match". Dict inits now pass through untouched.
# ---------------------------------------------------------------------------
@testset "R-16 panel_TFE MLE accepts a component Dict init" begin
    Random.seed!(216)
    Nf = 30; T = 5; n = Nf * T
    id = repeat(1:Nf, inner = T)
    X = hcat(randn(n), randn(n)); Z = reshape(randn(n), n, 1)   # no intercept (absorbed by FE)
    y = X * [0.5, -0.3] .+ 0.3 .* randn(n) .- abs.(0.4 .* randn(n))
    spec = sfmodel_spec(datatype = :panel_TFE, depvar = y, frontier = X, zvar = Z, id = id,
                        noise = :Normal, ineff = :HalfNormal, type = :prod)

    # Unit: the panel reorder must return a Dict init unchanged (before the fix it
    # collected the Dict into a Vector{Pair} in hash order).
    d = Dict(:frontier => [0.5, -0.3], :eqw => [0.1], :eqv => [0.1])
    @test SFrontiers._reorder_init_panel_to_mle(spec, d) === d

    # End-to-end: the fit runs with a component Dict init instead of crashing.
    opt = sfmodel_opt(warmstart_solver = NelderMead(), warmstart_opt = (iterations = 20,),
                      main_solver = NelderMead(), main_opt = (iterations = 40,))
    res = sfmodel_fit(spec = spec, method = sfmodel_method(method = :MLE),
                      init = Dict(:frontier => [0.5, -0.3], :eqw => [0.1], :eqv => [0.1]),
                      optim_options = opt, marginal = false, show_table = false, verbose = false)
    @test res isa SFResult
    @test isfinite(res.loglikelihood)
end

# ---------------------------------------------------------------------------
# R-17 — for panel_TFE with a non-MLE distribution, `_get_supported_methods`
# checked only mci_spec/msle_spec/mle_spec (all nothing — the Panel backend serves
# MCI/MSLE via panel_spec), so it reported an empty supported-methods set even
# though :MCI and :MSLE both work. The panel_spec branch now pushes :MCI, :MSLE.
# ---------------------------------------------------------------------------
@testset "R-17 panel_TFE reports MCI/MSLE as supported methods" begin
    Random.seed!(217)
    Nf = 25; T = 5; n = Nf * T
    id = repeat(1:Nf, inner = T)
    X = hcat(randn(n), randn(n)); Z = reshape(randn(n), n, 1)
    y = X * [0.5, -0.3] .+ 0.3 .* randn(n) .- abs.(0.4 .* randn(n))

    # panel_TFE + Weibull has no MLE route; MCI/MSLE serve it via panel_spec.
    spec = sfmodel_spec(datatype = :panel_TFE, depvar = y, frontier = X, zvar = Z, id = id,
                        noise = :Normal, ineff = :Weibull, type = :prod)
    @test SFrontiers._get_supported_methods(spec) == [:MCI, :MSLE]

    # Requesting :MLE errors, and the message lists the real supported methods
    # (before the fix it listed an empty set).
    err = try
        sfmodel_fit(spec = spec, method = sfmodel_method(method = :MLE),
                    show_table = false, verbose = false)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("MCI", err) && occursin("MSLE", err)

    # Cross-sectional HalfNormal is unaffected (mci/msle/mle, no panel double-count).
    spc = sfmodel_spec(depvar = y, frontier = hcat(ones(n), X), noise = :Normal,
                       ineff = :HalfNormal, type = :prod)
    @test SFrontiers._get_supported_methods(spc) == [:MCI, :MSLE, :MLE]
end

