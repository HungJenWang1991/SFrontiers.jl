# ---------------------------------------------------------------------------
# The library of the CPU-GPU equivalence sweep (README: test/GPU_VALIDATION.md).
#
# Compares the CPU and GPU code paths of SFrontiers at two levels:
#   fixed-θ  — evaluate NLL / jlms / bc / marginal at IDENTICAL parameter
#              vectors on the CPU-assembled vs GPU-assembled internal specs.
#              Sharp instrument: no optimizer in the loop. rtol 1e-10.
#   full-fit — independent GPU=true vs GPU=false fits from identical inputs
#              (the same draws). Coeff rtol 1e-6; when either fit did not
#              converge, the differences are recorded but not judged.
#              When the coefficients differ, the ridge rule decides: each
#              device's log-likelihoods at the two optima must agree within
#              ATOL_RIDGE (gpu_sweep_verdict.jl).
#
# Scalar indexing: everything runs under CUDA.allowscalar(false). A
# scalar-indexing error in the optimization loop is a failure; a post-fit
# stage that runs once per fit is retried under CUDA.@allowscalar and, if it
# then passes, recorded as :ok_with_allowscalar.
#
# This file assumes `using CUDA` happened BEFORE `using SFrontiers` (GPU
# method overloads register conditionally) and is include()d from
# gpu_sweep_runner.jl with CUDA, SFrontiers, Optim, DataFrames, Random,
# Statistics in scope.
# ---------------------------------------------------------------------------

include(joinpath(@__DIR__, "gpu_sweep_verdict.jl"))   # ATOL_RIDGE, ridge_pass, case_verdict

const RTOL_FIXED   = 1e-10   # fixed-θ comparisons (Float64 device math)
const RTOL_FIT     = 1e-6    # post-optimization coefficient comparisons
const RTOL_F32     = 5e-3    # Float32 smoke case, coefficient level

# ------------------------------------------------------------------ utilities

_errstr(e) = first(sprint(showerror, e), 400)

_is_scalar_err(e) = occursin("calar indexing", sprint(showerror, e))

"Max elementwise relative difference; NaN==NaN counts as agreement."
function maxreldiff(a, b)
    (a === nothing || b === nothing) && return nothing
    av = collect(Float64, vec(Array(a)))
    bv = collect(Float64, vec(Array(b)))
    length(av) == length(bv) || return Inf
    m = 0.0
    for i in eachindex(av)
        x, y = av[i], bv[i]
        (isnan(x) && isnan(y)) && continue
        d = abs(x - y) / max(abs(x), abs(y), 1.0)
        isnan(d) && (d = Inf)
        m = max(m, d)
    end
    return m
end

"""
Run stage `f`; classify errors. `retry_allowscalar=true` marks a post-fit
stage that runs once per fit, where a scalar-index error is retried under
@allowscalar. Returns Dict(:status, :value, [:error, :scalar_index, :error_retry]).
"""
function stage(f; retry_allowscalar::Bool=false)
    try
        return Dict{Symbol,Any}(:status => :ok, :value => f())
    catch e
        si = _is_scalar_err(e)
        d = Dict{Symbol,Any}(:status => :error, :value => nothing,
                             :error => _errstr(e), :scalar_index => si)
        if si && retry_allowscalar
            try
                d[:value]  = CUDA.@allowscalar f()
                d[:status] = :ok_with_allowscalar
            catch e2
                d[:error_retry] = _errstr(e2)
            end
        end
        return d
    end
end

# Minimal JSON writer (no JSON dep in the package env). Handles the types the
# harness emits: Dict{Symbol/String}, NamedTuple, Vector, Real, Bool, Symbol,
# String, Nothing. Non-finite reals are emitted as strings ("Inf", "NaN").
function _json(io::IO, x)
    if x === nothing;               print(io, "null")
    elseif x isa Bool;              print(io, x ? "true" : "false")
    elseif x isa Real
        xf = float(x)
        isfinite(xf) ? print(io, xf) : _json(io, string(xf))
    elseif x isa Symbol;            _json(io, string(x))
    elseif x isa AbstractString
        print(io, '"')
        for c in x
            c == '"'  ? print(io, "\\\"") :
            c == '\\' ? print(io, "\\\\") :
            c == '\n' ? print(io, "\\n")  :
            c == '\r' ? print(io, "\\r")  :
            c == '\t' ? print(io, "\\t")  :
            (c < ' '  ? print(io, "\\u", string(UInt16(c), base=16, pad=4)) : print(io, c))
        end
        print(io, '"')
    elseif x isa AbstractDict
        print(io, '{'); first = true
        for (k, v) in x
            first || print(io, ','); first = false
            _json(io, string(k)); print(io, ':'); _json(io, v)
        end
        print(io, '}')
    elseif x isa NamedTuple;        _json(io, Dict(pairs(x)))
    elseif x isa Union{AbstractVector,Tuple}
        print(io, '['); first = true
        for v in x
            first || print(io, ','); first = false
            _json(io, v)
        end
        print(io, ']')
    else
        _json(io, string(x))
    end
end
json_line(x) = sprint(_json, x)

# ------------------------------------------------------------ data generation

function make_xs_data(; N::Int=400, seed::Int=2026, T::Type=Float64, tiny_u::Bool=false)
    rng = Random.MersenneTwister(seed)
    x1 = randn(rng, N); x2 = randn(rng, N)
    X  = hcat(ones(N), x1, x2)
    z1 = abs.(randn(rng, N)) .+ 0.5
    z2 = abs.(randn(rng, N)) .+ 0.5
    Z  = hcat(ones(N), z1)                    # Z with the constant first
    Zs = hcat(z1, z2)                         # scaling zvar: NO constant column allowed
    v  = 0.3 .* randn(rng, N)
    u  = (tiny_u ? 0.005 : 0.5) .* abs.(randn(rng, N))
    y  = X * [1.0, 0.5, -0.3] .+ v .- u
    (y=T.(y), X=T.(X), Z=T.(Z), Zs=T.(Zs), N=N)
end

function make_panel_data(; F::Int=40, Tp::Int=5, seed::Int=2026)
    rng  = Random.MersenneTwister(seed)
    n    = F * Tp
    firm = repeat(1:F, inner=Tp)
    x1 = randn(rng, n); x2 = randn(rng, n)
    z1 = randn(rng, n)                        # varies within firms, as panel_TFE requires
    α  = repeat(randn(rng, F), inner=Tp)
    v  = 0.25 .* randn(rng, n)
    u  = 0.4 .* abs.(randn(rng, n))
    y  = 0.5 .* x1 .- 0.3 .* x2 .+ α .+ v .- u
    (y=y, X=hcat(x1, x2), z=z1, id=firm)      # panel_TFE: NO intercept column
end

# ------------------------------------------------------------ case definition

Base.@kwdef struct GPUCase
    name::String
    tier::Int
    backend::Symbol                 # :MCI | :MSLE
    dist::Symbol       = :HalfNormal
    type::Symbol       = :prod
    hetero             = Symbol[]   # Symbol[] | Vector{Symbol} | :scaling
    copula::Symbol     = :None
    chunks::Int        = 10
    multiRand::Bool    = true
    n_draws::Int       = 200
    panel::Bool        = false
    float32::Bool      = false
    tiny_u::Bool       = false      # degenerate / redflag case
    user_cpu_draws::Bool = false    # host user draws (device-mismatch path when GPU=true)
    user_gpu_draws::Bool = false    # CuArray user draws (device-mismatch path when GPU=false)
    nm_main::Bool      = false      # NelderMead main solver (hard dists / degenerate)
    consumers::Bool    = false      # post-fit consumers of a GPU result
    N::Int             = 400        # observations of the cross-sectional data (make_xs_data)
    fit_full::Bool     = true       # false: no fits; fixed-θ checks at the data-generating θ only
end

function build_spec(case::GPUCase, dat)
    kw = Dict{Symbol,Any}(:noise => :Normal, :ineff => case.dist, :type => case.type)
    case.copula != :None && (kw[:copula] = case.copula)
    if case.panel
        return sfmodel_spec(; depvar=dat.y, frontier=dat.X, zvar=dat.z, id=dat.id,
                              datatype=:panel_TFE, kw...)
    elseif case.hetero === :scaling
        return sfmodel_spec(; depvar=dat.y, frontier=dat.X, zvar=dat.Zs, hetero=:scaling, kw...)
    elseif !isempty(case.hetero)
        return sfmodel_spec(; depvar=dat.y, frontier=dat.X, zvar=dat.Z,
                              hetero=collect(case.hetero), kw...)
    else
        return sfmodel_spec(; depvar=dat.y, frontier=dat.X, kw...)
    end
end

function build_method(case::GPUCase, dat; gpu::Bool)
    kw = Dict{Symbol,Any}(:method => case.backend, :n_draws => case.n_draws,
                          :multiRand => case.multiRand, :chunks => case.chunks)
    gpu && (kw[:GPU] = true)
    if case.user_cpu_draws || case.user_gpu_draws
        rng  = Random.MersenneTwister(7)
        host = case.multiRand ? rand(rng, length(dat.y), case.n_draws) :
                                rand(rng, case.n_draws)
        # host on purpose for user_cpu_draws; device on purpose for user_gpu_draws —
        # both take the paths that move the draws to the device of the data
        kw[:draws] = case.user_gpu_draws ? CUDA.CuArray(host) : host
    end
    return sfmodel_method(; kw...)
end

function build_opt(case::GPUCase)
    main = case.nm_main ? NelderMead() : Newton()
    main_iters = case.nm_main ? 600 : 200
    return sfmodel_opt(warmstart_solver = NelderMead(),
                       warmstart_opt    = (iterations = 100, g_abstol = 1e-4),
                       main_solver      = main,
                       main_opt         = (iterations = main_iters, g_abstol = 1e-6,
                                           show_trace = false))
end

# ------------------------------------------------------- backend internal API

_backend_mod(case::GPUCase) =
    case.panel ? SFrontiers.Panel_Backend :
    case.backend === :MCI ? SFrontiers.MCI_Backend : SFrontiers.MSLE_Backend

"NLL at fixed θ on an internal spec (device decided by the spec's arrays)."
function eval_nll(case::GPUCase, ispec, θ; chunks::Int=case.chunks)
    B = _backend_mod(case)
    case.panel && return B.panel_nll(ispec, θ; chunks = chunks)
    return case.backend === :MCI ? B.MCI_nll(ispec, θ; chunks=chunks) :
                                   B.qmc_nll(ispec, θ; chunks=chunks)
end

"jlms/bc at fixed θ (cross-sectional only)."
function eval_jlms(case::GPUCase, ispec, θ)
    B = _backend_mod(case)
    return B.jlms_bc_indices(ispec, θ; chunks=case.chunks)
end

"Marginal effects at fixed θ (cross-sectional; nothing when not applicable)."
function eval_marginal(case::GPUCase, ispec, θ)
    B = _backend_mod(case)
    if case.hetero === :scaling
        return B.marginal_effects_scaling(ispec, θ)
    else
        return B.marginal_effects(ispec, θ)
    end
end

# ------------------------------------------------------------------ case run

function _fit(case::GPUCase, uspec, meth, opt)
    marg = case.hetero === :scaling || !isempty(case.hetero)
    sfmodel_fit(spec=uspec, method=meth, init=nothing, optim_options=opt,
                marginal=marg, show_table=false, verbose=false)
end

"Post-fit consumers of a GPU result (each runs once per fit, so a scalar-index error is retried)."
function run_consumers(case::GPUCase, res_g, dat)
    out = Dict{Symbol,Any}()
    out[:predict_jlms] = stage(retry_allowscalar=true) do
        predict(res_g; frontier=dat.X, depvar=dat.y, what=:jlms); true
    end
    out[:residuals] = stage(retry_allowscalar=true) do
        StatsAPI.residuals(res_g); true
    end
    out[:fitted] = stage(retry_allowscalar=true) do
        StatsAPI.fitted(res_g); true
    end
    out[:show] = stage(retry_allowscalar=true) do
        sprint(show, MIME("text/plain"), res_g); true
    end
    out[:dataframe] = stage(retry_allowscalar=true) do
        DataFrame(jlms=res_g.jlms, bc=res_g.bc); true
    end
    out[:plotrecipe] = stage(retry_allowscalar=true) do
        RecipesBase.apply_recipe(Dict{Symbol,Any}(), res_g); true
    end
    return out
end

"Internal spec assembled directly from the model spec, without a fit (cross-sectional cases)."
function assemble_direct(case::GPUCase, uspec, dat; gpu::Bool)
    B     = _backend_mod(case)
    bspec = case.backend === :MCI ? uspec.mci_spec : uspec.msle_spec
    asm   = case.backend === :MCI ? B._assemble_MCI_spec : B._assemble_MSLE_spec
    return asm(bspec, build_method(case, dat; gpu=gpu))
end

"""
θ at the values that generate `make_xs_data` (frontier [1.0, 0.5, -0.3], σᵤ² = 0.25, σᵥ² = 0.09),
placed through the spec's `idx`; Float32 for a Float32 case. A model with any other parameter is
refused.
"""
function dgp_theta(case::GPUCase, ispec)
    B   = _backend_mod(case)
    idx = ispec.idx
    θ   = fill(NaN, B.plen(ispec.model, ispec.K, ispec.L, ispec.hetero; L_scaling=ispec.L_scaling))
    θ[idx.beta] .= [1.0, 0.5, -0.3]
    θ[idx.ineff.sigma_sq] .= log(0.25)
    θ[idx.noise.ln_sigma_v_sq] = log(0.09)
    all(isfinite, θ) || error("the data-generating θ sets only the frontier, ln σᵤ² and ln σᵥ²")
    return case.float32 ? Float32.(θ) : θ
end

"""
Checks of a case with `fit_full = false`, which runs no fit: the CPU and the GPU internal specs are
assembled directly and compared at the data-generating θ (NLL, jlms, bc); both NLLs must be finite,
and every draw must lie below 1 on both devices.
"""
function run_fixed_only(case::GPUCase, uspec, dat)
    checks = Dict{Symbol,Any}()
    sc = stage(() -> assemble_direct(case, uspec, dat; gpu=false))
    sg = stage(() -> assemble_direct(case, uspec, dat; gpu=true))
    for (k, s) in ((:assemble_cpu, sc), (:assemble_gpu, sg))
        checks[k] = Dict(:status => s[:status],
                         (s[:status] != :ok ? (:error => s[:error],) : ())...)
    end
    mc, mg = sc[:value], sg[:value]
    (mc === nothing || mg === nothing) && return checks

    dc = stage(() -> maximum(Array(mc.draws_2D)))
    dg = stage(() -> maximum(Array(mg.draws_2D)))
    checks[:draws_below_one] = _check_both_stage(dc, dg, m -> m < 1f0)

    sθ = stage(() -> dgp_theta(case, mc))
    if sθ[:status] != :ok
        checks[:theta] = Dict(:status => :error, :error => sθ[:error])
        return checks
    end
    θ  = sθ[:value]
    nc = stage(() -> eval_nll(case, mc, θ))
    ng = stage(() -> eval_nll(case, mg, θ))
    checks[:fixed_nll_1] = _cmp_scalar_stage(nc, ng, case)
    # both NLLs finite: the comparison above counts NaN against NaN as agreement
    checks[:nll_finite]  = _check_both_stage(nc, ng, isfinite)
    jc = stage(() -> eval_jlms(case, mc, θ))
    jg = stage(() -> eval_jlms(case, mg, θ); retry_allowscalar=true)
    checks[:fixed_jlms] = _cmp_field_stage(jc, jg, :jlms, case)
    checks[:fixed_bc]   = _cmp_field_stage(jc, jg, :bc, case)
    return checks
end

"""
Execute one case; returns a Dict ready for JSONL. Never throws.
"""
function run_case(case::GPUCase)
    t0  = time()
    rec = Dict{Symbol,Any}(
        :case => case.name, :tier => case.tier, :backend => case.backend,
        :dist => case.dist, :type => case.type,
        :hetero => case.hetero === :scaling ? "scaling" : string(case.hetero),
        :copula => case.copula, :chunks => case.chunks,
        :multiRand => case.multiRand, :n_draws => case.n_draws,
        :panel => case.panel, :float32 => case.float32, :tiny_u => case.tiny_u,
        :checks => Dict{Symbol,Any}())
    checks = rec[:checks]

    Tel = case.float32 ? Float32 : Float64
    dat = case.panel ? make_panel_data() : make_xs_data(N=case.N, T=Tel, tiny_u=case.tiny_u)

    s_spec = stage(() -> build_spec(case, dat))
    if s_spec[:status] != :ok
        checks[:spec] = s_spec; rec[:elapsed_s] = round(time()-t0, digits=1)
        return rec
    end
    uspec = s_spec[:value]
    opt   = build_opt(case)

    # ---- fixed-θ checks only (fit_full = false): no fits -----------------
    if !case.fit_full
        merge!(checks, run_fixed_only(case, uspec, dat))
        rec[:elapsed_s] = round(time() - t0, digits=1)
        return rec
    end

    # ---- full-fit mode: CPU reference, then GPU --------------------------
    s_cpu = stage(() -> _fit(case, uspec, build_method(case, dat; gpu=false), opt))
    checks[:fit_cpu] = Dict(:status => s_cpu[:status],
                            (s_cpu[:status] != :ok ? (:error => s_cpu[:error],) : ())...)
    s_gpu = stage(() -> _fit(case, uspec, build_method(case, dat; gpu=true), opt))
    checks[:fit_gpu] = Dict(:status => s_gpu[:status],
                            (s_gpu[:status] != :ok ? (:error => s_gpu[:error],
                                :scalar_index => get(s_gpu, :scalar_index, false)) : ())...)

    res_c = s_cpu[:value]; res_g = s_gpu[:value]

    if res_c !== nothing && res_g !== nothing
        both_conv = (res_c.converged == true) && (res_g.converged == true)
        d = Dict{Symbol,Any}(
            :converged_cpu => res_c.converged, :converged_gpu => res_g.converged,
            :redflag_cpu   => res_c.redflag,   :redflag_gpu   => res_g.redflag,
            :dll   => maxreldiff([res_c.loglikelihood], [res_g.loglikelihood]),
            :dcoef => maxreldiff(res_c.coeff, res_g.coeff),
            :dse   => maxreldiff(res_c.std_err, res_g.std_err),
            :djlms => maxreldiff(res_c.jlms, res_g.jlms),
            :dbc   => maxreldiff(res_c.bc,   res_g.bc))
        if res_c.marginal !== nothing && res_g.marginal !== nothing
            d[:dmarg] = maxreldiff(Matrix(res_c.marginal), Matrix(res_g.marginal))
        end
        rtol = case.float32 ? RTOL_F32 : RTOL_FIT
        if !both_conv
            d[:status] = :flags_only
        elseif something(d[:dcoef], Inf) <= rtol
            # equal coefficients must give equal log-likelihoods and JLMS
            d[:status] = something(d[:dll],   Inf) <= rtol &&
                         something(d[:djlms], Inf) <= rtol ? :pass : :fail
        else
            # The coefficients differ: the ridge rule. Each device evaluates its log-likelihood
            # at both optima; the case passes when the two agree within ATOL_RIDGE on each device.
            s_ridge = stage() do
                θ_c = collect(Float64, Array(res_c.coeff))
                θ_g = collect(Float64, Array(res_g.coeff))
                mc, mg = res_c.model, res_g.model
                ℓ_cc = -eval_nll(case, mc, θ_c); ℓ_cg = -eval_nll(case, mc, θ_g)
                ℓ_gc = -eval_nll(case, mg, θ_c); ℓ_gg = -eval_nll(case, mg, θ_g)
                Dict(:ll_cpu_at_cpu_opt => ℓ_cc, :ll_cpu_at_gpu_opt => ℓ_cg,
                     :ll_gpu_at_cpu_opt => ℓ_gc, :ll_gpu_at_gpu_opt => ℓ_gg,
                     :dll_cpu => ℓ_cc - ℓ_cg, :dll_gpu => ℓ_gc - ℓ_gg)
            end
            if s_ridge[:status] == :ok
                r = s_ridge[:value]
                d[:ridge]  = r
                d[:status] = ridge_pass(r[:ll_cpu_at_cpu_opt], r[:ll_cpu_at_gpu_opt]) &&
                             ridge_pass(r[:ll_gpu_at_cpu_opt], r[:ll_gpu_at_gpu_opt]) ? :pass_ridge : :fail
            else
                d[:status] = :error
                d[:ridge_error] = s_ridge[:error]
            end
        end
        d[:gpu_result_device_fields] = join(
            [string(f) for f in (:jlms, :bc) if getproperty(res_g, f) !== nothing &&
                 !(getproperty(res_g, f) isa Array)], ",")
        checks[:fullfit] = d
    end

    # ---- fixed-θ mode (cross-sectional; panel uses the fits' stored specs)
    θ = res_c !== nothing ? collect(Float64, Array(res_c.coeff)) : nothing
    if θ !== nothing && all(isfinite, θ)
        θs = case.float32 ? Any[Float32.(θ)] : Any[θ, θ .* 0.95 .+ 0.01]
        mc = res_c.model
        mg = res_g !== nothing ? res_g.model : nothing
        if mg === nothing && !case.panel   # GPU fit died: assemble a GPU spec directly
            s_asm = stage() do
                B = _backend_mod(case)
                bspec = case.backend === :MCI ? uspec.mci_spec : uspec.msle_spec
                asm = case.backend === :MCI ? B._assemble_MCI_spec : B._assemble_MSLE_spec
                asm(bspec, build_method(case, dat; gpu=true))
            end
            checks[:assemble_gpu] = Dict(:status => s_asm[:status],
                (s_asm[:status] != :ok ? (:error => s_asm[:error],) : ())...)
            mg = s_asm[:value]
        end
        if mg !== nothing
            for (k, θk) in enumerate(θs)
                nc = stage(() -> eval_nll(case, mc, θk))
                ng = stage(() -> eval_nll(case, mg, θk))
                checks[Symbol("fixed_nll_$k")] = _cmp_scalar_stage(nc, ng, case)
            end
            if !case.panel
                jc = stage(() -> eval_jlms(case, mc, θs[1]))
                jg = stage(() -> eval_jlms(case, mg, θs[1]); retry_allowscalar=true)
                checks[:fixed_jlms] = _cmp_field_stage(jc, jg, :jlms, case)
                checks[:fixed_bc]   = _cmp_field_stage(jc, jg, :bc, case)
                if case.hetero === :scaling || !isempty(case.hetero)
                    mcm = stage(() -> eval_marginal(case, mc, θs[1]); retry_allowscalar=true)
                    mgm = stage(() -> eval_marginal(case, mg, θs[1]); retry_allowscalar=true)
                    checks[:fixed_marginal] = _cmp_marg_stage(mcm, mgm, case)
                end
                # chunk-invariance ON DEVICE: chunks=1 vs case.chunks at same θ
                c1 = stage(() -> eval_nll(case, mg, θs[1]; chunks=1))
                cN = stage(() -> eval_nll(case, mg, θs[1]; chunks=case.chunks))
                checks[:gpu_chunk_invariance] = _cmp_scalar_stage(c1, cN, case)
            end
        end
    end

    # ---- post-fit consumers of a GPU result --------------------------------
    if case.consumers && res_g !== nothing
        checks[:consumers] = run_consumers(case, res_g, dat)
    end

    rec[:elapsed_s] = round(time() - t0, digits=1)
    return rec
end

function _cmp_scalar_stage(sc, sg, case::GPUCase)
    d = Dict{Symbol,Any}()
    if sc[:status] == :ok && sg[:status] in (:ok, :ok_with_allowscalar)
        d[:diff] = maxreldiff([sc[:value]], [sg[:value]])
        rtol = case.float32 ? 1e-4 : RTOL_FIXED
        d[:status] = d[:diff] <= rtol ? :pass : :fail
        sg[:status] == :ok_with_allowscalar && (d[:allowscalar_needed] = true)
    else
        d[:status] = :error
        sc[:status] != :ok && (d[:error_cpu] = sc[:error])
        sg[:status] == :error && (d[:error_gpu] = sg[:error];
                                  d[:scalar_index] = get(sg, :scalar_index, false))
    end
    return d
end

function _cmp_field_stage(sc, sg, f::Symbol, case::GPUCase)
    d = Dict{Symbol,Any}()
    if sc[:status] == :ok && sg[:status] in (:ok, :ok_with_allowscalar)
        d[:diff] = maxreldiff(getproperty(sc[:value], f), getproperty(sg[:value], f))
        rtol = case.float32 ? 1e-4 : RTOL_FIXED
        d[:status] = d[:diff] <= rtol ? :pass : :fail
        sg[:status] == :ok_with_allowscalar && (d[:allowscalar_needed] = true)
    else
        d[:status] = :error
        sc[:status] != :ok && (d[:error_cpu] = sc[:error])
        sg[:status] == :error && (d[:error_gpu] = sg[:error];
                                  d[:scalar_index] = get(sg, :scalar_index, false))
    end
    return d
end

function _cmp_marg_stage(sc, sg, case::GPUCase)
    d = Dict{Symbol,Any}()
    if sc[:status] in (:ok, :ok_with_allowscalar) && sg[:status] in (:ok, :ok_with_allowscalar)
        vc, vg = sc[:value], sg[:value]
        mc = vc isa Tuple ? vc[1] : vc
        mg = vg isa Tuple ? vg[1] : vg
        if mc === nothing && mg === nothing
            d[:status] = :skipped
        else
            d[:diff] = maxreldiff(mc isa AbstractDataFrame ? Matrix(mc) : mc,
                                  mg isa AbstractDataFrame ? Matrix(mg) : mg)
            d[:status] = something(d[:diff], Inf) <= RTOL_FIXED ? :pass : :fail
        end
        (sc[:status] == :ok_with_allowscalar || sg[:status] == :ok_with_allowscalar) &&
            (d[:allowscalar_needed] = true)
    else
        d[:status] = :error
        sc[:status] == :error && (d[:error_cpu] = sc[:error])
        sg[:status] == :error && (d[:error_gpu] = sg[:error];
                                  d[:scalar_index] = get(sg, :scalar_index, false))
    end
    return d
end

"Pass when `pred` holds for the CPU and the GPU value of two stages; records both values."
function _check_both_stage(sc, sg, pred)
    d = Dict{Symbol,Any}(:cpu => sc[:value], :gpu => sg[:value])
    if sc[:status] == :ok && sg[:status] == :ok
        d[:status] = pred(sc[:value]) && pred(sg[:value]) ? :pass : :fail
    else
        d[:status] = :error
        sc[:status] != :ok && (d[:error_cpu] = sc[:error])
        sg[:status] != :ok && (d[:error_gpu] = sg[:error])
    end
    return d
end

# ------------------------------------------------------------------ case list

const HARD_DISTS = (:Gamma, :Lomax, :Weibull, :Lognormal)

_nm(d::Symbol) = d in HARD_DISTS
_mci_dists()  = [:HalfNormal, :TruncatedNormal, :Exponential, :Weibull,
                 :Lognormal, :Lomax, :Rayleigh, :Gamma,
                 :Erlang2, :Erlang3, :Erlang4]
_msle_dists() = [:HalfNormal, :TruncatedNormal, :Exponential, :Weibull,
                 :Lognormal, :Lomax, :Rayleigh,
                 :Erlang2, :Erlang3, :Erlang4]           # no Gamma: it is MCI-only

function build_cases(tiers::Vector{Int})
    cases = GPUCase[]
    # -- Tier 1: every dist × backend at defaults
    if 1 in tiers
        for d in _mci_dists()
            push!(cases, GPUCase(name="T1-MCI-$d",  tier=1, backend=:MCI,  dist=d, nm_main=_nm(d)))
        end
        for d in _msle_dists()
            push!(cases, GPUCase(name="T1-MSLE-$d", tier=1, backend=:MSLE, dist=d, nm_main=_nm(d)))
        end
    end
    # -- Tier 2: GPU mechanics on representative dists
    if 2 in tiers
        reps = [(:MCI, :HalfNormal), (:MSLE, :HalfNormal), (:MCI, :Gamma),
                (:MCI, :Weibull), (:MSLE, :Weibull)]
        for (b, d) in reps, ch in (1, 7, 403)     # N=400 ⇒ 403 forces empty trailing chunks
            push!(cases, GPUCase(name="T2-$b-$d-chunks$ch", tier=2, backend=b, dist=d,
                                 chunks=ch, nm_main=_nm(d)))
        end
        for (b, d) in reps
            push!(cases, GPUCase(name="T2-$b-$d-multiRandF", tier=2, backend=b, dist=d,
                                 multiRand=false, nm_main=_nm(d)))
            push!(cases, GPUCase(name="T2-$b-$d-cost", tier=2, backend=b, dist=d,
                                 type=:cost, nm_main=_nm(d)))
        end
        for b in (:MCI, :MSLE)
            push!(cases, GPUCase(name="T2-$b-HN-hetero", tier=2, backend=b,
                                 dist=:HalfNormal, hetero=[:sigma_u_sq]))
            push!(cases, GPUCase(name="T2-$b-TN-hetero-mu", tier=2, backend=b,
                                 dist=:TruncatedNormal, hetero=[:mu]))
            push!(cases, GPUCase(name="T2-$b-Exp-hetero", tier=2, backend=b,
                                 dist=:Exponential, hetero=[:sigma_u_sq]))
            push!(cases, GPUCase(name="T2-$b-TN-scaling", tier=2, backend=b,
                                 dist=:TruncatedNormal, hetero=:scaling))
            for cop in (:Gaussian, :Clayton, :Gumbel, :Clayton90)
                push!(cases, GPUCase(name="T2-$b-HN-copula$cop", tier=2, backend=b,
                                     dist=:HalfNormal, copula=cop))
            end
        end
        # Erlang mechanics: hetero-θ and scaling (MCI: delegated Gamma kernels with
        # the shape pinned; MSLE: standardized Newton quantile on the fixed draws;
        # θ-channel marginals compared on device)
        push!(cases, GPUCase(name="T2-MCI-Erlang2-hetero", tier=2, backend=:MCI,
                             dist=:Erlang2, hetero=[:theta]))
        push!(cases, GPUCase(name="T2-MCI-Erlang2-scaling", tier=2, backend=:MCI,
                             dist=:Erlang2, hetero=:scaling))
        push!(cases, GPUCase(name="T2-MSLE-Erlang2-hetero", tier=2, backend=:MSLE,
                             dist=:Erlang2, hetero=[:theta]))
    end
    # -- Tier 3: panel_TFE (only panel datatype with a GPU/simulation backend)
    if 3 in tiers
        for b in (:MCI, :MSLE), d in (:HalfNormal, :Weibull, :Erlang2)
            push!(cases, GPUCase(name="T3-panel-$b-$d", tier=3, backend=b, dist=d,
                                 panel=true, nm_main=_nm(d)))
        end
        push!(cases, GPUCase(name="T3-panel-MCI-HN-chunks1", tier=3, backend=:MCI,
                             dist=:HalfNormal, panel=true, chunks=1))
        # 40 firms in chunks of 6, the last of 4 (the per-chunk device path with an uneven
        # last chunk), and shared draws in a panel
        push!(cases, GPUCase(name="T3-panel-MCI-HN-chunks7", tier=3, backend=:MCI,
                             dist=:HalfNormal, panel=true, chunks=7))
        push!(cases, GPUCase(name="T3-panel-MSLE-HN-multiRandF", tier=3, backend=:MSLE,
                             dist=:HalfNormal, panel=true, multiRand=false))
    end
    # -- Tier 4: edges
    if 4 in tiers
        push!(cases, GPUCase(name="T4-userdraws-mismatch-MCI", tier=4, backend=:MCI,
                             dist=:HalfNormal, user_cpu_draws=true))
        # user draws on the other device: host draws × multiRand=false (MCI's 1×D
        # branch) and CuArray draws in a GPU=false fit (MSLE's _to_device_array)
        push!(cases, GPUCase(name="T4-userdraws-multiRandF-MCI", tier=4, backend=:MCI,
                             dist=:HalfNormal, user_cpu_draws=true, multiRand=false))
        push!(cases, GPUCase(name="T4-userdraws-multiRandF-MSLE", tier=4, backend=:MSLE,
                             dist=:HalfNormal, user_cpu_draws=true, multiRand=false))
        push!(cases, GPUCase(name="T4-userdrawsGPU-MCI", tier=4, backend=:MCI,
                             dist=:HalfNormal, user_gpu_draws=true))
        push!(cases, GPUCase(name="T4-userdrawsGPU-MSLE", tier=4, backend=:MSLE,
                             dist=:HalfNormal, user_gpu_draws=true))
        push!(cases, GPUCase(name="T4-consumers-MCI-HN", tier=4, backend=:MCI,
                             dist=:HalfNormal, consumers=true))
        push!(cases, GPUCase(name="T4-float32-MCI-HN", tier=4, backend=:MCI,
                             dist=:HalfNormal, float32=true))
        push!(cases, GPUCase(name="T4-float32-MSLE-HN", tier=4, backend=:MSLE,
                             dist=:HalfNormal, float32=true))
        # One Float32 Halton sequence of N × n_draws = 2^25 points: the point that rounds to
        # 1f0 in single precision is used and kept below 1. Fixed-θ checks only (no fit).
        push!(cases, GPUCase(name="T4-float32-long-MCI-HN", tier=4, backend=:MCI,
                             dist=:HalfNormal, float32=true, n_draws=8192, N=4096,
                             fit_full=false))
    end
    return cases
end

const SMOKE_CASES = ["T1-MCI-HalfNormal", "T1-MSLE-Weibull", "T2-MCI-HN-hetero",
                     "T1-MCI-Erlang2", "T1-MSLE-Erlang2",                    # Erlang: pinned-shape Gamma kernels (MCI) + Newton quantile (MSLE)
                     "T4-userdraws-multiRandF-MCI", "T4-userdrawsGPU-MSLE"]  # last two: user draws, other device
