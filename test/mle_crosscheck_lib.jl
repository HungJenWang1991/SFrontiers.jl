# ---------------------------------------------------------------------------
# MLE cross-check library.
#
# Cross-backend fixed-θ equivalence for the models shared by all three
# backends (Normal noise × HalfNormal / TruncatedNormal / Exponential,
# homoscedastic + heteroscedastic + TN-scaling):
#
#   MLE analytic  — SFmle.LL_T(modelid, ...) hand-assembled per case, but
#                   ANCHORED: at each case's own unified-MLE-fit optimum the
#                   hand-built closure must reproduce the fit's reported
#                   loglikelihood (self_check). θ maps MCI→MLE order via the
#                   package's _reorder_init_mci_to_mle (identity except
#                   TN+scaling), NOT hand permutation.
#   MCI / MSLE    — MCI_nll / qmc_nll on assembled internal specs at the SAME
#                   raw θ (QMC tolerance, 2^12 → 2^14 trend).
#   QuadGK oracle — sharp independent convolution integral (both levels AND
#                   the base→alt LL difference, to expose any dropped
#                   additive constants separately from level errors).
#   jlms/bc       — MLE's analytic jlms/bc at its fit optimum vs the QuadGK
#                   conditional moments vs MCI simulation at the same θ̂.
#
# Requires noise_oracle_lib.jl (stage/json/maxreldiff/quad_moments) to be
# include()d first.
# ---------------------------------------------------------------------------

const SFmle2 = SFrontiers.MLE_Backend.SFmle

# ------------------------------------------------------------ data generation

function make_crosscheck_data(; N::Int=80, seed::Int=2027, type::Symbol=:prod)
    rng = Random.MersenneTwister(seed)
    x1 = randn(rng, N); x2 = randn(rng, N)
    X  = hcat(ones(N), x1, x2)
    z1 = abs.(randn(rng, N)) .+ 0.5
    z2 = abs.(randn(rng, N)) .+ 0.5
    Zh = hcat(ones(N), z1)              # hetero zvar: _cons FIRST (a homoscedastic parameter needs a constant first zvar column)
    Zs = hcat(z1, z2)                   # scaling zvar: NO constant column
    v  = 0.3 .* randn(rng, N)
    u  = 0.5 .* abs.(randn(rng, N))
    y  = X * [1.0, 0.5, -0.3] .+ v .+ (type === :cost ? u : -u)
    (y=y, X=X, Zh=Zh, Zs=Zs, N=N)
end

# ------------------------------------------------------------ case definition

Base.@kwdef struct CrosscheckCase
    name::String
    ineff::Symbol                  # :HalfNormal | :TruncatedNormal | :Exponential
    type::Symbol  = :prod
    hetero        = Symbol[]       # Symbol[] | Vector{Symbol} | :scaling
end

# θ values per idx field name, per case (vectors match idx block lengths).
function theta_vals(case::CrosscheckCase, label::Symbol)
    b = label === :base
    d = Dict{Symbol,Vector{Float64}}(
        :beta          => b ? [1.0, 0.5, -0.3] : [0.8, 0.6, -0.2],
        :ln_sigma_v_sq => b ? [2 * log(0.3)]   : [2 * log(0.45)])
    het = case.hetero
    if case.ineff === :HalfNormal
        d[:sigma_sq] = (het isa Vector && :sigma_u_sq in het) ?
            (b ? [2 * log(0.5), 0.3] : [2 * log(0.7), -0.2]) :
            (b ? [2 * log(0.5)]      : [2 * log(0.7)])
    elseif case.ineff === :Exponential
        d[:sigma_sq] = (het isa Vector && :sigma_u_sq in het) ?
            (b ? [2 * log(0.4), 0.25] : [2 * log(0.6), -0.2]) :
            (b ? [2 * log(0.4)]       : [2 * log(0.6)])
    elseif case.ineff === :TruncatedNormal
        d[:mu] = (het isa Vector && :mu in het) ?
            (b ? [0.2, 0.15] : [0.35, -0.1]) :
            (b ? [0.2]       : [0.35])
        d[:sigma_u] = b ? [2 * log(0.5)] : [2 * log(0.7)]
        het === :scaling && (d[:delta] = b ? [0.2, -0.15] : [0.3, 0.1])
    end
    return d
end

"Build raw θ (MCI canonical order) from an assembled spec's idx + value dict."
function build_theta2(ispec, vals::Dict{Symbol,Vector{Float64}})
    idx = ispec.idx
    all_inds = Int[]
    assign = Tuple{Int,Float64}[]
    function put!(field, inds)
        haskey(vals, field) || error("no θ values for idx field $field")
        v = vals[field]
        ii = _asvec(inds)
        length(ii) == length(v) ||
            error("length mismatch for $field: idx $(length(ii)) vs vals $(length(v))")
        for (a, bv) in zip(ii, v)
            push!(assign, (a, bv))
        end
        append!(all_inds, ii)
    end
    put!(:beta, idx.beta)
    dind = _flatten_idx(hasproperty(idx, :delta) ? idx.delta : nothing)
    isempty(dind) || put!(:delta, dind)
    for (k, inds) in pairs(idx.ineff)
        inds isa Bool && continue
        put!(k, inds)
    end
    for (k, inds) in pairs(idx.noise)
        put!(k, inds)
    end
    isempty(_flatten_idx(idx.copula)) || error("copula not covered by the MLE cross-check")
    sort(all_inds) == collect(1:maximum(all_inds)) ||
        error("θ coverage gap: idx blocks $(sort(all_inds)) don't tile 1:n")
    θ = zeros(maximum(all_inds))
    for (a, bv) in assign
        θ[a] = bv
    end
    return θ
end

# -------------------------------------------------------------- spec builders

function build_uspec(case::CrosscheckCase, dat)
    kw = Dict{Symbol,Any}(:noise => :Normal, :ineff => case.ineff, :type => case.type)
    if case.hetero === :scaling
        return sfmodel_spec(; depvar=dat.y, frontier=dat.X, zvar=dat.Zs,
                              hetero=:scaling, kw...)
    elseif case.hetero isa Vector && !isempty(case.hetero)
        return sfmodel_spec(; depvar=dat.y, frontier=dat.X, zvar=dat.Zh,
                              hetero=collect(case.hetero), kw...)
    else
        return sfmodel_spec(; depvar=dat.y, frontier=dat.X, kw...)
    end
end

function assemble_sim(case::CrosscheckCase, uspec, backend::Symbol, n_draws::Int)
    meth = sfmodel_method(method=backend, n_draws=n_draws, multiRand=false, chunks=1)
    B = backend === :MCI ? SFrontiers.MCI_Backend : SFrontiers.MSLE_Backend
    bspec = backend === :MCI ? uspec.mci_spec : uspec.msle_spec
    asm   = backend === :MCI ? B._assemble_MCI_spec : B._assemble_MSLE_spec
    return asm(bspec, meth)
end

# ------------------------------------------------------------- MLE assembly

"""
Hand-assemble the LL_T closure inputs for a case, mirroring getvar(modelid):
MLE block order [x | z(μ) | q(hscale) | w(lnσᵤ²/lnλ²) | v(lnσᵥ²)], with a
synthesized _cons ones column for every homoscedastic channel. Returns
(modelid, f) where f(rho_mle) = LL_T(...) = NEGATIVE summed loglik.
"""
function mle_closure(case::CrosscheckCase, dat)
    N = dat.N
    Y = reshape(dat.y, N, 1)
    X = dat.X
    K = size(X, 2)
    ones1 = ones(N, 1)
    hetvec = case.hetero isa Vector ? case.hetero : Symbol[]

    if case.ineff === :HalfNormal
        modelid = SFmle2.Half
        W = :sigma_u_sq in hetvec ? dat.Zh : ones1
        Z = (); Q = ()
        nz = 0; nq = 0; nw = size(W, 2)
    elseif case.ineff === :Exponential
        modelid = SFmle2.Expo
        W = :sigma_u_sq in hetvec ? dat.Zh : ones1
        Z = (); Q = ()
        nz = 0; nq = 0; nw = size(W, 2)
    elseif case.ineff === :TruncatedNormal && case.hetero === :scaling
        modelid = SFmle2.Trun_Scale
        Z = ones1                       # μ (homoscedastic base)
        Q = dat.Zs                      # hscale
        W = ones1
        nz = 1; nq = size(Q, 2); nw = 1
    elseif case.ineff === :TruncatedNormal
        modelid = SFmle2.Trun
        Z = :mu in hetvec ? dat.Zh : ones1
        Q = ()
        W = :sigma_u_sq in hetvec ? dat.Zh : ones1
        nz = size(Z, 2); nq = 0; nw = size(W, 2)
    else
        error("unhandled ineff in the MLE cross-check: $(case.ineff)")
    end
    V = ones1; nv = 1

    begx = 1;                  endx = K
    begz = nz > 0 ? K + 1 : 0; endz = nz > 0 ? K + nz : 0
    begq = nq > 0 ? K + nz + 1 : 0
    endq = nq > 0 ? K + nz + nq : 0
    begw = K + nz + nq + 1;    endw = K + nz + nq + nw
    begv = endw + 1;           endv = endw + nv
    pos = (begx=begx, endx=endx, begz=begz, endz=endz,
           begq=begq, endq=endq, begw=begw, endw=endw,
           begv=begv, endv=endv)

    porc = case.type === :cost ? -1 : 1
    f = rho -> SFmle2.LL_T(modelid, Y, X, Z, Q, W, V,
                           porc, N, pos, collect(Float64, rho), (), nothing)
    return modelid, f, endv
end

# ----------------------------------------------------------------- oracles

"Per-observation inefficiency density closures for the quadrature oracle."
function crosscheck_fu_perobs(case::CrosscheckCase, dat, vals::Dict)
    N = dat.N
    hetvec = case.hetero isa Vector ? case.hetero : Symbol[]
    if case.ineff === :HalfNormal
        γ = vals[:sigma_sq]
        lin = :sigma_u_sq in hetvec ? dat.Zh * γ : fill(γ[1], N)
        return [let d = truncated(Normal(0.0, exp(0.5 * lin[i])), 0.0, Inf)
                    u -> pdf(d, u)
                end for i in 1:N]
    elseif case.ineff === :Exponential
        γ = vals[:sigma_sq]
        lin = :sigma_u_sq in hetvec ? dat.Zh * γ : fill(γ[1], N)
        return [let d = Exponential(sqrt(exp(lin[i])))   # exp(θ) = σᵤ² = Var(u)
                    u -> pdf(d, u)
                end for i in 1:N]
    elseif case.ineff === :TruncatedNormal
        δμ = vals[:mu]; γ = vals[:sigma_u]
        μs = :mu in hetvec ? dat.Zh * δμ : fill(δμ[1], N)
        σu = exp(0.5 * γ[1])
        if case.hetero === :scaling
            h = exp.(dat.Zs * vals[:delta])
            return [let d = truncated(Normal(μs[i], σu), 0.0, Inf), hi = h[i]
                        u -> pdf(d, u / hi) / hi
                    end for i in 1:N]
        else
            return [let d = truncated(Normal(μs[i], σu), 0.0, Inf)
                        u -> pdf(d, u)
                    end for i in 1:N]
        end
    end
    error("unhandled ineff $(case.ineff)")
end

# ------------------------------------------------------------------ case run

const CROSSCHECK_DRAWS = [2^12, 2^14]

"""
Execute one MLE cross-check case. Never throws. Steps:
 1. unified MLE fit → θ̂, reported LL, analytic jlms/bc
 2. hand-built LL_T self-check at θ̂ (anchors the closure wiring)
 3. fixed-θ battery at base/alt: MLE vs MCI vs MSLE vs QuadGK
    (levels AND base→alt differences)
 4. jlms/bc at θ̂: MLE analytic vs QuadGK vs MCI simulation
"""
function run_crosscheck_case(case::CrosscheckCase)
    t0  = time()
    rec = Dict{Symbol,Any}(
        :kind => :mle_crosscheck, :case => case.name, :ineff => case.ineff,
        :type => case.type,
        :hetero => case.hetero === :scaling ? "scaling" : string(case.hetero),
        :checks => Dict{Symbol,Any}())
    checks = rec[:checks]

    dat = make_crosscheck_data(type=case.type)
    s   = case.type === :cost ? -1 : 1

    s_us = stage(() -> build_uspec(case, dat))
    if s_us[:status] != :ok
        checks[:uspec] = s_us; rec[:elapsed_s] = round(time() - t0, digits=1)
        return rec
    end
    uspec = s_us[:value]

    s_mle = stage() do
        opt = sfmodel_opt(warmstart_solver = NelderMead(),
                          warmstart_opt    = (iterations = 200, g_abstol = 1e-4),
                          main_solver      = Newton(),
                          main_opt         = (iterations = 300, g_abstol = 1e-8,
                                              show_trace = false))
        sfmodel_fit(spec=uspec, method=sfmodel_method(method=:MLE), init=nothing,
                    optim_options=opt, marginal=false, show_table=false, verbose=false)
    end
    checks[:mle_fit] = Dict(:status => s_mle[:status],
        (s_mle[:status] != :ok ? (:error => s_mle[:error],) : ())...)
    res = s_mle[:value]

    s_cl = stage(() -> mle_closure(case, dat))
    if s_cl[:status] != :ok
        checks[:mle_closure] = s_cl; rec[:elapsed_s] = round(time() - t0, digits=1)
        return rec
    end
    modelid, nll_mle, nparams_mle = s_cl[:value]

    # ---- self-check: hand-built closure vs the fit's reported loglik -------
    if res !== nothing && res.coeff !== nothing && all(isfinite, res.coeff)
        sc = stage() do
            rho = SFrontiers._reorder_init_mci_to_mle(uspec, collect(Float64, res.coeff))
            length(rho) == nparams_mle || error("param count mismatch: " *
                "reorder gave $(length(rho)), closure expects $nparams_mle")
            nll_mle(rho)
        end
        if sc[:status] == :ok
            dself = abs(sc[:value] - (-res.loglikelihood)) /
                    max(abs(res.loglikelihood), 1.0)
            checks[:self_check] = Dict(:nll_closure => sc[:value],
                :nll_reported => -res.loglikelihood, :reldiff => dself,
                :status => dself <= 1e-8 ? :pass : :attention)
        else
            checks[:self_check] = Dict(:status => :error, :error => sc[:error])
        end
    end

    # ---- assembled sim specs -----------------------------------------------
    s_specs = stage() do
        Dict(b => [assemble_sim(case, uspec, b, D) for D in CROSSCHECK_DRAWS]
             for b in (:MCI, :MSLE))
    end
    if s_specs[:status] != :ok
        checks[:assemble] = s_specs; rec[:elapsed_s] = round(time() - t0, digits=1)
        return rec
    end
    specs = s_specs[:value]

    # ---- fixed-θ battery ----------------------------------------------------
    levels = Dict{Symbol,Dict{Symbol,Float64}}()   # θlabel → source → NLL
    for θlabel in (:base, :alt)
        vals = theta_vals(case, θlabel)
        s_θ = stage(() -> build_theta2(specs[:MCI][1], vals))
        if s_θ[:status] != :ok
            checks[Symbol("theta_$θlabel")] = s_θ
            continue
        end
        θ = s_θ[:value]
        ε = dat.y .- dat.X * vals[:beta]
        σv = exp(0.5 * vals[:ln_sigma_v_sq][1])
        fv = let nd = Normal(0.0, σv); z -> pdf(nd, z) end
        fus = crosscheck_fu_perobs(case, dat, vals)

        d = Dict{Symbol,Any}()
        lv = Dict{Symbol,Float64}()

        s_q = stage() do
            mom = [quad_moments(ε[i], s, fv, fus[i]) for i in 1:dat.N]
            (mom=mom, nll=-sum(log(m.den) for m in mom))
        end
        if s_q[:status] != :ok
            d[:quad_error] = s_q[:error]
            checks[Symbol("fixed_$θlabel")] = d
            continue
        end
        quad = s_q[:value]
        d[:quad_nll] = quad.nll; lv[:quad] = quad.nll

        s_m = stage() do
            rho = SFrontiers._reorder_init_mci_to_mle(uspec, θ)
            nll_mle(rho)
        end
        if s_m[:status] == :ok
            v = Float64(s_m[:value])
            d[:mle_nll] = v; lv[:mle] = v
            d[:dmle_quad_perobs] = abs(v - quad.nll) / dat.N
            d[:mle_status] = d[:dmle_quad_perobs] <= 1e-8 ? :pass : :attention
        else
            d[:mle_error] = s_m[:error]; d[:mle_status] = :error
        end

        for b in (:MCI, :MSLE)
            B = b === :MCI ? SFrontiers.MCI_Backend : SFrontiers.MSLE_Backend
            diffs = Float64[]
            for (k, D) in enumerate(CROSSCHECK_DRAWS)
                s_n = stage(() -> b === :MCI ?
                    B.MCI_nll(specs[b][k], θ; chunks=1) :
                    B.qmc_nll(specs[b][k], θ; chunks=1))
                if s_n[:status] == :ok
                    v = Float64(s_n[:value])
                    d[Symbol("$(b)_nll_$D")] = v
                    push!(diffs, abs(v - quad.nll) / dat.N)
                    d[Symbol("d$(b)_quad_perobs_$D")] = diffs[end]
                    k == length(CROSSCHECK_DRAWS) && (lv[b] = v)
                else
                    d[Symbol("$(b)_error_$D")] = s_n[:error]
                    push!(diffs, NaN)
                end
            end
            trend_ok = !any(isnan, diffs) &&
                       (diffs[end] <= 1.5 * diffs[1] || diffs[end] <= 1e-6)
            d[Symbol("$(b)_status")] =
                (!isempty(diffs) && !isnan(diffs[end]) && diffs[end] <= 1e-4 &&
                 trend_ok) ? :pass : :attention
        end
        checks[Symbol("fixed_$θlabel")] = d
        levels[θlabel] = lv
    end

    # ---- dropped-constant detector: base→alt differences --------------------
    if haskey(levels, :base) && haskey(levels, :alt)
        lb, la = levels[:base], levels[:alt]
        dd = Dict{Symbol,Any}()
        if haskey(lb, :quad)
            Δq = la[:quad] - lb[:quad]
            for src in (:mle, :MCI, :MSLE)
                (haskey(lb, src) && haskey(la, src)) || continue
                Δ = la[src] - lb[src]
                dd[Symbol("ddiff_$(src)_perobs")] = abs(Δ - Δq) / dat.N
                dd[Symbol("offset_$(src)_base_perobs")] = (lb[src] - lb[:quad]) / dat.N
                dd[Symbol("offset_$(src)_alt_perobs")]  = (la[src] - la[:quad]) / dat.N
            end
            dd[:status] = get(dd, :ddiff_mle_perobs, Inf) <= 1e-8 ? :pass : :attention
        end
        checks[:ll_differences] = dd
    end

    # ---- jlms/bc at the MLE optimum -----------------------------------------
    if res !== nothing && res.coeff !== nothing && all(isfinite, res.coeff) &&
       res.jlms !== nothing
        s_jb = stage() do
            θh = collect(Float64, res.coeff)
            # oracle at θ̂: rebuild per-obs params from θ̂ via the same idx map
            idx = specs[:MCI][1].idx
            vals = Dict{Symbol,Vector{Float64}}()
            vals[:beta] = θh[_asvec(idx.beta)]
            dind = _flatten_idx(hasproperty(idx, :delta) ? idx.delta : nothing)
            isempty(dind) || (vals[:delta] = θh[dind])
            for (k, inds) in pairs(idx.ineff)
                inds isa Bool && continue
                vals[k] = θh[_asvec(inds)]
            end
            for (k, inds) in pairs(idx.noise)
                vals[k] = θh[_asvec(inds)]
            end
            ε = dat.y .- dat.X * vals[:beta]
            σv = exp(0.5 * vals[:ln_sigma_v_sq][1])
            fv = let nd = Normal(0.0, σv); z -> pdf(nd, z) end
            fus = crosscheck_fu_perobs(case, dat, vals)
            nsel = min(N_JLMS_OBS, dat.N)
            mom = [quad_moments(ε[i], s, fv, fus[i]) for i in 1:nsel]
            B = SFrontiers.MCI_Backend
            rmci = B.jlms_bc_indices(specs[:MCI][end], θh; chunks=1)
            (jq=[m.jlms for m in mom], bq=[m.bc for m in mom],
             jm=collect(Float64, res.jlms[1:nsel]),
             bm=res.bc === nothing ? nothing : collect(Float64, res.bc[1:nsel]),
             jc=collect(Float64, rmci.jlms[1:nsel]),
             bc=collect(Float64, rmci.bc[1:nsel]))
        end
        jd = Dict{Symbol,Any}()
        if s_jb[:status] == :ok
            v = s_jb[:value]
            jd[:jlms_mle]  = v.jm; jd[:jlms_quad] = v.jq; jd[:jlms_mci] = v.jc
            jd[:djlms_mle_quad] = maxreldiff(v.jm, v.jq)
            jd[:djlms_mci_quad] = maxreldiff(v.jc, v.jq)
            if v.bm !== nothing
                jd[:dbc_mle_quad] = maxreldiff(v.bm, v.bq)
                jd[:dbc_mci_quad] = maxreldiff(v.bc, v.bq)
            end
            jd[:status] = (something(jd[:djlms_mle_quad], Inf) <= 1e-6 &&
                           something(jd[:djlms_mci_quad], Inf) <= 2e-3) ? :pass : :attention
        else
            jd[:status] = :error; jd[:error] = s_jb[:error]
        end
        checks[:jlms_bc_at_opt] = jd
    end

    rec[:elapsed_s] = round(time() - t0, digits=1)
    return rec
end

# ------------------------------------------------------------------ case list

function build_crosscheck_cases()
    return CrosscheckCase[
        CrosscheckCase(name="HN-homo-prod",     ineff=:HalfNormal),
        CrosscheckCase(name="HN-homo-cost",     ineff=:HalfNormal, type=:cost),
        CrosscheckCase(name="HN-hetero-prod",   ineff=:HalfNormal, hetero=[:sigma_u_sq]),
        CrosscheckCase(name="TN-homo-prod",     ineff=:TruncatedNormal),
        CrosscheckCase(name="TN-heteromu-prod", ineff=:TruncatedNormal, hetero=[:mu]),
        CrosscheckCase(name="Exp-homo-prod",    ineff=:Exponential),
        CrosscheckCase(name="Exp-hetero-prod",  ineff=:Exponential, hetero=[:sigma_u_sq]),
        CrosscheckCase(name="TN-scaling-prod",  ineff=:TruncatedNormal, hetero=:scaling),
    ]
end

const CROSSCHECK_SMOKE = ["HN-homo-prod", "HN-homo-cost"]
