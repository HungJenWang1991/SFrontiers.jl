# ---------------------------------------------------------------------------
# Noise-oracle library.
#
# Numerically verifies the StudentT / Laplace noise kernels
# against independent oracles:
#   kernel level — log-pdf / pdf / CDF of the noise kernels vs Distributions.jl
#                  (TDist scaled, Laplace), incl. tail arguments.
#   model level  — backend NLL / jlms / bc / per-obs likelihood at FIXED θ vs a
#                  QuadGK integral of the convolution density
#                  f(ε) = ∫ f_v(ε + s·u) f_u(u) [· c(F_v,F_u)] du,  s=+1 prod/−1 cost.
#                  Normal noise runs as a CONTROL:
#                  if the control fails, the oracle wiring is wrong, not the code.
#
# Tolerances: model-level comparisons are bounded by the QMC error of the draw
# count (n_draws 2^10 → 2^14 must show a shrinking trend), NOT machine eps.
# The Laplace kernels of both backends smooth |z| as sqrt(z^2 + 1e-16); the
# oracle therefore ALSO evaluates the same smoothed density (sharp comparison),
# and records the exact-|z| quadrature separately (expected ~1e-8/b-scale gap).
#
# Assumes SFrontiers, Distributions, QuadGK, Random, Statistics, Dates are in
# scope. QuadGK and Distributions are dependencies of the package.
# Patterns reused from test/gpu_equivalence_lib.jl: case structs, stage()
# error classification, maxreldiff (NaN==NaN, denom max(|x|,|y|,1)),
# hand-rolled JSONL writer, incremental append+flush output.
# ---------------------------------------------------------------------------

# ------------------------------------------------------------------ utilities

_errstr(e) = first(sprint(showerror, e), 400)

"Run stage `f`; classify errors. Returns Dict(:status, :value, [:error])."
function stage(f)
    try
        return Dict{Symbol,Any}(:status => :ok, :value => f())
    catch e
        return Dict{Symbol,Any}(:status => :error, :value => nothing,
                                :error => _errstr(e))
    end
end

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

# Minimal JSON writer (no JSON dep in the package env).
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

function make_noise_oracle_data(; N::Int=50, seed::Int=2026, type::Symbol=:prod)
    rng = Random.MersenneTwister(seed)
    x1 = randn(rng, N); x2 = randn(rng, N)
    X  = hcat(ones(N), x1, x2)
    v  = 0.3 .* randn(rng, N)
    u  = 0.5 .* abs.(randn(rng, N))
    y  = X * [1.0, 0.5, -0.3] .+ v .+ (type === :cost ? u : -u)
    (y=y, X=X, N=N)
end

# ----------------------------------------------------------------- parameters
# One named set per θ point; keys match the backend idx field names.

const NOISE_ORACLE_THETA = Dict(
    :base => Dict(
        :beta          => [1.0, 0.5, -0.3],
        :ln_sigma_u_sq => 2 * log(0.5),   # HalfNormal σ_u = 0.5
        :ln_lambda     => log(0.5),       # Weibull λ = 0.5
        :ln_k          => log(1.5),       # Weibull k = 1.5
        :ln_sigma_v_sq => 2 * log(0.3),   # σ_v = 0.3
        :ln_nu_minus_2 => log(3.0),       # StudentT ν = 5
        :ln_b          => log(0.25),      # Laplace b = 0.25
        :rho_raw       => 0.3),
    :alt => Dict(
        :beta          => [0.8, 0.6, -0.2],
        :ln_sigma_u_sq => 2 * log(0.7),
        :ln_lambda     => log(0.7),
        :ln_k          => log(2.2),
        :ln_sigma_v_sq => 2 * log(0.45),
        :ln_nu_minus_2 => log(7.0),       # ν = 9
        :ln_b          => log(0.4),
        :rho_raw       => -0.4))

_asvec(x::Integer) = Int[x]
_asvec(x::AbstractVector) = collect(Int, x)
_asvec(x::AbstractRange) = collect(Int, x)

_flatten_idx(x::Integer) = Int[x]
_flatten_idx(x::Bool) = Int[]                 # *_hetero flags inside ineff idx
_flatten_idx(x::AbstractVector) = collect(Int, x)
_flatten_idx(x::AbstractRange) = collect(Int, x)
_flatten_idx(x::NamedTuple) =
    reduce(vcat, [_flatten_idx(v) for v in values(x)]; init=Int[])
_flatten_idx(::Nothing) = Int[]
_flatten_idx(x) = Int[]

"""
Build the raw θ vector for an assembled internal spec at the named parameter
set. Homoscedastic-only (single index per ineff/noise channel).
"""
function build_theta(ispec, pars::Dict)
    idx = ispec.idx
    nparams = maximum(_flatten_idx((beta=idx.beta, noise=idx.noise,
                                    ineff=idx.ineff, copula=idx.copula)))
    θ = zeros(nparams)
    bidx = _asvec(idx.beta)
    length(bidx) == length(pars[:beta]) ||
        error("beta length mismatch: idx has $(length(bidx))")
    for (j, b) in enumerate(bidx)
        θ[b] = pars[:beta][j]
    end
    for (k, inds) in pairs(idx.ineff)
        inds isa Bool && continue
        v = k === :sigma_sq ? pars[:ln_sigma_u_sq] :
            k === :lambda   ? pars[:ln_lambda] :
            k === :k        ? pars[:ln_k] :
            error("unhandled ineff idx field $k (extend NOISE_ORACLE_THETA mapping)")
        ii = _asvec(inds)
        length(ii) == 1 || error("hetero ineff channel $k not supported by the noise oracle")
        θ[ii[1]] = v
    end
    for (k, ind) in pairs(idx.noise)
        v = k === :ln_sigma_v_sq ? pars[:ln_sigma_v_sq] :
            k === :ln_nu_minus_2 ? pars[:ln_nu_minus_2] :
            k === :ln_b          ? pars[:ln_b] :
            error("unhandled noise idx field $k")
        θ[first(_asvec(ind))] = v
    end
    cinds = _flatten_idx(idx.copula)
    if !isempty(cinds)
        length(cinds) == 1 || error("multi-parameter copula not supported by the noise oracle")
        θ[cinds[1]] = pars[:rho_raw]
    end
    return θ
end

# ------------------------------------------------------------------- oracles

"Noise density closure. `smoothed=true` reproduces the Laplace sqrt(z²+1e-16) kernel of both backends."
function oracle_fv(noise::Symbol, pars::Dict; smoothed::Bool=false)
    if noise === :Normal
        σ = exp(0.5 * pars[:ln_sigma_v_sq])
        return z -> pdf(Normal(0.0, σ), z)
    elseif noise === :StudentT
        σ = exp(0.5 * pars[:ln_sigma_v_sq])
        ν = exp(pars[:ln_nu_minus_2]) + 2
        td = TDist(ν)
        return z -> pdf(td, z / σ) / σ
    elseif noise === :Laplace
        b = exp(pars[:ln_b])
        if smoothed
            return z -> exp(-sqrt(z^2 + 1e-16) / b) / (2b)
        else
            return z -> pdf(Laplace(0.0, b), z)
        end
    else
        error("unknown noise $noise")
    end
end

"Noise CDF closure (for copula oracle)."
function oracle_Fv(noise::Symbol, pars::Dict)
    if noise === :Normal
        σ = exp(0.5 * pars[:ln_sigma_v_sq])
        return z -> cdf(Normal(0.0, σ), z)
    elseif noise === :StudentT
        σ = exp(0.5 * pars[:ln_sigma_v_sq])
        ν = exp(pars[:ln_nu_minus_2]) + 2
        td = TDist(ν)
        return z -> cdf(td, z / σ)
    elseif noise === :Laplace
        b = exp(pars[:ln_b])
        return z -> cdf(Laplace(0.0, b), z)
    else
        error("unknown noise $noise")
    end
end

"Inefficiency density/CDF closures from Distributions.jl."
function oracle_fu(ineff::Symbol, pars::Dict)
    if ineff === :HalfNormal
        σu = exp(0.5 * pars[:ln_sigma_u_sq])
        d = truncated(Normal(0.0, σu), 0.0, Inf)
        return (u -> pdf(d, u)), (u -> cdf(d, u))
    elseif ineff === :Weibull
        λ = exp(pars[:ln_lambda]); k = exp(pars[:ln_k])
        d = Weibull(k, λ)
        return (u -> pdf(d, u)), (u -> cdf(d, u))
    else
        error("unhandled ineff $ineff")
    end
end

"Gaussian copula density c(a,b;ρ) on (0,1)²."
function gauss_copula_density(ρ::Float64)
    n01 = Normal()
    return function (a, b)
        (a <= 0 || a >= 1 || b <= 0 || b >= 1) && return 1.0   # degenerate tail guard
        x = quantile(n01, a); y = quantile(n01, b)
        return exp((2ρ * x * y - ρ^2 * (x^2 + y^2)) / (2 * (1 - ρ^2))) / sqrt(1 - ρ^2)
    end
end

"""
Quadrature of the convolution density and conditional moments at one ε.
Returns (den, jlms, bc): den = ∫ w(u) du, jlms = ∫ u·w / den, bc = ∫ e⁻ᵘ·w / den,
with w(u) = f_v(ε + s·u) f_u(u) [· c(F_v(ε+s·u), F_u(u))].
"""
function quad_moments(ε::Float64, s::Int, fv, fu; cop=nothing, Fv=nothing, Fu=nothing,
                      rtol=1e-11)
    w = if cop === nothing
        u -> fv(ε + s * u) * fu(u)
    else
        u -> fv(ε + s * u) * fu(u) * cop(Fv(ε + s * u), Fu(u))
    end
    den  = quadgk(w, 0.0, Inf; rtol=rtol, atol=1e-300)[1]
    num1 = quadgk(u -> u * w(u), 0.0, Inf; rtol=rtol, atol=1e-300)[1]
    num2 = quadgk(u -> exp(-u) * w(u), 0.0, Inf; rtol=rtol, atol=1e-300)[1]
    return (den=den, jlms=num1 / den, bc=num2 / den)
end

# ------------------------------------------------------------ case definition

Base.@kwdef struct NoiseOracleCase
    name::String
    backend::Symbol                # :MCI | :MSLE
    noise::Symbol                  # :Normal (control) | :StudentT | :Laplace
    ineff::Symbol  = :HalfNormal   # :HalfNormal | :Weibull
    type::Symbol   = :prod
    copula::Symbol = :None
end

_backend_mod(case::NoiseOracleCase) =
    case.backend === :MCI ? SFrontiers.MCI_Backend : SFrontiers.MSLE_Backend

function assemble_ispec(case::NoiseOracleCase, dat, n_draws::Int)
    kw = Dict{Symbol,Any}(:noise => case.noise, :ineff => case.ineff, :type => case.type)
    case.copula != :None && (kw[:copula] = case.copula)
    uspec = sfmodel_spec(; depvar=dat.y, frontier=dat.X, kw...)
    meth  = sfmodel_method(method=case.backend, n_draws=n_draws,
                           multiRand=false, chunks=1)
    B = _backend_mod(case)
    bspec = case.backend === :MCI ? uspec.mci_spec : uspec.msle_spec
    asm   = case.backend === :MCI ? B._assemble_MCI_spec : B._assemble_MSLE_spec
    return asm(bspec, meth)
end

eval_nll(case::NoiseOracleCase, ispec, θ) =
    case.backend === :MCI ? _backend_mod(case).MCI_nll(ispec, θ; chunks=1) :
                            _backend_mod(case).qmc_nll(ispec, θ; chunks=1)

eval_jlms(case::NoiseOracleCase, ispec, θ) =
    _backend_mod(case).jlms_bc_indices(ispec, θ; chunks=1)

# ------------------------------------------------------------------ case run

const NOISE_ORACLE_DRAWS = [2^10, 2^12, 2^14]
const N_JLMS_OBS = 6

"""
Execute one model-level case; returns a Dict ready for JSONL. Never throws.
Compares backend NLL (per draw count) / jlms / bc / per-obs likelihood at
fixed θ against QuadGK oracles.
"""
function run_noise_oracle_case(case::NoiseOracleCase)
    t0  = time()
    rec = Dict{Symbol,Any}(
        :kind => :model, :case => case.name, :backend => case.backend,
        :noise => case.noise, :ineff => case.ineff, :type => case.type,
        :copula => case.copula, :checks => Dict{Symbol,Any}())
    checks = rec[:checks]

    dat = make_noise_oracle_data(type=case.type)
    s   = case.type === :cost ? -1 : 1        # composite error z = ε + s·u

    s_asm = stage(() -> [assemble_ispec(case, dat, D) for D in NOISE_ORACLE_DRAWS])
    if s_asm[:status] != :ok
        checks[:assemble] = s_asm
        rec[:elapsed_s] = round(time() - t0, digits=1)
        return rec
    end
    ispecs = s_asm[:value]

    θlabels = case.copula === :None ? (:base, :alt) : (:base,)
    for θlabel in θlabels
        pars = NOISE_ORACLE_THETA[θlabel]
        s_θ = stage(() -> build_theta(ispecs[1], pars))
        if s_θ[:status] != :ok
            checks[Symbol("theta_$θlabel")] = s_θ
            continue
        end
        θ = s_θ[:value]
        ε = dat.y .- dat.X * pars[:beta]

        # --- oracle quantities -------------------------------------------
        smoothed = case.noise === :Laplace
        fv  = oracle_fv(case.noise, pars; smoothed=smoothed)
        fve = oracle_fv(case.noise, pars; smoothed=false)
        fu, Fu = oracle_fu(case.ineff, pars)
        cop = nothing; Fv = nothing
        if case.copula !== :None
            B = _backend_mod(case)
            cv = B.get_copula_vals(ispecs[1].model.copula, θ, ispecs[1].idx,
                                   ispecs[1].constants)
            ρ = Float64(first(values(cv)))
            rec[:copula_rho] = ρ
            cop = gauss_copula_density(ρ)
            Fv  = oracle_Fv(case.noise, pars)
        end

        s_quad = stage() do
            mom = [quad_moments(ε[i], s, fv, fu; cop=cop, Fv=Fv, Fu=Fu)
                   for i in 1:dat.N]
            nll_sm = -sum(log(m.den) for m in mom)
            nll_ex = if smoothed   # also record the exact-|z| oracle
                mome = [quad_moments(ε[i], s, fve, fu; cop=cop, Fv=Fv, Fu=Fu)
                        for i in 1:dat.N]
                -sum(log(m.den) for m in mome)
            else
                nll_sm
            end
            (mom=mom, nll=nll_sm, nll_exact=nll_ex)
        end
        if s_quad[:status] != :ok
            checks[Symbol("quad_$θlabel")] = Dict(:status => :error,
                                                  :error => s_quad[:error])
            continue
        end
        quad = s_quad[:value]

        # --- backend NLL per draw count, convergence trend ----------------
        d = Dict{Symbol,Any}(:quad_nll => quad.nll)
        if smoothed
            d[:quad_nll_exact_abs]    = quad.nll_exact
            d[:smoothing_gap_perobs]  = abs(quad.nll_exact - quad.nll) / dat.N
        end
        diffs = Float64[]
        for (k, D) in enumerate(NOISE_ORACLE_DRAWS)
            s_nll = stage(() -> eval_nll(case, ispecs[k], θ))
            if s_nll[:status] != :ok
                d[Symbol("nll_error_$D")] = s_nll[:error]
                push!(diffs, NaN)
                continue
            end
            nb = Float64(s_nll[:value])
            d[Symbol("nll_$D")] = nb
            push!(diffs, abs(nb - quad.nll) / dat.N)
            d[Symbol("dnll_perobs_$D")] = diffs[end]
        end
        final = diffs[end]
        trend_ok = !isnan(final) && !isnan(diffs[1]) &&
                   (final <= 1.5 * diffs[1] || final <= 1e-6)
        d[:trend_ok] = trend_ok
        d[:status] = (!isnan(final) && final <= 1e-4 && trend_ok) ? :pass : :attention
        checks[Symbol("nll_$θlabel")] = d

        # --- jlms / bc / per-obs likelihood at max draws (base θ only) ----
        if θlabel === :base
            s_j = stage(() -> eval_jlms(case, ispecs[end], θ))
            jd = Dict{Symbol,Any}()
            if s_j[:status] != :ok
                jd[:status] = :error; jd[:error] = s_j[:error]
            else
                r = s_j[:value]
                idxs = 1:N_JLMS_OBS
                jq = [quad.mom[i].jlms for i in idxs]
                bq = [quad.mom[i].bc   for i in idxs]
                lq = [quad.mom[i].den  for i in idxs]
                jd[:jlms_backend] = collect(Float64, r.jlms[idxs])
                jd[:jlms_quad]    = jq
                jd[:djlms] = maxreldiff(r.jlms[idxs], jq)
                jd[:bc_backend]   = collect(Float64, r.bc[idxs])
                jd[:bc_quad]      = bq
                jd[:dbc]   = maxreldiff(r.bc[idxs], bq)
                if hasproperty(r, :likelihood) && r.likelihood !== nothing
                    jd[:dlik] = maxreldiff(r.likelihood[idxs], lq)
                end
                jd[:status] = (something(jd[:djlms], Inf) <= 2e-3 &&
                               something(jd[:dbc],   Inf) <= 2e-3) ? :pass : :attention
            end
            checks[:jlms_bc] = jd
        end
    end

    rec[:elapsed_s] = round(time() - t0, digits=1)
    return rec
end

# ------------------------------------------------------- kernel-level checks

"""
Kernel spot checks vs Distributions.jl to machine precision, incl. tails.
Exercises BOTH backend modules' myStudentTPDF(!), myLaplacePDF(!),
log_noise_pdf!, noise_cdf!, and the get_noise_vals θ→parameter mapping.
Returns a vector of JSONL-ready Dicts.
"""
function run_kernel_checks()
    recs = Dict{Symbol,Any}[]
    for (bname, B, mkT, mkL) in (
            (:MCI,  SFrontiers.MCI_Backend,
             SFrontiers.MCI_Backend.StudentTNoise_MCI(),
             SFrontiers.MCI_Backend.LaplaceNoise_MCI()),
            (:MSLE, SFrontiers.MSLE_Backend,
             SFrontiers.MSLE_Backend.StudentTNoise_MSLE(),
             SFrontiers.MSLE_Backend.LaplaceNoise_MSLE()))

        model_t = B._build_model(:StudentT, :HalfNormal)
        model_l = B._build_model(:Laplace,  :HalfNormal)
        c_t = B.make_constants(model_t, Float64)
        c_l = B.make_constants(model_l, Float64)

        # ---------------- StudentT ---------------------------------------
        for (σv, ν) in ((0.3, 5.0), (1.0, 2.1), (0.05, 30.0), (2.0, 2.05))
            rec = Dict{Symbol,Any}(:kind => :kernel, :backend => bname,
                :case => "kernel-$bname-StudentT-sv$(σv)-nu$(ν)",
                :noise => :StudentT, :sigma_v => σv, :nu => ν,
                :checks => Dict{Symbol,Any}())
            ch = rec[:checks]
            z = vcat([σv * m for m in
                      (0.0, 1e-9, 1e-3, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0)],
                     [-σv * m for m in (1e-9, 0.1, 1.0, 5.0, 30.0, 100.0)])
            td = TDist(ν)
            ref_log = [logpdf(td, zi / σv) - log(σv) for zi in z]
            ref_pdf = exp.(ref_log)

            s1 = stage(() -> B.myStudentTPDF(z=z, σ_v=σv, ν=ν))
            ch[:pdf_slow] = s1[:status] == :ok ?
                Dict(:diff => maxreldiff(s1[:value], ref_pdf),
                     :status => maxreldiff(s1[:value], ref_pdf) <= 1e-11 ? :pass : :attention) :
                Dict(:status => :error, :error => s1[:error])

            # fast path through get_noise_vals mapping (θ → σ_v, ν, log_const)
            p   = [2 * log(σv), log(ν - 2)]
            idx = (noise=(ln_sigma_v_sq=1, ln_nu_minus_2=2),)
            s2 = stage() do
                nv = B.get_noise_vals(mkT, p, idx, c_t)
                out = similar(z)
                B.myStudentTPDF!(out; z=z, log_const=nv.log_const,
                                 inv_σ_v=nv.inv_sigma_v,
                                 half_nu_plus_one=nv.half_nu_plus_one)
                (nv=nv, out=out)
            end
            if s2[:status] == :ok
                nv = s2[:value].nv
                dmap = max(abs(nv.sigma_v - σv) / σv, abs(nv.nu - ν) / ν)
                dpdf = maxreldiff(s2[:value].out, ref_pdf)
                ch[:pdf_fast_via_noise_vals] = Dict(:diff => dpdf, :map_diff => dmap,
                    :status => (dpdf <= 1e-11 && dmap <= 1e-12) ? :pass : :attention)
                s3 = stage() do
                    out = similar(z)
                    B.log_noise_pdf!(mkT, out, z, nv, c_t)
                    out
                end
                if s3[:status] == :ok
                    dlog = maximum(abs.(s3[:value] .- ref_log) ./
                                   max.(abs.(ref_log), 1.0))
                    ch[:logpdf] = Dict(:diff => dlog,
                                       :status => dlog <= 1e-11 ? :pass : :attention)
                else
                    ch[:logpdf] = Dict(:status => :error, :error => s3[:error])
                end
            else
                ch[:pdf_fast_via_noise_vals] = Dict(:status => :error, :error => s2[:error])
            end
            push!(recs, rec)
        end

        # ---------------- Laplace ----------------------------------------
        for b in (0.25, 1.0, 0.01)
            rec = Dict{Symbol,Any}(:kind => :kernel, :backend => bname,
                :case => "kernel-$bname-Laplace-b$(b)",
                :noise => :Laplace, :b => b, :checks => Dict{Symbol,Any}())
            ch = rec[:checks]
            z = vcat([b * m for m in
                      (0.0, 1e-9, 1e-6, 1e-3, 0.1, 1.0, 5.0, 30.0, 100.0)],
                     [-b * m for m in (1e-9, 1e-3, 1.0, 30.0, 100.0)])
            ld = Laplace(0.0, b)
            ref_pdf = pdf.(ld, z)
            ref_log = logpdf.(ld, z)
            ref_cdf = cdf.(ld, z)

            p   = [log(b)]
            idx = (noise=(ln_b=1,),)
            s2 = stage() do
                nv = B.get_noise_vals(mkL, p, idx, c_l)
                pdf_out = similar(z)
                B.myLaplacePDF!(pdf_out; z=z, inv_b=nv.inv_b, inv_2b=nv.inv_2b)
                log_out = similar(z)
                B.log_noise_pdf!(mkL, log_out, z, nv, c_l)
                cdf_out = similar(z)
                B.noise_cdf!(mkL, cdf_out, z, nv, c_l)
                (nv=nv, pdf=pdf_out, lg=log_out, cd=cdf_out)
            end
            if s2[:status] == :ok
                v = s2[:value]
                dmap = abs(v.nv.b - b) / b
                # away from 0: machine precision expected. Near 0: the smoothing
                # bound |Δlogf| ≤ 1e-8/b of both backends applies (recorded separately).
                far = abs.(z) .> 1e-6 * b .+ 1e-12
                dpdf_far = maxreldiff(v.pdf[far], ref_pdf[far])
                dlog_near = maximum(abs.(v.lg .- ref_log))
                dpdf_all = maxreldiff(v.pdf, ref_pdf)
                dcdf = maxreldiff(v.cd, ref_cdf)
                smooth_bound = 1.0e-8 / b + 1e-11
                ch[:map] = Dict(:diff => dmap,
                                :status => dmap <= 1e-12 ? :pass : :attention)
                ch[:pdf] = Dict(:diff_far => dpdf_far, :diff_all => dpdf_all,
                    :smooth_bound => smooth_bound,
                    :status => (dpdf_far <= smooth_bound &&
                                dpdf_all <= smooth_bound) ? :pass : :attention)
                ch[:logpdf] = Dict(:absdiff => dlog_near, :smooth_bound => smooth_bound,
                    :status => dlog_near <= smooth_bound ? :pass : :attention)
                ch[:cdf] = Dict(:diff => dcdf,
                                :status => dcdf <= 1e-12 ? :pass : :attention)
            else
                ch[:kernel_eval] = Dict(:status => :error, :error => s2[:error])
            end
            push!(recs, rec)
        end
    end
    return recs
end

# ------------------------------------------------------------------ case list

function build_noise_oracle_cases()
    cases = NoiseOracleCase[]
    for backend in (:MCI, :MSLE), noise in (:Normal, :StudentT, :Laplace),
        ineff in (:HalfNormal, :Weibull), type in (:prod, :cost)
        push!(cases, NoiseOracleCase(name="$backend-$noise-$ineff-$type",
                                     backend=backend, noise=noise, ineff=ineff, type=type))
    end
    # Laplace noise + Gaussian copula (exercises noise_cdf! Laplace in a full NLL)
    for backend in (:MCI, :MSLE)
        push!(cases, NoiseOracleCase(name="$backend-Laplace-HalfNormal-prod-gausscop",
                                     backend=backend, noise=:Laplace, ineff=:HalfNormal,
                                     type=:prod, copula=:Gaussian))
    end
    return cases
end

const NOISE_ORACLE_SMOKE = ["MCI-Normal-HalfNormal-prod",       # control: oracle wiring
                            "MCI-StudentT-HalfNormal-prod",
                            "MSLE-Laplace-HalfNormal-prod"]
