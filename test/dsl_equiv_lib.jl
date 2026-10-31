# ---------------------------------------------------------------------------
# DSL-equivalence library.
#
# DSL-vs-keyword spec equivalence: build every spec BOTH ways — DSL macros
# (@useData/@depvar/@frontier/@zvar/@id) from a DataFrame vs the keyword API
# from matrices with matching column order — and require:
#   level 1 (all cases)    — bit-identical spec arrays (depvar / frontier /
#                            zvar / scaling zvar / panel id) on every backend
#                            spec the unified layer built (MCI/MSLE/MLE/panel).
#   level 2 (fit subset)   — bit-identical fits (== on coeff, loglikelihood,
#                            jlms, bc; flag parity) with identical solver opts
#                            (Float64/CPU determinism makes == achievable).
#   level 3 (predict)      — bit-identical predict(:jlms) on both results.
# Plus a check that the DSL path with bare-Symbol hetero errors with the same
# clean message as the keyword path (both messages are recorded verbatim).
#
# Requires noise_oracle_lib.jl (stage/json/maxreldiff) include()d first,
# plus DataFrames and Optim in scope.
# ---------------------------------------------------------------------------

# ------------------------------------------------------------ data generation

function make_dsl_equiv_data(; N::Int=60, seed::Int=2028)
    rng = Random.MersenneTwister(seed)
    x1 = randn(rng, N); x2 = randn(rng, N)
    x3 = rand(rng, 1:5, N)                     # Int column (conversion path)
    z1 = abs.(randn(rng, N)) .+ 0.5
    z2 = abs.(randn(rng, N)) .+ 0.5
    v  = 0.3 .* randn(rng, N)
    u  = 0.5 .* abs.(randn(rng, N))
    y  = 1.0 .+ 0.5 .* x1 .- 0.3 .* x2 .+ v .- u
    DataFrame(y=y, cons=ones(N), x1=x1, x2=x2, x3=x3, z1=z1, z2=z2,
              unused1=randn(rng, N), unused2=randn(rng, N))
end

function make_dsl_equiv_panel(; F::Int=30, Tp::Int=4, seed::Int=2029)
    rng  = Random.MersenneTwister(seed)
    n    = F * Tp
    firm = repeat(1:F, inner=Tp)
    x1 = randn(rng, n); x2 = randn(rng, n)
    z1 = randn(rng, n)                          # within-firm varying
    α  = repeat(randn(rng, F), inner=Tp)
    v  = 0.25 .* randn(rng, n)
    u  = 0.4 .* abs.(randn(rng, n))
    y  = 0.5 .* x1 .- 0.3 .* x2 .+ α .+ v .- u
    DataFrame(y=y, x1=x1, x2=x2, z1=z1, firm=firm)
end

_mat(df, names...) = Matrix{Float64}(df[!, collect(names)])
_vec(df, name)     = Vector{Float64}(df[!, name])

# ------------------------------------------------------------ case definition

struct DslEquivCase
    name::String
    dsl_build::Function          # () -> UnifiedSpec (DSL path)
    kw_build::Function           # () -> UnifiedSpec (keyword path, matrices)
    fit_methods::Vector{Symbol}  # subset of [:MCI, :MSLE, :MLE] to fit both ways
    do_predict::Bool
end
DslEquivCase(name, d, k; fit=Symbol[], predict=false) = DslEquivCase(name, d, k, fit, predict)

# ---------------------------------------------------------------- comparison

const _SPEC_ARRAY_FIELDS = (:depvar, :frontier, :zvar, :scaling_zvar, :id,
                            :offsets, :T_max)

"Compare the array payloads of two backend specs bitwise."
function _cmp_backend_spec(a, b)
    a === nothing && b === nothing && return Dict(:status => :skipped)
    (a === nothing) != (b === nothing) &&
        return Dict(:status => :fail,
                    :error => "one path built this backend spec, the other did not " *
                              "(dsl=$(a !== nothing), kw=$(b !== nothing))")
    d = Dict{Symbol,Any}(); nfields = 0; bad = String[]
    for f in _SPEC_ARRAY_FIELDS
        (hasproperty(a, f) && hasproperty(b, f)) || continue
        va, vb = getproperty(a, f), getproperty(b, f)
        (va === nothing && vb === nothing) && continue
        nfields += 1
        if !isequal(va, vb)
            push!(bad, string(f))
            d[Symbol("maxreldiff_$f")] = (va isa AbstractArray && vb isa AbstractArray) ?
                maxreldiff(va, vb) : "type/shape mismatch"
        end
    end
    d[:fields_compared] = nfields
    d[:status] = nfields == 0 ? :skipped : (isempty(bad) ? :pass : :fail)
    isempty(bad) || (d[:mismatched] = join(bad, ","))
    return d
end

function compare_spec_arrays(us_dsl, us_kw)
    out = Dict{Symbol,Any}()
    for (label, fa) in ((:mci, :mci_spec), (:msle, :msle_spec),
                        (:mle, :mle_spec), (:panel, :panel_spec))
        a = hasproperty(us_dsl, fa) ? getproperty(us_dsl, fa) : nothing
        b = hasproperty(us_kw, fa)  ? getproperty(us_kw, fa)  : nothing
        out[label] = _cmp_backend_spec(a, b)
    end
    return out
end

# ---------------------------------------------------------------------- fits

function dsl_equiv_opt()
    sfmodel_opt(warmstart_solver = NelderMead(),
                warmstart_opt    = (iterations = 100, g_abstol = 1e-4),
                main_solver      = Newton(),
                main_opt         = (iterations = 200, g_abstol = 1e-6,
                                    show_trace = false))
end

function dsl_equiv_fit(uspec, method_sym)
    meth = method_sym === :MLE ? sfmodel_method(method=:MLE) :
           sfmodel_method(method=method_sym, n_draws=100, multiRand=true, chunks=1)
    sfmodel_fit(spec=uspec, method=meth, init=nothing, optim_options=dsl_equiv_opt(),
                marginal=false, show_table=false, verbose=false)
end

"Fit both spec styles with one method and require bit-identical results."
function fit_and_compare(us_dsl, us_kw, method_sym)
    d = Dict{Symbol,Any}()
    sa = stage(() -> dsl_equiv_fit(us_dsl, method_sym))
    sb = stage(() -> dsl_equiv_fit(us_kw,  method_sym))
    if sa[:status] != :ok || sb[:status] != :ok
        d[:status] = :error
        sa[:status] != :ok && (d[:error_dsl] = sa[:error])
        sb[:status] != :ok && (d[:error_kw]  = sb[:error])
        return d, nothing, nothing
    end
    ra, rb = sa[:value], sb[:value]
    bad = String[]
    for f in (:coeff, :loglikelihood, :jlms, :bc, :converged, :redflag)
        (hasproperty(ra, f) && hasproperty(rb, f)) || continue
        va, vb = getproperty(ra, f), getproperty(rb, f)
        if !isequal(va, vb)
            push!(bad, string(f))
            (va isa Union{AbstractArray,Number}) && (va !== nothing) &&
                (d[Symbol("maxreldiff_$f")] = maxreldiff(
                    va isa Number ? [va] : va, vb isa Number ? [vb] : vb))
        end
    end
    d[:status] = isempty(bad) ? :pass : :fail
    isempty(bad) || (d[:mismatched] = join(bad, ","))
    return d, ra, rb
end

"predict(:jlms) on both results (estimation data, matrix form) must be ==."
function predict_and_compare(ra, rb, X, y)
    s1 = stage(() -> predict(ra; frontier=X, depvar=y, what=:jlms))
    s2 = stage(() -> predict(rb; frontier=X, depvar=y, what=:jlms))
    d = Dict{Symbol,Any}()
    if s1[:status] != :ok || s2[:status] != :ok
        d[:status] = :error
        s1[:status] != :ok && (d[:error_dsl] = s1[:error])
        s2[:status] != :ok && (d[:error_kw]  = s2[:error])
        return d
    end
    eqok = isequal(s1[:value], s2[:value])
    d[:status] = eqok ? :pass : :fail
    eqok || (d[:maxreldiff] = maxreldiff(s1[:value], s2[:value]))
    return d
end

# ------------------------------------------------------------------ case run

function run_dsl_equiv_case(case::DslEquivCase, df, dfp)
    t0  = time()
    rec = Dict{Symbol,Any}(:kind => :dsl_equiv, :case => case.name,
                           :checks => Dict{Symbol,Any}())
    checks = rec[:checks]

    s_d = stage(case.dsl_build)
    s_k = stage(case.kw_build)
    if s_d[:status] != :ok || s_k[:status] != :ok
        checks[:build] = Dict(:status => :error,
            (s_d[:status] != :ok ? (:error_dsl => s_d[:error],) : ())...,
            (s_k[:status] != :ok ? (:error_kw  => s_k[:error],) : ())...)
        rec[:elapsed_s] = round(time() - t0, digits=1)
        return rec
    end
    us_dsl, us_kw = s_d[:value], s_k[:value]

    checks[:spec_arrays] = compare_spec_arrays(us_dsl, us_kw)

    for m in case.fit_methods
        fd, ra, rb = fit_and_compare(us_dsl, us_kw, m)
        checks[Symbol("fit_$m")] = fd
        if case.do_predict && m === :MCI && ra !== nothing && rb !== nothing
            X = hasproperty(us_kw, :mci_spec) && us_kw.mci_spec !== nothing ?
                us_kw.mci_spec.frontier : nothing
            y = X === nothing ? nothing : us_kw.mci_spec.depvar
            X !== nothing &&
                (checks[:predict] = predict_and_compare(ra, rb, X, y))
        end
    end

    rec[:elapsed_s] = round(time() - t0, digits=1)
    return rec
end

# ------------------------------------------------------------------ case list

"""
Build the DSL-equivalence case list. Each case closes over the SAME DataFrame for both
paths; the keyword path extracts matrices with the intended column order, so
any silent column-order/selection bug in the DSL extraction shows up as a
spec-array or fit mismatch.
"""
function build_dsl_equiv_cases(df, dfp)
    cases = DslEquivCase[]

    # A1 — baseline HN homoscedastic; fits on all three backends + predict
    push!(cases, DslEquivCase("A1-HN-baseline",
        () -> sfmodel_spec(@useData(df), @depvar(y), @frontier(cons, x1, x2);
                           noise=:Normal, ineff=:HalfNormal),
        () -> sfmodel_spec(depvar=_vec(df, :y), frontier=_mat(df, :cons, :x1, :x2),
                           noise=:Normal, ineff=:HalfNormal);
        fit=[:MCI, :MSLE, :MLE], predict=true))

    # A2 — permuted frontier column order (macro order must be honored)
    push!(cases, DslEquivCase("A2-frontier-permuted",
        () -> sfmodel_spec(@useData(df), @depvar(y), @frontier(cons, x2, x1);
                           noise=:Normal, ineff=:HalfNormal),
        () -> sfmodel_spec(depvar=_vec(df, :y), frontier=_mat(df, :cons, :x2, :x1),
                           noise=:Normal, ineff=:HalfNormal);
        fit=[:MCI]))

    # A3 — subset selection amid unused df columns; macros out of order
    push!(cases, DslEquivCase("A3-subset-unused-cols",
        () -> sfmodel_spec(@frontier(cons, x1), @useData(df), @depvar(y);
                           noise=:Normal, ineff=:HalfNormal),
        () -> sfmodel_spec(depvar=_vec(df, :y), frontier=_mat(df, :cons, :x1),
                           noise=:Normal, ineff=:HalfNormal)))

    # A4 — Int column in the frontier (Float64 conversion path)
    push!(cases, DslEquivCase("A4-int-column",
        () -> sfmodel_spec(@useData(df), @depvar(y), @frontier(cons, x1, x3);
                           noise=:Normal, ineff=:HalfNormal),
        () -> sfmodel_spec(depvar=_vec(df, :y), frontier=_mat(df, :cons, :x1, :x3),
                           noise=:Normal, ineff=:HalfNormal)))

    # A5 — hetero Vector{Symbol} with zvar (_cons first)
    push!(cases, DslEquivCase("A5-hetero-zvar",
        () -> sfmodel_spec(@useData(df), @depvar(y), @frontier(cons, x1, x2),
                           @zvar(cons, z1);
                           noise=:Normal, ineff=:HalfNormal, hetero=[:sigma_u_sq]),
        () -> sfmodel_spec(depvar=_vec(df, :y), frontier=_mat(df, :cons, :x1, :x2),
                           zvar=_mat(df, :cons, :z1),
                           noise=:Normal, ineff=:HalfNormal, hetero=[:sigma_u_sq]);
        fit=[:MCI]))

    # A6 — permuted zvar column order (3 columns)
    push!(cases, DslEquivCase("A6-zvar-permuted",
        () -> sfmodel_spec(@useData(df), @depvar(y), @frontier(cons, x1, x2),
                           @zvar(cons, z2, z1);
                           noise=:Normal, ineff=:HalfNormal, hetero=[:sigma_u_sq]),
        () -> sfmodel_spec(depvar=_vec(df, :y), frontier=_mat(df, :cons, :x1, :x2),
                           zvar=_mat(df, :cons, :z2, :z1),
                           noise=:Normal, ineff=:HalfNormal, hetero=[:sigma_u_sq])))

    # A7 — TN with hetero [:mu, :sigma_u_sq]; MSLE fit
    push!(cases, DslEquivCase("A7-TN-hetero-mu-sigma",
        () -> sfmodel_spec(@useData(df), @depvar(y), @frontier(cons, x1, x2),
                           @zvar(cons, z1);
                           noise=:Normal, ineff=:TruncatedNormal,
                           hetero=[:mu, :sigma_u_sq]),
        () -> sfmodel_spec(depvar=_vec(df, :y), frontier=_mat(df, :cons, :x1, :x2),
                           zvar=_mat(df, :cons, :z1),
                           noise=:Normal, ineff=:TruncatedNormal,
                           hetero=[:mu, :sigma_u_sq]);
        fit=[:MSLE]))

    # A8 — scaling property (zvar with NO constant column)
    push!(cases, DslEquivCase("A8-TN-scaling",
        () -> sfmodel_spec(@useData(df), @depvar(y), @frontier(cons, x1, x2),
                           @zvar(z1, z2);
                           noise=:Normal, ineff=:TruncatedNormal, hetero=:scaling),
        () -> sfmodel_spec(depvar=_vec(df, :y), frontier=_mat(df, :cons, :x1, :x2),
                           zvar=_mat(df, :z1, :z2),
                           noise=:Normal, ineff=:TruncatedNormal, hetero=:scaling);
        fit=[:MCI]))

    # A9 — Gaussian copula
    push!(cases, DslEquivCase("A9-copula",
        () -> sfmodel_spec(@useData(df), @depvar(y), @frontier(cons, x1, x2);
                           noise=:Normal, ineff=:HalfNormal, copula=:Gaussian),
        () -> sfmodel_spec(depvar=_vec(df, :y), frontier=_mat(df, :cons, :x1, :x2),
                           noise=:Normal, ineff=:HalfNormal, copula=:Gaussian);
        fit=[:MCI]))

    # A10 — name collisions with canonical auto-names:
    # df columns literally named _cons/x1/x2/z1, with _cons NOT all-ones
    push!(cases, DslEquivCase("A10-canonical-name-collision",
        () -> begin
            dfc = DataFrame(y=df.y, _cons=df.z2,   # non-constant column named _cons!
                            x1=df.x2, x2=df.x1, z1=df.z1)
            sfmodel_spec(@useData(dfc), @depvar(y), @frontier(x1, _cons, x2);
                         noise=:Normal, ineff=:HalfNormal)
        end,
        () -> sfmodel_spec(depvar=_vec(df, :y),
                           frontier=hcat(_vec(df, :x2), _vec(df, :z2), _vec(df, :x1)),
                           noise=:Normal, ineff=:HalfNormal);
        fit=[:MCI]))

    # A12 — panel_TFE with @id (frontier has NO intercept; zvar within-varying)
    push!(cases, DslEquivCase("A12-panel-TFE",
        () -> sfmodel_spec(@useData(dfp), @depvar(y), @frontier(x1, x2),
                           @zvar(z1), @id(firm);
                           noise=:Normal, ineff=:HalfNormal, datatype=:panel_TFE),
        () -> sfmodel_spec(depvar=_vec(dfp, :y), frontier=_mat(dfp, :x1, :x2),
                           zvar=_vec(dfp, :z1), id=Vector(dfp.firm),
                           noise=:Normal, ineff=:HalfNormal, datatype=:panel_TFE);
        fit=[:MCI]))

    return cases
end

"""
Bare-Symbol hetero must produce the SAME clean error on the DSL path as the
keyword path. Records both error texts verbatim.
"""
function run_dsl_error_parity_case(df)
    rec = Dict{Symbol,Any}(:kind => :dsl_equiv, :case => "A14-bare-symbol-hetero",
                           :checks => Dict{Symbol,Any}())
    s_kw = stage(() -> sfmodel_spec(depvar=_vec(df, :y),
                                    frontier=_mat(df, :cons, :x1, :x2),
                                    zvar=_mat(df, :cons, :z1),
                                    noise=:Normal, ineff=:HalfNormal,
                                    hetero=:sigma_u_sq))
    s_dsl = stage(() -> sfmodel_spec(@useData(df), @depvar(y),
                                     @frontier(cons, x1, x2), @zvar(cons, z1);
                                     noise=:Normal, ineff=:HalfNormal,
                                     hetero=:sigma_u_sq))
    d = Dict{Symbol,Any}(
        :kw_errored   => s_kw[:status]  == :error,
        :dsl_errored  => s_dsl[:status] == :error,
        :kw_error     => get(s_kw,  :error, nothing),
        :dsl_error    => get(s_dsl, :error, nothing))
    # both must error; DSL message should be the clean guard, not a MethodError
    clean(e) = e !== nothing && occursin("hetero", e) && !occursin("MethodError", e)
    d[:status] = (d[:kw_errored] && d[:dsl_errored] &&
                  clean(d[:kw_error]) && clean(d[:dsl_error])) ? :pass : :attention
    rec[:checks][:error_parity] = d
    return rec
end
