# Regression tests: an error in the final Hessian or re-evaluation keeps the fit.
#
# The final ForwardDiff Hessian of every fit (MLE sfmodel_fit, sfmodel_MCI_fit, sfmodel_MSLE_fit,
# sfmodel_panel_fit) and SFRedflag.sf_final_fg's evaluation again at the estimate ran outside any
# try block: an error there ended sfmodel_fit and the estimate was lost. In
# MLE_Backend.sfmodel_boot_marginal an error in a replicate's Hessian or get_marg ended the
# bootstrap, and the replicates already done were lost. Now:
#   - the Hessian is computed by SFRedflag.sf_classify_hessian(f, x): an error there gives
#     redflag 1, an all-NaN Hessian and var-cov matrix (NaN standard errors) and a message with
#     the error text, printed whatever `verbose` is; the coefficients, log-likelihood, jlms, bc
#     and marginal effects are those of the same fit without the error;
#   - sf_final_fg: gradient_norm NaN; loglikelihood = f at the estimate when it can be
#     evaluated there, else Optim's own minimum; a message;
#   - in the bootstrap such an error is a failed resample of a new kind (hesserror, margerror),
#     drawn again and counted;
#   - SFRedflag.sf_fatal_error decides what is never swallowed: InterruptException,
#     OutOfMemoryError, StackOverflowError and CUDA's out-of-GPU-memory error (recognised by its
#     type name or its message, without loading CUDA), also wrapped by a spawned task.
# The errors are injected by a model id of the test's own that delegates to the MLE half normal
# (the error comes only from nested-dual input, i.e. the Hessian, or after a forced line-search
# failure) and by a small objective with a switch. The fits use BFGS or gradient descent: with
# Newton() as the main solver Optim computes the Hessian itself at every point it accepts, the
# estimate included, so an error on nested-dual input ends such a fit inside optimize before the
# guard is reached. The MCI, MSLE and panel fit functions are
# checked in the source: they call the same guarded function as the MLE fit. On the package
# before the fix every fit- and bootstrap-level case throws instead of returning, and the
# helpers do not exist.

using Test
using SFrontiers
using Random
using LinearAlgebra
using Optim
using ForwardDiff
using DataFrames
using ADTypes: AutoForwardDiff

const _BF39RF = SFrontiers.SFRedflag
const _BF39MB = SFrontiers.MLE_Backend
const _BF39M = _BF39MB.SFmle

# Exceptions of the test's own: one named like CUDA's (the type-name rule), one with CUDA's
# text under another name (the message rule), one whose message cannot be printed.
struct _BF39OutOfGPUMemoryError <: Exception end
Base.showerror(io::IO, ::_BF39OutOfGPUMemoryError) = print(io, "a mock of CUDA's error (test)")
struct _BF39GPUText <: Exception end
Base.showerror(io::IO, ::_BF39GPUText) = print(io, "Out of GPU memory trying to allocate 2.000 GiB (mock, test)")
struct _BF39BadShow <: Exception end
Base.showerror(io::IO, ::_BF39BadShow) = error("showerror of _BF39BadShow throws (test)")

_bf39_nested(::Type{<:ForwardDiff.Dual{T,V}}) where {T,V} = V <: ForwardDiff.Dual
_bf39_nested(::Type) = false
_bf39_dual(::Type{<:ForwardDiff.Dual}) = true
_bf39_dual(::Type) = false

# One small objective for the unit checks (one function, so ForwardDiff compiles it once):
# _BF39_FMODE[] = :smooth, :singular (a Hessian of rank 1), or :nested / :dual / :all, which
# throw _BF39_FEXC[] on nested-dual input / on any dual input / on any input.
const _BF39_FMODE = Ref(:smooth)
const _BF39_FEXC = Ref{Any}(nothing)
function _bf39_obj(x)
    m = _BF39_FMODE[]; T = eltype(x)
    ((m === :nested && _bf39_nested(T)) || (m === :dual && _bf39_dual(T)) || m === :all) && throw(_BF39_FEXC[])
    m === :singular && return (x[1] + x[2])^2
    return (x[1] - 1)^2 + 3 * (x[2] + 0.5)^2 + 0.1 * x[1]^4 + x[1] * x[2]
end
function _bf39_with(f, mode, exc = nothing)
    _BF39_FMODE[] = mode; _BF39_FEXC[] = exc
    try
        return f()
    finally
        _BF39_FMODE[] = :smooth; _BF39_FEXC[] = nothing
    end
end

# A model id of the test's own: the MLE half normal, plus an error chosen by _BF39_MODE.
#   :domain, :interrupt, :gpuoom, :gputext, :stackover  -- on nested-dual input (the Hessian)
#   :hessonce   -- a DomainError on the _BF39_AT[]-th nested-dual call only
#   :margonce   -- a DomainError on the _BF39_AT[]-th call of get_marg only
#   :armed_dual, :armed_all -- once _BF39_ARMED[] is set: on dual input / on any input
const _BF39_MODE = Ref(:off)
const _BF39_AT = Ref(0)
const _BF39_CALLS = Ref(0)
const _BF39_ARMED = Ref(false)
struct _BF39Half <: _BF39M.Sfmodeltype end
_BF39M.getvar(::Type{_BF39Half}, dat::DataFrame) = _BF39M.getvar(_BF39M.Half, dat)
_BF39M.jlmsbc(::Type{_BF39Half}, args...) = _BF39M.jlmsbc(_BF39M.Half, args...)
_BF39M._mle_scale_at_clamp(::Type{_BF39Half}, pos, coef, wvar, vvar) =
    _BF39M._mle_scale_at_clamp(_BF39M.Half, pos, coef, wvar, vvar)
function _BF39M.LL_T(::Type{_BF39Half}, Y, X, z, q, W, V, PorC, nobs, po, rho, dum2, misc)
    T = eltype(rho); m = _BF39_MODE[]
    if _bf39_nested(T)
        m === :domain    && throw(DomainError(-1.0, "injected by the test"))
        m === :interrupt && throw(InterruptException())
        m === :gpuoom    && throw(_BF39OutOfGPUMemoryError())
        m === :gputext   && throw(ErrorException("CUDA error: Out of GPU memory (mock, test)"))
        m === :stackover && throw(StackOverflowError())
        if m === :hessonce
            _BF39_CALLS[] += 1
            _BF39_CALLS[] == _BF39_AT[] && throw(DomainError(-3.0, "a replicate's Hessian, test"))
        end
    end
    if _BF39_ARMED[] && (m === :armed_all || (m === :armed_dual && _bf39_dual(T)))
        throw(DomainError(-2.0, "after the forced line-search failure, test"))
    end
    return _BF39M.LL_T(_BF39M.Half, Y, X, z, q, W, V, PorC, nobs, po, rho, dum2, misc)
end
function _BF39M.get_marg(::Type{_BF39Half}, args...)
    if _BF39_MODE[] === :margonce
        _BF39_CALLS[] += 1
        _BF39_CALLS[] == _BF39_AT[] && throw(DomainError(-4.0, "a replicate's marginal effects, test"))
    end
    return _BF39M.get_marg(_BF39M.Half, args...)
end

@testset "an error in the final Hessian or re-evaluation keeps the fit; bootstrap redraws" begin
    RF = _BF39RF
    fixed = isdefined(RF, :sf_fatal_error)
    function capture(f)                       # stdout captured; an exception is returned, not thrown
        path, io = mktemp()
        r = try
            redirect_stdout(f, io)
        catch e
            e
        end
        close(io)
        txt = read(path, String); rm(path; force = true)
        return r, txt
    end
    allnan(A) = A isa AbstractArray && !isempty(A) && all(isnan, A)
    tf(ex) = try fetch(Threads.@spawn throw(ex)) catch e; e end      # an error as a spawned task delivers it

    @testset "sf_fatal_error: what is never swallowed" begin
        if fixed
            F = RF.sf_fatal_error
            @test F(InterruptException()) && F(OutOfMemoryError()) && F(StackOverflowError())
            @test F(_BF39OutOfGPUMemoryError())                         # type name
            @test F(_BF39GPUText())                                     # message (showerror)
            @test F(ErrorException("CUDA error: OUT OF GPU MEMORY"))    # message (msg field), any case
            @test F(tf(InterruptException())) && F(tf(_BF39OutOfGPUMemoryError()))
            @test F(CompositeException([tf(DomainError(-1.0)), tf(_BF39GPUText())]))
            @test F(CapturedException(InterruptException(), backtrace()))
            @test !F(DomainError(-1.0)) && !F(ArgumentError("x")) && !F(LinearAlgebra.SingularException(2))
            @test !F(tf(DomainError(-1.0))) && !F(CompositeException([tf(DomainError(-1.0))]))
            @test !F(_BF39BadShow()) && !F(ErrorException("out of memory"))
            @test !F(MethodError(+, (1, "a")))
            # the MLE bootstrap applies the same rule
            @test _BF39M._boot_fatal(_BF39OutOfGPUMemoryError()) && _BF39M._boot_fatal(tf(InterruptException())) &&
                  !_BF39M._boot_fatal(DomainError(-1.0))
        else
            @test fixed
        end
    end

    @testset "sf_classify_hessian(f, x): the Hessian with the guard" begin
        f = _bf39_obj
        x0 = [0.7, -0.2]
        if fixed
            H = ForwardDiff.hessian(f, x0)
            r = RF.sf_classify_hessian(f, x0; message = false)
            @test isequal(r.hessian, H) && isequal(r.var_cov_matrix, RF.sf_classify_hessian(H; message = false).var_cov_matrix)
            @test r.redflag == 0 && r.hessian_flag == 0
            # a singular Hessian: as the matrix method
            rs, Hs = _bf39_with(() -> (RF.sf_classify_hessian(f, x0; message = false), ForwardDiff.hessian(f, x0)), :singular)
            @test rs.redflag == 1 && rs.hessian_flag == 1 && allnan(rs.var_cov_matrix) && isequal(rs.hessian, Hs)
            # an error on nested-dual input only: NaN matrices, redflag 1, the message (also with message = false)
            de = DomainError(-1.0, "nested, test")
            (re, txt) = _bf39_with(() -> capture(() -> RF.sf_classify_hessian(f, x0; message = false)), :nested, de)
            @test re.redflag == 1 && re.hessian_flag == 1 && allnan(re.hessian) && allnan(re.var_cov_matrix)
            @test size(re.hessian) == (2, 2) && size(re.var_cov_matrix) == (2, 2) && eltype(re.hessian) == Float64
            @test occursin("The Hessian matrix could not be computed at the estimate (DomainError with -1.0: nested, test). Standard errors are set to NaN.", txt)
            (rb, txtb) = _bf39_with(() -> capture(() -> RF.sf_classify_hessian(f, x0; at_clamp = true, message = true)), :nested, de)
            @test rb.redflag == 1 && occursin("boundary solution", txtb) && occursin("could not be computed", txtb)
            (rq, txtq) = _bf39_with(() -> capture(() -> RF.sf_classify_hessian(f, x0; at_clamp = true, message = false)), :nested, de)
            @test !occursin("boundary solution", txtq) && occursin("could not be computed", txtq)
            # a message that cannot be printed: the error's type instead
            (rbs, txtbs) = _bf39_with(() -> capture(() -> RF.sf_classify_hessian(f, x0)), :nested, _BF39BadShow())
            @test rbs.redflag == 1 && occursin("_BF39BadShow", txtbs)
            # fatal errors are rethrown
            for ex in (InterruptException(), _BF39OutOfGPUMemoryError(), _BF39GPUText(), StackOverflowError())
                @test _bf39_with(() -> capture(() -> RF.sf_classify_hessian(f, x0))[1], :nested, ex) === ex
            end
        else
            @test fixed
        end
    end

    @testset "sf_final_fg: an error when evaluating again at the estimate" begin
        LS = Optim.LineSearches
        f = _bf39_obj
        x0 = [0.3, 0.4]
        # a line search that throws on its first call: Optim stops with ls_failed
        lsfail = (d, x, s, alpha, x_ls, phi0, dphi0) -> throw(LS.LineSearchException("forced by the test", 1e-3))
        rl = optimize(f, x0, GradientDescent(linesearch = lsfail), Optim.Options(iterations = 20); autodiff = AutoForwardDiff())
        @test RF.sf_ls_failed(rl)
        xm = Optim.minimizer(rl)
        if fixed
            # no error: what the function returned before, the same objects
            @test RF.sf_final_fg(rl, f) === (f(xm), maximum(abs, ForwardDiff.gradient(f, xm)))
            # the gradient fails: f at the minimizer and NaN
            (v, txt) = _bf39_with(() -> capture(() -> RF.sf_final_fg(rl, f)), :dual, DomainError(-2.0, "gradient, test"))
            @test v isa Tuple && v[1] === f(xm) && isnan(v[2]) && v[2] isa Float64
            @test occursin("The gradient of the log-likelihood could not be evaluated at the estimate (DomainError with -2.0", txt)
            # f itself fails: Optim's own minimum and NaN
            (v, txt) = _bf39_with(() -> capture(() -> RF.sf_final_fg(rl, f)), :all, DomainError(-6.0, "value, test"))
            @test v isa Tuple && v[1] === Optim.minimum(rl) && isnan(v[2])
            @test occursin("The log-likelihood could not be evaluated at the estimate (DomainError with -6.0", txt)
            # a gradient-free solver: Optim's minimum and NaN
            rm_ = optimize(f, x0, NelderMead(), Optim.Options(iterations = 200))
            (v, txt) = _bf39_with(() -> capture(() -> RF.sf_final_fg(rm_, f)), :dual, DomainError(-2.0, "gradient, test"))
            @test v[1] === Optim.minimum(rm_) && isnan(v[2]) && occursin("gradient of the log-likelihood", txt)
            # fatal errors are rethrown
            @test _bf39_with(() -> capture(() -> RF.sf_final_fg(rl, f))[1], :dual, InterruptException()) isa InterruptException
            @test _bf39_with(() -> capture(() -> RF.sf_final_fg(rm_, f))[1], :all, _BF39OutOfGPUMemoryError()) isa _BF39OutOfGPUMemoryError
        else
            @test fixed
        end
    end

    # ---- MLE fits through the unified API, with the test's own model id ----------------------
    rng = Xoshiro(2024); n = 200
    X = hcat(ones(n), randn(rng, n)); z = randn(rng, n)
    y = X * [1.0, 0.5] .+ 0.3 .* randn(rng, n) .- abs.(0.5 .* randn(rng, n) .* exp.(0.3 .* z))
    Z = hcat(ones(n), z)
    usp = sfmodel_spec(depvar = y, frontier = X, zvar = Z, noise = :Normal, ineff = :HalfNormal,
                       hetero = [:sigma_u_sq], type = :prod)
    msp = usp.mle_spec
    msp39 = typeof(msp)((f === :modelid ? _BF39Half : getfield(msp, f) for f in fieldnames(typeof(msp)))...)
    usp39 = typeof(usp)((f === :mle_spec ? msp39 : getfield(usp, f) for f in fieldnames(typeof(usp)))...)
    ini = sfmodel_init(spec = usp, frontier = X \ y, ln_sigma_u_sq = [log(0.25), 0.0], ln_sigma_v_sq = log(0.09))
    bfgs = sfmodel_opt(warmstart_solver = nothing, main_solver = BFGS(), main_opt = (iterations = 200, g_abstol = 1e-6))
    mle = sfmodel_method(method = :MLE)
    fitm(sp; opt = bfgs, init = ini, verbose = false) =
        capture(() -> sfmodel_fit(spec = sp, method = mle, init = copy(init), optim_options = opt,
                                  marginal = true, show_table = false, verbose = verbose))
    _BF39_MODE[] = :off; _BF39_ARMED[] = false
    r0, _ = fitm(usp)
    r0b, _ = fitm(usp39)
    @test r0.converged && r0.redflag == 0
    same(a, b, k) = isequal(getproperty(a, k), getproperty(b, k))
    @test all(k -> same(r0, r0b, k), (:coeff, :std_err, :Hessian, :loglikelihood, :gradient_norm, :jlms, :bc, :marginal))

    @testset "MLE: an error in the final Hessian keeps the fit" begin
        _BF39_MODE[] = :domain
        r1, txt = fitm(usp39)
        _BF39_MODE[] = :off
        @test !(r1 isa Exception)
        if !(r1 isa Exception)
            @test all(k -> same(r0, r1, k), (:coeff, :loglikelihood, :gradient_norm, :converged, :jlms, :bc, :marginal))
            @test r1.redflag == 1
            @test allnan(r1.std_err) && allnan(r1.list[:var_cov_mat]) && allnan(r1.Hessian)
            @test size(r1.Hessian) == size(r0.Hessian)
            @test occursin("The Hessian matrix could not be computed at the estimate (DomainError with -1.0: injected by the test). Standard errors are set to NaN.", txt)
        end
        # a verbose fit prints the same message and returns the same result
        _BF39_MODE[] = :domain
        r1v, txtv = fitm(usp39; verbose = true)
        _BF39_MODE[] = :off
        @test !(r1v isa Exception) && occursin("could not be computed at the estimate", txtv) &&
              isequal(r1v isa Exception ? nothing : r1v.coeff, r1 isa Exception ? 0 : r1.coeff)
    end

    @testset "MLE: Ctrl-C, out of memory, stack overflow and CUDA's error still end the fit" begin
        for (mode, check) in ((:interrupt, e -> e isa InterruptException),
                              (:gpuoom,    e -> e isa _BF39OutOfGPUMemoryError),
                              (:gputext,   e -> e isa ErrorException && occursin("Out of GPU memory", e.msg)),
                              (:stackover, e -> e isa StackOverflowError))
            _BF39_MODE[] = mode
            r, _ = fitm(usp39)
            _BF39_MODE[] = :off
            @test check(r)
        end
    end

    @testset "MLE: an error when evaluating again at the estimate (sf_final_fg)" begin
        LS = Optim.LineSearches
        # a line search that sets the trap and throws on its first call: Optim stops with
        # ls_failed, and sf_final_fg evaluates the objective and its gradient again
        armls = (d, x, s, alpha, x_ls, phi0, dphi0) -> (_BF39_ARMED[] = true;
                     throw(LS.LineSearchException("forced by the test", 1e-3)))
        gd = sfmodel_opt(warmstart_solver = nothing, main_solver = GradientDescent(linesearch = armls),
                         main_opt = (iterations = 20,))
        at0 = sfmodel_opt(warmstart_solver = nothing, main_solver = Newton(), main_opt = (iterations = 0,))
        # only dual input fails: loglikelihood at coeff, gradient_norm NaN, and the Hessian fails too
        # (started from `ini`, away from the optimum, so that the first iteration calls the line search)
        _BF39_MODE[] = :armed_dual; _BF39_ARMED[] = false
        r2, txt2 = fitm(usp39; opt = gd)
        _BF39_MODE[] = :off; _BF39_ARMED[] = false
        @test !(r2 isa Exception)
        if !(r2 isa Exception)
            e2, _ = fitm(usp; opt = at0, init = collect(r2.coeff))        # the objective at r2's coeff
            @test isnan(r2.gradient_norm) && r2.redflag == 1 && allnan(r2.std_err)
            # sf_final_fg evaluates f in Float64, the zero-iteration Newton fit with nested dual
            # numbers: the two paths can differ in the last bits, so isapprox. Under Optim 2.0 the
            # failed line search moved coeff away from the start, where the log-likelihood is far
            # lower, so rtol = 1e-12 still tells "f at coeff" from Optim's own minimum. Under
            # Optim 2.2.1 and later a failed line search leaves coeff at the start, and both
            # values are then at the start.
            @test isapprox(r2.loglikelihood, e2.loglikelihood; rtol = 1e-12)
            @test occursin("The gradient of the log-likelihood could not be evaluated at the estimate", txt2) &&
                  occursin("The gradients are problematic", txt2) && occursin("could not be computed at the estimate", txt2)
            @test isequal(r2.jlms, e2.jlms) && isequal(r2.bc, e2.bc)
        end
        # every input fails: Optim's own minimum, which is the objective at the start (the line
        # search failed on its first call; under Optim 2.0 coeff has moved away from the start,
        # under Optim 2.2.1 and later it is the start). Both values come from dual numbers
        # (gradient descent's first value-and-gradient call, the zero-iteration Newton fit's
        # value-gradient-Hessian call), whose values can differ in the last bits with the Julia version,
        # so rtol = 1e-12, as for r2 above.
        _BF39_MODE[] = :armed_all; _BF39_ARMED[] = false
        r3, txt3 = fitm(usp39; opt = gd)
        _BF39_MODE[] = :off; _BF39_ARMED[] = false
        @test !(r3 isa Exception)
        if !(r3 isa Exception)
            e3, _ = fitm(usp; opt = at0)                                   # the objective at the start
            @test isnan(r3.gradient_norm) && r3.redflag == 1 &&
                  isapprox(r3.loglikelihood, e3.loglikelihood; rtol = 1e-12)
            @test occursin("The log-likelihood could not be evaluated at the estimate", txt3)
        end
    end

    @testset "MLE bootstrap: an error in a replicate's Hessian or marginal effects is a redraw" begin
        boot = _BF39MB.sfmodel_boot_marginal
        rb, _ = fitm(usp39)                                  # the fit the bootstrap refers to (mode :off)
        @test rb.converged && rb.redflag == 0
        names0 = copy(_BF39M._dicM[:σᵤ²])
        function bootrun(mode, at; R = 5)
            _BF39_MODE[] = mode; _BF39_AT[] = at; _BF39_CALLS[] = 0
            out = nothing; logs = Any[]
            try
                logs, out = Test.collect_test_logs() do
                    try
                        boot(result = rb, R = R, seed = 11, getBootData = true, every = 1000)
                    catch e
                        e
                    end
                end
            finally
                _BF39_MODE[] = :off; _BF39_AT[] = 0
            end
            return out, logs
        end
        o0, l0 = bootrun(:off, 0)
        @test o0 isa Tuple && !any(l -> occursin("threw an error", string(l.message)), l0)   # no error, none counted
        # the second replicate's Hessian throws: drawn again, R replicates, a warning naming it
        o1, l1 = bootrun(:hessonce, 2)
        @test o1 isa Tuple && size(o1[2], 1) == 5 && all(isfinite, o1[2])
        w1 = join(string.(getfield.(l1, :message)), " ")
        @test occursin("computing the Hessian threw an error: 1", w1) &&
              occursin("the last error in a Hessian or the marginal effects: DomainError with -3.0", w1)
        @test o1 isa Tuple && isequal(o1[2][1, :], o0[2][1, :])   # the replicate done before it is kept as it was
        # the third replicate's marginal effects throw
        o2, l2 = bootrun(:margonce, 3)
        @test o2 isa Tuple && size(o2[2], 1) == 5 && all(isfinite, o2[2])
        w2 = join(string.(getfield.(l2, :message)), " ")
        @test occursin("computing the marginal effects threw an error: 1", w2) && occursin("DomainError with -4.0", w2)
        @test o2 isa Tuple && isequal(o2[2][1:2, :], o0[2][1:2, :])
        # every replicate's Hessian throws: the 10R limit, with the new kind in the message
        o3, _ = bootrun(:domain, 0; R = 2)
        @test o3 isa ErrorException && occursin("(the limit is 10R = 20)", o3.msg) &&
              occursin(r"computing the Hessian threw an error: \d+", o3.msg)
        # Ctrl-C in a replicate's Hessian ends the bootstrap; the names are restored
        o4, _ = bootrun(:interrupt, 0)
        @test o4 isa InterruptException
        @test _BF39M._dicM[:σᵤ²] == names0
    end

    # The MCI, MSLE and panel fit functions call the same sf_classify_hessian(f, x) (checked in
    # the source below; an end-to-end check of each, with an error injected on nested-dual
    # input, is not run here, because compiling those backends would double this file's
    # run time).
    @testset "the four fit functions compute the final Hessian through the guard (source)" begin
        root = joinpath(pkgdir(SFrontiers), "src")
        for f in (joinpath("sf_MLE", "SFmainfun.jl"), joinpath("sf_MCI", "sf_MCI_v21.jl"),
                  joinpath("sf_MSLE", "sf_MSLE_v21.jl"), joinpath("sf_panel", "sf_panel_v20.jl"))
            s = read(joinpath(root, f), String)
            @test count("sf_classify_hessian(_lik, _coevec;", s) == 1
            @test count(r"(?<![_a-z])hessian\(_lik, _coevec\)", s) == 0
        end
    end
end
