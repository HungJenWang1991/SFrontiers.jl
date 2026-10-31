# gpu_memory_probe.jl -- the memory check that GPU fits make before the optimizer, on the GPU, in a process of
# its own.
#
# (a) SFDraws.sf_gpu_free_bytes reads the GPU memory this process can still allocate: an Int above 0 and not
#     above the device's total memory.
# (b) Four small GPU fits held at their starting values, with 256 draws and the default chunks = 10:
#     cross-sectional MCI and MSLE (normal-half-normal, N = 200) and panel MSLE and MCI (true fixed effects,
#     half-normal, 50 firms × T = 4). Each fit makes the check once, and its debug record carries the estimate
#     that sf_hessian_pass_bytes gives for the fit's work arrays, rows and parameters; no fit warns about GPU
#     memory.
# (c) An estimate far above any free memory (3 arrays, 10^6 rows, 2^15 draws, chunks = 1, 12 parameters: about
#     1.3e14 bytes, computed, not allocated) gives one warning, and the warning names `chunks`.
#
# Run in a separate process because `using CUDA` must precede `using SFrontiers` for the GPU overloads to
# register; test_gpu_equivalence.jl launches it after its smoke sweep, with the default load path restored
# because CUDA is not a test dependency. Prints each check and its outcome, then a final "GPU MEMORY PROBE: PASS"
# or "GPU MEMORY PROBE: FAIL", and exits with 0 or 1.
using CUDA
using SFrontiers, Optim, Test, Random

CUDA.allowscalar(false)

const _GMP_FAILED = String[]

# Runs one check: prints its name, then its outcome; a check that throws or returns anything but `true` fails.
function _gmp_check(ok, name::AbstractString)
    println(name)
    pass = try
        ok() === true
    catch e
        println("    error: ", typeof(e), ": ", first(split(sprint(showerror, e), "\n")))
        false
    end
    pass || push!(_GMP_FAILED, name)
    println("    ", pass ? "PASS" : "FAIL")
    flush(stdout)
    return pass
end

# Cross-sectional data, normal-half-normal, N = 200: y = 1 + 0.5x + v - u, v ~ N(0, 0.3^2), u = |N(0, 0.4^2)|.
function _gmp_cross_spec()
    rng = Xoshiro(62)
    n = 200
    X = hcat(ones(n), randn(rng, n))
    y = X * [1.0, 0.5] .+ 0.3 .* randn(rng, n) .- abs.(0.4 .* randn(rng, n))
    return sfmodel_spec(depvar = y, frontier = X, noise = :Normal, ineff = :HalfNormal)
end

# Panel data, Wang-Ho true fixed effects with half-normal inefficiency and one scaling variable: 50 firms × T = 4.
function _gmp_panel_spec()
    rng = Xoshiro(44)
    Nf, Tp = 50, 4
    NT = Nf * Tp
    idp = repeat(1:Nf, inner = Tp)
    Xp = hcat(randn(rng, NT), randn(rng, NT))
    zp = randn(rng, NT)
    yp = repeat(randn(rng, Nf), inner = Tp) .+ Xp * [0.6, -0.4] .+ 0.3 .* randn(rng, NT) .-
         exp.(0.3 .* zp) .* repeat(abs.(0.4 .* randn(rng, Nf)), inner = Tp)
    return sfmodel_spec(depvar = yp, frontier = Xp, zvar = zp, id = idp, noise = :Normal, ineff = :HalfNormal,
                        type = :prod, datatype = :panel_TFE)
end

# Fits held at their starting values: no warm start and no Newton iteration.
const _GMP_HELD = sfmodel_opt(main_solver = Newton(), main_opt = (iterations = 0,))

# One GPU fit with n_draws = 256 and the default chunks, its logs collected down to the debug level. It must make
# the memory check once, with the estimate for `narrays` work arrays of cld(`nrows`, 10) rows, and must not warn.
function _gmp_fit_check(name, mkspec, method::Symbol, narrays::Int, nrows::Int)
    _gmp_check("(b) $name: one memory check, the estimate for its size, no warning") do
        spec = mkspec()
        logs, r = Test.collect_test_logs(; min_level = Base.CoreLogging.Debug) do
            sfmodel_fit(spec = spec, method = sfmodel_method(method = method, n_draws = 256, GPU = true),
                        optim_options = _GMP_HELD, marginal = false, show_table = false, verbose = false)
        end
        recs = [l for l in logs if l.level == Base.CoreLogging.Debug &&
                                   occursin("GPU memory check", string(l.message))]
        warns = [l for l in logs if l.level == Base.CoreLogging.Warn && occursin("GPU memory", string(l.message))]
        want = SFrontiers.SFDraws.sf_hessian_pass_bytes(narrays, nrows, 256, 10, length(r.coeff))
        est = length(recs) == 1 ? recs[1].kwargs[:est] : nothing
        free = length(recs) == 1 ? get(recs[1].kwargs, :free, nothing) : nothing
        println("    check records: $(length(recs)); est = $(repr(est)), expected $want ($narrays arrays, ",
                "$nrows rows, $(length(r.coeff)) parameters); free = $(repr(free)); GPU memory warnings: ",
                length(warns))
        length(recs) == 1 && est == want && isempty(warns)
    end
end

if !CUDA.functional()
    println("CUDA is not functional: the probe needs a working GPU.")
    println("GPU MEMORY PROBE: FAIL")
    exit(1)
end
println("GPU memory probe on ", CUDA.name(CUDA.device()), ", CUDA.jl ", pkgversion(CUDA))

_gmp_check("(a) sf_gpu_free_bytes() is an Int in (0, CUDA.total_memory()]") do
    f = SFrontiers.SFDraws.sf_gpu_free_bytes()
    t = CUDA.total_memory()
    println("    sf_gpu_free_bytes() = $(repr(f)); CUDA.total_memory() = $t")
    f isa Int && 0 < f <= t
end

_gmp_fit_check("cross-sectional MCI", _gmp_cross_spec, :MCI, SFrontiers.SFDraws.SF_WORK_ARRAYS.mci, 200)
_gmp_fit_check("cross-sectional MSLE", _gmp_cross_spec, :MSLE, SFrontiers.SFDraws.SF_WORK_ARRAYS.msle, 200)
_gmp_fit_check("panel MSLE", _gmp_panel_spec, :MSLE, SFrontiers.SFDraws.SF_WORK_ARRAYS.panel_msle, 50)
_gmp_fit_check("panel MCI", _gmp_panel_spec, :MCI, SFrontiers.SFDraws.SF_WORK_ARRAYS.panel_mci, 50)

_gmp_check("(c) an estimate of about 1.3e14 bytes gives one warning, which names `chunks`") do
    logs, w = Test.collect_test_logs() do
        SFrontiers.SFDraws.sf_check_gpu_memory(3, 10^6, 2^15, 1, 12; unit = "observations")
    end
    warns = [l for l in logs if l.level == Base.CoreLogging.Warn]
    foreach(l -> println("    warning: ", l.message), warns)
    w == true && length(warns) == 1 && occursin("chunks", string(warns[1].message))
end

println(isempty(_GMP_FAILED) ? "all checks passed" : "failed: " * join(_GMP_FAILED, "; "))
println(isempty(_GMP_FAILED) ? "GPU MEMORY PROBE: PASS" : "GPU MEMORY PROBE: FAIL")
exit(isempty(_GMP_FAILED) ? 0 : 1)
