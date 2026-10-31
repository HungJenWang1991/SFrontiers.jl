# ---------------------------------------------------------------------------
# gpu_sweep_runner.jl — the CPU-GPU equivalence sweep (README: test/GPU_VALIDATION.md).
# Designed to run detached: writes the environment record and then one JSON
# line per case (append + flush) so results survive any interruption; give
# each run its own --out path.
#
# Usage (from the package root):
#   julia --project=. test/gpu_sweep_runner.jl [--tiers=1,2,3,4] [--out=PATH] [--smoke]
#
# Exit code: 0 when no case needs attention, 1 otherwise.
#
# NOTE: `using CUDA` must precede `using SFrontiers` — the package registers
# its GPU method overloads only when CUDA is already loaded.
# ---------------------------------------------------------------------------

using Random, Statistics, LinearAlgebra, Dates, Printf
using CUDA
@assert CUDA.functional() "CUDA is not functional on this machine; cannot run the GPU sweep."
using SFrontiers, Optim, DataFrames, StatsAPI, RecipesBase

include(joinpath(@__DIR__, "gpu_equivalence_lib.jl"))   # also includes gpu_sweep_verdict.jl

# ---------------------------------------------------------------- arguments
tiers = [1, 2, 3, 4]
out   = joinpath(dirname(@__DIR__), "gpu_sweep_results.jsonl")
smoke = false
for a in ARGS
    if startswith(a, "--tiers=")
        global tiers = parse.(Int, split(chopprefix(a, "--tiers="), ","))
    elseif startswith(a, "--out=")
        global out = chopprefix(a, "--out=")
    elseif a == "--smoke"
        global smoke = true
    end
end
progress_file = out * ".progress.txt"

CUDA.allowscalar(false)

cases = build_cases(tiers)
smoke && (cases = [c for c in cases if c.name in SMOKE_CASES])

println("GPU sweep: $(length(cases)) cases, tiers=$(tiers), device=$(CUDA.name(CUDA.device()))")
println("Results -> $out")

# -------------------------------------------------------------- environment
# Printed here and written as the run's first line of the output file.
env = Dict(:kind => "environment", :julia => string(VERSION),
           :sfrontiers => string(pkgversion(SFrontiers)), :cuda_jl => string(pkgversion(CUDA)),
           :cuda_runtime => string(CUDA.runtime_version()), :cuda_driver => string(CUDA.driver_version()),
           :device => CUDA.name(CUDA.device()),
           :date_utc => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS\Z"),
           :tiers => tiers, :n_cases => length(cases))
println("Environment: ", json_line(env))
flush(stdout)

# ---------------------------------------------------------------------- run
io = open(out, "a")
println(io, json_line(env)); flush(io)
counts = Dict(:pass => 0, :allowscalar => 0, :ridge => 0, :attention => 0)
attn   = String[]
ridges = Tuple{String,Any,Any}[]                # (case, dll_cpu, dll_gpu)
# largest :diff of the passing checks whose names start with "fixed_", for the Float64
# (false) and the Float32 (true) cases: (value, case, check)
fixed_max = Dict(false => (-Inf, "", ""), true => (-Inf, "", ""))
t_start = time()
for (i, case) in enumerate(cases)
    print("[$i/$(length(cases))] $(case.name) ... "); flush(stdout)
    rec = try
        run_case(case)
    catch e   # run_case shouldn't throw; belt and braces
        Dict{Symbol,Any}(:case => case.name, :tier => case.tier,
                         :checks => Dict{Symbol,Any}(),
                         :harness_error => _errstr(e))
    end
    rec[:ts] = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS\Z")
    println(io, json_line(rec)); flush(io)
    v = case_verdict(rec)
    counts[v] += 1
    v == :attention && push!(attn, case.name)
    if v == :ridge
        r = get(get(rec[:checks], :fullfit, Dict{Symbol,Any}()), :ridge, Dict{Symbol,Any}())
        push!(ridges, (case.name, get(r, :dll_cpu, nothing), get(r, :dll_gpu, nothing)))
    end
    for (k, c) in rec[:checks]
        startswith(string(k), "fixed_") && c isa AbstractDict && get(c, :status, nothing) == :pass || continue
        c[:diff] > fixed_max[case.float32][1] && (fixed_max[case.float32] = (c[:diff], case.name, string(k)))
    end
    println("$(v)  ($(get(rec, :elapsed_s, "?")) s)"); flush(stdout)
    open(progress_file, "w") do p
        println(p, "$(i)/$(length(cases)) done  last=$(case.name)  verdict=$(v)  " *
                   "elapsed_total_min=$(round((time()-t_start)/60, digits=1))")
    end
end
close(io)

n_clean = counts[:pass] + counts[:allowscalar]
n_ridge = counts[:ridge]
n_attn  = counts[:attention]
_fmt(x) = x isa Real ? @sprintf("%.1e", x) : string(x)
_fmt_max(t) = t[1] == -Inf ? "none" : "$(_fmt(t[1])) ($(t[2]), $(t[3]))"

println("\n===== SWEEP COMPLETE =====")
println("cases: $(length(cases)); clean: $n_clean; ridge passes: $n_ridge; needs-attention: $n_attn")
println("largest fixed-parameter relative difference: Float64 cases $(_fmt_max(fixed_max[false])); " *
        "Float32 cases $(_fmt_max(fixed_max[true]))")
isempty(ridges) || println("ridge passes:\n  " *
    join(["$c  dll_cpu = $(_fmt(a)), dll_gpu = $(_fmt(b))" for (c, a, b) in ridges], "\n  "))
isempty(attn) || println("needs attention:\n  " * join(attn, "\n  "))
println("total wall: $(round((time()-t_start)/60, digits=1)) min")
open(progress_file, "w") do p
    println(p, "COMPLETE $(length(cases)) cases, $(n_attn) need attention, " *
               "$(round((time()-t_start)/60, digits=1)) min")
end
exit(n_attn == 0 ? 0 : 1)
