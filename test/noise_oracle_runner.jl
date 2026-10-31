# ---------------------------------------------------------------------------
# noise_oracle_runner.jl — standalone StudentT/Laplace noise oracle sweep.
# Designed to run detached: writes one JSON line per record (append + flush)
# so results survive any interruption.
#
# Usage (from the package root):
#   julia --project=. test/noise_oracle_runner.jl [--out=PATH] [--smoke]
#
# QuadGK and Distributions are dependencies of the package;
# --project=. provides them.
# ---------------------------------------------------------------------------

using Random, Statistics, LinearAlgebra, Dates, Printf
using Distributions, QuadGK
using SFrontiers

include(joinpath(@__DIR__, "noise_oracle_lib.jl"))

# ---------------------------------------------------------------- arguments
out   = joinpath(dirname(@__DIR__), "noise_oracle_results.jsonl")
smoke = false
for a in ARGS
    if startswith(a, "--out=")
        global out = chopprefix(a, "--out=")
    elseif a == "--smoke"
        global smoke = true
    end
end
progress_file = out * ".progress.txt"

cases = build_noise_oracle_cases()
smoke && (cases = [c for c in cases if c.name in NOISE_ORACLE_SMOKE])

println("Noise-oracle sweep: kernel checks + $(length(cases)) model cases")
println("Results -> $out")
flush(stdout)

# ------------------------------------------------------------------- verdict
"Worst status across a record's checks: pass < attention/error."
function rec_verdict(rec)
    worst = :pass
    function scan(d)
        for (k, v) in d
            v isa AbstractDict || continue
            if haskey(v, :status) && v[:status] in (:attention, :fail, :error)
                worst = :attention
            end
            scan(v)
        end
    end
    scan(rec[:checks])
    return worst
end

# ---------------------------------------------------------------------- run
io = open(out, "a")
n_pass = 0; n_attn = 0; attn = String[]
t_start = time()

_emit(rec) = begin
    rec[:ts] = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS\Z")
    println(io, json_line(rec)); flush(io)
    v = rec_verdict(rec)
    if v == :attention
        global n_attn += 1; push!(attn, String(rec[:case]))
    else
        global n_pass += 1
    end
    v
end

print("[kernel] spot checks ... "); flush(stdout)
krecs = try
    run_kernel_checks()
catch e
    [Dict{Symbol,Any}(:kind => :kernel, :case => "kernel-harness-error",
                      :checks => Dict{Symbol,Any}(), :harness_error => _errstr(e))]
end
kv = [_emit(r) for r in krecs]
println("$(count(==(:pass), kv))/$(length(kv)) pass"); flush(stdout)

for (i, case) in enumerate(cases)
    print("[$i/$(length(cases))] $(case.name) ... "); flush(stdout)
    rec = try
        run_noise_oracle_case(case)
    catch e   # run_noise_oracle_case shouldn't throw; belt and braces
        Dict{Symbol,Any}(:kind => :model, :case => case.name,
                         :checks => Dict{Symbol,Any}(), :harness_error => _errstr(e))
    end
    haskey(rec, :harness_error) && (global n_attn += 1; push!(attn, case.name))
    v = haskey(rec, :harness_error) ? :attention : _emit(rec)
    println("$(v)  ($(get(rec, :elapsed_s, "?")) s)"); flush(stdout)
    open(progress_file, "w") do p
        println(p, "kernel done; $(i)/$(length(cases)) model cases  last=$(case.name)  " *
                   "verdict=$(v)  elapsed_total_min=$(round((time()-t_start)/60, digits=1))")
    end
end
close(io)

println("\n===== NOISE-ORACLE SWEEP COMPLETE =====")
println("records: $(length(krecs) + length(cases)); clean: $n_pass; needs-attention: $n_attn")
isempty(attn) || println("needs attention:\n  " * join(unique(attn), "\n  "))
println("total wall: $(round((time()-t_start)/60, digits=1)) min")
open(progress_file, "w") do p
    println(p, "COMPLETE $(length(krecs) + length(cases)) records, $(n_attn) need attention, " *
               "$(round((time()-t_start)/60, digits=1)) min")
end
