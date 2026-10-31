# ---------------------------------------------------------------------------
# mle_crosscheck_runner.jl — standalone MLE vs MCI/MSLE fixed-θ
# cross-backend equivalence sweep.
#
# Usage (from the package root):
#   julia --project=. test/mle_crosscheck_runner.jl [--out=PATH] [--smoke]
#
# QuadGK and Distributions are dependencies of the package.
# ---------------------------------------------------------------------------

using Random, Statistics, LinearAlgebra, Dates, Printf
using Distributions, QuadGK
using SFrontiers, Optim

include(joinpath(@__DIR__, "noise_oracle_lib.jl"))
include(joinpath(@__DIR__, "mle_crosscheck_lib.jl"))

# ---------------------------------------------------------------- arguments
out   = joinpath(dirname(@__DIR__), "mle_crosscheck_results.jsonl")
smoke = false
for a in ARGS
    if startswith(a, "--out=")
        global out = chopprefix(a, "--out=")
    elseif a == "--smoke"
        global smoke = true
    end
end
progress_file = out * ".progress.txt"

cases = build_crosscheck_cases()
smoke && (cases = [c for c in cases if c.name in CROSSCHECK_SMOKE])

println("MLE cross-check sweep: $(length(cases)) cases")
println("Results -> $out")
flush(stdout)

function rec_verdict(rec)
    worst = :pass
    function scan(d)
        for (k, v) in d
            v isa AbstractDict || continue
            if haskey(v, :status) && v[:status] in (:attention, :fail, :error)
                worst = :attention
            end
            for kk in (:mle_status, :MCI_status, :MSLE_status)
                haskey(v, kk) && v[kk] in (:attention, :error) && (worst = :attention)
            end
            scan(v)
        end
    end
    scan(rec[:checks])
    return worst
end

io = open(out, "a")
n_pass = 0; n_attn = 0; attn = String[]
t_start = time()
for (i, case) in enumerate(cases)
    print("[$i/$(length(cases))] $(case.name) ... "); flush(stdout)
    rec = try
        run_crosscheck_case(case)
    catch e
        Dict{Symbol,Any}(:kind => :mle_crosscheck, :case => case.name,
                         :checks => Dict{Symbol,Any}(), :harness_error => _errstr(e))
    end
    rec[:ts] = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS\Z")
    println(io, json_line(rec)); flush(io)
    v = haskey(rec, :harness_error) ? :attention : rec_verdict(rec)
    if v == :attention
        global n_attn += 1; push!(attn, case.name)
    else
        global n_pass += 1
    end
    println("$(v)  ($(get(rec, :elapsed_s, "?")) s)"); flush(stdout)
    open(progress_file, "w") do p
        println(p, "$(i)/$(length(cases)) done  last=$(case.name)  verdict=$(v)  " *
                   "elapsed_total_min=$(round((time()-t_start)/60, digits=1))")
    end
end
close(io)

println("\n===== MLE CROSS-CHECK SWEEP COMPLETE =====")
println("cases: $(length(cases)); clean: $n_pass; needs-attention: $n_attn")
isempty(attn) || println("needs attention:\n  " * join(attn, "\n  "))
println("total wall: $(round((time()-t_start)/60, digits=1)) min")
open(progress_file, "w") do p
    println(p, "COMPLETE $(length(cases)) cases, $(n_attn) need attention, " *
               "$(round((time()-t_start)/60, digits=1)) min")
end
