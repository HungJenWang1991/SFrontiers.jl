# ---------------------------------------------------------------------------
# dsl_equiv_runner.jl — standalone DSL-vs-keyword spec-equivalence sweep.
#
# Usage (from the package root):
#   julia --project=. test/dsl_equiv_runner.jl [--out=PATH] [--smoke]
# ---------------------------------------------------------------------------

using Random, Statistics, LinearAlgebra, Dates, Printf
using DataFrames
using Distributions, QuadGK          # noise_oracle_lib deps (oracle unused here)
using SFrontiers, Optim

include(joinpath(@__DIR__, "noise_oracle_lib.jl"))
include(joinpath(@__DIR__, "dsl_equiv_lib.jl"))

out   = joinpath(dirname(@__DIR__), "dsl_equiv_results.jsonl")
smoke = false
for a in ARGS
    if startswith(a, "--out=")
        global out = chopprefix(a, "--out=")
    elseif a == "--smoke"
        global smoke = true
    end
end
progress_file = out * ".progress.txt"

const DF  = make_dsl_equiv_data()
const DFP = make_dsl_equiv_panel()
cases = build_dsl_equiv_cases(DF, DFP)
smoke && (cases = cases[1:2])

println("DSL-equivalence sweep: $(length(cases)) cases + bare-Symbol error-parity probe")
println("Results -> $out")
flush(stdout)

function rec_verdict(rec)
    worst = :pass
    function scan(d)
        for (k, v) in d
            v isa AbstractDict || continue
            haskey(v, :status) && v[:status] in (:attention, :fail, :error) &&
                (worst = :attention)
            scan(v)
        end
    end
    scan(rec[:checks])
    return worst
end

io = open(out, "a")
n_pass = 0; n_attn = 0; attn = String[]
t_start = time()

function emit(rec, i, total)
    rec[:ts] = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS\Z")
    println(io, json_line(rec)); flush(io)
    v = haskey(rec, :harness_error) ? :attention : rec_verdict(rec)
    if v == :attention
        global n_attn += 1; push!(attn, String(rec[:case]))
    else
        global n_pass += 1
    end
    println("$(v)  ($(get(rec, :elapsed_s, "?")) s)"); flush(stdout)
    open(progress_file, "w") do p
        println(p, "$(i)/$(total) done  last=$(rec[:case])  verdict=$(v)  " *
                   "elapsed_total_min=$(round((time()-t_start)/60, digits=1))")
    end
end

total = length(cases) + 1
for (i, case) in enumerate(cases)
    print("[$i/$total] $(case.name) ... "); flush(stdout)
    rec = try
        run_dsl_equiv_case(case, DF, DFP)
    catch e
        Dict{Symbol,Any}(:kind => :dsl_equiv, :case => case.name,
                         :checks => Dict{Symbol,Any}(), :harness_error => _errstr(e))
    end
    emit(rec, i, total)
end

print("[$total/$total] A14-bare-symbol-hetero ... "); flush(stdout)
rec = try
    run_dsl_error_parity_case(DF)
catch e
    Dict{Symbol,Any}(:kind => :dsl_equiv, :case => "A14-bare-symbol-hetero",
                     :checks => Dict{Symbol,Any}(), :harness_error => _errstr(e))
end
emit(rec, total, total)
close(io)

println("\n===== DSL-EQUIVALENCE SWEEP COMPLETE =====")
println("records: $total; clean: $n_pass; needs-attention: $n_attn")
isempty(attn) || println("needs attention:\n  " * join(attn, "\n  "))
println("total wall: $(round((time()-t_start)/60, digits=1)) min")
open(progress_file, "w") do p
    println(p, "COMPLETE $total records, $(n_attn) need attention, " *
               "$(round((time()-t_start)/60, digits=1)) min")
end
