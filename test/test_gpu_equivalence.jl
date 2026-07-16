# ---------------------------------------------------------------------------
# Gated GPU/CUDA equivalence smoke tests (GPU audit, 06/07_gpu_audit docs).
#
# CUDA.jl must be loaded BEFORE SFrontiers for the GPU overloads to register,
# and runtests.jl (necessarily) loads SFrontiers first — so the GPU checks run
# in a SEPARATE Julia process with the correct load order, via the sweep
# runner's --smoke subset (3 cases: plain MCI, hard-dist MSLE, hetero+marginal).
#
# Gate: skipped (no-op testset) unless CUDA.jl is installed AND functional.
# The full tiered sweep (67 cases) is run manually:
#   julia --project=. test/gpu_sweep_runner.jl
# ---------------------------------------------------------------------------

@testset "GPU equivalence (gated smoke)" begin
    pkgroot = dirname(@__DIR__)
    julia   = joinpath(Sys.BINDIR, Base.julia_exename())

    # Pkg.test sandboxes JULIA_LOAD_PATH, hiding user-env packages. CUDA is
    # deliberately NOT a test dependency, so restore the default load path in
    # the subprocesses ("@v#.#" = the version default env) to find it there.
    _lp = join(["@", "@v#.#", "@stdlib"], Sys.iswindows() ? ";" : ":")
    probe_cmd = addenv(
        `$julia --project=$pkgroot --startup-file=no -e "try; using CUDA; print(CUDA.functional()); catch; print(false); end"`,
        "JULIA_LOAD_PATH" => _lp)
    gpu_ok = try
        strip(read(probe_cmd, String)) == "true"
    catch
        false
    end

    if !gpu_ok
        @test_skip "CUDA not available/functional - GPU equivalence smoke skipped"
    else
        mktempdir() do tmp
            outfile = joinpath(tmp, "gpu_smoke.jsonl")
            runner  = joinpath(@__DIR__, "gpu_sweep_runner.jl")
            cmd = addenv(
                `$julia --project=$pkgroot --startup-file=no $runner --smoke --out=$outfile`,
                "JULIA_LOAD_PATH" => _lp)
            log = IOBuffer()
            ok  = success(pipeline(cmd; stdout=log, stderr=log))
            out = String(take!(log))
            @test ok
            @test occursin("SWEEP COMPLETE", out)
            @test occursin("needs-attention: 0", out)
            # Every smoke case produced a result line and none carry a harness error
            m = match(r"GPU sweep: (\d+) cases", out)
            @test m !== nothing
            lines = filter(!isempty, readlines(outfile))
            @test length(lines) == parse(Int, m.captures[1])
            @test !any(l -> occursin("harness_error", l), lines)
            # GPU-01: result jlms/bc must come back on host after a GPU fit
            @test !any(l -> occursin("\"gpu_result_device_fields\":\"jlms", l), lines)
        end
    end
end
