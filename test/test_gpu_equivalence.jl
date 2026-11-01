# ---------------------------------------------------------------------------
# Gated GPU tests: the CPU-GPU equivalence smoke sweep and the GPU memory probe.
#
# CUDA.jl must be loaded BEFORE SFrontiers for the GPU overloads to register,
# and runtests.jl (necessarily) loads SFrontiers first — so the GPU checks run
# in SEPARATE Julia processes with the correct load order: first the sweep
# runner's --smoke subset (the 7 smoke cases in SMOKE_CASES), then
# test/gpu_memory_probe.jl (the GPU memory estimate and warning).
#
# Gate: skipped (no-op testset) unless CUDA.jl is installed AND functional, and
# its version is one SFrontiers supports (6.4 or earlier; with CUDA.jl 6.5 the
# GPU fits fail to compile). A newer version is named in a warning.
# The full sweep (all tiers; the case count is in test/GPU_VALIDATION.md) is run manually:
#   julia --project=. test/gpu_sweep_runner.jl
# ---------------------------------------------------------------------------

@testset "GPU equivalence (gated smoke)" begin
    julia   = joinpath(Sys.BINDIR, Base.julia_exename())

    # Pkg.test runs the tests in a temporary environment that holds SFrontiers and its
    # test dependencies, and its load path hides the user's default environment. CUDA
    # is deliberately NOT a test dependency, so the subprocesses get this process's
    # load path followed by the default environment ("@v#.#" = the version default
    # env), where they find CUDA. They do not take the package folder as their
    # project: a package installed from a registry has no Manifest.toml there.
    _lp = join([Base.load_path(); "@v#.#"; "@stdlib"], Sys.iswindows() ? ";" : ":")
    probe_code = "try; using CUDA; println(CUDA.functional()); print(pkgversion(CUDA)); catch; print(false); end"
    probe_cmd = addenv(`$julia --startup-file=no -e $probe_code`, "JULIA_LOAD_PATH" => _lp)
    probe = try
        split(read(probe_cmd, String))
    catch
        String[]
    end
    gpu_ok = !isempty(probe) && probe[1] == "true"
    cuda_v = length(probe) > 1 ? tryparse(VersionNumber, probe[2]) : nothing

    if !gpu_ok
        @test_skip "CUDA not available/functional - GPU equivalence smoke skipped"
    elseif cuda_v !== nothing && cuda_v >= v"6.5"
        @warn("GPU equivalence smoke skipped: CUDA.jl $cuda_v in the default environment is " *
              "newer than the versions SFrontiers supports (6.4 or earlier)")
        @test_skip "CUDA.jl newer than 6.4 - GPU equivalence smoke skipped"
    else
        mktempdir() do tmp
            outfile = joinpath(tmp, "gpu_smoke.jsonl")
            runner  = joinpath(@__DIR__, "gpu_sweep_runner.jl")
            cmd = addenv(
                `$julia --startup-file=no $runner --smoke --out=$outfile`,
                "JULIA_LOAD_PATH" => _lp)
            log = IOBuffer()
            ok  = success(pipeline(cmd; stdout=log, stderr=log))
            out = String(take!(log))
            @test ok
            @test occursin("SWEEP COMPLETE", out)
            @test occursin("needs-attention: 0", out)
            # One environment record, then a result line for every smoke case; none
            # carries a harness error
            m = match(r"GPU sweep: (\d+) cases", out)
            @test m !== nothing
            lines = filter(!isempty, readlines(outfile))
            is_env = l -> occursin("\"kind\":\"environment\"", l)
            @test count(is_env, lines) == 1
            @test count(!is_env, lines) == parse(Int, m.captures[1])
            @test !any(l -> occursin("harness_error", l), lines)
            # the jlms and bc of a GPU fit's result come back as host arrays
            @test !any(l -> occursin("\"gpu_result_device_fields\":\"jlms", l), lines)
        end

        # The memory check that GPU fits make before the optimizer, run the same way:
        # gpu_memory_probe.jl prints "GPU MEMORY PROBE: PASS" when all its checks hold.
        mem_probe = joinpath(@__DIR__, "gpu_memory_probe.jl")
        mem_cmd = addenv(`$julia --startup-file=no $mem_probe`,
                         "JULIA_LOAD_PATH" => _lp)
        mem_log = IOBuffer()
        mem_ok  = success(pipeline(mem_cmd; stdout=mem_log, stderr=mem_log))
        mem_out = String(take!(mem_log))
        @test mem_ok
        @test occursin("GPU MEMORY PROBE: PASS", mem_out)
    end
end
