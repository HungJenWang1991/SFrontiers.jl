# The draw-memory checks on the GPU (gated). `using CUDA` must precede `using SFrontiers` for the
# GPU overloads to register, and this file is included after SFrontiers is loaded, so the checks run
# in a separate process (gpu_draw_memory_runner.jl), launched as test_gpu_equivalence.jl launches
# its sweep: with the test environment's load path followed by the default environment, because
# CUDA is not a test dependency. Skipped when CUDA is not available or not functional, and when its
# version is newer than the versions SFrontiers supports (6.4 or earlier).

@testset "draw memory on the GPU (gated)" begin
    julia   = joinpath(Sys.BINDIR, Base.julia_exename())
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
        @test_skip "CUDA not available/functional - draw-memory GPU checks skipped"
    elseif cuda_v !== nothing && cuda_v >= v"6.5"
        @warn("Draw-memory GPU checks skipped: CUDA.jl $cuda_v in the default environment is " *
              "newer than the versions SFrontiers supports (6.4 or earlier)")
        @test_skip "CUDA.jl newer than 6.4 - draw-memory GPU checks skipped"
    else
        runner = joinpath(@__DIR__, "gpu_draw_memory_runner.jl")
        cmd = addenv(`$julia --startup-file=no $runner`,
                     "JULIA_LOAD_PATH" => _lp)
        log = IOBuffer()
        ok  = success(pipeline(cmd; stdout = log, stderr = log))
        out = String(take!(log))
        @test ok
        m = match(r"DRAW-MEMORY GPU CHECKS COMPLETE: (\d+) checks, (\d+) mismatches", out)
        @test m !== nothing
        if m !== nothing
            @test parse(Int, m.captures[1]) == 24
            @test parse(Int, m.captures[2]) == 0
        end
        @test !occursin("Scalar indexing is disallowed", out)
    end
end
