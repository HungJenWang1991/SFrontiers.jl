# The draw-memory checks on the GPU (gated). `using CUDA` must precede `using SFrontiers` for the
# GPU overloads to register, and this file is included after SFrontiers is loaded, so the checks run
# in a separate process (gpu_draw_memory_runner.jl), launched as test_gpu_equivalence.jl launches
# its sweep, with the default load path restored because CUDA is not a test dependency. Skipped
# when CUDA is not available or not functional.

@testset "draw memory on the GPU (gated)" begin
    pkgroot = dirname(@__DIR__)
    julia   = joinpath(Sys.BINDIR, Base.julia_exename())
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
        @test_skip "CUDA not available/functional - draw-memory GPU checks skipped"
    else
        runner = joinpath(@__DIR__, "gpu_draw_memory_runner.jl")
        cmd = addenv(`$julia --project=$pkgroot --startup-file=no $runner`,
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
