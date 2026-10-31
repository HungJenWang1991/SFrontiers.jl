# The verdict rules of the CPU-GPU equivalence sweep (test/gpu_sweep_verdict.jl, shared with the sweep runner),
# checked on synthetic case records: how the statuses of a case's checks give its verdict, and the ridge rule's
# comparison of two log-likelihoods. No GPU needed.
using Test

isdefined(@__MODULE__, :case_verdict) || include(joinpath(@__DIR__, "gpu_sweep_verdict.jl"))

@testset "The verdict rules of the CPU-GPU equivalence sweep (no GPU needed)" begin
    # A case record as the runner writes it, and one check with a given status.
    _vrd_rec(checks) = Dict{Symbol,Any}(:case => "synthetic", :tier => 0, :checks => checks)
    _vrd_chk(s) = Dict{Symbol,Any}(:status => s, :diff => 0.0)

    @testset "case_verdict on synthetic records" begin
        # every check passes
        @test case_verdict(_vrd_rec(Dict{Symbol,Any}(:fit_cpu => _vrd_chk(:ok), :fit_gpu => _vrd_chk(:ok),
            :fullfit => _vrd_chk(:pass), :fixed_nll_1 => _vrd_chk(:pass), :fixed_jlms => _vrd_chk(:pass)))) === :pass

        # one ridge pass among passing checks
        ridge = Dict{Symbol,Any}(:status => :pass_ridge,
            :ridge => Dict{Symbol,Any}(:dll_cpu => 2.0e-13, :dll_gpu => -1.0e-13))
        @test case_verdict(_vrd_rec(Dict{Symbol,Any}(:fullfit => ridge, :fixed_nll_1 => _vrd_chk(:pass)))) === :ridge

        # one post-fit stage that passed under CUDA.@allowscalar
        consumers = Dict{Symbol,Any}(:show => _vrd_chk(:ok), :predict_jlms => _vrd_chk(:ok_with_allowscalar))
        @test case_verdict(_vrd_rec(Dict{Symbol,Any}(:fullfit => _vrd_chk(:pass), :consumers => consumers))) ===
              :allowscalar

        # any :fail or :error, at the top of the checks and nested two levels deep
        for s in (:fail, :error)
            @test case_verdict(_vrd_rec(Dict{Symbol,Any}(:fullfit => _vrd_chk(:pass), :fixed_bc => _vrd_chk(s)))) ===
                  :attention
            deep = Dict{Symbol,Any}(:show => _vrd_chk(:ok),
                :stage => Dict{Symbol,Any}(:inner => _vrd_chk(s)))
            @test case_verdict(_vrd_rec(Dict{Symbol,Any}(:fullfit => _vrd_chk(:pass), :consumers => deep))) ===
                  :attention
        end

        # :flags_only and :skipped are neutral
        @test case_verdict(_vrd_rec(Dict{Symbol,Any}(:fullfit => _vrd_chk(:flags_only)))) === :pass
        @test case_verdict(_vrd_rec(Dict{Symbol,Any}(:fixed_marginal => _vrd_chk(:skipped)))) === :pass
        @test case_verdict(_vrd_rec(Dict{Symbol,Any}(:fullfit => _vrd_chk(:flags_only),
            :fixed_marginal => _vrd_chk(:skipped)))) === :pass

        # a case with fixed-parameter checks only (no fits) is judged by those checks
        fixed_only = Dict{Symbol,Any}(:assemble_cpu => _vrd_chk(:ok), :assemble_gpu => _vrd_chk(:ok),
            :draws_below_one => _vrd_chk(:pass), :fixed_nll_1 => _vrd_chk(:pass), :nll_finite => _vrd_chk(:pass),
            :fixed_jlms => _vrd_chk(:pass), :fixed_bc => _vrd_chk(:pass))
        @test case_verdict(_vrd_rec(fixed_only)) === :pass
        fixed_only[:nll_finite] = _vrd_chk(:fail)
        @test case_verdict(_vrd_rec(fixed_only)) === :attention

        # a harness error with no checks
        rec = _vrd_rec(Dict{Symbol,Any}())
        rec[:harness_error] = "the case could not be run"
        @test case_verdict(rec) === :attention

        # a failure outweighs a ridge pass, and a ridge pass outweighs an allowscalar stage
        @test case_verdict(_vrd_rec(Dict{Symbol,Any}(:fullfit => ridge, :fixed_nll_1 => _vrd_chk(:fail)))) ===
              :attention
        @test case_verdict(_vrd_rec(Dict{Symbol,Any}(:fullfit => ridge, :consumers => consumers))) === :ridge

        # a record whose only statuses are :ok
        @test case_verdict(_vrd_rec(Dict{Symbol,Any}(:spec => _vrd_chk(:ok), :fit_cpu => _vrd_chk(:ok)))) === :pass
    end

    @testset "ridge_pass" begin
        @test ridge_pass(-100.0, -100.0 + 1e-12)
        @test !ridge_pass(-100.0, -100.0 + 1e-6)
        @test !ridge_pass(NaN, NaN)
    end
end
