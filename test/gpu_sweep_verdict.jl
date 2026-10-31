# ---------------------------------------------------------------------------
# The verdict rules of the CPU-GPU equivalence sweep (README: test/GPU_VALIDATION.md).
#
# No CUDA here: gpu_equivalence_lib.jl includes this file for the sweep runner,
# and test/test_gpu_sweep_verdict.jl tests the rules on the CPU.
# ---------------------------------------------------------------------------

# Two correct fits that stop at different points of a flat ridge agree to well below 1e-8 nats
# (1e-13 to 1e-9 on an RTX 5090).
const ATOL_RIDGE = 1e-8

"""
    ridge_pass(ll_a, ll_b; atol = ATOL_RIDGE)

`true` when the log-likelihoods `ll_a` and `ll_b` are both finite and differ by at most `atol`
nats (absolute).
"""
ridge_pass(ll_a, ll_b; atol = ATOL_RIDGE) = isfinite(ll_a) && isfinite(ll_b) && abs(ll_a - ll_b) <= atol

"""
    case_verdict(rec)

The verdict of one case of the sweep from its record `rec`, by the statuses of every nested Dict
of `rec[:checks]` that has a `:status`:
- `:attention` when the record has a `:harness_error`, or when any status is `:fail` or `:error`;
- otherwise `:ridge` when any status is `:pass_ridge` (coefficients that differ, with equal
  log-likelihoods at both optima on each device);
- otherwise `:allowscalar` when any status is `:ok_with_allowscalar` (a post-fit stage that
  passed under `CUDA.@allowscalar`);
- otherwise `:pass`. The statuses `:ok`, `:pass`, `:flags_only` and `:skipped` are neutral.
"""
function case_verdict(rec)
    haskey(rec, :harness_error) && return :attention
    found = Set{Any}()
    function scan(d)
        for (_, v) in d
            v isa AbstractDict || continue
            haskey(v, :status) && push!(found, v[:status])
            scan(v)
        end
    end
    scan(get(rec, :checks, Dict{Symbol,Any}()))
    (:fail in found || :error in found) && return :attention
    :pass_ridge in found && return :ridge
    :ok_with_allowscalar in found && return :allowscalar
    return :pass
end
