# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

# load_MSLE.jl
# Module isolation wrapper for the sf_MSLE project.
# Wraps all MSLE definitions inside MSLE_Backend to avoid name conflicts with sf_MCI.

module MSLE_Backend

using DataFrames
# The shared redflag rule (src/sf_redflag.jl): one Hessian classifier and one
# clamp test for all four backends.
using ..SFRedflag: sf_classify_hessian, sf_scale_at_clamp, sf_final_fg, sf_convergence_flag,
                   SF_BOUND_RTOL, sf_bound_side, SFBoundEntry, sf_bound_entries!, sf_bound_message, sf_bound_move,
                   sf_start_check, sf_classify_opg
# Shared numerical kernels (src/sf_numerics.jl).
using ..SFNumerics: sf_tn_quantile_tail, sf_logaddexp, sf_log_neglogcdf_normal, sf_log_neglogcdf_laplace
using ..SFNumerics: sf_tn_mean
# Shared Halton generator (src/sf_draws.jl).
using ..SFDraws: sf_halton_draws, sf_halton_period, sf_draw_scheme, sf_recycling_request, sf_warn_repeated_rows,
                 sf_check_gpu_memory, SF_WORK_ARRAYS
# The optional integration check after a fit (src/sf_accuracy.jl).
using ..SFAccuracy: sf_integration_settings, sf_integration_unavailable, sf_run_integration_check,
                    sf_integration_line, sf_shape_entry, sf_shape_check, sf_warn_lomax_mean, sf_print_warning

# If CUDA was loaded in Main, bring it into this module so that
# @isdefined(CuArray) inside the backend code returns true.
if isdefined(Main, :CUDA)
    using CUDA
end

include(joinpath(@__DIR__, "sf_MSLE", "sf_MSLE_v21.jl"))

# Flag: set to true when GPU overloads are successfully defined at runtime.
const _gpu_overloads_defined = Ref(false)

# Re-define GPU overloads at runtime (the if-isdefined block in
# sf_MSLE_v21.jl fires during precompilation when CUDA is absent,
# so the overloads are lost in the precompile cache).
function __init__()
    if isdefined(Main, :CUDA)
        @eval begin
            _maximum_msle(A::Main.CUDA.AnyCuArray; dims) = Main.CUDA.maximum(A; dims=dims)
            _sum_msle(A::Main.CUDA.AnyCuArray; dims)     = Main.CUDA.sum(A; dims=dims)
            _sum_scalar_msle(v::Main.CUDA.AnyCuArray)     = sum(v)
        end
        _gpu_overloads_defined[] = true
    end
end

"""
    check_gpu_overloads()

Call this when `GPU=true` is requested.  Gives an informative error if CUDA
was loaded *after* SFrontiers (so the `__init__` overloads were missed).
"""
function check_gpu_overloads()
    if !_gpu_overloads_defined[]
        if isdefined(Main, :CUDA)
            error("CUDA.jl was loaded after SFrontiers. " *
                  "Please load CUDA first:\n" *
                  "    using CUDA\n" *
                  "    using SFrontiers\n" *
                  "Then restart Julia and try again.")
        else
            error("GPU=true requires CUDA.jl. " *
                  "Please run `using CUDA` before `using SFrontiers`.")
        end
    end
end

end # module MSLE_Backend
