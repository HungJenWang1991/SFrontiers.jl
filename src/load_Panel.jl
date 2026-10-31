# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

# load_Panel.jl
# Module isolation wrapper for the sf_panel project (Wang and Ho 2010 panel SF).
# Wraps all panel definitions inside Panel_Backend to avoid name conflicts
# with sf_MCI and sf_MSLE.

module Panel_Backend

using DataFrames

# The standardized Erlang quantile kernel is shared with the MSLE backend rather
# than duplicated: it is a subtle safeguarded-Newton kernel with a single
# copy (sf_MSLE_v21.jl), and importing two names (the kernel and its draw-clamp
# helper _draw_clamp_hi) cannot create the conflicts the module isolation guards
# against. Order-safe: load_MSLE.jl is included before
# load_Panel.jl in SFrontiers.jl.
using ..MSLE_Backend: myErlangStdQuantile, _draw_clamp_hi

# The shared redflag rule (src/sf_redflag.jl): one Hessian classifier and one
# clamp test for all four backends.
using ..SFRedflag: sf_classify_hessian, sf_scale_at_clamp, sf_final_fg, sf_convergence_flag,
                   SF_BOUND_RTOL, sf_bound_side, SFBoundEntry, sf_bound_entries!, sf_bound_message, sf_bound_move,
                   sf_start_check
# Shared numerical kernels (src/sf_numerics.jl).
using ..SFNumerics: sf_normlogcdf_lowtail, sf_tn_quantile_tail
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

include(joinpath(@__DIR__, "sf_panel", "sf_panel_v20.jl"))

# Flag: set to true when GPU overloads are successfully defined at runtime.
const _gpu_overloads_defined = Ref(false)

# Re-define GPU overloads at runtime (the if-isdefined block in
# sf_panel_v20.jl fires during precompilation when CUDA is absent,
# so the overloads are lost in the precompile cache).
function __init__()
    if isdefined(Main, :CUDA)
        @eval begin
            _maximum_panel(A::Main.CUDA.AnyCuArray; dims) = Main.CUDA.maximum(A; dims=dims)
            _sum_panel(A::Main.CUDA.AnyCuArray; dims)     = Main.CUDA.sum(A; dims=dims)
            _sum_scalar_panel(v::Main.CUDA.AnyCuArray)     = sum(v)
            _to_device_panel(x::AbstractArray, ::Main.CUDA.AnyCuArray) = Main.CUDA.CuArray(x)
            _to_device_panel(x::Main.CUDA.AnyCuArray, ::Main.CUDA.AnyCuArray) = x
            _panel_draw_rows(draws::Main.CUDA.AnyCuArray, a::Int, b::Int) = Array(draws[a:b, :])
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

end # module Panel_Backend
