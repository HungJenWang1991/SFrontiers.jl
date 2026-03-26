# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

# load_Panel.jl
# Module isolation wrapper for the sf_panel project (Wang and Ho 2010 panel SF).
# Wraps all panel definitions inside Panel_Backend to avoid name conflicts
# with sf_MCI and sf_MSLE.
# The original sf_panel files are NOT modified.

module Panel_Backend

using DataFrames

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
