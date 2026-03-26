# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

# load_MCI.jl
# Module isolation wrapper for the sf_MCI project.
# Wraps all MCI definitions inside MCI_Backend to avoid name conflicts with sf_MSLE.
# The original sf_MCI files are NOT modified.

module MCI_Backend

using DataFrames

# If CUDA was loaded in Main, bring it into this module so that
# @isdefined(CuArray) inside the backend code returns true.
if isdefined(Main, :CUDA)
    using CUDA
end

include(joinpath(@__DIR__, "sf_MCI", "sf_MCI_v21.jl"))

# Flag: set to true when GPU overloads are successfully defined at runtime.
const _gpu_overloads_defined = Ref(false)

# Re-define GPU overloads at runtime (the if-isdefined block in
# sf_MCI_v21.jl fires during precompilation when CUDA is absent,
# so the overloads are lost in the precompile cache).
function __init__()
    if isdefined(Main, :CUDA)
        @eval begin
            _maximum(A::Main.CUDA.AnyCuArray; dims) = Main.CUDA.maximum(A; dims=dims)
            _sum(A::Main.CUDA.AnyCuArray; dims)     = Main.CUDA.sum(A; dims=dims)
            _sum_scalar(v::Main.CUDA.AnyCuArray)     = sum(v)
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

end # module MCI_Backend
