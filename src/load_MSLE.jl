# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

# load_MSLE.jl
# Module isolation wrapper for the sf_MSLE project.
# Wraps all MSLE definitions inside MSLE_Backend to avoid name conflicts with sf_MCI.
# The original sf_MSLE files are NOT modified.

module MSLE_Backend

using DataFrames

include(joinpath(@__DIR__, "sf_MSLE", "sf_MSLE_v21.jl"))

end # module MSLE_Backend
