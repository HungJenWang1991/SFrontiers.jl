# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

# load_MCI.jl
# Module isolation wrapper for the sf_MCI project.
# Wraps all MCI definitions inside MCI_Backend to avoid name conflicts with sf_MSLE.
# The original sf_MCI files are NOT modified.

module MCI_Backend

using DataFrames

include(joinpath(@__DIR__, "sf_MCI", "sf_MCI_v21.jl"))

end # module MCI_Backend
