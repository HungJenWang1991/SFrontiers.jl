# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

# load_MLE.jl
# Module isolation wrapper for the sf_MLE project (analytic MLE estimation).
# Wraps all MLE definitions inside MLE_Backend to avoid name conflicts with
# sf_MCI, sf_MSLE, and sf_panel.
# The original sf_MLE files are NOT modified.

module MLE_Backend

using DataFrames

include(joinpath(@__DIR__, "sf_MLE", "SFmle.jl"))
using .SFmle

end # module MLE_Backend
