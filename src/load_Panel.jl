# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

# load_Panel.jl
# Module isolation wrapper for the sf_panel project (Wang and Ho 2010 panel SF).
# Wraps all panel definitions inside Panel_Backend to avoid name conflicts
# with sf_MCI and sf_MSLE.
# The original sf_panel files are NOT modified.

module Panel_Backend

using DataFrames

include(joinpath(@__DIR__, "sf_panel", "sf_panel_v20.jl"))

end # module Panel_Backend
