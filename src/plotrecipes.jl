# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later
#
# RecipesBase-based plotting recipes for SFResult. Consumed automatically
# by any Plots.jl-compatible backend when the user loads `using Plots`.
# No heavy dependency is imposed on users who do not plot.
#
#   plot(result)                 -- default 2x2 diagnostic panel
#   plot(result, :efficiency)    -- JLMS + BC histograms
#   plot(result, :residuals)     -- residuals vs fitted + histogram + QQ
#   plot(result, :marginal)      -- grid of marginal-effect scatters
#   plot(result, :marginal, z)   -- marginal effect vs a named covariate
#   plot(result, :convergence)   -- optimizer trace (if stored)
#
# Note on per-subplot title/xlabel/ylabel: we set these as plot-level
# row-vector attributes (one entry per subplot) rather than inside each
# @series block. This matters when a subplot carries multiple series
# (e.g. a scatter plus a zero-line :hline): attributes set inside the
# secondary @series would otherwise wipe the primary series' labels.

using RecipesBase
using Measures: mm

# ----- Default: 2x2 diagnostic panel ---------------------------------

@recipe function f(r::SFResult)
    yhat = StatsAPI.fitted(r; type = :frontier)
    eps  = StatsAPI.residuals(r; type = :composed)

    has_marg = hasproperty(r, :marginal) && r.marginal !== nothing &&
               !_marginal_empty(r.marginal)

    if has_marg
        colname = _first_marginal_name(r.marginal)
        yvals   = _marginal_column(r.marginal, colname)
        p4_title  = "Marginal effect: $colname"
        p4_xlabel = "observation index"
        p4_ylabel = "∂E(u)/∂z"
    else
        vhat  = vec(StatsAPI.residuals(r; type = :v))
        std_v = _standardize(vhat)
        theo  = _normal_quantiles(length(std_v))
        p4_title  = "QQ: v̂ vs N(0, σ̂_v²)"
        p4_xlabel = "theoretical N(0,1) quantile"
        p4_ylabel = "sample quantile of std. v̂"
    end

    layout        --> (2, 2)
    size          --> (1000, 800)
    legend        --> false
    left_margin   --> 10mm
    bottom_margin --> 8mm
    top_margin    --> 4mm
    right_margin  --> 4mm
    title  --> ["JLMS inefficiency index" "Battese–Coelli efficiency" "Residuals vs fitted frontier" p4_title]
    xlabel --> ["E(u|ε)" "E(exp(-u)|ε)" "x'β̂" p4_xlabel]
    ylabel --> ["frequency" "frequency" "ε̂" p4_ylabel]

    # Panel 1: JLMS histogram
    @series begin
        seriestype := :histogram
        subplot    := 1
        bins       --> 30
        collect(r.jlms)
    end

    # Panel 2: BC histogram
    @series begin
        seriestype := :histogram
        subplot    := 2
        bins       --> 30
        collect(r.bc)
    end

    # Panel 3: residuals vs fitted + zero line
    @series begin
        seriestype := :scatter
        subplot    := 3
        markersize --> 3
        yhat, eps
    end
    @series begin
        seriestype := :hline
        subplot    := 3
        linestyle  := :dash
        linecolor  := :gray
        [0.0]
    end

    # Panel 4: conditional on availability of marginal effects
    if has_marg
        @series begin
            seriestype := :scatter
            subplot    := 4
            markersize --> 3
            1:length(yvals), yvals
        end
        @series begin
            seriestype := :hline
            subplot    := 4
            linestyle  := :dash
            linecolor  := :gray
            [0.0]
        end
    else
        @series begin
            seriestype := :scatter
            subplot    := 4
            markersize --> 3
            theo, sort(std_v)
        end
        @series begin
            seriestype := :path
            subplot    := 4
            linestyle  := :dash
            linecolor  := :gray
            [-3.0, 3.0], [-3.0, 3.0]
        end
    end
end

# ----- :efficiency --------------------------------------------------

@recipe function f(r::SFResult, which::Val{:efficiency})
    layout        --> (1, 2)
    size          --> (1000, 520)
    legend        --> false
    left_margin   --> 10mm
    bottom_margin --> 8mm
    top_margin    --> 4mm
    right_margin  --> 4mm
    title  --> ["JLMS inefficiency" "Battese–Coelli efficiency"]
    xlabel --> ["E(u|ε)" "E(exp(-u)|ε)"]
    ylabel --> ["frequency" "frequency"]

    @series begin
        seriestype := :histogram
        subplot    := 1
        bins       --> 30
        collect(r.jlms)
    end
    @series begin
        seriestype := :histogram
        subplot    := 2
        bins       --> 30
        collect(r.bc)
    end
end

# ----- :residuals ---------------------------------------------------

@recipe function f(r::SFResult, which::Val{:residuals})
    yhat  = StatsAPI.fitted(r; type = :frontier)
    eps   = StatsAPI.residuals(r; type = :composed)
    vhat  = vec(StatsAPI.residuals(r; type = :v))
    std_v = _standardize(vhat)
    theo  = _normal_quantiles(length(std_v))

    layout        --> (1, 3)
    size          --> (1500, 520)
    legend        --> false
    left_margin   --> 10mm
    bottom_margin --> 8mm
    top_margin    --> 4mm
    right_margin  --> 4mm
    title  --> ["Residuals vs fitted" "Residual histogram" "QQ: v̂ vs N(0, σ̂_v²)"]
    xlabel --> ["x'β̂" "ε̂" "theoretical N(0,1) quantile"]
    ylabel --> ["ε̂" "frequency" "sample quantile of std. v̂"]

    @series begin
        seriestype := :scatter
        subplot    := 1
        markersize --> 3
        yhat, eps
    end
    @series begin
        seriestype := :histogram
        subplot    := 2
        bins       --> 30
        eps
    end
    @series begin
        seriestype := :scatter
        subplot    := 3
        markersize --> 3
        theo, sort(std_v)
    end
    @series begin
        seriestype := :path
        subplot    := 3
        linestyle  := :dash
        linecolor  := :gray
        [-3.0, 3.0], [-3.0, 3.0]
    end
end

# ----- :marginal ----------------------------------------------------

@recipe function f(r::SFResult, which::Val{:marginal})
    (!hasproperty(r, :marginal) || r.marginal === nothing || _marginal_empty(r.marginal)) &&
        error("plot(:marginal) requires a model fit with `zvar` and `marginal=true`.")
    cols = _marginal_names(r.marginal)
    n    = length(cols)

    title_row  = reshape([String(c) for c in cols], 1, :)
    xlabel_row = fill("observation index", 1, n)
    ylabel_row = fill("∂E(u)/∂z",          1, n)

    layout        --> (1, n)
    size          --> (440 * n, 520)
    legend        --> false
    left_margin   --> 10mm
    bottom_margin --> 8mm
    top_margin    --> 4mm
    right_margin  --> 4mm
    title  --> title_row
    xlabel --> xlabel_row
    ylabel --> ylabel_row

    for (i, c) in enumerate(cols)
        y = _marginal_column(r.marginal, c)
        @series begin
            seriestype := :scatter
            subplot    := i
            markersize --> 3
            1:length(y), y
        end
        @series begin
            seriestype := :hline
            subplot    := i
            linestyle  := :dash
            linecolor  := :gray
            [0.0]
        end
    end
end

@recipe function f(r::SFResult, which::Val{:marginal}, covar::AbstractVector)
    (!hasproperty(r, :marginal) || r.marginal === nothing || _marginal_empty(r.marginal)) &&
        error("plot(:marginal, covar) requires a model fit with `zvar` and `marginal=true`.")
    cols = _marginal_names(r.marginal)
    isempty(cols) && error("No marginal effects found on the result.")
    y = _marginal_column(r.marginal, first(cols))
    length(covar) == length(y) || error(
        "Covariate length ($(length(covar))) does not match marginal-effect length ($(length(y))).")

    size          --> (720, 520)
    legend        --> false
    left_margin   --> 10mm
    bottom_margin --> 8mm
    top_margin    --> 4mm
    right_margin  --> 4mm
    title  --> "Marginal effect vs covariate"
    xlabel --> "z"
    ylabel --> "∂E(u)/∂z"

    @series begin
        seriestype := :scatter
        markersize --> 3
        covar, y
    end
    @series begin
        seriestype := :hline
        linestyle  := :dash
        linecolor  := :gray
        [0.0]
    end
end

# ----- :convergence -------------------------------------------------

@recipe function f(r::SFResult, which::Val{:convergence})
    if !hasproperty(r, :trace) || r.trace === nothing
        error("plot(:convergence) requires fitting with `store_trace=true` " *
              "so that the optimizer trace is captured.")
    end
    tr = r.trace
    layout        --> (1, 2)
    size          --> (1000, 520)
    legend        --> false
    left_margin   --> 10mm
    bottom_margin --> 8mm
    top_margin    --> 4mm
    right_margin  --> 4mm
    title  --> ["Objective value" "Gradient norm"]
    xlabel --> ["iteration" "iteration"]
    ylabel --> ["log-likelihood" "||∇ℓ||"]

    @series begin
        seriestype := :path
        subplot    := 1
        tr.iteration, tr.value
    end
    @series begin
        seriestype := :path
        subplot    := 2
        yaxis      := :log10
        tr.iteration, tr.g_norm
    end
end

# ----- Symbol-to-Val dispatch convenience --------------------------

@recipe function f(r::SFResult, which::Symbol, args...)
    return (r, Val(which), args...)
end

# ----- Small helpers -----------------------------------------------

_marginal_empty(df) =
    isnothing(df) ||
    (hasmethod(Base.isempty, Tuple{typeof(df)}) && isempty(df)) ||
    (hasproperty(df, :nrow) && df.nrow == 0) ||
    (hasmethod(size, Tuple{typeof(df)}) && size(df, 1) == 0)

function _marginal_names(df)
    try
        return [Symbol(c) for c in propertynames(df) if c != :nrow]
    catch
        return Symbol[]
    end
end

function _first_marginal_name(df)
    ns = _marginal_names(df)
    return isempty(ns) ? :marg_1 : ns[1]
end

function _marginal_column(df, name)
    return collect(getproperty(df, Symbol(name)))
end

function _standardize(x)
    μ = mean(x)
    σ = std(x)
    σ == 0 && return x .- μ
    return (x .- μ) ./ σ
end

function _normal_quantiles(n::Integer)
    # Plotting positions (i - 0.5)/n → z = Φ⁻¹(p). Use the bisection
    # helper already defined in accessors.jl.
    return [_normal_quantile((i - 0.5) / n) for i in 1:n]
end

# ----- Helpful hint when `plot` is called without Plots.jl loaded -----
#
# `plot` is owned by Plots.jl; without `using Plots` it is not in the
# user's namespace, so `plot(result, :efficiency)` raises
# `UndefVarError: plot not defined` before any SFrontiers code can run.
# We cannot intercept that call, but we can attach a hint to the error
# message via Base.Experimental.register_error_hint so the user sees a
# directive to load Plots.jl alongside the standard error.

function __init__()
    Base.Experimental.register_error_hint(UndefVarError) do io, exc
        if exc.var === :plot
            print(io,
                "\nHint: SFrontiers's diagnostic plot recipes require Plots.jl. " *
                "Please load it with `using Plots` (or another Plots-compatible backend).")
        end
    end
end
