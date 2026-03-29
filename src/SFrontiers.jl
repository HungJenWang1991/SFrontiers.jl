# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

module SFrontiers

using DataFrames

# ============================================================================
# Section 1: Load backends into isolated modules
# ============================================================================

include(joinpath(@__DIR__, "load_MCI.jl"))
include(joinpath(@__DIR__, "load_MSLE.jl"))
include(joinpath(@__DIR__, "load_MLE.jl"))
include(joinpath(@__DIR__, "load_Panel.jl"))

# ============================================================================
# Section 2: Wrapper types
# ============================================================================

"""
    MLEMethodSpec

Marker type for analytic MLE estimation. MLE does not require simulation
parameters (draws, GPU, chunks), so this is a simple empty struct used
for dispatch in `sfmodel_fit()`.
"""
struct MLEMethodSpec end

"""
    UnifiedMethod

Type alias for the union of all backends' method specification types.
The concrete type encodes which backend to use, enabling zero-cost dispatch.
"""
const UnifiedMethod = Union{MCI_Backend.SFMethodSpec, MSLE_Backend.SFMethodSpec_MSLE, MLEMethodSpec, Panel_Backend.PanelMethodSpec}

"""
    UnifiedSpec{T<:AbstractFloat}

Wrapper holding model specifications from one or more backends.
Created by `sfmodel_spec()`.

When `datatype=:cross_sectional` (default): builds MCI spec (always), MSLE spec
(unless `ineff=:Gamma`), and MLE spec (when noise=:Normal, no copula, and
ineff ∈ {:HalfNormal, :TruncatedNormal, :Exponential}).

When `datatype=:panel_TFE`: builds Panel spec (for MCI/MSLE) and optionally MLE spec.
When `datatype=:panel_TFE_CSW` or `:panel_TRE`: builds MLE spec only.

# Fields
- `datatype`: `:cross_sectional`, `:panel_TFE`, `:panel_TFE_CSW`, or `:panel_TRE`
- `mci_spec`: MCI backend specification
- `msle_spec`: MSLE backend specification
- `mle_spec`: MLE backend specification (analytic maximum likelihood)
- `panel_spec`: Panel backend specification (Wang and Ho 2010, simulation-based)
- `noise`, `ineff`, `copula`, `hetero`: Stored for informational access
"""
struct UnifiedSpec{T<:AbstractFloat}
    datatype::Symbol
    mci_spec::Union{MCI_Backend.SFModelSpec{T}, Nothing}
    msle_spec::Union{MSLE_Backend.SFModelSpec_MSLE{T}, Nothing}
    mle_spec::Union{MLE_Backend.SFModelSpec_MLE{T}, Nothing}
    panel_spec::Union{Panel_Backend.PanelModelSpec{T}, Nothing}
    noise::Symbol
    ineff::Symbol
    copula::Symbol
    hetero::Union{Vector{Symbol}, Symbol}
end

# ============================================================================
# Section 3: DSL macros (DataFrame-based specification)
# ============================================================================

using DataFrames

struct WUseDataSpec
    df::DataFrame
end

struct WDepvarSpec
    name::Symbol
end

struct WFrontierSpec
    names::Vector{Symbol}
end

struct WZvarSpec
    names::Vector{Symbol}
end

"""
    @useData(df)

Mark a DataFrame as the data source for the model specification.
"""
macro useData(df)
    :(WUseDataSpec($(esc(df))))
end

"""
    @depvar(varname)

Specify the dependent variable column name from the DataFrame.
"""
macro depvar(var)
    :(WDepvarSpec($(QuoteNode(var))))
end

"""
    @frontier(var1, var2, ...)

Specify the frontier variable column names from the DataFrame.
"""
macro frontier(vars...)
    names = [QuoteNode(v) for v in vars]
    :(WFrontierSpec(Symbol[$(names...)]))
end

"""
    @zvar(var1, var2, ...)

Specify the Z variable column names from the DataFrame.
"""
macro zvar(vars...)
    names = [QuoteNode(v) for v in vars]
    :(WZvarSpec(Symbol[$(names...)]))
end

struct WIdSpec
    name::Symbol
end

"""
    @id(varname)

Specify the panel unit identifier column name from the DataFrame (unbalanced panels).
"""
macro id(var)
    :(WIdSpec($(QuoteNode(var))))
end

# Union type for all DSL macro wrapper types (allows arbitrary macro ordering)
const DSLArg = Union{WUseDataSpec, WDepvarSpec, WFrontierSpec, WZvarSpec, WIdSpec}

# ============================================================================
# Section 4: sfmodel_spec — Unified model specification
# ============================================================================

"""
    sfmodel_spec(; depvar, frontier, zvar=nothing, noise, ineff, copula=:None,
                 hetero=Symbol[], datatype=:cross_sectional,
                 T_periods=nothing, id=nothing,
                 varnames=nothing, eqnames=nothing, eq_indices=nothing, type=:prod)

Construct a unified model specification.

When `datatype=:cross_sectional` (default): calls MCI, MSLE, and MLE backends internally
as applicable. MCI supports all distributions; MSLE excludes Gamma; MLE requires
Normal noise, no copula, and ineff ∈ {:HalfNormal, :TruncatedNormal, :Exponential}.

When `datatype=:panel_TFE`: Wang and Ho (2010) true fixed-effect model. Builds Panel
spec (for MCI/MSLE) and MLE spec (when ineff is HalfNormal or TruncatedNormal).

When `datatype=:panel_TFE_CSW`: Chen, Schmidt, and Wang (2014) fixed-effect model.
MLE only, requires `ineff=:HalfNormal`.

When `datatype=:panel_TRE`: True random-effect model. MLE only, requires
`ineff=:HalfNormal` or `:TruncatedNormal`.

Panel models require `id` to specify panel structure. For balanced panels, use
`id = repeat(1:N, inner=T)`. Panel models do not support `copula` or `hetero`;
do NOT include constant columns in `frontier` or `zvar` (within-demeaning eliminates them).

Returns a `UnifiedSpec{T}` that can be used with `sfmodel_fit()`.

# Arguments
- `depvar`: Response vector. Cross-sectional: N obs. Panel: N*T stacked by firm.
- `frontier`: Frontier design matrix. Cross-sectional: N×K. Panel: NT×K (no constant).
- `zvar=nothing`: Z variable matrix. Cross-sectional: N×L. Panel: NT×L (no constant).
- `noise::Symbol`: Noise distribution. Cross-sectional: `:Normal`, `:StudentT`, `:Laplace`.
  Panel: `:Normal` only.
- `ineff::Symbol`: Inefficiency distribution (`:HalfNormal`, `:TruncatedNormal`, `:Exponential`,
  `:Weibull`, `:Lognormal`, `:Lomax`, `:Rayleigh`, `:Gamma`)
- `copula::Symbol=:None`: Copula (cross-sectional only). Not available for panel models.
- `hetero::Vector{Symbol}=Symbol[]`: Heteroscedasticity (cross-sectional only). Not available for panel models.
- `datatype::Symbol=:cross_sectional`: Data type. `:cross_sectional`, `:panel_TFE`,
  `:panel_TFE_CSW`, or `:panel_TRE`.
- `T_periods::Union{Int,Nothing}=nothing`: Periods per firm (balanced panel).
- `id=nothing`: Unit identifier column (unbalanced panel).
- `varnames=nothing`: Variable names (auto-generated if not provided)
- `eqnames=nothing`: Equation names (auto-generated if not provided)
- `eq_indices=nothing`: Equation indices (auto-generated if not provided)
- `type::Symbol=:prod`: Frontier type (`:prod`, `:production`, or `:cost`)
"""
function sfmodel_spec(; depvar, frontier, zvar=nothing,
                noise::Symbol, ineff::Symbol,
                copula::Symbol=:None,
                hetero::Union{Vector{Symbol}, Symbol}=Symbol[],
                datatype::Symbol=:cross_sectional,
                T_periods::Union{Int,Nothing}=nothing,
                id=nothing,
                varnames::Union{Nothing, Vector{String}}=nothing,
                eqnames::Union{Nothing, Vector{String}}=nothing,
                eq_indices::Union{Nothing, Vector{Int}}=nothing,
                type::Symbol=:prod)

    # --- Helpers: validate copula/hetero for panel models ---
    _validate_panel_no_copula = () -> begin
        copula != :None && error("Panel models do not support `copula`. Remove the `copula` argument.")
    end
    _validate_panel_hetero_scaling_only = () -> begin
        # panel_TFE: hetero=:scaling is the only permissible option (and the default)
        if hetero === :scaling
            # OK — explicit :scaling accepted
        elseif hetero isa Symbol
            error("Panel TFE models only support `hetero=:scaling` (the default). Got `hetero=:$hetero`.")
        elseif !isempty(hetero)
            error("Panel TFE models only support `hetero=:scaling` (the default). " *
                "Individual heteroscedastic parameters (e.g., [:mu, :sigma_sq]) are not available for panel models.")
        end
    end
    _validate_panel_no_hetero = () -> begin
        # panel_TFE_CSW and panel_TRE: no hetero at all
        _has_hetero = hetero isa Symbol ? true : !isempty(hetero)
        _has_hetero && error("This panel model does not support `hetero`. Remove the `hetero` argument.")
    end

    # --- Helper: build MLE spec for cross-sectional models (returns nothing if not applicable) ---
    _build_mle_cross = () -> begin
        # MLE scaling only supports TruncatedNormal; skip for other distributions
        if hetero === :scaling && ineff != :TruncatedNormal
            return nothing
        end
        if noise == :Normal && copula == :None &&
           ineff in (:HalfNormal, :TruncatedNormal, :Exponential)
            MLE_Backend.sfmodel_spec(;
                depvar=depvar, frontier=frontier, zvar=zvar,
                noise=:Normal, ineff=ineff, hetero=hetero, type=type)
        else
            nothing
        end
    end

    if datatype == :panel_TFE
        # Wang and Ho (2010) true fixed-effect model
        # Supported by Panel_Backend (MCI/MSLE) and MLE_Backend (HalfNormal/TruncatedNormal)
        _validate_panel_no_copula()
        _validate_panel_hetero_scaling_only()
        !isnothing(T_periods) && error("T_periods is no longer supported. Use id instead. For balanced panels: id = repeat(1:N, inner=T).")

        # Always build Panel spec (for MCI/MSLE — supports all 8 distributions)
        panel_spec = Panel_Backend.sfmodel_panel_spec(;
            depvar=depvar, frontier=frontier, zvar=zvar,
            id=id,
            noise=noise, ineff=ineff, type=type,
            varnames=varnames, eqnames=eqnames, eq_indices=eq_indices)

        # Build MLE spec when applicable (Normal noise + HalfNormal/TruncatedNormal)
        mle_spec = if noise == :Normal && ineff in (:HalfNormal, :TruncatedNormal)
            MLE_Backend.sfmodel_spec(;
                depvar=depvar, frontier=frontier, zvar=zvar,
                ineff=ineff, panel=:TFE_WH2010, id=id, type=type)
        else
            nothing
        end

        T_el = eltype(panel_spec.depvar)
        return UnifiedSpec{T_el}(:panel_TFE, nothing, nothing, mle_spec, panel_spec,
                                 noise, ineff, :None, Symbol[])

    elseif datatype == :panel_TFE_CSW
        # Chen, Schmidt, and Wang (2014) true fixed-effect model — MLE only
        _validate_panel_no_copula()
        _validate_panel_no_hetero()
        !isnothing(T_periods) && error("T_periods is no longer supported. Use id instead. For balanced panels: id = repeat(1:N, inner=T).")
        noise == :Normal || error("Panel TFE_CSW models require `noise=:Normal`.")
        ineff == :HalfNormal || error("Panel TFE_CSW models require `ineff=:HalfNormal`.")

        mle_spec = MLE_Backend.sfmodel_spec(;
            depvar=depvar, frontier=frontier,
            ineff=:HalfNormal, panel=:TFE_CSW2014, id=id, type=type)

        T_el = eltype(mle_spec.depvar)
        return UnifiedSpec{T_el}(:panel_TFE_CSW, nothing, nothing, mle_spec, nothing,
                                 noise, ineff, :None, Symbol[])

    elseif datatype == :panel_TRE
        # True random-effect model — MLE only
        _validate_panel_no_copula()
        _validate_panel_no_hetero()
        !isnothing(T_periods) && error("T_periods is no longer supported. Use id instead. For balanced panels: id = repeat(1:N, inner=T).")
        noise == :Normal || error("Panel TRE models require `noise=:Normal`.")
        ineff in (:HalfNormal, :TruncatedNormal) || error(
            "Panel TRE models require `ineff=:HalfNormal` or `:TruncatedNormal`. " *
            "Got `ineff=:$ineff`.")

        mle_spec = MLE_Backend.sfmodel_spec(;
            depvar=depvar, frontier=frontier, zvar=zvar,
            ineff=ineff, panel=:TRE, id=id, type=type)

        T_el = eltype(mle_spec.depvar)
        return UnifiedSpec{T_el}(:panel_TRE, nothing, nothing, mle_spec, nothing,
                                 noise, ineff, :None, Symbol[])

    elseif datatype == :cross_sectional

        if hetero === :scaling
            # Scaling property model: build MCI, MSLE, and MLE specs
            mci_spec = MCI_Backend.sfmodel_spec(;
                depvar=depvar, frontier=frontier, zvar=zvar,
                noise=noise, ineff=ineff, copula=copula, hetero=:scaling,
                varnames=varnames, eqnames=eqnames,
                eq_indices=eq_indices, type=type)

            msle_spec = if ineff == :Gamma
                nothing
            else
                MSLE_Backend.sfmodel_spec(;
                    depvar=depvar, frontier=frontier, zvar=zvar,
                    noise=noise, ineff=ineff, copula=copula, hetero=:scaling,
                    varnames=varnames, eqnames=eqnames,
                    eq_indices=eq_indices, type=type)
            end

            # MLE scaling requires TruncatedNormal (handled internally by MLE_Backend)
            mle_spec = _build_mle_cross()

            T = eltype(mci_spec.depvar)
            return UnifiedSpec{T}(:cross_sectional, mci_spec, msle_spec, mle_spec, nothing,
                                  noise, ineff, copula, :scaling)
        end

        # Standard (non-scaling) cross-sectional path
        # Always build MCI spec (supports all distributions including Gamma)
        mci_spec = MCI_Backend.sfmodel_spec(;
            depvar=depvar, frontier=frontier, zvar=zvar,
            noise=noise, ineff=ineff, copula=copula, hetero=hetero,
            varnames=varnames, eqnames=eqnames,
            eq_indices=eq_indices, type=type)

        # Build MSLE spec only if ineff is not :Gamma (MSLE does not support Gamma)
        msle_spec = if ineff == :Gamma
            nothing
        else
            MSLE_Backend.sfmodel_spec(;
                depvar=depvar, frontier=frontier, zvar=zvar,
                noise=noise, ineff=ineff, copula=copula, hetero=hetero,
                varnames=varnames, eqnames=eqnames,
                eq_indices=eq_indices, type=type)
        end

        # Build MLE spec when applicable
        mle_spec = _build_mle_cross()

        T = eltype(mci_spec.depvar)
        return UnifiedSpec{T}(:cross_sectional, mci_spec, msle_spec, mle_spec, nothing,
                              noise, ineff, copula, hetero)

    else
        error("Unknown `datatype`: :$datatype. " *
              "Use `:cross_sectional` (default), `:panel_TFE`, `:panel_TFE_CSW`, or `:panel_TRE`.")
    end
end

"""
    sfmodel_spec(args::DSLArg...; noise, ineff, kwargs...)

DSL-style model specification using DataFrame column names.
Macros can appear in **any order**.

**Required macros:** `@useData(df)`, `@depvar(y)`, `@frontier(x1, x2, ...)`
**Optional macros:** `@zvar(z1, z2, ...)`, `@id(idvar)` (panel only)

# Examples
```julia
# Cross-sectional (macros in any order)
spec = sfmodel_spec(@depvar(yvar), @useData(df), @frontier(cons, x1, x2), @zvar(cons, z1);
    noise=:Normal, ineff=:TruncatedNormal, hetero=[:mu])

# Panel TFE (with @id)
spec = sfmodel_spec(@id(firm), @useData(df), @depvar(y), @frontier(x1, x2), @zvar(z1);
    noise=:Normal, ineff=:HalfNormal, datatype=:panel_TFE)

# Panel TRE (with @id)
spec = sfmodel_spec(@useData(df), @depvar(y), @frontier(cons, x1, x2), @id(firm);
    noise=:Normal, ineff=:HalfNormal, datatype=:panel_TRE)
```
"""
function sfmodel_spec(args::DSLArg...;
                      noise::Symbol, ineff::Symbol,
                      copula::Symbol=:None,
                      hetero::Union{Vector{Symbol}, Symbol}=Symbol[],
                      datatype::Symbol=:cross_sectional,
                      T_periods::Union{Int,Nothing}=nothing,
                      eqnames::Union{Nothing, Vector{String}}=nothing,
                      eq_indices::Union{Nothing, Vector{Int}}=nothing,
                      type::Symbol=:prod)

    # --- Extract each component by type ---
    data    = nothing
    dv      = nothing
    fr      = nothing
    zv      = nothing
    idspec  = nothing

    for arg in args
        if arg isa WUseDataSpec
            !isnothing(data)   && error("Duplicate @useData specification.")
            data = arg
        elseif arg isa WDepvarSpec
            !isnothing(dv)     && error("Duplicate @depvar specification.")
            dv = arg
        elseif arg isa WFrontierSpec
            !isnothing(fr)     && error("Duplicate @frontier specification.")
            fr = arg
        elseif arg isa WZvarSpec
            !isnothing(zv)     && error("Duplicate @zvar specification.")
            zv = arg
        elseif arg isa WIdSpec
            !isnothing(idspec) && error("Duplicate @id specification.")
            idspec = arg
        end
    end

    # --- Validate required macros ---
    isnothing(data) && error("@useData is required in DSL-style specification.")
    isnothing(dv)   && error("@depvar is required in DSL-style specification.")
    isnothing(fr)   && error("@frontier is required in DSL-style specification.")

    # --- Helpers: validate copula/hetero for panel models (DSL) ---
    _validate_panel_no_copula_dsl = () -> begin
        copula != :None && error("Panel models do not support `copula`.")
    end
    _validate_panel_hetero_scaling_only_dsl = () -> begin
        if hetero === :scaling
            # OK — explicit :scaling accepted for panel_TFE
        elseif hetero isa Symbol
            error("Panel TFE models only support `hetero=:scaling` (the default). Got `hetero=:$hetero`.")
        elseif !isempty(hetero)
            error("Panel TFE models only support `hetero=:scaling` (the default). " *
                "Individual heteroscedastic parameters (e.g., [:mu, :sigma_sq]) are not available for panel models.")
        end
    end
    _validate_panel_no_hetero_dsl = () -> begin
        _has_hetero = hetero isa Symbol ? true : !isempty(hetero)
        _has_hetero && error("This panel model does not support `hetero`. Remove the `hetero` argument.")
    end

    # --- Helper: build Panel spec from DSL args ---
    _build_panel_spec_dsl = () -> begin
        !isnothing(T_periods) && error("T_periods is no longer supported. Use @id instead. For balanced panels, add an id column to your DataFrame.")

        data_p = Panel_Backend.UseDataSpec(data.df)
        dv_p   = Panel_Backend.DepvarSpec(dv.name)
        fr_p   = Panel_Backend.FrontierSpec(fr.names)

        if !isnothing(idspec)
            id_p = Panel_Backend.IdSpec(idspec.name)
            if !isnothing(zv)
                Panel_Backend.sfmodel_panel_spec(
                    data_p, dv_p, fr_p,
                    Panel_Backend.ZvarSpec(zv.names), id_p;
                    noise=noise, ineff=ineff, type=type)
            else
                Panel_Backend.sfmodel_panel_spec(
                    data_p, dv_p, fr_p, id_p;
                    noise=noise, ineff=ineff, type=type)
            end
        else
            error("Panel models require @id(...). For balanced panels, add an id column to your DataFrame.")
        end
    end

    # --- Helper: extract DataFrame arrays for MLE keyword API ---
    _dsl_extract_data = () -> begin
        depvar_data   = Vector{Float64}(data.df[!, dv.name])
        frontier_data = Matrix{Float64}(data.df[!, fr.names])
        zvar_data = !isnothing(zv) ? Matrix{Float64}(data.df[!, zv.names]) : nothing
        id_data = !isnothing(idspec) ? Vector(data.df[!, idspec.name]) : nothing
        (depvar_data, frontier_data, zvar_data, id_data)
    end

    # --- Helper: build MLE spec for cross-sectional DSL ---
    _build_mle_cross_dsl = () -> begin
        if noise == :Normal && copula == :None &&
           ineff in (:HalfNormal, :TruncatedNormal, :Exponential)
            (depvar_data, frontier_data, zvar_data, _) = _dsl_extract_data()
            MLE_Backend.sfmodel_spec(;
                depvar=depvar_data, frontier=frontier_data, zvar=zvar_data,
                noise=:Normal, ineff=ineff, hetero=hetero, type=type)
        else
            nothing
        end
    end

    # --- Dispatch based on datatype ---
    if datatype == :panel_TFE
        _validate_panel_no_copula_dsl()
        _validate_panel_hetero_scaling_only_dsl()

        # Build Panel spec (for MCI/MSLE)
        panel_spec = _build_panel_spec_dsl()

        # Build MLE spec when applicable
        mle_spec = if noise == :Normal && ineff in (:HalfNormal, :TruncatedNormal)
            (depvar_data, frontier_data, zvar_data, id_data) = _dsl_extract_data()
            MLE_Backend.sfmodel_spec(;
                depvar=depvar_data, frontier=frontier_data, zvar=zvar_data,
                ineff=ineff, panel=:TFE_WH2010, id=id_data, type=type)
        else
            nothing
        end

        T_el = eltype(panel_spec.depvar)
        return UnifiedSpec{T_el}(:panel_TFE, nothing, nothing, mle_spec, panel_spec,
                                 noise, ineff, :None, Symbol[])

    elseif datatype == :panel_TFE_CSW
        _validate_panel_no_copula_dsl()
        _validate_panel_no_hetero_dsl()
        noise == :Normal || error("Panel TFE_CSW models require `noise=:Normal`.")
        ineff == :HalfNormal || error("Panel TFE_CSW models require `ineff=:HalfNormal`.")

        (depvar_data, frontier_data, _, id_data) = _dsl_extract_data()
        mle_spec = MLE_Backend.sfmodel_spec(;
            depvar=depvar_data, frontier=frontier_data,
            ineff=:HalfNormal, panel=:TFE_CSW2014, id=id_data, type=type)

        T_el = eltype(mle_spec.depvar)
        return UnifiedSpec{T_el}(:panel_TFE_CSW, nothing, nothing, mle_spec, nothing,
                                 noise, ineff, :None, Symbol[])

    elseif datatype == :panel_TRE
        _validate_panel_no_copula_dsl()
        _validate_panel_no_hetero_dsl()
        noise == :Normal || error("Panel TRE models require `noise=:Normal`.")
        ineff in (:HalfNormal, :TruncatedNormal) || error(
            "Panel TRE models require `ineff=:HalfNormal` or `:TruncatedNormal`. " *
            "Got `ineff=:$ineff`.")

        (depvar_data, frontier_data, zvar_data, id_data) = _dsl_extract_data()
        mle_spec = MLE_Backend.sfmodel_spec(;
            depvar=depvar_data, frontier=frontier_data, zvar=zvar_data,
            ineff=ineff, panel=:TRE, id=id_data, type=type)

        T_el = eltype(mle_spec.depvar)
        return UnifiedSpec{T_el}(:panel_TRE, nothing, nothing, mle_spec, nothing,
                                 noise, ineff, :None, Symbol[])

    elseif datatype == :cross_sectional
        !isnothing(idspec) && datatype == :cross_sectional &&
            error("`@id()` is only valid for panel models (e.g., `datatype=:panel_TFE`).")

        kw = (noise=noise, ineff=ineff, copula=copula, hetero=hetero,
              eqnames=eqnames, eq_indices=eq_indices, type=type)

        if hetero === :scaling
            # Scaling property model: build MCI, MSLE, and MLE specs
            data_m = MCI_Backend.UseDataSpec(data.df)
            dv_m   = MCI_Backend.DepvarSpec(dv.name)
            fr_m   = MCI_Backend.FrontierSpec(fr.names)
            mci_spec = if !isnothing(zv)
                MCI_Backend.sfmodel_spec(
                    data_m, dv_m, fr_m, MCI_Backend.ZvarSpec(zv.names); kw...)
            else
                MCI_Backend.sfmodel_spec(data_m, dv_m, fr_m; kw...)
            end

            msle_spec = if ineff == :Gamma
                nothing
            else
                data_s = MSLE_Backend.UseDataSpec(data.df)
                dv_s   = MSLE_Backend.DepvarSpec(dv.name)
                fr_s   = MSLE_Backend.FrontierSpec(fr.names)
                if !isnothing(zv)
                    MSLE_Backend.sfmodel_spec(
                        data_s, dv_s, fr_s, MSLE_Backend.ZvarSpec(zv.names); kw...)
                else
                    MSLE_Backend.sfmodel_spec(data_s, dv_s, fr_s; kw...)
                end
            end

            mle_spec = _build_mle_cross_dsl()

            T = eltype(mci_spec.depvar)
            return UnifiedSpec{T}(:cross_sectional, mci_spec, msle_spec, mle_spec, nothing,
                                  noise, ineff, copula, :scaling)
        end

        # Standard (non-scaling) cross-sectional path
        data_m = MCI_Backend.UseDataSpec(data.df)
        dv_m   = MCI_Backend.DepvarSpec(dv.name)
        fr_m   = MCI_Backend.FrontierSpec(fr.names)

        mci_spec = if !isnothing(zv)
            MCI_Backend.sfmodel_spec(
                data_m, dv_m, fr_m, MCI_Backend.ZvarSpec(zv.names); kw...)
        else
            MCI_Backend.sfmodel_spec(data_m, dv_m, fr_m; kw...)
        end

        msle_spec = if ineff == :Gamma
            nothing
        else
            data_s = MSLE_Backend.UseDataSpec(data.df)
            dv_s   = MSLE_Backend.DepvarSpec(dv.name)
            fr_s   = MSLE_Backend.FrontierSpec(fr.names)
            if !isnothing(zv)
                MSLE_Backend.sfmodel_spec(
                    data_s, dv_s, fr_s, MSLE_Backend.ZvarSpec(zv.names); kw...)
            else
                MSLE_Backend.sfmodel_spec(data_s, dv_s, fr_s; kw...)
            end
        end

        mle_spec = _build_mle_cross_dsl()

        T = eltype(mci_spec.depvar)
        return UnifiedSpec{T}(:cross_sectional, mci_spec, msle_spec, mle_spec, nothing,
                              noise, ineff, copula, hetero)

    else
        error("Unknown `datatype`: :$datatype. " *
              "Use `:cross_sectional` (default), `:panel_TFE`, `:panel_TFE_CSW`, or `:panel_TRE`.")
    end
end

# ============================================================================
# Section 5: sfmodel_method — Unified method specification
# ============================================================================

"""
    sfmodel_method(; method::Symbol, transformation=nothing, draws=nothing,
                   n_draws=nothing, multiRand=true, GPU=false, chunks=10, distinct_Halton_length=2^15-1)

Specify the estimation method.

# Arguments
- `method::Symbol`: `:MCI` for Monte Carlo Integration, `:MSLE` for Maximum Simulated
  Likelihood, or `:MLE` for analytic Maximum Likelihood Estimation.
- `transformation=nothing`: Transformation rule (MCI only).
- `draws=nothing`: User-supplied Halton draws (MCI/MSLE only). Auto-generated if `nothing`.
- `n_draws=nothing`: Number of quasi-random draws (MCI/MSLE only). Default: 1024.
- `multiRand::Bool=true`: Per-observation draws (MCI/MSLE only).
- `GPU::Bool=false`: Use GPU acceleration (MCI/MSLE only).
- `chunks::Int=10`: Memory chunking for GPU (MCI/MSLE only).
- `distinct_Halton_length::Int=2^15-1`: Maximum Halton sequence length (MCI/MSLE only).

# Returns
- `MCI_Backend.SFMethodSpec` if `method=:MCI`
- `MSLE_Backend.SFMethodSpec_MSLE` if `method=:MSLE`
- `MLEMethodSpec` if `method=:MLE`
"""
function sfmodel_method(;
    method::Symbol,
    transformation::Union{Symbol,Nothing} = nothing,
    draws::Union{Nothing, AbstractVecOrMat{<:AbstractFloat}} = nothing,
    n_draws::Union{Int,Nothing} = nothing,
    multiRand::Bool = true,
    GPU::Bool = false,
    chunks::Int = 10,
    distinct_Halton_length::Int = 2^15-1)

    # Early check: if GPU requested, ensure CUDA was loaded before SFrontiers
    if GPU
        if method == :MCI
            MCI_Backend.check_gpu_overloads()
        elseif method == :MSLE
            MSLE_Backend.check_gpu_overloads()
        end
    end

    if method == :MCI
        _n_draws = isnothing(n_draws) ? 1024 : n_draws
        return MCI_Backend.sfmodel_method(;
            method=:MCI,
            transformation=transformation,
            draws=draws,
            n_draws=_n_draws,
            multiRand=multiRand,
            GPU=GPU,
            chunks=chunks,
            distinct_Halton_length=distinct_Halton_length)

    elseif method == :MSLE
        if !isnothing(transformation)
            @warn "`transformation` is only used by method=:MCI. Ignored for method=:MSLE."
        end
        _n_draws = isnothing(n_draws) ? 1024 : n_draws
        return MSLE_Backend.sfmodel_method(;
            method=:MSLE,
            draws=draws,
            n_draws=_n_draws,
            multiRand=multiRand,
            GPU=GPU,
            chunks=chunks,
            distinct_Halton_length=distinct_Halton_length)

    elseif method == :MLE
        # MLE is analytic — simulation arguments are not applicable
        _has_sim_args = !isnothing(transformation) || !isnothing(draws) || !isnothing(n_draws) || GPU
        if _has_sim_args
            @warn "Simulation arguments (transformation, draws, n_draws, GPU, chunks) " *
                  "are not used by `method=:MLE` and will be ignored."
        end
        return MLEMethodSpec()

    else
        error("Unknown method: :$method. Use `:MCI`, `:MSLE`, or `:MLE`.")
    end
end

# ============================================================================
# Section 6: sfmodel_init — Unified initial values
# ============================================================================

# Helper to resolve a method argument to a Symbol (:MCI, :MSLE, :MLE, or nothing)
_resolve_method(::Nothing) = nothing
_resolve_method(m::Symbol)  = m
_resolve_method(::MCI_Backend.SFMethodSpec)        = :MCI
_resolve_method(::MSLE_Backend.SFMethodSpec_MSLE)  = :MSLE
_resolve_method(::MLEMethodSpec)                   = :MLE
_resolve_method(::Panel_Backend.PanelMethodSpec)    = :Panel

"""
    sfmodel_init(; spec::UnifiedSpec, method=nothing, kwargs...)

Compute or assemble initial values for the estimation.

For cross-sectional models, `method` controls which backend is used for initial values.
It accepts a `Symbol` (`:MCI`, `:MSLE`, `:MLE`) or a method object returned by
`sfmodel_method()`. If omitted, the backend is chosen automatically based on spec
availability (MCI > MSLE > MLE).

**Cross-sectional keyword arguments (MCI/MSLE):** `init`, `frontier`, `mu`,
`ln_sigma_sq`, `ln_sigma_v_sq`, `ln_nu_minus_2`, `ln_b`, `ln_lambda`, `ln_k`,
`ln_alpha`, `ln_theta` (Gamma only), `theta_rho` (copula), `message`.

**Cross-sectional keyword arguments (MLE):** `init`, `frontier`, `mu`,
`ln_sigma_u_sq`, `ln_sigma_v_sq`, `hscale`, `message`.

**Panel TFE keyword arguments (Panel_Backend):** `init`, `frontier`, `delta`, `mu`,
`ln_sigma_u_sq`, `ln_sigma_v_sq`, `ln_lambda`, `ln_k`, `ln_sigma_sq`,
`ln_alpha`, `ln_theta`.

**Panel TFE_CSW/TRE keyword arguments (MLE):** `init`, `frontier`, `mu`,
`ln_sigma_u_sq`, `ln_sigma_v_sq`, `hscale`, `sigma_a_sq`, `message`.

Dispatches based on `method` (if provided) or `spec.datatype`.

# Arguments
- `spec::UnifiedSpec`: Unified model specification from `sfmodel_spec()`.
- `method=nothing`: Estimation method — a `Symbol` (`:MCI`, `:MSLE`, `:MLE`) or method
  object from `sfmodel_method()`. Used for cross-sectional dispatch.
- Remaining keyword arguments are forwarded to the appropriate backend.

# Returns
Initial values (format depends on backend: Vector or Dict).
"""
function sfmodel_init(; spec::UnifiedSpec, method=nothing, kwargs...)
    if spec.datatype == :panel_TFE
        # Panel TFE: dispatch to Panel_Backend (for MCI/MSLE compatibility)
        # When method=:MLE is used, sfmodel_fit passes init=nothing so MLE uses OLS defaults
        return Panel_Backend.sfmodel_panel_init(; spec=spec.panel_spec, kwargs...)
    elseif spec.datatype in (:panel_TFE_CSW, :panel_TRE)
        # MLE-only panel models: use MLE init
        return MLE_Backend.sfmodel_init(; spec=spec.mle_spec, kwargs...)
    elseif spec.datatype == :cross_sectional
        m = _resolve_method(method)
        if m == :MCI
            spec.mci_spec === nothing && error("Method :MCI requested but no MCI spec available.")
            return MCI_Backend.sfmodel_MCI_init(; spec=spec.mci_spec, kwargs...)
        elseif m == :MSLE
            spec.msle_spec === nothing && error("Method :MSLE requested but no MSLE spec available.")
            return MSLE_Backend.sfmodel_MSLE_init(; spec=spec.msle_spec, kwargs...)
        elseif m == :MLE
            spec.mle_spec === nothing && error("Method :MLE requested but no MLE spec available.")
            return MLE_Backend.sfmodel_init(; spec=spec.mle_spec, kwargs...)
        elseif m === nothing
            # Fallback: priority-based dispatch (backwards compatibility)
            if spec.mci_spec !== nothing
                return MCI_Backend.sfmodel_MCI_init(; spec=spec.mci_spec, kwargs...)
            elseif spec.msle_spec !== nothing
                return MSLE_Backend.sfmodel_MSLE_init(; spec=spec.msle_spec, kwargs...)
            elseif spec.mle_spec !== nothing
                return MLE_Backend.sfmodel_init(; spec=spec.mle_spec, kwargs...)
            else
                error("No backend specification available for initial value computation.")
            end
        else
            error("Unknown method: :$m. Use `:MCI`, `:MSLE`, or `:MLE`.")
        end
    else
        error("Unknown datatype: :$(spec.datatype)")
    end
end

# ============================================================================
# Section 7: sfmodel_opt — Unified optimization options
# ============================================================================

"""
    sfmodel_opt(; warmstart_solver=nothing, warmstart_opt=nothing,
                main_solver, main_opt)

Specify optimization options (solvers and convergence criteria).

All estimation methods (MCI, MSLE, and MLE) use the same optimizer interface,
so no `method` argument is needed.

# Arguments
- `warmstart_solver=nothing`: Optional warmstart optimizer (e.g., `NelderMead()`).
- `warmstart_opt=nothing`: NamedTuple of warmstart options (e.g., `(iterations=200,)`).
- `main_solver`: Main optimizer (e.g., `Newton()`). Required.
- `main_opt`: NamedTuple of main optimization options. Required.
"""
function sfmodel_opt(; warmstart_solver = nothing,
                     warmstart_opt = nothing,
                     main_solver,
                     main_opt)
    return MCI_Backend.sfmodel_MCI_opt(;
        warmstart_solver=warmstart_solver,
        warmstart_opt=warmstart_opt,
        main_solver=main_solver,
        main_opt=main_opt)
end

# ============================================================================
# Section 8: sfmodel_fit — Unified model estimation
# ============================================================================

# --- Internal helpers: convert cross-sectional method/opt to panel equivalents ---

# Convert MCI/MSLE method spec → PanelMethodSpec
_to_panel_method(m::MCI_Backend.SFMethodSpec) =
    Panel_Backend.PanelMethodSpec(m.method, m.transformation, m.draws, m.n_draws, m.multiRand, m.GPU, m.chunks, m.distinct_Halton_length)

_to_panel_method(m::MSLE_Backend.SFMethodSpec_MSLE) =
    Panel_Backend.PanelMethodSpec(m.method, nothing, m.draws, m.n_draws, m.multiRand, m.GPU, m.chunks, m.distinct_Halton_length)

_to_panel_method(m::Panel_Backend.PanelMethodSpec) = m   # already correct type

# Convert MCI opt spec → PanelOptSpec (identical field layout)
function _to_panel_opt(opt)
    isnothing(opt) && return nothing
    return Panel_Backend.PanelOptSpec(opt.warmstart_solver, opt.warmstart_opt,
                                      opt.main_solver, opt.main_opt)
end

# Convert MCI opt struct → MLE Dict{Symbol,Any}
function _to_mle_opt(opt)
    isnothing(opt) && return nothing
    return Dict{Symbol, Any}(
        :warmstart_solver => opt.warmstart_solver,
        :warmstart_maxIT  => isnothing(opt.warmstart_opt) ? nothing :
                             opt.warmstart_opt.iterations,
        :main_solver      => opt.main_solver,
        :main_maxIT       => opt.main_opt.iterations,
        :tolerance        => opt.main_opt.g_abstol,
        :verbose          => true,
        :banner           => true,
        :ineff_index      => true,
        :marginal         => true,
        :table_format     => :text
    )
end

# Reorder init vector from MCI order → MLE order for TruncatedNormal+scaling.
# MCI order: [frontier(K), scaling(L), μ(1), σ²(1), σᵥ²(1)]
# MLE order: [frontier(K), μ(1), hscale(L), σᵤ²(1), σᵥ²(1)]
function _reorder_init_mci_to_mle(spec::UnifiedSpec, init)
    isnothing(init) && return nothing
    if spec.ineff == :TruncatedNormal && spec.hetero === :scaling
        K = spec.mci_spec.K
        L = spec.mci_spec.L
        v = collect(init)
        return vcat(
            v[1:K],            # frontier
            v[K+L+1:K+L+1],   # μ (was after scaling in MCI)
            v[K+1:K+L],       # scaling → hscale
            v[K+L+2:end]      # σ², σᵥ²
        )
    end
    return init
end

# Reorder MLE result coefficients → MCI canonical order for TruncatedNormal+scaling.
# MLE order: [frontier(K), μ(1), hscale(L), σᵤ²(1), σᵥ²(1)]
# MCI order: [frontier(K), scaling(L), μ(1), σ²(1), σᵥ²(1)]
function _reorder_result_mle_to_mci(spec::UnifiedSpec, result)
    if spec.ineff == :TruncatedNormal && spec.hetero === :scaling
        K = spec.mci_spec.K
        L = spec.mci_spec.L
        n = length(result.coeff)
        # perm: [1:K, (K+2):(K+1+L), K+1, (K+2+L):n]
        perm = vcat(1:K, (K+2):(K+1+L), K+1, (K+2+L):n)
        new_coeff    = result.coeff[perm]
        new_std_err  = result.std_err[perm]
        new_vcov     = result.var_cov_mat[perm, perm]
        new_hessian  = result.Hessian[perm, perm]
        return merge(result, (coeff = new_coeff,
                              std_err = new_std_err,
                              var_cov_mat = new_vcov,
                              Hessian = new_hessian))
    end
    return result
end

"""
    sfmodel_fit(; spec::UnifiedSpec, method::UnifiedMethod, init=nothing,
                optim_options=nothing, jlms_bc_index=true, marginal=true,
                show_table=true, verbose=true)

Estimate the stochastic frontier model using the specified method.

Dispatches based on `method` type and `spec.datatype`:
- `method=:MLE` → MLE backend (analytic maximum likelihood)
- `method=:MCI` → MCI backend (Monte Carlo Integration)
- `method=:MSLE` → MSLE backend (Maximum Simulated Likelihood)
- Panel TFE with MCI/MSLE → Panel backend (Wang and Ho 2010, simulation-based)

Issues informative errors when a method is not available for the given model
configuration, listing the supported alternatives.

# Arguments
- `spec::UnifiedSpec`: Model specification from `sfmodel_spec()`.
- `method::UnifiedMethod`: Method specification from `sfmodel_method()`.
- `init=nothing`: Initial parameter values from `sfmodel_init()`.
- `optim_options=nothing`: Optimization options from `sfmodel_opt()`.
- `jlms_bc_index::Bool=true`: Compute JLMS and BC efficiency indices.
- `marginal::Bool=true`: Compute marginal effects (cross-sectional) or ignored (panel).
- `show_table::Bool=true`: Display estimation results table.
- `verbose::Bool=true`: Verbose output during estimation.

# Returns
Backend-specific result (NamedTuple with fields including `converged`, `loglikelihood`,
`coeff`, `std_err`, `jlms`, `bc`, etc.)
"""
function sfmodel_fit(;
    spec::UnifiedSpec,
    method::UnifiedMethod,
    init = nothing,
    optim_options = nothing,
    jlms_bc_index::Bool = true,
    marginal::Bool = true,
    show_table::Bool = true,
    verbose::Bool = true)

    # --- MLE dispatch (works for all datatypes) ---
    if method isa MLEMethodSpec
        if spec.mle_spec === nothing
            _mle_unavailable_error(spec)
        end
        # For panel_TFE, init was built by Panel_Backend — pass init=nothing
        # so MLE uses its own OLS-based defaults
        mle_init = (spec.datatype == :panel_TFE) ? nothing :
                   _reorder_init_mci_to_mle(spec, init)
        mle_opt  = _to_mle_opt(optim_options)
        raw = MLE_Backend.sfmodel_fit(;
            spec = spec.mle_spec,
            init = mle_init,
            optim_options = mle_opt,
            jlms_bc_index = jlms_bc_index,
            marginal = marginal,
            show_table = show_table,
            verbose = verbose)
        return _reorder_result_mle_to_mci(spec, raw)

    # --- Panel TFE with simulation methods (MCI/MSLE → Panel_Backend) ---
    elseif spec.datatype == :panel_TFE && !(method isa MLEMethodSpec)
        if spec.panel_spec === nothing
            error("No Panel spec available for `datatype=:panel_TFE`.")
        end
        panel_method = _to_panel_method(method)
        if panel_method.GPU
            Panel_Backend.check_gpu_overloads()
        end
        panel_opt    = _to_panel_opt(optim_options)

        return Panel_Backend.sfmodel_panel_fit(;
            spec          = spec.panel_spec,
            method        = panel_method,
            init          = init,
            optim_options = panel_opt,
            jlms_bc_index = jlms_bc_index,
            marginal      = marginal,
            show_table    = show_table,
            verbose       = verbose)

    # --- Cross-sectional MCI ---
    elseif method isa MCI_Backend.SFMethodSpec
        if spec.mci_spec === nothing
            error("`method=:MCI` is not available for this model configuration.\n" *
                  "  Supported methods: " * join(string.(_get_supported_methods(spec)), ", "))
        end
        return MCI_Backend.sfmodel_MCI_fit(;
            spec = spec.mci_spec,
            method = method,
            init = init,
            optim_options = optim_options,
            jlms_bc_index = jlms_bc_index,
            marginal = marginal,
            show_table = show_table,
            verbose = verbose)

    # --- Cross-sectional MSLE ---
    elseif method isa MSLE_Backend.SFMethodSpec_MSLE
        if spec.msle_spec === nothing
            error("`method=:MSLE` is not available for this model configuration.\n" *
                  "  Supported methods: " * join(string.(_get_supported_methods(spec)), ", "))
        end
        return MSLE_Backend.sfmodel_MSLE_fit(;
            spec = spec.msle_spec,
            method = method,
            init = init,
            optim_options = optim_options,
            jlms_bc_index = jlms_bc_index,
            marginal = marginal,
            show_table = show_table,
            verbose = verbose)

    else
        error("Unrecognized method type: $(typeof(method))")
    end
end

# --- Helper: list supported methods for a given spec ---
function _get_supported_methods(spec::UnifiedSpec)
    methods = Symbol[]
    spec.mci_spec  !== nothing && push!(methods, :MCI)
    spec.msle_spec !== nothing && push!(methods, :MSLE)
    spec.mle_spec  !== nothing && push!(methods, :MLE)
    return methods
end

# --- Helper: informative error when MLE is not available ---
function _mle_unavailable_error(spec::UnifiedSpec)
    parts = String[]
    if spec.noise != :Normal
        push!(parts, "`noise=:$(spec.noise)` is not supported (MLE requires `noise=:Normal`)")
    end
    if spec.copula != :None
        push!(parts, "`copula=:$(spec.copula)` is not supported (MLE does not support copulas)")
    end
    if spec.ineff ∉ (:HalfNormal, :TruncatedNormal, :Exponential)
        push!(parts, "`ineff=:$(spec.ineff)` is not supported by MLE " *
              "(MLE supports `:HalfNormal`, `:TruncatedNormal`, `:Exponential`)")
    end
    if spec.datatype == :panel_TFE && spec.ineff ∉ (:HalfNormal, :TruncatedNormal)
        push!(parts, "panel TFE with `ineff=:$(spec.ineff)` is not supported by MLE " *
              "(MLE supports `:HalfNormal` or `:TruncatedNormal` for panel TFE)")
    end
    supported = _get_supported_methods(spec)
    msg = "`method=:MLE` is not available for this model configuration.\n" *
          "  Reason: " * join(parts, "; ") * "\n" *
          "  Supported methods for this configuration: " *
          join(string.(supported), ", ")
    error(msg)
end

# ============================================================================
# Section 9: Panel model wrappers — Wang and Ho (2010) panel SF
# ============================================================================

"""
    sfmodel_panel_spec(; depvar, frontier, zvar, id, noise, ineff, type=:prod, varnames=nothing, eqnames=nothing, eq_indices=nothing)

Construct a panel model specification for the Wang and Ho (2010) stochastic frontier model.

# Arguments
- `depvar`: Response vector (N*T observations, stacked by firm).
- `frontier`: Frontier design matrix (N*T × K).
- `zvar`: Scaling function variables (N*T × L). Z is NOT demeaned; `h(z)=exp(z'δ)` is computed then demeaned.
- `id`: Unit identifier column (required). For balanced panels: `id = repeat(1:N, inner=T)`.
- `noise::Symbol`: Noise distribution (`:Normal`).
- `ineff::Symbol`: Inefficiency distribution (`:HalfNormal`).
- `type::Symbol=:prod`: `:prod` (production) or `:cost`.
- `varnames`, `eqnames`, `eq_indices`: Optional variable/equation names.

# Returns
`Panel_Backend.PanelModelSpec{T}`
"""
function sfmodel_panel_spec(; kwargs...)
    return Panel_Backend.sfmodel_panel_spec(; kwargs...)
end

"""
    sfmodel_panel_method(; method, n_draws=1024, transformation=nothing,
                           multiRand=true, GPU=false, chunks=10, distinct_Halton_length=2^15-1)

Specify the estimation method for the panel model.

# Arguments
- `method::Symbol`: `:MSLE` or `:MCI`.
- `n_draws::Int=1024`: Number of quasi-random draws.
- `transformation=nothing`: MCI transformation (`:expo_rule`, `:logistic_1_rule`). Ignored for MSLE.
- `multiRand::Bool=true`: Per-firm draws (`true`: N×D matrix) or shared draws (`false`: 1D vector).
- `GPU::Bool=false`: Use GPU acceleration.
- `chunks::Int=10`: Memory chunking for GPU.
- `distinct_Halton_length::Int=2^15-1`: Max Halton sequence length for multiRand.
"""
function sfmodel_panel_method(; kwargs...)
    return Panel_Backend.sfmodel_panel_method(; kwargs...)
end

"""
    sfmodel_panel_init(; spec, kwargs...)

Compute or assemble initial values for the panel model.

# Arguments
- `spec`: Panel model specification from `sfmodel_panel_spec()`.
- Keyword overrides: `frontier`, `delta`, `ln_sigma_u_sq`, `ln_sigma_v_sq`.

# Returns
`Vector{Float64}` of initial parameter values.
"""
function sfmodel_panel_init(; kwargs...)
    return Panel_Backend.sfmodel_panel_init(; kwargs...)
end

"""
    sfmodel_panel_opt(; warmstart_solver=nothing, warmstart_opt=nothing, main_solver, main_opt)

Specify optimization options for the panel model.

# Arguments
- `warmstart_solver`: Optional warmstart optimizer (e.g., `NelderMead()`).
- `warmstart_opt`: NamedTuple of warmstart options.
- `main_solver`: Main optimizer (e.g., `Newton()`). Required.
- `main_opt`: NamedTuple of main optimization options. Required.
"""
function sfmodel_panel_opt(; kwargs...)
    return Panel_Backend.sfmodel_panel_opt(; kwargs...)
end

"""
    sfmodel_panel_fit(; spec, method, init=nothing, optim_options=nothing, jlms_bc=true, show_table=true, verbose=true)

Estimate the Wang and Ho (2010) panel stochastic frontier model.

# Arguments
- `spec`: Panel model specification from `sfmodel_panel_spec()`.
- `method`: Method specification from `sfmodel_panel_method()`.
- `init=nothing`: Initial values from `sfmodel_panel_init()`.
- `optim_options=nothing`: Optimization options from `sfmodel_panel_opt()`.
- `jlms_bc::Bool=true`: Compute JLMS and BC firm-level efficiency indices.
- `show_table::Bool=true`: Display estimation results table.
- `verbose::Bool=true`: Verbose output during estimation.

# Returns
NamedTuple with fields: `converged`, `loglikelihood`, `coeff`, `std_err`, `vcov`,
`jlms`, `bc`, `redflag`, `table`, `optim_result`, `spec`, `method`.
"""
function sfmodel_panel_fit(; kwargs...)
    return Panel_Backend.sfmodel_panel_fit(; kwargs...)
end

# ============================================================================
# Exports
# ============================================================================

export sfmodel_spec, sfmodel_method, sfmodel_init, sfmodel_opt, sfmodel_fit
export sfmodel_panel_spec, sfmodel_panel_method, sfmodel_panel_init
export sfmodel_panel_opt, sfmodel_panel_fit
export sfmodel_MixTable, sfmodel_ChiSquareTable, sfmodel_CI, sfmodel_MoMTest
export @useData, @depvar, @frontier, @zvar, @id

# Re-export MLE utility functions applicable to all backends
using .MLE_Backend: sfmodel_MixTable, sfmodel_ChiSquareTable, sfmodel_CI, sfmodel_MoMTest

end # module SFrontiers
