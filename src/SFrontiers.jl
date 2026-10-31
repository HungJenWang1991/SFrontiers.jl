# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

module SFrontiers

using DataFrames

# ============================================================================
# Section 1: Load backends into isolated modules
# ============================================================================

# The Hessian part of `redflag` and the simulation-clamp test, shared by all four backends.
# Loaded first: each backend module imports it.
include(joinpath(@__DIR__, "sf_redflag.jl"))
# Small numerical kernels shared by all four backends and predict.
include(joinpath(@__DIR__, "sf_numerics.jl"))
# The Halton generator of the simulation backends and the row source through which predict reads a fit's
# stored draw block.
include(joinpath(@__DIR__, "sf_draws.jl"))
# Accuracy diagnostics of simulated likelihoods: the optional integration check of MCI and MSLE fits.
include(joinpath(@__DIR__, "sf_accuracy.jl"))

include(joinpath(@__DIR__, "load_MCI.jl"))
include(joinpath(@__DIR__, "load_MSLE.jl"))
include(joinpath(@__DIR__, "load_MLE.jl"))
include(joinpath(@__DIR__, "load_Panel.jl"))

# StatsAPI-conformant accessor layer (SFResult wrapper + coef/vcov/...).
include(joinpath(@__DIR__, "accessors.jl"))

# Numeric diagnostics: residuals, fitted, lrtest, efficiency_summary, ...
include(joinpath(@__DIR__, "diagnostics.jl"))

# RecipesBase plotting recipes for SFResult (backend-agnostic).
include(joinpath(@__DIR__, "plotrecipes.jl"))

# Prediction on new data (`predict(r; frontier, zvar, depvar, what)`).
include(joinpath(@__DIR__, "predict.jl"))

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
(unless `ineff=:Gamma`, the only MCI-only distribution), and
MLE spec (when noise=:Normal, no copula, and
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
    # NOT parameterised on T: the MLE backend is Float64 by construction (SFmainfun.jl hard-types
    # its depvar/zvar), so demanding SFModelSpec_MLE{T} made every Float32 spec whose ineff has an
    # MLE analogue (HalfNormal/TruncatedNormal/Exponential) die on a raw convert MethodError while
    # the other eight distributions built fine. Float32 data now builds for all of them and the
    # MLE path runs in Float64, which is what the manual promises.
    mle_spec::Union{MLE_Backend.SFModelSpec_MLE, Nothing}
    panel_spec::Union{Panel_Backend.PanelModelSpec{T}, Nothing}
    noise::Symbol
    ineff::Symbol
    copula::Symbol
    hetero::Union{Vector{Symbol}, Symbol}
end

# Inefficiency distributions available only through the MCI backend: the MSLE spec
# is skipped for these. Gamma is MCI-only because its estimated shape k sits inside
# the quantile function, which would force a slow, non-AD, non-GPU numeric inversion
# at every likelihood evaluation. The Erlang orders do NOT share this obstacle: their
# shape is pinned at an integer and θ is a pure scale, so the MSLE quantile factorizes
# as θ·q_m(p) with q_m parameter-free (see myErlangStdQuantile in the MSLE backend).
const _MCI_ONLY_INEFF = (:Gamma,)

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

# `:panel_TFE_CSW` and `:panel_TRE` have no inefficiency-determinant equation, so a `zvar`
# given to either would be dropped without a word: the coefficients and the log-likelihood would
# be identical to the last digit with and without it, and the user would believe they had
# estimated a model with inefficiency determinants and had not. Neither model can take them
# as it stands. `:panel_TFE_CSW` is half-normal by construction, so it has no mean parameter,
# and its likelihood requires the inefficiency variance to be one number per firm
# (`checkConst(wvar, :σᵤ², 1)` in SFgetvars.jl). `:panel_TRE` requires the same of the
# variance, and its half-normal form has no mean parameter either. The argument is therefore
# refused outright rather than warned about.
function _panel_zvar_unsupported(datatype::Symbol, ineff::Symbol)
    _why = datatype === :panel_TFE_CSW ?
        "it is half-normal by construction, so it has no inefficiency mean, and its " *
        "likelihood requires the inefficiency variance to be constant" :
        (ineff === :HalfNormal ?
         "its half-normal form has no inefficiency mean, and its likelihood requires the " *
         "inefficiency variance to be constant" :
         "its likelihood requires the inefficiency variance to be constant and treats the " *
         "inefficiency mean as a constant")
    error("`zvar` is not a parameter of the `:$(datatype)` model. This model has no " *
          "inefficiency-determinant equation: $(_why), so there is nothing for `zvar` to " *
          "enter and it would be silently ignored. Remove the `zvar` argument. For a panel " *
          "model with inefficiency determinants, use `datatype=:panel_TFE`, whose scaling " *
          "function h(z) = exp(z'δ) is exactly that.")
end

# `T_periods` was removed in favour of `id`. Accepted and dropped on a cross-sectional spec
# (keyword or DSL), it would let a script carrying the old argument keep running and quietly
# estimate something else than its author had written. One check, ahead of the datatype
# dispatch, refuses it everywhere.
_reject_T_periods(T_periods, idhint::String) =
    isnothing(T_periods) ||
    error("`T_periods` is no longer supported. Panel structure is specified with `$idhint`; " *
          "for balanced panels use `id = repeat(1:N, inner=T)`. `T_periods` is not accepted " *
          "in cross-sectional models either: they have no period dimension. Remove the argument.")

# A cross-sectional model has no panel dimension, so `id` selects nothing: the fit's
# log-likelihood would be identical with and without it. The DSL sibling refuses `@id()` here; the
# keyword form refuses `id=` the same way.
_reject_cross_id(id) =
    isnothing(id) ||
    error("`id` is not accepted in cross-sectional models: it is only valid for panel models, " *
          "and a cross-sectional model has no panel structure to identify. Remove it, or set " *
          "`datatype=:panel_TFE`, `:panel_TFE_CSW`, or `:panel_TRE`.")

# With no `hetero` block and no `:scaling`, every inefficiency parameter is a constant read
# at `coef·zvar[:,1]`, and `zvar` cannot reach the likelihood: at fixed parameters the NLL is
# byte-identical with `zvar`, with `1000 * zvar`, and with no `zvar` at all. (Under
# `hetero=[:sigma_u_sq]` the same two z matrices give 22.66 and 7.3e22, so this is a property
# of the specification, not of the argument.) Refusing it here turns a
# silent no-op into the missing-`hetero` error it always was.
_reject_unusable_zvar(zvar, hetero) =
    (isnothing(zvar) || hetero === :scaling || !(hetero isa AbstractVector) ||
     !isempty(hetero)) ||
    error("`zvar` cannot be used by this model: `hetero` is empty and the model is not a " *
          "scaling-property model, so every inefficiency parameter is a constant and there " *
          "is no equation for `zvar` to enter — it would be silently ignored. Either name " *
          "the parameters that depend on it (e.g. `hetero=[:sigma_u_sq]`), use " *
          "`hetero=:scaling` for the scaling-property form h(z)=exp(z'δ), or drop `zvar`.")

"""
    sfmodel_spec(; depvar, frontier, zvar=nothing, noise, ineff, copula=:None,
                 hetero=Symbol[], datatype=:cross_sectional,
                 T_periods=nothing, id=nothing,
                 varnames=nothing, eqnames=nothing, eq_indices=nothing, type=:prod)

Construct a unified model specification.

When `datatype=:cross_sectional` (default): calls MCI, MSLE, and MLE backends internally
as applicable. MCI supports all distributions; MSLE excludes only Gamma; MLE requires
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
  Requires a `hetero` block or `hetero=:scaling` to read it: a cross-sectional spec with
  neither has no equation for `zvar` to enter, so the argument is refused rather than ignored.
- `noise::Symbol`: Noise distribution. Cross-sectional: `:Normal`, `:StudentT`, `:Laplace`.
  Panel: `:Normal` only.
- `ineff::Symbol`: Inefficiency distribution (`:HalfNormal`, `:TruncatedNormal`, `:Exponential`,
  `:Weibull`, `:Lognormal`, `:Lomax`, `:Rayleigh`, `:Gamma`, `:Erlang2`, `:Erlang3`, `:Erlang4`)
- `copula::Symbol=:None`: Copula (cross-sectional only). Not available for panel models.
- `hetero::Vector{Symbol}=Symbol[]`: Heteroscedasticity (cross-sectional only). Not available for panel models.
  The symbols of versions before 2.0.0, `:sigma_sq` (and `:lambda` for `ineff = :Exponential`), are accepted
  with a deprecation warning and mean `:sigma_u_sq`.
- `datatype::Symbol=:cross_sectional`: Data type. `:cross_sectional`, `:panel_TFE`,
  `:panel_TFE_CSW`, or `:panel_TRE`.
- `T_periods=nothing`: Removed. Kept in the signature only so an explicit value is refused
  by name; panel structure is specified with `id`.
- `id=nothing`: Unit identifier column (panel models only; refused on `:cross_sectional`).
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

    datatype === :cross_sectional && (hetero = _alias_hetero(hetero, ineff))   # deprecated symbols
    _reject_T_periods(T_periods, "id")

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
                "Individual heteroscedastic parameters (e.g., [:mu, :sigma_u_sq]) are not available for panel models.")
        end
    end
    _validate_panel_no_hetero = () -> begin
        # panel_TFE_CSW and panel_TRE: no hetero at all
        _has_hetero = hetero isa Symbol ? true : !isempty(hetero)
        _has_hetero && error("This panel model does not support `hetero`. Remove the `hetero` argument.")
    end

    # --- Helper: build MLE spec for cross-sectional models (returns nothing if not applicable) ---
    # `vn`: the varnames for the analytic backend, which labels its table rows by position in
    # its own parameter order (see _varnames_canon_to_mle).
    _build_mle_cross = (vn = varnames) -> begin
        # MLE scaling only supports TruncatedNormal; skip for other distributions
        if hetero === :scaling && ineff != :TruncatedNormal
            return nothing
        end
        if noise == :Normal && copula == :None &&
           ineff in (:HalfNormal, :TruncatedNormal, :Exponential)
            MLE_Backend.sfmodel_spec(;
                depvar=depvar, frontier=frontier, zvar=zvar,
                noise=:Normal, ineff=ineff, hetero=hetero, type=type,
                varnames=vn, eqnames=eqnames, eq_indices=eq_indices)
        else
            nothing
        end
    end

    if datatype == :panel_TFE
        # Wang and Ho (2010) true fixed-effect model
        # Supported by Panel_Backend (MCI/MSLE) and MLE_Backend (HalfNormal/TruncatedNormal)
        _validate_panel_no_copula()
        _validate_panel_hetero_scaling_only()

        # Always build Panel spec (for MCI/MSLE; supports all 11 distributions,
        # with Gamma gated to MCI inside the panel backend)
        panel_spec = Panel_Backend.sfmodel_panel_spec(;
            depvar=depvar, frontier=frontier, zvar=zvar,
            id=id,
            noise=noise, ineff=ineff, type=type,
            varnames=varnames, eqnames=eqnames, eq_indices=eq_indices)

        # Build MLE spec when applicable (Normal noise + HalfNormal/TruncatedNormal)
        mle_spec = if noise == :Normal && ineff in (:HalfNormal, :TruncatedNormal)
            MLE_Backend.sfmodel_spec(;
                depvar=depvar, frontier=frontier, zvar=zvar,
                ineff=ineff, panel=:TFE_WH2010, id=id, type=type,
                # The analytic backend's own order (μ before the scaling block)
                varnames=(ineff == :TruncatedNormal ?
                          _varnames_canon_to_mle(varnames, panel_spec.K, panel_spec.L) : varnames),
                eqnames=eqnames, eq_indices=eq_indices)
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
        isnothing(zvar) || _panel_zvar_unsupported(:panel_TFE_CSW, ineff)
        noise == :Normal || error("Panel TFE_CSW models require `noise=:Normal`.")
        ineff == :HalfNormal || error("Panel TFE_CSW models require `ineff=:HalfNormal`.")

        mle_spec = MLE_Backend.sfmodel_spec(;
            depvar=depvar, frontier=frontier,
            ineff=:HalfNormal, panel=:TFE_CSW2014, id=id, type=type,
            varnames=varnames, eqnames=eqnames, eq_indices=eq_indices)

        T_el = eltype(mle_spec.depvar)
        return UnifiedSpec{T_el}(:panel_TFE_CSW, nothing, nothing, mle_spec, nothing,
                                 noise, ineff, :None, Symbol[])

    elseif datatype == :panel_TRE
        # True random-effect model — MLE only
        _validate_panel_no_copula()
        _validate_panel_no_hetero()
        isnothing(zvar) || _panel_zvar_unsupported(:panel_TRE, ineff)
        noise == :Normal || error("Panel TRE models require `noise=:Normal`.")
        ineff in (:HalfNormal, :TruncatedNormal) || error(
            "Panel TRE models require `ineff=:HalfNormal` or `:TruncatedNormal`. " *
            "Got `ineff=:$ineff`.")

        mle_spec = MLE_Backend.sfmodel_spec(;
            depvar=depvar, frontier=frontier,
            ineff=ineff, panel=:TRE, id=id, type=type,
            varnames=varnames, eqnames=eqnames, eq_indices=eq_indices)

        T_el = eltype(mle_spec.depvar)
        return UnifiedSpec{T_el}(:panel_TRE, nothing, nothing, mle_spec, nothing,
                                 noise, ineff, :None, Symbol[])

    elseif datatype == :cross_sectional
        _reject_cross_id(id)
        _reject_unusable_zvar(zvar, hetero)

        if hetero === :scaling
            # Scaling property model: build MCI, MSLE, and MLE specs
            mci_spec = MCI_Backend.sfmodel_spec(;
                depvar=depvar, frontier=frontier, zvar=zvar,
                noise=noise, ineff=ineff, copula=copula, hetero=:scaling,
                varnames=varnames, eqnames=eqnames,
                eq_indices=eq_indices, type=type)

            msle_spec = if ineff in _MCI_ONLY_INEFF
                nothing
            else
                MSLE_Backend.sfmodel_spec(;
                    depvar=depvar, frontier=frontier, zvar=zvar,
                    noise=noise, ineff=ineff, copula=copula, hetero=:scaling,
                    varnames=varnames, eqnames=eqnames,
                    eq_indices=eq_indices, type=type)
            end

            # MLE scaling requires TruncatedNormal (handled internally by MLE_Backend).
            # User varnames follow the canonical order (scaling block before μ);
            # the analytic backend labels its rows in its own order (μ first).
            mle_spec = _build_mle_cross(_varnames_canon_to_mle(varnames, mci_spec.K, mci_spec.L_scaling))

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

        # Build MSLE spec only for distributions MSLE supports (all except Gamma)
        msle_spec = if ineff in _MCI_ONLY_INEFF
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

The symbols of versions before 2.0.0, `:sigma_sq` (and `:lambda` for `ineff = :Exponential`), are accepted with
a deprecation warning and mean `:sigma_u_sq`.
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

    datatype === :cross_sectional && (hetero = _alias_hetero(hetero, ineff))   # deprecated symbols
    _reject_T_periods(T_periods, "@id(...)")

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
                "Individual heteroscedastic parameters (e.g., [:mu, :sigma_u_sq]) are not available for panel models.")
        end
    end
    _validate_panel_no_hetero_dsl = () -> begin
        _has_hetero = hetero isa Symbol ? true : !isempty(hetero)
        _has_hetero && error("This panel model does not support `hetero`. Remove the `hetero` argument.")
    end

    # --- Helper: build Panel spec from DSL args ---
    _build_panel_spec_dsl = () -> begin
        data_p = Panel_Backend.UseDataSpec(data.df)
        dv_p   = Panel_Backend.DepvarSpec(dv.name)
        fr_p   = Panel_Backend.FrontierSpec(fr.names)

        if !isnothing(idspec)
            id_p = Panel_Backend.IdSpec(idspec.name)
            if !isnothing(zv)
                Panel_Backend.sfmodel_panel_spec(
                    data_p, dv_p, fr_p,
                    Panel_Backend.ZvarSpec(zv.names), id_p;
                    noise=noise, ineff=ineff, type=type,
                    eqnames=eqnames, eq_indices=eq_indices)
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
        # MLE scaling only supports TruncatedNormal; skip for other distributions
        # (mirror the keyword sibling `_build_mle_cross`).
        if hetero === :scaling && ineff != :TruncatedNormal
            return nothing
        end
        if noise == :Normal && copula == :None &&
           ineff in (:HalfNormal, :TruncatedNormal, :Exponential)
            (depvar_data, frontier_data, zvar_data, _) = _dsl_extract_data()
            MLE_Backend.sfmodel_spec(;
                depvar=depvar_data, frontier=frontier_data, zvar=zvar_data,
                noise=:Normal, ineff=ineff, hetero=hetero, type=type,
                eqnames=eqnames, eq_indices=eq_indices,
                frontier_names=fr.names,
                zvar_names=(!isnothing(zv) ? zv.names : nothing))
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
        _dsl_fr_names = fr.names
        _dsl_zv_names = !isnothing(zv) ? zv.names : nothing
        mle_spec = if noise == :Normal && ineff in (:HalfNormal, :TruncatedNormal)
            (depvar_data, frontier_data, zvar_data, id_data) = _dsl_extract_data()
            MLE_Backend.sfmodel_spec(;
                depvar=depvar_data, frontier=frontier_data, zvar=zvar_data,
                ineff=ineff, panel=:TFE_WH2010, id=id_data, type=type,
                eqnames=eqnames, eq_indices=eq_indices,
                frontier_names=_dsl_fr_names, zvar_names=_dsl_zv_names)
        else
            nothing
        end

        T_el = eltype(panel_spec.depvar)
        return UnifiedSpec{T_el}(:panel_TFE, nothing, nothing, mle_spec, panel_spec,
                                 noise, ineff, :None, Symbol[])

    elseif datatype == :panel_TFE_CSW
        _validate_panel_no_copula_dsl()
        _validate_panel_no_hetero_dsl()
        isnothing(zv) || _panel_zvar_unsupported(:panel_TFE_CSW, ineff)
        noise == :Normal || error("Panel TFE_CSW models require `noise=:Normal`.")
        ineff == :HalfNormal || error("Panel TFE_CSW models require `ineff=:HalfNormal`.")

        (depvar_data, frontier_data, _, id_data) = _dsl_extract_data()
        mle_spec = MLE_Backend.sfmodel_spec(;
            depvar=depvar_data, frontier=frontier_data,
            ineff=:HalfNormal, panel=:TFE_CSW2014, id=id_data, type=type,
            eqnames=eqnames, eq_indices=eq_indices,
            frontier_names=fr.names)

        T_el = eltype(mle_spec.depvar)
        return UnifiedSpec{T_el}(:panel_TFE_CSW, nothing, nothing, mle_spec, nothing,
                                 noise, ineff, :None, Symbol[])

    elseif datatype == :panel_TRE
        _validate_panel_no_copula_dsl()
        _validate_panel_no_hetero_dsl()
        isnothing(zv) || _panel_zvar_unsupported(:panel_TRE, ineff)
        noise == :Normal || error("Panel TRE models require `noise=:Normal`.")
        ineff in (:HalfNormal, :TruncatedNormal) || error(
            "Panel TRE models require `ineff=:HalfNormal` or `:TruncatedNormal`. " *
            "Got `ineff=:$ineff`.")

        (depvar_data, frontier_data, _, id_data) = _dsl_extract_data()
        mle_spec = MLE_Backend.sfmodel_spec(;
            depvar=depvar_data, frontier=frontier_data,
            ineff=ineff, panel=:TRE, id=id_data, type=type,
            eqnames=eqnames, eq_indices=eq_indices,
            frontier_names=fr.names)

        T_el = eltype(mle_spec.depvar)
        return UnifiedSpec{T_el}(:panel_TRE, nothing, nothing, mle_spec, nothing,
                                 noise, ineff, :None, Symbol[])

    elseif datatype == :cross_sectional
        !isnothing(idspec) && datatype == :cross_sectional &&
            error("`@id()` is only valid for panel models (e.g., `datatype=:panel_TFE`).")
        _reject_unusable_zvar(isnothing(zv) ? nothing : zv.names, hetero)

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

            msle_spec = if ineff in _MCI_ONLY_INEFF
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

        msle_spec = if ineff in _MCI_ONLY_INEFF
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
                   n_draws=nothing, multiRand=true, GPU=false, chunks=10, recycling=false,
                   distinct_Halton_length=nothing)

Specify the estimation method.

# Arguments
- `method::Symbol`: `:MCI` for Monte Carlo Integration, `:MSLE` for Maximum Simulated
  Likelihood, or `:MLE` for analytic Maximum Likelihood Estimation.
- `transformation=nothing`: Transformation rule (MCI only). Under the default rule, MCI fits whose
  estimated shape lies in a range where that rule is unreliable (Gamma k, Weibull k, Lomax α below a
  threshold) print a warning (unless the fit passes `shape_warning = false`) and store the verdict in
  `result.shape_check`.
- `draws=nothing`: User-supplied draws in [0, 1); auto-generated Halton if `nothing`. The element
  type must be an `AbstractFloat` (`Float32`, `Float64`, ...); an integer, rational or `Real`
  array is refused by the keyword's own type, and `predict(draws=)` enforces the same rule, so a
  draw object one boundary takes is a draw object the other takes. The ROW COUNT
  decides how they are read, and `multiRand` is not part of it: a length-D vector, a 1×D row matrix or
  a D×1 column matrix is one sequence of D draws shared by every observation; an N×D matrix is one row of D
  draws per observation. Any other row count is an error. Supplying `draws` forces `multiRand=false` and makes
  `n_draws` irrelevant (D comes from the draws themselves). The fit keeps the draws for `predict()`. When
  the array already has the data's element type and lives where the data live (a host `Array`, or a
  `CuArray` with `GPU = true`), the fit uses the array itself, not a copy, so changing the array afterwards
  changes what `predict()` computes.
- `n_draws=nothing`: Number of Halton draws per observation (per firm in panel models; MCI/MSLE only), used
  only when `draws` is not supplied. Default: 4096 (2^12). Use powers of two: without recycling, the draws of
  each observation then contain exactly one point in each interval [(k - 1)/n_draws, k/n_draws). Time and
  memory grow in proportion to N × `n_draws`.
- `multiRand::Bool=true`: generation mode, used only when `draws` is not supplied. `true` gives observation i
  (firm i in panel models) the `n_draws` consecutive points (i − 1)·n_draws + 1, …, i·n_draws of one base-2
  Halton sequence of N × `n_draws` points, so no point is used twice unless `recycling` or the deprecated
  `distinct_Halton_length` is given; `false` gives every observation the same first `n_draws` points.
- `GPU::Bool=false`: Use GPU acceleration (MCI/MSLE only).
- `chunks::Int=10`: Number of groups of observations (firms, for panel models) processed at a time (MCI/MSLE
  only). The memory of the likelihood, its derivatives and the efficiency indices is proportional to
  N × `n_draws` / `chunks`, on the CPU and on the GPU.
- `recycling=false`: reuse of Halton points across observations, for generated per-observation draws only
  (`multiRand = true`, no `draws`). `false` (default): no point is reused. `true`: the sequence is capped at
  L = 2^24 − 1 = 16,777,215 points and restarts from its first point when N × `n_draws` exceeds L; observation
  i receives the points at positions mod1((i − 1)·n_draws + j, L), j = 1, …, n_draws. An integer L with
  `n_draws` ≤ L ≤ 2^24 − 1 sets another cap. When N × `n_draws` exceeds L, observations i and i + L/g receive
  identical draws, where g is the greatest common divisor of `n_draws` and L; the fit warns when the sample
  has more than L/g observations. With `recycling = true` (L = 2^24 − 1, an odd number) any power of two for
  `n_draws` gives g = 1. With `draws` or `multiRand = false`, `recycling` is an error.
- `distinct_Halton_length=nothing`: deprecated; it gives a warning. Earlier `:MCI` and `:MSLE` scripts that set
  it still run and, with the same `n_draws`, get the draws they got before 2.0.0. To reproduce the draws of a
  fit of those versions that used the default length, give `recycling = 2^15 - 1` and that fit's `n_draws`
  (1,024 by default): the draws are identical whenever N × `n_draws` is at least 32,767. Use `recycling` for
  new work; giving both is an error.

`method=:MLE` is analytic and reads none of the simulation settings above; a nondefault value
for any of them is refused rather than ignored (any value of `recycling` other than `false`, and
any value of `distinct_Halton_length`, 2^15 − 1 included). `transformation` is MCI-only and is
likewise refused under `method=:MSLE`.

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
    recycling::Union{Bool,Integer} = false,
    distinct_Halton_length::Union{Int,Nothing} = nothing)

    # Early check: if GPU requested, ensure CUDA was loaded before SFrontiers
    if GPU
        if method == :MCI
            MCI_Backend.check_gpu_overloads()
        elseif method == :MSLE
            MSLE_Backend.check_gpu_overloads()
        end
    end

    if method == :MCI
        _n_draws = isnothing(n_draws) ? 2^12 : n_draws
        return MCI_Backend.sfmodel_method(;
            method=:MCI,
            transformation=transformation,
            draws=draws,
            n_draws=_n_draws,
            multiRand=multiRand,
            GPU=GPU,
            chunks=chunks,
            recycling=recycling,
            distinct_Halton_length=distinct_Halton_length)

    elseif method == :MSLE
        # `transformation` is an MCI-only device — it reparameterizes the MCI inefficiency
        # integral. MSLE draws from the inefficiency distribution directly and has no integral
        # to transform, so the value would be stored and never read: at fixed parameters
        # `transformation=:expo_rule` and no transformation give the same NLL to the last bit.
        # A warning would leave the caller with a fit they believed used a rule it cannot use,
        # so the argument is refused.
        isnothing(transformation) ||
            error("`transformation` is only effective with `method=:MCI`: it reparameterizes the " *
                  "MCI inefficiency integral, and `method=:MSLE` has no such integral. Remove " *
                  "`transformation`, or use `method=:MCI`.")
        _n_draws = isnothing(n_draws) ? 2^12 : n_draws
        return MSLE_Backend.sfmodel_method(;
            method=:MSLE,
            draws=draws,
            n_draws=_n_draws,
            multiRand=multiRand,
            GPU=GPU,
            chunks=chunks,
            recycling=recycling,
            distinct_Halton_length=distinct_Halton_length)

    elseif method == :MLE
        # `method=:MLE` is the analytic likelihood: it draws nothing, so not one simulation
        # setting reaches it — `MLEMethodSpec` has no fields to put them in. Each setting is
        # compared against its own default (several are not `nothing`) and named in the refusal.
        _mle_supplied = String[]
        isnothing(transformation)                    || push!(_mle_supplied, "transformation")
        isnothing(draws)                             || push!(_mle_supplied, "draws")
        isnothing(n_draws)                           || push!(_mle_supplied, "n_draws")
        multiRand === true                           || push!(_mle_supplied, "multiRand")
        GPU === false                                || push!(_mle_supplied, "GPU")
        chunks == 10                                 || push!(_mle_supplied, "chunks")
        # Each simulation setting is compared with its own default: `recycling` with `false`,
        # `distinct_Halton_length` with `nothing`.
        recycling === false                          || push!(_mle_supplied, "recycling")
        distinct_Halton_length === nothing           || push!(_mle_supplied, "distinct_Halton_length")
        isempty(_mle_supplied) ||
            error("$(join(_mle_supplied, ", ")) " *
                  (length(_mle_supplied) == 1 ? "is a simulation setting and is" :
                                                "are simulation settings and are") *
                  " not used by `method=:MLE`, which evaluates the likelihood analytically " *
                  "and draws nothing. Remove " *
                  (length(_mle_supplied) == 1 ? "it" : "them") *
                  ", or use `method=:MCI` / `method=:MSLE`, which do read them.")
        return MLEMethodSpec()

    else
        error("Unknown method: :$method. Use `:MCI`, `:MSLE`, or `:MLE`.")
    end
end

# ============================================================================
# Section 6: sfmodel_init — Unified initial values
# ============================================================================
# In versions before 2.0.0 `sfmodel_init` took a `method` argument, and for cross-sectional
# models it handed back a different kind of object for each value: a `Vector{Float64}` under
# `:MCI` and `:MSLE`, a component-keyed `Dict` under `:MLE`. The argument is removed. Every
# datatype and every method receives the same vector, so it would select nothing; and the
# same typed vector meant two different things on a scaling model, because each route read
# it in its own parameter order.

"""
    sfmodel_init(; spec::UnifiedSpec, kwargs...)

Compute or assemble starting values for the estimation.

Returns a `Vector{Float64}` in the model's own parameter order, for every `datatype` and
for every estimation method. The same vector is accepted by `sfmodel_fit` under
`method=:MCI`, `:MSLE` and `:MLE` alike (each where the model supports it); where two
backends order a block differently, `sfmodel_fit` converts at the fit boundary, so the
order documented for the model is the only order a user needs to know.

**Every component is optional.** A component that is not supplied takes exactly the value
the fit would have used had no starting values been supplied at all: the OLS estimate for
the frontier coefficients, and `0.1` for every other parameter, together with the panel
builder's distribution-specific overrides where it has them (`mu = 0.0`, and for Lomax
`ln_lambda = -1.0` and `ln_alpha = 0.5`). So `sfmodel_init(spec = myspec)` returns exactly
the vector the fit builds for itself from `init = nothing`.

**Keyword arguments by datatype.**
- Cross-sectional: `init`, `frontier`, `scaling`, `mu`, `ln_sigma_u_sq`, `ln_sigma_v_sq`,
  `ln_nu_minus_2`, `ln_b`, `ln_lambda`, `ln_k`, `ln_alpha`, `ln_theta` (Gamma: MCI only;
  Erlang: MCI and MSLE), `theta_rho` (copula), `message`. `hscale` is accepted as a synonym
  for `scaling`, which is the documented name.
- `:panel_TFE`: `init`, `frontier`, `scaling`, `mu`, `ln_sigma_u_sq`, `ln_sigma_v_sq`,
  `ln_lambda`, `ln_k`, `ln_alpha`, `ln_theta`. This route takes no `message` argument.
- `:panel_TFE_CSW` and `:panel_TRE`: `init`, `frontier`, `ln_sigma_u_sq`, `ln_sigma_v_sq`,
  `message`, plus `sigma_a_sq` and `mu` for `:panel_TRE` with `ineff=:TruncatedNormal`.
  Neither model has a scaling function, so neither accepts `scaling` or `hscale`.

The names of versions before 2.0.0 are accepted with a deprecation warning: `ln_sigma_sq` for `ln_sigma_u_sq`,
and for `ineff = :Exponential` also `ln_lambda`. Giving an old and a new name together is an error.

A component argument that is not part of the model is refused by name rather than dropped.

# Arguments
- `spec::UnifiedSpec`: model specification from `sfmodel_spec()`.
- Remaining keyword arguments are the starting-value components listed above.

# Returns
`Vector{Float64}` — the starting values, in the model's parameter order.
"""
function sfmodel_init(; spec::UnifiedSpec, kwargs...)
    kw = _init_kwargs(spec, kwargs)
    if spec.datatype == :panel_TFE
        return Panel_Backend.sfmodel_panel_init(; spec=spec.panel_spec, kw...)
    elseif spec.datatype in (:panel_TFE_CSW, :panel_TRE)
        # MLE-only panel models: flatten the analytic backend's component dictionary into
        # the vector the fit assembles from it, so these two return the same shape as the
        # rest (see `MLE_Backend.sfmodel_init_vec`).
        return MLE_Backend.sfmodel_init_vec(; spec=spec.mle_spec, kw...)
    elseif spec.datatype == :cross_sectional
        # One canonical vector, whatever the estimation method will be. The simulation
        # builders are the canonical ones: they cover every cross-sectional model the
        # package fits, including those with no analytic MLE route, and `sfmodel_fit`
        # converts to the analytic backend's order where the two differ
        # (`_reorder_init_mci_to_mle`).
        if spec.mci_spec !== nothing
            return MCI_Backend.sfmodel_MCI_init(; spec=spec.mci_spec, kw...)
        elseif spec.msle_spec !== nothing
            return MSLE_Backend.sfmodel_MSLE_init(; spec=spec.msle_spec, kw...)
        else
            # INTERNAL INVARIANT, not a user-facing path: every cross-sectional specification
            # `sfmodel_spec` accepts owns an MCI spec (and an MSLE one except for Gamma), so
            # this branch is unreachable from any public call (checked over all 66 cross
            # configurations). Kept so that a future change which stops building one says so
            # by name.
            error("No backend specification available for initial value computation.")
        end
    else
        error("Unknown datatype: :$(spec.datatype)")
    end
end

# Refuse the removed `method` argument with the reason, rather than letting it reach a
# backend as an unknown keyword and surface as a MethodError; and accept the analytic
# backend's `hscale` as a synonym for the canonical `scaling` on cross-sectional models, so
# that a call which named the scaling coefficients under the old `method=:MLE` route keeps
# working.
function _init_kwargs(spec::UnifiedSpec, kwargs)
    kw = Dict{Symbol,Any}(kwargs)
    if haskey(kw, :method)
        # The message does not say the vector is accepted "under `:MCI`, `:MSLE` and `:MLE`
        # alike", which is false for any model that does not support all three: a Weibull
        # specification would get that message from `sfmodel_init(method = :MLE)` and then its
        # actual MLE fit errors "`method=:MLE` is not available", and an MLE-only CSW
        # specification is the mirror image under `:MCI`. The vector is method-independent;
        # which methods can be run is a property of the model, not of the starting values.
        error("`sfmodel_init` no longer takes a `method` argument (removed in version 2.0.0). " *
              "Starting values are the same for every estimation method: one " *
              "`Vector{Float64}` in the model's own parameter order, accepted by " *
              "`sfmodel_fit` under every estimation method this model supports. " *
              "(Which those are is a property of the model — `sfmodel_fit` names them if " *
              "you ask for one it does not have.) " *
              "Delete `method=...` from the `sfmodel_init` call.")
    end
    _alias_init_names!(kw, spec.ineff)
    # `nothing` is what every optional component defaults to and is the documented way to say
    # "not supplied" — `mu = nothing` behaves exactly like omitting `mu`. The alias check below
    # therefore tests the value, not `haskey`: otherwise `hscale = 0.3, scaling = nothing`
    # would be read as a conflict and refused, as would `scaling = nothing, hscale = nothing`,
    # which supplies nothing at all. A key whose value is `nothing` is dropped, so it behaves
    # as absence does, and only two genuine values count as a conflict.
    _supplied(k) = haskey(kw, k) && kw[k] !== nothing
    _h, _s = _supplied(:hscale), _supplied(:scaling)
    for k in (:hscale, :scaling)
        haskey(kw, k) && kw[k] === nothing && delete!(kw, k)
    end

    if spec.datatype == :cross_sectional
        if _h && _s
            # The conflict message explains the two names as belonging to
            # "the same scaling function", which is true only of a scaling-property model. On
            # a homoscedastic specification there is no scaling function to have two names
            # for, and each alias ON ITS OWN already gets the right answer there — the
            # unused-component error. So say that instead, rather than describing a function
            # the model does not have.
            spec.hetero === :scaling ? error(
                "Give either `scaling` or `hscale`, not both: they are two names for the " *
                "coefficients of the same scaling function. `scaling` is the documented name.") :
                error(
                "Neither `scaling` nor `hscale` is part of the current model specification: " *
                "this model has no scaling function, so there are no scaling coefficients to " *
                "set. (The two are names for the same component where it exists; on a " *
                "scaling-property model — `hetero = :scaling` — supply one of them, not both.)")
        end
        _h && (kw[:scaling] = pop!(kw, :hscale))

    elseif spec.datatype in (:panel_TFE_CSW, :panel_TRE)
        # Neither MLE-only panel model has a scaling function, and the analytic backend refuses
        # `hscale` by name with exactly that reason. It also takes `scaling`, the canonical
        # spelling the unified API documents, as a synonym, so both spellings reach the same
        # guard, which names whichever was typed.
        _h && _s && error(
            "Give either `scaling` or `hscale`, not both: they are two names for the same " *
            "component. Neither is a parameter of a `:$(spec.datatype)` model, which has no " *
            "scaling function.")
    end
    return (; kw...)
end

# The names of versions before 2.0.0 for the variance parameter of the inefficiency distribution, accepted at the
# exported entry points with a deprecation warning (one log id per old name) and moved, value unchanged, to the
# slot of the new name. Only the distributions that own `ln_sigma_u_sq` take them; for any other the old name stays
# in place, so that the existing error names it.
const _OWNS_LN_SIGMA_U_SQ = (:HalfNormal, :TruncatedNormal, :Lognormal, :Rayleigh, :Exponential)

function _alias_init_names!(kw::Dict{Symbol,Any}, ineff::Symbol)
    # a value of `nothing` counts as absent, as for every optional component
    for k in (:ln_sigma_sq, :ln_lambda, :ln_sigma_u_sq)
        haskey(kw, k) && kw[k] === nothing && delete!(kw, k)
    end
    if haskey(kw, :ln_sigma_sq) && ineff in _OWNS_LN_SIGMA_U_SQ
        haskey(kw, :ln_sigma_u_sq) && throw(ArgumentError(
            "Give `ln_sigma_u_sq` or its deprecated name `ln_sigma_sq`, not both."))
        @warn "`ln_sigma_sq` is deprecated; use `ln_sigma_u_sq`, the name of the same parameter. " *
              "The value is used unchanged." _id=:sf_deprecated_ln_sigma_sq maxlog=1
        kw[:ln_sigma_u_sq] = pop!(kw, :ln_sigma_sq)
    end
    if haskey(kw, :ln_lambda) && ineff === :Exponential
        haskey(kw, :ln_sigma_u_sq) && throw(ArgumentError(
            "Give `ln_sigma_u_sq` or its deprecated name `ln_lambda` (for `ineff = :Exponential`), not both."))
        @warn "`ln_lambda` is deprecated for `ineff = :Exponential`; use `ln_sigma_u_sq`. For the " *
              "Exponential both names denote the log of σᵤ² = Var(u), so the value is used unchanged; " *
              "`ln_lambda` remains the log scale of `:Weibull` and `:Lomax`." _id=:sf_deprecated_ln_lambda maxlog=1
        kw[:ln_sigma_u_sq] = pop!(kw, :ln_lambda)
    end
    return kw
end

# The same for the `hetero` symbols of a cross-sectional specification: each old symbol is replaced in place, so
# the vector keeps the order the user gave (the specification stores and prints it).
function _alias_hetero(hetero, ineff::Symbol)
    hetero isa Vector{Symbol} || return hetero
    old = Symbol[]
    new = Symbol[]
    for h in hetero
        if (h === :sigma_sq && ineff in _OWNS_LN_SIGMA_U_SQ) || (h === :lambda && ineff === :Exponential)
            push!(old, h)
            push!(new, :sigma_u_sq)
        else
            push!(new, h)
        end
    end
    isempty(old) && return hetero
    count(==(:sigma_u_sq), new) > 1 && throw(ArgumentError(
        "Give `:sigma_u_sq` or its deprecated name (`:sigma_sq`; `:lambda` for `ineff = :Exponential`), not both."))
    if :sigma_sq in old
        @warn "`hetero = [:sigma_sq]` is deprecated; " *
              "use `hetero = [:sigma_u_sq]`." _id=:sf_deprecated_hetero_sigma_sq maxlog=1
    end
    if :lambda in old
        @warn "`hetero = [:lambda]` is deprecated for `ineff = :Exponential`; use `hetero = [:sigma_u_sq]`. " *
              "`:lambda` remains the scale of `:Weibull` and `:Lomax`." _id=:sf_deprecated_hetero_lambda maxlog=1
    end
    return new
end

# ============================================================================
# Section 7: sfmodel_opt — Unified optimization options
# ============================================================================

"""
    sfmodel_opt(; warmstart_solver=nothing, warmstart_opt=nothing,
                main_solver, main_opt, store_trace=false)

Specify optimization options (solvers and convergence criteria).

All estimation methods (MCI, MSLE, and MLE) use the same optimizer interface,
so no `method` argument is needed.

# Arguments
- `warmstart_solver=nothing`: Optional warmstart optimizer (e.g., `NelderMead()`).
- `warmstart_opt=nothing`: NamedTuple of warmstart options (e.g., `(iterations=200,)`).
  Requires `warmstart_solver`: without a solver there is no warmstart stage to configure,
  so an orphan `warmstart_opt` is refused rather than discarded.
- `main_solver`: Main optimizer (e.g., `Newton()`). Required.
- `main_opt`: NamedTuple of main optimization options. Required.
- `store_trace=false`: When `true`, the main optimizer's iteration trace is captured
  onto the result as `result.trace` so it can be drawn later with
  `plot(result, :convergence)`. Left `false` by default (no overhead).

Every `Optim.Options` keyword given in `warmstart_opt` or `main_opt` is honoured under
every method, `:MLE` included. `warmstart_opt = (iterations = 0,)` skips the warmstart.
When `sfmodel_fit` is called without `optim_options`, every method uses the same
default: a NelderMead warmstart (200 iterations, `g_abstol = 1e-3`) followed by Newton
(200 iterations, `g_abstol = 1e-7`).

With `noise = :Laplace`, a fit whose main stage stops without convergence is continued with `BFGS()` from its
endpoint with the same `main_opt`; not when the main solver is `BFGS()` or `LBFGS()`, or when `main_opt` allows no
iterations. With the default optimizer settings, whether omitted or written out, the Newton stage of a Laplace fit
runs at most 30 iterations before this finish; pass other main settings in `optim_options` to choose other budgets.
"""
function sfmodel_opt(; warmstart_solver = nothing,
                     warmstart_opt = nothing,
                     main_solver,
                     main_opt,
                     store_trace::Bool = false)
    return MCI_Backend.sfmodel_MCI_opt(;
        warmstart_solver=warmstart_solver,
        warmstart_opt=warmstart_opt,
        main_solver=main_solver,
        main_opt=main_opt,
        store_trace=store_trace)
end

# The optimizer used when `sfmodel_fit` is called without `optim_options`.
# Built here once and handed to whichever backend the call reaches, so every method starts
# from the same options. A fit that needs many more Newton iterations than this is usually
# an odd one, so the budget stays at 200.
# The backends' own fallbacks for a direct call (sfmodel_MCI_fit, sfmodel_MSLE_fit,
# sfmodel_panel_fit) hold the same values.
# With noise = :Laplace, MCI and MSLE fits cap the Newton stage of these options at 30 iterations and continue an
# unconverged fit with BFGS.
_default_optim_options() = sfmodel_opt(
    warmstart_solver = MCI_Backend.NelderMead(),
    warmstart_opt    = (iterations = 200, g_abstol = 1e-3),
    main_solver      = MCI_Backend.Newton(),
    main_opt         = (iterations = 200, g_abstol = 1e-7))

# ============================================================================
# Section 8: sfmodel_fit — Unified model estimation
# ============================================================================

# --- Internal helpers: convert cross-sectional method/opt to panel equivalents ---

# Convert MCI/MSLE method spec → PanelMethodSpec
_to_panel_method(m::MCI_Backend.SFMethodSpec) =
    Panel_Backend.PanelMethodSpec(m.method, m.transformation, m.draws, m.n_draws, m.multiRand, m.GPU, m.chunks, m.recycling, m.halton_legacy)

_to_panel_method(m::MSLE_Backend.SFMethodSpec_MSLE) =
    Panel_Backend.PanelMethodSpec(m.method, nothing, m.draws, m.n_draws, m.multiRand, m.GPU, m.chunks, m.recycling, m.halton_legacy)

_to_panel_method(m::Panel_Backend.PanelMethodSpec) = m   # already correct type

# Convert MCI opt spec → PanelOptSpec (identical field layout)
function _to_panel_opt(opt)
    isnothing(opt) && return nothing
    return Panel_Backend.PanelOptSpec(opt.warmstart_solver, opt.warmstart_opt,
                                      opt.main_solver, opt.main_opt)
end

# Convert MCI opt struct → MLE Dict{Symbol,Any}
# The two `Optim.Options` objects travel whole (`:main_opt`, `:warmstart_opt`), so `:MLE`
# honours every option the simulation methods honour (f/x tolerances, time_limit, callback,
# the warmstart's own g_abstol, ...). The scalar keys stay: the legacy `_dicOPT` flow and the
# bootstrap read them.
function _to_mle_opt(opt)
    isnothing(opt) && return nothing
    return Dict{Symbol, Any}(
        :warmstart_solver => opt.warmstart_solver,
        :warmstart_opt    => opt.warmstart_opt,
        :warmstart_maxIT  => isnothing(opt.warmstart_opt) ? nothing :
                             opt.warmstart_opt.iterations,
        :main_solver      => opt.main_solver,
        :main_opt         => opt.main_opt,
        :main_maxIT       => opt.main_opt.iterations,
        :tolerance        => opt.main_opt.g_abstol,
        :store_trace      => opt.main_opt.store_trace,   # propagate trace request to MLE backend
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
    init isa Dict && return init  # MLE backend handles Dicts natively; only reorder numeric vectors
    if spec.ineff == :TruncatedNormal && spec.hetero === :scaling
        K = spec.mci_spec.K
        L = spec.mci_spec.L_scaling  # true scaling-δ count; mci_spec.L is pinned to 1 for :scaling
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

# Reorder init vector from Panel-Backend order → MLE order for panel_TFE.
# Panel HN order: [frontier(K), δ_hscale(L), lnσᵤ²(1), lnσᵥ²(1)]    — matches MLE
# Panel TN order: [frontier(K), δ_hscale(L), μ(1), lnσᵤ²(1), lnσᵥ²(1)]
# MLE   TN order: [frontier(K), μ(1), δ_hscale(L), lnσᵤ²(1), lnσᵥ²(1)]
function _reorder_init_panel_to_mle(spec::UnifiedSpec, init)
    isnothing(init) && return nothing
    init isa Dict && return init  # MLE backend handles Dicts natively; only reorder numeric vectors
    spec.panel_spec === nothing && return init
    K = spec.panel_spec.K
    L = spec.panel_spec.L
    v = collect(init)
    if spec.ineff == :TruncatedNormal
        return vcat(
            v[1:K],            # frontier
            v[K+L+1:K+L+1],    # μ (was after scaling in Panel)
            v[K+1:K+L],        # scaling → hscale
            v[K+L+2:end]       # lnσᵤ², lnσᵥ²
        )
    end
    return v
end

# Reorder MLE result coefficients → MCI canonical order for TruncatedNormal+scaling.
# MLE order: [frontier(K), μ(1), hscale(L), σᵤ²(1), σᵥ²(1)]
# MCI order: [frontier(K), scaling(L), μ(1), σ²(1), σᵥ²(1)]
# Attach the X, y, and frontier-sign used at fit time to the MLE result so
# that diagnostics (residuals, fitted, ...) can operate uniformly across
# backends. MLE backends store data references only as symbols.
# Attach the fit-time X, y, and frontier sign to a backend result so the shared
# diagnostics layer (residuals/fitted/skewness/plot) works uniformly across every
# backend. Backends store data references only as symbols/specs, so we recover the
# actual arrays here from whichever spec the fit used.
function _attach_diagnostic_data(spec::UnifiedSpec, result)
    # panel_TFE fit (via MLE *or* the MCI/MSLE simulation backends): the Wang–Ho
    # likelihood is evaluated on the within-firm demeaned data (see getvar(PFEWHH/
    # PFEWHT), SFgetvars.jl, and the panel-sim _PanelInternalSpec y_tilde/x_tilde).
    # Attach the demeaned depvar/frontier — not the raw stacked NT data — so
    # diagnostics operate on the same scale as the fit. Per-row within-firm
    # demeaning is grouping-invariant, so it reproduces each backend's own
    # transform. The frontier sign is preserved unchanged.
    if spec.panel_spec !== nothing
        ps = spec.panel_spec
        # Attach the firm boundary offsets too: the Wang–Ho jlms uses the
        # *non-demeaned* scaling h_it, so `residuals(:v)`/`fitted(:response)` must
        # within-firm demean the inefficiency to stay on the demeaned scale;
        # diagnostics uses these offsets to do that.
        return merge(result, (_sf_depvar        = Panel_Backend.sf_panel_demean(ps.depvar, ps.offsets),
                              _sf_frontier      = Panel_Backend.sf_panel_demean(ps.frontier, ps.offsets),
                              _sf_sign          = ps.sign,
                              _sf_panel_offsets = ps.offsets))
    end
    # Cross-sectional MLE/MCI/MSLE: the raw fit-time data live on the MCI spec.
    src = spec.mci_spec
    if src !== nothing
        return merge(result, (_sf_depvar   = src.depvar,
                              _sf_frontier = src.frontier,
                              _sf_sign     = src.sign))
    end
    # panel_TFE_CSW / panel_TRE: MLE-only estimators with no mci_spec or panel_spec.
    # Recover the raw stacked data from the MLE spec so their diagnostic surface is
    # available too. These are not Wang–Ho within-demeaned models,
    # so residuals are on the raw scale (`type_sign` is +1 prod / −1 cost).
    ms = spec.mle_spec
    ms === nothing && return result
    return merge(result, (_sf_depvar   = ms.depvar,
                          _sf_frontier = ms.frontier,
                          _sf_sign     = ms.type_sign))
end

# The start vectors of a reordered MLE result: `main_ini` and `warmstart_ini`
# are coefficient vectors like `coeff`, so they are published in the same canonical order.
# `nothing` (no warmstart ran) passes through; so does anything that is not a
# coefficient-length vector. The trace needs no reordering: it holds only the iteration,
# the objective and the gradient norm.
_permute_start(v::AbstractVector, perm, n) = length(v) == n ? v[perm] : v
_permute_start(v, perm, n) = v

# User-supplied `varnames` follow the canonical order of Section 8, which for the truncated
# normal with hetero = :scaling and for panel_TFE with a truncated normal is K frontier
# names, L scaling names, μ, ln σᵤ², ln σᵥ². The analytic MLE
# backend orders these parameters frontier, μ, scaling, ln σᵤ², ln σᵥ² and puts the names on
# its rows by position (the table printed during the fit, `result.table`,
# `result.list[:varnames]`), so it receives them in that order: the inverse of `perm` in the
# two reorders below, which then publish `varnames` in the canonical order again. Names of any
# other length pass through unchanged (the MCI and panel specs, built first, have already
# refused a wrong length).
function _varnames_canon_to_mle(varnames, K::Int, L::Int)
    (varnames === nothing || length(varnames) != K + L + 3) && return varnames
    return varnames[vcat(1:K, K+L+1, (K+1):(K+L), (K+L+2):(K+L+3))]
end

function _reorder_result_mle_to_mci(spec::UnifiedSpec, result)
    if spec.ineff == :TruncatedNormal && spec.hetero === :scaling
        K = spec.mci_spec.K
        L = spec.mci_spec.L_scaling  # true scaling-δ count; mci_spec.L is pinned to 1 for :scaling
        n = length(result.coeff)
        # perm: [1:K, (K+2):(K+1+L), K+1, (K+2+L):n]
        perm = vcat(1:K, (K+2):(K+1+L), K+1, (K+2+L):n)
        new_coeff    = result.coeff[perm]
        new_std_err  = result.std_err[perm]
        new_vcov     = result.var_cov_mat[perm, perm]
        new_hessian  = result.Hessian[perm, perm]
        merged = merge(result, (coeff = new_coeff,
                                std_err = new_std_err,
                                var_cov_mat = new_vcov,
                                Hessian = new_hessian,
                                main_ini = _permute_start(result.main_ini, perm, n),
                                warmstart_ini = _permute_start(result.warmstart_ini, perm, n)))
        # `varnames` is consumed by accessors.jl `_row_names` → `coeftable(r)`.
        # Published labels must be in MCI order. The MLE backend returns its names in
        # MLE order: auto-generated ones and user-supplied ones alike (handed to it
        # through _varnames_canon_to_mle); they are permuted here. The skip when they
        # equal the MCI-spec names only covers names that the permutation leaves unchanged.
        if hasproperty(result, :varnames) && length(result.varnames) >= n &&
           result.varnames != spec.mci_spec.varnames
            new_varnames = result.varnames[perm]
            merged = merge(merged, (varnames = new_varnames,))
        end
        return merged
    end
    return result
end

# Reorder MLE result coefficient vector → Panel-Backend canonical order for panel_TFE.
# Only TruncatedNormal needs reordering (μ and δ_hscale swap); HN orders match.
# MLE   TN order: [frontier(K), μ(1), δ_hscale(L), lnσᵤ²(1), lnσᵥ²(1)]
# Panel TN order: [frontier(K), δ_hscale(L), μ(1), lnσᵤ²(1), lnσᵥ²(1)]
function _reorder_result_mle_to_panel(spec::UnifiedSpec, result)
    if spec.ineff == :TruncatedNormal && spec.panel_spec !== nothing
        K = spec.panel_spec.K
        L = spec.panel_spec.L
        n = length(result.coeff)
        # perm: [1:K, (K+2):(K+1+L), K+1, (K+2+L):n]
        perm = vcat(1:K, (K+2):(K+1+L), K+1, (K+2+L):n)
        new_coeff    = result.coeff[perm]
        new_std_err  = result.std_err[perm]
        new_vcov     = result.var_cov_mat[perm, perm]
        new_hessian  = result.Hessian[perm, perm]
        merged = merge(result, (coeff = new_coeff,
                                std_err = new_std_err,
                                var_cov_mat = new_vcov,
                                Hessian = new_hessian,
                                main_ini = _permute_start(result.main_ini, perm, n),
                                warmstart_ini = _permute_start(result.warmstart_ini, perm, n)))
        # `varnames` is consumed by accessors.jl `_row_names` → `coeftable(r)`.
        # Keep labels aligned with the reordered values for the public API.
        # (The CoefTable stored at `result.table` was built by the MLE backend
        # before this reorder and intentionally left in MLE order — it is the
        # during-fit display artifact, not a public-API surface.)
        # The backend's names are in MLE order, user-supplied ones too
        # (_varnames_canon_to_mle); the skip when they equal the panel-spec names only
        # covers names the permutation leaves unchanged (mirrors the cross-sectional sibling).
        if hasproperty(result, :varnames) && length(result.varnames) >= n &&
           result.varnames != spec.panel_spec.varnames
            new_varnames = result.varnames[perm]
            merged = merge(merged, (varnames = new_varnames,))
        end
        return merged
    end
    return result
end

"""
    sfmodel_fit(; spec::UnifiedSpec, method::UnifiedMethod, init=nothing,
                optim_options=nothing, jlms_bc_index=true, marginal=true,
                integration_check=false, show_table=true, shape_warning=true, verbose=true)

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
- `init=nothing`: Initial parameter values from `sfmodel_init()`. Under `:MCI` and `:MSLE`, and
  under `:MLE` for `panel_TFE_CSW` and `panel_TRE`, starting values at or beyond a parameter's
  numerical bounds are refused (an error) for a homoscedastic parameter and give a warning for a
  heteroscedastic one; a fit held at its starting values (no warmstart stage and `iterations = 0`
  in the main stage) is not checked.
- `optim_options=nothing`: Optimization options from `sfmodel_opt()`. If `nothing`, every
  method uses the same default: NelderMead (200 iterations, `g_abstol = 1e-3`) as a
  warmstart, then Newton (200 iterations, `g_abstol = 1e-7`). For `noise = :Laplace` the
  Newton stage of this default runs at most 30 iterations, and a fit it leaves unconverged
  continues with BFGS (see `sfmodel_opt`).
- `jlms_bc_index::Bool=true`: Compute JLMS and BC efficiency indices.
- `marginal::Bool=true`: Compute the observation-level marginal effects of the inefficiency
  determinants on E(u): cross-sectional models and `:panel_TFE` (every method). For a Lomax fit
  they are NaN where α ≤ 1, where E(u) does not exist, with a warning (unless
  `shape_warning = false`). `:panel_TFE_CSW` and
  `:panel_TRE` have no inefficiency determinants; their `marginal` and `marginal_mean` are empty
  tuples.
- `integration_check=false`: Run the integration check after an MCI or MSLE fit: at the
  estimate, the simulated log-likelihood is evaluated again with four times the draws, and the
  one-step shift −V·(g_4S − g_S) is reported in standard errors; each part reads OK, CAUTION or
  FAIL and the result is stored in `result.integration_check`, separately from `converged` and
  `redflag`. `true` uses the default thresholds (0.1 and 1 for the change in the
  log-likelihood, 0.1 and 0.5 standard errors for the shift); a NamedTuple with any of
  `ll_caution`, `ll_fail`, `shift_caution`, `shift_fail` changes them. The thresholds are
  heuristics. Cross-sectional and `panel_TFE` fits; ignored by `method = :MLE`. See Convergence
  Warnings in the manual.
- `show_table::Bool=true`: Display estimation results table.
- `shape_warning::Bool=true`: Print the warning of the shape-range check (an MCI Gamma, Weibull or
  Lomax fit under the default transformation rule, or an MSLE Lomax fit, whose estimated shape lies
  below its threshold) and the warning that a Lomax fit's marginal effects are undefined where
  α ≤ 1. `false` prints neither; `result.shape_check` is stored either way. Ignored by `method = :MLE`.
- `verbose::Bool=true`: Verbose output during estimation.

# Returns
Backend-specific result (NamedTuple with fields including `converged`, `loglikelihood`,
`coeff` (under `:MCI` and `:MSLE`, and under `:MLE` for `panel_TFE_CSW` and `panel_TRE`, a homoscedastic
coefficient that ended beyond its numerical bound is reported at the bound; see `bound_check`), `std_err`,
`jlms`, `bc`, `redflag` (0, 1 or 2; 1 also for a boundary solution),
`hessian_flag` (the Hessian part of `redflag`), `bound_check` (each parameter at or beyond its
numerical bound; see Parameter Bounds in the manual), etc.). MCI and MSLE results, cross-sectional
and `panel_TFE`,
also carry `integration_check`: the result of the integration check
(`sfmodel_fit(..., integration_check = true)`), or `nothing` when it is off. They also carry
`shape_check`: the verdict of the shape-range check (MCI Gamma, Weibull and Lomax fits under the
default transformation rule, and MSLE Lomax fits, print a warning when the estimated shape lies where
the simulated likelihood is unreliable, unless `shape_warning = false`), or `nothing` for the models that
it does not cover.
"""
function sfmodel_fit(;
    spec::UnifiedSpec,
    method::UnifiedMethod,
    init = nothing,
    optim_options = nothing,
    jlms_bc_index::Bool = true,
    marginal::Bool = true,
    integration_check = false,
    show_table::Bool = true,
    shape_warning::Bool = true,
    verbose::Bool = true)

    # An invalid `integration_check` is an error before any fit, under every method; `:MLE` ignores a
    # valid one.
    SFAccuracy.sf_integration_settings(integration_check)

    # One default for every method, built here once.
    optim_options === nothing && (optim_options = _default_optim_options())

    # --- MLE dispatch (works for all datatypes) ---
    if method isa MLEMethodSpec
        if spec.mle_spec === nothing
            _mle_unavailable_error(spec)
        end
        # For panel_TFE, init was built by Panel_Backend (Panel order); translate
        # to MLE order. For cross-sectional, reorder MCI/MSLE init to MLE order.
        mle_init = (spec.datatype == :panel_TFE) ?
                   _reorder_init_panel_to_mle(spec, init) :
                   _reorder_init_mci_to_mle(spec, init)
        # The backend always gets an explicit dictionary, never `nothing`. The MLE backend
        # keeps a module-level `_dicOPT`, and one written by the deprecated
        # `MLE_Backend.sfmodel_opt()` carries a `_user_set` marker so that a bare
        # *deprecated* `sfmodel_fit()` keeps honouring it. That marker says nothing about
        # this call, and `nothing` could not be told apart from a bare legacy fit.
        # (`optim_options` is never `nothing` here: the unified default was filled in above.)
        mle_opt  = _to_mle_opt(optim_options)
        raw = MLE_Backend.sfmodel_fit(;
            spec = spec.mle_spec,
            init = mle_init,
            optim_options = mle_opt,
            jlms_bc_index = jlms_bc_index,
            marginal = marginal,
            show_table = show_table,
            verbose = verbose)
        reordered = (spec.datatype == :panel_TFE) ?
                    _reorder_result_mle_to_panel(spec, raw) :
                    _reorder_result_mle_to_mci(spec, raw)
        return _wrap_result(_attach_diagnostic_data(spec, reordered), :MLE, spec)

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

        # Attach demeaned fit-time data so residuals/fitted/plot work for panel-sim
        # (MCI/MSLE) fits, whose result.model is a _PanelInternalSpec carrying only
        # y_tilde/x_tilde — not the depvar/frontier the diagnostics layer expects.
        panel_result = Panel_Backend.sfmodel_panel_fit(;
                spec          = spec.panel_spec,
                method        = panel_method,
                init          = init,
                optim_options = panel_opt,
                jlms_bc_index = jlms_bc_index,
                marginal      = marginal,
                integration_check = integration_check,
                show_table    = show_table,
                shape_warning = shape_warning,
                verbose       = verbose)
        return _wrap_result(_attach_diagnostic_data(spec, panel_result), :Panel, spec)

    # --- Cross-sectional MCI ---
    elseif method isa MCI_Backend.SFMethodSpec
        if spec.mci_spec === nothing
            error("`method=:MCI` is not available for this model configuration.\n" *
                  "  Supported methods: " * join(string.(_get_supported_methods(spec)), ", "))
        end
        return _wrap_result(
            MCI_Backend.sfmodel_MCI_fit(;
                spec = spec.mci_spec,
                method = method,
                init = init,
                optim_options = optim_options,
                jlms_bc_index = jlms_bc_index,
                marginal = marginal,
                integration_check = integration_check,
                show_table = show_table,
                shape_warning = shape_warning,
                verbose = verbose),
            :MCI, spec)

    # --- Cross-sectional MSLE ---
    elseif method isa MSLE_Backend.SFMethodSpec_MSLE
        if spec.msle_spec === nothing
            error("`method=:MSLE` is not available for this model configuration.\n" *
                  "  Supported methods: " * join(string.(_get_supported_methods(spec)), ", "))
        end
        return _wrap_result(
            MSLE_Backend.sfmodel_MSLE_fit(;
                spec = spec.msle_spec,
                method = method,
                init = init,
                optim_options = optim_options,
                jlms_bc_index = jlms_bc_index,
                marginal = marginal,
                integration_check = integration_check,
                show_table = show_table,
                shape_warning = shape_warning,
                verbose = verbose),
            :MSLE, spec)

    else
        error("Unrecognized method type: $(typeof(method))")
    end
end

# --- Helper: list supported methods for a given spec ---
function _get_supported_methods(spec::UnifiedSpec)
    methods = Symbol[]
    spec.mci_spec   !== nothing && push!(methods, :MCI)
    spec.msle_spec  !== nothing && push!(methods, :MSLE)
    spec.panel_spec !== nothing && push!(methods, :MCI, :MSLE)  # panel_TFE serves MCI/MSLE via panel_spec
    spec.mle_spec   !== nothing && push!(methods, :MLE)
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
- `ineff::Symbol`: Inefficiency distribution (`:HalfNormal`, `:TruncatedNormal`, `:Exponential`, `:Weibull`, `:Lognormal`, `:Lomax`, `:Rayleigh`, `:Gamma` (MCI only), `:Erlang2`, `:Erlang3`, `:Erlang4`).
- `type::Symbol=:prod`: `:prod` (production) or `:cost`.
- `varnames`, `eqnames`, `eq_indices`: Optional variable/equation names.

# Returns
`Panel_Backend.PanelModelSpec{T}`
"""
function sfmodel_panel_spec(; kwargs...)
    return Panel_Backend.sfmodel_panel_spec(; kwargs...)
end

"""
    sfmodel_panel_method(; method, n_draws=4096, transformation=nothing,
                           multiRand=true, GPU=false, chunks=10, recycling=false,
                           distinct_Halton_length=nothing)

Specify the estimation method for the panel model.

# Arguments
- `method::Symbol`: `:MSLE` or `:MCI`.
- `n_draws::Int=4096`: Number of Halton draws per firm (2^12). Use powers of two.
- `transformation=nothing`: MCI transformation (`:expo_rule`, `:logistic_1_rule`, `:logistic_2_rule`).
  MCI only; `method = :MSLE` refuses it. Under the default rule, MCI fits whose estimated shape lies
  in a range where that rule is unreliable (Gamma k, Weibull k, Lomax α below a threshold) print a
  warning (unless the fit passes `shape_warning = false`) and store the verdict in `result.shape_check`.
- `draws=nothing`: User-supplied uniform draws in `[0, 1)`. A length-D vector, a 1×D row or a
  D×1 column is one sequence shared by every firm; an N×D matrix is one row of draws per FIRM
  (N = number of firms, not NT). Any other row count is an error. Supplying `draws` forces
  `multiRand=false` and makes `n_draws` irrelevant. If `nothing`, Halton draws are generated. The fit keeps
  the draws for `predict()`. When the array already has the data's element type and lives where the data
  live (a host `Array`, or a `CuArray` with `GPU = true`), the fit uses the array itself, not a copy, so
  changing the array afterwards changes what `predict()` computes.
- `multiRand::Bool=true`: generation mode, used only when `draws` is not supplied. `true` gives observation i
  (firm i in panel models) the `n_draws` consecutive points (i − 1)·n_draws + 1, …, i·n_draws of one base-2
  Halton sequence of N × `n_draws` points, so no point is used twice unless `recycling` or the deprecated
  `distinct_Halton_length` is given; `false` gives every observation the same first `n_draws` points.
- `GPU::Bool=false`: Use GPU acceleration.
- `chunks::Int=10`: Number of groups of observations (firms, for panel models) processed at a time (MCI/MSLE
  only). The memory of the likelihood, its derivatives and the efficiency indices is proportional to
  N × `n_draws` / `chunks`, on the CPU and on the GPU.
- `recycling=false`: reuse of Halton points across firms, for generated per-firm draws only
  (`multiRand = true`, no `draws`). `false` (default): no point is reused. `true`: the sequence is capped at
  L = 2^24 − 1 = 16,777,215 points and restarts from its first point when N × `n_draws` exceeds L; firm i
  receives the points at positions mod1((i − 1)·n_draws + j, L), j = 1, …, n_draws. An integer L with
  `n_draws` ≤ L ≤ 2^24 − 1 sets another cap. When N × `n_draws` exceeds L, firms i and i + L/g receive
  identical draws, where g is the greatest common divisor of `n_draws` and L; the fit warns when the sample
  has more than L/g firms. With `recycling = true` (L = 2^24 − 1, an odd number) any power of two for
  `n_draws` gives g = 1. With `draws` or `multiRand = false`, `recycling` is an error.
- `distinct_Halton_length=nothing`: deprecated; it gives a warning. Earlier `:MCI` and `:MSLE` scripts that set
  it still run and, with the same `n_draws`, get the draws they got before 2.0.0. To reproduce the draws of a
  fit of those versions that used the default length, give `recycling = 2^15 - 1` and that fit's `n_draws`
  (1,024 by default): the draws are identical whenever N × `n_draws` is at least 32,767. Use `recycling` for
  new work; giving both is an error.
"""
function sfmodel_panel_method(; kwargs...)
    return Panel_Backend.sfmodel_panel_method(; kwargs...)
end

"""
    sfmodel_panel_init(; spec, kwargs...)

Compute or assemble initial values for the panel model.

# Arguments
- `spec`: Panel model specification from `sfmodel_panel_spec()`.
- Keyword overrides: `frontier`, `scaling`, `ln_sigma_u_sq`, `ln_sigma_v_sq`.

The names of versions before 2.0.0 are accepted with a deprecation warning: `ln_sigma_sq` for `ln_sigma_u_sq`,
and for `ineff = :Exponential` also `ln_lambda`. Giving an old and a new name together is an error.

# Returns
`Vector{Float64}` of initial parameter values.
"""
function sfmodel_panel_init(; kwargs...)
    kw = Dict{Symbol,Any}(kwargs)
    if get(kw, :spec, nothing) isa Panel_Backend.PanelModelSpec
        _alias_init_names!(kw, kw[:spec].ineff)
    end
    return Panel_Backend.sfmodel_panel_init(; kw...)
end

"""
    sfmodel_panel_opt(; warmstart_solver=nothing, warmstart_opt=nothing, main_solver, main_opt,
                        store_trace=false)

Specify optimization options for the panel model.

# Arguments
- `warmstart_solver`: Optional warmstart optimizer (e.g., `NelderMead()`).
- `warmstart_opt`: NamedTuple of warmstart options.
- `main_solver`: Main optimizer (e.g., `Newton()`). Required.
- `main_opt`: NamedTuple of main optimization options. Required.
- `store_trace=false`: When `true`, or when `main_opt` itself sets `store_trace = true`, the main
  optimizer's iteration trace (iterations, objective values and gradient norms) is stored in the
  result as `trace`; otherwise `trace` is `nothing`. `plot(result, :convergence)` draws it from the
  result of `sfmodel_fit` or `sfmodel_panel_fit`.
"""
function sfmodel_panel_opt(; kwargs...)
    return Panel_Backend.sfmodel_panel_opt(; kwargs...)
end

"""
    sfmodel_panel_fit(; spec, method, init=nothing, optim_options=nothing, jlms_bc_index=true,
                        marginal=true, integration_check=false, show_table=true, shape_warning=true,
                        verbose=true)

Estimate the Wang and Ho (2010) panel stochastic frontier model.

# Arguments
- `spec`: Panel model specification from `sfmodel_panel_spec()`.
- `method`: Method specification from `sfmodel_panel_method()`.
- `init=nothing`: Initial values from `sfmodel_panel_init()`. Starting values at or beyond a
  parameter's numerical bounds are refused (an error) for a homoscedastic parameter and give a
  warning for a heteroscedastic one; a fit held at its starting values (no warmstart stage and
  `iterations = 0` in the main stage) is not checked.
- `optim_options=nothing`: Optimization options from `sfmodel_panel_opt()`.
- `jlms_bc_index::Bool=true`: Compute JLMS and BC efficiency indices.
- `marginal::Bool=true`: Compute the observation-level marginal effects of the inefficiency
  determinants on E(u), stored in `marginal` (their sample means in `marginal_mean`). For a
  Lomax fit they are NaN where α ≤ 1, where E(u) does not exist, with a warning (unless
  `shape_warning = false`).
- `integration_check=false`: Run the integration check after the fit: at the estimate, the simulated
  log-likelihood is evaluated again with four times the draws, and the one-step shift −V·(g_4S − g_S)
  is reported in standard errors; each part reads OK, CAUTION or FAIL and the result is stored in
  `result.integration_check`, separately from `converged` and `redflag`. `true` uses the default
  thresholds (0.1 and 1 for the change in the log-likelihood, 0.1 and 0.5 standard errors for the
  shift); a NamedTuple with any of `ll_caution`, `ll_fail`, `shift_caution`, `shift_fail` changes
  them. The thresholds are heuristics.
- `show_table::Bool=true`: Display estimation results table.
- `shape_warning::Bool=true`: Print the warning of the shape-range check (an MCI Gamma, Weibull or
  Lomax fit under the default transformation rule, or an MSLE Lomax fit, whose estimated shape lies
  below its threshold) and the warning that a Lomax fit's marginal effects are undefined where
  α ≤ 1. `false` prints neither; `result.shape_check` is stored either way.
- `verbose::Bool=true`: Verbose output during estimation.

# Returns
NamedTuple with fields including `converged`, `loglikelihood`, `coeff` (a homoscedastic coefficient that
ended beyond its numerical bound is reported at the bound; see `bound_check`), `std_err`, `var_cov_mat`,
`jlms` (observation-level, length Σ Tᵢ), `bc` (observation-level, length Σ Tᵢ), `redflag` (0, 1
or 2; 1 also for a boundary solution), `hessian_flag` (the Hessian part of `redflag`), `bound_check`
(each parameter at or beyond its numerical bound; see Parameter Bounds in the manual),
`integration_check`, `shape_check` and `table`.

`integration_check` is the result of the integration check (the option `integration_check`), or
`nothing` when the check is off: a NamedTuple with `status` (`:OK`, `:CAUTION`, `:FAIL`,
`:not_available` or `:error`), `reason`, `n_draws` and `n_draws_check` (the fit's draws and
four times as many), `loglik`, `loglik_check`, `delta_loglik`, `delta_loglik_per_obs`,
`ll_verdict`, `shift_available`, `shift_note`, `shift`, `shift_se`, `max_shift_se`,
`max_shift_param`, `max_shift_estimate`, `max_shift_onestep` (the largest shift in standard
errors, the parameter it belongs to, its estimate and its value after the shift),
`shift_verdict`, `thresholds` and `seconds`. It is separate from `converged` and `redflag`.

`shape_check` is the verdict of the shape-range check, or `nothing` for the models that it does not
cover: a NamedTuple with `method`, `family`, `parameter`, `rule`, `threshold`, `in_range`, `n_below`,
`n` (1: one shape per model), `share` and `min_value`. MCI Gamma, Weibull and Lomax fits under the
default transformation rule, and MSLE Lomax fits, print a warning when the estimated shape lies below
`threshold` (`in_range` is then `true`), unless the fit passes `shape_warning = false`. It is separate
from `converged`, `redflag` and `integration_check`.
"""
function sfmodel_panel_fit(; kwargs...)
    return _wrap_result(Panel_Backend.sfmodel_panel_fit(; kwargs...), :Panel)
end

# ============================================================================
# Exports
# ============================================================================

export sfmodel_spec, sfmodel_method, sfmodel_init, sfmodel_opt, sfmodel_fit
export sfmodel_panel_spec, sfmodel_panel_method, sfmodel_panel_init
export sfmodel_panel_opt, sfmodel_panel_fit
export sfmodel_MixTable, sfmodel_ChiSquareTable, sfmodel_CI, sfmodel_MoMTest
export @useData, @depvar, @frontier, @zvar, @id
export SFResult
export lrtest, sf_vs_ols, efficiency_summary, residual_diagnostics, sf_skewness_test, LRTestResult, EfficiencySummary

# Re-export StatsAPI functions extended for SFResult
using StatsAPI: coef, vcov, stderror, loglikelihood, nobs, dof, dof_residual,
                aic, bic, confint, coeftable, predict, residuals, fitted
export coef, vcov, stderror, loglikelihood, nobs, dof, dof_residual,
       aic, bic, confint, coeftable, predict, residuals, fitted

# Re-export MLE utility functions applicable to all backends
using .MLE_Backend: sfmodel_MixTable, sfmodel_ChiSquareTable, sfmodel_CI, sfmodel_MoMTest

end # module SFrontiers
