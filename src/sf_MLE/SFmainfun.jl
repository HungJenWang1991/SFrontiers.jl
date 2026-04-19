# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

########################################################
####                sfmodel_spec()                  ####
########################################################

"""
    sfmodel_spec(; depvar, frontier, zvar=nothing, noise=:Normal, ineff,
                 hetero=Symbol[], type=:prod, panel=nothing,
                 id=nothing, message=false)

Specify a stochastic frontier model for analytic MLE estimation.

# Arguments
- `depvar`: Vector or matrix of the dependent variable.
- `frontier`: Matrix of frontier variables (include a constant column if needed).
- `zvar=nothing`: Matrix of variables for heteroscedasticity / scaling function.
  Shared across all heteroscedastic parameters specified by `hetero`.
- `noise::Symbol=:Normal`: Noise distribution (only `:Normal` supported).
- `ineff::Symbol`: Inefficiency distribution — `:HalfNormal`, `:TruncatedNormal`,
  or `:Exponential`.
- `hetero=Symbol[]`: Which inefficiency parameters are heteroscedastic (functions
  of `zvar`). Valid options depend on `ineff`:
  - `:HalfNormal` → `[:sigma_sq]`
  - `:TruncatedNormal` → `[:mu]`, `[:sigma_sq]`, or both
  - `:Exponential` → `[:lambda]`
  - Use `hetero=:scaling` for the scaling-property model (requires `zvar`).
- `type::Symbol=:prod`: `:prod` or `:production` (production frontier) or
  `:cost` (cost frontier).
- `panel=nothing`: Panel model type — `nothing` for cross-sectional, or one of
  `:TFE_WH2010`, `:TFE_CSW2014`, `:TRE`, `:TimeDecay`, `:Kumbhakar1990`.
- `id=nothing`: Vector of individual/firm identifiers (panel only).
- `message::Bool=false`: Print confirmation message.

# Returns
An `SFModelSpec_MLE{T}` struct containing all model specifications.

# Examples
```julia
# Cross-sectional, half-normal
spec = sfmodel_spec(depvar=y, frontier=X, ineff=:HalfNormal, type=:prod)

# Cross-sectional, truncated-normal with heteroscedastic μ
spec = sfmodel_spec(depvar=y, frontier=X, zvar=Z,
    ineff=:TruncatedNormal, hetero=[:mu], type=:prod)

# Scaling-property model
spec = sfmodel_spec(depvar=y, frontier=X, zvar=Z,
    ineff=:TruncatedNormal, hetero=:scaling, type=:prod)

# Panel model
spec = sfmodel_spec(depvar=y, frontier=X, zvar=Z,
    ineff=:TruncatedNormal, panel=:TFE_WH2010,
    id=id, type=:prod)
```
"""
function sfmodel_spec(;
    depvar, frontier, zvar=nothing,
    noise::Symbol=:Normal,
    ineff::Symbol,
    hetero::Union{Vector{Symbol}, Symbol}=Symbol[],
    type::Symbol=:prod,
    panel::Union{Nothing, Symbol}=nothing,
    id=nothing,
    varnames::Union{Nothing, Vector{String}}=nothing,
    eqnames::Union{Nothing, Vector{String}}=nothing,
    eq_indices::Union{Nothing, Vector{Int}}=nothing,
    frontier_names::Union{Nothing, Vector{Symbol}}=nothing,
    zvar_names::Union{Nothing, Vector{Symbol}}=nothing,
    message::Bool=false)

    #* ---- Validate noise ----
    noise == :Normal || throw("Only `noise=:Normal` is supported for analytic MLE. " *
        "For other noise distributions, use SFrontiers.")

    #* ---- Validate type ----
    local type_sign::Int
    if type in (:prod, :production)
        type_sign = 1
    elseif type == :cost
        type_sign = -1
    else
        throw("Invalid `type=:$type`. Use `:prod`, `:production`, or `:cost`.")
    end

    #* ---- Map ineff symbol to dist code and validate ----
    local dist_code::Symbol
    if ineff == :HalfNormal
        dist_code = :h
    elseif ineff == :TruncatedNormal
        dist_code = :t
    elseif ineff == :Exponential
        dist_code = :e
    else
        throw("Invalid `ineff=:$ineff`. Supported: `:HalfNormal`, `:TruncatedNormal`, `:Exponential`.")
    end

    #* ---- Handle scaling vs standard hetero ----
    local is_scaling::Bool = (hetero === :scaling)

    if is_scaling
        # Scaling-property model requires TruncatedNormal
        ineff == :TruncatedNormal || throw(
            "The scaling-property model (`hetero=:scaling`) requires `ineff=:TruncatedNormal`.")
        zvar === nothing && throw(
            "The scaling-property model (`hetero=:scaling`) requires `zvar` to be specified.")
        dist_code = :s  # internal code for Trun_Scale
    end

    #* ---- Validate hetero options against ineff ----
    if hetero isa Vector{Symbol} && !isempty(hetero)
        valid_hetero = if ineff == :HalfNormal
            [:sigma_sq]
        elseif ineff == :TruncatedNormal
            [:mu, :sigma_sq]
        elseif ineff == :Exponential
            [:lambda]
        else
            Symbol[]
        end
        for h in hetero
            h in valid_hetero || throw(
                "Invalid `hetero` option `:$h` for `ineff=:$ineff`. " *
                "Valid options: $(valid_hetero).")
        end
        zvar === nothing && throw(
            "`zvar` must be specified when `hetero` is non-empty.")
    end

    #* ---- Determine modelid ----
    local modelid::Type{<:Sfmodeltype}

    if panel === nothing  # cross-sectional
        if dist_code == :t
            modelid = Trun
        elseif dist_code == :h
            modelid = Half
        elseif dist_code == :e
            modelid = Expo
        elseif dist_code == :s
            modelid = Trun_Scale
        end
    elseif panel == :TFE_WH2010
        if dist_code == :t;  modelid = PFEWHT
        elseif dist_code == :h;  modelid = PFEWHH
        else; throw("Panel TFE_WH2010 only supports `ineff=:TruncatedNormal` or `:HalfNormal`.")
        end
    elseif panel == :TFE_CSW2014
        dist_code == :h || throw("Panel TFE_CSW2014 only supports `ineff=:HalfNormal`.")
        modelid = PFECSWH
    elseif panel == :TRE
        if dist_code == :h;  modelid = PTREH
        elseif dist_code == :t;  modelid = PTRET
        else; throw("Panel TRE only supports `ineff=:HalfNormal` or `:TruncatedNormal`.")
        end
    #= TimeDecay and Kumbhakar1990 disabled in keyword API — require time-varying gamma variables
    elseif panel == :TimeDecay
        dist_code == :t || throw("Panel TimeDecay only supports `ineff=:TruncatedNormal`.")
        modelid = PanDecay
    elseif panel == :Kumbhakar1990
        dist_code == :t || throw("Panel Kumbhakar1990 only supports `ineff=:TruncatedNormal`.")
        modelid = PanKumb90
    =#
    else
        throw("Invalid `panel=:$panel`. Supported: `:TFE_WH2010`, `:TFE_CSW2014`, `:TRE`.")
    end

    #* ---- Normalize data to arrays ----
    local depvar_vec::Vector{Float64}
    local frontier_mat::Matrix{Float64}

    if depvar isa Vector
        depvar_vec = convert(Vector{Float64}, depvar)
    elseif depvar isa Matrix
        size(depvar, 2) == 1 || throw("`depvar` must be a vector or single-column matrix.")
        depvar_vec = convert(Vector{Float64}, vec(depvar))
    else
        throw("`depvar` must be a Vector or Matrix.")
    end

    if frontier isa Vector
        frontier_mat = reshape(convert(Vector{Float64}, frontier), :, 1)
    elseif frontier isa Matrix
        frontier_mat = convert(Matrix{Float64}, frontier)
    else
        throw("`frontier` must be a Vector or Matrix.")
    end

    N = length(depvar_vec)
    K = size(frontier_mat, 2)
    size(frontier_mat, 1) == N || throw("`frontier` row count must match `depvar` length.")

    local zvar_mat::Matrix{Float64}
    local scaling_zvar_mat::Union{Nothing, Matrix{Float64}} = nothing
    local L::Int

    if zvar !== nothing
        if zvar isa Vector
            zvar_mat = reshape(convert(Vector{Float64}, zvar), :, 1)
        elseif zvar isa Matrix
            zvar_mat = convert(Matrix{Float64}, zvar)
        else
            throw("`zvar` must be a Vector or Matrix.")
        end
        size(zvar_mat, 1) == N || throw("`zvar` row count must match `depvar` length.")
        L = size(zvar_mat, 2)
    else
        zvar_mat = ones(N, 1)
        L = 1
    end

    # For scaling model, store the original zvar and use ones internally
    if is_scaling
        scaling_zvar_mat = zvar_mat
        zvar_mat = ones(N, 1)
        L = 1
    end

    # Normalize panel vectors
    local idvar_vec = nothing
    if id !== nothing
        idvar_vec = id isa Vector ? id : vec(id)
    end

    #* ---- Populate _dicM and tagD for backward compatibility ----
    _sfmodel_spec_populate_dicM!(
        depvar_vec, frontier_mat, zvar_mat, scaling_zvar_mat,
        dist_code, type, type_sign, panel, modelid,
        hetero, is_scaling, ineff,
        idvar_vec, N, K, L;
        frontier_names=frontier_names, zvar_names=zvar_names)

    #* ---- Syntax check (uses _dicM) ----
    SFmle.checksyn(tagD[:modelid])

    #* ---- Build varnames/eqnames/eq_indices if not provided ----
    if varnames === nothing || eqnames === nothing || eq_indices === nothing
        # Use the varlist from getvar at fit time; provide empty placeholders
        varnames = varnames === nothing ? String[] : varnames
        eqnames  = eqnames === nothing ? String[] : eqnames
        eq_indices = eq_indices === nothing ? Int[] : eq_indices
    end

    #* ---- Build and return struct ----
    spec = SFModelSpec_MLE{Float64}(
        depvar_vec, frontier_mat, zvar_mat,
        noise, ineff, hetero, type_sign,
        panel, idvar_vec,
        is_scaling, scaling_zvar_mat,
        N, K, L, modelid,
        varnames, eqnames, eq_indices,
        frontier_names, zvar_names)

    if message
        printstyled("A SFModelSpec_MLE from sfmodel_spec() is generated.\n"; color=:green)
    end

    global _MLE_spec = spec
    return spec

end  # end of sfmodel_spec()


#* ---- Internal: populate _dicM and tagD for backward compatibility ----

function _sfmodel_spec_populate_dicM!(
        depvar_vec, frontier_mat, zvar_mat, scaling_zvar_mat,
        dist_code, type_sym, type_sign, panel, modelid,
        hetero, is_scaling, ineff,
        idvar_vec, N, K, L;
        frontier_names::Union{Nothing,Vector{Symbol}}=nothing,
        zvar_names::Union{Nothing,Vector{Symbol}}=nothing)

    global _dicM
    _dicM = Dict{Symbol, Any}()

    # Initialize all keys to nothing
    for k in (:panel, :idvar, :dist, :type, :depvar, :frontier,
              :μ, :hscale, :gamma, :σᵤ², :λ, :σᵥ², :σₐ², :hasDF, :transfer, :misc, :sdf)
        _dicM[k] = nothing
    end

    # Build a combined DataFrame with auto-generated column names
    comDF = depvar_vec
    all_varnames = [:depvar]
    _dicM[:depvar] = [:depvar]

    # frontier columns (auto-generated unique names for internal DataFrame)
    fr_names = [Symbol("frontier_var$i") for i in 1:K]
    comDF = hcat(comDF, frontier_mat)
    append!(all_varnames, fr_names)
    _dicM[:frontier] = fr_names

    # constant column (for homoscedastic equations)
    const_col = ones(N, 1)
    comDF = hcat(comDF, const_col)
    push!(all_varnames, :_cons)

    # Map hetero/ineff to the old equation keys
    if is_scaling
        # Cross-sectional scaling model: hscale = original zvar, μ/σᵤ²/σᵥ² = constant
        hs_names = [Symbol("hscale_var$i") for i in 1:size(scaling_zvar_mat, 2)]
        comDF = hcat(comDF, scaling_zvar_mat)
        append!(all_varnames, hs_names)
        _dicM[:hscale] = hs_names
        _dicM[:μ]   = [:_cons]
        _dicM[:σᵤ²] = [:_cons]
        _dicM[:σᵥ²] = [:_cons]
    elseif panel in (:TFE_WH2010,)
        # Panel WH2010: zvar maps to hscale (scaling function)
        hs_names = [Symbol("hscale_var$i") for i in 1:size(zvar_mat, 2)]
        comDF = hcat(comDF, zvar_mat)
        append!(all_varnames, hs_names)
        _dicM[:hscale] = hs_names
        # μ only for truncated-normal (PFEWHT)
        if ineff == :TruncatedNormal
            _dicM[:μ] = [:_cons]
        end
        _dicM[:σᵤ²] = [:_cons]
        _dicM[:σᵥ²] = [:_cons]
    else
        hetero_vec = hetero isa Vector ? hetero : Symbol[]

        # μ equation (TruncatedNormal only)
        if ineff == :TruncatedNormal
            if :mu in hetero_vec
                z_names_mu = [Symbol("μ_var$i") for i in 1:size(zvar_mat, 2)]
                comDF = hcat(comDF, zvar_mat)
                append!(all_varnames, z_names_mu)
                _dicM[:μ] = z_names_mu
            else
                _dicM[:μ] = [:_cons]
            end
        end

        # σᵤ² / λ equation
        if ineff == :Exponential
            if :lambda in hetero_vec
                z_names_lam = [Symbol("λ_var$i") for i in 1:size(zvar_mat, 2)]
                comDF = hcat(comDF, zvar_mat)
                append!(all_varnames, z_names_lam)
                _dicM[:λ] = z_names_lam
            else
                _dicM[:λ] = [:_cons]
            end
        elseif :sigma_sq in hetero_vec
            z_names_su = [Symbol("σᵤ²_var$i") for i in 1:size(zvar_mat, 2)]
            comDF = hcat(comDF, zvar_mat)
            append!(all_varnames, z_names_su)
            _dicM[:σᵤ²] = z_names_su
        else
            _dicM[:σᵤ²] = [:_cons]
        end

        # σᵥ² equation (always homoscedastic)
        _dicM[:σᵥ²] = [:_cons]
    end

    # Store display names for table output (only when real names provided from DSL)
    if frontier_names !== nothing || zvar_names !== nothing
        _dicM[:_display_frontier] = frontier_names
        _dicM[:_display_zvar] = zvar_names
    end

    # Panel-specific keys
    if panel !== nothing
        _dicM[:panel] = [panel]
    end

    if idvar_vec !== nothing
        comDF = hcat(comDF, idvar_vec)
        push!(all_varnames, :idvar_col)
        _dicM[:idvar] = [:idvar_col]
    end

    # Panel-specific equations (σₐ², gamma) — set from constant
    if panel in (:TRE,)
        _dicM[:σₐ²] = [:_cons]
    end
    #= TimeDecay/Kumbhakar1990 disabled in keyword API
    if panel in (:TimeDecay, :Kumbhakar1990)
        _dicM[:gamma] = [:_cons]
        if _dicM[:μ] === nothing
            _dicM[:μ] = [:_cons]
        end
    end
    =#

    # Type and dist
    _dicM[:type] = type_sym == :cost ? [:cost] : [:production]
    _dicM[:dist] = [dist_code]

    # Create the DataFrame
    _dicM[:sdf] = DataFrame(comDF, all_varnames)
    _dicM[:hasDF] = false
    _dicM[:transfer] = false

    # Set tagD
    global tagD
    tagD = Dict{Symbol, Type{<:Sfmodeltype}}()
    tagD[:modelid] = modelid

    return nothing
end



########################################################
####     sfmodel_spec() — DSL (DataFrame) style    ####
########################################################

"""
    sfmodel_spec(args::DSLArg_MLE...; noise=:Normal, ineff, hetero=Symbol[],
                 type=:prod, panel=nothing, message=false)

DSL-style model specification using DataFrame column names.
Arguments can appear in **any order**.

**Required:** `@useData(df)`, `sf_depvar(:y)`, `sf_frontier(:x1, :x2, ...)`
**Optional:** `@zvar(z1, z2, ...)`, `@id(idvar)`

# Examples
```julia
# Cross-sectional
spec = sfmodel_spec(@useData(df), sf_depvar(:y), sf_frontier(:cons, :x1, :x2),
    @zvar(cons, z1);
    ineff=:TruncatedNormal, hetero=[:mu])

# Panel
spec = sfmodel_spec(@useData(df), sf_depvar(:y), sf_frontier(:x1, :x2),
    @zvar(z1), @id(firm);
    ineff=:TruncatedNormal, panel=:TFE_WH2010)
```
"""
function sfmodel_spec(args::DSLArg_MLE...;
    noise::Symbol=:Normal,
    ineff::Symbol,
    hetero::Union{Vector{Symbol}, Symbol}=Symbol[],
    type::Symbol=:prod,
    panel::Union{Nothing, Symbol}=nothing,
    message::Bool=false)

    # Extract each component by type (any order)
    local data = nothing
    local dv   = nothing
    local fr   = nothing
    local zv   = nothing
    local idspec   = nothing

    for arg in args
        if arg isa WUseDataSpec_MLE
            !isnothing(data) && throw("Duplicate @useData specification.")
            data = arg
        elseif arg isa WDepvarSpec_MLE
            !isnothing(dv) && throw("Duplicate @depvar specification.")
            dv = arg
        elseif arg isa WFrontierSpec_MLE
            !isnothing(fr) && throw("Duplicate @frontier specification.")
            fr = arg
        elseif arg isa WZvarSpec_MLE
            !isnothing(zv) && throw("Duplicate @zvar specification.")
            zv = arg
        elseif arg isa WIdSpec_MLE
            !isnothing(idspec) && throw("Duplicate @id / @idvar specification.")
            idspec = arg
        end
    end

    # Validate required macros
    isnothing(data) && throw("@useData is required in DSL-style sfmodel_spec().")
    isnothing(dv)   && throw("@depvar is required in DSL-style sfmodel_spec().")
    isnothing(fr)   && throw("@frontier is required in DSL-style sfmodel_spec().")

    df = data.df

    # Extract data from DataFrame
    depvar_data   = Vector{Float64}(df[!, dv.name])
    frontier_data = Matrix{Float64}(df[!, fr.names])

    zvar_data = if !isnothing(zv)
        Matrix{Float64}(df[!, zv.names])
    else
        nothing
    end

    id_data = if !isnothing(idspec)
        Vector(df[!, idspec.name])
    else
        nothing
    end

    # Delegate to the keyword method, passing actual column names
    return sfmodel_spec(;
        depvar=depvar_data, frontier=frontier_data, zvar=zvar_data,
        noise=noise, ineff=ineff, hetero=hetero, type=type,
        panel=panel, id=id_data,
        frontier_names=fr.names,
        zvar_names=isnothing(zv) ? nothing : zv.names,
        message=message)
end


########################################################
####  sfmodel_spec() — old Vararg API (deprecated)  ####
########################################################

# Backward-compatible method for the old calling convention using
# function/macro helpers that return tuples:
#   sfmodel_spec(sftype(prod), sfdist(half), depvar(y), frontier(x), σᵤ²(w), σᵥ²(v))
# This method is DEPRECATED. Use the keyword API or DSL macros instead.

function sfmodel_spec(arg::Vararg; message::Bool=false)

    global _dicM
           _dicM = Dict{Symbol, Any}()

        for k in (:panel, :idvar, :dist, :type, :depvar, :frontier, :μ, :hscale, :gamma, :σᵤ², :σᵥ², :σₐ², :hasDF, :transfer, :misc)
            _dicM[k] = nothing
        end

        for d in :($(arg))
            _dicM[d[1]] = d[2]
        end

           _dicM[:hasDF]    = true
           _dicM[:transfer] = false

        if typeof(_dicM[:depvar]) != Array{Symbol,1}

           _dicM[:hasDF] = false

           isa(_dicM[:depvar][1], Vector) || isa(_dicM[:depvar][1], Matrix) || throw(
           "`depvar()` has to be a Vector or Matrix.")

           comDF = _dicM[:depvar][1]
           varname = [:depvar]
           _dicM[:depvar] = [:depvar]

           for k in (:idvar, :frontier, :μ, :hscale, :gamma, :σₐ², :σᵤ², :σᵥ²)
               if _dicM[k] !== nothing
                  isa(_dicM[k], Vector) || isa(_dicM[k][1], Vector) || isa(_dicM[k][1], Matrix) || throw(
                     "`$k` has to be a Vector or Matrix.")

                  (isa(_dicM[k], Vector)  && length(_dicM[k][1]) == 1) ?  _dicM[k] = [_dicM[k]] : nothing

                  @views comDF = hcat(comDF, _dicM[k][1])
                  aa = Symbol[]
                  for i in axes(_dicM[k][1], 2)
                      push!(aa, Symbol(String(k)*"_var$(i)"))
                  end
                  varname = vcat(varname, aa)
                  _dicM[k] = aa
               end
           end

           comDF = DataFrame(comDF, varname)
           _dicM[:sdf] = comDF

        end

        (_dicM[:dist] !== nothing) || throw("You need to specify dist().")
        (_dicM[:type] !== nothing) || throw("You need to specify type().")

        s = uppercase(String(_dicM[:dist][1])[1:1])

        global tagD
        if _dicM[:panel] === nothing
            if (s == "T")
                tagD = Dict{Symbol, Type{Trun}}()
                tagD[:modelid] = Trun
            elseif (s == "H")
                tagD = Dict{Symbol, Type{Half}}()
                tagD[:modelid] = Half
            elseif (s == "E")
                tagD = Dict{Symbol, Type{Expo}}()
                tagD[:modelid] = Expo
            elseif (s == "S")
                tagD = Dict{Symbol, Type{Trun_Scale}}()
                tagD[:modelid] = Trun_Scale
            end
        elseif (_dicM[:panel] == [:TFE_WH2010]) && (s=="T")
            tagD = Dict{Symbol, Type{PFEWHT}}()
            tagD[:modelid] = PFEWHT
        elseif (_dicM[:panel] == [:TFE_WH2010]) && (s == "H")
            tagD = Dict{Symbol, Type{PFEWHH}}()
            tagD[:modelid] = PFEWHH
        elseif (_dicM[:panel] == [:TFE_WH2010])
            throw("The panel TFE_WH2010 model can only have `sfdist(trun)` or `sfdist(half)`.")
        elseif (_dicM[:panel] == [:TFE_CSW2014]) && (s=="H")
            tagD = Dict{Symbol, Type{PFECSWH}}()
            tagD[:modelid] = PFECSWH
        elseif (_dicM[:panel] == [:TFE_CSW2014])
            throw("The panel TFE_CSW2014 model can only have `sfdist(half)`.")
        elseif (_dicM[:panel] == [:TRE]) && (s=="H")
            tagD = Dict{Symbol, Type{PTREH}}()
            tagD[:modelid] = PTREH
        elseif (_dicM[:panel] == [:TRE]) && (s=="T")
            tagD = Dict{Symbol, Type{PTRET}}()
            tagD[:modelid] = PTRET
        elseif (_dicM[:panel] == [:TRE])
            throw("The panel TRE model can only have `sfdist(half)` or `sfdist(trun)`.")
        elseif (_dicM[:panel] == [:TimeDecay]) && (s == "T")
            tagD = Dict{Symbol, Type{PanDecay}}()
            tagD[:modelid] = PanDecay
        elseif (_dicM[:panel] == [:TimeDecay])
            throw("The panel time-decay model can only have `sfdist(trun)`.")
        elseif (_dicM[:panel] == [:Kumbhakar1990]) && (s == "T")
            tagD = Dict{Symbol, Type{PanKumb90}}()
            tagD[:modelid] = PanKumb90
        elseif (_dicM[:panel] == [:Kumbhakar1990])
            throw("The panel Kumbhakar 1990 model can only have `sfdist(trun)`.")
        else
            throw("The `sfpanel()` and/or `sfdist()` are not specified correctly.")
        end

        SFmle.checksyn(tagD[:modelid])

        if message
          printstyled("A dictionary from sfmodel_spec() is generated.\n"; color = :green)
        end
        return _dicM
end


########################################################
####                sfmodel_init()                  ####
########################################################

# --- Helper: normalize scalar/vector/tuple/matrix to Vector{Float64} ---
_to_init_vec(x::Vector{<:Real}) = convert(Vector{Float64}, x)
_to_init_vec(x::Real) = [Float64(x)]
_to_init_vec(x::Tuple) = collect(Float64, x)
_to_init_vec(x::AbstractMatrix) = vec(convert(Matrix{Float64}, x))

"""
    sfmodel_init(; spec, init, frontier, mu, ln_sigma_u_sq, ln_sigma_v_sq,
                   hscale, gamma, sigma_a_sq, message)

Provide initial values for the stochastic frontier model estimation.
Creates a global dictionary `_dicINI` consumed by `sfmodel_fit()`. Optional.

# Arguments
- `spec::SFModelSpec_MLE`: Model specification from `sfmodel_spec()`.
  Defaults to the global `_MLE_spec` set by `sfmodel_spec()`.
- `init`: Full initial-value vector for all parameters (bypasses component mode).
- `frontier`: Initial values for frontier coefficients.
- `mu`: Initial values for the μ equation (truncated normal models).
- `ln_sigma_u_sq`: Initial values for log σᵤ² equation.
- `ln_sigma_v_sq`: Initial values for log σᵥ² equation.
- `hscale`: Initial values for the scaling-property hscale equation.
- `gamma`: Initial values for the gamma equation (TimeDecay/Kumbhakar1990).
- `sigma_a_sq`: Initial values for σₐ² (TRE panel models).
- `message::Bool=false`: Print confirmation message.

# Examples
```julia
# Component mode
sfmodel_init(frontier=X\\y, ln_sigma_u_sq=-0.1, ln_sigma_v_sq=-0.1)

# Full vector mode
sfmodel_init(init=[0.1, 0.2, 0.5, 0.0, -0.1, -0.1, -0.1])

# With explicit spec
sfmodel_init(spec=myspec, frontier=[0.1, 0.2, 0.5], mu=zeros(3),
             ln_sigma_u_sq=-0.1, ln_sigma_v_sq=-0.1)
```
"""
function sfmodel_init(;
    spec::SFModelSpec_MLE = _MLE_spec,
    init = nothing,
    frontier = nothing,
    mu = nothing,
    ln_sigma_u_sq = nothing,
    ln_sigma_v_sq = nothing,
    hscale = nothing,
    gamma = nothing,
    sigma_a_sq = nothing,
    message::Bool = false)

    global _dicINI
    _dicINI = Dict{Symbol, Any}()

    # Mode 1: Full vector — bypass component assembly
    if init !== nothing
        _dicINI[:all_init] = _to_init_vec(init)
        if message
            printstyled("Initial values (full vector) set.\n"; color=:green)
        end
        return _dicINI
    end

    # Mode 2: Component-by-component
    if frontier !== nothing
        _dicINI[:frontier] = _to_init_vec(frontier)
    end
    if mu !== nothing
        v = _to_init_vec(mu)
        _dicINI[:μ]   = v
        _dicINI[:eqz] = v
    end
    if ln_sigma_u_sq !== nothing
        v = _to_init_vec(ln_sigma_u_sq)
        _dicINI[:σᵤ²] = v
        _dicINI[:eqw] = v
    end
    if sigma_a_sq !== nothing
        v = _to_init_vec(sigma_a_sq)
        _dicINI[:σₐ²] = v
        _dicINI[:eqw] = v
    end
    if ln_sigma_v_sq !== nothing
        v = _to_init_vec(ln_sigma_v_sq)
        _dicINI[:σᵥ²] = v
        _dicINI[:eqv] = v
    end
    if hscale !== nothing
        v = _to_init_vec(hscale)
        _dicINI[:hscale] = v
        _dicINI[:eqq]   = v
    end
    if gamma !== nothing
        v = _to_init_vec(gamma)
        _dicINI[:gamma] = v
        _dicINI[:eqq]  = v
    end

    if message
        printstyled("A dictionary from sfmodel_init() is generated.\n"; color=:green)
    end
    return _dicINI
end

########################################################
####                sfmodel_opt()                   ####
########################################################
"""
    sfmodel_opt(; warmstart_solver, warmstart_opt, main_solver, main_opt)

Specify optimization options (solvers and convergence criteria).

# Arguments
- `warmstart_solver=nothing`: Optional warmstart optimizer (e.g., `NelderMead()`).
  Set to `nothing` to skip warmstart (default).
- `warmstart_opt=nothing`: NamedTuple of warmstart options
  (e.g., `(iterations=200,)`). Requires trailing comma for single entries.
- `main_solver=Newton()`: Main optimizer.
- `main_opt=(iterations=2000, g_abstol=1e-8)`: NamedTuple of main optimization
  options passed to `Optim.Options`.

# Examples
```julia
sfmodel_opt(warmstart_solver=NelderMead(),
            warmstart_opt=(iterations=200,),
            main_solver=Newton(),
            main_opt=(iterations=2000, g_abstol=1e-8))

# No warmstart (default)
sfmodel_opt(main_solver=BFGS(), main_opt=(iterations=500, g_abstol=1e-6))
```
"""
function sfmodel_opt(;
    warmstart_solver = nothing,
    warmstart_opt = nothing,
    main_solver = Newton(),
    main_opt = (iterations=2000, g_abstol=1e-8))

    if main_opt !== nothing && !(main_opt isa NamedTuple)
        error("main_opt must be a NamedTuple, e.g. (iterations=2000, g_abstol=1e-8). " *
              "Hint: single-element tuples need a trailing comma: (iterations=200,)")
    end
    if warmstart_opt !== nothing && !(warmstart_opt isa NamedTuple)
        error("warmstart_opt must be a NamedTuple, e.g. (iterations=200,). " *
              "Hint: single-element tuples need a trailing comma: (iterations=200,)")
    end

    global _dicOPT
    _dicOPT = Dict{Symbol, Any}()

    _dicOPT[:warmstart_solver] = warmstart_solver
    _dicOPT[:warmstart_maxIT]  = warmstart_opt === nothing ? nothing : get(warmstart_opt, :iterations, 100)
    _dicOPT[:main_solver]      = main_solver
    _dicOPT[:main_maxIT]       = get(main_opt, :iterations, 2000)
    _dicOPT[:tolerance]        = get(main_opt, :g_abstol, 1e-8)
    # Display options — now controlled by sfmodel_fit() keywords; defaults here for compat
    _dicOPT[:verbose]      = true
    _dicOPT[:banner]       = true
    _dicOPT[:ineff_index]  = true
    _dicOPT[:marginal]     = true
    _dicOPT[:table_format] = :text

    return _dicOPT
end    # end of sfmodel_opt()





########################################################
###                 sfmodel_fit()                   ####
########################################################
"""
    sfmodel_fit(<keyword arguments>)

Maximum likelihood estimation of the stochastic frontier model specified
in `sfmodel_spec(...)`. Estimate the model parameters, calculate Jondrow et al.
(1982) inefficiency index and Battese and Coelli (1988) efficiency index,
compute marginal effects of inefficiency determinants (if any).
Return a NamedTuple with results.

# Arguments
- `spec::SFModelSpec_MLE`: The model specification from `sfmodel_spec()`.
  Defaults to the global `_MLE_spec`.
- `init`: Initial values vector from `sfmodel_init()`, or `nothing` for
  auto-generated OLS-based defaults.
- `optim_options`: Optimization options dict from `sfmodel_opt()`, or `nothing`
  for defaults.
- `jlms_bc_index::Bool=true`: Whether to compute JLMS and BC efficiency indices.
- `marginal::Bool=true`: Whether to compute marginal effects.
- `show_table::Bool=true`: Whether to print the results table and banner.
- `verbose::Bool=true`: Whether to print optimization progress messages.

# Examples
```julia-repl
spec = sfmodel_spec(sftype=:prod, sfdist=(:h, :n),
                    depvar=y, frontier=X)
result = sfmodel_fit(spec=spec)

# With all options
init = sfmodel_init(frontier=ones(5)*0.1, ln_sigma_u_sq=-0.1, ln_sigma_v_sq=-0.1)
opt  = sfmodel_opt(main_solver=Newton(), main_opt=(iterations=2000, g_abstol=1e-8))
result = sfmodel_fit(spec=spec, init=init, optim_options=opt,
                     jlms_bc_index=true, marginal=true, show_table=true, verbose=true)
```
"""
function sfmodel_fit(;
    spec::SFModelSpec_MLE = _MLE_spec,
    init = nothing,
    optim_options = nothing,
    jlms_bc_index::Bool = true,
    marginal::Bool = true,
    show_table::Bool = true,
    verbose::Bool = true)

   #* --- Set up _dicINI from init keyword ---

   global _dicINI
   if init !== nothing
       if init isa Dict
           _dicINI = init  # already a dict from sfmodel_init()
       else
           _dicINI = Dict{Symbol, Any}(:all_init => init)  # raw vector
       end
   else
       # Always reset to fresh defaults — avoids stale _dicINI from a previous model
       sfmodel_init()
   end

   #* --- Set up _dicOPT from optim_options keyword ---

   global _dicOPT
   if optim_options !== nothing
       _dicOPT = optim_options
   else
       @isdefined(_dicOPT) || sfmodel_opt()
   end

   # Populate display-related keys in _dicOPT from keyword args
   _dicOPT[:verbose]      = verbose
   _dicOPT[:banner]       = show_table
   _dicOPT[:ineff_index]  = jlms_bc_index
   _dicOPT[:marginal]     = marginal
   _dicOPT[:table_format] = :text

   #* for simulation, add a flag
   redflag::Bool = 0

   sfdat = _dicM[:sdf]

   # Reset display names in _dicM from the spec to guard against stale global
   # state when multiple specs are created before fitting (bug 8d).
   _dicM[:_display_frontier] = spec.display_frontier
   _dicM[:_display_zvar]     = spec.display_zvar

   if show_table
      printstyled("\n###------------------------------------###\n"; color=:yellow)
      printstyled("###  Estimating SF models using Julia  ###\n"; color=:yellow)
      printstyled("###------------------------------------###\n\n"; color=:yellow)
   end

  #* ##### Get variables from dataset #######

     # pos: (begx, endx, begz, endz, ...); variables' positions in the parameter vector.
     # num: (nofobs, nofx, ..., nofpara); number of variables in each equation
     # eqvec: ("frontier"=2, "mu"=6,...); named tuple of equation names and equation position in the table
     # eqvec2: (xeq=(1,3), zeq=(4,5),...); named tuple of equation and parameter positions, for sfmodel_predict
     # varlist: ("x1", "x2",...); variable names for making table

     (minfo1, minfo2, pos, num, eqvec, eqvec2, yvar, xvar, zvar, qvar, wvar,
      vvar,         rowIDT, varlist) = SFmle.getvar(tagD[:modelid], sfdat)

     # Override varlist with real column names from DSL macros (if provided)
     if get(_dicM, :_display_frontier, nothing) !== nothing
         dnames_fr = String.(get(_dicM, :_display_frontier, Symbol[]))
         _raw_zv = get(_dicM, :_display_zvar, nothing)
         dnames_zv = _raw_zv !== nothing ? String.(_raw_zv) : String[]
         vi = 2  # varlist[1] is " "
         # frontier names
         for j in 1:num.nofx
             varlist[vi] = dnames_fr[j]
             vi += 1
         end
         # z equation names (μ hetero for TruncatedNormal)
         for j in 1:num.nofz
             varlist[vi] = num.nofz > 1 ? dnames_zv[j] : varlist[vi]
             vi += 1
         end
         # q equation names (scaling hscale)
         for j in 1:num.nofq
             varlist[vi] = (!isempty(dnames_zv) && num.nofq > 1) ? dnames_zv[j] : varlist[vi]
             vi += 1
         end
         # w equation names (σᵤ²/λ hetero)
         for j in 1:num.nofw
             varlist[vi] = num.nofw > 1 ? dnames_zv[j] : varlist[vi]
             vi += 1
         end
         # v equation names (σᵥ²) — always scalar, keep as-is
     end

     # Fallback: use spec.varnames from keyword API (when DSL names not available)
     if get(_dicM, :_display_frontier, nothing) === nothing &&
        !isempty(spec.varnames) && length(spec.varnames) >= length(varlist) - 1
         for j in 1:(length(varlist)-1)
             varlist[j+1] = spec.varnames[j]
         end
     end

  #* ### print preliminary information ########

    if verbose

      printstyled("*********************************\n "; color=:cyan)
      printstyled("      Model Specification:\n"; color=:cyan);
      printstyled("*********************************\n"; color=:cyan)

      print("Model type: "); printstyled(minfo1; color=:yellow); println();println()
      printstyled(minfo2; color=:yellow); println()
    end

  #* ##### Get the type parameter #######

     _porc::Int64 = 1

     if (_dicM[:type] == [:cost])
         _porc = -1
     end

  #* ########## Process initial value dictionary  #####
     #* --- Get OLS results and other auxiliary values. --- #

     noffixed = 0  # number of fixed effect parameter
     if _dicM[:panel] == [:TFE_WH2010] || _dicM[:panel] == [:TFE_CSW2014]  # the fixed effects and differenced off
        noffixed = size(rowIDT,1)
     end

     beta0  = xvar \ yvar;  # OLS estimate, uses a pivoted QR factorization;
     resid  = yvar - xvar*beta0
     sse    = sum((resid).^2)
     ssd    = sqrt(sse/size(resid,1)) # MLE standard deviation; σ² = (1/N)* Σ ϵ^2
     ll_ols = sum(normlogpdf.(0, ssd, resid)) # ols log-likelihood
     sk_ols = sum((resid).^3) / ((ssd^3)*(size(resid,1))) # skewness of ols residuals

     #* --- Create the dictionary -----------

     if (:all_init in keys(_dicINI))
         sf_init = _dicINI[:all_init]
     else
         #*  Create ini vectors from user's values; if none, use the default.--- #
         b_ini  = get(_dicINI, :frontier, beta0)
         d1_ini = get(_dicINI, :eqz, ones(num.nofz) * 0.1)
         t_ini  = get(_dicINI, :eqq, ones(num.nofq) * 0.1)
         d2_ini = get(_dicINI, :eqw, ones(num.nofw) * 0.1)
         g_ini  = get(_dicINI, :eqv, ones(num.nofv) * 0.1)

         #*  Make it Array{Float64,1}; otherwise Array{Float64,2}. ---#
         sf_init = vcat(b_ini, d1_ini, t_ini, d2_ini, g_ini)
         sf_init = vec(sf_init)
     end # if :all_init


  #* ############ Misc.  ################
     # --- check if the number of initial values is correct
        (length(sf_init) == num.nofpara) ||  throw("The number of initial values does not match the number of parameters to be estimated. Make sure the number of init values in sfmodel_init() matches the number of variabls in sfmodel_spec().")

     # --- Make sure there is no numerical issue arising from int vs. Float64.
        sf_init = convert(Array{Float64,1}, sf_init)

  #* ############# process optimization options  #######

         if (_dicOPT[:warmstart_solver] === nothing) || (_dicOPT[:warmstart_maxIT] === nothing)
             do_warmstart_search = 0
         else
             do_warmstart_search = 1
             sf_ini_algo  = _dicOPT[:warmstart_solver]  # warmstart search algorithms
             sf_ini_maxit = _dicOPT[:warmstart_maxIT]    # warmstart search iter limit
         end

     # ---- main maximization algorithm -----
         sf_algo  = _dicOPT[:main_solver]    # main algorithm
         sf_maxit = _dicOPT[:main_maxIT]
         sf_tol   = _dicOPT[:tolerance]
         sf_table = _dicOPT[:table_format]

  #* ########  Start the Estimation  ##########

    #* ----- Define the objective function -----#

     _lik = rho -> SFmle.LL_T(tagD[:modelid],
                           yvar, xvar, zvar, qvar, wvar, vvar,
                           _porc, num.nofobs, pos, rho,
                                   rowIDT, _dicM[:misc])


    #* ---- Make placeholders for dictionary recording purposes *#

    sf_init_1st_dic  = 0
    sf_init_2nd_dic  = 0
    sf_ini_algo_dic  = nothing
    sf_ini_maxit_dic = 0
    sf_total_iter    = 0

    _run = 1  # a counter; use the -if- instead of -for- to avoid using global variables

    if (do_warmstart_search == 1) && (_run == 1)

        if verbose
            printstyled("The warmstart run...\n\n"; color = :green)
        end

        sf_init_1st_dic  = copy(sf_init) # for dict recording
        sf_ini_algo_dic  = sf_ini_algo
        sf_ini_maxit_dic = copy(sf_ini_maxit)

               _optres = optimize(_lik,
                               sf_init,         # initial values
                               sf_ini_algo,
                               Optim.Options(g_tol = sf_tol,
                                             iterations  = sf_ini_maxit,
                                             store_trace = true,
                                             show_trace  = false))


        sf_total_iter += Optim.iterations(_optres) # for later use

        sf_init = Optim.minimizer(_optres)  # save as initials for the next run
        _run    = 2                      # modify the flag

        if verbose
            println()
            print("$_optres \n")
            print("The warmstart results are:\n"); printstyled(Optim.minimizer(_optres); color=:yellow); println("\n")
        end

   end  # if  (do_warmstart_search == 1) && (_run == 1)

   if (do_warmstart_search == 0 ) || (_run == 2) # either no warmstart run and go straight here, or the 2nd run

       sf_init_2nd_dic = copy(sf_init) # for dict recording

       if verbose
           println()
           printstyled("Starting the optimization run...\n\n" ; color = :green)
       end

              _optres = optimize(_lik,
                              sf_init,       # initial values
                              sf_algo,       # different from search run
                              Optim.Options(g_tol = sf_tol,
                                            iterations  = sf_maxit, # different from search run
                                            store_trace = true,
                                            show_trace  = false);
                              autodiff = AutoForwardDiff())
       sf_total_iter += Optim.iterations(_optres)

       if verbose
             println()
             print("$_optres \n")
             print("The resulting coefficient vector is:\n"); printstyled(Optim.minimizer(_optres); color=:yellow); println("\n")
       end


       if isnan(Optim.g_residual(_optres)) || (Optim.g_residual(_optres) > 0.1)
            redflag = 1
            printstyled("Note that the estimation may not have converged properly. The gradients are problematic (too large, > 0.1, or others).\n\n", color = :red)
       end


       if Optim.iteration_limit_reached(_optres)
             redflag = 1
             printstyled("Caution: The number of iterations reached the limit.\n\n"; color= :red)
       end

   end     # if (do_warmstart_search == 0 )....


  #* ###### Post-estimation process ###############

      _coevec            = Optim.minimizer(_optres)  # coef. vec.
      numerical_hessian  = ForwardDiff.hessian(_lik, _coevec)  # Hessian

     #* ------ Check if the matrix is invertible. ----

     var_cov_matrix = try
                         inv(numerical_hessian)
                      catch err
                         redflag = 1
                         checkCollinear(tagD[:modelid], xvar, zvar, qvar, wvar, vvar)
                         throw("The Hessian matrix is not invertible, indicating the model does not converge properly. The estimation is abort.")
                      end

          #* In some cases the matrix is invertible but the resulting diagonal
          #*    elements are negative. Check.

          if !all( diag(var_cov_matrix) .> 0 ) # not all are positive
               redflag = 1
               printstyled("Some of the diagonal elements of the var-cov matrix are non-positive, indicating problems in the convergence. The estimation is abort.\n\n"; color = :red)
               checkCollinear(tagD[:modelid], xvar, zvar, qvar, wvar, vvar)
          end

     #* ------- JLMS and BC index -------------------

     if jlms_bc_index
        @views (_jlms, _bc) = jlmsbc(tagD[:modelid], _porc, pos, _coevec,
                                     yvar, xvar, zvar, qvar, wvar, vvar,         rowIDT)
        _jlmsM = mean(_jlms)
        _bcM   = mean(_bc)
     else
        _jlms  = nothing
        _bc    = nothing
        _jlmsM = nothing
        _bcM   = nothing
     end


     #* ---- marginal effect on E(u) --------------

     # Override _dicM equation entries with common column names before
     # calling get_marg, so that addDataFrame correctly merges same-variable
     # effects across equations and margMinfo/margeff use real names.
     # Save originals so _dicM is not permanently mutated (allows re-running
     # sfmodel_fit without re-running sfmodel_spec).
     _saved_dicM_keys = Dict{Symbol, Any}()
     for key in (:μ, :σᵤ², :λ, :hscale)
         v = get(_dicM, key, nothing)
         if v !== nothing
             _saved_dicM_keys[key] = copy(v)
         end
     end

     if get(_dicM, :_display_zvar, nothing) !== nothing
         # DSL path: use real column names
         dnames_zv = get(_dicM, :_display_zvar, Symbol[])
         for key in (:μ, :σᵤ², :λ, :hscale)
             v = get(_dicM, key, nothing)
             if v isa Vector{Symbol} && length(v) == length(dnames_zv)
                 _dicM[key] = dnames_zv
             end
         end
     else
         # Keyword API: generate common names so addDataFrame merges correctly
         ref_len = 0
         for key in (:μ, :σᵤ², :λ, :hscale)
             v = get(_dicM, key, nothing)
             if v isa Vector{Symbol}
                 ref_len = max(ref_len, length(v))
             end
         end
         if ref_len > 1
             common_names = [Symbol("z_var$i") for i in 1:ref_len]
             for key in (:μ, :σᵤ², :λ, :hscale)
                 v = get(_dicM, key, nothing)
                 if v isa Vector{Symbol} && length(v) == ref_len
                     _dicM[key] = common_names
                 end
             end
         end
     end

     if marginal
        margeff, margMinfo = get_marg(tagD[:modelid], pos, num, _coevec, zvar, qvar, wvar)
     else
        margeff, margMinfo = nothing, ()
     end

     # Restore _dicM keys so sfmodel_fit can be re-run without re-running sfmodel_spec.
     for (key, val) in _saved_dicM_keys
         _dicM[key] = val
     end

     #* ------- Make Table ------------------

     stddev  = sqrt.(diag(var_cov_matrix)) # standard error
     t_stats = _coevec ./ stddev          # t statistics
     p_value = zeros(num.nofpara)   # p values
     ci_low  = zeros(num.nofpara) # confidence interval
     ci_upp  = zeros(num.nofpara)
     tt      = cquantile(Normal(0,1), 0.025)

     for i = 1:num.nofpara
         @views p_value[i,1] = 2 * ccdf(TDist(num.nofobs - num.nofpara), abs(t_stats[i,1]))
         @views ci_low[i,1] = _coevec[i,1] - tt*stddev[i,1]
         @views ci_upp[i,1] = _coevec[i,1] + tt*stddev[i,1]
     end

       #* Build the table columns *#

    table = zeros(num.nofpara, 7)  # 7 columns in the table
    table[:,2] = _coevec   # estimated coefficients
    table[:,3] = stddev    # std deviation
    table[:,4] = t_stats   # t statistic
    table[:,5] = p_value   # p value
    table[:,6] = ci_low
    table[:,7] = ci_upp
    table      = [" " "Coef." "Std. Err." "z" "P>|z|" "95%CI_l" "95%CI_u"; table]  # add to top of the table

       #*  creating a column of function names

    table[:, 1] .= ""
    for i in 1:length(eqvec)
        @views j = eqvec[i]
        @views table[j,1] = keys(eqvec)[i]
    end

       #*  Add the column of variable names

    table = hcat(varlist, table)                      # combine the variable names column
    table[:,1], table[:,2] = table[:,2], table[:,1]   # swap the first name column and the function column
    table[1,2] = "Var."

     # * ------ Print Results ----------- *#

     if show_table

         printstyled("*********************************\n"; color=:cyan)
         printstyled("      Estimation Results\n"; color=:cyan)
         printstyled("*********************************\n"; color=:cyan)

         print("Method: "); printstyled("MLE"; color=:yellow); println()
         print("Model type: "); printstyled("noise=$(spec.noise), ineff=$(spec.ineff)"; color=:yellow); println()
         if spec.hetero isa Symbol && spec.hetero == :scaling
             print("Heteroscedastic parameters: "); printstyled(":scaling"; color=:yellow); println()
         elseif spec.hetero isa Vector{Symbol} && !isempty(spec.hetero)
             print("Heteroscedastic parameters: "); printstyled(spec.hetero; color=:yellow); println()
         else
             println("Homoscedastic model (no heteroscedasticity)")
         end
         print("Number of observations: "); printstyled(num.nofobs; color=:yellow); println()
         print("Number of frontier regressors (K): "); printstyled(spec.K; color=:yellow); println()
         print("Number of Z columns (L): "); printstyled(spec.L; color=:yellow); println()
         print("Frontier type: "); printstyled(spec.type_sign == 1 ? "production" : "cost"; color=:yellow); println()
         print("Number of iterations: "); printstyled(sf_total_iter; color=:yellow); println()
         _converged = Optim.converged(_optres)
         print("Converged: "); printstyled(_converged; color=_converged ? :yellow : :red); println()
         if !_converged
             redflag = 1
         end
         print("Log-likelihood: "); printstyled(round(-1*Optim.minimum(_optres); digits=5); color=:yellow); println()
         println()

         pretty_table(table[2:end,:];
                      column_labels=["", "Var.", "Coef.", "Std.Err.", "z", "P>|z|",
                              "95%CI_l", "95%CI_u"],
                      formatters = [fmt__printf("%.4f", collect(3:8))],
                      compact_printing = true,
                      backend = sf_table)
         println()


         # *----- Auxiliary Table, log parameters to original scales --------

         auxtable = Array{Any}(undef,2,3)
         rn = 0 # row index

         if size(wvar,2) == 1 # single variable
            varstd = sqrt(sum((wvar .- sum(wvar)/length(wvar)).^2)/length(wvar))
            if varstd  <= 1e-7 # constant, assuming =1 anyway
                rn += 1
                auxtable[rn, 1] = :σᵤ²
                auxtable[rn, 2] = exp(_coevec[pos.begw])
                auxtable[rn, 3] = exp(_coevec[pos.begw])*stddev[pos.begw]
            end
         end

         if size(vvar,2) == 1 # single variable
            varstd = sqrt(sum((vvar .- sum(vvar)/length(vvar)).^2)/length(vvar))
            if varstd  <= 1e-7 # constant
                rn += 1
                auxtable[rn, 1] = :σᵥ²
                auxtable[rn, 2] = exp(_coevec[pos.begv])
                auxtable[rn, 3] = exp(_coevec[pos.begv])*stddev[pos.begv]
            end
         end

         if rn >= 1  # table is non-empty
             println("Log-parameters converted to original scale (σ² = exp(log_σ²)):")
             pretty_table(auxtable[1:rn,:];
                          column_labels=["", "Coef.", "Std.Err."],
                          formatters = [fmt__printf("%.4f", [2, 3])],
                          compact_printing = true,
                          backend = sf_table)
             println()
         end

         print("Table format: "); printstyled(sf_table; color=:yellow); println()

         printstyled("***** Additional Information *********\n"; color=:cyan)

         print("* OLS (frontier-only) log-likelihood: "); printstyled(round(ll_ols; digits=5); color=:yellow); println("")
         print("* Skewness of OLS residuals: "); printstyled(round(sk_ols; digits=5); color=:yellow); println("")
         if jlms_bc_index
            print("* The sample mean of the JLMS inefficiency index: "); printstyled(round(_jlmsM; digits=5); color=:yellow); println("")
            print("* The sample mean of the BC efficiency index: "); printstyled(round(_bcM; digits=5); color=:yellow); println("\n")
         end
         if length(margMinfo) >= 1
            print("* The sample mean of inefficiency determinants' marginal effects on E(u): " ); printstyled(margMinfo; color=:yellow); println("")
            println("* Marginal effects of the inefficiency determinants at the observational level are saved in the return. See the follows.\n")
         end

         println("* Use `name.list` to see saved results (keys and values) where `name` is the return specified in `name = sfmodel_fit(..)`. Values may be retrieved using the keys. For instance:")
         println("   ** `name.loglikelihood`: the log-likelihood value of the model;")
         println("   ** `name.jlms`: Jondrow et al. (1982) inefficiency index;")
         println("   ** `name.bc`: Battese and Coelli (1988) efficiency index;")
         println("   ** `name.marginal`: a DataFrame with variables' (if any) marginal effects on E(u).")
         println("* Use `keys(name)` to see available keys.")

         printstyled("**************************************\n\n\n"; color=:cyan)

     end  # if show_table

  #* ########### create a dictionary and make a tuple for return ########### *#

      _dicRES = OrderedDict{Symbol, Any}()
      _dicRES[:converged]          = Optim.converged(_optres)
      _dicRES[:iter_limit_reached] = Optim.iteration_limit_reached(_optres)
      _dicRES[:_______________] = "___________________"  #33
      _dicRES[:n_observations]  = num.nofobs
      _dicRES[:loglikelihood]   = -Optim.minimum(_optres)
      _dicRES[:table]           = [table][1]
      _dicRES[:coeff]           = _coevec
      _dicRES[:std_err]         = stddev
      _dicRES[:var_cov_mat]     = [var_cov_matrix][1]
      _dicRES[:jlms]            = _jlms
      _dicRES[:bc]              = _bc
      _dicRES[:OLS_loglikelihood] = ll_ols
      _dicRES[:OLS_resid_skew]    = sk_ols
      _dicRES[:marginal]      = margeff
      _dicRES[:marginal_mean] = margMinfo
      _dicRES[:_____________] = "___________________"  #31
      _dicRES[:model]         = minfo1
      _dicRES[:depvar]        = _dicM[:depvar]
      _dicRES[:frontier]      = _dicM[:frontier]
      _dicRES[:μ]             = _dicM[:μ]
      _dicRES[:hscale]        = _dicM[:hscale]
      _dicRES[:gamma]         = _dicM[:gamma]
      _dicRES[:σₐ²]           = _dicM[:σₐ²]
      _dicRES[:σᵤ²]           = _dicM[:σᵤ²]
      _dicRES[:σᵥ²]           = _dicM[:σᵥ²]
      _dicRES[:log_σₐ²]       = _dicM[:σₐ²]
      _dicRES[:log_σᵤ²]       = _dicM[:σᵤ²]
      _dicRES[:log_σᵥ²]       = _dicM[:σᵥ²]
      _dicRES[:type]          = _dicM[:type]
      _dicRES[:dist]          = _dicM[:dist]
      _dicRES[:PorC]          = _porc
      _dicRES[:idvar]         = _dicM[:idvar] # for bootstrap marginal effect
      _dicRES[:table_format]  = sf_table
      _dicRES[:modelid]       = tagD[:modelid]
      _dicRES[:verbose]       = verbose
      _dicRES[:hasDF]         = _dicM[:hasDF]
      _dicRES[:transfer]      = _dicM[:transfer]

    for i in 1:length(eqvec2)
        _dicRES[keys(eqvec2)[i]] = _coevec[eqvec2[i]]
    end

      _dicRES[:________________]  = "___________________" #34
      _dicRES[:Hessian]           = [numerical_hessian][1]
      _dicRES[:gradient_norm]     = Optim.g_residual(_optres)
      _dicRES[:actual_iterations] = Optim.iterations(_optres)
      _dicRES[:______________] = "______________________" #32
      _dicRES[:warmstart_solver] = sf_ini_algo_dic
      _dicRES[:warmstart_ini]    = sf_init_1st_dic
      _dicRES[:warmstart_maxIT]  = sf_ini_maxit_dic
      _dicRES[:main_solver]      = sf_algo
      _dicRES[:main_ini]         = sf_init_2nd_dic
      _dicRES[:main_maxIT]       = sf_maxit
      _dicRES[:tolerance]        = sf_tol
      _dicRES[:eqpo]             = eqvec2

      _dicRES[:redflag]          = redflag
      _dicRES[:varnames]         = varlist[2:end]  # real variable names (strip header placeholder)

     #* ----- Delete optional keys that have value nothing,

         for k in (:μ, :hscale,  :gamma)
             if _dicRES[k] === nothing
                delete!(_dicRES, k)
             end
         end

     #* ----- Create a NamedTuple from the dic as the final output;
     #* -----     put the dic in the tuple.

         _ntRES = NamedTuple{Tuple(keys(_dicRES))}(values(_dicRES))
         _ntRES = (; _ntRES..., list    = _dicRES)

     #* ---- Create a global dictionary for sf_predict ----

        global _eqncoe
        _eqncoe = Dict{Symbol, Vector}()

        for i in 1:length(eqvec2)
            _eqncoe[keys(eqvec)[i]]  = _coevec[eqvec2[i]] # for sf_predict
        end

  #* ############  make returns  ############ *#

      return _ntRES

end # sfmodel_fit






#############################################################
###  bootstrapping std.err. of mean marginal effect   #######
#############################################################
"""
    sfmodel_boot_marginal(<keyword arguments>)

Bootstrap standard errors and obtain bias-corrected (BC) confidence intervals
for the mean marginal effects of inefficiency determinants. Note that the 
standard error may be influenced by extreme values of the bootstrapped while 
the percentile-based CI is less sensitive to them. In default, return a
``K x 2``` matrix of standard errors (1st column) and confidence intervals
(tuples, 2nd column), where ``K`` is the number of exogenous inefficiency
determinants. With `getBootData=true`, return two matrices: the first is the
same as in the default return, and the second is the ``R x K`` bootstrapped
data.

See also the help file on `sfmodel_CI()`.

# Arguments
- result=<returned result>: The returned result from `sfmodel_fit()`.
- data=<dataset>: The DataFrame dataset containing the model's data.
  Same as the one used in `sfmodel_fit()`. If the data was supplied by matrix
  (instead of DataFrame; i.e., the Method 2 of `sfmodel_spec()`), this option
  should be skipped.
- R::Integer=<number>: The number of bootstrapped samples. The default is 500.
- level::Real=<number>: The significance level (default=0.05) of the bias-corrected
  confidence intervals. If `level`>0.5, it is automatically transformed to
  `1-level`, such that `level=0.05` and `level=0.95` both return 95%
  confidene intervals at the 5% significance level.
- seed::Integer=<number>: A postive integer used to seed the random
  number generator (rng) for resampling, which ensures reproducibility.
  This rng is not global and is only effective in this function. If not
  specified, the global random number generator is used, and the bootstrap
  result may change (slightly) between different runs.
- iter::Integer=<number>: The maximum number of iterations for each bootstrapped
  sample. If the number is larger than 0, it overwrites the `main_maxIT` 
  specified in `sfmodel_opt()` which is the default.
- getBootData::Bool=false: Whether to return the bootstrapped data which is
  ``R x K`` where K is the number of exogenous determinants of inefficiency.
- every::Integer=10: Print bootstrapping progress for every `every` samples.

# Remarks
- Bootstrap samples are with replacement. For panel data, it samples
  cross-sectional units with replacement.
- In the MLE estimation, estimated
  coefficients from the main result is used as initial values. There is no
  `warmstart`. The `main_solver`, `main_maxIT`, and `tolerance` specified in
  `sfmodel_opt()` are used as default, but the value of `main_maxIT` may be
  replaced by the `iter` option.

# Examples
```julia
julia> std_ci = sfmodel_boot_marginal(result=res, data=df, R=250, seed=123)
bootstrap in progress..10..20..30..40..50..60..70..80..90..100..110..120..130..140..150..160..170..180..190..200..210..220..230..240..250..Done!

┌────────┬──────────────────────┬─────────────────┬──────────────────────┐
│        │ mean of the marginal │ std.err. of the │       bias-corrected │
│        │       effect on E(u) │     mean effect │    95.0%  conf. int. │
├────────┼──────────────────────┼─────────────────┼──────────────────────┤
│    age │             -0.00264 │         0.00225 │   (-0.00734, 0.0016) │
│ school │             -0.01197 │         0.01765 │    (-0.048, 0.01224) │
│     yr │             -0.02650 │         0.01221 │ (-0.05257, -0.00447) │
└────────┴──────────────────────┴─────────────────┴──────────────────────┘

3×2 Matrix{Any}:
 0.00224786  (-0.00734, 0.0016)
 0.0176472   (-0.048, 0.01224)
 0.012213    (-0.05257, -0.00447)

julia> std_ci, bsdata = sfmodel_boot_marginal(result=res, data=df, R=250, seed=123, getBootData=true);

 (output omitted)

julia> bsdata
250×3 adjoint(::Matrix{Real}) with eltype Real:
-0.000493033  -0.00840051   -0.0339088
-0.00510175   -0.0128184    -0.0147759
-0.0024987    -0.00971428   -0.00989262
-0.00157346   -0.0265515    -0.0140001
-0.00352179   -0.00670365   -0.0246122
-0.00375162   -0.0070496    -0.0306034
-0.00153094   -0.0154201    -0.0367731
0.000149329   0.00672036   -0.0461389
⋮
-0.00373306   -0.0108364    -0.00871898
-0.00170254   -0.0393002    -0.0500638
0.000686169   0.00241594   -0.018542
0.000258745   0.000183392  -0.039621
-0.00408104   -0.014574     -0.024126
-0.00417206   -0.0192443    -0.0406959
0.00266017   -0.0396552    -0.0359759
```
"""
function sfmodel_boot_marginal(; result::Any=nothing,  data::Any=nothing, 
                                 R::Integer=500, 
                                 level::Real=0.05,
                                 mymisc=nothing,
                                 iter::Integer=-1,
                                 getBootData::Bool=false,
                                 seed::Integer=-1,
                                 every::Integer=10) 

  if data === nothing
      if result.transfer == true
          data = _dicM[:sdf]
      else 
         throw("Need the `data=<DataFrame name>` option.")
      end
  elseif typeof(data) != DataFrame
      throw("`data=` needs to be a DataFrame.")
  end      

   # getvar still use info from _dicM 

  if result.marginal_mean==NamedTuple() || result.marginal_mean === nothing
      printstyled("The model does not have exogenous determinants of inefficiency, ", #=
               =# "or you didn't compute the marginal effect in the main estimation. Abort the bootstrap.\n"; color = :red)
      throw("No statistics to bootstrap.")               
  end    


  ((level > 0.0) && (level < 1.0)) || throw("The significance level (`level`) should be between 0 and 1.")
  
  if level > 0.5
     level = 1-level  # 0.95 -> 0.05
  end

     # In the following lines, the integer part had been taken care of in Type.
  (seed == -1) || ( seed > 0) || throw("`seed` needs to be a positive integer.")
  (iter == -1) || ( iter > 0) || throw("`iter` needs to be a positive integer.")
  (R > 0) || throw("`R` needs to be a positive integer.")

#* ##### Get parameters #######

          _porc = result.PorC
        sf_init = result.coeff
        sf_algo = eval(result.main_solver) 
       sf_maxit = result.main_maxIT  
         sf_tol = result.tolerance   
       sf_table = result.table_format 

       if iter > 0
           sf_maxit = iter
       end 

#* ########  parepare data  ##########

  (minfo1, minfo2, pos, num, eqvec, eqvec2, yvar, xvar, zvar, qvar, wvar, 
   vvar,         rowIDT, varlist) = SFmle.getvar(result.modelid, data)

  #----- make an orginal copy; also make later code easier --------

   yvar0 = deepcopy(yvar); xvar0 = deepcopy(xvar); zvar0 = deepcopy(zvar); 
   qvar0 = deepcopy(qvar); wvar0 = deepcopy(wvar); vvar0 = deepcopy(vvar);
   rowIDT0 = deepcopy(rowIDT)


#* ####### begin looping #############

 sim_res = Array{Real}(undef, ncol(result.marginal), R)

 if seed > 0
    rng = Xoshiro(seed)
 end


 for i in 1:R

    @label start1

   # ----- sampling data for cross-sectional data --------------

   if result.idvar === nothing  # i.e., cross-sectional

     if seed == -1
        select_row = sample(     1:nrow(data), nrow(data); replace=true)  # require StatsBase.jl
     else
        select_row = sample(rng, 1:nrow(data), nrow(data); replace=true)  # require StatsBase.jl
     end

     yvar =  yvar0[select_row, :]
     xvar =  xvar0[select_row, :]
     (zvar0 == ()) ||  (zvar =  zvar0[select_row, :])
     (qvar0 == ()) ||  (qvar =  qvar0[select_row, :])
     (wvar0 == ()) ||  (wvar =  wvar0[select_row, :])
     (vvar0 == ()) ||  (vvar =  vvar0[select_row, :])

  end

   # ------ sampling panel data ------

   if result.idvar !== nothing  # i.e., panel data  

      if seed == -1
        select_row = sample(     axes(rowIDT0,1), size(rowIDT0,1); replace=true)  # require StatsBase.jl
      else
        select_row = sample(rng, axes(rowIDT0,1), size(rowIDT0,1); replace=true)  # require StatsBase.jl
      end

      yvar, newidvar = mySample(rowIDT0, yvar0, select_row)
      xvar, _ = mySample(rowIDT0, xvar0, select_row)

      (zvar0 == ()) ||  ((zvar, _) = mySample(rowIDT0, zvar0, select_row))
      (qvar0 == ()) ||  ((qvar, _) = mySample(rowIDT0, qvar0, select_row))
      (wvar0 == ()) ||  ((wvar, _) = mySample(rowIDT0, wvar0, select_row))
      (vvar0 == ()) ||  ((vvar, _) = mySample(rowIDT0, vvar0, select_row))

      rowIDT = get_rowIDT(vec(newidvar))

    end

   
 #*  --- other small things -----  

   nofobs1 = size(yvar,1)

  #* ----- Define the objective function -----#

   _lik2 = rho -> SFmle.LL_T(result.modelid,
                         yvar, xvar, zvar, qvar, wvar, vvar,
                         _porc, nofobs1, pos, rho,
                                 rowIDT, mymisc)

  #* ---- estimate ----------------- *#

             _optres = try
                       optimize(_lik2,
                                sf_init,       # initial values
                                sf_algo,       # different from search run
                                Optim.Options(g_tol = sf_tol,
                                              f_tol=0.0, # force `Optim` to ignore this, but sometimes it does meet the 0.0 criterion
                                              x_tol=0.0, # same above
                                              iterations  = sf_maxit, # different from search run
                                              store_trace = false,
                                              show_trace  = false);
                                autodiff = AutoForwardDiff())
                    catch err
                       @goto start1
                    end 

#* ###### check if valid ############### 


if (Optim.iteration_limit_reached(_optres) ) || 
     (isnan(Optim.g_residual(_optres)) ) ||  
     (Optim.g_residual(_optres) > sf_tol)  # hjw!! was: 1e-1
         @goto start1
  end  

    _coevec            = Optim.minimizer(_optres)  # coef. vec.
    numerical_hessian  = ForwardDiff.hessian(_lik2, _coevec)  # Hessian

   #* ------ Check if the matrix is invertible. ----

   var_cov_matrix = try
                       inv(numerical_hessian)
                    catch err 
                       @goto start1
                    end 

        if !all( diag(var_cov_matrix) .> 0 ) # not all are positive
              @goto start1
        end              

   #* ---- marginal effect on E(u) -------------- 

   margeff, margMinfo = get_marg(result.modelid, pos, num, _coevec, zvar, qvar, wvar)

   theM = mean.(eachcol(margeff))  # mean marginal effect

   if sum(isnan.(theM)) != 0  # in this run some of the element is NaN
      @goto start1
   end 

   sim_res[:, i] = theM

   if result.verbose 
      if i == 1
         printstyled("bootstrap in progress.."; color = :cyan)
      end    
      if i%every == 0   # print for every 10 replications
          print("$(i)..")
      end    
      if i == R
          printstyled("Done!\n\n"; color = :cyan)    
      end
   end

end   # for i=1:R

 sim_res = sim_res'  # nofobs x K; ith row = ith replication, column = statistics

 #*###### compute statistics for the mean marginal effect #####*#

   theMean = mean.(eachcol(result.marginal))  # Kx1 vector
   theSTD  = sqrt.( sum((sim_res .- theMean').^2, dims=1) ./(R-1)) # 1 x K

   ci_mat = sfmodel_CI(bootdata=sim_res, observed=theMean, level=level, verbose=false);

   if result.verbose 

      table = Array{Any}(undef, length(theMean), 3+1)
      vname = names(result.marginal)

      for j in 1:length(theMean) 
          table[j, 1] = string(vname[j])[6:end]
      end
      
      table[:,2], table[:,3] = theMean, theSTD
      table[:,4] = ci_mat

       mylevel = 100*(1-level)

      table = [" " "mean of the marginal" "std.err. of the"  "bias-corrected"; 
               " " "effect on E(u)"       "mean effect"      "$(mylevel)%  conf. int.";
               table]  

            pretty_table(table;
                         show_header = false,
                         body_hlines = [2],
                         formatters = [fmt__printf("%.5f", collect(2:4))],
                         compact_printing = true,
                         backend = sf_table)
            println()

   end

   if getBootData == false   # most likely
      return hcat(theSTD', ci_mat) # K x 1
   else
      return hcat(theSTD', ci_mat), sim_res
   end

end # sfmodel_boot_marginal

# -----------------------------------------------#
# --- utilities for sfmodel_boot_marginal -------#
# -----------------------------------------------#


"""
    mySample(rowIDT, data, select)

A utility to construct sampled panel data, for both balanced and 
unbalanced panels. The `select` indicates the ith firm that is chosen 
to be included in the sample. The firm's row number is in `rowIDT`. 
The information is then used to obtained the ith firm data from `data`. 
It returns the sampled data and a vector of generated individual id.

# Arguments
- rowIDT: Typically from `get_rowIDT()`, which is a Nx2 matrix of the 
  following form:

  [ [1, 2, 3]     3
    [4, 5, 6, 7]  4 ]

   where the first colum is the row number of an individual, and the 2nd
   column is the number of time periods in the firm.
 - data: The data to take the sample from.
 - select: a vector indicating the selected rows. Typically an output
   from `sample()`.
"""

function mySample(rowIDT, data, select)
  newobs = 0.0  # number of obs in the new dataset
  for i in 1:length(select)  
      newobs += rowIDT[select[i], 2]  
  end

  sampled = Array{Float64}(undef, Int(newobs), size(data,2))  # this is why we need newobs
    newid = Array{Float64}(undef, Int(newobs), 1)

  start_row = 1  
  for i in 1:length(select)
      nrows = rowIDT[select[i], 2]  # n of obs of the chosen individual
      sampled[start_row : start_row + nrows-1, :] .= data[rowIDT[select[i],1], :]  # the ith firm data
        newid[start_row : start_row + nrows-1, :] .= reshape(ones(nrows)*i, :, 1)
      start_row += nrows  
  end
  return sampled, newid
end


"""
    sfmodel_CI(<keyword arguments>)

A general purpose (not specific to stochastic frontier models) function for
obtaining the standard error and the bias-correctred (BC) confidence intervals 
from bootstrapped data. Note that the standard error may be influenced by
extreme values of the bootstrapped while the percentile-based CI is less
sensitive to them. Return a ``K x 1`` matrix of confidence intervals in the 
form of tuples, where ``K`` is the number of bootstrap statistics.

See also the help file on `sfmodel_boot_marginal()`.

# Arguments
- bootdata::Array=<data>: The bootstrapped data of size ``R x K``,
  where ``R`` is the number of bootstrap samples (replications) and ``K`` is
  the number of statistics. An example is the bootstrapped data from
  `sfmodel_boot_marginal(, ... getBootData=true)`.
- observed::Union{Vector, Real, Tuple, NamedTuple}=<a vector of numbers>: The
  observed values of the statistics to which the confidence intervals are to
  be calculated. The length of `observed` should be equal to ``K``. It could take
  the form of a single value (if ``K=1``), a vector, a tuple, or a NamedTuple.
- level::Real=<number>: The significance level (default=0.05) of the bias-corrected
  confidence intervals. If `level`>0.5, it is automatically transformed to
  `1-level`, such that `level=0.05` and `level=0.95` both return 95%
  confidene intervals at the 5% significance level.  
- verbose::Bool=true: Print the result.

# Examples
```julia
julia> myans = sfmodel_fit(useData(df));

 (output omitted)

julia> std_ci, bsdata = sfmodel_boot_marginal(result=myans, data=df, R=250, seed=123, getBootData=true);

 (output omitted)

julia> sfmodel_CI(bootdata=bsdata, observed=myans.marginal_mean, level=0.10) 

Bias-Corrected 90.0% Confidence Interval:
 
3×1 Matrix{Any}:
 (-0.00655, 0.0009)
 (-0.04064, 0.0086)
 (-0.04663, -0.00872)

julia> # manually input observed values

julia> sfmodel_CI(bootdata=bsdata, observed=(-0.00264, -0.01197, -0.0265), level=0.10)
 
Bias-Corrected 90.0% Confidence Interval:
 
3×1 Matrix{Any}:
 (-0.00655, 0.0009)
 (-0.04064, 0.0086)
 (-0.04663, -0.00872)
```
"""
function sfmodel_CI(; bootdata::Any=nothing, observed::Union{Vector, Real, Tuple, NamedTuple}=nothing, level::Real=0.05, verbose::Bool=true)

    # bias-corrected (but not accelerated) confidence interval 
    # For the "accelerated" factor, need to estimate the SF model 
    #    for every jack-knifed sample, which is expensive.

    if (observed isa NamedTuple)
        observed = values(observed)
    end

   ((level > 0.0) && (level < 1.0)) || throw("The significance level (`level`) should be between 0 and 1.")

   if level > 0.50
      level = 1-level  # 0.95 -> 0.05
   end

    nofobs, nofK = size(bootdata)  # number of statistics

    (nofK == length(observed)) || throw("The number of statistics (`observed`) does not fit the number of columns of bootstrapped data.")
    
    ci = Array{Any}(undef, nofK, 1)

    z1 = quantile(Normal(), level/2)
    z2 = quantile(Normal(), 1 - level/2)  #! why z1 != z2?


    for i in 1:nofK
        @views data = bootdata[:,i]

        count = sum(data .< observed[i])
        z0 = quantile(Normal(), count/nofobs) # bias corrected factor

        alpha1 = cdf(Normal(), z0 + ((z0 + z1) ))
        alpha2 = cdf(Normal(), z0 + ((z0 + z2) ))

        order_data = sort(data)

        pos1 = max(Int(ceil(nofobs*alpha1)), 1) # make sure it is not 0 which will cause error
        pos2 = max(Int(ceil(nofobs*alpha2)), 1)

        ci[i,1] = (  round(order_data[pos1], digits=5),    round(order_data[pos2], digits=5)  )
    end

    if verbose == true
       mylevel = 100*(1-level)
       println("\nBias-Corrected $(mylevel)% Confidence Interval:\n")
    end

    return ci
end

########################################################
####            catching type error                 ####
########################################################


function sfmodel_init(arg::Vector) 
    throw(ArgumentError("The initial values in sfmodel_init() are specified incorrectly. They must be supplied in macros such as all_init(0.1, 0.1)."))
end    

function sfmodel_init(args::Number) 
    throw(ArgumentError("The initial values in sfmodel_init() are specified incorrectly. They must be supplied in macros such as all_init(0.1, 0.1)."))
end    

function sfmodel_fit(sfdat::Any) 
    throw(ArgumentError("The dataset specified in sfmodel_fit() must be a DataFrame."))
end

