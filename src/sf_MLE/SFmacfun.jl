# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

###################################################################
##  macro and functions for sfmodel_spec(), _init(), and _fit()  ##
###################################################################


#? ---- macros for sfmodel_spec() and functions for sfmodel_init() ------------

#=
macro sfdist(arg::Vararg)
 
    (length(arg)==1)||throw("@sfdist in sfmodel_spec should have one and only one word.")
 
     if arg[1] ∈ (:Trun, :Truncated, :truncated, :trun, :t)
        ha = :t
     elseif arg[1] ∈ (:Half, :half, :h)
        ha = :h
     elseif arg[1] ∈ (:Expo, :expo, :e)
        ha = :e
     elseif arg[1] ∈ (:Trun_Scale, :Trun_Scaling, :trun_scale, :trun_scaling, :ts, :s)
        ha = :s
     elseif arg[1] ∈ (:Half_Scale, :Half_Scaling, :half_scale, :half_scaling, :hs)
        throw("The half-normal model is naturally a scaling-property model. Please use @sfdist(half) specification.")
     else   
        throw("The keyword of @sfdist in sfmodel_spec() is specified incorrectly.")             
     end     
 
    return (:dist, [ha])
 end   
 =#

"""
 sfdist(arg::Vararg)

An argument in `sfmodel_sepc()`. Specify the distribution assumption of the one-sided stochastic
variable (aka inefficiency term) of the model. Possible choices include `truncated` (or `trun`, `t`), `half` (or `h`),
`exponential` (or `expo`, `e`), and `trun_scale` (or `trun_scaling`, `ts`).

See the help on `sfmodel_spec()` for more information.

# Examples
```julia
sfmodel_spec(sfdist(t), ...)
sfmodel_spec(sfdist(h), ...)
```
"""    
function sfdist(arg::Vararg)
 
    (length(arg)==1) || throw("`sfdist()` in sfmodel_spec should have one and only one word.")
 
     if arg[1] ∈ (Trun, trun, t)
        ha = :t
     elseif arg[1] ∈ (Half, half, h)
        ha = :h
     elseif arg[1] ∈ (Expo, expo, e)
        ha = :e
     elseif arg[1] ∈ (Trun_Scale, trun_scale, trun_scaling, s)
        ha = :s
     elseif arg[1] ∈ (Half_Scale, half_scale, half_scaling)
        throw("The half-normal model is naturally a scaling-property model. Use `sfdist(half)` specification.")
     else   
        throw("The inefficiency distribution in `sfdist()` of sfmodel_spec() is specified incorrectly.")             
     end     
 
    return (:dist, [ha])
 end  
 

  # -------------------
 
#=  
macro sftype(arg::Vararg)
  (length(arg)==1) || throw("`@sftype()` in sfmodel_spec should be either production, prod, or cost.")    
  if (arg[1]!=:production) && (arg[1]!=:prod) && (arg[1]!=:cost)
       throw("@sftype in sfmodel_spec should be either production, prod, or cost.")    
  end
  return (:type, collect(:($(arg))))
end   
=# 


"""
 sftype(arg::Vararg)

An argument in `sfmodel_sepc()`. Specify whether the model is a `production` (or `prod`) frontier
or a `cost` frontier.

See the help on `sfmodel_spec()` for more information.

# Examples
```julia
sftype(production)
sftype(cost)
```
"""    
function sftype(arg::Vararg) 
   (length(arg)==1) || throw("`sftype()` in sfmodel_spec should be either production, prod, or cost.")    
   if (arg[1]!=production) && (arg[1]!=prod) && (arg[1]!=cost)
        throw("`sftype()` in sfmodel_spec should be either production, prod, or cost.")    
   end
   return (:type, [Symbol(arg[1])])
end  


 
  # -------------------
 #= 
 macro model(arg::Vararg)
     (arg[1]==:BC1995) ||(arg[1]==:Wang2002)  || throw("@model only supports BC1995, Wang2002")
     return (:type, collect(:($(arg))))
 end
  =#

# Old @depvar macro removed — replaced by new version below (line ~1016)
 


"""
depvar(arg::Vararg)
 
An argument in `sfmodel_sepc()`. Specify the maxrix name where the matrix
is used as the data of the dependent variable.
 
See the help on `sfmodel_spec()` for more information.
 
# Examples
```julia
julia> ymat
100×1 Matrix{Float64}:
  0.0005852467656204083
 -0.9128698116696892
 -1.1087862468093093
 -1.0714018769972091
 -0.606943663382492
  0.37648407866805467
 -0.1281971631844683
  ⋮
  0.8541895741866585
  1.5109216952026845
 -0.3519833126683764
 -1.0378799750720447
 -0.9990384371507885
  0.18858962788775305

sfmodel_spec(depvar(ymat), ...)
```
"""
 function depvar(arg::Vararg)
    return (:depvar, collect(:($(arg))))
end

  # -------------------
# Old @frontier macro removed — replaced by new version below (line ~989) 
 


"""
frontier(arg::Vararg)
 
An argument in `sfmodel_sepc()`. Specify the name of the maxrix
used as the data of the `frontier` function.
 
See the help on `sfmodel_spec()` for more information.
 
# Examples
```julia
julia> xmat
100×3 Matrix{Float64}:
 -0.682942    0.440045   1.0
 -0.680902   -1.68124    1.0
  1.29108    -1.5516     1.0
  0.683652   -0.0319451  1.0
 -0.973079    1.11636    1.0
 -0.343229    0.314457   1.0
  0.107583    0.688177   1.0
  ⋮
  0.0943377  -0.781928   1.0
 -0.599142   -1.01591    1.0
 -0.56726    -1.03394    1.0
  1.33522     0.135763   1.0
  1.13235     0.0177493  1.0
 -0.310638   -0.314166   1.0

sfmodel_spec(frontier(xmat), ...)
```
"""
 function frontier(arg::Vararg)    # for sfmodel_init(0.1, 0.1)
     return (:frontier, collect(:($(arg))) )
 end
 


 
  # -------------------
 
  """
  @μ(arg::Vararg)
   
  An argument in `sfmodel_sepc()`. Specify the variables in the `μ`
  function using column names from a DataFrame. 
   
  See the help on `sfmodel_spec()` for more information.
   
  # Examples
  ```julia
  julia> df
  100×5 DataFrame
  │ Row │ yvar  │ xvar1     │ xvar2     │ zvar      │ _cons   │
  │     │ Int64 │ Float64   │ Float64   │ Float64   │ Float64 │
  ├─────┼───────┼───────────┼───────────┼───────────┼─────────┤
  │ 1   │ 1     │ 0.0306449 │ 0.452148  │ 0.808817  │ 1.0     │
  │ 2   │ 2     │ 0.460691  │ 0.296092  │ 0.454545  │ 1.0     │
  │ 3   │ 3     │ 0.897503  │ 0.376972  │ 0.907454  │ 1.0     │
  │ 4   │ 4     │ 0.682894  │ 0.776861  │ 0.161721  │ 1.0     │
  ⋮
  │ 96  │ 96    │ 0.329647  │ 0.0914057 │ 0.825032  │ 1.0     │
  │ 97  │ 97    │ 0.0781165 │ 0.338999  │ 0.761652  │ 1.0     │
  │ 98  │ 98    │ 0.41394   │ 0.0063118 │ 0.295372  │ 1.0     │
  │ 99  │ 99    │ 0.516381  │ 0.285415  │ 1.91995   │ 1.0     │
  │ 100 │ 100   │ 0.944     │ 0.702226  │ -0.539848 │ 1.0     │
  
  sfmodel_spec( @μ(zvar, _cons), ...)
  ```
  """  
 macro μ(arg::Vararg)
     return (:μ, collect(:($(arg))))
 end
 



 """
 μ(arg::Vararg)
  
 An argument in `sfmodel_sepc()`. Specify the name of the maxrix
 used as the data of the `μ` function.
  
 See the help on `sfmodel_spec()` for more information.
  
 # Examples
 ```julia
 julia> mumat
 100×2 Matrix{Float64}:
  -0.943133   1.0
  -0.897392   1.0
   0.585447   1.0
  -0.46106    1.0
  -0.54563    1.0
  -0.619428   1.0
   0.0575559  1.0
   ⋮
   0.0844192  1.0
  -1.3339     1.0
   1.29332    1.0
   0.691466   1.0
   0.422962   1.0
   0.374425   1.0
 
 sfmodel_spec(μ(mumat), ...)
 ```
 """
 function μ(arg::Vararg)
     return (:μ, collect(:($(arg))))
 end
 


 
 
  # -------------------
 
  """
  @mu(arg::Vararg)
   
  alias of @μ. See help on @μ.
  """  
 macro mu(arg::Vararg)
     return (:μ, collect(:($(arg))))
 end

 
 """
 mu(arg::Vararg)
  
 alias of μ. See help on μ.
 """  
 function mu(arg::Vararg)
     return (:μ, collect(:($(arg))))
 end
 
 
  # -------------------
 
  """
  @σᵤ²(arg::Vararg)
   
  An argument in `sfmodel_sepc()`. Specify the variables in the `σᵤ²`
  function using column names from a DataFrame. 
   
  See the help on `sfmodel_spec()` for more information.
   
  # Examples
  ```julia
  julia> df
  100×5 DataFrame
  │ Row │ yvar  │ xvar1     │ xvar2     │ zvar      │ _cons   │
  │     │ Int64 │ Float64   │ Float64   │ Float64   │ Float64 │
  ├─────┼───────┼───────────┼───────────┼───────────┼─────────┤
  │ 1   │ 1     │ 0.0306449 │ 0.452148  │ 0.808817  │ 1.0     │
  │ 2   │ 2     │ 0.460691  │ 0.296092  │ 0.454545  │ 1.0     │
  │ 3   │ 3     │ 0.897503  │ 0.376972  │ 0.907454  │ 1.0     │
  │ 4   │ 4     │ 0.682894  │ 0.776861  │ 0.161721  │ 1.0     │
  ⋮
  │ 96  │ 96    │ 0.329647  │ 0.0914057 │ 0.825032  │ 1.0     │
  │ 97  │ 97    │ 0.0781165 │ 0.338999  │ 0.761652  │ 1.0     │
  │ 98  │ 98    │ 0.41394   │ 0.0063118 │ 0.295372  │ 1.0     │
  │ 99  │ 99    │ 0.516381  │ 0.285415  │ 1.91995   │ 1.0     │
  │ 100 │ 100   │ 0.944     │ 0.702226  │ -0.539848 │ 1.0     │
  
  sfmodel_spec( @σᵤ²(zvar, _cons), ...)
  ```
  """  
 macro σᵤ²(arg::Vararg)
     return (:σᵤ², collect(:($(arg))))
 end
 

 """
 σᵤ²(arg::Vararg)
  
 An argument in `sfmodel_sepc()`. Specify the name of the maxrix
 used as the data of the `σᵤ²` function.
  
 See the help on `sfmodel_spec()` for more information.
  
 # Examples
 ```julia
 julia> zmat
 100×2 Matrix{Float64}:
  -0.943133   1.0
  -0.897392   1.0
   0.585447   1.0
  -0.46106    1.0
  -0.54563    1.0
  -0.619428   1.0
   0.0575559  1.0
   ⋮
   0.0844192  1.0
  -1.3339     1.0
   1.29332    1.0
   0.691466   1.0
   0.422962   1.0
   0.374425   1.0
 
 sfmodel_spec(σᵤ²(zmat), ...)
 ```
 """
 function σᵤ²(arg::Vararg)
     return (:σᵤ², collect(:($(arg))))
 end
 

 
 
  # -------------------
 
  """
  @sigma_u_2(arg::Vararg)
   
  alias of @σᵤ². See help on @σᵤ².
  """  
 macro sigma_u_2(arg::Vararg)
     return (:σᵤ², collect(:($(arg))))
 end
 
 """
 sigma_u_2(arg::Vararg)
  
 alias of σᵤ². See help on σᵤ².
 """ 
 function sigma_u_2(arg::Vararg)
     return (:σᵤ², collect(:($(arg))))
 end
 
 
 
  # -------------------
 
  """
  @σᵥ²(arg::Vararg)
   
  An argument in `sfmodel_sepc()`. Specify the variables in the `σᵥ²`
  function using column names from a DataFrame. 
   
  See the help on `sfmodel_spec()` for more information.
   
  # Examples
  ```julia
  julia> df
  100×5 DataFrame
  │ Row │ yvar  │ xvar1     │ xvar2     │ zvar      │ _cons   │
  │     │ Int64 │ Float64   │ Float64   │ Float64   │ Float64 │
  ├─────┼───────┼───────────┼───────────┼───────────┼─────────┤
  │ 1   │ 1     │ 0.0306449 │ 0.452148  │ 0.808817  │ 1.0     │
  │ 2   │ 2     │ 0.460691  │ 0.296092  │ 0.454545  │ 1.0     │
  │ 3   │ 3     │ 0.897503  │ 0.376972  │ 0.907454  │ 1.0     │
  │ 4   │ 4     │ 0.682894  │ 0.776861  │ 0.161721  │ 1.0     │
  ⋮
  │ 96  │ 96    │ 0.329647  │ 0.0914057 │ 0.825032  │ 1.0     │
  │ 97  │ 97    │ 0.0781165 │ 0.338999  │ 0.761652  │ 1.0     │
  │ 98  │ 98    │ 0.41394   │ 0.0063118 │ 0.295372  │ 1.0     │
  │ 99  │ 99    │ 0.516381  │ 0.285415  │ 1.91995   │ 1.0     │
  │ 100 │ 100   │ 0.944     │ 0.702226  │ -0.539848 │ 1.0     │
  
  sfmodel_spec( @σᵥ²(zvar, _cons), ...)
  ```
  """    
 macro σᵥ²(arg::Vararg)
     return (:σᵥ², collect(:($(arg))))
 end
 

 """
 σᵥ²(arg::Vararg)
  
 An argument in `sfmodel_sepc()`. Specify the name of the maxrix
 used as the data of the `σᵥ²` function.
  
 See the help on `sfmodel_spec()` for more information.
  
 # Examples
 ```julia
 julia> sigv2mat
 100×2 Matrix{Float64}:
  -0.943133   1.0
  -0.897392   1.0
   0.585447   1.0
  -0.46106    1.0
  -0.54563    1.0
  -0.619428   1.0
   0.0575559  1.0
   ⋮
   0.0844192  1.0
  -1.3339     1.0
   1.29332    1.0
   0.691466   1.0
   0.422962   1.0
   0.374425   1.0
 
 sfmodel_spec(σᵥ²(sigv2mat), ...)
 ```
 """
 function σᵥ²(arg::Vararg)
     return (:σᵥ², collect(:($(arg))))
 end
 
 
  # -------------------
 

"""
  @sigma_v_2(arg::Vararg)
  
 alias of @σᵥ². See help on @σᵥ².
"""   
 macro sigma_v_2(arg::Vararg)
     return (:σᵥ², collect(:($(arg))))
 end
 
 """
  sigma_v_2(arg::Vararg)
  
 alias of σᵥ². See help on σᵥ².
 """  
 function sigma_v_2(arg::Vararg)
     return (:σᵥ², collect(:($(arg))))
 end
 
 
 # -------------------

 """
 @σₐ²(arg::Vararg)
  
 An argument in `sfmodel_sepc()`. Specify the variables in the `σₐ²`
 function using column names from a DataFrame. 
  
 See the help on `sfmodel_spec()` for more information.
  
 # Examples
 ```julia
 julia> df
 100×5 DataFrame
 │ Row │ yvar  │ xvar1     │ xvar2     │ zvar      │ _cons   │
 │     │ Int64 │ Float64   │ Float64   │ Float64   │ Float64 │
 ├─────┼───────┼───────────┼───────────┼───────────┼─────────┤
 │ 1   │ 1     │ 0.0306449 │ 0.452148  │ 0.808817  │ 1.0     │
 │ 2   │ 2     │ 0.460691  │ 0.296092  │ 0.454545  │ 1.0     │
 │ 3   │ 3     │ 0.897503  │ 0.376972  │ 0.907454  │ 1.0     │
 │ 4   │ 4     │ 0.682894  │ 0.776861  │ 0.161721  │ 1.0     │
 ⋮
 │ 96  │ 96    │ 0.329647  │ 0.0914057 │ 0.825032  │ 1.0     │
 │ 97  │ 97    │ 0.0781165 │ 0.338999  │ 0.761652  │ 1.0     │
 │ 98  │ 98    │ 0.41394   │ 0.0063118 │ 0.295372  │ 1.0     │
 │ 99  │ 99    │ 0.516381  │ 0.285415  │ 1.91995   │ 1.0     │
 │ 100 │ 100   │ 0.944     │ 0.702226  │ -0.539848 │ 1.0     │
 
 sfmodel_spec( @σₐ²(_cons), ...)
 ```
 """   
macro σₐ²(arg::Vararg)
    return (:σₐ², collect(:($(arg))))
end


"""
σₐ²(arg::Vararg)
 
An argument in `sfmodel_sepc()`. Specify the name of the maxrix
used as the data of the `σₐ²` function.
 
See the help on `sfmodel_spec()` for more information.
 
# Examples
```julia
julia> amat
100×2 Matrix{Float64}:
 -0.943133   1.0
 -0.897392   1.0
  0.585447   1.0
 -0.46106    1.0
 -0.54563    1.0
 -0.619428   1.0
  0.0575559  1.0
  ⋮
  0.0844192  1.0
 -1.3339     1.0
  1.29332    1.0
  0.691466   1.0
  0.422962   1.0
  0.374425   1.0

sfmodel_spec(σₐ²(amat), ...)
```
"""
function σₐ²(arg::Vararg)
    return (:σₐ², collect(:($(arg))))
end



 # -------------------

 """
 @sigma_a_2(arg::Vararg)
  
 alias of @σₐ². See help on @σₐ².
 """   
macro sigma_a_2(arg::Vararg)
    return (:σₐ², collect(:($(arg))))
end

"""
sigma_a_2(arg::Vararg)
 
alias of σₐ². See help on σₐ².
"""  
function sigma_a_2(arg::Vararg)
    return (:σₐ², collect(:($(arg))))
end


  # -------------------

#=  
  
 macro λ(arg::Vararg)
     return (:λ, collect(:($(arg))))
 end
 
 function λ(arg::Vararg)
     return (:λ, collect(:($(arg))))
 end
 
 function λ(arg::Vector)
     return (:λ, arg)
 end

 =#
 
  # -------------------
 
#=

 macro lambda(arg::Vararg)
     return (:λ, collect(:($(arg))))
 end
 
 function lambda(arg::Vararg)
     return (:λ, collect(:($(arg))))
 end
 
 function lambda(arg::Vector)
     return (:λ, arg)
 end

=#

# -------------------

#=

macro τ(arg::Vararg)
    return (:τ, collect(:($(arg))))
end

function τ(arg::Vararg)
    return (:τ, collect(:($(arg))))
end

function τ(arg::Vector)
    return (:τ, arg)
end

=#

 # -------------------

#=

macro tau(arg::Vararg)
    return (:τ, collect(:($(arg))))
end

function tau(arg::Vararg)
    return (:τ, collect(:($(arg))))
end

function tau(arg::Vector)
    return (:τ, arg)
end

=#

 # -------------------

 """
 @hscale(arg::Vararg)
  
 An argument in `sfmodel_sepc()`. Specify the variables in the `hscale`
 function using column names from a DataFrame. 
  
 See the help on `sfmodel_spec()` for more information.
  
 # Examples
 ```julia
 julia> df
 100×5 DataFrame
 │ Row │ yvar  │ xvar1     │ xvar2     │ zvar      │ _cons   │
 │     │ Int64 │ Float64   │ Float64   │ Float64   │ Float64 │
 ├─────┼───────┼───────────┼───────────┼───────────┼─────────┤
 │ 1   │ 1     │ 0.0306449 │ 0.452148  │ 0.808817  │ 1.0     │
 │ 2   │ 2     │ 0.460691  │ 0.296092  │ 0.454545  │ 1.0     │
 │ 3   │ 3     │ 0.897503  │ 0.376972  │ 0.907454  │ 1.0     │
 │ 4   │ 4     │ 0.682894  │ 0.776861  │ 0.161721  │ 1.0     │
 ⋮
 │ 96  │ 96    │ 0.329647  │ 0.0914057 │ 0.825032  │ 1.0     │
 │ 97  │ 97    │ 0.0781165 │ 0.338999  │ 0.761652  │ 1.0     │
 │ 98  │ 98    │ 0.41394   │ 0.0063118 │ 0.295372  │ 1.0     │
 │ 99  │ 99    │ 0.516381  │ 0.285415  │ 1.91995   │ 1.0     │
 │ 100 │ 100   │ 0.944     │ 0.702226  │ -0.539848 │ 1.0     │
 
 sfmodel_spec( @hscale(zvar, _cons), ...)
 ```
 """   
macro hscale(arg::Vararg)
    return (:hscale, collect(:($(arg))))
end

"""
hscale(arg::Vararg)
 
An argument in `sfmodel_sepc()`. Specify the name of the maxrix
used as the data of the `hscale` function.
 
See the help on `sfmodel_spec()` for more information.
 
# Examples
```julia
julia> hmat
100×2 Matrix{Float64}:
 -0.943133   1.0
 -0.897392   1.0
  0.585447   1.0
 -0.46106    1.0
 -0.54563    1.0
 -0.619428   1.0
  0.0575559  1.0
  ⋮
  0.0844192  1.0
 -1.3339     1.0
  1.29332    1.0
  0.691466   1.0
  0.422962   1.0
  0.374425   1.0

sfmodel_spec(hscale(hmat), ...)
```
"""
function hscale(arg::Vararg)
    return (:hscale, collect(:($(arg))))
end

 


 # ---- time decay model ---------------

 """
 @gamma(arg::Vararg)
  
 An argument in `sfmodel_sepc()`. Specify the variables in the `gamma`
 function using column names from a DataFrame. 
  
 See the help on `sfmodel_spec()` for more information.
  
 # Examples
 ```julia
 julia> df
 100×5 DataFrame
 │ Row │ yvar  │ xvar1     │ xvar2     │ zvar      │ _cons   │
 │     │ Int64 │ Float64   │ Float64   │ Float64   │ Float64 │
 ├─────┼───────┼───────────┼───────────┼───────────┼─────────┤
 │ 1   │ 1     │ 0.0306449 │ 0.452148  │ 0.808817  │ 1.0     │
 │ 2   │ 2     │ 0.460691  │ 0.296092  │ 0.454545  │ 1.0     │
 │ 3   │ 3     │ 0.897503  │ 0.376972  │ 0.907454  │ 1.0     │
 │ 4   │ 4     │ 0.682894  │ 0.776861  │ 0.161721  │ 1.0     │
 ⋮
 │ 96  │ 96    │ 0.329647  │ 0.0914057 │ 0.825032  │ 1.0     │
 │ 97  │ 97    │ 0.0781165 │ 0.338999  │ 0.761652  │ 1.0     │
 │ 98  │ 98    │ 0.41394   │ 0.0063118 │ 0.295372  │ 1.0     │
 │ 99  │ 99    │ 0.516381  │ 0.285415  │ 1.91995   │ 1.0     │
 │ 100 │ 100   │ 0.944     │ 0.702226  │ -0.539848 │ 1.0     │
 
 sfmodel_spec( @gamma(zvar, _cons), ...)
 ```
 """   
macro gamma(arg::Vararg)
    return (:gamma, collect(:($(arg))))
end

"""
gamma(arg::Vararg)
 
An argument in `sfmodel_sepc()`. Specify the name of the maxrix
used as the data of the `gamma` function.
 
See the help on `sfmodel_spec()` for more information.
 
# Examples
```julia
julia> gmat
100×2 Matrix{Float64}:
 -0.943133   1.0
 -0.897392   1.0
  0.585447   1.0
 -0.46106    1.0
 -0.54563    1.0
 -0.619428   1.0
  0.0575559  1.0
  ⋮
  0.0844192  1.0
 -1.3339     1.0
  1.29332    1.0
  0.691466   1.0
  0.422962   1.0
  0.374425   1.0

sfmodel_spec(gamma(gmat), ...)
```
"""
function gamma(arg::Vararg)
    return (:gamma, collect(:($(arg))))
end


 # ------ panel model -------------

  macro sfpanel(arg::Vararg)
     
    (length(arg)==1)||throw("@sfpanel in sfmodel_spec should have one single string.")
    
    if !(arg[1] ∈ (:TFE_WH2010, :TFE_CSW2014, :TFE_G2005, :TRE, :TimeDecay, :Kumbhakar1990 ))
         throw("The keyword of @sfpanel in sfmodel_spec is specified incorrectly.")
    end 
    
    if arg[1] == :TFE_G2005
        throw("The TEF_G2005 panel model is essentially a cross-sectional model with dummies of individuals. Please use non-panel specifications and add individual dummies to @frontier().")
    end 

    return (:panel, collect(:($(arg))))
 end 


 function sfpanel(arg::Vararg)
     
    (length(arg)==1) || throw("`sfpanel()` in sfmodel_spec should have one single string.")
    
    if !(arg[1] ∈ (TFE_WH2010, TFE_CSW2014, TFE_G2005, TRE, TimeDecay, Kumbhakar1990 ))
         throw("The keyword of `sfpanel` in sfmodel_spec is specified incorrectly.")
    end 
    
    if arg[1] == TFE_G2005
        throw("The TEF_G2005 panel model is essentially a cross-sectional model with dummies of individuals. Please use non-panel specifications and add individual dummies to @frontier().")
    end 

    return (:panel, [Symbol(arg[1])])
 end


"""
@idvar(arg::Vararg)
 
An argument in `sfmodel_sepc()` for panel data models. Specify the column
name of a DataFrame that contain the individual's id information of the panel data. 

See the help on `sfmodel_spec()` for more information.
 
# Examples
```julia
julia> df
100×4 DataFrame
│ Row │ year  │ firm  │ yvar  │ xvar1     │
│     │ Int64 │ Int64 │ Int64 │ Float64   │
├─────┼───────┼───────┼───────┼───────────┤
│ 1   │ 2019  │ 1     │ 1     │ 0.77645   │
│ 2   │ 2020  │ 1     │ 2     │ 0.0782388 │
│ 3   │ 2021  │ 1     │ 3     │ 0.222884  │
│ 4   │ 2022  │ 1     │ 4     │ 0.762864  │
⋮
│ 96  │ 2022  │ 24    │ 96    │ 0.590184  │
│ 97  │ 2019  │ 25    │ 97    │ 0.364425  │
│ 98  │ 2020  │ 25    │ 98    │ 0.639463  │
│ 99  │ 2021  │ 25    │ 99    │ 0.500526  │
│ 100 │ 2022  │ 25    │ 100   │ 0.239137  │

sfmodel_spec( @idvar(firm), ...)
```
"""   
macro idvar(arg::Vararg)
    return (:idvar, collect(:($(arg))))
end


"""
idvar(arg::Vararg)
 
An argument in `sfmodel_sepc()` for panel data models. Specify the name of the
matrix containing the individual's id information of the panel data. 
 
See the help on `sfmodel_spec()` for more information.
 
# Examples
```julia
julia> idmat
100-element Vector{Int64}:
  1
  1
  1
  1
  2
  2
  2
  ⋮
 24
 24
 25
 25
 25
 25

sfmodel_spec( idvar(idmat), ...)
```
"""   
function idvar(arg::Vararg)
    return (:idvar, collect(:($(arg))))
end








     #* ------  variables by vector type, not in use -----
 #=
 macro xvar(arg::Vararg)
     return (:xvar, collect(:($(arg))))
 end

  # -------------------
 
 macro zvar(arg::Vararg)
     return (:zvar, collect(:($(arg))))
 end
 
  # -------------------
 
 macro etavar(arg::Vararg)
     return (:etavar, collect(:($(arg))))
 end
 =#
 
 
 
 #? ----  (sfmodel_opt helper functions removed — now uses keyword API directly) --------
 
 #? ---- New DSL types and macros (SFrontiers-style) ----

 # Struct types for the new DSL-style sfmodel_spec().
 # Only @useData, @zvar, and @id are new macros.
 # The existing @depvar, @frontier, @idvar macros are NOT
 # redefined here to preserve backward compatibility with sfmodel_MoMTest
 # and other code that uses the old tuple-returning macros.
 # Users construct DSLArg_MLE structs directly or via the new macros.

 abstract type DSLArg_MLE end

 struct WUseDataSpec_MLE <: DSLArg_MLE
     df::DataFrame
 end

 struct WDepvarSpec_MLE <: DSLArg_MLE
     name::Symbol
 end

 struct WFrontierSpec_MLE <: DSLArg_MLE
     names::Vector{Symbol}
 end

 struct WZvarSpec_MLE <: DSLArg_MLE
     names::Vector{Symbol}
 end

 struct WIdSpec_MLE <: DSLArg_MLE
     name::Symbol
 end

 # New macros — return DSLArg_MLE wrapper types (override old tuple-returning macros)
 macro useData(df)
     :(SFmle.WUseDataSpec_MLE($(esc(df))))
 end

 macro depvar(var)
     :(SFmle.WDepvarSpec_MLE($(QuoteNode(var))))
 end

 macro frontier(vars...)
     names = [QuoteNode(v) for v in vars]
     :(SFmle.WFrontierSpec_MLE(Symbol[$(names...)]))
 end

 macro zvar(vars...)
     names = [QuoteNode(v) for v in vars]
     :(SFmle.WZvarSpec_MLE(Symbol[$(names...)]))
 end

 macro id(var)
     :(SFmle.WIdSpec_MLE($(QuoteNode(var))))
 end

 # Convenience constructors for building DSLArg_MLE from code
 # (used when @depvar/@frontier are not available as DSL macros)
 sf_depvar(name::Symbol) = WDepvarSpec_MLE(name)
 sf_frontier(names::Symbol...) = WFrontierSpec_MLE(collect(names))
 sf_idvar(name::Symbol) = WIdSpec_MLE(name)


 #? ---- macros for sf_predict -------------

 """
@eq(arg)
 
An argument in `sfmodel_predict()`. Specify the name of the function to be
predicted. 
 
See the help on `sfmodel_predict()` for more information.
 
# Examples
```julia
sfmodel_predict( @eq(frontier), ...)
```
"""
 macro eq(arg)   
    return [:($(arg))]
end 


