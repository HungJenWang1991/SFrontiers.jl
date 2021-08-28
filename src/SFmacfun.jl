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
```julia-repl
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
```julia-repl
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

"""
@depvar(arg::Vararg)
 
An argument in `sfmodel_sepc()`. Specify the dependent variable using a
column name from a DataFrame. 
 
See the help on `sfmodel_spec()` for more information.
 
# Examples
```julia-repl
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

sfmodel_spec(@depvar(yvar), ...)
```
"""
 macro depvar(arg::Vararg)
     return (:depvar, collect(:($(arg))))
 end
 


"""
depvar(arg::Vararg)
 
An argument in `sfmodel_sepc()`. Specify the maxrix name where the matrix
is used as the data of the dependent variable.
 
See the help on `sfmodel_spec()` for more information.
 
# Examples
```julia-repl
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
"""
@frontier(arg::Vararg)
 
An argument in `sfmodel_sepc()`. Specify the variables in the `frontier`
function using column names from a DataFrame. 
 
See the help on `sfmodel_spec()` for more information.
 
# Examples
```julia-repl
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

sfmodel_spec(@frontier(xvar1, xvar2, _cons), ...)
```
"""
 macro frontier(arg::Vararg)   # for sfmodel_spec()          
     return (:frontier, collect(:($(arg))))
 end 
 


"""
frontier(arg::Vararg)
 
An argument in `sfmodel_sepc()`. Specify the name of the maxrix
used as the data of the `frontier` function.
 
See the help on `sfmodel_spec()` for more information.
 
# Examples
```julia-repl
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
 


"""
frontier(arg::Vector)
 
An argument in `sfmodel_init()`. Specify the initial values for coefficients in
the `frontier` function.
 
See the help on `sfmodel_init()` for more information.
 
# Examples
```julia-repl
sfmodel_init(frontier(0.1, 0.2, 0.5), ...)
b0 = ones(3)*0.1
sfmodel_init( frontier(b0), ...)
```
"""
 function frontier(arg::Vector)    # for smodel_init(b0)
     return (:frontier,  arg)
 end
 
 
  # -------------------
 
  """
  @μ(arg::Vararg)
   
  An argument in `sfmodel_sepc()`. Specify the variables in the `μ`
  function using column names from a DataFrame. 
   
  See the help on `sfmodel_spec()` for more information.
   
  # Examples
  ```julia-repl
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
 ```julia-repl
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
 


"""
 μ(arg::Vector)
 
An argument in `sfmodel_init()`. Specify the initial values for coefficients in
the `μ` function.
 
See the help on `sfmodel_init()` for more information.
 
# Examples
```julia-repl
sfmodel_init( μ(0.1, 0.2, 0.5), ...)

b0 = ones(3)*0.1
sfmodel_init( μ(b0), ...)
```
"""
 function μ(arg::Vector)
     return (:μ, arg)
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
 
 """
 mu(arg::Vector)
  
 alias of μ. See help on μ.
 """  
 function mu(arg::Vector)
     return (:μ, arg)
 end
 
  # -------------------
 
  """
  @σᵤ²(arg::Vararg)
   
  An argument in `sfmodel_sepc()`. Specify the variables in the `σᵤ²`
  function using column names from a DataFrame. 
   
  See the help on `sfmodel_spec()` for more information.
   
  # Examples
  ```julia-repl
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
 ```julia-repl
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
 

 """
 σᵤ²(arg::Vector)
 
An argument in `sfmodel_init()`. Specify the initial values for coefficients in
the `σᵤ²` function.
 
See the help on `sfmodel_init()` for more information.
 
# Examples
```julia-repl
sfmodel_init( σᵤ²(0.1, 0.2, 0.5), ...)

b0 = ones(3)*0.1
sfmodel_init( σᵤ²(b0), ...)
```
"""
 function σᵤ²(arg::Vector)
     return (:σᵤ², arg)
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
 
 """
 sigma_u_2(arg::Vector)
  
 alias of σᵤ². See help on σᵤ².
 """ 
 function sigma_u_2(arg::Vector)
     return (:σᵤ², arg)
 end
 
 
  # -------------------
 
  """
  @σᵥ²(arg::Vararg)
   
  An argument in `sfmodel_sepc()`. Specify the variables in the `σᵥ²`
  function using column names from a DataFrame. 
   
  See the help on `sfmodel_spec()` for more information.
   
  # Examples
  ```julia-repl
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
 ```julia-repl
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
 
 """
 σᵥ²(arg::Vector)
 
An argument in `sfmodel_init()`. Specify the initial values for coefficients in
the `σᵥ²` function.
 
See the help on `sfmodel_init()` for more information.
 
# Examples
```julia-repl
sfmodel_init( σᵥ²(0.1, 0.2, 0.5), ...)

b0 = ones(3)*0.1
sfmodel_init( σᵥ²(b0), ...)
```
"""
 function σᵥ²(arg::Vector)
     return (:σᵥ², arg)
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
 
 """
 sigma_v_2(arg::Vector)
 
alias of σᵥ². See help on σᵥ².
"""  
 function sigma_v_2(arg::Vector)
     return (:σᵥ², arg)
 end
 
 # -------------------

 """
 @σₐ²(arg::Vararg)
  
 An argument in `sfmodel_sepc()`. Specify the variables in the `σₐ²`
 function using column names from a DataFrame. 
  
 See the help on `sfmodel_spec()` for more information.
  
 # Examples
 ```julia-repl
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
```julia-repl
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


"""
σₐ²(arg::Vector)
 
An argument in `sfmodel_init()`. Specify the initial values for coefficients in
the `σₐ²` function.
 
See the help on `sfmodel_init()` for more information.
 
# Examples
```julia-repl
sfmodel_init( σₐ²(0.1, 0.2, 0.5), ...)

b0 = ones(3)*0.1
sfmodel_init( σₐ²(b0), ...)
```
"""
function σₐ²(arg::Vector)
    return (:σₐ², arg)
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

"""
sigma_a_2(arg::Vector)
 
alias of σₐ². See help on σₐ².
""" 
function sigma_a_2(arg::Vector)
    return (:σₐ², arg)
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
 ```julia-repl
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
```julia-repl
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

"""
hscale(arg::Vector)
 
An argument in `sfmodel_init()`. Specify the initial values for coefficients in
the `hscale` function.
 
See the help on `sfmodel_init()` for more information.
 
# Examples
```julia-repl
sfmodel_init( hscale(0.1, 0.2, 0.5), ...)

b0 = ones(3)*0.1
sfmodel_init( hscale(b0), ...)
```
"""
function hscale(arg::Vector)
    return (:hscale, arg)
end
 


 # ---- time decay model ---------------

 """
 @gamma(arg::Vararg)
  
 An argument in `sfmodel_sepc()`. Specify the variables in the `gamma`
 function using column names from a DataFrame. 
  
 See the help on `sfmodel_spec()` for more information.
  
 # Examples
 ```julia-repl
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
```julia-repl
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

"""
gamma(arg::Vector)
 
An argument in `sfmodel_init()`. Specify the initial values for coefficients in
the `gamma` function.
 
See the help on `sfmodel_init()` for more information.
 
# Examples
```julia-repl
sfmodel_init( gamma(0.1, 0.2, 0.5), ...)

b0 = ones(3)*0.1
sfmodel_init( gamma(b0), ...)
```
"""
function gamma(arg::Vector)
    return (:gamma, arg)
end

 # ------ panel model -------------

  macro sfpanel(arg::Vararg)
     
    (length(arg)==1)||throw("@sfpanel in sfmodel_spec should have one single string.")
    
    if !(arg[1] ∈ (:TFE_WH2010, :TFE_CSW2014, :TFE_G2005, :TRE, :TimeDecay ))
         throw("The keyword of @sfpanel in sfmodel_spec is specified incorrectly.")
    end 
    
    if arg[1] == :TFE_G2005
        throw("The TEF_G2005 panel model is essentially a cross-sectional model with dummies of individuals. Please use non-panel specifications and add individual dummies to @frontier().")
    end 

    return (:panel, collect(:($(arg))))
 end 


 function sfpanel(arg::Vararg)
     
    (length(arg)==1) || throw("`sfpanel()` in sfmodel_spec should have one single string.")
    
    if !(arg[1] ∈ (TFE_WH2010, TFE_CSW2014, TFE_G2005, TRE, TimeDecay ))
         throw("The keyword of `sfpanel` in sfmodel_spec is specified incorrectly.")
    end 
    
    if arg[1] == TFE_G2005
        throw("The TEF_G2005 panel model is essentially a cross-sectional model with dummies of individuals. Please use non-panel specifications and add individual dummies to @frontier().")
    end 

    return (:panel, [Symbol(arg[1])])
 end


 """
 @timevar(arg::Vararg)
  
 An argument in `sfmodel_sepc()` for panel data models. Specify the column
 name of a DataFrame that contain the time information of the panel data. 
 
 See the help on `sfmodel_spec()` for more information.
  
 # Examples
 ```julia-repl
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
 
 sfmodel_spec( @timvar(year), ...)
 ```
 """   
macro timevar(arg::Vararg)
    return (:timevar, collect(:($(arg))))
end



"""
timevar(arg::Vararg)
 
An argument in `sfmodel_sepc()` for panel data models. Specify the name of the
matrix containing the time information of the panel data. 
 
See the help on `sfmodel_spec()` for more information.
 
# Examples
```julia-repl
julia> timemat
100-element Vector{Int64}:
 2019
 2020
 2021
 2022
 2019
 2020
 2021
    ⋮
 2021
 2022
 2019
 2020
 2021
 2022

sfmodel_spec( timvar(timemat), ...)
```
"""   
function timevar(arg::Vararg)
    return (:timevar, collect(:($(arg))))
end


"""
@idvar(arg::Vararg)
 
An argument in `sfmodel_sepc()` for panel data models. Specify the column
name of a DataFrame that contain the individual's id information of the panel data. 

See the help on `sfmodel_spec()` for more information.
 
# Examples
```julia-repl
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
```julia-repl
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




function misc(arg::Any)
    return (:misc, arg)
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
 
 
 function all_init(arg::Vararg)
     return (:all_init, collect(:($(arg))))
 end
 

"""
all_init(arg::Vector)
 
An argument in `sfmodel_init()`. Specify the initial values for all the
parameters in the model.
 
See the help on `sfmodel_init()` for more information.
 
# Examples
```julia-repl
sfmodel_init( all_init(0.1, 0.2, 0.5, -0.1, -0.1, -0.2), ...)

b0 = ones(6)*0.1
sfmodel_init( all_init(b0), ...)
```
"""
 function all_init(arg::Vector)
     return (:all_init, arg)
 end
 
 #? ----  functions for sfmodel_opt --------
 
 
 #= macro warmstart_solver(arg=nothing)
     return (:warmstart_solver, arg)
 end =#
 


"""
warmstart_solver(arg)
 
An argument in `sfmodel_opt()`. Specify the algorithm used in the first-stage ("warmstart")
  optimization process.

The default is `NelderMead()`. Others include `SimulatedAnnealing()`, `SAMIN()`, `ParticleSwarm()`,
  `ConjugateGradient()`, `GradientDescent()`, `BFGS()`, `LBFGS()`,
  `Newton()`, `NewtonTrustRegion()`, and `IPNewton()`. See
  http://julianlsolvers.github.io/Optim.jl/stable/ for details.
  Non-gradient based algorithms are recommended for the warmstart solver. 

See the help on `sfmodel_opt()` for more information.
 
# Examples
```julia-repl
sfmodel_opt( warmstart_solver(NelderMead()), ...)
```
"""
 function warmstart_solver(arg=nothing)
     return (:warmstart_solver, arg)
 end
 
 
  # -------------------
 
 #= macro warmstart_maxIT(arg=nothing)
     return (:warmstart_maxIT, arg)
 end =#
 
 """
warmstart_maxIT(arg)
 
An argument in `sfmodel_opt()`. Specify the iteration limit for the warmstart. Default
is 100.

See the help on `sfmodel_opt()` for more information.
 
# Examples
```julia-repl
sfmodel_opt( warmstart_maxIT(400), ...)
```
"""
 function warmstart_maxIT(arg=nothing)
     return (:warmstart_maxIT, arg)
 end
 
 
  # -------------------
 
 #= macro main_solver(arg=nothing)
     return (:main_solver, arg)
 end =#
 
 
 """
main_solver(arg)
 
An argument in `sfmodel_opt()`. Specify the algorithm used in the 2nd-stage ("main")
  optimization process.

  The default is `Newton()`. Others include `SimulatedAnnealing()`, `SAMIN()`, `ParticleSwarm()`,
  `ConjugateGradient()`, `GradientDescent()`, `BFGS()`, `LBFGS()`,
  `NewtonTrustRegion()`, and `IPNewton()`. See
  http://julianlsolvers.github.io/Optim.jl/stable/ for details.


See the help on `sfmodel_opt()` for more information.
 
# Examples
```julia-repl
sfmodel_opt( main_solver(Newton()), ...)
```
"""
 function main_solver(arg=nothing)
     return (:main_solver, arg)
 end
 
 
  # -------------------
 
 #= macro main_maxIT(arg=nothing)
     return (:main_maxIT, arg)
 end =#
 


 """
main_maxIT(arg)
  
 An argument in `sfmodel_opt()`. Specify the iteration limit for the main solver. Default
 is 2000.
 
 See the help on `sfmodel_opt()` for more information.
  
 # Examples
 ```julia-repl
 sfmodel_opt( main_maxIT(2500), ...)
 ```
 """
 function main_maxIT(arg::Any=nothing)
     return (:main_maxIT, arg)
 end
 
 
  # -------------------
 
 #= macro tolerance(arg=nothing)
     return (:tolerance, arg)
 end =#
 

 """
tolerance(arg::Float64)
 
An argument in `sfmodel_opt()`. Specify the convergence criterion ("tolerance") based on the
absolute value of gradients. Default is 1.0e-8. For non-gradient algorithms,
it controls the main convergence tolerance, which is solver specific. 
See `Optim`'s `g_tol` option for more information.

Also see the help on `sfmodel_opt()` for more information.
 
# Examples
```julia-repl
sfmodel_opt( tolerance(1.0e-6), ...)
```
"""
 function tolerance(arg::Float64=nothing)
     return (:tolerance, arg)
 end
 
   # --------------------------
 """
verbose(arg::Bool)
 
An argument in `sfmodel_opt()`. Specify whether to print on screen (`true`,
the default) the information of the model and the optimization results.

See the help on `sfmodel_opt()` for more information.
 
# Examples
```julia-repl
sfmodel_opt( verbose(false), ...)
```
""" 
 function verbose(arg::Bool=false)
     return (:verbose, arg)
 end
 

 """
 banner(arg::Bool)
  
 An argument in `sfmodel_opt()`. Specify whether to print on screen (`true`,
 the default) a banner to serve as a visual indicator of the start of the 
 estimation.
 
 See the help on `sfmodel_opt()` for more information.
  
 # Examples
 ```julia-repl
 sfmodel_opt( banner(false), ...)
 ```
 """ 
 function banner(arg::Bool=true)
    return (:banner, arg)
 end
 

 """
 ineff_index(arg::Bool)
  
 An argument in `sfmodel_opt()`. Specify whether (`true`, the default) to compute the Jondrow et al. (1982)
 inefficiency index and the Battese and Coelli (1988) efficiency index.
 
 See the help on `sfmodel_opt()` for more information.
  
 # Examples
 ```julia-repl
 sfmodel_opt( ineff_index(false), ...)
 ```
 """ 
 function ineff_index(arg::Bool=true)
    return (:ineff_index, arg)
 end



"""
marginal(arg::Bool)
  
 An argument in `sfmodel_opt()`. Specify whether (`true`, the default) to
 compute the marginal effects of the exogenous determinants of inefficiency (if any).
 
 See the help on `sfmodel_opt()` for more information.
  
 # Examples
 ```julia-repl
 sfmodel_opt( marginal(false), ...)
 ```
""" 
 function marginal(arg::Bool=true)
    return (:marginal, arg)
 end

  # -------------------------


  """
  table_format(arg)
    
   An argument in `sfmodel_opt()`. Specify the format to print the coefficient
   tables on the screen: `text` (default), `html`, or `latex`. A wrapper of `PrettyTables.jl`'s
   `backend` option.

   See the help on `sfmodel_opt()` for more information.
    
   # Examples
   ```julia-repl
   sfmodel_opt( table_format(html), ...)
   ```
  """ 
 function table_format(arg=nothing)
    
    if !(arg ∈ (text, html, latex ))
        throw("The keyword of `table_format` in `sfmodel_opt()` is specified incorrectly. Only allow `text`, `html`, or `latex`. Got `$(arg)` instead.")
   end 

    return (:table_format, Symbol(arg))
end


 
  #? ------ functions for sfmodel_fit ----------
 
 # macro dataframe(arg)  
 #     return arg
 # end
 
 """
 useData(D::DataFrame)
   
  An argument in `sfmodel_fit()`. Specify the name of the DataFrame that
  contains the estimation data.

  See the help on `sfmodel_fit()` for more information.
   
# Examples
```julia-repl
  julia> mydf
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

  sfmodel_fit(useData(mydf), ...)
```
""" 
 function useData(D::DataFrame)  # doesn't work using macro (perhaps because of data), so...
     return D
 end
 
 #? ---- macros for sf_predict -------------

 """
@eq(arg)
 
An argument in `sfmodel_predict()`. Specify the name of the function to be
predicted. 
 
See the help on `sfmodel_predict()` for more information.
 
# Examples
```julia-repl
sfmodel_predict( @eq(frontier), ...)
```
"""
 macro eq(arg)   
    return [:($(arg))]
end 


