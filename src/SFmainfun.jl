########################################################
####                sfmodel_spec()                  ####
########################################################

"""
sfmodel_spec(<keyword arguments>)

Provide specifications of the stochastic frontier model, including the type of model
and names of variables or matrix used in estimating the model. Two ways to
specify: Method 1: Use DataFrame as input, and Method 2: use matrix as input.

# Method 1 (DataFrame input)
Variables come from a DataFrame, and column names of the Dataframe are used in
the variable input. With this method,
equations are identified by macros but not functions (e.g., `@depvar()` but
not `depvar()`).

## Arguments of Method 1
- `sfdist(::Vararg)`: the distribution assumption of the one-sided stochastic
  variable (aka inefficiency term) of the model;
  possible choices include `truncated` (or `trun`, `t`), `half` (or `h`),
  `exponential` (or `expo`, `e`), and `trun_scale` (or `trun_scaling`, `ts`).
- `sftype(::Vararg)`: whether the model is a `production` (or `prod`) frontier
  or a `cost` frontier.
- `sfpanel(::Vararg)`: the type of panel model. Choices include `TFE_WH2010`
  (true fixed effect model of Wang and Ho 2010 JE), `TFE_CSW2014` (true fixed
  model of Chen, Schmidt, and Wang 2014 JE),  `TRE` (true random effect model
  of Greene 2004), `TimeDecay` (time decay model of Battese and Coelli 1992).
- `@depvar(::Vararg)`: the dependent variable from a DataFrame.
- `@frontier(::Vararg)`: a list of variables, separated by commas, in the frontier function.
- `@μ(::Vararg)` or `@mu(::Vararg)`: a list of variable, separated by comma,
  in the linear function of μ. (`sftype(trun)` only).
- `@σᵥ²(::Vararg)` or `@sigma_v_2(::Vararg)`: a list of variable, separated by comma, in the σᵥ²
  equation.
- `@σᵤ²(::Vararg)` or `@sigma_u_2(::Vararg)`: a list of variable, separated by comma, in the σᵤ²
  equation.
- `@σₐ²(::Vararg)` or `@sigma_a_2(::Vararg)`: a list of variable, separated by comma, in the σₐ²
  equation. `sfpanel(TRE)` only.
- `@gamma(::Vararg)`: a list of variables, separated by commas, in the gamma
  equation. `sfpanel(TimeDecay)` only.
- `@timevar(::Vararg)`: the variable containing the time period information.
  Panel data model only.
- `@idvar(::Vararg)`: the variable identifying each individual. Panel data
  model only.
- message::Bool: Whether printing (=true) or not (=false, the default) the
  confirmation message "A dictionary from sfmodel_spec() is generated."
  on the screen after `sfmodel_spec()` is successfully executed.  


# Method 2 (matrix/vector input)
Data of the variables are
provided by individual matrices or vectors, and names of the mat/vec are used
in the equations. With this method, equations are identified by functions but
not macros (e.g., `depvar()` but not `@depvar()`). Note that if, for instance,
the name of `depvar` or `σᵤ²` has been used elsewhere in the program, using
these names to read in mat/vec will cause name conflict (`MethodError: objects
of type ... are not callable`). The workaround is to fully qualify the function
names, e.g., `SFrontiers.depvar`, `SFrontiers.σᵤ²`, etc. Or, use the alias (if
available), e.g., `sigma_u_2` instead of `σᵤ²`.

## Arguments of Method 2
- `sfdist(::Vararg)`: the distribution assumption on the inefficiency term;
  possible choices include `truncated` (or `trun`, `t`), `half` (or `h`),
  `exponential` (or `expo`, `e`), and `trun_scale` (or `trun_scaling`, `ts`).
- `sftype(::Vararg)`: whether the model is a `production` (or `prod`) frontier
  or a `cost` frontier.
- `sfpanel(::Vararg)`: the type of panel model. Choices include `TFE_WH2010`
  (true fixed effect model of Wang and Ho 2010 JE), `TFE_CSW2014` (true fixed
  model of Chen, Schmidt, and Wang 2014 JE),  `TRE` (true random effect model
  of Greene 2004), `TimeDecay` (time decay model of Battese and Coelli 1992).
- `depvar(::Matrix)`: Matrix or vector of the dependent variable.
- `frontier(::Matrix)`: matrix or vector for frontier function.
- `μ(::Matrix)` or `mu(::Matrix)`: matrix or vector for the (linear) μ equation (`trun` type only).
- `σᵤ²(::Matrix)` or `sigma_u_2(::Matrix)`: matrix or vector for the σᵤ² equation.
- `σₐ²(::Matrix)` or `sigma_a_2(::Matrix)`: matrix or vector for the σₐ² equation.
- message::Bool: Whether printing (=true) or not (=false, the default) the
  confirmation message "A dictionary from sfmodel_spec() is generated."
  on the screen after `sfmodel_spec()` is successfully executed.

# Examples
```julia-repl
sfmodel_spec(sftype(prod), sfdist(trun),
             @depvar(output), 
             @frontier(land, , labor, bull, year, _cons), 
             @μ(age, school, year, _cons),
             @σᵤ²(age, school, year, _cons),
             @σᵥ²(_cons));

sfmodel_spec(sfpanel(TRE), sftype(prod), sfdist(half),
             @timevar(yr), @idvar(id),
             @depvar(y), 
             @frontier(x1, x2, _cons), 
             @σₐ²(_cons),
             @σᵤ²(_cons),
             @σᵥ²(_cons),
             message = false);
```
"""
function sfmodel_spec(arg::Vararg; message::Bool=false) 

    global _dicM 
           _dicM = Dict{Symbol, Any}()  # nullify and initiate new dictionaries when a new model is specified

         #* -- creates default values ---

        for k in (:panel, :timevar, :idvar, :dist, :type, :depvar, :frontier, :μ, :hscale, :gamma, :σᵤ², :σᵥ², :σₐ², :hasDF, :transfer, :misc) # take out :data, :η, :λ, :τ, in this revision
            _dicM[k] = nothing
        end
     
        #* -- replace the defaults with user's values ---          

        for d in :($(arg))
            _dicM[d[1]] = d[2]
        end 

        #* ==== Method 2, matrix input (such as in simulations), create a DataFrame

           _dicM[:hasDF]    = true
           _dicM[:transfer] = false


        if typeof(_dicM[:depvar]) != Array{Symbol,1} # not a DataFrame

           _dicM[:hasDF] = false 

           isa(_dicM[:depvar][1], Vector) || isa(_dicM[:depvar][1], Matrix) || throw("
           `depvar()` has to be a Vector or Matrix (e.g., Array{Float64, 1} or Array{Float64, 2}). 
           Check with `isa(your_thing, Matrix)` or `isa(your_thing, Vector)`. 
           Try `convert()`, `reshape()`, `Matrix()`, or something similar.")

           comDF = _dicM[:depvar][1]  # create the first data column of comDF
           varname = [:depvar]
           _dicM[:depvar] = [:depvar]

           for k in (:timevar, :idvar, :frontier, :μ, :hscale, :gamma, :σₐ², :σᵤ², :σᵥ²) 
               if _dicM[k] !== nothing # if not nothing, must be Array
                  isa(_dicM[k], Vector) || isa(_dicM[k][1], Vector) || isa(_dicM[k][1], Matrix) || throw("
                     `k` has to be a Vector or Matrix (e.g., Array{Float64, 1} or Array{Float64, 2}). 
                     Check with `isa(your_thing, Matrix)` or `isa(your_thing, Vector)`. 
                     To convert, try `convert()`, `reshape()`, `Matrix()`, or something similar.")
   
                  (isa(_dicM[k], Vector)  && length(_dicM[k][1]) == 1) ?  _dicM[k] = [_dicM[k]] : nothing # ugly fix for pure vector input
   
                  @views comDF = hcat(comDF, _dicM[k][1]) # combine the data
                  aa = Symbol[]
                  for i in axes(_dicM[k][1], 2)
                      push!(aa, Symbol(String(k)*"_var$(i)")) # create name for the data
                  end                  
                  varname = vcat(varname, aa) # combine the dataname
                  _dicM[k] = aa
               end 
           end # for k in (...)


           comDF = DataFrame(comDF, varname)
           _dicM[:sdf] = comDF

        end # if typeof(...)

     
        #* -- check the model identifier and the model type ---

        (_dicM[:dist] !== nothing) || throw("You need to specify dist().")
        (_dicM[:type] !== nothing) || throw("You need to specify type().")

        #* --- get the model identifier -------

        s = uppercase(String(_dicM[:dist][1])[1:1])

        global tagD
        if _dicM[:panel] === nothing  # not panel 
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
        elseif (_dicM[:panel] == [:TFE_WH2010]) && (s == "H")  # panel and dist=half
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

        
        #* ---- check if the model has the correct syntax ---

        SFrontiers.checksyn(tagD[:modelid])

        #* ----- make return ----------- 
        if message 
          printstyled("A dictionary from sfmodel_spec() is generated.\n"; color = :green)  
        end  
        return _dicM # for debugging purpose

end  # end of sfmodel_spec()



########################################################
####                sfmodel_init()                  ####
########################################################
"""
    sfmodel_init(<keyword arguments>)

Provide initial values for the stochastic frontier model estimation. The
values could be a vector or scalars. It creates a global dictionary `_dicINI`. Optional.

# Arguments
- `all_init(::Union{Vector, Real})`: initial values of all the parameters in the model
- `frontier(::Union{Vector, Real})`: initial values of parameters in
  the `frontier()` function
- `μ(::Union{Vector, Real})` or `mu(::Union{Vector, Real})`: initial values of
  parameters in the `μ` function
- `hscale(::Union{Vector, Real})`: initial values of parameters in the `hscale()` function
- `gamma(::Union{Vector, Real})`: initial values of parameters in the `gamma()` function
- `σᵤ²(::Union{Vector, Real})` or `sigma_u_2(::Union{Vector, Real})`: initial values of parameters in the
   `σᵤ²` function
- `σᵥ²(::Union{Vector, Real})` or `sigma_v_2(::Union{Vector, Real})`: initial values of parameters in the
   `σᵥ²` function    
- `σₐ²(::Union{Vector, Real})` or `sigma_a_2(::Union{Vector, Real})`: initial
  values of parameters in the `σₐ²` function
- message::Bool: Whether printing (=true) or not (=false, the default) the
  confirmation message "A dictionary from sfmodel_init() is generated."
  on the screen after `sfmodel_init()` is successfully executed.

# Remarks
- Equations do not have to follow specific orders.
- `sfmodel_init(...)` is optional but is highly recommended. If it is not
  specified or is specified as an empty set, default values are used.
- It is not necessary to specify a complete set of equations. A partial list 
  or even empty lists are acceptable. Default values will be substituted for the
  missing equations.
- The generated `_dicINI` is inheritable in the sense that an exiting
  `_dicINI` (from the previous run of the same or a different model, for
  example) will be used if the current model does not have its own
  `sfmodel_init(...)`. This design has advantages in a simulations study where
  `sfmodel_init(...)` needs to be specified only once.

# Examples
```julia-repl
b_ini = ones(2)*0.2
sfmodel_init( # frontier(bb),             # may skip and use default
             μ(b_ini),                    # may use a vector
             σᵤ²(-0.1, -0.1),  
             σᵥ²(-0.1) )                   

sfmodel_init(all_init(0.1, 0.2, 0.5, 0.0, -0.1, -0.1, -0.1),
             message = false)             
```
"""
function sfmodel_init(arg::Vararg; message::Bool =false) # create a dictionary of inital vectors

   global _dicINI
          _dicINI = Dict{Symbol, Any}()

    for d in :($(arg))
        _dicINI[d[1]] = d[2]
    end        

    #* If has the key, creates the alias key with the same value.
    
    !(haskey(_dicINI, :μ))      || (_dicINI[:eqz] = _dicINI[:μ])
    !(haskey(_dicINI, :hscale)) || (_dicINI[:eqq] = _dicINI[:hscale]) 
    !(haskey(_dicINI, :gamma))  || (_dicINI[:eqq] = _dicINI[:gamma]) 
    !(haskey(_dicINI, :σₐ²))    || (_dicINI[:eqw] = _dicINI[:σₐ²])
    !(haskey(_dicINI, :σᵤ²))    || (_dicINI[:eqw] = _dicINI[:σᵤ²])
    !(haskey(_dicINI, :σᵥ²))    || (_dicINI[:eqv] = _dicINI[:σᵥ²])

    if message 
      printstyled("A dictionary from sfmodel_init() is generated.\n"; color = :green) 
    end
    return _dicINI # for debugging purpose
end    

########################################################
####                sfmodel_opt()                   ####
########################################################
"""
    sfmodel_opt(<keyword arguments>)

Provide options to the optimization algorithms for the maiximum likelihood
estimation. It creates a global dictionary `_dicOPT`. Optional. The `Optim`
package is used for the optimization, and a subset of
`Optim`'s keywords are directly accessible from this API. 

# Arguments
- `warmstart_solver(algorithm)`: The algorithm used in the first-stage ("warmstart")
  optimization process, which serves the purpose of improving upon the initial
  values for the second-stage ("main") estimation. The default is
  `NelderMead()`. Others include `SimulatedAnnealing()`, `SAMIN()`, `ParticleSwarm()`,
  `ConjugateGradient()`, `GradientDescent()`, `BFGS()`, `LBFGS()`,
  `Newton()`, `NewtonTrustRegion()`, and `IPNewton()`. See
  http://julianlsolvers.github.io/Optim.jl/stable/ for details.
  Non-gradient based algorithms are recommended for the warmstart solver. 
- `warmstart_maxIT(::Int64)`: The iteration limit for the warmstart. Default
  is 100.
- `main_solver(algorithm)`: The algorithm used in the main opimization process.
  The default is `Newton()`. Others include `SimulatedAnnealing()`, `SAMIN()`, `ParticleSwarm()`,
  `ConjugateGradient()`, `GradientDescent()`, `BFGS()`, `LBFGS()`,
  `NewtonTrustRegion()`, and `IPNewton()`. See
  http://julianlsolvers.github.io/Optim.jl/stable/ for details.
- `main_maxIT(::Int64)`: The iteration limit for the main estimation. Default
  is 2000.
- `tolerance(::Float64)`: The convergence criterion ("tolerance") based on the
  absolute value of gradients. Default is 1.0e-8. For non-gradient algorithms,
  it controls the main convergence tolerance, which is solver specific. 
  See `Optim`'s `g_tol` option for more information.
- `verbose(::Bool)`: Print on screen (`true`, the default) the information of
  the model and the optimization results.
- `banner(::Bool)`: Print on screen (`true`, the default) a banner to serve as
  a visual indicator of the start of the estimation.
- `ineff_index(::Bool)`: Whether to compute the Jondrow et al. (1982)
  inefficiency index and the Battese and Coelli (1988) efficiency index. The
  defauis `true`.
- `marginal(::Bool)`: Whether to compute the marginal effects of the exogenous
  determinants of inefficiency (if any).
- `table_format()`: The format to print the coefficient tables on the screen:
  `text` (default), `html`, or `latex`. A wrapper of `PrettyTables.jl`'s
  `backend` option.
- message::Bool: Whether printing (=true) or not (=false, the default) the
  confirmation message "A dictionary from sfmodel_opt() is generated."
  on the screen after `sfmodel_opt()` is successfully executed.


# Remarks
- `sfmodel_opt(...)` is optional. It can be omitted entirely, or specifying
  only a partial list of the keywords.
- If any of the keywords are missing, default values are used.
- If warmstart is not needed, you need to give empty keyword values to
  warmstart related keys. E.g., either `warmstart_solver()` or
  `warmstart_maxIT()`, or both. Omitting the keyword entirely (i.e., not
  writing down `warmstart_solver` or `warmstart_maxIT`) will not skip the
  warmstart, but will reinstate the default. 
- Users do not need to provide gradient or Hessian functions even if 
  gradient-based optimization algorithms are used. The package uses automatic
  differentiation (https://en.wikipedia.org/wiki/Automatic_differentiation) to 
  compute the derivatives. It is not numerical finite differentiation. It is
  fast and as accurate as the symbolic differentiation.
- The `_dicOPT` is inheritable in the sense that an exiting `_dicOPT` (from
  the previous run of the same or a different model, for example) will be used
  if the current model does not have its own `sfmodel_opt(...)`. This design
  has advantages in simulation studies where `sfmodel_opt(...)` needs to be
  specified only once.

# Examples
```julia-repl
sfmodel_opt(warmstart_solver(NelderMead()),   
            warmstart_maxIT(200),
            main_solver(Newton()), 
            main_maxIT(2000), 
            tolerance(1e-8),
            message = false)
```
"""
function sfmodel_opt(arg::Vararg; message::Bool=false) # create a dictionary of maximization options

    global _dicOPT
           _dicOPT = Dict{Symbol, Any}()

    #* -- creates the default ---

    _dicOPT[:warmstart_solver] = :(NelderMead())
    _dicOPT[:warmstart_maxIT]  =  100
    _dicOPT[:main_solver]      = :(Newton())
    _dicOPT[:main_maxIT]       =  2000
    _dicOPT[:tolerance]        =  1.0e-8
    _dicOPT[:verbose]          =  true
    _dicOPT[:banner]           =  true
    _dicOPT[:ineff_index]      =  true
    _dicOPT[:marginal]         =  true
    _dicOPT[:table_format]     = :(text)

    #* -- replace the defaults with the user's value ---

    for d in :($(arg))
        _dicOPT[d[1]] = d[2]
    end    
    
    #* ---- error checking --

    if (_dicOPT[:main_solver] === nothing) || (_dicOPT[:main_maxIT] === nothing) || (_dicOPT[:tolerance] === nothing)
         throw("You cannot give empty keyword values to `main_solver()`, `main_maxIT()`, or `tolerance()`. If you want to use the default, you may do so by dropping (not emptying keyword values) the keywords.")
    end
    
    if message 
      printstyled("A dictionary from sfmodel_opt() is generated.\n"; color = :green)  
    end  
    return _dicOPT # for debugging purpose

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
Return a dictionary with results.

# Arguments
- `useData(::DataFrame)`: The DataFrame used with the Method 1 of
  `sfmodel_spec(...)`. If use Method 2 of `sfmodel_spec(...)` (viz., data
  is supplied by individual matrices), do not need this keyword argument.

# Remarks
- Use `Optim.jl` to carry out the estimation.
- Users do not need to provide gradient or Hessian functions even if 
  gradient-based optimization algorithms are used. The package uses automatic
  differentiation (https://en.wikipedia.org/wiki/Automatic_differentiation) to 
  compute the derivatives. AD is not numerical finite differentiation. AD is
  fast and as accurate as the symbolic differentiation.

# Examples
```julia-repl
sfmodel_fit(useData(df))    # Method 1
sfmodel_fit()               # Method 2
```
"""
function sfmodel_fit()
     # For Method 2 of `sfmodel_spec()`.
   
    !(_dicM[:hasDF]) || throw("Need to specify DataFrame in `sfmodel_fit()`.")
    
    _dicM[:transfer] = true
    sfmodel_fit(_dicM[:sdf])
   
end    



function sfmodel_fit(sfdat::DataFrame) #, D1::Dict = _dicM, D2::Dict = _dicINI, D3::Dict = _dicOPT)

    (_dicM[:hasDF] || _dicM[:transfer])  || throw("You provided matrix in `sfmodel_spec()` so you cannot specify a DataFrame in `sfmodel_fit()`. Leave it blank.")

   #* for simulation, add a flag
   redflag::Bool = 0

  #* ###### Check if the OPT dictionary exists #####

      @isdefined(_dicINI) || sfmodel_init()  # if not exist, create one with default values
      @isdefined(_dicOPT) || sfmodel_opt()  
      

     if _dicOPT[:banner] 
        printstyled("\n###------------------------------------###\n"; color=:yellow)
        printstyled("###  Estimating SF models using Julia  ###\n"; color=:yellow)
        printstyled("###------------------------------------###\n\n"; color=:yellow)
      end  


  #* ##### Get variables from dataset #######
    
     # pos: (begx, endx, begz, endz, ...); variables' positions in the parameter vector.
     # num: (nofobs, nofx, ..., nofpara); number of variables in each equation
     # eqvec: ("frontier"=2, "μ"=6,...); named tuple of equation names and equation position in the table
     # eqvec2: (xeq=(1,3), zeq=(4,5),...); named tuple of equation and parameter positions, for sfmodel_predict
     # varlist: ("x1", "x2",...); variable names for making table

     (minfo1, minfo2, pos, num, eqvec, eqvec2, yvar, xvar, zvar, qvar, wvar, 
      vvar,         rowIDT, varlist) = SFrontiers.getvar(tagD[:modelid], sfdat)

  #* ### print preliminary information ########

    if _dicOPT[:verbose] 

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

     β0     = xvar \ yvar;  # OLS estiamte, uses a pivoted QR factorization;
     resid  = yvar - xvar*β0
     sse    = sum((resid).^2)  
     ssd    = sqrt(sse/(size(resid,1)-(num.nofx + noffixed ))) # sample standard deviation; σ² = (1/(N-K))* Σ ϵ^2
     ll_ols = sum(normlogpdf.(0, ssd, resid)) # ols log-likelihood
     sk_ols = sum((resid).^3) / ((ssd^3)*(size(resid,1))) # skewnewss of ols residuals

     #* --- Create the dictionary -----------

     if (:all_init in keys(_dicINI))
         sf_init = _dicINI[:all_init]
     else
         #*  Create ini vectors from user's values; if none, use the default.--- #      
         b_ini  = get(_dicINI, :frontier, β0)
         d1_ini = get(_dicINI, :eqz, ones(num.nofz) * 0.1)
         t_ini  = get(_dicINI, :eqq, ones(num.nofq) * 0.1)
         d2_ini = get(_dicINI, :eqw, ones(num.nofw) * 0.1)
         g_ini  = get(_dicINI, :eqv, ones(num.nofv) * 0.1)

         #*  Make it Array{Float64,1}; otherwise Array{Float64,2}. ---#     
         #*       Could also use sf_init[:,1]. *#
         sf_init = vcat(b_ini, d1_ini, t_ini, d2_ini, g_ini)  
         sf_init = vec(sf_init)   
     end # if :all_init


  #* ############ Misc.  ################     
     # --- check if the number of initial values is correct 
        (length(sf_init) == num.nofpara) ||  throw("The number of initial values does not match the number of parameters to be estimated. Make sure the number of init values in sfmodel_init() matches the number of variabls in sfmodel_spec().") 

     # --- Make sure there is no numerical issue arising from int vs. Float64.
        sf_init = convert(Array{Float64,1}, sf_init) 

  #* ############# process optimization dictionary  #######

         if (_dicOPT[:warmstart_solver] === nothing) || (_dicOPT[:warmstart_maxIT] === nothing)
             do_warmstart_search = 0
         else 
             do_warmstart_search = 1
             sf_ini_algo  = eval(_dicOPT[:warmstart_solver])  # warmstart search algorithms
             sf_ini_maxit = _dicOPT[:warmstart_maxIT]         # warmstart search iter limit
         end    
             
     # ---- main maximization algorithm -----
         sf_algo  = eval(_dicOPT[:main_solver])    # main algorithm
         sf_maxit = _dicOPT[:main_maxIT] 
         sf_tol   = _dicOPT[:tolerance] 
         sf_table = _dicOPT[:table_format]

  #* ########  Start the Estimation  ##########

    #* ----- Define the problem's Hessian -----#


     _myfun = TwiceDifferentiable(rho -> SFrontiers.LL_T(tagD[:modelid], 
                           yvar, xvar, zvar, qvar, wvar, vvar, 
                           _porc, num.nofobs, pos, rho,
                                   rowIDT, _dicM[:misc]),
                     sf_init;               
                    autodiff = :forward); 


    #* ---- Make placeholders for dictionary recording purposes *#

    sf_init_1st_dic  = 0
    sf_init_2nd_dic  = 0
    sf_ini_algo_dic  = nothing
    sf_ini_maxit_dic = 0
    sf_total_iter    = 0

    _run = 1  # a counter; use the -if- instead of -for- to avoid using global variables

    if (do_warmstart_search == 1) && (_run == 1)  
 
        if _dicOPT[:verbose] 
            printstyled("The warmstart run...\n\n"; color = :green)
        end
 
        sf_init_1st_dic  = copy(sf_init) # for dict recording
        sf_ini_algo_dic  = sf_ini_algo
        sf_ini_maxit_dic = copy(sf_ini_maxit)

        # @time  
               _optres = optimize(_myfun, 
                               sf_init,         # initial values  
                               sf_ini_algo,                   
                               Optim.Options(g_tol = sf_tol,
                                             iterations  = sf_ini_maxit, 
                                             store_trace = true,
                                             show_trace  = false))


        sf_total_iter += Optim.iterations(_optres) # for later use

        sf_init = Optim.minimizer(_optres)  # save as initials for the next run
        _run    = 2                      # modify the flag

        if _dicOPT[:verbose] 
            println()
            print("$_optres \n")
            print("The warmstart results are:\n"); printstyled(Optim.minimizer(_optres); color=:yellow); println("\n")
        end

   end  # if  (do_warmstart_search == 1) && (_run == 1)  

   if (do_warmstart_search == 0 ) || (_run == 2) # either no warmstart run and go straight here, or the 2nd run

       sf_init_2nd_dic = copy(sf_init) # for dict recording 

       if _dicOPT[:verbose] 
           println()
           printstyled("Starting the optimization run...\n\n" ; color = :green)
       end 
       
       # @time 
              _optres = optimize(_myfun, 
                              sf_init,       # initial values  
                              sf_algo,       # different from search run
                              Optim.Options(g_tol = sf_tol,
                                            iterations  = sf_maxit, # different from search run
                                            store_trace = true,
                                            show_trace  = false))
       sf_total_iter += Optim.iterations(_optres)

       if _dicOPT[:verbose] 
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
      numerical_hessian  = hessian!(_myfun, _coevec)  # Hessain

     #* ------ Check if the matrix is invertible. ----

     var_cov_matrix = try
                         inv(numerical_hessian)
                      catch err 
                         redflag = 1
                         checkCollinear(tagD[:modelid], xvar, zvar, qvar, wvar, vvar) # check if it is b/c of multi-collinearity in the data         
                         throw("The Hessian matrix is not invertible, indicating the model does not converge properly. The estimation is abort.")
                      end 
                      
          #* In some cases the matrix is invertible but the resulting diagonal
          #*    elements are negative. Check.

          if !all( diag(var_cov_matrix) .> 0 ) # not all are positive
               redflag = 1
               printstyled("Some of the diagonal elements of the var-cov matrix are non-positive, indicating problems in the convergence. The estimation is abort.\n\n"; color = :red)
               checkCollinear(tagD[:modelid], xvar, zvar, qvar, wvar, vvar) # check if it is b/c of multi-collinearity in the data
          end              

     #* ------- JLMS and BC index -------------------

     if _dicOPT[:ineff_index] 
        @views (_jlms, _bc) = jlmsbc(tagD[:modelid], _porc, pos, _coevec, 
                                     yvar, xvar, zvar, qvar, wvar, vvar,         rowIDT)
        _jlmsM = mean(_jlms) # sum(_jlms)/length(_jlms)
        _bcM   = mean(_bc)   # sum(_bc)/length(_bc)
     else
        _jlms  = nothing
        _bc    = nothing
        _jlmsM = nothing
        _bcM   = nothing
     end 


     #* ---- marginal effect on E(u) -------------- 
 
     if _dicOPT[:marginal] 
        margeff, margMinfo = get_marg(tagD[:modelid], pos, num, _coevec, zvar, qvar, wvar)
     else
        margeff, margMinfo = nothing, ()
     end                         


     #* ------- Make Table ------------------

     stddev  = sqrt.(diag(var_cov_matrix)) # standard error
     t_stats = _coevec ./ stddev          # t statistics
     p_value = zeros(num.nofpara)   # p values
     ci_low  = zeros(num.nofpara) # confidence interval
     ci_upp  = zeros(num.nofpara) 
     tt      = cquantile(Normal(0,1), 0.025)

     for i = 1:num.nofpara 
         @views p_value[i,1] = pvalue(TDist(num.nofobs - num.nofpara), t_stats[i,1]; tail=:both)
         @views ci_low[i,1] = _coevec[i,1] - tt*stddev[i,1]
         @views ci_upp[i,1] = _coevec[i,1] + tt*stddev[i,1]
     end  

       #* Build the table columns *#

    table = zeros(num.nofpara, 7)  # 7 columns in the table
    table[:,2] = _coevec   # estiamted coefficients
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

    table = hcat(varlist, table)                      # combine the variable names column (hcat, horizontal concatenate; see also vcat)
    table[:,1], table[:,2] = table[:,2], table[:,1]   # swap the first name column and the function column
    table[1,2] = "Var."

     # * ------ Print Results ----------- *#

     if _dicOPT[:verbose] 

         printstyled("*********************************\n "; color=:cyan)
         printstyled("      Estimation Results:\n"; color=:cyan); 
         printstyled("*********************************\n"; color=:cyan)

         print("Model type: "); printstyled(minfo1; color=:yellow); println()
         print("Number of observations: "); printstyled(num.nofobs; color=:yellow); println()
         print("Number of total iterations: "); printstyled(sf_total_iter; color=:yellow); println()
         if Optim.converged(_optres) 
             print("Converged successfully: "); printstyled(Optim.converged(_optres); color=:yellow); println()
         elseif Optim.converged(_optres) == false
             print("Converged successfully: "); printstyled(Optim.converged(_optres); color=:red); println()
             redflag = 1
         end         
         print("Log-likelihood value: "); printstyled(round(-1*Optim.minimum(_optres); digits=5); color=:yellow); println()
         println()
     
         pretty_table(table[2:end,:],    # could print the whole table as is, but this prettier
                      header=["", "Var.", "Coef.", "Std.Err.", "z", "P>|z|", 
                              "95%CI_l", "95%CI_u"],
                      formatters = ft_printf("%5.4f", 3:8),
                      compact_printing = true,
                      backend = Val(sf_table))
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
             println("Convert the constant log-parameter to its original scale, e.g., σ² = exp(log_σ²):")   
             pretty_table(auxtable[1:rn,:],
                          header=["", "Coef.", "Std.Err."],
                          formatters = ft_printf("%5.4f", 2:3),
                          compact_printing = true,
                          backend = Val(sf_table))

             print("\nTable format: "); printstyled("$(sf_table)"; color=:yellow); println(". Use sfmodel_opt() to choose between text, html, and latex.")
             println()
         end

         printstyled("***** Additional Information *********\n"; color=:cyan)
 

         print("* OLS (frontier-only) log-likelihood: "); printstyled(round(ll_ols; digits=5); color=:yellow); println("")
         print("* Skewness of OLS residuals: "); printstyled(round(sk_ols; digits=5); color=:yellow); println("")
         if _dicOPT[:ineff_index] 
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

     end  # if_verbose

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
#     _dicRES[:data]          = "$sfdat"
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
      _dicRES[:timevar]       = _dicM[:timevar]  
      _dicRES[:idvar]         = _dicM[:idvar] # for bootstrap marginal effect
      _dicRES[:table_format]  = _dicOPT[:table_format]
      _dicRES[:modelid]       = tagD[:modelid]
      _dicRES[:verbose]       = _dicOPT[:verbose]
      _dicRES[:hasDF]         = _dicM[:hasDF]
      _dicRES[:transfer]      = _dicM[:transfer]

    for i in 1:length(eqvec2)
        _dicRES[keys(eqvec2)[i]] = _coevec[eqvec2[i]]
    end

      _dicRES[:________________]  = "___________________" #34
      _dicRES[:Hessian]           = [numerical_hessian][1]
      _dicRES[:gradient_norm]     = Optim.g_residual(_optres)
    # _dicRES[:trace]             = Optim.trace(_optres)     # comment out because not very informative and size could be large
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

     #* ---- Create a gloal dictionary for sf_predict ---- 

        global _eqncoe 
        _eqncoe = Dict{Symbol, Vector}()  # nullify and initiate new dictionaries when a new model is specified

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
   vvar,         rowIDT, varlist) = SFrontiers.getvar(result.modelid, data)

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

  #* ----- Define the problem's Hessian -----#


   _myfun = TwiceDifferentiable(rho -> SFrontiers.LL_T(result.modelid, 
                         yvar, xvar, zvar, qvar, wvar, vvar, 
                         _porc, nofobs1, pos, rho,
                                 rowIDT, mymisc),
                          sf_init;               
                          autodiff = :forward); 

  #* ---- estimate ----------------- *#

             _optres = try
                       optimize(_myfun, 
                                sf_init,       # initial values  
                                sf_algo,       # different from search run
                                Optim.Options(g_tol = sf_tol,
                                              f_tol=0.0, # force `Optim` to ignore this, but sometimes it does meet the 0.0 criterion
                                              x_tol=0.0, # same above                                
                                              iterations  = sf_maxit, # different from search run
                                              store_trace = false,
                                              show_trace  = false))
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
    numerical_hessian  = hessian!(_myfun, _coevec)  # Hessain

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

            pretty_table(table,
                         show_header = false,
                         body_hlines = [2],
                         formatters = ft_printf("%0.5f", 2:4),
                         compact_printing = true,
                         backend = Val(sf_table))
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

