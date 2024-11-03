"""
    sfmodel_MoMTest(<keyword arguments>)

Provide method-of-moments (MoM) tests on the distribution assumptions of the
composed error (v-u or v+u) of stochastic frontier models. Also provide MoM
estimates of the model parameters as well as the inefficiency (JLMS) and
efficiency (BC) index. Return various results in a distionary.


# Arguments

- `sfdist(::Vararg)`: the distribution assumption of the one-sided stochastic
  variable (aka inefficiency term) of the model. Currently the
  choices include `half` (or `h`) and `exponential` (or `expo`, `e`)
- `sftype(::Vararg)`: whether the model is a `production` (or `prod`) frontier
  or a `cost` frontier.
- `@depvar(::Vararg)`: the dependent variable from the `data`.
- `@frontier(::Vararg)`: a list of variables, separated by commas, in the frontier function.
- `data::DataFrames`: the data in the DataFrames format.
- `omega (or ω)::Union{Real, Vector, Tuple}=<a vector of numbers>`:
  The `ω` parameter used in the test. It could take the form of a single value
  (e.g., `ω=1`) or a vector or tuple (e.g., `ω=(0.5, 1, 2)`). Default is
  `ω=1.0`.
- `level::Real=<number>`: The significance level (default=0.05) of the bias-corrected
  confidence intervals. If `level`>0.5, it is automatically transformed to
  `1-level`, such that `level=0.05` and `level=0.95` both return 95%
  confidene intervals at the 5% significance level.  
- `verbose::Bool`: whether print results on screen. Default is `=true`.
- `testonly::Bool`: whether to print only the test results. Default is
  `false`, which print both the test and the estimation results on
  the screen.

# Remarks

- Currently, the two-sided error (aka `v`) of the model is assumed to
  follow a normal distribuion in the test.
- Reference: Chen, Y.T., & Wang, H.J. (2012). Centered-Residuals-Based Moment
  Estimator and Test for Stochastic Frontier Models. Econometric Reviews,
  31(6), 625-653. 

# Examples
```julia-repl
julia> df = DataFrame(CSV.File("demodata.csv"))
julia> df[!, :_cons] .=1.0;
julia> res = sfmodel_MoMTest(sftype(prod), sfdist(expo),
                             @depvar(y), @frontier(_cons, x1, x2, x3),
                             data=df, ω=(0.5,1,2))          
julia> res.list  # a list of the saved results
julia> res.jlms  # the JLMS inefficiency index
```
"""
function sfmodel_MoMTest(arg::Vararg; data::DataFrame, ω::Union{Vector, Real, Tuple}=1.0, 
        omega::Union{Vector, Real, Tuple}=1.0, level::Real=0.05,
        verbose::Bool=true, testonly::Bool=false)

    _dicMoM = Dict{Symbol,Any}()

    for k in (:type, :dist, :depvar, :frontier)
        _dicMoM[k] = nothing
    end

    for d in :($(arg))
        _dicMoM[d[1]] = d[2]
    end

    # --- check syntax; in house so that no need using global dic ----

    for k in (:type, :dist, :depvar, :frontier) 
        if  (_dicMoM[k] === nothing) 
            throw("For the method of moment test/estimation, the `$k` equation is missing in sfmodel_MoMTest().")
        end 
    end

    if length(keys(_dicMoM)) > 4  # should have only 4 at this moment
       throw("For the MoM model specification, only `sftype`, `sfdist`, `@depvar`, and `@frontier` are needed. You apparently provided more than these.")  
    end        

    if typeof(data) != DataFrame
       throw("`data=` needs to be a DataFrame.")
    end    

    #-- retrieve parameters -------

    if !(1 in ω) || (length(ω) > 1)
        ω = ω
    elseif !(1 in omega) || (length(omega) > 1)
        ω = omega
    end

    s = uppercase(String(_dicMoM[:dist][1])[1:1])  # s = "H"(half), s = "E"(exponential)
    if _dicMoM[:type][1] == :cost
        _porc = -1.0
    else
        _porc = 1.0
    end

    ((level > 0.0) && (level < 1.0)) || throw("The significance level (`level`) should be between 0 and 1.")
    if level > 0.5
       level = 1-level  # 0.95 -> 0.05
    end
    α = level


    ##### print the title ############ 

     if verbose

        println("\n****************************************")
        println("** Moment Based Tests and Estimations **")
        println("****************************************")

     end

    ############################### Start Calculation #####################################################


    if (s == "H") || (s == "E")
    
        yvar = data[:, _dicMoM[:depvar]]
        xvar = data[:, _dicMoM[:frontier]]

        sT = size(yvar,1)


        #---- pick the constant ----------

        cnsname, cnspos, cnsnum = pickConsNameFromDF(xvar)
        if cnsnum == 0
           printstyled("\nThere is no intercept (no constant variable) in the model. Are you sure?\n"; color = :green)
        elseif cnsnum == 1
           cnspos = Int(cnspos[1])
        elseif cnsnum > 1
           printstyled("\nThere are multiple constant varibles in the model resulting in multicollinearity. The estimation is abort.\n"; color = :red) 
           throw("Model specification errors.")
        else
           throw("Something is not right about the specification of constant variables in the model")     
        end


        yvar  =  convert(Array{Float64}, Matrix(yvar))
        xvar  =  convert(Array{Float64}, Matrix(xvar))

        β_ols  = xvar \ yvar  # OLS estiamte, uses a pivoted QR factorization;
        resid  = yvar - xvar*β_ols
        sse    = sum((resid).^2)  
        ssd    = sqrt(sse/(size(resid,1)-size(xvar,2))) # sample standard deviation; σ² = (1/(N-K))* Σ ϵ^2
        ll_ols = sum(normlogpdf.(0, ssd, resid)) # ols log-likelihood
        sk_ols = sum((resid).^3) / ((ssd^3)*(size(resid,1))) # skewnewss of ols residuals
        std_ols = sqrt.(diag((ssd^2)*inv(xvar'*xvar)))
        Resid   = _porc*resid  # prod: v-(u-E(u)),  cost: v+(u-E(u))


        CResid = Resid.-mean(Resid)

        dm2 = CResid.^2
        m2 = mean(dm2)
        dm3 = CResid.^3
        m3 = mean(dm3)

        if s == "H"
            minfo1 = "normal and half-normal"
            λ₂ = 1-2/π
            λ₃ = (1-4/π) * sqrt(2/π)
        elseif s == "E"
            minfo1 = "normal and exponential"
            λ₂ = 1
            λ₃ = -2
        end

        σᵤ² = ((m3/λ₃)^2)^(1/3)
        σᵥ² = m2 - λ₂ * σᵤ²


        if cnsnum == 1
            if s == "H"
                β_ols[cnspos] = β_ols[cnspos] .+  _porc*sqrt(σᵤ²)*sqrt(2/π)
            elseif s == "E"
                β_ols[cnspos] = β_ols[cnspos] .+  _porc*sqrt(σᵤ²)
            end
            
            v_adj  = m2 * (inv(xvar' * xvar))     
            std_ols[cnspos] = Real(sqrt(v_adj[cnspos, cnspos]))
        end

        if s == "H"
            Resid2 = Resid .- _porc*sqrt(σᵤ²)*sqrt(2/π)
        elseif s == "E"
            Resid2 = Resid .- _porc*sqrt(σᵤ²)
        end
        
        Grad2 = CResid.^2 .- (σᵥ²+λ₂*σᵤ²)
        Grad3 = CResid.^3 .- λ₃*(σᵤ²^1.5).-3*(σᵥ²+λ₂*σᵤ²)*(CResid)  
        HessE = [-1 -λ₂ ; 0  -1.5*λ₃*sqrt(σᵤ²)]

        Grad22=Grad2.^2
        Grad33=Grad3.^2
        Grad23=Grad2.*Grad3
        mGrad22 = mean(Grad22)
        mGrad33 = mean(Grad33)
        mGrad23 = mean(Grad23)

        mMat = [mGrad22 mGrad23 ; mGrad23 mGrad33]
        vcovM  = (1/sT)*inv(HessE)*mMat*inv(HessE)'
        myvstd = sqrt(vcovM[1,1])
        myustd = sqrt(vcovM[2,2])

        nofom = length(ω)
        mycos = zeros(nofom)
        mysin = zeros(nofom)
        testcs = zeros(nofom)


        for i in 1:nofom
            Ecoswv = exp(-0.5*(ω[i]^2)*σᵥ²)
            EDcoswv= -0.5*(ω[i]^2)*Ecoswv

            if s == "H"  #The normal-half normal model
                μ = sqrt(σᵤ²)*sqrt(2/π)
                Dmu= sqrt(2/π)/(2*sqrt(σᵤ²))
                Ecoswu = exp(-0.5*(ω[i]^2)*σᵤ²)
                Edcoswu = -0.5*(ω[i]^2)*Ecoswu
                t = ω[i]*sqrt(σᵤ²/2)
                erfiw  = erfi(t)
                Esinwu = Ecoswu*erfiw
                EDsinwu =Edcoswu*erfiw+ω[i]/sqrt(2*π*σᵤ²)
            elseif s == "E" #The normal-exponential model
                μ = sqrt(σᵤ²)
                Dmu = 1/(2*sqrt(σᵤ²))
                Ecoswu = 1/(1+σᵤ²*ω[i]^2)
                Edcoswu = -(ω[i]^2)/((1+σᵤ²*ω[i]^2)^2)
                Esinwu = ω[i]*sqrt(σᵤ²)/(1+σᵤ²*(ω[i]^2))
                EDsinwu = ω[i]/(2*sqrt(σᵤ²)*(1+σᵤ²*ω[i]^2))-ω[i]^3*sqrt(σᵤ²)/(1+σᵤ²*(ω[i]^2))^2
            end

            hc = Ecoswu*cos(ω[i]*μ)+Esinwu*sin(ω[i]*μ)
            Mc = Ecoswv*hc
            Dhc= Edcoswu*cos(ω[i]*μ)+EDsinwu*sin(ω[i]*μ)-Ecoswu*sin(ω[i]*μ)*ω[i]*Dmu+Esinwu*cos(ω[i]*μ)*ω[i]*Dmu
            Dc = [-EDcoswv*hc , -Ecoswv*Dhc]

            hs = Ecoswu*sin(ω[i]*μ)-Esinwu*cos(ω[i]*μ)
            Ms = Ecoswv*hs
            Dhs = Edcoswu*sin(ω[i]*μ)-EDsinwu*cos(ω[i]*μ)+Ecoswu*cos(ω[i]*μ)*ω[i]*Dmu+Esinwu*sin(ω[i]*μ)*ω[i]*Dmu
            Ds = [-EDcoswv*hs , -Ecoswv*Dhs ]

            TFc=cos.(ω[i]*CResid).-Mc  
            TFs=sin.(ω[i]*CResid).-Ms 
            Gradc=TFc.+Ms*ω[i]*CResid 
            Grads=TFs.-Mc*ω[i]*CResid 

            invHess = inv(HessE')
            Xi = [Gradc,Grads] - ([Dc Ds]' * invHess' * [Grad2,Grad3])
            xic = Xi[1]
            xis = Xi[2]

            mTFc = mean(TFc)
            mTFs = mean(TFs)
            mTFcs = [mTFc, mTFs]


            # mat CTestcNH = sT*mTFcNH*inv(xicNH'*xicNH/sT)*mTFcNH
            xic2 = xic.^2
            xic2sum = 1/mean(xic2)
            mycosine = sT * mTFc * xic2sum * mTFc
            mycos[i] = mycosine

            # mat CTestsNH = sT*mTFsNH*inv(xisNH'*xisNH/sT)*mTFsNH
            xis2 = xis.^2 
            xis2sum = 1/mean(xis2)
            mysine = sT * mTFs * xis2sum * mTFs
            mysin[i] = mysine


            # mat CTestcsNH = sT*mTFcsNH'*inv(xicsNH'*xicsNH/sT)*mTFcsNH
            xics2 = xic.*xis 
            inv11 = sum(xic2)
            inv22 =sum(xis2)
            inv12 = sum(xics2)
            invmat = [inv11 inv12 ; inv12 inv22]/sT

            CTestcs = sT*mTFcs'*inv(invmat)*mTFcs
            Testcs = CTestcs[1,1]
            testcs[i] = Testcs

        end   
    end


    # ********** calculate log-likelihood value ***********
    
    if s == "H"
        σ²  = σᵤ² + σᵥ²
        μₛ  = ( - σᵤ² .* Resid2) ./ σ²
        σₛ² = (σᵥ² * σᵤ²) / σ²

        llike = sum( .- 0.5 * log(σ²) 
                     .+ normlogpdf.( Resid2 ./ sqrt(σ²))  
                     .+ normlogcdf.(μₛ ./ sqrt(σₛ²))  
                     .- normlogcdf(0) )
    elseif s == "E"
        σᵤ = sqrt(σᵤ²)
        σᵥ = sqrt(σᵥ²)

        llike = sum( -log(σᵤ) .+ normlogcdf.(-(Resid2 ./ σᵥ)  .- (σᵥ/σᵤ))
                    .+ Resid2 ./ σᵤ .+ σᵥ²/(2*σᵤ²) )
    end

    #***** JLMS & BC ********************************
    
    if s == "H" 
        myeps = Resid .- (_porc)*sqrt(σᵤ²)*sqrt(2/π)  # prod: v-(u-E(u)) - E(u) = v-u;  cost: v+(u-E(u)) + E(u) = v+u
        mustar  = -σᵤ²*myeps/(σᵤ² + σᵥ²) 
        sig2star = σᵤ²*σᵥ²/(σᵤ² + σᵥ²) 
        rat1 = mustar/sqrt(sig2star)
        _jlms = sqrt(sig2star)*normpdf.(rat1)./normcdf.(rat1) .+ (mustar) 
        _bc  = exp.(-mustar .+ 0.5*sig2star).*normcdf.(rat1 .- sqrt(sig2star))./normcdf.(rat1)
    elseif s == "E" 
        myeps = Resid .- (_porc)*sqrt(σᵤ²) # prod: v-(u-E(u)) - E(u) = v-u;  cost: v+(u-E(u)) + E(u) = v+u
        mustar = -myeps .- σᵥ²/sqrt(σᵤ²) 
        rat1 = mustar/sqrt(σᵥ²)
        _jlms = sqrt(σᵥ²)*normpdf.(rat1)./normcdf.(rat1) .+ mustar 
        _bc = exp.(-mustar .+ 0.5*σᵥ²).*normcdf.(rat1 .- sqrt(σᵥ²))./normcdf.(rat1)
    end

    _jlmsM = mean(_jlms) 
    _bcM   = mean(_bc)   


    #***** critical values **************************

    crit1 = chisqinvcdf(1,1-0.01)
    crit2 = chisqinvcdf(1,1-0.05)
    crit3 = chisqinvcdf(1,1-0.10)


    #***** prepare for printing test results***********

    omega_list = [i for i in ω]
    test_res = hcat(omega_list, mysin, mycos)
    test_header = (["ω", "sine", "cosine"])

    test_res2 = vcat(reshape(test_header, 1, :), test_res)

    #***********************************

   if verbose

    if s == "H" 
        print( "\n* Null Hypothesis: v is "); printstyled("normal"; bold=true, color=:yellow); print(" AND u is"); printstyled(" half-normal"; bold=true, color=:yellow); println(".")
    elseif s == "E" 
        print( "\n* Null Hypothesis: v is "); printstyled("normal"; bold=true, color=:yellow); print(" AND u is"); printstyled(" exponential"; bold=true, color=:yellow); println(".")
    end
    println(" ")

    println("  Test Statistics (χ² distribution)")

    #* make PrettyTable for sine and cosine test

    pretty_table(test_res; 
                 header = test_header,
                 header_crayon = crayon"yellow bold",
                 formatters = ft_printf("%4.5f", 2:3))
    println("  Note: Chen and Wang (2012 EReviews) indicates that cosine test with ω=1 has good overall performance.\n")


    println("\n  Critical Values (χ²(1))")
    pretty_table([crit1 crit2 crit3]; 
                 header = ["1%","5%","10%"],
                 header_crayon = crayon"yellow bold",
                 formatters = ft_printf("%4.5f", 1:3))

  end # if_verbose

    #***** prepare for printing estimation results *******

    nofpar = length(_dicMoM[:frontier]) + 2
        var_list  = cat([String(i) for i in _dicMoM[:frontier]], ["σᵥ²", "σᵤ²"], dims=1)
        coef_list = cat(β_ols, [σᵥ², σᵤ²], dims=1)
        std_list  = cat(std_ols, [myvstd, myustd], dims=1)
        t_stats = cat(zeros(nofpar-2), ["n.a.","n.a."], dims=1)
        pv_list = cat(zeros(nofpar-2), ["n.a.","n.a."], dims=1)
        ci_down = zeros(nofpar)
        ci_up = zeros(nofpar)
        tt = cquantile(Normal(0,1), α/2)
        for i in 1:nofpar
            if i <= nofpar-2
                t_stats[i] = coef_list[i] / std_list[i]
                pv_list[i] = pvalue(TDist(sT - nofpar), t_stats[i]; tail=:both)
                ci_down[i] = coef_list[i] - tt * std_list[i]
                ci_up[i] = coef_list[i] + tt * std_list[i]
            else
                ci_down[i] = (sT-1)/cquantile(Chisq(sT-1),α/2) * coef_list[i]
                ci_up[i] = (sT-1)/cquantile(Chisq(sT-1),1-(α/2)) * coef_list[i]
            end
        end 

    est_res = hcat(var_list, coef_list, std_list, t_stats, pv_list, ci_down, ci_up)
    est_header = ([" ", "Coef.", "Std. Err.", "z", "P>|z|", "$(Int(100-100α))%CI_l", "$(Int(100-100α))%CI_u"])

    est_res2 = vcat(reshape(est_header, 1, :), est_res)  # for save in dic


   #******* make PrettyTable for estimation results *******

   if verbose && !testonly

    println("\n\n* Method of Moments Estimates of the Model (Chen and Wang 2012 EReviews)")

    print("  ** Model type: "); printstyled(minfo1; color=:yellow); println()
    print("  ** The constant variable (for intercept) in the model: "); 
          if  cnsnum == 0
            printstyled("n.a.\n"; color = :yellow)
          else
            printstyled("$(cnsname[1]).\n"; color = :yellow)
          end  
    print("  ** Number of observations: "); printstyled(sT; color=:yellow); println()
    print("  ** Log-likelihood value: "); printstyled(round(llike; digits=5); color=:yellow); println()
    println()

    pretty_table(est_res; 
                 header = est_header,
                 header_crayon = crayon"yellow bold",
                 formatters = ft_printf("%4.5f", 2:7))
    println("  Note: CI of σᵥ² and σᵤ² is calculated based on the χ² distribution.\n\n")


    # --- print additional information -----

    printstyled("***** Additional Information *********\n"; color=:cyan)

    print("* OLS (frontier-only) log-likelihood: "); printstyled(round(ll_ols; digits=5); color=:yellow); println("")
    print("* Skewness of OLS residuals: "); printstyled(round(sk_ols; digits=5); color=:yellow); println("")
    print("* The sample mean of the JLMS inefficiency index: "); printstyled(round(_jlmsM; digits=5); color=:yellow); println("")
    print("* The sample mean of the BC efficiency index: "); printstyled(round(_bcM; digits=5); color=:yellow); println("\n")

    println("* Use `name.list` to see saved results (keys and values) where `name` is the return specified in `name = sfmodel_MoMTest(..)`. Values may be retrieved using the keys. For instance:")
    println("   ** `name.MoM_loglikelihood`: the log-likelihood value of the model;")
    println("   ** `name.jlms`: Jondrow et al. (1982) inefficiency index;")
    println("   ** `name.bc`: Battese and Coelli (1988) efficiency index;")
    println("* Use `keys(name)` to see available keys.")

    printstyled("**************************************\n\n\n"; color=:cyan)

   end  # if_verbose

    
 #* ########### create a dictionary and make a tuple for return ########### *#
     
      _dicRES = OrderedDict{Symbol, Any}()     
      _dicRES[:n_observations]  = sT    
      _dicRES[:MoMtest]         = [test_res2][1] 
      _dicRES[:table]           = [est_res2][1]  
      _dicRES[:coeff]           = coef_list 
      _dicRES[:std_err]         = std_list 
      _dicRES[:jlms]            = _jlms     
      _dicRES[:bc]              = _bc       
      _dicRES[:MoM_loglikelihood] = llike #! hjw: see SFloglikefun
      _dicRES[:OLS_loglikelihood] = ll_ols  
      _dicRES[:OLS_resid_skew]    = sk_ols  
      _dicRES[:depvar]        = _dicMoM[:depvar]    
      _dicRES[:frontier]      = _dicMoM[:frontier]  
      _dicRES[:type]          = _dicMoM[:type]      
      _dicRES[:dist]          = _dicMoM[:dist]      
      _dicRES[:PorC]          = _porc              

     #* ----- Delete optional keys that have value nothing, 


     #* ----- Create a NamedTuple from the dic as the final output; 
     #* -----     put the dic in the tuple.

         _ntRES = NamedTuple{Tuple(keys(_dicRES))}(values(_dicRES))
         _ntRES = (; _ntRES..., list    = _dicRES)

     #* ---- Create a gloal dictionary for sf_predict ---- 

  #* ############  make returns  ############ *#

      return _ntRES

end


