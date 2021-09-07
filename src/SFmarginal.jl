#########################################################
#                                                       #
#  marginal effects of exogenous determinants on E(u)   #
#                                                       #
#########################################################

#? ------------ Utilities --------------------- #

function nonConsDataFrame(D::DataFrame, M::Matrix)
       # Given a DataFrame containing the marginal effects 
       # of a set of exogenous determinants $(x1, x2, ..., xn)$
       # on E(u), it return the DataFrame where the marginal 
       # effect of constant $x$s are removed.

       # D: the marginal effect DataFrame; 
       # M: the matrix of (x1, .., xn) where the marginal 
       #    efect is calculated from.

      counter = 0      
      for w in collect(names(D),)
           counter += 1
           if length(unique(M[:, counter])) == 1 # is a constant
               select!(D, Not(Symbol(w)))
           end
      end 
      return D
end

         #? ------------------------ #

function addDataFrame(Main::DataFrame, A::DataFrame)
         # Combine two DataFrame with unions of columns.
         # For same-name columns, the values are added together.

       for k in collect(names(A),) # deal with the wvar
                 if k ∈ names(Main)
                      Main[:, Symbol(k)] = Main[:, Symbol(k)] + A[:, Symbol(k)]
                 else 
                      insertcols!(Main, Symbol(k) => A[:, Symbol(k)])
                 end
        end 
        return Main  
end 



#? ----------- truncated normal, marginal effect function -------


function marg_trun( # PorC::Int64, 
                    pos::NamedTuple, 
                    coef::Array{Float64, 1},
                    Zmarg, Wmarg)

         #* No need `modelid` because each has to be customized (cannot accept
         #*   qvar())

         #* Zmarg is the marginal effect, which acts like "coeff" in a usual
         #*   MLE model.
         #*   (1) This is obs-by-obs, so for example, 
         #*       use X'*coef[...] rather than X*coef[....].
         #*   (2) Marg eff is on E(u), not BC or JLMS.
         #*   (3) The line `return ....` cannot be used.

       z_pre = Zmarg'*coef[pos.begz : pos.endz]  # mu
       w_pre = Wmarg'*coef[pos.begw : pos.endw] # log_σᵤ²

       μ   = z_pre 
       σᵤ  = exp(0.5 * w_pre)

       Λ = μ/σᵤ 

       uncondU = σᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
end   

#? ----------- truncated normal, get marginal effect -------
 
function get_marg(::Type{Trun}, # PorC::Int64, 
                  pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
                  Z::Matrix, q, W::Matrix)

    mm_z = Array{Float64}(undef, num.nofz, num.nofobs)             
    mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    

    @inbounds for i in 1:num.nofobs   
        @views marg = ForwardDiff.gradient(marg -> marg_trun(pos, coef, 
                                                      marg[1 : num.nofz],
                                                      marg[num.nofz+1 : num.nofmarg]),
                                    vcat(Z[i,:], W[i,:]) );   

        mm_z[:,i] = marg[1 : num.nofz]
        mm_w[:,i] = marg[num.nofz+1 : end]

     end  # for i in 1:num.nofobs

     margeff = DataFrame(mm_z', _dicM[:μ]) # the base set
        mm_w = DataFrame(mm_w', _dicM[:σᵤ²])

      #* purge off the constant var's marginal effect from the DataFrame
         margeff = nonConsDataFrame(margeff, Z)
            mm_w = nonConsDataFrame(mm_w, W)

      #* if same var in different equations, add up the marg eff
         margeff = addDataFrame(margeff, mm_w)

      #* prepare info for printing
         margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
         newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
         margeff = rename!(margeff, vec(newname))

     return  margeff, margMean

end  

#? ---------------------------------------------------------
#? ----------- half normal, marginal effect function -------
#? --------------------------------------------------------- 

function marg_half( # PorC::Int64, 
     pos::NamedTuple, coef::Array{Float64, 1},
     Wmarg)

     w_pre = Wmarg'*coef[pos.begw : pos.endw] # log_σᵤ²
     σᵤ  = exp(0.5 * w_pre)
     uncondU = sqrt(2/π) * σᵤ # kx1

end   


#? ----------- half normal, get marginal effect -------
 
function get_marg(::Type{Half}, # PorC::Int64, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     z, q, W::Matrix)

     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    

     @inbounds for i in 1:num.nofobs   
     @views marg = ForwardDiff.gradient(marg -> marg_half(pos, coef, 
                                             marg[1 : num.nofw]),
                         vcat(W[i,:]) );   

     mm_w[:,i] = marg[1 : end]

     end  # for i in 1:num.nofobs


     margeff = DataFrame(mm_w', _dicM[:σᵤ²]) # the base set

     #* purge off the constant var's marginal effect from the DataFrame
         margeff = nonConsDataFrame(margeff, W)

     #* if same var in different equations, add up the marg eff
         #margeff = addDataFrame(margeff, mm_w)

      #* prepare info for printing
         margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
         newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
         margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  


#? ---------------------------------------------------------
#? ---------- Exponential normal, marginal effect function -
#? --------------------------------------------------------- 

function marg_expo( pos::NamedTuple, coef::Array{Float64, 1},
     Wmarg)

     w_pre = Wmarg'*coef[pos.begw : pos.endw] # log_σᵤ²
     λ   = exp(0.5*w_pre)
     uncondU = λ # kx1

end   


#? ----------- Exponential, get marginal effect -------
 
function get_marg(::Type{Expo}, # PorC::Int64, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     z, q, W::Matrix)

     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    

     @inbounds for i in 1:num.nofobs   
     @views       marg = ForwardDiff.gradient(marg -> marg_expo( pos, coef, 
                                           marg[1 : num.nofw]),
                                 vcat(W[i,:]) );   

     mm_w[:,i] = marg[1 : end]

     end  # for i in 1:num.nofobs

     margeff = DataFrame(mm_w', _dicM[:σᵤ²]) # the base set

     #* purge off the constant var's marginal effect from the DataFrame
         margeff = nonConsDataFrame(margeff, W)

     #* if same var in different equations, add up the marg eff
         #margeff = addDataFrame(margeff, mm_w)

      #* prepare info for printing
         margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
         newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
         margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  


#? ---------------------------------------------------------
#? ------ Scaling Property Model, marginal effect function -
#? --------------------------------------------------------- 

function marg_scal( # PorC::Int64, 
     pos::NamedTuple, coef::Array{Float64, 1},
     Zmarg, Qmarg, Wmarg)

     z_pre = Zmarg'*coef[pos.begz : pos.endz]  # mu
     q_pre = Qmarg'*coef[pos.begq : pos.endq]
     w_pre = Wmarg'*coef[pos.begw : pos.endw] # log_σᵤ²
 
     μ   = (z_pre) * exp(q_pre)             # Q is _cons; μ here is the after-mutiplied-by-scaling function
     σᵤ  = exp(0.5 * w_pre + q_pre)
     Λ   = μ/σᵤ
 
     uncondU = σᵤ* (Λ + normpdf(Λ) / normcdf(Λ))

end   

#? ----------- scaling property model, get marginal effect -------
 
function get_marg(::Type{Trun_Scale}, # PorC::Int64, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Z::Matrix, Q::Matrix, W::Matrix)

     mm_z = Array{Float64}(undef, num.nofz, num.nofobs)  
     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    

     @inbounds for i in 1:num.nofobs   
     @views marg = ForwardDiff.gradient(marg -> marg_scal( pos, coef, 
                                             marg[1 : num.nofz],
                                             marg[num.nofz+1 : num.nofz+num.nofq],
                                             marg[num.nofz+num.nofq+1 : num.nofmarg]),
                         vcat(Z[i,:], Q[i,:], W[i,:]) );   

     mm_z[:,i] = marg[1 : num.nofz]
     mm_q[:,i] = marg[num.nofz+1 : num.nofz+num.nofq]
     mm_w[:,i] = marg[num.nofz+num.nofq+1 : end]

     end  # for i in 1:num.nofobs


     margeff = DataFrame(mm_z', _dicM[:μ]) # the base set
     mm_q = DataFrame(mm_q', _dicM[:hscale])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Z)
     mm_q = nonConsDataFrame(mm_q, Q)
     mm_w = nonConsDataFrame(mm_w, W)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_q)
     margeff = addDataFrame(margeff, mm_w)

     #* prepare info for printing
     margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

     #* modify variable names to indicate marginal effects
     newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
     margeff = rename!(margeff, vec(newname))

     return  margeff, margMean

end  


#? ---------------------------------------------------------
#? - panel FE Wang and Ho, Half normal, marginal effect function -
#? --------------------------------------------------------- 

function marg_fewhh( # PorC::Int64, 
     pos::NamedTuple, coef::Array{Float64, 1},
     Qmarg, Wmarg)

     q_pre = Qmarg'*coef[pos.begq : pos.endq]
     w_pre = Wmarg'*coef[pos.begw : pos.endw] # log_σᵤ²

     σᵤ  = exp(0.5*w_pre + q_pre)
     uncondU = sqrt(2/π) * σᵤ
end   


#? -- panel FE Wang and Ho, Half normal, , get marginal effect ----
 
function get_marg(::Type{PFEWHH}, # PorC::Int64, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     z, Q::Matrix, W::Matrix)


        #* Note that Y and X are within-transformed by `getvar`, 
        #* but Q, W, V are still in the original level.

     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    

     @inbounds for i in 1:num.nofobs   
          @views marg = ForwardDiff.gradient(marg -> marg_fewhh( pos, coef, 
                                             marg[1 : num.nofq],
                                             marg[num.nofq+1 : num.nofmarg]),
                         vcat( Q[i,:], W[i,:]) );     
                         
     mm_q[:,i] = marg[1 : num.nofq]
     mm_w[:,i] = marg[num.nofq+1 : end]

     end  # for i in 1:num.nofobs

     margeff = DataFrame(mm_q', _dicM[:hscale])
     mm_w    = DataFrame(mm_w', _dicM[:σᵤ²])

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Q)
     mm_w    = nonConsDataFrame(mm_w, W)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_w)

     #* prepare info for printing
     margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

     #* modify variable names to indicate marginal effects
     newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
     margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  



#? --------------------------------------------------------------------
#? - panel FE Wang and Ho, truncated normal, marginal effect function -
#? -------------------------------------------------------------------- 

function marg_fewht( # PorC::Int64, 
     pos::NamedTuple, coef::Array{Float64, 1},
     Zmarg, Qmarg, Wmarg)

       z_pre = Zmarg'*coef[pos.begz : pos.endz]  # mu, a scalar
       q_pre = Qmarg'*coef[pos.begq : pos.endq]
       w_pre = Wmarg'*coef[pos.begw : pos.endw] # log_σᵤ²
   
      μ  = exp(q_pre)*z_pre
      σᵤ = exp(q_pre + 0.5*w_pre) 
      Λ  = μ/σᵤ 
    
    uncondU = σᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
   
end   


#? -- panel FE Wang and Ho, truncated normal, , get marginal effect ----
 
function get_marg(::Type{PFEWHT}, # PorC::Int64, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Z::Matrix, Q::Matrix, W::Matrix)

        #* Note that Y and X are within-transformed by `getvar`, 
        #* but Q, W, V are still in the original level.

     mm_z = Array{Float64}(undef, num.nofz, num.nofobs)  
     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    

     @inbounds for i in 1:num.nofobs   
     @views marg = ForwardDiff.gradient(marg -> marg_fewht( pos, coef, 
                                             marg[1 : num.nofz],
                                             marg[num.nofz+1 : num.nofz+num.nofq],
                                             marg[num.nofz+num.nofq+1 : num.nofmarg]),
                         vcat( Z[i,:], Q[i,:], W[i,:]) );                            

     mm_z[:,i] = marg[1 : num.nofz]
     mm_q[:,i] = marg[num.nofz+1 : num.nofz+num.nofq]
     mm_w[:,i] = marg[num.nofz+num.nofq+1 : end]

     end  # for i in 1:num.nofobs

     margeff = DataFrame(mm_z', _dicM[:μ]) # the base set
     mm_q = DataFrame(mm_q', _dicM[:hscale])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Z)
     mm_q = nonConsDataFrame(mm_q, Q)
     mm_w = nonConsDataFrame(mm_w, W)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_q)
     margeff = addDataFrame(margeff, mm_w)

      #* prepare info for printing
      margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
      newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
      margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  


#? --------------------------------------------------------------------
#? - Panel Time Decay Model, marginal effect function -
#? -------------------------------------------------------------------- 

function marg_pdecay( # PorC::Int64, 
     pos::NamedTuple, coef::Array{Float64, 1},
     Zmarg, Qmarg, Wmarg)
     
     #* same as marg_fewht

     z_pre = Zmarg'*coef[pos.begz : pos.endz]  # mu, a scalar
     q_pre = Qmarg'*coef[pos.begq : pos.endq]
     w_pre = Wmarg'*coef[pos.begw : pos.endw] # log_σᵤ²

     μ  = exp(q_pre)*z_pre
     σᵤ = exp(q_pre + 0.5*w_pre) 
     Λ  = μ/σᵤ 
    
     uncondU = σᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
   
end   


#? -- panel time decay model, get marginal effect ----
 
function get_marg(::Type{PanDecay}, # PorC::Int64, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Z::Matrix, Q::Matrix, W::Matrix)

        #* same as fewht

     mm_z = Array{Float64}(undef, num.nofz, num.nofobs)  
     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    


     @inbounds for i in 1:num.nofobs   
     @views marg = ForwardDiff.gradient(marg -> marg_pdecay( pos, coef, 
                                             marg[1 : num.nofz],
                                             marg[num.nofz+1 : num.nofz+num.nofq],
                                             marg[num.nofz+num.nofq+1 : num.nofmarg]),
                         vcat( Z[i,:], Q[i,:], W[i,:]) );                            

     mm_z[:,i] = marg[1 : num.nofz]
     mm_q[:,i] = marg[num.nofz+1 : num.nofz+num.nofq]
     mm_w[:,i] = marg[num.nofz+num.nofq+1 : end]

     end  # for i in 1:num.nofobs

     margeff = DataFrame(mm_z', _dicM[:μ]) # the base set
     mm_q = DataFrame(mm_q', _dicM[:gamma])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Z)
     mm_q = nonConsDataFrame(mm_q, Q)
     mm_w = nonConsDataFrame(mm_w, W)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_q)
     margeff = addDataFrame(margeff, mm_w)

     #* prepare info for printing
     margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

     #* modify var name to indicate marginal effect
     newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
     margeff = rename!(margeff, vec(newname))

     return  margeff, margMean

end  


#? --------------------------------------------------------------------
#? - Panel Kumbhakar 1990 Model, marginal effect function -
#? -------------------------------------------------------------------- 

function marg_pkumb90( # PorC::Int64, 
     pos::NamedTuple, coef::Array{Float64, 1},
     Zmarg, Qmarg, Wmarg)
     
     #* same as marg_fewht

     z_pre = Zmarg'*coef[pos.begz : pos.endz]  # mu, a scalar
     q_pre = Qmarg'*coef[pos.begq : pos.endq]
     w_pre = Wmarg'*coef[pos.begw : pos.endw] # log_σᵤ²

     μ  = (2 ./(1+exp(q_pre)))*z_pre # exp(q_pre)*z_pre
     σᵤ = (2 ./(1+exp(q_pre)))*exp(0.5*w_pre)    # exp(q_pre + 0.5*w_pre) 
     Λ  = μ/σᵤ 
    
     uncondU = σᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
   
end   


#? -- panel Kumbhakar 1990 model, get marginal effect ----
 
function get_marg(::Type{PanKumb90}, # PorC::Int64, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Z::Matrix, Q::Matrix, W::Matrix)

        #* same as fewht

     mm_z = Array{Float64}(undef, num.nofz, num.nofobs)  
     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    

     @inbounds for i in 1:num.nofobs   
     @views marg = ForwardDiff.gradient(marg -> marg_pkumb90( pos, coef, 
                                             marg[1 : num.nofz],
                                             marg[num.nofz+1 : num.nofz+num.nofq],
                                             marg[num.nofz+num.nofq+1 : num.nofmarg]),
                         vcat( Z[i,:], Q[i,:], W[i,:]) );                            

     mm_z[:,i] = marg[1 : num.nofz]
     mm_q[:,i] = marg[num.nofz+1 : num.nofz+num.nofq]
     mm_w[:,i] = marg[num.nofz+num.nofq+1 : end]

     end  # for i in 1:num.nofobs

     margeff = DataFrame(mm_z', _dicM[:μ]) # the base set
     mm_q = DataFrame(mm_q', _dicM[:gamma])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Z)
     mm_q = nonConsDataFrame(mm_q, Q)
     mm_w = nonConsDataFrame(mm_w, W)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_q)
     margeff = addDataFrame(margeff, mm_w)

     #* prepare info for printing
     margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

     #* modify var name to indicate marginal effect
     newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
     margeff = rename!(margeff, vec(newname))

     return  margeff, margMean

end  



#? -----------------------------------------------------------------------------
#? - panel FE CSW (JoE 2014), CSN model; Half normal, marginal effect function -
#? ----------------------------------------------------------------------------- 


#? -- panel FE CSW (JoE 2014), CSN model; Half normal, get marginal effect ----
 
function get_marg(::Type{PFECSWH}, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     z, q, w)

     # This model does not allow exogenous determinants of inefficiency
     # thus no marginal effect.

     return  (), ()
end  



#? -----------------------------------------------------------------------------
#? - panel true random effect model, half normal and truncated normal
#? ----------------------------------------------------------------------------- 


#? --  Half normal ---------------
 
function get_marg(::Type{PTREH}, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     z, q, w)

     # This model does not allow exogenous determinants of inefficiency
     # thus no marginal effect.

     return  (), ()
end  

#? --  truncated normal ---------------
 
function get_marg(::Type{PTRET}, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     z, q, w)

     # This model does not allow exogenous determinants of inefficiency
     # thus no marginal effect.

     return  (), ()
end  
