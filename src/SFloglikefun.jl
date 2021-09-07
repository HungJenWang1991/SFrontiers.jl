# y x         σᵤ²  σᵥ² -- expo      
# y x         σᵤ²  σᵥ² -- half
# y x μ       σᵤ²  σᵥ² -- trun     
# y x μ  h    σᵤ²  σᵥ² -- scal  
# y x    h    σᵤ²  σᵥ² -- TFE_WH2010, half  
# y x μ  h    σᵤ²  σᵥ² -- TFE_WH2010, truncated  
# y x μ  g    σᵤ²  σᵥ² -- decay  
# y x         σᵤ²  σᵥ² -- panel half (2014 JoE)
# y x    σₐ²  σᵤ²  σᵥ² -- TRE  
# ------------------------------------------
# y x z  q    w    v   -- generic varname
#   β δ1 τ    δ2   γ   -- coeff 


    # reduce the overhead of normpdf(a,b,c); safe in how it is used in the current module
sfnormpdf(μ::Real, σ::Real, x::Number) = 
  exp(-0.5 * StatsFuns.abs2( StatsFuns.zval(μ, σ, x) )) * invsqrt2π / σ



 #* ----  Normal Truncated-Normal -------------

function LL_T(::Type{Trun}, Y::Matrix, X::Matrix, Z::Matrix, q, W::Matrix, V::Matrix,                   
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, dum2, ::Nothing )

    β  = rho[po.begx : po.endx] 
    δ1 = rho[po.begz : po.endz]
    δ2 = rho[po.begw : po.endw]
    γ  = rho[po.begv : po.endv]

    μ   = Z * δ1 
    σᵤ² = exp.(W * δ2)
    σᵤ  = exp.(0.5 * W * δ2)
    σᵥ² = exp.(V * γ)    
    σ²  = σᵤ² + σᵥ²
    ϵ   = PorC*(Y - X * β)
    μₛ  = (σᵥ² .* μ - σᵤ² .* ϵ) ./ σ²
    σₛ² = (σᵥ² .* σᵤ²) ./ σ²

    @floop begin
      llike = zero(eltype(Y))
    # Threads.@threads  # using this leads to incorrect answers, prob. b/c data race
      @inbounds for i in 1:nobs  # faster than the vectorized version
            @views llike += (- 0.5 * log(σ²[i]) 
                            + normlogpdf((μ[i] + ϵ[i]) / sqrt(σ²[i]))  
                            + normlogcdf((μₛ[i]) / sqrt(σₛ²[i]))  
                            - normlogcdf((μ[i]) / σᵤ[i]) )
      end
    end  

    
    return -llike 

end    



#* -------   Normal Half-Normal -----------------# 

function LL_T(::Type{Half}, Y::Matrix, X::Matrix, z, q, W::Matrix, V::Matrix,                   
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, dum2, ::Nothing  )

    β  = rho[po.begx : po.endx] 
    δ2 = rho[po.begw : po.endw]
    γ  = rho[po.begv : po.endv]
 
    σᵤ² = exp.(W * δ2)
    σᵤ  = exp.(0.5 * W * δ2)
    σᵥ² = exp.(V * γ)    
    σ²  = σᵤ² + σᵥ²
    ϵ   = PorC*(Y - X * β)
    μₛ  = ( - σᵤ² .* ϵ) ./ σ²
    σₛ² = (σᵥ² .* σᵤ²) ./ σ²

    @floop begin
      llike = zero(eltype(ϵ))
      # Threads.@threads  # using this leads to incorrect answers, data race
      @inbounds for i in (1:nobs)  
                  @views llike += ( - 0.5 * log(σ²[i]) 
                                    + normlogpdf( (ϵ[i]) / sqrt(σ²[i]))  
                                    + normlogcdf((μₛ[i]) / sqrt(σₛ²[i]))  
                                    - normlogcdf(0)  )
      end
    end

    return -llike 
end  




#* ------ Exponential --------------------#


function LL_T(::Type{Expo}, Y::Matrix, X::Matrix, z, q, W::Matrix, V::Matrix,                   
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, dum2, ::Nothing  )


    β  = rho[po.begx : po.endx] 
    δ2 = rho[po.begw : po.endw]
    γ  = rho[po.begv : po.endv]

    
    λ²  = exp.(W * δ2)        # in the printout, changed to σᵤ²
    λ   = exp.(0.5 * W * δ2)  # in the printout, changed to σᵤ
    σᵥ² = exp.(V * γ)    
    σᵥ  = exp.(0.5 * V * γ)

    ϵ   = PorC*(Y - X * β)

    @floop begin
      llike = zero(eltype(ϵ))
      # Threads.@threads  # using this leads to incorrect answers, strange
      @inbounds for i in (1:nobs)  # faster than the vectorized version
                    @views llike += ( -log(λ[i]) 
                            + normlogcdf(-(ϵ[i]/σᵥ[i])  - (σᵥ[i]/λ[i]))
                            + ϵ[i]/λ[i]  
                            + σᵥ²[i]/(2*λ²[i]) )
                end
    end
    return -llike 

end  

#* ----- scaling property -------------

function LL_T(::Type{Trun_Scale}, Y::Matrix, X::Matrix, Z::Matrix, Q::Matrix, W::Matrix,  V::Matrix, 
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, dum2, ::Nothing )

    β  = rho[po.begx : po.endx] 
    δ1 = rho[po.begz : po.endz]
    τ  = rho[po.begq : po.endq]
    δ2 = rho[po.begw : po.endw]
    γ  = rho[po.begv : po.endv]

 #  h   = exp(Z*δ1)
    μ   = (Z*δ1) .* exp.(Q*τ)        # Q is _cons; μ is the after-mutiplied-by-scaling function
    σᵤ² = exp.(W * δ2 + 2*Q*τ)       # The notation of σᵤ² is different from the paper. W is _cons
    σᵤ  = exp.(0.5 * W * δ2 + Q*τ)
    σᵥ² = exp.(V * γ)    
    σ²  = σᵤ² + σᵥ²
    ϵ   = PorC*(Y - X * β)
    μₛ  = (σᵥ² .* μ - σᵤ² .* ϵ) ./ σ²
    σₛ² = (σᵥ² .* σᵤ²) ./ σ²
    
    @floop begin
      llike = zero(eltype(ϵ))
      # Threads.@threads  # using this leads to incorrect answers, strange
      @inbounds for i in (1:nobs)  
                   @views llike += ( - 0.5 * log(σ²[i]) 
                              + normlogpdf((μ[i] + ϵ[i]) / sqrt(σ²[i]))  
                              + normlogcdf((μₛ[i]) / sqrt(σₛ²[i]))  
                              - normlogcdf((μ[i]) / σᵤ[i])  )
      end  # for
    end  # begin
    return -llike 

end    


#* -------- panel fixed effect, Half normal, Wang and Ho 2010 --------


function LL_T(::Type{PFEWHH}, Ỹ::Union{Vector,Matrix}, X̃::Matrix, z, Q::Matrix, w, v,
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, idt::Matrix{Any}, ::Nothing)

    β  = rho[1:po.endx]
    τ  = rho[po.begq : po.endq]
    δ2 = rho[po.begw]  
    γ  = rho[po.begv]  # may use rho[po.begw : po.endw][1]

     
    μ   = 0.0
    σᵤ² = exp(δ2)     
    σᵥ² = exp(γ)  
    h   = exp.(Q*τ)
    ϵ̃   = PorC*(Ỹ - X̃ * β)

    nofid = size(idt,1)


    @floop begin
      lik = zero(eltype(ϵ̃))
  
      @inbounds for i in 1:nofid
  
        # @views Π⁻  = INVtrM[i]*(1/σᵥ²) 
          @views ind = idt[i,1]
  
        # @views trM = INVtrM[i] # this is exactly Matrix(I, idt[i,2], idt[i,2]) .- (1/idt[i,2])
          @views   h̃ = sf_demean(h[ind]) # INVtrM[i] * h[ind]  #trM * h[ind]; trM == INVtrM
                 σₛₛ² = 1.0/(h̃'*h̃*(1/σᵥ²) + 1/σᵤ²)  #  σₛₛ² = 1.0/(h̃'*Π⁻*h̃ + 1/σᵤ²); h̃ is already de-meaned, so skip Π⁻ but mind 1/σᵥ² 
          @views μₛₛ  = (μ/σᵤ² - ϵ̃[ind]'*h̃*(1/σᵥ²)) * σₛₛ²   # (μ/σᵤ² - ϵ̃[ind]'*Π⁻*h̃) * σₛₛ² ; h̃ is already de-meaned, so skip Π⁻ but mind 1/σᵥ² 
          @views es2 = -0.5*sum(sf_demean(ϵ̃[ind]).^2)*(1/σᵥ²) # -0.5*ϵ̃[ind]'*Π⁻*ϵ̃[ind]

          @views KK   = -0.5*(idt[i,2] - 1)*log(2π) - 0.5*(idt[i,2] - 1)*γ 
  
          lik += (KK + es2 
                  + 0.5*((μₛₛ^2)/σₛₛ² - μ^2/σᵤ²) 
                  + 0.5*log(σₛₛ²) + normlogcdf(μₛₛ/sqrt(σₛₛ²)) 
                  - 0.5*δ2 - normlogcdf(μ/sqrt(σᵤ²)) )
      end # for i=1:N
    end
    return -lik
end


#* -------- panel fixed effect, truncated normal, Wang and Ho 2010 --------


function LL_T(::Type{PFEWHT}, Ỹ::Union{Vector,Matrix}, X̃::Matrix, z, Q::Matrix, w, v,
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, idt::Matrix{Any}, ::Nothing)

    β  = rho[1:po.endx]
    δ1 = rho[po.begz]
    τ  = rho[po.begq:po.endq]
    δ2 = rho[po.begw]  
    γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
     
    μ   = δ1
    σᵤ² = exp(δ2) 
    σᵥ² = exp(γ)  
    h   = exp.(Q*τ)
    ϵ̃   = PorC*(Ỹ - X̃ * β)

    nofid = size(idt,1)

    @floop begin
      lik = zero(eltype(Ỹ))
  
      @inbounds for i in 1:nofid
  
       #  @views Π⁻  = INVtrM[i]*(1/σᵥ²) 
          @views ind = idt[i,1]
  
        # @views trM = Matrix(I, idt[i,2], idt[i,2]) .- (1/idt[i,2])
          @views   h̃ = sf_demean(h[ind]) # INVtrM[i] * h[ind]   # INVtrM is the same as trM; trM * h[ind]
                 σₛₛ² = 1.0/(h̃'*h̃*(1/σᵥ²) + 1/σᵤ²)  # 1.0/(h̃'*Π⁻*h̃ + 1/σᵤ²) ; h̃ is already de-meaned, so skip Π⁻ but mind 1/σᵥ² 
          @views μₛₛ  = (μ/σᵤ² - ϵ̃[ind]'*h̃*(1/σᵥ²)) * σₛₛ²   #(μ/σᵤ² - ϵ̃[ind]'*Π⁻*h̃) * σₛₛ²; h̃ is already de-meaned, so skip Π⁻ but mind 1/σᵥ² 
          @views es2 = -0.5*sum(sf_demean(ϵ̃[ind]).^2)*(1/σᵥ²) # -0.5*ϵ̃[ind]'*Π⁻*ϵ̃[ind]

          @views KK   = -0.5*(idt[i,2] - 1)*log(2π) - 0.5*(idt[i,2] - 1)*γ 
  
          lik += (KK + es2 
                  + 0.5*((μₛₛ^2)/σₛₛ² - μ^2/σᵤ²) 
                  + 0.5*log(σₛₛ²) + normlogcdf(μₛₛ/sqrt(σₛₛ²)) 
                  - 0.5*δ2 - normlogcdf(μ/sqrt(σᵤ²)) )
       end # for i=1:N
     end
     return -lik
end


#* -------- panel BC1992, Time Decay Model --------------

function LL_T(::Type{PanDecay}, Y::Union{Vector,Matrix}, X::Matrix, Z::Matrix, Q::Matrix, W::Matrix, V::Matrix, 
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, idt::Matrix{Any}, ::Nothing)


    β  = rho[1:po.endx]
    δ1 = rho[po.begz:po.endz]
    τ  = rho[po.begq:po.endq]
    δ2 = rho[po.begw]  
    γ  = rho[po.begv]  # rho[po.begw : po.endw][1]

    nofid = size(idt,1)

    σᵤ² = exp(δ2)    # should be W*log_σᵤ² where W is a _cons =1; make a short cut here
    σᵤ  = exp(0.5*δ2)
    σᵥ² = exp(γ)
    σᵥ  = exp(0.5*γ)
    σₛ² = σᵤ² + σᵥ²
    
    @floop begin
      lnf = zero(eltype(Y))
      @inbounds for i = 1:nofid 
          @views ind = idt[i,1]
          @views Tᵢ  = idt[i,2] 
          @views ε   = PorC*(Y[ind] - X[ind, :] * β)
          @views μ   = (Z[ind, :] * δ1)[1]  # ok becuase time-invariant
          @views Gₜ  = exp.(Q[ind, :] * τ ) 
                 Gε  = Gₜ .* ε
  
          Σε²  = sum_kbn(ε.^2)  #*  sensitive in this part, need precision
          ΣGε  = sum_kbn(Gε)    
          ΣGₜ² = sum_kbn(Gₜ.^2) 
          # ΣGₜ  = sum_kbn(Gₜ)    
        
          μₛ      = (μ * σᵥ² - ΣGε * σᵤ²) / (σᵥ² + σᵤ² * ΣGₜ²)
          σₛ²     =  σᵥ² * σᵤ² / (σᵥ² + σᵤ² * ΣGₜ²)
          comp1   = (μ * σᵥ² - σᵤ² * ΣGε)^2 / (σᵥ² * σᵤ² * (σᵥ² + σᵤ² * ΣGₜ²))
          bigterm =  Σε²/σᵥ² + (μ)^2 / σᵤ² - comp1
  
          lnf += ( - 0.5*(Tᵢ)*log(2π) + normlogcdf(μₛ/sqrt(σₛ²)) 
                  + 0.5*log(σₛ²) - 0.5*bigterm - Tᵢ*0.5*γ
                  - 0.5*δ2 - normlogcdf(μ/σᵤ)  )
      end
    end  
    return -lnf
end



#* -------- panel Kumbhakar Model --------------

function LL_T(::Type{PanKumb90}, Y::Union{Vector,Matrix}, X::Matrix, Z::Matrix, Q::Matrix, W::Matrix, V::Matrix, 
  PorC::Int64, nobs::Int64, po::NamedTuple, rho, idt::Matrix{Any}, ::Nothing)


  β  = rho[1:po.endx]
  δ1 = rho[po.begz:po.endz]
  τ  = rho[po.begq:po.endq]
  δ2 = rho[po.begw]  
  γ  = rho[po.begv]  # rho[po.begw : po.endw][1]

  nofid = size(idt,1)

  σᵤ² = exp(δ2)    # should be W*log_σᵤ² where W is a _cons =1; make a short cut here
  σᵤ  = exp(0.5*δ2)
  σᵥ² = exp(γ)
  σᵥ  = exp(0.5*γ)
  σₛ² = σᵤ² + σᵥ²
  
  @floop begin
    lnf = zero(eltype(Y))
    @inbounds for i = 1:nofid 
        @views ind = idt[i,1]
        @views Tᵢ  = idt[i,2] 
        @views ε   = PorC*(Y[ind] - X[ind, :] * β)
        @views μ   = (Z[ind, :] * δ1)[1]  # ok becuase time-invariant
        @views Gₜ  = 2 ./ (1 .+ exp.(Q[ind, :] * τ ) )
               Gε  = Gₜ .* ε

        Σε²  = sum_kbn(ε.^2)  #*  sensitive in this part, need precision
        ΣGε  = sum_kbn(Gε)    
        ΣGₜ² = sum_kbn(Gₜ.^2) 
        # ΣGₜ  = sum_kbn(Gₜ)    
      
        μₛ      = (μ * σᵥ² - ΣGε * σᵤ²) / (σᵥ² + σᵤ² * ΣGₜ²)
        σₛ²     =  σᵥ² * σᵤ² / (σᵥ² + σᵤ² * ΣGₜ²)
        comp1   = (μ * σᵥ² - σᵤ² * ΣGε)^2 / (σᵥ² * σᵤ² * (σᵥ² + σᵤ² * ΣGₜ²))
        bigterm =  Σε²/σᵥ² + (μ)^2 / σᵤ² - comp1

        lnf += ( - 0.5*(Tᵢ)*log(2π) + normlogcdf(μₛ/sqrt(σₛ²)) 
                + 0.5*log(σₛ²) - 0.5*bigterm - Tᵢ*0.5*γ
                - 0.5*δ2 - normlogcdf(μ/σᵤ)  )
    end
  end  
  return -lnf
end



#* -------- panel fixed effect, Half normal, CSW JoE 2014 (CSN) --------


function joe2014lnmvnpdf(x, s2) 

    k = length(x)
    
    # lnf = -k/2*log(2π)-0.5*log(det(σ))-0.5*x'*inv(σ)*x
    
    invSig = Matrix(I, k, k).*(1/s2) .+ (1/s2) # the analytic form of inv(σ)
    detSig = (s2^k)/(k+1) # analytic form of det(σ)
    res = -k/2*log(2π) - 0.5*log(detSig) - 0.5*x'*invSig*x

    return res
end



function LL_T(::Type{PFECSWH}, Ỹ::Union{Vector,Matrix}, X̃::Matrix, z, q, w, v,
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, idt::Matrix{Any}, ::Nothing)

    β   = rho[1:po.endx]
    σᵤ² = exp(rho[po.begw]) 
    σᵥ² = exp(rho[po.begv])  

    σᵤ = exp(0.5*rho[po.begw])
    σᵥ = exp(0.5*rho[po.begv])
    λ  =  σᵤ / σᵥ       
    σ² =  σᵤ² + σᵥ²
    ϵ  = PorC*(Ỹ - X̃ * β)

    nofid = size(idt,1)

    @floop begin
      loglik = zero(eltype(Ỹ))
       @inbounds for i in 1:nofid
           @views Tᵢ    = idt[i, 2]
           @views ind_a = idt[i, 1][1: Tᵢ-1]  # rowID[i][1:Tᵢ[i]-1] 
           @views ind   = idt[i, 1]              # rowID[i]
           @views A     = joe2014lnmvnpdf(ϵ[ind_a], σ²)
                  B     = quadgk(u -> normpdf(u)*prod(normcdf.(-λ/sqrt(σ²)*ϵ[ind].-λ/sqrt(Tᵢ)*u)), -Inf, Inf, rtol=1e-8)
           @views loglik += A + log(B[1])  + log(2)*Tᵢ 
        end
    end
    return -loglik
end



#* ------------------ true random effect model, Half Normal --------------


function LL_T(::Type{PTREH}, Y::Matrix, X::Matrix, z, Q::Matrix, w, v,
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, idt::Matrix{Any}, ::Nothing)

     β   = rho[1:po.endx]
     σₐ² = exp(rho[po.begq])
     σᵤ² = exp(rho[po.begw])  
     σᵥ² = exp(rho[po.begv])  
     σᵤ  = exp(0.5*rho[po.begw])

     ε1  = PorC*(Y - X * β)

    nofid = size(idt,1)

    @floop begin
      lnf = zero(eltype(Y))
      for i = 1:nofid
          @views ind  = idt[i, 1]
          @views T    = idt[i, 2]
          @views ε    = ε1[ind]
                 μ    = zeros(T)
              #  Σ    = ones(T,T)*σₐ² + Matrix(σᵥ²*I,T,T)
              logdetΣ = log(T*σₐ²*(σᵥ²)^(T-1) + (σᵥ²)^T)
              #  Σᵤ   = Matrix(σᵤ²*I,T,T)
                 invΣ = (1/σᵥ²) * (Matrix(I,T,T) - σₐ²/(σᵥ² + T*σₐ²)*ones(T,T))
                invΣᵤ = (1/σᵤ²) * Matrix(I,T,T) 
                 K    = ε'*invΣ - μ'*invΣᵤ
  
        # below is the analytic form of G = inv(inv(Σ)+inv(Σᵤ))
          γ²   = 1/(1/σᵤ²+1/σᵥ²)
          g    = -T*1/(1+σₐ²/σᵥ²*T)*σₐ²/σᵥ²^2*1/(1/σᵤ²+1/σᵥ²) 
          ρ²   = 1/(1+g) * 1/(1/σᵤ²+1/σᵥ²)^2 * 1/(1 + σₐ²/σᵥ²*T) *(σₐ²/σᵥ²^2)
          G    = γ² * Matrix(I,T,T) + ρ² * ones(T,T)
          logdetG = log(T*ρ²*(γ²)^(T-1) + (γ²)^T)
  
          B    = (-K*G)'
          γ    = sqrt(γ²)
          ρ    = sqrt(ρ²)
          cdfBG= quadgk(u -> normpdf(u)*prod(normcdf.((B.-ρ*u)/γ)), -Inf, Inf, rtol=1e-8)         
  
          @views lnf += (log(cdfBG[1]) + 0.5*logdetG - 0.5*logdetΣ
                  - sum(log.(σᵤ*normcdf.(μ/σᵤ))) 
                  - 0.5*(ε'*invΣ*ε) - 0.5*(μ'*invΣᵤ*μ) 
                  + 0.5*(B'*(invΣ + invΣᵤ)*B) -T*0.5*log(2*π) )                 
  
      end
    end  
    return -lnf
end


#* ------ true random effect model, truncated normal ---------


function LL_T(::Type{PTRET}, Y::Matrix, X::Matrix, Z::Matrix, Q::Matrix, w, v,
    PorC::Int64, nobs::Int64, po::NamedTuple, rho, idt::Matrix{Any}, ::Nothing)

    β   = rho[1:po.endx]
    δ   = rho[po.begz : po.endz]

    σₐ² = exp(rho[po.begq])
    σᵤ² = exp(rho[po.begw]) 
    σᵥ² = exp(rho[po.begv]) 

    σᵤ  = exp(0.5*rho[po.begw])
    
    nofid = size(idt, 1)

    @floop begin
      lnf = zero(eltype(Y))
      for i = 1:nofid
          @views ind = idt[i, 1]
          @views T   = idt[i, 2]
          @views ε   = PorC*(Y[ind] - X[ind, :] * β)
          @views μ   = (Z[ind, :] * δ) 
   #             Σ   = ones(T,T)*σₐ² + Matrix(σᵥ²*I,T,T)
          logdetΣ   = log(T*σₐ²*(σᵥ²)^(T-1) + (σᵥ²)^T)
  
   #      Σᵤ   = Matrix(σᵤ²*I,T,T)
          invΣ = (1/σᵥ²) * (Matrix(I,T,T) - σₐ²/(σᵥ² + T*σₐ²)*ones(T,T))
          invΣᵤ= (1/σᵤ²) * Matrix(I,T,T) 
          K    = ε'*invΣ - μ'*invΣᵤ
  
        # below is the analytic form of G = inv(inv(Σ)+inv(Σᵤ))
          γ²   = 1/(1/σᵤ²+1/σᵥ²)
          g    = -T*1/(1+σₐ²/σᵥ²*T)*σₐ²/σᵥ²^2*1/(1/σᵤ²+1/σᵥ²) 
          ρ²   = 1/(1+g) * 1/(1/σᵤ²+1/σᵥ²)^2 * 1/(1 + σₐ²/σᵥ²*T) *(σₐ²/σᵥ²^2)
          G    = γ² * Matrix(I,T,T) + ρ² * ones(T,T)
          logdetG = log(T*ρ²*(γ²)^(T-1) + (γ²)^T)      
  
          B    = (-K*G)'
          γ    = sqrt(γ²)
          ρ    = sqrt(ρ²)
          cdfBG= quadgk(u -> normpdf(u)*prod(normcdf.((B.-ρ*u)/γ)), -Inf, Inf, rtol=1e-8)
          
          @views lnf += (log(cdfBG[1]) + 0.5*logdetG - 0.5*logdetΣ 
                        - sum(log.(σᵤ*normcdf.(μ/σᵤ)))  #sum_kbn
                        - 0.5*(ε'*invΣ*ε) - 0.5*(μ'*invΣᵤ*μ) 
                        + 0.5*(B'*(invΣ + invΣᵤ)*B) -T*0.5*log(2*π) )
      end
    end       
    
    return -lnf
end
