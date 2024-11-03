
#########################################
####        JLMS and BC index        ####
#########################################

#? --------------- Truncated Normal --------------

function jlmsbc(::Type{Trun}, PorC::Int64, pos::NamedTuple, coef::Array{Float64, 1}, 
                Y::Matrix, X::Matrix, Z::Matrix, q, W::Matrix, V::Matrix, # dum1, 
                dum2)

    x_pre = X*coef[pos.begx : pos.endx]
    z_pre = Z*coef[pos.begz : pos.endz]  # mu
  # q_pre = Q*coef[pos.begq : pos.endq]
    w_pre = W*coef[pos.begw : pos.endw] # log_σᵤ²
    v_pre = V*coef[pos.begv : pos.endv] # log_σᵥ²

    μ   = z_pre 
    σᵤ² = @. exp(w_pre)
    σᵤ  = @. exp(0.5 * w_pre)
    σᵥ² = @. exp(v_pre)    
    σ²  = σᵤ² + σᵥ²
    ϵ   = PorC*(Y - x_pre)
    μₛ  = @. (σᵥ² * μ - σᵤ² * ϵ) / σ²
    σₛ  = @. sqrt((σᵥ² * σᵤ²) / σ²)

   jlms = @. (σₛ * normpdf(μₛ / σₛ )) / normcdf(μₛ / σₛ) + μₛ
     bc = @. exp( -μₛ + 0.5 * (σₛ)^2  ) * ( normcdf( (μₛ / σₛ) - σₛ ) / normcdf(μₛ / σₛ)  )

   return jlms, bc  
end


#? ------------------ Half normal model -------------

function jlmsbc(::Type{Half}, PorC::Int64, pos::NamedTuple, coef::Array{Float64, 1}, 
                Y::Matrix, X::Matrix, z, q, W::Matrix, V::Matrix, # dum1, 
                dum2)


    x_pre = X*coef[pos.begx : pos.endx]
  # z_pre = Z*coef[pos.begz : pos.endz]  # mu
  # q_pre = Q*coef[pos.begq : pos.endq]
    w_pre = W*coef[pos.begw : pos.endw] # log_σᵤ²
    v_pre = V*coef[pos.begv : pos.endv] # log_σᵥ²

    μ   = 0.0
    σᵤ² = @. exp(w_pre)
    σᵥ² = @. exp(v_pre)    
    σ²  = σᵤ² + σᵥ²
    ϵ   = PorC*(Y - x_pre)
    μₛ  = @. (σᵥ² * μ - σᵤ² * ϵ) / σ²
    σₛ  = @. sqrt((σᵥ² * σᵤ²) / σ²)

   jlms = @. (σₛ * normpdf(μₛ / σₛ )) / normcdf(μₛ / σₛ) + μₛ
     bc = @. exp( -μₛ + 0.5 * (σₛ)^2  ) * ( normcdf( (μₛ / σₛ) - σₛ ) / normcdf(μₛ / σₛ)  )

   return jlms, bc  
end



#? ------------------ Exponential model -------------


function jlmsbc(::Type{Expo}, PorC::Int64, pos::NamedTuple, coef::Array{Float64, 1}, 
                Y::Matrix, X::Matrix, z, q, W::Matrix, V::Matrix, # dum1, 
                dum2)

    x_pre = X*coef[pos.begx : pos.endx]
  # z_pre = Z*coef[pos.begz : pos.endz]  # mu
  # q_pre = Q*coef[pos.begq : pos.endq]
    w_pre = W*coef[pos.begw : pos.endw] # log_σᵤ²
    v_pre = V*coef[pos.begv : pos.endv] # log_σᵥ²

    λ   = @. exp(0.5*w_pre)
    σᵥ² = @. exp(v_pre)  
    σᵥ  = @. exp(0.5*v_pre) 
    
    ϵ   = PorC*(Y - x_pre)
    μₛ  = (- ϵ) -(σᵥ² ./ λ) # don't know why this line cannot use @.

    jlms = @. (σᵥ * normpdf(μₛ / σᵥ )) / normcdf(μₛ / σᵥ) + μₛ
      bc = @. exp( -μₛ + 0.5 * σᵥ²  ) * ( normcdf( (μₛ / σᵥ) - σᵥ ) / normcdf(μₛ / σᵥ)  )

    return jlms, bc  
end

#? -------------- Scaling Property Model ------------


function jlmsbc(::Type{Trun_Scale}, PorC::Int64, pos::NamedTuple, coef::Array{Float64, 1}, 
                Y::Matrix, X::Matrix, Z::Matrix, Q::Matrix, W::Matrix, V::Matrix, # dum1, 
                dum2)

    x_pre = X*coef[pos.begx : pos.endx]
    z_pre = Z*coef[pos.begz : pos.endz]  # mu
    q_pre = Q*coef[pos.begq : pos.endq]
    w_pre = W*coef[pos.begw : pos.endw] # log_σᵤ²
    v_pre = V*coef[pos.begv : pos.endv] # log_σᵥ²

    μ   = @. (z_pre) * exp(q_pre)             # Q is _cons; μ here is the after-mutiplied-by-scaling function
    σᵤ² = @. exp(w_pre + 2*q_pre)           # The notation of σᵤ² is different from the paper. W is _cons
    σᵥ² = @. exp(v_pre)    
    σ²  = σᵤ² + σᵥ²
    ϵ   = PorC*(Y - x_pre)
    μₛ  = @. (σᵥ² * μ - σᵤ² * ϵ) / σ²
    σₛ  = @. sqrt((σᵥ² * σᵤ²) / σ²)

   jlms = @. (σₛ * normpdf(μₛ / σₛ )) / normcdf(μₛ / σₛ) + μₛ
     bc = @. exp( -μₛ + 0.5 * (σₛ)^2  ) * ( normcdf( (μₛ / σₛ) - σₛ ) / normcdf(μₛ / σₛ)  )

   return jlms, bc  
end

#? ---------- panel FE Wang and Ho (2010 JoE), Half normal -----------

function jlmsbc(::Type{PFEWHH}, PorC::Int64, pos::NamedTuple, coef::Array{Float64, 1}, 
                Y::Vector, X::Matrix, z, Q::Matrix, W::Matrix, V::Matrix, idt::Matrix{Any})
  
    x_pre = X*coef[pos.begx : pos.endx]
  # z_pre = Z*coef[pos.begz : pos.endz]  # mu
    q_pre = Q*coef[pos.begq : pos.endq]
    w_pre =   coef[pos.begw] # log_σᵤ²
    v_pre =   coef[pos.begv] # log_σᵥ²

    ϵ̃   = PorC*(Y - x_pre)  
    h   = exp.(q_pre) 
    σᵤ² = exp(w_pre)
    σᵥ² = exp(v_pre)

    μ = 0.0
    
    nofobs = length(Y)
    N = size(idt,1)

    jlms = zeros(nofobs, 1)
      bc = zeros(nofobs, 1)

    for i in 1:N
      # @views Π⁻  = INVtrM[i]*(1/σᵥ²) 
        @views ind = idt[i,1]
      # @views trM = Matrix(I, idt[i,2], idt[i,2]) .- (1/idt[i,2])
        @views h̃   = sf_demean(h[ind]) # trM*h[ind]
              σₛₛ² = σₛₛ² = 1.0/(h̃'*h̃*(1/σᵥ²) + 1/σᵤ²)  # h̃ is already de-meaned, so skip Π⁻ but mind 1/σᵥ² # 1.0/(h̃'*Π⁻*h̃ + 1/σᵤ²)  
              σₛₛ  = sqrt(σₛₛ²)
        @views μₛₛ = (μ/σᵤ² - ϵ̃[ind]'*h̃*(1/σᵥ²)) * σₛₛ² # (μ/σᵤ² - ϵ̃[ind]'*Π⁻*h̃) * σₛₛ²

        jlms[ind] = @. h[ind] * (μₛₛ + normpdf(μₛₛ/σₛₛ)*σₛₛ/normcdf(μₛₛ/σₛₛ))
        bc[ind]   = @. ((normcdf(μₛₛ/σₛₛ - h[ind]*σₛₛ))/normcdf(μₛₛ/σₛₛ))*exp(-h[ind]*μₛₛ + 0.5*(h[ind]^2)*σₛₛ²)
    end # for_i  

    return jlms, bc  
end

#? ---------- panel FE Wang and Ho (2010 JoE), truncated normal -----------

function jlmsbc(::Type{PFEWHT}, PorC::Int64, pos::NamedTuple, coef::Array{Float64, 1}, 
    Y::Vector, X::Matrix, Z::Matrix, Q::Matrix, W::Matrix, V::Matrix, idt::Matrix{Any})

    x_pre = X*coef[pos.begx : pos.endx]
    z_pre =   coef[pos.begz]  # mu, a scalar
    q_pre = Q*coef[pos.begq : pos.endq]
    w_pre =   coef[pos.begw] # log_σᵤ²
    v_pre =   coef[pos.begv] # log_σᵥ²
    
    ϵ̃   = PorC*(Y - x_pre)  
    μ   = z_pre
    h   = exp.(q_pre) 
    σᵤ² = exp(w_pre)
    σᵥ² = exp(v_pre)
    
    nofobs = length(Y)
         N = size(idt,1)
    
    jlms = zeros(nofobs, 1)
      bc = zeros(nofobs, 1)
    
    for i in 1:N
      # @views Π⁻  = INVtrM[i]*(1/σᵥ²) 
        @views ind = idt[i,1]
      # @views trM = Matrix(I, idt[i,2], idt[i,2]) .- (1/idt[i,2])
        @views h̃   = sf_demean(h[ind]) # trM*h[ind]
              σₛₛ² = 1.0/(h̃'*h̃*(1/σᵥ²) + 1/σᵤ²)  # 1.0/(h̃'*Π⁻*h̃ + 1/σᵤ²)  
              σₛₛ  = sqrt(σₛₛ²)
        @views μₛₛ = (μ/σᵤ² - ϵ̃[ind]'*h̃*(1/σᵥ²)) * σₛₛ² # (μ/σᵤ² - ϵ̃[ind]'*Π⁻*h̃) * σₛₛ²

        jlms[ind] = @. h[ind] * (μₛₛ + normpdf(μₛₛ/σₛₛ)*σₛₛ/normcdf(μₛₛ/σₛₛ))
        bc[ind]   = @. ((normcdf(μₛₛ/σₛₛ - h[ind]*σₛₛ))/normcdf(μₛₛ/σₛₛ))*exp(-h[ind]*μₛₛ + 0.5*(h[ind]^2)*σₛₛ²)
    end # for_i  
    
    return jlms, bc  
end


#? ---------- panel time decay model of Battese and Coelli (1992) -----------

function jlmsbc(::Type{PanDecay}, PorC::Int64, pos::NamedTuple, coef::Array{Float64, 1}, 
  Y::Matrix, X::Matrix, Z::Matrix, Q::Matrix, W::Matrix, V::Matrix, # dum1, 
  idt::Matrix{Any})

  x_pre = X*coef[pos.begx : pos.endx]
  z_pre = Z*coef[pos.begz : pos.endz]  
  q_pre = Q*coef[pos.begq : pos.endq]
  w_pre =   coef[pos.begw] # log_σᵤ²
  v_pre =   coef[pos.begv] # log_σᵥ²
  
  nofobs = length(Y)
  N = size(idt,1)

  σᵤ² = exp(w_pre)    # should be W*log_σᵤ² where W is a _cons =1; make a short cut here
  σᵥ² = exp(v_pre)
  σₛ² = σᵤ² + σᵥ²

  jlms_ui  = zeros(nofobs, 1)
  jlms_uit = zeros(nofobs, 1)
  bc_uit   = zeros(nofobs, 1)
  
  @inbounds for i in 1:N
      @views ind = idt[i,1]
      @views Tᵢ  = idt[i,2] 
      @views ε   = PorC*(Y[ind] - x_pre[ind])
      @views μ   = (z_pre[ind])[1]  # ok becuase it is time-invariant
      @views Gₜ  = exp.(q_pre[ind]) 
      Gε  = Gₜ .* ε
  
      ΣGε  = sum_kbn(Gε)  
      ΣGₜ² = sum_kbn(Gₜ.^2) 
     
      μₛ   = (μ * σᵥ² - ΣGε * σᵤ²) / (σᵥ² + σᵤ² * ΣGₜ²)
      σₛ²  =  σᵥ² * σᵤ² / (σᵥ² + σᵤ² * ΣGₜ²)
      σₛ   = sqrt(σₛ²)
  
             jlms_ui[ind] .= μₛ + σₛ*(normpdf(μₛ/σₛ) / normcdf(μₛ/σₛ))  # identical within the panel
      @views jlms_uit[ind] = Gₜ .* jlms_ui[ind]

      bc_uit[ind] = (normcdf.(μₛ/σₛ .- Gₜ*σₛ  ) ./ normcdf( μₛ/σₛ )) .* exp.(-Gₜ.*μₛ + 0.5*((Gₜ).^2)*σₛ²)

  end

  return jlms_uit, bc_uit
end


#? ---------- panel Kumbhakar 1990 -----------

function jlmsbc(::Type{PanKumb90}, PorC::Int64, pos::NamedTuple, coef::Array{Float64, 1}, 
  Y::Matrix, X::Matrix, Z::Matrix, Q::Matrix, W::Matrix, V::Matrix, # dum1, 
  idt::Matrix{Any})

  x_pre = X*coef[pos.begx : pos.endx]
  z_pre = Z*coef[pos.begz : pos.endz]  
  q_pre = Q*coef[pos.begq : pos.endq]
  w_pre =   coef[pos.begw] # log_σᵤ²
  v_pre =   coef[pos.begv] # log_σᵥ²
  
  nofobs = length(Y)
  N = size(idt,1)

  σᵤ² = exp(w_pre)    # should be W*log_σᵤ² where W is a _cons =1; make a short cut here
  σᵥ² = exp(v_pre)
  σₛ² = σᵤ² + σᵥ²

  jlms_ui  = zeros(nofobs, 1)
  jlms_uit = zeros(nofobs, 1)
  bc_uit   = zeros(nofobs, 1)
  
  @inbounds for i in 1:N
      @views ind = idt[i,1]
      @views Tᵢ  = idt[i,2] 
      @views ε   = PorC*(Y[ind] - x_pre[ind])
      @views μ   = (z_pre[ind])[1]  # ok becuase it is time-invariant
      @views Gₜ  = 2 ./ (1 .+ exp.(q_pre[ind])  ) # Gₜ  = exp.(q_pre[ind]) 
      Gε  = Gₜ .* ε
  
      ΣGε  = sum_kbn(Gε)  
      ΣGₜ² = sum_kbn(Gₜ.^2) 
     
      μₛ   = (μ * σᵥ² - ΣGε * σᵤ²) / (σᵥ² + σᵤ² * ΣGₜ²)
      σₛ²  =  σᵥ² * σᵤ² / (σᵥ² + σᵤ² * ΣGₜ²)
      σₛ   = sqrt(σₛ²)
  
             jlms_ui[ind] .= μₛ + σₛ*(normpdf(μₛ/σₛ) / normcdf(μₛ/σₛ))  # identical within the panel
      @views jlms_uit[ind] = Gₜ .* jlms_ui[ind]

      bc_uit[ind] = (normcdf.(μₛ/σₛ .- Gₜ*σₛ  ) ./ normcdf( μₛ/σₛ )) .* exp.(-Gₜ.*μₛ + 0.5*((Gₜ).^2)*σₛ²)

  end

  return jlms_uit, bc_uit
end





#? ---------- panel FE CSW (JoE 2014), Half normal -----------

function jlmsbc(::Type{PFECSWH}, PorC::Int64, pos::NamedTuple, coef::Array{Float64, 1}, 
  Y::Vector, X::Matrix, z, q, w, v, idt::Matrix{Any})


   nofobs = length(Y)
   N = size(idt,1)
  #=
    for i in 1:N
        ind = idt[i,1]
        Y[ind] = Y[ind] .+ idt[i,3][1]  # add back the mean for the panel
        X[ind] = X[ind] .+ idt[i,3][2:end]
    end 
  =#   

    x_pre = X*coef[pos.begx : pos.endx]
    # z_pre = Z*coef[pos.begz : pos.endz]  # mu
    # q_pre = Q*coef[pos.begq : pos.endq]
    w_pre =   coef[pos.begw] # log_σᵤ²
    v_pre =   coef[pos.begv] # log_σᵥ²

    αϵ   = (Y - x_pre)  # αᵢ + ϵᵢ
    σᵤ² = exp(w_pre)
    σᵥ² = exp(v_pre)
    σᵤ = exp(0.5*w_pre)

    μ = 0.0

    jlms = zeros(nofobs, 1)
    bc = zeros(nofobs, 1)

    for i in 1:N   

        @views ind = idt[i,1]
        @views αᵢᴹ = mean(αϵ[ind]) + PorC*( sqrt(2/π)*σᵤ )
        @views ϵ   = αϵ[ind] .- αᵢᴹ
               μₛ  = -(σᵤ²*ϵ) ./ (σᵤ² + σᵥ²)
               σₛ² = (σᵤ² * σᵥ²)/(σᵤ² + σᵥ²)
               r   =  μₛ ./ sqrt(σₛ²)
         jlms[ind] =  (sqrt.(σₛ²) * normpdf.(r) ./ normcdf.(r)) .+μₛ 
         bc[ind]   =  exp.( -μₛ .+ 0.5 .* σₛ²  ) .* ( normcdf.( r .- sqrt.(σₛ²) ) ./ normcdf.(r)  )
   
    end # for_i  

    return jlms, bc  
end

#? ----- Panel TRE, Half Normal --------------

function jlmsbc(::Type{PTREH}, PorC::Int64, pos::NamedTuple, coef::Array{Float64, 1}, 
  Y::Matrix, X::Matrix, z, q, w, v, idt::Matrix{Any})

   #! only BC, no JLMS


    nofobs = length(Y)

    β   = coef[1:pos.endx]

    σₐ² = exp(coef[pos.begq])
    σᵤ² = exp(coef[pos.begw])  
    σᵥ² = exp(coef[pos.begv])  

    σᵤ  = exp(0.5*coef[pos.begw])
    σᵥ  = exp(0.5*coef[pos.begv])

    ε   = PorC*(Y - X * β)


  # μ   = (z[ind, :] * δ) #! truncated normal
    μ   = zeros(nofobs, 1)

    bc  = zeros(nofobs, 1)
    

    nofid = size(idt, 1)

    for i in 1:nofid
        @views ind  = idt[i, 1]
        @views T    = idt[i, 2]

              Σ     = ones(T,T)*σₐ² + Matrix(σᵥ²*I,T,T)
              Σᵤ    = Matrix(σᵤ²*I,T,T)
              invΣ  = (1/σᵥ²) * (Matrix(I,T,T) - σₐ²/(σᵥ² + T*σₐ²)*ones(T,T))
              invΣᵤ = (1/σᵤ²) * Matrix(I,T,T) 
        @views K    = ε[ind]'*invΣ - μ[ind]'*invΣᵤ

      # analytic form of G = inv(inv(Σ)+inv(Σᵤ))
        γ²   = 1/(1/σᵤ²+1/σᵥ²)
        g    = -T*1/(1+σₐ²/σᵥ²*T)*σₐ²/σᵥ²^2*1/(1/σᵤ²+1/σᵥ²) 
        ρ²   = 1/(1+g) * 1/(1/σᵤ²+1/σᵥ²)^2 * 1/(1 + σₐ²/σᵥ²*T) *(σₐ²/σᵥ²^2)
        G    = γ² * Matrix(I,T,T) + ρ² * ones(T,T)
        
        B    = (-K*G)'
        γ    = sqrt(γ²)
        ρ    = sqrt(ρ²)


        MCDF1 = quadgk(u -> normpdf(u)*prod(normcdf.((B.-ρ*u)/γ)), -Inf, Inf, rtol=1e-8)

        h1 = MCDF1[1]*exp(0.5*(B'*(invΣ + invΣᵤ)*B))

        bc_i = zeros(T)
        for j in 1:T
           l_T    = zeros(T)
           l_T[j] = 1.0
           K2 = K + l_T'
           B2 = (-K2*G)'
           MCDF2 = quadgk(u -> normpdf(u)*prod(normcdf.((B2.-ρ*u)/γ)), -Inf, Inf, rtol=1e-8)
           h2 = MCDF2[1]*exp(0.5*(B2'*(invΣ + invΣᵤ)*B2))
           bc_i[j] = h2/h1
        end
           bc[ind] = bc_i

      end
           jlms = 0

    return jlms, bc  
end


#? ----- Panel TRE, truncated normal ---------------

function jlmsbc(::Type{PTRET}, PorC::Int64, pos::NamedTuple, coef::Array{Float64, 1}, 
  Y::Matrix, X::Matrix, Z::Matrix, q, w, v, idt::Matrix{Any})

    #! only bc, no jlms

    nofobs = length(Y)

    β   = coef[1 : pos.endx]
    δ   = coef[pos.begz : pos.endz]

    σₐ² = exp(coef[pos.begq])
    σᵤ² = exp(coef[pos.begw])  
    σᵥ² = exp(coef[pos.begv])  

    ε   = PorC*(Y - X * β)

    μ   = (Z * δ) 

    bc  = zeros(nofobs, 1)
    

    nofid = size(idt, 1)

    for i in 1:nofid
        @views ind  = idt[i, 1]
        @views T    = idt[i, 2]
              invΣ  = (1/σᵥ²) * (Matrix(I,T,T) - σₐ²/(σᵥ² + T*σₐ²)*ones(T,T))
              invΣᵤ = (1/σᵤ²) * Matrix(I,T,T) 
        @views K     = ε[ind]'*invΣ - μ[ind]'*invΣᵤ

      # the analytic form of G = inv(inv(Σ)+inv(Σᵤ))
        γ²   = 1/(1/σᵤ²+1/σᵥ²)
        g    = -T*1/(1+σₐ²/σᵥ²*T)*σₐ²/σᵥ²^2*1/(1/σᵤ²+1/σᵥ²) 
        ρ²   = 1/(1+g) * 1/(1/σᵤ²+1/σᵥ²)^2 * 1/(1 + σₐ²/σᵥ²*T) *(σₐ²/σᵥ²^2)
        G    = γ² * Matrix(I,T,T) + ρ² * ones(T,T)
        
        B    = (-K*G)'
        γ    = sqrt(γ²)
        ρ    = sqrt(ρ²)

        MCDF1 = quadgk(u -> normpdf(u)*prod(normcdf.((B.-ρ*u)/γ)), -Inf, Inf, rtol=1e-8)

        h1 = MCDF1[1]*exp(0.5*(B'*(invΣ + invΣᵤ)*B))

        bc_i = zeros(T)
        for j in 1:T
           l_T    = zeros(T)
           l_T[j] = 1.0
           K2 = K + l_T'
           B2 = (-K2*G)'
           MCDF2 = quadgk(u -> normpdf(u)*prod(normcdf.((B2.-ρ*u)/γ)), -Inf, Inf, rtol=1e-8)
           h2 = MCDF2[1]*exp(0.5*(B2'*(invΣ + invΣᵤ)*B2))
           bc_i[j] = h2/h1
        end
           bc[ind] = bc_i

      end
           jlms = 0

    return jlms, bc  
end
