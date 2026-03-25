# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

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
    σᵤ² = @. clamp(exp(w_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵤ  = @. sqrt(σᵤ²)
    σᵥ² = @. clamp(exp(v_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σ²  = σᵤ² + σᵥ²
    ϵ   = PorC*(Y - x_pre)
    μₛ  = @. (σᵥ² * μ - σᵤ² * ϵ) / σ²
    σₛ  = @. sqrt((σᵥ² * σᵤ²) / σ²)

   r    = @. μₛ / σₛ
   jlms = @. σₛ * exp(normlogpdf(r) - normlogcdf(r)) + μₛ
     bc = @. exp(clamp(-μₛ + 0.5 * σₛ^2 + normlogcdf(r - σₛ) - normlogcdf(r), -500.0, 500.0))

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
    σᵤ² = @. clamp(exp(w_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵥ² = @. clamp(exp(v_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σ²  = σᵤ² + σᵥ²
    ϵ   = PorC*(Y - x_pre)
    μₛ  = @. (σᵥ² * μ - σᵤ² * ϵ) / σ²
    σₛ  = @. sqrt((σᵥ² * σᵤ²) / σ²)

   r    = @. μₛ / σₛ
   jlms = @. σₛ * exp(normlogpdf(r) - normlogcdf(r)) + μₛ
     bc = @. exp(clamp(-μₛ + 0.5 * σₛ^2 + normlogcdf(r - σₛ) - normlogcdf(r), -500.0, 500.0))

   return jlms, bc
end



#? ------------------ Exponential model -------------


function jlmsbc(::Type{Expo}, PorC::Int64, pos::NamedTuple, coef::Array{Float64, 1},
                Y::Matrix, X::Matrix, z, q, W::Matrix, V::Matrix, # dum1,
                dum2)

    x_pre = X*coef[pos.begx : pos.endx]
  # z_pre = Z*coef[pos.begz : pos.endz]  # mu
  # q_pre = Q*coef[pos.begq : pos.endq]
    w_pre = W*coef[pos.begw : pos.endw] # log_λ
    v_pre = V*coef[pos.begv : pos.endv] # log_σᵥ²

    λ   = @. sqrt(clamp(exp(w_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi))
    σᵥ² = @. clamp(exp(v_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵥ  = @. sqrt(σᵥ²)

    ϵ   = PorC*(Y - x_pre)
    μₛ  = (- ϵ) -(σᵥ² ./ λ) # don't know why this line cannot use @.

    r    = @. μₛ / σᵥ
    jlms = @. σᵥ * exp(normlogpdf(r) - normlogcdf(r)) + μₛ
      bc = @. exp(clamp(-μₛ + 0.5 * σᵥ² + normlogcdf(r - σᵥ) - normlogcdf(r), -500.0, 500.0))

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

    hscale = @. clamp(exp(q_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    μ   = @. (z_pre) * hscale             # Q is _cons; μ here is the after-mutiplied-by-scaling function
    σᵤ² = @. clamp(exp(w_pre) * hscale^2, _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)  # The notation of σᵤ² is different from the paper. W is _cons
    σᵥ² = @. clamp(exp(v_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σ²  = σᵤ² + σᵥ²
    ϵ   = PorC*(Y - x_pre)
    μₛ  = @. (σᵥ² * μ - σᵤ² * ϵ) / σ²
    σₛ  = @. sqrt((σᵥ² * σᵤ²) / σ²)

   r    = @. μₛ / σₛ
   jlms = @. σₛ * exp(normlogpdf(r) - normlogcdf(r)) + μₛ
     bc = @. exp(clamp(-μₛ + 0.5 * σₛ^2 + normlogcdf(r - σₛ) - normlogcdf(r), -500.0, 500.0))

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
    h   = clamp.(exp.(q_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵤ² = clamp(exp(w_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵥ² = clamp(exp(v_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)

    μ = 0.0

    nofobs = length(Y)
    N = size(idt,1)

    jlms = zeros(nofobs, 1)
      bc = zeros(nofobs, 1)

    for i in 1:N
        @views ind = idt[i,1]
        @views h̃   = sf_demean(h[ind])
              σₛₛ² = 1.0/(h̃'*h̃*(1/σᵥ²) + 1/σᵤ²)
              σₛₛ  = sqrt(σₛₛ²)
        @views μₛₛ = (μ/σᵤ² - ϵ̃[ind]'*h̃*(1/σᵥ²)) * σₛₛ²

        r = μₛₛ/σₛₛ
        mills = exp(normlogpdf(r) - normlogcdf(r))
        jlms[ind] = @. h[ind] * (μₛₛ + mills*σₛₛ)
        bc[ind]   = @. exp(clamp(-h[ind]*μₛₛ + 0.5*(h[ind]^2)*σₛₛ² + normlogcdf(r - h[ind]*σₛₛ) - normlogcdf(r), -500.0, 500.0))
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
    h   = clamp.(exp.(q_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵤ² = clamp(exp(w_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵥ² = clamp(exp(v_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)

    nofobs = length(Y)
         N = size(idt,1)

    jlms = zeros(nofobs, 1)
      bc = zeros(nofobs, 1)

    for i in 1:N
        @views ind = idt[i,1]
        @views h̃   = sf_demean(h[ind])
              σₛₛ² = 1.0/(h̃'*h̃*(1/σᵥ²) + 1/σᵤ²)
              σₛₛ  = sqrt(σₛₛ²)
        @views μₛₛ = (μ/σᵤ² - ϵ̃[ind]'*h̃*(1/σᵥ²)) * σₛₛ²

        r = μₛₛ/σₛₛ
        mills = exp(normlogpdf(r) - normlogcdf(r))
        jlms[ind] = @. h[ind] * (μₛₛ + mills*σₛₛ)
        bc[ind]   = @. exp(clamp(-h[ind]*μₛₛ + 0.5*(h[ind]^2)*σₛₛ² + normlogcdf(r - h[ind]*σₛₛ) - normlogcdf(r), -500.0, 500.0))
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

  σᵤ² = clamp(exp(w_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
  σᵥ² = clamp(exp(v_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
  σₛ² = σᵤ² + σᵥ²

  jlms_ui  = zeros(nofobs, 1)
  jlms_uit = zeros(nofobs, 1)
  bc_uit   = zeros(nofobs, 1)

  @inbounds for i in 1:N
      @views ind = idt[i,1]
      @views Tᵢ  = idt[i,2]
      @views ε   = PorC*(Y[ind] - x_pre[ind])
      @views μ   = (z_pre[ind])[1]  # ok becuase it is time-invariant
      @views Gₜ  = clamp.(exp.(q_pre[ind]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
      Gε  = Gₜ .* ε

      ΣGε  = sum(Gε)
      ΣGₜ² = sum(Gₜ.^2)

      μₛ   = (μ * σᵥ² - ΣGε * σᵤ²) / (σᵥ² + σᵤ² * ΣGₜ²)
      σₛ²  =  σᵥ² * σᵤ² / (σᵥ² + σᵤ² * ΣGₜ²)
      σₛ   = sqrt(σₛ²)

      r = μₛ/σₛ
      mills = exp(normlogpdf(r) - normlogcdf(r))
             jlms_ui[ind] .= μₛ + σₛ*mills
      @views jlms_uit[ind] = Gₜ .* jlms_ui[ind]

      bc_uit[ind] = exp.(clamp.(-Gₜ.*μₛ .+ 0.5*((Gₜ).^2)*σₛ² .+ normlogcdf.(r .- Gₜ*σₛ) .- normlogcdf(r), -500.0, 500.0))

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

  σᵤ² = clamp(exp(w_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
  σᵥ² = clamp(exp(v_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
  σₛ² = σᵤ² + σᵥ²

  jlms_ui  = zeros(nofobs, 1)
  jlms_uit = zeros(nofobs, 1)
  bc_uit   = zeros(nofobs, 1)

  @inbounds for i in 1:N
      @views ind = idt[i,1]
      @views Tᵢ  = idt[i,2]
      @views ε   = PorC*(Y[ind] - x_pre[ind])
      @views μ   = (z_pre[ind])[1]  # ok becuase it is time-invariant
      @views Gₜ  = 2 ./ (1 .+ clamp.(exp.(q_pre[ind]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi))
      Gε  = Gₜ .* ε

      ΣGε  = sum(Gε)
      ΣGₜ² = sum(Gₜ.^2)

      μₛ   = (μ * σᵥ² - ΣGε * σᵤ²) / (σᵥ² + σᵤ² * ΣGₜ²)
      σₛ²  =  σᵥ² * σᵤ² / (σᵥ² + σᵤ² * ΣGₜ²)
      σₛ   = sqrt(σₛ²)

      r = μₛ/σₛ
      mills = exp(normlogpdf(r) - normlogcdf(r))
             jlms_ui[ind] .= μₛ + σₛ*mills
      @views jlms_uit[ind] = Gₜ .* jlms_ui[ind]

      bc_uit[ind] = exp.(clamp.(-Gₜ.*μₛ .+ 0.5*((Gₜ).^2)*σₛ² .+ normlogcdf.(r .- Gₜ*σₛ) .- normlogcdf(r), -500.0, 500.0))

  end

  return jlms_uit, bc_uit
end





#? ---------- panel FE CSW (JoE 2014), Half normal -----------

function jlmsbc(::Type{PFECSWH}, PorC::Int64, pos::NamedTuple, coef::Array{Float64, 1},
  Y::Vector, X::Matrix, z, q, w, v, idt::Matrix{Any})


   nofobs = length(Y)
   N = size(idt,1)

    x_pre = X*coef[pos.begx : pos.endx]
    w_pre =   coef[pos.begw] # log_σᵤ²
    v_pre =   coef[pos.begv] # log_σᵥ²

    αϵ   = (Y - x_pre)  # αᵢ + ϵᵢ
    σᵤ² = clamp(exp(w_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵥ² = clamp(exp(v_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵤ = sqrt(σᵤ²)

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
         jlms[ind] =  sqrt.(σₛ²) .* exp.(normlogpdf.(r) .- normlogcdf.(r)) .+ μₛ
         bc[ind]   =  exp.(clamp.(-μₛ .+ 0.5 .* σₛ² .+ normlogcdf.( r .- sqrt.(σₛ²) ) .- normlogcdf.(r), -500.0, 500.0))

    end # for_i

    return jlms, bc
end

#? ----- Panel TRE, Half Normal --------------

function jlmsbc(::Type{PTREH}, PorC::Int64, pos::NamedTuple, coef::Array{Float64, 1},
  Y::Matrix, X::Matrix, z, q, w, v, idt::Matrix{Any})

    nofobs = length(Y)

    β   = coef[1:pos.endx]

    σₐ² = clamp(exp(coef[pos.begq]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵤ² = clamp(exp(coef[pos.begw]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵥ² = clamp(exp(coef[pos.begv]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)

    σᵤ  = sqrt(σᵤ²)
    σᵥ  = sqrt(σᵥ²)

    ε   = PorC*(Y - X * β)


  # μ   = (z[ind, :] * δ) #! truncated normal
    μ   = zeros(nofobs, 1)

    jlms = zeros(nofobs, 1)
    bc   = zeros(nofobs, 1)


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
        γ²   = max(1/(1/σᵤ²+1/σᵥ²), _MLE_CLAMP.exp_lo)
        g    = -T*1/(1+σₐ²/σᵥ²*T)*σₐ²/σᵥ²^2*1/(1/σᵤ²+1/σᵥ²)
        ρ²   = max(1/(1+g) * 1/(1/σᵤ²+1/σᵥ²)^2 * 1/(1 + σₐ²/σᵥ²*T) *(σₐ²/σᵥ²^2), zero(eltype(Y)))
        G    = γ² * Matrix(I,T,T) + ρ² * ones(T,T)

        B    = (-K*G)'
        γ    = sqrt(γ²)
        ρ    = sqrt(ρ²)


        MCDF1 = quadgk(u -> normpdf(u)*prod(normcdf.(clamp.((B.-ρ*u)/γ, -500.0, 500.0))), -Inf, Inf, rtol=1e-8)

        h1 = max(MCDF1[1], _MLE_CLAMP.log_lo)*exp(clamp(0.5*(B'*(invΣ + invΣᵤ)*B), -500.0, 500.0))

        # JLMS: E[u_j | ε, u ≥ 0] via 1-d quadrature over the common factor
        jlms_i = zeros(T)
        for j in 1:T
           JLMS_int = quadgk(u -> begin
               a = clamp.((B .- ρ*u) ./ γ, -500.0, 500.0)
               normpdf(u) * γ * (a[j]*normcdf(a[j]) + normpdf(a[j])) *
                   prod(normcdf.(a[1:end .!= j]))
           end, -Inf, Inf, rtol=1e-8)
           jlms_i[j] = JLMS_int[1] / max(MCDF1[1], _MLE_CLAMP.log_lo)
        end
        jlms[ind] = jlms_i

        # BC: E[exp(-u_j) | ε, u ≥ 0]
        bc_i = zeros(T)
        for j in 1:T
           l_T    = zeros(T)
           l_T[j] = 1.0
           K2 = K + l_T'
           B2 = (-K2*G)'
           MCDF2 = quadgk(u -> normpdf(u)*prod(normcdf.(clamp.((B2.-ρ*u)/γ, -500.0, 500.0))), -Inf, Inf, rtol=1e-8)
           h2 = max(MCDF2[1], _MLE_CLAMP.log_lo)*exp(clamp(0.5*(B2'*(invΣ + invΣᵤ)*B2), -500.0, 500.0))
           bc_i[j] = h2/h1
        end
        bc[ind] = bc_i

      end

    return jlms, bc
end


#? ----- Panel TRE, truncated normal ---------------

function jlmsbc(::Type{PTRET}, PorC::Int64, pos::NamedTuple, coef::Array{Float64, 1},
  Y::Matrix, X::Matrix, Z::Matrix, q, w, v, idt::Matrix{Any})

    nofobs = length(Y)

    β   = coef[1 : pos.endx]
    δ   = coef[pos.begz : pos.endz]

    σₐ² = clamp(exp(coef[pos.begq]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵤ² = clamp(exp(coef[pos.begw]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵥ² = clamp(exp(coef[pos.begv]), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)

    ε   = PorC*(Y - X * β)

    μ   = (Z * δ)

    jlms = zeros(nofobs, 1)
    bc   = zeros(nofobs, 1)


    nofid = size(idt, 1)

    for i in 1:nofid
        @views ind  = idt[i, 1]
        @views T    = idt[i, 2]
              invΣ  = (1/σᵥ²) * (Matrix(I,T,T) - σₐ²/(σᵥ² + T*σₐ²)*ones(T,T))
              invΣᵤ = (1/σᵤ²) * Matrix(I,T,T)
        @views K     = ε[ind]'*invΣ - μ[ind]'*invΣᵤ

      # the analytic form of G = inv(inv(Σ)+inv(Σᵤ))
        γ²   = max(1/(1/σᵤ²+1/σᵥ²), _MLE_CLAMP.exp_lo)
        g    = -T*1/(1+σₐ²/σᵥ²*T)*σₐ²/σᵥ²^2*1/(1/σᵤ²+1/σᵥ²)
        ρ²   = max(1/(1+g) * 1/(1/σᵤ²+1/σᵥ²)^2 * 1/(1 + σₐ²/σᵥ²*T) *(σₐ²/σᵥ²^2), zero(eltype(Y)))
        G    = γ² * Matrix(I,T,T) + ρ² * ones(T,T)

        B    = (-K*G)'
        γ    = sqrt(γ²)
        ρ    = sqrt(ρ²)

        MCDF1 = quadgk(u -> normpdf(u)*prod(normcdf.(clamp.((B.-ρ*u)/γ, -500.0, 500.0))), -Inf, Inf, rtol=1e-8)

        h1 = max(MCDF1[1], _MLE_CLAMP.log_lo)*exp(clamp(0.5*(B'*(invΣ + invΣᵤ)*B), -500.0, 500.0))

        # JLMS: E[u_j | ε, u ≥ 0] via 1-d quadrature over the common factor
        jlms_i = zeros(T)
        for j in 1:T
           JLMS_int = quadgk(u -> begin
               a = clamp.((B .- ρ*u) ./ γ, -500.0, 500.0)
               normpdf(u) * γ * (a[j]*normcdf(a[j]) + normpdf(a[j])) *
                   prod(normcdf.(a[1:end .!= j]))
           end, -Inf, Inf, rtol=1e-8)
           jlms_i[j] = JLMS_int[1] / max(MCDF1[1], _MLE_CLAMP.log_lo)
        end
        jlms[ind] = jlms_i

        # BC: E[exp(-u_j) | ε, u ≥ 0]
        bc_i = zeros(T)
        for j in 1:T
           l_T    = zeros(T)
           l_T[j] = 1.0
           K2 = K + l_T'
           B2 = (-K2*G)'
           MCDF2 = quadgk(u -> normpdf(u)*prod(normcdf.(clamp.((B2.-ρ*u)/γ, -500.0, 500.0))), -Inf, Inf, rtol=1e-8)
           h2 = max(MCDF2[1], _MLE_CLAMP.log_lo)*exp(clamp(0.5*(B2'*(invΣ + invΣᵤ)*B2), -500.0, 500.0))
           bc_i[j] = h2/h1
        end
        bc[ind] = bc_i

      end

    return jlms, bc
end
