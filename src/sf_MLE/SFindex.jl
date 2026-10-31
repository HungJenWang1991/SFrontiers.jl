# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

#########################################
####        JLMS and BC index        ####
#########################################

# With r = μₛ/σₛ the closed forms below are JLMS = σₛ·m(r), m(r) = r + φ(r)/Φ(r), and
# log BC = -μₛ + σₛ²/2 + log Φ(r - σₛ) - log Φ(r) (h·σₛ in place of σₛ where the model scales u
# by h; σᵥ in place of σₛ for Expo).
# Computed directly, as σₛ·exp(normlogpdf(r) - normlogcdf(r)) + μₛ and as that sum of four
# terms, both cancel as r → -∞ (μ < 0 with σᵤ → 0, or ϵ ≫ 0 with σᵥ → 0): the relative
# error of JLMS grows like eps·r⁴/4 (1.5e-10 near r = -40, 5e-5 at -1e3, 13% at -1e4, then
# any sign), and log BC carries an absolute error of about eps·r², so BC can exceed 1. For
# r < -40 they are therefore σₛ·sf_tn_mean(r) and exp(sf_tn_logmgf(r, σₛ)) (src/sf_numerics.jl),
# exact because r·σₛ = μₛ; there JLMS > 0 and 0 < BC ≤ 1. For r ≥ -40 the direct
# expressions are evaluated.

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
   # r < -40 takes the stable forms (see the note at the top of this file)
   jlms = @. ifelse(r < -40, σₛ * sf_tn_mean(r), σₛ * exp(normlogpdf(r) - normlogcdf(r)) + μₛ)
     bc = @. ifelse(r < -40, exp(sf_tn_logmgf(r, σₛ)),
                    exp(clamp(-μₛ + 0.5 * σₛ^2 + normlogcdf(r - σₛ) - normlogcdf(r), -500.0, 500.0)))

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
   # r < -40 takes the stable forms (see the note at the top of this file)
   jlms = @. ifelse(r < -40, σₛ * sf_tn_mean(r), σₛ * exp(normlogpdf(r) - normlogcdf(r)) + μₛ)
     bc = @. ifelse(r < -40, exp(sf_tn_logmgf(r, σₛ)),
                    exp(clamp(-μₛ + 0.5 * σₛ^2 + normlogcdf(r - σₛ) - normlogcdf(r), -500.0, 500.0)))

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

    # σᵤ is the exponential's scale, sqrt of the squared scale the w-block carries
    σᵤ  = @. sqrt(clamp(exp(w_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi))
    σᵥ² = @. clamp(exp(v_pre), _MLE_CLAMP.exp_lo, _MLE_CLAMP.exp_hi)
    σᵥ  = @. sqrt(σᵥ²)

    ϵ   = PorC*(Y - x_pre)
    μₛ  = (- ϵ) -(σᵥ² ./ σᵤ) # don't know why this line cannot use @.

    r    = @. μₛ / σᵥ
    # r < -40 takes the stable forms (see the note at the top of this file)
    jlms = @. ifelse(r < -40, σᵥ * sf_tn_mean(r), σᵥ * exp(normlogpdf(r) - normlogcdf(r)) + μₛ)
      bc = @. ifelse(r < -40, exp(sf_tn_logmgf(r, σᵥ)),
                     exp(clamp(-μₛ + 0.5 * σᵥ² + normlogcdf(r - σᵥ) - normlogcdf(r), -500.0, 500.0)))

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
   # r < -40 takes the stable forms (see the note at the top of this file)
   jlms = @. ifelse(r < -40, σₛ * sf_tn_mean(r), σₛ * exp(normlogpdf(r) - normlogcdf(r)) + μₛ)
     bc = @. ifelse(r < -40, exp(sf_tn_logmgf(r, σₛ)),
                    exp(clamp(-μₛ + 0.5 * σₛ^2 + normlogcdf(r - σₛ) - normlogcdf(r), -500.0, 500.0)))

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
        if r < -40   # the stable forms (see the note at the top of this file)
            m_r = sf_tn_mean(r)
            jlms[ind] = @. h[ind] * (σₛₛ * m_r)
            bc[ind]   = @. exp(sf_tn_logmgf(r, h[ind]*σₛₛ))
        else
            mills = exp(normlogpdf(r) - normlogcdf(r))
            jlms[ind] = @. h[ind] * (μₛₛ + mills*σₛₛ)
            bc[ind]   = @. exp(clamp(-h[ind]*μₛₛ + 0.5*(h[ind]^2)*σₛₛ² + normlogcdf(r - h[ind]*σₛₛ) - normlogcdf(r), -500.0, 500.0))
        end
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
        if r < -40   # the stable forms (see the note at the top of this file)
            m_r = sf_tn_mean(r)
            jlms[ind] = @. h[ind] * (σₛₛ * m_r)
            bc[ind]   = @. exp(sf_tn_logmgf(r, h[ind]*σₛₛ))
        else
            mills = exp(normlogpdf(r) - normlogcdf(r))
            jlms[ind] = @. h[ind] * (μₛₛ + mills*σₛₛ)
            bc[ind]   = @. exp(clamp(-h[ind]*μₛₛ + 0.5*(h[ind]^2)*σₛₛ² + normlogcdf(r - h[ind]*σₛₛ) - normlogcdf(r), -500.0, 500.0))
        end
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
      if r < -40   # the stable forms (see the note at the top of this file)
             jlms_ui[ind] .= σₛ*sf_tn_mean(r)
      @views jlms_uit[ind] = Gₜ .* jlms_ui[ind]
             bc_uit[ind] = exp.(sf_tn_logmgf.(r, Gₜ*σₛ))
      else
      mills = exp(normlogpdf(r) - normlogcdf(r))
             jlms_ui[ind] .= μₛ + σₛ*mills
      @views jlms_uit[ind] = Gₜ .* jlms_ui[ind]

      bc_uit[ind] = exp.(clamp.(-Gₜ.*μₛ .+ 0.5*((Gₜ).^2)*σₛ² .+ normlogcdf.(r .- Gₜ*σₛ) .- normlogcdf(r), -500.0, 500.0))
      end

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
      if r < -40   # the stable forms (see the note at the top of this file)
             jlms_ui[ind] .= σₛ*sf_tn_mean(r)
      @views jlms_uit[ind] = Gₜ .* jlms_ui[ind]
             bc_uit[ind] = exp.(sf_tn_logmgf.(r, Gₜ*σₛ))
      else
      mills = exp(normlogpdf(r) - normlogcdf(r))
             jlms_ui[ind] .= μₛ + σₛ*mills
      @views jlms_uit[ind] = Gₜ .* jlms_ui[ind]

      bc_uit[ind] = exp.(clamp.(-Gₜ.*μₛ .+ 0.5*((Gₜ).^2)*σₛ² .+ normlogcdf.(r .- Gₜ*σₛ) .- normlogcdf(r), -500.0, 500.0))
      end

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
        @views ϵ   = PorC .* (αϵ[ind] .- αᵢᴹ)   # ε in v−u form for both types (no-op for :prod)
               μₛ  = -(σᵤ²*ϵ) ./ (σᵤ² + σᵥ²)
               σₛ² = (σᵤ² * σᵥ²)/(σᵤ² + σᵥ²)
               r   =  μₛ ./ sqrt(σₛ²)
         # r < -40 takes the stable forms (see the note at the top of this file)
         jlms[ind] =  ifelse.(r .< -40, sqrt.(σₛ²) .* sf_tn_mean.(r),
                              sqrt.(σₛ²) .* exp.(normlogpdf.(r) .- normlogcdf.(r)) .+ μₛ)
         bc[ind]   =  ifelse.(r .< -40, exp.(sf_tn_logmgf.(r, sqrt.(σₛ²))),
                              exp.(clamp.(-μₛ .+ 0.5 .* σₛ² .+ normlogcdf.( r .- sqrt.(σₛ²) ) .- normlogcdf.(r), -500.0, 500.0)))

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
        # Cancellation-free Sherman-Morrison form, identical to LL_T(PTREH) in
        # SFloglikefun.jl so the index is built from the same G the likelihood used.
        c    = 1/σᵤ² + 1/σᵥ²
        cTd  = 1/σᵤ² + 1/(σᵥ² + T*σₐ²)
        γ²   = 1/c
        ρ²   = σₐ²/(σᵥ²*(σᵥ² + T*σₐ²)) / (c*cTd)
        G    = γ² * Matrix(I,T,T) + ρ² * ones(T,T)

        B    = (-K*G)'
        γ    = sqrt(γ²)
        ρ    = sqrt(ρ²)


        # Every number this function reports is a RATIO of orthant probabilities, so all
        # three are formed in log space; see _tre_logorthant in SFloglikefun.jl.  Neither h1
        # nor h2 survives on its own -- MCDF underflows to 0, so it would need a floor such
        # as 1e-15, and the exponent of the quadratic form a clamp at ±500 -- but both faults
        # cancel out of the ratio once it is taken in logs.  With that floor, on a T = 5 firm
        # sitting four units on the inefficient side of the frontier, the floor reaches
        # MCDF2 before MCDF1 and bc comes out 2.86 for a true 0.906: an efficiency above 1,
        # which E[exp(-u)] with u ≥ 0 cannot be.  Two units further out it is 7.42 for a
        # true 0.930.  Once both sit on the floor and the exponent clamp saturates as well,
        # bc is exactly 1 and jlms exactly 0 -- a firm the model thinks impossible, reported
        # as fully efficient.
        #
        # The exponents of h1 and h2 need not be formed at all.  B2 = B - G e_j and
        # (Σ⁻¹+Σᵤ⁻¹)G = I, so ½B2'(Σ⁻¹+Σᵤ⁻¹)B2 - ½B'(Σ⁻¹+Σᵤ⁻¹)B is exactly -B_j + ½G_jj.
        # As two quadratic forms they are each of order 1/σᵥ² or 1/σᵤ² and their difference
        # is O(1): at σᵥ² = 2.5e-12 it comes out 1291 wrong, and bc with it.  As -B_j + ½G_jj
        # it is right to 4e-7 there and to rounding everywhere else.
        # A firm with r·√T > 100, r = ρ/γ
        # (_tre_past_switch in SFloglikefun.jl, the switch of LL_T(PTRET), which every firm
        # whose integral LL_T(PTREH) takes piecewise also passes), gets jlms and bc from the
        # piecewise branches of the helpers, with B formed again without the cancellation of
        # its 1/σᵥ²-sized terms (_tre_Bx) and the firm's integral computed once (Px), and bc's
        # log-probability difference is taken in one piece, as in jlmsbc(PTRET) below: formed
        # as two logarithms from two separately rounded B's it was off by up to 4e-5 relative
        # at the σᵥ² bound. Every other firm uses B with _tre_logorthant and _tre_orthant_jlms.
        past     = _tre_past_switch(ρ, γ, T)
        Bx       = past ? _tre_Bx(ε[ind], μ[ind], σₐ², σᵤ², σᵥ², γ², ρ², T) : B
        Px       = past ? _tre_orthant_x(Bx, ρ, γ) : nothing
        logMCDF1 = past ? NaN : _tre_logorthant(B, ρ, γ)        # not needed past the switch

        # JLMS: E[u_j | ε, u ≥ 0] via 1-d quadrature over the common factor.  The numerator
        # integrand φ(u)γ(a_jΦ(a_j) + φ(a_j))∏_{k≠j}Φ(a_k) is φ(u)γ·m(a_j)·∏ₖΦ(aₖ), so it
        # shares the denominator's integrand and the ratio needs no floor at all.
        jlms_i = past ? _tre_orthant_jlms_x(Bx, ρ, γ, Px) : _tre_orthant_jlms(B, ρ, γ)
        jlms[ind] = jlms_i

        # BC: E[exp(-u_j) | ε, u ≥ 0]
        bc_i = zeros(T)
        for j in 1:T
           if past
               bc_i[j] = exp(_tre_logorthant_diff_x(Bx, ρ, γ, -G[:, j], Px) - Bx[j] + 0.5*G[j,j])
               continue
           end
           l_T    = zeros(T)
           l_T[j] = 1.0
           K2 = K + l_T'
           B2 = (-K2*G)'
           bc_i[j] = exp(_tre_logorthant(B2, ρ, γ) - logMCDF1 - B[j] + 0.5*G[j,j])
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
        # Cancellation-free Sherman-Morrison form, identical to LL_T(PTREH) in
        # SFloglikefun.jl so the index is built from the same G the likelihood used.
        c    = 1/σᵤ² + 1/σᵥ²
        cTd  = 1/σᵤ² + 1/(σᵥ² + T*σₐ²)
        γ²   = 1/c
        ρ²   = σₐ²/(σᵥ²*(σᵥ² + T*σₐ²)) / (c*cTd)
        G    = γ² * Matrix(I,T,T) + ρ² * ones(T,T)

        B    = (-K*G)'
        γ    = sqrt(γ²)
        ρ    = sqrt(ρ²)

        # Every number this function reports is a RATIO of orthant probabilities: jlms
        # shares one integrand between numerator and denominator, and bc's ratio is taken
        # by _tre_logorthant_diff in SFloglikefun.jl without forming either logarithm.
        # Neither h1 nor h2 survives on its own -- MCDF underflows to 0, so it would need a
        # floor such as 1e-15, and the exponent of the quadratic form a clamp at ±500 -- but
        # both faults cancel out of the ratio.  With that floor, on a T = 5 firm
        # sitting four units on the inefficient side of the frontier, the floor reaches
        # MCDF2 before MCDF1 and bc comes out 2.86 for a true 0.906: an efficiency above 1,
        # which E[exp(-u)] with u ≥ 0 cannot be.  Two units further out it is 7.42 for a
        # true 0.930.  Once both sit on the floor and the exponent clamp saturates as well,
        # bc is exactly 1 and jlms exactly 0 -- a firm the model thinks impossible, reported
        # as fully efficient.
        #
        # The exponents of h1 and h2 need not be formed at all.  B2 = B - G e_j and
        # (Σ⁻¹+Σᵤ⁻¹)G = I, so ½B2'(Σ⁻¹+Σᵤ⁻¹)B2 - ½B'(Σ⁻¹+Σᵤ⁻¹)B is exactly -B_j + ½G_jj.
        # As two quadratic forms they are each of order 1/σᵥ² or 1/σᵤ² and their difference
        # is O(1): at σᵥ² = 2.5e-12 it comes out 1291 wrong, and bc with it.  As -B_j + ½G_jj
        # it is right to 4e-7 there and to rounding everywhere else.
        # A firm with r·√T > 100 (_tre_past_switch in SFloglikefun.jl, the switch of LL_T(PTRET))
        # gets jlms and bc from the piecewise branches of the helpers, with B formed again without
        # the cancellation of its 1/σᵥ²-sized terms (_tre_Bx) and the firm's integral computed once
        # (Px); every other firm uses B with _tre_orthant_jlms and _tre_logorthant_diff.
        past = _tre_past_switch(ρ, γ, T)
        Bx   = past ? _tre_Bx(ε[ind], μ[ind], σₐ², σᵤ², σᵥ², γ², ρ², T) : B
        Px   = past ? _tre_orthant_x(Bx, ρ, γ) : nothing
        # JLMS: E[u_j | ε, u ≥ 0] via 1-d quadrature over the common factor.  The numerator
        # integrand φ(u)γ(a_jΦ(a_j) + φ(a_j))∏_{k≠j}Φ(a_k) is φ(u)γ·m(a_j)·∏ₖΦ(aₖ), so it
        # shares the denominator's integrand and the ratio needs no floor at all.
        jlms_i = past ? _tre_orthant_jlms_x(Bx, ρ, γ, Px) : _tre_orthant_jlms(B, ρ, γ)
        jlms[ind] = jlms_i

        # BC: E[exp(-u_j) | ε, u ≥ 0] = exp(logP₂ - logP₁ - B_j + ½G_jj), B₂ = B - Ge_j.
        # logP₂ and logP₁ each carry the same ∑ₜlog Φ(μₜ/σᵤ)-sized bulk, about -5.9e14 at
        # the σᵤ² clamp, that cancels in the difference: formed apart the difference is
        # quantised at 0.06 and bc came out 1.101 for three of forty observations, where
        # E[exp(-u)] with u ≥ 0 cannot exceed 1.  _tre_logorthant_diff takes the
        # difference with the exact shift S = -Ge_j (a₂ₜ - a₁ₜ = Sₜ/γ exactly) and never
        # forms either log; see SFloglikefun.jl.
        bc_i = zeros(T)
        for j in 1:T
           bc_i[j] = past ? exp(_tre_logorthant_diff_x(Bx, ρ, γ, -G[:, j], Px) - Bx[j] + 0.5*G[j,j]) :
                            exp(_tre_logorthant_diff(B, ρ, γ, -G[:, j]) - B[j] + 0.5*G[j,j])
        end
        bc[ind] = bc_i

      end

    return jlms, bc
end
