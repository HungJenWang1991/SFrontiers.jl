# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

########################################################
####             equation prediction                ####
########################################################

"""
    sfmodel_predict(@eq(eq_name))

Return predicted values of an equation (`eq_name`) of a stochastic frontier
model specified and estimated by `sfmodel_spec()` and `sfmodel_fit()`.

# Arguments:
- `@eq(eq_name)`: where `eq_name` is the name of the equation to which the
  value is to be predicted. A SF model has various equations, and different
  SF models have different sets of equations. Eligible `eq_name`s follow
  those specified in `sfmodel_spec()`. The `eq_name` may be specified using
  unicodes or their alias names; viz., `@eq(σᵤ²)` is the same as
  `@eq(sigma_u_2)`. Also, `log_σᵤ²` = `log_sigma_u_2`, `σᵥ²`= `sigma_v_2`,
  `log_σᵥ²` = `log_sigma_v_2`, `σₐ²`= `sigma_a_2`, `log_σₐ²` = `log_sigma_a_2`,
  `μ` = `mu`, and (Exponential inefficiency) `λ` = `lambda` with
  `log_λ²` = `log_lambda_2` for the linear predictor (`λ = exp(0.5·log_λ²)`).

# Remarks:
The predicted value is computed based on the equation's variable list and the
estimated coefficient vector. For instance, if the `frontier` function is a
linear function of variables `X` and coefficient vector `β`,
`sfmodel_predict(@eq(frontier))` returns `X*β̂ `. If the variance
function `σᵤ²` is parameterized by an exponential function of `Z` and `δ`
(i.e., `σᵤ² = exp(Zδ)`), `sfmodel_predict(@eq(log_σᵤ²))` returns `Z*δ̂`
and `sfmodel_predict(@eq(σᵤ²))` returns `exp(Z*δ̂)`.

# Examples
```julia
frontier_hat = sfmodel_predict(@eq(frontier));
sigma_u_2_hat = sfmodel_predict(@eq(σᵤ²));
```
"""
function sfmodel_predict(eq::Vector{Symbol})

  sfdata = _dicM[:sdf]
  eqname = eq[1]

  takeexp = false
  takesqrt = false     # R-09: Exponential λ = exp(0.5·W·δ̂) = sqrt(exp(W·δ̂))
  gamma_kumb = false   # Kumbhakar-1990 uses Gₜ = 2/(1+exp(Q·τ̂)) instead of exp()
  log_gamma_kumb = false  # R-10: Kumbhakar-1990 log Gₜ = log(2) − log1p(exp(Q·τ̂)) ≠ Q·τ̂

  if eqname == :frontier
      eq_var = :frontier
      eq_coe = :frontier
  elseif eqname == :mu || eqname == :μ
      eq_var = :μ
      eq_coe = :μ
  elseif eqname == :sigma_a_2 || eqname == :σₐ²
      eq_var  = :σₐ²
      eq_coe  = :log_σₐ²
      takeexp = true
  elseif eqname == :sigma_u_2 || eqname == :σᵤ²
      eq_var  = :σᵤ²
      eq_coe  = :log_σᵤ²
      takeexp = true
  elseif eqname == :sigma_v_2 || eqname == :σᵥ²
      eq_var  = :σᵥ²
      eq_coe  = :log_σᵥ²
      takeexp = true
  elseif eqname == :log_sigma_u_2 || eqname == :log_σᵤ²
      eq_var  = :σᵤ²
      eq_coe  = :log_σᵤ²
  elseif eqname == :log_sigma_v_2 || eqname == :log_σᵥ²
      eq_var  = :σᵥ²
      eq_coe  = :log_σᵥ²
  elseif eqname == :lambda || eqname == :λ
      # R-09: Exponential inefficiency scale. The coefficient block is `log_λ²`
      # and λ = exp(0.5·(W·δ̂)) = sqrt(exp(W·δ̂)), matching SFindex.jl jlmsbc(Expo).
      eq_var   = :λ
      eq_coe   = :log_λ²
      takesqrt = true
  elseif eqname == :log_lambda_2 || eqname == :log_λ²
      eq_var  = :λ
      eq_coe  = :log_λ²          # raw linear predictor W·δ̂
  elseif eqname == :gamma
      eq_var  = :gamma
      eq_coe  = :log_gamma
      if _dicM[:panel] == [:Kumbhakar1990]
          takeexp    = false
          gamma_kumb = true          # Gₜ = 2/(1+exp(Q·τ̂)) for Kumbhakar-1990
      else
          takeexp = true             # Gₜ = exp(Q·τ̂) for TimeDecay
      end
  elseif eqname == :log_gamma
      eq_var  = :gamma
      eq_coe  = :log_gamma
      if _dicM[:panel] == [:Kumbhakar1990]
          # R-10: the τ block is the pre-link predictor Q·τ̂; for Kumbhakar-1990
          # Gₜ = 2/(1+exp(Q·τ̂)) so log Gₜ = log(2) − log1p(exp(Q·τ̂)) ≠ Q·τ̂. (For
          # TimeDecay Gₜ = exp(Q·τ̂) so log Gₜ = Q·τ̂ — the raw predictor, unchanged.)
          log_gamma_kumb = true
      end
  elseif eqname == :hscale
      eq_var  = :hscale
      eq_coe  = :log_hscale
      takeexp = true
  elseif eqname == :log_hscale
      eq_var  = :hscale
      eq_coe  = :log_hscale
  else
      throw("@eq() is not specified correctly")
  end

  #* -- begin processing the data --------

  pvar = Matrix(sfdata[:, _dicM[eq_var]])  # get the variable vector/Matrix

  value = pvar*_eqncoe[eq_coe]

  if takeexp
     value = exp.(value)
  end

  if gamma_kumb
     value = 2.0 ./ (1.0 .+ exp.(value))   # Kumbhakar-1990 link Gₜ = 2/(1+exp(Q·τ̂))
  end

  if takesqrt
     value = exp.(0.5 .* value)            # R-09: λ = exp(0.5·W·δ̂) = sqrt(exp(W·δ̂))
  end

  if log_gamma_kumb
     value = log(2.0) .- log1p.(exp.(value))   # R-10: log Gₜ for Kumbhakar-1990
  end

  return value
end
