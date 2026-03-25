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
  and `μ` = `mu`.

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
  elseif eqname == :gamma
      eq_var  = :gamma
      eq_coe  = :log_gamma
      takeexp = true
  elseif eqname == :log_gamma
      eq_var  = :gamma
      eq_coe  = :log_gamma
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

  return value
end
