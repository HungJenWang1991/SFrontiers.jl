# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

########################################################
####             equation prediction                ####
########################################################

# Is the module-level model state still the one the last fit used?
#
# `sfmodel_predict` and `sfmodel_boot_marginal` read `_dicM` at CALL time but pair it with
# coefficients only a fit writes, so without this guard a spec built in between would hand the
# new spec's data to the old fit's coefficients without a word: `sfmodel_predict([:frontier])`
# would return `X_new * beta_old` exactly.  Every spec route assigns a fresh `_dicM` Dict, so
# identity is enough to tell.
function _mle_state_matches_last_fit()
    isdefined(@__MODULE__, :_fitted_dicM) || return false
    isdefined(@__MODULE__, :_dicM)        || return false
    return _dicM === _fitted_dicM
end

function _mle_require_state_of_last_fit(who::String)
    _mle_state_matches_last_fit() && return nothing
    error(who * ": the model state this reads has been replaced since the last " *
          "`sfmodel_fit`. " * who * " takes its DATA from the module-level model state at " *
          "call time and its COEFFICIENTS from the last fit, so a `sfmodel_spec` built in " *
          "between would pair one model's data with another model's estimates. Call it " *
          "immediately after the fit it refers to, or re-run that fit. (`predict(result; ...)` " *
          "reads the spec attached to its result object and is not affected.)")
end

# Is `result` the result of the last :MLE fit? `sfmodel_boot_marginal` takes the data and the
# model's equations from the module-level state and everything else (starting values, solver
# settings, labels, point estimates) from `result`. The check above only confirms that the
# state is the last fit's: after fitting a model A and then a model B,
# `sfmodel_boot_marginal(result = A)` would resample B's data with A's settings and return its
# standard errors without a word. Two links, both exact for the result of the last fit:
#   (1) its `frontier` entry IS the name vector of the `_dicM` it was fitted on: the fit stores
#       `_dicM[:frontier]` itself, not a copy; every keyword or DSL spec, and every fit handed
#       a spec object, builds a fresh `_dicM`; and nothing replaces that entry afterwards (the
#       bootstrap's name override touches only :μ, :σᵤ² and :hscale). A rerun of the same fit
#       builds a new `_dicM` too, so the result of the earlier run fails this link.
#   (2) the coefficients it starts the bootstrap from (`list[:coeff]`, else `coeff`)
#       equal, bit for bit, those the last fit left in `_eqncoe`, keyed by equation name, at the
#       positions `eqpo` (getvar's `eqvec2`) records. This separates two fits after one
#       deprecated Vararg `sfmodel_spec`, which share one `_dicM`.
# Both links read only what `sfmodel_fit` already stores.
function _mle_result_is_last_fit(result)
    _mle_state_matches_last_fit() || return false
    isdefined(@__MODULE__, :_eqncoe) || return false
    (hasproperty(result, :frontier) && hasproperty(result, :eqpo)) || return false
    fr = result.frontier
    (fr !== nothing && fr === get(_fitted_dicM, :frontier, nothing)) || return false     # (1)
    coef = (hasproperty(result, :list) && haskey(result.list, :coeff)) ? result.list[:coeff] :
           hasproperty(result, :coeff) ? result.coeff : nothing
    coef isa AbstractVector || return false
    eqpo = result.eqpo                                                                    # (2)
    (eqpo isa NamedTuple && length(eqpo) == length(_eqncoe) &&
     sum(length, eqpo; init = 0) == length(coef)) || return false
    for k in keys(eqpo)
        name = Symbol(chopprefix(String(k), "coeff_"))
        (haskey(_eqncoe, name) && checkbounds(Bool, coef, eqpo[k]) &&
         isequal(_eqncoe[name], coef[eqpo[k]])) || return false
    end
    return true
end

function _mle_require_result_of_last_fit(who::String, result)
    _mle_require_state_of_last_fit(who)
    _mle_result_is_last_fit(result) && return nothing
    error(who * ": `result` is not the result of the last `sfmodel_fit` with `method = :MLE`. " *
          who * " takes its DATA and the model's equations from the module-level model state, " *
          "which belongs to that last fit, and its starting values, solver settings and labels " *
          "from `result`, so bootstrapping an earlier result after another model has been " *
          "fitted would resample that other model's data. Call it immediately after the fit " *
          "it refers to, or re-run that fit and pass the result of the new run. (The result of " *
          "an earlier run of the same fit is refused as well, and so is a deep copy of a result, " *
          "such as one saved and loaded again: pass the object the fit returned.)")
end

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
  `μ` = `mu`, and `σᵤ` = `sigma_u`.
  For the Exponential inefficiency the squared scale is `σᵤ²` = `sigma_u_2` like every
  other distribution, and `σᵤ` = `sigma_u` returns the scale itself, `exp(0.5·log_σᵤ²)`,
  which is also `E(uᵢ)`.

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

  _mle_require_state_of_last_fit("sfmodel_predict")

  sfdata = _dicM[:sdf]
  eqname = eq[1]

  takeexp = false
  takesqrt = false     # the scale σᵤ = exp(0.5·W·δ̂) = sqrt(exp(W·δ̂))
  gamma_kumb = false   # Kumbhakar-1990 uses Gₜ = 2/(1+exp(Q·τ̂)) instead of exp()
  log_gamma_kumb = false  # Kumbhakar-1990 log Gₜ = log(2) − log1p(exp(Q·τ̂)) ≠ Q·τ̂

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
  elseif eqname == :sigma_u || eqname == :σᵤ
      # The inefficiency distribution's SCALE, σᵤ = exp(0.5·(W·δ̂)) = sqrt(exp(W·δ̂)),
      # matching SFindex.jl. For the Exponential that is also E(uᵢ). The block it reads is
      # `log_σᵤ²` for every distribution that owns one.
      eq_var   = :σᵤ²
      eq_coe   = :log_σᵤ²
      takesqrt = true
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
          # The τ block is the pre-link predictor Q·τ̂; for Kumbhakar-1990
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
     value = exp.(0.5 .* value)            # σᵤ = exp(0.5·W·δ̂) = sqrt(exp(W·δ̂))
  end

  if log_gamma_kumb
     value = log(2.0) .- log1p.(exp.(value))   # log Gₜ for Kumbhakar-1990
  end

  return value
end
