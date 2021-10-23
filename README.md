# SFrontiers.jl
A Julia Package for Estimating Stochastic Frontier Models


[![Build Status](https://travis-ci.com/HungJenWang1991/SFrontiers.jl.svg?branch=main)](https://travis-ci.com/HungJenWang1991/SFrontiers.jl)
[![codecov.io](https://codecov.io/github/HungJenWang1991/SFrontiers.jl/coverage.svg?branch=main)](https://codecov.io/github/HungJenWang1991/SFrontiers.jl?branch=main)



__SFrontiers.jl__ provides commands for estimating various parametric _stochastic frontier models_ in Julia. The commands estimate model parameters, calculate efficiency and inefficiency index, compute marginal effects of inefficiency determinants (if any), and with the option of bootstrapping standard errors of the mean marginal effects. Estimation methods include the maximum likelihood estimator (MLE) and the method of moments (MoM) estimator.

The package uses `Optim.jl` as the main driver for the maximum likelihood estimation.

#### Updates
* Oct. 23, 2021: New command `sfmodel_MoMTest()` for testing distribution assumptions and estimating model parameters using the MoM.

#### Coverage
* cross-sectional: models where the one-sided stochastic term (i.e., $u_i$) follows half normal, truncated normal, or exponential distributions, with the distributions flexibly parameterized by vectors of exogenous determinants. Also, the scaling property model. 
* panel: various flavors of true fixed effect models, true random effect model, and time-decay model.
* in the pipeline: two-tier frontier models with exogenous determinants, time series autocorrelation models, semi-parametric models, and models with exotic distributions estimated by maximum simulated likelihoods.

Collaboration to add models is welcome!


## Documentation

[Current Version](https://hungjenwang1991.github.io/SFrontiers.jl/). It has a detailed example (also available in Jupyter notebook) of using `SFrontiers.jl` to conduct a stochastic frontier analysis.

## Installation

    julia> using Pkg
    julia> Pkg.add("SFrontiers")

## Citation

(TBA)