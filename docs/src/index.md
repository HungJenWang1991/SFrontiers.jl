# SFrontiers.jl

A Julia package for stochastic frontier analysis with multiple estimation methods.

## Overview

SFrontiers.jl provides a unified interface for estimating stochastic frontier models using four backends:

- **MLE**: Analytic maximum likelihood estimation (closed-form, fastest)
- **MCI**: Monte Carlo integration with change-of-variable transforms
- **MSLE**: Maximum simulated likelihood estimation via inverse CDF
- **Panel**: Wang and Ho (2010) true fixed-effect panel models

## Supported Distributions

**Noise (v)**: Normal, Student-t, Laplace

**Inefficiency (u)**: Half-Normal, Truncated-Normal, Exponential, Weibull, Lognormal, Lomax, Rayleigh, Gamma

## API Reference

```@docs
sfmodel_spec
sfmodel_method
sfmodel_init
sfmodel_opt
sfmodel_fit
```
