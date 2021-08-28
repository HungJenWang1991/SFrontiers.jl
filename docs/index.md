__SFrontiers.jl__ provides commands for estimating various parametric _stochastic frontier models_ in __Julia__. The commands estimate model parameters, calculate efficiency and inefficiency index, compute marginal effects of inefficiency determinants (if any), and with the option of bootstrapping standard errors of the mean marginal effects. Collaboration to add models is welcome.


```@meta
@contents
Pages = ["literature.md", "overview.md"]
Depth = 3
```


## What is a stochastic frontier model?

!!! note "in a nutshell"
    An econometric model where the composed error consists of a one-sided (either positive or negative) term and the usual zero-mean error term. The model is usually estimated by MLE. Applications include production efficiency and others that may have upper bounds (e.g., potential output) or lower bounds (e.g., minimum cost) interpretations on the dependent variable.


A stochastic frontier (SF) model was originally proposed in the production function context to account for the production frontier (potential output) and the shortfall of the output from the frontier due to technical inefficiency. The latter is usually modeled as a one-sided deviation from the frontier. The model has since been extended to study other issues, such as underpayment in the job market due to labor market frictions, the financing constraints hypothesis of capital investment, growth convergence, underpricing of IPO, etc..

A typical SF model in the production function context may be represented as
```math
Y_i = F(X_i; \beta)e^{v_i - u_i},
```
taking logarithms on both sides and imposing simplifying assumptions on ``F(.)``, we have
```math
 y_i  = \alpha_0 + x_i' \beta + v_i - u_i.
```
Here ``v_i`` is the zero-mean random error and ``u_i>0`` is output loss due to technical inefficiency. Both are usually assumed to be stochastic and follow certain distributions. 
We may define ``y_i^* = \alpha_0 + x_i' \beta + v_i`` as the _stochastic frontier_, and observed output ``y_i`` is the frontier output minus the inefficiency effect, i.e., ``y_i = y_i^* - u_i``.
The model is often estimated by the maximum likelihood method based on the distribution assumptions of ``v_i`` and ``u_i``.

```@meta
 v_i \sim N(0, \sigma_v^2),\\
 u_i \sim N^+(\mu, \sigma_u^2),
Here ``N^+(\cdot)`` denotes a non-negative truncation of the underlying normal distribution
from below. 
```

Exogenous determinants of inefficiency, ``z_i``, may be parameterized into the distribution function of ``u_i``, and the marginal effects of ``z_i`` on the unconditional mean of inefficiency ``E(u_i)`` can be computed. The focus of a SF analysis also includes calculating the inefficiency index ``E(u\mid v_i - u_i)`` and the efficiency index ``E(\exp(u_i) \mid v_i - u_i)`` for each observation in the data.

A typical panel SF model can be similarly defined:
```math  
y_{it} = \alpha_i + x_{it}' \beta + v_{it} - u_{it}.
```
Here, ``\alpha_i`` is the individual's unobservable effect which is different from the inefficiency effect. The model is the so-called _true fixed effect panel SF model_ if ``\alpha_i`` is assumed to be a fixed parameter, and the _true random effect panel SF model_ if it is a value from a random variable. Other variants of the panel model may be obtained by making extra assumptions on ``\alpha_i``, ``u_{it}``, and ``v_{it}``.


## Why Julia?


!!! note "TL;DR"
    Speed and accuracy. The latter also translates into easier convergence in maximum likelihood estimation. In some cases, Julia helps to turn the problem's convergence issue from _art_ to _science_. 


Parametric SF models are highly nonlinear, which often presents challenges to maximum likelihood estimation. Recent developments in the literature also grow reliance on simulations and bootstrapping. Julia's strength helps to meet the demand.

* __SFrontiers__ uses Julia's [`Optim`](https://julianlsolvers.github.io/Optim.jl/stable/#) package as the basis to do MLE, and I found it versatile and robust. It offers an array of solvers, including non-gradient (e.g., `Nelder Mead`), gradient-based (e.g., `BFGS`), and Hessian-based algorithms (e.g., `Newton`). It also has a global optimization algorithm (i.e., `Particle Swarm`).  An effective estimation strategy for difficult problems is to use non-gradient methods in the first stage of the estimation to zoom in to the neighborhood of the solution and then switch to gradient-based ones. The strategy is easily implemented in __SFrontiers__ using `Optim,` and I found it very useful.

* `Optim` could use [_automatic differentiation_](https://en.wikipedia.org/wiki/Automatic_differentiation) from the package [`ForwardDiff`](https://juliadiff.org/ForwardDiff.jl/stable/) to calculate gradients and Hessians for the gradient- and Hessian-based algorithms. The automatic differentiation is different from symbolic differentiation and numerical finite differentiation. It in fact has the best of both worlds: it is as accurate as the analytic derivatives and it does not require users' supply of analytic formula (nor supply anything other than the log-likelihood function itself). Using automatic differentiation significantly improves estimation speed and accuracy. All of the models estimated by __SFrontiers__ use automatic differentiation by default.

* The _automatic differentiation_ also provides an easy way to compute marginal effects of inefficiency determinants.

* Julia is fast. Very fast. Some claim it can be as fast as C. This gives Julia an important edge in simulation-based estimation.


* Julia has a large ecosystem for optimization and modeling, including [`JuMP`](https://jump.dev/JuMP.jl/stable/), [`Optim`](https://julianlsolvers.github.io/Optim.jl/stable/#), [`BlackBoxOptim`](https://github.com/robertfeldt/BlackBoxOptim.jl), [`NLopt`](https://github.com/JuliaOpt/NLopt.jl), [`NLsolve`](https://github.com/JuliaNLSolvers/NLsolve.jl), and others. Some of which could be useful for SF research in the future.

* Julia language's _multiple dispatch_ feature makes it easy to collaborate and contribute. People who wish to make other models available via __SFrontiers__ can add their own relatively easily without worry too much about breaking other parts of the package.

* Julia's is free and open source.


## [What Models are Covered by __SFrontiers__?](@id literature)

!!! note "quick peek"
    * cross-sectional: ``v_i`` is a normal distribution and ``u_i`` follows half normal, truncated normal, or exponential distributions, with the distribution flexibly parameterized by vectors of exogenous determinants. Also, the scaling property model. 
    * panel: various flavors of true fixed effect models (Greene 2005, Wang and Ho 2010, Chen, Wang, and Schmidt 2014), true random effect model, time decay model.
    * in the pipeline: two-tier frontier models with exogenous determinants, moment estimator and moment tests, time series autocorrelation models, semi-parametric models, and models with exotic distributions estimated by maximum simulated likelihoods.

The following is a partial list of articles for which the models are covered by (can be estimated by) routines in __SFrontiers__. If there is anything missing, please inform me and I will be happy to update the list. 

* Aigner, D., Lovell, C. A. K., & Schmidt, P. (1977). Formulation and Estimation of Stochastic Frontier Production Function Models. Journal of Econometrics, 6, pp. 21-37. 
* Alvarez, A., Amsler, C., & Schmidt, P. (2006). Interpreting and Testing the Scaling Property in Models where Inefficiency Depends on Firm Characteristics. Journal of Productivity Analysis, 25, pp. 201-212. 
* Battese, G. E., & Coelli, T. J. (1988). Prediction of Firm-Level Technical Efficiencies with a Generalized Frontier Production Function and Panel Data. Journal of Econometrics, 38, pp. 387-399.
* Battese, G. E., & Coelli, T. J. (1992). Frontier Production Functions, Technical Efficiency and Panel Data: With Application to Paddy Farmers in India. Journal of Productivity Analysis, 3, pp. 153-169. 
* Battese, G. E., & Coelli, T. J. (1995). A Model for Technical Inefficiency Effects in a Stochastic Frontier Production Function for Panel Data. Empirical Economics, 20, pp. 325-332. 
* Bera, A. K., & Sharma, S. C. (1999). Estimating Production Uncertainty in Stochastic Frontier Production Function Models. Journal of Productivity Analysis, 12, pp. 187-210. 
* Caudill, S. B., & Ford, J. M. (1993). Biases in Frontier Estimation Due to Heteroscedasticity. In Economics Letters, 41, pp. 17-20.
* Chen, Y. T., & Wang, H. J. (2012). Centered-Residuals-Based Moment Estimator and Test for Stochastic Frontier Models. Econometric Reviews, 31(6), 625-653. 
* Chen, Y. Y., Schmidt, P., & Wang, H. J. (2014). Consistent estimation of the fixed effects stochastic frontier model. Journal of Econometrics, 181(2), 65-76. 
* Coelli, T., Perelman, S., & Romano, E. (1999). Accounting for Environmental Influences in Stochastic Frontier Models: With Application to International Airlines. Journal of Productivity Analysis, 11, pp. 251-273. 
* Colombi, R., Kumbhakar, S. C., Martini, G., & Vittadini, G. (2014). Closed-skew normality in stochastic frontiers with individual effects and long/short-run efficiency. Journal of Productivity Analysis, 42(2), 123-136. 
* Greene, W. (2004). Distinguishing between heterogeneity and inefficiency: stochastic frontier analysis of the World Health Organization's panel data on national health care systems. Health Economics, 13(10), 959-980. 
* Greene, W. (2005). Reconsidering heterogeneity in panel data estimators of the stochastic frontier model. Journal of Econometrics, 126, 269-303. 
* Huang, C. J., & Liu, J. T. (1994). Estimation of a Non-neutral Stochastic Frontier Production Function. Journal of Productivity Analysis, 5, pp. 171-180. 
* Huang, Y. F., Luo, S., & Wang, H. J. (2018). Flexible panel stochastic frontier model with serially correlated errors. Economics Letters, 163, 55-58.
* Jondrow, J., Lovell, C. A. K., Materov, I. S., & Schmidt, P. (1982). On the Estimation of Technical Inefficiency in the Stochastic Frontier Production Function Model. Journal of Econometrics, 19, pp. 233-238. 
* Kumbhakar, S. C., Ghosh, S., & McGuckin, J. T. (1991). A Generalized Production Frontier Approach for Estimating Determinants of Inefficiency in U.S. Dairy Farms. In Journal of Business and Economic Statistics, 9, pp. 279-286.
* Kumbhakar, S. C., & Wang, H.-J. (2005). Estimation of Growth Convergence Using a Stochastic Production Frontier Approach. In Economics Letters, 88, pp. 300-305.
* Stevenson, R. E. (1980). Likelihood Functions for Generalized Stochastic Frontier Estimation. Journal of Econometrics, 13, pp. 57-66.
* Wang, H.-J. (2002). Heteroscedasticity and Non-monotonic Efficiency Effects of a Stochastic Frontier Model. Journal of Productivity Analysis, 18, pp. 241-253.
* Wang, H.-J. (2003). A Stochastic Frontier Analysis of Financing Constraints on Investment: The Case of Financial Liberalization in Taiwan. Journal of Business and Economic Statistics, 21, pp. 406-419.
* Wang, H.-J., & Ho, C.-W. (2010). Estimating Fixed-Effect Panel Stochastic Frontier Models by Model Transformation. Journal of Econometrics, 157, 289-296. 
* Wang, H.-J., & Schmidt, P. (2002). One-Step and Two-Step Estimation of the Effects of Exogenous Variables on Technical Efficiency Levels. Journal of Productivity Analysis, 18, pp. 129-144.




## Citing __SFrontiers__

```@meta
If you find __SFrontiers.jl__ useful for your research, please cite the following paper:

@Article{wang_2021,
  author        = {Wang, Hung-Jen},
  journal       = {n.a.},
  title         = {SFrontiers: Stochastic Frontier Analysis using Julia},
  year          = {hopefully 2021},
  archiveprefix = {arXiv},
  eprint        = {n.a.},
  keywords      = {stochastic frontier analysis},
  primaryclass  = {??},
  url           = {n.a.},
}
```