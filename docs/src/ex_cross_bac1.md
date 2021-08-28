# Cross Section Models

```@meta
## Basic Model
```

Consider the following equation:

```math
 \mathbf{y}  =  \mathbf{x} \beta + \mathbf{v} - \mathbf{u},
```
where ``\mathbf{y}`` is ``(N\times 1)``, ``\mathbf{x}`` is ``(N\times k)`` and includes a column of 1 for intercept, and ``\beta`` is ``(k \times 1)``. ``\mathbf{v}`` and ``\mathbf{u}`` are stochastic and assumed to follow certain distribution assumption. 

Our goal is to estimate ``\beta`` and parameters associated with ``\mathbf{v}`` and ``\mathbf{u}``.

In the follows, we use ``x_i`` (which is (``1 \times k``)) to denote the ``i``th observation of ``\mathbf{x}``. Other notations follow similarly.



## Normal Truncated-Normal Model

```@meta
!!! note "Info"
    The first example goes through the details. Other examples follow the similar pattern and will only highlight the differences.
```    

For the individual ``i``, the composed error consists of ``v_i`` and ``u_i`` with the following distribution assumptions:

```math
\begin{aligned}
 v_i & \sim N(0, \sigma_v^2),\\
 u_i & \sim N^+(\mu, \sigma_u^2),
 \end{aligned}
```

where ``N^+(\mu, \sigma_u^2)`` denotes a _truncated normal distribution_ obtained by truncating the normal distribution ``N(\mu, \sigma_u^2)`` from below at 0. The ``\mu`` and ``\sigma_u^2`` are thus the mean and the variance of the normal distribution _before_ the truncation. 

In the likelihood function, the variance ``\sigma_v^2`` and ``\sigma_u^2`` are specified using exponential functions to ensure positive values. That is,

```math
  \sigma_v^2 = \exp(z_i^v \gamma),\\
  \sigma_u^2 = \exp(z_i^u \delta),
```

where ``z_i^v`` and ``z_i^u`` are vectors of variables. If an intercept is desired (most likely), one of variables in the vector needs to be the _constant variable_ with a value equal to 1. 

There are two ways to provide data for estimation. One is to use data from a DataFrame where column names of the data are variable names. The other is to use data from matrix or vectors. This is the likely scenario in simulation studies where computer generated data is used in estimation. Different ways of obtaining data would require different ways to specify the model. We start with the DataFrame approach.


#### Step 1: Use DataFrame and Specify the Model using `sfmodel_spec()`

We use the production data from paddy farmers in India. The ``\mathbf{y}`` is the annual rice production and ``\mathbf{x}`` is agricultural inputs. The dataset is in the `.csv` format and we read it in using the `CSV` package and save it as a DataFrame with the name `df`.

```julia
using SFrontiers, Optim    # main packages
using DataFrames, CSV   # handling data

df = CSV.read("sampledata.csv", DataFrame; header=1, delim=",")
df[!, :_cons] .= 1.0;         # append a column of 1 to be used as a constant
```

Note that we append a column of 1 to `df` with the column name `_cons`. This is essential because we need the constant variable to estimate intercepts. Before estimation, users should make sure that `df` contains no missing values or any anomaly that may affect the estimation.

Next, we give specifications of the type of model we wish to estimate using `sfmodel_spec()`. We start with the specification of the Battese and Coelli (1995) model where ``\mu`` is parameterized as a linear function of a vector of exogenous variables and ``\sigma_u^2`` and ``\sigma_v^2`` are constant parameters. 


```julia
sfmodel_spec(SFtype(prod), SFdist(trun),
             @depvar(yvar), 
             @frontier(Lland, PIland, Llabor, Lbull, Lcost, yr, _cons), 
             @μ(age, school, yr, _cons),
             @σᵤ²(_cons),
             @σᵥ²(_cons))
```              

- `SFtype(prod)` indicates a production-frontier type of model. The alternative is `cost` for cost frontier where the composed error is ``v_i + u_i``.
- `SFdist(trun)` specifies the truncated-normal distribution assumption on ``u_i``. Alternatives include `half`, `expo`, and `trun_scaling`.
- `@depvar(.)` specifies the dependent variable.
- `@frontier(.)` specifies the list of variables used in the frontier equation (i.e., the data of ``\mathbf{x}``). The variables are assumes to be linear in the equation.
- `@μ(.)` (or `@mu(.)`) specifies the variables of inefficiency determinants as a linear function in `μ`.
- `@σᵤ²(.)` (or `@sigma_u_2(.)`) and `@σᵥ²(.)` (or `@sigma_v_2(.)`) specify the variables to parameterize the variances. Note that here we include only the variable `_cons` and so ``\sigma_v^2`` and ``\sigma_u^2`` are both estimated as constant parameters.


Note how the constant parameters ``\sigma_v^2`` and ``\sigma_u^2`` are estimated. Using ``\sigma_u^2`` as an example, we have
```math
  \sigma_u^2 = \exp(z_i^u \delta)
```
where ``z_i^2`` is the constant variable `_cons`.


!!! tip "Note on intercepts and constant parameters" 
    __SFrontiers__ estimates intercepts and constant parameters as the coefficient of a constant variable. That is, if a parameter is constant, __SFrontiers__ requires a variable with values equal to 1. This is true for all equations. 



#### Step 2: Specify Initial Vaules using `sfmodel_init()` _(optional)_

The specification is optional. You can skip it entirely or specify the initial values for a partial list of the equations. If missing (all or part of the equations), default values will be used. Currently, the default uses the OLS estimates as initial values for the `frontier()` equation and ``0.1`` for all of the other parameters. The following example shows a mix of the strategies.

```julia
b1 = ones(4)*(0.1)

sfmodel_init(    # frontier(),
             μ(b1),
                 # σᵤ²(-0.1),
             σᵥ²(-0.1)) 
```             
The order of equations in `sfmodel_init()` is not important. 

Note that because of the exponential parameterization on `σᵤ²` and `σᵥ²`, the initial values are given to the `log(σᵤ²)` and `log(σᵥ²)`. In the above example where we have `σᵥ²(-0.1)`, it means we have in mind the initial value of `σᵥ²` being ``\exp(-0.1) = 0.905``.

You may notice that equation names (`frontier`, `μ`, etc.) are similar to those in `sfmodel_spec()` but different. `sfmodel_spec()` uses the macros type of names (those with "`@`") and here `sfmodel_init()` uses function type of names (those without "`@`").


!!! tip "Note on `@frontier( )` vs. `frontier( )`"
    Macro type of equation names (`@frontier()`, `@μ()`, etc.) are used only when the arguments are column names from DataFrames. For everything else, function type of equation names (`frontier()`, `μ()`, etc.) are used.


!!! tip "Note on name conflict"
    __SFrontiers__ uses names such as `μ`, `σᵥ²`, `gamma`, `depvar`, `frontier`, etc.. If users had used the same names elsewhere in the program for different purposes (for instance, using `μ` to denote the value of a parameter), or users import other packages that use the same names, name conflicts would arise. Signs of the problem include error messages such as 
    ```
    MethodError: objects of type ... are not callable
    ```
    There are simple ways to work around:
    - Use fully qualified function names. For instance, use `SFrontiers.σᵥ²` instead of `σᵥ²`.
    - Use alias if there is one. For instance, use `sigma_v_2` instead of `σᵥ²`. Check the section of `API Reference` for the information.


#### Step 3: Specify Maximization Options (and others) using `sfmodel_opt()` _(optional)_

The main purpose of this function is to select options for the numerical maximization procedures, including the choice of optimization algorithms, the number of iterations, the convergence criterion, and others. __SFrontiers__ uses Julia's [`Optim`](https://julianlsolvers.github.io/Optim.jl/stable/) package to do the maximization, and a subset of `Optim`'s options are directly accessible through `sfmodel_init()`.

An effective estimation strategy for challenging optimization problems is to use a non-gradient algorithm in the first stage (`warmstart`) for a few iterations and then switch to gradient-based algorithms in the second stage (`main`) for speed and accuracy. __SFrontiers__ uses the two-step strategy as default.

This function is also optional. All or part of the specifications may be skipped. If missing, default values will be used. The following example uses the default values.


```julia
sfmodel_opt(warmstart_solver(NelderMead()),   
            warmstart_maxIT(100),
            main_solver(Newton()), 
            main_maxIT(2000), 
            tolerance(1e-8))
```             
- `warmstart_solver(NelderMead())`: specifies the `Nelder-Mead` algorithm in the first stage estimation. Mind the braces "`()`" which is part of the algorithm name. Using a non-gradient algorithm in the first stage is recommended.
- `warmstart_maxIT(100)` and `main_maxIT(2000)`: the maximum numbers of iterations in the first and the second stage estimation.
- `main_solver(Newton())`: specifies the `Newton` method in the second stage estimation. 
- `tolerance(1e-8)`: set the convergence criterion, which is with respect to the absolute tolerance in the gradient, to `1e-8`. For non-gradient algorithms, it controls the main convergence tolerance, which is solver specific. This is a wrapper of `Optim`'s `g_tol` option.


If the two-stage strategy is not required, the `warmstart` stage can be skipped by giving empty keyword values to the first stage options, such as `warmstart_solver()` or `warmstart_maxIT()`, or both. Note that if we simply skip the keywords (i.e., missing `warmstart_solver` or `warmstart_maxIT` entirely), the default will be reinstate. Again, the first stage estimation will only be skipped when empty values are explicitly given to the the related options.

In __SFrontiers__, _missing_ = "_Whatever. Just use the default_", and _empty value_ = "_I know, but don't do it_".


In addition to controlling the maximization procedures, `sfmodel_opt()` also provides options to control other parts of the estimation. They and their default values are the follows. 

* `banner(true)`: show banner to help to visually identify the start of the estimation. 
* `verbose(true)`: show interim and final results.
* `ineff_index(true)`: compute the Jondrow et al. (1982) inefficiency index and the Battese and Coelli (1987) efficiency index.
* `marginal(true)`: calculate the marginal effect of the inefficiency determinants (if any) on the unconditional mean of inefficiency.

Turning these options to `false` may be useful particularly in simulation settings.


#### Step 4: Estimation

Step 1 to 3 prepare the model, and now we are ready to estimate it. We use `sfmodel_fit()` to start the estimation, and it returns a dictionary containing estimation results and other information of the model. In the following example, we save the returned dictionary in `res`.

```julia
res = sfmodel_fit(useData(df))
```
- `useData(df)`: name of the DataFrame (see Step 1).


#### Step 5: Results and Post Estimation Analysis

The main estimation results are shown in the terminal after it is done. Let's see what we have.

```
*********************************
       Estimation Results:
*********************************
Model type: the normal and truncated-normal model of Battese and Coelli (1995, JoE)
Number of observations: 271
Number of total iterations: 114
Converged successfully: true
Log-likelihood value: -87.21356

┌──────────┬────────┬─────────┬──────────┬──────────┬────────┬─────────┬─────────┐
│          │   Var. │   Coef. │ Std.Err. │        z │  P>|z| │ 95%CI_l │ 95%CI_u │
├──────────┼────────┼─────────┼──────────┼──────────┼────────┼─────────┼─────────┤
│ frontier │  Lland │  0.3030 │   0.0704 │   4.3024 │ 0.0000 │  0.1650 │  0.4410 │
│          │ PIland │  0.2479 │   0.1762 │   1.4069 │ 0.1607 │ -0.0975 │  0.5933 │
│          │ Llabor │  1.1265 │   0.0843 │  13.3658 │ 0.0000 │  0.9613 │  1.2917 │
│          │  Lbull │ -0.4038 │   0.0602 │  -6.7130 │ 0.0000 │ -0.5217 │ -0.2859 │
│          │  Lcost │  0.0142 │   0.0130 │   1.0915 │ 0.2761 │ -0.0113 │  0.0396 │
│          │     yr │  0.0146 │   0.0104 │   1.4028 │ 0.1619 │ -0.0058 │  0.0350 │
│          │  _cons │  1.6699 │   0.3612 │   4.6234 │ 0.0000 │  0.9620 │  2.3778 │
│        μ │    age │ -0.0045 │   0.0144 │  -0.3154 │ 0.7527 │ -0.0327 │  0.0236 │
│          │ school │  0.0304 │   0.0701 │   0.4336 │ 0.6650 │ -0.1070 │  0.1678 │
│          │     yr │ -0.2238 │   0.2263 │  -0.9889 │ 0.3237 │ -0.6674 │  0.2198 │
│          │  _cons │ -0.1747 │   1.4744 │  -0.1185 │ 0.9058 │ -3.0644 │  2.7150 │
│  log_σᵤ² │  _cons │ -0.4083 │   1.0082 │  -0.4050 │ 0.6858 │ -2.3842 │  1.5677 │
│  log_σᵥ² │  _cons │ -3.2328 │   0.2984 │ -10.8324 │ 0.0000 │ -3.8177 │ -2.6479 │
└──────────┴────────┴─────────┴──────────┴──────────┴────────┴─────────┴─────────┘

Convert the constant log-parameter to its original scale, e.g., σ² = exp(log_σ²):
┌─────┬────────┬──────────┐
│     │  Coef. │ Std.Err. │
├─────┼────────┼──────────┤
│ σᵤ² │ 0.6648 │   0.6702 │
│ σᵥ² │ 0.0394 │   0.0118 │
└─────┴────────┴──────────┘

***** Additional Information *********
* OLS (frontier-only) log-likelihood: -104.96993
* Skewness of OLS residuals: -0.70351
* The sample mean of the JLMS inefficiency index: 0.33006
* The sample mean of the BC efficiency index: 0.74921

* The sample mean of inefficiency determinants' marginal effects on E(u): (age = -0.00061, school = 0.00407, yr = -0.02997)
* Marginal effects of the inefficiency determinants at the observational level are saved in the return. See the follows.

* Use `name.list` to see saved results (keys and values) where `name` is the return specified in `name = sfmodel_fit(..)`. Values may be retrieved using the keys. For instance:
   ** `name.loglikelihood`: the log-likelihood value of the model;
   ** `name.jlms`: Jondrow et al. (1982) inefficiency index;
   ** `name.bc`: Battese and Coelli (1988) efficiency index;
   ** `name.marginal`: a DataFrame with variables' (if any) marginal effects on E(u).
* Use `keys(name)` to see available keys.
**************************************
```

As reminded in the printout, many of the model information are saved in the dictionary which can be called later for further investigation. Let's see the list of available keys first.

```julia
julia> keys(res)
(:converged, :iter_limit_reached, :_______________, :n_observations, :loglikelihood, :table, :coeff, :std_err, :var_cov_mat, :jlms, :bc, :OLS_loglikelihood, :OLS_resid_skew, :marginal, :marginal_mean, :_____________, :model, :depvar, :frontier, :μ, :σₐ², :σᵤ², :σᵥ², :log_σₐ², :log_σᵤ², :log_σᵥ², :type, :dist, :coeff_frontier, :coeff_μ, :coeff_log_σᵤ², :coeff_log_σᵥ², :________________, :Hessian, :gradient_norm, :actual_iterations, :______________, :warmstart_solver, :warmstart_ini, :warmstart_maxIT, :main_solver, :main_ini, :main_maxIT, :tolerance, :eqpo, :redflag, :list)
```

Among the keywords is the term `:coeff`, which is the keyword of the saved coefficient vector. We may retrieve the coefficient vector and save it in the name `b0` possibly for later use.
```julia
julia> b0 = res.coeff
13-element Vector{Float64}:
  0.3029823393273174
  0.24794010653151866
  1.1265299143721006
 -0.4038266728402889
  0.014152608439534585
  0.014589911311642377
  1.6699115039893075
 -0.004532647263411585
  0.03039490853353706
 -0.22379268672696456
 -0.17467371095755305
 -0.40828210156522676
 -3.2327957145786477
```

The estimation table shown above with the coefficients, standard errors, etc., may also be retrieved using the keyword `:table`, though it may not be formatted as pretty.



#### Inefficiency and Efficiency Index

The Jondrow et al. (1982) inefficiency index (``E(u|\widehat{v_i-u_i})``) and the Battese and Coelli (1987) efficiency index (``E(\exp(-u)|\widehat{v_i-u_i})``) may also be retrieved using the keywords `:jlms` and `:bc`. Here we show them in a ``N\times 2`` matrix.


```julia
julia> [res.jlms res.bc]
271×2 Matrix{Float64}:
 0.536473   0.595487
 0.47957    0.630042
 0.0924123  0.914553
 0.239773   0.795486
 0.114407   0.89573
 ⋮
 0.410421   0.674444
 0.859055   0.431521
 0.119179   0.891702
 0.18474    0.838167
 0.169893   0.850018
```

#### Marginal Effects

Let's also show the marginal effects of the inefficient determinants at the observational level, which are saved as a DataFrame object.

```julia
julia> res.marginal
271×3 DataFrame
│ Row │ marg_age     │ marg_school │ marg_yr    │
│     │ Float64      │ Float64     │ Float64    │
├─────┼──────────────┼─────────────┼────────────┤
│ 1   │ -0.000482249 │ 0.00323385  │ -0.0238103 │
│ 2   │ -0.000419427 │ 0.00281258  │ -0.0207086 │
│ 3   │ -0.000366921 │ 0.00246049  │ -0.0181162 │
│ 4   │ -0.000322813 │ 0.00216471  │ -0.0159384 │
│ 5   │ -0.000285561 │ 0.00191491  │ -0.0140992 │
⋮
│ 266 │ -0.000482007 │ 0.00323223  │ -0.0237984 │
│ 267 │ -0.000419226 │ 0.00281123  │ -0.0206987 │
│ 268 │ -0.00119146  │ 0.00798967  │ -0.0588266 │
│ 269 │ -0.00100688  │ 0.00675194  │ -0.0497134 │
│ 270 │ -0.000853674 │ 0.00572455  │ -0.0421489 │
│ 271 │ -0.000727005 │ 0.00487513  │ -0.0358948 │
```

#### Hypothesis Testing

We may conduct a likelihood ration (LR) test to see if the frontier specification is supported by the data. The null hypothesis is that the inefficiency term ``u_i`` is not warranted and the fit of the model is no better than the OLS.

First, we calculate the test statistics using the log-likelihood values of the OLS model and the current model.

```julia
julia> -2*(res.OLS_loglikelihood - res.loglikelihood)
35.51273933147917
```

Because the test amounts to testing ``u_i =0`` which is on the boundary of the parameter's support, the appropriate distribution for the test statistic is the mixed ``\chi^2`` distribution. Critical values may be retrieved using `sfmodel_MixTable(dof)` where `dof` is the degree of freedom of the test. In this example, `dof=5` because there are five parameters involved in ``u_i``.

```julia
julia> sfmodel_MixTable(5)

  * Significance levels and critical values of the mixed χ² distribution
┌─────┬───────┬────────┬────────┬────────┐
│ dof │  0.10 │   0.05 │  0.025 │   0.01 │
├─────┼───────┼────────┼────────┼────────┤
│ 5.0 │ 8.574 │ 10.371 │ 12.103 │ 14.325 │
└─────┴───────┴────────┴────────┴────────┘

source: Table 1, Kodde and Palm (1986, Econometrica).
```

Since the test statistic ``35.513`` is much larger than the critical value at the ``1\%`` level (which is ``14.325``), the result overwhelmingly rejects the null hypothesis of an OLS model.

#### Others

__SFrontiers__ provides the function `sfmodel_predict()` to obtain predicted values of the equations after the model is estimated. The following returns the predicted value of the `frontier` equation, i.e., ``\mathbf{x} \hat{\beta}``.

```julia
julia> xb = sfmodel_predict(@eq(frontier), df)
271-element Vector{Float64}:
 5.837738490387098
 5.467350523453003
 5.145107451544124
 5.494031744452952
 6.062700939460845
 6.097179092569791
 ⋮
 4.861054485639938
 6.393104647266437
 8.207717614833687
 7.894552876042939
 8.537385909249965
 6.373480344886016
``` 

`df` in the function is the name of the DataFrame which we had used to estimate the model.


We may also use it to predict values of ``\sigma_u^2`` for each observation, though in the current example it is trivial since ``\sigma_u^2`` is a constant.

```julia
julia> sfmodel_predict(@eq( σᵤ² ), df)
271-element Vector{Float64}:
 0.6647913136972949
 0.6647913136972949
 0.6647913136972949
 ⋮
 0.6647913136972949
 0.6647913136972949
 0.6647913136972949
```



## Normal Truncated-Normal Model with the Scaling Property


## Normal Half-Normal Model

## Normal Exponential Model


