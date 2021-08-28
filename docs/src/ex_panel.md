# [Examples of Panel Stochastic Frontier Models](@id panel)

```@contents
Pages = ["ex_panel.md"]
Depth = 5
```


!!! note "Reminder" 
    As shown in the [A Detailed Example](@ref detailedexample) section, __SFrontiers__ estimates stochastic frontier models in four steps:
      1. model specification using `sfmodel_spec()`,
      2. initial values using `sfmodel_init()` (_optional_),
      3. maximization options (and others) using `sfmodel_opt()` (_optional_),
      4. estimation using `sfmodel_fit()`.
      Here we only highlight the first step of using `sfmodel_spec()` for different models.



!!! note "Additional Tags for Panel Model"
    In general, the panel model requires the additional tags of:
      - `sfpanel()`: the model id, which currently includes `TFE_CSW2014`， `TFE_WH2010`, `TRE`, `TimeDecay`.
      - `timevar()`: variable of time period,
      - `idvar()`: variable for individual identification.




## Panel True Fixed Effect Model

A general setup of this model is:

```math
\begin{aligned}
  & y_{it}  = \alpha_i +  x_{it} \beta + \epsilon_{it},\\
  & \epsilon_{it}  = v_{it} - u_{it},\\
  & v_{it} \sim N(0, \sigma_v^2),   \quad u_{it}  \sim N^+(\mu, \sigma_u^2),
\end{aligned} 
```

where ``\alpha_i`` is the time-invariant and individual-specific effect which is not directly observable. It is not the same as the inefficiency effect, the latter of which is represented by ``u_{it}``. Here ``\alpha_i`` is assumed to be a fixed parameter which allows arbitrary correlations with ``x_{it}``. 

Greene (2005) coins the term _true fixed effect_ for this setup in order to distinguish it from other panel SF models where the fixed effect has different interpretations. For instance, Schmidt and Sickles (1984) have ``\alpha_i`` in the model and is also a fixed parameter. Their model, however, does not have ``u_{it}``. They interpret the estimated ``\alpha_i``, after normalization, as the inefficiency effect. Thus their ``\alpha_i`` is not the kind of fixed effect in the traditional sense and could be argued as a mixture of the individual effect and the inefficiency effect. Thus, _true fixed effect_ emphasizes the existence of both of the individual effect ``\alpha_i`` and the inefficiency effect ``u_{it}`` in the model, both of distinct interpretations.


The challenge of estimating such a model is the incidental parameters problem arising from ``\alpha_i``. For a linear panel data model, it is common to get rid of ``\alpha_i`` before estimation by first-differencing or within-transforming the model. This approach, however, is thought to be infeasible to the above panel SF model because we could not derive the closed-form likelihood function after the model transformation. It was later proven to be not true (see Chen, Schmidt, and Wang 2014).




```@meta
Here ``N^+(0, \sigma_u^2)`` is a _half-normal distribution_ obtained by truncating the normal distribution ``N(0, \sigma_u^2)`` from below at 0. ``z_i^v`` and ``z_i^u`` are vectors of exogenous variables including a constant, and the two vectors need not be the same. If, for instance, ``z_i^v`` has only a constant, ``\sigma_v^2`` is a constant parameter.

Both ``\sigma_v^2`` and ``\sigma_u^2`` are parameterized using exponential functions to ensure positive values. In the case of ``\sigma_v^2``, it is ``\sigma_v^2 = \exp(c_v)``, where ``c_v \in R`` is an unconstrained constant, and the log-likelihood maximization is w.r.t. ``c_v`` (among others).
```

Let's assume we have a panel dataset containing the production data of ``N`` farmers at the annual frequency for ``T`` years. We could have ``T`` to be different across farmers, i.e., ``T_i``. We assume the dataset is named `df` and is [in the DataFrame format](@ref useDataFrame) and has the following column names (_aka_ variables):
  * `y`: production output,
  * `x1`, `x2`, `x3`: production input,
  * `z1`, `z2`: factors that may affect production efficiency,
  * `_cons`: a constant variable with values equal to 1,
  * `yr`: year of production,
  * `id`: individual farmer's identification number.

\


First, call in the main packages.
```julia
using SFrontiers
using CSV, DataFrames
```

#### TFE with dummy variables ([Greene 2005, Wang 2003](@ref literature))

This approach simply uses dummy variables to estimate ``\alpha_i``, ``i=1,\ldots,N``. Thus, it can be estimated using  [cross-sectional models](@ref crosssectional) by adding dummy variables in the `frontier()` equation. To generate dummy variables from `id` in the original DataFrame, one may try [this method](https://stackoverflow.com/questions/64565276/julia-dataframes-how-to-do-one-hot-encoding):

```julia
df = CSV.read("panel_example.csv", DataFrame; header=1, delim=",")
df[!, :_cons] .= 1.0

uid = unique(df.id);
df = transform(df, @. :id => ByRow(isequal(uid)) .=> Symbol(:dummy, uid))
```
We may want to see what is inside the data now.

```julia
julia> describe(df)
106×8 DataFrame
│ Row │ variable │ mean      │ min      │ median     │ max     │ nunique │ nmissing │ eltype   │
│     │ Symbol   │ Float64   │ Real     │ Float64    │ Real    │ Nothing │ Nothing  │ DataType │
├─────┼──────────┼───────────┼──────────┼────────────┼─────────┼─────────┼──────────┼──────────┤
│ 1   │ id       │ 50.5      │ 1        │ 50.5       │ 100     │         │          │ Int64    │
│ 2   │ time     │ 3.5       │ 1        │ 3.5        │ 6       │         │          │ Int64    │
│ 3   │ y        │ 0.425722  │ -3.76222 │ 0.400012   │ 3.66662 │         │          │ Float64  │
│ 4   │ x1       │ 0.510121  │ -2.41952 │ 0.517722   │ 3.79116 │         │          │ Float64  │
│ 5   │ x2       │ -0.019732 │ -3.09178 │ 0.00775225 │ 2.99767 │         │          │ Float64  │
│ 6   │ _cons    │ 1.0       │ 1.0      │ 1.0        │ 1.0     │         │          │ Float64  │
│ 7   │ dummy1   │ 0.01      │ 0        │ 0.0        │ 1       │         │          │ Bool     │
│ 8   │ dummy2   │ 0.01      │ 0        │ 0.0        │ 1       │         │          │ Bool     │
⋮
│ 98  │ dummy92  │ 0.01      │ 0        │ 0.0        │ 1       │         │          │ Bool     │
│ 99  │ dummy93  │ 0.01      │ 0        │ 0.0        │ 1       │         │          │ Bool     │
│ 100 │ dummy94  │ 0.01      │ 0        │ 0.0        │ 1       │         │          │ Bool     │
│ 101 │ dummy95  │ 0.01      │ 0        │ 0.0        │ 1       │         │          │ Bool     │
│ 102 │ dummy96  │ 0.01      │ 0        │ 0.0        │ 1       │         │          │ Bool     │
│ 103 │ dummy97  │ 0.01      │ 0        │ 0.0        │ 1       │         │          │ Bool     │
│ 104 │ dummy98  │ 0.01      │ 0        │ 0.0        │ 1       │         │          │ Bool     │
│ 105 │ dummy99  │ 0.01      │ 0        │ 0.0        │ 1       │         │          │ Bool     │
│ 106 │ dummy100 │ 0.01      │ 0        │ 0.0        │ 1       │         │          │ Bool     │
```

This approach may be easier to carry out using __SFrontiers__'s [matrix-input method](@id matrixinput) in order to avoid manually inputting the dummy variable names which are quite many.

```julia
xvar = Matrix(df[:, [:x1, :x2]])
alldummy = Matrix(df[:, 8:106]) # skip the first dummy to avoid multicollinearity
xMat = hcat(xvar, alldummy) # combine all of the frontier var
yMat = Matrix(df[:, [:y]])
cMat = Matrix(df[:, [:_cons]])

sfmodel_spec(sftype(prod), sfdist(half),  
             depvar(yMat), 
             frontier(xMat), 
             σᵤ²(cMat),
             σᵥ²(cMat))
```

The rest of the estimation procedures is the same as in other models. In addition to the computational problem (having to estimate a large number of parameters), the approach suffers from the incidental parameters problem (Wang and Ho 2010). The estimates are consistent only when ``T`` is large (Greene 2005).

\

#### TFE with skew-normal approach ([Chen, Schmidt, and Wang 2014](@ref literature))

The authors use results from the closed skew-normal literature and derive the model's closed-form likelihood function after first-differencing or within transforming the model. The model id is `TFE_CSW2014`.

```julia
sfmodel_spec(sfpanel(TFE_CSW2014), sftype(prod), sfdist(half),
             @timevar(yr), @idvar(id),
             @depvar(y), 
             @frontier(x1, x2, x3), 
             @σᵤ²(_cons),
             @σᵥ²(_cons))
```              
* The user does not have to first-difference or within-transform the data; the program will perform the transformation.
* `_cons` cannot be included in `@frontier()`, because ``\alpha_i`` is a fixed parameter, and including `_cons` in addition to a full set of ``\alpha_i`` would cause multicollinearity.
* The model does not support exogenous determinants of inefficiency.
* `sfdist()` only supports `half`.


\


#### TFE with scaling property and inefficiency determinants ([Wang and Ho 2010](@ref literature))

Wang and Ho (2010) propose a model where the ``u_{it}`` is modeled as

```math
\begin{aligned}
  u_{it} & = h(z_{it}; \delta)\cdot u_i^*,\\
  u_i^* & \sim N^+(\mu, \sigma_u^2),
\end{aligned}
```

where

```math
\begin{aligned}
   h(z_{it}; \delta) & = \exp(z_{it}\delta),\\
   \sigma_u^2 & = \exp(c_u).
\end{aligned}
```

Both of ``\mu`` and ``\sigma_u^2`` are constant, and ``\mu`` may equal 0 for a half-normal assumption.

The specification has two advantages: (1) The closed-form likelihood function can be derived using the conventional method. (2) The model can easily accommodate exogenous determinants of inefficiency. The model is `TFE_WH2010`.

```julia
sfmodel_spec(sfpanel(TFE_WH2010), sftype(prod), sfdist(trun),
             @timevar(yr), @idvar(id),
             @depvar(y), 
             @frontier(x1, x2, x3), 
             @μ(_cons),
             @hscale(z1, z2),                # h(.) function
             @σᵤ²(_cons),
             @σᵥ²(_cons))
```             
* The user does not have to first-difference or within-transform the data; the program will perform the transformation.
* `_cons` cannot be included in `@frontier()`, because ``\alpha_i`` is a fixed parameter.
* Giving `sfdist(half)` and omitting `@μ()` will estimate the model with the half-normal distribution.

\

## Panel True Random Effect Model

The model is attributed to [Greene (2005)](@ref literature). It assumes ``\alpha_i`` to be a value from a random variable. We assume that 

```math
\begin{aligned}
  \alpha_i  &  \sim N(0, \sigma_a^2),\\
  \sigma_a^2  & = \exp(c_a),
\end{aligned}  
```
where ``c_a \in R`` is a constant. The model id is `TRE`.

```julia
sfmodel_spec(sfpanel(TRE), sftype(prod), sfdist(half),
             @timevar(yr), @idvar(id),
             @depvar(y), 
             @frontier(x1, x2, _cons), 
             @σₐ²(_cons),
             @σᵤ²(_cons),
             @σᵥ²(_cons))
```              
* The constant `_cons` is included in `@frontier()` since ``\alpha_i`` is random.
* Giving `sfdist(trun)` and `@μ(_cons)` will estimate the model with the truncated-normal distribution.

\

## Panel Time Decay Model


The model is attributed to [Battese and Coelli (1992)](@ref literature).


```math
\begin{aligned}
  & y_{it}  = \alpha_0 +  x_{it} \beta + \epsilon_{it},\\
  & \epsilon_{it}  = v_{it} - u_{it},\\
  & v_{it} \sim N(\mu_i, \sigma_v^2),   \\
  & u_{it} \sim G(t) u_i^*, \quad u_i^* \sim N^+(0, \sigma_u^2).
\end{aligned} 
```
where 

```math 
\begin{aligned}
  G(t) & = \exp[\gamma (t_i − T_i)],\\
  \sigma_u^2 & = \exp(c_u).
\end{aligned}  
```

In the specification, ``G(t)`` is a function of time ``t_i``, and ``T_i = \mathrm{max}(t_i)`` is fixed for an individual. A positive estimate of ``\gamma`` thus indicates a decreasing of inefficiency over time (i.e., _time decay_). The ``\mu_i`` can be a function of individual specific variable or a constant (``\mu_i = \mu``). 

Note that the frontier function has an overall intercept ``\alpha_0`` but there is no individual effect ``\alpha_i`` in the model, and so it does not belong to the class of true-fixed or true-random effect model. The model is `TimeDecay`.


Kumbhakar and Wang (2005) use a variant of the model to study growth convergence of a panel of countries, where they have ``\mu_i`` as the country's initial capital stock per capita when data began in the 1960s. We assume ``w`` to be such a variable in the following example.



```julia
using Statistics, DataFramesMeta  # helps to get `yearT_i = yr_i - max(yr_i)`

# assume `df` is already loaded 
gdf = groupby(df, :id)               # info of data grouping
df = @transform(gdf, yearT = :yr .-  maximum(:yr))   # create yearT


sfmodel_spec(sfpanel(TimeDecay), sftype(prod), sfdist(trun),
             @timevar(yr), @idvar(id),
             @depvar(y), 
             @frontier(x1, x2, x3, _cons), 
             @μ(w, _cons),
             @gamma(yearT),              # the G(.) function
             @σᵤ²(_cons),
             @σᵥ²(_cons))
```