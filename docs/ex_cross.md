# [Examples of Cross-Sectional Stochastic Frontier Models](@id crosssectional)


```@contents
Pages = ["ex_cross.md"]
Depth = 5
```


!!! note "Reminder" 
    As shown in the [A Detailed Example](@ref detailedexample) section, __SFrontiers__ estimates stochastic frontier models in four steps:
    1. model specification using `sfmodel_spec()`,
    2. initial values using `sfmodel_init()` (_optional_),
    3. maximization options (and others) using `sfmodel_opt()` (_optional_),
    4. estimation using `sfmodel_fit()`.
    Here we only highlight the first step of using `sfmodel_spec()` for different models.


## Normal Half-Normal


A general setup of the model is:

```math
\begin{aligned}
  y_i & = x_i \beta + \epsilon_i,\\
  \epsilon_i & = v_i - u_i,\\
  v_i \sim N(0, \sigma_v^2),  & \quad u_i  \sim N^+(0, \sigma_u^2),
\end{aligned} 
```
where
```math
\begin{aligned}
  \sigma_v^2  = \exp(z_i^v \rho), & \quad  \sigma_u^2  = \exp(z_i^u \gamma).
\end{aligned}
```

Here ``N^+(0, \sigma_u^2)`` is a _half-normal distribution_ obtained by truncating the normal distribution ``N(0, \sigma_u^2)`` from below at 0. ``z_i^v`` and ``z_i^u`` are vectors of exogenous variables including a constant, and the two vectors need not be the same. If, for instance, ``z_i^v`` has only a constant, ``\sigma_v^2`` is a constant parameter.

Both ``\sigma_v^2`` and ``\sigma_u^2`` are parameterized using exponential functions to ensure positive values. In the case of ``\sigma_v^2``, it is ``\sigma_v^2 = \exp(c_v)``, where ``c_v \in R`` is an unconstrained constant, and the log-likelihood maximization is w.r.t. ``c_v`` (among others).





We continue the example of estimating the stochastic production frontier of Indian farmers. Some specifications may not make much economic sense, and they are used only for the sake of examples.


```julia
using SFrontiers        # main packages
using DataFrames, CSV   # handling data

df = CSV.read("sampledata.csv", DataFrame; header=1, delim=",")
df[!, :_cons] .= 1.0;         # append column _cons as a column of 1 
```

This is the content of the data.

```julia
julia> describe(df)
11×8 DataFrame
│ Row │ variable │ mean     │ min      │ median  │ max     │ nunique │ nmissing │ eltype   │
│     │ Symbol   │ Float64  │ Real     │ Float64 │ Real    │ Nothing │ Nothing  │ DataType │
├─────┼──────────┼──────────┼──────────┼─────────┼─────────┼─────────┼──────────┼──────────┤
│ 1   │ yvar     │ 7.27812  │ 3.58666  │ 7.28586 │ 9.80335 │         │          │ Float64  │
│ 2   │ Lland    │ 1.05695  │ -1.60944 │ 1.14307 │ 3.04309 │         │          │ Float64  │
│ 3   │ PIland   │ 0.146997 │ 0.0      │ 0.0     │ 1.0     │         │          │ Float64  │
│ 4   │ Llabor   │ 6.84951  │ 3.2581   │ 6.72263 │ 9.46622 │         │          │ Float64  │
│ 5   │ Lbull    │ 5.64161  │ 2.07944  │ 5.68358 │ 8.37008 │         │          │ Float64  │
│ 6   │ Lcost    │ 4.6033   │ 0.0      │ 5.1511  │ 8.73311 │         │          │ Float64  │
│ 7   │ yr       │ 5.38007  │ 1        │ 5.0     │ 10      │         │          │ Int64    │
│ 8   │ age      │ 53.8856  │ 26       │ 53.0    │ 90      │         │          │ Int64    │
│ 9   │ school   │ 2.02583  │ 0        │ 0.0     │ 10      │         │          │ Int64    │
│ 10  │ yr_1     │ 5.38007  │ 1        │ 5.0     │ 10      │         │          │ Int64    │
│ 11  │ _cons    │ 1.0      │ 1        │ 1.0     │ 1       │         │          │ Int64    │
```

\

#### vanilla [(Aigner, Lovell, and Schmidt 1977)](@ref literature)

```julia
sfmodel_spec(sftype(prod), sfdist(half),
             @depvar(yvar), 
             @frontier(Lland, PIland, Llabor, Lbull, Lcost, yr, _cons), 
             @σᵤ²(_cons),
             @σᵥ²(_cons))
```

\

#### with inefficiency determinants [(Caudill and Ford 1993)](@ref literature)

```julia
sfmodel_spec(sftype(prod), sfdist(half),
             @depvar(yvar), 
             @frontier(Lland, PIland, Llabor, Lbull, Lcost, yr, _cons), 
             @σᵤ²(age, school, yr, _cons),
             @σᵥ²(_cons))
```

\


#### with inefficiency determinants and production uncertainty 

The ``\sigma_v^2`` is sometimes interpreted as production uncertainty (Bera and Sharma 1999). We show that the uncertainty can be parameterized by exogenous variables in the model. Here we assume that the uncertainty may be influenced by year (`yr`) effect (changes in the weather condition, etc.).

```julia
sfmodel_spec(sftype(prod), sfdist(half),
             @depvar(yvar), 
             @frontier(Lland, PIland, Llabor, Lbull, Lcost, yr, _cons), 
             @σᵤ²(age, school, _cons),
             @σᵥ²(yr, _cons))
```

\

## Normal Truncated-Normal

The setup is the same as in the [A Detailed Example](@ref detailedexample) section. We repeated it here for clarity.


```math
\begin{aligned}
 y_i & = x_i \beta + \epsilon_i,\\
 \epsilon_i & = v_i - u_i,\\
 v_i \sim N(0, \sigma_v^2), \quad & u_i  \sim N^+(\mu, \sigma_u^2), 
\end{aligned}
```
where

```math
\begin{aligned}
  \sigma_v^2  = & \exp(z_i^v \rho),\\
  \mu  = z_i^m \delta, & \quad  \sigma_u^2  = \exp(z_i^u \gamma).
\end{aligned}
```

Again, ``z_i^v``, ``z_i^m``, ``z_i^u`` could be a vector of variables or a single constant variable.


\


#### vanilla [(Stevenson 1980)](@ref literature)


```julia
sfmodel_spec(sftype(prod), sfdist(trun),
             @depvar(yvar), 
             @frontier(Lland, PIland, Llabor, Lbull, Lcost, yr, _cons), 
             @mu(_cons),
             @σᵤ²(_cons),
             @σᵥ²(_cons))
```


\


#### inefficiency determinants in ``\mu`` [(Kumbhakar, Ghosh, and McGuckin, 1991; Huang and Liu, 1994; Battese and Coelli, 1995)](@ref literature)


```julia
sfmodel_spec(sftype(prod), sfdist(trun),
             @depvar(yvar), 
             @frontier(Lland, PIland, Llabor, Lbull, Lcost, yr, _cons), 
             @mu(age, school, _cons),
             @σᵤ²(_cons),
             @σᵥ²(_cons))
```

\


#### inefficiency determinants in ``\mu`` and ``\sigma_u^2``, _aka_ the non-monotonic effecct model ([Wang 2002](@ref literature))

We show details of this model in the section [A Detailed Example](@ref detailedexample).

```julia
sfmodel_spec(sftype(prod), sfdist(trun),
             @depvar(yvar), 
             @frontier(Lland, PIland, Llabor, Lbull, Lcost, yr, _cons), 
             @mu(age, school, _cons),
             @σᵤ²(age, school, _cons),
             @σᵥ²(_cons))
```

\

## Normal Truncated-Normal with the Scaling Property


```math
\begin{aligned}
 y_i & = x_i \beta + \epsilon_i,\\
 \epsilon_i & = v_i - u_i,\\
 v_i \sim N(0, \sigma_v^2), \quad & u_i \sim h(z_i^h, \delta) N^+(\mu, \sigma_u^2),
\end{aligned}
```
where
```math
\begin{aligned}
 & \sigma_v^2  =  \exp(z_i^v \rho),  \\
 h(z_i^h, \delta) = & \exp(z_i^h \delta),  \quad   \sigma_u^2  = \exp(c_u).
 \end{aligned}
```

Both of the ``\mu`` and ``\sigma_u^2`` are constant parameters in this model, and thus ``z_i^h`` *__cannot__* contain a constant for the identification purpose.  The ``h(\cdot)`` is a non-negative function, and we assume an exponential function with linear ``z_i^h`` in this model.


\

#### the specification ([Wang and Schmidt 2002, Alvarez, Amsler, and Schmidt 2006](@ref literature))

```julia
sfmodel_spec(sftype(prod), sfdist(trun_scale), 
             @depvar(ly), 
             @frontier(llabor, lfeed, lcattle, lland, _cons), 
             @hscale(comp),       # cannot contain a constant
             @μ(_cons),
             @σᵤ²(_cons),
             @σᵥ²(_cons))
```              


\

## Normal Exponential

This model assumes ``u_i`` follows an exponential distribution. 

```math
\begin{aligned}
u_i  \sim \mathrm{Exp}(\sigma_u^2),
\end{aligned} 
```
where ``\sigma_u^2`` is the scale parameter such that ``E(u_i) = \sigma_u`` and ``Var(u_i) = \sigma_u^2``. The ``\sigma_u^2`` may be parameterized by a vector of variables, as we show in the following example.

```julia
sfmodel_spec(sftype(prod), sfdist(expo),
             @depvar(ly), 
             @frontier(llabor, lfeed, lcattle, lland, _cons), 
             @σᵤ²(comp, _cons),
             @σᵥ²(_cons))
```              
