### Estimation Overview

Consider a typical stochastic frontier model,
```math
  \begin{aligned}
   y_i = f(x_i; \beta) + v_i - u_i(z_i),
  \end{aligned}
```
where $v_i$ and $u_i(z_i) >0$ are both stochastic. 



`SFrontiers` provides utilities to estimate the model using the maximum likelihood approach.


##### 1. Specify the model using `sfmodel_spec()`
* a _cross-sectional_ (the default) or a _panel data_ (require tags) model
* a _production_ or a _cost_  type of model
* distribution assumptions on $u_i$
* names of variables for $y_i$, $x_i$, and $z_i$ (if any)


##### 2. (optional) Provide initial values using `sfmodel_init()`
* initial values for the maximum likelihood estimation
* a full list or a partial list for some of the equations

##### 3. (optional) Provide parameters for numerical maximization and other controls using `sfmodel_opt()`
* algorithms (e.g., `Nelder-Mead`, `BFGS`, `Newton`, etc.)
* maximum iteration number
* convergence criterion
* information to print on screen
* whether calculating inefficiency index, marginal effect, etc.



##### 4. Start the Numerical Maximization Process using `sfmodel_fit()`
* name of the dataset


##### 5. Conduct Post-Estimation Analysis
* hypothesis testing 
* the inefficiency index of Jondrow et al. (1982) or the efficiency index of Battese and Coelli (1988)
* marginal effect of the inefficiency determinants


