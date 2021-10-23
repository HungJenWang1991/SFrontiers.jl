using SFrontiers, DataFrames, Test


@testset verbose=true "Method of Moments" begin

	data= [-1.06966    1.53076     0.346522  1
			-1.55598   -1.54822     0.0296991  1
			-1.06309    0.0860199  -0.546911  1
			0.396344  -1.59922    -1.62234  1
			-0.367106  -1.31003     1.67005  1]

	 df = DataFrame(data, [:y, :x1, :x2, :cons])

    re1 = sfmodel_MoMTest(sftype(prod), sfdist(half),
                     @depvar(y), @frontier( cons, x1),
                     data = df,
                     ω = (0.5, 1, 2),
                     verbose=false )

	@test re1.coeff[1]  ≈ -0.31582 atol = 1e-5
	@test re1.jlms[1]   ≈  0.44173 atol = 1e-5
	@test re1.MoM_loglikelihood ≈ -4.74344 atol = 1e-5
end


@testset verbose=true "Part 1 (cross-section) of log-likelihood functions, marginal effect, jlms & bc index" begin

	data= [-1.06966    1.53076     0.346522  1
			-1.55598   -1.54822     0.0296991  1
			-1.06309    0.0860199  -0.546911  1
			0.396344  -1.59922    -1.62234  1
			-0.367106  -1.31003     1.67005  1]

	y = reshape(data[:,1], length(data[:,1]), 1)  # matrix     
	x = z = w = q = data[:,2:3]
	v = data[:,4:4]
	PorC = 1
	nobs = 5
	dum2 = ()
	tvar = [1,2,3,1,2]
	ivar = [1,1,1,2,2]
	temp1 = [[1,2,3],[4,5]]
	temp2 = [3,2]
	idt = hcat(temp1, temp2)

	#--- N-HN ----

	posvec = (begx=1, endx=2, begz=0, endz=0, begq=0, endq=0, begw=3, endw=4, begv=5, endv=5)
	rho = ones(5,1); rho2=vec(rho);
	@test  LL_T(Half, y,x,(),(),w,v,1,5,posvec,rho, dum2, nothing) ≈ 10.32890 atol=1e-5

	jlms, bc = jlmsbc(Half, PorC, posvec, rho2, y, x, (), (), w, v, nothing)
    @test jlms[1,1] ≈ 2.27294 atol = 1e-5
    @test bc[1,1] ≈ 0.190717 atol = 1e-5	

	nofvar = (nofobs=5, nofx=2, nofz=0, nofq=0,	nofw=2, nofv=1, nofpara=5, nofmarg = 0+0+2)
    sfmodel_spec(sftype(prod), sfdist(half), depvar(y), frontier(x),  σᵤ²(w), σᵥ²(v))
    aa, bb = get_marg(Half, posvec, nofvar, rho2, z, (), w)
    @test aa[1,1] ≈ 1.0199 atol=1e-5
	@test bb[1] ≈ 0.41615 atol = 1e-5


	# -- N-TN ----

	posvec = (begx=1, endx=2, begz=3, endz=4, begq=0, endq=0, begw=5, endw=6, begv=7, endv=7)
	rho = ones(7,1); rho2 = vec(rho);
	@test LL_T(Trun, y,x,z,(),w,v,1,5,posvec,rho, dum2, nothing)  ≈ 10.10842 atol=1e-5
	
	jlms, bc = jlmsbc(Trun, PorC, posvec, rho2, y, x, z, (), w, v, nothing)
    @test jlms[1,1] ≈ 2.72632 atol = 1e-5
    @test bc[1,1] ≈ 0.13462 atol = 1e-5	

	nofvar = (nofobs=5, nofx=2, nofz=2, nofq=0,	nofw=2, nofv=1, nofpara=7, nofmarg = 2+0+2)
    sfmodel_spec(sftype(prod), sfdist(trun), depvar(y), frontier(x), μ(z), σᵤ²(z), σᵥ²(v))
    aa, bb = get_marg(Trun, posvec, nofvar, rho2, z, (), w)
    @test aa[1,1] ≈ 0.55182 atol=1e-5
	@test bb[1] ≈ 0.26175 atol = 1e-5


	# -- N-Expo ----

	posvec = (begx=1, endx=2, begz=0, endz=0, begq=0, endq=0, begw=3, endw=4, begv=5, endv=5)
	rho = ones(5,1); rho2=vec(rho)
	@test LL_T(Expo, y,x,(),(),w,v,1,5,posvec,rho, dum2, nothing) ≈ 10.66291 atol=1e-5

	jlms, bc = jlmsbc(Expo, PorC, posvec, rho2, y, x, (), (), w, v, nothing)
    @test jlms[1,1] ≈ 2.27578 atol = 1e-5
    @test bc[1,1] ≈ 0.207597 atol = 1e-5	

	nofvar = (nofobs=5, nofx=2, nofz=0, nofq=0,	nofw=2, nofv=1, nofpara=5, nofmarg = 0+0+2)
    sfmodel_spec(sftype(prod), sfdist(e), depvar(y), frontier(x),  σᵤ²(w), σᵥ²(v))
    aa, bb = get_marg(Expo, posvec, nofvar, rho2, z, (), w)
    @test aa[1,1] ≈ 1.27825 atol=1e-5
	@test bb[1] ≈ 0.52157 atol = 1e-5


	# -- scaling ---

	posvec = (begx=1, endx=2, begz=3, endz=3, begq=4, endq=5, begw=6, endw=7, begv=8, endv=8)
	rho = ones(8,1); rho2=vec(rho);
	@test  LL_T(Trun_Scale, y,x,v,q,w,v,1,5,posvec,rho, dum2, nothing)  ≈ 11.93947 atol=1e-5

	jlms, bc = jlmsbc(Trun_Scale, PorC, posvec, rho2, y, x, v, q, w, v, nothing)
    @test jlms[1,1] ≈ 3.11162 atol = 1e-5
    @test bc[1,1] ≈ 0.11505 atol = 1e-5	

	nofvar = (nofobs=5, nofx=2, nofz=1, nofq=2,	nofw=2, nofv=1, nofpara=8, nofmarg = 1+2+2)
	sfmodel_spec(sftype(prod), sfdist(trun_scale), depvar(y), frontier(x), hscale(q), μ(v), σᵤ²(w), σᵥ²(v))
    aa, bb = get_marg(Trun_Scale, posvec, nofvar, rho2, v, q, w)
    @test aa[1,1] ≈ 16.00399 atol=1e-5
	@test bb[1] ≈ 3.80746 atol = 1e-5

end 	


@testset verbose=true "Part 2 (panel) of log-likelihood functions, marginal effect, jlms & bc index" begin

	data= [-1.06966    1.53076     0.346522  1
			-1.55598   -1.54822     0.0296991  1
			-1.06309    0.0860199  -0.546911  1
			0.396344  -1.59922    -1.62234  1
			-0.367106  -1.31003     1.67005  1]

	y = reshape(data[:,1], length(data[:,1]), 1)  # matrix     
	x = z = w = q = data[:,2:3]
	v = data[:,4:4]
	PorC = 1
	nobs = 5
	dum2 = ()
	tvar = [1,2,3,1,2]
	ivar = [1,1,1,2,2]
	temp1 = [[1,2,3],[4,5]]
	temp2 = [3,2]
	idt = hcat(temp1, temp2)


	# -- Wang and Ho 2010, half ----

	posvec = (begx=1, endx=2, begz=0, endz=0, begq=3, endq=4, begw=5, endw=5, begv=6, endv=6)
	rho = ones(6,1); rho2=vec(rho)
	@test LL_T(PFEWHH, data[:,1],x,(),q,(),(),1,5,posvec,rho, idt, nothing)  ≈ 5.97091 atol=1e-5

	jlms, bc = jlmsbc(PFEWHH, PorC, posvec, rho2, data[:,1], x, (), q, w, v, idt)
    @test jlms[1,1] ≈ 3.13039 atol = 1e-5
    @test bc[1,1] ≈ 0.13926 atol = 1e-5	

	nofvar = (nofobs=5, nofx=2, nofz=0, nofq=2,	nofw=1, nofv=1, nofpara=6, nofmarg = 0+2+1)
	sfmodel_spec(sfpanel(TFE_WH2010), sftype(prod), sfdist(half), timevar(tvar), idvar(ivar), 
	             depvar(y), frontier(x), hscale(q),  σᵤ²(v), σᵥ²(v))
    aa, bb = get_marg(PFEWHH, posvec, nofvar, rho2, v, q, v)

    @test aa[1,1] ≈ 8.59766 atol=1e-5
	@test bb[1] ≈ 2.33071 atol = 1e-5

	# --- Wang and Ho 2010, truncated ----

	posvec = (begx=1, endx=2, begz=3, endz=3, begq=4, endq=5, begw=6, endw=6, begv=7, endv=7)
	rho = ones(7,1); rho2=vec(rho)
	@test LL_T(PFEWHT, data[:,1],x,(),q,(),(),1,5,posvec,rho, idt, nothing)  ≈ 6.19169 atol=1e-5

	jlms, bc = jlmsbc(PFEWHT, PorC, posvec, rho2, data[:,1], x, v, q, w, v, idt)
    @test jlms[1,1] ≈ 3.31418 atol = 1e-5
    @test bc[1,1] ≈ 0.124392 atol = 1e-5	

	nofvar = (nofobs=5, nofx=2, nofz=1, nofq=2,	nofw=1, nofv=1, nofpara=7, nofmarg = 1+2+1)
	sfmodel_spec(sfpanel(TFE_WH2010), sftype(prod), sfdist(trun), timevar(tvar), idvar(ivar), 
	             depvar(y), frontier(x), hscale(q), μ(v), σᵤ²(v), σᵥ²(v))
    aa, bb = get_marg(PFEWHT, posvec, nofvar, rho2, v, q, v)

    @test aa[1,1] ≈ 11.44913 atol=1e-5
	@test bb[1] ≈ 3.10371 atol = 1e-5

	# --- time decay ----

	posvec = (begx=1, endx=2, begz=3, endz=4, begq=5, endq=6, begw=7, endw=7, begv=8, endv=8)
	rho = ones(8,1); rho2 = vec(rho)
	@test LL_T(PanDecay, y, x, z, q, w,v,1,5,posvec,rho, idt, nothing)  ≈ 11.79293 atol=1e-5

	jlms, bc = jlmsbc(PanDecay, PorC, posvec, rho2, y, x, z, q, w, v, idt)
    @test jlms[1,1] ≈ 3.282 atol = 1e-5
    @test bc[1,1] ≈ 0.10012 atol = 1e-5	

	nofvar = (nofobs=5, nofx=2, nofz=2, nofq=2,	nofw=1, nofv=1, nofpara=8, nofmarg = 2+2+1)
	sfmodel_spec(sfpanel(TimeDecay), sftype(prod), sfdist(trun), timevar(tvar), idvar(ivar), 
	             depvar(y), frontier(x), μ(z), gamma(q), σᵤ²(v), σᵥ²(v))
    aa, bb = get_marg(PanDecay, posvec, nofvar, rho2, z, q, v)
    @test aa[1,1] ≈ 4.38273 atol=1e-5
	@test bb[1] ≈ 1.04393 atol = 1e-5

	# --- Kumbhakar 1990 ----

	posvec = (begx=1, endx=2, begz=3, endz=4, begq=5, endq=6, begw=7, endw=7, begv=8, endv=8)
	rho = ones(8,1); rho2 = vec(rho)
	@test LL_T(PanKumb90, y, x, z, q, w,v,1,5,posvec,rho, idt, nothing)  ≈ 13.42116 atol=1e-5

	jlms, bc = jlmsbc(PanKumb90, PorC, posvec, rho2, y, x, z, q, w, v, idt)
    @test jlms[1,1] ≈ 0.23543 atol = 1e-5
    @test bc[1,1] ≈ 0.79868 atol = 1e-5	

	nofvar = (nofobs=5, nofx=2, nofz=2, nofq=2,	nofw=1, nofv=1, nofpara=8, nofmarg = 2+2+1)
	sfmodel_spec(sfpanel(Kumbhakar1990), sftype(prod), sfdist(trun), timevar(tvar), idvar(ivar), 
	             depvar(y), frontier(x), μ(z), gamma(q), σᵤ²(v), σᵥ²(v))
    aa, bb = get_marg(PanKumb90, posvec, nofvar, rho2, z, q, v)
    @test aa[1,1] ≈ 0.17797 atol=1e-5
	@test bb[1] ≈ 0.29235 atol = 1e-5


	# --- CSW 2014, half ---------

	posvec = (begx=1, endx=2, begz=0, endz=0, begq=0, endq=0, begw=3, endw=3, begv=4, endv=4)
	rho = ones(4,1); rho2=vec(rho)
	@test LL_T(PFECSWH, data[:,1], x, (), (), (), (),1,5,posvec,rho, idt, nothing) ≈ 8.61254 atol=1e-5

	jlms, bc = jlmsbc(PFECSWH, PorC, posvec, rho2, data[:,1], x, z, (), v, v, idt)
    @test jlms[1,1] ≈ 1.74963 atol = 1e-5
    @test bc[1,1] ≈ 0.26295 atol = 1e-5	

    # This model does not allow exog determinants hence no marginal effect.

	# --- true random effect, half ----------

	posvec = (begx=1, endx=2, begz=0, endz=0, begq=3, endq=3, begw=4, endw=4, begv=5, endv=5)
	rho = ones(5,1); rho2=vec(rho)
	@test LL_T(PTREH, y, x, (), q, (), (),1,5,posvec,rho, idt, nothing) ≈ 11.70517 atol=1e-5

	jlms, bc = jlmsbc(PTREH, PorC, posvec, rho2, y, x, (), q, v, v, idt)
    @test jlms[1,1] ≈ 0 atol = 1e-5  # not available
    @test bc[1,1] ≈ 0.26956 atol = 1e-5	

	# This model does not allow exog determinants hence no marginal effect.

	# --- true random effect, trun -----

	posvec = (begx=1, endx=2, begz=1, endz=1, begq=4, endq=4, begw=5, endw=5, begv=6, endv=6)
	rho = ones(6,1); rho2=vec(rho)
	@test LL_T(PTRET, y, x, v, q, (), (),1,5,posvec,rho, idt, nothing) ≈ 11.98406 atol=1e-5

	jlms, bc = jlmsbc(PTRET, PorC, posvec, rho2, y, x, v, q, v, v, idt)
    @test jlms[1,1] ≈ 0 atol = 1e-5  # not available
    @test bc[1,1] ≈ 0.18474 atol = 1e-5	

	# This model does not allow exog determinants hence no marginal effect.

end


@testset verbose=true "full MLE: N-HN(z), boot (may take time)" begin

	data = [-0.930756     0.679107   -0.117138   1.0
			-0.611645     0.828413   -0.601254   1.0
			-1.31106     -0.353007    1.14228    1.0
			-0.00456725  -0.134854   -0.0886163  1.0
			-3.36723      0.586617    0.279466   1.0
			-1.49024      0.297336    0.111422   1.0
			1.33926      0.0649475  -0.357884   1.0
			0.0539396   -0.109017    0.473714   1.0
			-3.75177     -0.51421     0.300234   1.0
			-0.485357     1.57433    -0.762677   1.0
			-0.643492    -0.688907    1.42305    1.0
			0.675002    -0.762804    0.408387   1.0
			-2.48183      0.397482    0.588621   1.0
			-2.96432      0.81163    -0.296278   1.0
			-1.04406     -0.346355    0.691111   1.0
			-4.63685     -0.187573    0.506874   1.0
			-2.07605     -1.60726    -0.0569299  1.0
			-2.85586     -2.48079    -1.77102    1.0
			0.207866     2.27623     1.59062    1.0
			-1.38462      0.219693    1.39706    1.0]

	y = data[:,1];
	xvar = hcat(data[:,2], data[:,4]);
	_con = data[:,4];
	zvar = hcat(data[:,3], data[:,4]);

	myini = [0.5, 0.5, 0.5, log(3), log(1)]

	sfmodel_spec(sftype(prod), sfdist(half), 
			depvar(y), frontier(xvar), 
			sigma_u_2(zvar), sigma_v_2(_con))

	sfmodel_init(all_init(myini))

	sfmodel_opt(warmstart_solver(NelderMead()),  
			warmstart_maxIT(200),
			main_solver(Newton()), 
			main_maxIT(2000), 
			tolerance(1e-8), 
			verbose(false), banner(false)
			)

	res = sfmodel_fit();

	boot1, d1 = sfmodel_boot_marginal(result=res, R=10, seed=123, getBootData=true)

	@test res.coeff[1] ≈ 0.30200 atol=1e-5 
	@test res.coeff[2] ≈ 0.13454 atol=1e-5 
	@test res.marginal[1,1] ≈ -0.08340  atol=1e-5
	@test res.jlms[1] ≈ 1.21327  atol=1e-5
	@test res.bc[1] ≈ 0.37075 atol=1e-5
	@test boot1[1] ≈ 1.60067 atol=7e-5
	@test d1[1] ≈ -0.47043 atol=1e-5

    ci90 = sfmodel_CI(bootdata=d1, observed=res.marginal_mean, level=0.10, verbose=false);
    @test ci90[1][1] ≈ -0.38963 atol=1e-5


	pred1 = sfmodel_predict(@eq(frontier))
	@test pred1[1] ≈ 0.33963 atol=1e-5 

end



@testset verbose=true "full MLE: N-Exp (may take time)" begin

	data = [-0.930756     0.679107   -0.117138   1.0
			-0.611645     0.828413   -0.601254   1.0
			-1.31106     -0.353007    1.14228    1.0
			-0.00456725  -0.134854   -0.0886163  1.0
			-3.36723      0.586617    0.279466   1.0
			-1.49024      0.297336    0.111422   1.0
			1.33926      0.0649475  -0.357884   1.0
			0.0539396   -0.109017    0.473714   1.0
			-3.75177     -0.51421     0.300234   1.0
			-0.485357     1.57433    -0.762677   1.0
			-0.643492    -0.688907    1.42305    1.0
			0.675002    -0.762804    0.408387   1.0
			-2.48183      0.397482    0.588621   1.0
			-2.96432      0.81163    -0.296278   1.0
			-1.04406     -0.346355    0.691111   1.0
			-4.63685     -0.187573    0.506874   1.0
			-2.07605     -1.60726    -0.0569299  1.0
			-2.85586     -2.48079    -1.77102    1.0
			0.207866     2.27623     1.59062    1.0
			-1.38462      0.219693    1.39706    1.0]

	y = data[:,1];
	xvar = hcat(data[:,2], data[:,4]);
	_con = data[:,4];
	zvar = hcat(data[:,3], data[:,4]);

	myini = [0.5, 0.5,  log(3), log(1)]

	sfmodel_spec(sftype(prod), sfdist(e), 
			depvar(y), frontier(xvar), 
			sigma_u_2(_con), sigma_v_2(_con))

	sfmodel_init(all_init(myini))

	sfmodel_opt(warmstart_solver(NelderMead()),  
			warmstart_maxIT(200),
			main_solver(Newton()), 
			main_maxIT(2000), 
			tolerance(1e-8),
			verbose(false), banner(false)
			)

	res = sfmodel_fit();

	@test res.coeff[1] ≈ 0.34725 atol=1e-5 
	@test res.coeff[2] ≈ -0.51886 atol=1e-5 
	@test res.jlms[1] ≈ 0.67725 atol=1e-5
	@test res.bc[1] ≈ 0.57816 atol=1e-5

	pred1 = sfmodel_predict(@eq(frontier))
	@test pred1[1] ≈  -0.28303 atol=1e-5 


end


@testset verbose=true "full MLE: N-HN (may take time)" begin

	data = [-0.930756     0.679107   -0.117138   1.0
			-0.611645     0.828413   -0.601254   1.0
			-1.31106     -0.353007    1.14228    1.0
			-0.00456725  -0.134854   -0.0886163  1.0
			-3.36723      0.586617    0.279466   1.0
			-1.49024      0.297336    0.111422   1.0
			1.33926      0.0649475  -0.357884   1.0
			0.0539396   -0.109017    0.473714   1.0
			-3.75177     -0.51421     0.300234   1.0
			-0.485357     1.57433    -0.762677   1.0
			-0.643492    -0.688907    1.42305    1.0
			0.675002    -0.762804    0.408387   1.0
			-2.48183      0.397482    0.588621   1.0
			-2.96432      0.81163    -0.296278   1.0
			-1.04406     -0.346355    0.691111   1.0
			-4.63685     -0.187573    0.506874   1.0
			-2.07605     -1.60726    -0.0569299  1.0
			-2.85586     -2.48079    -1.77102    1.0
			0.207866     2.27623     1.59062    1.0
			-1.38462      0.219693    1.39706    1.0]

	y = data[:,1];
	xvar = hcat(data[:,2], data[:,4]);
	_con = data[:,4];
	zvar = hcat(data[:,3], data[:,4]);

	myini = [0.5, 0.5,  log(3), log(1)]

	sfmodel_spec(sftype(prod), sfdist(half), 
			depvar(y), frontier(xvar), 
			sigma_u_2(_con), sigma_v_2(_con))

	sfmodel_init(all_init(myini))

	sfmodel_opt(warmstart_solver(NelderMead()),  
			warmstart_maxIT(200),
			main_solver(Newton()), 
			main_maxIT(2000), 
			tolerance(1e-8),
			verbose(false), banner(false)
			)

	res = sfmodel_fit();

	@test res.coeff[1] ≈ 0.32107 atol=1e-5 
	@test res.coeff[2] ≈ 0.11681 atol=1e-5 
	@test res.jlms[1] ≈ 1.20146 atol=1e-5
	@test res.bc[1] ≈ 0.37468 atol=1e-5

	pred1 = sfmodel_predict(@eq(frontier))
	@test pred1[1] ≈  0.33485 atol=1e-5 

end


@testset verbose=true "full MLE: N-TN(z) (may take time)" begin

	data = [-0.930756     0.679107   -0.117138   1.0
			-0.611645     0.828413   -0.601254   1.0
			-1.31106     -0.353007    1.14228    1.0
			-0.00456725  -0.134854   -0.0886163  1.0
			-3.36723      0.586617    0.279466   1.0
			-1.49024      0.297336    0.111422   1.0
			1.33926      0.0649475  -0.357884   1.0
			0.0539396   -0.109017    0.473714   1.0
			-3.75177     -0.51421     0.300234   1.0
			-0.485357     1.57433    -0.762677   1.0
			-0.643492    -0.688907    1.42305    1.0
			0.675002    -0.762804    0.408387   1.0
			-2.48183      0.397482    0.588621   1.0
			-2.96432      0.81163    -0.296278   1.0
			-1.04406     -0.346355    0.691111   1.0
			-4.63685     -0.187573    0.506874   1.0
			-2.07605     -1.60726    -0.0569299  1.0
			-2.85586     -2.48079    -1.77102    1.0
			0.207866     2.27623     1.59062    1.0
			-1.38462      0.219693    1.39706    1.0]

	y = data[:,1];
	xvar = hcat(data[:,2], data[:,4]);
	_con = data[:,4];
	zvar = hcat(data[:,3], data[:,4]);

	myini = [0.5, 0.5, 0.1,  0.5, log(3), log(1)]

	sfmodel_spec(sftype(prod), sfdist(trun), mu(_con),
			depvar(y), frontier(xvar), 
			sigma_u_2(zvar), sigma_v_2(_con))

	sfmodel_init(all_init(myini))

	sfmodel_opt(warmstart_solver(NelderMead()),  
			warmstart_maxIT(200),
			main_solver(Newton()), # may try warmstart_delta=0.2
			main_maxIT(2000), 
			tolerance(1e-8),
			verbose(false),	banner(false)
			)

	res = sfmodel_fit();

	@test res.coeff[1] ≈ 0.29602 atol=1e-5 
	@test res.coeff[2] ≈ 0.27351 atol=1e-5 
	@test res.marginal[1,1] ≈ -0.07106 atol=1e-5
	@test res.jlms[1] ≈ 1.35874 atol=1e-5
	@test res.bc[1] ≈ 0.32483 atol=1e-5
	
	pred1 = sfmodel_predict(@eq(frontier))
	@test pred1[1] ≈  0.47454 atol=1e-5 

end



@testset verbose=true "full MLE: N-TN (may take time) " begin

	data = [-0.930756     0.679107   -0.117138   1.0
			-0.611645     0.828413   -0.601254   1.0
			-1.31106     -0.353007    1.14228    1.0
			-0.00456725  -0.134854   -0.0886163  1.0
			-3.36723      0.586617    0.279466   1.0
			-1.49024      0.297336    0.111422   1.0
			1.33926      0.0649475  -0.357884   1.0
			0.0539396   -0.109017    0.473714   1.0
			-3.75177     -0.51421     0.300234   1.0
			-0.485357     1.57433    -0.762677   1.0
			-0.643492    -0.688907    1.42305    1.0
			0.675002    -0.762804    0.408387   1.0
			-2.48183      0.397482    0.588621   1.0
			-2.96432      0.81163    -0.296278   1.0
			-1.04406     -0.346355    0.691111   1.0
			-4.63685     -0.187573    0.506874   1.0
			-2.07605     -1.60726    -0.0569299  1.0
			-2.85586     -2.48079    -1.77102    1.0
			0.207866     2.27623     1.59062    1.0
			-1.38462      0.219693    1.39706    1.0]

		y = data[:,1];
		xvar = hcat(data[:,2], data[:,4]);
		_con = data[:,4];
		zvar = hcat(data[:,3], data[:,4]);

		myini = [0.5, 0.5, 0.1, log(3), log(1)]

		sfmodel_spec(sftype(prod), sfdist(trun), mu(_con),
				depvar(y), frontier(xvar), 
				sigma_u_2(_con), sigma_v_2(_con))

		sfmodel_init(all_init(myini))

		sfmodel_opt(warmstart_solver(NelderMead()),  
				warmstart_maxIT(200),
				main_solver(Newton()), 
				main_maxIT(2000), 
				tolerance(1e-8),
				verbose(false),	banner(false)
				)

		res = sfmodel_fit();

		@test res.coeff[1] ≈ 0.31051 atol=1e-5 
		@test res.coeff[2] ≈ 0.31477 atol=1e-5 
		@test res.jlms[1]  ≈ 1.41041 atol=1e-5
		@test res.bc[1]    ≈ 0.30956 atol=1e-5

		pred1 = sfmodel_predict(@eq(frontier))
		@test pred1[1] ≈  0.52564 atol=1e-5 

end


@testset verbose=true "misc" begin

	a1 = sfmodel_MixTable(5)
	@test a1[1,3]  ≈ 10.371

end