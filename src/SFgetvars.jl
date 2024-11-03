########################################################
####       process data for estimation              ####
########################################################



function get_rowIDT(ivar) 

    # Create a matrix of panel information.
    # The `ivar` is a vector of firm id, such as `bb=[1,1,1,2,2,3,3,3,3]`.
    # The output is 
    # julia> get_rowIDT(bb)
    #    3×2 Matrix{Any}:
    #     [1, 2, 3]     3
    #     [4, 5]        2
    #     [6, 7, 8, 9]  4

    N  = length(unique(ivar)) # N=number of panels
    id = Array{Any}(undef,N,2)
    id[:,1] = unique(ivar)    # list of id with no repetition
    @inbounds for i = 1:N
        @views id[i,2]= sum(ivar.== id[i,1])    # number of periods for the panel
    end
    @views Tᵢ = id[:,2]

    rowID =  Vector{Vector}(undef, N)  
    @inbounds for i=1:N
        @views dd = findall(x-> x == id[i,1], ivar) # row index of i'th firm
        rowID[i] = dd # put the id info in the vector; faster than using UnitRange
    end    
  
    rowIDT = hcat(rowID, Tᵢ) 

    return rowIDT # (Nx2): col_1 is panel's row info; col_2 is panel's number of periods
end

#=  replaced by sf_demean()
function get_INVtrM(rowIDT) 

    # Create the within-transformation matrix for panels.
    # The matrix is panel-specific.

    N = size(rowIDT,1)
    INVtrM = Vector{Matrix{Float64}}()  # store the inverted matrix
    @inbounds for i=1:N
        @views trM = Matrix(I, rowIDT[i, 2], rowIDT[i, 2]) .- (1/rowIDT[i,2]) # transformation matrix
      # push!(INVtrM, pinv(trM)) # put the inverted matrix in the vector; note, trM is an idempotent matrix, and so the inverse is itself
        push!(INVtrM, trM) # put the inverted matrix in the vector; note, trM is an idempotent matrix, and so the inverse is itself
      end    
    return INVtrM # (Nx1): ith row is the transformation matrix for the ith panel
end
=#

function sf_demean(a::AbstractArray)  # subtract mean from columns
    return a .- mean(a, dims=1)
end

########################################################
####        process variables for estimation        ####
########################################################

#? ----- truncated normal -----------------

function getvar(::Type{Trun}, dat::DataFrame)
    
    yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
    xvar = dat[:, _dicM[:frontier]]
    zvar = dat[:, _dicM[:μ]]
    wvar = dat[:, _dicM[:σᵤ²]]
    vvar = dat[:, _dicM[:σᵥ²]]
 
    #* --- model info printout --------- 

    modelinfo1 = "normal and truncated-normal"

    if _dicM[:μ] == _dicM[:σᵤ²] && length(_dicM[:μ]) > 1
      modelinfo1 = "the non-monotonic model of Wang (2002, JPA), normal and truncated-normal"
    elseif _dicM[:μ] == _dicM[:σᵤ²] && length(_dicM[:μ]) == 1
      modelinfo1 = "the normal and truncated-normal model (Stevenson 1980, JoE)"  
    elseif length(_dicM[:μ]) > 1 && length(_dicM[:σᵤ²] ) == 1 
      modelinfo1 = "the normal and truncated-normal model of Battese and Coelli (1995, JoE)"
    else 
      modelinfo1 = "the normal and truncated-normal model" 
    end

    modelinfo2 = begin
     """
     * In the case of type(cost), "- uᵢ" below is changed to "+ uᵢ".

     $(_dicM[:depvar][1]) = frontier($(_dicM[:frontier])) + vᵢ - uᵢ,

     where vᵢ ∼ N(0, σᵥ²),
                σᵥ² = exp(log_σᵥ²) 
                    = exp($(_dicM[:σᵥ²]));
           uᵢ ∼ N⁺(μ, σᵤ²),
                μ = $(_dicM[:μ]),
                σᵤ² = exp(log_σᵤ²) 
                    = exp($(_dicM[:σᵤ²]));
     """
    end


    #* --- retrieve and generate important parameters -----

    #*  number of obs and number of variables
    nofx = nofz = nofq = nofw = nofv = 0  # to make a complete list
    nofobs  = nrow(dat)    
    nofx    = size(xvar,2)  # nofx: number of x vars
    nofz    = size(zvar,2)
    nofw    = size(wvar,2)
    nofv    = size(vvar,2)
    nofpara = nofx + nofz + nofw + nofv
    nofvar = (nofobs=nofobs, nofx=nofx, nofz=nofz, nofq=nofq,
              nofw=nofw, nofv=nofv, nofpara=nofpara, nofmarg = nofz+nofq+nofw)

    #* positions of the variables/parameters
    begx=endx=begz=endz=begq=endq=begw=endw=begv=endv=0
    begx = 1
    endx = (nofx)
    begz = (nofx) + 1
    endz = (nofx) + (nofz )
    begw = (nofx) + (nofz ) + 1
    endw = (nofx) + (nofz ) + (nofw )
    begv = (nofx) + (nofz ) + (nofw ) + 1
    endv = nofpara
    posvec = (begx=begx, endx=endx, begz=begz, endz=endz,
              begq=begq, endq=endq, begw=begw, endw=endw,
              begv=begv, endv=endv)

    #* create equation names and mark positions for making tables
    eqvec = (frontier = begx + 1, 
                    μ = begz + 1,
              log_σᵤ² = begw + 1,
              log_σᵥ² = begv + 1)

    #* create equation names and mark positions 
    eqvec2 = (coeff_frontier = (begx:endx), 
              coeff_μ        = (begz:endz),
              coeff_log_σᵤ²  = (begw:endw),
              coeff_log_σᵥ²  = (begv:endv))

    #* retrieve variable names for making tables
    xnames  = names(xvar)
    znames  = names(zvar)
    wnames  = names(wvar)
    vnames  = names(vvar)
    varlist = vcat(" ", xnames, znames, wnames, vnames)
    
    #* Converting the dataframe to matrix in order to do computation
        # Matrix(.) converts DataFrame to Matrix, but DataFrame(load(.)) would
        # have Union{Missing, Float64} type, which causes problems in
        # `ForwardDiff` in `SFmarginal.jl`.
    yvar = convert(Array{Float64}, Matrix(yvar))
    xvar = convert(Array{Float64}, Matrix(xvar))
    zvar = convert(Array{Float64}, Matrix(zvar))
    wvar = convert(Array{Float64}, Matrix(wvar))
    vvar = convert(Array{Float64}, Matrix(vvar))
    
    qvar = () 

    dum2 = ()

    return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, zvar, qvar, wvar, vvar,       dum2, varlist


  end


#? --------- half-normal ----------------

function getvar(::Type{Half}, dat::DataFrame)

    yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
    xvar = dat[:, _dicM[:frontier]]
  # zvar = dat[:, _dicM[:μ]]
    wvar = dat[:, _dicM[:σᵤ²]]
    vvar = dat[:, _dicM[:σᵥ²]]

    #* --- model info printout --------- 

    modelinfo1 = "normal and half-normal"

    modelinfo2 = begin
     """
     * In the case of type(cost), "- uᵢ" below is changed to "+ uᵢ".

     $(_dicM[:depvar][1]) = frontier($(_dicM[:frontier])) + vᵢ - uᵢ,
     
     where vᵢ ∼ N(0, σᵥ²),
                σᵥ² = exp(log_σᵥ²) 
                    = exp($(_dicM[:σᵥ²]));
           uᵢ ∼ N⁺(0, σᵤ²),
                σᵤ² = exp(log_σᵤ²) 
                    = exp($(_dicM[:σᵤ²]));
     """
    end


    #* --- retrieve and generate important parameters -----

    #*  number of obs and number of variables
    nofx = nofz = nofq = nofw = nofv = 0  # to make a complete list

    nofobs  = nrow(dat)    
    nofx    = size(xvar,2)  # nofx: number of x vars
  # nofz    = size(zvar,2)
    nofw    = size(wvar,2)
    nofv    = size(vvar,2)
    nofpara = nofx + nofz + nofw + nofv

    nofvar = (nofobs=nofobs, nofx=nofx, nofz=nofz, nofq=nofq,
              nofw=nofw, nofv=nofv, nofpara=nofpara, nofmarg = nofz+nofq+nofw)

    #* positions of the variables/parameters
    begx=endx=begz=endz=begq=endq=begw=endw=begv=endv=0

    begx = 1
    endx = (nofx)
 #  begz = (nofx) + 1
 #  endz = (nofx) + (nofz )
    begw = (nofx) + (nofz ) + 1
    endw = (nofx) + (nofz ) + (nofw )
    begv = (nofx) + (nofz ) + (nofw ) + 1
    endv = nofpara

    posvec = (begx=begx, endx=endx, begz=begz, endz=endz,
              begq=begq, endq=endq, begw=begw, endw=endw,
              begv=begv, endv=endv)

    #* create equation names and mark positions for making tables
    eqvec = (frontier = begx + 1, 
           #        μ = begz + 1,
              log_σᵤ² = begw + 1,
              log_σᵥ² = begv + 1)

    #* create equation names and mark positions 
    eqvec2 = (coeff_frontier = (begx:endx), 
              coeff_log_σᵤ²  = (begw:endw),
              coeff_log_σᵥ²  = (begv:endv))

    #* retrieve variable names for making tables
    xnames = names(xvar)
  # znames = names(zvar)
    wnames = names(wvar)
    vnames = names(vvar)
    varlist= vcat(" ", xnames, # znames, 
                   wnames, vnames)
    
    #* Converting the dataframe to matrix in order to do computation
    yvar = convert(Array{Float64}, Matrix(yvar))
    xvar = convert(Array{Float64}, Matrix(xvar))
  # zvar = convert(Array{Float64}, Matrix(zvar))
    wvar = convert(Array{Float64}, Matrix(wvar))
    vvar = convert(Array{Float64}, Matrix(vvar))

    zvar = () 
    qvar = () 
    dum2 = ()

    return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, zvar, qvar, wvar, vvar,       dum2, varlist

end


#? ---------- Exponential (same as Half, only change equation names) ----------------

function getvar(::Type{Expo}, dat::DataFrame)

    yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
    xvar = dat[:, _dicM[:frontier]]
  # zvar = dat[:, _dicM[:μ]]
    wvar = dat[:, _dicM[:σᵤ²]]
    vvar = dat[:, _dicM[:σᵥ²]]
  
    #* --- model info printout --------- 

    modelinfo1 = "normal and exponential"

    modelinfo2 = begin
     """
     * In the case of type(cost), "- uᵢ" below is changed to "+ uᵢ".

     $(_dicM[:depvar][1]) = frontier($(_dicM[:frontier])) + vᵢ - uᵢ,

     where vᵢ ∼ N(0, σᵥ²),
                σᵥ² = exp(log_σᵥ²) 
                    = exp($(_dicM[:σᵥ²]));
           uᵢ ∼ Exp(1/σᵤ) = (1/σᵤ)e^(-(1/σᵤ)U), 
                σᵤ² = exp(log_σᵤ²) 
                    = exp($(_dicM[:σᵤ²]));
     """
    end



    #* --- retrieve and generate important parameters -----
  
    #*   number of obs and number of variables
    nofx = nofz = nofq = nofw = nofv = 0  # to make a complete list
  
    nofobs  = nrow(dat)    
    nofx    = size(xvar,2)  # nofx: number of x vars
  # nofz    = size(zvar,2)
    nofw    = size(wvar,2)
    nofv    = size(vvar,2)
    nofpara = nofx + nofz + nofw + nofv
  
    nofvar = (nofobs=nofobs, nofx=nofx, nofz=nofz, nofq=nofq,
              nofw=nofw, nofv=nofv, nofpara=nofpara, nofmarg = nofz+nofq+nofw)
  
    #* positions of the variables/parameters
    begx=endx=begz=endz=begq=endq=begw=endw=begv=endv=0
  
    begx = 1
    endx = (nofx)
  # begz = (nofx) + 1
  # endz = (nofx) + (nofz )
    begw = (nofx) + (nofz ) + 1
    endw = (nofx) + (nofz ) + (nofw )
    begv = (nofx) + (nofz ) + (nofw ) + 1
    endv = nofpara
  
    posvec = (begx=begx, endx=endx, begz=begz, endz=endz,
              begq=begq, endq=endq, begw=begw, endw=endw,
              begv=begv, endv=endv)
  
    #* create equation names and mark positions for making tables
    eqvec = (frontier = begx + 1, 
           #        μ = begz + 1,
              log_σᵤ² = begw + 1,  # old: λ
              log_σᵥ² = begv + 1)
  
 
    #* create equation names and mark positions 
    eqvec2 = (coeff_frontier = (begx:endx), 
              coeff_log_σᵤ²  = (begw:endw),
              coeff_log_σᵥ²  = (begv:endv))

    #* retrieve variable names for making tables
    xnames  = names(xvar)
  # znames  = names(zvar)
    wnames  = names(wvar)
    vnames  = names(vvar)
    varlist = vcat(" ", xnames, # znames, 
                   wnames, vnames)

                   

    #* Converting the dataframe to matrix in order to do computation
  
    yvar  = convert(Array{Float64}, Matrix(yvar))
    xvar  = convert(Array{Float64}, Matrix(xvar))
  # zvar  = Matrix(zvar)
    wvar  = convert(Array{Float64}, Matrix(wvar))
    vvar  = convert(Array{Float64}, Matrix(vvar))

    zvar = () 
    qvar = () 

    dum2 = ()
 
    return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, zvar, qvar, wvar, vvar,       dum2, varlist


end


#? --------- scaling property model ----------------

function getvar(::Type{Trun_Scale}, dat::DataFrame)

    yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
    xvar = dat[:, _dicM[:frontier]]
    zvar = dat[:, _dicM[:μ]]
    qvar = dat[:, _dicM[:hscale]]
    wvar = dat[:, _dicM[:σᵤ²]]
    vvar = dat[:, _dicM[:σᵥ²]]
  
    #* --- model info printout --------- 

    modelinfo1 = "normal and truncated-normal with the scaling property (e.g., Wang and Schmidt (2002 JPA))"

    modelinfo2 = begin
     """
     * In the case of type(cost), "- uᵢ" below is changed to "+ uᵢ".

     $(_dicM[:depvar][1]) = frontier($(_dicM[:frontier])) + vᵢ - uᵢ,
     
     where vᵢ ∼ N(0, σᵥ²),
                σᵥ² = exp(log_σᵥ²) 
                    = exp($(_dicM[:σᵥ²]));
           uᵢ ∼ hscaleᵢ * N⁺(μ, σᵤ²),
                hscaleᵢ = exp($(_dicM[:hscale])),
                μ = $(_dicM[:μ]),
                σᵤ² = exp(log_σᵤ²) 
                    = exp($(_dicM[:σᵤ²]));
     """
    end


    #* --- retrieve and generate important parameters -----
  
    #*   number of obs and number of variables
    nofx = nofz = nofq = nofw = nofv = 0  # to make a complete list
  
    nofobs  = nrow(dat)    
    nofx    = size(xvar,2)  # nofx: number of x vars
    nofz    = size(zvar,2)
    nofq    = size(qvar,2)  
    nofw    = size(wvar,2)
    nofv    = size(vvar,2)
    nofpara = nofx + nofz + nofq + nofw + nofv
  
    nofvar = (nofobs=nofobs, nofx=nofx, nofz=nofz, nofq=nofq,
              nofw=nofw, nofv=nofv, nofpara=nofpara, nofmarg = nofz+nofq+nofw)
  
    #* positions of the variables/parameters
    begx=endx=begz=endz=begq=endq=begw=endw=begv=endv=0
  
    begx = 1
    endx = nofx
    begz = endx + 1
    endz = begz + nofz-1
    begq = endz + 1
    endq = begq + nofq-1
    begw = endq + 1
    endw = begw + nofw-1
    begv = endw + 1
    endv = nofpara
  
    posvec = (begx=begx, endx=endx, begz=begz, endz=endz,
              begq=begq, endq=endq, begw=begw, endw=endw,
              begv=begv, endv=endv)
  
    #* create equation names and mark positions for making tables
    eqvec = (frontier = begx + 1, 
                    μ = begz + 1,
           log_hscale = begq + 1,
              log_σᵤ² = begw + 1,
              log_σᵥ² = begv + 1)

    #* create equation names and mark positions 
    eqvec2 = (coeff_frontier = (begx:endx), 
                     coeff_μ = (begz:endz),
            coeff_log_hscale = (begq:endq),
               coeff_log_σᵤ² = (begw:endw),
               coeff_log_σᵥ² = (begv:endv))

    #* retrieve variable names for making tables
    xnames  = names(xvar)
    znames  = names(zvar)
    qnames  = names(qvar)
    wnames  = names(wvar)
    vnames  = names(vvar)
    varlist = vcat(" ", xnames, znames, qnames, wnames, vnames)
   
    #* Converting the dataframe to matrix in order to do computation
    yvar  = convert(Array{Float64}, Matrix(yvar))
    xvar  = convert(Array{Float64}, Matrix(xvar))
    zvar  = convert(Array{Float64}, Matrix(zvar))
    qvar  = convert(Array{Float64}, Matrix(qvar))
    wvar  = convert(Array{Float64}, Matrix(wvar))
    vvar  = convert(Array{Float64}, Matrix(vvar))
 
    #* various functions can and cannot contain a constant, check! ---- *#
    checkConst(zvar, :μ,      @requireConst(1))
    checkConst(qvar, :hscale, @requireConst(0))
    checkConst(wvar, :σᵤ²,    @requireConst(1))

    dum2 = ()
 
    return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, zvar, qvar, wvar, vvar,       dum2, varlist

end


#? --------- panel FE Wang and Ho, half-normal ----------------

function getvar(::Type{PFEWHH}, dat::DataFrame)

  yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
  xvar = dat[:, _dicM[:frontier]]
# zvar = dat[:, _dicM[:μ]]
  qvar = dat[:, _dicM[:hscale]]
  wvar = dat[:, _dicM[:σᵤ²]]
  vvar = dat[:, _dicM[:σᵥ²]]

  tvar = dat[:, _dicM[:timevar]]
  ivar = dat[:, _dicM[:idvar]] # need the [1] otherwise causes problems down the road

    #* --- model info printout --------- 

    modelinfo1 = "true fixed effect of Wang and Ho (2010 JE), normal and half-normal"

    modelinfo2 = begin
     """
     * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

     $(_dicM[:depvar][1]) = frontier(αᵢ + $(_dicM[:frontier])) + vᵢₜ - uᵢₜ,
     
     where vᵢₜ ∼ N(0, σᵥ²),
                 σᵥ² = exp(log_σᵥ²) 
                     = exp($(_dicM[:σᵥ²]));
           uᵢₜ ∼ hscaleᵢₜ * uᵢ,
                 hscaleᵢₜ = exp($(_dicM[:hscale])),
           uᵢ ∼ N⁺(0, σᵤ²),
                σᵤ² = exp(log_σᵤ²) 
                    = exp($(_dicM[:σᵤ²]));
     """
    end


  #* --- retrieve and generate important parameters -----

  #*   number of obs and number of variables
  nofx = nofz = nofq = nofw = nofv = 0  # to make a complete list

  nofobs  = nrow(dat)    
  nofx    = size(xvar,2)  # nofx: number of x vars
# nofz    = size(zvar,2)
  nofq    = size(qvar,2)  
  nofw    = size(wvar,2)
  nofv    = size(vvar,2)
  nofpara = nofx + nofz + nofq + nofw + nofv

  nofvar = (nofobs=nofobs, nofx=nofx, nofz=nofz, nofq=nofq,
            nofw=nofw, nofv=nofv, nofpara=nofpara, nofmarg = nofz+nofq+nofw)

  #* positions of the variables/parameters
  begx=endx=begz=endz=begq=endq=begw=endw=begv=endv=0

  begx = 1
  endx = nofx
# begz = endx + 1
# endz = begz + nofz -1
  begq = endx + 1
  endq = begq + nofq -1
  begw = endq + 1
  endw = begw + nofw -1
  begv = endw + 1
  endv = nofpara

  posvec = (begx=begx, endx=endx, begz=begz, endz=endz,
            begq=begq, endq=endq, begw=begw, endw=endw,
            begv=begv, endv=endv)

  #* create equation names and mark positions for making tables
  eqvec = (frontier = begx + 1, 
           #      μ = begz + 1,
         log_hscale = begq + 1,
            log_σᵤ² = begw + 1,
            log_σᵥ² = begv + 1)

  #* create equation names and mark positions 
  eqvec2 = (coeff_frontier = (begx:endx), 
          coeff_log_hscale = (begq:endq),
             coeff_log_σᵤ² = (begw:endw),
             coeff_log_σᵥ² = (begv:endv))             

  #* retrieve variable names for making tables
  xnames  = names(xvar)
# znames  = names(zvar)
  qnames  = names(qvar)
  wnames  = names(wvar)
  vnames  = names(vvar)
  varlist = vcat(" ", xnames, # znames, 
                      qnames, wnames, vnames)
 
  #* Converting the dataframe to matrix in order to do computation
  yvar = convert(Array{Float64}, Matrix(yvar))
  xvar = convert(Array{Float64}, Matrix(xvar))
# zvar = convert(Array{Float64}, Matrix(zvar))
  qvar = convert(Array{Float64}, Matrix(qvar))
  wvar = convert(Array{Float64}, Matrix(wvar))
  vvar = convert(Array{Float64}, Matrix(vvar))

  tvar = convert(Array{Float64}, Matrix(tvar))
  ivar = convert(Array{Float64}, Matrix(ivar))

  zvar = ()

  #* various functions can and cannot contain a constant, check! ---- *#
  checkConst(xvar, :frontier, @requireConst(0))
  checkConst(qvar, :hscale,   @requireConst(0))
# checkConst(zvar, :μ,        @requireConst(1))
  checkConst(wvar, :σᵤ²,      @requireConst(1))
  checkConst(vvar, :σᵥ²,      @requireConst(1))
  
  #* panel info and within transformation

  rowIDT = get_rowIDT(vec(ivar))   # (Nx2): col_1 is panel's row info; col_2 is panel's number of periods
 
  yxdata = hcat(yvar, xvar) 
       D = zeros(nofobs, 1+nofx)  # pre-allocate the transformed dataset
       N = length(unique(vec(ivar)))   # N=number of panels

  for i=1:N  # within-transform the data
      @views D[rowIDT[i,1], :] = sf_demean(yxdata[rowIDT[i,1], :]) # INVtrM[i] * yxdata[rowIDT[i,1], :] # transform the data
  end    
  
  ỹ = (D[:, 1]) 
  x̃ = (D[:, 2:end]) 

  return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, ỹ, x̃, zvar, qvar, wvar, vvar,         rowIDT, varlist

end


#?--------- panel FE Wang and Ho, truncated normal ----------------

function getvar(::Type{PFEWHT}, dat::DataFrame)

  yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
  xvar = dat[:, _dicM[:frontier]]
  zvar = dat[:, _dicM[:μ]]
  qvar = dat[:, _dicM[:hscale]]  
  wvar = dat[:, _dicM[:σᵤ²]]
  vvar = dat[:, _dicM[:σᵥ²]]

  tvar = dat[:, _dicM[:timevar]]
  ivar = dat[:, _dicM[:idvar]] 

  #* --- model info printout --------- 
  modelinfo1 = "true fixed effect model of Wang and Ho (2010 JE), normal and truncated-normal"
  modelinfo2 = begin
   """
   * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

   $(_dicM[:depvar][1]) = frontier(αᵢ + $(_dicM[:frontier])) + vᵢₜ - uᵢₜ,
   
   where vᵢₜ ∼ N(0, σᵥ²),
               σᵥ² = exp(log_σᵥ²) 
                   = exp($(_dicM[:σᵥ²]));
         uᵢₜ ∼ hscaleᵢₜ * uᵢ,
               hscaleᵢₜ = exp($(_dicM[:hscale])),
         uᵢ ∼ N⁺(μ, σᵤ²),
              μ = $(_dicM[:μ])
              σᵤ² = exp(log_σᵤ²) 
                  = exp($(_dicM[:σᵤ²]));
   """
  end

  #* --- retrieve and generate important parameters -----

  #*   number of obs and number of variables
  nofx = nofz = nofq = nofw = nofv = 0  # to make a complete list

  nofobs  = nrow(dat)    
  nofx    = size(xvar,2)  # nofx: number of x vars
  nofz    = size(zvar,2)
  nofq    = size(qvar,2)  
  nofw    = size(wvar,2)
  nofv    = size(vvar,2)
  nofpara = nofx + nofz + nofq + nofw + nofv

  nofvar = (nofobs=nofobs, nofx=nofx, nofz=nofz, nofq=nofq,
            nofw=nofw, nofv=nofv, nofpara=nofpara, nofmarg = nofz+nofq+nofw)

  #* positions of the variables/parameters
  begx=endx=begz=endz=begq=endq=begw=endw=begv=endv=0

  begx = 1
  endx = nofx
  begz = endx + 1
  endz = begz + nofz-1
  begq = endz + 1
  endq = begq + nofq-1
  begw = endq + 1
  endw = begw + nofw-1
  begv = endw + 1
  endv = nofpara

  posvec = (begx=begx, endx=endx, begz=begz, endz=endz,
            begq=begq, endq=endq, begw=begw, endw=endw,
            begv=begv, endv=endv)

  #* create equation names and mark positions for making tables
  eqvec = (frontier = begx + 1, 
                  μ = begz + 1,
         log_hscale = begq + 1,
            log_σᵤ² = begw + 1,
            log_σᵥ² = begv + 1)

  #* create equation names and mark positions 
  eqvec2 = (coeff_frontier = (begx:endx), 
                   coeff_μ = (begz:endz),
          coeff_log_hscale = (begq:endq),
             coeff_log_σᵤ² = (begw:endw),
             coeff_log_σᵥ² = (begv:endv))             

  #* retrieve variable names for making tables
  xnames  = names(xvar)
  znames  = names(zvar)
  qnames  = names(qvar)
  wnames  = names(wvar)
  vnames  = names(vvar)
  varlist = vcat(" ", xnames, znames, qnames, wnames, vnames)
 
  #* Converting the dataframe to matrix in order to do computation
  yvar  = convert(Array{Float64}, Matrix(yvar))
  xvar  = convert(Array{Float64}, Matrix(xvar))
  zvar  = convert(Array{Float64}, Matrix(zvar))
  qvar  = convert(Array{Float64}, Matrix(qvar))
  wvar  = convert(Array{Float64}, Matrix(wvar))
  vvar  = convert(Array{Float64}, Matrix(vvar))

  tvar  = convert(Array{Float64}, Matrix(tvar))
  ivar  = convert(Array{Float64}, Matrix(ivar))



  #* various functions can and cannot contain a constant, check! ---- *#
  checkConst(xvar, :frontier, @requireConst(0))
  checkConst(zvar, :μ,        @requireConst(1))
  checkConst(qvar, :hscale,   @requireConst(0)) 
  checkConst(wvar, :σᵤ²,      @requireConst(1))
  checkConst(vvar, :σᵥ²,      @requireConst(1))

  #* Within transformation


  rowIDT = get_rowIDT(vec(ivar))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of periods
 
  yxdata = hcat(yvar, xvar) 
       D = zeros(nofobs, 1+nofx)      # pre-allocate the transformed dataset
       N = length(unique(vec(ivar))) # N=number of panels


  for i=1:N
      @views D[rowIDT[i,1], :] = sf_demean(yxdata[rowIDT[i,1], :]) # INVtrM[i] * yxdata[rowIDT[i,1], :] # transform the data
  end    
  
  ỹ = (D[:, 1]) 
  x̃ = (D[:, 2:end]) 

  return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, ỹ, x̃, zvar, qvar, wvar, vvar,         rowIDT, varlist


end


#?--------- time decay model   ----------------

# function getvar(::Type{PanDecay}, dat::DataFrame)

function getvar(::Union{Type{PanDecay}, Type{PanKumb90}}, dat::DataFrame)

  yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
  xvar = dat[:, _dicM[:frontier]]
  zvar = dat[:, _dicM[:μ]]
  qvar = dat[:, _dicM[:gamma]] 
  wvar = dat[:, _dicM[:σᵤ²]]
  vvar = dat[:, _dicM[:σᵥ²]]

  ivar = dat[:, _dicM[:idvar]] # need the [1] otherwise causes problems down the road

  #* --- model info printout --------- 


 if _dicM[:panel][1]  == :TimeDecay


      modelinfo1 = "panel time-decay model of Battese and Coelli (1992) (see also Wang and Kumbhakar 2005)"
      modelinfo2 = begin
      """
      * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

      $(_dicM[:depvar][1]) = frontier($(_dicM[:frontier])) + vᵢₜ - uᵢₜ,
      
      where vᵢₜ ∼ N(0, σᵥ²),
                  σᵥ² = exp(log_σᵥ²) 
                      = exp($(_dicM[:σᵥ²]));
            uᵢₜ ∼ gammaₜ * uᵢ,
                  gammaₜ = exp($(_dicM[:gamma])), 
            uᵢ ∼ N⁺(μ, σᵤ²),
                  μ = $(_dicM[:μ])
                  σᵤ² = exp(log_σᵤ²) 
                      = exp($(_dicM[:σᵤ²]));
      """
      end

  end


  if _dicM[:panel][1]  == :Kumbhakar1990

    modelinfo1 = "panel Kumbhakar (1990) model"
    modelinfo2 = begin
    """
    * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

    $(_dicM[:depvar][1]) = frontier($(_dicM[:frontier])) + vᵢₜ - uᵢₜ,
    
    where vᵢₜ ∼ N(0, σᵥ²),
                σᵥ² = exp(log_σᵥ²) 
                    = exp($(_dicM[:σᵥ²]));
          uᵢₜ ∼ gammaₜ * uᵢ,
                gammaₜ = 2/(1 + exp($(_dicM[:gamma]))), 
          uᵢ ∼ N⁺(μ, σᵤ²),
                μ = $(_dicM[:μ])
                σᵤ² = exp(log_σᵤ²) 
                    = exp($(_dicM[:σᵤ²]));
    """
    end

  end

  




  #* --- retrieve and generate important parameters -----

  #*   number of obs and number of variables
  nofx = nofz = nofq = nofw = nofv = 0  # to make a complete list

  nofobs  = nrow(dat)    
  nofx    = size(xvar,2)  # nofx: number of x vars
  nofz    = size(zvar,2)
  nofq    = size(qvar,2)  
  nofw    = size(wvar,2)
  nofv    = size(vvar,2)
  nofpara = nofx + nofz + nofq + nofw + nofv

  nofvar = (nofobs=nofobs, nofx=nofx, nofz=nofz, nofq=nofq,
            nofw=nofw, nofv=nofv, nofpara=nofpara, nofmarg = nofz+nofq+nofw)

  #* positions of the variables/parameters
  begx=endx=begz=endz=begq=endq=begw=endw=begv=endv=0

  begx = 1
  endx = nofx
  begz = endx + 1
  endz = begz + nofz-1
  begq = endz + 1
  endq = begq + nofq-1
  begw = endq + 1
  endw = begw + nofw-1
  begv = endw + 1
  endv = nofpara

  posvec = (begx=begx, endx=endx, begz=begz, endz=endz,
            begq=begq, endq=endq, begw=begw, endw=endw,
            begv=begv, endv=endv)

  #* create equation names and mark positions for making tables
  eqvec = (frontier = begx + 1, 
                  μ = begz + 1,
          log_gamma = begq + 1,
            log_σᵤ² = begw + 1,
            log_σᵥ² = begv + 1)

  #* create equation names and mark positions 
  eqvec2 = (coeff_frontier = (begx:endx), 
                   coeff_μ = (begz:endz),
           coeff_log_gamma = (begq:endq),
             coeff_log_σᵤ² = (begw:endw),
             coeff_log_σᵥ² = (begv:endv))             

  #* retrieve variable names for making tables
  xnames  = names(xvar)
  znames  = names(zvar)
  qnames  = names(qvar)
  wnames  = names(wvar)
  vnames  = names(vvar)
  varlist = vcat(" ", xnames, znames, qnames, wnames, vnames)
 
  #* Converting the dataframe to matrix in order to do computation
  yvar  = convert(Array{Float64}, Matrix(yvar))
  xvar  = convert(Array{Float64}, Matrix(xvar))
  zvar  = convert(Array{Float64}, Matrix(zvar))
  qvar  = convert(Array{Float64}, Matrix(qvar))
  wvar  = convert(Array{Float64}, Matrix(wvar))
  vvar  = convert(Array{Float64}, Matrix(vvar))
  ivar  = convert(Array{Float64}, Matrix(ivar))

  #* get panel related variables
  rowIDT = get_rowIDT(vec(ivar))   # (Nx2): col_1 is panel's row info; col_2 is panel's number of periods

  #* various functions should and/or should not contain a constant, check! ---- *#
  checkConst(zvar, :μ,      @requireConst(1), rowIDT[:,1])
  checkConst(qvar, :gamma,  @requireConst(0))
  checkConst(wvar, :σᵤ²,    @requireConst(1))
  checkConst(vvar, :σᵥ²,    @requireConst(1))

  return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2,  yvar, xvar, zvar, qvar, wvar, vvar,       rowIDT, varlist


end


#? --------- panel FE CSW 2014 JoE (CSN), half-normal ----------------

function getvar(::Type{PFECSWH}, dat::DataFrame)

  yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
  xvar = dat[:, _dicM[:frontier]]
# zvar = dat[:, _dicM[:μ]]
# qvar = dat[:, _dicM[:hscale]]
  wvar = dat[:, _dicM[:σᵤ²]]
  vvar = dat[:, _dicM[:σᵥ²]]

  tvar = dat[:, _dicM[:timevar]]
  ivar = dat[:, _dicM[:idvar]]
    #* --- model info printout --------- 

    modelinfo1 = "true fixed effect of Chen, Schmidt, and Wang (2014 JE), normal and half-normal"

    modelinfo2 = begin
     """
     * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

     $(_dicM[:depvar][1]) = frontier(αᵢ + $(_dicM[:frontier])) + vᵢₜ - uᵢₜ,
     
     where vᵢₜ ∼ N(0, σᵥ²),
                 σᵥ² = exp(log_σᵥ²) 
                     = exp($(_dicM[:σᵥ²]));
           uᵢₜ ∼ N⁺(0, σᵤ²),
                σᵤ² = exp(log_σᵤ²) 
                    = exp($(_dicM[:σᵤ²]));
     """
    end


  #* --- retrieve and generate important parameters -----

  #*   number of obs and number of variables
  nofx = nofz = nofq = nofw = nofv = 0  # to make a complete list

  nofobs  = nrow(dat)    
  nofx    = size(xvar,2)  # nofx: number of x vars
# nofz    = size(zvar,2)
# nofq    = size(qvar,2)  
  nofw    = size(wvar,2)
  nofv    = size(vvar,2)
  nofpara = nofx + nofz + nofq + nofw + nofv

  nofvar = (nofobs=nofobs, nofx=nofx, nofz=nofz, nofq=nofq,
            nofw=nofw, nofv=nofv, nofpara=nofpara, nofmarg = nofz+nofq+nofw)

  #* positions of the variables/parameters
  begx=endx=begz=endz=begq=endq=begw=endw=begv=endv=0

  begx = 1
  endx = nofx
# begz = endx + 1
# endz = begz + nofz -1
# begq = endx + 1
# endq = begq + nofq -1
  begw = endx + 1
  endw = begw + nofw -1
  begv = endw + 1
  endv = nofpara

  posvec = (begx=begx, endx=endx, begz=begz, endz=endz,
            begq=begq, endq=endq, begw=begw, endw=endw,
            begv=begv, endv=endv)

  #* create equation names and mark positions for making tables
  eqvec = (frontier = begx + 1, 
           #      μ = begz + 1,
       # log_hscale = begq + 1,
            log_σᵤ² = begw + 1,
            log_σᵥ² = begv + 1)

  #* create equation names and mark positions 
  eqvec2 = (coeff_frontier = (begx:endx), 
             coeff_log_σᵤ² = (begw:endw),
             coeff_log_σᵥ² = (begv:endv))             

  #* retrieve variable names for making tables
  xnames  = names(xvar)
# znames  = names(zvar)
# qnames  = names(qvar)
  wnames  = names(wvar)
  vnames  = names(vvar)
  varlist = vcat(" ", xnames, # znames, 
                      # qnames, 
                      wnames, vnames)
 
  #* Converting the dataframe to matrix in order to do computation
  yvar = convert(Array{Float64}, Matrix(yvar))
  xvar = convert(Array{Float64}, Matrix(xvar))
# zvar = convert(Array{Float64}, Matrix(zvar))
# qvar = convert(Array{Float64}, Matrix(qvar))
  wvar = convert(Array{Float64}, Matrix(wvar))
  vvar = convert(Array{Float64}, Matrix(vvar))

  tvar = convert(Array{Float64}, Matrix(tvar))
  ivar = convert(Array{Float64}, Matrix(ivar))

  zvar = ()
  qvar = ()

  #* various functions can and cannot contain a constant, check! ---- *#
  checkConst(xvar, :frontier, @requireConst(0))
# checkConst(qvar, :hscale,   @requireConst(0))
# checkConst(zvar, :μ,        @requireConst(1))
  checkConst(wvar, :σᵤ²,      @requireConst(1))
  checkConst(vvar, :σᵥ²,      @requireConst(1))


  #* panel info and within transformation

  rowIDT = get_rowIDT(vec(ivar))   # (Nx2): col_1 is panel's row info; col_2 is panel's number of periods

  yxdata = hcat(yvar, xvar) 
       D = zeros(nofobs, 1+nofx)  # pre-allocate the transformed dataset
       N = length(unique(vec(ivar)))   # N=number of panels

  for i=1:N  # within-transform the data
      @views D[rowIDT[i,1], :] = sf_demean(yxdata[rowIDT[i,1], :]) # INVtrM[i] * yxdata[rowIDT[i,1], :] # transform the data
  end    
  
  ỹ = (D[:, 1]) 
  x̃ = (D[:, 2:end]) 

  return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, ỹ, x̃, zvar, qvar, wvar, vvar,         rowIDT, varlist

end



#? --------- panel TRE (true random effect), half-normal ----------------

function getvar(::Type{PTREH}, dat::DataFrame)

  yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
  xvar = dat[:, _dicM[:frontier]]
# zvar = dat[:, _dicM[:μ]]
  qvar = dat[:, _dicM[:σₐ²]]
  wvar = dat[:, _dicM[:σᵤ²]]
  vvar = dat[:, _dicM[:σᵥ²]]

  tvar = dat[:, _dicM[:timevar]]
  ivar = dat[:, _dicM[:idvar]] # need the [1] otherwise causes problems down the road



    #* --- model info printout --------- 

    modelinfo1 = "true random effect model of Greene (2005 JoE), normal and half-normal"

    modelinfo2 = begin
     """
     * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

     $(_dicM[:depvar][1]) = frontier(αᵢ + $(_dicM[:frontier])) + vᵢₜ - uᵢₜ,
     
     where 
            αᵢ ∼ N(0, σₐ²)
                 σₐ² = exp(log_σₐ²) 
                     = exp($(_dicM[:σₐ²]));
           vᵢₜ ∼ N(0, σᵥ²),
                 σᵥ² = exp(log_σᵥ²) 
                     = exp($(_dicM[:σᵥ²]));
           uᵢₜ ∼ N⁺(0, σᵤ²),
                σᵤ² = exp(log_σᵤ²) 
                    = exp($(_dicM[:σᵤ²]));
     """
    end


  #* --- retrieve and generate important parameters -----

  #*   number of obs and number of variables
  nofx = nofz = nofq = nofw = nofv = 0  # to make a complete list

  nofobs  = nrow(dat)    
  nofx    = size(xvar,2)  # nofx: number of x vars
# nofz    = size(zvar,2)
  nofq    = size(qvar,2)  
  nofw    = size(wvar,2)
  nofv    = size(vvar,2)
  nofpara = nofx + nofz + nofq + nofw + nofv

  nofvar = (nofobs=nofobs, nofx=nofx, nofz=nofz, nofq=nofq,
            nofw=nofw, nofv=nofv, nofpara=nofpara, nofmarg = nofz+nofq+nofw)

  #* positions of the variables/parameters
  begx=endx=begz=endz=begq=endq=begw=endw=begv=endv=0

  begx = 1
  endx = nofx
# begz = endx + 1
# endz = begz + nofz -1
  begq = endx + 1
  endq = begq + nofq -1
  begw = endq + 1
  endw = begw + nofw -1
  begv = endw + 1
  endv = nofpara

  posvec = (begx=begx, endx=endx, begz=begz, endz=endz,
            begq=begq, endq=endq, begw=begw, endw=endw,
            begv=begv, endv=endv)

  #* create equation names and mark positions for making tables
  eqvec = (frontier = begx + 1, 
           #      μ = begz + 1,
            log_σₐ² = begq + 1,
            log_σᵤ² = begw + 1,
            log_σᵥ² = begv + 1)

  #* create equation names and mark positions 
  eqvec2 = (coeff_frontier = (begx:endx), 
             coeff_log_σₐ² = (begq:endq),
             coeff_log_σᵤ² = (begw:endw),
             coeff_log_σᵥ² = (begv:endv))             

  #* retrieve variable names for making tables
  xnames  = names(xvar)
# znames  = names(zvar)
  qnames  = names(qvar)
  wnames  = names(wvar)
  vnames  = names(vvar)
  varlist = vcat(" ", xnames, # znames, 
                      qnames, 
                      wnames, vnames)
 
  #* Converting the dataframe to matrix in order to do computation

  yvar = convert(Array{Float64}, Matrix(yvar))
  xvar = convert(Array{Float64}, Matrix(xvar))
  # zvar = convert(Array{Float64}, Matrix(zvar))
  qvar = convert(Array{Float64}, Matrix(qvar))
  wvar = convert(Array{Float64}, Matrix(wvar))
  vvar = convert(Array{Float64}, Matrix(vvar))

  tvar = convert(Array{Float64}, Matrix(tvar))
  ivar = convert(Array{Float64}, Matrix(ivar))

  zvar = ()

  #* various functions can and cannot contain a constant, check! ---- *#
# checkConst(xvar, :frontier, @requireConst(0))
# checkConst(qvar, :hscale,   @requireConst(0))
  checkConst(qvar, :σₐ²,      @requireConst(1))
  checkConst(wvar, :σᵤ²,      @requireConst(1))
  checkConst(vvar, :σᵥ²,      @requireConst(1))


  #* panel info and within transformation

  rowIDT = get_rowIDT(vec(ivar))   # (Nx2): col_1 is panel's row info; col_2 is panel's number of periods

  return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, zvar, qvar, wvar, vvar,    rowIDT, varlist


end

#? --------- panel TRE (true random effect), truncated-normal ----------------

function getvar(::Type{PTRET}, dat::DataFrame)

  yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
  xvar = dat[:, _dicM[:frontier]]
  zvar = dat[:, _dicM[:μ]]
  qvar = dat[:, _dicM[:σₐ²]]
  wvar = dat[:, _dicM[:σᵤ²]]
  vvar = dat[:, _dicM[:σᵥ²]]

  tvar = dat[:, _dicM[:timevar]]
  ivar = dat[:, _dicM[:idvar]] # need the [1] otherwise causes problems down the road

    #* --- model info printout --------- 

    modelinfo1 = "true random effect model of Greene (2005 JoE), normal and truncated-normal"

    modelinfo2 = begin
     """
     * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

     $(_dicM[:depvar][1]) = frontier(αᵢ + $(_dicM[:frontier])) + vᵢₜ - uᵢₜ,
     
     where 
            αᵢ ∼ N(0, σₐ²)
                 σₐ² = exp(log_σₐ²) 
                     = exp($(_dicM[:σₐ²]));
           vᵢₜ ∼ N(0, σᵥ²),
                 σᵥ² = exp(log_σᵥ²) 
                     = exp($(_dicM[:σᵥ²]));
           uᵢₜ ∼ N⁺(μ, σᵤ²),
                   μ = $(_dicM[:μ]),
                 σᵤ² = exp(log_σᵤ²) 
                     = exp($(_dicM[:σᵤ²]));
     """
    end


  #* --- retrieve and generate important parameters -----

  #*   number of obs and number of variables
  nofx = nofz = nofq = nofw = nofv = 0  # to make a complete list

  nofobs  = nrow(dat)    
  nofx    = size(xvar,2)  # nofx: number of x vars
  nofz    = size(zvar,2)
  nofq    = size(qvar,2)  
  nofw    = size(wvar,2)
  nofv    = size(vvar,2)
  nofpara = nofx + nofz + nofq + nofw + nofv

  nofvar = (nofobs=nofobs, nofx=nofx, nofz=nofz, nofq=nofq,
            nofw=nofw, nofv=nofv, nofpara=nofpara, nofmarg = nofz+nofq+nofw)

  #* positions of the variables/parameters
  begx=endx=begz=endz=begq=endq=begw=endw=begv=endv=0

  begx = 1
  endx = nofx
  begz = endx + 1
  endz = begz + nofz -1
  begq = endz + 1
  endq = begq + nofq -1
  begw = endq + 1
  endw = begw + nofw -1
  begv = endw + 1
  endv = nofpara

  posvec = (begx=begx, endx=endx, begz=begz, endz=endz,
            begq=begq, endq=endq, begw=begw, endw=endw,
            begv=begv, endv=endv)

  #* create equation names and mark positions for making tables
  eqvec = (frontier = begx + 1, 
                  μ = begz + 1,
            log_σₐ² = begq + 1,
            log_σᵤ² = begw + 1,
            log_σᵥ² = begv + 1)

  #* create equation names and mark positions 
  eqvec2 = (coeff_frontier = (begx:endx), 
             coeff_μ       = (begz:endz),
             coeff_log_σₐ² = (begq:endq),
             coeff_log_σᵤ² = (begw:endw),
             coeff_log_σᵥ² = (begv:endv))             

  #* retrieve variable names for making tables
  xnames  = names(xvar)
  znames  = names(zvar)
  qnames  = names(qvar)
  wnames  = names(wvar)
  vnames  = names(vvar)
  varlist = vcat(" ", xnames,  znames, 
                      qnames, 
                      wnames, vnames)
 
  #* Converting the dataframe to matrix in order to do computation
  yvar = convert(Array{Float64}, Matrix(yvar))
  xvar = convert(Array{Float64}, Matrix(xvar))
  zvar = convert(Array{Float64}, Matrix(zvar))
  qvar = convert(Array{Float64}, Matrix(qvar))
  wvar = convert(Array{Float64}, Matrix(wvar))
  vvar = convert(Array{Float64}, Matrix(vvar))

  tvar = convert(Array{Float64}, Matrix(tvar))
  ivar = convert(Array{Float64}, Matrix(ivar))

  #* various functions can and cannot contain a constant, check! ---- *#
# checkConst(xvar, :frontier, @requireConst(0))
# checkConst(qvar, :hscale,   @requireConst(0))
  checkConst(qvar, :σₐ²,      @requireConst(1))
  checkConst(wvar, :σᵤ²,      @requireConst(1))
  checkConst(vvar, :σᵥ²,      @requireConst(1))


  #* panel info and within transformation

  rowIDT = get_rowIDT(vec(ivar))   # (Nx2): col_1 is panel's row info; col_2 is panel's number of periods

  return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, zvar, qvar, wvar, vvar,         rowIDT, varlist


end
