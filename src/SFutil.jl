########################################################
####                  check syntax                  ####
########################################################


#------ truncated-normal ---------

function checksyn(::Type{Trun})
    for k in (:depvar, :frontier, :μ, :σᵤ², :σᵥ²) 
        if  (_dicM[k] === nothing) 
            throw("For the truncated-normal model, the `$k` equation is missing in sfmodel_spec().")
        end 
    end


    for k in (:hscale, :timevar, :σₐ², :idvar, :gamma) 
        if  (_dicM[k] !== nothing) 
            throw("For the truncated-normal model, the `$k` equation is not needed in sfmodel_spec().")
        end 
    end
end

 #------- half-normal -------------
 
function checksyn(::Type{Half})
    for k in (:depvar, :frontier, :σᵤ², :σᵥ²) 
        if  (_dicM[k] === nothing) 
            throw("For the half-normal model, the `$k` equation is missing in sfmodel_spec().")
        end 
    end

    for k in (:μ, :hscale, :timevar, :σₐ², :idvar, :gamma) 
        if  (_dicM[k] !== nothing) 
            throw("For the half-normal model, the `$k` equation is not needed in sfmodel_spec().")
        end 
    end
end

# ------- Exponential ----------------

function checksyn(::Type{Expo})
    for k in (:depvar, :frontier, :σᵤ², :σᵥ²) 
        if  (_dicM[k] === nothing) 
            throw("For the exponential-normal model, the `$k` equation is missing in sfmodel_spec().")
        end 
    end

    for k in (:μ, :hscale, :timevar, :σₐ², :idvar, :gamma ) 
        if  (_dicM[k] !== nothing) 
            throw("For the exponential-normal model, the `$k` equation is not needed in sfmodel_spec().")
        end 
    end
end

# ------- scaling property -------------

function checksyn(::Type{Trun_Scale})
    for k in (:depvar, :frontier, :hscale, :μ, :σᵤ², :σᵥ²) 
        if  (_dicM[k] === nothing) 
            throw("For the scaling-property model, the `$k` equation is required but is currently missing in your sfmodel_spec().")
        end 
    end
    for k in ( :timevar, :idvar, :σₐ², :gamma) 
        if  (_dicM[k] !== nothing) 
            throw("For the scaling-property model, the `$k` equation is not needed in sfmodel_spec().")
        end 
    end
end

# ------- panel Wang and Ho 2020, with Truncated Dist -------------

function checksyn(::Type{PFEWHT})
    for k in (:depvar, :frontier, :hscale, :μ, :σᵤ², :σᵥ², :timevar, :idvar) 
        if  (_dicM[k] === nothing) 
            throw("For the panel fixed effect model of Wang and Ho (2010, JE) with the truncatd-normal assumption, the `$k` equation is required but is currently missing in your sfmodel_spec().")
        end 
    end
    for k in (:gamma, :σₐ²) 
        if  (_dicM[k] !== nothing) 
            throw("For the panel fixed effect model of Wang and Ho (2010, JE) with the truncated-normal assumption, the `$k` equation is not needed in sfmodel_spec().")
        end 
    end
end


# ------- panel Wang and Ho 2020, with Halfnormal Dist -------------

function checksyn(::Type{PFEWHH})
    for k in (:depvar, :frontier, :hscale, :σᵤ², :σᵥ², :timevar, :idvar) 
        if  (_dicM[k] === nothing) 
            throw("For the panel fixed effect model of Wang and Ho (2010, JE) with the half-normal assumption, the `$k` equation is required but is currently missing in your sfmodel_spec().")
        end 
    end
    for k in (:μ, :σₐ², :gamma) 
        if  (_dicM[k] !== nothing) 
            throw("For the panel fixed effect model of Wang and Ho (2010, JE) with the half-normal assumption, the `$k` equation is not needed in sfmodel_spec().")
        end 
    end
end


# ------- panel Time Decay of BC1992 -------------

function checksyn(::Type{PanDecay})
    for k in (:depvar, :frontier, :μ, :gamma, :σᵤ², :σᵥ², :idvar) 
        if  (_dicM[k] === nothing) 
            throw("For the panel time-decay model of Battese and Coelli (1992), the `$k` equation is required but is currently missing in your sfmodel_spec().")
        end 
    end
    for k in (:σₐ², :hscale, ) 
        if  (_dicM[k] !== nothing) 
            throw("For the panel time-decay model of Battese and Coelli (1992), the `$k` equation is not needed in sfmodel_spec().")
        end 
    end
end

# ------- panel Kumbhakar 1990 model  -------------

function checksyn(::Type{PanKumb90})
    for k in (:depvar, :frontier, :μ, :gamma, :σᵤ², :σᵥ², :idvar) 
        if  (_dicM[k] === nothing) 
            throw("For the panel Kumbhakar (1990) model, the `$k` equation is required but is currently missing in your sfmodel_spec().")
        end 
    end
    for k in (:σₐ², :hscale, ) 
        if  (_dicM[k] !== nothing) 
            throw("For the panel Kumbhakar (1990) model, the `$k` equation is not needed in sfmodel_spec().")
        end 
    end
end




# ------- panel CSW (2014 JoE) CSN model, with Halfnormal Dist ----------

function checksyn(::Type{PFECSWH})
    for k in (:depvar, :frontier, :σᵤ², :σᵥ², :timevar, :idvar) 
        if  (_dicM[k] === nothing) 
            throw("For the panel fixed effect model of CSW (2014, JE) with the half-normal assumption, the `$k` equation is required but is currently missing in your sfmodel_spec().")
        end 
    end
    for k in (:μ, :gamma, :σₐ², :hscale) 
        if  (_dicM[k] !== nothing) 
            throw("For the panel fixed effect model of CSW (2014, JE) with the half-normal assumption, the `$k` equation is not needed in sfmodel_spec().")
        end 
    end
end


# ------- panel TRE with Halfnormal Dist ----------

function checksyn(::Type{PTREH})
    for k in (:depvar, :frontier, :σₐ², :σᵤ², :σᵥ², :timevar, :idvar) 
        if  (_dicM[k] === nothing) 
            throw("For the panel true random effect model of Greene (2005, JoE) with the half-normal assumption, the `$k` equation is required but is currently missing in your sfmodel_spec().")
        end 
    end
    for k in (:μ, :gamma, :hscale) 
        if  (_dicM[k] !== nothing) 
            throw("For the panel true random effect model of Greene (2005, JoE) with the half-normal assumption, the `$k` equation is not needed in sfmodel_spec().")
        end 
    end
end


# ------- panel TRE with truncated normal dist ----------

function checksyn(::Type{PTRET})
    for k in (:depvar, :frontier, :μ, :σₐ², :σᵤ², :σᵥ², :timevar, :idvar) 
        if  (_dicM[k] === nothing) 
            throw("For the panel true random effect model of Greene (2005, JoE) with the truncated-normal assumption, the `$k` equation is required but is currently missing in your sfmodel_spec().")
        end 
    end
    for k in ( :gamma, :hscale) 
        if  (_dicM[k] !== nothing) 
            throw("For the panel true random effect model of Greene (2005, JoE) with the truncated-normal assumption, the `$k` equation is not needed in sfmodel_spec().")
        end 
    end
end


########################################################
####           check, pick, remove constant         ####
########################################################


macro requireConst(arg)
    return arg
end

    # ---- check overall constant --------------

function checkConst(mat::Array, S::Symbol, requireConst::Int)
    for i in 1:size(mat,2)
        aa = mat[:,i]
        # varstd = sqrt(sum((aa .- sum(aa)/length(aa)).^2)/length(aa)) ; then varstd <= 1e-8
        bb = unique(aa)
        if requireConst == 1 
            (length(bb) == 1) || throw("The variable in the $(S) function has to be a constant for this model.")
        else
            (length(bb)  > 1) || throw("The $(S) function cannot have a constant for this model.")
        end
    end
end


     # ---- check constant panel-by-panel -----------

function checkConst(mat::Array, S::Symbol, requireConst::Int, rowid::Vector)
    for i in 1:size(mat,2)
        for j in 1:size(rowid,1)
            aa = mat[rowid[j], i]
            # varstd = sqrt(sum((aa .- sum(aa)/length(aa)).^2)/length(aa)) 
            bb = unique(aa)
            if requireConst == 1 
               (length(bb) == 1) || throw("The variable in the $(S) function has to be a constant within a panel for this model.")
            else
               (length(bb)  > 1)  || throw("The $(S) function cannot have a constant within a panel for this model.")
            end
        end
    end
end


# ------- remove constant from DataFrame ---------

function nonConsDataFrame(D::DataFrame, M::Matrix)
    # Given a DataFrame containing the marginal effects 
    # of a set of exogenous determinants $(x1, x2, ..., xn)$
    # on E(u), it return the DataFrame where the marginal 
    # effect of constant $x$s are removed.

    # D: the marginal effect DataFrame; 
    # M: the matrix of (x1, .., xn) where the marginal 
    #    efect is calculated from.

   counter = 0      
   for w in collect(names(D),)
        counter += 1
        if length(unique(M[:, counter])) == 1 # is a constant
            select!(D, Not(Symbol(w)))
        end
   end 
   return D
end

# ------- pick constant name from DataFrame ---------


function pickConsNameFromDF(D::DataFrame)

    # Pick the name of constant columns from a DataFrame
    # Returns: the constant var name, the column index in the DF, and the
    #          number of the constants

   consname = []
   conspos  = []
   for w in 1:size(D,2)
        if length(unique(D[:, w])) == 1 # is a constant
            push!(consname, Symbol(names(D)[w]))
            push!(conspos,  w)
        end
    end
   return consname, conspos, length(consname)
end



# ---- combine two DataFrames ---------------- #

function addDataFrame(Main::DataFrame, A::DataFrame)
      # Combine two DataFrame with unions of columns.
      # For same-name columns, the values are added together.

    for k in collect(names(A),) # deal with the wvar
              if k ∈ names(Main)
                   Main[:, Symbol(k)] = Main[:, Symbol(k)] + A[:, Symbol(k)]
              else 
                   insertcols!(Main, Symbol(k) => A[:, Symbol(k)])
              end
     end 
     return Main  
end 



########################################################
####          Chi-equare related tables             ####
########################################################

#* --------- mixed Chi-square table ---------
"""
    sfmodel_MixTable(<keyword arguments>)

Display and return critical values of the mixed `χ²` (ch-square) distribution.
The values are taken from Table 1, Kodde and Palm (1986, Econometrica).

## Argument
- `dof::Integer`: the degree of freedom. Currently support `dof` between 1 and 40.

## Examples
```julia
julia> sfmodel_MixTable(3)

  * Significance levels and critical values of the mixed χ² distribution
┌─────┬───────┬───────┬───────┬────────┐
│ dof │  0.10 │  0.05 │ 0.025 │   0.01 │
├─────┼───────┼───────┼───────┼────────┤
│ 3.0 │ 5.528 │ 7.045 │ 8.542 │ 10.501 │
└─────┴───────┴───────┴───────┴────────┘

source: Table 1, Kodde and Palm (1986, Econometrica).
```  

"""
function sfmodel_MixTable(dof::Real=9999) # not using ::Int64 in order to thow informative error

    (dof isa(Integer)) || throw("The degree of freedom (dof) has to be an integer.")

    @isdefined(_dicOPT) || sfmodel_opt()  
    sf_table = _dicOPT[:table_format]

    table = Array{Float32}(undef, 40, 5)

    table[1 ,1] = 1;   table[1 ,2] =   1.642; table[1 ,3] =   2.705;  table[1 ,4] =   3.841;   table[1 ,5]=    5.412 ; 
    table[2 ,1] = 2;   table[2 ,2] =   3.808; table[2 ,3] =   5.138;  table[2 ,4] =   6.483;   table[2 ,5]=    8.273 ; 
    table[3 ,1] = 3;   table[3 ,2] =   5.528; table[3 ,3] =   7.045;  table[3 ,4] =   8.542;   table[3 ,5]=   10.501 ; 
    table[4 ,1] = 4;   table[4 ,2] =   7.094; table[4 ,3] =   8.761;  table[4 ,4] =  10.383;   table[4 ,5]=   12.483 ; 
    table[5 ,1] = 5;   table[5 ,2] =   8.574; table[5 ,3] =  10.371;  table[5 ,4] =   12.103;  table[5 ,5] =  14.325 ; 
    table[6 ,1] = 6;   table[6 ,2] =   9.998; table[6 ,3] =  11.911;  table[6 ,4] =   13.742;  table[6 ,5] =  16.704 ; 
    table[7 ,1] = 7;   table[7 ,2] =  11.383; table[7 ,3] =  13.401;  table[7 ,4] =   15.321;  table[7 ,5] =  17.755 ; 
    table[8 ,1] = 8;   table[8 ,2] =  12.737; table[8 ,3] =  14.853;  table[8 ,4] =   16.856;  table[8 ,5] =  19.384 ; 
    table[9 ,1] = 9;   table[9 ,2] =  14.067; table[9 ,3] =  16.274;  table[9 ,4] =   18.354;  table[9 ,5] =  20.972 ; 
    table[10,1] = 10;  table[10,2] =  15.377; table[10,3] =  17.670;  table[10,4] =   19.824;  table[10,5] =  22.525 ; 
    table[11,1] = 11;  table[11,2] =  16.670; table[11,3] =  19.045;  table[11,4] =   21.268;  table[11,5] =  24.049 ; 
    table[12,1] = 12;  table[12,2] =  17.949; table[12,3] =  20.410;  table[12,4] =   22.691;  table[12,5] =  25.549 ; 
    table[13,1] = 13;  table[13,2] =  19.216; table[13,3] =  21.742;  table[13,4] =   24.096;  table[13,5] =  27.026 ; 
    table[14,1] = 14;  table[14,2] =  20.472; table[14,3] =  23.069;  table[14,4] =   25.484;  table[14,5] =  28.485 ; 
    table[15,1] = 15;  table[15,2] =  21.718; table[15,3] =  24.384;  table[15,4] =   26.856;  table[15,5] =  29.927 ; 
    table[16,1] = 16;  table[16,2] =  22.956; table[16,3] =  25.689;  table[16,4] =   28.219;  table[16,5] =  31.353 ; 
    table[17,1] = 17;  table[17,2] =  24.186; table[17,3] =  26.983;  table[17,4] =   29.569;  table[17,5] =  32.766 ; 
    table[18,1] = 18;  table[18,2] =  25.409; table[18,3] =  28.268;  table[18,4] =   30.908;  table[18,5] =  34.167 ; 
    table[19,1] = 19;  table[19,2] =  26.625; table[19,3] =  29.545;  table[19,4] =   32.237;  table[19,5] =  35.556 ; 
    table[20,1] = 20;  table[20,2] =  27.835; table[20,3] =  30.814;  table[20,4] =   33.557;  table[20,5] =  36.935 ; 
    table[21,1] = 21;  table[21,2] =  29.040; table[21,3] =  32.077;  table[21,4] =   34.869;  table[21,5] =  38.304 ; 
    table[22,1] = 22;  table[22,2] =  30.240; table[22,3] =  33.333;  table[22,4] =   36.173;  table[22,5] =  39.664 ; 
    table[23,1] = 23;  table[23,2] =  31.436; table[23,3] =  34.583;  table[23,4] =   37.470;  table[23,5] =  41.016 ; 
    table[24,1] = 24;  table[24,2] =  32.627; table[24,3] =  35.827;  table[24,4] =   38.761;  table[24,5] =  42.360 ; 
    table[25,1] = 25;  table[25,2] =  33.813; table[25,3] =  37.066;  table[25,4] =   40.045;  table[25,5] =  43.696 ; 
    table[26,1] = 26;  table[26,2] =  34.996; table[26,3] =  38.301;  table[26,4] =   41.324;  table[26,5] =  45.026 ; 
    table[27,1] = 27;  table[27,2] =  36.176; table[27,3] =  39.531;  table[27,4] =   42.597;  table[27,5] =  46.349 ; 
    table[28,1] = 28;  table[28,2] =  37.352; table[28,3] =  40.756;  table[28,4] =   43.865;  table[28,5] =  47.667 ; 
    table[29,1] = 29;  table[29,2] =  38.524; table[29,3] =  41.977;  table[29,4] =   45.128;  table[29,5] =  48.978 ; 
    table[30,1] = 30;  table[30,2] =  39.694; table[30,3] =  43.194;  table[30,4] =   46.387;  table[30,5] =  50.284 ; 
    table[31,1] = 31;  table[31,2] =  40.861; table[31,3] =  44.408;  table[31,4] =   47.641;  table[31,5] =  51.585 ; 
    table[32,1] = 32;  table[32,2] =  42.025; table[32,3] =  45.618;  table[32,4] =   48.891;  table[32,5] =  52.881 ; 
    table[33,1] = 33;  table[33,2] =  43.186; table[33,3] =  46.825;  table[33,4] =   50.137;  table[33,5] =  54.172 ; 
    table[34,1] = 34;  table[34,2] =  44.345; table[34,3] =  48.029;  table[34,4] =   51.379;  table[34,5] =  55.459 ; 
    table[35,1] = 35;  table[35,2] =  45.501; table[35,3] =  49.229;  table[35,4] =   52.618;  table[35,5] =  56.742 ; 
    table[36,1] = 36;  table[36,2] =  46.655; table[36,3] =  50.427;  table[36,4] =   53.853;  table[36,5] =  58.020 ; 
    table[37,1] = 37;  table[37,2] =  47.808; table[37,3] =  51.622;  table[37,4] =   55.085;  table[37,5] =  59.295 ; 
    table[38,1] = 38;  table[38,2] =  48.957; table[38,3] =  52.814;  table[38,4] =   56.313;  table[38,5] =  60.566 ; 
    table[39,1] = 39;  table[39,2] =  50.105; table[39,3] =  54.003;  table[39,4] =   57.539;  table[39,5] =  61.833 ; 
    table[40,1] = 40;  table[40,2] =  51.251; table[40,3] =  55.190;  table[40,4] =   58.762;  table[40,5] =  63.097 ; 

    println()
    printstyled("  * Significance levels and critical values of the mixed χ² distribution\n"; color = :yellow)

    pretty_table(if 1<=dof<=40       # correct instance
                    table[dof,:]'
                 elseif dof == 9999  # user did not specify dof
                    printstyled("  * Use `sfmodel_MixTable(d)` to show the specific critical values for the d.o.f. equal to `d`.\n"; color = :red)
                    table            # show all
                 elseif (dof < 1) || (dof > 40)                 
                    throw("The table only allows the degree of freedom between 1 and 40.")
                 else 
                    throw("There is a problem in your specification of the degree of freedom.")   
                 end,
                 header=["dof", "0.10", "0.05", "0.025", "0.01"],
                 formatters = ft_printf("%2.3f", 2:5),
                 compact_printing = true,
                 backend = Val(sf_table))
    println()
    printstyled("source: Table 1, Kodde and Palm (1986, Econometrica)."; color = :yellow)
    println()

    if 1<=dof<=40
        return table[dof, :]'
    elseif dof == 9999
        return table
    end
end

# ---------  Chi-square table ---------


function sfmodel_ChiSquareTable(dof::Real=1) # not using ::Int64 in order to throw informative error

    (dof isa(Integer)) || throw("The degree of freedom (dof) has to be an integer.")

    table = Array{Float64}(undef, 1, 5)

    table[1, 1] = dof
    table[1, 2] = chisqinvcdf(dof, 1-0.10)
    table[1, 3] = chisqinvcdf(dof, 1-0.05)
    table[1, 4] = chisqinvcdf(dof, 1-0.025)
    table[1, 5] = chisqinvcdf(dof, 1-0.01)

    println()
    printstyled("  * significance levels and critical values of the χ² distribution\n"; color = :yellow)

    pretty_table(table,
                 ["dof", "0.10", "0.05", "0.025", "0.01"],
                 formatters = ft_printf( "%2.3f", 2:5),
                 compact_printing = true,
                 backend = Val(sf_table))

end


# ---- check multi-collinearity in the data --------

function checkCollinear(modelid, xvar, zvar, qvar, wvar, vvar)                  

   # check multicollinearity in the dataset (matrix)

    for i in 1:5

        if i == 1 
            themat = xvar
            eqname = " `frontier` " 
            eqsym  = :frontier
        elseif i == 2
            themat = zvar
            eqname = " `μ` "
            eqsym  = :μ
        elseif i == 3
            themat = qvar
            if modelid == PanDecay
                eqname = " `gamma` "
                eqsym  = :gamma
            elseif modelid == TRE
                eqname = " `σₐ²` "
                eqsym  = :σₐ²
            else
                eqname = " `hscale` "    
                eqsym  = :hscale
            end    
        elseif i == 4
            themat = wvar        
            eqname = " `σᵤ²` "   
            eqsym  =  :σᵤ²
        elseif i == 5
            themat = vvar            
            eqname = " `σᵥ²` "
            eqsym  = :σᵥ²
        end


        if length(themat) > 0 && size(themat, 2) > 1  # check only if have more than 1 var

            _, pivots = rref_with_pivots(themat)  # pivots has unique,
                                                  # non-collinear columns;
                                                  # require `using RowEchelon`
                                                
            if length(pivots) != size(themat, 2) # number of columns are different, meaning collinear is dropped
                theColl = filter(x -> x ∉ pivots, 1:size(themat, 2)) # positions of problematic columns
                bb = ""
                for i in theColl  # get problematic var's name from the equation
                    bb = bb*String(_dicM[eqsym][i])*String(", ")
                end
                printstyled(bb, "appear to have perfect collinearity with other variables in the equation", eqname, ". Try dropping this/these variable(s) from the equation.\n"; color = :red)
                throw("Multi-collinearity in the variables.")
            end  # if length(pivots)       
        end # if size(themat, 2) > 1
    end # for i in 1:5
  end



