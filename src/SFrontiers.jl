module SFrontiers

export sfmodel_spec, sfmodel_init, sfmodel_opt, 
       sfmodel_fit, sfmodel_predict, drawTrun,
       genTrun, sf_demean, sfmodel_boot_marginal,
       # likelihood functions 
        LL_T, 
       # macros for _spec; 
        depvar, timevar, idvar,
        sfdist, sftype,
        @sfdist, @sftype,
        @model, @depvar, 
        @frontier,
        @μ, @mu,  @σᵤ², @sigma_u_2,  @σᵥ², @sigma_v_2, 
        @σₐ², @sigma_a_2, 
        @hscale,  
        @sfpanel, 
        sfpanel,
        @timevar, @idvar,
        @gamma,
        @eq,
       # functions for sfmodel_init 
        frontier, 
        μ, mu, σᵤ², sigma_u_2, σᵥ², sigma_v_2, 
        σₐ², sigma_a_2, 
        misc, 
        hscale,  gamma,
        all_init,
       # functions for sfmodel_opt
        warmstart_solver, warmstart_maxIT,
        main_solver, main_maxIT, tolerance, verbose, banner,
        ineff_index, marginal, table_format,
       # functions for sfmodel_fit
        useData,
        sfmodel_CI,
       # MoM Test
        sfmodel_MoMTest,
       # functions for JLMS and BC index
        jlmsbc, jlmsbc_marg,
       # the table for regular and mixed Chi-square test
        sfmodel_MixTable, sfmodel_ChiSquareTable,
       # struct 
        Sfmodeltype,
        Trun, trun, truncated, t, 
        Half, half, h,
        Expo, expo, e,
        Trun_Scale, trun_scale, trun_scaling, s,
        Half_Scale, half_scale, half_scaling,
        MoM,
        production, cost, #* not prod, b/c conflict with Base
        text, html, latex,
        PFEWHH, PFEWHT, PTREH,  PTRET,  PFECSWH, PanDecay, get_marg, PanKumb90,  #* export for testing purpose
        TFE_WH2010, TFE_CSW2014, TFE_G2005, TRE, TimeDecay, Kumbhakar1990,
      # Optim's algorithms  
        NelderMead, SimulatedAnnealing, SAMIN, ParticleSwarm,
        ConjugateGradient, GradientDescent, BFGS, LBFGS,
        Newton, NewtonTrustRegion, IPNewton, 
      # sfmodel_mtest()
        pickConsNameFromDF
 



using DataFrames
using DataStructures             # for OrderedDict
using Distributions              # for TDist, Normal
using FLoops                     # multithreading
using ForwardDiff                # for marginal effect
using HypothesisTests            # for pvalue()
using KahanSummation             # for time decay model, true random effect model
using LinearAlgebra              # extract diagnol and Matrix(I,...)
using NLSolversBase              # for hessian!
using Optim
using PrettyTables               # making tables 
using QuadGK                     # for TFE_CSW2014 model
using Random                     # for sfmodel_boot_marginal
using RowEchelon                 # for checkCollinear, check multi-collinearity
using SpecialFunctions           # for erfi used in sfmodel_MoMTest 
using StatsFuns                  # for normlogpdf(), normlogcdf()
using Statistics                 #



#############################
##   Define Model Types    ##
#############################


abstract type Sfmodeltype end
  struct Trun      <: Sfmodeltype end
  struct truncated <: Sfmodeltype end
  struct trun      <: Sfmodeltype end
  struct t         <: Sfmodeltype end 
  struct Half      <: Sfmodeltype end
  struct half      <: Sfmodeltype end
  struct h         <: Sfmodeltype end 
  struct Expo      <: Sfmodeltype end
  struct expo      <: Sfmodeltype end
  struct e <: Sfmodeltype end 
  struct Trun_Scale   <: Sfmodeltype end
  struct trun_scale   <: Sfmodeltype end
  struct trun_scaling <: Sfmodeltype end
  struct s <: Sfmodeltype end  
  struct Half_Scale   <: Sfmodeltype end
  struct half_scale   <: Sfmodeltype end
  struct half_scaling <: Sfmodeltype end
  struct MoM          <: Sfmodeltype end

  struct PFEWHT   <: Sfmodeltype end # panel fixed-effet of Wang and Ho 2010, truncated normal
  struct PFEWHH   <: Sfmodeltype end # panel fixed-effet of Wang and Ho 2010, half normal
  struct PTREH    <: Sfmodeltype end # panel true random effect model, half normal
  struct PTRET    <: Sfmodeltype end # panel true random effect model, truncated normal
  struct PFECSWH <: Sfmodeltype end
  struct PanDecay <: Sfmodeltype end # Panel Time Decay Model of Battese and Coelli (1992)
  struct PanKumb90   <: Sfmodeltype end # Kumbhakar (1990)

abstract type PorC end
  struct production <: PorC end
  struct cost <: PorC end

abstract type PanelModel end
  struct TFE_WH2010  <: PanelModel end
  struct TFE_CSW2014 <: PanelModel end
  struct TFE_G2005   <: PanelModel end
  struct TRE         <: PanelModel end
  struct TimeDecay   <: PanelModel end
  struct Kumbhakar1990 <: PanelModel end

abstract type TableFormat end
  struct text   <: TableFormat end
  struct html   <: TableFormat end
  struct latex  <: TableFormat end



################################################
##    include other files; order important    ##
################################################


include("SFmacfun.jl")
include("SFloglikefun.jl")
include("SFutil.jl")
include("SFmtest.jl")
include("SFgetvars.jl")
include("SFindex.jl")
include("SFpredict.jl")
include("SFmarginal.jl")
include("SFmainfun.jl")


end # SFrontiers module
