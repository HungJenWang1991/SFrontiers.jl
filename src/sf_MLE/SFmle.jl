# Copyright (C) 2025 Hung-Jen Wang
# SPDX-License-Identifier: GPL-3.0-or-later

module SFmle

export sfmodel_spec, sfmodel_init, sfmodel_opt,
  sfmodel_fit, sfmodel_predict, drawTrun,
  genTrun, sf_demean, sfmodel_boot_marginal,
  get_rowIDT, #! experiment
  # likelihood functions
  LL_T,
  # new SFrontiers-style DSL types and macros
  SFModelSpec_MLE,
  DSLArg_MLE,
  WUseDataSpec_MLE, WDepvarSpec_MLE, WFrontierSpec_MLE,
  WZvarSpec_MLE, WIdSpec_MLE,
  @useData, @zvar, @id,
  sf_depvar, sf_frontier, sf_idvar,
  # old macros for _spec (kept for backward compat)
  @depvar, @frontier, @idvar,
  depvar, idvar,
  sfdist, sftype,
  @sfdist, @sftype,
  @model,
  @μ, @mu, @σᵤ², @sigma_u_2, @σᵥ², @sigma_v_2,
  @σₐ², @sigma_a_2,
  @hscale,
  @sfpanel,
  sfpanel,
  @gamma,
  @eq,
  # functions for sfmodel_spec (also used by old Vararg sfmodel_spec)
  frontier,
  μ, mu, σᵤ², sigma_u_2, σᵥ², sigma_v_2,
  σₐ², sigma_a_2,
  hscale, gamma,
  # (sfmodel_opt helper functions removed — now uses keyword API directly)
  # (useData function removed — sfmodel_fit now uses keyword API directly)
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
  PFEWHH, PFEWHT, PTREH, PTRET, PFECSWH, PanDecay, get_marg, PanKumb90,  #* export for testing purpose
  TFE_WH2010, TFE_CSW2014, TFE_G2005, TRE, TimeDecay, Kumbhakar1990,
  # Optim's algorithms
  NelderMead, SimulatedAnnealing, SAMIN, ParticleSwarm,
  ConjugateGradient, GradientDescent, BFGS, LBFGS,
  Newton, NewtonTrustRegion, IPNewton,
  # sfmodel_mtest()
  pickConsNameFromDF
  


using DataFrames
using OrderedCollections: OrderedDict
using Distributions              # for TDist, Normal
using FLoops                     # multithreading
using ForwardDiff                # for marginal effect
# using HaltonSequences, Sobol, RCall #! experiment, for genSequence; ENV["R_HOME"] = "d:/MathStat/R/R-4.0.4";  Pkg.build("RCall");  using RCall
# HypothesisTests removed — pvalue() replaced by ccdf() from Distributions
# KahanSummation removed — sum_kbn() replaced by sum()
using LinearAlgebra              # extract diagnol and Matrix(I,...)
# NLSolversBase removed — using direct Optim.optimize + ForwardDiff.hessian instead
using Optim
using PrettyTables: pretty_table, fmt__printf  # making tables
using QuadGK                     # for TFE_CSW2014 model
using Random                     # for sfmodel_boot_marginal
# RowEchelon removed — checkCollinear now uses rank/QR from LinearAlgebra
using SpecialFunctions           # for erfi used in sfmodel_MoMTest 
using ADTypes                    # for AutoForwardDiff() in TwiceDifferentiable
using StatsFuns                  # for normlogpdf(), normlogcdf()
using Statistics                 #

#############################
##   Define Model Types    ##
#############################


abstract type Sfmodeltype end
struct Trun <: Sfmodeltype end
struct truncated <: Sfmodeltype end
struct trun <: Sfmodeltype end
struct t <: Sfmodeltype end
struct Half <: Sfmodeltype end
struct half <: Sfmodeltype end
struct h <: Sfmodeltype end
struct Expo <: Sfmodeltype end
struct expo <: Sfmodeltype end
struct e <: Sfmodeltype end
struct Trun_Scale <: Sfmodeltype end
struct trun_scale <: Sfmodeltype end
struct trun_scaling <: Sfmodeltype end
struct s <: Sfmodeltype end
struct Half_Scale <: Sfmodeltype end
struct half_scale <: Sfmodeltype end
struct half_scaling <: Sfmodeltype end
struct MoM <: Sfmodeltype end

struct PFEWHT <: Sfmodeltype end # panel fixed-effet of Wang and Ho 2010, truncated normal
struct PFEWHH <: Sfmodeltype end # panel fixed-effet of Wang and Ho 2010, half normal
struct PTREH <: Sfmodeltype end # panel true random effect model, half normal
struct PTRET <: Sfmodeltype end # panel true random effect model, truncated normal
struct PFECSWH <: Sfmodeltype end  # panel true fixed-effect of Chen, Schmidt, and Wang 2014 JE
struct PFECSWHMM <: Sfmodeltype end  # CSW model using MM, #! experiment
struct PanDecay <: Sfmodeltype end # Panel Time Decay Model of Battese and Coelli (1992)
struct PanKumb90 <: Sfmodeltype end # Kumbhakar (1990)

abstract type PorC end
struct production <: PorC end
struct cost <: PorC end

abstract type PanelModel end
struct TFE_WH2010 <: PanelModel end
struct TFE_CSW2014 <: PanelModel end
struct TFE_G2005 <: PanelModel end
struct TRE <: PanelModel end
struct TimeDecay <: PanelModel end
struct Kumbhakar1990 <: PanelModel end

abstract type TableFormat end
struct text <: TableFormat end
struct html <: TableFormat end
struct latex <: TableFormat end




######################################
##   Model Specification Struct     ##
######################################

struct SFModelSpec_MLE{T<:AbstractFloat}
    depvar::Vector{T}
    frontier::Matrix{T}
    zvar::Matrix{T}                           # shared Z matrix (ones(N,1) if no zvar)
    noise::Symbol                             # :Normal (only supported for analytic MLE)
    ineff::Symbol                             # :HalfNormal, :TruncatedNormal, :Exponential
    hetero::Union{Vector{Symbol}, Symbol}     # which params use zvar, or :scaling
    type_sign::Int                            # +1 (production) or -1 (cost)
    # Panel fields
    panel::Union{Nothing, Symbol}             # nothing=cross-sectional, or :TFE_WH2010, etc.
    idvar::Union{Nothing, Vector}
    # Scaling
    scaling::Bool
    scaling_zvar::Union{Nothing, Matrix{T}}
    # Dimensions
    N::Int
    K::Int                                    # frontier columns
    L::Int                                    # zvar columns
    # Model dispatch
    modelid::Type{<:Sfmodeltype}              # Half, Trun, Expo, PFEWHT, etc.
    # Metadata
    varnames::Vector{String}
    eqnames::Vector{String}
    eq_indices::Vector{Int}
end


################################################
##    include other files; order important    ##
################################################

include("SFmacfun.jl")
# include("0000SFtemp.jl")    #! experiment; quasi random number generator used for MSLE
include("SFloglikefun.jl")
include("SFutil.jl")
include("SFmtest.jl")
include("SFgetvars.jl")
include("SFindex.jl")
include("SFpredict.jl")
include("SFmarginal.jl")
include("SFmainfun.jl")


end # SFmle module
