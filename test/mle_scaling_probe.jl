# ---------------------------------------------------------------------------
# mle_scaling_probe.jl — focused arbitration of the TN-scaling-prod jlms
# mismatch (MLE analytic vs MCI simulation at the MLE fit optimum θ̂).
#
# Oracles:
#   analytic  — UNCLAMPED closed-form E[u|ε], E[e⁻ᵘ|ε] for the scaled TN model
#               (u = h·u*, u*~TN(μ0,σu0); u|ε ~ TN(μₛ,σₛ) truncated at 0)
#   p-space   — quadrature over p∈(0,1) with u = h·Q_TN(p) (what MSLE with
#               infinite draws computes), log-max-stabilized
# Evaluated at BOTH the healthy base θ and the fit optimum θ̂.
#
# Usage: julia --project=. test/mle_scaling_probe.jl
# ---------------------------------------------------------------------------

using Random, Statistics, LinearAlgebra, Dates, Printf
using Distributions, QuadGK
using SFrontiers, Optim

include(joinpath(@__DIR__, "noise_oracle_lib.jl"))
include(joinpath(@__DIR__, "mle_crosscheck_lib.jl"))

const NPROBE = 8

case  = CrosscheckCase(name="TN-scaling-prod", ineff=:TruncatedNormal, hetero=:scaling)
dat   = make_crosscheck_data(type=:prod)
uspec = build_uspec(case, dat)

opt = sfmodel_opt(warmstart_solver = NelderMead(),
                  warmstart_opt    = (iterations = 200, g_abstol = 1e-4),
                  main_solver      = Newton(),
                  main_opt         = (iterations = 300, g_abstol = 1e-8,
                                      show_trace = false))
res = sfmodel_fit(spec=uspec, method=sfmodel_method(method=:MLE), init=nothing,
                  optim_options=opt, marginal=false, show_table=false, verbose=false)

ispec  = assemble_sim(case, uspec, :MCI,  2^16)
ispec_s = assemble_sim(case, uspec, :MSLE, 2^16)
idx = ispec.idx

function unpack(θ)
    β   = θ[_asvec(idx.beta)]
    δ   = θ[_flatten_idx(idx.delta)]
    μ0  = θ[_asvec(idx.ineff.mu)[1]]
    w   = θ[_asvec(idx.ineff.sigma_u)[1]]
    γv  = θ[first(_asvec(idx.noise.ln_sigma_v_sq))]
    (β=β, δ=δ, μ0=μ0, σu0=exp(0.5w), σv=exp(0.5γv), w=w, γv=γv)
end

"Unclamped closed-form jlms/bc for the scaled TN model."
function analytic_jlms_bc(θ)
    p = unpack(θ)
    h   = exp.(dat.Zs * p.δ)
    ε   = dat.y .- dat.X * p.β
    μi  = p.μ0 .* h
    σu2 = (p.σu0^2) .* h.^2
    σv2 = p.σv^2
    σ2  = σu2 .+ σv2
    μs  = (σv2 .* μi .- σu2 .* ε) ./ σ2
    σs  = sqrt.(σv2 .* σu2 ./ σ2)
    n01 = Normal()
    r   = μs ./ σs
    jl  = @. σs * exp(logpdf(n01, r) - logcdf(n01, r)) + μs
    bc  = @. exp(-μs + 0.5 * σs^2 + logcdf(n01, r - σs) - logcdf(n01, r))
    (jlms=jl, bc=bc, h=h, mu=μi, su2=σu2)
end

"p-space quadrature E[u|ε], E[e⁻ᵘ|ε] for obs i (log-max stabilized)."
function pquad_jlms_bc(θ, i)
    p = unpack(θ)
    h  = exp(dot(dat.Zs[i, :], p.δ))
    ε  = dat.y[i] - dot(dat.X[i, :], p.β)
    tn = truncated(Normal(p.μ0, p.σu0), 0.0, Inf)
    nv = Normal(0.0, p.σv)
    uofp(q) = h * quantile(tn, q)
    lw(q)   = logpdf(nv, ε + uofp(q))          # s=+1 (prod)
    ps = range(1e-12, 1 - 1e-12, length=4001)
    M  = maximum(lw.(ps))
    den = quadgk(q -> exp(lw(q) - M), 0.0, 1.0; rtol=1e-10)[1]
    nj  = quadgk(q -> uofp(q) * exp(lw(q) - M), 0.0, 1.0; rtol=1e-10)[1]
    nb  = quadgk(q -> exp(-uofp(q)) * exp(lw(q) - M), 0.0, 1.0; rtol=1e-10)[1]
    (jlms=nj / den, bc=nb / den)
end

"Direct MLE jlmsbc call (with its clamps), rho in MLE order."
function mle_jlms_bc(θ)
    rho = collect(Float64, SFrontiers._reorder_init_mci_to_mle(uspec, θ))
    N = dat.N
    pos = (begx=1, endx=3, begz=4, endz=4, begq=5, endq=6,
           begw=7, endw=7, begv=8, endv=8)
    SFmle2.jlmsbc(SFmle2.Trun_Scale, 1, pos, rho,
                  reshape(dat.y, N, 1), dat.X, ones(N, 1), dat.Zs,
                  ones(N, 1), ones(N, 1), ())
end

θbase = build_theta2(ispec, theta_vals(case, :base))
θhat  = collect(Float64, res.coeff)

out = open(joinpath(dirname(@__DIR__), "mle_scaling_probe.jsonl"), "w")
for (label, θ) in ((:base, θbase), (:hat, θhat))
    pβ = unpack(θ)
    ana = analytic_jlms_bc(θ)
    jm, bm = mle_jlms_bc(θ)
    rm = SFrontiers.MCI_Backend.jlms_bc_indices(ispec, θ; chunks=1)
    rs = SFrontiers.MSLE_Backend.jlms_bc_indices(ispec_s, θ; chunks=1)
    pq = [stage(() -> pquad_jlms_bc(θ, i)) for i in 1:NPROBE]
    rec = Dict{Symbol,Any}(
        :theta => label,
        :params => Dict(:mu0 => pβ.μ0, :sigma_u0 => pβ.σu0, :sigma_v => pβ.σv,
                        :delta => pβ.δ, :w_raw => pβ.w,
                        :h_first8 => collect(ana.h[1:NPROBE]),
                        :sigma_u2_scaled_first8 => collect(ana.su2[1:NPROBE])),
        :jlms_mle_clamped   => collect(jm[1:NPROBE]),
        :jlms_analytic_uncl => collect(ana.jlms[1:NPROBE]),
        :jlms_mci_sim       => collect(rm.jlms[1:NPROBE]),
        :jlms_msle_sim      => collect(rs.jlms[1:NPROBE]),
        :jlms_pquad         => [s[:status] == :ok ? s[:value].jlms : s[:error] for s in pq],
        :bc_mle_clamped     => collect(bm[1:NPROBE]),
        :bc_analytic_uncl   => collect(ana.bc[1:NPROBE]),
        :bc_mci_sim         => collect(rm.bc[1:NPROBE]),
        :bc_pquad           => [s[:status] == :ok ? s[:value].bc : s[:error] for s in pq],
        :jlms_mle_fitstored => label === :hat ? collect(res.jlms[1:NPROBE]) : nothing)
    println(out, json_line(rec)); flush(out)
    println("=== θ = $label ===")
    println("  mu0=$(pβ.μ0)  sigma_u0=$(pβ.σu0)  sigma_v=$(pβ.σv)  delta=$(pβ.δ)")
    @printf("  %-4s %-13s %-13s %-13s %-13s %-13s\n",
            "obs", "mle_clamped", "analytic", "mci_sim", "msle_sim", "pquad")
    for i in 1:NPROBE
        pqv = pq[i][:status] == :ok ? pq[i][:value].jlms : NaN
        @printf("  %-4d %-13.6g %-13.6g %-13.6g %-13.6g %-13.6g\n",
                i, jm[i], ana.jlms[i], rm.jlms[i], rs.jlms[i], pqv)
    end
end
close(out)
println("probe complete")
