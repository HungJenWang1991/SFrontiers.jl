# gpu_draw_memory_runner.jl -- the draw-memory checks on the GPU, in a process of their own.
#
# The panel likelihood on a device spec (per-firm draws, processed one group of firms at a time) must
# equal the copies of the functions before the per-group rewrite (draw_memory_reference.jl), value,
# gradient and Hessian, bit for bit.
#
# Run in a separate process because `using CUDA` must precede `using SFrontiers` for the GPU
# overloads to register; test_gpu_draw_memory.jl launches it the way test_gpu_equivalence.jl
# launches its sweep. Prints one line per check and a final
# "DRAW-MEMORY GPU CHECKS COMPLETE: n checks, m mismatches" summary.
using CUDA
using SFrontiers, ForwardDiff, Optim, Random

CUDA.allowscalar(false)
println("cuda_functional=", CUDA.functional())

include(joinpath(@__DIR__, "draw_memory_reference.jl"))
const PB = SFrontiers.Panel_Backend

const NCHECKS = Ref(0)
const NBAD    = Ref(0)

# One check: prints its name and outcome; a check that throws counts as a mismatch.
function check(name, ok)
    NCHECKS[] += 1
    same = try
        ok()
    catch e
        println(name, " ERROR ", typeof(e), ": ", first(split(sprint(showerror, e), "\n")))
        false
    end
    same || (NBAD[] += 1)
    println(name, " equal=", same)
    flush(stdout)
end

grad(f, p) = ForwardDiff.gradient(f, p, ForwardDiff.GradientConfig(nothing, p), Val{false}())
hess(f, p) = ForwardDiff.hessian(f, p, ForwardDiff.HessianConfig(nothing, p), Val{false}())

# Per-firm draws, 40 firms, S = 128: MCI half-normal with Float64 data at chunks 7 (six groups of
# 6 firms and one of 4) and MSLE Weibull with Float32 data at chunks 10.
d = _ref_panel_data(40, 11)
for (meth, ineff, T, k) in ((:MCI, :HalfNormal, Float64, 7), (:MSLE, :Weibull, Float32, 10))
    isp = _ref_panel_spec(d, ineff; method = meth, D = 128, multiRand = true, T = T, GPU = true)
    θ = _ref_theta(d, ineff; T = T)
    fnew = p -> PB.panel_nll(isp, p; chunks = k)
    fold = p -> _ref_panel_nll(isp, p; chunks = k)
    name = "panel/$meth/$ineff/$T/chunks$k"
    check("$name/value",    () -> isequal(fnew(θ), fold(θ)))
    check("$name/gradient", () -> isequal(grad(fnew, θ), grad(fold, θ)))
    check("$name/Hessian",  () -> isequal(hess(fnew, θ), hess(fold, θ)))
end

# The efficiency indices of GPU fits, computed per group. A cross-sectional MCI fit (Float64, half-normal,
# N = 400, S = 256, chunks 10): its jlms and bc against the indices computed at once (chunks = 1), to 1e-12,
# because the device's row reduction depends on the number of rows. A panel MCI fit (half-normal, 40 firms,
# S = 128, chunks 7): its indices per group against the copy of the function before the per-group rewrite,
# bit for bit, and the fit's jlms against them.
og = sfmodel_opt(main_solver = NelderMead(), main_opt = (iterations = 1,))
rng = Xoshiro(4002)
xg = randn(rng, 400)
Xg = hcat(ones(400), xg)
yg = Xg * [1.0, 0.5] .+ 0.3 .* randn(rng, 400) .- abs.(0.6 .* randn(rng, 400))
spg = sfmodel_spec(depvar = yg, frontier = Xg, noise = :Normal, ineff = :HalfNormal, type = :prod)
rg = sfmodel_fit(spec = spg, method = sfmodel_method(method = :MCI, n_draws = 256, GPU = true, chunks = 10),
                 optim_options = og, marginal = false, show_table = false, verbose = false)
refg = SFrontiers.MCI_Backend.jlms_bc_indices(rg.model, rg.coeff; chunks = 1)
check("cross/MCI/HalfNormal/GPU fit/jlms vs chunks=1", () -> isapprox(rg.jlms, Array(refg.jlms); rtol = 1e-12))
check("cross/MCI/HalfNormal/GPU fit/bc vs chunks=1",   () -> isapprox(rg.bc, Array(refg.bc); rtol = 1e-12))
dp = _ref_panel_data(40, 11)
rp = sfmodel_fit(spec = _ref_unified_spec(dp, :HalfNormal),
                 method = sfmodel_method(method = :MCI, n_draws = 128, GPU = true, chunks = 7),
                 optim_options = og, marginal = false, show_table = false, verbose = false)
outp = PB.panel_jlms_bc(rp.model, rp.coeff; chunks = 7)
refp = _ref_panel_jlms_bc(rp.model, rp.coeff)
check("panel/MCI/HalfNormal/GPU fit/jlms chunks7", () -> outp.jlms == refp.jlms)
check("panel/MCI/HalfNormal/GPU fit/bc chunks7",   () -> outp.bc == refp.bc)
check("panel/MCI/HalfNormal/GPU fit/fit jlms",     () -> rp.jlms == outp.jlms)

# predict on GPU fits reads the draw block stored on the result, which stays on the device. Without `draws=`
# it must agree bit for bit with predict given that block back as a host Float64 array: for the cross-sectional
# MCI fit above, jlms and bc on its frame and jlms on a 430-row frame (the 400 rows followed by rows 1-30, so
# that the draw rows of the last group are not consecutive); for a cross-sectional MSLE fit with Float32 data
# (the same data, half-normal, N = 400, S = 256, chunks 10), jlms and bc; and for the panel MCI fit above,
# jlms and bc on its frame plus a new firm, id 41, with the four rows of firm 1, which uses the draws of
# estimation firm 1 (the last group of firms then reads draw rows 37-40 and 1). predict agrees with the MCI
# fit's jlms to 1e-10, because the fit computes them on the device. A GPU fit given a CuArray of draws in the
# data's element type keeps that array, not a copy.
hg = Array(rg.model.draws_2D)                           # the stored block of rg, copied to the host
Xg430 = vcat(Xg, Xg[1:30, :])
yg430 = vcat(yg, yg[1:30])
check("cross/MCI/HalfNormal/GPU fit/predict jlms, stored vs passed draws",
      () -> predict(rg; frontier = Xg, depvar = yg, what = :jlms) ==
            predict(rg; frontier = Xg, depvar = yg, draws = hg, what = :jlms))
check("cross/MCI/HalfNormal/GPU fit/predict bc, stored vs passed draws",
      () -> predict(rg; frontier = Xg, depvar = yg, what = :bc) ==
            predict(rg; frontier = Xg, depvar = yg, draws = hg, what = :bc))
check("cross/MCI/HalfNormal/GPU fit/predict jlms 430 rows, stored vs passed draws",
      () -> predict(rg; frontier = Xg430, depvar = yg430, what = :jlms) ==
            predict(rg; frontier = Xg430, depvar = yg430, draws = hg, what = :jlms))
check("cross/MCI/HalfNormal/GPU fit/predict jlms vs fit jlms",
      () -> isapprox(predict(rg; frontier = Xg, depvar = yg, what = :jlms), rg.jlms; rtol = 1e-10))
Xg32 = Float32.(Xg)
yg32 = Float32.(yg)
rs32 = sfmodel_fit(spec = sfmodel_spec(depvar = yg32, frontier = Xg32, noise = :Normal, ineff = :HalfNormal,
                                       type = :prod),
                   method = sfmodel_method(method = :MSLE, n_draws = 256, GPU = true, chunks = 10),
                   optim_options = og, marginal = false, show_table = false, verbose = false)
hs32 = Float64.(Array(rs32.model.draws_2D))             # the stored Float32 block, widened on the host
check("cross/MSLE/HalfNormal/Float32/GPU fit/predict jlms, stored vs passed draws",
      () -> predict(rs32; frontier = Xg32, depvar = yg32, what = :jlms) ==
            predict(rs32; frontier = Xg32, depvar = yg32, draws = hs32, what = :jlms))
check("cross/MSLE/HalfNormal/Float32/GPU fit/predict bc, stored vs passed draws",
      () -> predict(rs32; frontier = Xg32, depvar = yg32, what = :bc) ==
            predict(rs32; frontier = Xg32, depvar = yg32, draws = hs32, what = :bc))
hp = Array(rp.model.draws)                              # the stored block of rp, copied to the host
r1 = findall(==(1), dp.id)                              # firm 1: 3 + mod(1, 3) = 4 periods, rows 1-4
Xp41 = vcat(dp.X, dp.X[r1, :])
zp41 = vcat(dp.z, dp.z[r1])
yp41 = vcat(dp.y, dp.y[r1])
idp41 = vcat(dp.id, fill(41, length(r1)))
check("panel/MCI/HalfNormal/GPU fit/predict jlms with a new firm, stored vs passed draws",
      () -> predict(rp; frontier = Xp41, zvar = reshape(zp41, :, 1), depvar = yp41, id = idp41,
                    what = :jlms) ==
            predict(rp; frontier = Xp41, zvar = reshape(zp41, :, 1), depvar = yp41, id = idp41,
                    draws = hp, what = :jlms))
check("panel/MCI/HalfNormal/GPU fit/predict bc with a new firm, stored vs passed draws",
      () -> predict(rp; frontier = Xp41, zvar = reshape(zp41, :, 1), depvar = yp41, id = idp41,
                    what = :bc) ==
            predict(rp; frontier = Xp41, zvar = reshape(zp41, :, 1), depvar = yp41, id = idp41,
                    draws = hp, what = :bc))
dgpu = CuArray(rand(Xoshiro(5), 400, 256))
ru = sfmodel_fit(spec = spg,
                 method = sfmodel_method(method = :MCI, n_draws = 256, GPU = true, chunks = 10, draws = dgpu),
                 optim_options = og, marginal = false, show_table = false, verbose = false)
check("cross/MCI/HalfNormal/GPU fit/user CuArray draws kept without a copy", () -> ru.model.draws_2D === dgpu)

# The MSLE Laplace density on the device: the negative log-likelihood of a GPU spec against that of a CPU spec, at
# chunks 1 and 10, to 1e-10 relative. Laplace noise (b = 0.3) and half-normal inefficiency (σᵤ = 0.6), N = 1,000,
# production frontier, 512 shared draws, at the true parameters.
rl = Xoshiro(21)
xl = randn(rl, 1000)
U1l = rand(rl, 1000)
U2l = rand(rl, 1000)
zl = randn(rl, 1000)
Xl = hcat(ones(1000), xl)
yl = Xl * [1.0, 0.5] .+ 0.3 .* (log.(U1l) .- log.(U2l)) .- abs.(0.6 .* zl)
spl = sfmodel_spec(depvar = yl, frontier = Xl, noise = :Laplace, ineff = :HalfNormal, type = :prod)
θl = sfmodel_init(spec = spl, frontier = [1.0, 0.5], ln_sigma_u_sq = log(0.36), ln_b = log(0.3), message = false)
MSB = SFrontiers.MSLE_Backend
ispl_gpu = MSB._assemble_MSLE_spec(spl.msle_spec,
                                   sfmodel_method(method = :MSLE, n_draws = 512, multiRand = false, GPU = true))
ispl_cpu = MSB._assemble_MSLE_spec(spl.msle_spec, sfmodel_method(method = :MSLE, n_draws = 512, multiRand = false))
for k in (1, 10)
    check("cross/MSLE/Laplace/HalfNormal/GPU vs CPU spec/nll chunks$k",
          () -> isapprox(MSB.qmc_nll(ispl_gpu, θl; chunks = k), MSB.qmc_nll(ispl_cpu, θl; chunks = k); rtol = 1e-10))
end

# The copula Gamma likelihood on the device, where the regularized incomplete gamma that gives the copula the CDF of
# the inefficiency runs inside the broadcasts, with ForwardDiff duals for the gradient: the negative log-likelihood
# of a GPU spec against that of a CPU spec to 1e-10 relative, and its gradient to 1e-8 relative, at chunks 10.
# Gamma inefficiency (k = 2, θ = 0.4) with a Gaussian copula, normal noise (σᵥ = 0.3), N = 300, 256 draws,
# production frontier, at k = 2.
rc = Xoshiro(67)
Xc = [ones(300) randn(rc, 300) randn(rc, 300)]
uc = 0.4 .* (randexp(rc, 300) .+ randexp(rc, 300))
yc = Xc * [1.0, 0.5, -0.3] .+ 0.3 .* randn(rc, 300) .- uc
spc = sfmodel_spec(depvar = yc, frontier = Xc, noise = :Normal, ineff = :Gamma, copula = :Gaussian, type = :prod)
θc = sfmodel_init(spec = spc, frontier = [1.0, 0.5, -0.3], ln_k = log(2.0), ln_theta = log(0.4),
                  ln_sigma_v_sq = log(0.09), theta_rho = [0.3], message = false)
MCB = SFrontiers.MCI_Backend
ispc_gpu = MCB._assemble_MCI_spec(spc.mci_spec, sfmodel_method(method = :MCI, n_draws = 256, GPU = true))
ispc_cpu = MCB._assemble_MCI_spec(spc.mci_spec, sfmodel_method(method = :MCI, n_draws = 256))
fc_gpu = p -> MCB.MCI_nll(ispc_gpu, p; chunks = 10)
fc_cpu = p -> MCB.MCI_nll(ispc_cpu, p; chunks = 10)
check("cross/MCI/Gamma/Gaussian copula/GPU vs CPU spec/nll chunks10",
      () -> isapprox(fc_gpu(θc), fc_cpu(θc); rtol = 1e-10))
check("cross/MCI/Gamma/Gaussian copula/GPU vs CPU spec/gradient chunks10",
      () -> isapprox(ForwardDiff.gradient(fc_gpu, θc), ForwardDiff.gradient(fc_cpu, θc); rtol = 1e-8))

println("DRAW-MEMORY GPU CHECKS COMPLETE: $(NCHECKS[]) checks, $(NBAD[]) mismatches")
exit(NBAD[] == 0 ? 0 : 1)
