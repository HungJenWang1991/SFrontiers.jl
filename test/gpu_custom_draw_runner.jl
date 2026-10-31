# gpu_custom_draw_runner.jl — predict with the GPU draw object that a GPU fit took
# (README: test/GPU_VALIDATION.md).
#
# A GPU fit accepts a `CuArray` as `draws=`. predict uses the draws stored on the result when
# none are passed and accepts the same draws when they are passed, the `CuArray` itself
# included, although every predict path runs on the host.
#
# The check: for each backend and each draw layout, prediction with the GPU object must equal
# prediction with its CPU copy, bit for bit. Whether either equals the jlms and bc that the GPU
# fit computed on the device is not asserted here: the two devices can differ by about 1e-15.
#
# Run in a separate process because `using CUDA` must precede `using SFrontiers` for the GPU
# overloads to register; `test_input_forms_and_refusals.jl` launches it the way
# `test_gpu_equivalence.jl` launches its sweep. Prints one line per case and a final
# "CUSTOM-DRAW ROUNDTRIP COMPLETE: n cases, m mismatches" summary; exits 0 when there is no
# mismatch, 1 otherwise.
using CUDA
using SFrontiers
using Optim
using Logging

CUDA.allowscalar(false)
println("cuda_functional=", CUDA.functional())

const NCASES = Ref(0)
const NBAD   = Ref(0)

with_logger(NullLogger()) do
    og = sfmodel_opt(main_solver = NelderMead(), main_opt = (iterations = 1,))

    # ---- cross-sectional: shared 1 x D draws and per-observation N x D draws ----
    Ng, Dg = 5, 7
    Xg = hcat(ones(Ng), collect(range(-0.8, 0.8; length = Ng)))
    yg = Xg * [0.85, -0.2] .- abs.(0.25 .* sin.(1:Ng)) .+ 0.06 .* cos.(1:Ng)
    spg = sfmodel_spec(depvar = yg, frontier = Xg, noise = :Normal, ineff = :HalfNormal)

    for meth in (:MCI, :MSLE), shared in (true, false)
        dcpu = shared ? collect(range(0.0625, 0.875; length = Dg)) :
               reshape(collect(range(0.015625, 0.984375; length = Ng * Dg)), Ng, Dg)
        dgpu = CuArray(dcpu)
        r = sfmodel_fit(spec = spg,
                method = sfmodel_method(method = meth, draws = dgpu, GPU = true, chunks = 2),
                init = sfmodel_init(spec = spg, message = false), optim_options = og,
                marginal = false, show_table = false, verbose = false)
        args = (frontier = Xg, depvar = yg)
        for what in (:jlms, :bc)
            NCASES[] += 1
            same = try
                predict(r; args..., what = what, draws = dgpu) ==
                predict(r; args..., what = what, draws = dcpu)
            catch e
                println("cross/$meth/$(shared ? "shared" : "per")/$what ERROR ",
                        typeof(e), ": ", first(split(sprint(showerror, e), "\n")))
                false
            end
            same || (NBAD[] += 1)
            println("cross/$meth/$(shared ? "shared" : "per")/$what gpu_object==cpu_copy=", same)
        end
    end

    # ---- panel_TFE: shared draws and per-firm N_firms x D draws ----
    Nf, Ti, Dp = 3, 2, 7
    NT = Nf * Ti
    idp = repeat(1:Nf, inner = Ti)
    Xp = hcat(collect(range(-0.8, 0.8; length = NT)), sin.(range(0.1, 1.8; length = NT)))
    zp = cos.(range(0.2, 1.9; length = NT))
    yp = Xp * [0.65, -0.18] .- exp.(0.1 .* zp) .* repeat([0.12, 0.2, 0.28], inner = Ti) .+
         0.05 .* sin.(1:NT)
    spp = sfmodel_spec(depvar = yp, frontier = Xp, zvar = zp, id = idp,
                       noise = :Normal, ineff = :HalfNormal, datatype = :panel_TFE)

    for meth in (:MCI, :MSLE), shared in (true, false)
        dcpu = shared ? collect(range(0.0625, 0.875; length = Dp)) :
               reshape(collect(range(0.015625, 0.984375; length = Nf * Dp)), Nf, Dp)
        dgpu = CuArray(dcpu)
        r = sfmodel_fit(spec = spp,
                method = sfmodel_panel_method(method = meth, draws = dgpu, GPU = true, chunks = 2),
                init = sfmodel_init(spec = spp), optim_options = og,
                marginal = false, show_table = false, verbose = false)
        args = (frontier = Xp, zvar = reshape(zp, :, 1), depvar = yp, id = idp)
        for what in (:jlms, :bc)
            NCASES[] += 1
            same = try
                predict(r; args..., what = what, draws = dgpu) ==
                predict(r; args..., what = what, draws = dcpu)
            catch e
                println("panel/$meth/$(shared ? "shared" : "per")/$what ERROR ",
                        typeof(e), ": ", first(split(sprint(showerror, e), "\n")))
                false
            end
            same || (NBAD[] += 1)
            println("panel/$meth/$(shared ? "shared" : "per")/$what gpu_object==cpu_copy=", same)
        end
    end
end

println("CUSTOM-DRAW ROUNDTRIP COMPLETE: $(NCASES[]) cases, $(NBAD[]) mismatches")
exit(NBAD[] == 0 ? 0 : 1)
