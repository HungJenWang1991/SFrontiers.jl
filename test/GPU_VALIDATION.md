# GPU validation: the CPU–GPU equivalence sweep

SFrontiers evaluates its simulated likelihoods (MCI and MSLE, cross-sectional and `panel_TFE`) on an NVIDIA GPU when
`sfmodel_method` is given `GPU = true`. The CPU and GPU code paths must compute the same likelihood, efficiency
indices (JLMS and BC) and marginal effects. The sweep checks this at fixed parameters and by full fits from the same
inputs.

## Requirements

- An NVIDIA GPU on which CUDA.jl is functional.
- CUDA.jl installed in the default Julia environment. It is not a package dependency of SFrontiers; with
  `--project=.` Julia still finds the packages of the default environment through the standard load path.
- The runners load CUDA before SFrontiers, as the GPU methods of SFrontiers require.

## Commands

From the package root:

```
julia --project=. test/gpu_sweep_runner.jl                  # the full sweep, all tiers
julia --project=. test/gpu_sweep_runner.jl --tiers=1,2      # the tiers named
julia --project=. test/gpu_sweep_runner.jl --smoke          # the 7 smoke cases
julia --project=. test/gpu_sweep_runner.jl --out=PATH       # results to PATH
julia --project=. test/gpu_custom_draw_runner.jl            # predict with a GPU fit's own draw object
```

The options of the sweep runner can be combined. Without `--out`, the results go to `gpu_sweep_results.jsonl` in the
package root, and the progress to that path with `.progress.txt` appended.

## What the tiers cover

The full sweep has 83 cases:

- **Tier 1** (21 cases): every inefficiency distribution each method offers (the Gamma under MCI only).
- **Tier 2** (44 cases): `chunks` 1, 7 and 403 (empty trailing chunks), shared draws, cost frontiers,
  heteroscedastic and scaling models, the four copulas, Erlang.
- **Tier 3** (9 cases): the Wang–Ho panel (`panel_TFE`) under MCI and MSLE (half normal, Weibull, Erlang2; `chunks`
  1, 7 and the default; shared draws).
- **Tier 4** (9 cases): user draws on either device, post-fit consumers of a GPU result, Float32 data, and a long
  Float32 Halton sequence (fixed-parameter checks only).

## Tolerances

A relative difference is |a − b| divided by max(|a|, |b|, 1).

- **At fixed parameters** (the same parameter vector on a CPU and a GPU specification, no optimizer involved):
  relative 1e-10 for the log-likelihood, JLMS, BC and marginal effects (1e-4 with Float32 data).
- **Full fits** (a CPU fit and a GPU fit from the same inputs and the same draws): relative 1e-6 for the
  coefficients, the log-likelihood and JLMS (5e-3 with Float32 data).
- **The ridge rule**: when both fits converged and their coefficients differ by more than that tolerance, each
  device evaluates its log-likelihood at the CPU fit's and at the GPU fit's coefficients, and the case passes as a
  ridge pass when the two values agree within 1e-8 nats on each device. A flat likelihood lets two correct
  optimizers stop at different points with equal log-likelihoods; the fixed-parameter checks still test the devices
  against each other. When the coefficients agree but the log-likelihoods or JLMS do not, the case fails.

The fits use 200 draws because they check device equivalence, not accuracy (the package default is 4,096).

## Expected output

The sweep runner prints the environment record, one line per case with its verdict and time, and at the end:

```
===== SWEEP COMPLETE =====
cases: 83; clean: …; ridge passes: …; needs-attention: 0
largest fixed-parameter relative difference: Float64 cases … (case, check); Float32 cases … (case, check)
```

followed by the ridge passes, each with `dll_cpu` and `dll_gpu` (on each device, the log-likelihood at the CPU fit's
coefficients minus that at the GPU fit's), and by the cases that need attention. A case needs attention when a check
failed or raised an error, and is a ridge pass when its full fits passed by the ridge rule; otherwise it is clean,
also when a post-fit stage passed only under `CUDA.@allowscalar` or its full fits are recorded as `flags_only`. The
exit code is 0 when no case needs attention and 1 otherwise.

The custom-draw runner prints one line per case and `CUSTOM-DRAW ROUNDTRIP COMPLETE: n cases, m mismatches`, and
exits 0 when there is no mismatch and 1 otherwise.

### Reading the JSONL file

Each run first writes its environment record (`"kind":"environment"`: the versions of Julia, SFrontiers, CUDA.jl
and the CUDA runtime, the CUDA version that the driver supports, the device, the date in UTC, the tiers and the
number of cases), then one line per case: its settings and its `checks`, each with a `status`:

| `status` | Meaning |
|---|---|
| `ok` | a stage that ran without error: building the specification, a fit, a post-fit consumer |
| `pass` | a comparison within its tolerance |
| `pass_ridge` | full fits whose coefficients differ, passed by the ridge rule (the four log-likelihoods are under `ridge`) |
| `flags_only` | full fits of which at least one did not converge: their flags and differences are recorded, not judged |
| `skipped` | a comparison with nothing to compare (no marginal effects on either device) |
| `ok_with_allowscalar` | a post-fit stage that failed on scalar indexing and then ran under `CUDA.@allowscalar` |
| `fail` | a comparison outside its tolerance |
| `error` | a stage or a comparison that raised an error |

The runner appends to its output file, so give each run its own `--out` path.

## Run time

On an NVIDIA GeForce RTX 5090 (Julia 1.12.6 with 12 threads, CUDA.jl 5.9.6, CUDA runtime 13.1), each time
including the start of Julia: the full sweep about 16 minutes, the smoke subset about 2.4 minutes, the
custom-draw runner about 1.5 minutes. The release record below gives the time of the release run.

## The GPU tests inside `Pkg.test()`

Each runs in a subprocess and is skipped when CUDA is not functional:

- `test/test_gpu_equivalence.jl`: the 7 smoke cases of the sweep, then `test/gpu_memory_probe.jl` (the GPU memory
  estimate and warning).
- `test/test_input_forms_and_refusals.jl`: the custom-draw round trip (`test/gpu_custom_draw_runner.jl`).
- `test/test_gpu_draw_memory.jl` (`test/gpu_draw_memory_runner.jl`): the draw block, the grouped panel likelihood
  and the indices on the device, the MSLE Laplace likelihood, and the likelihood and gradient of the MCI Gamma model
  with a Gaussian copula.

## Release record

The record is the release run of the full sweep on an NVIDIA GeForce RTX 5090.

| GPU | Driver | CUDA runtime | CUDA.jl | Julia | SFrontiers | Date | Cases | Clean | Ridge passes | Needs attention | Largest fixed-parameter relative difference (Float64; Float32 cases) | Run time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 5090 | 591.86 (CUDA 13.1.0) | 13.1.0 | 5.9.6 | 1.12.6 | 2.0.0 | 2026-10-30 | 83 | 80 | 3 | 0 | Float64 8.9e-13 (T2-MCI-HN-copulaClayton, fixed_jlms); Float32 9.5e-06 (T4-float32-long-MCI-HN, fixed_jlms) | 15.9 min |
| NVIDIA GeForce RTX 5090 | 591.86 (CUDA 13.1.0) | 13.4.0 | 6.4.2 | 1.13.3 | 2.0.0 | 2026-10-31 | 83 | 81 | 2 | 0 | Float64 1.6e-12 (T2-MCI-HN-copulaClayton, fixed_jlms); Float32 9.5e-06 (T4-float32-long-MCI-HN, fixed_jlms) | 14.3 min |
