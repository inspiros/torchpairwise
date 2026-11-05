# Reproducing Operator Benchmarks

The public harness compares equivalent outputs and first gradients before it
reports timing. Run it from a checkout with a compatible torchpairwise source
build; it needs PyTorch and the package, with no extra benchmark dependencies.

```bash
python examples/benchmark_operators.py --case distance-sum --device cpu
python examples/benchmark_operators.py --case rbf-matmul --device cuda \
    --n 2048 --m 4096 --d 32 --channels 16 --chunk-size 128 \
    --backward --output develop/rbf-matmul.json
python examples/benchmark_operators.py --case distance-quantile --device cuda \
    --n 2049 --m 8193 --d 8 --q 0.9 --backward \
    --output develop/distance-quantile.json
python examples/benchmark_operators.py --case kernel-regression --device cuda \
    --n 2048 --m 4096 --d 32 --channels 4 --chunk-size 512 \
    --backward --output develop/kernel-regression.json
python examples/benchmark_operators.py --case attention --device cuda \
    --n 2048 --m 4096 --d 32 --channels 16 --chunk-size 512 \
    --output develop/attention.json
```

Use `CUDA_VISIBLE_DEVICES` to select an available GPU. Choose shapes that fit
the dense baselines first; dense validation is part of this harness. A failed or
out-of-memory path is recorded as an error, never as a speed measurement, and
the process exits nonzero. CPU uses one Torch thread by default; `--threads`
changes it. All size, chunk, warmup and repeat arguments are positive integers.
`--q` is finite and in [0,1]. Empty-input and invalid metric contracts belong to
unit tests rather than throughput measurements.

## Equivalent Workloads

| Case | Output | Formula / settings |
|---|---|---|
| `distance-sum` | `[N]` | Squared Euclidean distances summed over M |
| `rbf-matmul` | `[N,channels]` | RBF with explicit gamma=1/D, multiplied by V `[M,channels]` |
| `distance-quantile` | `[N]` | Linear q-quantile of squared Euclidean distances over M, stable tie order |
| `kernel-regression` | `[N,channels]` | Nadaraya-Watson RBF smoothing, gamma=1/D; target channels and row mass share one product |
| `attention` | `[N,channels]` | Single-head `softmax((Q @ K.T)/sqrt(D)) @ V`, no mask/dropout |

The [ML example](../examples/streaming_ml.py) contains these two algorithms plus
unweighted k-NN regression, each with a small dense reference and first-gradient
checks. k-NN is a correctness demonstration, not a timed case: the dense oracle
uses stable sorting to match the streamed selector's index tie rule. Its
gradients flow to selected targets only. Regression and attention are timed
end to end, including normalization. Regression floors row masses at
`torch.finfo(dtype).tiny`, returning zero when all RBF weights underflow.

Paths include eager PyTorch, fullgraph compiled PyTorch, native dense functions,
common scalar operators, and named streaming helpers where an equivalent helper
exists. Residual-event streaming is included for the distance cases, reducing D
before the scalar sample consumer. This illustrates that equal results can use
different algorithms and workspace. It does not make generic event streaming
an expected speed winner over a specialized metric.

For distance/RBF cases, the literal PyTorch baseline computes broadcast
differences, squares them and sums features. Its eager version can materialize
N*M*D; its compiled version can
fuse that intermediate. This is **not** a comparison against every optimized
PyTorch distance algorithm. For example, a norm/matmul formulation or
`torch.cdist` has different rounding and sometimes different intermediate
memory. Add another explicitly labeled baseline when that is the workload of
interest. Do not present these measurements as a universal ranking of packages.

Attention instead uses the dense `Q @ K.T` formula directly, without an N*M*D
broadcast intermediate. The comparison is with that dense formula, not with
PyTorch's specialized scaled-dot-product attention backends. Those can already
avoid storing scores and may be faster. The example demonstrates how the same
consumer applies to other scalar operator scores, rather than replacing an
optimized Transformer attention layer.

Inputs are finite, contiguous, unbatched random tensors shared by all paths.
The seed, shape, dtype, gamma convention, q, chunk widths and thread count are
recorded. Quantile uses stable sort plus linear interpolation in the dense
baselines too, matching the operator's tie rule. This finite-input helper is
not an alternative implementation of every `torch.quantile` edge convention.
The harness sets float32 matmul precision to `highest` for all paths.

## Timing and Memory

Without `--backward`, timing runs with autograd disabled. With it, each sample
includes forward plus `grad(output.sum())` for X, Y and V when present. Input
creation and correctness checks are outside timing; operator construction and
native source preparation are inside. Gradients use `autograd.grad`, so there
is no accumulation into persistent `.grad` buffers between samples.

Every path is validated and warmed before measurement. Compiled-PyTorch first
call latency, including compilation/autotuning, is recorded separately; it is
excluded from steady-state samples. `--skip-compile` deliberately omits that
baseline for a quick smoke run. Repeats shuffle path order using the recorded
seed to reduce ordering bias. CUDA is synchronized around wall-clock timing;
results include every sample, median, minimum and maximum. Repeat on a quiet
machine before drawing performance conclusions.

GPU memory is extra `max_memory_allocated` above live allocations just before
each timed sample. The result also records that live baseline and excludes
allocator reservation. No `empty_cache` is inserted between variants. Outputs,
source gradient buffers, temporaries and saved backward state contribute when
live. CPU peak tensor memory is **not measured** and is JSON null; process RSS
high-water marks cannot fairly isolate sequential variants. Logical dense score
bytes and input bytes are reported separately from measured allocation peaks.

Sweep `--chunk-size` at fixed shapes to see the storage/launch tradeoff. Very
small tiles can make execution substantially slower. Some consumers' tile
workspace can even exceed a compiled dense formula's peak, so a logical
streamed source alone does not guarantee lower measured allocation.

## Results and Provenance

The console summary goes to stderr; stdout is JSON. `--output` also writes the
same JSON to a chosen file. Normal failures preserve per-path error messages and
produce a nonzero exit code. The record includes validation tolerances, software
versions, GPU model/capability when applicable, extension SHA256/size, Git commit
and whether the checkout is dirty. A commit hash alone does not identify an
uncommitted source tree; retain its source snapshot when publishing a result.
Environment metadata omits hostname, username and import paths. Exception text
is retained for diagnosis and may contain local paths; inspect failed-run files
before sharing them. There is no telemetry or upload.

These are correctness-checked performance experiments, not a new unit-test grid
or release validation. Keep disposable outputs in ignored `develop/`.

## Complete Kernel Learning Workloads

`examples/benchmark_ml.py` measures source construction, model fitting and held-out
prediction for ridge regression, landmark regression, PCA and LDA. Rectangular SVD
measures fitting its leading factors. These comparisons use dense and streamed
RBF kernels with identical data and settings, avoiding a broadcast `[N,M,D]`
baseline. Choose sizes that fit the dense reference.

```sh
python examples/benchmark_ml.py --case pca --device cuda \
    --n 2048 --d 8 --rank 2 --subspace-size 10 --chunk-size 128 \
    --repeats 3 --output develop/pca-memory.json
python examples/benchmark_ml.py --case ridge --device cuda \
    --n 2048 --d 8 --dense-direct --output develop/ridge-memory.json
python examples/benchmark_ml.py --case landmark --device cuda \
    --n 1024 --m 128 --backward --dense-direct \
    --output develop/landmark-backward.json
```

The default `dense-iterative` baseline stores the kernel but uses the same native
solver, initialization and stopping controls as `streamed`. Stored sources now
use direct ATen products inside the native solver; generated sources evaluate
tiles. This compares retaining versus regenerating kernel entries. Optional
`--dense-direct` separately measures a direct solve, augmented QR least squares,
full symmetric eigendecomposition or full SVD. It solves the same mathematical
problem using a different numerical algorithm. Use this optional baseline when
comparing against conventional dense direct fitting, rather than interpreting
iterative timings as direct dense factorization performance. LDA retains the example's
coefficient-space regularization. No external-library performance ranking is made.

GPU peaks include source construction, solver work, output blocks, held-out
products and optional backward. All variants retain the same live inputs and
common initial/class blocks. Outputs from validation and previous repetitions
are released before measuring another variant. Reported memory is additional
`torch.cuda.max_memory_allocated` above the live baseline, excluding reserved
allocator cache and allocations owned directly by CUDA libraries or the driver.
It is not total device memory or process memory. CPU tensor peaks remain null.
The logical matrix size and common input/auxiliary bytes are reported separately.

Every path must pass convergence and numerical comparisons before timing.
Spectral checks compare values and subspaces, avoiding arbitrary vector signs;
held-out maps must transform consistently. `--backward` applies to ridge and
landmark fits and includes the mean squared prediction objective plus gradients
of every floating input. It uses implicit streaming solves and the corresponding
dense computation, with no dense matrix cotangent in the streamed source path.
Damping is positive in landmark fits. Failures/OOMs are JSON errors and nonzero
exits, never reported as successful performance results.

Warmup precedes synchronized timings, method order is shuffled, and the harness
reports all timing/peak samples, diagnostics, configuration, source checksum,
PyTorch/CUDA/GPU information and extension identity. Repeated source passes can
make streaming slower; report both time and memory rather than savings alone.


## Measured Kernel Fit Memory

One CUDA measurement on a Quadro RTX 6000, PyTorch 2.12.1+cu126, float64.
Three synchronized timing repeats after warmup; maximum extra peak tensor
allocation above the pre-call live allocation baseline, with Python garbage
collection before each sample. Warmed persistent workspaces are included in the
baseline, so these figures are additional allocations, not total device usage. RBF gamma=1/8, feature width 8, chunk 128,
query count 128, RHS width 2, regularization 0.5, rtol=1e-7, atol=1e-9.
PCA/SVD retain rank 2 with search width 10; LDA uses four classes. All cases
passed numerical checks and convergence before timing.

| Workload | Logical Kernel | Dense MiB | Streamed MiB | Dense Seconds | Streamed Seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| Ridge | `2048 x 2048` | 34.20 | 0.45 | 10.461 | 10.390 |
| Landmark | `2048 x 256` | 4.57 | 0.35 | 4.361 | 3.947 |
| PCA | `2048 x 2048` | 34.30 | 2.28 | 3.942 | 2.939 |
| LDA | `1024 x 1024` | 9.26 | 0.60 | 16.016 | 16.365 |
| SVD | `2048 x 1024` | 18.29 | 2.28 | 2.276 | 1.828 |
| Ridge + backward | `1024 x 1024` | 40.23 | 0.99 | 5.056 | 5.341 |
| Landmark + backward | `1024 x 128` | 5.11 | 0.93 | 2.461 | 2.928 |

These recorded measurements predate the stored-matrix product optimization.
They used the tiled matrix consumer; rerunning the current harness uses direct
ATen products for stored matrices and can give different timings.

Here dense means a stored RBF matrix with the same native iterative solver,
start, stopping rules and tiled matrix consumer as streaming. These timings
isolate storing versus recomputing entries; they do not represent a direct
dense BLAS fit. All fits except SVD include held-out prediction. Backward rows
also include mean squared predictions and every floating input gradient.

For a conventional direct dense comparison on the same 2,048-sample ridge
problem, the separate run measured:

| Ridge Variant | Extra Peak MiB | Median Seconds |
| --- | ---: | ---: |
| Dense direct solve | 96.12 | 0.051 |
| Stored matrix, matched iterative solver | 34.20 | 6.883 |
| Streamed iterative solver | 0.45 | 7.468 |

The direct solver uses a different algorithm and is faster in this experiment;
streaming saves matrix storage through repeated source evaluation. These are
workload observations on a shared machine, with no general speed guarantee.
CUDA-library/driver allocations and reserved cache are excluded; CPU peak is
unmeasured. Narrower problems, wider search spaces or other hardware can change
the balance. [Raw samples, validation errors and checksums](benchmarks/ml-memory.json)
provide the exact configurations and provenance.
