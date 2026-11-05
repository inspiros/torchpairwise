<div class="eyebrow">Integration</div>

# Benchmarking

Compare equivalent formulas, output shapes, dtypes, gradient work and hardware.
Streaming trades intermediate storage for tiled execution and sometimes replay;
it is not a universal throughput improvement.

The public harness validates values and first gradients before timing. Cases
cover distance sums, RBF products, distance quantiles, kernel regression and
scaled dot-product attention. Variants include eager
and fullgraph-compiled PyTorch, native dense named functions, streamed `tp.lazy`
reductions and products, and event streaming where equivalent.

```sh
python examples/benchmark_operators.py --case rbf-matmul --device cuda   --n 2048 --m 4096 --d 32 --channels 16 --chunk-size 128   --backward --output develop/rbf-matmul.json
```

Choose sizes that fit the dense reference. Warmup and first-call compilation
are separated from steady state. CUDA wall times are synchronized, path order
is shuffled, and extra peak allocated bytes are measured above a live baseline.
CPU tensor peak is not measured. Error paths record failure, not timing claims.

The eager direct PyTorch baseline may allocate N×M×D; compiled PyTorch can fuse
that intermediate. This does not cover every optimized torch.cdist/norm-matmul
formulation. Results include configuration, extension identity and environment
metadata. Inspect exception paths before sharing logs.

The [ML examples](https://github.com/inspiros/torchpairwise/blob/master/examples/streaming_ml.py)
show k-NN regression, normalized RBF smoothing and attention with dense value
and first-gradient checks. Benchmark `--case kernel-regression` or
`--case attention` for end-to-end comparisons. Attention uses a dense matmul
reference, without a broadcast feature intermediate; specialized PyTorch
attention backends may already save score storage and run faster.

Read the [benchmark methodology](https://github.com/inspiros/torchpairwise/blob/master/docs/benchmarking.md)
and [harness source](https://github.com/inspiros/torchpairwise/blob/master/examples/benchmark_operators.py).


## Complete Kernel Learning Workloads

`examples/benchmark_ml.py` compares complete ridge, landmark regression, PCA,
LDA and rectangular SVD fits. Dense and streamed paths use the same inputs,
initialization and numerical settings. Default `dense-iterative` stores the
kernel and runs the same iterative method with direct ATen stored-matrix products;
generated sources evaluate tiles. Its timing does not represent a direct dense
factorization; use the optional direct baseline for that. Optional `dense-direct` uses a direct solve, augmented QR,
full eigh or full SVD, labeled as a different numerical algorithm.

```sh
python examples/benchmark_ml.py --case ridge --device cuda --n 2048 \
    --dense-direct --output develop/ridge-memory.json
python examples/benchmark_ml.py --case landmark --device cuda --n 1024 --m 128 \
    --backward --dense-direct --output develop/landmark-backward.json
```

The timed scope includes source construction, fitting and held-out prediction;
SVD fits leading factors. `--backward` adds a mean squared prediction objective
and input VJPs for ridge/landmark. GPU memory is extra peak allocated bytes above
live inputs/common auxiliary blocks. Validation/previous outputs are released
before timing another variant; reserved cache and external CUDA allocations
are excluded. CPU tensor peaks are null. JSON records samples, diagnostics,
numerical errors, parameters, environment and harness/extension checksums.

Convergence and value/gradient or spectral-subspace checks must pass before
measurement. A direct decomposition can be faster while using more memory;
a small problem or a wide subspace can favor dense storage. Failures/OOMs are
errors with nonzero exits, never performance results. See
[full methodology](https://github.com/inspiros/torchpairwise/blob/master/docs/benchmarking.md#complete-kernel-learning-workloads).


### Measured Kernel Fit Memory

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
the balance. [Raw samples, validation errors and checksums](https://github.com/inspiros/torchpairwise/blob/master/docs/benchmarks/ml-memory.json)
provide the exact configurations and provenance.
