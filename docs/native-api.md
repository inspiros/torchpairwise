# Native C++ Comparisons and Deployment

The native API uses the same ATen algorithms as the Python bindings. CPU and
CUDA execution comes from PyTorch operations; a separate CUDA kernel is not
required for every metric. Include `torchpairwise.h` and link both LibTorch and
the torchpairwise shared library built against the same PyTorch minor and C++
ABI. The vector layer covers the following additions:

| Family | Native named functions |
| --- | --- |
| Vector | angular, Lorentzian, Clark, Soergel and Ruzicka distances |
| Probability/compositional | KL, symmetric KL, Bhattacharyya distance/coefficient, Aitchison, total variation, Fisher-Rao, Rényi, Pearson/Neyman chi-squared, triangular discrimination and Itakura-Saito |
| Histograms/missing data | Hellinger, histogram intersection, shared-grid 1D Wasserstein and NaN Euclidean |
| Radial kernels | rational quadratic and Matérn |

Every listed function has an all-pairs native entry point and a corresponding
`paired_` entry point. Symmetric functions also have a `condensed_` entry point.
Names match the Python named functions, including `_distances`, `_divergences`
or `_kernel` suffixes. The header is `csrc/ops/vector_comparisons.h`.
Forward and first backward execute in C++; importing the Python package is not
required by a standalone consumer. Native backward replay retains original
operands and recomputes bounded tiles rather than saving a pair-feature matrix.
Higher derivatives of these replay operators are unsupported.

```cpp
#include "torchpairwise.h"

// X: [...,N,D], Y: [...,M,D]; only leading batch axes broadcast.
auto distances = torchpairwise::ops::kl_divergences(X, Y);
auto aligned = torchpairwise::ops::paired_kl_divergences(X, aligned_Y);
auto unordered = torchpairwise::ops::condensed_symmetric_kl_divergences(X);
auto kernel = torchpairwise::ops::matern_kernel(X, Y, 0.7, 1.5);
```

The [standalone example](../examples/libtorch/vector_comparisons.cpp) checks
values, first gradients, pairing modes, and loading a scripted native call.
Deployment must arrange the shared library's PyTorch/CUDA dependency paths;
using a wheel does not remove those runtime dependencies. The standalone
[CMake package](cmake.md) installs the library, headers and
`torchpairwise::torchpairwise` target without a Python runtime requirement.

The linear-map overload `A.centered(reference_mean)` applies training-based
centering to rectangular held-out kernels, using an ATen tensor `[...,N]` of
training column means. Both forward and transposed actions preserve first
gradients without forming a matrix; see [centering contracts](linear-operators.md#centering-and-diagonal-shifts).

## Building on torchpairwise from C++

A Python package needs only `import torchpairwise`. A package with its own C++
extension can reach the native code in three ways:

| Route | Build needs | Tied to | Use for |
| --- | --- | --- | --- |
| Dispatcher calls by schema | PyTorch headers only | operator names and schemas | redistributable extensions (recommended) |
| Native headers from the wheel | `get_include()`, `library_path()`, C++20 | this exact torchpairwise and PyTorch build | source builds on Linux that need the C++ API, such as streaming or solvers |
| [Standalone CMake package](cmake.md) | LibTorch and CMake | the installed build | C++ applications without Python |

The [downstream example](../examples/downstream/) builds one extension of each
of the first two kinds with `torch.utils.cpp_extension.load`, then checks
values and gradients against Python.

**Dispatcher calls.** Public operators are registered as `torchpairwise::<name>`;
`torch.ops.torchpairwise.<name>.default._schema` prints a schema. Look the
operator up on first use, so the extension may load before torchpairwise:

```cpp
#include <ATen/core/dispatch/Dispatcher.h>

// torchpairwise::rbf_kernel(Tensor x1, Tensor? x2=None, float? gamma=None) -> Tensor
at::Tensor rbf_kernel(const at::Tensor& x, const at::Tensor& y, double gamma) {
  static const auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::rbf_kernel", "")
          .typed<at::Tensor(const at::Tensor&, const c10::optional<at::Tensor>&,
                            c10::optional<double>)>();
  return op.call(x, y, gamma);
}
```

Nothing links against torchpairwise, so load it before the first call:
`import torchpairwise` in Python, or `torch::ops::load_library` in C++. Calls take
the same autograd, CUDA and override paths as Python calls. The typed signature
must match the schema, otherwise the lookup throws. The supported contract is
the public operator names and schemas. Operators whose names start with an
underscore are private and may change in any release; a change to a public
schema is a breaking change and is listed in the [migration notes](migration.md).
PyTorch's stable ABI makes the same call by name (`torch_call_dispatcher`) for
extensions that must not depend on the PyTorch minor; that route is untested here.

**Native headers from the wheel.** `torchpairwise.get_include()` returns the
header directory and `torchpairwise.library_path()` the compiled `_C` library.
Include `<torchpairwise.h>`, which includes every public header. Individual
headers are `ops/<name>.h` relative to the same directory, in the wheel and in
the CMake install. With setuptools:

```python
import torchpairwise as tp
from torch.utils.cpp_extension import BuildExtension, CppExtension

extension = CppExtension(
    'mypkg._C', ['mypkg/csrc/ops.cpp'], include_dirs=[tp.get_include()],
    extra_compile_args=['-std=c++20'], extra_link_args=[tp.library_path()])
```

- Build against the PyTorch that torchpairwise was built for: same minor
  version, C++ ABI and CUDA runtime.
- The linker records the absolute path of `_C`, so the result is valid only for
  that installation. Build with `pip install --no-build-isolation` so the path is
  the runtime one, rebuild after upgrading or reinstalling torchpairwise, and do
  not ship the result as a binary wheel; use dispatcher calls for that.
- Windows wheels ship no import library, and macOS linking is untested; use
  dispatcher calls on those platforms.
- CUDA headers (`*.cuh`) are implementation details; there is no device-side API.

## Streaming Namespace

`torchpairwise::streaming` is available from `<torchpairwise.h>` and the
self-contained `<streaming.h>`. These header facades call the existing typed
native wrappers; they add no registrations and preserve serialized schemas.
Python uses `tp.lazy` and `tp.as_lazy` for the same source/consumer model.

```cpp
#include <torchpairwise.h>
namespace s = torchpairwise::streaming;

s::PairwiseOptions options;
options.chunk_size = 128;
options.comparison.gamma = 0.2;
auto K = s::kernel(X, Y, "rbf", options);
auto product = K.matmul(V);                  // [...,N,C]
auto spread = K.std(-1, 0.);                 // [...,N]
auto neighbors = K.topk(3);                  // values and indices
```

Short C++ scalar factories return the existing `ScalarExpression`, exposing its
full common consumer/algebra interface. Specialized `s::KernelOperator` remains
available for vector `displacement` and `residual_outer_product`, as in the
existing native API. `s::kernel` has vector and structured input overloads;
`s::pairwise` has typed structured, image/audio and sequence/token/MMD overloads.
Inputs and numerical options retain the family contracts. Vector `pairwise`
accepts distances and kernels, resolving canonical names and shared aliases
from the same native catalog as Python. The `kernel` factory restricts role to
kernels, so aliases such as `bhattacharyya` have the same meaning as in Python.

| Short factory | Native options / result |
| --- | --- |
| `s::pairwise(X,Y,metric,options)` | `PairwiseOptions` / `ScalarExpression` |
| `s::pairwise(tensors,other,metric,options)` | `StructuredOperationOptions` / `ScalarExpression` |
| `s::pairwise(X,Y,metric,comparison,options)` | Typed image/audio/ordered options plus `PairwiseOptions` / `ScalarExpression` |
| `s::kernel(X,Y,name,options)` | `PairwiseOptions` / `ScalarExpression` |
| `s::kernel(tensors,other,name,options)` | `StructuredOperationOptions` / `ScalarExpression` |
| `s::matrix(values,options)` | `MatrixOptions` / `ScalarExpression` |
| `s::residual(X,Y,transform,chunk,feature_chunk)` | `ResidualOperator` |
| `s::image_map(X,Y,comparison,options)` | `ImageMapOptions` / `ImageMapOperator` |

The namespace also exposes existing typed family factory names, source classes
and free consumers. Replacing `torchpairwise::ops` with
`torchpairwise::streaming` works for those re-exported APIs without changing
their signatures. The `ops` namespace remains the native implementation and
dispatcher layer. Python uses `tp.lazy` with custom and encoded sources.

`s::linalg::solve(A, rhs, SolveOptions)` provides a common entry for square
systems. Set `SolveOptions.method` explicitly to `"cg"`, `"minres"` or `"gmres"`.
It returns `SolveResult` with `solution`, `residual_norm`, `iterations`,
`converged` and `status`; status meanings remain method-specific. Common options
are `rtol`, `atol`, `max_iter`, `initial`. `preconditioner` is CG-only; optional
`restart` is GMRES-only. Unsupported options raise. No method detection/fallback.
`s::linalg::lstsq(A, rhs, LstsqOptions)` defaults to method `"lsqr"` and returns
`LSQRResult`, including the normal residual. Neither computes a dense inverse,
pseudoinverse, rank or singular values. They reuse existing native solver
registrations and compile/export behavior. See [solver contracts](linear-solves.md).

The inference-only native solver is `s::linalg::cg(A, rhs, CGOptions)`.
It accepts a `LinearOperator` and returns `CGResult` with Tensor fields
`solution`, `residual_norm`, `iterations`, `converged`, and `status`.
`CGOptions` holds `rtol`, `atol`, `max_iter`, optional `initial`, and optional
`preconditioner` (LinearOperator). The latter applies a fixed SPD approximation
to the inverse, not the matrix to invert. `s::diagonal(d)` stores only `[...,N]`
entries; `s::linalg::jacobi(d)` supplies the checked reciprocal diagonal action.
`s::linalg::lowrank_inverse(F, shift)` prepares the inverse of `F F.T + shift I`
from stored features; `s::linalg::nystrom_inverse(C, W, shift, jitter)` prepares
it from explicit cross/landmark kernel blocks. Both use inference-only native
setup with a frozen double spectral state, support RHS gradients after setup,
and never form an N-by-N inverse. See [preconditioning](preconditioning.md).
Use an explicit `at::NoGradGuard` when inputs require gradients. See
[CG shapes, convergence and status codes](linear-operators.md#conjugate-gradient).

For rectangular least squares use `s::linalg::lsqr(A, rhs, LSQROptions)`.
`LSQROptions` has `damp`, `rtol`, `atol`, `max_iter`, and optional `initial`.
`LSQRResult` returns `solution`, `residual_norm`, `normal_residual_norm`,
`iterations`, `converged`, and `status`. All numerical iteration is native;
inference-only and compile/export contracts match the Python API. See
[least-squares shapes and convergence](least-squares.md) and the
[landmark regression example](../examples/landmark_regression.py).

For real nonsymmetric square systems use
`s::linalg::gmres(A, rhs, GMRESOptions)`. Options contain `rtol`, `atol`, positive
`restart`, total-step `max_iter`, and optional `initial`. `GMRESResult` exposes
`solution`, `residual_norm`, `iterations`, `converged`, and `status`. Inference
and compile/export contracts match Python. See [GMRES contracts](gmres.md) and
the [PageRank example](../examples/pagerank.py).

`s::block({{A,B},{C,D}})` builds compact block maps; `hstack`/`vstack` expose
horizontal/vertical concatenation. Every C++ entry is a LinearOperator, adapted
with `s::aslinearoperator` where needed. Batch axes broadcast; matrix row/column
partitions agree. See [block-map contracts](block-operators.md).

For symmetric indefinite systems use `s::linalg::minres(A, rhs, MINRESOptions)`.
Options are `rtol`, `atol`, total-step `max_iter` and optional `initial`.
`MINRESResult` exposes `solution`, `residual_norm`, `iterations`, `converged` and
`status`. It uses short recurrence workspace and assumes symmetry. See
[MINRES contracts](minres.md) and the
[kernel intercept example](../examples/kernel_ridge_intercept.py).

`s::group_projection(labels, X)` constructs a `LinearOperator` whose actions
replace sample values by their group mean. Int64 labels `[...,N]` may contain
arbitrary signed IDs; X supplies dtype/device only. Partition batches broadcast
with RHS blocks, and the prepared partition snapshots labels. Both products and
native CG graphs reuse the IDs/counts. See [group projection](linear-operators.md#group-projection).

`s::linalg::eigh(A,EighOptions)` computes leading eigenpairs of a symmetric PSD
map. Set `k`, optional `subspace_size`, `max_iter`, `rtol`, `atol`, and optional
`initial`. `EigenResult` exposes `eigenvalues`, `eigenvectors`, `residual_norm`,
`iterations`, `converged`, `status`. Execution is inference-only; use explicit
no-grad when source tensors require gradients. See [eigenpair contracts](linear-operators.md#leading-eigenpairs).

`s::optimize::box_qp(Q,q,BoxQPOptions)` minimizes a convex quadratic with required
finite `lower`/`upper` Tensor bounds, optional `QPEquality{vector,target}`, optional
`initial`, `rtol`, `atol`, and `max_iter`. `QPResult` has `solution`, `objective`,
`iterations`, `converged`, `status`, `optimality`, `feasibility`. Execution is
inference-only; see [optimization shapes and diagnostics](quadratic-optimization.md).

`ops::sinkhorn(C,a,b,SinkhornOptions)` accepts a scalar `ScalarExpression`
cost and returns detached potentials, a differentiable objective and per-batch
diagnostics in `SinkhornResult`. `ops::sinkhorn_plan(C,result,a,b,epsilon)`
returns a streamed scalar expression. See [entropic transport](entropic-transport.md)
for masses, objectives and convergence.

## Native Source and Dispatch Conventions

The [linear-map interface](linear-operators.md#c--libtorch) adapts native
ScalarExpression sources or dense tensors with `streaming::aslinearoperator`.
Its finite graph composes source products and shared native linear primitives;
the latter use composite ATen autograd. C++ execution has no Python graph or
callback requirement. Graph composition does not materialize a matrix product.

Public C++ operator functions use cached typed PyTorch dispatcher handles.
Their registered implementations are separate functions. This gives direct
C++ calls the same dispatch boundary as `torch.ops` calls, including backend
overrides; family options structs flatten into typed schema parameters there.
C++ convenience functions without their own schema compose these wrappers.

Schemas and numerical forward/backward implementations live in `csrc/ops`.
Autograd Functions and Autograd-key registrations live in `ops/autograd`;
specialized device kernels and backend registrations live in `ops/cpu` and
`ops/cuda`. Related comparisons share family files and internal helper headers.
Legacy functors have been replaced by exported free functions. The root
`.clang-format` applies the same formatting to old and new native sources.

The native canonical-name catalog is shared by complete comparison selection
and the legacy `cdist`/`pdist`/kernel selectors. Legacy signatures and accepted
names/options remain available; they no longer maintain a separate functor
routing layer or a separate condensed algorithm.

The catalog uses static descriptors and typed function pointers for common
signatures, with explicit adapters for parameterized metrics and family APIs.
Selectors call public typed wrappers; named functions and family functions each
retain their own override boundary. An override of a named vector function
affects that named call, while overriding `vector_comparisons` affects family
selection. This avoids imposing one oversized callable signature on every metric.

## Parameters Across Languages

Use a named function when the metric is known. Its signature contains only its
own typed parameters. A generic dispatcher needs the union of relevant options;
we retain explicit optional arguments in its operator schema. Supplying an
option that the selected metric does not use raises an error, even if that
option equals another metric's default. Omission is represented by `None` /
`c10::nullopt`. Parameters are not silently ignored.

C++ callers can use the ordinary `VectorComparisonOptions` struct to avoid a
long positional argument list. The struct is a C++ convenience; it is not a
TorchScript custom class and does not add a Python runtime dependency:

```cpp
torchpairwise::ops::VectorComparisonOptions options;
options.length_scale = 0.7;
options.alpha = 0.8;
auto result = torchpairwise::ops::vector_comparisons(
    X, Y, "rational_quadratic", "all_pairs", options);
```

Python can use the typed dispatcher directly. Generic parameters are
`base`, `length_scale`, `alpha`, `nu`, `squared`, and `bin_positions` for this
migrated subset; arbitrary Python dictionaries are not part of its schema:

```python
result = torch.ops.torchpairwise.vector_comparisons(
    X, Y, "rational_quadratic", "all_pairs",
    length_scale=0.7, alpha=0.8,
)
```

Migrated named Python functions have a direct native call when scripted. Their
usual eager adapters retain the established Python validation errors. All-pairs,
paired and condensed shapes, numerical conventions and scalar domains remain
those documented in each metric's guide. Native invalid inputs raise a
synchronous `c10::Error` on CPU and CUDA; value checks cost one host
synchronization per call (tiled kernels check once per call, not per tile).
Inside tracing (Fake/meta tensors) and CUDA graph capture they remain
asynchronous tensor assertions, and a failed CUDA device assertion leaves the
CUDA context unusable. Floating comparison schemas register a native autocast
policy (float16/bfloat16 operands run in float32), identical in Python and
LibTorch. There is no general
second-derivative contract for the native replay layer.

## Weighted Comparisons

The shared weighted entry point covers cosine, correlation, Euclidean, squared
Euclidean, Manhattan, Minkowski, Chebyshev, Canberra, Bray-Curtis, Hamming,
Jaccard, Kulsinski, Kulczynski1, Yule, Dice, Rogers-Tanimoto, Russell-Rao,
Sokal-Michener and Sokal-Sneath. The legacy `wminkowski` name and vector aliases
are accepted. All three pairing modes execute in C++, including first backward
for inputs and feature weights where defined. Boolean/Hamming inputs have no
input derivative; weights can differentiate. Boolean outputs follow the weights'
floating dtype. The original unweighted operator schemas remain available.

```cpp
// Header: csrc/ops/weighted_comparisons.h (also included by torchpairwise.h).
// X: [...,N,D], Y: [...,M,D], W: [...,D]; batch axes broadcast together.
auto distance = torchpairwise::ops::weighted_comparisons(X, Y, W, "cosine");
auto aligned = torchpairwise::ops::weighted_comparisons(
    X, aligned_Y, W, "minkowski", "paired", 1.7);
auto unordered = torchpairwise::ops::weighted_comparisons(
    X, c10::nullopt, W, "sqeuclidean", "condensed");
```

Python users keep the existing named/generic API, such as
`cosine_distances(X, Y, w=W)` or `paired_distances(X, Y, metric="cosine", w=W)`.
The native schema is also directly callable:

```python
result = torch.ops.torchpairwise.weighted_comparisons(
    X, Y, W, "minkowski", "all_pairs", p=1.7,
)
```

`p` is accepted only for Minkowski (including aliases); other metrics reject an
explicit value. Zero weights ignore their coordinates before arithmetic. Ratio
metrics require positive total weight. See the [weighting guide](weighted-distances.md)
for boundary conventions. Native backward retains operands. Formulas that construct pair-feature values
replay bounded tiles; matrix products and existing native reductions retain
their full-size optimized operations. Higher derivatives are unsupported. The [weighted C++ example](../examples/libtorch/weighted_comparisons.cpp)
checks CPU/CUDA values, input/weight gradients, modes and option validation
without a Python interpreter. The [Python example](../examples/compiled_weighted.py)
compiles a public function and saves/reloads an export with changing sample counts.

## Nearest-Point Set Comparisons

Chamfer and directed/symmetric Hausdorff now share native layout normalization,
batch broadcasting, valid prefix lengths, pairing modes and first backward.
Inputs are dense `[...,N,P,C]` tensors; `batched` and `feature_dim` also support
spatial layouts such as `[B,N,C,H,W]`. This does not require sparse tensors. This shared API uses float32/float64
outside autocast and the usual float32 promotion inside autocast.

```cpp
torchpairwise::ops::PointsetComparisonOptions options;
options.batched = true;
options.feature_dim = 2;  // X: [B,N,C,H,W].
options.x1_lengths = lengths_X;
options.x2_lengths = lengths_Y;
options.chunk_size = 8;
options.squared = true;
auto distances = torchpairwise::ops::pointset_comparisons(
    X, Y, "chamfer", "all_pairs", options);
options.x2_lengths = corresponding_lengths_Y;
auto paired = torchpairwise::ops::pointset_comparisons(
    X, corresponding_Y, "chamfer", "paired", options);
```

The header is `csrc/ops/pointset_comparisons.h`, included by `torchpairwise.h`.
`pointset_comparisons` returns values for all three metrics; the separate
`directed_pointset_comparisons` returns `(values, source_indices, target_indices)`.
Only symmetric metrics accept condensed mode, with one input and `x1_lengths`.
`squared` is a Chamfer option; explicitly supplying it to Hausdorff is rejected.
Shuffling/generators are Hausdorff options and cannot accompany lengths or chunks.
The existing legacy dispatcher schemas remain available.

```python
# Existing Python APIs remain unchanged.
d = torchpairwise.chamfer_distances(X, Y, x1_lengths=lengths_X)
# Direct typed dispatcher, also available without the Python adapters:
d = torch.ops.torchpairwise.pointset_comparisons(
    X, Y, "chamfer", "all_pairs", x1_lengths=lengths_X,
)
```

Deterministic execution retains operands and replays bounded cloud tiles in
first backward. Nearest-point backward gathers only selected witnesses in point
tiles and executes in C++; it has no Python autograd dependency. Full-cloud
Hausdorff preserves the optimized early-exit forward search. Shuffled search
retains its actual forward witnesses and does not replay the RNG in backward.
Shuffled/generator execution remains outside the compile/export contract.
Native replay supports first derivatives; higher derivatives are unsupported.

The [standalone C++ example](../examples/libtorch/pointset_comparisons.cpp)
checks layouts, ragged values, gradients, modes and witnesses on CPU/CUDA.
The [compile/export example](../examples/compiled_pointsets.py) changes both
cloud counts and point counts after export. These guarantees cover the three
nearest-point metrics. The remaining cloud comparisons use the transport layer below.

## Complete Built-in Families

Built-in forward and first backward execute in C++ for all implemented families.
Python adapters preserve public validation and names. The umbrella header exposes
these typed C++ front ends:

| C++ selector | Coverage |
| --- | --- |
| `builtin_comparisons` | All registry vector/cloud distances and vector kernels, all supported modes |
| `transport_comparisons` | Energy, named-kernel MMD, sliced/exact Wasserstein and Sinkhorn |
| `mmd_comparisons` | Named-kernel MMD directly, using `TransportComparisonOptions` |
| `mmd_statistics` | Signed biased/unbiased MMD estimates, separate from distance dispatch |
| `gaussian_comparisons` | Gaussian Wasserstein |
| `covariance_comparisons` | Log-Euclidean, affine-invariant and Stein SPD comparisons |
| `mixture_comparisons` | Gaussian-mixture overlap, L2, Cauchy-Schwarz, exact/entropic transport and Sinkhorn |
| `image_comparisons` | PSNR, SSIM and multiscale SSIM |
| `audio_comparisons` | SI-SDR, signal SNR and spectral comparisons |
| `sequence_comparisons` | DTW and Soft-DTW divergence |
| `token_comparisons` | Levenshtein and indel |
| `streaming_reductions` | Registry neighbors, min/max/sum/prod/all/any and kernel payload sums |
| `kernel_operations` | Streamed matrix products, bilinear/quadratic forms, scalar/row/column reductions, diagonals and selected blocks |
| `builtin_neighbor_mask` | Dense k/radius masks over the vector/cloud registry |

Each family has typed parameters or an options struct; unsupported/unused
parameters are errors. Family headers document signatures and mode restrictions.
The [LibTorch examples](../examples/libtorch/) exercise each family without
initializing Python. Image/audio feature transforms are reused before batch
broadcasting. Native replay bounds retained comparison state; structured
comparisons retain their existing training-memory behavior.

```cpp
torchpairwise::ops::BuiltinComparisonOptions options;
options.p = 2.;
auto distances = torchpairwise::ops::builtin_comparisons(
    clouds, references, "wasserstein", "all_pairs", false, options);

torchpairwise::ops::StreamingReductionOptions reduction;
reduction.chunk_size = 128;
reduction.k = 4;
auto [values, indices] = torchpairwise::ops::streaming_reductions(
    X, Y, "euclidean", "neighbors", false, reduction);
```

Python's [unified selector](dispatchers.md) accepts all comparison families;
`torch.ops.torchpairwise.builtin_comparisons` exposes the full vector/cloud
selector with optional vector/cloud parameters. Native `cdist`/`pdist` schemas
now expose the broad role-aware catalog and expanded options; the family selector
remains available for direct vector/cloud use.

Compile/export capture native opaque operators and their registered metadata
kernels. They preserve native first backward and dynamic sample/batch shapes
where supported. Exact transport still runs a CPU-controlled native solver and
synchronizes CUDA costs; compilation does not make the solver a GPU algorithm.
Shuffled Hausdorff/generator execution remains outside this compiler contract.
Arbitrary Python predicates, user metrics and encoder factories retain their
separate deployment requirements. A Python callback is not made portable merely
by passing it to a wrapper.

TorchScript is [deprecated by PyTorch](https://docs.pytorch.org/tutorials/advanced/torch_script_custom_ops.html).
We retain compatible native schemas for existing users and test `torch.compile`
and `torch.export` independently. For serialized deployment, prefer an enclosing module whose forward binds the
metric options. Saving a bare ScriptFunction as a module can lose its Python
default-argument metadata, requiring explicit arguments when reloaded.

A scriptable operator does not automatically
become a fused CUDA kernel, and exporting a program does not automatically give
it an ONNX implementation or a universal standalone runner.

### Named-Kernel MMD from C++

```cpp
torchpairwise::ops::TransportComparisonOptions options;
options.kernel = "polynomial";
options.degree = 3;
options.gamma = .5;
options.coef0 = 1.;
options.squared = true;
auto statistic = torchpairwise::ops::mmd_comparisons(X, Y, "all_pairs", options);
```

`X` and `Y` contain collections of point clouds. The same options work through
`transport_comparisons(..., "mmd", ...)` and `BuiltinComparisonOptions` with
`builtin_comparisons`. The new native MMD forward/backward schemas have metadata
registrations for compile/export. Existing transport schemas and their defaults
remain registered for serialized graphs. All numerical forward and first backward
work executes in C++; Python is not needed by a standalone LibTorch program.
See [distribution comparisons](distribution-comparisons.md) for kernel domains,
Laplacian compatibility and the biased-statistic convention.

### Signed MMD Statistics from C++

```cpp
torchpairwise::ops::TransportComparisonOptions options;
options.kernel = "polynomial";
options.degree = 3;
options.gamma = 1. / X.size(-1);
options.coef0 = 1.;
auto statistic = torchpairwise::ops::mmd_statistics(
    X, Y, "all_pairs", "unbiased", options);
```

Do not set `squared` for statistics: they return the signed squared estimate
directly. Select `"paired"` or `"condensed"` for the other output modes; condensed
uses `c10::nullopt` for the second input. `"unbiased"` requires at least two valid
points per cloud and removes within-cloud diagonals. `"biased"` includes them.
The native schema has the same name and kernel/estimator options, plus the
layout/length selectors. The signed API uses overloads of the existing native
MMD forward/backward operators, preserving all prior serialized schemas.

## Streamed Interaction Maps

`KernelOperator::displacement(bool normalize=false)` streams a kernel-weighted
coordinate displacement. `sinkhorn_map(x, y, SinkhornMapOptions)` computes a
finite-iteration conditional balanced entropic transport map; options include
source/target masses, p, regularization, iteration/tile counts and displacement.
See [interaction contracts](interaction-maps.md) for zero-mass, memory and
first-derivative semantics. Both execute forward/backward without Python.

`image_maps` with `ImageMapOptions` returns dense maps or streams reference
reductions; see [image maps](image-maps.md). Kernel residual second moments
use `kernel_residual_outer_product` or the matching `KernelOperator` method;
see [kernel moments](kernel-moments.md) for centering and normalization.


## Streaming Operator Entry Points

Python and standalone LibTorch share native source preparation, numerical tiles
and first backward. For the full scalar consumer interface, construct a
`ScalarExpression` with a typed factory. Existing family facades keep their
original method signatures and remain available.

| Source | Header under `<ops/...>` | Full scalar expression entry |
| --- | --- | --- |
| Vector/cloud distances and vector kernels | `expression_operations.h` | `pairwise_expression(x,y,name,is_kernel,options)` |
| Covariance, Gaussian and mixture collections | `expression_operations.h` | `structured_expression(inputs,other_inputs,name,options)` |
| Image and audio scalar scores | `expression_operations.h` | Overloaded `media_expression(x,y,name,family_options,matrix_options)` |
| Sequence, token and MMD comparisons | `expression_operations.h` | Overloaded `ordered_expression(x,y,name,family_options,matrix_options)` |
| Caller-owned scalar matrices | `expression_operations.h` | `matrix_expression(scores,options)` |
| Feature-contracted residuals | `residual_operations.h` | `residual_operator(...).mean(-1)` and other feature methods |
| Fully contracted image events | `image_map_operations.h` | `image_map_operator(...).mean()` or `.events()...scalar()` |

`PairwiseOperationOptions` holds comparison parameters in `options.comparison`.
Its factory binding accepts comparison, chunk-size and transpose settings;
provide payloads, selection indices and reduction settings to the consumer.
For kernels, pass `true` explicitly to `pairwise_expression`'s kernel flag.

```cpp
#include <ops/expression_operations.h>
namespace ops = torchpairwise::ops;

ops::PairwiseOperationOptions options;
options.chunk_size = 64;
options.comparison.gamma = 0.2;
auto k = ops::pairwise_expression(x, y, "rbf", true, options);
auto product = k.matmul(values);
auto form = k.bilinear(left, right);
auto [lowest_values, lowest_indices] = k.topk(8, -1, false);
auto column_mean = k.reduce("mean", -2);
auto row_variance = k.var(-1, 0.);  // correction follows dim in C++
auto joint_variance = k.statistics(
    "var", at::IntArrayRef{-2,-1}, false, false, 0.);
```

### Common Consumers and Algebra

`ScalarExpression` exposes the same 25 numerical consumers as the Python scalar
interface through convenience methods and reducer selectors: `matmul`,
`softmax_matmul`, `bilinear`, `quadratic`, `frobenius_inner_product`, `hsic`, `diagonal`,
`block`, `topk`, seven basic reducers, four expanded statistics, three
normalized weighted statistics, and exact `median`/`quantile`. Basic reducers use `reduce("sum",dim,...)`,
`reduce("mean",dim,...)`, etc. `topk` returns a `std::tuple<Tensor,Tensor>`;
min/max reductions return values only. `transpose()` is a lazy logical view,
including for directed metrics. Exact selection takes one N/M axis and returns
values only: `median(dim=-1, keepdim=false, exclude_self=false)` and
`quantile(q, dim=-1, keepdim=false, interpolation="linear", exclude_self=false)`.
It retains complete rows in budgeted blocks; see [selection contracts](operator-statistics.md#exact-median-and-quantile).
`a.frobenius_inner_product(b, true, true)` computes centered alignment;
`a.hsic(b, true)` selects unbiased HSIC. See [coordinated statistics](kernel-alignment.md).

Expressions support `+`, `-`, entrywise `*`, unary negation, `scale`, `shift`,
`square`, `exp`, `weight_rows` and `weight_columns`:

```cpp
auto distance = ops::pairwise_expression(x, y, "sqeuclidean");
auto affinity = distance.scale(-0.5).exp();
auto smoothed = distance.scale(-0.5).softmax_matmul(values);
auto weighted = distance.weighted_var(affinity, -1, 1.);
auto combined = k.scale(0.7) + affinity;
auto robust = combined.median(-1);
auto threshold = distance.quantile(0.9, -1, false, "linear", false);
```

C++ arithmetic accepts scalar numbers directly (`distance + 1`, `distance / 2`);
`scale(...)` also accepts scalar tensors. Python additionally provides reflected
scalar arithmetic. Tensor factors must be zero-dimensional; row/column weighting
is explicit. No lazy operator matrix multiplication or
arbitrary ATen-method graph is implied.

Additional entrywise methods support scalar or expression operands:

| Operation | C++ methods |
| --- | --- |
| Arithmetic | `/`, `pow`, `minimum`, `maximum` |
| Scalar bounds | `clamp_min`, `clamp_max` (number or zero-dimensional tensor) |
| Unary transforms | `sqrt`, `log`, `sin`, `cos`, `tanh`, `sigmoid`, `abs` |
| Boolean expressions | `lt`, `le`, `gt`, `ge`, `eq`, `ne`, `logical_and`, `logical_or`, `logical_not` |
| Selection | `a.where(condition, other)`, `a.masked(mask, value=0)` |
| Axis expressions | `row_values`, `column_values`, `row_index`, `column_index`, `add_rows`, `add_columns` |

`where` keeps `a` where its Boolean condition is true. Axis expressions use the
receiver's matrix shape; value tensors have shape `[..., N]` or `[..., M]`.
Free functions `same_group_mask`, `different_group_mask`, `causal_mask`,
`band_mask` and `diagonal_mask` return Boolean expressions bound to that shape.
Group masks accept a second label tensor for rectangular matrices; causal masks
keep `j <= i + offset`, and band masks keep `-lower <= j-i <= upper`.

```cpp
auto a = ops::pairwise_expression(x, y, "sqeuclidean");
auto groups = ops::same_group_mask(a, labels);
auto off_diagonal = ops::diagonal_mask(a).logical_not();
auto mask = groups.logical_and(off_diagonal);
auto row_sums = a.masked(mask).reduce("sum", -1);
```

For a single axis, `statistics(reduction,dim,keepdim,exclude_self,correction)`
selects var/std/logsumexp/count_nonzero. `weighted_statistics(weights,reduction,
dim,keepdim,exclude_self,correction)` selects normalized mean/var/std with either
an axis confidence tensor or another ScalarExpression. Weighted convenience
var/std take `(weights,dim=-1,correction=0,keepdim=false,exclude_self=false)`;
unweighted var/std default to correction 1. Dimension-list overloads exist for
`reduce`, `statistics` and operator-weighted `weighted_statistics`; vector
confidence requires one selected axis. See [statistics](operator-statistics.md).

### Existing Family Facades

| Facade / free function | Header |
| --- | --- |
| `PairwiseOperator`, `pairwise_operations`, `PairwiseOperationOptions` | `pairwise_operations.h` |
| `KernelOperator`, `kernel_operations`, `KernelOperationOptions` | `kernel_operations.h` |
| `StructuredPairwiseOperator`, `structured_pairwise_operations`, `StructuredOperationOptions` | `structured_operations.h` |
| `ImagePairwiseOperator`, `AudioPairwiseOperator`, overloaded `media_pairwise_operations` | `media_operations.h` |
| `SequencePairwiseOperator`, `TokenPairwiseOperator`, `MMDPairwiseOperator`, overloaded `ordered_pairwise_operations` | `ordered_operations.h` |
| `MatrixOperator`, `matrix_operations`, `MatrixOperationOptions` | `matrix_operations.h` |

They retain products/forms, basic reductions and entry selectors from their
family API. Use the expression factories above for expanded statistics,
softmax products and top-k rather than assuming Python method parity on every
older C++ class. KernelOperator inherits PairwiseOperator and adds vector
`displacement` and `residual_outer_product`; mixture geometry uses its separate
structured input contract. `matrix_expression` starts from a dense caller-owned
allocation and does not remove that input's memory cost.

### Residual Event Sources

Residuals have shape `[...,N,M,D]` and sign X-Y. The integer feature-axis methods
return a ScalarExpression; explicit dimension-list overloads return terminal
Tensors. This distinction makes C++ return types static:

```cpp
#include <ops/residual_operations.h>
auto g = ops::residual_operator(x, y, "abs", 64, 128);
auto mae = g.mean(-1);                    // ScalarExpression
// Even a dimension list selecting only D is a terminal Tensor overload.
auto row_spread = g.var(std::vector<int64_t>{-2,-1}, 0.);
auto mean_features = g.mean(std::vector<int64_t>{-2});
auto [best, ids] = mae.topk(8, -1, false);
auto [feature_values, feature_ids] = g.topk(4);  // terminal [...,N,M,4]
```

The exported `residual_reduce` free function accepts selected N/M/D dimensions,
keepdim and correction, plus sample/feature tile sizes. The exported
`residual_topk(x, k, y, ...)` selects D and returns terminal values/indices.
Feature-only keepdim is
not a lazy residual view. See [residual contracts](residual-operators.md).

### Image Event Sources

Image maps begin with `[...,N,M,C,Hout,Wout]` event metadata. Pixel residuals use
Y-X; error and SSIM maps keep their canonical image formulas. The full-event
methods on `ImageMapOperator` return ScalarExpression. `events()` provides an
`ImageMapExpression` for selected event stages, with explicit `scalar()` after
contracting all events:

```cpp
#include <ops/image_map_operations.h>
ops::ImageMapOptions image_options;
image_options.data_range = 1.;
image_options.window_size = 3;
image_options.chunk_size = 2;
auto maps = ops::image_map_operator(images, references, "ssim", image_options);
auto pooled = maps.mean();                  // ScalarExpression
// Ordered spatial mean -> channel maximum -> common scalar consumer.
auto channels = maps.events().mean({-2,-1}); // ImageMapExpression
auto [channel_values, channel_ids] = channels.topk(2); // terminal [...,N,M,2]
// Event-only keepdim is the second mean argument, or third var/std argument.
auto kept = maps.events().mean({-2,-1}, true);
auto strongest_channel = channels.max({-1}).scalar();
auto [scores, indices] = strongest_channel.topk(2, -1, true);
auto channel_mean = channels.reduce_samples("mean", {-2});  // Tensor [...,N,C]
auto joint = maps.events().reduce_samples(
    "var", {-4,-2,-1}, false, 0.);           // Tensor [...,N,C]
```

`ImageMapOperator::topk` and `ImageMapExpression::topk` select one integer event
axis and return terminal values/indices; `image_event_topk` is the exported
free-function entry. `ImageMapExpression::reduce` takes `(reduction,dims,keepdim,correction)` and keeps
an event expression return type. `reduce_samples` is the explicit terminal
Tensor entry; its dimensions refer to the current shape. The exported
`image_event_reduce` free function accepts the encoded program and terminal
settings; prefer the facade when constructing programs by hand. All nine event
reducers are available. Large-map arithmetic and backward replay operate on
native spatial/channel regions, with no Python execution dependency. Details
and remaining full-event-buffer limits are in
[image-map operators](image-map-operators.md).

### Execution Boundary

All built-in numerical work and Autograd registrations execute without Python.
Canonical expression calls flatten source tensors into a typed dispatcher entry;
older serialized schema names remain registered. Python supplies metadata for
compile/export; it is not a hidden dependency of native forward/backward.
Loading an exported graph still requires its native operators and a compatible
runtime. Arbitrary user callbacks and encoder factories have separate deployment
requirements.

Native replay supports first derivatives and preserves comparison parameter,
input and coefficient gradients where defined; higher derivatives are unsupported.
Shapes, empty/exclusion/tie rules and memory follow the
[operator guide](operators.md). Standalone CMake/CTest validation of new facades
is reserved for review; the implementation-stage extension build does not by
itself establish standalone execution coverage. See [CMake](cmake.md).


## Unified Collection and Single-Sample Selectors

`comparison_selectors.h` provides `compare` and `single_comparison` with
`ComparisonOptions`. Tensor collections have the named family layouts;
`ComparisonCollection` groups Gaussian/mixture tensors. Modes are all_pairs,
paired, condensed and single. The native catalog also supplies Python metadata.
Distance cdist/pdist wrappers use its distance role; every kernel is selectable
through compare with role="kernel". Standalone execution needs no Python.

```cpp
using namespace torchpairwise::ops;
ComparisonOptions options;
options.squared = true;
// mu:[D], covariance:[D,D], optionally with leading broadcast batches.
auto value = single_comparison(ComparisonCollection{mu, covariance},
    ComparisonCollection{other_mu, other_covariance},
    "gaussian_wasserstein_distances", options);
// X:[N,D], Y:[M,D]. Broad options use the same native cdist as Python.
ComparisonOptions metric_options;
metric_options.alpha = 0.5;
auto divergences = cdist(X, c10::optional<at::Tensor>(Y),
    "renyi_divergences", metric_options);
```

Unused supplied options are rejected natively. The window option is an integer
DTW band or a string STFT window according to the selected family; it is the one
union-valued field. All other fields retain typed Tensor/scalar/list options.
Single mode creates singleton sample views and reuses native paired execution.
Native schema expansion requires rebuilding consumers and re-exporting graphs.
Standalone tests for this addition are deferred to the maintainer's review.

## User-Defined Comparisons

`custom_compare` exposes all four output modes for registered paired-tile
operators. `custom_expression` binds the same user operators to the scalar
streaming engine and its algebra/consumers. Both are declared in
`custom_comparisons.h`, included by `<torchpairwise.h>`.

```cpp
namespace tp = torchpairwise::ops;
auto values = tp::custom_compare(X, Y, {weights},
    "my_metrics::paired_forward", "my_metrics::paired_vjp");
auto A = tp::custom_expression(X, Y,
    "my_metrics::paired_forward", "my_metrics::paired_vjp", {weights});
auto mean = A.reduce("mean", -1);
```

Forward has signature `(Tensor, Tensor, Tensor[]) -> Tensor` over P corresponding
samples and returns `[P]`. VJP has signature `(Tensor, Tensor, Tensor, Tensor[])
-> Tensor[]` for upstream gradients, paired inputs and explicit parameters.
It returns input-shaped gradients and parameter gradients already summed over
pairs. Omitting the VJP requires forward's own registered autograd behavior.
Metadata, backend implementations and registration lifetimes are owned by the
user extension. Python callbacks can bind the same engine but retain their
Python dependency. See [custom contracts](custom-metrics.md) and the
[native extension example](../examples/native_custom_metric.cpp).


### C++ Callable Convenience

`torchpairwise::ops::make_pairwise` accepts ordinary C++ scalar callables and
returns a four-mode `CustomComparison`. Call it for a dense result or bind
`.expression(X,Y)` for the shared scalar streaming engine. Optional
`CustomComparisonBackward` and `CustomComparisonOptions` provide VJPs, explicit
Tensor parameters and batching options. TorchPairwise manages registrations;
users can keep the lower-level named-operator API for deployment. See
[custom metric examples and lifetime contracts](custom-metrics.md#c-callables).

For registered broadcast operators, `custom_compare` has a final
`bool broadcast=false` argument after `symmetric`, and `custom_expression` has
a final `bool broadcast=false` argument after `PairwiseOperationOptions`.
Passing `true` calls Cartesian tiles with `[n,1,*sample]` and `[1,m,*sample]`,
requiring `[n,m]` outputs; other tiles use paired `[P,*sample]` operands and
require `[P]`. VJPs must reduce broadcast axes and return the exact passed
operand shapes, with shared parameter gradients already summed. The default
preserves existing calls. C++ callable factories select the same contract with
`CustomComparisonOptions.broadcast=true` (default false), without requiring
`vectorized=true`. See [broadcast callbacks](custom-metrics.md#broadcast-form).

## Streamed Low-Rank GP and Landmark Selection

`linear_lowrank.h`, included by `<torchpairwise.h>`, exports
`pivoted_cholesky(op, rank, rtol, atol)`, `lanczos(op, k, initial,
reorthogonalize)`, `love_cache(op, rank, initial)`,
`predictive_variance(cache, cross, prior)`, `farthest_point_sampling(op, k,
start)` and `kmeans_plusplus(op, k, generator)`. Operators use the existing
`LinearOperator` facade. Pivoted Cholesky and selection require a single scalar
source (`LinearOperator(pairwise_expression(...))` or a matrix); linear-only
compositions are rejected. Lanczos and LOVE accept symmetric linear maps.

Setup is inference-only, performs double native arithmetic and returns source
dtype factors. Pivoted Cholesky reads one column per step; selection reads one
row. Both reuse prepared scalar sources and bounded tiles without constructing
the full kernel. Lanczos uses streamed matvecs and full reorthogonalization;
LOVE factors its small tridiagonal and returns a reusable `LOVECache.root`.
`predictive_variance` keeps cache state constant, supports cross/prior gradients
and clamps marginal variances at zero. See [GP prediction](gp-prediction.md)
for shapes, stopping, rank/accuracy tradeoffs and generator determinism.
Standalone build/runtime validation is deferred to maintainer review.
