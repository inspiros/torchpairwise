# Streaming Operator Guide

An operator binds inputs and a comparison formula. It evaluates tiles when a
consumer requests a result, avoiding a full computed pairwise intermediate.
Use a dense comparison when the matrix itself is the output; use an operator
when the result is a product, neighbors, statistic or selected block.

Use `tp.lazy` for comparisons and kernels, `tp.as_lazy` for existing tensors,
and `tp.residual_operator` or `tp.image_map_operator` for event-valued sources.
Custom comparisons use `tp.lazy(X, Y, comparison)`; prepared feature comparisons
use `tp.lazy(prepared)`.

```python
import torch
import torchpairwise as tp

X = torch.randn(20, 6)
Y = torch.randn(30, 6)
V = torch.randn(30, 4)

A = tp.lazy(X, Y, metric="sqeuclidean", chunk_size=16)
nearest = A.topk(3, largest=False)       # values/indices: [20,3]
row_mean = A.mean(dim=-1)               # tensor [20]
threshold = A.quantile(0.9, dim=-1)     # exact row thresholds, tensor [20]
smoothed = (-0.5 * A).softmax_matmul(V) # tensor [20,4]
```

The binding stores references and options. It creates no comparison matrix or
persistent numerical cache. Each consumer evaluates the formula again and sees
input mutations. Several consumers on one binding are separate computations;
combine entries into one expression before a consumer when that is the intended
formula.

For mode names, compatibility helpers and defaults, see the
[API naming guide](api-design.md). For reproducible comparisons, use the
[benchmark harness](benchmarking.md).

## Choose a Source

| Source | Logical output shape | Use it for |
|---|---|---|
| `tp.lazy(X,Y,metric=...)` | `[...,N,M]` | Any of the 87 canonical scalar comparisons, including structured and modality families |
| `tp.lazy(X,Y, metric=...)` | `[...,N,M]` | The 13 kernel comparisons and coordinate interactions for vector kernels |
| `tp.as_lazy(scores)` | `[...,N,M]` | Caller-owned dense values using the common interface; the input matrix already occupies memory |
| Scalar algebra, e.g. `alpha*K1 + K2` | `[...,N,M]` | Combine scalar entries before a common consumer |
| `tp.residual_operator(X,Y,transform=...)` | `[...,N,M,D]` | Signed X-Y, absolute or squared coordinate residuals; reduce features or samples |
| `tp.image_map_operator(X,Y,comparison=...)` | `[...,N,M,C,Hout,Wout]` | Aligned Y-X, absolute/squared error or valid-window SSIM maps; retain selected event axes |

`A.shape`, `A.dtype` and `A.device` describe source entries without evaluating
the matrix. A consumer can have a different output dtype, such as int64 counts.
Image-map sources also expose their current `event_shape`.

Leading batch dimensions broadcast. N indexes queries, M references; trailing
**event axes** describe the result of one pair comparison. They are distinct
from input features: a distance computed from D input coordinates still has a
scalar event shape.

Kernel sources share the common interface and add vector-coordinate
`displacement` and `residual_outer_product`. General expressions do not
provide these geometric methods.

Scalar comparison names and tensor/tuple inputs follow [compare](dispatchers.md).
Use full names such as `cosine_distances` or `cosine_similarity` when an alias
is ambiguous. These family guides cover preparation and options:

| Family | Guide |
|---|---|
| Vector and point-cloud comparisons | [Scalar pairwise operators](pairwise-operators.md) |
| Covariance, Gaussian and Gaussian mixture | [Structured operators](structured-operators.md) |
| Image and audio scalar scores | [Media operators](media-operators.md) |
| DTW/Soft-DTW, integer tokens and signed MMD | [Ordered operators](ordered-operators.md) |
| Supplied dense matrices | [Matrix operators](matrix-operators.md) |
| Kernels and vector interactions | [Kernel operators](kernel-operators.md), [interaction maps](interaction-maps.md), [kernel moments](kernel-moments.md) |
| Coordinate residuals | [Residual operators](residual-operators.md) |
| Image maps with event axes | [Image-map operators](image-map-operators.md) |

## Common Scalar Consumers

Entrywise lazy pairwise objects support all consumers below. Products `A @ B`,
`softmax`, `eye_like`, `diag_embed`, `block`, `group_projection`, and
preconditioners have linear capability: they support `@`, `.mT`, `sum`, `mean`,
scalar multiplication, addition/subtraction with other lazy objects, and
solvers. Other consumers raise with a `.to_dense()` hint. See
[linear operators](linear-operators.md).

`torch.sum(K, -1)`, `torch.matmul(K, V)`, and `torch.softmax(K, -1) @ V`
operate lazily. `torch.linalg.*` is rejected; use `tp.linalg`. Use `to_dense()`
to materialize, and `amin`/`amax` or `argmin`/`argmax` for extrema.
`min(dim)` and `max(dim)` return values and indices; `median(dim)` returns
values alone, unlike torch. `.mT` is the lazy transpose.

Every consumer below returns a tensor, or a named pair for selection.

| Consumer | Meaning / output for A: `[...,N,M]` |
|---|---|
| `A.matmul(V)` | \(\sum_j A_{ij}V_{jq}\); `[...,N,Q]` for V: `[...,M,Q]` |
| `A.softmax_matmul(V)` | \(\sum_j\operatorname{softmax}_j(A)_{ij}V_{jq}\); `[...,N,Q]` |
| `A.bilinear(L,R)` | \(L^TAR\); `[...,P,Q]` for L: `[...,N,P]`, R: `[...,M,Q]` |
| `A.hsic(B, unbiased=False)` | Biased or unbiased HSIC for square Gram sources; one scalar per batch |
| `A.frobenius_inner_product(B, center=False, normalize=False)` | Entry inner product; optional double centering/normalization, scalar per batch |
| `A.quadratic(V)` | \(V^TAV\); square A required, symmetry/positivity not assumed |
| `A.diagonal()` | `[...,min(N,M)]` |
| `A.block(rows,columns)` | Only the selected dense row/column cross product |
| `A.topk(k,dim=-1,largest=False)` | `values` and int64 `indices`, both `[...,N,k]` |
| `sum`, `mean`, `min`, `max`, `prod`, `all`, `any` | Reduce either or both sample axes |
| `var`, `std`, `logsumexp`, `count_nonzero` | Sample-axis statistics; see [statistics](operator-statistics.md) |
| `weighted_mean`, `weighted_var`, `weighted_std` | Normalized confidence-weighted statistics |
| `median`, `quantile` | Exact selection over one N/M axis; values only, row-block sorting |

Ordinary reducers accept one axis or a tuple/list of axes; top-k, median and
quantile each require one integer axis. Event-only ordinary reductions stay
lazy; selection returns tensors. This boundary is shared across source families.

Products/forms also accept unbatched coefficient vectors, removing the
corresponding output channel axes. Coefficient batches can add broadcast batch
axes. Dimensioned min/max return values and indices; `amin`/`amax` return values only.

```python
K = tp.lazy(X, Y, metric="rbf", gamma=0.2)
U = torch.randn(20, 2)
product = K.matmul(V)                  # [20,4]
form = K.bilinear(U, V)               # [2,4]
column_spread = K.std(-2, correction=0) # [30]
confidence = torch.ones(30)
weighted = A.weighted_var(confidence, dim=-1, correction=0)  # [20]
```

A confidence tensor belongs to one selected axis, M or N. Pair-specific weights
use another scalar operator or `tp.as_lazy(weights)`. Joint N/M weighted
statistics require operator weights. Ordinary var/std default to correction 1;
weighted var/std default to 0. `weight_rows` and `weight_columns` multiply entries
without normalizing; see [weighted statistics](operator-statistics.md#normalized-weighted-statistics).

[Coordinated Frobenius products](kernel-alignment.md) support centered kernel
alignment and biased HSIC. Centering needs O(N+M) state and a separate mean pass;
normalization can be combined with centering. This remains a terminal consumer.

## Algebra Stays Lazy Until a Consumer

Supported scalar entry operations are `+`, `-`, entrywise `*`, unary negation,
`square()`, `exp()`, `weight_rows` and `weight_columns`:

```python
K1 = tp.lazy(X, Y, metric="rbf", gamma=0.2)
K2 = tp.lazy(X, Y, metric="laplacian", gamma=0.3)
mixture = 0.7 * K1 + 0.3 * K2          # lazy [...,N,M]
features = mixture.matmul(V)           # tensor [...,N,4]
alignment = (K1 * K2).sum()            # tensor [...], uncentered entry inner product
```

Tensor scalar factors must have zero dimensions. Row/column weights have shapes
`[...,N]`/`[...,M]`. Different comparison sources need equal positional N/M counts
and matching sample order; batches broadcast. `A.T` preserves directed formulas:
it transposes entries of KL(X,Y), rather than evaluating KL(Y,X).

`A * B` multiplies entries. A matrix product of two lazy operators, arbitrary
Tensor methods on a lazy object, and user tile producers are outside this API.
The [algebra guide](operator-algebra.md) gives promotion, gradient and memory
contracts.

## Dimensions and Event Reductions

`dim` indexes the object's **current shape**, including leading batch axes for
positive indices. Integers and tuples/lists select logical axes; batch axes are
preserved and repeated dimensions are errors. A tuple is one joint reduction,
with one observation count and one correction, rather than successive statistics.

| Object | Default reduction dimensions | Explicit `dim=None` or `[]` | Event-only result | Selecting N or M |
|---|---|---|---|---|
| Scalar `[...,N,M]` | N and M; weighted methods default to M | N and M | No event axes | Tensor |
| Residual `[...,N,M,D]` | D | N, M and D | lazy pairwise object; feature keepdim unsupported | Tensor |
| Image map `[...,N,M,C,H,W]` | All currently remaining event axes | N, M and all current events | ImageMapOperator if events remain; lazy pairwise object after full contraction | Tensor |

Residual and image sources expose nine reducers: `sum`, `mean`, `var`, `std`,
`min`, `max`, `prod`, `all`, `any`. Expanded scalar statistics and weighted
statistics become available **after** event contraction. Residual `topk` selects
one D axis and returns terminal `[...,N,M,k]` values/indices. Scalar top-k accepts
one N/M dimension. Image-event top-k selects one current event axis, replacing
it by k. All top-k methods accept one integer `dim`, following PyTorch;
dimension lists and joint-axis top-k are unsupported. Exact median and quantile
accept one N/M axis after all event axes have been contracted. They retain
complete rows in budgeted blocks; see [exact selection](operator-statistics.md#exact-median-and-quantile).

```python
G = tp.residual_operator(X, Y, transform="abs", feature_chunk_size=32)
mae = G.mean(-1)                  # lazy [20,30], contracts D
nearest_mae = mae.topk(3, largest=False)
largest_coordinates = G.topk(2)  # tensors [20,30,2], selects D within each pair
mean_by_feature = G.mean(-2)      # tensor [20,6], contracts M
joint_spread = G.var((-2,-1), correction=0)  # tensor [20], joint M/D
```

Images retain several event axes, so partial event chains are useful:

```python
images = torch.rand(4, 3, 12, 14)
references = torch.rand(7, 3, 12, 14)
G = tp.image_map_operator(images, references, comparison="absolute_error",
                          chunk_size=2)
H = G.mean((-2,-1))               # lazy [4,7,3], spatial mean
A = H.max(-1)                    # lazy [4,7], maximum channel error
nearest_images = A.topk(2, largest=False)
mean_per_channel = H.mean(-2)     # tensor [4,3]; -2 now means M
joint_variance = G.var((-4,-2,-1), correction=0)  # tensor [4,3]
kept = G.mean((-2,-1), keepdim=True)  # lazy [4,7,3,1,1]
```

`joint_variance` reduces M/H/W once for each N/C. In contrast,
`H.var(-2, correction=0)` measures M variance **after** spatial averaging and
usually differs. Likewise width mean followed by height max runs in that order.
Image event keepdim retains singleton metadata; scalar and terminal consumers
retain reduced singleton axes when requested.

The streaming boundary is the return type. `H.mean(-2)` returns a tensor;
calling `.max(-1)` on that tensor is an ordinary PyTorch reduction after N*C
values have been produced. It does not fuse back into the source. Event-only
steps stay inside the native tile program until a terminal consumer runs.
See [event shapes](operator-event-shapes.md) and the family guides for details.

## Self Exclusion, Truth and Selection

Operator consumers include self terms by default, even when Y is omitted.
`exclude_self=True` on supported scalar reductions, top-k and softmax products
removes the positional diagonal and requires a square operator.
Use `exclude_self=True` to omit the diagonal; see [row reducers](reductions.md).
Uncontracted residual/image event reducers do not take `exclude_self`.

`all`/`any` use nonzero truth, including numeric distances from Boolean inputs.
A Python predicate may be passed to `all`/`any` (it is not exportable),
for example `D.any(-1, predicate=lambda d: d < t)`;
comparison syntax such as `A < t` is not a scalar algebra node.

Top-k returns indices along the selected current sample axis. Ties choose lower
indices; `largest` defaults to True, so set False for nearest distances. The
[selection contract](pairwise-operators.md#top-k-selection) covers NaNs, k=0,
exclusion and output storage.

## How Sample and Feature Tiles Cooperate

Both named scalar metrics and residual feature reductions stream sample pairs.
`chunk_size=C` limits each N/M tile to `min(C,N_remaining)` by
`min(C,M_remaining)`. Residual `feature_chunk_size=F` independently limits its D
slice to `min(F,D_remaining)`. Edge tiles are shorter; no padding is required.

Chunk sizes are not presets: any built-in Python `int` from 1 through
`2**63 - 1` is accepted, including 7, 129 or 300. Values larger than an axis
simply cover its remaining entries in one tile. Zero, negatives, booleans,
floats (including `128.0`), infinity, NaN and integers outside that range are
invalid. There is no special 0, -1 or infinity sentinel for an unbounded tile.
`None` selects the existing native/default policy only for APIs whose signature
allows it, such as scalar pairwise/kernel/matrix operators; residual and image
map operators require integer widths. The same range applies to feature,
component and encoder chunk widths where those options are exposed.

For a squared-residual feature mean, native execution is conceptually:

```text
for each X sample block:
    for each Y sample block:
        initialize one sum per pair in this sample tile
        for each feature block:
            residual = X_block[..., features] - Y_block[..., features]
            accumulate sum(residual ** 2, feature_axis)
            discard residual and feature temporaries
        scalar_tile = accumulated_sum / full_D
        pass scalar_tile to the selected matrix consumer
```

With C=128 and F=32, the residual tile has at most128*128*32 float32 entries,
about2 MiB. Double accumulation and additional forward/backward temporaries
increase total workspace. A mean divides by the full D once; variance merges
counts,means and centered-square sums; extrema retain a value and witness index.
First backward replays only feature values needed by the derivative instead of
saving every pair's residual. Identity sum/mean over D has a constant derivative
along D, so its backward contracts sample cotangents once and broadcasts into
input gradients without reconstructing residual blocks.
If the consumer is a row sum, it also merges each scalar tile immediately.

The residual/event working tensors scale with C*C*F plus per-pair state, rather
than N*M*D; inputs,output,state and their gradients remain. C controls two sample
axes, so halving it can quarter a feature tile; halving F halves that tile.
Smaller tiles typically require more launches and replay iterations.

An optimized named scalar metric such as squared Euclidean can reduce D inside
a fused kernel or use norms and matrix multiplication, avoiding a pairwise
feature tile altogether. The residual path supplies configurable feature
statistics through ATen operations; it is not automatically the faster or smaller
replacement for a named metric. Other named metric families have their own
internal workspace requirements.

Expression consumers inherit source chunk widths. Built-in vector pairwise and
kernel sources default to 2048-sample tiles; point-set, matrix, residual and
custom sources default to 128. Mixed sources use the smallest source width, so
a 128-wide source can limit a mixed expression. Explicit larger source widths
are honored instead of being capped at 128 by the planner.

At a fixed sample width of 128 on the same residual workload, reducing the
feature width from 128 to 64 to 32 reduced extra peak allocation from 68.6 to
36.6 to 20.6 MiB; forward plus backward took 561, 754 and 1,217 ms respectively.
Outputs and both input gradients matched an independent double-precision
formula. This illustrates the memory/launch tradeoff independently of sample
width, rather than establishing a generally optimal chunk choice.

## Dense Tensors versus Generated Comparisons

If a tensor `A` with shape `[N,M,D]` already exists, `A.mean(-1)` normally needs
little extra workspace beyond its `[N,M]` result. Reading `A` in chunks cannot
recover its existing allocation. Use normal PyTorch reductions in that case.

Streaming matters when the pairwise values are **generated from smaller inputs**:

```python
residuals = tp.residual_operator(
    X, Y, transform="square", chunk_size=256, feature_chunk_size=64)
row_totals = residuals.mean(-1).sum(-1)  # [N], no dense [N,M,D] or [N,M]
```

For a generated `[N,M,D]` comparison, dense storage alone costs
`N*M*D*element_size`; streaming uses working tiles, reduction state and requested
outputs, plus inputs and gradients. A requested full `[N,M]` output still costs
`N*M*element_size`. Working-tile entry counts are not total-memory guarantees:
double accumulation and temporary tensors add storage.

Dense eager evaluation can be faster because it uses fewer, larger operations.
Streaming adds launches and often backward recomputation. Larger chunks can
improve throughput while using more memory. Which wins depends on the formula,
reduction, hardware and compiler; PyTorch compilation can also fuse some dense
expressions, so eager comparisons do not establish a universal advantage.

One development comparison used float32 `N=M=2048`, `D=128`, squared residuals
averaged over D and summed over M, including first backward:

| Evaluation | Additional peak allocation | Forward + backward |
|---|---:|---:|
| Dense eager PyTorch expression | 10,240 MiB | 72.2 ms |
| Residual streaming, chunks 512 / 128 | 1,033 MiB | 218.2 ms |
| Residual streaming, chunks 256 / 64 | 134 MiB | 307.4 ms |
| Named squared-Euclidean streaming, sample chunk 256 | 5.9 MiB | 70.6 ms |

The residual chunk columns mean executed sample width / feature width. These
numbers were rerun after fixing an expression-planner cap that had silently
limited sample widths to128. Earlier 69/37 MiB results used128-sample tiles,
despite larger requested widths; they should not be read as512/256-tile results.
The named metric
computes the equivalent statistic with
`tp.lazy(X, Y, metric="sqeuclidean", chunk_size=256).sum(-1) / X.shape[-1]`,
which avoids the vector-valued reduction path. Prefer a named scalar metric when
it expresses the comparison you need. Outputs and both input gradients agreed
within float32 tolerance; these are not promises of bitwise equality.

Measurements used PyTorch 2.12.1+cu126 on a Quadro RTX 6000, one warmup and two
timed repetitions. Allocation is above persistent inputs and includes backward
work. This compares the expressions as written on a shared system; compiler
fusion or formula-specific algebra can change the outcome. In particular,
squared-distance row sums also admit an algebraic shortcut. The generic stream
saves memory here at substantial latency cost, rather than a guaranteed small
penalty.

Choose dense comparisons or operator bindings explicitly. `chunk_size` controls
sample tiles; residual `feature_chunk_size` controls D chunks. Within image
operators, large-map regional execution uses an automatic threshold, with no
separate force-regional/force-dense flag. These controls preserve the logical
reduction and its `dim` meaning.

## Fused CUDA Reductions

Some CUDA reductions evaluate each pair and reduce it in one kernel,
without sample tiles. A reduction takes this path when all of these hold:

- both inputs are float32 or float64 CUDA tensors of the same dtype, shaped
  `[..., N, D]`, with any broadcast batch and 1 ≤ D ≤ 64;
- the source is `sqeuclidean`, `manhattan`, `chebyshev`, `minkowski` with
  finite `p > 0` other than 2, `rbf`, `laplacian`, `matern` with `nu` in
  {0.5, 1.5, 2.5}, `rational_quadratic`, `linear`, `polynomial` or `sigmoid`,
  with scalar parameters;
- the expression is that one source, possibly transposed, followed by up to
  eight steps. Each step adds, subtracts, multiplies or divides by a Python
  number or by row or column values (`add_rows`, `add_columns`, `weight_rows`,
  `weight_columns`, or `row_values`/`column_values` operands), or applies
  `neg`, `exp`, or `where` on boolean row/column values with a number fill,
  such as `(-0.5 * D).add_rows(f).add_columns(g).exp()`.
  Row/column value tensors may require gradients;
- the consumer is `matmul`, sum or mean, `logsumexp`, `softmax_matmul`
  (with or without `exclude_self`), or top-k, min, max, argmin or argmax with
  k ≤ 64. Selections on the dot-product sources (`linear`, `polynomial`,
  `sigmoid`) keep the tile path: native forms their dot products with a matrix
  product, whose rounding a fused sum cannot reproduce, so near-ties could
  select different pairs;
- the shape is one where fusing is faster. Minkowski with `p` other than 1
  fuses up to D = 4, Chebyshev up to D = 16, and `matmul` with more than 16
  payload columns up to D = 4. Beyond these, the tiles were faster in
  benchmarks.

First derivatives of `matmul`, sum, mean, `logsumexp` and `softmax_matmul`
are fused too, except for Chebyshev. Its native backward splits tied maxima
evenly, so it keeps the tile path. So does the `softmax_matmul` backward of
`polynomial` with degree 2 or more: the fused backward centers with the
rounded output, and polynomial slopes amplify that rounding. Top-k backward
already evaluates only the selected pairs. Under CUDA autocast, `matmul`, sum and `softmax_matmul` cast
their operands to float32 and still fuse; `logsumexp` uses tiles.

Everything else uses the tile path. This includes:

- `euclidean`, and `minkowski` with `p=2`. These use `torch.cdist`, whose
  matrix-product form gives a small rounding residual at coincident points
  and NaN for infinite coordinates. A fused difference loop would give 0 and
  inf instead, changing native results. `cosine` also uses tiles.
- tensor parameters, several sources, 0-dim tensor constants, row/column
  indices, a number divided by the expression, `square`, `sqrt`, `pow`,
  `minimum`/`maximum`, and conditions other than one row/column mask or the
  AND of a row and a column mask. Pair masks from `tp.masks` compare labels
  or indices across the two axes, so they use tiles.

Fused values use the native formulas but a different summation order, so they
agree with the tile path within floating tolerance rather than bitwise. On
GPUs with reduced float64 throughput, fused float64 kernels may be slower
than float32 by a large factor; they still avoid the tile workspace.
Selections are exact, including tie order, NaN placement and signed zeros.
`chunk_size` is ignored on this path: it bounds tile memory, and the fused
kernels have no tile. The kernels run inside the registered native
operations, so `torch.compile`, `torch.export` and standalone LibTorch use
them too.

## Optional KeOps Backend

Native execution is the default. Install `torchpairwise[keops]` to enable
optional Python-side KeOps evaluation of supported unbatched float32/float64
vector distances, kernels and scalar expressions:

```python
with tp.backend("keops"):
    K = tp.lazy(X, Y, metric="rbf", gamma=0.5)
    result = K.matmul(V)

# A leaf override persists through transpose and surrounding contexts.
K = tp.lazy(X, Y, metric="rbf", backend="native")
```

Contexts nest and accept `"native"`, `"keops"` or `"auto"`. Explicit leaf
choices take precedence; conflicting choices, or any native leaf, select
native execution for the expression. `"keops"` requires the optional dependency;
`"auto"` silently uses native execution when it is unavailable. Auto selects
only supported float32 CUDA workloads with at least 2**25 pairs (about
N=M=5800) and at most 64 features, using consumers with matching semantics.
These thresholds were measured on one GPU; set the backend explicitly when your
own measurements differ. Unsupported sources,
options and consumers fall back to native execution. `tp.linalg` solvers and
`bilinear`/`quadratic` run their native implementations. `torch.compile` and
`torch.export` always use native execution; standalone LibTorch also uses native.

Results agree within floating tolerance rather than bitwise because summation
order differs. The first call for a formula JIT-compiles for several seconds;
KeOps caches compiled formulas in `KEOPS_CACHE_FOLDER`. Scalar parameter changes
reuse the formula. Under `"keops"`, selection tie order and NaN placement are
unspecified. Auto does not select these ordering-sensitive consumers.

PyKeOps autograd supports first derivatives through supported inputs, values,
row/column weights and scalar tensors. Chebyshev and general Minkowski distances
use KeOps only when gradients are not needed; unsupported gradient cases use
native execution. Higher derivatives are outside the contract, as for native
execution.

## Memory, Gradients and Deployment

Streaming avoids storing the full **computed intermediate**, not inputs,
outputs or gradients. Blocks and large top-k requests need their natural output
storage. `tp.as_lazy` starts from an existing dense allocation. Tile width
also does not bound a metric's internal point/sequence/image workspace.

`chunk_size` controls sample tiles and, for common scalar products/forms,
payload-channel tiles; [fused CUDA reductions](#fused-cuda-reductions) ignore
it. Residual `feature_chunk_size` bounds D workspace. Image
programs use spatial/channel regions for large-map arithmetic and VJPs. A first
event sum/mean/var/std/min/max/prod/all/any contracts directly into reduction
state, avoiding its full per-sample-tile event map and cotangent. Direct
sample/event terminal reductions merge regions without that intermediate. Prepared SSIM features and output/state buffers
remain; this is not a fixed total-memory budget. See [image memory details](image-map-operators.md#memory-and-native-execution).
Float64 accumulation favors accuracy and can limit GPU throughput.

Built-in forward and first backward are registered native C++/ATen operations.
Backward replays tiles; higher derivatives are unsupported. Replay means
recomputing generated pairwise values from the retained operands during backward.
It is bounded because only one limited-size sample/feature/image region and its
cotangent are materialized at a time, then discarded after accumulating input
gradients. This exchanges extra computation for less temporary storage, not a
fixed bound on total application memory. Differentiability
and domains still follow each source. Boolean/token inputs have no derivatives;
floating coefficients may still differentiate. Python callbacks and encoders
have their separate [custom](custom-metrics.md) and
[feature-reuse](learned-comparisons.md) deployment requirements.

Build bindings and expressions inside `forward`, use `.matmul()` explicitly,
and return tensors when using `torch.compile` or `torch.export`:

```python
class Smoother(torch.nn.Module):
    def forward(self, x, y, values):
        distance = tp.lazy(x, y, metric="sqeuclidean", chunk_size=16)
        return (-0.5 * distance).softmax_matmul(values)

model = Smoother()
compiled = torch.compile(model, fullgraph=True, dynamic=True)
result = compiled(X, Y, V)
```

Exported execution requires the native library. Metric choices, event layout
and other structural options can specialize graphs; dynamic sample/batch support
is family-specific. Compilation does not turn native ATen tiles or a
CPU-coordinated transport solver into a new fused CUDA kernel. See
[PyTorch integration](pytorch-integration.md).

Standalone LibTorch shares the numerical implementations. Use typed
`ScalarExpression` factories for the full common consumer interface; older C++
family facades retain their signatures. Native image partial chains use
`maps.events().mean({-2,-1}).max({-1}).scalar()` for an explicit static-type
conversion. See [native API](native-api.md#streaming-operator-entry-points) and
[CMake installation](cmake.md).
