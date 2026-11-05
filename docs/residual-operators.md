# Feature-Axis Residual Reductions

For source selection and shared axis rules, start with the [operator guide](operators.md).

`residual_operator(X, Y)` binds a feature-valued source with logical shape
`[..., N, M, D]`, where inputs have shapes `[..., N, D]` and `[..., M, D]`:

\[
G_{ijd}=X_{id}-Y_{jd}\]

It constructs metadata and keeps references to the inputs. It does not allocate
the pair-feature tensor. Leading batch axes broadcast. Omit Y for a self-source;
mutations of the tensors remain visible on later consumer calls.

## One Reduction Spelling, Axes Defined by Shape

Use the usual `dim=` spelling rather than a separate `reduce_features` method:

```python
G = tp.residual_operator(X, Y, chunk_size=64, feature_chunk_size=128)
H = G.var(dim=-1, correction=0)  # lazy [..., N, M], variance over D
best_spread = H.amin(dim=-1)      # tensor [..., N], minimum over M
global_spread = H.mean()        # tensor [...], mean over N and M per batch
smoothed = (-gamma * H).softmax_matmul(V)  # tensor [..., N, C]
```

| Object | Logical shape | `dim=-1` | `dim=-2` | `dim=-3` |
|---|---|---|---|---|
| Residual source G | `[..., N, M, D]` | Feature D, lazy | Reference M, tensor | Query N, tensor |
| Scalar source H | `[..., N, M]` | Reference M | Query N | Batch axis, unsupported by scalar consumers |

Scalar operators reduce N/M; residual operators reduce any subset of N/M/D.
`dim` accepts an integer, tuple or list, with equivalent positive/negative indices
relative to the advertised shape. Duplicates, out-of-range dimensions and batch
axes raise errors. `None` and an empty tuple/list select all **logical operator
axes**, preserving batches: N/M for a scalar source, N/M/D for residuals.
Defaults remain unchanged: residual methods default to D; scalar methods default
to N/M together. This per-batch convention differs from reducing all dimensions
of an arbitrary Tensor.

```python
G = tp.residual_operator(X, Y, chunk_size=64, feature_chunk_size=128)
H = G.var(dim=[-1], correction=0)          # lazy [..., N, M]
row_spread = G.var(dim=[-2, -1], correction=0)  # tensor [..., N]
feature_mean = G.mean(dim=(-3, -2))        # tensor [..., D]
all_spread = G.std(dim=None, correction=0) # tensor [...], all N/M/D
row_mean = G.mean(dim=-2, keepdim=True)    # tensor [..., N, 1, D]
scalar_spread = H.var(dim=[-2, -1])        # tensor [...], over N/M
```

**Feature-only reduction remains lazy; any sample-axis reduction is terminal.**
`G.reduce("var", dim=[-1], correction=0)` and `G.var(-1, correction=0)` return
a lazy pairwise object, with the [common scalar consumers](operators.md#common-scalar-consumers)
and algebra.
`H.matmul(V)` then streams without a full computed N*M or N*M*D intermediate.
A sample reduction streams its selected axes into an ordinary Tensor. Operations
on that Tensor work normally, but do not fuse back into the completed source
reduction. For example, `G.mean(-2).std(-1)` computes std of per-feature reference
means after producing N*D values. It is a different computation from joint
`G.std((-2,-1))`.

Multi-axis statistics are **joint** reductions: selected axes form one set of
observations, and correction applies once to its total count. For M/D,

\[
\operatorname{var}_{M,D}(G)_i =
\frac{\sum_{j,d}(G_{ijd}-\mu_i)^2}{\max(0,MD-\delta)},
\qquad \mu_i=\frac1{MD}\sum_{j,d}G_{ijd}\]

The implementation merges centered tile moments over that combined set. It does
not rewrite this as successive variance calls. The order of user-written
reductions remains significant.

`keepdim=True` is supported for terminal residual reductions and scalar consumers.
Feature-only `keepdim=True` is rejected: there is no lazy singleton-feature view
yet, and requesting it must not silently materialize a dense N*M result. Keep the
feature contraction scalar to continue streaming. No arbitrary Tensor operation
is made lazy by this dimension support.

## Sources and Useful Comparisons

The supported `transform` options are `"identity"` (signed residuals), `"abs"`
and `"square"`. Reducers are `sum`, `mean`, `var`, `std`, `min`, `max`, `prod`, `all` and `any`.
They share the same dimension selection and lazy/terminal boundary.

```python
mae = tp.residual_operator(X, Y, transform="abs").mean(-1)
mse = tp.residual_operator(X, Y, transform="square").mean(-1)
nearest_mse = mse.amin(-1)
```

For ordinary named distances, use their direct `tp.lazy` formulas;
the residual source is useful when the per-feature statistic defines the desired
comparison. Population residual variance has a useful interpretation:

\[
\operatorname{var}_d(X_{id}-Y_{jd})
=\min_b\frac1D\sum_d(X_{id}-Y_{jd}-b)^2\]

It compares image patches or signals after removing the best constant brightness
or DC offset. Constant offsets intentionally have zero variance. Absolute-residual
std measures the consistency of errors across features. Image or waveform users
can flatten their chosen event axes into D before binding the source; this API
does not infer channel/spatial semantics or compute local-window SSIM maps.

## Selecting the Largest or Smallest Feature Residuals

`topk` selects coordinates within each pair, returning ordinary tensors rather
than another lazy source:

```python
G = tp.residual_operator(X, Y, transform="abs", chunk_size=32,
                         feature_chunk_size=64)
errors = G.topk(4, dim=-1)        # values and int64 indices: [...,N,M,4]
loss = errors.values.mean()      # mean of the four largest errors per pair
```

This is useful for sparse feature mismatches, hard-coordinate losses, and
identifying which coordinates explain a comparison. `largest=False` selects
the smallest entries. `dim` must be one integer selecting D (`-1` or the
equivalent positive axis); sample-axis selection remains available after a
scalar feature contraction such as `G.mean(-1).topk(...)`. Dimension lists,
batch axes and direct N/M top-k on a residual source are unsupported.

`0 <= k <= D`; zero is allowed, including D=0. Empty N/M or batch dimensions
produce correspondingly empty outputs. Equal entries use the lower feature
index, independently of chunk sizes. NaNs rank above finite values: they are
selected first for largest and last for smallest. Unlike `torch.topk`, whose
tied indices are unspecified, this tie policy is deterministic. `sorted=False`
permits a sorted result. Indices have no gradients; values support first
gradients through the selected witnesses and broadcast input batches.

The algorithm merges each feature chunk with the current k candidates. Forward
working storage scales with a sample tile times `feature_chunk_size + k`,
plus backend sorting workspace. Values and indices themselves require
O(batch*N*M*k) storage. Large k can therefore approach the dense output size.
Backward saves those indices. Signed residuals scatter the selected gradients
directly, and absolute residuals gather only the selected coordinates before
scattering. Squared residuals replay bounded feature tiles, including unselected
nonfinite coordinates, to retain their source chain-rule behavior.
Input gradients and their float64 accumulators also require natural input-sized
storage. Smaller tiles do not set a fixed total application memory budget.

The result is terminal: subsequent tensor operations act on the allocated
N*M*k values. It does not retain a lazy streaming link into another consumer.
Image maps also support top-k along one current event axis; see the
[image-event selection guide](image-map-operators.md#top-k-along-one-event-axis).
Feature median/quantile remain future extensions.

## Numerical, Memory and Deployment Contract

Inputs must be dense real float32/float64 tensors with matching dtype/device and
equal feature counts. CPU/CUDA autocast promotes half/bfloat16 inputs to float32.
Subtraction and the optional absolute/square transform use the promoted input
dtype. Numerical reductions accumulate in float64 and return the input dtype.
`all`/`any` use nonzero truth and return nondifferentiable bool entries, including
when used as lazy feature-contracted sources. Variance
uses centered partial moments and a Chan merge, with finite anchors to retain
exact zero variance for constant inputs; it does not subtract raw second moments.
For float64 entries, feature and joint moments keep an anchor and mean offset
separately across tiles and in backward. Float32 retains its existing float64
absolute-mean path and state size. This avoids losing small variations when an
absolute float64 mean cannot be represented accurately. It does not provide
full-range overflow protection: subtraction/squaring at the public dtype and
moment arithmetic can still overflow. Relative accuracy of nearly cancelling
gradients is also limited by floating-point precision. See the
[statistics numerical contract](operator-statistics.md#numerical-and-memory-contract).

`var`/`std` use a finite real `correction` defaulting to 1, matching PyTorch's
sample variance. Set 0 for population variance. Empty feature sum is zero;
empty mean/var/std are NaN. Empty product is 1, all is True and any is False;
min/max reject empty selected axes. Min/max return values, propagate NaNs,
and send gradients to the first logical row-major witness across the selected
axes, independent of tile sizes. This matches first-witness min/max, rather than
the split-tie derivatives of `torch.amin`/`torch.amax`. Product replay counts
zeros and nonfinite factors separately. Its derivative at each factor is the
product of the other factors, including deletion of a sole infinity/NaN; this
explicit exclusive-product convention can differ from ATen's output/input
nonfinite derivatives. Nonpositive degrees of freedom follow PyTorch's
explicit-axis values and first derivatives, including nonfinite results.
Finite zero std with positive degrees of freedom has zero derivative. Features
containing NaN/infinity propagate according to the selected transform/statistic.

`chunk_size` bounds sample-axis tiles; `feature_chunk_size` bounds D tiles.
Numerical workspace includes a sample-pair tile times a feature tile, moment
state proportional to the retained output axes, and local input-gradient tiles. No full computed N*M*D tensor or
N*M comparison matrix is retained by streamed consumers. Inputs, consumer outputs
and gradients occupy their intrinsic memory; explicitly selecting a full block
still allocates its requested N*M output. First backward replays only the values
needed by the derivative. For identity sum/mean over D, it reduces the sample
cotangents once and broadcasts into the input gradients, without rebuilding a
pair/feature residual tile. Identity min/max adjoints accumulate at saved
witness coordinates without constructing a feature-wide selection mask.
Joint reductions visit blocks of the retained output shape; repeated selected
indices accumulate in float64. Paired feature extrema have one contribution
per row and can copy directly. Transformed extrema retain replay for their full
abs/square chain rule, including nonfinite factors. Mean adjoints divide
the upstream cotangent before expanding it into a feature block, avoiding a
separate divided-cotangent buffer. Forward reductions retain their existing
float64 accumulation and centered moment arithmetic.
Higher derivatives are unsupported. Float64 accumulation favors accuracy over
peak throughput, particularly on GPUs with limited float64 arithmetic.

For float32 N=257, M=259, D=513 with sample width 128 and feature width 128,
removing unnecessary backward value/cotangent buffers reduced additional peak
CUDA allocation in a forward-plus-first-backward comparison:

| Consumer | Previous | Optimized backward |
|---|---:|---:|
| Feature mean then scalar sum | 43.5 MiB | 26.9 MiB |
| Absolute feature mean then matmul | 67.3 MiB | 35.4 MiB |
| Feature min then scalar sum | 45.7 MiB | 37.4 MiB |
| Joint M/D mean | 42.2 MiB | 24.0 MiB |

These measurements include outputs, gradients and workspace above persistent
inputs, on a Quadro RTX 6000. One warmup and three timed repetitions showed
variable latency, so these results establish memory savings for this workload,
not a general speedup or a fixed memory budget.

Using saved witnesses further reduced **backward-only** extra allocation for
joint identity extrema on that workload from about 36 to 3 MiB. Backward began
with forward already complete and its inputs/state retained; these numbers
include the newly allocated input gradients. Full forward-plus-backward peaks
were unchanged at 74 to 106 MiB because the forward search remained dominant.
Feature-only extrema feeding scalar consumers still replay a local witness
search during backward, so their peak can also remain unchanged. Floating
accumulation order can differ; bitwise gradient equality is not promised, and
ATen honors PyTorch's deterministic-algorithm setting.

Construct the source inside `forward` for `torch.compile` and `torch.export`.
Numerical forward/backward are native, with typed dispatcher schemas. Standalone
LibTorch has the same source and reduction methods:

```cpp
#include <ops/residual_operations.h>
auto g = torchpairwise::ops::residual_operator(x, y, "identity", 64, 128);
auto h = g.var(-1, 0.);
auto smoothed = h.scale(-0.5).softmax_matmul(values);
auto [best, ids] = g.max().topk(8, -1, false);
auto [largest_features, feature_ids] = g.topk(4);  // [...,N,M,4] tensors
// Dimension-list overloads are explicitly typed Tensor consumers in C++:
auto row_spread = g.var(std::vector<int64_t>{-2, -1}, 0.);
auto feature_mean = g.mean(std::vector<int64_t>{-3, -2});
```

C++ has static return types: the existing integer feature-axis methods return
`ScalarExpression`; dimension-list overloads return Tensor consumers, including
when the list selects only D. `residual_reduce` is the equivalent exported free
function. `residual_topk(x, k, y, dim, largest, sorted, transform, chunk_size,
feature_chunk_size)` is the exported terminal feature-selection entry. Use the
integer reduction method when building a lazy scalar source.
ScalarExpression also accepts dimension lists in `reduce`, `statistics` and
operator-weighted `weighted_statistics`.

The API adds no learned model or dependency. General feature-expression algebra,
arbitrary event-axis sources and feature median/quantile remain separate extensions.
