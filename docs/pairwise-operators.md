# Streamed Scalar Comparison Operators

For source selection and shared axis rules, start with the [operator guide](operators.md).

`tp.lazy` binds a scalar comparison to its inputs and
settings. It represents the scalar comparison values from `compare`, evaluating
tiles when an operation needs them:

```python
A = tp.lazy(X, Y, metric="kl_divergences", chunk_size=64)
features = A @ V              # [...,N,C] for V:[...,M,C]
reverse = A.T @ U             # [...,M,C] for U:[...,N,C]
form = A.bilinear(L, R)       # L.T @ A @ R; no A @ R intermediate retained
row_min = A.amin(dim=-1)      # [...,N]
column_mean = A.mean(dim=-2)  # [...,M]
total = A.sum()               # one scalar per broadcast batch
```

Directed Hausdorff uses its distance values; witness indices remain on the
named comparison API. The vector/cloud scope is all 52 distances and 12 kernels in
the vector and point-cloud registries, including weighted/discrete comparisons, probability divergences,
missing-value Euclidean distances, point-set comparisons and transport
comparisons. Names and aliases follow [compare](dispatchers.md); use full
names such as `cosine_distances` or `cosine_similarity` when the short name is
ambiguous. The default is Euclidean distance. Select a kernel by name to use the same consumers and kernel geometry methods.

Covariance, Gaussian and mixture comparisons are also accepted using their
existing tensor/tuple collection forms; see [structured operators](structured-operators.md).
Image/audio comparisons also support the facade with shared native preprocessing;
see [media operators](media-operators.md). DTW, Soft-DTW, token edit distances and
signed MMD statistics support the same operations; see [ordered operators](ordered-operators.md).
Caller-owned matrices use `tp.as_lazy`; operators over user block producers
remain future work; no arbitrary Python callback is involved here.

## Operation Coverage

The Python common interface has twenty-five distinct numerical consumers:

| Group | Operations |
|---|---|
| Products and forms | `matmul`, `softmax_matmul`, `bilinear`, `quadratic`, `frobenius_inner_product`, `hsic` |
| Selectors | `diagonal`, `block`, `topk` |
| Reductions | `sum`, `mean`, `min`, `max`, `prod`, `all`, `any`, `var`, `std`, `logsumexp`, `count_nonzero`, `weighted_mean`, `weighted_var`, `weighted_std`, `median`, `quantile` |

`.mT` is a logical view, `reduce` selects a reducer, and `@` applies or composes maps.
They do not add numerical consumers. The facade accepts all 87 canonical named
comparisons, including thirteen kernel comparisons (twelve vector kernels and
Gaussian-mixture overlap). Aliases are not additional comparisons.
Kernel sources share the common consumers and provide
vector-coordinate displacement and
residual outer-product methods. See [kernel operators](kernel-operators.md).
For caller-owned values, [tp.as_lazy](matrix-operators.md) reuses the same
consumer interface.
Ordinary reducer `dim` accepts integer or tuple/list dimensions of the advertised
operator shape. Scalar lists select either or both N/M axes; batch axes are
preserved. Residual sources additionally support joint N/M/D reductions, and
[image-map sources](image-map-operators.md) support current C/H/W event axes
and joint sample/event reductions. An event-only contraction remains lazy;
a sample-axis consumer returns a Tensor.
See [axes and the streaming boundary](residual-operators.md#one-reduction-spelling-axes-defined-by-shape).

Scalar [operator algebra](operator-algebra.md) combines entries before these
consumers run. See [streamed statistics](operator-statistics.md) for correction,
numerical and native API contracts, including exact single-axis median/quantile.
General feature-valued expressions remain in the [design](operator-algebra-plan.md). Both [residual](residual-operators.md) and [image-map](image-map-operators.md)
sources contract events into scalar operators with the same `dim=` reduction
spelling; partial image event reductions retain their remaining event shape.

## Softmax Products

`A.softmax_matmul(V)` computes a stable row-normalized contraction:

\[
P_{ij}=\frac{\exp(A_{ij}-s_i)}{\sum_k\exp(A_{ik}-s_i)},\quad
s_i=\max_j A_{ij},\qquad R_{ic}=\sum_j P_{ij}V_{jc}\]

```python
scores = -gamma * tp.lazy(X, Y, metric="sqeuclidean")
smoothed = scores.softmax_matmul(values)  # [..., N, C], values: [..., M, C]
# Dot-product attention, with caller-owned features and payload:
scores = tp.lazy(Q, K, metric="linear") * (Q.shape[-1] ** -0.5)
attended = scores.softmax_matmul(V)
```

Scores are logits: use negative distances for similarity, or `linear` for
dot-product attention. An RBF operator already contains exponentiated affinities;
applying softmax to it exponentiates those affinities again.

`V` accepts `[M]` (returning `[..., N]`) or `[..., M, C]`, with positive C.
Score and payload batches broadcast, including after logical transpose.
Scores and payload must share float32/float64 dtype and device; autocast
promotes low-precision inputs to float32. `exclude_self=True` removes positional
diagonal entries before normalization and requires a square operator. Empty
reference sets and entirely excluded rows return zero with zero gradients.
Otherwise NaN, positive-infinity and all-negative-infinity score rows follow
`torch.softmax` and produce NaNs. An individual negative-infinity score has zero
probability when the row contains a finite score.

Native forward merges row maxima and scaled masses, retaining two float64
normalizer values per query. Forward and first backward replay bounded sample
and payload-channel tiles; neither a full computed score nor probability matrix
is retained. The output, source operands and their gradients still occupy memory.
Accumulation uses float64 for stability; this trades throughput for accuracy on
GPUs with limited float64 performance. This bounded consumer has no dropout,
causal mask or arbitrary attention mask; it does not promise the throughput of
a specialized fused attention kernel. Higher derivatives remain unsupported.

The common Python interface and the C++ `ScalarExpression::softmax_matmul`
use the same registered native implementation and support compile/export.
Construct a C++ source with `pairwise_expression`, `matrix_expression` or the
structured/media/ordered expression factories. The original C++ family facade
signatures remain unchanged.

## Products, Forms and Entries

`A.matmul(V)` and `A @ V` accept a vector `[M]` or matrix `[...,M,C]`.
`A.bilinear(L,R)` accepts vectors or matrices with N/M rows respectively, and
returns a scalar, vector, or `[...,R,C]` matrix. `A.quadratic(V)` computes
`V.T @ A @ V` for square A. Neither symmetry nor positive semidefiniteness is
assumed, even for a quadratic form. Operands must share the comparison output's
floating dtype and device. Token edit operators instead accept float32/float64
coefficients while retaining int64 entries; see the ordered-operator contract. Unweighted discrete distances produce float32, including Hamming on floating
inputs; weighted discrete distances follow their weights' dtype.

`A.diagonal()` evaluates the first `min(N,M)` corresponding entries.
`A.block(rows,columns)` explicitly materializes only those selected rows/columns.
Indices must be one-dimensional int64 tensors on the input device. Repeated and
empty selections are supported. Input and parameter batch axes broadcast, as do
product/form operands' batch axes. `A.shape`, `A.dtype` and `A.device` describe
the comparison matrix, excluding any future product operand batches.

`A.T` is a **logical matrix transpose**, including for directed metrics. If A
represents `KL(Xi,Yj)`, A.T still evaluates that formula, in transposed matrix
positions. Swapping X/Y would instead compute `KL(Yj,Xi)` and give different
values. Double transpose restores the original orientation. Metric parameters
and point lengths stay assigned to their original operands.

## Reductions

`sum`, `mean`, `min`, `max`, `prod`, `all` and `any` reduce references with
`dim=-1`, queries with `dim=-2`, or both sample axes with `dim=None` (default).
Batch axes are preserved. `keepdim=True` keeps the reduced singleton axes.
`exclude_self=True` removes the positional diagonal and requires a square
matrix; self terms are included by default. This is a positional rule for
explicit X/Y as well as self calls.

Min/max return values, propagate NaNs, and select the first logical row-major
entry on ties. Empty sums/products/all/any use 0/1/True/False; empty means are
NaN, and empty min/max raise. `all`/`any` use PyTorch's nonzero truth rule,
including for numeric distances from Boolean inputs. Arbitrary tile predicates
remain on the existing explicit Python callback reduction path.

Float32/float64 `prod` uses PyTorch's entry-level derivative convention: if the logical
input contains a zero, derivatives use products excluding the differentiated
entry; otherwise they use the total divided by that entry. The zero test spans
output rows and broadcast batches, as in PyTorch. Separate zero/Inf/NaN counts
preserve zero-times-infinity and zero-times-NaN behavior; ignored diagonal
entries are identities and do not trigger this test. Joint N/M products reduce
entries directly rather than differentiating nested row products. The reference
for joint axes is `dense.flatten(-2).prod(-1)`, preserving batches. At nonfinite
values, PyTorch's dimensionless `torch.prod(dense)` can have different gradients.

Forward product rounding depends on tile order. Finite partial products can still
overflow or underflow; this is not a full-range exclusive-product guarantee.
Nonlinear source derivatives retain their own nonfinite limits. Previously
exported/compiled graphs containing legacy `_streaming_reduce` or
`_expression_consume` product cores keep their old numerical behavior; recompile
to select the corrected `_expression_product` core. Existing schemas and default
auxiliary layouts remain available for those graphs.

## Preparation, Memory and Deployment

The native prepared comparison is shared with existing row reducers. Global
variance/inverse covariance for self standardized-Euclidean/Mahalanobis calls
is inferred once from the original collection, before tiling or entry selection.
Explicit Y still requires V/VI, matching the named metrics. No gradient is taken
through inferred statistics. Explicit differentiable tensor parameters keep
their existing gradients. Repeated calls prepare again, so tensor mutations are
visible; there is no hidden matrix or cross-call feature cache.

`chunk_size` bounds sample and product/form channel tiles. Cloud comparisons
can additionally use `metric_chunk_size` for their internal point tiles, keeping
these two limits separate. Other metric options retain their named meanings,
including `batched`, `feature_dim`, lengths and MMD's `kernel` selector. Shuffled
Hausdorff search/generators are not supported by streaming execution.

Forward and first backward are native C++/ATen. Backward replays comparison
tiles and accumulates input, tensor-parameter and coefficient gradients. It
retains no dense comparison or intermediate `A @ R` for bilinear forms. Natural
input/gradient/output memory remains; min/max/product/global reductions also
retain row-sized state, and each comparison has its own inner workspace. Tile
width is not a fixed total-memory budget. Higher derivatives are unsupported.

CPU/CUDA float32/float64 and autocast promotion follow the named comparisons.
The API supports `torch.compile` and `torch.export`; use the explicit `matmul`
method in compiled/exported functions and construct the binding
inside `forward` for dynamic inputs. Standalone C++ exposes `PairwiseOperator`,
`pairwise_operations` and `PairwiseOperationOptions` from
`<ops/pairwise_operations.h>`. C++ supplies the existing comparison
name and its kernel flag, with metric settings in `options.comparison`.

See [the example](../examples/pairwise_operators.py).

## Top-K Selection

All scalar operators and expressions support the same terminal selection:

```python
A = tp.lazy(X, Y, metric="euclidean")
nearest = A.topk(8, dim=-1, largest=False)
# nearest.values, nearest.indices: [..., N, 8], indices refer to Y
best_queries = A.topk(3, dim=-2, largest=False)
# [..., 3, M], indices refer to X
H = tp.residual_operator(X, Y, transform="abs").max(dim=-1)
values, indices = H.topk(8, largest=False)  # minimax residual neighbors
```

`dim` is one integer selecting N or M, with positive or negative indexing.
Like `torch.topk`, selection replaces that axis by k and returns a named pair
of tensors (`values`, int64 `indices`). Lists/tuples and batch axes are rejected.
Indices refer to the selected logical axis, including after `.T`; the operation
does not change a directed metric's formula. Equal values choose lower axis
indices. NaNs rank above finite values: first for largest, last for smallest.
`sorted=False` permits a sorted result. This deterministic tie rule is stronger
than PyTorch's unspecified tie ordering.

`exclude_self=True` removes the positional diagonal and requires a square
operator. k must lie between zero and the eligible axis length, inclusive.
Empty axes support k=0. Boolean/integer sources preserve their entry dtype;
indices are nondifferentiable. Floating values propagate first derivatives only
through selected entries; higher derivatives are unsupported.

Tiles are merged with the best k candidates so far, without retaining a full
computed N*M matrix. Output storage is intrinsically N*k (or k*M); choosing a
large k can therefore require substantial memory. Selection is terminal, so
subsequent tensor operations do not fuse back into the source. Feature-axis
selection on residual N*M*D and image event sources is also available. All
variants accept one integer axis; multi-axis top-k is omitted, following PyTorch.
Exact [median/quantile](operator-statistics.md#exact-median-and-quantile) over one
scalar sample axis sort complete rows in budgeted blocks and retain two witness
indices per result; they do not use constant candidate storage.

Native callers use `ScalarExpression::topk(k, dim, largest, sorted, exclude_self)`
and receive `std::tuple<Tensor, Tensor>`. Family expression factories and
feature-contracted residual sources share the implementation. Existing typed
family facades retain their original signatures; use their expression factory
for the expanded consumer interface. Built-in forward/backward are native with
fake metadata for `torch.compile` and `torch.export`.
