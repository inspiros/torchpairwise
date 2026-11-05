# Streamed Scalar Operator Algebra

For source selection and shared axis rules, start with the [operator guide](operators.md).

Scalar algebra combines comparison entries before a matrix consumer runs. The
[residual feature source](residual-operators.md) and
[image-map event source](image-map-operators.md) can contract their event axes
into scalar sources before this algebra. It
supports mixed distance affinities, mixtures/products of kernels, weighted
interactions and kernel alignment without retaining the dense expression matrix.

```python
rbf = tp.lazy(X, Y, metric="rbf", gamma=.5, chunk_size=64)
laplacian = tp.lazy(X, Y, metric="laplacian", gamma=.3, chunk_size=64)
mixture = alpha * rbf + (1-alpha) * laplacian
features = mixture.matmul(V)
alignment = (rbf * laplacian).sum()
```

The entries and product are

\[
C_{ij}=\alpha K^{\mathrm{rbf}}_{ij}+(1-\alpha)K^{\mathrm{laplacian}}_{ij},
\qquad R_{iq}=\sum_j C_{ij}V_{jq}\]

`alpha` can be a Python number or a zero-dimensional tensor with gradients.
`alignment` computes \(\sum_{ij}K^{\mathrm{rbf}}_{ij}K^{\mathrm{laplacian}}_{ij}\)
per batch; it is the unnormalized entrywise inner product, without centering.
See the [example](../examples/operator_algebra.py).

## Operations and Shapes

Supported expression constructors are `+`, `-`, entrywise `*`, unary negation,
`square()`, `exp()`, `weight_rows(r)` and `weight_columns(c)`. An expression
represents scalar entries `[...,N,M]`; feature axes have already been consumed
by its comparisons. There is no generic vector-valued expression or implicit
feature reduction in this API.

All existing scalar source families can be leaves, including structured/media/
ordered comparisons and [caller-owned matrices](matrix-operators.md). Different
sources must have equal positional N/M counts; the caller ensures sample order
agrees. Source, weight and consumer batch axes broadcast. Sources and tensor
factors share a device. Numeric entry operations follow PyTorch promotion;
integer source entries retain integer arithmetic until an operation promotes
it. The entry dtype does not necessarily equal a consumer's output dtype.

Unrestricted tensor broadcasting is deliberately excluded from `*`:

```python
D = tp.lazy(X, Y, metric="sqeuclidean", chunk_size=64)
C = (-gamma * D).exp().weight_rows(r).weight_columns(c)
result = C.matmul(V)
```

Here \(C_{ij}=r_i\exp(-\gamma D_{ij})c_j\), with `r:[...,N]` and
`c:[...,M]`. Signed weights are allowed. A tensor scalar must be zero-dimensional;
for pairwise tensor weights, use `C * tp.as_lazy(weights)` explicitly.
Low-precision source/factor tensors require autocast, which promotes them to
float32. Plain Python numbers follow the usual weak scalar promotion rules.
Complex and sparse inputs are outside the current contract.

Arithmetic produces a lazy pairwise expression: generic
combinations need not preserve symmetry, positivity or positive semidefiniteness.
Kernel-specific displacement and residual geometry are therefore not inherited.
`.T` is logical transpose, including for directed comparison sources and row/
column weighting; it does not exchange a directed metric's input roles.

## Entrywise functions and masks

Expressions also support division and powers (`/`, `**`), `minimum`, `maximum`,
`clamp`, `sqrt`, `log`, `sin`, `cos`, `tanh`, `sigmoid` and `abs`. Their
`torch` function forms work on lazy operands. Dtypes follow the corresponding
PyTorch operations, including weak Python scalar promotion, floating results
for integer true division and square root, and boolean comparison results.
Clamp bounds must be Python numbers or zero-dimensional tensors.

Comparisons `<`, `<=`, `>`, `>=`, `.eq()` and `.ne()` produce boolean
expressions. Combine masks with `&`, `|` and `~`, or the logical methods.
Equality operators retain object identity semantics. `a.where(condition, b)`
and `torch.where(condition, a, b)` select entries as in PyTorch. Non-scalar
dense tensor branches must be wrapped with `tp.as_lazy`.

`a.row_index()` and `a.column_index()` give global int64 sample indices;
`a.row_values(r)` and `a.column_values(c)` broadcast vectors over the opposite
axis. Their `like` expression supplies metadata only. `a.add_rows(r)` and
`a.add_columns(c)` add these vectors to entries, with batch broadcasting and
gradients to the vectors. For example:

```python
i, j = a.row_index(), a.column_index()
causal = a.where(j <= i, -float('inf'))
same_group = a.where(a.row_values(row_labels).eq(a.column_values(col_labels)), 0.)
```

Gradients follow PyTorch, including `torch.where`: an unselected branch with
an infinite derivative can still produce NaN gradients. Use the usual
double-where pattern to make that branch's inputs safe before evaluating it.

KeOps lowers division, scalar powers, the listed unary functions where
available, scalar clamp, and integer index/value comparisons and masks for
supported unbatched graphs. Integer values must be exactly representable in
the formula dtype; unsupported graphs fall back to native. Clamp and where
fall back when gradients are required. Fused CUDA reducers never execute the
new nodes.

## Consumers and Memory

Expressions share the [common scalar consumers](operators.md#common-scalar-consumers):
products/forms, blocks/diagonal/top-k, ordinary and weighted statistics, and exact
single-axis median/quantile. See
[operator statistics](operator-statistics.md) for expanded and normalized weighted statistics. `*` multiplies entries;
`matmul`/`@` contracts against a dense tensor. Multiplying two lazy operators as
matrices is not supported. Consumer results are ordinary tensors; subsequent
PyTorch operations do not automatically become streaming expression nodes.

Native preparation runs once per distinct leaf/orientation in a consumer call,
including whole-collection statistics and reusable media/ordered features.
Repeated expression nodes are evaluated once per tile/orientation. First
backward replays tiles, differentiates the local expression, and accumulates
input, parameter, factor, weight and coefficient gradients. No Python numerical
callback or backward orchestration is involved; higher derivatives are unsupported.
Within each backward tile, the consumer and expression gradient share the
evaluated source values and their native gradient graph. Those values are
released when the tile finishes; they are not cached between tiles or calls.
Softmax still makes a separate pass to compute its row centering term before
the pass that accumulates source gradients.
Inputs are retained by reference; subsequent calls see mutations. There is no
persistent numerical cache or retained dense comparison matrix. Natural inputs,
gradients, consumer outputs and reducer state still occupy memory. Tile workspace
also grows with expression size: this is not an unlimited constant-workspace
formula compiler. The default sample/channel tile width is the smallest source
width; explicit widths above 128 are honored.

Empty, exclusion, NaN/tie and reduction-axis rules follow the scalar consumer
contract. Float32/float64 products retain separate exceptional-factor state and follow
PyTorch's zero-dependent entry derivative convention, including cross-row/batch
zeros and joint sample reductions; see [product limits and legacy graphs](pairwise-operators.md#reductions).
Finite partial overflow/underflow and nonfinite source chain rules remain outside
a universal accuracy guarantee. Skewness, kurtosis and
event median/quantiles remain future work; scalar top-k and exact single-axis
median/quantile are supported. Planned statistics and boundaries are described in the
[design](operator-algebra-plan.md).

Construct expressions inside `forward` and use explicit `.matmul()` for
`torch.compile` and `torch.export`. The native C++ API is declared in
`<ops/expression_operations.h>`:

```cpp
auto a = torchpairwise::ops::pairwise_expression(x, y, "rbf", true);
auto b = torchpairwise::ops::pairwise_expression(x, y, "laplacian", true);
auto mixture = a.scale(alpha) + b.scale(0.7);
auto result = mixture.matmul(values);
auto alignment = (a * b).reduce("sum");
```

Typed factories also expose structured, media, ordered and supplied-matrix
sources. Standalone execution checks are deferred to review.
