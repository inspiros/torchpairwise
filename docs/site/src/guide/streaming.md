<div class="eyebrow">Streamed operators</div>

# Sources and Consumers

A source describes a logical array. A consumer evaluates tiles and combines them
into the requested Tensor. For scalar comparisons, the logical shape is
\([\ldots,N,M]\). No dense generated matrix is retained merely by binding it.

Use `tp.lazy` to bind scalar comparisons in Python and
`torchpairwise::streaming` in C++. Custom metrics and feature comparisons can
also be passed to `tp.lazy`.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X: [N,D], Y: [M,D], V: [M,C]
A = tp.lazy(X, Y, metric="sqeuclidean")
row_sum = A.sum(dim=-1)                  # [N]
values, indices = A.topk(8, largest=False)  # [N,8] each
product = A.matmul(V)                   # [N,C]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X:[N,D], Y:[M,D], V:[M,C]
namespace s = torchpairwise::streaming;
s::PairwiseOptions options;
options.chunk_size = 128;
auto A = s::pairwise(X, Y, "sqeuclidean", options);
auto row_sum = A.reduce("sum", -1);        // [N]
auto [values, indices] = A.topk(8, -1, false); // [N,8] each
auto product = A.matmul(V);                // [N,C]
```

</div>
</div>

The examples assume at least eight references. The language switch changes the
frontend spelling; mathematical results and native algorithms are shared.

## Source Families

| Source | Logical array | Python | C++ |
|---|---|---|---|
| Scalar vector/cloud | `[...,N,M]` | `tp.lazy` | `streaming::pairwise` |
| Kernels | `[...,N,M]` | `tp.lazy` | `streaming::kernel` with vector/structured options |
| Gaussian/covariance/mixture | `[...,N,M]` | `tp.lazy` | `streaming::pairwise` with structured options |
| Images/audio | `[...,N,M]` | `tp.lazy` | `streaming::pairwise` with media options |
| Sequences/tokens/MMD statistic | `[...,N,M]` | `tp.lazy` | `streaming::pairwise` with ordered options |
| Existing caller matrix | `[...,N,M]` | `tp.as_lazy` | `streaming::matrix` |
| User-defined scalar comparison | `[...,N,M]` | `tp.lazy(x, y, comparison)` | `custom_expression` with registered user operators |
| Encoded features with a head | `[...,N,M]` | `tp.lazy(x, y, feature_comparison)` | Run the encoder, then bind the feature tensors |
| Feature residuals | `[...,N,M,D]` | `tp.residual_operator` | `streaming::residual` |
| Image maps | `[...,N,M,C,Hout,Wout]` | `tp.image_map_operator` | `streaming::image_map` |

C++ family factories use typed family options; the common `ScalarExpression`
consumer interface applies after binding. See [native options](cpp.md).
Wrapping an existing matrix does not remove its input storage.

## Shared Scalar Consumers

| Purpose | Methods |
|---|---|
| Products and forms | `matmul`, `softmax_matmul`, `bilinear`, `quadratic`, `frobenius_inner_product`, `hsic` |
| Selected entries | `diagonal`, `block`, `topk` |
| Reductions | `sum`, `mean`, `min`, `max`, `prod`, `all`, `any` |
| Statistics | `var`, `std`, `logsumexp`, `count_nonzero`, `weighted_mean`, `weighted_var`, `weighted_std`, `median`, `quantile` |

These are 25 scalar consumers. Python method names mirror Tensor operations;
C++ uses `reduce("sum",...)` and named statistics on ScalarExpression.
Top-k returns values and indices. Python `min(dim)` and `max(dim)` return
values and indices; `amin` and `amax` return values. Unlike PyTorch,
`median(dim)` returns values only.
See [reductions](statistics.md) for dimensions and defaults.

Python also provides `argmin`, `argmax`, `amin`, `amax` and `softmax`. Reductions,
selections and `softmax_matmul` accept `exclude_self` and, where meaningful, a
[pair mask](statistics.md#pair-masks). Entrywise lazy objects support these consumers. Linear-only objects, including
products of lazy objects, softmax, identity, diagonal and block maps, group
projections and preconditioners, support `@`, `.mT`, `sum`, `mean`, scalar
multiplication, addition and subtraction with lazy objects, and the solvers.
Other consumers raise with a `.to_dense()` hint.

## Execution Backends

Tiled native execution is the default and is shared by Python, C++, compilation
and export. Two accelerations are layered on top of it:

- **Fused CUDA reducers** apply automatically to float32 and float64 CUDA
  sources with 1 to 64 features. They cover `sqeuclidean`, `manhattan`,
  `chebyshev`, `minkowski` with p other than 2, `rbf`, `laplacian`, `matern`,
  `rational_quadratic`, `linear`, `polynomial` and `sigmoid` with scalar
  parameters (selections on the last three use tiles), under `matmul`, sum, mean, `logsumexp`, `softmax_matmul` and top-k
  or min/max selection, including number, row and column offsets, weights,
  `exp` and row/column boolean masks. They evaluate each pair and reduce it in
  one kernel, so `chunk_size` is ignored; `tp.masks` pair masks use tiles.
- **KeOps** is an optional Python backend for eager, unbatched float32/float64
  vector distances, kernels and scalar expressions. Install it with
  `pip install "torchpairwise[keops]"`. Select it per context or per source:
  `"native"`, `"keops"` or `"auto"`. Unsupported sources, options and consumers
  fall back to native execution.

```python
with tp.backend("keops"):
    K = tp.lazy(X, Y, metric="rbf", gamma=0.5)
    result = K.matmul(V)                        # one KeOps reduction

K = tp.lazy(X, Y, metric="rbf", gamma=0.5, backend="native")  # leaf override
```

`backend="keops"` raises when KeOps is not installed, while `"auto"` falls back
silently and only selects large float32 CUDA workloads. `torch.compile`,
`torch.export`, standalone LibTorch and `tp.linalg` solvers always run native
code. Results agree within floating tolerance, not bitwise, and selection tie
order is unspecified under `"keops"`. See the
[fused reducer rules](https://github.com/inspiros/torchpairwise/blob/master/docs/operators.md#fused-cuda-reductions)
and [KeOps backend contract](https://github.com/inspiros/torchpairwise/blob/master/docs/operators.md#optional-keops-backend).

## Memory and Chunking

`chunk_size` controls outer sample tiles. `feature_chunk_size` slices residual
features; metric/component controls bound inner cloud or mixture work. Widths
are arbitrary positive integers up to int64max, not powers-of-two presets.
They are not a total-memory budget.

Inputs, prepared features, output buffers and gradients remain in memory.
Exact median/quantile may sort whole rows in budgeted row blocks. Training uses
bounded replay for supported sources: save source inputs and compact state,
recompute source tiles during backward, accumulate their adjoints. Recomputing
costs time; choose by workload rather than assuming streaming is always faster.

Bindings store references and options, not feature/matrix caches. Each consumer
recomputes; input mutations are visible. Ordinary Tensor operations after a
consumer do not stream back into its source. This is a finite comparison and
consumer API, not an arbitrary lazy PyTorch execution system.
