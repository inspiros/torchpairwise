<div class="eyebrow">Streamed operators</div>

# Reductions and Statistics

For scalar source \(A[\ldots,N,M]\), dim=-1 selects references M, dim=-2
queries N, and dim=None selects both sample axes. Batches persist. Basic
reducers and moments accept dimension lists/tuples for one joint reduction;
`keepdim=True` retains reduced axes. Python uses `.sum(...)`; C++ uses
`.reduce("sum",...)` on ScalarExpression.

| Operation | Result / convention |
|---|---|
| `sum`, `mean`, `prod` | Numeric values; operation-specific empty-axis identities |
| `min`, `max` | With an integer dimension, values and indices; first eligible reference on ties |
| `amin`, `amax` | Values only, propagating NaNs |
| `argmin`, `argmax` | Indices only |
| `all`, `any` | Boolean values under PyTorch nonzero truth |
| `var`, `std` | correction=1 by default |
| `weighted_mean`, `weighted_var`, `weighted_std` | Nonnegative confidence weights, normalized over the selected axis; correction=0 by default |
| `logsumexp`, `count_nonzero` | Stable log-sum-exp / integer nonzero count |
| `topk` | One integer axis; largest=True by default; values and indices |
| `median`, `quantile` | One integer sample axis, default -1; values only; scalar q in [0,1] |

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
A = tp.lazy(X, Y, metric="sqeuclidean")         # [N,M]
mean = A.mean(dim=-1)                           # [N]
spread = A.var(dim=(-2, -1), correction=0)      # scalar per batch
cutoff = A.quantile(0.9, dim=-1)                # [N]
weighted = A.weighted_mean(confidence, dim=-1)  # confidence:[M], result:[N]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
auto A = tp::pairwise_expression(X, Y, "sqeuclidean");
auto mean = A.reduce("mean", -1);  // [N]
auto spread = A.statistics("var", at::IntArrayRef{-2, -1}, false, false, 0.);
auto cutoff = A.quantile(0.9, -1);                // [N]
auto weighted = A.weighted_mean(confidence, -1);  // confidence:[M] -> [N]
```

</div>
</div>

## Selection Details

Top-k accepts one integer axis, as PyTorch does. Stable ties use smaller original
indices; NaNs have high ordering, so largest=True puts them first.
Operators include self unless `exclude_self=True`; exclusion
removes positional i=j entries from a square comparison, including separate X/Y
inputs.

Unlike PyTorch, `median(dim)` returns values only. Median selects the lower middle value. Quantile supports linear, lower, higher,
nearest and midpoint interpolation, with a deterministic witness policy. Exact
selection stores compact witnesses but sorts complete reduction-axis rows in
budgeted blocks; workspace scales with row length. It is not a constant-memory
approximate quantile sketch.

## Pair Masks

A pair mask selects the eligible \((i,j)\) entries of the logical `[...,N,M]`
source without materializing it. Pass it as `mask=` to a scalar consumer, or bind
it with `A.masked(mask, value=0)` to get a lazy expression for products, solvers
and further algebra. A masked consumer equals the dense consumer applied to
`torch.where(mask, A, fill)`: the fill removes an entry from that reduction
(`0` for sums, `+inf` for minima, `-inf` for maxima and `logsumexp`). Means,
variances and weighted statistics instead divide by the eligible count.
`median`, `quantile` and predicate `all`/`any` reject masks. A row without any
eligible entry returns the reduction identity, such as `NaN` for a mean, instead
of raising.

The `tp.masks` constructors are `same_group(labels, other=None)`,
`different_group(labels, other=None)`, `causal(offset=0)`, `band(lower=0, upper=0)`
and `diagonal()`. Combine masks with `&`, `|` and `~`. Any Boolean lazy
expression with the same axes, such as `D < radius`, and any dense Boolean
`[...,N,M]` tensor are masks too.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], labels:[N] int64, V:[N,C]
D = tp.lazy(X, metric="sqeuclidean")
positives = tp.masks.same_group(labels) & ~tp.masks.diagonal()
negatives = tp.masks.different_group(labels)
hardest_positive = D.amax(-1, mask=positives)               # [N]
hardest_negative, nearest = D.min(-1, mask=negatives)       # values, indices [N]
mean_positive = D.mean(-1, mask=positives)                  # divided by eligible count
attention = (-D).softmax_matmul(V, mask=tp.masks.causal())  # [N,C]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
#include <limits>
namespace tp = torchpairwise::ops;
// X:[N,D], labels:[N] int64, V:[N,C]. Bind masks, then fill explicitly.
constexpr double inf = std::numeric_limits<double>::infinity();
auto D = tp::pairwise_expression(X, c10::nullopt, "sqeuclidean");
auto positives = tp::same_group_mask(D, labels)
                     .logical_and(tp::diagonal_mask(D).logical_not());
auto negatives = tp::different_group_mask(D, labels);
auto hardest_positive = D.masked(positives, -inf).reduce("max", -1);  // [N]
auto hardest_negative = D.masked(negatives, inf).reduce("min", -1);   // [N]
auto mean_positive = D.masked(positives).reduce("sum", -1) /
                     positives.reduce("sum", -1);  // [N]
auto attention = D.scale(-1.).masked(tp::causal_mask(D), -inf)
                     .softmax_matmul(V);  // [N,C]
```

</div>
</div>

Masks are fills, not structural exclusion: when a row has fewer than `k`
eligible entries, `topk` returns masked positions holding fill values, so check
`values.isfinite()`. `exclude_self` still removes the diagonal structurally and
composes with a mask. Gradients follow `torch.where`; label tensors receive none.
Masked consumers use tiled execution rather than the fused CUDA reductions. See
the [pair mask notes](https://github.com/inspiros/torchpairwise/blob/master/docs/pair-masks.md)
for the complete consumer table, gradient rules and KeOps lowering.

## Event Axes versus Sample Axes

Event-valued sources expose more axes. Reducing only event axes keeps a lazy
source; selecting N or M returns a Tensor. Operation order is preserved, and
statistics are not generally associative: a mean of local variances is different
from one joint variance. See [feature and image axes](events.md).

For empty exclusions, nonfinite values, precision and all interpolation details,
see the [statistical contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/operator-statistics.md).
