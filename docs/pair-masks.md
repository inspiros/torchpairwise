# Pair Masks

```python
import torchpairwise as tp

D = tp.lazy(X, metric="sqeuclidean")                        # X:[N,D], labels:[N]
positives = tp.masks.same_group(labels) & ~tp.masks.diagonal()
negatives = tp.masks.different_group(labels)

hardest_positive = D.amax(-1, mask=positives)               # [N]
hardest_negative = D.min(-1, mask=negatives)                # values, indices [N]
mean_positive = D.mean(-1, mask=positives)                  # normalized by eligible count
attention = (-D).softmax_matmul(V, mask=tp.masks.causal())  # V:[N,C]
graph = tp.lazy(X, metric="rbf", gamma=.5).masked(D < radius ** 2)  # lazy masked kernel
```

A mask selects the eligible `(i, j)` entries of a logical `[...,N,M]` matrix
without materializing it. Pass it as `mask=` to a scalar consumer, or bind it
with `A.masked(mask, value=0)` to obtain a lazy expression for products, solvers
and further algebra.

## Mask Constructors

| Constructor | Keeps |
| --- | --- |
| `same_group(labels, other=None)` | `labels[i] == other[j]` |
| `different_group(labels, other=None)` | `labels[i] != other[j]` |
| `causal(offset=0)` | `j <= i + offset` |
| `band(lower=0, upper=0)` | `-lower <= j - i <= upper` |
| `diagonal()` | `i == j` |

`labels:[...,N]` labels rows and `other:[...,M]` labels columns. With `other`
omitted, the operand must be square and `labels` labels both axes. Mask batch
axes (of labels, lazy or dense masks) must broadcast to the operand's batch
shape; a mask cannot add batch axes. Labels compare exactly, so use integer or
Boolean labels. Combine masks with `&`, `|` and `~`.
Any Boolean lazy expression with the operand's N/M axes is also a mask, for
example `D < radius` or `tp.lazy(X, Y, "cosine") > .5`. So is a dense Boolean
`[...,N,M]` tensor, which is then caller-owned memory.

A mask has no shape of its own. It binds the axes of the operand it is applied
to, so `A.T.sum(-1, mask=m)` labels the rows of `A.T`.

## Consumer Semantics

A masked consumer computes the dense reference
`consumer(torch.where(mask, A, fill))`. The fill removes an entry from that
reduction:

| Consumer | Fill / normalization |
| --- | --- |
| `sum`, `count_nonzero`, `masked` default | `0` |
| `prod` | `1` |
| `all` / `any` | `True` / `False` |
| `amin`, `min`, `argmin`, `topk(largest=False)` | `+inf` (integer: the dtype maximum) |
| `amax`, `max`, `argmax`, `topk(largest=True)` | `-inf` (integer: the dtype minimum) |
| `logsumexp`, `softmax`, `softmax_matmul` | `-inf` |
| `mean`, `var`, `std`, `weighted_*` | mask as 0/1 confidence weights: the eligible count normalizes; `correction` keeps its meaning |

The rest of each consumer's contract is unchanged, including dims, keepdim,
batch axes and `exclude_self` (which composes with a mask).
`median`/`quantile` and predicate `all`/`any` reject masks.

Rows with no eligible entry return the reduction identity: sum `0`, mean `NaN`,
amin `+inf`, logsumexp `-inf`. Unlike unmasked `min`/`max`, no error is raised.
A row of a masked softmax with no eligible entry follows `torch.softmax` of
`-inf` scores.

**Masks are fills, not structural exclusion.** `exclude_self` removes the
diagonal structurally: an excluded item never displaces a valid candidate. A
masked entry instead holds its fill value. When a row has fewer than `k`
eligible entries, `topk` returns masked positions with fill values. Valid
entries equal to the fill (for example a genuine `+inf` distance under `min`)
tie with masked ones and follow the usual lower-index tie rule. Check
`values.isfinite()` when rows may have too few eligible entries.

## Gradients, Execution and Backends

Gradients follow `torch.where`: masked positions receive zero cotangents, and
label tensors receive none. The usual `torch.where` caveat applies to `masked`
expressions built on unsafe branches; see
[operator algebra](operator-algebra.md).

Masks lower to native expression nodes for row/column indices and values,
comparisons, logical operators and `where`. They therefore stream on CPU and
CUDA, compile and export like other scalar expressions, and reach `tp.linalg`
through `masked`. Masked consumers use tiled execution: the
[fused CUDA reductions](operators.md#fused-cuda-reductions) do not take masked
expressions. With the optional KeOps backend, integer label and index masks
(including `causal` and `band`) lower to exact KeOps conditions. Labels must be exactly representable in the
formula dtype. Floating-point comparisons fall back to native execution, as do
masked `logsumexp`, `softmax` and `softmax_matmul`: their `-inf` fill is not
safe in KeOps' streaming exponential sums.

See the [example](../examples/pair_masks.py) for supervised contrastive loss,
hard negative mining, a Borůvka nearest-component step and causal kernel
attention, each checked against dense references.
