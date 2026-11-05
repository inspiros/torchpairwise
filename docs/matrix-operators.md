# Caller-Owned Matrix Operators

For source selection and shared axis rules, start with the [operator guide](operators.md).

`tp.as_lazy` gives a dense scalar tensor the same matrix operations as
`tp.lazy`:

```python
A = tp.as_lazy(scores, chunk_size=128)  # scores: [...,N,M]
product = A.matmul(values)      # [...,N,C]
form = A.bilinear(left, right)  # left.T @ scores @ right
column_min = A.amin(-2)
block = A.block(rows, columns)
reverse = A.T.matmul(other_values)
```

The caller has already allocated `scores`. This binding stores a reference,
adds no persistent copy, and does not remove that input allocation. Its purpose
is to reuse the same algebra for user-provided matrices, external native
operators and library comparisons. For built-in comparisons, prefer
`tp.lazy(X,Y,metric=...)` to avoid constructing the matrix in the first
place. This source can participate in [scalar algebra](operator-algebra.md). Arbitrary
user tile callbacks and operator matrix products remain future work.

## Shared Operations

The common Python consumers are shared with scalar pairwise operators, including
softmax products, weighted statistics and top-k. See the canonical
[operation coverage](pairwise-operators.md#operation-coverage) rather than a
separate matrix-only list. `reduce` selects a reducer, `@` aliases `matmul`, and
`.T` is a logical transpose view. Reduction axes, exclusion, empty results,
ties, coefficient shapes and batches follow that shared contract.

Only dense strided real-valued tensors `[...,N,M]` are accepted. Source batch
axes and coefficient batch axes broadcast. Noncontiguous sources and repeated
block indices retain correct gradients. Subsequent calls see tensor mutations.

Float32/float64 sources retain their dtype. Half/bfloat16 require autocast,
which promotes source values and low-precision coefficients to float32. Standard
uint8/int8/int16/int32/int64 and bool sources retain their entry dtype for blocks,
diagonals, min/max and top-k. Their sums/products accumulate in int64; means accumulate in int64 before
float64 division, and all/any always return bool, including for uint8 inputs. Ordinary integer overflow rules apply. Integer
and bool products/forms accept floating float32/float64 coefficients; only the
coefficients receive gradients. `A.dtype` describes source entries, not every
possible reduction or contraction result. Complex and sparse inputs are outside
this interface's current contract.

Forward and first backward run in shared registered C++/ATen consumers. Bilinear
forms do not retain an additional full `A @ right` intermediate. The source,
requested output, natural dense source gradient and per-tile workspaces still
need storage; this is not a total-memory limit. Higher derivatives are unsupported.
Global products of noncontiguous sources may also need a temporary contiguous
copy to preserve PyTorch's product and gradient semantics.

Construct inside `forward` and use `.matmul()` for `torch.compile`/`torch.export`.
Standalone C++ exposes `MatrixOperator`, `matrix_operations` and
`MatrixOperationOptions` from `<ops/matrix_operations.h>`.
Use `matrix_expression` for the expanded native ScalarExpression consumers.
Standalone execution checks are left to review. See the
[runnable example](../examples/matrix_operators.py).
