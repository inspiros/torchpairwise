# Streamed Kernel Operators

For source selection and shared axis rules, start with the [operator guide](operators.md).

Bind tensors and kernel options once, then request products, contractions or
aggregates without constructing the full Gram matrix:

```python
K = tp.lazy(X, Y, metric="rbf", gamma=0.2)
product = K @ V                     # [..., N, R]
energy = K.bilinear(u, v)           # [...] for u:[N], v:[M]
cross = K.bilinear(L, R)            # [..., P, Q]
total = K.sum()                     # [...], both sample axes
average = K.mean(dim=-1)            # [..., N], reference axis
columns = K.sum(dim=-2)             # [..., M], query axis
diagonal = K.diagonal()             # [..., min(N, M)]
block = K.block(rows, columns_ids)  # selected dense block
reverse_product = K.T @ L

self_kernel = tp.lazy(X, metric="polynomial", degree=3, gamma=.2)
quadratic = self_kernel.quadratic(u)  # u.T @ K(X,X) @ u
off_diagonal_mean = self_kernel.mean(exclude_self=True)
```

Kernel sources share all
common matrix consumers. It accepts all thirteen named kernel comparisons:
the twelve vector kernels from `pairwise_kernels`, plus
`gaussian_mixture_overlap_kernel` with tuple inputs `(means,covariances,weights)`.
Both its full name and `gaussian_mixture_overlap` alias are accepted.
Vector kernel names and parameter validation match `pairwise_kernels`.
`quadratic` requires equal sample counts; an explicit cross collection is
allowed, and the result is not assumed positive.

## Shapes and Memory

Vector inputs have shapes `[...,N,D]` and `[...,M,D]`. Gaussian-mixture overlap
uses the [mixture collection contract](gaussian-mixtures.md). Coefficient matrices have shapes
`[...,N,P]` and `[...,M,Q]`. All batch axes broadcast. Plain vectors `[N]` and
`[M]` remove the corresponding output channel axis; represent batched vectors
as `[...,N,1]` or `[...,M,1]`. A vector product `K @ v` returns `[...,N]`.

Forms tile both sample axes and coefficient channels, accumulating the requested
output directly. They retain the four operands for backward and recompute
kernel tiles; neither a full Gram matrix nor a full intermediate `K @ R` is
saved. Outputs and operand gradients still require their natural storage.
`chunk_size` bounds the sample/channel tile widths, rather than total memory.
Global scalar reducers currently compose row reductions and retain O(N) row
state, plus kernel tile workspace. Selecting a block materializes that block.

The facade stores tensor references and options only. Mutating a bound input
changes subsequent results; there is no hidden feature or kernel cache. Each
method evaluates the kernel again. `.dtype` follows the pairwise facade: the comparison dtype, promoted to
float32 inside autocast for half/bfloat16 inputs.

## Reduction Contracts

All common scalar reductions and exact median/quantile are inherited, including
var/std/logsumexp/count_nonzero and normalized weighted statistics; see [statistics](operator-statistics.md).
The seven basic reducers `sum`, `mean`, `min`, `max`, `prod`, `all` and `any` accept `dim=None` (both
sample axes), `dim=-1` (references), or `dim=-2` (queries). Equivalent positive
indices and tuple/list dimensions are supported. Batch axes are
preserved. `keepdim=True` retains the reduced sample axes. `all` and `any` use
PyTorch's nonzero truth rule. Empty sum/product/all/any return 0/1/True/False;
empty mean returns NaN; reducing an empty axis with min/max raises. Min/max
propagate NaNs and route ties to the first position along the reduced axis,
with row-major order for a global reduction.

`exclude_self=True` omits the positional diagonal and requires a square
operator, including when X and Y are different tensors. Block indices must be
one-dimensional int64 tensors on the input device; repeated indices are valid
and bounds follow `torch.index_select`.

Lazy reductions accept `dim=-1` and `dim=None`,
`dim=-2` and `reduction="mean"`. Arbitrary Python predicates remain available
only on its existing reference-axis reducers; the operator facade accepts
native built-in reductions.

## Deployment

Numerical forward and backward execute in registered C++ operators. Python and
standalone LibTorch share the same kernel catalog. CPU/CUDA float32 and float64
are supported, with float16/bfloat16 promotion under autocast. First derivatives
are supported; higher derivatives of streamed forms are explicitly rejected.
Construct the facade inside `forward` for `torch.compile` or `torch.export`:

```python
class Objective(torch.nn.Module):
    def forward(self, x, coefficients):
        return tp.lazy(x, metric="rbf", gamma=.2).quadratic(coefficients)
```

Use `K.matmul(V)` inside compiled/exported functions. It has the same numerical
behavior as `K @ V`; the tested PyTorch Dynamo versions do not trace `@` on
ordinary Python facade objects. The explicit methods avoid that frontend limit.

C++ vector-kernel callers can use `torchpairwise::ops::KernelOperator`, a
`PairwiseOperator` specialization with `transpose()`, from
`<ops/kernel_operations.h>`, or the exported free function
`kernel_operations` with `KernelOperationOptions`. Structured C++ inputs retain
the typed `StructuredPairwiseOperator`/`structured_pairwise_operations` API.

## Vector Interactions

The kernel specialization adds two consumers for vector inputs: displacement
and residual outer products. They require coordinates, so structured mixture
inputs reject these methods with a clear error. A weighted displacement sum
needs no `[N,M,D]` tensor:

```python
displacement = K @ Y - X * K.sum(dim=-1, keepdim=True)
```

This is useful for geometric updates and message passing. Use `K.displacement()` or `K.displacement(normalize=True)` for a native
weighted sum or normalized field. Streamed entropic transport maps are also
available; see [interaction maps](interaction-maps.md). General kernel gradients
and arbitrary vector-valued interaction operators remain future extensions. See the runnable [example](../examples/kernel_operator.py).

Residual outer-product sums, normalized second moments and centered local
covariance are available through `K.residual_outer_product`; see
[kernel moments](kernel-moments.md).


Common matrix operations now delegate to the shared [general scalar operator](pairwise-operators.md)
engine. kernel sources have geometric methods;
`tp.lazy` accepts all 87 named comparisons across vector, point-cloud,
structured, image, audio, sequence and token families, preserving directed
comparison orientation under transpose. Both Python facades expose the same
[common consumers](pairwise-operators.md#operation-coverage) and logical transpose. The coordinate-specific
geometry methods are conveniences of the kernel specialization.
`tp.as_lazy(A)` exposes those consumers for caller-owned values; see
[matrix operators](matrix-operators.md).

## Comparing Gram Matrices

`K.frobenius_inner_product(L, center=True, normalize=True)` streams centered
kernel alignment between two representations on the same ordered observations.
`K.hsic(L, unbiased=False)` supplies biased HSIC directly, while
`unbiased=True` selects the diagonal-excluding estimator. All scalar operator
sources share these consumers; Frobenius products also accept rectangular
sources. See [formulas, gradients and memory](kernel-alignment.md).
