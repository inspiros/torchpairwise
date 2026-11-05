# Block Linear Operators

`tp.block([[A,B],[C,D]])` and native `s::block(...)` combine a
rectangular grid of maps without constructing the full matrix. Entries can be
lazy pairwise objects or dense tensors; Python adapts tensors automatically.
C++ entries are explicitly adapted with `s::aslinearoperator` where needed. Every entry must be present. A small explicit zero tensor
can represent a constraint corner; scalar and None placeholders are unsupported.

\[
\begin{bmatrix}A&B\\C&D\end{bmatrix}
\begin{bmatrix}u\\v\end{bmatrix}
=\begin{bmatrix}Au+Bv\\Cu+Dv\end{bmatrix}
\]

The constructor requires a nonempty rectangular list/tuple of rows. Each block
row shares one height and each block column one width. Sample dimensions match
exactly, including empty partitions; leading batch dimensions broadcast. All
entries use the same real float32/float64 dtype and device. Shape is
`[...,sum(row_heights),sum(column_widths)]`. Reconstruct the map after resizing
any bound tensors; ordinary value mutations remain visible.

Blocks are assembled from horizontal/vertical map concatenations. These are
also available as `torch.hstack((A, B))` and `torch.vstack((A, B))` in Python. Horizontal
concatenation splits the RHS sample axis, applies the two maps and sums results;
vertical concatenation applies both maps and concatenates their results. The
transpose reverses these roles. A vector RHS `[M]` or block `[...,M,R]` follows
the usual linear-map shape/batch convention; R must be positive.

## Symmetric Constraint System

For A `[N,N]`, constraints C `[P,N]` and optional small corner D `[P,P]`, use:

```python
import torch
import torchpairwise as tp

A = (lambda K: K + 0.3 * tp.eye_like(K))(tp.lazy(X, metric="rbf", gamma=0.3))
C = tp.as_lazy(constraints)  # [P,N]
D = X.new_zeros(constraints.shape[-2], constraints.shape[-2])
KKT = tp.block([[A, C.T], [C, D]])  # logical [N+P,N+P]
result = KKT.matmul(values)         # values:[N+P,R]
```

```cpp
#include <torchpairwise.h>
namespace s = torchpairwise::streaming;

s::PairwiseOptions options;
options.comparison.gamma = 0.3;
auto A = s::aslinearoperator(s::kernel(X, c10::nullopt, "rbf", options)).add_diagonal(0.3);
auto C = s::aslinearoperator(constraints);
auto D = s::aslinearoperator(at::zeros(
    {constraints.size(-2), constraints.size(-2)}, X.options()));
auto KKT = s::block({{A, C.transpose()}, {C, D}});
auto result = KKT.matmul(values);
```

A symmetric A and D make KKT symmetric; a zero corner generally makes it
indefinite. [MINRES](minres.md) can solve such maps under its contracts. No
automatic symmetry, rank or feasibility proof is provided. Dependent constraints
can give a singular system.

## Execution, Gradients and Storage

Public C++ actions, Python actions and native solver plans implement the same
block split/product/combine rules. The finite serialized graph adds two map
opcodes without changing existing dispatcher schema signatures. Built-in source
numerical execution and first backward remain native. Python defines the graph;
no Python callback is needed when using it through the native facade.

Block actions support first gradients, compile/export, transpose, composition
and the native solvers. Solvers keep their own assumptions and inference-only
contracts; differentiable block products do not make a solve differentiable.

Storage includes every explicit dense block, source data/preparation, RHS and
output blocks, and intermediate block products. No full generated matrix or
large zero padding is allocated. Result blocks may expand across broadcast
batches before concatenation; the concatenated output is stored. Shared graph
nodes share descriptions/preparation but are not a general matvec cache: a
repeated block may be evaluated more than once during an action. Autograd can
retain intermediate value blocks for backward. Fully dense input blocks retain
their dense storage; this factory does not compress them.
