# Streamed Nonsymmetric Linear Systems

`torchpairwise.linalg.gmres` and
`torchpairwise::streaming::linalg::gmres` solve real square systems with restarted
GMRES. Unlike CG, they require neither symmetry nor positive definiteness. A
nonsingular operator gives a unique solution; convergence is not guaranteed for
all restart widths or badly conditioned/nonnormal operators.

\[
AX=B,\qquad \|B_{:r}-AX_{:r}\|_2\le
\max(\mathrm{atol},\mathrm{rtol}\|B_{:r}\|_2)
\]

Use [LSQR](least-squares.md) for rectangular or inconsistent least-squares
problems. GMRES does not promise a least-squares or minimum-norm answer for
singular systems.

## Shapes and Options

A accepts a lazy pairwise object or dense matrix `[...,N,N]`.
B has shape `[N]` or `[...,N,R]`; use `[...,N,1]` for batched vectors. Map/RHS
batches broadcast, with independent recurrences and stopping per batch/column.
The solution is `[...,N]` or `[...,N,R]`. R is positive; empty sample/batch axes
are supported. Map, RHS and initial must share real float32/float64 dtype/device.

`initial` defaults to zero and can broadcast to the solution without adding
batch axes; sample/channel dimensions must match. `max_iter=0` only evaluates
that guess.

| Option | Default | Meaning |
| --- | --- | --- |
| `rtol` | `1e-5` | Nonnegative finite relative true-residual tolerance |
| `atol` | `0.` | Nonnegative finite absolute true-residual tolerance |
| `restart` | `20` | Positive integer limiting Arnoldi steps per cycle, clamped to N |
| `max_iter` | `1000` | Nonnegative total Arnoldi-step budget per batch/RHS across all cycles |
| `initial` | `None` | Optional initial solution |

`max_iter` counts inner steps, not restart cycles. This differs from libraries
whose iteration budget counts cycles. Smaller restart widths save basis memory
but may converge more slowly or stagnate. No preconditioning is applied.

## Numerical Method and Diagnostics

Each cycle uses two passes of modified Gram-Schmidt and Givens rotations to
solve the small Arnoldi least-squares problem. Recurrence estimates nominate
convergence candidates. Actual A actions verify their residuals using the
returned-precision solution, at restart boundaries, numerical breakdowns and
final termination. A false estimate restarts from the actual residual.

Basis vectors, orthogonalization, small QR/back-substitution and norms use
float64. Source products and output coefficients retain the supplied dtype.
This still does not guarantee accurate coefficients from a small residual for
ill-conditioned systems. A tiny Arnoldi remainder relative to the source-action
norm is treated as a numerical breakdown; success still requires true residual
verification. Restart stagnation exhausts the total iteration budget.

`GMRESResult` contains `solution`, float64 `residual_norm`, int64 `iterations`
and `status`, and bool `converged`. Diagnostics have shape `[...]` for a vector
RHS or `[...,R]` for a block.

| Status | Python Enum | Meaning |
| --- | --- | --- |
| 0 | `GMRESStatus.CONVERGED` | Actual residual meets tolerance |
| 1 | `GMRESStatus.ITERATION_LIMIT` | Total Arnoldi-step budget exhausted |
| 2 | `GMRESStatus.BREAKDOWN` | Arnoldi/triangular breakdown without verified convergence |
| 3 | `GMRESStatus.NONFINITE` | Nonfinite input/action/arithmetic |

The current iterate is retained on failure or exhaustion. Invalid options raise;
inspect status before using any result. Happy breakdown is reported as success
only when its actual residual passes. Diagnostics do not certify nonsingularity.

## Python

```python
import torch
import torchpairwise as tp

# X, keys:[N,D]; separate views of the same N vertices
with torch.no_grad():
    W = tp.lazy(X, keys, metric="rbf", gamma=0.3) + X.new_tensor(0.01 / X.shape[0])
    P = W.weight_rows(W.sum(dim=-1).reciprocal())
    A = -0.85 * P.mT + tp.eye_like(P)
    rhs = X.new_full((X.shape[0], 1), 0.15 / X.shape[0])
    fit = tp.linalg.gmres(A, rhs, restart=12, rtol=1e-8, max_iter=200)
    if not fit.converged.all():
        raise RuntimeError(f"GMRES status: {fit.status.tolist()}")
    ranks = fit.solution  # [N,1]
```

## C++

```cpp
#include <torchpairwise.h>
namespace s = torchpairwise::streaming;

at::NoGradGuard no_grad;
s::PairwiseOptions source;
source.comparison.gamma = 0.3;
auto W = s::kernel(X, keys, "rbf", source) +
    s::ScalarExpression(at::scalar_tensor(0.01/X.size(0), X.options()));
auto P = W.weight_rows(W.reduce("sum", -1).reciprocal());
auto A = s::aslinearoperator(P).transpose().scale(-0.85).add_diagonal(1.);
auto rhs = at::full({X.size(0), 1}, 0.15/X.size(0), X.options());
s::linalg::GMRESOptions options;
options.restart = 12;
options.rtol = 1e-8;
options.max_iter = 200;
auto fit = s::linalg::gmres(A, rhs, options);
TORCH_CHECK(fit.converged.all().item<bool>(), "GMRES did not converge");
auto ranks = fit.solution;  // [N,1]
```

## Memory and Deployment

With effective width m=min(restart,N,max_iter), workspace includes
O(NRm+Rm²) double basis/QR blocks per broadcast batch, solution/RHS/residual
blocks, source preparation and tile/consumer workspace. It does not retain the
generated matrix, but full-width GMRES with m near N can approach quadratic
storage. Each source action still evaluates all pairs. There is no general
speed advantage over dense factorization when the matrix fits in memory.

Numerical iteration is C++/ATen with a typed dispatcher schema and thin Python
bindings. Compatible compile/export preserves an opaque native solve and result
fields. Arbitrary custom source callbacks retain their own deployment needs.
Execution is inference-only: gradient-bearing inputs require explicit no-grad.
Autocast is excluded inside the solver. Host-controlled native stopping and
candidate verification synchronize CUDA, so CUDA graph capture is unsupported.

## PageRank Example

[Personalized PageRank](../examples/pagerank.py) uses the directed affinity
W=K(query,key)+floor/N, row-normalizes it to P, and solves:

\[
(I-\beta P^\mathsf T)R=(1-\beta)V,\qquad 0\le\beta<1
\]

Each V column is a nonnegative personalization vector summing to one. The
positive outgoing floor avoids dangling rows. Distinct query/key features mean
W need not be symmetric. This is a generated dense affinity graph with self
edges, not a sparse web-graph implementation. Only row mass, solver blocks and
ranks are retained. The small dense-reference option checks an independent
`torch.linalg.solve`, stationarity, nonnegativity and rank mass. Returned ranks
are not clipped or renormalized to conceal a solver error.
