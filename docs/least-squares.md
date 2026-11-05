# Streamed Least Squares

`torchpairwise.linalg.lsqr` and
`torchpairwise::streaming::linalg::lsqr` solve real rectangular least squares:

\[
\min_X\ \|AX-B\|_F^2+\delta^2\|X\|_F^2
\]

`damp=delta` is a finite nonnegative scalar. The regularization coefficient is
`damp**2`, not `damp`. This is ordinary Euclidean coefficient regularization;
applying it to a kernel dictionary does not make it exact kernel ridge or
Nyström-whitened regression.

The common entry point `streaming.linalg.lstsq` / native `s::linalg::lstsq`
selects this same LSQR implementation and returns the same `LSQRResult`.
Its only method is `"lsqr"` (default). See [unified solves](linear-solves.md)
for the distinction between iterative minimum-norm targets and an SVD
pseudoinverse with a singular-value cutoff.

## Inputs and Shapes

A accepts a lazy pairwise object or a dense tensor with
shape `[...,N,M]`. B is `[N]` or `[...,N,R]`; use `[...,N,1]` for batched vectors.
Operator and RHS batches broadcast. The solution is `[...,M]` for a vector RHS
or `[...,M,R]` for a block. R must be positive; empty N, M and batch axes are
supported. All floating tensors share a real float32/float64 dtype and device.
No automatic device or dtype conversion is applied.

`initial` has the solution's sample/channel axes and can broadcast to its batch
axes without adding batch dimensions. It defaults to zero. Damping penalizes
the final X, not X-initial. `max_iter=0` evaluates the initial guess without an
iteration. Tolerances and damping must be finite/nonnegative; max_iter is a
nonnegative integer.

With no damping, a zero start targets the minimum-norm least-squares solution
in exact arithmetic. A nonzero initial can preserve null-space components.
Finite precision, conditioning and stopping tolerances limit coefficient
accuracy; this is not a numerical-rank estimator or a pseudoinverse API.

## Algorithm and Convergence

Independent Golub-Kahan bidiagonal recurrences use A and its transpose. For
positive damping, they operate on the virtual augmented system `[A; damp I]`
with RHS `[B; 0]`. The code does not construct A.T A or the augmented matrix.
Source preparation is shared across iterations. Unlike CG, A can be rectangular,
nonsymmetric or rank deficient; no symmetry/PSD assumption is required.

Define the data residual and normal residual:

\[
R=B-AX,\qquad G=A^\mathsf T R-\delta^2X,\qquad
\|G_{:r}\|_2\le\max(\mathrm{atol},\mathrm{rtol}\|(A^\mathsf T B)_{:r}\|_2)
\]

This is a stationarity test, not a bound on solution error. An inconsistent
least-squares problem normally has a nonzero data residual at its optimum.
Absolute tolerance is in normal-residual units. Scaling A/B changes these units.
A small normal residual may coexist with inaccurate coefficients for an
ill-conditioned problem. Regularization can improve conditioning while changing
the fitted objective.

Recurrence estimates nominate converged columns. Actual A/A.T actions verify
these candidates, breakdowns and final diagnostics. Each batch/RHS stops
independently. `iterations` counts bidiagonal updates. Extra verification
products and native stopping checks add work and CUDA synchronization.

`LSQRResult` contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `solution` | `[...,M]` or `[...,M,R]` | Final coefficient vector/block |
| `residual_norm` | `[...]` or `[...,R]` | True data residual norm, float64 |
| `normal_residual_norm` | `[...]` or `[...,R]` | True stationarity residual norm, float64 |
| `iterations` | `[...]` or `[...,R]` | int64 update counts |
| `converged` | `[...]` or `[...,R]` | Boolean success flags |
| `status` | `[...]` or `[...,R]` | int64 status codes |

| Status | Python Enum | Meaning |
| --- | --- | --- |
| 0 | `LSQRStatus.CONVERGED` | True normal residual meets tolerance |
| 1 | `LSQRStatus.ITERATION_LIMIT` | Update budget exhausted |
| 2 | `LSQRStatus.BREAKDOWN` | Bidiagonal recurrence terminated without verified convergence |
| 3 | `LSQRStatus.NONFINITE` | Nonfinite input/action/arithmetic |

The result retains the current iterate on failure or iteration exhaustion.
Inspect `converged`/`status` before using it. Invalid arguments raise. A finite
returned array alone does not imply success, especially for nonfinite failures.

## Python

```python
import torch
import torchpairwise as tp

# X:[N,D], landmarks:[L,D], targets:[N,R], X_test:[T,D]
with torch.no_grad():
    A = tp.lazy(X, landmarks, metric="rbf", gamma=0.3, chunk_size=64)
    fit = tp.linalg.lsqr(A, targets, damp=0.2, rtol=1e-6, max_iter=300)
    if not fit.converged.all():
        raise RuntimeError(f"LSQR status: {fit.status.tolist()}")
    prediction = tp.lazy(X_test, landmarks, metric="rbf", gamma=0.3).matmul(fit.solution)
    # coefficients:[L,R], prediction:[T,R]
```

## C++

```cpp
#include <torchpairwise.h>
namespace s = torchpairwise::streaming;

at::NoGradGuard no_grad;
s::PairwiseOptions source;
source.comparison.gamma = 0.3;
source.chunk_size = 64;
auto A = s::aslinearoperator(s::kernel(X, landmarks, "rbf", source));
s::linalg::LSQROptions solve;
solve.damp = 0.2;
solve.rtol = 1e-6;
solve.max_iter = 300;
auto fit = s::linalg::lsqr(A, targets, solve);
TORCH_CHECK(fit.converged.all().item<bool>(), "LSQR did not converge");
auto prediction = s::kernel(X_test, landmarks, "rbf", source).matmul(fit.solution);
```

## Deployment and Memory

Numerical execution is C++/ATen with a typed dispatcher schema. Python builds
an operator description and wraps its result, with no Python numerical loop.
Compile/export preserve an opaque native solve and serialized diagnostics.

The solver is inference-only. Gradient-bearing inputs, source options and initial
guesses require explicit no-grad; they are not silently detached. CUDA graph
capture is unsupported because native early stopping and candidate verification
synchronize the host. Autocast is excluded inside the solver; supplied precision
is preserved for source products and solution blocks, while scalar recurrence
calculations and diagnostics use float64.

Solver storage is several `[...,N,R]` and `[...,M,R]` blocks (with an additional
augmented vector block when damped), plus source inputs/preparation/tiles and
consumer workspace. It is not a tile-only memory bound. Each action still
computes all pairs and may be repeated across many iterations. LSQR does not
promise the speed of a dense factorization that fits in memory.

Run [landmark regression](../examples/landmark_regression.py) with
`--dense-reference` on small inputs to compare against an independent augmented
`torch.linalg.lstsq`, including objective, coefficients and held-out predictions.
