# Solves and Least Squares

Use `streaming.linalg.solve` to apply an inverse through a square-system solve,
or `streaming.linalg.lstsq` for rectangular/rank-deficient least squares.
These entry points select existing native algorithms, without constructing
an inverse, pseudoinverse or normal matrix. The named solver APIs remain
available for direct use.

## Explicit Method Selection

| Entry | Method | Assumption | Method-Specific Options |
| --- | --- | --- | --- |
| `solve` | `"cg"` | Real symmetric positive definite A | Optional fixed SPD inverse-action `preconditioner` |
| `solve` | `"minres"` | Real symmetric A, possibly indefinite | None |
| `solve` | `"gmres"` | General real square A | Optional `restart`, default 20 |
| `lstsq` | `"lsqr"` (default) | General real rectangular/square A | Optional `damp`, default 0 |

`solve` requires an explicit method; it performs no symmetry/PSD detection,
automatic solver selection or fallback. Unsupported options raise, including an
explicit `restart=20` for CG or `preconditioner` for MINRES/GMRES. `lstsq` only
supports LSQR. Both accept `rtol`, `atol`, `max_iter` and optional `initial`.
Python method strings are case-sensitive and static within a compiled/exported
graph. Changing the method may require recompilation.

Python accepts a dense Tensor or lazy pairwise object.
C++ accepts a LinearOperator; use `s::aslinearoperator` for source/dense adapters.
All operands use matching float32/64 dtype/device, with broadcast leading batches
and independent RHS convergence. A `[N]` RHS produces batched vectors; use
`[...,N,1]` for batched single-column inputs.

## Square Systems

For A `[...,N,N]` and B `[N]` or `[...,N,R]`, solve for X `[...,N]` or
`[...,N,R]`:

\[
AX=B,\qquad X=A^{-1}B
\]

The inverse notation assumes a nonsingular system. Iterative convergence verifies
the actual residual, independently for each RHS column:

\[
\|b-Ax\|_2\le\max(\mathrm{atol},\mathrm{rtol}\|b\|_2)
\]

```python
import torchpairwise as tp

# X:[N,D], targets:[N,R]
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=0.2)
    A = K + 0.1 * tp.eye_like(K)
    fit = tp.linalg.solve(A, targets, method="cg", rtol=1e-6)
    if not fit.converged.all():
        raise RuntimeError(f"CG status: {fit.status.tolist()}")
    coefficients = fit.solution  # [N,R]
```

```cpp
#include <torchpairwise.h>
namespace s = torchpairwise::streaming;
at::NoGradGuard no_grad;
s::PairwiseOptions source;
source.comparison.gamma = 0.2;
auto K = s::kernel(X, c10::nullopt, "rbf", source);
auto A = s::aslinearoperator(K).add_diagonal(0.1);
s::linalg::SolveOptions options;
options.method = "cg";  // Required; an empty/unknown method raises.
options.rtol = 1e-6;
auto fit = s::linalg::solve(A, targets, options);
TORCH_CHECK(fit.converged.all().item<bool>(), "CG did not converge");
auto coefficients = fit.solution;  // [N,R]
```

`SolveResult` has `solution`, float64 `residual_norm`, int64 `iterations` and
`status`, and bool `converged`. Diagnostics are `[...]` for vector RHS or
`[...,R]` for block RHS. Numerical failure returns the current iterate and
diagnostics; no exception or retry is automatically triggered. Malformed
arguments raise. Status values preserve the selected solver's meanings:

| Code | CG | MINRES | GMRES |
| --- | --- | --- | --- |
| 0 | Converged | Converged | Converged |
| 1 | Iteration limit | Iteration limit | Iteration limit |
| 2 | Nonpositive curvature | Lanczos/QR breakdown | Arnoldi/triangular breakdown |
| 3 | Nonfinite arithmetic | Nonfinite arithmetic | Nonfinite arithmetic |
| 4 | Invalid preconditioner | Unused | Unused |

Use `CGStatus`, `MINRESStatus` or `GMRESStatus` for the selected method. There
is no separate status enum that erases these distinctions. Symmetry/definiteness
and a suitable method remain caller responsibilities. Small residuals do not
guarantee accurate coefficients on ill-conditioned systems. Singular inconsistent
systems belong to least squares; `solve` does not promise minimum-norm results.

## Implicit Solution Gradients

Set `differentiable=True` on `solve` to differentiate a converged nonsingular
square system. The same `SolveOptions.differentiable` field is available in C++.
For incoming solution cotangent G, backward solves an adjoint system:

\[
A^\mathsf T\Lambda=G,\qquad
\overline B=\Lambda,\qquad
\overline A=-\Lambda X^\mathsf T
\]

The last expression is contracted through source inputs and map factors using
streamed product replay; no dense matrix cotangent is formed for a streamed
source. Dense matrix inputs receive a matrix-shaped gradient. Iteration history
is not saved, but fitted solution blocks, original operands and replay workspace
remain allocated. Broadcast operand gradients sum to their original shapes.

```python
# features:[N,D], targets:[N,R]; either may require gradients.
A = (lambda K: K + 0.1 * tp.eye_like(K))(tp.lazy(features, metric="rbf", gamma=0.2))
fit = tp.linalg.solve(A, targets, method="cg", differentiable=True, rtol=1e-7)
loss = fit.solution.square().mean()
loss.backward()
```

```cpp
// Use an ordinary grad-enabled scope, unlike the inference example above.
s::linalg::SolveOptions differentiable;
differentiable.method = "cg";
differentiable.differentiable = true;
differentiable.rtol = 1e-7;
auto fitted = s::linalg::solve(A, targets, differentiable);
fitted.solution.square().mean().backward();
```

CG/MINRES retain their symmetry assumptions; GMRES uses the actual transpose.
Both solves use the selected method, tolerance, iteration budget and GMRES
restart. Failure to converge raises, including during backward and in no-grad
execution of this opt-in mode. Only the solution is differentiable, with first
derivatives only. This differentiates the mathematical solution under a local
nonsingularity assumption, not finite iteration decisions. Loose tolerances or
poor conditioning can still make gradients inaccurate. Rank-changing/singular
systems are outside this contract.

Initial guesses are algorithmic aids and must be detached. CG supports a fixed
SPD inverse-action preconditioner in both the forward and adjoint solve. Its
algorithmic-only operands are detached, so setup has no backward edge; operands shared with A or B
still receive the mathematical system gradients. The preconditioner must remain
fixed during each solve, with the existing dtype/batch/SPD assumptions. Existing source-specific gradient
and deployment restrictions still apply; Python callbacks require Python.
Named solvers remain inference-only. Unified `lstsq` supports the damped
implicit mode described below.

## Least Squares and Pseudoinverse Actions

For A `[...,N,M]` and B `[N]` or `[...,N,R]`, `lstsq` returns solution
`[...,M]` or `[...,M,R]` for:

\[
\min_X\ \|AX-B\|_F^2+\lambda^2\|X\|_F^2
\]

Here lambda is `damp`, not `damp` squared. With `damp=0` and the default zero
start, LSQR targets the minimum-norm least-squares solution, approximating the
pseudoinverse action A-dagger B:

\[
X=A^\dagger B,\qquad
X\in\operatorname*{argmin}_Z\|AZ-B\|_F,\quad
\|X\|_F\ \text{minimal}
\]

This is not an SVD pseudoinverse with an `rcond`/`rtol` rank cutoff. No rank or
singular values are estimated. A nonzero initial guess can retain components
in A's null space. Positive damping gives a regularized solution, not the
undamped pseudoinverse action, and penalizes the final solution itself.
Normal-residual stopping and poor conditioning can leave coefficient error;
inspect diagnostics and select tolerances for the task.

```python
# X:[N,D], landmarks:[L,D], targets:[N,R]
with torch.no_grad():
    A = tp.lazy(X, landmarks, metric="rbf", gamma=0.3)
    fit = tp.linalg.lstsq(A, targets, damp=0.2, rtol=1e-6)
    if not fit.converged.all():
        raise RuntimeError(f"LSQR status: {fit.status.tolist()}")
    coefficients = fit.solution  # [L,R]
```

```cpp
at::NoGradGuard no_grad;
s::PairwiseOptions source;
source.comparison.gamma = 0.3;
auto A = s::aslinearoperator(s::kernel(X, landmarks, "rbf", source));
s::linalg::LstsqOptions options;  // method defaults to "lsqr".
options.damp = 0.2;
options.rtol = 1e-6;
auto fit = s::linalg::lstsq(A, targets, options);
TORCH_CHECK(fit.converged.all().item<bool>(), "LSQR did not converge");
auto coefficients = fit.solution;  // [L,R]
```

The returned `LSQRResult` retains both `residual_norm` (data residual) and
`normal_residual_norm` (stationarity), plus `solution`, `iterations`, `converged`
and `status`. A nonzero data residual may be optimal. Convergence verifies:

\[
\|A^T(b-Ax)-\lambda^2x\|_2
\le\max(\mathrm{atol},\mathrm{rtol}\|A^Tb\|_2)
\]

It does not return the `rank`, `singular_values` or squared `residuals` fields
of `torch.linalg.lstsq`. See [LSQR contracts](least-squares.md) for diagnostics,
warm starts and rank-deficient behavior. For an exact materialized dense
inverse or SVD pseudoinverse, use `torch.linalg.inv`/`pinv` or their ATen APIs;
the output storage remains quadratic for square systems.

## Differentiable Damped Least Squares

`lstsq(..., damp=alpha, differentiable=True)` differentiates the unique minimizer
for finite positive alpha, including tall, wide and rank-deficient A. Alpha is
a fixed scalar; its square must be finite and strictly positive in the source
dtype. Undamped differentiation is not supported by this entry point.

For solution X, residual R and incoming cotangent G, backward applies:

\[
(A^\mathsf T A+\alpha^2I)\Lambda=G,\qquad
\overline B=A\Lambda,\qquad
\overline A=R\Lambda^\mathsf T-(A\Lambda)X^\mathsf T,\qquad R=B-AX
\]

Forward uses native LSQR. Backward uses CG on the damped normal action, then
two streamed product VJPs; neither a normal matrix nor the matrix cotangent is
stored for a streamed source. Source/factor/RHS first gradients are supported.
Both solves must converge or raise; diagnostics are detached and initial guesses
must be detached. Higher derivatives are unsupported. Positive damping gives
uniqueness, but small damping and ill conditioning can still amplify solution or
gradient error. The backward normal action squares singular values; choose
regularization and stopping tolerances accordingly.

```python
# X:[N,D], landmarks:[L,D], targets:[N,R], with grad-enabled operands
A = tp.lazy(X, landmarks, metric="rbf", gamma=0.3)
fit = tp.linalg.lstsq(A, targets, damp=0.2, differentiable=True, rtol=1e-7)
fit.solution.square().mean().backward()
```

```cpp
// Use a grad-enabled scope.
s::PairwiseOptions source;
source.comparison.gamma = 0.3;
auto A = s::aslinearoperator(s::kernel(X, landmarks, "rbf", source));
s::linalg::LstsqOptions options;
options.damp = 0.2;
options.differentiable = true;
options.rtol = 1e-7;
auto fit = s::linalg::lstsq(A, targets, options);
fit.solution.square().mean().backward();
```

## Deployment and Storage

Python selectors call the same canonical native solver operators as the named
APIs. Native free functions select those same implementations. Inference calls retain their existing schemas. Implicit differentiation uses
registered native forward/adjoint-VJP operators; no Python numerical loop is
required. Compatible compile/export retains opaque native solves and result metadata.

The default square and least-squares modes are inference-only. Their
gradient-bearing operands require explicit
`torch.no_grad()` / `at::NoGradGuard`. CUDA graph capture remains unsupported
because native stopping synchronizes the host. Workspace and iteration meanings
are those of [CG](linear-operators.md#conjugate-gradient),
[MINRES](minres.md), [GMRES](gmres.md) or [LSQR](least-squares.md).

No lazy generic inverse object is returned. RHS-dependent iterative stopping
does not automatically provide the fixed SPD inverse action required by CG
preconditioning. Use explicit [prepared inverse actions](preconditioning.md)
for that contract. Named solver functions remain useful for method-specific
work; using the unified entry points does not alter their numerical behavior.

For trace estimates, SPD log-determinants, graph diffusion and inverse-square-root
actions, see [spectral consumers](spectral-consumers.md).
