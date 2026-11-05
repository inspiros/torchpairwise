# Constrained Quadratic Optimization

`tp.optimize.box_qp` and `torchpairwise::streaming::optimize::box_qp`
solve a real symmetric convex quadratic with finite elementwise bounds and at
most one affine equality per objective:

\[
\min_x\ \tfrac12x^\mathsf TQx+q^\mathsf Tx,\qquad
\ell\le x\le u,\qquad a^\mathsf Tx=b\quad\text{(optional)}
\]

Q is a lazy pairwise object or a caller-owned dense Tensor.
It must be symmetric PSD; the solver does not prove those assumptions. It uses
native C++/ATen iteration without materializing a generated comparison matrix.
This is a general projected method, not a dedicated SVM/SMO implementation.

## Python and C++

```python
import torchpairwise as tp

# X:[N,D], labels:[N] float labels {-1,+1}; both classes must be present
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=0.5)
    Q = K.weight_rows(labels).weight_columns(labels)
    fit = tp.optimize.box_qp(Q, -torch.ones_like(labels), lower=0., upper=1.,
                            equality=(labels, 0.), rtol=1e-6, max_iter=1500)
    if not fit.converged:
        raise RuntimeError(f"QP status: {fit.status.item()}")
    alpha = fit.solution                     # [N]
    raw_scores = K.matmul(alpha * labels)     # [N], before intercept
```

```cpp
#include <torchpairwise.h>
namespace s = torchpairwise::streaming;

at::NoGradGuard no_grad;
s::PairwiseOptions kernel_options;
kernel_options.comparison.gamma = 0.5;
kernel_options.chunk_size = 128;
auto K = s::kernel(X, c10::nullopt, "rbf", kernel_options);
auto Q = s::aslinearoperator(K.weight_rows(labels).weight_columns(labels));
s::optimize::BoxQPOptions options;
options.lower = at::scalar_tensor(0., X.options());
options.upper = at::scalar_tensor(1., X.options());
options.equality = s::optimize::QPEquality{
    labels, at::scalar_tensor(0., X.options())};
options.rtol = 1e-6;
options.max_iter = 1500;
auto fit = s::optimize::box_qp(Q, -at::ones_like(labels), options);
// Inspect fit.converged / fit.status before using approximate coefficients.
auto raw_scores = K.matmul(fit.solution * labels);
```

This is the C-SVM dual: `Q=diag(labels) K diag(labels)`, `q=-ones`, bounds `[0,C]`,
and `labels.T @ alpha=0`. A free support vector gives the intercept
`b=y_i-raw_score_i`. If no support vector is strictly inside the box, use the
KKT interval for b instead. The [binary SVM example](../examples/kernel_svm.py)
implements both cases, reports a primal-dual gap and optionally checks a small
dense version. A box-only solve does not represent an SVM with a free intercept.

## Shapes and Options

q `[N]` produces solution `[...,N]` and scalar-per-batch diagnostics `[...]`.
q `[...,N,R]` defines R independent QPs and produces `[...,N,R]`, with diagnostics
`[...,R]`. Q and q batches broadcast. Use `[...,N,1]` for batched vectors.

`lower`/`upper` are required and finite. Python numbers become scalar tensors
matching q; tensors must already share its float32/float64 dtype and device.
Bounds broadcast to the solution block, and `[N]` bounds mean per-sample bounds.
Use `[...,N,1]` for batch-specific sample bounds. Bounds cannot add batch axes;
use `[1,R]` to select RHS columns. `initial` follows the same broadcast rules and
is projected onto the feasible set. Without it, the projected zero point is used.
`max_iter=0` projects/evaluates the initial guess without optimizer updates.
Empty N/batch axes are supported; R must be positive.

`equality=(a,b)` takes a vector broadcasting like bounds and a target broadcasting
to `[...,R]`. They share q's dtype/device. A `[N]` vector is shared across RHS
columns. The zero vector is valid only for a zero target. Internally a and b
are divided by `||a||_inf` when nonzero, making projection and feasibility
scale-invariant. There are no additional batch axes from constraints or initial.

## Method and Diagnostics

Projection onto the box plus equality has the form:

\[
\Pi_{\mathcal C}(z)=\operatorname{clip}(z-\tau a,\ell,u),\qquad
 a^\mathsf T\operatorname{clip}(z-\tau a,\ell,u)=b
\]

The scalar root uses safeguarded Newton with bisection, capped at 80 inner steps.
Endpoint constraints and zero coefficients have explicit handling. Spectral
projected-gradient steps use a quadratic minimization on a feasible segment:

\[
g=Qx+q,\qquad d=\Pi_{\mathcal C}(x-\alpha g)-x,\qquad
\eta=\operatorname{clip}\!\left(-\frac{g^\mathsf Td}{d^\mathsf TQd},0,1\right),\qquad
x_{+}=x+\eta d
\]

A zero-curvature descent direction takes the full feasible step. The next
spectral step is `clip((d.T d)/(d.T Qd),1e-12,1e12)` when curvature is positive.
Equality-normal gradient components are removed in float64 before projection
and line calculations; this is projection-invariant in exact arithmetic and
avoids a float32 feasibility rounding error reversing a near-stationary step.
Projection, internal feasible iterates, inner products and diagnostics use
float64. Source products and returned solutions retain the requested
float32/float64 dtype. Each update's
objective and stationarity use a fresh true Qx product at the returned-precision
point, and feasibility is checked on that same point.

`QPResult` exposes `solution`, float64 `objective`, int64 `iterations`, bool
`converged`, int64 `status`, float64 `optimality` and `feasibility`.
For the normalized equality `(a_hat,b_hat)`, diagnostics are:

\[
\begin{aligned}
\mathrm{optimality}&=\|x-\Pi_{\mathcal C}(x-(Qx+q))\|_\infty\\
\mathrm{feasibility}&=\max\left(\|(\ell-x)_+\|_\infty,
\|(x-u)_+\|_\infty,|\hat a^\mathsf Tx-\hat b|\right)
\end{aligned}
\]

The equality term is omitted for box-only QPs. Optimality must be at most
`max(atol,rtol*max(1,||q||_inf))`; feasibility must be at most
`max(atol,rtol*max(1,abs(b_hat),sum(abs(a_hat*x))))`. Projection solves its scalar
root to a float64 roundoff bound before casting to solution dtype. Very tight
float32 requests may stall rather than converge.

| Status | Python Name | Meaning |
| --- | --- | --- |
| 0 | `QPStatus.CONVERGED` | Optimality and feasibility checks pass |
| 1 | `QPStatus.ITERATION_LIMIT` | Optimizer iteration budget exhausted |
| 2 | `QPStatus.INFEASIBLE` | Inconsistent box or unreachable equality target |
| 3 | `QPStatus.NONFINITE` | Nonfinite q, initial, equality, map action or arithmetic |
| 4 | `QPStatus.NONCONVEX` | A significantly negative search curvature was found |
| 5 | `QPStatus.PROJECTION_FAILED` | Projection could not meet its numerical criterion |
| 6 | `QPStatus.STALLED` | No representable step before convergence |

C++ enum members are `Converged`, `IterationLimit`, `Infeasible`, `Nonfinite`,
`Nonconvex`, `ProjectionFailed`, `Stalled`. Each batch/RHS stops independently.
Iterations count attempted optimizer updates, excluding inner projection steps.
Iteration exhaustion retains the approximate solution and diagnostics. Status
2-6 returns NaN solution/objective/optimality/feasibility; inspect the status.
Malformed options/shapes and nonfinite bounds raise. Negative-curvature detection
is not an exhaustive convexity test; a stationary point of a nonconvex map can
still pass the local residual check, so convexity remains a caller assumption.

## Memory, Deployment and Gradients

Retained workspace consists of several solution-sized blocks, normalized
constraints, source preparation, outputs and source tile workspace. There is no
implicit dense Q or inverse. Matvec work, source pair evaluation, projection
iterations and CUDA host synchronization can outweigh dense methods on small
problems. A caller-owned dense Q keeps its allocation. This method favors memory
savings; it makes no performance guarantee against specialized QP/SVM solvers.

The solver is inference-only: gradient-bearing inputs need explicit
`torch.no_grad()` / `at::NoGradGuard`. This includes bounds, equality, initial,
source tensors and tensor options. Compatible compile/export graphs retain one
native solve plus Tensor-valued diagnostics. Native stopping/root iterations
synchronize CUDA, so CUDA graph capture is unsupported. User callbacks retain
their existing deployment requirements; adaptation does not make Python
callbacks independently deployable.
