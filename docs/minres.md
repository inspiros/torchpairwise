# Streamed Symmetric Indefinite Systems

`tp.linalg.minres` and
`torchpairwise::streaming::linalg::minres` solve real symmetric square systems
that may be indefinite. Symmetry is a caller assumption; it is not exhaustively
checked. Positive definiteness, required by CG, is not required by MINRES.

\[
AX=B,\qquad \|B_{:r}-AX_{:r}\|_2\le
\max(\mathrm{atol},\mathrm{rtol}\|B_{:r}\|_2)
\]

A nonsingular map gives a unique solution. For singular or inconsistent maps,
this API promises neither a minimum-norm solution nor a least-squares answer.
This is basic MINRES, without MINRES-QLP or a preconditioner. Use
[LSQR](least-squares.md) for the least-squares objective, or
[GMRES](gmres.md) for general nonsymmetric square systems.

## Shapes and Diagnostics

A is a lazy pairwise object or dense tensor `[...,N,N]`; B is `[N]` or
`[...,N,R]`. Use `[...,N,1]` for batched vectors. Batches broadcast and each RHS
stops independently. The solution is `[...,N]` or `[...,N,R]`. All floating
inputs share float32/float64 dtype and device; no implicit conversion is added.
Empty N or batch axes are supported, with positive R. Optional `initial`
broadcasts to the solution without adding batch axes, with matching sample and
channel dimensions. `max_iter=0` evaluates that guess, defaulting to zero.

Options are finite nonnegative `rtol` (default `1e-5`), `atol` (default `0.`),
and nonnegative integer `max_iter` (default `1000`). The budget counts total
Lanczos steps, including reliable restarts; no restart-width parameter exists.

`MINRESResult` returns `solution`, float64 `residual_norm`, int64 `iterations`
and `status`, and bool `converged`. Diagnostics have shape `[...]` for vector B
or `[...,R]` for blocks.

| Status | Python Enum | Meaning |
| --- | --- | --- |
| 0 | `MINRESStatus.CONVERGED` | Actual residual meets tolerance |
| 1 | `MINRESStatus.ITERATION_LIMIT` | Total step budget exhausted |
| 2 | `MINRESStatus.BREAKDOWN` | Lanczos/QR breakdown without verified convergence |
| 3 | `MINRESStatus.NONFINITE` | Nonfinite input/action/arithmetic |

Failures retain the current iterate. Invalid arguments raise; inspect diagnostics
before using a result. Happy breakdown is successful only when the actual
residual passes. A small residual does not bound solution error for a poorly
conditioned map or certify that the map was symmetric.

## Numerical Method and Deployment

The native C++/ATen implementation uses a short Lanczos recurrence and rolling
QR rotations. Internal vectors/scalars use float64; source products and the
returned solution retain the supplied precision. There is no stored Lanczos
basis or reorthogonalization, so finite precision may lose orthogonality and
require more iterations. Candidate convergence, breakdown and final termination
check the actual residual of the returned-precision point. A false recurrence
estimate can restart unfinished recurrences from their actual residual. A
Lanczos remainder tiny relative to the source action is treated as numerical
breakdown; actual convergence still decides success.

Solver workspace is O(NR) vector blocks per broadcast batch, plus source data,
preparation, tile/consumer workspace and outputs. Block operators add their
intermediate products. No generated matrix is retained; repeated map actions
still evaluate all pairs, so a small memory footprint is not a speed guarantee.

Typed dispatcher schemas, Python fake metadata and serialized MINRESResult
support compatible compile/export as one opaque native solve. Built-in numerical
execution has no Python callback dependency. Inference-only execution requires
explicit no-grad for gradient-bearing inputs. Autocast is excluded inside the
solver. Native host stopping synchronizes CUDA, so CUDA graph capture is
unsupported. First gradients through a block product are separate from solver
derivatives, which are not supported.

## Kernel Ridge with an Unpenalized Intercept

For a symmetric PSD kernel K and positive alpha, fit coefficients W `[N,R]`
and intercept b `[1,R]` using the symmetric indefinite KKT system:

\[
\begin{bmatrix}K+\alpha I&\mathbf1\\\mathbf1^\mathsf T&0\end{bmatrix}
\begin{bmatrix}W\\b\end{bmatrix}
=\begin{bmatrix}Y\\0\end{bmatrix},\qquad
\widehat Y_{\mathrm{test}}=K(X_{\mathrm{test}},X)W+\mathbf1 b
\]

The constraint \(\mathbf1^\mathsf T W=0\) permits an unpenalized intercept;
this differs from the earlier ridge example with no fitted intercept. The
[example](../examples/kernel_ridge_intercept.py) constructs the KKT map with
[block operators](block-operators.md); its zero corner is only `[1,1]`.
Held-out predictions use a rectangular streamed kernel product. Its small dense
reference checks direct KKT factorization, an independent Schur-complement
solution, coefficient constraint and held-out predictions.

```python
import torch
import torchpairwise as tp

# X:[N,D], targets:[N,R], N>0
with torch.no_grad():
    A = (lambda K: K + 0.3 * tp.eye_like(K))(tp.lazy(X, metric="rbf", gamma=0.3))
    C = tp.as_lazy(X.new_ones(1, X.shape[0]))
    KKT = tp.block([[A, C.T], [C, X.new_zeros(1, 1)]])
    rhs = torch.cat((targets, targets.new_zeros(1, targets.shape[-1])), -2)
    fit = tp.linalg.minres(KKT, rhs, rtol=1e-8, max_iter=300)
    if not fit.converged.all():
        raise RuntimeError(f"MINRES status: {fit.status.tolist()}")
    weights, intercept = fit.solution[:-1], fit.solution[-1:]
```

```cpp
#include <torchpairwise.h>
namespace s = torchpairwise::streaming;

at::NoGradGuard no_grad;
s::PairwiseOptions options;
options.comparison.gamma = 0.3;
auto A = s::aslinearoperator(s::kernel(X, c10::nullopt, "rbf", options)).add_diagonal(0.3);
auto C = s::aslinearoperator(at::ones({1, X.size(0)}, X.options()));
auto D = s::aslinearoperator(at::zeros({1, 1}, X.options()));
auto KKT = s::block({{A, C.transpose()}, {C, D}});
auto rhs = at::cat({targets, at::zeros({1, targets.size(-1)}, targets.options())}, -2);
s::linalg::MINRESOptions solve;
solve.rtol = 1e-8;
solve.max_iter = 300;
auto fit = s::linalg::minres(KKT, rhs, solve);
TORCH_CHECK(fit.converged.all().item<bool>(), "MINRES did not converge");
auto weights = fit.solution.slice(-2, 0, X.size(0));
auto intercept = fit.solution.slice(-2, X.size(0), X.size(0)+1);
```
