<div class="eyebrow">Streamed operators</div>

# Algebra and Kernel Methods

The [example gallery](../examples/index.md) connects these operators to kernel
ridge regression, PCA, LDA, SVM, landmark regression and PageRank, with shapes and runnable
dense checks.

Scalar source algebra transforms entries before a consumer. It supports
arithmetic with sources, tensors and numbers, powers, square, exp, log, sqrt,
trigonometric and sigmoid maps, minimum, maximum and clamping, comparisons,
`where`, masking and row/column weighting. Shapes must obey the operator
contracts; this is not lazy matrix composition. See the
[operator algebra notes](https://github.com/inspiros/torchpairwise/blob/master/docs/operator-algebra.md)
for the complete list, and [pair masks](statistics.md#pair-masks) for
eligibility-based consumers.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], Y:[M,D], V:[M,C]
K1 = tp.lazy(X, Y, metric="rbf", gamma=0.2)
K2 = tp.lazy(X, Y, metric="laplacian", gamma=0.3)
K = 0.7 * K1 + 0.3 * K2
features = K.matmul(V)               # [N,C]
energy = K.bilinear(left, right)     # scalar for left[N], right[M]
soft_features = K.softmax_matmul(V)  # [N,C]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X:[N,D], Y:[M,D], V:[M,C]
tp::PairwiseOperationOptions first, second;
first.comparison.gamma = 0.2;
second.comparison.gamma = 0.3;
auto K1 = tp::pairwise_expression(X, Y, "rbf", true, first);
auto K2 = tp::pairwise_expression(X, Y, "laplacian", true, second);
auto K = K1.scale(0.7) + K2.scale(0.3);
auto features = K.matmul(V);               // [N,C]
auto energy = K.bilinear(left, right);     // left[N], right[M] -> scalar
auto soft_features = K.softmax_matmul(V);  // [N,C]
```

</div>
</div>

## Products and Scalar Forms

\[
(AV)_{ic}=\sum_jA_{ij}V_{jc},\qquad
B=L^\mathsf{T}AR,\qquad Q=V^\mathsf{T}AV\]

`matmul` gives the product; `bilinear` computes B without retaining A or AR;
`quadratic` gives Q for a square source. Symmetry and PSD are not inferred.
`softmax_matmul` normalizes each source row over references before multiplying.

## Linear Maps

Lazy objects support matrix composition alongside entrywise source algebra. Both sample axes can be
centered, and square maps can receive a diagonal shift.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
import torchpairwise as tp
# X:[N,D], Y:[M,D], V:[N,R]
K = tp.lazy(X, Y, metric="rbf", gamma=0.2)
A = K.centered()  # H_N K H_M, [N,M]
B = A @ A.mT
B = B + 0.01 * tp.eye_like(B)  # [N,N]
result = B.matmul(V)           # [N,R]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
namespace s = torchpairwise::streaming;
s::PairwiseOptions options;
options.comparison.gamma = 0.2;
auto K = s::kernel(X, Y, "rbf", options);
auto A = s::aslinearoperator(K).centered();
auto B = A.compose(A.transpose()).add_diagonal(0.01);
auto result = B.matmul(V);  // [N,R]
```

</div>
</div>

`A @ B` composes two lazy objects; `A @ V` applies one to a Tensor. Centering
uses input/output sample means, while a diagonal shift adds a scaled RHS:

\[
H_L=I-\mathbf1\mathbf1^\mathsf{T}/L,\qquad
A_cV=H_N\bigl(A(H_MV)\bigr),\qquad
(A+\alpha I)V=AV+\alpha V
\]

Keep the entrywise source for entrywise square, products, statistics and selection.
Linear maps support composition, transpose, addition/subtraction, scalar scaling,
centering (including fitted `A.centered(reference_mean)` for held-out kernels)
and diagonal shifts, with matching float32/float64 inputs and
broadcast batches. Factor tensors may receive gradients. The map does not form
its matrix product, but intermediate vector blocks still use storage and may be
retained for backward. Reconstruct maps after resizing bound tensors.
See [full linear-map contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/linear-operators.md)
and the [dense-checked example](https://github.com/inspiros/torchpairwise/blob/master/examples/linear_operators.py).

## Block Maps

`tp.block([[A,B],[C,D]])` combines an explicit rectangular grid of maps.
Each block row shares a height and each block column a width; batches broadcast,
with matching float32/float64 dtype/device. Entries can be maps, scalar sources
or dense tensors in Python. In C++, adapt entries with `s::aslinearoperator` and
use `s::block({{A,B},{C,D}})`. `torch.hstack((A, B))` and `torch.vstack((A, B))` expose the two
building operations. Empty sample partitions are allowed; the grid itself must
be nonempty. Scalar/None placeholders are unsupported.

\[
\begin{bmatrix}A&B\\C&D\end{bmatrix}
\begin{bmatrix}u\\v\end{bmatrix}
=\begin{bmatrix}Au+Bv\\Cu+Dv\end{bmatrix}
\]

Actions split RHS blocks and combine native products without constructing a
full block matrix or large zero padding. Transpose, first gradients,
compile/export and native solvers use the finite block graph. Explicit dense
blocks and intermediate result blocks still use memory; shared descriptions do
not imply cached products. See [block contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/block-operators.md)
and [kernel fitting with an intercept](../examples/index.md#kernel-fitting-with-an-intercept).

## Choosing a Solver

Use `tp.linalg.solve` for a square system and `tp.linalg.lstsq`
for general least squares. Both select existing native algorithms and retain
convergence diagnostics. They do not construct an inverse or pseudoinverse.

| Entry | Method | Assumption | Extra Options |
| --- | --- | --- | --- |
| `solve` | `"cg"` | Real symmetric positive definite | `preconditioner` |
| `solve` | `"minres"` | Real symmetric, possibly indefinite | None |
| `solve` | `"gmres"` | General real square | `restart` (default 20) |
| `lstsq` | `"lsqr"` (default) | General real rectangular/square | `damp` (default 0) |

`solve` requires an explicit method. There is no automatic method detection or
fallback. Unsupported options raise; preconditioning belongs to CG and restart
to GMRES. Common controls are `rtol`, `atol`, `max_iter` and `initial`. Inputs
use matching float32/64 dtype/device with broadcast batches. Python accepts a
map, scalar streaming source or dense Tensor; C++ adapts sources with
`aslinearoperator`. For compile/export, the method string is static.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], targets:[N,R], landmarks:[L,D]
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=0.3)
    A = K + 0.2 * tp.eye_like(K)
    fit = tp.linalg.solve(A, targets, method="cg", rtol=1e-6)
    if not fit.converged.all():
        raise RuntimeError(f"CG status: {fit.status.tolist()}")
    coefficients = fit.solution  # [N,R]
    rectangular = tp.lazy(X, landmarks, metric="rbf", gamma=0.3)
    least = tp.linalg.lstsq(rectangular, targets, damp=0.2, rtol=1e-6)
    if not least.converged.all():
        raise RuntimeError(f"LSQR status: {least.status.tolist()}")
    landmark_coefficients = least.solution  # [L,R]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X:[N,D], targets:[N,R], landmarks:[L,D]
at::NoGradGuard no_grad;
s::PairwiseOptions source;
source.comparison.gamma = 0.3;
auto K = s::kernel(X, c10::nullopt, "rbf", source);
auto A = s::aslinearoperator(K).add_diagonal(0.2);
s::linalg::SolveOptions options;
options.method = "cg";
options.rtol = 1e-6;
auto fit = s::linalg::solve(A, targets, options);
TORCH_CHECK(fit.converged.all().item<bool>(), "CG did not converge");
auto coefficients = fit.solution;  // [N,R]
auto rectangular = s::aslinearoperator(s::kernel(X, landmarks, "rbf", source));
s::linalg::LstsqOptions least_options;  // method defaults to "lsqr".
least_options.damp = 0.2;
least_options.rtol = 1e-6;
auto least = s::linalg::lstsq(rectangular, targets, least_options);
TORCH_CHECK(least.converged.all().item<bool>(), "LSQR did not converge");
auto landmark_coefficients = least.solution;  // [L,R]
```

</div>
</div>

Square A `[...,N,N]` with b `[N]` or `[...,N,R]` returns `[...,N]` or
`[...,N,R]`. `SolveResult` contains solution, data residual norm, iterations,
converged and status. Status meanings remain those of the selected method:
code 2 is nonpositive curvature for CG, Lanczos breakdown for MINRES or Arnoldi
breakdown for GMRES. Code 0 is convergence, 1 iteration limit, 3 nonfinite
arithmetic; CG also uses 4 for an invalid preconditioner. Numerical failure
returns diagnostics and the current iterate; inspect them before using it.

Least squares A `[...,N,M]` returns `[...,M]` or `[...,M,R]`. Its `LSQRResult`
also reports `normal_residual_norm`: a nonzero data residual may be optimal.
With `damp=0` and a zero start, LSQR targets the minimum-norm solution:

\[
X\approx A^\dagger B,\qquad
X\in\operatorname*{argmin}_Z\|AZ-B\|_F,\quad\|X\|_F\ \text{minimal}
\]

This is an iterative target, not an SVD rank cutoff. A nonzero initial may
retain null-space components. Damping instead minimizes
\(\|AX-B\|_F^2+\mathrm{damp}^2\|X\|_F^2\), penalizing the final X.
No rank or singular-value estimates are returned. Stopping tolerance and poor
conditioning can leave coefficient error. For dense inverse/pseudoinverse
matrices use `torch.linalg.inv`/`pinv` or ATen; their output still needs storage.

The default square and least-squares modes are inference-only, support compatible compile/export, and
exclude CUDA graph capture because native stopping synchronizes the host.
The named methods below expose the same numerical implementations. See
[full unified contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/linear-solves.md).

## Conjugate Gradient

For real SPD maps, `tp.linalg.cg` solves kernel ridge systems without
storing the generated Gram matrix. Each RHS column has its own recurrence and
convergence diagnostics. Symmetry and positive definiteness are caller assumptions.

\[
(K+\alpha I)C=Y,\qquad \widehat Y=KC,\qquad
\|Y_{:r}-AC_{:r}\|_2\le\max(\mathrm{atol},\mathrm{rtol}\|Y_{:r}\|_2)
\]

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], targets:[N,R]
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=0.2)
    A = K + 0.1 * tp.eye_like(K)
    fit = tp.linalg.cg(A, targets, rtol=1e-6, max_iter=300)
    if not fit.converged.all():
        raise RuntimeError(f"CG status: {fit.status.tolist()}")
    predictions = K.matmul(fit.solution)  # [N,R]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X:[N,D], targets:[N,R]
at::NoGradGuard no_grad;
s::PairwiseOptions options;
options.comparison.gamma = 0.2;
options.chunk_size = 128;
auto K = s::kernel(X, c10::nullopt, "rbf", options);
auto A = s::aslinearoperator(K).add_diagonal(0.1);
s::linalg::CGOptions solve_options;
solve_options.rtol = 1e-6;
solve_options.max_iter = 300;
auto fit = s::linalg::cg(A, targets, solve_options);
// Inspect fit.converged / fit.status before using the result.
auto predictions = K.matmul(fit.solution);  // [N,R]
```

</div>
</div>

RHS `[N]` returns `[...,N]`; RHS `[...,N,R]` returns `[...,N,R]`, with broadcast
batches and positive R. Diagnostics have shape `[...]` or `[...,R]` respectively.
`CGResult` contains `solution`, float64 `residual_norm`, int64 `iterations` and
`status`, and bool `converged`. Status codes are 0 success, 1 iteration limit,
2 nonpositive curvature, 3 nonfinite arithmetic, 4 nonpositive preconditioned
residual energy. The actual residual verifies
convergence; numerical failure returns diagnostics. `initial` broadcasts to
the solution, and `max_iter=0` only evaluates that guess.

The solver is inference-only; gradient-bearing inputs require explicit no-grad.
Native early stopping synchronizes CUDA, so CUDA graph capture is unsupported.
Compile/export retain a native solve rather than unrolling iterations. Each
matvec still evaluates all pairs; storage includes source tiles, preparation and
solution-sized blocks. The [kernel ridge example](https://github.com/inspiros/torchpairwise/blob/master/examples/kernel_ridge.py)
compares with a small dense solve. See [full solver contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/linear-operators.md#conjugate-gradient).

### Diagonal Preconditioning

`tp.diag_embed(d)` represents a diagonal map from `d:[...,N]` without
allocating `[N,N]`. `tp.linalg.jacobi(d)` checks positive finite entries
and finite reciprocals, then stores their reciprocals once. Rebuild it if the
system changes. General diagonal maps retain the entries and support gradients.

CG accepts `preconditioner=M`, a fixed SPD inverse action approximating A inverse:

\[
z_k=M r_k,\qquad \rho_k=r_k^Tz_k
\]

The original residual still determines convergence. M uses matching `[N,N]`
dimensions/dtype/device and may broadcast to the solution without adding batches.
Symmetry and positive definiteness remain caller assumptions; a nonpositive
residual energy produces status 4. Gradient-bearing M operands require no-grad.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], targets:[N,R], d:[N] strictly positive
with torch.no_grad():
    D = tp.diag_embed(d.sqrt())
    K = tp.lazy(X, metric="rbf", gamma=0.2)
    K = K + 2. * tp.eye_like(K)
    A = D @ K @ D  # D(K+2I)D, diagonal = 3*d
    fit = tp.linalg.cg(A, targets, preconditioner=tp.linalg.jacobi(3. * d), rtol=1e-8)
    if not fit.converged.all():
        raise RuntimeError(f"PCG status: {fit.status.tolist()}")
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X:[N,D], targets:[N,R], d:[N] strictly positive
at::NoGradGuard no_grad;
auto D = s::diagonal(d.sqrt());
s::PairwiseOptions source;
source.comparison.gamma = 0.2;
auto K = s::aslinearoperator(s::kernel(X, c10::nullopt, "rbf", source)).add_diagonal(2.);
auto A = D.compose(K).compose(D);
s::linalg::CGOptions solve;
solve.preconditioner = s::linalg::jacobi(3. * d);
solve.rtol = 1e-8;
auto fit = s::linalg::cg(A, targets, solve);
TORCH_CHECK(fit.converged.all().item<bool>(), "PCG did not converge");
```

</div>
</div>

PCG adds one M action and keeps O(NR) vector blocks plus preconditioner state.
Fewer steps need not mean lower wall time. A constant-diagonal RBF Gram matrix
has no Jacobi iteration benefit; uneven diagonal scaling can benefit. The
[comparison example](https://github.com/inspiros/torchpairwise/blob/master/examples/preconditioned_cg.py)
reports both paths on one system, with an optional small dense oracle. See
[full contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/preconditioning.md).
This option belongs to CG; the other solvers retain their existing interfaces.

### Low-Rank and Nyström Preconditioning

`tp.linalg.lowrank_inverse(F, shift)` prepares the inverse action of
\(FF^T+\alpha I\) from stored `F:[...,N,L]`. For kernel systems,
`nystrom_inverse(C, W, shift, jitter=0.)` uses explicit landmark blocks
C=K(X,Z) `[...,N,L]` and W=K(Z,Z) `[...,L,L]`:

\[
\widetilde K=C(W+\delta I)^{-1}C^T,\qquad
M=(\widetilde K+\alpha I)^{-1}
\]

Alpha is `shift`, delta is `jitter`. C/W share float32/64 dtype/device with
broadcast batches. W must be symmetric, W+delta I positive definite; alpha
must be a positive finite scalar with finite reciprocal. Duplicate landmarks
commonly need explicit jitter. `nystrom_inverse` does not choose landmarks or
retry with hidden regularization; select them with
[pivoted Cholesky pivots](https://github.com/inspiros/torchpairwise/blob/master/docs/preconditioning.md#pivoted-cholesky-landmarks-and-inverse-actions)
or [streamed farthest-point sampling](spectral.md#gaussian-process-prediction).
Cholesky whitening and a thin SVD form the map without constructing an N-by-N
inverse.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], targets:[N,R], ids:[L] int64 landmark sample indices
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=0.2)
    rows = torch.arange(X.shape[-2], device=X.device)
    C, W = K.block(rows, ids), K.block(ids, ids)  # stored [N,L], [L,L]
    M = tp.linalg.nystrom_inverse(C, W, 0.03, jitter=1e-8)
    A = K + 0.03 * tp.eye_like(K)
    fit = tp.linalg.cg(A, targets, preconditioner=M, rtol=1e-8)
    if not fit.converged.all():
        raise RuntimeError(f"PCG status: {fit.status.tolist()}")
    # fit.solution:[N,R]; the original full kernel system is solved.
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X:[N,D], targets:[N,R], ids:[L] int64 landmark sample indices
at::NoGradGuard no_grad;
s::PairwiseOptions source;
source.comparison.gamma = 0.2;
source.chunk_size = 128;
auto K = s::kernel(X, c10::nullopt, "rbf", source);
auto rows = at::arange(X.size(-2), ids.options());
auto C = K.block(rows, ids), W = K.block(ids, ids);
auto M = s::linalg::nystrom_inverse(C, W, 0.03, 1e-8);
auto A = s::aslinearoperator(K).add_diagonal(0.03);
s::linalg::CGOptions solve;
solve.preconditioner = M;
solve.rtol = 1e-8;
auto fit = s::linalg::cg(A, targets, solve);
TORCH_CHECK(fit.converged.all().item<bool>(), "PCG did not converge");
// Ordinary features: s::linalg::lowrank_inverse(F, 0.03).
```

</div>
</div>

Setup is inference-only and snapshots the inputs. Reuse the map across solves;
rebuild after changing the kernel, features, landmarks or shift. Applying the
prepared map supports RHS gradients, transpose, composition, native solvers and
compile/export. The double basis/spectrum need O(Nr) storage with r=min(N,L),
plus scalar shift. Setup stores C/W and temporary double Cholesky/SVD blocks;
it does not stream the low-rank factor. With L near N, storage becomes quadratic.

For a thin spectral basis, the action is:

\[
MB=\frac{B-U(U^TB)}{\alpha}
 + U\operatorname{diag}\!\left((\sigma_k^2+\alpha)^{-1}\right)U^TB
\]

Double arithmetic reduces cancellation before casting back to the source dtype;
very small shifts can still amplify roundoff. There is no silent rank cutoff.
A preconditioner adds work to every iteration and has a setup cost, so compare
end-to-end time and memory as well as iteration counts. **CG still solves the
full original kernel system**, using the approximate inverse only to choose
search directions. The [Nyström example](https://github.com/inspiros/torchpairwise/blob/master/examples/nystrom_cg.py)
compares plain CG, Jacobi and Nyström, reporting setup separately and checking
held-out predictions against a dense reference. See the
[full contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/preconditioning.md#low-rank-inverse-actions).

## Symmetric Indefinite Systems

`tp.linalg.minres` solves real symmetric square maps that may be
indefinite, such as block KKT systems. Symmetry is a caller assumption, not an
exhaustive runtime check. A short Lanczos/QR recurrence retains O(NR) vector
blocks per batch rather than a full search basis. No preconditioner or
MINRES-QLP variant is provided.

The shape conventions match CG: A `[...,N,N]`, b `[N]` or `[...,N,R]`, with
broadcast batches, independent RHS convergence and an optional broadcast initial
guess. `max_iter=0` evaluates that guess; the budget counts all Lanczos steps.
Actual \(\|b-Ax\|_2\le\max(\mathrm{atol},\mathrm{rtol}\|b\|_2)\)
verifies convergence on the returned-precision solution. A false recurrence
estimate may restart unfinished columns from their actual residual.

`MINRESResult` returns `solution`, float64 `residual_norm`, int64 `iterations`
and `status`, and bool `converged`. Status 0 is convergence, 1 iteration limit,
2 Lanczos/QR breakdown without verified convergence, 3 nonfinite arithmetic.
Singular/inconsistent maps have no minimum-norm or least-squares guarantee;
use LSQR for that objective. Failure retains the current iterate.

Internal vectors/scalars use float64, while source products and returned
solutions retain supplied float32/float64 precision. Native execution is
inference-only, with compile/export support but no CUDA graph capture. See
[MINRES contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/minres.md)
and the [KKT gallery example](../examples/index.md#kernel-fitting-with-an-intercept).

## Nonsymmetric Systems

`tp.linalg.gmres` solves real square systems without CG's symmetry/PSD
assumptions. It uses a restarted Arnoldi basis, two modified Gram-Schmidt passes
and Givens rotations. `restart` bounds the basis width (default 20, clamped to N);
`max_iter` counts total Arnoldi steps per batch/RHS, not restart cycles.

Convergence checks the actual \(\|b-Ax\|_2\le\max(\mathrm{atol},\mathrm{rtol}\|b\|_2)\).
Map/RHS batches broadcast, with b `[N]` or `[...,N,R]` and matching solution
shape. `initial` broadcasts without adding batches. `max_iter=0` evaluates it.
`GMRESResult` returns `solution`, float64 `residual_norm`, int64 `iterations`
and `status`, and bool `converged`. Status 0 is convergence, 1 total iteration
limit, 2 Arnoldi/triangular breakdown without verified convergence, 3 nonfinite
arithmetic. Failure retains the current iterate; inspect diagnostics.

The solver is inference-only and supports compile/export, with no CUDA graph
capture or preconditioner. Source products/output use float32/float64 as supplied;
basis, orthogonalization and small QR blocks use float64. Effective restart width
m gives O(NRm+Rm²) workspace per batch plus source/consumer workspace. Restarts
may stagnate, and singular systems have no least-squares/minimum-norm guarantee.
See [personalized PageRank](../examples/index.md#personalized-pagerank) for
Python/C++ code and the [full GMRES contract](https://github.com/inspiros/torchpairwise/blob/master/docs/gmres.md).

## Rectangular Least Squares

`tp.linalg.lsqr(A, B, damp=0.2)` minimizes
\(\|AX-B\|_F^2+0.2^2\|X\|_F^2\) for real rectangular maps.
It uses A and A.T actions without constructing A.T A; no symmetry or PSD
assumption is required. A has shape `[...,N,M]`, B is `[N]` or `[...,N,R]`,
and the solution is `[...,M]` or `[...,M,R]`, with broadcast batches.

Convergence uses the actual normal residual
\(G=A^\mathsf T(B-AX)-\mathrm{damp}^2X\):

\[
\|G_{:r}\|_2\le\max(\mathrm{atol},\mathrm{rtol}\|(A^\mathsf T B)_{:r}\|_2)
\]

`LSQRResult` returns `solution`, float64 `residual_norm` (data residual) and
`normal_residual_norm` (stationarity), int64 `iterations`/`status` and bool
`converged`. A nonzero data residual may be optimal. Status 0 is convergence,
1 iteration limit, 2 recurrence breakdown without verified convergence, and
3 nonfinite arithmetic. Check diagnostics before using the result. Damping
always penalizes the final solution, including with a nonzero `initial`.

The solver is inference-only, with float32/float64 source products and solution
blocks, and double scalar recurrences. It supports native compile/export but
not CUDA graph capture. It retains O((N+M)R) solver blocks plus source workspace.
The [landmark regression gallery example](../examples/index.md#landmark-kernel-regression)
shows Python/C++ calls, held-out prediction and an independent dense check.
See [full LSQR contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/least-squares.md).

## Differentiable Square Solves

`solve(..., differentiable=True)` differentiates a converged nonsingular system
without retaining its iteration history. For incoming gradient G:

\[
AX=B,\qquad A^\mathsf T\Lambda=G,\qquad
\overline B=\Lambda,\qquad \overline A=-\Lambda X^\mathsf T
\]

The matrix cotangent is contracted through streamed source products and map
factors. Retained state includes operands, the fitted solution and narrow replay
blocks; a streamed source needs no dense gradient matrix.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], targets:[N,R], grad-enabled tensors
A = tp.lazy(X, metric="rbf", gamma=0.2)
A = A + 0.1 * tp.eye_like(A)
fit = tp.linalg.solve(A, targets, method="cg", differentiable=True, rtol=1e-7)
fit.solution.square().mean().backward()
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X:[N,D], targets:[N,R]; use a grad-enabled scope.
s::PairwiseOptions source;
source.comparison.gamma = 0.2;
auto A = s::aslinearoperator(s::kernel(X, c10::nullopt, "rbf", source)).add_diagonal(0.1);
s::linalg::SolveOptions options;
options.method = "cg";
options.differentiable = true;
options.rtol = 1e-7;
auto fit = s::linalg::solve(A, targets, options);
fit.solution.square().mean().backward();
```

</div>
</div>

The selected CG/MINRES/GMRES method also solves the adjoint, using the same
stopping options. Both solves must converge; failure raises. Only the solution
has first gradients; diagnostics are detached. Initial guesses must be detached,
CG supports a fixed SPD inverse-action preconditioner in both solves, with
no backward edge through its algorithmic-only operands. Shared system operands still
receive their mathematical gradients. Higher derivatives remain unsupported. Symmetry
assumptions still apply, and poor conditioning or loose tolerances can leave
inaccurate gradients. Named solvers remain inference-only.
`lstsq(..., damp=0.2, differentiable=True)` adds implicit gradients for uniquely
regularized rectangular least squares; see below.
See [full contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/linear-solves.md#implicit-solution-gradients).

## Differentiable Damped Least Squares

Positive damping gives a unique solution even for wide or rank-deficient maps.
For fixed damping alpha, residual R and solution cotangent G:

\[
(A^\mathsf T A+\alpha^2I)\Lambda=G,\qquad
\overline B=A\Lambda,\qquad
\overline A=R\Lambda^\mathsf T-(A\Lambda)X^\mathsf T,\qquad R=B-AX
\]

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], landmarks:[L,D], targets:[N,R]; grad-enabled operands
A = tp.lazy(X, landmarks, metric="rbf", gamma=0.3)
fit = tp.linalg.lstsq(A, targets, damp=0.2, differentiable=True, rtol=1e-7)
fit.solution.square().mean().backward()
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X:[N,D], landmarks:[L,D], targets:[N,R]; grad-enabled operands
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

</div>
</div>

Forward uses LSQR, backward CG on the damped normal action and streamed product
VJPs. No normal matrix, dense matrix cotangent or iteration history is stored
for a streamed source. Both solves must converge or raise. Only first gradients
through source inputs/factors and RHS are supported; damping is a fixed scalar,
initial guesses must be detached. Damping and its square must be finite and
strictly positive in the source dtype. Small damping/poor conditioning can
amplify errors; the backward normal action squares singular values. See
[full contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/linear-solves.md#differentiable-damped-least-squares).

## Group Projection and Kernel LDA

Group projection replaces each sample value by the mean of its group. Labels
are int64 `[...,N]`, with arbitrary signed IDs and broadcast batches. The map
snapshots group membership, stores IDs/counts, and supports value gradients.
`like` supplies only float32/float64 dtype and device.

\[
P=E\operatorname{diag}(n_c^{-1})E^\mathsf T,\qquad
(PV)_i=\frac{1}{n_{g_i}}\sum_{j:g_j=g_i}V_j,\qquad
P^\mathsf T=P,\quad P^2=P
\]

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# labels:[N], X:[N,D], V:[N,R]
P = tp.group_projection(labels, like=X)  # logical [N,N]
means = P.matmul(V)                      # [N,R], repeated group means
K = tp.lazy(X, metric="rbf", gamma=0.2)
R = K - P @ K  # (I-P)K
Sw = R.mT @ R
Sw = Sw + 0.1 * tp.eye_like(Sw)  # K(I-P)K + 0.1 I
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// labels:[N], X:[N,D], V:[N,R]
auto P = s::group_projection(labels, X);
auto means = P.matmul(V);
s::PairwiseOptions options;
options.comparison.gamma = 0.2;
auto K = s::aslinearoperator(s::kernel(X, c10::nullopt, "rbf", options));
auto R = K - P.compose(K);
auto Sw = R.transpose().compose(R).add_diagonal(0.1);
```

</div>
</div>

For symmetric K, `Sw` is the coefficient-regularized within-class scatter map.
The [kernel LDA example](https://github.com/inspiros/torchpairwise/blob/master/examples/kernel_lda.py)
solves against a class block F `[N,C]` with native CG, then diagonalizes
`F.T @ Sw^{-1}F` `[C,C]`. It retains at most C-1 positive discriminant directions.
No dense `[N,N]` array is needed in that path. The example checks a small dense
generalized eigenproblem, including eigenvalues and scatter orthogonality.

Group preparation sorts labels once, roughly O(N log N). Each action uses
scatter/gather with O(NR) work/workspace per broadcast partition. The map stores
two `[N]` integer arrays, not an `[N,N]` projector. The LDA class blocks are
separate `[N,C]` allocations; the advantage decreases as C approaches N.
See [partition contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/linear-operators.md#group-projection)
and [LDA formulas and dimensions](https://github.com/inspiros/torchpairwise/blob/master/docs/kernel-learning.md#kernel-lda-with-group-projection).

## Truncated Kernel PCA

Ordinary PCA covariance and LDA scatters are `[d,d]` for X `[N,d]`. With a
nonlinear kernel, the implicit feature dimension can be infinite, and exact
dual formulations use `[N,N]` matrices. A small input d does not reduce that
dual decomposition size.

An iterative approximation can instead build a basis Q `[N,r]`, using only
streamed centered products, then decompose a small projected matrix:

\[
H=I-\mathbf1\mathbf1^\mathsf{T}/N,\qquad
K_cV=H\bigl(K(HV)\bigr),\qquad B=Q^\mathsf{T}K_cQ
\]

The [kernel PCA example](https://github.com/inspiros/torchpairwise/blob/master/examples/kernel_pca.py)
uses native `tp.linalg.eigh`, with QR-stabilized block subspace iteration
and Rayleigh-Ritz projection. It returns training coordinates `[N,k]`, after
checking residuals and convergence. Each pass still evaluates all sample pairs;
storage includes `[N,r]` blocks, `[r,r]` projections and kernel tile workspace.
The no-grad example assumes a symmetric PSD kernel. Its optional dense reference
compares spectra/subspaces and checks linear and zero-rank cases.
[Held-out projection](../examples/index.md#pca-projection-for-new-samples) retains the
training kernel column means and applies `C.centered(reference_mean)` to the
rectangular test-to-training map. The same overload is available in C++.
It preserves query-batch independence and supports first gradients in map
actions; the eigensolve remains inference-only.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], N>=12
with torch.no_grad():
    A = tp.lazy(X, metric="rbf", gamma=0.2).centered()
    fit = tp.linalg.eigh(A, k=4, subspace_size=12, max_iter=100,
                       rtol=1e-6, atol=1e-10)
    if not fit.converged.all():
        raise RuntimeError(f"Eigen status: {fit.status.tolist()}")
    scores = fit.eigenvectors * fit.eigenvalues.clamp_min(0).sqrt()  # [N,4]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X:[N,D], N>=12
at::NoGradGuard no_grad;
s::PairwiseOptions options;
options.comparison.gamma = 0.2;
auto A = s::aslinearoperator(s::kernel(X, c10::nullopt, "rbf", options)).centered();
s::linalg::EighOptions eigen_options;
eigen_options.k = 4;
eigen_options.subspace_size = 12;
eigen_options.max_iter = 100;
eigen_options.rtol = 1e-6;
eigen_options.atol = 1e-10;
auto fit = s::linalg::eigh(A, eigen_options);
// Inspect fit.converged / fit.status before using the result.
auto scores = fit.eigenvectors * fit.eigenvalues.clamp_min(0).sqrt().unsqueeze(-2);
```

</div>
</div>

`EigenResult` contains descending `eigenvalues` `[...,k]`, `eigenvectors`
`[...,N,k]`, and `residual_norm`, `iterations`, `converged`, `status`, each
`[...,k]`. Residual norms are float64; counts/status int64 and convergence bool.
Each batch stops independently after all requested ranks converge. The criterion
checks the actual source action and vector orthogonality:

\[
\|Au_i-\lambda_i u_i\|_2\le\max(\mathrm{atol},\mathrm{rtol}|\lambda_i|),\qquad
U^\mathsf TU\approx I
\]

The width r defaults to `min(N,2*k+8)` or `initial.shape[-1]`. Require
`1<=k<=r<=N`. `initial` must have full numerical column rank, shape `[...,N,r]`,
and broadcast to the map's batches without adding batches. Otherwise the native
solver draws a random start from the current PyTorch RNG state. `max_iter=0`
evaluates the starting subspace without updates. The iteration count repeats
the batch's number of subspace updates across its k sorted ranks; individual
vectors are not locked and can rotate within repeated eigenspaces.

| Status | Meaning |
| --- | --- |
| 0 | Residual and orthogonality checks pass |
| 1 | Iteration limit reached |
| 2 | Rank-deficient initial block |
| 3 | Nonfinite input/action/arithmetic |
| 4 | Significantly negative projected direction |
| 5 | Loss of orthogonality |

Numerical failures return diagnostics; malformed options raise. Status 2-5 fails
the batch with NaN values/vectors/residuals. Iteration exhaustion retains finite
estimates. The solver does not clip eigenvalues; PCA clips tiny roundoff-negative
variance only when forming scores. Native early stopping synchronizes CUDA,
so CUDA graph capture is unsupported. Compatible compile/export retains a native
solve. Gradient-bearing sources/initial blocks need an explicit no-grad scope.

There is no smallest/indefinite eigenvalue mode, and PSD detection is not
exhaustive. Small residuals do not certify that a caller-supplied invariant
starting subspace did not miss larger eigenvalues. Use random starts, sufficient
width and more updates for flat spectra. A positive `atol` helps null directions;
negative Ritz values within `max(atol,roundoff_bound)` are tolerated. With r near
N the retained basis itself approaches dense storage.
See [solver contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/linear-operators.md#leading-eigenpairs)
and [scatter formulas and dimensions](https://github.com/inspiros/torchpairwise/blob/master/docs/kernel-learning.md).

## Rectangular Truncated SVD

For a rectangular scalar map A `[N,M]`, use `tp.linalg.svd` to estimate
leading singular triplets. Two-sided QR-stabilized iteration applies A and A.T,
then decomposes an `[r,r]` projection. It never forms A.T A.

\[
AV_k\approx U_k\Sigma_k,\qquad A^\mathsf T U_k\approx V_k\Sigma_k,\qquad
A_k=U_k\Sigma_kV_k^\mathsf T
\]

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], Y:[M,D], min(N,M)>=12; float32/64 inference inputs
A = tp.lazy(X, Y, metric="rbf", gamma=0.2)
fit = tp.linalg.svd(A, k=4, subspace_size=12, rtol=1e-6, atol=1e-9)
if not fit.converged.all():
    raise RuntimeError(f"SVD status: {fit.status.tolist()}")
# B:[M,R]. Reuse fitted factors without evaluating more source pairs.
approximation = fit.left_vectors @ (fit.singular_values[:, None] * (fit.right_vectors.T @ B))
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X:[N,D], Y:[M,D], min(N,M)>=12; float32/64 inference inputs
s::PairwiseOptions source;
source.comparison.gamma = 0.2;
auto A = s::aslinearoperator(s::kernel(X, Y, "rbf", source));
s::linalg::SVDOptions options;
options.k = 4;
options.subspace_size = 12;
options.rtol = 1e-6;
options.atol = 1e-9;
auto fit = s::linalg::svd(A, options);
TORCH_CHECK(fit.converged.all().item<bool>(), "SVD did not converge");
auto approximation = at::matmul(fit.left_vectors,
    fit.singular_values.unsqueeze(-1) *
    at::matmul(fit.right_vectors.transpose(-1,-2), B));  // [N,R]
```

</div>
</div>

The approximation code is unbatched. Results contain descending
`singular_values:[...,k]`, `left_vectors:[...,N,k]`, `right_vectors:[...,M,k]`
(columns V, not Vh), and per-triplet `residual_norm`, `iterations`, `converged`,
`status`. Residuals report the maximum of the two singular-equation errors;
orthogonality is also checked. Require `1<=k<=r<=min(N,M)`. The default width
is `min(N,M,2*k+8)` or `initial.shape[-1]`; initial is a full-rank right block
`[...,M,r]`, otherwise a random start is drawn from the current PyTorch RNG.
Initial batches may broadcast to the map but cannot add batches.

Retained numerical state is `O((N+M)*r+r*r)` double blocks plus source inputs,
preparation, tiles and outputs. Each pass still evaluates source pairs. After
fitting, factor products evaluate the approximate map without further pairs.
Width near min(N,M) approaches dense storage. Finite rank collapse and zero maps
are valid; use positive `atol` for null triplets. Flat spectra may need wider
subspaces or more updates, and small residuals alone do not certify global
leading order if the initial subspace misses a leading direction.

`max_iter` counts updates after the initial two-sided extraction; zero checks
only that extraction. Each batch stops when all requested triplets converge.
Status 0 converged, 1 iteration limit (finite estimates), 2 rank-deficient start,
3 nonfinite action/arithmetic, 4 loss of orthogonality. Status 2-4 returns NaN
estimates for the failed batch. Invalid arguments raise. This leading-only
solver is inference-only and excludes CUDA graph capture; compatible
compile/export retains a native operator. See
[full contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/truncated-svd.md)
and the [runnable approximation example](../examples/index.md#rectangular-low-rank-approximation).

## Alignment and Independence

For square Gram sources over the same N observations, let
\(H=I-\mathbf1\mathbf1^\mathsf{T}/N\).

\[
\operatorname{CKA}(K,L)=\frac{\langle HKH,HLH\rangle_F}{\|HKH\|_F\|HLH\|_F},\qquad
\operatorname{HSIC}_{b}=\frac{\langle HKH,HLH\rangle_F}{(N-1)^2}\]

Use `frobenius_inner_product(...,center=True,normalize=True)` for centered
alignment and `hsic` for biased or unbiased HSIC. Zero alignment norms give
zero values and first gradients. Biased HSIC requires N≥2; unbiased N≥4.
See [full alignment conventions](https://github.com/inspiros/torchpairwise/blob/master/docs/kernel-alignment.md).

## Geometric Interactions

Vector kernels additionally support weighted displacement and residual outer
products. For weights Kij, these aggregate terms such as
\(\sum_jK_{ij}(y_j-x_i)\) and
\(\sum_jK_{ij}(y_j-x_i)(y_j-x_i)^\mathsf{T}\) without storing all N×M
vectors or matrices. Normalized and centered variants have explicit contracts.
Finite-iteration Sinkhorn maps instead use transport-plan conditional means.
See [kernel geometry](https://github.com/inspiros/torchpairwise/blob/master/docs/kernel-moments.md)
and [transport maps](https://github.com/inspiros/torchpairwise/blob/master/docs/interaction-maps.md).


## Quadratic Optimization and Kernel SVM

`tp.optimize.box_qp` minimizes a real symmetric convex quadratic within
finite box bounds, optionally with one affine equality. It uses native source
products and projected steps; a generated `[N,N]` matrix is not retained.

\[
\min_x\ \tfrac12x^\mathsf TQx+q^\mathsf Tx,\qquad
\ell\le x\le u,\qquad a^\mathsf Tx=b
\]

For a binary C-SVM, use `Q=diag(y) K diag(y)`, `q=-ones`, bounds `[0,C]` and
`y.T @ alpha=0`. The equality is needed for a free intercept.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], labels:[N] floats {-1,+1}, both classes present
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=0.5)
    Q = K.weight_rows(labels).weight_columns(labels)
    fit = tp.optimize.box_qp(Q, -torch.ones_like(labels), lower=0., upper=1.,
                            equality=(labels, 0.), rtol=1e-6, max_iter=1500)
    if not fit.converged:
        raise RuntimeError(f"QP status: {fit.status.item()}")
    raw_scores = K.matmul(fit.solution * labels)  # [N], before intercept
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X:[N,D], labels:[N] floats {-1,+1}
at::NoGradGuard no_grad;
s::PairwiseOptions kernel_options;
kernel_options.comparison.gamma = 0.5;
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
// Inspect fit.converged / fit.status before using coefficients.
auto raw_scores = K.matmul(fit.solution * labels);
```

</div>
</div>

q `[N]` returns solution `[...,N]`; q `[...,N,R]` defines independent objectives
and returns `[...,N,R]`. Q/q batches broadcast; use `[...,N,1]` for batched
vectors. Bounds, equality vectors and initial guesses broadcast to the solution
without adding batches; `[N]` bounds/vectors mean samples. Equality targets
broadcast to `[...,R]`. Numeric bounds/targets become tensors matching q; Tensor
arguments already need the same float32/float64 dtype and device. Bounds must
be finite. `max_iter=0` projects/evaluates the initial guess, defaulting to zero.

`QPResult` exposes `solution`, `objective`, `iterations`, `converged`, `status`,
`optimality` and `feasibility`. Diagnostics shape is `[...]` for vector q or
`[...,R]` for block q. Objective/residuals are float64; status/counts int64 and
convergence bool. Each batch/RHS stops independently. Optimality is the infinity
norm of the projected-gradient mapping; feasibility covers box violations and
the normalized equality gap. The native method uses safeguarded Newton/bisection
for equality projection and a quadratic line minimization along feasible segments.

| Status | Meaning |
| --- | --- |
| 0 | Optimality and feasibility checks pass |
| 1 | Iteration limit |
| 2 | Infeasible bounds/equality |
| 3 | Nonfinite input/action/arithmetic |
| 4 | Negative search curvature |
| 5 | Projection failed numerically |
| 6 | No representable progress before convergence |

Malformed arguments raise. Iteration exhaustion retains finite approximations;
status 2-6 returns NaN solution/objective/residuals. Symmetry/PSD are assumptions:
negative curvature detection is not an exhaustive convexity test. Inference-only
execution requires explicit no-grad for gradient-bearing inputs. Compatible
compile/export retains one native solve; CUDA graph capture is unsupported.
Vector-sized solver/constraint blocks and source tiles still consume storage;
repeated products/projections can be slower than specialized dense solvers.

The [binary kernel SVM example](https://github.com/inspiros/torchpairwise/blob/master/examples/kernel_svm.py)
recovers an intercept from free supports or its KKT interval, reports a primal-dual
gap, and optionally compares the same problem with a dense matrix. See
[complete QP contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/quadratic-optimization.md).

## Entropic Transport

`tp.optimize.sinkhorn` solves entropic optimal transport for any scalar lazy
cost: a named metric, an entrywise expression graph, a `make_pairwise`
comparison or a dense matrix wrapped with `tp.as_lazy`. Masses `a:[...,N]` and
`b:[...,M]` may be nonuniform. Balanced problems need equal totals; masses are
never silently normalized. The solver returns log-domain potentials `f` and `g`,
and the coupling is never stored:

\[
\pi_{ij}=a_ib_j\exp\bigl((f_i+g_j-C_{ij})/\varepsilon\bigr)
\]

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# x:[N,D], y:[M,D], a:[N], b:[M] masses with equal totals
C = tp.lazy(x, y, metric="sqeuclidean", chunk_size=64)
fit = tp.optimize.sinkhorn(C, a, b, epsilon=0.1)
if not fit.converged.all():
    raise RuntimeError(f"Sinkhorn status: {fit.status.tolist()}")
plan = tp.optimize.sinkhorn_plan(C, fit, a, b, epsilon=0.1)  # lazy [N,M]
row_mass = plan.sum(-1)                                      # [N]
mapped = (plan @ y) / row_mass.unsqueeze(-1)                 # [N,D]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
namespace tp = torchpairwise::ops;
// x:[N,D], y:[M,D], a:[N], b:[M] masses with equal totals
tp::PairwiseOperationOptions source;
source.chunk_size = 64;
auto C = tp::pairwise_expression(x, y, "sqeuclidean", false, source);
tp::SinkhornOptions options;
options.epsilon = 0.1;
auto fit = tp::sinkhorn(C, a, b, options);
TORCH_CHECK(fit.converged.all().item<bool>(), "Sinkhorn did not converge");
auto plan = tp::sinkhorn_plan(C, fit, a, b, options.epsilon);  // lazy [N,M]
auto row_mass = plan.reduce("sum", -1);                        // [N]
auto mapped = plan.matmul(y) / row_mass.unsqueeze(-1);         // [N,D]
```

</div>
</div>

The plan stays lazy, so its sums and products stream cost tiles. The map is
defined only for rows with positive transported mass. Pass the same cost,
masses and `epsilon` to `sinkhorn_plan` that were used for the solve; the result
does not store them. `epsilon` has the units of the cost and must be positive.
Check `converged` before using potentials: `max_iter`, `rtol`, `atol` and
`check_every` control stopping, `initial=(f, g)` warm-starts, `epsilon_scaling`
anneals small regularization and `reach=rho` replaces the marginal constraints
by KL penalties for unbalanced problems.

`tp.optimize.sinkhorn_divergence` combines three solves into
\(\mathrm{OT}(a,b)-\tfrac12\mathrm{OT}(a,a)-\tfrac12\mathrm{OT}(b,b)\) and is
available in Python only:

```python
C_xx = tp.lazy(x, metric="sqeuclidean", chunk_size=64)
C_yy = tp.lazy(y, metric="sqeuclidean", chunk_size=64)
loss = tp.optimize.sinkhorn_divergence(C, C_xx, C_yy, a, b, epsilon=0.1)
```

Gradients use the envelope derivative of the converged dual: the cost
derivative is the coupling, and it reaches metric inputs, parameters and masses.
The iterations and returned potentials are not differentiated. The potentials
and objective use float64. Each half-step still visits every pair. For
point-set layouts and the fixed-cost catalog entry, see
[Sinkhorn divergence](../metrics/sinkhorn_divergences.md). The
[entropic transport notes](https://github.com/inspiros/torchpairwise/blob/master/docs/entropic-transport.md)
give the objectives, unbalanced duals and gradient conventions.

## Traces and Matrix Functions

The [spectral consumers](spectral.md) provide exact/Hutchinson traces, exact/SLQ
SPD log-determinants, symmetric exponential actions and SPD inverse-square-root
actions, plus [Gaussian process predictive variance](spectral.md#gaussian-process-prediction)
and landmark selection. They consume the same maps and expose Python/C++ APIs
with explicit probe counts, randomness and Krylov budgets. Trace/logdet also expose opt-in first
estimate gradients; SLQ selects its solve-based derivative explicitly. The [gallery example](../examples/index.md#kernel-log-determinants-and-graph-diffusion)
combines them for graph diffusion, whitening and kernel log-determinants.
