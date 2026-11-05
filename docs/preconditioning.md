# Preconditioned Conjugate Gradient

`streaming.linalg.cg(A, b, preconditioner=M)` solves the original SPD system
using a fixed SPD **inverse action** M approximating A inverse. Pass an action
that already applies the approximation to the inverse, not a matrix for the
solver to invert. The recurrence replaces the residual search vector with:

\[
z_k=M r_k,\qquad \rho_k=r_k^Tz_k,\qquad
\alpha_k=\frac{\rho_k}{p_k^TAp_k},\qquad
\beta_k=\frac{\rho_{k+1}}{\rho_k}
\]

The solver applies A and M separately, with independent RHS/batch convergence.
Success still requires the actual residual of the original system:

\[
\|b-Ax\|_2\le\max(\mathrm{atol},\mathrm{rtol}\|b\|_2)
\]

## Diagonal Maps and Jacobi

`tp.diag_embed(d)` represents diag(d) for real float32/64 `d:[...,N]`. Its action
is `d.unsqueeze(-1) * V`, using O(N) stored entries per batch and no `[N,N]`
allocation. It supports transpose, composition, first gradients, native solver
plans and compile/export. Entries may be zero or negative for general maps;
this does not make them valid CG preconditioners. Value mutations remain visible;
reconstruct the map after resizing the entries.

`tp.linalg.jacobi(d)` checks finite, strictly positive entries with finite
reciprocals and returns `tp.diag_embed(1/d)`. The reciprocals are computed once.
Rebuild this action after changing d or the system; values are not automatically
refreshed. The helper accepts a supplied diagonal, not an arbitrary composed map.
Scalar comparison sources have `.diagonal()` when extraction is needed.

```python
import torchpairwise as tp

# X:[N,D], targets:[N,R], d:[N] strictly positive
with torch.no_grad():
    D = tp.diag_embed(d.sqrt())
    K = tp.lazy(X, metric="rbf", gamma=0.2)
    K = K + 2. * tp.eye_like(K)
    A = D @ K @ D              # D(K+2I)D
    M = tp.linalg.jacobi(3. * d)             # diagonal(A) = 3*d
    fit = tp.linalg.cg(A, targets, preconditioner=M, rtol=1e-8)
    if not fit.converged.all():
        raise RuntimeError(f"PCG status: {fit.status.tolist()}")
    weights = fit.solution                  # [N,R]
```

```cpp
#include <torchpairwise.h>
namespace s = torchpairwise::streaming;
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
auto weights = fit.solution;                // [N,R]
```

## Low-Rank Inverse Actions

`tp.linalg.lowrank_inverse(F, shift)` prepares a fixed inverse action for
`F:[...,N,L]` and a strictly positive scalar shift:

\[
M=(FF^T+\alpha I)^{-1},\qquad F=U\Sigma V^T
\]

A thin SVD gives r=min(N,L) columns. For an RHS block B, application separates
its components in and outside that basis:

\[
MB=\frac{B-U(U^TB)}{\alpha}
  +U\operatorname{diag}\!\left(\frac{1}{\sigma_k^2+\alpha}\right)U^TB
\]

This also accepts dependent/zero feature columns and empty sample/feature axes.
It forms no N-by-N inverse or complementary projector. The stored basis and
spectrum use double precision, with double internal applications and output
cast to the supplied float32/64 dtype. This reduces cancellation from a
source-precision Woodbury subtraction; very small shifts relative to the
spectrum can still amplify roundoff. No numerical rank is chosen or silently
truncated.

Setup is inference-only. Use explicit no-grad when F or shift requires gradients.
After setup, the map supports RHS gradients, transpose, composition and native
solver plans. It snapshots F and shift; changing the original tensors does not
refresh the factorization. Setup and actions support compile/export, but setup
does not support CUDA graph capture. Reuse the map outside a compiled solve to
avoid repeating setup. Passing a prepared map into CG does not modify its state.

F must be finite strided float32/64. `shift` can be a number or zero-dimensional
Tensor matching F dtype/device; it must be finite, positive with a finite
reciprocal. A Tensor shift has no batch axis. The returned map has shape
`[...,N,N]`, with the feature batch shape. Its persistent state is a double
`[...,N,r]` basis, double `[...,r]` spectrum and scalar shift. Setup allocates
double feature/SVD blocks, including a thin `[...,r,L]` right factor and vendor
workspace. Each action uses O(NrR) work and O(NR+rR) temporary blocks per batch.
No streaming claim is made for the low-rank factor itself: its entries are stored.

## Nyström Kernel Preconditioning

`tp.linalg.nystrom_inverse(C, W, shift, jitter=0.)` accepts explicit kernel blocks
C=K(X,Z) `[...,N,L]` and W=K(Z,Z) `[...,L,L]`. It prepares:

\[
\widetilde K=C(W+\delta I)^{-1}C^T,\qquad
M=(\widetilde K+\alpha I)^{-1}
\]

Here alpha is `shift`, while delta is `jitter`. A Cholesky factor
`W+delta I = LL.T` whitens C to `F=C L^{-T}` before the thin SVD. No explicit
inverse is formed. Kernel symmetry/PSD and useful landmarks are the caller's
responsibility. C/W use matching float32/64 dtype/device and broadcast batches.
W must be finite/symmetric within roundoff; the regularized W must be positive
definite. Jitter is a finite nonnegative number and defaults to zero. Dependent
landmarks commonly need positive jitter. There is no hidden jitter, retry,
landmark selection or automatic approximation of A.

The inverse action approximates the original system's inverse. **CG still
solves the full original kernel system** and checks its true residual; it does
not switch to fitting the approximate kernel. Landmark construction materializes
N-by-L and L-by-L blocks. Setup uses O(NL+L²) blocks plus SVD workspace. Once
prepared, C/W may be released; the map retains only its spectral snapshot.
With L near N, the storage advantage disappears. Small L can give an ineffective
preconditioner; a larger L increases setup and per-iteration cost.

```python
# X:[N,D], targets:[N,R], ids:[L] int64 landmark sample indices
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=0.2)
    rows = torch.arange(X.shape[-2], device=X.device)
    C, W = K.block(rows, ids), K.block(ids, ids)   # stored [N,L], [L,L]
    M = tp.linalg.nystrom_inverse(C, W, 0.03, jitter=1e-8)
    A = K + 0.03 * tp.eye_like(K)
    fit = tp.linalg.cg(A, targets, preconditioner=M, rtol=1e-8)
    if not fit.converged.all():
        raise RuntimeError(f"PCG status: {fit.status.tolist()}")
```

```cpp
at::NoGradGuard no_grad;
s::PairwiseOptions source;
source.comparison.gamma = 0.2;
auto K = s::kernel(X, c10::nullopt, "rbf", source);
auto rows = at::arange(X.size(-2), ids.options());
auto C = K.block(rows, ids), W = K.block(ids, ids);
auto M = s::linalg::nystrom_inverse(C, W, 0.03, 1e-8);
// For ordinary features: auto M = s::linalg::lowrank_inverse(F, 0.03);
auto A = s::aslinearoperator(K).add_diagonal(0.03);
s::linalg::CGOptions solve;
solve.preconditioner = M;
solve.rtol = 1e-8;
auto fit = s::linalg::cg(A, targets, solve);
TORCH_CHECK(fit.converged.all().item<bool>(), "PCG did not converge");
```

The [Nyström example](../examples/nystrom_cg.py) fits exact kernel ridge, then
predicts held-out samples through the full rectangular kernel. It compares CG,
constant-diagonal Jacobi-PCG and Nyström-PCG at the same residual tolerance.
Setup time/peak memory and iterative solve time/peak memory are reported
separately, alongside setup plus one solve. One explicit landmark subset is
selected by the example; it is not an automatic library policy. The optional
dense oracle verifies coefficients and held-out predictions after measurements.

## Contract and Tradeoffs

A and M must be fixed real symmetric positive definite maps. This is a caller
assumption, not an exhaustive runtime test. Python accepts the same dense tensors,
lazy pairwise objects as A; C++ uses `CGOptions.preconditioner`,
an optional lazy pairwise object. Dtype/device and `[N,N]` dimensions must match.
M batches may broadcast to the existing solution batches but cannot add batches.

Nonpositive residual energy `r.T @ M @ r` gives
`CGStatus.INVALID_PRECONDITIONER` (4), named `InvalidPreconditioner` in C++.
Nonfinite preconditioner actions use existing status 3. Other statuses, initial
guesses and diagnostics follow [CG](linear-operators.md#conjugate-gradient).
`max_iter=0` evaluates the initial guess without applying M. Numerical failures
retain the current iterate. Shape/type errors raise.

CG remains inference-only: all gradient-bearing system and preconditioner operands
need explicit no-grad. Diagonal actions and reciprocal construction support
ordinary gradients outside a solve. Compile/export preserve a native solve;
native stopping still synchronizes CUDA and excludes CUDA graph capture.
The old unpreconditioned serialized operator schema remains available.

PCG adds an M action each iteration and retains O(NR) vector blocks, plus the
preconditioner's own state and workspace. Fewer steps do not guarantee less wall
time. Jacobi addresses uneven diagonal scaling; it cannot remove all spectral
ill-conditioning. For constant-diagonal kernels such as a plain RBF Gram matrix,
it is just scalar scaling and offers no iteration benefit. No preconditioner is
chosen or computed automatically. MINRES, GMRES and LSQR do not accept this
option.

The runnable [comparison](../examples/preconditioned_cg.py) reports CG/PCG steps,
solve time and incremental CUDA peak memory on the same system. The reciprocal
setup is outside the timed solves; source preparation is included. CPU timings
do not report memory. Both paths avoid dense storage. `--dense-reference` adds
a small direct oracle after those measurements; the dense path is not timed.

## Pivoted Cholesky Landmarks and Inverse Actions

`tp.linalg.pivoted_cholesky(K, rank, rtol=0., atol=0.)` accepts a scalar
entrywise symmetric PSD operator, including a kernel or `tp.as_lazy(matrix)`.
Linear-only compositions are rejected: the algorithm needs the diagonal and
individual columns. Native setup reads the diagonal, then evaluates one column
block per pivot and updates a double residual diagonal. It evaluates O(Nr)
entries and performs O(Nr²) arithmetic without forming N-by-N storage.

`PivotedCholeskyResult` contains `factor:[...,N,rank]`, integer
`pivots:[...,rank]`, and batch-shaped `error` and effective `rank`.
`error` is the residual diagonal trace. Each batch stops independently when
that trace is at most `max(atol, rtol * initial_trace)`, or at the rank budget.
Unused factor columns are zero and unused pivots are -1. Numerical outputs use
the source dtype after float64 internal work. Setup is inference-only.

```python
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=.2)
    result = tp.linalg.pivoted_cholesky(K, rank=16, rtol=1e-6)
    M = tp.linalg.lowrank_inverse(result.factor, noise)
    fit = tp.linalg.cg(K + noise * tp.eye_like(K), targets,
                       preconditioner=M, rtol=1e-8)
```

This still solves the original system. Rank controls approximation quality and
preconditioner cost; compare actual CG iterations and setup time. The pivot
criterion is the largest residual diagonal, with the lowest index breaking ties,
following Harbrecht et al. (2012).

Pivots also provide automatic Nyström landmarks. For an unbatched kernel, remove
padding before extracting the blocks:

```python
with torch.no_grad():
    ids = result.pivots[result.pivots >= 0]
    rows = torch.arange(X.shape[-2], device=X.device)
    C, W = K.block(rows, ids), K.block(ids, ids)
    M = tp.linalg.nystrom_inverse(C, W, noise, jitter=1e-8)
```

Select each batch's valid pivots independently when effective ranks differ.
The low-rank inverse recipe accepts the padded factor directly.
