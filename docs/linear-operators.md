# Linear Operators

Lazy pairwise objects act as linear maps. Products of lazy objects have linear
capability, preserving matrix actions without exposing entrywise operations.
Python and C++ assemble graphs of source products and native linear primitives.

```python
import torchpairwise as tp

# X:[N,D], Y:[M,D], V:[N,R]
K = tp.lazy(X, Y, metric="rbf", gamma=0.2)
A = K.centered()       # [N,M]
B = A @ A.mT + 0.01 * tp.eye_like(A @ A.mT)      # [N,N]
result = B.matmul(V)                      # [N,R]
```

Use `tp.as_lazy` to wrap a caller-owned dense matrix. Event sources must
contract every event axis first. Custom or prepared-encoder sources retain their
callback, feature-retention and deployment contracts.

## Matrix Algebra and Entrywise Algebra

| Operation | Meaning |
| --- | --- |
| `A @ B` | Matrix composition |
| `A @ V`, `A.matmul(V)` | Apply to dense `[M]` or `[...,M,R]` values |
| `A.mT` | Logical real transpose; reverse composition order |
| `A + B`, `A - B`, `-A` | Add/subtract map actions or negate one |
| `alpha * A` | Scalar scaling |
| `A + alpha * tp.eye_like(A)` | Square A plus alpha times the identity |
| `A.centered()` | Center both sample axes |
| `A.centered(reference_mean)` | Center against fitted training column means |
| `tp.diag_embed(d)` | Compact diagonal map from `d:[...,N]` |

For entrywise sources, `K.square()` and `K1 * K2` operate on individual
entries. Matrix products have linear capability; retain an entrywise source
when you need its statistics or selected entries.

## Centering and Diagonal Shifts

With H_L = I - 11.T/L, a rectangular centered map applies:

\[
A_c=H_N A H_M,\qquad A_c V=H_N\bigl(A(H_M V)\bigr)
\]

Input and output sample means implement these actions without storing H.
For a held-out kernel C `[T,N]`, retain the column means mu `[N]` of the
training Gram matrix and call `C.centered(mu)`:

\[
C_c=(C-\mathbf1_T\mu^\mathsf T)H_N,\qquad
(C_c)_{ij}=C_{ij}-\frac1N\sum_l C_{il}-\mu_j+\frac1N\sum_l\mu_l
\]

Compute `mu = K_train.reduce("mean", dim=-2)` once; use exactly the same
training references, order and kernel settings at projection time. This avoids
query-batch-dependent coordinates. No test-set column mean is estimated.
Python and C++ expose the same `centered(mu)` overload. The mean must be strided
float32/64 `[...,N]` matching source dtype/device, with N positive; leading
batches broadcast and query axes may be empty. It remains referenced, supports
first gradients, and value mutations affect later products. For fitted inference
state use `mu.detach().clone()` explicitly. Both actions and transposes stream;
only source tiles and vector blocks are evaluated.
For square A, adding alpha to its diagonal means:

\[
(A+\alpha I)V=AV+\alpha V
\]

This is different from the scalar source expression `K + alpha`, which shifts
every entry. Adding a diagonal does not by itself establish positive
definiteness. Symmetry and positive definiteness are not inferred.

## Shapes, Gradients and Storage

Maps and RHS/factor tensors use matching float32/float64 dtype and device.
Python numeric factors are converted to that dtype; Tensor factors must have
zero dimensions and may receive gradients. Leading batches broadcast. A vector
RHS `[M]` produces `[...,N]`; a matrix RHS `[...,M,R]` produces `[...,N,R]`.
Use a final singleton R axis for batched vectors to avoid shape ambiguity.
R must be positive, matching the scalar-source product contract. Empty sample
axes are supported; they are distinct from an empty RHS channel axis.

The adapters retain source references and shape metadata. Value mutations are
visible on later products. Reconstruct maps after resizing or changing the
layout/dtype/device of bound tensors. Composite sources retain the existing
first-derivative restrictions; differentiable scalar factors and RHS tensors
use the native ATen adjoints. Dynamic compile/export is supported for compatible
sources; Python callbacks do not gain deployment support through adaptation.

A composition `(A @ B) @ V` applies B to V and then A to the result. It does not
form AB, but intermediate vector blocks use storage, and autograd may retain
them for backward. Reusing a graph node recomputes its products; no persistent
matrix or result cache is implied. Work and memory depend on graph depth, block
width, source tiles and backward state. Dense V is never automatically split
into a separate user-visible lazy graph.

## C++ / LibTorch

```cpp
#include <torchpairwise.h>
namespace s = torchpairwise::streaming;

s::PairwiseOptions options;
options.chunk_size = 128;
options.comparison.gamma = 0.2;
auto K = s::kernel(X, Y, "rbf", options);
auto A = s::aslinearoperator(K).centered();
auto B = A.compose(A.transpose()).add_diagonal(0.01);
auto result = B.matmul(V);                 // [...,N,R]
```

C++ uses `compose` for matrix composition and `scale` for scalar multiplication.
Both interfaces expose `shape`, `dtype` and `device`; C++ uses accessors such as
`A.shape()`. Source factories return the same native ScalarExpression/event
types as before. The [runnable Python example](../examples/linear_operators.py)
checks a centered kernel normal-map product and its first gradients against a
small dense reference.

## Solve Entry Points

`streaming.linalg.solve(A, B, method=...)` selects `"cg"`, `"minres"` or `"gmres"`
for square maps. `streaming.linalg.lstsq(A, B)` selects LSQR for general least
squares. Matching C++ `solve`/`lstsq` use `SolveOptions`/`LstsqOptions`. No dense
inverse or pseudoinverse is created; diagnostics and numerical behavior come
from the selected native solver. Opt-in `solve(..., differentiable=True)` adds
implicit first gradients through a converged nonsingular square solution;
CG supports fixed preconditioners in that mode; higher derivatives remain
unsupported. `lstsq(..., differentiable=True)` supports positive damping.
See [method selection, shapes and inverse-action
contracts](linear-solves.md). The named solvers below remain available.

## Conjugate Gradient

`tp.linalg.cg` solves a real symmetric positive-definite system using
native C++/ATen iterations. Pass a lazy pairwise object or
dense Tensor. It does not check symmetry or prove positive definiteness.
Each batch/RHS column follows an independent CG recurrence, not block CG.

For kernel ridge regression without a fitted intercept:

\[
(K+\alpha I)C=Y,\qquad \widehat Y=KC,\qquad \alpha>0
\]

```python
# X:[N,D], targets:[N,R]
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=0.2)
    A = K + 0.1 * tp.eye_like(K)
    fit = tp.linalg.cg(A, targets, rtol=1e-6, max_iter=300)
    if not fit.converged.all():
        raise RuntimeError(f"CG status: {fit.status.tolist()}")
    predictions = K.matmul(fit.solution)        # [N,R]
```

```cpp
at::NoGradGuard no_grad;
auto K = s::kernel(X, c10::nullopt, "rbf", options);
s::linalg::CGOptions solve_options;
solve_options.rtol = 1e-6;
solve_options.max_iter = 300;
auto fit = s::linalg::cg(
    s::aslinearoperator(K).add_diagonal(0.1), targets, solve_options);
auto predictions = K.matmul(fit.solution);     // [N,R]
// Inspect fit.converged / fit.status before using the approximate solution.
```

The small dense equivalent is `torch.linalg.solve(K_dense + alpha * I, targets)`.
See the [runnable dense-checked example](../examples/kernel_ridge.py).
CG evaluates all pairs for every source matvec. It trades repeated computation
for avoiding the full generated matrix; it is not necessarily faster than a
dense factorization. Storage includes source preparation, source tiles, several
solution-sized vector blocks, and any intermediate blocks from map composition.
A wrapped dense Tensor retains its existing matrix allocation.

### Shapes and Diagnostics

RHS `[N]` returns solution `[...,N]` and diagnostics `[...]`. RHS `[...,N,R]`
returns solution `[...,N,R]` and diagnostics `[...,R]`. Operator/RHS batches
broadcast; use `[...,N,1]` for batched vectors. An optional `initial` must have
the same sample/channel axes and may broadcast to the solution without adding
batch dimensions. `max_iter=0` evaluates the initial guess, which defaults to
zero. Empty sample/batch axes work; the RHS channel count R must be positive.

`CGResult` has `solution`, `residual_norm`, `iterations`, `converged`, `status`.
The norm is float64; counts/status are int64 and convergence is bool. Norms and
inner products accumulate in float64 for either float32 or float64 systems.
A column succeeds only when the true residual satisfies:

\[
\|b-Ax\|_2\le\max(\mathrm{atol},\mathrm{rtol}\|b\|_2)
\]

| Status | Python Name | Meaning |
| --- | --- | --- |
| 0 | `CGStatus.CONVERGED` | True residual meets tolerance |
| 1 | `CGStatus.ITERATION_LIMIT` | Iteration budget exhausted |
| 2 | `CGStatus.NONPOSITIVE_CURVATURE` | Search direction has nonpositive curvature |
| 3 | `CGStatus.NONFINITE` | Nonfinite RHS, initial state, operator action or arithmetic |
| 4 | `CGStatus.INVALID_PRECONDITIONER` | Nonpositive residual energy under the inverse action |

The C++ enum uses `Converged`, `IterationLimit`, `NonpositiveCurvature`,
`Nonfinite`, `InvalidPreconditioner`. Converged/failed columns stop updating while others continue.
Iterations count attempted steps. Invalid shapes/options raise; numerical
nonconvergence returns a result for inspection. Failure detection is not an
exhaustive test of the SPD assumption.

### Preconditioning

Pass `preconditioner=M` for a fixed SPD inverse action approximating A inverse.
`tp.linalg.jacobi(d)` builds `tp.diag_embed(1/d)` from a supplied positive finite
system diagonal. The original residual still determines convergence. No
preconditioner is inferred automatically; a constant-diagonal RBF system gains
nothing from Jacobi. See [preconditioning](preconditioning.md) for Python/C++
examples, shapes, mutation semantics and the runnable comparison.
`lowrank_inverse(F, shift)` and `nystrom_inverse(C, W, shift, jitter=...)`
prepare inverse actions from stored features or landmark blocks. Their native
setup is inference-only; the prepared maps support RHS gradients and snapshots
that can be reused across solves. CG still applies the original full system.

### Deployment and Gradients

The solver is inference-only. Gradient-bearing inputs require an explicit
`torch.no_grad()` / `at::NoGradGuard`; otherwise the native entry raises.
This includes source tensors, tensor options, factors, RHS and initial guesses.
Source product gradients remain available outside the solve.

`torch.compile` and `torch.export` preserve a single native solve operator,
including Tensor-valued diagnostics, rather than unrolling its iterations.
Compatible source deployment requirements still apply. Native early stopping
uses host scalar synchronization; CUDA graph capture is unsupported and raises.

## Group Projection

`streaming.group_projection(labels, like=X)` constructs the symmetric projector
that replaces each sample value by the mean of its group. Let E be a conceptual
one-hot membership matrix `[N,C]` and let n_c be the group counts:

\[
P=E\operatorname{diag}(n_c^{-1})E^\mathsf T,\qquad
(PV)_i=\frac{1}{n_{g_i}}\sum_{j:g_j=g_i}V_j,\qquad P^\mathsf T=P,\quad P^2=P
\]

```python
# labels:[N], X:[N,D], V:[N,R]
P = tp.group_projection(labels, like=X)    # logical [N,N]
group_means = P.matmul(V)                # [N,R], means repeated per sample
within_residuals = V - group_means       # [N,R]
K = tp.lazy(X, metric="rbf", gamma=0.2)
R = K - P @ K                    # (I-P)K
Sw = R.mT @ R + 0.1 * tp.eye_like(R.mT @ R)    # K(I-P)K + 0.1 I for symmetric K
```

```cpp
// labels:[N], X:[N,D], V:[N,R]
auto P = s::group_projection(labels, X);
auto group_means = P.matmul(V);
auto within_residuals = V - group_means;
auto K = s::aslinearoperator(s::kernel(X, c10::nullopt, "rbf", options));
auto R = K - P.compose(K);
auto Sw = R.transpose().compose(R).add_diagonal(0.1);
```

The factory accepts strided int64 labels `[...,N]`, including negative and
nonconsecutive IDs. Each leading batch defines its own partition. Label batches
and RHS batches broadcast. `like` supplies only the float32/float64 dtype and
device; its values and shape are unused, and it receives no gradient from this
factory. Labels must already share its device. Prepared IDs/counts snapshot the
partition: later label mutations have no effect, so reconstruct to change groups.
Values retain the ordinary lazy pairwise object `[N]` / `[...,N,R]` contracts, including
empty sample/batch axes. Groups must be discrete; labels have no gradient.

The implementation sorts labels once and retains two `[...,N]` integer arrays.
Application uses scatter/gather with vector-sized workspace `[...,N,R]`, rather
than constructing P `[N,N]` or E `[N,C]`. For each partition, setup is dominated
by sorting, roughly O(N log N); each application takes O(NR) work and workspace.
Leading batch broadcasting also contributes to output/workspace size. The count
storage reserves N slots to keep shapes independent of the number of groups.
CUDA scatter sums may differ slightly with accumulation order.

The map supports ordinary matrix algebra, native CG graphs, value gradients,
compilation and export. The [kernel LDA example](../examples/kernel_lda.py) uses
it with CG and a class-sized eigensolve; it retains `[N,C]` blocks for the class
subspace, but no `[N,N]` array in the streamed path.

## Leading Eigenpairs

`streaming.linalg.eigh(A,k=...)` computes leading eigenpairs of a real symmetric
PSD map with QR-stabilized block subspace iteration and Rayleigh-Ritz projection.
Only the small projected matrix is diagonalized. It supports float32/float64,
leading batches, native Python/C++ execution and compile/export. It is
inference-only, as is CG; native early stopping does not support CUDA graph capture.

```python
# X:[N,D]
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=0.2)
    A = K.centered()
    fit = tp.linalg.eigh(A, k=4, subspace_size=12, max_iter=100,
                       rtol=1e-6, atol=1e-10)
    if not fit.converged.all():
        raise RuntimeError(f"Eigen status: {fit.status.tolist()}")
    coordinates = fit.eigenvectors * fit.eigenvalues.clamp_min(0).sqrt()  # [N,4]
```

```cpp
at::NoGradGuard no_grad;
auto K = s::kernel(X, c10::nullopt, "rbf", options);
auto A = s::aslinearoperator(K).centered();
s::linalg::EighOptions eigen_options;
eigen_options.k = 4;
eigen_options.subspace_size = 12;
eigen_options.max_iter = 100;
eigen_options.rtol = 1e-6;
eigen_options.atol = 1e-10;
auto fit = s::linalg::eigh(A, eigen_options);
// Inspect fit.converged / fit.status before using the result.
auto coordinates = fit.eigenvectors * fit.eigenvalues.clamp_min(0).sqrt().unsqueeze(-2);
```

The eigenvalues are in descending order. For A `[...,N,N]`, `EigenResult` holds
`eigenvalues` `[...,k]`, `eigenvectors` `[...,N,k]`, float64 `residual_norm`
`[...,k]`, int64 `iterations` / `status` and bool `converged`, also `[...,k]`.
For normalized Ritz vectors u, the actual map is applied to verify:

\[
\|Au_i-\lambda_i u_i\|_2\le\max(\mathrm{atol},\mathrm{rtol}|\lambda_i|),\qquad
U^\mathsf TU\approx I
\]

The working width r is `subspace_size`, with `1 <= k <= r <= N`. Its default is
`min(N,2*k+8)`. An optional `initial` block `[...,N,r]` sets r when
`subspace_size` is omitted. It must have full numerical column rank and may
broadcast to operator batches, without adding batch dimensions. Otherwise a
random block is drawn using the current PyTorch RNG state. Seed the RNG or
supply a fixed block for reproducible starts. `max_iter=0` evaluates the initial
subspace, including its projected eigensolve, without any subspace updates.
Empty leading batches work; N must be positive because k is positive.

Each batch continues until all requested eigenpairs converge or it fails or
exhausts its budget. Individual converged vectors are not locked: they can rotate
as the subspace changes, especially around repeated eigenvalues. `iterations`
counts subspace updates and repeats the final batch count across its k sorted
ranks. Compare invariant subspaces rather than vector signs for repeated spectra.
Use a positive `atol` when retaining null directions. Finite rank collapse of
AQ is allowed; QR completion supplies orthogonal nullspace directions without
dividing by eigenvalues.

| Status | Python Name | Meaning |
| --- | --- | --- |
| 0 | `EigenStatus.CONVERGED` | Residual and orthogonality checks pass |
| 1 | `EigenStatus.ITERATION_LIMIT` | Unconverged pair at the iteration limit |
| 2 | `EigenStatus.RANK_DEFICIENT_INITIAL` | Starting block lacks numerical rank |
| 3 | `EigenStatus.NONFINITE` | Nonfinite start, source action or arithmetic |
| 4 | `EigenStatus.NON_PSD` | A significantly negative projected direction was found |
| 5 | `EigenStatus.LOSS_OF_ORTHOGONALITY` | Returned block failed its orthogonality check |

C++ enum members are `Converged`, `IterationLimit`, `RankDeficientInitial`,
`Nonfinite`, `NonPSD`, `LossOfOrthogonality`. Status 2-5 fails the batch and returns
NaN eigenvalues/vectors/residuals; `converged` is false. Iteration exhaustion
retains finite Ritz estimates for inspection. Invalid shapes/options raise.
Eigenvalues are not clipped inside the solver; PCA clips only the tiny negative
roundoff values when turning variance into coordinates.

Symmetry and PSD are caller assumptions. Negative-direction detection is not
an exhaustive PSD test, and there is no smallest/indefinite eigenvalue mode.
The projected matrix removes roundoff asymmetry; the original map is unchanged.
Negative Ritz values within `max(atol,32*eps*r*max(abs(spectrum)))` are tolerated,
where eps is the input dtype precision.
Residual convergence establishes approximate eigenpairs, **not a certificate
that no larger eigenvalue was missed**. A supplied invariant start orthogonal
to leading directions can converge to other eigenpairs. Use a random start with
adequate width; flat spectra may require more iterations or a wider subspace.

Storage includes several `[N,r]` blocks and `[r,r]` projections per batch,
source preparation, tile workspace and outputs. If r approaches N, the subspace
storage itself approaches `[N,N]`. There is no implicit full comparison matrix,
but a caller choosing r=N requests a full basis. Each iteration still evaluates
all source pairs for its products, and checks true residuals with another product.
QR and small projected eigensolves also have a cost. See the
[native kernel PCA example](../examples/kernel_pca.py).

## Rectangular Singular Triplets

`streaming.linalg.svd` estimates leading singular values and left/right vectors
of real rectangular maps using native two-sided products. Python/C++ expose
matching result fields and options. See [truncated SVD](truncated-svd.md) for
residual, initialization, shape and storage contracts. It is inference-only;
it does not form a normal matrix.

## Traces and Matrix Functions

`streaming.linalg.trace` exposes exact blocked actions and Hutchinson estimates.
`logdet` explicitly selects dense Cholesky or matrix-free SPD Lanczos quadrature.
`expm_multiply` applies a symmetric matrix exponential, and `inv_sqrt_multiply`
applies an SPD inverse square root to vectors/blocks. All four have matching C++
entry points. See [spectral consumers](spectral-consumers.md) for shapes, explicit
randomness, inference requirements, diagnostics and basis-memory costs.
