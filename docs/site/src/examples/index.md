<div class="eyebrow">Example gallery</div>

# Streaming ML Algorithms

Use a comparison matrix through its action on vectors or small blocks. These
examples keep the algorithm's coefficients, projections or scores, while
generating kernel tiles as needed. The diagrams show computation paths, not
measured performance.

<div class="example-grid">
<a class="example-card" href="#kernel-ridge-regression">
<img src="../assets/gallery-ridge.svg" width="560" height="224" alt="Kernel ridge: apply a shifted kernel through CG to obtain an N by R coefficient block." />
<span class="example-card-body"><span class="feature-number">01 / SOLVE</span><strong>Kernel Ridge Regression</strong><span class="badge mode-all-pairs">CG</span><span class="badge mode-condensed">Kernel products</span><p>Fit regularized coefficients by solving a system through streamed products.</p><span class="example-shape">X [N,D], targets [N,R] → predictions [N,R]</span></span>
</a>
<a class="example-card" href="#truncated-kernel-pca">
<img src="../assets/gallery-pca.svg" width="560" height="224" alt="Kernel PCA: apply a centered kernel to a narrow basis and solve an r by r eigenproblem." />
<span class="example-card-body"><span class="feature-number">02 / EMBED</span><strong>Truncated Kernel PCA</strong><span class="badge mode-paired">Leading eigenpairs</span><span class="badge mode-condensed">Centering</span><p>Find a low-dimensional embedding with a narrow subspace and a small eigensolve.</p><span class="example-shape">X [N,D] → coordinates [N,k]</span></span>
</a>
<a class="example-card" href="#kernel-lda">
<img src="../assets/gallery-lda.svg" width="560" height="224" alt="Kernel LDA: group projection and scatter products lead to a C by C class-space eigenproblem." />
<span class="example-card-body"><span class="feature-number">03 / SEPARATE</span><strong>Kernel LDA</strong><span class="badge mode-all-pairs">Group projection</span><span class="badge mode-paired">Class-space solve</span><p>Apply within-class scatter as a map, then diagonalize in the smaller class space.</p><span class="example-shape">X [N,D], labels [N] → coordinates [N,k]</span></span>
</a>
<a class="example-card" href="#binary-kernel-svm">
<img src="../assets/gallery-svm.svg" width="560" height="224" alt="Binary SVM: a label-weighted kernel feeds a box-and-equality QP, yielding N dual coefficients." />
<span class="example-card-body"><span class="feature-number">04 / OPTIMIZE</span><strong>Binary Kernel SVM</strong><span class="badge mode-single">Constrained QP</span><span class="badge mode-condensed">KKT intercept</span><p>Train a binary C-SVM with box bounds and the equality required for a free intercept.</p><span class="example-shape">X [N,D], labels [N] → scores [N]</span></span>
</a>
<a class="example-card" href="#landmark-kernel-regression">
<img src="../assets/gallery-lsqr.svg" width="560" height="224" alt="Landmark regression: a rectangular training kernel feeds LSQR, then a held-out kernel applies the landmark coefficients." />
<span class="example-card-body"><span class="feature-number">05 / LEAST SQUARES</span><strong>Landmark Kernel Regression</strong><span class="badge mode-paired">Rectangular LSQR</span><span class="badge mode-all-pairs">Held-out prediction</span><p>Fit a kernel dictionary without storing its design matrix or normal matrix. Apply the coefficients to new samples.</p><span class="example-shape">X [N,D], landmarks [L,D] → coefficients [L,R]<br />X_test [T,D] → predictions [T,R]</span></span>
</a>
<a class="example-card" href="#personalized-pagerank">
<img src="../assets/gallery-pagerank.svg" width="560" height="224" alt="PageRank: row-normalized directed affinity tiles feed a nonsymmetric GMRES solve, producing personalized vertex ranks." />
<span class="example-card-body"><span class="feature-number">06 / RANK</span><strong>Personalized PageRank</strong><span class="badge mode-single">Restarted GMRES</span><span class="badge mode-all-pairs">Directed graph</span><p>Rank vertices of a generated directed affinity graph through a nonsymmetric linear system.</p><span class="example-shape">Query/key views [N,D], personalization [N,R]<br />→ ranks [N,R]</span></span>
</a>
<a class="example-card" href="#kernel-fitting-with-an-intercept">
<img src="../assets/gallery-kkt.svg" width="560" height="224" alt="Kernel fitting: a streamed kernel and small constraint blocks form an indefinite KKT map, solved by short-recurrence MINRES for coefficients and an intercept." />
<span class="example-card-body"><span class="feature-number">07 / CONSTRAIN</span><strong>Kernel Fitting with an Intercept</strong><span class="badge mode-condensed">Symmetric indefinite MINRES</span><span class="badge mode-all-pairs">Block KKT map</span><p>Fit kernel coefficients and an unpenalized intercept through a compact block system, then predict held-out samples.</p><span class="example-shape">X [N,D], targets [N,R]<br />→ coefficients [N,R], intercept [1,R]</span></span>
</a>
<a class="example-card" href="#rectangular-low-rank-approximation">
<img src="../assets/gallery-svd.svg" width="560" height="224" alt="Rectangular SVD: streamed forward and transpose products fit narrow left and right factors, which reuse new matrix products without evaluating pairs." />
<span class="example-card-body"><span class="feature-number">08 / APPROXIMATE</span><strong>Rectangular Truncated SVD</strong><span class="badge mode-paired">Two-sided iteration</span><span class="badge mode-condensed">Reusable factors</span><p>Fit a low-rank rectangular kernel map, then apply its approximation through narrow factors.</p><span class="example-shape">A [N,M] → U [N,k], singular values [k], V [M,k]</span></span>
</a>
</div>

Use `tp.linalg.solve(..., method=...)` for square systems and
`tp.linalg.lstsq(...)` for least squares; the named solver APIs remain
available. These entries preserve native convergence diagnostics and do not
form an inverse or pseudoinverse. See [choosing a solver](../guide/algebra.md#choosing-a-solver).

## Storage and Computation

Here N is the training sample count, D the input feature count, R the number of
target columns, C the number of classes, k the requested embedding dimension,
and r the PCA search-subspace width. Landmark regression uses L landmarks and
T held-out samples.

| Example | Dense formulation | Main retained algorithm blocks |
| --- | --- | --- |
| Ridge | `[N,N]` shifted Gram matrix and a direct solve | Coefficients, RHS and CG vectors `[N,R]` |
| PCA | `[N,N]` centered Gram matrix and a full eigendecomposition | Basis/products `[N,r]`, projection `[r,r]`, coordinates `[N,k]` |
| LDA | `[N,N]` Gram and scatter matrices | Class blocks `[N,C]`, class eigenproblem `[C,C]`, coordinates `[N,k]` |
| SVM | `[N,N]` label-weighted Gram matrix | Dual coefficients, gradients and directions `[N]` |
| Landmark regression | `[N,L]` kernel design and an augmented least-square factorization | LSQR blocks `[N,R]` and `[L,R]`, predictions `[T,R]` |
| PageRank | `[N,N]` transition matrix and a nonsymmetric solve | Row mass `[N]`, ranks `[N,R]`, restart basis `[N,R,m]` and QR blocks `[R,m,m]` |
| Ridge with intercept | `[N+1,N+1]` symmetric indefinite KKT matrix | MINRES blocks `[N+1,R]`, constraint `[1,N]`, intercept `[1,R]` |

These are block shapes, not total peak-memory estimates. Inputs, kernel
preparation, tile workspace, diagnostics and temporary solver blocks also use
memory. A streamed product still evaluates all pairs, and iterative methods may
repeat that work many times. Streaming trades stored matrices for recomputation;
it does not guarantee a faster solve. PCA with r near N, or LDA with C near N,
loses much of the storage advantage.
For GMRES, m is the effective restart width; a width near N can also approach
quadratic basis storage. Its basis and QR workspace use float64.

<div class="example-scope">These examples use real float32/float64 inputs.
Most run without autograd; learning examples explicitly enable first gradients.
Check convergence diagnostics before using results. Native solvers are available in Python and C++; the runnable teaching
scripts are Python. They illustrate algorithms rather than fitted-estimator APIs.</div>

### Measure Complete Fits

Use the [algorithm benchmark harness](https://github.com/inspiros/torchpairwise/blob/master/examples/benchmark_ml.py)
for end-to-end ridge, landmark, PCA, LDA and SVD comparisons. It reports time,
extra peak allocated CUDA bytes, numerical errors and convergence diagnostics.
The default dense baseline uses the same iterative solver and settings as
streaming; `--dense-direct` adds a separately labeled direct dense solve or
full decomposition. Kernel construction and held-out prediction are included;
SVD measures factor fitting. Ridge/landmark also accept `--backward`.

```sh
python examples/benchmark_ml.py --case pca --device cuda --n 2048 --d 8 \
    --rank 2 --subspace-size 10 --chunk-size 128 --output develop/pca-memory.json
```

The [measured comparison table](../guide/benchmarks.md#measured-kernel-fit-memory)
reports complete fits and backward workloads, plus a separate direct-dense
ridge comparison.

CPU tensor peaks are unmeasured. GPU figures exclude reserved allocator cache
and driver/external-library allocations. Streaming may take more time to save
memory; see [measurement scope](../guide/benchmarks.md#complete-kernel-learning-workloads).

## Kernel Ridge Regression

For a symmetric PSD kernel K and positive regularization \(\alpha\), solve for
coefficients W, then apply the kernel to predict:

\[
(K+\alpha I)W=Y,\qquad \widehat Y=KW
\]

The example uses no fitted intercept. The dense equivalent is
`torch.linalg.solve(K + alpha * I, Y)`, followed by `K @ W`.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
import torch
import torchpairwise as tp

# X:[N,D], targets:[N,R], matching float32/float64 dtype and device
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=0.3, chunk_size=64)
    A = K + 0.2 * tp.eye_like(K)
    fit = tp.linalg.solve(A, targets, method="cg", rtol=1e-6, max_iter=300)
    if not fit.converged.all():
        raise RuntimeError(f"CG status: {fit.status.tolist()}")
    predictions = K.matmul(fit.solution)  # [N,R]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
#include <torchpairwise.h>
namespace s = torchpairwise::streaming;

// X:[N,D], targets:[N,R], matching float32/float64 dtype and device
at::NoGradGuard no_grad;
s::PairwiseOptions options;
options.comparison.gamma = 0.3;
options.chunk_size = 64;
auto K = s::kernel(X, c10::nullopt, "rbf", options);
auto A = s::aslinearoperator(K).add_diagonal(0.2);
s::linalg::SolveOptions solve;
solve.method = "cg";
solve.rtol = 1e-6;
solve.max_iter = 300;
auto fit = s::linalg::solve(A, targets, solve);
TORCH_CHECK(fit.converged.all().item<bool>(), "CG did not converge");
auto predictions = K.matmul(fit.solution);  // [N,R]
```

</div>
</div>

[Full ridge example](https://github.com/inspiros/torchpairwise/blob/master/examples/kernel_ridge.py)
prints coefficient shape, iterations, residuals and training RMSE.
Run from the repository root:

```sh
python examples/kernel_ridge.py --device cuda --dense-check
```

For unevenly scaled systems, the [preconditioned CG example](https://github.com/inspiros/torchpairwise/blob/master/examples/preconditioned_cg.py)
compares iterations, solve time and incremental CUDA memory with Jacobi-PCG.
It uses the same streamed matrix in both paths and a separate optional dense
oracle. Jacobi does not improve a constant-diagonal RBF Gram matrix by itself.
The [Nyström-preconditioned ridge example](https://github.com/inspiros/torchpairwise/blob/master/examples/nystrom_cg.py)
stores landmark kernel blocks to prepare a reusable inverse action, then solves
the full original kernel system. It compares CG, Jacobi and Nyström, reports
setup separately from solves, and checks coefficients and held-out predictions
against a small dense oracle. See [low-rank preconditioning](../guide/algebra.md#low-rank-and-nyström-preconditioning).

### Learning Through a Solve

The [differentiable ridge example](https://github.com/inspiros/torchpairwise/blob/master/examples/differentiable_kernel_ridge.py)
learns an RBF feature scale from a held-out regression loss. It uses
preconditioned `solve(..., differentiable=True)` followed by a rectangular prediction product;
the optional dense check compares both loss and scale gradient. Backward solves
an adjoint system and replays streamed products, retaining no iteration history.
This mode requires nonsingular systems and convergence of both solves.
[Python and C++ examples](../guide/algebra.md#differentiable-square-solves)
show its options and restrictions. `--landmarks 8` switches to a damped
rectangular fit using `lstsq(..., differentiable=True)`; the same script checks
the resulting loss and scale gradient against augmented dense QR.

```sh
python examples/differentiable_kernel_ridge.py --device cuda --dense-check
python examples/differentiable_kernel_ridge.py --device cuda --landmarks 8 --dense-check
```

## Truncated Kernel PCA

Center the Gram operator, estimate its leading k eigenpairs using a search
subspace of width r, and form unwhitened training coordinates:

\[
H=I-\mathbf1\mathbf1^\mathsf T/N,\qquad
K_c=HKH,\qquad K_cU_k\approx U_k\Lambda_k,\qquad
Z=U_k\Lambda_k^{1/2}
\]

The dense equivalent centers K explicitly and calls `torch.linalg.eigh(Kc)`,
retaining the largest k eigenpairs. Gram eigenvalues divided by N-1 give the
corresponding sample-covariance eigenvalues. The example also projects held-out
samples using the training kernel means and fitted directions.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# Reuse the imports above. X:[N,D], N>=12
with torch.no_grad():
    A = tp.lazy(X, metric="rbf", gamma=0.2).centered()
    fit = tp.linalg.eigh(A, k=4, subspace_size=12, max_iter=100,
                       rtol=1e-6, atol=1e-10)
    if not fit.converged.all():
        raise RuntimeError(f"Eigen status: {fit.status.tolist()}")
    coordinates = fit.eigenvectors * fit.eigenvalues.clamp_min(0).sqrt()
    # coordinates:[N,4]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// Reuse the include/namespace above. X:[N,D], N>=12
at::NoGradGuard no_grad;
s::PairwiseOptions options;
options.comparison.gamma = 0.2;
auto A = s::aslinearoperator(s::kernel(X, c10::nullopt, "rbf", options)).centered();
s::linalg::EighOptions solve;
solve.k = 4;
solve.subspace_size = 12;
solve.max_iter = 100;
solve.rtol = 1e-6;
solve.atol = 1e-10;
auto fit = s::linalg::eigh(A, solve);
TORCH_CHECK(fit.converged.all().item<bool>(), "Eigensolve did not converge");
auto coordinates = fit.eigenvectors * fit.eigenvalues.clamp_min(0).sqrt(); // [N,4]
```

</div>
</div>

[Full PCA example](https://github.com/inspiros/torchpairwise/blob/master/examples/kernel_pca.py)
prints coordinate shape, leading eigenvalues, iterations and eigen-residuals.
The dense check compares spectra/subspaces rather than eigenvector signs.

```sh
python examples/kernel_pca.py --device cuda --dense-reference
```

### PCA Projection for New Samples

Fit the training column means once. With C `[T,N]` comparing held-out inputs
against the training inputs, the unwhitened transform is:

\[
\mu_j=\frac1N\sum_i K_{ij},\qquad
C_c=(C-\mathbf1_T\mu^\mathsf T)H_N,\qquad
Z_{\rm new}=C_c U_k\Lambda_k^{-1/2}\in\mathbb R^{T\times k}
\]

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# Inside the no-grad scope above; X_new:[T,D]. Reuse the fitted eigensolve.
K = tp.lazy(X, metric="rbf", gamma=0.2)
reference_mean = K.reduce("mean", dim=-2)       # [N], fit once
cutoff = torch.finfo(X.dtype).eps * X.shape[0] * fit.eigenvalues.abs().max()
positive = fit.eigenvalues > cutoff
safe = torch.where(positive, fit.eigenvalues, torch.ones_like(fit.eigenvalues))
inverse_root = torch.where(positive, safe.rsqrt(), torch.zeros_like(safe))
weights = fit.eigenvectors * inverse_root     # [N,k]
C = tp.lazy(X_new, X, metric="rbf", gamma=0.2)
held_out = C.centered(reference_mean).matmul(weights)  # [T,k]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// Inside the no-grad scope above; X_new:[T,D], same training references/options.
auto K = s::kernel(X, c10::nullopt, "rbf", options);
auto reference_mean = K.reduce("mean", -2);   // [N], fit once
auto cutoff = std::numeric_limits<double>::epsilon() * X.size(0) *
              fit.eigenvalues.abs().max();  // X is float64 in this example
auto positive = fit.eigenvalues > cutoff;
auto safe = at::where(positive, fit.eigenvalues, at::ones_like(fit.eigenvalues));
auto inverse_root = at::where(positive, safe.rsqrt(), at::zeros_like(safe));
auto weights = fit.eigenvectors * inverse_root;  // [N,k]
auto C = s::aslinearoperator(s::kernel(X_new, X, "rbf", options));
auto held_out = C.centered(reference_mean).matmul(weights);  // [T,k]
```

</div>
</div>

Include `<limits>` for the C++ cutoff. Null/tiny eigenvalues receive zero
projection weight. Keep training samples, kernel settings, means and weights
for reuse. Coordinates are independent of which other queries accompany a
sample. Only kernel tiles and projection blocks are evaluated; fitting still
needs the training samples and narrow solver state.

## Kernel LDA

Let E `[N,C]` be the class-indicator matrix, \(n_c\) the class counts, and
\(P=E\operatorname{diag}(n_c^{-1})E^\mathsf T\) the group-mean projector.
The example uses coefficient-space regularization \(\alpha I\):

\[
S_w=K(I-P)K+\alpha I,\qquad
F=KH E\operatorname{diag}(n_c^{-1/2}),\qquad S_b=FF^\mathsf T
\]

Solve \(S_w Z=F\) with CG and diagonalize
\(F^\mathsf T Z\), which is `[C,C]`. For its positive eigenpairs
\((\theta,u)\), a discriminant direction is \(Zu/\sqrt\theta\).
There are at most C-1 nonzero directions. The dense equivalent forms the
`[N,N]` scatter matrices and solves the same generalized eigenproblem.
This regularizer differs from adding \(\alpha I\) in implicit feature space.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], labels:[N] int64. Core scatter map; full algorithm linked below.
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=0.3)
    P = tp.group_projection(labels, like=X)
    R = K - P @ K                  # (I-P)K
    Sw = R.mT @ R
    Sw = Sw + 0.2 * tp.eye_like(Sw)  # logical [N,N]
    # For class block F:[N,C], solve Sw Z = F via tp.linalg.cg(Sw, F).
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X:[N,D], labels:[N] int64. Core scatter map.
at::NoGradGuard no_grad;
s::PairwiseOptions options;
options.comparison.gamma = 0.3;
auto K = s::aslinearoperator(s::kernel(X, c10::nullopt, "rbf", options));
auto P = s::group_projection(labels, X);
auto R = K - P.compose(K);
auto Sw = R.transpose().compose(R).add_diagonal(0.2);
// For F:[N,C], solve Sw Z = F via s::linalg::cg(Sw, F, solve_options).
```

</div>
</div>

[Full LDA example](https://github.com/inspiros/torchpairwise/blob/master/examples/kernel_lda.py)
constructs F, solves the class block, extracts directions and prints coordinate
shapes, eigenvalues and CG diagnostics. The dense check verifies the generalized
eigenproblem and scatter orthogonality.

```sh
python examples/kernel_lda.py --device cuda --dense-check
```

## Binary Kernel SVM

For labels \(y_i\in\{-1,+1\}\) and penalty C, the dual problem is:

\[
\min_\alpha\ \tfrac12\alpha^\mathsf TQ\alpha-\mathbf1^\mathsf T\alpha,
\qquad Q=\operatorname{diag}(y)K\operatorname{diag}(y),\qquad
0\le\alpha_i\le C,\quad y^\mathsf T\alpha=0
\]

The equality permits a free intercept. Training scores are
\(K(\alpha\odot y)+b\). The example obtains b from free support vectors or,
when none are free, the KKT intercept interval. Its dense reference uses the
same QP with a materialized Q, comparing objective and predictions; the example
also checks the primal-dual gap.

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
s::PairwiseOptions options;
options.comparison.gamma = 0.5;
options.chunk_size = 128;
auto K = s::kernel(X, c10::nullopt, "rbf", options);
auto Q = s::aslinearoperator(K.weight_rows(labels).weight_columns(labels));
s::optimize::BoxQPOptions solve;
solve.lower = at::scalar_tensor(0., X.options());
solve.upper = at::scalar_tensor(1., X.options());
solve.equality = s::optimize::QPEquality{labels, at::scalar_tensor(0., X.options())};
solve.rtol = 1e-6;
solve.max_iter = 1500;
auto fit = s::optimize::box_qp(Q, -at::ones_like(labels), solve);
TORCH_CHECK(fit.converged.all().item<bool>(), "QP did not converge");
auto raw_scores = K.matmul(fit.solution * labels);  // [N], before intercept
```

</div>
</div>

[Full SVM example](https://github.com/inspiros/torchpairwise/blob/master/examples/kernel_svm.py)
includes the intercept, training scores, constraint
residuals and primal-dual gap. A second run with a small penalty exercises the
case with no free support vectors on this seeded dataset.

```sh
python examples/kernel_svm.py --device cuda --dense-reference
python examples/kernel_svm.py --device cuda --C 0.01 --dense-reference
```

## Landmark Kernel Regression

A kernel dictionary gives each input L explicit features, one per landmark.
Unlike the square Gram systems above, its design matrix A is rectangular:

\[
\begin{aligned}
A&=K(X,Z)\in\mathbb R^{N\times L}\\
W_*&\in\arg\min_W\ \|AW-Y\|_F^2+\delta^2\|W\|_F^2\\
\widehat Y_{\mathrm{test}}&=K(X_{\mathrm{test}},Z)W_*
\end{aligned}
\]

LSQR uses A and its transpose without forming A.T A. A nonzero data residual
is expected when the targets are not exactly representable by the dictionary.
Convergence checks the true normal residual
\(A^\mathsf T(Y-AW)-\delta^2 W\). This example uses Euclidean coefficient
regularization and no fitted intercept; it is distinct from exact kernel ridge
and Nyström-whitened features.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], landmarks:[L,D], targets:[N,R], X_test:[T,D]
with torch.no_grad():
    A = tp.lazy(X, landmarks, metric="rbf", gamma=0.3, chunk_size=64)
    fit = tp.linalg.lstsq(A, targets, damp=0.2, rtol=1e-6, max_iter=300)
    if not fit.converged.all():
        raise RuntimeError(f"LSQR status: {fit.status.tolist()}")
    predictions = tp.lazy(X_test, landmarks, metric="rbf", gamma=0.3).matmul(fit.solution)
    # fit.solution:[L,R], predictions:[T,R]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X:[N,D], landmarks:[L,D], targets:[N,R], X_test:[T,D]
at::NoGradGuard no_grad;
s::PairwiseOptions options;
options.comparison.gamma = 0.3;
options.chunk_size = 64;
auto A = s::aslinearoperator(s::kernel(X, landmarks, "rbf", options));
s::linalg::LstsqOptions solve;
solve.damp = 0.2;
solve.rtol = 1e-6;
solve.max_iter = 300;
auto fit = s::linalg::lstsq(A, targets, solve);
TORCH_CHECK(fit.converged.all().item<bool>(), "LSQR did not converge");
auto predictions = s::kernel(X_test, landmarks, "rbf", options).matmul(fit.solution);
// fit.solution:[L,R], predictions:[T,R]
```

</div>
</div>

[Full landmark regression example](https://github.com/inspiros/torchpairwise/blob/master/examples/landmark_regression.py)
prints held-out RMSE, coefficient/output shapes and data/normal residuals. The
dense reference solves `[A; damp I] W ≈ [Y; 0]` with `torch.linalg.lstsq`, then
compares coefficients, objective and held-out predictions. LSQR also accepts
wide, rank-deficient and nonsymmetric maps. See the
[complete least-squares contract](https://github.com/inspiros/torchpairwise/blob/master/docs/least-squares.md)
for minimum-norm, warm-start and stopping conventions.

```sh
python examples/landmark_regression.py --device cuda --dense-reference
```

## Personalized PageRank

Give each vertex a query and a key feature vector. Their RBF affinity need not
be symmetric. Add a small uniform weight to guarantee positive outgoing mass,
then normalize each row:

\[
\begin{aligned}
W_{ij}&=K(q_i,k_j)+\varepsilon/N\\
P_{ij}&=W_{ij}/\sum_jW_{ij}\\
(I-\beta P^\mathsf T)R&=(1-\beta)V
\end{aligned}
\]

Each column of V is a nonnegative personalization vector summing to one. With
\(0\le\beta<1\), the resulting ranks give the stationary distribution of the
teleporting walk. Restarted GMRES uses actions of \(I-\beta P^\mathsf T\),
without constructing the identity or the transition matrix. This is a generated
dense graph with self edges, not a sparse web-graph implementation. Row
normalization of a symmetric affinity would instead give a reversible walk;
the separate query/key features make this example directed.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X, keys:[N,D], two views of the same N vertices; N>0
with torch.no_grad():
    W = tp.lazy(X, keys, metric="rbf", gamma=0.3) + X.new_tensor(0.01 / X.shape[0])
    P = W.weight_rows(W.sum(dim=-1).reciprocal())
    A = tp.eye_like(P) - 0.85 * P.mT
    rhs = X.new_full((X.shape[0], 1), 0.15 / X.shape[0])
    fit = tp.linalg.gmres(A, rhs, restart=12, rtol=1e-8, max_iter=200)
    if not fit.converged.all():
        raise RuntimeError(f"GMRES status: {fit.status.tolist()}")
    ranks = fit.solution  # [N,1]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X, keys:[N,D], N>0
at::NoGradGuard no_grad;
s::PairwiseOptions options;
options.comparison.gamma = 0.3;
auto W = s::kernel(X, keys, "rbf", options) +
    s::ScalarExpression(at::scalar_tensor(0.01/X.size(0), X.options()));
auto P = W.weight_rows(W.reduce("sum", -1).reciprocal());
auto A = s::aslinearoperator(P).transpose().scale(-0.85).add_diagonal(1.);
auto rhs = at::full({X.size(0), 1}, 0.15/X.size(0), X.options());
s::linalg::GMRESOptions solve;
solve.restart = 12;
solve.rtol = 1e-8;
solve.max_iter = 200;
auto fit = s::linalg::gmres(A, rhs, solve);
TORCH_CHECK(fit.converged.all().item<bool>(), "GMRES did not converge");
auto ranks = fit.solution;  // [N,1]
```

</div>
</div>

The dense equivalent is `torch.linalg.solve(I-beta*P.T, (1-beta)*V)`.
The [full PageRank example](https://github.com/inspiros/torchpairwise/blob/master/examples/pagerank.py)
uses two personalization columns and prints iteration counts, residuals and
rank-vector mass. The dense check also verifies stationarity, nonnegativity and
mass; it does not clip or renormalize solver output. See
[GMRES contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/gmres.md)
for restart storage, stopping and breakdown conventions.

```sh
python examples/pagerank.py --device cuda --restart 12 --dense-reference
```

## Kernel Fitting with an Intercept

The first ridge example fixes the intercept at zero. To learn an unpenalized
intercept b `[1,R]`, solve for coefficients W `[N,R]` under
\(\mathbf1^\mathsf T W=0\):

\[
\begin{bmatrix}K+\alpha I&\mathbf1\\\mathbf1^\mathsf T&0\end{bmatrix}
\begin{bmatrix}W\\b\end{bmatrix}
=\begin{bmatrix}Y\\0\end{bmatrix}
\]

For a symmetric PSD kernel and positive alpha, the KKT map is symmetric
indefinite. `tp.block` combines the streamed kernel with a stored ones
vector and a `[1,1]` zero corner. MINRES acts on this map without storing the
full system. Held-out predictions are \(K(X_{\mathrm{test}},X)W+\mathbf1 b\).

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], targets:[N,R], N>0
with torch.no_grad():
    A = tp.lazy(X, metric="rbf", gamma=0.3)
    A = A + 0.3 * tp.eye_like(A)
    C = tp.as_lazy(X.new_ones(1, X.shape[0]))
    KKT = tp.block([[A, C.T], [C, X.new_zeros(1, 1)]])
    rhs = torch.cat((targets, targets.new_zeros(1, targets.shape[-1])), -2)
    fit = tp.linalg.minres(KKT, rhs, rtol=1e-8, max_iter=300)
    if not fit.converged.all():
        raise RuntimeError(f"MINRES status: {fit.status.tolist()}")
    weights, intercept = fit.solution[:-1], fit.solution[-1:]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X:[N,D], targets:[N,R], N>0
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

</div>
</div>

The [full intercept example](https://github.com/inspiros/torchpairwise/blob/master/examples/kernel_ridge_intercept.py)
prints coefficients, intercepts, Lanczos steps, KKT residuals and held-out RMSE.
The dense reference checks both a direct KKT solve and an independent Schur
complement using SPD solves, plus the zero-sum coefficient constraint and
held-out predictions. The Schur formulation is another valid approach for this
small constraint system; the example demonstrates direct indefinite block solves.
See [block maps](../guide/algebra.md#block-maps) and
[MINRES contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/minres.md).

```sh
python examples/kernel_ridge_intercept.py --device cuda --dense-reference
```

## Rectangular Low-Rank Approximation

The [rectangular SVD example](https://github.com/inspiros/torchpairwise/blob/master/examples/rectangular_svd.py)
decomposes an RBF cross-kernel `[N,M]` through `tp.linalg.svd`, retaining
left `[N,k]` and right `[M,k]` factors. It then applies an approximate product
with new RHS `[M,R]` using only those factors. The dense check compares singular
values, reconstruction and that product against `torch.linalg.svd`. Fitting
still requires repeated pairwise passes; subsequent approximate products reuse
the factors without pair evaluations.

\[
AV_k\approx U_k\Sigma_k,\qquad
A_k B=U_k\bigl(\Sigma_k(V_k^\mathsf T B)\bigr)\in\mathbb R^{N\times R}
\]

```sh
python examples/rectangular_svd.py --device cuda --dense-check
```

Use the [Python/C++ SVD guide](../guide/algebra.md#rectangular-truncated-svd)
for result fields, shapes, numerical diagnostics and memory costs. The method
estimates leading triplets; flat spectra can require more width or iterations.

## More Streamed Consumers

The [regression and attention examples](https://github.com/inspiros/torchpairwise/blob/master/examples/streaming_ml.py)
show three smaller patterns, with dense value/first-gradient checks:

| Algorithm | Operator consumer | Retained result |
| --- | --- | --- |
| k-NN regression | Distance `topk(k, largest=False)` and target averaging | Neighbors `[N,k]`, predictions `[N,R]` |
| Nadaraya-Watson regression | One kernel product carrying targets and a normalizer | Weighted targets and mass `[N,R+1]` |
| Scaled dot-product attention | `softmax_matmul(values)` on scaled dot products | Attended values `[N,R]` |

```sh
python examples/streaming_ml.py --device cuda
```

Use `--device cpu` in any command for CPU execution. Dense-reference flags are
intended for small inputs: they deliberately allocate the full comparison matrix.
The examples print their own results; numeric outputs depend on device, dtype
and convergence. See [algebra and solver contracts](../guide/algebra.md) for
convergence and deployment limits, and [benchmarking](../guide/benchmarks.md)
to measure time and memory for a workload.

### Masks, Transport and Predictive Variance

The [pair mask example](https://github.com/inspiros/torchpairwise/blob/master/examples/pair_masks.py)
applies masked consumers to supervised contrastive loss, hard negative mining, a
Borůvka nearest-component step and causal kernel attention, each checked against
a dense reference. Python and C++ snippets for these consumers are in
[pair masks](../guide/statistics.md#pair-masks). Two further patterns are
documented with code rather than a script: Sinkhorn transport over any lazy cost
in [entropic transport](../guide/algebra.md#entropic-transport), and LOVE
predictive variances with landmark selection in
[Gaussian process prediction](../guide/spectral.md#gaussian-process-prediction).

```sh
python examples/pair_masks.py --device cuda
```

## Kernel Log-Determinants and Graph Diffusion

A kernel uncertainty or model-selection objective can use
\(\log\det(K+\alpha I)\). A symmetric affinity graph defines the Laplacian
\(L=\operatorname{diag}(K\mathbf1)-K\); diffusion acts on signals V directly:

\[
Y(t)=e^{-tL}V
\]

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], signals:[N,R], matching float32/64 dtype/device
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=.2, chunk_size=64)
    A = K + .5 * tp.eye_like(K)
    uncertainty = tp.linalg.logdet(A, method="slq", num_probes=32,
                                  seed=19, max_iter=32, block_size=4)
    L = tp.diag_embed(K.reduce("sum", dim=-1)) - K
    diffused = tp.linalg.expm_multiply(L, signals, scale=-.1, max_iter=32)
    whitened = tp.linalg.inv_sqrt_multiply(A, signals, max_iter=32)
    Y = diffused.action  # [N,R]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
at::NoGradGuard no_grad;
s::PairwiseOptions source;
source.comparison.gamma = .2;
source.chunk_size = 64;
auto K = s::kernel(X, c10::nullopt, "rbf", source);
auto A = s::aslinearoperator(K).add_diagonal(.5);
s::linalg::LogdetOptions estimate;
estimate.method = "slq";
estimate.num_probes = 32;
estimate.seed = 19;
estimate.max_iter = 32;
estimate.block_size = 4;
auto uncertainty = s::linalg::logdet(A, estimate);
auto L = s::diagonal(K.reduce("sum", -1)) - s::aslinearoperator(K);
s::linalg::MatrixFunctionOptions action;
action.max_iter = 32;
auto diffused = s::linalg::expm_multiply(L, signals, -.1, action);
auto whitened = s::linalg::inv_sqrt_multiply(A, signals, action);
```

</div>
</div>

The [spectral example](https://github.com/inspiros/torchpairwise/blob/master/examples/spectral_actions.py)
prints runtime, extra peak CUDA tensor memory, log-determinant sampling error
and per-column action diagnostics. `--dense-reference` separately measures
small dense Cholesky/logdet, matrix-exponential and eigendecomposition references,
including kernel construction. It checks constant-signal preservation under
diffusion. Finite Krylov/probe counts approximate results; increase the budgets
and inspect dense errors where feasible.

```sh
python examples/spectral_actions.py --device cuda --samples 128 --dense-reference
```

Only Krylov bases, scalar samples, signal blocks and source tiles are retained;
the generated N-by-N kernel, Laplacian and matrix functions are not stored.
Large Krylov or block sizes can still require substantial basis memory.
See [spectral methods and diagnostics](../guide/spectral.md).

## Learning Kernel Bandwidth and Noise

The [likelihood example](https://github.com/inspiros/torchpairwise/blob/master/examples/kernel_likelihood.py)
learns two scalar parameters through a native implicit CG solve and a
log-determinant estimate. For A = K + noise I and one target column y,

\[
\mathcal L=\tfrac12y^\mathsf TA^{-1}y+\tfrac12\log\det A+
\tfrac N2\log(2\pi)
\]

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], y:[N,1]; log_length/log_noise are trainable scalar tensors.
features = X / log_length.exp()
K = tp.lazy(features, metric="rbf", gamma=.5)
A = K + log_noise.exp() * tp.eye_like(K)
fit = tp.linalg.solve(A, y, method="cg", differentiable=True,
                     rtol=1e-8, max_iter=500)
determinant = tp.linalg.logdet(A, method="slq", num_probes=32,
    seed=17, max_iter=24, differentiable=True, gradient="solve",
    solve_rtol=1e-8, solve_max_iter=500)
loss = .5 * (y * fit.solution).sum() + .5 * determinant.estimate
loss.backward()
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
auto features = X / log_length.exp();
s::PairwiseOptions source;
source.comparison.gamma = .5;
auto A = s::aslinearoperator(s::kernel(features, c10::nullopt, "rbf", source))
             .add_diagonal(log_noise.exp());
s::linalg::SolveOptions solve;
solve.method = "cg";
solve.differentiable = true;
solve.rtol = 1e-8;
solve.max_iter = 500;
auto fit = s::linalg::solve(A, y, solve);
s::linalg::LogdetOptions options;
options.method = "slq";
options.num_probes = 32;
options.seed = 17;
options.max_iter = 24;
options.differentiable = true;
options.gradient = "solve";
options.solve_rtol = 1e-8;
options.solve_max_iter = 500;
auto determinant = s::linalg::logdet(A, options);
auto loss = .5 * (y * fit.solution).sum() + .5 * determinant.estimate;
loss.backward();
```

</div>
</div>

SLQ values approximate the objective, while `gradient="solve"` estimates the
underlying log-determinant derivative using fixed probes and converged CG
solutions. It does not differentiate finite quadrature steps. Sampling error
omits quadrature bias; inspect both budgets separately. The executable example
reuses explicit probes across optimizer steps and optionally checks gradients
against the same dense trace/solve estimator. `--exact-logdet` instead selects
dense Cholesky and checks the ordinary exact log-determinant gradient.

```sh
python examples/kernel_likelihood.py --device cuda --dense-reference
```

For a linear kernel, the existing map composition also exposes an explicit
factorized action, useful when the feature width is small:

\[
(XY^\mathsf T)V=X(Y^\mathsf TV)
\]

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], Y:[M,D], V:[M,R]
A = tp.as_lazy(X) @ tp.as_lazy(Y).mT
result = A.matmul(V)  # [N,R]; intermediate [D,R]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
auto A = s::aslinearoperator(X).compose(s::aslinearoperator(Y).transpose());
auto result = A.matmul(V);  // [N,R]; intermediate [D,R]
```

</div>
</div>

This explicitly selects evaluation order; ordinary floating-point reassociation,
overflow and nonfinite propagation can differ from a stored or tiled Gram product.
