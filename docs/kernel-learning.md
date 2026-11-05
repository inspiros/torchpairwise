# Scatter Matrices and Kernel PCA

Streaming matrix products support iterative algorithms. They do not turn
`torch.linalg.eigh` or `svd` into consumers of logical operators. The distinction
between input features, kernel features and dual coefficients determines the
decomposition size.

## Ordinary PCA and LDA

For X `[N,d]`, let class c have n_c samples and mean mu_c; mu is the global mean.
Using column vectors for individual observations:

\[
S_w=\sum_c\sum_{i\in c}(x_i-\mu_c)(x_i-\mu_c)^\mathsf{T},\qquad
S_b=\sum_c n_c(\mu_c-\mu)(\mu_c-\mu)^\mathsf{T}
\]

Both matrices are `[d,d]`. These are unnormalized scatter sums; class-weighting
and covariance-normalization conventions can vary. The total scatter is
`S_t = S_w + S_b`. PCA instead decomposes the covariance:

\[
C_X=\frac{(X-\mathbf1\mu^\mathsf{T})^\mathsf{T}
                (X-\mathbf1\mu^\mathsf{T})}{N-1}
\]

This is also `[d,d]`. When d is small, accumulating these matrices from sample
blocks and doing a dense decomposition is appropriate. No kernel matrix is
needed. LDA solves a generalized eigenproblem involving both scatters, commonly
with regularization when the within-class matrix is singular.

## Exact Kernel Formulations

Let Phi `[N,p]` contain explicit kernel feature rows, so `K = Phi @ Phi.T` is
`[N,N]`. The original d need not equal p; p may be infinite for an RBF kernel.
Define P `[N,N]` by P_ij = 1/n_c when i and j belong to class c, and zero
otherwise. Let `J = 11.T/N` and `H = I-J`. Feature-space scatters are:

\[
S_w^\phi=\Phi^\mathsf{T}(I-P)\Phi,\qquad
S_b^\phi=\Phi^\mathsf{T}(P-J)\Phi
\]

These are `[p,p]` operators. For a discriminant direction `w = Phi.T @ alpha`,
alpha is `[N]`, and the corresponding quadratic forms use the dual matrices:

\[
\widetilde S_w=K(I-P)K,\qquad
\widetilde S_b=K(P-J)K
\]

Both dual matrices are `[N,N]`. The low rank of between-class scatter (at most
number of classes minus one) does not by itself remove the within-class solve.
Regularization must be defined in the chosen primal or dual formulation; they
are not interchangeable by simply adding the same identity matrix.

Kernel PCA similarly uses `K_c = H K H`, `[N,N]`. Its nonzero eigenvalues divided
by `N-1` equal those of the centered feature-space covariance. A small input d
does not make an exact nonlinear kernel decomposition small.

Computing an ordinary covariance of the rows of K instead treats those rows
as explicit N-dimensional features. That produces `K.T @ H @ K / (N-1)`,
also `[N,N]`, rather than the standard kernel PCA eigenproblem. For an already
centered Gram matrix, that covariance squares its eigenvalues before dividing
by `N-1`.

## A Small Projected Decomposition

The [runnable kernel PCA example](../examples/kernel_pca.py) calls native
`tp.linalg.eigh` on a centered lazy kernel. It builds an
orthonormal subspace Q `[N,r]` and performs a small Rayleigh-Ritz decomposition:

\[
K_cV=H\bigl(K(HV)\bigr),\qquad
B=Q^\mathsf{T}K_cQ\in\mathbb R^{r\times r}
\]

The solver checks actual eigenpair residuals and orthogonality, returning
convergence/status per sorted eigenpair. Training coordinates are
`U * sqrt(lambda)` `[N,k]`. Retained state includes several sample blocks of
width r, a small projected matrix, inputs, outputs and product tile workspace.
Every kernel product still evaluates all pairs. Flat spectra can require more
updates or a wider subspace, increasing work/storage. Random starts need overlap
with leading directions; small residuals cannot certify global eigenvalue order
for an arbitrary caller-supplied invariant start.

```sh
python examples/kernel_pca.py --device cuda --components 4 --oversampling 8 \
    --max-iter 100 --rtol 1e-7 --chunk-size 128 --dense-reference
```

The optional dense reference checks the spectrum and reports subspace error;
small linear and zero-rank cases also run. Omit it for large inputs. The example
is unbatched and inference-only, assumes a symmetric PSD kernel, and computes
training and held-out coordinates. It raises on solver nonconvergence. The underlying
[native leading eigensolver](linear-operators.md#leading-eigenpairs) also supports
broadcast batches and Python/C++ deployment.

The example fits training column means `mu = K.reduce("mean", dim=-2)` and
projection weights `U * lambda.rsqrt()`. Held-out C `[T,N]` is centered with
`tp.as_lazy(C).centered(mu)` before applying these `[N,k]` weights:

\[
Z_{\rm new}=\left(C-\mathbf1_T\mu^\mathsf T\right)H_N
                 U_k\Lambda_k^{-1/2}\in\mathbb R^{T\times k}
\]

The example gives zero weight to eigenvalues at or below
`eps * N * max(abs(lambda))`, avoiding division by zero for zero-rank data.
Queries may be transformed independently or in batches with identical results.
Fitted means, weights and training inputs remain allocated; no cross-kernel
matrix is stored. This is an unwhitened projection, consistent with the training
scores. The optional dense check also verifies held-out projections.

Explicit approximate features such as random features or Nyström features give
another route: materialize covariance/scatter matrices `[r,r]` and decompose
those. That changes the representation; it does not make exact kernel scatters
independent of N.

## Kernel LDA with Group Projection

The [kernel LDA example](../examples/kernel_lda.py) implements a specific
coefficient-regularized dual formulation using streamed RBF products, group
projection, native CG, and a small dense eigensolve. No general inverse is formed.
For memberships E `[N,C]`, counts n_c and sample centering H, define:

\[
P=E\operatorname{diag}(n_c^{-1})E^\mathsf T,\qquad
S_w=K(I-P)K+\alpha I,\qquad
F=KHE\operatorname{diag}(n_c^{-1/2}),\qquad S_b=FF^\mathsf T
\]

The scatter maps are logically `[N,N]`; F and the solution Z are `[N,C]`.
Solve against the C columns, then diagonalize a `[C,C]` matrix:

\[
S_wZ=F,\qquad Tu=\theta u,\quad T=F^\mathsf TZ,\qquad
v=Zu/\sqrt\theta,\qquad S_bv=\theta S_wv
\]

Only positive directions are kept, at most C-1. The normalization makes the
returned directions orthonormal in the S_w inner product. Training coordinates
are `K @ directions`, shape `[N,k]`. Out-of-sample projection uses the training
samples as references, `streaming.kernel(X_new,X_train,...).matmul(directions)`.
This example uses alpha times the identity in coefficient space. Adding alpha
times the identity in the implicit RKHS yields a different dual regularizer.

P is applied with `streaming.group_projection(labels,like=X)`, storing group
IDs/counts. `R=K-P @ K` applies `(I-P)K`, and
`R.mT @ R + alpha * tp.eye_like(R.mT @ R)` implements S_w for symmetric K. The small
class subspace and CG workspace are still material allocations; when C approaches
N, this approach loses its memory advantage. The example's optional dense check
verifies eigenvalues, the generalized eigen residual, and S_w orthogonality.

The runnable LDA example also projects independent held-out inputs and checks
them against the dense rectangular kernel, without requiring query labels.
