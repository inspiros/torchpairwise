# SPD Covariance Distances

These standalone functions compare symmetric positive-definite (SPD) matrices.
They accept covariance samples with shape `[..., N, D, D]`; a second operand
has shape `[..., M, D, D]`. Leading batch dimensions broadcast. The functions
are separate from the vector metric registry and generic `cdist`/`pdist` APIs.

| Geometry | All pairs | Paired samples | Condensed self distances |
| --- | --- | --- | --- |
| Log-Euclidean | `log_euclidean_distances` | `paired_log_euclidean_distances` | `condensed_log_euclidean_distances` |
| Affine-invariant Riemannian | `affine_invariant_distances` | `paired_affine_invariant_distances` | `condensed_affine_invariant_distances` |
| Rooted Stein | `stein_distances` | `paired_stein_distances` | `condensed_stein_distances` |

All-pairs functions return `[..., N, M]`. Omit the second operand for self
distances. Paired functions require equal sample counts and return `[..., N]`.
Condensed functions accept one operand and return `[..., N*(N-1)//2]` in row-major
upper-triangle order: `(0,1), (0,2), ..., (1,2), ...`. Empty sample and batch
axes are supported; matrix dimension `D` must be positive.

Native standalone evaluation already tiles sample pairs, including condensed
comparisons, to bound pairwise matrix workspaces. Requested all-pairs/condensed
outputs still require their natural storage. For reductions and products that
avoid retaining that output, use the [structured operator](structured-operators.md)
and its configurable `chunk_size`.

```python
import torch
import torchpairwise as tp

covariances = torch.tensor([
    [[2., .3], [.3, 1.]],
    [[1., 0.], [0., 3.]],
], dtype=torch.float64, requires_grad=True)

pairwise = tp.affine_invariant_distances(covariances)
condensed = tp.condensed_stein_distances(covariances)
condensed.sum().backward()
```

## Definitions and Normalization

With the principal symmetric matrix logarithm, Log-Euclidean distance is

\[
d_{\mathrm{LE}}(A,B)=\lVert\log A-\log B\rVert_F\]

Affine-invariant Riemannian distance is

\[
d_{\mathrm{AI}}(A,B)=\lVert\log(A^{-1/2}BA^{-1/2})\rVert_F
=\left(\sum_i\log^2\lambda_i\right)^{1/2},
\]

where \(\lambda_i\) are the generalized eigenvalues of \((B,A)\).
The Stein/Jensen-Bregman LogDet divergence and its rooted distance are

\[
S(A,B)=\log\det\frac{A+B}{2}
-\frac{\log\det A+\log\det B}{2},\qquad
 d_{\mathrm{Stein}}(A,B)=\sqrt{S(A,B)}\]

Every function defaults to `squared=False`. For Log-Euclidean and affine-invariant
geometry, `squared=True` returns the squared distance. For Stein, it returns
\(S(A,B)\) itself: the divergence, not its square, and with no extra factor of
two. All logarithms are natural logarithms.

All three distances are symmetric, invariant under common positive scalar
rescaling, invariant under simultaneous inversion, and invariant under common
orthogonal congruence. Affine-invariant and Stein distances additionally remain
unchanged under any common invertible congruence \(A\mapsto PAP^T\).
Log-Euclidean distance does not have this last property in general.

## Domain, Precision, and Derivatives

Inputs must be dense real tensors on the same CPU or CUDA device, with the same
dtype. Float32 and float64 are supported. Under autocast, float16 and bfloat16
inputs are promoted to float32, computations run with autocast disabled, and
outputs have dtype float32. Input gradients follow PyTorch's usual cast rules.

All entries must be finite and matrices must be symmetric and strictly positive
definite. Symmetry allows relative roundoff of
`64 * D * finfo(computation_dtype).eps` times each matrix's largest absolute entry;
accepted matrices are symmetrized. Zero and negative eigenvalues are rejected.
There is no added diagonal jitter, eigenvalue clipping, or positive eigenvalue
floor. Every supplied covariance is validated even if the other operand is empty.
In eager execution invalid inputs raise `ValueError`.

Very small positive eigenvalues are allowed, but conditioning is limited by the
computation dtype. A mathematically SPD matrix may lose positive definiteness
numerically; whitened spectra must remain finite and positive. Float64 is useful
when covariance condition numbers or scale ratios are large. These routines do
not promise meaningful results when intermediate arithmetic is unrepresentable.

First derivatives support repeated positive eigenvalues, including scalar
multiples of the identity. The matrix logarithm uses its continuous Fréchet
derivative, with the diagonal/repeated-eigenvalue divided difference
\(1/\lambda\). The other two geometries differentiate scalar spectral functions
using generalized eigenvectors; they do not differentiate eigenvectors.
**Higher derivatives are unsupported** and custom backward functions use
`once_differentiable`.

Identical covariances produce exactly zero values. Unsquared distances have a
chosen zero subgradient there. Squared distances and Stein divergence also have
zero gradients there. This policy applies to equal values in separate input
tensors as well as self comparisons.

For nearby covariances, affine-invariant and Stein computations diagonalize
\(L^{-1}(B-A)L^{-T}\), where \(A=LL^T\), and evaluate the resulting logarithms
with `log1p`. Stein divergence is accumulated as
\(\sum_i\log\cosh(\tfrac12\log\lambda_i)\), using a cancellation-resistant
formula near zero and an overflow-resistant formula away from zero. This avoids
subtracting nearly equal log determinants. Log-Euclidean geometry subtracts
computed matrix logarithms, so resolving extremely small differences still
depends on the precision of those individual logarithms.

## Memory and Execution

Per-input decompositions process at most 16 samples per batch at a time.
All-pairs evaluation uses tiles of up to 16 by 16 samples; paired and condensed
evaluation use at most 16 pairs per batch at a time. Condensed evaluation computes
upper-triangle pairs directly without allocating a full square distance map.
Batch axes are not tiled. Matrix work within each tile still scales quadratically
with `D`, and dense decompositions scale cubically with `D`.

Tiling limits individual forward decomposition/residual temporaries. It does
**not** provide constant-memory training: autograd retains state from all tiles.
Pairwise backward state can scale as `batch_size * N * M * D * D`; output storage
and per-input decompositions also remain. Use smaller sample batches or explicit
training-loop chunking when this state is too large.

The implementation consists of PyTorch operations and first-order custom
autograd functions. No claim of compiler/backend coverage follows from source
inspection; use eager execution unless the intended compiled configuration has
been validated.
