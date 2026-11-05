# Truncated Singular Value Decomposition

`streaming.linalg.svd(A, k=...)` estimates the leading singular triplets of a
real rectangular map A `[...,N,M]`. It accepts dense tensors, scalar streaming
sources and LinearOperators. The shared native C++ API is
`streaming::linalg::svd(A, SVDOptions)`.

\[
AV_k\approx U_k\Sigma_k,\qquad A^\mathsf T U_k\approx V_k\Sigma_k,\qquad
A_k=U_k\Sigma_k V_k^\mathsf T
\]

`SVDResult` contains descending `singular_values` `[...,k]`, `left_vectors`
`[...,N,k]`, `right_vectors` `[...,M,k]`, and `residual_norm`, `iterations`,
`converged`, `status` `[...,k]`. Right vectors are columns V, rather than Vh.
Norms are float64, counters/status int64 and convergence bool. Vector/value
outputs preserve source float32/64 dtype and device. Leading source batches
broadcast normally.

## Usage

```python
import torchpairwise as tp

# X:[N,D], Y:[M,D]; float32/64, inference inputs
A = tp.lazy(X, Y, metric="rbf", gamma=0.2)
fit = tp.linalg.svd(A, k=4, subspace_size=12, rtol=1e-6, atol=1e-9)
if not fit.converged.all():
    raise RuntimeError(f"SVD status: {fit.status.tolist()}")
# V:[M,R]. Apply the fitted approximation without further pair evaluations.
result = fit.left_vectors @ (fit.singular_values[:, None] * (fit.right_vectors.T @ V))
```

The last line is unbatched; use `transpose(-1,-2)` and `unsqueeze(-1)` for
batched factors. `A.centered(mean)` can also be decomposed.

```cpp
#include <torchpairwise.h>
namespace s = torchpairwise::streaming;
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
auto result = at::matmul(fit.left_vectors,
    fit.singular_values.unsqueeze(-1) *
    at::matmul(fit.right_vectors.transpose(-1,-2), V));
```

Require `1<=k<=r<=min(N,M)`. Width r defaults to `min(N,M,2*k+8)` or the
last dimension of an explicit `initial:[...,M,r]`. That right search block must
have full numerical column rank and broadcast to operator batches without adding
batches. Otherwise the solver draws a random block using the current PyTorch
RNG state. Use an explicit block for reproducibility across repeated calls.

## Iteration, Convergence and Memory

Alternating products with A and A.T are stabilized by QR. A small `[r,r]`
projected SVD gives candidates. Neither A nor a normal matrix A.T A is formed.
True forward and adjoint source products check each triplet:

\[
\max\bigl(\|Av_i-s_i u_i\|_2,\|A^\mathsf T u_i-s_i v_i\|_2\bigr)
\le\max(\mathrm{atol},\mathrm{rtol}s_i),\qquad
U^\mathsf T U\approx I,\quad V^\mathsf T V\approx I
\]

Every pass evaluates the underlying source pairs. Retained numerical state uses
`O((N+M)*r+r*r)` double blocks plus inputs, source preparation, tiles and outputs.
Width near `min(N,M)` approaches dense storage. An existing dense matrix input
keeps its allocation. Fitted factors can subsequently apply an approximate
matrix product in `O((N+M)*k*R)` work, without reevaluating source entries.

`max_iter` counts updates after an initial two-sided extraction. Zero evaluates
only that extraction. Each batch stops when all requested triplets converge;
individual vectors are not locked and may rotate within repeated singular
subspaces. Finite rank collapse is valid, including zero maps; positive `atol`
is useful for null directions. Diagnostics do not certify global leading order
if a caller-supplied invariant start misses larger singular directions. Random
starts, sufficient width and updates are needed for flat spectra.

| Status | Meaning |
| --- | --- |
| 0 | Residual and orthogonality checks pass |
| 1 | Iteration limit; finite estimates retained |
| 2 | Rank-deficient initial search block |
| 3 | Nonfinite action or arithmetic |
| 4 | Loss of vector orthogonality |

Status 2-4 fails the batch with NaN estimates; malformed options raise. Native
stopping synchronizes CUDA, so graph capture is unsupported. This decomposition
is inference-only: use explicit no-grad for gradient-bearing inputs. Compatible
compile/export retains a native operator. It computes leading triplets only,
without a smallest-value mode or derivative through singular vectors.

The [rectangular SVD example](../examples/rectangular_svd.py) checks fitted
singular values, low-rank reconstruction and an approximate RHS action against
a small dense SVD. Omit `--dense-check` for large inputs.
