# GP Predictive Variance with LOVE

`tp.linalg.love_cache(A, rank, initial=v)` prepares an inference-only inverse
root approximation for a symmetric positive definite streamed training system
`A = K + noise I`. Native Lanczos forms an orthonormal basis Q and its small
tridiagonal projection T. Factoring `T = L L.T` gives `root = Q L^{-T}`, so
`A^{-1}` is approximated by `root @ root.T`. No training N-by-N matrix is
formed. The returned `LOVECache` stores `root:[...,N,rank]`; early breakdown
leaves zero padded columns.

```python
import torch
import torchpairwise as tp

# X:[N,D], targets:[N], test:[Ntest,D]; noise is a positive variance.
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=.2)
    A = K + noise * tp.eye_like(K)
    cache = tp.linalg.love_cache(A, rank=32, initial=targets)

cross = tp.lazy(test, X, metric="rbf", gamma=.2)
prior = tp.lazy(test, metric="rbf", gamma=.2).diagonal()
variance = tp.linalg.predictive_variance(cache, cross, prior)
```

`predictive_variance` returns the latent marginal variance
`prior - ((cross @ cache.root)**2).sum(-1)`, **clamped at zero**. Add observation
noise separately for noisy predictive observations. The clamp prevents negative
reported variances from approximation or roundoff; a clamped entry has zero
clamp gradient. This helper composes the native streamed matmul and ordinary
Tensor operations. Gradients flow through the cross kernel and supplied prior;
the training cache is constant. It does not differentiate the training solve or
cache construction.

Reuse one cache for multiple test batches. Each call streams the rectangular
cross kernel rather than storing all training-to-test entries. Cache storage is
O(Nr); the test action stores O(Ntest r) projected values. Larger rank improves
the inverse approximation at increased setup, storage and prediction cost. Full
rank recovers the dense SPD result up to numerical error when the initial vector
spans every eigenspace needed by the cross kernel. An invariant-subspace
breakdown can finish earlier; increasing the requested rank cannot introduce
components absent from that starting Krylov space. Try a different nonzero
initial vector when necessary, and compare ranks on representative predictions.
This is the LOVE construction of Pleiss et al. (2018).

## Inspecting the Lanczos Projection

`tp.linalg.lanczos(A, k, initial=v, reorthogonalize=True)` exposes the primitive
used by LOVE. A may be a scalar kernel or a symmetric linear composition such
as `K + noise I`; only matvecs are required. Its `LanczosResult` contains
`basis:[...,N,k]`, `alpha:[...,k]`, `beta:[...,k-1]`, and batch-shaped integer
`iterations` and `status`. The effective tridiagonal is built from the leading
`iterations` alpha entries and one fewer beta entries. Unused entries are zero.
Statuses distinguish completing the budget, invariant-subspace breakdown,
nonfinite arithmetic and an invalid initial vector.

The recurrence computes in float64 with full reorthogonalization by default;
returned numerical tensors use the operator dtype. Setup is inference-only and
requires explicit no-grad for gradient-bearing inputs. Supply a nonzero
`initial:[...,N]` matching dtype/device. Neither symmetry nor global positive
definiteness can be established by finitely many products. LOVE requires a
positive definite projected system and adds no hidden jitter.

## Streaming Landmark Selection

`tp.neighbors.farthest_point_sampling(D, k, start=0)` accepts a scalar
self-distance operator and returns `(indices, distances)`. It updates the nearest
selected distance using one native row block per step, evaluating O(Nk) entries.
Returned distances are the insertion distances: the first is infinity, and each
subsequent entry is its distance to the nearest previously selected point.
Ties use the lowest unused index, including coincident points.

`tp.neighbors.kmeans_plusplus(D, k, generator=None)` returns indices for a
**squared-distance** self operator. The first index is uniform; subsequent
weights are the minimum supplied squared distance to the selected set. Do not
square these weights again. When all remaining weights are zero, the lowest
unused index is selected. Both selectors are native and inference-only.

A supplied `torch.Generator` must match the operator device. Reusing its seed
reproduces selection for the same inputs and device; sequences need not match
across devices. For example, use `torch.Generator(device=X.device).manual_seed(7)`.
The resulting indices can select Nyström landmarks as described in
[preconditioning](preconditioning.md).
