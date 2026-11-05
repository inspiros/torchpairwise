<div class="eyebrow">Getting started</div>

# Using Torchpairwise in Models

Many models compute a pairwise score matrix and immediately reduce it:
attention normalizes and applies it, a contrastive loss takes a row
log-sum-exp, a kernel method solves with it, and retrieval keeps a few best
entries. In PyTorch, each of these allocates the full `[N,M]` matrix and keeps
it for backward. In torchpairwise, you bind the scores with `tp.lazy`, then call
the consumer. The consumer evaluates the pairs in bounded tiles, or on fused
CUDA kernels, and never stores the matrix. First derivatives replay the tiles
instead of saving them.

```python
import torch
import torchpairwise as tp

scores = tp.lazy(Q, K, metric="linear") * Q.shape[-1] ** -0.5  # [N,M], not stored
out = scores.softmax_matmul(V)                                 # [N,C]
out.sum().backward()                                           # grads for Q, K, V
```

Memory then scales with the inputs and outputs, `(N+M)*(D+C)`, plus a tile
workspace. The time cost is unchanged: every pair is still evaluated, and
backward evaluates them again. Streaming pays off when N*M is the memory
bottleneck, not when the matrix is small.

## Task Map

Each row is a native operation with first-order autograd on CPU and CUDA.
"Fused" marks consumers that run on fused CUDA kernels for float32 or float64
inputs with 1 to 64 features and a supported source; see
[execution backends](streaming.md#execution-backends).

| Task | Call | Notes |
|---|---|---|
| Attention | `(tp.lazy(Q, K, "linear") * s).softmax_matmul(V, mask=...)` | Fused without a pair mask; masks: `tp.masks.causal()`, `band`, `same_group` |
| Kernel attention, Nadaraya-Watson | `tp.lazy(X, Y, "rbf", gamma=g) @ torch.cat([V, ones], -1)` | Fused; the last column is the row mass |
| InfoNCE, NT-Xent | `S.logsumexp(-1, exclude_self=True)` on `S = tp.lazy(z, z, "linear") / t` | Fused; recipe in `tests/test_ml_recipes.py` |
| k-NN, retrieval | `tp.lazy(X, R, "sqeuclidean").topk(k, largest=False)` | Fused, exact tie order; `argmin`, `min` |
| Chamfer-style losses | `D.min(-1)` and `D.min(-2)` on `D = tp.lazy(X, Y, "sqeuclidean")` | `min` returns values and indices |
| MMD loss, two-sample test | `tp.mmd_statistics(X[None], Y[None], kernel="rbf", gamma=g)` | Unbiased by default |
| HSIC, CKA | `Kx.hsic(Ky)`, `Kx.frobenius_inner_product(Ky, center=True, normalize=True)` | Two lazy kernels on the same samples |
| Kernel ridge, GP mean | `tp.linalg.solve(K + a * tp.eye_like(K), Y, method="cg")` | Preconditioners: `tp.linalg.pivoted_cholesky`, Nyström |
| GP variance | `tp.linalg.love_cache`, `tp.linalg.predictive_variance` | See [GP prediction](spectral.md#gaussian-process-prediction) |
| Kernel PCA, spectral embedding | `tp.linalg.eigh(K.centered(), k=r)` | Low-rank: `tp.linalg.svd` |
| Kernel SVM | `tp.optimize.box_qp(Q, q, lower=0., upper=C, equality=...)` | See the [example gallery](../examples/index.md#binary-kernel-svm) |
| Optimal transport loss | `tp.optimize.sinkhorn_divergence(C_xy, C_xx, C_yy, a, b, epsilon=e)` | Any lazy cost; see [entropic transport](algebra.md#entropic-transport) |
| Landmarks, coresets | `tp.neighbors.farthest_point_sampling(D, k)`, `tp.neighbors.kmeans_plusplus(D, k)` | Greedy selection; returns indices |
| Learned metrics | `tp.make_pairwise(fn)`, encoder heads | See [custom comparisons](../metrics/custom-comparisons.md) |

`X[None]` in the MMD row is not a typo: `mmd_statistics` compares batches of
point clouds, so one feature set is one cloud.

## Score Expressions

Build scores with ordinary arithmetic on the lazy object, then consume once.
Offsets and weights per row or column (`add_rows`, `add_columns`,
`weight_rows`, `weight_columns`), `exp`, and boolean row/column masks stay
fused on CUDA. Their row and column tensors can require gradients, so learned
biases and per-sample weights train through the same streamed backward.

```python
D = tp.lazy(X, Y, metric="sqeuclidean")
logits = (-0.5 / tau * D).add_rows(row_bias).add_columns(column_bias)
loss = -logits.logsumexp(-1).mean()  # fused on float32 CUDA
```

Pair masks from `tp.masks` (`causal`, `band`, `same_group`) and other
expression nodes such as `sqrt`, `pow` or `minimum` work on every consumer
but use tiles.

## Stein Variational Gradient Descent

SVGD moves particles `x` toward a target density `p` with the direction

\[
\phi(x_i)=\frac{1}{n}\sum_j k(x_j,x_i)\nabla\log p(x_j)+\nabla_{x_j}k(x_j,x_i)
\]

For an RBF kernel `exp(-gamma*|x-y|^2)`, the repulsive term is
`-2*gamma*sum_j k_ij*(x_j-x_i)`, which the kernel's `displacement` method
streams. One update therefore takes one kernel product and one displacement,
with no `[n,n]` matrix:

```python
def svgd_direction(x, score, *, gamma):
    K = tp.lazy(x, x, metric="rbf", gamma=gamma)
    return (K @ score - 2 * gamma * K.displacement()) / x.shape[-2]

for _ in range(steps):
    x = x + step_size * svgd_direction(x, score_fn(x), gamma=gamma)
```

`tests/test_ml_recipes.py` checks this direction against a dense autograd
oracle. A kernel Stein discrepancy also needs score inner products and the
kernel's second derivatives per pair. No single consumer computes them, so the
library does not provide one.

## What to Check

- **Precision.** Fused CUDA kernels sum in a different order from the tiles
  (and from the matrix product behind dot-product scores), so results agree
  within floating tolerance, not bitwise. See
  [precision and deployment](deployment.md).
- **Gradients.** Supported vector comparisons and streamed sums, means,
  matrix products, logsumexp and softmax products support second derivatives,
  including gradient penalties and Hessian-vector products. Higher-order replay
  retains tile intermediates and may use O(N*M) memory. Selections (`topk`,
  `argmin`) route gradients through the selected pairs only, not the indices.
  Other operations may raise for higher derivatives.
- **Solvers.** Results carry `converged` and `status`; check them before using
  `solution`. Differentiable solves need `differentiable=True`.
- **Deployment.** `torch.compile`, `torch.export` and LibTorch run the same
  native operations, including the fused kernels. The optional KeOps backend
  is Python-only and eager; see
  [execution backends](streaming.md#execution-backends).
