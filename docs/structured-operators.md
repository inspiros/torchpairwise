# Streamed Covariance, Gaussian and Mixture Operators

`tp.lazy` accepts the same structured collections as `compare`:

```python
# Means:[...,N,D], covariances:[...,N,D,D]
A = tp.lazy(
    (means_x, covariances_x), (means_y, covariances_y),
    metric="gaussian_wasserstein", squared=True, chunk_size=32,
)
product = A.matmul(V)
column_means = A.mean(dim=-2)
form = A.bilinear(L, R)

# SPD covariance collections:[...,N,D,D]
B = tp.lazy(covariances_x, covariances_y, metric="log_euclidean")

# Mixture means/covariances/weights:[...,N,K,D]/[...,N,K,D,D]/[...,N,K]
C = tp.lazy(
    (mixture_means_x, mixture_covariances_x, weights_x),
    (mixture_means_y, mixture_covariances_y, weights_y),
    metric="gaussian_mixture_sinkhorn", regularization=.1, max_iter=50,
    x1_lengths=lengths_x, x2_lengths=lengths_y, chunk_size=8,
)
reverse_product = C.T.matmul(U)
```

The common [operator contracts](pairwise-operators.md) apply: products,
bilinear/quadratic forms, sum/mean/min/max/prod/all/any along either or both
sample axes, logical transpose, diagonals and selected blocks. Batch axes of
all collection components, lengths and coefficients broadcast. Sample axes
and feature/component axes must match the named comparison's requirements.
`shape`, `dtype` and `device` describe the scalar matrix.

Supported comparisons are Gaussian Wasserstein, covariance log-Euclidean,
affine-invariant and Stein, and Gaussian-mixture overlap, L2,
Cauchy-Schwarz, exact component transport, entropic transport cost and Sinkhorn
divergence. Aliases and full names follow `list_comparisons`. Covariance/Gaussian
`squared` and mixture parameters retain their named meanings; unused options
raise. Exact mixture transport retains its existing active-component limit and
CPU solver/device-transfer requirements. Streaming the outer sample matrix does
not make exact component transport GPU-only or remove its component workspace.

## Shared Preparation and Gradients

Native preparation validates the original collections before handling empty
outputs or selecting entries, using the same metadata checks as the named
structured comparisons. Gaussian covariance roots, covariance logs/Cholesky
factors, normalized mixture weights and mixture covariance transforms are built
once per collection per forward call. For mixture L2/Cauchy-Schwarz and Sinkhorn,
per-sample self-overlap/self-transport terms are prepared once as well. A self
collection reuses preparation when its lengths agree. Different prefix lengths
require separate preparation, even when raw tensors are shared.

Matrix operations consume slices of these prepared features. First backward
replays pair tiles, accumulates feature cotangents and differentiates collection
preparation once per operand, including gradients through weight normalization
and cached self terms. This reuses the canonical structured pair formulas and
matrix derivatives. No Python metric callback or backward callback is used.
Higher derivatives are explicitly unsupported.

The complete `[N,M]` comparison matrix is not kept for reductions, products or
forms. Features, input/feature gradients and requested outputs retain their
natural storage; prepared features may include broadcast batches. Scalar
reducers retain row-sized state. `chunk_size` bounds outer sample/product-channel
tiles; density mixtures also accept `component_chunk_size` for component work.
Transport algorithms retain their own inner component working sets. A tile width
is not a fixed total-memory budget. Explicit `block` requests materialize the
requested block only, while `diagonal` evaluates aligned entries.

No feature or matrix cache survives a call. Later tensor mutations are visible;
repeated methods prepare again. First-derivative, zero/empty, PSD/SPD and padded
component conventions remain those of the named APIs. Signed divergences or
transport values are not silently converted into positive kernels.

## Deployment

Forward/backward are C++/ATen on CPU/CUDA, with float32/float64 and autocast
promotion. Use `.matmul()` in compiled/exported modules and construct the binding
inside `forward`, as for vector operators. Tuples retain their component shapes
in dynamic `torch.export` inputs.

Standalone C++ uses `StructuredPairwiseOperator` and
`structured_pairwise_operations` from
`<ops/structured_operations.h>`. A covariance collection is a
one-tensor vector, a Gaussian collection a two-tensor vector, and a mixture
collection a three-tensor vector. `StructuredOperationOptions::matrix` contains
common matrix controls; squared/regularization/length/component options have
their own fields. Its vector/cloud comparison packet must remain unset.

See [the example](../examples/structured_operators.py).
