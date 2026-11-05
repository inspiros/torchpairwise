# Streamed Residual Outer Products

Compute a second-moment matrix per query point without keeping pairwise
outer products or building a target feature bank of shape `[M,D,D]`:

```python
K = tp.lazy(X, Y, metric="rbf", gamma=.2, chunk_size=64)
second = K.residual_outer_product()  # [...,N,D,D]
mean_second = K.residual_outer_product(normalize=True)
local_covariance = K.residual_outer_product(normalize=True, center=True)
# Functional form:
second = tp.lazy(X, Y, "rbf", gamma=.2).residual_outer_product()
```

Let `delta_ij=Yj-Xi` and `mass_i=sum_j Kij`. Without centering, the result is
`sum_j Kij * delta_ij * delta_ij.T`; `normalize=True` divides by row mass.
With `center=True`, replace each residual by its deviation from the row's
kernel-weighted mean residual. Normalized centered output is the population
local covariance for nonnegative weights with positive mass; no sample/bias
correction is applied. Unnormalized centered output is the weighted centered
sum. All twelve registered vector kernels are supported, with their existing
parameters/domains. Signed kernels are algebraically supported and carry no
probability or positive-semidefinite covariance guarantee.

Exactly zero mass gives a zero matrix when centering or normalization is
requested, with zero gradients through that branch. Unnormalized, uncentered
moments preserve their algebraic sum even when signed weights cancel in mass.
Empty reference collections therefore give zero; empty queries give an empty
output. There is no epsilon clamping of small nonzero mass. Self calls include
self partners; their uncentered residual is zero, but their weight contributes
to normalization and centering.

## Accuracy, Memory and Gradients

The uncentered algorithm accumulates second/first moments and row mass in one
native pass. Centering uses two passes: first compute the mean residual, then
accumulate outer products of centered coordinates. This avoids subtracting two
large raw second moments to recover a small covariance. Accumulation and
coordinate differences use float64; kernel values and their derivatives keep
the canonical native kernel's compute policy. Floating-point conditioning and
representability still apply.

Both sample axes and coordinate channels are tiled. A pair-feature tile is
contracted as matrix products; a pair-of-points outer-product tensor
`[N,M,D,D]` is never needed. The requested `[N,D,D]` output and its gradient
still need their natural storage. Centering additionally retains row means;
normalization/centering may retain output-sized intermediates for their VJPs.
`chunk_size` bounds sample/channel tile widths, not all allocations. Broadcast
batches are supported, and noncontiguous inputs are accepted. There is no
hidden kernel/feature cache.

Backward replays kernel tiles, applies analytic residual VJPs, and delegates
kernel derivatives to the canonical native catalog. First coordinate gradients
are supported, including batch reductions and derivatives through the computed
mean/mass; higher derivatives are explicitly unsupported. Numeric kernel
settings are not differentiable tensor parameters. CPU/CUDA float32/float64 and
autocast promotion match [kernel operators](kernel-operators.md).

Python methods and functions support `torch.compile`/`torch.export`. Standalone
C++ exposes `KernelOperator::residual_outer_product(normalize, center)` and the
exported `kernel_residual_outer_product` function with `KernelOperationOptions`.
No Python callback is used by built-in forward or backward.

This is a specialized consumer of a virtual pairwise matrix-valued interaction,
with event shape `(D,D)`. It does not expose an unreduced lazy outer-product
source. See [multiple feature axes](operator-event-shapes.md) for the general
shape contract and why this factorized implementation should remain specialized.
