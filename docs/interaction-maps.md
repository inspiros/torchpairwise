# Streamed Displacement and Transport Maps

These APIs reduce vector-valued pair interactions without saving a full
`[..., N, M, D]` tensor. Inputs are dense point collections `[..., N, D]` and
`[..., M, D]`; all leading batch axes broadcast. They do not require sparse
PyTorch tensors or compare an outer collection of clouds against another.
The result is one vector per source point, `[..., N, D]`.

## Kernel-Weighted Displacement

```python
K = tp.lazy(X, Y, metric="rbf", gamma=.2)
force = K.displacement()                # sum_j Kij * (Yj - Xi)
shift = K.displacement(normalize=True)  # force / sum_j Kij
# Equivalent functional entry point:
shift = tp.lazy(X, Y, "rbf", gamma=.2).displacement(normalize=True)
```

This supports all registered vector kernels. A single streamed product with
coordinate and constant channels computes both weighted coordinates and row
mass. Backward reuses the native product's tile replay. This is useful for
registration updates and geometric message passing. Signed/indefinite kernels
are allowed: a normalized field has a barycentric interpretation when its
weights are nonnegative and have positive row mass. Exactly zero row mass gives
zero normalized displacement, with zero derivatives through that branch. An
empty reference collection also gives zero. Unnormalized displacement remains
the weighted sum even if signed weights cancel in the row mass. The factored
expression can lose precision when large coordinate offsets cancel; center
coordinates when small displacements matter at those scales.

## Balanced Entropic Transport

```python
mapped = tp.sinkhorn_barycentric_map(
    X, Y, source_weights=a, target_weights=b,
    regularization=.1, max_iter=100, p=2, chunk_size=128,
)
shift = tp.sinkhorn_displacement(X, Y, regularization=.1, max_iter=100)
```

The ground cost is `||Xi-Yj||**p`, for `p=1` or `p=2`. These functions use
balanced entropic transport with the supplied regularization; they do not
subtract self costs as a Sinkhorn divergence does. Positive regularization
spreads transport across partners: a self-map is generally not the identity.
This is not the exact unregularized Wasserstein flow solver.

Masses have shapes `[..., N]` and `[..., M]`, matching the coordinate dtype and
device; their batch axes also broadcast. They default to uniform masses, are
normalized independently, and must be finite/nonnegative with positive finite
totals. Collections must be nonempty, and coordinates finite. Zero target
masses contribute no final transport. Zero source-mass rows return zero in both
the map and displacement. Coordinates of zero-mass points must still be finite;
there is no prefix-length/padded-coordinate API here.

Starting with log target scaling zero, each iteration first updates the source
log scaling and then the target log scaling using tiled logsumexp. The final
barycentric map normalizes each positive-source row of the resulting plan.
This defines a finite-iteration approximation: row conditional weights sum to
one, but its normalized plan need not satisfy both requested marginals until
convergence. `max_iter` is a fixed positive iteration count, not a tolerance or
convergence guarantee. Check stability as the iteration budget increases for
your workload. The returned displacement is this map minus X, on positive-mass
source rows. Neither function returns or caches a dense transport plan.

## Memory, Gradients and Deployment

The numerical algorithms and backward execute in C++/ATen on CPU or CUDA.
They support float32/float64, with half/bfloat16 inputs promoted to float32 under
autocast. Outside autocast, reduced precision is rejected by these APIs.
`chunk_size` bounds sample-axis tiles, not total memory. Batched inputs process
all broadcast batches in a tile; inputs and output require their natural space.

Transport backward differentiates the fixed iterations, including coordinates
and nonzero masses; it does not substitute an envelope or implicit derivative
for a converged solution. Each logsumexp/map operation retains its operands and
row output, then replays cost tiles for its VJP. Saved solver state scales as
`O(max_iter * batch * (N+M))`, plus shared input references and the final vector
output. Cost workspace is tiled in forward and backward. No full cost/plan
matrix or pair-feature tensor is retained. Higher transport-map derivatives
are explicitly unsupported. At zero mass the selected support is fixed;
no derivative for inserting a new support point is promised.

Both functions can be used inside `torch.compile`/`torch.export` modules.
Construct kernel facades inside `forward` and call explicit methods.
As in other native comparison APIs, invalid eager values raise synchronous
errors; graph/capture checks use the shared asynchronous assertion policy.
Scalar options (iteration count, tile size, p and regularization) are execution
settings, not differentiable tensors.

Standalone LibTorch uses `KernelOperator::displacement(bool normalize)` and
`sinkhorn_map(x, y, SinkhornMapOptions)` from
`<ops/kernel_operations.h>` and
`<ops/transport_maps.h>`. Set `options.displacement=true` for a
transport displacement; the numerical path is shared with Python.

See [the runnable example](../examples/interaction_maps.py).
[Image maps](image-maps.md) and [residual outer products](kernel-moments.md)
are available as separate APIs. Arbitrary vector interactions and transport-plan
operator products remain future extensions.
