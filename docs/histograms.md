# Histogram Distances and Kernels

These functions compare ordinary dense vectors of bin masses. Inputs use the
same `[N,D]`, `[M,D]` and broadcast-batch conventions as other vector metrics.
Every bin must be finite and nonnegative. Sparse tensors are not supported.

## Hellinger Distance

`hellinger_distances(X,Y)` normalizes each row independently to a probability
distribution and computes

\[
H(x,y)=\frac{\lVert\sqrt{p}-\sqrt{q}\rVert_2}{\sqrt{2}},\qquad
p=\frac{x}{\sum_i x_i},\quad q=\frac{y}{\sum_i y_i}\]

Rows must have positive finite total mass. All-zero rows are rejected; zeros
within a positive-mass row are valid. Identical distributions have distance zero
and disjoint distributions have distance one, up to floating-point rounding.
Multiplying a row by a positive scalar does not change its distribution or
distance. Input gradients include the effect of normalization.

```python
import torch
import torchpairwise as tp

X = torch.tensor([[1., 3., 0.], [0., 0., 2.]])
Y = torch.tensor([[2., 6., 0.], [1., 0., 0.]])
matrix = tp.hellinger_distances(X, Y)  # [2,2]
paired = tp.paired_distances(X, Y, metric="hellinger")  # [2]
condensed = tp.pdist(X, metric="hellinger")  # [1]
```

The string name is `hellinger`, including in `cdist`, `pdist` and neighbor masks.
This is the distance, without squaring. It is not the Bhattacharyya distance
`-log(sum(sqrt(p*q)))`.

Gradients are tested for strictly positive bins. The derivative of the square-root
transform is singular at zero; this implementation assigns that transform a
zero derivative at zero, without adding epsilon or changing forward values.
Normalization still couples the bins in each row, so a zero-valued input bin
need not have zero total input gradient. Coincident distances use a zero
subgradient. Probability boundaries are outside the smooth-gradient contract.

## Histogram Intersection

`histogram_intersection_kernel(X,Y)` computes

\[
K(x,y)=\sum_i\min(x_i,y_i)\]

It compares raw masses without normalizing. All-zero rows are valid. Self
similarity equals the row's mass; when both inputs are probability distributions,
values lie between zero and one. Scaling both inputs by a positive factor scales
the kernel by that factor. Normalize explicitly before calling if that is the
desired meaning for your data.

```python
matrix = tp.histogram_intersection_kernel(X, Y)
paired = tp.paired_kernels(X, Y, kernel="histogram_intersection")
condensed = tp.condensed_kernels(X, kernel="histogram_intersection")
```

At an equal pair of bin values, the derivative is split equally between the
inputs, matching `torch.minimum`. This includes ties at zero. Other bins give
their derivative to the smaller input. These derivatives are conventions at
nondifferentiable ties, not smooth derivatives there.

The all-pairs kernel sums minima directly in fused CPU/CUDA reductions. It does
not allocate `[N,M,D]` feature intermediates or recover intersection by
subtracting L1 distance from row sums. Direct accumulation preserves small
overlaps beside large bins that the subtraction can lose. Backward scans pairs
to accumulate each input gradient without a feature-sized pairwise buffer.

## KL and Symmetric KL

`kl_divergences(X,Y)` independently normalizes nonnegative positive-mass rows
to p and q, then returns

\[
D_{KL}(p\Vert q)=\sum_i p_i\log(p_i/q_i)\]

`symmetric_kl_divergences(X,Y)` returns the Jeffreys divergence
`KL(p||q) + KL(q||p)`. It is the sum, with no half factor or square root.
Neither divergence is a metric with a triangle inequality. KL is asymmetric;
it supports all-pairs and paired outputs. Symmetric KL also supports condensed
output. Generic strings are `"kl"` and `"symmetric_kl"`, including dense
neighbor masks. The existing Jensen-Shannon distance remains a different
bounded, rooted comparison.

```python
X = torch.tensor([[1., 2., 3.], [2., 1., 4.]])
Y = torch.tensor([[3., 1., 2.], [1., 4., 2.]])
matrix = tp.kl_divergences(X, Y)
paired = tp.paired_distances(X, Y, metric="kl", base=2)
condensed = tp.pdist(X, metric="symmetric_kl")
```

Natural logarithms are the default; `base` can be a finite scalar greater than
one (base 2 gives bits). Inputs must be finite/nonnegative, with positive finite
row totals; no epsilon smoothing is added. Scaling a row by a positive constant
leaves its distribution unchanged. The forward conventions follow
[SciPy relative entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html):
zero p contributes zero, including when q is zero; positive p with zero q makes
KL infinite. Thus Jeffreys is infinite if either directional support condition
fails. Different supports do not necessarily make directed KL infinite.

Positive-bin derivatives include normalization. Float32 KL uses float64
intermediates and rounds its final output to float32, preserving small values
and normalization residuals in near-identity gradients; this trades throughput
for accuracy. At a zero raw input bin, this
API assigns zero input derivative, including through normalization. A whole
infinite comparison receives zero derivatives for both operands, including the
finite direction of an infinite Jeffreys pair. These are explicit boundary
conventions, not smooth derivatives or a way to make infinite losses valid
training targets. Use strictly positive distributions for smooth-gradient
optimization. Clamping tiny negative roundoff to zero does not smooth supports.

All-pairs uses bounded sample/feature tiles and checkpoint recomputation during
training, rather than retaining a full `[N,M,D]` intermediate. A local
relative-entropy series preserves small near-identity results. Paired execution
uses direct corresponding-row reductions; condensed uses row suffixes. Tile
loops may specialize on sample/feature counts under compilation. Float32/float64,
autocast and compiled first gradients are supported. Numerical representability
limits still apply; positive raw support is retained even when its normalized
probability underflows. See the [example](../examples/probability_and_weights.py).

## Execution

Hellinger and histogram intersection support float32/float64, AMP promotion, input gradients, compiled
forward/backward, noncontiguous inputs, broadcasting and all three comparison
modes. Paired execution uses direct row reductions; condensed execution uses
strict row suffixes, without computing a full square matrix. Empty sample/batch
axes are supported; empty feature axes are rejected.

Dense all-pairs output still requires N*M storage. Autograd may retain input and
distance intermediates. No new general reduction API is introduced. Domain
validation follows the [shared eager/compiled policy](pytorch-integration.md).
The public functions and string dispatch are Python APIs; the fused intersection
primitive is private and does not add these names to the legacy native
`cdist`/`pdist` catalog. Rebuild the extension after updating this stage.

See the [example](../examples/histograms.py) and
[comparison contracts](api-contracts.md).
