# Missing Features and Ordered Histograms

These vector distances support all-pairs, paired and condensed outputs, leading
batch broadcasting, float32/float64, autocast and compiled first gradients.
The additional semantics below determine what the feature axis means.

## NaN-Aware Euclidean Distance

`nan_euclidean_distances(X,Y)` treats only NaN as missing. Let V be the features
observed in both vectors, K their count, and D the full feature count:

\[
d(x,y)=\sqrt{\frac{D}{K}\sum_{i\in V}(x_i-y_i)^2}\]

This matches the weighting in scikit-learn's
[`nan_euclidean_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.nan_euclidean_distances.html).
`squared=True` returns the expression under the square root. The distance equals
ordinary Euclidean when there are no missing features. It does not impute data.

```python
import torch
import torchpairwise as tp

X = torch.tensor([[1., float("nan"), 3.], [2., 4., 6.]])
Y = torch.tensor([[2., 2., 3.], [float("nan"), 5., 7.]])
D = tp.nan_euclidean_distances(X, Y)
paired = tp.paired_distances(X, Y, metric="nan_euclidean")
condensed = tp.pdist(X, metric="nan_euclidean", squared=True)
```

Inputs may contain finite real values and NaNs; infinities are rejected. If no
feature overlaps, the output is NaN, including a fully missing row's self
comparison. Such pairs receive a chosen zero gradient; a NaN output does not
become a valid training target because of that convention. Missing coordinates
also receive zero gradients. Gradients are tested with fixed missingness
patterns, and zero distances use a zero subgradient. Changes in missingness
are not differentiable.

The all-pairs implementation uses fused CPU/CUDA reductions and stores one int64
observed count per compared pair for backward. It does not allocate `[N,M,D]`
differences or masks. Paired execution uses corresponding-row masks/reductions;
condensed execution computes row suffixes. This distance also works in the
existing dense neighbor-mask APIs. A radius comparison against NaN is false;
the k-neighbor function still selects exactly k entries and orders NaNs after
finite distances, so it can select undefined pairs when too few finite ones exist.

## One-Dimensional Wasserstein Distance

`wasserstein_1d_distances(X,Y)` treats each feature as histogram mass at one
shared bin position. Inputs are finite nonnegative masses with positive finite
row totals, normalized independently to probability distributions.

By default bins are at positions `0,1,...,D-1`. Supply a shared one-dimensional
`bin_positions` tensor with D entries for an uneven grid. It must have the
input dtype/device and finite, strictly increasing entries with finite gaps.
The same grid applies to every sample and batch in both inputs.

For normalized masses p,q at sorted positions z, with cumulative sums P,Q:

\[
W_1(p,q)=\sum_{i=0}^{D-2}|P_i-Q_i|(z_{i+1}-z_i)\]

This is the exact one-dimensional earth mover's distance for these discrete
distributions. A shared bin grid makes it a weighted L1 distance between CDFs;
the implementation reuses existing native/PyTorch L1 reductions. It never
constructs a pairwise feature tensor. See SciPy's
[`wasserstein_distance`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)
for the distribution interpretation; use its bin positions as support values
and histogram rows as weights when comparing values.

```python
X = torch.tensor([[1., 0., 0.], [0., 0., 2.]])
Y = torch.tensor([[0., 0., 1.], [0., 3., 0.]])
positions = torch.tensor([0., 1., 4.])
D = tp.wasserstein_1d_distances(X, Y, bin_positions=positions)
paired = tp.paired_distances(X, Y, metric="wasserstein_1d",
                             bin_positions=positions)
condensed = tp.pdist(X, metric="wasserstein_1d", bin_positions=positions)
```

Transport across farther bins costs more. Multiplying an individual row by a
positive scalar leaves its normalized distribution unchanged. Translating every
bin position leaves the distance unchanged; scaling the grid scales distance.
A single shared bin gives zero distances and zero gradients. The API does not
interpret row entries as raw support samples, compare separate grids, or solve
multidimensional optimal transport.

For point distributions in more than one coordinate dimension, see the
[point-set transport guide](transport.md). Its exact, entropic and sliced
functions use support vectors and retain a separate API from these histograms.

Mass gradients include normalization. Grid positions also support gradients
while their strict order is maintained; cumulative ties use a zero subgradient
for absolute value. All-zero histograms are rejected, while empty sample/batch
axes remain supported. Grids are not sorted automatically, since that would
also require permuting histogram masses. Floating-point overflow/underflow
limits still apply to masses, normalization and spacings.

## API Scope

Both names work in public `cdist`, `paired_distances`, `pdist` and neighbor masks.
`bin_positions` belongs to Wasserstein; `squared` belongs to NaN-aware Euclidean
and the existing Chamfer metric. Unsupported options are rejected.
Inputs must be dense strided tensors. The legacy native `torch.ops` string
catalog is unchanged; use the public Python functions for these additions.
Rebuild the extension to obtain the new private NaN-aware primitive.

See the [runnable example](../examples/missing_and_ordered.py) and
[shared contracts](api-contracts.md).
