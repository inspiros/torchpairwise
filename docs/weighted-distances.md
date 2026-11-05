# Feature-Weighted Distances

`cosine_distances(X,Y,w=w)` and `correlation_distances(X,Y,w=w)` accept
nonnegative feature weights with positive total weight in each batch. The
optional `w` tensor ends in `[D]`; preceding axes broadcast with input batch
axes. It must match input dtype/device and contain finite values. These
functions use dense strided float32/float64 tensors, with the usual promotion
to float32 under autocast. Weight gradients are supported for positive weights.

Weighted cosine is

\[
d(x,y)=1-\frac{\sum_i w_i x_i y_i}
 {\sqrt{\sum_i w_i x_i^2}\sqrt{\sum_i w_i y_i^2}}\]

Weighted correlation applies this formula to vectors centered at their own
weighted means, `sum(w*x)/sum(w)`. Both results are clipped to `[0,2]` to remove
roundoff beyond the mathematical interval. Multiplying all weights by the same
positive scalar leaves the result unchanged. Weights are normalized internally
after max rescaling so summing large finite weights does not overflow.

```python
import torch
import torchpairwise as tp

X = torch.rand(4, 5)
Y = torch.rand(3, 5)
w = torch.tensor([1., 2., 0., .5, 1.])
matrix = tp.cosine_distances(X, Y, w=w)
paired = tp.paired_distances(X[:3], Y, metric="correlation", w=w)
condensed = tp.pdist(X, metric="cosine", w=w)
```

The formulas follow SciPy's
[weighted cosine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html)
and [weighted correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html).
Our existing zero-norm cosine convention still applies: cross-input distance is
one if either supported vector has zero norm, with a chosen zero gradient for
that comparison. Self comparisons explicitly zero the diagonal when Y is omitted
or the same tensor object; paired comparisons of the same object also return
zero. A separate equal-valued tensor uses cross-input semantics. Weighted
correlation is NaN for a row constant over its positive-weight support, even
when zero-weight coordinates vary. A single supported feature therefore makes
correlation undefined. Undefined-correlation gradients are not training targets.

Zero weights ignore their feature values, while derivatives with respect to a
weight at zero are boundary conventions rather than a smooth contract. Omitting
`w`, or passing `w=None`, retains the existing unweighted implementation. Point-set/transport functions, probability histogram comparisons, and kernels
do not acquire generic feature weighting. Weighted
Minkowski retains its existing formula and continues to allow all-zero weights.

All-pairs uses weighted row statistics and matrix multiplication without an
`[N,M,D]` feature buffer. Paired uses direct row reductions; condensed uses row
suffix comparisons. Batch broadcasting, input/weight gradients, autocast and
compiled first gradients follow the [shared integration contract](pytorch-integration.md).
Generic `cdist`, `paired_distances`, `pdist` and dense neighbor masks accept these
weights. Legacy native `torch.ops` schemas retain their original signatures.

See the [example](../examples/probability_and_weights.py).

## Remaining SciPy-Style Vector Catalog

Optional `w` also applies to Euclidean, squared Euclidean, Manhattan/cityblock,
Minkowski, Chebyshev, Canberra, Bray-Curtis, Hamming and the boolean catalog
(Jaccard, Dice, Yule, Rogers-Tanimoto, Russell-Rao, Sokal-Michener,
Sokal-Sneath, Kulczynski1 and legacy Kulsinski). Aliases accept the same option.
Standardized Euclidean retains its variance `V` and Mahalanobis its `VI`; these
already express coordinate scaling rather than accepting a redundant `w`.
Haversine, SNR and point-set metrics have their own geometry and no added `w`.

| Family | Weighted formula or interpretation | Zero-total weights |
| --- | --- | --- |
| Lp, finite positive p | `(sum(w*abs(x-y)**p))**(1/p)` | Zero distances |
| Squared Euclidean | `sum(w*(x-y)**2)` | Zero distances |
| Chebyshev / infinite p | Maximum difference over features with `w>0`; magnitudes are irrelevant | Zero distances |
| Canberra | `sum(w*abs(x-y)/(abs(x)+abs(y)))`, with coordinate 0/0=0 | Zero distances |
| Bray-Curtis | `sum(w*abs(x-y))/sum(w*abs(x+y))` | Rejected |
| Hamming | Weighted fraction of unequal feature values | Rejected |
| Boolean catalog | Replace each contingency count by its weight sum | Rejected |

Canberra scales with a uniform scaling of weights. Ratio/contingency metrics
are invariant to that scaling. Bray-Curtis retains undefined 0/0 outputs when
its weighted denominator is zero. Boolean empty-support conventions follow the
[shared contracts](api-contracts.md): Jaccard/Yule return zero for all-false
vectors, Dice/Kulczynski1/Sokal-Sneath return NaN, and Russell-Rao/Kulsinski
return one. Weighted Kulsinski is the legacy formula with D replaced by the
weight sum; it remains distinct from Kulczynski1. Zero weights omit features.

Boolean inputs remain `bool`; their `w` is float32 or float64 on the same device.
Hamming accepts numeric values and compares equality without converting them
to bool. Its weights may also be float32/float64 independently of input dtype.
Outputs use the weights' floating dtype for these nondifferentiable-input
comparisons. Weight gradients are available where the weighted formula is
finite; boolean/equality inputs themselves have no derivative. Smooth numeric
metrics support both input and positive-weight gradients. Chebyshev support
changes and zero weights remain outside smooth-gradient guarantees.

Lp and Bray-Curtis reuse existing native primitives; boolean contingency
reductions use weighted matrix products, while Hamming/Canberra/squared-Euclidean
use budgeted sample/feature tiles with checkpointing. Condensed output evaluates
row blocks against their column suffix. No new native kernels or
legacy dispatcher schemas are introduced. This implementation prioritizes
formula and gradient parity; no throughput improvement is claimed. References:
[SciPy Canberra](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.canberra.html),
[Bray-Curtis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.braycurtis.html),
[weighted Jaccard](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html),
and [weighted Chebyshev](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.chebyshev.html).

## Native Execution and Deployment

Weighted formulas and first backward execute in shared C++ for all-pairs, paired
and condensed calls. Existing optimized Minkowski primitives are reused.
`torch.compile(..., fullgraph=True)` covers representative forward/backward calls;
dynamic export/save/load tests include changing sample counts, input/weight
gradients and Boolean/Hamming comparisons. Metric names, ranks and scalar options
remain structural. The [compiled example](../examples/compiled_weighted.py)
shows public Python use; the [native API](native-api.md#weighted-comparisons)
provides the standalone LibTorch entry point. TorchScript is optional; higher
derivatives of native replay are unsupported.
