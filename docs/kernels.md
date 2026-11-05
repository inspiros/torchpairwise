# Rational Quadratic and MatéRn Kernels

These isotropic kernels compare vector samples, with the same shapes as RBF:
`[N,D]` and `[M,D]` produce `[N,M]`; leading batch dimensions broadcast.
They have unit amplitude: finite coincident inputs give one. Multiply the result
by a variance/amplitude externally when needed.

`length_scale` is a strictly positive finite real scalar, defaulting to 1.
It measures distance in input units, independent of the feature count. Larger
values make similarity fall more slowly. Parameters are fixed scalars;
learnable tensor parameters and per-feature length scales are not supported.
Input gradients, float32/float64, autocast, and compiled forward/backward are
supported in all three comparison modes. Higher derivatives are not guaranteed.

## Rational Quadratic

With Euclidean distance \(d=\lVert x-y\rVert_2\), scale \(\ell\), and positive
finite \(\alpha\):

\[
k(x,y)=\left(1+\frac{d^2}{2\alpha\ell^2}\right)^{-\alpha}\]

`alpha` defaults to 1. The kernel combines RBF behavior at different scales;
smaller alpha gives heavier tails. As alpha grows, it approaches RBF with
`gamma=1/(2*length_scale**2)`. Evaluation uses `log1p` to retain small changes
and the large-alpha limit.
Division is ordered to avoid forming `2*alpha`, which can overflow for a
representable finite alpha even when the kernel has a finite RBF limit.

```python
K = tp.rational_quadratic_kernel(X, Y, length_scale=2., alpha=.5)
# Equivalent string dispatch:
K = tp.pairwise_kernels(X, Y, kernel="rational_quadratic",
                        length_scale=2., alpha=.5)
```

The normalization and scalar parameters match scikit-learn's
[`RationalQuadratic`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RationalQuadratic.html).

## MatéRn

`nu` selects smoothness and accepts only 0.5, 1.5, or 2.5 (default: 1.5).
For \(r=d/\ell\):

| nu | Kernel |
| --- | --- |
| 0.5 | \(\exp(-r)\) |
| 1.5 | \((1+\sqrt{3}r)\exp(-\sqrt{3}r)\) |
| 2.5 | \((1+\sqrt{5}r+5r^2/3)\exp(-\sqrt{5}r)\) |

Larger nu produces smoother covariance functions. These common closed forms
avoid a special-function dependency. Arbitrary nu and infinity are rejected;
use `rbf_kernel` for the infinite-smoothness limit. Unlike
[`Matern`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html),
this API currently supports only scalar length scales and the three nu values.

Matérn with nu=0.5 uses **Euclidean** distance. The existing `laplacian_kernel`
uses **Manhattan** distance, so they differ for multidimensional vectors.
Coincident inputs use a zero subgradient, including the cusp at nu=0.5.
All-pairs Matérn uses direct Euclidean distance accumulation to avoid the
cancellation of close large-offset vectors in a squared-norm matrix product.
Far tails that round to zero are evaluated with a zero gradient, bounding the
polynomial argument before multiplication to avoid `infinity * zero` in values
or backward. As with other floating-point operations, inputs and scalar options
must be considered together with the computation dtype; extremely small scales
or parameters outside its representable range can still lose precision or overflow.

```python
K = tp.matern_kernel(X, Y, length_scale=2., nu=2.5)
paired = tp.paired_kernels(X, Y, kernel="matern", length_scale=2., nu=2.5)
condensed = tp.condensed_kernels(X, kernel="matern", length_scale=2., nu=2.5)
```

## Execution and Output Modes

Paired mode requires equal sample counts and computes only corresponding rows;
it does not allocate a square matrix. Condensed mode computes strict row suffixes
without a square intermediate, in the usual `pdist` order. Empty samples/batches
follow the [shared contracts](api-contracts.md). Native first backward retains
operands and replays comparison tiles; dense all-pairs output still requires
N*M storage.

Forward and first backward use shared C++/ATen algorithms. Named
`torch.ops.torchpairwise` schemas and the `vector_comparisons` and
`builtin_comparisons` selectors expose the same modes to Python and standalone
LibTorch. Rebuild the matching native extension when updating these operators;
no separate radial-kernel dependency is required. See the
[native API guide](native-api.md) and [Python example](../examples/radial_kernels.py).
