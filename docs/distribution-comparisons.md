# Comparing Distributions and Compositions

These comparisons answer different questions. Bhattacharyya and Aitchison take
one histogram/composition vector per sample. Energy and MMD take a collection
of support points per sample. Gaussian W2 takes explicit means and covariance
matrices. Inputs are ordinary dense tensors; no sparse storage or distribution
object is required.

## Bhattacharyya Overlap and Distance

For normalized histogram masses p and q, the coefficient is
`BC(p,q)=sum(sqrt(p*q))` and the distance is `-log(BC(p,q))`.
`bhattacharyya_kernel(X,Y)` returns BC, while `bhattacharyya_distances(X,Y)`
returns its negative natural logarithm. Rows are finite/nonnegative, with
positive finite total mass. Independent row rescaling does not change results.

The coefficient is zero for disjoint supports and one for equal distributions.
The distance is infinite for disjoint supports and zero for equal distributions.
It is not a triangle-inequality metric. Hellinger instead returns
`sqrt(1-BC)`; the distinct transforms remain separately named.

Both support all-pairs, paired and condensed outputs. Use generic
`kernel="bhattacharyya"` in kernel dispatchers and `metric="bhattacharyya"` in
distance dispatchers. Raw zero bins have chosen zero derivatives, including
through normalization. Disjoint infinite-distance pairs receive zero whole-pair
gradients; these are boundary conventions, not smooth derivatives. No epsilon
is added. Float32 uses float64 intermediates and returns float32. Near identity,
the distance evaluates `1-BC` via squared root-probability differences to avoid
subtracting almost equal numbers. Extreme float64 overlaps may still underflow.

## Aitchison Distance for Compositions

`aitchison_distances(X,Y)` computes the Euclidean distance between centered
log-ratios:

\[
\operatorname{clr}(x)_i=\log x_i-\frac1D\sum_j\log x_j,\qquad
d_A(x,y)=\|\operatorname{clr}(x)-\operatorname{clr}(y)\|_2\]

Each row must have strictly positive finite entries. Zeros are rejected; no
pseudocount is inferred. Raw log centering avoids computing a normalization
sum, so even finite entries whose total overflows can be compared. Independent
positive rescaling of rows leaves the mathematical distance unchanged. D=1
gives zero. Float32 uses float64 intermediates; the final dtype is preserved.
All three modes are supported through `metric="aitchison"`. Coincident rows
use zero norm derivatives. Extremely small entries can have large log gradients.

## Energy Distance and Kernel MMD

`energy_distances` and `mmd_distances` take point-set inputs `[N,P,C]` and
`[M,Q,C]`, with the existing batch/layout selectors and prefix lengths.
Valid points have uniform probability mass. Define empirical averages that
include within-cloud diagonal terms:

\[
E^2(X,Y)=2\,\mathbb E\|X-Y\|_2-\mathbb E\|X-X'\|_2-\mathbb E\|Y-Y'\|_2,
\]

\[
\operatorname{MMD}^2(X,Y)=\mathbb E k(X,X')+\mathbb E k(Y,Y')
                         -2\mathbb E k(X,Y)\]

The default returns the square root. `squared=True` returns the expression
above; these are biased empirical V statistics, not unbiased U estimates.
Negative floating-point roundoff is clamped to zero, with a chosen zero root
derivative at zero. Energy uses Euclidean point distances, so its squared
statistic scales linearly with coordinate scale rather than quadratically.

MMD selects the native pairwise kernel catalog by short name or full function
name. It supports polynomial, RBF, linear, cosine, Manhattan Laplacian,
rational quadratic, Matérn, histogram intersection, chi-squared, additive
chi-squared and Bhattacharyya kernels. Sigmoid is rejected because it is not
generally positive semidefinite. The additive chi-squared kernel is conditionally
positive semidefinite: its separable terms cancel in the MMD statistic.

| Selector | Parameters / convention |
| --- | --- |
| `rbf` / `rbf_kernel` | `gamma`; omitted gamma is `1/(2*length_scale**2)` |
| `polynomial` / `polynomial_kernel` | `degree=3`, `gamma=1/C`, `coef0=1`; non-negative integer degree and finite non-negative gamma/coef0 |
| `linear` / `linear_kernel` | Euclidean distance between cloud means |
| `cosine` / `cosine_similarity` | Existing normalized-dot-product and zero-vector conventions |
| `laplacian` / `euclidean_laplacian` | Historical Euclidean `exp(-norm(x-y)/length_scale)` |
| `laplacian_kernel` | Registry Manhattan kernel, `gamma=1/C` by default |
| `rational_quadratic` / `rational_quadratic_kernel` | Positive `length_scale`, positive `alpha=1` |
| `matern` / `matern_kernel` | Positive `length_scale`, `nu=0.5`, `1.5` (default), or `2.5` |
| `histogram_intersection` / `histogram_intersection_kernel` | Non-negative coordinates |
| `chi2` / `chi2_kernel` | Non-negative coordinates, `gamma=1` by default |
| `additive_chi2` / `additive_chi2_kernel` | Non-negative coordinates |
| `bhattacharyya` / `bhattacharyya_kernel` | Non-negative coordinates, positive finite mass per point; each point is normalized |

`length_scale=1.0` remains positive and finite for all MMD selectors, with no
effect on kernels that do not use it. Other kernel options are rejected when
unused. RBF and Euclidean Laplacian preserve their original definitions.
In particular, **`laplacian` and `laplacian_kernel` intentionally differ** to
preserve the historical MMD API while exposing the registry kernel explicitly.

Linear and finite-degree polynomial MMD compare selected moments and do not
distinguish every pair of distributions. RBF and Euclidean Laplacian are
characteristic on Euclidean supports. Kernel selection preserves each kernel's
input domain and boundary derivative conventions.

Nonlinear expectations use bounded point tiles and reuse the native kernel
implementations; linear MMD uses cloud means directly. All-pairs still returns
the outer cloud-distance matrix. For a signed statistic with a general
similarity, compose three kernel sums as in the
[matrix-free kernel example](../examples/kernel_methods.py). That example
retains roundoff without applying a root or a clamp.

```python
import torch
import torchpairwise as tp

X, Y = torch.rand(3, 20, 2), torch.rand(4, 15, 2)
matrix = tp.energy_distances(X, Y)
paired = tp.paired_distances(X, Y[:3], metric="mmd", kernel="rbf", length_scale=.5)
condensed = tp.pdist(X, metric="energy", squared=True)
polynomial = tp.mmd_distances(
    X, Y, kernel="polynomial", degree=3, gamma=.5, coef0=1., squared=True)
```

Both support all three modes, finite valid coordinates, NaN padding with prefix
lengths, and first input gradients. Point comparisons use fixed-size reduction
tiles and checkpoint recomputation in training. This avoids retaining full
point-cost matrices, but graph metadata still grows with the number of tiles.
All-pairs cloud output is dense; optional `chunk_size` additionally tiles the
outer comparisons. Self expectations are reused within each evaluator tile.
Linear MMD computes means directly. Subtracting nearly equal expectations
retains floating-point limitations, particularly for rooted values near zero.

## Signed MMD Estimates and KID

`mmd_statistics`, `paired_mmd_statistics` and `condensed_mmd_statistics` return
squared empirical MMD estimates directly. They use the same cloud layout,
kernel options, prefix lengths, batching and feature-axis selectors as
`mmd_distances`. `estimator="unbiased"` is the default:

\[
\widehat{\operatorname{MMD}}_u^2(X,Y)=
\frac{\sum_{i\ne j}k(x_i,x_j)}{P(P-1)}+
\frac{\sum_{i\ne j}k(y_i,y_j)}{Q(Q-1)}-
\frac{2\sum_{i,j}k(x_i,y_j)}{PQ}\]

Each cloud needs at least two valid points. Only within-cloud diagonals are
removed; all cross terms remain. For independent i.i.d. samples from the two
populations this is an unbiased estimate of population MMD squared. The output
can be negative: **no clamp or square root is applied**. Comparing a finite
sample with itself therefore need not yield zero. `estimator="biased"` includes
within-cloud diagonals and permits singleton clouds; it also returns the direct
statistic, with no root or clamp. Linear biased statistics use the stable
squared difference between means.

```python
# Features supplied by user code: [points,features]. No model is bundled.
real_features, generated_features = torch.randn(100, 64), torch.randn(120, 64)
statistic = tp.mmd_statistics(
    real_features[None], generated_features[None],
    kernel="polynomial", degree=3, gamma=1/64, coef0=1.,
).squeeze()  # scalar signed estimate

# Collections of clouds: output [N,M], [N], or [N*(N-1)//2].
clouds = torch.randn(3, 20, 64)
paired = tp.paired_mmd_statistics(clouds, clouds + .1, kernel="rbf")
condensed = tp.compare(clouds, metric="mmd_statistic", mode="condensed")
```

KID uses unbiased polynomial MMD on image features, typically with the cubic
kernel `(dot(x,y)/C+1)**3`. Matching a published KID score also requires the
specified encoder, weights, preprocessing, and subset/aggregation protocol.
This API provides the statistic; user code owns those choices. See the
[runnable feature example](../examples/mmd_statistics.py).

The estimate has dedicated typed C++/dispatcher entry points and full-graph
compile/export metadata. It shares native point tiling, self-expectation caches
and first-backward replay with MMD distances. Higher derivatives are unsupported.
`compare` dispatches it by `mmd_statistics` or `mmd_statistic`; signed statistics
are separate from the distance registry used by `cdist`, `pdist` and reductions.
The output modes remain available through their named functions or `compare`.

## Gaussian Bures-Wasserstein Distance

For Gaussian distributions with means m,n and PSD covariances A,B, squared W2 is

\[
\|m-n\|_2^2+\operatorname{tr}(A)+\operatorname{tr}(B)
-2\operatorname{tr}\big((A^{1/2}BA^{1/2})^{1/2}\big)\]

It compares the supplied Gaussians, not arbitrary distributions sharing those
moments. Covariance matrices must be supplied explicitly; the API does not
estimate covariances from points. Three structured entry points keep the extra
inputs clear:

```python
means = torch.rand(3, 2)                        # [N,D]
covariances = torch.eye(2).expand(3, 2, 2)       # [N,D,D]
matrix = tp.gaussian_wasserstein_distances(means, covariances)
paired = tp.paired_gaussian_wasserstein_distances(
    means, covariances, means + 1, covariances * 2)
condensed = tp.condensed_gaussian_wasserstein_distances(means, covariances)
```

All-pairs outputs `[...,N,M]`, paired `[...,N]`, and condensed
`[...,N*(N-1)//2]`. Batch axes broadcast, including between one input's means
and covariances. Sample and feature dimensions must match within each input.
Supply both second inputs or omit both. `squared=False` returns rooted W2;
`squared=True` omits the root. These functions are separate from `cdist`/`pdist`
and the generic metric registry.

Finite means and finite symmetric PSD covariances are required. Relative
roundoff tolerance is `64*D*finfo(dtype).eps` times each matrix's scale;
material asymmetry/negative eigenvalues are rejected. Accepted matrices are
symmetrized and tiny negative eigenvalues clipped to zero, without adding a
diagonal jitter. An equivalent orthogonal Procrustes residual evaluates the
covariance term without subtracting almost equal traces.

First derivatives on positive-definite covariances use a matrix square-root
Sylvester equation and a Procrustes envelope derivative. Repeated positive
eigenvalues remain supported. At singular PSD boundaries no general finite
derivative exists: zero/zero Sylvester denominators choose zero, and the SVD
chooses an optimal Procrustes continuation. These conventions are not a claim
of smooth differentiability at singular covariance matrices. Higher derivatives
are explicitly unsupported. Coincident distances use zero root derivatives.

The covariance root is computed once per sample. All-pairs uses 16-by-16 sample
tiles; paired computes only aligned comparisons; condensed computes row suffixes.
Tiling bounds decomposition temporaries, but backward retains per-pair matrix
residuals/rotations, so its state still scales as `N*M*D*D`. Matrix work per pair
is cubic in D. No speed comparison with transport solvers is implied.

## Supplied Image-Feature Distributions

The Gaussian W2 `squared=True` operation also supplies the Gaussian comparison
used by FID when callers provide matching feature means and covariances. The
encoder and preprocessing stay in user code. This does not make arbitrary
feature-distribution scores equivalent to a named FID benchmark. Biased MMD
is not the unbiased KID estimator. See
[learned representations](learned-comparisons.md) for
a feature-moment example and the model ownership policy.

## Integration

These APIs support CPU/CUDA float32/float64 and autocast promotion to float32,
with first gradients and compiled execution as described in the
[integration contract](pytorch-integration.md). Gaussian inputs must share their
original dtype/device; Gaussian FakeTensor/meta execution is not advertised.
No new native kernels or runtime dependencies are introduced. Numerical
overflow/underflow limits remain. See [the example](../examples/distributions.py).
