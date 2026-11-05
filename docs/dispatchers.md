# Selecting a Comparison

`torchpairwise.compare` is the general selector for distances, divergences,
kernels, scores, costs and statistics. The name is deliberately neutral: a
similarity or signed MMD statistic is not a distance. Its `mode` controls pairing.
Use canonical full names to make formulas clear across families.

```python
import torchpairwise as tp

matrix = tp.compare(X, Y, metric='sqeuclidean_distances')
aligned = tp.compare(X, Y, metric='sqeuclidean_distances', mode='paired')
unordered = tp.compare(X, metric='sqeuclidean_distances', mode='condensed')
value = tp.single_comparison(x, y, metric='sqeuclidean_distances')
# Equivalent: tp.compare(x, y, metric='sqeuclidean_distances', mode='single')
```

## Four Pairing Modes

| Mode | Input collections / samples | Output |
|---|---|---|
| `all_pairs` (default) | X with N samples, optional Y with M samples | `[...,N,M]` |
| `paired` | X and Y with equal N | `[...,N]` |
| `condensed` | One collection X, supported symmetric comparison | `[...,N*(N-1)//2]`, pairs i<j in row order |
| `single` | One sample x and one sample y per leading batch | `[...]`; scalar for unbatched samples |

Single mode **omits** the collection sample axis. The chosen metric defines the
sample dimensions; leading dimensions broadcast as batches. There is no inferred
N/M axis and no automatic mode detection. Thus `[B,D]` means B batched vectors
in single mode, and a collection of B vectors in all-pairs mode.

| Family | One sample | Batched examples |
|---|---|---|
| Vector / histogram | `[D]` | `[B,D]` |
| Point cloud / empirical distribution | `[P,D]` | `[B,P,D]` |
| Image | `[C,H,W]` | `[B,C,H,W]` |
| Audio | `[C,T]` | `[B,C,T]` |
| Sequence | `[T,D]` | `[B,T,D]` |
| Tokens | `[T]` | `[B,T]` |
| Covariance | `[D,D]` | `[B,D,D]` |
| Gaussian | `(mean[D], covariance[D,D])` | `(means[B,D], covariances[B,D,D])` |
| Gaussian mixture | `(means[K,D], covariances[K,D,D], weights[K])` | Add matching leading batches to each tensor |

Single point sets use canonical `[P,D]` samples; flatten spatial point grids
first and omit `batched`/`feature_dim`. Optional lengths omit the collection
sample axis too: use scalar lengths for an unbatched point set/sequence, or
`[B]` lengths for B batched samples. Feature weights and V/VI keep their normal
parameter shapes. Standardized Euclidean/Mahalanobis require explicit V/VI for
single comparison; one pair does not supply a reference population estimate.

```python
# Each result is one scalar, or one value per leading broadcast batch.
w2 = tp.single_comparison((mu, covariance), (other_mu, other_covariance),
                          metric='gaussian_wasserstein_distances')
chamfer = tp.single_comparison(points, other_points, metric='chamfer_distances')
ssim = tp.single_comparison(image, reference, metric='ssim_scores', data_range=1.)
```

Single mode adds a singleton sample axis, invokes the existing native paired
implementation, then removes that axis. It does not compute a full batch-by-batch
matrix or add a new numerical implementation.

## Selector Coverage and Its Justification

One native catalog supplies public names, aliases, roles, sample ranks, keyword
parameters and supported modes to Python and C++. `list_comparisons()` returns
that metadata. Filter with `family='image'` or `role='kernel'`, for example.
Roles are `distance` (including divergences and transport costs), `kernel`,
`score` and `statistic`; they do not assert mathematical metric axioms or PSD.

| Selector | Canonical catalog entries | Scope rule |
|---|---|---|
| `cdist`, `pairwise_distances` | 68 | Every distance/divergence/cost across modalities |
| `paired_distances` | Same 68 | Same role, corresponding samples |
| `pdist`, `condensed_distances` | 59 | Distance role with supported condensed mode |
| `pairwise_kernels`, `paired_kernels`, `condensed_kernels` | Same 13 | Every catalog kernel, including Gaussian-mixture overlap |
| `compare` | 87; all paired/single, 76 condensed | Every role, filtered by supported mode |
| `tp.lazy` | Same 87 scalar sources | Same formulas, consumed without retaining their full matrix |
| `tp.lazy` with a kernel metric | Same 13 kernels | Kernel specialization of the scalar operator |

Counts exclude spelling aliases but include mathematical synonyms. Image SSIM
and PSNR are scores, so use `compare` rather than a distance selector for them.
Audio log-spectral distance and Gaussian Wasserstein are distance entries and
are available through distance selectors. Different sample modalities are not
an exclusion criterion.

Full canonical names and catalog aliases work in both languages. Shared short
`cosine` selects cosine distance in distance/general selection, and cosine
similarity in kernel selection. Use `cosine_distances` or `cosine_similarity`
when moving across those roles. MMD's historical kernel-option conventions are
separate; see [API naming](api-design.md#formula-ambiguities-to-avoid).

Named functions keep their unprefixed all-pairs meaning. Existing `paired_` and
`condensed_` named functions remain conveniences; not every comparison exposes
all these individual Python names. Use the generic selectors for uniform access;
there is no separate `single_*` wrapper family for every metric.

`cdist` and `pdist` default to Minkowski p=2; `pairwise_distances`,
`paired_distances`, `compare` and `single_comparison` default to Euclidean.
`condensed_distances` retains the pdist default. Specify the metric when passing
p or transferring code between these interfaces. Omitted Y means self comparison
for all-pairs; paired/single require Y and condensed accepts only X.

All generic selectors return one Tensor. Named directed Hausdorff retains its
historical `(distances, x_witness, y_witness)` result; use it when witnesses are
needed. This is an intentional simplification of generic `compare`'s earlier
Hausdorff return. See [migration](migration.md#unified-selectors-and-single-samples).

## Python and Native Routing

Python `cdist`/`pdist` call the native operators with those names. Tensor inputs
use the default overload; Gaussian/mixture tuples use the `.collections`
Tensor-list overload. Their roles, aliases, parameters, defaults and numerical
execution agree. `compare` and the other mode/role frontends use the native
`compare` operator. The common selector validates options and calls the existing
vector, point-set, structured, image, audio, sequence or token family operator.
Built-in forward/backward execution has no Python numerical callback dependency.

Standalone C++ includes `comparison_selectors.h` (also in `torchpairwise.h`).
Use `ComparisonOptions` for explicitly supplied parameters and `compare`,
`single_comparison`, `cdist` or `pdist` for selection; structured collections use
`ComparisonCollection`. Existing positional C++ cdist/pdist calls forward through
the same broad operators rather than a separate legacy whitelist. Rebuild native
consumers and re-export/recompile graphs after this schema expansion.
See [native API](native-api.md).

A **comparison selector** chooses a formula/mode from its name. The **PyTorch
operator dispatcher** separately routes an operator to numerical, autograd or
metadata implementations. An exposed schema name does not imply an identically
named Python function or standalone C++ symbol.

## Streaming and Convenience Helpers

`tp.lazy(X,Y,metric=...)` uses the same scalar catalog as dense compare.
Kernel metrics add coordinate geometry, which
requires vector inputs. Residual and image-map factories bind event-valued
sources. See the [operator guide](operators.md).

Use lazy selection and reduction methods for uniform catalog coverage; [the naming guide](api-design.md)
explains those contracts. Arbitrary callbacks retain separate deployment needs.
