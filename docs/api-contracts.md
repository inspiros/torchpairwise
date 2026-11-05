# Comparison Modes

`torchpairwise` distinguishes the comparison mode from batching:

| Mode | Inputs | Output | Status |
| --- | --- | --- | --- |
| All-pairs | `[N, D]`, `[M, D]` | `[N, M]` | Implemented |
| Paired | `[N, D]`, `[N, D]` | `[N]` | Implemented |
| Condensed | `[N, D]` | `[N * (N - 1) / 2]` | Implemented |

The table shows vector inputs; point-set metrics use the same output modes with
additional point axes described below.

`pdist(X, metric=...)` (also `condensed_distances`) stores each unordered pair of
distinct samples once, in order `(0,1), (0,2), ..., (1,2), ...`. It excludes the
diagonal. `condensed_kernels(X, kernel=...)` uses the same layout for kernels.
SNR and directed Hausdorff are asymmetric and are rejected in condensed mode.
KL divergence is also asymmetric and rejected there; symmetric KL uses the
Jeffreys sum and supports condensed output. See the
[histogram guide](histograms.md) for support/infinity/gradient conventions.
Optional feature weights on cosine and correlation follow the
[weighted-distance contract](weighted-distances.md).

Any leading batch dimensions broadcast, following PyTorch rules. For example,
`[2, 1, N, D]` and `[3, M, D]` produce `[2, 3, N, M]`. Features must match;
paired mode additionally requires exactly matching sample counts. It does not
broadcast a single row across multiple rows. Features must be non-empty; empty
sample or batch dimensions are supported.

```python
import torchpairwise as tp

matrix = tp.cdist(X, Y, metric="canberra")
paired = tp.paired_distances(X, Y, metric="canberra")
kernel_matrix = tp.pairwise_kernels(X, Y, kernel="rbf", gamma=0.5)
kernel_pairs = tp.paired_kernels(X, Y, kernel="rbf", gamma=0.5)
distance_vector = tp.pdist(X, metric="canberra")
kernel_vector = tp.condensed_kernels(X, kernel="rbf", gamma=0.5)
```

Existing named functions such as `canberra_distances` and `rbf_kernel` support the
same batch broadcasting. Omitting the second input in a named all-pairs function
compares the first input to itself. Paired functions require both inputs and
compute only the corresponding-row comparisons, without a square intermediate.
Gradients through broadcast batches reduce back to the original operand shapes.

Float32 and float64 are the primary correctness targets on CPU and CUDA.
Autocast promotes half/bfloat16 operands of floating distances and kernels to
float32 and computes with autocast disabled; float64 stays float64. This applies
to all three comparison modes. Gradients flow through the casts back to the
original input dtype, and CUDA GradScaler can be used normally. Tensor parameters
may remain float32 when inputs arrive in half/bfloat16 under autocast. Outside
autocast, reduced precision support remains operation-specific; use float32/64
or autocast for a common floating contract across the catalog. Direct float16
Canberra uses float32 term arithmetic and accumulation on CPU/CUDA, retaining
float16 outputs and gradients. Supported vector comparisons and streamed
consumers provide second derivatives through native tile replay: squared
Euclidean, Euclidean, Manhattan, Minkowski, cosine, RBF, Laplacian, linear,
polynomial, sigmoid, rational quadratic, and Matérn (nu 0.5, 1.5, or 2.5).
Supported consumers include sums, means, weighted means, matrix products,
selected values, logsumexp and softmax products, as well as MMD statistics for
supported MMD kernels. The higher-order graph retains tile intermediates and
can use O(N*M) memory; other operations may raise for higher derivatives.
Higher-order replay can differ in rounding from first-order kernels, and
derivatives at nonsmooth points follow the selected subgradient. Weighted-mean
replay uses direct numerator and mass sums, so extreme-range inputs may lose
accuracy or overflow compared with the anchored first-order path. Gradient checks apply
to smooth interior points; probability/domain boundaries can be nondifferentiable.
Inputs must share a device and, outside autocast, a dtype. Boolean distance families convert nonzero values to true; Hamming
instead compares the original values for equality. Boolean/Hamming distances do
not provide useful gradients.

See [AMP and compiler integration](pytorch-integration.md) for autocast policy,
compiled training, and transform limitations.

## Parameters and Numerical Conventions

- Feature weights `w` and variance `V` end in `[D]`; inverse covariance `VI` ends
  in `[D,D]`. Their leading batch dimensions broadcast with the input batches.
  They must share input dtype/device. Weights are finite and non-negative;
  variances are finite and strictly positive. VI should describe a valid inverse
  covariance for Mahalanobis distances.
- Standardized Euclidean and Mahalanobis require explicit V/VI for cross-input
  and paired comparisons. Self comparisons may infer detached statistics from
  each batch independently. Inferred variances must be positive; singular
  covariance matrices are rejected by inversion. Covariance inference requires
  at least two samples.
- Condensed V/VI inference uses the full input sample set, once per batch, with
  detached statistics. Supply explicit V/VI for empty or singleton sample sets.
  Other metrics return an empty condensed vector for zero or one sample.
- Minkowski accepts p >= 0, including infinity. Weighted Minkowski requires p > 0;
  at infinity, positive weights select coordinates and their magnitudes do not
  rescale the result. Weight gradients at infinity are zero while support is fixed.
- Default polynomial/sigmoid/RBF/laplacian gamma is `1 / D` for every batch rank.
- Rational quadratic and Matérn use a positive finite scalar `length_scale=1`,
  independent of feature count. Rational quadratic adds positive finite
  `alpha=1`; Matérn accepts `nu` in `{0.5,1.5,2.5}`, default 1.5. They support
  all-pairs, paired, and condensed kernels. These fixed Python scalars are not
  learnable tensor parameters. See [kernel formulas and conventions](kernels.md).
- Jensen-Shannon normalizes each nonnegative, positive-mass row. The logarithm
  base, when given, must exceed 1. Zero-mass rows are rejected.
- NaN-aware Euclidean rescales the squared difference over jointly observed
  features by D/observed_count. NaN denotes missingness; infinities are rejected.
  No-overlap pairs return NaN with a chosen zero gradient. `squared=True` omits
  the square root. Missing coordinates and zero distances receive zero gradients.
- `wasserstein_1d` compares finite nonnegative histogram masses after independent
  row normalization. Positive finite mass is required. A shared unit-spaced
  grid is the default; optional `[D]` `bin_positions` must be finite and strictly
  increasing with finite gaps and input dtype/device. Masses and grid positions
  support gradients. See [missing and ordered data](missing-and-ordered-data.md).
- Hellinger normalizes each finite nonnegative row to unit mass and uses the
  Euclidean distance between square roots divided by sqrt(2). Positive finite
  row mass is required. Histogram intersection sums minima of finite
  nonnegative bins without normalization; zero-mass rows are valid. Both support
  all three modes. See the [histogram guide](histograms.md) for gradients at zero
  and ties, and for the distinction between raw counts and distributions.
- Zero-distance Mahalanobis/Jensen-Shannon/weighted Lp/SNR and coincident Haversine
  comparisons use a zero subgradient at the singular point. Canberra coordinates
  with both values zero contribute zero to both value and gradient. Haversine's
  intermediate is clamped to its valid interval; antipodal singular gradients
  use the same zero convention.
- Cosine similarity of a zero vector is zero, with zero subgradient. Cross-input
  cosine distance is consequently 1 for such a pair; self-comparison diagonal
  distances are explicitly zero when Y is omitted or is the same tensor object. Correlation involving a constant row is NaN.
- Other undefined formulas retain their formula-specific NaN/infinity behavior,
  including zero denominators in Bray-Curtis or Kulczynski1. The obsolete
  Kulsinski formula remains available and differs from Kulczynski1.

For two all-false boolean vectors, Jaccard/Hamming/Yule/Rogers-Tanimoto/
Sokal-Michener return zero; Russell-Rao/Kulsinski return one; and
Dice/Kulczynski1/Sokal-Sneath return NaN. These follow the corresponding formulas
and reference conventions, rather than forcing a zero diagonal for every family.

## Point-Set Distances and Neighbors

These APIs take ordinary dense tensors, where each sample contains an unordered
collection of vectors. Sparse COO/CSR layouts are not supported. The
[point-set guide](pointsets.md) explains cloud/image layouts, lengths, and their
relationship to ordinary vector comparisons.

Directed Hausdorff, symmetric Hausdorff and Chamfer support arbitrary point axes
within each sample:

| Layout option | Inputs, with features last | Output |
| --- | --- | --- |
| `batched=False` | `[N,*points,C]`, `[M,*other_points,C]` | `[N,M]` |
| `batched=True` | `[B,N,*points,C]`, `[B2,M,*other_points,C]` | `[broadcast(B,B2),N,M]` |
| `batched=None` (default) | `[...,N,P,C]`, `[...,M,Q,C]` | `[...,N,M]` |

Exact uniform point-set Wasserstein, sliced Wasserstein and Sinkhorn divergence
use these same layouts and prefix lengths, with all three comparison modes.
Their generic names are `wasserstein`, `sliced_wasserstein`, and `sinkhorn`.
Sliced transport requires explicit shared projection directions. Sinkhorn is a
fixed-budget regularized divergence estimate, without a square root; exact
transport uses a native CPU-controlled solver even with CUDA inputs. Opaque
native calls support compile/export while preserving CPU coordination. See the
[transport guide](transport.md) for their distinct formulas and execution limits.

Explicit `batched` selects zero or one batch axis. Its omission preserves the
established point-cloud contract and leading batch broadcasting; it does not
infer spatial axes. For example, `[N,H,W,C]` requires `batched=False` to treat
H*W as points rather than interpret N as a batch axis. Each operand must include
the batch axis when `batched=True`; use a size-one batch to broadcast.

`feature_dim=-1` selects the feature/coordinate axis; set it to 2 for
`[B,N,C,H,W]` or 1 for `[N,C,H,W]`. The same selector applies to both operands
and must follow their batch/sample axes. All other axes after N are flattened
as points, preserving their original axis order. Point shapes/counts may differ
between operands, while C must match. Point and feature dimensions must be
non-empty; empty sample/batch axes are allowed. Reordering/flattening can copy
noncontiguous inputs, including conventional contiguous channel-first storage.

`directed_hausdorff_distances` returns `(distances, x1_indices, x2_indices)`, all with the
output shape above. Witness indices address each sample's flattened point set;
use `torch.unravel_index(indices, point_shape)` to recover spatial coordinates.
Distances use the Euclidean ground metric, without squaring. Deterministic
backward recomputes witnesses during native tile replay. Shuffled backward uses
the actual saved forward witnesses and does not advance the RNG. `cdist(..., metric="directed_hausdorff")` accepts the same layout
options and returns only distances. `hausdorff_distances` returns the maximum
of both directed distances. `chamfer_distances` returns the sum of the two
directional means of nearest-point distances; it defaults to Euclidean ground
distances, with `squared=True` selecting squared Euclidean. Neither function
reduces across samples/batches, and both return only the distance tensor.

String names are `directed_hausdorff`, `hausdorff`, and `chamfer`. All three work
through `cdist`, `paired_distances`, and both neighbor-mask functions. Paired
execution requires equal sample counts and returns `[...,N]`; it computes only
corresponding cloud comparisons. `pdist`/`condensed_distances` support the two
symmetric functions, storing each unordered cloud pair once. Layout and length
options apply to these point-set functions, not vector metrics or kernels.

Optional `x1_lengths`/`x2_lengths` are int64 tensors on the input device, ending
in `[N]`/`[M]`; leading axes broadcast within the existing input batch axes,
without introducing additional batch axes. Values must lie
between 1 and the padded point count. Each length selects a prefix of that
sample's flattened point axes; padded coordinates are ignored, including NaNs,
and receive zero gradient. Empty clouds are rejected, while zero samples and
empty batches remain supported. With Y omitted, `x2_lengths` defaults to
`x1_lengths`. With explicit Y, supply its lengths independently. Condensed calls
accept `x1_lengths` only and use it for both sides. Arbitrary point masks are
not yet supported. Shuffling/generators are available only for Hausdorff calls
without lengths; the length-aware reduction uses deterministic first-index
nearest-point tie selection. Distance gradients at zero use a zero subgradient.

`chunk_size=None` selects the native default execution. A positive integer tiles
all-pairs cloud comparisons to at most that many samples on each side, per
broadcast batch. Paired comparisons chunk that many corresponding cloud pairs;
condensed comparisons chunk row suffixes. Deterministic first backward retains
operands and replays point-level intermediates for one native tile at a time.
The final output is unchanged, at the cost of extra searches and launches.
Chunking cannot be combined with Hausdorff shuffle/generator options.

For tile size K, active nearest-point storage scales with `B*K*K*(P+Q)`
for all-pairs execution. Backward also uses feature-sized tile workspace.
Input/layout copies and the final `[B,N,M]` matrix still occupy memory; this
option is not a total-memory cap. Tile loops execute inside opaque native
operators; public compile/export checks cover changing sample and point counts.

```python
# X: [B,N,C,H,W], Y: [B,M,C,H2,W2]
distance, x_point, y_point = tp.directed_hausdorff_distances(
    X, Y, batched=True, feature_dim=2)
chamfer = tp.chamfer_distances(X, Y, batched=True, feature_dim=2)
paired = tp.paired_distances(X, X, metric="hausdorff", batched=True, feature_dim=2)
condensed = tp.pdist(X, metric="chamfer", batched=True, feature_dim=2)
```

Image feature-set Hausdorff compares C-dimensional descriptors, ignoring pixel
positions. Permuting the descriptors leaves the result unchanged. To compare
foreground geometry in masks, first extract foreground spatial coordinates;
mask values/classes are not coordinate features. See the
[library comparison](library-comparison.md) for these differing conventions.

Neighbor masks have the all-pairs output shape. `k_neighbors_mask` selects exactly
k eligible columns per row, breaking ties by column index; k=0 selects nothing.
k cannot exceed the eligible column count. Self entries are excluded by default
when Y is omitted, and included for explicit cross-input comparisons. Both mask
functions accept `exclude_self` to override that policy. Radius uses an inclusive
`distance <= epsilon` threshold and requires non-negative epsilon. Masks still
require dense all-pairs storage.

The public native family selectors support the same batch broadcasting and
comparison modes as the Python adapters. `builtin_comparisons` covers the full
vector/cloud distance and vector-kernel catalogs; `pointset_comparisons` also
exposes Chamfer and directed/symmetric Hausdorff with lengths and layout options.
`pdist`/`cdist` now share the broad native/Python role-aware catalog and expanded
optional parameters. Low-level family primitives retain their specialized signatures.
Use the [native API guide](native-api.md) and the typed family headers to choose
the appropriate entry point. CUDA operations use PyTorch's current stream.

Rational quadratic, Matérn, Hellinger and histogram intersection have named
native schemas and shared C++/ATen forward and first backward. Point-set
Chamfer and length-aware Hausdorff use a native nearest-point reduction that
keeps one distance and witness per source point in each active cloud tile,
rather than a dense inner `[P,Q]` matrix. Deterministic backward replays tile
witnesses from saved operands; shuffled Hausdorff retains its actual forward
witnesses. The final dense outer result and any directed witness outputs still
require their full output storage. Float32/float64 and reduced-precision inputs
under autocast are supported; these family APIs require float32/float64 outside
autocast.

## Additional Comparison Families

[Probability divergences](probability-divergences.md) distinguish normalized
histograms from raw positive Itakura-Saito spectra. Directed divergences have
all-pairs and paired outputs; symmetric comparisons also support condensed
output. Rényi uses the positive finite scalar `alpha`, with natural logarithms
and the KL limit at one. `cdist` and dense neighbor masks forward `alpha`.

[Additional vector distances](additional-vector-distances.md) use radians for
angular distance and require nonnegative inputs for Clark/Soergel/Ruzicka.
`continuous_jaccard` aliases `ruzicka`; Boolean `jaccard` keeps its meaning.

[Covariance geometry](covariance-distances.md) takes `[...,N,D,D]` SPD samples
through standalone functions for all three output modes. These structured
functions are separate from the vector dispatcher. They support first
derivatives, including repeated positive spectra; higher derivatives are
unsupported. No jitter or eigenvalue floor is inserted.

## Whole Images and Custom Functions

[Image quality scores](image-comparisons.md) take `[...,N,C,H,W]` images and
explicit `data_range`. PSNR, SSIM and multiscale SSIM have all-pairs, paired and
condensed score APIs outside the distance/kernel dispatcher. See their window,
pooling, channel-reduction and identity-gradient conventions before comparing
results with another library.

The experimental [custom-function adapter](custom-metrics.md) transforms a pure
PyTorch two-sample scalar function using `vmap`, optionally compiling its tiled
evaluation. Explicit `sample_ndim` describes vectors, clouds, or images. The
callable owns sample-shape compatibility and its gradient/numerical semantics;
condensed output requires an explicit symmetry assertion.

## Streaming Operators and Event Axes

The [operator guide](operators.md) is the entry point for scalar comparisons,
caller-owned matrices, scalar algebra, coordinate residuals and image-map
programs. A scalar metric consumes its input coordinates and represents
`[...,N,M]`; event sources retain `[...,N,M,*event_shape]`. A method's `dim`
indexes that current shape, not the input tensor. Batch axes are preserved.

Selected event-only reductions remain lazy; fully contracted events yield a
scalar expression. Selecting N or M returns a terminal tensor, including when
other selected event axes participate in one joint statistic. Subsequent tensor
operations do not fuse back into that consumer. Nine event reducers and the
21 common scalar consumers have distinct scopes; see the
[dimension/default table](operators.md#dimensions-and-event-reductions).

Chunk widths accept arbitrary built-in Python integers in `[1, 2**63 - 1]`,
not just presets. Widths larger than an axis are clipped to its remaining length;
tail tiles need no padding. Zero, negative values, booleans, floats, infinity,
NaN and larger integers are rejected. `None` remains available only where the
signature declares an optional chunk width; it selects that API's default policy.
Native typed schemas use signed 64-bit integers and reject nonpositive widths.

`chunk_size` bounds tile widths rather than total memory. Scalar intermediate
matrices need not be stored, but outputs, inputs, gradients and merge state still
occupy memory. Residual D tiling and image regional contraction have different
limits. Large image programs avoid full event tiles for a first
sum/mean/var/std/min/max/prod/all/any and for direct sample/event terminal
contractions with no prior event program. Backward recomputes one region at a
time rather than retaining each pairwise map.
Prepared image features remain. Image residuals use Y-X; coordinate residuals
use X-Y.
See the family guides for precision, nonfinite and first-derivative conventions.
