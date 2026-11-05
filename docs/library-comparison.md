# Choosing a Pairwise Library

External libraries reviewed against official documentation and source on
2026-10-29; local operator coverage updated on 2026-10-31. This compares
the API included in TorchPairwise 1.0. It is a
capability comparison, not a runtime ranking. Features describe the listed APIs;
users can compose additional operations in each framework.

torchpairwise's useful niche is a broad set of vector distances and kernels
with a shared PyTorch interface for all-pairs, paired, and condensed execution,
CPU/CUDA autograd, batch broadcasting, autocast, and compiler integration.
It supports padded point clouds, weighted Gaussian mixtures, continuous/token
sequences and aligned image/audio comparisons. The [operator interface](operators.md)
streams all named scalar families through products/forms, top-k and statistics,
with bounded scalar algebra and residual/image event programs. Remaining gaps
include arbitrary point masks, segmentation evaluation, sparse layouts, user
block producers and general lazy vector/matrix-valued formulas.

## Coverage and Tradeoffs

| Package / APIs | Comparisons and coverage | Where it is a better fit; gaps relative to torchpairwise |
| --- | --- | --- |
| **torchpairwise** | Broad vector/boolean/probability/kernel catalog; clouds, Gaussian mixtures, sequence alignment/token edits and aligned image/audio comparisons; all-pairs/paired/condensed where symmetric | Differentiable CPU/CUDA APIs where the comparison admits gradients; user-owned encoder feature reuse. Common operators stream all named scalar families into products, top-k and statistics. Residual/image event reductions remain lazy until a sample consumer. Sparse layouts, arbitrary point masks, mesh losses and general lazy formula programs remain gaps. |
| **PyTorch built-ins** | `cdist` for all-pairs Lp with leading batches; `F.pairwise_distance` for corresponding vectors; `F.pdist` for condensed Lp on a rank2 input | Best first choice for standard Lp operations: already installed, supported by PyTorch. Does not supply our broader distance/kernel catalog or batched condensed dispatcher. |
| **SciPy spatial.distance** | Broad `cdist` / condensed `pdist`, `squareform`, and directed Hausdorff between two point sets | Mature reference for scientific distances and condensed ordering. These APIs do not expose torchpairwise's broadcast batch and PyTorch training contracts. The standard weighted vector catalog is now available here; match normalization and boundary conventions when comparing results. |
| **scikit-learn metrics.pairwise** | Broad all-pairs distances/kernels; paired Euclidean, Manhattan, cosine and callables; chunked distances, argmin utilities, and sparse support for selected metrics | Strong fit for sklearn pipelines and sparse features. No built-in Hausdorff or generic condensed dispatcher in this module. Standard inputs are `[N,D]`, not broadcast batches. Selected operations also support experimental Array API GPU dispatch. |
| **PyTorch3D** | Differentiable geometry: paired batches of point clouds, Chamfer/Hausdorff reductions, KNN, lengths for padded clouds, mesh distances | Better coverage for geometry and variable-length clouds. Its cloud loss compares corresponding clouds, not every pair of N and M clouds. It is not a broad SciPy-style vector distance/kernel catalog. |
| **TorchMetrics** | Pairwise cosine, linear similarity, Euclidean, Manhattan, Minkowski on `[N,D]` / `[M,D]`; segmentation Hausdorff; extensive stateful evaluation metrics | A good fit for evaluation and distributed metric aggregation. Its pairwise module is smaller and does not provide our general batched/paired/condensed dispatchers. Segmentation Hausdorff consumes masks, not learned feature-point sets. |
| **pytorch-metric-learning** | Lp, cosine, dot-product, SNR distances connected to embedding losses and miners; corresponding-vector methods on distance classes | Better for complete metric-learning objectives/mining. Its standard distance inputs are rank2 embeddings, and normalization is enabled by default. `BatchedDistance` chunks queries via a callback; it does not mean independent tensor batch axes. |
| **KeOps** | Symbolic pairwise formulas with sum, logsumexp, min/max and K-min/argmin reductions, batching and autodiff | Better when the result is a reduction or neighbor query and the full matrix cannot fit. Formulas can express many distances/kernels; it is not the same ready-made named catalog. Returning every pair still requires quadratic output storage. |
| **POT (Python Optimal Transport)** | Exact, regularized and sliced transport between weighted distributions; backend-compatible operations including selected PyTorch paths | Better for transport solvers, couplings and related distribution problems. Backend compatibility does not imply every solver executes on CUDA: `emd2` uses a CPU solver. Our exact uniform point-set solver is also host-coordinated; our histogram API covers shared-grid 1D masses. |
| **GeomLoss** | Differentiable Sinkhorn divergences and geometric losses for sampled measures, with tensorized and online backends | Better for scalable differentiable regularized transport. Online evaluation reduces inner point-cost storage; it does not automatically produce our outer `[N,M]` cloud-distance matrix. Its regularized divergence is different from exact Wasserstein. |

Sources: [PyTorch cdist](https://docs.pytorch.org/docs/stable/generated/torch.cdist.html),
[pairwise_distance](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pairwise_distance.html),
[pdist](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pdist.html);
[SciPy distance catalog](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html);
[sklearn pairwise metrics](https://scikit-learn.org/stable/modules/metrics.html),
[paired dispatcher](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_distances.html),
[Array API support and supported operations](https://scikit-learn.org/stable/modules/array_api.html);
[PyTorch3D losses](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html),
[KNN](https://pytorch3d.readthedocs.io/en/latest/modules/ops.html#pytorch3d.ops.knn_points);
[TorchMetrics pairwise exports](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/pairwise/__init__.py),
[segmentation Hausdorff](https://lightning.ai/docs/torchmetrics/stable/segmentation/hausdorff_distance.html);
[metric-learning distances](https://kevinmusgrave.github.io/pytorch-metric-learning/distances/);
[KeOps](https://www.kernel-operations.io/keops/index.html).

## Hausdorff Semantics Must Match

For point sets X and Y, our directed distance is

\[
h(X,Y)=\max_{x\in X}\min_{y\in Y}\lVert x-y\rVert_2\]

The symmetric distance is `maximum(h(X,Y), h(Y,X))`, with the reverse all-pairs
matrix transposed to align samples. It is not the sum of the directions.
We return Euclidean distance and flattened witness indices, like SciPy's
point-coordinate interpretation extended to multiple samples and batches.
[SciPy reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html).

PyTorch3D uses `[B,P,C]` and supports per-cloud lengths. With
`point_reduction="max"`, `batch_reduction=None`, and `single_directional=True`,
it returns one directed result per corresponding cloud pair. With `norm=2`, its
KNN ground distances are **squared** Euclidean, so a value comparison against
ours must account for that square. Its bidirectional max reduction takes the
maximum of the two directions. The default Chamfer mean reduction and batch
mean describe a different output and should not be benchmarked as identical
work. See [Chamfer implementation](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/chamfer.html)
and [KNN contract](https://pytorch3d.readthedocs.io/en/latest/modules/ops.html#pytorch3d.ops.knn_points).

TorchMetrics Hausdorff measures segmentation surfaces with options for class
encoding, pixel spacing, direction, and ground metric. Its functional API returns
per-sample/per-class values; its metric class aggregates them. scikit-image
instead compares coordinates of all nonzero mask pixels. Neither should be
equated to interpreting every pixel's C channels as a descriptor. Our spatial
feature layout `[B,N,H,W,C]` means H*W unordered C-dimensional vectors. It does
not infer foreground, extract surfaces, or include pixel positions in the metric.
Sources: [TorchMetrics implementation](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/segmentation/hausdorff_distance.py),
[metric class](https://lightning.ai/docs/torchmetrics/stable/segmentation/hausdorff_distance.html),
[scikit-image](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.hausdorff_distance).

The development kernel catalog also includes isotropic rational quadratic and
Matérn (nu=0.5/1.5/2.5) in all three output modes. Their scalar parameterization
matches scikit-learn's Gaussian-process kernels, which live outside
`sklearn.metrics.pairwise`. See the [kernel guide](kernels.md) for supported
scales, formulas, and the Euclidean-versus-Manhattan distinction.

Hellinger distance and histogram intersection also support all three modes.
Hellinger normalizes positive-mass rows to distributions; intersection compares
raw bin masses using native ATen reductions. See the [histogram guide](histograms.md).

Normalized KL and Jeffreys divergence add directed and symmetric distribution
comparisons, respectively. Jeffreys supports all three modes; directed KL excludes
condensed output. Cosine and correlation now accept broadcast feature weights,
matching the corresponding SciPy formulas while preserving this package's
zero-norm cosine convention. See [feature weighting](weighted-distances.md).

## Priorities Suggested by the Comparison

Symmetric Hausdorff, Chamfer, paired/condensed point sets and per-sample prefix
lengths are implemented. Chamfer defaults to unsquared Euclidean ground
distance and sums directional means; `squared=True` aligns its ground distance
with PyTorch3D norm2. The next useful extensions are:

1. Point-set masks beyond contiguous prefixes, and profiling/tiling the new
   nearest-point kernels. Deterministic backward replays nearest witnesses from
   saved operands one cloud tile at a time; `chunk_size` controls outer cloud
   tiles. Further backward optimization remains useful.
2. Profile streaming reductions and assess measured native fusion. The full
   registries now replay selected pairs/tiles instead of retaining dense
   candidate graphs. Existing neighbor masks remain dense. The common operator now supports
   Gaussian/image/audio/sequence/token and signed-MMD scalar comparisons, while
   custom-callable block producers remain future scope. Cloud reductions retain
   their inner solver costs; image-map programs still retain full events per
   sample tile.
3. NaN-aware Euclidean and shared-grid 1D histogram Wasserstein are implemented
   across all three modes. The standard weighted SciPy-style vector catalog is also implemented.
   Broader kernels can follow without
   making every metric accept every parameter.
4. Profile condensed per-row launches and channel-first feature conversion.
   A feature-axis option improves usability but cannot promise a zero-copy
   native layout. Preserve fast paths for standard contiguous point arrays.

## Performance Comparison Rules

No claim that torchpairwise is faster than these packages follows from this
audit. Compare identical formulas, inputs, output mode, dtype, normalization,
direction, and reductions. Report CPU separately from GPU, forward separately
from forward-plus-backward, and peak memory alongside latency. Warm up compiled
code, exclude compilation/build time from steady-state timings but report it,
and synchronize CUDA timing. Include ordinary torch expressions and compiled
torch as baselines. Measure CPU tensor/NumPy transfer costs separately.
Align `F.pairwise_distance`'s epsilon, TorchMetrics' diagonal policy, and
pytorch-metric-learning's default embedding normalization before comparing values.

In particular, extracting a diagonal/triangle from a full matrix is an all-pairs
baseline for our paired/condensed APIs, not evidence of a faster implementation
of the same amount of work. Avoiding unnecessary comparisons reduces the amount
of work; report that separately from kernel throughput gains. KeOps reductions should be
compared with reduced/neighbor outputs, not charged for a dense output they do
not compute. Broad cross-library runtime benchmarks remain separate work.

See [dense outputs and reductions](reductions.md) for when a dense pairwise
matrix is the desired output and when it is only an avoidable intermediate.

For transport-specific sources, semantics and a proposed extension order, see
the [point-set transport guide](transport.md). Exact uniform W1/W2, sliced
Wasserstein and a fixed-budget Sinkhorn divergence estimate are implemented.
Exact transport uses a native CPU-controlled solver, including when captured by
compile/export. Sinkhorn retains dense point costs and differentiates its fixed
iteration budget inside native tile replay. No performance
ranking against POT or GeomLoss follows from this addition.

The development catalog also covers Bhattacharyya coefficient/distance,
Aitchison distance, empirical energy/MMD, and structured Gaussian W2. These
compare distinct distribution representations; see
[distribution comparison contracts](distribution-comparisons.md). Feature
weighting now extends the standard SciPy-style numeric and boolean vector
catalog, with zero/undefined conventions in [feature weights](weighted-distances.md).

## Image and Custom Comparisons

The [image quality family](image-comparisons.md) adds whole-image PSNR, SSIM and
MS-SSIM in all three pairwise modes. These require spatial alignment and fixed
image/data-range conventions; scores are not assumed metric distances or PSD
kernels. TorchMetrics and specialist image-quality libraries remain useful for
quality-assessment pipelines and pretrained perceptual comparisons. The library leaves learned models and weights to the caller and skips dedicated
LPIPS/DISTS wrappers. Gaussian W2 already supplies the FID formula for supplied
feature moments. See [learned representations](learned-comparisons.md) for
feature reuse and model-free extension scope.

The experimental [custom adapter](custom-metrics.md) uses PyTorch `vmap` and
optional `torch.compile` for a user-supplied scalar comparison. Callable support
is restricted by those transforms; it is not arbitrary Python/NumPy execution
and does not promise specialized-native throughput or total bounded training
memory. No mandatory external runtime dependency is added.

The general `make_feature_pairwise` wrapper separately composes a caller-owned
single encoder or query/reference towers with an explicit `metric=` or `head=`.
Built-in metric/kernel heads and `make_pairwise` heads support lazy streaming;
plain callable collection heads are eager-only. It reuses features within each
call and preserves model mode and gradient contexts, with no cross-call cache.
The [mixture](gaussian-mixtures.md), [sequence](sequence-comparisons.md),
[token](token-comparisons.md), and [audio](audio-comparisons.md) guides cover the
additional modalities and their explicit normalization/gradient contracts.
POT, tslearn, RapidFuzz and specialist audio libraries remain useful references;
these additions do not establish a cross-library performance ranking.
