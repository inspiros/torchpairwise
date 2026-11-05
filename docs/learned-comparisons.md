# Learned Representations and Comparison Scores

The library operates on supplied tensors and scalar PyTorch comparison
functions. It does not provide learned encoders, pretrained checkpoints, weight
loaders, or model-specific image preprocessing. Users own their models, loaded
weights, device placement, training/evaluation state, and gradient policy.

`make_feature_pairwise` composes the user's encoder with an explicit metric or collection
head and reuses each original collection's features within the call.
[Custom comparisons](custom-metrics.md) adapt compatible two-sample scalar
functions, including functions that capture trainable parameters. The two
factories can be composed when a comparison head is naturally scalar.

## Where Execution Can Save Work

PyTorch model code already calls native CPU/CUDA operators. Wrapping that code
in another Python function does not itself make those operators faster.
Vectorization or compilation may help a particular function, but performance
needs an equivalent-work benchmark.

When a score factorizes as

\[
s(x,y)=g(f(x),f(y)),
\]

compute the encoder f once per input sample and reuse its outputs. For all-pairs
comparisons this requires N+M sample encodings, usually in batches, rather than
repeating two encodings for every one of the N*M pairs. Self comparisons with a single encoder can
reuse the same feature collection on both sides. The comparison stage
still evaluates N*M scores if all-pairs output is requested. Paired mode already
needs only N comparisons, so the opportunity for reuse differs.

The caller can perform this separation directly, then use vector distances,
kernels, point-set comparisons, or the custom adapter on the supplied features.
A two-image model passed to `make_pairwise` is not automatically separated into
an encoder and comparison head. It can repeat the complete model per pair.
`vmap` does not memoize image features across pairs. Models that jointly process
the two inputs may not admit independent feature caching at all.

## Feature-Reuse Wrapper

```python
compare = tp.make_feature_pairwise(
    encoder, "euclidean", sample_ndim=1, encode_chunk_size=None,
)
result = compare(x1, x2, **head_options)
```

Inputs are dense strided tensors shaped `[*batch, N, *sample]`, where
`sample_ndim` (a positive integer) counts trailing sample axes. For images
`[N,C,H,W]`, use `sample_ndim=3`; a mono waveform collection `[N,1,T]` uses
`sample_ndim=2`. The wrapper flattens each original collection's batch and
sample-count axes into `[samples,*sample]` for the encoder. It preserves the
order of samples and may copy noncontiguous inputs during reshaping.

The encoder returns either a tensor or an ordinary nonempty tuple/list/dict
of feature trees. Every leaf must be a dense strided tensor with leading
axis equal to the encoded minibatch count; remaining feature axes are
unrestricted, including per-sample scalar output `[samples]`. Containers cannot
contain metadata, `None`, or empty subcontainers. Named tuples and custom
container types are unsupported. Each leaf is restored to
`[*original_batch,N,*feature]`, preserving ordinary container kinds and dict
keys. With chunks, structures, dict keys, and each leaf's trailing shape,
dtype, and device must agree across chunks. Output trees from distinct input
collections need only satisfy the user's comparison contract.

Each collection is encoded **before** the comparison performs any batch
broadcasting. A `[1,N,*sample]` collection therefore encodes only N samples,
even when compared against `[B,M,*sample]`. The wrapper does not infer or
validate comparison output shapes, symmetry, matching sample counts, feature
compatibility, or batch broadcasting. Those belong to the selected head, which
receives complete restored feature collections:

- For a single encoder, omitted `x2` supplies one feature collection to the
  head; built-in heads compare that collection with itself.
- `compare(x, y)` calls `head(features_x, features_y)`.
- With a single encoder, `compare(x, x)` encodes once and supplies the very same feature tree as both
  comparison arguments. It still calls the two-argument comparison form.
- Distinct tensor objects are encoded separately, even if they share storage
  or contain equal values.

Choose exactly one string `metric` (positional or keyword) or `head=`.
Metric keywords such as `gamma` belong to the factory only with `metric=`.
A callable positional second argument is rejected; write `head=my_head`.
Additional arguments pass to custom collection heads. The encoder receives only
its flat tensor; bind encoder-specific options with a closure or partial.

```python
import torchpairwise as tp

all_pairs = tp.make_feature_pairwise(encoder, "euclidean", sample_ndim=3)
paired = tp.make_feature_pairwise(
    encoder, "euclidean", mode="paired", sample_ndim=3)
condensed = tp.make_feature_pairwise(
    encoder, "euclidean", mode="condensed", sample_ndim=3)
custom = tp.make_feature_pairwise(encoder, head=my_head, sample_ndim=3)
```

Construction `mode=` supplies the default; call-time `mode=` overrides it.
Metric heads default to `all_pairs`; CustomComparison heads default to their
own mode. Plain callable heads use their `.mode` attribute if present, otherwise
receive no implicit mode. A CustomComparison's sample axes describe features;
`sample_ndim` here describes the original input samples. Feature trees require
an explicit head that reads their leaves.

### Two-Tower Retrieval

```python
retrieval = tp.make_feature_pairwise(
    (query_encoder, reference_encoder), "sqeuclidean", sample_ndim=3)
values, indices = tp.lazy(query_images, reference_images, retrieval).topk(
    3, largest=False)
query_features = retrieval.encode(query_images, tower="query")
reference_features = retrieval.encode(reference_images, tower="reference")
```

The query tower encodes the first input and the reference tower the second.
With omitted `x2`, both towers independently encode the first input; explicit
`retrieval(X, X)` also runs both towers. The result need not be symmetric, so
condensed mode is rejected. `encode` requires `tower="query"` or
`tower="reference"` for two towers; a single encoder requires `tower=None`.
Both towers follow the same chunking, gradient and model-mode rules.

There is **no cache across calls**. Changes to inputs, model parameters, model
mode, and preprocessing take effect on the next invocation. The wrapper does
not detach features, disable gradients, call `eval()`, move inputs or models,
convert dtypes, load weights, or compile user code. Caller-selected autograd,
autocast, and train/eval contexts govern both stages. Gradients can reach inputs,
encoder parameters, and tensors captured by the comparison head. Integer token
inputs are accepted when the encoder supports them. Device/dtype compatibility
is left to the encoder and comparison.

`encode_chunk_size` is `None` by default, giving one encoder invocation per
distinct original collection. A positive integer splits the flat sample axis
into ordered minibatches and concatenates each feature leaf. It does not tile
pair comparisons: use the comparison's own controls for that. Chunks can change
results for batch-dependent encoders, including BatchNorm in training mode,
cross-sample attention, batch statistics, and stochastic computations. Flattening
original batch axes also means those axes are not separate encoder batches.
Use this wrapper only when these encoder batching semantics are intended.
Chunking limits each encoder invocation's sample count; it does not bound the
total feature storage, autograd graph, or saved activations from all chunks.
Failures do not roll back user model state changes from completed encoder calls.

Empty original collections, including zero-size batch axes, raise `ValueError`
**before calling the encoder or comparison**. Arbitrary user encoders need not
support empty minibatches, and probing one would introduce side effects to
infer feature structure. A nonempty singleton can still produce an empty
condensed comparison according to its comparison's contract.

See [the runnable example](../examples/feature_comparisons.py). Performance
claims require equivalent-work measurements; feature extraction and pair
comparison remain ordinary PyTorch operations.

## Explicit Preparation and Streamed Heads

The returned FeatureComparison object also provides three reusable paths:

```python
compare = tp.make_feature_pairwise(
    encoder, "sqeuclidean", sample_ndim=3)
features_x = compare.encode(images_x)  # [...,N,*feature]
prepared = compare.prepare(images_x, images_y)
A = tp.lazy(prepared, chunk_size=16)   # logical [...,N,M]
mean = A.mean(-1)
nearest = A.topk(3, largest=False)
dense = prepared()                     # only materializes when requested
```

`encode` returns an ordinary feature tree for arbitrary other heads. `prepare`
explicitly computes both feature trees now; calls and operator consumers on the
returned binding do not rerun the encoder. `tp.lazy(X,Y,compare)` is a shortcut
that encodes once and binds the head. Subsequent consumers recompute head tiles,
while reusing that encoding. Normal `compare(X,Y)` calls still encode anew.
Prepared features are an explicit snapshot, not an automatically invalidated
cache. Prepare again when model/input/preprocessing changes should take effect.

Head choice explicitly determines streaming support:

| Head kind | Eager | Lazy | KeOps / fused CUDA eligibility |
|---|---|---|---|
| Built-in `metric="name"` | Yes | Yes | When the metric and consumer support the backend |
| `head=tp.make_pairwise(...)` | Yes | Yes, native tile replay | No |
| Plain callable or `nn.Module` collection `head=` | Yes | No; wrap a pairwise function with `tp.make_pairwise` | No |

Built-in heads bind the canonical metric/kernel operator directly, including
kernel names such as `"rbf"` with `gamma=...`. Lazy binding requires all-pairs
mode and one feature tensor. CustomComparison heads use their own streaming
operator. Plain callable collection heads remain eager-only: there is no dense
fallback or callable introspection; wrap the callable with `tp.make_pairwise`
for lazy evaluation.

Single mode can be requested with `mode="single"` when the comparison accepts
it, or selected by a CustomComparison factory default. Raw inputs then omit the
N/M axis; an image is `[C,H,W]` and batched single images are `[B,C,H,W]`.
The encoder still receives a sample minibatch. Restored feature leaves omit the
collection axis before calling the single comparison. `encode(single=True)`
and `prepare(...,mode="single")` expose this explicitly.

Encoder gradients remain attached to prepared features. Combine consumer losses
into one backward pass, or use PyTorch's normal retain_graph rules for repeated
backward through one preparation. Head replay bounds pairwise workspace but does
not replay the encoder or bound encoder activation/feature storage. Use encoder
checkpointing in the application if needed. No automatic detach or model-mode
changes are performed. Native C++ applications can run their loaded encoder and
pass its features directly to pairwise_expression or custom_expression.

## LPIPS and DISTS

These scores include specific feature normalization, layer aggregation and
learned calibration. A generic distance between pooled image embeddings does
not reproduce them. A faithful implementation needs their supplied feature
extractor and comparison head, including the loaded calibration parameters.

Use the user's chosen implementation directly. A pure, transform-compatible
scalar call may be adapted with `make_pairwise`, subject to its batching,
output-dtype and compilation restrictions. The adapter does not load weights,
change model mode, disable gradients, or claim to accelerate the model. Feature
reuse requires an explicit decomposition provided by the caller; it is not
inferred from an arbitrary model. Dedicated LPIPS/DISTS APIs are skipped for now.

## FID and Feature-Distribution Comparisons

FID compares two collections of extracted features, not two individual images.
Its Gaussian comparison is squared Gaussian Wasserstein distance, which the
library already supplies. Given feature tensors `features_x` and `features_y`
of shape `[samples, features]`, on the same device, the following composition
uses float64 moments and the sample covariance with denominator n-1:

```python
import torchpairwise as tp

# Extract these once using the user's loaded model and chosen preprocessing.
# Choose model mode and gradient tracking explicitly in the calling application.
def feature_moments(features):
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("Expected at least two feature vectors in [N,D] form")
    features = features.double()
    mean = features.mean(0, keepdim=True)
    centered = features - mean
    covariance = centered.mT @ centered / (features.shape[0] - 1)
    return mean, covariance.unsqueeze(0)

mean_x, cov_x = feature_moments(features_x)
mean_y, cov_y = feature_moments(features_y)
score = tp.gaussian_wasserstein_distances(
    mean_x, cov_x, mean_y, cov_y, squared=True,
)[0, 0]
```

This computes the Gaussian feature-distribution score for the supplied moments.
Reproducing a particular FID benchmark also requires matching its encoder,
weights, image preprocessing, feature layer, covariance estimator and numerical
protocol. Scores from arbitrary embeddings are not interchangeable with that
benchmark. The library makes no inference about those choices and does not
provide a named image-to-FID model pipeline.

KID uses an unbiased polynomial-kernel MMD estimator. Use
`mmd_statistics(real_features[None], generated_features[None],
kernel="polynomial", degree=3, gamma=1/C, coef0=1.)` for that signed statistic
on supplied features. The caller supplies matching features and implements any
benchmark subset sampling/aggregation. Negative estimates are retained.
`mmd_distances` remains the biased rooted distance by default; selecting a
polynomial kernel alone does not make that distance an unbiased KID estimate.
See [distribution comparisons](distribution-comparisons.md) and the
[example](../examples/mmd_statistics.py).
