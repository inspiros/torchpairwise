<div class="eyebrow">Custom Comparisons</div>

# Encoded Feature Reuse

For comparisons that factor into an encoder and a head, compute features before
forming sample pairs. **Example composition:**

\[
f(x,y)=g(e(x),e(y))
\]

Here \(e\) is any caller-supplied encoder and \(g\) is a compatible comparison head;
neither is restricted to a particular model or distance formula. This changes
the encoder workload from two encodings per pair to one per input sample. It
does not separate a model that jointly processes both samples, and it does not
supply pretrained models, weights or preprocessing. Python orchestration examples
remain visible in either language view; the native deployment example switches
between Python and C++.

## Encode Before Pairing

```python
import torch
import torchpairwise as tp

encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3*8*8, 16))
images = torch.randn(3, 3, 8, 8)      # [N,C,H,W]
references = torch.randn(5, 3, 8, 8)  # [M,C,H,W]
comparison = tp.make_feature_pairwise(encoder,
    "sqeuclidean", sample_ndim=3)
values = comparison(images, references)                       # [3,5]
single = comparison(images[0], references[0], mode="single")  # scalar
```

| Stage | Input | Output |
|---|---|---|
| Raw collection | `[...,N,*sample]` | Encoder receives `[original_samples,*sample]` |
| Encoder | `[original_samples,*sample]` | `[original_samples,*feature]` per leaf |
| Restored features | Encoded leaves | `[...,N,*feature]` |
| Single mode | `[...,*sample]` | `[...,*feature]`, compared without an N/M axis |
| Scalar all-pairs head | Two feature collections | `[...,N,M]` |

Each original collection is encoded before comparison batch broadcasting. A
`[1,N,*sample]` input still encodes N samples against a `[B,M,*sample]` input.
With a single encoder, omitted Y encodes once; explicit `comparison(X,X)` reuses the same feature tree
on both sides. Distinct tensor objects are encoded independently.

The encoder can return nonempty ordinary tuple/list/dict trees of tensor leaves.
Every leaf needs the encoded count as its first axis. A head owns how those
leaves become comparison inputs; a generic nested feature tree is not silently
flattened into one vector.

## Prepare Once, Use Several Consumers

```python
prepared = comparison.prepare(images, references)
A = tp.lazy(prepared, chunk_size=2)  # logical [3,5]
row_mean = A.mean(-1)                # [3]
nearest = A.topk(2, largest=False)   # values/indices [3,2]
V = torch.randn(5, 4)
product = A.matmul(V)  # [3,4]
dense = prepared()     # [3,5], only when requested
```

Neither binding A nor its consumers reruns the encoder. Each consumer evaluates
the comparison head's tiles again, reusing prepared features. Preparation is an
explicit snapshot of computed features. Prepare again when the model, inputs or
preprocessing should change those features. Normal comparison calls encode anew.

For several different heads, extract features directly:

```python
features = comparison.encode(images)            # [3,16]
other_features = comparison.encode(references)  # [5,16]
cosine = tp.lazy(features, other_features, metric="cosine_distances")
rbf = tp.lazy(features, other_features, metric="rbf", gamma=0.2)
```

`tp.lazy(images, references, comparison)` encodes once and binds the head.
`make_feature_pairwise` takes exactly one head: a string `metric` (positional or
keyword) or `head=`; callable positional heads are rejected. Metric options such
as `gamma` are allowed only with `metric=`. Construction `mode=` sets the
default and call-time `mode=` overrides it. Built-in heads default to all-pairs;
CustomComparison heads use their own default. Plain callable heads use `.mode`
if present and otherwise receive no implicit mode.

| Head kind | Eager | Lazy | KeOps / fused CUDA eligibility |
|---|---|---|---|
| Built-in `metric="name"` | Yes | Yes | For supported metric/consumer combinations |
| `head=tp.make_pairwise(...)` | Yes | Yes, native tile replay | KeOps only for a `keops=True` broadcast head; no fused CUDA |
| Plain callable or `nn.Module` collection `head=` | Yes | No; use `tp.make_pairwise` for a pairwise function | No |

Built-in heads bind the canonical distance/kernel operator, including `"rbf"`.
Lazy bindings require all-pairs mode and a single feature tensor for metric
heads. A feature tree needs an explicit head. Plain collection heads have no
dense streaming fallback; wrap a pairwise function with `tp.make_pairwise` to
stream it. A single-mode preparation cannot become a streaming matrix.

## Two-Tower Retrieval

```python
# query_encoder and reference_encoder are caller-supplied modules.
retrieval = tp.make_feature_pairwise(
    (query_encoder, reference_encoder), "sqeuclidean", sample_ndim=3)
values, indices = tp.lazy(images, references, retrieval).topk(2, largest=False)
queries = retrieval.encode(images, tower="query")
references_encoded = retrieval.encode(references, tower="reference")
```

The first input uses the query encoder and the second uses the reference
encoder. Omitting the second input runs both towers on the first input;
`retrieval(images, images)` also runs both. These features can be asymmetric, so
condensed mode is rejected. Two-tower `encode` requires `tower="query"` or
`tower="reference"`; a single encoder requires `tower=None`. Both towers obey
the same chunking and model-state rules.

## Training and Memory

Features remain attached to the encoder's autograd graph. A streamed head's
backward accumulates gradients into feature tensors, which then flow through the
encoder. Combine losses into one backward call, or follow PyTorch's normal
`retain_graph` rules when reusing one prepared graph across backward calls.

`encode_chunk_size` limits original samples per encoder invocation. Comparison
`chunk_size` instead limits sample tiles in the head. Encoder chunks can change
BatchNorm, cross-sample attention and other batch-dependent behavior. Neither
control bounds total feature storage or retained encoder activations. Head replay
does not recompute the encoder; checkpoint it separately if the application needs
that tradeoff.

The wrapper preserves model train/eval state, gradients, device, dtype and the
caller's autocast context. Choose these in the application. Empty original
collections are rejected before invoking arbitrary user code, since an encoder's
feature structure cannot be inferred safely.

## Native Deployment

The Python encoder wrapper is orchestration. In LibTorch, run the caller's loaded
encoder with its chosen runtime and pass the resulting feature tensors to
`compare`, `pairwise_expression` or `custom_expression`. These use the same native
head/reduction code. Python callbacks or encoder factories do not become native
merely because the head supports C++.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# The caller supplies a loaded encoder.
features = encoder(images)   # [N,F]
other = encoder(references)  # [M,F]
A = tp.lazy(features, other, metric="sqeuclidean")
mean = A.mean(-1)  # [N]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// encoder is the caller's loaded, typed LibTorch module.
auto features = encoder->forward(images);   // [N,F]
auto other = encoder->forward(references);  // [M,F]
auto A = torchpairwise::ops::pairwise_expression(
    features, other, "sqeuclidean");
auto mean = A.reduce("mean", -1);  // [N]
```

</div>
</div>

See the runnable [feature example](https://github.com/inspiros/torchpairwise/blob/master/examples/feature_comparisons.py).
Feature reuse alone does not reproduce LPIPS/DISTS calibration or a benchmark's
FID preprocessing. [Learned-comparison conventions](https://github.com/inspiros/torchpairwise/blob/master/docs/learned-comparisons.md)
explain feature-distribution scores and caller-owned models.
