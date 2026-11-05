<div class="home-intro">
<div class="eyebrow">TorchPairwise / documentation</div>
<h1>Comparisons and<br />Streamed Operators</h1>
<p class="lede">Distances, divergences, kernels and scores for tensor samples.
One comparison API, four pairing modes, shared native execution in Python and C++.</p>
<div class="badges"><span class="badge mode-all-pairs">87 comparison entries</span><span class="badge mode-condensed">13 kernels</span><span class="badge mode-paired">9 input families</span><span class="badge mode-single">CPU / CUDA</span></div>
</div>

<img class="hero-art" src="assets/hero.svg" width="1440" height="790" alt="Four comparison modes: all-pairs matrix, paired values, condensed upper triangle and one single-sample value. Streaming computes tiles and combines them into the requested result." />

<div class="feature-grid">
<a class="feature-card" href="guide/modes.html"><span class="feature-number">01 / OUTPUT SHAPES</span><strong>Four Comparison Modes</strong><p>All-pairs, paired, condensed and single. Leading batch axes broadcast.</p></a>
<a class="feature-card" href="guide/streaming.html"><span class="feature-number">02 / TILED EXECUTION</span><strong>Stream a Result</strong><p>Reductions, top-k, statistics and matrix products without retaining a full generated matrix.</p></a>
<a class="feature-card" href="metrics/index.html"><span class="feature-number">03 / METRIC REFERENCE</span><strong>Start with the Equation</strong><p>Find a comparison by formula, sample layout and supported modes.</p></a>
</div>

## A Comparison in Either Language

The top-bar Python / C++ switch changes paired examples throughout this guide.
Use the same metric name and mode in both languages.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
import torch
import torchpairwise as tp

X = torch.tensor([[0., 0.], [3., 4.]])  # [N=2,D=2]
Y = torch.tensor([[0., 4.], [3., 0.]])  # [M=2,D=2]

scores = tp.compare(X, Y, metric="sqeuclidean")  # [2,2]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
#include <torchpairwise.h>
namespace tp = torchpairwise::ops;

auto X = at::tensor({0.f, 0.f, 3.f, 4.f}).reshape({2, 2});
auto Y = at::tensor({0.f, 4.f, 3.f, 0.f}).reshape({2, 2});
auto scores = tp::compare(X, Y, "sqeuclidean");  // [2,2]
```

</div>
</div>

<div class="numeric-output">

Result: `[[16, 9], [9, 16]]`.

</div>

Start with the [quick start](guide/start.md), or browse the
[comparison catalog](metrics/index.md). Both numeric and nonnumeric samples use
ordinary dense tensors; structured distributions group their component tensors.

Explore the [example gallery](examples/index.md) for kernel ridge regression,
truncated kernel PCA, kernel LDA, binary SVM, landmark regression, PageRank and
kernel fitting with an intercept, graph diffusion and matrix-function actions. Each pairs a streamed
algorithm with a small dense correctness reference.

Streamed sources also support [pair masks](guide/statistics.md#pair-masks),
[entropic transport](guide/algebra.md#entropic-transport) over any lazy cost,
[Gaussian process prediction](guide/spectral.md#gaussian-process-prediction),
[user-defined metrics](metrics/custom-metrics.md), [encoded features](metrics/encoded-features.md)
with one or two encoders, and optional [execution backends](guide/streaming.md#execution-backends).

> This guide describes the source-tree API. Published 0.3 wheels expose an older
> interface. Build this checkout for the API documented here; see [installation](guide/install.md).

Python and native callers share built-in forward and first-backward algorithms.
Metric-specific domains, autocast, compilation and boundary derivatives are
part of each comparison's contract. A symmetric entry is not necessarily a
triangle-inequality metric, and a score need not be a positive-semidefinite kernel.
