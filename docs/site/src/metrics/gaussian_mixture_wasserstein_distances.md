<div class="eyebrow">Gaussian mixtures</div>

# Gaussian-Mixture Wasserstein

<div class="metric-key">gaussian_mixture_wasserstein_distances</div>
<div class="badges"><span class="badge mode-all-pairs">all-pairs</span><span class="badge mode-paired">paired</span><span class="badge mode-condensed">condensed</span><span class="badge mode-single">single</span><span class="badge">distance</span><span class="badge">Mixtures</span></div>

\[
MW_2(p,q)=\sqrt{\min_{\pi\in\Pi(a,b)}\sum_{i,j}\pi_{ij}W_2^2(\mathcal N_i,\mathcal N_j)}
\]

PSD covariances. Restricts transport to Gaussian component couplings: not exact W2 between full mixture densities. At most 4096 positive-mass component pairs; CPU-controlled solve. squared=True omits the root.

## Shapes and Modes

| Input | Layout |
|---|---|
| One sample | `(means[K,D], covariances[K,D,D], weights[K])` |
| Collection X | `(means[...,N,K,D], covariances[...,N,K,D,D], weights[...,N,K])` |
| Collection Y | `(means[...,M,L,D], covariances[...,M,L,D,D], weights[...,M,L])` |

| Mode | Selector | Output |
|---|---|---|
| <span class="badge mode-all-pairs">all-pairs</span> | `compare(X,Y,metric=...)` | `[...,N,M]` |
| <span class="badge mode-paired">paired</span> | `compare(X,Y,metric=...,mode="paired")` | `[...,N]`; equal sample counts |
| <span class="badge mode-condensed">condensed</span> | `compare(X,metric=...,mode="condensed")` | `[...,N*(N-1)//2]`; strict upper triangle |
| <span class="badge mode-single">single</span> | `compare(x,y,metric=...,mode="single")` | `[...]`; leading batches only |

Leading batch dimensions broadcast. See [mode and shape conventions](../guide/modes.md).

## Entry Points

| Named Python entry | Function |
|---|---|
| All-pairs | `gaussian_mixture_wasserstein_distances` |
| Paired | `paired_gaussian_mixture_wasserstein_distances` |
| Condensed | `condensed_gaussian_mixture_wasserstein_distances` |

The Python and C++ `compare` selectors cover every mode shown above. `cdist` / `pairwise_distances` and `paired_distances` select this comparison; `pdist` / `condensed_distances` select its condensed mode. `single_comparison` is shorthand for single mode.

Catalog aliases: `gaussian_mixture_wasserstein`.

Accepted metric options: `squared`, `x1_lengths`, `x2_lengths`, `chunk_size`. Omitted options keep the metric defaults; unused options are rejected.

## Example

Choose Python or C++ in the top bar; the setting applies across this guide.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
import torch
import torchpairwise as tp

X = (torch.rand(2, 2, 2), torch.eye(2).repeat(2, 2, 1, 1), torch.ones(2, 2))
Y = (torch.rand(3, 2, 2), torch.eye(2).repeat(3, 2, 1, 1), torch.ones(3, 2))
# (means [N,K,D], covariances [N,K,D,D], weights [N,K]); K=D=2

metric = "gaussian_mixture_wasserstein_distances"
all_pairs = tp.compare(X, Y, metric=metric)  # [2,3]
paired_Y = tuple(t[:2] for t in Y)
paired = tp.compare(X, paired_Y, metric=metric, mode="paired")  # [2]
condensed = tp.compare(X, metric=metric, mode="condensed")      # [1]
x, y = tuple(t[0] for t in X), tuple(t[0] for t in Y)
single = tp.compare(x, y, metric=metric, mode="single")  # scalar
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
#include <torchpairwise.h>

namespace tp = torchpairwise::ops;
tp::ComparisonCollection X{at::rand({2, 2, 2}),
    at::eye(2).repeat({2, 2, 1, 1}), at::ones({2, 2})};
tp::ComparisonCollection Y{at::rand({3, 2, 2}),
    at::eye(2).repeat({3, 2, 1, 1}), at::ones({3, 2})};

const std::string metric = "gaussian_mixture_wasserstein_distances";
tp::ComparisonOptions options;
auto all_pairs = tp::compare(X, Y, metric, "all_pairs", options);  // [2,3]
tp::ComparisonCollection paired_Y{Y[0].narrow(0, 0, 2), Y[1].narrow(0, 0, 2), Y[2].narrow(0, 0, 2)};
auto paired = tp::compare(X, paired_Y, metric, "paired", options);  // [2]
// [1]
auto condensed = tp::compare(X, c10::nullopt, metric, "condensed", options);
tp::ComparisonCollection x{X[0][0], X[1][0], X[2][0]};
tp::ComparisonCollection y{Y[0][0], Y[1][0], Y[2][0]};
auto single = tp::compare(x, y, metric, "single", options);  // scalar
```

</div>
</div>

## Streamed Evaluation

`tp.lazy(x, y, name)` binds this distance for reductions, selection and products without retaining its full generated comparison matrix. Native source factories depend on the sample family; see [streamed operators](../guide/streaming.md) and [C++ use](../guide/cpp.md).

## Numerical Contract

Floating CPU/CUDA inputs use matching `float32` / `float64`, with lower precision promoted under autocast where supported. First derivatives follow the metric’s smooth and boundary conventions. See the [detailed conventions](https://github.com/inspiros/torchpairwise/blob/master/docs/gaussian-mixtures.md) and [precision and deployment](../guide/deployment.md).
