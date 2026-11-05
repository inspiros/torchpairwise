<div class="eyebrow">Gaussians and covariances</div>

# Gaussian Wasserstein / Bures

<div class="metric-key">gaussian_wasserstein_distances</div>
<div class="badges"><span class="badge mode-all-pairs">all-pairs</span><span class="badge mode-paired">paired</span><span class="badge mode-condensed">condensed</span><span class="badge mode-single">single</span><span class="badge">distance</span><span class="badge">Gaussians</span></div>

\[
W_2^2=\|\mu-\nu\|_2^2+\operatorname{tr}\!\left(\Sigma+\Lambda-2(\Sigma^{1/2}\Lambda\Sigma^{1/2})^{1/2}\right)
\]

Means and finite symmetric positive-semidefinite covariances. Default returns W2; squared=True returns the expression shown. Singular covariance boundaries use a finite derivative continuation.

## Shapes and Modes

| Input | Layout |
|---|---|
| One sample | `(mean[D], covariance[D,D])` |
| Collection X | `(means[...,N,D], covariances[...,N,D,D])` |
| Collection Y | `(means[...,M,D], covariances[...,M,D,D])` |

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
| All-pairs | `gaussian_wasserstein_distances` |
| Paired | `paired_gaussian_wasserstein_distances` |
| Condensed | `condensed_gaussian_wasserstein_distances` |

The Python and C++ `compare` selectors cover every mode shown above. `cdist` / `pairwise_distances` and `paired_distances` select this comparison; `pdist` / `condensed_distances` select its condensed mode. `single_comparison` is shorthand for single mode.

Catalog aliases: `gaussian_wasserstein`.

Accepted metric options: `squared`. Omitted options keep the metric defaults; unused options are rejected.

## Example

Choose Python or C++ in the top bar; the setting applies across this guide.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
import torch
import torchpairwise as tp

X = (torch.rand(2, 2), torch.eye(2).repeat(2, 1, 1))
Y = (torch.rand(3, 2), 2 * torch.eye(2).repeat(3, 1, 1))
# (means [N,D], covariances [N,D,D])

metric = "gaussian_wasserstein_distances"
all_pairs = tp.compare(X, Y, metric=metric)  # [2,3]
paired_Y = tuple(t[:2] for t in Y)
paired = tp.compare(X, paired_Y, metric=metric, mode="paired")  # [2]
condensed = tp.compare(X, metric=metric, mode="condensed")  # [1]
x, y = tuple(t[0] for t in X), tuple(t[0] for t in Y)
single = tp.compare(x, y, metric=metric, mode="single")  # scalar
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
#include <torchpairwise.h>

namespace tp = torchpairwise::ops;
tp::ComparisonCollection X{at::rand({2, 2}), at::eye(2).repeat({2, 1, 1})};
tp::ComparisonCollection Y{at::rand({3, 2}), 2 * at::eye(2).repeat({3, 1, 1})};

const std::string metric = "gaussian_wasserstein_distances";
tp::ComparisonOptions options;
auto all_pairs = tp::compare(X, Y, metric, "all_pairs", options);  // [2,3]
tp::ComparisonCollection paired_Y{Y[0].narrow(0, 0, 2), Y[1].narrow(0, 0, 2)};
auto paired = tp::compare(X, paired_Y, metric, "paired", options);  // [2]
auto condensed = tp::compare(X, c10::nullopt, metric, "condensed", options);  // [1]
tp::ComparisonCollection x{X[0][0], X[1][0]};
tp::ComparisonCollection y{Y[0][0], Y[1][0]};
auto single = tp::compare(x, y, metric, "single", options);  // scalar
```

</div>
</div>

## Streamed Evaluation

`tp.lazy(x, y, name)` binds this distance for reductions, selection and products without retaining its full generated comparison matrix. Native source factories depend on the sample family; see [streamed operators](../guide/streaming.md) and [C++ use](../guide/cpp.md).

## Numerical Contract

Floating CPU/CUDA inputs use matching `float32` / `float64`, with lower precision promoted under autocast where supported. First derivatives follow the metric’s smooth and boundary conventions. See the [detailed conventions](https://github.com/inspiros/torchpairwise/blob/master/docs/distribution-comparisons.md) and [precision and deployment](../guide/deployment.md).
