<div class="eyebrow">Gaussians and covariances</div>

# Affine-Invariant Riemannian

<div class="metric-key">affine_invariant_distances</div>
<div class="badges"><span class="badge mode-all-pairs">all-pairs</span><span class="badge mode-paired">paired</span><span class="badge mode-condensed">condensed</span><span class="badge mode-single">single</span><span class="badge">distance</span><span class="badge">Covariances</span></div>

\[
d(A,B)=\|\log(A^{-1/2}BA^{-1/2})\|_F
\]

Symmetric positive-definite matrices. squared=True returns the squared distance. No jitter or eigenvalue clipping.

## Shapes and Modes

| Input | Layout |
|---|---|
| One sample | `[D,D]` |
| Collection X | `[...,N,D,D]` |
| Collection Y | `[...,M,D,D]` |

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
| All-pairs | `affine_invariant_distances` |
| Paired | `paired_affine_invariant_distances` |
| Condensed | `condensed_affine_invariant_distances` |

The Python and C++ `compare` selectors cover every mode shown above. `cdist` / `pairwise_distances` and `paired_distances` select this comparison; `pdist` / `condensed_distances` select its condensed mode. `single_comparison` is shorthand for single mode.

Catalog aliases: `affine_invariant`.

Accepted metric options: `squared`. Omitted options keep the metric defaults; unused options are rejected.

## Example

Choose Python or C++ in the top bar; the setting applies across this guide.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
import torch
import torchpairwise as tp

X = torch.eye(2).repeat(2, 1, 1)      # [N=2,D=2,D=2]
Y = 2 * torch.eye(2).repeat(3, 1, 1)  # [M=3,D=2,D=2]

metric = "affine_invariant_distances"
all_pairs = tp.compare(X, Y, metric=metric)  # [2,3]
paired_Y = Y[:2]
paired = tp.compare(X, paired_Y, metric=metric, mode="paired")  # [2]
condensed = tp.compare(X, metric=metric, mode="condensed")      # [1]
x, y = X[0], Y[0]
single = tp.compare(x, y, metric=metric, mode="single")  # scalar
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
#include <torchpairwise.h>

namespace tp = torchpairwise::ops;
auto X = at::eye(2).repeat({2, 1, 1});
auto Y = 2 * at::eye(2).repeat({3, 1, 1});

const std::string metric = "affine_invariant_distances";
tp::ComparisonOptions options;
auto all_pairs = tp::compare(X, Y, metric, "all_pairs", options);  // [2,3]
auto paired_Y = Y.narrow(0, 0, 2);
auto paired = tp::compare(X, paired_Y, metric, "paired", options);  // [2]
// [1]
auto condensed = tp::compare(X, c10::nullopt, metric, "condensed", options);
auto x = X[0], y = Y[0];
auto single = tp::compare(x, y, metric, "single", options);  // scalar
```

</div>
</div>

## Streamed Evaluation

`tp.lazy(x, y, name)` binds this distance for reductions, selection and products without retaining its full generated comparison matrix. Native source factories depend on the sample family; see [streamed operators](../guide/streaming.md) and [C++ use](../guide/cpp.md).

## Numerical Contract

Floating CPU/CUDA inputs use matching `float32` / `float64`, with lower precision promoted under autocast where supported. First derivatives follow the metric’s smooth and boundary conventions. See the [detailed conventions](https://github.com/inspiros/torchpairwise/blob/master/docs/covariance-distances.md) and [precision and deployment](../guide/deployment.md).
