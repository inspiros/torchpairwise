<div class="eyebrow">Point sets and transport</div>

# Sliced Wasserstein

<div class="metric-key">sliced_wasserstein_distances</div>
<div class="badges"><span class="badge mode-all-pairs">all-pairs</span><span class="badge mode-paired">paired</span><span class="badge mode-condensed">condensed</span><span class="badge mode-single">single</span><span class="badge">distance</span><span class="badge">Point sets</span></div>

\[
\widehat{SW}_p(X,Y)=\left[\frac1L\sum_{\ell=1}^L\int_0^1|F^{-1}_{X\theta_\ell}(t)-F^{-1}_{Y\theta_\ell}(t)|^p\,dt\right]^{1/p}
\]

Explicit shared unit projections [D,L] are required; p=1 or 2. Handles unequal point counts by exact projected quantile integration. This is sliced, not full-dimensional Wp.

## Shapes and Modes

| Input | Layout |
|---|---|
| One sample | `[P,D]` |
| Collection X | `[...,N,P,D]` |
| Collection Y | `[...,M,Q,D]` |

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
| All-pairs | `sliced_wasserstein_distances` |

The Python and C++ `compare` selectors cover every mode shown above. `cdist` / `pairwise_distances` and `paired_distances` select this comparison; `pdist` / `condensed_distances` select its condensed mode. `single_comparison` is shorthand for single mode.

Catalog aliases: `sliced_wasserstein`.

Accepted metric options: `p`, `projections`, `x1_lengths`, `x2_lengths`, `chunk_size`, `batched`, `feature_dim`. Omitted options keep the metric defaults; unused options are rejected.

## Example

Choose Python or C++ in the top bar; the setting applies across this guide.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
import torch
import torchpairwise as tp

X = torch.rand(2, 3, 2)  # [N=2,P=3,D=2]
Y = torch.rand(3, 4, 2)  # [M=3,Q=4,D=2]

metric = "sliced_wasserstein_distances"
options = dict(projections=torch.eye(2))
all_pairs = tp.compare(X, Y, metric=metric, **options)  # [2,3]
paired_Y = Y[:2]
paired = tp.compare(X, paired_Y, metric=metric, mode="paired", **options)  # [2]
condensed = tp.compare(X, metric=metric, mode="condensed", **options)      # [1]
x, y = X[0], Y[0]
single = tp.compare(x, y, metric=metric, mode="single", **options)  # scalar
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
#include <torchpairwise.h>

namespace tp = torchpairwise::ops;
auto X = at::rand({2, 3, 2});  // [N=2,P=3,D=2]
auto Y = at::rand({3, 4, 2});  // [M=3,Q=4,D=2]

const std::string metric = "sliced_wasserstein_distances";
tp::ComparisonOptions options;
options.projections = at::eye(2);
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

Floating CPU/CUDA inputs use matching `float32` / `float64`, with lower precision promoted under autocast where supported. First derivatives follow the metric’s smooth and boundary conventions. See the [detailed conventions](https://github.com/inspiros/torchpairwise/blob/master/docs/transport.md) and [precision and deployment](../guide/deployment.md).

## Related Implementations

- [POT sliced_wasserstein_distance](https://pythonot.github.io/gen_modules/ot.sliced.html#ot.sliced.sliced_wasserstein_distance).

Related APIs can differ in reduction, normalization, units or boundary behavior; match these before comparing values.
