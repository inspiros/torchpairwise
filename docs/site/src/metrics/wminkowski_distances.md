<div class="eyebrow">Vector geometry</div>

# Weighted Minkowski

<div class="metric-key">wminkowski_distances</div>
<div class="badges"><span class="badge mode-all-pairs">all-pairs</span><span class="badge mode-paired">paired</span><span class="badge mode-condensed">condensed</span><span class="badge mode-single">single</span><span class="badge">distance</span><span class="badge">Vectors</span></div>

\[
d_{p,w}(x,y)=\left(\sum_i w_i|x_i-y_i|^p\right)^{1/p}
\]

Provide nonnegative w ending in [D]; p must be positive. Weights multiply the powered differences, rather than being raised to p.

## Shapes and Modes

| Input | Layout |
|---|---|
| One sample | `[D]` |
| Collection X | `[...,N,D]` |
| Collection Y | `[...,M,D]` |

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
| All-pairs | `wminkowski_distances` |

The Python and C++ `compare` selectors cover every mode shown above. `cdist` / `pairwise_distances` and `paired_distances` select this comparison; `pdist` / `condensed_distances` select its condensed mode. `single_comparison` is shorthand for single mode.

Catalog aliases: `wminkowski`.

Accepted metric options: `p`, `w`. Omitted options keep the metric defaults; unused options are rejected.

## Example

Choose Python or C++ in the top bar; the setting applies across this guide.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
import torch
import torchpairwise as tp

X = torch.rand(2, 2) + 0.2  # [N=2,D=2]
Y = torch.rand(3, 2) + 0.2  # [M=3,D=2]

metric = "wminkowski_distances"
options = dict(w=torch.tensor([1., 2.]))
all_pairs = tp.compare(X, Y, metric=metric, **options)  # [2,3]
paired_Y = Y[:2]
paired = tp.compare(X, paired_Y, metric=metric, mode="paired", **options)  # [2]
condensed = tp.compare(X, metric=metric, mode="condensed", **options)  # [1]
x, y = X[0], Y[0]
single = tp.compare(x, y, metric=metric, mode="single", **options)  # scalar
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
#include <torchpairwise.h>

namespace tp = torchpairwise::ops;
auto X = at::rand({2, 2}) + 0.2;  // [N=2,D=2]
auto Y = at::rand({3, 2}) + 0.2;  // [M=3,D=2]

const std::string metric = "wminkowski_distances";
tp::ComparisonOptions options;
options.w = at::tensor({1.f, 2.f});
auto all_pairs = tp::compare(X, Y, metric, "all_pairs", options);  // [2,3]
auto paired_Y = Y.narrow(0, 0, 2);
auto paired = tp::compare(X, paired_Y, metric, "paired", options);  // [2]
auto condensed = tp::compare(X, c10::nullopt, metric, "condensed", options);  // [1]
auto x = X[0], y = Y[0];
auto single = tp::compare(x, y, metric, "single", options);  // scalar
```

</div>
</div>

## Streamed Evaluation

`tp.lazy(x, y, name)` binds this distance for reductions, selection and products without retaining its full generated comparison matrix. Native source factories depend on the sample family; see [streamed operators](../guide/streaming.md) and [C++ use](../guide/cpp.md).

## Numerical Contract

Floating CPU/CUDA inputs use matching `float32` / `float64`, with lower precision promoted under autocast where supported. First derivatives follow the metric’s smooth and boundary conventions. See the [detailed conventions](https://github.com/inspiros/torchpairwise/blob/master/docs/weighted-distances.md) and [precision and deployment](../guide/deployment.md).

## Related Implementations

- [SciPy minkowski](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.minkowski.html).

Related APIs can differ in reduction, normalization, units or boundary behavior; match these before comparing values.
