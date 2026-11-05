<div class="eyebrow">Point sets and transport</div>

# Squared MMD Statistic

<div class="metric-key">mmd_statistics</div>
<div class="badges"><span class="badge mode-all-pairs">all-pairs</span><span class="badge mode-paired">paired</span><span class="badge mode-condensed">condensed</span><span class="badge mode-single">single</span><span class="badge">statistic</span><span class="badge">Point sets</span></div>

\[
\widehat{\operatorname{MMD}}_u^2=\frac{\sum_{i\ne j}k(x_i,x_j)}{P(P-1)}+\frac{\sum_{i\ne j}k(y_i,y_j)}{Q(Q-1)}-\frac{2\sum_{i,j}k(x_i,y_j)}{PQ}
\]

estimator="unbiased" is the default and requires at least two valid points on each side. estimator="biased" includes diagonals. Signed result: no clamp or root.

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
| All-pairs | `mmd_statistics` |
| Paired | `paired_mmd_statistics` |
| Condensed | `condensed_mmd_statistics` |

The Python and C++ `compare` selectors cover every mode shown above. Use `compare` for this statistic; distance/kernel selectors have different roles. `single_comparison` is shorthand for single mode.

Catalog aliases: `mmd_statistic`.

Accepted metric options: `kernel`, `estimator`, `length_scale`, `gamma`, `degree`, `coef0`, `alpha`, `nu`, `batched`, `feature_dim`, `x1_lengths`, `x2_lengths`, `chunk_size`. Omitted options keep the metric defaults; unused options are rejected.

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

metric = "mmd_statistics"
all_pairs = tp.compare(X, Y, metric=metric)  # [2,3]
paired_Y = Y[:2]
paired = tp.compare(X, paired_Y, metric=metric, mode="paired")  # [2]
condensed = tp.compare(X, metric=metric, mode="condensed")  # [1]
x, y = X[0], Y[0]
single = tp.compare(x, y, metric=metric, mode="single")  # scalar
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
#include <torchpairwise.h>

namespace tp = torchpairwise::ops;
auto X = at::rand({2, 3, 2});  // [N=2,P=3,D=2]
auto Y = at::rand({3, 4, 2});  // [M=3,Q=4,D=2]

const std::string metric = "mmd_statistics";
tp::ComparisonOptions options;
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

`tp.lazy(x, y, name)` binds this statistic for reductions, selection and products without retaining its full generated comparison matrix. Native source factories depend on the sample family; see [streamed operators](../guide/streaming.md) and [C++ use](../guide/cpp.md).

## Numerical Contract

Floating CPU/CUDA inputs use matching `float32` / `float64`, with lower precision promoted under autocast where supported. First derivatives follow the metric’s smooth and boundary conventions. See the [detailed conventions](https://github.com/inspiros/torchpairwise/blob/master/docs/distribution-comparisons.md) and [precision and deployment](../guide/deployment.md).
