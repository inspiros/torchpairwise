<div class="eyebrow">Images</div>

# Peak Signal-to-Noise Ratio

<div class="metric-key">psnr_scores</div>
<div class="badges"><span class="badge mode-all-pairs">all-pairs</span><span class="badge mode-paired">paired</span><span class="badge mode-condensed">condensed</span><span class="badge mode-single">single</span><span class="badge">score</span><span class="badge">Images</span></div>

\[
\operatorname{PSNR}(x,y)=10\log_{10}\frac{R^2}{(CHW)^{-1}\sum_{c,h,w}(x_{chw}-y_{chw})^2}
\]

Aligned whole images; data_range R is required. Returns dB. Identical images give +infinity with a chosen zero derivative.

## Shapes and Modes

| Input | Layout |
|---|---|
| One sample | `[C,H,W]` |
| Collection X | `[...,N,C,H,W]` |
| Collection Y | `[...,M,C,H,W]` |

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
| All-pairs | `psnr_scores` |
| Paired | `paired_psnr_scores` |
| Condensed | `condensed_psnr_scores` |

The Python and C++ `compare` selectors cover every mode shown above. Use `compare` for this score; distance/kernel selectors have different roles. `single_comparison` is shorthand for single mode.

Catalog aliases: `psnr`.

Accepted metric options: `data_range`, `chunk_size`. Omitted options keep the metric defaults; unused options are rejected.

## Example

Choose Python or C++ in the top bar; the setting applies across this guide.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
import torch
import torchpairwise as tp

X = torch.rand(2, 1, 8, 8)  # [N=2,C=1,H=8,W=8]
Y = torch.rand(3, 1, 8, 8)  # [M=3,C=1,H=8,W=8]

metric = "psnr_scores"
options = dict(data_range=1.0)
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
auto X = at::rand({2, 1, 8, 8});
auto Y = at::rand({3, 1, 8, 8});

const std::string metric = "psnr_scores";
tp::ComparisonOptions options;
options.data_range = 1.0;
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

`tp.lazy(x, y, name)` binds this score for reductions, selection and products without retaining its full generated comparison matrix. Native source factories depend on the sample family; see [streamed operators](../guide/streaming.md) and [C++ use](../guide/cpp.md).

## Numerical Contract

Floating CPU/CUDA inputs use matching `float32` / `float64`, with lower precision promoted under autocast where supported. First derivatives follow the metric’s smooth and boundary conventions. See the [detailed conventions](https://github.com/inspiros/torchpairwise/blob/master/docs/image-comparisons.md) and [precision and deployment](../guide/deployment.md).

## Related Implementations

- [TorchMetrics PSNR](https://lightning.ai/docs/torchmetrics/stable/image/peak_signal_noise_ratio.html).

Related APIs can differ in reduction, normalization, units or boundary behavior; match these before comparing values.
