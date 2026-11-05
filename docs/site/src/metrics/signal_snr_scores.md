<div class="eyebrow">Audio</div>

# Signal SNR

<div class="metric-key">signal_snr_scores</div>
<div class="badges"><span class="badge mode-all-pairs">all-pairs</span><span class="badge mode-paired">paired</span><span class="badge mode-single">single</span><span class="badge">score</span><span class="badge">Audio</span></div>

\[
\operatorname{SNR}=\operatorname{mean}_c10\log_{10}\frac{\max(\|r\|_2^2,\delta)}{\max(\|e-r\|_2^2,\delta)}
\]

First operand is estimate e, second reference r. zero_mean=False; energy_floor=1e-8. This audio dB score differs from vector snr_distances.

## Shapes and Modes

| Input | Layout |
|---|---|
| One sample | `[C,T]` |
| Collection X | `[...,N,C,T]` |
| Collection Y | `[...,M,C,T]` |

| Mode | Selector | Output |
|---|---|---|
| <span class="badge mode-all-pairs">all-pairs</span> | `compare(X,Y,metric=...)` | `[...,N,M]` |
| <span class="badge mode-paired">paired</span> | `compare(X,Y,metric=...,mode="paired")` | `[...,N]`; equal sample counts |
| <span class="badge mode-single">single</span> | `compare(x,y,metric=...,mode="single")` | `[...]`; leading batches only |

Leading batch dimensions broadcast. Condensed mode is not exposed for this directional comparison. See [mode and shape conventions](../guide/modes.md).

## Entry Points

| Named Python entry | Function |
|---|---|
| All-pairs | `signal_snr_scores` |
| Paired | `paired_signal_snr_scores` |

The Python and C++ `compare` selectors cover every mode shown above. Use `compare` for this score; distance/kernel selectors have different roles. `single_comparison` is shorthand for single mode.

Catalog aliases: `signal_snr`.

Accepted metric options: `zero_mean`, `energy_floor`, `chunk_size`. Omitted options keep the metric defaults; unused options are rejected.

## Example

Choose Python or C++ in the top bar; the setting applies across this guide.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
import torch
import torchpairwise as tp

X = torch.rand(2, 1, 24)  # [N=2,C=1,T=24]
Y = torch.rand(3, 1, 24)  # [M=3,C=1,T=24]

metric = "signal_snr_scores"
all_pairs = tp.compare(X, Y, metric=metric)  # [2,3]
paired_Y = Y[:2]
paired = tp.compare(X, paired_Y, metric=metric, mode="paired")  # [2]
x, y = X[0], Y[0]
single = tp.compare(x, y, metric=metric, mode="single")  # scalar
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
#include <torchpairwise.h>

namespace tp = torchpairwise::ops;
auto X = at::rand({2, 1, 24});
auto Y = at::rand({3, 1, 24});

const std::string metric = "signal_snr_scores";
tp::ComparisonOptions options;
auto all_pairs = tp::compare(X, Y, metric, "all_pairs", options);  // [2,3]
auto paired_Y = Y.narrow(0, 0, 2);
auto paired = tp::compare(X, paired_Y, metric, "paired", options);  // [2]
auto x = X[0], y = Y[0];
auto single = tp::compare(x, y, metric, "single", options);  // scalar
```

</div>
</div>

## Streamed Evaluation

`tp.lazy(x, y, name)` binds this score for reductions, selection and products without retaining its full generated comparison matrix. Native source factories depend on the sample family; see [streamed operators](../guide/streaming.md) and [C++ use](../guide/cpp.md).

## Numerical Contract

Floating CPU/CUDA inputs use matching `float32` / `float64`, with lower precision promoted under autocast where supported. First derivatives follow the metric’s smooth and boundary conventions. See the [detailed conventions](https://github.com/inspiros/torchpairwise/blob/master/docs/audio-comparisons.md) and [precision and deployment](../guide/deployment.md).

## Related Implementations

- [TorchMetrics SNR](https://lightning.ai/docs/torchmetrics/stable/audio/signal_noise_ratio.html).

Related APIs can differ in reduction, normalization, units or boundary behavior; match these before comparing values.
