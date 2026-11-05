<div class="eyebrow">Point sets and transport</div>

# Directed Hausdorff

<div class="metric-key">directed_hausdorff_distances</div>
<div class="badges"><span class="badge mode-all-pairs">all-pairs</span><span class="badge mode-paired">paired</span><span class="badge mode-single">single</span><span class="badge">distance</span><span class="badge">Point sets</span></div>

\[
h(X,Y)=\max_{x\in X}\min_{y\in Y}\|x-y\|_2
\]

Generic selectors return values. The named all-pairs function also returns source/target witness indices. Directional: no condensed mode.

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
| <span class="badge mode-single">single</span> | `compare(x,y,metric=...,mode="single")` | `[...]`; leading batches only |

Leading batch dimensions broadcast. Condensed mode is not exposed for this directional comparison. See [mode and shape conventions](../guide/modes.md).

## Entry Points

| Named Python entry | Function |
|---|---|
| All-pairs | `directed_hausdorff_distances` |

The Python and C++ `compare` selectors cover every mode shown above. `cdist` / `pairwise_distances` and `paired_distances` select this comparison; condensed distance selectors reject it. `single_comparison` is shorthand for single mode.

Catalog aliases: `directed_hausdorff`.

Accepted metric options: `shuffle`, `generator`, `x1_lengths`, `x2_lengths`, `chunk_size`, `batched`, `feature_dim`. Omitted options keep the metric defaults; unused options are rejected.

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

metric = "directed_hausdorff_distances"
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
auto X = at::rand({2, 3, 2});  // [N=2,P=3,D=2]
auto Y = at::rand({3, 4, 2});  // [M=3,Q=4,D=2]

const std::string metric = "directed_hausdorff_distances";
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

`tp.lazy(x, y, name)` binds this distance for reductions, selection and products without retaining its full generated comparison matrix. Native source factories depend on the sample family; see [streamed operators](../guide/streaming.md) and [C++ use](../guide/cpp.md).

## Numerical Contract

Floating CPU/CUDA inputs use matching `float32` / `float64`, with lower precision promoted under autocast where supported. First derivatives follow the metric’s smooth and boundary conventions. See the [detailed conventions](https://github.com/inspiros/torchpairwise/blob/master/docs/pointsets.md) and [precision and deployment](../guide/deployment.md).

## Related Implementations

- [SciPy directed_hausdorff](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html).

Related APIs can differ in reduction, normalization, units or boundary behavior; match these before comparing values.
