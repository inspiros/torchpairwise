<div class="eyebrow">Sequences and tokens</div>

# Levenshtein

<div class="metric-key">levenshtein_distances</div>
<div class="badges"><span class="badge mode-all-pairs">all-pairs</span><span class="badge mode-paired">paired</span><span class="badge mode-condensed">condensed</span><span class="badge mode-single">single</span><span class="badge">distance</span><span class="badge">Tokens</span></div>

\[
L_{ij}=\min\{L_{i-1,j}+1,L_{i,j-1}+1,L_{i-1,j-1}+\mathbf1[x_i\ne y_j]\}
\]

Integer token IDs, unit insertion/deletion/substitution costs. Returns int64; valid prefix lengths may be zero. Tokens are labels, not numeric features.

## Shapes and Modes

| Input | Layout |
|---|---|
| One sample | `[T]` |
| Collection X | `[...,N,T]` |
| Collection Y | `[...,M,S]` |

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
| All-pairs | `levenshtein_distances` |
| Paired | `paired_levenshtein_distances` |
| Condensed | `condensed_levenshtein_distances` |

The Python and C++ `compare` selectors cover every mode shown above. `cdist` / `pairwise_distances` and `paired_distances` select this comparison; `pdist` / `condensed_distances` select its condensed mode. `single_comparison` is shorthand for single mode.

Catalog aliases: `levenshtein`.

Accepted metric options: `x1_lengths`, `x2_lengths`, `chunk_size`. Omitted options keep the metric defaults; unused options are rejected.

## Example

Choose Python or C++ in the top bar; the setting applies across this guide.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
import torch
import torchpairwise as tp

X = torch.tensor([[1, 2, 3], [3, 2, 1]])             # [N=2,T=3]
Y = torch.tensor([[1, 2, 2], [3, 3, 1], [2, 2, 1]])  # [M=3,S=3]

metric = "levenshtein_distances"
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
auto X = at::tensor({1, 2, 3, 3, 2, 1}, at::kLong).reshape({2, 3});
auto Y = at::tensor({1, 2, 2, 3, 3, 1, 2, 2, 1}, at::kLong).reshape({3, 3});

const std::string metric = "levenshtein_distances";
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

Token inputs are integer tensors with integer outputs; no input derivatives. See the [detailed conventions](https://github.com/inspiros/torchpairwise/blob/master/docs/token-comparisons.md) and [precision and deployment](../guide/deployment.md).

## Related Implementations

- [RapidFuzz Levenshtein](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html).

Related APIs can differ in reduction, normalization, units or boundary behavior; match these before comparing values.
