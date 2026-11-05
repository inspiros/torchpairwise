<div class="eyebrow">Getting started</div>

# Quick Start

Choose a formula, then a pairing mode. These small squared-Euclidean examples
have the same values in Python and C++. N and M are the collection sizes;
D is the number of features in one vector.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
import torch
import torchpairwise as tp

X = torch.tensor([[0., 0.], [3., 4.]])  # [N=2,D=2]
Y = torch.tensor([[0., 4.], [3., 0.]])  # [M=2,D=2]
metric = "sqeuclidean"

matrix = tp.compare(X, Y, metric=metric)                    # [2,2]
paired = tp.compare(X, Y, metric=metric, mode="paired")     # [2]
condensed = tp.compare(X, metric=metric, mode="condensed") # [1]
single = tp.compare(X[0], Y[0], metric=metric, mode="single")  # scalar
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
#include <torchpairwise.h>
namespace tp = torchpairwise::ops;

auto X = at::tensor({0.f, 0.f, 3.f, 4.f}).reshape({2, 2});
auto Y = at::tensor({0.f, 4.f, 3.f, 0.f}).reshape({2, 2});
const std::string metric = "sqeuclidean";

auto matrix = tp::compare(X, Y, metric);                          // [2,2]
auto paired = tp::compare(X, Y, metric, "paired");               // [2]
auto condensed = tp::compare(X, c10::nullopt, metric, "condensed"); // [1]
auto single = tp::compare(X[0], Y[0], metric, "single");         // scalar
```

</div>
</div>

| Mode | Values | Shape |
|---|---|---|
| <span class="badge mode-all-pairs">all-pairs</span> | `[[16,9],[9,16]]` | `[2,2]` |
| <span class="badge mode-paired">paired</span> | `[16,16]` | `[2]` |
| <span class="badge mode-condensed">condensed</span> | `[25]` | `[1]` |
| <span class="badge mode-single">single</span> | `16` | scalar |

Condensed compares X with itself, taking only i<j. Single inputs omit the
collection axis. `single_comparison` is an optional convenience for the same
single mode. See [modes and shapes](modes.md) for batches and structured samples.

## A Streamed Result

If the algorithm needs a reduction or selected neighbors, bind the formula
instead of returning every comparison. Here the logical source is `[2,2]`;
the consumer returns `[2]`.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
A = tp.lazy(X, Y, metric="sqeuclidean", chunk_size=1)
row_sums = A.sum(dim=-1)                # [2], values [25,25]
neighbors = A.topk(1, largest=False)    # values/indices [2,1]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
tp::PairwiseOperationOptions options;
options.chunk_size = 1;
auto A = tp::pairwise_expression(X, Y, "sqeuclidean", false, options);
auto row_sums = A.reduce("sum", -1);     // [2], values [25,25]
auto [values, indices] = A.topk(1, -1, false);  // [2,1] each
```

</div>
</div>

The nearest values are `[[9],[9]]`, with reference indices `[[1],[0]]`.
Bindings retain input references and options, not computed matrices. Each
consumer evaluates the necessary tiles again. Continue with
[sources and consumers](streaming.md).
