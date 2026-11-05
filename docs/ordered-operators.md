# Sequence, Token and MMD Matrix Operators

The shared `tp.lazy` facade accepts the names and options from
`compare`, including DTW, Soft-DTW divergence, Levenshtein/indel and signed MMD
statistics. Products, bilinear/quadratic forms, reductions, logical transpose,
diagonals and selected blocks all use native C++ execution.

```python
A = tp.lazy(sequences, references, metric="soft_dtw",
                         gamma=0.5, x1_lengths=lengths,
                         x2_lengths=reference_lengths, chunk_size=8)
features = A.matmul(values)
closest = A.amin(-1)

E = tp.lazy(tokens, other_tokens, metric="levenshtein")
edit_counts = E.sum(-1)             # int64
mean_count = E.mean()              # float64
weighted_counts = E.matmul(weights)  # weights' float32/float64 dtype

S = tp.lazy(clouds, other_clouds, metric="mmd_statistic", metric="polynomial", degree=3, gamma=0.1,
                         estimator="unbiased")
statistic_sum = S.sum()            # signed; negative values remain negative
```

Sequence layouts remain `[...,N,T,D]` with prefix lengths `[...,N]`; token
layouts remain `[...,N,T]` with integer IDs. Only batch dimensions broadcast.
Padding and infeasible alignment paths keep the named comparison's conventions.
Tokens allow empty sequences; DTW requires positive time and feature sizes.
A token length tensor broadcasts into its own input sample/batch shape, matching
the canonical edit-distance API. MMD retains point-layout, prefix-length,
kernel parameter and biased/unbiased estimator contracts. Ordinary `mmd`
remains rooted by default; `mmd_statistic` is a separate signed estimate.

Token IDs never pass through floating point. Entries, diagonals, selected blocks,
min/max, sums and products return int64; ordinary integer overflow rules apply.
Means return float64; all/any return bool under numeric nonzero semantics.
Products/forms accept float32 or float64 coefficients on the input device, and
use that dtype for floating distance arithmetic. Coefficients must share a
dtype. Half/bfloat16 coefficients require autocast and produce float32. Only
coefficients receive gradients. `A.dtype` describes the int64 comparison entries.

Soft-DTW self costs and MMD within-cloud terms are prepared once per operand
per operation and reused across tiles. Self calls reuse one preparation when
second-operand lengths are omitted. Nothing is cached across public calls, so
input mutations remain visible. Backward replays comparison tiles; sequence
backward still needs its dynamic-programming workspace, and MMD still needs
point-pair tiles. `chunk_size` bounds each sample axis and coefficient channel
tile; it is not a total-memory limit. MMD accepts `metric_chunk_size` (default 128) for internal point tiles;
sequence and token facades do not expose a separate inner tile parameter.

CPU/CUDA float32/float64, autocast, first derivatives, `torch.compile` and
`torch.export` are supported. Construct the operator inside `forward` and use
`.matmul()` for compilation/export. Native C++ APIs are described in
[native API](native-api.md). See the [runnable example](../examples/ordered_operators.py)
and [general operator contract](pairwise-operators.md).
