# Streamed Operator Statistics

For source selection and shared axis rules, start with the [operator guide](operators.md).

Scalar operators and expressions support `var`, `std`, `logsumexp` and
`count_nonzero` in addition to the seven basic reducers. They evaluate comparison
tiles without retaining the full scalar matrix.

```python
D = tp.lazy(X, Y, metric="sqeuclidean")
spread = D.std(-1, correction=0)  # spread of each query's distances
log_density = (-gamma * D).logsumexp(-1)  # log of summed affinities
nonzero_pairs = D.count_nonzero()  # one int64 count per batch
```

These are useful for neighborhood diagnostics, distance-based losses and stable
log-domain kernel sums. Each call is a terminal consumer returning an ordinary
tensor. It does not make subsequent normalization or other PyTorch operations
lazy. A log mean affinity is `log_density - math.log(Y.shape[-2])` when the
reference count is positive and no entries are excluded.

## Axes and Correction

For inputs `X: [..., N, D]` and `Y: [..., M, D]`, the metric consumes `D`
and defines scalar entries `A: [..., N, M]`. These statistics act on **N/M**;
they do not compute statistics over input features or unreduced image/point
coordinates. For example, `A.var(-1)` has shape `[..., N]` and measures how
each query's comparison scores vary across references. For feature statistics,
[residual_operator](residual-operators.md) represents `[...,N,M,D]` and its
`var(-1)` contracts D into a lazy scalar source instead. The dimension always
refers to the shape of the object on which the method is called.

[Image-map operators](image-map-operators.md) similarly support event and joint
sample/event var/std: spatial mean can retain C lazily, while joint M/H/W
variance returns a terminal N/C tensor. These event sources expose nine basic
reducers; logsumexp, count_nonzero and normalized weighted statistics belong to
the scalar interface after all event axes have been contracted.

`dim=-1` reduces references, `dim=-2` reduces queries and `dim=None` reduces both
sample axes. An integer, tuple/list such as `dim=[-2,-1]`, or equivalent positive
indices are accepted. Empty tuples/lists also select both logical sample axes;
duplicates and batch-axis selection raise errors. Multiple dimensions define
one joint statistic, with correction applied once to the total eligible count. Batch axes are preserved. `keepdim=True` retains reduced axes.
`exclude_self=True` removes the positional diagonal and requires a square
operator; it changes the eligible count as well as the accumulated statistic.
Logical transpose preserves directed metric formulas, as with other consumers.

Variance and standard deviation accept a finite real `correction` (default 1,
as in PyTorch; 0 gives population variance). The generic spelling is
`A.reduce("var", dim=-1, correction=0)`. The correction option applies only to
variance and standard deviation. For eligible count \(n\), mean \(\mu\) and
correction \(\delta\),

\[
\operatorname{var}(A)=\frac{\sum_k(A_k-\mu)^2}{\max(0,n-\delta)},
\qquad \operatorname{std}(A)=\sqrt{\operatorname{var}(A)}\]

Zero degrees of freedom can produce infinity or NaN, following PyTorch values
and first derivatives for explicit sample axes. Empty variance/std are NaN. Zero variance with positive
degrees of freedom has a zero std derivative. `logsumexp` of no eligible entries
is negative infinity. `count_nonzero` of no entries is zero; NaNs count as
nonzero, as in PyTorch.

## Numerical and Memory Contract

Variance uses a Chan merge of centered tile moments, not subtraction of two
large raw moments. Each row/global state retains an anchor and a mean offset
separately, including for backward. This keeps small variations from disappearing
when an absolute mean cannot be represented accurately. Finite constant rows
near the float64 limit avoid overflowing a raw sum. Opposite finite extremes can
rebase the anchor to zero, with normalized accumulation if the offset sum
overflows. The variance's squared deviations can still exceed float64 range;
this is not a full-range or bitwise-equivalence guarantee. True nonfinite entries
retain nonfinite variance semantics.

For example, population variance of `[1e16, 1e16+2, ..., 1e16+8]` is 8 across
sample tiles, and finite constant `1e308` rows have zero variance and zero
first derivatives. These translated and exact-constant oracles are useful at
extreme magnitudes where PyTorch CPU/CUDA algorithms can themselves differ.
The anchor/offset state is always retained. Extreme-range rebasing and normalized
fallbacks run only for float64 entries: finite float32 entries cannot overflow
the float64 moment accumulator, so they skip that work automatically. This
choice uses dtype metadata and adds no data-dependent synchronization or public
mode switch.

Normalized weighted statistics retain separate anchors and mean offsets for
float64 entries; float32 retains its existing float64 absolute-mean path. They
use their own weight-scaled moment implementation, without the full-range
rebasing/normalization fallbacks above. Float64 residual/image feature and joint
statistics also retain anchors separately from mean offsets; local image event
var/std center their inputs before calling ATen. Float32 keeps the existing
float64 accumulation paths. These feature/event implementations do not add the
scalar full-range fallbacks: public transforms, differences, sums and squared
deviations can still overflow or lose precision. Nearly cancelling gradients
can also have large relative errors, especially after squaring large residuals.

Mean offsets and centered sums accumulate in float64 for float32 and
float64 entries. `logsumexp` merges tile log sums with `logaddexp`, also in
float64. Floating results use the entry dtype; autocast promotes low-precision
sources and factors to float32. Float64 accumulation trades some throughput for
accuracy, especially on GPUs with limited float64 throughput.

Floating statistics require float32/float64 entries. Integer/Boolean sources can
use `count_nonzero`, which returns int64 and has no gradient; explicit floating
algebra can promote entries when that is intended. Counts retain integer truth
semantics without conversion through float64.

Per-row/column statistics retain state proportional to that output axis; global
statistics merge directly across all tiles. First backward replays tiles and
accumulates gradients through sources, scalar parameters and row/column weights.
No dense intermediate matrix or Python numerical callback is retained. Higher
derivatives remain unsupported.

## Native and Compiled Use

Construct operators inside `forward` for `torch.compile` and `torch.export`.
The registered native frontend uses the same typed source packet as scalar
algebra. C++ uses the existing typed expression factories for any source family:

```cpp
auto a = torchpairwise::ops::pairwise_expression(x, y, "sqeuclidean");
auto spread = a.std(-1, 0.);  // population standard deviation
// Generic equivalent: a.statistics("std", -1, false, false, 0.).
auto log_density = a.scale(-0.5).logsumexp(-1);
```

Matrix, structured, media and ordered factories support the same statistics.
The original C++ family facades keep their original consumer signatures; use
these expression factories for the expanded statistics. Standalone C++,
installed-wheel and latest-CPU validation are performed at review time.

## Normalized Weighted Statistics

```python
D = tp.lazy(X, Y, metric="sqeuclidean")
mean_distance = D.weighted_mean(confidence, dim=-1)  # confidence:[...,M]
spread = D.weighted_std(confidence, dim=-1)
K = (-gamma * D).exp()
local_variance = D.weighted_var(K, dim=-1, correction=1)
```

`weighted_mean`, `weighted_var` and `weighted_std` normalize their weights;
`weight_rows`/`weight_columns` remain entrywise multiplication. A tensor argument
is confidence along the explicitly reduced axis: `[...,M]` for `dim=-1` (the
default), `[...,N]` for `dim=-2`. Its batch dimensions broadcast. A scalar operator
or expression supplies pair-specific weights with matching positional N/M axes.
For existing `[...,N,M]` weight tensors, use `tp.as_lazy(weights)` explicitly.
A global reduction (`dim=None`) requires an operator, avoiding an ambiguous axis
for a bare confidence vector. `keepdim` and `exclude_self` have the usual sample
axis semantics. Transpose the weight operator too when transposing its values.

For eligible weights \(w_k\), define \(W=\sum_k w_k\),
\(W_2=\sum_k w_k^2\) and \(\mu_w=\sum_k w_k A_k/W\). Then

\[
\operatorname{weighted\_var}(A,w)
=\frac{\sum_k w_k(A_k-\mu_w)^2}
       {W-\delta W_2/W}\]

The default `correction=0` is population variance; `correction=1` is the standard
unbiased reliability-weight correction. This is a confidence/reliability
interpretation, not a count of repeated observations. Constant positive weights
recover ordinary variance with the same correction. Finite nonnegative real
corrections are accepted. Zero total mass or a nonpositive corrected denominator
produces NaN, with undefined gradients. Weighted std is the square root of this
variance and has a zero derivative at zero valid variance.

Evaluated weights must be finite and nonnegative, including excluded positions.
Integer/Boolean weights are supported without weight gradients. Values must be
floating entries; explicitly promote an integer-valued operator when intended.
Zero weights ignore nonfinite entry values; their value and weight cotangents
are zero by convention. Source expressions retain their own nonfinite derivative
behavior, so this is not a guarantee of finite gradients for every source input. At zero weights on finite values, weight gradients
follow the normalized formula. Positive-weight nonfinite values propagate the
usual nonfinite results.

Centered moments accumulate in float64. A running weight scale prevents common
rescaling of finite weights from overflowing the mass or squared-mass sums.
For float64 entries each row/global state keeps an anchor and weighted mean
offset separately across tiles and in backward. This retains small variations around a large translation,
including derivatives with respect to confidence weights. The public weighted
mean is rounded to the entry dtype; its derivatives use the separate offset.
This is not a full-range guarantee: extreme differences, weighted offset sums
and squared deviations can still overflow float64.
Float32 retains the previous absolute-mean arithmetic in float64, avoiding
persistent-anchor work for these lower-precision entries. Dtype selects the
path automatically.
The normalized result is invariant to a positive common scale. The native graph
shares prepared sources and evaluates shared nodes once per tile, including when
weights depend on the values. Backward accumulates both value and weight
cotangents through that graph. Per-output state has six float64 values for float64 entries and five for
float32 entries; no dense
pairwise matrix or persistent numeric cache is retained. Higher derivatives remain
unsupported. Float64 accumulation has the same throughput tradeoff as unweighted
statistics.

C++ expression factories expose the same methods, accepting either an axis
confidence tensor or another lazy pairwise object:

```cpp
auto d = torchpairwise::ops::pairwise_expression(x, y, "sqeuclidean");
auto k = d.scale(-0.5).exp();
auto average = d.weighted_mean(k, -1);
auto variance = d.weighted_var(k, -1, 1.);
```

Dimension tuples/lists also work for weighted statistics. A confidence tensor
still requires exactly one selected sample axis; joint N/M weighting requires a
scalar weight operator. For example, `A.weighted_var(W, dim=[-2,-1])` accepts a
lazy scalar W and computes one reliability-weighted variance per batch.

For positive weighted dimensions, index the full broadcast weighted shape. For
example, an unbatched N*M value source with B*M confidence weights has logical
weighted shape B*N*M, so `dim=2` selects references. Negative -1/-2 keep their
usual M/N meanings regardless of added batch axes.

## Exact Median and Quantile

```python
D = tp.lazy(X, Y, metric="sqeuclidean", chunk_size=64)
robust_loss = D.median(dim=-1).mean()          # lower middle per query
thresholds = D.quantile(0.9, dim=-1)           # per-query distance threshold
column_medians = D.median(dim=-2, keepdim=True)
# Also works after scalar algebra or complete event contraction:
robust_error = tp.residual_operator(X, Y, transform="abs").mean(-1).quantile(0.5)
```

Both methods return values as ordinary tensors, consistent with operator
min/max. They accept **one integer N/M dimension**, including its positive
shape index, and preserve leading batches. The default is `dim=-1`.
`keepdim=True` retains that axis; `exclude_self=True` removes the positional
diagonal of a square operator before selection. Empty eligible axes raise;
an empty retained sample axis or batch produces an empty tensor. `dim=None`,
lists/tuples, vector/tensor q, joint/global selection and event-axis selection
are not part of this first API. Approximate quantiles are separate future work.

For sorted eligible entries \(a_{(0)},\ldots,a_{(L-1)}\), median selects
\(a_{(\lfloor(L-1)/2\rfloor)}\), the lower middle for even L. Quantile accepts
one finite Python real \(q\in[0,1]\). With \(r=q(L-1)\), linear interpolation is

\[
Q(q)=(1-w)a_{(\lfloor r\rfloor)}+w a_{(\lceil r\rceil)},
\qquad w=r-\lfloor r\rfloor\]

`interpolation` accepts PyTorch's `linear` (default), `lower`, `higher`,
`nearest` (rank ties to even), and `midpoint`. Thus `median()` and
`quantile(0.5)` differ on even counts. Quantile requires float32/float64 entries;
median also accepts integer entries without converting their values to floats.
Boolean entries are rejected. Native autocast promotes low-precision sources
and factors to float32.

Sorting is stable in the original logical axis order. A tied rank sends its
cotangent to that entry rather than averaging across all tied values.
A row containing any NaN has a NaN result: median chooses the first logical
NaN; quantile uses the last stable sorted NaN. PyTorch propagates a NaN too,
but its tied sort witnesses can differ between CPU and CUDA; those tied
gradients need not match this deterministic ordering.
Interpolated infinities follow ATen `lerp`, including its NaN results at some
endpoints. First backward applies the selected/interpolated cotangents.
Composed expressions and general comparisons replay all source tiles, including
zero-cotangent entries, preserving native nonfinite chain rules. Plain matrix
sources scatter the sparse cotangents directly. Plain unweighted squared-Euclidean
sources use a CUDA adjoint that reads selection indices in registers and sums
all partner terms, including zero-cotangent nonfinite products. It avoids
per-tile autograd graphs and dense cotangent matrices; it does not skip those
products. CPU keeps the replay adjoint. Higher derivatives are unsupported.

This is exact row-block sorting, with **axis-dependent workspace**, rather
than a constant-state reducer. For an axis of length L, row-block width is
`min(chunk_size, max(1, floor(2**20/L)))`; one full row can exceed that element
target. Each batch is processed separately. Source evaluations still use
sample tiles bounded by `chunk_size`. Temporary values, sort indices, sort
workspace and optional diagonal compaction scale with the row-block size times
L. Output and two saved int64 indices per result scale with the retained sample
axis. Backward needs source-gradient buffers and tile replay but no full score
matrix or saved per-tile graph. Small dense workloads can be substantially faster;
select this path when avoiding the full computed matrix matters. Caller-owned
`tp.as_lazy` inputs already occupy their dense storage.

One CUDA float32 squared-Euclidean workload (N=2049, M=8193, D=8, q=0.9,
chunk_size=128, forward plus first backward) measured:

| Path | Extra peak GPU allocation | Median time |
|---|---|---|
| Dense distance followed by `torch.quantile` | 576 MiB | 15.9 ms |
| Exact operator quantile before targeted optimization | 48.3 MiB | 709 ms |
| Exact operator quantile with the squared-Euclidean optimization | 48.3 MiB | 27.7 ms |

Values and gradients were checked before timing. These are three warmed,
non-interleaved samples on a shared host, illustrating a memory/time tradeoff;
they are not a general performance guarantee. The optimized case retains the
same sorted values, stable witness rules and interpolation API. Other comparisons
and composed expressions retain their general execution path. Matrix sources
also avoid repeated gathers by sorting row views, with a sparse scatter adjoint.

The shared native implementation covers scalar pairwise/kernel, structured,
media, ordered, matrix and scalar-expression sources. Residual/image operators
can use it after contracting **all** event axes to a scalar source. C++ uses
`ScalarExpression::median(dim, keepdim, exclude_self)` and
`ScalarExpression::quantile(q, dim, keepdim, interpolation, exclude_self)`.
Typed dispatcher schemas and Fake metadata support `torch.compile` and
`torch.export`; built-in execution has no Python numerical dependency.
