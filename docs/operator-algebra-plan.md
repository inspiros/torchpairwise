# Operator Design and Remaining Extensions

For the implemented API, start with the [operator guide](operators.md).
Scalar algebra, all 25 scalar consumers, residual contractions and partial/joint
image event reductions are implemented; the design below also discusses future
feature graphs, selection statistics and operator composition. Implementation
status is explicit in each section. See [algebra](operator-algebra.md),
[statistics](operator-statistics.md) and [event shapes](operator-event-shapes.md)
for current contracts.

## Scope Relative to KeOps

KeOps has its own symbolic formula language and a documented set of operations;
it does not make arbitrary PyTorch operations lazy. Its
[LazyTensor introduction](https://www.kernel-operations.io/keops/python/LazyTensor.html)
shows feature summation remaining symbolic, followed by a sample reduction that
returns a PyTorch tensor. Its [operation reference](https://www.kernel-operations.io/keops/python/api/common/GenericLazyTensor.html)
distinguishes those feature operations from reductions over the two sample axes.
Long finite compositions of supported formula operations are possible; this does
not imply support for an arbitrary sequence of PyTorch calls or intermediate
reductions staying lazy. KeOps also provides specialized consumers, including
weighted softmax reductions and a conjugate-gradient solve.

TorchPairwise will keep a narrower boundary. An initial expression node must be
evaluable for a pair `(i,j)` from its source entries, scalar constants, row `i`
weights and column `j` weights. Combining more approved local nodes does not
require supporting more operation types. Bounded residual feature contractions are now available; general feature
expressions remain future work; a sample reduction or product with a dense tensor is
a terminal consumer. No promise is made that every chain of PyTorch operations
can be fused, or that arbitrarily large expressions use constant workspace.

| Example | Boundary |
|---|---|
| `(alpha * A + B).square().sum()` | Scalar expression followed by a consumer |
| `G.mean(dim=-1).matmul(V)` for a residual source | Per-pair feature contraction, then a scalar consumer |
| `A.mean(-1).mean()` | First call returns a tensor; second call is ordinary PyTorch |
| Row centering/softmax followed by a product | Dedicated consumer; `softmax_matmul` implemented, general nonlocal nodes deferred |
| `A @ B` for two lazy operators | Deferred: contracts an additional sample axis |
| Arbitrary convolution, FFT or Python callback on a lazy matrix | Outside the initial algebra API |

This targets useful combinations of the named comparisons and existing
consumers, not a general tensor compiler. Regular PyTorch remains available for
input preprocessing and for composing the comparatively small consumer outputs.

## Entrywise Scalar Algebra

Let \(A_{ij}=a(x_i,y_j)\) and \(B_{ij}=b(u_i,v_j)\). Both operators must have
the same logical sample dimensions \((N,M)\); leading batch dimensions may
broadcast. Combining operators is positional: the caller is responsible for
making their sample order agree. Transpose preserves each source's original
comparison formula.

The scalar algebra layer constructs lazy operators:

| Operation | Entries | Spelling |
|---|---|---|
| Scale | \(C_{ij}=\alpha A_{ij}\) | `alpha * A` |
| Shift | \(C_{ij}=A_{ij}+\beta\) | `A + beta` |
| Sum / difference | \(C_{ij}=A_{ij}\pm B_{ij}\) | `A + B`, `A - B` |
| Entrywise product | \(C_{ij}=A_{ij}B_{ij}\) | `A * B` |
| Row weights | \(C_{ij}=r_i A_{ij}\) | `A.weight_rows(r)` |
| Column weights | \(C_{ij}=A_{ij}c_j\) | `A.weight_columns(c)` |
| Selected entrywise transforms | \(C_{ij}=f(A_{ij})\) | initially `A.square()`, `A.exp()` |

Scalar constants may be Python numbers or zero-dimensional tensors. A tensor
scalar can receive gradients. Row/column weights use explicit methods with
shapes `[...,N]` and `[...,M]`; unrestricted tensor broadcasting in `A * tensor`
is excluded from the first API to avoid confusing weights with pairwise entries.
An existing `[...,N,M]` tensor can instead be bound with `tp.as_lazy` and
combined as another source. Normal PyTorch dtype promotion will apply to the
expression; integer entries must not be converted to floating point merely to
evaluate integer algebra. Sources must use the same device.

For example, the expression

```python
C = (alpha * A + beta * B).weight_rows(r).weight_columns(c)
result = C.matmul(V)
```

means

\[
R_{iq}=\sum_{j=1}^{M}r_i(\alpha A_{ij}+\beta B_{ij})c_jV_{jq}\]

Likewise, `(A * B).sum()` computes
\(\sum_i\sum_j A_{ij}B_{ij}\) per batch. This is useful for kernel alignment
and related contractions without storing two Gram matrices. `(A * B).matmul(V)`
weights values by a product of pairwise scores. An RBF-like transform of squared
distances is \(C_{ij}=\exp(-\gamma A_{ij})\), written `(-gamma * A).exp()`.
This generic formula retains the source's diagonal and numerical conventions;
it need not be bitwise identical to a dedicated named RBF implementation.

`*` combines entries; `@` contracts the reference axis with a dense tensor:
\((AV)_{iq}=\sum_j A_{ij}V_{jq}\). Operator-by-operator matrix composition
\((AB)_{ik}=\sum_j A_{ij}B_{jk}\) is deferred: it introduces another sample
axis and nested streaming. A general transformed expression also has no promise
of symmetry, nonnegative entries, or positive semidefiniteness, even if its
sources were kernels. It returns a generic scalar expression, not a kernel
specialization with inherited geometric methods.

Consumers evaluate the expression on matching tiles and retain no dense
expression matrix. The native implementation must prepare each source once
per consumer call, replay tiles for backward, and accumulate gradients through
source parameters, scalar factors and weights. Reused expression nodes should
be evaluated once per tile, with their cotangents accumulated. Mutations remain
visible on subsequent calls; there is no persistent numerical cache.

## Sample Axes and Feature Axes

Multiple trailing feature dimensions are part of the shape design, not just D.
See [event shapes and extension priorities](operator-event-shapes.md) for existing
image/matrix-valued interactions, axis rules and the bounded next stage.
[Image-map event reductions](image-map-operators.md) preserve selected event
axes lazily and feed scalar algebra after full contraction. Joint sample/event
reductions are terminal tensor consumers.

A scalar operator represents `[...,N,M]`. The `D` in input vectors `[...,N,D]`
is already consumed by the comparison and is not a scalar operator axis.
Thus `A.mean(dim=-1)` reduces references, `A.mean(dim=-2)` reduces queries,
and `A.mean()` reduces both sample axes, preserving batches. These are terminal
consumers returning ordinary tensors; they do not create another lazy matrix.

A feature-valued source represents `[...,N,M,*event]`, with event axes distinct
from the two sample axes. The first bounded source is
[residual_operator](residual-operators.md), representing `[...,N,M,D]`. For a
single feature axis, \(G_{ijd}=g(x_i,y_j)_d\), a feature contraction is

\[
H_{ij}=\frac{1}{D}\sum_{d=1}^{D}G_{ijd}\]

`G.mean(dim=-1)` returns a scalar operator `H`. We use the familiar `dim=`
spelling on each object shape rather than introducing `reduce_features`. Feature
reduction is lazy; scalar N/M reduction is terminal. The two user examples have
precise meanings:

```python
H = G.mean(dim=-1)        # lazy [...,N,M], reduce D
values = H.matmul(V)      # tensor [...,N,Q]
row_max = H.amax(dim=-1)  # tensor [...,N]
global_max = H.max()      # tensor [...], per batch
```

\[
\mathrm{values}_{iq}=\sum_j\left(\frac1D\sum_dG_{ijd}\right)V_{jq},
\qquad
\mathrm{row\_max}_i=\max_j\left(\frac1D\sum_dG_{ijd}\right)\]

Order matters: averaging per-feature maxima generally differs from taking the
maximum of feature averages. There must be no optimizer rewrite that exchanges
these reductions. Feature reduction can remain inside each pair tile; it must
also chunk large event axes where the source supports that. Native scalar
comparisons should keep their specialized formulas instead of expanding into
an unnecessary `N*M*D` tensor.

The initial `residual_operator` supports identity/absolute/squared residuals
and feature sum/mean/var/std/min/max/prod/all/any, including chunked feature
replay for backward. Scalar top-k over N/M is also implemented as a terminal
consumer. Residual feature top-k is now a terminal `[...,N,M,k]` values/indices
consumer. Image-event top-k likewise selects one current event axis. Every
top-k API follows PyTorch's single-integer dimension convention; joint-axis
selection is outside the API. Quantiles remain deferred.
It also supports terminal N/M and joint N/M/D statistics through `dim=-2/-3`
and dimension tuples/lists. Joint moments are accumulated over the combined
selected set, not composed from separate variances. Scalar consumers accept
N/M dimension lists too. Feature-only contraction stays lazy; sample reduction
returns a Tensor. Arbitrary vector/event graphs still require separate work.
Existing displacement/image-map/residual-outer APIs
retain their documented tensor outputs; ordinary PyTorch operations on them
do not gain automatic streaming fusion.

Cross-pair transformations such as row centering or row softmax are not
entrywise algebra. For example,
\(C_{ij}=A_{ij}-M^{-1}\sum_k A_{ik}\) needs a row statistic. Such operations
require explicit planners or dedicated consumers with documented passes and
row state. They must not be smuggled into a supposedly local transform callback.

The bounded terminal consumer `A.softmax_matmul(V)` now implements the common
row-softmax contraction explicitly. It retains row maxima/scaled masses and
replays sample/channel tiles in native forward and first backward. It does not
create a reusable lazy softmax node or enable arbitrary nonlocal compositions;
see [softmax products](pairwise-operators.md#softmax-products).

## Practical Reduction Expansion

The statistics plan covers losses, diagnostics,
normalization and neighborhood analysis:

| Priority | Operations | Streaming contract |
|---|---|---|
| Implemented | `var`, `std` | Mergeable count/mean/second central moment; stable accumulation |
| Implemented | `logsumexp`, `count_nonzero` | Stable max/exp-sum or integer counts |
| Implemented | Weighted mean, variance and std | Explicit nonnegative weights, normalization and correction rules |
| Implemented for one scalar N/M axis | `median`, `quantile` | Exact row-block sorting; full-axis workspace and two saved indices per output |
| Workload-driven | `kthvalue`, event/global quantiles | Separate axis, return-index and workspace contracts |
| Workload-driven | Approximate quantiles, skewness, kurtosis, histograms | Separate error, numerical and gradient contracts |

For \(n\) eligible entries, mean \(\mu\) and correction \(\delta\),

\[
\operatorname{var}=\frac{1}{n-\delta}\sum_k(a_k-\mu)^2,
\qquad \operatorname{std}=\sqrt{\operatorname{var}}\]

Use a Chan/Welford merge of partial counts, means and centered sums of squares,
not `mean(A.square()) - mean(A).square()`, which loses accuracy for nearly
constant large values. Match PyTorch's correction default and documented
empty/insufficient-count and zero-variance derivatives. Start with floating
inputs; integer statistics need an explicit conversion policy. Exclusion changes
the eligible count, not just the numerator. Global variance must merge row
states; averaging row variances gives the wrong result in general.

Exact scalar median/quantile now support one N/M axis; see the
[implemented contract](operator-statistics.md#exact-median-and-quantile). They are useful for robust losses and distance thresholds, but
do not have the same cheap merge state as mean or variance. Sorting may buffer
a complete row; bounded-workspace selection can need more passes. For even
counts, PyTorch median selects the lower middle entry, whereas default quantile
at 0.5 interpolates. Interpolation, ties, NaNs, gradients and global versus row
workspace must be explicit. Approximate quantiles, if added, should be named
and documented separately, including their error/gradient contract.

The project should cover this finite set of useful reducers rather than promise
arbitrary nested streaming of every PyTorch operation. Local expression nodes,
explicit feature contractions, and terminal sample consumers provide clear
extension points. Nonlocal transforms and operator matrix products require
their own plans before being exposed.
