# Streamed Frobenius Products and Kernel Alignment

`A.frobenius_inner_product(B, center=False, normalize=False)` consumes any two
scalar operators with matching positional N/M axes. Batches broadcast. It returns
one floating scalar per batch and does not construct either full comparison
matrix. Use `tp.as_lazy` for an existing tensor; that tensor's allocation
remains. Both sources must refer to the same ordered row/column observations for
an alignment statistic to be meaningful.

```python
import torchpairwise as tp

K = tp.lazy(X, metric="rbf", gamma=0.2)
L = tp.lazy(features, metric="linear")
cka = K.frobenius_inner_product(L, center=True, normalize=True)
# Biased, centered HSIC convention for N > 1; includes diagonal entries.
hsic = K.hsic(L)
unbiased_hsic = K.hsic(L, unbiased=True)  # at least four observations
```

CKA compares representations on the same observations even when their input
feature dimensions differ. It is invariant to positive rescaling of either
Gram matrix in exact arithmetic. The HSIC expression above uses the common
`(N-1)**2` denominator; another biased convention divides by `N**2`. The dedicated `hsic` consumer also provides the unbiased diagonal-excluding
estimator through `unbiased=True`. A kernel can be signed or
indefinite; this interface does not promise positive semidefiniteness or a
nonnegative result for arbitrary sources.

## Formula and Shapes

For A,B of logical shape `[...,N,M]`, the default returns

\[
F(A,B)=\sum_{i=1}^N\sum_{j=1}^M A_{ij}B_{ij}\]

`center=True` double-centers **each** source, including rectangular sources:

\[
\widetilde A_{ij}=A_{ij}-\frac1M\sum_l A_{il}
-\frac1N\sum_k A_{kj}+\frac1{NM}\sum_{k,l}A_{kl},
\qquad F=\sum_{ij}\widetilde A_{ij}\widetilde B_{ij}\]

For square Gram matrices this is the entry inner product of `H K H` and
`H L H`, with `H=I-11^T/N`. For real symmetric Gram matrices it also equals
`trace(K H L H)`. Directed comparisons are permitted, but the trace
interpretation requires symmetry.

`normalize=True` divides the selected inner product by both Frobenius norms:

\[
S(A,B)=\frac{F(A,B)}{\|A\|_F\|B\|_F},
\]

using centered entries when requested. A zero norm returns zero with zero first
cotangents at the entry level. Empty N/M axes return zero; a finite constant
matrix double-centers to zero. There is no `dim`, self exclusion or estimator
correction option: this consumer contracts both sample axes and includes every
entry. Use ordinary scalar reductions when another statistic is intended.

## Dedicated HSIC Estimators

`K.hsic(L, unbiased=False)` requires matching square Gram operators with at least
2 observations. Its value is the centered Frobenius product divided by `(N-1)**2`.
Both Gram matrices must describe the same ordered observations; their source
feature dimensions can differ. Symmetry and positive semidefiniteness are caller
contracts, not runtime checks.

For `unbiased=True`, N must be at least 4. Let `A` have its diagonal set to zero,
row sums r, column sums c and total sum t. U-centering defines, for i != j,

\[
U(A)_{ij}=A_{ij}-\frac{r_i}{N-2}-\frac{c_j}{N-2}
+\frac{t}{(N-1)(N-2)},\qquad U(A)_{ii}=0\]

The estimator is

\[
\widehat{\mathrm{HSIC}}_u=\frac{1}{N(N-3)}
\sum_{i\ne j}U(K)_{ij}U(L)_{ij}\]

For symmetric Gram matrices this is the usual unbiased HSIC estimator.
The estimate can be negative; it is not clamped. Diagonal entries are excluded
before centering, including NaN/Inf diagonals in supplied matrices. Nonlinear
source derivatives at nonfinite inputs retain their own limitations. For
nonsymmetric matrices the API still returns the defined U-centered product and
its unconstrained first derivatives, but its usual HSIC interpretation does not
apply. No statistical significance test or permutation p-value is provided.

Both estimators use two forward tile passes and O(N) saved state, followed by
bounded replay in first backward. There is no dense centering matrix, Gram
matrix, or transport plan. The unbiased gradient applies U twice; for symmetric
matrices U is a projection, while nonsymmetric inputs require the additional
adjoint terms.

## Execution, Gradients and Numerical Limits

The native implementation shares the existing typed scalar-expression packet.
It supports source mixtures, transposes, tensor factors and row/column weights.
Equal source nodes share tile evaluation, and both root cotangents accumulate
through the source inputs and parameters. Matching sample counts do not imply
matching observations; positional correspondence remains the caller's contract.

Uncentered products need one forward tile pass and constant scalar state per
batch. Centered products need a first pass for row/column means, then an entry
product pass, retaining O(N+M) floating state per batch. First backward replays
bounded source tiles; it does not retain the full matrices. Inputs, source
preparation, gradients and requested outputs remain allocated. Tile limits
follow the smallest participating source chunk width.

Floating entries accumulate in float64 and the result uses the promoted entry
dtype. Normalized products maintain separate running scales for the two operands
to avoid overflow/underflow from squaring otherwise finite entries. Integer-only sources are rejected; explicitly promote them in scalar
algebra when a floating statistic is intended. Autocast uses existing source
promotion. Centering retains a separate finite anchor and mean offsets to avoid
losing small variation on large common translations. Public source formulas
and centered differences can still overflow. Nonfinite source entries are
outside the finite-input accuracy guarantees; the explicit zero-norm convention
can mask a nonfinite partner. This is not a universal full-range numerical guarantee.
Higher derivatives remain unsupported.

Construct operators inside `forward` when compiling or exporting. The shared
native implementation is available from standalone C++:

```cpp
auto k = torchpairwise::ops::pairwise_expression(x, {}, "rbf", true);
auto l = torchpairwise::ops::pairwise_expression(features, {}, "linear", true);
auto cka = k.frobenius_inner_product(l, true, true);
auto hsic_unbiased = k.hsic(l, true);
```

Exported execution requires the native library. CMake/standalone and release
wheel validation follow the repository's review-stage policy; they are separate
from routine Python CPU/CUDA and compilation checks.
