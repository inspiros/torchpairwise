# Additional Vector and Histogram Distances

These comparisons accept dense float32 or float64 vectors. Under autocast,
float16 and bfloat16 inputs are promoted to float32 and return float32 results.
Only batch axes broadcast; feature dimensions must match. Each comparison is
available as an all-pairs function, through `cdist`, and through
`paired_distances` and `condensed_distances` with the metric names below.

```python
torchpairwise.angular_distances(x, y)
torchpairwise.cdist(x, y, metric="lorentzian")
torchpairwise.paired_distances(x, y, metric="clark")
torchpairwise.condensed_distances(x, metric="ruzicka")
```

## Angular

`angular` measures the angle in **radians**, from zero to pi, between finite,
nonzero real vectors. With unit vectors \(u=x/\lVert x\rVert_2\) and
\(v=y/\lVert y\rVert_2\), its formula is

\[
d(x,y)=2\operatorname{atan2}\left(\lVert u-v\rVert_2,
                                      \lVert u+v\rVert_2\right)\]

The result is invariant under positive scaling of either row. It is not divided
by pi. Zero vectors have no direction and raise `ValueError`; no angle is
assigned to them. Empty sample axes remain valid.

The two-norm formula retains small angles and angles close to pi without
differentiating `acos` near its singular endpoints. First derivatives exist
away from coincident and antipodal directions. At either endpoint the
implementation chooses zero subgradients for both operands, including when
backward receives a zero upstream gradient. Row and feature-norm scaling avoids
overflow for finite inputs with very large magnitude.

## Lorentzian

`lorentzian` accepts finite, signed real vectors and computes

\[
d(x,y)=\sum_k\log\left(1+|x_k-y_k|\right)\]

The logarithm is natural. This definition uses the sum, with no feature-count
normalization. An exact coordinate equality has the zero subgradient of the
absolute value. Finite, oppositely signed extreme values are handled without
overflowing their difference before taking its logarithm.

## Clark

`clark` accepts finite, nonnegative vectors and computes

\[
d(x,y)=\sqrt{\sum_k\left(\frac{|x_k-y_k|}{x_k+y_k}\right)^2}\]

A coordinate with both values zero contributes zero and has zero chosen
derivatives. Other zero coordinates are allowed. Both-zero vectors consequently
have distance zero, and coincident vectors use zero subgradients. Per-coordinate
scaling preserves small bins even beside large bins. A zero vector versus a
vector with \(r\) positive coordinates has distance \(\sqrt r\).

## Soergel and Ruzicka (Continuous Jaccard)

`soergel` and `ruzicka` accept finite, nonnegative vectors and compute the same
distance on this domain:

\[
d(x,y)=\frac{\sum_k |x_k-y_k|}{\sum_k\max(x_k,y_k)}
      =1-\frac{\sum_k\min(x_k,y_k)}{\sum_k\max(x_k,y_k)}\]

This is the continuous min/max Jaccard distance, also called Ruzicka distance.
It differs from the Boolean `jaccard` metric. The result lies between zero and
one. Both-zero vectors have distance zero with zero chosen derivatives; a zero
vector versus any nonzero vector has distance one. Equal coordinates use the
zero subgradient of the absolute value. Inputs are not normalized to unit mass,
and arbitrary separate rescaling of rows changes the result. Simultaneously
rescaling both rows by a positive scalar preserves it.

## Evaluation and Precision

All five names support first-order autograd, broadcast batches, noncontiguous
inputs, and empty sample axes. No feature weights or smoothing parameters are
added. All-pairs, paired and condensed evaluations use sample tiles sized from a
fixed element budget (about four million pair-feature entries, counting several
float64 temporaries per feature) and feature tiles of at most 64 entries.
Formulas that reduce to distance or product matrices of transformed rows
(Hellinger, Bhattacharyya, Aitchison, rational quadratic and Matérn) use
pair-sized tiles. Condensed output evaluates row blocks against their column
suffix. Feature reductions are checkpointed while tracking gradients, bounding
retained pair-by-feature intermediates. The final
all-pairs output still requires storage proportional to the number of pairs.

Float32 inputs use float64 arithmetic internally and round the final output to
float32. Values below the output dtype's representable range can underflow, and
derivatives whose magnitudes exceed its range can overflow. Domain assertions
remain in compiled graphs. Eager invalid values raise `ValueError` from Python
and a synchronous `RuntimeError` from the native operators, on CPU and CUDA;
compiled CUDA assertions can report errors asynchronously.
