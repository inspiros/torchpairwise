# Ordered Continuous Sequences

DTW compares ordered feature sequences, including time series, trajectories,
audio frames and video embeddings. Each sample has shape `[T,D]`; a collection
has shape `[...,N,T,D]`. Feature extraction and feature units remain user-owned.
DTW preserves monotone time alignment; point-set comparisons answer a different
question because they discard the ordering.

```python
import torch
import torchpairwise as tp

x = torch.tensor([[[0.], [1.], [2.]], [[0.], [2.], [99.]]])
lengths = torch.tensor([3, 2], dtype=torch.int64)
cost = tp.dtw_distances(x, x1_lengths=lengths, squared=True)
loss = tp.soft_dtw_divergences(x, x1_lengths=lengths, gamma=0.5)
```

Only leading batch axes broadcast. The two collections can have different
padded time dimensions, but their feature dimensions must match. Optional
`x1_lengths` and `x2_lengths` are int64 tensors on the input device, with shapes
`[...,N]` and `[...,M]`. Their batch axes broadcast with the input batch axes,
but cannot introduce more batch axes than the inputs already have. Each length
must lie in `[1,T]`; empty sequences are rejected. Frames beyond a valid prefix
are ignored, may contain NaNs or infinities, and have exactly zero gradients.
Valid frames must be finite. Empty sample collections and empty batches produce
correctly shaped, differentiable empty outputs.

When the second collection is omitted, the comparison is against the first
collection and `x1_lengths` also supplies `x2_lengths` unless explicitly
specified. The output forms are:

| Form | DTW function | Soft-DTW function | Output |
| --- | --- | --- | --- |
| All pairs | `dtw_distances` | `soft_dtw_divergences` | `[...,N,M]` |
| Aligned samples | `paired_dtw_distances` | `paired_soft_dtw_divergences` | `[...,N]` |
| Self upper triangle | `condensed_dtw_distances` | `condensed_soft_dtw_divergences` | `[...,N*(N-1)//2]` |

Paired inputs require equal sample counts. Condensed functions take one
collection and `x_lengths`; entries follow row-major `i<j` order. They compute
these pairs directly without forming a square output.

## Cost, Paths and Warping Window

The fixed frame cost is **squared Euclidean**:

`c(i,j) = sum_d (x[i,d] - y[j,d])**2`.

Paths start at `(0,0)`, end at `(length1-1,length2-1)`, and use steps `(1,0)`,
`(0,1)` or `(1,1)`. DTW minimizes the **sum** of frame costs along a path. No
path-length averaging or sequence-length normalization is applied.
`dtw_distances(..., squared=True)` returns that minimum sum. The default
`squared=False` returns its square root. Neither convention is a triangle
metric; repeated frames can give zero cost between distinct sequences.

`window=None` allows every such path. A non-negative integer `window=w` imposes
`abs(i-j)<=w` on every cell, using the original frame indices. This is an
unscaled Sakoe-Chiba band, including when the lengths differ; it is not a band
around a rescaled diagonal. In particular `abs(length1-length2)>w` is infeasible.
An infeasible pair returns `+inf` with zero whole-pair gradients for both APIs.
With `window=0`, equal-length comparisons use only the diagonal.

## Debiased Soft-DTW

Raw Soft-DTW `S_gamma(x,y)` replaces the predecessor minimum in the DTW
recurrence by

`softmin(a,b,c) = -gamma * log(exp(-a/gamma) + exp(-b/gamma) + exp(-c/gamma))`.

Equivalently, it is the negative scaled log-sum-exp over all path costs.
`gamma` is a required positive finite real scalar, measured in squared-feature
units. It must be representable as a positive normal value in the computation
dtype. A tensor gamma, including one requiring gradients, is not accepted.
Smaller gamma approaches hard DTW; gamma is not an iteration tolerance.

Raw Soft-DTW can be negative, including on the self diagonal. The public API
returns the debiased divergence

`D_gamma(x,y) = S_gamma(x,y) - 0.5*S_gamma(x,x) - 0.5*S_gamma(y,y)`.

The same gamma and window apply to all three terms. Self terms are computed
once per sample per call and reused across partners. Output is unrooted,
unclamped and unnormalized. Identical valid sequences produce exactly zero
with zero whole-pair gradient, even when their padding differs. The divergence
is symmetric and is not a triangle metric. Nonnegativity and separation
results for Soft-DTW divergences depend on conditions on the ground cost and
alignment family. This squared-Euclidean API makes no blanket nonnegativity or
separation guarantee for arbitrary data and window choices; it does not silently
clamp signed results or apply an invalid square root.

## Differentiation, Dtype and Working Memory

Inputs support float32 and float64 on CPU and CUDA. Half and bfloat16 inputs
require autocast and produce float32; computations run with autocast disabled.
Only first derivatives are supported. Hard DTW is piecewise differentiable:
adjoints split equally among tied predecessor cells, defining a subgradient
policy at alignment ties. Rooted zero costs have zero gradients. Soft-DTW uses
stable shifted exponentials and differentiates both cross and self terms.

`chunk_size` is a positive integer (default 16), bounding sample pairs in a tile
**per batch**. Diagonal wavefronts evaluate all cells in parallel within each
tile using C++/ATen operations. Forward retains two
wavefronts with width `min(T,S)` and the sequence inputs. Backward recomputes the
quadratic wavefront table for one tile, then propagates adjoints explicitly.
This avoids retaining a Python autograd graph for every cell or a quadratic
alignment table for every collection pair. Output assembly, input storage,
batch size and the chunk's backward table still consume memory. Time remains
quadratic in sequence lengths. Specialized fused kernels may be preferable for
long sequences; no CUDA speedup claim is made without benchmarking.

The typed `sequence_comparisons` dispatcher and C++ entry point own all three
output modes, padding sanitization, recurrence and first backward. They are
usable from standalone LibTorch without Python initialization; see the
[native API guide](native-api.md) and
[standalone example](../examples/libtorch/sequence_comparisons.cpp).

References:

- Cuturi & Blondel (2017), [Soft-DTW: a Differentiable Loss Function for Time-Series](https://proceedings.mlr.press/v70/cuturi17a.html).
- Blondel, Mensch & Vert (2021), [Differentiable Divergences Between Time Series](https://proceedings.mlr.press/v130/blondel21a.html).
