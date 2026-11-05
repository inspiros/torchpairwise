<div class="eyebrow">Getting started</div>

# Modes and Shapes

A metric describes how to compare two samples. A mode describes which samples
are compared. Batch axes are independent of that choice.

For a scalar comparison \(f(x,y)\):

\[
A_{ij}=f(X_i,Y_j),\qquad p_i=f(X_i,Y_i),\qquad
c_{ij}=f(X_i,X_j)\ (i<j),\qquad s=f(x,y)\]

| Mode | Vector inputs | Output | Meaning |
|---|---|---|---|
| <span class="badge mode-all-pairs">all-pairs</span> | `[...,N,D]`, `[...,M,D]` | `[...,N,M]` | Every X/Y combination; omitted Y means self comparison |
| <span class="badge mode-paired">paired</span> | `[...,N,D]`, `[...,N,D]` | `[...,N]` | Equal sample counts; both collections required |
| <span class="badge mode-condensed">condensed</span> | `[...,N,D]` | `[...,N*(N-1)//2]` | Unordered self pairs, no diagonal; supported symmetric comparisons |
| <span class="badge mode-single">single</span> | `[...,D]`, `[...,D]` | `[...]` | One sample per broadcast batch; both samples required |

Condensed ordering is `(0,1), (0,2), …, (1,2), …`. Do not interpret `pdist` as
corresponding rows. Use paired mode for that operation.

## What Is One Sample?

| Family | One sample | Collection X |
|---|---|---|
| Vectors, histograms | `[D]` | `[...,N,D]` |
| Point sets | `[P,D]` | `[...,N,P,D]` |
| Images | `[C,H,W]` | `[...,N,C,H,W]` |
| Audio | `[C,T]` | `[...,N,C,T]` |
| Sequences | `[T,D]` | `[...,N,T,D]` |
| Token sequences | `[T]` | `[...,N,T]` |
| Covariances | `[D,D]` | `[...,N,D,D]` |
| Gaussians | `(mean[D], covariance[D,D])` | `(means[...,N,D], covariances[...,N,D,D])` |
| Gaussian mixtures | `(means[K,D], covariances[K,D,D], weights[K])` | Add the N axis before each component's sample dimensions |

Point sets are dense tensors, not sparse storage. A cloud may contain coordinates
or feature vectors. Whole-image comparisons instead preserve aligned channels
and spatial axes; they do not turn pixels into unordered points.

## Leading Batches

Leading dimensions follow PyTorch broadcasting. All-pairs vectors shaped
`[2,1,N,D]` and `[3,M,D]` produce `[2,3,N,M]`. In single mode, `[B,D]` versus
`[D]` produces `[B]`; the same `[B,D]` in all-pairs mode is a collection of B
vectors. No mode is inferred from shape.

Sample and feature axes must meet the metric's contract. Paired sample counts
must match exactly; an N=1 collection does not broadcast to N corresponding
samples. Point counts or sequence lengths can differ where documented.

## Lengths and Structured Inputs

Point-set, sequence, token and mixture comparisons can accept valid prefix
lengths. Collection lengths end in `[N]`; single-mode lengths omit that axis:
use a scalar for one unbatched sample or `[B]` for B batched samples. Token
lengths may be zero; point distributions must be nonempty. Masked padding is
ignored according to the family contract.

Single point sets use canonical `[P,D]` samples. Flatten spatial point grids
first; omit the collection-oriented `batched` and `feature_dim` controls.
Use a tuple/ComparisonCollection for Gaussian or mixture components, not an
extra synthetic feature dimension. See each [metric page](../metrics/index.md).
