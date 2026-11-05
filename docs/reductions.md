# Dense Pairwise Outputs and Reduced Outputs

For `cdist(X,Y)`, the dense `[N,M]` matrix is the requested result. Every element
must be stored, so any implementation returning that tensor needs storage
proportional to N*M. Dense output is useful when callers inspect all comparisons,
reuse them in several computations, or pass a precomputed matrix to another
algorithm. It remains a central torchpairwise API.

In other workloads, the caller immediately reduces the matrix. It is then an
intermediate rather than the final result:

| Use case | Logical operation | Required final output |
| --- | --- | --- |
| Nearest neighbor matching | Minimum and argmin across Y | `[N]` distances and indices |
| Exact top-k neighbors | k smallest distances per row | `[N,k]` distances and indices |
| Kernel regression / kernel matrix-vector products | Sum K(X,Y)*V across Y | `[N,D]` for D values per reference sample |
| Kernel density estimation | Sum kernel weights across Y | `[N]` |
| Chamfer / Hausdorff of two clouds | Nearest point per source, followed by mean / max | One scalar per compared cloud pair |

For example, 100,000 queries and 100,000 references produce 10 billion entries:
**40 GB** for float32 distances. Keeping only ten neighbors needs about **12 MB**
for float32 distances and int64 indices, excluding inputs and working memory.
The dense matrix can be prohibitive even when the required result is small.

An exact reduction can process a tile of pairs, update running minima, top-k
candidates or sums, then discard that tile. It need not write the full matrix to
global memory. This reduces storage and memory traffic; exhaustive nearest
neighbor search can still evaluate every pair. Approximation or pruning are
separate algorithmic choices. Floating-point summation order and tie selection
can differ across implementations and must be considered when comparing results.

[KeOps](https://www.kernel-operations.io/keops/index.html) specializes in this
workflow: its LazyTensor records the pairwise formula, and computation happens
when a reduction such as sum, min or argmin is requested. It supports kernel
matrix-vector products, neighbor queries and autograd without materializing the
symbolic matrix. See the [official LazyTensor example](https://www.kernel-operations.io/keops/python/LazyTensor.html).
That does not remove the storage requirement if the requested result really is
the full dense matrix.

## The Two Levels of Pairing in Point-Set Distances

Point-set inputs `[N,P,C]` and `[M,Q,C]` have **two** comparison levels:

1. Compare every pair of cloud samples, producing the desired `[N,M]` output.
2. Within each cloud pair, compare P source points against Q target points to
   calculate the cloud distance.

The second level does not need to return `[P,Q]`. Our directed Hausdorff kernel
already reduces those distances directly. Chamfer and length-aware Hausdorff
now use a native nearest-point reduction, retaining distances and witnesses per
source point rather than all point pairs. Backward uses the selected witnesses.
The outer `[N,M]` cloud matrix is still returned normally.

This is not a general KeOps backend. Witness storage still scales with
N*M*(P+Q), with additional feature-sized backward workspace. Large outer cloud
collections may still need chunking. Paired cloud comparisons avoid the outer
square entirely; condensed comparisons store only unordered cloud pairs.

An optional `chunk_size` tiles outer cloud comparisons and checkpoints point-level
intermediates in training. It bounds each tile's workspace while still returning
the full cloud-distance result. Inputs/output storage remain, and recomputation
costs additional work. See the [point-set guide](pointsets.md) for the dense
layouts, lengths and chunking contract.

## Choosing the Reduction Entry Point

Use [lazy pairwise objects](operators.md) for reductions, products, selection,
statistics and weighting. Event-valued [residual](residual-operators.md) and
[image-map](image-map-operators.md) sources also support event-axis reductions.

## Streaming Neighbor and Kernel APIs

```python
match = tp.lazy(X, Y, "euclidean").min(-1)
distances, indices = tp.lazy(X, Y, "cosine").topk(8, largest=False)
self_neighbors = tp.lazy(X).topk(8, largest=False, exclude_self=True)
K = tp.lazy(X, Y, "rbf", gamma=0.2)
total = K.sum(-1)
weighted = K @ V
farthest = tp.lazy(X, Y).topk(3, largest=True)
minimum = tp.lazy(P, Q, "kl").amin(-1)
product = K.prod(-1)
```

All distance and kernel registry keys support streaming reductions: vector,
weighted, Boolean/equality, probability, missing-data, point-set and empirical
transport comparisons. Their existing parameters, normalization, dtype and
domain contracts apply. Structured Gaussian/mixture, image, audio, sequence
and token functions remain standalone and are not registry reduction inputs.
The existing neighbor-mask functions still return dense masks. No reduced
operation silently falls back to building the full pairwise matrix.

Point-set inputs use `[...,N,P,C]` and `[...,M,Q,C]`, with existing `batched`,
`feature_dim` and prefix-length options. The returned reference index identifies
a cloud, not a point within it. Shuffling/generators are unsupported in reduced
Hausdorff execution. Point-set inner solvers retain their existing memory and
execution costs: exact Wasserstein remains CPU-controlled, and finite-budget
Sinkhorn remains an estimate. Sliced projections are explicit and shared.

Standardized Euclidean variance and Mahalanobis inverse covariance are inferred
once from the complete original collection for omitted-reference calls. They
are detached as in the dense API. Explicit references still require `V`/`VI`.
Inference never uses an individual tile. Other tensor options, including
feature weights, histogram positions and projection matrices, retain their
existing gradient behavior.

Vector inputs `[...,N,D]` and `[...,M,D]` broadcast over batch axes. Neighbor
results are a `torch.return_types.topk` named tuple with `values` and `indices`, both
`[...,N,k]`; indices are int64 on the input device. `min(-1)` returns
values and indices with the last axis removed. Distances retain the selected metric's units.
Results are sorted by computed value; ties choose lower reference indices.
NaNs rank above finite values, as in `torch.topk`. Floating-point differences
across tile sizes/backends can change near ties.

`exclude_self=False` includes the diagonal by default. `exclude_self=True`
requires equal sample counts and asserts aligned samples; it does not detect
identity. Exclusion is structural: an excluded item cannot displace a valid
infinite/NaN candidate. Cosine retains the dense API's zero diagonal when the
same tensor object is supplied on both sides, including zero vectors.

`k` must be an integer, excluding bool, from zero through the eligible reference
count. Excessive k is rejected. Empty references require k=0; nearest-neighbor
search raises when no eligible reference exists, including singleton self
search. Empty query or batch axes return empty results. Shape/domain validation
still applies, and differentiable empty results have connected zero gradients.

`K.sum(-1)` has shape `[...,N]`. With
`values` of shape `[...,M,Cv]`, `K @ values` has shape `[...,N,Cv]`.
A vector `[M]` is also supported. The sample count M must match. Payload batch axes may introduce new broadcast axes.
Payloads must have the same dtype/device as the inputs. Sums include all self
terms and apply no normalization. Empty references return connected zeros.
Signed kernels/payloads and ordinary nonfinite arithmetic are preserved;
in particular a zero payload does not mask an infinite kernel entry.

## Basic Reference-Axis Reductions

`D.reduce(...)` accepts `"min"`, `"max"`, `"sum"`, `"prod"`, `"all"` or `"any"`.
With `dim=-1` it reduces the reference axis M,
returning `[...,N]`; `keepdim=True` returns `[...,N,1]`. Callers can further
reduce those outputs using ordinary torch operations. Self terms are included
by default; `exclude_self=True` removes the aligned positional diagonal and
requires N=M.

| Operation | Dense reference expression | Empty eligible references |
| --- | --- | --- |
| min | `D.min(dim=-1).values` | Raises ValueError |
| max | `D.max(dim=-1).values` | Raises ValueError |
| sum | `D.sum(dim=-1)` | Zero |
| prod | `D.prod(dim=-1)` | One |
| all | `D.bool().all(dim=-1)` | True |
| any | `D.bool().any(dim=-1)` | False |

`amin`/`amax` return values only and propagate NaNs. On ties, including multiple NaNs,
they choose the first reference for backward, matching indexed torch min/max
rather than the evenly split tie derivative of `amin`/`amax`. This differs
deliberately from neighbor ranking, which places NaNs last. Use `topk`
when indices or top-k results are needed. Product uses ordinary floating-point
multiplication, with one/multiple-zero gradient handling; it is not a log-domain
product. Tiled multiplication/summation can change rounding and underflow/
overflow compared with dense evaluation. Nonfinite values may produce NaNs.

Boolean-input metrics such as Jaccard produce **numeric distances**. Their
all/any reductions use nonzero truth, as does torch: zero is False and NaN is
True. For a similarity threshold, supply an explicit tile predicate:

```python
has_close_reference = tp.lazy(boolean_features, references, "jaccard").any(
    -1, predicate=lambda distances: distances < 0.2,
)
all_finite = tp.lazy(X, Y, "linear").all(-1, predicate=torch.isfinite)
```

The predicate receives a numeric pair tile and must return a dense Boolean
tensor with the same shape/device. It is allowed only with all/any, should be
pure and tile-independent, and is not exportable. Logical outputs
have no gradients. Every tile is evaluated; no data-dependent host-side early
exit is introduced. Empty outputs do not probe the predicate with fabricated
values. The same [feature wrapper](learned-comparisons.md) can compose an
encoder with one of these collection reductions.

## Memory, Gradients and Execution

`chunk_size=None` chooses a tile side of 128. A positive integer caps query and
reference tile sizes; payload channels are tiled too. Neighbor forward keeps
only a distance tile and running candidates, followed by `[...,N,k]` output.
Kernel forward keeps a kernel tile and output accumulators. Neither constructs
the full `[N,M]` intermediate. Broadcast inputs remain views, and batches are
visited individually rather than flattened into expanded input copies.

First backward saves original operands plus selected indices for neighbors,
and original operands/payloads for sums. Product additionally saves per-query
products and zero counts. Backward recomputes only selected neighbor/extremum
pairs or one pair tile at a time. Candidate autograd graphs and pair tiles are
not retained. Gradients accumulate into each original operand shape, including
repeated/broadcast references and tensor parameters. Selection
indices are held fixed; ties differentiate the chosen branch. Higher derivatives
are not supported. Payloads and the existing differentiable feature weights
receive gradients; scalar gamma follows the existing non-tensor convention.

Sum forward uses compensated cross-tile accumulation in the compute dtype.
Within-tile sums/matmuls retain PyTorch rounding. Cross-tile backward accumulation
uses float64 buffers before casting to each input dtype. Inputs, outputs,
gradients and metric-local feature workspace still consume memory; large k can
itself approach dense output size. This is not a constant-total-memory promise.

CPU/CUDA float32/float64 are supported for numeric metrics. Boolean/equality
metrics additionally accept their existing non-floating input dtypes and
retain the base comparison's numeric output dtype. Autocast promotes half/bfloat16 to
float32 for input/weight/payload computation; float64 stays float64. PyTorch's
deterministic-algorithm policy applies to the underlying operations, including
scatter/index accumulation. CUDA operations use the caller's current stream.

Built-in traversal, selection and first backward execute in C++ and use ATen
operations on CPU/CUDA. Native opaque operators support fullgraph compilation
and export with dynamic sample/batch dimensions; no dense pair matrix is added
by capture. Standalone LibTorch callers use `streaming_reductions` with typed
`StreamingReductionOptions`. Arbitrary predicate callbacks remain an explicit
Python path. This is tiled native composition; it does not depend on KeOps.
Benchmark equivalent dense-plus-reduction outputs before performance claims.
See the [example](../examples/reductions.py).

## Matrix-Free Kernel Methods

`K @ V` applies the kernel matrix to one or
more vectors: `V` has shape `[M,Cv]`, and the result is `[N,Cv]`. All registry
kernels and their typed parameters are accepted. Each iteration recomputes
kernel tiles, trading arithmetic for avoiding storage of the full Gram matrix.
First backward also replays tiles instead of retaining the matrix.

For kernel PCA, an iterative eigensolver needs products with the centered
operator `H K H`. Center `V`, apply `K @ V`, and center
the result. For kernel ridge regression, conjugate gradients can apply
`K V + regularization * V`. SVM prediction is a kernel product with signed
support-vector coefficients; training needs a solver designed for products or
on-demand kernel blocks. Kernel Fisher discriminant analysis similarly needs
an operator-aware solver. Existing estimators that require a dense precomputed
Gram matrix do not become matrix-free just by changing kernel evaluation.
Ordinary linear LDA does not need a kernel Gram matrix.

The [kernel-method example](../examples/kernel_methods.py) checks centered
products and a polynomial-kernel MMD statistic, including first gradients.
It uses dense matrices only as small validation oracles. For positive-semidefinite
kernels, biased MMD squared is the sum of the two within-set mean kernels minus
twice the cross-set mean. Three streamed sums compute it without a Gram matrix.
This example works on two `[samples,features]` collections; the dedicated
point-set `mmd_distances` also supports multiple clouds, prefix lengths and
pairing modes, using the same native named-kernel catalog subject to its
MMD validity restrictions.

### Other Kernel-Method Operations

A streamed product is the main primitive for iterative kernel ridge solves,
centered kernel PCA eigensolvers and kernel Fisher discriminant methods. Those
algorithms also need vector arithmetic, regularization and stopping criteria;
a kernel product does not turn an existing dense estimator into an iterative one.

Other useful access patterns include:

- **Diagonals:** `paired_kernels(X, X, kernel=...)` evaluates `K[i,i]`
  without the full Gram matrix, for trace estimates or diagonal normalization.
- **Selected blocks:** `pairwise_kernels(X[rows], Y[columns], kernel=...)`
  supplies SVM working sets or Nyström landmarks. Only that block is dense.
- **Quadratic/bilinear forms:** `K.quadratic(V)` and `K.bilinear(L, R)` contract
  kernel tiles directly, without retaining the intermediate `K @ V`.
- **Entrywise aggregates:** row sums, minima, maxima, products and boolean
  predicates use lazy reduction methods. Exact Frobenius norms, HSIC and cross-kernel
  alignment additionally need coordinated products of kernel entries; these
  are future specialized reductions, not currently inferred from two row sums.

Dedicated `mmd_distances(..., kernel="polynomial", squared=True, ...)` now
provides the same biased empirical statistic as three kernel sums, with cloud
layouts, prefix lengths and all-pairs/paired/condensed outputs. Its native
execution shares the kernel catalog. The separate signed `mmd_statistics` API
supplies the unbiased polynomial
statistic used by KID on caller-provided features. The
[kernel operator](kernel-operators.md) binds these products, forms, diagonals,
blocks and scalar/row/column reductions in one Python/C++ interface.


Float32/float64 product consumers share the corrected native scalar-product state and
PyTorch's zero-dependent entry derivative convention. The whole-input zero test
includes other output rows/batches; actual NaN/Inf factors are counted separately.
See [product semantics and legacy compiled graphs](pairwise-operators.md#reductions).
Finite intermediate overflow/underflow and source-specific nonfinite VJPs remain
separate limits. No new product option or dense intermediate is required.
