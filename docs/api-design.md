# API Names and Recommended Usage

A comparison formula, a pairing mode and an execution strategy are separate
choices. Use the same formula name when comparing their values or performance.

Lazy sources and consumers live at package level. Use `tp.lazy` for scalar
comparisons, `tp.as_lazy` for stored matrices, and `tp.residual_operator` or
`tp.image_map_operator` for event-valued sources. See the
[migration guide](migration.md#stream-the-result-your-algorithm-needs).

## Select the Output Mode

| Need | Recommended generic call | Output |
|---|---|---|
| Every X/Y sample pair | `compare(X,Y,metric=name)` | `[...,N,M]` |
| Corresponding samples | `compare(X,Y,metric=name,mode="paired")` | `[...,N]` |
| Each unordered self pair once | `compare(X,metric=name,mode="condensed")` | `[...,N*(N-1)//2]` |
| One x/y sample per batch | `single_comparison(x,y,metric=name)` or `compare(...,mode="single")` | `[...]` |
| A product, reduction or selection | `streaming.pairwise(X,Y,metric=name)` followed by a consumer | Consumer-dependent Tensor |

`compare` and `tp.lazy` span all scalar comparison families and accept
canonical full names from `list_comparisons()`. Full names are clearest across
families: `cosine_distances`, `cosine_similarity`, `ssim_scores`,
`gaussian_wasserstein_distances`. Short aliases are conveniences, not additional
formulas. `mode="paired"` requires Y; condensed accepts only X and a comparison
that provides a symmetric condensed mode. Leading batches are independent of
the pairing mode.

```python
name = "sqeuclidean_distances"
all_pairs = tp.compare(X, Y, metric=name)
aligned = tp.compare(X, Y, metric=name, mode="paired")  # equal sample counts
unordered = tp.compare(X, metric=name, mode="condensed")

A = tp.lazy(X, Y, metric=name)
row_sum = A.sum(dim=-1)
nearest = A.topk(8, dim=-1, largest=False)
```

The distance/divergence/cost trio is `pairwise_distances`, `paired_distances`,
`condensed_distances`. Its scope now spans all sample modalities. The kernel trio
is `pairwise_kernels`, `paired_kernels`, `condensed_kernels`, including mixture
overlap. All use the same native role/mode catalog as `compare`; canonical full
names and short aliases are accepted. `cdist`/`pdist` are compact all-pairs/
condensed distance entry points; Python now calls these same native schemas.
`cdist(X)` and `pairwise_distances(X)` support self comparison. See the
[coverage and routing guide](dispatchers.md#selector-coverage-and-its-justification).

`compare` is a useful role-neutral name within the torchpairwise namespace.
Keep its pairing mode explicit when leaving the all-pairs default.
`single_comparison` makes the fourth use case discoverable without generating
another large family of per-metric aliases. Named functions keep their current
all-pairs meaning; callers need not rewrite every metric with a verbose prefix.

Default metric keys differ historically: `cdist`, `pdist` and
`condensed_distances` default to Minkowski with p=2; `pairwise_distances`,
`paired_distances`, `compare`, `single_comparison` and `tp.lazy` default to Euclidean.
Their default unweighted formulas agree, but passing p without naming a
Minkowski metric need not be accepted by an Euclidean entry point. Specify the
metric when moving between interfaces.

## Named Functions

Named comparisons use their mathematical role in the suffix:

| Role | All-pairs name | Other modes where exposed |
|---|---|---|
| Distance | `gaussian_wasserstein_distances` | `paired_gaussian_wasserstein_distances`, `condensed_gaussian_wasserstein_distances` |
| Divergence | `soft_dtw_divergences` | `paired_soft_dtw_divergences`, `condensed_soft_dtw_divergences` |
| Score | `ssim_scores` | `paired_ssim_scores`, `condensed_ssim_scores` |
| Kernel | `gaussian_mixture_overlap_kernel` | `paired_gaussian_mixture_overlap_kernel`, `condensed_gaussian_mixture_overlap_kernel` |

The lack of a prefix means all-pairs. Preserve the existing `paired_` and
`condensed_` prefixes; another `all_pairs_` alias family is unnecessary.
`cosine_similarity` is an established suffix exception. Scores/similarities
are not automatically distances or positive-semidefinite kernels.

Not every vector comparison exposes individual Python `paired_*` or
`condensed_*` functions, even when a native mode or named LibTorch function
exists. Use the generic selector for uniform mode coverage rather than guessing
an attribute name. C++ uses typed family selectors for heterogeneous input
layouts; a native registered name is not automatically a Python top-level name.

Some named structured functions take means/covariances as separate parameters;
`compare` and operators use one tuple per collection. This groups the data that
represent one sample collection without a large universal C++ signature.

## Prefer Operators for Streaming

Factories bind an unevaluated source; ordinary methods consume it:

```python
K = tp.lazy(X, Y, metric="rbf_kernel", gamma=0.2)
product = K.matmul(V)
total = K.sum()                         # both N/M, preserving batches
mean_for_each_query = K.mean(dim=-1)
energy = K.bilinear(left, right)

G = tp.residual_operator(X, Y, transform="abs", feature_chunk_size=32)
mae = G.mean(dim=-1)                    # lazy scalar source [...,N,M]
robust_neighbors = mae.topk(4, largest=False)

maps = tp.image_map_operator(images, references, comparison="squared_error")
channels = maps.mean(dim=(-2,-1))        # lazy [...,N,M,C]
scores = channels.mean(dim=-1)          # lazy [...,N,M]
threshold = scores.quantile(0.9, dim=-1) # terminal [...,N]
```

Kernel metrics share the common interface and add coordinate geometry.
`tp.as_lazy` wraps a caller-owned dense matrix. Consumers recompute tiles;
a binding is not a cache. Construct it inside `forward` for compilation or export.

## Defaults and Return Types Matter

| Operation | Contract |
|---|---|
| Ordinary scalar reducers | `dim=None` selects N/M together; lists/tuples are joint reductions; batches persist |
| Residual reducers | Default dim=-1 contracts D lazily; selecting N or M is terminal |
| Image-map reducers | Omitted dim contracts current events; explicit None/[] selects all non-batch axes |
| Scalar median/quantile | One integer N/M axis, default -1; values only |
| Top-k | One integer axis, default -1, largest=True; values and indices |
| `min(dim)`, `max(dim)` | Values and indices; `amin` and `amax` return values only |

`median(dim)` returns values only, unlike PyTorch. `topk` returns
`.values` and `.indices`.

Operator consumers include self unless `exclude_self=True` is explicit.
Exclusion removes positional i=j entries of a square comparison, even for
separate X and Y tensors. Use explicit dims and exclusion in examples that move
between helpers. Event contractions and terminal consumers preserve the written
order; Tensor operations after a consumer do not fuse back into the source.

`chunk_size` controls outer sample tiles; `feature_chunk_size` controls residual
D slices. Some point/distribution operators expose `metric_chunk_size` or
`component_chunk_size` for inner work. These parameters control different axes;
none is a total-memory limit. See the [operator guide](operators.md).

## Formula Ambiguities to Avoid

- `cosine` means a distance in `compare`/`tp.lazy`, but similarity in
  a kernel selector. Use `cosine_distances` or `cosine_similarity` across families.
- MMD retains a historical `kernel="laplacian"` Euclidean-ground convention;
  `kernel="laplacian_kernel"` selects the Manhattan-ground registry kernel.
  Its RBF default scale also differs from the vector kernel's gamma=1/D.
  Use canonical kernel names and explicit gamma/scale when matching formulas.
- `soergel_distances`, `ruzicka_distances` and continuous Jaccard share a
  nonnegative-input formula. Some Boolean names also have equivalent formulas.
  Mathematical synonyms need not imply redundant implementations or that every
  catalog name is a different mathematical expression.
- Named directed Hausdorff retains witnesses. All generic selectors, including
  `compare`, return one distance/value Tensor. Use the named function for witnesses.
- Arbitrary Boolean predicates currently belong to the explicit Python callback
  reduction helpers. Operators' built-in all/any use nonzero truth; the finite
  scalar algebra does not yet provide arbitrary comparisons such as `A < t`.

No named metric formula or function was renamed. Native selector schemas were
expanded and generic Hausdorff now consistently returns values only; rebuild
consumers and re-export graphs. See [migration](migration.md#unified-selectors-and-single-samples).
Further renaming/deprecation should be an explicit migration.
