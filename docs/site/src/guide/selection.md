<div class="eyebrow">Getting started</div>

# Comparison Selection

`compare` covers all scalar roles: distances, divergences, costs, kernels,
scores and statistics. Full canonical names are clearest across families.
Modes and parameters are checked against one native catalog in both languages.

| Selector | Catalog scope |
|---|---|
| `compare` | 87 entries, all paired/single; 76 condensed |
| `cdist` / `pairwise_distances` | 68 distances, divergences and costs |
| `paired_distances` | Same distance role, paired samples |
| `pdist` / `condensed_distances` | 59 supported symmetric distance entries |
| `pairwise_kernels`, `paired_kernels`, `condensed_kernels` | 13 kernels, including Gaussian-mixture overlap |
| `lazy` | The 87 scalar formulas, streamed |

Counts include mathematically equivalent named entries, not just distinct
formulas. `list_comparisons(family=..., role=...)` exposes names, aliases,
sample layouts, modes and accepted parameters in Python; use the searchable
[catalog](../metrics/index.md) in either language.

## Named Functions and Aliases

Unprefixed named functions retain all-pairs behavior. Some also expose
`paired_` / `condensed_` functions. Use the generic mode selector for consistent
access rather than guessing a named function. `single_comparison` is a shorthand
for `compare(...,mode="single")`, not another numerical implementation.

Shared short names are role-dependent. `cosine` means distance in `compare` or
a distance selector, similarity in a kernel selector or `lazy`. Use `cosine_distances`
and `cosine_similarity` when switching roles. MMD kernel parameters have their
own [conventions](../metrics/mmd_distances.md).

## Defaults and Options

`compare`, `single_comparison`, `pairwise_distances`, `paired_distances` and
`lazy` default to Euclidean. `cdist`, `pdist` and
`condensed_distances` retain Minkowski with p=2. Default values agree, but
accepted parameters differ: specify a Minkowski metric when passing p.

Python uses named keyword parameters; native C++ uses `ComparisonOptions`.
Omitted parameters differ from explicitly supplied ones. Parameters unused by
a metric are errors. Scalar controls, metric names and modes are structural
choices; they are not learnable tensor parameters.

All generic selectors return one Tensor. Named directed Hausdorff additionally
returns witness indices; use the named API when those are needed.

## Native Calls

Python cdist/pdist call the same native schemas as C++ wrappers. Gaussian/mixture
inputs use collection overloads. Family numerical operators provide shared
forward and first-backward execution. Native opaque operators and metadata
support compilation/export; see [deployment](deployment.md).
