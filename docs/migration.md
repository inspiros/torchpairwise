# Moving from the Published 0.3 Release

This guide covers migration from 0.3 to 1.0. New APIs require a 1.0 source
installation or wheel; already published 0.3 wheels do not include them.
Check the imported package's `__version__` and `__file__` when comparing builds.
See the [release notes](../CHANGELOG.md) for the scope of the update.

## Build Against Your Installed PyTorch

Install your chosen compatible PyTorch first, then build with
`python -m pip install --no-build-isolation --no-deps .`. The native extension
must match the PyTorch minor used to compile it; rebuild after changing that
minor. The tested stacks are PyTorch 2.12.1 on CPU/CUDA and 2.15.0 on CPU.
Latest-PyTorch CUDA validation is still outstanding. See the
[build instructions](building.md) for prerequisites and configuration.

## Choose the Output You Need

The documented `pairwise_distances(X,Y=None,metric="euclidean")` frontend is now
exported alongside `paired_distances` and `condensed_distances`; `cdist` and
`pdist` retain their existing signatures/defaults. See [API naming](api-design.md).

Existing named vector functions retain their all-pairs meaning: `[N,D]` and
`[M,D]` produce `[N,M]`. The development API also exposes corresponding-row
paired results and condensed self comparisons:

```python
import torchpairwise as tp

matrix = tp.cdist(X, Y, metric="canberra")        # [N,M]
paired = tp.paired_distances(X, Y, metric="canberra")  # [N], same N required
condensed = tp.pdist(X, metric="canberra")        # [N*(N-1)//2]
```

Use `pairwise_kernels`, `paired_kernels` and `condensed_kernels` for similarities.
Paired execution computes corresponding rows directly; condensed execution
computes only pairs `i<j` in row order. Asymmetric SNR and directed Hausdorff
are rejected in condensed mode.

Leading batch axes broadcast consistently in the public Python functions.
Feature counts must match; paired sample counts must also match. A singleton
sample axis does not broadcast into multiple corresponding samples. Inputs
share dtype/device, except for the documented autocast promotion.

## Stream the Result Your Algorithm Needs

Construct a lazy pairwise object with `tp.lazy(X, Y, metric=...)`, or wrap an
existing tensor with `tp.as_lazy`. Custom comparisons use `tp.lazy(X, Y, comparison)`;
prepared feature comparisons use `tp.lazy(prepared)`.

C++ callers can use `torchpairwise::streaming` from `<torchpairwise.h>` or
`<streaming.h>` with the installed include directory. It re-exports existing
typed factories/classes and adds short source factories. The `ops` namespace
and serialized dispatcher schemas remain unchanged. See the
[native namespace guide](native-api.md#streaming-namespace).

Existing dense comparisons retain their meaning. New [operator bindings](operators.md)
evaluate a comparison only when a terminal consumer needs tiles:

```python
A = tp.lazy(X, Y, metric="euclidean")
nearest = A.topk(8, largest=False)  # values/indices, no dense N*M matrix
average = A.mean(-1)              # tensor [...,N], reduce M
```

Kernel metrics also provide vector geometry methods. Use `tp.as_lazy` to apply
common consumers to an existing dense tensor; its storage is still required.

Residual/image sources retain event axes. `residual_operator(X,Y).var(-1)`
contracts D lazily; `image_map_operator(...).mean((-2,-1))` retains channels
lazily. After full event contraction, scalar consumers reduce N/M. Selecting a
sample axis on an event source is terminal, including joint sample/event
statistics. `dim` always refers to the current shape. The
[axis table](operators.md#dimensions-and-event-reductions) records different
method defaults and keepdim support.

Consumers include self by default. Use `topk(k, largest=False, exclude_self=True)`
for nearest neighbors excluding the diagonal. `min(dim)` and `max(dim)` return
values and indices; use `amin(dim)` and `amax(dim)` for values alone.

## Review Behavior Affected by Correctness Fixes

| Area | Development behavior |
| --- | --- |
| Default kernel gamma | Polynomial, sigmoid, RBF and Laplacian use `1/D`, including batched inputs. |
| Inferred statistics | Self standardized Euclidean and Mahalanobis infer detached statistics independently within each batch. Cross-input and paired calls require explicit `V`/`VI`. |
| Nearest-neighbor masks | Select exactly `k` eligible columns; `k=0` selects none. Ties follow column index. Self entries are excluded when Y is omitted, included for explicit Y, with `exclude_self` overriding that choice. |
| Radius masks | Use the inclusive `distance <= epsilon` threshold and the same self policy. |
| Empty samples/batches | Return correctly shaped empty outputs and support connected zero gradients where applicable. Feature axes and point clouds must remain nonempty. |
| Validation | Wrong ranks, mismatched sample/features, invalid domains and incompatible kwargs produce explicit errors. Previously accepted invalid calls may now fail. |
| CUDA streams | Native kernels use the current PyTorch stream and guard the input device, including backward. |
| Weighted Minkowski at `p=inf` | `wminkowski_distances(..., p=inf, w=w)` returns the maximum absolute difference over coordinates with `w > 0` (SciPy's convention). The 0.3 release returned `max(w * abs(x - y))`; multiply coordinates by `w` first to reproduce it. |
| Native input errors | Native operators raise ordinary synchronous errors for invalid eager values on CPU and CUDA, rather than CUDA device-side assertions. Compiled graphs and CUDA graph capture keep tensor assertions. |
| Mixed precision | The float32 autocast policy for comparison schemas is registered natively, so C++/LibTorch callers get the same promotion as Python. |

Numerical boundary conventions are specified in the [API contracts](api-contracts.md),
including zero-distance subgradients, undefined boolean formulas and cosine zero
vectors. Do not assume every distance family has a zero diagonal for every input.

## Keep Point Sets Distinct from Vectors

Existing Hausdorff cloud layouts `[N,P,C]` and `[B,N,P,C]` keep their meaning.
Directed Hausdorff's named function returns `(distances, x_indices, y_indices)`;
`cdist(..., metric="directed_hausdorff")` returns distances only. The witness
indices address each sample's flattened point axes.

For spatial feature sets, select the layout explicitly:

```python
# X: [B,N,C,H,W], Y: [B,M,C,H2,W2]
D = tp.hausdorff_distances(X, Y, batched=True, feature_dim=2)
```

This compares unordered C-dimensional descriptors, without adding pixel positions
to them. `batched=False` selects `[N,*points,C]` with zero batch axes;
`batched=True` selects one batch axis. Omitting it preserves the existing
`[...,N,P,C]` interpretation. These layout options apply to point-set functions.

Symmetric Hausdorff and Chamfer support all three comparison modes. Per-cloud
prefix lengths ignore dense padding; no sparse tensor is needed. Chamfer sums
the two directional means and defaults to Euclidean point distances; use
`squared=True` for squared Euclidean. Optional `chunk_size` trades recomputation
for lower intermediate storage during training. See the [point-set guide](pointsets.md).

## Mixed Precision and Compilation

Under autocast, floating metrics/kernels promote half/bfloat16 operands to
float32; double inputs stay double. Gradients return through the casts. Direct
half/bfloat16 execution outside autocast is operation-specific. Compiled training
is supported through tested native primitives and public routes; condensed/tiled
sample loops may specialize on shape. Higher derivatives, vmap and other
`torch.func` transforms are outside the current integration contract. See
[AMP and compilation](pytorch-integration.md).

Built-in metrics and reductions execute through native C++ implementations.
Use the public named API or the complete comparison selectors for rational
quadratic, Matérn, Hellinger, histogram intersection and point-set options.
The legacy `cdist`/`pdist` schemas retain their original name and parameter scope;
the native `builtin_comparisons` selector covers the complete vector/cloud
registry. Rebuild the extension and C++ consumers when upgrading.

## Native C++ Source Conventions

Use exported free functions such as `torchpairwise::ops::euclidean_distances`
instead of the old `*_functor::call` classes. Function names and signatures,
Python APIs and registered operator schemas are preserved; the functor classes
are removed. Rebuild C++ consumers against the new headers/library. Public C++
operator calls now pass through the same typed dispatcher as `torch.ops`, so
backend overrides apply at that boundary. See the [native API guide](native-api.md).

Extensions built on an installed wheel can use `torchpairwise.get_include()` and
`torchpairwise.library_path()` instead of hard-coded package paths. Header
includes are `<torchpairwise.h>` or `<ops/<name>.h>`; the
`<torchpairwise/ops/...>` spelling in earlier documentation did not resolve
against either the wheel or the CMake install. Redistributable extensions should
call operators through the dispatcher instead
([building on torchpairwise](native-api.md#building-on-torchpairwise-from-c)).


## Unified Selectors and Single Samples

Python and native cdist/pdist now use one role-aware catalog: 68 distance/
divergence/cost entries (59 condensed) across all modalities; all kernel mode
selectors cover13 kernels. Full canonical names and aliases work consistently.
Python cdist now permits omitted Y for self comparison. Defaults retain their
previous formulas. Separate legacy native name/option allowlists are removed.
Native schemas keep their operator names but expand optional parameters and
add Tensor-list overloads for Gaussian/mixture collections. Rebuild C++ consumers
and re-export/recompile serialized graphs; old binary/schema compatibility is
not promised across this change. Positional C++ option overloads still forward
to the common selector. Use ComparisonOptions for additional parameters.

`compare(...,mode="single")` and `single_comparison(x,y,metric=...)` compare one
sample per leading broadcast batch. Sample layouts omit N/M; optional lengths
omit that axis too. Generic compare now returns only the value Tensor for directed
Hausdorff, matching all other generic selectors. The named directed Hausdorff
function still returns witnesses. ComparisonInfo adds role/sample_ndims, and all
entries advertise single mode. Custom user callback wrappers are unchanged.
