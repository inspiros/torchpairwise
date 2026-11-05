# Pairwise Outputs with Multiple Feature Axes

The general pairwise shape is `[..., N, M, *event_shape]`. Leading dimensions
are batches, N/M index the two collections, and trailing event dimensions
belong to one comparison result. A scalar comparison has an empty event shape;
a residual vector has event shape `(D,)`. Multiple event axes are useful and
already occur in specialized APIs.

| Interaction | Logical unreduced shape | Current support |
|---|---|---|
| Scalar distance or kernel | `[...,N,M]` | Common scalar operators and algebra |
| Coordinate residual | `[...,N,M,D]` | Lazy residual source and reductions |
| Pixel error or SSIM map | `[...,N,M,C,Hout,Wout]` | `image_maps`, reference reductions, and lazy partial/full-event and joint sample/event reductions with `image_map_operator` |
| Residual outer product | `[...,N,M,D,D]` | Specialized kernel-weighted consumer returns `[...,N,D,D]`; no public lazy outer-product source |
| Local point-to-point costs between clouds | `[...,N,M,P,Q]` | Intermediate of some point-set algorithms; no public lazy local-cost source |
| Matrix-valued kernel | `[...,N,M,Dout,Din]` | Future use case; existing named kernels return scalars |

The last two examples describe possible sources, not currently exposed generic
operators. Gaussian/covariance inputs alone do not imply matrix-valued output:
their existing distance functions still return one scalar per pair.

See [image maps](image-maps.md), [lazy image-map contractions](image-map-operators.md), [residual operators](residual-operators.md) and
[kernel moments](kernel-moments.md) for implemented numerical contracts.

## Axes Must Be Explicit

If there are r event axes, N is at `-(r+2)` and M at `-(r+1)`. For an image map
with shape `[...,N,M,C,H,W]`, `dim=-1` is width, `dim=-2` is height,
`dim=-3` is channel, `dim=-4` is M and `dim=-5` is N. For a scalar operator,
`dim=-1` is M. Always interpret dim relative to the object's current shape.
A tuple/list selects several axes jointly for ordinary reductions, preserving
batch axes. `topk` follows PyTorch and accepts one integer axis only; joint-axis
top-k is outside the API. Residual sources select D, and image sources select
one current event axis. Both return terminal values and indices tensors.

Do not infer these roles from tensor rank alone. A source family knows its
event layout; a future generic tensor-valued source would need explicit
`event_ndim` or equivalent metadata. Event metadata must survive reductions,
so remaining event axes never silently become batch or sample axes.

Flattening all event coordinates is valid for a full elementwise event
contraction, such as a mean over C/H/W. Flattening is not a substitute for the
public shape contract: spatial pooling, channel-only statistics, local-window
SSIM, covariance structure and matrix contractions distinguish event axes.

## Streaming Boundary

The following contract applies to image-map event programs and guides future
bounded sources. Partial-event nodes and joint sample/event reductions are
implemented for image maps.

1. A reduction over event axes only stays local to each sample pair. Reducing
   all event axes produces a scalar operator compatible with existing scalar
   algebra, top-k and products. Reducing a strict subset preserves an explicit
   lazy node with the remaining event shape.
2. Selecting N or M is a terminal consumer returning an ordinary tensor, with
   unreduced event axes retained. Joint sample/event moments must reduce the
   combined observation set once, not compute moments of separately reduced
   moments. Later tensor operations do not fuse back into this consumer.
3. Event contractions execute in the written order. A mean over image width
   followed by a maximum over height differs from a joint reduction and must
   not be reordered or flattened into one statistic.
4. Specialized contractions may consume several axes together. They need an
   explicit mathematical and memory contract, rather than inheriting scalar
   `matmul` semantics by broadcasting. Feature/event top-k uses terminal tensors
   and indices along one selected axis; exact quantiles need explicit
   pass/workspace limits.

For a partial-event reduction of an image map G, spatial pooling has entries

\[
H_{ijc}=\frac{1}{HW}\sum_{h,w}G_{ijchw},
\]

retaining C. A subsequent channel contraction could produce a scalar score
\(A_{ij}=C^{-1}\sum_c H_{ijc}\), which could feed the existing scalar consumers.
Write this as `H = G.mean((-2,-1))`, then `A = H.mean(-1)`. In contrast,
`image_maps` returns a tensor and materializes the map. `tp.image_map_operator(...).reduce(...)`
streams references while retaining all C/H/W axes.

`G.var((-4,-2,-1), correction=0)` computes joint M/H/W variance and returns
`[...,N,C]`. `H.var(-2, correction=0)` computes M variance **after** spatial
pooling and generally has different values. Both are supported, with the
written stage order preserved. Event-only keepdim retains singleton event
metadata. Omitted dim selects current events; explicit None/[] selects all
logical axes. Residual top-k selects D; image-event top-k selects one current
event axis. Both return terminal values/indices. Event median/quantile remain
deferred; exact scalar median/quantile are available after full event contraction.

For a matrix-valued kernel, a useful dedicated contraction is

\[
U_{ia}=\sum_{j=1}^{M}\sum_{b=1}^{D_{in}}K_{ijab}V_{jb}\]

It returns `[...,N,Dout]` while streaming both M and Din. Such kernels are useful
for coupled vector fields and multi-output kernel learning. They are a separate
extension from the current scalar kernel product
\(U_{iq}=\sum_j K_{ij}V_{jq}\); this document does not claim existing support.

## Implementation Priorities

The aligned image-map source starts with `(C,Hout,Wout)` event metadata and
retains selected event axes through ordered native reduction stages. Terminal
N/M and joint reductions preserve pixel/SSIM conventions and existing schemas.
Large-map arithmetic/VJPs now use native spatial/channel regions with SSIM
window halos. First-event sum/mean/var/std/min/max/prod/all/any contracts directly
into regional state. Direct sample/event terminal contractions also merge
regions into output-shaped state and replay regional cotangents. Product replay
retains compact exceptional-factor state rather than full event buffers.
A general event graph, arbitrary pairwise callbacks, FFT/convolution nodes and
matrix-valued kernel families are not prerequisites for this bounded step.

Keep residual outer products on their specialized contraction path. Its
factorization avoids constructing pairwise D*D values; a generic event engine
must not replace it with a slower expansion. General outer-product sources or
matrix-valued kernels should follow a concrete workload.

Memory depends on all retained output and state axes. Pair tiles do not bound
total memory if each pair has a large event tensor: event axes also need bounded
workspace or a specialized factorization. An explicitly requested tensor still
requires its natural output storage. Built-in numerical forward/backward must
remain native C++/ATen, with typed schemas and compile/export metadata; Python
should only assemble the bounded source/consumer description.
