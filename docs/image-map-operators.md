# Lazy Image-Map Event Reductions

For source selection and shared axis rules, start with the [operator guide](operators.md).

`image_map_operator` binds aligned pairwise image maps with logical shape
`[...,N,M,C,Hout,Wout]`. It retains references to the inputs and constructs no
pairwise map until a consumer runs. Inputs are `[...,N,C,H,W]` and
`[...,M,C,H,W]`; leading batch dimensions broadcast.

```python
G = tp.image_map_operator(X, Y, comparison="absolute_error", chunk_size=4)
worst_pixel = G.max(dim=(-3, -2, -1))  # lazy [...,N,M]
nearest = worst_pixel.topk(8, largest=False)
channels = G.mean(dim=(-2, -1))      # lazy [...,N,M,C]
channel_worst = channels.max(-1)    # lazy [...,N,M]
nearest_channel_worst = channel_worst.topk(8, largest=False)
reference_average = channels.mean(-2)  # tensor [...,N,C]; -2 is now M
joint_variance = G.var(dim=(-4, -2, -1), correction=0)  # tensor [...,N,C]

S = tp.image_map_operator(
    X, Y, comparison="ssim", data_range=1., window_size=11, chunk_size=4,
)
local_variation = S.var(correction=0)  # lazy [...,N,M]
pooled = S.mean()                    # lazy [...,N,M]
smoothed = (gamma * pooled).softmax_matmul(V)
```

The scalar expressions support the existing scalar algebra and
[common consumers](pairwise-operators.md#operation-coverage), including top-k,
matmul, forms and N/M statistics. Build the binding inside `forward` when using
`torch.compile` or `torch.export`. The named whole-image comparisons remain
available through `pairwise_operator`; use them when their scalar formulas are
already the desired score.

## Explicit Event Shape and Bounded Scope

`G.shape` includes the two sample axes and all three event axes;
`G.event_shape` is `(C,Hout,Wout)`. Pixel maps retain H/W. Valid-window SSIM has
`Hout=H-window_size+1` and `Wout=W-window_size+1`.

| Current shape | N axis | M axis | Remaining event axes |
|---|---|---|---|
| `[...,N,M,C,Hout,Wout]` | -5 | -4 | -3, -2, -1 |
| After spatial mean: `[...,N,M,C]` | -3 | -2 | -1 |
| After channel contraction: `[...,N,M]` | -2 | -1 | None; common scalar consumers apply |
| After spatial keepdim: `[...,N,M,C,1,1]` | -5 | -4 | -3, -2, -1, including the singleton axes |

The reducers `sum`, `mean`, `var`, `std`, `min`, `max`, `prod`, `all`, `any`
accept an integer, tuple or list of current logical dimensions. An event-only
reduction stays lazy and preserves the remaining event shape. Contracting every
event axis with `keepdim=False` produces a scalar expression. `keepdim=True`
retains singleton event axes and therefore an image-map operator.

Omitting `dim` contracts **all currently remaining event axes**, preserving the
original full-event convenience API. Explicit `dim=None` or an empty list/tuple
selects N, M and every current event axis. Batch axes are always preserved.
Positive indices include leading batch axes; negative indices are relative to
the current shape. Repeated dimensions and batch selections raise errors.

Selecting N or M returns an ordinary tensor computed by native sample tiling.
Other selected event axes participate in the same reduction. For example,
`G.var((-4,-2,-1), correction=0)` computes one variance over M/H/W for each N/C;
it does not compute variance of independently pooled variances. Later tensor
operations do not fuse back into this consumer. Event stages run in the order
written: width mean followed by height max differs from a joint mean or max.

`tp.image_map_operator(...).reduce(...)` supports reference-only
sum/mean/min/max retaining C/H/W. Calling `image_maps(...).mean(...)` returns an
ordinary tensor and materializes the maps.

After event contraction, the resulting scalar expression has shape
`[...,N,M]`: its `dim=-1` now refers to M, and `dim=-2` to N. There is no
sample-transpose method on the uncontracted image source yet. Contract first,
then use the scalar expression's `.T`.

See [event shapes](operator-event-shapes.md) for the shared axis contract.
General vector/event algebra remains outside this API.

## Top-K Along One Event Axis

`topk` follows PyTorch's one-integer `dim` convention. It returns terminal
`values` and int64 `indices` tensors, replacing the selected current event axis
with k. Batch, N/M and other event axes persist:

```python
G = tp.image_map_operator(X, Y, comparison="absolute_error", chunk_size=4)
columns = G.topk(2, dim=-1)       # [...,N,M,C,H,2], two columns per row/channel
channels = G.mean((-2,-1))       # lazy [...,N,M,C]
strongest = channels.topk(2)     # [...,N,M,2], two channels per pair
```

Indices refer only to the selected axis, after any preceding reductions. The
column example selects two columns separately within each retained row/channel;
it does not select the globally largest pixels across H/W. Dimension lists,
joint-axis selection, batch axes and direct N/M selection on an image-event
source are unsupported. Contract all events into a scalar expression before
selecting image neighbors along N/M.

Require `0 <= k <= axis_length`. `k=0` and empty sample/batch collections yield
empty results. `largest=False` selects the smallest entries. Ties use the lower
axis index across region boundaries; NaNs rank highest. This tie rule is more
specific than `torch.topk`'s unspecified tied indices. The existing finite-input
image validation remains in force; ranking rules apply to computed map/program
entries. `sorted=False` permits
sorted output. Floating values have native first gradients; indices and integer
count programs are nondifferentiable. Boolean entries are rejected, as in
PyTorch. Integer counts remain int64 throughout selection.

Raw maps merge candidates from bounded C/H/W regions, without allocating the
full pairwise map. Backward scatters gradients into matching regions and reuses
the existing pixel/SSIM VJPs, including nonfinite source chain rules. Ordered
reduction programs reuse the regional first contraction and existing program
evaluator, then select from its remaining events within bounded sample tiles.
Those program tiles can still need substantial retained-event workspace; if one
pair's reduced event grid is large, it is not split by this terminal selector.

The selected output and indices need their natural storage. Large k or large
retained event axes can keep them large. Candidate sort workspace, packed image
features, input gradients and program replay also need memory. There is no fixed
whole-application memory budget. Results are tensors: subsequent tensor methods
do not fuse back into the streaming source. Higher derivatives are unsupported.

## Map and Numerical Conventions

`comparison` selects `residual` (Y-X), `absolute_error`, `squared_error`, or
`ssim`. These share [image_maps](image-maps.md)'s native formulas and SSIM
options, including all-one maps and zero pair gradients for identical images.
Signed residual orientation differs from `residual_operator`'s X-Y definition.
No padding, channel averaging, color conversion or clipping is implicit.
SSIM requires data_range; SSIM-only options are rejected for pixel maps.

Map arithmetic uses float64, then each map entry is rounded to the public
float32/float64 dtype before its event contraction. Floating event reductions
accumulate those entries in float64 and round each stage to the public dtype. This preserves
float32 map tie decisions, while sums/products/moments need not be bitwise
identical to reducing the map directly in float32. Half/bfloat16 inputs require
CPU/CUDA autocast and produce float32 outputs.

`var`/`std` use a finite real correction defaulting to 1; set 0 for population
moments. The correction applies once across the axes selected in that stage or terminal
reduction. Terminal moments merge count/mean/centered-square state across tiles.
Float64 terminal and regional moments keep an anchor and mean offset separately,
including in backward. Small local event var/std center their float64 inputs
before calling ATen; finite constant `1e308` pixel maps then have zero population variance
and zero first derivatives. Float32 keeps its existing arithmetic and state
size. Public map rounding remains in place. These paths do not add full-range
overflow fallbacks: map transforms, differences and squared deviations can still
overflow, and nearly cancelling gradients can lose relative accuracy. See the
[statistics numerical contract](operator-statistics.md#numerical-and-memory-contract).
Min/max return values, propagate NaNs and send gradients to the first C/H/W
row-major witness over the selected axes rather than splitting ties. all/any use
numeric nonzero truth and produce nondifferentiable bool entries. Subsequent
sum/prod promotes these to int64 and preserves integer accumulation. mean/var/std
require floating entries. Scalar expressions using floating weights can still
differentiate those weights.

Event product stages follow ATen's product VJP on the recomputed entries,
including its nonfinite behavior. Terminal streamed products use the shared
residual reducer's exclusive-product convention: the cotangent at a factor is
the upstream gradient times the product of all other factors. Zero, infinity
and NaN counts are tracked separately.

Images must be finite, dense float32/float64 with matching dtype/device and
nonempty equal C/H/W. Maps can overflow when their public dtype cannot
represent the computed error. Sample collections and broadcast batches may be
empty; scalar consumers retain their usual empty/exclusion contracts. Higher
derivatives are unsupported.

## Memory and Native Execution

Numerical forward/backward are registered C++/ATen operations. Collection SSIM
features are prepared once per source operand in each consumer call. A replayed
backward accumulates packed feature gradients, then differentiates preparation
once per operand. No Python tile callback or hidden persistent cache is used.
Mutations remain visible on later consumer calls.

The source executes its ordered event stages inside each pair-map tile. Fully
contracted entries feed the common scalar consumer; terminal reductions merge
output-shaped state over sample tiles. It retains packed collection features and requested consumer
state, not full computed N*M*C*Hout*Wout maps or an N*M score matrix. Selected
blocks and large top-k requests still require their natural output memory.
Large pixel/SSIM map tiles evaluate their arithmetic in channel/spatial regions,
with valid-window halos for SSIM. Backward replays those regions instead of
retaining a complete map-formula graph. The SSIM identity predicate still belongs
to the whole image pair, including when a smaller patch happens to match.
Small maps retain the original local ATen graph to avoid replay overhead. The
current internal switch is more than 4,194,304 public map entries per sample
tile; it is a heuristic, not a user memory budget.

For a large tile whose **first event stage is sum, mean, var, std, min, max,
prod, all or any**, the
native core contracts the selected C/H/W dimensions directly from regions into
reduction state. It avoids the complete public map, double reduction input and
map cotangent for that stage. Remaining event dimensions retain their natural
output-shaped state; subsequent ordered stages run on that contracted result.
This includes spatial mean followed by channel max and full-event variance.
The C++ full-event convenience API and older serialized image sources share
this optimization.

Direct terminal sample/event reductions with no preceding event stage also
merge regions straight into output-shaped state. All nine reducers use this
path for large tiles, including joint M/H/W variance and reductions that retain
full C/H/W outputs. Their backward replay forms only regional cotangents.
A retained output still needs its natural storage.

Identity image-residual (`Y - X`) event sum/mean and direct joint sample/event
sum/mean use shape-only adjoints. Contracted C/H/W axes stay singleton while
incoming cotangents are normalized, rounded to the public map dtype and summed
across samples. Only then are they broadcast into input-shaped packed gradients.
This avoids converting an expanded pairwise pixel cotangent. Retained image
axes still use bounded regions; input gradients and forward search/reduction
storage remain. Prior event stages and transformed maps keep their existing
ordered chain rules.

With float32 N=5/M=7 RGB 129x263 images and sample width 4, a development
measurement reduced backward-only additional peak allocation from 67 to 17 MiB
for global mean and 54 to 17 MiB for joint M/H/W sum. Full-call peaks fell from
85 and 73 MiB respectively to 37 MiB. Backward times were 2.0 versus 2.4 ms
for global mean and 2.7 versus 2.2 ms for joint sum. Smaller event-only tiles
retained their existing ATen path and allocations. One warmup and three timed
repetitions establish these workload-specific memory savings; latency varied.

Regional absolute- and squared-error backward reuse their freshly computed
residual buffer for the local derivative and cotangent product. Caller-owned
cotangents and saved state are not used as scratch. The public-dtype rounding
boundary, double arithmetic and sample sums remain; all derivative entries are
evaluated, including unselected zero cotangents multiplied by infinite residuals.
This removes separate derivative/product buffers without a new CUDA kernel.

For float32 N=M=8 RGB 256x256 images and sample width 8, this buffer reuse
reduced event-mean backward additional peak allocation from 192 to 144 MiB for
both absolute and squared error; full-call peaks fell from 240 to 192 MiB.
Absolute-error global mean fell from 192 to 144 MiB full-call peak, and
squared-error joint M/H/W variance from 204 to 180 MiB. Full event-mean times
were 11.7 versus 11.6 ms for absolute error and 13.7 versus 11.5 ms for squared
error. Other measured timings stayed close. These development measurements used
one warmup and three repetitions; they do not establish a general speedup.

CUDA regional pixel-map generation also combines double subtraction, optional
absolute value or square, and public-dtype rounding in one kernel. Reduction
state can receive a double promotion of the explicitly rounded entry directly,
avoiding intermediate double maps and float32 cast buffers. This preserves
rounding before moments, extrema selection and truth tests, including float32
subnormals. Paired and Cartesian regions read broadcast strides; CPU builds use
the same rounding boundaries through ATen. Forward contractions, regional public
maps and replay share this path. Requested outputs, prepared features and input
gradients retain their natural storage costs.

A subsequent comparison on the same float32 N=M=8/RGB 256x256/chunk-8 workload
measured full-call absolute-error event mean at 15.5 versus 11.8 ms and squared
error at 13.8 versus 9.8 ms after regional map fusion. Full-call CUDA launches
fell from 530 to 494, with 24 fewer dtype-copy operations. Peak allocation stayed
unchanged because later gradient work dominated. These one-warmup/three-repeat
measurements describe this workload, rather than a general latency guarantee.

For identity image residuals (`Y - X`), regional event min/max and direct joint
sample/event min/max backward scatter cotangents at the saved first winning
pixels. They avoid expanding a regional selection mask or replaying the map
just to recover that mask. Retained image axes are visited in bounded output
blocks, and repeated pixel gradients accumulate in double after public-dtype
cotangent rounding. Input-sized packed gradients and forward winner search
remain. Scalar consumers can still replay the local forward search in backward;
this optimization does not remove all replay. Absolute-error, squared-error and
SSIM maps retain their existing chain rules, including exceptional-value terms.

In a CUDA development measurement with float32 N=5/M=7 RGB 129x263 images
and sample width 4, the saved-witness adjoint reduced backward-only additional
peak allocation from 79 to 10 MiB for global max and joint M/H/W min. Full-call
peaks fell from 98 to 44 and 68 MiB respectively. Global-max backward took
3.6 versus 2.2 ms; joint M/H/W backward took 3.5 versus 3.3 ms. Small event-only
tiles stayed on their existing ATen graph, with unchanged allocation. These
results used one warmup and three timed repetitions on the same development
GPU; they establish workload-specific memory savings, not a general speedup.

First-event prod also contracts regions and replays bounded map/cotangent
regions. CUDA combines all product-state fields in one reduction for sufficiently
long event rows, splitting long rows into small partials with a fixed-order
merge. Short rows retain ATen reductions. Its state retains the raw product, exceptional-factor counts and finite
product/sign information, with a log-magnitude fallback for finite products
that overflow or underflow. It preserves the event stage's ATen quotient versus
exclusive-product branch, including when a zero in another output row selects
the exclusive branch. Terminal products keep their existing merge-state gradient
convention. As with other tiled floating reductions, grouping and extreme
finite overflow/underflow can differ from a dense product; bitwise equivalence
and identical overflow thresholds are not promised. Regional min/max preserves
the first logical witness across region boundaries, including NaNs, ties after
public-dtype rounding and signed zeros; all/any returns nondifferentiable boolean
results. Packed collection features, their gradients, identity checks, consumer
state and convolution workspace also remain. There is no fixed total-memory
bound. `chunk_size` bounds sample tile widths; smaller values may still be
needed for large images. Specialized factorized outer-product contractions
are unaffected.

In a development measurement with float32 RGB 512x512 images, N=M=8 and
chunk_size=8, absolute-error spatial mean -> channel max -> global sum used
672 MiB of peak additional CUDA allocation versus 1,056 MiB with the preceding
regional-map implementation. Forward plus backward took 53 ms versus 48 ms.
Full-event variance used 684 MiB versus 1,632 MiB, at 86 ms versus 62 ms. SSIM
for the spatial chain used 2,983 MiB versus 3,167 MiB, at 592 ms versus 588 ms;
its prepared features account for much of the remaining allocation.
Measurements used PyTorch 2.12.1+cu126 on a Quadro RTX 6000, one warmup and two
timed repetitions. Peak allocation is above persistent inputs and includes
preparation, outputs and backward work; it is not pure scratch space or a
portable performance guarantee. Regional replay trades recomputation for memory.

For direct joint M/H/W variance on the same absolute-error image workload,
regional terminals reduced additional peak allocation from 1,632 to 492 MiB
with sample width 8. Forward plus backward took 41 ms versus 44 ms; two timed
repetitions are insufficient to establish a latency improvement. Width 2 stayed
at 456 MiB because its tile was already below the regional threshold. The
requested output and the prior numerical/gradient checks agreed.

A first-event product benchmark used float32 RGB 256x256 images, N=M=8,
sample width 8 and residuals near one to avoid an all-zero underflow result.
Fused CUDA state and adjoint kernels avoid the temporary mask/arithmetic tensors
and repeated reductions of the first regional implementation:

| Full-event product implementation | Extra peak allocation | Forward + backward |
|---|---:|---:|
| Previous full event tile | 420 MiB | 12 ms |
| First regional implementation | 267 MiB | 68 ms |
| Regional replay with fused CUDA adjoint | 164 MiB | 47 ms |
| Fused CUDA state, log shortcut and adjoint | 164 MiB | 16 ms |

Spatial product followed by channel mean improved from 69 to 53 to 15 ms,
at the same 164 MiB peak. Scalar results and gradient summaries agreed; separate
dense tests checked full input gradients. Normal finite partial products need
one logarithm each; overflowing or subnormal partials replay only their bounded
slice to sum factor logs. The device-side zero branch and finite range-loss
fallback remain. Direct terminal sum/mean also skips recomputing contracted
values when only their shape is needed for the cotangent. Scalar consumers
reuse each backward tile's evaluated source values for its expression gradient.
CUDA product kernels read rounded public float32/float64 map entries directly
and promote factors in registers, keeping state and cotangent arithmetic in
float64. This avoids a separate float64 copy of each float32 product region;
public map and incoming-cotangent rounding still apply. Other map/dtype copies
remain performance targets. These paths trade
extra work for lower memory; the environment, warmup/repetition count and
allocation conventions match the measurements above.

A subsequent comparison combined per-tile source reuse with direct float32
product inputs. On the same product workload, forward plus backward improved
from 15.5 to 11.5 ms and additional peak allocation fell from 164 to 160 MiB.
For SSIM followed by event mean and sample sum, source reuse improved 142 to
121 ms with peak allocation unchanged at 1,161 MiB. These later measurements
used one warmup and three timed repetitions; they do not establish a general
speedup across sources or devices.

Python and native LibTorch use the same typed expression frontend with a flat
list of source tensors. All source operands therefore participate in dispatcher
autograd discovery, including source-only losses and top-k after opaque SSIM
preparation. Earlier nested-packet schemas remain registered for serialized
programs. Numerical cores remain opaque for tracing. Native usage:

```cpp
#include <ops/image_map_operations.h>
torchpairwise::ops::ImageMapOptions options;
options.data_range = 1.;
options.window_size = 11;
options.chunk_size = 4;
auto maps = torchpairwise::ops::image_map_operator(x, y, "ssim", options);
auto scores = maps.mean();  // ScalarExpression
auto [best, ids] = scores.topk(8);
auto spread = maps.var({-3,-2,-1}, 0.).matmul(values);
// Explicit conversion keeps C++ return types static; Python infers the result.
auto channels = maps.events().mean({-2,-1});
auto [channel_values, channel_ids] = channels.topk(2);  // terminal [...,N,M,2]
auto worst_channel = channels.max({-1}).scalar();
auto neighbors = worst_channel.topk(8);
auto per_query_channel = channels.reduce_samples("mean", {-2});
auto joint = maps.events().reduce_samples("var", {-4,-2,-1}, false, 0.);
```

No learned model or new dependency is introduced. CUDA builds include an
internal fused product-state and product-adjoint kernels. CPU builds, and short
CUDA state reductions, retain the shared ATen algorithms. Both use the same
public native operator schemas.

See the [runnable example](../examples/image_map_operators.py).
