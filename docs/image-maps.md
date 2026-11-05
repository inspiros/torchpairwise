# Image Maps and Streamed Reference Reductions

Compare aligned images while retaining channel and spatial coordinates:

```python
maps = tp.image_maps(X, Y, comparison="squared_error")
# X:[...,N,C,H,W], Y:[...,M,C,H,W]
# maps:[...,N,M,C,H,W]

average_error = tp.image_map_operator(X, Y, comparison="absolute_error").reduce("mean", dim=-4)
# [...,N,C,H,W], without constructing the full map tensor
best_ssim = tp.image_map_operator(
    X, Y, comparison="ssim", data_range=1., window_size=11,
).reduce("max", dim=-4)
```

All batch axes broadcast. `comparison` selects `residual` (Y-X),
`absolute_error`, `squared_error`, or `ssim`. Pixel maps keep H/W. SSIM keeps
valid-window positions of size `H-window_size+1` by `W-window_size+1`, with C
unchanged. The SSIM Gaussian window, population moments, intensity range and
stabilizers match [the existing image scores](image-comparisons.md). `data_range`
is required for SSIM; window/range options apply only to SSIM. No padding,
channel averaging, color conversion or pixel clipping is performed.

Averaging the SSIM map over C and valid spatial positions reproduces
`ssim_scores`. Exactly identical whole-image pairs give an all-one SSIM map
with zero pair gradients, matching that API. Negative SSIM values are retained.
Multi-scale spatial map layouts remain future work; existing MS-SSIM scalar
scores keep their current definition.

## Pairing and Reductions

`image_maps(..., mode="paired")` requires two collections of equal sample count
and returns `[...,N,C,Hout,Wout]`. `mode="condensed"` requires one collection,
returns `[...,N*(N-1)//2,C,Hout,Wout]` in row-major i<j order, and excludes the
directed signed residual. Omit Y for self all-pairs maps. A dense map request
needs the complete output's natural memory even though work is tiled.

`image_map_operator(...).reduce(...)` streams sum, mean, min or max over **references only**.
Channels/spatial axes are retained, and batch axes broadcast and remain intact.
With `keepdim=True`, the result is `[...,N,1,C,Hout,Wout]`. Min/max choose the
first reference on a tie and propagate NaNs. Comparisons use the returned
map dtype, including float32 rounding, so their winners match materialized maps. Empty reference sums are zero;
empty means are NaN; empty reference min/max raise. Reductions include the
positional diagonal for self inputs. Product, logical and predicate reductions
are not currently part of this image-map API.

Min/max here choose a reference independently at every channel/pixel. They do
not select one best reference image by its whole-image score.

## Native Execution and Memory

Numerical forward and first backward execute in C++/ATen on CPU/CUDA. Inputs
must be finite float32/float64, with matching nonempty C/H/W and dtype/device.
Autocast promotes half/bfloat16 and produces float32. Computation uses float64
internally and casts final values back to float32/float64. Raw pixel maps can
still overflow when their requested output dtype cannot represent the result.

`chunk_size` bounds reference-image tiles; paired mode also tiles aligned rows.
SSIM features are computed once per operand in each forward/backward call.
Backward accumulates gradients of collection features while replaying pair
maps, then differentiates the feature transforms once per operand. No dense
pair-map tensor is retained for reduced execution. Collection/feature state,
output and gradients have their natural storage; min/max additionally retain
one int64 reference index per output element. Tile workspace scales with image
size, broadcast batches and chunk size. It is not a fixed total-memory budget.
There is no hidden cross-call cache. Higher map derivatives are explicitly
unsupported.

The functions support `torch.compile` and `torch.export` with native Fake
metadata. Standalone C++ uses `image_maps(x, y, comparison, mode, reduction,
ImageMapOptions)` from `<ops/image_comparisons.h>`; forward and
backward share the Python implementation's numerical path.

See [the runnable example](../examples/image_maps_and_moments.py).

A map has three event axes, C/H/W, after its two sample axes.
[image_map_operator](image-map-operators.md) supports lazy partial-event
reductions, full contraction into scalar entries, and terminal joint sample/event
reductions. Scalar entries support top-k, algebra and products. The functions on
this page retain their tensor outputs; see [multiple feature axes](operator-event-shapes.md).
