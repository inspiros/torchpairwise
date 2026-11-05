<div class="eyebrow">Streamed operators</div>

# Feature and Image Axes

Scalar comparisons return one value per pair. Residual and image-map sources
retain multiple values for each pair, motivating feature/spatial tiling.

| Source | Logical shape | Event axes |
|---|---|---|
| Feature residual | `[...,N,M,D]` | D |
| Image map | `[...,N,M,C,Hout,Wout]` | C, Hout, Wout |

Residuals use X−Y, with identity, abs or square transforms. Image sources expose
aligned Y−X, absolute error, squared error, SSIM or dissimilarity maps according
to their selected comparison. They do not align, resize or register images.

## Event-Only Reductions Stay Lazy

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], Y:[M,D]
G = tp.residual_operator(X, Y, transform="abs",
                         chunk_size=64, feature_chunk_size=32)  # [N,M,D]
A = G.mean(dim=-1)                          # lazy [N,M]
values, indices = A.topk(3, largest=False)  # [N,3]
joint = G.mean(dim=(-2, -1))                # Tensor [N], joint over M,D
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// X:[N,D], Y:[M,D]
tp::ResidualOperator G(X, Y, "abs", 64, 32);    // [N,M,D]
auto A = G.mean(-1);                            // lazy [N,M]
auto [values, indices] = A.topk(3, -1, false);  // [N,3]
auto joint = G.mean(at::IntArrayRef{-2, -1});   // Tensor [N]
```

</div>
</div>

The basic reducers sum/mean/var/std/min/max/prod/all/any apply to current logical
axes. Selecting only events keeps the source lazy; once N or M is selected, the
result is an ordinary Tensor and later operations do not fuse back into it.

An image chain can reduce Hout/Wout first, leaving channels, then reduce C,
then select over M. `maps.mean(dim=(-2,-1)).mean(dim=-1).amax(dim=-1)` therefore
means average spatial values, average channels, then the largest reference score.
It is not interchangeable with a joint maximum over all axes.

Residual feature top-k and image-event top-k return terminal selected values
and indices along one integer axis. Their outputs are Tensors, not new lazy
sources. Scalar median/quantile require all event axes to be contracted first.

## Regional Execution

Large image reductions can evaluate only the spatial regions needed for their
output and replay regions during backward. `feature_chunk_size` similarly slices
residual D. This bounds generated pair-event workspaces; source images, collection
features, output tensors and gradients still exist. Some event chains retain
whole earlier-stage maps; chunk controls do not guarantee a fixed total peak.
See [event shape rules](https://github.com/inspiros/torchpairwise/blob/master/docs/operator-event-shapes.md)
and [image-map contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/image-map-operators.md).
