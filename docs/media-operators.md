# Streamed Image and Audio Comparison Operators

Image and audio scalar comparisons use the same `tp.lazy` facade:

```python
# Images:[...,N,C,H,W], reference images:[...,M,C,H,W]
A = tp.lazy(images, references, metric="ssim",
                         data_range=1., window_size=11, chunk_size=16)
weighted_scores = A.matmul(values)  # values:[...,M,K] -> [...,N,K]
column_means = A.mean(dim=-2)
form = A.bilinear(left, right)

# Aligned waveforms:[...,N,C,T], references:[...,M,C,T]
B = tp.lazy(audio, reference_audio, metric="log_spectral",
                         n_fft=512, hop_length=128, chunk_size=16)
reverse_product = B.T.matmul(query_values)
```

Supported comparisons are PSNR, SSIM, multiscale SSIM, SI-SDR, signal SNR,
log-spectral distance, spectral convergence and multiresolution STFT distance.
Names, parameters and collection layouts follow `compare` and the named
[image](image-comparisons.md) and [audio](audio-comparisons.md) functions. Image
scores require an explicit `data_range`. STFT resolution triples retain their
Python tuple/list form, including default hop/window lengths.

All 21 [common scalar consumers](pairwise-operators.md#operation-coverage)
apply, including top-k, softmax products and normalized weighted statistics.
Logical transpose, diagonals and repeated/empty selected blocks use the same
interface. Collection
and coefficient batch axes broadcast. `A.T` preserves the original comparison's
argument order: for directional audio comparisons, swapping estimates and
references would give a different matrix. Scores retain their original units
and direction; no conversion into a distance or positive kernel is implied.
In particular, perfect-image PSNR remains positive infinity.

## Preparation, Precision and Memory

Validation and numeric preparation are shared with the canonical native family.
Per-image SSIM statistics/pyramids and per-waveform centering/STFT features are
computed once per collection per method call. Backward accumulates feature
cotangents across pair tiles, then differentiates preparation once per operand.
Equal self inputs reuse forward preparation. Repeated methods prepare again;
later tensor mutations are visible. There is no persistent feature cache.

Native feature storage packs each sample's features into one contiguous feature
axis; variable image pyramid/STFT layouts are unpacked as views during tile
execution. This supports any valid scale/resolution count, without padding every
level to the largest spatial size. Feature storage and gradients still cost
memory proportional to the collected features. `chunk_size` bounds outer sample
and product channel tiles, not a fixed total-byte budget. Image windows and
transport/FFT/other inner workspaces retain their own storage requirements.
Products and forms do not retain the full `[N,M]` comparison matrix or an
intermediate `A @ right` for a bilinear form. Explicit `block` requests allocate
the selected block output.

Float32/float64 and autocast promotion follow the named functions. Internal SSIM
and audio arithmetic retains its canonical double precision; each scalar entry
is converted to the public dtype before matrix operations and winner selection.
This matters for min/max ties on float32 scores. Inputs must be finite, including
when the opposite operand or selection is empty. PSNR can produce infinite
scores from valid finite inputs. Standard nonfinite arithmetic applies to their
products and sums; `exclude_self` can remove the positional diagonal of a square
operator before reduction. Higher derivatives remain unsupported.

## Deployment

Forward and first backward are C++/ATen with no Python numerical callback. Use
`.matmul()` in compiled/exported modules and construct the binding inside
`forward`. Dynamic sample and batch axes are supported; channel/spatial/time
feature dimensions are specialized in the exported operator. Export separate
models when those event dimensions differ. This specialization does not change
the named dense functions' existing shape contracts.

C++ provides `ImagePairwiseOperator`, `AudioPairwiseOperator` and overloaded
`media_pairwise_operations` from `<ops/media_operations.h>`.
Construct the operator with tensors, a metric name, its `ImageComparisonOptions`
or `AudioComparisonOptions`, and optional `PairwiseOperationOptions` for matrix
controls. `matrix.chunk_size` overrides the family's default; the vector/cloud
`matrix.comparison` packet must remain unset.

See [the example](../examples/media_operators.py). Unreduced pixel error/SSIM
maps remain a separate [image-map API](image-maps.md).

## Whole-Image Scores versus Image Maps

`tp.lazy(...,metric="ssim",data_range=...)` has scalar entries and
shape `[...,N,M]`; its `mean(-1)` reduces references.
`image_map_operator(...,comparison="ssim",data_range=...)` retains local
`[...,N,M,C,Hout,Wout]` entries; its `mean(-1)` reduces window-map width.
Partial event stages stay lazy, full event contraction produces the common
scalar interface, and selecting N/M returns a terminal tensor. Use the
[image-map guide](image-map-operators.md) for current-shape axes and joint
statistics. `image_maps` remains the API for an explicitly materialized map.
