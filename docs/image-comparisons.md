# Whole-Image Quality Comparisons

PSNR, SSIM, and multiscale SSIM compare aligned whole images. A color image is
one sample: spatial pixels are not point-set samples, and the implementation
does not match, shift, register, or resize them. Larger scores indicate greater
similarity. These scores are neither promised triangle-inequality distances nor
positive-semidefinite kernels, and do not enter `cdist`, `pdist`, or the vector
metric registry.

| Comparison | All pairs | Aligned pairs | Condensed self pairs |
| --- | --- | --- | --- |
| PSNR | `psnr_scores` | `paired_psnr_scores` | `condensed_psnr_scores` |
| SSIM | `ssim_scores` | `paired_ssim_scores` | `condensed_ssim_scores` |
| MS-SSIM | `multiscale_ssim_scores` | `paired_multiscale_ssim_scores` | `condensed_multiscale_ssim_scores` |

Every input uses the explicit shape `[..., N, C, H, W]`. Leading batch axes
broadcast; channels, height, and width must match exactly. All-pairs calls accept
an optional second input and return `[..., N, M]`. Omitting it compares the first
input with itself. Paired calls require both inputs, equal sample counts, and
return `[..., N]`. Condensed calls take one input and return
`[..., N*(N-1)//2]`, ordered `(0,1), (0,2), ..., (1,2), ...`. A single image
therefore needs shape `[1, C, H, W]`, not `[C, H, W]`.

```python
import torchpairwise as tp

# predictions: [8, 3, 256, 256], references: [8, 3, 256, 256]
scores = tp.paired_ssim_scores(predictions, references, data_range=1.0)
all_pairs = tp.psnr_scores(predictions, references, data_range=1.0)
within_predictions = tp.condensed_multiscale_ssim_scores(
    predictions, data_range=1.0
)
```

`data_range` is a required positive finite Python real scalar for every call.
Use `1.0` for a declared unit intensity range, or `255.0` for intensities in the
usual 8-bit scale after conversion to floating point. The same declared range
must apply to all compared images. The code never infers a range from either
input or from a pair, clamps raw pixels, converts color spaces, downloads model
weights, or guesses a tensor layout. Finite values outside the declared range
are accepted, including negative values; the caller chooses their meaning.

Both inputs must have the same device and dtype. Float32 and float64 work on CPU
and CUDA. Under autocast, float16 and bfloat16 inputs are promoted to float32;
the image computation runs with autocast disabled and returns float32. SSIM and
MS-SSIM use float64 intermediates, then cast scores to the promised output dtype;
PSNR computes in float32 or float64 directly. Half inputs outside autocast are
rejected. Noncontiguous tensors and broadcast batch
views are supported. Sample axes and batch axes may be empty, with correctly
shaped outputs connected to zero input gradients. C/H/W must remain nonempty,
and SSIM window-size requirements still apply to empty collections.

## PSNR

PSNR is measured in decibels:

\[
\operatorname{PSNR}(x,y)
=10\log_{10}\!\left(\frac{R^2}{\frac1{CHW}\sum_{c,h,w}(x_{chw}-y_{chw})^2}\right)\]

Here \(R\) is `data_range`. Reduction includes every channel and pixel. An
identical pair returns positive infinity, with a chosen zero derivative for the
whole pair with respect to both inputs. This is a boundary convention: PSNR has
no finite mathematical derivative at zero error. Error scaling before the
logarithm avoids directly squaring very small residuals; a dyadic input
downscale prevents finite opposite-sign pixels from overflowing subtraction.

## SSIM

The defaults are `window_size=11`, `sigma=1.5`, `k1=0.01`, and `k2=0.03`.
The window size must be positive and odd, sigma/K1/K2 positive and finite, and
H/W each at least the window size. The discrete Gaussian has coordinates
\(-\lfloor w/2\rfloor,\ldots,\lfloor w/2\rfloor\), is normalized to sum one,
and is applied separably with **valid convolution**, without padding.

Each channel is processed independently using Gaussian-weighted population
moments, with no sample-covariance correction:

\[
l=\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1},\qquad
cs=\frac{2\sigma_{xy}+C_2}{\sigma_x^2+\sigma_y^2+C_2},\qquad
C_1=(K_1R)^2,\quad C_2=(K_2R)^2\]

SSIM averages \(l\,cs\) over valid window locations, then averages the channel
scores equally. It does not multiply separately averaged luminance and contrast
scores. Negative SSIM values are retained. Identical pairs return exactly one,
with zero whole-pair gradients.

For numerical stability, the implementation scales each image channel by its
own detached magnitude, caches centered moments, and reconciles magnitudes
within each pair before forming score ratios. The complete moment and score
calculation uses float64 so large intermediate derivatives at tiny stabilizers
can combine before gradients return to float32 inputs. It clips
roundoff-negative variances to zero, and forms the equivalent stabilizer ratios
in log space. This avoids
directly squaring extreme finite scalar ranges or intensities. If both terms
of a ratio underflow to zero, its limiting equal-zero value is one with zero
derivative. As with other floating-point reductions, extreme scale differences
within one image pair can still lose information.

## Multiscale SSIM

MS-SSIM uses the same Gaussian window and constants, with default `weights`:

```python
(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
```

Weights must be a nonempty tuple or list of positive finite real scalars. They
are used exactly as supplied, without renormalization. There is one scale per
weight. Between scales, the code average-pools 2 by 2 blocks with stride 2 and
no padding. Odd final rows/columns are dropped. For \(L\) scales, H/W must each
be at least `window_size * 2**(L-1)`: the five-scale defaults require at least
176 by 176 pixels. There is no automatic scale reduction for smaller images.

For channel \(c\), it computes

\[
\prod_{j=1}^{L-1}\max(\operatorname{mean}_{h,w}cs_{j,c},0)^{w_j}
\;\max(\operatorname{mean}_{h,w}(l_{L,c}cs_{L,c}),0)^{w_L},
\]

then averages over channels. Nonpositive factors map to zero with zero
derivative before taking powers; this avoids complex values and singular
fractional-power gradients. This is score-factor clipping, with no clamping of
raw pixels. A one-scale weight `(1.0,)` thus equals nonnegative SSIM per channel,
followed by channel averaging. Identical pairs return exactly one with zero
whole-pair gradients. Other libraries differ in padding, odd-size pooling,
covariance correction, channel reduction, or negative-factor policy; match
these conventions before comparing numbers.

## Computation and Gradients

`chunk_size=16` is a positive integer controlling how many image pairs per batch
are evaluated at once. All-pairs and condensed calls process one source image
against a target chunk; paired calls chunk the aligned sample axis. No full
`N*M*C*H*W` pair-image tensor is constructed. Condensed calls evaluate only the
strict upper triangle. SSIM image pyramids, local means, and local variances
are computed once per image and reused across its partners.

The implementation runs in C++/ATen and supports standalone LibTorch through
`image_comparisons` in `ops/image_comparisons.h`. Python named functions validate
arguments and dispatch to the same native implementation. See the
[LibTorch example](../examples/libtorch/modality_comparisons.cpp).

Native autograd retains the collection operands and replays the image features
once per collection during backward. It differentiates pair-local residuals and
cross moments one tile at a time, accumulates feature gradients, then propagates
through each collection's features once. This avoids retaining every pair's
spatial maps. Collection features, their gradients, and the output still require
memory: `chunk_size` bounds pair intermediates, while the all-pairs output remains
quadratic. First derivatives are supported; higher derivatives raise an explicit
error. Exact identity and nonpositive MS-SSIM factors use the boundary choices above.
Float64 SSIM intermediates also require more memory and may run more slowly on
devices with limited float64 throughput.

Learned models and checkpoint management are outside this library's scope.
Dedicated LPIPS/DISTS wrappers are skipped for now; callers can use their own
implementations or compatible [custom comparisons](custom-metrics.md). Feature
caching must be explicit. FID/KID concern distributions of features rather than
individual image pairs. See [learned representations](learned-comparisons.md)
for the existing model-free Gaussian comparison and the efficiency boundary.

For pixelwise residual/error maps, unreduced SSIM maps and streamed reductions
over reference images, see [image maps](image-maps.md).
