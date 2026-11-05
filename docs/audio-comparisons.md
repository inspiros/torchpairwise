# Audio Comparisons

Audio functions compare aligned waveforms with shape `[..., N, C, T]`. `N`
indexes recordings, `C` channels and `T` time samples. Mono inputs retain the
channel axis (`[N, 1, T]`). Both collections need matching nonempty `C` and `T`;
only preceding batch axes broadcast. Recordings may be noncontiguous. Sample
counts and batch dimensions may be empty.

All-pairs functions return `[..., N, M]`; omitting `x2` compares `x1` with
itself. `paired_` functions return `[..., N]` and require equal sample counts.
The symmetric `condensed_log_spectral_distances` returns the self-comparison
upper triangle in row-major `i < j` order. Directional functions have no
condensed variant. These functions have dedicated APIs, outside the vector
metric dispatch; `signal_snr_scores` differs from the existing vector
`snr_distances`.

`x1` is the **estimate** and `x2` the **reference**. Higher waveform scores are
better; lower spectral discrepancies are better. No sample rate is inferred or
required by these formulas. The caller owns loading, sample rate agreement,
resampling, channel correspondence, amplitude calibration, cropping, time
alignment, and preprocessing. No speech model, PESQ/STOI, perceptual guarantee,
phase alignment, or learned feature extraction is included.

## Waveform Scores

The formulas below operate independently per channel, then average **dB scores**
over channels. They do not sum channels before taking a logarithm. With
`zero_mean=True`, subtract each waveform channel's time mean first. This is the
default for SI-SDR and is disabled by default for conventional signal SNR.

Let `e` be the estimate, `r` the reference and `E(z) = sum_t z[t]^2`.
`signal_snr_scores` uses target `s = r`.
`si_sdr_scores` uses the least-squares projection
`s = (sum_t e[t]*r[t] / E(r)) * r`; when `E(r)=0`, set `s=0`.
The reported score for each channel is

```text
10 * (log10(max(E(s), energy_floor))
      - log10(max(E(e-s), energy_floor)))
```

`energy_floor` is a positive finite Python scalar (default `1e-8`) in **summed
squared waveform units**, applied separately to numerator and denominator.
It regularizes the conventional ratio. A perfect channel reports
`10*log10(max(E(s), floor)/floor)` instead of infinity. Two silent channels
report zero. A silent reference and nonzero estimate reports
`10*log10(floor/max(E(e), floor))`. An orthogonal SI-SDR estimate has the same
zero-target policy. A constant channel becomes silent when centering is on.
Floored scores lose exact scale invariance near silence; changing waveform
units or duration changes the meaning of the absolute energy floor. The
unfloored SI-SDR definition removes gain errors, whereas signal SNR retains
them. Both API contracts treat the second argument as the directional
reference, so neither exposes condensed output.

## STFT Protocol

`log_spectral_distances` and `spectral_convergence_distances` share these
keyword options:

| Option | Default | Meaning |
| --- | --- | --- |
| `n_fft` | 512 | Positive FFT size |
| `hop_length` | `max(1, n_fft//4)` | Positive frame stride in samples |
| `win_length` | `n_fft` | Positive window length, at most `n_fft` |
| `window` | `'hann'` | Periodic Hann, periodic `'hamming'`, or `'rectangular'` |
| `center` | `True` | Pad each time end by `n_fft//2` before framing |
| `pad_mode` | `'constant'` | Zero padding or `'reflect'` |
| `normalized` | `False` | When true, divide the FFT by `sqrt(n_fft)` |

A short window is centered with zeros inside the FFT frame, following
`torch.stft`. Spectra are complex one-sided FFTs of real inputs; comparisons
use their absolute magnitudes. There is no detrending, DC removal, power
squaring, frequency weighting, phase term or extra window-energy correction.
Only complete frames are retained after optional centering.
`center=False` requires `T >= n_fft`; centered reflection requires
`T > n_fft//2`. Centered constant padding also supports shorter waveforms.

For magnitudes `A` (estimate) and `B` (reference), log-spectral distance is

```text
mean_{C,F,frames} abs(ln(max(A, log_floor)) - ln(max(B, log_floor)))
```

This uses natural logarithms, magnitude units and L1 averaging. `log_floor`
is positive and finite, default `1e-7`. It is symmetric and zero for identical
magnitudes, but different waveforms can share magnitudes. It is not a metric
on waveforms. Both-silent input returns zero; a silent channel enters through
the explicit magnitude floor. Values below the floor have zero derivative
through the clamp.

Spectral convergence is

```text
||A - B||_F / max(||B||_F, norm_floor)
```

The Frobenius norms span **all channels, frequencies and frames together**.
`norm_floor` is positive and finite, default `1e-7`, in magnitude-norm units.
This is directed by its reference denominator. Both-silent input returns zero;
a silent reference and nonzero estimate returns `||A||_F / norm_floor`.

## Multi-Resolution Discrepancy

`multiresolution_stft_distances` and its paired variant use a nonempty
`resolutions` list/tuple of `(n_fft, hop_length, win_length)` triples. Each triple
uses the same window, centering, padding and normalization protocol above.
The defaults are `((512,128,512), (1024,256,1024), (2048,512,2048))`.
The scalar result is the arithmetic mean over resolutions of

```text
spectral_convergence(A, B) + mean(abs(ln(max(A, log_floor))
                                  - ln(max(B, log_floor))))
```

Every resolution and each of the two terms has equal weight. Duplicated triples
retain their repeated weight while reusing their transform. This explicit
unweighted sum is a discrepancy, generally directed by the convergence term;
it has no condensed variant. There is no hidden mel transform, perceptual
weighting, random resolution sampling, waveform term or sample-rate conversion.
FFT sizes refer to samples, so their physical time spans depend on caller-owned
sample rates.

## Dtype, Gradients and Feature Reuse

Inputs must be finite, dense strided CPU/CUDA tensors of the same dtype and
device. Float32 and float64 are supported. Float16/bfloat16 are accepted inside
CPU/CUDA autocast and produce float32 outputs; otherwise they are rejected.
Computation uses float64 internally, including the STFT, and returns the input
float32/float64 dtype (float32 for autocast half inputs). Gradients return in the
input dtype. This choice improves energy/floor arithmetic but can cost
substantial GPU throughput. Extreme float64 values may still overflow, and a
large result can overflow a float32 output. No performance advantage over
specialized audio libraries is claimed.

Ordinary PyTorch autograd reaches both inputs. Absolute-value, norm-zero,
projection-silence, and clamp boundaries use PyTorch's derivatives/subgradients;
these functions are not smooth everywhere. A positive floor does not make SI-SDR
projection continuous as reference energy approaches zero from arbitrary
directions. Two silent inputs and exact spectral equality have zero returned gradients.
The shared implementation runs in C++/ATen and supports standalone LibTorch
through `audio_comparisons` in `ops/audio_comparisons.h`. Python named functions
validate arguments and dispatch to the same native implementation. The
[LibTorch example](../examples/libtorch/modality_comparisons.cpp) exercises both
image and audio families. Higher derivatives raise an explicit error.

Within each forward call, the STFT of each original collection is computed once
per distinct resolution and reused for every partner. Explicitly passing the
same tensor object as both collections shares the forward features. Broadcasting
happens after extraction. Native autograd retains collection operands and replays
each collection's features once during backward. It differentiates pair tiles,
accumulates feature gradients, then propagates through the collection transforms
once. `chunk_size` (positive integer, default 16) bounds pair-feature
intermediates per batch. Collection features, their gradients, and output tensors
still consume memory, and batch size/channels/frames remain memory factors.
Features are not cached across calls. No persistent cache or user model is owned
by these functions.

See [the runnable example](../examples/audio_comparisons.py).
