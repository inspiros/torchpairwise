"""Aligned whole-image quality scores, separate from vector distance dispatch."""
import numbers

import torch
from ._validation import _check_chunk_size

from ._batching import validate_values
from ._batching import is_finite_real


_MS_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _positive_scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f'{name} must be a real scalar')
    if not is_finite_real(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return float(value)


def _prepare(x, y, mode, chunk_size):
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError('Image inputs must be tensors')
    if x.layout != torch.strided or y.layout != torch.strided:
        raise TypeError('Image inputs must be dense strided tensors')
    if x.ndim < 4 or y.ndim < 4:
        raise ValueError('Images must have shape [...,N,C,H,W]')
    if x.shape[-3:] != y.shape[-3:]:
        raise ValueError('Image C/H/W dimensions must match')
    if any(size == 0 for size in x.shape[-3:]):
        raise ValueError('Image C/H/W dimensions must be non-empty')
    if mode == 'paired' and x.shape[-4] != y.shape[-4]:
        raise ValueError('Paired image inputs must have equal sample counts')
    if x.device != y.device:
        raise ValueError('Image inputs must have the same device')
    if x.dtype != y.dtype:
        raise ValueError('Image inputs must have the same dtype')
    if x.device.type not in ('cpu', 'cuda'):
        raise ValueError('Image inputs must be on CPU or CUDA')
    allowed = ((torch.float32, torch.float64, torch.float16, torch.bfloat16)
               if torch.is_autocast_enabled(x.device.type) else (torch.float32, torch.float64))
    if x.dtype not in allowed:
        raise TypeError('Image inputs require float32 or float64 outside autocast')
    if type(chunk_size) is not int:
        raise TypeError('chunk_size must be an integer')
    _check_chunk_size(chunk_size)
    try:
        return torch.broadcast_shapes(x.shape[:-4], y.shape[:-4])
    except RuntimeError as exc:
        raise ValueError('Image batch dimensions are not broadcastable') from exc


def _window_options(x, window_size, sigma, k1, k2, weights):
    if type(window_size) is not int:
        raise TypeError('window_size must be an integer')
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError('window_size must be positive and odd')
    sigma = _positive_scalar(sigma, 'sigma')
    k1, k2 = _positive_scalar(k1, 'k1'), _positive_scalar(k2, 'k2')
    if weights is not None:
        if not isinstance(weights, (tuple, list)) or not weights:
            raise TypeError('weights must be a non-empty tuple or list of positive real scalars')
        weights = tuple(_positive_scalar(weight, 'weights') for weight in weights)
    levels = len(weights) if weights is not None else 1
    minimum = window_size * 2 ** (levels - 1)
    if min(x.shape[-2:]) < minimum:
        raise ValueError(f'Image H/W must be at least {minimum} for this window and scale count')
    return sigma, k1, k2, weights


def _call(x, y, *, mode, kind, data_range, chunk_size, window_size=11, sigma=1.5,
          k1=.01, k2=.03, weights=None):
    # Native validation is retained in traced graphs. Eager adapters preserve
    # Python exception types without tracing Python scalar math into Dynamo.
    if not torch.compiler.is_compiling():
        data_range = _positive_scalar(data_range, 'data_range')
        _prepare(x, y, mode, chunk_size)
        if kind != 'psnr':
            sigma, k1, k2, weights = _window_options(x, window_size, sigma, k1, k2, weights)
        validate_values(torch.isfinite(x).all() & torch.isfinite(y).all(), 'Image inputs must be finite')
    return torch.ops.torchpairwise.image_comparisons(
        x, None if y is x and mode != 'paired' else y, kind,
        'all_pairs' if mode == 'allpairs' else mode, data_range, chunk_size,
        window_size, sigma, k1, k2, [] if weights is None else list(weights))


def psnr_scores(x1, x2=None, *, data_range, chunk_size=16):
    """Whole-image PSNR in dB, returning ``[...,N,M]`` (higher is better).

    Inputs are finite float32/float64 tensors ``[...,N,C,H,W]`` on CPU/CUDA;
    only batch axes broadcast. Omit x2 for self comparisons. Half/bfloat16
    inputs require autocast and produce float32. ``data_range`` must be an
    explicit positive finite real scalar. No pixel clipping or range inference
    occurs. PSNR is ``10*log10(data_range**2/MSE)`` over all C/H/W entries;
    exact equality returns +inf with zero whole-pair gradients.

    ``chunk_size`` bounds pair-image intermediates per batch. Pair computations
    use native autograd replay; image-local state and the output still consume
    memory. These scores are not distance/kernel dispatch
    metrics. See ``docs/image-comparisons.md`` for all image conventions.
    """
    return _call(x1, x1 if x2 is None else x2, mode='allpairs', kind='psnr',
                 data_range=data_range, chunk_size=chunk_size)


def paired_psnr_scores(x1, x2, *, data_range, chunk_size=16):
    """Aligned PSNR ``[...,N]``; see :func:`psnr_scores`."""
    return _call(x1, x2, mode='paired', kind='psnr', data_range=data_range, chunk_size=chunk_size)


def condensed_psnr_scores(x, *, data_range, chunk_size=16):
    """Self PSNR in row-major i<j order ``[...,N*(N-1)//2]``."""
    return _call(x, x, mode='condensed', kind='psnr', data_range=data_range, chunk_size=chunk_size)


def ssim_scores(x1, x2=None, *, data_range, window_size=11, sigma=1.5, k1=.01, k2=.03, chunk_size=16):
    """Whole-image SSIM ``[...,N,M]`` using a valid Gaussian window.

    See :func:`psnr_scores` for image, dtype, batching and memory contracts.
    ``window_size`` is positive and odd; H/W must each be at least that size.
    ``sigma``, ``k1`` and ``k2`` are positive finite scalars. Gaussian-weighted
    population moments use C1=(k1*data_range)**2 and C2=(k2*data_range)**2.
    Average each channel's SSIM map over valid locations, then over channels.
    No padding, sample covariance correction, color transform or pixel clipping
    is applied. Negative scores are retained. Identity returns one with zero
    whole-pair gradients; the score is not a PSD kernel or a metric distance.
    Moments and scores use float64 internally to avoid overflowing intermediate
    gradients at tiny stabilizers; outputs retain float32/float64 as above.
    """
    return _call(x1, x1 if x2 is None else x2, mode='allpairs', kind='ssim',
                 data_range=data_range, chunk_size=chunk_size, window_size=window_size,
                 sigma=sigma, k1=k1, k2=k2)


def paired_ssim_scores(x1, x2, *, data_range, window_size=11, sigma=1.5, k1=.01, k2=.03, chunk_size=16):
    """Aligned SSIM ``[...,N]``; see :func:`ssim_scores`."""
    return _call(x1, x2, mode='paired', kind='ssim', data_range=data_range,
                 chunk_size=chunk_size, window_size=window_size, sigma=sigma, k1=k1, k2=k2)


def condensed_ssim_scores(x, *, data_range, window_size=11, sigma=1.5, k1=.01, k2=.03, chunk_size=16):
    """Self SSIM in row-major i<j order ``[...,N*(N-1)//2]``."""
    return _call(x, x, mode='condensed', kind='ssim', data_range=data_range,
                 chunk_size=chunk_size, window_size=window_size, sigma=sigma, k1=k1, k2=k2)


def multiscale_ssim_scores(x1, x2=None, *, data_range, window_size=11, sigma=1.5,
                           k1=.01, k2=.03, weights=_MS_WEIGHTS, chunk_size=16):
    """Whole-image MS-SSIM ``[...,N,M]``, with higher scores indicating similarity.

    Uses :func:`ssim_scores` moments at each scale. Between scales average-pool
    2x2 with stride 2, no padding, dropping unmatched last rows/columns. For L
    positive weights, H/W must each be >= window_size*2**(L-1). Default weights
    are (0.0448, 0.2856, 0.3001, 0.2363, 0.1333), used without renormalization.
    Compute the weighted product of spatial-mean contrast/structure terms at
    the first L-1 scales and spatial-mean SSIM at the last, independently per
    channel, then average channels. Nonpositive factors map to zero with zero
    derivative before fractional powers. Raw image intensities are not clamped.
    Identity returns one with zero whole-pair gradients.
    """
    if weights is None:
        raise TypeError('weights must be a non-empty tuple or list of positive real scalars')
    return _call(x1, x1 if x2 is None else x2, mode='allpairs', kind='multiscale_ssim',
                 data_range=data_range, chunk_size=chunk_size, window_size=window_size,
                 sigma=sigma, k1=k1, k2=k2, weights=weights)


def paired_multiscale_ssim_scores(x1, x2, *, data_range, window_size=11, sigma=1.5,
                                  k1=.01, k2=.03, weights=_MS_WEIGHTS, chunk_size=16):
    """Aligned MS-SSIM ``[...,N]``; see :func:`multiscale_ssim_scores`."""
    if weights is None:
        raise TypeError('weights must be a non-empty tuple or list of positive real scalars')
    return _call(x1, x2, mode='paired', kind='multiscale_ssim', data_range=data_range,
                 chunk_size=chunk_size, window_size=window_size, sigma=sigma,
                 k1=k1, k2=k2, weights=weights)


def condensed_multiscale_ssim_scores(x, *, data_range, window_size=11, sigma=1.5,
                                     k1=.01, k2=.03, weights=_MS_WEIGHTS, chunk_size=16):
    """Self MS-SSIM in row-major i<j order ``[...,N*(N-1)//2]``."""
    if weights is None:
        raise TypeError('weights must be a non-empty tuple or list of positive real scalars')
    return _call(x, x, mode='condensed', kind='multiscale_ssim', data_range=data_range,
                 chunk_size=chunk_size, window_size=window_size, sigma=sigma,
                 k1=k1, k2=k2, weights=weights)
