"""Validation adapters for native aligned waveform comparisons."""
import numbers

import torch

from ._batching import validate_values
from ._batching import is_finite_real


_RESOLUTIONS = ((512, 128, 512), (1024, 256, 1024), (2048, 512, 2048))


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f'{name} must be a real scalar')
    if not is_finite_real(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return float(value)


def _integer(value, name):
    if type(value) is not int:
        raise TypeError(f'{name} must be an integer')
    if value < 1:
        raise ValueError(f'{name} must be positive')
    return value


def _prepare(x, y, mode, chunk_size):
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError('Audio inputs must be tensors')
    if x.layout != torch.strided or y.layout != torch.strided:
        raise TypeError('Audio inputs must be dense strided tensors')
    if x.ndim < 3 or y.ndim < 3:
        raise ValueError('Audio inputs must have shape [...,N,C,T]')
    if x.shape[-2:] != y.shape[-2:]:
        raise ValueError('Audio channel and time dimensions must match')
    if not x.shape[-2] or not x.shape[-1]:
        raise ValueError('Audio channel and time dimensions must be non-empty')
    if x.dtype != y.dtype or x.device != y.device:
        raise ValueError('Audio inputs must have the same dtype and device')
    if x.device.type not in ('cpu', 'cuda'):
        raise ValueError('Audio inputs must be on CPU or CUDA')
    allowed = ((torch.float32, torch.float64, torch.float16, torch.bfloat16)
               if torch.is_autocast_enabled(x.device.type) else (torch.float32, torch.float64))
    if x.dtype not in allowed:
        raise TypeError('Audio inputs require float32 or float64 outside autocast')
    if mode == 'paired' and x.shape[-3] != y.shape[-3]:
        raise ValueError('Paired audio inputs must have equal sample counts')
    _integer(chunk_size, 'chunk_size')
    try:
        batch = torch.broadcast_shapes(x.shape[:-3], y.shape[:-3])
    except RuntimeError as exc:
        raise ValueError('Audio batch dimensions are not broadcastable') from exc
    validate_values(torch.isfinite(x).all() & torch.isfinite(y).all(), 'Audio inputs must be finite')
    return batch


def _stft_options(length, n_fft, hop_length, win_length, window, center, pad_mode, normalized):
    n_fft = _integer(n_fft, 'n_fft')
    hop_length = _integer(max(1, n_fft // 4) if hop_length is None else hop_length, 'hop_length')
    win_length = _integer(n_fft if win_length is None else win_length, 'win_length')
    if win_length > n_fft:
        raise ValueError('win_length must not exceed n_fft')
    if window not in ('hann', 'hamming', 'rectangular'):
        raise ValueError('window must be hann, hamming, or rectangular')
    if type(center) is not bool or type(normalized) is not bool:
        raise TypeError('center and normalized must be booleans')
    if pad_mode not in ('constant', 'reflect'):
        raise ValueError('pad_mode must be constant or reflect')
    if center and pad_mode == 'reflect' and length <= n_fft // 2:
        raise ValueError('Reflect padding requires T > n_fft//2')
    if not center and length < n_fft:
        raise ValueError('center=False requires T >= n_fft')
    return n_fft, hop_length, win_length


def _call(x, y, *, kind, mode, chunk_size=16, zero_mean=False, energy_floor=1e-8,
          n_fft=512, hop_length=None, win_length=None, window='hann', center=True,
          pad_mode='constant', normalized=False, log_floor=1e-7, norm_floor=1e-7,
          resolutions=None):
    # C++ validates options in compiled/exported graphs; retain Python error
    # types for eager callers without introducing scalar-math graph breaks.
    if not torch.compiler.is_compiling():
        _prepare(x, y, mode, chunk_size)
        if type(zero_mean) is not bool:
            raise TypeError('zero_mean must be a boolean')
        energy_floor = _positive(energy_floor, 'energy_floor')
        log_floor = _positive(log_floor, 'log_floor')
        norm_floor = _positive(norm_floor, 'norm_floor')
        if kind not in ('si_sdr', 'signal_snr'):
            if kind == 'multiresolution_stft':
                if not isinstance(resolutions, (tuple, list)) or not resolutions:
                    raise ValueError('resolutions must be a non-empty list/tuple of (n_fft, hop_length, win_length)')
                if any(not isinstance(r, (tuple, list)) or len(r) != 3 for r in resolutions):
                    raise ValueError('Each resolution must contain n_fft, hop_length, win_length')
            else:
                resolutions = ((n_fft, hop_length, win_length),)
            resolutions = tuple(_stft_options(x.shape[-1], *r, window, center, pad_mode, normalized)
                                for r in resolutions)
    return torch.ops.torchpairwise.audio_comparisons(
        x, None if y is x and mode != 'paired' else y, kind,
        'all_pairs' if mode == 'allpairs' else mode, chunk_size, zero_mean,
        energy_floor, n_fft, hop_length, win_length, window, center, pad_mode,
        normalized, log_floor, norm_floor,
        [] if resolutions is None else [v for n, hop, win in resolutions
                                       for v in (n, max(1, n // 4) if hop is None else hop,
                                                 n if win is None else win)])


def si_sdr_scores(x1, x2=None, *, zero_mean=True, energy_floor=1e-8, chunk_size=16):
    """Directional SI-SDR in dB for [...,N,C,T] estimates and references.

    Project each estimated channel onto its reference, then average channel dB
    scores. Time centering defaults to True. Each summed target/error energy is
    clamped to energy_floor; silent/silent is zero and perfect signals are finite.
    Floors affect scale invariance near silence. See docs/audio-comparisons.md.
    """
    return _call(x1, x1 if x2 is None else x2, kind='si_sdr', mode='allpairs',
                 zero_mean=zero_mean, energy_floor=energy_floor, chunk_size=chunk_size)


def paired_si_sdr_scores(x1, x2, *, zero_mean=True, energy_floor=1e-8, chunk_size=16):
    """Aligned SI-SDR [...,N]; x2 is the reference. See si_sdr_scores."""
    return _call(x1, x2, kind='si_sdr', mode='paired', zero_mean=zero_mean,
                 energy_floor=energy_floor, chunk_size=chunk_size)


def signal_snr_scores(x1, x2=None, *, zero_mean=False, energy_floor=1e-8, chunk_size=16):
    """Signal SNR dB = 10 log10(reference energy / error energy), per channel.

    Both energies are floored by energy_floor before the channel dB average.
    Inputs are [...,N,C,T]; this is distinct from vector snr_distances.
    See docs/audio-comparisons.md for boundaries, dtype and memory contracts.
    """
    return _call(x1, x1 if x2 is None else x2, kind='signal_snr', mode='allpairs',
                 zero_mean=zero_mean, energy_floor=energy_floor, chunk_size=chunk_size)


def paired_signal_snr_scores(x1, x2, *, zero_mean=False, energy_floor=1e-8, chunk_size=16):
    """Aligned signal SNR [...,N]; x2 is the reference. See signal_snr_scores."""
    return _call(x1, x2, kind='signal_snr', mode='paired', zero_mean=zero_mean,
                 energy_floor=energy_floor, chunk_size=chunk_size)


def log_spectral_distances(x1, x2=None, *, n_fft=512, hop_length=None,
                                   win_length=None, window='hann', center=True,
                                   pad_mode='constant', normalized=False, log_floor=1e-7,
                                   chunk_size=16):
    """Mean absolute difference of natural log STFT magnitudes (symmetric).

    Inputs [...,N,C,T]; periodic windows, one-sided spectra and no detrending.
    See docs/audio-comparisons.md for shapes, floors, dtype and STFT protocol.
    """
    return _call(x1, x1 if x2 is None else x2, kind='log_spectral', mode='allpairs',
                 n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
                 center=center, pad_mode=pad_mode, normalized=normalized,
                 log_floor=log_floor, chunk_size=chunk_size)


def paired_log_spectral_distances(x1, x2, *, n_fft=512, hop_length=None,
                                   win_length=None, window='hann', center=True,
                                   pad_mode='constant', normalized=False, log_floor=1e-7,
                                   chunk_size=16):
    """Mean absolute difference of natural log STFT magnitudes (symmetric).

    Inputs [...,N,C,T]; periodic windows, one-sided spectra and no detrending.
    See docs/audio-comparisons.md for shapes, floors, dtype and STFT protocol.
    """
    return _call(x1, x2, kind='log_spectral', mode='paired',
                 n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
                 center=center, pad_mode=pad_mode, normalized=normalized,
                 log_floor=log_floor, chunk_size=chunk_size)


def condensed_log_spectral_distances(x, *, n_fft=512, hop_length=None,
                                   win_length=None, window='hann', center=True,
                                   pad_mode='constant', normalized=False, log_floor=1e-7,
                                   chunk_size=16):
    """Mean absolute difference of natural log STFT magnitudes (symmetric).

    Inputs [...,N,C,T]; periodic windows, one-sided spectra and no detrending.
    See docs/audio-comparisons.md for shapes, floors, dtype and STFT protocol.
    """
    return _call(x, x, kind='log_spectral', mode='condensed',
                 n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
                 center=center, pad_mode=pad_mode, normalized=normalized,
                 log_floor=log_floor, chunk_size=chunk_size)


def spectral_convergence_distances(x1, x2=None, *, n_fft=512, hop_length=None,
                                   win_length=None, window='hann', center=True,
                                   pad_mode='constant', normalized=False, norm_floor=1e-7,
                                   chunk_size=16):
    """Frobenius STFT magnitude error divided by the x2 reference norm.

    Inputs [...,N,C,T]; periodic windows, one-sided spectra and no detrending.
    See docs/audio-comparisons.md for shapes, floors, dtype and STFT protocol.
    """
    return _call(x1, x1 if x2 is None else x2, kind='spectral_convergence', mode='allpairs',
                 n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
                 center=center, pad_mode=pad_mode, normalized=normalized,
                 norm_floor=norm_floor, chunk_size=chunk_size)


def paired_spectral_convergence_distances(x1, x2, *, n_fft=512, hop_length=None,
                                   win_length=None, window='hann', center=True,
                                   pad_mode='constant', normalized=False, norm_floor=1e-7,
                                   chunk_size=16):
    """Frobenius STFT magnitude error divided by the x2 reference norm.

    Inputs [...,N,C,T]; periodic windows, one-sided spectra and no detrending.
    See docs/audio-comparisons.md for shapes, floors, dtype and STFT protocol.
    """
    return _call(x1, x2, kind='spectral_convergence', mode='paired',
                 n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
                 center=center, pad_mode=pad_mode, normalized=normalized,
                 norm_floor=norm_floor, chunk_size=chunk_size)


def multiresolution_stft_distances(x1, x2=None, *, resolutions=_RESOLUTIONS,
                                             window='hann', center=True, pad_mode='constant',
                                             normalized=False, log_floor=1e-7, norm_floor=1e-7,
                                             chunk_size=16):
    """Mean over resolutions of spectral convergence + mean absolute log error.

    Resolutions are explicit (n_fft, hop_length, win_length) triples. The default
    triples are (512,128,512), (1024,256,1024), (2048,512,2048). x2 is the
    directional reference. See docs/audio-comparisons.md for the full protocol.
    """
    return _call(x1, x1 if x2 is None else x2, kind='multiresolution_stft', mode='allpairs',
                 resolutions=resolutions, window=window, center=center, pad_mode=pad_mode,
                 normalized=normalized, log_floor=log_floor, norm_floor=norm_floor,
                 chunk_size=chunk_size)


def paired_multiresolution_stft_distances(x1, x2, *, resolutions=_RESOLUTIONS,
                                             window='hann', center=True, pad_mode='constant',
                                             normalized=False, log_floor=1e-7, norm_floor=1e-7,
                                             chunk_size=16):
    """Mean over resolutions of spectral convergence + mean absolute log error.

    Resolutions are explicit (n_fft, hop_length, win_length) triples. The default
    triples are (512,128,512), (1024,256,1024), (2048,512,2048). x2 is the
    directional reference. See docs/audio-comparisons.md for the full protocol.
    """
    return _call(x1, x2, kind='multiresolution_stft', mode='paired',
                 resolutions=resolutions, window=window, center=center, pad_mode=pad_mode,
                 normalized=normalized, log_floor=log_floor, norm_floor=norm_floor,
                 chunk_size=chunk_size)
