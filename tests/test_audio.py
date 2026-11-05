"""Independent waveform/STFT definitions, modes, gradients and feature reuse."""
import numpy as np
import pytest
import torch

import torchpairwise as audio


KINDS = ('si_sdr_scores', 'signal_snr_scores', 'log_spectral_distances',
         'spectral_convergence_distances', 'multiresolution_stft_distances')


def options(kind):
    if kind.endswith('scores'):
        return dict(zero_mean=True, energy_floor=2e-7, chunk_size=2)
    if kind.startswith('multiresolution'):
        return dict(resolutions=((8, 3, 6), (12, 4, 8)), log_floor=2e-6,
                    norm_floor=3e-6, chunk_size=2)
    result = dict(n_fft=8, hop_length=3, win_length=6, chunk_size=2)
    result['log_floor' if kind.startswith('log_') else 'norm_floor'] = 2e-6
    return result


def numpy_magnitude(signal, resolution, *, window='hann', center=True,
                    pad_mode='constant', normalized=False):
    n_fft, hop, win = resolution
    indices = np.arange(win)
    if window == 'hann':
        weights = .5 - .5 * np.cos(2 * np.pi * indices / win) if win > 1 else np.ones(1)
    elif window == 'hamming':
        weights = .54 - .46 * np.cos(2 * np.pi * indices / win) if win > 1 else np.ones(1)
    else:
        weights = np.ones(win)
    weights = np.pad(weights, ((n_fft - win) // 2, (n_fft - win + 1) // 2))
    if center:
        signal = np.pad(signal, ((0, 0), (n_fft // 2, n_fft // 2)), mode=pad_mode)
    frames = np.lib.stride_tricks.sliding_window_view(signal, n_fft, axis=-1)[..., ::hop, :]
    result = np.abs(np.fft.rfft(frames * weights, axis=-1))
    if normalized:
        result /= np.sqrt(n_fft)
    return result


def numpy_pair(estimate, reference, kind, **kwargs):
    if kind.endswith('scores'):
        if kwargs.get('zero_mean', kind == 'si_sdr_scores'):
            estimate = estimate - estimate.mean(-1, keepdims=True)
            reference = reference - reference.mean(-1, keepdims=True)
        target = reference
        if kind == 'si_sdr_scores':
            energy = np.sum(reference ** 2, axis=-1, keepdims=True)
            alpha = np.sum(estimate * reference, axis=-1, keepdims=True) / np.where(energy > 0, energy, 1)
            target = alpha * reference
        floor = kwargs.get('energy_floor', 1e-8)
        return np.mean(10 * np.log10(np.maximum(np.sum(target ** 2, axis=-1), floor)
                                    / np.maximum(np.sum((estimate - target) ** 2, axis=-1), floor)))
    resolutions = kwargs.get('resolutions', ((kwargs.get('n_fft', 512), kwargs.get('hop_length', 128),
                                            kwargs.get('win_length', 512)),))
    scores = []
    for resolution in resolutions:
        a, b = (numpy_magnitude(x, resolution, **{k: kwargs[k] for k in
                 ('window', 'center', 'pad_mode', 'normalized') if k in kwargs}) for x in (estimate, reference))
        log_error = np.abs(np.log(np.maximum(a, kwargs.get('log_floor', 1e-7)))
                           - np.log(np.maximum(b, kwargs.get('log_floor', 1e-7)))).mean()
        sc = np.linalg.norm(a - b) / max(np.linalg.norm(b), kwargs.get('norm_floor', 1e-7))
        scores.append(log_error if kind.startswith('log_') else sc if kind.startswith('spectral_') else sc + log_error)
    return np.mean(scores)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_audio_numpy_modes_broadcast_noncontiguous(kind, dtype, device):
    rng = np.random.default_rng(735)
    a = rng.normal(size=(2, 1, 3, 24, 2))
    b = rng.normal(size=(1, 2, 3, 24, 2))
    x = torch.tensor(a, dtype=dtype, device=device).transpose(-1, -2)
    y = torch.tensor(b, dtype=dtype, device=device).transpose(-1, -2)
    kwargs = options(kind)
    expected = np.empty((2, 2, 3, 3))
    for i, j, n, m in np.ndindex(expected.shape):
        expected[i, j, n, m] = numpy_pair(a[i, 0, n].T, b[0, j, m].T, kind, **kwargs)
    result = getattr(audio, kind)(x, y, **kwargs)
    expected = torch.tensor(expected, dtype=dtype, device=device)
    torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(getattr(audio, 'paired_' + kind)(x, y, **kwargs),
                               expected.diagonal(dim1=-2, dim2=-1), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(getattr(audio, kind)(x, y[..., :2, :, :], **kwargs), expected[..., :2],
                               atol=2e-5, rtol=2e-5)
    if kind == 'log_spectral_distances':
        self_result = getattr(audio, kind)(x, **kwargs)
        i, j = torch.triu_indices(3, 3, 1, device=device)
        torch.testing.assert_close(audio.condensed_log_spectral_distances(x, **kwargs), self_result[..., i, j])
        torch.testing.assert_close(getattr(audio, kind)(y, x, **kwargs).transpose(-1, -2), result)


@pytest.mark.parametrize('kind', KINDS)
def test_audio_gradcheck(kind, device):
    torch.manual_seed(893)
    x = torch.randn(2, 1, 16, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: getattr(audio, 'paired_' + kind)(a, b, **options(kind)),
                                   (x, y), fast_mode=True)


@pytest.mark.parametrize('kind', KINDS)
def test_audio_silence_empty_and_singleton(kind, device):
    x = torch.zeros(2, 2, 24, dtype=torch.float64, device=device, requires_grad=True)
    result = getattr(audio, kind)(x, **options(kind))
    torch.testing.assert_close(result, torch.zeros_like(result))
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))
    empty = getattr(audio, kind)(x[..., :0, :, :], x, **options(kind))
    assert empty.shape == (0, 2) and empty.requires_grad
    empty.sum().backward()
    if kind == 'log_spectral_distances':
        assert audio.condensed_log_spectral_distances(x[:1], **options(kind)).shape == (0,)


def test_audio_scores_boundaries_and_direction(device):
    ref = torch.tensor([[[1., -1., 1., -1.]]], dtype=torch.float64, device=device)
    estimate = 2 * ref
    # Gain changes are removed by projection; the perfect score uses a floor.
    torch.testing.assert_close(audio.si_sdr_scores(estimate, ref, zero_mean=False),
                               torch.tensor([[10 * np.log10(16 / 1e-8)]], dtype=ref.dtype, device=device))
    torch.testing.assert_close(audio.signal_snr_scores(estimate, ref), torch.zeros(1, 1, dtype=ref.dtype, device=device))
    reverse = audio.signal_snr_scores(ref, estimate)
    torch.testing.assert_close(reverse, torch.tensor([[10 * np.log10(4)]], dtype=ref.dtype, device=device))
    zero = torch.zeros_like(ref)
    for name in ('si_sdr_scores', 'signal_snr_scores'):
        result = getattr(audio, name)(estimate, zero, zero_mean=False)
        torch.testing.assert_close(result, torch.tensor([[10 * np.log10(1e-8 / 16)]], dtype=ref.dtype, device=device))
    offset = ref + 5
    torch.testing.assert_close(audio.si_sdr_scores(offset, ref), audio.si_sdr_scores(ref, ref))


@pytest.mark.parametrize('window', ['hann', 'hamming', 'rectangular'])
@pytest.mark.parametrize('center,pad_mode', [(False, 'constant'), (True, 'reflect')])
def test_audio_stft_protocol(window, center, pad_mode, device):
    x = torch.arange(24, dtype=torch.float64, device=device).reshape(1, 1, 24) / 24
    y = x.roll(3, -1)
    kwargs = dict(n_fft=9, hop_length=2, win_length=5, window=window,
                  center=center, pad_mode=pad_mode, normalized=True, log_floor=.01)
    expected = numpy_pair(x.cpu().numpy()[0], y.cpu().numpy()[0], 'log_spectral_distances', **kwargs)
    result = audio.log_spectral_distances(x, y, **kwargs)
    torch.testing.assert_close(result, torch.tensor([[expected]], dtype=x.dtype, device=device))


def test_audio_transforms_shared_across_pairs_and_backward(device):
    x = torch.randn(1, 3, 2, 24, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.randn(2, 4, 2, 24, dtype=torch.float64, device=device, requires_grad=True)
    resolutions = ((8, 2, 8), (12, 3, 10), (8, 2, 8))
    # Native transforms are observable through ATen, rather than torch.stft.
    # Duplicate resolutions share transforms. Native backward replays each
    # collection transform once, independently of the number of pair tiles.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
        result = audio.multiresolution_stft_distances(x, y, resolutions=resolutions, chunk_size=2)
    assert sorted(tuple(event.input_shapes[0]) for event in profile.events()
                  if event.name == 'aten::stft') == [(6, 24), (6, 24), (16, 24), (16, 24)]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
        result.sum().backward()
    assert sorted(tuple(event.input_shapes[0]) for event in profile.events()
                  if event.name == 'aten::stft') == [(6, 24), (6, 24), (16, 24), (16, 24)]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
        audio.multiresolution_stft_distances(x, resolutions=resolutions)
    assert [tuple(event.input_shapes[0]) for event in profile.events()
            if event.name == 'aten::stft'] == [(6, 24), (6, 24)]


@pytest.mark.parametrize('kwargs,exception', [({'n_fft': 0}, ValueError), ({'hop_length': True}, TypeError),
    ({'win_length': 9, 'n_fft': 8}, ValueError), ({'window': 'unknown'}, ValueError),
    ({'center': 1}, TypeError), ({'normalized': 1}, TypeError), ({'log_floor': 0}, ValueError),
    ({'pad_mode': 'reflect', 'n_fft': 64}, ValueError), ({'center': False, 'n_fft': 64}, ValueError),
    ({'chunk_size': 0}, ValueError)])
def test_audio_rejects_stft_options(kwargs, exception, device):
    x = torch.ones(2, 1, 16, device=device)
    with pytest.raises(exception):
        audio.log_spectral_distances(x, **kwargs)


def test_audio_rejects_shapes_and_values(device):
    x = torch.ones(2, 1, 16, device=device)
    for bad in (x[..., 0, :], x[..., :0], x[..., :8], x.double(), x.long(), x * float('nan')):
        with pytest.raises((ValueError, TypeError)):
            audio.signal_snr_scores(x, bad)
    with pytest.raises(ValueError, match='equal sample counts'):
        audio.paired_signal_snr_scores(x, x[:1])
    with pytest.raises(ValueError):
        audio.multiresolution_stft_distances(x, resolutions=())
    with pytest.raises(ValueError):
        audio.si_sdr_scores(x, energy_floor=-1)


def test_audio_dtype_autocast(device):
    x = torch.randn(2, 1, 24, device=device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        half = x.to(torch.bfloat16)
        result = audio.log_spectral_distances(half, n_fft=8)
        assert result.dtype == torch.float32
    with pytest.raises(TypeError):
        audio.log_spectral_distances(half, n_fft=8)


def test_audio_spectral_silent_reference_and_direction(device):
    estimate = torch.ones(1, 1, 4, dtype=torch.float64, device=device)
    silence = torch.zeros_like(estimate)
    kwargs = dict(n_fft=4, hop_length=4, window='rectangular', center=False)
    # Single FFT frame: magnitude [4,0,0], so its Frobenius norm is four.
    forward = audio.spectral_convergence_distances(estimate, silence, norm_floor=.5, **kwargs)
    reverse = audio.spectral_convergence_distances(silence, estimate, norm_floor=.5, **kwargs)
    torch.testing.assert_close(forward, torch.tensor([[8.]], dtype=estimate.dtype, device=device))
    torch.testing.assert_close(reverse, torch.tensor([[1.]], dtype=estimate.dtype, device=device))
    log = audio.log_spectral_distances(estimate, silence, log_floor=1., **kwargs)
    torch.testing.assert_close(log, torch.tensor([[np.log(4) / 3]], dtype=estimate.dtype, device=device))
