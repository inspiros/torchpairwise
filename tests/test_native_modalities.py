"""Native image/audio deployment; numerical oracles live in test_image/audio."""
import weakref

import pytest
import torch

import torchpairwise as tp


def test_native_modality_dispatch_and_direct_formulas(device):
    x = torch.rand(3, 2, 5, 6, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand_like(x, requires_grad=True)
    actual = torch.ops.torchpairwise.image_comparisons(x, y, 'psnr', 'paired', 1.7, 2)
    expected = 20 * torch.tensor(1.7, device=device, dtype=x.dtype).log10() - 10 * (x-y).square().mean((-3, -2, -1)).log10()
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual.sum(), (x, y)),
                    torch.autograd.grad(expected.sum(), (x, y))):
        torch.testing.assert_close(a, b)
    x = torch.randn(3, 2, 24, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)
    actual = torch.ops.torchpairwise.audio_comparisons(x, y, 'signal_snr', 'paired', 2)
    expected = (10 * (y.square().sum(-1).clamp_min(1e-8).log10() -
                     (x-y).square().sum(-1).clamp_min(1e-8).log10())).mean(-1)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual.sum(), (x, y)),
                    torch.autograd.grad(expected.sum(), (x, y))):
        torch.testing.assert_close(a, b)
    with pytest.raises(RuntimeError, match='Condensed'):
        torch.ops.torchpairwise.audio_comparisons(x, None, 'signal_snr', 'condensed')
    if x.device.type == 'cpu':
        with pytest.raises(RuntimeError, match='finite'):
            torch.ops.torchpairwise.audio_comparisons(x.detach().fill_(float('nan')), None, 'si_sdr')


def test_modality_large_chunk_and_scale_options(device):
    x = torch.rand(3, 1, 1, 1, device=device)
    torch.testing.assert_close(tp.condensed_psnr_scores(x, data_range=1., chunk_size=2**63-1),
                               tp.condensed_psnr_scores(x, data_range=1., chunk_size=1))
    with pytest.raises(RuntimeError, match='too small'):
        torch.ops.torchpairwise.image_comparisons(x, None, 'multiscale_ssim', 'all_pairs', 1.,
                                                 window_size=1, weights=[1.] * 64)


@pytest.mark.parametrize('family', ['image', 'audio'])
def test_modality_replay_retains_collection_operands(family, device):
    shape = (5, 2, 6, 7) if family == 'image' else (5, 2, 32)
    x = torch.rand(shape, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.rand(shape, dtype=x.dtype, device=device, requires_grad=True)
    references = []

    class Saved:
        pass

    def pack(tensor):
        saved = Saved()
        saved.tensor = tensor.detach()
        references.append(weakref.ref(saved))
        return saved

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda saved: saved.tensor):
        if family == 'image':
            output = tp.multiscale_ssim_scores(x, y, data_range=1., window_size=3,
                                              weights=(.4, .6), chunk_size=2)
        else:
            output = tp.multiresolution_stft_distances(x, y, resolutions=((8, 2, 8), (16, 4, 16)), chunk_size=2)
    shapes = [tuple(saved().tensor.shape) for saved in references if saved() is not None]
    assert shapes and all(shape == tuple(x.shape) for shape in shapes)
    assert all(gradient.isfinite().all() for gradient in torch.autograd.grad(output.sum(), (x, y)))


@pytest.mark.compiler
@pytest.mark.parametrize('family', ['image', 'audio'])
@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed'])
def test_modality_export_dynamic_roundtrip_and_compile(family, mode, device, tmp_path):
    class Comparison(torch.nn.Module):
        def forward(self, x, y):
            if family == 'image':
                kwargs = dict(data_range=1.7, window_size=3, weights=(.4, .6), chunk_size=2)
                if mode == 'all_pairs':
                    return tp.multiscale_ssim_scores(x, y, **kwargs)
                if mode == 'paired':
                    return tp.paired_multiscale_ssim_scores(x, y, **kwargs)
                return tp.condensed_multiscale_ssim_scores(x, **kwargs)
            kwargs = dict(n_fft=8, hop_length=3, win_length=6, chunk_size=2)
            if mode == 'all_pairs':
                return tp.log_spectral_distances(x, y, **kwargs)
            if mode == 'paired':
                return tp.paired_log_spectral_distances(x, y, **kwargs)
            return tp.condensed_log_spectral_distances(x, **kwargs)

    def inputs(n, m):
        shape = (2, 6, 7) if family == 'image' else (2, 24)
        return (torch.rand((n, *shape), device=device, requires_grad=True),
                torch.rand((m, *shape), device=device, requires_grad=True))

    n = torch.export.Dim('n', min=3 if mode == 'condensed' else 2)
    m = n if mode == 'paired' else torch.export.Dim('m', min=2)
    module = Comparison()
    exported = torch.export.export(module, inputs(3, 3 if mode == 'paired' else 4),
                                   dynamic_shapes=({0: n}, {0: m}))
    path = tmp_path / f'{family}_{mode}.pt2'
    torch.export.save(exported, path)
    restored = torch.export.load(path).module()
    args = inputs(5, 5 if mode == 'paired' else 6)
    actual, expected = restored(*args), module(*args)
    torch.testing.assert_close(actual, expected)
    operands = args[:1] if mode == 'condensed' else args
    for a, b in zip(torch.autograd.grad(actual.sum(), operands),
                    torch.autograd.grad(expected.sum(), operands)):
        torch.testing.assert_close(a, b)
    torch.compiler.reset()
    compiled = torch.compile(module, fullgraph=True, dynamic=True, backend='aot_eager')
    actual, expected = compiled(*args), module(*args)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual.sum(), operands),
                    torch.autograd.grad(expected.sum(), operands)):
        torch.testing.assert_close(a, b)
    torch.compiler.reset()


@pytest.mark.compiler
@pytest.mark.parametrize('family', ['image', 'audio'])
@pytest.mark.parametrize('n', [0, 1, 2])
def test_modality_fixed_condensed_boundary_deployment(family, n, device, tmp_path):
    # Export specializes zero/singleton output sizes; cover them separately
    # from the dynamic N>=3 contract (whose condensed output has size >=3).
    class Comparison(torch.nn.Module):
        def forward(self, x):
            if family == 'image':
                return tp.condensed_ssim_scores(x, data_range=1., window_size=3)
            return tp.condensed_log_spectral_distances(x, n_fft=8, hop_length=3, win_length=6)

    shape = (n, 1, 6, 7) if family == 'image' else (n, 1, 24)
    x = torch.rand(shape, device=device, requires_grad=True)
    module = Comparison()
    expected = module(x)
    assert expected.shape == (n * (n - 1) // 2,)
    exported = torch.export.export(module, (x,))
    path = tmp_path / f'{family}_boundary_{n}.pt2'
    torch.export.save(exported, path)
    actual = torch.export.load(path).module()(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                               torch.autograd.grad(expected.sum(), x)[0])
    torch.compiler.reset()
    compiled = torch.compile(module, fullgraph=True, backend='aot_eager')
    actual, expected = compiled(x), module(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                               torch.autograd.grad(expected.sum(), x)[0])
    torch.compiler.reset()
