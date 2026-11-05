"""Whole-image quality definitions, independent patch oracle, and gradients."""
import numpy as np
import pytest
import torch

import torchpairwise as image


KINDS = ('psnr', 'ssim', 'multiscale_ssim')


def options(kind):
    result = dict(data_range=1.7, chunk_size=2)
    if kind != 'psnr':
        result.update(window_size=3, sigma=.9, k1=.02, k2=.04)
    if kind == 'multiscale_ssim':
        result['weights'] = (.35, .65)
    return result


def numpy_pair(x, y, kind, *, data_range=1.7, window_size=3, sigma=.9,
               k1=.02, k2=.04, weights=(.35, .65), **ignored):
    # Explicit windows and centered patch moments, independently of the
    # implementation's separable convolution and image-local moment cache.
    if kind == 'psnr':
        mse = np.mean((x - y) ** 2)
        return np.inf if mse == 0 else 10 * np.log10(data_range ** 2 / mse)
    coordinates = np.arange(window_size) - window_size // 2
    gaussian = np.exp(-coordinates ** 2 / (2 * sigma ** 2))
    gaussian /= gaussian.sum()
    kernel = np.outer(gaussian, gaussian)
    factors = []
    levels = len(weights) if kind == 'multiscale_ssim' else 1
    for level in range(levels):
        a = np.lib.stride_tricks.sliding_window_view(x, (window_size, window_size), axis=(-2, -1))
        b = np.lib.stride_tricks.sliding_window_view(y, (window_size, window_size), axis=(-2, -1))
        ma, mb = (np.sum(t * kernel, axis=(-2, -1)) for t in (a, b))
        da, db = a - ma[..., None, None], b - mb[..., None, None]
        va, vb = (np.sum(t ** 2 * kernel, axis=(-2, -1)) for t in (da, db))
        cov = np.sum(da * db * kernel, axis=(-2, -1))
        c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
        luminance = (2 * ma * mb + c1) / (ma ** 2 + mb ** 2 + c1)
        contrast = (2 * cov + c2) / (va + vb + c2)
        score = np.mean(luminance * contrast, axis=(-2, -1))
        if kind == 'ssim':
            return score.mean()
        factor = score if level + 1 == levels else contrast.mean(axis=(-2, -1))
        factors.append(np.maximum(factor, 0) ** weights[level])
        if level + 1 < levels:
            h, w = (size // 2 * 2 for size in x.shape[-2:])
            x, y = (t[..., :h, :w].reshape(*t.shape[:-2], h // 2, 2, w // 2, 2).mean(axis=(-3, -1))
                    for t in (x, y))
    return np.prod(factors, axis=0).mean()


def numpy_allpairs(x, y, kind, kwargs):
    x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()
    batch = np.broadcast_shapes(x.shape[:-4], y.shape[:-4])
    x, y = (np.broadcast_to(t, (*batch, *t.shape[-4:])) for t in (x, y))
    result = np.empty((*batch, x.shape[-4], y.shape[-4]))
    for index in np.ndindex(batch):
        for i in range(x.shape[-4]):
            for j in range(y.shape[-4]):
                result[index + (i, j)] = numpy_pair(x[index + (i,)], y[index + (j,)], kind, **kwargs)
    return torch.as_tensor(result)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_image_modes_broadcast_noncontiguous_and_numpy(kind, dtype, device):
    torch.manual_seed(516)
    x = torch.rand(2, 1, 3, 2, 11, 9, device=device, dtype=dtype).transpose(-1, -2)
    y = torch.rand(1, 2, 3, 2, 11, 9, device=device, dtype=dtype).transpose(-1, -2) * .1 + .45
    assert not x.is_contiguous()
    kwargs = options(kind)
    allpairs = getattr(image, kind + '_scores')
    paired = getattr(image, 'paired_' + kind + '_scores')
    condensed = getattr(image, 'condensed_' + kind + '_scores')
    expected = numpy_allpairs(x, y, kind, kwargs).to(device=device, dtype=dtype)
    torch.testing.assert_close(allpairs(x, y, **kwargs), expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(paired(x, y, **kwargs), expected.diagonal(dim1=-2, dim2=-1),
                               atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(allpairs(x, y[..., :2, :, :, :], **kwargs), expected[..., :2],
                               atol=2e-5, rtol=2e-5)
    expected_self = numpy_allpairs(x, x, kind, kwargs).to(device=device, dtype=dtype)
    i, j = torch.triu_indices(3, 3, 1, device=device)
    torch.testing.assert_close(condensed(x, **kwargs), expected_self[..., i, j], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(allpairs(x, **kwargs), expected_self, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(allpairs(y, x, **kwargs).transpose(-1, -2), expected,
                               atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('mode', ['', 'paired_', 'condensed_'])
def test_image_first_gradients(kind, mode, device):
    torch.manual_seed(873)
    x = (.2 + .4 * torch.rand(2, 1, 6, 7, dtype=torch.float64, device=device)).requires_grad_()
    y = (.25 + .3 * torch.rand_like(x)).requires_grad_()
    kwargs = options(kind)
    fn = getattr(image, mode + kind + '_scores')
    args = (x,) if mode == 'condensed_' else (x, y)
    assert torch.autograd.gradcheck(lambda *values: fn(*values, **kwargs), args,
                                   fast_mode=True, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize('kind', KINDS)
def test_image_identity_empty_and_chunked_gradients(kind, device):
    torch.manual_seed(704)
    x = torch.rand(2, 3, 2, 6, 7, device=device, requires_grad=True)
    y = (torch.rand(1, 3, 2, 6, 7, device=device) * .1 + .45).requires_grad_()
    kwargs = options(kind)
    paired = getattr(image, 'paired_' + kind + '_scores')
    expected = torch.full((2, 3), float('inf') if kind == 'psnr' else 1., device=device)
    identity = paired(x, x.clone(), **kwargs)
    torch.testing.assert_close(identity, expected)
    torch.testing.assert_close(torch.autograd.grad(identity.sum(), x)[0], torch.zeros_like(x))
    for prefix, args in (('', (x, y)), ('paired_', (x, y)), ('condensed_', (x,))):
        fn = getattr(image, prefix + kind + '_scores')
        actual = fn(*args, **kwargs)
        options_large = dict(kwargs, chunk_size=7)
        reference = fn(*args, **options_large)
        torch.testing.assert_close(actual, reference)
        inputs = (x,) if prefix == 'condensed_' else (x, y)
        for grad, ref in zip(torch.autograd.grad(actual.sum(), inputs),
                             torch.autograd.grad(reference.sum(), inputs)):
            torch.testing.assert_close(grad, ref)
            assert torch.isfinite(grad).all()
    allpairs = getattr(image, kind + '_scores')
    condensed = getattr(image, 'condensed_' + kind + '_scores')
    cases = [(allpairs(x[:, :0], y, **kwargs), (2, 0, 3)),
             (allpairs(x[:0], y, **kwargs), (0, 3, 3)),
             (paired(x[:, :0], y[:, :0], **kwargs), (2, 0)),
             (condensed(x[:, :0], **kwargs), (2, 0)),
             (condensed(x[:, :1], **kwargs), (2, 0))]
    for result, shape in cases:
        assert result.shape == shape
        torch.testing.assert_close(torch.autograd.grad(result.sum(), x)[0], torch.zeros_like(x))


def test_psnr_definition_range_scaling_and_small_differences(device):
    x = torch.zeros(1, 2, 3, 4, device=device, requires_grad=True)
    y = torch.full_like(x, .25, requires_grad=True)
    actual = image.paired_psnr_scores(x, y, data_range=1.)
    torch.testing.assert_close(actual, torch.full_like(actual, 20 * np.log10(4)))
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), y)[0],
                               torch.full_like(y, -20 / (np.log(10) * .25 * y.numel())))
    torch.testing.assert_close(image.paired_psnr_scores(x * 255, y * 255, data_range=255.), actual)
    # Squaring a representable float32 residual directly would underflow here.
    tiny = torch.full_like(x, 1e-30)
    torch.testing.assert_close(image.paired_psnr_scores(x, tiny, data_range=1.),
                               torch.full_like(actual, 600.))


def test_ssim_constants_negative_factors_and_default_pyramid(device):
    x = torch.ones(1, 2, 3, 3, device=device, dtype=torch.float64)
    y = x * .5
    score = image.paired_ssim_scores(x, y, data_range=1., window_size=3)
    torch.testing.assert_close(score, torch.full_like(score, (1 + .01 ** 2) / (1.25 + .01 ** 2)))
    # Contrast reversal can give negative SSIM. MS-SSIM clips score factors,
    # rather than raw image values, and defines finite zero derivatives there.
    checker = ((torch.arange(6, device=device)[:, None] + torch.arange(6, device=device)) % 2)
    x = checker.to(torch.float64)[None, None].requires_grad_()
    y = (1 - x.detach()).requires_grad_()
    ssim = image.paired_ssim_scores(x, y, data_range=1., window_size=3)
    assert (ssim < 0).all()
    ms = image.paired_multiscale_ssim_scores(x, y, data_range=1., window_size=3, weights=(.5, .5))
    torch.testing.assert_close(ms, torch.zeros_like(ms))
    for grad in torch.autograd.grad(ms.sum(), (x, y)):
        torch.testing.assert_close(grad, torch.zeros_like(grad))
    # Exercise conventional five-level defaults at the exact valid-size bound.
    x = torch.ones(1, 1, 176, 177, device=device)
    value = image.multiscale_ssim_scores(x, data_range=1.)
    torch.testing.assert_close(value, torch.ones_like(value))


def test_image_extreme_finite_scalars_and_pixels(device):
    x = torch.full((1, 1, 3, 3), 3e38, device=device)
    y = -x
    psnr = image.paired_psnr_scores(x, y, data_range=1.)
    torch.testing.assert_close(psnr, torch.full_like(psnr, -20 * np.log10(6e38)))
    if device == 'cpu':
        # CUDA configurations may flush minimum subnormal inputs to zero.
        a = torch.tensor([[[[3e38, 1.401298464324817e-45]]]], device=device)
        b = torch.tensor([[[[3e38, 0.]]]], device=device)
        assert torch.isfinite(image.paired_psnr_scores(a, b, data_range=1.)).all()
    # Scores must be independent of unrelated image samples in a collection.
    a = torch.tensor([0., 3e38], device=device).reshape(2, 1, 1, 1)
    b = torch.ones(1, 1, 1, 1, device=device)
    torch.testing.assert_close(image.ssim_scores(a, b, data_range=1., window_size=1)[0],
                               image.ssim_scores(a[:1], b, data_range=1., window_size=1)[0])
    for data_range, amplitude in ((1e308, 1.), (1e-300, 0.), (1., 1e20)):
        a = torch.full((1, 1, 3, 3), amplitude, device=device, requires_grad=True)
        b = torch.full_like(a, amplitude * .5, requires_grad=True)
        score = image.paired_ssim_scores(a, b, data_range=data_range, window_size=3, sigma=1e-300)
        assert torch.isfinite(score).all()
        for grad in torch.autograd.grad(score.sum(), (a, b)):
            assert torch.isfinite(grad).all()


@pytest.mark.parametrize('kind', KINDS)
def test_image_validation(kind, device):
    x = torch.ones(2, 1, 6, 7, device=device)
    fn = getattr(image, kind + '_scores')
    paired = getattr(image, 'paired_' + kind + '_scores')
    kwargs = options(kind)
    with pytest.raises(TypeError, match='data_range'):
        fn(x)
    for value in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='data_range'):
            fn(x, **dict(kwargs, data_range=value))
    for value in (True, torch.tensor(1., device=device), '1'):
        with pytest.raises(TypeError, match='data_range'):
            fn(x, **dict(kwargs, data_range=value))
    for bad, error, message in [
        (x[0], ValueError, 'shape'), (x.to_sparse(), TypeError, 'dense strided'),
        (x.long(), TypeError, 'float32 or float64'), (x.half(), TypeError, 'float32 or float64'),
        (x[:, :, :0], ValueError, 'non-empty'),
    ]:
        with pytest.raises(error, match=message):
            fn(bad, **kwargs)
    with pytest.raises(ValueError, match='C/H/W'):
        fn(x, x[..., :6], **kwargs)
    with pytest.raises(ValueError, match='same dtype'):
        fn(x, x.double(), **kwargs)
    with pytest.raises(ValueError, match='equal sample counts'):
        paired(x, x[:1], **kwargs)
    with pytest.raises(ValueError, match='batch dimensions'):
        fn(x.expand(2, *x.shape), x.expand(3, *x.shape), **kwargs)
    for value, error in [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)]:
        with pytest.raises(error, match='chunk_size'):
            fn(x, **dict(kwargs, chunk_size=value))
    for bad_value in (float('nan'), float('inf')):
        bad = x.clone()
        bad[0, 0, 0, 0] = bad_value
        with pytest.raises(ValueError, match='finite'):
            fn(bad, **kwargs)
    if kind != 'psnr':
        for value, error in [(0, ValueError), (2, ValueError), (True, TypeError), (3.5, TypeError)]:
            with pytest.raises(error, match='window_size'):
                fn(x, **dict(kwargs, window_size=value))
        with pytest.raises(ValueError, match='at least'):
            fn(x, **dict(kwargs, window_size=7))
        for key in ('sigma', 'k1', 'k2'):
            with pytest.raises(ValueError, match=key):
                fn(x, **dict(kwargs, **{key: 0.}))
    if kind == 'multiscale_ssim':
        for weights, error in [(None, TypeError), ((), TypeError), ((0., 1.), ValueError),
                               ((float('nan'),), ValueError), ((True,), TypeError)]:
            with pytest.raises(error, match='weights'):
                fn(x, **dict(kwargs, weights=weights))


@pytest.mark.parametrize('kind', KINDS)
def test_image_autocast(kind, device):
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    x = torch.rand(2, 1, 6, 7, device=device, dtype=dtype, requires_grad=True)
    y = (torch.rand_like(x) * .1 + .45).requires_grad_()
    kwargs = options(kind)
    for prefix, args in (('', (x, y)), ('paired_', (x, y)), ('condensed_', (x,))):
        fn = getattr(image, prefix + kind + '_scores')
        with torch.autocast(device, dtype=dtype):
            actual = fn(*args, **kwargs)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, fn(*(t.float() for t in args), **kwargs))
        assert torch.isfinite(torch.autograd.grad(actual.sum(), x)[0]).all()


@pytest.mark.compiler
@pytest.mark.parametrize('kind', KINDS)
def test_image_compiled_values_and_gradients(kind, device):
    torch.compiler.reset()
    x = torch.rand(2, 1, 6, 7, device=device, requires_grad=True)
    y = (torch.rand(2, 1, 6, 7, device=device) * .1 + .45).requires_grad_()
    kwargs = options(kind)

    def call(a, b):
        return (getattr(image, kind + '_scores')(a, b, **kwargs).sum()
                + getattr(image, 'paired_' + kind + '_scores')(a, b, **kwargs).sum()
                + getattr(image, 'condensed_' + kind + '_scores')(a, **kwargs).sum())

    expected = call(x, y)
    actual = torch.compile(call, fullgraph=True)(x, y)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    for grad, ref in zip(torch.autograd.grad(actual, (x, y)), torch.autograd.grad(expected, (x, y))):
        torch.testing.assert_close(grad, ref, atol=1e-4, rtol=1e-4)
    torch.compiler.reset()
