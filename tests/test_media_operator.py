"""Image/audio scalar operators against their canonical values and gradients."""
import io

import pytest
import torch
import torchpairwise as tp


METRICS = ('psnr', 'ssim', 'multiscale_ssim', 'si_sdr', 'signal_snr',
           'log_spectral', 'spectral_convergence', 'multiresolution_stft')


def image_metric(metric):
    return metric in METRICS[:3]


def options(metric):
    if metric == 'psnr':
        return dict(data_range=1.)
    if image_metric(metric):
        kw = dict(data_range=1., window_size=3)
        if metric == 'multiscale_ssim':
            kw['weights'] = (.4, .6)
        return kw
    if metric == 'multiresolution_stft':
        return dict(resolutions=((8, 2, 8), (12, 3, 10)))
    if metric in ('log_spectral', 'spectral_convergence'):
        return dict(n_fft=8, hop_length=2)
    return {}


def inputs(metric, n, device, batch=(), dtype=torch.float64):
    event = (2, 8, 10) if image_metric(metric) else (2, 24)
    x = torch.rand((*batch, n, *event), device=device, dtype=dtype) + .2
    return x.requires_grad_()


@pytest.mark.parametrize('metric', METRICS)
def test_media_operator_catalog_values_and_gradients(device, metric):
    torch.manual_seed(91)
    x, y = inputs(metric, 2, device), inputs(metric, 3, device)
    kw = options(metric)
    dense = tp.compare(x, y, metric=metric, **kw).T
    a = tp.lazy(x, y, metric=metric, chunk_size=1, **kw).T
    l = torch.randn(3, 2, device=device, dtype=x.dtype, requires_grad=True)
    r = torch.randn(2, 2, device=device, dtype=x.dtype, requires_grad=True)
    actual = a.matmul(r), a.bilinear(l, r), a.mean(-2), a.diagonal()
    expected = dense @ r, l.T @ dense @ r, dense.mean(-2), dense.diagonal()
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
    cot = tuple(torch.randn_like(t) for t in actual)
    raw = (x, y, l, r)
    torch.testing.assert_close(torch.autograd.grad(actual, raw, cot),
                               torch.autograd.grad(expected, raw, cot), rtol=2e-7, atol=2e-8)
    rows, cols = torch.tensor([2, 0, 2], device=device), torch.tensor([1, 0], device=device)
    got = a.block(rows, cols)
    fresh = tp.compare(x, y, metric=metric, **kw).T.index_select(0, rows).index_select(1, cols)
    torch.testing.assert_close(got, fresh, rtol=1e-8, atol=1e-9)
    if metric in ('ssim', 'multiresolution_stft'):
        torch.testing.assert_close(torch.autograd.grad(got.sum(), (x, y)),
                                   torch.autograd.grad(fresh.sum(), (x, y)), rtol=2e-7, atol=2e-8)
    assert a.shape == dense.shape and a.dtype == dense.dtype and a.device == dense.device
    assert a.block(rows[:0], cols).shape == (0, 2)


def test_media_operator_broadcast_reductions_and_directed_transpose(device):
    torch.manual_seed(92)
    for metric in ('ssim', 'signal_snr'):
        x, y = inputs(metric, 2, device, (2, 1)), inputs(metric, 3, device, (1, 2))
        kw = options(metric)
        dense = tp.compare(x, y, metric=metric, **kw)
        a = tp.lazy(x, y, metric=metric, chunk_size=1, **kw)
        assert a.shape == dense.shape
        for transpose in (False, True):
            operator = a.T if transpose else a
            matrix = dense.transpose(-1, -2) if transpose else dense
            for dim in (-1, -2, None):
                source = matrix.flatten(-2) if dim is None else matrix
                axis = -1 if dim is None else dim
                for reduction in ('sum', 'mean', 'min', 'max', 'prod', 'all', 'any'):
                    ref = getattr(source, reduction)(axis)
                    if reduction in ('min', 'max'):
                        ref = ref.values
                    torch.testing.assert_close(getattr(operator, {'min': 'amin', 'max': 'amax'}.get(reduction, reduction))(dim), ref)
        selfop = tp.lazy(x, metric=metric, chunk_size=1, **kw)
        selfmatrix = tp.compare(x, metric=metric, **kw)
        torch.testing.assert_close(selfop.sum(exclude_self=True),
            selfmatrix.sum((-2, -1)) - selfmatrix.diagonal(dim1=-2, dim2=-1).sum(-1))
        coefficients = torch.randn(2, device=device, dtype=x.dtype)
        torch.testing.assert_close(selfop.quadratic(coefficients),
                                   torch.einsum('i,...ij,j->...', coefficients, selfmatrix, coefficients))
        l = torch.randn(1, 2, 2, 2, device=device, dtype=x.dtype, requires_grad=True)
        r = torch.randn(2, 1, 3, 2, device=device, dtype=x.dtype, requires_grad=True)
        actual, expected = a.bilinear(l, r), l.transpose(-1, -2) @ dense @ r
        torch.testing.assert_close(actual, expected)
        raw = (x, y, l, r)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), raw),
                                   torch.autograd.grad(expected.sum(), raw), rtol=2e-7, atol=2e-8)
        if metric == 'signal_snr':
            assert not torch.allclose(dense, tp.compare(y, x, metric=metric).transpose(-1, -2))


def test_media_operator_empty_validation_precision_and_special_values(device):
    torch.manual_seed(93)
    for metric in ('ssim', 'multiresolution_stft'):
        kw = options(metric)
        x = inputs(metric, 2, device)
        empty = x[:0]
        a = tp.lazy(x, empty, metric=metric, chunk_size=1, **kw)
        assert a.sum().item() == 0 and a.prod().item() == 1
        assert a.all().item() and not a.any().item() and a.mean().isnan()
        assert a.diagonal().shape == (0,)
        torch.testing.assert_close(torch.autograd.grad(a.sum(), x)[0], torch.zeros_like(x))
        with pytest.raises((RuntimeError, IndexError)):
            a.min()
        bad = x.detach().clone(); bad.flatten()[0] = float('nan')
        with pytest.raises((RuntimeError, ValueError)):
            tp.lazy(bad, empty, metric=metric, **kw).sum()
        with pytest.raises(TypeError):
            tp.lazy(x, metric=metric, squared=True, **kw)
        for dtype in (torch.float32, torch.float64):
            data = x.detach().to(dtype)
            op = tp.lazy(data, metric=metric, chunk_size=1, **kw)
            assert op.dtype == dtype
            torch.testing.assert_close(op.sum(-1), tp.compare(data, metric=metric, **kw).sum(-1))
        dtype = torch.bfloat16 if device == 'cpu' else torch.float16
        low = x.detach().to(dtype).requires_grad_()
        with torch.autocast(device, dtype=dtype):
            op = tp.lazy(low, metric=metric, chunk_size=1, **kw)
            value = op.sum()
            assert op.dtype == torch.float32 and value.dtype == torch.float32
        assert torch.autograd.grad(value, low)[0].isfinite().all()
    image = inputs('psnr', 2, device)
    a = tp.lazy(image, metric='psnr', data_range=1.0, chunk_size=1)
    assert a.diagonal().isposinf().all()
    assert a.amax(-1).isposinf().all()
    dense = tp.compare(image, metric='psnr', data_range=1.)
    masked = dense.masked_fill(torch.eye(2, device=device, dtype=torch.bool), 0.)
    torch.testing.assert_close(a.sum(exclude_self=True), masked.sum())
    with pytest.raises((RuntimeError, ValueError)):
        tp.lazy(image, metric='ssim').sum()
    # Select ties after rounding entries to the public float32 score dtype.
    x = image[:1].detach().float().requires_grad_()
    y = torch.stack((x[0] + 1e-4, x[0])).detach().requires_grad_()
    dense = tp.compare(x, y, metric='ssim', data_range=1., window_size=3)
    assert dense[0, 0] == dense[0, 1]
    actual = tp.lazy(x, y, metric='ssim', data_range=1.0, window_size=3, chunk_size=1).amax(-1)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), (x, y)),
                               torch.autograd.grad(dense.max(-1).values.sum(), (x, y)),
                               rtol=3e-5, atol=1e-9)


def test_media_operator_preparation_reuse_and_stft_layouts(device):
    torch.manual_seed(94)
    for metric in ('multiscale_ssim', 'multiresolution_stft'):
        x, y = inputs(metric, 2, device), inputs(metric, 3, device)
        kw = options(metric)
        a = tp.lazy(x, y, metric=metric, chunk_size=1, **kw)
        r = torch.randn(3, 2, device=device, dtype=x.dtype)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            result = a.matmul(r)
            grads = torch.autograd.grad(result.sum(), (x, y))
        names = [event.name for event in profile.events()]
        assert names.count('torchpairwise::_media_features') == 2
        assert names.count('torchpairwise::_media_features_backward') == 2
        dense = tp.compare(x, y, metric=metric, **kw) @ r
        torch.testing.assert_close(grads, torch.autograd.grad(dense.sum(), (x, y)), rtol=2e-7, atol=2e-8)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            tp.lazy(x, metric=metric, chunk_size=1, **kw).sum()
        assert sum(event.name == 'torchpairwise::_media_features' for event in profile.events()) == 1
        before = a.sum().detach()
        with torch.no_grad():
            x.mul_(.7)
        torch.testing.assert_close(a.sum(), tp.compare(x, y, metric=metric, **kw).sum())
        assert not torch.allclose(a.sum(), before)
    x, y = inputs('log_spectral', 2, device), inputs('log_spectral', 3, device)
    configurations = (
        ('log_spectral', dict(n_fft=9, hop_length=3, win_length=7, pad_mode='reflect')),
        ('spectral_convergence', dict(n_fft=8, hop_length=3, win_length=6, center=False, window='hamming')),
        ('multiresolution_stft', dict(resolutions=((8, None, None), (8, None, None), (9, 2, 7)), normalized=True)),
    )
    for metric, kw in configurations:
        a = tp.lazy(x, y, metric=metric, chunk_size=1, **kw)
        dense = tp.compare(x, y, metric=metric, **kw)
        torch.testing.assert_close(a.sum(-1), dense.sum(-1))
        torch.testing.assert_close(torch.autograd.grad(a.sum(), (x, y)),
                                   torch.autograd.grad(dense.sum(), (x, y)), rtol=2e-7, atol=2e-8)


@pytest.mark.compiler
@pytest.mark.parametrize('metric', ('ssim', 'multiscale_ssim', 'si_sdr', 'multiresolution_stft'))
def test_media_operator_compile_export_and_opcheck(device, metric):
    class Model(torch.nn.Module):
        def forward(self, x, y, l, r):
            a = tp.lazy(x, y, metric=metric, chunk_size=1, **options(metric)).T
            result=(a.matmul(r),a.bilinear(l,r),a.mean(-2),a.diagonal())
            if metric=='ssim':
                g=tp.image_map_operator(x,y,comparison='ssim',chunk_size=1,**options(metric))
                h=g.var(correction=0)
                result += (h.matmul(l),h.topk(2,dim=-2).values,h.sum(),
                           tp.image_map_operator(x,y,comparison='absolute_error',chunk_size=1).max().matmul(l),
                           g.mean((-2,-1)).max(-1).matmul(l),
                           g.var((-4,-2,-1),correction=0),
                           g.mean((-2,-1),keepdim=True).mean(-4))
            return result
    torch.manual_seed(95); torch._dynamo.reset()
    model = Model()
    batched = metric in ('ssim', 'si_sdr')
    batch = (2,) if batched else ()
    x, y = inputs(metric, 2, device, batch), inputs(metric, 3, device, batch)
    args = x, y, torch.randn(3, 2, device=device, dtype=x.dtype), torch.randn(2, 2, device=device, dtype=x.dtype)
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    actual, expected = compiled(*args), model(*args)
    torch.testing.assert_close(actual, expected)
    assert [t.requires_grad for t in actual]==[t.requires_grad for t in expected]
    loss = lambda values: sum(t.square().sum() for t in values)
    torch.testing.assert_close(torch.autograd.grad(loss(actual), (x, y)),
                               torch.autograd.grad(loss(expected), (x, y)), rtol=2e-6, atol=2e-7)
    n, m = torch.export.Dim('queries', min=2, max=6), torch.export.Dim('references', min=2, max=6)
    b = torch.export.Dim('batches', min=1, max=4)
    shape = lambda size: {0:b, 1:size} if batched else {0:size}
    graph = torch.export.export(model, args, dynamic_shapes=(shape(n), shape(m), {0:m}, {0:n}))
    buffer = io.BytesIO(); torch.export.save(graph, buffer); buffer.seek(0)
    restored = torch.export.load(buffer).module()
    other_batch = (3,) if batched else ()
    other = inputs(metric, 4, device, other_batch), inputs(metric, 2, device, other_batch), torch.randn(2, 2, device=device, dtype=x.dtype), torch.randn(4, 2, device=device, dtype=x.dtype)
    torch.testing.assert_close(restored(*other), model(*other))
    if metric == 'ssim':
        torch.library.opcheck(torch.ops.torchpairwise._media_features.default,
            (x, 1, [2, 8, 10, 3], [1., 1.5, .01, .03]))
        packed=torch.ops.torchpairwise._media_features(x,1,[2,8,10,3],[1.,1.5,.01,.03]).reshape(-1,2*8*10+2*8*10+3*2*6*8+2)
        torch.library.opcheck(torch.ops.torchpairwise._image_map_values.default,
            (packed[:2],packed[2:],3,[2,8,10,3],[1.,1.5,.01,.03],7,False))
        torch.library.opcheck(torch.ops.torchpairwise._image_event_reduce.default,
            (packed[:2],packed[2:],3,[2,8,10,3],[1.,1.5,.01,.03],7,
             [1,6,0],[1.],'var',[1],False,0.,1,False))
        torch.library.opcheck(torch.ops.torchpairwise._image_map_reduce.default,
            (packed[:2],packed[2:],3,'var',[2,8,10,3],[1.,1.5,.01,.03],7,.5,False))

        raw_gradient=torch.rand(2,2,160,device=device,dtype=x.dtype)
        raw=x.detach()
        with torch.no_grad():
            torch.library.opcheck(torch.ops.torchpairwise._media_features_backward.default,
                (raw_gradient,raw,0,[2,8,10,3],[1.,1.5,.01,.03]),
                test_utils=('test_schema','test_faketensor'))
