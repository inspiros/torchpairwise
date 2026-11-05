"""Signed empirical MMD estimates, including the polynomial statistic used by KID."""
import io
import pytest
import torch
import torchpairwise as tp


KERNELS = [
    ('polynomial', dict(degree=3, gamma=.4, coef0=.7)),
    ('rbf', dict(gamma=.6)),
    ('laplacian_kernel', dict(gamma=.8)),
    ('linear', {}),
    ('cosine', {}),
    ('rational_quadratic', dict(length_scale=.8, alpha=.7)),
    ('matern', dict(length_scale=.9, nu=2.5)),
    ('histogram_intersection', {}),
    ('chi2', dict(gamma=.3)),
    ('additive_chi2', {}),
    ('bhattacharyya', {}),
    ('laplacian', dict(length_scale=.8)),
]


def reference(a, b, kernel, options, estimator='unbiased'):
    def gram(x, y):
        if kernel == 'laplacian':
            return (-torch.linalg.vector_norm(x[:, None] - y[None], dim=-1) / options['length_scale']).exp()
        return tp.pairwise_kernels(x, y, kernel='laplacian' if kernel == 'laplacian_kernel' else kernel, **options)
    xx, yy, xy = gram(a, a), gram(b, b), gram(a, b)
    if estimator == 'unbiased':
        n, m = a.shape[0], b.shape[0]
        return ((xx.sum() - xx.diagonal().sum()) / (n * (n - 1))
                + (yy.sum() - yy.diagonal().sum()) / (m * (m - 1)) - 2 * xy.mean())
    return xx.mean() + yy.mean() - 2 * xy.mean()


@pytest.mark.parametrize('kernel,options', KERNELS)
def test_mmd_statistics_kernels_layout_lengths_modes_and_gradients(kernel, options, device):
    torch.manual_seed(1701)
    x = torch.rand(2, 3, 4, 2, dtype=torch.float64, device=device) + .2
    y = torch.rand(1, 3, 4, 2, dtype=x.dtype, device=device) + .4
    xl = torch.tensor([[2, 4, 3], [3, 2, 4]], device=device)
    yl = torch.tensor([[4, 2, 3]], device=device)
    x[torch.arange(4, device=device) >= xl[..., None]] = float('nan')
    y[torch.arange(4, device=device) >= yl[..., None]] = float('nan')
    # Noncontiguous channel-first spatial layout, with broadcasted batches.
    x = x.reshape(2, 3, 2, 2, 2).movedim(-1, 2).requires_grad_()
    y = y.reshape(1, 3, 2, 2, 2).movedim(-1, 2).requires_grad_()
    xf, yf = x.movedim(2, -1).flatten(2, -2), y.movedim(2, -1).flatten(2, -2)
    for estimator in ('unbiased', 'biased'):
        expected = torch.stack([torch.stack([torch.stack([
            reference(xf[b, i, :xl[b, i]], yf[0, j, :yl[0, j]], kernel, options, estimator)
            for j in range(3)]) for i in range(3)]) for b in range(2)])
        kwargs = dict(kernel=kernel, estimator=estimator, batched=True, feature_dim=2,
                      x1_lengths=xl, x2_lengths=yl, **options)
        actual = tp.mmd_statistics(x, y, **kwargs)
        torch.testing.assert_close(actual, expected)
        paired = tp.paired_mmd_statistics(x, y, **kwargs)
        torch.testing.assert_close(paired, expected.diagonal(dim1=-2, dim2=-1))
        weights = torch.randn_like(expected)
        for a, e in zip(torch.autograd.grad((actual * weights).sum(), (x, y)),
                        torch.autograd.grad((expected * weights).sum(), (x, y))):
            torch.testing.assert_close(a, e)
            assert a.isfinite().all()
        kwargs.pop('x2_lengths')
        full = tp.mmd_statistics(x, **kwargs)
        i, j = torch.triu_indices(3, 3, 1, device=device)
        condensed = tp.condensed_mmd_statistics(x, **kwargs)
        torch.testing.assert_close(condensed, full[..., i, j])
        torch.testing.assert_close(tp.compare(x, metric='mmd_statistic', mode='condensed', **kwargs), condensed)
        torch.testing.assert_close(torch.autograd.grad(condensed.sum(), x)[0],
                                   torch.autograd.grad(full[..., i, j].sum(), x)[0])


def test_mmd_statistics_signed_polynomial_and_degenerate_boundaries(device):
    x = torch.tensor([[[-1.], [1.]]], dtype=torch.float64, device=device, requires_grad=True)
    # For k(a,b)=(a*b+1)^3, within off-diagonal=0 and cross mean=4.
    result = tp.mmd_statistics(x, x, kernel='polynomial', gamma=1.)
    torch.testing.assert_close(result, x.new_tensor([[-8.]]))
    assert (result < 0).all()
    expected = reference(x[0], x[0], 'polynomial', dict(gamma=1.))
    torch.testing.assert_close(torch.autograd.grad(result.sum(), x)[0],
                               torch.autograd.grad(expected, x)[0])
    for kernel in ('linear', 'polynomial', 'rbf'):
        constant = torch.ones(2, 3, 2, dtype=x.dtype, device=device, requires_grad=True)
        zero = tp.mmd_statistics(constant, kernel=kernel)
        torch.testing.assert_close(zero, torch.zeros_like(zero), atol=1e-14, rtol=0)
        torch.testing.assert_close(torch.autograd.grad(zero.sum(), constant)[0],
                                   torch.zeros_like(constant), atol=1e-14, rtol=0)
        for n in (0, 1):
            empty = tp.condensed_mmd_statistics(constant[:n], kernel=kernel)
            assert empty.shape == (0,)
            torch.testing.assert_close(torch.autograd.grad(empty.sum(), constant)[0], torch.zeros_like(constant))
    assert tp.mmd_statistics(x[:0], x).shape == (0, 1)
    assert tp.paired_mmd_statistics(x[:0], x[:0]).shape == (0,)
    # Biased estimates permit singleton clouds and remain the squared distance statistic.
    singleton = x[:, :1]
    torch.testing.assert_close(tp.mmd_statistics(singleton, estimator='biased'),
                               tp.mmd_distances(singleton, squared=True))


def test_mmd_statistics_point_tiles_explicit_polynomial_gradcheck(device):
    torch.manual_seed(941)
    x = torch.rand(1, 129, 2, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.rand(1, 131, 2, dtype=x.dtype, device=device, requires_grad=True)
    def polynomial(a, b):
        return (.3 * a.matmul(b.T) + 1).pow(3)
    a, b = x[0], y[0]
    xx, yy = polynomial(a, a), polynomial(b, b)
    expected = ((xx.sum() - xx.diagonal().sum()) / (129 * 128)
                + (yy.sum() - yy.diagonal().sum()) / (131 * 130)
                - 2 * polynomial(a, b).mean())
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
        actual = tp.mmd_statistics(x, y, kernel='polynomial', gamma=.3)
        gradients = torch.autograd.grad(actual.sum(), (x, y))
    torch.testing.assert_close(actual.squeeze(), expected)
    for a, e in zip(gradients, torch.autograd.grad(expected, (x, y))):
        torch.testing.assert_close(a, e)
    calls = [e for e in profile.events() if e.name == 'torchpairwise::_legacy_compare']
    assert calls and all(e.input_shapes[0][-2] <= 128 and e.input_shapes[1][-2] <= 128 for e in calls)
    assert torch.autograd.gradcheck(lambda a, b: tp.mmd_statistics(a, b, kernel='polynomial', gamma=.3),
                                   (x[:, :2].detach().requires_grad_(), y[:, :3].detach().requires_grad_()))


def test_mmd_statistics_validation_and_amp(device):
    x = torch.ones(2, 3, 2, device=device)
    for call in (tp.mmd_statistics, torch.ops.torchpairwise.mmd_statistics):
        with pytest.raises((RuntimeError, ValueError), match='estimator'):
            call(x, estimator='other')
        with pytest.raises(RuntimeError, match='two valid points'):
            call(x[:, :1])
        for bad in (0, 1, 4):
            with pytest.raises(RuntimeError, match='two valid points'):
                call(x, x1_lengths=torch.tensor([bad, 3], device=device))
        with pytest.raises(RuntimeError, match='Lengths'):
            call(x, estimator='biased', x1_lengths=torch.tensor([0, 3], device=device))
        with pytest.raises(RuntimeError, match='sigmoid'):
            call(x, kernel='sigmoid')
    with pytest.raises(TypeError, match='degree'):
        tp.mmd_statistics(x, kernel='polynomial', degree=True)
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    low = torch.rand(2, 3, 2, dtype=dtype, device=device, requires_grad=True)
    with pytest.raises(RuntimeError, match='float32 or float64'):
        tp.mmd_statistics(low)
    with torch.autocast(device, dtype=dtype):
        actual = tp.mmd_statistics(low, low + .5, kernel='polynomial')
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, tp.mmd_statistics(low.float(), (low + .5).float(), kernel='polynomial'))
    assert torch.autograd.grad(actual.sum(), low)[0].isfinite().all()
    # Second derivatives are checked in test_higher_order.py.
    value = tp.mmd_statistics(x.requires_grad_(), kernel='polynomial')
    assert torch.autograd.grad(value.sum(), x, create_graph=True)[0].requires_grad



@pytest.mark.compiler
def test_mmd_statistics_native_contract_and_export(device):
    torch.manual_seed(2717)
    x, y = [torch.rand(3, p, 2, device=device, requires_grad=True) for p in (4, 5)]
    empty = torch.empty(0, device=device, dtype=torch.int64)
    args = (x, y, empty, empty, 'polynomial', 0, .4, 3, 1., 1., 1., 1.5, True)
    checked = torch.library.opcheck(torch.ops.torchpairwise._mmd_compare.statistic, args)
    assert all(v == 'SUCCESS' for v in checked.values())
    for mode in ('all_pairs', 'paired', 'condensed'):
        class Statistic(torch.nn.Module):
            def forward(self, a, b, al, bl):
                options = dict(kernel='polynomial', gamma=.4, x1_lengths=al)
                if mode != 'condensed':
                    options['x2_lengths'] = bl
                return tp.compare(a, None if mode == 'condensed' else b,
                                  metric='mmd_statistics', mode=mode, **options)
        xl = torch.tensor([2, 3, 4], device=device)
        yl = torch.tensor([3, 4, 5], device=device)
        model = Statistic()
        compiled = torch.compile(model, backend='aot_eager', fullgraph=True)
        actual, expected = compiled(x, y, xl, yl), model(x, y, xl, yl)
        torch.testing.assert_close(actual, expected)
        operands = (x,) if mode == 'condensed' else (x, y)
        for a, e in zip(torch.autograd.grad(actual.sum(), operands), torch.autograd.grad(expected.sum(), operands)):
            torch.testing.assert_close(a, e)
        torch.compiler.reset()
        n = torch.export.Dim('n', min=3, max=6)
        p = torch.export.Dim('p', min=2, max=8)
        q = torch.export.Dim('q', min=2, max=8)
        exported = torch.export.export(model, (x, y, xl, yl),
                                       dynamic_shapes=({0: n, 1: p}, {0: n, 1: q}, {0: n}, {0: n}))
        buffer = io.BytesIO()
        torch.export.save(exported, buffer)
        buffer.seek(0)
        loaded = torch.export.load(buffer).module()
        a, b = torch.rand(4, 6, 2, device=device), torch.rand(4, 7, 2, device=device)
        al, bl = torch.tensor([2, 3, 4, 2], device=device), torch.tensor([3, 4, 5, 2], device=device)
        torch.testing.assert_close(loaded(a, b, al, bl), model(a, b, al, bl))
        invalid = al.clone()
        invalid[0] = 1
        with pytest.raises(RuntimeError, match='two valid points'):
            loaded(a, b, invalid, bl)
        # A rejected call must leave the CUDA context usable for a valid call.
        torch.testing.assert_close(loaded(a, b, al, bl), model(a, b, al, bl))
