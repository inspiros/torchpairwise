"""Empirical distribution comparisons against explicit finite-sum formulas."""
import pytest
import torch
import torchpairwise as tp


_CASES = [('energy', None), ('mmd', 'rbf'), ('mmd', 'laplacian'), ('mmd', 'linear')]


def _options(metric, kernel, squared=False):
    result = dict(squared=squared)
    if metric == 'mmd':
        result.update(kernel=kernel, length_scale=.7)
    return result


def _reference(x, y, metric, kernel, squared=False, length_scale=.7):
    def mean(a, b):
        difference = a[:, None, :] - b[None, :, :]
        if kernel == 'linear':
            value = (a[:, None, :] * b[None, :, :]).sum(-1)
        elif kernel == 'rbf':
            value = (-difference.square().sum(-1) / (2 * length_scale ** 2)).exp()
        else:
            value = torch.linalg.vector_norm(difference, dim=-1)
            if kernel == 'laplacian':
                value = (-value / length_scale).exp()
        return value.mean()
    cross, xx, yy = mean(x, y), mean(x, x), mean(y, y)
    value = (2 * cross - xx - yy if metric == 'energy' else xx + yy - 2 * cross).clamp_min(0)
    return value if squared else value.sqrt()


@pytest.mark.parametrize('metric,kernel', _CASES)
@pytest.mark.parametrize('squared', [False, True])
def test_distribution_formula_batches_layout_modes_and_gradients(metric, kernel, squared, device):
    torch.manual_seed(271)
    x = torch.rand(2, 3, 4, 2, device=device, dtype=torch.float64)
    y = torch.rand(1, 3, 3, 2, device=device, dtype=x.dtype) + .9
    xl = torch.tensor([[4, 1, 2], [3, 2, 4]], device=device)
    yl = torch.tensor([[2, 3, 1]], device=device)
    x[torch.arange(4, device=device) >= xl[..., None]] = float('nan')
    y[torch.arange(3, device=device) >= yl[..., None]] = float('nan')
    x = x.reshape(2, 3, 2, 2, 2).movedim(-1, 2).contiguous().requires_grad_()
    y = y.reshape(1, 3, 1, 3, 2).movedim(-1, 2).contiguous().requires_grad_()
    xf, yf = x.movedim(2, -1).flatten(2, -2), y.movedim(2, -1).flatten(2, -2)
    options = _options(metric, kernel, squared)
    expected = torch.stack([torch.stack([torch.stack([
        _reference(xf[b, i, :int(xl[b, i])], yf[0, j, :int(yl[0, j])],
                   metric, kernel, squared) for j in range(3)]) for i in range(3)]) for b in range(2)])
    options.update(x1_lengths=xl, x2_lengths=yl, batched=True, feature_dim=2)
    actual = getattr(tp, metric + '_distances')(x, y, **options)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(tp.cdist(x, y, metric=metric, chunk_size=2, **options), expected)
    paired = tp.paired_distances(x, y, metric=metric, chunk_size=2, **options)
    torch.testing.assert_close(paired, expected.diagonal(dim1=-2, dim2=-1))
    for result, oracle in ((actual, expected), (paired, expected.diagonal(dim1=-2, dim2=-1))):
        for grad, ref in zip(torch.autograd.grad(result.sum(), (x, y), retain_graph=True),
                             torch.autograd.grad(oracle.sum(), (x, y), retain_graph=True)):
            torch.testing.assert_close(grad, ref)
            assert torch.isfinite(grad).all()
    options.pop('x2_lengths')
    full = getattr(tp, metric + '_distances')(x, **options)
    i, j = torch.triu_indices(3, 3, 1, device=device)
    condensed = tp.pdist(x, metric=metric, chunk_size=2, **options)
    torch.testing.assert_close(condensed, full[..., i, j])
    torch.testing.assert_close(torch.autograd.grad(condensed.sum(), x)[0],
                               torch.autograd.grad(full[..., i, j].sum(), x)[0])


@pytest.mark.parametrize('metric,kernel', _CASES)
def test_distribution_singletons_repetition_zero_and_empty(metric, kernel, device):
    x = torch.tensor([[[0., 0.]], [[3., 4.]]], device=device, dtype=torch.float64,
                     requires_grad=True)
    options = _options(metric, kernel)
    call = getattr(tp, metric + '_distances')
    result = call(x, **options)
    if metric == 'energy':
        expected_squared = 10.
    elif kernel == 'rbf':
        expected_squared = 2 * (1 - torch.exp(torch.tensor(-25 / (2 * .7 ** 2))).item())
    elif kernel == 'laplacian':
        expected_squared = 2 * (1 - torch.exp(torch.tensor(-5 / .7)).item())
    else:
        expected_squared = 25.
    torch.testing.assert_close(result[0, 1].square(), x.new_tensor(expected_squared))
    torch.testing.assert_close(call(x.repeat_interleave(3, -2), **options), result)
    zeros = tp.paired_distances(x, x, metric=metric, **options)
    torch.testing.assert_close(zeros, torch.zeros_like(zeros))
    torch.testing.assert_close(torch.autograd.grad(zeros.sum(), x)[0], torch.zeros_like(x))
    duplicate = torch.ones(2, 3, 2, device=device, dtype=x.dtype, requires_grad=True)
    zero = call(duplicate, **options)
    torch.testing.assert_close(zero, torch.zeros_like(zero))
    torch.testing.assert_close(torch.autograd.grad(zero.sum(), duplicate)[0], torch.zeros_like(duplicate))
    for n in (0, 1):
        empty = tp.pdist(x[:n], metric=metric, **options)
        assert empty.shape == (0,)
        torch.testing.assert_close(torch.autograd.grad(empty.sum(), x)[0], torch.zeros_like(x))
    assert call(x[:0], x, **options).shape == (0, 2)
    assert tp.paired_distances(x[:0], x[:0], metric=metric, **options).shape == (0,)
    assert call(x.expand(0, *x.shape), x.unsqueeze(0), **options).shape == (0, 2, 2)


@pytest.mark.parametrize('metric,kernel', _CASES)
def test_distribution_first_derivatives_and_symmetry(metric, kernel, device):
    torch.manual_seed(713)
    x = torch.randn(1, 2, 2, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.randn(1, 3, 2, dtype=x.dtype, device=device, requires_grad=True)
    call = getattr(tp, metric + '_distances')
    options = _options(metric, kernel)
    assert torch.autograd.gradcheck(lambda a, b: call(a, b, **options), (x, y))
    torch.testing.assert_close(call(x, y, **options), call(y, x, **options).T)


def test_distribution_linear_kernel_only_compares_means(device):
    x = torch.tensor([[[-1.], [1.]]], device=device)
    y = torch.tensor([[[0.], [0.]]], device=device)
    assert tp.mmd_distances(x, y, kernel='linear').item() == 0
    assert tp.mmd_distances(x, y, kernel='rbf').item() > 0
    assert tp.energy_distances(x, y).item() > 0


def test_distribution_padding_large_translation_gradients(device):
    x = torch.tensor([[[1e20], [float('nan')]]], device=device, requires_grad=True)
    y = torch.tensor([[[1e20]]], device=device, requires_grad=True)
    lengths = torch.tensor([1], device=device)
    for metric, kernel in _CASES:
        result = getattr(tp, metric + '_distances')(
            x, y, x1_lengths=lengths, **_options(metric, kernel))
        torch.testing.assert_close(result, torch.zeros_like(result))
        for grad in torch.autograd.grad(result.sum(), (x, y)):
            torch.testing.assert_close(grad, torch.zeros_like(grad))


@pytest.mark.parametrize('metric,kernel', _CASES[:-1])
def test_distribution_point_tiles_forward_backward(metric, kernel, device):
    torch.manual_seed(891)
    # Cross the native point-tile boundary and compare values/first gradients
    # against an independent dense formula. Profiling observes ATen work in C++.
    x = torch.rand(1, 129, 2, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(1, 131, 2, device=device, dtype=x.dtype, requires_grad=True)
    expected = _reference(x[0], y[0], metric, kernel, squared=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as forward:
        actual = getattr(tp, metric + '_distances')(x, y, **_options(metric, kernel, squared=True))
    torch.testing.assert_close(actual[0, 0], expected)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as backward:
        gradients = torch.autograd.grad(actual.sum(), (x, y))
    for grad, ref in zip(gradients, torch.autograd.grad(expected, (x, y))):
        torch.testing.assert_close(grad, ref)
    for profile in (forward, backward):
        calls = [event for event in profile.events() if event.name in ('aten::cdist', 'torchpairwise::_ppminkowski')]
        assert calls
        assert all(event.input_shapes[0][-2] <= 128 and event.input_shapes[1][-2] <= 128 for event in calls)


@pytest.mark.parametrize('metric,kernel', _CASES)
def test_distribution_validation_and_amp(metric, kernel, device):
    call = getattr(tp, metric + '_distances')
    x = torch.ones(2, 3, 2, device=device)
    options = _options(metric, kernel)
    with pytest.raises(TypeError, match='squared'):
        call(x, **dict(options, squared=1))
    for bad in (float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite'):
            call(x * bad, **options)
    if metric == 'mmd':
        for bad in ('unsupported', None, 3):
            with pytest.raises(ValueError, match='kernel'):
                call(x, **dict(options, kernel=bad))
        for bad in (0., -1., float('inf'), float('nan'), True, torch.tensor(1.)):
            with pytest.raises(ValueError, match='length_scale'):
                call(x, **dict(options, length_scale=bad))
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    low = torch.rand(2, 3, 2, device=device, dtype=dtype, requires_grad=True)
    with pytest.raises(TypeError, match='float32 or float64'):
        call(low, **options)
    with torch.autocast(device, dtype=dtype):
        actual = call(low, low + 1, **options)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, call(low.float(), (low + 1).float(), **options))
    assert torch.isfinite(torch.autograd.grad(actual.sum(), low)[0]).all()


@pytest.mark.compiler
@pytest.mark.parametrize('metric,kernel', _CASES)
def test_distribution_compile_ragged_modes(metric, kernel, device):
    torch.compiler.reset()
    x = torch.rand(2, 3, 2, device=device, requires_grad=True)
    y = torch.rand(2, 4, 2, device=device, requires_grad=True) + .5
    xl = torch.tensor([2, 3], device=device)
    yl = torch.tensor([4, 2], device=device)
    options = _options(metric, kernel, squared=True)
    def call(a, b, al, bl):
        kwargs = dict(options, x1_lengths=al, x2_lengths=bl, chunk_size=1)
        return (getattr(tp, metric + '_distances')(a, b, **kwargs).sum()
                + tp.paired_distances(a, b, metric=metric, **kwargs).sum()
                + tp.pdist(a, metric=metric, x1_lengths=al, chunk_size=1, **options).sum())
    expected = call(x, y, xl, yl)
    actual = torch.compile(call, fullgraph=True)(x, y, xl, yl)
    torch.testing.assert_close(actual, expected)
    for grad, ref in zip(torch.autograd.grad(actual, (x, y)), torch.autograd.grad(expected, (x, y))):
        torch.testing.assert_close(grad, ref)
    torch.compiler.reset()
