"""Independent KL references, support boundaries, and public execution modes."""
from decimal import Decimal, localcontext
import numpy as np
import pytest
from scipy.special import rel_entr
import torch
import torchpairwise as tp


def named(kind):
    return tp.kl_divergences if kind == 'kl' else tp.symmetric_kl_divergences


def reference(x, y, symmetric=False):
    p = x / x.sum(-1, keepdims=True)
    q = y / y.sum(-1, keepdims=True)
    result = rel_entr(p[..., :, None, :], q[..., None, :, :]).sum(-1)
    if symmetric:
        result = result + rel_entr(q[..., None, :, :], p[..., :, None, :]).sum(-1)
    return result


@pytest.mark.parametrize('kind', ['kl', 'symmetric_kl'])
def test_kl_values_broadcast_modes_and_gradients(kind, device):
    torch.manual_seed(73)
    fn = named(kind)
    for dtype in (torch.float32, torch.float64):
        x = (torch.rand(2, 1, 3, 6, device=device, dtype=dtype) + .1)[..., ::2].requires_grad_()
        y = (torch.rand(1, 2, 4, 6, device=device, dtype=dtype) + .2)[..., ::2].requires_grad_()
        actual = fn(x, y)
        expected = reference(x.detach().cpu().double().numpy(), y.detach().cpu().double().numpy(),
                             kind == 'symmetric_kl')
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=dtype, device=device),
                                   atol=2e-6 if dtype == torch.float32 else 1e-12, rtol=1e-5)
        torch.testing.assert_close(tp.cdist(x, y, metric=kind), actual)
        p, q = x / x.sum(-1, keepdim=True), y / y.sum(-1, keepdim=True)
        direct = (p.unsqueeze(-2) * (p.unsqueeze(-2).log() - q.unsqueeze(-3).log())).sum(-1)
        if kind == 'symmetric_kl':
            direct = direct + (q.unsqueeze(-3) * (q.unsqueeze(-3).log() - p.unsqueeze(-2).log())).sum(-1)
        weights = torch.linspace(.2, 1., actual.numel(), device=device, dtype=dtype).reshape(actual.shape)
        for grad, ref in zip(torch.autograd.grad((actual * weights).sum(), (x, y)),
                             torch.autograd.grad((direct * weights).sum(), (x, y))):
            torch.testing.assert_close(grad, ref, atol=2e-6, rtol=1e-5)
        paired = tp.paired_distances(x, y[..., :3, :], metric=kind)
        # Use new forward graphs after the independent all-pairs gradient check.
        diagonal = fn(x, y[..., :3, :]).diagonal(dim1=-2, dim2=-1)
        torch.testing.assert_close(paired, diagonal, atol=2e-6, rtol=1e-5)
        for grad, ref in zip(torch.autograd.grad(paired.sum(), (x, y)),
                             torch.autograd.grad(diagonal.sum(), (x, y))):
            torch.testing.assert_close(grad, ref, atol=2e-6, rtol=1e-5)
        if kind == 'symmetric_kl':
            i, j = torch.triu_indices(3, 3, 1, device=device)
            condensed = tp.condensed_distances(x, metric=kind)
            triangle = fn(x)[..., i, j]
            torch.testing.assert_close(condensed, triangle, atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(torch.autograd.grad(condensed.sum(), x)[0],
                                       torch.autograd.grad(triangle.sum(), x)[0], atol=2e-6, rtol=1e-5)
    a, b = [(torch.rand(2, 3, device=device, dtype=torch.float64) + .2).requires_grad_()
            for _ in range(2)]
    assert torch.autograd.gradcheck(fn, (a, b), fast_mode=True)


def test_kl_direction_normalization_and_log_base(device):
    x = torch.tensor([[1., 3., 6.], [3., 2., 1.]], device=device, dtype=torch.float64)
    y = torch.tensor([[5., 4., 1.]], device=device, dtype=torch.float64)
    forward, reverse = tp.kl_divergences(x, y), tp.kl_divergences(y, x).T
    assert not torch.allclose(forward, reverse)
    jeffreys = tp.symmetric_kl_divergences(x, y)
    torch.testing.assert_close(jeffreys, forward + reverse)
    for fn in (tp.kl_divergences, tp.symmetric_kl_divergences):
        value = fn(x, y)
        torch.testing.assert_close(fn(x * torch.tensor([[2.], [7.]], device=device), y * 4), value)
        torch.testing.assert_close(fn(x, y, base=2), value / np.log(2))
        torch.testing.assert_close(tp.paired_distances(x, y.expand_as(x),
                                                     metric='kl' if fn is tp.kl_divergences else 'symmetric_kl',
                                                     base=10), value[:, 0] / np.log(10))
    with pytest.raises(ValueError, match='symmetric'):
        tp.condensed_distances(x, metric='kl')


@pytest.mark.parametrize('kind', ['kl', 'symmetric_kl'])
def test_kl_near_identity_keeps_small_values_and_gradients(kind, device):
    # The divergence is much smaller than the individual entropy terms. Decimal
    # supplies an independent reference without their subtractive cancellation.
    for dtype, delta, tolerance in ((torch.float32, 1e-3, .002),
                                    (torch.float64, 1e-7, 1e-7)):
        x = torch.tensor([[2., 3., 5.]], dtype=dtype, device=device, requires_grad=True)
        y = torch.tensor([[2. + delta, 3. - delta, 5.]], dtype=dtype, device=device, requires_grad=True)
        with localcontext() as context:
            context.prec = 60
            a = [Decimal(float(v)) for v in x.detach().cpu()[0]]
            b = [Decimal(float(v)) for v in y.detach().cpu()[0]]
            p, q = [v / sum(a) for v in a], [v / sum(b) for v in b]
            logs = [(u / v).ln() for u, v in zip(p, q)]
            forward = sum(u * log for u, log in zip(p, logs))
            reverse = -sum(v * log for v, log in zip(q, logs))
            expected = forward + (reverse if kind == 'symmetric_kl' else 0)
            gx = [(log - forward) / sum(a) for log in logs]
            gy = [(1 - u / v) / sum(b) for u, v in zip(p, q)]
            if kind == 'symmetric_kl':
                gx = [g + (1 - v / u) / sum(a) for g, u, v in zip(gx, p, q)]
                gy = [g + (-log - reverse) / sum(b) for g, log in zip(gy, logs)]
        actual = named(kind)(x, y)
        assert actual.item() > 0
        torch.testing.assert_close(actual, torch.tensor([[float(expected)]], dtype=dtype, device=device),
                                   atol=0, rtol=tolerance)
        for grad, ref in zip(torch.autograd.grad(actual.sum(), (x, y)), (gx, gy)):
            torch.testing.assert_close(grad, torch.tensor([[float(v) for v in ref]], dtype=dtype, device=device),
                                       atol=1e-12 if dtype == torch.float32 else 1e-17, rtol=tolerance)


def test_kl_pair_and_feature_tile_boundaries(device):
    torch.manual_seed(9)
    x = (torch.rand(33, 65, dtype=torch.float64, device=device) + .1).requires_grad_()
    y = (torch.rand(2, 65, dtype=torch.float64, device=device) + .1).requires_grad_()
    actual = tp.kl_divergences(x, y)
    expected = reference(x.detach().cpu().numpy(), y.detach().cpu().numpy())
    torch.testing.assert_close(actual, torch.as_tensor(expected, device=device))
    p, q = x / x.sum(-1, keepdim=True), y / y.sum(-1, keepdim=True)
    direct = (p[:, None] * (p[:, None].log() - q[None].log())).sum(-1)
    for grad, ref in zip(torch.autograd.grad(actual.sum(), (x, y)),
                         torch.autograd.grad(direct.sum(), (x, y))):
        torch.testing.assert_close(grad, ref)


@pytest.mark.parametrize('kind', ['kl', 'symmetric_kl'])
def test_kl_zero_support_and_infinite_pair_gradients(kind, device):
    fn = named(kind)
    x = torch.tensor([[0., 1., 0.], [1., 0., 0.], [1., 1., 0.]],
                     device=device, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[0., 1., 0.], [1., 1., 0.], [0., 0., 1.]],
                     device=device, dtype=torch.float64, requires_grad=True)
    actual = fn(x, y)
    expected = reference(x.detach().cpu().numpy(), y.detach().cpu().numpy(), kind == 'symmetric_kl')
    torch.testing.assert_close(actual, torch.as_tensor(expected, device=device))
    # An infinite comparison contributes no gradients, even when only one of
    # the two directional KL terms is infinite in the symmetric case.
    for grad in torch.autograd.grad(fn(x[:1], y[1:2] if kind == 'symmetric_kl' else y[2:]).sum(), (x, y)):
        torch.testing.assert_close(grad, torch.zeros_like(grad))
    gradients = torch.autograd.grad(actual.sum(), (x, y))
    for grad, value in zip(gradients, (x, y)):
        assert torch.isfinite(grad).all()
        assert grad[value == 0].eq(0).all()
    finite = torch.where(torch.isfinite(fn(x, y)), fn(x, y), 0.).sum()
    for grad, ref in zip(gradients, torch.autograd.grad(finite, (x, y))):
        torch.testing.assert_close(grad, ref)
    for grad in torch.autograd.grad(fn(x, x).diagonal().sum(), x):
        torch.testing.assert_close(grad, torch.zeros_like(grad))
    torch.testing.assert_close(tp.paired_distances(x, y, metric=kind), actual.diagonal())
    if kind == 'symmetric_kl':
        i, j = torch.triu_indices(3, 3, 1, device=device)
        torch.testing.assert_close(tp.condensed_distances(x, metric=kind), fn(x)[i, j])
    # Raw positive support must survive probability underflow: both directional
    # divergences stay finite despite this very small normalized first bin.
    tiny = torch.finfo(torch.float32).tiny
    a = torch.tensor([[tiny, 1e20]], device=device)
    b = torch.tensor([[1., 1.]], device=device)
    assert torch.isfinite(fn(a, b)).all()
    assert torch.isfinite(fn(b, a)).all()


@pytest.mark.parametrize('kind', ['kl', 'symmetric_kl'])
def test_kl_empty_and_amp(kind, device):
    fn = named(kind)
    x, y = [(torch.rand(3, 4, device=device) + .1).requires_grad_() for _ in range(2)]
    for a, b, shape in ((x[:0], y, (0, 3)), (x, y[:0], (3, 0)),
                        (x.expand(0, *x.shape), y, (0, 3, 3))):
        value = fn(a, b)
        assert value.shape == shape
        for grad in torch.autograd.grad(value.sum(), (x, y)):
            torch.testing.assert_close(grad, torch.zeros_like(grad))
    value = tp.paired_distances(x[:0], y[:0], metric=kind)
    assert value.shape == (0,)
    for grad in torch.autograd.grad(value.sum(), (x, y)):
        torch.testing.assert_close(grad, torch.zeros_like(grad))
    if kind == 'symmetric_kl':
        for n in (0, 1):
            value = tp.condensed_distances(x[:n], metric=kind)
            assert value.shape == (0,)
            torch.testing.assert_close(torch.autograd.grad(value.sum(), x)[0], torch.zeros_like(x))
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    a, b = [value.detach().to(dtype).requires_grad_() for value in (x, y)]
    calls = [(fn, (a, b), {}), (tp.paired_distances, (a, b), {'metric': kind})]
    if kind == 'symmetric_kl':
        calls.append((tp.condensed_distances, (a,), {'metric': kind}))
    for call, args, options in calls:
        with torch.autocast(device, dtype=dtype):
            actual = call(*args, **options)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, call(*(v.float() for v in args), **options))
        assert torch.isfinite(torch.autograd.grad(actual.sum(), a)[0]).all()


@pytest.mark.parametrize('kind', ['kl', 'symmetric_kl'])
def test_kl_domain_validation(kind, device):
    fn = named(kind)
    x = torch.ones(2, 3, device=device)
    for value in (-1., float('inf'), float('nan')):
        bad = x.clone()
        bad[0, 0] = value
        with pytest.raises(ValueError, match='finite and non-negative'):
            fn(x, bad)
    for bad in (torch.zeros_like(x), torch.full_like(x, torch.finfo(x.dtype).max)):
        with pytest.raises(ValueError, match='positive finite row mass'):
            fn(bad)
    with pytest.raises(TypeError, match='dense strided'):
        fn(x.to_sparse())
    for base in (0, 1, -2, float('nan')):
        with pytest.raises(ValueError, match='base'):
            fn(x, base=base)


@pytest.mark.compiler
def test_kl_compiled_modes(device):
    torch.compiler.reset()

    def call(x, y):
        return torch.stack([tp.kl_divergences(x, y).sum(),
                            tp.symmetric_kl_divergences(x, y, base=2).sum(),
                            tp.paired_distances(x, y, metric='kl').sum(),
                            tp.paired_distances(x, y, metric='symmetric_kl').sum(),
                            tp.condensed_distances(x, metric='symmetric_kl').sum()])

    compiled = torch.compile(call, fullgraph=True)
    for zeros in (False, True):
        x, y = [(torch.rand(3, 4, device=device) + .2) for _ in range(2)]
        if zeros:
            x[0, 0] = 0.
            y[1, 1] = 0.
        x, y = x.requires_grad_(), y.requires_grad_()
        expected, actual = call(x, y), compiled(x, y)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        for grad, ref in zip(torch.autograd.grad(actual.sum(), (x, y)),
                             torch.autograd.grad(expected.sum(), (x, y))):
            torch.testing.assert_close(grad, ref, atol=2e-5, rtol=1e-4)
    torch.compiler.reset()
