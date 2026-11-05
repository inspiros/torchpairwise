"""Probability references, support conventions and public execution contracts."""
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest
import torch
import torchpairwise as tp


OPERATIONS = {
    'total_variation': 'total_variation_distances',
    'fisher_rao': 'fisher_rao_distances',
    'renyi': 'renyi_divergences',
    'pearson_chi2': 'pearson_chi2_divergences',
    'neyman_chi2': 'neyman_chi2_divergences',
    'triangular_discrimination': 'triangular_discrimination_distances',
    'itakura_saito': 'itakura_saito_divergences',
}
SYMMETRIC = {'total_variation', 'fisher_rao', 'triangular_discrimination'}


def reference(x, y, kind, alpha=1.7):
    if kind != 'itakura_saito':
        x, y = x / x.sum(-1, keepdims=True), y / y.sum(-1, keepdims=True)
    p, q = x[..., :, None, :], y[..., None, :, :]
    if kind == 'total_variation':
        return .5 * np.abs(p - q).sum(-1)
    if kind == 'fisher_rao':
        return 2 * np.arccos(np.clip(np.sqrt(p * q).sum(-1), 0., 1.))
    if kind == 'renyi':
        return np.log((p ** alpha * q ** (1. - alpha)).sum(-1)) / (alpha - 1.)
    if kind == 'pearson_chi2':
        return ((p - q) ** 2 / q).sum(-1)
    if kind == 'neyman_chi2':
        return ((p - q) ** 2 / p).sum(-1)
    if kind == 'triangular_discrimination':
        return ((p - q) ** 2 / (p + q)).sum(-1)
    return (p / q - np.log(p / q) - 1.).sum(-1)


@pytest.mark.parametrize('kind', OPERATIONS)
def test_probability_extra_reference_public_modes_and_gradcheck(kind, device):
    torch.manual_seed(201)
    fn = getattr(tp, OPERATIONS[kind])
    options = {'alpha': 1.7} if kind == 'renyi' else {}
    x = (torch.rand(2, 1, 3, 6, device=device, dtype=torch.float64) + .4)[..., ::2].requires_grad_()
    y = (torch.rand(1, 2, 4, 6, device=device, dtype=torch.float64) + .3)[..., ::2].requires_grad_()
    value = fn(x, y, **options)
    expected = reference(x.detach().cpu().numpy(), y.detach().cpu().numpy(), kind)
    torch.testing.assert_close(value, torch.as_tensor(expected, device=device), atol=2e-13, rtol=2e-11)
    torch.testing.assert_close(tp.cdist(x, y, metric=kind, **options), value)
    paired = tp.paired_distances(x, y[..., :3, :], metric=kind, **options)
    diagonal = fn(x, y[..., :3, :], **options).diagonal(dim1=-2, dim2=-1)
    torch.testing.assert_close(paired, diagonal)
    for actual, expected in zip(torch.autograd.grad(paired.sum(), (x, y)),
                                torch.autograd.grad(diagonal.sum(), (x, y))):
        torch.testing.assert_close(actual, expected)
    if kind in SYMMETRIC:
        i, j = torch.triu_indices(3, 3, 1, device=device)
        condensed = tp.condensed_distances(x, metric=kind)
        triangle = fn(x)[..., i, j]
        torch.testing.assert_close(condensed, triangle)
        torch.testing.assert_close(torch.autograd.grad(condensed.sum(), x)[0],
                                   torch.autograd.grad(triangle.sum(), x)[0])
    else:
        with pytest.raises(ValueError, match='symmetric'):
            tp.condensed_distances(x, metric=kind, **options)
    a, b = [(torch.rand(2, 3, dtype=torch.float64, device=device) + .5).requires_grad_()
            for _ in range(2)]
    assert torch.autograd.gradcheck(lambda u, v: fn(u, v, **options), (a, b), fast_mode=True)
    if kind != 'itakura_saito':
        torch.testing.assert_close(fn(7 * x, 3 * y, **options), value)
    else:
        torch.testing.assert_close(fn(7 * x, 7 * y), value)
        assert not torch.allclose(fn(7 * x, y), value)


def test_probability_extra_support_and_boundary_gradients(device):
    x = torch.tensor([[1., 0., 0.]], dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tensor([[0., 1., 0.]], dtype=torch.float64, device=device, requires_grad=True)
    expected = {'total_variation': 1., 'fisher_rao': math.pi, 'renyi': math.inf,
                'pearson_chi2': math.inf, 'neyman_chi2': math.inf,
                'triangular_discrimination': 2.}
    for kind, target in expected.items():
        fn = getattr(tp, OPERATIONS[kind])
        options = {'alpha': .5} if kind == 'renyi' else {}
        value = fn(x, y, **options)
        torch.testing.assert_close(value, torch.tensor([[target]], dtype=x.dtype, device=device))
        for grad in torch.autograd.grad(value.sum(), (x, y)):
            torch.testing.assert_close(grad, torch.zeros_like(grad))
        for grad in torch.autograd.grad(fn(x, y, **options), (x, y), grad_outputs=torch.zeros_like(value)):
            assert torch.isfinite(grad).all()
        same = fn(x, x, **options)
        assert same.eq(0).all()
        torch.testing.assert_close(torch.autograd.grad(same.sum(), x)[0], torch.zeros_like(x))
    p = torch.tensor([[1., 0.]], dtype=x.dtype, device=device, requires_grad=True)
    q = torch.tensor([[1., 3.]], dtype=x.dtype, device=device, requires_grad=True)
    # The q-only bin contributes zero to the Renyi power sum for every alpha>0.
    for alpha in (.5, 1., 2.):
        value = tp.renyi_divergences(p, q, alpha=alpha)
        torch.testing.assert_close(value, torch.full_like(value, math.log(4.)))
        gx, gy = torch.autograd.grad(value.sum(), (p, q))
        assert gx[0, 1] == 0
        assert torch.isfinite(gx).all() and torch.isfinite(gy).all()
    a = torch.tensor([[1., 0., 0.], [1., 2., 0.]], dtype=x.dtype, device=device, requires_grad=True)
    b = torch.tensor([[0., 1., 0.], [2., 2., 0.]], dtype=x.dtype, device=device, requires_grad=True)
    for kind in expected:
        options = {'alpha': .5} if kind == 'renyi' else {}
        value = getattr(tp, OPERATIONS[kind])(a, b, **options)
        for gradient, raw in zip(torch.autograd.grad(value.sum(), (a, b)), (a, b)):
            assert torch.isfinite(gradient).all()
            assert gradient[raw == 0].eq(0).all()
    # Neyman(p,q) is Pearson(q,p), including which support mismatch is infinite.
    torch.testing.assert_close(tp.neyman_chi2_divergences(p, q), tp.pearson_chi2_divergences(q, p).T)


def test_probability_extra_near_identity_decimal_and_renyi_limit(device):
    for dtype, delta, tolerance in ((torch.float32, 1e-3, 5e-5), (torch.float64, 1e-7, 5e-7)):
        x = torch.tensor([[2., 3., 5.]], dtype=dtype, device=device, requires_grad=True)
        y = torch.tensor([[2. + delta, 3. - delta, 5.]], dtype=dtype, device=device, requires_grad=True)
        with localcontext() as context:
            context.prec = 70
            a = [Decimal(float(v)) for v in x.detach().cpu()[0]]
            b = [Decimal(float(v)) for v in y.detach().cpu()[0]]
            p, q = [v / sum(a) for v in a], [v / sum(b) for v in b]
            chord = sum((u.sqrt() - v.sqrt()) ** 2 for u, v in zip(p, q)).sqrt()
            # asin(z)=z+O(z^3), far below this check's relative tolerance.
            fisher = 2 * chord
            pearson = sum((u - v) ** 2 / v for u, v in zip(p, q))
            triangular = sum((u - v) ** 2 / (u + v) for u, v in zip(p, q))
            is_value = sum(u / v - (u / v).ln() - 1 for u, v in zip(a, b))
            targets = {'fisher_rao': fisher, 'pearson_chi2': pearson,
                       'triangular_discrimination': triangular, 'itakura_saito': is_value}
            for alpha in (.5, 1. + 1e-10, 2., 1.e6):
                exponent = Decimal(alpha)
                power_sum = sum((exponent * u.ln() + (1 - exponent) * v.ln()).exp()
                                for u, v in zip(p, q))
                target = power_sum.ln() / (exponent - 1)
                value = tp.renyi_divergences(x, y, alpha=alpha)
                torch.testing.assert_close(value, torch.full_like(value, float(target)), atol=0, rtol=tolerance)
                assert torch.isfinite(torch.autograd.grad(value.sum(), (x, y))[0]).all()
        for kind, target in targets.items():
            value = getattr(tp, OPERATIONS[kind])(x, y)
            assert value.item() > 0
            torch.testing.assert_close(value, torch.full_like(value, float(target)), atol=0, rtol=tolerance)
        torch.testing.assert_close(tp.renyi_divergences(x, y, alpha=1), tp.kl_divergences(x, y))
    # Preserve float32 raw support where normalization in float32 underflows.
    p = torch.tensor([[torch.finfo(torch.float32).tiny, 1.e20]], device=device)
    q = torch.tensor([[1., 1.]], device=device)
    assert torch.isfinite(tp.renyi_divergences(q, p, alpha=2)).all()
    assert torch.isfinite(tp.pearson_chi2_divergences(p, q)).all()
    assert torch.isinf(tp.pearson_chi2_divergences(q, p)).all()  # float32 output overflow


def test_probability_extra_tile_boundaries_and_empty_gradients(device):
    torch.manual_seed(204)
    x, y = [(torch.rand(n, 65, device=device, dtype=torch.float64) + .3).requires_grad_()
            for n in (33, 2)]
    for kind, alpha in (('renyi', .7), ('renyi', 2.3), ('pearson_chi2', None), ('itakura_saito', None)):
        fn = getattr(tp, OPERATIONS[kind])
        options = {'alpha': alpha} if kind == 'renyi' else {}
        value = fn(x, y, **options)
        target = reference(x.detach().cpu().numpy(), y.detach().cpu().numpy(), kind, alpha=alpha)
        torch.testing.assert_close(value, torch.as_tensor(target, device=device), atol=2e-12, rtol=2e-11)
        # Small independent dense formulas check gradients across both tile axes.
        p, q = x[:, None], y[None, :]
        if kind != 'itakura_saito':
            p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
        if kind == 'renyi':
            direct = (p.pow(alpha) * q.pow(1 - alpha)).sum(-1).log() / (alpha - 1)
        elif kind == 'pearson_chi2':
            direct = ((p - q).square() / q).sum(-1)
        else:
            direct = (p / q - (p / q).log() - 1).sum(-1)
        for actual, expected in zip(torch.autograd.grad(value.sum(), (x, y)),
                                    torch.autograd.grad(direct.sum(), (x, y))):
            torch.testing.assert_close(actual, expected)
    for kind, name in OPERATIONS.items():
        fn = getattr(tp, name)
        for left, right, shape in ((x[:0], y, (0, 2)), (x, y[:0], (33, 0))):
            empty = fn(left, right)
            assert empty.shape == shape
            for grad in torch.autograd.grad(empty.sum(), (x, y)):
                torch.testing.assert_close(grad, torch.zeros_like(grad))
        empty = tp.paired_distances(x[:0], y[:0], metric=kind)
        assert empty.shape == (0,)
        for grad in torch.autograd.grad(empty.sum(), (x, y)):
            torch.testing.assert_close(grad, torch.zeros_like(grad))


def test_probability_extra_validation_and_amp(device):
    x, y = [(torch.rand(3, 4, device=device) + .3).requires_grad_() for _ in range(2)]
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    for kind, name in OPERATIONS.items():
        fn = getattr(tp, name)
        with pytest.raises(TypeError, match='dense strided'):
            fn(x.to_sparse())
        with pytest.raises(TypeError, match='float32 or float64'):
            fn(x.to(dtype))
        for bad in (torch.zeros_like(x), -torch.ones_like(x), torch.full_like(x, float('inf'))):
            with pytest.raises(ValueError):
                fn(bad)
        # Master leaves stay float32; low precision casts are part of the graph.
        a, b = x.to(dtype), y.to(dtype)
        for call, args, kwargs in ((fn, (a, b), {}),
                                   (tp.paired_distances, (a, b), {'metric': kind})):
            with torch.autocast(device, dtype=dtype):
                value = call(*args, **kwargs)
            assert value.dtype == torch.float32
            torch.testing.assert_close(value, call(*(v.float() for v in args), **kwargs))
            assert torch.isfinite(torch.autograd.grad(value.sum(), x, retain_graph=True)[0]).all()
    for alpha in (0, -1, True, float('inf'), float('nan')):
        with pytest.raises(ValueError, match='alpha'):
            tp.renyi_divergences(x, alpha=alpha)
    huge = torch.full_like(x, torch.finfo(x.dtype).max)
    assert tp.itakura_saito_divergences(huge).eq(0).all()
    with pytest.raises(ValueError, match='positive finite row mass'):
        tp.total_variation_distances(huge)


@pytest.mark.compiler
def test_probability_extra_compiled_values_gradients_and_zero_upstream(device):
    torch.compiler.reset()

    def call(x, y):
        return torch.stack([tp.total_variation_distances(x, y).sum(),
                            tp.fisher_rao_distances(x, y).sum(),
                            tp.renyi_divergences(x, y, alpha=.7).sum(),
                            tp.pearson_chi2_divergences(x, y).sum(),
                            tp.neyman_chi2_divergences(x, y).sum(),
                            tp.triangular_discrimination_distances(x, y).sum(),
                            tp.itakura_saito_divergences(x + .1, y + .1).sum(),
                            tp.paired_distances(x, y, metric='renyi', alpha=2.).sum(),
                            tp.condensed_distances(x, metric='fisher_rao').sum()])

    compiled = torch.compile(call, fullgraph=True)
    for zeros in (False, True):
        x, y = [(torch.rand(3, 4, device=device) + .4) for _ in range(2)]
        if zeros:
            x[0, 0] = 0.
            y[1, 1] = 0.
        x, y = x.requires_grad_(), y.requires_grad_()
        actual, expected = compiled(x, y), call(x, y)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        for grad, target in zip(torch.autograd.grad(actual.sum(), (x, y)),
                                torch.autograd.grad(expected.sum(), (x, y))):
            torch.testing.assert_close(grad, target, atol=2e-5, rtol=2e-4)
        for grad in torch.autograd.grad(compiled(x, y), (x, y), grad_outputs=torch.zeros_like(actual)):
            torch.testing.assert_close(grad, torch.zeros_like(grad))
    torch.compiler.reset()
