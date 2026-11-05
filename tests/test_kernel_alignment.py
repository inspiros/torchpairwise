"""Coordinated Gram statistics: dense formulas, VJPs and boundary contracts."""
import pytest
import torch
import torchpairwise as tp


def test_frobenius_centering_normalization_and_source_gradients(device):
    torch.manual_seed(235)
    x = torch.randn(2, 1, 5, 3, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, 2, 7, 3, device=device, dtype=x.dtype, requires_grad=True)
    # Noncontiguous source operands exercise the backward Fake stride contract.
    raw = torch.randn(7, 5, device=device, dtype=torch.float32, requires_grad=True)
    factor = torch.tensor(.4, device=device, dtype=x.dtype, requires_grad=True)
    a = (-factor * tp.lazy(x, y, metric='sqeuclidean', chunk_size=3)).exp()
    b = tp.as_lazy(raw.T, chunk_size=2)
    dense_a = (-factor * (x.unsqueeze(-2) - y.unsqueeze(-3)).square().sum(-1)).exp()
    actual, expected = [], []
    for center in (False, True):
        aa, bb = dense_a, raw.T.double()
        if center:
            aa = aa - aa.mean(-1, keepdim=True) - aa.mean(-2, keepdim=True) + aa.mean((-2, -1), keepdim=True)
            bb = bb - bb.mean(-1, keepdim=True) - bb.mean(-2, keepdim=True) + bb.mean((-2, -1), keepdim=True)
        dot = (aa * bb).sum((-2, -1))
        for normalize in (False, True):
            actual.append(a.frobenius_inner_product(b, center=center, normalize=normalize))
            expected.append(dot / (aa.square().sum((-2, -1)).sqrt() * bb.square().sum((-2, -1)).sqrt()) if normalize else dot)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    cot = [torch.randn_like(v) for v in actual]
    torch.testing.assert_close(torch.autograd.grad(actual, (x, y, raw, factor), cot),
                               torch.autograd.grad(expected, (x, y, raw, factor), cot),
                               rtol=2e-5, atol=1e-7)
    # Shared roots accumulate both cotangents, including transpose orientation.
    shared = tp.as_lazy(raw.T, chunk_size=2)
    norm = shared.frobenius_inner_product(shared, center=True, normalize=True)
    torch.testing.assert_close(norm, torch.ones_like(norm))
    torch.testing.assert_close(torch.autograd.grad(norm, raw)[0], torch.zeros_like(raw), atol=1e-6, rtol=0)
    with torch.no_grad():
        raw.add_(.2)
    torch.testing.assert_close(shared.T.frobenius_inner_product(shared.T), raw.double().square().sum().float())

    # Independent conventional HSIC formula on symmetric Gram matrices.
    features = torch.randn(5, 3, device=device, dtype=x.dtype, requires_grad=True)
    target = torch.randn(5, 2, device=device, dtype=x.dtype, requires_grad=True)
    k, l = tp.lazy(features, metric='rbf', gamma=0.3, chunk_size=2), tp.lazy(target, metric='linear', chunk_size=3)
    kk, ll = tp.rbf_kernel(features, gamma=.3), target @ target.T
    kc = kk - kk.mean(-1, keepdim=True) - kk.mean(-2, keepdim=True) + kk.mean()
    lc = ll - ll.mean(-1, keepdim=True) - ll.mean(-2, keepdim=True) + ll.mean()
    ko = kk - kk.diag().diag(); lo = ll - ll.diag().diag()
    expected = ((kc * lc).sum() / 16,
                ((ko * lo).sum() + ko.sum() * lo.sum() / 12 -
                 2 * (ko.sum(-1) * lo.sum(-1)).sum() / 3) / 10)
    retained = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: (retained.append(tuple(t.shape)), t)[1], lambda t: t):
        actual = (k.hsic(l), k.hsic(l, unbiased=True))
    assert (5, 5) not in retained
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(torch.autograd.grad(actual, (features, target)),
                               torch.autograd.grad(expected, (features, target)), rtol=1e-8, atol=1e-9)
    # U-centered product remains defined for non-symmetric score matrices.
    scores = torch.randn(5, 5, device=device, dtype=x.dtype, requires_grad=True)
    def ucenter(t):
        off = t.masked_fill(torch.eye(5, device=device, dtype=torch.bool), 0)
        centered = off - off.sum(-1, keepdim=True)/3 - off.sum(-2, keepdim=True)/3 + off.sum()/12
        return centered.masked_fill(torch.eye(5, device=device, dtype=torch.bool), 0)
    a = tp.as_lazy(scores, chunk_size=2)
    actual = a.hsic(a.T, unbiased=True)
    expected = (ucenter(scores) * ucenter(scores.T)).sum()/10
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, scores), torch.autograd.grad(expected, scores))


def test_frobenius_zero_empty_translation_and_validation(device):
    for shape in ((3, 4), (0, 4), (3, 0), (1, 4)):
        data = torch.ones(shape, device=device, dtype=torch.float64, requires_grad=True)
        a = tp.as_lazy(data, chunk_size=2)
        result = a.frobenius_inner_product(a, center=True, normalize=True)
        torch.testing.assert_close(result, torch.zeros_like(result))
        torch.testing.assert_close(torch.autograd.grad(result, data)[0], torch.zeros_like(data))
    z = torch.zeros(3, 4, device=device, requires_grad=True)
    a = tp.as_lazy(z)
    zero = a.frobenius_inner_product(a, normalize=True)
    assert zero.item() == 0 and torch.autograd.grad(zero, z)[0].eq(0).all()
    for scale in (1e-200, 1e200):
        values = (torch.arange(1, 13, device=device, dtype=torch.float64).reshape(3, 4) * scale).requires_grad_()
        a = tp.as_lazy(values, chunk_size=2)
        result = a.frobenius_inner_product(a, normalize=True)
        torch.testing.assert_close(result, torch.ones_like(result))
        assert torch.autograd.grad(result, values)[0].isfinite().all()
    offsets = torch.tensor([[0., 2., 8.], [6., 4., 10.]], device=device, dtype=torch.float64)
    a, b = tp.as_lazy(offsets + 1e+16, chunk_size=2), tp.as_lazy(offsets, chunk_size=1)
    torch.testing.assert_close(a.frobenius_inner_product(b, center=True), b.frobenius_inner_product(b, center=True))
    with pytest.raises(TypeError, match='scalar operator'):
        a.frobenius_inner_product(offsets)
    with pytest.raises(TypeError, match='booleans'):
        a.frobenius_inner_product(b, center=1)
    with pytest.raises(RuntimeError, match='N/M'):
        a.frobenius_inner_product(tp.as_lazy(offsets.T))
    with pytest.raises(RuntimeError, match='floating'):
        tp.as_lazy(offsets.long()).frobenius_inner_product(tp.as_lazy(offsets.long()))

    with pytest.raises(RuntimeError, match='square'):
        a.hsic(b)
    for count, unbiased in ((1, False), (3, True)):
        small = tp.as_lazy(torch.ones(count, count, device=device))
        with pytest.raises(RuntimeError, match='observations'):
            small.hsic(small, unbiased=unbiased)
    bad = torch.eye(4, device=device, dtype=torch.float64).masked_fill(torch.eye(4, device=device, dtype=torch.bool), float('nan')).requires_grad_()
    source = tp.as_lazy(bad, chunk_size=1)
    result = source.hsic(source, unbiased=True)
    assert result.item() == 0
    assert torch.autograd.grad(result, bad)[0].eq(0).all()
