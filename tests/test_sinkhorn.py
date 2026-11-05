"""Focused values, finite-budget gradients and assembly checks for Sinkhorn."""
import pytest
import torch

import torchpairwise as tp


def _primal_reference(x, y, p, regularization):
    """Independent positive-domain alternating scaling, only for tiny tests."""
    cost = torch.linalg.vector_norm(x[:, None] - y[None], dim=-1).pow(p)
    a = x.new_full((len(x),), 1 / len(x))
    b = y.new_full((len(y),), 1 / len(y))
    kernel = (-cost / regularization).exp()
    v = torch.ones_like(b)
    for _ in range(1000):
        u = a / (kernel @ v)
        v = b / (kernel.T @ u)
    plan = u[:, None] * kernel * v[None]
    torch.testing.assert_close(plan.sum(-1), a, atol=1e-12, rtol=1e-12)
    return (plan * cost).sum() + regularization * (
        plan * (plan.log() - a.log()[:, None] - b.log()[None])).sum()


@pytest.mark.parametrize('p', [1, 2])
def test_sinkhorn_regularized_objective_reference(p, device):
    x = torch.tensor([[-.2, .1], [.4, .6], [.3, -.1]], dtype=torch.float64, device=device)
    y = torch.tensor([[.1, -.3], [.6, .2]], dtype=x.dtype, device=device)
    reg = .7
    expected = (_primal_reference(x, y, p, reg)
                - .5 * _primal_reference(x, x, p, reg)
                - .5 * _primal_reference(y, y, p, reg))
    actual = tp.sinkhorn_divergences(x[None], y[None], p=p, regularization=reg, max_iter=250)
    torch.testing.assert_close(actual.squeeze(), expected, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize('p', [1, 2])
def test_sinkhorn_singletons_finite_budget_symmetry_and_self(p, device):
    x = torch.tensor([[[0., 0.]], [[1., 2.]]], dtype=torch.float64, device=device,
                     requires_grad=True)
    y = torch.tensor([[[3., 4.]], [[-1., 1.]], [[.5, .25]]], dtype=x.dtype, device=device)
    actual = tp.sinkhorn_divergences(x, y, p=p, max_iter=1)
    expected = torch.cdist(x[:, 0], y[:, 0]).pow(p)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                               torch.autograd.grad(expected.sum(), x)[0])
    # Symmetry is retained even when only a few scaling updates are executed.
    torch.manual_seed(20)
    a = torch.rand(2, 3, 2, dtype=x.dtype, device=device, requires_grad=True)
    b = torch.rand(3, 4, 2, dtype=x.dtype, device=device)
    forward = tp.sinkhorn_divergences(a, b, p=p, max_iter=3)
    torch.testing.assert_close(forward, tp.sinkhorn_divergences(b, a, p=p, max_iter=3).T)
    own = tp.paired_distances(a, a, metric='sinkhorn', p=p, max_iter=3)
    torch.testing.assert_close(own, torch.zeros_like(own), atol=1e-14, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(own.sum(), a)[0], torch.zeros_like(a),
                               atol=1e-14, rtol=0)


def test_sinkhorn_ragged_layout_modes_chunking_and_gradients(device):
    torch.manual_seed(58)
    x = torch.rand(2, 3, 4, 2, dtype=torch.float64, device=device)
    y = torch.rand(1, 3, 3, 2, dtype=x.dtype, device=device)
    xl = torch.tensor([[4, 2, 1], [3, 4, 2]], device=device)
    yl = torch.tensor([[2, 3, 1]], device=device)
    x[torch.arange(4, device=device) >= xl[..., None]] = float('nan')
    y[torch.arange(3, device=device) >= yl[..., None]] = float('nan')
    x = x.reshape(2, 3, 2, 2, 2).movedim(-1, 2).contiguous().requires_grad_()
    y = y.movedim(-1, 2).contiguous().requires_grad_()
    options = dict(batched=True, feature_dim=2, x1_lengths=xl, x2_lengths=yl,
                   regularization=.4, max_iter=8)
    dense = tp.cdist(x, y, metric='sinkhorn', **options)
    paired = tp.paired_distances(x, y, metric='sinkhorn', **options)
    torch.testing.assert_close(paired, dense.diagonal(dim1=-2, dim2=-1))
    for fn, expected in ((tp.cdist, dense), (tp.paired_distances, paired)):
        actual = fn(x, y, metric='sinkhorn', chunk_size=2, **options)
        torch.testing.assert_close(actual, expected)
        for grad, ref in zip(torch.autograd.grad(actual.sum(), (x, y), retain_graph=True),
                             torch.autograd.grad(expected.sum(), (x, y), retain_graph=True)):
            torch.testing.assert_close(grad, ref)
            assert torch.isfinite(grad).all()
    condensed_options = {key: value for key, value in options.items() if key != 'x2_lengths'}
    condensed = tp.pdist(x, metric='sinkhorn', chunk_size=2, **condensed_options)
    own = tp.cdist(x, None, metric='sinkhorn', **condensed_options)
    i, j = torch.triu_indices(3, 3, 1, device=device)
    torch.testing.assert_close(condensed, own[..., i, j])
    torch.testing.assert_close(torch.autograd.grad(condensed.sum(), x, retain_graph=True)[0],
                               torch.autograd.grad(own[..., i, j].sum(), x)[0])


def test_sinkhorn_finite_algorithm_gradcheck(device):
    x = torch.tensor([[[.1, -.2], [.7, .4]]], dtype=torch.float64, device=device,
                     requires_grad=True)
    y = torch.tensor([[[-.3, .5], [.2, .9], [.8, -.1]]], dtype=x.dtype, device=device,
                     requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: tp.sinkhorn_divergences(a, b, regularization=.6, max_iter=4), (x, y))


def test_sinkhorn_small_cost_and_translated_padding(device):
    small = torch.tensor([[[1e-4, 0.]]], device=device, requires_grad=True)
    value = tp.sinkhorn_divergences(torch.zeros_like(small), small,
                                    regularization=1., max_iter=1)
    torch.testing.assert_close(value, small.new_tensor([[1e-8]]), atol=1e-14, rtol=2e-6)
    torch.testing.assert_close(torch.autograd.grad(value.sum(), small)[0], 2 * small)
    padded = torch.tensor([[[1e20], [float('nan')]]], device=device, requires_grad=True)
    target = torch.tensor([[[1e20]]], device=device)
    result = tp.sinkhorn_divergences(padded, target, x1_lengths=torch.tensor([1], device=device),
                                     max_iter=2)
    torch.testing.assert_close(result, torch.zeros_like(result))
    torch.testing.assert_close(torch.autograd.grad(result.sum(), padded)[0], torch.zeros_like(padded))


def test_sinkhorn_empty_samples_low_precision_and_validation(device):
    x = torch.ones(2, 3, 2, device=device, requires_grad=True)
    y = torch.zeros(2, 4, 2, device=device)
    for fn, args, shape in ((tp.cdist, (x[:0], y), (0, 2)),
                            (tp.paired_distances, (x[:0], y[:0]), (0,)),
                            (tp.pdist, (x[:1],), (0,))):
        actual = fn(*args, metric='sinkhorn', max_iter=2)
        assert actual.shape == shape
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0], torch.zeros_like(x))
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    low_x, low_y = x.to(dtype), y.to(dtype)
    with torch.autocast(device, dtype=dtype):
        actual = tp.sinkhorn_divergences(low_x, low_y, max_iter=3)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, tp.sinkhorn_divergences(low_x.float(), low_y.float(), max_iter=3))
    for option, error in ((dict(p=3), ValueError), (dict(p=True), ValueError),
                           (dict(regularization=0), ValueError),
                           (dict(regularization=float('nan')), ValueError),
                           (dict(max_iter=0), ValueError), (dict(max_iter=True), TypeError),
                           (dict(regularization=1e-100), ValueError)):
        with pytest.raises(error):
            tp.sinkhorn_divergences(x, y, **option)
    with pytest.raises(ValueError, match='finite'):
        tp.sinkhorn_divergences(x * float('inf'), y)
