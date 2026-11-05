"""Unregularized multidimensional uniform transport and chosen-plan gradients."""
import math

import pytest
import torch
import torchpairwise as tp


@pytest.mark.parametrize('p', [1, 2])
def test_exact_wasserstein_joint_geometry_and_unequal_counts(p, device):
    x = torch.tensor([[[-1., -1.], [1., 1.]]], dtype=torch.float64, device=device)
    y = torch.tensor([[[-1., 1.], [1., -1.]]], dtype=x.dtype, device=device)
    # Coordinate marginals agree, but the joint transport distance is two.
    torch.testing.assert_close(tp.wasserstein_distances(x, y, p=p), x.new_tensor([[2.]]))
    a = x.new_tensor([[[0.], [3.]]])
    b = x.new_tensor([[[0.], [1.], [3.]]])
    expected = .5 if p == 1 else math.sqrt(5 / 6)
    torch.testing.assert_close(tp.cdist(a, b, metric='wasserstein', p=p), a.new_tensor([[expected]]))
    # A cheapest-edge greedy coupling would take 0 -> 1 first and force the
    # expensive 2 -> -2 edge. Residual reverse arcs must undo that first choice.
    a, b = x.new_tensor([[[0.], [2.]]]), x.new_tensor([[[1.], [-2.]]])
    expected = 1.5 if p == 1 else math.sqrt(2.5)
    torch.testing.assert_close(tp.wasserstein_distances(a, b, p=p), a.new_tensor([[expected]]))
    # A singleton comparison distinguishes rooted Euclidean W2 from squared
    # cost and distinguishes multidimensional W1 from Manhattan ground cost.
    torch.testing.assert_close(tp.wasserstein_distances(x.new_tensor([[[0., 0.]]]),
                                                      x.new_tensor([[[3., 4.]]]), p=p),
                               x.new_tensor([[5.]]))


def _linear_program_value(x, y, p):
    scipy_optimize = pytest.importorskip('scipy.optimize')
    np = pytest.importorskip('numpy')
    x, y = x.detach().double().cpu().numpy(), y.detach().double().cpu().numpy()
    cost = np.linalg.norm(x[:, None] - y[None], axis=-1) ** p
    n, m = cost.shape
    rows = np.kron(np.eye(n), np.ones((1, m)))
    columns = np.tile(np.eye(m), (1, n))
    result = scipy_optimize.linprog(cost.ravel(), A_eq=np.concatenate((rows, columns)),
                                    b_eq=np.concatenate((np.full(n, 1 / n), np.full(m, 1 / m))),
                                    bounds=(0, None), method='highs')
    assert result.success, result.message
    return max(0., result.fun) ** (1 / p)


@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('n,m', [(2, 3), (3, 5), (4, 4), (7, 5)])
def test_exact_wasserstein_against_independent_linear_program(p, n, m, device):
    generator = torch.Generator().manual_seed(170 + n + m)
    x = torch.randn(2, n, 3, generator=generator, dtype=torch.float64).to(device)
    y = torch.randn(3, m, 3, generator=generator, dtype=torch.float64).to(device)
    expected = x.new_tensor([[_linear_program_value(a, b, p) for b in y] for a in x])
    actual = tp.wasserstein_distances(x, y, p=p)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(tp.wasserstein_distances(y, x, p=p).T, expected)
    torch.testing.assert_close(tp.wasserstein_distances(x.flip(-2), y.flip(-2), p=p), expected)


@pytest.mark.parametrize('p', [1, 2])
def test_exact_wasserstein_broadcast_layout_ragged_modes_and_chunking(p, device):
    generator = torch.Generator().manual_seed(812)
    x = torch.randn(2, 3, 4, 2, generator=generator, dtype=torch.float64).to(device)
    y = torch.randn(1, 3, 3, 2, generator=generator, dtype=torch.float64).to(device)
    xl = torch.tensor([[4, 2, 1], [3, 4, 2]], device=device)
    yl = torch.tensor([[3, 1, 2]], device=device)
    x[torch.arange(4, device=device) >= xl[..., None]] = float('nan')
    y[torch.arange(3, device=device) >= yl[..., None]] = float('nan')
    x = x.reshape(2, 3, 2, 2, 2).movedim(-1, 2).contiguous().requires_grad_()
    y = y.reshape(1, 3, 1, 3, 2).movedim(-1, 2).contiguous().requires_grad_()
    kwargs = dict(p=p, batched=True, feature_dim=2, x1_lengths=xl, x2_lengths=yl)
    dense = tp.wasserstein_distances(x, y, **kwargs)
    chunked = tp.cdist(x, y, metric='wasserstein', chunk_size=2, **kwargs)
    torch.testing.assert_close(chunked, dense)
    paired = tp.paired_distances(x, y, metric='wasserstein', **kwargs)
    torch.testing.assert_close(paired, dense.diagonal(dim1=-2, dim2=-1))
    paired_chunked = tp.paired_distances(x, y, metric='wasserstein', chunk_size=2, **kwargs)
    torch.testing.assert_close(paired_chunked, paired)
    for actual, expected in ((chunked, dense), (paired, dense.diagonal(dim1=-2, dim2=-1)),
                             (paired_chunked, paired)):
        for grad, ref in zip(torch.autograd.grad(actual.sum(), (x, y), retain_graph=True),
                             torch.autograd.grad(expected.sum(), (x, y), retain_graph=True)):
            torch.testing.assert_close(grad, ref)
            assert torch.isfinite(grad).all()
    kwargs.pop('x2_lengths')
    own = tp.wasserstein_distances(x, **kwargs)
    i, j = torch.triu_indices(3, 3, 1, device=device)
    condensed = tp.pdist(x, metric='wasserstein', chunk_size=2, **kwargs)
    torch.testing.assert_close(condensed, own[..., i, j])
    torch.testing.assert_close(torch.autograd.grad(condensed.sum(), x)[0],
                               torch.autograd.grad(own[..., i, j].sum(), x)[0])


@pytest.mark.parametrize('p', [1, 2])
def test_exact_wasserstein_coordinate_gradcheck(p, device):
    x = torch.tensor([[[.1, .2], [1.7, -.4]]], device=device,
                     dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[[-.2, .6], [.9, 1.3], [2.2, -.3]]], device=device,
                     dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: tp.wasserstein_distances(a, b, p=p), (x, y))


@pytest.mark.parametrize('p', [1, 2])
def test_exact_wasserstein_zero_ties_and_empty_gradients(p, device):
    x = torch.tensor([[[1., 2.], [1., 2.]], [[3., 4.], [3., 4.]]],
                     device=device, dtype=torch.float64, requires_grad=True)
    value = tp.paired_distances(x, x, metric='wasserstein', p=p)
    torch.testing.assert_close(value, torch.zeros_like(value))
    torch.testing.assert_close(torch.autograd.grad(value.sum(), x)[0], torch.zeros_like(x))
    for compare, args in ((tp.cdist, (x[:0], x)),
                           (tp.paired_distances, (x[:0], x[:0])),
                           (tp.pdist, (x[:1],)),
                           (tp.cdist, (x.expand(0, *x.shape), x.unsqueeze(0)))):
        result = compare(*args, metric='wasserstein', p=p)
        assert result.numel() == 0
        torch.testing.assert_close(torch.autograd.grad(result.sum(), x)[0], torch.zeros_like(x))


def test_exact_wasserstein_direct_paired_does_not_build_cross_pairs(device):
    x = torch.arange(12., device=device).reshape(3, 2, 2)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
        result = tp.paired_distances(x, x + .25, metric='wasserstein')
    assert result.shape == (3,)
    calls = [event for event in profile.events() if event.name == 'aten::cdist']
    assert len(calls) == 3
    torch.testing.assert_close(result, torch.full_like(result, 2 ** .5 * .25))


def test_exact_wasserstein_validation_and_eager_contract(device):
    x = torch.ones(2, 3, 2, device=device)
    for p in (0, 3, -1, float('nan'), float('inf'), True, '1'):
        with pytest.raises((TypeError, ValueError), match='p|non-negative'):
            tp.wasserstein_distances(x, p=p)
    for bad in (float('nan'), float('inf')):
        invalid = x.clone()
        invalid[0, 0, 0] = bad
        with pytest.raises(ValueError, match='finite'):
            tp.wasserstein_distances(invalid)
    for lengths in (torch.tensor([0, 2], device=device), torch.tensor([4, 2], device=device)):
        with pytest.raises(ValueError, match='between 1'):
            tp.wasserstein_distances(x, x1_lengths=lengths)
    from torchpairwise.ops._wasserstein import wasserstein
    assert wasserstein(torch.empty(2, 3, 2, device='meta'),
                       torch.empty(2, 3, 2, device='meta')).shape == (2, 2)


def test_exact_wasserstein_autocast(device):
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    x = torch.arange(12., device=device).reshape(3, 2, 2).to(dtype).requires_grad_()
    y = (x.detach() + .5).requires_grad_()
    for compare, args in ((tp.cdist, (x, y)), (tp.paired_distances, (x, y)), (tp.pdist, (x,))):
        with torch.autocast(device, dtype=dtype):
            actual = compare(*args, metric='wasserstein', p=2)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, compare(*(a.float() for a in args), metric='wasserstein', p=2))
        assert torch.isfinite(torch.autograd.grad(actual.sum(), x)[0]).all()
