"""Projected empirical transport, including exact unequal mass intervals."""
from fractions import Fraction

import pytest
import torch
import torchpairwise as tp


def _reference(x, y, directions, p):
    """Independent rational quantile partition for two unpadded clouds."""
    a, b = (value @ directions for value in (x, y))
    a, b = a.sort(0).values, b.sort(0).values
    boundaries = sorted({Fraction(i, len(x)) for i in range(len(x) + 1)}
                        | {Fraction(i, len(y)) for i in range(len(y) + 1)})
    result = a.sum() * 0.
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        midpoint = (left + right) / 2
        ai, bi = int(midpoint * len(x)), int(midpoint * len(y))
        result = result + (a[ai] - b[bi]).abs().pow(p).mean() * float(right - left)
    return result if p == 1 else result.sqrt()


@pytest.mark.parametrize('p', [1, 2])
def test_sliced_exact_quantiles_layout_batches_modes_and_gradients(p, device):
    torch.manual_seed(193)
    x = torch.rand(2, 3, 4, 2, device=device, dtype=torch.float64)
    y = torch.rand(1, 3, 3, 2, device=device, dtype=x.dtype) + .2
    xl = torch.tensor([[4, 1, 2], [3, 2, 4]], device=device)
    yl = torch.tensor([[2, 3, 1]], device=device)
    x[torch.arange(4, device=device) >= xl[..., None]] = float('nan')
    y[torch.arange(3, device=device) >= yl[..., None]] = float('nan')
    # Explicit channel-first spatial layout exercises the common point contract.
    x = x.reshape(2, 3, 2, 2, 2).movedim(-1, 2).contiguous().requires_grad_()
    y = y.reshape(1, 3, 1, 3, 2).movedim(-1, 2).contiguous().requires_grad_()
    xf, yf = x.movedim(2, -1).flatten(2, -2), y.movedim(2, -1).flatten(2, -2)
    directions = torch.tensor([[1., 0., .6], [0., 1., .8]], dtype=x.dtype, device=device,
                              requires_grad=True)
    kwargs = dict(projections=directions, p=p, batched=True, feature_dim=2,
                  x1_lengths=xl, x2_lengths=yl)
    expected = torch.stack([torch.stack([torch.stack([
        _reference(xf[b, i, :int(xl[b, i])], yf[0, j, :int(yl[0, j])], directions, p)
        for j in range(3)]) for i in range(3)]) for b in range(2)])
    actual = tp.sliced_wasserstein_distances(x, y, **kwargs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(tp.cdist(x, y, metric='sliced_wasserstein', chunk_size=2, **kwargs), expected)
    paired = tp.paired_distances(x, y, metric='sliced_wasserstein', **kwargs)
    torch.testing.assert_close(paired, expected.diagonal(dim1=-2, dim2=-1))
    for output, oracle in ((actual, expected), (paired, expected.diagonal(dim1=-2, dim2=-1))):
        for grad, ref in zip(torch.autograd.grad(output.sum(), (x, y, directions), retain_graph=True),
                             torch.autograd.grad(oracle.sum(), (x, y, directions), retain_graph=True)):
            torch.testing.assert_close(grad, ref)
            assert torch.isfinite(grad).all()
    kwargs.pop('x2_lengths')
    self_matrix = tp.sliced_wasserstein_distances(x, **kwargs)
    i, j = torch.triu_indices(3, 3, 1, device=device)
    condensed = tp.pdist(x, metric='sliced_wasserstein', chunk_size=2, **kwargs)
    torch.testing.assert_close(condensed, self_matrix[..., i, j])
    torch.testing.assert_close(torch.autograd.grad(condensed.sum(), x)[0],
                               torch.autograd.grad(self_matrix[..., i, j].sum(), x)[0])


@pytest.mark.parametrize('p', [1, 2])
def test_sliced_equal_counts_fast_path_unequal_counts_and_zero_gradients(p, device):
    directions = torch.eye(2, device=device, dtype=torch.float64)
    x = torch.tensor([[[0., 1.], [3., 4.]], [[2., -1.], [5., 3.]]], device=device,
                     dtype=directions.dtype, requires_grad=True)
    y = torch.tensor([[[0., 2.], [1., 4.], [2., 5.]]], device=device, dtype=x.dtype)
    expected = torch.stack([_reference(cloud, y[0], directions, p) for cloud in x])[:, None]
    torch.testing.assert_close(tp.sliced_wasserstein_distances(x, y, projections=directions, p=p), expected)
    options = dict(projections=directions, p=p)
    full = tp.sliced_wasserstein_distances(x, x.flip(0), **options)
    lengths = torch.full((2,), 2, device=device, dtype=torch.int64)
    slow = tp.sliced_wasserstein_distances(x, x.flip(0), x1_lengths=lengths, x2_lengths=lengths, **options)
    torch.testing.assert_close(full, slow)
    torch.testing.assert_close(full, tp.sliced_wasserstein_distances(x, x.flip(0), chunk_size=1, **options))
    paired = tp.paired_distances(x, x, metric='sliced_wasserstein', **options)
    torch.testing.assert_close(paired, torch.zeros_like(paired))
    torch.testing.assert_close(torch.autograd.grad(paired.sum(), x)[0], torch.zeros_like(x))
    torch.testing.assert_close(tp.pdist(x, metric='sliced_wasserstein', **options), full[0, :1])
    for n in (0, 1):
        empty = tp.pdist(x[:n], metric='sliced_wasserstein', **options)
        assert empty.shape == (0,)
        torch.testing.assert_close(torch.autograd.grad(empty.sum(), x)[0], torch.zeros_like(x))
    assert tp.sliced_wasserstein_distances(x[:0], y, **options).shape == (0, 1)
    assert tp.paired_distances(x[:0], x[:0], metric='sliced_wasserstein', **options).shape == (0,)
    empty_batch = tp.sliced_wasserstein_distances(x.expand(0, *x.shape), y.unsqueeze(0), **options)
    assert empty_batch.shape == (0, 2, 1)


def test_sliced_validation_and_autocast(device):
    x = torch.ones(2, 3, 2, device=device)
    directions = torch.eye(2, device=device)
    for p in (0, 3, True, float('inf')):
        with pytest.raises(ValueError, match='p must be'):
            tp.sliced_wasserstein_distances(x, projections=directions, p=p)
    for bad in (torch.ones(2, device=device), directions[:, :0], directions * 2):
        with pytest.raises(ValueError, match='projections'):
            tp.sliced_wasserstein_distances(x, projections=bad)
    with pytest.raises(ValueError, match='dtype and device'):
        tp.sliced_wasserstein_distances(x, projections=directions.double())
    with pytest.raises(ValueError, match='finite'):
        tp.sliced_wasserstein_distances(x * float('nan'), projections=directions)
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    low = torch.rand(2, 3, 2, device=device, dtype=dtype, requires_grad=True)
    low_directions = directions.to(dtype)
    with pytest.raises(TypeError, match='float32 or float64'):
        tp.sliced_wasserstein_distances(low, projections=low_directions)
    with torch.autocast(device, dtype=dtype):
        actual = tp.sliced_wasserstein_distances(low, low.flip(0), projections=low_directions)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, tp.sliced_wasserstein_distances(
        low.float(), low.flip(0).float(), projections=directions))
    assert torch.isfinite(torch.autograd.grad(actual.sum(), low)[0]).all()


@pytest.mark.compiler
def test_sliced_compile_ragged_modes(device):
    torch.compiler.reset()
    x = torch.rand(3, 4, 2, device=device, requires_grad=True)
    y = torch.rand(3, 3, 2, device=device, requires_grad=True)
    xl = torch.tensor([2, 3, 4], device=device)
    yl = torch.tensor([3, 1, 2], device=device)
    directions = torch.eye(2, device=device)
    def call(a, b, al, bl):
        kwargs = dict(projections=directions, x1_lengths=al, x2_lengths=bl, chunk_size=2)
        return (tp.sliced_wasserstein_distances(a, b, **kwargs).sum()
                + tp.paired_distances(a, b, metric='sliced_wasserstein', **kwargs).sum()
                + tp.pdist(a, metric='sliced_wasserstein', projections=directions,
                           x1_lengths=al, chunk_size=2).sum())
    expected = call(x, y, xl, yl)
    actual = torch.compile(call, fullgraph=True)(x, y, xl, yl)
    torch.testing.assert_close(actual, expected)
    for grad, ref in zip(torch.autograd.grad(actual, (x, y)), torch.autograd.grad(expected, (x, y))):
        torch.testing.assert_close(grad, ref)
    torch.compiler.reset()
