"""A dense oracle for the bounded softmax contraction, without a source grid."""
import pytest
import torch
import torchpairwise as tp


def test_softmax_matmul_values_gradients_and_boundaries(device):
    torch.manual_seed(129)
    matrix = torch.randn(2, 1, 3, 4, device=device, dtype=torch.float64, requires_grad=True)
    values = torch.randn(1, 2, 3, 5, device=device, dtype=matrix.dtype, requires_grad=True)
    # Directed transpose, independent broadcast batches, partial sample and
    # channel tiles, and gradients into both sources and payload.
    op = tp.as_lazy(matrix, chunk_size=2).T
    actual = op.softmax_matmul(values)
    expected = matrix.transpose(-1, -2).softmax(-1) @ values
    torch.testing.assert_close(actual, expected)
    cot = torch.randn_like(actual)
    torch.testing.assert_close(torch.autograd.grad(actual, (matrix, values), cot),
                               torch.autograd.grad(expected, (matrix, values), cot))
    x = torch.rand(4, 2, device=device, dtype=matrix.dtype, requires_grad=True)
    gamma = torch.tensor(.7, device=device, dtype=matrix.dtype, requires_grad=True)
    v = torch.randn(4, device=device, dtype=matrix.dtype, requires_grad=True)
    score = -gamma * tp.lazy(x, metric='sqeuclidean', chunk_size=2)
    dense = -gamma * (x[:, None] - x[None]).square().sum(-1)
    mask = torch.eye(4, device=device, dtype=torch.bool)
    actual = score.softmax_matmul(v, exclude_self=True)
    expected = dense.masked_fill(mask, -torch.inf).softmax(-1) @ v
    torch.testing.assert_close(actual, expected)
    cot = torch.randn_like(actual)
    torch.testing.assert_close(torch.autograd.grad(actual, (x, gamma, v), cot),
                               torch.autograd.grad(expected, (x, gamma, v), cot))
    # A single logsumexp in entry dtype loses the normalizer for huge offsets.
    # Keep maximum and scaled mass separately; also allow early all -inf tiles.
    for data in ([[1e30] * 5], [[-torch.inf, -torch.inf, 1000., 999., -torch.inf]],
                 [[-torch.inf] * 5], [[torch.inf, 1., 0., 2., 3.]],
                 [[torch.nan, 1., 0., 2., 3.]]):
        raw = torch.tensor(data, device=device, dtype=torch.float32, requires_grad=True)
        v = torch.arange(5, device=device, dtype=raw.dtype, requires_grad=True)
        actual = tp.as_lazy(raw, chunk_size=2).softmax_matmul(v)
        expected = raw.softmax(-1) @ v
        torch.testing.assert_close(actual, expected, equal_nan=True)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), (raw, v)),
                                   torch.autograd.grad(expected.sum(), (raw, v)), equal_nan=True)
    for n, m, exclude in ((0, 3, False), (2, 0, False), (1, 1, True)):
        raw = torch.zeros(n, m, device=device, requires_grad=True)
        v = torch.ones(m, 3, device=device, requires_grad=True)
        actual = tp.as_lazy(raw, chunk_size=1).softmax_matmul(v, exclude_self=exclude)
        assert actual.shape == (n, 3) and actual.eq(0).all()
        gradients = torch.autograd.grad(actual.sum(), (raw, v))
        assert all(g.eq(0).all() for g in gradients)
    with pytest.raises(RuntimeError, match='square'):
        op.softmax_matmul(values, exclude_self=True)
    with pytest.raises(RuntimeError, match='float32/float64'):
        tp.as_lazy(torch.ones(2, 3, device=device, dtype=torch.int64)).softmax_matmul(torch.ones(3, device=device))
