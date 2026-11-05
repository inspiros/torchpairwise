"""Streaming pivot choices, stopping diagnostics and GP inverse actions."""
import pytest
import torch
import torchpairwise as tp


def _reference(a, rank):
    residual = a.diagonal().clone()
    factor = a.new_zeros(a.size(0), rank)
    pivots = []
    for j in range(rank):
        p = int(residual.argmax())
        pivots.append(p)
        factor[:, j] = (a[:, p] - factor[:, :j] @ factor[p, :j]) / residual[p].sqrt()
        residual = (residual - factor[:, j].square()).clamp_min(0)
        residual[pivots] = 0
    return factor, torch.tensor(pivots, device=a.device), residual.sum()


def test_pivots_dense_reference_and_full_rank(device):
    torch.manual_seed(401)
    f = torch.randn(9, 9, dtype=torch.float64, device=device)
    a = f @ f.T + torch.eye(9, dtype=f.dtype, device=device)
    expected, pivots, error = _reference(a, 5)
    fit = tp.linalg.pivoted_cholesky(tp.as_lazy(a), 5)
    torch.testing.assert_close(fit.factor, expected)
    torch.testing.assert_close(fit.pivots, pivots)
    torch.testing.assert_close(fit.error, error, atol=1e-12, rtol=1e-10)
    assert fit.rank == 5
    full = tp.linalg.pivoted_cholesky(tp.as_lazy(a), 9)
    torch.testing.assert_close(full.factor @ full.factor.T, a)


def test_independent_trace_stopping_and_padding(device):
    d = torch.tensor([[4., 0., 0., 0.], [4., 3., 2., 1.]], device=device, dtype=torch.float64)
    fit = tp.linalg.pivoted_cholesky(tp.as_lazy(torch.diag_embed(d)), 4, rtol=.31)
    assert fit.rank.tolist() == [1, 2]
    assert fit.pivots.tolist() == [[0, -1, -1, -1], [0, 1, -1, -1]]
    assert not fit.factor[0, :, 1:].any() and not fit.factor[1, :, 2:].any()
    torch.testing.assert_close(fit.error, d.new_tensor([0., 3.]))
    stopped = tp.linalg.pivoted_cholesky(tp.as_lazy(torch.diag(d[1])), 4, atol=10.)
    assert stopped.rank == 0 and not stopped.factor.any()


def test_kernel_preconditioner_reduces_cg_steps(device):
    x = torch.linspace(-2, 2, 24, device=device, dtype=torch.float64)[:, None]
    k = tp.lazy(x, metric='rbf', gamma=.8, chunk_size=7)
    a = k + .001 * tp.eye_like(k)
    rhs = torch.cos(3 * x)
    fit = tp.linalg.pivoted_cholesky(k, 12)
    inverse = tp.linalg.lowrank_inverse(fit.factor, .001)
    plain = tp.linalg.cg(a, rhs, rtol=1e-8, max_iter=150)
    preconditioned = tp.linalg.cg(a, rhs, preconditioner=inverse, rtol=1e-8, max_iter=150)
    assert plain.converged.all() and preconditioned.converged.all()
    assert (preconditioned.iterations < plain.iterations).all()


def test_pivoted_cholesky_validation(device):
    a = tp.as_lazy(torch.eye(4, dtype=torch.float64, device=device))
    for options in ({'rank': -1}, {'rank': 5}, {'rank': 2, 'rtol': -1.}, {'rank': 2, 'atol': float('nan')}):
        with pytest.raises((ValueError, RuntimeError)):
            tp.linalg.pivoted_cholesky(a, **options)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        tp.linalg.pivoted_cholesky(a + .1 * tp.eye_like(a), 2)
    with pytest.raises(RuntimeError, match='inference-only'):
        tp.linalg.pivoted_cholesky(torch.eye(4, device=device, requires_grad=True), 2)
    with pytest.raises(RuntimeError, match='diagonal'):
        tp.linalg.pivoted_cholesky(-torch.eye(4, device=device), 2)


def test_scalar_expression_columns_and_float32(device):
    x = torch.linspace(-1., 2., 7, device=device)[:, None]
    weights = torch.linspace(.5, 1.5, 7, device=device)
    kernel = tp.lazy(x, metric='rbf', gamma=.8, chunk_size=2)
    expression = kernel.weight_rows(weights).weight_columns(weights).T
    fit = tp.linalg.pivoted_cholesky(expression, 5)
    dense = tp.rbf_kernel(x.double(), gamma=.8) * weights.double()[:, None] * weights.double()[None, :]
    reference, pivots, _ = _reference(dense, 5)
    assert fit.factor.dtype == x.dtype and fit.error.dtype == x.dtype
    torch.testing.assert_close(fit.factor, reference.float(), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(fit.pivots, pivots)
