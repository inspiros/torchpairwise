"""Lanczos projection identities and reusable GP variance caches."""
import pytest
import torch
import torchpairwise as tp


def _lanczos_reference(a, initial, k):
    q = initial / initial.norm()
    basis, alpha, beta = [], [], []
    previous = torch.zeros_like(q)
    coupling = a.new_zeros(())
    for j in range(k):
        basis.append(q)
        v = a @ q - coupling * previous
        diagonal = q @ v
        alpha.append(diagonal)
        v = v - diagonal * q
        Q = torch.stack(basis, -1)
        v = v - Q @ (Q.T @ v)
        if j + 1 < k:
            coupling = v.norm()
            beta.append(coupling)
            previous, q = q, v / coupling
    return torch.stack(basis, -1), torch.stack(alpha), torch.stack(beta)


def test_lanczos_dense_projection_and_reference(device):
    a = torch.diag(torch.arange(1., 10., device=device, dtype=torch.float64))
    initial = torch.linspace(.5, 2., 9, device=device, dtype=a.dtype)
    fit = tp.linalg.lanczos(tp.as_lazy(a), 5, initial=initial)
    q, alpha, beta = _lanczos_reference(a, initial, 5)
    torch.testing.assert_close(fit.basis, q)
    torch.testing.assert_close(fit.alpha, alpha)
    torch.testing.assert_close(fit.beta, beta)
    tridiagonal = torch.diag(fit.alpha) + torch.diag(fit.beta, 1) + torch.diag(fit.beta, -1)
    torch.testing.assert_close(fit.basis.T @ a @ fit.basis, tridiagonal, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(fit.basis.T @ fit.basis, torch.eye(5, device=device, dtype=a.dtype), atol=1e-12, rtol=1e-10)
    assert fit.iterations == 5 and fit.status == tp.linalg.LanczosStatus.COMPLETED


def test_lanczos_independent_breakdown_and_invalid_initial(device):
    a = torch.diag_embed(torch.tensor([[2., 2., 2., 2.], [1., 2., 3., 4.]], device=device, dtype=torch.float64))
    fit = tp.linalg.lanczos(tp.as_lazy(a), 3, initial=torch.ones(2, 4, device=device, dtype=a.dtype))
    assert fit.iterations.tolist() == [1, 3]
    assert fit.status.tolist() == [int(tp.linalg.LanczosStatus.BREAKDOWN), int(tp.linalg.LanczosStatus.COMPLETED)]
    assert not fit.basis[0, :, 1:].any() and not fit.alpha[0, 1:].any() and not fit.beta[0].any()
    invalid = tp.linalg.lanczos(tp.as_lazy(a[0]), 3, initial=torch.zeros(4, device=device, dtype=a.dtype))
    assert invalid.status == tp.linalg.LanczosStatus.INVALID_INITIAL
    assert invalid.iterations == 0


def test_love_full_rank_variance_and_cross_gradients(device):
    x = torch.linspace(-2., 2., 8, dtype=torch.float64, device=device)[:, None]
    test = torch.tensor([[-1.7], [-.3], [.8]], device=device, dtype=x.dtype, requires_grad=True)
    k = tp.lazy(x, metric='rbf', gamma=.7)
    a = k + .2 * tp.eye_like(k)
    initial = torch.arange(1., 9., device=device, dtype=x.dtype)
    cache = tp.linalg.love_cache(a, 8, initial=initial)
    cross = tp.lazy(test, x, metric='rbf', gamma=.7, chunk_size=3)
    actual = tp.linalg.predictive_variance(cache, cross, torch.ones(3, device=device, dtype=x.dtype))
    dense_a = tp.rbf_kernel(x, gamma=.7) + .2 * torch.eye(8, device=device, dtype=x.dtype)
    dense_cross = tp.rbf_kernel(test, x, gamma=.7)
    expected = (1 - (dense_cross * torch.linalg.solve(dense_a, dense_cross.T).T).sum(-1)).clamp_min(0)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    gradient, = torch.autograd.grad(actual.sum(), test)
    reference, = torch.autograd.grad(expected.sum(), test)
    torch.testing.assert_close(gradient, reference, atol=1e-9, rtol=1e-8)
    # A cache is independent of the number of test rows and the clamp is explicit.
    clamped = tp.linalg.predictive_variance(cache, cross, torch.zeros(3, device=device, dtype=x.dtype))
    assert not clamped.any() and not cache.root.requires_grad


def test_love_nested_rank_error_decreases(device):
    x = torch.linspace(-2., 2., 9, device=device, dtype=torch.float64)[:, None]
    k = tp.lazy(x, metric='rbf', gamma=.6)
    a = k + .3 * tp.eye_like(k)
    dense = tp.rbf_kernel(x, gamma=.6)
    exact = 1 - (dense * torch.linalg.solve(dense + .3 * torch.eye(9, device=device, dtype=x.dtype), dense.T).T).sum(-1)
    initial = torch.arange(1., 10., device=device, dtype=x.dtype)
    errors = []
    for rank in (2, 4, 9):
        cache = tp.linalg.love_cache(a, rank, initial=initial)
        variance = tp.linalg.predictive_variance(cache, k, torch.ones(9, device=device, dtype=x.dtype))
        errors.append((variance - exact).norm())
    assert errors[1] <= errors[0] + 1e-10 and errors[2] <= errors[1] + 1e-10


def test_lanczos_and_love_validation(device):
    a = tp.as_lazy(torch.eye(4, device=device, dtype=torch.float64))
    initial = torch.ones(4, device=device, dtype=torch.float64)
    for rank in (0, 5):
        with pytest.raises((ValueError, RuntimeError)):
            tp.linalg.lanczos(a, rank, initial=initial)
        with pytest.raises((ValueError, RuntimeError)):
            tp.linalg.love_cache(a, rank, initial=initial)
    with pytest.raises((ValueError, RuntimeError)):
        tp.linalg.lanczos(a, 2, initial=initial[:3])
    with pytest.raises(RuntimeError, match='inference-only'):
        tp.linalg.love_cache(a, 2, initial=initial.clone().requires_grad_())
    with pytest.raises(RuntimeError, match='positive definite'):
        tp.linalg.love_cache(-torch.eye(4, device=device, dtype=initial.dtype), 2, initial=initial)
    nonfinite = tp.linalg.lanczos(torch.full((4, 4), float('nan'), device=device,
                                            dtype=initial.dtype), 2, initial=initial)
    assert nonfinite.status == tp.linalg.LanczosStatus.NONFINITE
    assert nonfinite.iterations == 0
    with pytest.raises(RuntimeError, match='training axis'):
        tp.linalg.predictive_variance(tp.linalg.LOVECache(initial[:, None]),
                                      torch.ones(2, 3, device=device, dtype=initial.dtype),
                                      initial[:2])


def test_love_independent_breakdown_and_empty_batch(device):
    a = torch.diag_embed(torch.tensor([[2., 2., 2., 2.], [1., 2., 3., 4.]],
                                      device=device, dtype=torch.float64))
    initial = torch.ones(4, device=device, dtype=a.dtype)
    cache = tp.linalg.love_cache(a, 4, initial=initial)
    assert not cache.root[0, :, 1:].any()
    torch.testing.assert_close(cache.root[1] @ cache.root[1].T, torch.linalg.inv(a[1]))
    empty = tp.linalg.lanczos(a[:0], 3, initial=initial)
    assert empty.basis.shape == (0, 4, 3) and empty.status.shape == (0,)


@pytest.mark.compiler
def test_lanczos_compile_and_export(device):
    class Projection(torch.nn.Module):
        def forward(self, a, initial):
            result = tp.linalg.lanczos(tp.as_lazy(a), 3, initial=initial)
            return result.basis, result.alpha, result.beta, result.iterations, result.status

    module = Projection()
    a = torch.diag(torch.arange(1., 6., device=device, dtype=torch.float64))
    initial = torch.ones(5, device=device, dtype=a.dtype)
    expected = module(a, initial)
    torch.testing.assert_close(torch.compile(module, fullgraph=True)(a, initial), expected)
    exported = torch.export.export(module, (a, initial))
    torch.testing.assert_close(exported.module()(a, initial), expected)
