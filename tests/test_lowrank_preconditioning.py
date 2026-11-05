"""Spectral inverse actions: dense references, Nyström setup and deployment."""
import io

import pytest
import torch
import torchpairwise as tp
import torchpairwise as s


def test_lowrank_inverse_dense_actions_and_boundaries(device):
    torch.manual_seed(118)
    for dtype in (torch.float32, torch.float64):
        features = torch.randn(2, 8, 3, device=device, dtype=dtype)
        # A repeated column is valid: no full-column-rank assumption.
        features[..., 2] = features[..., 0]
        rhs = torch.randn(8, 2, device=device, dtype=dtype, requires_grad=True)
        M = s.linalg.lowrank_inverse(features, .3)
        actual = M.T.matmul(rhs)
        matrix = features @ features.transpose(-1, -2) + .3 * torch.eye(8, device=device, dtype=dtype)
        # Match the prepared double arithmetic before inversion; rounding F F.T
        # to float32 first perturbs a poorly conditioned dense gradient oracle.
        accurate = features.double() @ features.double().transpose(-1, -2)
        accurate = accurate + features.new_tensor(.3).double() * torch.eye(8, device=device, dtype=torch.float64)
        expected = torch.linalg.solve(accurate, rhs.double()).to(dtype)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        nystrom = s.linalg.nystrom_inverse(features, torch.eye(3, device=device, dtype=dtype), .3)
        torch.testing.assert_close(nystrom.matmul(rhs), actual)
        torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), rhs),
                                   torch.autograd.grad(expected.square().sum(), rhs), rtol=3e-5, atol=1e-5)
        result = s.linalg.cg(matrix, rhs.detach(), preconditioner=M, rtol=1e-5 if dtype == torch.float32 else 1e-10)
        assert result.converged.all() and (result.iterations <= (2 if dtype == torch.float32 else 1)).all()
        torch.testing.assert_close(result.solution, expected.detach(), rtol=3e-5, atol=3e-6)
        # State is a factorization snapshot, not a reference to features/shift.
        alpha = features.new_tensor(.3)
        snapshot = s.linalg.lowrank_inverse(features, alpha)
        previous = snapshot.matmul(rhs).detach()
        features.zero_()
        alpha.fill_(2.)
        torch.testing.assert_close(snapshot.matmul(rhs), previous)
    for shape in ((0, 3), (4, 0), (0, 4, 2), (3, 5)):
        f = torch.zeros(shape, device=device, dtype=torch.float64)
        rhs = torch.randn((*shape[:-1], 2), device=device, dtype=f.dtype)
        torch.testing.assert_close(s.linalg.lowrank_inverse(f, .5).matmul(rhs), 2. * rhs)
    f = torch.randn(5, 2, device=device, dtype=torch.float64)
    with pytest.raises(RuntimeError, match='inference-only'):
        s.linalg.lowrank_inverse(f.requires_grad_(), .2)
    with torch.no_grad():
        assert not s.linalg.lowrank_inverse(f, .2).matmul(f).requires_grad
    for shift in (0., -1., float('nan'), float('inf'), 1e-320):
        with pytest.raises(RuntimeError, match='shift'):
            s.linalg.lowrank_inverse(f.detach(), shift)
    with pytest.raises(RuntimeError, match='finite'):
        s.linalg.lowrank_inverse(f.detach().fill_(float('nan')), .2)


def test_nystrom_inverse_dense_and_pcg(device):
    torch.manual_seed(119)
    # Broadcast C/W batches and compare the exact regularized approximation.
    cross = torch.randn(2, 9, 3, device=device, dtype=torch.float64)
    q = torch.randn(1, 3, 3, device=device, dtype=cross.dtype)
    w = q @ q.transpose(-1, -2)
    rhs = torch.randn(9, 2, device=device, dtype=cross.dtype)
    M = s.linalg.nystrom_inverse(cross, w, .4, jitter=.1)
    approximation = cross @ torch.linalg.solve(w + .1 * torch.eye(3, device=device, dtype=w.dtype), cross.transpose(-1, -2))
    matrix = approximation + .4 * torch.eye(9, device=device, dtype=cross.dtype)
    torch.testing.assert_close(M.matmul(rhs), torch.linalg.solve(matrix, rhs), rtol=1e-10, atol=1e-10)
    fit = s.linalg.cg(matrix, rhs, preconditioner=M, rtol=1e-10)
    assert fit.converged.all() and (fit.iterations == 1).all()
    # True constant-diagonal RBF system. Exact CG still uses the full kernel.
    x = torch.randn(32, 2, device=device, dtype=cross.dtype) * .3
    K = s.lazy(x, gamma=0.2, chunk_size=9, metric='rbf')
    rows = torch.arange(32, device=device)
    ids = torch.arange(0, 32, 4, device=device)
    C, W = K.block(rows, ids), K.block(ids, ids)
    M = s.linalg.nystrom_inverse(C, W, .05, jitter=1e-8)
    A = K + 0.05 * tp.eye_like(K)
    rhs = torch.randn(32, 2, device=device, dtype=cross.dtype)
    fit = s.linalg.cg(A, rhs, preconditioner=M, rtol=1e-9, max_iter=100)
    plain = s.linalg.cg(A, rhs, rtol=1e-9, max_iter=100)
    assert fit.converged.all() and plain.converged.all()
    assert (fit.iterations < plain.iterations).all()
    dense = A.matmul(torch.eye(32, device=device, dtype=cross.dtype))
    torch.testing.assert_close(fit.solution, torch.linalg.solve(dense, rhs), rtol=1e-7, atol=1e-7)
    # Explicit jitter handles dependent landmarks; no implicit retry.
    singular = cross.new_ones(3, 3)
    with pytest.raises(RuntimeError, match='positive definite'):
        s.linalg.nystrom_inverse(cross, singular, .4)
    assert s.linalg.nystrom_inverse(cross, singular, .4, jitter=.1).shape == (2, 9, 9)
    for shape in ((0, 3), (4, 0), (0, 4, 2)):
        c = torch.zeros(shape, device=device, dtype=cross.dtype)
        w = torch.eye(shape[-1], device=device, dtype=cross.dtype)
        v = torch.randn((*shape[:-1], 2), device=device, dtype=cross.dtype)
        torch.testing.assert_close(s.linalg.nystrom_inverse(c, w, .5).matmul(v), 2 * v)
    w = torch.eye(3, device=device, dtype=cross.dtype)
    w[0, 1] = .5
    with pytest.raises(RuntimeError, match='symmetric'):
        s.linalg.nystrom_inverse(cross, w, .4)
    with pytest.raises(RuntimeError, match='inference-only'):
        s.linalg.nystrom_inverse(cross, w.requires_grad_(), .4)
    with pytest.raises(RuntimeError, match='jitter'):
        s.linalg.nystrom_inverse(cross, w.detach(), .4, jitter=-1.)


@pytest.mark.compiler
def test_lowrank_dynamic_compile_export_and_rhs_gradients(device):
    def action(features, rhs):
        return s.linalg.lowrank_inverse(features, .3).matmul(rhs)
    f = torch.randn(7, 3, device=device, dtype=torch.float64)
    rhs = torch.randn(7, 2, device=device, dtype=f.dtype, requires_grad=True)
    compiled_action = torch.compile(action, fullgraph=True, dynamic=True)
    actual = compiled_action(f, rhs)
    expected = action(f, rhs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), rhs),
                               torch.autograd.grad(expected.square().sum(), rhs))
    # Thin rank changes from L to N when the feature matrix becomes wide.
    small_f, small_rhs = f[:2].clone(), rhs[:2].detach().clone()
    torch.testing.assert_close(compiled_action(small_f, small_rhs), action(small_f, small_rhs))
    class Solve(torch.nn.Module):
        def forward(self, c, w, b):
            M = s.linalg.nystrom_inverse(c, w, .3, jitter=.1)
            F = s.as_lazy(c)
            gram = F.matmul(F.T)
            A = gram + 0.3 * tp.eye_like(gram)
            return s.linalg.cg(A, b, preconditioner=M, rtol=1e-8, max_iter=60)
    module = Solve()
    def inputs(n):
        return (torch.randn(n, 3, device=device, dtype=f.dtype),
                torch.eye(3, device=device, dtype=f.dtype),
                torch.randn(n, 2, device=device, dtype=f.dtype))
    compiled = torch.compile(module, fullgraph=True, dynamic=True)
    for n in (7, 10):
        values = inputs(n)
        torch.testing.assert_close(compiled(*values), module(*values))
    n = torch.export.Dim('samples', min=4, max=16)
    values = inputs(7)
    exported = torch.export.export(module, values, dynamic_shapes=({0: n}, {}, {0: n}))
    buffer = io.BytesIO()
    torch.export.save(exported, buffer)
    buffer.seek(0)
    restored = torch.export.load(buffer).module()
    values = inputs(11)
    torch.testing.assert_close(restored(*values), module(*values))
