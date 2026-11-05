"""Compact diagonal actions and fixed SPD inverse actions for native CG."""
import io

import pytest
import torch
import torchpairwise as tp
import torchpairwise as s


def test_diagonal_actions_and_jacobi_contract(device):
    d = torch.tensor([[1., 2., 4.], [3., 5., 7.]], device=device, dtype=torch.float64, requires_grad=True)
    v = torch.randn(3, 2, device=device, dtype=d.dtype, requires_grad=True)
    a = s.diag_embed(d)
    actual = a.T.matmul(v) + s.linalg.jacobi(d).matmul(v)
    expected = (torch.diag_embed(d) + torch.diag_embed(d.reciprocal())) @ v
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), (d, v)),
                               torch.autograd.grad(expected.square().sum(), (d, v)))
    with torch.no_grad():
        a = s.diag_embed(d)
        inverse = s.linalg.jacobi(d)
        previous = inverse.matmul(v).clone()
        d.mul_(2)
        torch.testing.assert_close(a.matmul(v), d.unsqueeze(-1) * v)
        torch.testing.assert_close(inverse.matmul(v), previous)  # Reciprocal snapshot.
    for shape in ((0,), (0, 3)):
        diagonal = torch.ones(shape, device=device, dtype=d.dtype)
        values = torch.empty((*shape, 2), device=device, dtype=d.dtype)
        assert s.diag_embed(diagonal).matmul(values).shape == values.shape
        assert s.linalg.cg(s.diag_embed(diagonal), values, preconditioner=s.linalg.jacobi(diagonal)).converged.all()
    for bad in ([0., 1.], [-1., 2.], [float('nan'), 1.], [float('inf'), 1.], [1e-320, 1.]):
        with pytest.raises(RuntimeError, match='Jacobi diagonal'):
            s.linalg.jacobi(torch.tensor(bad, device=device, dtype=d.dtype))
    with pytest.raises(ValueError, match='Diagonal entries'):
        s.diag_embed(torch.tensor(1.0, device=device))


def test_pcg_dense_oracle_independent_status_and_broadcast(device):
    torch.manual_seed(27)
    x = torch.randn(2, 24, 4, device=device, dtype=torch.float64)
    d = torch.logspace(-3, 3, 24, device=device, dtype=x.dtype)
    D = s.diag_embed(d.sqrt())
    K = s.lazy(x, gamma=0.2, chunk_size=7, metric='rbf') + 2.0 * tp.eye_like(s.lazy(x, gamma=0.2, chunk_size=7, metric='rbf'))
    A = D @ K @ D
    rhs = torch.randn(24, 2, device=device, dtype=x.dtype)
    M = s.linalg.jacobi(3. * d)
    fit = s.linalg.cg(A, rhs, preconditioner=M, rtol=1e-9, max_iter=100)
    plain = s.linalg.cg(A, rhs, rtol=1e-9, max_iter=300)
    matrix = A.matmul(torch.eye(24, device=device, dtype=x.dtype))
    assert fit.converged.all() and plain.converged.all()
    assert (fit.iterations < plain.iterations).all()
    torch.testing.assert_close(fit.solution, torch.linalg.solve(matrix, rhs), rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(fit.residual_norm, torch.linalg.vector_norm(rhs - A.matmul(fit.solution), dim=-2), rtol=1e-5, atol=1e-14)
    initial = s.linalg.cg(A, rhs, initial=fit.solution, preconditioner=M, rtol=1e-9, max_iter=0)
    assert initial.converged.all() and not initial.iterations.any()
    # Independent RHS/batches: successful zero column, invalid and nonfinite maps.
    rhs = x.new_tensor([[0., 1.], [0., 2.], [0., 3.]])
    inverse = x.new_tensor([[1., 1., 1.], [-1., -1., -1.], [float('nan'), 1., 1.]])
    fit = s.linalg.cg(s.diag_embed(torch.ones_like(inverse)), rhs, preconditioner=s.diag_embed(inverse))
    assert fit.status.tolist() == [[0, 0], [0, 4], [0, 3]]
    assert fit.iterations.tolist() == [[0, 1], [0, 0], [0, 0]]
    assert torch.equal(fit.converged, fit.status == 0)
    for dtype in (torch.float32, torch.float64):
        d = torch.tensor([1., 4., 16.], device=device, dtype=dtype)
        fit = s.linalg.cg(s.diag_embed(d), d, preconditioner=s.linalg.jacobi(d))
        torch.testing.assert_close(fit.solution, torch.ones_like(d))
        assert fit.converged and fit.iterations == 1
    # An arbitrary composed SPD inverse action and RHS-defined batch axes.
    inverse = s.diag_embed(d.rsqrt()) @ s.diag_embed(d.rsqrt())
    rhs = d.expand(2, 3).unsqueeze(-1)
    fit = s.linalg.cg(s.diag_embed(d), rhs, preconditioner=inverse)
    torch.testing.assert_close(fit.solution, torch.ones_like(rhs))
    fit = s.linalg.cg(s.diag_embed(d), d, preconditioner=torch.diag(d.reciprocal()))
    torch.testing.assert_close(fit.solution, torch.ones_like(d))
    # Zero budget evaluates only the original guess, even with an invalid M.
    fit = s.linalg.cg(s.diag_embed(d), d, preconditioner=s.diag_embed(-d), max_iter=0)
    assert fit.status == 1 and fit.iterations == 0
    with pytest.raises((ValueError, RuntimeError), match='batch'):
        s.linalg.cg(s.diag_embed(d), d, preconditioner=s.diag_embed(d.expand(2, 3)))
    with pytest.raises(RuntimeError, match='dimensions'):
        s.linalg.cg(s.diag_embed(d), d, preconditioner=s.diag_embed(d[:2]))
    with pytest.raises(RuntimeError, match='inference-only'):
        s.linalg.cg(s.diag_embed(d), d, preconditioner=s.diag_embed(d.clone().requires_grad_()))
    with torch.no_grad():
        assert s.linalg.cg(s.diag_embed(d), d, preconditioner=s.diag_embed(d.clone().requires_grad_())).converged


@pytest.mark.compiler
def test_pcg_dynamic_compile_and_export(device):
    def action(d, v):
        return s.diag_embed(d).T.matmul(v) + s.linalg.jacobi(d).matmul(v)
    d = torch.tensor([1., 2., 4.], device=device, dtype=torch.float64, requires_grad=True)
    v = torch.randn(3, 2, device=device, dtype=d.dtype, requires_grad=True)
    actual = torch.compile(action, fullgraph=True)(d, v)
    expected = (d + d.reciprocal()).unsqueeze(-1) * v
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), (d, v)),
                               torch.autograd.grad(expected.square().sum(), (d, v)))
    class Solve(torch.nn.Module):
        def forward(self, x, d, b):
            D = s.diag_embed(d.sqrt())
            kernel = s.lazy(x, gamma=0.2, chunk_size=3, metric='rbf')
            K = kernel + 2.0 * tp.eye_like(kernel)
            A = D.matmul(K).matmul(D)
            return s.linalg.cg(A, b, preconditioner=s.linalg.jacobi(3. * d), rtol=1e-8, max_iter=60)
    module = Solve()
    def inputs(n):
        x = torch.randn(n, 3, device=device, dtype=torch.float64)
        return x, torch.logspace(-2, 2, n, device=device, dtype=x.dtype), torch.randn(n, 2, device=device, dtype=x.dtype)
    compiled = torch.compile(module, backend='inductor', fullgraph=True, dynamic=True)
    for n in (7, 10):
        values = inputs(n)
        torch.testing.assert_close(compiled(*values), module(*values))
    n = torch.export.Dim('samples', min=4, max=16)
    values = inputs(7)
    exported = torch.export.export(module, values, dynamic_shapes=({0: n}, {0: n}, {0: n}))
    buffer = io.BytesIO()
    torch.export.save(exported, buffer)
    buffer.seek(0)
    restored = torch.export.load(buffer).module()
    values = inputs(11)
    torch.testing.assert_close(restored(*values), module(*values))
