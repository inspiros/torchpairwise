"""Restarted GMRES: nonsymmetric oracles, independent status and deployment."""
import io

import pytest
import torch
import torchpairwise as tp
import torchpairwise as s


def test_gmres_dense_and_streamed_restarts(device):
    torch.manual_seed(921)
    for dtype in (torch.float32, torch.float64):
        a = torch.randn(2, 9, 9, device=device, dtype=dtype)*.12
        a += 2*torch.eye(9, device=device, dtype=dtype)
        b = torch.randn(9, 3, device=device, dtype=dtype)
        initial = torch.randn(9, 3, device=device, dtype=dtype)
        rtol = 1e-6 if dtype == torch.float32 else 1e-11
        fit = s.linalg.gmres(a, b, initial=initial, restart=3, max_iter=100, rtol=rtol)
        assert fit.converged.all(), fit
        assert (fit.residual_norm <= rtol*torch.linalg.vector_norm(b.double(), dim=0)).all()
        assert (fit.iterations > 3).any()  # Exercises multiple restart cycles.
        torch.testing.assert_close(fit.solution, torch.linalg.solve(a, b),
                                   rtol=1e-4 if dtype == torch.float32 else 1e-9,
                                   atol=1e-5 if dtype == torch.float32 else 1e-10)
        dense_norm = torch.linalg.vector_norm((b-a@fit.solution).double(), dim=-2)
        # Prepared/tiled and dense BLAS products may accumulate float32 terms
        # differently; bound the induced residual-norm discrepancy explicitly.
        roundoff = 8*torch.finfo(dtype).eps * (
            torch.linalg.matrix_norm(a.double())*torch.linalg.matrix_norm(fit.solution.double())
            + torch.linalg.vector_norm(b.double(), dim=0).max())
        assert ((fit.residual_norm-dense_norm).abs() <= roundoff.unsqueeze(-1)+1e-12).all()
        assert fit.solution.shape == (2, 9, 3) and fit.status.shape == (2, 3)
    # Distinct query/key embeddings of the same vertices: a directed transition.
    x = torch.randn(2, 10, 3, device=device, dtype=torch.float64)
    keys = torch.randn(1, 10, 3, device=device, dtype=x.dtype)
    weights = s.lazy(x, keys, gamma=0.2, chunk_size=3, metric='rbf')
    p = weights.weight_rows(1/weights.sum(dim=-1))
    operator = -0.85 * p.T + 1.0 * tp.eye_like(-0.85 * p.T)
    rhs = torch.ones(10, 2, device=device, dtype=x.dtype)*.015
    dense = tp.rbf_kernel(x, keys, gamma=.2)
    dense = dense/dense.sum(-1, keepdim=True)
    matrix = torch.eye(10, device=device, dtype=x.dtype) - .85*dense.transpose(-1,-2)
    fit = s.linalg.gmres(operator, rhs, restart=4, rtol=1e-10, max_iter=80)
    assert fit.converged.all(), fit
    torch.testing.assert_close(fit.solution, torch.linalg.solve(matrix, rhs), rtol=1e-8, atol=1e-10)
    exact = s.linalg.gmres(operator, rhs, initial=fit.solution, max_iter=0, rtol=1e-9)
    assert exact.converged.all() and not exact.iterations.any()
    assert fit.residual_norm.dtype == torch.float64
    # Happy breakdown (including an indefinite operator) needs true residuals.
    a = torch.diag(torch.tensor([-1., 2., -3.], device=device, dtype=x.dtype))
    fit = s.linalg.gmres(a, torch.ones(3, device=device, dtype=x.dtype), restart=10, rtol=1e-12)
    assert fit.converged and fit.iterations <= 3
    torch.testing.assert_close(fit.solution, 1/a.diag())


def test_gmres_status_limits_and_inference(device):
    dtype = torch.float64
    def t(value):
        return torch.tensor(value, device=device, dtype=dtype)
    identity = torch.eye(3, device=device, dtype=dtype)
    matrices = torch.stack((identity, identity*0, torch.diag(t([1., 2., 4.]))))
    rhs = t([[0., 1., float('nan')], [0., 2., 1.], [0., 3., 1.]])
    fit = s.linalg.gmres(matrices, rhs, max_iter=1, rtol=1e-12)
    assert fit.status.tolist() == [[0, 0, 3], [0, 2, 3], [0, 1, 3]]
    assert fit.iterations.tolist() == [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert torch.equal(fit.converged, fit.status == 0)
    torch.testing.assert_close(fit.residual_norm,
        torch.linalg.vector_norm((rhs-matrices@fit.solution).double(), dim=-2), equal_nan=True)
    # Singular, consistent problem: successful solution without uniqueness claim.
    singular = s.linalg.gmres(torch.diag(t([1., 0., 0.])), t([2., 0., 0.]))
    assert singular.converged
    torch.testing.assert_close(singular.solution, t([2., 0., 0.]))
    # max_iter is total Arnoldi steps, not restart cycles; restart1 may stagnate.
    rotation = t([[0., -1.], [1., 0.]])
    exhausted = s.linalg.gmres(rotation, t([1., 0.]), restart=1, max_iter=3, rtol=1e-12)
    assert exhausted.status == 1 and exhausted.iterations == 3
    limited = s.linalg.gmres(identity, t([1., 2., 3.]), max_iter=0)
    assert limited.status == 1 and not limited.iterations
    for shape in ((0, 0), (0, 3, 3)):
        result = s.linalg.gmres(torch.empty(shape, device=device, dtype=dtype),
                                torch.empty((*shape[:-1], 1), device=device, dtype=dtype))
        assert result.converged.all() and result.solution.shape == (*shape[:-1], 1)
    with pytest.raises(RuntimeError, match='inference-only'):
        s.linalg.gmres(identity.requires_grad_(), t([1.,2.,3.]))
    with torch.no_grad():
        assert not s.linalg.gmres(identity, t([1.,2.,3.])).solution.requires_grad
    for options in ({'restart':0}, {'restart':-1}, {'rtol':-1}, {'atol':float('nan')}, {'max_iter':-1}):
        with pytest.raises(RuntimeError, match='nonnegative|positive'):
            s.linalg.gmres(identity.detach(), t([1.,2.,3.]), **options)
    for options in ({'restart':True}, {'restart':2.5}, {'max_iter':True}):
        with pytest.raises(TypeError, match='integers'):
            s.linalg.gmres(identity.detach(), t([1.,2.,3.]), **options)
    with pytest.raises(RuntimeError, match='square'):
        s.linalg.gmres(torch.ones(2,3,device=device,dtype=dtype), t([1.,2.]))
    with pytest.raises(RuntimeError, match='sample count'):
        s.linalg.gmres(identity.detach(), t([1.,2.]))
    with pytest.raises(RuntimeError, match='initial'):
        s.linalg.gmres(identity.detach(), t([1.,2.,3.]), initial=t([1.,2.]))


@pytest.mark.compiler
def test_gmres_compile_export(device):
    class Solve(torch.nn.Module):
        def forward(self, x, keys, rhs):
            k = s.lazy(x, keys, gamma=0.2, chunk_size=3, metric='rbf')
            a = 0.05 * k + 1.0 * tp.eye_like(0.05 * k)
            return s.linalg.gmres(a, rhs, restart=3, rtol=1e-9, max_iter=60)
    torch.manual_seed(14)
    def inputs(n):
        return tuple(torch.randn(n, d, device=device, dtype=torch.float64) for d in (3,3,2))
    module, args = Solve(), inputs(7)
    compiled = torch.compile(module, backend='inductor', fullgraph=True, dynamic=True)
    for n in (7,10):
        values = inputs(n)
        expected = module(*values)
        assert expected.converged.all()
        torch.testing.assert_close(compiled(*values), expected)
    n = torch.export.Dim('samples', min=3, max=16)
    exported = torch.export.export(module, args, dynamic_shapes=({0:n},{0:n},{0:n}))
    buffer = io.BytesIO()
    torch.export.save(exported, buffer)
    buffer.seek(0)
    restored = torch.export.load(buffer).module()
    values = inputs(11)
    torch.testing.assert_close(restored(*values), module(*values))
