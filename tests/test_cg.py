"""Native CG: dense oracles, independent diagnostics and serialized solves."""
import io

import pytest
import torch
import torchpairwise as tp
import torchpairwise as s


def test_cg_streamed_graph_and_broadcast_rhs(device):
    torch.manual_seed(83)
    dtype = torch.float64
    x = torch.randn(2, 9, 3, device=device, dtype=dtype)
    y = torch.randn(1, 6, 3, device=device, dtype=dtype)
    v = torch.randn(9, 3, device=device, dtype=dtype)
    k = s.lazy(x, y, gamma=0.2, chunk_size=4, metric='rbf')
    a = (k + 0.25).square().centered()
    operator = a @ a.T + 0.4 * tp.eye_like(a @ a.T)
    dense = (tp.rbf_kernel(x, y, gamma=.2) + .25).square()
    dense = dense - dense.mean(-1, keepdim=True) - dense.mean(-2, keepdim=True) + dense.mean((-1, -2), keepdim=True)
    matrix = dense @ dense.transpose(-1, -2) + .4 * torch.eye(9, device=device, dtype=dtype)
    result = s.linalg.cg(operator, v, rtol=1e-10, max_iter=60)
    torch.testing.assert_close(result.solution, torch.linalg.solve(matrix, v), rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(result.residual_norm,
        torch.linalg.vector_norm(v - operator.matmul(result.solution), dim=-2), rtol=1e-4, atol=1e-13)
    torch.testing.assert_close(result.residual_norm, torch.linalg.vector_norm(v - matrix @ result.solution, dim=-2), rtol=1e-3, atol=1e-13)
    assert result.converged.all() and (result.status == s.linalg.CGStatus.CONVERGED).all()
    assert result.solution.shape == (2, 9, 3) and result.iterations.shape == (2, 3)
    assert result.residual_norm.dtype == torch.float64
    # The same graph with distinct source dimensions and a broadcast initial.
    initial = torch.linalg.solve(matrix, v)
    result = s.linalg.cg(operator, v, initial=initial, rtol=1e-9, max_iter=0)
    assert result.converged.all() and not result.iterations.any()
    for dtype in (torch.float32, torch.float64):
        b = torch.arange(1, 5, device=device, dtype=dtype)
        result = s.linalg.cg(2 * s.as_lazy(torch.eye(4, device=device, dtype=dtype)), b)
        torch.testing.assert_close(result.solution, b / 2)
        assert result.converged.shape == () and result.converged


def test_cg_diagnostics_boundaries_and_inference(device):
    dtype = torch.float64
    def t(value):
        return torch.tensor(value, device=device, dtype=dtype)
    matrices = torch.stack([torch.eye(3, device=device, dtype=dtype), -torch.eye(3, device=device, dtype=dtype), torch.diag(t([1., 2., 4.]))])
    b = t([[0., 1., float('nan')], [0., 2., 1.], [0., 3., 1.]])
    result = s.linalg.cg(matrices, b, max_iter=1, rtol=1e-12)
    assert result.status.tolist() == [[0, 0, 3], [0, 2, 3], [0, 1, 3]]
    assert result.iterations.tolist() == [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert torch.equal(result.converged, result.status == 0)
    torch.testing.assert_close(result.residual_norm, torch.linalg.vector_norm(b - matrices @ result.solution, dim=-2), equal_nan=True)
    exhausted = s.linalg.cg(torch.eye(3, device=device, dtype=dtype), t([1., 2., 3.]), max_iter=0)
    assert exhausted.status == 1 and exhausted.iterations == 0
    for shape in ((0, 0), (0, 3, 3)):
        matrix = torch.empty(shape, device=device, dtype=dtype)
        rhs = torch.empty((*shape[:-1], 1), device=device, dtype=dtype)
        result = s.linalg.cg(matrix, rhs)
        assert result.converged.all() and result.solution.shape == rhs.shape
    grad_matrix = torch.eye(3, device=device, dtype=dtype, requires_grad=True)
    with pytest.raises(RuntimeError, match='inference-only'):
        s.linalg.cg(grad_matrix, t([1., 2., 3.]))
    with torch.no_grad():
        assert not s.linalg.cg(grad_matrix, t([1., 2., 3.])).solution.requires_grad
    for options in ({'rtol': -1}, {'atol': float('nan')}, {'max_iter': -1}):
        with pytest.raises(RuntimeError, match='nonnegative'):
            s.linalg.cg(matrices, t([1., 2., 3.]), **options)
    with pytest.raises(TypeError, match='integer'):
        s.linalg.cg(matrices, t([1., 2., 3.]), max_iter=True)
    with pytest.raises(RuntimeError, match='square'):
        s.linalg.cg(torch.ones(2, 3, device=device, dtype=dtype), t([1., 2.]))
    with pytest.raises(RuntimeError, match='initial'):
        s.linalg.cg(matrices, t([1., 2., 3.]), initial=t([1., 2.]))


@pytest.mark.compiler
def test_cg_compile_and_export(device):
    class Solve(torch.nn.Module):
        def forward(self, x, b):
            a = s.lazy(x, gamma=0.3, chunk_size=3, metric='rbf') + 0.5 * tp.eye_like(s.lazy(x, gamma=0.3, chunk_size=3, metric='rbf'))
            return s.linalg.cg(a, b, rtol=1e-8, max_iter=40)
    module = Solve()
    x = torch.randn(7, 3, device=device, dtype=torch.float64)
    b = torch.randn(7, 2, device=device, dtype=x.dtype)
    expected = module(x, b)
    compiled = torch.compile(module, backend='inductor', fullgraph=True, dynamic=True)
    for n in (7, 10):
        xx, bb = torch.randn(n, 3, device=device, dtype=x.dtype), torch.randn(n, 2, device=device, dtype=x.dtype)
        torch.testing.assert_close(compiled(xx, bb), module(xx, bb))
    n = torch.export.Dim('samples', min=4, max=16)
    exported = torch.export.export(module, (x, b), dynamic_shapes=({0: n}, {0: n}))
    buf = io.BytesIO()
    torch.export.save(exported, buf)
    buf.seek(0)
    restored = torch.export.load(buf).module()
    torch.testing.assert_close(restored(x, b), expected)
    xx, bb = torch.randn(11, 3, device=device, dtype=x.dtype), torch.randn(11, 2, device=device, dtype=x.dtype)
    torch.testing.assert_close(restored(xx, bb), module(xx, bb))
