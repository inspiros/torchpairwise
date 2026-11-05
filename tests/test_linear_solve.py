"""Unified solver selection, preserved diagnostics and native deployment."""
import io

import pytest
import torch
import torchpairwise as tp
import torchpairwise as s


def test_solve_selection_and_least_squares_contract(device):
    def t(value):
        return torch.tensor(value, device=device, dtype=torch.float64)
    rhs = t([[1., 0.], [2., 0.]])
    for method, matrix, options in (
        ('cg', t([[4., 1.], [1., 3.]]), {'preconditioner': s.linalg.jacobi(t([4., 3.]))}),
        ('minres', t([[0., 1.], [1., 0.]]), {}),
        ('gmres', t([[2., 3.], [0., 1.]]), {'restart': 2}),
    ):
        result = s.linalg.solve(matrix, rhs, method=method, rtol=1e-10, max_iter=30, **options)
        direct = getattr(s.linalg, method)(matrix, rhs, rtol=1e-10, max_iter=30, **options)
        assert isinstance(result, s.linalg.SolveResult)
        torch.testing.assert_close(tuple(result), tuple(direct))
        torch.testing.assert_close(result.solution, torch.linalg.solve(matrix, rhs), rtol=1e-9, atol=1e-9)
        assert result.converged.all() and result.iterations[1] == 0
    limited = s.linalg.solve(t([[1., 0.], [0., 1.]]), rhs[:, 0], method='cg', max_iter=0)
    assert limited.status == s.linalg.CGStatus.ITERATION_LIMIT and limited.iterations == 0
    failed = s.linalg.solve(-t([[1., 0.], [0., 1.]]), rhs[:, 0], method='cg')
    assert failed.status == s.linalg.CGStatus.NONPOSITIVE_CURVATURE
    for method, options, message in (
        ('auto', {}, 'method'), ('CG', {}, 'method'),
        ('cg', {'restart': 20}, 'restart'),
        ('minres', {'preconditioner': t([[1., 0.], [0., 1.]])}, 'preconditioner'),
    ):
        with pytest.raises(ValueError, match=message):
            s.linalg.solve(t([[1., 0.], [0., 1.]]), rhs, method=method, **options)
    with pytest.raises(TypeError, match='method'):
        s.linalg.solve(t([[1., 0.], [0., 1.]]), rhs)
    # Wide rank-deficient system: the default zero start targets A^dagger b.
    matrix = t([[1., 0., 0., 0.], [0., 2., 0., 0.], [1., 0., 0., 0.]])
    values = t([[1., 0.], [2., 1.], [3., 0.]])
    result = s.linalg.lstsq(matrix, values, rtol=1e-11, max_iter=30)
    assert isinstance(result, s.linalg.LSQRResult) and result.converged.all()
    torch.testing.assert_close(result, s.linalg.lsqr(matrix, values, rtol=1e-11, max_iter=30))
    torch.testing.assert_close(result.solution, torch.linalg.pinv(matrix) @ values, rtol=1e-9, atol=1e-9)
    assert result.residual_norm[0] > 1. and result.normal_residual_norm.max() < 1e-10
    initial = torch.zeros_like(result.solution)
    initial[3] = 7.
    warm = s.linalg.lstsq(matrix, values, initial=initial, rtol=1e-11)
    torch.testing.assert_close(warm.solution[:3], result.solution[:3])
    torch.testing.assert_close(warm.solution[3], initial[3])
    damped = s.linalg.lstsq(matrix, values, damp=.4, initial=initial, rtol=1e-11)
    oracle = torch.linalg.solve(matrix.T @ matrix + .4**2 * torch.eye(4, device=device, dtype=matrix.dtype), matrix.T @ values)
    assert damped.converged.all()
    torch.testing.assert_close(damped.solution, oracle, rtol=1e-9, atol=1e-9)
    with pytest.raises(ValueError, match='method'):
        s.linalg.lstsq(matrix, values, method='svd')


@pytest.mark.compiler
def test_solve_dispatch_compile_and_export(device):
    class Model(torch.nn.Module):
        def forward(self, x, y, b):
            K = s.lazy(x, gamma=0.3, chunk_size=3, metric='rbf')
            A = K + 0.5 * tp.eye_like(K)
            solved = s.linalg.solve(A, b, method='cg',
                                    preconditioner=s.linalg.jacobi(x.new_full((x.shape[0],), 1.5)),
                                    rtol=1e-8, max_iter=40)
            fitted = s.linalg.lstsq(s.lazy(x, y, gamma=0.3, chunk_size=3, metric='rbf'), b, damp=0.2, rtol=1e-08, max_iter=40)
            return solved, fitted
    torch.manual_seed(124)
    module = Model()
    def inputs(n):
        return (torch.randn(n, 3, device=device, dtype=torch.float64),
                torch.randn(4, 3, device=device, dtype=torch.float64),
                torch.randn(n, 2, device=device, dtype=torch.float64))
    compiled = torch.compile(module, fullgraph=True, dynamic=True)
    for n in (7, 10):
        values = inputs(n)
        torch.testing.assert_close(compiled(*values), module(*values))
    values = inputs(7)
    n = torch.export.Dim('samples', min=4, max=16)
    exported = torch.export.export(module, values, dynamic_shapes=({0: n}, {}, {0: n}))
    buffer = io.BytesIO()
    torch.export.save(exported, buffer)
    buffer.seek(0)
    restored = torch.export.load(buffer).module()
    values = inputs(11)
    actual = restored(*values)
    assert isinstance(actual[0], s.linalg.SolveResult) and isinstance(actual[1], s.linalg.LSQRResult)
    torch.testing.assert_close(actual, module(*values))
