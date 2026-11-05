"""Rectangular LSQR: independent dense oracles, diagnostics and deployment."""
import io

import pytest
import torch
import torchpairwise as tp
import torchpairwise as s


def test_lsqr_rectangular_oracles_and_streamed_map(device):
    torch.manual_seed(901)
    for rows, columns, deficient in ((9, 5, False), (4, 7, False), (8, 5, True)):
        a = torch.randn(rows, columns, dtype=torch.float64, device=device)
        if deficient:
            a[:, -1] = a[:, 0] + a[:, 1]
        b = torch.randn(rows, 2, dtype=a.dtype, device=device)
        fit = s.linalg.lsqr(a, b, rtol=1e-11, max_iter=100)
        expected = torch.linalg.pinv(a) @ b
        assert fit.converged.all(), fit
        torch.testing.assert_close(fit.solution, expected, rtol=1e-7, atol=1e-8)
        torch.testing.assert_close(fit.residual_norm, torch.linalg.vector_norm(b-a@fit.solution, dim=0))
        torch.testing.assert_close(fit.normal_residual_norm,
                                   torch.linalg.vector_norm(a.T @ (b-a@fit.solution), dim=0),
                                   rtol=1e-3, atol=1e-13)
    # A composed rectangular, batched generated map, broadcast RHS/initial and
    # nonzero damping. Compare with an explicitly augmented dense least square.
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(2, 11, 3, dtype=dtype, device=device)
        y = torch.randn(1, 5, 3, dtype=dtype, device=device)
        b = torch.randn(11, 2, dtype=dtype, device=device)
        initial = torch.randn(5, 2, dtype=dtype, device=device)
        source = s.lazy(x, y, gamma=0.3, chunk_size=4, metric='rbf')
        operator = source.centered()
        matrix = tp.rbf_kernel(x, y, gamma=.3)
        matrix = matrix - matrix.mean(-1, keepdim=True) - matrix.mean(-2, keepdim=True) + matrix.mean((-1, -2), keepdim=True)
        identity = torch.eye(5, dtype=dtype, device=device).expand(2, 5, 5)
        augmented = torch.cat((matrix, .3*identity), dim=-2).double()
        augmented_b = torch.cat((b, torch.zeros(5, 2, dtype=dtype, device=device))).expand(2, 16, 2).double()
        expected = torch.linalg.lstsq(augmented, augmented_b).solution.to(dtype)
        fit = s.linalg.lsqr(operator, b, damp=.3, initial=initial,
                             rtol=2e-5 if dtype == torch.float32 else 1e-10, max_iter=100)
        assert fit.converged.all(), fit
        tolerance = 2e-3 if dtype == torch.float32 else 1e-8
        torch.testing.assert_close(fit.solution, expected, rtol=tolerance, atol=tolerance)
        assert fit.solution.shape == (2, 5, 2) and fit.status.shape == (2, 2)
        assert fit.residual_norm.dtype == fit.normal_residual_norm.dtype == torch.float64
        # Damping penalizes the actual x, including a nonzero initial guess.
        residual = b - operator.matmul(fit.solution)
        normal = operator.T.matmul(residual).double() - .09*fit.solution.double()
        # The public transpose and the prepared solver plan may accumulate
        # tiles in a different order. Bound their float32 action roundoff.
        roundoff = 8*torch.finfo(dtype).eps * (
            torch.linalg.matrix_norm(matrix.double()) * torch.linalg.matrix_norm(residual.double())
            + .09*torch.linalg.matrix_norm(fit.solution.double()))
        error = (fit.normal_residual_norm - torch.linalg.vector_norm(normal, dim=-2)).abs()
        assert (error <= roundoff.unsqueeze(-1) + 1e-12).all()


def test_lsqr_diagnostics_empty_initial_and_inference(device):
    def t(value):
        return torch.tensor(value, dtype=torch.float64, device=device)
    identity = torch.eye(3, dtype=torch.float64, device=device)
    a = torch.stack((identity, identity*0, torch.diag(t([1., 2., 4.]))))
    b = t([[0., 1., float('nan')], [0., 2., 1.], [0., 3., 1.]])
    fit = s.linalg.lsqr(a, b, max_iter=1, rtol=1e-12)
    assert fit.status.tolist() == [[0, 0, 3], [0, 0, 3], [0, 1, 3]]
    assert fit.iterations.tolist() == [[0, 1, 0], [0, 0, 0], [0, 1, 0]]
    torch.testing.assert_close(fit.converged, fit.status == 0)
    # Zero A and nonzero b is stationary but has a nonzero data residual.
    assert fit.residual_norm[1, 1] > 0 and fit.normal_residual_norm[1, 1] == 0
    for rows, columns, damp in ((0, 3, 0.), (0, 3, .2), (3, 0, 0.), (0, 0, .2)):
        initial = torch.ones(columns, dtype=torch.float64, device=device)
        result = s.linalg.lsqr(torch.empty(rows, columns, dtype=initial.dtype, device=device),
                               torch.ones(rows, dtype=initial.dtype, device=device), damp=damp, initial=initial)
        assert result.solution.shape == (columns,) and result.converged
        expected = initial if damp == 0 else initial*0
        torch.testing.assert_close(result.solution, expected)
    empty = s.linalg.lsqr(torch.empty(0, 3, 2, dtype=b.dtype, device=device), torch.ones(3, 1, dtype=b.dtype, device=device))
    assert empty.solution.shape == (0, 2, 1) and empty.converged.all()
    a = t([[1., 2.], [3., 1.], [2., -1.]])
    b, initial = t([1., 2., 4.]), t([.4, -.2])
    result = s.linalg.lsqr(a, b, damp=.2, initial=initial, max_iter=0)
    torch.testing.assert_close(result.solution, initial)
    torch.testing.assert_close(result.normal_residual_norm,
                               torch.linalg.vector_norm(a.T @ (b-a@initial)-.04*initial))
    assert result.status == s.linalg.LSQRStatus.ITERATION_LIMIT and result.iterations == 0
    zero = s.linalg.lsqr(a, torch.zeros_like(b), damp=.2, initial=initial)
    assert zero.converged and not zero.solution.any()
    # An initial null-space component is retained for an undamped wide system.
    wide = t([[1., 0., 0.], [0., 1., 0.]])
    kept = s.linalg.lsqr(wide, t([1., 2.]), initial=t([0., 0., 7.]), rtol=1e-12)
    torch.testing.assert_close(kept.solution, t([1., 2., 7.]))
    grad = a.detach().requires_grad_()
    with pytest.raises(RuntimeError, match='inference-only'):
        s.linalg.lsqr(grad, b)
    with torch.no_grad():
        assert not s.linalg.lsqr(grad, b).solution.requires_grad
    for options in ({'damp': -1}, {'rtol': float('inf')}, {'atol': -1}, {'max_iter': -1}):
        with pytest.raises(RuntimeError, match='nonnegative'):
            s.linalg.lsqr(a, b, **options)
    with pytest.raises(TypeError, match='integer'):
        s.linalg.lsqr(a, b, max_iter=True)
    with pytest.raises(RuntimeError, match='sample count'):
        s.linalg.lsqr(a, b[:2])
    with pytest.raises(RuntimeError, match='initial'):
        s.linalg.lsqr(a, b, initial=torch.ones_like(b))


@pytest.mark.compiler
def test_lsqr_compile_export_rectangular(device):
    class LeastSquares(torch.nn.Module):
        def forward(self, x, landmarks, b):
            a = s.lazy(x, landmarks, gamma=0.3, chunk_size=3, metric='rbf')
            return s.linalg.lsqr(a, b, damp=.2, rtol=1e-8, max_iter=80)
    module = LeastSquares()
    dtype = torch.float64
    def inputs(n, m):
        return (torch.randn(n, 3, device=device, dtype=dtype),
                torch.randn(m, 3, device=device, dtype=dtype),
                torch.randn(n, 2, device=device, dtype=dtype))
    args = inputs(8, 5)
    compiled = torch.compile(module, backend='inductor', dynamic=True, fullgraph=True)
    for n, m in ((8, 5), (11, 7)):
        values = inputs(n, m)
        expected = module(*values)
        assert expected.converged.all()
        torch.testing.assert_close(compiled(*values), expected)
    n, m = torch.export.Dim('rows', min=4, max=16), torch.export.Dim('columns', min=3, max=10)
    exported = torch.export.export(module, args, dynamic_shapes=({0: n}, {0: m}, {0: n}))
    buffer = io.BytesIO()
    torch.export.save(exported, buffer)
    buffer.seek(0)
    restored = torch.export.load(buffer).module()
    for values in (args, inputs(12, 6)):
        torch.testing.assert_close(restored(*values), module(*values))
