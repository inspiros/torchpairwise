"""Implicit solution gradients: adjoints, streaming replay and opaque deployment."""
import io

import pytest
import torch
import torchpairwise as s
import torchpairwise as tp


def test_implicit_solve_dense_and_streamed_vjps(device):
    torch.manual_seed(33)
    def t(value):
        return torch.tensor(value, dtype=torch.float64, device=device, requires_grad=True)
    for method, data in (('cg', [[4., 1.], [1., 3.]]),
                         ('minres', [[1., 2.], [2., -1.]]),
                         ('gmres', [[2., 3.], [0., 1.]])):
        matrix = t(data).repeat(2, 1, 1).detach().requires_grad_()
        b = t([1., 2.])  # broadcast vector RHS; tests adjoint block normalization
        options = {'restart': 2} if method == 'gmres' else {}
        result = s.linalg.solve(matrix, b, method=method, differentiable=True,
                                rtol=1e-11, max_iter=40, **options)
        expected = torch.linalg.solve(matrix, b)
        torch.testing.assert_close(result.solution, expected)
        torch.testing.assert_close(torch.autograd.grad(result.solution.square().sum(), (matrix, b)),
                                   torch.autograd.grad(expected.square().sum(), (matrix, b)))
        assert result.converged.all() and not result.residual_norm.requires_grad
    x = torch.randn(2, 7, 3, dtype=torch.float64, device=device, requires_grad=True)
    weight = t(1.2)
    shift = t(.7)
    factor = t(.9)
    b = torch.randn(7, 2, dtype=x.dtype, device=device, requires_grad=True)
    # A scalar-expression factor and linear-map factor have separate packet slots.
    source = s.lazy(x, gamma=0.3, chunk_size=3, metric='rbf') * weight
    a = factor * source.centered() + shift * tp.eye_like(factor * source.centered())
    fit = s.linalg.solve(a, b, method='cg', differentiable=True, rtol=1e-10, max_iter=80)
    dense = tp.rbf_kernel(x, gamma=.3) * weight
    dense = dense - dense.mean(-1, keepdim=True) - dense.mean(-2, keepdim=True) + dense.mean((-1, -2), keepdim=True)
    matrix = dense * factor + shift * torch.eye(7, device=device, dtype=x.dtype)
    oracle = torch.linalg.solve(matrix, b)
    torch.testing.assert_close(fit.solution, oracle, rtol=1e-8, atol=1e-8)
    variables = (x, weight, factor, shift, b)
    torch.testing.assert_close(torch.autograd.grad(fit.solution.square().sum(), variables),
                               torch.autograd.grad(oracle.square().sum(), variables), rtol=1e-7, atol=1e-7)
    # Native dictionary tensor option, distinct from expression/linear factors.
    w = t([.7, 1.1, 1.6])
    distances = s.lazy(x, metric='cosine_distances', w=w, chunk_size=3)
    linear = distances
    a = linear.T @ linear + 2.0 * tp.eye_like(linear.T @ linear)
    fit = s.linalg.solve(a, b, method='cg', differentiable=True, rtol=1e-10)
    d = tp.cosine_distances(x, w=w)
    dense = d.transpose(-1, -2) @ d + 2. * torch.eye(7, device=device, dtype=x.dtype)
    oracle = torch.linalg.solve(dense, b)
    torch.testing.assert_close(torch.autograd.grad(fit.solution.sum(), (x, w, b)),
                               torch.autograd.grad(oracle.sum(), (x, w, b)), rtol=1e-7, atol=1e-7)
    # Forward may converge for a special RHS while the adjoint exhausts its budget.
    matrix = t([[2., 0.], [0., 3.]])
    b = t([1., 0.])
    fit = s.linalg.solve(matrix, b, method='cg', differentiable=True, max_iter=1, rtol=1e-12)
    with pytest.raises(RuntimeError, match='converged'):
        torch.autograd.grad(fit.solution.sum(), b)
    with pytest.raises(RuntimeError, match='converged'):
        s.linalg.solve(matrix, b, method='cg', differentiable=True, max_iter=0)
    with pytest.raises(RuntimeError, match='detach'):
        s.linalg.solve(matrix, b, method='cg', differentiable=True, initial=b)
    fit = s.linalg.solve(matrix, b, method='cg', differentiable=True)
    with pytest.raises(RuntimeError, match='Higher derivatives'):
        torch.autograd.grad(fit.solution.sum(), b, create_graph=True)
    fit = s.linalg.solve(matrix, b, method='cg', differentiable=True)
    with torch.no_grad():
        matrix.add_(.1)
    with pytest.raises(RuntimeError, match='modified by an inplace'):
        fit.solution.sum().backward()


@pytest.mark.compiler
def test_implicit_solve_compile_export_gradients(device):
    class Model(torch.nn.Module):
        def forward(self, x, shift, b):
            a = s.lazy(x * (1 + shift), gamma=0.2, chunk_size=3, metric='rbf') + shift * tp.eye_like(s.lazy(x * (1 + shift), gamma=0.2, chunk_size=3, metric='rbf'))
            return s.linalg.solve(a, b.tanh(), method='cg', differentiable=True, rtol=1e-10, max_iter=80, preconditioner=s.diag_embed(x.new_ones(x.shape[0]) / (1 + shift)))
    torch.manual_seed(67)
    model = Model()
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    def inputs(n):
        return (torch.randn(n, 3, device=device, dtype=torch.float64, requires_grad=True),
                torch.tensor(.8, device=device, dtype=torch.float64, requires_grad=True),
                torch.randn(n, 2, device=device, dtype=torch.float64, requires_grad=True))
    for n in (5, 8):
        values = inputs(n)
        oracle = model(*values)
        actual = compiled(*values)
        torch.testing.assert_close(actual, oracle)
        torch.testing.assert_close(torch.autograd.grad(actual.solution.square().sum(), values),
                                   torch.autograd.grad(oracle.solution.square().sum(), values))
    n = torch.export.Dim('samples', min=3, max=20)
    exported = torch.export.export(model, inputs(7), dynamic_shapes=({0: n}, {}, {0: n}))
    buffer = io.BytesIO()
    torch.export.save(exported, buffer)
    buffer.seek(0)
    restored = torch.export.load(buffer).module()
    values = inputs(9)
    actual, oracle = restored(*values), model(*values)
    assert isinstance(actual, s.linalg.SolveResult)
    torch.testing.assert_close(actual, oracle)
    torch.testing.assert_close(torch.autograd.grad(actual.solution.sum(), values),
                               torch.autograd.grad(oracle.solution.sum(), values))


def test_implicit_pcg_algorithmic_and_shared_operands(device):
    dtype = torch.float64
    diagonal = torch.tensor([.01, .1, 1., 10.], device=device, dtype=dtype, requires_grad=True)
    inverse = diagonal.detach().reciprocal().requires_grad_()
    rhs = torch.randn(2, 4, 2, device=device, dtype=dtype, requires_grad=True)
    a = s.diag_embed(diagonal)
    fit = s.linalg.solve(a, rhs, method='cg', differentiable=True, preconditioner=s.diag_embed(inverse), rtol=1e-12, max_iter=1)
    expected = rhs / diagonal[:, None]
    assert fit.converged.all() and (fit.iterations == 1).all()
    torch.testing.assert_close(fit.solution, expected)
    actual = torch.autograd.grad(fit.solution.square().sum(), (diagonal, rhs, inverse), allow_unused=True)
    reference = torch.autograd.grad(expected.square().sum(), (diagonal, rhs))
    torch.testing.assert_close(actual[:2], reference)
    assert actual[2] is None
    # A and M share an operand. Only the mathematical system contributes.
    rhs = rhs.detach().requires_grad_()
    fit = s.linalg.solve(a, rhs, method='cg', differentiable=True, preconditioner=s.diag_embed(diagonal.reciprocal()), rtol=1e-12, max_iter=1)
    expected = rhs / diagonal[:, None]
    torch.testing.assert_close(torch.autograd.grad(fit.solution.sum(), (diagonal, rhs)),
                               torch.autograd.grad(expected.sum(), (diagonal, rhs)))
    with pytest.raises(RuntimeError, match='converged'):
        s.linalg.solve(a, rhs, method='cg', differentiable=True, preconditioner=s.diag_embed(-inverse))


def test_implicit_damped_lstsq_gradients_and_failures(device):
    torch.manual_seed(71)
    def tensor(*shape):
        return torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True)
    # Tall, wide and rank-deficient maps: damping supplies uniqueness.
    for n, m in ((7, 4), (4, 7)):
        base, rhs = tensor(1, n, m-1), tensor(2, n, 2)
        matrix = torch.cat((base, base[..., :1]), -1)
        fit = s.linalg.lstsq(matrix, rhs, damp=.7, differentiable=True, rtol=1e-10, max_iter=100)
        normal = matrix.transpose(-1,-2) @ matrix + .7**2 * torch.eye(m, device=device, dtype=matrix.dtype)
        expected = torch.linalg.solve(normal, matrix.transpose(-1,-2) @ rhs)
        assert fit.converged.all() and not fit.normal_residual_norm.requires_grad
        torch.testing.assert_close(fit.solution, expected, rtol=1e-8, atol=1e-9)
        torch.testing.assert_close(torch.autograd.grad(fit.solution.square().sum(), (base, rhs)),
                                   torch.autograd.grad(expected.square().sum(), (base, rhs)), rtol=1e-7, atol=1e-8)
    x, y, rhs, factor = tensor(6, 2), tensor(4, 2), tensor(6), tensor()
    source = s.lazy(x, y, gamma=0.3, chunk_size=3, metric='rbf') * factor
    fit = s.linalg.lstsq(source, rhs, damp=.5, differentiable=True, rtol=1e-10)
    matrix = tp.rbf_kernel(x, y, gamma=.3) * factor
    expected = torch.linalg.solve(matrix.T @ matrix + .25 * torch.eye(4, device=device, dtype=x.dtype), matrix.T @ rhs)
    torch.testing.assert_close(torch.autograd.grad(fit.solution.square().sum(), (x,y,rhs,factor)),
                               torch.autograd.grad(expected.square().sum(), (x,y,rhs,factor)), rtol=1e-7, atol=1e-8)
    for damp in (0., -.1, float('inf'), 1e-250):
        with pytest.raises(RuntimeError, match='damp'):
            s.linalg.lstsq(matrix.detach(), rhs, damp=damp, differentiable=True)
    with pytest.raises(RuntimeError, match='converged'):
        s.linalg.lstsq(matrix.detach(), rhs, damp=.5, differentiable=True, max_iter=0)
    with pytest.raises(RuntimeError, match='detach'):
        s.linalg.lstsq(matrix.detach(), rhs, damp=.5, differentiable=True, initial=y[:,0])
    # Adjoint convergence is checked independently from the LSQR forward solve.
    a = torch.diag(torch.tensor([1.,2.],device=device,dtype=x.dtype))
    b = torch.tensor([1.,0.],device=device,dtype=x.dtype,requires_grad=True)
    fit = s.linalg.lstsq(a, b, damp=.5, differentiable=True, rtol=1e-12, max_iter=1)
    with pytest.raises(RuntimeError, match='converged'):
        torch.autograd.grad(fit.solution.sum(), b)


@pytest.mark.compiler
def test_implicit_lstsq_compile_export(device):
    class Model(torch.nn.Module):
        def forward(self, x, y, b):
            return s.linalg.lstsq(s.lazy(x.sin(), y.cos(), gamma=0.3, chunk_size=3, metric='rbf'), b.tanh(), damp=0.5, differentiable=True, rtol=1e-10, max_iter=100)
    def compare_fit(actual, expected):
        # Compiled sin/cos/tanh can differ by one last bit. A stopping boundary
        # may then take another iteration; solution/gradient tolerances and the
        # convergence/status contract still have to agree.
        torch.testing.assert_close((actual.solution, actual.residual_norm,
                                    actual.normal_residual_norm, actual.converged, actual.status),
                                   (expected.solution, expected.residual_norm,
                                    expected.normal_residual_norm, expected.converged, expected.status))
        assert actual.iterations.shape == expected.iterations.shape
        assert ((actual.iterations >= 0) & (actual.iterations <= 100)).all()
    model=Model()
    compiled=torch.compile(model,fullgraph=True,dynamic=True)
    def inputs(n,m):
        return tuple(torch.randn(*shape,device=device,dtype=torch.float64,requires_grad=True)
                     for shape in ((n,2),(m,2),(n,2)))
    for n,m in ((7,4),(5,8)):
        torch.manual_seed(2)  # Exercises a CPU wide-system stopping boundary.
        values=inputs(n,m)
        actual,expected=compiled(*values),model(*values)
        compare_fit(actual,expected)
        torch.testing.assert_close(torch.autograd.grad(actual.solution.square().sum(),values),
                                   torch.autograd.grad(expected.solution.square().sum(),values))
    n,m=(torch.export.Dim(name,min=3,max=20) for name in ('rows','columns'))
    exported=torch.export.export(model,inputs(7,4),dynamic_shapes=({0:n},{0:m},{0:n}))
    buffer=io.BytesIO();torch.export.save(exported,buffer);buffer.seek(0)
    restored=torch.export.load(buffer).module();values=inputs(6,9)
    actual,expected=restored(*values),model(*values)
    assert isinstance(actual,s.linalg.LSQRResult)
    compare_fit(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual.solution.sum(),values),
                               torch.autograd.grad(expected.solution.sum(),values))
