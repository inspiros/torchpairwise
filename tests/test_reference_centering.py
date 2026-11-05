"""Training-based centering: held-out actions, adjoints, gradients and deployment."""
import io

import pytest
import torch
import torchpairwise as s
import torchpairwise as tp


def test_reference_centered_actions_and_gradients(device):
    torch.manual_seed(912)
    def tensor(*shape):
        return torch.randn(shape, dtype=torch.float64, device=device, requires_grad=True)
    x, y, mean, rhs, adjoint = tensor(3, 2), tensor(5, 2), tensor(2, 5), tensor(5, 2), tensor(3, 2)
    source = s.lazy(x, y, gamma=0.4, chunk_size=2, metric='rbf')
    operator = source.centered(mean)
    matrix = tp.rbf_kernel(x, y, gamma=.4)
    centered = matrix - matrix.mean(-1, keepdim=True) - mean.unsqueeze(-2) + mean.mean(-1, keepdim=True).unsqueeze(-2)
    assert operator.shape == (2, 3, 5)
    actual = [operator @ rhs, operator.T @ adjoint]
    expected = [centered @ rhs, centered.transpose(-1, -2) @ adjoint]
    torch.testing.assert_close(actual, expected)
    variables = (x, y, mean, rhs, adjoint)
    torch.testing.assert_close(
        torch.autograd.grad(sum(v.square().sum() for v in actual), variables),
        torch.autograd.grad(sum(v.square().sum() for v in expected), variables),
        rtol=1e-8, atol=1e-8)
    with torch.no_grad():
        # A fitted mean is independent of the query batch; values remain referenced.
        mean.add_(.2 * torch.arange(5, device=device))
        solo = s.lazy(x[:1], y, gamma=0.4, metric='rbf').centered(mean) @ rhs
        torch.testing.assert_close(solo, (operator @ rhs)[:, :1])
        empty = s.lazy(x[:0], y, gamma=0.4, metric='rbf').centered(mean)
        torch.testing.assert_close(empty @ rhs, torch.empty(2, 0, 2, device=device, dtype=x.dtype))
        torch.testing.assert_close(empty.T @ adjoint[:0], torch.zeros(2, 5, 2, device=device, dtype=x.dtype))
        train = s.lazy(y, gamma=0.4, chunk_size=2, metric='rbf')
        mu = train.reduce('mean', dim=-2)
        torch.testing.assert_close(train.centered(mu) @ rhs, train.centered() @ rhs)
    for invalid in (mean[:, :4], mean.to(torch.float32), mean.unsqueeze(-1)):
        with pytest.raises((ValueError, RuntimeError), match='[Rr]eference'):
            source.centered(invalid)
    with pytest.raises(RuntimeError, match='nonempty'):
        s.as_lazy(torch.empty(2, 0, device=device, dtype=x.dtype)).centered(mean[0, :0])


@pytest.mark.compiler
def test_reference_centered_plan_compile_export(device):
    class Model(torch.nn.Module):
        def forward(self, x, y, mean, rhs):
            a = s.lazy(x, y, gamma=0.3, chunk_size=3, metric='rbf').centered(mean)
            return a.matmul(rhs), a.T.matmul(a.matmul(rhs))
    model = Model()
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    def inputs(n, m, grad=False):
        return tuple(torch.randn(*shape, device=device, dtype=torch.float64, requires_grad=grad)
                     for shape in ((n, 2), (m, 2), (m,), (m, 2)))
    for n, m in ((4, 7), (6, 5)):
        values = inputs(n, m, True)
        expected = model(*values)
        actual = compiled(*values)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(sum(v.square().sum() for v in actual), values),
                                   torch.autograd.grad(sum(v.square().sum() for v in expected), values))
    n, m = (torch.export.Dim(name, min=2) for name in ('queries', 'references'))
    exported = torch.export.export(model, inputs(4, 7), dynamic_shapes=({0: n}, {0: m}, {0: m}, {0: m}))
    buffer = io.BytesIO()
    torch.export.save(exported, buffer)
    buffer.seek(0)
    restored = torch.export.load(buffer).module()
    values = inputs(8, 6)
    torch.testing.assert_close(restored(*values), model(*values))
    # Native solver preparation sees the new graph opcode as well as direct actions.
    x, y, mean, _ = inputs(4, 7)
    a = s.lazy(x, y, gamma=0.3, chunk_size=3, metric='rbf').centered(mean)
    normal = a.T @ a + 0.5 * tp.eye_like(a.T @ a)
    rhs = torch.randn(7, 2, device=device, dtype=x.dtype)
    fit = s.linalg.solve(normal, rhs, method='cg', rtol=1e-10, max_iter=50)
    matrix = tp.rbf_kernel(x, y, gamma=.3)
    matrix = matrix - matrix.mean(-1, keepdim=True) - mean + mean.mean()
    torch.testing.assert_close(fit.solution, torch.linalg.solve(matrix.T @ matrix + .5 * torch.eye(7, device=device, dtype=x.dtype), rhs))
    assert fit.converged.all()
