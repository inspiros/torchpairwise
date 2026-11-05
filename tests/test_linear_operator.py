"""Distinct linear-map actions; dense small oracles and deployment coverage."""
import io

import pytest
import torch
import torchpairwise as s
import torchpairwise as tp


def center(matrix):
    return matrix - matrix.mean(-2, keepdim=True) - matrix.mean(-1, keepdim=True) + matrix.mean((-2, -1), keepdim=True)


def test_linear_composition_transpose_shifts_and_gradients(device):
    torch.manual_seed(71)
    def tensor(*shape):
        return torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True)
    x, y, b, e = tensor(2, 4, 3), tensor(1, 6, 3), tensor(1, 6, 5), tensor(4, 5)
    v, u, p, factor, diagonal = tensor(5, 2), tensor(4), tensor(4, 2), tensor(), tensor()
    source = s.lazy(x, y, metric='sqeuclidean', chunk_size=3)
    a = source
    assert a.shape == source.shape and a.dtype == x.dtype and a.device == x.device
    c = factor * (a.centered() @ s.as_lazy(b)) + s.as_lazy(e)
    dense = tp.sqeuclidean_distances(x, y)
    expected_c = (center(dense) @ b) * factor + e
    square = a @ a.T + diagonal * tp.eye_like(a @ a.T)
    expected_square = dense @ dense.transpose(-1, -2) + diagonal * torch.eye(4, device=device, dtype=x.dtype)
    actual = [c @ v, c.T @ u, square @ p, (c - c).matmul(v), (-c).matmul(v)]
    expected = [expected_c @ v, expected_c.transpose(-1, -2) @ u,
                expected_square @ p, (expected_c - expected_c) @ v, -expected_c @ v]
    torch.testing.assert_close(actual, expected)
    variables = (x, y, b, e, v, u, p, factor, diagonal)
    torch.testing.assert_close(
        torch.autograd.grad(sum(z.square().sum() for z in actual), variables),
        torch.autograd.grad(sum(z.square().sum() for z in expected), variables),
        rtol=1e-8, atol=1e-8)
    # Binding observes value mutation; it does not cache the dense result.
    with torch.no_grad():
        e.add_(.1)
        torch.testing.assert_close(c @ v, ((center(dense) @ b) * factor + e) @ v)


def test_linear_empty_blocks_and_strided_inputs(device):
    for rows, columns, channels in ((3, 4, 2), (0, 4, 2), (3, 0, 2)):
        base = torch.randn(columns, rows, device=device, dtype=torch.float64, requires_grad=True)
        values = torch.randn(columns, channels, device=device, dtype=base.dtype, requires_grad=True)
        matrix = base.T
        operator = s.as_lazy(matrix).centered()
        actual = operator @ values
        # Center matrix products through vector means as in H A H.
        reference = matrix @ (values - values.mean(-2, keepdim=True))
        reference = reference - reference.mean(-2, keepdim=True)
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), (base, values)),
                                   torch.autograd.grad(reference.sum(), (base, values)))


def test_linear_contracts():
    a = s.as_lazy(torch.ones(3, 4)).centered()
    with pytest.raises(RuntimeError, match='square'):
        a + 1.0 * tp.eye_like(a)
    with pytest.raises(RuntimeError, match='contracted'):
        a @ s.as_lazy(torch.ones(5, 2))
    with pytest.raises(TypeError, match='real'):
        True * a
    with pytest.raises(ValueError, match='scalar'):
        torch.ones(1) * a
    with pytest.raises(ValueError, match='dtype'):
        a.matmul(torch.ones(4, 2, dtype=torch.float64))
    with pytest.raises(RuntimeError, match='positive'):
        a.matmul(torch.ones(4, 0))
    with pytest.raises(TypeError, match='[Tt]ensor'):
        s.as_lazy(s.residual_operator(torch.ones(3, 2)))
    with pytest.raises(TypeError, match='float32'):
        s.as_lazy(torch.ones(3, 3, dtype=torch.int64)).centered()


@pytest.mark.compiler
def test_linear_compile_export_and_first_gradients(device):
    class Model(torch.nn.Module):
        def forward(self, x, y, b, v, shift):
            a = s.lazy(x, y, gamma=0.2, chunk_size=3, metric='rbf')
            # Dynamo cannot trace `@` between two lazy objects; use matmul.
            c = 0.5 * a.matmul(s.as_lazy(b)).centered()
            gram = c.matmul(c.T)
            return (gram + shift * tp.eye_like(gram)).matmul(v)
    model = Model()
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    for n, m, width in ((5, 7, 4), (8, 4, 6)):
        x = torch.randn(n, 3, device=device, requires_grad=True)
        y = torch.randn(m, 3, device=device, requires_grad=True)
        b = torch.randn(m, width, device=device, requires_grad=True)
        v = torch.randn(n, 2, device=device, requires_grad=True)
        shift = torch.tensor(.1, device=device, requires_grad=True)
        c = center(tp.rbf_kernel(x, y, gamma=.2) @ b) * .5
        expected = (c @ c.T) @ v + shift * v
        for actual in (model(x, y, b, v, shift), compiled(x, y, b, v, shift)):
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(torch.autograd.grad(actual.sum(), (x, y, b, v, shift)),
                                       torch.autograd.grad(expected.sum(), (x, y, b, v, shift), retain_graph=True),
                                       rtol=3e-4, atol=3e-5)
    n_dim, m_dim, width_dim = (torch.export.Dim(name, min=2) for name in ('n', 'm', 'width'))
    exported = torch.export.export(model, (x, y, b, v, shift), dynamic_shapes=(
        {0: n_dim}, {0: m_dim}, {0: m_dim, 1: width_dim}, {0: n_dim}, {}))
    buffer = io.BytesIO()
    torch.export.save(exported, buffer)
    buffer.seek(0)
    restored = torch.export.load(buffer).module()
    x = torch.randn(6, 3, device=device)
    y = torch.randn(9, 3, device=device)
    b = torch.randn(9, 5, device=device)
    v = torch.randn(6, 2, device=device)
    shift = torch.tensor(.2, device=device)
    torch.testing.assert_close(restored(x, y, b, v, shift), model(x, y, b, v, shift))
