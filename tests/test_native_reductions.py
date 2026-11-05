"""Native streaming deployment; dense/catalog oracles cover all builtins."""
import pytest
import torch
import torchpairwise as tp


def test_builtin_reductions_do_not_use_python_replay(device, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Builtin execution reached Python replay')
    monkeypatch.setattr(tp.LazyPairwise, 'block', forbidden)
    x = torch.rand(3, 2, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(4, 2, device=device, dtype=x.dtype, requires_grad=True)
    values = torch.rand(4, 2, device=device, dtype=x.dtype, requires_grad=True)
    outputs = []
    for operation in ('min', 'max', 'sum', 'prod', 'all', 'any'):
        result = tp.lazy(x, y, chunk_size=2, metric='sqeuclidean').reduce(operation, dim=-1)
        dense = (x[:, None] - y[None]).square().sum(-1)
        expected = getattr(dense, operation)(-1)
        expected = expected.values if operation in ('min', 'max') else expected
        torch.testing.assert_close(result, expected)
        if result.requires_grad:
            outputs.append(result.sum())
    match = tp.lazy(x, y, chunk_size=2, metric='euclidean').topk(2, exclude_self=False, largest=False)
    outputs.append(match.values.sum())
    weighted = tp.lazy(x, y, chunk_size=2, metric='linear').matmul(values)
    torch.testing.assert_close(weighted, (x @ y.T) @ values)
    outputs.append(weighted.sum())
    assert all(g.isfinite().all() for g in torch.autograd.grad(sum(outputs), (x, y, values)))


def test_native_streaming_direct_api_and_large_ties(device):
    native = torch.ops.torchpairwise.streaming_reductions
    x = torch.tensor([[1.], [0.]], device=device, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[2.], [2.], [-1.]], device=device, dtype=x.dtype, requires_grad=True)
    result, indices = native(x, y, 'linear', 'neighbors', True, k=2, largest=True, chunk_size=1)
    torch.testing.assert_close(indices, torch.tensor([[0, 1], [0, 1]], device=device))
    torch.testing.assert_close(result, torch.tensor([[2., 2.], [0., 0.]], device=device, dtype=x.dtype))
    gradient = torch.autograd.grad(result.sum(), y)[0]
    torch.testing.assert_close(gradient, torch.tensor([[1.], [1.], [0.]], device=device, dtype=x.dtype))
    rows, empty = native(x, y, 'linear', 'max', True, keepdim=True, chunk_size=1)
    assert rows.shape == (2, 1) and empty.shape == (0,)
    torch.testing.assert_close(torch.autograd.grad(rows.sum(), y)[0],
                               torch.tensor([[1.], [0.], [0.]], device=device, dtype=x.dtype))
    with pytest.raises(RuntimeError, match='chunk_size'):
        native(x, y, chunk_size=0)
    with pytest.raises(RuntimeError, match='values'):
        native(x, y, 'linear', 'kernel_sum', True)


@pytest.mark.compiler
def test_streaming_export_compile_dynamic_and_parameter_gradients(device, tmp_path):
    class Streaming(torch.nn.Module):
        def forward(self, x, y, w, values):
            match = tp.lazy(x, y, w=w, chunk_size=2, metric='sqeuclidean').topk(2, exclude_self=False, largest=False)
            rows = tuple((tp.lazy(x, y, w=w, chunk_size=2, metric='sqeuclidean').reduce(operation, dim=-1) for operation in ('min', 'max', 'sum', 'prod', 'all', 'any')))
            payload = tp.lazy(x, y, gamma=0.4, chunk_size=2, metric='rbf').matmul(values)
            return match.values, match.indices, *rows, payload
    n, m = torch.export.Dim('n', min=3), torch.export.Dim('m', min=4)
    def inputs(n, m):
        return (torch.rand(1, n, 3, device=device, dtype=torch.float64, requires_grad=True),
                torch.rand(2, m, 3, device=device, dtype=torch.float64, requires_grad=True),
                (torch.rand(2, 3, device=device, dtype=torch.float64) + .2).requires_grad_(),
                torch.rand(1, m, 2, device=device, dtype=torch.float64, requires_grad=True))
    module = Streaming()
    exported = torch.export.export(module, inputs(3, 4),
                                   dynamic_shapes=({1: n}, {1: m}, {}, {1: m}))
    path = tmp_path / 'streaming.pt2'
    torch.export.save(exported, path)
    restored = torch.export.load(path).module()
    args = inputs(5, 6)
    actual, expected = restored(*args), module(*args)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    def objective(result):
        return sum(value.sum() for value in result if value.is_floating_point())
    for a, b in zip(torch.autograd.grad(objective(actual), args),
                    torch.autograd.grad(objective(expected), args)):
        torch.testing.assert_close(a, b)
    torch.compiler.reset()
    compiled = torch.compile(module, fullgraph=True, dynamic=True)
    compiled(*inputs(3, 4))
    actual, expected = compiled(*args), module(*args)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    for a, b in zip(torch.autograd.grad(objective(actual), args),
                    torch.autograd.grad(objective(expected), args)):
        torch.testing.assert_close(a, b)
    torch.compiler.reset()


@pytest.mark.compiler
def test_cloud_streaming_export_dynamic_lengths(device, tmp_path):
    class Clouds(torch.nn.Module):
        def forward(self, x, y, xl, yl):
            nearest = tp.lazy(x, y, x1_lengths=xl, x2_lengths=yl, chunk_size=2, metric='chamfer').min(-1, exclude_self=False)
            total = tp.lazy(x, y, x1_lengths=xl, x2_lengths=yl, chunk_size=2, metric='chamfer').reduce('sum', dim=-1)
            return nearest.values, nearest.indices, total
    n, m = torch.export.Dim('n', min=3), torch.export.Dim('m', min=3)
    p, q = torch.export.Dim('p', min=3), torch.export.Dim('q', min=3)
    def inputs(n, m, p, q):
        x = torch.rand(n, p, 2, device=device, requires_grad=True)
        y = torch.rand(m, q, 2, device=device, requires_grad=True)
        xl = torch.full((n,), p - 1, device=device, dtype=torch.int64)
        yl = torch.full((m,), q - 1, device=device, dtype=torch.int64)
        with torch.no_grad():
            x[:, -1] = float('nan')
            y[:, -1] = float('nan')
        return x, y, xl, yl
    module = Clouds()
    exported = torch.export.export(module, inputs(3, 4, 4, 5),
                                   dynamic_shapes=({0: n, 1: p}, {0: m, 1: q}, {0: n}, {0: m}))
    path = tmp_path / 'cloud-streaming.pt2'
    torch.export.save(exported, path)
    args = inputs(5, 6, 6, 7)
    actual, expected = torch.export.load(path).module()(*args), module(*args)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    for a, b in zip(torch.autograd.grad(actual[0].sum() + actual[2].sum(), args[:2]),
                    torch.autograd.grad(expected[0].sum() + expected[2].sum(), args[:2])):
        torch.testing.assert_close(a, b)
