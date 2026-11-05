"""Common native consumers over caller-owned real scalar matrices."""
import io

import pytest
import torch
import torchpairwise as tp


def test_matrix_operator_values_gradients_and_transpose(device):
    torch.manual_seed(111)
    matrix = torch.randn(2, 1, 3, 8, device=device, dtype=torch.float64)[..., ::2].requires_grad_()
    a = tp.as_lazy(matrix, chunk_size=2)
    assert a.shape == matrix.shape and a.dtype == matrix.dtype and a.device == matrix.device
    for operator, dense in ((a, matrix), (a.T, matrix.transpose(-1, -2))):
        n, m = dense.shape[-2:]
        l = torch.randn(1, 2, n, 2, device=device, dtype=matrix.dtype, requires_grad=True)
        r = torch.randn(2, 1, m, 3, device=device, dtype=matrix.dtype, requires_grad=True)
        rows = torch.tensor([n-1, 0, n-1], device=device)
        columns = torch.tensor([m-1, 0], device=device)
        actual = (operator.matmul(r), operator.bilinear(l, r), operator.mean(-2),
                  operator.diagonal(), operator.block(rows, columns))
        expected = (dense @ r, l.transpose(-1, -2) @ dense @ r, dense.mean(-2),
                    dense.diagonal(dim1=-2, dim2=-1), dense.index_select(-2, rows).index_select(-1, columns))
        torch.testing.assert_close(actual, expected)
        loss = lambda out: sum(t.square().sum() for t in out)
        torch.testing.assert_close(torch.autograd.grad(loss(actual), (matrix, l, r)),
                                   torch.autograd.grad(loss(expected), (matrix, l, r)))
        assert operator.block(rows[:0], columns).shape[-2:] == (0, 2)
    square = matrix[..., :3]
    c = torch.randn(3, 2, device=device, dtype=matrix.dtype, requires_grad=True)
    actual = tp.as_lazy(square, chunk_size=1).quadratic(c)
    expected = c.T @ square @ c
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), (matrix, c)),
                               torch.autograd.grad(expected.sum(), (matrix, c)))
    with torch.no_grad():
        matrix.add_(1.)
    torch.testing.assert_close(a.sum(), matrix.sum((-1, -2)))
    assert a.T.T.matrix is matrix


def test_matrix_operator_reductions_and_exact_entries(device):
    def check(dtype):
        if dtype == torch.int64:
            matrix = torch.tensor([[2**60+1, 2**60, -2**60], [2**60+2, -2**60+1, 2**60]], device=device)
        else:
            matrix = torch.tensor([[1, 0, 2], [1, 3, 4]], device=device, dtype=dtype)
        a = tp.as_lazy(matrix, chunk_size=1)
        ids, columns = torch.tensor([1, 0, 1], device=device), torch.tensor([2, 0], device=device)
        torch.testing.assert_close(a.block(ids, columns), matrix.index_select(0, ids).index_select(1, columns))
        torch.testing.assert_close(a.diagonal(), matrix.diagonal())
        assert a.dtype == dtype
        # Dtypes share the same dispatcher/reducer paths. Check every supported
        # dtype once, then exercise axes/transposes only for representative float,
        # exact int64 and Boolean inputs instead of multiplying the complete grid.
        if dtype not in (torch.float64, torch.int64, torch.bool):
            for name in ('sum', 'min', 'max', 'prod', 'all', 'any'):
                value = matrix.ne(0) if name in ('all', 'any') else matrix
                ref = getattr(value, name)(-1)
                if name in ('min', 'max'): ref = ref.values
                torch.testing.assert_close(getattr(a, {'min': 'amin', 'max': 'amax'}.get(name, name))(-1), ref)
            return
        for operator, dense in ((a, matrix), (a.T, matrix.T)):
            for dim in (None, -1, -2):
                source = dense.flatten() if dim is None else dense
                for reduction in ('sum', 'mean', 'min', 'max', 'prod', 'all', 'any'):
                    value = source
                    if reduction in ('all', 'any'):
                        value = value.ne(0)
                    axis = -1 if dim is None else dim
                    ref = (value.sum(axis).double() / value.shape[axis]
                           if reduction == 'mean' and not matrix.is_floating_point()
                           else getattr(value, reduction)(axis))
                    if reduction in ('min', 'max'):
                        ref = ref.values
                    torch.testing.assert_close(getattr(operator, {'min': 'amin', 'max': 'amax'}.get(reduction, reduction))(dim), ref)
        if dtype == torch.int64:
            cancellation = torch.tensor([[2**60, 1], [-2**60, 0]], device=device)
            assert tp.as_lazy(cancellation, chunk_size=1).mean().item() == .25
        if not matrix.is_floating_point():
            r = torch.randn(3, 2, device=device, dtype=torch.float64, requires_grad=True)
            l = torch.randn(2, device=device, dtype=torch.float64, requires_grad=True)
            actual = a.matmul(r), a.bilinear(l, r)
            expected = matrix.double() @ r, l @ matrix.double() @ r
            torch.testing.assert_close(actual, expected)
            loss = lambda values: sum(t.square().sum() for t in values)
            torch.testing.assert_close(torch.autograd.grad(loss(actual), (l, r)),
                                       torch.autograd.grad(loss(expected), (l, r)))
    for dtype in (torch.float32, torch.float64, torch.uint8, torch.int8,
                  torch.int16, torch.int32, torch.int64, torch.bool):
        check(dtype)


def test_matrix_operator_boundary_gradients_and_empty(device):
    for data in ([[1., 0.], [0., 2.]], [[float('nan'), 1.], [float('nan'), 2.]]):
        matrix = torch.tensor(data, device=device, dtype=torch.float64, requires_grad=True)
        a = tp.as_lazy(matrix, chunk_size=1).T
        for reduction in ('min', 'max'):
            actual = getattr(a, {'min': 'amin', 'max': 'amax'}.get(reduction, reduction))()
            ref = getattr(matrix.T.flatten(), reduction)(-1).values
            torch.testing.assert_close(actual, ref, equal_nan=True)
            torch.testing.assert_close(torch.autograd.grad(actual, matrix)[0],
                                       torch.autograd.grad(ref, matrix)[0])
    for data in ([[1., 1.], [2., 3.]], [[0., 1.], [2., 3.]], [[0., 0.], [2., 3.]], [[0., 1.], [2., float('inf')]]):
        matrix = torch.tensor(data, device=device, dtype=torch.float64, requires_grad=True)
        actual, ref = tp.as_lazy(matrix, chunk_size=1).prod(dim=[-2, -1]), matrix.prod()
        torch.testing.assert_close(actual, ref, equal_nan=True)
        torch.testing.assert_close(torch.autograd.grad(actual, matrix)[0], torch.autograd.grad(ref, matrix)[0], equal_nan=True)
    for n, m in ((0, 3), (2, 0), (0, 0)):
        matrix = torch.empty(2, n, m, device=device, dtype=torch.float64, requires_grad=True)
        a = tp.as_lazy(matrix, chunk_size=1)
        assert a.sum().eq(0).all() and a.prod().eq(1).all() and a.all().all() and not a.any().any()
        assert a.mean().isnan().all() and a.diagonal().shape == (2, 0)
        r = torch.ones(m, 2, device=device, dtype=matrix.dtype, requires_grad=True)
        l = torch.ones(n, 2, device=device, dtype=matrix.dtype, requires_grad=True)
        result = a.bilinear(l, r)
        assert result.eq(0).all()
        assert all(g.eq(0).all() for g in torch.autograd.grad(result.sum(), (matrix, l, r)))
        with pytest.raises((RuntimeError, IndexError)):
            a.min()
    matrix = torch.randn(3, 3, device=device, dtype=torch.float64, requires_grad=True)
    mask = ~torch.eye(3, device=device, dtype=torch.bool)
    a = tp.as_lazy(matrix, chunk_size=1)
    torch.testing.assert_close(a.sum(exclude_self=True), matrix[mask].sum())
    torch.testing.assert_close(a.mean(-2, keepdim=True, exclude_self=True),
                               matrix.masked_fill(~mask, 0).sum(-2, keepdim=True) / 2)
    assert tp.as_lazy(matrix.expand(0, *matrix.shape)).sum().shape == (0,)
    with pytest.raises(RuntimeError, match='bounds'):
        a.block(torch.tensor([-1], device=device), torch.tensor([0], device=device))


def test_kernel_specialization_and_matrix_validation_amp(device):
    torch.manual_seed(112)
    x = torch.rand(3, 2, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(4, 2, device=device, dtype=x.dtype, requires_grad=True)
    k = tp.lazy(x, y, metric='rbf', gamma=0.4, chunk_size=1)
    assert isinstance(k, tp.LazyPairwise) and isinstance(k.T, tp.LazyPairwise)
    a = tp.lazy(x, y, metric='rbf_kernel', gamma=0.4, chunk_size=1)
    torch.testing.assert_close(k.T.sum(-1), a.T.sum(-1))
    weights = tp.rbf_kernel(x, y, gamma=.4).T
    delta = x.unsqueeze(0) - y.unsqueeze(1)
    expected = ((weights[..., None] * delta).sum(1),
                torch.einsum('ij,ijd,ije->ide', weights, delta, delta))
    actual = k.T.displacement(), k.T.residual_outer_product()
    torch.testing.assert_close(actual, expected)
    loss = lambda out: sum(t.square().sum() for t in out)
    torch.testing.assert_close(torch.autograd.grad(loss(actual), (x, y)),
                               torch.autograd.grad(loss(expected), (x, y)))
    dtype = torch.bfloat16 if device == 'cpu' else torch.float16
    matrix = torch.rand(3, 4, device=device, dtype=dtype, requires_grad=True)
    c = torch.rand(4, 2, device=device, dtype=dtype, requires_grad=True)
    with torch.autocast(device, dtype=dtype):
        op = tp.as_lazy(matrix, chunk_size=1)
        actual = op.matmul(c)
        assert op.dtype == torch.float32 and actual.dtype == torch.float32
        assert tp.lazy(x.to(dtype), y.to(dtype), metric='rbf').dtype == torch.float32
    torch.testing.assert_close(actual, matrix.float() @ c.float())
    assert all(g.isfinite().all() for g in torch.autograd.grad(actual.sum(), (matrix, c)))
    with pytest.raises((TypeError, ValueError)):
        tp.as_lazy(torch.ones(3, device=device))
    with pytest.raises(RuntimeError):
        tp.as_lazy(torch.ones(2, 2, device=device, dtype=torch.complex64)).sum()
    with pytest.raises(RuntimeError):
        tp.as_lazy(torch.ones(2, 2, device=device, dtype=torch.int64)).matmul(torch.ones(2, device=device, dtype=torch.int64))


@pytest.mark.compiler
@pytest.mark.parametrize('integer', (False, True))
def test_matrix_operator_compile_export(device, integer):
    class Model(torch.nn.Module):
        def forward(self, matrix, l, r):
            a = tp.as_lazy(matrix, chunk_size=2).T
            rows = torch.tensor([1, 0, 1], device=matrix.device)
            cols = torch.tensor([1, 0], device=matrix.device)
            return a.matmul(r), a.bilinear(l, r), a.mean(-2), a.sum(-1), a.diagonal(), a.block(rows, cols)
    def inputs(b, n, m):
        matrix = (torch.randint(0, 5, (b, n, m), device=device) if integer
                  else torch.rand(b, n, m, device=device, dtype=torch.float64).requires_grad_())
        return (matrix, torch.rand(m, 2, device=device, dtype=torch.float64, requires_grad=True),
                torch.rand(n, 2, device=device, dtype=torch.float64, requires_grad=True))
    torch.manual_seed(113); torch._dynamo.reset()
    args = inputs(2, 3, 4)
    model = Model()
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    actual, expected = compiled(*args), model(*args)
    torch.testing.assert_close(actual, expected)
    loss = lambda out: sum(t.double().square().sum() for t in out)
    leaves = args[1:] if integer else args
    torch.testing.assert_close(torch.autograd.grad(loss(actual), leaves),
                               torch.autograd.grad(loss(expected), leaves))
    b, n, m = torch.export.Dim('batch', min=1, max=4), torch.export.Dim('n', min=2, max=6), torch.export.Dim('m', min=2, max=6)
    graph = torch.export.export(model, args, dynamic_shapes=({0:b, 1:n, 2:m}, {0:m}, {0:n}))
    buffer = io.BytesIO(); torch.export.save(graph, buffer); buffer.seek(0)
    restored = torch.export.load(buffer).module()
    other = inputs(3, 4, 2)
    torch.testing.assert_close(restored(*other), model(*other))
    if not integer:
        matrix = args[0]
        values = [matrix, torch.arange(matrix.shape[-1], device=device).unsqueeze(-1)]
        values += [matrix.new_empty(0)] * 8
        torch.library.opcheck(torch.ops.torchpairwise._streaming_reduce.default,
            (values, '_matrix', 'sum', False, 2, 1, False, False, False, False,
             2, 7, 3, [], [], [], '_matrix'))


def test_structured_kernel_specialization(device):
    torch.manual_seed(114)
    def mixture(n):
        means = torch.rand(n, 2, 2, device=device, dtype=torch.float64, requires_grad=True)
        covariance = torch.eye(2, device=device, dtype=means.dtype).expand(n, 2, 2, 2).clone().requires_grad_()
        weights = torch.rand(n, 2, device=device, dtype=means.dtype).requires_grad_()
        return means, covariance, weights
    x, y = mixture(2), mixture(3)
    l = torch.randn(3, 2, device=device, dtype=torch.float64, requires_grad=True)
    r = torch.randn(2, 2, device=device, dtype=torch.float64, requires_grad=True)
    k = tp.lazy(x, y, metric='gaussian_mixture_overlap', chunk_size=1, component_chunk_size=1).T
    a = tp.lazy(x, y, metric='gaussian_mixture_overlap', chunk_size=1, component_chunk_size=1).T
    assert isinstance(k, tp.LazyPairwise) and k.shape == a.shape and k.dtype == a.dtype
    actual = k.matmul(r), k.bilinear(l, r), k.mean(), k.diagonal()
    expected = a.matmul(r), a.bilinear(l, r), a.mean(), a.diagonal()
    torch.testing.assert_close(actual, expected)
    raw = (*x, *y, l, r)
    loss = lambda out: sum(t.square().sum() for t in out)
    torch.testing.assert_close(torch.autograd.grad(loss(actual), raw), torch.autograd.grad(loss(expected), raw))
    torch.testing.assert_close(tp.lazy(x, y, metric='gaussian_mixture_overlap').bilinear(r, l), a.T.bilinear(r, l))
    with pytest.raises(ValueError, match='vector'):
        k.displacement()
    with pytest.raises(ValueError, match='vector'):
        k.residual_outer_product()
    # Compile/export coverage is shared with the existing structured operator.
    class Model(torch.nn.Module):
        def forward(self, x, y, values):
            return tp.lazy(x, y, metric='gaussian_mixture_overlap').matmul(values)
    model = Model()
    detached = tuple(t.detach() for t in x), tuple(t.detach() for t in y), l.detach()
    torch.testing.assert_close(torch.compile(model, backend='eager', fullgraph=True)(*detached), model(*detached))
    graph = torch.export.export(model, detached)
    torch.testing.assert_close(graph.module()(*detached), model(*detached))
