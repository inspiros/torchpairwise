"""Entrywise expression nodes against dense PyTorch and backend contracts."""
import pytest
import torch
import torchpairwise as tp
from torchpairwise.backends import keops
from torchpairwise.ops._operator_algebra import ScalarExpression


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_transforms_consumers_and_gradients(device, dtype):
    torch.manual_seed(230)
    x = torch.rand(2, 3, 2, device=device, dtype=dtype, requires_grad=True)
    y = torch.rand(1, 4, 2, device=device, dtype=dtype, requires_grad=True)
    rows = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
    cols = torch.rand(1, 4, device=device, dtype=dtype, requires_grad=True)
    bound = torch.tensor(.4, device=device, dtype=dtype, requires_grad=True)
    payload = torch.rand(2, 4, 2, device=device, dtype=dtype, requires_grad=True)
    a = tp.lazy(x, y, metric='sqeuclidean', chunk_size=2)
    d = tp.sqeuclidean_distances(x, y)
    expr = (((a + 1.2).sqrt().log().sin().cos().tanh().sigmoid().abs() / 1.3) ** 2
            + (2. / (a + 2.)) + 2. ** (-a)).clamp(min=bound, max=2.)
    dense = (((d + 1.2).sqrt().log().sin().cos().tanh().sigmoid().abs() / 1.3) ** 2
             + (2. / (d + 2.)) + 2. ** (-d)).clamp_min(bound).clamp_max(2.)
    expr = expr.minimum(a + 2.).maximum(a / 4.).add_rows(rows).add_columns(cols)
    dense = torch.maximum(torch.minimum(dense, d + 2.), d / 4.) + rows[..., :, None] + cols[..., None, :]
    operands = (x, y, rows, cols, bound, payload)
    ids = torch.tensor([0, 2], device=device)
    jds = torch.tensor([3, 1], device=device)
    actual = (expr.sum(-1), expr.matmul(payload), expr.logsumexp(-1), expr.var(-1),
              expr.diagonal(), expr.block(ids, jds), expr.T.sum(-1), expr.topk(2).values)
    expected = (dense.sum(-1), dense @ payload, dense.logsumexp(-1), dense.var(-1),
                dense.diagonal(dim1=-2, dim2=-1), dense.index_select(-2, ids).index_select(-1, jds),
                dense.transpose(-1, -2).sum(-1), dense.topk(2).values)
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)
    cot = [torch.randn_like(v) for v in actual]
    ga = torch.autograd.grad(actual, operands, cot)
    gb = torch.autograd.grad(expected, operands, cot)
    torch.testing.assert_close(ga, gb, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_masks_axes_transpose_and_fill(device, dtype):
    torch.manual_seed(231)
    x = torch.rand(3, 2, device=device, dtype=dtype, requires_grad=True)
    y = torch.rand(4, 2, device=device, dtype=dtype, requires_grad=True)
    a = tp.lazy(x, y, metric='rbf', chunk_size=2)
    dense = a.to_dense()
    i = torch.arange(3, device=device)[:, None]
    j = torch.arange(4, device=device)[None, :]
    ri, cj = a.row_index(), a.column_index()
    labels_r = torch.tensor([0, 1, 1], device=device)
    labels_c = torch.tensor([1, 0, 1, 2], device=device)
    same = a.row_values(labels_r).eq(a.column_values(labels_c))
    mask = ((cj <= ri) | same) & ~(cj > ri + 2)
    ref = ((j <= i) | (labels_r[:, None] == labels_c[None, :])) & ~(j > i + 2)
    for operation in ('any', 'all', 'count_nonzero'):
        torch.testing.assert_close(getattr(mask, operation)(-1), getattr(ref, operation)(-1))
    for fill in (0., -float('inf'), float('inf')):
        expr = torch.where(mask, a, fill)
        expected = torch.where(ref, dense, fill)
        torch.testing.assert_close(expr.block(torch.tensor([2, 0], device=device),
                                               torch.tensor([3, 1], device=device)),
                                   expected[[2, 0]][:, [3, 1]])
        torch.testing.assert_close(expr.diagonal(), expected.diagonal())
        torch.testing.assert_close(expr.T.topk(2).values, expected.T.topk(2).values)
        torch.testing.assert_close(expr.sum(-1), expected.sum(-1))
        torch.testing.assert_close(expr.logsumexp(-1), expected.logsumexp(-1), equal_nan=True)
        torch.testing.assert_close(expr.var(-1), expected.var(-1), equal_nan=True)
    torch.testing.assert_close(ri.block(torch.arange(3, device=device), torch.arange(4, device=device)), i.expand(3, 4))
    torch.testing.assert_close(cj.T.sum(-1), j.expand(3, 4).T.sum(-1))
    for lazy, reference in ((ri < cj, i < j), (ri <= cj, i <= j), (ri > cj, i > j),
                            (ri >= cj, i >= j), (ri.eq(cj), i == j), (ri.ne(cj), i != j)):
        torch.testing.assert_close(lazy.any(-1), reference.any(-1))


def test_dispatch_typing_and_pair_validation(device):
    data = torch.rand(3, 4, device=device)
    a = tp.as_lazy(data)
    expr = torch.where(torch.less(a, .5), torch.sqrt(a + 1.), torch.pow(a + 1., 2.))
    assert expr.dtype == torch.float32
    torch.testing.assert_close(expr.sum(-1), torch.where(data < .5, torch.sqrt(data + 1.), (data + 1.).pow(2.)).sum(-1))
    assert a != object() and hash(a) == hash(a)
    with pytest.raises((ValueError, TypeError), match='scalar|as_lazy'):
        torch.where(a < .5, a, data)
    with pytest.raises(ValueError, match='Unsupported'):
        ScalarExpression('pair', a, a)._packet()
    with pytest.raises((TypeError, ValueError)):
        a.clamp(min=torch.ones(2, device=device))


@pytest.mark.compiler
def test_compile_new_nodes_cpu():
    def run(x, y):
        a = tp.lazy(x, y, metric='sqeuclidean')
        return ((a + 1.).sqrt() / 2.).sum(-1)
    x, y = torch.rand(3, 2), torch.rand(4, 2)
    torch.testing.assert_close(torch.compile(run, backend='eager', fullgraph=True)(x, y), run(x, y))


def test_new_nodes_do_not_use_fused_cuda(device):
    if device != 'cuda':
        pytest.skip('CUDA only')
    x, y = torch.rand(16, 3, device=device), torch.rand(19, 3, device=device)
    a = tp.lazy(x, y, metric='sqeuclidean')
    before = torch.ops.torchpairwise._fused_reducer_calls()
    actual = (a + 1.).sqrt().sum(-1)
    assert torch.ops.torchpairwise._fused_reducer_calls() == before
    torch.testing.assert_close(actual, (tp.sqeuclidean_distances(x, y) + 1.).sqrt().sum(-1))


def test_keops_new_nodes_and_fallback(device):
    if not keops.available():
        pytest.skip('KeOps unavailable')
    x, y = torch.rand(4, 3, device=device), torch.rand(5, 3, device=device)
    a = tp.lazy(x, y, metric='sqeuclidean')
    mask = a.column_index() <= a.row_index()
    supported = ((a + 1.).sqrt() / 2.).where(mask, 0.)
    with tp.backend('native'):
        expected = supported.sum(-1)
    before = keops._stats['calls']
    with tp.backend('keops'):
        actual = supported.sum(-1)
    torch.testing.assert_close(actual, expected)
    assert keops._stats['calls'] > before
    for expr in (a.minimum(a + 1.), a.where(a < .5, 0.)):
        with tp.backend('native'):
            expected = expr.sum(-1)
        before = keops._stats['calls']
        with tp.backend('keops'):
            actual = expr.sum(-1)
        torch.testing.assert_close(actual, expected)
        assert keops._stats['calls'] == before


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_clamp_boundary_and_axis_value_gradients(device, dtype):
    data = torch.tensor([[0., 1., 2.], [2., 1., 0.]], device=device, dtype=dtype, requires_grad=True)
    rows = torch.tensor([.2, .3], device=device, dtype=dtype, requires_grad=True)
    columns = torch.tensor([.1, .4, .6], device=device, dtype=dtype, requires_grad=True)
    lower = torch.tensor(1., device=device, dtype=dtype, requires_grad=True)
    a = tp.as_lazy(data, chunk_size=1)
    expr = a.clamp(min=lower).add_rows(rows).add_columns(columns)
    dense = data.clamp_min(lower) + rows[:, None] + columns[None, :]
    actual = expr.sum(-1)
    expected = dense.sum(-1)
    torch.testing.assert_close(actual, expected)
    args = (data, rows, columns, lower)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), args),
                               torch.autograd.grad(expected.sum(), args))
    torch.testing.assert_close(a.row_values(rows).T.sum(-1), rows.expand(3, 2).sum(-1))
    torch.testing.assert_close(a.column_values(columns).sum(-1), columns.expand(2, 3).sum(-1))


def test_integer_promotion_and_torch_functions(device):
    matrix = torch.tensor([[1, 2], [3, 4]], device=device)
    a = tp.as_lazy(matrix)
    for expr, dense in ((torch.sqrt(a), torch.sqrt(matrix)),
                        (torch.div(a, 2), torch.div(matrix, 2)),
                        (torch.sin(a), torch.sin(matrix)),
                        (torch.clamp(a, min=2), torch.clamp(matrix, min=2))):
        assert expr.dtype == dense.dtype
        torch.testing.assert_close(expr.sum(-1), dense.sum(-1))
    for func in (torch.lt, torch.le, torch.gt, torch.ge, torch.eq, torch.ne,
                 torch.less, torch.less_equal, torch.greater, torch.greater_equal):
        expr, dense = func(a, 2), func(matrix, 2)
        assert expr.dtype == torch.bool
        torch.testing.assert_close(expr.any(-1), dense.any(-1))
    expr = torch.logical_or(torch.logical_not(torch.eq(a, 2)), torch.logical_and(a > 1, a < 4))
    dense = torch.logical_or(torch.logical_not(torch.eq(matrix, 2)), torch.logical_and(matrix > 1, matrix < 4))
    torch.testing.assert_close(expr.all(-1), dense.all(-1))


def test_where_gradients_and_linear_consumer(device):
    x = torch.rand(4, 3, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(5, 3, device=device, dtype=x.dtype, requires_grad=True)
    a = tp.lazy(x, y, metric='sqeuclidean', chunk_size=2)
    i, j = a.row_index(), a.column_index()
    mask = j <= i
    result = a.where(mask, 0.).sum(-1)
    dense = tp.sqeuclidean_distances(x, y)
    ref = torch.where(torch.arange(5, device=device)[None, :] <=
                      torch.arange(4, device=device)[:, None], dense, 0.).sum(-1)
    torch.testing.assert_close(result, ref)
    torch.testing.assert_close(torch.autograd.grad(result.sum(), (x, y)),
                               torch.autograd.grad(ref.sum(), (x, y)))

    matrix = torch.eye(4, device=device, dtype=x.dtype) * 3
    matrix += torch.full((4, 4), .1, device=device, dtype=x.dtype)
    lazy = (tp.as_lazy(matrix) + 1.).sqrt()
    rhs = torch.arange(1, 5, device=device, dtype=x.dtype)
    solution = tp.linalg.gmres(lazy, rhs, restart=4, max_iter=20, rtol=1e-10).solution
    torch.testing.assert_close(solution, torch.linalg.solve((matrix + 1.).sqrt(), rhs), rtol=1e-7, atol=1e-8)


def test_keops_integer_labels_and_grad_fallback(device):
    if not keops.available():
        pytest.skip('KeOps unavailable')
    x = torch.rand(4, 3, device=device, dtype=torch.float64)
    y = torch.rand(5, 3, device=device, dtype=x.dtype)
    a = tp.lazy(x, y, metric='rbf')
    rows = torch.tensor([0, 1, 2, 1], device=device)
    cols = torch.tensor([1, 1, 0, 2, 3], device=device)
    mask = a.row_values(rows).eq(a.column_values(cols))
    expr = a.where(mask, 0.).T
    with tp.backend('native'):
        expected = expr.sum(-1)
    before = keops._stats['calls']
    with tp.backend('keops'):
        actual = expr.sum(-1)
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)
    assert keops._stats['calls'] > before
    grad_a = tp.lazy(x.requires_grad_(), y, metric='rbf')
    for expr in (grad_a.clamp(min=.3), grad_a.where(mask, 0.)):
        before = keops._stats['calls']
        with tp.backend('keops'):
            value = expr.sum(-1)
        assert keops._stats['calls'] == before
        assert torch.autograd.grad(value.sum(), x, retain_graph=True)[0] is not None
