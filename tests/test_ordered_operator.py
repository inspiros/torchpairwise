"""Native sequence/token/MMD operators against canonical dense comparisons."""
import io

import pytest
import torch
import torchpairwise as tp


METRICS = ('dtw', 'soft_dtw', 'levenshtein', 'indel', 'mmd', 'mmd_statistic')


def options(metric):
    return dict(gamma=.7) if metric == 'soft_dtw' else dict(estimator='unbiased') if metric == 'mmd_statistic' else {}


def inputs(metric, n, device, batch=()):
    if metric in ('levenshtein', 'indel'):
        # Adjacent IDs above float64's exact integer range must remain distinct.
        return torch.randint(0, 4, (*batch, n, 4), device=device) + 2**60
    return torch.rand((*batch, n, 4, 2), device=device, dtype=torch.float64).requires_grad_()


@pytest.mark.parametrize('metric', METRICS)
def test_ordered_operator_consumers_and_gradients(device, metric):
    torch.manual_seed(104)
    x, y = inputs(metric, 2, device), inputs(metric, 3, device)
    kw = options(metric)
    a = tp.lazy(x, y, metric=metric, chunk_size=1, **kw).T
    dense = tp.compare(x, y, metric=metric, **kw).T
    assert a.shape == dense.shape and a.dtype == dense.dtype and a.device == dense.device
    l = torch.randn(3, 2, device=device, dtype=torch.float64, requires_grad=True)
    r = torch.randn(2, 2, device=device, dtype=torch.float64, requires_grad=True)
    numeric = dense.double()
    actual = a.matmul(r), a.bilinear(l, r), a.mean(-2), a.diagonal()
    expected = numeric @ r, l.T @ numeric @ r, numeric.mean(-2), dense.diagonal()
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
    raw = (x, y, l, r) if x.is_floating_point() else (l, r)
    loss = lambda outputs: sum(t.double().square().sum() for t in outputs)
    torch.testing.assert_close(torch.autograd.grad(loss(actual), raw),
                               torch.autograd.grad(loss(expected), raw), rtol=3e-7, atol=3e-8)
    rows, cols = torch.tensor([2, 0, 2], device=device), torch.tensor([1, 0], device=device)
    got = a.block(rows, cols)
    ref = tp.compare(x, y, metric=metric, **kw).T.index_select(0, rows).index_select(1, cols)
    torch.testing.assert_close(got, ref)
    if x.is_floating_point():
        torch.testing.assert_close(torch.autograd.grad(got.sum(), (x, y)),
                                   torch.autograd.grad(ref.sum(), (x, y)), rtol=3e-7, atol=3e-8)
    assert a.block(rows[:0], cols).shape == (0, 2)


def test_ordered_operator_broadcast_reductions_lengths_and_empty(device):
    torch.manual_seed(105)
    for metric in ('soft_dtw', 'levenshtein', 'mmd_statistic'):
        x, y = inputs(metric, 2, device, (2, 1)), inputs(metric, 3, device, (1, 2))
        lx, ly = torch.tensor([2, 3], device=device), torch.tensor([3, 2, 4], device=device)
        if x.is_floating_point():
            with torch.no_grad():
                x[..., 0, 2:, :] = float('nan')
                x[..., 1, 3:, :] = float('nan')
                y[..., 0, 3:, :] = float('nan')
                y[..., 1, 2:, :] = float('nan')
        kw = dict(**options(metric), x1_lengths=lx, x2_lengths=ly)
        dense = tp.compare(x, y, metric=metric, **kw)
        a = tp.lazy(x, y, metric=metric, chunk_size=1, **kw)
        assert a.shape == dense.shape
        for operator, matrix in ((a, dense), (a.T, dense.transpose(-1, -2))):
            for dim in (-1, -2, None):
                data = matrix.flatten(-2) if dim is None else matrix
                for reduction in ('sum', 'mean', 'prod', 'min', 'max', 'all', 'any'):
                    source = data.double() if reduction == 'mean' else data
                    ref = getattr(source, reduction)(-1 if dim is None else dim)
                    if reduction in ('min', 'max'):
                        ref = ref.values
                    torch.testing.assert_close(getattr(operator, {'min': 'amin', 'max': 'amax'}.get(reduction, reduction))(dim), ref)
        if x.is_floating_point():
            actual_grad = torch.autograd.grad(a.sum().sum(), (x, y))
            torch.testing.assert_close(actual_grad, torch.autograd.grad(dense.sum(), (x, y)), rtol=3e-7, atol=3e-8)
            assert all(g.isfinite().all() for g in actual_grad)
            assert actual_grad[0][..., 0, 2:, :].eq(0).all()
        selfop = tp.lazy(x, metric=metric, chunk_size=1, x1_lengths=lx, **options(metric))
        matrix = tp.compare(x, metric=metric, x1_lengths=lx, **options(metric))
        torch.testing.assert_close(selfop.sum(exclude_self=True), matrix.sum((-1, -2)) - matrix.diagonal(dim1=-2, dim2=-1).sum(-1))
        c = torch.randn(2, device=device, dtype=torch.float64)
        torch.testing.assert_close(selfop.quadratic(c), torch.einsum('i,...ij,j->...', c, matrix.double(), c))
        clean_x, clean_y = (x.nan_to_num(), y.nan_to_num()) if x.is_floating_point() else (x, y)
        axis = -3 if x.is_floating_point() else -2
        for xx, yy in ((clean_x.narrow(axis, 0, 0), clean_y), (clean_x, clean_y.narrow(axis, 0, 0))):
            empty = tp.lazy(xx, yy, metric=metric, chunk_size=1, **options(metric))
            assert (empty.sum() == 0).all() and (empty.prod() == 1).all()
            assert empty.all().all() and not empty.any().any() and empty.mean().isnan().all()
            assert empty.diagonal().shape[-1] == 0
            with pytest.raises((RuntimeError, IndexError)):
                empty.min()


def test_ordered_operator_sequence_paths_tokens_and_mmd_kernels(device):
    # Infeasible paths and identical Soft-DTW must have zero whole-pair gradients.
    x = torch.ones(2, 3, 1, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.ones(2, 4, 1, device=device, dtype=torch.float64, requires_grad=True)
    for metric in ('dtw', 'soft_dtw'):
        a = tp.lazy(x, y, metric=metric, window=0, chunk_size=1, **options(metric))
        assert a.diagonal().isposinf().all()
        g = torch.autograd.grad(a.sum(), (x, y))
        assert all(t.eq(0).all() for t in g)
        if metric == 'dtw':
            actual = torch.autograd.grad(a.prod(), (x, y))
            dense = tp.dtw_distances(x, y, window=0)
            torch.testing.assert_close(actual, torch.autograd.grad(dense.prod(), (x, y)))
            assert all(t.eq(0).all() for t in actual)
        zero = tp.lazy(x, metric=metric, chunk_size=1, **options(metric)).diagonal()
        assert zero.eq(0).all() and torch.autograd.grad(zero.sum(), x)[0].eq(0).all()
    empty_tokens = torch.empty(2, 0, device=device, dtype=torch.int64)
    ids = torch.tensor([[2**60, 2**60+1], [2**60+2, 2**60+2]], device=device)
    for metric in ('levenshtein', 'indel'):
        a = tp.lazy(empty_tokens, ids, metric=metric, chunk_size=1)
        assert a.sum().dtype == torch.int64 and a.sum() == 8
        assert a.mean().dtype == torch.float64 and a.mean() == 2
        torch.testing.assert_close(a.amin(-1), torch.tensor([2, 2], device=device))
        single = tp.lazy(ids[:, :1], metric=metric, chunk_size=1)
        assert single.sum() == (2 if metric == 'levenshtein' else 4)
        with pytest.raises(RuntimeError):
            a.matmul(torch.ones(2, device=device, dtype=torch.int64))
    torch.manual_seed(106)
    x, y = inputs('mmd', 2, device), inputs('mmd', 3, device)
    for kernel, params in (('linear', {}), ('polynomial', dict(degree=2, gamma=.5)),
                           ('laplacian_kernel', {}), ('matern', dict(nu=2.5))):
        for metric in ('mmd', 'mmd_statistic'):
            kw = dict(kernel=kernel, **params, **options(metric))
            a = tp.lazy(x, y, metric=metric, chunk_size=1, metric_chunk_size=2, **kw)
            dense = tp.compare(x, y, metric=metric, **kw)
            torch.testing.assert_close(a.sum(-1), dense.sum(-1))
            torch.testing.assert_close(torch.autograd.grad(a.sum(), (x, y)),
                                       torch.autograd.grad(dense.sum(), (x, y)), rtol=3e-7, atol=3e-8)
    torch.testing.assert_close(tp.lazy(x, y, metric='mmd_statistic').sum(), tp.mmd_statistics(x, y).sum())
    # Unbiased statistics retain their negative diagonal, not a distance clamp.
    assert tp.lazy(x, metric='mmd_statistic', estimator='unbiased').diagonal().lt(0).all()


@pytest.mark.compiler
@pytest.mark.parametrize('metric', ('soft_dtw', 'levenshtein', 'mmd_statistic'))
@pytest.mark.filterwarnings('error:.*_streaming_reduce.*:UserWarning')
def test_ordered_operator_compile_export_and_preparation(device, metric):
    class Model(torch.nn.Module):
        def forward(self, x, y, r):
            a = tp.lazy(x, y, metric=metric, chunk_size=1, **options(metric)).T
            return a.matmul(r), a.mean(-2), a.diagonal()
    torch.manual_seed(107); torch._dynamo.reset()
    model = Model()
    x, y = inputs(metric, 2, device, (2,)), inputs(metric, 3, device, (2,))
    r = torch.randn(2, 2, device=device, dtype=torch.float64, requires_grad=True)
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    actual, expected = compiled(x, y, r), model(x, y, r)
    torch.testing.assert_close(actual, expected)
    leaves = (x, y, r) if x.is_floating_point() else (r,)
    loss = lambda out: sum(t.double().square().sum() for t in out)
    torch.testing.assert_close(torch.autograd.grad(loss(actual), leaves),
                               torch.autograd.grad(loss(expected), leaves), rtol=3e-6, atol=3e-7)
    b, n, m = torch.export.Dim('batch', min=1, max=4), torch.export.Dim('n', min=2, max=6), torch.export.Dim('m', min=2, max=6)
    graph = torch.export.export(model, (x, y, r), dynamic_shapes=({0:b, 1:n}, {0:b, 1:m}, {0:n}))
    buffer = io.BytesIO(); torch.export.save(graph, buffer); buffer.seek(0)
    restored = torch.export.load(buffer).module()
    args = inputs(metric, 4, device, (3,)), inputs(metric, 2, device, (3,)), torch.randn(4, 2, device=device, dtype=torch.float64)
    torch.testing.assert_close(restored(*args), model(*args))
    if metric == 'soft_dtw':
        lengths = torch.full(x.shape[:-2], 4, device=device, dtype=torch.int64)
        torch.library.opcheck(torch.ops.torchpairwise._sequence_operator_cost.default,
            (x, x, lengths, lengths, .7, None, 1, True))
    if metric == 'mmd_statistic':
        lengths = torch.full(x.shape[:-2], 4, device=device, dtype=torch.int64)
        torch.library.opcheck(torch.ops.torchpairwise._mmd_features.default,
            (x, lengths, 'rbf', -1., 3, 1., 1., 1., 1.5, True, True))


def test_ordered_preparation_reuse_and_validation(device):
    x = inputs('soft_dtw', 3, device)
    for metric, opname in (('soft_dtw', '_sequence_operator_cost'), ('mmd_statistic', '_mmd_features')):
        a = tp.lazy(x, metric=metric, chunk_size=1, **options(metric))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            value = a.sum()
            torch.autograd.grad(value, x)
        names = [event.name for event in profile.events()]
        if metric == 'soft_dtw':
            # One self preparation, nine cross tiles (replayed for backward).
            assert names.count('torchpairwise::' + opname) == 19
            assert names.count('torchpairwise::' + opname + '_backward') == 10
        else:
            assert names.count('torchpairwise::' + opname) == 1
            assert names.count('torchpairwise::' + opname + '_backward') == 1
        with torch.no_grad():
            x.mul_(.8)
        torch.testing.assert_close(a.sum(), tp.compare(x, metric=metric, **options(metric)).sum())
    for metric, kwargs in (('soft_dtw', {}), ('dtw', dict(gamma=.2)),
                           ('mmd_statistic', dict(estimator='invalid'))):
        with pytest.raises((TypeError, RuntimeError)):
            tp.lazy(x, metric=metric, **kwargs).sum()
    ids = inputs('levenshtein', 2, device)
    with pytest.raises(RuntimeError):
        tp.lazy(ids.bool(), metric='levenshtein').sum()
    with pytest.raises(TypeError):
        tp.lazy(ids, metric='indel', squared=True)
