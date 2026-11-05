"""Second derivatives of the supported native replay paths."""
import pytest
import torch
import torchpairwise as tp


CASES = [
    ('sqeuclidean', {}, False), ('euclidean', {}, False),
    ('manhattan', {}, False), ('minkowski', {'p': 3.}, False),
    ('cosine', {}, False),
    ('rbf', {'gamma': .4}, True), ('laplacian', {'gamma': .4}, True),
    ('linear', {}, True), ('polynomial', {'gamma': .4, 'coef0': 1., 'degree': 3}, True),
    ('sigmoid', {'gamma': .4, 'coef0': .2}, True),
    ('cosine_similarity', {}, True),
    *[('matern', {'length_scale': 1.3, 'nu': n}, True) for n in (.5, 1.5, 2.5)],
    ('rational_quadratic', {'length_scale': 1.3, 'alpha': .7}, True),
]


def inputs(device, dtype=torch.float64):
    torch.manual_seed(487)
    return (torch.randn(2, 3, device=device, dtype=dtype, requires_grad=True),
            (torch.randn(3, 3, device=device, dtype=dtype) + 2).requires_grad_())


@pytest.mark.parametrize('metric,options,kernel', CASES)
def test_dense_and_lazy_metrics_gradgrad(device, metric, options, kernel):
    x, y = inputs(device)
    name = 'cosine' if metric == 'cosine_similarity' else metric
    dense = (lambda a, b: tp.pairwise_kernels(a, b, kernel=name, **options)) if kernel else (
        lambda a, b: tp.pairwise_distances(a, b, metric=name, **options))
    lazy = lambda a, b: tp.lazy(a, b, metric, chunk_size=2, **options).sum()
    assert torch.autograd.gradgradcheck(dense, (x, y), fast_mode=True)
    assert torch.autograd.gradgradcheck(lazy, (x, y), fast_mode=True)


@pytest.mark.parametrize('consumer', ['matmul', 'sum', 'mean', 'logsumexp', 'softmax_matmul', 'topk', 'min', 'max'])
def test_streamed_consumers_against_dense(device, consumer):
    x, y = inputs(device)
    v = torch.randn(3, 2, device=device, dtype=x.dtype, requires_grad=True)
    cot = torch.randn(2, device=device, dtype=x.dtype, requires_grad=True)
    d = (x[:, None] - y[None]).square().sum(-1)
    a = tp.lazy(x, y, 'sqeuclidean', chunk_size=2)
    if consumer == 'matmul':
        actual, expected = a.matmul(v), d @ v
    elif consumer == 'sum':
        actual, expected = a.sum(-1), d.sum(-1)
    elif consumer == 'mean':
        actual, expected = a.mean(-1), d.mean(-1)
    elif consumer == 'logsumexp':
        actual, expected = a.logsumexp(-1), d.logsumexp(-1)
    elif consumer == 'softmax_matmul':
        actual, expected = a.softmax_matmul(v), d.softmax(-1) @ v
    elif consumer == 'topk':
        actual, expected = a.topk(2, largest=False).values, d.topk(2, largest=False).values
    elif consumer == 'min':
        actual, expected = a.min(-1).values, d.min(-1).values
    else:
        actual, expected = a.max(-1).values, d.max(-1).values
    # Cotangent also belongs to the differentiated graph.
    aa = (actual * (cot[:, None] if actual.ndim == 2 else cot)).sum()
    bb = (expected * (cot[:, None] if expected.ndim == 2 else cot)).sum()
    args = (x, y, v, cot) if consumer in ('matmul', 'softmax_matmul') else (x, y, cot)
    ga = torch.autograd.grad(aa, args, create_graph=True)
    gb = torch.autograd.grad(bb, args, create_graph=True)
    torch.testing.assert_close(ga, gb)
    ha = torch.autograd.grad(sum(g.square().sum() for g in ga), args, allow_unused=True)
    hb = torch.autograd.grad(sum(g.square().sum() for g in gb), args, allow_unused=True)
    for a, b in zip(ha, hb):
        torch.testing.assert_close(a, b)


def test_expression_rows_columns_and_penalty(device):
    x, y = inputs(device)
    rows = torch.randn(2, device=device, dtype=x.dtype, requires_grad=True)
    cols = torch.randn(3, device=device, dtype=x.dtype, requires_grad=True)
    def f(a, b, r, c):
        return (-.5 * tp.lazy(a, b, 'sqeuclidean', chunk_size=2)).add_rows(r).add_columns(c).exp().sum()
    assert torch.autograd.gradgradcheck(f, (x, y, rows, cols), fast_mode=True)
    grad, = torch.autograd.grad(f(x, y, rows, cols), x, create_graph=True)
    penalty = grad.norm()
    penalty.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    direction = torch.randn_like(x)
    grad, = torch.autograd.grad(f(x, y, rows, cols), x, create_graph=True)
    hvp, = torch.autograd.grad((grad * direction).sum(), x)
    d = (x[:, None] - y[None]).square().sum(-1)
    ref = (-.5 * d + rows[:, None] + cols[None, :]).exp().sum()
    g, = torch.autograd.grad(ref, x, create_graph=True)
    expected, = torch.autograd.grad((g * direction).sum(), x)
    torch.testing.assert_close(hvp, expected)


def test_first_order_repeatable(device):
    x, y = inputs(device)
    def run():
        return torch.autograd.grad(tp.lazy(x, y, 'rbf', gamma=.4, chunk_size=2).sum(), (x, y))
    for a, b in zip(run(), run()):
        assert torch.equal(a, b)


@pytest.mark.cuda
@pytest.mark.parametrize('consumer', ['sum', 'matmul', 'softmax_matmul'])
def test_cuda_fused_first_order_replayed_for_second(consumer, device):
    if device != 'cuda':
        pytest.skip('CUDA only')
    x, y = inputs(device, torch.float32)
    v = torch.randn(3, 2, device=device, requires_grad=True)
    a = tp.lazy(x, y, 'rbf', gamma=.4, chunk_size=2)
    d = (-.4 * (x[:, None] - y[None]).square().sum(-1)).exp()
    if consumer == 'sum':
        out, ref = a.sum(), d.sum()
    elif consumer == 'matmul':
        out, ref = a.matmul(v).square().sum(), (d @ v).square().sum()
    else:
        out, ref = a.softmax_matmul(v).square().sum(), (d.softmax(-1) @ v).square().sum()
    g, = torch.autograd.grad(out, x, create_graph=True)
    gr, = torch.autograd.grad(ref, x, create_graph=True)
    h, = torch.autograd.grad(g.square().sum(), x)
    hr, = torch.autograd.grad(gr.square().sum(), x)
    torch.testing.assert_close(h, hr, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed'])
@pytest.mark.parametrize('metric,options', [
    ('sqeuclidean', {}), ('matern', {'nu': 1.5, 'length_scale': 1.3})])
def test_dense_pairing_modes_gradgrad(device, mode, metric, options):
    x, _ = inputs(device)
    y = (x.detach() + .7).requires_grad_()
    if mode == 'condensed':
        def f(a):
            return tp.compare(a, metric=metric, mode=mode, **options)
        operands = (x,)
    else:
        def f(a, b):
            return tp.compare(a, b, metric=metric, mode=mode, **options)
        operands = (x, y)
    assert torch.autograd.gradgradcheck(f, operands, fast_mode=True)


@pytest.mark.parametrize('metric', ['canberra', 'angular'])
def test_unsupported_dense_higher_order_raises(device, metric):
    x, y = inputs(device)
    # The first-order path remains available; requesting a derivative graph
    # must report the limit rather than returning detached or zero gradients.
    out = tp.compare(x, y, metric=metric).sum()
    first = torch.autograd.grad(out, x)[0]
    assert torch.isfinite(first).all()
    out = tp.compare(x, y, metric=metric).sum()
    with pytest.raises(RuntimeError, match='Higher derivatives.*unsupported'):
        torch.autograd.grad(out, x, create_graph=True)


def test_dense_empty_higher_order_remains_connected(device):
    x = torch.empty(0, 3, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.ones(2, 3, dtype=torch.float64, device=device, requires_grad=True)
    out = tp.compare(x, y, metric='sqeuclidean').sum()
    gx, gy = torch.autograd.grad(out, (x, y), create_graph=True)
    assert gx.requires_grad and gy.requires_grad
    hx, hy = torch.autograd.grad(gx.sum() + gy.sum(), (x, y))
    torch.testing.assert_close(hx, torch.zeros_like(x))
    torch.testing.assert_close(hy, torch.zeros_like(y))


def test_self_comparison_hessian(device):
    x, _ = inputs(device)
    actual = tp.lazy(x, metric='rbf', gamma=.4, chunk_size=1).sum()
    expected = (-.4 * (x[:, None] - x[None]).square().sum(-1)).exp().sum()
    g, = torch.autograd.grad(actual, x, create_graph=True)
    ref, = torch.autograd.grad(expected, x, create_graph=True)
    torch.testing.assert_close(g, ref)
    h, = torch.autograd.grad(g.square().sum(), x)
    hr, = torch.autograd.grad(ref.square().sum(), x)
    torch.testing.assert_close(h, hr)


def test_weighted_mean_second_derivative(device):
    x, y = inputs(device)
    weights = (torch.rand(3, device=device, dtype=x.dtype) + .5).requires_grad_()
    dense = (x[:, None] - y[None]).square().sum(-1)
    actual = tp.lazy(x, y, 'sqeuclidean', chunk_size=2).weighted_mean(weights, dim=-1)
    expected = (dense * weights).sum(-1) / weights.sum()
    args = (x, y, weights)
    ga = torch.autograd.grad(actual.square().sum(), args, create_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), args, create_graph=True)
    torch.testing.assert_close(ga, gb)
    ha = torch.autograd.grad(sum(g.square().sum() for g in ga), args)
    hb = torch.autograd.grad(sum(g.square().sum() for g in gb), args)
    torch.testing.assert_close(ha, hb)


def test_softmax_operator_second_derivative(device):
    x, y = inputs(device)
    values = torch.randn(3, 2, device=device, dtype=x.dtype, requires_grad=True)
    dense = (x[:, None] - y[None]).square().sum(-1)
    actual = tp.lazy(x, y, 'sqeuclidean', chunk_size=2).softmax().matmul(values)
    expected = dense.softmax(-1) @ values
    args = (x, y, values)
    ga = torch.autograd.grad(actual.square().sum(), args, create_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), args, create_graph=True)
    ha = torch.autograd.grad(sum(g.square().sum() for g in ga), args)
    hb = torch.autograd.grad(sum(g.square().sum() for g in gb), args)
    torch.testing.assert_close(ga, gb)
    torch.testing.assert_close(ha, hb)
@pytest.mark.parametrize('estimator', ['biased', 'unbiased'])
@pytest.mark.parametrize('kernel,options', [('rbf', {'gamma': .4}), ('linear', {}), ('polynomial', {'gamma': .4, 'degree': 3})])
def test_mmd_statistics_gradgrad(device, estimator, kernel, options):
    torch.manual_seed(48)
    x = torch.randn(1, 3, 2, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.randn(1, 4, 2, dtype=torch.float64, device=device, requires_grad=True)
    fn = lambda a, b: tp.mmd_statistics(a, b, kernel=kernel, estimator=estimator, **options)
    assert torch.autograd.gradgradcheck(fn, (x, y), fast_mode=True)


def test_weighted_mean_gradgrad(device):
    x, y = inputs(device)
    w = torch.rand(3, dtype=torch.float64, device=device, requires_grad=True)
    fn = lambda a, b, c: tp.lazy(a, b, 'rbf', gamma=.4, chunk_size=2).weighted_mean(c)
    assert torch.autograd.gradgradcheck(fn, (x, y, w), fast_mode=True)


@pytest.mark.parametrize('method', ['min', 'max'])
def test_global_extrema_second_derivative(device, method):
    x, y = inputs(device)
    dense = (x[:, None] - y[None]).square().sum(-1)
    actual = getattr(tp.lazy(x, y, 'sqeuclidean', chunk_size=2), method)()
    expected = getattr(dense, method)()
    ga = torch.autograd.grad(actual, (x, y), create_graph=True)
    gb = torch.autograd.grad(expected, (x, y), create_graph=True)
    ha = torch.autograd.grad(sum(g.square().sum() for g in ga), (x, y))
    hb = torch.autograd.grad(sum(g.square().sum() for g in gb), (x, y))
    torch.testing.assert_close(ga, gb)
    torch.testing.assert_close(ha, hb)
