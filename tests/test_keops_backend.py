"""Optional KeOps selection, numerical parity and native fallback contracts."""
import pytest
import torch
import torchpairwise as tp
from torchpairwise import backends
from torchpairwise.backends import keops


LEAVES = [
    ('sqeuclidean', {}), ('euclidean', {}), ('manhattan', {}),
    ('chebyshev', {}), ('minkowski', {'p': 3.}),
    ('cosine_distances', {}), ('cosine_similarity', {}), ('linear', {}),
    ('polynomial', {}), ('sigmoid', {}), ('rbf', {}), ('laplacian', {}),
    ('rational_quadratic', {}), ('matern', {'nu': .5}),
    ('matern', {'nu': 1.5}), ('matern', {'nu': 2.5}),
]


@pytest.fixture
def installed_keops():
    pytest.importorskip('pykeops')
    assert keops.available()


def test_selection_nesting_and_missing_dependency(monkeypatch):
    monkeypatch.setattr(keops, 'available', lambda: False)
    assert backends._selection.get() == 'native'
    with pytest.raises(ValueError):
        with tp.backend('unknown'):
            pass
    with pytest.raises(ImportError, match=r'pip install torchpairwise\[keops\]'):
        with tp.backend('keops'):
            pass
    with pytest.raises(ImportError, match=r'pip install torchpairwise\[keops\]'):
        tp.lazy(torch.randn(3, 2), backend='keops')
    with tp.backend('auto'):
        assert backends._selection.get() == 'auto'
        with tp.backend('native'):
            assert backends._selection.get() == 'native'
        assert backends._selection.get() == 'auto'
        a = tp.lazy(torch.randn(3, 2))
        torch.testing.assert_close(a.sum(), a._native_evaluate('reduce', reduction='sum', dim=None))
    assert backends._selection.get() == 'native'


def test_leaf_override_transpose_and_conflicts(monkeypatch):
    monkeypatch.setattr(keops, 'available', lambda: True)
    seen = []
    def lower(*args, **kwargs):
        seen.append((args, kwargs))
        return NotImplemented
    monkeypatch.setattr(keops, '_lower', lower)
    x = torch.randn(3, 2)
    for metric in ('sqeuclidean', 'rbf'):
        a = tp.lazy(x, metric=metric, backend='keops')
        assert a.T._backend == 'keops'
        a.T.sum(-1)
        assert seen
        seen.clear()
        native = tp.lazy(x, metric=metric, backend='native')
        auto = tp.lazy(x, metric=metric, backend='auto')
        with tp.backend('keops'):
            native.T.sum(-1)
            (a + native).sum(-1)
            (a + auto).sum(-1)
        assert not seen
    with pytest.raises(ValueError):
        tp.lazy(x, backend='unknown')


@pytest.mark.parametrize('owner,name', [
    (torch.compiler, 'is_compiling'), (torch.jit, 'is_scripting'),
    (torch.jit, 'is_tracing'), (torch._C, '_are_functorch_transforms_active'),
])
def test_capture_guard_precedes_context(monkeypatch, owner, name):
    class ForbiddenContext:
        def get(self):
            raise AssertionError('capture must return before context access')
    monkeypatch.setattr(backends, '_selection', ForbiddenContext())
    monkeypatch.setattr(owner, name, lambda: True)
    assert backends._dispatch(object(), 'matmul', {}) is NotImplemented


@pytest.mark.compiler
def test_compile_export_never_lower(monkeypatch):
    monkeypatch.setattr(keops, 'available', lambda: True)
    def forbidden(*args, **kwargs):
        raise AssertionError('KeOps reached during deployment')
    monkeypatch.setattr(keops, '_lower', forbidden)
    class Model(torch.nn.Module):
        def forward(self, x, y, v):
            return tp.lazy(x, y, metric='rbf').matmul(v)
    args = (torch.randn(4, 2), torch.randn(5, 2), torch.randn(5, 3))
    model = Model()
    expected = model(*args)
    with tp.backend('keops'):
        torch.testing.assert_close(torch.compile(model, backend='eager', fullgraph=True)(*args), expected)
        torch.testing.assert_close(torch.export.export(model, args).module()(*args), expected)


def _compare(call, *, lowered=True, rtol=3e-5, atol=3e-6):
    with tp.backend('native'):
        expected = call()
    before = keops._stats['calls']
    with tp.backend('keops'):
        actual = call()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    if lowered:
        assert keops._stats['calls'] > before
    else:
        assert keops._stats['calls'] == before
    return actual


@pytest.mark.parametrize('metric,options', LEAVES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_leaf_consumers(installed_keops, device, dtype, metric, options):
    torch.manual_seed(320)
    x = torch.randn(6, 6, device=device, dtype=dtype)[:, ::2]
    y = torch.randn(7, 6, device=device, dtype=dtype)[:, ::2]
    a = tp.lazy(x, y, metric=metric, **options)
    # Optional tanh support is determined by the installed PyKeOps version.
    if metric == 'sigmoid':
        from pykeops.torch import LazyTensor
        if not hasattr(LazyTensor, 'tanh'):
            _compare(lambda: a.sum(-1), lowered=False)
            return
    for source in (a, a.T):
        v = torch.randn(source.shape[-1], 2, device=device, dtype=dtype)
        _compare(lambda: source.matmul(v))
        _compare(lambda: source.matmul(v[:, 0]))
        _compare(lambda: source.softmax_matmul(v))
        for dim in (-1, -2):
            for keepdim in (False, True):
                for name in ('sum', 'mean', 'logsumexp', 'amin', 'amax'):
                    _compare(lambda: getattr(source, name)(dim, keepdim=keepdim))
            for largest in (False, True):
                selected = _compare(lambda: source.topk(2, dim, largest=largest))
                assert selected.indices.dtype == torch.int64
            _compare(lambda: source.argmin(dim))
            _compare(lambda: source.argmax(dim))
        for name in ('sum', 'mean'):
            _compare(lambda: getattr(source, name)(keepdim=True))


@pytest.mark.parametrize('metric,options', [v for v in LEAVES if v[0] not in ('chebyshev', 'minkowski', 'sigmoid')])
def test_coincident_and_zero_gradients(installed_keops, device, metric, options):
    torch.manual_seed(321)
    x = torch.randn(4, 3, device=device, dtype=torch.float64)
    x[0] = 0
    x.requires_grad_()
    a = tp.lazy(x, metric=metric, **options)
    with tp.backend('native'):
        expected = a.sum()
        grad = torch.autograd.grad(expected, x)[0]
    with tp.backend('keops'):
        before = keops._stats['calls']
        actual = a.sum()
        actual_grad = torch.autograd.grad(actual, x)[0]
    assert keops._stats['calls'] > before
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(actual_grad, grad, rtol=1e-7, atol=1e-8)


def test_expression_scalar_weights_and_value_gradients(installed_keops, device):
    torch.manual_seed(322)
    def tensor(*shape):
        return torch.rand(shape, device=device, dtype=torch.float64, requires_grad=True)
    x, y, gamma, scalar = tensor(4, 3), tensor(5, 3), tensor(), tensor()
    rows, cols, values = tensor(4), tensor(5), tensor(4, 2)
    a = tp.lazy(x, y, metric='rbf', gamma=gamma)
    b = tp.lazy(x, y, metric='linear')
    expr = ((-(a * b + scalar - a)).square().exp()).weight_rows(rows).weight_columns(cols).T
    operands = (x, y, gamma, scalar, rows, cols, values)
    with tp.backend('native'):
        # Native metric schemas take real options; scalar algebra retains the
        # tensor parameter's gradient for the reference expression.
        native_a = (-gamma * tp.lazy(x, y, metric='sqeuclidean')).exp()
        native_expr = ((-(native_a * b + scalar - native_a)).square().exp()).weight_rows(rows).weight_columns(cols).T
        expected = native_expr.matmul(values)
        expected_grad = torch.autograd.grad(expected.sum(), operands)
    with tp.backend('keops'):
        before = keops._stats['calls']
        actual = expr.matmul(values)
        actual_grad = torch.autograd.grad(actual.sum(), operands)
    assert keops._stats['calls'] > before
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('consumer', ['matmul', 'softmax_matmul', 'sum', 'mean', 'logsumexp'])
def test_consumer_gradients(installed_keops, device, consumer):
    torch.manual_seed(324)
    x = torch.randn(4, 3, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.randn(5, 3, device=device, dtype=x.dtype, requires_grad=True)
    values = torch.randn(4, 2, device=device, dtype=x.dtype, requires_grad=True)
    a = tp.lazy(x, y, metric='rbf').T
    def evaluate():
        return getattr(a, consumer)(values) if consumer in ('matmul', 'softmax_matmul') else getattr(a, consumer)(-2)
    operands = (x, y, values) if consumer in ('matmul', 'softmax_matmul') else (x, y)
    with tp.backend('native'):
        expected = evaluate()
        expected_grad = torch.autograd.grad(expected.sum(), operands)
    with tp.backend('keops'):
        actual = evaluate()
        actual_grad = torch.autograd.grad(actual.sum(), operands)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-5, atol=5e-6)


@pytest.mark.parametrize('metric', ['polynomial', 'sigmoid', 'rational_quadratic', 'matern'])
def test_metric_scalar_gradients(installed_keops, device, metric):
    if metric == 'sigmoid':
        from pykeops.torch import LazyTensor
        if not hasattr(LazyTensor, 'tanh'):
            pytest.skip('Installed PyKeOps has no tanh')
    torch.manual_seed(325)
    x = torch.rand(4, 3, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(5, 3, device=device, dtype=x.dtype, requires_grad=True)
    first = torch.tensor(.7, device=device, dtype=x.dtype, requires_grad=True)
    second = torch.tensor(1.2, device=device, dtype=x.dtype, requires_grad=True)
    dot, d2 = x @ y.T, (x[:, None] - y[None]).square().sum(-1)
    if metric in ('polynomial', 'sigmoid'):
        options = dict(gamma=first, coef0=second)
        base = first * dot + second
        dense = base**3 if metric == 'polynomial' else base.tanh()
        operands = (x, y, first, second)
    elif metric == 'rational_quadratic':
        options = dict(alpha=first, length_scale=second)
        dense = (1 + d2 / (2 * first * second.square())).pow(-first)
        operands = (x, y, first, second)
    else:
        options = dict(length_scale=first)
        r = d2.sqrt() * 3**.5 / first
        dense = (1 + r) * (-r).exp()
        operands = (x, y, first)
    expected_grad = torch.autograd.grad(dense.sum(), operands)
    with tp.backend('keops'):
        actual = tp.lazy(x, y, metric=metric, **options).sum()
        actual_grad = torch.autograd.grad(actual, operands)
    torch.testing.assert_close(actual, dense.sum(), rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)


def test_gamma_formula_cache_and_cg(installed_keops, device):
    torch.manual_seed(323)
    x = torch.randn(6, 3, device=device, dtype=torch.float64)
    formulas = []
    for gamma in (.2, .7, torch.tensor(.4, device=device, dtype=x.dtype)):
        _compare(lambda: tp.lazy(x, metric='rbf', gamma=gamma).sum(-1))
        formulas.append(keops._stats['formula'])
    assert formulas[0] == formulas[1] == formulas[2]
    k = tp.lazy(x, metric='rbf')
    a = k + .5 * tp.eye_like(k)
    b = torch.randn(6, device=device, dtype=x.dtype)
    _compare(lambda: tp.linalg.cg(a, b, rtol=1e-10, max_iter=30).solution,
             lowered=False, rtol=1e-7, atol=1e-8)


def test_unsupported_consumers_fall_back(installed_keops, device):
    x = torch.rand(5, 3, device=device, dtype=torch.float64)
    a = tp.lazy(x, metric='rbf')
    v = torch.rand(5, 2, device=device, dtype=x.dtype)
    ids = torch.tensor([0, 2], device=device)
    calls = [lambda: a.sum(exclude_self=True), lambda: a.sum([-1]),
             lambda: a.softmax_matmul(v, exclude_self=True),
             lambda: a.matmul(v.unsqueeze(0)), lambda: a.diagonal(),
             lambda: a.block(ids, ids), lambda: a.median(-1),
             lambda: a.quantile(.4, -1), lambda: a.weighted_mean(x[:, 0]),
             lambda: a.frobenius_inner_product(a), lambda: a.hsic(a),
             lambda: a.bilinear(v, v), lambda: a.quadratic(v),
             lambda: a.displacement().sum(), lambda: a.residual_outer_product().sum()]
    calls += [lambda name=name: getattr(a, name)(-1) for name in
              ('var', 'std', 'count_nonzero', 'prod', 'all', 'any')]
    for call in calls:
        _compare(call, lowered=False)
    for source in (tp.as_lazy(x @ x.T), tp.lazy(x, metric='canberra'),
                   tp.lazy(x, metric='euclidean', w=torch.ones(3, device=device, dtype=x.dtype)),
                   tp.lazy(x.unsqueeze(0), metric='rbf'),
                   tp.lazy(x[:0], x, metric='rbf')):
        _compare(lambda: source.sum(-1), lowered=False)
    for metric, options in (('chebyshev', {}), ('minkowski', {'p': 3.})):
        source = tp.lazy(x.requires_grad_(), metric=metric, **options)
        _compare(lambda: source.sum(-1), lowered=False)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _compare(lambda: a.sum(-1), lowered=False)
    large = tp.lazy(torch.rand(66, 3, device=device, dtype=x.dtype), metric='rbf')
    _compare(lambda: large.topk(65), lowered=False)
    # Scalar tensors requiring grad also disallow forward-only lowering.
    p = torch.tensor(3., device=device, dtype=x.dtype, requires_grad=True)
    source = tp.lazy(x.detach(), metric='minkowski', p=p)
    _compare(lambda: source.sum(-1), lowered=False)


def test_auto_thresholds_and_ordering(installed_keops, device, monkeypatch):
    x = torch.randn(4, 3, device=device)
    y = torch.randn(5, 3, device=device)
    a = tp.lazy(x, y, metric='rbf')
    with tp.backend('native'):
        expected = a.sum(-1)
    for min_pairs, max_dim, lower in ((21, 3, False), (20, 2, False), (20, 3, device == 'cuda')):
        monkeypatch.setattr(keops, 'AUTO_MIN_PAIRS', min_pairs)
        monkeypatch.setattr(keops, 'AUTO_MAX_DIM', max_dim)
        before = keops._stats['calls']
        with tp.backend('auto'):
            actual = a.sum(-1)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        assert (keops._stats['calls'] > before) == lower
    before = keops._stats['calls']
    with tp.backend('auto'):
        tp.lazy(x.double(), y.double(), metric='rbf').sum(-1)
    assert keops._stats['calls'] == before
    with tp.backend('auto'):
        a.topk(2)
        a.amin(-1)
    assert keops._stats['calls'] == before


def test_singleton_axes_and_forward_only_no_grad(installed_keops, device):
    for n, m in ((1, 3), (3, 1)):
        x = torch.randn(n, 2, device=device)
        y = torch.randn(m, 2, device=device)
        a = tp.lazy(x, y, metric='rbf')
        _compare(lambda: a.sum(-1))
        _compare(lambda: a.T.matmul(torch.ones(n, device=device)))
        _compare(lambda: a.topk(1, -2, largest=False))
    x = torch.randn(3, 2, device=device, requires_grad=True)
    with torch.no_grad():
        for metric in ('chebyshev', 'minkowski'):
            a = tp.lazy(x, metric=metric, **({'p': 3.} if metric == 'minkowski' else {}))
            _compare(lambda: a.sum(-1))


def test_broadcast_custom_keops_opt_in(installed_keops, device):
    torch.manual_seed(323)
    x = torch.randn(6, 3, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.randn(5, 3, device=device, dtype=torch.float64, requires_grad=True)
    w = torch.rand(3, device=device, dtype=torch.float64, requires_grad=True)
    values = torch.randn(5, 2, device=device, dtype=torch.float64)
    ones = torch.ones(6, 1, device=device, dtype=torch.float64)
    # Asymmetric in its arguments, so a transposed leaf must keep fn's order.
    fn = lambda a, b, w: (-((a - b) * (a - b) * w).sum(-1)).exp() * (a * w).sum(-1).exp()
    opted = tp.make_pairwise(fn, form='broadcast', parameters=(w,), keops=True)
    plain = tp.make_pairwise(fn, form='broadcast', parameters=(w,))
    with tp.backend('native'):
        expected = (tp.lazy(x, y, plain) @ values, tp.lazy(x, y, plain).T @ ones)
        expected_grad = torch.autograd.grad(sum(e.sum() for e in expected), (x, y, w))
    with tp.backend('keops'):
        before = keops._stats['calls']
        actual = (tp.lazy(x, y, opted) @ values, tp.lazy(x, y, opted).T @ ones)
        actual_grad = torch.autograd.grad(sum(a.sum() for a in actual), (x, y, w))
        assert keops._stats['calls'] == before + 2
        # Without the opt-in, or when fn fails on LazyTensors, tiles stay native.
        tp.lazy(x, y, plain) @ values
        tp.lazy(x, y, tp.make_pairwise(lambda a, b: torch.nan_to_num(a - b).sum(-1), form='broadcast',
                                       keops=True)).sum(-1)
        assert keops._stats['calls'] == before + 2
    for got, want in zip((*actual, *actual_grad), (*expected, *expected_grad)):
        torch.testing.assert_close(got, want, rtol=1e-7, atol=1e-8)
    with pytest.raises(ValueError, match='keops=True'):
        tp.make_pairwise(fn, parameters=(w,), keops=True)
