"""CUDA fused reducers versus native tiles, with dense float64 error oracles."""
from contextlib import contextmanager

import pytest
import torch
import torchpairwise as tp


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available'),
]

FUSED_TOPK = True
OPS = torch.ops.torchpairwise
DTYPES = [torch.float32, torch.float64]
METRICS = [
    pytest.param('sqeuclidean_distances', {}, True, id='sqeuclidean'),
    pytest.param('euclidean_distances', {}, False, id='euclidean'),
    pytest.param('manhattan_distances', {}, True, id='manhattan'),
    pytest.param('chebyshev_distances', {}, True, id='chebyshev'),
    pytest.param('minkowski_distances', {'p': 3.}, True, id='minkowski-3'),
    pytest.param('minkowski_distances', {'p': .5}, True, id='minkowski-half'),
    pytest.param('minkowski_distances', {'p': 2.}, False, id='minkowski-2'),
    pytest.param('minkowski_distances', {'p': 1.}, True, id='minkowski-1'),
    pytest.param('minkowski_distances', {'p': 1 / 3}, True, id='minkowski-third'),
    pytest.param('rbf_kernel', {'gamma': .3}, True, id='rbf-gamma'),
    pytest.param('rbf_kernel', {}, True, id='rbf-default'),
    pytest.param('laplacian_kernel', {}, True, id='laplacian'),
    pytest.param('matern_kernel', {'nu': .5, 'length_scale': .7}, True, id='matern-half'),
    pytest.param('matern_kernel', {'nu': 1.5}, True, id='matern-1.5'),
    pytest.param('matern_kernel', {'nu': 2.5, 'length_scale': 2.}, True, id='matern-2.5'),
    pytest.param('rational_quadratic_kernel', {'alpha': .7, 'length_scale': 1.3},
                 True, id='rational-quadratic'),
    pytest.param('linear_kernel', {}, True, id='linear'),
    pytest.param('polynomial_kernel', {}, True, id='polynomial-default'),
    pytest.param('polynomial_kernel', {'degree': 2, 'gamma': .4, 'coef0': .3},
                 True, id='polynomial-2'),
    pytest.param('polynomial_kernel', {'degree': 5, 'coef0': 0.}, True, id='polynomial-5'),
    pytest.param('sigmoid_kernel', {'gamma': .2, 'coef0': -.1}, True, id='sigmoid'),
]
CONSUMERS = [
    'matmul_1', 'matmul_20', 'sum_rows', 'sum_columns', 'sum_all',
    'mean_rows', 'mean_columns', 'mean_all',
    'logsumexp_rows', 'logsumexp_columns', 'softmax', 'softmax_20',
]
EXPR_CONSUMERS = ['matmul_1', 'sum_rows', 'logsumexp_rows', 'softmax']
EXPRESSIONS = [
    pytest.param(lambda a: -0.7 * a, id='neg-scaled'),
    pytest.param(lambda a: 2. * a + 1, id='affine'),
    pytest.param(lambda a: 1 - a, id='integer-one-minus'),
    pytest.param(lambda a: -(a * .5) - 3, id='chain4'),
]


@contextmanager
def _fused(enabled):
    previous = OPS._set_fused_reducers(enabled)
    calls = [0]
    try:
        before = OPS._fused_reducer_calls()
        yield calls
    finally:
        try:
            torch.cuda.synchronize()
            calls[0] = OPS._fused_reducer_calls() - before
        finally:
            OPS._set_fused_reducers(previous)


def _run(enabled, fn):
    with _fused(enabled) as calls:
        result = fn()
    return result, calls[0]


def _assert_calls(calls, expected):
    if expected:
        assert calls > 0
    else:
        assert calls == 0


def _rand(shape, dtype, device='cuda', seed=0):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def _inputs(features, dtype, consumer, variant='plain', n=37, m=53):
    batch = (2,) if variant == 'batched' else ()
    x = _rand((*batch, n, features), dtype, seed=11)
    y = _rand((*((1,) if batch else ()), m, features), dtype, seed=12)
    references = n if variant in ('self', 'exclude', 'transposed') else m
    # 20 channels span two 16-channel kernel blocks.
    channels = 20 if consumer.endswith('_20') else 1
    v = _rand((*((1,) if batch else ()), references, channels), dtype, seed=13)
    return x, y, v


def _consume(a, v, consumer, *, dense=False, exclude=False):
    if consumer.startswith('matmul'):
        return a @ v if dense else a.matmul(v)
    if not dense:
        if consumer.startswith('softmax'):
            return a.softmax_matmul(v, exclude_self=exclude)
        operation, axis = consumer.rsplit('_', 1)
        dim = {'rows': -1, 'columns': -2, 'all': None}[axis]
        return getattr(a, operation)(dim=dim, exclude_self=exclude)
    operation, axis = (('softmax', 'rows') if consumer.startswith('softmax')
                       else consumer.rsplit('_', 1))
    dim = {'rows': -1, 'columns': -2, 'all': (-2, -1)}[axis]
    if exclude:
        diagonal = torch.eye(a.shape[-1], dtype=torch.bool, device=a.device)
        a = a.masked_fill(diagonal, -torch.inf if operation in ('logsumexp', 'softmax') else 0)
    if operation == 'softmax':
        return a.softmax(-1) @ v
    if operation == 'mean' and exclude:
        count = a.shape[dim] - 1 if isinstance(dim, int) else a.shape[-1] * (a.shape[-2] - 1)
        return a.sum(dim) / count
    return getattr(a, operation)(dim)


def _call(metric, options, consumer, variant='plain', *, dense=False,
          expression=None, chunk_size=None):
    def evaluate(x, y, v):
        a = tp.lazy(x, None if variant in ('self', 'exclude') else y,
                    metric, chunk_size=chunk_size, **options)
        if dense:
            a = a.to_dense()
        if variant == 'transposed':
            a = a.mT if dense else a.T
        if expression is not None:
            a = expression(a)
        return _consume(a, v, consumer, dense=dense, exclude=variant == 'exclude')
    return evaluate


def _evaluate(fn, args, enabled, gradients=False, grad_indices=None):
    grad_indices = tuple(range(len(args))) if grad_indices is None else grad_indices
    args = tuple(t.detach().clone().requires_grad_(gradients and i in grad_indices)
                 for i, t in enumerate(args))
    out, forward_calls = _run(enabled, lambda: fn(*args))
    if not gradients:
        return (out,), forward_calls, 0
    # Generate in float32 even for the float64 oracle: all runs get the same
    # random cotangent, with exactly representable values after conversion.
    cotangent = _rand(out.shape, torch.float32, device=out.device, seed=91).to(out.dtype)
    targets = tuple(args[i] for i in grad_indices)
    grads, backward_calls = _run(enabled, lambda: torch.autograd.grad(
        out, targets, grad_outputs=cotangent, allow_unused=True))
    grads = tuple(torch.zeros_like(t) if g is None else g for g, t in zip(grads, targets))
    return (out.detach(), *grads), forward_calls, backward_calls


def _assert_accuracy(native, fused, reference=None, *, equal_nan=False):
    assert fused.shape == native.shape
    assert fused.dtype == native.dtype
    if native.dtype == torch.float64 or equal_nan:
        # Summation order differs from the tiles (and from GEMM for dot
        # products): relative to the largest magnitude, not entrywise.
        finite = native.isfinite()
        scale = native[finite].abs().max().item() + 1 if finite.any() else 1
        tolerance = 1e-10 if native.dtype == torch.float64 else 1e-5
        torch.testing.assert_close(fused, native, rtol=tolerance, atol=tolerance * scale,
                                   equal_nan=True)
        return
    assert reference is not None
    assert reference.shape == fused.shape
    native_error = (native.double() - reference).abs().max().item()
    fused_error = (fused.double() - reference).abs().max().item()
    scale = reference.abs().max().item() + 1
    assert fused_error <= max(2 * native_error, 1e-5 * scale)


def _parity(fn, dense_fn, args, dtype, *, expected=True, gradients=False,
            backward_expected=None, grad_indices=None, equal_nan=False):
    native, forward_calls, backward_calls = _evaluate(
        fn, args, False, gradients, grad_indices)
    _assert_calls(forward_calls, False)
    _assert_calls(backward_calls, False)
    fused, forward_calls, backward_calls = _evaluate(
        fn, args, True, gradients, grad_indices)
    _assert_calls(forward_calls, expected)
    if gradients:
        _assert_calls(backward_calls, expected if backward_expected is None else backward_expected)
    reference = (None,) * len(native)
    if dtype == torch.float32 and not equal_nan:
        reference, forward_calls, backward_calls = _evaluate(
            dense_fn, tuple(t.double() for t in args), False, gradients, grad_indices)
        _assert_calls(forward_calls, False)
        _assert_calls(backward_calls, False)
    for a, b, r in zip(native, fused, reference):
        _assert_accuracy(a, b, r, equal_nan=equal_nan)
    return fused


def _fusable(metric, options, consumer, features):
    # fused_reducers.cpp's measured crossovers: tiles win beyond these.
    if metric.startswith('minkowski') and options.get('p') != 1. and features > 4:
        return False
    if metric.startswith('chebyshev') and features > 16:
        return False
    return not (consumer == 'matmul_20' and features > 4)


def _grid(dimensions):
    # Cross metrics and consumers fully; distribute shape variants over the D
    # boundaries instead of multiplying equivalent dispatcher paths again.
    return [pytest.param(consumer, features,
                         'self' if variant == 'exclude' and consumer.startswith('matmul') else variant,
                         id=f'{consumer}-D{features}-{variant}')
            for consumer in CONSUMERS for features, variant in dimensions]


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('metric,options,eligible', METRICS)
@pytest.mark.parametrize('consumer,features,variant', _grid([
    (1, 'plain'), (4, 'transposed'), (5, 'batched'),
    (16, 'exclude'), (17, 'self'), (64, 'plain'),
]))
def test_forward(metric, options, eligible, consumer, features, variant, dtype):
    args = _inputs(features, dtype, consumer, variant)
    eligible = eligible and _fusable(metric, options, consumer, features)
    _parity(_call(metric, options, consumer, variant),
            _call(metric, options, consumer, variant, dense=True), args, dtype,
            expected=eligible)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('metric,options,eligible', METRICS)
@pytest.mark.parametrize('consumer,features,variant', _grid([
    (1, 'plain'), (5, 'batched'), (16, 'exclude'), (17, 'transposed'), (5, 'self'),
]))
def test_gradients(metric, options, eligible, consumer, features, variant, dtype):
    args = _inputs(features, dtype, consumer, variant)
    eligible = eligible and _fusable(metric, options, consumer, features)
    _parity(_call(metric, options, consumer, variant),
            _call(metric, options, consumer, variant, dense=True), args, dtype,
            expected=eligible, gradients=True,
            backward_expected=eligible and features <= 16 and metric != 'chebyshev_distances'
            and not (metric == 'polynomial_kernel' and consumer.startswith('softmax')))


@pytest.mark.parametrize('consumer', ['matmul_1', 'sum_rows'])
def test_large_chunks(consumer):
    # Seven repeated queries keep the dense oracle small while both streamed
    # paths still process all 70000 x 70000 pairs and all reduction chunks.
    x = _rand((7, 3), torch.float32, seed=21).repeat(10000, 1)
    y = _rand((70000, 3), torch.float32, seed=22)
    v = _rand((70000, 1), torch.float32, seed=23)

    def dense(x, y, v):
        a = tp.lazy(x[:7], y, 'rbf').to_dense()
        return _consume(a, v, consumer, dense=True).repeat(10000, *([1] if consumer == 'matmul_1' else []))

    _parity(_call('rbf', {}, consumer), dense, (x, y, v), torch.float32)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('metric', ['sqeuclidean', 'rbf'])
@pytest.mark.parametrize('consumer', CONSUMERS)
def test_feature_limit_fallback(metric, consumer, dtype):
    args = _inputs(65, dtype, consumer)
    _parity(_call(metric, {}, consumer), _call(metric, {}, consumer, dense=True),
            args, dtype, expected=False, gradients=True)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('metric,options', [
    pytest.param('manhattan', {}, id='manhattan'),
    pytest.param('minkowski', {'p': 3.}, id='minkowski-3'),
    pytest.param('minkowski', {'p': .5}, id='minkowski-half'),
    pytest.param('matern', {'nu': .5}, id='matern-half'),
    pytest.param('matern', {'nu': 1.5}, id='matern-1.5'),
    pytest.param('matern', {'nu': 2.5}, id='matern-2.5'),
    pytest.param('laplacian', {}, id='laplacian'),
    pytest.param('rational_quadratic', {}, id='rational-quadratic'),
])
@pytest.mark.parametrize('consumer', EXPR_CONSUMERS)
@pytest.mark.parametrize('alias', [False, True])
def test_coincident_gradients(metric, options, consumer, alias, dtype):
    x, y, v = _inputs(4, dtype, consumer, n=37, m=37)
    x[1] = x[0]
    y[2] = x[0]
    y[3] = y[2]

    def call(x, y, v):
        return _consume(tp.lazy(x, x if alias else y, metric, **options), v, consumer)

    # These are native subgradient conventions, not a smooth dense oracle.
    _parity(call, None, (x, y, v), dtype, gradients=True, equal_nan=True)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('metric,options,eligible', METRICS)
@pytest.mark.parametrize('consumer', EXPR_CONSUMERS)
def test_nonfinite(metric, options, eligible, consumer, dtype):
    x, y, v = _inputs(5, dtype, consumer)
    x[2, 1] = torch.nan
    y[7, 0] = torch.inf
    _parity(_call(metric, options, consumer), None, (x, y, v), dtype,
            expected=eligible and _fusable(metric, options, consumer, 5), equal_nan=True)


def test_logsumexp_all_negative_infinity():
    x = torch.zeros(37, 3, device='cuda')
    y = torch.full((53, 3), 1e10, device='cuda')
    v = torch.ones(53, 1, device='cuda')
    fn = _call('sqeuclidean', {}, 'logsumexp_rows', expression=lambda a: -1e30 * a)
    (result,) = _parity(fn, None, (x, y, v), torch.float32, equal_nan=True)
    assert torch.isneginf(result).all()


_EDGE = 2e38  # finite, but twice it overflows float32
OVERFLOW_EDGES = [
    # A zero subgradient at delta == 0 whatever the (infinite) cotangent.
    pytest.param('manhattan', {}, [[0.]], [[0.]], [[_EDGE, _EDGE]], 'matmul',
                 id='manhattan-coincident'),
    # a * value overflows; native a * s^(a - 1) does not.
    pytest.param('minkowski', {'p': .5}, [[_EDGE]], [[0.]], None, 'sum',
                 id='minkowski-root'),
    # Underflowed kernel value times 2 * delta.
    pytest.param('rbf', {'gamma': 1.}, [[_EDGE]], [[0.]], None, 'sum', id='rbf-square'),
    # (1 / l)^2 overflows; native applies the factors one at a time.
    pytest.param('rational_quadratic', {'length_scale': 1e-20, 'alpha': 1.},
                 [[1.]], [[0.]], None, 'sum', id='rational-quadratic-scale'),
]


@pytest.mark.parametrize('metric,options,x,y,v,consumer', OVERFLOW_EDGES)
def test_backward_overflow_edges(metric, options, x, y, v, consumer):
    x, y = (torch.tensor(t, device='cuda') for t in (x, y))
    v = None if v is None else torch.tensor(v, device='cuda')

    def gradients(enabled):
        args = [t.clone().requires_grad_() for t in (x, y)]

        def run():
            a = tp.lazy(*args, metric, **options)
            out = a.matmul(v) if consumer == 'matmul' else a.sum()
            return (out, *torch.autograd.grad(out, args, torch.ones_like(out)))
        return _run(enabled, run)
    (native, _), (fused, calls) = gradients(False), gradients(True)
    assert calls > 0
    torch.testing.assert_close(fused, native, equal_nan=True)
    assert all(t.isfinite().all() for t in native)


def test_softmax_exclusion_keeps_non_finite_payload():
    # Native weighs the excluded pair 0 and multiplies: 0 * inf is NaN.
    x = torch.zeros(2, 1, device='cuda')
    v = torch.tensor([[torch.inf], [1.]], device='cuda')
    fn = _call('sqeuclidean', {}, 'softmax', 'exclude', expression=lambda a: -a)
    (result,) = _parity(fn, None, (x, x, v), torch.float32, equal_nan=True)
    assert result[0].isnan().all() and result[1].isinf().all()


def _selection_operator(pattern, dtype, exclude):
    generator = torch.Generator(device='cuda').manual_seed(31)
    x = torch.randint(0, 3, (67, 5), device='cuda', generator=generator).to(dtype)
    y = torch.randint(0, 3, (67, 5), device='cuda', generator=generator).to(dtype)
    x[1] = x[0]
    y[0] = x[0]
    y[1] = y[0]
    if pattern == 'nan':
        x[2] = torch.nan
        y[7] = torch.nan
    a = tp.lazy(x, None if exclude else y, 'sqeuclidean')
    return -a if pattern == 'signed_zero' else a


def _assert_exact(native, fused):
    assert native.shape == fused.shape and native.dtype == fused.dtype
    if native.is_floating_point():
        nan = native.isnan()
        assert torch.equal(nan, fused.isnan())
        assert torch.equal(native[~nan], fused[~nan])
        torch.testing.assert_close(fused, native, rtol=0, atol=0, equal_nan=True)
        zeros = (native == 0) & ~nan
        assert torch.equal(native[zeros].signbit(), fused[zeros].signbit())
    else:
        assert torch.equal(native, fused)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('pattern', ['ties', 'nan', 'signed_zero'])
@pytest.mark.parametrize('k', [1, 8, 64, 'eligible'])
@pytest.mark.parametrize('largest', [False, True])
@pytest.mark.parametrize('exclude', [False, True])
@pytest.mark.parametrize('dim', [-1, -2])
def test_topk_exact(pattern, k, largest, exclude, dim, dtype):
    a = _selection_operator(pattern, dtype, exclude)
    k = 67 - int(exclude) if k == 'eligible' else k
    fn = lambda: a.topk(k, dim=dim, largest=largest, exclude_self=exclude)
    native, calls = _run(False, fn)
    _assert_calls(calls, False)
    fused, calls = _run(True, fn)
    if FUSED_TOPK:
        _assert_calls(calls, k <= 64)
    _assert_exact(native.values, fused.values)
    _assert_exact(native.indices, fused.indices)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('pattern', ['ties', 'nan', 'signed_zero'])
@pytest.mark.parametrize('operation', ['min', 'max', 'argmin', 'argmax'])
@pytest.mark.parametrize('exclude', [False, True])
@pytest.mark.parametrize('dim', [-1, -2])
def test_extrema_exact(pattern, operation, exclude, dim, dtype):
    a = _selection_operator(pattern, dtype, exclude)
    fn = lambda: getattr(a, operation)(dim=dim, exclude_self=exclude)
    native, calls = _run(False, fn)
    _assert_calls(calls, False)
    fused, calls = _run(True, fn)
    if FUSED_TOPK:
        _assert_calls(calls, True)
    if operation.startswith('arg'):
        _assert_exact(native, fused)
    else:
        _assert_exact(native.values, fused.values)
        _assert_exact(native.indices, fused.indices)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('metric', ['sqeuclidean', 'rbf'])
@pytest.mark.parametrize('expression', EXPRESSIONS)
@pytest.mark.parametrize('consumer', EXPR_CONSUMERS)
def test_expressions(metric, expression, consumer, dtype):
    args = _inputs(5, dtype, consumer)
    _parity(_call(metric, {}, consumer, expression=expression),
            _call(metric, {}, consumer, expression=expression, dense=True),
            args, dtype, gradients=True)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('requires_grad', [False, True])
@pytest.mark.parametrize('consumer', EXPR_CONSUMERS)
def test_tensor_constant_fallback(consumer, requires_grad, dtype):
    args = (*_inputs(5, dtype, consumer), torch.tensor(2., device='cuda', dtype=dtype))

    def call(dense):
        def evaluate(x, y, v, constant):
            a = tp.lazy(x, y, 'rbf')
            a = a.to_dense() if dense else a
            return _consume(constant * a, v, consumer, dense=dense)
        return evaluate

    _parity(call(False), call(True), args, dtype, expected=False, gradients=True,
            grad_indices=(0, 1, 2, 3) if requires_grad else (0, 1, 2))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('consumer', EXPR_CONSUMERS)
def test_cpu_fallback(consumer, dtype):
    args = tuple(t.cpu() for t in _inputs(5, dtype, consumer))
    _parity(_call('rbf', {}, consumer), _call('rbf', {}, consumer, dense=True),
            args, dtype, expected=False, gradients=True)


@pytest.mark.parametrize('consumer', EXPR_CONSUMERS)
def test_autocast_fallback(consumer):
    args = _inputs(5, torch.float16, consumer)

    def call(dense):
        fn = _call('sqeuclidean', {}, consumer, dense=dense, expression=lambda a: 2. * a)

        def evaluate(*args):
            with torch.autocast('cuda'):
                return fn(*args)
        return evaluate

    # Streaming reductions and softmax_matmul cast half operands once and run
    # with autocast excluded (float32 throughout), so they fuse; logsumexp
    # keeps autocast active and falls back.
    _parity(call(False), call(True), args, torch.float32,
            expected=consumer != 'logsumexp_rows')


@pytest.mark.parametrize('dtype', DTYPES)
def test_zero_features_fallback(dtype):
    args = _inputs(0, dtype, 'sum_rows')
    fn = _call('sqeuclidean', {}, 'sum_rows')
    try:
        native, calls, _ = _evaluate(fn, args, False)
    except (ValueError, RuntimeError) as error:
        if not any(word in str(error).lower() for word in ('feature', 'dimension', 'size', 'empty')):
            raise
        pytest.skip(f'The native API rejects D=0: {error}')
    _assert_calls(calls, False)
    fused, calls, _ = _evaluate(fn, args, True)
    _assert_calls(calls, False)
    reference, _, _ = _evaluate(_call('sqeuclidean', {}, 'sum_rows', dense=True),
                                tuple(t.double() for t in args), False)
    _assert_accuracy(native[0], fused[0], reference[0])


def _masks(rows, columns, batch=()):
    generator = torch.Generator(device='cuda').manual_seed(43)
    row_mask = torch.rand((*batch, rows), device='cuda', generator=generator) > .2
    column_mask = torch.rand((columns,), device='cuda', generator=generator) > .2
    row_mask[..., 0] = column_mask[0] = True
    return row_mask, column_mask


def _where(condition, a, other):
    from torchpairwise.ops._operator_algebra import ScalarExpression
    return ScalarExpression('where', condition, (other, a))


# (lazy, dense) pairs of expressions over row values r and column values c;
# masks select rows/columns as Sinkhorn plans do.
AXIS_EXPRESSIONS = [
    pytest.param(lambda a, r, c, m: a.add_rows(r).add_columns(c),
                 lambda a, r, c, m: a + r[..., :, None] + c[..., None, :], id='offsets'),
    pytest.param(lambda a, r, c, m: (2. * a).weight_rows(r).weight_columns(c),
                 lambda a, r, c, m: 2. * a * r[..., :, None] * c[..., None, :], id='weights'),
    pytest.param(lambda a, r, c, m: (a - a.column_values(c)) / 3,
                 lambda a, r, c, m: (a - c[..., None, :]) / 3, id='sub-div'),
    pytest.param(lambda a, r, c, m: (a.row_values(r) - a) * a.column_values(c),
                 lambda a, r, c, m: (r[..., :, None] - a) * c[..., None, :], id='sub-from-mul'),
    pytest.param(lambda a, r, c, m: (-a / .5).add_rows(r).exp(),
                 lambda a, r, c, m: (-a / .5 + r[..., :, None]).exp(), id='exp'),
    pytest.param(lambda a, r, c, m: (-a).add_rows(r).add_columns(c).where(
                     a.row_values(m[0]) & a.column_values(m[1]), -torch.inf).exp(),
                 lambda a, r, c, m: torch.where(m[0][..., :, None] & m[1][..., None, :],
                                                -a + r[..., :, None] + c[..., None, :],
                                                -torch.inf).exp(), id='sinkhorn-plan'),
    pytest.param(lambda a, r, c, m: _where(a.column_values(m[1]), a.add_rows(r), .5),
                 lambda a, r, c, m: torch.where(m[1][..., None, :], .5, a + r[..., :, None]),
                 id='drop'),
]


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('variant', ['plain', 'transposed', 'batched'])
@pytest.mark.parametrize('expression,dense_expression', AXIS_EXPRESSIONS)
@pytest.mark.parametrize('consumer', EXPR_CONSUMERS + ['sum_columns', 'matmul_20'])
def test_row_column_expressions(consumer, expression, dense_expression, variant, dtype):
    x, y, v = _inputs(4 if consumer == 'matmul_20' else 5, dtype, consumer, variant)
    rows, columns = (53, 37) if variant == 'transposed' else (37, 53)
    batch = (2,) if variant == 'batched' else ()
    r = _rand((*batch, rows), dtype, seed=44)
    c = _rand((columns,), dtype, seed=45)
    masks = _masks(rows, columns, batch)

    def call(dense):
        def evaluate(x, y, v, r, c):
            a = tp.lazy(x, y, 'rbf')
            a = a.to_dense() if dense else a
            if variant == 'transposed':
                a = a.mT if dense else a.T
            a = (dense_expression if dense else expression)(a, r, c, masks)
            return _consume(a, v, consumer, dense=dense)
        return evaluate

    _parity(call(False), call(True), (x, y, v, r, c), dtype, gradients=True)


@pytest.mark.parametrize('largest', [False, True])
@pytest.mark.parametrize('operation', ['topk', 'argmin'])
def test_row_column_selection_exact(operation, largest):
    a = _selection_operator('ties', torch.float32, False)
    r = _rand((67,), torch.float32, seed=46).round()
    c = _rand((67,), torch.float32, seed=47).round()
    masks = _masks(67, 67)
    b = a.add_rows(r).weight_columns(c).where(a.column_values(masks[1]), torch.inf)
    fn = ((lambda: b.topk(8, largest=largest)) if operation == 'topk'
          else (lambda: b.argmin(dim=-2)))
    native, calls = _run(False, fn)
    _assert_calls(calls, False)
    fused, calls = _run(True, fn)
    _assert_calls(calls, True)
    if operation == 'topk':
        _assert_exact(native.values, fused.values)
        _assert_exact(native.indices, fused.indices)
    else:
        _assert_exact(native, fused)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('consumer', EXPR_CONSUMERS)
def test_explicit_chunk_size(consumer, dtype):
    args = _inputs(5, dtype, consumer)
    _parity(_call('rbf', {}, consumer, chunk_size=7),
            _call('rbf', {}, consumer, dense=True), args, dtype)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('metric', ['linear', 'polynomial', 'sigmoid'])
@pytest.mark.parametrize('operation', ['topk', 'argmax'])
def test_dot_product_selection_keeps_tiles(metric, operation, dtype):
    # Native forms dot products with a GEMM; fused sums would round
    # differently and could reorder near ties, so selections stay on tiles.
    x, y, _ = _inputs(5, dtype, 'sum_rows')
    a = tp.lazy(x, y, metric)
    fn = (lambda: a.topk(8, dim=-1)) if operation == 'topk' else (lambda: a.argmax(dim=-1))
    native, calls = _run(False, fn)
    _assert_calls(calls, False)
    fused, calls = _run(True, fn)
    _assert_calls(calls, False)
    if operation == 'argmax':
        _assert_exact(native, fused)
    else:
        _assert_exact(native.values, fused.values)
        _assert_exact(native.indices, fused.indices)


@pytest.mark.parametrize('consumer', EXPR_CONSUMERS)
def test_attention_scores(consumer):
    # Scaled dot-product attention: (Q K^T) / sqrt(D) under softmax_matmul.
    x, y, v = _inputs(16, torch.float32, consumer)
    expression = lambda a: a * 16 ** -.5
    _parity(_call('linear', {}, consumer, expression=expression),
            _call('linear', {}, consumer, expression=expression, dense=True),
            (x, y, v), torch.float32, gradients=True)


@pytest.mark.parametrize('dtype', DTYPES)
def test_global_logsumexp_fallback(dtype):
    args = _inputs(5, dtype, 'sum_all')
    _parity(_call('rbf', {}, 'logsumexp_all'),
            _call('rbf', {}, 'logsumexp_all', dense=True), args, dtype,
            expected=False, gradients=True)


@pytest.mark.compiler
def test_compile_forward_backward():
    from torch._inductor import config

    args = _inputs(5, torch.float32, 'matmul_1')
    fn = _call('rbf', {}, 'matmul_1')
    dense = _call('rbf', {}, 'matmul_1', dense=True)
    eager = _parity(fn, dense, args, torch.float32, gradients=True)
    torch.compiler.reset()
    try:
        with config.patch(compile_threads=1):
            compiled = torch.compile(fn, fullgraph=True)
            actual = _parity(compiled, dense, args, torch.float32, gradients=True)
        for result, expected in zip(actual, eager):
            torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)
    finally:
        torch.compiler.reset()


@pytest.mark.compiler
def test_export_forward():
    class Matmul(torch.nn.Module):
        def forward(self, x, y, v):
            return tp.lazy(x, y, 'rbf').matmul(v)

    args = _inputs(5, torch.float32, 'matmul_1')
    module = Matmul()
    dense = _call('rbf', {}, 'matmul_1', dense=True)
    eager = _parity(module, dense, args, torch.float32)
    exported = torch.export.export(module, args).module()
    actual = _parity(exported, dense, args, torch.float32)
    torch.testing.assert_close(actual[0], eager[0], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('dtype', DTYPES)
def test_info_nce_scores(dtype):
    # NT-Xent denominators: the same tensor on both sides with exclude_self.
    x, _, v = _inputs(8, dtype, 'logsumexp_rows', n=37, m=37)

    def call(x, v):
        return (tp.lazy(x, x, 'linear') / .2).logsumexp(-1, exclude_self=True)

    _parity(call, call, (x, v), dtype, gradients=True, grad_indices=(0,))
