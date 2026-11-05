"""Independent scalar references for the experimental user-function adapter."""
import itertools

import pytest
import torch
import torchpairwise as tp

from torchpairwise import make_pairwise


def _reference(fn, x, y=None, *, mode='all_pairs', sample_ndim=1):
    y = x if y is None else y
    rank = sample_ndim + 1
    batch = torch.broadcast_shapes(x.shape[:-rank], y.shape[:-rank])
    x = x.expand(*batch, *x.shape[-rank:])
    y = y.expand(*batch, *y.shape[-rank:])
    n, m = x.shape[-rank], y.shape[-rank]
    results = []
    for b in itertools.product(*(range(length) for length in batch)):
        if mode == 'all_pairs':
            values = torch.stack([torch.stack([fn(x[b][i], y[b][j])
                                              for j in range(m)]) for i in range(n)])
        elif mode == 'paired':
            values = torch.stack([fn(x[b][i], y[b][i]) for i in range(n)])
        else:
            values = torch.stack([fn(x[b][i], x[b][j])
                                  for i in range(n) for j in range(i + 1, n)])
        results.append(values)
    return torch.stack(results).reshape(*batch, *results[0].shape)


@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed'])
@pytest.mark.parametrize('chunk_size', [None, 1, 5])
def test_custom_values_batches_noncontiguous_and_gradients(mode, chunk_size, device):
    torch.manual_seed(610)
    x = torch.randn(2, 1, 4, 4, 2, dtype=torch.float64, device=device).transpose(-1, -2)
    y = torch.randn(1, 3, 4 if mode == 'paired' else 3, 4, 2,
                    dtype=x.dtype, device=device).transpose(-1, -2)
    x, y = x.requires_grad_(), y.requires_grad_()
    weight = torch.rand(2, 4, dtype=x.dtype, device=device, requires_grad=True)

    def metric(a, b):
        return ((a - b).square() * weight).sum()

    adapted = make_pairwise(metric, mode=mode, sample_ndim=2,
                            chunk_size=chunk_size, symmetric=True)
    args = (x,) if mode == 'condensed' else (x, y)
    differentiable = (*args, weight)
    expected = _reference(metric, *args, mode=mode, sample_ndim=2)
    actual = adapted(*args)
    torch.testing.assert_close(actual, expected)
    for grad, ref in zip(torch.autograd.grad(actual.sum(), differentiable),
                         torch.autograd.grad(expected.sum(), differentiable)):
        torch.testing.assert_close(grad, ref)


def test_custom_image_samples_and_self_comparison(device):
    x = torch.randn(2, 3, 2, 3, 4, device=device)
    y = torch.randn(1, 4, 2, 3, 4, device=device)

    def image_metric(a, b):
        # The function receives [C,H,W], with every sample axis intact.
        return (a.mean((-1, -2)) - b.mean((-1, -2))).square().sum()

    compare = make_pairwise(image_metric, sample_ndim=3, chunk_size=5)
    torch.testing.assert_close(compare(x, y), _reference(image_metric, x, y, sample_ndim=3))
    torch.testing.assert_close(compare(x), _reference(image_metric, x, sample_ndim=3))


@pytest.mark.parametrize('mode', ['all_pairs', 'paired'])
def test_custom_distinct_sample_extents_belong_to_metric(mode, device):
    x = torch.randn(2, 1, 3, 4, 2, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, 2, 3, 7, 2, device=device, dtype=x.dtype, requires_grad=True)
    metric = lambda a, b: (a.mean(0) - b.mean(0)).square().sum()
    fn = make_pairwise(metric, mode=mode, sample_ndim=2, chunk_size=5)
    actual = fn(x, y)
    expected = _reference(metric, x, y, mode=mode, sample_ndim=2)
    torch.testing.assert_close(actual, expected)
    for grad, ref in zip(torch.autograd.grad(actual.sum(), (x, y)),
                         torch.autograd.grad(expected.sum(), (x, y))):
        torch.testing.assert_close(grad, ref)


def test_custom_asymmetric_function_and_condensed_order(device):
    x = torch.arange(15, dtype=torch.float64, device=device).reshape(5, 3)
    asymmetric = lambda a, b: a.sum() - 2 * b.sum()
    torch.testing.assert_close(make_pairwise(asymmetric)(x), _reference(asymmetric, x))
    # A symmetric encoding makes every unordered pair identifiable.
    encode = lambda a, b: (a[0] + 1) * (b[0] + 1)
    actual = make_pairwise(encode, mode='condensed', symmetric=True, chunk_size=3)(x)
    expected = torch.stack([encode(x[i], x[j]) for i in range(5) for j in range(i + 1, 5)])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed'])
def test_custom_pair_tiles_are_bounded_across_batches(mode, device, monkeypatch):
    actual_vmap = torch.vmap
    tile_shapes = []

    def inspected_vmap(fn, **kwargs):
        mapped = actual_vmap(fn, **kwargs)

        def call(a, b):
            tile_shapes.append((a.shape, b.shape))
            return mapped(a, b)

        return call

    monkeypatch.setattr(torch, 'vmap', inspected_vmap)
    x = torch.randn(2, 7, 3, device=device)
    adapted = make_pairwise(lambda a, b: (a - b).square().sum(), mode=mode,
                            symmetric=True, chunk_size=5)
    result = adapted(x, x) if mode == 'paired' else adapted(x)
    assert sum(a[0] for a, _ in tile_shapes) == result.numel()
    assert all(a == b and 0 < a[0] <= 5 and a[1:] == (3,) for a, b in tile_shapes)


@pytest.mark.parametrize('mode,n', [('all_pairs', 0), ('paired', 0),
                                   ('condensed', 0), ('condensed', 1)])
def test_custom_empty_never_probes_metric_and_preserves_input_gradients(mode, n, device):
    def forbidden(a, b):
        raise AssertionError('an empty comparison must not call the metric')

    x = torch.full((2, n, 3), float('nan'), device=device, requires_grad=True)
    y = torch.full((1, 3 if mode == 'all_pairs' else n, 3), float('nan'),
                   device=device, requires_grad=True)
    fn = make_pairwise(forbidden, mode=mode, symmetric=True, output_dtype=torch.float64)
    args = (x,) if mode == 'condensed' else (x, y)
    result = fn(*args)
    assert result.shape == ((2, 0, 3) if mode == 'all_pairs' else (2, 0))
    assert result.dtype == torch.float64 and result.device == x.device
    for grad, value in zip(torch.autograd.grad(result.sum(), args), args):
        torch.testing.assert_close(grad, torch.zeros_like(value))


@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed'])
def test_custom_empty_batch_and_unused_capture(mode, device):
    x = torch.empty(0, 3, 4, device=device, requires_grad=True)
    y = torch.ones(1, 3, 4, device=device, requires_grad=True)
    captured = torch.ones((), device=device, requires_grad=True)
    fn = make_pairwise(lambda a, b: captured * (a - b).sum(), mode=mode, symmetric=True)
    result = fn(x) if mode == 'condensed' else fn(x, y)
    assert result.shape == ((0, 3, 3) if mode == 'all_pairs' else (0, 3))
    dx, dc = torch.autograd.grad(result.sum(), (x, captured), allow_unused=True)
    torch.testing.assert_close(dx, torch.zeros_like(x))
    assert dc is None


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_custom_dtype_contract_and_explicit_accumulation(dtype, device):
    x = torch.ones(2, 3, device=device, dtype=dtype, requires_grad=True)
    metric = lambda a, b: (a.float() - 2 * b.float()).square().sum()
    fn = make_pairwise(metric, output_dtype=torch.float32)
    actual = fn(x)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, _reference(metric, x))
    assert torch.autograd.grad(actual.sum(), x)[0].dtype == dtype
    if dtype != torch.float32:
        with pytest.raises(TypeError, match='scalar dtype'):
            make_pairwise(metric)(x)


def test_custom_autocast_respects_declared_output_dtype(device):
    low = torch.float16 if device == 'cuda' else torch.bfloat16
    x = torch.randn(3, 4, device=device)

    def metric(a, b):
        return (a.unsqueeze(0) @ b.unsqueeze(1)).reshape(())

    with torch.autocast(device, dtype=low):
        actual = make_pairwise(metric, output_dtype=low)(x)
        torch.testing.assert_close(actual, _reference(metric, x))
        with pytest.raises(TypeError, match='scalar dtype'):
            make_pairwise(metric)(x)
    assert actual.dtype == low


@pytest.mark.parametrize('fn,error,message', [
    (lambda a, b: 1.0, TypeError, 'scalar Tensor'),
    (lambda a, b: (a - b).sum().reshape(1), ValueError, 'zero dimensions'),
    (lambda a, b: ((a - b).sum(),), TypeError, 'scalar Tensor'),
    (lambda a, b: (a - b).double().sum(), TypeError, 'scalar dtype'),
    (lambda a, b: torch.zeros((), device='meta'), ValueError, 'input device'),
])
def test_custom_scalar_contract(fn, error, message, device):
    with pytest.raises(error, match=message):
        make_pairwise(fn)(torch.ones(2, 3, device=device))


@pytest.mark.parametrize('kwargs,error', [
    ({'mode': 'bad'}, ValueError), ({'mode': 'condensed'}, ValueError),
    ({'sample_ndim': 0}, ValueError), ({'sample_ndim': True}, ValueError),
    ({'chunk_size': 0}, ValueError), ({'chunk_size': True}, ValueError),
    ({'compile': 'yes'}, TypeError), ({'symmetric': 1}, TypeError),
    ({'output_dtype': torch.int64}, TypeError),
])
def test_custom_factory_validation(kwargs, error):
    with pytest.raises(error):
        make_pairwise(lambda a, b: (a - b).sum(), **kwargs)
    with pytest.raises(TypeError, match='callable'):
        make_pairwise(None)


def test_custom_input_validation(device):
    metric = lambda a, b: (a - b).sum()
    fn = make_pairwise(metric)
    x = torch.ones(2, 3, device=device)
    for other, error, message in [
        ([1, 2], TypeError, 'tensors'),
        (x[0], ValueError, 'sample-count'),
        (x.double(), ValueError, 'same dtype and device'),
        (x.to_sparse(), TypeError, 'dense strided'),
        (x.long(), TypeError, 'floating point'),
        (torch.ones(2, 3, device='meta'), ValueError, 'same dtype and device'),
    ]:
        with pytest.raises(error, match=message):
            fn(x, other)
    with pytest.raises(ValueError, match='batch dimensions'):
        fn(x.expand(2, 2, 3), x.expand(3, 2, 3))
    with pytest.raises(TypeError, match='both input'):
        make_pairwise(metric, mode='paired')(x)
    with pytest.raises(ValueError, match='equal sample counts'):
        make_pairwise(metric, mode='paired')(x, x[:1])
    with pytest.raises(TypeError, match='one input'):
        make_pairwise(metric, mode='condensed', symmetric=True)(x, x)


def test_custom_rejects_randomness_and_data_dependent_python(device):
    x = torch.ones(2, 3, device=device)
    with pytest.raises(RuntimeError, match='random'):
        make_pairwise(lambda a, b: torch.rand_like(a).sum())(x)
    with pytest.raises(RuntimeError, match='item|vmap'):
        make_pairwise(lambda a, b: a if a.sum().item() else b)(x)


@pytest.mark.compiler
@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed'])
def test_custom_compiled_first_gradients_and_captured_parameter(mode, device):
    torch.compiler.reset()
    try:
        x = torch.randn(2, 4, 3, device=device, requires_grad=True)
        y = torch.randn(1, 4, 3, device=device, requires_grad=True)
        weight = torch.rand(3, device=device, requires_grad=True)
        metric = lambda a, b: ((a - b).square() * weight).sum()
        fn = make_pairwise(metric, mode=mode, symmetric=True, chunk_size=5, compile=True)
        args = (x,) if mode == 'condensed' else (x, y)
        actual = fn(*args)
        expected = _reference(metric, *args, mode=mode)
        torch.testing.assert_close(actual, expected)
        for grad, ref in zip(torch.autograd.grad(actual.sum(), (*args, weight)),
                             torch.autograd.grad(expected.sum(), (*args, weight))):
            torch.testing.assert_close(grad, ref, rtol=1e-4, atol=1e-5)
        # Reuse the factory's compiler wrapper after another parameter value.
        with torch.no_grad():
            weight.add_(.25)
        torch.testing.assert_close(fn(*args), _reference(metric, *args, mode=mode))
    finally:
        torch.compiler.reset()


def test_custom_creates_one_compiler_wrapper_per_factory(device, monkeypatch):
    calls = []

    def compile_spy(fn, **kwargs):
        calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, 'compile', compile_spy)
    fn = make_pairwise(lambda a, b: (a - b).square().sum(), chunk_size=2, compile=True)
    x = torch.ones(3, 4, device=device)
    fn(x)
    fn(x[:2])
    assert calls == [{'fullgraph': True, 'dynamic': True}]


@pytest.mark.compiler
def test_custom_compiler_failure_is_explicit(device):
    torch.compiler.reset()
    try:
        @torch.compiler.disable
        def metric(a, b):
            return (a - b).square().sum()

        x = torch.ones(2, 3, device=device)
        fn = make_pairwise(metric, compile=True)
        with pytest.raises(torch._dynamo.exc.Unsupported):
            fn(x)
    finally:
        torch.compiler.reset()


def test_custom_four_modes_explicit_surrogate_vjp(device):
    x = torch.randn(2, 3, 4, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.randn(1, 5, 4, dtype=x.dtype, device=device, requires_grad=True)
    w = torch.rand(4, dtype=x.dtype, device=device, requires_grad=True)
    def forward(a, b, weight):
        return ((a - b).square() * weight).sum()
    def vjp(g, a, b, weight):
        delta = a - b
        return 5*g*weight*delta, -5*g*weight*delta, 3*g*delta.square()
    metric = make_pairwise(forward, backward=vjp, parameters=(w,), symmetric=True, chunk_size=7)
    for mode in ('all_pairs', 'paired', 'condensed', 'single'):
        a, b = (x[:, 0], y[:, 0]) if mode == 'single' else (x, y[:, :3] if mode == 'paired' else y)
        args = (a,) if mode == 'condensed' else (a, b)
        if mode == 'all_pairs':
            expected = ((a.unsqueeze(-2)-b.unsqueeze(-3)).square()*w).sum(-1)
        elif mode == 'paired' or mode == 'single':
            expected = ((a-b).square()*w).sum(-1)
        else:
            ids = torch.triu_indices(3, 3, 1, device=device)
            expected = ((a[:, ids[0]]-a[:, ids[1]]).square()*w).sum(-1)
        actual = metric(*args, mode=mode)
        torch.testing.assert_close(actual, expected)
        variables = (a, w) if mode == 'condensed' else (a, b, w)
        upstream = torch.randn_like(actual)
        grads = torch.autograd.grad(actual, variables, upstream)
        oracle = torch.autograd.grad(expected, variables, upstream)
        for k, (g, ref) in enumerate(zip(grads, oracle)):
            torch.testing.assert_close(g, ref*(3 if k == len(variables)-1 else 2.5))
    with pytest.raises(TypeError, match='both input'):
        metric(x, mode='single')


def test_custom_streaming_shared_consumers_and_parameter_vjps(device):
    x = torch.randn(2, 3, 4, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.randn(1, 5, 4, dtype=x.dtype, device=device, requires_grad=True)
    w = torch.rand(4, dtype=x.dtype, device=device, requires_grad=True)
    offset = torch.tensor(0.3, dtype=x.dtype, device=device, requires_grad=True)
    def forward(a, b, weight, bias):
        return ((a-b).square()*weight).sum()+bias
    def vjp(g, a, b, weight, bias):
        delta = a-b
        return 5*g*weight*delta, -5*g*weight*delta, 3*g*delta.square(), g
    metric = make_pairwise(forward, backward=vjp, parameters=(w, offset))
    A = tp.lazy(x, y, metric, chunk_size=2)
    dense = ((x.unsqueeze(-2)-y.unsqueeze(-3)).square()*w).sum(-1)+offset
    coefficients = torch.randn(5, 2, dtype=x.dtype, device=device)
    actual = A.mean(dim=-1)+A.matmul(coefficients).sum(-1)
    expected = dense.mean(-1)+(dense@coefficients).sum(-1)
    torch.testing.assert_close(actual, expected)
    grads = torch.autograd.grad(actual.sum(), (x,y,w,offset))
    oracle = torch.autograd.grad(expected.sum(), (x,y,w,offset))
    for scale, g, ref in zip((2.5,2.5,3.,1.), grads, oracle):
        torch.testing.assert_close(g, ref*scale)
    torch.testing.assert_close(A.T.sum(dim=-1), dense.detach().sum(-2))
    torch.testing.assert_close(A.var(dim=(-2,-1), correction=0), dense.detach().var((-2,-1), correction=0))
    torch.testing.assert_close(A.topk(2, largest=False)[0], dense.detach().topk(2, largest=False).values)
    torch.testing.assert_close(A.quantile(0.7), dense.detach().quantile(0.7, dim=-1))
    builtin = __import__('torchpairwise').lazy(x, y, metric='sqeuclidean', chunk_size=2)
    torch.testing.assert_close((A.square()+builtin).sum(), (dense.detach().square()+(x.unsqueeze(-2)-y.unsqueeze(-3)).square().sum(-1)).sum((-2,-1)))
    hidden = make_pairwise(lambda a,b: ((a-b).square()*w).sum())
    with pytest.raises(ValueError, match='explicit parameters'):
        tp.lazy(x, y, hidden)


def test_custom_registered_tile_protocol_native_dense_and_stream(device):
    lib = torch.library.Library('tp_custom_protocol_test', 'FRAGMENT')
    lib.define('forward(Tensor x, Tensor y, Tensor[] parameters) -> Tensor')
    lib.define('backward(Tensor g, Tensor x, Tensor y, Tensor[] parameters) -> Tensor[]')
    def forward(x,y,parameters):
        return ((x-y).square()*parameters[0]).sum(-1)
    def backward(g,x,y,parameters):
        delta = x-y
        value = 5*g.unsqueeze(-1)*delta*parameters[0]
        return [value,-value,(3*g.unsqueeze(-1)*delta.square()).sum(0)]
    lib.impl('forward',forward,'CompositeExplicitAutograd')
    lib.impl('backward',backward,'CompositeExplicitAutograd')
    x = torch.randn(2,3,4,device=device,dtype=torch.float64,requires_grad=True)
    y = torch.randn(1,5,4,device=device,dtype=x.dtype,requires_grad=True)
    w = torch.rand(4,device=device,dtype=x.dtype,requires_grad=True)
    metric = make_pairwise('tp_custom_protocol_test::forward', backward='tp_custom_protocol_test::backward', parameters=(w,), symmetric=True, chunk_size=7)
    expected = ((x.unsqueeze(-2)-y.unsqueeze(-3)).square()*w).sum(-1)
    actual = metric(x,y)
    torch.testing.assert_close(actual,expected)
    g = torch.randn_like(actual)
    grads = torch.autograd.grad(actual,(x,y,w),g)
    refs = torch.autograd.grad(expected,(x,y,w),g)
    for scale,value,ref in zip((2.5,2.5,3.),grads,refs):
        torch.testing.assert_close(value,ref*scale)
    torch.testing.assert_close(tp.lazy(x, y, metric, chunk_size=2).sum(), expected.detach().sum((-2, -1)))
    torch.testing.assert_close(metric(x[:,0],y[:,0],mode='single'), ((x[:,0]-y[:,0]).square()*w).sum(-1))
    torch.testing.assert_close(metric(x,y[:,:3],mode='paired'), ((x-y[:,:3]).square()*w).sum(-1))
    ids = torch.triu_indices(3,3,1,device=device)
    torch.testing.assert_close(metric(x,mode='condensed'), ((x[:,ids[0]]-x[:,ids[1]]).square()*w).sum(-1))


@pytest.mark.compiler
def test_custom_autograd_function_compiled_streaming_edge_tiles(device):
    # Dense gathers and partial Cartesian tiles used to produce enough stride /
    # grad-mode specializations to exhaust Dynamo's default compilation cache.
    class SquaredDistance(torch.autograd.Function):
        generate_vmap_rule = True

        @staticmethod
        def forward(x, y):
            return (x - y).square().sum()

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(*inputs)

        @staticmethod
        def backward(ctx, gradient):
            x, y = ctx.saved_tensors
            dx = 2 * gradient * (x - y)
            return dx, -dx

    torch.compiler.reset()
    try:
        x = torch.randn(3, 4, device=device, requires_grad=True)
        y = torch.randn(5, 4, device=device, requires_grad=True)
        metric = make_pairwise(SquaredDistance.apply, compile=True)
        expected = (x[:, None] - y[None]).square().sum(-1)
        torch.testing.assert_close(metric(x, y), expected)
        loss = tp.lazy(x, y, metric, chunk_size=2).mean()
        torch.testing.assert_close(loss, expected.mean())
        for got, want in zip(torch.autograd.grad(loss, (x, y)),
                             torch.autograd.grad(expected.mean(), (x, y))):
            torch.testing.assert_close(got, want)
    finally:
        torch.compiler.reset()


def test_custom_registered_operator_reregistration():
    # User libraries can unload and register the same name again. A cached
    # external TypedOperatorHandle used to outlive its registration and crash.
    x = torch.randn(2, 4, requires_grad=True)
    y = torch.randn(3, 4, requires_grad=True)
    for factor in (1., 2.):
        lib = torch.library.Library('torchpairwise_lifecycle_test', 'FRAGMENT')
        try:
            lib.define('pairs(Tensor x, Tensor y, Tensor[] parameters) -> Tensor')
            lib.define('vjp(Tensor g, Tensor x, Tensor y, Tensor[] parameters) -> Tensor[]')

            def forward(a, b, parameters):
                return factor * (a - b).square().sum(-1)

            def backward(g, a, b, parameters):
                dx = 3 * factor * g[:, None] * (a - b)
                return [dx, -dx]

            lib.impl('pairs', forward, 'CompositeImplicitAutograd')
            lib.impl('vjp', backward, 'CompositeImplicitAutograd')
            metric = make_pairwise('torchpairwise_lifecycle_test::pairs',
                                   backward='torchpairwise_lifecycle_test::vjp')
            delta = x[:, None] - y[None]
            expected = factor * delta.square().sum(-1)
            torch.testing.assert_close(metric(x, y), expected)
            loss = tp.lazy(x, y, metric, chunk_size=2).mean()
            torch.testing.assert_close(loss, expected.mean())
            dx, dy = torch.autograd.grad(loss, (x, y))
            torch.testing.assert_close(dx, 3 * factor * delta.sum(1) / 6)
            torch.testing.assert_close(dy, -3 * factor * delta.sum(0) / 6)
        finally:
            lib._destroy()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed', 'single'])
def test_custom_broadcast_scalar_parity(mode, dtype, device):
    torch.manual_seed(721)
    x = torch.randn(2, 1, 3, 2, 4, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn(1, 2, 3 if mode == 'paired' else 5, 2, 4,
                    device=device, dtype=dtype, requires_grad=True)
    weight = torch.rand(2, 4, device=device, dtype=dtype, requires_grad=True)
    scalar = make_pairwise(lambda a, b, w: ((a-b).square()*w).sum(),
                           parameters=(weight,), sample_ndim=2, symmetric=True, chunk_size=5)
    broadcast = make_pairwise(lambda a, b, w: ((a-b).square()*w).sum((-2, -1)),
                              form='broadcast', parameters=(weight,), sample_ndim=2,
                              symmetric=True, chunk_size=5)
    a, b = (x[:, :, 0], y[:, :, 0]) if mode == 'single' else (x, y)
    args = (a,) if mode == 'condensed' else (a, b)
    actual, expected = broadcast(*args, mode=mode), scalar(*args, mode=mode)
    torch.testing.assert_close(actual, expected)
    variables = (*args, weight)
    upstream = torch.randn_like(actual)
    for got, want in zip(torch.autograd.grad(actual, variables, upstream),
                         torch.autograd.grad(expected, variables, upstream)):
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize('consumer', ['sum', 'mean', 'matmul', 'topk', 'logsumexp',
                                      'var', 'softmax_matmul', 'diagonal', 'block'])
def test_custom_broadcast_streaming_consumers(consumer, device):
    torch.manual_seed(722)
    x = torch.randn(2, 3, 4, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, 5, 4, device=device, dtype=x.dtype, requires_grad=True)
    weight = torch.rand(4, device=device, dtype=x.dtype, requires_grad=True)
    values = torch.randn(5, 2, device=device, dtype=x.dtype, requires_grad=True)
    metric = make_pairwise(lambda a, b, w: ((a-b).square()*w).sum(-1),
                           form='broadcast', parameters=(weight,))
    source = tp.lazy(x, y, metric, chunk_size=2)
    dense = ((x.unsqueeze(-2)-y.unsqueeze(-3)).square()*weight).sum(-1)
    if consumer == 'sum':
        actual, expected = source.sum(-1), dense.sum(-1)
    elif consumer == 'mean':
        actual, expected = source.mean(), dense.mean((-2, -1))
    elif consumer == 'matmul':
        actual, expected = source.matmul(values), dense @ values
    elif consumer == 'topk':
        actual, indices = source.topk(2, largest=False)
        expected, expected_indices = dense.topk(2, largest=False)
        torch.testing.assert_close(indices, expected_indices)
    elif consumer == 'logsumexp':
        actual, expected = source.logsumexp(-1), dense.logsumexp(-1)
    elif consumer == 'var':
        actual, expected = source.var(correction=0), dense.var((-2, -1), correction=0)
    elif consumer == 'softmax_matmul':
        actual, expected = source.softmax_matmul(values), dense.softmax(-1) @ values
    elif consumer == 'diagonal':
        actual, expected = source.diagonal(), dense.diagonal(dim1=-2, dim2=-1)
    else:
        rows = torch.tensor([2, 0], device=device)
        cols = torch.tensor([4, 1, 3], device=device)
        actual = source.block(rows, cols)
        expected = dense.index_select(-2, rows).index_select(-1, cols)
    torch.testing.assert_close(actual, expected)
    variables = (x, y, weight, values) if consumer in ('matmul', 'softmax_matmul') else (x, y, weight)
    upstream = torch.randn_like(actual)
    for got, want in zip(torch.autograd.grad(actual, variables, upstream),
                         torch.autograd.grad(expected, variables, upstream)):
        torch.testing.assert_close(got, want)


def test_custom_broadcast_backward_operand_shapes(device):
    calls = []

    def forward(a, b, weight):
        return ((a-b).abs()*weight).sum(-1)

    def backward(g, a, b, weight):
        calls.append((a.shape, b.shape, g.shape))
        delta = (a-b).sign()*g.unsqueeze(-1)*weight
        return delta.sum_to_size(a.shape), -delta.sum_to_size(b.shape), ((a-b).abs()*g.unsqueeze(-1)).sum_to_size(weight.shape)

    x = torch.randn(3, 4, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.randn(5, 4, device=device, dtype=x.dtype, requires_grad=True)
    weight = torch.rand(4, device=device, dtype=x.dtype, requires_grad=True)
    metric = make_pairwise(forward, form='broadcast', backward=backward,
                           parameters=(weight,), chunk_size=4)
    expected = ((x[:, None]-y[None]).abs()*weight).sum(-1)
    for actual, reference in [(metric(x, y), expected),
                              (tp.lazy(x, y, metric, chunk_size=2).sum(-1), expected.sum(-1))]:
        torch.testing.assert_close(actual, reference)
        for got, want in zip(torch.autograd.grad(actual.sum(), (x, y, weight)),
                             torch.autograd.grad(reference.sum(), (x, y, weight), retain_graph=True)):
            torch.testing.assert_close(got, want)
    assert any(len(a) == 3 and a[1] == 1 and b[0] == 1 and g == (a[0], b[1])
               for a, b, g in calls)

    bad = make_pairwise(forward, form='broadcast', parameters=(weight,),
                        backward=lambda g, a, b, w: (torch.zeros_like(a).reshape(-1),
                                                     torch.zeros_like(b), torch.zeros_like(w)))
    with pytest.raises((ValueError, RuntimeError), match='shape'):
        torch.autograd.grad(bad(x, y).sum(), (x, y, weight))


@pytest.mark.parametrize('fn,error,message', [
    (lambda a, b: (a-b).sum(), ValueError, 'shape'),
    (lambda a, b: (a-b).double().sum(-1), TypeError, 'dtype'),
    (lambda a, b: torch.empty(torch.broadcast_shapes(a.shape[:-1], b.shape[:-1]), device='meta'),
     ValueError, 'device'),
    (lambda a, b: (a-b).sum(-1).to_sparse(), TypeError, 'strided'),
])
def test_custom_broadcast_output_contract(fn, error, message, device):
    with pytest.raises(error, match=message):
        make_pairwise(fn, form='broadcast')(torch.ones(2, 3, device=device))


def test_custom_broadcast_tiles_do_not_vmap_or_flatten_pairs(device, monkeypatch):
    shapes = []

    def forbidden(*args, **kwargs):
        raise AssertionError('broadcast form must not vmap')

    def forward(a, b):
        shapes.append((a.shape, b.shape))
        assert a.ndim == b.ndim == 3
        assert a.shape[1] == b.shape[0] == 1
        assert a.shape[0]*b.shape[1] <= 5
        return (a-b).square().sum(-1)

    monkeypatch.setattr(torch, 'vmap', forbidden)
    x = torch.randn(7, 3, device=device)
    y = torch.randn(4, 3, device=device)
    metric = make_pairwise(forward, form='broadcast', chunk_size=5)
    expected = (x[:, None]-y[None]).square().sum(-1)
    torch.testing.assert_close(metric(x, y), expected)
    torch.testing.assert_close(tp.lazy(x, y, metric, chunk_size=2).sum(-1), expected.sum(-1))
    assert shapes


@pytest.mark.compiler
def test_custom_broadcast_compiled_dense_and_streaming(device):
    torch.compiler.reset()
    try:
        x = torch.randn(3, 4, device=device, requires_grad=True)
        y = torch.randn(5, 4, device=device, requires_grad=True)
        metric = make_pairwise(lambda a, b: (a-b).square().sum(-1),
                               form='broadcast', compile=True, chunk_size=2)
        expected = (x[:, None]-y[None]).square().sum(-1)
        torch.testing.assert_close(metric(x, y), expected)
        actual = tp.lazy(x, y, metric, chunk_size=2).mean()
        torch.testing.assert_close(actual, expected.mean())
        for got, want in zip(torch.autograd.grad(actual, (x, y)),
                             torch.autograd.grad(expected.mean(), (x, y))):
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
    finally:
        torch.compiler.reset()


def test_custom_broadcast_registered_native_protocol(device):
    lib = torch.library.Library('tp_custom_broadcast_test', 'FRAGMENT')
    try:
        lib.define('forward(Tensor x, Tensor y, Tensor[] parameters) -> Tensor')
        lib.define('backward(Tensor g, Tensor x, Tensor y, Tensor[] parameters) -> Tensor[]')

        def forward(a, b, parameters):
            return ((a-b).square()*parameters[0]).sum(-1)

        def backward(g, a, b, parameters):
            delta = 2*g.unsqueeze(-1)*(a-b)*parameters[0]
            return [delta.sum_to_size(a.shape), -delta.sum_to_size(b.shape),
                    (g.unsqueeze(-1)*(a-b).square()).sum_to_size(parameters[0].shape)]

        lib.impl('forward', forward, 'CompositeExplicitAutograd')
        lib.impl('backward', backward, 'CompositeExplicitAutograd')
        x = torch.randn(3, 4, device=device, dtype=torch.float64, requires_grad=True)
        y = torch.randn(5, 4, device=device, dtype=x.dtype, requires_grad=True)
        weight = torch.rand(4, device=device, dtype=x.dtype, requires_grad=True)
        metric = make_pairwise('tp_custom_broadcast_test::forward', form='broadcast',
                               backward='tp_custom_broadcast_test::backward', parameters=(weight,),
                               symmetric=True)
        expected = ((x[:, None]-y[None]).square()*weight).sum(-1)
        dense = metric(x, y)
        torch.testing.assert_close(dense, expected)
        for got, want in zip(torch.autograd.grad(dense.sum(), (x, y, weight)),
                             torch.autograd.grad(expected.sum(), (x, y, weight), retain_graph=True)):
            torch.testing.assert_close(got, want)
        torch.testing.assert_close(metric(x, y[:3], mode='paired'),
                                   ((x-y[:3]).square()*weight).sum(-1))
        torch.testing.assert_close(metric(x[0], y[0], mode='single'),
                                   ((x[0]-y[0]).square()*weight).sum(-1))
        ids = torch.triu_indices(3, 3, 1, device=device)
        torch.testing.assert_close(metric(x, mode='condensed'),
                                   ((x[ids[0]]-x[ids[1]]).square()*weight).sum(-1))
        actual = tp.lazy(x, y, metric, chunk_size=2).sum(-1)
        torch.testing.assert_close(actual, expected.sum(-1))
        for got, want in zip(torch.autograd.grad(actual.sum(), (x, y, weight)),
                             torch.autograd.grad(expected.sum(), (x, y, weight))):
            torch.testing.assert_close(got, want)
    finally:
        lib._destroy()


@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed'])
def test_custom_broadcast_empty_never_calls_function(mode, device):
    def forbidden(a, b):
        raise AssertionError('empty broadcast comparison must not call fn')

    x = torch.empty(2, 0, 3, device=device, requires_grad=True)
    y = torch.empty(1, 4 if mode == 'all_pairs' else 0, 3,
                    device=device, requires_grad=True)
    metric = make_pairwise(forbidden, form='broadcast', symmetric=True,
                           output_dtype=torch.float64)
    args = (x,) if mode == 'condensed' else (x, y)
    actual = metric(*args, mode=mode)
    assert actual.shape == ((2, 0, 4) if mode == 'all_pairs' else (2, 0))
    assert actual.dtype == torch.float64 and actual.device == x.device
    for got, value in zip(torch.autograd.grad(actual.sum(), args), args):
        torch.testing.assert_close(got, torch.zeros_like(value))


def test_custom_broadcast_invalid_form():
    with pytest.raises((TypeError, ValueError), match='form'):
        make_pairwise(lambda a, b: (a-b).sum(-1), form='unknown')


def test_custom_broadcast_streaming_multiaxis_samples_and_batches(device):
    x = torch.randn(2, 1, 3, 2, 4, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, 2, 5, 2, 4, device=device, dtype=x.dtype, requires_grad=True)
    weight = torch.rand(2, 4, device=device, dtype=x.dtype, requires_grad=True)
    metric = make_pairwise(lambda a, b, w: ((a-b).square()*w).sum((-2, -1)),
                           form='broadcast', sample_ndim=2, parameters=(weight,))
    actual = tp.lazy(x, y, metric, chunk_size=2).sum(-1)
    expected = ((x.unsqueeze(-3)-y.unsqueeze(-4)).square()*weight).sum((-2, -1)).sum(-1)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    for got, want in zip(torch.autograd.grad(actual, (x, y, weight), upstream),
                         torch.autograd.grad(expected, (x, y, weight), upstream)):
        torch.testing.assert_close(got, want)


class _ModuleMetric(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(3, device=device, dtype=torch.float64))
        self.register_buffer('scale', torch.tensor(0.5, device=device, dtype=torch.float64))

    def forward(self, a, b):
        return ((a-b).square()*self.weight).sum(-1)*self.scale


def test_custom_module_broadcast_streaming_parameters_and_buffer(device):
    module = _ModuleMetric(device).eval()
    x = torch.randn(4, 3, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.randn(5, 3, device=device, dtype=torch.float64, requires_grad=True)
    metric = make_pairwise(module, form='broadcast')
    assert metric.parameters == tuple(module.parameters())
    expected = module(x[:, None], y[None])
    torch.testing.assert_close(metric(x, y), expected)
    actual = tp.lazy(x, y, metric, chunk_size=3).sum()
    torch.testing.assert_close(actual, expected.sum())
    operands = (x, y, module.weight)
    for got, want in zip(torch.autograd.grad(actual, operands),
                         torch.autograd.grad(expected.sum(), operands)):
        torch.testing.assert_close(got, want)
    assert not module.training
    with pytest.raises(TypeError, match='omit parameters'):
        make_pairwise(module, parameters=(module.weight,))


def test_custom_module_scalar_and_explicit_backward(device):
    module = _ModuleMetric(device)
    x = torch.randn(3, 3, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.randn(4, 3, device=device, dtype=torch.float64, requires_grad=True)
    scale = module.scale

    def backward(g, a, b, weight):
        delta = 2*g*(a-b)*weight*scale
        return delta, -delta, g*(a-b).square()*scale

    metric = make_pairwise(module, backward=backward)
    actual = tp.lazy(x, y, metric, chunk_size=3).sum()
    expected = module(x[:, None], y[None]).sum()
    operands = (x, y, module.weight)
    for got, want in zip(torch.autograd.grad(actual, operands),
                         torch.autograd.grad(expected, operands)):
        torch.testing.assert_close(got, want)
    assert module.training


def test_custom_module_hidden_backward_parameter_still_rejected(device):
    module = _ModuleMetric(device)
    hidden = torch.ones(3, device=device, dtype=torch.float64, requires_grad=True)

    def backward(g, a, b, weight):
        return g*a*hidden, g*b, g*weight

    metric = make_pairwise(module, backward=backward, form='broadcast')
    x = torch.randn(3, 3, device=device, dtype=torch.float64)
    with pytest.raises(ValueError, match='explicit parameters'):
        tp.lazy(x, x, metric)


@pytest.mark.compiler
def test_custom_module_compiled_broadcast(device):
    torch.compiler.reset()
    try:
        module = _ModuleMetric(device)
        x = torch.randn(3, 3, device=device, dtype=torch.float64)
        y = torch.randn(4, 3, device=device, dtype=torch.float64)
        metric = make_pairwise(module, form='broadcast', compile=True)
        actual = tp.lazy(x, y, metric, chunk_size=3).sum()
        expected = module(x[:, None], y[None]).sum()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, module.weight)[0],
                                   torch.autograd.grad(expected, module.weight)[0])
        assert module.training
    finally:
        torch.compiler.reset()
