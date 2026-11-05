"""Full-graph compilation through the native forwards and their backwards."""
import pytest
import torch
import torchpairwise as tp
from test_metrics import parameters

pytestmark = pytest.mark.compiler


@pytest.fixture(autouse=True)
def reset_compiler_cache():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


# Native opcheck covers each primitive; exercise representative public routes
# here rather than every metric x comparison mode x compiler combination.
METRICS = ['euclidean', 'canberra', 'wminkowski', 'mahalanobis']
KERNELS = ['linear', 'rbf', 'laplacian', 'chi2']


def compare_forward_backward(fn, args, backend='aot_eager', dynamic=True):
    expected = fn(*args)
    expected_grads = torch.autograd.grad(expected.sum(), args)
    compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=dynamic)
    actual = compiled(*args)
    actual_grads = torch.autograd.grad(actual.sum(), args)
    torch.testing.assert_close(actual, expected)
    for grad, ref in zip(actual_grads, expected_grads):
        torch.testing.assert_close(grad, ref, atol=1e-5, rtol=1e-4)
    return compiled


@pytest.mark.parametrize('metric', METRICS)
@pytest.mark.parametrize('paired', [False, True])
def test_compiled_distances_and_parameter_gradients(metric, paired, device):
    d = 2 if metric == 'haversine' else 5
    x = (torch.rand(2, 1, 3, d, device=device) + .4).requires_grad_()
    y = (torch.rand(1, 2, 3, d, device=device) + .6).requires_grad_()
    kwargs = parameters(metric, x)
    parameter_keys = [key for key, value in kwargs.items() if isinstance(value, torch.Tensor)]
    args = (x, y, *(kwargs[key].requires_grad_() for key in parameter_keys))
    scalars = {key: value for key, value in kwargs.items() if key not in parameter_keys}
    fn = tp.paired_distances if paired else tp.cdist

    def call(a, b, *params):
        return fn(a, b, metric=metric, **scalars, **dict(zip(parameter_keys, params)))

    compiled = compare_forward_backward(call, args)
    # Different sample counts must execute the same dynamic graph correctly.
    next_args = (x[..., :2, :], y[..., :2, :], *args[2:])
    torch.testing.assert_close(compiled(*next_args), call(*next_args))


@pytest.mark.parametrize('kernel', KERNELS)
@pytest.mark.parametrize('paired', [False, True])
def test_compiled_kernels(kernel, paired, device):
    x, y = [(torch.rand(3, 5, device=device) + .2).requires_grad_() for _ in range(2)]
    fn = tp.paired_kernels if paired else tp.pairwise_kernels
    compare_forward_backward(lambda a, b: fn(a, b, kernel=kernel), (x, y))


def test_compiled_hausdorff(device):
    x = torch.rand(1, 2, 3, 2, 4, device=device, requires_grad=True)
    y = torch.rand(2, 3, 3, 3, 2, device=device, requires_grad=True)
    compare_forward_backward(lambda a, b: tp.directed_hausdorff_distances(
        a, b, batched=True, feature_dim=2)[0], (x, y), backend='inductor')


@pytest.mark.parametrize('metric', ['hamming', 'jaccard', 'yule'])
def test_compiled_boolean_metrics(metric, device):
    x, y = [torch.rand(3, 5, device=device) > .5 for _ in range(2)]
    fn = lambda a, b: tp.cdist(a, b, metric=metric)
    compiled = torch.compile(fn, backend='aot_eager', fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(x, y), fn(x, y), equal_nan=True)


def test_compiled_autocast(device):
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    x, y = [torch.rand(3, 5, device=device, dtype=dtype).requires_grad_() for _ in range(2)]

    def call(a, b):
        with torch.autocast(device, dtype=dtype):
            return tp.canberra_distances(a, b) + tp.rbf_kernel(a, b)

    compare_forward_backward(call, (x, y))


def test_compiled_native_autocast_cosine_identity(device):
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    x = torch.zeros(3, 5, device=device, dtype=dtype, requires_grad=True)

    def call(a):
        with torch.autocast(device, dtype=dtype):
            return (torch.ops.torchpairwise.cosine_distances(a, a)
                    + torch.ops.torchpairwise.cdist(a, a, metric='cosine'))

    compare_forward_backward(call, (x,), backend='inductor', dynamic=False)


@pytest.mark.parametrize('metric', ['seuclidean', 'mahalanobis', 'cosine'])
def test_compiled_self_comparisons(metric, device):
    x = (torch.rand(2, 7, 3, device=device) + .2).requires_grad_()
    compare_forward_backward(lambda a: getattr(tp, metric + '_distances')(a), (x,))


@pytest.mark.parametrize('kind', ['paired', 'canberra', 'wminkowski', 'jensenshannon'])
def test_inductor_forward_backward(kind, device):
    d = 2 if kind == 'haversine' else 8
    x, y = [(torch.rand(4, d, device=device) + .3).requires_grad_() for _ in range(2)]
    kwargs = parameters(kind, x)
    parameter_keys = [key for key, value in kwargs.items() if isinstance(value, torch.Tensor)]
    args = (x, y, *(kwargs[key].requires_grad_() for key in parameter_keys))
    scalars = {key: value for key, value in kwargs.items() if key not in parameter_keys}

    def fn(a, b, *params):
        if kind == 'paired':
            return tp.paired_kernels(a, b, kernel='rbf')
        if kind == 'additive_chi2':
            return tp.additive_chi2_kernel(a, b)
        return tp.cdist(a, b, metric=kind, **scalars, **dict(zip(parameter_keys, params)))

    compare_forward_backward(fn, args, backend='inductor', dynamic=False)


def test_export_dynamic_inference_roundtrip(device, tmp_path):
    class Distances(torch.nn.Module):
        def forward(self, x, y):
            return tp.canberra_distances(x, y) + tp.rbf_kernel(x, y)

    x, y = torch.rand(3, 5, device=device), torch.rand(4, 5, device=device)
    n, m = torch.export.Dim('n', min=2), torch.export.Dim('m', min=2)
    program = torch.export.export(Distances(), (x, y), dynamic_shapes=({0: n}, {0: m}))
    path = tmp_path / 'distances.pt2'
    torch.export.save(program, path)
    restored = torch.export.load(path).module()
    x, y = torch.rand(5, 5, device=device), torch.rand(6, 5, device=device)
    torch.testing.assert_close(restored(x, y), Distances()(x, y))


@pytest.mark.parametrize('metric', ['canberra', 'sqeuclidean', 'euclidean', 'jaccard'])
def test_dynamic_sample_counts_share_one_graph(metric, device):
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    fn = torch.compile(lambda a, b: tp.cdist(a, b, metric=metric),
                       backend=backend, fullgraph=True, dynamic=True)
    for n, m in ((3, 4), (6, 7)):
        x, y = torch.rand(n, 5, device=device), torch.rand(m, 5, device=device)
        if metric == 'jaccard':
            x, y = x > .5, y > .5
        torch.testing.assert_close(fn(x, y), tp.cdist(x, y, metric=metric))
    assert len(graphs) == 1


def test_inferred_covariance_dynamic_samples_share_one_graph(device):
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    fn = torch.compile(tp.mahalanobis_distances, backend=backend, fullgraph=True, dynamic=True)
    for n in (7, 9):
        x = torch.rand(2, n, 3, device=device)
        torch.testing.assert_close(fn(x), tp.mahalanobis_distances(x))
    assert len(graphs) == 1
