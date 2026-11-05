"""Registry-wide numerical coverage and focused reduction/gradient contracts."""
import math

import pytest
import torch

import torchpairwise as tp
from torchpairwise.ops._registry import DISTANCES, KERNELS


def case(name, spec, device):
    x = torch.tensor([[.2, .7, 1.1], [1.5, .3, .8], [.4, 1.3, .1]], device=device, dtype=torch.float64)
    y = torch.tensor([[.5, .9, .2], [.8, .4, 1.7], [1.3, .6, .4], [.3, 1.6, .9]], device=device, dtype=x.dtype)
    kwargs = {}
    if spec.boolean:
        x, y = x > .5, y > .5
    if name == 'haversine':
        x, y = x[:, :2], y[:, :2]
    if name == 'wminkowski':
        kwargs['w'] = x.new_tensor([.7, 1.2, .4])
    if name == 'seuclidean':
        kwargs['V'] = torch.ones(3, device=device, dtype=x.dtype)
    if name == 'mahalanobis':
        kwargs['VI'] = torch.eye(3, device=device, dtype=x.dtype)
    if spec.event_ndim == 3:
        x = x[:, :2, None] + x.new_tensor([0., .15])
        y = y[:, :2, None] + y.new_tensor([.05, -.1])
    if name == 'sliced_wasserstein':
        kwargs['projections'] = torch.eye(2, device=device, dtype=x.dtype)
    if name == 'sinkhorn':
        kwargs['max_iter'] = 8
    return x, y, kwargs


@pytest.mark.parametrize('name', [name for name in DISTANCES if name != 'continuous_jaccard'])
def test_every_registered_distance_sum_and_neighbors(name, device):
    x, y, kwargs = case(name, DISTANCES[name], device)
    dense = tp.cdist(x, y, metric=name, **kwargs)
    actual = tp.lazy(x, y, chunk_size=2, **kwargs, metric=DISTANCES[name].op).reduce('sum', dim=-1)
    torch.testing.assert_close(actual, dense.sum(-1), equal_nan=True)
    selected = tp.lazy(x, y, chunk_size=2, **kwargs, metric=DISTANCES[name].op).topk(2, exclude_self=False, largest=False)
    ids = dense.argsort(dim=-1, stable=True)[..., :2]
    torch.testing.assert_close(selected.indices, ids)
    torch.testing.assert_close(selected.values, dense.gather(-1, ids), equal_nan=True)


@pytest.mark.parametrize('name', list(KERNELS))
def test_every_registered_kernel_rows_payload_and_gradients(name, device):
    x, y, kwargs = case(name, KERNELS[name], device)
    x, y = x.requires_grad_(), y.requires_grad_()
    values = torch.tensor([[1., -.4], [.3, .8], [-.2, 1.1], [.5, -.7]], device=device,
                          dtype=x.dtype, requires_grad=True)
    dense = tp.pairwise_kernels(x, y, kernel=name)
    rows = tp.lazy(x, y, chunk_size=2, metric=name).reduce('sum', dim=-1)
    weighted = tp.lazy(x, y, chunk_size=2, metric=name).matmul(values)
    torch.testing.assert_close(rows, dense.sum(-1))
    torch.testing.assert_close(weighted, dense @ values)
    expected = dense.sum() + (dense @ values).square().sum()
    actual = rows.sum() + weighted.square().sum()
    for got, want in zip(torch.autograd.grad(actual, (x, y, values)),
                         torch.autograd.grad(expected, (x, y, values))):
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize('operation', ['min', 'max', 'sum', 'prod', 'all', 'any'])
def test_basic_reductions_values_keepdim_and_gradients(operation, device):
    x = torch.tensor([[1., .2], [-.5, .7]], device=device, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[.8, .3], [-.1, 1.3], [.4, -.2]], device=device, dtype=x.dtype, requires_grad=True)
    dense = tp.linear_kernel(x, y)
    expected = getattr(dense, operation)(dim=-1, keepdim=True)
    if operation in ('min', 'max'):
        expected = expected.values
    actual = tp.lazy(x, y, chunk_size=1, metric='linear').reduce(operation, dim=-1, keepdim=True)
    torch.testing.assert_close(actual, expected)
    if operation in ('all', 'any'):
        assert actual.dtype == torch.bool and not actual.requires_grad
    else:
        for got, want in zip(torch.autograd.grad(actual.sum(), (x, y)), torch.autograd.grad(expected.sum(), (x, y))):
            torch.testing.assert_close(got, want)


def test_products_zero_counts_and_negative_factors(device):
    for factors in ([0., 2., 3.], [0., 0., 3.], [0., 0., float("inf")], [0., 0., float("nan")], [-2., .3, 1.5], [0., float('inf')], [float('nan'), 2.]):
        x = torch.ones(1, 1, device=device, dtype=torch.float64)
        y = torch.tensor(factors, device=device, dtype=x.dtype).reshape(-1, 1).requires_grad_()
        expected = tp.linear_kernel(x, y).prod(-1)
        actual = tp.lazy(x, y, chunk_size=1, metric='linear').reduce('prod', dim=-1)
        torch.testing.assert_close(actual, expected, equal_nan=True)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), y)[0],
                                   torch.autograd.grad(expected.sum(), y)[0], equal_nan=True)

    # Shared any-zero predicate spans output rows/batches. Joint products must
    # reduce entries directly, not differentiate a nested product of row totals.
    for values in ([[0., 0., float('inf')]], [[0., 0., float('nan')]],
                   [[float('inf'), 2.], [0., 3.]],
                   [[float('nan'), 2.], [0., 3.]],
                   [[float('inf'), 2.], [3., 4.]],
                   [[[float('inf'), 2.]], [[0., 3.]]]):
        raw = torch.tensor(values, device=device, dtype=torch.float64, requires_grad=True)
        for dim in (-1, -2, None):
            expected = raw.flatten(-2).prod(-1) if dim is None else raw.prod(dim)
            for size in (1, 2):
                actual = (tp.as_lazy(raw, chunk_size=size) + 0.0).prod(dim)
                torch.testing.assert_close(actual, expected, equal_nan=True)
                torch.testing.assert_close(torch.autograd.grad(actual.sum(), raw)[0],
                                           torch.autograd.grad(expected.sum(), raw, retain_graph=True)[0], equal_nan=True)
    raw = torch.tensor([[float('nan'), 0., float('inf')],
                        [2., float('inf'), 0.], [3., 4., float('nan')]],
                       device=device, dtype=torch.float64, requires_grad=True)
    allowed = ~torch.eye(3, device=device, dtype=torch.bool)
    dense = torch.where(allowed, raw, 1.)
    source = tp.as_lazy(raw, chunk_size=2) + 0.
    for dim in (-1, -2, None):
        actual = source.prod(dim, keepdim=True, exclude_self=True)
        expected = dense.flatten(-2).prod(-1).reshape(1, 1) if dim is None else dense.prod(dim, keepdim=True)
        torch.testing.assert_close(actual, expected, equal_nan=True)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), raw)[0],
                                   torch.autograd.grad(expected.sum(), raw, retain_graph=True)[0], equal_nan=True)


def test_boolean_metrics_predicates_and_weight_gradients(device):
    x = torch.tensor([[True, False, True], [False, True, True]], device=device)
    y = torch.tensor([[True, True, False], [False, True, True], [True, False, True]], device=device)
    w = torch.tensor([.5, 1.2, .8], device=device, dtype=torch.float64, requires_grad=True)
    dense = tp.cdist(x, y, metric='jaccard', w=w)
    for operation in ('all', 'any'):
        actual = tp.lazy(x, y, w=w, chunk_size=1, metric='jaccard').reduce(operation, dim=-1)
        torch.testing.assert_close(actual, getattr(dense, operation)(-1))
        selected = getattr(tp.lazy(x, y, w=w, chunk_size=1, metric='jaccard'), operation)(dim=-1, predicate=lambda d: d < 0.3)
        torch.testing.assert_close(selected, getattr(dense < .3, operation)(-1))
        assert not selected.requires_grad
    actual = tp.lazy(x, y, w=w, chunk_size=2, metric='jaccard').reduce('sum', dim=-1)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), w)[0],
                               torch.autograd.grad(dense.sum(), w)[0])
    # Hamming accepts numeric rows but has no input-value derivatives.
    hamming = tp.lazy(x.double().requires_grad_(), y.double(), metric='hamming').reduce('sum', dim=-1)
    assert not hamming.requires_grad


def test_largest_ties_nonfinites_and_nan_propagating_minmax(device):
    x = torch.zeros(2, 1, device=device)
    y = torch.tensor([[1.], [-1.], [float('inf')], [float('nan')]], device=device)
    for size in (1, 3):
        result = tp.lazy(x, y, chunk_size=size, metric='euclidean').topk(4, exclude_self=False, largest=True)
        assert result.indices.tolist() == [[3, 2, 0, 1], [3, 2, 0, 1]]
        assert tp.lazy(x, y, chunk_size=size, metric='euclidean').reduce('min', dim=-1).isnan().all()
        assert tp.lazy(x, y, chunk_size=size, metric='euclidean').reduce('max', dim=-1).isnan().all()
    a = torch.tensor([[1.]], device=device, dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[2.], [2.], [-1.]], device=device, dtype=a.dtype, requires_grad=True)
    maximum = tp.lazy(a, b, chunk_size=1, metric='linear').reduce('max', dim=-1)
    assert torch.autograd.grad(maximum.sum(), b)[0].flatten().tolist() == [1., 0., 0.]


@pytest.mark.parametrize('metric', ['seuclidean', 'mahalanobis'])
def test_inferred_statistics_use_complete_collection(metric, device):
    torch.manual_seed(824)
    x = torch.randn(2, 7, 3, device=device, dtype=torch.float64, requires_grad=True)
    # Named self API infers the detached statistics, generic cdist requires y.
    dense = getattr(tp, metric + '_distances')(x)
    expected = dense.sum(-1)
    actual = tp.lazy(x, chunk_size=2, metric=metric).reduce('sum', dim=-1)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(torch.autograd.grad(actual.sum(), x), torch.autograd.grad(expected.sum(), x)):
        torch.testing.assert_close(got, want)
    with pytest.raises(RuntimeError, match='required'):
        tp.lazy(x, x, metric=metric).reduce('sum', dim=-1)


@pytest.mark.parametrize('metric', ['chamfer', 'directed_hausdorff', 'sliced_wasserstein', 'sinkhorn'])
def test_cloud_lengths_selected_and_row_gradients(metric, device):
    torch.manual_seed(715)
    x = torch.rand(2, 3, 2, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(3, 4, 2, device=device, dtype=x.dtype, requires_grad=True)
    xl, yl = torch.tensor([2, 3], device=device), torch.tensor([4, 2, 3], device=device)
    with torch.no_grad():
        x[0, 2] = float('nan')
        y[1, 2:] = float('nan')
        y[2, 3:] = float('nan')
    kwargs = {'x1_lengths': xl, 'x2_lengths': yl}
    if metric == 'sliced_wasserstein':
        kwargs['projections'] = torch.eye(2, device=device, dtype=x.dtype)
    if metric == 'sinkhorn':
        kwargs['max_iter'] = 6
    dense = tp.cdist(x, y, metric=metric, **kwargs)
    actual = tp.lazy(x, y, chunk_size=1, **kwargs, metric=metric).reduce('sum', dim=-1)
    selected = tp.lazy(x, y, chunk_size=1, **kwargs, metric=metric).min(-1, exclude_self=False)
    expected_selected = dense.min(-1).values
    torch.testing.assert_close(actual, dense.sum(-1))
    torch.testing.assert_close(selected.values, expected_selected)
    for got, want in zip(torch.autograd.grad(actual.sum() + selected.values.sum(), (x, y)),
                         torch.autograd.grad(dense.sum() + expected_selected.sum(), (x, y))):
        torch.testing.assert_close(got, want)


def test_empty_identities_exclusion_contracts_and_predicate_validation(device):
    x = torch.rand(2, 3, device=device, requires_grad=True)
    y = torch.empty(0, 3, device=device, requires_grad=True)
    for operation, identity in [('sum', 0), ('prod', 1), ('all', True), ('any', False)]:
        actual = tp.lazy(x, y, metric='euclidean').reduce(operation, dim=-1)
        assert actual.tolist() == [identity, identity]
        if operation in ('sum', 'prod'):
            for g in torch.autograd.grad(actual.sum(), (x, y)):
                assert not g.any()
    for operation in ('min', 'max'):
        with pytest.raises(RuntimeError, match='eligible|empty'):
            tp.lazy(x, y, metric='euclidean').reduce(operation, dim=-1)
    dense = tp.euclidean_distances(x)
    torch.testing.assert_close(tp.lazy(x, metric='euclidean').reduce('sum', dim=-1, exclude_self=True), dense.sum(-1))
    assert tp.lazy(x, metric='euclidean').reduce('prod', dim=-1, exclude_self=True).shape == (2,)
    with pytest.raises(TypeError, match='predicate'):
        tp.lazy(x, metric='euclidean').reduce('sum', dim=-1, predicate=lambda t: t > 0)
    with pytest.raises(TypeError, match='Boolean'):
        tp.lazy(x, metric='euclidean').any(dim=-1, predicate=lambda t: t)
    with pytest.raises(ValueError, match='tile shape'):
        tp.lazy(x, metric='euclidean').all(dim=-1, predicate=lambda t: t.any())


def test_reduction_saved_state_is_bounded_for_new_operators(device):
    x = torch.rand(5, 3, device=device, requires_grad=True)
    y = torch.rand(17, 3, device=device, requires_grad=True)
    for operation in ('sum', 'prod', 'min', 'max'):
        shapes = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: (shapes.append(tuple(t.shape)), t)[1], lambda t: t):
            value = tp.lazy(x, y, chunk_size=2, metric='rbf').reduce(operation, dim=-1)
        # Products save their slot packet and O(N+M) row state; no saved
        # tensor may span both sample axes.
        assert all(math.prod(shape) < 5 * 17 and not {5, 17} <= set(shape)
                   for shape in shapes), shapes
        value.sum().backward()


@pytest.mark.parametrize('metric', ['seuclidean', 'mahalanobis', 'wasserstein_1d', 'sliced_wasserstein'])
def test_shared_and_batched_tensor_parameter_gradients(metric, device):
    torch.manual_seed(619)
    x = (torch.rand(1, 3, 3, device=device, dtype=torch.float64) + .2).requires_grad_()
    y = (torch.rand(4, 3, device=device, dtype=x.dtype) + .2).requires_grad_()
    if metric == 'seuclidean':
        parameter = torch.tensor([[[.5, 1.2, .8]], [[1.1, .7, .4]]], device=device,
                                 dtype=x.dtype, requires_grad=True)
        kwargs = {'V': parameter}
    elif metric == 'mahalanobis':
        parameter = torch.eye(3, device=device, dtype=x.dtype).repeat(2, 1, 1).requires_grad_()
        kwargs = {'VI': parameter}
    elif metric == 'wasserstein_1d':
        parameter = torch.tensor([.1, .8, 2.], device=device, dtype=x.dtype, requires_grad=True)
        kwargs = {'bin_positions': parameter}
    else:
        x, y = x.reshape(1, 3, 3, 1), y.reshape(4, 3, 1)
        parameter = torch.ones(1, 1, device=device, dtype=x.dtype, requires_grad=True)
        kwargs = {'projections': parameter}
    matrix = tp.cdist(x, y, metric=metric, **kwargs)
    actual = tp.lazy(x, y, chunk_size=2, **kwargs, metric=metric).reduce('sum', dim=-1)
    nearest = tp.lazy(x, y, chunk_size=2, **kwargs, metric=metric).min(-1, exclude_self=False)
    torch.testing.assert_close(actual, matrix.sum(-1))
    torch.testing.assert_close(nearest.values, matrix.min(-1).values)
    for got, want in zip(torch.autograd.grad(actual.sum() + nearest.values.sum(), (x, y, parameter)),
                         torch.autograd.grad(matrix.sum() + matrix.min(-1).values.sum(), (x, y, parameter))):
        torch.testing.assert_close(got, want)


def test_reduced_cloud_spatial_layout_and_batch_lengths(device):
    torch.manual_seed(938)
    x = torch.rand(1, 3, 2, 2, 2, device=device, dtype=torch.float64)
    y = torch.rand(2, 4, 2, 2, 2, device=device, dtype=x.dtype)
    xl, yl = torch.tensor([[3, 4, 2]], device=device), torch.tensor([[4, 2, 3, 4]], device=device)
    kwargs = dict(metric='chamfer', batched=True, feature_dim=-1, x1_lengths=xl, x2_lengths=yl)
    matrix = tp.cdist(x, y, **kwargs)
    actual = tp.lazy(x, y, chunk_size=2, **kwargs).reduce('max', dim=-1)
    match = tp.lazy(x, y, chunk_size=2, **kwargs).topk(2, exclude_self=False, largest=True)
    torch.testing.assert_close(actual, matrix.max(-1).values)
    expected = matrix.argsort(dim=-1, descending=True, stable=True)[..., :2]
    torch.testing.assert_close(match.indices, expected)
    assert actual.shape == (2, 3)


def test_new_reduction_autocast_and_singleton_exclusion(device):
    low = torch.bfloat16 if device == 'cpu' else torch.float16
    x = torch.rand(2, 3, device=device, dtype=low)
    y = torch.rand(3, 3, device=device, dtype=low)
    with torch.autocast(device, dtype=low):
        for operation in ('min', 'max', 'sum', 'prod'):
            value = tp.lazy(x, y, chunk_size=1, metric='euclidean').reduce(operation, dim=-1)
            reference = tp.lazy(x.float(), y.float(), chunk_size=1, metric='euclidean').reduce(operation, dim=-1)
            assert value.dtype == torch.float32
            torch.testing.assert_close(value, reference)
        truth = tp.lazy(x, y, metric='euclidean').any(dim=-1, predicate=lambda d: d < 0.5)
        assert truth.dtype == torch.bool
    singleton = torch.ones(1, 3, device=device)
    for operation, expected in [('sum', 0.), ('prod', 1.), ('all', True), ('any', False)]:
        assert tp.lazy(singleton, metric='euclidean').reduce(operation, dim=-1, exclude_self=True).item() == expected
