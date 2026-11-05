"""Dense numerical oracles, stable eligibility, replay gradients and saved state."""
import pytest
import torch

import torchpairwise as tp


def dense_neighbors(x, y, k, metric, exclusion=False, **kwargs):
    value = tp.cdist(x, y, metric=metric, **kwargs)
    # An explicit Python ordering is independent of the streaming sort/merge.
    flat = value.detach().cpu().reshape(-1, *value.shape[-2:])
    ids = []
    for matrix in flat:
        rows = []
        for i, row in enumerate(matrix):
            allowed = [j for j in range(row.numel()) if not exclusion or i != j]
            allowed.sort(key=lambda j: (bool(row[j].isnan()), float(row[j]) if not row[j].isnan() else 0., j))
            rows.append(allowed[:k])
        ids.append(rows)
    index = torch.tensor(ids, device=x.device, dtype=torch.int64).reshape(*value.shape[:-1], k)
    return value.gather(-1, index), index


@pytest.mark.parametrize('metric', ['euclidean', 'sqeuclidean', 'manhattan', 'cosine_distances'])
def test_reduced_neighbors_dense_and_broadcast_gradients(metric, device):
    torch.manual_seed(712)
    x = torch.randn(2, 1, 3, 4, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, 2, 5, 4, device=device, dtype=x.dtype, requires_grad=True)
    w = torch.tensor([.5, 1.2, .3, .7], device=device, dtype=x.dtype, requires_grad=True)
    actual = tp.lazy(x, y, w=w, chunk_size=2, metric=metric).topk(3, exclude_self=False, largest=False)
    expected, index = dense_neighbors(x, y, 3, metric, w=w)
    torch.testing.assert_close(actual.indices, index)
    torch.testing.assert_close(actual.values, expected)
    for got, want in zip(torch.autograd.grad(actual.values.square().sum(), (x, y, w)),
                         torch.autograd.grad(expected.square().sum(), (x, y, w))):
        torch.testing.assert_close(got, want)
    closest = tp.lazy(x, y, w=w, chunk_size=1, metric=metric).min(-1, exclude_self=False)
    torch.testing.assert_close(closest.values, expected[..., 0])
    torch.testing.assert_close(closest.indices, index[..., 0])


def test_reduced_ties_self_cosine_and_nonfinite_order(device):
    x = torch.tensor([[0., 0.], [1., 0.], [1., 0.], [-1., 0.]], device=device)
    for size in (1, 3):
        actual = tp.lazy(x, chunk_size=size, metric='cosine_distances').topk(3, exclude_self=True, largest=False)
        dense = tp.cosine_distances(x)
        expected = torch.tensor([[1, 2, 3], [2, 0, 3], [1, 0, 3], [0, 1, 2]], device=device)
        torch.testing.assert_close(actual.indices, expected)
        torch.testing.assert_close(actual.values, dense.gather(-1, expected))
        explicit = tp.lazy(x, x, chunk_size=size, metric='cosine_distances').min(-1, exclude_self=False)
        assert explicit.indices.tolist() == [0, 1, 1, 3]
        assert not explicit.values.any()
    y = torch.tensor([[float('inf')], [float('nan')], [1.], [-float('inf')]], device=device)
    order = tp.lazy(torch.zeros(1, 1, device=device), y, chunk_size=2, metric='euclidean').topk(4, exclude_self=False, largest=False)
    assert order.indices.tolist() == [[2, 0, 3, 1]]
    assert order.values[0, -1].isnan()
    nan = torch.full((3, 1), float('nan'), device=device)
    selected = tp.lazy(nan, chunk_size=1, metric='cosine_distances').topk(2, exclude_self=True, largest=False)
    assert selected.indices.tolist() == [[1, 2], [0, 2], [0, 1]]


@pytest.mark.parametrize('kernel', ['linear', 'rbf', 'laplacian'])
@pytest.mark.parametrize('with_values', [False, True])
def test_kernel_sum_dense_values_and_replayed_gradients(kernel, with_values, device):
    torch.manual_seed(932)
    x = torch.randn(1, 3, 4, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 5, 4, device=device, dtype=x.dtype, requires_grad=True)
    values = torch.randn(2, 1, 5, 3, device=device, dtype=x.dtype, requires_grad=True) if with_values else None
    kwargs = {} if kernel == 'linear' else {'gamma': .4}
    operator = tp.lazy(x, y, metric=kernel, chunk_size=2, **kwargs)
    actual = operator @ values if with_values else operator.sum(-1)
    matrix = tp.pairwise_kernels(x, y, kernel=kernel, **kwargs)
    expected = matrix @ values if with_values else matrix.sum(-1)
    torch.testing.assert_close(actual, expected)
    inputs = (x, y, values) if with_values else (x, y)
    for got, want in zip(torch.autograd.grad(actual.square().sum(), inputs),
                         torch.autograd.grad(expected.square().sum(), inputs)):
        torch.testing.assert_close(got, want)


def test_reduction_gradchecks_and_self_alias_accumulation(device):
    torch.manual_seed(340)
    x = torch.randn(3, 2, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.randn(4, 2, device=device, dtype=x.dtype, requires_grad=True)
    v = torch.randn(4, 2, device=device, dtype=x.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: tp.lazy(a, b, chunk_size=2, metric='euclidean').topk(2, exclude_self=False, largest=False).values, (x, y), fast_mode=True)
    assert torch.autograd.gradcheck(lambda a, b, p: tp.lazy(a, b, chunk_size=2, metric='rbf').matmul(p), (x, y, v), fast_mode=True)
    actual = tp.lazy(x, chunk_size=2, metric='rbf').sum(-1)
    expected = tp.rbf_kernel(x).sum(-1)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                               torch.autograd.grad(expected.sum(), x)[0])


def test_reduced_empty_contracts_and_connected_gradients(device):
    x = torch.randn(2, 3, device=device, requires_grad=True)
    y = torch.empty(0, 3, device=device, requires_grad=True)
    v = torch.empty(0, 2, device=device, requires_grad=True)
    neighbors = tp.lazy(x, y, metric='euclidean').topk(0, exclude_self=False, largest=False)
    assert neighbors.values.shape == (2, 0)
    total = tp.lazy(x, y, metric='rbf').matmul(v)
    assert total.shape == (2, 2) and not total.any()
    for gradient in torch.autograd.grad(total.sum() + neighbors.values.sum(), (x, y, v)):
        assert not gradient.any()
    with pytest.raises(RuntimeError, match='eligible'):
        tp.lazy(x, y, metric='euclidean').min(-1, exclude_self=False)
    with pytest.raises(RuntimeError, match='eligible'):
        tp.lazy(x[:1], metric='euclidean').min(-1, exclude_self=True)
    with pytest.raises(ValueError, match='integer'):
        tp.lazy(x, metric='euclidean').topk(True, exclude_self=True, largest=False)
    with pytest.raises(ValueError, match='Unknown'):
        tp.lazy(x, metric='unknown').topk(1, exclude_self=True, largest=False)
    with pytest.raises(ValueError, match='Unknown'):
        tp.lazy(x, metric='unknown').sum(-1)
    with pytest.raises(TypeError, match='does not accept'):
        tp.lazy(x, gamma=0.1, metric='linear').sum(-1)
    with pytest.raises(RuntimeError, match='matching M'):
        tp.lazy(x, metric='rbf').matmul(torch.ones(3, device=device))


def test_kernel_sum_compensation_and_nonfinite_semantics(device):
    x = torch.ones(1, 1, device=device)
    y = torch.ones(3, 1, device=device)
    values = torch.tensor([[1e8], [1.], [-1e8]], device=device)
    # Tile chunks merge with compensation; fused CUDA kernels ignore
    # chunk_size and compensate per group of 8 pairs, so test the tiles.
    previous = torch.ops.torchpairwise._set_fused_reducers(False)
    try:
        torch.testing.assert_close(tp.lazy(x, y, chunk_size=1, metric='linear').matmul(values), x)
    finally:
        torch.ops.torchpairwise._set_fused_reducers(previous)
    inf = torch.full((1, 1), float('inf'), device=device)
    assert tp.lazy(x, inf, chunk_size=1, metric='linear').sum(-1).isposinf().all()
    assert tp.lazy(x, inf, metric='linear').matmul(torch.zeros_like(inf)).isnan().all()


def test_reduction_saved_tensors_do_not_retain_pair_tiles(device):
    x = torch.rand(7, 3, device=device, requires_grad=True)
    y = torch.rand(19, 3, device=device, requires_grad=True)
    values = torch.rand(19, 2, device=device, requires_grad=True)
    for operation in (lambda: tp.lazy(x, y, chunk_size=4, metric='euclidean').topk(2, exclude_self=False, largest=False).values,
                      lambda: tp.lazy(x, y, chunk_size=4, metric='rbf').matmul(values)):
        shapes = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: (shapes.append(tuple(t.shape)), t)[1], lambda t: t):
            result = operation()
        # Inputs, lengths and k-wide outputs only: nothing tile-sized (4x4).
        inputs = {(7, 3), (19, 3), (19, 2)}
        assert not any(len(shape) >= 2 and min(shape[-2:]) >= 4 and shape not in inputs
                       for shape in shapes), shapes
        result.sum().backward()


def test_reduction_autocast_noncontiguous_and_zero_batch(device):
    x = torch.rand(2, 3, 4, device=device).transpose(-1, -2)
    y = torch.rand(2, 3, 5, device=device).transpose(-1, -2)
    low = torch.bfloat16 if device == 'cpu' else torch.float16
    with torch.autocast(device, dtype=low):
        result = tp.lazy(x.to(low), y.to(low), chunk_size=2, metric='euclidean').topk(2, exclude_self=False, largest=False)
        summed = tp.lazy(x.to(low), y.to(low), chunk_size=2, metric='rbf').sum(-1)
    assert result.values.dtype == summed.dtype == torch.float32
    empty = x[:0]
    assert tp.lazy(empty, y[:0], metric='euclidean').topk(0, exclude_self=False, largest=False).values.shape == (0, 4, 0)
    assert tp.lazy(empty, y[:0], metric='rbf').sum(-1).shape == (0, 4)


@pytest.mark.cuda
def test_reductions_current_cuda_stream():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.tensor([[0., 0.], [1., 0.]], device='cuda', requires_grad=True)
        value = tp.lazy(x, chunk_size=1, metric='euclidean').min(-1, exclude_self=True).values
        value.sum().backward()
        event = torch.cuda.Event()
        event.record()
    event.synchronize()
    assert value.tolist() == [1., 1.]
    torch.testing.assert_close(x.grad, torch.tensor([[-2., 0.], [2., 0.]], device='cuda'))
