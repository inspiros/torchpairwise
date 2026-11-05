import pytest
import torch
import torchpairwise as tp
from test_metrics import DISTANCES, KERNELS, inputs, parameters


@pytest.mark.parametrize('metric', DISTANCES)
def test_broadcast_batches(metric, device):
    x, y = inputs(metric, device)
    xx = x.expand(2, 1, *x.shape)
    yy = y.expand(1, 3, *y.shape)
    kw = parameters(metric, x)
    result = tp.cdist(xx, yy, metric=metric, **kw)
    expected = tp.cdist(x, y, metric=metric, **kw).expand(2, 3, len(x), len(y))
    torch.testing.assert_close(result, expected, equal_nan=True)
    torch.testing.assert_close(tp.cdist(x, yy, metric=metric, **kw), expected[:1], equal_nan=True)


@pytest.mark.parametrize('metric', DISTANCES)
def test_paired_matches_small_diagonal(metric, device):
    x, y = inputs(metric, device)
    x = x[:len(y)]
    kw = parameters(metric, x)
    result = tp.paired_distances(x, y, metric=metric, **kw)
    expected = tp.cdist(x, y, metric=metric, **kw).diagonal()
    assert result.shape == (len(x),)
    torch.testing.assert_close(result, expected, equal_nan=True)
    batched = tp.paired_distances(x.expand(2, 1, *x.shape), y.expand(1, 3, *y.shape),
                                  metric=metric, **kw)
    torch.testing.assert_close(batched, expected.expand(2, 3, -1), equal_nan=True)


@pytest.mark.parametrize('kernel', KERNELS)
def test_paired_kernels(kernel, device):
    x, y = inputs('euclidean', device)
    x = x[:len(y)]
    key = 'cosine' if kernel == 'cosine_similarity' else kernel
    expected = tp.pairwise_kernels(x, y, kernel=key).diagonal()
    torch.testing.assert_close(tp.paired_kernels(x, y, kernel=key), expected)


@pytest.mark.parametrize('paired', [False, True])
@pytest.mark.parametrize('metric', ['canberra', 'rbf', 'wminkowski', 'seuclidean', 'mahalanobis'])
def test_broadcast_gradients_reduce_to_original_shapes(metric, paired, device):
    g = torch.Generator().manual_seed(65)
    x = (torch.rand(2, 1, 3, 2, generator=g, dtype=torch.float64) + .2).to(device).requires_grad_()
    y = (torch.rand(1, 2, 3, 2, generator=g, dtype=torch.float64) + .4).to(device).requires_grad_()
    if metric == 'rbf':
        fn = tp.paired_kernels if paired else tp.pairwise_kernels
        call = lambda a, b: fn(a, b, kernel='rbf')
        args = (x, y)
    else:
        fn = tp.paired_distances if paired else tp.cdist
        if metric == 'wminkowski':
            param = torch.tensor([.7, 1.3], device=device, dtype=x.dtype).requires_grad_()
            call = lambda a, b, w: fn(a, b, metric=metric, w=w, p=1.7)
            args = (x, y, param)
        elif metric == 'seuclidean':
            param = torch.tensor([.7, 1.3], device=device, dtype=x.dtype).requires_grad_()
            call = lambda a, b, v: fn(a, b, metric=metric, V=v)
            args = (x, y, param)
        elif metric == 'mahalanobis':
            param = torch.eye(2, device=device, dtype=x.dtype).requires_grad_()
            call = lambda a, b, vi: fn(a, b, metric=metric, VI=vi)
            args = (x, y, param)
        else:
            call = lambda a, b: fn(a, b, metric=metric)
            args = (x, y)
    assert torch.autograd.gradcheck(call, args, fast_mode=True, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('name', ['paired_distances', 'paired_kernels'])
def test_paired_rejects_sample_broadcast(name, device):
    with pytest.raises(ValueError, match='sample counts'):
        getattr(tp, name)(torch.ones(1, 3, device=device), torch.ones(5, 3, device=device))


def test_paired_large_n_has_linear_output(device):
    x = torch.ones(20000, 4, device=device)
    y = x * 2
    result = tp.paired_distances(x, y, metric='sqeuclidean')
    torch.testing.assert_close(result, torch.full((20000,), 4., device=device))
    for kernel in ('rational_quadratic', 'matern', 'histogram_intersection'):
        result = tp.paired_kernels(x, y, kernel=kernel)
        assert result.shape == (20000,)
        assert torch.isfinite(result).all()
    assert tp.paired_distances(x, y, metric='hellinger').shape == (20000,)
    for metric in ('nan_euclidean', 'wasserstein_1d'):
        assert tp.paired_distances(x, y, metric=metric).shape == (20000,)


def test_parameter_batches_can_broadcast(device):
    x, y = inputs('euclidean', device)
    w = torch.ones(2, 1, x.shape[-1], dtype=x.dtype, device=device)
    w[1] *= 3
    out = tp.wminkowski_distances(x, y, w=w)
    assert out.shape == (2, 1, len(x), len(y))
    torch.testing.assert_close(out[1], out[0] * 3 ** .5)


@pytest.mark.parametrize('kernel,kwargs', [('linear', {}), ('polynomial', {'degree': 2, 'gamma': .7, 'coef0': -.2}),
                                        ('sigmoid', {'gamma': .4, 'coef0': -.3}), ('rbf', {'gamma': .9})])
def test_paired_row_reduction_values_and_gradients(kernel, kwargs, device):
    x = torch.rand(2, 1, 4, 3, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(1, 3, 4, 3, device=device, dtype=torch.float64, requires_grad=True)
    paired = tp.paired_kernels(x, y, kernel=kernel, **kwargs)
    diagonal = tp.pairwise_kernels(x, y, kernel=kernel, **kwargs).diagonal(dim1=-2, dim2=-1)
    torch.testing.assert_close(paired, diagonal)
    for actual, expected in zip(torch.autograd.grad(paired.sum(), (x, y)),
                                torch.autograd.grad(diagonal.sum(), (x, y))):
        torch.testing.assert_close(actual, expected)


def test_paired_squared_reduction_zero_distance_and_empty(device):
    x = torch.rand(2, 4, 3, device=device, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    out = tp.paired_distances(x, y, metric='sqeuclidean')
    torch.testing.assert_close(out, torch.zeros_like(out))
    for gradient in torch.autograd.grad(out.sum(), (x, y)):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))
    assert tp.paired_distances(x[:, :0], y[:, :0], metric='sqeuclidean').shape == (2, 0)


@pytest.mark.parametrize('paired', [False, True])
def test_empty_batch(paired, device):
    x = torch.empty(0, 3, 2, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.ones(3, 2, dtype=x.dtype, device=device, requires_grad=True)
    fn = tp.paired_distances if paired else tp.cdist
    out = fn(x, y, metric='canberra')
    assert out.shape == ((0, 3) if paired else (0, 3, 3))
    gx, gy = torch.autograd.grad(out.sum(), (x, y))
    torch.testing.assert_close(gx, torch.zeros_like(x))
    torch.testing.assert_close(gy, torch.zeros_like(y))
