"""Condensed ordering and assembly; metric formulas are tested separately."""
import pytest
import scipy.spatial.distance as sd
import torch
import torchpairwise as tp

from test_metrics import DISTANCES, KERNELS, inputs, parameters


@pytest.mark.parametrize('metric', [name for name in DISTANCES if name != 'snr'])
def test_condensed_distances(metric, device):
    x, _ = inputs(metric, device)
    kwargs = parameters(metric, x)
    expected = tp.cdist(x, x, metric=metric, **kwargs)
    i, j = torch.triu_indices(len(x), len(x), offset=1, device=device)
    actual = tp.pdist(x, metric=metric, **kwargs)
    torch.testing.assert_close(actual, expected[i, j], equal_nan=True)
    torch.testing.assert_close(torch.ops.torchpairwise.pdist(x, metric=metric, **kwargs),
                               actual, equal_nan=True)
    # Independent order/value oracle for the usual pdist baseline.
    if metric == 'euclidean':
        torch.testing.assert_close(actual.cpu(), torch.from_numpy(sd.pdist(x.cpu().numpy())))


@pytest.mark.parametrize('kernel', KERNELS)
def test_condensed_kernels(kernel, device):
    key = 'cosine' if kernel == 'cosine_similarity' else kernel
    x, _ = inputs('euclidean', device)
    i, j = torch.triu_indices(len(x), len(x), offset=1, device=device)
    kwargs = {'gamma': .7} if key in ('rbf', 'chi2') else {}
    actual = tp.condensed_kernels(x, kernel=key, **kwargs)
    torch.testing.assert_close(actual, tp.pairwise_kernels(x, kernel=key, **kwargs)[i, j])
    torch.testing.assert_close(torch.ops.torchpairwise.pdist_kernel(x, kernel=key, **kwargs), actual)


@pytest.mark.parametrize('metric', ['canberra', 'wminkowski', 'mahalanobis'])
def test_condensed_parameter_batches_and_gradients(metric, device):
    x = (torch.rand(2, 1, 4, 3, device=device, dtype=torch.float64) + .3).requires_grad_()
    kwargs = parameters(metric, x)
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            kwargs[key] = value.expand(1, 3, *value.shape).clone().requires_grad_()
    args = (x, *(value for value in kwargs.values() if isinstance(value, torch.Tensor)))
    actual = tp.condensed_distances(x, metric=metric, **kwargs)
    i, j = torch.triu_indices(4, 4, 1, device=device)
    expected = tp.cdist(x, x, metric=metric, **kwargs)[..., i, j]
    torch.testing.assert_close(actual, expected)
    for grad, ref in zip(torch.autograd.grad(actual.sum(), args), torch.autograd.grad(expected.sum(), args)):
        torch.testing.assert_close(grad, ref)


@pytest.mark.parametrize('metric', ['seuclidean', 'mahalanobis'])
def test_condensed_infers_full_sample_statistics(metric, device):
    x = torch.randn(2, 7, 3, device=device, dtype=torch.float64, requires_grad=True)
    i, j = torch.triu_indices(7, 7, 1, device=device)
    actual = tp.pdist(x, metric=metric)
    expected = getattr(tp, metric + '_distances')(x)[..., i, j]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.ops.torchpairwise.pdist(x, metric=metric), actual)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                               torch.autograd.grad(expected.sum(), x)[0])


@pytest.mark.parametrize('n', [0, 1, 2])
def test_condensed_small_and_empty_inputs(n, device):
    x = torch.rand(2, n, 3, device=device, requires_grad=True)
    for fn in (tp.pdist, tp.condensed_kernels):
        actual = fn(x)
        assert actual.shape == (2, n * (n - 1) // 2)
        if n < 2:
            torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0], torch.zeros_like(x))
    torch.testing.assert_close(torch.ops.torchpairwise.pdist(x), tp.pdist(x))
    assert tp.pdist(torch.empty(0, n, 3, device=device)).shape == (0, n * (n - 1) // 2)


def test_condensed_autocast_noncontiguous_and_kernel_gradient(device):
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    x = torch.rand(2, 5, 6, device=device, dtype=dtype)[..., ::2].requires_grad_()
    with torch.autocast(device, dtype=dtype):
        actual = tp.condensed_kernels(x, kernel='rbf')
        native = torch.ops.torchpairwise.pdist_kernel(x, kernel='rbf')
        distances = torch.ops.torchpairwise.pdist(x, metric='canberra')
    torch.testing.assert_close(actual, tp.condensed_kernels(x.float(), kernel='rbf'))
    torch.testing.assert_close(native, actual)
    torch.testing.assert_close(distances, tp.pdist(x.float(), metric='canberra'))
    assert actual.dtype == torch.float32
    assert torch.isfinite(torch.autograd.grad(actual.sum(), x)[0]).all()


def test_condensed_rejects_invalid_contracts():
    x = torch.ones(4, 3)
    for metric in ('snr', 'directed_hausdorff'):
        with pytest.raises(ValueError, match='symmetric'):
            tp.pdist(x, metric=metric)
        with pytest.raises(RuntimeError, match='symmetric'):
            torch.ops.torchpairwise.pdist(x, metric=metric)
    with pytest.raises(TypeError, match='does not accept'):
        tp.condensed_kernels(x, kernel='linear', gamma=.5)
    with pytest.raises(ValueError, match='non-negative'):
        tp.pdist(x, metric='wminkowski', w=-torch.ones(3))


@pytest.mark.compiler
def test_condensed_compiled_forward_backward(device):
    x = (torch.rand(4, 3, device=device) + .2).requires_grad_()

    def call(value):
        return tp.pdist(value, metric='canberra') + tp.condensed_kernels(value, kernel='rbf')

    expected = call(x)
    actual = torch.compile(call, fullgraph=True)(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                               torch.autograd.grad(expected.sum(), x)[0])
