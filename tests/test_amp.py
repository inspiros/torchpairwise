"""Autocast precision, mixed parameter dtypes, and first derivative contracts."""
import pytest
import torch
import torchpairwise as tp

from torchpairwise.ops._registry import OPS
from test_metrics import parameters


FLOAT_OPS = sorted({spec.op for spec in OPS.values()
                    if spec.differentiable and spec.event_ndim == 2})


def test_canberra_direct_half_uses_float_accumulation(device):
    # Feature sums and both reference-axis gradient sums exceed Half's
    # consecutive-integer range. Output and input gradients still use Half.
    x = torch.zeros(1, 4096, device=device, dtype=torch.float16)
    y = torch.ones_like(x)
    actual = tp.canberra_distances(x, y)
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, actual.new_full((1, 1), 4096))
    for rows, others in ((1, 4096), (4096, 1)):
        x = torch.zeros(rows, 1, device=device, dtype=torch.float16, requires_grad=True)
        y = torch.ones(others, 1, device=device, dtype=torch.float16, requires_grad=True)
        gx, gy = torch.autograd.grad(tp.canberra_distances(x, y).sum(), (x, y))
        torch.testing.assert_close(gx, torch.full_like(x, -others))
        torch.testing.assert_close(gy, torch.zeros_like(y))
        # Swap operands to exercise the other native gradient accumulator.
        gy, gx = torch.autograd.grad(tp.canberra_distances(y, x).sum(), (y, x))
        torch.testing.assert_close(gx, torch.full_like(x, -others))
        torch.testing.assert_close(gy, torch.zeros_like(y))
    # Both |a| + |b| and a - b overflow if evaluated before promotion.
    x = torch.tensor([[60000., 0.]], device=device, dtype=torch.float16, requires_grad=True)
    y = torch.tensor([[-60000., 0.]], device=device, dtype=torch.float16, requires_grad=True)
    actual = torch.ops.torchpairwise._canberra(x, y)
    torch.testing.assert_close(actual, actual.new_ones((1, 1)))
    gx, gy = torch.autograd.grad(actual.sum(), (x, y))
    torch.testing.assert_close(gx, torch.zeros_like(x))
    torch.testing.assert_close(gy, torch.zeros_like(y))


@pytest.mark.parametrize('name', FLOAT_OPS)
def test_autocast_matches_float32(name, device):
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    d = 2 if name == 'haversine_distances' else 5
    x = (torch.rand(2, 1, 3, d, device=device) + .3).to(dtype).requires_grad_()
    y = (torch.rand(1, 2, 3, d, device=device) + .5).to(dtype).requires_grad_()
    # Parameters commonly stay float32 when upstream layers produce half output.
    kwargs = parameters(name.removesuffix('_distances'), x.float())
    kwargs = {key: value.requires_grad_() if isinstance(value, torch.Tensor) else value
              for key, value in kwargs.items()}
    fn = getattr(tp, name)
    expected = fn(x.float(), y.float(), **kwargs)
    actual_inputs = [x, y] + [value for value in kwargs.values() if isinstance(value, torch.Tensor)]
    expected_grads = torch.autograd.grad(expected.sum(), actual_inputs)
    with torch.autocast(device, dtype=dtype):
        actual = fn(x, y, **kwargs)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    grads = torch.autograd.grad(actual.sum(), actual_inputs)
    for grad, expected_grad, value in zip(grads, expected_grads, actual_inputs):
        assert grad.dtype == value.dtype
        torch.testing.assert_close(grad, expected_grad)


@pytest.mark.parametrize('metric', ['mahalanobis', 'wminkowski', 'canberra', 'sqeuclidean'])
def test_autocast_paired_and_parameters(metric, device):
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    x = torch.rand(2, 3, 5, device=device).to(dtype).requires_grad_()
    y = torch.rand(3, 5, device=device).to(dtype).requires_grad_()
    kwargs = parameters(metric, x.float())
    fn = tp.paired_distances
    expected = fn(x.float(), y.float(), metric=metric, **kwargs)
    with torch.autocast(device, dtype=dtype):
        actual = fn(x, y, metric=metric, **kwargs)
    torch.testing.assert_close(actual, expected)
    for grad in torch.autograd.grad(actual.sum(), (x, y)):
        assert torch.isfinite(grad).all()


@pytest.mark.parametrize('kernel', ['linear', 'polynomial', 'sigmoid', 'rbf', 'chi2'])
def test_paired_kernel_autocast_and_double(kernel, device):
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        x = torch.rand(3, 5, device=device).to(dtype).requires_grad_()
        y = torch.rand(3, 5, device=device).to(dtype).requires_grad_()
        ref_x, ref_y = (x, y) if dtype == torch.float64 else (x.float(), y.float())
        expected = tp.paired_kernels(ref_x, ref_y, kernel=kernel)
        with torch.autocast(device, dtype=torch.float16 if device == 'cuda' else torch.bfloat16):
            actual = tp.paired_kernels(x, y, kernel=kernel)
        assert actual.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
        torch.testing.assert_close(actual, expected)
        torch.autograd.grad(actual.sum(), (x, y))


@pytest.mark.parametrize('name', ['canberra_distances', 'linear_kernel', 'sqeuclidean_distances'])
def test_native_autocast_and_double(name, device):
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        x = torch.rand(3, 5, device=device).to(dtype).requires_grad_()
        fn = getattr(torch.ops.torchpairwise, name)
        expected = fn(x if dtype == torch.float64 else x.float())
        with torch.autocast(device, dtype=torch.float16 if device == 'cuda' else torch.bfloat16):
            actual = fn(x)
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected)
        torch.autograd.grad(actual.sum(), (x,))


def test_autocast_preserves_self_comparison_and_boolean_equality(device):
    x = torch.zeros(3, 5, device=device, dtype=torch.float16)
    with torch.autocast(device, dtype=torch.float16 if device == 'cuda' else torch.bfloat16):
        torch.testing.assert_close(tp.cosine_distances(x).diagonal(), torch.zeros(3, device=device))
        torch.testing.assert_close(tp.cosine_distances(x, x).diagonal(), torch.zeros(3, device=device))
        torch.testing.assert_close(tp.hamming_distances(x), torch.zeros(3, 3, device=device))


def test_native_autocast_preserves_cosine_identity(device):
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        x = torch.zeros(3, 5, device=device, dtype=dtype, requires_grad=True)
        y = x.detach().clone()
        with torch.autocast(device, dtype=torch.float16 if device == 'cuda' else torch.bfloat16):
            same = torch.ops.torchpairwise.cosine_distances(x, x)
            cross = torch.ops.torchpairwise.cosine_distances(x, y)
            dispatched = torch.ops.torchpairwise.cdist(x, x, metric='cosine')
        torch.testing.assert_close(same.diagonal(), torch.zeros_like(same.diagonal()))
        torch.testing.assert_close(dispatched, same)
        torch.testing.assert_close(cross.diagonal(), torch.ones_like(cross.diagonal()))
        gradient, = torch.autograd.grad(same.diagonal().sum(), x)
        torch.testing.assert_close(gradient, torch.zeros_like(x))


@pytest.mark.cuda
def test_cuda_grad_scaler_training_step():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    layer = torch.nn.Linear(5, 4, device='cuda')
    optimizer = torch.optim.SGD(layer.parameters(), lr=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=16.)
    before = layer.weight.detach().clone()
    with torch.autocast('cuda', dtype=torch.float16):
        features = layer(torch.randn(6, 5, device='cuda'))
        loss = tp.rbf_kernel(features).mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(before, layer.weight)
    assert torch.isfinite(layer.weight).all()
