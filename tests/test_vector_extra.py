"""Independent vector/histogram formulas, singular boundaries, and public modes."""
import math
import numpy as np
import pytest
import torch
import torchpairwise as tp


NAMES = ('angular', 'lorentzian', 'clark', 'soergel', 'ruzicka')


def reference(x, y, name):
    a, b = x[..., :, None, :], y[..., None, :, :]
    if name == 'angular':
        # Ordinary-angle reference independent of the implementation's atan2.
        product = (a * b).sum(-1)
        return np.arccos(np.clip(product / np.sqrt((a * a).sum(-1) * (b * b).sum(-1)), -1, 1))
    if name == 'lorentzian':
        return np.log1p(np.abs(a - b)).sum(-1)
    if name == 'clark':
        ratio = np.divide(np.abs(a - b), a + b, out=np.zeros(np.broadcast_shapes(a.shape, b.shape)),
                          where=(a + b) != 0)
        return np.sqrt((ratio * ratio).sum(-1))
    maximum = np.maximum(a, b).sum(-1)
    # Min/max intersection form is an independent Ruzicka identity.
    intersection = np.minimum(a, b).sum(-1)
    return np.divide(maximum - intersection, maximum, out=np.zeros_like(maximum), where=maximum != 0)


@pytest.mark.parametrize('name', NAMES)
def test_vector_extra_reference_broadcast_modes_and_gradcheck(name, device):
    torch.manual_seed(91)
    fn = getattr(tp, name + '_distances')
    for dtype in (torch.float32, torch.float64):
        x = (torch.rand(2, 1, 3, 8, device=device, dtype=dtype) + .2)[..., ::2].requires_grad_()
        y = (torch.rand(1, 2, 4, 8, device=device, dtype=dtype) + .4)[..., ::2].requires_grad_()
        actual = fn(x, y)
        expected = reference(x.detach().cpu().double().numpy(), y.detach().cpu().double().numpy(), name)
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=dtype, device=device))
        torch.testing.assert_close(tp.cdist(x, y, metric=name), actual)
        paired = tp.paired_distances(x, y[..., :3, :], metric=name)
        diagonal = fn(x, y[..., :3, :]).diagonal(dim1=-2, dim2=-1)
        torch.testing.assert_close(paired, diagonal)
        for grad, ref in zip(torch.autograd.grad(paired.sum(), (x, y)),
                             torch.autograd.grad(diagonal.sum(), (x, y))):
            torch.testing.assert_close(grad, ref, atol=3e-6, rtol=2e-5)
        i, j = torch.triu_indices(3, 3, 1, device=device)
        condensed = tp.condensed_distances(x, metric=name)
        triangle = fn(x)[..., i, j]
        torch.testing.assert_close(condensed, triangle)
        torch.testing.assert_close(torch.autograd.grad(condensed.sum(), x)[0],
                                   torch.autograd.grad(triangle.sum(), x)[0], atol=3e-6, rtol=2e-5)
    a = torch.tensor([[.2, .9, .4], [.5, .3, 1.2]], dtype=torch.float64, device=device, requires_grad=True)
    b = torch.tensor([[.7, .4, 1.1], [1.1, .8, .2]], dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (a, b), fast_mode=True)


@pytest.mark.parametrize('name', NAMES)
def test_vector_extra_feature_and_sample_tiles(name, device):
    torch.manual_seed(92)
    x = (torch.rand(17, 65, dtype=torch.float64, device=device) + .2).requires_grad_()
    y = (torch.rand(2, 65, dtype=torch.float64, device=device) + .3).requires_grad_()
    actual = getattr(tp, name + '_distances')(x, y)
    expected = reference(x.detach().cpu().numpy(), y.detach().cpu().numpy(), name)
    torch.testing.assert_close(actual, torch.as_tensor(expected, device=device))
    gradients = torch.autograd.grad(actual.sum(), (x, y))
    assert all(torch.isfinite(grad).all() for grad in gradients)


def test_angular_small_and_antipodal_gradients(device):
    for dtype, delta, tolerance in ((torch.float32, 1e-4, 3e-6), (torch.float64, 1e-10, 3e-14)):
        for sign in (1., -1.):
            x = torch.tensor([[1., 0.]], dtype=dtype, device=device, requires_grad=True)
            y = torch.tensor([[sign, delta]], dtype=dtype, device=device, requires_grad=True)
            value = tp.angular_distances(x, y)
            expected = math.atan(delta) if sign > 0 else math.pi - math.atan(delta)
            torch.testing.assert_close(value, torch.tensor([[expected]], dtype=dtype, device=device),
                                       atol=tolerance, rtol=tolerance)
            gx, gy = torch.autograd.grad(value.sum(), (x, y))
            torch.testing.assert_close(gx, torch.tensor([[0., -1.]], dtype=dtype, device=device),
                                       atol=tolerance, rtol=tolerance)
            target = torch.tensor([[-delta, sign]], dtype=dtype, device=device) / (1 + delta * delta)
            torch.testing.assert_close(gy, target, atol=tolerance, rtol=tolerance)
    for sign in (1., -1.):
        x = torch.tensor([[1., 2., -3.]], dtype=torch.float64, device=device, requires_grad=True)
        y = (x.detach() * sign).requires_grad_()
        for upstream in (0., 1.):
            value = tp.angular_distances(x, y)
            torch.testing.assert_close(value, torch.tensor([[0. if sign > 0 else math.pi]], device=device,
                                                           dtype=torch.float64))
            for gradient in torch.autograd.grad((value * upstream).sum(), (x, y)):
                torch.testing.assert_close(gradient, torch.zeros_like(gradient))
    # Row normalization must not overflow on otherwise finite inputs.
    huge = torch.tensor([[torch.finfo(torch.float64).max, 0.]], dtype=torch.float64, device=device)
    orthogonal = huge.flip(-1)
    torch.testing.assert_close(tp.angular_distances(huge, orthogonal),
                               torch.tensor([[math.pi / 2]], device=device, dtype=huge.dtype))


@pytest.mark.parametrize('name', ['clark', 'soergel', 'ruzicka'])
def test_histogram_extra_zero_conventions_and_large_inputs(name, device):
    fn = getattr(tp, name + '_distances')
    x = torch.tensor([[0., 0., 0.], [0., 2., 3.]], dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tensor([[0., 0., 0.], [1., 0., 3.]], dtype=torch.float64, device=device, requires_grad=True)
    actual = fn(x, y)
    expected = reference(x.detach().cpu().numpy(), y.detach().cpu().numpy(), name)
    torch.testing.assert_close(actual, torch.as_tensor(expected, device=device))
    for gradient in torch.autograd.grad(actual[0, 0], (x, y)):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))
    same = fn(x, x).diagonal()
    assert same.eq(0).all()
    torch.testing.assert_close(torch.autograd.grad(same.sum(), x)[0], torch.zeros_like(x))
    huge = torch.finfo(torch.float64).max
    a = torch.tensor([[huge, huge]], dtype=torch.float64, device=device)
    b = torch.tensor([[huge / 2, 0.]], dtype=torch.float64, device=device)
    target = math.sqrt(1 + 1 / 9) if name == 'clark' else .75
    torch.testing.assert_close(fn(a, b), torch.tensor([[target]], dtype=torch.float64, device=device))
    if name == 'clark':
        # Per-coordinate ratios must retain a tiny bin beside a huge bin.
        b = torch.tensor([[huge, 1e-300]], dtype=torch.float64, device=device)
        a = torch.tensor([[huge, 2e-300]], dtype=torch.float64, device=device)
        torch.testing.assert_close(fn(a, b), torch.tensor([[1 / 3]], dtype=torch.float64, device=device))
    else:
        torch.testing.assert_close(fn(x, y), tp.ruzicka_distances(x, y))


def test_lorentzian_signed_extremes_and_identity(device):
    maximum = torch.finfo(torch.float64).max
    x = torch.tensor([[maximum, -maximum]], dtype=torch.float64, device=device, requires_grad=True)
    y = (-x.detach()).requires_grad_()
    value = tp.lorentzian_distances(x, y)
    expected = 2 * (math.log(maximum) + math.log(2))
    torch.testing.assert_close(value, torch.tensor([[expected]], dtype=torch.float64, device=device))
    for grad in torch.autograd.grad(value.sum(), (x, y)):
        assert torch.isfinite(grad).all()
    same = tp.lorentzian_distances(x, x)
    assert same.eq(0).all()
    torch.testing.assert_close(torch.autograd.grad(same.sum(), x)[0], torch.zeros_like(x))


@pytest.mark.parametrize('name', NAMES)
def test_vector_extra_validation_and_empty(name, device):
    fn = getattr(tp, name + '_distances')
    x = torch.tensor([[1., 2.], [2., 1.]], device=device, requires_grad=True)
    y = torch.tensor([[2., 3.]], device=device, requires_grad=True)
    for a, b, shape in ((x[:0], y, (0, 1)), (x, y[:0], (2, 0)),
                        (x.expand(0, 2, 2), y, (0, 2, 1))):
        value = fn(a, b)
        assert value.shape == shape
        for grad in torch.autograd.grad(value.sum(), (x, y)):
            torch.testing.assert_close(grad, torch.zeros_like(grad))
    assert tp.paired_distances(x[:0], x[:0], metric=name).shape == (0,)
    assert tp.condensed_distances(x[:1], metric=name).shape == (0,)
    for invalid in (float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite'):
            fn(torch.full_like(x, invalid))
    if name in ('clark', 'soergel', 'ruzicka'):
        with pytest.raises(ValueError, match='non-negative'):
            fn(-x)
    if name == 'angular':
        with pytest.raises(ValueError, match='nonzero rows'):
            fn(torch.zeros_like(x))
    with pytest.raises(TypeError, match='dense strided'):
        fn(x.to_sparse())
    with pytest.raises(TypeError, match='float32 or float64'):
        fn(x.half())


@pytest.mark.compiler
def test_vector_extra_compile_amp_master_leaf_and_fake(device):
    torch.compiler.reset()
    amp_dtype = torch.float16 if device == 'cuda' else torch.bfloat16

    def call(x, y):
        with torch.autocast(device, dtype=amp_dtype):
            return torch.stack([operation(x, *args, metric=name).sum()
                                for name in NAMES
                                for operation, args in ((tp.cdist, (y,)),
                                                        (tp.paired_distances, (y,)),
                                                        (tp.condensed_distances, ()))])

    # Float32 master leaves expose autocast/backward mismatches that low
    # precision leaves can conceal by rounding their gradients.
    x = torch.tensor([[.2, .7, 1.1], [.8, .3, .5]], device=device, requires_grad=True)
    y = torch.tensor([[.5, 1.2, .4], [.3, .9, 1.4]], device=device, requires_grad=True)
    expected = call(x, y)
    actual = torch.compile(call, fullgraph=True)(x, y)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    for grad, ref in zip(torch.autograd.grad(actual.sum(), (x, y)),
                         torch.autograd.grad(expected.sum(), (x, y))):
        assert grad.dtype == torch.float32
        assert torch.isfinite(grad).all()
        torch.testing.assert_close(grad, ref, atol=3e-6, rtol=2e-5)
    with torch.autocast(device, dtype=amp_dtype):
        low = tp.angular_distances(x.detach().to(amp_dtype), y.detach().to(amp_dtype))
    assert low.dtype == torch.float32
    from torch._subclasses.fake_tensor import FakeTensorMode
    real_a, real_b = x.detach(), y.detach()
    with FakeTensorMode() as mode:
        a, b = mode.from_tensor(real_a), mode.from_tensor(real_b)
        assert call(a, b).shape == (3 * len(NAMES),)
    torch.compiler.reset()
