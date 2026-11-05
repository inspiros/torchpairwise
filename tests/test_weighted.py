"""Weighted cosine/correlation contracts and independent SciPy references."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist
import torch
import torchpairwise as tp


@pytest.mark.parametrize('metric', ['cosine', 'correlation'])
def test_weighted_reference_modes_and_gradients(metric, device):
    torch.manual_seed(37)
    named = getattr(tp, metric + '_distances')
    x = torch.randn(2, 1, 3, 10, device=device, dtype=torch.float64)[..., ::2].requires_grad_()
    y = torch.randn(1, 2, 4, 10, device=device, dtype=torch.float64)[..., ::2].requires_grad_()
    w = torch.rand(2, 2, 5, device=device, dtype=torch.float64) + .2
    w[0, 0, 0] = 0
    w.requires_grad_()
    a, b, weights = [value.detach().cpu().numpy() for value in (x, y, w)]
    expected = np.asarray([[cdist(a[i, 0], b[0, j], metric=metric, w=weights[i, j])
                            for j in range(2)] for i in range(2)])
    matrix = named(x, y, w=w)
    torch.testing.assert_close(matrix, torch.as_tensor(expected, device=device))
    torch.testing.assert_close(tp.cdist(x, y, metric=metric, w=w), matrix)
    torch.testing.assert_close(named(x, y, w=w * 7.5), matrix)
    # Omission and explicit None retain precisely the existing native path.
    torch.testing.assert_close(named(x, y), named(x, y, w=None), atol=0, rtol=0)

    paired = tp.paired_distances(x, y[..., :3, :], metric=metric, w=w)
    diagonal = matrix[..., :3].diagonal(dim1=-2, dim2=-1)
    torch.testing.assert_close(paired, diagonal)
    for actual, reference in zip(torch.autograd.grad(paired.sum(), (x, y, w)),
                                 torch.autograd.grad(diagonal.sum(), (x, y, w))):
        torch.testing.assert_close(actual, reference)
    i, j = torch.triu_indices(3, 3, 1, device=device)
    condensed = tp.condensed_distances(x, metric=metric, w=w)
    triangle = named(x, w=w)[..., i, j]
    torch.testing.assert_close(condensed, triangle)
    for actual, reference in zip(torch.autograd.grad(condensed.sum(), (x, w)),
                                 torch.autograd.grad(triangle.sum(), (x, w))):
        torch.testing.assert_close(actual, reference)

    # Numerical derivatives independently check both input and weight gradients.
    p, q = [torch.randn(2, 4, device=device, dtype=torch.float64).requires_grad_()
            for _ in range(2)]
    weight = (torch.rand(4, device=device, dtype=torch.float64) + .5).requires_grad_()
    for fn in (lambda a, b, c: named(a, b, w=c),
               lambda a, b, c: tp.paired_distances(a, b, metric=metric, w=c)):
        assert torch.autograd.gradcheck(fn, (p, q, weight), fast_mode=True)


def test_weighted_degenerate_and_zero_weight_features(device):
    x = torch.tensor([[8., 0., 0.], [-9., 1., 0.], [4., 0., 2.]],
                     device=device, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    w = torch.tensor([0., 2., 3.], device=device, dtype=x.dtype, requires_grad=True)
    cross = tp.cosine_distances(x, y, w=w)
    torch.testing.assert_close(cross[0], torch.ones(3, device=device, dtype=x.dtype))
    torch.testing.assert_close(cross[:, 0], torch.ones(3, device=device, dtype=x.dtype))
    for args in ((x,), (x, x)):
        torch.testing.assert_close(tp.cosine_distances(*args, w=w).diagonal(),
                                   torch.zeros(3, device=device, dtype=x.dtype))
    torch.testing.assert_close(tp.paired_distances(x, x, metric='cosine', w=w),
                               torch.zeros(3, device=device, dtype=x.dtype))
    zero_pair = tp.paired_distances(x[:1], y[:1], metric='cosine', w=w)
    torch.testing.assert_close(zero_pair, torch.ones(1, device=device, dtype=x.dtype))
    for gradient in torch.autograd.grad(zero_pair.sum(), (x, y, w)):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))
    # A row constant on the positive-weight support has undefined correlation,
    # even if its ignored coordinate differs or the weights are not uniform.
    constant = torch.tensor([[7., 2., 2.]], device=device, dtype=x.dtype)
    assert tp.correlation_distances(constant, x, w=w).isnan().all()
    assert tp.paired_distances(constant, constant, metric='correlation', w=w).isnan().all()
    single = torch.tensor([0., 1., 0.], device=device, dtype=x.dtype)
    assert tp.condensed_distances(x, metric='correlation', w=single).isnan().all()
    for metric in ('cosine', 'correlation'):
        named = getattr(tp, metric + '_distances')
        # Large ignored coordinates neither overflow nor affect input gradients.
        a = torch.tensor([[1e30, 2e20, -3e20]], device=device, requires_grad=True)
        b = torch.tensor([[-1e30, 4e20, 1e20]], device=device, requires_grad=True)
        weight = torch.tensor([0., 2., 3.], device=device)
        actual = named(a, b, w=weight)
        reference = cdist(a.detach().cpu().double().numpy()[:, 1:],
                          b.detach().cpu().double().numpy()[:, 1:],
                          metric=metric, w=np.asarray([2., 3.]))
        torch.testing.assert_close(actual, torch.as_tensor(reference, device=device, dtype=a.dtype))
        for gradient in torch.autograd.grad(actual.sum(), (a, b)):
            assert torch.isfinite(gradient).all()
            assert gradient[0, 0] == 0
        # Raw weight sums overflow, but individually finite weights are valid.
        large = torch.full_like(w, torch.finfo(w.dtype).max)
        torch.testing.assert_close(named(x, y, w=large), named(x, y, w=torch.ones_like(w)))
        for n in (0, 1):
            empty = tp.condensed_distances(x[:n], metric=metric, w=w)
            assert empty.shape == (0,)
            for gradient in torch.autograd.grad(empty.sum(), (x, w)):
                torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize('metric', ['cosine', 'correlation'])
def test_weighted_parameter_validation(metric, device):
    x = torch.ones(2, 3, device=device)
    named = getattr(tp, metric + '_distances')
    with pytest.raises(TypeError, match='w must be a tensor'):
        named(x, w=[1., 1., 1.])
    with pytest.raises(ValueError, match='w must end with shape'):
        named(x, w=torch.ones(2, device=device))
    with pytest.raises(ValueError, match='same dtype and device'):
        named(x, w=torch.ones(3, device=device, dtype=torch.float64))
    with pytest.raises(ValueError, match='not broadcastable'):
        named(x.expand(2, 2, 3), w=torch.ones(3, 3, device=device))
    for value in (-1., float('nan'), float('inf')):
        w = torch.tensor([1., value, 1.], device=device)
        with pytest.raises(ValueError, match='non-negative'):
            named(x, w=w)
    # Positive total is required in each batch and in every comparison mode.
    w = torch.tensor([[1., 2., 3.], [0., 0., 0.]], device=device)
    for fn in (lambda: named(x, w=w),
               lambda: tp.paired_distances(x, x, metric=metric, w=w),
               lambda: tp.condensed_distances(x, metric=metric, w=w)):
        with pytest.raises(ValueError, match='positive total'):
            fn()
    torch.testing.assert_close(tp.wminkowski_distances(x, w=torch.zeros(3, device=device)),
                               torch.zeros(2, 2, device=device))


@pytest.mark.compiler
def test_weighted_compiled_modes(device):
    torch.compiler.reset()
    x, y = [torch.randn(3, 4, device=device).requires_grad_() for _ in range(2)]
    w = (torch.rand(4, device=device) + .5).requires_grad_()

    def call(a, b, c):
        return torch.stack([value.sum() for metric in ('cosine', 'correlation')
                            for value in (getattr(tp, metric + '_distances')(a, b, w=c),
                                          tp.paired_distances(a, b, metric=metric, w=c),
                                          tp.condensed_distances(a, metric=metric, w=c))])

    expected = call(x, y, w)
    actual = torch.compile(call, fullgraph=True)(x, y, w)
    torch.testing.assert_close(actual, expected)
    for gradient, reference in zip(torch.autograd.grad(actual.sum(), (x, y, w)),
                                   torch.autograd.grad(expected.sum(), (x, y, w))):
        torch.testing.assert_close(gradient, reference, atol=2e-5, rtol=2e-4)
    torch.compiler.reset()


def test_weighted_autocast(device):
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    x, y = [torch.randn(3, 4, device=device).to(dtype).requires_grad_() for _ in range(2)]
    w = (torch.rand(4, device=device).to(dtype) + .5).requires_grad_()
    for metric in ('cosine', 'correlation'):
        named = getattr(tp, metric + '_distances')
        for fn, args, options in ((named, (x, y), {}),
                                   (tp.paired_distances, (x, y), {'metric': metric}),
                                   (tp.condensed_distances, (x,), {'metric': metric})):
            with torch.autocast(device, dtype=dtype):
                actual = fn(*args, w=w, **options)
            assert actual.dtype == torch.float32
            torch.testing.assert_close(actual, fn(*(a.float() for a in args), w=w.float(), **options))
            for gradient in torch.autograd.grad(actual.sum(), (*args, w)):
                assert torch.isfinite(gradient).all()
