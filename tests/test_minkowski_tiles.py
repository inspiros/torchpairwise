"""Direct-difference tiles and radial kernels at coincident input rows."""
import math

import pytest
import torch

import torchpairwise as tp


RADIAL_CASES = [
    ('laplacian', {'gamma': .7}),
    *[('matern', {'length_scale': .7, 'nu': nu}) for nu in (.5, 1.5, 2.5)],
    ('rational_quadratic', {'length_scale': 1.7, 'alpha': .4}),
]


def _radial_reference(x, y, kernel, kwargs):
    # Force direct cdist, retaining its zero subgradient at coincident rows.
    if kernel == 'laplacian':
        return (-kwargs['gamma'] * torch.cdist(x, y, p=1)).exp()
    distance = torch.cdist(x, y, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    if kernel == 'rational_quadratic':
        return (1 + distance.square() / (2 * kwargs['alpha'] * kwargs['length_scale'] ** 2)).pow(-kwargs['alpha'])
    nu = kwargs['nu']
    r = distance * math.sqrt(2 * nu) / kwargs['length_scale']
    polynomial = 1 if nu == .5 else 1 + r
    if nu == 2.5:
        polynomial = polynomial + r.square() / 3
    return polynomial * (-r).exp()


@pytest.mark.parametrize('features', [1, 3, 33, 130])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_minkowski_tile_values(features, batched, dtype, device):
    torch.manual_seed(815)
    batch = (2,) if batched else ()
    x = torch.randn(*batch, 37, features, dtype=dtype, device=device)
    y = torch.randn(*batch, 70, features, dtype=dtype, device=device)
    delta = (x.double().unsqueeze(-2) - y.double().unsqueeze(-3)).abs()
    cases = [
        (tp.sqeuclidean_distances, {}, delta.square().sum(-1)),
        (tp.manhattan_distances, {}, delta.sum(-1)),
        (tp.minkowski_distances, {'p': 3}, delta.pow(3).sum(-1).pow(1 / 3)),
        (tp.chebyshev_distances, {}, delta.amax(-1)),
    ]
    for function, kwargs, expected in cases:
        actual = function(x, y, **kwargs)
        tolerance = 3e-6 if dtype == torch.float32 else 2e-13
        torch.testing.assert_close(actual, expected.to(dtype), rtol=tolerance, atol=tolerance)
    # Several public distances still use cdist in this checkout. Exercise the
    # rewritten primitive directly as well, including its negative-inf branch.
    for power, expected in [(0, delta.new_full(delta.shape[:-1], features)),
                            (1, delta.sum(-1)), (2, delta.square().sum(-1)),
                            (3, delta.pow(3).sum(-1)),
                            (float('inf'), delta.amax(-1)),
                            (-float('inf'), delta.amin(-1))]:
        actual = torch.ops.torchpairwise._ppminkowski(x, y, power)
        torch.testing.assert_close(actual, expected.to(dtype), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('kernel,kwargs', RADIAL_CASES)
def test_minkowski_radial_gradients(kernel, kwargs, device):
    torch.manual_seed(816)
    x = torch.randn(5, 4, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.randn(7, 4, dtype=x.dtype, device=device)
    y[0] = x[0].detach()  # An exactly duplicated row exercises zero distance.
    y.requires_grad_()
    function = getattr(tp, kernel + '_kernel')
    assert torch.autograd.gradcheck(lambda a, b: function(a, b, **kwargs),
                                    (x, y), atol=1e-5, fast_mode=True)
    actual = function(x, y, **kwargs)
    expected = _radial_reference(x, y, kernel, kwargs)
    weights = torch.randn_like(actual)
    gradients = torch.autograd.grad((actual * weights).sum(), (x, y), retain_graph=True)
    references = torch.autograd.grad((expected * weights).sum(), (x, y))
    for gradient, reference in zip(gradients, references):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, reference, atol=1e-12, rtol=1e-11)
    for gradient in torch.autograd.grad(actual[0, 0], (x, y)):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)


@pytest.mark.parametrize('kernel,kwargs', RADIAL_CASES)
def test_minkowski_radial_values(kernel, kwargs, device):
    torch.manual_seed(817)
    function = getattr(tp, kernel + '_kernel')
    for dtype in (torch.float32, torch.float64):
        # Multiple leading dimensions also exercise flattening broadcast batches.
        x = torch.randn(2, 1, 5, 6, dtype=dtype, device=device)[..., ::2]
        y = torch.randn(1, 2, 7, 6, dtype=dtype, device=device)[..., ::2]
        expected = _radial_reference(x.double(), y.double(), kernel, kwargs).to(dtype)
        torch.testing.assert_close(function(x, y, **kwargs), expected)
