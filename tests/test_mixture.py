"""Analytic mixture-density and weighted component-transport contracts."""
import math

import pytest
import torch
import torchpairwise as tp


FAMILIES = (
    'gaussian_mixture_overlap_kernel',
    'gaussian_mixture_l2_distances',
    'gaussian_mixture_cauchy_schwarz_divergences',
    'gaussian_mixture_wasserstein_distances',
    'gaussian_mixture_entropic_transport_costs',
    'gaussian_mixture_sinkhorn_divergences',
)


def _one_dimensional_overlap(a, av, aw, b, bv, bw):
    aw, bw = aw / aw.sum(-1, keepdim=True), bw / bw.sum(-1, keepdim=True)
    variance = av[..., :, None] + bv[..., None, :]
    return ((aw[..., :, None] * bw[..., None, :])
            * (-(a[..., :, None] - b[..., None, :]).square() / (2 * variance)).exp()
            / (2 * math.pi * variance).sqrt()).sum((-1, -2))


def test_mixture_weighted_overlap_and_density_gradients(device):
    a = torch.tensor([[[-1.], [2.]]], dtype=torch.float64, device=device, requires_grad=True)
    b = torch.tensor([[[.2], [3.], [4.]]], dtype=a.dtype, device=device, requires_grad=True)
    av = torch.tensor([[.7, 1.2]], dtype=a.dtype, device=device, requires_grad=True)
    bv = torch.tensor([[1.1, 2., .9]], dtype=a.dtype, device=device, requires_grad=True)
    aw = torch.tensor([[2., 3.]], dtype=a.dtype, device=device, requires_grad=True)
    bw = torch.tensor([[1., 4., 2.]], dtype=a.dtype, device=device, requires_grad=True)
    overlap = tp.gaussian_mixture_overlap_kernel(a, av[..., None, None], aw,
                                                 b, bv[..., None, None], bw,
                                                 component_chunk_size=1)
    reference = _one_dimensional_overlap(a[..., 0], av, aw, b[..., 0], bv, bw)
    torch.testing.assert_close(overlap[0, 0], reference[0])
    inputs = (a, av, aw, b, bv, bw)
    for actual, expected in zip(torch.autograd.grad(overlap.sum(), inputs, retain_graph=True),
                                torch.autograd.grad(reference.sum(), inputs, retain_graph=True)):
        torch.testing.assert_close(actual, expected)
    own_a = _one_dimensional_overlap(a[..., 0], av, aw, a[..., 0], av, aw)
    own_b = _one_dimensional_overlap(b[..., 0], bv, bw, b[..., 0], bv, bw)
    squared = tp.gaussian_mixture_l2_distances(a, av[..., None, None], aw,
                                              b, bv[..., None, None], bw, squared=True)
    cs = tp.gaussian_mixture_cauchy_schwarz_divergences(a, av[..., None, None], aw,
                                                       b, bv[..., None, None], bw)
    torch.testing.assert_close(squared[0, 0], (own_a + own_b - 2 * reference)[0])
    torch.testing.assert_close(cs[0, 0], (.5 * own_a.log() + .5 * own_b.log() - reference.log())[0])


def test_mixture_cs_stays_finite_when_overlap_underflows(device):
    mean = torch.zeros(1, 1, 1, dtype=torch.float64, device=device)
    covariance = torch.ones(1, 1, 1, 1, dtype=mean.dtype, device=device)
    weights = torch.ones(1, 1, dtype=mean.dtype, device=device)
    overlap = tp.gaussian_mixture_overlap_kernel(mean, covariance, weights, mean + 100, covariance, weights)
    cs = tp.gaussian_mixture_cauchy_schwarz_divergences(mean, covariance, weights,
                                                       mean + 100, covariance, weights)
    torch.testing.assert_close(overlap, torch.zeros_like(overlap), atol=0, rtol=0)
    torch.testing.assert_close(cs, torch.full_like(cs, 2500.))


@pytest.mark.parametrize('family', FAMILIES)
def test_mixture_broadcast_modes_chunking_and_permutation(device, family):
    torch.manual_seed(490)
    mean = torch.randn(2, 1, 3, 2, 2, dtype=torch.float64, device=device)
    covariance = (torch.rand(1, 2, 3, 2, 2, dtype=mean.dtype, device=device) + .5).diag_embed()
    weights = torch.tensor([[1., 3.], [2., 1.], [1., 1.]], dtype=mean.dtype, device=device)
    fn, paired, condensed = (getattr(tp, prefix + family) for prefix in ('', 'paired_', 'condensed_'))
    kwargs = {'max_iter': 15} if family.endswith(('transport_costs', 'sinkhorn_divergences')) else {}
    full = fn(mean, covariance, weights, **kwargs)
    chunked = fn(mean, covariance, weights, chunk_size=1, **kwargs)
    assert full.shape == (2, 2, 3, 3)
    torch.testing.assert_close(full, chunked)
    aligned = paired(mean, covariance, weights, mean, covariance, weights, **kwargs)
    torch.testing.assert_close(aligned, full.diagonal(dim1=-2, dim2=-1))
    i, j = torch.triu_indices(3, 3, 1, device=device)
    torch.testing.assert_close(condensed(mean, covariance, weights, **kwargs), full[..., i, j])
    permuted = fn(mean, covariance, weights, mean.flip(-2), covariance.flip(-3), weights.flip(-1), **kwargs)
    torch.testing.assert_close(full, permuted, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(full, full.transpose(-1, -2), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize('family', FAMILIES)
def test_mixture_padding_zero_weights_and_empty_gradients(device, family):
    mean = torch.tensor([[[0.], [float('nan')], [float('nan')]],
                         [[2.], [float('nan')], [float('nan')]]],
                        dtype=torch.float64, device=device, requires_grad=True)
    covariance = torch.ones(2, 3, 1, 1, dtype=mean.dtype, device=device)
    covariance[:, 1:] = float('nan')
    covariance.requires_grad_()
    weights = torch.tensor([[1., float('nan'), float('nan')],
                            [3., float('nan'), float('nan')]],
                           dtype=mean.dtype, device=device, requires_grad=True)
    lengths = torch.ones(2, dtype=torch.int64, device=device)
    fn = getattr(tp, family)
    kwargs = {'max_iter': 10} if family.endswith(('transport_costs', 'sinkhorn_divergences')) else {}
    actual = fn(mean, covariance, weights, x1_lengths=lengths, chunk_size=1, **kwargs)
    reference = fn(mean[:, :1], covariance[:, :1], weights[:, :1], **kwargs)
    torch.testing.assert_close(actual, reference)
    for gradient in torch.autograd.grad(actual.sum(), (mean, covariance, weights), retain_graph=True):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient[:, 1:], torch.zeros_like(gradient[:, 1:]), atol=0, rtol=0)
    empty = fn(mean[:0], covariance[:0], weights[:0], x1_lengths=lengths[:0], **kwargs)
    assert empty.shape == (0, 0)
    for gradient in torch.autograd.grad(empty.sum(), (mean, covariance, weights)):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)
    # A tile containing exclusively zero-mass components must have finite gradients.
    clean_mean = torch.nan_to_num(mean.detach()).requires_grad_()
    clean_covariance = torch.ones_like(covariance).requires_grad_()
    clean_weights = torch.tensor([[1., 0., 0.], [1., 0., 0.]], dtype=mean.dtype, device=device, requires_grad=True)
    options = {'component_chunk_size': 1} if family in FAMILIES[:3] else kwargs
    result = fn(clean_mean, clean_covariance, clean_weights, **options)
    for gradient in torch.autograd.grad(result.sum(), (clean_mean, clean_covariance, clean_weights)):
        assert torch.isfinite(gradient).all()


def test_mixture_exact_unequal_masses_values_and_mass_derivatives(device):
    mean = torch.tensor([[[0.], [4.]]], dtype=torch.float64, device=device, requires_grad=True)
    other = torch.tensor([[[1.], [6.]]], dtype=mean.dtype, device=device, requires_grad=True)
    covariance = torch.ones(1, 2, 1, 1, dtype=mean.dtype, device=device, requires_grad=True)
    weights = torch.tensor([[.8, .2]], dtype=mean.dtype, device=device, requires_grad=True)
    target = torch.tensor([[.3, .7]], dtype=mean.dtype, device=device, requires_grad=True)
    actual = tp.gaussian_mixture_wasserstein_distances(mean, covariance, weights,
                                                       other, covariance, target, squared=True)
    a, b = weights / weights.sum(-1, keepdim=True), target / target.sum(-1, keepdim=True)
    costs = (mean[..., :, None, :] - other[..., None, :, :]).square().sum(-1)
    reference = (b[:, 0] * costs[:, 0, 0] + (a[:, 0] - b[:, 0]) * costs[:, 0, 1]
                 + a[:, 1] * costs[:, 1, 1])
    torch.testing.assert_close(actual[0, 0], torch.tensor(19.1, dtype=mean.dtype, device=device))
    for gradient, expected in zip(torch.autograd.grad(actual.sum(), (mean, other, weights, target)),
                                  torch.autograd.grad(reference.sum(), (mean, other, weights, target))):
        torch.testing.assert_close(gradient, expected)
    rooted = tp.gaussian_mixture_wasserstein_distances(mean, covariance, weights, other, covariance, target)
    torch.testing.assert_close(rooted.square(), actual)


def test_mixture_exact_rectangular_psd_and_regularized_conventions(device):
    mean = torch.tensor([[[0.], [3.]]], dtype=torch.float64, device=device)
    other = torch.tensor([[[1.], [2.], [6.]]], dtype=mean.dtype, device=device)
    covariance = torch.zeros(1, 2, 1, 1, dtype=mean.dtype, device=device)
    target_covariance = torch.zeros(1, 3, 1, 1, dtype=mean.dtype, device=device)
    weights = torch.tensor([[.25, .75]], dtype=mean.dtype, device=device)
    target = torch.tensor([[.5, .3, .2]], dtype=mean.dtype, device=device)
    actual = tp.gaussian_mixture_wasserstein_distances(mean, covariance, weights,
                                                       other, target_covariance, target, squared=True)
    torch.testing.assert_close(actual, torch.full_like(actual, 3.35))
    # Uniform self transport on two equal-covariance components has a closed
    # form full KL objective; its debiased version is exactly zero.
    weights = torch.ones_like(weights)
    epsilon = .7
    raw = tp.gaussian_mixture_entropic_transport_costs(mean, covariance, weights,
                                                       regularization=epsilon, max_iter=200)
    expected = epsilon * (math.log(2) - math.log1p(math.exp(-9 / epsilon)))
    torch.testing.assert_close(raw, torch.full_like(raw, expected))
    debiased = tp.gaussian_mixture_sinkhorn_divergences(mean, covariance, weights,
                                                       regularization=epsilon, max_iter=200)
    torch.testing.assert_close(debiased, torch.zeros_like(debiased), atol=1e-14, rtol=0)


@pytest.mark.parametrize('family', ['gaussian_mixture_overlap_kernel', 'gaussian_mixture_entropic_transport_costs'])
def test_mixture_spd_gradcheck_means_covariance_weights(device, family):
    mean = torch.tensor([[[.2], [1.1]]], dtype=torch.float64, device=device, requires_grad=True)
    covariance = torch.tensor([[[[1.]], [[2.]]]], dtype=mean.dtype, device=device, requires_grad=True)
    weights = torch.tensor([[.4, .6]], dtype=mean.dtype, device=device, requires_grad=True)
    other = mean.detach() + .7
    target_covariance = covariance.detach() * 1.3
    target_weights = weights.detach().flip(-1)
    kwargs = {'max_iter': 12} if family.endswith('transport_costs') else {'component_chunk_size': 1}
    assert torch.autograd.gradcheck(lambda m, c, w: getattr(tp, family)(
        m, c, w, other, target_covariance, target_weights, **kwargs), (mean, covariance, weights),
        atol=2e-5, rtol=2e-4)


def test_mixture_covariance_and_parameter_validation(device):
    mean = torch.zeros(1, 1, 1, dtype=torch.float64, device=device)
    covariance = torch.ones(1, 1, 1, 1, dtype=mean.dtype, device=device)
    weights = torch.ones(1, 1, dtype=mean.dtype, device=device)
    with pytest.raises(ValueError, match='positive definite'):
        tp.gaussian_mixture_overlap_kernel(mean, covariance * 0, weights)
    with pytest.raises(ValueError, match='positive semidefinite'):
        tp.gaussian_mixture_wasserstein_distances(mean, -covariance, weights)
    with pytest.raises(ValueError, match='positive total'):
        tp.gaussian_mixture_overlap_kernel(mean, covariance, weights * 0)
    with pytest.raises(ValueError, match='non-negative'):
        tp.gaussian_mixture_overlap_kernel(mean, covariance, -weights)
    with pytest.raises(ValueError, match='all three'):
        tp.gaussian_mixture_overlap_kernel(mean, covariance, weights, mean)
    with pytest.raises(TypeError, match='integer'):
        tp.gaussian_mixture_overlap_kernel(mean, covariance, weights, chunk_size=True)
    with pytest.raises(ValueError, match='between 1'):
        tp.gaussian_mixture_overlap_kernel(mean, covariance, weights,
                                           x1_lengths=torch.zeros(1, dtype=torch.int64, device=device))
    with pytest.raises(ValueError, match='regularization'):
        tp.gaussian_mixture_sinkhorn_divergences(mean, covariance, weights, regularization=0)
    with pytest.raises(ValueError, match='4096'):
        tp.gaussian_mixture_wasserstein_distances(mean.expand(1, 65, 1), covariance.expand(1, 65, 1, 1),
                                                   weights.expand(1, 65))
