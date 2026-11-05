"""Structured Gaussian Bures-Wasserstein values and first derivatives."""
import pytest
import torch
import torchpairwise as tp


def _diagonal_reference(a, av, b, bv, squared):
    value = ((a[..., :, None, :] - b[..., None, :, :]).square().sum(-1)
             + (av.sqrt()[..., :, None, :] - bv.sqrt()[..., None, :, :]).square().sum(-1))
    return value if squared else value.sqrt()


@pytest.mark.parametrize('squared', [False, True])
@pytest.mark.parametrize('dimension', [1, 3])
def test_gaussian_diagonal_closed_form_and_gradients(device, dimension, squared):
    torch.manual_seed(281)
    a = torch.randn(3, dimension, dtype=torch.float64, device=device, requires_grad=True)
    b = torch.randn(2, dimension, dtype=a.dtype, device=device, requires_grad=True)
    av = (torch.rand_like(a) + .5).requires_grad_()
    bv = (torch.rand_like(b) + 2).requires_grad_()
    actual = tp.gaussian_wasserstein_distances(a, av.diag_embed(), b, bv.diag_embed(), squared=squared)
    expected = _diagonal_reference(a, av, b, bv, squared)
    torch.testing.assert_close(actual, expected)
    for value, reference in zip(torch.autograd.grad(actual.sum(), (a, av, b, bv)),
                                torch.autograd.grad(expected.sum(), (a, av, b, bv))):
        torch.testing.assert_close(value, reference)


@pytest.mark.parametrize('squared', [False, True])
def test_gaussian_rotated_commuting_closed_form(device, squared):
    # A fixed non-coordinate eigenbasis gives full, commuting covariances.
    q = torch.tensor([[.6, -.8], [.8, .6]], device=device, dtype=torch.float64)
    av = torch.tensor([[1., 4.], [9., 16.]], device=device, dtype=q.dtype)
    bv = torch.tensor([[4., 25.], [1., 1.]], device=device, dtype=q.dtype)
    a = torch.tensor([[0., 1.], [2., 3.]], device=device, dtype=q.dtype)
    b = a + 1
    actual = tp.gaussian_wasserstein_distances(
        a, q @ av.diag_embed() @ q.T, b, q @ bv.diag_embed() @ q.T, squared=squared)
    torch.testing.assert_close(actual, _diagonal_reference(a, av, b, bv, squared))


@pytest.mark.parametrize('squared', [False, True])
def test_gaussian_noncommuting_trace_reference_symmetry(device, squared):
    a = torch.tensor([[[2., .6], [.6, 1.]], [[1., -.2], [-.2, 3.]]],
                     device=device, dtype=torch.float64)
    b = torch.tensor([[[3., -.5], [-.5, 2.]]], device=device, dtype=a.dtype)
    mean = torch.zeros(2, 2, device=device, dtype=a.dtype)
    other = torch.ones(1, 2, device=device, dtype=a.dtype)
    values, vectors = torch.linalg.eigh(a)
    root = (vectors * values.sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    trace_root = torch.linalg.eigvalsh(root @ b @ root).sqrt().sum(-1)
    reference = 2 + a.diagonal(dim1=-2, dim2=-1).sum(-1) + b.diagonal(dim1=-2, dim2=-1).sum(-1) - 2 * trace_root
    reference = reference if squared else reference.sqrt()
    actual = tp.gaussian_wasserstein_distances(mean, a, other, b, squared=squared)
    torch.testing.assert_close(actual[:, 0], reference)
    torch.testing.assert_close(actual, tp.gaussian_wasserstein_distances(
        other, b, mean, a, squared=squared).T)


@pytest.mark.parametrize('squared', [False, True])
@pytest.mark.parametrize('repeated', [False, True])
def test_gaussian_spd_gradcheck_including_repeated_eigenvalues(device, repeated, squared):
    mean = torch.tensor([[.2, -.3]], device=device, dtype=torch.float64, requires_grad=True)
    other = torch.tensor([[1.2, .4]], device=device, dtype=mean.dtype, requires_grad=True)
    covariance = torch.eye(2, device=device, dtype=mean.dtype)[None]
    target = covariance * 2 if repeated else torch.tensor([[[2., .4], [.4, 3.]]], device=device, dtype=mean.dtype)
    covariance, target = covariance.requires_grad_(), target.requires_grad_()

    def call(m, a, n, b):
        # Numerical perturbations remain symmetric; the public API intentionally
        # rejects materially nonsymmetric covariance arguments.
        a, b = (a + a.transpose(-1, -2)) / 2, (b + b.transpose(-1, -2)) / 2
        return tp.gaussian_wasserstein_distances(m, a, n, b, squared=squared)

    assert torch.autograd.gradcheck(call, (mean, covariance, other, target), atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('squared', [False, True])
def test_gaussian_identity_zero_and_translation(device, squared):
    mean = torch.tensor([[1., 2.], [-2., 3.]], device=device, dtype=torch.float64, requires_grad=True)
    covariance = torch.eye(2, device=device, dtype=mean.dtype).expand(2, 2, 2).clone().requires_grad_()
    zero = tp.paired_gaussian_wasserstein_distances(mean, covariance, mean, covariance, squared=squared)
    torch.testing.assert_close(zero, torch.zeros_like(zero), atol=0, rtol=0)
    for gradient in torch.autograd.grad(zero.sum(), (mean, covariance)):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)
    translated = tp.paired_gaussian_wasserstein_distances(mean, covariance, mean + 2, covariance, squared=squared)
    torch.testing.assert_close(translated, torch.full_like(translated, 8 if squared else 8 ** .5))
    torch.testing.assert_close(torch.autograd.grad(translated.sum(), covariance)[0], torch.zeros_like(covariance))
    self_map = tp.gaussian_wasserstein_distances(mean, covariance, squared=squared)
    torch.testing.assert_close(self_map.diagonal(), torch.zeros_like(zero), atol=0, rtol=0)


@pytest.mark.parametrize('squared', [False, True])
def test_gaussian_psd_boundary_policy(device, squared):
    mean = torch.zeros(3, 2, device=device, dtype=torch.float64, requires_grad=True)
    variances = torch.tensor([[0., 0.], [0., 4.], [1., 0.]], device=device, dtype=mean.dtype)
    covariance = variances.diag_embed().requires_grad_()
    actual = tp.gaussian_wasserstein_distances(mean, covariance, squared=squared)
    reference = _diagonal_reference(mean, variances, mean, variances, squared)
    torch.testing.assert_close(actual, reference)
    assert (actual >= 0).all()
    for gradient in torch.autograd.grad(actual.sum(), (mean, covariance)):
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize('squared', [False, True])
def test_gaussian_broadcast_noncontiguous_modes_and_gradients(device, squared):
    torch.manual_seed(315)
    mean = torch.randn(2, 1, 3, 4, device=device, dtype=torch.float64)[..., ::2].requires_grad_()
    covariance = (torch.rand(1, 4, 3, 2, device=device, dtype=mean.dtype) + 1).diag_embed()
    covariance = covariance.transpose(-1, -2).requires_grad_()
    other = torch.randn(1, 4, 3, 2, device=device, dtype=mean.dtype, requires_grad=True)
    target = (torch.rand(2, 1, 3, 2, device=device, dtype=mean.dtype) + 2).diag_embed().requires_grad_()
    full = tp.gaussian_wasserstein_distances(mean, covariance, other, target, squared=squared)
    assert full.shape == (2, 4, 3, 3)
    paired = tp.paired_gaussian_wasserstein_distances(mean, covariance, other, target, squared=squared)
    torch.testing.assert_close(paired, full.diagonal(dim1=-2, dim2=-1))
    inputs = (mean, covariance, other, target)
    for gradient, reference in zip(torch.autograd.grad(paired.sum(), inputs, retain_graph=True),
                                    torch.autograd.grad(full.diagonal(dim1=-2, dim2=-1).sum(), inputs)):
        torch.testing.assert_close(gradient, reference)
    self_map = tp.gaussian_wasserstein_distances(mean, covariance, squared=squared)
    condensed = tp.condensed_gaussian_wasserstein_distances(mean, covariance, squared=squared)
    i, j = torch.triu_indices(3, 3, 1, device=device)
    torch.testing.assert_close(condensed, self_map[..., i, j])
    for gradient, reference in zip(torch.autograd.grad(condensed.sum(), (mean, covariance)),
                                    torch.autograd.grad(self_map[..., i, j].sum(), (mean, covariance))):
        torch.testing.assert_close(gradient, reference)


def test_gaussian_empty_samples_and_batch(device):
    mean = torch.ones(2, 2, device=device, dtype=torch.float64, requires_grad=True)
    covariance = torch.eye(2, device=device, dtype=mean.dtype).expand(2, 2, 2).requires_grad_()
    for n in (0, 1):
        value = tp.condensed_gaussian_wasserstein_distances(mean[:n], covariance[:n])
        assert value.shape == (0,)
        for gradient in torch.autograd.grad(value.sum(), (mean, covariance)):
            torch.testing.assert_close(gradient, torch.zeros_like(gradient))
    assert tp.gaussian_wasserstein_distances(mean[:0], covariance[:0], mean, covariance).shape == (0, 2)
    assert tp.paired_gaussian_wasserstein_distances(mean[:0], covariance[:0], mean[:0], covariance[:0]).shape == (0,)
    value = tp.gaussian_wasserstein_distances(mean.expand(0, 2, 2), covariance.expand(0, 2, 2, 2))
    assert value.shape == (0, 2, 2)


def test_gaussian_nearby_covariances_avoid_trace_cancellation(device):
    mean = torch.zeros(1, 2, device=device)
    variance = torch.ones(1, 2, device=device)
    other = variance + torch.tensor([[1e-4, 2e-4]], device=device)
    actual = tp.gaussian_wasserstein_distances(
        mean, variance.diag_embed(), mean, other.diag_embed(), squared=True)
    expected = (variance.sqrt() - other.sqrt()).square().sum(-1).unsqueeze(-1)
    assert actual.item() > 0
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_gaussian_covariance_domain_validation_and_roundoff(device, dtype):
    mean = torch.zeros(1, 2, device=device, dtype=dtype)
    identity = torch.eye(2, device=device, dtype=dtype)[None]
    for bad, message in ((identity * float('nan'), 'finite'),
                          (torch.tensor([[[1., .1], [0., 1.]]], device=device, dtype=dtype), 'symmetric'),
                          (torch.tensor([[[1., 0.], [0., -.1]]], device=device, dtype=dtype), 'positive semidefinite')):
        with pytest.raises(ValueError, match=message):
            tp.gaussian_wasserstein_distances(mean, bad)
    tiny = 8 * torch.finfo(dtype).eps
    covariance = torch.tensor([[[1., 0.], [0., -tiny]]], device=device, dtype=dtype)
    projected = torch.tensor([[[1., 0.], [0., 0.]]], device=device, dtype=dtype)
    torch.testing.assert_close(tp.gaussian_wasserstein_distances(mean, covariance, mean, identity),
                               tp.gaussian_wasserstein_distances(mean, projected, mean, identity))
    with pytest.raises(ValueError, match='finite'):
        tp.gaussian_wasserstein_distances(mean + float('inf'), identity)


def test_gaussian_shape_type_validation_and_autocast(device):
    mean = torch.ones(2, 2, device=device)
    covariance = torch.eye(2, device=device).expand(2, 2, 2)
    with pytest.raises(ValueError, match='both'):
        tp.gaussian_wasserstein_distances(mean, covariance, mean)
    with pytest.raises(ValueError, match='dimensions must match'):
        tp.gaussian_wasserstein_distances(mean, covariance[:1])
    with pytest.raises(ValueError, match='equal sample counts'):
        tp.paired_gaussian_wasserstein_distances(mean, covariance, mean[:1], covariance[:1])
    with pytest.raises(ValueError, match='same dtype'):
        tp.gaussian_wasserstein_distances(mean, covariance.double())
    with pytest.raises(TypeError, match='boolean'):
        tp.gaussian_wasserstein_distances(mean, covariance, squared=1)
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    low = mean.to(dtype).requires_grad_()
    low_covariance = covariance.to(dtype).requires_grad_()
    with pytest.raises(TypeError, match='float32 or float64'):
        tp.gaussian_wasserstein_distances(low, low_covariance)
    with torch.autocast(device, dtype=dtype):
        actual = tp.gaussian_wasserstein_distances(low, low_covariance, low + 1, low_covariance * 2)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, tp.gaussian_wasserstein_distances(
        low.float(), low_covariance.float(), (low + 1).float(), (low_covariance * 2).float()))
    for gradient in torch.autograd.grad(actual.sum(), (low, low_covariance)):
        assert torch.isfinite(gradient).all()


@pytest.mark.compiler
def test_gaussian_compile_first_gradients_all_modes(device):
    torch.compiler.reset()
    mean = torch.tensor([[0., 1.], [2., 3.]], device=device, requires_grad=True)
    covariance = torch.eye(2, device=device).expand(2, 2, 2).clone().requires_grad_()

    def call(m, c):
        return (tp.gaussian_wasserstein_distances(m, c, m + 1, c * 2).sum()
                + tp.paired_gaussian_wasserstein_distances(m, c, m + 2, c * 3, squared=True).sum()
                + tp.condensed_gaussian_wasserstein_distances(m, c).sum())

    expected = call(mean, covariance)
    actual = torch.compile(call, fullgraph=True)(mean, covariance)
    torch.testing.assert_close(actual, expected)
    for gradient, reference in zip(torch.autograd.grad(actual, (mean, covariance)),
                                    torch.autograd.grad(expected, (mean, covariance))):
        torch.testing.assert_close(gradient, reference)
    torch.compiler.reset()
