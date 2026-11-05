"""SPD covariance geometry: independent values, invariances, and first derivatives."""
import pytest
import torch
import torchpairwise as tp


METRICS = ('log_euclidean', 'affine_invariant', 'stein')


def _function(metric, mode=''):
    return getattr(tp, f'{mode}{metric}_distances')


def _diagonal_reference(a, b, metric, squared):
    a, b = a[..., :, None, :], b[..., None, :, :]
    if metric == 'stein':
        value = .5 * torch.log1p((a - b).square() / (4 * a * b)).sum(-1)
    else:
        value = (a.log() - b.log()).square().sum(-1)
    return value if squared else value.sqrt()


@pytest.mark.parametrize('metric', METRICS)
@pytest.mark.parametrize('squared', [False, True])
@pytest.mark.parametrize('dimension', [1, 3])
def test_covariance_diagonal_reference_and_gradients(device, metric, squared, dimension):
    torch.manual_seed(804)
    a = (torch.rand(3, dimension, device=device, dtype=torch.float64) + .5).requires_grad_()
    b = (torch.rand(2, dimension, device=device, dtype=a.dtype) + 2).requires_grad_()
    actual = _function(metric)(a.diag_embed(), b.diag_embed(), squared=squared)
    expected = _diagonal_reference(a, b, metric, squared)
    torch.testing.assert_close(actual, expected)
    for gradient, reference in zip(torch.autograd.grad(actual.sum(), (a, b)),
                                    torch.autograd.grad(expected.sum(), (a, b))):
        torch.testing.assert_close(gradient, reference)


@pytest.mark.parametrize('metric', METRICS)
def test_covariance_rotated_commuting_reference(device, metric):
    q = torch.tensor([[.6, -.8], [.8, .6]], device=device, dtype=torch.float64)
    a = torch.tensor([[1., 4.], [9., 16.]], device=device, dtype=q.dtype)
    b = torch.tensor([[4., 25.], [1., 1.]], device=device, dtype=q.dtype)
    actual = _function(metric)(q @ a.diag_embed() @ q.T, q @ b.diag_embed() @ q.T)
    torch.testing.assert_close(actual, _diagonal_reference(a, b, metric, False))


def _full_covariances(device):
    a = torch.tensor([[[2., .6], [.6, 1.]], [[1., -.2], [-.2, 3.]]],
                     device=device, dtype=torch.float64)
    b = torch.tensor([[[3., -.5], [-.5, 2.]]], device=device, dtype=a.dtype)
    return a, b


@pytest.mark.parametrize('metric', METRICS)
def test_covariance_noncommuting_reference_and_symmetry(device, metric):
    a, b = _full_covariances(device)
    function = _function(metric)
    actual = function(a, b, squared=True)
    if metric == 'stein':
        expected = (torch.linalg.slogdet((a + b) / 2).logabsdet
                    - (torch.linalg.slogdet(a).logabsdet + torch.linalg.slogdet(b).logabsdet) / 2)
    elif metric == 'affine_invariant':
        # A^-1 B has the same positive spectrum as the whitened matrix.
        values = torch.linalg.eigvals(torch.linalg.solve(a, b)).real
        expected = values.log().square().sum(-1)
    else:
        def matrix_log(value):
            values, vectors = torch.linalg.eigh(value)
            return (vectors * values.log().unsqueeze(-2)) @ vectors.transpose(-1, -2)
        expected = (matrix_log(a) - matrix_log(b)).square().sum((-1, -2))
    torch.testing.assert_close(actual[:, 0], expected)
    torch.testing.assert_close(actual, function(b, a, squared=True).T)


@pytest.mark.parametrize('metric', METRICS)
@pytest.mark.parametrize('squared', [False, True])
@pytest.mark.parametrize('repeated', [None, 1.125, 2.])
def test_covariance_gradcheck_repeated_spectra(device, metric, squared, repeated):
    a = torch.eye(2, device=device, dtype=torch.float64)[None].requires_grad_()
    b = (repeated * a.detach() if repeated else
         torch.tensor([[[2., .3], [.3, 3.]]], device=device, dtype=a.dtype)).requires_grad_()

    def call(x, y):
        return _function(metric)((x + x.transpose(-1, -2)) / 2,
                                 (y + y.transpose(-1, -2)) / 2, squared=squared)

    assert torch.autograd.gradcheck(call, (a, b), atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('metric', METRICS)
@pytest.mark.parametrize('squared', [False, True])
def test_covariance_coincidence_and_zero_upstream(device, metric, squared):
    # Separate leaves check each partial gradient, not just cancellation when
    # the same tensor occurs on both sides. Include repeated/full spectra.
    a = torch.tensor([[[1., 0.], [0., 1.]], [[2., .4], [.4, 3.]]],
                     device=device, dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    value = _function(metric, 'paired_')(a, b, squared=squared)
    torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)
    for gradient in torch.autograd.grad(value.sum(), (a, b)):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)
    full = _function(metric)(a, b * 2, squared=squared)
    for gradient in torch.autograd.grad(full, (a, b), grad_outputs=torch.zeros_like(full)):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)
    self_map = _function(metric)(a, squared=squared)
    torch.testing.assert_close(self_map.diagonal(), torch.zeros_like(value), atol=0, rtol=0)


@pytest.mark.parametrize('metric', METRICS)
def test_covariance_invariances(device, metric):
    a, b = _full_covariances(device)
    function = _function(metric)
    expected = function(a, b)
    p = torch.tensor([[.6, -.8], [.8, .6]], device=device, dtype=a.dtype)
    torch.testing.assert_close(function(p @ a @ p.T, p @ b @ p.T), expected)
    torch.testing.assert_close(function(a * 13, b * 13), expected)
    torch.testing.assert_close(function(torch.linalg.inv(a), torch.linalg.inv(b)), expected)
    if metric != 'log_euclidean':
        p = torch.tensor([[2., .7], [-.3, 1.]], device=device, dtype=a.dtype)
        torch.testing.assert_close(function(p @ a @ p.T, p @ b @ p.T), expected)


@pytest.mark.parametrize('metric', METRICS)
@pytest.mark.parametrize('squared', [False, True])
def test_covariance_broadcast_noncontiguous_modes_and_gradients(device, metric, squared):
    torch.manual_seed(814)
    a = (torch.rand(2, 1, 3, 2, device=device, dtype=torch.float64) + 1).diag_embed()
    a = a.transpose(-1, -2).requires_grad_()
    b = (torch.rand(1, 4, 3, 2, device=device, dtype=a.dtype) + 2).diag_embed().requires_grad_()
    full = _function(metric)(a, b, squared=squared)
    assert full.shape == (2, 4, 3, 3)
    paired = _function(metric, 'paired_')(a, b, squared=squared)
    torch.testing.assert_close(paired, full.diagonal(dim1=-2, dim2=-1))
    for gradient, reference in zip(torch.autograd.grad(paired.sum(), (a, b)),
                                    torch.autograd.grad(full.diagonal(dim1=-2, dim2=-1).sum(), (a, b))):
        torch.testing.assert_close(gradient, reference)
    self_map = _function(metric)(a, squared=squared)
    condensed = _function(metric, 'condensed_')(a, squared=squared)
    i, j = torch.triu_indices(3, 3, 1, device=device)
    torch.testing.assert_close(condensed, self_map[..., i, j])
    torch.testing.assert_close(torch.autograd.grad(condensed.sum(), a)[0],
                               torch.autograd.grad(self_map[..., i, j].sum(), a)[0])


@pytest.mark.parametrize('metric', METRICS)
def test_covariance_tile_boundaries_and_empty_outputs(device, metric):
    a = torch.linspace(1, 3, 18, device=device, dtype=torch.float64)[:, None, None].requires_grad_()
    b = torch.linspace(4, 6, 17, device=device, dtype=a.dtype)[:, None, None].requires_grad_()
    full = _function(metric)(a, b)
    torch.testing.assert_close(full, _diagonal_reference(a[..., 0], b[..., 0], metric, False))
    paired = _function(metric, 'paired_')(a[:17], b)
    torch.testing.assert_close(paired, full[:17].diagonal())
    condensed = _function(metric, 'condensed_')(a)
    i, j = torch.triu_indices(18, 18, 1, device=device)
    torch.testing.assert_close(condensed, _function(metric)(a)[i, j])
    for n in (0, 1):
        empty = _function(metric, 'condensed_')(a[:n])
        assert empty.shape == (0,)
        torch.testing.assert_close(torch.autograd.grad(empty.sum(), a)[0], torch.zeros_like(a))
    assert _function(metric)(a[:0], b).shape == (0, 17)
    assert _function(metric, 'paired_')(a[:0], b[:0]).shape == (0,)
    assert _function(metric)(a.expand(0, 18, 1, 1)).shape == (0, 18, 18)


@pytest.mark.parametrize('metric', METRICS)
def test_covariance_near_identity_avoids_cancellation(device, metric):
    a = torch.eye(2, device=device)[None]
    b = a.clone()
    b[0, 0, 1] = b[0, 1, 0] = 1e-5
    actual = _function(metric)(a, b, squared=True)
    # Evaluate the reference in double from the actual float input values.
    eigenvalues = torch.linalg.eigvalsh(b.double())
    expected = _diagonal_reference(torch.ones_like(eigenvalues), eigenvalues, metric, True)
    assert actual.item() > 0
    # Log-Euclidean computes input eigenvalues in float32; whitening metrics
    # diagonalize B-A and preserve considerably smaller relative differences.
    torch.testing.assert_close(actual.double(), expected,
                               rtol=.02 if metric == 'log_euclidean' else 2e-5, atol=0)


@pytest.mark.parametrize('metric', METRICS)
def test_covariance_domain_validation_and_no_jitter(device, metric):
    identity = torch.eye(2, device=device, dtype=torch.float64)[None]
    function = _function(metric)
    for bad, message in ((identity * float('nan'), 'finite'),
                         (identity * float('inf'), 'finite'),
                         (torch.tensor([[[1., .1], [0., 1.]]], device=device, dtype=identity.dtype), 'symmetric'),
                         (torch.diag_embed(torch.tensor([[1., 0.]], device=device, dtype=identity.dtype)), 'strictly positive'),
                         (torch.diag_embed(torch.tensor([[1., -1e-15]], device=device, dtype=identity.dtype)), 'strictly positive')):
        with pytest.raises(ValueError, match=message):
            function(bad, identity)
        # The nonempty operand is still validated when output has no pairs.
        with pytest.raises(ValueError, match=message):
            function(identity[:0], bad)
    # No positive eigenvalue floor: diagonal SPD inputs spanning many orders
    # of magnitude have finite distances when the arithmetic is representable.
    tiny = identity * 1e-20
    assert torch.isfinite(function(tiny, identity)).all()
    roundoff = identity.clone()
    roundoff[0, 0, 1] = 8 * torch.finfo(identity.dtype).eps
    symmetrized = (roundoff + roundoff.transpose(-1, -2)) / 2
    torch.testing.assert_close(function(roundoff, identity * 2), function(symmetrized, identity * 2))


@pytest.mark.parametrize('metric', METRICS)
def test_covariance_shape_dtype_validation_and_autocast(device, metric):
    a = torch.eye(2, device=device).expand(2, 2, 2).clone()
    function = _function(metric)
    for bad in (a[0], a[..., :1], torch.empty(2, 0, 0, device=device)):
        with pytest.raises(ValueError, match='shape|non-empty'):
            function(bad)
    with pytest.raises(ValueError, match='dimensions must match'):
        function(a, torch.ones(1, 1, 1, device=device))
    with pytest.raises(ValueError, match='equal sample counts'):
        _function(metric, 'paired_')(a, a[:1])
    with pytest.raises(ValueError, match='broadcastable'):
        function(a.expand(2, 2, 2, 2), a.expand(3, 2, 2, 2))
    with pytest.raises(ValueError, match='same dtype'):
        function(a, a.double())
    with pytest.raises(TypeError, match='boolean'):
        function(a, squared=1)
    with pytest.raises(TypeError, match='tensors'):
        function([1])
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    low = a.to(dtype).requires_grad_()
    with pytest.raises(TypeError, match='float32 or float64'):
        function(low)
    with torch.autocast(device, dtype=dtype):
        value = function(low, low * 2)
    assert value.dtype == torch.float32
    torch.testing.assert_close(value, function(low.float(), low.float() * 2))
    assert torch.isfinite(torch.autograd.grad(value.sum(), low)[0]).all()
