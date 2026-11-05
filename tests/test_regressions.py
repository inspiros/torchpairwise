import pytest
import torch
import torchpairwise as tp

from test_metrics import DISTANCES, KERNELS, inputs, parameters


@pytest.mark.parametrize('kernel', ['polynomial', 'sigmoid', 'rbf', 'laplacian'])
def test_batched_default_gamma_uses_features(kernel, device):
    x, y = inputs('euclidean', device)
    fn = getattr(tp, kernel + '_kernel')
    torch.testing.assert_close(fn(x[None], y[None])[0], fn(x, y))


@pytest.mark.parametrize('metric', ['seuclidean', 'mahalanobis'])
def test_inferred_statistics_are_per_batch(metric, device):
    g = torch.Generator().manual_seed(12)
    x = torch.randn(2, 9, 3, generator=g, dtype=torch.float64).to(device)
    x[1] *= 4
    fn = getattr(tp, metric + '_distances')
    expected = torch.stack([fn(batch) for batch in x])
    torch.testing.assert_close(fn(x), expected, atol=1e-10, rtol=1e-8)


def test_inferred_covariance_matches_detached_torch_cov(device):
    x = torch.randn(2, 9, 3, device=device, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        inverse = torch.stack([torch.cov(batch.T).inverse().T for batch in x])
    actual = tp.mahalanobis_distances(x)
    expected = tp.mahalanobis_distances(x, VI=inverse)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                               torch.autograd.grad(expected.sum(), x)[0])


@pytest.mark.parametrize('metric', DISTANCES)
@pytest.mark.parametrize('empty_x', [True, False])
def test_empty_rows(metric, empty_x, device):
    x, y = inputs(metric, device)
    x, y = (x[:0], y) if empty_x else (x, y[:0])
    kw = parameters(metric, x)
    out = getattr(tp, metric + '_distances')(x, y, **kw)
    assert out.shape == (len(x), len(y))
    if x.is_floating_point() and metric not in {'hamming'}:
        x.requires_grad_()
        y.requires_grad_()
        out = getattr(tp, metric + '_distances')(x, y, **kw)
        gx, gy = torch.autograd.grad(out.sum(), (x, y))
        torch.testing.assert_close(gx, torch.zeros_like(x))
        torch.testing.assert_close(gy, torch.zeros_like(y))


def test_canberra_zero_coordinates_have_finite_zero_gradients(device):
    x = torch.tensor([[0., 1.]], device=device, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[0., 2.]], device=device, dtype=torch.float64, requires_grad=True)
    gx, gy = torch.autograd.grad(tp.canberra_distances(x, y).sum(), (x, y))
    torch.testing.assert_close(gx, torch.tensor([[0., -4 / 9]], device=device, dtype=x.dtype))
    torch.testing.assert_close(gy, torch.tensor([[0., 2 / 9]], device=device, dtype=x.dtype))


@pytest.mark.parametrize('metric', ['euclidean', 'canberra', 'sqeuclidean', 'mahalanobis'])
def test_invalid_second_rank_rejected(metric, device):
    x, y = inputs(metric, device)
    with pytest.raises((ValueError, RuntimeError), match='dim|rank|2D|2-D|batch'):
        getattr(tp, metric + '_distances')(x, y[0], **parameters(metric, x))


@pytest.mark.parametrize('metric', ['wminkowski', 'mahalanobis', 'jensenshannon', 'haversine', 'snr'])
def test_identical_pairs_have_zero_subgradient(metric, device):
    x = torch.tensor([[.3, .7]], device=device, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    kw = parameters(metric, x)
    out = getattr(tp, metric + '_distances')(x, y, **kw)
    torch.testing.assert_close(out, torch.zeros_like(out))
    for grad in torch.autograd.grad(out.sum(), (x, y)):
        torch.testing.assert_close(grad, torch.zeros_like(grad))


def test_weighted_infinity_uses_weight_support(device):
    x = torch.tensor([[1., 2., 3.]], dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tensor([[3., 8., 6.]], dtype=x.dtype, device=device, requires_grad=True)
    w = torch.tensor([5., 0., .2], dtype=x.dtype, device=device, requires_grad=True)
    out = tp.wminkowski_distances(x, y, p=float('inf'), w=w)
    torch.testing.assert_close(out, torch.tensor([[3.]], dtype=x.dtype, device=device))
    gx, gy, gw = torch.autograd.grad(out.sum(), (x, y, w))
    torch.testing.assert_close(gx, torch.tensor([[0., 0., -1.]], dtype=x.dtype, device=device))
    torch.testing.assert_close(gy, -gx)
    torch.testing.assert_close(gw, torch.zeros_like(w))


def test_zero_vector_cosine_policy(device):
    x = torch.zeros(2, 3, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.ones(4, 3, dtype=x.dtype, device=device, requires_grad=True)
    out = tp.cosine_similarity(x, y)
    torch.testing.assert_close(out, torch.zeros_like(out))
    for grad in torch.autograd.grad(out.sum(), (x, y)):
        torch.testing.assert_close(grad, torch.zeros_like(grad))
    torch.testing.assert_close(tp.cosine_distances(x).diagonal(), torch.zeros(2, dtype=x.dtype, device=device))


def test_constant_correlation_rows_are_undefined(device):
    x = torch.ones(2, 3, dtype=torch.float64, device=device)
    assert tp.correlation_distances(x).isnan().all()


def test_haversine_antipodal_is_finite(device):
    x = torch.tensor([[.2, .3]], device=device, dtype=torch.float64)
    y = torch.tensor([[-.2, .3 + torch.pi]], device=device, dtype=x.dtype)
    torch.testing.assert_close(tp.haversine_distances(x, y), torch.full((1, 1), torch.pi, device=device, dtype=x.dtype))


@pytest.mark.parametrize('name,kwargs', [('jensenshannon', {'base': 1}),
                                        ('wminkowski', {'p': -1}),
                                        ('haversine', {})])
def test_invalid_domains(name, kwargs, device):
    x = torch.ones(2, 3, device=device)
    with pytest.raises(ValueError):
        getattr(tp, name + '_distances')(x, x, **kwargs)


def test_native_mahalanobis_rejects_nonsquare_vi(device):
    x = torch.ones(2, 3, dtype=torch.float64, device=device)
    y = torch.ones(4, 3, dtype=x.dtype, device=device)
    invalid = torch.ones(3, 2, dtype=x.dtype, device=device)
    with pytest.raises(RuntimeError, match='VI.*square'):
        torch.ops.torchpairwise.mahalanobis_distances(x, y, VI=invalid)


def test_cosine_same_tensor_self_policy(device):
    x = torch.zeros(3, 4, dtype=torch.float64, device=device, requires_grad=True)
    expected = torch.zeros(3, dtype=x.dtype, device=device)
    torch.testing.assert_close(tp.cosine_distances(x, x).diagonal(), expected)
    torch.testing.assert_close(tp.cdist(x, x, metric='cosine').diagonal(), expected)
    paired = tp.paired_distances(x, x, metric='cosine')
    torch.testing.assert_close(paired, expected)
    gradient, = torch.autograd.grad(paired.sum(), (x,))
    torch.testing.assert_close(gradient, torch.zeros_like(x))


@pytest.mark.parametrize('n,m', [(5, 3), (0, 3)])
def test_jensen_shannon_primitive_backward_work_counts(n, m, device):
    # Unequal counts span multiple CUDA blocks; an empty side must still give
    # connected zero gradients for the nonempty operand.
    x = (torch.rand(2, 257, n, device=device, dtype=torch.float64) + .3).transpose(-1, -2).requires_grad_()
    y = (torch.rand(2, 257, m, device=device, dtype=torch.float64) + .5).transpose(-1, -2).requires_grad_()
    result = torch.ops.torchpairwise._sqjensenshannon(x, y, 2.)
    p, q = x.unsqueeze(-2), y.unsqueeze(-3)
    midpoint = (p + q) * .5
    expected = ((p * (p / midpoint).log() + q * (q / midpoint).log()).sum(-1)
                / (2 * torch.log(x.new_tensor(2.))))
    torch.testing.assert_close(result, expected)
    cotangent = torch.randn_like(result)
    for actual, reference in zip(torch.autograd.grad(result, (x, y), cotangent),
                                 torch.autograd.grad(expected, (x, y), cotangent)):
        torch.testing.assert_close(actual, reference)
