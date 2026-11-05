"""Dispatcher contract checks against actual native layouts and derivatives."""
import pytest
import torch
import torchpairwise as tp


@pytest.mark.parametrize('name', ['_canberra', '_braycurtis', '_haversine', '_ppminkowski',
                                  '_wminkowski', '_sqmahalanobis', '_sqjensenshannon',
                                  '_snr', '_additive_chi2_kernel', '_directed_hausdorff',
                                  'pwand_sum', 'pwne_mean', 'prf_div', '_pointset_nn', '_histogram_intersection', '_nan_sqeuclidean'])
@pytest.mark.compiler
def test_native_operator_contract(name, device):
    namespace = torch.ops.torchpairwise
    torch.compiler.reset()
    d = 2 if name == '_haversine' else 3
    x, y = [(torch.rand(2, d, device=device) + .3).requires_grad_() for _ in range(2)]
    if name == '_directed_hausdorff':
        x, y = [value.unsqueeze(0).detach().requires_grad_() for value in (x, y)]
    if name == '_wminkowski':
        args = (x, y, torch.ones(d, device=device, requires_grad=True), 1.7)
    elif name == '_pointset_nn':
        x, y = [value.unsqueeze(0).unsqueeze(0) for value in (x, y)]
        lengths = torch.full((1, 1), 2, device=device, dtype=torch.int64)
        args = (x, y, lengths, lengths)
    elif name == '_sqmahalanobis':
        args = (x, y, torch.eye(d, device=device, requires_grad=True))
    elif name == '_sqjensenshannon':
        args = (x / x.sum(-1, keepdim=True), y / y.sum(-1, keepdim=True), None)
    elif name == 'pwand_sum':
        args = (x.detach() > .7, y.detach() > .7)
    elif name == 'pwne_mean':
        args = (x.detach(), y.detach())
    else:
        args = (x, y)
    result = torch.library.opcheck(getattr(namespace, name).default, args)
    assert all(value == 'SUCCESS' for value in result.values())


def test_public_fake_and_meta_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode
    for context in (FakeTensorMode(), torch.device('meta')):
        with context:
            x, y = torch.empty(2, 1, 3, 5), torch.empty(1, 4, 6, 5)
            result = tp.canberra_distances(x, y)
            assert result.shape == (2, 4, 3, 6)
            assert result.dtype == torch.float32
            assert tp.jensenshannon_distances(x, y).shape == (2, 4, 3, 6)
            assert tp.chi2_kernel(x, y).shape == (2, 4, 3, 6)
            assert tp.rational_quadratic_kernel(x, y).shape == (2, 4, 3, 6)
            assert tp.matern_kernel(x, y).shape == (2, 4, 3, 6)
            assert tp.hellinger_distances(x, y).shape == (2, 4, 3, 6)
            assert tp.histogram_intersection_kernel(x, y).shape == (2, 4, 3, 6)
            assert tp.nan_euclidean_distances(x, y).shape == (2, 4, 3, 6)
            assert tp.wasserstein_1d_distances(x, y, bin_positions=torch.arange(5.)).shape == (2, 4, 3, 6)
            assert tp.seuclidean_distances(x, y, V=torch.ones(5)).shape == (2, 4, 3, 6)


def test_legacy_backward_dispatcher_alias(device):
    x, y = [torch.rand(2, 3, device=device) for _ in range(2)]
    grad = torch.ones(2, 2, device=device)
    namespace = torch.ops.torchpairwise
    expected = namespace._canberra_backward(grad, x, y)
    actual = namespace.__canberra_backward(grad, x, y)
    for value, ref in zip(actual, expected):
        torch.testing.assert_close(value, ref)


@pytest.mark.parametrize('left', [False, True])
@pytest.mark.parametrize('scalar', [False, True])
def test_division_backward_reads_contiguous_promoted_operands(left, scalar, device):
    # Sum gradients have zero strides; mixed dtypes additionally require reads
    # from promoted buffers, rather than reinterpreting the original storage.
    x = torch.rand(4, 3, device=device, dtype=torch.float32)[::2].requires_grad_()
    y = 2. if scalar else (torch.rand(4, 3, device=device, dtype=torch.float64)[::2] + .3).requires_grad_()
    fn = torch.ops.torchpairwise.prf_ldiv if left else torch.ops.torchpairwise.prf_div
    args = (x,) if scalar else (x, y)
    expected = y / x if left else x / y
    actual = fn(x, y)
    torch.testing.assert_close(actual, expected)
    for grad, ref in zip(torch.autograd.grad(actual.sum(), args), torch.autograd.grad(expected.sum(), args)):
        torch.testing.assert_close(grad, ref)
