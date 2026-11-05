"""Named-kernel empirical MMD shares the native vector kernel catalog."""
import pytest
import torch
import torchpairwise as tp


KERNELS = [
    ('polynomial', dict(degree=3, gamma=.4, coef0=.7)),
    ('rbf_kernel', dict(gamma=.6)),
    ('laplacian_kernel', dict(gamma=.8)),
    ('linear_kernel', {}),
    ('cosine_similarity', {}),
    ('rational_quadratic', dict(length_scale=.8, alpha=.7)),
    ('matern', dict(length_scale=.9, nu=2.5)),
    ('histogram_intersection', {}),
    ('chi2', dict(gamma=.3)),
    ('additive_chi2', {}),
    ('bhattacharyya', {}),
]


def dense_statistic(a, b, kernel, options):
    def gram(x, y):
        from torchpairwise.ops._registry import KERNELS
        short = next((name for name, spec in KERNELS.items() if spec.op == kernel), kernel)
        return tp.pairwise_kernels(x, y, kernel=short, **options)
    return (gram(a, a).mean() + gram(b, b).mean() - 2 * gram(a, b).mean()).clamp_min(0)


@pytest.mark.parametrize('kernel,options', KERNELS)
def test_mmd_named_kernels_modes_lengths_and_gradients(kernel, options, device):
    torch.manual_seed(207)
    x = (torch.rand(3, 4, 2, dtype=torch.float64, device=device) + .2).requires_grad_()
    y = (torch.rand(3, 5, 2, dtype=x.dtype, device=device) + .6).requires_grad_()
    xl = torch.tensor([2, 4, 3], device=device)
    yl = torch.tensor([5, 2, 4], device=device)
    # Prefix masking must protect all kernel domains from NaN padding.
    with torch.no_grad():
        x[0, 2:] = float('nan')
        y[1, 2:] = float('nan')
    expected = torch.stack([torch.stack([
        dense_statistic(a[:al], b[:bl], kernel, options)
        for b, bl in zip(y, yl)]) for a, al in zip(x, xl)])
    kwargs = dict(kernel=kernel, squared=True, x1_lengths=xl, x2_lengths=yl, **options)
    actual = tp.mmd_distances(x, y, **kwargs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(tp.compare(x, y, 'mmd', **kwargs), expected)
    paired = tp.paired_distances(x, y, 'mmd', **kwargs)
    torch.testing.assert_close(paired, expected.diagonal())
    weights = torch.randn_like(expected)
    for a, e in zip(torch.autograd.grad((actual * weights).sum(), (x, y)),
                    torch.autograd.grad((expected * weights).sum(), (x, y))):
        torch.testing.assert_close(a, e)
        assert a.isfinite().all()
    self_kwargs = dict(kernel=kernel, squared=True, x1_lengths=xl, **options)
    full = tp.mmd_distances(x, **self_kwargs)
    i, j = torch.triu_indices(3, 3, 1, device=device)
    condensed = tp.pdist(x, 'mmd', **self_kwargs)
    torch.testing.assert_close(condensed, full[i, j])
    torch.testing.assert_close(torch.autograd.grad(condensed.sum(), x)[0],
                               torch.autograd.grad(full[i, j].sum(), x)[0])


def test_polynomial_mmd_tiled_native_reference_and_reduction(device):
    torch.manual_seed(367)
    x = torch.rand(1, 129, 3, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(1, 131, 3, device=device, dtype=x.dtype, requires_grad=True)
    def mean(a, b):
        return (.2 * (a @ b.T) + .7).pow(3).mean()
    expected = mean(x[0], x[0]) + mean(y[0], y[0]) - 2 * mean(x[0], y[0])
    options = dict(kernel='polynomial', degree=3, gamma=.2, coef0=.7, squared=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
        actual = tp.mmd_distances(x, y, **options)
        gradients = torch.autograd.grad(actual.sum(), (x, y))
    torch.testing.assert_close(actual[0, 0], expected)
    for a, e in zip(gradients, torch.autograd.grad(expected, (x, y))):
        torch.testing.assert_close(a, e)
    calls = [e for e in profile.events() if e.name == 'torchpairwise::_legacy_compare']
    assert calls and all(e.input_shapes[0][-2] <= 128 and e.input_shapes[1][-2] <= 128 for e in calls)
    # Registry reductions must forward the full MMD kernel option set.
    reduced = tp.lazy(x, y, **options, metric='mmd').reduce('sum', dim=-1)
    reference = tp.mmd_distances(x, y, **options).sum(-1)
    torch.testing.assert_close(reduced, reference)
    for a, e in zip(torch.autograd.grad(reduced.sum(), (x, y)),
                    torch.autograd.grad(reference.sum(), (x, y))):
        torch.testing.assert_close(a, e)
    for n in (0, 1):
        empty = tp.pdist(x[:n], 'mmd', **options)
        assert empty.numel() == 0
        torch.testing.assert_close(torch.autograd.grad(empty.sum(), x)[0], torch.zeros_like(x))


def test_mmd_kernel_validation_and_laplacian_compatibility(device):
    x = torch.tensor([[[0., 0.]], [[3., 4.]]], device=device)
    expected = 2 * (1 - torch.exp(x.new_tensor(-5.)))
    torch.testing.assert_close(tp.mmd_distances(x, kernel='laplacian', squared=True)[0, 1], expected)
    manhattan = tp.mmd_distances(x, kernel='laplacian_kernel', gamma=1., squared=True)
    torch.testing.assert_close(manhattan[0, 1], 2 * (1 - torch.exp(x.new_tensor(-7.))))
    for options, message in [
        (dict(kernel='sigmoid'), 'positive semidefinite'),
        (dict(kernel='polynomial', gamma=-1.), 'gamma'),
        (dict(kernel='polynomial', coef0=-1.), 'coef0'),
        (dict(kernel='polynomial', degree=-1), 'degree'),
        (dict(kernel='linear', degree=3), 'Unused parameter'),
        (dict(kernel='rbf', alpha=1.), 'Unused parameter'),
    ]:
        with pytest.raises((RuntimeError, ValueError), match=message):
            tp.mmd_distances(x, **options)
    # Direct native calls enforce the same contract without Python validation.
    with pytest.raises(RuntimeError, match='positive semidefinite'):
        torch.ops.torchpairwise.mmd_comparisons(x, kernel='sigmoid')


@pytest.mark.compiler
def test_mmd_operator_contract(device):
    x = torch.rand(2, 3, 2, device=device, requires_grad=True)
    y = torch.rand(2, 4, 2, device=device, requires_grad=True)
    empty = torch.empty(0, device=device, dtype=torch.int64)
    args = (x, y, empty, empty, 'polynomial', 0, .4, 3, 1., 1., 1., 1.5, True)
    result = torch.library.opcheck(torch.ops.torchpairwise._mmd_compare.default, args)
    assert all(v == 'SUCCESS' for v in result.values())
