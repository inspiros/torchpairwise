"""One selector preserves existing family input contracts and native execution."""
import pytest
import torch
import torchpairwise as tp
from torchpairwise.ops._comparison_dispatch import compare, list_comparisons


def test_comparison_catalog_names_and_contracts():
    catalog = list_comparisons()
    names = [entry.name for entry in catalog]
    assert names == sorted(set(names))
    assert {entry.family for entry in catalog} == {
        'vector', 'point_cloud', 'image', 'audio', 'sequence', 'tokens',
        'gaussian', 'covariance', 'mixture'}
    for family in {entry.family for entry in catalog}:
        assert all(entry.family == family for entry in list_comparisons(family=family))
    assert next(entry for entry in catalog if entry.name == 'ssim_scores').aliases == ('ssim', 'ssim_index')
    assert next(entry for entry in catalog if entry.name == 'signal_snr_scores').modes == ('all_pairs', 'paired', 'single')
    assert all('single' in entry.modes for entry in catalog)
    assert len(list_comparisons(role='distance')) == 68
    assert len(list_comparisons(role='kernel')) == 13
    assert all(not entry.differentiable for entry in list_comparisons(family='tokens'))
    assert 'squared' in next(entry for entry in catalog if entry.name == 'gaussian_wasserstein_distances').parameters
    with pytest.raises(ValueError, match='family'):
        list_comparisons(family='unknown')


def _example(family, device):
    options = dict(device=device, dtype=torch.float64)
    if family == 'image':
        a, b = torch.rand(3, 1, 5, 5, **options), torch.rand(3, 1, 5, 5, **options)
        return a, b, 'ssim_scores', dict(data_range=1., window_size=3)
    if family == 'audio':
        return torch.rand(3, 1, 24, **options), torch.rand(3, 1, 24, **options), 'log_spectral_distances', dict(n_fft=8)
    if family == 'sequence':
        return torch.rand(3, 4, 2, **options), torch.rand(3, 4, 2, **options), 'dtw_distances', dict(squared=True)
    if family == 'tokens':
        a = torch.tensor([[1, 2, 3], [2, 3, 4], [1, 1, 1]], device=device)
        return a, a.roll(1, 0), 'levenshtein_distances', {}
    if family == 'covariance':
        a = torch.eye(2, **options).repeat(3, 1, 1)
        return a, a * 2, 'stein_distances', dict(squared=True)
    if family in ('gaussian', 'mixture'):
        mean = torch.rand(3, 2, **options)
        covariance = torch.eye(2, **options).repeat(3, 1, 1)
        if family == 'gaussian':
            return (mean, covariance), (mean + 1, covariance * 2), 'gaussian_wasserstein_distances', dict(squared=True)
        weights = torch.ones(3, 2, **options)
        a = mean[:, None].repeat(1, 2, 1), covariance[:, None].repeat(1, 2, 1, 1), weights
        b = a[0] + 1, a[1] * 2, weights
        return a, b, 'gaussian_mixture_l2_distances', dict(squared=True)
    if family == 'point_cloud':
        return torch.rand(3, 4, 2, **options), torch.rand(3, 5, 2, **options), 'chamfer_distances', dict(squared=True)
    return torch.rand(3, 2, **options), torch.rand(3, 2, **options), 'euclidean_distances', {}


@pytest.mark.parametrize('family', ['vector', 'point_cloud', 'image', 'audio', 'sequence', 'tokens',
                                    'gaussian', 'covariance', 'mixture'])
@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed'])
def test_comparison_dispatch_matches_named_family_entrypoints(family, mode, device):
    a, b, name, options = _example(family, device)
    prefix = '' if mode == 'all_pairs' else mode + '_'
    if family in ('vector', 'point_cloud') and mode != 'all_pairs':
        function = tp.paired_distances if mode == 'paired' else tp.condensed_distances
        args = (a, b) if mode == 'paired' else (a,)
        expected = function(*args, metric='chamfer' if family == 'point_cloud' else 'euclidean', **options)
    else:
        function = getattr(tp, prefix + name)
        args = (*a, *b) if family in ('gaussian', 'mixture') and mode != 'condensed' else a if family in ('gaussian', 'mixture') else (a, b) if mode != 'condensed' else (a,)
        expected = function(*args, **options)
    actual = compare(a, None if mode == 'condensed' else b, metric=name, mode=mode, **options)
    torch.testing.assert_close(actual, expected)


def test_comparison_dispatch_aliases_lengths_and_errors(device):
    x = torch.tensor([[1, 2, 99], [2, 3, 99], [1, 1, 99]], device=device)
    lengths = torch.full((3,), 2, dtype=torch.int64, device=device)
    expected = tp.condensed_levenshtein_distances(x, x_lengths=lengths)
    torch.testing.assert_close(compare(x, metric='levenshtein', mode='condensed', x1_lengths=lengths), expected)
    y = torch.rand(3, 2, device=device)
    torch.testing.assert_close(compare(y, metric='l2'), tp.euclidean_distances(y))
    torch.testing.assert_close(compare(y, metric='rbf'), tp.rbf_kernel(y))
    # Complete the documented pairwise/paired/condensed distance vocabulary.
    # cdist/pdist and named mode helpers share the native role-filtered catalog.
    torch.testing.assert_close(tp.pairwise_distances(y), compare(y))
    torch.testing.assert_close(tp.pairwise_distances(y, y, metric='canberra'),
                               tp.cdist(y, y, metric='canberra'))
    assert tp.pairwise_distances(y[:0], y).shape == (0, 3)
    with pytest.raises((TypeError, ValueError), match='gamma'):
        tp.pairwise_distances(y, metric='euclidean', gamma=.2)
    with pytest.raises(ValueError, match='Unknown comparison'):
        compare(y, metric='missing')
    with pytest.raises(ValueError, match='x2'):
        compare(y, metric='ssim', mode='paired')
    with pytest.raises(ValueError, match='x1 only'):
        compare(y, y, mode='condensed')
    with pytest.raises(ValueError, match='does not support'):
        compare(y, metric='signal_snr', mode='condensed')
    with pytest.raises(TypeError, match='tuple of tensors'):
        compare(y, metric='gaussian_wasserstein')
    with pytest.raises(TypeError, match='x2_lengths'):
        compare(x, metric='levenshtein', mode='condensed', x2_lengths=lengths)
    with pytest.raises(TypeError, match='both'):
        compare(x, metric='levenshtein', mode='condensed', x1_lengths=lengths, x_lengths=lengths)
    image = torch.rand(2, 1, 5, 5, device=device)
    with pytest.raises(TypeError, match='squared'):
        compare(image, metric='psnr', data_range=1., squared=True)


@pytest.mark.compiler
@pytest.mark.parametrize('family', ['image', 'audio', 'sequence', 'gaussian', 'covariance', 'mixture'])
def test_comparison_dispatch_compile_export_and_gradients(family, device, tmp_path):
    a, b, name, options = _example(family, device)
    def require(value):
        return tuple(t.requires_grad_() for t in value) if isinstance(value, tuple) else value.requires_grad_()
    a, b = require(a), require(b)

    class Comparison(torch.nn.Module):
        def forward(self, x, y):
            return compare(x, y, metric=name, mode='paired', **options)

    module = Comparison()
    exported = torch.export.export(module, (a, b))
    path = tmp_path / f'comparison_dispatch_{family}.pt2'
    torch.export.save(exported, path)
    restored = torch.export.load(path).module()
    expected, actual = module(a, b), restored(a, b)
    torch.testing.assert_close(actual, expected)
    operands = a if isinstance(a, tuple) else (a,)
    for value, reference in zip(torch.autograd.grad(actual.sum(), operands),
                                torch.autograd.grad(expected.sum(), operands)):
        torch.testing.assert_close(value, reference)
    torch.compiler.reset()
    compiled = torch.compile(module, backend='aot_eager', fullgraph=True)
    actual, expected = compiled(a, b), module(a, b)
    torch.testing.assert_close(actual, expected)
    for value, reference in zip(torch.autograd.grad(actual.sum(), operands),
                                torch.autograd.grad(expected.sum(), operands)):
        torch.testing.assert_close(value, reference)
    torch.compiler.reset()


def test_single_comparison_native_selectors_and_batches(device):
    # One distinct sample layout per family; avoid multiplying aliases into grids.
    for family in ('vector', 'point_cloud', 'image', 'audio', 'sequence', 'tokens',
                   'covariance', 'gaussian', 'mixture'):
        a, b, name, options = _example(family, device)
        info = next(entry for entry in list_comparisons() if entry.name == name)
        expected = compare(a, b, metric=name, mode='paired', **options)
        # Collection sample count becomes a batch size in single mode.
        actual = tp.single_comparison(a, b, metric=name, **options)
        torch.testing.assert_close(actual, expected)
        aa = a if isinstance(a, tuple) else (a,)
        bb = b if isinstance(b, tuple) else (b,)
        single_a = tuple(t[0] for t in aa)
        single_b = tuple(t[0] for t in bb)
        unbatch_a = single_a if isinstance(a, tuple) else single_a[0]
        unbatch_b = single_b if isinstance(b, tuple) else single_b[0]
        scalar = tp.single_comparison(unbatch_a, unbatch_b, metric=name, **options)
        assert scalar.ndim == 0
        torch.testing.assert_close(scalar, expected[0])
        # Raw native single mode consumes the same logical sample layouts.
        native = torch.ops.torchpairwise.compare(single_a, single_b, name, 'single', **options)
        torch.testing.assert_close(native, scalar)
        broadcast = tp.single_comparison(a, unbatch_b, metric=name, **options)
        reference = compare(a, tuple(t[:1].expand_as(t) for t in bb) if isinstance(b, tuple)
                            else b[:1].expand_as(b), metric=name, mode='paired', **options)
        torch.testing.assert_close(broadcast, reference)
        if info.role == 'distance':
            dense = tp.cdist(a, b, metric=name, **options)
            torch.testing.assert_close(dense, compare(a, b, metric=name, **options))
            raw = (torch.ops.torchpairwise.cdist.collections(aa, bb, name, **options)
                   if isinstance(a, tuple) else torch.ops.torchpairwise.cdist(a, b, name, **options))
            torch.testing.assert_close(raw, dense)
    x = torch.rand(2, 3, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(3, device=device, dtype=torch.float64, requires_grad=True)
    value = tp.single_comparison(x, y, metric='sqeuclidean')
    reference = (x-y).square().sum(-1)
    torch.testing.assert_close(torch.autograd.grad(value.sum(), (x, y)),
                               torch.autograd.grad(reference.sum(), (x, y)))
    lengths = torch.tensor([3, 2], device=device)
    cloud = torch.rand(2, 3, 2, device=device, dtype=torch.float64)
    actual = tp.single_comparison(cloud, cloud.flip(0), metric='chamfer',
                                  x1_lengths=lengths, x2_lengths=lengths.flip(0))
    expected = tp.paired_distances(cloud[:, None], cloud.flip(0)[:, None], metric='chamfer',
                                   x1_lengths=lengths[:, None], x2_lengths=lengths.flip(0)[:, None]).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    a, b, _, _ = _example('mixture', device)
    for mode, helper in [('all_pairs', tp.pairwise_kernels), ('paired', tp.paired_kernels),
                         ('condensed', tp.condensed_kernels)]:
        expected = compare(a, None if mode == 'condensed' else b,
                           metric='gaussian_mixture_overlap_kernel', mode=mode)
        actual = helper(a, kernel='gaussian_mixture_overlap') if mode == 'condensed' else helper(a, b, kernel='gaussian_mixture_overlap')
        torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match='Unknown comparison'):
        tp.cdist(x, x, metric='ssim_scores', data_range=1.)
    with pytest.raises((RuntimeError, TypeError), match='parameter.*gamma|gamma'):
        torch.ops.torchpairwise.cdist(x, x, 'euclidean', gamma=.1)
    with pytest.raises(ValueError, match='x2'):
        compare(x, mode='single')
    with pytest.raises((RuntimeError, ValueError), match='rank'):
        tp.single_comparison(torch.tensor(1., device=device), y)

    for invalid in (0, -1, True, float('inf'), 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            tp.single_comparison(cloud, cloud, metric='chamfer', chunk_size=invalid)
