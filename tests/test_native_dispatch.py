"""Python and native selectors share catalog scope and reject unused options."""
import pytest
import torch
import torchpairwise as tp

# Representatives of the distinct legacy schema/parameter routes; aliases and
# numerical formulas are covered by comparison_dispatch and the metric suites.
_LEGACY = ('euclidean', 'canberra', 'wminkowski', 'mahalanobis',
           'directed_hausdorff', 'hamming', 'l2')
_BOOLEAN = {'hamming', 'jaccard', 'kulsinski', 'kulczynski1', 'yule', 'dice',
            'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath'}

@pytest.mark.parametrize('metric', _LEGACY)
def test_legacy_native_distance_selectors_share_complete_route(metric, device):
    x = torch.rand(3, 2, device=device, dtype=torch.float64) + .2
    y = torch.rand(4, 2, device=device, dtype=torch.float64) + .2
    options = {}
    if metric in _BOOLEAN:
        x, y = x > .6, y > .6
    elif metric == 'directed_hausdorff':
        x, y = x[:, None].repeat(1, 3, 1), y[:, None].repeat(1, 4, 1)
    elif metric in ('minkowski', 'wminkowski', 'lp'):
        options['p'] = 1.7
        if metric == 'wminkowski':
            options['w'] = torch.tensor([.7, 1.2], device=device, dtype=x.dtype)
    elif metric == 'seuclidean':
        options['V'] = torch.tensor([.8, 1.3], device=device, dtype=x.dtype)
    elif metric == 'mahalanobis':
        options['VI'] = torch.eye(2, device=device, dtype=x.dtype)
    elif metric == 'jensenshannon':
        options['base'] = 2.
    torch.testing.assert_close(torch.ops.torchpairwise.cdist(x, y, metric, **options),
                               tp.cdist(x, y, metric, **options), equal_nan=True)
    if metric not in ('snr', 'directed_hausdorff'):
        torch.testing.assert_close(torch.ops.torchpairwise.pdist(x, metric, **options),
                                   tp.pdist(x, metric, **options), equal_nan=True)

@pytest.mark.parametrize('kernel', ['linear', 'polynomial', 'chi2'])
def test_legacy_native_kernel_selector_shares_complete_route(kernel, device):
    x = torch.rand(3, 2, device=device, dtype=torch.float64) + .2
    options = {'gamma': .4} if kernel in ('polynomial', 'sigmoid', 'rbf', 'laplacian', 'chi2') else {}
    if kernel == 'polynomial':
        options.update(degree=2, coef0=.7)
    elif kernel == 'sigmoid':
        options['coef0'] = .7
    torch.testing.assert_close(torch.ops.torchpairwise.pdist_kernel(x, kernel, **options),
                               tp.condensed_kernels(x, kernel=kernel, **options))

def test_native_selector_catalog_expansion_and_unused_option_checks():
    x = torch.tensor([[1., 2.], [2., 3.], [3., 1.]])
    torch.testing.assert_close(torch.ops.torchpairwise.cdist(x, x, 'euclidean', w=torch.ones(2)),
                               tp.cdist(x, x, 'euclidean', w=torch.ones(2)))
    torch.testing.assert_close(torch.ops.torchpairwise.cdist(x, x, 'angular'), tp.cdist(x, x, 'angular'))
    with pytest.raises(TypeError, match='gamma'):
        torch.ops.torchpairwise.cdist(x, x, 'euclidean', gamma=.2)
    with pytest.raises(RuntimeError, match='Unknown'):
        torch.ops.torchpairwise.cdist(x, x, 'missing')
