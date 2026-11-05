"""Value oracles for every public distance/kernel, independently of dispatch."""
import numpy as np
import pytest
import scipy.spatial.distance as sd
import sklearn.metrics.pairwise as sk
import torch

import torchpairwise as tp

DISTANCES = [
    'euclidean', 'sqeuclidean', 'manhattan', 'minkowski', 'chebyshev',
    'cosine', 'correlation', 'braycurtis', 'canberra', 'jensenshannon',
    'hamming', 'jaccard', 'dice', 'yule', 'rogerstanimoto', 'russellrao',
    'sokalmichener', 'sokalsneath', 'kulsinski', 'kulczynski1', 'snr',
    'wminkowski', 'seuclidean', 'mahalanobis', 'haversine',
]
BOOLEAN = {'jaccard', 'dice', 'yule', 'rogerstanimoto', 'russellrao',
           'sokalmichener', 'sokalsneath', 'kulsinski', 'kulczynski1'}
KERNELS = ['linear', 'polynomial', 'sigmoid', 'rbf', 'laplacian',
           'cosine_similarity', 'additive_chi2', 'chi2']


def inputs(metric, device, dtype=torch.float64):
    g = torch.Generator().manual_seed(83)
    d = 2 if metric == 'haversine' else 5
    x = torch.rand(4, d, generator=g, dtype=dtype) + 0.2
    y = torch.rand(3, d, generator=g, dtype=dtype) + 0.3
    if metric in BOOLEAN:
        x, y = x > 0.7, y > 0.7
    return x.to(device), y.to(device)


def parameters(metric, x):
    if metric == 'wminkowski':
        return {'p': 1.7, 'w': torch.linspace(0.5, 1.5, x.shape[-1], device=x.device, dtype=x.dtype)}
    if metric == 'minkowski':
        return {'p': 1.7}
    if metric == 'seuclidean':
        return {'V': torch.linspace(0.5, 1.5, x.shape[-1], device=x.device, dtype=x.dtype)}
    if metric == 'mahalanobis':
        return {'VI': torch.eye(x.shape[-1], device=x.device, dtype=x.dtype) * 1.3}
    return {}


def reference(metric, x, y, kwargs):
    a, b = x.cpu().numpy(), y.cpu().numpy()
    kw = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    name = {'manhattan': 'cityblock', 'wminkowski': 'minkowski',
            'sokalmichener': 'rogerstanimoto'}.get(metric, metric)
    if metric == 'haversine':
        return sk.haversine_distances(a, b)
    if metric == 'snr':
        return sd.cdist(a, b, lambda u, v: (v - u).var() / u.var())
    if metric in {'kulsinski', 'kulczynski1'}:
        aa, bb = a[:, None], b[None, :]
        both = (aa & bb).sum(-1)
        different = (aa ^ bb).sum(-1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return ((different - both + a.shape[-1]) / (different + a.shape[-1])
                    if metric == 'kulsinski' else both / different)
    return sd.cdist(a, b, metric=name, **kw)


@pytest.mark.parametrize('metric', DISTANCES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_distances_against_reference(metric, dtype, device):
    x, y = inputs(metric, device, dtype)
    kw = parameters(metric, x)
    result = getattr(tp, metric + '_distances')(x, y, **kw)
    expected = torch.as_tensor(reference(metric, x, y, kw), device=device, dtype=result.dtype)
    torch.testing.assert_close(result, expected, atol=3e-6, rtol=3e-5, equal_nan=True)
    torch.testing.assert_close(tp.cdist(x, y, metric=metric, **kw), result, equal_nan=True)


@pytest.mark.parametrize('kernel', KERNELS)
def test_kernels_against_sklearn(kernel, device):
    x, y = inputs('euclidean', device)
    name = kernel if kernel == 'cosine_similarity' else kernel + '_kernel'
    result = getattr(tp, name)(x, y)
    expected = getattr(sk, name)(x.cpu().numpy(), y.cpu().numpy())
    torch.testing.assert_close(result, torch.as_tensor(expected, device=device), atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize('alias,canonical', [('l1', 'manhattan'), ('l2', 'euclidean'),
                                          ('lp', 'minkowski'), ('linf', 'chebyshev'),
                                          ('cityblock', 'manhattan')])
def test_aliases(alias, canonical, device):
    x, y = inputs('euclidean', device)
    torch.testing.assert_close(getattr(tp, alias + '_distances')(x, y),
                               getattr(tp, canonical + '_distances')(x, y))


@pytest.mark.parametrize('metric', DISTANCES)
def test_noncontiguous(metric, device):
    x, y = inputs(metric, device)
    xx = torch.stack((x, x), -1)[..., 0]
    yy = torch.stack((y, y), -1)[..., 0]
    assert not xx.is_contiguous()
    fn = getattr(tp, metric + '_distances')
    torch.testing.assert_close(fn(xx, yy, **parameters(metric, x)),
                               fn(x, y, **parameters(metric, x)), equal_nan=True)


@pytest.mark.parametrize('metric', sorted(BOOLEAN | {'hamming'}))
def test_all_false_boolean_conventions(metric, device):
    x = torch.zeros(2, 5, dtype=torch.bool, device=device)
    y = torch.zeros(3, 5, dtype=torch.bool, device=device)
    with np.errstate(invalid='ignore', divide='ignore'):
        expected = reference(metric, x, y, {})
    result = getattr(tp, metric + '_distances')(x, y)
    torch.testing.assert_close(result, torch.as_tensor(expected, dtype=result.dtype, device=device), equal_nan=True)
