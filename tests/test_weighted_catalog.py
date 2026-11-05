"""Weighted catalog values; common assembly is checked without a mode cross-product."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.spatial.distance import yule
import torch
import torchpairwise as tp


BOOLEAN = ('jaccard', 'kulsinski', 'kulczynski1', 'yule', 'dice',
           'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath')
NUMERIC = ('euclidean', 'sqeuclidean', 'manhattan', 'minkowski', 'chebyshev',
           'canberra', 'braycurtis', 'hamming')


@pytest.mark.parametrize('metric', (*NUMERIC, *BOOLEAN))
def test_weighted_catalog_reference(metric, device):
    x = torch.tensor([[0., 1., 2., 4.], [3., 2., 1., .5], [1., 3., 2., 0.]],
                     device=device, dtype=torch.float64)
    y = x.flip(0) + .3
    if metric in BOOLEAN:
        x, y = x > 1.5, y > 1.5
    w = torch.tensor([.5, 0., 2., 1.], device=device, dtype=torch.float64)
    kwargs = {'p': 1.7} if metric == 'minkowski' else {}
    a, b, weights = (v.cpu().numpy() for v in (x, y, w))
    if metric in ('kulsinski', 'kulczynski1'):
        different = ((a[:, None] != b[None]) * weights).sum(-1)
        both = ((a[:, None] & b[None]) * weights).sum(-1)
        with np.errstate(invalid='ignore', divide='ignore'):
            expected = ((different - both + weights.sum()) / (different + weights.sum())
                        if metric == 'kulsinski' else both / different)
    elif metric == 'yule':
        # Some SciPy cdist builds incorrectly return zero for fractional-weight
        # Yule. Its scalar implementation follows the documented formula.
        expected = np.asarray([[yule(u, v, w=weights) for v in b] for u in a])
    else:
        name = {'manhattan': 'cityblock', 'sokalmichener': 'rogerstanimoto'}.get(metric, metric)
        with np.errstate(invalid='ignore', divide='ignore'):
            expected = cdist(a, b, metric=name, w=weights, **kwargs)
    matrix = tp.cdist(x, y, metric=metric, w=w, **kwargs)
    torch.testing.assert_close(tp.paired_distances(x, y, metric=metric, w=None, **kwargs),
                               tp.paired_distances(x, y, metric=metric, **kwargs), equal_nan=True)
    torch.testing.assert_close(tp.pdist(x, metric=metric, w=None, **kwargs),
                               tp.pdist(x, metric=metric, **kwargs), equal_nan=True)
    torch.testing.assert_close(matrix, torch.as_tensor(expected, device=device), equal_nan=True)
    torch.testing.assert_close(getattr(tp, metric + '_distances')(x, y, w=w, **kwargs), matrix, equal_nan=True)
    torch.testing.assert_close(tp.paired_distances(x, y, metric=metric, w=w, **kwargs), matrix.diagonal(), equal_nan=True)
    i, j = torch.triu_indices(3, 3, 1, device=device)
    torch.testing.assert_close(tp.pdist(x, metric=metric, w=w, **kwargs),
                               tp.cdist(x, x, metric=metric, w=w, **kwargs)[i, j], equal_nan=True)
    result = tp.cdist(x.expand(2, 1, *x.shape), y.expand(1, 2, *y.shape),
                      metric=metric, w=w.expand(2, 2, -1), **kwargs)
    torch.testing.assert_close(result, matrix.expand(2, 2, *matrix.shape), equal_nan=True)


@pytest.mark.parametrize('metric', ['canberra', 'braycurtis', 'sqeuclidean', 'euclidean', 'minkowski'])
def test_weighted_numeric_gradients(metric, device):
    x = torch.tensor([[.1, .7, 1.4], [.4, 1.1, .3]], device=device, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[.3, .5, 1.8], [1.2, .2, .5]], device=device, dtype=x.dtype, requires_grad=True)
    w = torch.tensor([.5, 1.2, .8], device=device, dtype=x.dtype, requires_grad=True)
    kw = {'p': 1.7} if metric == 'minkowski' else {}
    assert torch.autograd.gradcheck(lambda a,b,c: tp.cdist(a,b,metric=metric,w=c,**kw), (x,y,w), fast_mode=True)
    paired = tp.paired_distances(x, y, metric=metric, w=w, **kw)
    diagonal = tp.cdist(x, y, metric=metric, w=w, **kw).diagonal()
    for actual, expected in zip(torch.autograd.grad(paired.sum(), (x,y,w)),
                                torch.autograd.grad(diagonal.sum(), (x,y,w))):
        torch.testing.assert_close(actual, expected)


def test_weighted_boolean_and_hamming_boundaries(device):
    zero = torch.zeros(2, 3, device=device, dtype=torch.bool)
    w = torch.tensor([0., 1., 2.], device=device, requires_grad=True)
    for metric in BOOLEAN:
        result = tp.cdist(zero, zero, metric=metric, w=w)
        expected = tp.cdist(zero, zero, metric=metric)
        torch.testing.assert_close(result, expected, equal_nan=True)
        with pytest.raises(ValueError, match='positive total'):
            tp.cdist(zero, zero, metric=metric, w=w * 0)
        assert tp.pdist(zero[:1], metric=metric, w=w).shape == (0,)
    # Hamming compares values, rather than converting numeric inputs to bool.
    a = torch.tensor([[1, 2, 3]], device=device)
    b = torch.tensor([[2, 2, 4]], device=device)
    result = tp.cdist(a, b, metric='hamming', w=w)
    torch.testing.assert_close(result, w.new_tensor([[2 / 3]]))
    assert torch.isfinite(torch.autograd.grad(result.sum(), w)[0]).all()
    x = torch.ones(2, 3, device=device)
    for metric in ('euclidean', 'sqeuclidean', 'manhattan', 'canberra', 'chebyshev'):
        result = tp.cdist(x, x * 2, metric=metric, w=torch.zeros_like(w))
        torch.testing.assert_close(result, torch.zeros_like(result))


def test_weighted_ignored_large_coordinate_and_chebyshev_weights(device):
    x = torch.tensor([[1e20, 1.]], device=device, requires_grad=True)
    y = torch.zeros_like(x)
    w = torch.tensor([0., 1.], device=device)
    for metric in ('sqeuclidean', 'euclidean', 'minkowski', 'canberra', 'braycurtis'):
        for call in (tp.cdist, tp.paired_distances):
            result = call(x, y, metric=metric, w=w)
            torch.testing.assert_close(result, torch.ones_like(result))
            assert torch.isfinite(torch.autograd.grad(result.sum(), x)[0]).all()
    result = tp.cdist(x, y, metric='chebyshev', w=torch.tensor([3e38, 3e38], device=device))
    torch.testing.assert_close(result, x.new_tensor([[1e20]]))


@pytest.mark.compiler
def test_weighted_catalog_compiled_first_gradients(device):
    torch.compiler.reset()
    x = torch.tensor([[.2, .7, 1.3], [.8, .3, .6]], device=device, requires_grad=True)
    y = torch.tensor([[.5, .4, 1.8], [.3, .9, .2]], device=device, requires_grad=True)
    w = torch.tensor([.5, 1.3, .8], device=device, requires_grad=True)

    def call(a, b, c):
        values = [tp.cdist(a, b, metric=m, w=c).sum()
                  + tp.paired_distances(a, b, metric=m, w=c).sum()
                  + tp.pdist(a, metric=m, w=c).sum()
                  for m in ('sqeuclidean', 'canberra', 'braycurtis')]
        values.append(tp.cdist(a > .5, b > .5, metric='jaccard', w=c).sum())
        return torch.stack(values)

    expected = call(x, y, w)
    actual = torch.compile(call, fullgraph=True)(x, y, w)
    torch.testing.assert_close(actual, expected)
    for value, reference in zip(torch.autograd.grad(actual.sum(), (x,y,w)),
                                torch.autograd.grad(expected.sum(), (x,y,w))):
        torch.testing.assert_close(value, reference, atol=2e-5, rtol=2e-4)
    torch.compiler.reset()
