import pytest
import torch
import torchpairwise as tp


@pytest.mark.parametrize('metric', ['canberra', 'braycurtis', 'haversine', 'sqeuclidean',
                                    'jensenshannon', 'snr', 'wminkowski', 'seuclidean', 'mahalanobis'])
def test_native_gradients(metric, device):
    g = torch.Generator().manual_seed(923)
    d = 2 if metric == 'haversine' else 3
    x = (torch.rand(2, d, generator=g, dtype=torch.float64) + .3).to(device).requires_grad_()
    y = (torch.rand(3, d, generator=g, dtype=torch.float64) + .5).to(device).requires_grad_()
    args = (x, y)
    fn = getattr(tp, metric + '_distances')
    if metric == 'wminkowski':
        w = torch.tensor([.7, 1.2, 1.4], dtype=x.dtype, device=device, requires_grad=True)
        args = (x, y, w)
        fn = lambda a, b, w: tp.wminkowski_distances(a, b, p=1.7, w=w)
    if metric == 'seuclidean':
        v = torch.tensor([.7, 1.2, 1.4], dtype=x.dtype, device=device, requires_grad=True)
        args = (x, y, v)
        fn = lambda a, b, v: tp.seuclidean_distances(a, b, V=v)
    if metric == 'mahalanobis':
        vi = torch.eye(d, dtype=x.dtype, device=device).requires_grad_()
        args = (x, y, vi)
        fn = lambda a, b, vi: tp.mahalanobis_distances(a, b, VI=vi)
    assert torch.autograd.gradcheck(fn, args, fast_mode=True, atol=2e-5, rtol=2e-4)


def test_chi2_gradient(device):
    x = torch.tensor([[.3, .8], [.9, .2]], device=device, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[.7, .5]], device=device, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(tp.additive_chi2_kernel, (x, y), fast_mode=True)
