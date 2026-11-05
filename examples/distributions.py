"""Histogram, empirical-distribution and Gaussian comparisons with gradients."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.tensor([[1., 2., 3.], [3., 1., 2.]], device=device, requires_grad=True)
    y = torch.tensor([[2., 3., 1.], [1., 3., 2.]], device=device, requires_grad=True)
    for fn in (tp.bhattacharyya_kernel, tp.bhattacharyya_distances, tp.aitchison_distances):
        value = fn(x, y)
        assert all(torch.isfinite(g).all() for g in torch.autograd.grad(value.sum(), (x,y)))
        print(fn.__name__, value.detach())
    a = torch.tensor([[[0., 0.], [1., 1.]], [[1., 0.], [0., 1.]]],
                     device=device, requires_grad=True)
    b = (a.detach() + .2).requires_grad_()
    for metric, kwargs in (('energy', {}), ('mmd', dict(kernel='rbf', length_scale=.7))):
        value = tp.cdist(a, b, metric=metric, **kwargs)
        assert all(torch.isfinite(g).all() for g in torch.autograd.grad(value.sum(), (a,b)))
        print(metric, value.detach(), 'paired:', tp.paired_distances(a,b,metric=metric,**kwargs).detach(),
              'condensed:', tp.pdist(a,metric=metric,**kwargs).detach())
    means = torch.tensor([[0., 0.], [1., 2.]], device=device, requires_grad=True)
    covariances = torch.eye(2, device=device).expand(2,2,2).clone().requires_grad_()
    value = tp.gaussian_wasserstein_distances(means, covariances, means + 1, covariances * 2)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(value.sum(), (means,covariances)))
    print('gaussian_wasserstein', value.detach())
    print('Distribution comparison gradients passed on', device)


if __name__ == '__main__':
    main()
