"""Normalized KL/Jeffreys and weighted angular distances with gradients."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.tensor([[1., 2., 3.], [2., 4., 1.]], device=device, requires_grad=True)
    y = torch.tensor([[2., 3., 1.], [1., 3., 4.]], device=device, requires_grad=True)
    w = torch.tensor([.5, 1., 2.], device=device, requires_grad=True)
    for metric, kwargs in (('kl', dict(base=2.)), ('symmetric_kl', {}),
                            ('cosine', dict(w=w)), ('correlation', dict(w=w))):
        matrix = tp.cdist(x, y, metric=metric, **kwargs)
        paired = tp.paired_distances(x, y, metric=metric, **kwargs)
        inputs = (x, y, w) if 'w' in kwargs else (x, y)
        assert all(torch.isfinite(g).all() for g in torch.autograd.grad(
            matrix.sum() + paired.sum(), inputs))
        print(metric, 'matrix:', matrix.detach(), 'paired:', paired.detach())
        if metric != 'kl':
            print('condensed:', tp.pdist(x, metric=metric, **kwargs).detach())
    print('KL/Jeffreys and weighted cosine/correlation gradients passed on', device)


if __name__ == '__main__':
    main()
