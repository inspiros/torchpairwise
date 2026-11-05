"""Compare exact, sliced, and regularized transport between small point clouds."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.tensor([[[-1., -1.], [1., 1.]],
                      [[-.5, -.5], [.5, .5]]], device=device, requires_grad=True)
    y = torch.tensor([[[-1., 1.], [1., -1.]],
                      [[-.2, .3], [.8, -.4]]], device=device, requires_grad=True)
    # Shared directions are explicit and reusable. Coordinate axes alone would
    # miss the first pair's joint-distribution difference; include diagonals.
    directions = torch.tensor([[1., 0., 1., 1.], [0., 1., 1., -1.]], device=device)
    directions = directions / torch.linalg.vector_norm(directions, dim=0)
    options = (
        ('wasserstein', dict(p=1)),
        ('sliced_wasserstein', dict(projections=directions, p=2)),
        ('sinkhorn', dict(regularization=.5, max_iter=100)),
    )
    for metric, kwargs in options:
        matrix = tp.cdist(x, y, metric=metric, **kwargs)
        paired = tp.paired_distances(x, y, metric=metric, **kwargs)
        condensed = tp.pdist(x, metric=metric, **kwargs)
        gradients = torch.autograd.grad(matrix.sum() + paired.sum() + condensed.sum(), (x, y))
        assert all(torch.isfinite(value).all() for value in gradients)
        print(metric, 'all-pairs:', matrix.detach(), 'paired:', paired.detach(),
              'condensed:', condensed.detach())
    print('Exact joint W1 is 2 despite identical coordinate marginals.')
    print('Exact solver uses CPU coordination for CUDA inputs; outputs stay on', device)


if __name__ == '__main__':
    main()
