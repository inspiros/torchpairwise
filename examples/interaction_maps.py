"""Geometric updates with no retained pair-displacement tensor or transport plan."""
import torch
import torchpairwise as tp


def main():
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(20, 3, device=device, requires_grad=True)
    y = torch.randn(24, 3, device=device, requires_grad=True)
    force = tp.lazy(x, y, metric='rbf', gamma=.2, chunk_size=8).displacement()
    shift = tp.sinkhorn_displacement(x, y, regularization=.5, max_iter=20, chunk_size=8)
    loss = force.square().mean() + shift.square().mean()
    loss.backward()
    print('kernel displacement:', tuple(force.shape))
    print('transport displacement:', tuple(shift.shape))
    print('finite coordinate gradients:', bool(x.grad.isfinite().all() & y.grad.isfinite().all()))


if __name__ == '__main__':
    main()
