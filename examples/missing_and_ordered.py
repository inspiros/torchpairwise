"""Missing features versus transport of masses on an ordered histogram grid."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.tensor([[1., float('nan'), 3.], [2., 4., 6.]], device=device, requires_grad=True)
    y = torch.tensor([[2., 2., 3.], [float('nan'), 5., 7.]], device=device, requires_grad=True)
    distance = tp.nan_euclidean_distances(x, y)
    torch.testing.assert_close(tp.paired_distances(x, y, metric='nan_euclidean'), distance.diagonal())
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(distance.sum(), (x, y)))
    print('NaN-aware Euclidean:', distance)

    masses = torch.tensor([[1., 0., 0.], [0., 0., 2.]], device=device, requires_grad=True)
    grid = torch.tensor([0., 1., 4.], device=device, requires_grad=True)
    distance = tp.wasserstein_1d_distances(masses, bin_positions=grid)
    condensed = tp.pdist(masses, metric='wasserstein_1d', bin_positions=grid)
    torch.testing.assert_close(condensed, distance[0, 1].reshape(1))
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(condensed.sum(), (masses, grid)))
    print('Wasserstein on shared bins:', distance)


if __name__ == '__main__':
    main()
