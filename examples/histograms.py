"""Distribution distance versus raw-mass similarity, with input gradients."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.tensor([[1., 3., 0.], [0., 0., 2.]], device=device, requires_grad=True)
    y = torch.tensor([[2., 6., 0.], [1., 0., 0.]], device=device, requires_grad=True)
    distance = tp.hellinger_distances(x, y)
    intersection = tp.histogram_intersection_kernel(x, y)
    torch.testing.assert_close(tp.paired_distances(x, y, metric='hellinger'), distance.diagonal())
    torch.testing.assert_close(tp.paired_kernels(x, y, kernel='histogram_intersection'), intersection.diagonal())
    for output in (distance, intersection, tp.pdist(x, metric='hellinger'),
                   tp.condensed_kernels(x, kernel='histogram_intersection')):
        assert all(torch.isfinite(g).all() for g in torch.autograd.grad(output.sum(), (x, y),
                                                                       allow_unused=True) if g is not None)
    print('Hellinger (unit-mass rows):', distance)
    print('Histogram intersection (raw mass):', intersection)


if __name__ == '__main__':
    main()
