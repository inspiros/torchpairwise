"""Compare Gaussian-mixture samples: python examples/gaussian_mixtures.py --device cpu."""
import argparse

import torch
import torchpairwise as tp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    options = {'device': args.device, 'dtype': torch.float64}
    means = torch.tensor([[[0., 0.], [2., 0.], [float('nan'), float('nan')]],
                          [[1., 1.], [3., 1.], [4., -1.]]], **options).requires_grad_()
    covariances = torch.eye(2, **options).expand(2, 3, 2, 2).clone()
    # These are explicit probability weights, chosen independently of opacity.
    weights = torch.tensor([[1., 3., 0.], [2., 1., 1.]], **options)
    lengths = torch.tensor([2, 3], device=args.device, dtype=torch.int64)
    print('Density overlap:')
    print(tp.gaussian_mixture_overlap_kernel(means, covariances, weights, x1_lengths=lengths))
    print('Condensed component-restricted MW2:')
    print(tp.condensed_gaussian_mixture_wasserstein_distances(
        means, covariances, weights, x1_lengths=lengths))
    print('Debiased entropic component transport (squared scale):')
    print(tp.gaussian_mixture_sinkhorn_divergences(
        means, covariances, weights, x1_lengths=lengths, regularization=.5, max_iter=100))

    target_means = torch.tensor([[[.5, .5]], [[2., .5]]], **options)
    target_covariances = torch.eye(2, **options).expand(2, 1, 2, 2)
    target_weights = torch.ones(2, 1, **options)
    loss = tp.paired_gaussian_mixture_l2_distances(
        means, covariances, weights, target_means, target_covariances, target_weights,
        squared=True, x1_lengths=lengths).sum()
    loss.backward()
    print('Paired squared density L2 loss:', loss.detach())
    print('Mean gradients (padding receives zero):')
    print(means.grad)


if __name__ == '__main__':
    main()
