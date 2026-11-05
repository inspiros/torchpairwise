"""Isotropic kernels in all-pairs, paired and condensed modes, with gradients."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(4, 3, device=device, requires_grad=True)
    y = torch.rand(4, 3, device=device, requires_grad=True)
    for kernel, options in [('rational_quadratic', {'alpha': .5}), ('matern', {'nu': 2.5})]:
        options['length_scale'] = 1.7
        matrix = tp.pairwise_kernels(x, y, kernel=kernel, **options)
        paired = tp.paired_kernels(x, y, kernel=kernel, **options)
        condensed = tp.condensed_kernels(x, kernel=kernel, **options)
        torch.testing.assert_close(paired, matrix.diagonal())
        gradients = torch.autograd.grad(matrix.sum() + paired.sum() + condensed.sum(), (x, y))
        assert all(torch.isfinite(value).all() for value in gradients)
        print(kernel, 'all-pairs:', tuple(matrix.shape),
              'paired:', tuple(paired.shape), 'condensed:', tuple(condensed.shape))


if __name__ == '__main__':
    main()
