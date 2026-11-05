"""Combine scalar kernels and distances before a streamed consumer."""
import argparse
import torch
import torchpairwise as tp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    device = parser.parse_args().device
    torch.manual_seed(125)
    x = torch.randn(7, 3, device=device, requires_grad=True)
    y = torch.randn(9, 3, device=device, requires_grad=True)
    values = torch.randn(9, 2, device=device, requires_grad=True)
    alpha = torch.tensor(.3, device=device, requires_grad=True)
    rbf = tp.lazy(x, y, metric='rbf', gamma=.5, chunk_size=4)
    laplacian = tp.lazy(x, y, metric='laplacian', gamma=.3, chunk_size=4)
    mixture = alpha * rbf + (1-alpha) * laplacian
    product = mixture.matmul(values)
    dense = alpha * tp.rbf_kernel(x, y, gamma=.5) + (1-alpha) * tp.laplacian_kernel(x, y, gamma=.3)
    torch.testing.assert_close(product, dense @ values)
    alignment = (rbf * laplacian).sum()
    torch.testing.assert_close(alignment, (tp.rbf_kernel(x, y, gamma=.5) * tp.laplacian_kernel(x, y, gamma=.3)).sum())
    distance = tp.lazy(x, y, metric='sqeuclidean', chunk_size=4)
    weights = torch.linspace(.2, 1., 9, device=device)
    affinity = (-.5 * distance).exp().weight_columns(weights)
    torch.testing.assert_close(affinity.matmul(values), ((-.5 * tp.sqeuclidean_distances(x, y)).exp() * weights) @ values)
    gradients = torch.autograd.grad(product.square().sum() + alignment, (x, y, values, alpha))
    assert all(g.isfinite().all() for g in gradients)
    print('Mixture product:', tuple(product.shape), 'kernel alignment:', float(alignment.detach()))


if __name__ == '__main__':
    main()
