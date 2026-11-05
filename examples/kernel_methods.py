"""Kernel products, centered Gram operators, and MMD without a dense Gram matrix."""
import torch
import torchpairwise as tp


def centered_kernel_product(x, values, *, kernel='rbf', **options):
    """Apply H K(X,X) H to one or more vectors; H centers the sample axis."""
    centered = values - values.mean(dim=-2, keepdim=True)
    result = tp.lazy(x, x, metric=kernel, **options) @ centered
    return result - result.mean(dim=-2, keepdim=True)


def biased_mmd_squared(x, y, *, kernel='rbf', **options):
    """Biased empirical MMD squared for two nonempty [samples,features] sets.

    This example composes the registry-wide reductions. For a distance, choose
    a positive-semidefinite kernel on your input domain. It retains finite-sum
    roundoff instead of clamping negative values or taking a square root.
    """
    xx = tp.lazy(x, x, metric=kernel, **options).sum() / x.shape[-2] ** 2
    yy = tp.lazy(y, y, metric=kernel, **options).sum() / y.shape[-2] ** 2
    xy = tp.lazy(x, y, metric=kernel, **options).sum() / (x.shape[-2] * y.shape[-2])
    return xx + yy - 2 * xy


def main():
    torch.manual_seed(1729)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(17, 5, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.randn(23, 5, device=device, dtype=x.dtype, requires_grad=True)
    values = torch.randn(17, 2, device=device, dtype=x.dtype, requires_grad=True)
    result = centered_kernel_product(x, values, gamma=.2, chunk_size=8)
    # Small dense oracles only: the production helpers above use streamed products.
    gram = tp.rbf_kernel(x, x, gamma=.2)
    centered = values - values.mean(dim=0, keepdim=True)
    expected = gram @ centered
    expected = expected - expected.mean(dim=0, keepdim=True)
    torch.testing.assert_close(result, expected)
    options = dict(degree=2, gamma=.2, coef0=1.)
    mmd2 = biased_mmd_squared(x, y, kernel='polynomial', chunk_size=8, **options)
    dense = (tp.polynomial_kernel(x, x, **options).mean()
             + tp.polynomial_kernel(y, y, **options).mean()
             - 2 * tp.polynomial_kernel(x, y, **options).mean())
    dedicated = tp.mmd_distances(x.unsqueeze(0), y.unsqueeze(0),
                                 kernel='polynomial', squared=True, **options).squeeze()
    torch.testing.assert_close(mmd2, dense)
    torch.testing.assert_close(dedicated, dense)
    actual = torch.autograd.grad(result.square().mean() + mmd2, (x, y, values))
    reference = torch.autograd.grad(expected.square().mean() + dense, (x, y, values))
    for gradient, oracle in zip(actual, reference):
        torch.testing.assert_close(gradient, oracle)
    print('Centered kernel operator and streamed polynomial MMD passed on', device)


if __name__ == '__main__':
    main()
