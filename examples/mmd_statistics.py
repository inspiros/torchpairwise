"""Unbiased polynomial MMD on supplied features, without an encoder dependency."""
import torch
import torchpairwise as tp


def main():
    torch.manual_seed(2026)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    real = torch.randn(17, 5, device=device, dtype=torch.float64, requires_grad=True)
    generated = torch.randn(23, 5, device=device, dtype=real.dtype, requires_grad=True)
    # Each sample in mmd_statistics is a distribution represented by points.
    # Unsqueeze to compare one feature distribution against one other.
    options = dict(kernel='polynomial', degree=3, gamma=1 / real.shape[-1], coef0=1.)
    statistic = tp.mmd_statistics(real[None], generated[None], **options).squeeze()

    # Small dense oracle only. The native operation evaluates bounded point tiles.
    xx = tp.polynomial_kernel(real, real, gamma=.2)
    yy = tp.polynomial_kernel(generated, generated, gamma=.2)
    xy = tp.polynomial_kernel(real, generated, gamma=.2)
    n, m = real.shape[0], generated.shape[0]
    expected = ((xx.sum() - xx.diagonal().sum()) / (n * (n - 1))
                + (yy.sum() - yy.diagonal().sum()) / (m * (m - 1)) - 2 * xy.mean())
    torch.testing.assert_close(statistic, expected)
    for actual, reference in zip(torch.autograd.grad(statistic, (real, generated)),
                                  torch.autograd.grad(expected, (real, generated))):
        torch.testing.assert_close(actual, reference)

    # The estimate is signed. Even two identical finite samples can give a
    # negative unbiased estimate; clamping it would change the estimator.
    identical = tp.mmd_statistics(real[None], real[None], **options).squeeze()
    assert identical < 0
    print('Unbiased polynomial MMD values/gradients passed on', device)
    print('Independent feature collections:', statistic.item())
    print('Identical finite sample:', identical.item())
    # To reproduce KID, the caller must also match the benchmark encoder,
    # feature preprocessing and any random subset/aggregation protocol.


if __name__ == '__main__':
    main()
