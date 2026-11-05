"""Kernel ridge coefficients from native CG, with an optional small dense oracle."""
import argparse

import torch
import torchpairwise as tp


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--dense-check', action='store_true')
    args = parser.parse_args(argv)
    generator = torch.Generator(device=args.device).manual_seed(19)
    x = torch.randn(args.samples, 3, dtype=torch.float64, device=args.device, generator=generator)
    y = (x[:, :1].sin() + .2 * x[:, 1:2].square())  # [N,1]
    gamma, alpha = .3, .2
    kernel = tp.lazy(x, metric='rbf', gamma=gamma, chunk_size=args.chunk_size)
    operator = kernel + alpha * tp.eye_like(kernel)
    result = tp.linalg.solve(operator, y, method="cg", rtol=1e-8, max_iter=300)
    if not result.converged.all():
        raise RuntimeError(f'CG failed: status={result.status.tolist()}')
    predictions = kernel.matmul(result.solution)  # [N,1], training values
    print('Coefficients:', list(result.solution.shape))
    print('Iterations:', result.iterations.tolist(), 'residual:', result.residual_norm.tolist())
    print('Training RMSE:', (predictions - y).square().mean().sqrt().item())
    if args.dense_check:
        # Only the reference allocates [N,N]. Same system, no fitted intercept.
        dense = tp.rbf_kernel(x, gamma=gamma)
        expected = torch.linalg.solve(dense + alpha * torch.eye(args.samples, device=x.device, dtype=x.dtype), y)
        torch.testing.assert_close(result.solution, expected, rtol=1e-6, atol=1e-7)
        print('Dense solve agrees')


if __name__ == '__main__':
    main()
