"""Approximate a rectangular kernel map with native truncated singular triplets.

python examples/rectangular_svd.py --device cuda --dense-check

The streamed decomposition retains narrow sample blocks and source tiles. Its
low-rank approximation can apply new RHS blocks without evaluating any further
pairs. Building the approximation still requires repeated full pairwise passes.
"""
import argparse

import torch
import torchpairwise as tp


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--rows', type=int, default=96)
    parser.add_argument('--columns', type=int, default=64)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--dense-check', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.rank <= min(args.rows, args.columns):
        parser.error('rank must be between 1 and min(rows,columns)')
    generator = torch.Generator(device=args.device).manual_seed(903)
    x = torch.randn(args.rows, 3, dtype=torch.float64, device=args.device, generator=generator)
    y = torch.randn(args.columns, 3, dtype=x.dtype, device=x.device, generator=generator)
    source = tp.lazy(x, y, gamma=.2, chunk_size=32, metric='rbf')
    width = min(args.rows, args.columns, args.rank + 8)
    initial = torch.randn(args.columns, width, dtype=x.dtype, device=x.device, generator=generator)
    fit = tp.linalg.svd(source, k=args.rank, initial=initial, max_iter=150, rtol=1e-7, atol=1e-10)
    if not fit.converged.all():
        raise RuntimeError(f'Singular status: {fit.status.tolist()}')
    rhs = torch.randn(args.columns, 2, dtype=x.dtype, device=x.device, generator=generator)
    # Reuse fitted factors: [N,k] @ ([k,1]*([k,M] @ [M,R])) -> [N,R].
    approximation = fit.left_vectors @ (fit.singular_values[:, None] * (fit.right_vectors.T @ rhs))
    print('Logical kernel:', [args.rows, args.columns], 'retained factors:', list(fit.left_vectors.shape), list(fit.right_vectors.shape))
    print('Singular values:', fit.singular_values.tolist())
    print('Iterations:', fit.iterations.tolist(), 'two-sided residual:', fit.residual_norm.tolist())
    print('Approximate product:', list(approximation.shape))
    if args.dense_check:
        matrix = tp.rbf_kernel(x, y, gamma=.2)
        u, values, vh = torch.linalg.svd(matrix, full_matrices=False)
        torch.testing.assert_close(fit.singular_values, values[:args.rank], rtol=1e-6, atol=1e-8)
        lowrank = (u[:, :args.rank] * values[:args.rank]) @ vh[:args.rank]
        reconstructed = (fit.left_vectors * fit.singular_values) @ fit.right_vectors.T
        torch.testing.assert_close(reconstructed, lowrank, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(approximation, lowrank @ rhs, rtol=2e-5, atol=2e-6)
        relative_error = (source.matmul(rhs) - approximation).norm() / source.matmul(rhs).norm()
        print('Relative product approximation error:', relative_error.item())


if __name__ == '__main__':
    main()
