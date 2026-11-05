"""Compare CG and Jacobi-PCG on an unevenly scaled streamed kernel system.

Solve A W = Y with A = D (K + 2 I) D, D = diag(sqrt(d)). This isolates
diagonal scaling, a case where Jacobi is useful. A constant-diagonal RBF Gram
matrix alone will not benefit: its Jacobi action is just a scalar multiple.
Both iterative paths store tiles and vector blocks, never the full Gram matrix.
The optional dense oracle is deliberately separate and only for small problems.
"""
import argparse
import time

import torch
import torchpairwise as tp


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--dense-reference', action='store_true')
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error('samples must be positive')
    generator = torch.Generator(device=args.device).manual_seed(27)
    x = torch.randn(args.samples, 4, device=args.device, dtype=torch.float64, generator=generator)
    targets = torch.randn(args.samples, 2, device=x.device, dtype=x.dtype, generator=generator)
    d = torch.logspace(-3, 3, args.samples, device=x.device, dtype=x.dtype)
    D = tp.diag_embed(d.sqrt())
    K = tp.lazy(x, gamma=.2, chunk_size=args.chunk_size, metric='rbf')
    K = K + 2. * tp.eye_like(K)
    A = D @ K @ D
    # K_ii = 1 for this RBF kernel; diagonal(A) = 3*d exactly.
    M = tp.linalg.jacobi(3. * d)

    def sync():
        if x.is_cuda:
            torch.cuda.synchronize(x.device)

    # Warm both dispatch paths before timing. No dense oracle allocation yet.
    for inverse in (None, M):
        tp.linalg.cg(A, targets, preconditioner=inverse, max_iter=1)
    fits = []
    for name, inverse in (('CG', None), ('Jacobi-PCG', M)):
        sync()
        if x.is_cuda:
            baseline = torch.cuda.memory_allocated(x.device)
            torch.cuda.reset_peak_memory_stats(x.device)
        start = time.perf_counter()
        # Unpreconditioned CG needs about 1,800 steps at the default 128 samples.
        fit = tp.linalg.cg(A, targets, preconditioner=inverse, rtol=1e-9, max_iter=5000)
        sync()
        seconds = time.perf_counter() - start
        memory = f', peak extra CUDA bytes={torch.cuda.max_memory_allocated(x.device)-baseline}' if x.is_cuda else ''
        print(f'{name}: steps={fit.iterations.tolist()}, seconds={seconds:.4f}, '
              f'residual={fit.residual_norm.tolist()}{memory}')
        if not fit.converged.all():
            raise RuntimeError(f'{name} status={fit.status.tolist()}')
        fits.append(fit)
    torch.testing.assert_close(fits[0].solution, fits[1].solution, rtol=1e-7, atol=1e-7)
    print('Stored diagonal:', list(d.shape), 'RHS/solution:', list(targets.shape),
          'hypothetical dense matrix bytes:', args.samples**2 * x.element_size())
    if args.dense_reference:
        gram = tp.rbf_kernel(x, gamma=.2) + 2. * torch.eye(args.samples, device=x.device, dtype=x.dtype)
        matrix = d.sqrt()[:, None] * gram * d.sqrt()[None, :]
        expected = torch.linalg.solve(matrix, targets)
        for fit in fits:
            torch.testing.assert_close(fit.solution, expected, rtol=1e-7, atol=1e-7)
        print('Dense solution checks passed')


if __name__ == '__main__':
    main()
