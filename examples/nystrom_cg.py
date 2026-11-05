"""Fit exact streamed kernel ridge using a reusable Nyström CG preconditioner.

The approximation is only an inverse action inside CG. The system and held-out
prediction still use the full RBF kernel. Compare CG, scalar Jacobi-PCG and
Nyström-PCG on the same inputs/tolerances; report setup separately from solves.
Small optional dense references are allocated only after streamed measurements.
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
    parser.add_argument('--landmarks', type=int, default=16)
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--dense-reference', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.landmarks <= args.samples:
        parser.error('require 1 <= landmarks <= samples')
    generator = torch.Generator(device=args.device).manual_seed(119)
    x = .4 * torch.randn(args.samples, 2, device=args.device, dtype=torch.float64, generator=generator)
    x_test = .4 * torch.randn(32, 2, device=x.device, dtype=x.dtype, generator=generator)
    def target(values):
        return torch.cat((values[:, :1].sin(), values[:, 1:2].cos()), -1)
    rhs = target(x)
    shift, gamma, jitter = .03, .2, 1e-8
    K = tp.lazy(x, gamma=gamma, chunk_size=args.chunk_size, metric='rbf')
    A = K + shift * tp.eye_like(K)
    rows = torch.arange(args.samples, device=x.device)
    ids = torch.randperm(args.samples, device=x.device, generator=generator)[:args.landmarks]

    def sync():
        if x.is_cuda:
            torch.cuda.synchronize(x.device)

    # Warm the source path before measuring setup/solves.
    A.matmul(rhs)
    sync()
    if x.is_cuda:
        baseline = torch.cuda.memory_allocated(x.device)
        torch.cuda.reset_peak_memory_stats(x.device)
    start = time.perf_counter()
    cross, landmarks = K.block(rows, ids), K.block(ids, ids)
    M = tp.linalg.nystrom_inverse(cross, landmarks, shift, jitter=jitter)
    sync()
    setup_seconds = time.perf_counter() - start
    memory = f', peak extra CUDA bytes={torch.cuda.max_memory_allocated(x.device)-baseline}' if x.is_cuda else ''
    print(f'Nyström setup: seconds={setup_seconds:.4f}{memory}')
    rank = min(args.samples, args.landmarks)
    print('Setup blocks:', list(cross.shape), list(landmarks.shape),
          'prepared state bytes:', 8 * (args.samples * rank + rank) + x.element_size())
    del cross, landmarks
    jacobi = tp.linalg.jacobi(x.new_full((args.samples,), 1. + shift))
    fits = []
    for name, inverse in (('CG', None), ('Jacobi-PCG', jacobi), ('Nyström-PCG', M)):
        tp.linalg.cg(A, rhs, preconditioner=inverse, max_iter=1)
        sync()
        if x.is_cuda:
            baseline = torch.cuda.memory_allocated(x.device)
            torch.cuda.reset_peak_memory_stats(x.device)
        start = time.perf_counter()
        fit = tp.linalg.cg(A, rhs, preconditioner=inverse, rtol=1e-9, max_iter=500)
        sync()
        seconds = time.perf_counter() - start
        memory = f', peak extra CUDA bytes={torch.cuda.max_memory_allocated(x.device)-baseline}' if x.is_cuda else ''
        print(f'{name}: steps={fit.iterations.tolist()}, solve seconds={seconds:.4f}, '
              f'residual={fit.residual_norm.tolist()}{memory}')
        if inverse is M:
            print(f'Nyström setup + one solve seconds={setup_seconds+seconds:.4f}')
        if not fit.converged.all():
            raise RuntimeError(f'{name} status={fit.status.tolist()}')
        fits.append(fit)
    test_kernel = tp.lazy(x_test, x, gamma=gamma, chunk_size=args.chunk_size, metric='rbf')
    scores = test_kernel.matmul(fits[-1].solution)
    print('Held-out shape:', list(scores.shape), 'RMSE:', (scores-target(x_test)).square().mean().sqrt().item())
    if args.dense_reference:
        matrix = tp.rbf_kernel(x, gamma=gamma) + shift * torch.eye(args.samples, device=x.device, dtype=x.dtype)
        reference = torch.linalg.solve(matrix, rhs)
        predictions = tp.rbf_kernel(x_test, x, gamma=gamma) @ reference
        for fit in fits:
            torch.testing.assert_close(fit.solution, reference, rtol=1e-7, atol=1e-7)
        torch.testing.assert_close(scores, predictions, rtol=1e-7, atol=1e-7)
        print('Dense coefficient and held-out checks passed')


if __name__ == '__main__':
    main()
