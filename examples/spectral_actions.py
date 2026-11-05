"""Kernel log-determinants, graph diffusion and whitening without dense functions.

The optional small-problem dense oracle is measured separately. Timings are
illustrative; allocated CUDA peaks exclude reserved cache and vendor workspaces.
"""
import argparse
import gc
import time

import torch
import torchpairwise as tp


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--probes', type=int, default=16)
    parser.add_argument('--krylov-size', type=int, default=32)
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--block-size', type=int, default=4)
    parser.add_argument('--dense-reference', action='store_true')
    args = parser.parse_args(argv)
    if any(v <= 0 for v in (args.samples, args.probes, args.krylov_size, args.chunk_size, args.block_size)):
        parser.error('sample/probe counts and chunk/block/Krylov sizes must be positive')
    generator = torch.Generator(device=args.device).manual_seed(81)
    x = torch.randn(args.samples, 4, device=args.device, dtype=torch.float64, generator=generator)
    values = torch.randn(args.samples, 2, device=x.device, dtype=x.dtype, generator=generator)
    kernel = tp.lazy(x, gamma=.2, chunk_size=args.chunk_size, metric='rbf')
    a = kernel + .5 * tp.eye_like(kernel)
    # A symmetric complete weighted graph, including self weights in K. Those
    # cancel in D-K. The constant signal must be invariant under diffusion.
    degree = kernel.reduce('sum', dim=-1)
    laplacian = tp.diag_embed(degree) - kernel
    options = dict(max_iter=args.krylov_size, block_size=args.block_size)

    def measure(label, operation):
        operation()  # Warm before the measured call; discard all returned state.
        gc.collect()
        if x.is_cuda:
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = operation()
        if x.is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = f', extra allocated CUDA bytes={torch.cuda.max_memory_allocated()-baseline}' if x.is_cuda else ''
        print(f'{label}: seconds={elapsed:.4f}{peak}')
        return result

    logdet = measure('Streamed SLQ logdet', lambda: tp.linalg.logdet(
        a, method='slq', num_probes=args.probes, seed=19, **options))
    diffusion = measure('Streamed graph diffusion', lambda: tp.linalg.expm_multiply(
        laplacian, values, scale=-.1, **options))
    whitening = measure('Streamed inverse-root action', lambda: tp.linalg.inv_sqrt_multiply(a, values, **options))
    invariant = tp.linalg.expm_multiply(laplacian, torch.ones_like(values[:, 0]), scale=-.1, **options)
    torch.testing.assert_close(invariant.action, torch.ones_like(invariant.action), rtol=1e-8, atol=1e-9)
    print('SLQ logdet:', logdet.estimate.item(), 'sampling SE:', logdet.standard_error.item())
    print('Shapes: X', list(x.shape), 'signals/actions', list(diffusion.action.shape))
    print('Diffusion heuristic errors:', diffusion.error_estimate.tolist())
    print('Whitening heuristic errors:', whitening.error_estimate.tolist())
    print('A hypothetical dense comparison matrix uses', args.samples**2*x.element_size(), 'bytes')
    print('Sampling SE omits quadrature bias; compare larger --krylov-size separately.')
    if args.dense_reference:
        def dense_logdet():
            k = tp.rbf_kernel(x, gamma=.2) + .5*torch.eye(args.samples, device=x.device, dtype=x.dtype)
            return torch.linalg.slogdet(k).logabsdet
        exact = measure('Dense direct logdet (construction included)', dense_logdet)
        def dense_diffusion():
            k = tp.rbf_kernel(x, gamma=.2)
            l = torch.diag(k.sum(-1)) - k
            return torch.matrix_exp(-.1*l) @ values
        reference = measure('Dense diffusion (construction included)', dense_diffusion)
        def dense_whitening():
            k = tp.rbf_kernel(x, gamma=.2) + .5*torch.eye(args.samples, device=x.device, dtype=x.dtype)
            d, u = torch.linalg.eigh(k)
            return u @ (d.rsqrt()[:, None]*(u.T @ values))
        whitened = measure('Dense whitening (construction included)', dense_whitening)
        print('SLQ absolute error:', (logdet.estimate-exact).abs().item())
        print('Diffusion relative error:', (diffusion.action-reference).norm().div(reference.norm()).item())
        print('Whitening relative error:', (whitening.action-whitened).norm().div(whitened.norm()).item())
        # No forced accuracy threshold: these are finite-budget approximations.
        assert torch.isfinite(diffusion.action).all() and torch.isfinite(whitening.action).all()


if __name__ == '__main__':
    main()
