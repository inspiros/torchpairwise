"""Learn kernel bandwidth/noise with an implicit solve and native logdet gradients.

This is a small teaching example, not a fitted GP estimator. SLQ forward and
trace/solve backward are distinct approximations, selected explicitly. Fixed
probes are reused across steps. Dense references are optional and small only.
"""
import argparse
import math

import torch
import torchpairwise as tp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--num-probes', type=int, default=16)
    parser.add_argument('--krylov-size', type=int, default=24)
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--exact-logdet', action='store_true', help='Explicit dense Cholesky instead of SLQ')
    parser.add_argument('--dense-reference', action='store_true')
    args = parser.parse_args(argv)
    if min(args.samples, args.num_probes, args.krylov_size, args.chunk_size, args.steps) <= 0:
        parser.error('counts, budgets and chunk sizes must be positive')
    rng = torch.Generator(device=args.device).manual_seed(331)
    x = torch.randn(args.samples, 3, device=args.device, dtype=torch.float64, generator=rng)
    y = x[:, :1].sin()+.1*torch.randn(args.samples, 1, device=x.device, dtype=x.dtype, generator=rng)
    probes = torch.randint(0, 2, (args.samples, args.num_probes), device=x.device, generator=rng).to(x.dtype)*2-1
    log_length = torch.nn.Parameter(x.new_tensor(0.))
    log_noise = torch.nn.Parameter(x.new_tensor(-1.))
    params = (log_length, log_noise)
    optimizer = torch.optim.Adam(params, lr=.03)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        features = x/log_length.exp()
        a = tp.lazy(features, gamma=.5, chunk_size=args.chunk_size, metric='rbf')
        a = a + log_noise.exp() * tp.eye_like(a)
        fit = tp.linalg.solve(a, y, method='cg', differentiable=True, rtol=1e-10, max_iter=500)
        if args.exact_logdet:
            determinant = tp.linalg.logdet(a, method='exact', differentiable=True, block_size=4)
        else:
            determinant = tp.linalg.logdet(a, method='slq', probes=probes,
                max_iter=args.krylov_size, block_size=4, differentiable=True,
                gradient='solve', solve_rtol=1e-10, solve_max_iter=500)
        loss = .5*(y*fit.solution).sum()+.5*determinant.estimate+.5*args.samples*math.log(2*math.pi)
        gradients = torch.autograd.grad(loss, params)
        if args.dense_reference:
            dense = tp.rbf_kernel(x/log_length.exp(), gamma=.5)+log_noise.exp()*torch.eye(args.samples, device=x.device, dtype=x.dtype)
            dense_alpha = torch.linalg.solve(dense, y)
            dense_logdet = torch.linalg.slogdet(dense).logabsdet
            if args.exact_logdet:
                reference_loss = .5*(y*dense_alpha).sum()+.5*dense_logdet
                torch.testing.assert_close(determinant.estimate, dense_logdet)
            else:
                # Check the chosen stochastic gradient using the same probes.
                # This proxy's VALUE is not the logdet approximation.
                inverse_probes = torch.linalg.solve(dense, probes).detach()
                logdet_gradient_proxy = (inverse_probes*(dense@probes)).sum()/args.num_probes
                reference_loss = .5*(y*dense_alpha).sum()+.5*logdet_gradient_proxy
                print('  logdet absolute error:', (determinant.estimate-dense_logdet).abs().item())
            reference_gradients = torch.autograd.grad(reference_loss, params)
            torch.testing.assert_close(gradients, reference_gradients, rtol=1e-7, atol=1e-8)
            torch.testing.assert_close(fit.solution, dense_alpha, rtol=1e-7, atol=1e-8)
        for param, grad in zip(params, gradients):
            if not torch.isfinite(grad).all():
                raise RuntimeError('Nonfinite likelihood gradient')
            param.grad = grad
        print(f'step={step} objective={loss.item():.6f} sampling_SE={determinant.standard_error.item():.4g} '
              f'length={log_length.exp().item():.4f} noise={log_noise.exp().item():.4f} '
              f'grad={[g.item() for g in gradients]}')
        optimizer.step()
    print('Only estimate receives logdet gradients; diagnostics/probes are fixed.')
    if not args.exact_logdet:
        print('SLQ values and trace/solve gradients have different finite-budget errors.')


if __name__ == '__main__':
    main()
