"""Learn an RBF feature scale through a native streaming kernel-ridge solve.

python examples/differentiable_kernel_ridge.py --device cuda --dense-check

Small training/validation blocks stay dense; the streamed path stores neither
Gram/cross matrices nor an iteration trace. Both solves must converge. The
optional dense oracle uses the same inputs and compares the objective/gradient.
Pass --landmarks 8 for a damped rectangular least-squares fit instead of ridge.
The square case demonstrates the preconditioner option; its constant Jacobi
scaling does not itself reduce the RBF system condition number.
"""
import argparse

import torch
import torchpairwise as tp


def objective(x, y, targets, validation, log_scale, *, dense=False, landmarks=None):
    train, query = x * log_scale.exp(), y * log_scale.exp()
    references = train if landmarks is None else landmarks * log_scale.exp()
    if dense:
        matrix = tp.rbf_kernel(train, references, gamma=.3)
        if landmarks is None:
            weights = torch.linalg.solve(matrix + .4 * torch.eye(x.shape[0], device=x.device, dtype=x.dtype), targets)
        else:
            augmented = torch.cat((matrix, .4 * torch.eye(references.shape[0], device=x.device, dtype=x.dtype)), 0)
            rhs = torch.cat((targets, targets.new_zeros(references.shape[0], targets.shape[-1])), 0)
            q, r = torch.linalg.qr(augmented, mode='reduced')
            weights = torch.linalg.solve_triangular(r, q.T @ rhs, upper=True)
        prediction = tp.rbf_kernel(query, references, gamma=.3) @ weights
    else:
        source = tp.lazy(train, references, gamma=.3, chunk_size=7, metric='rbf')
        if landmarks is None:
            a = source + .4 * tp.eye_like(source)
            inverse = tp.diag_embed(train.new_full((train.shape[0],), 1 / 1.4))
            fit = tp.linalg.solve(a, targets, method='cg', differentiable=True,
                                 preconditioner=inverse, rtol=1e-10, max_iter=200)
        else:
            fit = tp.linalg.lstsq(source, targets, damp=.4, differentiable=True,
                                 rtol=1e-10, max_iter=200)
        prediction = tp.lazy(query, references, gamma=.3, chunk_size=7, metric='rbf').matmul(fit.solution)
    return (prediction - validation).square().mean()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--dense-check', action='store_true')
    parser.add_argument('--landmarks', type=int, default=0, help='use a damped rectangular dictionary of this size; 0 uses square ridge')
    args = parser.parse_args(argv)
    if args.steps < 1 or not 0 <= args.landmarks <= 32:
        parser.error('steps must be positive and landmarks must lie in [0,32]')
    generator = torch.Generator(device=args.device).manual_seed(81)
    x = torch.randn(32, 3, device=args.device, dtype=torch.float64, generator=generator)
    y = torch.randn(12, 3, device=args.device, dtype=torch.float64, generator=generator)
    targets, validation = x[:, :1].sin(), y[:, :1].sin()
    log_scale = torch.tensor(0., device=args.device, dtype=x.dtype, requires_grad=True)
    landmarks = x[:args.landmarks] if args.landmarks else None
    for step in range(args.steps):
        loss = objective(x, y, targets, validation, log_scale, landmarks=landmarks)
        gradient, = torch.autograd.grad(loss, log_scale)
        if args.dense_check:
            dense = objective(x, y, targets, validation, log_scale, dense=True, landmarks=landmarks)
            expected, = torch.autograd.grad(dense, log_scale)
            torch.testing.assert_close(loss, dense, rtol=1e-8, atol=1e-10)
            torch.testing.assert_close(gradient, expected, rtol=1e-7, atol=1e-9)
        print(f'Step {step}: validation MSE={loss.item():.6g}, scale gradient={gradient.item():.6g}')
        with torch.no_grad():
            log_scale.sub_(.1 * gradient)
    print('Training:[32,3], validation:[12,3], targets/predictions:[N,1]')


if __name__ == '__main__':
    main()
