"""Landmark-kernel feature regression with native rectangular LSQR.

python examples/landmark_regression.py --device cuda --dense-reference

Fit ||K(X,Z) W-Y||^2 + damp^2 ||W||^2, then predict K(X_test,Z) W.
Landmarks Z give L explicit kernel features, so coefficients are [L,R]. This is
Euclidean coefficient ridge on a kernel dictionary, not exact kernel ridge or
Nyström-whitened features. No [N,L] design or [L,L] normal matrix is stored in
the streamed path. Dense references are only for small correctness checks.
"""
import argparse

import torch
import torchpairwise as tp


@torch.no_grad()
def fit_landmark_regression(x, landmarks, targets, *, gamma=.3, damp=.2,
                            chunk_size=64, rtol=1e-8, max_iter=300):
    """Return coefficients [L,R] and LSQR diagnostics; no fitted intercept."""
    design = tp.lazy(x, landmarks, gamma=gamma, chunk_size=chunk_size, metric='rbf')
    fit = tp.linalg.lstsq(design, targets, damp=damp, rtol=rtol, max_iter=max_iter)
    if not fit.converged.all():
        raise RuntimeError(f'LSQR failed: status={fit.status.tolist()}, '
                           f'normal_residual={fit.normal_residual_norm.tolist()}')
    return fit.solution, fit


@torch.no_grad()
def predict(x, landmarks, coefficients, *, gamma=.3, chunk_size=64):
    """Map held-out X:[T,D] to predictions [T,R] using the same landmarks."""
    return tp.lazy(x, landmarks, gamma=gamma, chunk_size=chunk_size, metric='rbf').matmul(coefficients)


def targets(x):
    return x[:, :1].sin() + .2*x[:, 1:2].square()


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--samples', type=int, default=192)
    parser.add_argument('--landmarks', type=int, default=24)
    parser.add_argument('--test-samples', type=int, default=48)
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--dense-reference', action='store_true')
    args = parser.parse_args(argv)
    if min(args.samples, args.landmarks, args.test_samples) < 1 or args.landmarks > args.samples:
        parser.error('positive sample counts are required and landmarks cannot exceed training samples')
    generator = torch.Generator(device=args.device).manual_seed(137)
    x = torch.randn(args.samples, 3, dtype=torch.float64, device=args.device, generator=generator)
    test = torch.randn(args.test_samples, 3, dtype=x.dtype, device=x.device, generator=generator)
    # This teaching example uses the first L training samples as landmarks.
    # Landmark selection is a separate modeling choice.
    landmarks, y = x[:args.landmarks].clone(), targets(x)
    gamma, damp = .3, .2
    coefficients, fit = fit_landmark_regression(x, landmarks, y, gamma=gamma, damp=damp,
                                               chunk_size=args.chunk_size, rtol=1e-10)
    scores = predict(test, landmarks, coefficients, gamma=gamma, chunk_size=args.chunk_size)
    print('Coefficients:', list(coefficients.shape), 'held-out predictions:', list(scores.shape))
    print('Iterations:', fit.iterations.tolist(), 'data residual:', fit.residual_norm.tolist(),
          'normal residual:', fit.normal_residual_norm.tolist())
    print('Held-out RMSE:', (scores-targets(test)).square().mean().sqrt().item())
    if args.dense_reference:
        design = tp.rbf_kernel(x, landmarks, gamma=gamma)  # [N,L], oracle only
        augmented = torch.cat((design, damp*torch.eye(args.landmarks, dtype=x.dtype, device=x.device)))
        rhs = torch.cat((y, y.new_zeros(args.landmarks, y.shape[-1])))
        expected = torch.linalg.lstsq(augmented, rhs).solution
        torch.testing.assert_close(coefficients, expected, rtol=2e-6, atol=2e-7)
        dense_scores = tp.rbf_kernel(test, landmarks, gamma=gamma) @ expected
        torch.testing.assert_close(scores, dense_scores, rtol=2e-6, atol=2e-7)
        objective = (design@coefficients-y).square().sum()+damp**2*coefficients.square().sum()
        reference = (design@expected-y).square().sum()+damp**2*expected.square().sum()
        torch.testing.assert_close(objective, reference)
        print('Dense augmented least-square coefficients, objective and held-out predictions agree')


if __name__ == '__main__':
    main()
