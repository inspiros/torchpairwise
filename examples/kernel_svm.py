"""Binary C-SVM with a streamed kernel dual and an explicit free intercept.

The optimization enforces 0<=alpha<=C and y.T@alpha=0 for labels {-1,+1}.
No Gram matrix is retained unless --dense-reference is requested. This is a
small example, not a general fitted-estimator API or a dedicated SMO solver.
"""
import argparse

import torch
import torchpairwise as tp


def intercept_from_kkt(alpha, labels, raw_scores, C, tolerance):
    """Free supports determine b; otherwise choose the midpoint of its KKT interval."""
    lower, upper = alpha <= tolerance, alpha >= C - tolerance
    free = ~lower & ~upper
    residual = labels - raw_scores
    if free.any():
        return residual[free].mean()
    needs_lower = ((labels > 0) & lower) | ((labels < 0) & upper)
    needs_upper = ((labels > 0) & upper) | ((labels < 0) & lower)
    lo = residual[needs_lower].max() if needs_lower.any() else residual.new_tensor(-torch.inf)
    hi = residual[needs_upper].min() if needs_upper.any() else residual.new_tensor(torch.inf)
    if lo.isfinite() and hi.isfinite():
        return (lo + hi) * .5
    return lo if lo.isfinite() else hi if hi.isfinite() else residual.new_zeros(())


@torch.no_grad()
def fit_binary_svm(x, labels, *, C=1., gamma=.5, chunk_size=32,
                   rtol=1e-6, max_iter=1500):
    if x.ndim != 2 or labels.shape != x.shape[:1]:
        raise ValueError('Expected X:[N,D] and labels:[N]')
    if not ((labels == -1) | (labels == 1)).all() or labels.unique().numel() != 2:
        raise ValueError('Binary SVM needs both labels -1 and +1')
    if labels.dtype != x.dtype or labels.device != x.device or C <= 0:
        raise ValueError('Labels must match X dtype/device and C must be positive')
    kernel = tp.lazy(x, metric='rbf', gamma=gamma, chunk_size=chunk_size)
    Q = kernel.weight_rows(labels).weight_columns(labels)
    fit = tp.optimize.box_qp(Q, -torch.ones_like(labels), lower=0., upper=C,
                            equality=(labels, 0.), rtol=rtol, atol=1e-9, max_iter=max_iter)
    if not fit.converged:
        raise RuntimeError(f'SVM QP failed: status={fit.status.item()}, '
                           f'optimality={fit.optimality.item()}, feasibility={fit.feasibility.item()}')
    alpha = fit.solution
    signed = alpha * labels
    raw = kernel.matmul(signed)
    tolerance = max(1e-7, 10 * rtol * max(1., C))
    bias = intercept_from_kkt(alpha, labels, raw, C, tolerance)
    scores = raw + bias
    norm = signed @ raw
    primal = .5 * norm + C * (1 - labels * scores).clamp_min(0).sum()
    gap = primal + fit.objective  # fit.objective is the minimized negative dual.
    return alpha, bias, scores, gap, fit


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--samples-per-class', type=int, default=16)
    parser.add_argument('--C', type=float, default=1.)
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--max-iter', type=int, default=1500)
    parser.add_argument('--dense-reference', action='store_true')
    args = parser.parse_args(argv)
    generator = torch.Generator(device=args.device).manual_seed(53)
    labels = torch.tensor([-1., 1.], device=args.device, dtype=torch.float64).repeat_interleave(args.samples_per_class)
    x = .65 * torch.randn(labels.numel(), 2, device=args.device, dtype=labels.dtype, generator=generator)
    x[:, 0] += labels * 1.2
    gamma = .5
    alpha, bias, scores, gap, fit = fit_binary_svm(
        x, labels, C=args.C, gamma=gamma, chunk_size=args.chunk_size, max_iter=args.max_iter)
    print('Dual coefficients:', list(alpha.shape), 'intercept:', bias.item())
    print('Training accuracy:', ((scores >= 0) == (labels > 0)).double().mean().item())
    print('Iterations:', fit.iterations.item(), 'optimality:', fit.optimality.item(),
          'feasibility:', fit.feasibility.item(), 'primal-dual gap:', gap.item())
    if args.dense_reference:
        # Same dual problem with explicitly stored K and Q, plus independent
        # feasibility/KKT/primal-dual checks. PSD dual solutions can be nonunique.
        dense = tp.rbf_kernel(x, gamma=gamma)
        Q = labels[:, None] * dense * labels[None, :]
        oracle = tp.optimize.box_qp(Q, -torch.ones_like(labels), lower=0., upper=args.C,
                                   equality=(labels, 0.), rtol=1e-6, atol=1e-9, max_iter=args.max_iter)
        assert oracle.converged
        torch.testing.assert_close(fit.objective, oracle.objective, rtol=1e-5, atol=2e-6)
        reference_raw = dense @ (oracle.solution * labels)
        reference_bias = intercept_from_kkt(oracle.solution, labels, reference_raw, args.C,
                                            max(1e-7, 1e-5 * max(1., args.C)))
        torch.testing.assert_close(scores, reference_raw + reference_bias, rtol=2e-4, atol=2e-4)
        assert (alpha >= -1e-8).all() and (alpha <= args.C + 1e-8).all()
        assert abs(labels @ alpha) < 1e-7
        assert gap >= -1e-7 and gap < 2e-4 * max(1., abs(fit.objective.item()))
        print('Dense dual values/predictions and primal-dual checks agree')


if __name__ == '__main__':
    main()
