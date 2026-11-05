"""Regularized dual kernel LDA with streamed scatters and a class-sized eigensolve.

This example uses coefficient-space regularization alpha*I in the within-class
matrix. It has no fitted-estimator API and does not claim that this equals an
alpha*I regularizer in the implicit RKHS. No [N,N] array is needed in the streamed
path; class blocks [N,C] and the small [C,C] eigensolve are retained.
"""
import argparse

import torch
import torchpairwise as tp


@torch.no_grad()
def kernel_lda(x, labels, *, gamma=.3, regularization=.2, chunk_size=64,
               components=None, rtol=1e-8, max_iter=500):
    """Return eigenvalues [k], dual directions [N,k], coordinates [N,k]."""
    if x.ndim != 2 or labels.ndim != 1 or labels.shape[0] != x.shape[0]:
        raise ValueError('Expected X:[N,D] and labels:[N]')
    if labels.dtype != torch.int64 or labels.device != x.device:
        raise ValueError('labels must be int64 on the X device')
    if regularization <= 0:
        raise ValueError('regularization must be positive')
    classes, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    c = classes.numel()
    if c < 2:
        raise ValueError('Kernel LDA needs at least two nonempty classes')
    k = min(c-1, x.shape[0]-1) if components is None else components
    if type(k) is not int or not 1 <= k <= min(c-1, x.shape[0]-1):
        raise ValueError('components must lie between 1 and min(C-1,N-1)')
    kernel = tp.lazy(x, metric='rbf', gamma=gamma, chunk_size=chunk_size)
    a = kernel
    p = tp.group_projection(labels, like=x)
    residual = a - p @ a                 # (I-P)K
    within = residual.mT @ residual
    within = within + regularization * tp.eye_like(within)
    # E diag(1/sqrt(n_c)) has shape [N,C]; center over samples to apply H.
    e = torch.nn.functional.one_hot(inverse, num_classes=c).to(x.dtype)
    normalized = e / counts.to(x.dtype).sqrt()
    centered = normalized - normalized.mean(0, keepdim=True)
    f = kernel.matmul(centered)                 # F=KH E diag(1/sqrt(n_c))
    fit = tp.linalg.solve(within, f, method='cg', rtol=rtol, max_iter=max_iter)
    if not fit.converged.all():
        raise RuntimeError(f'Within-class CG failed: {fit.status.tolist()}')
    z = fit.solution                            # Z=Sw^{-1}F, [N,C]
    small = f.T @ z                             # [C,C]
    eigenvalues, u = torch.linalg.eigh((small + small.T) * .5)
    eigenvalues = eigenvalues[-k:].flip(0)
    u = u[:, -k:].flip(1)
    cutoff = torch.finfo(x.dtype).eps * max(x.shape[0], c) * eigenvalues.abs().max()
    if (eigenvalues <= cutoff).any():
        raise RuntimeError('Requested more discriminant directions than the numerical between-class rank')
    directions = z @ (u / eigenvalues.sqrt())    # Sw-orthonormal, [N,k]
    coordinates = kernel.matmul(directions)     # training projections, [N,k]
    return eigenvalues, directions, coordinates, fit


@torch.no_grad()
def transform_kernel_lda(x_new, x_train, directions, *, gamma=.3, chunk_size=64):
    """Project [T,D] to [T,k] using fitted directions; no held-out labels needed."""
    return tp.lazy(x_new, x_train, metric='rbf', gamma=gamma,
                    chunk_size=chunk_size).matmul(directions)


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--samples-per-class', type=int, default=24)
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--dense-check', action='store_true')
    args = parser.parse_args(argv)
    generator = torch.Generator(device=args.device).manual_seed(29)
    means = torch.tensor([[-2., 0.], [2., 0.], [0., 2.]], device=args.device, dtype=torch.float64)
    x = means.repeat_interleave(args.samples_per_class, 0)
    x += .6 * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    labels = torch.tensor([-10, 42, 999], device=x.device).repeat_interleave(args.samples_per_class)
    gamma, alpha = .3, .2
    eigenvalues, directions, coordinates, fit = kernel_lda(
        x, labels, gamma=gamma, regularization=alpha, chunk_size=args.chunk_size)
    x_new = means.repeat_interleave(4, 0)
    x_new += .6 * torch.randn(x_new.shape, device=x.device, dtype=x.dtype, generator=generator)
    held_out = transform_kernel_lda(x_new, x, directions, gamma=gamma, chunk_size=args.chunk_size)
    print('Held-out coordinates:', list(held_out.shape))
    print('Dual directions:', list(directions.shape), 'training coordinates:', list(coordinates.shape))
    print('Small eigensolve:', [3, 3], 'eigenvalues:', eigenvalues.tolist())
    print('CG iterations:', fit.iterations.tolist(), 'residual:', fit.residual_norm.tolist())
    if args.dense_check:
        # Only this oracle allocates [N,N] matrices. Compare generalized eigen
        # residuals/Sw-orthogonality, avoiding eigenvector sign ambiguity.
        matrix = tp.rbf_kernel(x, gamma=gamma)
        same = labels[:, None] == labels[None, :]
        p = same.to(x.dtype) / same.sum(-1, keepdim=True)
        identity = torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
        h = identity - 1 / x.shape[0]
        sw = matrix @ (identity - p) @ matrix + alpha * identity
        sb = matrix @ (p - torch.ones_like(p)/x.shape[0]) @ matrix
        torch.testing.assert_close(directions.T @ sw @ directions, torch.eye(2, device=x.device, dtype=x.dtype), rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(sb @ directions, (sw @ directions) * eigenvalues, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(coordinates, matrix @ directions)
        torch.testing.assert_close(held_out, tp.rbf_kernel(x_new, x, gamma=gamma) @ directions)
        torch.testing.assert_close(transform_kernel_lda(x_new[:1], x, directions, gamma=gamma), held_out[:1])
        # Independent dense reduction to the same C-by-C eigenproblem.
        _, ids, counts = torch.unique(labels, return_inverse=True, return_counts=True)
        e = torch.nn.functional.one_hot(ids).to(x.dtype) / counts.to(x.dtype).sqrt()
        f = matrix @ h @ e
        small = f.T @ torch.linalg.solve(sw, f)
        expected = torch.linalg.eigvalsh((small + small.T)*.5)[-2:].flip(0)
        torch.testing.assert_close(eigenvalues, expected, rtol=2e-6, atol=2e-7)
        print('Dense generalized eigenproblem agrees')


if __name__ == '__main__':
    main()
