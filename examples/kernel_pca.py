"""Truncated kernel PCA using native streamed leading eigenpairs.

python examples/kernel_pca.py --device cuda --dense-reference

The streamed path never forms the Gram matrix. --dense-reference explicitly
adds a small dense oracle; omit it for large inputs. This unbatched no-grad
example assumes a symmetric PSD kernel and projects held-out samples using
means fitted once on the training Gram matrix.
"""
import argparse

import torch
import torchpairwise as tp


@torch.no_grad()
def truncated_kernel_pca(x, components, *, kernel='rbf', oversampling=8,
                         max_iter=100, rtol=1e-7, atol=1e-10, chunk_size=128,
                         seed=1729, **options):
    """Return eigenvalues [k], projection weights [N,k], scores [N,k], fit, mean [N].

    Gram eigenvalues / (N-1) give covariance eigenvalues. Coordinates are
    U*sqrt(lambda), without whitening. Retains O(N*r+r*r) solver blocks plus
    input/output/source tile workspace; r=min(k+oversampling,N-1). Every product
    still evaluates all pairs. Check convergence before using an embedding.
    """
    if x.ndim != 2 or x.dtype not in (torch.float32, torch.float64):
        raise ValueError('x must be a float32/float64 tensor [N,D]')
    n = x.shape[0]
    if type(components) is not int or not 1 <= components < n:
        raise ValueError('components must be an integer in [1,N-1]')
    if type(oversampling) is not int or oversampling < 0:
        raise ValueError('oversampling must be a nonnegative integer')
    width = min(components + oversampling, n - 1)
    source = tp.lazy(x, metric=kernel, chunk_size=chunk_size, **options)
    reference_mean = source.reduce('mean', dim=-2)  # column means, [N]
    operator = source.centered(reference_mean)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    initial = torch.randn(n, width, dtype=x.dtype, device=x.device, generator=generator)
    fit = tp.linalg.eigh(operator, k=components, initial=initial, max_iter=max_iter,
                        rtol=rtol, atol=atol)
    if not fit.converged.all():
        raise RuntimeError(
            f'Kernel PCA failed: status={fit.status.tolist()}, residual={fit.residual_norm.tolist()}')
    # Tiny roundoff-negative Ritz values have no variance. The solver itself
    # returns raw Ritz values; this embedding convention clips only the scores.
    cutoff = torch.finfo(x.dtype).eps * n * fit.eigenvalues.abs().max()
    positive = fit.eigenvalues > cutoff
    safe_values = torch.where(positive, fit.eigenvalues, torch.ones_like(fit.eigenvalues))
    inverse_root = torch.where(positive, safe_values.rsqrt(), torch.zeros_like(safe_values))
    weights = fit.eigenvectors * inverse_root
    coordinates = fit.eigenvectors * torch.where(positive, safe_values.sqrt(), torch.zeros_like(safe_values))
    return fit.eigenvalues, weights, coordinates, fit, reference_mean


@torch.no_grad()
def transform_kernel_pca(x_new, x_train, weights, reference_mean, *, kernel='rbf',
                         chunk_size=128, **options):
    """Project [T,D] to [T,k] with fitted training state and the same kernel."""
    cross = tp.lazy(x_new, x_train, metric=kernel, chunk_size=chunk_size, **options)
    return cross.centered(reference_mean).matmul(weights)


def dense_reference(x, components, *, kernel='rbf', **options):
    gram = tp.compare(x, metric=kernel, **options)  # [N,N], oracle only
    centered = gram - gram.mean(0, keepdim=True) - gram.mean(1, keepdim=True) + gram.mean()
    values, vectors = torch.linalg.eigh((centered + centered.T) * .5)
    return centered, values[-components:].flip(0), vectors[:, -components:].flip(1)


@torch.no_grad()
def check_linear_case(device):
    generator = torch.Generator(device=device).manual_seed(43)
    x = torch.randn(19, 3, dtype=torch.float64, device=device, generator=generator)
    values, weights, coordinates, fit, mean = truncated_kernel_pca(
        x, 2, kernel='linear', oversampling=1, max_iter=20, chunk_size=7, rtol=1e-9)
    vectors = fit.eigenvectors
    centered, expected, reference = dense_reference(x, 2, kernel='linear')
    torch.testing.assert_close(values, expected)
    torch.testing.assert_close((vectors.T @ reference).square().sum(), values.new_tensor(2.))
    torch.testing.assert_close(centered @ vectors, vectors * values)
    torch.testing.assert_close(coordinates.T @ coordinates, torch.diag(values))
    torch.testing.assert_close(transform_kernel_pca(x, x, weights, mean, kernel='linear'), coordinates)
    query = x[:5] + .3
    projected = transform_kernel_pca(query, x, weights, mean, kernel='linear', chunk_size=7)
    cross = tp.linear_kernel(query, x)
    centered_cross = cross - cross.mean(-1, keepdim=True) - mean + mean.mean()
    torch.testing.assert_close(projected, centered_cross @ weights)
    torch.testing.assert_close(transform_kernel_pca(query[:1], x, weights, mean, kernel='linear'), projected[:1])
    zeros, zero_weights, scores, zero_fit, zero_mean = truncated_kernel_pca(x * 0, 2, kernel='linear', chunk_size=7)
    torch.testing.assert_close(transform_kernel_pca(query, x * 0, zero_weights, zero_mean, kernel='linear'), torch.zeros_like(projected))
    assert fit.converged.all() and zero_fit.converged.all()
    torch.testing.assert_close(zeros, torch.zeros_like(zeros))
    torch.testing.assert_close(scores, torch.zeros_like(scores))
    print('Small linear and zero-rank dense checks passed on', device)


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--features', type=int, default=8)
    parser.add_argument('--components', type=int, default=4)
    parser.add_argument('--oversampling', type=int, default=8)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--rtol', type=float, default=1e-7)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--dense-reference', action='store_true')
    args = parser.parse_args(argv)
    if args.samples < 2 or args.features < 1:
        parser.error('samples must be at least two and features must be positive')
    generator = torch.Generator(device=args.device).manual_seed(7)
    x = torch.randn(args.samples, args.features, dtype=torch.float64,
                    device=args.device, generator=generator)
    options = dict(gamma=1. / args.features)
    values, weights, coordinates, fit, mean = truncated_kernel_pca(
        x, args.components, oversampling=args.oversampling, max_iter=args.max_iter,
        rtol=args.rtol, chunk_size=args.chunk_size, **options)
    vectors = fit.eigenvectors
    x_new = torch.randn(11, args.features, dtype=x.dtype, device=x.device, generator=generator)
    held_out = transform_kernel_pca(x_new, x, weights, mean, chunk_size=args.chunk_size, **options)
    print('Training coordinates:', list(coordinates.shape), 'held-out coordinates:', list(held_out.shape))
    print('Covariance eigenvalues:', (values / (args.samples - 1)).tolist())
    print('Native iterations:', fit.iterations.tolist(), 'true residuals:', fit.residual_norm.tolist())
    if args.dense_reference:
        cross = tp.rbf_kernel(x_new, x, **options)
        expected_cross = cross - cross.mean(-1, keepdim=True) - mean + mean.mean()
        torch.testing.assert_close(held_out, expected_cross @ weights)
        centered, expected, reference = dense_reference(x, args.components, **options)
        torch.testing.assert_close(centered @ vectors, vectors * values, rtol=2e-5, atol=2e-6)
        eigen_error = (values - expected).norm() / expected.norm().clamp_min(torch.finfo(x.dtype).tiny)
        overlap = (vectors.T @ reference).square().sum()
        subspace_error = ((args.components - overlap).clamp_min(0) / args.components).sqrt()
        print(f'Dense relative eigenvalue error: {eigen_error.item():.6g}')
        print(f'Dense subspace error: {subspace_error.item():.6g} (0 matches, 1 orthogonal)')
        torch.testing.assert_close(values, expected, rtol=1e-5, atol=1e-6)
        check_linear_case(args.device)


if __name__ == '__main__':
    main()
