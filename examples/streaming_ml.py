"""Three ML consumers of logical matrices, with small dense correctness checks.

Run: python examples/streaming_ml.py --device cuda
Inputs are unbatched vectors [N,D], references [M,D], and targets/values [M,C].
These are teaching examples, not fitted estimators or optimized attention layers.
"""
import argparse

import torch
import torchpairwise as tp


def knn_regression(query, reference, targets, *, k=5, chunk_size=128):
    """Average the targets of the k nearest references; return [N,C].

    Selection ties use the lower reference index. Gradients flow to selected
    targets, not through the discrete neighbor indices to query/reference.
    """
    distances = tp.lazy(
        query, reference, metric="sqeuclidean", chunk_size=chunk_size)
    neighbors = distances.topk(k, largest=False).indices  # [N,k]
    return targets[neighbors].mean(dim=1)


def dense_knn_regression(query, reference, targets, *, k=5):
    distances = tp.sqeuclidean_distances(query, reference)  # [N,M]
    # Stable sorting reproduces the streamed selector's tie convention.
    neighbors = distances.argsort(dim=-1, stable=True)[:, :k]
    return targets[neighbors].mean(dim=1)


def _normalize(weighted):
    # RBF weights can all underflow to zero for a very distant query. This
    # example returns zero there and floors subnormal masses at dtype.tiny.
    return weighted[:, :-1] / weighted[:, -1:].clamp_min(
        torch.finfo(weighted.dtype).tiny)


def kernel_regression(query, reference, targets, *, gamma, chunk_size=128):
    """Nadaraya-Watson RBF smoothing: sum_j Kij*tj / sum_j Kij, [N,C]."""
    kernel = tp.lazy(
        query, reference, metric="rbf", gamma=gamma, chunk_size=chunk_size)
    # Carry the normalizer as one more value channel: one streamed product,
    # rather than separate products for the numerator and the row mass.
    values = torch.cat((targets, torch.ones_like(targets[:, :1])), dim=-1)
    return _normalize(kernel @ values)


def dense_kernel_regression(query, reference, targets, *, gamma):
    kernel = tp.rbf_kernel(query, reference, gamma=gamma)  # [N,M]
    values = torch.cat((targets, torch.ones_like(targets[:, :1])), dim=-1)
    return _normalize(kernel @ values)


def attention(query, keys, values, *, chunk_size=128):
    """Single-head scaled dot-product attention, without masks/dropout, [N,C]."""
    scores = tp.lazy(query, keys, metric="linear", chunk_size=chunk_size)
    return (scores * query.shape[-1] ** -0.5).softmax_matmul(values)


def dense_attention(query, keys, values):
    scores = (query @ keys.T) * query.shape[-1] ** -0.5  # [N,M]
    return scores.softmax(dim=-1) @ values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    generator = torch.Generator(device=args.device).manual_seed(1729)
    def tensor(shape):
        return torch.randn(shape, device=args.device, dtype=torch.float64,
                           generator=generator, requires_grad=True)
    query, reference, targets = tensor((11, 5)), tensor((17, 5)), tensor((17, 3))
    cases = (
        ("k-NN regression", knn_regression, dense_knn_regression, {"k": 3}),
        ("RBF regression", kernel_regression, dense_kernel_regression, {"gamma": .2}),
        ("Attention", attention, dense_attention, {}),
    )
    for name, streamed, dense, options in cases:
        actual = streamed(query, reference, targets, chunk_size=4, **options)
        expected = dense(query, reference, targets, **options)
        torch.testing.assert_close(actual, expected)
        # k-NN routing has no gradient to the query/reference coordinates.
        operands = (targets,) if name == "k-NN regression" else (query, reference, targets)
        gradients = torch.autograd.grad(actual.square().sum(), operands)
        oracle = torch.autograd.grad(expected.square().sum(), operands)
        torch.testing.assert_close(gradients, oracle)
        print(f"{name}: {list(actual.shape)}, values and first gradients match on {args.device}")
    # A separate tie example makes the deterministic neighbor convention visible.
    tied = query.new_zeros((3, 5))
    labels = query.new_tensor([[1.], [3.], [9.]])
    torch.testing.assert_close(knn_regression(tied[:1], tied, labels, k=2,
                                             chunk_size=1), labels.new_tensor([[2.]]))


if __name__ == "__main__":
    main()
