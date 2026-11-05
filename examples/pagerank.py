"""Personalized PageRank on a streamed directed kernel-affinity graph.

python examples/pagerank.py --device cuda --dense-reference

Each vertex has distinct query/key features. W=K(query,key) can be asymmetric.
Normalize outgoing rows to P, then solve (I-beta P.T) ranks=(1-beta) v using
native restarted GMRES. Teleport columns are nonnegative and normalized. A small
uniform outgoing weight prevents dangling rows without materializing a matrix.
This is a dense affinity graph, not a sparse web-graph implementation.
"""
import argparse

import torch
import torchpairwise as tp


@torch.no_grad()
def pagerank(query, keys, personalization, *, beta=.85, gamma=.3, floor=.01,
             chunk_size=64, restart=12, max_iter=200, rtol=1e-8):
    """Return ranks [N,R], native diagnostics and transition source.

    A per-edge floor/N adds total floor outgoing mass to each vertex. Self-edges
    are included. Query/key features are two views of the same N vertices.
    """
    if (query.ndim != 2 or keys.shape != query.shape or query.shape[0] == 0
            or query.dtype not in (torch.float32, torch.float64)
            or keys.dtype != query.dtype or keys.device != query.device):
        raise ValueError('query/keys must be matching nonempty float32/64 [N,D] tensors')
    if not 0 <= beta < 1 or not 0 < floor < float('inf'):
        raise ValueError('beta must be in [0,1) and floor finite/positive')
    if (personalization.ndim != 2 or personalization.shape[0] != query.shape[0]
            or personalization.shape[-1] == 0 or personalization.dtype != query.dtype
            or personalization.device != query.device):
        raise ValueError('personalization must be [N,R] matching features')
    if not torch.isfinite(personalization).all() or (personalization < 0).any():
        raise ValueError('personalization must be finite and nonnegative')
    mass = personalization.sum(0, keepdim=True)
    if not mass.isfinite().all() or (mass <= 0).any():
        raise ValueError('each personalization column needs finite positive mass')
    v = personalization/mass
    # An explicit scalar Tensor preserves the requested precision in source
    # algebra, including when the input is float64.
    per_edge = query.new_tensor(floor/query.shape[0])
    affinity = tp.lazy(query, keys, gamma=gamma, chunk_size=chunk_size, metric='rbf') + per_edge
    row_mass = affinity.sum(dim=-1)                    # [N]
    transition = affinity.weight_rows(row_mass.reciprocal())
    operator = tp.eye_like(transition) - beta * transition.mT
    fit = tp.linalg.gmres(operator, (1-beta)*v, restart=restart, max_iter=max_iter, rtol=rtol)
    if not fit.converged.all():
        raise RuntimeError(f'PageRank failed: status={fit.status.tolist()}, '
                           f'residual={fit.residual_norm.tolist()}')
    # Do not clip or renormalize: these values solve the stated linear system.
    return fit.solution, fit, transition


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--samples', type=int, default=192)
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--restart', type=int, default=12)
    parser.add_argument('--dense-reference', action='store_true')
    args = parser.parse_args(argv)
    if args.samples < 2:
        parser.error('samples must be at least two')
    generator = torch.Generator(device=args.device).manual_seed(211)
    query = torch.randn(args.samples, 3, device=args.device, dtype=torch.float64, generator=generator)
    keys = torch.randn(query.shape, device=query.device, dtype=query.dtype, generator=generator)
    v = torch.ones(args.samples, 2, device=query.device, dtype=query.dtype)
    v[:, 1] = torch.sigmoid(query[:, 0])
    v = v/v.sum(0, keepdim=True)
    beta, gamma, floor = .85, .3, .01
    ranks, fit, transition = pagerank(query, keys, v, beta=beta, gamma=gamma, floor=floor,
                                     chunk_size=args.chunk_size, restart=args.restart, rtol=1e-10)
    print('Personalized ranks:', list(ranks.shape), 'mass:', ranks.sum(0).tolist())
    print('Arnoldi steps:', fit.iterations.tolist(), 'residual:', fit.residual_norm.tolist())
    stationarity = (1-beta)*v+beta*transition.T.matmul(ranks)
    torch.testing.assert_close(ranks, stationarity, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(ranks.sum(0), torch.ones_like(ranks.sum(0)), rtol=1e-8, atol=1e-10)
    assert (ranks >= -1e-10).all()
    if args.dense_reference:
        weights = tp.rbf_kernel(query, keys, gamma=gamma) + floor/args.samples
        p = weights/weights.sum(-1, keepdim=True)
        matrix = torch.eye(args.samples, device=query.device, dtype=query.dtype)-beta*p.T
        expected = torch.linalg.solve(matrix, (1-beta)*v)
        torch.testing.assert_close(ranks, expected, rtol=1e-8, atol=1e-10)
        print('Dense nonsymmetric solve and stationarity/mass checks agree')


if __name__ == '__main__':
    main()
