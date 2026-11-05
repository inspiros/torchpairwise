"""Shared dispatcher forwarding and compiled transport training contracts."""
import pytest
import torch
import torchpairwise as tp


def test_transport_neighbor_forwarding_and_option_isolation(device):
    x = torch.tensor([[[0., 0.]], [[1., 0.]], [[3., 0.]]], device=device)
    for metric, options in (
        ('wasserstein', {}),
        ('sliced_wasserstein', dict(projections=torch.eye(2, device=device))),
        ('sinkhorn', dict(regularization=.5, max_iter=3)),
    ):
        dense = tp.cdist(x, x, metric=metric, **options)
        radius = tp.radius_neighbors_mask(x, epsilon=.75, metric=metric, **options)
        expected = dense <= .75
        expected.fill_diagonal_(False)
        torch.testing.assert_close(radius, expected)
        masked = dense.clone()
        masked.fill_diagonal_(float('inf'))
        indices = masked.argsort(dim=-1, stable=True)[:, :1]
        expected = torch.zeros_like(dense, dtype=torch.bool).scatter_(1, indices, True)
        torch.testing.assert_close(tp.k_neighbors_mask(x, metric=metric, **options), expected)
    for options in (dict(projections=torch.eye(2, device=device)),
                    dict(regularization=.5), dict(max_iter=3)):
        with pytest.raises(TypeError, match='does not accept'):
            tp.cdist(x[:, 0], x[:, 0], metric='euclidean', **options)
    with pytest.raises(TypeError, match='projections'):
        tp.pdist(x, metric='sliced_wasserstein')


@pytest.mark.compiler
def test_sinkhorn_compiled_training_modes(device):
    torch.compiler.reset()
    x = torch.tensor([[[.1, -.2], [.7, .4]], [[.2, .3], [-.4, .1]]],
                     device=device, requires_grad=True)
    y = torch.tensor([[[-.3, .5], [.2, .9], [.8, -.1]],
                      [[.4, -.5], [.3, .8], [.1, .2]]], device=device, requires_grad=True)

    def call(a, b):
        options = dict(regularization=.6, max_iter=4)
        return (tp.sinkhorn_divergences(a, b, **options).sum()
                + tp.paired_distances(a, b, metric='sinkhorn', **options).sum()
                + tp.pdist(a, metric='sinkhorn', **options).sum())

    expected = call(x, y)
    actual = torch.compile(call, fullgraph=True)(x, y)
    torch.testing.assert_close(actual, expected)
    for gradient, reference in zip(torch.autograd.grad(actual, (x, y)),
                                    torch.autograd.grad(expected, (x, y))):
        torch.testing.assert_close(gradient, reference)
    torch.compiler.reset()
