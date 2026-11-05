"""New histogram/weight options through the existing generic neighbor API."""
import pytest
import torch
import torchpairwise as tp


def test_probability_weighted_neighbor_forwarding(device):
    x = torch.tensor([[1., 2., 3.], [3., 2., 1.], [1., 3., 2.]], device=device)
    for metric, options in (
        ('kl', dict(base=2.)), ('symmetric_kl', dict(base=2.)),
        ('bhattacharyya', {}), ('aitchison', {}),
        ('cosine', dict(w=torch.tensor([.5, 1., 2.], device=device))),
        ('correlation', dict(w=torch.tensor([.5, 1., 2.], device=device))),
    ):
        matrix = tp.cdist(x, x, metric=metric, **options)
        expected = matrix <= .1
        expected.fill_diagonal_(False)
        torch.testing.assert_close(tp.radius_neighbors_mask(x, metric=metric, epsilon=.1, **options), expected)
        eligible = matrix.clone()
        eligible.fill_diagonal_(float('inf'))
        indices = eligible.argsort(dim=-1, stable=True)[..., :1]
        expected = torch.zeros_like(matrix, dtype=torch.bool).scatter_(-1, indices, True)
        torch.testing.assert_close(tp.k_neighbors_mask(x, metric=metric, **options), expected)
    with pytest.raises(ValueError, match='symmetric'):
        tp.pdist(x, metric='kl')
    with pytest.raises(TypeError, match='does not accept'):
        tp.cdist(x, x, metric='kl', w=torch.ones(3, device=device))
    clouds = x.unsqueeze(-2)
    for metric, options in (('energy', {}), ('mmd', dict(kernel='laplacian', length_scale=.7))):
        matrix = tp.cdist(clouds, clouds, metric=metric, **options)
        expected = matrix <= .1
        expected.fill_diagonal_(False)
        torch.testing.assert_close(tp.radius_neighbors_mask(clouds, metric=metric,
                                                            epsilon=.1, **options), expected)


def test_new_probability_fake_meta_contracts():
    from torch._subclasses.fake_tensor import FakeTensorMode
    for context in (FakeTensorMode(), torch.device('meta')):
        with context:
            x, y = torch.empty(2, 1, 3, 4), torch.empty(1, 5, 3, 4)
            for metric in ('kl', 'symmetric_kl', 'cosine', 'correlation'):
                kwargs = {'w': torch.ones(4)} if metric in ('cosine', 'correlation') else {}
                assert tp.cdist(x, y, metric=metric, **kwargs).shape == (2, 5, 3, 3)
                assert tp.paired_distances(x, y, metric=metric, **kwargs).shape == (2, 5, 3)
                if metric != 'kl':
                    assert tp.pdist(x, metric=metric, **kwargs).shape == (2, 1, 3)
