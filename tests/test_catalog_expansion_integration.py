"""Option routing and aliases across the new public catalog."""
import pytest
import torch
import torchpairwise as tp


def test_expansion_aliases_neighbors_and_options(device):
    x = torch.tensor([[1., 2., 4.], [4., 1., 2.]], device=device)
    y = torch.tensor([[3., 2., 1.], [2., 4., 1.], [1., 3., 2.]], device=device)
    for fn in (tp.continuous_jaccard_distances, tp.soergel_distances):
        torch.testing.assert_close(fn(x, y), tp.ruzicka_distances(x, y))
    torch.testing.assert_close(tp.cdist(x, y, metric='continuous_jaccard'),
                               tp.ruzicka_distances(x, y))
    d = tp.renyi_divergences(x, y, alpha=2.)
    expected = torch.zeros_like(d, dtype=torch.bool).scatter_(-1, d.argmin(-1, keepdim=True), True)
    assert torch.equal(tp.k_neighbors_mask(x, y, metric='renyi', alpha=2.), expected)
    assert torch.equal(tp.radius_neighbors_mask(x, y, metric='renyi', alpha=2., epsilon=.3), d <= .3)
    for name in ('renyi', 'pearson_chi2', 'neyman_chi2', 'itakura_saito'):
        with pytest.raises(ValueError, match='symmetric'):
            tp.pdist(x, metric=name)
    with pytest.raises(TypeError, match='alpha'):
        tp.cdist(x, y, metric='total_variation', alpha=2.)
