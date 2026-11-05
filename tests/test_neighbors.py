import pytest
import torch
import torchpairwise as tp


@pytest.mark.parametrize('k', [0, 1, 2])
def test_exact_k_and_self_exclusion(k, device):
    x = torch.tensor([[0.], [1.], [3.]], device=device)
    result = tp.k_neighbors_mask(x, k=k)
    assert result.dtype == torch.bool
    assert not result.diagonal().any()
    assert (result.sum(-1) == k).all()


def test_cross_neighbors_and_stable_ties(device):
    x = torch.tensor([[1.], [2.]], device=device)
    y = torch.tensor([[0.], [2.], [4.]], device=device)
    expected = torch.tensor([[True, False, False], [False, True, False]], device=device)
    torch.testing.assert_close(tp.k_neighbors_mask(x, y, k=1), expected)
    torch.testing.assert_close(tp.k_neighbors_mask(x[None, None], y.expand(3, 1, -1, -1), k=1),
                               expected.expand(3, 1, -1, -1))


def test_radius_and_self_policy(device):
    x = torch.tensor([[0.], [1.], [3.]], device=device)
    result = tp.radius_neighbors_mask(x, epsilon=1.)
    expected = torch.tensor([[False, True, False], [True, False, False], [False, False, False]], device=device)
    torch.testing.assert_close(result, expected)
    assert tp.radius_neighbors_mask(x, x, epsilon=0.).diagonal().all()


def test_invalid_k(device):
    x = torch.ones(3, 2, device=device)
    for k in [-1, 3, 1.5, True]:
        with pytest.raises(ValueError):
            tp.k_neighbors_mask(x, k=k)


def test_native_neighbor_parity(device):
    x = torch.tensor([[[0.], [1.], [3.]]], device=device)
    torch.testing.assert_close(torch.ops.torchpairwise.k_neighbors_mask(x, k=1),
                               tp.k_neighbors_mask(x, k=1))
    torch.testing.assert_close(torch.ops.torchpairwise.radius_neighbors_mask(x, epsilon=1.),
                               tp.radius_neighbors_mask(x, epsilon=1.))
