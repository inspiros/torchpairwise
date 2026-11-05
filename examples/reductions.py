"""Neighbor matching and kernel regression without a dense pairwise matrix."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(17, 5, device=device, requires_grad=True)
    y = torch.randn(23, 5, device=device, requires_grad=True)
    values = torch.randn(23, 2, device=device, requires_grad=True)
    match = tp.lazy(x, y, chunk_size=8).topk(3, largest=False)
    # One extra channel obtains the normalization mass in the same traversal.
    augmented = torch.cat((values, torch.ones_like(values[:, :1])), -1)
    total = tp.lazy(x, y, metric='rbf', gamma=.2, chunk_size=8) @ augmented
    prediction = total[:, :-1] / total[:, -1:].clamp_min(1e-12)
    (match.values.mean() + prediction.square().mean()).backward()
    assert match.indices.shape == (17, 3)
    assert all(torch.isfinite(t.grad).all() for t in (x, y, values))
    farthest = tp.lazy(x, y, chunk_size=8).topk(2, largest=True)
    maximum = tp.lazy(x, y, chunk_size=8).amax(-1)
    torch.testing.assert_close(farthest.values[:, 0], maximum)
    product = tp.lazy(x, y, metric='rbf', chunk_size=8).prod(-1)
    assert product.shape == (17,)
    flags = torch.tensor([[True, False], [True, True]], device=device)
    any_close = tp.lazy(flags, flags, metric='jaccard', chunk_size=1).any(-1, predicate=lambda d: d < .1)
    assert any_close.all()
    print('Neighbor matching and normalized kernel regression passed on', device)


if __name__ == '__main__':
    main()
