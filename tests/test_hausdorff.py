import numpy as np
import pytest
import scipy.spatial.distance as sd
import torch
import torchpairwise as tp


def sets(device):
    g = torch.Generator().manual_seed(72)
    x = torch.rand(2, 3, 2, generator=g, dtype=torch.float64).to(device)
    y = (torch.rand(3, 4, 2, generator=g, dtype=torch.float64) + .4).to(device)
    return x, y


@pytest.mark.parametrize('shuffle', [False, True])
def test_hausdorff_values_witnesses_and_broadcast(shuffle, device):
    x, y = sets(device)
    generator = torch.Generator(device=device).manual_seed(19)
    values, xi, yi = tp.directed_hausdorff_distances(x, y, shuffle=shuffle, generator=generator)
    expected = np.array([[sd.directed_hausdorff(a.cpu(), b.cpu())[0] for b in y] for a in x])
    torch.testing.assert_close(values, torch.as_tensor(expected, device=device))
    assert xi.dtype == yi.dtype == torch.int64
    for i in range(len(x)):
        for j in range(len(y)):
            torch.testing.assert_close((x[i, xi[i, j]] - y[j, yi[i, j]]).norm(), values[i, j])
    result = tp.directed_hausdorff_distances(x.expand(2, 1, *x.shape), y.expand(1, 3, *y.shape))
    torch.testing.assert_close(result[0], values.expand(2, 3, -1, -1))


def test_hausdorff_gradients_and_rng(device):
    x, y = sets(device)
    x.requires_grad_()
    y.requires_grad_()
    generator = torch.Generator(device=device).manual_seed(92)
    fn = lambda a, b: tp.directed_hausdorff_distances(a, b, shuffle=True, generator=generator)[0]
    assert torch.autograd.gradcheck(fn, (x, y), fast_mode=True)
    value = fn(x, y)
    after_forward = generator.get_state().clone()
    torch.autograd.grad(value.sum(), (x, y))
    assert torch.equal(generator.get_state(), after_forward)


def test_hausdorff_identical_sets_zero_gradient(device):
    x, _ = sets(device)
    x.requires_grad_()
    values, _, _ = tp.directed_hausdorff_distances(x)
    assert values.diagonal().eq(0).all()
    gradient, = torch.autograd.grad(values.diagonal().sum(), (x,))
    torch.testing.assert_close(gradient, torch.zeros_like(x))


def test_hausdorff_empty_samples_and_point_sets(device):
    x, y = sets(device)
    for value in tp.directed_hausdorff_distances(x[:0], y):
        assert value.shape == (0, len(y))
    with pytest.raises(ValueError, match='point sets'):
        tp.directed_hausdorff_distances(x[:, :0], y)


def test_hausdorff_broadcast_gradients(device):
    x, y = sets(device)
    x = x.expand(2, 1, *x.shape).clone().requires_grad_()
    y = y.expand(1, 2, *y.shape).clone().requires_grad_()
    fn = lambda a, b: tp.directed_hausdorff_distances(a, b)[0]
    assert torch.autograd.gradcheck(fn, (x, y), fast_mode=True)


@pytest.mark.parametrize('batched,feature_dim', [(False, -1), (False, -3), (True, 2)])
def test_hausdorff_spatial_features_values_witnesses_and_gradients(batched, feature_dim, device):
    torch.manual_seed(48)
    x = torch.rand(2, 2, 3, 4, dtype=torch.float64, device=device)
    y = torch.rand(3, 3, 3, 4, dtype=torch.float64, device=device) + .3
    if batched:
        x = x.expand(2, *x.shape).clone()
        y = y.unsqueeze(0)  # Broadcast one batch to two.
    x = x.movedim(-1, feature_dim).contiguous().requires_grad_()
    y = y.movedim(-1, feature_dim).contiguous().requires_grad_()
    sample_dim = int(batched)
    xf = x.movedim(feature_dim, -1).flatten(sample_dim + 1, -2)
    yf = y.movedim(feature_dim, -1).flatten(sample_dim + 1, -2)
    # Independent exhaustive ground distances, including every point pair.
    ground = torch.cdist(xf.unsqueeze(-3), yf.unsqueeze(-4))
    expected = ground.amin(-1).amax(-1)
    values, xi, yi = tp.directed_hausdorff_distances(
        x, y, batched=batched, feature_dim=feature_dim)
    torch.testing.assert_close(values, expected)
    torch.testing.assert_close(ground.gather(-1, yi[..., None, None].expand(*yi.shape, xf.shape[-2], 1))
                               .squeeze(-1).gather(-1, xi[..., None]).squeeze(-1), values)
    for actual, reference in zip(torch.autograd.grad(values.sum(), (x, y)),
                                 torch.autograd.grad(expected.sum(), (x, y))):
        torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(tp.cdist(x, y, metric='directed_hausdorff',
                                       batched=batched, feature_dim=feature_dim), values)
    torch.testing.assert_close(tp.radius_neighbors_mask(
        x, y, epsilon=.7, metric='directed_hausdorff', batched=batched,
        feature_dim=feature_dim), expected <= .7)
    mask = tp.k_neighbors_mask(x, y, k=1, metric='directed_hausdorff',
                              batched=batched, feature_dim=feature_dim)
    assert mask.sum(-1).eq(1).all()
    torch.testing.assert_close(mask.to(torch.int64).argmax(-1), expected.argmin(-1))


def test_hausdorff_spatial_self_empty_and_layout_errors(device):
    x = torch.rand(2, 3, 4, 5, device=device)
    assert tp.directed_hausdorff_distances(x)[0].shape == (2, 3, 3)
    values = tp.directed_hausdorff_distances(x, batched=False)[0]
    assert values.shape == (2, 2)
    assert values.diagonal().eq(0).all()
    assert tp.directed_hausdorff_distances(x[:0], x, batched=False)[0].shape == (0, 2)
    xb = x.unsqueeze(0)
    assert tp.directed_hausdorff_distances(xb[:0], xb, batched=True)[0].shape == (0, 2, 2)
    with pytest.raises(ValueError, match='point sets'):
        tp.directed_hausdorff_distances(x[:, :0], batched=False)
    for kwargs, error, message in [
        ({'batched': 1}, TypeError, 'batched'),
        ({'feature_dim': 1.5}, TypeError, 'integer'),
        ({'feature_dim': 4}, ValueError, 'out of range'),
        ({'batched': True, 'feature_dim': 1}, ValueError, 'sample axes'),
    ]:
        with pytest.raises(error, match=message):
            tp.directed_hausdorff_distances(x, **kwargs)
    with pytest.raises(TypeError, match='does not accept'):
        tp.cdist(x, x, metric='euclidean', feature_dim=2)
