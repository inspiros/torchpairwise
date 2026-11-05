"""Native neighbor selection across catalog metrics and non-finite candidates."""
import pytest
import torch
import torchpairwise as tp


def _reference(distances, k, exclude):
    order = distances.argsort(dim=-1, stable=True)
    if exclude:
        diagonal = torch.arange(distances.shape[-2], device=distances.device).unsqueeze(-1)
        flags = order == diagonal
        order = order.gather(-1, flags.to(torch.uint8).argsort(dim=-1, stable=True))
    return torch.zeros_like(distances, dtype=torch.bool).scatter_(-1, order[..., :k], True)


@pytest.mark.parametrize('exclude', [False, True])
@pytest.mark.parametrize('k', [0, 1, 2])
def test_native_mask_stable_nonfinite_and_self_policy(k, exclude, device):
    distances = torch.tensor([[float('nan'), float('inf'), float('inf')],
                              [1., 1., float('nan')],
                              [float('nan'), float('-inf'), 0.]], device=device)
    original = distances.clone()
    actual = torch.ops.torchpairwise.neighbor_mask_from_distances(distances, 'k', k, None, exclude)
    torch.testing.assert_close(actual, _reference(distances, k, exclude))
    torch.testing.assert_close(distances, original, equal_nan=True)
    expected = distances <= float('inf')
    if exclude:
        expected.diagonal().fill_(False)
    torch.testing.assert_close(torch.ops.torchpairwise.neighbor_mask_from_distances(
        distances, 'radius', None, float('inf'), exclude), expected)
    assert not actual.requires_grad


def test_native_masks_cloud_catalog_and_explicit_cross_exclusion(device):
    x = torch.tensor([[[0.], [2.]], [[1.], [3.]], [[5.], [7.]]], device=device)
    y = x + 1
    distances = tp.chamfer_distances(x, y, squared=True, chunk_size=1)
    expected = _reference(distances, 1, True)
    actual = tp.k_neighbors_mask(x, y, k=1, metric='chamfer', squared=True,
                                exclude_self=True, chunk_size=1)
    torch.testing.assert_close(actual, expected)
    assert not actual.diagonal().any()
    with pytest.raises(ValueError, match='equal sample counts'):
        tp.k_neighbors_mask(x, y[:2], k=1, metric='chamfer', exclude_self=True)


@pytest.mark.parametrize('exclude', [False, True])
def test_native_neighbor_empty_outputs_and_invalid_options(exclude, device):
    distances = torch.empty(2, 0, 0, device=device)
    assert torch.ops.torchpairwise.neighbor_mask_from_distances(
        distances, 'k', 0, None, exclude).shape == (2, 0, 0)
    assert torch.ops.torchpairwise.neighbor_mask_from_distances(
        distances, 'radius', None, 0., exclude).shape == (2, 0, 0)
    with pytest.raises(RuntimeError, match='eligible'):
        torch.ops.torchpairwise.neighbor_mask_from_distances(distances, 'k', 1, None, exclude)
    with pytest.raises(RuntimeError, match='Unused'):
        torch.ops.torchpairwise.neighbor_mask_from_distances(distances, 'k', 0, 1., exclude)
    with pytest.raises(RuntimeError, match='epsilon'):
        torch.ops.torchpairwise.neighbor_mask_from_distances(distances, 'radius', None, float('nan'), exclude)


@pytest.mark.compiler
@pytest.mark.parametrize('selection', ['k', 'radius'])
def test_native_neighbor_compile_export_dynamic(selection, device, tmp_path):
    class Neighbors(torch.nn.Module):
        def forward(self, x, y):
            if selection == 'k':
                return tp.k_neighbors_mask(x, y, k=2, exclude_self=True, metric='angular')
            return tp.radius_neighbors_mask(x, y, epsilon=.7, exclude_self=True, metric='angular')

    module = Neighbors()
    shape = torch.export.Dim('samples', min=3)
    x, y = torch.rand(4, 2, device=device) + .2, torch.rand(4, 2, device=device) + .2
    exported = torch.export.export(module, (x, y), dynamic_shapes=({0: shape}, {0: shape}))
    path = tmp_path / f'neighbors_{selection}.pt2'
    torch.export.save(exported, path)
    restored = torch.export.load(path).module()
    x, y = torch.rand(5, 2, device=device) + .2, torch.rand(5, 2, device=device) + .2
    torch.testing.assert_close(restored(x, y), module(x, y))
    torch.compiler.reset()
    compiled = torch.compile(module, backend='aot_eager', fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(x, y), module(x, y))
    torch.compiler.reset()
