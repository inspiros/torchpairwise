"""Greedy streamed rows, stable ties and device-local seeding."""
import pytest
import torch
import torchpairwise as tp


def _fps(d, k, start):
    selected, distances = [], []
    nearest = d.new_full((d.size(0),), float('inf'))
    p = start
    for j in range(k):
        selected.append(p)
        distances.append(d.new_full((), float('inf')) if j == 0 else nearest[p].clone())
        nearest = torch.minimum(nearest, d[p])
        scores = nearest.clone()
        scores[selected] = -float('inf')
        p = int(scores.argmax())
    return torch.tensor(selected, device=d.device), torch.stack(distances)


def test_fps_reference_batches_and_duplicate_ties(device):
    x = torch.tensor([[[0.], [1.], [3.], [7.]], [[0.], [0.], [0.], [0.]]], dtype=torch.float64, device=device)
    d = (x - x.transpose(-1, -2)).square()
    indices, distances = tp.neighbors.farthest_point_sampling(tp.as_lazy(d), 4, start=1)
    for b in range(2):
        ids, values = _fps(d[b], 4, 1)
        torch.testing.assert_close(indices[b], ids)
        torch.testing.assert_close(distances[b], values)
    lazy = tp.lazy(x, metric='sqeuclidean', chunk_size=2)
    streamed_indices, streamed_distances = tp.neighbors.farthest_point_sampling(lazy, 4, start=1)
    torch.testing.assert_close(streamed_indices, indices)
    torch.testing.assert_close(streamed_distances, distances)
    assert indices[1].tolist() == [1, 0, 2, 3]


def test_kmeans_device_generator_and_zero_probability_fallback(device):
    d = torch.zeros(6, 6, device=device, dtype=torch.float64)
    generator = torch.Generator(device=device).manual_seed(83)
    first = tp.neighbors.kmeans_plusplus(tp.as_lazy(d), 6, generator=generator)
    generator.manual_seed(83)
    second = tp.neighbors.kmeans_plusplus(tp.as_lazy(d), 6, generator=generator)
    torch.testing.assert_close(first, second)
    assert first[1:].tolist() == [i for i in range(6) if i != int(first[0])]
    # Only the opposite cluster has nonzero D² probability after either pick.
    d = torch.tensor([[0., 0., 9.], [0., 0., 9.], [9., 9., 0.]], device=device, dtype=torch.float64)
    ids = tp.neighbors.kmeans_plusplus(tp.as_lazy(d), 2, generator=generator)
    assert (int(ids[0]) == 2) != (int(ids[1]) == 2)


def test_selection_validation(device):
    d = tp.as_lazy(torch.zeros(4, 4, dtype=torch.float64, device=device))
    for k in (-1, 5):
        with pytest.raises((ValueError, RuntimeError)):
            tp.neighbors.farthest_point_sampling(d, k)
        with pytest.raises((ValueError, RuntimeError)):
            tp.neighbors.kmeans_plusplus(d, k)
    with pytest.raises((ValueError, RuntimeError)):
        tp.neighbors.farthest_point_sampling(d, 2, start=4)
    rectangle = tp.as_lazy(torch.zeros(3, 4, device=device, dtype=torch.float64))
    with pytest.raises((ValueError, RuntimeError)):
        tp.neighbors.kmeans_plusplus(rectangle, 2)
    with pytest.raises(RuntimeError, match='inference-only'):
        tp.neighbors.farthest_point_sampling(torch.zeros(4, 4, device=device,
                                                         requires_grad=True), 2)
    with pytest.raises(RuntimeError, match='nonnegative'):
        tp.neighbors.kmeans_plusplus(-torch.ones(4, 4, device=device), 2)
