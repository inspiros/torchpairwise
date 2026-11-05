"""Stable streamed top-k must retain global-index ties across tile merges."""
import pytest
import torch

import torchpairwise as tp


def _dense_topk(scores, k, largest, exclude_self=False):
    # Stable sorting preserves the original (global column index) tie order.
    allowed = torch.ones_like(scores, dtype=torch.bool)
    if exclude_self:
        allowed &= ~torch.eye(scores.shape[-1], device=scores.device, dtype=torch.bool)
    if scores.is_floating_point():
        nan = scores.isnan()
        key = torch.where(nan, 0., scores)
        category = torch.where(allowed, (~nan if largest else nan).to(torch.int8), 2)
    else:
        key = scores.to(torch.int64)
        category = (~allowed).to(torch.int8)
    order = torch.argsort(key, dim=-1, descending=largest, stable=True)
    by_category = torch.argsort(category.gather(-1, order), dim=-1, stable=True)
    indices = order.gather(-1, by_category)[..., :k]
    return scores.gather(-1, indices), indices


def _assert_selection(actual, expected):
    values, indices = expected
    torch.testing.assert_close(actual.indices, indices, rtol=0, atol=0)
    # Compare NaN payloads and the sign of zero as well as numerical values.
    if values.is_floating_point():
        bits = torch.int32 if values.dtype == torch.float32 else torch.int64
        torch.testing.assert_close(actual.values.contiguous().view(bits),
                                   values.contiguous().view(bits), rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual.values, values, rtol=0, atol=0)


def test_topk_merge_pairwise_ties_and_exclusion(device):
    x = (torch.arange(75, device=device).remainder(5).float())[:, None]
    scores = (x[:, None, :] - x[None, :, :]).square().sum(-1)
    for chunk in (3, 64, None):
        source = tp.lazy(x, metric='sqeuclidean_distances', chunk_size=chunk)
        for largest in (False, True):
            _assert_selection(
                source.topk(8, largest=largest, exclude_self=True),
                _dense_topk(scores, 8, largest, exclude_self=True))
        # These boundaries cover skipped tile loops, tiles narrower than k,
        # and the maximum public k with and without diagonal exclusion.
        for k, exclude in ((0, True), (75, False), (74, True)):
            _assert_selection(source.topk(k, largest=False, exclude_self=exclude),
                              _dense_topk(scores, k, False, exclude))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_topk_merge_nonfinite_and_signed_zero(device, dtype):
    pattern = torch.tensor([float('nan'), -0., 0., float('inf'),
                            -float('inf'), 1., -1.], device=device, dtype=dtype)
    scores = pattern.repeat(11)[:75].repeat(3, 1)
    scores[1].fill_(float('nan'))
    scores[2].zero_()
    scores[2, ::2] = -0.
    for chunk in (3, 64, None):
        source = tp.as_lazy(scores, chunk_size=chunk)
        for largest in (False, True):
            for k in (1, 8, 75):
                _assert_selection(source.topk(k, largest=largest),
                                  _dense_topk(scores, k, largest))


def test_topk_merge_integer_extremes(device):
    limits = torch.iinfo(torch.int64)
    scores = torch.tensor([limits.min, limits.max, 0, 0, 1, -1],
                          device=device).repeat(13)[:75].repeat(2, 1)
    scores[1].fill_(limits.min)
    for chunk in (3, 64, None):
        source = tp.as_lazy(scores, chunk_size=chunk)
        for largest in (False, True):
            for k in (1, 8, 75):
                _assert_selection(source.topk(k, largest=largest),
                                  _dense_topk(scores, k, largest))
