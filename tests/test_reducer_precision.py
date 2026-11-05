"""Tile precision and running-maximum regressions for native reducers."""
import pytest
import torch
import torchpairwise as tp


@pytest.mark.parametrize('channels', [1, 3])
def test_float32_reducer_rescaling_and_backward(channels, device):
    # Tile size two: Q=1 fits one channel tile; Q=3 replays channel tiles. Row zero
    # starts with a zero-mass tile, then its running maximum changes twice.
    raw = torch.tensor([
        [-torch.inf, -torch.inf, 4., 3., 8.],
        [1000., 999., 998., 1001., -torch.inf],
        [1e30, 1e30, 1e30, 1e30, 1e30],
        [-3., -2., -1., 0., 1.],
        [-torch.inf, -torch.inf, -torch.inf, -torch.inf, -torch.inf],
    ], device=device, requires_grad=True)
    values = torch.linspace(-.7, 1.1, 5 * channels, device=device).reshape(5, channels)
    values.requires_grad_()
    op = tp.as_lazy(raw, chunk_size=2)
    masked = raw.masked_fill(torch.eye(5, device=device, dtype=torch.bool), -torch.inf)
    actual = op.softmax_matmul(values, exclude_self=True)
    expected = masked.softmax(-1) @ values
    torch.testing.assert_close(actual, expected, equal_nan=True)
    cotangent = torch.linspace(.2, 1., actual.numel(), device=device).reshape_as(actual)
    # Ignore the all -inf row for the finite gradient comparison: softmax's
    # NaN probabilities would otherwise contaminate every payload gradient.
    finite_actual = tp.as_lazy(raw[:4], chunk_size=2).softmax_matmul(values)
    finite_expected = raw[:4].softmax(-1) @ values
    actual_grad = torch.autograd.grad(finite_actual, (raw, values), cotangent[:4])
    expected_grad = torch.autograd.grad(finite_expected, (raw, values), cotangent[:4])
    torch.testing.assert_close(actual_grad, expected_grad)
    actual_lse = op.logsumexp(-1, exclude_self=True)
    expected_lse = masked.logsumexp(-1)
    torch.testing.assert_close(actual_lse, expected_lse, equal_nan=True)
    torch.testing.assert_close(torch.autograd.grad(actual_lse.sum(), raw),
                               torch.autograd.grad(expected_lse.sum(), raw),
                               equal_nan=True)


def test_softmax_normalizes_before_contracting_large_payload(device):
    raw = torch.zeros(1, 5, device=device, dtype=torch.float64)
    values = torch.full((5, 1), torch.finfo(raw.dtype).max / 2,
                        device=device, dtype=raw.dtype)
    # An unnormalized online numerator overflows despite a finite answer.
    actual = tp.as_lazy(raw, chunk_size=2).softmax_matmul(values)
    torch.testing.assert_close(actual, raw.softmax(-1) @ values)


_FUSED_GAMMAS = [.1, 10., 1000.]
_FUSED_CASES = (
    [('softmax', q, exclude) for q in (1, 32) for exclude in (False, True)]
    + [('logsumexp', None, exclude) for exclude in (False, True)]
    + [('matmul', q, False) for q in (1, 32)])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
@pytest.mark.parametrize('n', [257, 4096])
@pytest.mark.parametrize('gamma', _FUSED_GAMMAS)
@pytest.mark.parametrize('consumer,channels,exclude', _FUSED_CASES)
def test_float32_fused_error_within_tile_error(n, gamma, consumer, channels,
                                               exclude):
    # Fused and tile results against a float64 dense reference on the same
    # float32 inputs; fused max abs error may exceed the tile path's by 1.5x.
    ops = torch.ops.torchpairwise
    generator = torch.Generator(device='cuda').manual_seed(n + int(gamma))
    x = torch.randn(n, 3, device='cuda', generator=generator)
    y = torch.randn(n, 3, device='cuda', generator=generator)
    values = torch.randn(n, channels or 1, device='cuda', generator=generator)
    sq = (x.double()[:, None] - y.double()[None]).square().sum(-1)
    if consumer == 'matmul':
        def run():
            return tp.lazy(x, y, metric='rbf', gamma=gamma).matmul(values)
        expected = (-gamma * sq).exp() @ values.double()
    else:
        scores = -gamma * sq
        if exclude:
            scores.fill_diagonal_(-torch.inf)
        if consumer == 'softmax':
            expected = scores.softmax(-1) @ values.double()
        else:
            expected = scores.logsumexp(-1)

        def run():
            lazy = -gamma * tp.lazy(x, y, metric='sqeuclidean')
            if consumer == 'softmax':
                return lazy.softmax_matmul(values, exclude_self=exclude)
            return lazy.logsumexp(-1, exclude_self=exclude)
    errors = {}
    previous = ops._set_fused_reducers(False)
    try:
        for fused in (False, True):
            ops._set_fused_reducers(fused)
            calls = ops._fused_reducer_calls()
            actual = run()
            assert (ops._fused_reducer_calls() > calls) == fused
            errors[fused] = (actual.double() - expected).abs().max().item()
    finally:
        ops._set_fused_reducers(previous)
    floor = 2 * torch.finfo(torch.float32).eps * expected.abs().max().item()
    assert errors[True] <= 1.5 * errors[False] + floor, errors
