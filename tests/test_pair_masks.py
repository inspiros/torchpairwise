"""Pair masks: consumers on where(mask, A, fill) against per-row dense references."""
import pytest
import torch
import torchpairwise as tp

N = 9
LABELS = torch.arange(N) % 3


def _masks(device):
    labels = LABELS.to(device)
    radius = tp.lazy(torch.zeros(N, 1, device=device), metric='sqeuclidean')
    return {
        'same': (tp.masks.same_group(labels), labels[:, None] == labels[None]),
        'different': (tp.masks.different_group(labels), labels[:, None] != labels[None]),
        'causal': (tp.masks.causal(), torch.ones(N, N, device=device, dtype=torch.bool).tril()),
        'band': (tp.masks.band(1, 2), torch.ones(N, N, device=device, dtype=torch.bool).triu(-1).tril(2)),
        'off_diagonal_same': (tp.masks.same_group(labels) & ~tp.masks.diagonal(),
                              (labels[:, None] == labels[None]) & ~torch.eye(N, device=device, dtype=torch.bool)),
        # A lazy Boolean expression on the left defers to PairMask.__rand__.
        'lazy_and': (((radius + 1) > 0.5) & tp.masks.causal(),
                     torch.ones(N, N, device=device, dtype=torch.bool).tril()),
        'or': (tp.masks.causal(-3) | tp.masks.diagonal(),
               torch.ones(N, N, device=device, dtype=torch.bool).tril(-3) | torch.eye(N, device=device, dtype=torch.bool)),
        'dense': (torch.eye(N, device=device, dtype=torch.bool).roll(1, 1) | torch.eye(N, device=device, dtype=torch.bool),
                  torch.eye(N, device=device, dtype=torch.bool).roll(1, 1) | torch.eye(N, device=device, dtype=torch.bool)),
        # A lazy Boolean expression whose shape matches the operand.
        'lazy': ((radius + 1) > 0.5, torch.ones(N, N, device=device, dtype=torch.bool)),
    }


def _reference(dense, mask, reduction, dim):
    """Reduce only the eligible entries of each row/column/whole matrix."""
    def reduce(values):
        if reduction in ('var', 'std'):
            return getattr(values, reduction)(correction=1)
        if reduction == 'logsumexp':
            return values.logsumexp(0)
        if reduction == 'count_nonzero':
            return values.count_nonzero()
        return getattr(values, reduction)()
    if dim is None:
        return reduce(dense[mask])
    if dim == -2:
        dense, mask = dense.T, mask.T
    return torch.stack([reduce(dense[i][mask[i]]) for i in range(dense.shape[0])])


@pytest.mark.parametrize('kind', ['same', 'different', 'causal', 'band', 'off_diagonal_same', 'or', 'dense', 'lazy',
                                  'lazy_and'])
def test_masked_reductions_match_eligible_entries(device, kind):
    torch.manual_seed(301)
    x = torch.randn(N, 3, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.randn(N, 3, device=device, dtype=torch.float64, requires_grad=True)
    mask, dense_mask = _masks(device)[kind]
    A = tp.lazy(x, y, 'rbf', gamma=.3, chunk_size=4)
    dense = tp.rbf_kernel(x, y, gamma=.3)
    actual, expected = [], []
    for dim in (-1, -2, None):
        counts = dense_mask.sum() if dim is None else dense_mask.sum(dim)
        reductions = ['sum', 'mean', 'logsumexp', 'prod', 'amin', 'amax', 'count_nonzero']
        if bool((counts >= 2).all()):
            reductions += ['var', 'std']
        for reduction in reductions:
            value = getattr(A, reduction)(dim, mask=mask)
            ref = _reference(dense, dense_mask, reduction, dim)
            torch.testing.assert_close(value, ref, msg=f'{reduction} dim={dim}')
            if value.is_floating_point():
                actual.append(value); expected.append(ref)
    cot = [torch.randn_like(v) for v in actual]
    torch.testing.assert_close(torch.autograd.grad(actual, (x, y), cot),
                               torch.autograd.grad(expected, (x, y), cot))


def test_masked_selection_softmax_and_weights(device):
    torch.manual_seed(302)
    x = torch.randn(2, N, 3, device=device, dtype=torch.float64, requires_grad=True)
    v = torch.randn(N, 2, device=device, dtype=x.dtype, requires_grad=True)
    labels = torch.stack([LABELS, torch.arange(N) // 3]).to(device)  # two different partitions [2,N]
    mask = tp.masks.different_group(labels)
    dense_mask = labels[..., :, None] != labels[..., None, :]
    A = tp.lazy(x, metric='sqeuclidean', chunk_size=4)
    dense = tp.sqeuclidean_distances(x)
    filled_small = dense.masked_fill(~dense_mask, float('inf'))
    filled_large = dense.masked_fill(~dense_mask, -float('inf'))
    nearest = A.min(-1, mask=mask)
    farthest = A.topk(2, largest=True, mask=mask)
    ref_nearest = filled_small.min(-1)
    ref_farthest = filled_large.topk(2, largest=True)
    assert torch.equal(nearest.indices, ref_nearest.indices)
    assert torch.equal(farthest.indices, ref_farthest.indices)
    assert torch.equal(A.argmin(-1, mask=mask), ref_nearest.indices)
    scores = -A
    attention = scores.softmax_matmul(v, mask=mask)
    ref_attention = torch.softmax(-filled_small, -1) @ v
    weights = torch.rand(N, device=device, dtype=x.dtype) + .1
    weighted = A.weighted_mean(weights, -1, mask=mask)
    ref_weighted = (dense * weights * dense_mask).sum(-1) / (weights * dense_mask).sum(-1)
    masked = A.masked(mask) @ v
    ref_masked = dense.masked_fill(~dense_mask, 0) @ v
    softmax = scores.softmax(-1, mask=mask) @ v
    actual = (nearest.values, farthest.values, attention, weighted, masked, softmax)
    expected = (ref_nearest.values, ref_farthest.values, ref_attention, ref_weighted, ref_masked, ref_attention)
    torch.testing.assert_close(actual, expected)
    cot = [torch.randn_like(t) for t in actual]
    torch.testing.assert_close(torch.autograd.grad(actual, (x, v), cot),
                               torch.autograd.grad(expected, (x, v), cot))


def test_causal_kernel_attention_and_transposed_rectangular_labels(device):
    torch.manual_seed(303)
    q = torch.randn(7, 4, device=device, dtype=torch.float64, requires_grad=True)
    k = torch.randn(7, 4, device=device, dtype=q.dtype, requires_grad=True)
    v = torch.randn(7, 3, device=device, dtype=q.dtype, requires_grad=True)
    causal = torch.ones(7, 7, device=device, dtype=torch.bool).tril()
    out = tp.lazy(q, k, 'linear', chunk_size=3).softmax_matmul(v, mask=tp.masks.causal())
    ref = torch.softmax((q @ k.T).masked_fill(~causal, -float('inf')), -1) @ v
    torch.testing.assert_close(out, ref)
    # Masks bind the axes of the operand they are applied to, including A.T.
    y = torch.randn(5, 4, device=device, dtype=q.dtype)
    rows, columns = torch.tensor([0, 1, 0, 1, 2], device=device), torch.tensor([1, 1, 0, 2, 0, 1, 2], device=device)
    A = tp.lazy(q, y, 'manhattan', chunk_size=2).T  # [5,7]
    dense = tp.manhattan_distances(q, y).T
    dense_mask = rows[:, None] == columns[None]
    total = A.sum(-1, mask=tp.masks.same_group(rows, columns))
    torch.testing.assert_close(total, (dense * dense_mask).sum(-1))
    cot = torch.randn_like(out)
    torch.testing.assert_close(torch.autograd.grad(out, (q, k, v), cot), torch.autograd.grad(ref, (q, k, v), cot))
    # exclude_self composes with a mask.
    B = tp.lazy(q, metric='sqeuclidean', chunk_size=3)
    same = torch.tensor([0, 0, 1, 1, 0, 1, 0], device=device)
    keep = (same[:, None] == same[None]) & ~torch.eye(7, device=device, dtype=torch.bool)
    torch.testing.assert_close(B.sum(-1, mask=tp.masks.same_group(same), exclude_self=True),
                               (tp.sqeuclidean_distances(q) * keep).sum(-1))


def test_fully_masked_rows_follow_reduction_identities(device):
    x = torch.randn(4, 2, device=device, dtype=torch.float64)
    A = tp.lazy(x, metric='sqeuclidean')
    nothing = tp.masks.different_group(torch.zeros(4, device=device, dtype=torch.long))
    assert torch.equal(A.sum(-1, mask=nothing), torch.zeros(4, device=device, dtype=x.dtype))
    assert torch.isnan(A.mean(-1, mask=nothing)).all()
    assert torch.isposinf(A.amin(-1, mask=nothing)).all()
    assert torch.isneginf(A.logsumexp(-1, mask=nothing)).all()
    assert torch.equal(A.count_nonzero(-1, mask=nothing), torch.zeros(4, device=device, dtype=torch.long))


def test_mask_contract_errors(device):
    x = torch.randn(4, 2, device=device)
    y = torch.randn(3, 2, device=device)
    A = tp.lazy(x, y)
    labels = torch.arange(4, device=device)
    with pytest.raises(ValueError, match='square'):
        A.sum(-1, mask=tp.masks.same_group(labels))
    with pytest.raises(ValueError, match='labels'):
        A.sum(-1, mask=tp.masks.same_group(labels, labels))
    with pytest.raises(ValueError, match='mask'):
        A.median(-1, mask=tp.masks.causal())
    with pytest.raises(ValueError, match='predicate'):
        A.any(-1, mask=tp.masks.causal(), predicate=torch.isfinite)
    with pytest.raises(TypeError, match='Boolean'):
        A.sum(-1, mask=A)
    with pytest.raises(TypeError, match='integer'):
        tp.masks.causal(1.5)
    with pytest.raises(ValueError, match='empty'):
        tp.masks.band(-2, 1)
    with pytest.raises(ValueError, match='batch'):  # masks may not add batch axes
        tp.as_lazy(torch.rand(4, 4, device=device)).min(1, mask=torch.ones(2, 4, 4, dtype=torch.bool, device=device))


def test_masked_statistics_gradients_ignore_excluded_entries(device):
    # values[mask] references: excluded entries get zero gradient even on rows
    # whose normalization is invalid (empty mean, single-entry var).
    data = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=device, dtype=torch.float64, requires_grad=True)
    keep = torch.tensor([[False, False, False], [True, False, False]], device=device)
    A = tp.as_lazy(data, chunk_size=2)
    for reduction in ('mean', 'var'):
        value = getattr(A, reduction)(-1, mask=keep)
        grad, = torch.autograd.grad(value, data, torch.ones_like(value))
        assert torch.equal(grad[~keep], torch.zeros_like(grad[~keep])), reduction


@pytest.mark.skipif(not tp.backends.keops.available(), reason='KeOps unavailable')
def test_masked_keops_lowering_matches_native(device):
    if device == 'cpu' and not torch.cuda.is_available():
        pytest.skip('KeOps parity is checked on the CUDA environment')
    torch.manual_seed(304)
    x = torch.randn(64, 3, device=device)
    v = torch.randn(64, 2, device=device)
    labels = torch.arange(64, device=device) % 5
    mask = tp.masks.same_group(labels) & ~tp.masks.diagonal()
    A = tp.lazy(x, metric='rbf_kernel', gamma=.5)
    native = (A.masked(mask) @ v, A.logsumexp(-1, mask=mask), A.topk(3, mask=mask).values,
              A.softmax_matmul(v, mask=tp.masks.causal()))
    calls = tp.backends.keops._stats['calls']
    with tp.backend('keops'):
        lowered = (A.masked(mask) @ v, A.logsumexp(-1, mask=mask), A.topk(3, mask=mask).values,
                   A.softmax_matmul(v, mask=tp.masks.causal()))
    # -inf fills stay native for exp-shift reductions (KeOps NaNs on leading -inf).
    assert tp.backends.keops._stats['calls'] == calls + 2
    torch.testing.assert_close(lowered, native, rtol=1e-4, atol=1e-5)
    # causal is lowered exactly: an index-difference margin, filled with 0.
    with tp.backend('keops'):
        lowered = A.masked(tp.masks.causal()) @ v
    assert tp.backends.keops._stats['calls'] == calls + 3
    torch.testing.assert_close(lowered, A.masked(tp.masks.causal()) @ v, rtol=1e-4, atol=1e-5)
