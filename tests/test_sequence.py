"""Sequence values/first derivatives against independently enumerated paths."""
import pytest
import torch
import torchpairwise as tp


def _reference(a, b, *, gamma=None, window=None):
    paths = []

    def visit(i, j, path):
        if i >= a.shape[0] or j >= b.shape[0] or (window is not None and abs(i - j) > window):
            return
        path = path + [(i, j)]
        if i == a.shape[0] - 1 and j == b.shape[0] - 1:
            paths.append(path)
        else:
            for di, dj in ((1, 0), (0, 1), (1, 1)):
                visit(i + di, j + dj, path)

    visit(0, 0, [])
    if not paths:
        return a[:0].sum() + b[:0].sum() + float('inf')
    costs = torch.stack([torch.stack([(a[i] - b[j]).square().sum() for i, j in path]).sum()
                         for path in paths])
    return costs.min() if gamma is None else -gamma * torch.logsumexp(-costs / gamma, 0)


def _divergence(a, b, gamma, window=None):
    return (_reference(a, b, gamma=gamma, window=window)
            - .5 * _reference(a, a, gamma=gamma, window=window)
            - .5 * _reference(b, b, gamma=gamma, window=window))


def _functions(soft):
    if soft:
        return (tp.soft_dtw_divergences, tp.paired_soft_dtw_divergences,
                tp.condensed_soft_dtw_divergences, {'gamma': .7})
    return (tp.dtw_distances, tp.paired_dtw_distances,
            tp.condensed_dtw_distances, {'squared': True})


@pytest.mark.parametrize('soft', [False, True])
@pytest.mark.parametrize('window', [None, 1])
def test_sequence_path_enumeration_values_and_gradients(device, soft, window):
    a = torch.tensor([[[.2, -.1], [1.3, .5]]], device=device, dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[[-.4, .2], [.8, .7], [2., .3]]], device=device,
                     dtype=a.dtype, requires_grad=True)
    allpairs, _, _, kwargs = _functions(soft)
    actual = allpairs(a, b, window=window, **kwargs)[0, 0]
    expected = (_divergence(a[0], b[0], kwargs['gamma'], window) if soft
                else _reference(a[0], b[0], window=window))
    torch.testing.assert_close(actual, expected)
    for gradient, reference in zip(torch.autograd.grad(actual, (a, b)),
                                    torch.autograd.grad(expected, (a, b))):
        torch.testing.assert_close(gradient, reference)
    # Also exercise the internal shorter-axis transpose.
    torch.testing.assert_close(actual, allpairs(b, a, window=window, **kwargs)[0, 0])


@pytest.mark.parametrize('soft', [False, True])
def test_sequence_lengths_padding_values_and_gradients(device, soft):
    a = torch.tensor([[[.1], [.7], [1.9]], [[-.3], [float('nan')], [float('inf')]]],
                     device=device, dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[[.5], [1.2], [float('-inf')]], [[-.1], [.9], [2.4]]],
                     device=device, dtype=a.dtype, requires_grad=True)
    la = torch.tensor([3, 1], device=device)
    lb = torch.tensor([2, 3], device=device)
    allpairs, _, _, kwargs = _functions(soft)
    actual = allpairs(a, b, x1_lengths=la, x2_lengths=lb, chunk_size=1, **kwargs)
    expected = torch.stack([torch.stack([
        _divergence(a[i, :ta], b[j, :tb], kwargs['gamma']) if soft
        else _reference(a[i, :ta], b[j, :tb])
        for j, tb in enumerate((2, 3))]) for i, ta in enumerate((3, 1))])
    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(actual.sum(), (a, b))
    references = torch.autograd.grad(expected.sum(), (a, b))
    for gradient, reference in zip(gradients, references):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, reference)
    assert torch.count_nonzero(gradients[0][1, 1:]) == 0
    assert torch.count_nonzero(gradients[1][0, 2:]) == 0


@pytest.mark.parametrize('soft', [False, True])
def test_sequence_broadcast_noncontiguous_modes_and_gradients(device, soft):
    torch.manual_seed(551)
    a = torch.randn(2, 1, 3, 2, 4, device=device, dtype=torch.float64)[..., ::2].requires_grad_()
    b = torch.randn(1, 2, 3, 3, 2, device=device, dtype=a.dtype, requires_grad=True)
    la = torch.tensor([[[2, 1, 2]]], device=device)
    lb = torch.tensor([[[3, 2, 1]], [[2, 3, 2]]], device=device)
    allpairs, paired, condensed, kwargs = _functions(soft)
    full = allpairs(a, b, x1_lengths=la, x2_lengths=lb, chunk_size=2, **kwargs)
    aligned = paired(a, b, x1_lengths=la, x2_lengths=lb, chunk_size=1, **kwargs)
    assert full.shape == (2, 2, 3, 3)
    torch.testing.assert_close(aligned, full.diagonal(dim1=-2, dim2=-1))
    for gradient, reference in zip(torch.autograd.grad(aligned.sum(), (a, b)),
                                    torch.autograd.grad(full.diagonal(dim1=-2, dim2=-1).sum(), (a, b))):
        torch.testing.assert_close(gradient, reference)
    self_map = allpairs(a, x1_lengths=la, chunk_size=1, **kwargs)
    compact = condensed(a, x_lengths=la, chunk_size=2, **kwargs)
    i, j = torch.triu_indices(3, 3, 1, device=device)
    torch.testing.assert_close(compact, self_map[..., i, j])
    for gradient, reference in zip(torch.autograd.grad(compact.sum(), (a,)),
                                    torch.autograd.grad(self_map[..., i, j].sum(), (a,))):
        torch.testing.assert_close(gradient, reference)


@pytest.mark.parametrize('soft', [False, True])
def test_sequence_double_gradcheck(device, soft):
    a = torch.tensor([[[.2], [1.4]]], device=device, dtype=torch.float64, requires_grad=True)
    # Avoid the hard-DTW path tie at .8; a finite-difference derivative is
    # only unique away from alignment changes. Ties have separate tests.
    b = torch.tensor([[[-.3], [.9], [2.]]], device=device, dtype=a.dtype, requires_grad=True)
    _, paired, _, kwargs = _functions(soft)
    assert torch.autograd.gradcheck(lambda x, y: paired(x, y, **kwargs), (a, b),
                                    atol=2e-5, rtol=2e-4)


def test_sequence_identity_root_zero_window_and_repeated_frames(device):
    a = torch.tensor([[[.2], [.8], [1.9]]], device=device, dtype=torch.float64, requires_grad=True)
    b = (a.detach() + .5).requires_grad_()
    expected = (a - b).square().sum((-1, -2))
    for squared in (False, True):
        actual = tp.paired_dtw_distances(a, b, squared=squared, window=0)
        torch.testing.assert_close(actual, expected if squared else expected.sqrt())
        zero = tp.paired_dtw_distances(a, a, squared=squared)
        torch.testing.assert_close(zero, torch.zeros_like(zero), atol=0, rtol=0)
        torch.testing.assert_close(torch.autograd.grad(zero.sum(), a)[0], torch.zeros_like(a), atol=0, rtol=0)
    torch.testing.assert_close(tp.paired_soft_dtw_divergences(a, b, gamma=.7, window=0), expected)
    zero = tp.paired_soft_dtw_divergences(a, a.clone(), gamma=.7)
    torch.testing.assert_close(zero, torch.zeros_like(zero), atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(zero.sum(), a)[0], torch.zeros_like(a), atol=0, rtol=0)
    repeat = torch.tensor([[[1.], [1.], [2.]]], device=device)
    short = torch.tensor([[[1.], [2.]]], device=device)
    torch.testing.assert_close(tp.dtw_distances(repeat, short), torch.zeros(1, 1, device=device))
    # Debiasing differs from raw Soft-DTW, which is negative on this diagonal.
    assert _reference(a[0], a[0], gamma=.7) < 0


@pytest.mark.parametrize('soft', [False, True])
def test_sequence_infeasible_band_has_zero_whole_pair_gradient(device, soft):
    a = torch.tensor([[[.2], [1.3], [2.2]]], device=device, dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[[.7]]], device=device, dtype=a.dtype, requires_grad=True)
    allpairs, _, _, kwargs = _functions(soft)
    actual = allpairs(a, b, window=1, **kwargs)
    assert torch.isposinf(actual).all()
    for gradient in torch.autograd.grad(actual.sum(), (a, b)):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)


@pytest.mark.parametrize('soft', [False, True])
def test_sequence_empty_collections_and_batch(device, soft):
    a = torch.ones(2, 2, 1, device=device, dtype=torch.float64, requires_grad=True)
    allpairs, paired, condensed, kwargs = _functions(soft)
    for n in (0, 1):
        actual = condensed(a[:n], **kwargs)
        assert actual.shape == (0,)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), a)[0], torch.zeros_like(a))
    assert allpairs(a[:0], a, **kwargs).shape == (0, 2)
    assert paired(a[:0], a[:0], **kwargs).shape == (0,)
    assert allpairs(a.expand(0, 2, 2, 1), **kwargs).shape == (0, 2, 2)


def test_sequence_length_and_parameter_validation(device):
    a = torch.ones(2, 3, 1, device=device)
    with pytest.raises(ValueError, match='shape'):
        tp.dtw_distances(a[0])
    with pytest.raises(ValueError, match='non-empty'):
        tp.dtw_distances(a[:, :0])
    with pytest.raises(ValueError, match='equal sample counts'):
        tp.paired_dtw_distances(a, a[:1])
    with pytest.raises(ValueError, match='same device and dtype'):
        tp.dtw_distances(a, a.double())
    with pytest.raises(TypeError, match='boolean'):
        tp.dtw_distances(a, squared=1)
    with pytest.raises(TypeError, match='int64'):
        tp.dtw_distances(a, x1_lengths=torch.ones(2, device=device))
    with pytest.raises(ValueError, match='sample count'):
        tp.dtw_distances(a, x1_lengths=torch.ones(1, device=device, dtype=torch.int64))
    for bad in (0, 4):
        with pytest.raises(ValueError, match='between 1'):
            tp.dtw_distances(a, x1_lengths=torch.full((2,), bad, device=device, dtype=torch.int64))
    with pytest.raises(ValueError, match='finite'):
        tp.dtw_distances(a * float('nan'))
    for bad in (True, .5):
        with pytest.raises(TypeError, match='window'):
            tp.dtw_distances(a, window=bad)
    with pytest.raises(ValueError, match='window'):
        tp.dtw_distances(a, window=-1)
    with pytest.raises(ValueError, match='chunk_size'):
        tp.dtw_distances(a, chunk_size=0)
    with pytest.raises(TypeError, match='chunk_size'):
        tp.dtw_distances(a, chunk_size=True)
    for bad in (None, True, torch.tensor(1., device=device)):
        with pytest.raises(TypeError, match='gamma'):
            tp.soft_dtw_divergences(a, gamma=bad)
    for bad in (0, -.1, float('inf'), float('nan'), 1e-50):
        with pytest.raises(ValueError, match='gamma'):
            tp.soft_dtw_divergences(a, gamma=bad)


def test_sequence_autocast_contract(device):
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    a = torch.tensor([[[.2], [.8]], [[.4], [1.7]]], device=device, dtype=dtype, requires_grad=True)
    for soft in (False, True):
        allpairs, _, _, kwargs = _functions(soft)
        with pytest.raises(TypeError, match='float32 or float64'):
            allpairs(a, **kwargs)
        with torch.autocast(device, dtype=dtype):
            actual = allpairs(a, a + 1, **kwargs)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, allpairs(a.float(), (a + 1).float(), **kwargs))
        assert torch.isfinite(torch.autograd.grad(actual.sum(), a)[0]).all()
