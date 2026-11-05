"""Missing-feature distances and transport of histogram masses on a shared grid."""
import numpy as np
import pytest
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import nan_euclidean_distances
import torch
import torchpairwise as tp


@pytest.mark.parametrize('metric', ['nan_euclidean', 'wasserstein_1d'])
def test_missing_wasserstein_oracles_modes_and_gradients(metric, device):
    torch.manual_seed(149)
    named = getattr(tp, metric + '_distances')
    for dtype in (torch.float32, torch.float64):
        x = (torch.rand(2, 1, 3, 8, device=device, dtype=dtype) + .1)[..., ::2]
        y = (torch.rand(1, 2, 4, 8, device=device, dtype=dtype) + .2)[..., ::2]
        if metric == 'nan_euclidean':
            x[..., 0, 1] = float('nan')
            y[..., 1, 2] = float('nan')
        x.requires_grad_()
        y.requires_grad_()
        positions = torch.tensor([0., .7, 2.5, 5.], device=device, dtype=dtype, requires_grad=True)
        options = {} if metric == 'nan_euclidean' else {'bin_positions': positions}
        expected = []
        for a in x[:, 0].detach().cpu().double().numpy():
            row = []
            for b in y[0].detach().cpu().double().numpy():
                if metric == 'nan_euclidean':
                    row.append(nan_euclidean_distances(a, b))
                else:
                    grid = positions.detach().cpu().double().numpy()
                    row.append([[wasserstein_distance(grid, grid, u, v) for v in b] for u in a])
            expected.append(row)
        actual = named(x, y, **options)
        torch.testing.assert_close(actual, torch.as_tensor(np.asarray(expected), device=device, dtype=dtype))
        torch.testing.assert_close(tp.cdist(x, y, metric=metric, **options), actual)
        if metric == 'nan_euclidean':
            torch.testing.assert_close(named(x, y, squared=True), actual.square())
        paired = tp.paired_distances(x, y[..., :3, :], metric=metric, **options)
        diagonal = actual[..., :3].diagonal(dim1=-2, dim2=-1)
        args = (x, y) if metric == 'nan_euclidean' else (x, y, positions)
        torch.testing.assert_close(paired, diagonal)
        for g, r in zip(torch.autograd.grad(paired.sum(), args), torch.autograd.grad(diagonal.sum(), args)):
            torch.testing.assert_close(g, r)
        i, j = torch.triu_indices(3, 3, 1, device=device)
        condensed = tp.pdist(x, metric=metric, **options)
        ref = named(x, **options)[..., i, j]
        torch.testing.assert_close(condensed, ref)
        args = (x,) if metric == 'nan_euclidean' else (x, positions)
        for g, r in zip(torch.autograd.grad(condensed.sum(), args), torch.autograd.grad(ref.sum(), args)):
            torch.testing.assert_close(g, r)
    a, b = [(torch.rand(n, 4, device=device, dtype=torch.float64) + .2).requires_grad_() for n in (2, 3)]
    if metric == 'nan_euclidean':
        # Gradcheck finite underlying variables, keeping the missingness pattern fixed.
        ma = torch.tensor([False, True, False, False], device=device)
        mb = torch.tensor([False, False, True, False], device=device)
        for squared in (False, True):
            assert torch.autograd.gradcheck(lambda u, v: named(u.masked_fill(ma, float('nan')),
                v.masked_fill(mb, float('nan')), squared=squared), (a, b), fast_mode=True)
    else:
        assert torch.autograd.gradcheck(lambda u, v, p: named(u, v, bin_positions=p),
                                        (a, b, positions), fast_mode=True)
        # Default unit-spaced grid and an explicit arange are equivalent.
        torch.testing.assert_close(named(a, b), named(a, b, bin_positions=torch.arange(4, device=device, dtype=a.dtype)))


def test_nan_euclidean_missing_zero_and_no_overlap(device):
    nan = float('nan')
    x = torch.tensor([[1., nan, 3.], [nan, nan, nan], [2., 4., nan]], device=device, requires_grad=True)
    y = torch.tensor([[nan, 2., 3.], [1., nan, nan]], device=device, requires_grad=True)
    actual = tp.nan_euclidean_distances(x, y)
    expected = torch.tensor([[0., 0.], [nan, nan], [2 * 3**.5, 3**.5]], device=device)
    torch.testing.assert_close(actual, expected, equal_nan=True)
    gx, gy = torch.autograd.grad(actual.sum(), (x, y))
    torch.testing.assert_close(gx, torch.tensor([[0., 0., 0.], [0., 0., 0.], [3**.5, 3**.5, 0.]], device=device))
    torch.testing.assert_close(gy, torch.tensor([[0., -3**.5, 0.], [-3**.5, 0., 0.]], device=device))
    a = torch.tensor([[nan, 1.]], device=device, requires_grad=True)
    b = torch.tensor([[2., nan]], device=device, requires_grad=True)
    for output in (tp.nan_euclidean_distances(a, b), tp.paired_distances(a, b, metric='nan_euclidean'),
                   tp.pdist(torch.cat((a, b)), metric='nan_euclidean')):
        assert output.isnan().all()
        for g, v in zip(torch.autograd.grad(output.sum(), (a, b)), (a, b)):
            torch.testing.assert_close(g, torch.zeros_like(v))
    assert tp.nan_euclidean_distances(x)[1, 1].isnan()  # No artificial self diagonal.


def test_wasserstein_1d_grid_scaling_and_single_bin(device):
    x = torch.tensor([[2., 0., 0.], [0., 0., 3.]], device=device, requires_grad=True)
    p = torch.tensor([0., 1., 4.], device=device, requires_grad=True)
    matrix = tp.wasserstein_1d_distances(x, bin_positions=p)
    torch.testing.assert_close(matrix, torch.tensor([[0., 4.], [4., 0.]], device=device))
    torch.testing.assert_close(tp.wasserstein_1d_distances(x * torch.tensor([[3.], [7.]], device=device), bin_positions=p), matrix)
    torch.testing.assert_close(tp.wasserstein_1d_distances(x, bin_positions=p + 100), matrix)
    torch.testing.assert_close(tp.wasserstein_1d_distances(x, bin_positions=p * 3), matrix * 3)
    torch.testing.assert_close(torch.autograd.grad(matrix[0, 1], p)[0], torch.tensor([-1., 0., 1.], device=device))
    y = x.detach().clone().requires_grad_()
    coincident = tp.paired_distances(x, y, metric='wasserstein_1d', bin_positions=p)
    for g, v in zip(torch.autograd.grad(coincident.sum(), (x, y, p)), (x, y, p)):
        torch.testing.assert_close(g, torch.zeros_like(v))
    a = torch.tensor([[2.], [5.]], device=device, requires_grad=True)
    grid = torch.tensor([7.], device=device, requires_grad=True)
    for output in (tp.wasserstein_1d_distances(a, bin_positions=grid),
                   tp.paired_distances(a, a, metric='wasserstein_1d', bin_positions=grid),
                   tp.pdist(a, metric='wasserstein_1d', bin_positions=grid)):
        torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
        for g, v in zip(torch.autograd.grad(output.sum(), (a, grid)), (a, grid)):
            torch.testing.assert_close(g, torch.zeros_like(v))
    torch.testing.assert_close(tp.radius_neighbors_mask(x, metric='wasserstein_1d', bin_positions=p, epsilon=0.),
                               torch.zeros(2, 2, dtype=torch.bool, device=device))


@pytest.mark.parametrize('metric', ['nan_euclidean', 'wasserstein_1d'])
def test_missing_wasserstein_empty_and_amp(metric, device):
    x, y = [torch.rand(3, 4, device=device, requires_grad=True) + .2 for _ in range(2)]
    named = getattr(tp, metric + '_distances')
    for n in (0, 1):
        empty = tp.pdist(x[:n], metric=metric)
        assert empty.shape == (0,)
        torch.testing.assert_close(torch.autograd.grad(empty.sum(), x)[0], torch.zeros_like(x))
    for fn, a, b, shape in ((named, x[:0], y, (0, 3)), (named, x, y[:0], (3, 0)),
                             (named, x.expand(0, *x.shape), y, (0, 3, 3)),
                             (tp.paired_distances, x[:0], y[:0], (0,))):
        out = fn(a, b, **({'metric': metric} if fn is tp.paired_distances else {}))
        assert out.shape == shape
        for g, v in zip(torch.autograd.grad(out.sum(), (x, y)), (x, y)):
            torch.testing.assert_close(g, torch.zeros_like(v))
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    a, b = [(torch.rand(3, 4, device=device) + .1).to(dtype).requires_grad_() for _ in range(2)]
    if metric == 'nan_euclidean':
        a = a.masked_fill(torch.tensor([False, True, False, False], device=device), float('nan'))
    options = {} if metric == 'nan_euclidean' else {'bin_positions': torch.tensor([0., .5, 1., 3.], device=device)}
    for fn, args in ((tp.paired_distances, (a, b)), (tp.pdist, (a,))):
        with torch.autocast(device, dtype=dtype):
            out = fn(*args, metric=metric, **options)
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, fn(*(v.float() for v in args), metric=metric, **options))
        assert torch.isfinite(torch.autograd.grad(out.sum(), a)[0]).all()


def test_missing_wasserstein_validation(device):
    x = torch.ones(2, 3, device=device)
    for val in (float('inf'), -float('inf')):
        with pytest.raises(ValueError, match='not infinities'):
            tp.nan_euclidean_distances(x, torch.full_like(x, val))
    with pytest.raises(TypeError, match='boolean'):
        tp.nan_euclidean_distances(x, squared=1)
    for metric in ('nan_euclidean', 'wasserstein_1d'):
        with pytest.raises(TypeError, match='dense strided'):
            tp.cdist(x.to_sparse(), x, metric=metric)
    for value in (0., -1., float('inf'), float('nan'), torch.finfo(x.dtype).max):
        with pytest.raises(ValueError):
            tp.wasserstein_1d_distances(torch.full_like(x, value))
    for positions, message in ((torch.ones(2, device=device), 'D entries'),
                                (torch.ones(1, 3, device=device), 'one-dimensional'),
                                (torch.arange(3, dtype=torch.float64, device=device), 'dtype and device'),
                                (torch.tensor([0., 0., 1.], device=device), 'strictly increasing'),
                                (torch.tensor([2., 1., 0.], device=device), 'strictly increasing'),
                                (torch.tensor([0., 1., float('inf')], device=device), 'finite')):
        with pytest.raises(ValueError, match=message):
            tp.wasserstein_1d_distances(x, bin_positions=positions)
    with pytest.raises(TypeError, match='does not accept'):
        tp.cdist(x, x, metric='nan_euclidean', bin_positions=torch.arange(3., device=device))
    with pytest.raises(TypeError, match='does not accept'):
        tp.pdist(x, metric='wasserstein_1d', squared=True)


@pytest.mark.compiler
def test_missing_wasserstein_compiled_modes(device):
    torch.compiler.reset()
    x, y = [(torch.rand(3, 4, device=device) + .2).requires_grad_() for _ in range(2)]
    p = torch.tensor([0., .5, 2., 4.], device=device, requires_grad=True)
    ma = torch.tensor([False, True, False, False], device=device)
    mb = torch.tensor([False, False, True, False], device=device)

    def call(a, b, positions):
        u, v = a.masked_fill(ma, float('nan')), b.masked_fill(mb, float('nan'))
        return torch.stack([tp.nan_euclidean_distances(u, v).sum(),
                            tp.paired_distances(u, v, metric='nan_euclidean', squared=True).sum(),
                            tp.pdist(u, metric='nan_euclidean').sum(),
                            tp.wasserstein_1d_distances(a, b, bin_positions=positions).sum(),
                            tp.paired_distances(a, b, metric='wasserstein_1d', bin_positions=positions).sum(),
                            tp.pdist(a, metric='wasserstein_1d', bin_positions=positions).sum()])

    expected = call(x, y, p)
    actual = torch.compile(call, fullgraph=True)(x, y, p)
    torch.testing.assert_close(actual, expected)
    for g, r in zip(torch.autograd.grad(actual.sum(), (x, y, p)), torch.autograd.grad(expected.sum(), (x, y, p))):
        torch.testing.assert_close(g, r, atol=1e-5, rtol=1e-4)
    torch.compiler.reset()


@pytest.mark.cuda
def test_nan_euclidean_current_stream():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    x, y = [torch.rand(n, 4, device='cuda', dtype=torch.float64) for n in (3, 4)]
    mask = torch.tensor([False, True, False, False], device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        a, b = (x * 2).masked_fill(mask, float('nan')).requires_grad_(), (y * 3).requires_grad_()
        value, counts = torch.ops.torchpairwise._nan_sqeuclidean(a, b)
        gradients = torch.autograd.grad(value.sum(), (a, b))
        consumed = (value.clone(), counts.clone(), *(g.clone() for g in gradients))
    torch.cuda.current_stream().wait_stream(stream)
    a, b = (x * 2).masked_fill(mask, float('nan')).requires_grad_(), (y * 3).requires_grad_()
    valid = ~a[:, None].isnan() & ~b[None].isnan()
    diff = torch.where(valid, a[:, None], 0.) - torch.where(valid, b[None], 0.)
    counts = valid.sum(-1)
    ref = diff.square().sum(-1) * (4 / counts.to(a.dtype))
    for actual, expected in zip(consumed, (ref, counts, *torch.autograd.grad(ref.sum(), (a, b)))):
        torch.testing.assert_close(actual, expected)
