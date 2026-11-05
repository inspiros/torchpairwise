"""Point-set definitions, ragged prefixes, and comparison-mode assembly."""
import pytest
import torch
import torchpairwise as tp


def reference(x, y, xl, yl, metric, squared=False):
    # Inputs here are [B,N,P,C], already expanded to common B.
    batches = []
    for b in range(len(x)):
        rows = []
        for i in range(x.shape[1]):
            cols = []
            for j in range(y.shape[1]):
                ground = torch.cdist(x[b, i, :xl[b, i]], y[b, j, :yl[b, j]])
                if squared:
                    ground = ground.square()
                forward = ground.min(-1).values
                reverse = ground.min(-2).values
                if metric == 'directed_hausdorff':
                    value = forward.max()
                elif metric == 'hausdorff':
                    value = torch.maximum(forward.max(), reverse.max())
                else:
                    value = forward.mean() + reverse.mean()
                cols.append(value)
            rows.append(torch.stack(cols))
        batches.append(torch.stack(rows))
    return torch.stack(batches)


@pytest.mark.parametrize('metric,squared', [('directed_hausdorff', False), ('hausdorff', False),
                                          ('chamfer', False), ('chamfer', True)])
def test_ragged_pointset_modes_values_and_gradients(metric, squared, device):
    torch.manual_seed(87)
    xl = torch.tensor([[4, 2, 1], [3, 4, 2]], device=device)
    yl = torch.tensor([[3, 1, 2]], device=device)
    x = torch.rand(2, 3, 4, 2, dtype=torch.float64, device=device)
    y = torch.rand(1, 3, 3, 2, dtype=x.dtype, device=device) + .3
    x[torch.arange(4, device=device) >= xl[..., None]] = float('nan')
    y[torch.arange(3, device=device) >= yl[..., None]] = float('nan')
    # Exercise channel-first spatial layouts and broadcasted Y batches.
    x = x.reshape(2, 3, 2, 2, 2).movedim(-1, 2).contiguous().requires_grad_()
    y = y.reshape(1, 3, 1, 3, 2).movedim(-1, 2).contiguous().requires_grad_()
    xf = x.movedim(2, -1).flatten(2, -2)
    yf = y.movedim(2, -1).flatten(2, -2).expand(2, -1, -1, -1)
    kwargs = dict(batched=True, feature_dim=2, x1_lengths=xl, x2_lengths=yl)
    if metric == 'chamfer':
        kwargs['squared'] = squared
    expected = reference(xf, yf, xl, yl.expand(2, -1), metric, squared)
    actual = tp.cdist(x, y, metric=metric, **kwargs)
    named = getattr(tp, metric + '_distances')(x, y, **kwargs)
    torch.testing.assert_close(named[0] if isinstance(named, tuple) else named, expected)
    torch.testing.assert_close(actual, expected)
    paired = tp.paired_distances(x, y, metric=metric, **kwargs)
    torch.testing.assert_close(paired, expected.diagonal(dim1=-2, dim2=-1))
    for output, ref in ((actual, expected), (paired, expected.diagonal(dim1=-2, dim2=-1))):
        for grad, oracle in zip(torch.autograd.grad(output.sum(), (x, y), retain_graph=True),
                                torch.autograd.grad(ref.sum(), (x, y), retain_graph=True)):
            torch.testing.assert_close(grad, oracle)
            assert torch.isfinite(grad).all()  # NaN padding must receive zero gradient.
    if metric == 'directed_hausdorff':
        _, xi, yi = named
        assert (xi < xl[..., None]).all()
        assert (yi < yl[:, None, :]).all()
    else:
        self_expected = reference(xf, xf, xl, xl, metric, squared)
        i, j = torch.triu_indices(3, 3, 1, device=device)
        kwargs.pop('x2_lengths')
        condensed = tp.pdist(x, metric=metric, **kwargs)
        torch.testing.assert_close(condensed, self_expected[..., i, j])
        torch.testing.assert_close(torch.autograd.grad(condensed.sum(), x)[0],
                                   torch.autograd.grad(self_expected[..., i, j].sum(), x)[0])


@pytest.mark.parametrize('metric', ['hausdorff', 'chamfer'])
def test_pointset_full_lengths_zero_gradients_and_empty_samples(metric, device):
    x = torch.rand(2, 3, 2, device=device, requires_grad=True)
    y = torch.rand(3, 4, 2, device=device)
    xl = torch.full((2,), 3, device=device, dtype=torch.int64)
    yl = torch.full((3,), 4, device=device, dtype=torch.int64)
    actual = tp.cdist(x, y, metric=metric)
    torch.testing.assert_close(actual, tp.cdist(x, y, metric=metric, x1_lengths=xl, x2_lengths=yl))
    torch.testing.assert_close(actual, tp.cdist(y, x, metric=metric).T)
    paired = tp.paired_distances(x, x, metric=metric)
    torch.testing.assert_close(paired, torch.zeros_like(paired))
    torch.testing.assert_close(torch.autograd.grad(paired.sum(), x)[0], torch.zeros_like(x))
    for n in (0, 1):
        empty = tp.pdist(x[:n], metric=metric)
        assert empty.shape == (0,)
        torch.testing.assert_close(torch.autograd.grad(empty.sum(), x)[0], torch.zeros_like(x))
    assert tp.paired_distances(x[:0], x[:0], metric=metric).shape == (0,)
    assert tp.cdist(x[:0], y, metric=metric).shape == (0, 3)
    assert tp.cdist(x.expand(0, *x.shape), y.unsqueeze(0), metric=metric).shape == (0, 2, 3)


def test_pointset_lengths_validation(device):
    x = torch.ones(2, 3, 2, device=device)
    for lengths, error, message in [
        (torch.tensor([0, 2], device=device), ValueError, 'between 1'),
        (torch.tensor([4, 2], device=device), ValueError, 'between 1'),
        (torch.tensor([2], device=device), ValueError, 'sample count'),
        (torch.ones(2, device=device), TypeError, 'int64'),
        (torch.ones(2, 2, dtype=torch.int64, device=device), ValueError, 'batch axes'),
    ]:
        with pytest.raises(error, match=message):
            tp.chamfer_distances(x, x1_lengths=lengths)
    for size, error in [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)]:
        with pytest.raises(error, match='chunk_size'):
            tp.chamfer_distances(x, chunk_size=size)
    with pytest.raises(ValueError, match='shuffle or generator'):
        tp.hausdorff_distances(x, chunk_size=2, shuffle=True)
    with pytest.raises(TypeError, match='does not accept'):
        tp.cdist(x, x, metric='euclidean', chunk_size=2)
    with pytest.raises(TypeError, match='dense strided'):
        tp.chamfer_distances(x.to_sparse())
    lengths = torch.tensor([2, 3], device=device)
    with pytest.raises(ValueError, match='Shuffling'):
        tp.hausdorff_distances(x, x1_lengths=lengths, shuffle=True)
    with pytest.raises(TypeError, match='x2_lengths'):
        tp.pdist(x, metric='chamfer', x2_lengths=lengths)
    with pytest.raises(TypeError, match='boolean'):
        tp.chamfer_distances(x, squared=1)
    with pytest.raises(ValueError, match='equal sample'):
        tp.paired_distances(x, x[:1], metric='chamfer')


def test_pointset_autocast(device):
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    x = torch.rand(3, 4, 2, device=device, dtype=dtype, requires_grad=True)
    y = torch.rand(3, 5, 2, device=device, dtype=dtype, requires_grad=True)
    for metric in ('hausdorff', 'chamfer'):
        for fn, args in ((tp.cdist, (x, y)), (tp.paired_distances, (x, y)), (tp.pdist, (x,))):
            with torch.autocast(device, dtype=dtype):
                actual = fn(*args, metric=metric, chunk_size=2)
            assert actual.dtype == torch.float32
            torch.testing.assert_close(actual, fn(*(a.float() for a in args), metric=metric))
            assert torch.isfinite(torch.autograd.grad(actual.sum(), x)[0]).all()


@pytest.mark.compiler
def test_compiled_pointset_modes_and_lengths(device):
    torch.compiler.reset()
    x = torch.rand(3, 4, 2, device=device, requires_grad=True)
    y = torch.rand(3, 65, 2, device=device, requires_grad=True)
    xl = torch.tensor([4, 2, 3], device=device)
    yl = torch.tensor([33, 65, 2], device=device)
    def call(a, b, al, bl):
        return (tp.chamfer_distances(a, b, x1_lengths=al, x2_lengths=bl, chunk_size=2).sum()
                + tp.paired_distances(a, b, metric='chamfer', squared=True,
                                      x1_lengths=al, x2_lengths=bl, chunk_size=2).sum()
                + tp.pdist(a, metric='chamfer', x1_lengths=al, chunk_size=2).sum())
    expected = call(x, y, xl, yl)
    actual = torch.compile(call, fullgraph=True)(x, y, xl, yl)
    torch.testing.assert_close(actual, expected)
    for grad, ref in zip(torch.autograd.grad(actual, (x, y)), torch.autograd.grad(expected, (x, y))):
        torch.testing.assert_close(grad, ref)
    torch.compiler.reset()


@pytest.mark.parametrize('metric', ['directed_hausdorff', 'hausdorff', 'chamfer'])
def test_chunked_pointset_values_witnesses_and_gradients(metric, device):
    torch.manual_seed(54)
    x = torch.rand(2, 5, 4, 3, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(1, 3, 7, 3, device=device, dtype=x.dtype, requires_grad=True)
    xl = torch.tensor([[4, 2, 1, 3, 4], [1, 2, 4, 4, 3]], device=device)
    yl = torch.tensor([[6, 1, 7]], device=device)
    fn = getattr(tp, metric + '_distances')
    # Uneven tile edges, distinct lengths, tuple witnesses and self comparisons.
    for kwargs in ({}, {'x1_lengths': xl, 'x2_lengths': yl}):
        if kwargs:
            with torch.no_grad():
                x[torch.arange(4, device=device) >= xl[..., None]] = float('nan')
                y[torch.arange(7, device=device) >= yl[..., None]] = float('nan')
        expected = fn(x, y, **kwargs)
        actual = fn(x, y, chunk_size=2, **kwargs)
        if isinstance(actual, tuple):
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)
            actual, expected = actual[0], expected[0]
        else:
            torch.testing.assert_close(actual, expected)
        for grad, ref in zip(torch.autograd.grad(actual.sum(), (x, y)),
                             torch.autograd.grad(expected.sum(), (x, y))):
            torch.testing.assert_close(grad, ref)
    for paired in (False, True):
        a, b = x[:, :3], y
        options = dict(metric=metric, x1_lengths=xl[:, :3], x2_lengths=yl)
        compare = tp.paired_distances if paired else tp.cdist
        with torch.no_grad():
            torch.testing.assert_close(compare(a, b, chunk_size=2, **options), compare(a, b, **options))
        actual = compare(a, b, chunk_size=2, **options)
        expected = compare(a, b, **options)
        torch.testing.assert_close(actual, expected)
        for grad, ref in zip(torch.autograd.grad(actual.sum(), (x, y)),
                             torch.autograd.grad(expected.sum(), (x, y))):
            torch.testing.assert_close(grad, ref)
    if metric != 'directed_hausdorff':
        actual = tp.pdist(x, metric=metric, x1_lengths=xl, chunk_size=2)
        expected = tp.pdist(x, metric=metric, x1_lengths=xl)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                                   torch.autograd.grad(expected.sum(), x)[0])
    self_actual = fn(x, x1_lengths=xl, chunk_size=2)
    self_expected = fn(x, x1_lengths=xl)
    if isinstance(self_actual, tuple):
        self_actual, self_expected = self_actual[0], self_expected[0]
    torch.testing.assert_close(self_actual, self_expected)
    for compare, args, options in [(tp.cdist, (x[:, :0], y), {'metric': metric}),
                                   (tp.paired_distances, (x[:, :0], x[:, :0]), {'metric': metric})]:
        empty = compare(*args, chunk_size=2, **options)
        assert empty.numel() == 0
        torch.testing.assert_close(torch.autograd.grad(empty.sum(), x)[0], torch.zeros_like(x))


@pytest.mark.cuda
@pytest.mark.parametrize('q', [31, 32, 65])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_cuda_nearest_point_search_boundaries_and_ties(q, dtype):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    torch.manual_seed(21)
    # Source padding and a short target prefix exercise partially active work;
    # duplicate witnesses span different warp lanes and search iterations.
    x = torch.rand(1, 2, 5, 3, dtype=dtype, device='cuda', requires_grad=True)
    y = torch.full((1, 2, q, 3), 10., dtype=dtype, device='cuda')
    y[:, :, 5] = 0
    y[:, :, min(38, q - 1)] = 0
    y[:, 1, 1] = .1
    y.requires_grad_()
    xl = torch.tensor([[5, 2]], device='cuda')
    yl = torch.tensor([[q, 2]], device='cuda')
    squared, indices = torch.ops.torchpairwise._pointset_nn(x, y, xl, yl)
    ground = (x[:, :, None, :, None] - y[:, None, :, None]).square().sum(-1)
    valid_targets = torch.arange(q, device='cuda') < yl[..., None]
    expected, expected_indices = ground.masked_fill(~valid_targets[:, None, :, None], float('inf')).min(-1)
    valid_sources = (torch.arange(5, device='cuda') < xl[..., None])[:, :, None]
    expected = expected.masked_fill(~valid_sources, 0.)
    expected_indices = expected_indices.masked_fill(~valid_sources, 0)
    torch.testing.assert_close(squared, expected)
    torch.testing.assert_close(indices, expected_indices)
    assert indices[0, 0, 0].eq(5).all()
    for grad, ref in zip(torch.autograd.grad(squared.sum(), (x, y)),
                         torch.autograd.grad(expected.sum(), (x, y))):
        torch.testing.assert_close(grad, ref)
