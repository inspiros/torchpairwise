"""Probability normalization, raw histogram mass, and fused intersection."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist
import torch
import torchpairwise as tp


def functions(kind):
    if kind == 'hellinger':
        return tp.hellinger_distances, tp.paired_distances, tp.condensed_distances, {'metric': kind}
    return tp.histogram_intersection_kernel, tp.paired_kernels, tp.condensed_kernels, {'kernel': kind}


@pytest.mark.parametrize('kind', ['hellinger', 'histogram_intersection'])
def test_histogram_values_modes_and_gradients(kind, device):
    torch.manual_seed(19)
    named, paired_fn, condensed_fn, options = functions(kind)
    for dtype in (torch.float32, torch.float64):
        x = (torch.rand(2, 1, 3, 6, device=device, dtype=dtype) + .1)[..., ::2].requires_grad_()
        y = (torch.rand(1, 2, 4, 6, device=device, dtype=dtype) + .2)[..., ::2].requires_grad_()
        expected = []
        for a in x[:, 0].detach().cpu().double().numpy():
            row = []
            for b in y[0].detach().cpu().double().numpy():
                if kind == 'hellinger':
                    row.append(cdist(np.sqrt(a / a.sum(-1, keepdims=True)),
                                     np.sqrt(b / b.sum(-1, keepdims=True))) / np.sqrt(2))
                else:
                    row.append(np.minimum(a[:, None, :], b[None, :, :]).sum(-1))
            expected.append(row)
        actual = named(x, y)
        torch.testing.assert_close(actual, torch.as_tensor(np.asarray(expected), dtype=dtype, device=device))
        dispatch = tp.cdist if kind == 'hellinger' else tp.pairwise_kernels
        torch.testing.assert_close(dispatch(x, y, **options), actual)
        paired = paired_fn(x, y[..., :3, :], **options)
        diagonal = actual[..., :3].diagonal(dim1=-2, dim2=-1)
        torch.testing.assert_close(paired, diagonal)
        for grad, ref in zip(torch.autograd.grad(paired.sum(), (x, y)),
                             torch.autograd.grad(diagonal.sum(), (x, y))):
            torch.testing.assert_close(grad, ref)
        i, j = torch.triu_indices(3, 3, 1, device=device)
        condensed = condensed_fn(x, **options)
        reference = named(x)[..., i, j]
        torch.testing.assert_close(condensed, reference)
        torch.testing.assert_close(torch.autograd.grad(condensed.sum(), x)[0],
                                   torch.autograd.grad(reference.sum(), x)[0])
    a, b = [torch.rand(2, 3, device=device, dtype=torch.float64).requires_grad_() for _ in range(2)]
    assert torch.autograd.gradcheck(named, (a, b), fast_mode=True)


@pytest.mark.parametrize('kind', ['hellinger', 'histogram_intersection'])
def test_histogram_zeros_empty_and_amp(kind, device):
    named, paired_fn, condensed_fn, options = functions(kind)
    x = torch.tensor([[0., 1., 0.], [1., 0., 0.], [1., 1., 0.]], device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    matrix = named(x, y)
    if kind == 'hellinger':
        torch.testing.assert_close(matrix[:2, :2], torch.tensor([[0., 1.], [1., 0.]], device=device))
        torch.testing.assert_close(named(x * torch.tensor([2., 3., 4.], device=device)[:, None], y), matrix)
        for g in torch.autograd.grad(matrix.diagonal().sum(), (x, y)):
            torch.testing.assert_close(g, torch.zeros_like(g))
    else:
        torch.testing.assert_close(matrix.diagonal(), x.sum(-1))
        torch.testing.assert_close(named(x * 2, y * 2), matrix * 2)
        assert named(torch.zeros_like(x), y).eq(0).all()
        # Gradients at ties, including zero bins, match torch.minimum exactly.
        weights = torch.arange(9, device=device, dtype=x.dtype).reshape(3, 3)
        ref = torch.minimum(x[:, None], y[None]).sum(-1)
        for g, r in zip(torch.autograd.grad((matrix * weights).sum(), (x, y)),
                        torch.autograd.grad((ref * weights).sum(), (x, y))):
            torch.testing.assert_close(g, r)
        # Avoid subtractive cancellation of a small overlap beside a large bin.
        a = torch.tensor([[1e8, 1.]], device=device)
        b = torch.tensor([[0., 1.]], device=device)
        torch.testing.assert_close(named(a, b), torch.ones(1, 1, device=device), atol=0, rtol=0)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(named(x, y).sum(), (x, y)))
    for n in (0, 1):
        empty = condensed_fn(x[:n], **options)
        assert empty.shape == (0,)
        torch.testing.assert_close(torch.autograd.grad(empty.sum(), x)[0], torch.zeros_like(x))
    for fn, a, b, shape in ((named, x[:0], y, (0, 3)),
                             (named, x, y[:0], (3, 0)),
                             (named, x.expand(0, *x.shape), y, (0, 3, 3)),
                             (paired_fn, x[:0], y[:0], (0,))):
        empty = fn(a, b, **(options if fn is paired_fn else {}))
        assert empty.shape == shape
        for g, value in zip(torch.autograd.grad(empty.sum(), (x, y)), (x, y)):
            torch.testing.assert_close(g, torch.zeros_like(value))
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    a, b = [torch.rand(3, 5, device=device).to(dtype).requires_grad_() for _ in range(2)]
    for fn, args in ((paired_fn, (a, b)), (condensed_fn, (a,))):
        with torch.autocast(device, dtype=dtype):
            actual = fn(*args, **options)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, fn(*(v.float() for v in args), **options))
        assert torch.isfinite(torch.autograd.grad(actual.sum(), a)[0]).all()


def test_histogram_domain_validation(device):
    x = torch.ones(2, 3, device=device)
    for fn in (tp.hellinger_distances, tp.histogram_intersection_kernel):
        for value in (-1., float('inf'), float('nan')):
            bad = x.clone()
            bad[0, 0] = value
            with pytest.raises(ValueError, match='finite and non-negative'):
                fn(x, bad)
        with pytest.raises(TypeError, match='dense strided'):
            fn(x.to_sparse())
    with pytest.raises(ValueError, match='positive finite row mass'):
        tp.hellinger_distances(torch.zeros_like(x))
    with pytest.raises(ValueError, match='positive finite row mass'):
        tp.hellinger_distances(torch.full_like(x, torch.finfo(x.dtype).max))
    # Hellinger is also available in generic nearest/radius distance masks.
    torch.testing.assert_close(tp.radius_neighbors_mask(x, metric='hellinger', epsilon=0.),
                               ~torch.eye(2, device=device, dtype=torch.bool))


@pytest.mark.compiler
def test_histogram_compiled_modes(device):
    torch.compiler.reset()
    x, y = [(torch.rand(3, 4, device=device) + .1).requires_grad_() for _ in range(2)]

    def call(a, b):
        return torch.stack([fn(*args, **options).sum()
                            for kind in ('hellinger', 'histogram_intersection')
                            for named, paired_fn, condensed_fn, options in (functions(kind),)
                            for fn, args, options in ((named, (a, b), {}),
                                                      (paired_fn, (a, b), options),
                                                      (condensed_fn, (a,), options))])

    expected = call(x, y)
    actual = torch.compile(call, fullgraph=True)(x, y)
    torch.testing.assert_close(actual, expected)
    for g, r in zip(torch.autograd.grad(actual.sum(), (x, y)), torch.autograd.grad(expected.sum(), (x, y))):
        torch.testing.assert_close(g, r, atol=1e-5, rtol=1e-4)
    torch.compiler.reset()


@pytest.mark.cuda
def test_histogram_intersection_current_stream():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    x, y = [torch.rand(n, 5, device='cuda', dtype=torch.float64) for n in (3, 4)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        a, b = [(value * scale).requires_grad_() for value, scale in ((x, 2), (y, 3))]
        # The primitive avoids public value validation synchronizing the host.
        value = torch.ops.torchpairwise._histogram_intersection(a, b)
        gradients = torch.autograd.grad(value.sum(), (a, b))
        consumed = (value.clone(), *(g.clone() for g in gradients))
    torch.cuda.current_stream().wait_stream(stream)
    a, b = (x * 2).requires_grad_(), (y * 3).requires_grad_()
    ref = torch.minimum(a[:, None], b[None]).sum(-1)
    for a, b in zip(consumed, (ref, *torch.autograd.grad(ref.sum(), (a, b)))):
        torch.testing.assert_close(a, b)
