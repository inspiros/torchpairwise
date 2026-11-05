"""Independent compositional references, support boundaries, and public modes."""
from decimal import Decimal, localcontext
import numpy as np
import pytest
import torch
import torchpairwise as tp


def functions(kind):
    if kind == 'bhattacharyya_kernel':
        return (tp.bhattacharyya_kernel, tp.pairwise_kernels, tp.paired_kernels,
                tp.condensed_kernels, {'kernel': 'bhattacharyya'})
    return (getattr(tp, kind + '_distances'), tp.cdist, tp.paired_distances,
            tp.condensed_distances, {'metric': kind})


def reference(x, y, kind):
    if kind == 'aitchison':
        # Independent log-ratio identity: the sum over unordered feature pairs
        # divided by D equals the squared Euclidean distance between CLRs.
        i, j = np.triu_indices(x.shape[-1], 1)
        a = np.log(x[..., i] / x[..., j])
        b = np.log(y[..., i] / y[..., j])
        return np.sqrt(((a[..., :, None, :] - b[..., None, :, :]) ** 2).sum(-1) / x.shape[-1])
    p, q = x / x.sum(-1, keepdims=True), y / y.sum(-1, keepdims=True)
    coefficient = np.sqrt(p[..., :, None, :] * q[..., None, :, :]).sum(-1)
    if kind == 'bhattacharyya_kernel':
        return coefficient
    with np.errstate(divide='ignore'):
        return -np.log(coefficient)


@pytest.mark.parametrize('kind', ['bhattacharyya_kernel', 'bhattacharyya', 'aitchison'])
def test_compositional_reference_batched_modes_and_gradients(kind, device):
    torch.manual_seed(58)
    fn, dispatch, paired_fn, condensed_fn, options = functions(kind)
    for dtype in (torch.float32, torch.float64):
        x = (torch.rand(2, 1, 3, 6, device=device, dtype=dtype) + .2)[..., ::2].requires_grad_()
        y = (torch.rand(1, 2, 4, 6, device=device, dtype=dtype) + .3)[..., ::2].requires_grad_()
        actual = fn(x, y)
        expected = reference(x.detach().cpu().double().numpy(), y.detach().cpu().double().numpy(), kind)
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=dtype, device=device))
        torch.testing.assert_close(dispatch(x, y, **options), actual)
        if kind == 'aitchison':
            a, b = x.log(), y.log()
            a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
            direct = torch.linalg.vector_norm(a.unsqueeze(-2) - b.unsqueeze(-3), dim=-1)
        else:
            p, q = x / x.sum(-1, keepdim=True), y / y.sum(-1, keepdim=True)
            direct = (p.unsqueeze(-2) * q.unsqueeze(-3)).sqrt().sum(-1)
            if kind == 'bhattacharyya':
                direct = -direct.log()
        weights = torch.linspace(.2, 1., actual.numel(), dtype=dtype, device=device).reshape(actual.shape)
        for grad, ref in zip(torch.autograd.grad((actual * weights).sum(), (x, y)),
                             torch.autograd.grad((direct * weights).sum(), (x, y))):
            torch.testing.assert_close(grad, ref, atol=3e-6, rtol=2e-5)
        paired = paired_fn(x, y[..., :3, :], **options)
        diagonal = fn(x, y[..., :3, :]).diagonal(dim1=-2, dim2=-1)
        torch.testing.assert_close(paired, diagonal)
        for grad, ref in zip(torch.autograd.grad(paired.sum(), (x, y)),
                             torch.autograd.grad(diagonal.sum(), (x, y))):
            torch.testing.assert_close(grad, ref, atol=3e-6, rtol=2e-5)
        i, j = torch.triu_indices(3, 3, 1, device=device)
        condensed = condensed_fn(x, **options)
        triangle = fn(x)[..., i, j]
        torch.testing.assert_close(condensed, triangle)
        torch.testing.assert_close(torch.autograd.grad(condensed.sum(), x)[0],
                                   torch.autograd.grad(triangle.sum(), x)[0], atol=3e-6, rtol=2e-5)
    x, y = [(torch.rand(2, 3, dtype=torch.float64, device=device) + .3).requires_grad_()
            for _ in range(2)]
    assert torch.autograd.gradcheck(fn, (x, y), fast_mode=True)


def test_bhattacharyya_support_normalization_and_infinite_gradients(device):
    x = torch.tensor([[0., 1., 0.], [1., 0., 0.], [1., 1., 0.]],
                     dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tensor([[0., 1., 0.], [1., 1., 0.], [0., 0., 1.]],
                     dtype=torch.float64, device=device, requires_grad=True)
    for kind in ('bhattacharyya_kernel', 'bhattacharyya'):
        fn, _, paired_fn, condensed_fn, options = functions(kind)
        actual = fn(x, y)
        expected = reference(x.detach().cpu().numpy(), y.detach().cpu().numpy(), kind)
        torch.testing.assert_close(actual, torch.as_tensor(expected, device=device))
        torch.testing.assert_close(fn(x * torch.tensor([[2.], [7.], [4.]], device=device), y * 3), actual)
        torch.testing.assert_close(paired_fn(x, y, **options), actual.diagonal())
        i, j = torch.triu_indices(3, 3, 1, device=device)
        torch.testing.assert_close(condensed_fn(x, **options), fn(x)[i, j])
        gradients = torch.autograd.grad(actual.sum(), (x, y))
        for grad, value in zip(gradients, (x, y)):
            assert torch.isfinite(grad).all()
            assert grad[value == 0].eq(0).all()
    for grad in torch.autograd.grad(tp.bhattacharyya_distances(x[:1], y[2:]).sum(), (x, y)):
        torch.testing.assert_close(grad, torch.zeros_like(grad))
    # Masking infinite pairs after evaluation produces the same finite-pair gradients.
    all_values = tp.bhattacharyya_distances(x, y)
    finite_values = tp.bhattacharyya_distances(x, y)
    for grad, ref in zip(torch.autograd.grad(all_values.sum(), (x, y)),
                         torch.autograd.grad(torch.where(torch.isfinite(finite_values), finite_values, 0.).sum(),
                                             (x, y))):
        torch.testing.assert_close(grad, ref)
    same = tp.bhattacharyya_distances(x, x).diagonal()
    assert same.eq(0).all()
    torch.testing.assert_close(torch.autograd.grad(same.sum(), x)[0], torch.zeros_like(x))
    # Preserve support when a float32 normalized probability would underflow.
    a = torch.tensor([[torch.finfo(torch.float32).tiny, 1e20]], device=device)
    b = torch.tensor([[1., 0.]], device=device)
    assert tp.bhattacharyya_kernel(a, b).item() > 0
    assert torch.isfinite(tp.bhattacharyya_distances(a, b)).all()


def test_bhattacharyya_near_identity_decimal_reference(device):
    for dtype, delta, rtol in ((torch.float32, 1e-3, 2e-5), (torch.float64, 1e-7, 2e-7)):
        x = torch.tensor([[2., 3., 5.]], dtype=dtype, device=device, requires_grad=True)
        y = torch.tensor([[2. + delta, 3. - delta, 5.]], dtype=dtype, device=device, requires_grad=True)
        with localcontext() as context:
            context.prec = 70
            a = [Decimal(float(v)) for v in x.detach().cpu()[0]]
            b = [Decimal(float(v)) for v in y.detach().cpu()[0]]
            p, q = [v / sum(a) for v in a], [v / sum(b) for v in b]
            coefficient = sum((u * v).sqrt() for u, v in zip(p, q))
            distance = -coefficient.ln()
            gx = [(1 - (v / u).sqrt() / coefficient) / (2 * sum(a)) for u, v in zip(p, q)]
            gy = [(1 - (u / v).sqrt() / coefficient) / (2 * sum(b)) for u, v in zip(p, q)]
        actual = tp.bhattacharyya_distances(x, y)
        assert actual.item() > 0
        torch.testing.assert_close(actual, torch.tensor([[float(distance)]], dtype=dtype, device=device),
                                   atol=0, rtol=rtol)
        for grad, ref in zip(torch.autograd.grad(actual.sum(), (x, y)), (gx, gy)):
            torch.testing.assert_close(grad, torch.tensor([[float(v) for v in ref]], dtype=dtype, device=device),
                                       atol=1e-12 if dtype == torch.float32 else 1e-16, rtol=rtol)


def test_bhattacharyya_small_overlap_gradient(device):
    x = torch.tensor([[99., 1.]], dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tensor([[1., 99.]], dtype=torch.float64, device=device, requires_grad=True)
    coefficient = 2 * np.sqrt(.99 * .01)
    torch.testing.assert_close(tp.bhattacharyya_kernel(x, y), torch.tensor([[coefficient]], device=device))
    torch.testing.assert_close(tp.bhattacharyya_distances(x, y),
                               torch.tensor([[-np.log(coefficient)]], device=device))
    # Exercise the direct logarithm branch away from the root-distance branch.
    assert torch.autograd.gradcheck(tp.bhattacharyya_distances, (x, y), fast_mode=True)


def test_aitchison_scale_identity_and_large_mass(device):
    x = torch.tensor([[1., 2., 4.], [3., 1., 2.]], dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tensor([[2., 3., 1.]], dtype=torch.float64, device=device)
    actual = tp.aitchison_distances(x, y)
    torch.testing.assert_close(tp.aitchison_distances(x * torch.tensor([[2.], [7.]], device=device), y * 5), actual)
    same = tp.aitchison_distances(x, x).diagonal()
    assert same.eq(0).all()
    torch.testing.assert_close(torch.autograd.grad(same.sum(), x)[0], torch.zeros_like(x))
    # Aitchison logs raw entries: finite entries remain valid even if their sum overflows.
    huge = torch.full((2, 3), torch.finfo(torch.float32).max, device=device)
    assert tp.aitchison_distances(huge).eq(0).all()
    single = torch.tensor([[1.], [7.]], dtype=torch.float64, device=device, requires_grad=True)
    value = tp.aitchison_distances(single)
    assert value.eq(0).all()
    torch.testing.assert_close(torch.autograd.grad(value.sum(), single)[0], torch.zeros_like(single))


@pytest.mark.parametrize('kind', ['bhattacharyya_kernel', 'bhattacharyya', 'aitchison'])
def test_compositional_empty_and_validation(kind, device):
    fn, _, paired_fn, condensed_fn, options = functions(kind)
    x, y = [(torch.rand(3, 4, device=device) + .2).requires_grad_() for _ in range(2)]
    for a, b, shape in ((x[:0], y, (0, 3)), (x, y[:0], (3, 0)),
                        (x.expand(0, *x.shape), y, (0, 3, 3))):
        value = fn(a, b)
        assert value.shape == shape
        for grad in torch.autograd.grad(value.sum(), (x, y)):
            torch.testing.assert_close(grad, torch.zeros_like(grad))
    value = paired_fn(x[:0], y[:0], **options)
    assert value.shape == (0,)
    for grad in torch.autograd.grad(value.sum(), (x, y)):
        torch.testing.assert_close(grad, torch.zeros_like(grad))
    for n in (0, 1):
        value = condensed_fn(x[:n], **options)
        assert value.shape == (0,)
        torch.testing.assert_close(torch.autograd.grad(value.sum(), x)[0], torch.zeros_like(x))
    for invalid in (-1., float('inf'), float('nan'), *([0.] if kind == 'aitchison' else [])):
        bad = y.detach().clone()
        bad[0, 0] = invalid
        with pytest.raises(ValueError, match='finite and strictly positive' if kind == 'aitchison'
                           else 'finite and non-negative'):
            fn(x, bad)
    if kind != 'aitchison':
        for bad in (torch.zeros_like(x), torch.full_like(x, torch.finfo(x.dtype).max)):
            with pytest.raises(ValueError, match='positive finite row mass'):
                fn(bad)
    with pytest.raises(TypeError, match='dense strided'):
        fn(x.to_sparse())
    with pytest.raises(TypeError, match='float32 or float64'):
        fn(x.half())


@pytest.mark.compiler
def test_compositional_compiler_and_amp_modes(device):
    torch.compiler.reset()
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16

    def call(x, y):
        with torch.autocast(device, dtype=dtype):
            return torch.stack([operation(*args, **kwargs).sum()
                                for kind in ('bhattacharyya_kernel', 'bhattacharyya', 'aitchison')
                                for fn, _, paired_fn, condensed_fn, options in (functions(kind),)
                                for operation, args, kwargs in ((fn, (x, y), {}),
                                                               (paired_fn, (x, y), options),
                                                               (condensed_fn, (x,), options))])

    torch.manual_seed(23)
    # AMP training normally keeps master parameters in float32. Quantize the
    # values first so low-precision input promotion is checked on identical data,
    # without comparing nine separately rounded bfloat16 leaf accumulations.
    low = [(torch.rand(3, 4, device=device) + .3).to(dtype).requires_grad_()
           for _ in range(2)]
    x, y = [value.detach().float().requires_grad_() for value in low]
    expected = call(x, y)
    actual = torch.compile(call, fullgraph=True)(x, y)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    promoted = call(*low)
    torch.testing.assert_close(expected, promoted, atol=2e-6, rtol=1e-5)
    for grad in torch.autograd.grad(promoted.sum(), low):
        assert torch.isfinite(grad).all()
    # Also check zero upstream entries, as when only one returned result is used.
    for output_grad in (torch.ones_like(actual), torch.eye(9, device=device)[0]):
        for grad, ref in zip(torch.autograd.grad(actual, (x, y), output_grad, retain_graph=True),
                             torch.autograd.grad(expected, (x, y), output_grad, retain_graph=True)):
            assert torch.isfinite(grad).all()
            torch.testing.assert_close(grad, ref, atol=3e-6, rtol=2e-5)
    torch.compiler.reset()
