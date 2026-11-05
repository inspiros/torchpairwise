"""Independent kernel values, derivatives and comparison-mode contracts."""
import pytest
import torch
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic

import torchpairwise as tp


CASES = [('rational_quadratic', {'length_scale': 1.7, 'alpha': .4}),
         *[('matern', {'length_scale': .7, 'nu': nu}) for nu in (.5, 1.5, 2.5)]]


@pytest.mark.parametrize('kernel,kwargs', CASES)
def test_radial_kernel_oracles_and_modes(kernel, kwargs, device):
    torch.manual_seed(125)
    oracle = (RationalQuadratic(**kwargs) if kernel == 'rational_quadratic' else Matern(**kwargs))
    for dtype in (torch.float32, torch.float64):
        # Distinct batches and strided features also check broadcast gradients.
        x = torch.rand(2, 1, 3, 6, device=device, dtype=dtype)[..., ::2].requires_grad_()
        y = torch.rand(1, 2, 4, 6, device=device, dtype=dtype)[..., ::2].requires_grad_()
        named = getattr(tp, kernel + '_kernel')
        actual = tp.pairwise_kernels(x, y, kernel=kernel, **kwargs)
        expected = torch.stack([torch.stack([
            torch.as_tensor(oracle(a.detach().cpu().numpy(), b.detach().cpu().numpy()),
                            device=device, dtype=dtype) for b in y[0]]) for a in x[:, 0]])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(named(x, y, **kwargs), actual)
        paired = tp.paired_kernels(x, y[..., :3, :], kernel=kernel, **kwargs)
        diagonal = actual[..., :3].diagonal(dim1=-2, dim2=-1)
        torch.testing.assert_close(paired, diagonal)
        for grad, ref in zip(torch.autograd.grad(paired.sum(), (x, y)),
                             torch.autograd.grad(diagonal.sum(), (x, y))):
            torch.testing.assert_close(grad, ref)
        i, j = torch.triu_indices(3, 3, 1, device=device)
        condensed = tp.condensed_kernels(x, kernel=kernel, **kwargs)
        self_matrix = named(x, **kwargs)
        torch.testing.assert_close(condensed, self_matrix[..., i, j])
        torch.testing.assert_close(torch.autograd.grad(condensed.sum(), x)[0],
                                   torch.autograd.grad(self_matrix[..., i, j].sum(), x)[0])
    if kernel == 'rational_quadratic' or kwargs['nu'] == 1.5:
        default_oracle = RationalQuadratic() if kernel == 'rational_quadratic' else Matern(nu=1.5)
        default = named(x[0, 0], y[0, 0])
        expected = torch.as_tensor(default_oracle(x[0, 0].detach().cpu().numpy(),
                                                y[0, 0].detach().cpu().numpy()), device=device)
        torch.testing.assert_close(default, expected)
    # Finite differences check input derivatives independently of our formulas.
    a = torch.rand(2, 3, dtype=torch.float64, device=device, requires_grad=True)
    b = torch.rand(3, 3, dtype=a.dtype, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(lambda u, v: named(u, v, **kwargs), (a, b), fast_mode=True)
    if kernel == 'rational_quadratic':
        torch.testing.assert_close(named(a, b, length_scale=1.7, alpha=1e8),
                                   tp.rbf_kernel(a, b, gamma=1 / (2 * 1.7 ** 2)), atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize('kernel,kwargs', CASES)
def test_radial_kernel_coincidence_empty_and_amp(kernel, kwargs, device):
    # N>25 exercises the shape where cdist normally switches to matrix multiply;
    # large common offsets would then lose small differences and exact diagonals.
    x = (1e4 + torch.arange(32, device=device)[:, None].expand(-1, 3) / 16).requires_grad_()
    y = x.detach().clone().requires_grad_()
    named = getattr(tp, kernel + '_kernel')
    matrix = named(x, y, **kwargs)
    torch.testing.assert_close(matrix.diagonal(), torch.ones(32, device=device), atol=0, rtol=0)
    for grad in torch.autograd.grad(matrix.diagonal().sum(), (x, y)):
        torch.testing.assert_close(grad, torch.zeros_like(grad))
    paired = tp.paired_kernels(x, y, kernel=kernel, **kwargs)
    torch.testing.assert_close(paired, torch.ones_like(paired))
    for grad in torch.autograd.grad(paired.sum(), (x, y)):
        torch.testing.assert_close(grad, torch.zeros_like(grad))
    distant = named(x, y + 1e4, **kwargs)
    assert torch.isfinite(distant).all() and (distant >= 0).all() and (distant < matrix.min()).all()
    for n in (0, 1):
        empty = tp.condensed_kernels(x[:n], kernel=kernel, **kwargs)
        assert empty.shape == (0,)
        torch.testing.assert_close(torch.autograd.grad(empty.sum(), x)[0], torch.zeros_like(x))
    for fn, a, b, shape in [(named, x[:0], y, (0, 32)),
                             (named, x.expand(0, *x.shape), y, (0, 32, 32)),
                             (tp.paired_kernels, x[:0], y[:0], (0,))]:
        options = {'kernel': kernel} if fn is tp.paired_kernels else {}
        empty = fn(a, b, **options, **kwargs)
        assert empty.shape == shape
        for grad, value in zip(torch.autograd.grad(empty.sum(), (x, y)), (x, y)):
            torch.testing.assert_close(grad, torch.zeros_like(value))
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    a, b = [torch.rand(3, 5, device=device).to(dtype).requires_grad_() for _ in range(2)]
    for fn, args in ((tp.paired_kernels, (a, b)), (tp.condensed_kernels, (a,))):
        with torch.autocast(device, dtype=dtype):
            actual = fn(*args, kernel=kernel, **kwargs)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, fn(*(v.float() for v in args), kernel=kernel, **kwargs))
        assert torch.isfinite(torch.autograd.grad(actual.sum(), a)[0]).all()


def test_radial_kernel_parameter_validation():
    x = torch.ones(2, 3)
    for kernel in ('rational_quadratic', 'matern'):
        for key in ('length_scale', 'alpha') if kernel == 'rational_quadratic' else ('length_scale',):
            for value in (0, -1, float('inf'), float('nan')):
                with pytest.raises(ValueError, match=key):
                    tp.pairwise_kernels(x, kernel=kernel, **{key: value})
            named = getattr(tp, kernel + '_kernel')
            for value in (True, None, [1., 2., 3.], torch.tensor(1.)):
                with pytest.raises(TypeError, match=key):
                    named(x, **{key: value})
                # Selector options are optional native arguments: None selects the default.
                if value is not None:
                    with pytest.raises(TypeError, match=key):
                        tp.pairwise_kernels(x, kernel=kernel, **{key: value})
        with pytest.raises(TypeError, match='does not accept'):
            tp.pairwise_kernels(x, kernel=kernel, gamma=1.)
    for nu in (0, 1., float('inf'), float('nan')):
        with pytest.raises(ValueError, match='nu'):
            tp.matern_kernel(x, nu=nu)
    for nu in (True, torch.tensor(1.5)):
        with pytest.raises(TypeError, match='nu'):
            tp.matern_kernel(x, nu=nu)


def test_matern_extreme_tails_and_gradients(device):
    # Both a distance-accumulation overflow and a finite r with overflowing r²
    # should give a zero tail, with finite zero first gradients in every mode.
    for dtype, far in ((torch.float32, 1e20), (torch.float64, 1e200)):
        for value, scale in ((far, 1.), (1., 1e-20)):
            x = torch.zeros(1, 2, dtype=dtype, device=device, requires_grad=True)
            y = torch.tensor([[value, 0.]], dtype=dtype, device=device, requires_grad=True)
            for nu in (.5, 1.5, 2.5):
                options = dict(length_scale=scale, nu=nu)
                outputs = (tp.matern_kernel(x, y, **options),
                           tp.paired_kernels(x, y, kernel='matern', **options),
                           tp.condensed_kernels(torch.cat((x, y)), kernel='matern', **options))
                for output in outputs:
                    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
                    for gradient, value in zip(torch.autograd.grad(output.sum(), (x, y)), (x, y)):
                        torch.testing.assert_close(gradient, torch.zeros_like(value), atol=0, rtol=0)


def test_rational_quadratic_large_finite_alpha(device):
    x = torch.tensor([[.2, .3]], dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tensor([[1.2, .3]], dtype=x.dtype, device=device, requires_grad=True)
    # alpha is finite, but 2*alpha overflows a Python float. The limit is RBF.
    outputs = (tp.rational_quadratic_kernel(x, y, alpha=1e308),
               tp.paired_kernels(x, y, kernel='rational_quadratic', alpha=1e308),
               tp.condensed_kernels(torch.cat((x, y)), kernel='rational_quadratic', alpha=1e308))
    expected = (-(x - y).square().sum(-1) / 2).exp()
    reference_gradients = torch.autograd.grad(expected.sum(), (x, y))
    for output in outputs:
        torch.testing.assert_close(output.reshape(-1), expected)
        for gradient, reference in zip(torch.autograd.grad(output.sum(), (x, y)), reference_gradients):
            torch.testing.assert_close(gradient, reference)


@pytest.mark.compiler
def test_radial_kernel_compiled_modes(device):
    torch.compiler.reset()
    x, y = [torch.rand(3, 4, device=device, requires_grad=True) for _ in range(2)]

    def call(a, b):
        outputs = []
        for kernel, kwargs in CASES:
            outputs.extend((tp.pairwise_kernels(a, b, kernel=kernel, **kwargs).sum(),
                            tp.paired_kernels(a, b, kernel=kernel, **kwargs).sum(),
                            tp.condensed_kernels(a, kernel=kernel, **kwargs).sum()))
            if kernel == 'matern':
                outputs.append(tp.matern_kernel(a, b, length_scale=1e-20, nu=kwargs['nu']).sum())
        return torch.stack(outputs)

    expected = call(x, y)
    actual = torch.compile(call, fullgraph=True)(x, y)
    torch.testing.assert_close(actual, expected)
    for grad, ref in zip(torch.autograd.grad(actual.sum(), (x, y)),
                         torch.autograd.grad(expected.sum(), (x, y))):
        torch.testing.assert_close(grad, ref, atol=1e-5, rtol=1e-4)
    torch.compiler.reset()
