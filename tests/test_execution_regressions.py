"""Regressions for native execution: tiling, validation, autocast and kernels."""
import math
import subprocess
import sys
import textwrap

import pytest
import torch
import torchpairwise as tp


def _upper(full):
    rows, columns = torch.triu_indices(full.shape[-1], full.shape[-1], 1)
    return full[..., rows, columns]


@pytest.mark.parametrize('metric,kwargs', [
    ('euclidean', {}), ('canberra', {}), ('cosine', {'w': True}),
    ('sqeuclidean', {'w': True}), ('hellinger', {}), ('kl_symmetric', {}),
])
def test_condensed_row_blocks_match_all_pairs(device, metric, kwargs):
    # Enough samples to span several budgeted condensed row blocks.
    n = 2100 if metric in ('euclidean', 'canberra') else 700
    torch.manual_seed(0)
    x = torch.rand(n, 3, dtype=torch.float64, device=device) + .1
    name = 'symmetric_kl' if metric == 'kl_symmetric' else metric
    options = {}
    if kwargs.get('w'):
        options['w'] = torch.rand(3, dtype=torch.float64, device=device) + .5
    a = x.clone().requires_grad_()
    condensed = tp.pdist(a, metric=name, **options)
    g = torch.randn_like(condensed)
    (grad,) = torch.autograd.grad(condensed, a, g)
    b = x.clone().requires_grad_()
    full = _upper(tp.cdist(b, b, metric=name, **options))
    (reference,) = torch.autograd.grad(full, b, g)
    torch.testing.assert_close(condensed, full)
    torch.testing.assert_close(grad, reference)


@pytest.mark.parametrize('metric', ['mahalanobis', 'seuclidean'])
def test_paired_statistics_match_all_pairs_diagonal(device, metric):
    torch.manual_seed(0)
    x = torch.randn(2, 6, 4, dtype=torch.float64, device=device)
    y = torch.randn(2, 6, 4, dtype=torch.float64, device=device)
    if metric == 'mahalanobis':
        a = torch.randn(2, 4, 4, dtype=torch.float64, device=device)
        # Deliberately nonsymmetric: gradients must follow the stated formula.
        parameter = {'VI': a @ a.mT + torch.eye(4, dtype=torch.float64, device=device) + .1 * a}
    else:
        parameter = {'V': torch.rand(2, 4, dtype=torch.float64, device=device) + .5}
    key, value = next(iter(parameter.items()))
    leaves = [t.clone().requires_grad_() for t in (x, y, value)]
    paired = tp.paired_distances(leaves[0], leaves[1], metric=metric, **{key: leaves[2]})
    grads = torch.autograd.grad(paired.sum(), leaves)
    others = [t.clone().requires_grad_() for t in (x, y, value)]
    full = tp.cdist(others[0], others[1], metric=metric, **{key: others[2]})
    diagonal = full.diagonal(dim1=-2, dim2=-1)
    references = torch.autograd.grad(diagonal.sum(), others)
    torch.testing.assert_close(paired, diagonal)
    for grad, reference in zip(grads, references):
        torch.testing.assert_close(grad, reference)


def test_weighted_minkowski_infinity_uses_positive_weight_support(device):
    x = torch.tensor([[0., 0., 9.]], device=device)
    y = torch.tensor([[1., 3., 0.]], device=device)
    w = torch.tensor([2., .5, 0.], device=device)
    # max over coordinates with w > 0 of |x - y|; the zero-weight 9 is ignored.
    assert tp.wminkowski_distances(x, y, p=math.inf, w=w).item() == 3.


@pytest.mark.parametrize('n,m', [(3, 5000), (40, 60)])
def test_legacy_backward_kernels_match_float64_reference(device, n, m):
    # Few query rows with many references exercises chunked reductions.
    torch.manual_seed(0)
    x = torch.randn(n, 3, device=device)
    y = torch.randn(m, 3, device=device)
    angles = torch.rand(n, 2, device=device)
    others = torch.rand(m, 2, device=device)
    a = torch.randn(3, 3, device=device)
    vi = a @ a.mT + torch.eye(3, device=device) + .2 * a
    w = torch.rand(3, device=device) + .5
    cases = {
        'sqeuclidean': lambda p, q: tp.sqeuclidean_distances(p, q),
        'minkowski3': lambda p, q: tp.minkowski_distances(p, q, p=3., w=w.to(p)),
        'mahalanobis': lambda p, q: tp.mahalanobis_distances(p, q, VI=vi.to(p)),
        'snr': lambda p, q: tp.snr_distances(p, q),
    }
    for name, function in cases.items():
        p, q = x.clone().requires_grad_(), y.clone().requires_grad_()
        g = torch.rand(n, m, device=device)
        grads = torch.autograd.grad(function(p, q), (p, q), g)
        p64, q64 = x.double().requires_grad_(), y.double().requires_grad_()
        references = torch.autograd.grad(function(p64, q64), (p64, q64), g.double())
        for grad, reference in zip(grads, references):
            torch.testing.assert_close(grad.double(), reference, rtol=1e-4, atol=1e-4)
    p, q = angles.clone().requires_grad_(), others.clone().requires_grad_()
    grads = torch.autograd.grad(tp.haversine_distances(p, q).sum(), (p, q))
    p64, q64 = angles.double().requires_grad_(), others.double().requires_grad_()
    references = torch.autograd.grad(tp.haversine_distances(p64, q64).sum(), (p64, q64))
    for grad, reference in zip(grads, references):
        torch.testing.assert_close(grad.double(), reference, rtol=1e-4, atol=1e-4)


def test_weight_and_inverse_covariance_gradients(device):
    torch.manual_seed(0)
    x = torch.randn(4, 3, dtype=torch.float64, device=device)
    y = torch.randn(70, 3, dtype=torch.float64, device=device)
    w = (torch.rand(3, dtype=torch.float64, device=device) + .5).requires_grad_()
    torch.autograd.gradcheck(lambda v: tp.wminkowski_distances(x, y, p=3., w=v), (w,))
    a = torch.randn(3, 3, dtype=torch.float64, device=device)
    vi = (a @ a.mT + torch.eye(3, dtype=torch.float64, device=device)).requires_grad_()
    torch.autograd.gradcheck(
        lambda v: torch.ops.torchpairwise._sqmahalanobis(x, y, v), (vi,))


def _run_child(code):
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(code)],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize('metric,kernel', [('hellinger', False), ('jensenshannon', False),
                                           ('euclidean', False), ('chi2', True)])
def test_native_invalid_values_raise_and_keep_device_usable(device, metric, kernel):
    # A child process isolates the CUDA context in case validation regresses.
    out = _run_child(f'''
        import torch, torchpairwise
        x = torch.tensor([[1., -1.]], device={device!r})
        options = {{}} if {metric!r} != 'euclidean' else {{'w': -torch.ones(2, device={device!r})}}
        try:
            torch.ops.torchpairwise.builtin_comparisons(
                x, None, {metric!r}, 'all_pairs', {kernel}, **options)
            print('accepted')
        except RuntimeError:
            print('raised')
        print(float((torch.ones(2, device={device!r}) + 1).sum()))
    ''')
    assert out.split() == ['raised', '4.0']


@pytest.mark.parametrize('function,kwargs,event', [
    ('matern_kernel', {}, 2), ('rational_quadratic_kernel', {'alpha': 2.}, 2),
    ('sinkhorn_divergences', {'max_iter': 3}, 3), ('mmd_distances', {}, 3),
])
@pytest.mark.compiler
def test_fullgraph_dynamic_compile_accepts_scalar_options(function, kwargs, event):
    torch._dynamo.reset()
    shape = (4, 3) if event == 2 else (3, 5, 2)
    x = torch.randn(*shape)
    y = torch.randn(*((6, 3) if event == 2 else (2, 4, 2)))
    f = getattr(tp, function)
    compiled = torch.compile(lambda a, b: f(a, b, **kwargs), fullgraph=True,
                             dynamic=True, backend='eager')
    torch.testing.assert_close(compiled(x, y), f(x, y, **kwargs))


def test_native_autocast_policy_without_python_package(device):
    # Only the native library is loaded: no Python fake or autocast registration.
    dtype = 'torch.bfloat16' if device == 'cpu' else 'torch.float16'
    out = _run_child(f'''
        import importlib.machinery, importlib.util, os, torch
        root = importlib.util.find_spec('torchpairwise').submodule_search_locations[0]
        # This interpreter's binary; a checkout may hold builds for others.
        path = next(p for suffix in importlib.machinery.EXTENSION_SUFFIXES
                    if os.path.exists(p := os.path.join(root, '_C' + suffix)))
        torch.ops.load_library(path)
        ops = torch.ops.torchpairwise
        x = torch.rand(4, 3, device={device!r}).to({dtype})
        y = torch.rand(5, 3, device={device!r}).to({dtype})
        with torch.autocast({device!r}, dtype={dtype}):
            dtypes = [ops.canberra_distances(x, y).dtype, ops.snr_distances(x, y).dtype]
            same = ops.cosine_distances(x, x)
        print(*dtypes, same.dtype, bool((same.diagonal() == 0).all()))
    ''')
    assert out.split() == ['torch.float32'] * 3 + ['True']
