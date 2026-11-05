"""Spectral consumers: dense numerical oracles, explicit randomness and deployment."""
import gc
import io
import weakref

import pytest
import torch
import torchpairwise as tp
import torchpairwise as s
from torchpairwise.linalg import _packet


def test_trace_methods_rng_and_prompt_lifetime(device):
    torch.manual_seed(306)
    x = torch.randn(2, 9, 3, dtype=torch.float64, device=device)
    source = s.lazy(x, gamma=0.3, chunk_size=4, metric='rbf')
    dense = tp.rbf_kernel(x, gamma=.3)
    a = source.centered() + 0.2 * tp.eye_like(source.centered())
    centered = dense - dense.mean(-1, keepdim=True) - dense.mean(-2, keepdim=True) + dense.mean((-1,-2), keepdim=True)
    dense = centered + .2*torch.eye(9, device=device, dtype=x.dtype)
    exact = s.linalg.trace(a, method='exact', block_size=4)
    torch.testing.assert_close(exact.estimate, dense.diagonal(dim1=-2, dim2=-1).sum(-1))
    assert not exact.standard_error.any() and exact.samples.shape == (2, 0)
    torch.testing.assert_close(s.linalg.trace(source, method='exact').estimate, x.new_full((2,), 9.))
    # Explicit probes give directly checkable quadratic samples, including a
    # non-symmetric transpose and composition, not just SPD kernel matrices.
    matrix = torch.randn(2, 9, 9, dtype=x.dtype, device=device)
    probes = torch.randn(1, 9, 5, dtype=x.dtype, device=device)
    general = s.as_lazy(matrix).T @ s.as_lazy(matrix)
    result = s.linalg.trace(general, method='hutchinson', probes=probes, block_size=3)
    expected = (probes*(matrix.transpose(-1,-2) @ (matrix @ probes))).sum(-2)
    torch.testing.assert_close(result.samples, expected)
    torch.testing.assert_close(result.estimate, expected.mean(-1))
    torch.testing.assert_close(result.standard_error, expected.std(-1)/5**.5)
    rng = torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state()
    first = s.linalg.trace(a, method='hutchinson', num_probes=7, seed=41, block_size=3)
    second = s.linalg.trace(a, method='hutchinson', num_probes=7, seed=41, block_size=3)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state(), rng)
    one = s.linalg.trace(a, method='hutchinson', num_probes=1, seed=41)
    assert one.standard_error.isnan().all()
    # Verify immediate reference release even with automatic cycle GC disabled.
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        def bind_and_pack():
            features = x.clone()
            expression = (s.lazy(features, metric='rbf') + 0.5).square()
            linear = expression + 0.1 * tp.eye_like(expression)
            refs = [weakref.ref(t) for t in (features, expression, linear)]
            _packet(linear)
            return refs
        assert all(ref() is None for ref in bind_and_pack())
    finally:
        if was_enabled:
            gc.enable()


def test_logdet_quadrature_and_matrix_function_actions(device):
    torch.manual_seed(307)
    for dtype in (torch.float32, torch.float64):
        features = torch.randn(2, 7, 3, dtype=dtype, device=device)
        kernel = s.lazy(features, gamma=0.15, chunk_size=3, metric='rbf')
        a = kernel + 0.5 * tp.eye_like(kernel)
        dense = tp.rbf_kernel(features, gamma=.15) + .5*torch.eye(7, dtype=dtype, device=device)
        d, u = torch.linalg.eigh(dense.double())
        log_matrix = (u*d.log().unsqueeze(-2)) @ u.transpose(-1,-2)
        # Complete scaled coordinate probes make the trace average exact;
        # no statistical interpretation of independent samples is assumed.
        probes = 7**.5*torch.eye(7, device=device, dtype=dtype)
        fit = s.linalg.logdet(a, method='slq', probes=probes, max_iter=7, block_size=3)
        tolerance = dict(rtol=3e-5, atol=3e-6) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(fit.samples, (probes.double()*(log_matrix@probes.double())).sum(-2), **tolerance)
        torch.testing.assert_close(fit.estimate, d.log().sum(-1), **tolerance)
        torch.testing.assert_close(s.linalg.logdet(a, method='exact', block_size=3).estimate, d.log().sum(-1), **tolerance)
        assert fit.iterations.shape == (2, 7) and (fit.iterations <= 7).all()
        rhs = torch.randn(1, 7, 4, dtype=dtype, device=device)
        rhs[..., 0].zero_()
        exp = s.linalg.expm_multiply(a, rhs, scale=-.2, max_iter=7, block_size=3)
        root = s.linalg.inv_sqrt_multiply(a, rhs, max_iter=7, block_size=3)
        torch.testing.assert_close(exp.action.double(), torch.matrix_exp(-.2*dense.double())@rhs.double(), **tolerance)
        torch.testing.assert_close(root.action.double(), ((u*d.rsqrt().unsqueeze(-2))@u.transpose(-1,-2))@rhs.double(), **tolerance)
        assert exp.action.shape == (2, 7, 4) and exp.action.dtype == dtype
        assert not exp.action[..., 0].any() and not exp.iterations[..., 0].any()
        # A vector RHS and a symmetric indefinite exponential are also supported.
        indefinite = dense[0] - 2*torch.eye(7, device=device, dtype=dtype)
        vector = rhs[0, :, 1]
        action = s.linalg.expm_multiply(indefinite, vector, max_iter=7)
        torch.testing.assert_close(action.action.double(), torch.matrix_exp(indefinite.double())@vector.double(), **tolerance)
        assert action.error_estimate.shape == action.iterations.shape == ()
        one = s.linalg.expm_multiply(dense[0], vector, max_iter=1)
        assert one.error_estimate.isinf()
        three = s.linalg.expm_multiply(dense[0], vector, max_iter=3)
        four = s.linalg.expm_multiply(dense[0], vector, max_iter=4)
        torch.testing.assert_close(four.error_estimate,
            torch.linalg.vector_norm(four.action.double()-three.action.double()), **tolerance)
    # Independent exact breakdowns: a scalar multiple of I, a zero RHS and
    # an empty axis must not divide by a norm or poison another probe/column.
    diag = s.diag_embed(torch.full((2, 6), 4.0, device=device, dtype=torch.float64))
    rhs = torch.randn(3, 2, 6, 2, device=device, dtype=torch.float64)
    fit = s.linalg.inv_sqrt_multiply(diag, rhs)
    torch.testing.assert_close(fit.action, rhs/2)
    assert (fit.iterations == 1).all() and not fit.error_estimate.any()
    for n in (0, 3):
        empty = torch.empty(0, n, n, device=device, dtype=torch.float64)
        assert s.linalg.logdet(empty, method='exact').estimate.shape == (0,)
        assert s.linalg.logdet(empty, method='slq', num_probes=3).samples.shape == (0, 3)
    empty = torch.empty(0, 0, device=device, dtype=torch.float64)
    assert s.linalg.trace(empty, method='exact').estimate == 0
    assert s.linalg.logdet(empty, method='slq', num_probes=2).estimate == 0
    assert s.linalg.expm_multiply(empty, torch.empty(0, device=device, dtype=torch.float64)).action.shape == (0,)


def test_spectral_contracts_and_nonfinite(device):
    matrix = torch.eye(4, device=device, dtype=torch.float64)
    for options in ({'method': 'auto'}, {'method': 'hutchinson'},
                    {'method': 'exact', 'seed': 3}, {'method': 'hutchinson', 'num_probes': True},
                    {'method': 'hutchinson', 'num_probes': 0},
                    {'method': 'hutchinson', 'num_probes': 2, 'seed': -1},
                    {'method': 'exact', 'block_size': float('inf')},
                    {'method': 'hutchinson', 'probes': matrix, 'seed': 1}):
        with pytest.raises((ValueError, RuntimeError)):
            s.linalg.trace(matrix, **options)
    for probes in (torch.ones(3, 2, device=device, dtype=matrix.dtype),
                   torch.ones(2, 4, 2, device=device, dtype=matrix.dtype),
                   matrix.float(), matrix*float('nan')):
        with pytest.raises((ValueError, RuntimeError)):
            s.linalg.trace(matrix, method='hutchinson', probes=probes)
    with pytest.raises(ValueError, match='stochastic'):
        s.linalg.logdet(matrix, method='exact', max_iter=2)
    for bad in (-matrix, matrix*float('nan')):
        with pytest.raises(RuntimeError):
            s.linalg.logdet(bad, method='slq', probes=matrix)
        with pytest.raises(RuntimeError):
            s.linalg.logdet(bad, method='exact')
        with pytest.raises(RuntimeError):
            s.linalg.inv_sqrt_multiply(bad, matrix)
    nonsymmetric = matrix.clone(); nonsymmetric[0, 1] = .5
    with pytest.raises(RuntimeError, match='symmetric'):
        s.linalg.logdet(nonsymmetric, method='exact')
    for options in ({'max_iter': 0}, {'max_iter': True}, {'block_size': -1}, {'scale': float('nan')}):
        with pytest.raises((RuntimeError, ValueError)):
            s.linalg.expm_multiply(matrix, matrix, **options)
    grad = matrix.clone().requires_grad_()
    for operation in (lambda: s.linalg.trace(grad, method='exact'),
                      lambda: s.linalg.logdet(grad, method='exact'),
                      lambda: s.linalg.expm_multiply(grad, matrix),
                      lambda: s.linalg.inv_sqrt_multiply(matrix, grad)):
        with pytest.raises(RuntimeError, match='inference-only'):
            operation()
        with torch.no_grad():
            operation()


@pytest.mark.compiler
def test_spectral_compile_export_and_current_stream(device):
    class Consumers(torch.nn.Module):
        def forward(self, x, probes):
            a = s.lazy(x, gamma=0.2, chunk_size=3, metric='rbf') + 0.5 * tp.eye_like(s.lazy(x, gamma=0.2, chunk_size=3, metric='rbf'))
            return (s.linalg.trace(a, method='exact', block_size=3),
                    s.linalg.trace(a, method='hutchinson', probes=probes, block_size=3),
                    s.linalg.logdet(a, method='slq', probes=probes, max_iter=6, block_size=3),
                    s.linalg.expm_multiply(a, probes, scale=-.3, max_iter=6, block_size=3),
                    s.linalg.inv_sqrt_multiply(a, probes, max_iter=6, block_size=3))
    torch.manual_seed(309)
    module = Consumers()
    compiled = torch.compile(module, fullgraph=True, dynamic=True, backend='inductor')
    def inputs(n):
        return (torch.randn(n, 3, device=device, dtype=torch.float64),
                torch.randn(n, 4, device=device, dtype=torch.float64))
    for n in (7, 9):
        args = inputs(n)
        torch.testing.assert_close(compiled(*args), module(*args))
    args = inputs(7)
    n = torch.export.Dim('samples', min=6, max=15)
    exported = torch.export.export(module, args, dynamic_shapes=({0:n}, {0:n}))
    buf = io.BytesIO(); torch.export.save(exported, buf); buf.seek(0)
    restored = torch.export.load(buf).module()
    args = inputs(8)
    torch.testing.assert_close(restored(*args), module(*args))
    # Unseeded native randomness must survive compilation as runtime RNG work.
    class RandomTrace(torch.nn.Module):
        def forward(self, x):
            return s.linalg.trace(s.lazy(x, gamma=0.2, metric='rbf'), method='hutchinson', num_probes=5).samples
    random_module = RandomTrace()
    random_compiled = torch.compile(random_module, fullgraph=True, backend='inductor')
    torch.manual_seed(79); expected = random_module(args[0])
    torch.manual_seed(79); actual = random_compiled(args[0])
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(random_compiled(args[0]), actual)
    if device == 'cuda':
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = module(*args)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(result, module(*args))
