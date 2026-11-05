"""First estimate derivatives and targeted common-power regression oracles."""
import io

import pytest
import torch
import torchpairwise as tp
import torchpairwise as s


def test_trace_estimate_gradients_and_probe_snapshots(device):
    torch.manual_seed(321)
    x = torch.randn(2, 6, 3, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, 6, 3, device=device, dtype=x.dtype, requires_grad=True)
    weights = torch.randn(6, device=device, dtype=x.dtype, requires_grad=True)
    scale = torch.tensor(.7, device=device, dtype=x.dtype, requires_grad=True)
    source = (s.lazy(x, y, metric='cosine_similarity', chunk_size=3) + 0.3).weight_rows(weights)
    a = scale * source.centered()
    dense = (tp.cosine_similarity(x, y)+.3)*weights[:, None]
    dense = scale*(dense-dense.mean(-1, keepdim=True)-dense.mean(-2, keepdim=True)+dense.mean((-1,-2), keepdim=True))
    probes = torch.randn(1, 6, 5, device=device, dtype=x.dtype)
    parameters = (x, y, weights, scale)
    upstream = x.new_tensor([.3, -.7])
    for method in ('exact', 'hutchinson'):
        kwargs = {} if method == 'exact' else {'probes': probes.clone()}
        actual = s.linalg.trace(a, method=method, differentiable=True, block_size=2, **kwargs)
        samples = (probes*(dense@probes)).sum(-2)
        expected = dense.diagonal(dim1=-2, dim2=-1).sum(-1) if method == 'exact' else samples.mean(-1)
        torch.testing.assert_close(actual.estimate, expected)
        assert actual.estimate.requires_grad and not actual.samples.requires_grad and not actual.standard_error.requires_grad
        if method == 'hutchinson':
            kwargs['probes'].add_(42.)  # Backward uses its fixed forward snapshot.
        gradients = torch.autograd.grad((actual.estimate*upstream).sum(), parameters)
        reference = torch.autograd.grad((expected*upstream).sum(), parameters, retain_graph=True)
        torch.testing.assert_close(gradients, reference, rtol=1e-9, atol=1e-10)
    # Generated probes exactly preserve the inference RNG contract; backward
    # consumes no random state and can replay the same saved probes twice.
    for seed in (17, None):
        torch.manual_seed(43)
        with torch.no_grad():
            reference = s.linalg.trace(a, method='hutchinson', num_probes=5, seed=seed, block_size=2)
        torch.manual_seed(43)
        actual = s.linalg.trace(a, method='hutchinson', num_probes=5, seed=seed, block_size=2, differentiable=True)
        torch.testing.assert_close(actual, reference)
        state = torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state()
        first = torch.autograd.grad(actual.estimate.sum(), parameters, retain_graph=True)
        second = torch.autograd.grad(actual.estimate.sum(), parameters)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state(), state)


def test_logdet_exact_and_explicit_solve_gradients(device):
    torch.manual_seed(322)
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(2, 7, 3, device=device, dtype=dtype, requires_grad=True)
        scale = torch.tensor(.7, device=device, dtype=dtype, requires_grad=True)
        shift = torch.tensor(.8, device=device, dtype=dtype, requires_grad=True)
        a = scale * s.lazy(x, gamma=0.3, chunk_size=3, metric='rbf') + shift * tp.eye_like(scale * s.lazy(x, gamma=0.3, chunk_size=3, metric='rbf'))
        dense = scale*tp.rbf_kernel(x, gamma=.3)+shift*torch.eye(7, device=device, dtype=dtype)
        params = (x, scale, shift)
        tolerance = dict(rtol=3e-4, atol=2e-5) if dtype == torch.float32 else dict(rtol=1e-8, atol=1e-10)
        exact = s.linalg.logdet(a, method='exact', block_size=3, differentiable=True)
        target = torch.linalg.slogdet(dense.double()).logabsdet
        torch.testing.assert_close(exact.estimate, target)
        torch.testing.assert_close(torch.autograd.grad(exact.estimate.sum(), params),
            torch.autograd.grad(target.sum(), params, retain_graph=True), **tolerance)
        for probes in (7**.5*torch.eye(7, device=device, dtype=dtype),
                       torch.randn(1, 7, 3, device=device, dtype=dtype)):
            fit = s.linalg.logdet(a, method='slq', probes=probes, max_iter=2,
                block_size=2, differentiable=True, gradient='solve',
                solve_rtol=1e-6 if dtype == torch.float32 else 1e-11, solve_max_iter=100)
            # Reference the explicitly documented trace/solve estimator, not
            # the derivative of a depth-two quadrature approximation.
            solved = torch.linalg.solve(dense.double(), probes.double()).detach().to(dtype)
            surrogate = (solved*(dense@probes)).sum()/probes.shape[-1]
            torch.testing.assert_close(torch.autograd.grad(fit.estimate.sum(), params),
                torch.autograd.grad(surrogate, params, retain_graph=True), **tolerance)
            assert not fit.samples.requires_grad and not fit.standard_error.requires_grad
            assert not fit.iterations.requires_grad
    # Empty systems return zero derivatives of the empty log-determinant/trace.
    empty = torch.empty(0, 0, device=device, dtype=torch.float64, requires_grad=True)
    for operation in (lambda: s.linalg.trace(empty, method='exact', differentiable=True),
                      lambda: s.linalg.logdet(empty, method='exact', differentiable=True),
                      lambda: s.linalg.logdet(empty, method='slq', num_probes=2,
                           differentiable=True, gradient='solve')):
        result = operation()
        assert result.estimate == 0
        assert torch.autograd.grad(result.estimate, empty)[0].shape == (0, 0)


def test_spectral_gradient_contracts_and_failures(device):
    matrix = torch.eye(4, device=device, dtype=torch.float64, requires_grad=True)
    for kwargs in ({'differentiable': True}, {'differentiable': True, 'gradient': 'slq'},
                   {'differentiable': False, 'gradient': 'solve'},
                   {'differentiable': True, 'gradient': 'solve', 'solve_rtol': -1}):
        with pytest.raises(ValueError):
            s.linalg.logdet(matrix, method='slq', num_probes=2, **kwargs)
    with pytest.raises(ValueError, match='only used'):
        s.linalg.logdet(matrix, method='exact', differentiable=True, solve_max_iter=10)
    with pytest.raises(RuntimeError, match='fixed algorithmic'):
        s.linalg.trace(matrix, method='hutchinson', probes=matrix, differentiable=True)
    failed = s.linalg.logdet(matrix, method='slq', num_probes=2, seed=5,
        differentiable=True, gradient='solve', solve_max_iter=0)
    with pytest.raises(RuntimeError, match='converged'):
        failed.estimate.backward()
    result = s.linalg.trace(matrix, method='exact', differentiable=True)
    with pytest.raises(RuntimeError, match='Higher derivatives'):
        torch.autograd.grad(result.estimate, matrix, create_graph=True)
    result = s.linalg.trace(matrix, method='exact', differentiable=True)
    with torch.no_grad():
        matrix.add_(.1)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        result.estimate.backward()


def test_common_power_specializations_modes_values_and_gradients(device):
    torch.manual_seed(323)
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(6, 7, device=device, dtype=dtype, requires_grad=True)
        y = torch.randn(4, 7, device=device, dtype=dtype, requires_grad=True)
        for p in (1., 2., 1.7):
            actual = tp.minkowski_distances(x, y, p=p).pow(p)
            reference = (x[:, None]-y[None]).abs().pow(p).sum(-1)
            torch.testing.assert_close(actual, reference)
            torch.testing.assert_close(torch.autograd.grad(actual.sum(), (x, y), retain_graph=True),
                torch.autograd.grad(reference.sum(), (x, y), retain_graph=True))
        square = tp.sqeuclidean_distances(x)
        torch.testing.assert_close(tp.compare(x, x, metric='sqeuclidean', mode='paired'), square.diagonal())
        i, j = torch.triu_indices(6, 6, offset=1, device=device)
        torch.testing.assert_close(tp.compare(x, metric='sqeuclidean', mode='condensed'), square[i, j])
        torch.testing.assert_close(tp.compare(x[0], x[1], metric='sqeuclidean', mode='single'), square[0, 1])
        # Direct subtraction preserves small separations at a large offset.
        offset = 1e5 if dtype == torch.float32 else 1e12
        shifted = x.detach()+offset
        expected = (shifted[:, None]-shifted[None]).square().sum(-1)
        torch.testing.assert_close(tp.sqeuclidean_distances(shifted), expected)
        nonfinite = shifted.clone(); nonfinite[0, 0] = float('inf')
        torch.testing.assert_close(tp.sqeuclidean_distances(nonfinite),
            (nonfinite[:, None]-nonfinite[None]).square().sum(-1), equal_nan=True)
        values = torch.randn(4, 3, device=device, dtype=dtype)
        factorized = (s.as_lazy(x) @ s.as_lazy(y).T).matmul(values)
        torch.testing.assert_close(factorized, x@(y.T@values))
        torch.testing.assert_close(factorized, tp.linear_kernel(x, y)@values, rtol=1e-5, atol=2e-5)


@pytest.mark.compiler
def test_spectral_estimate_gradients_compile_export_and_stream(device):
    class Model(torch.nn.Module):
        def forward(self, x, scale, shift, probes):
            a = scale.exp() * s.lazy(x.sin(), gamma=0.2, chunk_size=3, metric='rbf') + (shift.exp() + 0.5) * tp.eye_like(scale.exp() * s.lazy(x.sin(), gamma=0.2, chunk_size=3, metric='rbf'))
            return (s.linalg.trace(a, method='exact', block_size=3, differentiable=True),
                    s.linalg.trace(a, method='hutchinson', probes=probes, block_size=2, differentiable=True),
                    s.linalg.logdet(a, method='exact', block_size=3, differentiable=True),
                    s.linalg.logdet(a, method='slq', probes=probes, max_iter=2, block_size=2,
                        differentiable=True, gradient='solve', solve_rtol=1e-10, solve_max_iter=100))
    def inputs(n):
        return (torch.randn(3, n, device=device, dtype=torch.float64).T.requires_grad_(),
                torch.tensor(-.2, device=device, dtype=torch.float64, requires_grad=True),
                torch.tensor(-.4, device=device, dtype=torch.float64, requires_grad=True),
                torch.randn(4, n, device=device, dtype=torch.float64).T)
    def check(actual, expected, args):
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(sum(r.estimate for r in actual), args[:3])
        ge = torch.autograd.grad(sum(r.estimate for r in expected), args[:3])
        torch.testing.assert_close(ga, ge, rtol=1e-7, atol=1e-9)
    torch.manual_seed(324)
    model = Model()
    compiled = torch.compile(model, fullgraph=True, dynamic=True, backend='inductor')
    for n in (6, 8):
        args = inputs(n)
        check(compiled(*args), model(*args), args)
    n = torch.export.Dim('samples', min=4, max=12)
    exported = torch.export.export(model, inputs(6), dynamic_shapes=({0:n}, {}, {}, {0:n}))
    buffer = io.BytesIO(); torch.export.save(exported, buffer); buffer.seek(0)
    restored = torch.export.load(buffer).module()
    args = inputs(7)
    check(restored(*args), model(*args), args)
    class RandomTrace(torch.nn.Module):
        def forward(self, x):
            return s.linalg.trace(s.lazy(x, gamma=0.2, chunk_size=3, metric='rbf'), method='hutchinson', num_probes=5, block_size=2, differentiable=True).estimate
    random_model = RandomTrace()
    random_compiled = torch.compile(random_model, fullgraph=True, dynamic=True, backend='inductor')
    x = inputs(7)[0]
    torch.manual_seed(83); actual = random_compiled(x)
    torch.manual_seed(83); expected = random_model(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, x), torch.autograd.grad(expected, x))
    if device == 'cuda':
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            args = inputs(7)
            actual = model(*args)
            ga = torch.autograd.grad(sum(r.estimate for r in actual), args[:3])
        torch.cuda.current_stream().wait_stream(stream)
        expected = model(*args)
        ge = torch.autograd.grad(sum(r.estimate for r in expected), args[:3])
        torch.testing.assert_close(ga, ge, rtol=1e-7, atol=1e-9)
