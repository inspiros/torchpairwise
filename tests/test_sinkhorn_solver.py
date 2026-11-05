"""Log-domain OT against independent dense, differentiable iterations."""
import pytest
import torch
import torchpairwise as tp


def _reference(cost, a, b, epsilon, reach=None, steps=1000):
    f, g = torch.zeros_like(a), torch.zeros_like(b)
    tau = 1. if reach is None else reach / (reach + epsilon)
    for _ in range(steps):
        f = -tau * epsilon * torch.logsumexp(b.log().unsqueeze(-2) +
            (g.unsqueeze(-2) - cost) / epsilon, -1)
        g = -tau * epsilon * torch.logsumexp(a.log().unsqueeze(-1) +
            (f.unsqueeze(-1) - cost) / epsilon, -2)
    plan = ((f.unsqueeze(-1) + g.unsqueeze(-2) - cost) / epsilon).exp() * a.unsqueeze(-1) * b.unsqueeze(-2)
    if reach is None:
        objective = (a * f).sum(-1) + (b * g).sum(-1)
    else:
        objective = -reach * ((a * torch.expm1(-f / reach)).sum(-1) +
                              (b * torch.expm1(-g / reach)).sum(-1))
        objective = objective - epsilon * (plan.sum((-2, -1)) - a.sum(-1) * b.sum(-1))
    return f, g, plan, objective


def _inputs(device):
    x = torch.tensor([[0., .1], [.4, .2], [.8, .6]], device=device, dtype=torch.float64)
    y = torch.tensor([[.1, .3], [.5, .7], [.9, .2], [.6, .4]], device=device, dtype=torch.float64)
    a = x.new_tensor([.2, .3, .5])
    b = x.new_tensor([.1, .2, .3, .4])
    return x, y, a, b


def _dense(x, y):
    return (x.unsqueeze(-2) - y.unsqueeze(-3)).square().sum(-1)


def test_sinkhorn_any_cost_and_streamed_plan(device):
    x, y, a, b = _inputs(device)
    dense = _dense(x, y)
    base = tp.lazy(x, y, 'sqeuclidean', chunk_size=2)
    custom = tp.make_pairwise(lambda u, v: (u-v).square().sum(-1), form='broadcast')
    for cost, expected in ((base, dense), (base.sqrt() + .1*base, dense.sqrt() + .1*dense),
                           (tp.lazy(x, y, custom, chunk_size=2), dense),
                           (tp.as_lazy(dense, chunk_size=2), dense), (dense, dense)):
        fit = tp.optimize.sinkhorn(cost, a, b, epsilon=.2, rtol=1e-10, check_every=1)
        _, _, plan, objective = _reference(expected, a, b, .2)
        actual = tp.optimize.sinkhorn_plan(cost, fit, a, b, epsilon=.2)
        assert fit.converged.all() and (fit.status == tp.optimize.SinkhornStatus.CONVERGED).all()
        torch.testing.assert_close(fit.objective, objective, rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(actual.sum(-1), a, rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(actual.sum(-2), b, rtol=1e-8, atol=1e-10)
        values = torch.arange(8, device=device, dtype=x.dtype).reshape(4, 2)
        torch.testing.assert_close(actual @ values, plan @ values, rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(fit.marginal_error, torch.zeros_like(fit.marginal_error), atol=1e-9, rtol=0)


def test_sinkhorn_objective_gradients(device):
    x, y, a, b = (v.requires_grad_() for v in _inputs(device))
    # Normalisation restricts mass perturbations to the balanced tangent space;
    # potentials, and therefore unconstrained mass gradients, have a gauge.
    aa, bb = a/a.sum(), b/b.sum()
    fit = tp.optimize.sinkhorn(tp.lazy(x, y, 'sqeuclidean', chunk_size=2), aa, bb,
                               epsilon=.3, rtol=1e-12, check_every=1)
    expected = _reference(_dense(x, y), aa, bb, .3, steps=300)[-1]
    actual_grad = torch.autograd.grad(fit.objective, (x, y, a, b), retain_graph=True)
    reference_grad = torch.autograd.grad(expected, (x, y, a, b))
    for actual, reference in zip(actual_grad, reference_grad):
        torch.testing.assert_close(actual, reference, rtol=1e-7, atol=1e-9)
    assert not fit.f.requires_grad and not fit.g.requires_grad


def test_sinkhorn_float32_plan_product(device):
    x, y, _, _ = (value.float() for value in _inputs(device))
    cost = tp.lazy(x, y, 'sqeuclidean', chunk_size=2)
    fit = tp.optimize.sinkhorn(cost, epsilon=.3)
    plan = tp.optimize.sinkhorn_plan(cost, fit, epsilon=.3)
    assert plan.dtype == x.dtype
    assert (plan @ y).dtype == y.dtype


def test_sinkhorn_unbalanced(device):
    x, y, a, b = _inputs(device)
    b = b * .7
    cost = _dense(x, y)
    fit = tp.optimize.sinkhorn(cost, a, b, epsilon=.2, reach=.6, rtol=1e-10, max_iter=2000)
    _, _, plan, objective = _reference(cost, a, b, .2, reach=.6)
    torch.testing.assert_close(fit.f, _reference(cost, a, b, .2, reach=.6)[0], rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(fit.objective, objective, rtol=1e-7, atol=1e-9)
    values = torch.eye(4, device=device, dtype=x.dtype)
    torch.testing.assert_close(tp.optimize.sinkhorn_plan(cost, fit, a, b, epsilon=.2) @ values, plan, rtol=1e-7, atol=1e-9)
    balanced = tp.optimize.sinkhorn(cost, a, b/.7, epsilon=.2, rtol=1e-10)
    large = tp.optimize.sinkhorn(cost, a, b/.7, epsilon=.2, reach=1e6, rtol=1e-9)
    torch.testing.assert_close(tp.optimize.sinkhorn_plan(cost, large, a, b/.7, epsilon=.2) @ values,
                               tp.optimize.sinkhorn_plan(cost, balanced, a, b/.7, epsilon=.2) @ values,
                               rtol=2e-5, atol=2e-7)


def test_sinkhorn_broadcast_zero_masses_and_initial(device):
    x, y, a, b = _inputs(device)
    cost = torch.stack((_dense(x, y), _dense(x, y)*.7))
    a = a.expand(2, -1).clone(); a[:, 0] = 0; a = a/a.sum(-1, keepdim=True)
    b = b.clone(); b[1] = 0; b = b/b.sum()
    fit = tp.optimize.sinkhorn(cost, a, b, epsilon=.3, rtol=1e-10, check_every=1)
    _, _, plan, objective = _reference(cost, a, b, .3)
    assert fit.f.shape == (2, 3) and fit.g.shape == (2, 4)
    assert fit.converged.all() and torch.isfinite(fit.objective).all()
    torch.testing.assert_close(fit.objective, objective, rtol=1e-8, atol=1e-10)
    actual = tp.optimize.sinkhorn_plan(cost, fit, a, b, epsilon=.3)
    torch.testing.assert_close(actual.sum(-1), plan.sum(-1), rtol=1e-8, atol=1e-10)
    warm = tp.optimize.sinkhorn(cost, a, b, epsilon=.3, initial=(fit.f, fit.g), rtol=1e-10, check_every=1)
    assert (warm.iterations <= fit.iterations).all()


def test_sinkhorn_scaling_and_divergence(device):
    x, y, a, b = _inputs(device)
    cost = _dense(x, y)
    plain = tp.optimize.sinkhorn(cost, a, b, epsilon=.01, rtol=1e-8, max_iter=5000, check_every=1)
    scaled = tp.optimize.sinkhorn(cost, a, b, epsilon=.01, epsilon_scaling=.5,
                                  rtol=1e-8, max_iter=5000, check_every=1)
    assert plain.converged.all() and scaled.converged.all()
    torch.testing.assert_close(scaled.objective, plain.objective, rtol=1e-6, atol=1e-8)
    assert (scaled.iterations <= plain.iterations).all()
    xx, yy = _dense(x, x), _dense(y, y)
    divergence = tp.optimize.sinkhorn_divergence(cost, xx, yy, a, b, epsilon=.2, rtol=1e-10)
    identical = tp.optimize.sinkhorn_divergence(xx, xx, xx, a, a, epsilon=.2, rtol=1e-10)
    assert (divergence >= -1e-8).all()
    torch.testing.assert_close(identical, torch.zeros_like(identical), rtol=0, atol=1e-10)


def test_sinkhorn_validation(device):
    x, y, a, b = _inputs(device)
    cost = _dense(x, y)
    for kwargs in ({'epsilon': 0.}, {'epsilon': -.1}):
        with pytest.raises((ValueError, RuntimeError), match='epsilon'):
            tp.optimize.sinkhorn(cost, a, b, **kwargs)
    with pytest.raises((ValueError, RuntimeError), match='total|mass|balanced'):
        tp.optimize.sinkhorn(cost, a, b*2, epsilon=.2)
    with pytest.raises((ValueError, RuntimeError), match='nonnegative|negative'):
        tp.optimize.sinkhorn(cost, -a, b, epsilon=.2)
    with pytest.raises((TypeError, ValueError, RuntimeError), match='entrywise|scalar|linear'):
        tp.optimize.sinkhorn(tp.eye_like(tp.as_lazy(cost[:3, :3])), epsilon=.2)


@pytest.mark.compiler
def test_sinkhorn_compile_export(device):
    class Solve(torch.nn.Module):
        def forward(self, x, y):
            return tp.optimize.sinkhorn(tp.lazy(x, y, 'sqeuclidean', chunk_size=2), epsilon=.3,
                                         rtol=1e-8, max_iter=500)
    model = Solve()
    x, y, _, _ = _inputs(device)
    expected = model(x, y)
    compiled = torch.compile(model, fullgraph=True, backend='inductor')
    torch.testing.assert_close(compiled(x, y), expected)
    exported = torch.export.export(model, (x, y)).module()
    torch.testing.assert_close(exported(x, y), expected)


@pytest.mark.parametrize('epsilon_scaling', [None, .5])
def test_sinkhorn_fused_cuda_matches_tile(device, epsilon_scaling):
    if torch.device(device).type != 'cuda':
        pytest.skip('fused reducers are CUDA only')
    ops = torch.ops.torchpairwise
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(2, 300, 3, generator=generator).to(device)
    y = torch.rand(2, 257, 3, generator=generator).to(device)
    a = torch.rand(2, 300, generator=generator).to(device)
    b = torch.rand(257, generator=generator).to(device)
    a, b = a / a.sum(-1, keepdim=True), b / b.sum()
    for make in (lambda: tp.lazy(x, y, 'sqeuclidean'), lambda: .5 * tp.lazy(x, y, 'manhattan') + .1):
        fits = []
        for enabled in (False, True):
            previous = ops._set_fused_reducers(enabled)
            try:
                before = ops._fused_reducer_calls()
                fits.append(tp.optimize.sinkhorn(make(), a, b, epsilon=.05, rtol=1e-6,
                                                 epsilon_scaling=epsilon_scaling, check_every=5))
                torch.cuda.synchronize()
                assert (ops._fused_reducer_calls() > before) == enabled
            finally:
                ops._set_fused_reducers(previous)
        tile, fused = fits
        assert tile.converged.all() and fused.converged.all()
        torch.testing.assert_close(fused.f, tile.f, rtol=0, atol=1e-4)
        torch.testing.assert_close(fused.g, tile.g, rtol=0, atol=1e-4)
        torch.testing.assert_close(fused.objective, tile.objective, rtol=1e-5, atol=1e-7)
    if epsilon_scaling is not None:
        # Potentials near 1/epsilon cancel the cost term; float32 gaps stay
        # accurate enough to meet a tight marginal tolerance.
        fit = tp.optimize.sinkhorn(tp.lazy(x, y, 'sqeuclidean'), a, b, epsilon=.002, rtol=1e-6,
                                   epsilon_scaling=epsilon_scaling, max_iter=5000, check_every=5)
        assert fit.converged.all()
