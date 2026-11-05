"""Dense independent oracles for streamed geometric interactions."""
import io

import pytest
import torch
import torchpairwise as tp


def _dense_map(x, y, a, b, epsilon, iterations, p):
    cost = torch.linalg.vector_norm(x.unsqueeze(-2) - y.unsqueeze(-3), dim=-1).pow(p) / epsilon
    a = torch.ones_like(x[..., 0]) if a is None else a
    b = torch.ones_like(y[..., 0]) if b is None else b
    loga = torch.where(a > 0, torch.where(a > 0, a, 1.).log() - a.sum(-1, keepdim=True).log(), -torch.inf)
    logb = torch.where(b > 0, torch.where(b > 0, b, 1.).log() - b.sum(-1, keepdim=True).log(), -torch.inf)
    v = torch.zeros_like(logb)
    for _ in range(iterations):
        u = loga - torch.logsumexp(v.unsqueeze(-2) - cost, -1)
        v = logb - torch.logsumexp(u.unsqueeze(-1) - cost, -2)
    result = (v.unsqueeze(-2) - cost).softmax(-1) @ y
    return torch.where((a > 0).unsqueeze(-1), result, 0.)


@pytest.mark.parametrize('kernel,kwargs', [('rbf', {'gamma': .4}), ('polynomial', {'degree': 3}), ('matern', {})])
@pytest.mark.parametrize('normalize', [False, True])
def test_displacement_matches_dense_values_and_gradients(device, kernel, kwargs, normalize):
    torch.manual_seed(2)
    # Batch broadcast, strided inputs, multiple sample/channel tiles.
    x = torch.randn(2, 1, 5, 6, device=device, dtype=torch.float64)[..., ::2].requires_grad_()
    y = torch.randn(1, 3, 7, 6, device=device, dtype=torch.float64)[..., ::2].requires_grad_()
    k = tp.pairwise_kernels(x, y, kernel=kernel, **kwargs)
    expected = (k.unsqueeze(-1) * (y.unsqueeze(-3) - x.unsqueeze(-2))).sum(-2)
    if normalize:
        expected = expected / k.sum(-1, keepdim=True)
    actual = tp.lazy(x, y, chunk_size=2, **kwargs, metric=kernel).displacement(normalize=normalize)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(tp.lazy(x, y, metric=kernel, chunk_size=2, **kwargs).displacement(normalize=normalize), expected)
    g = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, (x, y), g)
    references = torch.autograd.grad(expected, (x, y), g)
    for grad, reference in zip(grads, references):
        torch.testing.assert_close(grad, reference)


@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('zero_mass', [False, True])
def test_sinkhorn_map_dense_values_gradients_and_chunk_invariance(device, p, zero_mass):
    torch.manual_seed(4)
    x = torch.randn(2, 1, 5, 4, device=device, dtype=torch.float64)[..., ::2].requires_grad_()
    y = torch.randn(1, 3, 7, 4, device=device, dtype=torch.float64)[..., ::2].requires_grad_()
    a = torch.rand(5, device=device, dtype=torch.float64) + .2
    b = torch.rand(3, 7, device=device, dtype=torch.float64) + .2
    if zero_mass:
        a[1:3] = 0
        b[..., :3] = 0  # Entire target tile has -inf duals.
    a.requires_grad_(); b.requires_grad_()
    options = dict(source_weights=a, target_weights=b, regularization=.7, max_iter=4, p=p)
    actual = tp.sinkhorn_barycentric_map(x, y, chunk_size=2, **options)
    expected = _dense_map(x, y, a, b, .7, 4, p)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(tp.sinkhorn_barycentric_map(x, y, chunk_size=32, **options), expected)
    displacement = tp.sinkhorn_displacement(x, y, chunk_size=3, **options)
    torch.testing.assert_close(displacement, torch.where((a > 0).unsqueeze(-1), expected - x, 0.))
    g = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, (x, y, a, b), g)
    references = torch.autograd.grad(expected, (x, y, a, b), g)
    for grad, reference in zip(grads, references):
        torch.testing.assert_close(grad, reference, rtol=2e-8, atol=2e-9)


def test_sinkhorn_gradcheck_and_converged_marginal_mean(device):
    torch.manual_seed(9)
    x = (.1 * torch.randn(3, 2, device=device, dtype=torch.float64)).requires_grad_()
    y = (.1 * torch.randn(4, 2, device=device, dtype=torch.float64)).requires_grad_()
    a = torch.tensor([1., 2., 3.], device=device, dtype=torch.float64, requires_grad=True)
    b = torch.tensor([2., 3., 1., 2.], device=device, dtype=torch.float64, requires_grad=True)
    function = lambda xx, yy, aa, bb: tp.sinkhorn_barycentric_map(xx, yy, source_weights=aa, target_weights=bb, regularization=.5, max_iter=5, chunk_size=2)
    assert torch.autograd.gradcheck(function, (x, y, a, b))
    mapped = function(x, y, a, b)
    torch.testing.assert_close(((a/a.sum()).unsqueeze(-1) * mapped).sum(-2), ((b/b.sum()).unsqueeze(-1) * y).sum(-2), atol=1e-8, rtol=1e-6)


def test_empty_zero_rows_and_native_autocast(device):
    x = torch.randn(3, 2, device=device)
    y = torch.randn(0, 2, device=device)
    for normalize in (False, True):
        torch.testing.assert_close(tp.lazy(x, y, metric='rbf').displacement(normalize=normalize), torch.zeros_like(x))
    y = torch.tensor([[1., 2.]], device=device)
    torch.testing.assert_close(tp.sinkhorn_barycentric_map(x, y, max_iter=2), y.expand_as(x))
    dtype = torch.bfloat16 if device == 'cpu' else torch.float16
    with torch.autocast(device, dtype=dtype):
        result = torch.ops.torchpairwise.sinkhorn_map(x.to(dtype), y.to(dtype), max_iter=2)
        assert result.dtype == torch.float32
        assert tp.lazy(x.to(dtype), y.to(dtype), metric='rbf').displacement(normalize=False).dtype == torch.float32
    with pytest.raises(RuntimeError, match='Higher derivatives'):
        xx = x.requires_grad_()
        torch.autograd.grad(tp.sinkhorn_barycentric_map(xx, y, max_iter=2).sum(), xx, create_graph=True)


@pytest.mark.parametrize('options', [dict(source_weights=torch.zeros(3)), dict(target_weights=torch.tensor([1., -1., 2., 3.])), dict(regularization=0), dict(max_iter=0), dict(p=3), dict(chunk_size=0)])
def test_invalid_map_options(device, options):
    options = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in options.items()}
    with pytest.raises((ValueError, RuntimeError)):
        tp.sinkhorn_barycentric_map(torch.randn(3, 2, device=device), torch.randn(4, 2, device=device), **options)
    assert (torch.ones(1, device=device) + 1).item() == 2


def test_sinkhorn_saves_vectors_not_pair_matrices(device):
    x = torch.randn(17, 3, device=device, requires_grad=True)
    y = torch.randn(19, 3, device=device, requires_grad=True)
    saved = []
    def pack(t):
        saved.append(tuple(t.shape))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t:t):
        tp.sinkhorn_barycentric_map(x, y, max_iter=4, chunk_size=5).sum().backward()
    # Forward solver saves inputs/dual vectors/results. Backward cost replay
    # may save a cost tile, but never a full pair or pair-feature matrix.
    assert not any(17 in shape and 19 in shape for shape in saved)


@pytest.mark.compiler
def test_interaction_compile_export_and_opcheck(device):
    class Map(torch.nn.Module):
        def forward(self, x, y, a, b):
            return (tp.sinkhorn_displacement(x, y, source_weights=a, target_weights=b, max_iter=3, regularization=.8, chunk_size=2),
                    tp.lazy(x, y, gamma=0.2, chunk_size=2, metric='rbf').displacement(normalize=True))
    torch._dynamo.reset()
    args = (torch.randn(3, 2, device=device), torch.randn(4, 2, device=device), torch.rand(3, device=device) + .2, torch.rand(4, device=device) + .2)
    model = Map()
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(*args), model(*args))
    x,y,a,b = [t.detach().requires_grad_() for t in args]
    compiled_grads = torch.autograd.grad(sum(v.square().sum() for v in compiled(x,y,a,b)), (x,y,a,b))
    eager_grads = torch.autograd.grad(sum(v.square().sum() for v in model(x,y,a,b)), (x,y,a,b))
    torch.testing.assert_close(compiled_grads, eager_grads)
    n = torch.export.Dim('queries', min=2, max=8)
    m = torch.export.Dim('references', min=2, max=8)
    exported = torch.export.export(model, args, dynamic_shapes=({0:n},{0:m},{0:n},{0:m}))
    buffer = io.BytesIO(); torch.export.save(exported, buffer); buffer.seek(0)
    restored = torch.export.load(buffer).module()
    resized = (torch.randn(6, 2, device=device), torch.randn(2, 2, device=device), torch.ones(6, device=device), torch.ones(2, device=device))
    torch.testing.assert_close(restored(*resized), model(*resized))
    torch.library.opcheck(torch.ops.torchpairwise._transport_map_reduce.default,
        (args[0].requires_grad_(), args[1].requires_grad_(), args[3].log().requires_grad_(), 2, .8, 2, False))
