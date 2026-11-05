"""Native streamed kernel operators against explicit small Gram-matrix oracles."""
import io
import weakref
import pytest
import torch
import torchpairwise as tp


KERNELS = [
    ('rbf', dict(gamma=.4)), ('linear', {}),
    ('polynomial', dict(gamma=.2, degree=3, coef0=.7)),
    ('sigmoid', dict(gamma=.3, coef0=.2)), ('laplacian', dict(gamma=.5)),
    ('cosine', {}), ('chi2', dict(gamma=.3)), ('additive_chi2', {}),
    ('histogram_intersection', {}), ('bhattacharyya', {}),
    ('rational_quadratic', dict(length_scale=.8, alpha=.7)),
    ('matern', dict(length_scale=.9, nu=2.5)),
]


@pytest.mark.parametrize('kernel,options', KERNELS)
def test_kernel_forms_catalog_broadcast_values_and_gradients(kernel, options, device):
    torch.manual_seed(107)
    x = (torch.rand(2, 1, 3, 3, device=device, dtype=torch.float64) + .2).requires_grad_()
    y = (torch.rand(1, 3, 5, 3, device=device, dtype=x.dtype) + .3).requires_grad_()
    left = torch.rand(2, 1, 3, 2, device=device, dtype=x.dtype, requires_grad=True)
    right = torch.rand(3, 5, 3, device=device, dtype=x.dtype, requires_grad=True)
    operator = tp.lazy(x, y, metric=kernel, chunk_size=4, **options)
    assert isinstance(operator, tp.LazyPairwise)
    assert operator.shape == (2, 3, 3, 5)
    assert operator.dtype == x.dtype and operator.device == x.device
    gram = tp.pairwise_kernels(x, y, kernel=kernel, **options)
    expected = left.transpose(-1, -2) @ gram @ right
    actual = operator.bilinear(left, right)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(tp.lazy(x, y, chunk_size=4, **options, metric=kernel).bilinear(left, right), expected)
    weights = torch.randn_like(expected)
    for a, e in zip(torch.autograd.grad((actual * weights).sum(), (x, y, left, right)),
                    torch.autograd.grad((expected * weights).sum(), (x, y, left, right), retain_graph=True)):
        torch.testing.assert_close(a, e)
    torch.testing.assert_close(operator @ right, gram @ right)
    torch.testing.assert_close(operator.T @ left, gram.transpose(-1, -2) @ left)
    torch.testing.assert_close(operator.diagonal(), gram.diagonal(dim1=-2, dim2=-1))
    rows = torch.tensor([2, 0, 2], device=device)
    columns = torch.tensor([4, 1], device=device)
    block = operator.block(rows, columns)
    expected = gram.index_select(-2, rows).index_select(-1, columns)
    torch.testing.assert_close(block, expected)
    for a, e in zip(torch.autograd.grad(block.sum(), (x, y)),
                    torch.autograd.grad(expected.sum(), (x, y))):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize('reduction', ['sum', 'mean', 'min', 'max', 'prod', 'all', 'any'])
def test_kernel_reductions_axes_and_gradients(reduction, device):
    torch.manual_seed(319)
    x = (torch.rand(2, 3, 2, dtype=torch.float64, device=device) + .2).requires_grad_()
    y = (torch.rand(1, 4, 2, dtype=x.dtype, device=device) + .5).requires_grad_()
    operator = tp.lazy(x, y, metric='rbf', gamma=0.4, chunk_size=2)
    gram = tp.rbf_kernel(x, y, gamma=.4)
    for dim in (None, -1, -2):
        for keepdim in (False, True):
            source = gram.flatten(-2) if dim is None else gram
            axis = -1 if dim is None else dim
            if reduction in ('min', 'max'):
                expected = getattr(source, reduction)(dim=axis, keepdim=keepdim).values
            else:
                expected = getattr(source, reduction)(dim=axis, keepdim=keepdim)
            if dim is None and keepdim:
                expected = expected.unsqueeze(-1)
            actual = getattr(operator, {'min': 'amin', 'max': 'amax'}.get(reduction, reduction))(dim=dim, keepdim=keepdim)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(tp.lazy(x, y, gamma=0.4, chunk_size=2, metric='rbf').reduce(reduction, dim=dim, keepdim=keepdim), expected)
            if reduction not in ('all', 'any'):
                for a, e in zip(torch.autograd.grad(actual.sum(), (x, y), retain_graph=True),
                                torch.autograd.grad(expected.sum(), (x, y), retain_graph=True)):
                    torch.testing.assert_close(a, e)


def test_kernel_quadratic_vectors_ties_zeros_and_empty(device):
    torch.manual_seed(419)
    x = torch.rand(4, 3, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(5, 3, device=device, dtype=x.dtype, requires_grad=True)
    u = torch.rand(4, device=device, dtype=x.dtype, requires_grad=True)
    v = torch.rand(5, device=device, dtype=x.dtype, requires_grad=True)
    K = tp.lazy(x, y, metric='linear', chunk_size=2)
    gram = x @ y.T
    for a, b in ((u, v), (u[:, None], v), (u, v[:, None]), (u[:, None], v[:, None])):
        torch.testing.assert_close(K.bilinear(a, b), (a.T if a.ndim == 2 else a) @ gram @ b)
    torch.testing.assert_close(K @ v, gram @ v)
    actual = tp.lazy(x, chunk_size=2, metric='linear').quadratic(u)
    expected = u @ (x @ x.T) @ u
    torch.testing.assert_close(actual, expected)
    for a, e in zip(torch.autograd.grad(actual, (x, u)), torch.autograd.grad(expected, (x, u))):
        torch.testing.assert_close(a, e)
    # Global min/max tie routing follows the first flattened matrix position.
    a = torch.tensor([[1.], [1.], [2.]], device=device, dtype=x.dtype, requires_grad=True)
    b = torch.tensor([[1.], [1.]], device=device, dtype=x.dtype, requires_grad=True)
    for operation in ('min', 'max'):
        # torch.min() without dim splits ties; index-reduction chooses the first.
        dense = getattr((a @ b.T).flatten(), operation)(dim=0).values
        actual = getattr(tp.lazy(a, b, metric='linear', chunk_size=1), operation)()
        for value, reference in zip(torch.autograd.grad(actual, (a, b)), torch.autograd.grad(dense, (a, b))):
            torch.testing.assert_close(value, reference)
    for zero_count in (1, 2):
        a = torch.tensor([[0.], [2.]], device=device, dtype=x.dtype, requires_grad=True)
        b = torch.ones(zero_count, 1, device=device, dtype=x.dtype, requires_grad=True)
        actual = tp.lazy(a, b, metric='linear', chunk_size=1).prod()
        expected = (a @ b.T).prod()
        for value, reference in zip(torch.autograd.grad(actual, (a, b)), torch.autograd.grad(expected, (a, b))):
            torch.testing.assert_close(value, reference)
    for n, m in ((0, 5), (4, 0), (0, 0)):
        a, b = x[:n], y[:m]
        operator = tp.lazy(a, b, chunk_size=2, metric='rbf')
        assert operator.sum().item() == 0
        assert operator.prod().item() == 1
        assert operator.all().item() and not operator.any().item()
        assert operator.mean().isnan().item()
        assert operator.diagonal().numel() == 0
        actual = operator.bilinear(u[:n], v[:m])
        assert actual.item() == 0
        for g in torch.autograd.grad(actual, (x, y, u, v)):
            torch.testing.assert_close(g, torch.zeros_like(g))
        with pytest.raises((RuntimeError, ValueError, IndexError), match='min|max|eligible|dimension'):
            operator.min()
    assert tp.lazy(x.expand(0, *x.shape), y, metric='rbf').bilinear(u, v).shape == (0,)


def test_kernel_operator_exclusion_mutation_validation_amp(device):
    x = torch.rand(3, 2, device=device)
    y = torch.rand(4, 2, device=device)
    operator = tp.lazy(x, metric='linear', chunk_size=2)
    before = operator.sum()
    x.add_(1.)
    torch.testing.assert_close(operator.sum(), (x @ x.T).sum())
    assert operator.sum() != before
    gram = x @ x.T
    mask = ~torch.eye(3, device=device, dtype=torch.bool)
    torch.testing.assert_close(operator.mean(exclude_self=True), gram[mask].mean())
    torch.testing.assert_close(operator.sum(dim=-2, exclude_self=True), torch.where(mask, gram, 0.).sum(-2))
    with pytest.raises((RuntimeError, ValueError), match='square'):
        tp.lazy(x, y, metric='rbf').quadratic(torch.ones(3, device=device))
    with pytest.raises((RuntimeError, ValueError), match='sample count'):
        tp.lazy(x, y, metric='rbf').bilinear(torch.ones(2, device=device), torch.ones(4, device=device))
    torch.testing.assert_close(operator.sum(dim=0),gram.sum(0))
    with pytest.raises(ValueError, match='dim'):
        operator.sum(dim=2)
    with pytest.raises(TypeError, match='parameter'):
        tp.lazy(x, metric='linear', degree=3)
    with pytest.raises(RuntimeError, match='indices|int64'):
        operator.block(torch.tensor([1.], device=device), torch.tensor([1], device=device))
    torch.testing.assert_close(tp.lazy(x, metric='rbf').all(dim=None, predicate=lambda t: t > 0),
                               tp.rbf_kernel(x).gt(0).all())
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    a = torch.rand(3, 2, device=device, dtype=dtype, requires_grad=True)
    u = torch.rand(3, device=device, dtype=dtype, requires_grad=True)
    with torch.autocast(device, dtype=dtype):
        actual = tp.lazy(a, metric='rbf', gamma=0.4).quadratic(u)
    assert actual.dtype == torch.float32
    expected = tp.lazy(a.float(), metric='rbf', gamma=0.4).quadratic(u.float())
    torch.testing.assert_close(actual, expected)
    assert all(g.isfinite().all() for g in torch.autograd.grad(actual, (a, u)))


def test_kernel_form_retained_state_and_channel_tiles(device):
    torch.manual_seed(729)
    # Noncontiguous dense views, channels crossing the tile boundary, and input
    # batch broadcasting. Retained state must consist only of the four operands.
    x = torch.rand(3, 11, device=device, dtype=torch.float64).T.requires_grad_()
    y = torch.rand(3, 13, device=device, dtype=x.dtype).T.requires_grad_()
    left = torch.rand(6, 11, device=device, dtype=x.dtype).T.requires_grad_()
    right = torch.rand(5, 13, device=device, dtype=x.dtype).T.requires_grad_()
    saved = []
    class Packed:
        def __init__(self, tensor): self.tensor = tensor
    def pack(tensor):
        item = Packed(tensor)
        saved.append(weakref.ref(item))
        return item
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda item: item.tensor):
        actual = tp.lazy(x, y, gamma=0.3, chunk_size=4, metric='polynomial').bilinear(left, right)
    shapes = [tuple(item().tensor.shape) for item in saved if item() is not None]
    assert shapes and all(s in (x.shape, y.shape, left.shape, right.shape) for s in shapes)
    expected = left.T @ (.3 * (x @ y.T) + 1.).pow(3) @ right
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
        gradients = torch.autograd.grad(actual.sum(), (x, y, left, right))
    for a, e in zip(gradients, torch.autograd.grad(expected.sum(), (x, y, left, right))):
        torch.testing.assert_close(a, e)
    calls = [e for e in profile.events() if e.name == 'torchpairwise::_legacy_compare']
    assert calls and all(e.input_shapes[0][-2] <= 4 and e.input_shapes[1][-2] <= 4 for e in calls)
    value = tp.lazy(x, metric='rbf').bilinear(left, left)
    with pytest.raises(RuntimeError, match='Higher derivatives'):
        torch.autograd.grad(value.sum(), x, create_graph=True)


@pytest.mark.compiler
def test_kernel_operator_compile_export_contracts(device):
    torch.manual_seed(3201)
    x, y = [torch.rand(n, 2, device=device, requires_grad=True) for n in (3, 5)]
    left = torch.rand(3, 2, device=device, requires_grad=True)
    right = torch.rand(5, 2, device=device, requires_grad=True)
    checks = torch.library.opcheck(torch.ops.torchpairwise._kernel_form.default,
                                  (x, y, left, right, 'rbf', 2, [.4, None, None, None, None], None))
    assert all(v == 'SUCCESS' for v in checks.values())
    class Operations(torch.nn.Module):
        def forward(self, a, b, l, r):
            kernel = tp.lazy(a, b, metric='rbf', gamma=0.4, chunk_size=2)
            return kernel.matmul(r), kernel.bilinear(l, r), kernel.sum(), kernel.mean(dim=-2), kernel.diagonal()
    model = Operations()
    actual = torch.compile(model, fullgraph=True)(x, y, left, right)
    expected = model(x, y, left, right)
    for a, e in zip(actual, expected): torch.testing.assert_close(a, e)
    for a, e in zip(torch.autograd.grad(sum(t.sum() for t in actual), (x, y, left, right)),
                    torch.autograd.grad(sum(t.sum() for t in expected), (x, y, left, right))):
        torch.testing.assert_close(a, e)
    torch.compiler.reset()
    n, m = torch.export.Dim('n', min=2, max=8), torch.export.Dim('m', min=2, max=8)
    exported = torch.export.export(model, (x, y, left, right),
                                   dynamic_shapes=({0: n}, {0: m}, {0: n}, {0: m}))
    buffer = io.BytesIO()
    torch.export.save(exported, buffer)
    buffer.seek(0)
    loaded = torch.export.load(buffer).module()
    # Reverse N<M to N>M to exercise rectangular diagonal symbolic minimum.
    args = (torch.rand(6, 2, device=device), torch.rand(4, 2, device=device),
            torch.rand(6, 2, device=device), torch.rand(4, 2, device=device))
    for a, e in zip(loaded(*args), model(*args)): torch.testing.assert_close(a, e)
