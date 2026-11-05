"""Deployment boundaries for shared weighted C++ execution.

The weighted catalog tests provide independent numerical references. These tests
exercise native-only calls, dynamic graphs, and retained autograd state.
"""
import weakref
import pytest
import torch
import torchpairwise as tp


def test_native_weighted_batch_parameters_and_options(device):
    x = torch.rand(2, 1, 3, 5, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(1, 2, 4, 5, device=device, dtype=x.dtype, requires_grad=True)
    w = torch.rand(2, 2, 5, device=device, dtype=x.dtype, requires_grad=True)
    native = torch.ops.torchpairwise.weighted_comparisons
    for metric in ('sqeuclidean', 'cosine', 'hamming'):
        result = native(x, y, w, metric)
        reference = tp.cdist(x, y, metric=metric, w=w)
        torch.testing.assert_close(result, reference)
        operands = (w,) if metric == 'hamming' else (x, y, w)
        for actual, expected in zip(torch.autograd.grad(result.sum(), operands),
                                    torch.autograd.grad(reference.sum(), operands)):
            torch.testing.assert_close(actual, expected)
    with pytest.raises(RuntimeError, match='Unused parameter: p'):
        native(x, y, w, 'cosine', p=2.)
    with pytest.raises(RuntimeError, match='positive'):
        native(x, y, w, 'minkowski', p=0.)
    with pytest.raises(RuntimeError, match='two inputs'):
        native(x, None, w, 'cosine', 'paired')
    with pytest.raises(RuntimeError, match='one input'):
        native(x, y, w, 'cosine', 'condensed')
    with pytest.raises(RuntimeError, match='bool inputs'):
        native(x, y, w, 'jaccard')
    # Hamming must preserve raw integer equality, including adjacent int64s
    # that become equal if a floating scalar accidentally promotes operands.
    a = torch.tensor([[2**60, 2**60 + 1]], device=device)
    b = a.flip(-1)
    torch.testing.assert_close(native(a, b, w.new_ones(2), 'hamming'), w.new_ones(1, 1))
    assert not native(x, y, w.detach(), 'hamming').requires_grad


def test_native_weighted_retained_state(device):
    x = torch.rand(17, 65, device=device, requires_grad=True)
    y = torch.rand(23, 65, device=device, requires_grad=True)
    w = torch.rand(65, device=device, requires_grad=True)
    for metric in ('sqeuclidean', 'braycurtis', 'cosine'):
        packed = []
        class Saved:
            pass
        def pack(tensor):
            saved = Saved()
            saved.tensor = tensor.detach()
            packed.append(weakref.ref(saved))
            return saved
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda saved: saved.tensor):
            output = torch.ops.torchpairwise.weighted_comparisons(x, y, w, metric)
        shapes = [tuple(saved().tensor.shape) for saved in packed if saved() is not None]
        assert shapes and all(shape in ((17, 65), (23, 65), (65,)) for shape in shapes)
        assert all(gradient.isfinite().all() for gradient in torch.autograd.grad(output.sum(), (x, y, w)))


@pytest.mark.compiler
@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed'])
def test_weighted_export_dynamic_roundtrip(mode, device, tmp_path):
    class Weighted(torch.nn.Module):
        def forward(self, x, y, w):
            if mode == 'all_pairs':
                return tp.cosine_distances(x, y, w=w) + tp.cdist(x, y, metric='minkowski', p=1.7, w=w)
            if mode == 'paired':
                return tp.paired_distances(x, y, metric='correlation', w=w)
            return tp.condensed_distances(x, metric='sqeuclidean', w=w)
    # PyTorch specializes singleton dimensions. N=2 gives a singleton condensed
    # output, requiring a separate export from the general N>=3 graph.
    n = torch.export.Dim('n', min=3 if mode == 'condensed' else 2)
    m = n if mode == 'paired' else torch.export.Dim('m', min=2)
    # Weights introduce a batch axis absent from x/y, and remain graph inputs.
    x = torch.rand(3, 5, device=device)
    y = torch.rand(3 if mode == 'paired' else 4, 5, device=device)
    w = torch.rand(2, 5, device=device) + .2
    module = Weighted()
    exported = torch.export.export(module, (x, y, w), dynamic_shapes=(
        {0: n}, {0: m}, {0: torch.export.Dim('batch', min=1)}))
    path = tmp_path / 'weighted.pt2'
    torch.export.save(exported, path)
    restored = torch.export.load(path).module()
    targets = {node.target for node in exported.graph.nodes if node.op == 'call_function'}
    # The complete selector may remain opaque instead of tracing into the
    # weighted family/core; all of these are native deployment boundaries.
    assert targets.intersection({torch.ops.torchpairwise.weighted_comparisons.default,
                                 torch.ops.torchpairwise._weighted_compare.default,
                                 torch.ops.torchpairwise.builtin_comparisons.default,
                                 torch.ops.torchpairwise.compare.default})
    x = torch.rand(6, 5, device=device, requires_grad=True)
    y = torch.rand(6 if mode == 'paired' else 7, 5, device=device, requires_grad=True)
    w = (torch.rand(3, 5, device=device) + .2).requires_grad_()
    actual, expected = restored(x, y, w), module(x, y, w)
    torch.testing.assert_close(actual, expected)
    operands = (x, w) if mode == 'condensed' else (x, y, w)
    for gradient, reference in zip(torch.autograd.grad(actual.sum(), operands),
                                   torch.autograd.grad(expected.sum(), operands)):
        torch.testing.assert_close(gradient, reference)
    if mode == 'all_pairs':
        torch.compiler.reset()
        compiled = torch.compile(module, fullgraph=True, dynamic=True)
        compiled(x[:3], y[:4], w[:2])
        output = compiled(x, y, w)
        reference = module(x, y, w)
        torch.testing.assert_close(output, reference)
        for gradient, expected in zip(torch.autograd.grad(output.sum(), operands),
                                       torch.autograd.grad(reference.sum(), operands)):
            torch.testing.assert_close(gradient, expected)
        torch.compiler.reset()


@pytest.mark.compiler
def test_weighted_boolean_export_and_compiled_gradient(device, tmp_path):
    class WeightedBoolean(torch.nn.Module):
        def forward(self, x, y, w):
            return tp.cdist(x, y, metric='jaccard', w=w) + tp.cdist(x, y, metric='hamming', w=w)
    x = torch.rand(3, 5, device=device) > .5
    y = torch.rand(4, 5, device=device) > .5
    w = (torch.rand(5, device=device) + .2).requires_grad_()
    module = WeightedBoolean()
    torch.compiler.reset()
    actual = torch.compile(module, fullgraph=True)(x, y, w)
    expected = module(x, y, w)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), w)[0], torch.autograd.grad(expected.sum(), w)[0])
    exported = torch.export.export(module, (x, y, w), dynamic_shapes=(
        {0: torch.export.Dim('n', min=2)}, {0: torch.export.Dim('m', min=2)}, None))
    path = tmp_path / 'weighted_bool.pt2'
    torch.export.save(exported, path)
    restored = torch.export.load(path).module()
    x, y = x.repeat(2, 1), y.repeat(2, 1)
    torch.testing.assert_close(restored(x, y, w), module(x, y, w))
    torch.compiler.reset()
