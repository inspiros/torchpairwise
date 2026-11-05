"""Native schema visibility, Python scripting and representative graph export."""
import pytest
import inspect
import weakref
import torch
import torchpairwise as tp


@pytest.mark.parametrize('name', ('angular_distances', 'kl_divergences', 'matern_kernel'))
def test_native_named_script_serialization(name, device, tmp_path):
    x, y = [torch.rand(n, 4, device=device, dtype=torch.float64) + .3 for n in (3, 5)]
    operation = getattr(tp, name)
    scripted = torch.jit.script(operation)
    torch.testing.assert_close(scripted(x, y), operation(x, y))
    assert 'prim::PythonOp' not in str(scripted.graph)
    assert 'torchpairwise::' + name in str(scripted.graph)
    path = tmp_path / 'comparison.pt'
    scripted.save(str(path))
    restored = torch.jit.load(str(path), map_location=device)
    # Saving a bare ScriptFunction as a module does not preserve its Python
    # default-argument metadata. A deployed enclosing module fixes parameters.
    defaults = tuple(parameter.default for parameter in
                     list(inspect.signature(operation).parameters.values())[2:])
    torch.testing.assert_close(restored(x, y, *defaults), operation(x, y))
    # Scripted calls retain native first derivatives.
    x, y = x.requires_grad_(), y.requires_grad_()
    for actual, expected in zip(torch.autograd.grad(restored(x, y, *defaults).sum(), (x, y)),
                                torch.autograd.grad(operation(x, y).sum(), (x, y))):
        torch.testing.assert_close(actual, expected)


def test_native_generic_modes_and_rejected_options(device):
    x, y = [torch.rand(3, 4, device=device, requires_grad=True) + .3 for _ in range(2)]
    native = torch.ops.torchpairwise.vector_comparisons
    for name, parameter in [('angular_distances', {}), ('symmetric_kl_divergences', {'base': 2.}),
                            ('matern_kernel', {'length_scale': .7, 'nu': 2.5})]:
        matrix = getattr(tp, name)(x, y, **parameter)
        paired = native(x, y, name, 'paired', **parameter)
        torch.testing.assert_close(paired, matrix.diagonal())
        condensed = native(x, None, name, 'condensed', **parameter)
        i, j = torch.triu_indices(3, 3, 1, device=device)
        torch.testing.assert_close(condensed, getattr(tp, name)(x, **parameter)[i, j])
    with pytest.raises(RuntimeError, match='Unused parameter: alpha'):
        native(x, y, 'angular', alpha=2.)
    with pytest.raises(RuntimeError, match='symmetric'):
        native(x, None, 'kl', 'condensed')
    with pytest.raises(RuntimeError, match='two inputs'):
        native(x, None, 'kl', 'paired')


def test_native_forward_retains_operands_without_pair_feature_state(device):
    x = (torch.rand(17, 65, device=device) + .3).requires_grad_()
    y = (torch.rand(23, 65, device=device) + .3).requires_grad_()
    for operation in (tp.angular_distances, tp.renyi_divergences):
        packed = []
        def pack(tensor):
            # The tuple is weakly tracked: transient validation statistics
            # must not be mistaken for state retained until backward.
            class Saved:
                pass
            saved = Saved()
            saved.tensor = tensor.detach()
            packed.append(weakref.ref(saved))
            return saved
        with torch.autograd.graph.saved_tensors_hooks(
                pack, lambda saved: saved.tensor):
            result = operation(x, y)
        shapes = [tuple(saved().tensor.shape) for saved in packed if saved() is not None]
        assert shapes and all(shape in ((17, 65), (23, 65), (0,)) for shape in shapes)
        assert all(torch.isfinite(gradient).all()
                   for gradient in torch.autograd.grad(result.sum(), (x, y)))


@pytest.mark.compiler
def test_native_vector_export_dynamic_roundtrip(device, tmp_path):
    class Comparisons(torch.nn.Module):
        def forward(self, x, y):
            return tp.kl_divergences(x, y, base=2.) + tp.rational_quadratic_kernel(x, y, alpha=.7)
    x, y = [torch.rand(n, 4, device=device) + .3 for n in (3, 5)]
    n, m = torch.export.Dim('n', min=2), torch.export.Dim('m', min=2)
    exported = torch.export.export(Comparisons(), (x, y), dynamic_shapes=({0: n}, {0: m}))
    path = tmp_path / 'comparisons.pt2'
    torch.export.save(exported, path)
    loaded = torch.export.load(path).module()
    x, y = [torch.rand(count, 4, device=device) + .3 for count in (4, 6)]
    torch.testing.assert_close(loaded(x, y), Comparisons()(x, y))
