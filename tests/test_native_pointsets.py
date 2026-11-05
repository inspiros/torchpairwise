"""Native point-set deployment; independent formulas live in test_pointsets."""
import weakref
import pytest
import torch
import torchpairwise as tp


def test_native_pointset_layout_options_and_witnesses(device):
    x = torch.rand(2, 3, 2, 2, 2, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(1, 4, 2, 1, 3, device=device, dtype=x.dtype, requires_grad=True)
    xl = torch.tensor([[4, 2, 3], [1, 4, 2]], device=device)
    yl = torch.tensor([[3, 1, 2, 3]], device=device)
    native = torch.ops.torchpairwise.pointset_comparisons
    options = dict(batched=True, feature_dim=2, x1_lengths=xl, x2_lengths=yl, chunk_size=2)
    actual = native(x, y, 'chamfer', squared=True, **options)
    expected = tp.chamfer_distances(x, y, squared=True, **options)
    torch.testing.assert_close(actual, expected)
    for gradient, reference in zip(torch.autograd.grad(actual.sum(), (x, y)),
                                   torch.autograd.grad(expected.sum(), (x, y))):
        torch.testing.assert_close(gradient, reference)
    result = torch.ops.torchpairwise.directed_pointset_comparisons(x, y, **options)
    for a, b in zip(result, tp.directed_hausdorff_distances(x, y, **options)):
        torch.testing.assert_close(a, b)
    with pytest.raises(RuntimeError, match='Unused parameter: squared'):
        native(x, y, 'hausdorff', squared=False, batched=True, feature_dim=2)
    with pytest.raises(RuntimeError, match='Unused parameter: shuffle'):
        native(x, y, 'chamfer', shuffle=False, batched=True, feature_dim=2)
    with pytest.raises(RuntimeError, match='symmetric'):
        native(x, None, 'directed_hausdorff', 'condensed', batched=True, feature_dim=2)
    with pytest.raises(RuntimeError, match='batch axes'):
        native(x, y, 'chamfer', x1_lengths=xl.expand(2, 2, 3), batched=True, feature_dim=2)


def test_native_pointset_replay_retains_operands(device):
    x = torch.rand(17, 5, 3, device=device, requires_grad=True)
    y = torch.rand(19, 7, 3, device=device, requires_grad=True)
    packed = []
    class Saved:
        pass
    def pack(tensor):
        saved = Saved()
        saved.tensor = tensor.detach()
        packed.append(weakref.ref(saved))
        return saved
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda saved: saved.tensor):
        output = tp.chamfer_distances(x, y)
    shapes = [tuple(saved().tensor.shape) for saved in packed if saved() is not None]
    assert shapes and all(shape in ((1, 17, 5, 3), (1, 19, 7, 3), (0,)) for shape in shapes)
    assert all(gradient.isfinite().all() for gradient in torch.autograd.grad(output.sum(), (x, y)))


@pytest.mark.compiler
@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed'])
def test_pointset_export_dynamic_roundtrip(mode, device, tmp_path):
    class Pointsets(torch.nn.Module):
        def forward(self, x, y, xl, yl):
            if mode == 'all_pairs':
                d, i, j = tp.directed_hausdorff_distances(x, y, x1_lengths=xl, x2_lengths=yl, chunk_size=2)
                c = tp.chamfer_distances(x, y, x1_lengths=xl, x2_lengths=yl)
                full = tp.hausdorff_distances(torch.nan_to_num(x), torch.nan_to_num(y))
                return c + d + full, i, j
            if mode == 'paired':
                return tp.paired_distances(x, y, metric='hausdorff', x1_lengths=xl, x2_lengths=yl, chunk_size=2)
            return tp.pdist(x, metric='chamfer', squared=True, x1_lengths=xl, chunk_size=2)
    n = torch.export.Dim('n', min=3)
    m = n if mode == 'paired' else torch.export.Dim('m', min=3)
    p, q = torch.export.Dim('p', min=2), torch.export.Dim('q', min=2)
    def inputs(n, m, p, q):
        x = torch.rand(n, p, 2, device=device, requires_grad=True)
        y = torch.rand(m, q, 2, device=device, requires_grad=True)
        xl = torch.full((n,), p - 1, device=device, dtype=torch.int64)
        yl = torch.full((m,), q - 1, device=device, dtype=torch.int64)
        with torch.no_grad():
            x[:, -1] = float('nan')
            y[:, -1] = float('nan')
        return x, y, xl, yl
    args = inputs(3, 3 if mode == 'paired' else 4, 4, 5)
    module = Pointsets()
    exported = torch.export.export(module, args, dynamic_shapes=({0: n, 1: p}, {0: m, 1: q}, {0: n}, {0: m}))
    path = tmp_path / 'pointsets.pt2'
    torch.export.save(exported, path)
    restored = torch.export.load(path).module()
    args = inputs(5, 5 if mode == 'paired' else 6, 6, 7)
    actual, expected = restored(*args), module(*args)
    if mode == 'all_pairs':
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        actual, expected = actual[0], expected[0]
    else:
        torch.testing.assert_close(actual, expected)
    operands = (args[0],) if mode == 'condensed' else args[:2]
    for gradient, reference in zip(torch.autograd.grad(actual.sum(), operands),
                                   torch.autograd.grad(expected.sum(), operands)):
        torch.testing.assert_close(gradient, reference)
    if mode == 'all_pairs':
        torch.compiler.reset()
        compiled = torch.compile(module, fullgraph=True, dynamic=True)
        compiled(*inputs(3, 4, 4, 5))
        output, reference = compiled(*args), module(*args)
        for a, b in zip(output, reference):
            torch.testing.assert_close(a, b)
        for a, b in zip(torch.autograd.grad(output[0].sum(), operands),
                        torch.autograd.grad(reference[0].sum(), operands)):
            torch.testing.assert_close(a, b)
        torch.compiler.reset()


def test_native_nearest_backward_point_tiles_and_noncontiguous_inputs(device):
    x = torch.rand(1, 2, 3, 131, device=device, dtype=torch.float64).transpose(-1, -2)
    y = torch.rand(1, 2, 3, 5, device=device, dtype=x.dtype).transpose(-1, -2)
    xl = torch.tensor([[129, 1]], device=device)
    yl = torch.tensor([[5, 2]], device=device)
    xv = torch.arange(131, device=device) < xl[..., None]
    yv = torch.arange(5, device=device) < yl[..., None]
    x[~xv], y[~yv] = float('nan'), float('nan')
    x.requires_grad_()
    y.requires_grad_()
    actual, _ = torch.ops.torchpairwise._pointset_nn(x, y, xl, yl)
    a, b = torch.where(xv[..., None], x, 0.), torch.where(yv[..., None], y, 0.)
    ground = (a[:, :, None, :, None] - b[:, None, :, None]).square().sum(-1)
    expected = ground.masked_fill(~yv[:, None, :, None], float('inf')).min(-1).values
    expected = torch.where(xv[:, :, None], expected, 0.)
    torch.testing.assert_close(actual, expected)
    for gradient, reference in zip(torch.autograd.grad(actual.sum(), (x, y)),
                                   torch.autograd.grad(expected.sum(), (x, y))):
        torch.testing.assert_close(gradient, reference)
        assert gradient.isfinite().all()


def test_shuffled_native_pointset_modes_do_not_replay_rng(device):
    x = torch.rand(3, 4, 2, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(3, 5, 2, device=device, dtype=x.dtype, requires_grad=True)
    for paired in (True, False):
        generator = torch.Generator(device=device).manual_seed(81)
        if paired:
            value = tp.paired_distances(x, y, metric='directed_hausdorff', shuffle=True, generator=generator)
            reference = tp.paired_distances(x, y, metric='directed_hausdorff')
            operands = (x, y)
        else:
            value = tp.pdist(x, metric='hausdorff', shuffle=True, generator=generator)
            reference = tp.pdist(x, metric='hausdorff')
            operands = (x,)
        torch.testing.assert_close(value, reference)
        after_forward = generator.get_state().clone()
        assert all(g.isfinite().all() for g in torch.autograd.grad(value.sum(), operands))
        assert torch.equal(generator.get_state(), after_forward)
