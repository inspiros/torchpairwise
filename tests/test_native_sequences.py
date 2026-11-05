"""Native sequence deployment and local recurrence derivatives.

Independent path/edit oracles and exhaustive ragged cases also live in
``test_sequence.py`` and ``test_tokens.py``.
"""
import weakref

import pytest
import torch
import torchpairwise as tp


def _hard_recurrence(a, b):
    # Unlike enumerating equally weighted complete paths, amin at each cell
    # checks the documented local tie rule used by the explicit adjoint.
    infinity = a.new_tensor(float('inf'))
    rows = [[a.new_tensor(0.)] + [infinity] * len(b)]
    for i in range(len(a)):
        row = [infinity]
        for j in range(len(b)):
            predecessor = torch.stack((rows[-1][j], rows[-1][j + 1], row[-1])).amin(0)
            row.append((a[i] - b[j]).square().sum() + predecessor)
        rows.append(row)
    return rows[-1][-1]


def test_native_sequence_ties_and_window_zero_gradients(device):
    x = torch.tensor([[[0.], [1.]]], dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tensor([[[0.], [0.], [2.]]], dtype=x.dtype, device=device, requires_grad=True)
    native = torch.ops.torchpairwise.sequence_comparisons
    actual = native(x, y, 'paired', True)
    reference = _hard_recurrence(x[0], y[0])
    torch.testing.assert_close(actual[0], reference)
    for a, b in zip(torch.autograd.grad(actual.sum(), (x, y)),
                    torch.autograd.grad(reference, (x, y))):
        torch.testing.assert_close(a, b)
    y = (x.detach() + .25).requires_grad_()
    for gamma in (None, .7):
        actual = native(x, y, 'paired', True if gamma is None else False,
                        gamma, window=0, chunk_size=1)
        reference = (x - y).square().sum((-1, -2))
        torch.testing.assert_close(actual, reference)
        for a, b in zip(torch.autograd.grad(actual.sum(), (x, y)),
                        torch.autograd.grad(reference.sum(), (x, y))):
            torch.testing.assert_close(a, b)


def test_native_sequence_retains_only_inputs_and_masks(device):
    x = torch.randn(17, 5, 2, device=device, requires_grad=True)
    y = torch.randn(19, 7, 2, device=device, requires_grad=True)
    packed = []
    class Saved:
        pass
    def pack(tensor):
        saved = Saved()
        saved.tensor = tensor.detach()
        packed.append(weakref.ref(saved))
        return saved
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda saved: saved.tensor):
        output = tp.soft_dtw_divergences(x, y, gamma=.7, chunk_size=2)
    saved = [entry().tensor for entry in packed if entry() is not None]
    assert saved
    assert all(value.numel() <= max(x.numel(), y.numel()) for value in saved)
    assert all(value.isfinite().all() for value in torch.autograd.grad(output.sum(), (x, y)))


def test_native_sequence_and_token_contracts(device):
    x = torch.ones(2, 3, 1, device=device)
    native = torch.ops.torchpairwise.sequence_comparisons
    with pytest.raises(RuntimeError, match='gamma'):
        native(x, gamma=0.)
    with pytest.raises(RuntimeError, match='chunk_size'):
        native(x, chunk_size=0)
    with pytest.raises(RuntimeError, match='one input'):
        native(x, x, 'condensed')
    with pytest.raises(RuntimeError, match='squared'):
        native(x, gamma=.7, squared=True)
    token = torch.ops.torchpairwise.token_comparisons
    with pytest.raises(RuntimeError, match='integer'):
        token(x.squeeze(-1))
    with pytest.raises(RuntimeError, match='substitution_cost'):
        token(x.squeeze(-1).long(), substitution_cost=3)
    large = torch.tensor([[2 ** 60, 2 ** 60 + 1], [2 ** 60, 2 ** 60 + 2]], device=device)
    expected = torch.tensor([[0, 1], [1, 0]], device=device)
    torch.testing.assert_close(token(large), expected)


@pytest.mark.compiler
@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed'])
@pytest.mark.parametrize('soft', [False, True])
def test_sequence_export_dynamic_roundtrip(mode, soft, device, tmp_path):
    class Sequences(torch.nn.Module):
        def forward(self, x, y, xl, yl):
            kwargs = {'gamma': .7} if soft else {'squared': True}
            if mode == 'all_pairs':
                fn = tp.soft_dtw_divergences if soft else tp.dtw_distances
                return fn(x, y, x1_lengths=xl, x2_lengths=yl, chunk_size=2, **kwargs)
            if mode == 'paired':
                fn = tp.paired_soft_dtw_divergences if soft else tp.paired_dtw_distances
                return fn(x, y, x1_lengths=xl, x2_lengths=yl, chunk_size=2, **kwargs)
            fn = tp.condensed_soft_dtw_divergences if soft else tp.condensed_dtw_distances
            return fn(x, x_lengths=xl, chunk_size=2, **kwargs)
    n = torch.export.Dim('n', min=3)
    m = n if mode == 'paired' else torch.export.Dim('m', min=3)
    t, u = torch.export.Dim('t', min=3), torch.export.Dim('u', min=3)
    def inputs(n, m, t, u):
        x = torch.randn(n, t, 2, device=device, requires_grad=True)
        y = torch.randn(m, u, 2, device=device, requires_grad=True)
        xl = torch.full((n,), t - 1, device=device, dtype=torch.int64)
        yl = torch.full((m,), u - 1, device=device, dtype=torch.int64)
        with torch.no_grad():
            x[:, -1] = float('nan')
            y[:, -1] = float('nan')
        return x, y, xl, yl
    module = Sequences()
    args = inputs(3, 3 if mode == 'paired' else 4, 4, 5)
    exported = torch.export.export(module, args,
                                   dynamic_shapes=({0: n, 1: t}, {0: m, 1: u}, {0: n}, {0: m}))
    path = tmp_path / 'sequences.pt2'
    torch.export.save(exported, path)
    restored = torch.export.load(path).module()
    args = inputs(5, 5 if mode == 'paired' else 6, 6, 7)
    actual, expected = restored(*args), module(*args)
    torch.testing.assert_close(actual, expected)
    operands = args[:1] if mode == 'condensed' else args[:2]
    for a, b in zip(torch.autograd.grad(actual.sum(), operands),
                    torch.autograd.grad(expected.sum(), operands)):
        torch.testing.assert_close(a, b)
    if mode == 'all_pairs':
        torch.compiler.reset()
        compiled = torch.compile(module, fullgraph=True, dynamic=True)
        compiled(*inputs(3, 4, 4, 5))
        actual, expected = compiled(*args), module(*args)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(torch.autograd.grad(actual.sum(), operands),
                        torch.autograd.grad(expected.sum(), operands)):
            torch.testing.assert_close(a, b)
        torch.compiler.reset()


@pytest.mark.compiler
@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed'])
@pytest.mark.parametrize('substitution', [1, 2])
def test_token_export_dynamic_roundtrip(mode, substitution, device, tmp_path):
    class Tokens(torch.nn.Module):
        def forward(self, x, y, xl, yl):
            suffix = 'levenshtein_distances' if substitution == 1 else 'indel_distances'
            if mode == 'all_pairs':
                return getattr(tp, suffix)(x, y, x1_lengths=xl, x2_lengths=yl, chunk_size=3)
            if mode == 'paired':
                return getattr(tp, 'paired_' + suffix)(x, y, x1_lengths=xl, x2_lengths=yl, chunk_size=3)
            return getattr(tp, 'condensed_' + suffix)(x, x_lengths=xl, chunk_size=3)
    n = torch.export.Dim('n', min=3)
    m = n if mode == 'paired' else torch.export.Dim('m', min=3)
    t, u = torch.export.Dim('t', min=2), torch.export.Dim('u', min=2)
    def inputs(n, m, t, u):
        return (torch.randint(0, 4, (n, t), device=device),
                torch.randint(0, 4, (m, u), device=device),
                torch.arange(n, device=device).remainder(t + 1),
                torch.arange(m, device=device).remainder(u + 1))
    module = Tokens()
    args = inputs(3, 3 if mode == 'paired' else 4, 4, 5)
    exported = torch.export.export(module, args,
                                   dynamic_shapes=({0: n, 1: t}, {0: m, 1: u}, {0: n}, {0: m}))
    path = tmp_path / 'tokens.pt2'
    torch.export.save(exported, path)
    restored = torch.export.load(path).module()
    args = inputs(5, 5 if mode == 'paired' else 6, 6, 7)
    torch.testing.assert_close(restored(*args), module(*args))
    if mode == 'all_pairs':
        torch.compiler.reset()
        compiled = torch.compile(module, fullgraph=True, dynamic=True)
        compiled(*inputs(3, 4, 4, 5))
        torch.testing.assert_close(compiled(*args), module(*args))
        torch.compiler.reset()
