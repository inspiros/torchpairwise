"""Native transport deployment, modes and first backward without Python solvers."""
import io
import pytest
import torch
import torchpairwise as tp


@pytest.mark.compiler
@pytest.mark.parametrize('metric', ['energy', 'mmd', 'mmd_polynomial', 'sliced_wasserstein', 'sinkhorn', 'wasserstein'])
def test_native_transport_export_compile_modes(metric, device):
    polynomial = metric == 'mmd_polynomial'
    metric = 'mmd' if polynomial else metric
    class Comparison(torch.nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.mode = mode
            self.register_buffer('directions', torch.eye(2, device=device))

        def forward(self, x, y):
            options = {'max_iter': 5} if metric == 'sinkhorn' else {}
            if polynomial:
                options.update(kernel='polynomial', gamma=.4, degree=3, coef0=1., squared=True)
            if metric == 'sliced_wasserstein':
                options['projections'] = self.directions
            return tp.compare(x, None if self.mode == 'condensed' else y,
                              metric, mode=self.mode, **options)

    for mode in ('all_pairs', 'paired', 'condensed'):
        x = (torch.rand(3, 4, 2, device=device) + .1).requires_grad_()
        y = (torch.rand(3, 5, 2, device=device) + .3).requires_grad_()
        model = Comparison(mode)
        reference = model(x, y)
        compiled = torch.compile(model, backend='aot_eager', fullgraph=True)
        actual = compiled(x, y)
        torch.testing.assert_close(actual, reference)
        inputs = (x,) if mode == 'condensed' else (x, y)
        for result, expected in zip(torch.autograd.grad(actual.sum(), inputs),
                                    torch.autograd.grad(reference.sum(), inputs)):
            torch.testing.assert_close(result, expected)
        torch.compiler.reset()
        # PyTorch specializes output dimensions 0/1; condensed N=2 has one pair.
        count = torch.export.Dim('samples', min=3, max=6)
        exported = torch.export.export(model, (x, y), dynamic_shapes=({0: count}, {0: count}))
        buffer = io.BytesIO()
        torch.export.save(exported, buffer)
        buffer.seek(0)
        loaded = torch.export.load(buffer).module()
        bigger_x = torch.rand(4, 4, 2, device=device) + .1
        bigger_y = torch.rand(4, 5, 2, device=device) + .3
        torch.testing.assert_close(loaded(bigger_x, bigger_y), model(bigger_x, bigger_y))
