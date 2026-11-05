"""Lazy namespace integration and one shared compile/export integration path."""
import io

import pytest
import torch
import torchpairwise as tp
import torchpairwise as s


@pytest.mark.compiler
def test_lazy_namespace_compile_export_and_gradients(device):
    class Model(torch.nn.Module):
        def forward(self, x, y, v):
            distances = s.lazy(x, y, metric='sqeuclidean', chunk_size=3)
            kernel = s.lazy(x, y, metric='rbf', gamma=0.2, chunk_size=3)
            return ((-.2 * distances).exp() + kernel).matmul(v)

    torch.manual_seed(42)
    model = Model()
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    for n, m in ((5, 7), (8, 4)):
        x = torch.randn(2, n, 3, device=device, requires_grad=True)
        y = torch.randn(1, m, 3, device=device, requires_grad=True)
        v = torch.randn(m, 2, device=device, requires_grad=True)
        expected = 2 * tp.rbf_kernel(x, y, gamma=.2).matmul(v)
        for actual in (model(x, y, v), compiled(x, y, v)):
            torch.testing.assert_close(actual, expected)
            actual_grad = torch.autograd.grad(actual.sum(), (x, y, v))
            expected_grad = torch.autograd.grad(expected.sum(), (x, y, v), retain_graph=True)
            torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-4, atol=2e-5)
    n_dim, m_dim = torch.export.Dim('n', min=2), torch.export.Dim('m', min=2)
    exported = torch.export.export(model, (x, y, v), dynamic_shapes=(
        {1: n_dim}, {1: m_dim}, {0: m_dim}))
    buffer = io.BytesIO()
    torch.export.save(exported, buffer)
    buffer.seek(0)
    restored = torch.export.load(buffer).module()
    x = torch.randn(2, 6, 3, device=device)
    y = torch.randn(1, 9, 3, device=device)
    v = torch.randn(9, 2, device=device)
    torch.testing.assert_close(restored(x, y, v), model(x, y, v))
