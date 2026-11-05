"""Partition maps: projector identities, streamed scatters and deployment."""
import io

import pytest
import torch
import torchpairwise as tp
import torchpairwise as s


def dense_projector(labels, dtype):
    same = labels.unsqueeze(-1) == labels.unsqueeze(-2)
    return same.to(dtype) / same.sum(-1, keepdim=True)


def test_group_projection_actions_gradients_and_boundaries(device):
    # Different partitions per batch, arbitrary signed/gapped int64 IDs, strided
    # labels and values. The dense oracle is only for these small test matrices.
    labels = torch.tensor([[2**62, -7, 2**62, 99, -7],
                           [-2**63, -2**63, 4, 4, 4]], device=device).T.contiguous().T
    values = torch.randn(1, 3, 5, dtype=torch.float64, device=device).transpose(-1, -2).requires_grad_()
    projector = s.group_projection(labels, like=values)
    assert projector.shape == (2, 5, 5)
    assert projector.dtype == values.dtype and projector.device == values.device
    dense = dense_projector(labels, values.dtype)
    result = projector.matmul(values)
    expected = dense @ values
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(projector.T.matmul(values), result)
    torch.testing.assert_close(projector.matmul(result), result)
    torch.testing.assert_close(torch.autograd.grad(result.square().sum(), values),
                               torch.autograd.grad(expected.square().sum(), values))
    # Construction snapshots labels rather than storing mutable external IDs.
    labels.fill_(0)
    torch.testing.assert_close(projector.matmul(values), expected)
    for dtype in (torch.float32, torch.float64):
        v = torch.randn(5, device=device, dtype=dtype)
        for ids, want in ((torch.zeros(5, device=device, dtype=torch.int64), v.mean().expand_as(v)),
                          (torch.arange(5, device=device), v)):
            p = s.group_projection(ids, like=v)
            torch.testing.assert_close(p.matmul(v), want)
    for shape in ((0,), (0, 5)):
        ids = torch.empty(shape, device=device, dtype=torch.int64)
        v = torch.empty((*shape, 2), device=device, dtype=torch.float64, requires_grad=True)
        p = s.group_projection(ids, like=v)
        r = p.matmul(v)
        assert r.shape == v.shape and r.numel() == 0
        assert torch.autograd.grad(r.sum(), v)[0].shape == v.shape
    with pytest.raises(ValueError, match='int64'):
        s.group_projection(torch.ones(5, device=device), like=values)
    with pytest.raises(ValueError, match='float32'):
        s.group_projection(torch.ones(5, dtype=torch.int64, device=device), like=labels)


def test_group_projection_cg_and_scatter_maps(device):
    torch.manual_seed(89)
    x = torch.randn(2, 7, 3, device=device, dtype=torch.float64)
    labels = torch.tensor([10, 10, -4, -4, -4, 99, 99], device=device)
    p = s.group_projection(labels, like=x)
    k = s.lazy(x, gamma=0.25, chunk_size=3, metric='rbf')
    residual = k - p @ k          # (I-P)K
    within = residual.T @ residual + 0.5 * tp.eye_like(residual.T @ residual)
    b = torch.randn(7, 2, device=device, dtype=x.dtype)
    fit = s.linalg.cg(within, b, rtol=1e-9, max_iter=60)
    dense_k = torch.exp(-.25 * (x.unsqueeze(-2) - x.unsqueeze(-3)).square().sum(-1))
    dense_p = dense_projector(labels, x.dtype)
    w = dense_k @ (torch.eye(7, device=device, dtype=x.dtype) - dense_p) @ dense_k + .5 * torch.eye(7, device=device, dtype=x.dtype)
    torch.testing.assert_close(fit.solution, torch.linalg.solve(w, b), rtol=1e-7, atol=1e-8)
    assert fit.converged.all()
    # A partition-only graph has no scalar-source packet. Its nullspace becomes
    # invertible with the explicit diagonal shift.
    fit = s.linalg.cg(p + 0.25 * tp.eye_like(p), b, rtol=1e-10)
    torch.testing.assert_close(fit.solution, torch.linalg.solve(dense_p + .25 * torch.eye(7, device=device, dtype=x.dtype), b))
    assert fit.converged.all()
    with pytest.raises(ValueError, match='dtype'):
        s.linalg.cg(p + 0.25 * tp.eye_like(p), b.float())


@pytest.mark.compiler
def test_group_projection_compile_export_and_gradients(device):
    class Project(torch.nn.Module):
        def forward(self, labels, v):
            p = s.group_projection(labels, like=v)
            action = p.matmul(v)
            fit = s.linalg.cg(p + 0.5 * tp.eye_like(p), v.detach(), rtol=1e-09, max_iter=30)
            return action, fit
    model = Project()
    compiled = torch.compile(model, backend='inductor', fullgraph=True, dynamic=True)
    def inputs(n):
        return (torch.arange(n, device=device) % 3,
                torch.randn(n, 2, device=device, dtype=torch.float64, requires_grad=True))
    for n in (7, 10):
        labels, v = inputs(n)
        actual, expected = compiled(labels, v), model(labels, v)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual[0].square().sum(), v),
                                   torch.autograd.grad(expected[0].square().sum(), v))
    labels, v = inputs(7)
    n = torch.export.Dim('samples', min=2, max=20)
    exported = torch.export.export(model, (labels, v), dynamic_shapes=({0:n}, {0:n}))
    buffer = io.BytesIO()
    torch.export.save(exported, buffer)
    buffer.seek(0)
    restored = torch.export.load(buffer).module()
    labels, v = inputs(11)
    torch.testing.assert_close(restored(labels, v), model(labels, v))
