"""Rectangular singular triplets: dense spectral oracles and native deployment."""
import io

import pytest
import torch
import torchpairwise as tp
import torchpairwise as s


def test_svd_rectangular_spectrum_and_boundaries(device):
    torch.manual_seed(351)
    for dtype in (torch.float32, torch.float64):
        for n, m in ((11, 7), (7, 11)):
            u = torch.linalg.qr(torch.randn(2, n, 7, device=device, dtype=dtype)).Q
            v = torch.linalg.qr(torch.randn(2, m, 7, device=device, dtype=dtype)).Q
            spectrum = torch.tensor([9., 5., 2., .3, .1, .02, .005], device=device, dtype=dtype)
            matrix = (u * spectrum) @ v.transpose(-1, -2)
            rtol, atol = (3e-5, 1e-6) if dtype == torch.float32 else (1e-9, 1e-10)
            fit = s.linalg.svd(matrix, k=2, subspace_size=4, rtol=rtol, atol=atol, max_iter=80)
            assert fit.converged.all() and (fit.iterations > 0).all()
            torch.testing.assert_close(fit.singular_values, spectrum[:2].expand(2, 2), rtol=rtol, atol=atol)
            forward = (matrix @ fit.right_vectors - fit.left_vectors * fit.singular_values.unsqueeze(-2)).norm(dim=-2)
            adjoint = (matrix.transpose(-1, -2) @ fit.left_vectors - fit.right_vectors * fit.singular_values.unsqueeze(-2)).norm(dim=-2)
            threshold = (rtol * fit.singular_values).clamp_min(atol)
            assert (torch.maximum(forward, adjoint) <= threshold * 1.1 + 1e-13).all()
            identity = torch.eye(2, device=device, dtype=dtype).expand(2, 2, 2)
            torch.testing.assert_close(fit.left_vectors.transpose(-1, -2) @ fit.left_vectors, identity, rtol=rtol, atol=atol)
            torch.testing.assert_close(fit.right_vectors.transpose(-1, -2) @ fit.right_vectors, identity, rtol=rtol, atol=atol)
    x = torch.randn(2, 9, 3, device=device, dtype=torch.float64)
    y = torch.randn(1, 6, 3, device=device, dtype=x.dtype)
    source = s.lazy(x, y, metric='linear', chunk_size=3)
    fit = s.linalg.svd(source, k=3, subspace_size=4, rtol=1e-9, atol=1e-10)
    matrix = tp.linear_kernel(x, y)
    torch.testing.assert_close(fit.singular_values, torch.linalg.svdvals(matrix)[..., :3])
    torch.testing.assert_close((fit.left_vectors * fit.singular_values.unsqueeze(-2)) @ fit.right_vectors.transpose(-1, -2), matrix)
    zero = s.linalg.svd(torch.zeros(4, 6, device=device, dtype=x.dtype), k=3, atol=1e-12)
    assert zero.converged.all() and not zero.singular_values.any()
    limited = s.linalg.svd(matrix, k=2, subspace_size=2, max_iter=0, rtol=1e-14)
    assert (limited.status == s.linalg.SVDStatus.ITERATION_LIMIT).any()
    assert limited.singular_values.isfinite().all() and not limited.iterations.any()
    initial = torch.ones(6, 3, device=device, dtype=x.dtype)
    bad = s.linalg.svd(matrix, k=2, initial=initial)
    assert (bad.status == s.linalg.SVDStatus.RANK_DEFICIENT_INITIAL).all() and bad.singular_values.isnan().all()
    initial[0, 0] = float('nan')
    bad = s.linalg.svd(matrix, k=2, initial=initial)
    assert (bad.status == s.linalg.SVDStatus.NONFINITE).all()
    empty = s.linalg.svd(matrix[:0], k=2)
    assert empty.left_vectors.shape == (0, 9, 2)
    for options in ({'k': 0}, {'k': 7}, {'k': 2, 'subspace_size': 8}, {'k': 2, 'max_iter': -1},
                    {'k': 2, 'rtol': float('inf')}, {'k': 2, 'initial': torch.ones(8, 2, device=device, dtype=x.dtype)}):
        with pytest.raises(RuntimeError):
            s.linalg.svd(matrix, **options)
    with pytest.raises(RuntimeError, match='inference-only'):
        s.linalg.svd(matrix.requires_grad_(), k=2)


@pytest.mark.compiler
def test_svd_compile_export_rectangular_source(device):
    class Model(torch.nn.Module):
        def forward(self, x, y, mean, initial):
            cross = s.lazy(x, y, metric='linear', chunk_size=3).centered(mean)
            return s.linalg.svd(cross, k=2, initial=initial, rtol=1e-8, atol=1e-10, max_iter=40)
    model = Model()
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    def inputs(n, m):
        return tuple(torch.randn(*shape, device=device, dtype=torch.float64)
                     for shape in ((n, 2), (m, 2), (m,), (m, 4)))
    for n, m in ((8, 6), (6, 9)):
        values = inputs(n, m)
        # Same start; compare triplet equations rather than signs at null ranks.
        actual, expected = compiled(*values), model(*values)
        torch.testing.assert_close(actual, expected)
        assert actual.converged.all()
    n, m = (torch.export.Dim(name, min=4, max=20) for name in ('rows', 'columns'))
    exported = torch.export.export(model, inputs(7, 8), dynamic_shapes=({0: n}, {0: m}, {0: m}, {0: m}))
    buffer = io.BytesIO()
    torch.export.save(exported, buffer)
    buffer.seek(0)
    restored = torch.export.load(buffer).module()
    values = inputs(10, 7)
    actual, expected = restored(*values), model(*values)
    assert isinstance(actual, s.linalg.SVDResult)
    torch.testing.assert_close(actual, expected)
