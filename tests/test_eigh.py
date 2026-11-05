"""Leading PSD eigenpairs against dense spectra and native diagnostics."""
import io

import pytest
import torch
import torchpairwise as tp
import torchpairwise as s


def test_eigh_leading_kernel_spectrum_and_low_rank(device):
    torch.manual_seed(97)
    x = torch.randn(2, 17, 3, dtype=torch.float64, device=device)
    x[1].zero_()  # One zero centered batch must finish independently.
    operator = 0.75 * s.lazy(x, gamma=0.25, chunk_size=5, metric='rbf').centered()
    initial = torch.randn(1, 17, 8, dtype=x.dtype, device=device)
    result = s.linalg.eigh(operator, k=3, initial=initial, rtol=1e-9, atol=1e-12, max_iter=150)
    dense = tp.rbf_kernel(x, gamma=.25)
    dense = .75 * (dense - dense.mean(-1, keepdim=True) - dense.mean(-2, keepdim=True) + dense.mean((-1,-2), keepdim=True))
    values, vectors = torch.linalg.eigh(dense)
    expected_values, expected_vectors = values[..., -3:].flip(-1), vectors[..., -3:].flip(-1)
    torch.testing.assert_close(result.eigenvalues, expected_values, rtol=1e-7, atol=1e-10)
    torch.testing.assert_close((result.eigenvectors[0].T @ expected_vectors[0]).square().sum(), x.new_tensor(3.))
    torch.testing.assert_close(result.residual_norm,
        torch.linalg.vector_norm(operator.matmul(result.eigenvectors)-result.eigenvectors*result.eigenvalues.unsqueeze(-2), dim=-2), atol=1e-13, rtol=1e-3)
    torch.testing.assert_close(result.eigenvectors.transpose(-1,-2) @ result.eigenvectors,
                               torch.eye(3,device=device,dtype=x.dtype).expand(2,3,3))
    assert result.converged.all() and not result.status.any()
    assert result.eigenvalues.shape == result.iterations.shape == (2,3)
    assert result.eigenvectors.shape == (2,17,3) and result.residual_norm.dtype == torch.float64
    assert (result.iterations[1] == 0).all() and (result.iterations[0] > 0).all()
    for dtype in (torch.float32, torch.float64):
        # Repeated leading eigenvalues and a nullspace: compare invariant spaces.
        matrix = torch.diag(torch.tensor([9.,9.,4.,1.,0.,0.],device=device,dtype=dtype))
        fit = s.linalg.eigh(matrix, k=2, subspace_size=4, rtol=1e-5 if dtype==torch.float32 else 1e-9, atol=1e-10, max_iter=100)
        torch.testing.assert_close(fit.eigenvalues, matrix.diagonal()[:2])
        torch.testing.assert_close(fit.eigenvectors[:2].square().sum(), matrix.new_tensor(2.))
        assert fit.converged.all()
        # A zero map never divides by Ritz values and returns orthonormal zeros.
        zero = s.linalg.eigh(matrix*0, k=2, subspace_size=4)
        assert zero.converged.all() and not zero.eigenvalues.any() and not zero.residual_norm.any()


def test_eigh_diagnostics_initial_and_contracts(device):
    dtype = torch.float64
    eye = torch.eye(6, device=device, dtype=dtype)
    matrices = torch.stack([eye, -eye, eye*float('nan'), eye, torch.diag(torch.arange(1.,7.,device=device,dtype=dtype))])
    initial = torch.randn(5,6,3,device=device,dtype=dtype)
    initial[3].zero_()
    fit = s.linalg.eigh(matrices, k=2, initial=initial, max_iter=0, rtol=1e-9, atol=1e-12)
    assert fit.status.tolist() == [[0,0],[4,4],[3,3],[2,2],[1,1]]
    assert torch.equal(fit.converged, fit.status==0) and not fit.iterations.any()
    assert fit.eigenvalues[1:4].isnan().all()
    # A fully spanning initial basis makes the first Rayleigh-Ritz pass exact.
    exact = s.linalg.eigh(matrices[-1], k=3, initial=eye, max_iter=0)
    torch.testing.assert_close(exact.eigenvalues, torch.tensor([6.,5.,4.],device=device,dtype=dtype))
    assert exact.converged.all()
    empty = s.linalg.eigh(eye.expand(0,6,6), k=2, subspace_size=3)
    assert empty.eigenvalues.shape==(0,2) and empty.eigenvectors.shape==(0,6,2)
    bad_initial = initial[0].clone(); bad_initial[0,0]=float('inf')
    assert (s.linalg.eigh(eye,k=2,initial=bad_initial).status==3).all()
    for options in ({'k':0},{'k':7},{'k':2,'subspace_size':1},{'k':2,'subspace_size':7},
                    {'k':2,'rtol':-1},{'k':2,'atol':float('nan')},{'k':2,'max_iter':-1}):
        with pytest.raises(RuntimeError):
            s.linalg.eigh(eye,**options)
    with pytest.raises(TypeError,match='integer'):
        s.linalg.eigh(eye,k=True)
    with pytest.raises(RuntimeError,match='initial'):
        s.linalg.eigh(eye,k=2,initial=initial)
    grad = eye.clone().requires_grad_()
    with pytest.raises(RuntimeError,match='inference-only'):
        s.linalg.eigh(grad,k=2)
    with torch.no_grad():
        assert not s.linalg.eigh(grad,k=2).eigenvectors.requires_grad
    # Caller-supplied invariant starts need overlap with leading directions.
    # Residual diagnostics cannot prove a global leading-eigenvalue certificate.
    missed = s.linalg.eigh(matrices[-1],k=1,initial=eye[:,:1],max_iter=0)
    assert missed.converged.all() and missed.eigenvalues.item()==1


@pytest.mark.compiler
def test_eigh_compile_and_export(device):
    class Spectrum(torch.nn.Module):
        def forward(self, x, initial):
            a = s.lazy(x, gamma=0.25, chunk_size=3, metric='rbf').centered()
            return s.linalg.eigh(a,k=2,initial=initial,rtol=1e-8,atol=1e-11,max_iter=100)
    model = Spectrum()
    compiled = torch.compile(model,fullgraph=True,dynamic=True,backend='inductor')
    def args(n):
        return (torch.randn(n,3,device=device,dtype=torch.float64),
                torch.randn(n,5,device=device,dtype=torch.float64))
    for n in (9,12):
        inputs = args(n)
        actual, expected = compiled(*inputs), model(*inputs)
        torch.testing.assert_close(actual.eigenvalues,expected.eigenvalues)
        torch.testing.assert_close(actual.residual_norm,expected.residual_norm,atol=1e-12,rtol=1e-3)
        torch.testing.assert_close(actual.converged,expected.converged)
    inputs=args(9)
    n=torch.export.Dim('samples',min=6,max=20)
    exported=torch.export.export(model,inputs,dynamic_shapes=({0:n},{0:n}))
    buf=io.BytesIO();torch.export.save(exported,buf);buf.seek(0)
    restored=torch.export.load(buf).module()
    inputs=args(11)
    actual,expected=restored(*inputs),model(*inputs)
    torch.testing.assert_close(actual.eigenvalues,expected.eigenvalues)
    torch.testing.assert_close(actual.eigenvectors,expected.eigenvectors)
    torch.testing.assert_close(actual.status,expected.status)

    # A native random start consumes the runtime PyTorch RNG. Compilation must
    # preserve seeded behavior rather than treating the solve as pure CSE-able
    # constant work. Initial blocks above cover deterministic export/save-load.
    class RandomStart(torch.nn.Module):
        def forward(self, x):
            a = s.lazy(x, gamma=0.25, chunk_size=3, metric='rbf').centered()
            return s.linalg.eigh(a, k=2, subspace_size=5, rtol=1e-8, atol=1e-11, max_iter=100)
    random_model = RandomStart()
    random_compiled = torch.compile(random_model, fullgraph=True, dynamic=True, backend='inductor')
    x = args(9)[0]
    torch.manual_seed(109)
    expected = random_model(x)
    torch.manual_seed(109)
    actual = random_compiled(x)
    torch.testing.assert_close(actual.eigenvalues, expected.eigenvalues)
    torch.testing.assert_close(actual.eigenvectors, expected.eigenvectors)
    torch.testing.assert_close(actual.iterations, expected.iterations)
