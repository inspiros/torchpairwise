"""Short-recurrence indefinite solves and compact block-map actions."""
import io

import pytest
import torch
import torchpairwise as tp
import torchpairwise as s


def test_block_maps_transpose_gradients_and_shapes(device):
    torch.manual_seed(31)
    def t(*shape):
        return torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True)
    x, y, cross, bottom, v, u = t(2,4,3), t(1,5,3), t(4,2), t(3,2), t(7,2), t(7,3)
    a = s.lazy(x, y, gamma=0.2, chunk_size=3, metric='rbf')
    c = t(3,5)
    operator = s.block([[a,cross],[c,bottom]])
    dense_a = tp.rbf_kernel(x,y,gamma=.2)
    upper = torch.cat((dense_a,cross.expand(2,4,2)),-1)
    lower = torch.cat((c,bottom),-1).expand(2,3,7)
    dense = torch.cat((upper,lower),-2)
    actual = operator.matmul(v), operator.T.matmul(u)
    expected = dense@v, dense.transpose(-1,-2)@u
    torch.testing.assert_close(actual,expected)
    operands = x,y,cross,c,bottom,v,u
    torch.testing.assert_close(
        torch.autograd.grad(sum(a.square().sum() for a in actual),operands),
        torch.autograd.grad(sum(a.square().sum() for a in expected),operands))
    assert operator.shape == (2,7,7)
    for rows in (0,2):
        op = s.block([[torch.empty(rows,0,device=device,dtype=x.dtype),torch.ones(rows,3,device=device,dtype=x.dtype)],
                      [torch.empty(1,0,device=device,dtype=x.dtype),torch.ones(1,3,device=device,dtype=x.dtype)]])
        values=torch.randn(3,2,device=device,dtype=x.dtype)
        torch.testing.assert_close(op.matmul(values),torch.ones(rows+1,3,device=device,dtype=x.dtype)@values)
        torch.testing.assert_close(op.T.matmul(torch.ones(rows+1,1,device=device,dtype=x.dtype)),
                                   torch.full((3,1),float(rows+1),device=device,dtype=x.dtype))
    for grid in ([],[[]],[[torch.ones(2,2)],[]]):
        with pytest.raises(ValueError,match='grid'):
            s.block(grid)
    with pytest.raises(RuntimeError,match='widths'):
        s.block([[torch.ones(2,2)],[torch.ones(3,3)]])
    with pytest.raises(RuntimeError,match='heights'):
        s.block([[torch.ones(2,2),torch.ones(3,2)]])


def test_minres_indefinite_oracles_blocks_and_diagnostics(device):
    torch.manual_seed(511)
    for dtype in (torch.float32,torch.float64):
        q,_=torch.linalg.qr(torch.randn(2,9,9,device=device,dtype=dtype))
        diagonal=torch.tensor([-4.,-2.,-1.,-.5,.4,.8,1.,2.,4.],device=device,dtype=dtype)
        a=(q*diagonal)@q.transpose(-1,-2)
        b=torch.randn(9,2,device=device,dtype=dtype)
        initial=torch.randn(9,2,device=device,dtype=dtype)
        tolerance=2e-6 if dtype==torch.float32 else 1e-10
        fit=s.linalg.minres(a,b,initial=initial,rtol=tolerance,max_iter=80)
        assert fit.converged.all(),fit
        torch.testing.assert_close(fit.solution,torch.linalg.solve(a,b),
            rtol=2e-4 if dtype==torch.float32 else 1e-8,atol=2e-5 if dtype==torch.float32 else 1e-9)
        assert (fit.residual_norm<=tolerance*torch.linalg.vector_norm(b.double(),dim=0)).all()
        dense_norm=torch.linalg.vector_norm((b-a@fit.solution).double(),dim=-2)
        roundoff=8*torch.finfo(dtype).eps*(torch.linalg.matrix_norm(a.double())*
            torch.linalg.matrix_norm(fit.solution.double())+torch.linalg.vector_norm(b.double(),dim=0).max())
        assert ((fit.residual_norm-dense_norm).abs()<=roundoff.unsqueeze(-1)+1e-12).all()
    x=torch.randn(2,8,3,device=device,dtype=torch.float64)
    a=s.lazy(x, gamma=0.3, chunk_size=3, metric='rbf') + 0.4 * tp.eye_like(s.lazy(x, gamma=0.3, chunk_size=3, metric='rbf'))
    constraint=s.as_lazy(torch.ones(1, 8, device=device, dtype=x.dtype))
    kkt=s.block([[a,constraint.T],[constraint,torch.zeros(1,1,device=device,dtype=x.dtype)]])
    rhs=torch.randn(9,2,device=device,dtype=x.dtype)
    matrix=tp.rbf_kernel(x,gamma=.3)+.4*torch.eye(8,device=device,dtype=x.dtype)
    dense=torch.cat((torch.cat((matrix,torch.ones(2,8,1,device=device,dtype=x.dtype)),-1),
                     torch.cat((torch.ones(2,1,8,device=device,dtype=x.dtype),torch.zeros(2,1,1,device=device,dtype=x.dtype)),-1)),-2)
    fit=s.linalg.minres(kkt,rhs,rtol=1e-10,max_iter=80)
    assert fit.converged.all(),fit
    torch.testing.assert_close(fit.solution,torch.linalg.solve(dense,rhs),rtol=1e-8,atol=1e-9)
    exact=s.linalg.minres(kkt,rhs,initial=fit.solution,max_iter=0,rtol=1e-9)
    assert exact.converged.all() and not exact.iterations.any()
    def t(value):
        return torch.tensor(value,device=device,dtype=x.dtype)
    identity=torch.eye(3,device=device,dtype=x.dtype)
    matrices=torch.stack((identity,identity*0,torch.diag(t([-1.,2.,4.]))))
    rhs=t([[0.,1.,float('nan')],[0.,2.,1.],[0.,3.,1.]])
    result=s.linalg.minres(matrices,rhs,rtol=1e-12,max_iter=1)
    assert result.status.tolist()==[[0,0,3],[0,2,3],[0,1,3]]
    assert result.iterations.tolist()==[[0,1,0],[0,1,0],[0,1,0]]
    torch.testing.assert_close(result.converged,result.status==0)
    # Zero first Lanczos diagonal for an invertible indefinite matrix.
    swap=s.linalg.minres(t([[0.,1.],[1.,0.]]),t([1.,0.]),rtol=1e-12)
    assert swap.converged and swap.iterations==2
    torch.testing.assert_close(swap.solution,t([0.,1.]))
    consistent=s.linalg.minres(torch.diag(t([1.,0.,0.])),t([2.,0.,0.]))
    assert consistent.converged
    for shape in ((0,0),(0,3,3)):
        fit=s.linalg.minres(torch.empty(shape,device=device,dtype=x.dtype),
            torch.empty((*shape[:-1],1),device=device,dtype=x.dtype))
        assert fit.converged.all() and fit.solution.shape==(*shape[:-1],1)
    with pytest.raises(RuntimeError,match='inference-only'):
        s.linalg.minres(identity.requires_grad_(),t([1.,2.,3.]))
    with torch.no_grad():
        assert not s.linalg.minres(identity,t([1.,2.,3.])).solution.requires_grad
    for options in ({'rtol':-1},{'atol':float('nan')},{'max_iter':-1}):
        with pytest.raises(RuntimeError,match='nonnegative'):
            s.linalg.minres(identity.detach(),t([1.,2.,3.]),**options)
    with pytest.raises(TypeError,match='integer'):
        s.linalg.minres(identity.detach(),t([1.,2.,3.]),max_iter=True)
    with pytest.raises(RuntimeError,match='square'):
        s.linalg.minres(torch.ones(2,3,device=device,dtype=x.dtype),t([1.,2.]))
    with pytest.raises(RuntimeError,match='initial'):
        s.linalg.minres(identity.detach(),t([1.,2.,3.]),initial=t([1.,2.]))


@pytest.mark.compiler
def test_minres_block_compile_export(device):
    class Model(torch.nn.Module):
        def forward(self,x,b,v):
            a=s.lazy(x, gamma=0.3, chunk_size=3, metric='rbf') + 0.5 * tp.eye_like(s.lazy(x, gamma=0.3, chunk_size=3, metric='rbf'))
            c=s.as_lazy(torch.ones(1, x.shape[0], device=x.device, dtype=x.dtype))
            kkt=s.block([[a,c.T],[c,torch.zeros(1,1,device=x.device,dtype=x.dtype)]])
            rhs=torch.cat((b,b.new_zeros(1,b.shape[-1])),-2)
            return s.linalg.minres(kkt,rhs,rtol=1e-9,max_iter=80),kkt.matmul(v),kkt.T.matmul(v)
    torch.manual_seed(18)
    def inputs(n):
        return (torch.randn(n,3,device=device,dtype=torch.float64),
                torch.randn(n,2,device=device,dtype=torch.float64),
                torch.randn(n+1,2,device=device,dtype=torch.float64))
    model,args=Model(),inputs(7)
    compiled=torch.compile(model,backend='inductor',fullgraph=True,dynamic=True)
    for n in (7,10):
        values=inputs(n)
        expected=model(*values)
        assert expected[0].converged.all()
        torch.testing.assert_close(compiled(*values),expected)
    n=torch.export.Dim('samples',min=3,max=16)
    exported=torch.export.export(model,args,dynamic_shapes=({0:n},{0:n},{0:n+1}))
    buffer=io.BytesIO()
    torch.export.save(exported,buffer)
    buffer.seek(0)
    restored=torch.export.load(buffer).module()
    values=inputs(11)
    torch.testing.assert_close(restored(*values),model(*values))
    # Block products retain first gradients independently of inference-only
    # solver calls, including compiled native concat/transpose primitives.
    class Actions(torch.nn.Module):
        def forward(self,a,c,v):
            aa=s.as_lazy(a)
            cc=s.as_lazy(c)
            op=s.block([[aa,cc.T],[cc,a.new_zeros(c.shape[-2],c.shape[-2])]])
            return op.matmul(v),op.T.matmul(v)
    action=Actions()
    compiled_action=torch.compile(action,backend='inductor',fullgraph=True,dynamic=True)
    a=torch.randn(4,4,device=device,dtype=torch.float64,requires_grad=True)
    c=torch.randn(2,4,device=device,dtype=a.dtype,requires_grad=True)
    v=torch.randn(6,2,device=device,dtype=a.dtype,requires_grad=True)
    dense=torch.cat((torch.cat((a,c.T),-1),torch.cat((c,a.new_zeros(2,2)),-1)),-2)
    oracle=dense@v,dense.T@v
    actual=compiled_action(a,c,v)
    torch.testing.assert_close(actual,oracle)
    torch.testing.assert_close(torch.autograd.grad(sum(z.square().sum() for z in actual),(a,c,v)),
                               torch.autograd.grad(sum(z.square().sum() for z in oracle),(a,c,v)))
