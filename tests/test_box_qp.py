"""Convex box/equality QPs: independent face oracles, diagnostics and deployment."""
import io
import itertools

import pytest
import torch
import torchpairwise as tp
import torchpairwise as s


def face_oracle(matrix, q, lower, upper, a=None, target=0.):
    """Small CPU-only exhaustive face enumeration, independent of projection."""
    matrix, q, lower, upper = (t.detach().double().cpu() for t in (matrix,q,lower,upper))
    a = None if a is None else a.detach().double().cpu()
    best, best_value = None, float('inf')
    for face in itertools.product((-1,0,1), repeat=q.numel()):
        free = torch.tensor([i for i,v in enumerate(face) if v==0], dtype=torch.int64)
        fixed = torch.tensor([i for i,v in enumerate(face) if v!=0], dtype=torch.int64)
        x = torch.stack([lower[i] if v==-1 else upper[i] if v==1 else q.new_zeros(()) for i,v in enumerate(face)])
        rhs = -q[free] - matrix[free][:,fixed] @ x[fixed]
        if free.numel():
            block = matrix[free][:,free]
            remaining = target if a is None else target-a[fixed]@x[fixed]
            if a is not None and a[free].ne(0).any():
                af = a[free]
                block = torch.cat((torch.cat((block,af[:,None]),1),torch.cat((af,q.new_zeros(1)))[None,:]),0)
                rhs = torch.cat((rhs,torch.as_tensor(remaining).reshape(1)))
                x[free] = torch.linalg.solve(block,rhs)[:-1]
            else:
                if a is not None and abs(remaining)>1e-8:
                    continue
                x[free] = torch.linalg.solve(block,rhs)
        if (x<lower-1e-8).any() or (x>upper+1e-8).any():
            continue
        if a is not None and abs(a@x-target)>1e-8:
            continue
        value = (.5*x@matrix@x+q@x).item()
        if value<best_value:
            best,best_value=x.clone(),value
    assert best is not None
    return best,best_value


def test_box_qp_dense_faces_and_broadcast_columns(device):
    torch.manual_seed(113)
    dtype=torch.float64
    base=torch.randn(2,4,4,device=device,dtype=dtype)
    matrix=base.transpose(-1,-2)@base+.5*torch.eye(4,device=device,dtype=dtype)
    q=torch.randn(1,4,2,device=device,dtype=dtype)*2
    lower=torch.zeros(4,device=device,dtype=dtype)
    upper=torch.tensor([.7,1.,.6,1.2],device=device,dtype=dtype)
    a=torch.tensor([1.,-1.,1.,-1.],device=device,dtype=dtype)
    target=torch.tensor([0.,.2],device=device,dtype=dtype)
    for equation in (None,(a,target)):
        fit=s.optimize.box_qp(matrix,q,lower=lower,upper=upper,equality=equation,rtol=1e-8,atol=1e-10,max_iter=1500)
        assert fit.converged.all(), (fit.status,fit.optimality,fit.feasibility)
        assert fit.solution.shape==(2,4,2) and fit.objective.shape==(2,2)
        for batch in range(2):
            for column in range(2):
                expected,value=face_oracle(matrix[batch],q[0,:,column],lower,upper,a if equation else None,target[column].item() if equation else 0.)
                torch.testing.assert_close(fit.solution[batch,:,column].cpu(),expected,rtol=2e-6,atol=2e-7)
                torch.testing.assert_close(fit.objective[batch,column].cpu(),torch.tensor(value,dtype=dtype),rtol=1e-7,atol=1e-9)
        # Scaling the affine equation cannot change the solution/diagnostics.
        if equation:
            scaled=s.optimize.box_qp(matrix,q,lower=lower,upper=upper,equality=(a*1e-8,target*1e-8),rtol=1e-8,atol=1e-10,max_iter=1500)
            torch.testing.assert_close(scaled.solution,fit.solution,rtol=2e-6,atol=2e-7)
    for dtype in (torch.float32,torch.float64):
        diagonal=torch.diag(torch.tensor([1.,2.,4.],device=device,dtype=dtype))
        linear=torch.tensor([-2.,1.,-1.],device=device,dtype=dtype)
        fit=s.optimize.box_qp(diagonal,linear,lower=0.,upper=1.)
        torch.testing.assert_close(fit.solution,linear.new_tensor([1.,0.,.25]))
        assert fit.converged and fit.status==0 and fit.objective.dtype==torch.float64

    # Regression: equality-normal gradient components must not turn a tiny
    # float32 feasibility rounding error into a positive line derivative.
    torch.manual_seed(127)
    x=torch.randn(2,8,3,device=device,dtype=torch.float32)
    linear=-torch.ones(8,2,device=device,dtype=x.dtype)
    labels=(torch.arange(8,device=device)%2).to(x.dtype)*2-1
    operator=s.lazy(x, gamma=0.25, chunk_size=3, metric='rbf') + 0.5 * tp.eye_like(s.lazy(x, gamma=0.25, chunk_size=3, metric='rbf'))
    dense=tp.rbf_kernel(x, gamma=0.25) + 0.5 * torch.eye(8, device=device, dtype=x.dtype)
    for matrix in (operator,dense):
        fit=s.optimize.box_qp(matrix,linear,lower=0.,upper=.8,equality=(labels,0.),rtol=1e-5,atol=1e-9,max_iter=500)
        assert fit.converged.all(),(fit.status,fit.optimality)
        assert fit.optimality.max() <= 1e-5
        torch.testing.assert_close(fit.objective, (.5*fit.solution.double()*matrix.matmul(fit.solution).double()
            + linear.double()*fit.solution.double()).sum(-2))


def test_box_qp_diagnostics_projection_and_contracts(device):
    dtype=torch.float64
    eye=torch.eye(2,device=device,dtype=dtype)
    matrices=torch.stack((eye,-eye,eye*float('nan'),eye,eye))
    q=torch.tensor([-1.,-1.],device=device,dtype=dtype)
    lo=torch.zeros(5,2,1,device=device,dtype=dtype)
    hi=torch.ones_like(lo)*2;hi[3].fill_(-1)
    initial=torch.zeros_like(lo);initial[4].fill_(float('nan'))
    fit=s.optimize.box_qp(matrices,q,lower=lo,upper=hi,initial=initial,rtol=1e-8)
    assert fit.status.tolist()==[0,4,3,2,3]
    assert fit.solution[1:].isnan().all() and torch.equal(fit.converged,fit.status==0)
    a=torch.tensor([1.,-1.],device=device,dtype=dtype)
    # Initial projection and finite box endpoints, even with zero iterations.
    projected=s.optimize.box_qp(eye,q,lower=0.,upper=1.,equality=(a,.5),max_iter=0)
    torch.testing.assert_close(projected.solution,q.new_tensor([.5,0.]))
    assert projected.feasibility<1e-12 and projected.status==1
    endpoint=s.optimize.box_qp(eye,q,lower=0.,upper=1.,equality=(a,1.))
    torch.testing.assert_close(endpoint.solution,q.new_tensor([1.,0.]))
    assert endpoint.converged
    impossible=s.optimize.box_qp(eye,q,lower=0.,upper=1.,equality=(a,3.))
    assert impossible.status==s.optimize.QPStatus.INFEASIBLE
    redundant=s.optimize.box_qp(eye,q,lower=0.,upper=1.,equality=(a*0,0.))
    assert redundant.converged
    zero=s.optimize.box_qp(torch.empty(0,0,device=device,dtype=dtype),torch.empty(0,device=device,dtype=dtype),lower=0.,upper=1.)
    assert zero.converged and zero.objective==0 and zero.optimality==0
    with pytest.raises(RuntimeError,match='finite'):
        s.optimize.box_qp(eye,q,lower=0.,upper=float('inf'))
    with pytest.raises(RuntimeError,match='nonnegative'):
        s.optimize.box_qp(eye,q,lower=0.,upper=1.,max_iter=-1)
    with pytest.raises(TypeError,match='integer'):
        s.optimize.box_qp(eye,q,lower=0.,upper=1.,max_iter=True)
    with pytest.raises(RuntimeError,match='inference-only'):
        s.optimize.box_qp(eye,q.clone().requires_grad_(),lower=0.,upper=1.)
    with torch.no_grad():
        assert not s.optimize.box_qp(eye,q.clone().requires_grad_(),lower=0.,upper=1.).solution.requires_grad


@pytest.mark.compiler
def test_box_qp_streamed_compile_export(device):
    class Solve(torch.nn.Module):
        def forward(self,x,q,a):
            kernel=s.lazy(x, gamma=0.25, chunk_size=3, metric='rbf')
            operator=kernel + 0.5 * tp.eye_like(kernel)
            return s.optimize.box_qp(operator,q,lower=0.,upper=.8,equality=(a,0.),rtol=1e-6,atol=1e-9,max_iter=300)
    model=Solve()
    compiled=torch.compile(model,fullgraph=True,dynamic=True,backend='inductor')
    def args(n):
        x=torch.randn(n,3,device=device,dtype=torch.float64)
        q=-torch.ones(n,1,device=device,dtype=x.dtype)
        a=(torch.arange(n,device=device)%2).to(x.dtype)*2-1
        return x,q,a
    for n in (6,9):
        inputs=args(n)
        actual,expected=compiled(*inputs),model(*inputs)
        torch.testing.assert_close(actual,expected)
        assert actual.converged.all(),(actual.status,actual.optimality)
    inputs=args(7)
    n=torch.export.Dim('samples',min=4,max=16)
    exported=torch.export.export(model,inputs,dynamic_shapes=({0:n},{0:n},{0:n}))
    buffer=io.BytesIO();torch.export.save(exported,buffer);buffer.seek(0)
    restored=torch.export.load(buffer).module()
    inputs=args(8)
    torch.testing.assert_close(restored(*inputs),model(*inputs))
