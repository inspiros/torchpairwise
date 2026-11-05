"""Dense oracles for general scalar operator streaming and native deployment."""
import io
import pytest
import torch
import torchpairwise as tp
from torchpairwise.ops._registry import DISTANCES, KERNELS


def test_operator_registry_products(device):
    torch.manual_seed(61)
    seen=set()
    for registry in (DISTANCES,KERNELS):
        for spec in registry.values():
            if spec.op in seen:continue
            seen.add(spec.op)
            d=2 if spec.op=='haversine_distances' else 3
            x=torch.rand(3,d,device=device,dtype=torch.float64)+.3
            y=torch.rand(4,d,device=device,dtype=torch.float64)+.3
            kw={}
            if spec.event_ndim==3:
                x=x.unsqueeze(-2).repeat(1,2,1);y=y.unsqueeze(-2).repeat(1,2,1)
            if spec.boolean:x=x>.7;y=y>.7;x[...,0]=True;y[...,0]=True
            if spec.op=='wminkowski_distances':kw['w']=torch.ones(d,device=device,dtype=x.dtype)
            if spec.op=='seuclidean_distances':kw['V']=torch.ones(d,device=device,dtype=x.dtype)
            if spec.op=='mahalanobis_distances':kw['VI']=torch.eye(d,device=device,dtype=x.dtype)
            if spec.op=='sliced_wasserstein_distances':kw['projections']=torch.eye(d,device=device,dtype=x.dtype)
            dense=tp.compare(x,y,metric=spec.op,**kw)
            if isinstance(dense,tuple):dense=dense[0]  # directed Hausdorff also returns witnesses
            a=tp.lazy(x, y, metric=spec.op, chunk_size=2, **kw)
            payload=torch.randn(4,2,device=device,dtype=dense.dtype)
            torch.testing.assert_close(a@payload,dense@payload,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(a.sum(-2),dense.sum(-2),rtol=1e-5,atol=1e-6)
            assert a.shape==dense.shape and a.dtype==dense.dtype, spec.op


def test_operator_directed_transpose_forms_and_gradients(device):
    torch.manual_seed(62)
    x=(torch.rand(2,1,3,4,device=device,dtype=torch.float64)+.2).requires_grad_()
    y=(torch.rand(1,2,5,4,device=device,dtype=torch.float64)+.2).requires_grad_()
    l=torch.randn(2,1,5,3,device=device,dtype=x.dtype,requires_grad=True)
    r=torch.randn(1,2,3,2,device=device,dtype=x.dtype,requires_grad=True)
    a=tp.lazy(x, y, metric='kl', base=2.0, chunk_size=2).T
    dense=tp.kl_divergences(x,y,base=2.).transpose(-1,-2)
    assert a.shape==dense.shape and a.T.T.shape==a.shape
    values=[a@r,a.bilinear(l,r),a.sum(-1),a.sum(-2),a.mean(),a.diagonal()]
    refs=[dense@r,l.transpose(-1,-2)@dense@r,dense.sum(-1),dense.sum(-2),dense.mean((-2,-1)),dense.diagonal(dim1=-2,dim2=-1)]
    torch.testing.assert_close(values,refs)
    gs=[torch.randn_like(v) for v in values]
    torch.testing.assert_close(torch.autograd.grad(values,(x,y,l,r),gs),torch.autograd.grad(refs,(x,y,l,r),gs),rtol=1e-8,atol=1e-9)
    rows=torch.tensor([4,1,1],device=device);cols=torch.tensor([2,0],device=device)
    torch.testing.assert_close(a.block(rows,cols),dense.index_select(-2,rows).index_select(-1,cols))
    square=tp.lazy(x, x.clone(), metric='kl', chunk_size=2).T
    q=torch.randn(3,device=device,dtype=x.dtype,requires_grad=True)
    torch.testing.assert_close(square.quadratic(q),torch.einsum('i,...ij,j->...',q,tp.kl_divergences(x,x.clone()).transpose(-1,-2),q))


def test_operator_weights_statistics_and_discrete_payload_gradients(device):
    torch.manual_seed(63)
    x=torch.randn(6,3,device=device,dtype=torch.float64,requires_grad=True)
    y=torch.randn(4,3,device=device,dtype=x.dtype,requires_grad=True)
    w=torch.rand(2,1,3,device=device,dtype=x.dtype,requires_grad=True)+.3
    l=torch.randn(2,1,4,2,device=device,dtype=x.dtype,requires_grad=True)
    r=torch.randn(6,2,device=device,dtype=x.dtype,requires_grad=True)
    a=tp.lazy(x, y, metric='sqeuclidean', w=w, chunk_size=2).T
    dense=tp.sqeuclidean_distances(x,y,w=w).transpose(-1,-2)
    actual=a.bilinear(l,r);expected=l.transpose(-1,-2)@dense@r
    g=torch.randn_like(actual)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,(x,y,w,l,r),g),torch.autograd.grad(expected,(x,y,w,l,r),g))
    for metric in ('seuclidean','mahalanobis'):
        dense=tp.compare(x,metric=metric)
        a=tp.lazy(x, metric=metric, chunk_size=2)
        torch.testing.assert_close(a.sum(-1),dense.sum(-1))
        ids=torch.tensor([4,0,0],device=device)
        torch.testing.assert_close(a.block(ids,ids),dense.index_select(-2,ids).index_select(-1,ids))
        torch.testing.assert_close(a.diagonal(),dense.diagonal())
    discrete=tp.lazy(x, y, metric='hamming', chunk_size=2)
    assert not discrete.bilinear(torch.ones(6,device=device),torch.ones(4,device=device)).requires_grad
    bx=torch.tensor([[1,2],[0,3],[4,3]],device=device)
    by=torch.tensor([[1,3],[4,0]],device=device)
    v=torch.randn(3,2,device=device,requires_grad=True)
    dense=tp.hamming_distances(bx,by).T
    got=tp.lazy(bx, by, metric='hamming', chunk_size=1).T@v
    torch.testing.assert_close(got,dense@v)
    torch.testing.assert_close(torch.autograd.grad(got.sum(),v)[0],dense.T@torch.ones_like(got))


def test_operator_reduction_contracts_empty_ties_and_retained_state(device):
    x=torch.tensor([[0.,1.],[1.,0.],[0.,1.]],device=device,requires_grad=True)
    y=torch.tensor([[0.,1.],[1.,0.],[0.,1.]],device=device,requires_grad=True)
    dense=tp.sqeuclidean_distances(x,y)
    for transpose in (False,True):
        a=tp.lazy(x, y, metric='sqeuclidean', chunk_size=1)
        if transpose:a=a.T
        matrix=dense.T if transpose else dense
        for dim in (-1,-2,None):
            for reduction in ('sum','mean','min','max','prod','all','any'):
                data=matrix.flatten(-2) if dim is None else matrix
                axis=-1 if dim is None else dim
                expected=getattr(data,reduction)(axis)
                if reduction in ('min','max'):expected=expected.values
                actual=getattr(a,{'min': 'amin', 'max': 'amax'}.get(reduction, reduction))(dim)
                torch.testing.assert_close(actual,expected)
                if reduction in ('min','max','prod'):
                    torch.testing.assert_close(torch.autograd.grad(actual.sum(),(x,y),retain_graph=True),torch.autograd.grad(expected.sum(),(x,y),retain_graph=True))
        torch.testing.assert_close(a.sum(exclude_self=True),matrix.sum()-matrix.diagonal().sum())
        assert a.mean(-2,keepdim=True).shape==(1,3)
    for xx,yy in ((x[:0],y),(x,y[:0]),(x[:0],y[:0])):
        a=tp.lazy(xx, yy)
        assert a.sum().item()==0 and a.prod().item()==1 and a.all().item() and not a.any().item()
        assert a.mean().isnan()
        with pytest.raises((RuntimeError,IndexError)):a.min()
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append(t.shape) or t),lambda t:t):
        value=tp.lazy(x, y, chunk_size=1).bilinear(torch.ones(3, device=device), torch.ones(3, device=device))
    assert torch.Size([3,3]) not in saved
    with pytest.raises(RuntimeError,match='Higher derivatives'):torch.autograd.grad(value,x,create_graph=True)
    assert tp.lazy(torch.rand(2, 3, 1, device=device), metric='dtw').shape == (2,2)


def test_operator_cloud_lengths_transpose_and_gradients(device):
    torch.manual_seed(64)
    x=torch.rand(3,4,2,device=device,dtype=torch.float64,requires_grad=True)
    y=torch.rand(2,3,2,device=device,dtype=x.dtype,requires_grad=True)
    xl=torch.tensor([2,4,3],device=device);yl=torch.tensor([3,1],device=device)
    with torch.no_grad():x[0,2:]=float('nan');x[2,3:]=float('nan');y[1,1:]=float('nan')
    payload=torch.randn(3,2,device=device,dtype=x.dtype,requires_grad=True)
    for metric in ('chamfer','directed_hausdorff','sinkhorn'):
        kw=dict(x1_lengths=xl,x2_lengths=yl)
        if metric=='sinkhorn':kw['max_iter']=4
        dense=tp.cdist(x,y,metric=metric,**kw).T
        a=tp.lazy(x, y, metric=metric, chunk_size=1, metric_chunk_size=2, **kw).T
        actual=a.matmul(payload);expected=dense@payload
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(torch.autograd.grad(actual.square().sum(),(x,y,payload)),torch.autograd.grad(expected.square().sum(),(x,y,payload)))
        rows=torch.tensor([1,0,1],device=device);cols=torch.tensor([2,0],device=device)
        torch.testing.assert_close(a.block(rows,cols),dense.index_select(0,rows).index_select(1,cols))


@pytest.mark.compiler
def test_pairwise_operator_compile_export(device):
    class Model(torch.nn.Module):
        def forward(self,x,y,l,r):
            a=tp.lazy(x, y, metric='kl', chunk_size=2).T
            return a.matmul(r),a.bilinear(l,r),a.mean(-2)
    torch._dynamo.reset();model=Model()
    args=(torch.rand(3,4,device=device)+.3,torch.rand(5,4,device=device)+.3,torch.randn(5,2,device=device),torch.randn(3,2,device=device))
    compiled=torch.compile(model,fullgraph=True,dynamic=True)
    torch.testing.assert_close(compiled(*args),model(*args))
    leaves=tuple(v.clone().requires_grad_() for v in args)
    loss=lambda values:sum(v.square().sum() for v in values)
    torch.testing.assert_close(torch.autograd.grad(loss(compiled(*leaves)),leaves),torch.autograd.grad(loss(model(*leaves)),leaves),rtol=1e-4,atol=1e-5)
    n=torch.export.Dim('queries',min=2,max=8);m=torch.export.Dim('references',min=2,max=8)
    graph=torch.export.export(model,args,dynamic_shapes=({0:n},{0:m},{0:m},{0:n}))
    buffer=io.BytesIO();torch.export.save(graph,buffer);buffer.seek(0)
    restored=torch.export.load(buffer).module()
    other=(torch.rand(6,4,device=device)+.3,torch.rand(2,4,device=device)+.3,torch.randn(2,2,device=device),torch.randn(6,2,device=device))
    torch.testing.assert_close(restored(*other),model(*other))
    empty=args[0].new_empty(0)
    v=[args[0].requires_grad_(),args[1].requires_grad_()]+[empty]*8
    torch.library.opcheck(torch.ops.torchpairwise._pairwise_form.default,(v,args[3].requires_grad_(),args[2].requires_grad_(),'kl','sum',False,2,1,False,False,False,False,2,6,3,[None]*8,[None]*3,[None]*2,None))
