"""Structured scalar operators against canonical dense values and derivatives."""
import io
import pytest
import torch
import torchpairwise as tp


METRICS=('gaussian_wasserstein','log_euclidean','affine_invariant','stein',
    'gaussian_mixture_overlap','gaussian_mixture_l2','gaussian_mixture_cauchy_schwarz',
    'gaussian_mixture_wasserstein','gaussian_mixture_entropic_transport','gaussian_mixture_sinkhorn')


def inputs(metric,n,device,batch=()):
    d=2;k=2;mix=metric.startswith('gaussian_mixture')
    shape=(*batch,n,*((k,) if mix else ()),d)
    mean=torch.rand(shape,device=device,dtype=torch.float64)+.2
    factor=torch.rand((*shape,d),device=device,dtype=mean.dtype)
    cov=factor@factor.transpose(-1,-2)+torch.eye(d,device=device,dtype=mean.dtype)
    if metric in ('log_euclidean','affine_invariant','stein'):return cov.requires_grad_()
    if not mix:return mean.requires_grad_(),cov.requires_grad_()
    weights=(torch.rand((*batch,n,k),device=device,dtype=mean.dtype)+.4).requires_grad_()
    return mean.requires_grad_(),cov.requires_grad_(),weights


def leaves(collection):return collection if isinstance(collection,tuple) else (collection,)


def options(metric):
    if metric in ('gaussian_mixture_entropic_transport','gaussian_mixture_sinkhorn'):
        return dict(regularization=.8,max_iter=3)
    if metric in ('gaussian_mixture_overlap','gaussian_mixture_l2','gaussian_mixture_cauchy_schwarz'):
        return dict(component_chunk_size=1)
    return {}


@pytest.mark.parametrize('metric',METRICS)
def test_structured_operator_catalog_values_and_all_gradients(device,metric):
    torch.manual_seed(81)
    x=inputs(metric,2,device);y=inputs(metric,3,device)
    kw=options(metric)
    dense=tp.compare(x,y,metric=metric,**kw)
    a=tp.lazy(x, y, metric=metric, chunk_size=1, **kw).T
    l=torch.randn(3,2,device=device,dtype=dense.dtype,requires_grad=True)
    r=torch.randn(2,2,device=device,dtype=dense.dtype,requires_grad=True)
    actual=(a.matmul(r),a.bilinear(l,r),a.mean(-2),a.diagonal())
    matrix=dense.T
    expected=(matrix@r,l.T@matrix@r,matrix.mean(-2),matrix.diagonal())
    torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-9)
    tensors=(*leaves(x),*leaves(y),l,r)
    cotangents=tuple(torch.randn_like(t) for t in actual)
    torch.testing.assert_close(torch.autograd.grad(actual,tensors,cotangents),torch.autograd.grad(expected,tensors,cotangents),rtol=2e-7,atol=2e-8)
    rows=torch.tensor([2,0,2],device=device);columns=torch.tensor([1,0],device=device)
    got=a.block(rows,columns);ref=matrix.index_select(0,rows).index_select(1,columns)
    torch.testing.assert_close(got,ref,rtol=1e-8,atol=1e-9)
    if metric in ('gaussian_wasserstein','gaussian_mixture_sinkhorn'):
        fresh=tp.compare(x,y,metric=metric,**kw).T.index_select(0,rows).index_select(1,columns)
        raw=(*leaves(x),*leaves(y))
        torch.testing.assert_close(torch.autograd.grad(got.sum(),raw),torch.autograd.grad(fresh.sum(),raw),rtol=2e-7,atol=2e-8)
    assert a.shape==matrix.shape and a.dtype==matrix.dtype and a.device==matrix.device
    empty=rows[:0]
    assert a.block(empty,columns).shape==(0,2)


def test_structured_operator_broadcast_reductions_and_empty_contracts(device):
    torch.manual_seed(82)
    x=inputs('gaussian_wasserstein',2,device,(2,1))
    y=inputs('gaussian_wasserstein',3,device,(1,2))
    # Mean and covariance components can have independent broadcast batches.
    x=(x[0][0],x[1]);y=(y[0][:,0],y[1])
    dense=tp.compare(x,y,metric='gaussian_wasserstein',squared=True)
    a=tp.lazy(x, y, metric='gaussian_wasserstein', squared=True, chunk_size=1)
    for transpose in (False,True):
        matrix=dense.transpose(-1,-2) if transpose else dense
        operator=a.T if transpose else a
        assert operator.shape==matrix.shape
        for dim in (-1,-2,None):
            source=matrix.flatten(-2) if dim is None else matrix
            axis=-1 if dim is None else dim
            for reduction in ('sum','mean','min','max','prod','all','any'):
                expected=getattr(source,reduction)(axis)
                if reduction in ('min','max'):expected=expected.values
                torch.testing.assert_close(getattr(operator,{'min': 'amin', 'max': 'amax'}.get(reduction, reduction))(dim),expected)
    selfop=tp.lazy(x, metric='gaussian_wasserstein', chunk_size=1)
    full=tp.compare(x,metric='gaussian_wasserstein')
    torch.testing.assert_close(selfop.sum(exclude_self=True),full.sum((-2,-1))-full.diagonal(dim1=-2,dim2=-1).sum(-1))
    coeff=torch.randn(2,device=device,dtype=full.dtype)
    torch.testing.assert_close(selfop.quadratic(coeff),torch.einsum('i,...ij,j->...',coeff,full,coeff))
    for metric in ('gaussian_wasserstein','stein','gaussian_mixture_l2'):
        source=inputs(metric,2,device)
        empty=tuple(t[:0] for t in leaves(source));empty=empty[0] if not isinstance(source,tuple) else empty
        operator=tp.lazy(source, empty, metric=metric, chunk_size=1)
        assert operator.sum().item()==0 and operator.prod().item()==1
        assert operator.all().item() and not operator.any().item() and operator.mean().isnan()
        for reduction in ('min','max'):
            with pytest.raises((RuntimeError,IndexError)):getattr(operator,{'min': 'amin', 'max': 'amax'}.get(reduction, reduction))()
        assert operator.diagonal().shape==(0,)
        result=operator.sum();grads=torch.autograd.grad(result,leaves(source))
        assert all(torch.count_nonzero(g)==0 for g in grads)


def test_structured_operator_padded_components_self_terms_and_reuse(device):
    torch.manual_seed(83)
    metric='gaussian_mixture_sinkhorn';x=inputs(metric,3,device);y=inputs(metric,2,device)
    xl=torch.tensor([1,2,1],device=device);yl=torch.tensor([2,1],device=device)
    for collection,lengths in ((x,xl),(y,yl)):
        with torch.no_grad():
            valid=torch.arange(2,device=device)<lengths.unsqueeze(-1)
            collection[0][~valid]=float('nan');collection[1][~valid]=float('nan');collection[2][~valid]=float('nan')
    kw=dict(x1_lengths=xl,x2_lengths=yl,max_iter=3,regularization=.7)
    dense=tp.compare(x,y,metric=metric,**kw)
    a=tp.lazy(x, y, metric=metric, chunk_size=1, **kw).T
    torch.testing.assert_close(a.sum(-1),dense.T.sum(-1),rtol=1e-8,atol=1e-9)
    l=torch.randn(2,2,device=device,dtype=x[0].dtype);r=torch.randn(3,2,device=device,dtype=x[0].dtype)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        value=a.bilinear(l,r)
        grads=torch.autograd.grad(value.sum(),(*x,*y))
    names=[e.name for e in profile.events()]
    assert names.count('torchpairwise::_structured_features')==2
    assert names.count('torchpairwise::_structured_features_backward')==2
    ref=l.T@dense.T@r
    torch.testing.assert_close(grads,torch.autograd.grad(ref.sum(),(*x,*y)),rtol=2e-7,atol=2e-8)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        tp.lazy(x, metric=metric, x1_lengths=xl, max_iter=3, chunk_size=1).sum()
    assert sum(e.name=='torchpairwise::_structured_features' for e in profile.events())==1
    # Shared tensors with different valid prefixes are distinct collections.
    other_lengths=torch.tensor([2,1,2],device=device)
    clean=inputs(metric,3,device)
    kw=dict(x1_lengths=xl,x2_lengths=other_lengths,max_iter=3,regularization=.7)
    operator=tp.lazy(clean, metric=metric, chunk_size=1, **kw)
    torch.testing.assert_close(operator.sum(-1),tp.compare(clean,metric=metric,**kw).sum(-1))
    # All mixture components and coefficients broadcast their batch axes.
    bx=tuple(t.detach().unsqueeze(0).expand(2,*t.shape).requires_grad_() for t in clean)
    by=inputs(metric,2,device,(1,))
    coefficients=torch.randn(1,2,2,device=device,dtype=clean[0].dtype,requires_grad=True)
    kw=dict(max_iter=3,regularization=.7)
    actual=tp.lazy(bx, by, metric=metric, chunk_size=1, **kw).matmul(coefficients)
    expected=tp.compare(bx,by,metric=metric,**kw)@coefficients
    raw=(*bx,*by,coefficients)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),raw),torch.autograd.grad(expected.sum(),raw),rtol=2e-7,atol=2e-8)


def test_structured_operator_validation_and_precision(device):
    x=inputs('gaussian_wasserstein',2,device)
    with pytest.raises(TypeError):tp.lazy(x[0], metric='gaussian_wasserstein')
    with pytest.raises(TypeError):tp.lazy(x, metric='gaussian_wasserstein', regularization=1.0)
    bad=x[1].clone();bad[0,0,0]=-10.
    empty=tuple(t[:0] for t in x)
    with pytest.raises((RuntimeError,ValueError)):
        tp.lazy((x[0], bad), empty, metric='gaussian_wasserstein').sum()
    for dtype in (torch.float32,torch.float64):
        data=tuple(t.detach().to(dtype) for t in x)
        a=tp.lazy(data, metric='gaussian_wasserstein', chunk_size=1)
        torch.testing.assert_close(a.sum(-1),tp.compare(data,metric='gaussian_wasserstein').sum(-1))
    dtype=torch.bfloat16 if device=='cpu' else torch.float16
    low=tuple(t.detach().to(dtype) for t in x)
    with torch.autocast(device,dtype=dtype):
        a=tp.lazy(low, metric='gaussian_wasserstein', chunk_size=1)
        assert a.dtype==torch.float32 and a.sum().dtype==torch.float32


@pytest.mark.compiler
@pytest.mark.parametrize('metric',('gaussian_wasserstein','log_euclidean','gaussian_mixture_sinkhorn'))
def test_structured_operator_compile_export_and_opcheck(device,metric):
    class Model(torch.nn.Module):
        def forward(self,x,y,l,r):
            a=tp.lazy(x, y, metric=metric, chunk_size=1, **options(metric)).T
            return a.matmul(r),a.bilinear(l,r),a.mean(-2),a.diagonal()
    torch.manual_seed(84);torch._dynamo.reset();model=Model()
    x=inputs(metric,2,device);y=inputs(metric,3,device)
    args=(x,y,torch.randn(3,2,device=device,dtype=torch.float64),torch.randn(2,2,device=device,dtype=torch.float64))
    compiled=torch.compile(model,fullgraph=True,dynamic=True)
    expected=model(*args);actual=compiled(*args)
    torch.testing.assert_close(actual,expected)
    tensors=(*leaves(x),*leaves(y))
    loss=lambda values:sum(t.square().sum() for t in values)
    torch.testing.assert_close(torch.autograd.grad(loss(actual),tensors),torch.autograd.grad(loss(expected),tensors),rtol=2e-6,atol=2e-7)
    n=torch.export.Dim('queries',min=2,max=6);m=torch.export.Dim('references',min=2,max=6)
    sample=lambda value,dim:tuple({0:dim} for t in value) if isinstance(value,tuple) else {0:dim}
    graph=torch.export.export(model,args,dynamic_shapes=(sample(x,n),sample(y,m),{0:m},{0:n}))
    buffer=io.BytesIO();torch.export.save(graph,buffer);buffer.seek(0)
    restored=torch.export.load(buffer).module()
    other=(inputs(metric,4,device),inputs(metric,2,device),torch.randn(2,2,device=device,dtype=torch.float64),torch.randn(4,2,device=device,dtype=torch.float64))
    torch.testing.assert_close(restored(*other),model(*other))
    if metric=='gaussian_wasserstein':
        empty=x[0].new_empty(0)
        torch.library.opcheck(torch.ops.torchpairwise._structured_features.default,(x[0].unsqueeze(0),x[1].unsqueeze(0),empty,empty.long(),0,.1,3,2))
