"""Independent dense residual outer products, derivatives and deployment."""
import io
import pytest
import torch
import torchpairwise as tp
from torchpairwise.ops._registry import KERNELS


def dense_moment(x,y,kernel,normalize,center,**kwargs):
    k=tp.pairwise_kernels(x,y,kernel=kernel,**kwargs)
    delta=y.unsqueeze(-3)-x.unsqueeze(-2)
    mass=k.sum(-1)
    nonzero=mass!=0;safe=torch.where(nonzero,mass,1.)
    if center:
        mean=(k.unsqueeze(-1)*delta).sum(-2)/safe.unsqueeze(-1)
        delta=delta-mean.unsqueeze(-2)
    result=(k[...,None,None]*delta.unsqueeze(-1)*delta.unsqueeze(-2)).sum(-3)
    if normalize:result=result/safe[...,None,None]
    return torch.where(nonzero[...,None,None],result,0.) if normalize or center else result


@pytest.mark.parametrize('kernel,kwargs',[
    ('rbf',{'gamma':.4}),('linear',{}),('polynomial',{'degree':3,'gamma':.2}),
    ('matern',{'nu':2.5}),('chi2',{'gamma':.3}),('histogram_intersection',{})])
def test_kernel_moments_dense_values_and_gradients(device,kernel,kwargs):
    torch.manual_seed(4)
    x=(torch.rand(2,1,3,10,device=device,dtype=torch.float64)+.2)[...,::2].requires_grad_()
    y=(torch.rand(1,2,4,10,device=device,dtype=torch.float64)+.2)[...,::2].requires_grad_()
    for normalize,center in ((False,False),(True,False),(False,True),(True,True)):
        expected=dense_moment(x,y,kernel,normalize,center,**kwargs)
        actual=tp.lazy(x, y, metric=kernel, chunk_size=2, **kwargs).residual_outer_product(normalize=normalize, center=center)
        torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-10)
        # A nonsymmetric cotangent checks both coordinate-channel contributions.
        g=torch.randn_like(actual)
        grads=torch.autograd.grad(actual,(x,y),g)
        refs=torch.autograd.grad(expected,(x,y),g)
        torch.testing.assert_close(grads,refs,rtol=2e-8,atol=2e-9)


def test_kernel_moment_catalog_gradcheck_offset_and_empty(device):
    torch.manual_seed(19)
    x=(torch.rand(2,3,device=device,dtype=torch.float64)+.3).requires_grad_()
    y=(torch.rand(3,3,device=device,dtype=torch.float64)+.3).requires_grad_()
    for kernel in KERNELS:
        actual=tp.lazy(x, y, chunk_size=2, metric=kernel).residual_outer_product()
        torch.testing.assert_close(actual,dense_moment(x,y,kernel,False,False),rtol=1e-9,atol=1e-10)
    f=lambda a,b:tp.lazy(a, b, chunk_size=2, metric='rbf').residual_outer_product(normalize=True, center=True)
    assert torch.autograd.gradcheck(f,(x,y),fast_mode=True)
    torch.testing.assert_close(f(x+1e9,y+1e9),dense_moment(x+1e9,y+1e9,'rbf',True,True))
    torch.testing.assert_close(f(x,y+1e9),dense_moment(x,y+1e9,'rbf',True,True))
    # Constant weights isolate covariance when source/target centers are far apart.
    actual=tp.lazy(x, y + 1e9, 'rbf', gamma=0., chunk_size=2).residual_outer_product(normalize=True, center=True)
    torch.testing.assert_close(actual,dense_moment(x,y+1e9,'rbf',True,True,gamma=0.),atol=1e-12,rtol=1e-8)
    for xx,yy in ((x,y[:0]),(x[:0],y),(x.unsqueeze(0)[:0],y)):
        value=f(xx,yy)
        assert torch.count_nonzero(value)==0
        gx,gy=torch.autograd.grad(value.sum(),(x,y))
        torch.testing.assert_close(gx,torch.zeros_like(x));torch.testing.assert_close(gy,torch.zeros_like(y))


def test_kernel_moment_zero_mass_amp_and_retained_state(device):
    x=torch.tensor([[1.,0.]],device=device,dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[1.,1.],[-1.,2.]],device=device,dtype=torch.float64,requires_grad=True)
    raw=tp.lazy(x, y, chunk_size=1, metric='linear').residual_outer_product()
    torch.testing.assert_close(raw,dense_moment(x,y,'linear',False,False))
    assert torch.count_nonzero(raw)>0
    for normalize,center in ((True,False),(False,True),(True,True)):
        result=tp.lazy(x, y, chunk_size=1, metric='linear').residual_outer_product(normalize=normalize, center=center)
        torch.testing.assert_close(result,torch.zeros_like(result))
        torch.testing.assert_close(torch.autograd.grad(result.sum(),(x,y)),(torch.zeros_like(x),torch.zeros_like(y)))
    dtype=torch.bfloat16 if device=='cpu' else torch.float16
    with torch.autocast(device,dtype=dtype):
        assert torch.ops.torchpairwise.kernel_residual_outer_product(x.to(dtype),y.to(dtype)).dtype==torch.float32
    x=torch.randn(17,7,device=device,requires_grad=True);y=torch.randn(19,7,device=device,requires_grad=True)
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append(tuple(t.shape)) or t),lambda t:t):
        value=tp.lazy(x, y, chunk_size=3, metric='rbf').residual_outer_product(normalize=True, center=True)
    assert not any(17 in shape and 19 in shape for shape in saved)
    assert (19,7,7) not in saved
    with pytest.raises(RuntimeError,match='Higher derivatives'):torch.autograd.grad(value.sum(),x,create_graph=True)


@pytest.mark.compiler
def test_image_map_moment_compile_export_opcheck(device):
    class Interactions(torch.nn.Module):
        def forward(self,ix,iy,x,y):
            return (tp.image_map_operator(ix, iy, data_range=1.0, window_size=3, chunk_size=2, comparison='ssim').reduce('min', dim=-4),
                    tp.lazy(x, y, metric='rbf', gamma=0.3, chunk_size=2).residual_outer_product(normalize=True, center=True))
    torch._dynamo.reset()
    args=(torch.rand(3,1,5,6,device=device),torch.rand(4,1,5,6,device=device),torch.randn(3,5,device=device),torch.randn(4,5,device=device))
    model=Interactions();compiled=torch.compile(model,fullgraph=True,dynamic=True)
    torch.testing.assert_close(compiled(*args),model(*args))
    leaves=tuple(t.clone().requires_grad_() for t in args)
    loss=lambda result:sum(t.square().sum() for t in result)
    torch.testing.assert_close(torch.autograd.grad(loss(compiled(*leaves)),leaves),torch.autograd.grad(loss(model(*leaves)),leaves),rtol=1e-4,atol=1e-5)
    n=torch.export.Dim('queries',min=2,max=8);m=torch.export.Dim('references',min=2,max=8)
    exported=torch.export.export(model,args,dynamic_shapes=({0:n},{0:m},{0:n},{0:m}))
    buffer=io.BytesIO();torch.export.save(exported,buffer);buffer.seek(0)
    restored=torch.export.load(buffer).module()
    other=(torch.rand(6,1,5,6,device=device),torch.rand(2,1,5,6,device=device),torch.randn(6,5,device=device),torch.randn(2,5,device=device))
    torch.testing.assert_close(restored(*other),model(*other))
    torch.library.opcheck(torch.ops.torchpairwise._image_map.default,
        (args[0].double().requires_grad_(),args[1].double().requires_grad_(),3,5,2,3,[1.,1.5,.01,.03]))
    torch.library.opcheck(torch.ops.torchpairwise._kernel_moments.default,
        (args[2].requires_grad_(),args[3].requires_grad_(),args[2],'rbf',2,[.3,None,None,None,None],None,True))
