"""Pixel/window oracles for unreduced maps and streamed reference reductions."""
import pytest
import torch
import torchpairwise as tp


def dense_map(x, y, comparison):
    a, b = x.unsqueeze(-4), y.unsqueeze(-5)
    if comparison != 'ssim':
        delta = b-a
        return delta if comparison == 'residual' else delta.abs() if comparison == 'absolute_error' else delta.square()
    coords = torch.arange(3, dtype=x.dtype, device=x.device)-1
    g = (-.5*(coords/.9).square()).exp(); g = g/g.sum()
    kernel = g[:,None]*g[None,:]
    a, b = (t.unfold(-2,3,1).unfold(-2,3,1) for t in (a,b))
    ma, mb = ((t*kernel).sum((-1,-2)) for t in (a,b))
    da, db = a-ma[...,None,None], b-mb[...,None,None]
    va, vb = ((t.square()*kernel).sum((-1,-2)) for t in (da,db))
    covariance = (da*db*kernel).sum((-1,-2))
    c1, c2 = (.02*1.7)**2, (.04*1.7)**2
    return (2*ma*mb+c1)/(ma.square()+mb.square()+c1)*(2*covariance+c2)/(va+vb+c2)


def options(comparison):
    return dict(data_range=1.7,window_size=3,sigma=.9,k1=.02,k2=.04) if comparison == 'ssim' else {}


@pytest.mark.parametrize('comparison',['residual','absolute_error','squared_error','ssim'])
def test_image_maps_modes_reductions_and_gradients(device, comparison):
    torch.manual_seed(51)
    x = torch.rand(2,1,2,1,6,5,device=device,dtype=torch.float64).transpose(-1,-2).requires_grad_()
    y = torch.rand(1,2,3,1,6,5,device=device,dtype=torch.float64).transpose(-1,-2).requires_grad_()
    kw = options(comparison)
    for reduction in ('none','sum','mean','min','max'):
        full = dense_map(x,y,comparison)
        expected = full if reduction == 'none' else getattr(full,reduction)(dim=-4)
        if reduction in ('min','max'): expected = expected.values
        actual = (tp.image_maps(x,y,comparison=comparison,chunk_size=2,**kw) if reduction == 'none' else
                  tp.image_map_operator(x, y, chunk_size=2, **kw, comparison=comparison).reduce(reduction, dim=-4))
        torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-9)
        g = torch.randn_like(actual)
        gradients = torch.autograd.grad(actual,(x,y),g)
        references = torch.autograd.grad(expected,(x,y),g)
        for a,b in zip(gradients,references): torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-8)
    # One event statistic per source family, then common scalar consumers.
    name={'residual':'std','absolute_error':'max','squared_error':'var','ssim':'mean'}[comparison]
    opts={'correction':0} if name in ('var','std') else {}
    g=tp.image_map_operator(x,y,comparison=comparison,chunk_size=2,**kw)
    flat=dense_map(x,y,comparison).flatten(-3)
    pooled=getattr(flat,name)(dim=-1,**opts)
    if name=='max':pooled=pooled.values
    h=getattr(g,name)([-3,-2,-1],**opts)
    value=h.topk(2,largest=comparison=='ssim')
    ref=pooled.topk(2,largest=comparison=='ssim')
    torch.testing.assert_close(value,ref,atol=1e-10,rtol=1e-9)
    torch.testing.assert_close(torch.autograd.grad(value.values.sum(),(x,y)),
                               torch.autograd.grad(ref.values.sum(),(x,y)),atol=1e-9,rtol=1e-8)
    if comparison == 'ssim':
        # Replay boundaries cross both spatial tails; another case crosses the
        # channel boundary. Identity is a whole-image predicate, not per patch.
        for shape in ((1,73,263), (33,5,7)):
            a=torch.rand(1,*shape,device=device,dtype=torch.float64,requires_grad=True)
            b=torch.cat((a.detach(),torch.rand_like(a)),0).requires_grad_()
            c,h,w=shape
            settings=[c,h,w,3];parameters=[1.7,.9,.02,.04]
            packed=[torch.ops.torchpairwise._media_features(t,1,settings,parameters) for t in (a,b)]
            native=torch.ops.torchpairwise._image_map_values(*packed,3,settings,parameters,7,False)
            actual=native.mean((-2,-1)).max(-1).values.sum()+native.var((-4,-2,-1),correction=0).sum()
            maps=dense_map(a,b,'ssim')
            expected=maps.mean((-2,-1)).max(-1).values.sum()+maps.var((-4,-2,-1),correction=0).sum()
            torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-9)
            torch.testing.assert_close(torch.autograd.grad(actual,(a,b)),
                torch.autograd.grad(expected,(a,b)),rtol=2e-7,atol=2e-9)
    paired = tp.image_maps(x,y[...,:2,:,:,:],comparison=comparison,mode='paired',chunk_size=2,**kw)
    full = dense_map(x,y[...,:2,:,:,:],comparison)
    expected = torch.stack([full[...,i,i,:,:,:] for i in range(2)],dim=-4)
    torch.testing.assert_close(paired,expected,atol=1e-10,rtol=1e-9)
    if comparison != 'residual':
        condensed = tp.image_maps(x,comparison=comparison,mode='condensed',chunk_size=2,**kw)
        torch.testing.assert_close(condensed,dense_map(x,x,comparison)[...,0,1,:,:,:].unsqueeze(-4),atol=1e-10,rtol=1e-9)
        assert torch.autograd.grad(condensed.sum(),x)[0].isfinite().all()
    else:
        with pytest.raises(RuntimeError,match='directed'):
            tp.image_maps(x,comparison=comparison,mode='condensed')


def test_ssim_maps_pool_to_existing_score_and_gradcheck(device):
    torch.manual_seed(12)
    x = torch.rand(2,1,4,5,device=device,dtype=torch.float64,requires_grad=True)
    y = torch.rand(3,1,4,5,device=device,dtype=torch.float64,requires_grad=True)
    kw = options('ssim')
    maps = tp.image_maps(x,y,comparison='ssim',**kw)
    pooled = maps.mean((-3,-2,-1))
    score = tp.ssim_scores(x,y,**kw)
    torch.testing.assert_close(pooled,score)
    torch.testing.assert_close(torch.autograd.grad(pooled.sum(),(x,y)),torch.autograd.grad(score.sum(),(x,y)))
    assert torch.autograd.gradcheck(lambda a, b: tp.image_map_operator(a, b, **kw, comparison='ssim').reduce('mean', dim=-4), (x, y), fast_mode=True)
    identity = tp.image_maps(x,x.clone(),comparison='ssim',mode='paired',**kw)
    torch.testing.assert_close(identity,torch.ones_like(identity))
    torch.testing.assert_close(torch.autograd.grad(identity.sum(),x)[0],torch.zeros_like(x))


def test_image_map_empty_ties_keepdim_amp_and_saved_state(device):
    x = torch.zeros(2,1,3,4,device=device,requires_grad=True)
    y = torch.ones(5,1,3,4,device=device,requires_grad=True)
    for reduction in ('min','max'):
        value = tp.image_map_operator(x, y, chunk_size=2).reduce(reduction, keepdim=True, dim=-4)
        assert value.shape == (2,1,1,3,4)
        gx,gy = torch.autograd.grad(value.sum(),(x,y))
        torch.testing.assert_close(gx,torch.full_like(x,-2.))
        expected = torch.zeros_like(y);expected[0]=4.
        torch.testing.assert_close(gy,expected)
    empty = y[:0]
    for mode,args,shape in [('all_pairs',(x,empty),(2,0,1,3,4)),('paired',(x[:0],empty),(0,1,3,4)),('condensed',(x[:1],),(0,1,3,4))]:
        value = tp.image_maps(*args,mode=mode)
        assert value.shape == shape
        torch.testing.assert_close(torch.autograd.grad(value.sum(),x)[0],torch.zeros_like(x))
    zero = tp.image_map_operator(x, empty).reduce('sum', dim=-4)
    mean = tp.image_map_operator(x, empty).reduce('mean', dim=-4)
    torch.testing.assert_close(zero,torch.zeros_like(x));assert mean.isnan().all()
    for reduction in ('min','max'):
        with pytest.raises(RuntimeError,match='empty'):tp.image_map_operator(x, empty).reduce(reduction, dim=-4)
    dtype=torch.bfloat16 if device=='cpu' else torch.float16
    with torch.autocast(device,dtype=dtype):
        assert torch.ops.torchpairwise.image_maps(x.to(dtype),y.to(dtype),reduction='mean').dtype==torch.float32
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append(tuple(t.shape)) or t),lambda t:t):
        value=tp.image_map_operator(x, y, data_range=1.0, window_size=3, chunk_size=2, comparison='ssim').reduce('min', dim=-4)
    assert not any(len(shape)==5 and shape[:2]==(2,5) for shape in saved)
    with pytest.raises(RuntimeError,match='Higher.*derivatives'):
        torch.autograd.grad(value.sum(),x,create_graph=True)


def test_image_map_invalid_options(device):
    x=torch.rand(2,1,4,5,device=device)
    for kw in [dict(comparison='unknown'),dict(comparison='ssim'),dict(data_range=1.),dict(chunk_size=0),dict(comparison='ssim',data_range=1.,window_size=4)]:
        with pytest.raises((RuntimeError,ValueError,TypeError)):tp.image_maps(x,**kw)
        with pytest.raises((RuntimeError,ValueError,TypeError)):tp.image_map_operator(x,**kw).mean().sum()
    with pytest.raises(ValueError):tp.image_map_operator(x).reduce('missing', dim=-4)


def test_float32_reference_ties_match_materialized_maps(device):
    x=torch.ones(1,1,2,2,device=device,requires_grad=True)
    y=torch.tensor([0.,2.**-25],device=device).reshape(2,1,1,1).expand(2,1,2,2).clone().requires_grad_()
    for comparison,reduction in [('residual','max'),('absolute_error','min')]:
        full=tp.image_maps(x,y,comparison=comparison)
        torch.testing.assert_close(full[:,0],full[:,1],rtol=0,atol=0)
        expected=getattr(full,reduction)(dim=-4).values
        actual=tp.image_map_operator(x, y, chunk_size=1, comparison=comparison).reduce(reduction, dim=-4)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(),(x,y)),torch.autograd.grad(expected.sum(),(x,y)))
        pooled=tp.image_maps(x,y,comparison=comparison).flatten(-3).max(-1).values
        h=tp.image_map_operator(x,y,comparison=comparison,chunk_size=1).max()
        torch.testing.assert_close(h.sum(),pooled.sum())
        torch.testing.assert_close(torch.autograd.grad(h.sum(),(x,y)),torch.autograd.grad(pooled.sum(),(x,y)))

    # A promoted regional map must still round entries before computing moments.
    # Skipping the float32 boundary makes this variance four times smaller.
    a = torch.zeros(1, 2, device=device, dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[1.+2.**-25, 1.+3.*2.**-25]], device=device,
                     dtype=a.dtype, requires_grad=True)
    result, _ = torch.ops.torchpairwise._image_map_contract(
        a, b, 0, [1,1,2,1], [1.,1.,.01,.03], 6, 2, 7, False, 0., False)
    expected = (b.unsqueeze(0)-a.unsqueeze(1)).float().double().var(-1,correction=0).float()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(result.sum(), (a,b)),
                               torch.autograd.grad(expected.sum(), (a,b)), rtol=0, atol=0)



def test_lazy_image_map_full_event_reductions_and_boundaries(device):
    torch.manual_seed(140)
    x = (torch.rand(2,1,3,2,3,4,device=device,dtype=torch.float64)*.2).transpose(-1,-2).detach().requires_grad_()
    y = (.8+torch.rand(1,2,4,2,3,4,device=device,dtype=x.dtype)*.2).transpose(-1,-2).detach().requires_grad_()
    v = torch.randn(3,2,device=device,dtype=x.dtype,requires_grad=True)
    g = tp.image_map_operator(x,y,comparison='residual',chunk_size=2)
    raw = tp.image_maps(x,y,comparison='residual').flatten(-3)
    assert g.shape == (2,2,3,4,2,4,3) and g.event_shape == (2,4,3)
    actual,expected=[],[]
    for name in ('sum','mean','var','std','min','max','prod','all','any'):
        opts={'correction':.5} if name in ('var','std') else {}
        h=getattr(g,name)(dim=[4,5,6],**opts)
        ref=getattr(raw,name)(dim=-1,**opts)
        if name in ('min','max'):ref=ref.values
        assert isinstance(h,tp.LazyPairwise) and h.shape==ref.shape and h.dtype==ref.dtype
        if name in ('all','any'):
            torch.testing.assert_close(h.diagonal(),ref.diagonal(dim1=-2,dim2=-1))
            assert not h.sum().requires_grad
        else:
            actual.append(h.T.matmul(v));expected.append(ref.transpose(-1,-2)@v)
    cot=[torch.randn_like(t) for t in actual]
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,(x,y,v),cot),
                               torch.autograd.grad(expected,(x,y,v),cot),rtol=1e-7,atol=1e-9)
    scalar=torch.tensor(.3,device=device,dtype=x.dtype,requires_grad=True)
    truth=(g.all()*scalar).sum().sum()
    gx,gs=torch.autograd.grad(truth,(x,scalar))
    assert gx.eq(0).all() and gs.item()==48
    maps = tp.image_maps(x,y,comparison='residual')
    channels = g.mean(dim=(-2,-1))
    assert channels.event_shape == (2,) and channels.shape == (2,2,3,4,2)
    pooled = channels.max(-1)
    ref_channels = maps.mean((-2,-1))
    torch.testing.assert_close(pooled.sum(), ref_channels.max(-1).values.sum((-2,-1)))
    torch.testing.assert_close(g.mean(-1).max(-1).mean().sum(),
        maps.mean(-1).max(-1).values.mean(-1).sum((-2,-1)))
    retained = g.mean((-2,-1), keepdim=True)
    assert retained.event_shape == (2,1,1)
    torch.testing.assert_close(retained.mean().diagonal(),
        ref_channels.mean(-1).diagonal(dim1=-2,dim2=-1))
    # Joint sample/event statistics are one reduction, not variance of means.
    joint = g.var(dim=(-4,-2,-1), correction=.5, keepdim=True)
    reference = maps.var(dim=(-4,-2,-1), correction=.5, keepdim=True)
    torch.testing.assert_close(joint, reference)
    torch.testing.assert_close(torch.autograd.grad(joint.sum(),(x,y)),
        torch.autograd.grad(reference.sum(),(x,y),retain_graph=True),rtol=1e-7,atol=1e-9)
    # Local event and joint sample/event moments preserve translated precision.
    a=torch.zeros(1,1,1,5,device=device,dtype=x.dtype,requires_grad=True)
    b=(1e16+torch.arange(15,device=device,dtype=x.dtype).reshape(3,1,1,5)*2).requires_grad_()
    source=tp.image_map_operator(a,b,comparison='residual',chunk_size=2)
    centered=(b-1e16)[None]-a[:,None]
    for axes in ((-3,-2,-1),(-4,-3,-2,-1)):
        actual=source.var(axes,correction=0)
        ref=centered.var(axes,correction=0)
        if isinstance(actual,tp.LazyPairwise):actual=actual.sum();ref=ref.sum()
        torch.testing.assert_close(actual,ref)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(),(a,b)),
                                   torch.autograd.grad(ref.sum(),(a,b),retain_graph=True))
    torch.testing.assert_close(channels.mean(-2),ref_channels.mean(-2))
    torch.testing.assert_close(g.mean(None),maps.mean((-5,-4,-3,-2,-1)))
    torch.testing.assert_close(g.mean([]),g.mean(None))
    for name in ('sum','mean','var','std','min','max','prod','all','any'):
        opts={'correction':0} if name in ('var','std') else {}
        actual=getattr(channels,name)(-2,**opts)
        ref=getattr(ref_channels,name)(-2,**opts)
        if name in ('min','max'):ref=ref.values
        torch.testing.assert_close(actual,ref)
        if name not in ('all','any'):
            torch.testing.assert_close(torch.autograd.grad(actual.sum(),(x,y)),
                torch.autograd.grad(ref.sum(),(x,y),retain_graph=True),rtol=1e-7,atol=1e-9)
    truth = g.all((-2,-1))
    torch.testing.assert_close(truth.sum(-2), maps.ne(0).all(-1).all(-1).sum(-2))
    assert truth.sum(-2).dtype == torch.int64
    with pytest.raises(TypeError,match='floating'):
        truth.mean()
    with pytest.raises(ValueError,match='batch'):
        g.mean(0)
    with pytest.raises(ValueError,match='only used'):
        g.reduce('sum',correction=0)
    # Empty collections remain connected without invoking an empty convolution.
    for comparison in ('squared_error','ssim'):
        kw=options(comparison)
        h=tp.image_map_operator(x[...,:0,:,:,:],y,comparison=comparison,**kw).mean()
        value=h.sum().sum()
        assert value.item()==0
        assert torch.autograd.grad(value,x)[0].eq(0).all()

    # Shared SSIM features prepare once per operand in each consumer and first
    # backward. Only packed collections, not full pair maps, are retained.
    kw=options('ssim')
    h=tp.image_map_operator(x,y,comparison='ssim',chunk_size=2,**kw).var(correction=0)
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append(tuple(t.shape)),t)[1],lambda t:t):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            value=h.sum()
            grads=torch.autograd.grad(value.sum(),(x,y))
    names=[event.name for event in profile.events()]
    assert names.count('torchpairwise::_media_features')==2
    assert names.count('torchpairwise::_media_features_backward')==2
    assert (2,2,3,4,2,2,1) not in saved and (2,2,3,4) not in saved
    low=torch.bfloat16 if device=='cpu' else torch.float16
    a=x.detach().to(low).requires_grad_();b=y.detach().to(low).requires_grad_()
    with torch.autocast(device,dtype=low):
        result=tp.image_map_operator(a,b,comparison='ssim',chunk_size=2,**kw).var(correction=0).sum().sum()
    ref=tp.image_map_operator(a.float(),b.float(),comparison='ssim',chunk_size=2,**kw).var(correction=0).sum().sum()
    assert result.dtype==torch.float32
    torch.testing.assert_close(result,ref,rtol=3e-4,atol=3e-5)
    assert all(t.isfinite().all() for t in torch.autograd.grad(result,(a,b)))

    # Force the same regional contraction used for large maps with modest
    # inputs. Spatial tails and overlapping SSIM halos cross region borders.
    settings, parameters = [2, 73, 263, 3], [1., 1.5, .01, .03]
    for comparison, code, mask in ((0,0,5), (3,1,6), (2,2,7), (1,3,3),
                                   (0,4,7), (3,5,3), (0,7,6), (0,8,7)):
        a = torch.rand(1, *settings[:3], device=device, dtype=torch.float64)
        b = torch.rand(2, *settings[:3], device=device, dtype=torch.float64)
        if code == 4:
            a.zero_(); b.fill_(2.)
            # The first logical witness is visited in a later rectangle;
            # these distinct double values tie after the public float cast.
            b[0,0,0,260] = -1.+1e-8
            b[0,0,1,0] = -1.
        elif code >= 7:
            a.zero_(); b.fill_(int(code == 7))
            b[..., -1, -1] = int(code == 8)
        kind = int(comparison == 3)
        a = torch.ops.torchpairwise._media_features(a,kind,settings,parameters).detach().requires_grad_()
        b = torch.ops.torchpairwise._media_features(b,kind,settings,parameters).detach().requires_grad_()
        dtype = 7 if code == 5 else 6
        args = (a,b,comparison,settings,parameters,dtype,code,mask,True,.5,False)
        result,state = torch.ops.torchpairwise._image_map_contract(*args)
        dense = torch.ops.torchpairwise._image_map_values(a,b,comparison,settings,parameters,dtype,False)
        axes = tuple(2+d for d in range(3) if mask & (1 << d))
        name = ('sum','mean','var','std','min','max','prod','all','any')[code]
        retained = [d for d in range(dense.ndim) if d not in axes]
        shape = [dense.shape[d] for d in retained]
        flat = dense.double().permute(*retained,*axes).reshape(*shape,-1)
        kw = {'correction':.5} if code in (2,3) else {}
        expected = getattr(flat,name)(-1,**kw)
        if code in (4,5): expected = expected.values
        out_shape = [1 if d in axes else size for d,size in enumerate(dense.shape)]
        expected = expected.reshape(out_shape).to(result.dtype)
        torch.testing.assert_close(result,expected)
        if code >= 7:
            assert result.dtype == torch.bool and not result.requires_grad
        else:
            cot = torch.randn_like(result)
            original_cot = cot.clone()
            torch.testing.assert_close(torch.autograd.grad(result,(a,b),cot),
                torch.autograd.grad(expected,(a,b),cot),rtol=3e-5,atol=3e-8)
            torch.testing.assert_close(cot, original_cot)
        assert not state.requires_grad
        if code in (1,5,7):
            torch.library.opcheck(torch.ops.torchpairwise._image_map_contract.default,args)

    # The product core shares ATen's whole-tile zero branch, rather than the
    # exceptional-factor convention of a terminal sample-axis product.
    def check_product(a, b, settings, mask, *, comparison=0, paired=False, cotangent=1.):
        parameters = [1., 1.5, .01, .03]
        kind = int(comparison == 3)
        a, b = [torch.ops.torchpairwise._media_features(t, kind, settings, parameters)
                .detach().requires_grad_() for t in (a, b)]
        args = (a, b, comparison, settings, parameters, 7, 6, mask, True, 1., paired)
        result, state = torch.ops.torchpairwise._image_map_contract(*args)
        dense = torch.ops.torchpairwise._image_map_values(
            a, b, comparison, settings, parameters, 7, paired)
        rank = 1 if paired else 2
        axes = [rank+d for d in range(3) if mask & (1 << d)]
        retained = [d for d in range(dense.ndim) if d not in axes]
        shape = [dense.shape[d] for d in retained]
        expected = dense.permute(*retained, *axes).reshape(*shape, -1).prod(-1)
        expected = expected.reshape(result.shape)
        cot = torch.full_like(result, cotangent)
        torch.testing.assert_close(result, expected, equal_nan=True)
        torch.testing.assert_close(torch.autograd.grad(result, (a, b), cot),
            torch.autograd.grad(expected, (a, b), cot), equal_nan=True)
        assert state.shape[-1] == 7 and not state.requires_grad
        return args

    a = torch.rand(1, 35, 3, 259, device=device, dtype=torch.float64)*.001
    b = 1+torch.rand(2, 35, 3, 259, device=device, dtype=a.dtype)*.001
    for mask in range(1, 8):
        args = check_product(a, b, [35, 3, 259, 1], mask)
    torch.library.opcheck(torch.ops.torchpairwise._image_map_contract.default, args)
    check_product(a.expand(2, -1, -1, -1), b, [35, 3, 259, 1], 5, paired=True)
    # A zero in one pair switches every pair to ATen's exclusive-product VJP.
    for values in ((float('inf'), 2.), (float('nan'), 2.),
                   (0., float('inf')), (0., 0., float('inf')),
                   (0., 0., float('nan')), (-0., -2.), (float('inf'),), (float('nan'),)):
        for cot in (0., 1.):
            a = torch.zeros(1, 1, 1, len(values), device=device, dtype=torch.float64)
            b = torch.tensor(values, device=device, dtype=a.dtype).reshape_as(a)
            check_product(a, b, [1, 1, len(values), 1], 7, cotangent=cot)
    a = torch.zeros(1, 1, 1, 2, device=device, dtype=torch.float64)
    b = torch.tensor([[float('inf'), 2.], [0., 2.]],device=device,dtype=a.dtype).reshape(2,1,1,2)
    check_product(a, b, [1, 1, 2, 1], 7)
    for values in ([1e308,1e308,1e-308], [1e-308,1e-308,1e308]):
        a = torch.zeros(1,1,1,3,device=device,dtype=torch.float64)
        b = torch.tensor([values,[0.,1.,1.]],device=device,dtype=a.dtype).reshape(2,1,1,3)
        check_product(a,b,[1,1,3,1],7)
    a = torch.rand(1,2,7,263,device=device,dtype=torch.float64)
    b = a.expand(2,-1,-1,-1)+torch.rand(2,2,7,263,device=device,dtype=a.dtype)*.0001
    check_product(a, b, [2,7,263,3], 6, comparison=3)

    # CUDA amin/amax may select a different tied zero sign than indexed min.
    a = torch.zeros(1,1,1,2,device=device,dtype=torch.float64)
    b = a.clone(); b[...,0] = -0.
    settings, parameters = [1,1,2,1], [1.,1.,.01,.03]
    a,b = [torch.ops.torchpairwise._media_features(t,0,settings,parameters) for t in (a,b)]
    result,_ = torch.ops.torchpairwise._image_map_contract(a,b,0,settings,parameters,7,4,7,False,1.,False)
    assert result.signbit().all()

    # Repeated winning pixels accumulate after rounding each incoming map
    # cotangent to float32, without losing cancellation in the packed VJP.
    a = torch.zeros(1, 2, device=device, dtype=torch.float64, requires_grad=True)
    b = torch.ones(3, 2, device=device, dtype=a.dtype, requires_grad=True)
    cot = torch.tensor([2**25, 1., -2**25], device=device, dtype=torch.float32)
    result, _ = torch.ops.torchpairwise._image_map_contract(
        a, b, 0, settings, parameters, 6, 4, 7, False, 1., False)
    ga, gb = torch.autograd.grad(result, (a, b), cot.reshape_as(result))
    torch.testing.assert_close(ga, torch.tensor([[-1., 0.]], device=device, dtype=a.dtype))
    torch.testing.assert_close(gb[:, 0], cot.double())
    assert gb[:, 1].eq(0).all()
    for code, divisor in ((0, 1.), (1, 2.)):
        result, _ = torch.ops.torchpairwise._image_map_contract(
            a, b, 0, settings, parameters, 6, code, 7, False, 1., False)
        ga, gb = torch.autograd.grad(result, (a, b), cot.reshape_as(result))
        incoming = (cot.double() / divisor).float().double()
        torch.testing.assert_close(ga, -incoming.sum().expand_as(ga))
        torch.testing.assert_close(gb, incoming[:, None].expand_as(gb))

    # Cross the actual automatic threshold: direct joint M/H/W variance must
    # consume regions without constructing a complete pair map or cotangent.
    a=torch.rand(2,4,515,517,device=device,requires_grad=True)
    b=torch.rand_like(a,requires_grad=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        result=tp.image_map_operator(a,b,comparison='squared_error',chunk_size=2).var((-4,-2,-1),correction=0)
        gradients=torch.autograd.grad(result.sum(),(a,b))
    assert not any(e.name=='torchpairwise::_image_map_values' for e in profile.events())
    raw=tp.image_maps(a,b,comparison='squared_error',chunk_size=2)
    expected=raw.double().var((-4,-2,-1),correction=0).float()
    torch.testing.assert_close(result,expected)
    torch.testing.assert_close(gradients,torch.autograd.grad(expected.sum(),(a,b)),rtol=3e-5,atol=3e-8)

    # Exercise public automatic product contraction above the large-map
    # threshold, including backward; no full event map is materialized.
    a=(.002*a.detach()).requires_grad_()
    b=(1+.002*b.detach()).requires_grad_()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        result=tp.image_map_operator(a,b,comparison='residual',chunk_size=2).prod((-2,-1)).mean().sum()
        gradients=torch.autograd.grad(result.sum(),(a,b))
    names=[e.name for e in profile.events()]
    assert 'torchpairwise::_image_map_contract' in names
    assert 'torchpairwise::_image_map_values' not in names
    raw=tp.image_maps(a,b,comparison='residual',chunk_size=2)
    expected=raw.double().flatten(-2).prod(-1).float().double().mean(-1).float().sum()
    torch.testing.assert_close(result,expected)
    torch.testing.assert_close(gradients,torch.autograd.grad(expected,(a,b)),rtol=3e-5,atol=3e-8)


def test_image_event_topk_values_axes_programs_and_boundaries(device):
    torch.manual_seed(161)
    x = torch.rand(2, 1, 2, 3, 7, 5, device=device, dtype=torch.float64).transpose(-1, -2).requires_grad_()
    y = torch.rand(1, 2, 3, 3, 5, 7, device=device, dtype=x.dtype, requires_grad=True)
    for comparison, dim, largest in (('residual', -3, True), ('absolute_error', -2, False),
                                     ('squared_error', -1, True), ('ssim', 6, False)):
        raw = dense_map(x, y, comparison)
        ids = raw.argsort(dim=dim, descending=largest, stable=True).narrow(dim, 0, 2)
        expected = raw.gather(dim, ids)
        result = tp.image_map_operator(x, y, comparison=comparison, chunk_size=1,
                                       **options(comparison)).topk(2, dim=dim, largest=largest, sorted=False)
        assert isinstance(result, torch.return_types.topk)
        torch.testing.assert_close(result.values, expected, rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(result.indices, ids)
        cot = torch.randn_like(expected)
        torch.testing.assert_close(torch.autograd.grad(result.values, (x, y), cot),
                                   torch.autograd.grad(expected, (x, y), cot), rtol=1e-8, atol=1e-9)
    # The selected axis belongs to the current shape after ordered reductions.
    raw = dense_map(x, y, 'absolute_error')
    source = tp.image_map_operator(x, y, comparison='absolute_error', chunk_size=1)
    program = source.mean((-2, -1)).var(-1, correction=0, keepdim=True)
    expected = raw.mean((-2, -1)).var(-1, correction=0, keepdim=True)
    result = program.topk(1)
    torch.testing.assert_close(result.values, expected)
    assert result.indices.eq(0).all()
    torch.testing.assert_close(torch.autograd.grad(result.values.sum(), (x, y)),
                               torch.autograd.grad(expected.sum(), (x, y)))
    counts = source.any(-1).sum(-1)
    result = counts.topk(2)
    expected = raw.ne(0).any(-1).sum(-1).sort(dim=-1, descending=True, stable=True)
    torch.testing.assert_close(result.values, expected.values[..., :2])
    torch.testing.assert_close(result.indices, expected.indices[..., :2])
    assert result.values.dtype == torch.int64 and not result.values.requires_grad
    with pytest.raises(TypeError, match='numeric|boolean'):
        source.any(-1).topk(1)
    # Region boundaries: W=256, C=32, and valid SSIM Hout=64.
    a = torch.zeros(1, 1, 1, 270, device=device, requires_grad=True)
    b = torch.zeros_like(a, requires_grad=True)
    with torch.no_grad():
        b[..., 260] = float('nan')
        b[..., 268] = float('nan')
        b[..., 5] = float('inf')
    # Public image inputs must be finite. Exercise exceptional map entries
    # directly through the packed native core without relaxing that contract.
    def exceptional_topk(k, largest=True):
        values, indices = torch.ops.torchpairwise._image_event_topk(
            a.flatten(1).double(), b.flatten(1).double(), 2, [1, 1, 270, 11],
            [1., 1.5, .01, .03], 6, [], [], k, 2, largest, 1)
        return torch.return_types.topk((values, indices))
    for largest in (True, False):
        raw = dense_map(a, b, 'squared_error')
        ids = raw.argsort(dim=-1, descending=largest, stable=True)[..., :3]
        expected = raw.gather(-1, ids)
        result = exceptional_topk(3, largest)
        torch.testing.assert_close(result.values, expected, equal_nan=True)
        torch.testing.assert_close(result.indices, ids)
        torch.testing.assert_close(torch.autograd.grad(result.values.sum(), (a, b)),
                                   torch.autograd.grad(expected.sum(), (a, b)), equal_nan=True)
    empty = exceptional_topk(0)
    raw = dense_map(a, b, 'squared_error')[..., :0]
    assert empty.values.shape[-1] == 0
    torch.testing.assert_close(torch.autograd.grad(empty.values.sum(), (a, b)),
                               torch.autograd.grad(raw.sum(), (a, b)), equal_nan=True)
    a = torch.zeros(1, 33, 2, 3, device=device, requires_grad=True)
    b = torch.arange(33, device=device, dtype=a.dtype).reshape(1, 33, 1, 1).expand_as(a).clone().requires_grad_()
    with torch.no_grad(): b[:, 32].copy_(b[:, 31])
    result = tp.image_map_operator(a, b, comparison='residual').topk(2, dim=-3)
    assert result.indices[:, :, 0].eq(31).all() and result.indices[:, :, 1].eq(32).all()
    a = torch.rand(1, 1, 67, 5, device=device, dtype=torch.float64, requires_grad=True)
    b = torch.cat((a.detach(), torch.rand_like(a)), 0).requires_grad_()
    raw = dense_map(a, b, 'ssim')
    ids = raw.argsort(dim=-2, descending=True, stable=True)[..., :2, :]
    expected = raw.gather(-2, ids)
    result = tp.image_map_operator(a, b, comparison='ssim', **options('ssim')).topk(2, dim=-2)
    torch.testing.assert_close(result.values, expected, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(result.indices, ids)
    torch.testing.assert_close(torch.autograd.grad(result.values.sum(), (a, b)),
                               torch.autograd.grad(expected.sum(), (a, b)), rtol=1e-7, atol=1e-8)
    assert tp.image_map_operator(x[..., :0, :, :, :], y).topk(1).values.shape[:4] == (2, 2, 0, 3)
    for k in (-1, 8, True, 1.5):
        with pytest.raises((ValueError, RuntimeError), match='k|axis'):
            source.topk(k)
    for dim in (-4, [5, 6], None):
        with pytest.raises((TypeError, ValueError), match='dim|axis'):
            source.topk(1, dim=dim)
    with pytest.raises(TypeError, match='boolean'):
        source.topk(1, largest=1)


@pytest.mark.compiler
def test_image_event_topk_native_and_compiler(device):
    torch.manual_seed(162)
    a = torch.rand(2, 2, 5, 7, device=device, dtype=torch.float64, requires_grad=True)
    b = torch.rand(3, 2, 5, 7, device=device, dtype=a.dtype, requires_grad=True)
    settings, parameters = [2, 5, 7, 11], [1., 1.5, .01, .03]
    packed = [torch.ops.torchpairwise._media_features(t, 0, settings, parameters) for t in (a, b)]
    torch.library.opcheck(torch.ops.torchpairwise._image_event_topk.default,
                          (*packed, 2, settings, parameters, 7, [], [], 2, 2, False, 1))

    def call(x, y):
        return tp.image_map_operator(x, y, comparison='absolute_error', chunk_size=1).mean(-1).topk(2, dim=-1)

    compiled = torch.compile(call, fullgraph=True, dynamic=True, backend='aot_eager')
    for x, y in ((a, b), (a[:1], b[:2])):
        ref, out = call(x, y), compiled(x, y)
        torch.testing.assert_close(out, ref)
        torch.testing.assert_close(torch.autograd.grad(out.values.sum(), (x, y)),
                                   torch.autograd.grad(ref.values.sum(), (x, y)))
