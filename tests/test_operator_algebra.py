"""Scalar-expression workloads against dense values, derivatives and deployment."""
import io
import pytest
import torch
import torchpairwise as tp


def test_scalar_algebra_mixed_sources_weights_and_directed_transpose(device):
    torch.manual_seed(121)
    x = (torch.rand(2, 1, 3, 2, device=device, dtype=torch.float64) + .3).requires_grad_()
    y = (torch.rand(1, 2, 4, 2, device=device, dtype=x.dtype) + .3).requires_grad_()
    w = torch.rand(2, device=device, dtype=x.dtype, requires_grad=True)
    alpha = torch.tensor(.4, device=device, dtype=x.dtype, requires_grad=True)
    matrix = torch.rand(3, 4, device=device, dtype=x.dtype, requires_grad=True)
    rows = torch.rand(1, 2, 3, device=device, dtype=x.dtype, requires_grad=True)
    cols = torch.rand(2, 1, 4, device=device, dtype=x.dtype, requires_grad=True)
    left = torch.rand(4, 2, device=device, dtype=x.dtype, requires_grad=True)
    right = torch.rand(2, 1, 3, 2, device=device, dtype=x.dtype, requires_grad=True)
    a = tp.lazy(x, y, metric='kl', chunk_size=2)
    b = tp.lazy(x, y, metric='sqeuclidean', w=w, chunk_size=1)
    expr = ((-alpha * a - b).exp() + tp.as_lazy(matrix) * 0.2).weight_rows(rows).weight_columns(cols).T
    dense = ((-alpha * tp.kl_divergences(x, y) - tp.sqeuclidean_distances(x, y, w=w)).exp()
             + matrix * .2) * rows.unsqueeze(-1) * cols.unsqueeze(-2)
    dense = dense.transpose(-1, -2)
    assert isinstance(expr, tp.LazyPairwise) and expr.entrywise
    assert expr.shape == dense.shape and expr.dtype == dense.dtype and expr.device == dense.device
    rids = torch.tensor([3, 0, 3], device=device); cids = torch.tensor([2, 0], device=device)
    actual = (expr.matmul(right), expr.bilinear(left, right), expr.sum(), expr.mean(-2),
              expr.diagonal(), expr.block(rids, cids), expr.T.T.sum(-1))
    expected = (dense @ right, left.T @ dense @ right, dense.sum((-1, -2)), dense.mean(-2),
                dense.diagonal(dim1=-2, dim2=-1), dense.index_select(-2, rids).index_select(-1, cids), dense.sum(-1))
    torch.testing.assert_close(actual, expected)
    raw = (x, y, w, alpha, matrix, rows, cols, left, right)
    cot = [torch.randn_like(v) for v in actual]
    torch.testing.assert_close(torch.autograd.grad(actual, raw, cot), torch.autograd.grad(expected, raw, cot), rtol=1e-7, atol=1e-8)
    # An already-transposed expression can be combined with a different source.
    torch.testing.assert_close((expr + a.T).sum(-1), (dense + tp.kl_divergences(x,y).transpose(-1,-2)).sum(-1))


def test_scalar_algebra_repeated_nodes_consumers_and_boundaries(device):
    torch.manual_seed(122)
    matrix = (torch.rand(2, 3, 3, device=device, dtype=torch.float64) + .3).requires_grad_()
    a = tp.as_lazy(matrix, chunk_size=1)
    shared = a.square()
    expr, dense = shared * shared + shared - 1, matrix.square().square() + matrix.square() - 1
    actual, expected = [], []
    for dim in (None, -1, -2):
        for name in ('sum', 'mean', 'min', 'max', 'prod', 'all', 'any'):
            source = dense.flatten(-2) if dim is None else dense
            ref = getattr(source, name)(-1 if dim is None else dim)
            if name in ('min','max'): ref = ref.values
            value = getattr(expr, {'min': 'amin', 'max': 'amax'}.get(name, name))([-2,-1] if dim is None else [dim])
            torch.testing.assert_close(value, ref)
            if value.is_floating_point(): actual.append(value); expected.append(ref)
    c = torch.rand(3, 2, device=device, dtype=matrix.dtype, requires_grad=True)
    actual.append(expr.quadratic(c)); expected.append(c.T @ dense @ c)
    mask = ~torch.eye(3,device=device,dtype=torch.bool)
    actual.append(expr.mean(exclude_self=True)); expected.append(dense[...,mask].mean(-1))
    cot=[torch.randn_like(v) for v in actual]
    torch.testing.assert_close(torch.autograd.grad(actual,(matrix,c),cot),torch.autograd.grad(expected,(matrix,c),cot))
    with torch.no_grad(): matrix.add_(.1)
    torch.testing.assert_close(expr.sum(),(matrix.square().square()+matrix.square()-1).sum((-2,-1)))
    ids=torch.tensor([2,0,2],device=device)
    assert expr.block(ids[:0],ids).shape == (2,0,3)
    with pytest.raises(RuntimeError,match='bounds'):expr.block(torch.tensor([-1],device=device),ids)
    for n,m in ((0,3),(2,0),(0,0)):
        z=torch.empty(2,n,m,device=device,dtype=matrix.dtype,requires_grad=True)
        e=tp.as_lazy(z)*2+1
        assert e.sum().eq(0).all() and e.prod().eq(1).all() and e.mean().isnan().all()
        assert e.all().all() and not e.any().any()
        assert torch.autograd.grad(e.sum().sum(),z)[0].numel()==0
    nan=torch.tensor([[float('nan'),1.],[float('nan'),2.]],device=device,requires_grad=True)
    e=tp.as_lazy(nan)+0
    for name in ('min','max'):
        value=getattr(e.T,{'min': 'amin', 'max': 'amax'}.get(name, name))();ref=getattr(nan.T.flatten(),name)(-1).values
        torch.testing.assert_close(value,ref,equal_nan=True)
        torch.testing.assert_close(torch.autograd.grad(value,nan)[0],torch.autograd.grad(ref,nan)[0])

    # Explicit widths above128 must reach the native source. D tiling is a
    # separate control and leaves its final short block intact.
    x=torch.rand(257,5,device=device,dtype=torch.float64)
    y=torch.rand(263,5,device=device,dtype=x.dtype)
    source=tp.residual_operator(x,y,transform='square',chunk_size=256,feature_chunk_size=2)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as profile:
        value=source.mean().sum(-1)
    events=profile.events()
    packets=[e.input_shapes for e in events if e.name=='torchpairwise::_residual_features']
    assert max(p[0][0] for p in packets)==256 and max(p[1][0] for p in packets)==256
    widths=[e.input_shapes[0][-1] for e in events if e.name=='aten::sub' and e.input_shapes and len(e.input_shapes[0])==3]
    assert max(widths)==2 and 1 in widths
    expected=(x[:,None,:]-y[None,:,:]).square().mean(-1).sum(-1)
    torch.testing.assert_close(value,expected)


@pytest.mark.parametrize('dtype',(torch.int64,torch.bool,torch.float32))
def test_scalar_algebra_promotion_and_factor_contract(device,dtype):
    data=[[2**60+1,2**60],[-2**60,1]] if dtype==torch.int64 else [[1,0],[0,1]]
    matrix=torch.tensor(data,device=device,dtype=dtype)
    a=tp.as_lazy(matrix, chunk_size=1)
    e=a*2+1
    assert e.dtype==(matrix*2+1).dtype
    ids=torch.tensor([0,1],device=device)
    torch.testing.assert_close(e.block(ids,ids),matrix*2+1)
    torch.testing.assert_close(e.sum(),(matrix*2+1).sum())
    for scalar in (.5,-1,True,torch.tensor(.3,device=device,dtype=torch.float64)):
        expression=a+scalar
        assert expression.dtype==(matrix+scalar).dtype
        torch.testing.assert_close(expression.block(ids,ids),matrix+scalar)

    r=torch.rand(2,2,device=device,dtype=torch.float64,requires_grad=True)
    torch.testing.assert_close(e.matmul(r) if e.dtype==torch.int64 else (a*2.).matmul(r.float()),
                               (matrix*2+1).double()@r if e.dtype==torch.int64 else (matrix*2.).float()@r.float())
    for invalid in (torch.ones(2,device=device),object()):
        with pytest.raises((TypeError,ValueError)):a*invalid
    with pytest.raises(RuntimeError,match='count'):a.weight_rows(torch.ones(3,device=device)).sum()
    with pytest.raises(RuntimeError,match='N/M'):(a + tp.as_lazy(torch.ones(3, 2, device=device))).sum()
    categorical=torch.rand(3,2,device=device,requires_grad=True)
    h=tp.lazy(categorical, metric='hamming')+1
    assert not h.sum().requires_grad
    image=torch.rand(2,1,8,8,device=device)
    with pytest.raises((RuntimeError,ValueError)):
        (tp.lazy(image, metric='multiscale_ssim', data_range=1.0, window_size=3, weights=()) + 1).sum()



def test_scalar_algebra_numbers_keep_python_precision(device):
    # Python floats were lowered in the default dtype, so float64 sources saw
    # 0.3 rounded to float32 (0.30000001192092896).
    ids=torch.tensor([0,1],device=device)
    for dtype in (torch.float64,torch.float32,torch.int64):
        matrix=torch.tensor([[0,1],[2,3]],device=device,dtype=dtype)
        a=tp.as_lazy(matrix, chunk_size=1)
        weights=torch.tensor([.7,.1],device=device,dtype=torch.float64)
        pairs=[(a+.3,matrix+.3),((a+.1)*.3,(matrix+.1)*.3),(.3-a*a,.3-matrix*matrix),
               (a.weight_rows(weights)+.3,matrix*weights[:,None]+.3)]
        for expression,dense in pairs:
            assert expression.dtype==dense.dtype
            torch.testing.assert_close(expression.block(ids,ids),dense,atol=0,rtol=0)
        one=tp.as_lazy(matrix[:1, :1], chunk_size=1)+.3
        assert one.sum().item()==(matrix[:1,:1]+.3).sum().item()


@pytest.mark.parametrize('family',('vector','structured','media','ordered'))
def test_scalar_algebra_catalog_sources(device,family):
    torch.manual_seed(123)
    if family=='vector':
        x=torch.rand(2,3,device=device,dtype=torch.float64)+.3
        y=torch.rand(3,3,device=device,dtype=torch.float64)+.3
        cases=[('kl_divergences',x,y,{}),('rbf_kernel',x,y,{'gamma':.4}),
               ('hamming_distances',x>.6,y>.6,{})]
    else:
        if family=='structured':from test_structured_operator import METRICS,inputs,options
        elif family=='media':from test_media_operator import METRICS,inputs,options
        else:from test_ordered_operator import METRICS,inputs,options
        representatives = {'structured': ('stein', 'gaussian_wasserstein', 'gaussian_mixture_overlap'),
                           'media': ('ssim', 'multiresolution_stft'),
                           'ordered': ('soft_dtw', 'levenshtein', 'mmd_statistic')}
        cases=[(metric,inputs(metric,2,device),inputs(metric,3,device),options(metric))
               for metric in METRICS if metric in representatives[family]]
        assert len(cases) == len(representatives[family])
    for metric,x,y,kw in cases:
        a=tp.lazy(x, y, metric=metric, chunk_size=1, **kw)
        raw=tp.compare(x,y,metric=metric,**kw)
        if isinstance(raw,tuple):raw=raw[0]
        dense=(raw*.25+1).square().T
        e=(a*.25+1).square().T
        r=torch.rand(2,2,device=device,dtype=dense.dtype,requires_grad=True)
        actual=(e.matmul(r),e.sum(-2),e.diagonal())
        expected=(dense@r,dense.sum(-2),dense.diagonal())
        torch.testing.assert_close(actual,expected,rtol=3e-6,atol=3e-7)
        if family!='vector':
            leaves=(*x,*y) if isinstance(x,tuple) else (x,y)
            leaves=tuple(t for t in leaves if t.requires_grad)+(r,)
            assert [v.requires_grad for v in actual] == [v.requires_grad for v in expected]
            active=[v for v in actual if v.requires_grad]
            refs=[v for v in expected if v.requires_grad]
            cot=[torch.randn_like(v) for v in active]
            torch.testing.assert_close(torch.autograd.grad(active,leaves,cot),torch.autograd.grad(refs,leaves,cot),rtol=3e-6,atol=3e-7)


@pytest.mark.compiler
def test_scalar_algebra_compile_export_and_retained_state(device):
    class Model(torch.nn.Module):
        def forward(self,x,y,factor,rows,values):
            a=tp.lazy(x, y, metric='sqeuclidean', chunk_size=2)
            k=(-factor*a).exp().weight_rows(rows)
            e=k*k+k*.2
            return (e.matmul(values),e.T.mean(-1),e.diagonal(),e.sum(),
                    e.var([-1],correction=0),e.std((-2,-1)),e.logsumexp([-2]),
                    a.weighted_mean(k),e.weighted_var(k,dim=[-1],correction=1),
                    e.T.weighted_std(rows,dim=[-1]),(-factor*a).softmax_matmul(values),
                    tp.residual_operator(x,y,feature_chunk_size=2,chunk_size=2).var(-1,correction=0).matmul(values),
                    tp.residual_operator(x,y,feature_chunk_size=2,chunk_size=2).var([-3,-2],correction=0),
                    e.topk(2,dim=-2).values,e.topk(2).indices,
                    e.prod(-1), e.prod(),
                    k.frobenius_inner_product(e,center=True,normalize=True),
                    tp.lazy(x, metric='rbf').hsic(tp.lazy(x, metric='linear')),
                    tp.residual_operator(x,y,feature_chunk_size=2).all().diagonal(),
                    e.count_nonzero([-2,-1]))
    torch.manual_seed(124);torch._dynamo.reset()
    def inputs(n,m,d=3):
        return (torch.rand(n,d,device=device,dtype=torch.float64,requires_grad=True),
                torch.rand(m,d,device=device,dtype=torch.float64,requires_grad=True),
                torch.tensor(.4,device=device,dtype=torch.float64,requires_grad=True),
                torch.rand(n,device=device,dtype=torch.float64,requires_grad=True),
                torch.rand(m,2,device=device,dtype=torch.float64,requires_grad=True))
    model=Model();args=inputs(3,4)
    compiled=torch.compile(model,fullgraph=True,dynamic=True)
    actual,expected=compiled(*args),model(*args)
    torch.testing.assert_close(actual,expected)
    floating=[v for v in actual if v.is_floating_point()]
    refs=[v for v in expected if v.is_floating_point()]
    cot=[torch.randn_like(v) for v in floating]
    torch.testing.assert_close(torch.autograd.grad(floating,args,cot),torch.autograd.grad(refs,args,cot))
    n,m=torch.export.Dim('n',min=2,max=16),torch.export.Dim('m',min=2,max=16)
    graph=torch.export.export(model,args,dynamic_shapes=({0:n},{0:m},{},{0:n},{0:m}))
    stream=io.BytesIO();torch.export.save(graph,stream);stream.seek(0)
    restored=torch.export.load(stream).module();other=inputs(13,11)
    torch.testing.assert_close(restored(*other),model(*other))
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append(tuple(t.shape)),t)[1],lambda t:t):
        result=model(*other)
    assert not {(13,11),(11,13),(13,11,3),(11,13,3)}.intersection(saved)
    # Isolate the new feature source: older CPU vector preparation can pin D.
    # Keep the existing mixed-source N/M export and exercise dynamic D here.
    class ResidualModel(torch.nn.Module):
        def forward(self,x,y,values):
            g=tp.residual_operator(x,y,feature_chunk_size=2,chunk_size=2)
            return (g.var([-1],correction=0).matmul(values),
                    g.var([-2,-1],correction=0),g.mean([-3]),
                    g.std([-3,-2],correction=0,keepdim=True),
                    g.prod([-3,-1]),g.min().topk(2).values,g.any([-2,-1]))
    d=torch.export.Dim('features',min=2,max=8)
    residual_args=(args[0],args[1],args[-1])
    residual_graph=torch.export.export(ResidualModel(),residual_args,
        dynamic_shapes=({0:n,1:d},{0:m,1:d},{0:m}))
    stream=io.BytesIO();torch.export.save(residual_graph,stream);stream.seek(0)
    residual_export=torch.export.load(stream).module()
    changed=inputs(13,11,5);residual_args=(changed[0],changed[1],changed[-1])
    torch.testing.assert_close(residual_export(*residual_args),ResidualModel()(*residual_args))
    for device_type,low in ((device,torch.bfloat16 if device=='cpu' else torch.float16),):
        low_args=tuple(t.detach().to(low).requires_grad_() for t in args)
        with torch.autocast(device_type,dtype=low):value=model(*low_args)
        ref=model(*(t.float() for t in low_args))
        torch.testing.assert_close(value,ref,rtol=3e-4,atol=3e-5)
        assert all(t.dtype==torch.float32 for t in value if t.is_floating_point())
        assert value[-1].dtype==torch.int64 and value[-2].dtype==torch.bool
        gradients=torch.autograd.grad(sum(t.sum() for t in value if t.is_floating_point()),low_args)
        assert all(g.isfinite().all() for g in gradients)
    # Raw registered core: schema/fake/autograd/AOT agreement on a matrix leaf
    # and a differentiable scalar factor, without depending on Python replay.
    matrix=torch.rand(4,3,device=device,dtype=torch.float64).T.requires_grad_()
    empty=matrix.new_empty(0)
    v=[empty.clone() for _ in range(20)]
    v[0]=matrix.new_empty(3,0);v[1]=matrix.new_empty(4,0)
    v[7]=torch.arange(3,device=device);v[8]=torch.arange(4,device=device)
    v[10]=matrix;v[11]=torch.arange(4,device=device).unsqueeze(-1)
    v.append(torch.tensor(.4,device=device,dtype=matrix.dtype,requires_grad=True))
    torch.library.opcheck(torch.ops.torchpairwise._expression_consume.default,
        (v,['_matrix'],['_matrix'],[[2,2,7,0,0,0,3]],[[]],[[]],[[]],
         [[0,0,-1],[1,0,-1],[5,0,1]],[],'sum',7,2,False,False,False))

    torch.library.opcheck(torch.ops.torchpairwise._expression_consume.topk,
        (v,['_matrix'],['_matrix'],[[2,2,7,0,0,0,3]],[[]],[[]],[[]],
         [[0,0,-1],[1,0,-1],[5,0,1]],[],'neighbors',7,2,False,False,False,2,True))

    torch.library.opcheck(torch.ops.torchpairwise._expression_product.default,
        (v,['_matrix'],['_matrix'],[[2,2,7,0,0,0,3]],[[]],[[]],[[]],
         [[0,0,-1],[1,0,-1],[5,0,1]],[],'prod',7,2,False,False,True,0.))

    torch.library.opcheck(torch.ops.torchpairwise._expression_statistics.default,
        (v,['_matrix'],['_matrix'],[[2,2,7,0,0,0,3]],[[]],[[]],[[]],
         [[0,0,-1],[1,0,-1],[5,0,1]],[],'var',7,2,False,False,True,1.))

    torch.library.opcheck(torch.ops.torchpairwise._expression_statistics.weighted,
        (v,['_matrix'],['_matrix'],[[2,2,7,0,0,0,3]],[[]],[[]],[[]],
         [[0,0,-1],[1,0,-1],[5,0,1]],[],'var',7,2,False,False,True,0.,0,1))

    torch.library.opcheck(torch.ops.torchpairwise._expression_inner_product.default,
        (v,['_matrix'],['_matrix'],[[2,2,7,0,0,0,3]],[[]],[[]],[[]],
         [[0,0,-1],[1,0,-1],[5,0,1]],[],'inner_product',7,2,False,False,False,
         0,2,True,True))

    v[9]=torch.rand(4,2,device=device,dtype=matrix.dtype,requires_grad=True)
    torch.library.opcheck(torch.ops.torchpairwise._expression_softmax_matmul.default,
        (v,['_matrix'],['_matrix'],[[2,2,7,0,0,0,3]],[[]],[[]],[[]],
         [[0,0,-1],[1,0,-1],[5,0,1]],[],'softmax_matmul',7,2,False,False,False))

    rx=torch.rand(3,5,device=device,dtype=matrix.dtype,requires_grad=True)
    ry=torch.rand(4,5,device=device,dtype=matrix.dtype,requires_grad=True)
    torch.library.opcheck(torch.ops.torchpairwise._residual_features.default,
        (rx,ry,'var','abs',.5,2,False))

    torch.library.opcheck(torch.ops.torchpairwise._residual_reduce.default,
        (rx,ry,'var','square',[1,2],False,.5,2,2))
    torch.library.opcheck(torch.ops.torchpairwise._residual_features.default,
        (rx,ry,'min','identity',1.,2,False))
    torch.library.opcheck(torch.ops.torchpairwise._residual_reduce.default,
        (rx,ry,'prod','identity',[0,2],False,1.,2,2))



def test_scalar_topk_axes_ties_and_gradients(device):
    torch.manual_seed(131)
    x = torch.rand(3, 5, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(4, 5, device=device, dtype=x.dtype, requires_grad=True)
    a = tp.residual_operator(x, y, feature_chunk_size=2, chunk_size=2).min()
    dense = (x[:, None] - y[None]).min(-1).values
    for dim in (-1, 0):
        value = (a.square() + .1).topk(2, dim=dim, largest=False)
        ref = (dense.square() + .1).topk(2, dim=dim, largest=False)
        torch.testing.assert_close(value, ref)
        cot = torch.randn_like(value.values)
        torch.testing.assert_close(torch.autograd.grad(value.values, (x, y), cot),
                                   torch.autograd.grad(ref.values, (x, y), cot, retain_graph=True))
    # Stable ties and nonfinite ordering must not depend on tile boundaries.
    matrix = torch.tensor([[1., float('nan'), 1., float('inf')],
                           [3., 2., 2., float('-inf')]], device=device, requires_grad=True)
    op = tp.as_lazy(matrix, chunk_size=1)
    small = op.topk(3, largest=False)
    torch.testing.assert_close(small.indices, torch.tensor([[0, 2, 3], [3, 1, 2]], device=device))
    assert op.topk(1).indices[0, 0] == 1
    assert op.topk(0).values.shape == (2, 0)
    assert torch.autograd.grad(op.topk(0).values.sum(), matrix)[0].eq(0).all()
    integer = torch.tensor([[2**63-1, -2**63, 0, -2**63]], device=device)
    torch.testing.assert_close(tp.as_lazy(integer, chunk_size=1).topk(3, largest=False).indices, torch.tensor([[1, 3, 2]], device=device))
    square = torch.ones(4, 4, device=device, requires_grad=True)
    result = tp.as_lazy(square, chunk_size=1).topk(2, exclude_self=True, sorted=False)
    assert result.indices.ne(torch.arange(4, device=device)[:, None]).all()
    torch.testing.assert_close(result.indices, torch.tensor([[1, 2], [0, 2], [0, 1], [0, 1]], device=device))
    assert tp.as_lazy(matrix[:, :0]).topk(0).values.shape == (2, 0)
    with pytest.raises(TypeError, match='integer dim'):
        op.topk(1, dim=[-1])
    with pytest.raises(RuntimeError, match='eligible'):
        op.topk(5)
