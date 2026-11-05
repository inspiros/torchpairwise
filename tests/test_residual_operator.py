"""Feature contraction into the existing scalar streaming engine."""
import warnings
import pytest
import torch
import torchpairwise as tp


def test_residual_feature_axes_dense_oracle_and_boundaries(device):
    torch.manual_seed(130)
    x = torch.randn(2, 1, 3, 14, device=device, dtype=torch.float64)[..., ::2].requires_grad_()
    y = torch.randn(1, 2, 4, 7, device=device, dtype=x.dtype, requires_grad=True)
    values = torch.randn(3, 2, device=device, dtype=x.dtype, requires_grad=True)
    residual = x.unsqueeze(-2) - y.unsqueeze(-3)
    actual, expected = [], []
    # Distinct feature formulas, N/M consumers, transposed directed residuals,
    # batch broadcasting and noncontiguous features without a test grid.
    for transform, name, correction in (
        ('identity', 'sum', None), ('square', 'mean', None),
        ('identity', 'var', 0), ('abs', 'std', .5)):
        g = tp.residual_operator(x, y, transform=transform, chunk_size=2, feature_chunk_size=3)
        assert g.shape == residual.shape
        dense = residual if transform == 'identity' else getattr(residual, transform)()
        options = {} if correction is None else {'correction': correction}
        h = getattr(g, name)(dim=len(g.shape)-1, **options)
        dense = getattr(dense, name)(dim=-1, **options)
        assert isinstance(h, tp.LazyPairwise) and h.shape == dense.shape
        actual.append(h.T.matmul(values))
        expected.append(dense.transpose(-1, -2) @ values)
    actual.extend((h.var(-1, correction=0), h.diagonal()))
    expected.extend((dense.var(-1, correction=0), dense.diagonal(dim1=-2, dim2=-1)))
    # Joint axes use one statistic over the combined set, not nested variances.
    for name, axes, transform, keepdim, correction in (
        ('mean', -3, 'square', False, None),
        ('var', [-2], 'identity', True, .5),
        ('std', (-3,-2), 'abs', False, 0),
        ('var', [3,4], 'square', True, 1),
        ('sum', (-3,-1), 'abs', False, None),
        ('std', None, 'identity', True, 0)):
        source=tp.residual_operator(x,y,transform=transform,chunk_size=2,feature_chunk_size=3)
        dense=residual if transform=='identity' else getattr(residual,transform)()
        opts={} if correction is None else {'correction':correction}
        actual.append(getattr(source,name)(axes,keepdim=keepdim,**opts))
        expected.append(getattr(dense,name)(dim=(-3,-2,-1) if axes is None else axes,keepdim=keepdim,**opts))
    cot = [torch.randn_like(t) for t in actual]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, (x, y, values), cot),
                               torch.autograd.grad(expected, (x, y, values), cot), rtol=1e-7, atol=1e-8)
    # Selected entries use the paired core as well as the Cartesian tile core.
    g = tp.residual_operator(x, y, feature_chunk_size=2, chunk_size=2)
    raw = residual.mean(-1)
    torch.testing.assert_close(g.mean().diagonal(), raw.diagonal(dim1=-2, dim2=-1))
    ids = torch.tensor([2, 0, 2], device=device)
    torch.testing.assert_close(g.mean().block(ids, ids), raw.index_select(-2, ids).index_select(-1, ids))
    # Feature reduction then sample reduction is ordered, not interchangeable.
    constant = torch.full((2, 9), 1e6, device=device, requires_grad=True)
    zero = torch.zeros(3, 9, device=device, requires_grad=True)
    std = tp.residual_operator(constant, zero, feature_chunk_size=2).std(correction=0).sum()
    assert std.item() == 0.
    assert all(t.eq(0).all() for t in torch.autograd.grad(std, (constant, zero)))
    # Translation must not change moments or their VJPs across D/sample tiles.
    shifted=(torch.arange(10,device=device,dtype=torch.float64).reshape(2,5)%7)*2
    a=(shifted+1e16).requires_grad_()
    b=torch.zeros(3,5,device=device,dtype=a.dtype,requires_grad=True)
    source=tp.residual_operator(a,b,chunk_size=2,feature_chunk_size=3)
    centered=(a-1e16)[:,None]-b[None]
    for name in ('var','std'):
        for axes in (-1,(-2,-1)):
            actual=getattr(source,name)(axes,correction=0)
            ref=getattr(centered,name)(axes,correction=0)
            if isinstance(actual,tp.LazyPairwise):actual=actual.sum();ref=ref.sum()
            torch.testing.assert_close(actual,ref)
            torch.testing.assert_close(torch.autograd.grad(actual.sum(),(a,b)),
                                       torch.autograd.grad(ref.sum(),(a,b),retain_graph=True))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for d, correction in ((0, 1), (1, 1), (2, 3)):
            a = torch.arange(d, device=device, dtype=torch.float64).reshape(1, d).requires_grad_()
            b = torch.zeros(1, d, device=device, dtype=a.dtype, requires_grad=True)
            for name in ('sum', 'mean', 'var', 'std'):
                opts = {'correction': correction} if name in ('var', 'std') else {}
                value = getattr(tp.residual_operator(a, b, feature_chunk_size=1), name)(**opts).sum()
                ref = getattr(a[:, None] - b[None], name)(dim=-1, **opts).sum()
                torch.testing.assert_close(value, ref, equal_nan=True)
                torch.testing.assert_close(torch.autograd.grad(value, (a, b)),
                                           torch.autograd.grad(ref, (a, b)), equal_nan=True)
    for dim in ((-1,4), [0], [-6]):
        with pytest.raises(ValueError):
            g.var(dim)
    with pytest.raises(ValueError,match='keepdim'):
        g.mean([-1],keepdim=True)
    assert isinstance(g.mean([-1]),tp.LazyPairwise)
    with pytest.raises(ValueError, match='only used'):
        g.reduce('sum', correction=0)
    for n, m in ((0, 3), (2, 0)):
        a = torch.empty(n, 5, device=device, requires_grad=True)
        b = torch.empty(m, 5, device=device, requires_grad=True)
        value = tp.residual_operator(a, b).var(correction=0).sum()
        assert value.item() == 0.
        assert all(t.eq(0).all() for t in torch.autograd.grad(value, (a, b)))

    a=torch.randn(2,5,device=device,dtype=torch.float64,requires_grad=True)
    b=torch.empty(0,5,device=device,dtype=a.dtype,requires_grad=True)
    for name in ('sum','mean','var','std'):
        opts={'correction':0} if name in ('var','std') else {}
        value=getattr(tp.residual_operator(a,b),name)([-2],**opts)
        ref=getattr(a[:,None]-b[None],name)(dim=-2,**opts)
        torch.testing.assert_close(value,ref,equal_nan=True)
        torch.testing.assert_close(torch.autograd.grad(value.sum(),(a,b)),
                                   torch.autograd.grad(ref.sum(),(a,b)),equal_nan=True)

    # Chunk widths are arbitrary positive int64 values, not preset buckets.
    a=torch.randn(2,5,device=device,dtype=torch.float64)
    b=torch.randn(3,5,device=device,dtype=a.dtype)
    expected=(a[:,None]-b[None]).square().mean(-1).sum(-1)
    for size in (1,3,129,300,2**63-1):
        value=tp.residual_operator(a,b,transform='square',chunk_size=size,feature_chunk_size=size).mean().sum(-1)
        torch.testing.assert_close(value,expected)
    for bad in (None,0,-1,2**63,1.0,True,False,float('inf'),-float('inf'),float('nan')):
        with pytest.raises(ValueError,match='feature_chunk_size'):
            tp.residual_operator(a,b,feature_chunk_size=bad)
    factories=(lambda size:tp.residual_operator(a,b,chunk_size=size),
               lambda size:tp.lazy(a, b, chunk_size=size),
               lambda size:tp.lazy(a, b, chunk_size=size, metric='rbf'),
               lambda size:tp.as_lazy(a, chunk_size=size),
               lambda size:tp.image_map_operator(a[:,None,None,:],b[:,None,None,:],chunk_size=size))
    for factory in factories:
        with pytest.raises(ValueError,match='chunk_size'):
            factory(2**63)
    for factory in factories[1:4]:
        assert factory(None).chunk_size is None
    # Typed native entrypoints also reject nonpositive sizes before iteration,
    # including empty collections that might otherwise skip a bad loop step.
    for bad in (0,-1):
        with pytest.raises(RuntimeError,match='positive'):
            torch.ops.torchpairwise.residual_reduction(a[:0],b,'sum','identity',[-2],False,None,bad,1)


def test_residual_selection_products_and_truth(device):
    # Ties cross feature/sample tiles; select the first logical witness.
    x = torch.tensor([[1., 1., 2.], [1., 2., 1.]], device=device, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[0., 0., 0.], [0., 1., 0.]], device=device, dtype=x.dtype, requires_grad=True)
    g = tp.residual_operator(x, y, chunk_size=1, feature_chunk_size=1)
    raw = x[:, None] - y[None]
    actual = [g.min().sum(), g.max().diagonal().sum(), g.min([-3, -1]).sum(),
              g.max(None), g.prod().sum(), g.prod([-2]).sum(), g.prod(None)]
    expected = [raw.min(-1).values.sum(), raw.max(-1).values.diagonal().sum(),
                raw.permute(1, 0, 2).flatten(1).min(-1).values.sum(),
                raw.flatten().max(0).values, raw.prod(-1).sum(),
                raw.prod(-2).sum(), raw.flatten().prod(0)]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, (x, y)),
                               torch.autograd.grad(expected, (x, y)))
    for name in ('all', 'any'):
        source = getattr(g, name)()
        assert source.dtype == torch.bool
        torch.testing.assert_close(source.diagonal(), getattr(raw.ne(0), name)(-1).diagonal())
        value = getattr(g, name)([-3, -1], keepdim=True)
        ref = getattr(getattr(raw.ne(0), name)(-1, keepdim=True), name)(-3, keepdim=True)
        torch.testing.assert_close(value, ref)
        assert value.dtype == torch.bool and not value.requires_grad
        assert not source.sum().requires_grad
    a = torch.empty(2, 0, device=device)
    b = torch.empty(3, 0, device=device)
    for name, identity in (('prod', 1), ('all', True), ('any', False)):
        value = getattr(tp.residual_operator(a, b), name)().diagonal()
        assert value.eq(identity).all()
    with pytest.raises(RuntimeError, match='nonempty'):
        tp.residual_operator(a, b).min().sum()
    # NaNs propagate, and their first witness receives min/max gradients.
    a = torch.tensor([[2., float('nan'), float('nan')]], device=device, requires_grad=True)
    b = torch.zeros(1, 3, device=device, requires_grad=True)
    value = tp.residual_operator(a, b, feature_chunk_size=1).min().sum()
    assert value.isnan()
    torch.testing.assert_close(torch.autograd.grad(value, a)[0], a.new_tensor([[0., 1., 0.]]))
    # Repeated witnesses accumulate before float32 rounding. A float32 atomic
    # sum could lose the unit term between these opposing large cotangents.
    a = torch.zeros(2, 4, device=device, requires_grad=True)
    b = torch.zeros(3, 4, device=device, requires_grad=True)
    weights = a.new_tensor([[1e8], [1.], [-1e8]])
    value = tp.residual_operator(a, b, chunk_size=4).min().matmul(weights).sum()
    ga, gb = torch.autograd.grad(value, (a, b))
    torch.testing.assert_close(ga, a.new_tensor([[1., 0., 0., 0.]]).expand_as(a))
    torch.testing.assert_close(gb[:, 0], b.new_tensor([-2e8, -2., 2e8]))
    assert gb[:, 1:].eq(0).all()
    a = torch.zeros(3, 4, device=device, requires_grad=True)
    b = torch.zeros(2, 4, device=device, requires_grad=True)
    value = tp.residual_operator(a, b, chunk_size=2, feature_chunk_size=3).min(-2)
    ga, gb = torch.autograd.grad(value, (a, b), weights.expand_as(value))
    torch.testing.assert_close(ga, weights.expand_as(a))
    torch.testing.assert_close(gb[0], b.new_full((4,), -1.))
    assert gb[1].eq(0).all()
    # Exclusive-product VJP: deleting the sole exceptional factor leaves a
    # finite product; this path does not divide an infinite/NaN output by input.
    for special in (float('inf'), float('nan')):
        a = torch.tensor([[special, 2., 3.]], device=device, requires_grad=True)
        b = torch.zeros_like(a)
        value = tp.residual_operator(a, b, feature_chunk_size=1).prod().sum()
        grad = torch.autograd.grad(value, a)[0]
        assert grad[0, 0] == 6.


def test_residual_feature_topk_values_witnesses_and_gradients(device):
    torch.manual_seed(151)
    x = torch.randn(2, 1, 3, 14, device=device, dtype=torch.float64)[..., ::2].requires_grad_()
    y = torch.randn(1, 2, 4, 7, device=device, dtype=x.dtype, requires_grad=True)
    raw = x.unsqueeze(-2) - y.unsqueeze(-3)
    for transform, largest, k in (('identity', True, 3), ('abs', False, 2), ('square', True, 7)):
        dense = raw if transform == 'identity' else getattr(raw, transform)()
        expected_ids = dense.argsort(dim=-1, descending=largest, stable=True)[..., :k]
        expected = dense.gather(-1, expected_ids)
        result = tp.residual_operator(x, y, transform=transform, chunk_size=2,
                                     feature_chunk_size=2).topk(k, dim=4, largest=largest, sorted=False)
        assert isinstance(result, torch.return_types.topk)
        torch.testing.assert_close(result.values, expected)
        torch.testing.assert_close(result.indices, expected_ids)
        assert not result.indices.requires_grad
        cot = torch.randn_like(expected)
        torch.testing.assert_close(torch.autograd.grad(result.values, (x, y), cot),
                                   torch.autograd.grad(expected, (x, y), cot, retain_graph=True))
    # Ties/NaNs cross chunks; lower feature IDs win ties, for either direction.
    a = torch.tensor([[2., 2., float('nan'), 1., float('nan'), float('inf')]],
                     device=device, requires_grad=True)
    b = torch.zeros_like(a, requires_grad=True)
    for largest in (True, False):
        for transform in ('identity', 'abs', 'square'):
            dense = a[:, None] - b[None]
            if transform != 'identity':
                dense = getattr(dense, transform)()
            ids = dense.argsort(dim=-1, descending=largest, stable=True)[..., :3]
            expected = dense.gather(-1, ids)
            result = tp.residual_operator(a, b, transform=transform,
                                         feature_chunk_size=2).topk(3, largest=largest)
            torch.testing.assert_close(result.values, expected, equal_nan=True)
            torch.testing.assert_close(result.indices, ids)
            torch.testing.assert_close(torch.autograd.grad(result.values.sum(), (a, b)),
                                       torch.autograd.grad(expected.sum(), (a, b)), equal_nan=True)
    # Empty selections retain differentiable values, including nonlinear VJPs.
    for width, k in ((0, 0), (6, 0)):
        aa, bb = a[:, :width], b[:, :width]
        result = tp.residual_operator(aa, bb, transform='square', feature_chunk_size=2).topk(k)
        expected = (aa[:, None] - bb[None]).square()[..., :0]
        assert result.values.shape == (1, 1, 0)
        torch.testing.assert_close(torch.autograd.grad(result.values.sum(), (a, b)),
                                   torch.autograd.grad(expected.sum(), (a, b)), equal_nan=True)
    empty = tp.residual_operator(x[..., :0, :], y).topk(2)
    assert empty.values.shape == (2, 2, 0, 4, 2)
    self_values = tp.residual_operator(x, feature_chunk_size=2).topk(1).values
    self_dense = (x.unsqueeze(-2) - x.unsqueeze(-3)).sort(dim=-1, descending=True, stable=True).values[..., :1]
    torch.testing.assert_close(self_values, self_dense)
    torch.testing.assert_close(torch.autograd.grad(self_values.sum(), x)[0],
                               torch.autograd.grad(self_dense.sum(), x)[0])
    source = tp.residual_operator(x, y)
    for k in (-1, 8, True, 1.5):
        with pytest.raises((ValueError, RuntimeError), match='k|feature'):
            source.topk(k)
    for dim in (-2, [4], None):
        with pytest.raises((TypeError, ValueError), match='dim|feature'):
            source.topk(1, dim=dim)
    with pytest.raises(TypeError, match='boolean'):
        source.topk(1, sorted=1)
    with pytest.raises(RuntimeError, match='positive'):
        torch.ops.torchpairwise.residual_topk(x, 1, y, feature_chunk_size=0)


@pytest.mark.compiler
def test_residual_feature_topk_native_and_compiler(device):
    x = torch.randn(7, 3, device=device, dtype=torch.float64).t().requires_grad_()
    y = torch.randn(4, 7, device=device, dtype=x.dtype, requires_grad=True)
    torch.library.opcheck(torch.ops.torchpairwise._residual_topk.default,
                          (x, y, 2, False, 'abs', 2, 3))

    def call(a, b):
        return tp.residual_operator(a, b, transform='square', chunk_size=2,
                                    feature_chunk_size=3).topk(2, largest=False)

    compiled = torch.compile(call, backend='aot_eager', dynamic=True, fullgraph=True)
    for a, b in ((x, y), (x[:2], y[:3])):
        expected, actual = call(a, b), compiled(a, b)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual.values.sum(), (a, b)),
                                   torch.autograd.grad(expected.values.sum(), (a, b)))
