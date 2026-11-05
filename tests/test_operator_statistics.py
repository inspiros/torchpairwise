"""One dense oracle covers the distinct streaming-statistic contracts."""
import warnings
import io
import pytest
import torch
import torchpairwise as tp


def test_operator_statistics_values_gradients_and_boundaries(device):
    torch.manual_seed(127)
    matrix = torch.randn(2, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
    source = tp.as_lazy(matrix, chunk_size=2)
    # Cover each axis/global merge, transpose, exclusion and correction without
    # multiplying every source family by every statistic and option.
    actual, expected = [], []
    for name, dim, transpose, exclude, correction in (
        ('var', -1, False, False, 1), ('var', None, False, True, 0),
        ('std', -2, True, False, .5), ('std', None, True, False, -1),
        ('logsumexp', -2, False, False, 1), ('logsumexp', None, True, True, 1)):
        data, operator = (matrix.transpose(-1,-2), source.T) if transpose else (matrix, source)
        axis = -1 if dim is None else dim
        if exclude:
            data = data[..., ~torch.eye(4, device=device, dtype=torch.bool)]
        elif dim is None:
            data = data.flatten(-2)
        options = {'correction':correction} if name in ('var','std') else {}
        value = getattr(operator,name)(dim,keepdim=True,exclude_self=exclude,**options)
        ref = getattr(data,name)(dim=axis,**options)
        ref = ref.unsqueeze(-1).unsqueeze(-1) if dim is None else ref.unsqueeze(dim)
        torch.testing.assert_close(value,ref)
        actual.append(value);expected.append(ref)
    cot=[torch.randn_like(v) for v in actual]
    torch.testing.assert_close(torch.autograd.grad(actual,matrix,cot),
                               torch.autograd.grad(expected,matrix,cot))
    integral=torch.tensor([[0,2**60+1],[0,-2**60]],device=device)
    op=tp.as_lazy(integral, chunk_size=1)
    assert op.count_nonzero().dtype==torch.int64
    torch.testing.assert_close(op.count_nonzero(-2),integral.count_nonzero(-2))
    assert op.count_nonzero(exclude_self=True).item()==1
    assert not op.count_nonzero().requires_grad
    for bad in ('var','std','logsumexp'):
        with pytest.raises(RuntimeError,match='float32/float64'):getattr(op,bad)()
    # Large offset/small variance is evaluated against a float64 oracle, never
    # mean(square)-square(mean). Include a partial tile and a global merge.
    large=(torch.arange(35,device=device,dtype=torch.float32).reshape(5,7)%5)*.125+1e6
    stable=tp.as_lazy(large, chunk_size=3)
    for dim in (None,-1):
        ref=large.double().flatten() if dim is None else large.double()
        torch.testing.assert_close(stable.var(dim),ref.var(dim=-1).float())
    # Float32 entries need no float64-range fallback, even at both extremes.
    bound=torch.finfo(torch.float32).max
    extremes=torch.tensor([[bound,-bound,bound]],device=device,requires_grad=True)
    value=tp.as_lazy(extremes, chunk_size=2).var(-1, correction=0)
    ref=extremes.double().var(-1,correction=0).float()
    torch.testing.assert_close(value,ref)
    torch.testing.assert_close(torch.autograd.grad(value.sum(),extremes)[0],
                               torch.autograd.grad(ref.sum(),extremes)[0])
    # Keep an anchor and an offset mean across partial tiles, including in
    # backward: adding a large offset must not change variance or its VJP.
    shifted=(torch.arange(15,device=device,dtype=torch.float64).reshape(3,5)%5)*2
    offset=(shifted+1e16).requires_grad_()
    constant=torch.full((3,5),1e308,device=device,dtype=torch.float64,requires_grad=True)
    for dim in (None,-1):
        for name in ('var','std'):
            actual=getattr(tp.as_lazy(offset, chunk_size=3), name)(dim, correction=0)
            small=offset-1e16
            ref=getattr(small.flatten() if dim is None else small,name)(-1,correction=0)
            torch.testing.assert_close(actual,ref)
            torch.testing.assert_close(torch.autograd.grad(actual.sum(),offset)[0],
                                       torch.autograd.grad(ref.sum(),offset)[0])
            value=getattr(tp.as_lazy(constant, chunk_size=3), name)(dim, correction=0)
            assert value.eq(0).all()
            assert torch.autograd.grad(value.sum(),constant)[0].eq(0).all()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for data in ([],[2.],[2.,2.,2.],[1.,2.,3.]):
            raw=torch.tensor([data],device=device,dtype=torch.float64,requires_grad=True)
            operator=tp.as_lazy(raw, chunk_size=2)
            for name in ('var','std','logsumexp'):
                options={'correction':3} if name in ('var','std') else {}
                value=getattr(operator,name)(-1,**options)
                ref=getattr(raw,name)(dim=-1,**options)
                torch.testing.assert_close(value,ref,equal_nan=True)
                torch.testing.assert_close(torch.autograd.grad(value.sum(),raw)[0],
                                           torch.autograd.grad(ref.sum(),raw)[0],equal_nan=True)
    raw=torch.tensor([[float('inf'),float('-inf')],[float('-inf'),float('-inf')],
                      [float('nan'),0.]],device=device,requires_grad=True)
    value=tp.as_lazy(raw, chunk_size=1).logsumexp(-1);ref=raw.logsumexp(-1)
    torch.testing.assert_close(value,ref,equal_nan=True)
    torch.testing.assert_close(torch.autograd.grad(value.sum(),raw)[0],
                               torch.autograd.grad(ref.sum(),raw)[0],equal_nan=True)
    with pytest.raises(ValueError,match='only used'):source.reduce('sum',correction=0)


def test_weighted_operator_statistics_dense_oracle_and_scaling(device):
    torch.manual_seed(128)
    raw=torch.randn(2,1,3,4,device=device,dtype=torch.float64,requires_grad=True)
    weights=(torch.rand(1,2,4,device=device,dtype=raw.dtype)+.2).requires_grad_()
    with torch.no_grad():weights[...,0]=0
    operator=tp.as_lazy(raw, chunk_size=2)
    def reference(a,w,axis,correction=0,kind='mean'):
        mass=w.sum(axis,keepdim=True)
        mean=(a*w).sum(axis,keepdim=True)/mass
        if kind=='mean':return mean.squeeze(axis) if isinstance(axis,int) else mean.squeeze(-1).squeeze(-1)
        denominator=mass-correction*w.square().sum(axis,keepdim=True)/mass
        variance=(w*(a-mean).square()).sum(axis,keepdim=True)/denominator
        value=variance.sqrt() if kind=='std' else variance
        return value.squeeze(axis) if isinstance(axis,int) else value.squeeze(-1).squeeze(-1)
    a=raw.expand(2,2,3,4);w=weights.unsqueeze(-2).expand_as(a)
    actual=(operator.weighted_mean(weights,dim=[3]),operator.weighted_var(weights,correction=1),
            operator.weighted_std(weights),operator.T.weighted_mean(tp.as_lazy(w.transpose(-1, -2)), dim=-2))
    expected=(reference(a,w,-1),reference(a,w,-1,1,'var'),reference(a,w,-1,0,'std'),
              reference(a.transpose(-1,-2),w.transpose(-1,-2),-2))
    # Confidence weights may introduce an extra batch axis; positive indices
    # refer to the broadcast weighted shape, matching the native frontend.
    actual=(*actual,tp.as_lazy(raw[0, 0]).weighted_mean(weights[0], dim=[2]))
    expected=(*expected,reference(raw[0,0].unsqueeze(0),weights[0].unsqueeze(-2),-1))
    torch.testing.assert_close(actual,expected)
    cot=[torch.randn_like(v) for v in actual]
    torch.testing.assert_close(torch.autograd.grad(actual,(raw,weights),cot),
                               torch.autograd.grad(expected,(raw,weights),cot))
    # A weight operator derived from the same distance expression shares sources
    # and contributes through both the values and the normalized weights.
    x=torch.rand(3,2,device=device,dtype=raw.dtype,requires_grad=True)
    y=torch.rand(3,2,device=device,dtype=raw.dtype,requires_grad=True)
    gamma=torch.tensor(.7,device=device,dtype=raw.dtype,requires_grad=True)
    distance=tp.lazy(x, y, metric='sqeuclidean', chunk_size=2)
    affinity=(-gamma*distance).exp()
    dense=(x[:,None]-y[None]).square().sum(-1)
    w=(-gamma*dense).exp()
    mask=~torch.eye(3,device=device,dtype=torch.bool)
    actual=(distance.weighted_mean(affinity,dim=[-2,-1],exclude_self=True,keepdim=True),
            distance.weighted_var(affinity,dim=-2,correction=1))
    expected=(reference(dense,torch.where(mask,w,0.),(-2,-1)).reshape(1,1),
              reference(dense,w,-2,1,'var'))
    torch.testing.assert_close(actual,expected)
    cot=[torch.randn_like(v) for v in actual]
    torch.testing.assert_close(torch.autograd.grad(actual,(x,y,gamma),cot),
                               torch.autograd.grad(expected,(x,y,gamma),cot))
    # Scale invariance, partial tiles and large offsets without raw-moment
    # cancellation. No extra parameter grid or repeated metric catalog run.
    data=torch.arange(20,device=device,dtype=raw.dtype).reshape(4,5)/8+1e6
    w=torch.linspace(.1,1.,5,device=device,dtype=raw.dtype)
    op=tp.as_lazy(data, chunk_size=3)
    expected=reference(data,w.expand_as(data),-1,1,'var')
    for factor in (1e-180,1e180):
        torch.testing.assert_close(op.weighted_var(w*factor,correction=1),expected)
    # Keep the weighted mean offset separate across tiles and in both value/
    # confidence VJPs. Use a translated reference rather than an absolute mean.
    shifted=(torch.arange(15,device=device,dtype=raw.dtype).reshape(3,5)%7)*2
    translated=(shifted+1e16).requires_grad_()
    confidence=torch.linspace(.2,1.,5,device=device,dtype=raw.dtype,requires_grad=True)
    for name in ('mean','var','std'):
        actual=getattr(tp.as_lazy(translated, chunk_size=3), 'weighted_' + name)(confidence)
        ref=reference(translated-1e16,confidence,-1,kind=name)
        if name=='mean':ref=ref+1e16
        torch.testing.assert_close(actual,ref)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(),(translated,confidence)),
                                   torch.autograd.grad(ref.sum(),(translated,confidence)))
    zero=torch.zeros(5,device=device,dtype=raw.dtype)
    assert op.weighted_mean(zero).isnan().all()
    assert op.weighted_var(w*0).isnan().all()
    with pytest.raises(RuntimeError,match='nonnegative'):op.weighted_mean(-w)
    with pytest.raises(RuntimeError,match='finite'):op.weighted_mean(w*float('nan'))
    with pytest.raises(ValueError,match='as_lazy'):op.weighted_mean(w,dim=None)
    masked=torch.tensor([[float('nan'),2.,4.]],device=device,dtype=raw.dtype)
    result=tp.as_lazy(masked, chunk_size=2).weighted_mean(torch.tensor([0.0, 1.0, 1.0], device=device))
    torch.testing.assert_close(result,torch.tensor([3.],device=device,dtype=raw.dtype))
    empty=tp.as_lazy(data[:, :0])
    assert empty.weighted_mean(w[:0]).isnan().all()

    constant=torch.full((2,5),1.1,device=device,dtype=raw.dtype,requires_grad=True)
    zero_std=tp.as_lazy(constant, chunk_size=2).weighted_std(w)
    assert zero_std.eq(0).all()
    torch.testing.assert_close(torch.autograd.grad(zero_std.sum(),constant)[0],torch.zeros_like(constant))
    infinite=torch.tensor([[float('inf'),2.,4.]],device=device,dtype=raw.dtype)
    assert tp.as_lazy(infinite, chunk_size=1).weighted_mean(torch.ones(3, device=device)).isposinf().all()


def test_exact_selection_axes_interpolation_gradients_and_boundaries(device):
    torch.manual_seed(148)
    x = torch.rand(2, 1, 3, 5, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(1, 2, 4, 5, device=device, dtype=x.dtype, requires_grad=True)
    scale = torch.tensor(.7, device=device, dtype=x.dtype, requires_grad=True)
    a = tp.lazy(x, y, metric='sqeuclidean', chunk_size=2)
    op = (a * scale + .2).square().T
    dense = (tp.sqeuclidean_distances(x, y) * scale + .2).square().transpose(-1, -2)
    actual, expected = [], []
    for dim in (-1, 2):
        actual.append(op.median(dim, keepdim=True))
        expected.append(dense.median(dim, keepdim=True).values)
        for mode in ('linear', 'lower', 'higher', 'nearest', 'midpoint'):
            actual.append(op.quantile(.37, dim, interpolation=mode))
            expected.append(dense.quantile(.37, dim, interpolation=mode))
    torch.testing.assert_close(actual, expected)
    cot = [torch.randn_like(v) for v in actual]
    torch.testing.assert_close(torch.autograd.grad(actual, (x, y, scale), cot),
                               torch.autograd.grad(expected, (x, y, scale), cot))
    # Plain squared distances take a fused CUDA adjoint; composed expressions
    # above retain replay. Compare the two paths, including source/batch strides.
    for dim in (-1, -2):
        for source in (a, a.T):
            value = source.quantile(.37, dim)
            replay = (source + 0.).quantile(.37, dim)
            torch.testing.assert_close(value, replay)
            cot = torch.randn_like(value)
            torch.testing.assert_close(torch.autograd.grad(value, (x, y), cot),
                                       torch.autograd.grad(replay, (x, y), cot), rtol=1e-7, atol=1e-8)
    bad_x = torch.tensor([[0., 1.], [float('inf'), 2.], [float('nan'), 0.]],
                         device=device, dtype=x.dtype, requires_grad=True)
    bad_y = torch.tensor([[1., 1.], [-float('inf'), 0.], [3., 2.], [4., 0.]],
                         device=device, dtype=x.dtype, requires_grad=True)
    for dim in (-1, -2):
        plain = tp.lazy(bad_x, bad_y, metric='sqeuclidean', chunk_size=2)
        fast, replay = plain.quantile(.3, dim), (plain + 0.).quantile(.3, dim)
        torch.testing.assert_close(fast, replay, equal_nan=True)
        torch.testing.assert_close(torch.autograd.grad(fast.sum(), (bad_x, bad_y)),
                                   torch.autograd.grad(replay.sum(), (bad_x, bad_y)), equal_nan=True)
    # Lower middle and interpolation differ for even counts. Stable sort ties
    # choose the entry at that rank, rather than sharing its cotangent.
    matrix = torch.tensor([[1., 1., 1., 1.], [4., 2., 1., 3.]], device=device, requires_grad=True)
    a = tp.as_lazy(matrix, chunk_size=1)
    torch.testing.assert_close(a.median(), torch.tensor([1., 2.], device=device))
    torch.testing.assert_close(a.quantile(.5), torch.tensor([1., 2.5], device=device))
    torch.testing.assert_close(torch.autograd.grad(a.median().sum(), matrix)[0],
                               torch.tensor([[0., 1., 0., 0.], [0., 1., 0., 0.]], device=device))
    for q in (0., 1.):
        torch.testing.assert_close(a.quantile(q), matrix.quantile(q, -1))
    # torch.round's ties-to-even nearest ranks, including both parities. Ranks
    # use double q; torch 2.15 forms float32 ranks for float32 input, so
    # q=1/6 rounds up there. Its float64 reference keeps the exact tie.
    for q in (1./6., .5, 5./6.):
        torch.testing.assert_close(a.quantile(q, interpolation='nearest'),
                                   matrix.double().quantile(q, -1, interpolation='nearest').float())
    square = torch.rand(5, 5, device=device, dtype=torch.float64, requires_grad=True)
    mask = ~torch.eye(5, device=device, dtype=torch.bool)
    for dim in (-1, -2):
        e = square if dim == -1 else square.T
        expected = e[mask].reshape(5, 4).quantile(.3, -1)
        actual = tp.as_lazy(square, chunk_size=2).quantile(0.3, dim, exclude_self=True)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), square)[0],
                                   torch.autograd.grad(expected.sum(), square)[0])
    exceptional = torch.tensor([[float('nan'), 1., float('nan'), 2.],
                                [-float('inf'), 1., 2., float('inf')]], device=device, requires_grad=True)
    op = tp.as_lazy(exceptional, chunk_size=3)
    for q in (0., .25, .5, 1.):
        value, ref = op.quantile(q), exceptional.quantile(q, -1)
        torch.testing.assert_close(value, ref, equal_nan=True)
        expected_grad = torch.autograd.grad(ref.sum(), exceptional)[0]
        # PyTorch's sort does not promise tied NaN witnesses across devices.
        # This API explicitly orders ties by original axis index.
        expected_grad[0] = torch.tensor([0., 0., 1., 0.], device=device)
        torch.testing.assert_close(torch.autograd.grad(value.sum(), exceptional)[0],
                                   expected_grad, equal_nan=True)
    assert torch.autograd.grad(op.median().sum(), exceptional)[0][0, 0] == 1
    # Replaying a composed source preserves its zero-cotangent chain rules.
    value = op.square().quantile(.2)
    ref = exceptional.square().quantile(.2, -1)
    torch.testing.assert_close(torch.autograd.grad(value.sum(), exceptional)[0],
                               torch.autograd.grad(ref.sum(), exceptional)[0], equal_nan=True)
    integer = torch.tensor([[2**63-1, -2**63, 2**60, 0]], device=device)
    assert tp.as_lazy(integer, chunk_size=3).median().item() == 0
    assert tp.as_lazy(matrix[:0], chunk_size=3).quantile(0.5).shape == (0,)
    assert tp.as_lazy(matrix.expand(0, 2, 4)).median().shape == (0, 2)
    for q in (float('nan'), float('inf'), -1., 1.1):
        with pytest.raises(ValueError, match='q'): a.quantile(q)
    for q in (True, torch.tensor(.5), [.5]):
        with pytest.raises(TypeError, match='scalar'): a.quantile(q)
    for dim in (None, [-1], (-1, -2), True):
        with pytest.raises(TypeError, match='integer dim'): a.median(dim)
    with pytest.raises(ValueError, match='interpolation'): a.quantile(.5, interpolation='invalid')
    with pytest.raises(RuntimeError, match='float32/float64'): tp.as_lazy(integer).quantile(0.5)
    with pytest.raises(RuntimeError, match='Boolean'): tp.as_lazy(integer.bool()).median()
    for empty in (matrix[:, :0], torch.ones(1, 1, device=device)):
        with pytest.raises(RuntimeError, match='eligible'): tp.as_lazy(empty).median(exclude_self=empty.shape[-1] == 1)


@pytest.mark.compiler
def test_exact_selection_native_and_compiler(device):
    torch.manual_seed(149)
    x = torch.rand(3, 4, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(5, 4, device=device, dtype=x.dtype, requires_grad=True)
    def run(x, y):
        a = tp.lazy(x, y, metric='sqeuclidean', chunk_size=2)
        return a.quantile(.4, keepdim=True) + a.median(0).sum()
    eager = run(x, y)
    compiled = torch.compile(run, backend='aot_eager', fullgraph=True)
    torch.testing.assert_close(compiled(x, y), eager)
    torch.testing.assert_close(torch.autograd.grad(compiled(x, y).sum(), (x, y)),
                               torch.autograd.grad(eager.sum(), (x, y)))
    class M(torch.nn.Module):
        def forward(self, x, y): return run(x, y)
    program = torch.export.export(M(), (x, y))
    buffer = io.BytesIO(); torch.export.save(program, buffer); buffer.seek(0)
    torch.testing.assert_close(torch.export.load(buffer).module()(x, y), run(x, y))
    # Check the native opaque core, including saved int64 selection indices.
    from torch.utils._python_dispatch import TorchDispatchMode
    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.torchpairwise._expression_selection.default:
                self.args = args
            return func(*args, **(kwargs or {}))
    capture = Capture()
    with capture:
        run(x, y)
    torch.library.opcheck(torch.ops.torchpairwise._expression_selection.default, capture.args)
