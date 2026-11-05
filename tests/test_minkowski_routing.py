"""Minkowski routing, zero subgradients and infinity's split-tie contract."""
import math

import pytest
import torch
import torchpairwise as tp


def _assert_values_and_grads(actual, expected, inputs):
    torch.testing.assert_close(actual, expected)
    # Nonuniform upstream gradients expose wrong pair/feature accumulation.
    upstream = torch.linspace(.3, 1.7, actual.numel(), device=actual.device,
                              dtype=actual.dtype).reshape_as(actual)
    grads = torch.autograd.grad(actual, inputs, upstream)
    refs = torch.autograd.grad(expected, inputs, upstream)
    for grad, ref in zip(grads, refs):
        assert torch.isfinite(grad).all()
        torch.testing.assert_close(grad, ref)


@pytest.mark.parametrize('p', [.5, 1., 1.5, 3., math.inf])
def test_minkowski_random_and_batched_cdist(p, device):
    generator = torch.Generator().manual_seed(917)
    x = torch.randn(2, 4, 5, generator=generator, dtype=torch.float64).to(device).requires_grad_()
    y = torch.randn(1, 3, 5, generator=generator, dtype=torch.float64).to(device).requires_grad_()
    _assert_values_and_grads(tp.minkowski_distances(x, y, p=p),
                            torch.cdist(x, y, p=p), (x, y))


@pytest.mark.parametrize('p', [.5, 1., 1.5, 3., math.inf])
def test_minkowski_duplicate_and_zero_coordinates(p, device):
    # All pairs have zero coordinate differences; diagonal/duplicate pairs
    # additionally have zero distance. cdist defines both subgradients as zero.
    x = torch.tensor([[1., 2., 3.], [1., 2., 3.], [4., 2., 3.]],
                     device=device, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    _assert_values_and_grads(tp.minkowski_distances(x, y, p=p),
                            torch.cdist(x, y, p=p), (x, y))


def test_chebyshev_ties_split_and_zero_distance(device):
    x = torch.tensor([[0., 0., 0.], [1., -1., 1.], [0., 0., 0.]],
                     device=device, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[1., 1., -1.], [0., 0., 0.]],
                     device=device, dtype=torch.float64, requires_grad=True)
    expected = torch.linalg.vector_norm(x[:, None] - y[None], ord=math.inf, dim=-1)
    _assert_values_and_grads(tp.chebyshev_distances(x, y), expected, (x, y))


def test_weighted_infinity_masks_features_and_splits_ties(device):
    x = torch.tensor([[0., 0., 0.], [1., -1., 1.]],
                     device=device, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[1., 9., -1.], [0., 0., 0.]],
                     device=device, dtype=torch.float64, requires_grad=True)
    w = torch.tensor([.2, 0., 5.], device=device, dtype=torch.float64, requires_grad=True)
    mask = (w > 0).to(x.dtype)
    expected = torch.linalg.vector_norm((x[:, None] - y[None]) * mask,
                                       ord=math.inf, dim=-1) + w.sum() * 0
    _assert_values_and_grads(tp.wminkowski_distances(x, y, p=math.inf, w=w),
                            expected, (x, y, w))


def test_infinity_nan_in_nonfirst_feature(device):
    x = torch.tensor([[0., math.nan, 2.], [0., 1., 2.]], device=device)
    y = torch.tensor([[1., 2., 4.]], device=device)
    expected = torch.tensor([[math.nan], [2.]], device=device)
    # These are the three newly routed public branches (unweighted wminkowski
    # is tested separately below); NaN propagation is a deliberate cdist change.
    outputs = (tp.minkowski_distances(x, y, p=math.inf),
               tp.chebyshev_distances(x, y),
               tp.wminkowski_distances(x, y, p=math.inf, w=torch.ones(3, device=device)))
    for actual in outputs:
        torch.testing.assert_close(actual, expected, equal_nan=True)


def test_unweighted_wminkowski_zero_coordinate_subgradient(device):
    x = torch.tensor([[0., 2., 3.], [1., 2., 3.]],
                     device=device, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    _assert_values_and_grads(tp.wminkowski_distances(x, y, p=.5),
                            torch.cdist(x, y, p=.5), (x, y))


@pytest.mark.parametrize('kind', ['minkowski', 'chebyshev', 'weighted_inf'])
def test_minkowski_routing_gradcheck(kind, device):
    # Fixed differences with separated absolute values avoid infinity ties and
    # keep finite differences away from zero and weight-mask boundaries.
    x = torch.tensor([[.2, 1.3, -.7], [2.1, -.4, .8]],
                     device=device, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[-1.1, .1, 2.7], [.7, -2.3, -.1]],
                     device=device, dtype=torch.float64, requires_grad=True)
    if kind == 'minkowski':
        fn, args = lambda a, b: tp.minkowski_distances(a, b, p=3.), (x, y)
    elif kind == 'chebyshev':
        fn, args = tp.chebyshev_distances, (x, y)
    else:
        # Gradcheck stays away from the zero-weight mask discontinuity.
        w = torch.tensor([.7, .3, 2.], device=device, dtype=x.dtype, requires_grad=True)
        fn = lambda a, b, weights: tp.wminkowski_distances(a, b, p=math.inf, w=weights)
        args = (x, y, w)
    assert torch.autograd.gradcheck(fn, args)


def test_streaming_chebyshev_matches_dense_tie_gradients(device):
    x = torch.tensor([[0., 0., 0.], [1., -1., 1.], [2., 2., 2.], [0., 0., 0.]],
                     device=device, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[1., 1., -1.], [0., 0., 0.], [-1., -1., 1.]],
                     device=device, dtype=torch.float64, requires_grad=True)
    v = torch.arange(6, device=device, dtype=x.dtype).reshape(3, 2).requires_grad_()
    actual = tp.lazy(x, y, metric='chebyshev', chunk_size=2).matmul(v)
    expected = tp.chebyshev_distances(x, y).matmul(v)
    _assert_values_and_grads(actual, expected, (x, y, v))


def test_paired_and_condensed_routing(device):
    x = torch.tensor([[0., 0., 0.], [1., 1., -1.], [1., 0., 0.]],
                     device=device, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[1., 1., -1.], [0., 0., 0.], [1., 0., 0.]],
                     device=device, dtype=torch.float64, requires_grad=True)
    paired = tp.paired_distances(x, y, metric='minkowski', p=.5)
    expected_paired = torch.cdist(x, y, p=.5).diagonal()
    _assert_values_and_grads(paired, expected_paired, (x, y))
    i, j = torch.triu_indices(3, 3, 1, device=device)
    condensed = tp.pdist(x, metric='chebyshev')
    expected_condensed = torch.linalg.vector_norm(x[i] - x[j], ord=math.inf, dim=-1)
    _assert_values_and_grads(condensed, expected_condensed, (x,))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_routed_dtype_values_and_gradients(dtype, device):
    x = torch.tensor([[0., 0., 0.], [1., 1., -1.]], device=device, dtype=dtype, requires_grad=True)
    y = torch.tensor([[1., 1., -1.], [0., 0., 0.]], device=device, dtype=dtype, requires_grad=True)
    for p in (.5, math.inf):
        actual = tp.minkowski_distances(x, y, p=p)
        assert actual.dtype == dtype
        # float reference works on CPU too, where cdist does not support Half.
        if math.isinf(p):
            expected = torch.linalg.vector_norm(x.float()[:, None] - y.float()[None], ord=p, dim=-1)
        else:
            expected = torch.cdist(x.float(), y.float(), p=p)
        _assert_values_and_grads(actual, expected.to(dtype), (x, y))


def test_routed_autocast(device):
    dtype = torch.float16 if device == 'cuda' else torch.bfloat16
    x = torch.tensor([[0., 0., 0.], [1., 1., -1.]], device=device,
                     dtype=dtype, requires_grad=True)
    y = torch.tensor([[1., 1., -1.], [0., 0., 0.]], device=device,
                     dtype=dtype, requires_grad=True)
    for p in (3., math.inf):
        expected = tp.minkowski_distances(x.float(), y.float(), p=p)
        with torch.autocast(device, dtype=dtype):
            actual = tp.minkowski_distances(x, y, p=p)
        assert actual.dtype == torch.float32
        _assert_values_and_grads(actual, expected, (x, y))


@pytest.mark.compiler
def test_minkowski_routing_compile_and_export(device):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return (tp.minkowski_distances(x, y, p=3.),
                    tp.chebyshev_distances(x, y))

    x = torch.tensor([[.1, .7, 1.3], [2.1, -.4, .8]], device=device, requires_grad=True)
    y = torch.tensor([[-1.1, .1, 2.7], [.7, -2.3, -.1]], device=device, requires_grad=True)
    model = Model()
    expected = model(x, y)
    compiled = torch.compile(model, fullgraph=True)
    actual = compiled(x, y)
    # One backward traverses both outputs of the compiled graph.
    _assert_values_and_grads(torch.stack(actual), torch.stack(expected), (x, y))
    exported = torch.export.export(model, (x, y)).module()
    for actual, reference in zip(exported(x, y), model(x, y)):
        torch.testing.assert_close(actual, reference)


def test_primitive_negative_infinity_ties_and_nan(device):
    x = torch.tensor([[0., 0., 0.], [1., 1., 1.]],
                     device=device, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[1., -1., 2.], [0., 0., 0.]],
                     device=device, dtype=torch.float64, requires_grad=True)
    expected = torch.linalg.vector_norm(x[:, None] - y[None], ord=-math.inf, dim=-1)
    _assert_values_and_grads(torch.ops.torchpairwise._ppminkowski(x, y, -math.inf),
                            expected, (x, y))
    nan_input = x.detach().clone()
    nan_input[0, 1] = math.nan
    result = torch.ops.torchpairwise._ppminkowski(nan_input, y.detach(), -math.inf)
    assert result[0].isnan().all()
    assert torch.isfinite(result[1]).all()


def test_routed_empty_feature_axis_is_rejected(device):
    x = torch.empty(2, 3, 0, device=device, dtype=torch.float64)
    for p in (.5, math.inf):
        with pytest.raises(ValueError, match='non-empty'):
            tp.minkowski_distances(x, x, p=p)
