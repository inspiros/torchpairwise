"""One lazy type: construction, capabilities, torch dispatch and deployment."""
from functools import partial
import importlib.util

import pytest
import torch
import torchpairwise as tp


def test_constructor_vector_metric_and_metadata(device):
    x = torch.randn(2, 3, 4, device=device, dtype=torch.float64)
    y = torch.randn(1, 5, 4, device=device, dtype=x.dtype)
    operator = tp.lazy(x, y, 'sqeuclidean', chunk_size=2)
    assert isinstance(operator, tp.LazyPairwise) and operator.entrywise
    assert operator.ndim == 3 and operator.size() == torch.Size((2, 3, 5))
    assert operator.size(-1) == 5 and operator.dtype == x.dtype and operator.device == x.device
    torch.testing.assert_close(operator.to_dense(), tp.sqeuclidean_distances(x, y))
    torch.testing.assert_close(operator.mT.to_dense(), operator.T.to_dense())
    with pytest.raises(AttributeError):
        operator.entrywise = False


def test_constructor_kernel_short_and_full_names(device):
    x = torch.randn(3, 2, device=device)
    y = torch.randn(5, 2, device=device)
    expected = tp.rbf_kernel(x, y, gamma=.3)
    short = tp.lazy(x, y, 'rbf', gamma=.3, chunk_size=2)
    full = tp.lazy(x, y, 'rbf_kernel', gamma=.3, chunk_size=2)
    assert short.entrywise and type(short) is type(full)
    torch.testing.assert_close(short.to_dense(), expected)
    torch.testing.assert_close(full.to_dense(), expected, rtol=0, atol=0)


def test_constructor_structured_family(device):
    means = torch.randn(3, 2, device=device, dtype=torch.float64)
    covariances = torch.eye(2, device=device, dtype=means.dtype).repeat(3, 1, 1)
    source = (means, covariances)
    operator = tp.lazy(source, metric='gaussian_wasserstein', squared=True, chunk_size=2)
    expected = tp.gaussian_wasserstein_distances(*source, squared=True)
    assert isinstance(operator, tp.LazyPairwise) and operator.entrywise
    torch.testing.assert_close(operator.to_dense(), expected)


def test_constructor_custom_comparison(device):
    comparison = tp.make_pairwise(lambda a, b: (a - b).square().sum())
    x, y = torch.randn(3, 2, device=device), torch.randn(5, 2, device=device)
    operator = tp.lazy(x, y, comparison, chunk_size=2)
    assert operator.entrywise
    torch.testing.assert_close(operator.to_dense(), comparison(x, y))


def test_constructor_feature_comparison_and_prepared_reuse(device):
    calls = []
    def encoder(samples):
        calls.append(samples.shape[0])
        return samples.square()
    comparison = tp.make_feature_pairwise(encoder, 'sqeuclidean')
    x, y = torch.randn(3, 2, device=device), torch.randn(5, 2, device=device)
    operator = tp.lazy(x, y, comparison, chunk_size=2)
    expected = tp.sqeuclidean_distances(x.square(), y.square())
    torch.testing.assert_close(operator.to_dense(), expected)
    assert calls == [3, 5]
    prepared = comparison.prepare(x, y)
    torch.testing.assert_close(tp.lazy(prepared, chunk_size=2).to_dense(), expected)
    assert calls == [3, 5, 3, 5]


def test_constructor_dense_leaf_preserves_mutation_and_gradient(device):
    matrix = torch.randn(3, 4, device=device, dtype=torch.float64, requires_grad=True)
    operator = tp.as_lazy(matrix, chunk_size=2)
    assert operator.entrywise
    dense = operator.to_dense()
    torch.testing.assert_close(dense, matrix, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(dense.sum(), matrix)[0], torch.ones_like(matrix))
    with torch.no_grad():
        matrix.add_(2)
    torch.testing.assert_close(operator.to_dense(), matrix, rtol=0, atol=0)


def test_entrywise_binary_capability_cells(device):
    a, b = torch.randn(3, 3, device=device), torch.randn(3, 3, device=device)
    A, B = tp.as_lazy(a), tp.as_lazy(b)
    for actual, expected in ((A + B, a + b), (A - B, a - b), (A * B, a * b)):
        assert actual.entrywise
        torch.testing.assert_close(actual.to_dense(), expected)
    composition = A @ B
    assert isinstance(composition, tp.LazyPairwise) and not composition.entrywise
    torch.testing.assert_close(composition.to_dense(), a @ b)


def test_mixed_and_linear_binary_capability_cells(device):
    a, b = torch.randn(3, 3, device=device), torch.randn(3, 3, device=device)
    A, B = tp.as_lazy(a), tp.as_lazy(b)
    L = tp.diag_embed(torch.tensor([1., 2., 3.], device=device))
    diagonal = torch.diag(torch.tensor([1., 2., 3.], device=device))
    for left, right, l_dense, r_dense in ((A, L, a, diagonal), (L, B, diagonal, b),
                                          (L, L, diagonal, diagonal)):
        for result, expected in ((left + right, l_dense + r_dense),
                                 (left - right, l_dense - r_dense),
                                 (left @ right, l_dense @ r_dense)):
            assert not result.entrywise
            torch.testing.assert_close(result.to_dense(), expected)
        with pytest.raises(TypeError, match='Hadamard.*entrywise.*@'):
            left * right


def test_scalar_and_tensor_capability_cells(device):
    matrix = torch.randn(3, 3, device=device)
    entrywise = tp.as_lazy(matrix)
    linear = entrywise @ entrywise
    for scalar in (2., matrix.new_tensor(2.)):
        for result, expected in ((entrywise + scalar, matrix + scalar),
                                 (scalar - entrywise, scalar - matrix),
                                 (entrywise * scalar, matrix * scalar),
                                 (scalar * entrywise, scalar * matrix)):
            assert result.entrywise
            torch.testing.assert_close(result.to_dense(), expected)
        for result in (linear * scalar, scalar * linear):
            assert not result.entrywise
            torch.testing.assert_close(result.to_dense(), (matrix @ matrix) * scalar)
        for operation in (lambda: linear + scalar, lambda: scalar + linear,
                          lambda: linear - scalar, lambda: scalar - linear):
            with pytest.raises(TypeError, match='eye_like'):
                operation()
    vector = torch.randn(3, device=device)
    torch.testing.assert_close(entrywise @ vector, matrix @ vector)
    torch.testing.assert_close(vector @ entrywise, vector @ matrix)
    torch.testing.assert_close(linear @ vector, matrix @ matrix @ vector)
    torch.testing.assert_close(vector @ linear, vector @ matrix @ matrix)
    for operator in (entrywise, linear):
        with pytest.raises((TypeError, ValueError)):
            operator * matrix
        with pytest.raises((TypeError, ValueError)):
            operator + matrix


def test_linear_sum_mean_are_linear_functionals(device):
    a = torch.randn(2, 3, 4, device=device, dtype=torch.float64)
    b = torch.randn(1, 4, 5, device=device, dtype=a.dtype)
    operator = tp.as_lazy(a) @ tp.as_lazy(b)
    dense = a @ b
    for dim in (None, -1, -2):
        for keepdim in (False, True):
            axis = (-2, -1) if dim is None else dim
            torch.testing.assert_close(operator.sum(dim, keepdim=keepdim), dense.sum(axis, keepdim=keepdim))
            torch.testing.assert_close(operator.mean(dim, keepdim=keepdim), dense.mean(axis, keepdim=keepdim))


@pytest.mark.parametrize('consumer', [
    lambda a: a.amin(-1), lambda a: a.max(-1), lambda a: a.argmax(-1),
    lambda a: a.prod(-1), lambda a: a.all(-1), lambda a: a.any(-1),
    lambda a: a.var(-1), lambda a: a.std(-1), lambda a: a.logsumexp(-1),
    lambda a: a.count_nonzero(-1), lambda a: a.topk(1), lambda a: a.median(-1),
    lambda a: a.quantile(.5), lambda a: a.exp(), lambda a: a.square(),
    lambda a: a.softmax(-1), lambda a: a.softmax_matmul(torch.ones(3, device=a.device)),
    lambda a: a.diagonal(), lambda a: a.weight_rows(torch.ones(3, device=a.device)),
])
def test_linear_consumer_errors_name_capability_and_escape(consumer, device):
    operator = tp.diag_embed(torch.ones(3, device=device))
    with pytest.raises(TypeError, match='linear.*to_dense|entrywise.*to_dense'):
        consumer(operator)


def test_torch_arithmetic_and_elementwise_groups(device):
    matrix = torch.randn(3, 3, device=device)
    operator = tp.as_lazy(matrix)
    scalar = matrix.new_tensor(.2)
    results = ((torch.add(input=operator, other=scalar, alpha=2), matrix + 2 * scalar),
               (torch.sub(scalar, operator), scalar - matrix),
               (torch.mul(input=operator, other=scalar), matrix * scalar),
               (torch.neg(operator), -matrix), (torch.exp(input=operator), matrix.exp()),
               (torch.square(input=operator), matrix.square()))
    for result, expected in results:
        torch.testing.assert_close(result.to_dense(), expected)


def test_torch_reduction_group_forwards_kwargs_to_methods(device):
    matrix = torch.rand(3, 4, device=device, dtype=torch.float64) + .5
    operator = tp.as_lazy(matrix, chunk_size=2)
    for function in (torch.sum, torch.mean, torch.amin, torch.amax, torch.min, torch.max,
                     torch.argmin, torch.argmax, torch.prod, torch.all, torch.any,
                     torch.var, torch.std, torch.logsumexp, torch.count_nonzero, torch.median):
        options = {'dim': -1}
        if function not in (torch.count_nonzero,):
            options['keepdim'] = True
        torch.testing.assert_close(function(input=operator, **options),
                                   getattr(operator, function.__name__)(**options))
    torch.testing.assert_close(torch.topk(input=operator, k=2, dim=-1, largest=False),
                               operator.topk(2, dim=-1, largest=False))
    torch.testing.assert_close(torch.quantile(input=operator, q=.3, dim=-1, keepdim=True),
                               operator.quantile(.3, dim=-1, keepdim=True))


def test_torch_linear_group_and_concatenation(device):
    matrix = torch.randn(3, 3, device=device)
    operator = tp.as_lazy(matrix)
    rhs, vector = torch.randn(3, 2, device=device), torch.randn(3, device=device)
    torch.testing.assert_close(torch.matmul(operator, rhs), operator.matmul(rhs))
    torch.testing.assert_close(torch.mm(operator, rhs), operator.matmul(rhs))
    torch.testing.assert_close(torch.mv(operator, vector), operator.matmul(vector))
    torch.testing.assert_close(torch.matmul(rhs.T, operator), rhs.T @ matrix)
    torch.testing.assert_close(torch.softmax(input=operator, dim=-1) @ rhs, operator.softmax(-1) @ rhs)
    torch.testing.assert_close(torch.diagonal(input=operator, offset=0, dim1=-2, dim2=-1), operator.diagonal())
    torch.testing.assert_close(torch.trace(operator), operator.diagonal().sum())
    torch.testing.assert_close(torch.transpose(input=operator, dim0=-2, dim1=-1).to_dense(), operator.mT.to_dense())
    for result, expected in ((torch.hstack((operator, operator)), torch.hstack((matrix, matrix))),
                             (torch.vstack((operator, operator)), torch.vstack((matrix, matrix))),
                             (torch.block_diag(operator, operator), torch.block_diag(matrix, matrix))):
        assert not result.entrywise
        torch.testing.assert_close(result.to_dense(), expected)


def test_unsupported_torch_calls_and_softmax_axis_errors(device):
    operator = tp.as_lazy(torch.eye(3, device=device))
    for function in (torch.erf, torch.atan, torch.lgamma):  # not expression nodes
        with pytest.raises(TypeError, match='not lazy.*to_dense.*torchpairwise.linalg'):
            function(operator)
    with pytest.raises(TypeError, match='tp.linalg|torchpairwise.linalg'):
        torch.linalg.solve(operator, torch.ones(3, device=device))
    with pytest.raises((TypeError, ValueError), match='mT.softmax'):
        operator.softmax(-2)
    with pytest.raises(TypeError, match='offset 0.*last two'):
        torch.diagonal(operator, offset=1)


def test_min_max_namedtuples_nan_rows_ties_and_arg_reductions(device):
    raw = torch.tensor([[2., 1., 1., 4.], [float('nan'), 3., float('nan'), 2.],
                        [5., 5., 2., 2.]], device=device, dtype=torch.float64, requires_grad=True)
    operator = tp.as_lazy(raw, chunk_size=1)
    for method, result_type in (('min', torch.return_types.min), ('max', torch.return_types.max)):
        for dim in (-1, -2):
            for keepdim in (False, True):
                actual = getattr(operator, method)(dim, keepdim=keepdim)
                expected = getattr(raw, method)(dim, keepdim=keepdim)
                assert isinstance(actual, result_type)
                torch.testing.assert_close(actual, expected, equal_nan=True)
                torch.testing.assert_close(getattr(operator, 'a' + method)(dim, keepdim=keepdim), expected.values, equal_nan=True)
                torch.testing.assert_close(getattr(operator, 'arg' + method)(dim, keepdim=keepdim), expected.indices)
                torch.testing.assert_close(torch.autograd.grad(actual.values.sum(), raw)[0],
                                           torch.autograd.grad(expected.values.sum(), raw)[0])
        actual = getattr(operator, method)()
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, getattr(raw, method)(), equal_nan=True)
    torch.testing.assert_close(operator.argmin(), raw.argmin())
    torch.testing.assert_close(operator.argmax(), raw.argmax())
    # Existing native median returns values; the public method preserves that contract.
    torch.testing.assert_close(operator.median(-1), raw.median(-1).values, equal_nan=True)


def test_predicate_escape_hatch(device):
    matrix = torch.tensor([[0., 2.], [3., 4.]], device=device)
    operator = tp.as_lazy(matrix)
    torch.testing.assert_close(operator.all(-1, predicate=lambda tile: tile > 1), (matrix > 1).all(-1))
    torch.testing.assert_close(operator.any(-2, predicate=lambda tile: tile > 1), (matrix > 1).any(-2))


def test_identity_diagonal_block_and_group_constructors(device):
    matrix = torch.randn(3, 3, device=device, dtype=torch.float64)
    operator = tp.as_lazy(matrix)
    shifted = operator + .7 * tp.eye_like(operator)
    assert not shifted.entrywise and shifted._kind == 'add_diagonal'
    torch.testing.assert_close(shifted.to_dense(), matrix + .7 * torch.eye(3, device=device, dtype=matrix.dtype))
    diagonal = torch.tensor([1., 2., 3.], device=device, dtype=matrix.dtype)
    torch.testing.assert_close(tp.diag_embed(diagonal).to_dense(), torch.diag_embed(diagonal))
    torch.testing.assert_close(tp.block([[operator, tp.eye_like(operator)]]).to_dense(),
                               torch.cat((matrix, torch.eye(3, device=device, dtype=matrix.dtype)), -1))
    labels = torch.tensor([1, 1, 8], device=device)
    projection = tp.group_projection(labels, like=operator)
    expected = torch.tensor([[.5, .5, 0.], [.5, .5, 0.], [0., 0., 1.]], device=device, dtype=matrix.dtype)
    assert not projection.entrywise
    torch.testing.assert_close(projection.to_dense(), expected)


def test_solver_on_composed_lazy_objects(device):
    matrix = torch.randn(5, 3, device=device, dtype=torch.float64)
    operator = tp.as_lazy(matrix)
    system = operator.mT @ operator
    system = system + .5 * tp.eye_like(system)
    rhs = torch.randn(3, 2, device=device, dtype=matrix.dtype)
    result = tp.linalg.cg(system, rhs, rtol=1e-11, max_iter=30)
    expected = torch.linalg.solve(matrix.T @ matrix + .5 * torch.eye(3, device=device, dtype=matrix.dtype), rhs)
    assert result.converged.all()
    torch.testing.assert_close(result.solution, expected, rtol=1e-8, atol=1e-9)


def test_removed_names_are_absent():
    removed = ('streaming', 'pairwise_operator', 'kernel_operator', 'matrix_operator',
               'PairwiseOperator', 'KernelOperator', 'MatrixOperator', 'ScalarExpression',
               'CustomPairwiseOperator', 'LinearOperator', 'NeighborResult', 'aslinearoperator',
               'k_neighbors', 'nearest_neighbors', 'kernel_sum', 'reduce_distances', 'reduce_kernels',
               'kernel_bilinear', 'kernel_quadratic', 'kernel_displacement',
               'kernel_residual_outer_product', 'reduce_image_maps')
    for name in removed:
        assert not hasattr(tp, name), name
    assert importlib.util.find_spec('torchpairwise.streaming') is None
    comparison = tp.make_pairwise(lambda a, b: (a - b).square().sum())
    features = tp.make_feature_pairwise(lambda x: x, 'sqeuclidean')
    assert not hasattr(comparison, 'operator') and not hasattr(features, 'operator')
    linear = tp.diag_embed(torch.ones(2))
    for name in ('compose', 'scale', 'add_diagonal', 'transpose', 'hstack', 'vstack'):
        assert not hasattr(linear, name), name
    assert len(tp.__all__) == len(set(tp.__all__))


@pytest.mark.compiler
@pytest.mark.parametrize('operation', ['matmul', 'sum', 'softmax', 'cg'])
def test_lazy_compile_fullgraph(operation, device):
    class Model(torch.nn.Module):
        def forward(self, x, values):
            kernel = tp.lazy(x, metric='rbf', gamma=.3, chunk_size=2)
            if operation == 'matmul':
                return torch.matmul(kernel, values)
            if operation == 'sum':
                return torch.sum(kernel, -1)
            if operation == 'softmax':
                return kernel.softmax(-1).matmul(values)
            return tp.linalg.cg(kernel + .5 * tp.eye_like(kernel), values, rtol=1e-10, max_iter=30).solution
    x = torch.randn(5, 3, device=device, dtype=torch.float64)
    values = torch.randn(5, 2, device=device, dtype=x.dtype)
    dense = tp.rbf_kernel(x, gamma=.3)
    expected = {'matmul': lambda: dense @ values, 'sum': lambda: dense.sum(-1),
                'softmax': lambda: dense.softmax(-1) @ values,
                'cg': lambda: torch.linalg.solve(dense + .5 * torch.eye(5, device=device, dtype=x.dtype), values)}[operation]()
    model = Model()
    torch.compiler.reset()
    compiled = torch.compile(model, backend='aot_eager', fullgraph=True)
    torch.testing.assert_close(compiled(x, values), expected, rtol=1e-8, atol=1e-9)
    torch.compiler.reset()


@pytest.mark.compiler
def test_lazy_tensor_matmul_operator_is_rejected_under_dynamo(device):
    # Dynamo passes `A @ v` and `v @ A` to __torch_function__ identically.
    def forward(x, values):
        return tp.lazy(x, metric='rbf', gamma=.3, chunk_size=2) @ values
    x = torch.randn(5, 3, device=device, dtype=torch.float64)
    torch.compiler.reset()
    with pytest.raises(Exception, match='A.matmul'):
        torch.compile(forward, backend='eager', fullgraph=True)(x, torch.randn(5, 2, device=device, dtype=x.dtype))
    torch.compiler.reset()


@pytest.mark.compiler
def test_lazy_export_builds_binding_inside_forward(device):
    class Model(torch.nn.Module):
        def forward(self, x, y, values):
            kernel = tp.lazy(x, y, 'rbf', gamma=.2, chunk_size=2)
            return kernel @ values, torch.sum(kernel, dim=-1)
    x, y = torch.randn(3, 2, device=device), torch.randn(5, 2, device=device)
    values = torch.randn(5, 2, device=device)
    module = torch.export.export(Model(), (x, y, values)).module()
    dense = tp.rbf_kernel(x, y, gamma=.2)
    torch.testing.assert_close(module(x, y, values), (dense @ values, dense.sum(-1)))


def test_softmax_transpose_and_linear_sums(device):
    matrix = torch.randn(3, 4, device=device, dtype=torch.float64)
    operator = tp.as_lazy(matrix, chunk_size=2).mT.softmax(-1)
    dense = matrix.mT.softmax(-1)
    assert not operator.entrywise
    values = torch.randn(3, 2, device=device, dtype=matrix.dtype)
    torch.testing.assert_close(operator @ values, dense @ values)
    torch.testing.assert_close(operator.mT @ torch.ones(4, device=device, dtype=matrix.dtype), dense.sum(-2))
    torch.testing.assert_close(operator.sum(-1), dense.sum(-1))
    torch.testing.assert_close(operator.mean(-2), dense.mean(-2))


def test_consumers_share_evaluate_boundary(device, monkeypatch):
    matrix = torch.rand(3, 3, device=device, dtype=torch.float64)
    operator = tp.as_lazy(matrix, chunk_size=2)
    calls = []
    evaluate = tp.LazyPairwise._evaluate
    def record(self, operation, **options):
        calls.append(operation)
        return evaluate(self, operation, **options)
    monkeypatch.setattr(tp.LazyPairwise, '_evaluate', record)
    values = torch.ones(3, device=device, dtype=matrix.dtype)
    for operation in (lambda: operator @ values, lambda: operator.sum(-1),
                      lambda: operator.to_dense(), lambda: operator.topk(1),
                      lambda: operator.softmax(-1) @ values,
                      lambda: (operator @ operator) @ values):
        previous = len(calls)
        operation()
        assert len(calls) > previous
    assert 'matmul' in calls and 'reduce' in calls and 'block' in calls and 'softmax_matmul' in calls


def test_solver_on_softmax_map_uses_native_scalar_plan(device):
    matrix = torch.randn(3, 3, device=device, dtype=torch.float64)
    scores = tp.as_lazy(matrix, chunk_size=2)
    softmax = scores.softmax(-1)
    system = softmax + 2. * tp.eye_like(softmax)
    rhs = torch.randn(3, device=device, dtype=matrix.dtype)
    result = tp.linalg.gmres(system, rhs, restart=3, max_iter=20, rtol=1e-11)
    expected = torch.linalg.solve(matrix.softmax(-1) + 2. * torch.eye(3, device=device, dtype=matrix.dtype), rhs)
    assert result.converged.all()
    torch.testing.assert_close(result.solution, expected, rtol=1e-8, atol=1e-9)
    with pytest.raises(TypeError, match='solver plans.*to_dense'):
        tp.linalg.gmres(scores.softmax(-1, exclude_self=True), rhs)
