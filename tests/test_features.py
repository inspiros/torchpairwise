"""Within-call encoding reuse, collection layouts and user-owned gradient policy."""
import pytest
import torch

import torchpairwise as tp


def squared_pairs(x, y=None, *, scale=1.):
    y = x if y is None else y
    return scale * (x.unsqueeze(-2) - y.unsqueeze(-3)).square().sum(-1)


@pytest.mark.parametrize('chunk_size', [None, 1, 5])
def test_features_original_batches_noncontiguous_and_gradients(chunk_size, device):
    torch.manual_seed(455)
    x = torch.randn(2, 1, 3, 4, 2, dtype=torch.float64, device=device).transpose(-1, -2).requires_grad_()
    y = torch.randn(1, 3, 4, 4, 2, dtype=x.dtype, device=device).transpose(-1, -2).requires_grad_()
    model = torch.nn.Linear(8, 3, dtype=x.dtype, device=device)
    scale = torch.tensor(1.3, dtype=x.dtype, device=device, requires_grad=True)
    calls = []

    def encoder(flat):
        calls.append(tuple(flat.shape))
        return model(flat.flatten(1)).tanh()

    def head(a, b):
        return squared_pairs(a, b, scale=scale)

    wrapped = tp.make_feature_pairwise(encoder, head=head, sample_ndim=2, encode_chunk_size=chunk_size)
    actual = wrapped(x, y)
    assert actual.shape == (2, 3, 3, 4)
    counts = [6, 12] if chunk_size is None else [min(chunk_size, count - start)
              for count in (6, 12) for start in range(0, count, chunk_size)]
    assert calls == [(count, 2, 4) for count in counts]
    # Independent explicit sample encodings restore the two original layouts.
    a = torch.stack([model(sample.flatten()).tanh() for sample in x.reshape(6, 2, 4)]).reshape(2, 1, 3, 3)
    b = torch.stack([model(sample.flatten()).tanh() for sample in y.reshape(12, 2, 4)]).reshape(1, 3, 4, 3)
    expected = squared_pairs(a, b, scale=scale)
    torch.testing.assert_close(actual, expected)
    variables = (x, y, model.weight, model.bias, scale)
    actual_grads = torch.autograd.grad(actual.sum(), variables)
    expected_grads = torch.autograd.grad(expected.sum(), variables)
    for grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(grad, expected_grad)


def test_features_self_identity_arity_and_no_cross_call_cache(device):
    x = torch.randn(3, 4, device=device)
    calls, comparisons = [], []

    def encoder(flat):
        calls.append(flat.shape[0])
        return {'vector': flat * 2}

    def comparison(*features, **kwargs):
        comparisons.append((features, kwargs))
        return features[0]['vector'].sum()

    wrapped = tp.make_feature_pairwise(encoder, head=comparison)
    wrapped(x, flag='omitted')
    assert calls == [3] and len(comparisons[-1][0]) == 1
    wrapped(x, x, flag='explicit')
    a, b = comparisons[-1][0]
    assert a is b and a['vector'] is b['vector']
    assert calls == [3, 3] and comparisons[-1][1] == {'flag': 'explicit'}
    wrapped(x, x.view_as(x))
    assert calls == [3, 3, 3, 3]  # Equal storage is deliberately not object identity.
    before = wrapped(x).detach().clone()
    x.add_(1)
    after = wrapped(x)
    torch.testing.assert_close(after - before, x.new_tensor(24.))


@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed'])
def test_features_comparison_owns_modes_and_arguments(mode, device):
    x = torch.randn(3, 4, dtype=torch.float64, device=device)
    y = torch.randn_like(x)
    encoder = lambda batch: batch.square()
    custom = tp.make_pairwise(lambda a, b: (a - b).square().sum(), mode=mode, symmetric=True)
    wrapped = tp.make_feature_pairwise(encoder, head=custom)
    args = (x,) if mode == 'condensed' else (x, y)
    expected_args = (x.square(),) if mode == 'condensed' else (x.square(), y.square())
    torch.testing.assert_close(wrapped(*args), custom(*expected_args))
    if mode == 'all_pairs':
        direct = tp.make_feature_pairwise(encoder, head=tp.cdist)
        torch.testing.assert_close(direct(x, y, 'euclidean'), tp.cdist(x.square(), y.square(), metric='euclidean'))
    if mode == 'paired':
        direct = tp.make_feature_pairwise(encoder, head=tp.paired_distances)
        torch.testing.assert_close(direct(x, y, metric='euclidean'),
                                   tp.paired_distances(x.square(), y.square(), metric='euclidean'))
        with pytest.raises(TypeError):
            wrapped(x)


def test_features_nested_trees_chunks_and_comparison_outputs(device):
    x = torch.arange(24, dtype=torch.float64, device=device).reshape(2, 3, 4)

    def encoder(flat):
        return {'maps': (flat[:, :2], [flat[:, 2:].square()]), 'scalar': flat.sum(-1)}

    def head(tree):
        assert type(tree) is dict and type(tree['maps']) is tuple and type(tree['maps'][1]) is list
        assert tree['maps'][0].shape == (2, 3, 2)
        assert tree['scalar'].shape == (2, 3)
        return {'score': tree['maps'][0].sum(-1) + tree['maps'][1][0].sum(-1),
                'scalar': tree['scalar']}

    actual = tp.make_feature_pairwise(encoder, head=head, encode_chunk_size=4)(x)
    torch.testing.assert_close(actual['score'], x[..., :2].sum(-1) + x[..., 2:].square().sum(-1))
    torch.testing.assert_close(actual['scalar'], x.sum(-1))


def test_features_preserve_mode_grad_dtype_and_user_context(device):
    model = torch.nn.Linear(4, 2, device=device)
    x = torch.randn(3, 4, device=device, requires_grad=True)
    wrapped = tp.make_feature_pairwise(model, head=squared_pairs)
    model.train()
    assert wrapped(x).requires_grad and model.training
    model.eval()
    assert wrapped(x).requires_grad and not model.training
    with torch.no_grad():
        assert not wrapped(x).requires_grad and not model.training
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        features = tp.make_feature_pairwise(model, head=lambda f: f)(x)
        assert features.dtype == torch.bfloat16
    # Token IDs and encoder-produced dtypes are accepted without conversion.
    embedding = torch.nn.Embedding(8, 3, device=device)
    tokens = torch.tensor([[1, 2], [3, 4]], device=device)
    result = tp.make_feature_pairwise(embedding, head=lambda f: f)(tokens)
    torch.testing.assert_close(result, embedding(tokens))


def test_features_batch_dependent_chunking_is_explicit(device):
    x = torch.arange(12, dtype=torch.float64, device=device).reshape(4, 3)
    encoder = lambda samples: samples - samples.mean(0)
    complete = tp.make_feature_pairwise(encoder, head=lambda f: f)(x)
    chunked = tp.make_feature_pairwise(encoder, head=lambda f: f, encode_chunk_size=2)(x)
    assert not torch.equal(complete, chunked)


@pytest.mark.parametrize('shape', [(0, 4), (2, 0, 4), (0, 3, 4)])
def test_features_empty_policy_does_not_probe_encoder(shape, device):
    def forbidden(*args):
        raise AssertionError('empty collections must be rejected before calling user code')

    wrapped = tp.make_feature_pairwise(forbidden, head=forbidden)
    empty = torch.empty(shape, device=device)
    with pytest.raises(ValueError, match='empty feature collections'):
        wrapped(empty)
    with pytest.raises(ValueError, match='empty feature collections'):
        wrapped(torch.ones(2, 4, device=device), empty)


@pytest.mark.parametrize('output,exception', [
    (lambda x: x.sum(), ValueError), (lambda x: x[:1], ValueError),
    (lambda x: {'bad': 1}, TypeError), (lambda x: [], ValueError),
    (lambda x: {'bad': x[:1]}, ValueError)])
def test_features_reject_bad_encoder_outputs(output, exception, device):
    wrapped = tp.make_feature_pairwise(output, head=lambda f: f)
    with pytest.raises(exception):
        wrapped(torch.ones(3, 4, device=device))


@pytest.mark.parametrize('problem', ['structure', 'shape', 'dtype', 'keys', 'length'])
def test_features_chunk_outputs_must_be_consistent(problem, device):
    calls = []

    def encoder(x):
        calls.append(1)
        if problem == 'structure':
            return x if len(calls) == 1 else [x]
        if problem == 'shape':
            return x if len(calls) == 1 else x[:, :2]
        if problem == 'dtype':
            return x if len(calls) == 1 else x.double()
        if problem == 'keys':
            return {'a' if len(calls) == 1 else 'b': x}
        return [x] if len(calls) == 1 else [x, x]

    wrapped = tp.make_feature_pairwise(encoder, head=lambda f: f, encode_chunk_size=2)
    with pytest.raises((ValueError, TypeError), match='consistent across encoder chunks'):
        wrapped(torch.ones(3, 4, device=device))


@pytest.mark.parametrize('kwargs', [{'sample_ndim': 0}, {'sample_ndim': True},
    {'encode_chunk_size': 0}, {'encode_chunk_size': False}, {'encode_chunk_size': 1.5}])
def test_features_reject_bad_factory_options(kwargs):
    with pytest.raises(ValueError):
        tp.make_feature_pairwise(lambda f: f, head=squared_pairs, **kwargs)


def test_features_reject_bad_callables_and_inputs(device):
    with pytest.raises(TypeError):
        tp.make_feature_pairwise(None, head=squared_pairs)
    with pytest.raises(TypeError):
        tp.make_feature_pairwise(lambda f: f, None)
    wrapped = tp.make_feature_pairwise(lambda f: f, head=squared_pairs)
    with pytest.raises(TypeError):
        wrapped([1., 2.])
    with pytest.raises(ValueError):
        wrapped(torch.ones(4, device=device))
    with pytest.raises(TypeError):
        wrapped(torch.ones(2, 4, device=device).to_sparse())


def test_features_encoder_owns_input_extents_and_refreshes_parameters(device):
    parameter = torch.nn.Parameter(torch.tensor(2., device=device))
    x = torch.arange(12., device=device).reshape(3, 4)
    y = torch.arange(14., device=device).reshape(2, 7)
    calls = []

    def encoder(samples):
        calls.append(tuple(samples.shape))
        return samples.mean(-1, keepdim=True) * parameter

    wrapped = tp.make_feature_pairwise(encoder, head=squared_pairs)
    first = wrapped(x, y)
    assert calls == [(3, 4), (2, 7)]
    torch.testing.assert_close(first, squared_pairs(x.mean(-1, keepdim=True) * parameter,
                                                  y.mean(-1, keepdim=True) * parameter))
    with torch.no_grad():
        parameter.mul_(2)
    second = wrapped(x, y)
    torch.testing.assert_close(second, first * 4)
    assert len(calls) == 4


def test_features_explicit_preparation_single_and_streaming_reuse(device):
    x = torch.randn(3,4,device=device,dtype=torch.float64,requires_grad=True)
    y = torch.randn(5,4,device=device,dtype=x.dtype,requires_grad=True)
    calls = []
    model = torch.nn.Linear(4,2,device=device,dtype=x.dtype)
    def encoder(batch):
        calls.append(batch.shape[0])
        return model(batch).tanh()
    wrapped = tp.make_feature_pairwise(encoder, 'sqeuclidean')
    prepared = wrapped.prepare(x,y)
    assert calls == [3,5]
    dense = (prepared.features1[:,None]-prepared.features2[None]).square().sum(-1)
    A = tp.lazy(prepared, chunk_size=2)
    total = A.sum()+A.mean()
    torch.testing.assert_close(total,dense.sum()+dense.mean())
    torch.testing.assert_close(prepared(),dense)
    assert calls == [3,5]
    refs = torch.autograd.grad(dense.sum()+dense.mean(),(x,y,model.weight,model.bias),retain_graph=True)
    actual = torch.autograd.grad(total,(x,y,model.weight,model.bias))
    for value,ref in zip(actual,refs):
        torch.testing.assert_close(value,ref)
    wrapped(x,y)
    assert calls == [3,5,3,5]
    torch.testing.assert_close(wrapped(x[0],y[0],mode='single'), (model(x[0]).tanh()-model(y[0]).tanh()).square().sum())
    self_prepared = wrapped.prepare(x,x)
    assert self_prepared.features1 is self_prepared.features2
    with pytest.raises(ValueError,match='prepared collections'):
        tp.lazy(wrapped.prepare(x[0], y[0], mode='single'))
    custom = tp.make_pairwise(lambda a,b: (a-b).square().sum())
    encoded_custom = tp.make_feature_pairwise(encoder, head=custom)
    torch.testing.assert_close(tp.lazy(x, y, encoded_custom, chunk_size=2).sum(), (model(x).tanh()[:, None] - model(y).tanh()[None]).square().sum())
    opaque = tp.make_feature_pairwise(encoder, head=squared_pairs)
    with pytest.raises(TypeError,match='make_pairwise'):
        tp.lazy(x, y, opaque)


@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed'])
def test_features_metric_head_modes_and_override(mode, device):
    x = torch.randn(4, 3, dtype=torch.float64, device=device)
    y = torch.randn_like(x)
    wrapped = tp.make_feature_pairwise(torch.square, 'sqeuclidean', mode=mode)
    args = (x,) if mode == 'condensed' else (x, y)
    dense = tp.cdist(x.square(), x.square() if mode == 'condensed' else y.square(), metric='sqeuclidean')
    expected = dense if mode == 'all_pairs' else dense.diagonal() if mode == 'paired' else dense[torch.triu_indices(4, 4, 1, device=device).unbind()]
    torch.testing.assert_close(wrapped(*args), expected)
    torch.testing.assert_close(wrapped.prepare(*args)(), expected)
    torch.testing.assert_close(wrapped(x, y, mode='all_pairs'), tp.cdist(x.square(), y.square(), metric='sqeuclidean'))
    if mode != 'all_pairs':
        with pytest.raises(ValueError, match='all_pairs'):
            tp.lazy(x, None if mode == 'condensed' else y, wrapped)
        with pytest.raises(ValueError, match='all_pairs'):
            tp.lazy(wrapped.prepare(*args))
        torch.testing.assert_close(tp.lazy(x, y, wrapped, mode='all_pairs').to_dense(), tp.cdist(x.square(), y.square(), metric='sqeuclidean'))


def test_features_metric_streaming_encoder_gradients_and_kernel(device):
    x = torch.randn(4, 3, dtype=torch.float64, device=device)
    y = torch.randn(5, 3, dtype=x.dtype, device=device)
    encoder = torch.nn.Linear(3, 2, dtype=x.dtype, device=device)
    wrapped = tp.make_feature_pairwise(encoder, 'sqeuclidean')
    dense = tp.cdist(encoder(x), encoder(y), metric='sqeuclidean').sum(-1)
    streamed = tp.lazy(x, y, wrapped, chunk_size=2).sum(-1)
    torch.testing.assert_close(streamed, dense)
    actual = torch.autograd.grad(streamed.sum(), tuple(encoder.parameters()))
    expected = torch.autograd.grad(dense.sum(), tuple(encoder.parameters()))
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)
    kernel = tp.make_feature_pairwise(encoder, 'rbf', gamma=0.7)
    expected = tp.pairwise_kernels(encoder(x), encoder(y), kernel='rbf', gamma=0.7)
    torch.testing.assert_close(kernel(x, y), expected)
    torch.testing.assert_close(tp.lazy(x, y, kernel, chunk_size=2).to_dense(), expected)
    torch.testing.assert_close(tp.lazy(kernel.prepare(x, y), chunk_size=2).to_dense(), expected)


def test_features_head_argument_contract_and_lazy_trees(device):
    identity = lambda x: x
    for options in ({}, {'metric': 'euclidean', 'head': squared_pairs},
                    {'head': squared_pairs, 'gamma': 1.}):
        with pytest.raises(TypeError):
            tp.make_feature_pairwise(identity, **options)
    with pytest.raises(TypeError, match='head='):
        tp.make_feature_pairwise(identity, squared_pairs)
    x = torch.randn(3, 2, device=device)
    tree = tp.make_feature_pairwise(lambda x: {'vector': x}, 'euclidean')
    with pytest.raises(TypeError, match='one feature tensor'):
        tp.lazy(x, x, tree)
    with pytest.raises(TypeError, match='one feature tensor'):
        tp.lazy(tree.prepare(x, x))
    eager = tp.make_feature_pairwise(identity, head=squared_pairs)
    torch.testing.assert_close(eager(x), squared_pairs(x))
    with pytest.raises(TypeError, match='make_pairwise'):
        tp.lazy(eager.prepare(x))


def test_features_plain_head_mode_defaults_and_overrides(device):
    def head(x, y, *, mode):
        return tp.compare(x, y, metric='sqeuclidean', mode=mode)
    head.mode = 'paired'
    x = torch.randn(3, 2, device=device)
    y = torch.randn_like(x)
    inferred = tp.make_feature_pairwise(torch.square, head=head)
    expected = (x.square() - y.square()).square().sum(-1)
    torch.testing.assert_close(inferred(x, y), expected)
    explicit = tp.make_feature_pairwise(torch.square, head=head, mode='all_pairs')
    torch.testing.assert_close(explicit(x, y, mode='paired'), expected)
    torch.testing.assert_close(explicit.prepare(x, y)(mode='paired'), expected)
    torch.testing.assert_close(explicit(x, y), tp.cdist(x.square(), y.square(), metric='sqeuclidean'))


def test_features_two_towers_dense_lazy_self_and_gradients(device):
    query = torch.nn.Linear(3, 2, dtype=torch.float64, device=device).eval()
    reference = torch.nn.Linear(3, 2, dtype=torch.float64, device=device).train()
    with torch.no_grad():
        query.weight.fill_(0.5)
        query.bias.zero_()
        reference.weight.fill_(1.)
        reference.bias.fill_(2.)
    x = torch.arange(12, dtype=torch.float64, device=device).reshape(4, 3) / 10
    y = torch.randn(5, 3, dtype=x.dtype, device=device)
    wrapped = tp.make_feature_pairwise((query, reference), 'sqeuclidean', encode_chunk_size=2)
    assert wrapped.encoder == (query, reference)
    assert wrapped.query_encoder is query and wrapped.reference_encoder is reference
    expected = tp.cdist(query(x), reference(y), metric='sqeuclidean')
    torch.testing.assert_close(wrapped(x, y), expected)
    actual = tp.lazy(x, y, wrapped, chunk_size=2).sum()
    torch.testing.assert_close(actual, expected.sum())
    variables = tuple(query.parameters()) + tuple(reference.parameters())
    for value, target in zip(torch.autograd.grad(actual, variables), torch.autograd.grad(expected.sum(), variables)):
        torch.testing.assert_close(value, target)
    self_expected = tp.cdist(query(x), reference(x), metric='sqeuclidean')
    assert not torch.allclose(self_expected, self_expected.T)
    torch.testing.assert_close(wrapped(x), self_expected)
    torch.testing.assert_close(wrapped(x, x), self_expected)
    torch.testing.assert_close(tp.lazy(wrapped.prepare(x), chunk_size=2).to_dense(), self_expected)
    torch.testing.assert_close(wrapped.encode(x, tower='query'), query(x))
    torch.testing.assert_close(wrapped.encode(x, tower='reference'), reference(x))
    torch.testing.assert_close(wrapped.encode(x[0], single=True, tower='query'), query(x[0]))
    for tower in (None, 'other', 0):
        with pytest.raises(ValueError):
            wrapped.encode(x, tower=tower)
    single = tp.make_feature_pairwise(query, 'sqeuclidean')
    assert single.encoder is query
    with pytest.raises(ValueError):
        single.encode(x, tower='query')
    with pytest.raises(ValueError, match='condensed'):
        wrapped(x, mode='condensed')
    with pytest.raises(ValueError, match='condensed'):
        wrapped.prepare(x, mode='condensed')
    assert not query.training and reference.training


@pytest.mark.parametrize('encoder', [(lambda x: x,), (lambda x: x, None),
                                     (lambda x: x, lambda x: x, lambda x: x),
                                     [lambda x: x, lambda x: x]])
def test_features_reject_bad_towers(encoder):
    with pytest.raises(TypeError):
        tp.make_feature_pairwise(encoder, 'euclidean')
