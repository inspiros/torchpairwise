"""Standalone-native structured algorithms, stable spectra and exported graphs."""
import math

import pytest
import torch
import torchpairwise as tp


@pytest.mark.parametrize('metric', ['log_euclidean', 'affine_invariant', 'stein'])
def test_native_covariance_repeated_spectra_and_modes(metric, device):
    a = torch.eye(2, dtype=torch.float64, device=device).repeat(3, 1, 1).requires_grad_()
    b = (a.detach() * 2).requires_grad_()
    operation = torch.ops.torchpairwise.covariance_comparisons
    value = operation(a, b, metric, 'paired', True)
    expected = torch.full_like(value, 2 * math.log(2)**2)
    if metric == 'stein':
        expected = torch.full_like(value, 2 * math.log(1.5 / 2**.5))
    torch.testing.assert_close(value, expected, atol=1e-7, rtol=1e-7)
    assert all(g.isfinite().all() for g in torch.autograd.grad(value.sum(), (a, b)))
    full = operation(a, None, metric, 'all_pairs', True)
    condensed = operation(a, None, metric, 'condensed', True)
    i, j = torch.triu_indices(3, 3, 1, device=device)
    torch.testing.assert_close(condensed, full[i, j])
    torch.testing.assert_close(full, torch.zeros_like(full), atol=0, rtol=0)


def test_native_gaussian_diagonal_values_and_backward(device):
    means = torch.tensor([[0., 1.], [3., 2.]], device=device, dtype=torch.float64, requires_grad=True)
    variances = torch.tensor([[1., 4.], [4., 9.]], device=device, dtype=means.dtype, requires_grad=True)
    other = means.detach() + 1
    target = variances.detach() * 2
    value = torch.ops.torchpairwise.gaussian_comparisons(means, variances.diag_embed(), other,
                                                        target.diag_embed(), 'all_pairs', True)
    expected = ((means[:, None] - other[None]).square().sum(-1)
                + (variances.sqrt()[:, None] - target.sqrt()[None]).square().sum(-1))
    torch.testing.assert_close(value, expected)
    for actual, reference in zip(torch.autograd.grad(value.sum(), (means, variances)),
                                 torch.autograd.grad(expected.sum(), (means, variances))):
        torch.testing.assert_close(actual, reference)


def test_native_exact_mixture_nonuniform_plan_and_weight_derivative(device):
    mean = torch.tensor([[[0.], [3.]]], device=device, dtype=torch.float64, requires_grad=True)
    other = torch.tensor([[[1.], [5.]]], device=device, dtype=mean.dtype)
    covariance = torch.ones(1, 2, 1, 1, device=device, dtype=mean.dtype, requires_grad=True)
    weights = torch.tensor([[1., 3.]], device=device, dtype=mean.dtype, requires_grad=True)
    target_weights = torch.tensor([[2., 2.]], device=device, dtype=mean.dtype)
    value = tp.gaussian_mixture_wasserstein_distances(mean, covariance, weights, other,
                                                      covariance.detach(), target_weights, squared=True)
    # Monotone 1-D coupling: masses 1/4 at (0,0), 1/4 at (1,0), 1/2 at (1,1).
    torch.testing.assert_close(value, torch.full_like(value, 3.25))
    gm, gc, gw = torch.autograd.grad(value.sum(), (mean, covariance, weights))
    torch.testing.assert_close(gm, torch.tensor([[[-.5], [-1.]]], device=device, dtype=mean.dtype))
    torch.testing.assert_close(gc, torch.zeros_like(gc), atol=1e-14, rtol=0)
    torch.testing.assert_close(gw, torch.tensor([[-.5625, .1875]], device=device, dtype=mean.dtype))


@pytest.mark.compiler
@pytest.mark.parametrize('family', ['gaussian', 'log_euclidean', 'affine_invariant', 'stein',
                                     'overlap', 'l2', 'cs', 'wasserstein', 'entropic', 'sinkhorn'])
@pytest.mark.parametrize('mode', ['all_pairs', 'paired', 'condensed'])
def test_native_structured_compile_export_roundtrip(family, mode, device, tmp_path):
    class Comparison(torch.nn.Module):
        def forward(self, means, covariance, weights, other_means, other_covariance, other_weights):
            second = None if mode == 'condensed' else other_means
            second_covariance = None if mode == 'condensed' else other_covariance
            if family == 'gaussian':
                return torch.ops.torchpairwise.gaussian_comparisons(
                    means[..., 0, :], covariance[..., 0, :, :],
                    None if second is None else second[..., 0, :],
                    None if second_covariance is None else second_covariance[..., 0, :, :], mode, True)
            if family in ('log_euclidean', 'affine_invariant', 'stein'):
                return torch.ops.torchpairwise.covariance_comparisons(
                    covariance[..., 0, :, :],
                    None if second_covariance is None else second_covariance[..., 0, :, :], family, mode, True)
            # Exercise the public selector/adapter too, including scalar
            # regularization validation, not only the typed native schema.
            names = {'overlap': 'gaussian_mixture_overlap_kernel',
                     'l2': 'gaussian_mixture_l2_distances',
                     'cs': 'gaussian_mixture_cauchy_schwarz_divergences',
                     'wasserstein': 'gaussian_mixture_wasserstein_distances',
                     'entropic': 'gaussian_mixture_entropic_transport_costs',
                     'sinkhorn': 'gaussian_mixture_sinkhorn_divergences'}
            options = {'chunk_size': 2}
            if family in ('l2', 'wasserstein'):
                options['squared'] = True
            if family in ('entropic', 'sinkhorn'):
                options.update(regularization=.4, max_iter=4)
            return tp.compare((means, covariance, weights),
                              None if second is None else
                              (second, second_covariance, other_weights),
                              metric=names[family], mode=mode, **options)

    def inputs(n, m):
        means = torch.randn(n, 2, 2, device=device, requires_grad=True)
        other = torch.randn(m, 2, 2, device=device, requires_grad=True)
        covariance = torch.eye(2, device=device).expand(n, 2, 2, 2).clone().requires_grad_()
        target = (torch.eye(2, device=device) * 2).expand(m, 2, 2, 2).clone().requires_grad_()
        weights = torch.ones(n, 2, device=device, requires_grad=True)
        other_weights = torch.ones(m, 2, device=device, requires_grad=True)
        return means, covariance, weights, other, target, other_weights

    n = torch.export.Dim('n', min=3 if mode == 'condensed' else 2)
    m = n if mode == 'paired' else torch.export.Dim('m', min=2)
    module = Comparison()
    args = inputs(3, 3 if mode == 'paired' else 4)
    exported = torch.export.export(module, args, dynamic_shapes=({0: n}, {0: n}, {0: n},
                                                               {0: m}, {0: m}, {0: m}))
    path = tmp_path / f'{family}_{mode}.pt2'
    torch.export.save(exported, path)
    restored = torch.export.load(path).module()
    args = inputs(4, 4 if mode == 'paired' else 5)
    expected, actual = module(*args), restored(*args)
    torch.testing.assert_close(actual, expected)
    # Include only operands that the selected family actually consumes.
    operands = [args[1]] if family in ('log_euclidean', 'affine_invariant', 'stein') else [args[0], args[1]]
    if family not in ('gaussian', 'log_euclidean', 'affine_invariant', 'stein'):
        operands.append(args[2])
    for a, b in zip(torch.autograd.grad(actual.sum(), operands),
                    torch.autograd.grad(expected.sum(), operands)):
        torch.testing.assert_close(a, b)
    torch.compiler.reset()
    compiled = torch.compile(module, backend='aot_eager', fullgraph=True, dynamic=True)
    actual, expected = compiled(*args), module(*args)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual.sum(), operands),
                    torch.autograd.grad(expected.sum(), operands)):
        torch.testing.assert_close(a, b)
    torch.compiler.reset()


@pytest.mark.parametrize('family', ['gaussian', 'log_euclidean', 'affine_invariant', 'stein',
                                     'overlap', 'l2', 'cs', 'wasserstein', 'entropic', 'sinkhorn'])
def test_native_structured_validates_nonempty_operands_before_empty_result(family, device):
    mean = torch.zeros(1, 2, device=device, dtype=torch.float64)
    covariance = torch.eye(2, device=device, dtype=mean.dtype)[None]
    if family == 'gaussian':
        def call(m, c, condensed=False):
            if condensed:
                return torch.ops.torchpairwise.gaussian_comparisons(m, c, None, None, 'condensed')
            return torch.ops.torchpairwise.gaussian_comparisons(mean[:0], covariance[:0], m, c)
        with pytest.raises(RuntimeError, match='finite'):
            call(mean + float('nan'), covariance)
        for condensed in (False, True):
            with pytest.raises(RuntimeError, match='positive semidefinite'):
                call(mean, -covariance, condensed)
        return
    if family in ('log_euclidean', 'affine_invariant', 'stein'):
        for condensed in (False, True):
            with pytest.raises(RuntimeError, match='strictly positive'):
                torch.ops.torchpairwise.covariance_comparisons(
                    covariance * 0 if condensed else covariance[:0],
                    None if condensed else covariance * 0, family,
                    'condensed' if condensed else 'all_pairs')
        return
    mean = mean[:, None].repeat(1, 2, 1)
    covariance = covariance[:, None].repeat(1, 2, 1, 1)
    weights = torch.ones(1, 2, device=device, dtype=mean.dtype)

    def call(m, c, w, condensed=False, lengths=None):
        return torch.ops.torchpairwise.mixture_comparisons(
            m if condensed else mean[:0], c if condensed else covariance[:0],
            w if condensed else weights[:0], None if condensed else m,
            None if condensed else c, None if condensed else w, family,
            'condensed' if condensed else 'all_pairs',
            lengths if condensed else None, None if condensed else lengths)

    for condensed in (False, True):
        for bad_mean, bad_covariance, bad_weight, message in (
                (mean + float('nan'), covariance, weights, 'finite'),
                (mean, -covariance, weights, 'positive'),
                (mean, covariance, weights * 0, 'positive total'),
                (mean, covariance, weights * float('inf'), 'finite')):
            with pytest.raises(RuntimeError, match=message):
                call(bad_mean, bad_covariance, bad_weight, condensed)
        with pytest.raises(RuntimeError, match='between 1'):
            call(mean, covariance, weights, condensed, torch.zeros(1, device=device, dtype=torch.int64))
    # Invalid padding remains ignored even though the output contains no pairs.
    padded_mean, padded_covariance = mean.clone(), covariance.clone()
    padded_mean[:, 1] = float('nan')
    padded_covariance[:, 1] = float('nan')
    output = call(padded_mean, padded_covariance, weights,
                  lengths=torch.ones(1, device=device, dtype=torch.int64))
    assert output.shape == (0, 1)


@pytest.mark.compiler
@pytest.mark.parametrize('n', [0, 1, 2])
def test_structured_fixed_condensed_boundary_exports(n, device):
    class Comparison(torch.nn.Module):
        def forward(self, means, covariance, weights):
            return (tp.condensed_gaussian_wasserstein_distances(means, covariance),
                    tp.condensed_log_euclidean_distances(covariance),
                    tp.condensed_gaussian_mixture_l2_distances(
                        means.unsqueeze(-2), covariance.unsqueeze(-3), weights))
    means = torch.rand(n, 2, device=device, requires_grad=True)
    covariance = torch.eye(2, device=device).repeat(n, 1, 1).requires_grad_()
    weights = torch.ones(n, 1, device=device, requires_grad=True)
    model = Comparison()
    actual = torch.export.export(model, (means, covariance, weights)).module()(means, covariance, weights)
    expected = model(means, covariance, weights)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    for a, b in zip(torch.autograd.grad(sum(t.sum() for t in actual), (means, covariance, weights)),
                    torch.autograd.grad(sum(t.sum() for t in expected), (means, covariance, weights))):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize('family', ['gaussian', 'log_euclidean', 'affine_invariant', 'stein'])
@pytest.mark.parametrize('mode', ['all_pairs', 'paired'])
def test_structured_zero_broadcast_batch_preserves_raw_validation(family, mode, device):
    mean = torch.zeros(1, 2, 2, device=device, dtype=torch.float64)
    covariance = torch.eye(2, device=device, dtype=mean.dtype).expand(1, 2, 2, 2).clone()
    empty_mean, empty_covariance = mean[:0], covariance[:0]

    def call(m, c, reverse=False, python=False):
        left_mean, right_mean = (m, empty_mean) if reverse else (empty_mean, m)
        left_cov, right_cov = (c, empty_covariance) if reverse else (empty_covariance, c)
        if family == 'gaussian':
            if python:
                fn = tp.paired_gaussian_wasserstein_distances if mode == 'paired' else tp.gaussian_wasserstein_distances
                return fn(left_mean, left_cov, right_mean, right_cov)
            return torch.ops.torchpairwise.gaussian_comparisons(left_mean, left_cov, right_mean, right_cov, mode)
        if python:
            fn = getattr(tp, ('paired_' if mode == 'paired' else '') + family + '_distances')
            return fn(left_cov, right_cov)
        return torch.ops.torchpairwise.covariance_comparisons(left_cov, right_cov, family, mode)

    expected_shape = (0, 2) if mode == 'paired' else (0, 2, 2)
    for reverse in (False, True):
        assert call(mean, covariance, reverse).shape == expected_shape
        assert call(mean, covariance, reverse, python=True).shape == expected_shape
        bad_values = [(mean, covariance * float('nan'), 'finite'),
                      (mean, covariance * 0, 'strictly positive' if family != 'gaussian' else None),
                      (mean, -covariance, 'positive semidefinite' if family == 'gaussian' else 'strictly positive')]
        asymmetric = covariance.clone()
        asymmetric[..., 0, 1] = .1
        bad_values.append((mean, asymmetric, 'symmetric'))
        if family == 'gaussian':
            bad_values.append((mean + float('nan'), covariance, 'finite'))
        for bad_mean, bad_cov, message in bad_values:
            if message is None:  # Singular PSD covariances are valid Gaussians.
                assert call(bad_mean, bad_cov, reverse).shape == expected_shape
                continue
            with pytest.raises(RuntimeError, match=message):
                call(bad_mean, bad_cov, reverse)
            with pytest.raises(ValueError, match=message):
                call(bad_mean, bad_cov, reverse, python=True)

    if family == 'gaussian':
        # Within one collection, the covariance or mean batch can independently
        # erase its partner before collection-to-collection normalization.
        with pytest.raises(RuntimeError, match='positive semidefinite'):
            torch.ops.torchpairwise.gaussian_comparisons(empty_mean, -covariance, None, None, 'condensed')
        with pytest.raises(RuntimeError, match='finite'):
            torch.ops.torchpairwise.gaussian_comparisons(mean + float('nan'), empty_covariance)


@pytest.mark.compiler
@pytest.mark.parametrize('family', ['gaussian', 'log_euclidean', 'affine_invariant', 'stein'])
def test_structured_zero_batch_validation_export_and_compile(family, device, tmp_path):
    class Comparison(torch.nn.Module):
        def forward(self, empty_mean, empty_covariance, mean, covariance):
            if family == 'gaussian':
                return tp.gaussian_wasserstein_distances(empty_mean, empty_covariance, mean, covariance)
            return getattr(tp, family + '_distances')(empty_covariance, covariance)

    mean = torch.zeros(1, 2, 2, device=device, dtype=torch.float64, requires_grad=True)
    covariance = torch.eye(2, device=device, dtype=mean.dtype).expand(1, 2, 2, 2).clone().requires_grad_()
    args = (mean[:0], covariance[:0], mean, covariance)
    model = Comparison()
    exported = torch.export.export(model, args)
    # Export may retain the public family selector as one opaque node instead
    # of tracing its internal validation token. Either boundary must still run
    # native raw validation, even when the comparison produces no elements.
    native_boundaries = {torch.ops.torchpairwise._structured_validate_raw.default,
                         torch.ops.torchpairwise.gaussian_comparisons.default,
                         torch.ops.torchpairwise.covariance_comparisons.default}
    assert any(node.target in native_boundaries for node in exported.graph.nodes)
    path = tmp_path / f'{family}_zero_batch.pt2'
    torch.export.save(exported, path)
    restored = torch.export.load(path).module()
    torch.testing.assert_close(restored(*args), model(*args))
    expected = model(*args)
    actual = restored(*args)
    operands = (mean, covariance) if family == 'gaussian' else (covariance,)
    for a, b in zip(torch.autograd.grad(actual.sum(), operands),
                    torch.autograd.grad(expected.sum(), operands)):
        torch.testing.assert_close(a, b)
    bad = (args[0], args[1], mean, -covariance)
    message = 'positive semidefinite' if family == 'gaussian' else 'strictly positive'
    with pytest.raises(RuntimeError, match=message):
        restored(*bad)
    torch.compiler.reset()
    compiled = torch.compile(model, backend='aot_eager', fullgraph=True)
    torch.testing.assert_close(compiled(*args), model(*args))
    with pytest.raises(RuntimeError, match=message):
        compiled(*bad)
    torch.compiler.reset()


def _poison_cuda_pool():
    # Stale NaN bits in reused blocks made CUDA eigh return NaN eigenvalues for
    # exactly zero matrices; previously only allocation history exposed this.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    blocks = [torch.full((1 << 17,), math.nan, device='cuda', dtype=torch.float64) for _ in range(4)]
    del blocks


@pytest.mark.cuda
def test_zero_matrices_survive_stale_cuda_workspace():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from torchpairwise import linalg
    mean = torch.tensor([[0., 0.], [3., 4.]], device='cuda', dtype=torch.float64)
    zero = torch.zeros(2, 2, 2, device='cuda', dtype=mean.dtype)
    _poison_cuda_pool()
    value = torch.ops.torchpairwise.gaussian_comparisons(mean, zero)
    torch.testing.assert_close(value, torch.cdist(mean, mean))
    spd = torch.tensor([[2., .5], [.5, 1.]], device='cuda', dtype=mean.dtype).expand(2, 2, 2)
    for metric in ('affine_invariant', 'stein'):
        _poison_cuda_pool()
        value = torch.ops.torchpairwise.covariance_comparisons(spd, None, metric)
        torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)
    values = torch.rand(3, 2, device='cuda', dtype=mean.dtype)
    _poison_cuda_pool()
    action = linalg.expm_multiply(torch.zeros(3, 3, device='cuda', dtype=mean.dtype), values)
    torch.testing.assert_close(action.action, values)
    _poison_cuda_pool()
    result = linalg.eigh(torch.zeros(4, 4, device='cuda', dtype=mean.dtype), k=1, atol=1e-12)
    assert bool(result.converged.all()) and result.eigenvalues.eq(0).all()
