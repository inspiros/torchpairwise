"""Capabilities shared by the Python dispatchers, shape validation and tests."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    op: str
    parameters: tuple = ()
    event_ndim: int = 2  # sample axis plus feature/point axes
    symmetric: bool = True
    differentiable: bool = True
    boolean: bool = False
    native: bool = True  # whether this exact named op has a torch.ops schema; all builtins have native family APIs


DISTANCES = {
    name: Metric(name + '_distances') for name in (
        'euclidean', 'haversine', 'manhattan', 'cosine', 'sqeuclidean',
        'correlation', 'chebyshev', 'braycurtis', 'canberra',
    )
}
DISTANCES.update({
    'cosine': Metric('cosine_distances', ('w',)),
    'correlation': Metric('correlation_distances', ('w',)),
    'kl': Metric('kl_divergences', ('base',), symmetric=False, native=False),
    'symmetric_kl': Metric('symmetric_kl_divergences', ('base',), native=False),
    'bhattacharyya': Metric('bhattacharyya_distances', native=False),
    'aitchison': Metric('aitchison_distances', native=False),
    'energy': Metric('energy_distances', ('squared', 'x1_lengths', 'x2_lengths', 'chunk_size'), 3, native=False),
    'mmd': Metric('mmd_distances', ('squared', 'kernel', 'length_scale', 'gamma', 'degree', 'coef0', 'alpha', 'nu', 'x1_lengths', 'x2_lengths', 'chunk_size'), 3, native=False),
    'directed_hausdorff': Metric('directed_hausdorff_distances',
                               ('shuffle', 'generator', 'x1_lengths', 'x2_lengths', 'chunk_size'), 3, False),
    'hausdorff': Metric('hausdorff_distances',
                       ('shuffle', 'generator', 'x1_lengths', 'x2_lengths', 'chunk_size'), 3, native=False),
    'chamfer': Metric('chamfer_distances', ('squared', 'x1_lengths', 'x2_lengths', 'chunk_size'), 3, native=False),
    'wasserstein': Metric('wasserstein_distances', ('p', 'x1_lengths', 'x2_lengths', 'chunk_size'), 3, native=False),
    'sliced_wasserstein': Metric('sliced_wasserstein_distances',
                                ('p', 'projections', 'x1_lengths', 'x2_lengths', 'chunk_size'), 3, native=False),
    'sinkhorn': Metric('sinkhorn_divergences',
                       ('p', 'regularization', 'max_iter', 'x1_lengths', 'x2_lengths', 'chunk_size'), 3, native=False),
    'minkowski': Metric('minkowski_distances', ('p',)),
    'wminkowski': Metric('wminkowski_distances', ('p', 'w')),
    'seuclidean': Metric('seuclidean_distances', ('V',)),
    'mahalanobis': Metric('mahalanobis_distances', ('VI',)),
    'jensenshannon': Metric('jensenshannon_distances', ('base',)),
    'snr': Metric('snr_distances', symmetric=False),
    'hellinger': Metric('hellinger_distances', native=False),
    'nan_euclidean': Metric('nan_euclidean_distances', ('squared',), native=False),
    'wasserstein_1d': Metric('wasserstein_1d_distances', ('bin_positions',), native=False),
    'hamming': Metric('hamming_distances', differentiable=False),
})
DISTANCES.update({name: Metric(name + '_distances', differentiable=False, boolean=True)
                  for name in ('jaccard', 'kulsinski', 'kulczynski1', 'yule', 'dice',
                               'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath')})
# Standard SciPy feature weighting; keep native schemas unchanged when omitted.
for name in ('euclidean', 'manhattan', 'sqeuclidean', 'chebyshev', 'braycurtis',
             'canberra', 'minkowski', 'hamming', 'jaccard', 'kulsinski', 'kulczynski1',
             'yule', 'dice', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath'):
    spec = DISTANCES[name]
    DISTANCES[name] = Metric(spec.op, (*spec.parameters, 'w'), spec.event_ndim,
                            spec.symmetric, spec.differentiable, spec.boolean, spec.native)
for alias, name in {'l1': 'manhattan', 'cityblock': 'manhattan', 'l2': 'euclidean',
                    'lp': 'minkowski', 'linf': 'chebyshev'}.items():
    DISTANCES[alias] = DISTANCES[name]

DISTANCES.update({
    'total_variation': Metric('total_variation_distances', (), symmetric=True, native=False),
    'fisher_rao': Metric('fisher_rao_distances', (), symmetric=True, native=False),
    'renyi': Metric('renyi_divergences', ('alpha',), symmetric=False, native=False),
    'pearson_chi2': Metric('pearson_chi2_divergences', (), symmetric=False, native=False),
    'neyman_chi2': Metric('neyman_chi2_divergences', (), symmetric=False, native=False),
    'triangular_discrimination': Metric('triangular_discrimination_distances', (), symmetric=True, native=False),
    'itakura_saito': Metric('itakura_saito_divergences', (), symmetric=False, native=False),
    'angular': Metric('angular_distances', native=False),
    'lorentzian': Metric('lorentzian_distances', native=False),
    'clark': Metric('clark_distances', native=False),
    'soergel': Metric('soergel_distances', native=False),
    'ruzicka': Metric('ruzicka_distances', native=False),
})
DISTANCES['continuous_jaccard'] = DISTANCES['ruzicka']
PROBABILITY_EXTRA_OPS = ('total_variation_distances', 'fisher_rao_distances', 'renyi_divergences', 'pearson_chi2_divergences', 'neyman_chi2_divergences', 'triangular_discrimination_distances', 'itakura_saito_divergences')
VECTOR_EXTRA_OPS = ('angular_distances', 'lorentzian_distances', 'clark_distances', 'soergel_distances', 'ruzicka_distances')

KERNELS = {
    'bhattacharyya': Metric('bhattacharyya_kernel', native=False),
    'linear': Metric('linear_kernel'),
    'polynomial': Metric('polynomial_kernel', ('degree', 'gamma', 'coef0')),
    'sigmoid': Metric('sigmoid_kernel', ('gamma', 'coef0')),
    'rbf': Metric('rbf_kernel', ('gamma',)),
    'laplacian': Metric('laplacian_kernel', ('gamma',)),
    'cosine': Metric('cosine_similarity'),
    'additive_chi2': Metric('additive_chi2_kernel'),
    'chi2': Metric('chi2_kernel', ('gamma',)),
    'rational_quadratic': Metric('rational_quadratic_kernel', ('length_scale', 'alpha'), native=False),
    'matern': Metric('matern_kernel', ('length_scale', 'nu'), native=False),
    'histogram_intersection': Metric('histogram_intersection_kernel', native=False),
}
# Native ATen comparison layer: public schemas own all comparison modes.
NATIVE_POINTSET_OPS = ('chamfer_distances', 'hausdorff_distances', 'directed_hausdorff_distances')

NATIVE_VECTOR_OPS = ('aitchison_distances', 'angular_distances', 'bhattacharyya_distances', 'bhattacharyya_kernel', 'clark_distances', 'fisher_rao_distances', 'hellinger_distances', 'histogram_intersection_kernel', 'itakura_saito_divergences', 'kl_divergences', 'lorentzian_distances', 'matern_kernel', 'nan_euclidean_distances', 'neyman_chi2_divergences', 'pearson_chi2_divergences', 'rational_quadratic_kernel', 'renyi_divergences', 'ruzicka_distances', 'soergel_distances', 'symmetric_kl_divergences', 'total_variation_distances', 'triangular_discrimination_distances', 'wasserstein_1d_distances')
for registry in (DISTANCES, KERNELS):
    for key, spec in tuple(registry.items()):
        if spec.op in NATIVE_VECTOR_OPS:
            registry[key] = Metric(spec.op, spec.parameters, spec.event_ndim,
                                   spec.symmetric, spec.differentiable, spec.boolean, True)

OPS = {spec.op: spec for spec in (*DISTANCES.values(), *KERNELS.values())}
# Native aliases have their own schemas, but share canonical capabilities.
for alias in ('l1', 'l2', 'lp', 'linf', 'cityblock'):
    OPS[alias + '_distances'] = DISTANCES[alias]


def resolve(name, registry):
    if not isinstance(name, str) or name not in registry:
        raise ValueError(f'Unknown metric/kernel {name!r}; supported names: {", ".join(sorted(registry))}')
    return registry[name]
