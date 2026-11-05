"""All-family distance/divergence and kernel selectors sharing native catalogs."""

def cdist(x1, x2=None, metric='minkowski', **kwargs):
    """All-pairs distances/divergences/costs, matching native cdist coverage/options."""
    from ._comparison_dispatch import _select
    return _select(x1, x2, metric, 'all_pairs', role='distance', selector='cdist', **kwargs)


def pairwise_distances(x1, x2=None, metric='euclidean', **kwargs):
    """All-pairs distance selector; same catalog as cdist, Euclidean default."""
    return cdist(x1, x2, metric, **kwargs)


def pairwise_kernels(x1, x2=None, kernel='rbf', **kwargs):
    """All-pairs kernel selector over every catalog kernel, including mixtures."""
    from ._comparison_dispatch import _select
    return _select(x1, x2, kernel, 'all_pairs', role='kernel', **kwargs)


def pdist(input, metric='minkowski', **kwargs):
    """Strict upper-triangle distances for every supported symmetric comparison."""
    from ._comparison_dispatch import _select
    return _select(input, None, metric, 'condensed', role='distance', selector='pdist', **kwargs)


__all__ = ['cdist', 'pdist', 'pairwise_distances', 'pairwise_kernels']
