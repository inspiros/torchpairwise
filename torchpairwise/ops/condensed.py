"""Strict upper-triangle selectors over the shared native comparison catalog."""

def condensed_distances(x, metric='minkowski', **kwargs):
    """Strict upper-triangle distances/divergences/costs, across all sample modalities."""
    from ._comparison_dispatch import _select
    return _select(x, None, metric, 'condensed', role='distance', **kwargs)


def condensed_kernels(x, kernel='rbf', **kwargs):
    """Strict upper-triangle kernels, including Gaussian-mixture overlap."""
    from ._comparison_dispatch import _select
    return _select(x, None, kernel, 'condensed', role='kernel', **kwargs)


__all__ = ['condensed_distances', 'condensed_kernels']
