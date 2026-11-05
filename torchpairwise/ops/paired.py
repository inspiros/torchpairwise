"""Corresponding-sample selectors over the shared native comparison catalog."""

def paired_distances(x1, x2, metric='euclidean', **kwargs):
    """Corresponding-sample distances/divergences/costs, across all sample modalities."""
    from ._comparison_dispatch import _select
    return _select(x1, x2, metric, 'paired', role='distance', **kwargs)


def paired_kernels(x1, x2, kernel='rbf', **kwargs):
    """Corresponding-sample kernels, including Gaussian-mixture overlap."""
    from ._comparison_dispatch import _select
    return _select(x1, x2, kernel, 'paired', role='kernel', **kwargs)


__all__ = ['paired_distances', 'paired_kernels']
