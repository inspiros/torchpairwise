"""Validation for native point-set distribution/transport comparisons."""
import torch


def _validate_transport(spec, x, y, kwargs):
    # Established Python errors precede shared native execution in all modes.
    if spec.op in ('energy_distances', 'mmd_distances'):
        from ._distribution import validate_distribution
        validate_distribution(spec, x, y, kwargs)
    elif spec.op == 'sinkhorn_divergences':
        from ._sinkhorn import validate_sinkhorn
        validate_sinkhorn(x, y, kwargs)
    elif spec.op in ('wasserstein_distances', 'sliced_wasserstein_distances'):
        # Parameter/type/domain checks are shared below without solving pairs.
        from ._batching import validate_values
        if spec.op == 'sliced_wasserstein_distances':
            from ._sliced import _validate
            _validate(x, y, kwargs['projections'], kwargs.get('p', 2))
        p = kwargs.get('p', 1)
        if isinstance(p, bool) or not isinstance(p, (int, float)) or p not in (1, 2):
            raise ValueError('Wasserstein p must be 1 or 2')
        for value, lengths in ((x, kwargs.get('x1_lengths')), (y, kwargs.get('x2_lengths'))):
            valid = torch.ones(value.shape[:-1], device=value.device, dtype=torch.bool)
            if lengths is not None:
                valid = torch.arange(value.shape[-2], device=value.device) < lengths.unsqueeze(-1)
            validate_values((value.isfinite() | ~valid[..., None]).all(),
                            'Wasserstein valid point coordinates must be finite')
