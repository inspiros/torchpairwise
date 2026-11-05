"""Python exception adapters; comparison algorithms live in C++."""
import math
import torch

def validate_vector_extra_inputs(spec, x, y):
    """Value/layout checks following the shared public shape validation."""
    from ._batching import validate_values

    histogram = spec.op in ('clark_distances', 'soergel_distances',
                            'ruzicka_distances', 'continuous_jaccard_distances')
    for value in (x, y):
        if value.layout != torch.strided:
            raise TypeError('Inputs must be dense strided tensors')
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError('These distances require float32 or float64 inputs outside autocast')
        condition = torch.isfinite(value)
        if histogram:
            condition = condition & (value >= 0)
        validate_values(condition.all(), 'Histogram inputs must be finite and non-negative'
                        if histogram else 'Vector inputs must be finite')
        if spec.op == 'angular_distances':
            validate_values((value.abs().amax(-1) > 0).all(),
                            'Angular inputs must have nonzero rows')
