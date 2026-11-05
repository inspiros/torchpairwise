"""Python exception adapters; comparison algorithms live in C++."""
import math
import torch

def validate_bhattacharyya_inputs(x, y):
    """Validate values after the shared shape, dtype, and layout checks."""
    from ._batching import validate_values

    for value in (x, y):
        validate_values((torch.isfinite(value) & (value >= 0)).all(),
                        'Bhattacharyya inputs must be finite and non-negative')
        mass = value.sum(-1)
        validate_values((torch.isfinite(mass) & (mass > 0)).all(),
                        'Bhattacharyya inputs must have positive finite row mass')


def validate_aitchison_inputs(x, y):
    """Aitchison requires strictly positive entries, without a pseudocount."""
    from ._batching import validate_values

    for value in (x, y):
        validate_values((torch.isfinite(value) & (value > 0)).all(),
                        'Aitchison inputs must be finite and strictly positive')
