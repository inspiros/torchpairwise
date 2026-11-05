"""Python exception adapters; comparison algorithms live in C++."""
import math
import torch

def validate_kl_inputs(x, y):
    """Validate histogram values after the public shape/dtype/layout checks."""
    from ._batching import validate_values

    for value in (x, y):
        validate_values((torch.isfinite(value) & (value >= 0)).all(),
                        'KL inputs must be finite and non-negative')
        mass = value.sum(-1)
        validate_values((torch.isfinite(mass) & (mass > 0)).all(),
                        'KL inputs must have positive finite row mass')
