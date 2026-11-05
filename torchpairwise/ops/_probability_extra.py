"""Python exception adapters; comparison algorithms live in C++."""
import torch
from ._batching import is_finite_real

def validate_probability_extra(spec, x, y, kwargs):
    """Value/layout/dtype checks, called after the shared structural checks."""
    from ._batching import validate_values

    if x.layout != torch.strided or y.layout != torch.strided:
        raise TypeError('Probability comparisons require dense strided inputs')
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError('Probability comparisons require float32 or float64 inputs outside autocast')
    raw_positive = spec.op == 'itakura_saito_divergences'
    for value in (x, y):
        if raw_positive:
            validate_values((torch.isfinite(value) & (value > 0)).all(),
                            'Itakura-Saito inputs must be finite and strictly positive')
        else:
            validate_values((torch.isfinite(value) & (value >= 0)).all(),
                            'Probability inputs must be finite and non-negative')
            mass = value.sum(-1)
            validate_values((torch.isfinite(mass) & (mass > 0)).all(),
                            'Probability inputs must have positive finite row mass')
    if spec.op == 'renyi_divergences':
        alpha = kwargs.get('alpha', 1.)
        valid = not isinstance(alpha, bool) and isinstance(alpha, (int, float))
        if valid:
            try:
                valid = is_finite_real(alpha) and alpha > 0
            except OverflowError:
                valid = False
        if not valid:
            raise ValueError('Renyi alpha must be a finite scalar greater than zero')
