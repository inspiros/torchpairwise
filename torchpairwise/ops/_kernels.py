"""Python exception adapters; comparison algorithms live in C++."""
import torch
from ._batching import is_finite_real

def validate_radial_parameters(spec, kwargs):
    for key in ('length_scale', 'alpha') if spec.op == 'rational_quadratic_kernel' else ('length_scale',):
        value = kwargs.get(key, 1.)
        if type(value) not in (int, float):
            raise TypeError(f'{key} must be a real scalar')
        if not is_finite_real(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if spec.op == 'matern_kernel':
        nu = kwargs.get('nu', 1.5)
        if type(nu) not in (int, float):
            raise TypeError('nu must be a real scalar')
        if nu not in (.5, 1.5, 2.5):
            raise ValueError('nu must be one of 0.5, 1.5, or 2.5')
