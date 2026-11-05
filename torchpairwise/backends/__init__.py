"""Optional eager evaluation backends for lazy pairwise operators."""
from contextlib import contextmanager
from contextvars import ContextVar
from weakref import WeakSet

import torch

from . import keops

_selection = ContextVar('torchpairwise_backend', default='native')
# Overrides can be nested inside expressions; skip graph walks otherwise.
_overrides = WeakSet()


def _validate(name):
    if name not in ('native', 'keops', 'auto'):
        raise ValueError(f'Unknown backend {name!r}; expected native, keops or auto')
    if name == 'keops' and not torch.compiler.is_compiling() and not keops.available():
        raise ImportError('KeOps is unavailable; pip install torchpairwise[keops]')


@contextmanager
def backend(name):
    """Select native, keops or auto evaluation within a nestable context."""
    _validate(name)
    token = _selection.set(name)
    try:
        yield
    finally:
        _selection.reset(token)


def _set_backend(leaf, name):
    if name is not None and not torch.compiler.is_compiling():
        _overrides.add(leaf)
    leaf._backend = name
    return leaf


def _resolve(source, selected):
    names = set()

    def visit(value):
        if hasattr(value, '_operation'):
            op = value._operation
            if op not in ('row_values', 'column_values', 'row_index', 'column_index'):
                visit(value._left)
            if op == 'where':
                for branch in value._right:
                    visit(branch)
            elif op in ('add', 'sub', 'mul', 'div', 'pow', 'minimum', 'maximum',
                         'clamp_min', 'clamp_max', 'lt', 'le', 'gt', 'ge', 'eq',
                         'ne', 'logical_and', 'logical_or'):
                visit(value._right)
        else:
            name = getattr(value, '_backend', None)
            if name is not None:
                names.add(name)

    visit(source)
    if 'native' in names or len(names) > 1:
        return 'native'
    return next(iter(names)) if names else selected


def _dispatch(source, operation, options):
    if (torch.compiler.is_compiling() or torch.jit.is_scripting()
            or torch.jit.is_tracing() or torch._C._are_functorch_transforms_active()):
        return NotImplemented
    selected = _selection.get()
    if selected == 'native' and not _overrides:
        return NotImplemented
    selected = _resolve(source, selected)
    if selected == 'native' or not keops.available():
        return NotImplemented
    return keops._lower(source, operation, options, auto=selected == 'auto')


__all__ = ['backend', 'keops']
