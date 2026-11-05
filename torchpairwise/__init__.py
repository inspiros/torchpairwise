from .extension import _HAS_OPS, _assert_has_ops, has_ops, with_cuda, cuda_version, get_include, library_path

_assert_has_ops()

from . import ops
from . import backends
from .backends import backend
from .ops import cpdist, metrics, paired, condensed, neighbors, pairwise, masks
from ._ops import _ops
from ._version import __version__
from .ops import *

from . import _integration
from . import linalg, optimize

__all__ = list(ops.__all__) + ['backend', 'backends', 'linalg', 'optimize', 'has_ops', 'with_cuda', 'cuda_version',
                               'get_include', 'library_path',
                               'extension', 'ops', 'cpdist', 'metrics', 'paired', 'condensed', 'masks',
                               'neighbors', 'pairwise']
