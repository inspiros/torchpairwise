from .extension import _HAS_OPS, _assert_has_ops, has_ops, with_cuda, cuda_version

_assert_has_ops()

from . import ops
from ._ops import _ops
from ._version import __version__
from .ops import *
