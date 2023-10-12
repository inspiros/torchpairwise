from torch._ops import _OpNamespace

from .extension import _assert_has_ops

__all__ = ['_ops']


class _TorchPairwiseOpNameSpace(_OpNamespace):

    def __init__(self):
        super(_TorchPairwiseOpNameSpace, self).__init__('torchpairwise')

    def __getattr__(self, op_name):
        _assert_has_ops()
        return super(_TorchPairwiseOpNameSpace, self).__getattr__(op_name)


_ops = _TorchPairwiseOpNameSpace()
