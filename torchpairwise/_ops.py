"""Guarded access to the native operator namespace."""
import torch
from .extension import _assert_has_ops

__all__ = ['_ops']


class _TorchPairwiseOpNameSpace:
    def __getattr__(self, op_name):
        _assert_has_ops()
        return getattr(torch.ops.torchpairwise, op_name)


_ops = _TorchPairwiseOpNameSpace()
