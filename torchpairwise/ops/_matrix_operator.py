"""Common matrix consumers over caller-owned dense scalar matrices."""
import torch
from ._validation import _check_chunk_size
from ._pairwise_operator import _ScalarOperator


class MatrixOperator(_ScalarOperator):
    """Bind [...,N,M] values to the shared native matrix consumers.

    The caller already owns the dense matrix. This binding avoids duplicating
    it and provides the same algebra as PairwiseOperator; it does not remove
    the input matrix allocation. Mutations are visible on subsequent calls.
    """
    def __init__(self, matrix, *, chunk_size=None):
        if not isinstance(matrix, torch.Tensor):
            raise TypeError('matrix must be a tensor')
        if matrix.layout != torch.strided or matrix.ndim < 2:
            raise ValueError('matrix must be dense strided with shape [...,N,M]')
        _check_chunk_size(chunk_size, allow_none=True)
        self.matrix, self.chunk_size = matrix, chunk_size
        self._transposed = False

    @property
    def shape(self):
        shape = self.matrix.shape
        return torch.Size((*shape[:-2], shape[-1], shape[-2])) if self._transposed else shape

    @property
    def dtype(self):
        dtype = self.matrix.dtype
        return torch.float32 if dtype in (torch.float16, torch.bfloat16) and torch.is_autocast_enabled(self.device.type) else dtype

    @property
    def device(self):
        return self.matrix.device

    @property
    def T(self):
        result = MatrixOperator(self.matrix, chunk_size=self.chunk_size)
        result._transposed = not self._transposed
        return result

    def _native_evaluate(self, operation, **options):
        return torch.ops.torchpairwise.matrix_operations(self.matrix, operation,
            transposed=self._transposed, chunk_size=self.chunk_size, **options)



__all__ = ['MatrixOperator']
