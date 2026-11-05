from typing import Optional
import torch
from torch import Tensor
from ._pairwise_operator import _ScalarOperator

class MatrixOperator(_ScalarOperator):
    matrix: Tensor
    chunk_size: Optional[int]
    def __init__(self, matrix: Tensor, *, chunk_size: Optional[int] = None) -> None: ...
    @property
    def shape(self) -> torch.Size: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    @property
    def T(self) -> MatrixOperator: ...
