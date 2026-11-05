from typing import Any, Optional
from torch import Tensor
from ._pairwise_operator import PairwiseOperator
from ._comparison_dispatch import Collection

class KernelOperator(PairwiseOperator):
    x1: Collection
    x2: Optional[Collection]
    kernel: str
    chunk_size: Optional[int]
    kernel_kwargs: dict[str, Any]
    def __init__(self, x1: Collection, x2: Optional[Collection] = None, *, kernel: str = 'rbf', chunk_size: Optional[int] = None, **kernel_kwargs: Any) -> None: ...
    @property
    def T(self) -> KernelOperator: ...
    def displacement(self, *, normalize: bool = False) -> Tensor: ...
    def residual_outer_product(self, *, normalize: bool = False, center: bool = False) -> Tensor: ...
