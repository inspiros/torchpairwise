from typing import Optional
from torch import Tensor

def mmd_statistics(x1: Tensor, x2: Optional[Tensor] = None, *, kernel: str = 'rbf', estimator: str = 'unbiased',
                   length_scale: float = 1., gamma: Optional[float] = None,
                   degree: Optional[int] = None, coef0: Optional[float] = None,
                   alpha: Optional[float] = None, nu: Optional[float] = None,
                   batched: Optional[bool] = None, feature_dim: int = -1,
                   x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                   chunk_size: Optional[int] = None) -> Tensor: ...

def paired_mmd_statistics(x1: Tensor, x2: Tensor, *, kernel: str = 'rbf', estimator: str = 'unbiased',
                   length_scale: float = 1., gamma: Optional[float] = None,
                   degree: Optional[int] = None, coef0: Optional[float] = None,
                   alpha: Optional[float] = None, nu: Optional[float] = None,
                   batched: Optional[bool] = None, feature_dim: int = -1,
                   x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                   chunk_size: Optional[int] = None) -> Tensor: ...

def condensed_mmd_statistics(x1: Tensor, *, kernel: str = 'rbf', estimator: str = 'unbiased',
                   length_scale: float = 1., gamma: Optional[float] = None,
                   degree: Optional[int] = None, coef0: Optional[float] = None,
                   alpha: Optional[float] = None, nu: Optional[float] = None,
                   batched: Optional[bool] = None, feature_dim: int = -1,
                   x1_lengths: Optional[Tensor] = None,
                   chunk_size: Optional[int] = None) -> Tensor: ...
