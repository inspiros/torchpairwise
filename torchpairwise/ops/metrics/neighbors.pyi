from typing import Optional
from torch import Tensor, Generator


def k_neighbors_mask(
    x1: Tensor, x2: Optional[Tensor] = None, k: int = 1,
    metric: str = 'euclidean', *, exclude_self: Optional[bool] = None,
    w: Optional[Tensor] = None, V: Optional[Tensor] = None,
    VI: Optional[Tensor] = None, p: Optional[float] = None,
    base: Optional[float] = None, shuffle: Optional[bool] = None,
    generator: Optional[Generator] = None,
    batched: Optional[bool] = None, feature_dim: Optional[int] = None,
    squared: Optional[bool] = None, x1_lengths: Optional[Tensor] = None,
    x2_lengths: Optional[Tensor] = None, chunk_size: Optional[int] = None,
    bin_positions: Optional[Tensor] = None, projections: Optional[Tensor] = None,
    regularization: Optional[float] = None, max_iter: Optional[int] = None,
    kernel: Optional[str] = None, length_scale: Optional[float] = None,
    alpha: Optional[float] = None,
) -> Tensor: ...


def radius_neighbors_mask(
    x1: Tensor, x2: Optional[Tensor] = None, epsilon: float = 0.,
    metric: str = 'euclidean', *, exclude_self: Optional[bool] = None,
    w: Optional[Tensor] = None, V: Optional[Tensor] = None,
    VI: Optional[Tensor] = None, p: Optional[float] = None,
    base: Optional[float] = None, shuffle: Optional[bool] = None,
    generator: Optional[Generator] = None,
    batched: Optional[bool] = None, feature_dim: Optional[int] = None,
    squared: Optional[bool] = None, x1_lengths: Optional[Tensor] = None,
    x2_lengths: Optional[Tensor] = None, chunk_size: Optional[int] = None,
    bin_positions: Optional[Tensor] = None, projections: Optional[Tensor] = None,
    regularization: Optional[float] = None, max_iter: Optional[int] = None,
    kernel: Optional[str] = None, length_scale: Optional[float] = None,
    alpha: Optional[float] = None,
) -> Tensor: ...

from typing import Tuple, Union
from .._pairwise_operator import LazyPairwise

def farthest_point_sampling(D: Union[LazyPairwise, Tensor], k: int, *,
                            start: int = 0) -> Tuple[Tensor, Tensor]: ...
def kmeans_plusplus(D: Union[LazyPairwise, Tensor], k: int, *,
                    generator: Optional[Generator] = None) -> Tensor: ...
