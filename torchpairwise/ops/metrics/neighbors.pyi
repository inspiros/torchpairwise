from typing import Optional

from torch import Tensor


# noinspection PyPep8Naming
def k_neighbors_mask(x1: Tensor,
                     x2: Optional[Tensor] = None,
                     k: int = 0,
                     metric: str = "euclidean",
                     **kwargs) -> Tensor: ...


# noinspection PyPep8Naming
def radius_neighbors_mask(x1: Tensor,
                          x2: Optional[Tensor] = None,
                          epsilon: float = 0.,
                          metric: str = "euclidean",
                          **kwargs) -> Tensor: ...
