from typing import Optional
from torch import Tensor

def image_maps(x1: Tensor, x2: Optional[Tensor] = None, *, comparison: str = 'squared_error', mode: str = 'all_pairs', data_range: Optional[float] = None, window_size: int = 11, sigma: float = 1.5, k1: float = .01, k2: float = .03, chunk_size: int = 16) -> Tensor: ...
