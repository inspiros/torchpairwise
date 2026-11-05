from typing import Optional
from torch import Tensor


def dtw_distances(x1: Tensor, x2: Optional[Tensor] = None, *,
                  x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                  squared: bool = False, window: Optional[int] = None,
                  chunk_size: int = 16) -> Tensor: ...
def paired_dtw_distances(x1: Tensor, x2: Tensor, *,
                         x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                         squared: bool = False, window: Optional[int] = None,
                         chunk_size: int = 16) -> Tensor: ...
def condensed_dtw_distances(x: Tensor, *, x_lengths: Optional[Tensor] = None,
                            squared: bool = False, window: Optional[int] = None,
                            chunk_size: int = 16) -> Tensor: ...
def soft_dtw_divergences(x1: Tensor, x2: Optional[Tensor] = None, *, gamma: float,
                         x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                         window: Optional[int] = None, chunk_size: int = 16) -> Tensor: ...
def paired_soft_dtw_divergences(x1: Tensor, x2: Tensor, *, gamma: float,
                                x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                                window: Optional[int] = None, chunk_size: int = 16) -> Tensor: ...
def condensed_soft_dtw_divergences(x: Tensor, *, gamma: float, x_lengths: Optional[Tensor] = None,
                                   window: Optional[int] = None, chunk_size: int = 16) -> Tensor: ...
