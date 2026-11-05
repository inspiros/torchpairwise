from typing import Optional, Union, List, Tuple
from torch import Tensor


def psnr_scores(x1: Tensor, x2: Optional[Tensor] = None, *, data_range: float,
                chunk_size: int = 16) -> Tensor: ...
def paired_psnr_scores(x1: Tensor, x2: Tensor, *, data_range: float,
                       chunk_size: int = 16) -> Tensor: ...
def condensed_psnr_scores(x: Tensor, *, data_range: float,
                          chunk_size: int = 16) -> Tensor: ...

def ssim_scores(x1: Tensor, x2: Optional[Tensor] = None, *, data_range: float,
                window_size: int = 11, sigma: float = 1.5, k1: float = .01,
                k2: float = .03, chunk_size: int = 16) -> Tensor: ...
def paired_ssim_scores(x1: Tensor, x2: Tensor, *, data_range: float,
                       window_size: int = 11, sigma: float = 1.5, k1: float = .01,
                       k2: float = .03, chunk_size: int = 16) -> Tensor: ...
def condensed_ssim_scores(x: Tensor, *, data_range: float, window_size: int = 11,
                          sigma: float = 1.5, k1: float = .01, k2: float = .03,
                          chunk_size: int = 16) -> Tensor: ...

def multiscale_ssim_scores(x1: Tensor, x2: Optional[Tensor] = None, *,
                           data_range: float, window_size: int = 11,
                           sigma: float = 1.5, k1: float = .01, k2: float = .03,
                           weights: Union[Tuple[float, ...], List[float]] = ...,
                           chunk_size: int = 16) -> Tensor: ...
def paired_multiscale_ssim_scores(x1: Tensor, x2: Tensor, *, data_range: float,
                                  window_size: int = 11, sigma: float = 1.5,
                                  k1: float = .01, k2: float = .03,
                                  weights: Union[Tuple[float, ...], List[float]] = ...,
                                  chunk_size: int = 16) -> Tensor: ...
def condensed_multiscale_ssim_scores(x: Tensor, *, data_range: float,
                                     window_size: int = 11, sigma: float = 1.5,
                                     k1: float = .01, k2: float = .03,
                                     weights: Union[Tuple[float, ...], List[float]] = ...,
                                     chunk_size: int = 16) -> Tensor: ...
