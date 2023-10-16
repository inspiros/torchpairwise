from typing import Optional

from torch import Tensor, Generator
from torch.types import Number


# ~~~~~ sklearn ~~~~~
# noinspection PyPep8Naming
def euclidean_distances(x1: Tensor,
                        x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def haversine_distances(x1: Tensor,
                        x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def manhattan_distances(x1: Tensor,
                        x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def cosine_distances(x1: Tensor,
                     x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def linear_kernel(x1: Tensor,
                  x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def polynomial_kernel(x1: Tensor,
                      x2: Optional[Tensor] = None,
                      degree: int = 3,
                      gamma: Optional[float] = None,
                      coef0: float = 1.) -> Tensor: ...


# noinspection PyPep8Naming
def sigmoid_kernel(x1: Tensor,
                   x2: Optional[Tensor] = None,
                   gamma: Optional[float] = None,
                   coef0: float = 1.) -> Tensor: ...


# noinspection PyPep8Naming
def rbf_kernel(x1: Tensor,
               x2: Optional[Tensor] = None,
               gamma: Optional[float] = None) -> Tensor: ...


# noinspection PyPep8Naming
def laplacian_kernel(x1: Tensor,
                     x2: Optional[Tensor] = None,
                     gamma: Optional[float] = None) -> Tensor: ...


# noinspection PyPep8Naming
def cosine_similarity(x1: Tensor,
                      x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def additive_chi2_kernel(x1: Tensor,
                         x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def chi2_kernel(x1: Tensor,
                x2: Optional[Tensor] = None,
                gamma: float = 1.) -> Tensor: ...


# ~~~~~ scipy ~~~~~
# noinspection PyPep8Naming
def directed_hausdorff_distances(x1: Tensor,
                                 x2: Optional[Tensor] = None,
                                 *,
                                 shuffle: bool = False,
                                 generator: Optional[Generator] = None) -> Tensor: ...


# noinspection PyPep8Naming
def minkowski_distances(x1: Tensor,
                        x2: Optional[Tensor] = None,
                        p: float = 2) -> Tensor: ...


# noinspection PyPep8Naming
def wminkowski_distances(x1: Tensor,
                         x2: Optional[Tensor] = None,
                         p: float = 2,
                         w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def sqeuclidean_distances(x1: Tensor,
                          x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def correlation_distances(x1: Tensor,
                          x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def hamming_distances(x1: Tensor,
                      x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def jaccard_distances(x1: Tensor,
                      x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def kulsinski_distances(x1: Tensor,
                        x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def kulczynski1_distances(x1: Tensor,
                          x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def seuclidean_distances(x1: Tensor,
                         x2: Optional[Tensor] = None,
                         V: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def cityblock_distances(x1: Tensor,
                        x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def mahalanobis_distances(x1: Tensor,
                          x2: Optional[Tensor] = None,
                          VI: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def chebyshev_distances(x1: Tensor,
                        x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def braycurtis_distances(x1: Tensor,
                         x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def canberra_distances(x1: Tensor,
                       x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def jensenshannon_distances(x1: Tensor,
                            x2: Optional[Tensor] = None,
                            base: Optional[float] = None,
                            dim: int = -1,
                            keepdim: bool = False) -> Tensor: ...


# noinspection PyPep8Naming
def yule_distances(x1: Tensor,
                   x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def dice_distances(x1: Tensor,
                   x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def rogerstanimoto_distances(x1: Tensor,
                             x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def russellrao_distances(x1: Tensor,
                         x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def sokalmichener_distances(x1: Tensor,
                            x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def sokalsneath_distances(x1: Tensor,
                          x2: Optional[Tensor] = None) -> Tensor: ...


# ~~~~~ others ~~~~~
# noinspection PyPep8Naming
def snr_distances(x1: Tensor,
                  x2: Optional[Tensor] = None,
                  correction: Number = 1) -> Tensor: ...


# ~~~~~ aliases ~~~~~
def l1_distances(x1: Tensor,
                 x2: Optional[Tensor] = None) -> Tensor: ...


def l2_distances(x1: Tensor,
                 x2: Optional[Tensor] = None) -> Tensor: ...


def lp_distances(x1: Tensor,
                 x2: Optional[Tensor] = None,
                 p: float = 2) -> Tensor: ...
