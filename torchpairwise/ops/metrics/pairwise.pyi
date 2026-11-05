from typing import Optional, Tuple

from torch import Tensor, Generator
from torch.types import Number


# ~~~~~ sklearn ~~~~~
# noinspection PyPep8Naming
def euclidean_distances(x1: Tensor,
                        x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def haversine_distances(x1: Tensor,
                        x2: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def manhattan_distances(x1: Tensor,
                        x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def cosine_distances(x1: Tensor,
                     x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


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
                                 generator: Optional[Generator] = None,
                                 batched: Optional[bool] = None,
                                 feature_dim: int = -1,
                                 x1_lengths: Optional[Tensor] = None,
                                 x2_lengths: Optional[Tensor] = None,
                                 chunk_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor]: ...


# noinspection PyPep8Naming
def minkowski_distances(x1: Tensor,
                        x2: Optional[Tensor] = None,
                        p: float = 2, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def wminkowski_distances(x1: Tensor,
                         x2: Optional[Tensor] = None,
                         p: float = 2,
                         w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def sqeuclidean_distances(x1: Tensor,
                          x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def correlation_distances(x1: Tensor,
                          x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def hamming_distances(x1: Tensor,
                      x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def jaccard_distances(x1: Tensor,
                      x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def kulsinski_distances(x1: Tensor,
                        x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def kulczynski1_distances(x1: Tensor,
                          x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def seuclidean_distances(x1: Tensor,
                         x2: Optional[Tensor] = None,
                         V: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def cityblock_distances(x1: Tensor,
                        x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def mahalanobis_distances(x1: Tensor,
                          x2: Optional[Tensor] = None,
                          VI: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def chebyshev_distances(x1: Tensor,
                        x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def braycurtis_distances(x1: Tensor,
                         x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def canberra_distances(x1: Tensor,
                       x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def jensenshannon_distances(x1: Tensor,
                            x2: Optional[Tensor] = None,
                            base: Optional[float] = None) -> Tensor: ...


# noinspection PyPep8Naming
def yule_distances(x1: Tensor,
                   x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def dice_distances(x1: Tensor,
                   x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def rogerstanimoto_distances(x1: Tensor,
                             x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def russellrao_distances(x1: Tensor,
                         x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def sokalmichener_distances(x1: Tensor,
                            x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# noinspection PyPep8Naming
def sokalsneath_distances(x1: Tensor,
                          x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


# ~~~~~ others ~~~~~
# noinspection PyPep8Naming
def snr_distances(x1: Tensor,
                  x2: Optional[Tensor] = None) -> Tensor: ...


# ~~~~~ aliases ~~~~~
def l1_distances(x1: Tensor,
                 x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


def l2_distances(x1: Tensor,
                 x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


def lp_distances(x1: Tensor,
                 x2: Optional[Tensor] = None,
                 p: float = 2, *, w: Optional[Tensor] = None) -> Tensor: ...


def linf_distances(x1: Tensor, x2: Optional[Tensor] = None, *, w: Optional[Tensor] = None) -> Tensor: ...


def hausdorff_distances(x1: Tensor, x2: Optional[Tensor] = None, *,
                        shuffle: bool = False, generator: Optional[Generator] = None,
                        batched: Optional[bool] = None, feature_dim: int = -1,
                        x1_lengths: Optional[Tensor] = None,
                        x2_lengths: Optional[Tensor] = None,
                        chunk_size: Optional[int] = None) -> Tensor: ...


def chamfer_distances(x1: Tensor, x2: Optional[Tensor] = None, *, squared: bool = False,
                      batched: Optional[bool] = None, feature_dim: int = -1,
                      x1_lengths: Optional[Tensor] = None,
                      x2_lengths: Optional[Tensor] = None,
                      chunk_size: Optional[int] = None) -> Tensor: ...


def rational_quadratic_kernel(x1: Tensor, x2: Optional[Tensor] = None,
                              length_scale: float = 1.0, alpha: float = 1.0) -> Tensor: ...


def matern_kernel(x1: Tensor, x2: Optional[Tensor] = None,
                  length_scale: float = 1.0, nu: float = 1.5) -> Tensor: ...


def hellinger_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...


def histogram_intersection_kernel(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...


def nan_euclidean_distances(x1: Tensor, x2: Optional[Tensor] = None,
                            squared: bool = False) -> Tensor: ...


def wasserstein_1d_distances(x1: Tensor, x2: Optional[Tensor] = None,
                            bin_positions: Optional[Tensor] = None) -> Tensor: ...


def wasserstein_distances(x1: Tensor, x2: Optional[Tensor] = None, *, p: float = 1,
                          batched: Optional[bool] = None, feature_dim: int = -1,
                          x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                          chunk_size: Optional[int] = None) -> Tensor: ...


def sliced_wasserstein_distances(x1: Tensor, x2: Optional[Tensor] = None, *, projections: Tensor,
                                 p: float = 2, batched: Optional[bool] = None, feature_dim: int = -1,
                                 x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                                 chunk_size: Optional[int] = None) -> Tensor: ...


def sinkhorn_divergences(x1: Tensor, x2: Optional[Tensor] = None, *, p: float = 2,
                         regularization: float = 0.1, max_iter: int = 100,
                         batched: Optional[bool] = None, feature_dim: int = -1,
                         x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                         chunk_size: Optional[int] = None) -> Tensor: ...


def kl_divergences(x1: Tensor, x2: Optional[Tensor] = None,
                   base: Optional[float] = None) -> Tensor: ...


def symmetric_kl_divergences(x1: Tensor, x2: Optional[Tensor] = None,
                            base: Optional[float] = None) -> Tensor: ...


def bhattacharyya_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...


def bhattacharyya_kernel(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...


def aitchison_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...


def energy_distances(x1: Tensor, x2: Optional[Tensor] = None, *, squared: bool = False,
                      batched: Optional[bool] = None, feature_dim: int = -1,
                      x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                      chunk_size: Optional[int] = None) -> Tensor: ...


def mmd_distances(x1: Tensor, x2: Optional[Tensor] = None, *, squared: bool = False, kernel: str = 'rbf', length_scale: float = 1.0,
                      gamma: Optional[float] = None, degree: Optional[int] = None, coef0: Optional[float] = None,
                      alpha: Optional[float] = None, nu: Optional[float] = None,
                      batched: Optional[bool] = None, feature_dim: int = -1,
                      x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                      chunk_size: Optional[int] = None) -> Tensor: ...


def total_variation_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...

def fisher_rao_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...

def renyi_divergences(x1: Tensor, x2: Optional[Tensor] = None, alpha: float = 1.0) -> Tensor: ...

def pearson_chi2_divergences(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...

def neyman_chi2_divergences(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...

def triangular_discrimination_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...

def itakura_saito_divergences(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...

def angular_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...

def lorentzian_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...

def clark_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...

def soergel_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...

def ruzicka_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...

def continuous_jaccard_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor: ...
