from typing import Optional
from torch import Tensor


def gaussian_wasserstein_distances(x1_mean: Tensor, x1_covariance: Tensor,
                                   x2_mean: Optional[Tensor] = None,
                                   x2_covariance: Optional[Tensor] = None, *,
                                   squared: bool = False) -> Tensor: ...


def paired_gaussian_wasserstein_distances(x1_mean: Tensor, x1_covariance: Tensor,
                                          x2_mean: Tensor, x2_covariance: Tensor, *,
                                          squared: bool = False) -> Tensor: ...


def condensed_gaussian_wasserstein_distances(means: Tensor, covariances: Tensor, *,
                                             squared: bool = False) -> Tensor: ...
