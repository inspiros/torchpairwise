from typing import Optional
from torch import Tensor


def gaussian_mixture_overlap_kernel(
    x1_mean: Tensor,
    x1_covariance: Tensor,
    x1_weights: Tensor,
    x2_mean: Optional[Tensor] = None,
    x2_covariance: Optional[Tensor] = None,
    x2_weights: Optional[Tensor] = None,
    *,
    x1_lengths: Optional[Tensor] = None,
    x2_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
    component_chunk_size: int = 32,
) -> Tensor: ...


def paired_gaussian_mixture_overlap_kernel(
    x1_mean: Tensor,
    x1_covariance: Tensor,
    x1_weights: Tensor,
    x2_mean: Tensor,
    x2_covariance: Tensor,
    x2_weights: Tensor,
    *,
    x1_lengths: Optional[Tensor] = None,
    x2_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
    component_chunk_size: int = 32,
) -> Tensor: ...


def condensed_gaussian_mixture_overlap_kernel(
    means: Tensor,
    covariances: Tensor,
    weights: Tensor,
    *,
    x1_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
    component_chunk_size: int = 32,
) -> Tensor: ...


def gaussian_mixture_l2_distances(
    x1_mean: Tensor,
    x1_covariance: Tensor,
    x1_weights: Tensor,
    x2_mean: Optional[Tensor] = None,
    x2_covariance: Optional[Tensor] = None,
    x2_weights: Optional[Tensor] = None,
    *,
    squared: bool = False,
    x1_lengths: Optional[Tensor] = None,
    x2_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
    component_chunk_size: int = 32,
) -> Tensor: ...


def paired_gaussian_mixture_l2_distances(
    x1_mean: Tensor,
    x1_covariance: Tensor,
    x1_weights: Tensor,
    x2_mean: Tensor,
    x2_covariance: Tensor,
    x2_weights: Tensor,
    *,
    squared: bool = False,
    x1_lengths: Optional[Tensor] = None,
    x2_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
    component_chunk_size: int = 32,
) -> Tensor: ...


def condensed_gaussian_mixture_l2_distances(
    means: Tensor,
    covariances: Tensor,
    weights: Tensor,
    *,
    squared: bool = False,
    x1_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
    component_chunk_size: int = 32,
) -> Tensor: ...


def gaussian_mixture_cauchy_schwarz_divergences(
    x1_mean: Tensor,
    x1_covariance: Tensor,
    x1_weights: Tensor,
    x2_mean: Optional[Tensor] = None,
    x2_covariance: Optional[Tensor] = None,
    x2_weights: Optional[Tensor] = None,
    *,
    x1_lengths: Optional[Tensor] = None,
    x2_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
    component_chunk_size: int = 32,
) -> Tensor: ...


def paired_gaussian_mixture_cauchy_schwarz_divergences(
    x1_mean: Tensor,
    x1_covariance: Tensor,
    x1_weights: Tensor,
    x2_mean: Tensor,
    x2_covariance: Tensor,
    x2_weights: Tensor,
    *,
    x1_lengths: Optional[Tensor] = None,
    x2_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
    component_chunk_size: int = 32,
) -> Tensor: ...


def condensed_gaussian_mixture_cauchy_schwarz_divergences(
    means: Tensor,
    covariances: Tensor,
    weights: Tensor,
    *,
    x1_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
    component_chunk_size: int = 32,
) -> Tensor: ...


def gaussian_mixture_wasserstein_distances(
    x1_mean: Tensor,
    x1_covariance: Tensor,
    x1_weights: Tensor,
    x2_mean: Optional[Tensor] = None,
    x2_covariance: Optional[Tensor] = None,
    x2_weights: Optional[Tensor] = None,
    *,
    squared: bool = False,
    x1_lengths: Optional[Tensor] = None,
    x2_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
) -> Tensor: ...


def paired_gaussian_mixture_wasserstein_distances(
    x1_mean: Tensor,
    x1_covariance: Tensor,
    x1_weights: Tensor,
    x2_mean: Tensor,
    x2_covariance: Tensor,
    x2_weights: Tensor,
    *,
    squared: bool = False,
    x1_lengths: Optional[Tensor] = None,
    x2_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
) -> Tensor: ...


def condensed_gaussian_mixture_wasserstein_distances(
    means: Tensor,
    covariances: Tensor,
    weights: Tensor,
    *,
    squared: bool = False,
    x1_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
) -> Tensor: ...


def gaussian_mixture_entropic_transport_costs(
    x1_mean: Tensor,
    x1_covariance: Tensor,
    x1_weights: Tensor,
    x2_mean: Optional[Tensor] = None,
    x2_covariance: Optional[Tensor] = None,
    x2_weights: Optional[Tensor] = None,
    *,
    regularization: float = 0.1,
    max_iter: int = 100,
    x1_lengths: Optional[Tensor] = None,
    x2_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
) -> Tensor: ...


def paired_gaussian_mixture_entropic_transport_costs(
    x1_mean: Tensor,
    x1_covariance: Tensor,
    x1_weights: Tensor,
    x2_mean: Tensor,
    x2_covariance: Tensor,
    x2_weights: Tensor,
    *,
    regularization: float = 0.1,
    max_iter: int = 100,
    x1_lengths: Optional[Tensor] = None,
    x2_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
) -> Tensor: ...


def condensed_gaussian_mixture_entropic_transport_costs(
    means: Tensor,
    covariances: Tensor,
    weights: Tensor,
    *,
    regularization: float = 0.1,
    max_iter: int = 100,
    x1_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
) -> Tensor: ...


def gaussian_mixture_sinkhorn_divergences(
    x1_mean: Tensor,
    x1_covariance: Tensor,
    x1_weights: Tensor,
    x2_mean: Optional[Tensor] = None,
    x2_covariance: Optional[Tensor] = None,
    x2_weights: Optional[Tensor] = None,
    *,
    regularization: float = 0.1,
    max_iter: int = 100,
    x1_lengths: Optional[Tensor] = None,
    x2_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
) -> Tensor: ...


def paired_gaussian_mixture_sinkhorn_divergences(
    x1_mean: Tensor,
    x1_covariance: Tensor,
    x1_weights: Tensor,
    x2_mean: Tensor,
    x2_covariance: Tensor,
    x2_weights: Tensor,
    *,
    regularization: float = 0.1,
    max_iter: int = 100,
    x1_lengths: Optional[Tensor] = None,
    x2_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
) -> Tensor: ...


def condensed_gaussian_mixture_sinkhorn_divergences(
    means: Tensor,
    covariances: Tensor,
    weights: Tensor,
    *,
    regularization: float = 0.1,
    max_iter: int = 100,
    x1_lengths: Optional[Tensor] = None,
    chunk_size: int = 8,
) -> Tensor: ...
