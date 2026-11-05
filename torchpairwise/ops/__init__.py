from .cpdist import *
from .metrics import *
from .paired import *
from .condensed import *
from ._gaussian import (
    gaussian_wasserstein_distances,
    paired_gaussian_wasserstein_distances,
    condensed_gaussian_wasserstein_distances,
)

from ._covariance import (
    log_euclidean_distances,
    paired_log_euclidean_distances,
    condensed_log_euclidean_distances,
    affine_invariant_distances,
    paired_affine_invariant_distances,
    condensed_affine_invariant_distances,
    stein_distances,
    paired_stein_distances,
    condensed_stein_distances,
)

from ._custom import CustomComparison, make_pairwise
from ._residual_operator import ResidualOperator, residual_operator
from ._features import FeatureComparison, PreparedFeatureComparison, make_feature_pairwise

from ._image import (
    psnr_scores,
    paired_psnr_scores,
    condensed_psnr_scores,
    ssim_scores,
    paired_ssim_scores,
    condensed_ssim_scores,
    multiscale_ssim_scores,
    paired_multiscale_ssim_scores,
    condensed_multiscale_ssim_scores,
)

from ._tokens import (
    levenshtein_distances,
    paired_levenshtein_distances,
    condensed_levenshtein_distances,
    indel_distances,
    paired_indel_distances,
    condensed_indel_distances,
)

from ._audio import (
    si_sdr_scores,
    paired_si_sdr_scores,
    signal_snr_scores,
    paired_signal_snr_scores,
    log_spectral_distances,
    paired_log_spectral_distances,
    condensed_log_spectral_distances,
    spectral_convergence_distances,
    paired_spectral_convergence_distances,
    multiresolution_stft_distances,
    paired_multiresolution_stft_distances,
)

from ._sequence import (
    dtw_distances,
    paired_dtw_distances,
    condensed_dtw_distances,
    soft_dtw_divergences,
    paired_soft_dtw_divergences,
    condensed_soft_dtw_divergences,
)

from ._mixture import (
    gaussian_mixture_overlap_kernel,
    paired_gaussian_mixture_overlap_kernel,
    condensed_gaussian_mixture_overlap_kernel,
    gaussian_mixture_l2_distances,
    paired_gaussian_mixture_l2_distances,
    condensed_gaussian_mixture_l2_distances,
    gaussian_mixture_cauchy_schwarz_divergences,
    paired_gaussian_mixture_cauchy_schwarz_divergences,
    condensed_gaussian_mixture_cauchy_schwarz_divergences,
    gaussian_mixture_wasserstein_distances,
    paired_gaussian_mixture_wasserstein_distances,
    condensed_gaussian_mixture_wasserstein_distances,
    gaussian_mixture_entropic_transport_costs,
    paired_gaussian_mixture_entropic_transport_costs,
    condensed_gaussian_mixture_entropic_transport_costs,
    gaussian_mixture_sinkhorn_divergences,
    paired_gaussian_mixture_sinkhorn_divergences,
    condensed_gaussian_mixture_sinkhorn_divergences,
)

from ._distribution import mmd_statistics, paired_mmd_statistics, condensed_mmd_statistics

from ._comparison_dispatch import ComparisonInfo, compare, single_comparison, list_comparisons

from ._transport_maps import sinkhorn_barycentric_map, sinkhorn_displacement

from ._image_maps import image_maps
from ._image_map_operator import ImageMapOperator, image_map_operator

from ._lazy import *
from . import masks

from types import ModuleType as _ModuleType
__all__ = [name for name, value in globals().items()
           if not name.startswith('_') and not isinstance(value, _ModuleType)]
