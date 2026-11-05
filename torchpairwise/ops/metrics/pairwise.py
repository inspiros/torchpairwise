"""All-pairs distances and kernels with broadcastable leading batches."""
import torch
from typing import Optional, Tuple
from torch import Tensor, Generator
from .._batching import call_pairwise


def kl_divergences(x1: Tensor, x2: Optional[Tensor] = None, base: Optional[float] = None) -> Tensor:
    """KL(p||q) of independently normalized nonnegative histogram rows.

    Natural logarithms by default. Incompatible supports return infinity with
    chosen zero pair gradients. Positive-mass rows are required; see docs for
    probability-boundary gradient conventions. Asymmetric; no condensed mode.
    """
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.kl_divergences(x1, x2, base)
    return call_pairwise('kl_divergences', x1, x2, base=base)


def symmetric_kl_divergences(x1: Tensor, x2: Optional[Tensor] = None,
                            base: Optional[float] = None) -> Tensor:
    """Jeffreys divergence KL(p||q)+KL(q||p), without a half factor or root."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.symmetric_kl_divergences(x1, x2, base)
    return call_pairwise('symmetric_kl_divergences', x1, x2, base=base)



def bhattacharyya_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
    """Negative natural logarithm of the normalized Bhattacharyya coefficient."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.bhattacharyya_distances(x1, x2)
    return call_pairwise('bhattacharyya_distances', x1, x2)


def bhattacharyya_kernel(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
    """Bhattacharyya coefficient of independently normalized histogram masses."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.bhattacharyya_kernel(x1, x2)
    return call_pairwise('bhattacharyya_kernel', x1, x2)


def aitchison_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
    """Euclidean distance between centered log-ratios of strictly positive rows."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.aitchison_distances(x1, x2)
    return call_pairwise('aitchison_distances', x1, x2)


def energy_distances(x1: Tensor, x2: Optional[Tensor] = None, *, squared: bool = False,
                      batched: Optional[bool] = None, feature_dim: int = -1,
                      x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                      chunk_size: Optional[int] = None) -> Tensor:
    """Uniform empirical distribution distance; squared=True returns its V statistic."""
    return call_pairwise('energy_distances', x1, x2, squared=squared,
                         batched=batched, feature_dim=feature_dim, x1_lengths=x1_lengths,
                         x2_lengths=x2_lengths, chunk_size=chunk_size)


def mmd_distances(x1: Tensor, x2: Optional[Tensor] = None, *, squared: bool = False, kernel: str = 'rbf', length_scale: float = 1.0,
                      gamma: Optional[float] = None, degree: Optional[int] = None, coef0: Optional[float] = None,
                      alpha: Optional[float] = None, nu: Optional[float] = None,
                      batched: Optional[bool] = None, feature_dim: int = -1,
                      x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                      chunk_size: Optional[int] = None) -> Tensor:
    """Biased empirical MMD for named kernels; squared=True returns its V statistic.

    Kernel parameters follow their pairwise counterparts. RBF defaults to
    gamma=1/(2*length_scale**2). Historical 'laplacian' is Euclidean;
    'laplacian_kernel' selects the registry's Manhattan kernel. Sigmoid is
    unsupported because it is not generally positive semidefinite.
    """
    return call_pairwise('mmd_distances', x1, x2, squared=squared, kernel=kernel, length_scale=length_scale,
                         gamma=gamma, degree=degree, coef0=coef0, alpha=alpha, nu=nu,
                         batched=batched, feature_dim=feature_dim, x1_lengths=x1_lengths,
                         x2_lengths=x2_lengths, chunk_size=chunk_size)

def wasserstein_distances(x1: Tensor, x2: Optional[Tensor] = None, *, p: float = 1,
                          batched: Optional[bool] = None, feature_dim: int = -1,
                          x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                          chunk_size: Optional[int] = None) -> Tensor:
    """Exact W1/W2 between uniform point distributions; host-coordinated solver.

    Inputs are [...,N,P,C] and [...,M,Q,C]. CUDA tensors and first coordinate
    gradients are supported, with CPU synchronization for optimization. Native
    opaque calls support compile/export; the solver remains CPU-controlled.
    Use sliced Wasserstein or Sinkhorn for GPU-oriented losses.
    """
    return call_pairwise('wasserstein_distances', x1, x2, p=p, batched=batched,
                         feature_dim=feature_dim, x1_lengths=x1_lengths,
                         x2_lengths=x2_lengths, chunk_size=chunk_size)


def sliced_wasserstein_distances(x1: Tensor, x2: Optional[Tensor] = None, *, projections: Tensor,
                                 p: float = 2, batched: Optional[bool] = None, feature_dim: int = -1,
                                 x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                                 chunk_size: Optional[int] = None) -> Tensor:
    """Finite-direction sliced W1/W2 of uniform clouds, using shared unit [C,L] projections.

    This differs from full multidimensional Wasserstein. Projection directions
    remain explicit so callers can reuse them across modes and repeated calls.
    """
    return call_pairwise('sliced_wasserstein_distances', x1, x2, projections=projections,
                         p=p, batched=batched, feature_dim=feature_dim,
                         x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size)


def sinkhorn_divergences(x1: Tensor, x2: Optional[Tensor] = None, *, p: float = 2,
                         regularization: float = 0.1, max_iter: int = 100,
                         batched: Optional[bool] = None, feature_dim: int = -1,
                         x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                         chunk_size: Optional[int] = None) -> Tensor:
    """Debiased KL-regularized transport estimates after a fixed iteration budget.

    Ground cost is Euclidean distance to power p (1 or 2), without a 1/p factor.
    Returns the debiased dual objective, without root or clamp. Finite iterations
    do not guarantee convergence or nonnegativity; gradients differentiate them.
    """
    return call_pairwise('sinkhorn_divergences', x1, x2, p=p,
                         regularization=regularization, max_iter=max_iter,
                         batched=batched, feature_dim=feature_dim,
                         x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size)

def euclidean_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('euclidean_distances', x1, x2, **({} if w is None else {'w': w}))

def haversine_distances(x1: Tensor, x2: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('haversine_distances', x1, x2)

def manhattan_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('manhattan_distances', x1, x2, **({} if w is None else {'w': w}))

def cosine_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    """Cosine distance; optional nonnegative feature weights require positive total weight."""
    return call_pairwise('cosine_distances', x1, x2, **({} if w is None else {'w': w}))

def linear_kernel(x1: Tensor, x2: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('linear_kernel', x1, x2)

def polynomial_kernel(x1: Tensor, x2: Optional[Tensor]=None, degree: int=3, gamma: Optional[float]=None, coef0: float=1.0) -> Tensor:
    return call_pairwise('polynomial_kernel', x1, x2, degree=degree, gamma=gamma, coef0=coef0)

def sigmoid_kernel(x1: Tensor, x2: Optional[Tensor]=None, gamma: Optional[float]=None, coef0: float=1.0) -> Tensor:
    return call_pairwise('sigmoid_kernel', x1, x2, gamma=gamma, coef0=coef0)

def rbf_kernel(x1: Tensor, x2: Optional[Tensor]=None, gamma: Optional[float]=None) -> Tensor:
    return call_pairwise('rbf_kernel', x1, x2, gamma=gamma)

def laplacian_kernel(x1: Tensor, x2: Optional[Tensor]=None, gamma: Optional[float]=None) -> Tensor:
    return call_pairwise('laplacian_kernel', x1, x2, gamma=gamma)

def rational_quadratic_kernel(x1: Tensor, x2: Optional[Tensor]=None, length_scale: float=1.0, alpha: float=1.0) -> Tensor:
    """Isotropic rational quadratic kernel, with positive scalar scale and alpha.

    Returns ``[..., N, M]``; leading batches broadcast. The diagonal is one
    for finite coincident inputs. Parameters are fixed Python scalars.
    """
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.rational_quadratic_kernel(x1, x2, length_scale, alpha)
    return call_pairwise('rational_quadratic_kernel', x1, x2, length_scale=length_scale, alpha=alpha)

def matern_kernel(x1: Tensor, x2: Optional[Tensor]=None, length_scale: float=1.0, nu: float=1.5) -> Tensor:
    """Isotropic Matérn kernel for nu=0.5, 1.5 or 2.5 (default: 1.5).

    ``length_scale`` is a positive scalar; both options are fixed Python
    scalars. Uses Euclidean distance, including for nu=0.5. Gradients at
    coincident inputs use a zero subgradient.
    """
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.matern_kernel(x1, x2, length_scale, nu)
    return call_pairwise('matern_kernel', x1, x2, length_scale=length_scale, nu=nu)

def cosine_similarity(x1: Tensor, x2: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('cosine_similarity', x1, x2)

def hellinger_distances(x1: Tensor, x2: Optional[Tensor]=None) -> Tensor:
    """Hellinger distance after normalizing each nonnegative row to unit mass.

    Rows must be finite with positive finite mass. Returns ``[..., N, M]``.
    Coincident distances and the square-root transform at zero use zero
    gradient conventions; gradients on probability boundaries are not smooth.
    """
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.hellinger_distances(x1, x2)
    return call_pairwise('hellinger_distances', x1, x2)

def nan_euclidean_distances(x1: Tensor, x2: Optional[Tensor]=None, squared: bool=False) -> Tensor:
    """Euclidean distance on jointly observed features, rescaled by D/count.

    NaNs indicate missing features; infinities are rejected. No-overlap pairs
    return NaN with a chosen zero gradient. Missing coordinates and coincident
    distances receive zero gradients. ``squared=True`` omits the square root.
    """
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.nan_euclidean_distances(x1, x2, squared)
    return call_pairwise('nan_euclidean_distances', x1, x2, squared=squared)

def wasserstein_1d_distances(x1: Tensor, x2: Optional[Tensor]=None, bin_positions: Optional[Tensor]=None) -> Tensor:
    """1D Wasserstein distance of normalized histogram masses on a shared grid.

    Rows must be finite, nonnegative and have positive finite mass. Bins are
    unit-spaced by default; optional [D] positions must be finite and strictly
    increasing, with input dtype/device. Positions and masses support gradients.
    """
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.wasserstein_1d_distances(x1, x2, bin_positions)
    return call_pairwise('wasserstein_1d_distances', x1, x2, bin_positions=bin_positions)

def histogram_intersection_kernel(x1: Tensor, x2: Optional[Tensor]=None) -> Tensor:
    """Sum of per-feature minima of finite nonnegative vectors, without normalization.

    All-zero rows are valid. Tied bins split the derivative equally between
    inputs, matching ``torch.minimum``. Returns ``[..., N, M]``.
    """
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.histogram_intersection_kernel(x1, x2)
    return call_pairwise('histogram_intersection_kernel', x1, x2)

def additive_chi2_kernel(x1: Tensor, x2: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('additive_chi2_kernel', x1, x2)

def chi2_kernel(x1: Tensor, x2: Optional[Tensor]=None, gamma: float=1.0) -> Tensor:
    return call_pairwise('chi2_kernel', x1, x2, gamma=gamma)

def directed_hausdorff_distances(x1: Tensor, x2: Optional[Tensor]=None, *, shuffle: bool=False, generator: Optional[Generator]=None, batched: Optional[bool]=None, feature_dim: int=-1, x1_lengths: Optional[Tensor]=None, x2_lengths: Optional[Tensor]=None, chunk_size: Optional[int]=None) -> Tuple[Tensor, Tensor, Tensor]:
    """Compare point sets, returning distances and flattened point witnesses.

    ``batched=False`` accepts ``[N, *points, C]``; ``True`` accepts
    ``[B, N, *points, C]``. ``feature_dim`` selects C (default: last).
    All remaining axes after N are flattened as points in their original order.
    Omit ``batched`` to retain the legacy ``[..., N, P, C]`` interpretation.
    Optional int64 lengths select a non-empty prefix of each flattened point
    set; padding is ignored. Shuffling is available only without lengths.
    ``chunk_size`` tiles sample comparisons, checkpointing witnesses in training;
    it cannot be combined with shuffle/generator.
    """
    return call_pairwise('directed_hausdorff_distances', x1, x2, shuffle=shuffle,
                         generator=generator, batched=batched, feature_dim=feature_dim,
                         x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size)

def hausdorff_distances(x1: Tensor, x2: Optional[Tensor]=None, *, shuffle: bool=False, generator: Optional[Generator]=None, batched: Optional[bool]=None, feature_dim: int=-1, x1_lengths: Optional[Tensor]=None, x2_lengths: Optional[Tensor]=None, chunk_size: Optional[int]=None) -> Tensor:
    """Symmetric Hausdorff: maximum of the two directed Euclidean distances.

    Point layout, lengths and shuffle options follow directed Hausdorff.
    Returns only distances, with all-pairs shape ``[..., N, M]``.
    """
    return call_pairwise('hausdorff_distances', x1, x2, shuffle=shuffle,
                         generator=generator, batched=batched, feature_dim=feature_dim,
                         x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size)

def chamfer_distances(x1: Tensor, x2: Optional[Tensor]=None, *, squared: bool=False, batched: Optional[bool]=None, feature_dim: int=-1, x1_lengths: Optional[Tensor]=None, x2_lengths: Optional[Tensor]=None, chunk_size: Optional[int]=None) -> Tensor:
    """Sum of directional mean nearest-point distances, shape ``[..., N, M]``.

    Uses Euclidean ground distances, or squared Euclidean when ``squared=True``.
    Point layout, prefix lengths and optional sample chunking follow directed Hausdorff.
    """
    return call_pairwise('chamfer_distances', x1, x2, squared=squared,
                         batched=batched, feature_dim=feature_dim,
                         x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size)

def minkowski_distances(x1: Tensor, x2: Optional[Tensor]=None, p: float=2, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('minkowski_distances', x1, x2, p=p, **({} if w is None else {'w': w}))

def wminkowski_distances(x1: Tensor, x2: Optional[Tensor]=None, p: float=2, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('wminkowski_distances', x1, x2, p=p, w=w)

def sqeuclidean_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('sqeuclidean_distances', x1, x2, **({} if w is None else {'w': w}))

def correlation_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    """Correlation distance; optional weights define weighted centering and norms."""
    return call_pairwise('correlation_distances', x1, x2, **({} if w is None else {'w': w}))

def hamming_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('hamming_distances', x1, x2, **({} if w is None else {'w': w}))

def jaccard_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('jaccard_distances', x1, x2, **({} if w is None else {'w': w}))

def kulsinski_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('kulsinski_distances', x1, x2, **({} if w is None else {'w': w}))

def kulczynski1_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('kulczynski1_distances', x1, x2, **({} if w is None else {'w': w}))

def seuclidean_distances(x1: Tensor, x2: Optional[Tensor]=None, V: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('seuclidean_distances', x1, x2, V=V)

def cityblock_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('cityblock_distances', x1, x2, **({} if w is None else {'w': w}))

def mahalanobis_distances(x1: Tensor, x2: Optional[Tensor]=None, VI: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('mahalanobis_distances', x1, x2, VI=VI)

def chebyshev_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('chebyshev_distances', x1, x2, **({} if w is None else {'w': w}))

def braycurtis_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('braycurtis_distances', x1, x2, **({} if w is None else {'w': w}))

def canberra_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('canberra_distances', x1, x2, **({} if w is None else {'w': w}))

def jensenshannon_distances(x1: Tensor, x2: Optional[Tensor]=None, base: Optional[float]=None) -> Tensor:
    return call_pairwise('jensenshannon_distances', x1, x2, base=base)

def yule_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('yule_distances', x1, x2, **({} if w is None else {'w': w}))

def dice_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('dice_distances', x1, x2, **({} if w is None else {'w': w}))

def rogerstanimoto_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('rogerstanimoto_distances', x1, x2, **({} if w is None else {'w': w}))

def russellrao_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('russellrao_distances', x1, x2, **({} if w is None else {'w': w}))

def sokalmichener_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('sokalmichener_distances', x1, x2, **({} if w is None else {'w': w}))

def sokalsneath_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('sokalsneath_distances', x1, x2, **({} if w is None else {'w': w}))

def snr_distances(x1: Tensor, x2: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('snr_distances', x1, x2)

def l1_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('l1_distances', x1, x2, **({} if w is None else {'w': w}))

def l2_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('l2_distances', x1, x2, **({} if w is None else {'w': w}))

def lp_distances(x1: Tensor, x2: Optional[Tensor]=None, p: float=2, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('lp_distances', x1, x2, p=p, **({} if w is None else {'w': w}))

def linf_distances(x1: Tensor, x2: Optional[Tensor]=None, *, w: Optional[Tensor]=None) -> Tensor:
    return call_pairwise('linf_distances', x1, x2, **({} if w is None else {'w': w}))

def total_variation_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
    """Half the L1 distance between independently normalized histogram rows."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.total_variation_distances(x1, x2)
    return call_pairwise('total_variation_distances', x1, x2)


def fisher_rao_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
    """Fisher-Rao probability distance in radians, twice the angle of square roots."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.fisher_rao_distances(x1, x2)
    return call_pairwise('fisher_rao_distances', x1, x2)


def renyi_divergences(x1: Tensor, x2: Optional[Tensor] = None, alpha: float = 1.0) -> Tensor:
    """Directional normalized Rényi divergence; positive alpha, natural logs, KL at one."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.renyi_divergences(x1, x2, alpha)
    return call_pairwise('renyi_divergences', x1, x2, alpha=alpha)


def pearson_chi2_divergences(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
    """Normalized Pearson chi-square divergence, with the second operand in the denominator."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.pearson_chi2_divergences(x1, x2)
    return call_pairwise('pearson_chi2_divergences', x1, x2)


def neyman_chi2_divergences(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
    """Normalized Neyman chi-square divergence, with the first operand in the denominator."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.neyman_chi2_divergences(x1, x2)
    return call_pairwise('neyman_chi2_divergences', x1, x2)


def triangular_discrimination_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
    """Normalized triangular discrimination, sum((p-q)^2/(p+q)), without a half factor."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.triangular_discrimination_distances(x1, x2)
    return call_pairwise('triangular_discrimination_distances', x1, x2)


def itakura_saito_divergences(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
    """Raw positive-spectrum Itakura-Saito divergence; inputs are not normalized."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.itakura_saito_divergences(x1, x2)
    return call_pairwise('itakura_saito_divergences', x1, x2)


def angular_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
    """Angles in radians between nonzero real vectors, from zero to pi."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.angular_distances(x1, x2)
    return call_pairwise('angular_distances', x1, x2)


def lorentzian_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
    """Sum of log1p absolute coordinate differences between finite real vectors."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.lorentzian_distances(x1, x2)
    return call_pairwise('lorentzian_distances', x1, x2)


def clark_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
    """Euclidean norm of absolute coordinate differences divided by sums, for nonnegative vectors."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.clark_distances(x1, x2)
    return call_pairwise('clark_distances', x1, x2)


def soergel_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
    """Nonnegative continuous Jaccard distance; the same formula as Ruzicka."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.soergel_distances(x1, x2)
    return call_pairwise('soergel_distances', x1, x2)


def ruzicka_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
    """Nonnegative continuous Jaccard distance, sum(abs(x-y))/sum(max(x,y))."""
    if torch.jit.is_scripting():
        return torch.ops.torchpairwise.ruzicka_distances(x1, x2)
    return call_pairwise('ruzicka_distances', x1, x2)


def continuous_jaccard_distances(x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
    """Alias of Ruzicka distance for finite nonnegative vectors."""
    return call_pairwise('ruzicka_distances', x1, x2)

__all__ = ['total_variation_distances', 'fisher_rao_distances', 'renyi_divergences', 'pearson_chi2_divergences', 'neyman_chi2_divergences', 'triangular_discrimination_distances', 'itakura_saito_divergences', 'angular_distances', 'lorentzian_distances', 'clark_distances', 'soergel_distances', 'ruzicka_distances', 'continuous_jaccard_distances', 'bhattacharyya_distances', 'bhattacharyya_kernel', 'aitchison_distances', 'energy_distances', 'mmd_distances', 'kl_divergences', 'symmetric_kl_divergences', 'wasserstein_distances', 'sliced_wasserstein_distances', 'sinkhorn_divergences', 'nan_euclidean_distances', 'wasserstein_1d_distances', 'hellinger_distances', 'histogram_intersection_kernel', 'rational_quadratic_kernel', 'matern_kernel', 'hausdorff_distances', 'chamfer_distances', 'euclidean_distances', 'haversine_distances', 'manhattan_distances', 'cosine_distances', 'linear_kernel', 'polynomial_kernel', 'sigmoid_kernel', 'rbf_kernel', 'laplacian_kernel', 'cosine_similarity', 'additive_chi2_kernel', 'chi2_kernel', 'directed_hausdorff_distances', 'minkowski_distances', 'wminkowski_distances', 'sqeuclidean_distances', 'correlation_distances', 'hamming_distances', 'jaccard_distances', 'kulsinski_distances', 'kulczynski1_distances', 'seuclidean_distances', 'cityblock_distances', 'mahalanobis_distances', 'chebyshev_distances', 'braycurtis_distances', 'canberra_distances', 'jensenshannon_distances', 'yule_distances', 'dice_distances', 'rogerstanimoto_distances', 'russellrao_distances', 'sokalmichener_distances', 'sokalsneath_distances', 'snr_distances', 'l1_distances', 'l2_distances', 'lp_distances', 'linf_distances']
