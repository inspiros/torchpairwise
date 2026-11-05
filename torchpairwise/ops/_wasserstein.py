"""Exact native uniform transport with CPU flow control and device gradients."""
import torch


def wasserstein(x, y, *, p=1, x1_lengths=None, x2_lengths=None,
                paired=False, chunk_size=None):
    """Evaluate prepared point sets, including a direct paired path.

    Inputs are broadcast-prepared ``[...,N,P,C]`` / ``[...,M,Q,C]`` tensors.
    Uniform mass is assigned to each valid prefix point. ``p`` is 1 or 2 and
    the ground norm is Euclidean; W2 includes the square root. Output has the
    input dtype/device. At zero distances the coordinate derivative is zero.

    Each pair is solved serially, so ``chunk_size`` does not change peak dense
    solver storage: O(P*Q) per pair. It is accepted for the point-set contract.
    CUDA inputs synchronize and transfer detached ground costs to the CPU.
    Compile/export retain the native opaque solver; metadata execution reports
    shapes without solving transport. CUDA runtime still synchronizes with CPU.
    """
    if isinstance(p, bool) or not isinstance(p, (int, float)) or p not in (1, 2):
        raise ValueError('Wasserstein p must be 1 or 2')
    if x.device.type not in ('cpu', 'cuda', 'meta'):
        raise ValueError('Exact Wasserstein supports CPU and CUDA inputs')
    if paired and x.shape[-3] != y.shape[-3]:
        raise ValueError('Paired inputs must have equal sample counts')
    for value, lengths in ((x, x1_lengths), (y, x2_lengths)):
        finite = torch.isfinite(value).all(-1)
        if lengths is not None:
            finite = finite | (torch.arange(value.shape[-2], device=value.device) >= lengths.unsqueeze(-1))
        from ._batching import validate_values
        validate_values(finite.all(), 'Wasserstein valid point coordinates must be finite')
    return torch.ops.torchpairwise.transport_comparisons(
        x, y, 'wasserstein', 'paired' if paired else 'all_pairs', p=p,
        x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size)
