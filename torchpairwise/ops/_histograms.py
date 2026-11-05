"""Internal compatibility adapters for native histogram comparisons."""
import torch


def wasserstein_1d(x, y, *, bin_positions=None, paired=False):
    return torch.ops.torchpairwise.vector_comparisons(
        x, y, 'wasserstein_1d', 'paired' if paired else 'all_pairs',
        bin_positions=bin_positions)


def histogram_comparison(spec, x, y, *, paired=False):
    return torch.ops.torchpairwise.vector_comparisons(
        x, y, spec.op, 'paired' if paired else 'all_pairs')
