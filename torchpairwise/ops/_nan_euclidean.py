"""Internal compatibility adapter for the native missing-data comparison."""
import torch


def nan_euclidean(x, y, *, squared=False, paired=False):
    return torch.ops.torchpairwise.vector_comparisons(
        x, y, 'nan_euclidean', 'paired' if paired else 'all_pairs', squared=squared)
