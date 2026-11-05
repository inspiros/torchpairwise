"""Ordered trajectories with padding, pair modes and a differentiable loss.

Run from an installed torchpairwise environment:
    python examples/sequence_comparisons.py
"""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trajectories = torch.tensor([
        [[0., 0.], [1., .1], [2., 0.], [3., .2]],
        [[0., .1], [1.8, 0.], [3., .3], [float('nan'), float('nan')]],
        [[0., 1.], [.8, 1.1], [2., .9], [3., 1.]],
    ], device=device, requires_grad=True)
    lengths = torch.tensor([4, 3, 4], dtype=torch.int64, device=device)
    print('All-pairs rooted DTW:')
    print(tp.dtw_distances(trajectories, x1_lengths=lengths, chunk_size=2))
    print('Condensed Soft-DTW divergence:')
    print(tp.condensed_soft_dtw_divergences(
        trajectories, x_lengths=lengths, gamma=.5, chunk_size=2))
    target = torch.nan_to_num(trajectories.detach()) + .1
    loss = tp.paired_soft_dtw_divergences(
        trajectories, target, x1_lengths=lengths, x2_lengths=lengths,
        gamma=.5, window=1).mean()
    loss.backward()
    print('Aligned training loss:', loss.detach())
    print('Padded-frame gradient:', trajectories.grad[1, 3])


if __name__ == '__main__':
    main()
