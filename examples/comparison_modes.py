"""Run directly with a source-built torchpairwise installation."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() and tp.with_cuda() else 'cpu'
    x = torch.rand(2, 1, 8, 4, device=device)
    y = torch.rand(1, 3, 8, 4, device=device)
    print('All pairs:', tp.cdist(x, y, metric='canberra').shape)
    print('Paired rows:', tp.paired_distances(x, y, metric='canberra').shape)
    print('Paired RBF:', tp.paired_kernels(x, y, kernel='rbf').shape)
    print('Condensed distances:', tp.pdist(x, metric='canberra').shape)
    print('Condensed RBF:', tp.condensed_kernels(x, kernel='rbf').shape)


if __name__ == '__main__':
    main()
