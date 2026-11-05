"""Point-cloud distances with ragged prefixes and three comparison modes."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(12)
    x = torch.rand(3, 5, 2, device=device, requires_grad=True)
    y = torch.rand(3, 4, 2, device=device, requires_grad=True)
    xl = torch.tensor([5, 2, 4], device=device)
    yl = torch.tensor([3, 4, 2], device=device)
    for metric in ('hausdorff', 'chamfer'):
        matrix = tp.cdist(x, y, metric=metric, x1_lengths=xl, x2_lengths=yl)
        paired = tp.paired_distances(x, y, metric=metric, x1_lengths=xl, x2_lengths=yl)
        condensed = tp.pdist(x, metric=metric, x1_lengths=xl)
        chunked = tp.cdist(x, y, metric=metric, x1_lengths=xl, x2_lengths=yl, chunk_size=2)
        torch.testing.assert_close(chunked, matrix)
        torch.testing.assert_close(paired, matrix.diagonal())
        gx, gy = torch.autograd.grad(matrix.sum() + paired.sum() + condensed.sum(), (x, y))
        assert torch.isfinite(gx).all() and torch.isfinite(gy).all()
        print(metric, 'all-pairs:', matrix, 'paired:', paired, 'condensed:', condensed)


if __name__ == '__main__':
    main()
