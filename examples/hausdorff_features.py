"""Compare image descriptors as unordered point sets, with either channel layout."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(7)
    # One independent batch containing two/three image feature samples.
    x = torch.rand(1, 2, 4, 3, 5, device=device, requires_grad=True)
    y = torch.rand(1, 3, 4, 2, 3, device=device, requires_grad=True)
    distances, xi, yi = tp.directed_hausdorff_distances(
        x, y, batched=True, feature_dim=2)
    last = tp.cdist(x.movedim(2, -1), y.movedim(2, -1),
                    metric='directed_hausdorff', batched=True)
    torch.testing.assert_close(distances, last)
    distances.sum().backward()
    x_rows, x_columns = torch.unravel_index(xi, x.shape[-2:])
    print('All-pairs distances:', distances)
    print('X witness rows/columns:', x_rows, x_columns)
    print('Y witness rows/columns:', torch.unravel_index(yi, y.shape[-2:]))
    print('Input gradient shapes:', x.grad.shape, y.grad.shape)


if __name__ == '__main__':
    main()
