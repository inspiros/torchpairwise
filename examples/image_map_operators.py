"""Image event statistics feeding streamed scalar neighbors and smoothing."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(141)
    x = torch.rand(5, 3, 12, 14, device=device, requires_grad=True)
    y = torch.rand(9, 3, 12, 14, device=device, requires_grad=True)
    values = torch.rand(9, 4, device=device, requires_grad=True)
    maps = tp.image_map_operator(x, y, comparison='absolute_error', chunk_size=2)
    channels = maps.mean((-2, -1))  # Lazy [N,M,C].
    nearest = channels.max(-1).topk(3, largest=False)
    reference_average = channels.mean(-2)  # Terminal [N,C].
    ssim = tp.image_map_operator(x, y, comparison='ssim', data_range=1.,
                                 window_size=3, chunk_size=2)
    smoothed = (4. * ssim.mean()).softmax_matmul(values)
    loss = ssim.var((-4,-2,-1), correction=0).mean() + smoothed.square().mean() + reference_average.mean()
    loss.backward()
    print('image map:', tuple(maps.shape), 'events:', tuple(maps.event_shape))
    print('nearest values/indices:', tuple(nearest.values.shape), tuple(nearest.indices.shape))
    print('smoothed payload:', tuple(smoothed.shape), 'loss:', loss.item())
    assert all(t.grad is not None and t.grad.isfinite().all() for t in (x, y, values))


if __name__ == '__main__':
    main()
