"""Native streamed image scores and audio distances with reusable features."""
import torch
import torchpairwise as tp


def main():
    torch.manual_seed(9)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    images = torch.rand(3, 2, 12, 12, device=device, requires_grad=True)
    references = torch.rand(5, 2, 12, 12, device=device, requires_grad=True)
    values = torch.randn(5, 2, device=device, requires_grad=True)
    operator = tp.lazy(images, references, metric='ssim',
                                    data_range=1., window_size=3, chunk_size=2)
    scores = operator.matmul(values)
    scores.square().mean().backward()
    assert all(t.grad.isfinite().all() for t in (images, references, values))
    audio = torch.randn(3, 2, 64, device=device)
    targets = torch.randn(5, 2, 64, device=device)
    spectral = tp.lazy(audio, targets, metric='log_spectral',
                                    n_fft=16, hop_length=4, chunk_size=2)
    print('SSIM matrix:', tuple(operator.shape), 'product:', tuple(scores.shape))
    print('Audio column means:', tuple(spectral.mean(-2).shape))
    print('finite image/coefficient gradients: True')


if __name__ == '__main__':
    main()
