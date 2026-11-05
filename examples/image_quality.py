"""Whole aligned-image quality scores in all three comparison modes."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(71)
    x = torch.rand(3, 3, 32, 32, device=device, requires_grad=True)
    y = (x.detach()[:2] * .8 + .1).requires_grad_()
    loss = x.new_zeros(())
    for name in ('psnr', 'ssim', 'multiscale_ssim'):
        # Two MS scales keep this example small; the default has five scales.
        options = {'data_range': 1.}
        if name == 'multiscale_ssim':
            options['weights'] = (.5, .5)
        matrix = getattr(tp, name + '_scores')(x, y, **options)
        paired = getattr(tp, 'paired_' + name + '_scores')(x[:2], y, **options)
        condensed = getattr(tp, 'condensed_' + name + '_scores')(x, **options)
        print(name, 'all-pairs:', matrix.detach(), 'paired:', paired.detach(),
              'condensed:', condensed.detach())
        loss = loss + matrix.sum() + paired.sum() + condensed.sum()
    for gradient in torch.autograd.grad(loss, (x, y)):
        assert torch.isfinite(gradient).all()
    print('Whole-image quality scores and gradients passed on', device)


if __name__ == '__main__':
    main()
