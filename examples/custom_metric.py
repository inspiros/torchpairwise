"""Vectorize a scalar PyTorch comparison, explicit parameters and native streaming."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = torch.linspace(.5, 1.5, 4, device=device, requires_grad=True)

    def score(a, b, weight):
        return (weight * torch.log1p((a - b).square())).sum()

    compare = tp.make_pairwise(score, parameters=(weights,), chunk_size=16, compile=True, symmetric=True)
    x = torch.tensor([[1., 2., 3., 4.], [4., 2., 1., 3.], [3., 1., 4., 2.]],
                     device=device, requires_grad=True)
    y = x.detach()[:2].flip(0).clone().requires_grad_()
    matrix = compare(x, y)
    aligned = compare(x[:2], y, mode="paired")
    triangle = compare(x, mode="condensed")
    single = compare(x[0], y[0], mode="single")
    A = tp.lazy(x, y, compare, chunk_size=2)
    torch.testing.assert_close(A.mean(-1), matrix.mean(-1))
    print('custom all-pairs:', matrix.detach(), 'paired:', aligned.detach(),
          'condensed:', triangle.detach(), 'single:', single.detach())
    for gradient in torch.autograd.grad(matrix.sum() + aligned.sum() + triangle.sum(), (x, y, weights)):
        assert torch.isfinite(gradient).all()
    image_compare = tp.make_pairwise(lambda a, b: (a - b).abs().mean(), sample_ndim=3)
    images = torch.rand(2, 3, 4, 5, device=device)
    assert image_compare(images).shape == (2, 2)
    print('Custom vector/image comparisons and explicit-parameter gradients passed on', device)


if __name__ == '__main__':
    main()
