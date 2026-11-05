"""Compose centered kernel maps and diagonal shifts without forming their product."""
import argparse

import torch
import torchpairwise as tp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(argv)
    generator = torch.Generator(device=args.device).manual_seed(42)
    def tensor(shape):
        return torch.randn(shape, device=args.device, dtype=torch.float64,
                           generator=generator, requires_grad=True)
    x, y = tensor((11, 3)), tensor((17, 3))
    values = tensor((11, 2))
    regularization = torch.tensor(.1, device=args.device, dtype=x.dtype, requires_grad=True)
    kernel = tp.lazy(x, y, metric='rbf', gamma=.2, chunk_size=4)
    a = kernel.centered()       # H_11 K H_17
    normal = a @ a.mT
    normal = normal + regularization * tp.eye_like(normal) # logical [11,11]
    result = normal @ values                      # [11,2]; no K or A A.T stored

    # Small dense reference only. The streamed computation above applies a.T
    # then a to vector blocks and adds the scaled RHS.
    dense = tp.rbf_kernel(x, y, gamma=.2)
    dense = dense - dense.mean(0, keepdim=True) - dense.mean(1, keepdim=True) + dense.mean()
    expected = dense @ (dense.T @ values) + regularization * values
    torch.testing.assert_close(result, expected)
    operands = (x, y, values, regularization)
    torch.testing.assert_close(torch.autograd.grad(result.square().sum(), operands),
                               torch.autograd.grad(expected.square().sum(), operands))
    print('Linear map:', list(normal.shape), 'product:', list(result.shape))
    print('Values and first gradients match on', args.device)


if __name__ == '__main__':
    main()
