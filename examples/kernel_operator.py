"""Scalar reductions, bilinear forms and vector fields without a dense Gram matrix."""
import torch
import torchpairwise as tp


def main():
    torch.manual_seed(817)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(11, 3, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.randn(17, 3, dtype=x.dtype, device=device, requires_grad=True)
    u = torch.randn(11, dtype=x.dtype, device=device, requires_grad=True)
    v = torch.randn(17, dtype=x.dtype, device=device, requires_grad=True)
    kernel = tp.lazy(x, y, metric='rbf', gamma=.2, chunk_size=4)
    total = kernel.sum()
    form = kernel.bilinear(u, v)
    # Sum_j k(x_i,y_j) (y_j - x_i), using only two kernel products.
    field = kernel @ y - x * kernel.sum(dim=-1, keepdim=True)
    # Dense values below are small reference calculations only.
    gram = tp.rbf_kernel(x, y, gamma=.2)
    expected = (gram.unsqueeze(-1) * (y[None] - x[:, None])).sum(1)
    torch.testing.assert_close(total, gram.sum())
    torch.testing.assert_close(form, u @ gram @ v)
    torch.testing.assert_close(field, expected)
    torch.testing.assert_close(kernel.mean(dim=-2), gram.mean(-2))
    torch.testing.assert_close(kernel.diagonal(), gram.diagonal())
    self_kernel = tp.lazy(x, metric='linear', chunk_size=4)
    torch.testing.assert_close(self_kernel.quadratic(u), u @ (x @ x.T) @ u)
    actual_grad = torch.autograd.grad(total + form + field.square().sum(), (x, y, u, v))
    reference_grad = torch.autograd.grad(gram.sum() + u @ gram @ v + expected.square().sum(), (x, y, u, v))
    for actual, reference in zip(actual_grad, reference_grad):
        torch.testing.assert_close(actual, reference)
    print('Streamed kernel reductions, forms and displacement field passed on', device)


if __name__ == '__main__':
    main()
