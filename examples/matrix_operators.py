"""One matrix interface for comparisons, kernels and caller-owned values."""
import argparse
import torch
import torchpairwise as tp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    device = parser.parse_args().device
    torch.manual_seed(115)
    x, y = torch.rand(3, 2, device=device), torch.rand(4, 2, device=device)
    values = torch.rand(4, 2, device=device, requires_grad=True)
    k = tp.lazy(x, y, metric='rbf', gamma=.5, chunk_size=2)
    a = tp.lazy(x, y, metric='rbf_kernel', gamma=.5, chunk_size=2)
    assert isinstance(k, tp.LazyPairwise)
    torch.testing.assert_close(k.matmul(values), a.matmul(values))
    scores = torch.randn(3, 4, device=device, requires_grad=True)
    supplied = tp.as_lazy(scores, chunk_size=2)
    actual = supplied.matmul(values)
    torch.testing.assert_close(actual, scores @ values)
    assert all(g.isfinite().all() for g in torch.autograd.grad(actual.sum(), (scores, values)))
    print('same twelve consumers; kernel specialization adds vector geometry')
    print('caller-owned matrix result', tuple(actual.shape), supplied.sum().item())


if __name__ == '__main__':
    main()
