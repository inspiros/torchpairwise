"""Sequence/token/statistical comparisons without retaining their dense matrices."""
import argparse
import torch
import torchpairwise as tp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    device = parser.parse_args().device
    torch.manual_seed(108)
    x = torch.rand(4, 5, 2, device=device, requires_grad=True)
    y = torch.rand(6, 5, 2, device=device, requires_grad=True)
    values = torch.rand(6, 3, device=device, requires_grad=True)
    for metric, kw in (('soft_dtw', dict(gamma=.5)),
                       ('mmd_statistic', dict(estimator='unbiased', kernel='polynomial'))):
        a = tp.lazy(x, y, metric=metric, chunk_size=2, **kw)
        result = a.matmul(values)
        torch.testing.assert_close(result, tp.compare(x, y, metric=metric, **kw) @ values)
        assert all(g.isfinite().all() for g in torch.autograd.grad(result.sum(), (x, y, values)))
        print(metric, tuple(result.shape), a.mean().item())
    tokens = torch.tensor([[2**60, 2**60 + 1], [2**60 + 2, 2**60]], device=device)
    edits = tp.lazy(tokens, metric='levenshtein', chunk_size=1)
    assert edits.sum().dtype == torch.int64 and edits.mean().dtype == torch.float64
    print('token edit total', edits.sum().item())


if __name__ == '__main__':
    main()
