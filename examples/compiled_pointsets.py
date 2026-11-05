"""Compile ragged clouds and export dynamic cloud/point counts."""
import tempfile
from pathlib import Path
import torch
import torchpairwise as tp


class Clouds(torch.nn.Module):
    def forward(self, x, y, x_lengths, y_lengths):
        return tp.chamfer_distances(x, y, x1_lengths=x_lengths, x2_lengths=y_lengths, chunk_size=2)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def inputs(n, m, p, q):
        x, y = torch.rand(n, p, 3, device=device), torch.rand(m, q, 3, device=device)
        xl = torch.full((n,), p - 1, dtype=torch.int64, device=device)
        yl = torch.full((m,), q - 1, dtype=torch.int64, device=device)
        x[:, -1], y[:, -1] = float('nan'), float('nan')  # Ignored padding.
        return x.requires_grad_(), y.requires_grad_(), xl, yl
    args = inputs(3, 4, 5, 6)
    module = Clouds()
    compiled = torch.compile(module, fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(*args), module(*args))
    n, m, p, q = [torch.export.Dim(name, min=2) for name in ('n', 'm', 'p', 'q')]
    exported = torch.export.export(module, args, dynamic_shapes=(
        {0: n, 1: p}, {0: m, 1: q}, {0: n}, {0: m}))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'pointsets.pt2'
        torch.export.save(exported, path)
        restored = torch.export.load(path).module()
        args = inputs(5, 6, 7, 8)
        actual = restored(*args)
        torch.testing.assert_close(actual, module(*args))
        actual.sum().backward()
        assert all(value.grad.isfinite().all() for value in args[:2])
    print(f'Compiled/exported ragged point sets passed on {device}')


if __name__ == '__main__':
    main()
