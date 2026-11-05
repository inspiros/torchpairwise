"""Compile a weighted metric and serialize an export with dynamic sample axes."""
import tempfile
from pathlib import Path
import torch
import torchpairwise as tp


class WeightedComparison(torch.nn.Module):
    def forward(self, x, y, weights):
        return tp.cosine_distances(x, y, w=weights)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x, y = [torch.rand(n, 8, device=device) for n in (3, 5)]
    weights = torch.rand(8, device=device) + .1
    model = WeightedComparison()
    compiled = torch.compile(model, fullgraph=True)
    torch.testing.assert_close(compiled(x, y, weights), model(x, y, weights))
    exported = torch.export.export(model, (x, y, weights), dynamic_shapes=(
        {0: torch.export.Dim('n', min=2)}, {0: torch.export.Dim('m', min=2)}, None))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'weighted.pt2'
        torch.export.save(exported, path)
        restored = torch.export.load(path).module()
        # Changing N and M does not require another export.
        x, y = [torch.rand(n, 8, device=device, requires_grad=True) for n in (4, 7)]
        weights.requires_grad_()
        output = restored(x, y, weights)
        torch.testing.assert_close(output, model(x, y, weights))
        output.sum().backward()
        assert all(value.grad is not None for value in (x, y, weights))
    print(f'Compiled and exported weighted cosine passed on {device}')


if __name__ == '__main__':
    main()
