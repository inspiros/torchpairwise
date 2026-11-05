"""Build a user-owned native metric and bind it in Python; no Python numerical callback.

Requires this source checkout built against the active PyTorch and a C++ compiler.
Build output stays in develop/; this is separate from a standalone CMake example.
"""
import argparse
from pathlib import Path
import torch
import torchpairwise as tp
from torch.utils.cpp_extension import load


def load_example(build_directory=None):
    root = Path(__file__).resolve().parents[1]
    directory = root/'develop/native-custom-example' if build_directory is None else Path(build_directory)
    directory.mkdir(parents=True, exist_ok=True)
    return load(name='torchpairwise_custom_example',
                sources=[str(root/'examples/native_custom_metric.cpp')],
                extra_include_paths=[tp.get_include()],
                extra_cflags=['-std=c++20', '-O2'],
                extra_ldflags=[tp.library_path()],
                build_directory=str(directory), is_python_module=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    options = parser.parse_args()
    load_example()
    x = torch.randn(3,4,device=options.device,requires_grad=True)
    y = torch.randn(5,4,device=options.device,requires_grad=True)
    weights = torch.ones(4,device=options.device,requires_grad=True)
    metric = tp.make_pairwise('torchpairwise_example::weighted_pairs',
        backward='torchpairwise_example::weighted_pairs_vjp', parameters=(weights,), symmetric=True)
    values = metric(x,y)                       # [3,5]
    row_mean = tp.lazy(x,y,metric).mean(-1)    # [3]
    native_mean = torch.ops.torchpairwise_example.streamed_mean(x,y,weights)
    torch.testing.assert_close(row_mean,values.mean(-1))
    torch.testing.assert_close(native_mean,row_mean)
    row_mean.sum().backward()
    print('Dense:',tuple(values.shape),'Streamed:',tuple(row_mean.shape))
    print('Input and parameter gradients:',all(t.grad is not None for t in (x,y,weights)))


if __name__ == '__main__':
    main()
