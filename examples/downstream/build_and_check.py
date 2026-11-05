"""Build two downstream C++ extensions on top of torchpairwise and check them.

dispatcher_ops.cpp calls torchpairwise operators by schema and needs only the
PyTorch headers. header_ops.cpp includes <torchpairwise.h> and links the
installed library, so it is tied to this exact torchpairwise build (Linux).
Both need a C++ compiler; header_ops.cpp needs C++20.
"""
import argparse
from pathlib import Path

import torch
import torchpairwise as tp
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent


def build(build_directory=None, verbose=False):
    def directory(name):
        if build_directory is None:
            return None  # PyTorch's default extension cache
        path = Path(build_directory)/name
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    # Loading registers torch.ops.dispatcher_consumer and torch.ops.header_consumer.
    load(name='dispatcher_consumer', sources=[str(HERE/'dispatcher_ops.cpp')],
         extra_cflags=['-O2'], build_directory=directory('dispatcher'),
         is_python_module=False, verbose=verbose)
    load(name='header_consumer', sources=[str(HERE/'header_ops.cpp')],
         extra_include_paths=[tp.get_include()], extra_ldflags=[tp.library_path()],
         extra_cflags=['-std=c++20', '-O2'], build_directory=directory('header'),
         is_python_module=False, verbose=verbose)


def check(device='cpu'):
    options = dict(dtype=torch.float64, device=device)
    x = torch.randn(64, 5, **options, requires_grad=True)
    centers = torch.randn(200, 5, **options, requires_grad=True)
    coefficients = torch.randn(200, 3, **options, requires_grad=True)
    inputs = (x, centers, coefficients)
    reference = tp.rbf_kernel(x, centers, gamma=.5) @ coefficients
    dispatched = torch.ops.dispatcher_consumer.rbf_predict(x, centers, coefficients, .5)
    streamed = torch.ops.header_consumer.rbf_predict(x, centers, coefficients, .5, 32)
    expected = torch.autograd.grad(reference.sum(), inputs)
    for value in (dispatched, streamed):
        torch.testing.assert_close(value, reference)
        for grad, want in zip(torch.autograd.grad(value.sum(), inputs), expected):
            torch.testing.assert_close(grad, want)
    return tuple(reference.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--build-dir', default=None)
    parser.add_argument('--verbose', action='store_true')
    options = parser.parse_args()
    build(options.build_dir, options.verbose)
    shape = check(options.device)
    print('Dispatcher and header extensions match torchpairwise:', shape, 'with gradients')


if __name__ == '__main__':
    main()
