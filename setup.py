import json
import os
import re
import sys
import shlex
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import _get_cuda_arch_flags

PACKAGE_ROOT = 'torchpairwise'


def cuda_arch_targets(flags):
    """Read output targets from resolved nvcc gencode flags, never input archs.

    Other custom nvcc architecture controls may depend on the build machine or
    toolkit defaults. Report unknown rather than guessing their output targets.
    """
    text = ' '.join(flags)
    if re.search(r'(?:^|\s)(?:-arch|--gpu-architecture|-code|--gpu-code)(?:=|\s)', text):
        return []
    codes = re.findall(r'\bcode\s*=\s*(\[[^\]]*\]|[^,\s]+)', text)
    targets = set()
    for code in codes:
        for target in code.strip('[]').split(','):
            target = target.strip()
            if not re.fullmatch(r'(?:sm|compute)_\d+[a-z]?', target):
                return []
            targets.add(target)
    return sorted(targets)


def use_py_limited_api():
    # The extension is loaded through torch.ops and uses no CPython API beyond
    # an empty Windows module initializer, so release wheels can target the
    # stable ABI: one wheel per platform serves every supported Python.
    # Opt-in, because in-place builds for several interpreters would otherwise
    # share one _C.abi3 file name.
    mode = os.getenv('TORCHPAIRWISE_PY_LIMITED_API', '0')
    if mode not in ('0', '1'):
        raise ValueError('TORCHPAIRWISE_PY_LIMITED_API must be 0 or 1')
    return mode == '1'


def py_limited_api_tag():
    # torch compiles limited-API extensions with Py_LIMITED_API set to this.
    version = int(torch.utils.cpp_extension.min_supported_cpython, 16)
    return f'cp{version >> 24}{(version >> 16) & 0xff}'


def get_version(version_file='_version.py'):
    import importlib.util
    version_file_path = os.path.abspath(os.path.join(PACKAGE_ROOT, version_file))
    try:
        spec = importlib.util.spec_from_file_location('_version', version_file_path)
        version_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(version_module)
        version = str(version_module.__version__)
    except:
        version = '0.0.0'
    # A local label such as cu132 marks wheels for a non-default torch variant.
    # PyPI rejects local versions, so these wheels ship outside PyPI.
    local = os.getenv('TORCHPAIRWISE_LOCAL_VERSION', '')
    if local:
        if not re.fullmatch(r'[a-z0-9]+(?:\.[a-z0-9]+)*', local):
            raise ValueError('TORCHPAIRWISE_LOCAL_VERSION must be a PEP 440 local label, e.g. cu132')
        version += f'+{local}'
    return version


def get_parallel_options(backend=None):
    parallel_extra_compile_args = []
    parallel_define_macros = []

    if backend is not None:
        if backend.upper() not in ['OPENMP', 'NATIVE', 'NATIVE_TBB']:
            raise ValueError('Parallel backend options are OPENMP, NATIVE, or NATIVE_TBB. '
                             f'Got unknown backend {backend}.')
    else:  # detect torch parallel backend
        parallel_info_string = torch.__config__.parallel_info()
        parallel_info_array = parallel_info_string.splitlines()
        backend_lines = [line for line in parallel_info_array
                         if line.startswith('ATen parallel backend:')]
        if len(backend_lines):
            backend = backend_lines[0].rsplit(': ')[1]

    backend = backend.lower() if backend is not None else ''
    if backend == 'openmp':
        parallel_define_macros += [('AT_PARALLEL_OPENMP', None)]
        if sys.platform == 'darwin':
            parallel_extra_compile_args.append('-Xpreprocessor')
        parallel_extra_compile_args.append('/openmp' if sys.platform == 'win32' else '-fopenmp')
        if sys.platform == 'darwin':
            parallel_extra_compile_args.append('-lomp')
    elif backend.startswith('native'):
        if backend.endswith('tbb'):
            parallel_define_macros += [('AT_PARALLEL_NATIVE_TBB', None)]
        else:
            parallel_define_macros += [('AT_PARALLEL_NATIVE', None)]
    return parallel_extra_compile_args, parallel_define_macros


def get_extensions():
    extensions_dir = os.path.join(PACKAGE_ROOT, 'csrc')

    # Shared with CMake so Python and standalone LibTorch build the same code.
    source_groups = json.loads((Path(extensions_dir) / 'sources.json').read_text())
    sources = [str(Path(extensions_dir) / name)
               for group in ('common', 'cpu') for name in source_groups[group]]
    source_cuda = [str(Path(extensions_dir) / name) for name in source_groups['cuda']]
    extension = CppExtension
    extra_compile_args = {'cxx': []}
    define_macros = []

    print('Compiling extensions with following flags:')
    force_cuda = os.getenv('FORCE_CUDA', '0') == '1'
    print(f'  FORCE_CUDA: {force_cuda}')
    debug_mode = os.getenv('DEBUG', '0') == '1'
    print(f'  DEBUG: {debug_mode}')

    nvcc_flags = os.getenv('NVCC_FLAGS', '')
    print(f'  NVCC_FLAGS: {nvcc_flags}')
    py_limited_api = use_py_limited_api()
    print(f'  TORCHPAIRWISE_PY_LIMITED_API: {py_limited_api}')

    # enable cpu parallel
    parallel_extra_compile_args, parallel_define_macros = get_parallel_options()
    extra_compile_args['cxx'] += parallel_extra_compile_args
    define_macros += parallel_define_macros

    # enable cuda
    cuda_mode = os.getenv('TORCHPAIRWISE_BUILD_CUDA', 'auto').lower()
    if cuda_mode not in ('auto', '0', '1'):
        raise ValueError('TORCHPAIRWISE_BUILD_CUDA must be auto, 0, or 1')
    build_cuda = force_cuda or cuda_mode == '1' or (
        cuda_mode == 'auto' and CUDA_HOME is not None and torch.version.cuda is not None)
    if build_cuda and CUDA_HOME is None:
        raise RuntimeError('A CUDA toolkit is required for a CUDA build; set CUDA_HOME')
    if build_cuda:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        if nvcc_flags == '':
            nvcc_flags = []
        else:
            nvcc_flags = shlex.split(nvcc_flags)
        # Resolve once and pass these exact flags to nvcc as well as recording
        # their output targets. This includes PyTorch's named/default arches.
        nvcc_flags += _get_cuda_arch_flags(nvcc_flags)
        extra_compile_args['nvcc'] = nvcc_flags
        targets = ' '.join(cuda_arch_targets(nvcc_flags))
        prefix = '/D' if sys.platform == 'win32' else '-D'
        extra_compile_args['cxx'].append(
            f'{prefix}TORCHPAIRWISE_CUDA_ARCH_FLAGS="{targets}"')

    if sys.platform == 'win32':
        define_macros += [(f'{PACKAGE_ROOT}_EXPORTS', None)]
        define_macros += [('USE_PYTHON', None)]
        extra_compile_args['cxx'].append('/MP')
        # Operator registrations and dispatch templates exceed MSVC's default
        # section limit (C1128) in the larger translation units.
        extra_compile_args['cxx'].append('/bigobj')
        if 'nvcc' in extra_compile_args:
            extra_compile_args['nvcc'] += ['-Xcompiler', '/bigobj']

    if debug_mode:
        print('Compiling in debug mode')
        extra_compile_args['cxx'] += ['/Zi', '/Od'] if sys.platform == 'win32' else ['-g', '-O0']
        if 'nvcc' in extra_compile_args:
            # we have to remove '-OX' and '-g' flag if exists and append
            nvcc_flags = extra_compile_args['nvcc']
            extra_compile_args['nvcc'] = [f for f in nvcc_flags if not ('-O' in f or '-g' in f)]
            extra_compile_args['nvcc'].append('-O0')
            extra_compile_args['nvcc'].append('-g')

    include_dirs = [os.path.abspath(extensions_dir)]
    ext_modules = [
        extension(
            f'{PACKAGE_ROOT}._C',
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            py_limited_api=py_limited_api,
        )
    ]
    return ext_modules


def torch_requirement():
    major, minor = (int(part) for part in torch.__version__.split('.')[:2])
    return f'torch>={major}.{minor},<{major}.{minor + 1}'


def setup_package():
    options = {}
    if use_py_limited_api():
        options['bdist_wheel'] = {'py_limited_api': py_limited_api_tag()}
    setup(
        options=options,
        version=get_version(),
        # Native wheels must stay within the torch minor they were built against.
        install_requires=[torch_requirement()],
        ext_modules=get_extensions(),
        cmdclass={
            'build_ext': torch.utils.cpp_extension.BuildExtension
        },
    )


if __name__ == '__main__':
    setup_package()
