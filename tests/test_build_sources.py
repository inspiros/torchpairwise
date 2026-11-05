"""Keep both native build front ends complete when new operators are added."""
import json
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest


def test_shared_manifest_covers_every_native_translation_unit():
    root = Path(__file__).resolve().parents[1] / 'torchpairwise' / 'csrc'
    groups = json.loads((root / 'sources.json').read_text())
    listed = [name for names in groups.values() for name in names]
    actual = {path.relative_to(root).as_posix() for path in root.rglob('*')
              if path.suffix in ('.cpp', '.cu')}
    assert set(groups) == {'common', 'cpu', 'cuda'}
    assert len(listed) == len(set(listed))
    assert set(listed) == actual


def test_cuda_architecture_target_parsing(tmp_path):
    root = Path(__file__).resolve().parents[1]
    parse = runpy.run_path(str(root / 'setup.py'))['cuda_arch_targets']
    cases = [
        ([], [], ''),
        (['-gencode=arch=compute_75,code=sm_75'], ['sm_75'], 'OFF'),
        (['--generate-code', 'arch=compute_90a,code=[sm_90a,compute_90a]'],
         ['compute_90a', 'sm_90a'], 'OFF'),
        (['-gencode=arch=compute_75,code=sm_75', '-gencode=arch=compute_75,code=compute_75'],
         ['compute_75', 'sm_75'], 'OFF'),
        (['-arch=native'], [], 'OFF'),
        (['--gpu-code', 'sm_80'], [], 'OFF'),
        (['-gencode=arch=compute_80,code=[sm_80,lto_80]'], [], 'OFF'),
    ]
    for flags, expected, _ in cases:
        assert parse(flags) == expected
    cmake = shutil.which('cmake')
    if cmake is None:
        pytest.skip('CMake is needed for the build-metadata script check')
    # Run only a CMake script: no configure, native build, CTest or LibTorch test.
    lines = [f'include("{root.as_posix()}/cmake/CudaArchitectureInfo.cmake")']
    native_cases = [(flags, expected, arch) for flags, expected, arch in cases]
    native_cases += [([], ['compute_75', 'compute_90a', 'sm_75', 'sm_80'],
                      '75;80-real;90a-virtual'),
                     ([], [], 'native'), ([], [], 'all')]
    for flags, expected, arch in native_cases:
        lines += [f'torchpairwise_cuda_arch_targets(actual "{" ".join(flags)}" "{arch}")',
                  f'if(NOT actual STREQUAL "{" ".join(expected)}")',
                  '  message(FATAL_ERROR "Unexpected targets: ${actual}")', 'endif()']
    script = tmp_path / 'architectures.cmake'
    script.write_text('\n'.join(lines) + '\n')
    subprocess.run([cmake, '-P', str(script)], check=True, capture_output=True, text=True)


def test_cuda_architecture_runtime_metadata():
    import re
    import torch
    from torchpairwise import extension

    targets = extension.cuda_arch_list()
    assert targets == sorted(set(targets))
    assert all(re.fullmatch(r'(sm|compute)_\d+[a-z]?', item) for item in targets)
    assert torch.ops.torchpairwise._cuda_arch_flags() == ' '.join(targets)
    if not extension.with_cuda():
        assert targets == []
