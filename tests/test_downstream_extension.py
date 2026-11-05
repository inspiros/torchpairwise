"""Downstream C++ extensions built on an installed torchpairwise."""
import os
import runpy
import shutil
import sys
from pathlib import Path

import pytest
import torchpairwise as tp

EXAMPLE = Path(__file__).resolve().parents[1]/'examples/downstream/build_and_check.py'


def test_native_include_and_library_helpers():
    assert (Path(tp.get_include())/'torchpairwise.h').is_file()
    assert (Path(tp.get_include())/'ops/ops.h').is_file()
    assert Path(tp.library_path()).samefile(tp.extension._get_extension_path('_C'))


@pytest.fixture(scope='module')
def example(tmp_path_factory):
    if not sys.platform.startswith('linux'):
        pytest.skip('header linking is checked on Linux')
    if shutil.which(os.environ.get('CXX', 'c++')) is None:
        pytest.skip('needs a C++ compiler')
    namespace = runpy.run_path(str(EXAMPLE))
    # Built once: a second library would register the same operator namespaces.
    namespace['build'](tmp_path_factory.mktemp('downstream'))
    return namespace


@pytest.mark.compiler
def test_downstream_dispatcher_and_header_extensions(example, device):
    assert example['check'](device) == (64, 3)
