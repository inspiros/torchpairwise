import pytest
import torch
import os


def pytest_addoption(parser):
    parser.addoption('--device', choices=('all', 'cpu', 'cuda'), default='all',
                     help='Select devices before scheduling tests')


def pytest_generate_tests(metafunc):
    if 'device' in metafunc.fixturenames:
        selected = metafunc.config.getoption('--device')
        devices = ['cpu', 'cuda'] if selected == 'all' else [selected]
        metafunc.parametrize('device', devices, indirect=True)


def pytest_collection_modifyitems(config, items):
    if config.getoption('--device') == 'cpu':
        excluded = [item for item in items if item.get_closest_marker('cuda')]
        items[:] = [item for item in items if not item.get_closest_marker('cuda')]
        config.hook.pytest_deselected(items=excluded)


@pytest.fixture
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    return request.param


@pytest.fixture(autouse=True, scope='session')
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1 if os.environ.get('PYTEST_XDIST_WORKER') else 2)
    yield
    torch.set_num_threads(previous)
