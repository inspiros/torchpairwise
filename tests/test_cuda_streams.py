import pytest
import torch
import torchpairwise as tp

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize('metric', ['canberra', 'braycurtis', 'sqeuclidean', 'snr',
                                    'jensenshannon', 'cosine', 'haversine', 'chamfer'])
def test_native_follows_current_stream(metric):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    d = 2 if metric == 'haversine' else 4
    x = torch.full((5, d), .2, device='cuda', dtype=torch.float64)
    y = torch.full((3, d), .7, device='cuda', dtype=torch.float64)
    # Nonconstant rows make SNR/correlation references well-defined.
    x += torch.arange(d, device='cuda', dtype=x.dtype) * .03
    y += torch.arange(d, device='cuda', dtype=y.dtype) * .07
    if metric == 'chamfer':
        x, y = x.unsqueeze(-2), y.unsqueeze(-2).expand(-1, 65, -1)
    original = x.clone()
    expected = tp.cdist(x * 2, y, metric=metric)
    expected_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(expected_stream)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(3_000_000)
        x.mul_(2)
        result = tp.cdist(x, y, metric=metric)
        consumed = result.clone()
    expected_stream.wait_stream(stream)
    torch.testing.assert_close(consumed, expected, equal_nan=True)
    torch.testing.assert_close(x, original * 2)


@pytest.mark.parametrize('metric', ['canberra', 'braycurtis', 'sqeuclidean', 'jensenshannon', 'snr', 'chamfer'])
def test_native_backward_follows_current_stream(metric):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    g = torch.Generator().manual_seed(18)
    x = (torch.rand(3, 4, generator=g, dtype=torch.float64) + .2).cuda().requires_grad_()
    y = (torch.rand(2, 4, generator=g, dtype=torch.float64) + .4).cuda().requires_grad_()
    if metric == 'chamfer':
        x, y = x.unsqueeze(-2), y.unsqueeze(-2).expand(-1, 65, -1)
    reference = torch.autograd.grad(tp.cdist(x, y, metric=metric).sum(), (x, y))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        result = torch.autograd.grad(tp.cdist(x, y, metric=metric).sum(), (x, y))
        consumed = tuple(value.clone() for value in result)
    torch.cuda.current_stream().wait_stream(stream)
    for actual, expected in zip(consumed, reference):
        torch.testing.assert_close(actual, expected)
