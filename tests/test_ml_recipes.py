"""Recipes from the ML backend guide against dense autograd/PyTorch oracles."""
import pytest
import torch
import torch.nn.functional as F
import torchpairwise as tp


def svgd_direction(particles, score, *, gamma):
    """Stein variational gradient direction with an RBF kernel, [N,D].

    phi_i = mean_j k(x_j,x_i)*score_j + grad_{x_j} k(x_j,x_i). For RBF the
    second term is 2*gamma*sum_j k_ij*(x_i-x_j) = -2*gamma*displacement_i.
    """
    kernel = tp.lazy(particles, particles, metric='rbf', gamma=gamma)
    return (kernel @ score - 2 * gamma * kernel.displacement()) / particles.shape[-2]


def info_nce(z, *, temperature):
    """NT-Xent over 2B views, where rows i and i+B are positives; scalar."""
    z = F.normalize(z, dim=-1)
    similarity = tp.lazy(z, z, metric='linear') / temperature
    positives = (z * z.roll(z.shape[0] // 2, 0)).sum(-1) / temperature
    return (similarity.logsumexp(-1, exclude_self=True) - positives).mean()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_svgd_direction_matches_autograd(device, dtype):
    torch.manual_seed(5)
    x = torch.randn(37, 3, device=device, dtype=dtype)
    score = -x  # standard normal target
    gamma = .7
    leaf = x.clone().requires_grad_()
    kernel = torch.exp(-gamma * (leaf[:, None] - x[None]).square().sum(-1))
    # Radial kernels give grad_{x_j} k(x_j,x_i) = -grad_{x_i} k(x_i,x_j).
    attraction, = torch.autograd.grad(kernel.sum(), leaf)
    expected = (kernel.detach() @ score - attraction) / x.shape[0]
    tolerance = dict(rtol=1e-4, atol=1e-5) if dtype == torch.float32 else {}
    torch.testing.assert_close(svgd_direction(x, score, gamma=gamma), expected, **tolerance)


def test_svgd_moves_particles_toward_target(device):
    torch.manual_seed(6)
    x = torch.randn(64, 2, device=device, dtype=torch.float64) * .2 + 3
    for _ in range(300):
        x = x + .3 * svgd_direction(x, -x, gamma=.5)
    assert x.mean(0).norm() < .3
    assert .3 < x.var(0).mean() < 2


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_info_nce_matches_dense(device, dtype):
    torch.manual_seed(7)
    z = torch.randn(16, 8, device=device, dtype=dtype, requires_grad=True)
    loss = info_nce(z, temperature=.2)
    n = F.normalize(z, dim=-1)
    logits = (n @ n.T / .2).masked_fill(torch.eye(16, dtype=torch.bool, device=device), -torch.inf)
    expected = F.cross_entropy(logits, torch.arange(16, device=device).roll(8))
    tolerance = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else {}
    torch.testing.assert_close(loss, expected, **tolerance)
    torch.testing.assert_close(torch.autograd.grad(loss, z)[0],
                               torch.autograd.grad(expected, z)[0], **tolerance)
