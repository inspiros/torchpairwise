"""Masked streaming consumers for ML losses and graph steps, with dense checks.

Run: python examples/pair_masks.py --device cuda
Inputs are unbatched embeddings [N,D] with integer labels [N]. The dense
references are small validation oracles, not part of the streamed algorithms.
"""
import argparse

import torch
import torchpairwise as tp


def supervised_contrastive_loss(z, labels, *, temperature=.1, chunk_size=128):
    """SupCon loss (Khosla et al., 2020), averaging log-probabilities over positives.

    Uses the anchor-excluded softmax denominator; every label needs two samples.
    """
    z = torch.nn.functional.normalize(z, dim=-1)
    scores = tp.lazy(z, metric='linear', chunk_size=chunk_size) * (1 / temperature)
    positives = tp.masks.same_group(labels) & ~tp.masks.diagonal()
    log_normalizer = scores.logsumexp(-1, exclude_self=True)       # [N]
    mean_positive = scores.mean(-1, mask=positives)                # [N]
    return (log_normalizer - mean_positive).mean()


def dense_supervised_contrastive_loss(z, labels, *, temperature=.1):
    z = torch.nn.functional.normalize(z, dim=-1)
    scores = z @ z.T / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = (labels[:, None] == labels[None]) & ~eye
    log_normalizer = scores.masked_fill(eye, -float('inf')).logsumexp(-1)
    mean_positive = (scores * positives).sum(-1) / positives.sum(-1)
    return (log_normalizer - mean_positive).mean()


def batch_hard_triplet_loss(z, labels, *, margin=.2, chunk_size=128):
    """Hardest positive/negative per anchor (Hermans et al., 2017)."""
    D = tp.lazy(z, metric='euclidean', chunk_size=chunk_size)
    hardest_positive = D.amax(-1, mask=tp.masks.same_group(labels) & ~tp.masks.diagonal())
    hardest_negative = D.amin(-1, mask=tp.masks.different_group(labels))
    return torch.relu(hardest_positive - hardest_negative + margin).mean()


def dense_batch_hard_triplet_loss(z, labels, *, margin=.2):
    D = torch.cdist(z, z)
    same = labels[:, None] == labels[None]
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    hardest_positive = D.masked_fill(~same | eye, -float('inf')).amax(-1)
    hardest_negative = D.masked_fill(same, float('inf')).amin(-1)
    return torch.relu(hardest_positive - hardest_negative + margin).mean()


def nearest_other_component(x, components, *, chunk_size=128):
    """One Boruvka step: each point's nearest point in another component."""
    D = tp.lazy(x, metric='sqeuclidean', chunk_size=chunk_size)
    return D.min(-1, mask=tp.masks.different_group(components))


def causal_kernel_attention(q, k, v, *, gamma=1., chunk_size=128):
    """softmax over j <= i of -gamma * ||q_i - k_j||**2, applied to v."""
    scores = -gamma * tp.lazy(q, k, metric='sqeuclidean', chunk_size=chunk_size)
    return scores.softmax_matmul(v, mask=tp.masks.causal())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    torch.manual_seed(0)
    device, dtype = torch.device(args.device), torch.float64
    z = torch.randn(96, 8, device=device, dtype=dtype, requires_grad=True)
    labels = torch.arange(96, device=device) % 6
    for streamed, dense in ((supervised_contrastive_loss, dense_supervised_contrastive_loss),
                            (batch_hard_triplet_loss, dense_batch_hard_triplet_loss)):
        a, b = streamed(z, labels, chunk_size=32), dense(z, labels)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(torch.autograd.grad(a, z), torch.autograd.grad(b, z))
    components = torch.arange(96, device=device) // 24
    nearest = nearest_other_component(z.detach(), components, chunk_size=32)
    D = torch.cdist(z.detach(), z.detach()).square()
    reference = D.masked_fill(components[:, None] == components[None], float('inf')).min(-1)
    assert torch.equal(nearest.indices, reference.indices)
    q, k, v = (torch.randn(64, 4, device=device, dtype=dtype) for _ in range(3))
    causal = torch.ones(64, 64, device=device, dtype=torch.bool).tril()
    reference = torch.softmax((-torch.cdist(q, k).square()).masked_fill(~causal, -float('inf')), -1) @ v
    torch.testing.assert_close(causal_kernel_attention(q, k, v, chunk_size=16), reference)
    print('pair mask examples match dense references')


if __name__ == '__main__':
    main()
