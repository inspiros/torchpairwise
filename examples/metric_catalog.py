"""Probability, continuous histogram and SPD covariance comparison modes."""
import torch
import torchpairwise as tp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.tensor([[1., 2., 4.], [4., 1., 2.]], device=device, requires_grad=True)
    y = torch.tensor([[3., 2., 1.], [2., 4., 1.]], device=device, requires_grad=True)
    loss = x.new_zeros(())
    for name in ('total_variation', 'fisher_rao', 'renyi', 'pearson_chi2', 'neyman_chi2',
                 'triangular_discrimination', 'itakura_saito', 'angular', 'lorentzian',
                 'clark', 'soergel', 'ruzicka'):
        options = {'alpha': .7} if name == 'renyi' else {}
        matrix = tp.cdist(x, y, metric=name, **options)
        paired = tp.paired_distances(x, y, metric=name, **options)
        print(name, 'all-pairs:', matrix.detach(), 'paired:', paired.detach())
        loss = loss + matrix.sum() + paired.sum()
    print('continuous Jaccard condensed:', tp.pdist(x, metric='continuous_jaccard').detach())
    a = torch.tensor([[[2., .3], [.3, 1.]], [[1., 0.], [0., 3.]]],
                     dtype=torch.float64, device=device, requires_grad=True)
    b = (a.detach().flip(0) * 1.5).requires_grad_()
    covariance_loss = a.new_zeros(())
    for name in ('log_euclidean', 'affine_invariant', 'stein'):
        matrix = getattr(tp, name + '_distances')(a, b)
        paired = getattr(tp, 'paired_' + name + '_distances')(a, b)
        condensed = getattr(tp, 'condensed_' + name + '_distances')(a)
        print(name, 'all-pairs:', matrix.detach(), 'paired:', paired.detach(),
              'condensed:', condensed.detach())
        covariance_loss = covariance_loss + matrix.sum() + paired.sum() + condensed.sum()
    for gradient in (*torch.autograd.grad(loss, (x, y)),
                     *torch.autograd.grad(covariance_loss, (a, b))):
        assert torch.isfinite(gradient).all()
    print('Expanded metric catalog gradients passed on', device)


if __name__ == '__main__':
    main()
