import scipy.spatial.distance as sci_dist
import sklearn.metrics.pairwise as sklearn_pw
import torch

import torchpairwise


def test_boolean_kernels(device='cuda'):
    x1 = torch.randint(0, 3, (10, 5), device=device)
    x2 = torch.randint(0, 3, (8, 5), device=device)

    print('dice')
    output = torchpairwise.dice_distances(x1, x2)
    py_output = sklearn_pw.pairwise_distances(x1.detach().cpu(),
                                              x2.detach().cpu(),
                                              metric='dice')
    print(output.detach().cpu() - py_output)

    print('hamming')
    output = torchpairwise.hamming_distances(x1, x2)
    py_output = sklearn_pw.pairwise_distances(x1.detach().cpu(),
                                              x2.detach().cpu(),
                                              metric='hamming')
    print(output.detach().cpu() - py_output)

    print('jaccard')
    output = torchpairwise.jaccard_distances(x1, x2)
    py_output = sklearn_pw.pairwise_distances(x1.detach().cpu(),
                                              x2.detach().cpu(),
                                              metric='jaccard')
    print(output.detach().cpu() - py_output)

    print('kulsinski')
    output = torchpairwise.kulsinski_distances(x1, x2)
    py_output = sklearn_pw.pairwise_distances(x1.detach().cpu(),
                                              x2.detach().cpu(),
                                              metric='kulsinski')
    print(output.detach().cpu() - py_output)

    print('rogerstanimoto')
    output = torchpairwise.rogerstanimoto_distances(x1, x2)
    py_output = sklearn_pw.pairwise_distances(x1.detach().cpu(),
                                              x2.detach().cpu(),
                                              metric='rogerstanimoto')
    print(output.detach().cpu() - py_output)

    print('russellrao')
    output = torchpairwise.russellrao_distances(x1, x2)
    py_output = sklearn_pw.pairwise_distances(x1.detach().cpu(),
                                              x2.detach().cpu(),
                                              metric='russellrao')
    print(output.detach().cpu() - py_output)

    print('sokalmichener')
    output = torchpairwise.sokalmichener_distances(x1, x2)
    py_output = sklearn_pw.pairwise_distances(x1.detach().cpu(),
                                              x2.detach().cpu(),
                                              metric='sokalmichener')
    print(output.detach().cpu() - py_output)

    print('sokalsneath')
    output = torchpairwise.sokalsneath_distances(x1, x2)
    py_output = sklearn_pw.pairwise_distances(x1.detach().cpu(),
                                              x2.detach().cpu(),
                                              metric='sokalsneath')
    print(output.detach().cpu() - py_output)

    print('yule')
    output = torchpairwise.yule_distances(x1, x2)
    py_output = sklearn_pw.pairwise_distances(x1.detach().cpu(),
                                              x2.detach().cpu(),
                                              metric='yule')
    print(output.detach().cpu() - py_output)


def test_floating_kernels(dtype=torch.float64, device='cuda'):
    x1 = torch.rand(10, 5, dtype=dtype, device=device)
    x2 = torch.rand(8, 5, dtype=dtype, device=device)

    print('additive_chi2_kernel')
    output = torchpairwise.additive_chi2_kernel(x1, x2)
    py_output = sklearn_pw.additive_chi2_kernel(x1.detach().cpu(),
                                                x2.detach().cpu())
    print(output.detach().cpu() - py_output)

    print('mahalanobis')
    VI = torch.full((x1.size(-1), x2.size(-1)), 0.1, dtype=dtype, device=device)
    output = torchpairwise.mahalanobis_distances(x1, x2, VI)
    py_output = sklearn_pw.pairwise_distances(x1.detach().cpu(),
                                              x2.detach().cpu(),
                                              metric='mahalanobis',
                                              VI=VI.detach().cpu())
    print(output.detach().cpu() - py_output)

    print('seuclidean')
    V = torch.full((x1.size(-1),), 0.1, dtype=dtype, device=device)
    output = torchpairwise.seuclidean_distances(x1, x2, V)
    py_output = sklearn_pw.pairwise_distances(x1.detach().cpu(),
                                              x2.detach().cpu(),
                                              metric='seuclidean',
                                              V=V.detach().cpu())
    print(output.detach().cpu() - py_output)

    print('braycurtis')
    output = torchpairwise.braycurtis_distances(x1, x2)
    py_output = sklearn_pw.pairwise_distances(x1.detach().cpu(),
                                              x2.detach().cpu(),
                                              metric='braycurtis')
    print(output.detach().cpu() - py_output)

    print('canberra')
    output = torchpairwise.canberra_distances(x1, x2)
    py_output = sklearn_pw.pairwise_distances(x1.detach().cpu(),
                                              x2.detach().cpu(),
                                              metric='canberra')
    print(output.detach().cpu() - py_output)

    print('cosine')
    output = torchpairwise.cosine_distances(x1, x2)
    py_output = sklearn_pw.pairwise_distances(x1.detach().cpu(),
                                              x2.detach().cpu(),
                                              metric='cosine')
    print(output.detach().cpu() - py_output)

    print('correlation')
    output = torchpairwise.correlation_distances(x1, x2)
    py_output = sklearn_pw.pairwise_distances(x1.detach().cpu(),
                                              x2.detach().cpu(),
                                              metric='correlation')
    print(output.detach().cpu() - py_output)

    print('jensenshannon')
    output = torchpairwise.jensenshannon_distances(x1, x2)
    py_output = sci_dist.cdist(x1.detach().cpu(),
                               x2.detach().cpu(),
                               metric='jensenshannon')
    print(output.detach().cpu() - py_output)

    print('directed_hausdorff')
    x1 = torch.rand(10, 9, 3)
    x2 = torch.rand(8, 7, 3)
    output, x1_inds, x2_inds = torchpairwise.directed_hausdorff_distances(x1, x2, shuffle=True)
    print(output)
    print(x1_inds)
    print(x2_inds)
    py_output = sci_dist.directed_hausdorff(x1[-1].detach().cpu(),
                                            x2[-1].detach().cpu())
    print(py_output)
    # print(output.detach().cpu() - py_output)

    gen = torch.Generator()
    gen.manual_seed(1)
    x1 = x1.to(dtype=torch.float64, device=device)
    x2 = x2.to(dtype=torch.float64, device=device)
    x1.requires_grad_()
    x2.requires_grad_()
    grad_correct = torch.autograd.gradcheck(
        lambda x, y: torchpairwise.directed_hausdorff_distances(x, y, shuffle=True, generator=gen), inputs=(x1, x2))
    print('grad_correct:', grad_correct)


def test_cdist(dtype=torch.float64, device='cuda'):
    x1 = torch.rand(10, 5, dtype=dtype, device=device)
    x2 = torch.rand(8, 5, dtype=dtype, device=device)

    output = torchpairwise.cdist(x1, x2, metric='manhattan')
    print(output)
    print(output - torchpairwise.manhattan_distances(x1, x2))


if __name__ == '__main__':
    # test_boolean_kernels()
    # test_floating_kernels()
    test_cdist()
