import torch

# sklearn
euclidean_distances = torch.ops.torchpairwise.euclidean_distances
haversine_distances = torch.ops.torchpairwise.haversine_distances
manhattan_distances = torch.ops.torchpairwise.manhattan_distances
cosine_distances = torch.ops.torchpairwise.cosine_distances
linear_kernel = torch.ops.torchpairwise.linear_kernel
polynomial_kernel = torch.ops.torchpairwise.polynomial_kernel
sigmoid_kernel = torch.ops.torchpairwise.sigmoid_kernel
rbf_kernel = torch.ops.torchpairwise.rbf_kernel
laplacian_kernel = torch.ops.torchpairwise.laplacian_kernel
cosine_similarity = torch.ops.torchpairwise.cosine_similarity
additive_chi2_kernel = torch.ops.torchpairwise.additive_chi2_kernel
chi2_kernel = torch.ops.torchpairwise.chi2_kernel

# scipy
directed_hausdorff_distances = torch.ops.torchpairwise.directed_hausdorff_distances
minkowski_distances = torch.ops.torchpairwise.minkowski_distances
wminkowski_distances = torch.ops.torchpairwise.wminkowski_distances
sqeuclidean_distances = torch.ops.torchpairwise.sqeuclidean_distances
correlation_distances = torch.ops.torchpairwise.correlation_distances
hamming_distances = torch.ops.torchpairwise.hamming_distances
jaccard_distances = torch.ops.torchpairwise.jaccard_distances
kulsinski_distances = torch.ops.torchpairwise.kulsinski_distances
kulczynski1_distances = torch.ops.torchpairwise.kulczynski1_distances
seuclidean_distances = torch.ops.torchpairwise.seuclidean_distances
cityblock_distances = torch.ops.torchpairwise.cityblock_distances
mahalanobis_distances = torch.ops.torchpairwise.mahalanobis_distances
chebyshev_distances = torch.ops.torchpairwise.chebyshev_distances
braycurtis_distances = torch.ops.torchpairwise.braycurtis_distances
canberra_distances = torch.ops.torchpairwise.canberra_distances
jensenshannon_distances = torch.ops.torchpairwise.jensenshannon_distances
yule_distances = torch.ops.torchpairwise.yule_distances
dice_distances = torch.ops.torchpairwise.dice_distances
rogerstanimoto_distances = torch.ops.torchpairwise.rogerstanimoto_distances
russellrao_distances = torch.ops.torchpairwise.russellrao_distances
sokalmichener_distances = torch.ops.torchpairwise.sokalmichener_distances
sokalsneath_distances = torch.ops.torchpairwise.sokalsneath_distances

# aliases
l1_distances = torch.ops.torchpairwise.l1_distances
l2_distances = torch.ops.torchpairwise.l2_distances
lp_distances = torch.ops.torchpairwise.lp_distances
