#include "utils/comparison_dispatch.h"

#include <cmath>

#include "pairwise_metrics.h"
namespace torchpairwise::ops::detail {
namespace {
struct ComparisonEntry {
  const char* name;
  bool kernel;
  ComparisonFamily family;
};
constexpr ComparisonEntry comparisons[] = {
    {"euclidean", false, ComparisonFamily::Legacy},
    {"haversine", false, ComparisonFamily::Legacy},
    {"manhattan", false, ComparisonFamily::Legacy},
    {"cosine", false, ComparisonFamily::Legacy},
    {"sqeuclidean", false, ComparisonFamily::Legacy},
    {"correlation", false, ComparisonFamily::Legacy},
    {"chebyshev", false, ComparisonFamily::Legacy},
    {"braycurtis", false, ComparisonFamily::Legacy},
    {"canberra", false, ComparisonFamily::Legacy},
    {"minkowski", false, ComparisonFamily::Legacy},
    {"wminkowski", false, ComparisonFamily::Legacy},
    {"seuclidean", false, ComparisonFamily::Legacy},
    {"mahalanobis", false, ComparisonFamily::Legacy},
    {"jensenshannon", false, ComparisonFamily::Legacy},
    {"snr", false, ComparisonFamily::Legacy},
    {"hamming", false, ComparisonFamily::Legacy},
    {"jaccard", false, ComparisonFamily::Legacy},
    {"kulsinski", false, ComparisonFamily::Legacy},
    {"kulczynski1", false, ComparisonFamily::Legacy},
    {"yule", false, ComparisonFamily::Legacy},
    {"dice", false, ComparisonFamily::Legacy},
    {"rogerstanimoto", false, ComparisonFamily::Legacy},
    {"russellrao", false, ComparisonFamily::Legacy},
    {"sokalmichener", false, ComparisonFamily::Legacy},
    {"sokalsneath", false, ComparisonFamily::Legacy},
    {"bhattacharyya", false, ComparisonFamily::Vector},
    {"aitchison", false, ComparisonFamily::Vector},
    {"hellinger", false, ComparisonFamily::Vector},
    {"nan_euclidean", false, ComparisonFamily::Vector},
    {"wasserstein_1d", false, ComparisonFamily::Vector},
    {"total_variation", false, ComparisonFamily::Vector},
    {"fisher_rao", false, ComparisonFamily::Vector},
    {"renyi", false, ComparisonFamily::Vector},
    {"pearson_chi2", false, ComparisonFamily::Vector},
    {"neyman_chi2", false, ComparisonFamily::Vector},
    {"triangular_discrimination", false, ComparisonFamily::Vector},
    {"itakura_saito", false, ComparisonFamily::Vector},
    {"angular", false, ComparisonFamily::Vector},
    {"lorentzian", false, ComparisonFamily::Vector},
    {"clark", false, ComparisonFamily::Vector},
    {"soergel", false, ComparisonFamily::Vector},
    {"ruzicka", false, ComparisonFamily::Vector},
    {"kl", false, ComparisonFamily::Vector},
    {"symmetric_kl", false, ComparisonFamily::Vector},
    {"chamfer", false, ComparisonFamily::Pointset},
    {"hausdorff", false, ComparisonFamily::Pointset},
    {"directed_hausdorff", false, ComparisonFamily::Pointset},
    {"energy", false, ComparisonFamily::Transport},
    {"mmd", false, ComparisonFamily::Transport},
    {"sliced_wasserstein", false, ComparisonFamily::Transport},
    {"sinkhorn", false, ComparisonFamily::Transport},
    {"wasserstein", false, ComparisonFamily::Transport},
    {"linear", true, ComparisonFamily::Legacy},
    {"polynomial", true, ComparisonFamily::Legacy},
    {"sigmoid", true, ComparisonFamily::Legacy},
    {"rbf", true, ComparisonFamily::Legacy},
    {"laplacian", true, ComparisonFamily::Legacy},
    {"cosine", true, ComparisonFamily::Legacy},
    {"additive_chi2", true, ComparisonFamily::Legacy},
    {"chi2", true, ComparisonFamily::Legacy},
    {"bhattacharyya", true, ComparisonFamily::Vector},
    {"histogram_intersection", true, ComparisonFamily::Vector},
    {"rational_quadratic", true, ComparisonFamily::Vector},
    {"matern", true, ComparisonFamily::Vector}};
using BinaryFunction = at::Tensor (*)(const at::Tensor&,
                                      const c10::optional<at::Tensor>&);
struct NamedEntry {
  const char* name;
  bool kernel;
  BinaryFunction function;
};
const NamedEntry named_comparisons[] = {
    {"euclidean", false, euclidean_distances},
    {"haversine", false, haversine_distances},
    {"manhattan", false, manhattan_distances},
    {"cosine", false, cosine_distances},
    {"sqeuclidean", false, sqeuclidean_distances},
    {"correlation", false, correlation_distances},
    {"chebyshev", false, chebyshev_distances},
    {"braycurtis", false, braycurtis_distances},
    {"canberra", false, canberra_distances},
    {"snr", false, snr_distances},
    {"hamming", false, hamming_distances},
    {"jaccard", false, jaccard_distances},
    {"kulsinski", false, kulsinski_distances},
    {"kulczynski1", false, kulczynski1_distances},
    {"yule", false, yule_distances},
    {"dice", false, dice_distances},
    {"rogerstanimoto", false, rogerstanimoto_distances},
    {"russellrao", false, russellrao_distances},
    {"sokalmichener", false, sokalmichener_distances},
    {"sokalsneath", false, sokalsneath_distances},
    {"linear", true, linear_kernel},
    {"cosine", true, cosine_similarity},
    {"additive_chi2", true, additive_chi2_kernel}};
}  // namespace
std::string canonical_comparison_name(c10::string_view raw) {
  std::string name(raw);
  for (auto suffix : {"_distances", "_divergences", "_kernel"}) {
    auto len = std::char_traits<char>::length(suffix);
    if (name.size() > len &&
        name.compare(name.size() - len, len, suffix) == 0) {
      name.resize(name.size() - len);
      break;
    }
  }
  if (name == "l1" || name == "cityblock")
    name = "manhattan";
  if (name == "l2")
    name = "euclidean";
  if (name == "lp")
    name = "minkowski";
  if (name == "linf")
    name = "chebyshev";
  if (name == "continuous_jaccard")
    name = "ruzicka";
  if (name == "cosine_similarity")
    name = "cosine";
  return name;
}
int64_t ordered_comparison_kind(c10::string_view raw) {
  auto name = canonical_comparison_name(raw);
  if (name == "mmd_statistics" || name == "mmd_statistic")
    return 5;
  constexpr const char* names[] = {"dtw", "soft_dtw", "levenshtein", "indel",
                                   "mmd"};
  for (int64_t i = 0; i < 5; ++i)
    if (name == names[i])
      return i;
  return -1;
}
int64_t media_comparison_kind(c10::string_view raw) {
  constexpr const char* names[][2] = {
      {"psnr", "psnr_scores"},
      {"ssim", "ssim_scores"},
      {"multiscale_ssim", "multiscale_ssim_scores"},
      {"si_sdr", "si_sdr_scores"},
      {"signal_snr", "signal_snr_scores"},
      {"log_spectral", "log_spectral_distances"},
      {"spectral_convergence", "spectral_convergence_distances"},
      {"multiresolution_stft", "multiresolution_stft_distances"}};
  if (raw == "ssim_index")
    return 1;
  for (int64_t i = 0; i < 8; ++i)
    if (raw == names[i][0] || raw == names[i][1])
      return i;
  return -1;
}
int64_t structured_comparison_kind(c10::string_view raw) {
  auto name = canonical_comparison_name(raw);
  if (name == "gaussian_mixture_entropic_transport_costs")
    name = "gaussian_mixture_entropic_transport";
  constexpr const char* names[] = {"gaussian_wasserstein",
                                   "log_euclidean",
                                   "affine_invariant",
                                   "stein",
                                   "gaussian_mixture_overlap",
                                   "gaussian_mixture_l2",
                                   "gaussian_mixture_cauchy_schwarz",
                                   "gaussian_mixture_wasserstein",
                                   "gaussian_mixture_entropic_transport",
                                   "gaussian_mixture_sinkhorn"};
  for (int64_t i = 0; i < 10; ++i)
    if (name == names[i])
      return i;
  return -1;
}
ComparisonFamily comparison_family(c10::string_view name, bool kernel) {
  for (const auto& entry : comparisons)
    if (name == entry.name && kernel == entry.kernel)
      return entry.family;
  TORCH_CHECK(false, "Unknown builtin ",
              kernel ? "kernel: " : "distance: ", name);
}
at::Tensor call_named_comparison(const at::Tensor& x,
                                 const c10::optional<at::Tensor>& y,
                                 c10::string_view name, bool kernel, double p,
                                 double base, int64_t degree, double gamma,
                                 double coef0, const at::Tensor& variance,
                                 const at::Tensor& inverse) {
  if (kernel) {
    if (name == "polynomial")
      return polynomial_kernel(x, y, degree, gamma, coef0);
    if (name == "sigmoid")
      return sigmoid_kernel(x, y, gamma, coef0);
    if (name == "rbf")
      return rbf_kernel(x, y, gamma);
    if (name == "laplacian")
      return laplacian_kernel(x, y, gamma);
    if (name == "chi2")
      return chi2_kernel(x, y, gamma);
  } else {
    if (name == "minkowski" || name == "wminkowski")
      return minkowski_distances(x, y, p);
    if (name == "seuclidean")
      return seuclidean_distances(x, y, variance);
    if (name == "mahalanobis")
      return mahalanobis_distances(x, y, inverse);
    if (name == "jensenshannon")
      return jensenshannon_distances(
          x, y, std::isnan(base) ? c10::nullopt : c10::optional<double>(base));
  }
  for (const auto& entry : named_comparisons)
    if (name == entry.name && kernel == entry.kernel)
      return entry.function(x, y);
  TORCH_CHECK(false, "Unknown named comparison: ", name);
}
}  // namespace torchpairwise::ops::detail
