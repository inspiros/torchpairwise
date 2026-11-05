#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/library.h>

#include <unordered_map>

namespace torchpairwise::ops {
namespace {
// Accuracy-first mixed precision for floating comparisons: half/bfloat16
// operands on the autocast device run in float32; float64, integer, Boolean
// and other-device tensors are unchanged. Repeated operands are cast once so
// identity checks (for example cosine self distances) still hold.
template <c10::DeviceType device>
void comparison_autocast(const c10::OperatorHandle& op,
                         torch::jit::Stack* stack) {
  std::unordered_map<c10::TensorImpl*, at::Tensor> casts;
  auto cast = [&](const at::Tensor& value) {
    if (!value.defined() || value.device().type() != device ||
        (value.scalar_type() != at::kHalf &&
         value.scalar_type() != at::kBFloat16))
      return value;
    auto [item, inserted] = casts.try_emplace(value.unsafeGetTensorImpl());
    if (inserted)
      item->second = value.to(at::kFloat);
    return item->second;
  };
  const auto arguments = op.schema().arguments().size();
  for (size_t index = stack->size() - arguments; index < stack->size();
       ++index) {
    auto& value = (*stack)[index];
    if (value.isTensor()) {
      value = cast(value.toTensor());
    } else if (value.isTensorList()) {
      c10::List<at::Tensor> list;
      for (const at::Tensor& item : value.toTensorList())
        list.push_back(cast(item));
      value = list;
    }
  }
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKeySet(
      {c10::DispatchKey::AutocastCPU, c10::DispatchKey::AutocastCUDA}));
  op.callBoxed(stack);
}

// Public floating comparison schemas and the primitives they compose.
constexpr const char* operators[] = {
    "_additive_chi2_kernel",
    "_braycurtis",
    "_canberra",
    "_directed_hausdorff",
    "_haversine",
    "_histogram_intersection",
    "_nan_sqeuclidean",
    "_pointset_nn",
    "_ppminkowski",
    "_snr",
    "_sqjensenshannon",
    "_sqmahalanobis",
    "_wminkowski",
    "additive_chi2_kernel",
    "aitchison_distances",
    "angular_distances",
    "bhattacharyya_distances",
    "bhattacharyya_kernel",
    "braycurtis_distances",
    "canberra_distances",
    "cdist",
    "chebyshev_distances",
    "chi2_kernel",
    "cityblock_distances",
    "clark_distances",
    "correlation_distances",
    "cosine_distances",
    "cosine_similarity",
    "directed_hausdorff_distances",
    "euclidean_distances",
    "fisher_rao_distances",
    "haversine_distances",
    "hellinger_distances",
    "histogram_intersection_kernel",
    "itakura_saito_divergences",
    "jensenshannon_distances",
    "kl_divergences",
    "l1_distances",
    "l2_distances",
    "laplacian_kernel",
    "linear_kernel",
    "linf_distances",
    "lorentzian_distances",
    "lp_distances",
    "mahalanobis_distances",
    "manhattan_distances",
    "matern_kernel",
    "minkowski_distances",
    "nan_euclidean_distances",
    "neyman_chi2_divergences",
    "pdist",
    "pdist_kernel",
    "pearson_chi2_divergences",
    "polynomial_kernel",
    "rational_quadratic_kernel",
    "rbf_kernel",
    "renyi_divergences",
    "ruzicka_distances",
    "seuclidean_distances",
    "sigmoid_kernel",
    "snr_distances",
    "soergel_distances",
    "sqeuclidean_distances",
    "symmetric_kl_divergences",
    "total_variation_distances",
    "triangular_discrimination_distances",
    "wasserstein_1d_distances",
    "wminkowski_distances",
};

TORCH_LIBRARY_IMPL(torchpairwise, AutocastCPU, m) {
  for (const auto* name : operators)
    m.impl(name, torch::CppFunction::makeFromBoxedFunction<
                     &comparison_autocast<c10::DeviceType::CPU>>());
}
TORCH_LIBRARY_IMPL(torchpairwise, AutocastCUDA, m) {
  for (const auto* name : operators)
    m.impl(name, torch::CppFunction::makeFromBoxedFunction<
                     &comparison_autocast<c10::DeviceType::CUDA>>());
}
}  // namespace
}  // namespace torchpairwise::ops
