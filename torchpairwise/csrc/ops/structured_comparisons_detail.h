#pragma once
// Internal native bridges shared by forward/backward kernels and autograd.
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <array>
#include <vector>
namespace torchpairwise::ops::detail::structured {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
using GradResult = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
using Signature = Tensor(const Tensor&, const Tensor&, const Tensor&,
                         const Tensor&, const Tensor&, const Tensor&,
                         const Tensor&, const Tensor&, int64_t, int64_t, bool,
                         double, int64_t, int64_t, int64_t);
inline Tensor _structured_compare(const Tensor& x, const Tensor& xc,
                                  const Tensor& xw, const Tensor& y,
                                  const Tensor& yc, const Tensor& yw,
                                  const Tensor& xl, const Tensor& yl,
                                  int64_t kind, int64_t mode, bool squared,
                                  double epsilon, int64_t iterations,
                                  int64_t tile, int64_t component_tile) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_structured_compare", "")
          .typed<Signature>();
  return op.call(x, xc, xw, y, yc, yw, xl, yl, kind, mode, squared, epsilon,
                 iterations, tile, component_tile);
}
using Features = std::array<Tensor, 4>;
struct PreparedInputs {
  std::array<Tensor, 8> operands;
  std::vector<c10::SymInt> batch;
  int64_t kind, mode;
  bool squared;
  double epsilon;
  int64_t iterations, tile, component_tile;
};
PreparedInputs prepare_gaussians(const Tensor& mean, const Tensor& covariance,
                                 const c10::optional<Tensor>& other_mean,
                                 const c10::optional<Tensor>& other_covariance,
                                 c10::string_view mode, bool squared);
PreparedInputs prepare_covariances(const Tensor& covariance,
                                   const c10::optional<Tensor>& other,
                                   c10::string_view metric,
                                   c10::string_view mode, bool squared);
PreparedInputs prepare_mixtures(
    const Tensor& mean, const Tensor& covariance, const Tensor& weights,
    const c10::optional<Tensor>& other_mean,
    const c10::optional<Tensor>& other_covariance,
    const c10::optional<Tensor>& other_weights, c10::string_view metric,
    c10::string_view mode, const c10::optional<Tensor>& lengths,
    const c10::optional<Tensor>& other_lengths, c10::optional<bool> squared,
    c10::optional<double> regularization, c10::optional<int64_t> max_iter,
    int64_t chunk_size, c10::optional<int64_t> component_chunk_size);
Features prepare_features(const Tensor& x, const Tensor& covariance,
                          const Tensor& weights, const Tensor& lengths,
                          int64_t kind, double epsilon, int64_t iterations,
                          int64_t component_tile);
Tensor feature_pair(const Features& a, const Features& b, int64_t kind,
                    bool paired, bool squared, double epsilon,
                    int64_t iterations, int64_t component_tile);
using FeatureSignature = std::vector<Tensor>(const Tensor&, const Tensor&,
                                             const Tensor&, const Tensor&,
                                             int64_t, double, int64_t, int64_t);
inline std::vector<Tensor> features(const Tensor& x, const Tensor& covariance,
                                    const Tensor& weights,
                                    const Tensor& lengths, int64_t kind,
                                    double epsilon, int64_t iterations,
                                    int64_t component_tile) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_structured_features", "")
          .typed<FeatureSignature>();
  return op.call(x, covariance, weights, lengths, kind, epsilon, iterations,
                 component_tile);
}
using FeatureGradSignature = std::vector<Tensor>(at::TensorList, const Tensor&,
                                                 const Tensor&, const Tensor&,
                                                 const Tensor&, int64_t, double,
                                                 int64_t, int64_t);
Tensor matrix_root_autograd(Tensor x);
Tensor bures_autograd(Tensor a, Tensor b);
Tensor matrix_log_autograd(Tensor x);
Tensor congruence_autograd(Tensor a, Tensor b, Tensor factor, int64_t kind);

std::array<Tensor, 3> matrix_root_kernel(Tensor x);
Values matrix_root_backward_kernel(const Tensor& gradient, const Values& saved);
std::array<Tensor, 3> bures_kernel(Tensor a, Tensor b);
Values bures_backward_kernel(const Tensor& gradient, const Values& saved,
                             at::IntArrayRef a_shape, at::IntArrayRef b_shape);
std::array<Tensor, 3> matrix_log_kernel(Tensor x);
Values matrix_log_backward_kernel(const Tensor& gradient, const Values& saved);
std::array<Tensor, 4> congruence_kernel(Tensor a, Tensor b, Tensor factor,
                                        int64_t kind);
Values congruence_backward_kernel(const Tensor& gradient, const Values& saved,
                                  at::IntArrayRef a_shape,
                                  at::IntArrayRef b_shape, int64_t kind);
}  // namespace torchpairwise::ops::detail::structured
