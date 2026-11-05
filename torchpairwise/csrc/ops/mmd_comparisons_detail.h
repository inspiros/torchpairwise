#pragma once
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include "utils/clouds.h"

namespace torchpairwise::ops::detail::mmd {
struct KernelOptions {
  std::string name;
  double gamma;
  int64_t degree;
  double coef0, scale, alpha, nu;
  bool statistic = false, unbiased = false, exclude_diagonal = false;
  int64_t chunk_size = 128;
};
using Core = at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, c10::string_view, int64_t, double,
                        int64_t, double, double, double, double, bool);
using Grad = std::tuple<at::Tensor, at::Tensor>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, c10::string_view, int64_t, double, int64_t, double,
    double, double, double, bool);
inline at::Tensor _mmd_compare(const at::Tensor& x, const at::Tensor& y,
                               const at::Tensor& xl, const at::Tensor& yl,
                               const KernelOptions& o, int64_t mode,
                               bool squared) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_mmd_compare", "")
                       .typed<Core>();
  return op.call(x, y, xl, yl, o.name, mode, o.gamma, o.degree, o.coef0,
                 o.scale, o.alpha, o.nu, squared);
}
inline at::Tensor _mmd_statistic(const at::Tensor& x, const at::Tensor& y,
                                 const at::Tensor& xl, const at::Tensor& yl,
                                 const KernelOptions& o, int64_t mode) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_mmd_compare", "statistic")
          .typed<Core>();
  return op.call(x, y, xl, yl, o.name, mode, o.gamma, o.degree, o.coef0,
                 o.scale, o.alpha, o.nu, o.unbiased);
}
at::Tensor point_kernel(const at::Tensor& x, const at::Tensor& y,
                        const at::Tensor& xv, const at::Tensor& yv,
                        const KernelOptions& options);
at::Tensor point_autograd(at::Tensor x, at::Tensor y, at::Tensor xv,
                          at::Tensor yv, const KernelOptions& options);
}  // namespace torchpairwise::ops::detail::mmd

namespace torchpairwise::ops::detail::mmd {
using at::Tensor;
struct PreparedInputs {
  CloudInputs clouds;
  KernelOptions options;
  int64_t mode;
  bool squared;
};
PreparedInputs prepare(const Tensor& input, const c10::optional<Tensor>& other,
                       c10::string_view mode, c10::string_view raw,
                       bool squared, double scale, c10::optional<double> gamma,
                       c10::optional<int64_t> degree,
                       c10::optional<double> coef0, c10::optional<double> alpha,
                       c10::optional<double> nu,
                       const c10::optional<Tensor>& xl,
                       const c10::optional<Tensor>& yl,
                       c10::optional<int64_t> size, c10::optional<bool> batched,
                       int64_t feature_dim,
                       c10::optional<c10::string_view> estimator);
using FeatureSignature = std::vector<Tensor>(const Tensor&, const Tensor&,
                                             c10::string_view, double, int64_t,
                                             double, double, double, double,
                                             bool, bool, int64_t);
using FeatureGradSignature = Tensor(at::TensorList, const Tensor&,
                                    const Tensor&, c10::string_view, double,
                                    int64_t, double, double, double, double,
                                    bool, bool, int64_t);
inline std::vector<Tensor> features(const Tensor& x, const Tensor& lengths,
                                    const KernelOptions& o) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_mmd_features", "")
                       .typed<FeatureSignature>();
  return op.call(x, lengths, o.name, o.gamma, o.degree, o.coef0, o.scale,
                 o.alpha, o.nu, o.statistic, o.unbiased, o.chunk_size);
}
Tensor prepared_pair(const Tensor& x, const Tensor& y, const Tensor& xl,
                     const Tensor& yl, const Tensor& ax, const Tensor& by,
                     const KernelOptions& options, bool paired, bool squared,
                     int64_t chunk_size = 128);
}  // namespace torchpairwise::ops::detail::mmd
