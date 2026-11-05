#pragma once
#include "operator_sources_detail.h"

namespace torchpairwise::ops::detail::residual {
using FeatureOutputs = std::tuple<at::Tensor, at::Tensor>;
using TopKSignature = FeatureOutputs(const at::Tensor&, const at::Tensor&,
                                     int64_t, bool, c10::string_view, int64_t,
                                     int64_t);
using TopKGradSignature = FeatureOutputs(const at::Tensor&, const at::Tensor&,
                                         const at::Tensor&, const at::Tensor&,
                                         c10::string_view, int64_t, int64_t);
using FeatureSignature = FeatureOutputs(const at::Tensor&, const at::Tensor&,
                                        c10::string_view, c10::string_view,
                                        double, int64_t, bool);
using FeatureGradSignature = std::tuple<at::Tensor, at::Tensor>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    c10::string_view, c10::string_view, double, int64_t, bool);
using ReductionSignature = FeatureOutputs(const at::Tensor&, const at::Tensor&,
                                          c10::string_view, c10::string_view,
                                          at::IntArrayRef, bool, double,
                                          int64_t, int64_t);
using ReductionGradSignature = std::tuple<at::Tensor, at::Tensor>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    c10::string_view, c10::string_view, at::IntArrayRef, bool, double, int64_t,
    int64_t);
FeatureOutputs reduction_kernel(const at::Tensor& x, const at::Tensor& y,
                                c10::string_view reduction,
                                c10::string_view transform,
                                at::IntArrayRef axes, bool keepdim,
                                double correction, int64_t sample_size,
                                int64_t feature_size);
std::tuple<at::Tensor, at::Tensor> reduction_backward_kernel(
    const at::Tensor& gradient, const at::Tensor& x, const at::Tensor& y,
    const at::Tensor& state, c10::string_view reduction,
    c10::string_view transform, at::IntArrayRef axes, bool keepdim,
    double correction, int64_t sample_size, int64_t feature_size);
FeatureOutputs feature_kernel(const at::Tensor& x, const at::Tensor& y,
                              c10::string_view reduction,
                              c10::string_view transform, double correction,
                              int64_t size, bool paired);
std::tuple<at::Tensor, at::Tensor> feature_backward_kernel(
    const at::Tensor& gradient, const at::Tensor& x, const at::Tensor& y,
    const at::Tensor& state, c10::string_view reduction,
    c10::string_view transform, double correction, int64_t size, bool paired);
at::Tensor feature_reduce(const at::Tensor& x, const at::Tensor& y,
                          c10::string_view reduction,
                          c10::string_view transform, double correction,
                          int64_t size, bool paired);
}  // namespace torchpairwise::ops::detail::residual

namespace torchpairwise::ops::detail::streaming {
PreparedComparison prepare_residual_operator(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view reduction,
    c10::string_view transform, double correction, int64_t feature_size,
    const PairwiseOperationOptions& options);
}  // namespace torchpairwise::ops::detail::streaming
