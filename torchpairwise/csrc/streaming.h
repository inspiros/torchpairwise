#pragma once

#include <ATen/core/dispatch/Dispatcher.h>

#include <algorithm>

#include "ops/ops.h"

// Public namespace facade. Implementations and serialized schemas stay in ops.
namespace torchpairwise::streaming {
using ops::AudioPairwiseOperator;
using ops::ExpressionSource;
using ops::ImageMapExpression;
using ops::ImageMapOperator;
using ops::ImagePairwiseOperator;
using ops::KernelOperator;
using ops::LinearOperator;
using ops::MatrixOperator;
using ops::MediaPairwiseOperator;
using ops::MMDPairwiseOperator;
using ops::OrderedPairwiseOperator;
using ops::PairwiseOperator;
using ops::ResidualOperator;
using ops::ScalarExpression;
using ops::SequencePairwiseOperator;
using ops::StructuredPairwiseOperator;
using ops::TokenPairwiseOperator;

using ops::AudioComparisonOptions;
using ops::ImageComparisonOptions;
using ops::ImageMapOptions;
using ops::KernelOperationOptions;
using ops::MatrixOperationOptions;
using ops::MMDOperatorOptions;
using ops::PairwiseOperationOptions;
using ops::SequenceOperatorOptions;
using ops::SinkhornMapOptions;
using ops::StreamingReductionOptions;
using ops::StructuredOperationOptions;
using ops::TokenOperatorOptions;
using PairwiseOptions = ops::PairwiseOperationOptions;
using MatrixOptions = ops::MatrixOperationOptions;

// Existing typed family factories and free consumers can be used unchanged
// after replacing the ops namespace with streaming.
using ops::aslinearoperator;
using ops::block;
using ops::custom_expression;
using ops::diagonal;
using ops::group_projection;
using ops::image_event_reduce;
using ops::image_event_topk;
using ops::image_map_operator;
using ops::kernel_operations;
using ops::kernel_residual_outer_product;
using ops::matrix_expression;
using ops::matrix_operations;
using ops::media_expression;
using ops::media_pairwise_operations;
using ops::ordered_expression;
using ops::ordered_pairwise_operations;
using ops::pairwise_expression;
using ops::pairwise_operations;
using ops::residual_operator;
using ops::residual_reduce;
using ops::residual_topk;
using ops::sinkhorn_map;
using ops::streaming_reductions;
using ops::structured_expression;
using ops::structured_pairwise_operations;

namespace detail {
using CatalogEntry =
    std::tuple<std::string, std::string, std::string, std::vector<std::string>,
               std::vector<std::string>, bool, bool, std::vector<std::string>,
               std::string, std::vector<int64_t>>;
inline const CatalogEntry& source_info(c10::string_view name,
                                       bool kernels_only = false) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::comparison_catalog", "")
          .typed<std::vector<CatalogEntry>()>();
  static const auto catalog = op.call();
  // Use the authoritative catalog and its ordering, as in compare/Python.
  // Canonical names win; shared aliases prefer distance unless kernel-only.
  for (const auto& entry : catalog)
    if (std::get<0>(entry) == name &&
        (!kernels_only || std::get<8>(entry) == "kernel"))
      return entry;
  for (const auto& entry : catalog) {
    const auto& aliases = std::get<7>(entry);
    if (std::find(aliases.begin(), aliases.end(), std::string(name)) !=
            aliases.end() &&
        (!kernels_only || std::get<8>(entry) == "kernel"))
      return entry;
  }
  TORCH_CHECK(false, "Unknown ", kernels_only ? "kernel" : "comparison", " '",
              name, "'");
}
}  // namespace detail

// Short factories return the existing common ScalarExpression interface.
// Specialized KernelOperator remains available for vector geometry consumers.
inline ScalarExpression pairwise(
    const at::Tensor& x, const c10::optional<at::Tensor>& y = c10::nullopt,
    c10::string_view metric = "euclidean",
    const PairwiseOptions& options = {}) {
  const auto& info = detail::source_info(metric);
  TORCH_CHECK(
      std::get<1>(info) == "vector" || std::get<1>(info) == "point_cloud",
      "Use the typed family overload for ", std::get<0>(info));
  return ops::pairwise_expression(x, y, std::get<0>(info),
                                  std::get<8>(info) == "kernel", options);
}
inline ScalarExpression pairwise(
    at::TensorList x,
    const c10::optional<std::vector<at::Tensor>>& y = c10::nullopt,
    c10::string_view metric = "gaussian_wasserstein",
    const StructuredOperationOptions& options = {}) {
  return ops::structured_expression(x, y, metric, options);
}
inline ScalarExpression pairwise(const at::Tensor& x,
                                 const c10::optional<at::Tensor>& y,
                                 c10::string_view metric,
                                 const ImageComparisonOptions& comparison,
                                 const PairwiseOptions& options = {}) {
  return ops::media_expression(x, y, metric, comparison, options);
}
inline ScalarExpression pairwise(const at::Tensor& x,
                                 const c10::optional<at::Tensor>& y,
                                 c10::string_view metric,
                                 const AudioComparisonOptions& comparison,
                                 const PairwiseOptions& options = {}) {
  return ops::media_expression(x, y, metric, comparison, options);
}
inline ScalarExpression pairwise(const at::Tensor& x,
                                 const c10::optional<at::Tensor>& y,
                                 c10::string_view metric,
                                 const SequenceOperatorOptions& comparison,
                                 const PairwiseOptions& options = {}) {
  return ops::ordered_expression(x, y, metric, comparison, options);
}
inline ScalarExpression pairwise(const at::Tensor& x,
                                 const c10::optional<at::Tensor>& y,
                                 c10::string_view metric,
                                 const TokenOperatorOptions& comparison,
                                 const PairwiseOptions& options = {}) {
  return ops::ordered_expression(x, y, metric, comparison, options);
}
inline ScalarExpression pairwise(const at::Tensor& x,
                                 const c10::optional<at::Tensor>& y,
                                 c10::string_view metric,
                                 const MMDOperatorOptions& comparison,
                                 const PairwiseOptions& options = {}) {
  return ops::ordered_expression(x, y, metric, comparison, options);
}
inline ScalarExpression kernel(
    const at::Tensor& x, const c10::optional<at::Tensor>& y = c10::nullopt,
    c10::string_view name = "rbf", const PairwiseOptions& options = {}) {
  const auto& info = detail::source_info(name, true);
  TORCH_CHECK(
      std::get<1>(info) == "vector" || std::get<1>(info) == "point_cloud",
      "Use the structured kernel overload for ", std::get<0>(info));
  return ops::pairwise_expression(x, y, std::get<0>(info), true, options);
}
inline ScalarExpression kernel(
    at::TensorList x,
    const c10::optional<std::vector<at::Tensor>>& y = c10::nullopt,
    c10::string_view name = "gaussian_mixture_overlap_kernel",
    const StructuredOperationOptions& options = {}) {
  const auto& info = detail::source_info(name, true);
  return ops::structured_expression(x, y, std::get<0>(info), options);
}
inline ScalarExpression matrix(const at::Tensor& values,
                               const MatrixOptions& options = {}) {
  return ops::matrix_expression(values, options);
}
inline ResidualOperator residual(
    const at::Tensor& x, const c10::optional<at::Tensor>& y = c10::nullopt,
    c10::string_view transform = "identity", int64_t chunk_size = 128,
    int64_t feature_chunk_size = 128) {
  return ops::residual_operator(x, y, transform, chunk_size,
                                feature_chunk_size);
}
inline ImageMapOperator image_map(
    const at::Tensor& x, const c10::optional<at::Tensor>& y = c10::nullopt,
    c10::string_view comparison = "squared_error",
    const ImageMapOptions& options = {}) {
  return ops::image_map_operator(x, y, comparison, options);
}
}  // namespace torchpairwise::streaming

namespace torchpairwise::streaming::linalg {
using ops::trace;
using ops::TraceOptions;
using ops::TraceResult;
using ops::logdet;
using ops::LogdetOptions;
using ops::LogdetResult;
using ops::expm_multiply;
using ops::inv_sqrt_multiply;
using ops::MatrixFunctionOptions;
using ops::MatrixFunctionResult;
using ops::cg;
using ops::CGOptions;
using ops::CGResult;
using ops::CGStatus;
using ops::EigenResult;
using ops::EigenStatus;
using ops::eigh;
using ops::EighOptions;
using ops::gmres;
using ops::GMRESOptions;
using ops::GMRESResult;
using ops::GMRESStatus;
using ops::jacobi;
using ops::lowrank_inverse;
using ops::lsqr;
using ops::LSQROptions;
using ops::LSQRResult;
using ops::LSQRStatus;
using ops::lstsq;
using ops::LstsqOptions;
using ops::minres;
using ops::MINRESOptions;
using ops::MINRESResult;
using ops::MINRESStatus;
using ops::nystrom_inverse;
using ops::solve;
using ops::SolveOptions;
using ops::SolveResult;
using ops::svd;
using ops::SVDOptions;
using ops::SVDResult;
using ops::SVDStatus;
}  // namespace torchpairwise::streaming::linalg

namespace torchpairwise::streaming::optimize {
using ops::box_qp;
using ops::BoxQPOptions;
using ops::QPEquality;
using ops::QPResult;
using ops::QPStatus;
}  // namespace torchpairwise::streaming::optimize
