#pragma once
#include "expression_operations_detail.h"

namespace torchpairwise::ops::detail::streaming {
using StatisticsOutputs = std::tuple<Tensor, Tensor>;
using SelectionSignature = StatisticsOutputs(at::TensorList,
                                             TORCHPAIRWISE_EXPRESSION_TYPES,
                                             double, c10::string_view);
using SelectionGradSignature = Tensors(const Tensor&, at::TensorList,
                                       const Tensor&, const Tensor&,
                                       TORCHPAIRWISE_EXPRESSION_TYPES, double,
                                       c10::string_view);
StatisticsOutputs expression_selection_kernel(at::TensorList operands,
                                              TORCHPAIRWISE_EXPRESSION_DECL,
                                              double q,
                                              c10::string_view interpolation);
Tensors expression_selection_backward_kernel(
    const Tensor& gradient, at::TensorList operands, const Tensor& result,
    const Tensor& indices, TORCHPAIRWISE_EXPRESSION_DECL, double q,
    c10::string_view interpolation);
Tensor expression_selection(at::TensorList operands, const Settings& settings,
                            c10::string_view reduction, double q,
                            c10::string_view interpolation);
StatisticsOutputs expression_product_kernel(at::TensorList operands,
                                            TORCHPAIRWISE_EXPRESSION_DECL,
                                            double correction);
Tensors expression_product_backward_kernel(
    const Tensor& gradient, at::TensorList operands, const Tensor& result,
    const Tensor& state, TORCHPAIRWISE_EXPRESSION_DECL, double correction);
using InnerProductSignature = StatisticsOutputs(at::TensorList,
                                                TORCHPAIRWISE_EXPRESSION_TYPES,
                                                int64_t, int64_t, bool, bool);
using InnerProductGradSignature = Tensors(const Tensor&, at::TensorList,
                                          const Tensor&, const Tensor&,
                                          TORCHPAIRWISE_EXPRESSION_TYPES,
                                          int64_t, int64_t, bool, bool);
StatisticsOutputs inner_product_kernel(at::TensorList operands,
                                       TORCHPAIRWISE_EXPRESSION_DECL,
                                       int64_t left_root, int64_t right_root,
                                       bool center, bool normalize);
Tensors inner_product_backward_kernel(const Tensor& gradient,
                                      at::TensorList operands,
                                      const Tensor& result, const Tensor& state,
                                      TORCHPAIRWISE_EXPRESSION_DECL,
                                      int64_t left_root, int64_t right_root,
                                      bool center, bool normalize);
Tensor expression_inner_product(at::TensorList operands,
                                const Settings& settings, int64_t left_root,
                                int64_t right_root, bool center, bool normalize,
                                c10::string_view statistic = "inner_product");
using StatisticsSignature = StatisticsOutputs(at::TensorList,
                                              TORCHPAIRWISE_EXPRESSION_TYPES,
                                              double);
using StatisticsGradSignature = Tensors(const Tensor&, at::TensorList,
                                        const Tensor&, const Tensor&,
                                        TORCHPAIRWISE_EXPRESSION_TYPES, double);
using WeightedStatisticsSignature = StatisticsOutputs(
    at::TensorList, TORCHPAIRWISE_EXPRESSION_TYPES, double, int64_t, int64_t);
using WeightedStatisticsGradSignature = Tensors(const Tensor&, at::TensorList,
                                                const Tensor&, const Tensor&,
                                                TORCHPAIRWISE_EXPRESSION_TYPES,
                                                double, int64_t, int64_t);
Tensor legacy_weighted_statistics_state(const Tensor& state);
StatisticsOutputs weighted_statistics_kernel(at::TensorList operands,
                                             TORCHPAIRWISE_EXPRESSION_DECL,
                                             double correction,
                                             int64_t value_root,
                                             int64_t weight_root);
Tensors weighted_statistics_backward_kernel(
    const Tensor& gradient, at::TensorList operands, const Tensor& result,
    const Tensor& state, TORCHPAIRWISE_EXPRESSION_DECL, double correction,
    int64_t value_root, int64_t weight_root);
Tensor expression_weighted_statistics(at::TensorList operands,
                                      const Settings& settings,
                                      c10::string_view reduction,
                                      bool global_reduction, double correction,
                                      int64_t value_root, int64_t weight_root);
// The last expression metadata flag denotes a global sample reduction here,
// rather than a bilinear form. All source packets have the same typed layout.
Settings statistics_settings(const ExpressionPacket& packet,
                             c10::string_view reduction, int64_t dtype,
                             int64_t size, bool exclusion, bool transposed);
StatisticsOutputs expression_statistics_kernel(at::TensorList operands,
                                               TORCHPAIRWISE_EXPRESSION_DECL,
                                               double correction);
Tensors expression_statistics_backward_kernel(
    const Tensor& gradient, at::TensorList operands, const Tensor& result,
    const Tensor& center, TORCHPAIRWISE_EXPRESSION_DECL, double correction);
Tensor expression_statistics(at::TensorList operands, const Settings& settings,
                             c10::string_view reduction, bool global_reduction,
                             double correction);
}  // namespace torchpairwise::ops::detail::streaming
