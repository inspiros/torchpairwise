#pragma once
#include "linear_plan_detail.h"

namespace torchpairwise::ops::detail::linear {
using SolveTuple = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>;
using ImplicitSolveSignature = SolveTuple(
    TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef shape,
    const Tensor& rhs, c10::string_view method, double rtol, double atol,
    int64_t max_iter, int64_t restart, const c10::optional<Tensor>& initial);
using ImplicitGradSignature = Tensors(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                                      c10::SymIntArrayRef shape,
                                      const Tensor& rhs, const Tensor& solution,
                                      const Tensor& gradient,
                                      c10::string_view method, double rtol,
                                      double atol, int64_t max_iter,
                                      int64_t restart);
using LeastSquaresTuple =
    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
using ImplicitLstsqSignature = LeastSquaresTuple(
    TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef shape,
    const Tensor& rhs, double damp, double rtol, double atol, int64_t max_iter,
    const c10::optional<Tensor>& initial);
using ImplicitLstsqGradSignature =
    Tensors(TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef shape,
            const Tensor& rhs, const Tensor& solution, const Tensor& gradient,
            double damp, double rtol, double atol, int64_t max_iter);
Packet packet_from_args(TORCHPAIRWISE_LINEAR_PACKET_DECL);
// Flatten only tensor fields, in inputs/factors/linear_factors/dictionary
// order. Save these through save_for_backward, never in an unversioned metadata
// state.
Tensors packet_values(const Packet& packet);
Packet replace_values(const Packet& metadata, at::TensorList values);
SolveTuple implicit_solve(const Packet& packet, c10::SymIntArrayRef shape,
                          const Tensor& rhs, c10::string_view method,
                          double rtol, double atol, int64_t max_iter,
                          int64_t restart, const c10::optional<Tensor>& initial,
                          bool preconditioned = false);
}  // namespace torchpairwise::ops::detail::linear
