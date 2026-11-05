#pragma once
// Fused native reducers: CUDA kernels that evaluate low-dimensional
// Minkowski-family distances, radial kernels and dot-product kernels in
// registers and reduce them on the fly, replacing the materialized tile of
// the streaming path.
// Design: .maintainer/step4-fused-reducers.md (local).
#include <ATen/ATen.h>

#include <utility>
#include <vector>

#include "streaming_reductions_detail.h"

namespace torchpairwise::ops::detail::fused {
using at::Tensor;
// Longest expression chain (pairs::max_chain).
constexpr int max_chain = 8;
struct Plan {
  int feature, outer;  // pairs::Feature, pairs::Outer
  double power;        // Feature::Power exponent
  double a, b, c, d;   // OuterParams before the cast to the input dtype
  // Expression chain applied to the metric value (pairs::ChainOp codes). An
  // op's operand is a number (axis 0) or a value of the output row (axis 1)
  // or reduced column (axis 2). Numbers, and the fills of Keep/Drop, are the
  // lowered 0-dim device tensors in the value dtype (undefined for Neg and
  // Exp); kernels read them on device, so no host sync. Row/column values are
  // [B, samples] contiguous in the value dtype, or [1, samples] when shared
  // across the batch; slot is their operand slot when they take gradients
  // (floating), else -1.
  int chain_size;
  int chain_op[max_chain];
  int chain_axis[max_chain];
  Tensor chain_constant[max_chain];
  Tensor chain_values[max_chain];
  int64_t chain_slot[max_chain];
  // Output rows and reduced columns as [B, N, D] and [B, M, D], contiguous.
  Tensor rows, columns;
  // Operand slots the rows/columns came from, and (expressions) the index maps
  // from output rows/columns to those operands' sample axes.
  int64_t row_slot, column_slot;
  Tensor row_map, column_map;
  std::vector<int64_t> batch;
  bool differentiable;
};
// Largest feature count of the fused gradient kernels; larger rows keep the
// tile replay (per-row gradient registers would spill).
constexpr int64_t max_gradient_features = 16;
// Largest fused top-k size; larger k keeps the tile path.
constexpr int64_t max_topk = 64;
// The same plan with output rows and reduced columns exchanged. Every v1
// metric is symmetric, so this evaluates the transposed operator.
inline Plan swapped(Plan p) {
  std::swap(p.rows, p.columns);
  std::swap(p.row_slot, p.column_slot);
  std::swap(p.row_map, p.column_map);
  for (int k = 0; k < p.chain_size; ++k)
    if (p.chain_axis[k] != 0)
      p.chain_axis[k] = 3 - p.chain_axis[k];
  return p;
}
// Process-wide switch for A/B tests; on by default.
bool enabled();
// Returns a plan when settings, operands and device admit a fused kernel:
// float32 or float64 CUDA operands of a named leaf, or an expression of one
// leaf under add/sub/mul/div by numbers or row/column values, neg, exp,
// row/column weighting and where masks on row/column conditions. Selections
// of dot-product kernels keep the tile path (GEMM rounding).
c10::optional<Plan> plan(at::TensorList v, const streaming::Settings& s);
// Counts fused kernel entries (tests check dispatch through compiled graphs).
void count();
// Scatters a [B, samples, D] gradient of the plan's rows (or columns) back
// to the shape of its source operand.
Tensor to_operand(const Plan& p, const Tensor& gradient, bool rows,
                  at::TensorList v);

// Optional softmax weighting: value -> exp(value - shift[k]) / mass[k] with
// k the output row (or the reduced column when on_columns); center (gradient
// only) is subtracted from the channel dot product before weighting.
struct Probability {
  Tensor shift, mass, center;  // [B, samples] in the input dtype
  bool on_columns = false;
};

#ifdef WITH_CUDA
// sum_j value(rows_i, columns_j) * payload[j, :] as [B, N, C]; payload is
// [..., M, C]. exclusion drops j == i (reductions only).
Tensor matmul_cuda(const Plan& plan, const Tensor& payload, bool exclusion,
                   const Probability& probability = {});
// sum_j w_ij * d value(rows_i, columns_j) / d rows_i as [B, N, D] with
// w_ij = left_i . right_j (optionally weighted, see Probability); left is
// [..., N, C] and right [..., M, C]. exclusion zeroes w_ii. Requires
// plan.differentiable and D <= 16.
Tensor gradient_cuda(const Plan& plan, const Tensor& left, const Tensor& right,
                     bool exclusion, const Probability& probability = {});
// Gradients of every operand the pair values read, as gradient_cuda for the
// rows (left, rows weighting) and the columns (right, columns weighting),
// plus the chain's row/column values, written to their slots of result
// (zeros_like(v) on entry). Requires what gradient_cuda requires.
void operand_gradients(const Plan& plan, const Tensor& left,
                       const Tensor& right, bool exclusion,
                       const Probability& rows, const Probability& columns,
                       std::vector<Tensor>& result, at::TensorList v);
// Row log-sum-exp of the plan's scores, [B, N] float64, as the tile path's
// logsumexp (NaN and +inf propagate; empty or all -inf rows give -inf).
Tensor logsumexp_cuda(const Plan& plan, bool exclusion);
// Row log-sum-exp of scale * value(rows_i, columns_j) + bias[b, j], [B, N]
// float64, for the Sinkhorn half steps. scale is a 0-dim float64 device
// tensor (read on device), bias float64 [B, M] or shared [M], and reference
// a float64 [B, N] or [N] guess of the result: any finite guess gives the
// same value, and a close one keeps float32 score gaps accurate when the
// bias and the scaled cost are large and cancel.
Tensor biased_logsumexp_cuda(const Plan& plan, const Tensor& scale,
                             const Tensor& bias, const Tensor& reference);
// Softmax row state [B, N, 2] float64 (maximum, mass) as the tile path's
// softmax pass 1.
Tensor softmax_state_cuda(const Plan& plan, bool exclusion);
// The k first columns of each row under the tile path's selection order
// (category, key, column), as values [B, N, k] and column indices; requires
// 1 <= k <= min(max_topk, eligible columns).
std::pair<Tensor, Tensor> topk_cuda(const Plan& plan, int64_t k, bool largest,
                                    bool nan_first, bool exclusion);
#endif
}  // namespace torchpairwise::ops::detail::fused
