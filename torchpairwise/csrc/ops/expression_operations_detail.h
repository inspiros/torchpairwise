#pragma once
#include "pairwise_operations_detail.h"

namespace torchpairwise::ops::detail::streaming {
// Serialized native expression program; opcodes are part of the new schema ABI.
// 0 source, 1 scalar tensor, 2 scalar number, 3 add, 4 subtract, 5 multiply,
// 6 square, 7 exp, 8 negate, 9 row weighting, 10 column weighting.
// 12 divide, 13 power, 14 minimum, 15 maximum, 16/17 clamp min/max,
// 18 sqrt, 19 log, 20 sin, 21 cos, 22 tanh, 23 sigmoid, 24 abs,
// 25 logical not, 26..31 lt/le/gt/ge/eq/ne, 32/33 logical and/or,
// 34 where, 35 branch pair, 38/39 row/column values, 40/41 row/column ids.
// 11 is facade-only transpose; 36/37 are reserved.
inline bool expression_binary(int64_t op) {
  return op == 3 || op == 4 || op == 5 || op == 12 || op == 13 || op == 14 ||
         op == 15 || (op >= 26 && op <= 33);
}
inline bool expression_unary(int64_t op) {
  return op == 6 || op == 7 || op == 8 || (op >= 18 && op <= 25);
}
inline bool expression_axis_slot(int64_t op) {
  return op == 9 || op == 10 || op == 38 || op == 39;
}
inline bool expression_row_slot(int64_t op) {
  return op == 9 || op == 38;
}
inline bool expression_scalar_bound(int64_t op) {
  return op == 16 || op == 17;
}
struct ExpressionPacket {
  std::vector<std::string> names, families;
  std::vector<std::vector<int64_t>> headers, nodes;
  std::vector<Reals> reals;
  std::vector<Integers> integers;
  std::vector<Flags> flags;
  std::vector<at::Scalar> numbers;
};
struct ExpressionPlan {
  ExpressionPacket packet;
  std::vector<Settings> leaves;
};
// Dtype of a lowered scalar number combined with partner (undefined if none).
at::ScalarType expression_number_dtype(const at::Scalar& value,
                                       const Tensor& partner);
// Per-node lowering dtypes for scalar number nodes on probe operands.
std::vector<at::ScalarType> expression_number_dtypes(at::TensorList operands,
                                                     const ExpressionPlan& plan,
                                                     const Tensors& leaves);
at::ScalarType infer_expression_dtype(at::TensorList operands,
                                      const ExpressionPlan& plan,
                                      const Tensors& leaves, int64_t root = -1);
std::shared_ptr<const ExpressionPlan> expression_plan(
    const ExpressionPacket& packet);
Tensors expression_local_inputs(at::TensorList operands,
                                const Settings& settings,
                                const std::vector<int64_t>& index, int64_t left,
                                int64_t count, int64_t right, int64_t other,
                                const Tensor& rows, const Tensor& columns,
                                bool replay);
Tensor expression_evaluate(at::TensorList local, const Settings& settings,
                           bool paired, ExpressionReplay* replay = nullptr);
Tensor expression_evaluate_higher_order(at::TensorList local,
                                        const Settings& settings, bool paired);
Tensors expression_evaluate_roots(at::TensorList local,
                                  const Settings& settings,
                                  at::IntArrayRef roots,
                                  ExpressionReplay* replay = nullptr);
void expression_accumulate_roots(at::TensorList local,
                                 at::TensorList cotangents,
                                 at::IntArrayRef roots, Tensors& buffers,
                                 const Settings& settings,
                                 const std::vector<int64_t>& index,
                                 const ExpressionReplay* replay = nullptr);
void expression_accumulate(at::TensorList local, const Tensor& cotangent,
                           Tensors& buffers, const Settings& settings,
                           const std::vector<int64_t>& index,
                           const ExpressionReplay* replay = nullptr);
void expression_select(PreparedComparison& prepared, c10::string_view operation,
                       const PairwiseOperationOptions& options);

#define TORCHPAIRWISE_EXPRESSION_DECL                                       \
  const std::vector<std::string>&names,                                     \
      const std::vector<std::string>&families,                              \
      const std::vector<std::vector<int64_t>>&headers,                      \
      const std::vector<Reals>&reals, const std::vector<Integers>&integers, \
      const std::vector<Flags>&flags,                                       \
      const std::vector<std::vector<int64_t>>&nodes,                        \
      at::ArrayRef<at::Scalar> numbers, c10::string_view operation,         \
      int64_t dtype, int64_t size, bool exclusion, bool transposed, bool form
#define TORCHPAIRWISE_EXPRESSION_TYPES                                     \
  const std::vector<std::string>&, const std::vector<std::string>&,        \
      const std::vector<std::vector<int64_t>>&, const std::vector<Reals>&, \
      const std::vector<Integers>&, const std::vector<Flags>&,             \
      const std::vector<std::vector<int64_t>>&, at::ArrayRef<at::Scalar>,  \
      c10::string_view, int64_t, int64_t, bool, bool, bool
#define TORCHPAIRWISE_EXPRESSION_ARGS                                          \
  names, families, headers, reals, integers, flags, nodes, numbers, operation, \
      dtype, size, exclusion, transposed, form
using ExpressionSignature = Outputs(at::TensorList,
                                    TORCHPAIRWISE_EXPRESSION_TYPES);
using ExpressionTopKSignature = Outputs(at::TensorList,
                                        TORCHPAIRWISE_EXPRESSION_TYPES, int64_t,
                                        bool);
using ExpressionTopKGradSignature = Tensors(const Tensor&, at::TensorList,
                                            const Tensor&, const Tensor&,
                                            const Tensor&, const Tensor&,
                                            TORCHPAIRWISE_EXPRESSION_TYPES,
                                            int64_t, bool);
Outputs expression_topk(at::TensorList operands, const Settings& settings,
                        int64_t k, bool largest);
using ExpressionGradSignature = Tensors(const Tensor&, at::TensorList,
                                        const Tensor&, const Tensor&,
                                        const Tensor&, const Tensor&,
                                        TORCHPAIRWISE_EXPRESSION_TYPES);
Outputs expression_consume(at::TensorList operands,
                           TORCHPAIRWISE_EXPRESSION_DECL);
Tensors expression_consume_backward(const Tensor& gradient,
                                    at::TensorList operands,
                                    const Tensor& result, const Tensor& indices,
                                    const Tensor& zeros, const Tensor& nonzero,
                                    TORCHPAIRWISE_EXPRESSION_DECL);
}  // namespace torchpairwise::ops::detail::streaming
