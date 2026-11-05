#pragma once

#include "expression_operations.h"

namespace torchpairwise::ops {
// Shared real linear primitives. Composite ATen implementations provide native
// autograd and compiler decompositions; no Python numerical callback is used.
TORCHPAIRWISE_API at::Tensor linear_transform(
    const at::Tensor& values, c10::string_view operation,
    const c10::optional<at::Tensor>& other = c10::nullopt,
    const c10::optional<at::Tensor>& factor = c10::nullopt);
TORCHPAIRWISE_API at::Tensor linear_diagonal_apply(const at::Tensor& values,
                                                   const at::Tensor& diagonal);
TORCHPAIRWISE_API at::Tensor linear_jacobi(const at::Tensor& diagonal);
TORCHPAIRWISE_API at::Tensor linear_reference_center(
    const at::Tensor& product, const at::Tensor& values,
    const at::Tensor& reference_mean, bool transposed = false);
TORCHPAIRWISE_API at::Tensor linear_lowrank_apply(const at::Tensor& values,
                                                  const at::Tensor& basis,
                                                  const at::Tensor& spectrum,
                                                  const at::Tensor& shift);

// Partition preparation stores compact IDs/counts, independent of label range.
TORCHPAIRWISE_API std::tuple<at::Tensor, at::Tensor> group_partition(
    const at::Tensor& labels);
TORCHPAIRWISE_API at::Tensor group_projection_apply(const at::Tensor& values,
                                                    const at::Tensor& indices,
                                                    const at::Tensor& counts);

// Postorder graph: source, add, subtract, compose, scale, diagonal shift,
// center, transpose, group projection (8), horizontal/vertical blocks (9/10),
// diagonal map (11), prepared low-rank inverse action (12), reference center
// (13). Group operands index integer partition tensors in factors;
// source/factor operands index their own lists.
struct LinearOperatorGraph {
  std::vector<ExpressionGraph> sources;
  std::vector<std::vector<int64_t>> nodes;
  std::vector<at::Tensor> factors;
};
struct LinearOperatorNode;
class TORCHPAIRWISE_API LinearOperator {
 public:
  explicit LinearOperator(ScalarExpression source);
  explicit LinearOperator(at::Tensor matrix);
  const std::vector<c10::SymInt>& shape() const;
  at::ScalarType dtype() const;
  c10::Device device() const;
  at::Tensor matmul(const at::Tensor& values) const;
  LinearOperatorGraph graph() const;
  LinearOperator compose(const LinearOperator& other) const;
  LinearOperator hstack(const LinearOperator& other) const;
  LinearOperator vstack(const LinearOperator& other) const;
  LinearOperator transpose() const;
  LinearOperator centered() const;
  LinearOperator centered(const at::Tensor& reference_mean) const;
  LinearOperator scale(at::Scalar factor) const;
  LinearOperator scale(const at::Tensor& factor) const;
  LinearOperator add_diagonal(at::Scalar factor) const;
  LinearOperator add_diagonal(const at::Tensor& factor) const;
  LinearOperator operator+(const LinearOperator& other) const;
  LinearOperator operator-(const LinearOperator& other) const;
  LinearOperator operator-() const;

 private:
  friend LinearOperator group_projection(const at::Tensor&, const at::Tensor&);
  friend LinearOperator diagonal(const at::Tensor&);
  friend LinearOperator lowrank_inverse_operator(at::TensorList);
  std::shared_ptr<const LinearOperatorNode> node_;
  explicit LinearOperator(std::shared_ptr<const LinearOperatorNode> node);
  LinearOperator unary(int64_t kind) const;
  LinearOperator binary(const LinearOperator& other, int64_t kind) const;
  LinearOperator scaled(int64_t kind, const c10::optional<at::Tensor>& tensor,
                        const c10::optional<at::Scalar>& number) const;
};
// like supplies only dtype/device. labels [...,N] are snapshotted by value.
TORCHPAIRWISE_API LinearOperator group_projection(const at::Tensor& labels,
                                                  const at::Tensor& like);
// A rectangular grid of explicit blocks, sharing matrix row/column partitions.
// No generated block matrix or zero padding is allocated.
TORCHPAIRWISE_API LinearOperator
block(const std::vector<std::vector<LinearOperator>>& blocks);
// Retain [...,N] entries; apply without constructing a [...,N,N] matrix.
TORCHPAIRWISE_API LinearOperator diagonal(const at::Tensor& values);
// Inverse action diag(1/d); d must be finite, strictly positive and
// representable.
TORCHPAIRWISE_API LinearOperator jacobi(const at::Tensor& values);
// Internal prepared-state adapter. Public setup factories validate and snapshot
// the double spectral factors; shift supplies the returned action dtype/device.
TORCHPAIRWISE_API LinearOperator lowrank_inverse_operator(at::TensorList state);
inline LinearOperator aslinearoperator(const ScalarExpression& source) {
  return LinearOperator(source);
}
inline LinearOperator aslinearoperator(const at::Tensor& matrix) {
  return LinearOperator(matrix);
}
inline LinearOperator aslinearoperator(const LinearOperator& value) {
  return value;
}
}  // namespace torchpairwise::ops
