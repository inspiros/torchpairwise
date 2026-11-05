#include "linear_operations.h"

#include <ATen/ExpandUtils.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include <algorithm>

#include "utils/value_checks.h"

namespace torchpairwise::ops {
namespace {
void check_values(const at::Tensor& value) {
  TORCH_CHECK(value.layout() == at::kStrided && value.dim() >= 2,
              "Linear blocks require dense strided [...,N,R] tensors");
  TORCH_CHECK(
      value.scalar_type() == at::kFloat || value.scalar_type() == at::kDouble,
      "Linear operators require float32 or float64");
}
void check_diagonal(const at::Tensor& value) {
  TORCH_CHECK(value.layout() == at::kStrided && value.dim() >= 1 &&
                  (value.scalar_type() == at::kFloat ||
                   value.scalar_type() == at::kDouble),
              "Diagonal entries must be strided float32/float64 [...,N]");
}
at::Tensor linear_diagonal_apply_impl(const at::Tensor& values,
                                      const at::Tensor& diagonal) {
  check_values(values);
  check_diagonal(diagonal);
  TORCH_CHECK(diagonal.scalar_type() == values.scalar_type() &&
                  diagonal.device() == values.device(),
              "Diagonal and RHS must share dtype/device");
  TORCH_CHECK(diagonal.sym_size(-1) == values.sym_size(-2),
              "Diagonal RHS sample count mismatch");
  return diagonal.unsqueeze(-1) * values;
}
at::Tensor linear_jacobi_impl(const at::Tensor& diagonal) {
  check_diagonal(diagonal);
  auto inverse = diagonal.reciprocal();
  utils::check_values(
      (diagonal.isfinite() & diagonal.gt(0) & inverse.isfinite()).all(),
      "Jacobi diagonal must be finite, strictly positive and have a finite "
      "reciprocal");
  return inverse;
}
at::Tensor linear_reference_center_impl(const at::Tensor& product,
                                        const at::Tensor& values,
                                        const at::Tensor& mean,
                                        bool transposed) {
  check_values(product);
  check_values(values);
  check_diagonal(mean);
  TORCH_CHECK(mean.scalar_type() == product.scalar_type() &&
                  values.scalar_type() == product.scalar_type() &&
                  mean.device() == product.device() &&
                  values.device() == product.device(),
              "Reference centering operands must share dtype/device");
  TORCH_CHECK(mean.sym_size(-1) > 0 &&
                  mean.sym_size(-1) == (transposed ? product.sym_size(-2)
                                                   : values.sym_size(-2)) &&
                  product.sym_size(-1) == values.sym_size(-1),
              "Reference centering requires a nonempty matching reference axis "
              "and channel dimensions");
  if (!transposed)
    return product - at::matmul(mean.unsqueeze(-2), values);
  auto corrected = product - mean.unsqueeze(-1) * values.sum(-2, true);
  return corrected - corrected.mean(-2, true);
}
at::Tensor linear_lowrank_apply_impl(const at::Tensor& values,
                                     const at::Tensor& basis,
                                     const at::Tensor& spectrum,
                                     const at::Tensor& shift) {
  check_values(values);
  TORCH_CHECK(basis.layout() == at::kStrided && basis.dim() >= 2 &&
                  basis.scalar_type() == at::kDouble &&
                  spectrum.layout() == at::kStrided &&
                  spectrum.scalar_type() == at::kDouble &&
                  spectrum.dim() == basis.dim() - 1,
              "Low-rank inverse requires double spectral state");
  auto expected = basis.sym_sizes().vec();
  expected.erase(expected.end() - 2);
  TORCH_CHECK(spectrum.sym_sizes().equals(expected) &&
                  basis.sym_size(-2) == values.sym_size(-2) &&
                  shift.layout() == at::kStrided && shift.dim() == 0 &&
                  shift.scalar_type() == values.scalar_type() &&
                  basis.device() == values.device() &&
                  spectrum.device() == values.device() &&
                  shift.device() == values.device(),
              "Low-rank inverse state must match sample dimensions and RHS "
              "dtype/device");
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto v = values.to(at::kDouble), alpha = shift.to(at::kDouble);
  auto projected = at::matmul(basis.transpose(-1, -2), v);
  // Split into parallel and orthogonal components in double precision instead
  // of subtracting a nearly equal Woodbury correction in source precision.
  auto orthogonal = v - at::matmul(basis, projected);
  return (orthogonal / alpha +
          at::matmul(basis, projected / (spectrum + alpha).unsqueeze(-1)))
      .to(values.scalar_type());
}
at::Tensor linear_transform_impl(const at::Tensor& values,
                                 c10::string_view operation,
                                 const c10::optional<at::Tensor>& other,
                                 const c10::optional<at::Tensor>& factor) {
  check_values(values);
  bool addition = operation == "add" || operation == "sub";
  bool shift = operation == "add_diagonal", scale = operation == "scale";
  bool vstack = operation == "vstack";
  TORCH_CHECK(addition || shift || scale || vstack || operation == "center",
              "Unknown linear transform: ", operation);
  TORCH_CHECK(bool(other) == (addition || shift || vstack) &&
                  bool(factor) == (shift || scale),
              "Missing or unused linear transform operand");
  if (other) {
    check_values(*other);
    TORCH_CHECK(other->scalar_type() == values.scalar_type() &&
                    other->device() == values.device(),
                "Linear operands must share dtype and device");
    TORCH_CHECK((vstack || other->sym_size(-2) == values.sym_size(-2)) &&
                    other->sym_size(-1) == values.sym_size(-1),
                "Linear block sample/channel dimensions must match");
  }
  if (factor)
    TORCH_CHECK(factor->layout() == at::kStrided && factor->dim() == 0 &&
                    factor->scalar_type() == values.scalar_type() &&
                    factor->device() == values.device(),
                "Linear factors must be scalar tensors matching dtype/device");
  if (operation == "center")
    return values - values.mean(-2, true);
  if (vstack) {
    auto batch =
        at::infer_size_symint(values.sym_sizes().slice(0, values.dim() - 2),
                              other->sym_sizes().slice(0, other->dim() - 2));
    auto first = batch, second = batch;
    first.insert(first.end(), {values.sym_size(-2), values.sym_size(-1)});
    second.insert(second.end(), {other->sym_size(-2), other->sym_size(-1)});
    return at::cat({values.expand_symint(first), other->expand_symint(second)},
                   -2);
  }
  if (addition)
    return operation == "add" ? values + *other : values - *other;
  return shift ? values + *factor * *other : values * *factor;
}
constexpr int64_t Source = 0, Add = 1, Subtract = 2, Compose = 3, Scale = 4,
                  Shift = 5, Center = 6, Transpose = 7, Group = 8, HStack = 9,
                  VStack = 10, Diagonal = 11, LowrankInverse = 12,
                  ReferenceCenter = 13;
}  // namespace

struct LinearOperatorNode {
  int64_t kind = Source;
  ScalarExpressionMetadata metadata{{}, at::kFloat, c10::Device(at::kCPU)};
  std::shared_ptr<const LinearOperatorNode> left, right;
  c10::optional<ScalarExpression> source;
  c10::optional<at::Tensor> factor;
  c10::optional<at::Scalar> number;
  at::Tensor indices, counts;
  std::vector<at::Tensor> spectral;
};
namespace {
at::Tensor apply_linear(const std::shared_ptr<const LinearOperatorNode>& node,
                        const at::Tensor& values, bool transpose = false) {
  auto kind = node->kind;
  if (kind == Diagonal)
    return linear_diagonal_apply(values, *node->factor);
  if (kind == LowrankInverse)
    return linear_lowrank_apply(values, node->spectral[0], node->spectral[1],
                                node->spectral[2]);
  if (kind == Group)
    return group_projection_apply(values, node->indices, node->counts);
  if (kind == Source)
    return (transpose ? node->source->transpose() : *node->source)
        .matmul(values);
  if (kind == Transpose)
    return apply_linear(node->left, values, !transpose);
  if (kind == Compose) {
    auto inner = transpose ? node->left : node->right;
    auto outer = transpose ? node->right : node->left;
    return apply_linear(outer, apply_linear(inner, values, transpose),
                        transpose);
  }
  if (kind == HStack || kind == VStack) {
    bool split = (kind == HStack) != transpose;
    if (split) {
      auto axis = node->left->metadata.shape.size() - (transpose ? 2 : 1);
      auto size = node->left->metadata.shape[axis];
      return linear_transform(
          apply_linear(node->left, values.narrow_symint(-2, 0, size),
                       transpose),
          "add",
          apply_linear(
              node->right,
              values.narrow_symint(-2, size, values.sym_size(-2) - size),
              transpose));
    }
    return linear_transform(apply_linear(node->left, values, transpose),
                            "vstack",
                            apply_linear(node->right, values, transpose));
  }
  if (kind == Center)
    return linear_transform(
        apply_linear(node->left, linear_transform(values, "center"), transpose),
        "center");
  if (kind == ReferenceCenter) {
    auto rhs = transpose ? values : linear_transform(values, "center");
    return linear_reference_center(apply_linear(node->left, rhs, transpose),
                                   rhs, *node->factor, transpose);
  }
  auto value = apply_linear(node->left, values, transpose);
  if (kind == Add || kind == Subtract)
    return linear_transform(value, kind == Add ? "add" : "sub",
                            apply_linear(node->right, values, transpose));
  auto factor = node->factor.value_or(at::Tensor());
  if (!factor.defined())
    factor = at::scalar_tensor(*node->number, value.options());
  return linear_transform(
      value, kind == Scale ? "scale" : "add_diagonal",
      kind == Shift ? c10::optional<at::Tensor>(values) : c10::nullopt, factor);
}
}  // namespace

at::Tensor linear_transform(const at::Tensor& values,
                            c10::string_view operation,
                            const c10::optional<at::Tensor>& other,
                            const c10::optional<at::Tensor>& factor) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::linear_transform", "")
                       .typed<decltype(linear_transform_impl)>();
  return op.call(values, operation, other, factor);
}
at::Tensor linear_diagonal_apply(const at::Tensor& values,
                                 const at::Tensor& diagonal) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::linear_diagonal_apply", "")
          .typed<decltype(linear_diagonal_apply_impl)>();
  return op.call(values, diagonal);
}
at::Tensor linear_jacobi(const at::Tensor& diagonal) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::linear_jacobi", "")
                       .typed<decltype(linear_jacobi_impl)>();
  return op.call(diagonal);
}
at::Tensor linear_reference_center(const at::Tensor& product,
                                   const at::Tensor& values,
                                   const at::Tensor& mean, bool transposed) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::linear_reference_center", "")
          .typed<decltype(linear_reference_center_impl)>();
  return op.call(product, values, mean, transposed);
}
at::Tensor linear_lowrank_apply(const at::Tensor& values,
                                const at::Tensor& basis,
                                const at::Tensor& spectrum,
                                const at::Tensor& shift) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::linear_lowrank_apply", "")
          .typed<decltype(linear_lowrank_apply_impl)>();
  return op.call(values, basis, spectrum, shift);
}
LinearOperator::LinearOperator(std::shared_ptr<const LinearOperatorNode> node)
    : node_(std::move(node)) {}
LinearOperator::LinearOperator(ScalarExpression source) {
  auto node = std::make_shared<LinearOperatorNode>();
  node->metadata = source.metadata();
  TORCH_CHECK(
      node->metadata.dtype == at::kFloat || node->metadata.dtype == at::kDouble,
      "Linear operators require float32 or float64 entries");
  node->source = std::move(source);
  node_ = std::move(node);
}
LinearOperator::LinearOperator(at::Tensor matrix)
    : LinearOperator(matrix_expression(matrix)) {}
const std::vector<c10::SymInt>& LinearOperator::shape() const {
  return node_->metadata.shape;
}
at::ScalarType LinearOperator::dtype() const {
  return node_->metadata.dtype;
}
c10::Device LinearOperator::device() const {
  return node_->metadata.device;
}
at::Tensor LinearOperator::matmul(const at::Tensor& values) const {
  TORCH_CHECK(values.layout() == at::kStrided && values.dim() >= 1 &&
                  values.scalar_type() == dtype() &&
                  values.device() == device(),
              "Linear RHS must be dense, real and match operator dtype/device");
  bool vector = values.dim() == 1;
  auto block = vector ? values.unsqueeze(-1) : values;
  TORCH_CHECK(block.sym_size(-1) > 0,
              "Linear RHS channel count must be positive");
  TORCH_CHECK(block.sym_size(-2) == shape().back(),
              "Linear RHS sample count mismatch");
  auto result = apply_linear(node_, block);
  auto expected = at::infer_size_symint(
      c10::SymIntArrayRef(shape()).slice(0, shape().size() - 2),
      block.sym_sizes().slice(0, block.dim() - 2));
  expected.push_back(shape()[shape().size() - 2]);
  expected.push_back(block.sym_size(-1));
  TORCH_CHECK(result.sym_sizes().equals(expected),
              "Bound source shape changed; reconstruct the linear operator");
  return vector ? result.squeeze(-1) : result;
}
LinearOperatorGraph LinearOperator::graph() const {
  LinearOperatorGraph graph;
  std::vector<std::pair<const LinearOperatorNode*, int64_t>> seen;
  std::function<int64_t(const std::shared_ptr<const LinearOperatorNode>&)>
      visit;
  visit = [&](const std::shared_ptr<const LinearOperatorNode>& n) {
    for (const auto& entry : seen)
      if (entry.first == n.get())
        return entry.second;
    int64_t a = -1, b = -1;
    if (n->kind == LowrankInverse) {
      a = graph.factors.size();
      graph.factors.insert(graph.factors.end(), n->spectral.begin(),
                           n->spectral.end());
      b = a + 1;
    } else if (n->kind == Diagonal) {
      a = graph.factors.size();
      graph.factors.push_back(*n->factor);
    } else if (n->kind == Group) {
      a = graph.factors.size();
      graph.factors.push_back(n->indices);
      b = graph.factors.size();
      graph.factors.push_back(n->counts);
    } else if (n->kind == Source) {
      a = graph.sources.size();
      graph.sources.push_back(n->source->graph());
    } else {
      a = visit(n->left);
      if (n->kind == Add || n->kind == Subtract || n->kind == Compose ||
          n->kind == HStack || n->kind == VStack)
        b = visit(n->right);
      else if (n->kind == Scale || n->kind == Shift ||
               n->kind == ReferenceCenter) {
        b = graph.factors.size();
        graph.factors.push_back(
            n->factor
                ? *n->factor
                : at::scalar_tensor(
                      *n->number,
                      at::TensorOptions().dtype(dtype()).device(device())));
      }
    }
    auto id = int64_t(graph.nodes.size());
    graph.nodes.push_back({n->kind, a, b});
    seen.emplace_back(n.get(), id);
    return id;
  };
  visit(node_);
  return graph;
}
LinearOperator diagonal(const at::Tensor& values) {
  check_diagonal(values);
  auto node = std::make_shared<LinearOperatorNode>();
  node->kind = Diagonal;
  node->factor = values;
  node->metadata = {values.sym_sizes().vec(), values.scalar_type(),
                    values.device()};
  node->metadata.shape.push_back(values.sym_size(-1));
  return LinearOperator(std::move(node));
}
LinearOperator jacobi(const at::Tensor& values) {
  return torchpairwise::ops::diagonal(linear_jacobi(values));
}
LinearOperator lowrank_inverse_operator(at::TensorList state) {
  TORCH_CHECK(state.size() == 3 && state[0].dim() >= 2 && state[1].dim() >= 1 &&
                  state[2].dim() == 0,
              "Invalid prepared low-rank state");
  auto node = std::make_shared<LinearOperatorNode>();
  node->kind = LowrankInverse;
  node->spectral = state.vec();
  auto shape = state[0].sym_sizes().vec();
  shape.back() = shape[shape.size() - 2];
  node->metadata = {std::move(shape), state[2].scalar_type(),
                    state[2].device()};
  return LinearOperator(std::move(node));
}
LinearOperator group_projection(const at::Tensor& labels,
                                const at::Tensor& like) {
  TORCH_CHECK(
      like.layout() == at::kStrided && (like.scalar_type() == at::kFloat ||
                                        like.scalar_type() == at::kDouble),
      "Group projection like must be a strided float32/float64 Tensor");
  TORCH_CHECK(labels.device() == like.device(),
              "Group labels and like must share a device");
  auto [indices, counts] = group_partition(labels);
  auto node = std::make_shared<LinearOperatorNode>();
  node->kind = Group;
  node->indices = std::move(indices);
  node->counts = std::move(counts);
  node->metadata = {labels.sym_sizes().vec(), like.scalar_type(),
                    like.device()};
  node->metadata.shape.push_back(labels.sym_size(-1));
  return LinearOperator(std::move(node));
}
LinearOperator LinearOperator::unary(int64_t kind) const {
  auto node = std::make_shared<LinearOperatorNode>();
  node->kind = kind;
  node->left = node_;
  node->metadata = node_->metadata;
  if (kind == Transpose)
    std::iter_swap(node->metadata.shape.end() - 1,
                   node->metadata.shape.end() - 2);
  return LinearOperator(std::move(node));
}
LinearOperator LinearOperator::binary(const LinearOperator& other,
                                      int64_t kind) const {
  TORCH_CHECK(dtype() == other.dtype() && device() == other.device(),
              "Linear operators must share dtype and device");
  const auto& a = shape();
  const auto& b = other.shape();
  if (kind == Compose) {
    TORCH_CHECK(a.back() == b[b.size() - 2],
                "Linear composition contracted dimensions must match");
  } else if (kind == HStack || kind == VStack) {
    TORCH_CHECK(kind == HStack ? a[a.size() - 2] == b[b.size() - 2]
                               : a.back() == b.back(),
                "Block concatenation dimensions do not match");
  } else {
    TORCH_CHECK(a[a.size() - 2] == b[b.size() - 2] && a.back() == b.back(),
                "Linear addition/subtraction requires equal matrix dimensions");
  }
  auto node = std::make_shared<LinearOperatorNode>();
  node->kind = kind;
  node->left = node_;
  node->right = other.node_;
  node->metadata = node_->metadata;
  auto& out = node->metadata.shape;
  out = at::infer_size_symint(c10::SymIntArrayRef(a).slice(0, a.size() - 2),
                              c10::SymIntArrayRef(b).slice(0, b.size() - 2));
  out.push_back(kind == VStack ? a[a.size() - 2] + b[b.size() - 2]
                               : a[a.size() - 2]);
  out.push_back(kind == HStack    ? a.back() + b.back()
                : kind == Compose ? b.back()
                                  : a.back());
  return LinearOperator(std::move(node));
}
LinearOperator LinearOperator::scaled(
    int64_t kind, const c10::optional<at::Tensor>& tensor,
    const c10::optional<at::Scalar>& number) const {
  if (kind == Shift)
    TORCH_CHECK(shape()[shape().size() - 2] == shape().back(),
                "Diagonal shifts require a square linear operator");
  if (tensor) {
    TORCH_CHECK(tensor->layout() == at::kStrided && tensor->dim() == 0 &&
                    tensor->scalar_type() == dtype() &&
                    tensor->device() == device(),
                "Linear factors must be scalar tensors matching dtype/device");
  } else {
    TORCH_CHECK(number && !number->isBoolean() && !number->isComplex(),
                "Linear factors must be real scalars");
  }
  auto node = std::make_shared<LinearOperatorNode>();
  node->kind = kind;
  node->left = node_;
  node->metadata = node_->metadata;
  node->factor = tensor;
  node->number = number;
  return LinearOperator(std::move(node));
}
LinearOperator LinearOperator::compose(const LinearOperator& other) const {
  return binary(other, Compose);
}
LinearOperator LinearOperator::hstack(const LinearOperator& other) const {
  return binary(other, HStack);
}
LinearOperator LinearOperator::vstack(const LinearOperator& other) const {
  return binary(other, VStack);
}
LinearOperator block(const std::vector<std::vector<LinearOperator>>& blocks) {
  TORCH_CHECK(!blocks.empty() && !blocks[0].empty(),
              "block requires a nonempty rectangular grid");
  std::vector<LinearOperator> rows;
  for (const auto& row : blocks) {
    TORCH_CHECK(row.size() == blocks[0].size(),
                "block requires a rectangular grid");
    auto joined = row[0];
    for (size_t j = 0; j < row.size(); ++j) {
      TORCH_CHECK(row[j].shape().back() == blocks[0][j].shape().back(),
                  "block column widths must agree");
      if (j)
        joined = joined.hstack(row[j]);
    }
    rows.push_back(std::move(joined));
  }
  auto result = rows[0];
  for (size_t i = 1; i < rows.size(); ++i)
    result = result.vstack(rows[i]);
  return result;
}
LinearOperator LinearOperator::transpose() const {
  return unary(Transpose);
}
LinearOperator LinearOperator::centered() const {
  return unary(Center);
}
LinearOperator LinearOperator::centered(
    const at::Tensor& reference_mean) const {
  check_diagonal(reference_mean);
  TORCH_CHECK(reference_mean.scalar_type() == dtype() &&
                  reference_mean.device() == device(),
              "Reference mean must match linear operator dtype/device");
  TORCH_CHECK(
      reference_mean.sym_size(-1) == shape().back() && shape().back() > 0,
      "Reference mean must match the nonempty reference axis");
  auto node = std::make_shared<LinearOperatorNode>();
  node->kind = ReferenceCenter;
  node->left = node_;
  node->factor = reference_mean;
  node->metadata = node_->metadata;
  auto& out = node->metadata.shape;
  auto batch = at::infer_size_symint(
      c10::SymIntArrayRef(shape()).slice(0, shape().size() - 2),
      reference_mean.sym_sizes().slice(0, reference_mean.dim() - 1));
  batch.insert(batch.end(), {shape()[shape().size() - 2], shape().back()});
  out = std::move(batch);
  return LinearOperator(std::move(node));
}
LinearOperator LinearOperator::scale(at::Scalar factor) const {
  return scaled(Scale, c10::nullopt, factor);
}
LinearOperator LinearOperator::scale(const at::Tensor& factor) const {
  return scaled(Scale, factor, c10::nullopt);
}
LinearOperator LinearOperator::add_diagonal(at::Scalar factor) const {
  return scaled(Shift, c10::nullopt, factor);
}
LinearOperator LinearOperator::add_diagonal(const at::Tensor& factor) const {
  return scaled(Shift, factor, c10::nullopt);
}
LinearOperator LinearOperator::operator+(const LinearOperator& other) const {
  return binary(other, Add);
}
LinearOperator LinearOperator::operator-(const LinearOperator& other) const {
  return binary(other, Subtract);
}
LinearOperator LinearOperator::operator-() const {
  return scale(-1.);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def("linear_diagonal_apply(Tensor values, Tensor diagonal) -> Tensor");
  m.def("linear_jacobi(Tensor diagonal) -> Tensor");
  m.def(
      "linear_reference_center(Tensor product, Tensor values, Tensor "
      "reference_mean, bool transposed=False) -> Tensor");
  m.def(
      "linear_lowrank_apply(Tensor values, Tensor basis, Tensor spectrum, "
      "Tensor shift) -> Tensor");
  m.def(
      "linear_transform(Tensor values, str operation, Tensor? other=None, "
      "Tensor? factor=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeImplicitAutograd, m) {
  m.impl("linear_diagonal_apply", TORCH_FN(linear_diagonal_apply_impl));
  m.impl("linear_jacobi", TORCH_FN(linear_jacobi_impl));
  m.impl("linear_reference_center", TORCH_FN(linear_reference_center_impl));
  m.impl("linear_lowrank_apply", TORCH_FN(linear_lowrank_apply_impl));
  m.impl("linear_transform", TORCH_FN(linear_transform_impl));
}
}  // namespace torchpairwise::ops
