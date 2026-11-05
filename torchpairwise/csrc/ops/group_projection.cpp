#include <ATen/ExpandUtils.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include "linear_operations.h"

namespace torchpairwise::ops {
namespace {
using Partition = std::tuple<at::Tensor, at::Tensor>;
void check_partition_tensor(const at::Tensor& labels) {
  TORCH_CHECK(labels.layout() == at::kStrided && labels.dim() >= 1 &&
                  labels.scalar_type() == at::kLong,
              "Group labels/partition must be dense int64 with shape [...,N]");
}
Partition group_partition_impl(const at::Tensor& labels) {
  check_partition_tensor(labels);
  // Fixed-size IDs/counts avoid unique()'s data-dependent output shape. Labels
  // may be negative, noncontiguous or near int64 limits. No arithmetic on them.
  auto n = labels.sym_size(-1);
  auto [ordered, order] = labels.sort(-1);
  auto boundary = at::cat({at::ones_like(labels.narrow_symint(-1, 0, n.min(1)),
                                         labels.options().dtype(at::kBool)),
                           ordered.slice(-1, 1).ne(ordered.slice(-1, 0, -1))},
                          -1);
  auto sorted_ids = boundary.cumsum(-1, at::kLong) - 1;
  auto indices = at::zeros_like(labels).scatter(-1, order, sorted_ids);
  auto counts =
      at::zeros_like(labels).scatter_add(-1, indices, at::ones_like(labels));
  return {indices, counts};
}
Partition group_partition_meta(const at::Tensor& labels) {
  check_partition_tensor(labels);
  return {at::empty_symint(labels.sym_sizes(), labels.options()),
          at::empty_symint(labels.sym_sizes(), labels.options())};
}
at::Tensor group_projection_apply_impl(const at::Tensor& values,
                                       const at::Tensor& indices,
                                       const at::Tensor& counts) {
  check_partition_tensor(indices);
  check_partition_tensor(counts);
  TORCH_CHECK(indices.sym_sizes().equals(counts.sym_sizes()) &&
                  indices.device() == counts.device(),
              "Group partition shapes/devices must match");
  TORCH_CHECK(values.layout() == at::kStrided && values.dim() >= 2 &&
                  (values.scalar_type() == at::kFloat ||
                   values.scalar_type() == at::kDouble) &&
                  values.device() == indices.device(),
              "Group projection values must be strided float32/float64 on the "
              "partition device");
  TORCH_CHECK(
      values.sym_size(-2) == indices.sym_size(-1) && values.sym_size(-1) > 0,
      "Group projection values require matching sample count and positive "
      "channel count");
  auto shape =
      at::infer_size_symint(values.sym_sizes().slice(0, values.dim() - 2),
                            indices.sym_sizes().slice(0, indices.dim() - 1));
  shape.push_back(indices.sym_size(-1));
  auto count = counts.expand_symint(shape);
  shape.push_back(values.sym_size(-1));
  auto ids = indices.unsqueeze(-1).expand_symint(shape);
  auto input = values.expand_symint(shape);
  auto sums =
      at::zeros_symint(shape, input.options()).scatter_add(-2, ids, input);
  auto denominator = count.gather(-1, indices.expand_symint(count.sym_sizes()))
                         .clamp_min(1)
                         .to(values.scalar_type());
  return sums.gather(-2, ids) / denominator.unsqueeze(-1);
}
}  // namespace
Partition group_partition(const at::Tensor& labels) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_group_partition", "")
                       .typed<decltype(group_partition_impl)>();
  return op.call(labels);
}
at::Tensor group_projection_apply(const at::Tensor& values,
                                  const at::Tensor& indices,
                                  const at::Tensor& counts) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::group_projection_apply", "")
          .typed<decltype(group_projection_apply_impl)>();
  return op.call(values, indices, counts);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def("_group_partition(Tensor labels) -> (Tensor, Tensor)");
  m.def(
      "group_projection_apply(Tensor values, Tensor indices, Tensor counts) -> "
      "Tensor");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_group_partition", TORCH_FN(group_partition_impl));
}
TORCH_LIBRARY_IMPL(torchpairwise, Meta, m) {
  m.impl("_group_partition", TORCH_FN(group_partition_meta));
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeImplicitAutograd, m) {
  m.impl("group_projection_apply", TORCH_FN(group_projection_apply_impl));
}
}  // namespace torchpairwise::ops
