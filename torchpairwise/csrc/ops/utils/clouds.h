#pragma once
#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>

#include "value_checks.h"

namespace torchpairwise::ops::detail {
struct CloudInputs {
  at::Tensor x, y, xl, yl;
  std::vector<c10::SymInt> batch;
};
inline at::Tensor cloud_layout(const at::Tensor& x, c10::optional<bool> batched,
                               int64_t feature_dim) {
  TORCH_CHECK(x.layout() == at::kStrided,
              "Point-set inputs must be dense strided tensors");
  TORCH_CHECK(x.dim() >= (batched.value_or(false) ? 4 : 3),
              "Point-set inputs have insufficient rank");
  TORCH_CHECK(feature_dim >= -x.dim() && feature_dim < x.dim(),
              "feature_dim is out of range");
  auto axis = (feature_dim + x.dim()) % x.dim(),
       sample = batched.has_value() ? int64_t(*batched) : x.dim() - 3;
  TORCH_CHECK(axis > sample, "feature_dim must follow batch and sample axes");
  auto value = x.movedim(axis, -1);
  auto shape = value.sym_sizes().slice(0, sample + 1).vec();
  c10::SymInt points = 1;
  for (int64_t i = sample + 1; i < value.dim() - 1; ++i)
    points *= value.sym_size(i);
  shape.push_back(points);
  shape.push_back(value.sym_size(-1));
  return value.reshape_symint(shape);
}
inline CloudInputs cloud_inputs(const at::Tensor& input,
                                const c10::optional<at::Tensor>& other,
                                const c10::optional<at::Tensor>& x_lengths,
                                const c10::optional<at::Tensor>& y_lengths,
                                c10::optional<bool> batched,
                                int64_t feature_dim, int64_t mode,
                                bool defer_length_checks = false) {
  auto x = cloud_layout(input, batched, feature_dim),
       y = other.has_value() ? cloud_layout(*other, batched, feature_dim) : x;
  TORCH_CHECK(x.device() == y.device() && x.scalar_type() == y.scalar_type(),
              "Inputs must share dtype and device");
  TORCH_CHECK(x.sym_size(-1) > 0 && x.sym_size(-1) == y.sym_size(-1),
              "Input features must match and be non-empty");
  TORCH_CHECK(x.sym_size(-2) > 0 && y.sym_size(-2) > 0,
              "Input point sets must be non-empty");
  TORCH_CHECK(mode != 1 || x.sym_size(-3) == y.sym_size(-3),
              "Paired inputs must have equal sample counts");
  auto xl = x_lengths, yl = y_lengths;
  if (!other.has_value() && !yl.has_value())
    yl = xl;
  auto batch = at::infer_size_symint(x.sym_sizes().slice(0, x.dim() - 3),
                                     y.sym_sizes().slice(0, y.dim() - 3));
  auto rank = std::max(x.dim(), y.dim()) - 3;
  for (auto item : {std::make_pair(x, xl), std::make_pair(y, yl)})
    if (item.second.has_value()) {
      const auto& len = *item.second;
      TORCH_CHECK(len.layout() == at::kStrided &&
                      len.scalar_type() == at::kLong &&
                      len.device() == x.device(),
                  "Lengths must be dense int64 tensors on the input device");
      TORCH_CHECK(len.dim() >= 1 && len.sym_size(-1) == item.first.sym_size(-3),
                  "Lengths must end with sample count");
      TORCH_CHECK(len.dim() - 1 <= rank,
                  "Lengths cannot introduce additional batch axes");
      if (!defer_length_checks)
        utils::check_values(
            (len.ge(1) & len.le(item.first.sym_size(-2))).all(),
            "Lengths must be between 1 and the padded point count");
      batch =
          at::infer_size_symint(batch, len.sym_sizes().slice(0, len.dim() - 1));
    }
  auto normalize = [&](const at::Tensor& value) {
    auto shape = batch;
    for (int64_t i = -3; i < 0; ++i)
      shape.push_back(value.sym_size(i));
    return value.expand_symint(shape);
  };
  auto normalize_lengths = [&](const c10::optional<at::Tensor>& len,
                               const at::Tensor& value) {
    auto shape = batch;
    shape.push_back(value.sym_size(-3));
    return len.has_value() ? len->expand_symint(shape)
                           : at::empty({0}, value.options().dtype(at::kLong));
  };
  return {normalize(x), normalize(y), normalize_lengths(xl, x),
          normalize_lengths(yl, y), batch};
}
inline std::vector<c10::SymInt> cloud_output_shape(const at::Tensor& x,
                                                   const at::Tensor& y,
                                                   int64_t mode) {
  auto shape = x.sym_sizes().slice(0, x.dim() - 3).vec();
  auto n = x.sym_size(-3);
  shape.push_back(mode == 2 ? n * (n - 1) / 2 : n);
  if (mode == 0)
    shape.push_back(y.sym_size(-3));
  return shape;
}
inline at::Tensor safe_root(const at::Tensor& x) {
  auto nz = x.ne(0);
  return at::where(nz, at::where(nz, x, 1.).sqrt(), 0.);
}
}  // namespace torchpairwise::ops::detail
