#pragma once
#include <ATen/ATen.h>

#include <limits>

namespace torchpairwise::ops::detail::residual {
// State shared by Cartesian, paired and multi-axis residual consumers.
inline bool extra_reduction(c10::string_view name) {
  return name == "min" || name == "max" || name == "prod" || name == "all" ||
         name == "any";
}
inline bool boolean_reduction(c10::string_view name) {
  return name == "all" || name == "any";
}
inline int64_t reducer_state_width(c10::string_view name) {
  return name == "prod" ? 5 : boolean_reduction(name) ? 1 : 2;
}
inline at::Tensor reducer_state(at::IntArrayRef shape,
                                at::TensorOptions options,
                                c10::string_view name) {
  auto sizes = shape.vec();
  sizes.push_back(reducer_state_width(name));
  auto state = at::zeros(sizes, options.dtype(at::kDouble));
  if (name == "min" || name == "max") {
    state.select(-1, 0).fill_(name == "min"
                                  ? std::numeric_limits<double>::infinity()
                                  : -std::numeric_limits<double>::infinity());
    state.select(-1, 1).fill_(std::numeric_limits<double>::infinity());
  } else if (name == "prod")
    state.select(-1, 1).fill_(1.);
  else if (name == "all")
    state.select(-1, 0).fill_(1.);
  return state;
}
inline at::Tensor axis_product(at::Tensor value, at::IntArrayRef axes) {
  for (auto axis : axes)
    value = value.prod(axis, true);
  return value;
}
inline void reducer_update(at::Tensor state, const at::Tensor& value,
                           const at::Tensor& ids, at::IntArrayRef axes,
                           c10::string_view name) {
  if (boolean_reduction(name)) {
    auto truth = value.ne(0);
    for (auto axis : axes)
      truth = name == "all" ? truth.all(axis, true) : truth.any(axis, true);
    auto total = state.select(-1, 0);
    total.copy_(name == "all" ? total.ne(0) & truth : total.ne(0) | truth);
  } else if (name == "prod") {
    // Separate exceptional factors so a zero never destroys the product needed
    // by its own derivative. Infinities/NaNs are counted independently too.
    auto zero = value.eq(0), inf = value.isinf(), nan = value.isnan();
    state.select(-1, 0).add_(zero.sum(axes, true));
    state.select(-1, 1).mul_(
        axis_product(at::where(zero | inf | nan, 1., value), axes));
    state.select(-1, 2).add_(value.lt(0).sum(axes, true));
    state.select(-1, 3).add_(inf.sum(axes, true));
    state.select(-1, 4).add_(nan.sum(axes, true));
  } else {
    auto best = name == "min" ? value.amin(axes, true) : value.amax(axes, true);
    auto equal = value.eq(best) | (value.isnan() & best.isnan());
    auto witness = at::where(equal, ids.to(at::kDouble),
                             std::numeric_limits<double>::infinity())
                       .amin(axes, true);
    auto old = state.select(-1, 0), old_id = state.select(-1, 1);
    auto better = best.isnan() & ~old.isnan();
    better |= ~best.isnan() & ~old.isnan() &
              (name == "min" ? best.lt(old) : best.gt(old));
    better |=
        (best.eq(old) | (best.isnan() & old.isnan())) & witness.lt(old_id);
    old.copy_(at::where(better, best, old));
    old_id.copy_(at::where(better, witness, old_id));
  }
}
inline at::Tensor product_value(const at::Tensor& zeros,
                                const at::Tensor& finite,
                                const at::Tensor& negatives,
                                const at::Tensor& infs,
                                const at::Tensor& nans) {
  auto infinity = at::where(negatives.remainder(2).ne(0),
                            -std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity());
  auto result = at::where(infs.gt(0), infinity, finite);
  result = at::where(zeros.gt(0), at::zeros_like(result), result);
  return at::where(nans.gt(0) | (zeros.gt(0) & infs.gt(0)),
                   std::numeric_limits<double>::quiet_NaN(), result);
}
inline at::Tensor reducer_result(const at::Tensor& state,
                                 c10::string_view name) {
  if (name == "prod")
    return product_value(state.select(-1, 0), state.select(-1, 1),
                         state.select(-1, 2), state.select(-1, 3),
                         state.select(-1, 4));
  auto result = state.select(-1, 0);
  return boolean_reduction(name) ? result.ne(0) : result;
}
inline at::Tensor reducer_gradient(const at::Tensor& state,
                                   const at::Tensor& value,
                                   const at::Tensor& ids,
                                   const at::Tensor& grad,
                                   c10::string_view name) {
  if (name == "min" || name == "max")
    return at::where(ids.eq(state.select(-1, 1)), grad, 0.);
  auto zero = value.eq(0), inf = value.isinf(), nan = value.isnan();
  auto finite = state.select(-1, 1) / at::where(zero | inf | nan, 1., value);
  auto factor =
      product_value(state.select(-1, 0) - zero.to(at::kDouble), finite,
                    state.select(-1, 2) - value.lt(0).to(at::kDouble),
                    state.select(-1, 3) - inf.to(at::kDouble),
                    state.select(-1, 4) - nan.to(at::kDouble));
  return grad * factor;
}
}  // namespace torchpairwise::ops::detail::residual
