#pragma once
// Internal native bridges shared by forward/backward kernels and autograd.
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <array>
#include <vector>
namespace torchpairwise::ops::detail::builtin {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
using Core = Tensor(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                    c10::string_view, int64_t, bool, double, double, int64_t,
                    double, double, bool);
inline Tensor _legacy_compare(const Tensor& x, const Tensor& y, const Tensor& v,
                              const Tensor& vi, c10::string_view name,
                              int64_t mode, bool kernel, double p, double base,
                              int64_t degree, double gamma, double coef0,
                              bool self) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_legacy_compare", "")
                       .typed<Core>();
  return op.call(x, y, v, vi, name, mode, kernel, p, base, degree, gamma, coef0,
                 self);
}
bool contains_name(c10::string_view name,
                   std::initializer_list<const char*> names);
}  // namespace torchpairwise::ops::detail::builtin
