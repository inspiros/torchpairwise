#pragma once
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <array>
namespace torchpairwise::ops::detail::tokens {
using at::Tensor;
std::array<Tensor, 4> prepare(const Tensor& x,
                              const c10::optional<Tensor>& other,
                              c10::string_view mode, int64_t substitution,
                              const c10::optional<Tensor>& x1_lengths,
                              const c10::optional<Tensor>& x2_lengths,
                              int64_t chunk);
using Signature = Tensor(const Tensor&, const Tensor&, const Tensor&,
                         const Tensor&, int64_t, int64_t, int64_t);
inline Tensor compare(const Tensor& x, const Tensor& y, const Tensor& xl,
                      const Tensor& yl, int64_t mode, int64_t substitution,
                      int64_t chunk) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_token_compare", "")
                       .typed<Signature>();
  return op.call(x, y, xl, yl, mode, substitution, chunk);
}
}  // namespace torchpairwise::ops::detail::tokens
