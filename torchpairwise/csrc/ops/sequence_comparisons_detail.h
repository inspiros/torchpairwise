#pragma once
// Internal native bridges shared by forward/backward kernels and autograd.
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <array>
#include <vector>
namespace torchpairwise::ops::detail::sequence {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
using OptionalDouble = c10::optional<double>;
using OptionalInt = c10::optional<int64_t>;
using CoreSignature = Tensor(const Tensor&, const Tensor&, const Tensor&,
                             const Tensor&, int64_t, bool, OptionalDouble,
                             OptionalInt, int64_t);
inline Tensor _sequence_compare(const Tensor& x, const Tensor& y,
                                const Tensor& lx, const Tensor& ly,
                                int64_t mode, bool squared,
                                OptionalDouble gamma, OptionalInt window,
                                int64_t chunk) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_sequence_compare", "")
          .typed<CoreSignature>();
  return op.call(x, y, lx, ly, mode, squared, gamma, window, chunk);
}

}  // namespace torchpairwise::ops::detail::sequence

namespace torchpairwise::ops::detail::sequence {
std::array<Tensor, 4> prepare(const Tensor& input,
                              const c10::optional<Tensor>& other,
                              c10::string_view mode, bool squared,
                              OptionalDouble gamma,
                              const c10::optional<Tensor>& x1_lengths,
                              const c10::optional<Tensor>& x2_lengths,
                              OptionalInt window, int64_t chunk);
using CostSignature = Tensor(const Tensor&, const Tensor&, const Tensor&,
                             const Tensor&, OptionalDouble, OptionalInt,
                             int64_t, bool);
using CostGradSignature = Pair(const Tensor&, const Tensor&, const Tensor&,
                               const Tensor&, const Tensor&, OptionalDouble,
                               OptionalInt, int64_t, bool);
inline Tensor operator_cost(const Tensor& x, const Tensor& y, const Tensor& lx,
                            const Tensor& ly, OptionalDouble gamma,
                            OptionalInt window, int64_t chunk,
                            bool self = false) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_sequence_operator_cost", "")
          .typed<CostSignature>();
  return op.call(x, y, lx, ly, gamma, window, chunk, self);
}
Tensor prepared_pair(const Tensor& x, const Tensor& y, const Tensor& lx,
                     const Tensor& ly, const Tensor& sx, const Tensor& sy,
                     bool paired, bool squared, OptionalDouble gamma,
                     OptionalInt window, int64_t chunk);
}  // namespace torchpairwise::ops::detail::sequence
