#pragma once
// Internal native bridges shared by forward/backward kernels and autograd.
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <array>
#include <vector>
namespace torchpairwise::ops::detail::audio {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
using Core = Tensor(const Tensor&, const Tensor&, int64_t, int64_t, int64_t,
                    at::IntArrayRef, at::ArrayRef<double>);
inline Tensor _audio_compare(const Tensor& x, const Tensor& y, int64_t kind,
                             int64_t mode, int64_t size,
                             at::IntArrayRef settings, at::ArrayRef<double> p) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_audio_compare", "")
                       .typed<Core>();
  return op.call(x, y, kind, mode, size, settings, p);
}

}  // namespace torchpairwise::ops::detail::audio
