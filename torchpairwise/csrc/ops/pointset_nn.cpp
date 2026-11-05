#include "pointset_nn.h"

#include <torch/library.h>
namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Pair = std::tuple<Tensor, Tensor>;
Pair pointset_nn_backward_kernel(const Tensor& grad, const Tensor& x,
                                 const Tensor& y, const Tensor& lengths,
                                 const Tensor& indices) {
  auto gx = at::zeros(x.sizes(), x.options()),
       gy = at::zeros(y.sizes(), y.options());
  if (grad.numel() == 0)
    return {gx, gy};
  const auto B = x.size(0), N = x.size(1), P = x.size(2), C = x.size(3);
  const auto M = y.size(1), Q = y.size(2);
  // Gather only selected witnesses, in source-point tiles. There is no
  // P*Q matrix or full N*M*P*C gradient temporary.
  auto integer = x.options().dtype(at::kLong);
  auto offset = at::arange(B, integer).view({B, 1, 1, 1}) * (M * Q) +
                at::arange(M, integer).view({1, 1, M, 1}) * Q;
  for (int64_t start = 0; start < P; start += 128) {
    auto count = std::min(int64_t(128), P - start);
    auto index = indices.narrow(-1, start, count);
    auto expanded = index.unsqueeze(-1).expand({B, N, M, count, C});
    auto target = y.unsqueeze(1).expand({B, N, M, Q, C}).gather(3, expanded);
    auto valid = at::arange(start, start + count, integer)
                     .view({1, 1, count})
                     .lt(lengths.unsqueeze(-1));
    auto difference = x.narrow(2, start, count).unsqueeze(2) - target;
    difference = at::where(valid.unsqueeze(2).unsqueeze(-1), difference, 0.);
    auto contribution =
        difference * (2 * grad.narrow(-1, start, count).unsqueeze(-1));
    gx.narrow(2, start, count).add_(contribution.sum(2));
    gy.view({B * M * Q, C})
        .index_add_(0, (index + offset).reshape({-1}),
                    -contribution.reshape({-1, C}));
  }
  return {gx, gy};
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_pointset_nn_backward(Tensor grad, Tensor x, Tensor y, Tensor "
      "x_lengths, Tensor indices) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_pointset_nn_backward", TORCH_FN(pointset_nn_backward_kernel));
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_pointset_nn(Tensor x, Tensor y, Tensor x_lengths, Tensor y_lengths) -> "
      "(Tensor distances, Tensor indices)");
}
}  // namespace torchpairwise::ops
