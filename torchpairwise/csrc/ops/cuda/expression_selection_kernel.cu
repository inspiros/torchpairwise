#include <ATen/Dispatch.h>

#include "../expression_selection_detail.h"
#include "pair_reduction.cuh"

namespace torchpairwise::ops::detail::streaming {
namespace {
template <typename scalar_t>
struct SelectionDifferenceTerm {
  // Retain the public-dtype pair product, then accumulate in float64 like the
  // common streaming buffers. Do not factor x*sum(g)-sum(g*y): it loses the
  // zero-cotangent NaNs of 0*(inf-inf), 0*inf and overflowing differences.
  using acc_t = double;
  using math_t = at::opmath_type<scalar_t>;
  cuda_detail::Accessor3<scalar_t> own, other;
  cuda_detail::Accessor2<scalar_t> gradient;
  cuda_detail::Accessor3<int64_t> indices;
  bool transpose;
  math_t weight;
  __device__ acc_t operator()(int64_t b, int64_t r, int64_t o,
                              int64_t e) const {
    auto row = transpose ? o : r, column = transpose ? r : o;
    auto lo = indices[b][row][0], hi = indices[b][row][1];
    math_t coefficient = (column == lo ? math_t(1) - weight : math_t(0)) +
                         (column == hi ? weight : math_t(0));
    math_t g = coefficient != math_t(0) ? math_t(gradient[b][row]) * coefficient
                                        : math_t(0);
    math_t delta = math_t(own[b][r][e]) - math_t(other[b][o][e]);
    return acc_t(g * delta);
  }
};
}  // namespace
std::tuple<at::Tensor, at::Tensor> selection_sqeuclidean_backward_cuda(
    const at::Tensor& x, const at::Tensor& y, const at::Tensor& gradient,
    const at::Tensor& indices, double weight) {
  at::cuda::CUDAGuard device_guard(x.device());
  auto a = x.unsqueeze(0), b = y.unsqueeze(0), g = gradient.unsqueeze(0),
       ids = indices.unsqueeze(0);
  at::Tensor dx, dy;
  AT_DISPATCH_FLOATING_TYPES(
      x.scalar_type(), "selection_sqeuclidean_backward_cuda", [&] {
        auto accessor = [](const at::Tensor& t) {
          return t.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits,
                                           int64_t>();
        };
        auto reduce = [&](const at::Tensor& own, const at::Tensor& other,
                          bool transpose) {
          SelectionDifferenceTerm<scalar_t> term{
              accessor(own),
              accessor(other),
              g.generic_packed_accessor<scalar_t, 2, at::RestrictPtrTraits,
                                        int64_t>(),
              ids.generic_packed_accessor<int64_t, 3, at::RestrictPtrTraits,
                                          int64_t>(),
              transpose,
              static_cast<at::opmath_type<scalar_t>>(weight)};
          return cuda_detail::pair_reduction(term, 1, own.size(1), own.size(2),
                                             other.size(1), own.options())
              .squeeze(0)
              .mul_(2.);
        };
        dx = reduce(a, b, false);
        dy = reduce(b, a, true);
      });
  return {dx, dy};
}
}  // namespace torchpairwise::ops::detail::streaming
