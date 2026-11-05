#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../histogram_intersection.h"

namespace torchpairwise::ops {
namespace {
at::Tensor histogram_intersection_kernel(const at::Tensor& x,
                                         const at::Tensor& y) {
  check_histogram_inputs(x, y);
  auto a = x.contiguous(), b = y.contiguous();
  const auto n = x.size(-2), m = y.size(-2), d = x.size(-1);
  auto shape = x.sizes().vec();
  shape.back() = m;
  auto out = at::empty(shape, x.options());
  AT_DISPATCH_FLOATING_TYPES(
      x.scalar_type(), "histogram_intersection_cpu", [&] {
        const auto *xp = a.const_data_ptr<scalar_t>(),
                   *yp = b.const_data_ptr<scalar_t>();
        auto* op = out.mutable_data_ptr<scalar_t>();
        at::parallel_for(0, out.numel(), 64, [&](int64_t start, int64_t end) {
          for (auto t = start; t < end; ++t)
            op[t] = histogram_value(xp, yp, n, m, d, t);
        });
      });
  return out;
}

std::tuple<at::Tensor, at::Tensor> histogram_intersection_backward_kernel(
    const at::Tensor& grad, const at::Tensor& x, const at::Tensor& y) {
  check_histogram_grad(grad, x, y);
  auto a = x.contiguous(), b = y.contiguous(), g = grad.contiguous();
  auto gx = at::empty(x.sizes(), x.options()),
       gy = at::empty(y.sizes(), y.options());
  const auto n = x.size(-2), m = y.size(-2), d = x.size(-1);
  AT_DISPATCH_FLOATING_TYPES(
      x.scalar_type(), "histogram_intersection_backward_cpu", [&] {
        const auto *xp = a.const_data_ptr<scalar_t>(),
                   *yp = b.const_data_ptr<scalar_t>();
        const auto* gp = g.const_data_ptr<scalar_t>();
        auto *dx = gx.mutable_data_ptr<scalar_t>(),
             *dy = gy.mutable_data_ptr<scalar_t>();
        at::parallel_for(0, gx.numel(), 64, [&](int64_t start, int64_t end) {
          for (auto t = start; t < end; ++t)
            dx[t] = histogram_derivative<true>(gp, xp, yp, n, m, d, t);
        });
        at::parallel_for(0, gy.numel(), 64, [&](int64_t start, int64_t end) {
          for (auto t = start; t < end; ++t)
            dy[t] = histogram_derivative<false>(gp, xp, yp, n, m, d, t);
        });
      });
  return {gx, gy};
}
}  // namespace

TORCH_LIBRARY_IMPL(torchpairwise, CPU, m) {
  m.impl("_histogram_intersection", TORCH_FN(histogram_intersection_kernel));
  m.impl("_histogram_intersection_backward",
         TORCH_FN(histogram_intersection_backward_kernel));
}
}  // namespace torchpairwise::ops
