#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../nan_euclidean.h"

namespace torchpairwise::ops {
namespace {
std::tuple<at::Tensor, at::Tensor> nan_sqeuclidean_kernel(const at::Tensor& x,
                                                          const at::Tensor& y) {
  check_nan_inputs(x, y);
  auto a = x.contiguous(), b = y.contiguous();
  const auto n = x.size(-2), m = y.size(-2), d = x.size(-1);
  auto shape = x.sizes().vec();
  shape.back() = m;
  auto out = at::empty(shape, x.options()),
       counts = at::empty(shape, x.options().dtype(at::kLong));
  auto* cp = counts.mutable_data_ptr<int64_t>();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "nan_sqeuclidean_cpu", [&] {
    const auto *xp = a.const_data_ptr<scalar_t>(),
               *yp = b.const_data_ptr<scalar_t>();
    auto* op = out.mutable_data_ptr<scalar_t>();
    at::parallel_for(0, out.numel(), 64, [&](int64_t start, int64_t end) {
      for (auto t = start; t < end; ++t)
        nan_squared_value(xp, yp, n, m, d, t, op[t], cp[t]);
    });
  });
  return {out, counts};
}

std::tuple<at::Tensor, at::Tensor> nan_sqeuclidean_backward_kernel(
    const at::Tensor& grad, const at::Tensor& x, const at::Tensor& y,
    const at::Tensor& counts) {
  check_nan_backward(grad, x, y, counts);
  auto a = x.contiguous(), b = y.contiguous(), g = grad.contiguous(),
       c = counts.contiguous();
  auto gx = at::empty(x.sizes(), x.options()),
       gy = at::empty(y.sizes(), y.options());
  const auto n = x.size(-2), m = y.size(-2), d = x.size(-1);
  const auto* cp = c.const_data_ptr<int64_t>();
  AT_DISPATCH_FLOATING_TYPES(
      x.scalar_type(), "nan_sqeuclidean_backward_cpu", [&] {
        const auto *xp = a.const_data_ptr<scalar_t>(),
                   *yp = b.const_data_ptr<scalar_t>(),
                   *gp = g.const_data_ptr<scalar_t>();
        auto *dx = gx.mutable_data_ptr<scalar_t>(),
             *dy = gy.mutable_data_ptr<scalar_t>();
        at::parallel_for(0, gx.numel(), 64, [&](int64_t start, int64_t end) {
          for (auto t = start; t < end; ++t)
            dx[t] = nan_squared_derivative<true>(gp, xp, yp, cp, n, m, d, t);
        });
        at::parallel_for(0, gy.numel(), 64, [&](int64_t start, int64_t end) {
          for (auto t = start; t < end; ++t)
            dy[t] = nan_squared_derivative<false>(gp, xp, yp, cp, n, m, d, t);
        });
      });
  return {gx, gy};
}
}  // namespace

TORCH_LIBRARY_IMPL(torchpairwise, CPU, m) {
  m.impl("_nan_sqeuclidean", TORCH_FN(nan_sqeuclidean_kernel));
  m.impl("_nan_sqeuclidean_backward",
         TORCH_FN(nan_sqeuclidean_backward_kernel));
}
}  // namespace torchpairwise::ops
