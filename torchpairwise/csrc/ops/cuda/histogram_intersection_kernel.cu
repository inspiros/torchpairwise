#include <ATen/Dispatch.h>
#include <torch/library.h>

#include "../histogram_intersection.h"
#include "cuda_helpers.h"

namespace torchpairwise::ops {
namespace {
template <typename scalar_t>
__global__ void histogram_intersection_kernel_impl(int64_t count,
                                                   const scalar_t* x,
                                                   const scalar_t* y,
                                                   scalar_t* out, int64_t n,
                                                   int64_t m, int64_t d) {
  CUDA_1D_KERNEL_LOOP(t, count) {
    out[t] = histogram_value(x, y, n, m, d, t);
  }
}
template <bool left, typename scalar_t>
__global__ void histogram_intersection_backward_kernel_impl(
    int64_t count, const scalar_t* g, const scalar_t* x, const scalar_t* y,
    scalar_t* out, int64_t n, int64_t m, int64_t d) {
  CUDA_1D_KERNEL_LOOP(t, count) {
    out[t] = histogram_derivative<left>(g, x, y, n, m, d, t);
  }
}

at::Tensor histogram_intersection_kernel(const at::Tensor& x,
                                         const at::Tensor& y) {
  check_histogram_inputs(x, y);
  const at::cuda::CUDAGuard guard(x.device());
  auto a = x.contiguous(), b = y.contiguous();
  auto shape = x.sizes().vec();
  shape.back() = y.size(-2);
  auto out = at::empty(shape, x.options());
  if (!out.numel())
    return out;
  AT_DISPATCH_FLOATING_TYPES(
      x.scalar_type(), "histogram_intersection_cuda", [&] {
        histogram_intersection_kernel_impl<<<
            GET_BLOCKS(256, out.numel()), 256, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            out.numel(), a.const_data_ptr<scalar_t>(),
            b.const_data_ptr<scalar_t>(), out.mutable_data_ptr<scalar_t>(),
            x.size(-2), y.size(-2), x.size(-1));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::tuple<at::Tensor, at::Tensor> histogram_intersection_backward_kernel(
    const at::Tensor& grad, const at::Tensor& x, const at::Tensor& y) {
  check_histogram_grad(grad, x, y);
  const at::cuda::CUDAGuard guard(x.device());
  auto a = x.contiguous(), b = y.contiguous(), g = grad.contiguous();
  auto gx = at::empty(x.sizes(), x.options()),
       gy = at::empty(y.sizes(), y.options());
  AT_DISPATCH_FLOATING_TYPES(
      x.scalar_type(), "histogram_intersection_backward_cuda", [&] {
        if (gx.numel())
          histogram_intersection_backward_kernel_impl<true>
              <<<GET_BLOCKS(256, gx.numel()), 256, 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  gx.numel(), g.const_data_ptr<scalar_t>(),
                  a.const_data_ptr<scalar_t>(), b.const_data_ptr<scalar_t>(),
                  gx.mutable_data_ptr<scalar_t>(), x.size(-2), y.size(-2),
                  x.size(-1));
        if (gy.numel())
          histogram_intersection_backward_kernel_impl<false>
              <<<GET_BLOCKS(256, gy.numel()), 256, 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  gy.numel(), g.const_data_ptr<scalar_t>(),
                  a.const_data_ptr<scalar_t>(), b.const_data_ptr<scalar_t>(),
                  gy.mutable_data_ptr<scalar_t>(), x.size(-2), y.size(-2),
                  x.size(-1));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gx, gy};
}
}  // namespace
TORCH_LIBRARY_IMPL(torchpairwise, CUDA, m) {
  m.impl("_histogram_intersection", TORCH_FN(histogram_intersection_kernel));
  m.impl("_histogram_intersection_backward",
         TORCH_FN(histogram_intersection_backward_kernel));
}
}  // namespace torchpairwise::ops
