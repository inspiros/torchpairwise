#include <ATen/Dispatch.h>

#include "../streaming_reductions_detail.h"
#include "cuda_helpers.h"

namespace torchpairwise::ops::detail::streaming {
namespace {
template <typename scalar_t>
__global__ void compensated_kernel(int64_t count, const scalar_t* total,
                                   const scalar_t* correction,
                                   const scalar_t* term, scalar_t* output,
                                   scalar_t* error) {
  CUDA_1D_KERNEL_LOOP(i, count) {
    auto a = total[i], b = term[i], updated = a + b;
    bool valid = ::isfinite(a) && ::isfinite(b) && ::isfinite(updated);
    auto residual =
        ::fabs(a) >= ::fabs(b) ? (a - updated) + b : (b - updated) + a;
    output[i] = updated;
    error[i] = valid ? correction[i] + residual : scalar_t(0);
  }
}
}  // namespace
std::pair<Tensor, Tensor> compensated_cuda(const Tensor& total,
                                           const Tensor& correction,
                                           const Tensor& term) {
  const at::cuda::CUDAGuard guard(total.device());
  auto output = at::empty(total.sizes(), total.options());
  auto error = at::empty(total.sizes(), total.options());
  if (total.numel()) {
    AT_DISPATCH_FLOATING_TYPES(
        total.scalar_type(), "compensated_sum_cuda", [&] {
          compensated_kernel<<<GET_BLOCKS(256, total.numel()), 256, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
              total.numel(), total.const_data_ptr<scalar_t>(),
              correction.const_data_ptr<scalar_t>(),
              term.const_data_ptr<scalar_t>(),
              output.mutable_data_ptr<scalar_t>(),
              error.mutable_data_ptr<scalar_t>());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output, error};
}
}  // namespace torchpairwise::ops::detail::streaming
