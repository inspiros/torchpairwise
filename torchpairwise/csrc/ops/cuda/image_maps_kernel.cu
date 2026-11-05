#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>

#include "../image_map_operations_detail.h"
#include "cuda_helpers.h"

namespace torchpairwise::ops::detail::media {
namespace {
using Features =
    at::GenericPackedTensorAccessor<double, 5, at::RestrictPtrTraits, int64_t>;
template <typename scalar_t>
using Values = at::GenericPackedTensorAccessor<scalar_t, 5,
                                               at::RestrictPtrTraits, int64_t>;
template <typename scalar_t>
__global__ void pixel_map_region_kernel(int64_t count, Features x, Features y,
                                        Values<scalar_t> output, int comparison,
                                        bool round_to_float) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    auto w = index % output.size(4), t = index / output.size(4);
    auto h = t % output.size(3);
    t /= output.size(3);
    auto c = t % output.size(2);
    t /= output.size(2);
    auto j = t % output.size(1), i = t / output.size(1);
    double value = y[i][j][c][h][w] - x[i][j][c][h][w];
    if (comparison == 1)
      value = ::fabs(value);
    else if (comparison == 2)
      value *= value;
    // Keep the public boundary even for a double output accumulator. This
    // explicit round-to-nearest conversion preserves float32 subnormal entries.
    if (round_to_float)
      value = static_cast<double>(__double2float_rn(value));
    output[i][j][c][h][w] = static_cast<scalar_t>(value);
  }
}
}  // namespace
Tensor pixel_map_region_cuda(const Tensor& x, const Tensor& y,
                             int64_t comparison, int64_t dtype, bool promote) {
  const at::cuda::CUDAGuard guard(x.device());
  TORCH_INTERNAL_ASSERT(
      x.scalar_type() == at::kDouble && y.scalar_type() == at::kDouble &&
      x.dim() == y.dim() && (x.dim() == 4 || x.dim() == 5) && comparison >= 0 &&
      comparison <= 2 &&
      (dtype == int64_t(at::kFloat) || dtype == int64_t(at::kDouble)));
  auto shape = at::infer_size(x.sizes(), y.sizes());
  auto output = at::empty(
      shape, x.options().dtype(promote ? at::kDouble : at::ScalarType(dtype)));
  if (!output.numel())
    return output;
  // Broadcast through strides, never expanded storage. Paired regions add one
  // singleton sample axis so both modes use the same fixed-rank kernel.
  auto left = x.expand(shape), right = y.expand(shape);
  auto values = output;
  if (x.dim() == 4) {
    left = left.unsqueeze(0);
    right = right.unsqueeze(0);
    values = values.unsqueeze(0);
  }
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "pixel_map_region", [&] {
    pixel_map_region_kernel<<<GET_BLOCKS(256, output.numel()), 256, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
        output.numel(),
        left.packed_accessor64<double, 5, at::RestrictPtrTraits>(),
        right.packed_accessor64<double, 5, at::RestrictPtrTraits>(),
        values.packed_accessor64<scalar_t, 5, at::RestrictPtrTraits>(),
        int(comparison), dtype == int64_t(at::kFloat));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return output;
}
}  // namespace torchpairwise::ops::detail::media
