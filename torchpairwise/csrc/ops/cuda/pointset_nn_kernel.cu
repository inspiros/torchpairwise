#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include "../pointset_nn.h"
#include "cuda_helpers.h"

namespace torchpairwise::ops {
namespace {
template <typename scalar_t>
__global__ void pointset_nn_cuda_impl(int64_t total, const scalar_t* x,
                                      const scalar_t* y, const int64_t* xl,
                                      const int64_t* yl, scalar_t* out,
                                      int64_t* indices, int64_t N, int64_t M,
                                      int64_t P, int64_t Q, int64_t C) {
  CUDA_1D_KERNEL_LOOP(t, total) {
    const auto p = t % P, j = (t / P) % M;
    const auto i = (t / (P * M)) % N, b = t / (P * M * N);
    out[t] = 0;
    indices[t] = 0;
    if (p >= xl[b * N + i])
      continue;
    scalar_t best = INFINITY;
    int64_t nearest = 0;
    const auto raw_length = yl[b * M + j];
    const auto length = raw_length < 0 ? 0 : (raw_length > Q ? Q : raw_length);
    for (int64_t q = 0; q < length; ++q) {
      scalar_t squared = 0;
      for (int64_t c = 0; c < C; ++c) {
        const scalar_t diff =
            x[((b * N + i) * P + p) * C + c] - y[((b * M + j) * Q + q) * C + c];
        squared += diff * diff;
      }
      if (squared < best) {
        best = squared;
        nearest = q;
      }
    }
    out[t] = best;
    indices[t] = nearest;
  }
}

// One warp searches the target cloud for one source point. Lanes visit
// independent target points, then reduce (distance,index) with first-index
// ties. No point-by-point matrix or global reduction scratch is needed.
template <typename scalar_t>
__global__ void pointset_nn_warp_cuda_impl(int64_t total, const scalar_t* x,
                                           const scalar_t* y, const int64_t* xl,
                                           const int64_t* yl, scalar_t* out,
                                           int64_t* indices, int64_t N,
                                           int64_t M, int64_t P, int64_t Q,
                                           int64_t C) {
  constexpr int warps_per_block = 8;
  const int lane = threadIdx.x % 32;
  for (int64_t t = static_cast<int64_t>(blockIdx.x) * warps_per_block +
                   threadIdx.x / 32;
       t < total; t += static_cast<int64_t>(gridDim.x) * warps_per_block) {
    const auto p = t % P, j = (t / P) % M;
    const auto i = (t / (P * M)) % N, b = t / (P * M * N);
    if (p >= xl[b * N + i]) {
      if (lane == 0) {
        out[t] = 0;
        indices[t] = 0;
      }
      continue;
    }
    scalar_t best = INFINITY;
    int64_t nearest = 0;
    const auto raw_length = yl[b * M + j];
    const auto length = raw_length < 0 ? 0 : (raw_length > Q ? Q : raw_length);
    for (int64_t q = lane; q < length; q += 32) {
      scalar_t squared = 0;
      for (int64_t c = 0; c < C; ++c) {
        const scalar_t diff =
            x[((b * N + i) * P + p) * C + c] - y[((b * M + j) * Q + q) * C + c];
        squared += diff * diff;
      }
      if (squared < best) {
        best = squared;
        nearest = q;
      }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      const auto other = __shfl_down_sync(0xffffffff, best, offset);
      const auto other_index = __shfl_down_sync(0xffffffff, nearest, offset);
      if (other < best || (other == best && other_index < nearest)) {
        best = other;
        nearest = other_index;
      }
    }
    if (lane == 0) {
      out[t] = best;
      indices[t] = nearest;
    }
  }
}

std::tuple<at::Tensor, at::Tensor> pointset_nn_kernel(const at::Tensor& x,
                                                      const at::Tensor& y,
                                                      const at::Tensor& xl,
                                                      const at::Tensor& yl) {
  check_pointset_nn(x, y, xl, yl);
  const c10::cuda::CUDAGuard guard(x.device());
  auto xc = x.contiguous(), yc = y.contiguous();
  auto xlc = xl.contiguous(), ylc = yl.contiguous();
  auto distances =
      at::empty({x.size(0), x.size(1), y.size(1), x.size(2)}, x.options());
  auto indices = at::empty(distances.sizes(), x.options().dtype(at::kLong));
  if (distances.numel() != 0) {
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointset_nn_kernel", [&] {
      const auto kernel = y.size(2) >= 32 ? pointset_nn_warp_cuda_impl<scalar_t>
                                          : pointset_nn_cuda_impl<scalar_t>;
      const auto work_per_block = y.size(2) >= 32 ? 8 : 256;
      kernel<<<GET_BLOCKS(work_per_block, distances.numel()), 256, 0,
               at::cuda::getCurrentCUDAStream()>>>(
          distances.numel(), xc.const_data_ptr<scalar_t>(),
          yc.const_data_ptr<scalar_t>(), xlc.const_data_ptr<int64_t>(),
          ylc.const_data_ptr<int64_t>(), distances.mutable_data_ptr<scalar_t>(),
          indices.mutable_data_ptr<int64_t>(), x.size(1), y.size(1), x.size(2),
          y.size(2), x.size(3));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {distances, indices};
}
}  // namespace
TORCH_LIBRARY_IMPL(torchpairwise, CUDA, m) {
  m.impl("_pointset_nn", TORCH_FN(pointset_nn_kernel));
}
}  // namespace torchpairwise::ops
