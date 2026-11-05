#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/cuda/Atomic.cuh>
#include <type_traits>

#define CUDA_1D_KERNEL_LOOP_T(i, n, index_t)                            \
  for (index_t i =                                                      \
           static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
       i < (n); i += static_cast<index_t>(blockDim.x) * gridDim.x)

#define CUDA_1D_KERNEL_LOOP(i, n) \
  CUDA_1D_KERNEL_LOOP_T(          \
      i, n, std::remove_cv_t<std::remove_reference_t<decltype(n)>>)

inline unsigned int GET_BLOCKS(const unsigned int THREADS, const int64_t N) {
  TORCH_CHECK(N >= 0, "Kernel work count must be non-negative");
  const int64_t max_grid =
      at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  // Empty work is safe: every kernel guards its first access with i < N.
  // Keep a valid launch grid for forward and backward empty tensors alike.
  return static_cast<unsigned int>(std::max<int64_t>(
      1, std::min(max_grid, N / THREADS + (N % THREADS != 0))));
}

// Temporarily counter latest MSVC update that causes incompatibility with CUDA
#if (_MSC_VER >= 1928)
#define floor floorf
#define ceil ceilf
#endif
