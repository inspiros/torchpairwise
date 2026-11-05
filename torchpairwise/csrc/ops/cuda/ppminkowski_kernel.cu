#include <ATen/ATen.h>
#include <torch/library.h>

#include "../utils/dispatch.h"
#include "cuda_helpers.h"
#include "pair_reduction.cuh"
#include "signum.cuh"

namespace torchpairwise::ops {
namespace {
constexpr unsigned int GET_THREADS() {
  return 1024;
}

namespace impl {
// A 16x16 block computes a 32x32 output tile (2x2 pairs per thread).
// Transposed, padded shared tiles give coalesced global row loads and
// contiguous/broadcast shared reads while keeping each pair's k order intact.
constexpr int pair_tile = 32;
constexpr int feature_tile = 32;
constexpr int tile_threads = 16;

template <int power, bool infinite, bool neg, typename scalar_t,
          typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void _ppminkowski_kernel_impl(
    index_t n_tiles,
    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits,
                                          index_t>
        x1,
    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits,
                                          index_t>
        x2,
    scalar_t p,
    at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t>
        output) {
  __shared__ scalar_t left[feature_tile][pair_tile + 1];
  __shared__ scalar_t right[feature_tile][pair_tile + 1];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int tid = ty * tile_threads + tx;
  const index_t tiles_i = (x1.size(1) - 1) / pair_tile + 1;
  const index_t tiles_j = (x2.size(1) - 1) / pair_tile + 1;
  // Flatten batch and both sample axes so neither grid.y nor grid.z limits
  // the accepted shapes. All threads traverse the same tiles and barriers.
  for (index_t tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
    const index_t b = tile / (tiles_i * tiles_j);
    const index_t i = (tile / tiles_j % tiles_i) * pair_tile;
    const index_t j = (tile % tiles_j) * pair_tile;
    scalar_t values[2][2] = {};
    for (index_t start = 0; start < x1.size(2);) {
      const int count = static_cast<int>(x1.size(2) - start < feature_tile
                                             ? x1.size(2) - start
                                             : feature_tile);
      for (int slot = tid; slot < pair_tile * feature_tile;
           slot += tile_threads * tile_threads) {
        const int row = slot / feature_tile, k = slot % feature_tile;
        left[k][row] = i + row < x1.size(1) && k < count
                           ? x1[b][i + row][start + k]
                           : scalar_t(0);
        right[k][row] = j + row < x2.size(1) && k < count
                            ? x2[b][j + row][start + k]
                            : scalar_t(0);
      }
      __syncthreads();
      for (int k = 0; k < count; ++k) {
#pragma unroll
        for (int a = 0; a < 2; ++a) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            auto delta = left[k][ty + a * tile_threads] -
                         right[k][tx + c * tile_threads];
            auto& val = values[a][c];
            if constexpr (infinite) {
              scalar_t tmp = fabs(delta);
              // Every feature propagates NaNs, independent of its position.
              if (start == 0 && k == 0)
                val = tmp;
              else if (isnan(tmp) || (neg ? tmp < val : tmp > val))
                val = tmp;
            } else if constexpr (power == 2)
              val += delta * delta;
            else if constexpr (power == 1)
              val += fabs(delta);
            else
              val += pow(fabs(delta), p);
          }
        }
      }
      // Even threads with no valid output must finish reading this tile
      // before any warp starts overwriting it (including the next pair tile).
      __syncthreads();
      start += count;
    }
#pragma unroll
    for (int a = 0; a < 2; ++a) {
#pragma unroll
      for (int c = 0; c < 2; ++c)
        if (i + ty + a * tile_threads < x1.size(1) &&
            j + tx + c * tile_threads < x2.size(1))
          output[b][i + ty + a * tile_threads][j + tx + c * tile_threads] =
              values[a][c];
    }
  }
}
}  // namespace impl

at::Tensor ppminkowski_kernel(const at::Tensor& x1, const at::Tensor& x2,
                              double p) {
  at::CheckedFrom c = "_ppminkowski_forward";
  auto args = {at::TensorArg(x1, "x1", 1), at::TensorArg(x2, "x2", 2)};
  at::checkAllSameGPU(c, args);
  at::checkAllSameType(c, args);

  at::cuda::CUDAGuard device_guard(x1.get_device());
  bool unbatched = x1.ndimension() == 2;
  TORCH_CHECK(unbatched || x1.ndimension() == 3,
              "x1 must be 2-D (unbatched) or 3-D (batched) tensor.")
  TORCH_CHECK(x2.ndimension() == (unbatched ? 2 : 3),
              "x2 must be 2-D (unbatched) or 3-D (batched) tensor.")
  TORCH_CHECK(unbatched || (x1.size(0) == x2.size(0)),
              "batch_size of x1 and x2 do not match.")
  TORCH_CHECK((unbatched && x1.size(1) == x2.size(1)) ||
                  (!unbatched && x1.size(2) == x2.size(2)),
              "feature dimension of x1 and x2 do not match.")

  auto x1_c = x1.contiguous();
  auto x2_c = x2.contiguous();
  if (unbatched) {
    x1_c = x1_c.unsqueeze(0);
    x2_c = x2_c.unsqueeze(0);
  }

  int64_t batch_size = x1_c.size(0);
  auto output =
      at::empty({batch_size, x1_c.size(1), x2_c.size(1)}, x1_c.options());
  int64_t n_kernels = output.numel();

  if (p == 0) {
    output.fill_(x1_c.size(2));
  } else if (x1_c.size(2) == 0) {
    // Empty feature reductions have distance zero, including +/-infinity.
    output.zero_();
  } else if (n_kernels != 0) {
    const dim3 threads(impl::tile_threads, impl::tile_threads);
    const int64_t tiles_i = (x1_c.size(1) - 1) / impl::pair_tile + 1;
    const int64_t tiles_j = (x2_c.size(1) - 1) / impl::pair_tile + 1;
    const int64_t n_tiles = batch_size * tiles_i * tiles_j;
    const unsigned int blocks = GET_BLOCKS(1, n_tiles);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x1_c.scalar_type(),
        "_ppminkowski_forward_cuda", ([&] {
          TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(
              n_kernels, CUDA, ([&] {
                auto output_accessor = output.generic_packed_accessor<
                    scalar_t, 3, at::RestrictPtrTraits, index_t>();
                if (std::isinf(p)) {
                  TORCHPAIRWISE_DISPATCH_BOOL_NAME(
                      negative, p < 0, ([&] {
                        impl::_ppminkowski_kernel_impl<
                            0, true, negative, scalar_t,
                            index_t><<<blocks, threads, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
                            n_tiles,
                            x1_c.generic_packed_accessor<
                                scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                            x2_c.generic_packed_accessor<
                                scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                            static_cast<scalar_t>(p), output_accessor);
                      }));
                } else {
                  // Generic lambda: `.template` for nvcc on MSVC.
                  auto run = [&](auto power) {
                    impl::_ppminkowski_kernel_impl<
                        decltype(power)::value, false, false, scalar_t, index_t>
                        <<<blocks, threads, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
                            n_tiles,
                            x1_c.template generic_packed_accessor<
                                scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                            x2_c.template generic_packed_accessor<
                                scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                            static_cast<scalar_t>(p), output_accessor);
                  };
                  if constexpr (std::is_same_v<scalar_t, float> ||
                                std::is_same_v<scalar_t, double>) {
                    if (p == 2.)
                      run(std::integral_constant<int, 2>{});
                    else if (p == 1.)
                      run(std::integral_constant<int, 1>{});
                    else
                      run(std::integral_constant<int, 0>{});
                  } else {
                    // Keep reduced precision on its existing path without
                    // instantiating unused specialized kernel variants.
                    run(std::integral_constant<int, 0>{});
                  }
                }
              }));
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (unbatched)
    output.squeeze_(0);
  return output;
}

namespace impl {
// p * |own - other|^(p-1) * sign(own - other), weighted by the pair gradient.
template <typename scalar_t>
struct PowerTerm {
  using acc_t = at::opmath_type<scalar_t>;
  cuda_detail::Accessor3<scalar_t> grad, own, other;
  bool transpose;
  acc_t p;
  __device__ acc_t operator()(int64_t b, int64_t r, int64_t o,
                              int64_t e) const {
    acc_t val = acc_t(own[b][r][e]) - acc_t(other[b][o][e]);
    if (val == acc_t(0))
      return acc_t(0);
    return acc_t(cuda_detail::pair_entry(grad, transpose, b, r, o)) * p *
           ::pow(::fabs(val), p - 1) * m_signum(val);
  }
};

// Keep the extremum subgradient convention in one helper: split exact ties.
template <typename acc_t>
__device__ acc_t extremum_tie_weight(acc_t magnitude, acc_t distance,
                                     acc_t inverse_count) {
  return distance != acc_t(0) && magnitude == distance ? inverse_count
                                                       : acc_t(0);
}

template <bool neg, typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void extremum_pair_stats_kernel(
    int64_t n_kernels, cuda_detail::Accessor3<scalar_t> x1,
    cuda_detail::Accessor3<scalar_t> x2,
    cuda_detail::Accessor3<at::opmath_type<scalar_t>> distances,
    cuda_detail::Accessor3<at::opmath_type<scalar_t>> inverse_counts) {
  using acc_t = at::opmath_type<scalar_t>;
  CUDA_1D_KERNEL_LOOP_T(index, n_kernels, int64_t) {
    const int64_t j = index % x2.size(1);
    const int64_t i = index / x2.size(1) % x1.size(1);
    const int64_t b = index / (x2.size(1) * x1.size(1));
    acc_t distance = 0;
    int64_t count = 0;
    for (int64_t k = 0; k < x1.size(2); ++k) {
      // Match the forward subtraction in the input dtype.
      scalar_t delta = x1[b][i][k] - x2[b][j][k];
      const acc_t magnitude = ::fabs(acc_t(delta));
      if (k == 0 || isnan(magnitude) ||
          (neg ? magnitude < distance : magnitude > distance)) {
        distance = magnitude;
        count = 1;
      } else if (magnitude == distance) {
        ++count;
      }
    }
    distances[b][i][j] = distance;
    inverse_counts[b][i][j] = count == 0 ? acc_t(0) : acc_t(1) / count;
  }
}

template <typename scalar_t>
struct ExtremumTerm {
  using acc_t = at::opmath_type<scalar_t>;
  cuda_detail::Accessor3<scalar_t> grad, own, other;
  cuda_detail::Accessor3<acc_t> distances, inverse_counts;
  bool transpose;
  __device__ acc_t operator()(int64_t b, int64_t r, int64_t o,
                              int64_t e) const {
    scalar_t delta = own[b][r][e] - other[b][o][e];
    const acc_t val = acc_t(delta);
    const acc_t weight = extremum_tie_weight(
        ::fabs(val), cuda_detail::pair_entry(distances, transpose, b, r, o),
        cuda_detail::pair_entry(inverse_counts, transpose, b, r, o));
    if (weight == acc_t(0))
      return acc_t(0);
    return acc_t(cuda_detail::pair_entry(grad, transpose, b, r, o)) *
           m_signum(val) * weight;
  }
};
}  // namespace impl

std::tuple<at::Tensor, at::Tensor> ppminkowski_backward_kernel(
    const at::Tensor& grad_output, const at::Tensor& x1, const at::Tensor& x2,
    double p) {
  at::cuda::CUDAGuard device_guard(grad_output.get_device());
  bool unbatched = x1.ndimension() == 2;

  auto grad_output_c = grad_output.contiguous();
  auto x1_c = x1.contiguous();
  auto x2_c = x2.contiguous();
  if (unbatched) {
    grad_output_c = grad_output_c.unsqueeze(0);
    x1_c = x1_c.unsqueeze(0);
    x2_c = x2_c.unsqueeze(0);
  }

  int64_t batch_size = x1_c.size(0);
  auto grad_x1 = at::zeros_like(x1_c);
  auto grad_x2 = at::zeros_like(x2_c);

  if (p != 0) {
    const unsigned int threads = GET_THREADS();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, grad_output_c.scalar_type(),
        "_ppminkowski_backward_cuda", ([&] {
          if (std::isinf(p)) {
            using acc_t = at::opmath_type<scalar_t>;
            auto accessor = [](const at::Tensor& t) {
              return t.generic_packed_accessor<
                  scalar_t, 3, at::RestrictPtrTraits, int64_t>();
            };
            auto options =
                x1_c.options().dtype(c10::CppTypeToScalarType<acc_t>::value);
            auto distances = at::empty(grad_output_c.sizes(), options);
            auto inverse_counts = at::empty_like(distances);
            auto distances_accessor = distances.generic_packed_accessor<
                acc_t, 3, at::RestrictPtrTraits, int64_t>();
            auto inverse_counts_accessor =
                inverse_counts.generic_packed_accessor<
                    acc_t, 3, at::RestrictPtrTraits, int64_t>();
            const int64_t n_kernels = grad_output_c.numel();
            if (n_kernels != 0) {
              TORCHPAIRWISE_DISPATCH_BOOL_NAME(
                  negative, p < 0, ([&] {
                    impl::extremum_pair_stats_kernel<negative, scalar_t>
                        <<<GET_BLOCKS(threads, n_kernels), threads, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
                            n_kernels, accessor(x1_c), accessor(x2_c),
                            distances_accessor, inverse_counts_accessor);
                  }));
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            auto reduce = [&](const at::Tensor& own, const at::Tensor& other,
                              bool transpose) {
              impl::ExtremumTerm<scalar_t> term{
                  accessor(grad_output_c), accessor(own),
                  accessor(other),         distances_accessor,
                  inverse_counts_accessor, transpose};
              return cuda_detail::pair_reduction(term, batch_size, own.size(1),
                                                 own.size(2), other.size(1),
                                                 own.options())
                  .to(own.scalar_type());
            };
            grad_x1 = reduce(x1_c, x2_c, false);
            grad_x2 = reduce(x2_c, x1_c, true);
          } else {
            // Exact per-pair terms, reduced in parallel along each partner
            // axis.
            auto accessor = [](const at::Tensor& t) {
              return t.generic_packed_accessor<
                  scalar_t, 3, at::RestrictPtrTraits, int64_t>();
            };
            auto reduce = [&](const at::Tensor& own, const at::Tensor& other,
                              bool transpose) {
              auto options = own.options();
              if (p == 2) {
                cuda_detail::DifferenceTerm<scalar_t> term{
                    accessor(grad_output_c), accessor(own), accessor(other),
                    transpose};
                return cuda_detail::pair_reduction(term, batch_size,
                                                   own.size(1), own.size(2),
                                                   other.size(1), options)
                    .mul_(2)
                    .to(own.scalar_type());
              }
              impl::PowerTerm<scalar_t> term{
                  accessor(grad_output_c), accessor(own), accessor(other),
                  transpose, static_cast<at::opmath_type<scalar_t>>(p)};
              return cuda_detail::pair_reduction(term, batch_size, own.size(1),
                                                 own.size(2), other.size(1),
                                                 options)
                  .to(own.scalar_type());
            };
            grad_x1 = reduce(x1_c, x2_c, false);
            grad_x2 = reduce(x2_c, x1_c, true);
          }
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (unbatched) {
    grad_x1.squeeze_(0);
    grad_x2.squeeze_(0);
  }
  return std::make_tuple(grad_x1, grad_x2);
}
}  // namespace

TORCH_LIBRARY_IMPL(torchpairwise, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchpairwise::_ppminkowski"),
         TORCH_FN(ppminkowski_kernel));
  m.impl(TORCH_SELECTIVE_NAME("torchpairwise::_ppminkowski_backward"),
         TORCH_FN(ppminkowski_backward_kernel));
}
}  // namespace torchpairwise::ops
