#include <ATen/Dispatch.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "../image_map_operations_detail.h"
#include "cuda_helpers.h"

namespace torchpairwise::ops::detail::media {
namespace {
template <typename scalar_t>
using Values = at::GenericPackedTensorAccessor<scalar_t, 5,
                                               at::RestrictPtrTraits, int64_t>;
using ProductState =
    at::GenericPackedTensorAccessor<double, 6, at::RestrictPtrTraits, int64_t>;
// One reduction combines all seven product fields from rounded public map
// entries, with a rare bounded replay for unstable log magnitudes. The small
// partial buffer is proportional to output rows and split count, never to the
// full pair/event map.
struct ProductPartial {
  double raw, zeros, finite, negatives, infs, nans, log_magnitude;
};
__device__ __forceinline__ ProductPartial product_identity() {
  return {1., 0., 1., 0., 0., 0., 0.};
}
__device__ __forceinline__ ProductPartial merge_product(ProductPartial a,
                                                        ProductPartial b) {
  a.raw *= b.raw;
  a.zeros += b.zeros;
  a.finite *= b.finite;
  a.negatives += b.negatives;
  a.infs += b.infs;
  a.nans += b.nans;
  a.log_magnitude += b.log_magnitude;
  return a;
}
__device__ __forceinline__ ProductPartial product_entry(double value) {
  bool zero = value == 0., inf = ::isinf(value), nan = ::isnan(value);
  ProductPartial result;
  result.raw = value;
  result.zeros = double(zero);
  result.finite = zero || inf || nan ? 1. : value;
  result.negatives = ::signbit(value) ? 1. : 0.;
  result.infs = double(inf);
  result.nans = double(nan);
  result.log_magnitude = 0.;
  return result;
}
__device__ __forceinline__ void update_product(ProductState state,
                                               int64_t output,
                                               ProductPartial part) {
  auto w = output % state.size(4), t = output / state.size(4);
  auto h = t % state.size(3);
  t /= state.size(3);
  auto c = t % state.size(2);
  t /= state.size(2);
  auto j = t % state.size(1), i = t / state.size(1);
  auto saved = state[i][j][c][h][w];
  saved[0] *= part.raw;
  saved[1] += part.zeros;
  saved[2] *= part.finite;
  saved[3] += part.negatives;
  saved[4] += part.infs;
  saved[5] += part.nans;
  saved[6] += part.log_magnitude;
}
template <typename scalar_t>
__global__ void image_product_state_kernel(int64_t outputs, int64_t reduced,
                                           int64_t chunks, int mask,
                                           Values<scalar_t> value,
                                           ProductState state,
                                           double* partial) {
  extern __shared__ ProductPartial shared[];
  int lane = threadIdx.x;
  auto rh = mask & 2 ? value.size(3) : int64_t(1);
  auto rw = mask & 4 ? value.size(4) : int64_t(1);
  auto chunk_size = (reduced + chunks - 1) / chunks;
  for (int64_t block = blockIdx.x; block < outputs * chunks;
       block += gridDim.x) {
    auto output = block / chunks, chunk = block % chunks;
    auto ow = output % state.size(4), t = output / state.size(4);
    auto oh = t % state.size(3);
    t /= state.size(3);
    auto oc = t % state.size(2);
    t /= state.size(2);
    auto j = t % state.size(1), i = t / state.size(1);
    auto begin = chunk * chunk_size;
    auto end = begin + chunk_size < reduced ? begin + chunk_size : reduced;
    auto local = product_identity();
    for (int64_t r = begin + lane; r < end; r += blockDim.x) {
      auto w = mask & 4 ? r % rw : ow;
      auto h = mask & 2 ? (r / rw) % rh : oh;
      auto c = mask & 1 ? r / rw / rh : oc;
      local = merge_product(local, product_entry(value[i][j][c][h][w]));
    }
    shared[lane] = local;
    __syncthreads();
    for (int step = blockDim.x / 2; step; step /= 2) {
      if (lane < step)
        shared[lane] = merge_product(shared[lane], shared[lane + step]);
      __syncthreads();
    }
    // A representable normal partial already contains its finite factors'
    // magnitude. One logarithm replaces a logarithm at every map entry. If
    // multiplication lost that magnitude, replay only this bounded partial
    // and sum individual logs; no host predicate or temporary map is needed.
    auto magnitude = ::fabs(shared[0].finite);
    bool stable = ::isfinite(magnitude) &&
                  magnitude >= std::numeric_limits<double>::min();
    if (stable) {
      if (!lane)
        shared[0].log_magnitude = ::log(magnitude);
    } else {
      double logarithm = 0.;
      for (int64_t r = begin + lane; r < end; r += blockDim.x) {
        auto w = mask & 4 ? r % rw : ow;
        auto h = mask & 2 ? (r / rw) % rh : oh;
        auto c = mask & 1 ? r / rw / rh : oc;
        double v = value[i][j][c][h][w];
        if (v != 0. && ::isfinite(v))
          logarithm += ::log(::fabs(v));
      }
      shared[lane].log_magnitude = logarithm;
      __syncthreads();
      for (int step = blockDim.x / 2; step; step /= 2) {
        if (lane < step)
          shared[lane].log_magnitude += shared[lane + step].log_magnitude;
        __syncthreads();
      }
    }
    if (!lane) {
      auto part = shared[0];
      if (chunks == 1)
        update_product(state, output, part);
      else {
        auto base = partial + block * 7;
        base[0] = part.raw;
        base[1] = part.zeros;
        base[2] = part.finite;
        base[3] = part.negatives;
        base[4] = part.infs;
        base[5] = part.nans;
        base[6] = part.log_magnitude;
      }
    }
    __syncthreads();
  }
}
__global__ void image_product_merge_kernel(int64_t outputs, int64_t chunks,
                                           const double* partial,
                                           ProductState state) {
  CUDA_1D_KERNEL_LOOP_T(output, outputs, int64_t) {
    auto total = product_identity();
    for (int64_t k = 0; k < chunks; ++k) {
      auto base = partial + (output * chunks + k) * 7;
      ProductPartial part;
      part.raw = base[0];
      part.zeros = base[1];
      part.finite = base[2];
      part.negatives = base[3];
      part.infs = base[4];
      part.nans = base[5];
      part.log_magnitude = base[6];
      total = merge_product(total, part);
    }
    update_product(state, output, total);
  }
}
template <typename scalar_t>
__global__ void image_product_gradient_kernel(
    int64_t count, Values<scalar_t> value, Values<double> gradient,
    ProductState state, const bool* any_zero, Values<double> output) {
  CUDA_1D_KERNEL_LOOP_T(index, count, int64_t) {
    auto w = index % value.size(4), t = index / value.size(4);
    auto h = t % value.size(3);
    t /= value.size(3);
    auto c = t % value.size(2);
    t /= value.size(2);
    auto j = t % value.size(1), i = t / value.size(1);
    auto sc = state.size(2) == 1 ? 0 : c;
    auto sh = state.size(3) == 1 ? 0 : h;
    auto sw = state.size(4) == 1 ? 0 : w;
    auto si = state.size(0) == 1 ? 0 : i;
    auto sj = state.size(1) == 1 ? 0 : j;
    auto g =
        gradient[gradient.size(0) == 1 ? 0 : i][gradient.size(1) == 1 ? 0 : j]
                [gradient.size(2) == 1 ? 0 : c][gradient.size(3) == 1 ? 0 : h]
                [gradient.size(4) == 1 ? 0 : w];
    double v = value[i][j][c][h][w];
    auto saved = state[si][sj][sc][sh][sw];
    double factor;
    // CUDA branching skips exceptional-factor/log work on ordinary no-zero
    // inputs. No host predicate or synchronization is introduced.
    if (!any_zero[0]) {
      factor = saved[0] / v;
    } else {
      bool zero = v == 0., inf = ::isinf(v), nan = ::isnan(v);
      bool exceptional = zero || inf || nan;
      double zeros = saved[1] - double(zero);
      double negatives = saved[3] - double(::signbit(v));
      double infs = saved[4] - double(inf), nans = saved[5] - double(nan);
      bool negative = ::fmod(negatives, 2.) != 0.;
      if (nans > 0. || (zeros > 0. && infs > 0.))
        factor = std::numeric_limits<double>::quiet_NaN();
      else if (zeros > 0.)
        factor = negative ? -0. : 0.;
      else if (infs > 0.)
        factor = negative ? -std::numeric_limits<double>::infinity()
                          : std::numeric_limits<double>::infinity();
      else if (::isfinite(saved[2]) && saved[2] != 0.)
        factor = saved[2] / (exceptional ? 1. : v);
      else {
        double magnitude =
            ::exp(saved[6] - ::log(exceptional ? 1. : ::fabs(v)));
        factor = negative ? -magnitude : magnitude;
      }
    }
    output[i][j][c][h][w] = g * factor;
  }
}
}  // namespace
void image_product_update_cuda(Tensor state, const Tensor& value,
                               at::IntArrayRef axes) {
  const at::cuda::CUDAGuard guard(value.device());
  auto outputs = state.numel() / 7;
  if (!outputs)
    return;
  auto values = value.dim() == 4 ? value.unsqueeze(0) : value;
  auto saved = state.dim() == 5 ? state.unsqueeze(0) : state;
  int mask = 0;
  int64_t reduced = 1;
  for (auto axis : axes) {
    mask |= 1 << (axis + (value.dim() == 4 ? 1 : 0) - 2);
    reduced *= value.size(axis);
  }
  // Split long rows enough to fill the device, with at least 1024 factors per
  // partial. All merges have a fixed order and use no atomic updates.
  auto chunks = std::min(std::max<int64_t>(1, (512 + outputs - 1) / outputs),
                         std::max<int64_t>(1, reduced / 1024));
  int threads = 32;
  while (threads < 256 && threads < reduced)
    threads *= 2;
  auto partial = chunks > 1 ? at::empty({outputs, chunks, 7},
                                        value.options().dtype(at::kDouble))
                            : Tensor();
  auto state_accessor =
      saved.packed_accessor64<double, 6, at::RestrictPtrTraits>();
  AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "image_product_state", [&] {
    image_product_state_kernel<<<GET_BLOCKS(1, outputs * chunks), threads,
                                 threads * sizeof(ProductPartial),
                                 at::cuda::getCurrentCUDAStream()>>>(
        outputs, reduced, chunks, mask,
        values.packed_accessor64<scalar_t, 5, at::RestrictPtrTraits>(),
        state_accessor,
        chunks > 1 ? partial.mutable_data_ptr<double>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  if (chunks > 1) {
    image_product_merge_kernel<<<GET_BLOCKS(256, outputs), 256, 0,
                                 at::cuda::getCurrentCUDAStream()>>>(
        outputs, chunks, partial.const_data_ptr<double>(), state_accessor);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
Tensor image_product_gradient_cuda(const Tensor& state, const Tensor& value,
                                   const Tensor& gradient,
                                   const Tensor& any_zero) {
  const at::cuda::CUDAGuard guard(value.device());
  auto result = at::empty(value.sizes(), value.options().dtype(at::kDouble));
  if (!result.numel())
    return result;
  // Both Cartesian and paired image regions use the same fixed-rank kernel;
  // the paired case adds a singleton sample axis, without copying data.
  auto values = value.dim() == 4 ? value.unsqueeze(0) : value;
  auto gradients = gradient.dim() == 4 ? gradient.unsqueeze(0) : gradient;
  auto saved = state.dim() == 5 ? state.unsqueeze(0) : state;
  auto output = result.dim() == 4 ? result.unsqueeze(0) : result;
  AT_DISPATCH_FLOATING_TYPES(
      value.scalar_type(), "image_product_gradient", [&] {
        image_product_gradient_kernel<<<GET_BLOCKS(256, result.numel()), 256, 0,
                                        at::cuda::getCurrentCUDAStream()>>>(
            result.numel(),
            values.packed_accessor64<scalar_t, 5, at::RestrictPtrTraits>(),
            gradients.packed_accessor64<double, 5, at::RestrictPtrTraits>(),
            saved.packed_accessor64<double, 6, at::RestrictPtrTraits>(),
            any_zero.const_data_ptr<bool>(),
            output.packed_accessor64<double, 5, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return result;
}
}  // namespace torchpairwise::ops::detail::media
