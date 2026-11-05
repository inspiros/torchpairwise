#include <ATen/ATen.h>
#include <torch/library.h>

#include "../utils/dispatch.h"
#include "cuda_helpers.h"
#include "pair_reduction.cuh"

namespace torchpairwise::ops {
namespace {
constexpr unsigned int GET_THREADS() {
  return 1024;
}

namespace impl {
template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void _haversine_forward_kernel_impl(
    index_t n_kernels,
    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits,
                                          index_t>
        x1,
    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits,
                                          index_t>
        x2,
    at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t>
        output) {
  CUDA_1D_KERNEL_LOOP(index, n_kernels) {
    index_t j = index % x2.size(1);
    index_t i = (index / x2.size(1)) % x1.size(1);
    index_t b = index / (x2.size(1) * x1.size(1));

    scalar_t a = pow(sin((x1[b][i][0] - x2[b][j][0]) / 2), 2) +
                 cos(x1[b][i][0]) * cos(x2[b][j][0]) *
                     pow(sin((x1[b][i][1] - x2[b][j][1]) / 2), 2);
    a = a < 0 ? scalar_t(0) : (a > 1 ? scalar_t(1) : a);
    output[b][i][j] = 2 * asin(sqrt(a));
  }
}
}  // namespace impl

at::Tensor haversine_kernel(const at::Tensor& x1, const at::Tensor& x2) {
  at::CheckedFrom c = "_haversine_forward";
  auto args = {at::TensorArg(x1, "x1", 1), at::TensorArg(x2, "x2", 2)};
  at::checkAllSameGPU(c, args);
  at::checkAllSameType(c, args);

  at::cuda::CUDAGuard device_guard(x1.get_device());
  bool unbatched = x1.ndimension() == 2;
  TORCH_CHECK(unbatched || x1.ndimension() == 3,
              "x1 must be 2-D (unbatched) or 3-D (batched) tensor.")
  TORCH_CHECK(x2.ndimension() == (unbatched ? 2 : 3),
              "x2 must be 2-D (unbatched) or 3-D (batched) tensor.")
  TORCH_CHECK(unbatched || x1.size(0) == x2.size(0),
              "batch_size of x1 and x2 do not match.")
  TORCH_CHECK((unbatched && x1.size(1) == 2 && x2.size(1) == 2) ||
                  (!unbatched && x1.size(2) == 2 && x2.size(2) == 2),
              "feature dimension of x1 and x2 must be 2.")

  auto x1_c = x1.contiguous();
  auto x2_c = x2.contiguous();
  if (unbatched) {
    x1_c = x1_c.unsqueeze(0);
    x2_c = x2_c.unsqueeze(0);
  }

  int64_t batch_size = x1_c.size(0);
  int64_t n_kernels = batch_size * x1_c.size(1) * x2_c.size(1);
  auto output =
      at::empty({batch_size, x1_c.size(1), x2_c.size(1)}, x1_c.options());

  const unsigned int threads = GET_THREADS();
  const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x1_c.scalar_type(), "_haversine_forward_cuda",
      ([&] {
        TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(
            n_kernels, CUDA, ([&] {
              auto output_accessor = output.generic_packed_accessor<
                  scalar_t, 3, at::RestrictPtrTraits, index_t>();
              impl::_haversine_forward_kernel_impl<scalar_t, index_t>
                  <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                      n_kernels,
                      x1_c.generic_packed_accessor<
                          scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                      x2_c.generic_packed_accessor<
                          scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                      output_accessor);
            }));
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (unbatched)
    output.squeeze_(0);
  return output;
}

namespace impl {
template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void _haversine_backward_denom_kernel_impl(
    index_t n_kernels,
    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits,
                                          index_t>
        x1,
    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits,
                                          index_t>
        x2,
    at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t>
        denom) {
  CUDA_1D_KERNEL_LOOP(index, n_kernels) {
    index_t j = index % x2.size(1);
    index_t i = (index / x2.size(1)) % x1.size(1);
    index_t b = index / (x2.size(1) * x1.size(1));

    scalar_t val = pow(sin((x1[b][i][0] - x2[b][j][0]) / 2), 2) +
                   cos(x1[b][i][0]) * cos(x2[b][j][0]) *
                       pow(sin((x1[b][i][1] - x2[b][j][1]) / 2), 2);
    val = val < 0 ? scalar_t(0) : (val > 1 ? scalar_t(1) : val);
    denom[b][i][j] = sqrt(val) * sqrt(1 - val);
  }
}

// Latitude (e=0) and longitude (e=1) derivative terms of one pair; rows are
// x1 samples unless second, partners are the other operand.
template <typename scalar_t>
struct HaversineTerm {
  using acc_t = at::opmath_type<scalar_t>;
  cuda_detail::Accessor3<scalar_t> grad, denom, x1, x2;
  bool second;
  __device__ acc_t operator()(int64_t b, int64_t r, int64_t o,
                              int64_t e) const {
    int64_t i = second ? o : r, j = second ? r : o;
    acc_t d = denom[b][i][j];
    if (d == 0)
      return 0;
    acc_t g = acc_t(grad[b][i][j]) / d;
    acc_t lat1 = x1[b][i][0], lon1 = x1[b][i][1];
    acc_t lat2 = x2[b][j][0], lon2 = x2[b][j][1];
    acc_t half = ::sin((lon1 - lon2) / 2);
    if (e == 1) {
      acc_t value = half * ::cos(lat1) * ::cos(lat2) * ::cos((lon1 - lon2) / 2);
      return g * (second ? -value : value);
    }
    if (second)
      return g * (-::sin(lat2) * half * half * ::cos(lat1) -
                  ::sin(lat1 - lat2) / 2);
    return g *
           (-::sin(lat1) * half * half * ::cos(lat2) + ::sin(lat1 - lat2) / 2);
  }
};
}  // namespace impl

std::tuple<at::Tensor, at::Tensor> haversine_backward_kernel(
    const at::Tensor& grad_output, const at::Tensor& x1, const at::Tensor& x2) {
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
  int64_t n_kernels;
  auto grad_x1 = at::zeros_like(x1_c);
  auto grad_x2 = at::zeros_like(x2_c);

  const unsigned int threads = GET_THREADS();
  unsigned int blocks;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, grad_output_c.scalar_type(),
      "_haversine_backward_cuda", ([&] {
        auto denom = at::empty({batch_size, x1_c.size(1), x2_c.size(1)},
                               grad_output.options());
        n_kernels = denom.numel();
        blocks = GET_BLOCKS(threads, n_kernels);
        TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(
            n_kernels, CUDA, ([&] {
              auto denom_accessor = denom.generic_packed_accessor<
                  scalar_t, 3, at::RestrictPtrTraits, index_t>();
              impl::_haversine_backward_denom_kernel_impl<scalar_t, index_t>
                  <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                      n_kernels,
                      x1_c.generic_packed_accessor<
                          scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                      x2_c.generic_packed_accessor<
                          scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                      denom_accessor);
            }));

        C10_CUDA_KERNEL_LAUNCH_CHECK();
        auto accessor = [](const at::Tensor& t) {
          return t.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits,
                                           int64_t>();
        };
        // Parallel exact reductions along each partner axis.
        auto reduce = [&](const at::Tensor& own, const at::Tensor& other,
                          bool second) {
          impl::HaversineTerm<scalar_t> term{accessor(grad_output_c),
                                             accessor(denom), accessor(x1_c),
                                             accessor(x2_c), second};
          return cuda_detail::pair_reduction(term, batch_size, own.size(1),
                                             own.size(2), other.size(1),
                                             own.options())
              .to(own.scalar_type());
        };
        grad_x1 = reduce(x1_c, x2_c, false);
        grad_x2 = reduce(x2_c, x1_c, true);
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (unbatched) {
    grad_x1.squeeze_(0);
    grad_x2.squeeze_(0);
  }
  return std::make_tuple(grad_x1, grad_x2);
}
}  // namespace

TORCH_LIBRARY_IMPL(torchpairwise, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchpairwise::_haversine"),
         TORCH_FN(haversine_kernel));
  m.impl(TORCH_SELECTIVE_NAME("torchpairwise::_haversine_backward"),
         TORCH_FN(haversine_backward_kernel));
}
}  // namespace torchpairwise::ops
