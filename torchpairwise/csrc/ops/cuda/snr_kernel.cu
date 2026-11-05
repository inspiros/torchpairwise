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
__global__ void _snr_forward_kernel_impl(
    index_t n_kernels,
    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits,
                                          index_t>
        x1,
    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits,
                                          index_t>
        x2,
    const at::GenericPackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits,
                                          index_t>
        x1_var,
    at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t>
        output) {
  CUDA_1D_KERNEL_LOOP(index, n_kernels) {
    index_t j = index % x2.size(1);
    index_t i = (index / x2.size(1)) % x1.size(1);
    index_t b = index / (x2.size(1) * x1.size(1));

    scalar_t diff_mean = 0;
    for (index_t k = 0; k < x1.size(2); k++) {
      diff_mean += x2[b][j][k] - x1[b][i][k];
    }
    diff_mean /= x1.size(2);

    scalar_t diff_var = 0;
    for (index_t k = 0; k < x1.size(2); k++) {
      scalar_t tmp = x2[b][j][k] - x1[b][i][k] - diff_mean;
      diff_var += tmp * tmp;
    }
    output[b][i][j] = diff_var / x1_var[b][i];
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void _snr_forward_diff_kernel_impl(
    index_t n_kernels,
    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits,
                                          index_t>
        x1,
    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits,
                                          index_t>
        x2,
    at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t>
        diff_mean,
    at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t>
        diff_var) {
  CUDA_1D_KERNEL_LOOP(index, n_kernels) {
    index_t j = index % x2.size(1);
    index_t i = (index / x2.size(1)) % x1.size(1);
    index_t b = index / (x2.size(1) * x1.size(1));

    scalar_t mean_diff = 0;
    for (index_t k = 0; k < x1.size(2); k++) {
      mean_diff += x2[b][j][k] - x1[b][i][k];
    }
    mean_diff /= x1.size(2);

    scalar_t var_diff = 0;
    for (index_t k = 0; k < x1.size(2); k++) {
      scalar_t v_diff = x2[b][j][k] - x1[b][i][k] - mean_diff;
      var_diff += v_diff * v_diff;
    }
    diff_mean[b][i][j] = mean_diff;
    diff_var[b][i][j] = var_diff;
  }
}
}  // namespace impl

at::Tensor snr_kernel(const at::Tensor& x1, const at::Tensor& x2) {
  at::CheckedFrom c = "_snr_forward";
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
      at::empty({batch_size, x1_c.size(1), x2_c.size(1)}, x1.options());
  if (output.numel() == 0)
    return unbatched ? output.squeeze(0) : output;
  auto x1_var = at::var(x1_c, 2, 0).mul_(x1_c.size(2));
  int64_t n_kernels = output.numel();

  const unsigned int threads = GET_THREADS();
  const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x1.scalar_type(), "_snr_forward_cuda", ([&] {
        TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(
            n_kernels, CUDA, ([&] {
              auto output_accessor = output.generic_packed_accessor<
                  scalar_t, 3, at::RestrictPtrTraits, index_t>();
              impl::_snr_forward_kernel_impl<scalar_t, index_t>
                  <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                      n_kernels,
                      x1_c.generic_packed_accessor<
                          scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                      x2_c.generic_packed_accessor<
                          scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                      x1_var.generic_packed_accessor<
                          scalar_t, 2, at::RestrictPtrTraits, index_t>(),
                      output_accessor);
            }));
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (unbatched)
    output.squeeze_(0);
  return output;
}

namespace impl {
// Reference-row terms (rows i, partners j) of the noise/signal ratio.
template <typename scalar_t>
struct SignalTerm {
  using acc_t = at::opmath_type<scalar_t>;
  cuda_detail::Accessor3<scalar_t> grad, x1, x2, diff_mean, diff_var;
  cuda_detail::Accessor2<scalar_t> x1_mean, x1_var;
  __device__ acc_t operator()(int64_t b, int64_t i, int64_t j,
                              int64_t k) const {
    acc_t delta =
        acc_t(x2[b][j][k]) - acc_t(x1[b][i][k]) - acc_t(diff_mean[b][i][j]);
    acc_t var = x1_var[b][i];
    return -acc_t(grad[b][i][j]) * 2 *
           (delta * var + (acc_t(x1[b][i][k]) - acc_t(x1_mean[b][i])) *
                              acc_t(diff_var[b][i][j])) /
           (var * var);
  }
};

// Estimate-row terms (rows j, partners i).
template <typename scalar_t>
struct NoiseTerm {
  using acc_t = at::opmath_type<scalar_t>;
  cuda_detail::Accessor3<scalar_t> grad, x1, x2, diff_mean;
  cuda_detail::Accessor2<scalar_t> x1_var;
  __device__ acc_t operator()(int64_t b, int64_t j, int64_t i,
                              int64_t k) const {
    return acc_t(grad[b][i][j]) * 2 *
           (acc_t(x2[b][j][k]) - acc_t(x1[b][i][k]) -
            acc_t(diff_mean[b][i][j])) /
           acc_t(x1_var[b][i]);
  }
};
}  // namespace impl

std::tuple<at::Tensor, at::Tensor> snr_backward_kernel(
    const at::Tensor& grad_output, const at::Tensor& x1, const at::Tensor& x2) {
  if (grad_output.numel() == 0)
    return std::make_tuple(at::zeros_like(x1), at::zeros_like(x2));
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

  int64_t n_kernels;
  auto grad_x1 = at::zeros_like(x1_c);
  auto grad_x2 = at::zeros_like(x2_c);
  auto x1_mean = at::mean(x1_c, 2);
  auto x1_var = at::var(x1_c, 2, 0).mul_(x1_c.size(2));
  auto diff_mean = at::empty_like(grad_output_c);
  auto diff_var = at::empty_like(grad_output_c);

  const unsigned int threads = GET_THREADS();
  unsigned int blocks;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x1.scalar_type(), "_snr_backward_cuda", ([&] {
        n_kernels = grad_output_c.numel();
        blocks = GET_BLOCKS(threads, n_kernels);
        TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(
            n_kernels, CUDA, ([&] {
              auto diff_mean_accessor = diff_mean.generic_packed_accessor<
                  scalar_t, 3, at::RestrictPtrTraits, index_t>();
              auto diff_var_accessor = diff_var.generic_packed_accessor<
                  scalar_t, 3, at::RestrictPtrTraits, index_t>();
              impl::_snr_forward_diff_kernel_impl<scalar_t, index_t>
                  <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                      n_kernels,
                      x1_c.generic_packed_accessor<
                          scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                      x2_c.generic_packed_accessor<
                          scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                      diff_mean_accessor, diff_var_accessor);
            }));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        auto accessor = [](const at::Tensor& t) {
          return t.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits,
                                           int64_t>();
        };
        auto rows = [](const at::Tensor& t) {
          return t.generic_packed_accessor<scalar_t, 2, at::RestrictPtrTraits,
                                           int64_t>();
        };
        // Parallel exact reductions along each partner axis.
        impl::SignalTerm<scalar_t> signal{accessor(grad_output_c),
                                          accessor(x1_c),
                                          accessor(x2_c),
                                          accessor(diff_mean),
                                          accessor(diff_var),
                                          rows(x1_mean),
                                          rows(x1_var)};
        grad_x1 = cuda_detail::pair_reduction(signal, x1_c.size(0),
                                              x1_c.size(1), x1_c.size(2),
                                              x2_c.size(1), x1_c.options())
                      .to(x1_c.scalar_type());
        impl::NoiseTerm<scalar_t> noise{accessor(grad_output_c), accessor(x1_c),
                                        accessor(x2_c), accessor(diff_mean),
                                        rows(x1_var)};
        grad_x2 = cuda_detail::pair_reduction(noise, x2_c.size(0), x2_c.size(1),
                                              x2_c.size(2), x1_c.size(1),
                                              x2_c.options())
                      .to(x2_c.scalar_type());
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
  m.impl(TORCH_SELECTIVE_NAME("torchpairwise::_snr"), TORCH_FN(snr_kernel));
  m.impl(TORCH_SELECTIVE_NAME("torchpairwise::_snr_backward"),
         TORCH_FN(snr_backward_kernel));
}
}  // namespace torchpairwise::ops
