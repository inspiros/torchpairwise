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
__global__ void _sqmahalanobis_forward_kernel_impl(
    index_t n_kernels,
    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits,
                                          index_t>
        x1,
    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits,
                                          index_t>
        x2,
    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits,
                                          index_t>
        VI,
    at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t>
        output) {
  CUDA_1D_KERNEL_LOOP(index, n_kernels) {
    index_t j = index % x2.size(1);
    index_t i = (index / x2.size(1)) % x1.size(1);
    index_t b = index / (x2.size(1) * x1.size(1));

    scalar_t val = 0, diff_k, diff_l;
    for (index_t k = 0; k < x1.size(2); k++) {
      for (index_t l = 0; l < x2.size(2); l++) {
        diff_k = x1[b][i][k] - x2[b][j][k];
        diff_l = x1[b][i][l] - x2[b][j][l];
        val += diff_k * VI[b][k][l] * diff_l;
      }
    }
    output[b][i][j] = val;
  }
}
}  // namespace impl

at::Tensor sqmahalanobis_kernel(const at::Tensor& x1, const at::Tensor& x2,
                                const at::Tensor& VI) {
  at::CheckedFrom c = "_sqmahalanobis_forward";
  auto args = {at::TensorArg(x1, "x1", 1), at::TensorArg(x2, "x2", 2),
               at::TensorArg(VI, "VI", 3)};
  at::checkAllSameGPU(c, args);
  at::checkAllSameType(c, args);

  at::cuda::CUDAGuard device_guard(x1.get_device());
  bool unbatched = x1.ndimension() == 2;
  TORCH_CHECK(unbatched || x1.ndimension() == 3,
              "x1 must be 2-D (unbatched) or 3-D (batched) tensor.")
  TORCH_CHECK(x2.ndimension() == (unbatched ? 2 : 3),
              "x2 must be 2-D (unbatched) or 3-D (batched) tensor.")
  TORCH_CHECK(VI.ndimension() == (unbatched ? 2 : 3),
              "VI must be 2-D (unbatched) or 3-D (batched) tensor.")
  TORCH_CHECK(VI.size(-1) == VI.size(-2), "VI must be square");
  TORCH_CHECK(
      unbatched || (x1.size(0) == x2.size(0) && x1.size(0) == VI.size(0)),
      "batch_size of x1, x2, and VI do not match.")
  TORCH_CHECK(
      (unbatched && x1.size(1) == x2.size(1) && x1.size(1) == VI.size(0)) ||
          (!unbatched && x1.size(2) == x2.size(2) && x1.size(2) == VI.size(1)),
      "feature dimension of x1, x2, and VI do not match.")

  auto x1_c = x1.contiguous();
  auto x2_c = x2.contiguous();
  auto VI_c = VI.contiguous();
  if (unbatched) {
    x1_c = x1_c.unsqueeze(0);
    x2_c = x2_c.unsqueeze(0);
    VI_c = VI_c.unsqueeze(0);
  }

  int64_t batch_size = x1_c.size(0);
  int64_t n_kernels = batch_size * x1_c.size(1) * x2_c.size(1);
  auto output =
      at::empty({batch_size, x1_c.size(1), x2_c.size(1)}, x1_c.options());

  const unsigned int threads = GET_THREADS();
  const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x1_c.scalar_type(),
      "_sqmahalanobis_forward_cpu", ([&] {
        TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(
            n_kernels, CUDA, ([&] {
              auto output_accessor = output.generic_packed_accessor<
                  scalar_t, 3, at::RestrictPtrTraits, index_t>();
              impl::_sqmahalanobis_forward_kernel_impl<scalar_t, index_t>
                  <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                      n_kernels,
                      x1_c.generic_packed_accessor<
                          scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                      x2_c.generic_packed_accessor<
                          scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                      VI_c.generic_packed_accessor<
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
// Pair gradient times the outer product of the pair difference; o
// enumerates the flattened (i, j) pairs and e the flattened (k, l) entries.
template <typename scalar_t>
struct OuterTerm {
  using acc_t = at::opmath_type<scalar_t>;
  cuda_detail::Accessor3<scalar_t> grad, x1, x2;
  int64_t m, d;
  __device__ acc_t operator()(int64_t b, int64_t, int64_t o, int64_t e) const {
    int64_t i = o / m, j = o % m, k = e / d, l = e % d;
    return acc_t(grad[b][i][j]) * (acc_t(x1[b][i][k]) - acc_t(x2[b][j][k])) *
           (acc_t(x1[b][i][l]) - acc_t(x2[b][j][l]));
  }
};
}  // namespace impl

std::tuple<at::Tensor, at::Tensor, at::Tensor> sqmahalanobis_backward_kernel(
    const at::Tensor& grad_output, const at::Tensor& x1, const at::Tensor& x2,
    const at::Tensor& VI) {
  at::cuda::CUDAGuard device_guard(grad_output.get_device());
  bool unbatched = x1.ndimension() == 2;

  auto grad_output_c = grad_output.contiguous();
  auto x1_c = x1.contiguous();
  auto x2_c = x2.contiguous();
  auto VI_c = VI.contiguous();
  if (unbatched) {
    grad_output_c = grad_output_c.unsqueeze(0);
    x1_c = x1_c.unsqueeze(0);
    x2_c = x2_c.unsqueeze(0);
    VI_c = VI_c.unsqueeze(0);
  }

  int64_t batch_size = x1_c.size(0);
  auto grad_x1 = at::zeros_like(x1_c);
  auto grad_x2 = at::zeros_like(x2_c);
  auto grad_VI = at::zeros_like(VI_c);

  // d/dx (x-y)^T VI (x-y) = (VI + VI^T)(x-y): reduce the exact pair
  // differences first, then apply the symmetric matrix once per sample.
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, grad_output_c.scalar_type(),
      "_sqmahalanobis_backward_cuda", ([&] {
        auto accessor = [](const at::Tensor& t) {
          return t.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits,
                                           int64_t>();
        };
        auto difference = [&](const at::Tensor& own, const at::Tensor& other,
                              bool transpose) {
          cuda_detail::DifferenceTerm<scalar_t> term{
              accessor(grad_output_c), accessor(own), accessor(other),
              transpose};
          return cuda_detail::pair_reduction(term, batch_size, own.size(1),
                                             own.size(2), other.size(1),
                                             own.options());
        };
        auto u = difference(x1_c, x2_c, false);
        auto symmetric = VI_c.to(u.scalar_type());
        symmetric = symmetric + symmetric.transpose(-1, -2);
        grad_x1 = u.matmul(symmetric).to(x1_c.scalar_type());
        grad_x2 = difference(x2_c, x1_c, true)
                      .matmul(symmetric)
                      .to(x2_c.scalar_type());
        impl::OuterTerm<scalar_t> term{accessor(grad_output_c), accessor(x1_c),
                                       accessor(x2_c), x2_c.size(1),
                                       x1_c.size(2)};
        grad_VI = cuda_detail::pair_reduction(
                      term, batch_size, 1, x1_c.size(2) * x1_c.size(2),
                      x1_c.size(1) * x2_c.size(1), VI_c.options())
                      .reshape(VI_c.sizes())
                      .to(VI_c.scalar_type());
      }));
  if (unbatched) {
    grad_x1.squeeze_(0);
    grad_x2.squeeze_(0);
    grad_VI.squeeze_(0);
  }
  return std::make_tuple(grad_x1, grad_x2, grad_VI);
}
}  // namespace

TORCH_LIBRARY_IMPL(torchpairwise, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchpairwise::_sqmahalanobis"),
         TORCH_FN(sqmahalanobis_kernel));
  m.impl(TORCH_SELECTIVE_NAME("torchpairwise::_sqmahalanobis_backward"),
         TORCH_FN(sqmahalanobis_backward_kernel));
}
}  // namespace torchpairwise::ops
