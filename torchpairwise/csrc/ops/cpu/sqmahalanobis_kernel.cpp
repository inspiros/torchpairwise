#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <torch/library.h>

#include "../ppminkowski.h"
#include "../utils/dispatch.h"
#include "../utils/tiling.h"
#include "cpu_helpers.h"

namespace torchpairwise::ops {
namespace {
namespace impl {
template <typename scalar_t, typename index_t>
void _sqmahalanobis_forward_kernel_impl(
    index_t n_kernels, const at::TensorAccessor<scalar_t, 3> x1,
    const at::TensorAccessor<scalar_t, 3> x2,
    const at::TensorAccessor<scalar_t, 3> VI,
    at::TensorAccessor<scalar_t, 3> output) {
  CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
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
  at::checkAllSameType(c, args);

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

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x1_c.scalar_type(),
      "_sqmahalanobis_forward_cpu", ([&] {
        TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(
            n_kernels, CPU, ([&] {
              auto output_accessor = output.accessor<scalar_t, 3>();
              impl::_sqmahalanobis_forward_kernel_impl<scalar_t, index_t>(
                  n_kernels, x1_c.accessor<scalar_t, 3>(),
                  x2_c.accessor<scalar_t, 3>(), VI_c.accessor<scalar_t, 3>(),
                  output_accessor);
            }));
      }));
  if (unbatched)
    output.squeeze_(0);
  return output;
}

// Exact pair-difference outer products, accumulated over budgeted row
// blocks with batched matrix products.
at::Tensor outer_gradient(const at::Tensor& grad, const at::Tensor& x1,
                          const at::Tensor& x2) {
  auto acc = at::toOpMathType(x1.scalar_type());
  auto a = x1.to(acc), b = x2.to(acc), g = grad.to(acc);
  auto batch = x1.size(0), n = x1.size(1), m = x2.size(1), d = x1.size(2);
  auto result = at::zeros({batch, d, d}, a.options());
  for (int64_t i = 0, rows; i < n; i += rows) {
    rows = utils::tile_rows(m, d, n - i);
    auto delta =
        (a.narrow(1, i, rows).unsqueeze(2) - b.unsqueeze(1)).flatten(1, 2);
    auto weighted = delta * g.narrow(1, i, rows).reshape({batch, rows * m, 1});
    result.add_(weighted.transpose(1, 2).bmm(delta));
  }
  return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sqmahalanobis_backward_kernel(
    const at::Tensor& grad_output, const at::Tensor& x1, const at::Tensor& x2,
    const at::Tensor& VI) {
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

  // d/dx (x-y)^T VI (x-y) = (VI + VI^T)(x-y). The p=2 Minkowski backward
  // gives twice the exact pair-difference sums; apply the matrix once.
  auto differences =
      detail::_ppminkowski_backward(grad_output_c, x1_c, x2_c, 2.);
  auto acc = at::toOpMathType(x1_c.scalar_type());
  auto symmetric = VI_c.to(acc);
  symmetric = (symmetric + symmetric.transpose(-1, -2)).mul_(.5);
  auto grad_x1 =
      std::get<0>(differences).to(acc).matmul(symmetric).to(x1_c.scalar_type());
  auto grad_x2 =
      std::get<1>(differences).to(acc).matmul(symmetric).to(x2_c.scalar_type());
  auto grad_VI =
      outer_gradient(grad_output_c, x1_c, x2_c).to(VI_c.scalar_type());
  if (unbatched) {
    grad_x1.squeeze_(0);
    grad_x2.squeeze_(0);
    grad_VI.squeeze_(0);
  }
  return std::make_tuple(grad_x1, grad_x2, grad_VI);
}
}  // namespace

TORCH_LIBRARY_IMPL(torchpairwise, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchpairwise::_sqmahalanobis"),
         TORCH_FN(sqmahalanobis_kernel));
  m.impl(TORCH_SELECTIVE_NAME("torchpairwise::_sqmahalanobis_backward"),
         TORCH_FN(sqmahalanobis_backward_kernel));
}
}  // namespace torchpairwise::ops
