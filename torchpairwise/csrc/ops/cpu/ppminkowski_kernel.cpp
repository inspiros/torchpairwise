#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <torch/library.h>

#include "../utils/dispatch.h"
#include "cpu_helpers.h"
#include "signum.h"

namespace torchpairwise::ops {
namespace {
namespace impl {
template <int power = 0, typename scalar_t = float, typename index_t = int64_t>
void _ppminkowski_kernel_impl(index_t n_kernels,
                              const at::TensorAccessor<scalar_t, 3> x1,
                              const at::TensorAccessor<scalar_t, 3> x2,
                              scalar_t p,
                              at::TensorAccessor<scalar_t, 3> output) {
  CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
    index_t j = index % x2.size(1);
    index_t i = (index / x2.size(1)) % x1.size(1);
    index_t b = index / (x2.size(1) * x1.size(1));

    scalar_t val = 0;
    for (index_t k = 0; k < x1.size(2); k++) {
      auto delta = x1[b][i][k] - x2[b][j][k];
      if constexpr (power == 2)
        val += delta * delta;
      else if constexpr (power == 1)
        val += fabs(delta);
      else
        val += pow(fabs(delta), p);
    }
    output[b][i][j] = val;
  }
}

// Return the extremum and its multiplicity without depending on feature order.
template <bool neg, typename scalar_t, typename index_t>
std::pair<scalar_t, index_t> infinity_stats(
    const at::TensorAccessor<scalar_t, 3> x1,
    const at::TensorAccessor<scalar_t, 3> x2, index_t b, index_t i, index_t j) {
  scalar_t distance = 0;
  index_t count = 0;
  for (index_t k = 0; k < x1.size(2); ++k) {
    scalar_t candidate = fabs(x1[b][i][k] - x2[b][j][k]);
    if (std::isnan(static_cast<double>(candidate)))
      return {candidate, 0};
    if (count == 0 || (neg ? candidate < distance : candidate > distance)) {
      distance = candidate;
      count = 1;
    } else if (candidate == distance) {
      ++count;
    }
  }
  return {distance, count};
}

// Keep the extremum tie rule here: each attaining coordinate gets 1/count.
template <typename scalar_t, typename index_t>
scalar_t infinity_gradient(scalar_t delta, scalar_t distance, index_t count,
                           scalar_t grad) {
  if (count == 0 || delta == 0 || fabs(delta) != distance)
    return scalar_t(0);
  using acc_t = at::opmath_type<scalar_t>;
  return static_cast<scalar_t>(acc_t(grad) * m_signum(delta) / acc_t(count));
}

template <bool neg = false, typename scalar_t, typename index_t>
void _ppminkowski_inf_kernel_impl(index_t n_kernels,
                                  const at::TensorAccessor<scalar_t, 3> x1,
                                  const at::TensorAccessor<scalar_t, 3> x2,
                                  at::TensorAccessor<scalar_t, 3> output) {
  CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
    index_t j = index % x2.size(1);
    index_t i = (index / x2.size(1)) % x1.size(1);
    index_t b = index / (x2.size(1) * x1.size(1));
    output[b][i][j] = infinity_stats<neg>(x1, x2, b, i, j).first;
  }
}
}  // namespace impl

at::Tensor ppminkowski_kernel(const at::Tensor& x1, const at::Tensor& x2,
                              double p) {
  at::CheckedFrom c = "_ppminkowski_forward";
  auto args = {at::TensorArg(x1, "x1", 1), at::TensorArg(x2, "x2", 2)};
  at::checkAllSameType(c, args);

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
  int64_t n_kernels = batch_size * x1_c.size(1) * x2_c.size(1);
  auto output =
      at::empty({batch_size, x1_c.size(1), x2_c.size(1)}, x1_c.options());

  if (p == 0) {
    output.fill_(x1_c.size(2));
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x1_c.scalar_type(),
        "_ppminkowski_forward_cpu", ([&] {
          TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(
              n_kernels, CPU, ([&] {
                auto output_accessor = output.accessor<scalar_t, 3>();
                if (std::isinf(p)) {
                  TORCHPAIRWISE_DISPATCH_BOOL_NAME(
                      negative, p < 0, ([&] {
                        impl::_ppminkowski_inf_kernel_impl<negative, scalar_t,
                                                           index_t>(
                            n_kernels, x1_c.accessor<scalar_t, 3>(),
                            x2_c.accessor<scalar_t, 3>(), output_accessor);
                      }));
                } else {
                  auto run = [&](auto power) {
                    impl::_ppminkowski_kernel_impl<decltype(power)::value,
                                                   scalar_t, index_t>(
                        n_kernels, x1_c.accessor<scalar_t, 3>(),
                        x2_c.accessor<scalar_t, 3>(), static_cast<scalar_t>(p),
                        output_accessor);
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
  }
  if (unbatched)
    output.squeeze_(0);
  return output;
}

namespace impl {
template <typename scalar_t, typename index_t>
void _ppminkowski_backward_x1_kernel_impl(
    index_t n_kernels, const at::TensorAccessor<scalar_t, 3> grad_output,
    const at::TensorAccessor<scalar_t, 3> x1,
    const at::TensorAccessor<scalar_t, 3> x2, scalar_t p,
    at::TensorAccessor<scalar_t, 3> grad_x1) {
  scalar_t p_minus_1 = p - 1;
  CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
    index_t i = index % x1.size(1);
    index_t b = index / x1.size(1);

    scalar_t val;
    for (index_t j = 0; j < x2.size(1); j++) {
      for (index_t k = 0; k < x1.size(2); k++) {
        val = x1[b][i][k] - x2[b][j][k];
        if (val != 0)
          grad_x1[b][i][k] += grad_output[b][i][j] * p *
                              pow(fabs(val), p_minus_1) * m_signum(val);
      }
    }
  }
}

template <typename scalar_t, typename index_t>
void _ppminkowski_backward_x2_kernel_impl(
    index_t n_kernels, const at::TensorAccessor<scalar_t, 3> grad_output,
    const at::TensorAccessor<scalar_t, 3> x1,
    const at::TensorAccessor<scalar_t, 3> x2, scalar_t p,
    at::TensorAccessor<scalar_t, 3> grad_x2) {
  scalar_t p_minus_1 = p - 1;
  CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
    index_t j = index % x2.size(1);
    index_t b = index / x2.size(1);

    scalar_t val;
    for (index_t i = 0; i < x1.size(1); i++) {
      for (index_t k = 0; k < x1.size(2); k++) {
        val = x2[b][j][k] - x1[b][i][k];
        if (val != 0)
          grad_x2[b][j][k] += grad_output[b][i][j] * p *
                              pow(fabs(val), p_minus_1) * m_signum(val);
      }
    }
  }
}

template <bool neg, bool transpose, typename scalar_t, typename index_t>
void _ppminkowski_inf_backward_kernel_impl(
    index_t n_kernels, const at::TensorAccessor<scalar_t, 3> grad_output,
    const at::TensorAccessor<scalar_t, 3> own,
    const at::TensorAccessor<scalar_t, 3> other,
    at::TensorAccessor<scalar_t, 3> grad_own) {
  // Each worker owns a complete row of one input, so no updates race.
  CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
    index_t i = index % own.size(1);
    index_t b = index / own.size(1);
    for (index_t j = 0; j < other.size(1); ++j) {
      auto [distance, count] = infinity_stats<neg>(own, other, b, i, j);
      scalar_t grad = transpose ? grad_output[b][j][i] : grad_output[b][i][j];
      for (index_t k = 0; k < own.size(2); ++k) {
        scalar_t delta = own[b][i][k] - other[b][j][k];
        grad_own[b][i][k] += infinity_gradient(delta, distance, count, grad);
      }
    }
  }
}
}  // namespace impl

std::tuple<at::Tensor, at::Tensor> ppminkowski_backward_kernel(
    const at::Tensor& grad_output, const at::Tensor& x1, const at::Tensor& x2,
    double p) {
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

  if (p != 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, grad_output_c.scalar_type(),
        "_ppminkowski_backward_cpu", ([&] {
          if (std::isinf(p)) {
            TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(
                std::max(grad_output_c.numel(),
                         batch_size * std::max(x1_c.size(1), x2_c.size(1))),
                CPU, ([&] {
                  TORCHPAIRWISE_DISPATCH_BOOL_NAME(
                      negative, p < 0, ([&] {
                        auto grad_x1_accessor = grad_x1.accessor<scalar_t, 3>();
                        auto grad_x2_accessor = grad_x2.accessor<scalar_t, 3>();
                        impl::_ppminkowski_inf_backward_kernel_impl<
                            negative, false, scalar_t, index_t>(
                            batch_size * x1_c.size(1),
                            grad_output_c.accessor<scalar_t, 3>(),
                            x1_c.accessor<scalar_t, 3>(),
                            x2_c.accessor<scalar_t, 3>(), grad_x1_accessor);
                        impl::_ppminkowski_inf_backward_kernel_impl<
                            negative, true, scalar_t, index_t>(
                            batch_size * x2_c.size(1),
                            grad_output_c.accessor<scalar_t, 3>(),
                            x2_c.accessor<scalar_t, 3>(),
                            x1_c.accessor<scalar_t, 3>(), grad_x2_accessor);
                      }));
                }));
          } else {
            n_kernels = batch_size * x1_c.size(1);
            TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(
                n_kernels, CPU, ([&] {
                  auto grad_x1_accessor = grad_x1.accessor<scalar_t, 3>();
                  impl::_ppminkowski_backward_x1_kernel_impl<scalar_t, index_t>(
                      n_kernels, grad_output_c.accessor<scalar_t, 3>(),
                      x1_c.accessor<scalar_t, 3>(),
                      x2_c.accessor<scalar_t, 3>(), static_cast<scalar_t>(p),
                      grad_x1_accessor);
                }));

            n_kernels = batch_size * x2_c.size(1);
            TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(
                n_kernels, CPU, ([&] {
                  auto grad_x2_accessor = grad_x2.accessor<scalar_t, 3>();
                  impl::_ppminkowski_backward_x2_kernel_impl<scalar_t, index_t>(
                      n_kernels, grad_output_c.accessor<scalar_t, 3>(),
                      x1_c.accessor<scalar_t, 3>(),
                      x2_c.accessor<scalar_t, 3>(), static_cast<scalar_t>(p),
                      grad_x2_accessor);
                }));
          }
        }));
  }
  if (unbatched) {
    grad_x1.squeeze_(0);
    grad_x2.squeeze_(0);
  }
  return std::make_tuple(grad_x1, grad_x2);
}
}  // namespace

TORCH_LIBRARY_IMPL(torchpairwise, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchpairwise::_ppminkowski"),
         TORCH_FN(ppminkowski_kernel));
  m.impl(TORCH_SELECTIVE_NAME("torchpairwise::_ppminkowski_backward"),
         TORCH_FN(ppminkowski_backward_kernel));
}
}  // namespace torchpairwise::ops
