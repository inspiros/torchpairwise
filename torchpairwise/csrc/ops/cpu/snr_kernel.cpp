#include <ATen/ATen.h>
#include <torch/library.h>

#include "cpu_helpers.h"
#include "../utils/dispatch.h"

namespace torchpairwise {
    namespace ops {
        namespace {
            namespace impl {
                template<typename scalar_t, typename index_t>
                void _snr_forward_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        const at::TensorAccessor<scalar_t, 2> x1_var,
                        at::TensorAccessor<scalar_t, 3> output) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
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
                        output[b][i][j] = x1_var[b][i] / diff_var;
                    }
                }

                template<typename scalar_t, typename index_t>
                void _snr_forward_diff_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        at::TensorAccessor<scalar_t, 3> diff_mean,
                        at::TensorAccessor<scalar_t, 3> diff_var) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
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
            } // namespace impl

            at::Tensor _snr_forward_kernel(
                    const at::Tensor &x1,
                    const at::Tensor &x2) {
                at::CheckedFrom c = "_snr_forward";
                auto args = {
                        at::TensorArg(x1, "x1", 1),
                        at::TensorArg(x2, "x2", 2)};
                at::checkAllSameType(c, args);

                bool unbatched = x1.ndimension() == 2;
                TORCH_CHECK(unbatched || x1.ndimension() == 3,
                            "x1 must be 2-D (unbatched) or 3-D (batched) tensor.")
                TORCH_CHECK(unbatched || x2.ndimension() == 3,
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
                auto output = at::empty({batch_size, x1_c.size(1), x2_c.size(1)}, x1.options());
                auto x1_var = at::var(x1_c, 2, 0).mul_(x1_c.size(2));
                int64_t n_kernels = output.numel();

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(x1.scalar_type(), "_snr_forward_cpu", ([&] {
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto output_accessor =
                                output.accessor<scalar_t, 3>();
                        impl::_snr_forward_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                x1_var.accessor<scalar_t, 2>(),
                                output_accessor);
                    }));
                }));
                if (unbatched)
                    output.squeeze_(0);
                return output;
            }

            namespace impl {
                template<typename scalar_t, typename index_t>
                void _snr_backward_x1_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        const at::TensorAccessor<scalar_t, 2> x1_mean,
                        const at::TensorAccessor<scalar_t, 2> x1_var,
                        const at::TensorAccessor<scalar_t, 3> diff_mean,
                        const at::TensorAccessor<scalar_t, 3> diff_var,
                        at::TensorAccessor<scalar_t, 3> grad_x1) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t i = index % x1.size(1);
                        index_t b = index / x1.size(1);

                        for (index_t j = 0; j < x2.size(1); j++) {
                            for (index_t k = 0; k < x1.size(2); k++) {
                                grad_x1[b][i][k] += grad_output[b][i][j] * 2 * (
                                        (-x1_mean[b][i] + x1[b][i][k]) +
                                        (-diff_mean[b][i][j] + x2[b][j][k] - x1[b][i][k]) *
                                        x1_var[b][i] / diff_var[b][i][j]
                                ) / diff_var[b][i][j];
                            }
                        }
                    }
                }

                template<typename scalar_t, typename index_t>
                void _snr_backward_x2_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        const at::TensorAccessor<scalar_t, 2> x1_var,
                        const at::TensorAccessor<scalar_t, 3> diff_mean,
                        const at::TensorAccessor<scalar_t, 3> diff_var,
                        at::TensorAccessor<scalar_t, 3> grad_x2) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t b = index / x2.size(1);

                        for (index_t i = 0; i < x1.size(1); i++) {
                            for (index_t k = 0; k < x1.size(2); k++) {
                                grad_x2[b][j][k] += grad_output[b][i][j] * 2 *
                                                    (-diff_mean[b][i][j] + x2[b][j][k] - x1[b][i][k]) *
                                                    -x1_var[b][i] / (diff_var[b][i][j] * diff_var[b][i][j]);
                            }
                        }
                    }
                }
            } // namespace impl

            std::tuple<at::Tensor, at::Tensor> _snr_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &x1,
                    const at::Tensor &x2) {
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

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(x1.scalar_type(), "_snr_backward_cpu", ([&] {
                    n_kernels = grad_output_c.numel();
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto diff_mean_accessor =
                                diff_mean.accessor<scalar_t, 3>();
                        auto diff_var_accessor =
                                diff_var.accessor<scalar_t, 3>();
                        impl::_snr_forward_diff_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                diff_mean_accessor,
                                diff_var_accessor);
                    }));

                    n_kernels = x1_c.size(0) * x1_c.size(1);
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_x1_accessor =
                                grad_x1.accessor<scalar_t, 3>();
                        impl::_snr_backward_x1_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 3>(),
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                x1_mean.accessor<scalar_t, 2>(),
                                x1_var.accessor<scalar_t, 2>(),
                                diff_mean.accessor<scalar_t, 3>(),
                                diff_var.accessor<scalar_t, 3>(),
                                grad_x1_accessor);
                    }));

                    n_kernels = x2_c.size(0) * x2_c.size(1);
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_x2_accessor =
                                grad_x2.accessor<scalar_t, 3>();
                        impl::_snr_backward_x2_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 3>(),
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                x1_var.accessor<scalar_t, 2>(),
                                diff_mean.accessor<scalar_t, 3>(),
                                diff_var.accessor<scalar_t, 3>(),
                                grad_x2_accessor);
                    }));
                }));
                if (unbatched) {
                    grad_x1.squeeze_(0);
                    grad_x2.squeeze_(0);
                }
                return std::make_tuple(grad_x1, grad_x2);
            }
        }

        TORCH_LIBRARY_IMPL(torchpairwise, CPU, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_snr"),
                    TORCH_FN(_snr_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::__snr_backward"),
                    TORCH_FN(_snr_backward_kernel));
        }
    }
}
