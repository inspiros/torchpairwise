#include <ATen/ATen.h>
#include <torch/library.h>

#include "cpu_helpers.h"
#include "rel_entr.h"
#include "../utils/dispatch.h"

namespace torchpairwise {
    namespace ops {
        namespace {
            namespace impl {
                template<typename scalar_t, typename index_t>
                void _sqjensenshannon_forward_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        at::TensorAccessor<scalar_t, 3> output) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t i = (index / x2.size(1)) % x1.size(1);
                        index_t b = index / (x2.size(1) * x1.size(1));

                        scalar_t val = 0, m;
                        for (int64_t k = 0; k < x1.size(2); k++) {
                            m = (x1[b][i][k] + x2[b][j][k]) / static_cast<scalar_t>(2);
                            val += rel_entr(x1[b][i][k], m) + rel_entr(x2[b][j][k], m);
                        }
                        output[b][i][j] = val;
                    }
                }
            } // namespace impl

            at::Tensor _sqjensenshannon_forward_kernel(
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    c10::optional<double> base) {
                at::CheckedFrom c = "_sqjensenshannon_forward";
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
                int64_t n_kernels = output.numel();

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(x1.scalar_type(), "_sqjensenshannon_forward_cpu", ([&] {
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto output_accessor =
                                output.accessor<scalar_t, 3>();
                        impl::_sqjensenshannon_forward_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                output_accessor);
                        output.div_(base.has_value() ? 2 * log(static_cast<scalar_t>(base.value())) : 2);
                    }));
                }));
                if (unbatched)
                    output.squeeze_(0);
                return output;
            }

            namespace impl {
                template<typename scalar_t, typename index_t>
                void _sqjensenshannon_backward_x1_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        at::TensorAccessor<scalar_t, 3> grad_x1) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t k = index % x1.size(2);
                        index_t i = (index / x1.size(2)) % x1.size(1);
                        index_t b = index / (x1.size(2) * x1.size(1));

                        scalar_t val = 0, sum, m;
                        for (index_t j = 0; j < x2.size(1); j++) {
                            sum = x1[b][i][k] + x2[b][j][k];
                            m = sum / static_cast<scalar_t>(2);
                            if (m > static_cast<scalar_t>(0)) {
                                if (x1[b][i][k] > static_cast<scalar_t>(0)) {
                                    val += grad_output[b][i][j] *
                                           (x2[b][j][k] + sum * log(x1[b][i][k] / m)) / sum;
                                }
                                if (x2[b][j][k] > static_cast<scalar_t>(0)) {
                                    val += grad_output[b][i][j] * -x2[b][j][k] / sum;
                                }
                            }
                        }
                        grad_x1[b][i][k] += val;
                    }
                }

                template<typename scalar_t, typename index_t>
                void _sqjensenshannon_backward_x2_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        at::TensorAccessor<scalar_t, 3> grad_x2) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t k = index % x2.size(2);
                        index_t j = (index / x2.size(2)) % x2.size(1);
                        index_t b = index / (x2.size(2) * x2.size(1));

                        scalar_t val = 0, sum, m;
                        for (index_t i = 0; i < x1.size(1); i++) {
                            sum = x1[b][i][k] + x2[b][j][k];
                            m = sum / static_cast<scalar_t>(2);
                            if (m > static_cast<scalar_t>(0)) {
                                if (x1[b][i][k] > static_cast<scalar_t>(0)) {
                                    val += grad_output[b][i][j] * -x1[b][i][k] / sum;
                                }
                                if (x2[b][j][k] > static_cast<scalar_t>(0)) {
                                    val += grad_output[b][i][j] *
                                           (x1[b][i][k] + sum * log(x2[b][j][k] / m)) / sum;
                                }
                            }
                        }
                        grad_x2[b][j][k] += val;
                    }
                }
            } // namespace impl

            std::tuple<at::Tensor, at::Tensor> _sqjensenshannon_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    c10::optional<double> base) {
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

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(x1.scalar_type(), "_sqjensenshannon_backward_cpu", ([&] {
                    grad_output_c = grad_output_c.div(
                            base.has_value() ? 2 * log(static_cast<scalar_t>(base.value())) : 2);

                    n_kernels = x1_c.numel();
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_x1_accessor =
                                grad_x1.accessor<scalar_t, 3>();
                        impl::_sqjensenshannon_backward_x1_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 3>(),
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                grad_x1_accessor);
                    }));

                    n_kernels = x2_c.numel();
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_x2_accessor =
                                grad_x2.accessor<scalar_t, 3>();
                        impl::_sqjensenshannon_backward_x2_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 3>(),
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
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
                    TORCH_SELECTIVE_NAME("torchpairwise::_sqjensenshannon"),
                    TORCH_FN(_sqjensenshannon_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::__sqjensenshannon_backward"),
                    TORCH_FN(_sqjensenshannon_backward_kernel));
        }
    }
}
