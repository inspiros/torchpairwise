#include <ATen/ATen.h>
#include <torch/library.h>

#include "cpu_helpers.h"
#include "signum.h"
#include "../utils/dispatch.h"

namespace torchpairwise {
    namespace ops {
        namespace {
            namespace impl {
                template<typename scalar_t, typename index_t>
                void _braycurtis_forward_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        at::TensorAccessor<scalar_t, 3> output) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t i = (index / x2.size(1)) % x1.size(1);
                        index_t b = index / (x2.size(1) * x1.size(1));

                        scalar_t diff = 0, sum = 0;
                        for (index_t k = 0; k < x1.size(2); k++) {
                            diff += fabs(x1[b][i][k] - x2[b][j][k]);
                            sum += fabs(x1[b][i][k] + x2[b][j][k]);
                        }
                        output[b][i][j] = diff / sum;
                    }
                }
            } // namespace impl

            at::Tensor _braycurtis_forward_kernel(
                    const at::Tensor &x1,
                    const at::Tensor &x2) {
                at::CheckedFrom c = "_braycurtis_forward";
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

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(x1.scalar_type(), "_braycurtis_forward_cpu", ([&] {
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto output_accessor =
                                output.accessor<scalar_t, 3>();
                        impl::_braycurtis_forward_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                output_accessor);
                    }));
                }));
                if (unbatched)
                    output.squeeze_(0);
                return output;
            }

            namespace impl {
                template<typename scalar_t, typename index_t>
                void _braycurtis_forward_components_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        at::TensorAccessor<scalar_t, 3> sd,
                        at::TensorAccessor<scalar_t, 3> ss) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t i = (index / x2.size(1)) % x1.size(1);
                        index_t b = index / (x2.size(1) * x1.size(1));

                        scalar_t diff = 0, sum = 0;
                        for (index_t k = 0; k < x1.size(2); k++) {
                            diff += fabs(x1[b][i][k] - x2[b][j][k]);
                            sum += fabs(x1[b][i][k] + x2[b][j][k]);
                        }
                        sd[b][i][j] = diff;
                        ss[b][i][j] = sum;
                    }
                }

                template<typename scalar_t, typename index_t>
                void _braycurtis_backward_x1_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> sd,
                        const at::TensorAccessor<scalar_t, 3> ss,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        at::TensorAccessor<scalar_t, 3> grad_x1) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t k = index % x1.size(2);
                        index_t i = (index / x1.size(2)) % x1.size(1);
                        index_t b = index / (x1.size(2) * x1.size(1));

                        scalar_t val = 0, diff, sum;
                        for (index_t j = 0; j < x2.size(1); j++) {
                            diff = x1[b][i][k] - x2[b][j][k];
                            sum = x1[b][i][k] + x2[b][j][k];
                            val += grad_output[b][i][j] *
                                    (m_signum(diff) / ss[b][i][j] -
                                    m_signum(sum) * sd[b][i][j] / ss[b][i][j] / ss[b][i][j]);
                        }
                        grad_x1[b][i][k] = val;
                    }
                }

                template<typename scalar_t, typename index_t>
                void _braycurtis_backward_x2_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> sd,
                        const at::TensorAccessor<scalar_t, 3> ss,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        at::TensorAccessor<scalar_t, 3> grad_x2) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t k = index % x2.size(2);
                        index_t j = (index / x2.size(2)) % x2.size(1);
                        index_t b = index / (x2.size(2) * x2.size(1));

                        scalar_t val = 0, diff, sum;
                        for (index_t i = 0; i < x1.size(1); i++) {
                            diff = x1[b][i][k] - x2[b][j][k];
                            sum = x1[b][i][k] + x2[b][j][k];
                            val += grad_output[b][i][j] *
                                   (-m_signum(diff) / ss[b][i][j] -
                                    m_signum(sum) * sd[b][i][j] / ss[b][i][j] / ss[b][i][j]);
                        }
                        grad_x2[b][j][k] = val;
                    }
                }
            } // namespace impl

            std::tuple<at::Tensor, at::Tensor> _braycurtis_backward_kernel(
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
                auto sd = at::empty_like(grad_output_c);
                auto ss = at::empty_like(grad_output_c);

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(x1.scalar_type(), "_braycurtis_backward_cpu", ([&] {
                    n_kernels = grad_output_c.numel();
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto sd_accessor =
                                sd.accessor<scalar_t, 3>();
                        auto ss_accessor =
                                ss.accessor<scalar_t, 3>();
                        impl::_braycurtis_forward_components_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                sd_accessor,
                                ss_accessor);
                    }));

                    n_kernels = x1_c.numel();
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_x1_accessor =
                                grad_x1.accessor<scalar_t, 3>();
                        impl::_braycurtis_backward_x1_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 3>(),
                                sd.accessor<scalar_t, 3>(),
                                ss.accessor<scalar_t, 3>(),
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                grad_x1_accessor);
                    }));

                    n_kernels = x2_c.numel();
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_x2_accessor =
                                grad_x2.accessor<scalar_t, 3>();
                        impl::_braycurtis_backward_x2_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 3>(),
                                sd.accessor<scalar_t, 3>(),
                                ss.accessor<scalar_t, 3>(),
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
                    TORCH_SELECTIVE_NAME("torchpairwise::_braycurtis"),
                    TORCH_FN(_braycurtis_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::__braycurtis_backward"),
                    TORCH_FN(_braycurtis_backward_kernel));
        }
    }
}
