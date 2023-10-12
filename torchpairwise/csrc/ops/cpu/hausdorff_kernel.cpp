#include <ATen/ATen.h>
#include <torch/library.h>

#include "cpu_helpers.h"
#include "../utils/dispatch.h"
#include "../utils/scalar_type_utils.h"

namespace torchpairwise {
    namespace ops {
        namespace {
            namespace impl {
                template<typename scalar_t, typename index_t>
                void _directed_hausdorff_forward_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 4> x1,
                        const at::TensorAccessor<scalar_t, 4> x2,
                        at::TensorAccessor<scalar_t, 3> output,
                        at::TensorAccessor<int64_t, 3> x1_indices,
                        at::TensorAccessor<int64_t, 3> x2_indices) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t i = (index / x2.size(1)) % x1.size(1);
                        index_t b = index / (x2.size(1) * x1.size(1));

                        scalar_t cmax = 0, cmin, d, diff;
                        index_t m_val, n_val, m_tmp, n_tmp;
                        for (index_t m = 0; m < x1.size(2); m++) {
                            cmin = c10::CPPTypeLimits<scalar_t>::upper_bound();
                            for (index_t n = 0; n < x2.size(2); n++) {
                                d = 0;
                                for (index_t k = 0; k < x1.size(3); k++) {
                                    diff = x1[b][i][m][k] - x2[b][j][n][k];
                                    d += diff * diff;
                                }
                                if (d < cmax)
                                    break;
                                if (d < cmin) {
                                    cmin = d;
                                    m_tmp = m;
                                    n_tmp = n;
                                }
                            }
                            if (cmin >= cmax && d >= cmax) {
                                cmax = cmin;
                                m_val = m_tmp;
                                n_val = n_tmp;
                            }
                        }
                        output[b][i][j] = sqrt(cmax);
                        x1_indices[b][i][j] = m_val;
                        x2_indices[b][i][j] = n_val;
                    }
                }

                template<typename scalar_t, typename index_t>
                void _directed_hausdorff_shuffled_forward_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 4> x1,
                        const at::TensorAccessor<scalar_t, 4> x2,
                        const at::TensorAccessor<int64_t, 3> x1_perm,
                        const at::TensorAccessor<int64_t, 3> x2_perm,
                        at::TensorAccessor<scalar_t, 3> output,
                        at::TensorAccessor<int64_t, 3> x1_indices,
                        at::TensorAccessor<int64_t, 3> x2_indices) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t i = (index / x2.size(1)) % x1.size(1);
                        index_t b = index / (x2.size(1) * x1.size(1));

                        scalar_t cmax = 0, cmin, d, diff;
                        index_t m_val, n_val, m_tmp, n_tmp, m, n;
                        for (index_t m_ = 0; m_ < x1.size(2); m_++) {
                            m = x1_perm[b][i][m_];
                            cmin = c10::CPPTypeLimits<scalar_t>::upper_bound();
                            for (index_t n_ = 0; n_ < x2.size(2); n_++) {
                                n = x2_perm[b][j][n_];
                                d = 0;
                                for (index_t k = 0; k < x1.size(3); k++) {
                                    diff = x1[b][i][m][k] - x2[b][j][n][k];
                                    d += diff * diff;
                                }
                                if (d < cmax)
                                    break;
                                if (d < cmin) {
                                    cmin = d;
                                    m_tmp = m;
                                    n_tmp = n;
                                }
                            }
                            if (cmin >= cmax && d >= cmax) {
                                cmax = cmin;
                                m_val = m_tmp;
                                n_val = n_tmp;
                            }
                        }
                        output[b][i][j] = sqrt(cmax);
                        x1_indices[b][i][j] = m_val;
                        x2_indices[b][i][j] = n_val;
                    }
                }
            } // namespace impl

            std::tuple<at::Tensor, at::Tensor, at::Tensor> _directed_hausdorff_forward_kernel(
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    bool shuffle,
                    c10::optional<at::Generator> generator) {
                at::CheckedFrom c = "_directed_hausdorff_forward";
                auto args = {
                        at::TensorArg(x1, "x1", 1),
                        at::TensorArg(x2, "x2", 2)};
                at::checkAllSameType(c, args);

                bool unbatched = x1.ndimension() == 3;
                TORCH_CHECK(unbatched || x1.ndimension() == 4,
                            "x1 must be 3-D (unbatched) or 4-D (batched) tensor.")
                TORCH_CHECK(unbatched || x2.ndimension() == 4,
                            "x2 must be 3-D (unbatched) or 4-D (batched) tensor.")
                TORCH_CHECK(unbatched || (x1.size(0) == x2.size(0)),
                            "batch_size of x1 and x2 do not match.")
                TORCH_CHECK((unbatched && x1.size(2) == x2.size(2)) ||
                            (!unbatched && x1.size(3) == x2.size(3)),
                            "feature dimension of x1 and x2 do not match.")

                auto x1_c = x1.contiguous();
                auto x2_c = x2.contiguous();
                if (unbatched) {
                    x1_c = x1_c.unsqueeze(0);
                    x2_c = x2_c.unsqueeze(0);
                }

                int64_t batch_size = x1_c.size(0);
                auto output = at::empty({batch_size, x1_c.size(1), x2_c.size(1)},
                                        x1.options());
                auto x1_indices = at::empty({batch_size, x1_c.size(1), x2_c.size(1)},
                                            x1.options().dtype(at::kLong));
                auto x2_indices = at::empty({batch_size, x1_c.size(1), x2_c.size(1)},
                                            x1.options().dtype(at::kLong));
                int64_t n_kernels = output.numel();

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(x1.scalar_type(), "_directed_hausdorff_forward_cpu", ([&] {
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto output_accessor =
                                output.accessor<scalar_t, 3>();
                        auto x1_indices_accessor =
                                x1_indices.accessor<int64_t, 3>();
                        auto x2_indices_accessor =
                                x2_indices.accessor<int64_t, 3>();
                        if (shuffle) {
                            auto x1_perm = at::empty({batch_size, x1_c.size(2)}, x1_c.options().dtype(at::kLong));
                            auto x2_perm = at::empty({batch_size, x2_c.size(2)}, x2_c.options().dtype(at::kLong));
                            for (const auto b: c10::irange(batch_size)) {
                                auto x1_perm_b = x1_perm[b], x2_perm_b = x2_perm[b];
                                at::randperm_outf(x1_c.size(2), generator, x1_perm_b);
                                at::randperm_outf(x2_c.size(2), generator, x2_perm_b);
                            }
                            x1_perm = x1_perm.view({batch_size, 1, x1_c.size(2)}).expand({-1, x1_c.size(1), -1});
                            x2_perm = x2_perm.view({batch_size, 1, x2_c.size(2)}).expand({-1, x2_c.size(1), -1});
                            impl::_directed_hausdorff_shuffled_forward_kernel_impl<scalar_t, index_t>(
                                    n_kernels,
                                    x1_c.accessor<scalar_t, 4>(),
                                    x2_c.accessor<scalar_t, 4>(),
                                    x1_perm.accessor<int64_t, 3>(),
                                    x2_perm.accessor<int64_t, 3>(),
                                    output_accessor,
                                    x1_indices_accessor,
                                    x2_indices_accessor);
                        } else {
                            impl::_directed_hausdorff_forward_kernel_impl<scalar_t, index_t>(
                                    n_kernels,
                                    x1_c.accessor<scalar_t, 4>(),
                                    x2_c.accessor<scalar_t, 4>(),
                                    output_accessor,
                                    x1_indices_accessor,
                                    x2_indices_accessor);
                        }
                    }));
                }));
                if (unbatched) {
                    output.squeeze_(0);
                    x1_indices.squeeze_(0);
                    x2_indices.squeeze_(0);
                }
                return std::make_tuple(output, x1_indices, x2_indices);
            }

            namespace impl {
                template<typename scalar_t, typename index_t>
                void _directed_hausdorff_backward_x1_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> output,
                        const at::TensorAccessor<int64_t, 3> x1_indices,
                        const at::TensorAccessor<int64_t, 3> x2_indices,
                        const at::TensorAccessor<scalar_t, 4> x1,
                        const at::TensorAccessor<scalar_t, 4> x2,
                        at::TensorAccessor<scalar_t, 4> grad_x1) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t k = index % x1.size(3);
                        index_t i = (index / x1.size(3)) % x1.size(1);
                        index_t b = index / (x1.size(3) * x1.size(1));

                        scalar_t diff;
                        for (index_t j = 0; j < x2.size(1); j++) {
                            index_t m_val = x1_indices[b][i][j];
                            index_t n_val = x2_indices[b][i][j];
                            diff = x1[b][i][m_val][k] - x2[b][j][n_val][k];
                            grad_x1[b][i][m_val][k] += grad_output[b][i][j] * diff / output[b][i][j];
                        }
                    }
                }

                template<typename scalar_t, typename index_t>
                void _directed_hausdorff_backward_x2_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> output,
                        const at::TensorAccessor<int64_t, 3> x1_indices,
                        const at::TensorAccessor<int64_t, 3> x2_indices,
                        const at::TensorAccessor<scalar_t, 4> x1,
                        const at::TensorAccessor<scalar_t, 4> x2,
                        at::TensorAccessor<scalar_t, 4> grad_x2) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t k = index % x1.size(3);
                        index_t j = (index / x1.size(3)) % x2.size(1);
                        index_t b = index / (x1.size(3) * x2.size(1));

                        scalar_t diff;
                        for (index_t i = 0; i < x1.size(1); i++) {
                            index_t m_val = x1_indices[b][i][j];
                            index_t n_val = x2_indices[b][i][j];
                            diff = x2[b][j][n_val][k] - x1[b][i][m_val][k];
                            grad_x2[b][j][n_val][k] += grad_output[b][i][j] * diff / output[b][i][j];
                        }
                    }
                }
            } // namespace impl

            std::tuple<at::Tensor, at::Tensor> _directed_hausdorff_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    bool shuffle,
                    c10::optional<at::Generator> generator) {
                bool unbatched = x1.ndimension() == 3;

                auto grad_output_c = grad_output.contiguous();
                auto x1_c = x1.contiguous();
                auto x2_c = x2.contiguous();
                if (unbatched) {
                    grad_output_c = grad_output_c.unsqueeze(0);
                    x1_c = x1_c.unsqueeze(0);
                    x2_c = x2_c.unsqueeze(0);
                }

                auto grad_x1 = at::zeros_like(x1_c);
                auto grad_x2 = at::zeros_like(x2_c);


                int64_t batch_size = x1_c.size(0);
                auto output = at::empty({batch_size, x1_c.size(1), x2_c.size(1)},
                                        x1.options());
                auto x1_indices = at::empty({batch_size, x1_c.size(1), x2_c.size(1)},
                                            x1.options().dtype(at::kLong));
                auto x2_indices = at::empty({batch_size, x1_c.size(1), x2_c.size(1)},
                                            x1.options().dtype(at::kLong));
                int64_t n_kernels;

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(x1.scalar_type(), "_directed_hausdorff_backward_cpu", ([&] {
                    n_kernels = output.numel();
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto output_accessor =
                                output.accessor<scalar_t, 3>();
                        auto x1_indices_accessor =
                                x1_indices.accessor<int64_t, 3>();
                        auto x2_indices_accessor =
                                x2_indices.accessor<int64_t, 3>();
                        if (shuffle) {
                            auto x1_perm = at::empty({batch_size, x1_c.size(2)}, x1_c.options().dtype(at::kLong));
                            auto x2_perm = at::empty({batch_size, x2_c.size(2)}, x2_c.options().dtype(at::kLong));
                            for (const auto b: c10::irange(batch_size)) {
                                auto x1_perm_b = x1_perm[b], x2_perm_b = x2_perm[b];
                                at::randperm_outf(x1_c.size(2), generator, x1_perm_b);
                                at::randperm_outf(x2_c.size(2), generator, x2_perm_b);
                            }
                            x1_perm = x1_perm.view({batch_size, 1, x1_c.size(2)}).expand({-1, x1_c.size(1), -1});
                            x2_perm = x2_perm.view({batch_size, 1, x2_c.size(2)}).expand({-1, x2_c.size(1), -1});
                            impl::_directed_hausdorff_shuffled_forward_kernel_impl<scalar_t, index_t>(
                                    n_kernels,
                                    x1_c.accessor<scalar_t, 4>(),
                                    x2_c.accessor<scalar_t, 4>(),
                                    x1_perm.accessor<int64_t, 3>(),
                                    x2_perm.accessor<int64_t, 3>(),
                                    output_accessor,
                                    x1_indices_accessor,
                                    x2_indices_accessor);
                        } else {
                            impl::_directed_hausdorff_forward_kernel_impl<scalar_t, index_t>(
                                    n_kernels,
                                    x1_c.accessor<scalar_t, 4>(),
                                    x2_c.accessor<scalar_t, 4>(),
                                    output_accessor,
                                    x1_indices_accessor,
                                    x2_indices_accessor);
                        }
                    }));

                    n_kernels = batch_size * x1_c.size(1) * x1_c.size(3);
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_x1_accessor =
                                grad_x1.accessor<scalar_t, 4>();
                        impl::_directed_hausdorff_backward_x1_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 3>(),
                                output.accessor<scalar_t, 3>(),
                                x1_indices.accessor<int64_t, 3>(),
                                x2_indices.accessor<int64_t, 3>(),
                                x1_c.accessor<scalar_t, 4>(),
                                x2_c.accessor<scalar_t, 4>(),
                                grad_x1_accessor);
                    }));

                    n_kernels = batch_size * x2_c.size(1) * x1_c.size(3);
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_x2_accessor =
                                grad_x2.accessor<scalar_t, 4>();
                        impl::_directed_hausdorff_backward_x2_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 3>(),
                                output.accessor<scalar_t, 3>(),
                                x1_indices.accessor<int64_t, 3>(),
                                x2_indices.accessor<int64_t, 3>(),
                                x1_c.accessor<scalar_t, 4>(),
                                x2_c.accessor<scalar_t, 4>(),
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
                    TORCH_SELECTIVE_NAME("torchpairwise::_directed_hausdorff"),
                    TORCH_FN(_directed_hausdorff_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::__directed_hausdorff_backward"),
                    TORCH_FN(_directed_hausdorff_backward_kernel));
        }
    }
}
