#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise {
    namespace ops {
        namespace utils {
            template<bool is_boolean_metric = false, bool is_2d_metric = false>
            static C10_ALWAYS_INLINE std::pair<at::Tensor, at::Tensor> check_pairwise_inputs(
                    const at::CheckedFrom &c,
                    const at::Tensor &x1,
                    const c10::optional<at::Tensor> &x2 = c10::nullopt) {
                bool unbatched = x1.ndimension() == (is_2d_metric ? 3 : 2);
                if constexpr (is_2d_metric) {
                    TORCH_CHECK(unbatched || x1.ndimension() == 4,
                                "x1 must be 3D tensor (unbatched) or 4D tensor (batched)")
                } else {
                    TORCH_CHECK(unbatched || x1.ndimension() == 3,
                                "x1 must be 2D tensor (unbatched) or 3D tensor (batched)")
                }
                if (x2.has_value()) {
                    auto x2_ = x2.value();
                    if constexpr (is_2d_metric) {
                        TORCH_CHECK(unbatched || x2_.ndimension() == 4,
                                    "x2 must be 3D tensor (unbatched) or 4D tensor (batched)")
                        TORCH_CHECK((unbatched && x1.size(2) == x2_.size(2)) ||
                                    (!unbatched && x1.size(3) == x2_.size(3)),
                                    "x1 and x2 must have same number of features. Got: x1.size(",
                                    unbatched ? 2 : 3,
                                    ")=",
                                    x1.size(unbatched ? 2 : 3),
                                    ", x2.size(",
                                    unbatched ? 2 : 3, ")=",
                                    x2_.size(unbatched ? 2 : 3))
                    } else {
                        TORCH_CHECK(unbatched || x2_.ndimension() == 3,
                                    "x2 must be 2D tensor (unbatched) or 3D tensor (batched)")
                        TORCH_CHECK((unbatched && x1.size(1) == x2_.size(1)) ||
                                    (!unbatched && x1.size(2) == x2_.size(2)),
                                    "x1 and x2 must have same number of features. Got: x1.size(",
                                    unbatched ? 1 : 2,
                                    ")=",
                                    x1.size(unbatched ? 1 : 2),
                                    ", x2.size(",
                                    unbatched ? 1 : 2, ")=",
                                    x2_.size(unbatched ? 1 : 2))
                    }
                    if constexpr (is_boolean_metric) {
                        if (x1.scalar_type() != at::kBool || x2_.scalar_type() != at::kBool) {
                            TORCH_WARN("Data was converted to ", at::kBool, " for metric ", c)
                        }
                        return std::make_pair(x1.to(at::kBool), x2_.to(at::kBool));
                    } else
                        return std::make_pair(x1, x2_);
                } else {
                    if constexpr (is_boolean_metric) {
                        if (x1.scalar_type() != at::kBool) {
                            TORCH_WARN("Data was converted to ", at::kBool, " for metric ", c)
                        }
                        auto x1_ = x1.to(at::kBool);
                        return std::make_pair(x1_, x1_);
                    } else
                        return std::make_pair(x1, x1);
                }
            }
        } // namespace utils

        // ~~~~~ functors ~~~~~
        // Note: these functors do not follow torch's native ops convention
        // sklearn
        struct TORCHPAIRWISE_API euclidean_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::euclidean_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "euclidean_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API haversine_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::haversine_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "haversine_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API manhattan_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::manhattan_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "manhattan_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API cosine_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::cosine_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "cosine_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API linear_kernel_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::linear_kernel")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "linear_kernel(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API polynomial_kernel_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &, int64_t, c10::optional<double>, double);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::polynomial_kernel")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "polynomial_kernel(Tensor x1, Tensor? x2=None, int degree=3, float? gamma=None, float coef0=1) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2, int64_t degree, c10::optional<double> gamma, double coef0);
        };

        struct TORCHPAIRWISE_API sigmoid_kernel_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &, c10::optional<double>, double);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::sigmoid_kernel")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "sigmoid_kernel(Tensor x1, Tensor? x2=None, float? gamma=None, float coef0=1) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2, c10::optional<double> gamma, double coef0);
        };

        struct TORCHPAIRWISE_API rbf_kernel_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &, c10::optional<double>);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::rbf_kernel")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "rbf_kernel(Tensor x1, Tensor? x2=None, float? gamma=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2, c10::optional<double> gamma);
        };

        struct TORCHPAIRWISE_API laplacian_kernel_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &, c10::optional<double>);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::laplacian_kernel")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "laplacian_kernel(Tensor x1, Tensor? x2=None, float? gamma=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2, c10::optional<double> gamma);
        };

        struct TORCHPAIRWISE_API cosine_similarity_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::cosine_similarity")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "cosine_similarity(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API additive_chi2_kernel_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::additive_chi2_kernel")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "additive_chi2_kernel(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API chi2_kernel_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &, double);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::chi2_kernel")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "chi2_kernel(Tensor x1, Tensor? x2=None, float gamma=1.0) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2, double gamma);
        };

        // scipy
        struct TORCHPAIRWISE_API directed_hausdorff_distances_functor {
            using schema = std::tuple<at::Tensor, at::Tensor, at::Tensor> (const at::Tensor &, const c10::optional<at::Tensor> &, bool, c10::optional<at::Generator>);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::directed_hausdorff_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "directed_hausdorff_distances(Tensor x1, Tensor? x2=None, *, bool shuffle=False, Generator? generator=None) -> (Tensor output, Tensor x1_indices, Tensor x2_indices)")
            static std::tuple<at::Tensor, at::Tensor, at::Tensor> call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2, bool shuffle, c10::optional<at::Generator> generator);
        };

        struct TORCHPAIRWISE_API minkowski_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &, double);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::minkowski_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "minkowski_distances(Tensor x1, Tensor? x2=None, float p=2) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2, double p);
        };

        struct TORCHPAIRWISE_API wminkowski_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &, double, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::wminkowski_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "wminkowski_distances(Tensor x1, Tensor? x2=None, float p=2, Tensor? w=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2, double p, const c10::optional<at::Tensor> &w);
        };

        struct TORCHPAIRWISE_API sqeuclidean_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::sqeuclidean_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "sqeuclidean_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API correlation_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::correlation_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "correlation_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API hamming_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::hamming_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "hamming_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API jaccard_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::jaccard_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "jaccard_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API kulsinski_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::kulsinski_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "kulsinski_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API kulczynski1_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::kulczynski1_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "kulczynski1_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API seuclidean_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::seuclidean_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "seuclidean_distances(Tensor x1, Tensor? x2=None, Tensor? V=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2, const c10::optional<at::Tensor> &V);
        };

        struct TORCHPAIRWISE_API cityblock_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::cityblock_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "cityblock_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API mahalanobis_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::mahalanobis_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "mahalanobis_distances(Tensor x1, Tensor? x2=None, Tensor? VI=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2, const c10::optional<at::Tensor> &VI);
        };

        struct TORCHPAIRWISE_API chebyshev_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::chebyshev_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "chebyshev_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API braycurtis_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::braycurtis_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "braycurtis_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API canberra_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::canberra_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "canberra_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API jensenshannon_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &, c10::optional<double>);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::jensenshannon_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "jensenshannon_distances(Tensor x1, Tensor? x2=None, float? base=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2, c10::optional<double> base);
        };

        struct TORCHPAIRWISE_API yule_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::yule_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "yule_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API dice_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::dice_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "dice_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API rogerstanimoto_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::rogerstanimoto_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "rogerstanimoto_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API russellrao_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::russellrao_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "russellrao_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API sokalmichener_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::sokalmichener_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "sokalmichener_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API sokalsneath_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::sokalsneath_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "sokalsneath_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        // others
        struct TORCHPAIRWISE_API snr_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::snr_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "snr_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        // aliases
        struct TORCHPAIRWISE_API l1_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::l1_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "l1_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API l2_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::l2_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "l2_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        struct TORCHPAIRWISE_API lp_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &, double);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::lp_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "lp_distances(Tensor x1, Tensor? x2=None, float p=2) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2, double p);
        };

        struct TORCHPAIRWISE_API linf_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::linf_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "linf_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };

        // ~~~~~ functions ~~~~~
        // sklearn
        inline at::Tensor euclidean_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return euclidean_distances_functor::call(x1, x2);
        }

        inline at::Tensor haversine_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return haversine_distances_functor::call(x1, x2);
        }

        inline at::Tensor manhattan_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return manhattan_distances_functor::call(x1, x2);
        }

        inline at::Tensor cosine_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return cosine_distances_functor::call(x1, x2);
        }

        inline at::Tensor linear_kernel(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return linear_kernel_functor::call(x1, x2);
        }

        inline at::Tensor polynomial_kernel(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                int64_t degree = 3,
                c10::optional<double> gamma = c10::nullopt,
                double coef0 = 1) {
            return polynomial_kernel_functor::call(x1, x2, degree, gamma, coef0);
        }

        inline at::Tensor sigmoid_kernel(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                c10::optional<double> gamma = c10::nullopt,
                double coef0 = 1) {
            return sigmoid_kernel_functor::call(x1, x2, gamma, coef0);
        }

        inline at::Tensor rbf_kernel(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                c10::optional<double> gamma = c10::nullopt) {
            return rbf_kernel_functor::call(x1, x2, gamma);
        }

        inline at::Tensor laplacian_kernel(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                c10::optional<double> gamma = c10::nullopt) {
            return laplacian_kernel_functor::call(x1, x2, gamma);
        }

        inline at::Tensor cosine_similarity(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return cosine_similarity_functor::call(x1, x2);
        }

        inline at::Tensor additive_chi2_kernel(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return additive_chi2_kernel_functor::call(x1, x2);
        }

        inline at::Tensor chi2_kernel(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                double gamma = 1.0) {
            return chi2_kernel_functor::call(x1, x2, gamma);
        }

        // scipy
        inline std::tuple<at::Tensor, at::Tensor, at::Tensor> directed_hausdorff_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                bool shuffle = false,
                c10::optional<at::Generator> generator = c10::nullopt) {
            return directed_hausdorff_distances_functor::call(x1, x2, shuffle, generator);
        }

        inline at::Tensor minkowski_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                double p = 2) {
            return minkowski_distances_functor::call(x1, x2, p);
        }

        inline at::Tensor wminkowski_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                double p = 2,
                const c10::optional<at::Tensor> &w = c10::nullopt) {
            return wminkowski_distances_functor::call(x1, x2, p, w);
        }

        inline at::Tensor sqeuclidean_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return sqeuclidean_distances_functor::call(x1, x2);
        }

        inline at::Tensor correlation_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return correlation_distances_functor::call(x1, x2);
        }

        inline at::Tensor hamming_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return hamming_distances_functor::call(x1, x2);
        }

        inline at::Tensor jaccard_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return jaccard_distances_functor::call(x1, x2);
        }

        inline at::Tensor kulsinski_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return kulczynski1_distances_functor::call(x1, x2);
        }

        inline at::Tensor kulczynski1_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return kulczynski1_distances_functor::call(x1, x2);
        }

        inline at::Tensor seuclidean_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                const c10::optional<at::Tensor> &V = c10::nullopt) {
            return seuclidean_distances_functor::call(x1, x2, V);
        }

        inline at::Tensor cityblock_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return cityblock_distances_functor::call(x1, x2);
        }

        inline at::Tensor mahalanobis_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                const c10::optional<at::Tensor> &VI = c10::nullopt) {
            return mahalanobis_distances_functor::call(x1, x2, VI);
        }

        inline at::Tensor chebyshev_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return chebyshev_distances_functor::call(x1, x2);
        }

        inline at::Tensor braycurtis_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return braycurtis_distances_functor::call(x1, x2);
        }

        inline at::Tensor canberra_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return canberra_distances_functor::call(x1, x2);
        }

        inline at::Tensor jensenshannon_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                c10::optional<double> base = c10::nullopt) {
            return jensenshannon_distances_functor::call(x1, x2, base);
        }

        inline at::Tensor yule_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return yule_distances_functor::call(x1, x2);
        }

        inline at::Tensor dice_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return dice_distances_functor::call(x1, x2);
        }

        inline at::Tensor rogerstanimoto_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return rogerstanimoto_distances_functor::call(x1, x2);
        }

        inline at::Tensor russellrao_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return russellrao_distances_functor::call(x1, x2);
        }

        inline at::Tensor sokalmichener_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return sokalmichener_distances_functor::call(x1, x2);
        }

        inline at::Tensor sokalsneath_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return sokalsneath_distances_functor::call(x1, x2);
        }

        // others
        inline at::Tensor snr_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return snr_distances_functor::call(x1, x2);
        }

        // aliases
        inline at::Tensor l1_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return l1_distances_functor::call(x1, x2);
        }

        inline at::Tensor l2_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return l2_distances_functor::call(x1, x2);
        }

        inline at::Tensor lp_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                double p = 2) {
            return lp_distances_functor::call(x1, x2, p);
        }

        inline at::Tensor linf_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return linf_distances_functor::call(x1, x2);
        }
    }
}
