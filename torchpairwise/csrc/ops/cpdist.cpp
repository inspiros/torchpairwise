#include "cpdist.h"

#include <torch/library.h>

#include "pairwise_metrics.h"

// TODO: the design of this file is total garbage,
//  but I don't know how to do it properly atm.
#define DEFINE_CPDIST_METRIC_SPEC(NAME, HAS_w, HAS_V, HAS_VI, HAS_p, HAS_base, HAS_shuffle, HAS_generator) \
struct cpdist_##NAME##_spec {                                                                              \
    STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, #NAME)                                                  \
    using cdist_functor = NAME##_distances_functor;                                                        \
    using pdist_functor = nullptr_t;                                                                       \
    static constexpr cdist_functor::ptr_schema cdist_func = NAME##_distances;                              \
    static constexpr nullptr_t pdist_func = nullptr;                                                       \
    struct has_arg {                                                                                       \
        static constexpr bool w = HAS_w;                                                                   \
        static constexpr bool V = HAS_V;                                                                   \
        static constexpr bool VI = HAS_VI;                                                                 \
        static constexpr bool p = HAS_p;                                                                   \
        static constexpr bool base = HAS_base;                                                             \
        static constexpr bool shuffle = HAS_shuffle;                                                       \
        static constexpr bool generator = HAS_generator;                                                   \
    };                                                                                                     \
};

namespace torchpairwise {
    namespace ops {
        namespace detail {
            template<typename T>
            inline bool called_with(const c10::optional<T> &arg) {
                return arg.has_value();
            }

            template<bool cdist, typename metric>
            inline void check_extra_args(
                    const c10::optional<at::Tensor> &w,
                    const c10::optional<at::Tensor> &V,
                    const c10::optional<at::Tensor> &VI,
                    c10::optional<double> p,
                    c10::optional<double> base,
                    c10::optional<bool> shuffle,
                    c10::optional<at::Generator> generator) {
                std::vector<std::string> incompatible_args;
#define CHECK_CALLED_WITH(ARG) \
    if (called_with(ARG) && !metric::has_arg::ARG) incompatible_args.emplace_back(#ARG);
                CHECK_CALLED_WITH(w)
                CHECK_CALLED_WITH(V)
                CHECK_CALLED_WITH(VI)
                CHECK_CALLED_WITH(p)
                CHECK_CALLED_WITH(base)
                CHECK_CALLED_WITH(shuffle)
                CHECK_CALLED_WITH(generator)
                if constexpr (cdist) {
                    TORCH_CHECK_TYPE(
                            incompatible_args.empty(),
                            metric::cdist_functor::name,
                            " was called with incompatible arguments ",
                            std::accumulate(
                                    std::begin(incompatible_args), std::end(incompatible_args), std::string(),
                                    [](const std::string &ss, std::string _arg) {
                                        return ss.empty() ? _arg : ss + ", " + _arg;
                                    }),
                            ".\nThe following signature is supported:\n",
                            c10::str("    ", metric::cdist_functor::schema_str, "\n"))
                } else {
                    TORCH_CHECK_TYPE(
                            incompatible_args.empty(),
                            metric::pdist_functor::name,
                            " was called with incompatible arguments ",
                            std::accumulate(
                                    std::begin(incompatible_args), std::end(incompatible_args), std::string(),
                                    [](const std::string &ss, std::string _arg) {
                                        return ss.empty() ? _arg : ss + ", " + _arg;
                                    }),
                            ".\nThe following signature is supported:\n",
                            c10::str("    ", metric::pdist_functor::schema_str, "\n"))
                }
            }

            // sklearn
            DEFINE_CPDIST_METRIC_SPEC(euclidean,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(haversine,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(manhattan,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(cosine,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            // scipy
            DEFINE_CPDIST_METRIC_SPEC(directed_hausdorff,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ true, /* generator */ true)
            DEFINE_CPDIST_METRIC_SPEC(minkowski,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ true, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(wminkowski,
            /* w */ true, /* V */ 0, /* VI */ 0, /* p */ true, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(sqeuclidean,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(correlation,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(hamming,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(jaccard,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(kulsinski,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(kulczynski1,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(seuclidean,
            /* w */ 0, /* V */ true, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(cityblock,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(mahalanobis,
            /* w */ 0, /* V */ 0, /* VI */ true, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(chebyshev,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(braycurtis,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(canberra,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(jensenshannon,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ true, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(yule,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(dice,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(rogerstanimoto,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(russellrao,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(sokalmichener,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(sokalsneath,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            // aliases
            DEFINE_CPDIST_METRIC_SPEC(l1,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(l2,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ 0, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
            DEFINE_CPDIST_METRIC_SPEC(lp,
            /* w */ 0, /* V */ 0, /* VI */ 0, /* p */ true, /* base */ 0, /* shuffle */ 0, /* generator */ 0)
        }

        at::Tensor cdist(
                const at::Tensor &x1,
                const at::Tensor &x2,
                c10::string_view metric,
                const c10::optional<at::Tensor> &w,
                const c10::optional<at::Tensor> &V,
                const c10::optional<at::Tensor> &VI,
                c10::optional<double> p,
                c10::optional<double> base,
                c10::optional<bool> shuffle,
                c10::optional<at::Generator> generator) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.cpdist.cdist")
#define CDIST_TRY_CALL(METRIC)                                                                             \
if (metric == #METRIC) {                                                                                   \
    detail::check_extra_args<true, detail::cpdist_##METRIC##_spec>(w, V, VI, p, base, shuffle, generator); \
    return detail::cpdist_##METRIC##_spec::cdist_func(x1, x2);                                             \
}

#define CDIST_TRY_CALL_WITH(METRIC, ARG1)                                                                  \
if (metric == #METRIC) {                                                                                   \
    detail::check_extra_args<true, detail::cpdist_##METRIC##_spec>(w, V, VI, p, base, shuffle, generator); \
    return detail::cpdist_##METRIC##_spec::cdist_func(x1, x2, ARG1);                                       \
}

#define CDIST_TRY_CALL_WITH2(METRIC, ARG1, ARG2)                                                           \
if (metric == #METRIC) {                                                                                   \
    detail::check_extra_args<true, detail::cpdist_##METRIC##_spec>(w, V, VI, p, base, shuffle, generator); \
    return detail::cpdist_##METRIC##_spec::cdist_func(x1, x2, ARG1, ARG2);                                 \
}
            constexpr std::array supported_metrics = {
                    "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine",
                    "dice", "directed_hausdorff", "euclidean", "hamming", "haversine", "jaccard",
                    "jensenshannon", "kulczynski1", "kulsinski", "l1", "l2", "lp", "mahalanobis",
                    "manhattan", "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
                    "sokalmichener", "sokalsneath", "sqeuclidean", "wminkowski", "yule"
            };
            CDIST_TRY_CALL(euclidean)
            CDIST_TRY_CALL(haversine)
            CDIST_TRY_CALL(manhattan)
            CDIST_TRY_CALL(cosine)
            if (metric == "directed_hausdorff") {
                detail::check_extra_args<true, detail::cpdist_directed_hausdorff_spec>(
                        w, V, VI, p, base, shuffle, generator);
                auto result = directed_hausdorff_distances(x1, x2, shuffle.value_or(false), generator);
                return std::get<0>(result);
            }
            CDIST_TRY_CALL_WITH(minkowski, p.value_or(2))
            CDIST_TRY_CALL_WITH2(wminkowski, p.value_or(2), w)
            CDIST_TRY_CALL(sqeuclidean)
            CDIST_TRY_CALL(correlation)
            CDIST_TRY_CALL(hamming)
            CDIST_TRY_CALL(jaccard)
            CDIST_TRY_CALL(kulsinski)
            CDIST_TRY_CALL(kulczynski1)
            CDIST_TRY_CALL_WITH(seuclidean, V)
            CDIST_TRY_CALL(cityblock)
            CDIST_TRY_CALL_WITH(mahalanobis, VI)
            CDIST_TRY_CALL(chebyshev)
            CDIST_TRY_CALL(braycurtis)
            CDIST_TRY_CALL(canberra)
            CDIST_TRY_CALL_WITH(jensenshannon, base)
            CDIST_TRY_CALL(yule)
            CDIST_TRY_CALL(dice)
            CDIST_TRY_CALL(rogerstanimoto)
            CDIST_TRY_CALL(russellrao)
            CDIST_TRY_CALL(sokalmichener)
            CDIST_TRY_CALL(sokalsneath)
            CDIST_TRY_CALL(l1)
            CDIST_TRY_CALL(l2)
            CDIST_TRY_CALL_WITH(lp, p.value_or(2))
            TORCH_CHECK(false,
                        "Got unkown distance metric: ",
                        metric,
                        ". Supported metrics are [",
                        std::accumulate(
                                std::begin(supported_metrics), std::end(supported_metrics), std::string(),
                                [](const std::string &ss, std::string m) {
                                    return ss.empty() ? m : ss + ", " + m;
                                }),
                        "]")
        }

        at::Tensor pdist(
                const at::Tensor &input,
                c10::string_view metric,
                const c10::optional<at::Tensor> &w,
                const c10::optional<at::Tensor> &V,
                const c10::optional<at::Tensor> &VI,
                c10::optional<double> p,
                c10::optional<double> base,
                c10::optional<bool> shuffle,
                c10::optional<at::Generator> generator) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.cpdist.pdist")
            TORCH_CHECK_NOT_IMPLEMENTED(false,
                                        "pdist is yet to be implemented.")
        }

        TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
            m.def("torchpairwise::cdist(Tensor x1, Tensor x2, str metric=\"minkowski\", *, "
                  TORCHPAIRWISE_CPDIST_EXTRA_ARGS_SCHEMA_STR ") -> Tensor",
                  TORCH_FN(cdist));
            m.def("torchpairwise::pdist(Tensor input, str metric=\"minkowski\", *, "
                  TORCHPAIRWISE_CPDIST_EXTRA_ARGS_SCHEMA_STR ") -> Tensor",
                  TORCH_FN(pdist));
        }
    }
}
