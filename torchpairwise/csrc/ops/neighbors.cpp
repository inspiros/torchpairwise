#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include "pairwise_metrics.h"
#include "cpdist.h"

namespace torchpairwise {
    namespace ops {
        at::Tensor k_neighbors_mask(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2,
                int64_t k,
                c10::string_view metric,
                const c10::optional<at::Tensor> &w,
                const c10::optional<at::Tensor> &V,
                const c10::optional<at::Tensor> &VI,
                c10::optional<double> p,
                c10::optional<double> base,
                c10::optional<bool> shuffle,
                c10::optional<at::Generator> generator) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.neighbors.k_neighbors_mask")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("k_neighbors_mask", x1, x2);
            TORCH_CHECK(k >= 0,
                        "k must be non-negative. Got k=",
                        k)
            at::NoGradGuard no_grad_guard;
            auto dists = torchpairwise::ops::cdist(x1_, x2_, metric, w, V, VI, p, base, shuffle, generator);
            auto neighbors_inds = dists.argsort(1, false).slice(1, 0, k + 1);
            auto first_dim = at::arange(0, x1_.size(0), neighbors_inds.options()).view({-1, 1}).repeat({1, k + 1});
            auto output = at::zeros({x1.size(0), x2_.size(0)}, x1_.options().dtype(at::kBool));
            output.index_put_({first_dim.flatten(), neighbors_inds.flatten()}, true);
            return output;
        }

        at::Tensor radius_neighbors_mask(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2,
                double epsilon,
                c10::string_view metric,
                const c10::optional<at::Tensor> &w,
                const c10::optional<at::Tensor> &V,
                const c10::optional<at::Tensor> &VI,
                c10::optional<double> p,
                c10::optional<double> base,
                c10::optional<bool> shuffle,
                c10::optional<at::Generator> generator) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.neighbors.k_neighbors_mask")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("radius_neighbors_mask", x1, x2);
            TORCH_CHECK(epsilon >= 0.,
                        "epsilon must be non-negative. Got epsilon=",
                        epsilon)
            at::NoGradGuard no_grad_guard;
            auto dists = torchpairwise::ops::cdist(x1_, x2_, metric, w, V, VI, p, base, shuffle, generator);
            return dists.le(epsilon);
        }

        TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
            // utilities
            m.def("torchpairwise::k_neighbors_mask(Tensor x1, Tensor? x2=None, int k=1, "
                  "str metric=\"euclidean\", *, " TORCHPAIRWISE_CPDIST_EXTRA_ARGS_SCHEMA_STR ") -> Tensor",
                  TORCH_FN(k_neighbors_mask));
            m.def("torchpairwise::radius_neighbors_mask(Tensor x1, Tensor? x2=None, float epsilon=0., "
                  "str metric=\"euclidean\", *, " TORCHPAIRWISE_CPDIST_EXTRA_ARGS_SCHEMA_STR ") -> Tensor",
                  TORCH_FN(radius_neighbors_mask));
        }
    }
}
