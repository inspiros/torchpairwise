#pragma once

#include "pairwise_metrics.h"

#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include "additive_chi2_kernel.h"
#include "braycurtis.h"
#include "canberra.h"
#include "haversine.h"
#include "hausdorff.h"
#include "pairwise_binary.h"
#include "ppminkowski.h"
#include "prf_div.h"
#include "sqjensenshannon.h"
#include "sqmahalanobis.h"
#include "wminkowski.h"
#include "utils/scalar_type_utils.h"

namespace torchpairwise {
    namespace ops {
        at::Tensor euclidean_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.euclidean_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("euclidean_distances", x1, x2);
            return at::cdist(x1_, x2_, 2);
        }

        at::Tensor haversine_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.haversine_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("haversine_distances", x1, x2);
            return _haversine(x1_, x2_);
        }

        at::Tensor manhattan_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.manhattan_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("manhattan_distances", x1, x2);
            return at::cdist(x1_, x2_, 1);
        }

        at::Tensor cosine_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.cosine_distances")
            return cosine_similarity(x1, x2).neg_().add_(1).clip_(0, 2);
        }

        at::Tensor linear_kernel_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.linear_kernel")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("linear_kernel", x1, x2);
            return x1_.matmul(x2_.transpose(-1, -2));
        }

        at::Tensor polynomial_kernel_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2,
                int64_t degree,
                c10::optional<double> gamma,
                double coef0) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.polynomial_kernel")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("polynomial_kernel", x1, x2);
            auto gamma_ = gamma.value_or(1.0 / x1.size(1));
            return x1_.matmul(x2_.transpose(-1, -2)).mul_(gamma_).add_(coef0).pow_(degree);
        }

        at::Tensor sigmoid_kernel_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2,
                c10::optional<double> gamma,
                double coef0) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.sigmoid_kernel")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("sigmoid_kernel", x1, x2);
            auto gamma_ = gamma.value_or(1.0 / x1.size(1));
            return x1_.matmul(x2_.transpose(-1, -2)).mul_(gamma_).add_(coef0).tanh_();
        }

        at::Tensor rbf_kernel_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2,
                c10::optional<double> gamma) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.rbf_kernel")
            auto gamma_ = gamma.value_or(1.0 / x1.size(1));
            return sqeuclidean_distances_functor::call(x1, x2).mul_(-gamma_).exp_();
        }

        at::Tensor laplacian_kernel_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2,
                c10::optional<double> gamma) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.laplacian_kernel")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("laplacian_kernel", x1, x2);
            auto gamma_ = gamma.value_or(1.0 / x1.size(1));
            return at::cdist(x1_, x2_, 1).mul_(-gamma_).exp_();
        }

        at::Tensor cosine_similarity_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.cosine_similarity")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("cosine_similarity", x1, x2);
            auto x1_norm = x1.norm(2, -1, true);
            auto x2_norm = x2.has_value() ? x2_.norm(2, -1, true) : x1_norm;
            auto denom = x1_norm.mul(x2_norm.transpose(-1, -2));
            return prf_div(x1_.matmul(x2_.transpose(-1, -2)), denom, "identity");
        }

        at::Tensor additive_chi2_kernel_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.additive_chi2_kernel")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("additive_chi2_kernel", x1, x2);
            TORCH_CHECK(x1_.ge(0).all().item<bool>() && (!x2.has_value() || x2_.ge(0).all().item<bool>()),
                        "All elements of x1 and x2 must be positive")
            return _additive_chi2_kernel(x1_, x2_);
        }

        at::Tensor chi2_kernel_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2,
                double gamma) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.chi2_kernel")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("additive_chi2_kernel", x1, x2);
            TORCH_CHECK(x1_.ge(0).all().item<bool>() && (!x2.has_value() || x2_.ge(0).all().item<bool>()),
                        "All elements of x1 and x2 must be positive")
            return _additive_chi2_kernel(x1, x2_).mul_(gamma).exp_();
        }

        // scipy
        std::tuple<at::Tensor, at::Tensor, at::Tensor>
                directed_hausdorff_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2,
                bool shuffle,
                c10::optional<at::Generator> generator) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.directed_hausdorff_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs<false, true>("directed_hausdorff_distances", x1, x2);
            return _directed_hausdorff(x1_, x2_, shuffle, generator);
        }

        at::Tensor minkowski_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2,
                double p) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.minkowski_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("minkowski_distances", x1, x2);
            return at::cdist(x1_, x2_, p);
        }

        at::Tensor wminkowski_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2,
                double p,
                const c10::optional<at::Tensor> &w) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.wminkowski_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("wminkowski_distances", x1, x2);
            return w.has_value() ? _wminkowski(x1_, x2_, w.value(), p) : at::cdist(x1_, x2_, p);
        }

        at::Tensor sqeuclidean_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.sqeuclidean_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("sqeuclidean_distances", x1, x2);
            return _ppminkowski(x1_, x2_, 2);
        }

        at::Tensor correlation_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.correlation_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("correlation_distances", x1, x2);
            // centering
            x1_ = x1_ - x1_.mean(-1, true);
            x2_ = x2.has_value() ? x2_ - x2_.mean(-1, true) : x1_;

            auto x1_norm = x1_.norm(2, -1, true);
            auto x2_norm = x2.has_value() ? x2_.norm(2, -1, true) : x1_norm;
            auto denom = x1_norm.mul(x2_norm.transpose(-1, -2));
            return prf_div(x1_.matmul(x2_.transpose(-1, -2)), denom, "identity").neg_().add_(1);
        }

        at::Tensor hamming_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.hamming_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("hamming_distances", x1, x2);
            return pwne_mean(x1_, x2_);
        }

        at::Tensor jaccard_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.jaccard_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs<true>("jaccard_distances", x1, x2);
            return prf_div(pwxor_sum(x1_, x2_), pwor_sum(x1_, x2_));
        }

        at::Tensor kulsinski_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            TORCH_WARN_DEPRECATION("kulsinski_distances is deprecated in SciPy 1.9.0 and might "
                                   "be removed in the future. Please use kulczynski1_distances.")
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.kulsinski_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs<true>("kulsinski_distances", x1, x2);
            auto S = pwxor_sum(x1_, x2_);
            return (S - pwand_sum(x1_, x2_) + x1_.size(-1)) / (S + x1_.size(-1));
        }

        at::Tensor kulczynski1_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.kulczynski1_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs<true>("kulczynski1_distances", x1, x2);
            return pwand_sum(x1_, x2_) / pwxor_sum(x1_, x2_);
        }

        at::Tensor seuclidean_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2,
                const c10::optional<at::Tensor> &V) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.seuclidean_distances")
            TORCH_CHECK(!x2.has_value() || V.has_value(),
                        "V is required for seuclidean when x2 is passed.")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("euclidean_distances", x1, x2);
            at::Tensor V_;
            if (V.has_value()) {
                V_ = V.value();
            } else {
                at::NoGradGuard no_grad_guard;
                auto var_dims = x1_.ndimension() == 2 ? at::IntArrayRef{0} : at::IntArrayRef{0, 1};
                V_ = x1_.var(var_dims);
            }
            return _wminkowski(x1_, x2_, prf_ldiv(V_, 1, "identity"), 2);
        }

        at::Tensor cityblock_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.cityblock_distances")
            return manhattan_distances_functor::call(x1, x2);
        }

        at::Tensor mahalanobis_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2,
                const c10::optional<at::Tensor> &VI) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.mahalanobis_distances")
            TORCH_CHECK(!x2.has_value() || VI.has_value(),
                        "VI is required for manhalanobis when x2 is passed.")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("mahalanobis_distances", x1, x2);
            at::Tensor VI_;
            if (VI.has_value()) {
                VI_ = VI.value();
            } else {
                at::NoGradGuard no_grad_guard;
                if (x1_.ndimension() == 2)
                    VI_ = at::cov(x1_.transpose(-1, -2)).inverse().transpose_(-1, -2);
                else {
                    auto cov = at::empty({x1_.size(0), x1_.size(1), x1_.size(1)}, x1_.options());
                    for (const auto b: c10::irange(x1_.size(0))) {
                        cov.index_put_({b}, at::cov(x1_[b].transpose(-1, -2)));
                    }
                    VI_ = cov.inverse().transpose_(-1, -2);
                }
            }
            return _sqmahalanobis(x1_, x2_, VI_).sqrt_();
        }

        at::Tensor chebyshev_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.chebyshev_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("chebyshev_distances", x1, x2);
            return at::cdist(x1_, x2_, c10::CPPTypeLimits<double>::upper_bound());
        }

        at::Tensor braycurtis_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.braycurtis_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("braycurtis_distances", x1, x2);
            return _braycurtis(x1_, x2_);
        }

        at::Tensor canberra_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.canberra_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("canberra_distances", x1, x2);
            return _canberra(x1_, x2_);
        }

        at::Tensor jensenshannon_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2,
                c10::optional<double> base) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.jensenshannon_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("jensenshannon_distances", x1, x2);
            // normalize 0-1
            x1_ = x1_ / x1_.sum(-1, true);
            x2_ = x2.has_value() ? x2_ / x2_.sum(-1, true) : x1_;
            return _sqjensenshannon(x1_, x2_, base).sqrt_();
        }

        at::Tensor yule_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.yule_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs<true>("yule_distances", x1, x2);
            auto half_R = pwand_sum(x1_, ~x2_) * pwand_sum(~x1_, x2_);
            return prf_div(2 * half_R, (pwand_sum(x1_, x2_) * pwand_sum(~x1_, ~x2_) + half_R));
        }

        at::Tensor dice_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.dice_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs<true>("dice_distances", x1, x2);
            auto R = pwxor_sum(x1_, x2_);
            return R / (2 * pwand_sum(x1_, x2_) + R);
        }

        at::Tensor rogerstanimoto_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.rogerstanimoto_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs<true>("rogerstanimoto_distances", x1, x2);
            auto R = 2 * pwxor_sum(x1_, x2_);
            auto S = pweq_sum(x1_, x2_);
            return R / (S + R);
        }

        at::Tensor russellrao_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.russellrao_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs<true>("russellrao_distances", x1, x2);
            return (x1.size(-1) - pwand_sum(x1_, x2_)) / x1.size(-1);
        }

        at::Tensor sokalmichener_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.sokalmichener_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs<true>("sokalmichener_distances", x1, x2);
            auto R = 2 * pwxor_sum(x1_, x2_);
            auto S = pweq_sum(x1_, x2_);
            return R / (S + R);
        }

        at::Tensor sokalsneath_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.sokalsneath_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs<true>("sokalsneath_distances", x1, x2);
            auto R = 2 * pwxor_sum(x1_, x2_);
            // intentionally returns nan for entirely different pairs
            return R / (pwand_sum(x1_, x2_) + R);
        }

        // aliases
        at::Tensor l1_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.l1_distances")
            return manhattan_distances(x1, x2);
        }

        at::Tensor l2_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.l2_distances")
            return euclidean_distances(x1, x2);
        }

        at::Tensor lp_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2,
                double p) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.lp_distances")
            return minkowski_distances(x1, x2, p);
        }

        TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
#define REGISTER_FUNCTOR(FUNCTOR) \
    m.def(c10::str("torchpairwise::", FUNCTOR::schema_str).c_str(), TORCH_FN(FUNCTOR::call))
            // sklearn
            REGISTER_FUNCTOR(euclidean_distances_functor);
            REGISTER_FUNCTOR(haversine_distances_functor);
            REGISTER_FUNCTOR(manhattan_distances_functor);
            REGISTER_FUNCTOR(cosine_distances_functor);
            REGISTER_FUNCTOR(linear_kernel_functor);
            REGISTER_FUNCTOR(polynomial_kernel_functor);
            REGISTER_FUNCTOR(sigmoid_kernel_functor);
            REGISTER_FUNCTOR(rbf_kernel_functor);
            REGISTER_FUNCTOR(laplacian_kernel_functor);
            REGISTER_FUNCTOR(cosine_similarity_functor);
            REGISTER_FUNCTOR(additive_chi2_kernel_functor);
            REGISTER_FUNCTOR(chi2_kernel_functor);
            // scipy
            REGISTER_FUNCTOR(directed_hausdorff_distances_functor);
            REGISTER_FUNCTOR(minkowski_distances_functor);
            REGISTER_FUNCTOR(wminkowski_distances_functor);
            REGISTER_FUNCTOR(sqeuclidean_distances_functor);
            REGISTER_FUNCTOR(correlation_distances_functor);
            REGISTER_FUNCTOR(hamming_distances_functor);
            REGISTER_FUNCTOR(jaccard_distances_functor);
            REGISTER_FUNCTOR(kulsinski_distances_functor);
            REGISTER_FUNCTOR(kulczynski1_distances_functor);
            REGISTER_FUNCTOR(seuclidean_distances_functor);
            REGISTER_FUNCTOR(cityblock_distances_functor);
            REGISTER_FUNCTOR(mahalanobis_distances_functor);
            REGISTER_FUNCTOR(chebyshev_distances_functor);
            REGISTER_FUNCTOR(braycurtis_distances_functor);
            REGISTER_FUNCTOR(canberra_distances_functor);
            REGISTER_FUNCTOR(jensenshannon_distances_functor);
            REGISTER_FUNCTOR(yule_distances_functor);
            REGISTER_FUNCTOR(dice_distances_functor);
            REGISTER_FUNCTOR(rogerstanimoto_distances_functor);
            REGISTER_FUNCTOR(russellrao_distances_functor);
            REGISTER_FUNCTOR(sokalmichener_distances_functor);
            REGISTER_FUNCTOR(sokalsneath_distances_functor);
            // aliases
            REGISTER_FUNCTOR(l1_distances_functor);
            REGISTER_FUNCTOR(l2_distances_functor);
            REGISTER_FUNCTOR(lp_distances_functor);
        }
    }
}
