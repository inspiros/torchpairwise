#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <limits>

#include "../weighted_comparisons_detail.h"
namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;

using GradSignature = std::tuple<Tensor, Tensor, Tensor>(const Tensor&,
                                                         const Tensor&,
                                                         const Tensor&,
                                                         const Tensor&, int64_t,
                                                         int64_t, double, bool);
class WeightedFeatureFunction
    : public torch::autograd::Function<WeightedFeatureFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor w, int64_t kind, bool paired,
                        double p) {
    ctx->save_for_backward({x, y, w});
    ctx->saved_data["kind"] = kind;
    ctx->saved_data["paired"] = paired;
    ctx->saved_data["p"] = p;
    return detail::weighted::feature_kernel(x, y, w, kind, paired, p);
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of native weighted replay are unsupported");
    auto local = ctx->get_saved_variables();
    Values inputs, result(6);
    std::vector<int64_t> indices;
    for (int64_t i = 0; i < 3; ++i) {
      local[i] = local[i].detach();
      if (ctx->needs_input_grad(i) && local[i].is_floating_point()) {
        local[i].set_requires_grad(true);
        inputs.push_back(local[i]);
        indices.push_back(i);
      }
    }
    if (inputs.empty())
      return result;
    at::AutoGradMode enable(true);
    auto value = detail::weighted::feature_kernel(
        local[0], local[1], local[2], ctx->saved_data["kind"].toInt(),
        ctx->saved_data["paired"].toBool(), ctx->saved_data["p"].toDouble());
    if (!value.requires_grad())
      return result;
    auto grads =
        torch::autograd::grad({value}, inputs, gradients, false, false, true);
    for (size_t i = 0; i < indices.size(); ++i)
      result[indices[i]] = grads[i];
    return result;
  }
};
class WeightedComparisonFunction
    : public torch::autograd::Function<WeightedComparisonFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor w, int64_t kind, int64_t mode,
                        double p, bool self) {
    ctx->save_for_backward({x, y, w});
    ctx->saved_data["kind"] = kind;
    ctx->saved_data["mode"] = mode;
    ctx->saved_data["p"] = p;
    ctx->saved_data["self"] = self;
    at::AutoDispatchBelowADInplaceOrView guard;
    return detail::weighted::_weighted_compare(x, y, w, kind, mode, p, self);
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of native weighted replay are unsupported");
    auto v = ctx->get_saved_variables();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_weighted_compare_backward", "")
            .typed<GradSignature>();
    auto g = op.call(
        gradients[0], v[0], v[1], v[2], ctx->saved_data["kind"].toInt(),
        ctx->saved_data["mode"].toInt(), ctx->saved_data["p"].toDouble(),
        ctx->saved_data["self"].toBool());
    return {std::get<0>(g), std::get<1>(g), std::get<2>(g), Tensor(),
            Tensor(),       Tensor(),       Tensor()};
  }
};
Tensor _weighted_compare_autograd(const Tensor& x, const Tensor& y,
                                  const Tensor& w, int64_t kind, int64_t mode,
                                  double p, bool self) {
  return WeightedComparisonFunction::apply(x, y, w, kind, mode, p, self);
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_weighted_compare", TORCH_FN(_weighted_compare_autograd));
  m.impl("_weighted_compare_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
namespace detail::weighted {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
Tensor feature_autograd(Tensor x, Tensor y, Tensor w, int64_t kind, bool paired,
                        double p) {
  return WeightedFeatureFunction::apply(x, y, w, kind, paired, p);
}
}  // namespace detail::weighted
}  // namespace torchpairwise::ops
