#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <limits>

#include "../structured_comparisons_detail.h"
namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;

using GradResult = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
using GradSignature = GradResult(const Tensor&, const Tensor&, const Tensor&,
                                 const Tensor&, const Tensor&, const Tensor&,
                                 const Tensor&, const Tensor&, const Tensor&,
                                 int64_t, int64_t, bool, double, int64_t,
                                 int64_t, int64_t);

class MatrixRootFunction
    : public torch::autograd::Function<MatrixRootFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x) {
    auto state = detail::structured::matrix_root_kernel(x);
    ctx->save_for_backward({state[1], state[2]});
    return state[0];
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    return detail::structured::matrix_root_backward_kernel(
        gradients[0], ctx->get_saved_variables());
  }
};
class BuresFunction : public torch::autograd::Function<BuresFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor a,
                        Tensor b) {
    auto state = detail::structured::bures_kernel(a, b);
    ctx->save_for_backward({state[1], state[2]});
    ctx->saved_data["a"] = a.sizes().vec();
    ctx->saved_data["b"] = b.sizes().vec();
    return state[0];
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    return detail::structured::bures_backward_kernel(
        gradients[0], ctx->get_saved_variables(),
        ctx->saved_data["a"].toIntVector(), ctx->saved_data["b"].toIntVector());
  }
};
class MatrixLogFunction : public torch::autograd::Function<MatrixLogFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x) {
    auto state = detail::structured::matrix_log_kernel(x);
    ctx->save_for_backward({state[1], state[2]});
    return state[0];
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    return detail::structured::matrix_log_backward_kernel(
        gradients[0], ctx->get_saved_variables());
  }
};
class CongruenceFunction
    : public torch::autograd::Function<CongruenceFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor a,
                        Tensor b, Tensor factor, int64_t kind) {
    auto state = detail::structured::congruence_kernel(a, b, factor, kind);
    ctx->save_for_backward({state[1], state[2], state[3]});
    ctx->saved_data["a"] = a.sizes().vec();
    ctx->saved_data["b"] = b.sizes().vec();
    ctx->saved_data["kind"] = kind;
    return state[0];
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    return detail::structured::congruence_backward_kernel(
        gradients[0], ctx->get_saved_variables(),
        ctx->saved_data["a"].toIntVector(), ctx->saved_data["b"].toIntVector(),
        ctx->saved_data["kind"].toInt());
  }
};
class StructuredFeaturesFunction
    : public torch::autograd::Function<StructuredFeaturesFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor covariance, Tensor weights, Tensor lengths,
                        int64_t kind, double epsilon, int64_t iterations,
                        int64_t component_tile) {
    ctx->save_for_backward({x, covariance, weights, lengths});
    ctx->saved_data["kind"] = kind;
    ctx->saved_data["epsilon"] = epsilon;
    ctx->saved_data["iterations"] = iterations;
    ctx->saved_data["component_tile"] = component_tile;
    at::AutoDispatchBelowADInplaceOrView guard;
    auto result =
        detail::structured::features(x, covariance, weights, lengths, kind,
                                     epsilon, iterations, component_tile);
    if (kind == 2 || kind == 3)
      ctx->mark_non_differentiable({result[1]});
    return result;
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of structured features are unsupported");
    auto v = ctx->get_saved_variables();
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow(
                             "torchpairwise::_structured_features_backward", "")
                         .typed<detail::structured::FeatureGradSignature>();
    auto g = op.call(gradients, v[0], v[1], v[2], v[3],
                     ctx->saved_data["kind"].toInt(),
                     ctx->saved_data["epsilon"].toDouble(),
                     ctx->saved_data["iterations"].toInt(),
                     ctx->saved_data["component_tile"].toInt());
    return {g[0], g[1], g[2], Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
  }
};
Values structured_features_autograd(const Tensor& x, const Tensor& covariance,
                                    const Tensor& weights,
                                    const Tensor& lengths, int64_t kind,
                                    double epsilon, int64_t iterations,
                                    int64_t component_tile) {
  return StructuredFeaturesFunction::apply(x, covariance, weights, lengths,
                                           kind, epsilon, iterations,
                                           component_tile);
}
class StructuredComparisonFunction
    : public torch::autograd::Function<StructuredComparisonFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor xc, Tensor xw, Tensor y, Tensor yc, Tensor yw,
                        Tensor xl, Tensor yl, int64_t kind, int64_t mode,
                        bool squared, double epsilon, int64_t iterations,
                        int64_t tile, int64_t component_tile) {
    ctx->save_for_backward({x, xc, xw, y, yc, yw, xl, yl});
    ctx->saved_data["kind"] = kind;
    ctx->saved_data["mode"] = mode;
    ctx->saved_data["squared"] = squared;
    ctx->saved_data["epsilon"] = epsilon;
    ctx->saved_data["iterations"] = iterations;
    ctx->saved_data["tile"] = tile;
    ctx->saved_data["component_tile"] = component_tile;
    at::AutoDispatchBelowADInplaceOrView guard;
    return detail::structured::_structured_compare(
        x, xc, xw, y, yc, yw, xl, yl, kind, mode, squared, epsilon, iterations,
        tile, component_tile);
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(
        !at::GradMode::is_enabled(),
        "Higher derivatives of native structured comparisons are unsupported");
    auto s = ctx->get_saved_variables();
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow(
                             "torchpairwise::_structured_compare_backward", "")
                         .typed<GradSignature>();
    auto result = op.call(
        gradients[0], s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        ctx->saved_data["kind"].toInt(), ctx->saved_data["mode"].toInt(),
        ctx->saved_data["squared"].toBool(),
        ctx->saved_data["epsilon"].toDouble(),
        ctx->saved_data["iterations"].toInt(), ctx->saved_data["tile"].toInt(),
        ctx->saved_data["component_tile"].toInt());
    return {std::get<0>(result),
            std::get<1>(result),
            std::get<2>(result),
            std::get<3>(result),
            std::get<4>(result),
            std::get<5>(result),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor()};
  }
};
Tensor _structured_compare_autograd(const Tensor& x, const Tensor& xc,
                                    const Tensor& xw, const Tensor& y,
                                    const Tensor& yc, const Tensor& yw,
                                    const Tensor& xl, const Tensor& yl,
                                    int64_t kind, int64_t mode, bool squared,
                                    double epsilon, int64_t iterations,
                                    int64_t tile, int64_t component_tile) {
  return StructuredComparisonFunction::apply(x, xc, xw, y, yc, yw, xl, yl, kind,
                                             mode, squared, epsilon, iterations,
                                             tile, component_tile);
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_structured_features", TORCH_FN(structured_features_autograd));
  m.impl("_structured_features_backward",
         torch::CppFunction::makeFallthrough());
  m.impl("_structured_validate_raw", torch::CppFunction::makeFallthrough());
  m.impl("_structured_compare", TORCH_FN(_structured_compare_autograd));
  m.impl("_structured_compare_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
namespace detail::structured {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
using GradResult = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
Tensor matrix_root_autograd(Tensor x) {
  return MatrixRootFunction::apply(x);
}
Tensor bures_autograd(Tensor a, Tensor b) {
  return BuresFunction::apply(a, b);
}
Tensor matrix_log_autograd(Tensor x) {
  return MatrixLogFunction::apply(x);
}
Tensor congruence_autograd(Tensor a, Tensor b, Tensor factor, int64_t kind) {
  return CongruenceFunction::apply(a, b, factor, kind);
}
}  // namespace detail::structured
}  // namespace torchpairwise::ops
