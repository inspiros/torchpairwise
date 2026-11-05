#include <torch/autograd.h>
#include <torch/library.h>

#include "../residual_operations_detail.h"
#include "../residual_reducer.h"
#include "../utils/centered_moments.h"

namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = std::vector<Tensor>;
class ResidualFeatureReductionFunction
    : public torch::autograd::Function<ResidualFeatureReductionFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, std::string reduction, std::string transform,
                        double correction, int64_t size, bool paired) {
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_residual_features", "anchored")
            .typed<detail::residual::FeatureSignature>();
    auto [result, state] =
        op.call(x, y, reduction, transform, correction, size, paired);
    ctx->save_for_backward({x, y, state});
    ctx->mark_non_differentiable({state});
    if (detail::residual::boolean_reduction(reduction))
      ctx->mark_non_differentiable({result});
    ctx->saved_data["reduction"] = reduction;
    ctx->saved_data["transform"] = transform;
    ctx->saved_data["correction"] = correction;
    ctx->saved_data["size"] = size;
    ctx->saved_data["paired"] = paired;
    return {result, state};
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(
        !at::GradMode::is_enabled(),
        "Higher derivatives of residual feature reductions are unsupported");
    auto saved = ctx->get_saved_variables();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_residual_features_backward", "")
            .typed<detail::residual::FeatureGradSignature>();
    auto [gx, gy] = op.call(gradients[0], saved[0], saved[1], saved[2],
                            ctx->saved_data["reduction"].toStringRef(),
                            ctx->saved_data["transform"].toStringRef(),
                            ctx->saved_data["correction"].toDouble(),
                            ctx->saved_data["size"].toInt(),
                            ctx->saved_data["paired"].toBool());
    return {gx, gy, Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
  }
};
detail::residual::FeatureOutputs anchored_feature_autograd(
    const Tensor& x, const Tensor& y, c10::string_view reduction,
    c10::string_view transform, double correction, int64_t size, bool paired) {
  auto output = ResidualFeatureReductionFunction::apply(
      x, y, std::string(reduction), std::string(transform), correction, size,
      paired);
  return {output[0], output[1]};
}
detail::residual::FeatureOutputs feature_autograd(
    const Tensor& x, const Tensor& y, c10::string_view reduction,
    c10::string_view transform, double correction, int64_t size, bool paired) {
  auto [result, state] = anchored_feature_autograd(x, y, reduction, transform,
                                                   correction, size, paired);
  return {result, detail::legacy_moment_state(state, 2, 0)};
}
class ResidualReductionFunction
    : public torch::autograd::Function<ResidualReductionFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, std::string reduction, std::string transform,
                        std::vector<int64_t> axes, bool keepdim,
                        double correction, int64_t sample_size,
                        int64_t feature_size) {
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_residual_reduce", "anchored")
            .typed<detail::residual::ReductionSignature>();
    auto [result, state] = op.call(x, y, reduction, transform, axes, keepdim,
                                   correction, sample_size, feature_size);
    ctx->save_for_backward({x, y, state});
    ctx->mark_non_differentiable({state});
    if (detail::residual::boolean_reduction(reduction))
      ctx->mark_non_differentiable({result});
    ctx->saved_data["reduction"] = reduction;
    ctx->saved_data["transform"] = transform;
    ctx->saved_data["axes"] = axes;
    ctx->saved_data["keepdim"] = keepdim;
    ctx->saved_data["correction"] = correction;
    ctx->saved_data["sample_size"] = sample_size;
    ctx->saved_data["feature_size"] = feature_size;
    return {result, state};
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of residual reductions are unsupported");
    auto saved = ctx->get_saved_variables();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_residual_reduce_backward", "")
            .typed<detail::residual::ReductionGradSignature>();
    auto [gx, gy] = op.call(gradients[0], saved[0], saved[1], saved[2],
                            ctx->saved_data["reduction"].toStringRef(),
                            ctx->saved_data["transform"].toStringRef(),
                            ctx->saved_data["axes"].toIntVector(),
                            ctx->saved_data["keepdim"].toBool(),
                            ctx->saved_data["correction"].toDouble(),
                            ctx->saved_data["sample_size"].toInt(),
                            ctx->saved_data["feature_size"].toInt());
    return {gx,       gy,       Tensor(), Tensor(), Tensor(),
            Tensor(), Tensor(), Tensor(), Tensor()};
  }
};
detail::residual::FeatureOutputs anchored_reduction_autograd(
    const Tensor& x, const Tensor& y, c10::string_view reduction,
    c10::string_view transform, at::IntArrayRef axes, bool keepdim,
    double correction, int64_t sample_size, int64_t feature_size) {
  auto output = ResidualReductionFunction::apply(
      x, y, std::string(reduction), std::string(transform), axes.vec(), keepdim,
      correction, sample_size, feature_size);
  return {output[0], output[1]};
}
detail::residual::FeatureOutputs reduction_autograd(
    const Tensor& x, const Tensor& y, c10::string_view reduction,
    c10::string_view transform, at::IntArrayRef axes, bool keepdim,
    double correction, int64_t sample_size, int64_t feature_size) {
  auto [result, state] =
      anchored_reduction_autograd(x, y, reduction, transform, axes, keepdim,
                                  correction, sample_size, feature_size);
  return {result, detail::legacy_moment_state(state, 3, 1)};
}
class ResidualFeatureTopKFunction
    : public torch::autograd::Function<ResidualFeatureTopKFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, int64_t k, bool largest,
                        std::string transform, int64_t sample_size,
                        int64_t feature_size) {
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("torchpairwise::_residual_topk", "")
                         .typed<detail::residual::TopKSignature>();
    auto [values, indices] =
        op.call(x, y, k, largest, transform, sample_size, feature_size);
    ctx->save_for_backward({x, y, indices});
    ctx->mark_non_differentiable({indices});
    ctx->saved_data["transform"] = transform;
    ctx->saved_data["sample_size"] = sample_size;
    ctx->saved_data["feature_size"] = feature_size;
    return {values, indices};
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of residual topk are unsupported");
    auto saved = ctx->get_saved_variables();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_residual_topk_backward", "")
            .typed<detail::residual::TopKGradSignature>();
    auto [gx, gy] = op.call(gradients[0], saved[0], saved[1], saved[2],
                            ctx->saved_data["transform"].toStringRef(),
                            ctx->saved_data["sample_size"].toInt(),
                            ctx->saved_data["feature_size"].toInt());
    return {gx, gy, Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
  }
};
detail::residual::FeatureOutputs topk_autograd(const Tensor& x, const Tensor& y,
                                               int64_t k, bool largest,
                                               c10::string_view transform,
                                               int64_t sample_size,
                                               int64_t feature_size) {
  auto out = ResidualFeatureTopKFunction::apply(
      x, y, k, largest, std::string(transform), sample_size, feature_size);
  return {out[0], out[1]};
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_residual_topk", TORCH_FN(topk_autograd));
  m.impl("_residual_topk_backward", torch::CppFunction::makeFallthrough());
  m.impl("_residual_reduce", TORCH_FN(reduction_autograd));
  m.impl("_residual_reduce.anchored", TORCH_FN(anchored_reduction_autograd));
  m.impl("_residual_reduce_backward", torch::CppFunction::makeFallthrough());
  m.impl("_residual_features", TORCH_FN(feature_autograd));
  m.impl("_residual_features.anchored", TORCH_FN(anchored_feature_autograd));
  m.impl("_residual_features_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
}  // namespace torchpairwise::ops
