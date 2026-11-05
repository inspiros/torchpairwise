#include <torch/autograd.h>
#include <torch/library.h>

#include "../image_comparisons_detail.h"
#include "../image_map_operations_detail.h"
#include "../utils/centered_moments.h"
namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = std::vector<Tensor>;
class ImageMapFunction : public torch::autograd::Function<ImageMapFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, int64_t kind, int64_t mode, int64_t size,
                        int64_t window, std::vector<double> p,
                        bool single_precision) {
    at::AutoDispatchBelowADInplaceOrView guard;
    auto out = detail::image::image_map(x, y, kind, mode, size, window, p,
                                        single_precision);
    ctx->save_for_backward({x, y, std::get<1>(out)});
    ctx->mark_non_differentiable({std::get<1>(out)});
    ctx->saved_data["kind"] = kind;
    ctx->saved_data["mode"] = mode;
    ctx->saved_data["size"] = size;
    ctx->saved_data["window"] = window;
    ctx->saved_data["p"] = p;
    return {std::get<0>(out), std::get<1>(out)};
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of native image maps are unsupported");
    auto v = ctx->get_saved_variables();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_image_map_backward", "")
            .typed<detail::image::MapGrad>();
    auto g = op.call(
        gradients[0], v[0], v[1], v[2], ctx->saved_data["kind"].toInt(),
        ctx->saved_data["mode"].toInt(), ctx->saved_data["size"].toInt(),
        ctx->saved_data["window"].toInt(),
        ctx->saved_data["p"].toDoubleVector());
    return {std::get<0>(g), std::get<1>(g), Tensor(), Tensor(),
            Tensor(),       Tensor(),       Tensor(), Tensor()};
  }
};
std::tuple<Tensor, Tensor> image_map_autograd(const Tensor& x, const Tensor& y,
                                              int64_t kind, int64_t mode,
                                              int64_t size, int64_t window,
                                              at::ArrayRef<double> p,
                                              bool single_precision) {
  auto v = ImageMapFunction::apply(x, y, kind, mode, size, window, p.vec(),
                                   single_precision);
  return {v[0], v[1]};
}
class ImageMapValuesFunction
    : public torch::autograd::Function<ImageMapValuesFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, int64_t comparison,
                        std::vector<int64_t> settings,
                        std::vector<double> parameters, int64_t dtype,
                        bool paired) {
    at::AutoDispatchBelowADInplaceOrView guard;
    auto result = detail::media::map_values(x, y, comparison, settings,
                                            parameters, dtype, paired);
    ctx->save_for_backward({x, y});
    ctx->saved_data["comparison"] = comparison;
    ctx->saved_data["settings"] = settings;
    ctx->saved_data["parameters"] = parameters;
    ctx->saved_data["dtype"] = dtype;
    ctx->saved_data["paired"] = paired;
    return result;
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of image maps are unsupported");
    auto v = ctx->get_saved_variables();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_image_map_values_backward", "")
            .typed<detail::media::MapValuesGradSignature>();
    auto [gx, gy] = op.call(
        gradients[0], v[0], v[1], ctx->saved_data["comparison"].toInt(),
        ctx->saved_data["settings"].toIntVector(),
        ctx->saved_data["parameters"].toDoubleVector(),
        ctx->saved_data["dtype"].toInt(), ctx->saved_data["paired"].toBool());
    return {gx, gy, Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
  }
};
Tensor image_map_values_autograd(const Tensor& x, const Tensor& y,
                                 int64_t comparison, at::IntArrayRef settings,
                                 at::ArrayRef<double> parameters, int64_t dtype,
                                 bool paired) {
  return ImageMapValuesFunction::apply(x, y, comparison, settings.vec(),
                                       parameters.vec(), dtype, paired);
}
class ImageMapContractFunction
    : public torch::autograd::Function<ImageMapContractFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, int64_t comparison,
                        std::vector<int64_t> settings,
                        std::vector<double> parameters, int64_t dtype,
                        int64_t code, int64_t mask, bool keepdim,
                        double correction, bool paired) {
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_image_map_contract", "anchored")
            .typed<detail::media::MapContractSignature>();
    auto [result, state] =
        op.call(x, y, comparison, settings, parameters, dtype, code, mask,
                keepdim, correction, paired);
    ctx->save_for_backward({x, y, state});
    if (code >= 7)
      ctx->mark_non_differentiable({result, state});
    else
      ctx->mark_non_differentiable({state});
    ctx->saved_data["comparison"] = comparison;
    ctx->saved_data["settings"] = settings;
    ctx->saved_data["parameters"] = parameters;
    ctx->saved_data["dtype"] = dtype;
    ctx->saved_data["code"] = code;
    ctx->saved_data["mask"] = mask;
    ctx->saved_data["keepdim"] = keepdim;
    ctx->saved_data["correction"] = correction;
    ctx->saved_data["paired"] = paired;
    return {result, state};
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher image contraction derivatives are unsupported");
    auto v = ctx->get_saved_variables();
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow(
                             "torchpairwise::_image_map_contract_backward", "")
                         .typed<detail::media::MapContractGradSignature>();
    auto [gx, gy] = op.call(
        gradients[0], v[0], v[1], v[2], ctx->saved_data["comparison"].toInt(),
        ctx->saved_data["settings"].toIntVector(),
        ctx->saved_data["parameters"].toDoubleVector(),
        ctx->saved_data["dtype"].toInt(), ctx->saved_data["code"].toInt(),
        ctx->saved_data["mask"].toInt(), ctx->saved_data["keepdim"].toBool(),
        ctx->saved_data["correction"].toDouble(),
        ctx->saved_data["paired"].toBool());
    return {gx,       gy,       Tensor(), Tensor(), Tensor(), Tensor(),
            Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
  }
};
detail::media::EventOutputs anchored_image_map_contract_autograd(
    const Tensor& x, const Tensor& y, int64_t comparison,
    at::IntArrayRef settings, at::ArrayRef<double> parameters, int64_t dtype,
    int64_t code, int64_t mask, bool keepdim, double correction, bool paired) {
  auto v = ImageMapContractFunction::apply(x, y, comparison, settings.vec(),
                                           parameters.vec(), dtype, code, mask,
                                           keepdim, correction, paired);
  return {v[0], v[1]};
}
detail::media::EventOutputs image_map_contract_autograd(
    const Tensor& x, const Tensor& y, int64_t comparison,
    at::IntArrayRef settings, at::ArrayRef<double> parameters, int64_t dtype,
    int64_t code, int64_t mask, bool keepdim, double correction, bool paired) {
  auto [result, state] = anchored_image_map_contract_autograd(
      x, y, comparison, settings, parameters, dtype, code, mask, keepdim,
      correction, paired);
  return {result, detail::legacy_moment_state(state, 3, 1)};
}
class ImageMapReductionFunction
    : public torch::autograd::Function<ImageMapReductionFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, int64_t comparison, std::string reduction,
                        std::vector<int64_t> settings,
                        std::vector<double> parameters, int64_t dtype,
                        double correction, bool paired) {
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_image_map_reduce", "")
            .typed<detail::media::MapReductionSignature>();
    auto result = op.call(x, y, comparison, reduction, settings, parameters,
                          dtype, correction, paired);
    ctx->save_for_backward({x, y});
    ctx->saved_data["comparison"] = comparison;
    ctx->saved_data["reduction"] = reduction;
    ctx->saved_data["settings"] = settings;
    ctx->saved_data["parameters"] = parameters;
    ctx->saved_data["dtype"] = dtype;
    ctx->saved_data["correction"] = correction;
    ctx->saved_data["paired"] = paired;
    if (reduction == "all" || reduction == "any")
      ctx->mark_non_differentiable({result});
    return result;
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher image-map event derivatives are unsupported");
    auto v = ctx->get_saved_variables();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_image_map_reduce_backward", "")
            .typed<detail::media::MapReductionGradSignature>();
    auto [gx, gy] =
        op.call(gradients[0], v[0], v[1], ctx->saved_data["comparison"].toInt(),
                ctx->saved_data["reduction"].toStringRef(),
                ctx->saved_data["settings"].toIntVector(),
                ctx->saved_data["parameters"].toDoubleVector(),
                ctx->saved_data["dtype"].toInt(),
                ctx->saved_data["correction"].toDouble(),
                ctx->saved_data["paired"].toBool());
    return {gx,       gy,       Tensor(), Tensor(), Tensor(),
            Tensor(), Tensor(), Tensor(), Tensor()};
  }
};
Tensor image_map_reduce_autograd(const Tensor& x, const Tensor& y,
                                 int64_t comparison, c10::string_view reduction,
                                 at::IntArrayRef settings,
                                 at::ArrayRef<double> parameters, int64_t dtype,
                                 double correction, bool paired) {
  return ImageMapReductionFunction::apply(
      x, y, comparison, std::string(reduction), settings.vec(),
      parameters.vec(), dtype, correction, paired);
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_image_map_contract", TORCH_FN(image_map_contract_autograd));
  m.impl("_image_map_contract.anchored",
         TORCH_FN(anchored_image_map_contract_autograd));
  m.impl("_image_map_contract_backward", torch::CppFunction::makeFallthrough());
  m.impl("_image_map_values", TORCH_FN(image_map_values_autograd));
  m.impl("_image_map_values_backward", torch::CppFunction::makeFallthrough());
  m.impl("_image_map_reduce", TORCH_FN(image_map_reduce_autograd));
  m.impl("_image_map_reduce_backward", torch::CppFunction::makeFallthrough());
  m.impl("_image_map", TORCH_FN(image_map_autograd));
  m.impl("_image_map_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
}  // namespace torchpairwise::ops
