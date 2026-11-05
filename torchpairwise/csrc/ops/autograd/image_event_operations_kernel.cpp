#include <torch/autograd.h>
#include <torch/library.h>

#include "../image_map_operations_detail.h"
#include "../utils/centered_moments.h"

namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = std::vector<Tensor>;
class ImageEventReductionFunction
    : public torch::autograd::Function<ImageEventReductionFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, int64_t comparison,
                        std::vector<int64_t> settings,
                        std::vector<double> parameters, int64_t dtype,
                        std::vector<int64_t> program,
                        std::vector<double> corrections, std::string reduction,
                        std::vector<int64_t> axes, bool keepdim,
                        double correction, int64_t size, bool paired) {
    at::AutoDispatchBelowADInplaceOrView guard;
    auto [result, state] = detail::media::event_reduce(
        x, y, comparison, settings, parameters, dtype, program, corrections,
        reduction, axes, keepdim, correction, size, paired);
    ctx->save_for_backward({x, y, state});
    ctx->mark_non_differentiable({state});
    auto metadata = detail::media::event_metadata(comparison, settings, dtype,
                                                  program, corrections);
    if (!metadata.differentiable || !result.is_floating_point())
      ctx->mark_non_differentiable({result});
    ctx->saved_data["comparison"] = comparison;
    ctx->saved_data["settings"] = settings;
    ctx->saved_data["parameters"] = parameters;
    ctx->saved_data["dtype"] = dtype;
    ctx->saved_data["program"] = program;
    ctx->saved_data["corrections"] = corrections;
    ctx->saved_data["reduction"] = reduction;
    ctx->saved_data["axes"] = axes;
    ctx->saved_data["keepdim"] = keepdim;
    ctx->saved_data["correction"] = correction;
    ctx->saved_data["size"] = size;
    ctx->saved_data["paired"] = paired;
    return {result, state};
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher image event derivatives are unsupported");
    auto v = ctx->get_saved_variables();
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow(
                             "torchpairwise::_image_event_reduce_backward", "")
                         .typed<detail::media::EventGradSignature>();
    auto [gx, gy] = op.call(
        gradients[0], v[0], v[1], v[2], ctx->saved_data["comparison"].toInt(),
        ctx->saved_data["settings"].toIntVector(),
        ctx->saved_data["parameters"].toDoubleVector(),
        ctx->saved_data["dtype"].toInt(),
        ctx->saved_data["program"].toIntVector(),
        ctx->saved_data["corrections"].toDoubleVector(),
        ctx->saved_data["reduction"].toStringRef(),
        ctx->saved_data["axes"].toIntVector(),
        ctx->saved_data["keepdim"].toBool(),
        ctx->saved_data["correction"].toDouble(),
        ctx->saved_data["size"].toInt(), ctx->saved_data["paired"].toBool());
    return {gx,       gy,       Tensor(), Tensor(), Tensor(),
            Tensor(), Tensor(), Tensor(), Tensor(), Tensor(),
            Tensor(), Tensor(), Tensor(), Tensor()};
  }
};
detail::media::EventOutputs anchored_event_autograd(
    const Tensor& x, const Tensor& y, int64_t comparison,
    at::IntArrayRef settings, at::ArrayRef<double> parameters, int64_t dtype,
    at::IntArrayRef program, at::ArrayRef<double> corrections,
    c10::string_view reduction, at::IntArrayRef axes, bool keepdim,
    double correction, int64_t size, bool paired) {
  auto v = ImageEventReductionFunction::apply(
      x, y, comparison, settings.vec(), parameters.vec(), dtype, program.vec(),
      corrections.vec(), std::string(reduction), axes.vec(), keepdim,
      correction, size, paired);
  return {v[0], v[1]};
}
detail::media::EventOutputs event_autograd(
    const Tensor& x, const Tensor& y, int64_t comparison,
    at::IntArrayRef settings, at::ArrayRef<double> parameters, int64_t dtype,
    at::IntArrayRef program, at::ArrayRef<double> corrections,
    c10::string_view reduction, at::IntArrayRef axes, bool keepdim,
    double correction, int64_t size, bool paired) {
  auto [result, state] = anchored_event_autograd(
      x, y, comparison, settings, parameters, dtype, program, corrections,
      reduction, axes, keepdim, correction, size, paired);
  return {result, detail::legacy_moment_state(state, 3, 1)};
}
class ImageEventTopKFunction
    : public torch::autograd::Function<ImageEventTopKFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, int64_t comparison,
                        std::vector<int64_t> settings,
                        std::vector<double> parameters, int64_t dtype,
                        std::vector<int64_t> program,
                        std::vector<double> corrections, int64_t k,
                        int64_t axis, bool largest, int64_t size) {
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_image_event_topk", "")
            .typed<detail::media::EventTopKSignature>();
    auto [values, indices] =
        op.call(x, y, comparison, settings, parameters, dtype, program,
                corrections, k, axis, largest, size);
    ctx->save_for_backward({x, y, indices});
    ctx->mark_non_differentiable({indices});
    auto metadata = detail::media::event_metadata(comparison, settings, dtype,
                                                  program, corrections);
    if (!metadata.differentiable || !values.is_floating_point())
      ctx->mark_non_differentiable({values});
    ctx->saved_data["comparison"] = comparison;
    ctx->saved_data["settings"] = settings;
    ctx->saved_data["parameters"] = parameters;
    ctx->saved_data["dtype"] = dtype;
    ctx->saved_data["program"] = program;
    ctx->saved_data["corrections"] = corrections;
    ctx->saved_data["axis"] = axis;
    ctx->saved_data["size"] = size;
    return {values, indices};
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher image event topk derivatives are unsupported");
    auto saved = ctx->get_saved_variables();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_image_event_topk_backward", "")
            .typed<detail::media::EventTopKGradSignature>();
    auto [gx, gy] = op.call(gradients[0], saved[0], saved[1], saved[2],
                            ctx->saved_data["comparison"].toInt(),
                            ctx->saved_data["settings"].toIntVector(),
                            ctx->saved_data["parameters"].toDoubleVector(),
                            ctx->saved_data["dtype"].toInt(),
                            ctx->saved_data["program"].toIntVector(),
                            ctx->saved_data["corrections"].toDoubleVector(),
                            ctx->saved_data["axis"].toInt(),
                            ctx->saved_data["size"].toInt());
    return {gx,       gy,       Tensor(), Tensor(), Tensor(), Tensor(),
            Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
  }
};
detail::media::EventOutputs event_topk_autograd(
    const Tensor& x, const Tensor& y, int64_t comparison,
    at::IntArrayRef settings, at::ArrayRef<double> parameters, int64_t dtype,
    at::IntArrayRef program, at::ArrayRef<double> corrections, int64_t k,
    int64_t axis, bool largest, int64_t size) {
  auto result = ImageEventTopKFunction::apply(
      x, y, comparison, settings.vec(), parameters.vec(), dtype, program.vec(),
      corrections.vec(), k, axis, largest, size);
  return {result[0], result[1]};
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_image_event_topk", TORCH_FN(event_topk_autograd));
  m.impl("_image_event_topk_backward", torch::CppFunction::makeFallthrough());
  m.impl("_image_event_reduce", TORCH_FN(event_autograd));
  m.impl("_image_event_reduce.anchored", TORCH_FN(anchored_event_autograd));
  m.impl("_image_event_reduce_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
}  // namespace torchpairwise::ops
