#include <torch/autograd.h>
#include <torch/library.h>

#include "streaming_settings.h"
namespace torchpairwise::ops {
namespace {
using namespace detail::streaming;
class StreamingReductionFunction
    : public torch::autograd::Function<StreamingReductionFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor w, Tensor variance, Tensor inverse,
                        Tensor bins, Tensor projections, Tensor xl, Tensor yl,
                        Tensor payload, Settings s) {
    Tensors v{x, y, w, variance, inverse, bins, projections, xl, yl, payload},
        saved;
    for (int64_t slot = 0; slot < Slots; ++slot)
      if (s.has(slot))
        saved.push_back(v[slot]);
    save_settings(ctx, s);
    at::AutoDispatchBelowADInplaceOrView guard;
    auto result = _streaming_reduce_legacy(v, s);
    if (s.kind() == "neighbors")
      saved.push_back(std::get<1>(result));
    if (s.kind() == "prod") {
      saved.push_back(std::get<0>(result));
      saved.push_back(std::get<2>(result));
      saved.push_back(std::get<3>(result));
    }
    ctx->save_for_backward(saved);
    bool differentiable = false;
    for (int64_t slot = 0; slot < Slots; ++slot)
      differentiable |= s.differentiable(slot) && v[slot].requires_grad();
    ctx->mark_non_differentiable(
        {std::get<1>(result), std::get<2>(result), std::get<3>(result)});
    if (!differentiable || s.kind() == "all" || s.kind() == "any")
      ctx->mark_non_differentiable({std::get<0>(result), std::get<1>(result),
                                    std::get<2>(result), std::get<3>(result)});
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
            std::get<3>(result)};
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    auto s = load_settings(ctx);
    auto saved = ctx->get_saved_variables();
    auto empty = at::empty({0}, saved[0].options());
    Tensors v(Slots, empty);
    size_t offset = 0;
    for (int64_t slot = 0; slot < Slots; ++slot)
      if (s.has(slot))
        v[slot] = saved[offset++];
    Tensor result = empty, index = empty, zeros = empty, nonzero = empty;
    if (s.kind() == "neighbors")
      index = saved[offset++];
    if (s.kind() == "prod") {
      result = saved[offset++];
      zeros = saved[offset++];
      nonzero = saved[offset++];
    }
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_streaming_reduce_backward", "")
            .typed<GradSignature>();
    auto derivatives = op.call(gradients[0], v, result, index, zeros, nonzero,
                               TORCHPAIRWISE_STREAMING_META_ARGS);
    derivatives.push_back(Tensor());
    return derivatives;
  }
};
Outputs _streaming_reduce_autograd(at::TensorList v,
                                   TORCHPAIRWISE_STREAMING_META_DECL) {
  auto s = settings(metric, operation, kernel, rank, k, largest, exclusion,
                    self_cosine, nan_first, size, dtype, mask, std::move(reals),
                    std::move(integers), std::move(flags), family);
  auto values = StreamingReductionFunction::apply(
      v[X], v[Y], v[W], v[Variance], v[Inverse], v[Bins], v[Projections], v[XL],
      v[YL], v[Payload], s);
  return {values[0], values[1], values[2], values[3]};
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_streaming_reduce", TORCH_FN(_streaming_reduce_autograd));
  m.impl("_streaming_reduce_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
}  // namespace torchpairwise::ops
