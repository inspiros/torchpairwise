#include <torch/autograd.h>
#include <torch/library.h>

#include "../pairwise_operations_detail.h"
#include "streaming_settings.h"
namespace torchpairwise::ops {
namespace {
using namespace detail::streaming;
class PairwiseFormFunction
    : public torch::autograd::Function<PairwiseFormFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor w, Tensor variance, Tensor inverse,
                        Tensor bins, Tensor projections, Tensor xl, Tensor yl,
                        Tensor payload, Tensor left, Tensor right, Settings s) {
    Tensors operands{x,    y,           w,  variance, inverse,
                     bins, projections, xl, yl,       payload},
        saved;
    for (int64_t slot = 0; slot < Slots; ++slot)
      if (s.has(slot))
        saved.push_back(operands[slot]);
    saved.insert(saved.end(), {left, right});
    ctx->save_for_backward(saved);
    save_settings(ctx, s);
    at::AutoDispatchBelowADInplaceOrView guard;
    auto result = pairwise_form(
        {x, y, w, variance, inverse, bins, projections, xl, yl, payload}, left,
        right, s);
    bool differentiable = left.requires_grad() || right.requires_grad();
    for (int64_t slot = 0; slot < Slots; ++slot)
      differentiable |=
          s.differentiable(slot) && operands[slot].requires_grad();
    if (!differentiable)
      ctx->mark_non_differentiable({result});
    return result;
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of native pairwise forms are unsupported");
    auto saved = ctx->get_saved_variables();
    auto s = load_settings(ctx);
    auto empty = at::empty({0}, saved[0].options());
    Tensors v(Slots, empty);
    size_t offset = 0;
    for (int64_t slot = 0; slot < Slots; ++slot)
      if (s.has(slot))
        v[slot] = saved[offset++];
    auto left = saved[offset++], right = saved[offset];
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_pairwise_form_backward", "")
            .typed<FormGradSignature>();
    auto g = op.call(gradients[0], v, left, right,
                     TORCHPAIRWISE_STREAMING_META_ARGS);
    g.push_back(Tensor());
    return g;
  }
};
Tensor pairwise_form_autograd(at::TensorList v, const Tensor& left,
                              const Tensor& right,
                              TORCHPAIRWISE_STREAMING_META_DECL) {
  auto s = settings(metric, operation, kernel, rank, k, largest, exclusion,
                    self_cosine, nan_first, size, dtype, mask, std::move(reals),
                    std::move(integers), std::move(flags), family);
  return PairwiseFormFunction::apply(v[X], v[Y], v[W], v[Variance], v[Inverse],
                                     v[Bins], v[Projections], v[XL], v[YL],
                                     v[Payload], left, right, s);
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_pairwise_form", TORCH_FN(pairwise_form_autograd));
  m.impl("_pairwise_form_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
}  // namespace torchpairwise::ops
