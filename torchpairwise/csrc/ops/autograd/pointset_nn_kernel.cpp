#include <ATen/ATen.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Pair = std::tuple<Tensor, Tensor>;
using Values = torch::autograd::variable_list;
using Forward = Pair(const Tensor&, const Tensor&, const Tensor&,
                     const Tensor&);
using Backward = Pair(const Tensor&, const Tensor&, const Tensor&,
                      const Tensor&, const Tensor&);

class PointsetNearestFunction
    : public torch::autograd::Function<PointsetNearestFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor xl, Tensor yl) {
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("torchpairwise::_pointset_nn", "")
                         .typed<Forward>();
    auto result = op.call(x, y, xl, yl);
    ctx->save_for_backward({x, y, xl, std::get<1>(result)});
    ctx->mark_non_differentiable({std::get<1>(result)});
    return {std::get<0>(result), std::get<1>(result)};
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of native point-set nearest search are "
                "unsupported");
    auto saved = ctx->get_saved_variables();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_pointset_nn_backward", "")
            .typed<Backward>();
    auto result = op.call(gradients[0], saved[0], saved[1], saved[2], saved[3]);
    return {std::get<0>(result), std::get<1>(result), Tensor(), Tensor()};
  }
};
Pair _pointset_nn_autograd(const Tensor& x, const Tensor& y, const Tensor& xl,
                           const Tensor& yl) {
  auto result = PointsetNearestFunction::apply(x, y, xl, yl);
  return {result[0], result[1]};
}

TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_pointset_nn", TORCH_FN(_pointset_nn_autograd));
  m.impl("_pointset_nn_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
}  // namespace torchpairwise::ops
