#include <torch/autograd.h>
#include <torch/library.h>

#include "../histogram_intersection.h"
namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = torch::autograd::variable_list;
class HistogramIntersectionFunction
    : public torch::autograd::Function<HistogramIntersectionFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y) {
    at::AutoDispatchBelowADInplaceOrView guard;
    auto result = _histogram_intersection(x, y);
    ctx->save_for_backward({x, y});
    return result;
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    auto saved = ctx->get_saved_variables();
    auto result = detail::_histogram_intersection_backward(gradients[0],
                                                           saved[0], saved[1]);
    return {std::get<0>(result), std::get<1>(result)};
  }
};
at::Tensor _histogram_intersection_autograd(const Tensor& x, const Tensor& y) {
  auto result = HistogramIntersectionFunction::apply(x, y);
  return result;
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_histogram_intersection", TORCH_FN(_histogram_intersection_autograd));
  m.impl("_histogram_intersection_backward",
         torch::CppFunction::makeFallthrough());
}
}  // namespace
}  // namespace torchpairwise::ops
