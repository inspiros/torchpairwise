#include <torch/autograd.h>
#include <torch/library.h>

#include "../nan_euclidean.h"
namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = torch::autograd::variable_list;
class NanSquaredEuclideanFunction
    : public torch::autograd::Function<NanSquaredEuclideanFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y) {
    at::AutoDispatchBelowADInplaceOrView guard;
    auto result = _nan_sqeuclidean(x, y);
    ctx->save_for_backward({x, y, std::get<1>(result)});
    ctx->mark_non_differentiable({std::get<1>(result)});
    return {std::get<0>(result), std::get<1>(result)};
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    auto saved = ctx->get_saved_variables();
    auto result = detail::_nan_sqeuclidean_backward(gradients[0], saved[0],
                                                    saved[1], saved[2]);
    return {std::get<0>(result), std::get<1>(result)};
  }
};
std::tuple<at::Tensor, at::Tensor> _nan_sqeuclidean_autograd(const Tensor& x,
                                                             const Tensor& y) {
  auto result = NanSquaredEuclideanFunction::apply(x, y);
  return {result[0], result[1]};
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_nan_sqeuclidean", TORCH_FN(_nan_sqeuclidean_autograd));
  m.impl("_nan_sqeuclidean_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
}  // namespace torchpairwise::ops
