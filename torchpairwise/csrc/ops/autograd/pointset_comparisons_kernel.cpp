#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <limits>

#include "../pointset_comparisons_detail.h"
namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;

using GradSignature = Pair(const Tensor&, const Tensor&, const Tensor&,
                           const Tensor&, const Tensor&, int64_t, int64_t, bool,
                           int64_t, bool);
class PointsetComparisonFunction
    : public torch::autograd::Function<PointsetComparisonFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor xl, Tensor yl, int64_t kind,
                        int64_t mode, bool squared, int64_t size, bool legacy) {
    ctx->save_for_backward({x, y, xl, yl});
    ctx->saved_data["kind"] = kind;
    ctx->saved_data["mode"] = mode;
    ctx->saved_data["squared"] = squared;
    ctx->saved_data["size"] = size;
    ctx->saved_data["legacy"] = legacy;
    at::AutoDispatchBelowADInplaceOrView guard;
    auto result = detail::pointset::_pointset_compare(x, y, xl, yl, kind, mode,
                                                      squared, size, legacy);
    ctx->mark_non_differentiable({std::get<1>(result), std::get<2>(result)});
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(
        !at::GradMode::is_enabled(),
        "Higher derivatives of native point-set replay are unsupported");
    auto saved = ctx->get_saved_variables();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_pointset_compare_backward", "")
            .typed<GradSignature>();
    auto result = op.call(
        gradients[0], saved[0], saved[1], saved[2], saved[3],
        ctx->saved_data["kind"].toInt(), ctx->saved_data["mode"].toInt(),
        ctx->saved_data["squared"].toBool(), ctx->saved_data["size"].toInt(),
        ctx->saved_data["legacy"].toBool());
    return {std::get<0>(result),
            std::get<1>(result),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor()};
  }
};
Triple _pointset_compare_autograd(const Tensor& x, const Tensor& y,
                                  const Tensor& xl, const Tensor& yl,
                                  int64_t kind, int64_t mode, bool squared,
                                  int64_t size, bool legacy) {
  auto result = PointsetComparisonFunction::apply(x, y, xl, yl, kind, mode,
                                                  squared, size, legacy);
  return {result[0], result[1], result[2]};
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_pointset_compare", TORCH_FN(_pointset_compare_autograd));
  m.impl("_pointset_compare_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
namespace detail::pointset {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;

}  // namespace detail::pointset
}  // namespace torchpairwise::ops
