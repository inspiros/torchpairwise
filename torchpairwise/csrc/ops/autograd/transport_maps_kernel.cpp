#include <torch/autograd.h>
#include <torch/library.h>

#include "../transport_maps_detail.h"

namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = std::vector<Tensor>;
class TransportMapReductionFunction
    : public torch::autograd::Function<TransportMapReductionFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor logw, int64_t p, double epsilon,
                        int64_t size, bool barycentric) {
    at::AutoDispatchBelowADInplaceOrView guard;
    auto result = detail::transport_maps::reduce(x, y, logw, p, epsilon, size,
                                                 barycentric);
    ctx->save_for_backward({x, y, logw, result});
    ctx->saved_data["p"] = p;
    ctx->saved_data["epsilon"] = epsilon;
    ctx->saved_data["size"] = size;
    ctx->saved_data["barycentric"] = barycentric;
    return result;
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of streamed Sinkhorn maps are unsupported");
    auto v = ctx->get_saved_variables();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_transport_map_reduce_backward",
                               "")
            .typed<detail::transport_maps::Grad>();
    auto g = op.call(
        gradients[0], v[0], v[1], v[2], v[3], ctx->saved_data["p"].toInt(),
        ctx->saved_data["epsilon"].toDouble(), ctx->saved_data["size"].toInt(),
        ctx->saved_data["barycentric"].toBool());
    return {std::get<0>(g), std::get<1>(g), std::get<2>(g), Tensor(),
            Tensor(),       Tensor(),       Tensor()};
  }
};
Tensor transport_map_reduce_autograd(const Tensor& x, const Tensor& y,
                                     const Tensor& logw, int64_t p,
                                     double epsilon, int64_t size,
                                     bool barycentric) {
  return TransportMapReductionFunction::apply(x, y, logw, p, epsilon, size,
                                              barycentric);
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_transport_map_reduce", TORCH_FN(transport_map_reduce_autograd));
  m.impl("_transport_map_reduce_backward",
         torch::CppFunction::makeFallthrough());
}
}  // namespace
}  // namespace torchpairwise::ops
