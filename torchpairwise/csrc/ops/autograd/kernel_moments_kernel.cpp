#include <torch/autograd.h>
#include <torch/library.h>

#include "../kernel_moments_detail.h"
namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = std::vector<Tensor>;
class KernelMomentFunction
    : public torch::autograd::Function<KernelMomentFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor origins, std::string name,
                        int64_t size, detail::kernel_moments::Reals reals,
                        c10::optional<int64_t> degree, bool second) {
    ctx->save_for_backward({x, y, origins});
    ctx->saved_data["second"] = second;
    ctx->saved_data["name"] = name;
    ctx->saved_data["size"] = size;
    ctx->saved_data["degree"] = degree ? c10::IValue(*degree) : c10::IValue();
    ctx->saved_data["reals"] = reals;
    at::AutoDispatchBelowADInplaceOrView guard;
    auto v = detail::kernel_moments::moments(x, y, origins, name, size, reals,
                                             degree, second);
    return {std::get<0>(v), std::get<1>(v), std::get<2>(v)};
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(
        !at::GradMode::is_enabled(),
        "Higher derivatives of kernel residual moments are unsupported");
    auto v = ctx->get_saved_variables();
    auto reals = ctx->saved_data["reals"].to<detail::kernel_moments::Reals>();
    c10::optional<int64_t> degree =
        ctx->saved_data["degree"].isNone()
            ? c10::nullopt
            : c10::optional<int64_t>(ctx->saved_data["degree"].toInt());
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_kernel_moments_backward", "")
            .typed<detail::kernel_moments::Grad>();
    auto g = op.call(gradients[0], gradients[1], gradients[2], v[0], v[1], v[2],
                     ctx->saved_data["name"].toStringRef(),
                     ctx->saved_data["size"].toInt(), reals, degree,
                     ctx->saved_data["second"].toBool());
    return {std::get<0>(g), std::get<1>(g), std::get<2>(g), Tensor(),
            Tensor(),       Tensor(),       Tensor(),       Tensor()};
  }
};
std::tuple<Tensor, Tensor, Tensor> kernel_moments_autograd(
    const Tensor& x, const Tensor& y, const Tensor& origins,
    c10::string_view name, int64_t size, detail::kernel_moments::Reals reals,
    c10::optional<int64_t> degree, bool second) {
  auto v = KernelMomentFunction::apply(x, y, origins, std::string(name), size,
                                       std::move(reals), degree, second);
  return {v[0], v[1], v[2]};
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_kernel_moments", TORCH_FN(kernel_moments_autograd));
  m.impl("_kernel_moments_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
}  // namespace torchpairwise::ops
