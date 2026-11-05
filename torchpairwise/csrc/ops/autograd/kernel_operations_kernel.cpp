#include <torch/autograd.h>
#include <torch/library.h>

#include "../kernel_operations_detail.h"

namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = torch::autograd::variable_list;
using namespace detail::kernel_operations;
class KernelFormFunction
    : public torch::autograd::Function<KernelFormFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor left, Tensor right,
                        c10::string_view kernel, int64_t size, Reals reals,
                        c10::optional<int64_t> degree) {
    ctx->save_for_backward({x, y, left, right});
    ctx->saved_data["kernel"] = std::string(kernel);
    ctx->saved_data["size"] = size;
    ctx->saved_data["reals"] = reals;
    ctx->saved_data["degree"] = degree ? c10::IValue(*degree) : c10::IValue();
    at::AutoDispatchBelowADInplaceOrView guard;
    return kernel_form(x, y, left, right, kernel, size, reals, degree);
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of native kernel forms are unsupported");
    auto v = ctx->get_saved_variables();
    auto degree =
        ctx->saved_data["degree"].isNone()
            ? c10::nullopt
            : c10::optional<int64_t>(ctx->saved_data["degree"].toInt());
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_kernel_form_backward", "")
            .typed<Grad>();
    auto [gx, gy, gl, gr] =
        op.call(gradients[0], v[0], v[1], v[2], v[3],
                ctx->saved_data["kernel"].toStringRef(),
                ctx->saved_data["size"].toInt(),
                ctx->saved_data["reals"].to<Reals>(), degree);
    return {gx, gy, gl, gr, Tensor(), Tensor(), Tensor(), Tensor()};
  }
};
Tensor kernel_form_autograd(const Tensor& x, const Tensor& y,
                            const Tensor& left, const Tensor& right,
                            c10::string_view kernel, int64_t size, Reals reals,
                            c10::optional<int64_t> degree) {
  return KernelFormFunction::apply(x, y, left, right, kernel, size, reals,
                                   degree);
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_kernel_form", TORCH_FN(kernel_form_autograd));
  m.impl("_kernel_form_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
}  // namespace torchpairwise::ops
