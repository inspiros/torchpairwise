#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <limits>

#include "../transport_comparisons_detail.h"
namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;

using Grad = std::tuple<Tensor, Tensor, Tensor>(const Tensor&, const Tensor&,
                                                const Tensor&, const Tensor&,
                                                const Tensor&, const Tensor&,
                                                int64_t, int64_t, double,
                                                int64_t, double, double,
                                                int64_t, bool, int64_t);
class TransportPointFunction
    : public torch::autograd::Function<TransportPointFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor xv, Tensor yv, int64_t kind,
                        int64_t kernel, double scale) {
    ctx->save_for_backward({x, y, xv, yv});
    ctx->saved_data["kind"] = kind;
    ctx->saved_data["kernel"] = kernel;
    ctx->saved_data["scale"] = scale;
    return detail::transport::point_kernel(x, y, xv, yv, kind, kernel, scale);
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(
        !at::GradMode::is_enabled(),
        "Higher derivatives of native transport replay are unsupported");
    auto v = ctx->get_saved_variables();
    auto x = v[0].detach().set_requires_grad(true),
         y = v[1].detach().set_requires_grad(true);
    at::AutoGradMode enable(true);
    auto value = detail::transport::point_kernel(
        x, y, v[2], v[3], ctx->saved_data["kind"].toInt(),
        ctx->saved_data["kernel"].toInt(), ctx->saved_data["scale"].toDouble());
    auto grads =
        torch::autograd::grad({value}, {x, y}, gradients, false, false, true);
    return {grads[0], grads[1], Tensor(), Tensor(),
            Tensor(), Tensor(), Tensor()};
  }
};
class TransportComparisonFunction
    : public torch::autograd::Function<TransportComparisonFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor xl, Tensor yl, Tensor directions,
                        int64_t kind, int64_t mode, double p, int64_t kernel,
                        double scale, double epsilon, int64_t iterations,
                        bool squared, int64_t size) {
    ctx->save_for_backward({x, y, xl, yl, directions});
    ctx->saved_data["kind"] = kind;
    ctx->saved_data["mode"] = mode;
    ctx->saved_data["p"] = p;
    ctx->saved_data["kernel"] = kernel;
    ctx->saved_data["scale"] = scale;
    ctx->saved_data["epsilon"] = epsilon;
    ctx->saved_data["iterations"] = iterations;
    ctx->saved_data["squared"] = squared;
    ctx->saved_data["size"] = size;
    at::AutoDispatchBelowADInplaceOrView guard;
    return detail::transport::_transport_compare(
        x, y, xl, yl, directions, kind, mode, p, kernel, scale, epsilon,
        iterations, squared, size);
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(
        !at::GradMode::is_enabled(),
        "Higher derivatives of native transport replay are unsupported");
    auto v = ctx->get_saved_variables();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_transport_compare_backward", "")
            .typed<Grad>();
    auto result = op.call(
        gradients[0], v[0], v[1], v[2], v[3], v[4],
        ctx->saved_data["kind"].toInt(), ctx->saved_data["mode"].toInt(),
        ctx->saved_data["p"].toDouble(), ctx->saved_data["kernel"].toInt(),
        ctx->saved_data["scale"].toDouble(),
        ctx->saved_data["epsilon"].toDouble(),
        ctx->saved_data["iterations"].toInt(),
        ctx->saved_data["squared"].toBool(), ctx->saved_data["size"].toInt());
    return {std::get<0>(result),
            std::get<1>(result),
            Tensor(),
            Tensor(),
            std::get<2>(result),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor()};
  }
};
Tensor _transport_compare_autograd(const Tensor& x, const Tensor& y,
                                   const Tensor& xl, const Tensor& yl,
                                   const Tensor& directions, int64_t kind,
                                   int64_t mode, double p, int64_t kernel,
                                   double scale, double epsilon,
                                   int64_t iterations, bool squared,
                                   int64_t size) {
  return TransportComparisonFunction::apply(x, y, xl, yl, directions, kind,
                                            mode, p, kernel, scale, epsilon,
                                            iterations, squared, size);
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_transport_compare", TORCH_FN(_transport_compare_autograd));
  m.impl("_transport_compare_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
namespace detail::transport {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
Tensor point_autograd(Tensor x, Tensor y, Tensor xv, Tensor yv, int64_t kind,
                      int64_t kernel, double scale) {
  return TransportPointFunction::apply(x, y, xv, yv, kind, kernel, scale);
}
}  // namespace detail::transport
}  // namespace torchpairwise::ops
