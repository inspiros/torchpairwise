#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <limits>

#include "../vector_comparisons_detail.h"
namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::array<Tensor, 3>;

using GradSignature = std::tuple<Tensor, Tensor, Tensor>(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, int64_t,
    int64_t, double, double, double, double, bool);
class VectorFeatureFunction
    : public torch::autograd::Function<VectorFeatureFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, Tensor a,
                        Tensor b, Tensor c, Tensor d, Tensor e, Tensor f,
                        Tensor scale, int64_t kind, bool paired, double alpha,
                        double limit) {
    ctx->save_for_backward({a, b, c, d, e, f, scale});
    ctx->saved_data["kind"] = kind;
    ctx->saved_data["paired"] = paired;
    ctx->saved_data["alpha"] = alpha;
    ctx->saved_data["limit"] = limit;
    auto values = detail::vector::feature_kernel({a, b, c}, {d, e, f}, scale,
                                                 kind, paired, alpha, limit);
    Values flags;
    for (auto& t : values)
      if (!t.is_floating_point())
        flags.push_back(t);
    ctx->mark_non_differentiable(flags);
    return values;
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(
        !at::GradMode::is_enabled(),
        "Higher derivatives of native comparison replay are unsupported");
    auto local = ctx->get_saved_variables();
    Values inputs;
    std::vector<int64_t> positions;
    for (int64_t i = 0; i < 7; ++i)
      if (local[i].defined()) {
        local[i] = local[i].detach();
        if (ctx->needs_input_grad(i) && local[i].is_floating_point()) {
          local[i].set_requires_grad(true);
          inputs.push_back(local[i]);
          positions.push_back(i);
        }
      }
    Values result(11);
    if (inputs.empty())
      return result;
    at::AutoGradMode enable(true);
    auto outputs = detail::vector::feature_kernel(
        {local[0], local[1], local[2]}, {local[3], local[4], local[5]},
        local[6], ctx->saved_data["kind"].toInt(),
        ctx->saved_data["paired"].toBool(), ctx->saved_data["alpha"].toDouble(),
        ctx->saved_data["limit"].toDouble());
    Values active, cotangents;
    for (size_t i = 0; i < outputs.size(); ++i)
      if (outputs[i].requires_grad() && gradients[i].defined()) {
        active.push_back(outputs[i]);
        cotangents.push_back(gradients[i]);
      }
    if (active.empty())
      return result;
    auto grads =
        torch::autograd::grad(active, inputs, cotangents, false, false, true);
    for (size_t i = 0; i < positions.size(); ++i)
      result[positions[i]] = grads[i];
    return result;
  }
};
class VectorComparisonFunction
    : public torch::autograd::Function<VectorComparisonFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor grid, int64_t kind, int64_t mode,
                        double base, double length_scale, double alpha,
                        double nu, bool squared) {
    ctx->save_for_backward({x, y, grid});
    ctx->saved_data["kind"] = kind;
    ctx->saved_data["mode"] = mode;
    ctx->saved_data["base"] = base;
    ctx->saved_data["length_scale"] = length_scale;
    ctx->saved_data["alpha"] = alpha;
    ctx->saved_data["nu"] = nu;
    ctx->saved_data["squared"] = squared;
    at::AutoDispatchBelowADInplaceOrView guard;
    return detail::vector::_vector_compare(x, y, grid, kind, mode, base,
                                           length_scale, alpha, nu, squared);
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    auto saved = ctx->get_saved_variables();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_vector_compare_backward", "")
            .typed<GradSignature>();
    auto result = op.call(
        gradients[0], saved[0], saved[1], saved[2],
        ctx->saved_data["kind"].toInt(), ctx->saved_data["mode"].toInt(),
        ctx->saved_data["base"].toDouble(),
        ctx->saved_data["length_scale"].toDouble(),
        ctx->saved_data["alpha"].toDouble(), ctx->saved_data["nu"].toDouble(),
        ctx->saved_data["squared"].toBool());
    return {std::get<0>(result),
            std::get<1>(result),
            std::get<2>(result),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor()};
  }
};
Tensor _vector_compare_autograd(const Tensor& x, const Tensor& y,
                                const Tensor& grid, int64_t kind, int64_t mode,
                                double base, double length_scale, double alpha,
                                double nu, bool squared) {
  return VectorComparisonFunction::apply(x, y, grid, kind, mode, base,
                                         length_scale, alpha, nu, squared);
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_vector_compare", TORCH_FN(_vector_compare_autograd));
  m.impl("_vector_compare_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
namespace detail::vector {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::array<Tensor, 3>;
Values feature_autograd(Tensor a, Tensor b, Tensor c, Tensor d, Tensor e,
                        Tensor f, Tensor scale, int64_t kind, bool paired,
                        double alpha, double limit) {
  return VectorFeatureFunction::apply(a, b, c, d, e, f, scale, kind, paired,
                                      alpha, limit);
}
}  // namespace detail::vector
}  // namespace torchpairwise::ops
