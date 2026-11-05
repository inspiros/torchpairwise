#include <torch/autograd.h>
#include <torch/library.h>

#include "../mmd_comparisons_detail.h"

namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = torch::autograd::variable_list;
using detail::mmd::KernelOptions;
void save_options(torch::autograd::AutogradContext* ctx,
                  const KernelOptions& o) {
  ctx->saved_data["point_chunk_size"] = o.chunk_size;
  ctx->saved_data["kernel"] = o.name;
  ctx->saved_data["gamma"] = o.gamma;
  ctx->saved_data["degree"] = o.degree;
  ctx->saved_data["coef0"] = o.coef0;
  ctx->saved_data["scale"] = o.scale;
  ctx->saved_data["alpha"] = o.alpha;
  ctx->saved_data["nu"] = o.nu;
  ctx->saved_data["statistic"] = o.statistic;
  ctx->saved_data["unbiased"] = o.unbiased;
  ctx->saved_data["exclude_diagonal"] = o.exclude_diagonal;
}
KernelOptions load_options(torch::autograd::AutogradContext* ctx) {
  return {ctx->saved_data["kernel"].toStringRef(),
          ctx->saved_data["gamma"].toDouble(),
          ctx->saved_data["degree"].toInt(),
          ctx->saved_data["coef0"].toDouble(),
          ctx->saved_data["scale"].toDouble(),
          ctx->saved_data["alpha"].toDouble(),
          ctx->saved_data["nu"].toDouble(),
          ctx->saved_data["statistic"].toBool(),
          ctx->saved_data["unbiased"].toBool(),
          ctx->saved_data["exclude_diagonal"].toBool(),
          ctx->saved_data["point_chunk_size"].toInt()};
}
class MMDPointFunction : public torch::autograd::Function<MMDPointFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor xv, Tensor yv, KernelOptions o) {
    ctx->save_for_backward({x, y, xv, yv});
    save_options(ctx, o);
    return detail::mmd::point_kernel(x, y, xv, yv, o);
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of native MMD replay are unsupported");
    auto v = ctx->get_saved_variables();
    auto x = v[0].detach().set_requires_grad(true),
         y = v[1].detach().set_requires_grad(true);
    at::AutoGradMode enable(true);
    auto value = detail::mmd::point_kernel(x, y, v[2], v[3], load_options(ctx));
    auto g =
        torch::autograd::grad({value}, {x, y}, gradients, false, false, true);
    return {g[0], g[1], Tensor(), Tensor(), Tensor()};
  }
};
class MMDComparisonFunction
    : public torch::autograd::Function<MMDComparisonFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor xl, Tensor yl, c10::string_view kernel,
                        int64_t mode, double gamma, int64_t degree,
                        double coef0, double scale, double alpha, double nu,
                        bool squared, bool statistic) {
    ctx->save_for_backward({x, y, xl, yl});
    KernelOptions o{
        std::string(kernel), gamma, degree, coef0, scale, alpha, nu};
    o.statistic = statistic;
    o.unbiased = statistic && squared;
    save_options(ctx, o);
    ctx->saved_data["mode"] = mode;
    ctx->saved_data["squared"] = squared;
    at::AutoDispatchBelowADInplaceOrView guard;
    return statistic
               ? detail::mmd::_mmd_statistic(x, y, xl, yl, o, mode)
               : detail::mmd::_mmd_compare(x, y, xl, yl, o, mode, squared);
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(
        !at::GradMode::is_enabled() || ctx->saved_data["statistic"].toBool(),
        "Higher derivatives of native MMD distances are unsupported");
    auto v = ctx->get_saved_variables();
    auto o = load_options(ctx);
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_mmd_compare_backward", "")
            .typed<detail::mmd::Grad>();
    static auto statistic_op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_mmd_compare_backward",
                               "statistic")
            .typed<detail::mmd::Grad>();
    auto [gx, gy] =
        (o.statistic ? statistic_op : op)
            .call(gradients[0], v[0], v[1], v[2], v[3], o.name,
                  ctx->saved_data["mode"].toInt(), o.gamma, o.degree, o.coef0,
                  o.scale, o.alpha, o.nu, ctx->saved_data["squared"].toBool());
    return {gx,       gy,       Tensor(), Tensor(), Tensor(),
            Tensor(), Tensor(), Tensor(), Tensor(), Tensor(),
            Tensor(), Tensor(), Tensor(), Tensor()};
  }
};
template <bool Statistic>
Tensor mmd_compare_autograd(const Tensor& x, const Tensor& y, const Tensor& xl,
                            const Tensor& yl, c10::string_view kernel,
                            int64_t mode, double gamma, int64_t degree,
                            double coef0, double scale, double alpha, double nu,
                            bool squared) {
  return MMDComparisonFunction::apply(x, y, xl, yl, kernel, mode, gamma, degree,
                                      coef0, scale, alpha, nu, squared,
                                      Statistic);
}
class MMDFeaturesFunction
    : public torch::autograd::Function<MMDFeaturesFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor length, c10::string_view kernel, double gamma,
                        int64_t degree, double coef0, double scale,
                        double alpha, double nu, bool statistic, bool unbiased,
                        int64_t chunk_size) {
    ctx->save_for_backward({x, length});
    KernelOptions o{
        std::string(kernel), gamma, degree, coef0, scale, alpha, nu};
    o.statistic = statistic;
    o.unbiased = unbiased;
    o.chunk_size = chunk_size;
    save_options(ctx, o);
    at::AutoDispatchBelowADInplaceOrView guard;
    auto result = detail::mmd::features(x, length, o);
    if (o.name == "linear" && !o.unbiased)
      ctx->mark_non_differentiable({result[1]});
    return result;
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of MMD features are unsupported");
    auto v = ctx->get_saved_variables();
    auto o = load_options(ctx);
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_mmd_features_backward", "")
            .typed<detail::mmd::FeatureGradSignature>();
    auto g = op.call(gradients, v[0], v[1], o.name, o.gamma, o.degree, o.coef0,
                     o.scale, o.alpha, o.nu, o.statistic, o.unbiased,
                     ctx->saved_data["point_chunk_size"].toInt());
    return {g,        Tensor(), Tensor(), Tensor(), Tensor(), Tensor(),
            Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
  }
};
Values mmd_features_autograd(const Tensor& x, const Tensor& length,
                             c10::string_view kernel, double gamma,
                             int64_t degree, double coef0, double scale,
                             double alpha, double nu, bool statistic,
                             bool unbiased, int64_t chunk_size) {
  return MMDFeaturesFunction::apply(x, length, kernel, gamma, degree, coef0,
                                    scale, alpha, nu, statistic, unbiased,
                                    chunk_size);
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_mmd_features", TORCH_FN(mmd_features_autograd));
  m.impl("_mmd_features_backward", torch::CppFunction::makeFallthrough());
  m.impl("_mmd_compare.statistic", TORCH_FN(mmd_compare_autograd<true>));
  m.impl("_mmd_compare_backward.statistic",
         torch::CppFunction::makeFallthrough());
  m.impl("_mmd_compare", TORCH_FN(mmd_compare_autograd<false>));
  m.impl("_mmd_compare_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
namespace detail::mmd {
Tensor point_autograd(Tensor x, Tensor y, Tensor xv, Tensor yv,
                      const KernelOptions& o) {
  return MMDPointFunction::apply(x, y, xv, yv, o);
}
}  // namespace detail::mmd
}  // namespace torchpairwise::ops
