#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <limits>

#include "../builtin_comparisons_detail.h"
namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;

using Grad = std::tuple<Tensor, Tensor, Tensor, Tensor>;
using Backward = Grad(const Tensor&, const Tensor&, const Tensor&,
                      const Tensor&, const Tensor&, c10::string_view, int64_t,
                      bool, double, double, int64_t, double, double, bool);
class LegacyComparisonFunction
    : public torch::autograd::Function<LegacyComparisonFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor v, Tensor vi, std::string name,
                        int64_t mode, bool kernel, double p, double base,
                        int64_t degree, double gamma, double coef0, bool self) {
    ctx->save_for_backward({x, y, v, vi});
    ctx->saved_data["name"] = name;
    ctx->saved_data["mode"] = mode;
    ctx->saved_data["kernel"] = kernel;
    ctx->saved_data["p"] = p;
    ctx->saved_data["base"] = base;
    ctx->saved_data["degree"] = degree;
    ctx->saved_data["gamma"] = gamma;
    ctx->saved_data["coef0"] = coef0;
    ctx->saved_data["self"] = self;
    at::AutoDispatchBelowADInplaceOrView guard;
    return detail::builtin::_legacy_compare(x, y, v, vi, name, mode, kernel, p,
                                            base, degree, gamma, coef0, self);
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& g) {
    auto a = ctx->get_saved_variables();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_legacy_compare_backward", "")
            .typed<Backward>();
    auto result = op.call(
        g[0], a[0], a[1], a[2], a[3], ctx->saved_data["name"].toStringRef(),
        ctx->saved_data["mode"].toInt(), ctx->saved_data["kernel"].toBool(),
        ctx->saved_data["p"].toDouble(), ctx->saved_data["base"].toDouble(),
        ctx->saved_data["degree"].toInt(), ctx->saved_data["gamma"].toDouble(),
        ctx->saved_data["coef0"].toDouble(), ctx->saved_data["self"].toBool());
    return {std::get<0>(result),
            std::get<1>(result),
            std::get<2>(result),
            std::get<3>(result),
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
Tensor _legacy_compare_autograd(const Tensor& x, const Tensor& y,
                                const Tensor& v, const Tensor& vi,
                                c10::string_view name, int64_t mode,
                                bool kernel, double p, double base,
                                int64_t degree, double gamma, double coef0,
                                bool self) {
  if (!x.is_floating_point() ||
      (!kernel && detail::builtin::contains_name(
                      name, {"hamming", "jaccard", "kulsinski", "kulczynski1",
                             "yule", "dice", "rogerstanimoto", "russellrao",
                             "sokalmichener", "sokalsneath"}))) {
    at::AutoDispatchBelowADInplaceOrView guard;
    return detail::builtin::_legacy_compare(x, y, v, vi, name, mode, kernel, p,
                                            base, degree, gamma, coef0, self);
  }
  return LegacyComparisonFunction::apply(x, y, v, vi, std::string(name), mode,
                                         kernel, p, base, degree, gamma, coef0,
                                         self);
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_legacy_compare", TORCH_FN(_legacy_compare_autograd));
  m.impl("_legacy_compare_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
namespace detail::builtin {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;

}  // namespace detail::builtin
}  // namespace torchpairwise::ops
