#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <limits>

#include "../sequence_comparisons_detail.h"
namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;

using OptionalDouble = c10::optional<double>;
using OptionalInt = c10::optional<int64_t>;
using GradSignature = std::tuple<Tensor, Tensor>(const Tensor&, const Tensor&,
                                                 const Tensor&, const Tensor&,
                                                 const Tensor&, int64_t, bool,
                                                 OptionalDouble, OptionalInt,
                                                 int64_t);

class SequenceComparisonFunction
    : public torch::autograd::Function<SequenceComparisonFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor lx, Tensor ly, int64_t mode,
                        bool squared, OptionalDouble gamma, OptionalInt window,
                        int64_t chunk) {
    ctx->save_for_backward({x, y, lx, ly});
    ctx->saved_data["mode"] = mode;
    ctx->saved_data["squared"] = squared;
    ctx->saved_data["gamma"] = gamma.value_or(0.);
    ctx->saved_data["window"] = window.value_or(-1);
    ctx->saved_data["chunk"] = chunk;
    at::AutoDispatchBelowADInplaceOrView guard;
    return detail::sequence::_sequence_compare(x, y, lx, ly, mode, squared,
                                               gamma, window, chunk);
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of native sequence replay are unsupported");
    auto values = ctx->get_saved_variables();
    OptionalDouble gamma =
        ctx->saved_data["gamma"].toDouble() > 0
            ? OptionalDouble(ctx->saved_data["gamma"].toDouble())
            : c10::nullopt;
    OptionalInt window = ctx->saved_data["window"].toInt() >= 0
                             ? OptionalInt(ctx->saved_data["window"].toInt())
                             : c10::nullopt;
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_sequence_compare_backward", "")
            .typed<GradSignature>();
    auto derivative = op.call(gradients[0], values[0], values[1], values[2],
                              values[3], ctx->saved_data["mode"].toInt(),
                              ctx->saved_data["squared"].toBool(), gamma,
                              window, ctx->saved_data["chunk"].toInt());
    return {std::get<0>(derivative),
            std::get<1>(derivative),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor()};
  }
};
Tensor _sequence_compare_autograd(const Tensor& x, const Tensor& y,
                                  const Tensor& lx, const Tensor& ly,
                                  int64_t mode, bool squared,
                                  OptionalDouble gamma, OptionalInt window,
                                  int64_t chunk) {
  return SequenceComparisonFunction::apply(x, y, lx, ly, mode, squared, gamma,
                                           window, chunk);
}
class SequenceCostFunction
    : public torch::autograd::Function<SequenceCostFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, Tensor lx, Tensor ly, OptionalDouble gamma,
                        OptionalInt window, int64_t chunk, bool self) {
    ctx->save_for_backward({x, y, lx, ly});
    ctx->saved_data["gamma"] = gamma.value_or(0.);
    ctx->saved_data["window"] = window.value_or(-1);
    ctx->saved_data["chunk"] = chunk;
    ctx->saved_data["self"] = self;
    at::AutoDispatchBelowADInplaceOrView guard;
    return detail::sequence::operator_cost(x, y, lx, ly, gamma, window, chunk,
                                           self);
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of sequence costs are unsupported");
    auto v = ctx->get_saved_variables();
    OptionalDouble gamma =
        ctx->saved_data["gamma"].toDouble() > 0
            ? OptionalDouble(ctx->saved_data["gamma"].toDouble())
            : c10::nullopt;
    OptionalInt window = ctx->saved_data["window"].toInt() >= 0
                             ? OptionalInt(ctx->saved_data["window"].toInt())
                             : c10::nullopt;
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow(
                "torchpairwise::_sequence_operator_cost_backward", "")
            .typed<detail::sequence::CostGradSignature>();
    auto g = op.call(gradients[0], v[0], v[1], v[2], v[3], gamma, window,
                     ctx->saved_data["chunk"].toInt(),
                     ctx->saved_data["self"].toBool());
    return {std::get<0>(g), std::get<1>(g), Tensor(), Tensor(),
            Tensor(),       Tensor(),       Tensor(), Tensor()};
  }
};
Tensor sequence_cost_autograd(const Tensor& x, const Tensor& y,
                              const Tensor& lx, const Tensor& ly,
                              OptionalDouble gamma, OptionalInt window,
                              int64_t chunk, bool self) {
  return SequenceCostFunction::apply(x, y, lx, ly, gamma, window, chunk, self);
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_sequence_operator_cost", TORCH_FN(sequence_cost_autograd));
  m.impl("_sequence_operator_cost_backward",
         torch::CppFunction::makeFallthrough());
  m.impl("_sequence_compare", TORCH_FN(_sequence_compare_autograd));
  m.impl("_sequence_compare_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
namespace detail::sequence {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
using OptionalDouble = c10::optional<double>;
using OptionalInt = c10::optional<int64_t>;

}  // namespace detail::sequence
}  // namespace torchpairwise::ops
