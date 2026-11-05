// Load with native_custom_metric.py, or link into a LibTorch application.
#include <torch/autograd.h>
#include <torch/library.h>
#include <torchpairwise.h>

namespace {
at::Tensor weighted_pairs(const at::Tensor& x, const at::Tensor& y,
                          at::TensorList parameters) {
  TORCH_CHECK(parameters.size() == 1,
              "Expected one shared feature-weight tensor");
  return ((x - y).square() * parameters[0]).sum(-1);
}
std::vector<at::Tensor> weighted_pairs_vjp(const at::Tensor& gradient,
                                           const at::Tensor& x,
                                           const at::Tensor& y,
                                           at::TensorList parameters) {
  auto delta = x - y;
  auto g = gradient.unsqueeze(-1);
  auto dx = 2 * g * delta * parameters[0];
  return {dx, -dx, (g * delta.square()).sum(0)};
}
at::Tensor native_streamed_mean(const at::Tensor& x, const at::Tensor& y,
                                const at::Tensor& weights) {
  namespace tp = torchpairwise::ops;
  tp::PairwiseOperationOptions options;
  options.chunk_size = 2;
  auto source =
      tp::custom_expression(x, y, "torchpairwise_example::weighted_pairs",
                            "torchpairwise_example::weighted_pairs_vjp",
                            {weights}, 1, c10::nullopt, options);
  return source.reduce("mean", -1);
}
at::Tensor native_compare(const at::Tensor& x,
                          const c10::optional<at::Tensor>& y,
                          const at::Tensor& weights, c10::string_view mode) {
  return torchpairwise::ops::custom_compare(
      x, y, {weights}, "torchpairwise_example::weighted_pairs",
      "torchpairwise_example::weighted_pairs_vjp", mode, 1, c10::nullopt, 7,
      true);
}

// Ordinary C++ callable API: no registration is needed for the metric itself.
// This wrapper is registered only so this example can be exercised from Python.
at::Tensor callable_compare(const at::Tensor& x,
                            const c10::optional<at::Tensor>& y,
                            const at::Tensor& weight, c10::string_view mode,
                            bool vectorized, bool explicit_vjp) {
  namespace tp = torchpairwise::ops;
  tp::CustomComparisonOptions options;
  options.parameters = {weight};
  options.symmetric = true;
  options.vectorized = vectorized;
  options.chunk_size = 7;
  auto forward = [vectorized](const at::Tensor& a, const at::Tensor& b,
                              at::TensorList parameters) {
    auto value = (a - b).square() * parameters[0];
    return vectorized ? value.sum(-1) : value.sum();
  };
  tp::CustomComparisonBackward vjp;
  if (explicit_vjp) {
    vjp = [vectorized](const at::Tensor& gradient, const at::Tensor& a,
                       const at::Tensor& b, at::TensorList parameters) {
      auto delta = a - b;
      auto g = vectorized ? gradient.unsqueeze(-1) : gradient;
      // Deliberate surrogate VJP verifies use of the supplied callback.
      auto dx = 5 * g * delta * parameters[0];
      auto dw = 3 * g * delta.square();
      return std::vector<at::Tensor>{dx, -dx, vectorized ? dw.sum(0) : dw};
    };
  }
  auto metric = tp::make_pairwise(forward, vjp, options);
  if (mode == "mean") {
    tp::PairwiseOperationOptions tiles;
    tiles.chunk_size = 2;
    // Return after the factory goes out of scope: backward still owns callback
    // access through the internal process-lifetime registration.
    return metric.expression(x, y, tiles).reduce("mean", -1);
  }
  return metric(x, y, mode);
}

at::Tensor simple_callable_mean(const at::Tensor& x, const at::Tensor& y) {
  static const auto metric = torchpairwise::ops::make_pairwise(
      [](const at::Tensor& a, const at::Tensor& b) {
        return (a - b).square().sum();
      });
  return metric.expression(x, y).reduce("mean", -1);
}

class SquaredDistanceFunction
    : public torch::autograd::Function<SquaredDistanceFunction> {
 public:
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            const at::Tensor& x, const at::Tensor& y) {
    ctx->save_for_backward({x, y});
    return (x - y).square().sum();
  }
  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& gradients) {
    auto inputs = ctx->get_saved_variables();
    auto dx = 5 * gradients[0] * (inputs[0] - inputs[1]);
    return {dx, -dx};  // Deliberate surrogate for callback verification.
  }
};
at::Tensor function_callable_mean(const at::Tensor& x, const at::Tensor& y) {
  static const auto metric = torchpairwise::ops::make_pairwise(
      [](const at::Tensor& a, const at::Tensor& b) {
        return SquaredDistanceFunction::apply(a, b);
      });
  return metric.expression(x, y).reduce("mean", -1);
}
TORCH_LIBRARY(torchpairwise_example, m) {
  m.def("weighted_pairs(Tensor x, Tensor y, Tensor[] parameters) -> Tensor");
  m.def(
      "weighted_pairs_vjp(Tensor grad, Tensor x, Tensor y, Tensor[] "
      "parameters) -> Tensor[]");
  m.def("streamed_mean(Tensor x, Tensor y, Tensor weights) -> Tensor");
  m.def("compare(Tensor x, Tensor? y, Tensor weights, str mode) -> Tensor");
  m.def(
      "callable_compare(Tensor x, Tensor? y, Tensor weight, str mode, "
      "bool vectorized, bool explicit_vjp) -> Tensor");
  m.def("simple_callable_mean(Tensor x, Tensor y) -> Tensor");
  m.def("function_callable_mean(Tensor x, Tensor y) -> Tensor");
}
// ATen operations dispatch to CPU/CUDA; no specialized device kernel is needed.
TORCH_LIBRARY_IMPL(torchpairwise_example, CompositeImplicitAutograd, m) {
  m.impl("weighted_pairs", TORCH_FN(weighted_pairs));
  m.impl("weighted_pairs_vjp", TORCH_FN(weighted_pairs_vjp));
  m.impl("streamed_mean", TORCH_FN(native_streamed_mean));
  m.impl("compare", TORCH_FN(native_compare));
  m.impl("callable_compare", TORCH_FN(callable_compare));
  m.impl("simple_callable_mean", TORCH_FN(simple_callable_mean));
  m.impl("function_callable_mean", TORCH_FN(function_callable_mean));
}
}  // namespace
