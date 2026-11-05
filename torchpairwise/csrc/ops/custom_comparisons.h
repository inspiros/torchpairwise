#pragma once

#include <functional>

#include "expression_operations.h"

namespace torchpairwise::ops {
// Registered paired-tile protocol:
// forward(Tensor x, Tensor y, Tensor[] parameters) -> Tensor [P]
// backward(Tensor grad, Tensor x, Tensor y, Tensor[] parameters) -> Tensor[]
// Backward returns x/y gradients and shared (already pair-summed) parameter
// VJPs.
// With broadcast=true, Cartesian operands are [n,1,*sample]/[1,m,*sample]
// and return [n,m]. VJPs match each operand's passed shape.
TORCHPAIRWISE_API at::Tensor custom_compare(
    const at::Tensor& x, const c10::optional<at::Tensor>& y,
    at::TensorList parameters, c10::string_view forward,
    c10::string_view backward = "", c10::string_view mode = "all_pairs",
    int64_t sample_ndim = 1, c10::optional<int64_t> output_dtype = c10::nullopt,
    c10::optional<int64_t> chunk_size = c10::nullopt, bool symmetric = false,
    bool broadcast = false);
TORCHPAIRWISE_API ScalarExpression custom_expression(
    const at::Tensor& x, const c10::optional<at::Tensor>& y,
    c10::string_view forward, c10::string_view backward = "",
    at::TensorList parameters = {}, int64_t sample_ndim = 1,
    c10::optional<int64_t> output_dtype = c10::nullopt,
    const PairwiseOperationOptions& options = {}, bool broadcast = false);

// Scalar callbacks receive intact samples, like the Python factory. With
// vectorized=true they receive paired tiles [P,*sample], returning [P]; their
// parameter VJPs must then already be summed over P.
// broadcast=true passes broadcast tiles directly, without scalar adaptation.
using CustomComparisonForward = std::function<at::Tensor(
    const at::Tensor&, const at::Tensor&, at::TensorList)>;
using CustomComparisonBackward = std::function<std::vector<at::Tensor>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::TensorList)>;
struct CustomComparisonOptions {
  std::vector<at::Tensor> parameters;
  std::string mode = "all_pairs";
  int64_t sample_ndim = 1;
  c10::optional<int64_t> output_dtype = c10::nullopt;
  c10::optional<int64_t> chunk_size = c10::nullopt;
  bool symmetric = false;
  bool vectorized = false;
  bool broadcast = false;
};
class TORCHPAIRWISE_API CustomComparison {
 public:
  at::Tensor operator()(
      const at::Tensor& x,
      const c10::optional<at::Tensor>& y = c10::nullopt) const;
  at::Tensor operator()(const at::Tensor& x, const c10::optional<at::Tensor>& y,
                        c10::string_view mode) const {
    return compare(x, y, mode);
  }
  at::Tensor compare(const at::Tensor& x, const c10::optional<at::Tensor>& y,
                     c10::string_view mode) const;
  ScalarExpression expression(
      const at::Tensor& x, const c10::optional<at::Tensor>& y = c10::nullopt,
      const PairwiseOperationOptions& options = {}) const;

 private:
  friend TORCHPAIRWISE_API CustomComparison make_pairwise(
      CustomComparisonForward forward, CustomComparisonBackward backward,
      CustomComparisonOptions options);
  CustomComparison(std::string forward, std::string backward,
                   CustomComparisonOptions options);
  std::string forward_, backward_;
  CustomComparisonOptions options_;
};
// Construct once and reuse. Callback registrations/captures live for the
// process, so deferred expressions and backward survive destruction of the
// factory handle. They are native callbacks, not portable serialized code.
TORCHPAIRWISE_API CustomComparison make_pairwise(
    CustomComparisonForward forward, CustomComparisonBackward backward = {},
    CustomComparisonOptions options = {});
inline CustomComparison make_pairwise(CustomComparisonForward forward,
                                      CustomComparisonOptions options) {
  return make_pairwise(std::move(forward), {}, std::move(options));
}
// Parameter-free functions need only two arguments.
inline CustomComparison make_pairwise(
    std::function<at::Tensor(const at::Tensor&, const at::Tensor&)> forward,
    CustomComparisonOptions options = {}) {
  TORCH_CHECK(forward, "Custom forward must be callable");
  return make_pairwise(
      [forward = std::move(forward)](const at::Tensor& x, const at::Tensor& y,
                                     at::TensorList parameters) {
        TORCH_CHECK(parameters.empty(),
                    "Two-argument callbacks do not accept parameters");
        return forward(x, y);
      },
      {}, std::move(options));
}
}  // namespace torchpairwise::ops
