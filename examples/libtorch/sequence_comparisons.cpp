// Link against LibTorch and the matching torchpairwise shared library.
// All recurrence execution, mode assembly and first derivatives are native.
#include <ATen/ATen.h>
#include <torch/csrc/autograd/autograd.h>

#include <iostream>

#include "torchpairwise.h"

void check(at::Device device) {
  using torchpairwise::ops::sequence_comparisons;
  using torchpairwise::ops::token_comparisons;
  auto options = at::TensorOptions().device(device).dtype(at::kDouble);
  auto x = at::rand({3, 4, 2}, options).set_requires_grad(true);
  auto y = at::rand({3, 4, 2}, options).set_requires_grad(true);
  // A zero-width band admits exactly the aligned path; its independent
  // formula checks both hard and debiased soft costs and their derivatives.
  for (bool soft : {false, true}) {
    c10::optional<double> gamma =
        soft ? c10::optional<double>(.7) : c10::nullopt;
    auto actual = sequence_comparisons(x, y, "paired", !soft, gamma,
                                       c10::nullopt, c10::nullopt, 0, 2);
    auto expected = (x - y).square().sum({-1, -2});
    TORCH_CHECK(at::allclose(actual, expected),
                "Sequence aligned-path values differ");
    auto gradients = torch::autograd::grad({actual.sum()}, {x, y});
    auto reference = torch::autograd::grad({expected.sum()}, {x, y});
    for (size_t i = 0; i < gradients.size(); ++i)
      TORCH_CHECK(at::allclose(gradients[i], reference[i]),
                  "Sequence gradients differ");
    auto full =
        sequence_comparisons(x, c10::nullopt, "all_pairs", !soft, gamma);
    auto compact =
        sequence_comparisons(x, c10::nullopt, "condensed", !soft, gamma);
    auto triangle = at::stack({full.select(0, 0).select(0, 1),
                               full.select(0, 0).select(0, 2),
                               full.select(0, 1).select(0, 2)});
    TORCH_CHECK(at::allclose(compact, triangle),
                "Sequence condensed ordering differs");
    TORCH_CHECK(full.diagonal().eq(0).all().item<bool>(),
                "Self costs must be exactly zero");
  }
  auto tokens = at::tensor({int64_t(1), int64_t(2), int64_t(3), int64_t(1),
                            int64_t(3), int64_t(2)},
                           options.dtype(at::kLong))
                    .reshape({2, 3});
  auto length = at::tensor({int64_t(3), int64_t(0)}, tokens.options());
  for (int64_t substitution : {1, 2}) {
    auto full = token_comparisons(tokens, c10::nullopt, "all_pairs",
                                  substitution, length);
    auto expected = at::tensor({int64_t(0), int64_t(3), int64_t(3), int64_t(0)},
                               tokens.options())
                        .reshape({2, 2});
    TORCH_CHECK(at::equal(full, expected), "Ragged token values differ");
    auto compact = token_comparisons(tokens, c10::nullopt, "condensed",
                                     substitution, length);
    TORCH_CHECK(compact.item<int64_t>() == 3,
                "Token condensed ordering differs");
    auto paired = token_comparisons(tokens, tokens, "paired", substitution,
                                    length, length);
    TORCH_CHECK(paired.eq(0).all().item<bool>(),
                "Paired token self costs differ");
  }
  std::cout << device
            << ": native sequence values/gradients and token modes passed\n";
}
int main() {
  check(at::Device(at::kCPU));
  if (at::hasCUDA())
    check(at::Device(at::kCUDA, 0));
}
