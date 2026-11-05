// Link against LibTorch and the matching torchpairwise shared library.
// This executable uses no Python interpreter or Python registration code.
#include <torch/csrc/autograd/autograd.h>
#include <torch/script.h>

#include <iostream>

#include "torchpairwise.h"

int main(int argc, char** argv) {
  using namespace torchpairwise::ops;
  auto x = (at::rand({3, 4}, at::kDouble) + .3).set_requires_grad(true);
  auto y = (at::rand({5, 4}, at::kDouble) + .3).set_requires_grad(true);
  auto p = x / x.sum(-1, true), q = y / y.sum(-1, true);
  auto reference =
      (p.unsqueeze(-2) * (p.log().unsqueeze(-2) - q.log().unsqueeze(-3)))
          .sum(-1);
  auto actual = kl_divergences(x, y);
  TORCH_CHECK(at::allclose(actual, reference, 1.e-10, 1.e-12),
              "KL values differ");
  auto grads = torch::autograd::grad({actual.sum()}, {x, y});
  auto expected_grads = torch::autograd::grad({reference.sum()}, {x, y});
  for (size_t i = 0; i < grads.size(); ++i)
    TORCH_CHECK(at::allclose(grads[i], expected_grads[i], 1.e-10, 1.e-12),
                "Native KL gradient differs");
  TORCH_CHECK(paired_kl_divergences(x, y.narrow(0, 0, 3)).sizes() ==
                  at::IntArrayRef({3}),
              "Paired shape differs");
  TORCH_CHECK(
      condensed_symmetric_kl_divergences(x).sizes() == at::IntArrayRef({3}),
      "Condensed shape differs");
  VectorComparisonOptions options;
  options.length_scale = .7;
  options.alpha = .8;
  auto kernel =
      vector_comparisons(x, y, "rational_quadratic", "all_pairs", options);
  TORCH_CHECK(at::equal(kernel, rational_quadratic_kernel(x, y, .7, .8)),
              "Named/generic kernel differs");
  bool rejected = false;
  try {
    vector_comparisons(x, y, "angular", "all_pairs", options);
  } catch (const c10::Error&) {
    rejected = true;
  }
  TORCH_CHECK(rejected, "Unused options must be rejected");
  if (argc > 1) {
    auto module = torch::jit::load(argv[1]);
    auto scripted = module.forward({x, y}).toTensor();
    TORCH_CHECK(at::allclose(scripted, kl_divergences(x, y)),
                "Serialized KL values differ");
    auto scripted_grads = torch::autograd::grad({scripted.sum()}, {x, y});
    auto native_grads =
        torch::autograd::grad({kl_divergences(x, y).sum()}, {x, y});
    for (size_t i = 0; i < grads.size(); ++i)
      TORCH_CHECK(at::allclose(scripted_grads[i], native_grads[i]),
                  "Serialized native gradient differs");
  }
  std::cout << "Standalone LibTorch values, gradients, modes and option "
               "validation passed\n";
}
