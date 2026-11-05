// No Python initialization: native transport and complete vector/cloud
// selector.
#include <ATen/ATen.h>
#include <torch/csrc/autograd/autograd.h>

#include <iostream>

#include "torchpairwise.h"

int main() {
  for (auto device : {at::Device(at::kCPU), at::Device(at::kCUDA, 0)}) {
    if (device.is_cuda() && !at::globalContext().hasCUDA())
      continue;
    auto options = at::TensorOptions().device(device).dtype(at::kDouble);
    auto x = at::rand({3, 4, 2}, options).set_requires_grad(true);
    auto y = at::rand({3, 5, 2}, options).set_requires_grad(true);
    for (auto metric :
         {"energy", "mmd", "sliced_wasserstein", "sinkhorn", "wasserstein"}) {
      torchpairwise::ops::BuiltinComparisonOptions params;
      if (std::string(metric) == "sliced_wasserstein")
        params.projections = at::eye(2, options);
      if (std::string(metric) == "sinkhorn")
        params.max_iter = 5;
      auto full = torchpairwise::ops::builtin_comparisons(
          x, y, metric, "all_pairs", false, params);
      auto paired = torchpairwise::ops::builtin_comparisons(
          x, y, metric, "paired", false, params);
      TORCH_CHECK(at::allclose(paired, full.diagonal(0, -2, -1)),
                  "paired values differ");
      auto grads = torch::autograd::grad({full.sum()}, {x, y});
      TORCH_CHECK(grads[0].isfinite().all().item<bool>() &&
                      grads[1].isfinite().all().item<bool>(),
                  "nonfinite gradient");
      auto condensed = torchpairwise::ops::builtin_comparisons(
          x, c10::nullopt, metric, "condensed", false, params);
      TORCH_CHECK(condensed.numel() == 3, "condensed shape");
      grads = torch::autograd::grad({condensed.sum()}, {x});
      TORCH_CHECK(grads[0].isfinite().all().item<bool>(),
                  "nonfinite condensed gradient");
    }
    auto vectors = x.select(1, 0);
    for (auto mode : {"all_pairs", "paired", "condensed"}) {
      auto value = torchpairwise::ops::builtin_comparisons(
          vectors,
          std::string(mode) == "condensed"
              ? c10::nullopt
              : c10::optional<at::Tensor>(vectors + .2),
          "euclidean", mode);
      auto grad = torch::autograd::grad({value.sum()}, {x});
      TORCH_CHECK(grad[0].isfinite().all().item<bool>(),
                  "vector replay gradient");
    }
    std::cout << "Native transport/selector values and gradients passed on "
              << device << '\n';
  }
}
