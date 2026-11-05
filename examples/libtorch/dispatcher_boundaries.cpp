// Public C++ wrappers and generic selectors honor dispatcher overrides.
// No Python interpreter or Python registration code is used.
#include <torch/csrc/autograd/autograd.h>
#include <torch/library.h>

#include <iostream>

#include "ops/histogram_intersection.h"
#include "ops/nan_euclidean.h"
#include "torchpairwise.h"

namespace {
using at::Tensor;
Tensor constant_comparison(const Tensor& x, const c10::optional<Tensor>& y,
                           double value) {
  auto shape = x.sizes().slice(0, x.dim() - 2).vec();
  shape.push_back(x.size(-2));
  shape.push_back(y.value_or(x).size(-2));
  return at::full(shape, value, x.options());
}
Tensor named_override(const Tensor& x, const c10::optional<Tensor>& y) {
  return constant_comparison(x, y, 41.);
}
Tensor family_override(const Tensor& x, const c10::optional<Tensor>& y,
                       c10::string_view metric, c10::string_view mode,
                       c10::optional<double> base,
                       c10::optional<double> length_scale,
                       c10::optional<double> alpha, c10::optional<double> nu,
                       c10::optional<bool> squared,
                       const c10::optional<Tensor>& bins) {
  return constant_comparison(x, y, 42.);
}
}  // namespace
int main() {
  using namespace torchpairwise::ops;
  auto x = at::rand({3, 2}, at::kDouble) + .2;
  auto y = at::rand({4, 2}, at::kDouble) + .2;
  auto reference = angular_distances(x, y);
  {
    torch::Library overrides(torch::Library::IMPL, "torchpairwise",
                             c10::DispatchKey::CPU, __FILE__, __LINE__);
    overrides.impl("angular_distances", TORCH_FN(named_override));
    overrides.impl("euclidean_distances", TORCH_FN(named_override));
    overrides.impl("vector_comparisons", TORCH_FN(family_override));
    TORCH_CHECK(angular_distances(x, y).eq(41.).all().item<bool>());
    TORCH_CHECK(cdist(x, y, "euclidean").eq(41.).all().item<bool>());
    TORCH_CHECK(
        builtin_comparisons(x, y, "angular").eq(42.).all().item<bool>());
  }
  TORCH_CHECK(at::allclose(angular_distances(x, y), reference));
  auto a = x.detach().set_requires_grad(true);
  auto b = y.detach().set_requires_grad(true);
  auto intersection = _histogram_intersection(a, b);
  auto expected = at::minimum(a.unsqueeze(1), b.unsqueeze(0)).sum(-1);
  auto actual_grad = torch::autograd::grad({intersection.sum()}, {a, b});
  auto expected_grad = torch::autograd::grad({expected.sum()}, {a, b});
  TORCH_CHECK(at::allclose(intersection, expected));
  for (size_t i = 0; i < actual_grad.size(); ++i)
    TORCH_CHECK(at::allclose(actual_grad[i], expected_grad[i]));
  auto squared = _nan_sqeuclidean(a, b);
  expected = (a.unsqueeze(1) - b.unsqueeze(0)).square().sum(-1);
  actual_grad = torch::autograd::grad({std::get<0>(squared).sum()}, {a, b});
  expected_grad = torch::autograd::grad({expected.sum()}, {a, b});
  TORCH_CHECK(at::allclose(std::get<0>(squared), expected));
  TORCH_CHECK(std::get<1>(squared).eq(a.size(-1)).all().item<bool>());
  for (size_t i = 0; i < actual_grad.size(); ++i)
    TORCH_CHECK(at::allclose(actual_grad[i], expected_grad[i]));
  std::cout << "Standalone typed public/named/family dispatch and override "
               "removal passed\n";
}
