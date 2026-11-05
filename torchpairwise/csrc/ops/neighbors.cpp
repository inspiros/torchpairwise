#include "neighbors.h"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include <limits>

#include "cpdist.h"
#include "pairwise_metrics.h"

namespace torchpairwise::ops {
namespace {
at::Tensor neighbor_mask_from_distances_impl(const at::Tensor& distances,
                                             c10::string_view selection,
                                             c10::optional<int64_t> k,
                                             c10::optional<double> epsilon,
                                             bool exclude_self);
at::Tensor k_neighbors_mask_impl(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2, int64_t k,
    c10::string_view metric, const c10::optional<at::Tensor>& w,
    const c10::optional<at::Tensor>& V, const c10::optional<at::Tensor>& VI,
    c10::optional<double> p, c10::optional<double> base,
    c10::optional<bool> shuffle, c10::optional<at::Generator> generator);
at::Tensor radius_neighbors_mask_impl(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2, double epsilon,
    c10::string_view metric, const c10::optional<at::Tensor>& w,
    const c10::optional<at::Tensor>& V, const c10::optional<at::Tensor>& VI,
    c10::optional<double> p, c10::optional<double> base,
    c10::optional<bool> shuffle, c10::optional<at::Generator> generator);
}  // namespace

at::Tensor neighbor_mask_from_distances(const at::Tensor& distances,
                                        c10::string_view selection,
                                        c10::optional<int64_t> k,
                                        c10::optional<double> epsilon,
                                        bool exclude_self) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::neighbor_mask_from_distances", "")
          .typed<decltype(neighbor_mask_from_distances)>();
  return op.call(distances, selection, k, epsilon, exclude_self);
}

at::Tensor builtin_neighbor_mask(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
    c10::string_view selection, c10::optional<int64_t> k,
    c10::optional<double> epsilon, c10::optional<bool> exclude_self,
    c10::string_view metric, const BuiltinComparisonOptions& options) {
  at::NoGradGuard no_grad_guard;
  auto distances =
      builtin_comparisons(x1, x2, metric, "all_pairs", false, options);
  return neighbor_mask_from_distances(distances, selection, k, epsilon,
                                      exclude_self.value_or(!x2.has_value()));
}

at::Tensor k_neighbors_mask(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2, int64_t k,
    c10::string_view metric, const c10::optional<at::Tensor>& w,
    const c10::optional<at::Tensor>& V, const c10::optional<at::Tensor>& VI,
    c10::optional<double> p, c10::optional<double> base,
    c10::optional<bool> shuffle, c10::optional<at::Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::k_neighbors_mask", "")
                       .typed<decltype(k_neighbors_mask)>();
  return op.call(x1, x2, k, metric, w, V, VI, p, base, shuffle, generator);
}

at::Tensor radius_neighbors_mask(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2, double epsilon,
    c10::string_view metric, const c10::optional<at::Tensor>& w,
    const c10::optional<at::Tensor>& V, const c10::optional<at::Tensor>& VI,
    c10::optional<double> p, c10::optional<double> base,
    c10::optional<bool> shuffle, c10::optional<at::Generator> generator) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::radius_neighbors_mask", "")
          .typed<decltype(radius_neighbors_mask)>();
  return op.call(x1, x2, epsilon, metric, w, V, VI, p, base, shuffle,
                 generator);
}

TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "neighbor_mask_from_distances(Tensor distances, str selection, int? "
      "k=None, "
      "float? epsilon=None, bool exclude_self=False) -> Tensor",
      TORCH_FN(neighbor_mask_from_distances_impl));
  // utilities
  m.def(
      "torchpairwise::k_neighbors_mask(Tensor x1, Tensor? x2=None, int k=1, "
      "str metric=\"euclidean\", *, " TORCHPAIRWISE_CPDIST_EXTRA_ARGS_SCHEMA_STR
      ") -> Tensor",
      TORCH_FN(k_neighbors_mask_impl));
  m.def(
      "torchpairwise::radius_neighbors_mask(Tensor x1, Tensor? x2=None, float "
      "epsilon=0., "
      "str metric=\"euclidean\", *, " TORCHPAIRWISE_CPDIST_EXTRA_ARGS_SCHEMA_STR
      ") -> Tensor",
      TORCH_FN(radius_neighbors_mask_impl));
}
namespace {
at::Tensor neighbor_mask_from_distances_impl(const at::Tensor& distances,
                                             c10::string_view selection,
                                             c10::optional<int64_t> k,
                                             c10::optional<double> epsilon,
                                             bool exclude_self) {
  TORCH_CHECK(
      distances.layout() == at::kStrided && distances.dim() >= 2,
      "Neighbor distances must be dense strided tensors with shape [...,N,M]");
  TORCH_CHECK(selection == "k" || selection == "radius",
              "selection must be k or radius");
  TORCH_CHECK(!exclude_self || distances.sym_size(-2) == distances.sym_size(-1),
              "Self exclusion requires equal sample counts");
  at::NoGradGuard no_grad_guard;
  if (selection == "radius") {
    TORCH_CHECK(!k.has_value(), "Unused parameter: k");
    TORCH_CHECK(epsilon.has_value() && *epsilon >= 0.,
                "epsilon must be non-negative");
    auto output = distances.le(*epsilon);
    if (exclude_self)
      output.diagonal(0, -2, -1).fill_(false);
    return output;
  }
  TORCH_CHECK(!epsilon.has_value(), "Unused parameter: epsilon");
  TORCH_CHECK(k.has_value() && *k >= 0, "k must be a non-negative integer");
  // A zero request also handles an empty square self-collection.
  // Positive self requests leave one column for the excluded entry.
  auto columns = distances.sym_size(-1);
  TORCH_CHECK(*k <= columns && (!exclude_self || *k == 0 || *k < columns),
              "k exceeds the number of eligible columns");
  auto output = at::zeros_like(distances, distances.options().dtype(at::kBool));
  auto order = at::argsort(distances, true, -1, false);
  if (exclude_self) {
    // Move the excluded entry after even a NaN candidate. Sorting
    // the exclusion flag stably preserves all eligible ordering.
    output.diagonal(0, -2, -1).fill_(true);
    auto excluded = output.gather(-1, order);
    order =
        order.gather(-1, at::argsort(excluded.to(at::kByte), true, -1, false));
    output.zero_();
  }
  return output.scatter_(-1, order.slice(-1, 0, *k), true);
}

at::Tensor k_neighbors_mask_impl(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2, int64_t k,
    c10::string_view metric, const c10::optional<at::Tensor>& w,
    const c10::optional<at::Tensor>& V, const c10::optional<at::Tensor>& VI,
    c10::optional<double> p, c10::optional<double> base,
    c10::optional<bool> shuffle, c10::optional<at::Generator> generator) {
  C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.neighbors.k_neighbors_mask")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) = utils::check_pairwise_inputs("k_neighbors_mask", x1, x2);
  TORCH_CHECK(k >= 0, "k must be non-negative. Got k=", k)
  at::NoGradGuard no_grad_guard;
  auto dists =
      ops::cdist(x1_, x2_, metric, w, V, VI, p, base, shuffle, generator);
  return neighbor_mask_from_distances(dists, "k", k, c10::nullopt,
                                      !x2.has_value());
}

at::Tensor radius_neighbors_mask_impl(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2, double epsilon,
    c10::string_view metric, const c10::optional<at::Tensor>& w,
    const c10::optional<at::Tensor>& V, const c10::optional<at::Tensor>& VI,
    c10::optional<double> p, c10::optional<double> base,
    c10::optional<bool> shuffle, c10::optional<at::Generator> generator) {
  C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.neighbors.k_neighbors_mask")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("radius_neighbors_mask", x1, x2);
  TORCH_CHECK(epsilon >= 0.,
              "epsilon must be non-negative. Got epsilon=", epsilon)
  at::NoGradGuard no_grad_guard;
  auto dists =
      ops::cdist(x1_, x2_, metric, w, V, VI, p, base, shuffle, generator);
  return neighbor_mask_from_distances(dists, "radius", c10::nullopt, epsilon,
                                      !x2.has_value());
}
}  // namespace
}  // namespace torchpairwise::ops
