#pragma once
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <memory>
#include <vector>

#include "streaming_reductions.h"
namespace torchpairwise::ops::detail::streaming {
using at::Tensor;
using Tensors = std::vector<Tensor>;
using Reals = std::vector<c10::optional<double>>;
using Integers = std::vector<c10::optional<int64_t>>;
using Flags = std::vector<c10::optional<bool>>;
using Outputs = std::tuple<Tensor, Tensor, Tensor, Tensor>;
using Values = std::vector<Tensor>;
// Backward-only, single-tile source values and their native autograd graphs.
// Keep this on the tile's stack; never save it across tiles or operator calls.
struct ExpressionReplay {
  Tensors leaves;
};
enum Slot {
  X,
  Y,
  W,
  Variance,
  Inverse,
  Bins,
  Projections,
  XL,
  YL,
  Payload,
  Slots
};
constexpr int64_t ranks[Slots] = {0, 0, 1, 1, 2, 0, 0, 1, 1, 2};
struct ExpressionPlan;
struct Settings {
  std::string metric, operation;
  bool kernel, largest, exclusion, self_cosine, nan_first;
  int64_t rank, k, size, dtype, mask;
  Reals reals;
  Integers integers;
  Flags flags;
  c10::optional<std::string> family;
  std::shared_ptr<const ExpressionPlan> expression;
  bool cached_self() const {
    return family && (*family == "_sequence" || *family == "_mmd");
  }
  bool structured() const {
    return family && *family == "_structured";
  }
  int64_t operand_rank(int64_t slot) const {
    if (!structured())
      return cached_self() && (slot == W || slot == Variance) ? 2
             : slot < 2                                       ? rank
                                                              : ranks[slot];
    if (slot < 2)
      return rank;
    if (slot < 4)
      return integers[0].value() < 4 ? 3 : 4;
    return slot < 8 ? 2 : ranks[slot];
  }
  bool sampled(int64_t slot) const {
    return cached_self()  ? slot < 4 || slot == XL || slot == YL
           : structured() ? slot < 8
                          : slot == X || slot == Y || slot == XL || slot == YL;
  }
  bool first_operand(int64_t slot) const {
    return cached_self()  ? slot == X || slot == W || slot == XL
           : structured() ? slot % 2 == 0
                          : slot == X || slot == XL;
  }
  bool transposed() const {
    return operation.rfind("transpose_", 0) == 0;
  }
  c10::string_view kind() const {
    return c10::string_view(operation).substr(transposed() ? 10 : 0);
  }
  bool has(int64_t slot) const {
    return mask & (int64_t(1) << slot);
  }
  bool differentiable(int64_t slot) const {
    if (family && *family == "_tokens")
      return slot == Payload && has(slot);
    if (family && *family == "_image_event" && metric == "truth_event")
      return slot == Payload && has(slot);
    if (family && (*family == "_residual" || *family == "_image_map") &&
        (metric == "all" || metric == "any"))
      return slot == Payload && has(slot);
    if (structured())
      return has(slot);
    if (slot == XL || slot == YL || !has(slot))
      return false;
    if (slot >= W)
      return true;
    if (kernel)
      return true;
    auto name = metric;
    constexpr const char* suffix = "_distances";
    if (name.size() > 10 && name.compare(name.size() - 10, 10, suffix) == 0)
      name.resize(name.size() - 10);
    for (auto item :
         {"hamming", "jaccard", "kulsinski", "kulczynski1", "yule", "dice",
          "rogerstanimoto", "russellrao", "sokalmichener", "sokalsneath"})
      if (name == item)
        return false;
    return true;
  }
};
// Shared preparation and tile evaluation for reducers and matrix operators.
// Statistics are inferred over the original collections before any slicing.
struct PreparedComparison {
  Tensors values;
  Settings settings;
  std::vector<c10::SymInt> batch;
  c10::SymInt queries, references;
};
PreparedComparison prepare(const Tensor& input,
                           const c10::optional<Tensor>& other,
                           c10::string_view metric, c10::string_view operation,
                           bool kernel,
                           const StreamingReductionOptions& options,
                           bool transposed = false);
BuiltinComparisonOptions comparison_options(const Settings& s,
                                            at::TensorList v);
std::vector<int64_t> batch_shape(at::TensorList v, const Settings& s);
std::vector<int64_t> coordinates(int64_t flat,
                                 const std::vector<int64_t>& batch);
Tensor lookup(Tensor value, const std::vector<int64_t>& index, int64_t rank);
Tensors local_inputs(at::TensorList v, const Settings& s,
                     const std::vector<int64_t>& index, int64_t left,
                     int64_t left_size, int64_t right, int64_t right_size,
                     const Tensor& rows = Tensor(),
                     const Tensor& columns = Tensor(), bool replay = false);
Tensor evaluate(at::TensorList local, const Settings& s, bool paired,
                int64_t left, int64_t right, const Tensor& rows = Tensor(),
                const Tensor& columns = Tensor(),
                ExpressionReplay* replay = nullptr);
Tensor evaluate_higher_order(at::TensorList local, const Settings& s,
                             bool paired, int64_t left, int64_t right,
                             const Tensor& rows = Tensor(),
                             const Tensor& columns = Tensor());
void accumulate(at::TensorList local, const Tensor& value,
                const Tensor& cotangent, Tensors& buffers, const Settings& s,
                const std::vector<int64_t>& index, int64_t left, int64_t count,
                int64_t right, int64_t other, const Tensor& rows = Tensor(),
                const Tensor& columns = Tensor(),
                const ExpressionReplay* replay = nullptr);
std::pair<Tensor, Tensor> compensated(const Tensor& total,
                                      const Tensor& correction,
                                      const Tensor& term);
#ifdef WITH_CUDA
std::pair<Tensor, Tensor> compensated_cuda(const Tensor& total,
                                           const Tensor& correction,
                                           const Tensor& term);
#endif

#define TORCHPAIRWISE_STREAMING_META_DECL                                      \
  c10::string_view metric, c10::string_view operation, bool kernel,            \
      int64_t rank, int64_t k, bool largest, bool exclusion, bool self_cosine, \
      bool nan_first, int64_t size, int64_t dtype, int64_t mask, Reals reals,  \
      Integers integers, Flags flags, c10::optional<c10::string_view> family
#define TORCHPAIRWISE_STREAMING_META_TYPES                                \
  c10::string_view, c10::string_view, bool, int64_t, int64_t, bool, bool, \
      bool, bool, int64_t, int64_t, int64_t, Reals, Integers, Flags,      \
      c10::optional<c10::string_view>
#define TORCHPAIRWISE_STREAMING_META_ARGS                               \
  s.metric, s.operation, s.kernel, s.rank, s.k, s.largest, s.exclusion, \
      s.self_cosine, s.nan_first, s.size, s.dtype, s.mask, s.reals,     \
      s.integers, s.flags,                                              \
      s.family ? c10::optional<c10::string_view>(*s.family) : c10::nullopt
inline Settings settings(TORCHPAIRWISE_STREAMING_META_DECL) {
  return {
      std::string(metric),
      std::string(operation),
      kernel,
      largest,
      exclusion,
      self_cosine,
      nan_first,
      rank,
      k,
      size,
      dtype,
      mask,
      std::move(reals),
      std::move(integers),
      std::move(flags),
      family ? c10::optional<std::string>(std::string(*family)) : c10::nullopt};
}
using CoreSignature = Outputs(at::TensorList,
                              TORCHPAIRWISE_STREAMING_META_TYPES);
using GradSignature = Tensors(const Tensor&, at::TensorList, const Tensor&,
                              const Tensor&, const Tensor&, const Tensor&,
                              TORCHPAIRWISE_STREAMING_META_TYPES);
Outputs expression_reduce(at::TensorList operands, const Settings& settings);
// Corrected product consumer; legacy serialized cores retain their auxiliary
// layouts. global selects a joint N/M product rather than nested row products.
Outputs streaming_product(at::TensorList operands, const Settings& settings,
                          bool global = false);
Outputs streaming_reduce_kernel(at::TensorList operands,
                                const Settings& settings);
Tensors streaming_reduce_backward_kernel(
    const Tensor& gradient, at::TensorList operands, const Tensor& result,
    const Tensor& indices, const Tensor& zeros, const Tensor& nonzero,
    const Settings& settings);
inline Outputs _streaming_reduce_legacy(at::TensorList v, const Settings& s) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_streaming_reduce", "")
          .typed<CoreSignature>();
  return op.call(v, TORCHPAIRWISE_STREAMING_META_ARGS);
}
inline Outputs _streaming_reduce(at::TensorList v, const Settings& s) {
  if (s.kind() == "prod" &&
      (s.dtype == int64_t(at::kFloat) || s.dtype == int64_t(at::kDouble)))
    return streaming_product(v, s);
  if (s.expression)
    return expression_reduce(v, s);
  return _streaming_reduce_legacy(v, s);
}

}  // namespace torchpairwise::ops::detail::streaming
