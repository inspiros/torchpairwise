#pragma once
#include "linear_operations.h"
#include "streaming_reductions_detail.h"

namespace torchpairwise::ops::detail::linear {
using namespace streaming;
using TensorDict = c10::Dict<std::string, at::Tensor>;
using RealDict = c10::Dict<std::string, double>;
using IntDict = c10::Dict<std::string, int64_t>;
using BoolDict = c10::Dict<std::string, bool>;
using StringDict = c10::Dict<std::string, std::string>;
using RealLists = c10::Dict<std::string, c10::List<double>>;
using IntLists = c10::Dict<std::string, c10::List<int64_t>>;

// One typed flat packet serves Python/export and the native graph facade.
struct Packet {
  Tensors inputs, factors, linear_factors;
  std::vector<int64_t> counts, source_counts, factor_counts, linear_program;
  std::vector<std::string> metrics, families;
  TensorDict tensors;
  RealDict reals;
  IntDict integers;
  BoolDict flags;
  StringDict strings;
  RealLists real_lists;
  IntLists integer_lists;
  c10::List<bool> transposed;
  IntLists programs, kinds, integer_numbers;
  RealLists real_numbers;
};
#define TORCHPAIRWISE_LINEAR_PACKET_DECL                                     \
  at::TensorList inputs, at::IntArrayRef counts,                             \
      const std::vector<std::string>&metrics,                                \
      const std::vector<std::string>&families, const TensorDict &tensors,    \
      const RealDict &reals, const IntDict &integers, const BoolDict &flags, \
      const StringDict &strings, const RealLists &real_lists,                \
      const IntLists &integer_lists, const c10::List<bool>&transposed,       \
      at::IntArrayRef source_counts, const IntLists &programs,               \
      at::TensorList factors, at::IntArrayRef factor_counts,                 \
      const IntLists &kinds, const IntLists &integer_numbers,                \
      const RealLists &real_numbers, at::IntArrayRef linear_program,         \
      at::TensorList linear_factors
#define TORCHPAIRWISE_LINEAR_PACKET_ARGS                                       \
  inputs, counts, metrics, families, tensors, reals, integers, flags, strings, \
      real_lists, integer_lists, transposed, source_counts, programs, factors, \
      factor_counts, kinds, integer_numbers, real_numbers, linear_program,     \
      linear_factors
#define TORCHPAIRWISE_LINEAR_PACKED_ARGS(p)                                  \
  p.inputs, p.counts, p.metrics, p.families, p.tensors, p.reals, p.integers, \
      p.flags, p.strings, p.real_lists, p.integer_lists, p.transposed,       \
      p.source_counts, p.programs, p.factors, p.factor_counts, p.kinds,      \
      p.integer_numbers, p.real_numbers, p.linear_program, p.linear_factors

Packet pack_graph(const LinearOperatorGraph& graph);
LinearOperatorGraph unpack_graph(TORCHPAIRWISE_LINEAR_PACKET_DECL);
struct Plan {
  LinearOperatorGraph graph;
  std::vector<PreparedComparison> sources;
  std::vector<std::vector<c10::SymInt>> shapes;
  Plan(LinearOperatorGraph graph, const Tensor& like);
  Tensor apply(int64_t id, const Tensor& values, bool transpose = false) const;
  Tensor operator()(const Tensor& values) const;
};
}  // namespace torchpairwise::ops::detail::linear
