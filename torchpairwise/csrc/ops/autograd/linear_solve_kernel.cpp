#include <torch/autograd.h>
#include <torch/library.h>

#include <unordered_set>

#include "../linear_solve_detail.h"

namespace torchpairwise::ops::detail::linear {
namespace {
// Keep mathematical system edges, including shared operands, while stopping
// gradients through preconditioner-only construction (zero times a singular
// setup derivative would otherwise still produce NaNs).
Packet detach_preconditioner(Packet p) {
  auto graph = unpack_graph(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p));
  TORCH_CHECK(!graph.nodes.empty() && graph.nodes.back()[0] == 3,
              "Invalid implicit PCG packet");
  std::unordered_set<const void*> system_tensors;
  auto keep = [&](const Tensor& value) {
    system_tensors.insert(value.unsafeGetTensorImpl());
  };
  std::vector<bool> needed(graph.nodes.size());
  auto mark = [&](int64_t node) {
    TORCH_CHECK(node >= 0 && node < int64_t(needed.size()),
                "Invalid PCG system edge");
    needed[node] = true;
  };
  mark(graph.nodes.back()[1]);
  for (int64_t id = int64_t(graph.nodes.size()) - 1; id >= 0; --id) {
    if (!needed[id])
      continue;
    auto node = graph.nodes[id];
    auto kind = node[0], a = node[1], b = node[2];
    if (kind == 0) {
      const auto& expression = graph.sources.at(a);
      for (const auto& source : expression.sources) {
        for (const auto& value : source.inputs)
          keep(value);
        for (const auto& item : source.tensors)
          keep(item.value());
      }
      for (const auto& value : expression.factors)
        keep(value);
    } else if (kind == 8 || kind == 11 || kind == 12) {
      keep(graph.factors.at(a));
      if (kind == 8 || kind == 12)
        keep(graph.factors.at(b));
      if (kind == 12)
        keep(graph.factors.at(b + 1));
    } else {
      mark(a);
      if (kind == 1 || kind == 2 || kind == 3 || kind == 9 || kind == 10)
        mark(b);
      else if (kind == 4 || kind == 5 || kind == 13)
        keep(graph.factors.at(b));
    }
  }
  auto values = packet_values(p);
  for (auto& value : values)
    if (!system_tensors.count(value.unsafeGetTensorImpl()))
      value = value.detach();
  return replace_values(p, values);
}
struct ImplicitState {
  Packet packet;
  std::vector<c10::SymInt> shape;
  std::string method;
  double rtol, atol;
  int64_t max_iter, restart;
  bool preconditioned = false, least_squares = false;
  double damp = 0.;
};
class ImplicitSolveFunction
    : public torch::autograd::Function<ImplicitSolveFunction> {
 public:
  static Tensors forward(torch::autograd::AutogradContext* ctx,
                         at::TensorList values,
                         const c10::optional<Tensor>& initial,
                         ImplicitState state) {
    const auto& p = state.packet;
    ctx->saved_data["counts"] = p.counts;
    ctx->saved_data["metrics"] = p.metrics;
    ctx->saved_data["families"] = p.families;
    ctx->saved_data["reals"] = p.reals;
    ctx->saved_data["integers"] = p.integers;
    ctx->saved_data["flags"] = p.flags;
    ctx->saved_data["strings"] = p.strings;
    ctx->saved_data["real_lists"] = p.real_lists;
    ctx->saved_data["integer_lists"] = p.integer_lists;
    ctx->saved_data["transposed"] = p.transposed;
    ctx->saved_data["source_counts"] = p.source_counts;
    ctx->saved_data["programs"] = p.programs;
    ctx->saved_data["factor_counts"] = p.factor_counts;
    ctx->saved_data["kinds"] = p.kinds;
    ctx->saved_data["integer_numbers"] = p.integer_numbers;
    ctx->saved_data["real_numbers"] = p.real_numbers;
    ctx->saved_data["linear_program"] = p.linear_program;
    std::vector<std::string> keys;
    for (const auto& item : p.tensors)
      keys.push_back(item.key());
    ctx->saved_data["tensor_keys"] = keys;
    ctx->saved_data["sizes"] = std::vector<int64_t>{
        int64_t(p.inputs.size()), int64_t(p.factors.size()),
        int64_t(p.linear_factors.size())};
    ctx->saved_data["shape"] = c10::SymIntArrayRef(state.shape);
    ctx->saved_data["method"] = state.method;
    ctx->saved_data["rtol"] = state.rtol;
    ctx->saved_data["atol"] = state.atol;
    ctx->saved_data["max_iter"] = state.max_iter;
    ctx->saved_data["restart"] = state.restart;
    ctx->saved_data["preconditioned"] = state.preconditioned;
    ctx->saved_data["least_squares"] = state.least_squares;
    ctx->saved_data["damp"] = state.damp;
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::linear_implicit_solve", "")
            .typed<ImplicitSolveSignature>();
    static auto pcg =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::linear_implicit_pcg", "")
            .typed<ImplicitSolveSignature>();
    static auto least =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::linear_implicit_lstsq", "")
            .typed<ImplicitLstsqSignature>();
    Tensors result;
    if (state.least_squares) {
      auto fit = least.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), state.shape,
                            values.back(), state.damp, state.rtol, state.atol,
                            state.max_iter, initial);
      result = {std::get<0>(fit), std::get<1>(fit), std::get<2>(fit),
                std::get<3>(fit), std::get<4>(fit), std::get<5>(fit)};
    } else {
      auto fit = (state.preconditioned ? pcg : op)
                     .call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), state.shape,
                           values.back(), state.method, state.rtol, state.atol,
                           state.max_iter, state.restart, initial);
      result = {std::get<0>(fit), std::get<1>(fit), std::get<2>(fit),
                std::get<3>(fit), std::get<4>(fit)};
    }
    auto saved = values.vec();
    saved.push_back(result[0]);
    ctx->save_for_backward(saved);
    ctx->mark_non_differentiable(Tensors(result.begin() + 1, result.end()));
    return result;
  }

  static Tensors backward(torch::autograd::AutogradContext* ctx,
                          const Tensors& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of implicit solves are unsupported");
    auto saved = ctx->get_saved_variables();
    Packet p;
    p.counts = ctx->saved_data["counts"].to<std::vector<int64_t>>();
    p.metrics = ctx->saved_data["metrics"].to<std::vector<std::string>>();
    p.families = ctx->saved_data["families"].to<std::vector<std::string>>();
    p.reals = ctx->saved_data["reals"].to<RealDict>();
    p.integers = ctx->saved_data["integers"].to<IntDict>();
    p.flags = ctx->saved_data["flags"].to<BoolDict>();
    p.strings = ctx->saved_data["strings"].to<StringDict>();
    p.real_lists = ctx->saved_data["real_lists"].to<RealLists>();
    p.integer_lists = ctx->saved_data["integer_lists"].to<IntLists>();
    p.transposed = ctx->saved_data["transposed"].to<c10::List<bool>>();
    p.source_counts =
        ctx->saved_data["source_counts"].to<std::vector<int64_t>>();
    p.programs = ctx->saved_data["programs"].to<IntLists>();
    p.factor_counts =
        ctx->saved_data["factor_counts"].to<std::vector<int64_t>>();
    p.kinds = ctx->saved_data["kinds"].to<IntLists>();
    p.integer_numbers = ctx->saved_data["integer_numbers"].to<IntLists>();
    p.real_numbers = ctx->saved_data["real_numbers"].to<RealLists>();
    p.linear_program =
        ctx->saved_data["linear_program"].to<std::vector<int64_t>>();
    auto sizes = ctx->saved_data["sizes"].toIntVector();
    auto view = at::TensorList(saved);
    int64_t offset = 0;
    p.inputs = view.slice(offset, sizes[0]).vec();
    offset += sizes[0];
    p.factors = view.slice(offset, sizes[1]).vec();
    offset += sizes[1];
    p.linear_factors = view.slice(offset, sizes[2]).vec();
    offset += sizes[2];
    for (const auto& key :
         ctx->saved_data["tensor_keys"].to<std::vector<std::string>>())
      p.tensors.insert(key, saved[offset++]);
    auto shape = ctx->saved_data["shape"].toSymIntVector();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::linear_implicit_solve_backward",
                               "")
            .typed<ImplicitGradSignature>();
    static auto pcg = c10::Dispatcher::singleton()
                          .findSchemaOrThrow(
                              "torchpairwise::linear_implicit_pcg_backward", "")
                          .typed<ImplicitGradSignature>();
    static auto least =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::linear_implicit_lstsq_backward",
                               "")
            .typed<ImplicitLstsqGradSignature>();
    auto result =
        ctx->saved_data["least_squares"].toBool()
            ? least.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), shape,
                         saved[offset], saved[offset + 1], gradients[0],
                         ctx->saved_data["damp"].toDouble(),
                         ctx->saved_data["rtol"].toDouble(),
                         ctx->saved_data["atol"].toDouble(),
                         ctx->saved_data["max_iter"].toInt())
            : (ctx->saved_data["preconditioned"].toBool() ? pcg : op)
                  .call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), shape,
                        saved[offset], saved[offset + 1], gradients[0],
                        ctx->saved_data["method"].toStringRef(),
                        ctx->saved_data["rtol"].toDouble(),
                        ctx->saved_data["atol"].toDouble(),
                        ctx->saved_data["max_iter"].toInt(),
                        ctx->saved_data["restart"].toInt());
    result.push_back(
        Tensor());  // Initial guess: an algorithmic aid, not the solution map.
    result.push_back(Tensor());  // Non-tensor state.
    return result;
  }
};
SolveTuple implicit_autograd(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                             c10::SymIntArrayRef shape, const Tensor& rhs,
                             c10::string_view method, double rtol, double atol,
                             int64_t max_iter, int64_t restart,
                             const c10::optional<Tensor>& initial) {
  auto p = packet_from_args(TORCHPAIRWISE_LINEAR_PACKET_ARGS);
  auto values = packet_values(p);
  values.push_back(rhs);
  auto result = ImplicitSolveFunction::apply(
      at::TensorList(values), initial,
      ImplicitState{std::move(p), shape.vec(), std::string(method), rtol, atol,
                    max_iter, restart});
  return {result[0], result[1], result[2], result[3], result[4]};
}
SolveTuple implicit_pcg_autograd(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                                 c10::SymIntArrayRef shape, const Tensor& rhs,
                                 c10::string_view method, double rtol,
                                 double atol, int64_t max_iter, int64_t restart,
                                 const c10::optional<Tensor>& initial) {
  auto p =
      detach_preconditioner(packet_from_args(TORCHPAIRWISE_LINEAR_PACKET_ARGS));
  auto values = packet_values(p);
  values.push_back(rhs);
  auto result = ImplicitSolveFunction::apply(
      at::TensorList(values), initial,
      ImplicitState{std::move(p), shape.vec(), std::string(method), rtol, atol,
                    max_iter, restart, true});
  return {result[0], result[1], result[2], result[3], result[4]};
}
LeastSquaresTuple implicit_lstsq_autograd(
    TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef shape,
    const Tensor& rhs, double damp, double rtol, double atol, int64_t max_iter,
    const c10::optional<Tensor>& initial) {
  auto p = packet_from_args(TORCHPAIRWISE_LINEAR_PACKET_ARGS);
  auto values = packet_values(p);
  values.push_back(rhs);
  auto result = ImplicitSolveFunction::apply(
      at::TensorList(values), initial,
      ImplicitState{std::move(p), shape.vec(), "lsqr", rtol, atol, max_iter, 20,
                    false, true, damp});
  return {result[0], result[1], result[2], result[3], result[4], result[5]};
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("linear_implicit_solve", TORCH_FN(implicit_autograd));
  m.impl("linear_implicit_pcg", TORCH_FN(implicit_pcg_autograd));
  m.impl("linear_implicit_lstsq", TORCH_FN(implicit_lstsq_autograd));
  m.impl("linear_implicit_pcg_backward", torch::CppFunction::makeFallthrough());
  m.impl("linear_implicit_lstsq_backward",
         torch::CppFunction::makeFallthrough());
  m.impl("linear_implicit_solve_backward",
         torch::CppFunction::makeFallthrough());
}
}  // namespace
}  // namespace torchpairwise::ops::detail::linear
