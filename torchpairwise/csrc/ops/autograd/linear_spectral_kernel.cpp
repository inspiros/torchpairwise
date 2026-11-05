#include <torch/autograd.h>
#include <torch/library.h>

#include "../linear_spectral_detail.h"

namespace torchpairwise::ops::detail::spectral {
namespace {
struct SpectralState {
  Packet packet;
  std::vector<c10::SymInt> shape;
  std::string function, method, gradient;
  c10::optional<int64_t> num_probes, seed, max_iter;
  int64_t block_size;
  double solve_rtol, solve_atol;
  int64_t solve_max_iter;
};
class SpectralEstimateFunction
    : public torch::autograd::Function<SpectralEstimateFunction> {
 public:
  static Tensors forward(torch::autograd::AutogradContext* ctx,
                         at::TensorList values, const Tensor& anchor,
                         const c10::optional<Tensor>& probes,
                         SpectralState state) {
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

    ctx->saved_data["function"] = state.function;
    ctx->saved_data["method"] = state.method;
    ctx->saved_data["gradient"] = state.gradient;
    ctx->saved_data["block_size"] = state.block_size;
    ctx->saved_data["solve_rtol"] = state.solve_rtol;
    ctx->saved_data["solve_atol"] = state.solve_atol;
    ctx->saved_data["solve_max_iter"] = state.solve_max_iter;
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::linear_trace_differentiable", "")
            .typed<ForwardSignature>();
    auto result =
        op.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), state.shape, anchor,
                state.function, state.method, state.num_probes, probes,
                state.seed, state.max_iter, state.block_size, state.gradient,
                state.solve_rtol, state.solve_atol, state.solve_max_iter);
    auto saved = values.vec();
    saved.push_back(anchor);
    saved.push_back(std::get<4>(result));
    ctx->save_for_backward(saved);
    ctx->mark_non_differentiable({std::get<1>(result), std::get<2>(result),
                                  std::get<3>(result), std::get<4>(result)});
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
            std::get<3>(result), std::get<4>(result)};
  }
  static Tensors backward(torch::autograd::AutogradContext* ctx,
                          const Tensors& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of spectral estimates are unsupported");
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
            .findSchemaOrThrow("torchpairwise::linear_trace_backward", "")
            .typed<BackwardSignature>();
    auto result = op.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), shape,
                          saved[offset], saved[offset + 1], gradients[0],
                          ctx->saved_data["function"].toStringRef(),
                          ctx->saved_data["method"].toStringRef(),
                          ctx->saved_data["gradient"].toStringRef(),
                          ctx->saved_data["block_size"].toInt(),
                          ctx->saved_data["solve_rtol"].toDouble(),
                          ctx->saved_data["solve_atol"].toDouble(),
                          ctx->saved_data["solve_max_iter"].toInt());
    result.insert(result.end(), {Tensor(), Tensor(), Tensor()});
    return result;  // Metadata anchor, fixed probes, non-tensor state.
  }
};
Forward spectral_autograd(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                          c10::SymIntArrayRef shape, const Tensor& anchor,
                          c10::string_view function, c10::string_view method,
                          c10::optional<int64_t> num_probes,
                          const c10::optional<Tensor>& probes,
                          c10::optional<int64_t> seed,
                          c10::optional<int64_t> max_iter, int64_t block_size,
                          c10::string_view gradient, double solve_rtol,
                          double solve_atol, int64_t solve_max_iter) {
  TORCH_CHECK(
      !at::GradMode::is_enabled() || !probes || !probes->requires_grad(),
      "Spectral probes are fixed algorithmic inputs; detach probes explicitly");
  auto p = packet_from_args(TORCHPAIRWISE_LINEAR_PACKET_ARGS);
  auto values = packet_values(p);
  auto result = SpectralEstimateFunction::apply(
      at::TensorList(values), anchor, probes,
      SpectralState{std::move(p), shape.vec(), std::string(function),
                    std::string(method), std::string(gradient), num_probes,
                    seed, max_iter, block_size, solve_rtol, solve_atol,
                    solve_max_iter});
  return {result[0], result[1], result[2], result[3], result[4]};
}
}  // namespace
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("linear_trace", torch::CppFunction::makeFallthrough());
  m.impl("linear_function", torch::CppFunction::makeFallthrough());
  m.impl("linear_trace_differentiable", TORCH_FN(spectral_autograd));
  m.impl("linear_trace_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace torchpairwise::ops::detail::spectral
