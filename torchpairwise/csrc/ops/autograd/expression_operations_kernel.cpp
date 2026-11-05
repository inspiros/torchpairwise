#include <torch/autograd.h>
#include <torch/library.h>

#include "../expression_operations_detail.h"

namespace torchpairwise::ops {
namespace {
using namespace detail::streaming;
struct ExpressionContext {
  ExpressionPacket packet;
  std::string operation;
  int64_t dtype, size;
  bool exclusion, transposed, form;
  int64_t k = -1;
  bool largest = true;
};
class ScalarExpressionFunction
    : public torch::autograd::Function<ScalarExpressionFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx, at::TensorList v,
                        ExpressionContext state) {
    auto& p = state.packet;
    ctx->saved_data["names"] = p.names;
    ctx->saved_data["families"] = p.families;
    ctx->saved_data["headers"] = p.headers;
    ctx->saved_data["nodes"] = p.nodes;
    ctx->saved_data["reals"] = p.reals;
    ctx->saved_data["integers"] = p.integers;
    ctx->saved_data["flags"] = p.flags;
    c10::impl::GenericList numbers(c10::AnyType::get());
    for (const auto& n : p.numbers)
      numbers.push_back(n);
    ctx->saved_data["numbers"] = numbers;
    ctx->saved_data["operation"] = state.operation;
    ctx->saved_data["control"] = std::vector<int64_t>{
        state.dtype, state.size, state.exclusion, state.transposed,
        state.form,  state.k,    state.largest};
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_expression_consume", "")
            .typed<ExpressionSignature>();
    static auto topk_op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_expression_consume", "topk")
            .typed<ExpressionTopKSignature>();
    auto result = state.k >= 0
                      ? topk_op.call(v, p.names, p.families, p.headers, p.reals,
                                     p.integers, p.flags, p.nodes, p.numbers,
                                     state.operation, state.dtype, state.size,
                                     state.exclusion, state.transposed,
                                     state.form, state.k, state.largest)
                      : op.call(v, p.names, p.families, p.headers, p.reals,
                                p.integers, p.flags, p.nodes, p.numbers,
                                state.operation, state.dtype, state.size,
                                state.exclusion, state.transposed, state.form);
    ctx->mark_non_differentiable(
        {std::get<1>(result), std::get<2>(result), std::get<3>(result)});
    auto plan = expression_plan(p);
    bool differentiable = v[Payload].requires_grad();
    for (size_t leaf = 0; leaf < plan->leaves.size(); ++leaf)
      for (int64_t slot = 0; slot < Slots; ++slot)
        differentiable |= plan->leaves[leaf].differentiable(slot) &&
                          v[Slots * (leaf + 1) + slot].requires_grad();
    for (size_t slot = Slots * (1 + plan->leaves.size()); slot < v.size();
         ++slot)
      differentiable |= v[slot].requires_grad();
    if (!differentiable || state.operation == "all" || state.operation == "any")
      ctx->mark_non_differentiable({std::get<0>(result)});
    auto saved = v.vec();
    auto empty = at::empty({0}, v[0].options());
    saved.push_back(state.operation == "prod" ? std::get<0>(result) : empty);
    saved.push_back(state.operation == "neighbors" ||
                            state.operation == "maximum"
                        ? std::get<1>(result)
                        : empty);
    saved.push_back(state.operation == "prod" ? std::get<2>(result) : empty);
    saved.push_back(state.operation == "prod" ? std::get<3>(result) : empty);
    ctx->save_for_backward(saved);
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
            std::get<3>(result)};
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& g) {
    auto saved = ctx->get_saved_variables();
    Tensors v(saved.begin(), saved.end() - 4);
    auto names = ctx->saved_data["names"].to<std::vector<std::string>>();
    auto families = ctx->saved_data["families"].to<std::vector<std::string>>();
    auto headers =
        ctx->saved_data["headers"].to<std::vector<std::vector<int64_t>>>();
    auto nodes =
        ctx->saved_data["nodes"].to<std::vector<std::vector<int64_t>>>();
    auto reals = ctx->saved_data["reals"].to<std::vector<Reals>>();
    auto integers = ctx->saved_data["integers"].to<std::vector<Integers>>();
    auto flags = ctx->saved_data["flags"].to<std::vector<Flags>>();
    std::vector<at::Scalar> numbers;
    for (const auto& n : ctx->saved_data["numbers"].toListRef())
      numbers.push_back(n.toScalar());
    auto control = ctx->saved_data["control"].toIntVector();
    auto operation = ctx->saved_data["operation"].toStringRef();
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow(
                             "torchpairwise::_expression_consume_backward", "")
                         .typed<ExpressionGradSignature>();
    static auto topk_op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_expression_consume_backward",
                               "topk")
            .typed<ExpressionTopKGradSignature>();
    auto result =
        control[5] >= 0
            ? topk_op.call(g[0], v, saved[v.size()], saved[v.size() + 1],
                           saved[v.size() + 2], saved[v.size() + 3], names,
                           families, headers, reals, integers, flags, nodes,
                           numbers, operation, control[0], control[1],
                           bool(control[2]), bool(control[3]), bool(control[4]),
                           control[5], bool(control[6]))
            : op.call(g[0], v, saved[v.size()], saved[v.size() + 1],
                      saved[v.size() + 2], saved[v.size() + 3], names, families,
                      headers, reals, integers, flags, nodes, numbers,
                      operation, control[0], control[1], bool(control[2]),
                      bool(control[3]), bool(control[4]));
    result.push_back(Tensor());
    return result;
  }
};
Outputs expression_consume_autograd(at::TensorList v,
                                    TORCHPAIRWISE_EXPRESSION_DECL) {
  ExpressionContext state{
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      std::string(operation),
      dtype,
      size,
      exclusion,
      transposed,
      form};
  auto result = ScalarExpressionFunction::apply(v, state);
  return {result[0], result[1], result[2], result[3]};
}
Outputs expression_topk_autograd(at::TensorList v,
                                 TORCHPAIRWISE_EXPRESSION_DECL, int64_t k,
                                 bool largest) {
  TORCH_CHECK(k >= 0, "k must be nonnegative");
  ExpressionContext state{
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      std::string(operation),
      dtype,
      size,
      exclusion,
      transposed,
      form,
      k,
      largest};
  auto result = ScalarExpressionFunction::apply(v, state);
  return {result[0], result[1], result[2], result[3]};
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_expression_consume.topk", TORCH_FN(expression_topk_autograd));
  m.impl("_expression_consume_backward.topk",
         torch::CppFunction::makeFallthrough());
  m.impl("_expression_consume", TORCH_FN(expression_consume_autograd));
  m.impl("_expression_consume_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
}  // namespace torchpairwise::ops
