#include <torch/autograd.h>
#include <torch/library.h>

#include "../expression_statistics_detail.h"
#include "../softmax_operations_detail.h"

namespace torchpairwise::ops::detail::streaming {
Tensors statistics_higher_order_backward(const Tensor&, at::TensorList,
                                         Settings, bool, bool, int64_t,
                                         int64_t);
}
namespace torchpairwise::ops {
namespace {
using namespace detail::streaming;
struct StatisticsContext {
  ExpressionPacket packet;
  std::string operation;
  int64_t dtype, size;
  bool exclusion, transposed, global;
  double correction;
  bool weighted = false;
  bool softmax = false;
  bool inner_product = false, center = false, normalize = false;
  bool product = false;
  bool selection = false;
  std::string interpolation = "linear";
  int64_t value_root = -1, weight_root = -1;
};
class ExpressionStatisticsFunction
    : public torch::autograd::Function<ExpressionStatisticsFunction> {
 public:
  static Values forward(torch::autograd::AutogradContext* ctx,
                        at::TensorList operands, StatisticsContext state) {
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
    ctx->saved_data["control"] =
        std::vector<int64_t>{state.dtype, state.size, state.exclusion,
                             state.transposed, state.global};
    ctx->saved_data["correction"] = state.correction;
    ctx->saved_data["weighted"] = state.weighted;
    ctx->saved_data["softmax"] = state.softmax;
    ctx->saved_data["inner_product"] = state.inner_product;
    ctx->saved_data["product"] = state.product;
    ctx->saved_data["selection"] = state.selection;
    ctx->saved_data["interpolation"] = state.interpolation;
    ctx->saved_data["center"] = state.center;
    ctx->saved_data["normalize"] = state.normalize;
    ctx->saved_data["roots"] =
        std::vector<int64_t>{state.value_root, state.weight_root};
    at::AutoDispatchBelowADInplaceOrView guard;
    StatisticsOutputs output;
    if (state.selection) {
      static auto op =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow("torchpairwise::_expression_selection", "")
              .typed<SelectionSignature>();
      output =
          op.call(operands, p.names, p.families, p.headers, p.reals, p.integers,
                  p.flags, p.nodes, p.numbers, state.operation, state.dtype,
                  state.size, state.exclusion, state.transposed, state.global,
                  state.correction, state.interpolation);
    } else if (state.product) {
      static auto op =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow("torchpairwise::_expression_product", "")
              .typed<StatisticsSignature>();
      output = op.call(operands, p.names, p.families, p.headers, p.reals,
                       p.integers, p.flags, p.nodes, p.numbers, state.operation,
                       state.dtype, state.size, state.exclusion,
                       state.transposed, state.global, 0.);
    } else if (state.inner_product) {
      static auto op =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow("torchpairwise::_expression_inner_product", "")
              .typed<InnerProductSignature>();
      output =
          op.call(operands, p.names, p.families, p.headers, p.reals, p.integers,
                  p.flags, p.nodes, p.numbers, state.operation, state.dtype,
                  state.size, false, false, false, state.value_root,
                  state.weight_root, state.center, state.normalize);
    } else if (state.softmax) {
      static auto op = c10::Dispatcher::singleton()
                           .findSchemaOrThrow(
                               "torchpairwise::_expression_softmax_matmul", "")
                           .typed<SoftmaxSignature>();
      output =
          op.call(operands, p.names, p.families, p.headers, p.reals, p.integers,
                  p.flags, p.nodes, p.numbers, state.operation, state.dtype,
                  state.size, state.exclusion, state.transposed, false);
    } else if (state.weighted) {
      static auto op =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow("torchpairwise::_expression_statistics",
                                 "weighted_anchored")
              .typed<WeightedStatisticsSignature>();
      output =
          op.call(operands, p.names, p.families, p.headers, p.reals, p.integers,
                  p.flags, p.nodes, p.numbers, state.operation, state.dtype,
                  state.size, state.exclusion, state.transposed, state.global,
                  state.correction, state.value_root, state.weight_root);
    } else {
      static auto op =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow("torchpairwise::_expression_statistics",
                                 "anchored")
              .typed<StatisticsSignature>();
      output = op.call(operands, p.names, p.families, p.headers, p.reals,
                       p.integers, p.flags, p.nodes, p.numbers, state.operation,
                       state.dtype, state.size, state.exclusion,
                       state.transposed, state.global, state.correction);
    }
    auto [result, center] = output;
    ctx->mark_non_differentiable({center});
    auto plan = expression_plan(p);
    bool differentiable = false;
    if (state.softmax)
      differentiable |= operands[Payload].requires_grad();
    for (size_t leaf = 0; leaf < plan->leaves.size(); ++leaf)
      for (int64_t slot = 0; slot < Slots; ++slot)
        differentiable |= plan->leaves[leaf].differentiable(slot) &&
                          operands[Slots * (leaf + 1) + slot].requires_grad();
    for (size_t slot = Slots * (1 + plan->leaves.size());
         slot < operands.size(); ++slot)
      differentiable |= operands[slot].requires_grad();
    if (!differentiable || state.operation == "count_nonzero")
      ctx->mark_non_differentiable({result});
    auto saved = operands.vec();
    saved.insert(saved.end(), {result, center});
    ctx->save_for_backward(saved);
    return {result, center};
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(
        !at::GradMode::is_enabled() ||
            (ctx->saved_data["softmax"].toBool() ||
             (!ctx->saved_data["selection"].toBool() &&
              !ctx->saved_data["product"].toBool() &&
              !ctx->saved_data["inner_product"].toBool() &&
              ((!ctx->saved_data["weighted"].toBool() &&
                ctx->saved_data["operation"].toStringRef() == "logsumexp") ||
               (ctx->saved_data["weighted"].toBool() &&
                ctx->saved_data["operation"].toStringRef() == "mean")))),
        "Higher derivatives of streamed statistics are unsupported");
    auto saved = ctx->get_saved_variables();
    Tensors operands(saved.begin(), saved.end() - 2);
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
    Tensors result;
    if (ctx->saved_data["selection"].toBool()) {
      static auto op =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow(
                  "torchpairwise::_expression_selection_backward", "")
              .typed<SelectionGradSignature>();
      result = op.call(
          gradients[0], operands, saved[operands.size()],
          saved[operands.size() + 1], names, families, headers, reals, integers,
          flags, nodes, numbers, ctx->saved_data["operation"].toStringRef(),
          control[0], control[1], bool(control[2]), bool(control[3]),
          bool(control[4]), ctx->saved_data["correction"].toDouble(),
          ctx->saved_data["interpolation"].toStringRef());
    } else if (ctx->saved_data["product"].toBool()) {
      static auto op =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow("torchpairwise::_expression_product_backward",
                                 "")
              .typed<StatisticsGradSignature>();
      result = op.call(gradients[0], operands, saved[operands.size()],
                       saved[operands.size() + 1], names, families, headers,
                       reals, integers, flags, nodes, numbers,
                       ctx->saved_data["operation"].toStringRef(), control[0],
                       control[1], bool(control[2]), bool(control[3]),
                       bool(control[4]), 0.);
    } else if (ctx->saved_data["inner_product"].toBool()) {
      auto roots = ctx->saved_data["roots"].toIntVector();
      static auto op =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow(
                  "torchpairwise::_expression_inner_product_backward", "")
              .typed<InnerProductGradSignature>();
      result = op.call(gradients[0], operands, saved[operands.size()],
                       saved[operands.size() + 1], names, families, headers,
                       reals, integers, flags, nodes, numbers,
                       ctx->saved_data["operation"].toStringRef(), control[0],
                       control[1], false, false, false, roots[0], roots[1],
                       ctx->saved_data["center"].toBool(),
                       ctx->saved_data["normalize"].toBool());
    } else if (ctx->saved_data["softmax"].toBool()) {
      static auto op =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow(
                  "torchpairwise::_expression_softmax_matmul_backward", "")
              .typed<SoftmaxGradSignature>();
      result = op.call(gradients[0], operands, saved[operands.size()],
                       saved[operands.size() + 1], names, families, headers,
                       reals, integers, flags, nodes, numbers,
                       ctx->saved_data["operation"].toStringRef(), control[0],
                       control[1], bool(control[2]), bool(control[3]), false);
    } else if (ctx->saved_data["weighted"].toBool()) {
      auto roots = ctx->saved_data["roots"].toIntVector();
      static auto op =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow(
                  "torchpairwise::_expression_statistics_backward", "weighted")
              .typed<WeightedStatisticsGradSignature>();
      if (at::GradMode::is_enabled()) {
        auto s = statistics_settings(
            {names, families, headers, nodes, reals, integers, flags, numbers},
            "var", control[0], control[1], bool(control[2]), bool(control[3]));
        result = statistics_higher_order_backward(gradients[0], operands, s,
                                                  false, bool(control[4]),
                                                  roots[0], roots[1]);
      } else {
        result = op.call(
            gradients[0], operands, saved[operands.size()],
            saved[operands.size() + 1], names, families, headers, reals,
            integers, flags, nodes, numbers,
            ctx->saved_data["operation"].toStringRef(), control[0], control[1],
            bool(control[2]), bool(control[3]), bool(control[4]),
            ctx->saved_data["correction"].toDouble(), roots[0], roots[1]);
      }
    } else {
      static auto op =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow(
                  "torchpairwise::_expression_statistics_backward", "")
              .typed<StatisticsGradSignature>();
      result = op.call(
          gradients[0], operands, saved[operands.size()],
          saved[operands.size() + 1], names, families, headers, reals, integers,
          flags, nodes, numbers, ctx->saved_data["operation"].toStringRef(),
          control[0], control[1], bool(control[2]), bool(control[3]),
          bool(control[4]), ctx->saved_data["correction"].toDouble());
    }
    result.push_back(Tensor());
    return result;
  }
};
StatisticsOutputs anchored_statistics_autograd(at::TensorList operands,
                                               TORCHPAIRWISE_EXPRESSION_DECL,
                                               double correction) {
  StatisticsContext state{
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      std::string(operation),
      dtype,
      size,
      exclusion,
      transposed,
      form,
      correction};
  auto result = ExpressionStatisticsFunction::apply(operands, state);
  return {result[0], result[1]};
}
StatisticsOutputs statistics_autograd(at::TensorList operands,
                                      TORCHPAIRWISE_EXPRESSION_DECL,
                                      double correction) {
  auto [result, center] = anchored_statistics_autograd(
      operands, names, families, headers, reals, integers, flags, nodes,
      numbers, operation, dtype, size, exclusion, transposed, form, correction);
  if (operation == "var" || operation == "std")
    center = center.sum(-1);
  return {result, center};
}
StatisticsOutputs anchored_weighted_statistics_autograd(
    at::TensorList operands, TORCHPAIRWISE_EXPRESSION_DECL, double correction,
    int64_t value_root, int64_t weight_root) {
  StatisticsContext state{
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      std::string(operation),
      dtype,
      size,
      exclusion,
      transposed,
      form,
      correction};
  state.weighted = true;
  state.value_root = value_root;
  state.weight_root = weight_root;
  auto result = ExpressionStatisticsFunction::apply(operands, state);
  return {result[0], result[1]};
}
StatisticsOutputs weighted_statistics_autograd(at::TensorList operands,
                                               TORCHPAIRWISE_EXPRESSION_DECL,
                                               double correction,
                                               int64_t value_root,
                                               int64_t weight_root) {
  auto [result, state] = anchored_weighted_statistics_autograd(
      operands, names, families, headers, reals, integers, flags, nodes,
      numbers, operation, dtype, size, exclusion, transposed, form, correction,
      value_root, weight_root);
  return {result, legacy_weighted_statistics_state(state)};
}
StatisticsOutputs softmax_matmul_autograd(at::TensorList operands,
                                          TORCHPAIRWISE_EXPRESSION_DECL) {
  StatisticsContext state{
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      std::string(operation),
      dtype,
      size,
      exclusion,
      transposed,
      false,
      0.};
  state.softmax = true;
  auto result = ExpressionStatisticsFunction::apply(operands, state);
  return {result[0], result[1]};
}
StatisticsOutputs inner_product_autograd(

    at::TensorList operands, TORCHPAIRWISE_EXPRESSION_DECL, int64_t left_root,
    int64_t right_root, bool center, bool normalize) {
  TORCH_CHECK(!exclusion && !transposed && !form,
              "Unused Frobenius inner product metadata");
  StatisticsContext state{
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      std::string(operation),
      dtype,
      size,
      false,
      false,
      false,
      0.};
  state.inner_product = true;
  state.center = center;
  state.normalize = normalize;
  state.value_root = left_root;
  state.weight_root = right_root;
  auto result = ExpressionStatisticsFunction::apply(operands, state);
  return {result[0], result[1]};
}
StatisticsOutputs product_autograd(at::TensorList operands,
                                   TORCHPAIRWISE_EXPRESSION_DECL,
                                   double correction) {
  TORCH_CHECK(operation == "prod" && correction == 0.,
              "Invalid product consumer metadata");
  StatisticsContext state{
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      std::string(operation),
      dtype,
      size,
      exclusion,
      transposed,
      form,
      0.};
  state.product = true;
  auto result = ExpressionStatisticsFunction::apply(operands, state);
  return {result[0], result[1]};
}
StatisticsOutputs selection_autograd(at::TensorList operands,
                                     TORCHPAIRWISE_EXPRESSION_DECL, double q,
                                     c10::string_view interpolation) {
  StatisticsContext state{
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      std::string(operation),
      dtype,
      size,
      exclusion,
      transposed,
      form,
      q};
  state.selection = true;
  state.interpolation = std::string(interpolation);
  auto result = ExpressionStatisticsFunction::apply(operands, state);
  return {result[0], result[1]};
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_expression_selection", TORCH_FN(selection_autograd));
  m.impl("_expression_selection_backward",
         torch::CppFunction::makeFallthrough());
  m.impl("_expression_product", TORCH_FN(product_autograd));
  m.impl("_expression_product_backward", torch::CppFunction::makeFallthrough());
  m.impl("_expression_inner_product", TORCH_FN(inner_product_autograd));
  m.impl("_expression_inner_product_backward",
         torch::CppFunction::makeFallthrough());
  m.impl("_expression_softmax_matmul", TORCH_FN(softmax_matmul_autograd));
  m.impl("_expression_softmax_matmul_backward",
         torch::CppFunction::makeFallthrough());
  m.impl("_expression_statistics.weighted",
         TORCH_FN(weighted_statistics_autograd));
  m.impl("_expression_statistics.weighted_anchored",
         TORCH_FN(anchored_weighted_statistics_autograd));
  m.impl("_expression_statistics_backward.weighted",
         torch::CppFunction::makeFallthrough());
  m.impl("_expression_statistics", TORCH_FN(statistics_autograd));
  m.impl("_expression_statistics.anchored",
         TORCH_FN(anchored_statistics_autograd));
  m.impl("_expression_statistics_backward",
         torch::CppFunction::makeFallthrough());
}
}  // namespace
}  // namespace torchpairwise::ops
