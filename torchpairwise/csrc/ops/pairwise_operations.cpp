#include "pairwise_operations.h"

#include <ATen/ExpandUtils.h>
#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include "expression_operations_detail.h"
#include "operator_sources_detail.h"
#include "pairwise_operations_detail.h"
#include "utils/comparison_dispatch.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops {
using at::Tensor;
namespace detail::streaming {
std::vector<int64_t> form_batch(at::TensorList v, const Tensor& left,
                                const Tensor& right, const Settings& s) {
  auto batch = batch_shape(v, s);
  for (const auto& value : {left, right})
    batch = at::infer_size(batch, value.sizes().slice(0, value.dim() - 2));
  return batch;
}
Tensor expression_form_kernel(at::TensorList v, const Tensor& left,
                              const Tensor& right, const Settings& s) {
  auto size = s.size;
  utils::DeferredValueChecks checks;
  auto batch = form_batch(v, left, right, s), shape = batch;
  shape.insert(shape.end(), {left.size(-1), right.size(-1)});
  auto result = at::zeros(shape, v[X].options().dtype(at::ScalarType(s.dtype)));
  auto correction = at::zeros_like(result);
  int64_t batches = 1;
  for (auto d : batch)
    batches *= d;
  for (int64_t z = 0; z < batches; ++z) {
    auto index = coordinates(z, batch);
    auto l = lookup(left, index, 2), r = lookup(right, index, 2);
    for (int64_t i = 0; i < l.size(0); i += size)
      for (int64_t j = 0; j < r.size(0); j += size) {
        auto n = std::min(size, l.size(0) - i),
             m = std::min(size, r.size(0) - j);
        auto local = local_inputs(v, s, index, i, n, j, m);
        auto tile = evaluate(local, s, false, i, j).to(left.scalar_type());
        for (int64_t u = 0; u < l.size(-1); u += size) {
          auto p = std::min(size, l.size(-1) - u);
          auto lp =
              l.narrow(0, i, n).narrow(-1, u, p).transpose(0, 1).matmul(tile);
          for (int64_t c = 0; c < r.size(-1); c += size) {
            auto t = std::min(size, r.size(-1) - c);
            auto term = lp.matmul(r.narrow(0, j, m).narrow(-1, c, t));
            auto total =
                lookup(result, index, 2).narrow(0, u, p).narrow(1, c, t);
            auto error =
                lookup(correction, index, 2).narrow(0, u, p).narrow(1, c, t);
            auto update = compensated(total, error, term);
            total.copy_(update.first);
            error.copy_(update.second);
          }
        }
      }
  }
  checks.finish();
  return result + correction;
}
Tensors expression_form_backward_kernel(const Tensor& gradient,
                                        at::TensorList v, const Tensor& left,
                                        const Tensor& right,
                                        const Settings& s) {
  auto size = s.size;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  utils::DeferredValueChecks checks;
  Tensors buffers;
  for (const auto& value : v)
    buffers.push_back(at::zeros_like(
        value, value.options().dtype(value.is_floating_point()
                                         ? at::kDouble
                                         : value.scalar_type())));
  auto gl = at::zeros(left.sizes(), left.options().dtype(at::kDouble));
  auto gr = at::zeros(right.sizes(), right.options().dtype(at::kDouble));
  auto batch = form_batch(v, left, right, s);
  int64_t batches = 1;
  for (auto d : batch)
    batches *= d;
  at::AutoGradMode enable(true);
  for (int64_t z = 0; z < batches; ++z) {
    auto index = coordinates(z, batch);
    auto l = lookup(left, index, 2).detach(),
         r = lookup(right, index, 2).detach();
    auto g = lookup(gradient, index, 2).detach();
    for (int64_t i = 0; i < l.size(0); i += size)
      for (int64_t j = 0; j < r.size(0); j += size) {
        auto n = std::min(size, l.size(0) - i),
             m = std::min(size, r.size(0) - j);
        auto local =
            local_inputs(v, s, index, i, n, j, m, Tensor(), Tensor(), true);
        ExpressionReplay replay;
        auto tile =
            evaluate(local, s, false, i, j, Tensor(), Tensor(), &replay);
        auto cot = at::zeros({n, m}, gradient.options().dtype(at::kDouble));
        {
          at::NoGradGuard guard;
          auto k = tile.detach().to(at::kDouble);
          for (int64_t u = 0; u < l.size(-1); u += size) {
            auto p = std::min(size, l.size(-1) - u);
            auto lp = l.narrow(0, i, n).narrow(-1, u, p).to(at::kDouble);
            for (int64_t c = 0; c < r.size(-1); c += size) {
              auto t = std::min(size, r.size(-1) - c);
              auto rp = r.narrow(0, j, m).narrow(-1, c, t).to(at::kDouble);
              auto gp = g.narrow(0, u, p).narrow(1, c, t).to(at::kDouble);
              cot.add_(lp.matmul(gp).matmul(rp.transpose(0, 1)));
              lookup(gl, index, 2)
                  .narrow(0, i, n)
                  .narrow(1, u, p)
                  .add_(k.matmul(rp).matmul(gp.transpose(0, 1)));
              lookup(gr, index, 2)
                  .narrow(0, j, m)
                  .narrow(1, c, t)
                  .add_(k.transpose(0, 1).matmul(lp).matmul(gp));
            }
          }
        }
        accumulate(local, tile, cot.to(tile.scalar_type()), buffers, s, index,
                   i, n, j, m, Tensor(), Tensor(), &replay);
      }
  }
  for (size_t slot = 0; slot < v.size(); ++slot)
    buffers[slot] = buffers[slot].to(v[slot].scalar_type()).contiguous();
  buffers.push_back(gl.to(left.scalar_type()).contiguous());
  buffers.push_back(gr.to(right.scalar_type()).contiguous());
  checks.finish();
  return buffers;
}
Tensor pairwise_form_kernel(at::TensorList v, const Tensor& left,
                            const Tensor& right,
                            TORCHPAIRWISE_STREAMING_META_DECL) {
  auto s = settings(metric, operation, kernel, rank, k, largest, exclusion,
                    self_cosine, nan_first, size, dtype, mask, std::move(reals),
                    std::move(integers), std::move(flags), family);
  return expression_form_kernel(v, left, right, s);
}
Tensors pairwise_form_backward_kernel(const Tensor& gradient, at::TensorList v,
                                      const Tensor& left, const Tensor& right,
                                      TORCHPAIRWISE_STREAMING_META_DECL) {
  auto s = settings(metric, operation, kernel, rank, k, largest, exclusion,
                    self_cosine, nan_first, size, dtype, mask, std::move(reals),
                    std::move(integers), std::move(flags), family);
  return expression_form_backward_kernel(gradient, v, left, right, s);
}
void validate_pairwise_options(c10::string_view operation,
                               const PairwiseOperationOptions& o) {
  TORCH_CHECK(operation == "reduce" || operation == "matmul" ||
                  operation == "bilinear" || operation == "quadratic" ||
                  operation == "diagonal" || operation == "block",
              "Unknown pairwise operation");
  TORCH_CHECK(!o.left || operation == "bilinear" || operation == "quadratic",
              "Unused left operand");
  TORCH_CHECK(!o.right || operation == "bilinear" || operation == "matmul",
              "Unused right operand");
  TORCH_CHECK((!o.rows && !o.columns) || operation == "block",
              "Unused block indices");
  TORCH_CHECK(
      operation == "reduce" ||
          (!o.dim && !o.keepdim && !o.exclude_self && o.reduction == "sum"),
      "Reduction options require operation=reduce");
  TORCH_CHECK(!o.dim || *o.dim == -1 || *o.dim == -2,
              "dim must be None, -1 or -2");
  if (operation == "reduce")
    TORCH_CHECK(o.reduction == "sum" || o.reduction == "mean" ||
                    o.reduction == "min" || o.reduction == "max" ||
                    o.reduction == "prod" || o.reduction == "all" ||
                    o.reduction == "any",
                "Unknown pairwise reduction");
}
Tensor execute_prepared(PreparedComparison prepared, c10::string_view operation,
                        const PairwiseOperationOptions& o, bool autocast) {
  auto cast = [&](Tensor v) {
    return autocast && (v.scalar_type() == at::kHalf ||
                        v.scalar_type() == at::kBFloat16)
               ? v.to(at::kFloat)
               : v;
  };
  auto reduction = o.reduction;
  bool column = operation == "reduce" && o.dim && *o.dim == -2;
  auto& v = prepared.values;
  auto& s = prepared.settings;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  if (operation == "matmul" || operation == "reduce") {
    if (operation == "reduce" && reduction == "prod" &&
        (s.dtype == int64_t(at::kFloat) || s.dtype == int64_t(at::kDouble))) {
      auto value = std::get<0>(streaming_product(v, s, !o.dim));
      if (!o.keepdim)
        return value;
      return o.dim ? value.unsqueeze(column ? -2 : -1)
                   : value.unsqueeze(-1).unsqueeze(-1);
    }
    auto value = std::get<0>(_streaming_reduce(v, s));
    if (operation == "matmul")
      return o.right->dim() == 1 ? value.squeeze(-1) : value;
    if (reduction == "min" || reduction == "max")
      value = value.squeeze(-1);
    if (o.dim) {
      if (reduction == "mean") {
        if (!value.is_floating_point())
          value = value.to(at::kDouble);
        value = value / (prepared.references - int64_t(o.exclude_self));
      }
      return o.keepdim ? value.unsqueeze(column ? -2 : -1) : value;
    }
    if (reduction == "sum" || reduction == "mean")
      value = value.sum(-1);
    else if (reduction == "min")
      value = std::get<0>(value.min(-1));
    else if (reduction == "max")
      value = std::get<0>(value.max(-1));
    else if (reduction == "prod")
      value = value.prod(-1);
    else if (reduction == "all")
      value = value.all(-1);
    else
      value = value.any(-1);
    if (reduction == "mean") {
      if (!value.is_floating_point())
        value = value.to(at::kDouble);
      value = value / (prepared.queries * prepared.references -
                       (o.exclude_self ? prepared.queries : c10::SymInt(0)));
    }
    return o.keepdim ? value.unsqueeze(-1).unsqueeze(-1) : value;
  }
  if (operation == "block" || operation == "diagonal") {
    if (operation == "block") {
      TORCH_CHECK(o.rows && o.columns, "block requires rows and columns");
      for (const auto& ids : {*o.rows, *o.columns})
        TORCH_CHECK(ids.layout() == at::kStrided && ids.dim() == 1 &&
                        ids.scalar_type() == at::kLong &&
                        ids.device() == v[X].device(),
                    "Block indices must be int64 vectors on the input device");
    }
    auto count = prepared.queries.min(prepared.references);
    if (s.expression)
      expression_select(prepared, operation, o);
    else
      for (int64_t slot = 0; slot < Slots; ++slot)
        if (s.has(slot) && s.sampled(slot)) {
          auto axis = -s.operand_rank(slot);
          if (operation == "diagonal")
            v[slot] = v[slot].narrow_symint(axis, 0, count);
          else
            v[slot] = v[slot].index_select(
                axis, (s.first_operand(slot) != s.transposed()) ? *o.rows
                                                                : *o.columns);
        }
    bool self_cosine = s.self_cosine;
    s.self_cosine = false;
    s.operation = (s.transposed() ? "transpose_" : "") +
                  std::string(operation == "diagonal" ? "diagonal" : "entries");
    auto value = std::get<0>(_streaming_reduce(v, s));
    if (self_cosine) {
      if (operation == "diagonal")
        value = at::zeros_like(value) + value * 0.;
      else
        value = at::where(o.rows->unsqueeze(-1).eq(o.columns->unsqueeze(0)), 0.,
                          value);
    }
    return value;
  }
  auto payload = [&](const c10::optional<Tensor>& raw_value,
                     c10::SymInt count) {
    TORCH_CHECK(raw_value, "Missing form operand");
    auto value = cast(*raw_value);
    TORCH_CHECK(
        value.layout() == at::kStrided && value.dim() >= 1 &&
            value.device() == v[X].device() &&
            value.scalar_type() == at::ScalarType(s.dtype),
        "Form operands must be dense and match comparison output dtype/device");
    if (value.dim() == 1)
      value = value.unsqueeze(-1);
    TORCH_CHECK(
        value.sym_size(-2) == count && value.sym_size(-1) > 0,
        "Form operand sample count mismatch or nonpositive channel count");
    return value;
  };
  if (operation == "quadratic")
    TORCH_CHECK(prepared.queries == prepared.references,
                "Quadratic forms require a square operator");
  auto l = payload(o.left, prepared.queries),
       r = payload(operation == "quadratic" ? o.left : o.right,
                   prepared.references);
  auto batch = prepared.batch;
  for (const auto& value : {l, r})
    batch = at::infer_size_symint(batch,
                                  value.sym_sizes().slice(0, value.dim() - 2));
  auto value = pairwise_form(v, l, r, s);
  bool rv = (operation == "quadratic" ? o.left : o.right)->dim() == 1,
       lv = o.left->dim() == 1;
  if (rv)
    value = value.squeeze(-1);
  if (lv)
    value = value.squeeze(rv ? -1 : -2);
  return value;
}
PreparedComparison prepare_pairwise_operator(
    const Tensor& input, const c10::optional<Tensor>& other,
    c10::string_view raw, bool kernel, c10::string_view operation,
    const PairwiseOperationOptions& o) {
  validate_pairwise_options(operation, o);
  bool amp = (input.device().is_cpu() || input.device().is_cuda()) &&
             at::autocast::is_autocast_enabled(input.device().type());
  auto cast = [&](Tensor v) {
    return amp && (v.scalar_type() == at::kHalf ||
                   v.scalar_type() == at::kBFloat16)
               ? v.to(at::kFloat)
               : v;
  };
  StreamingReductionOptions ro;
  ro.comparison = o.comparison;
  ro.chunk_size = o.chunk_size;
  ro.exclude_self = o.exclude_self;
  if (operation == "matmul") {
    TORCH_CHECK(o.right, "matmul requires values");
    ro.values = cast(*o.right);
    if (ro.values->dim() == 1)
      ro.values = ro.values->unsqueeze(-1);
  }
  auto reduction = o.reduction;
  bool column = operation == "reduce" && o.dim && *o.dim == -2;
  auto prepared = prepare(input, other, detail::canonical_comparison_name(raw),
                          operation == "matmul" ? "matmul"
                          : operation == "reduce" && reduction != "mean"
                              ? c10::string_view(reduction)
                              : "sum",
                          kernel, ro, o.transposed != column);
  return prepared;
}
}  // namespace detail::streaming
namespace {
using namespace detail::streaming;
Tensor pairwise_operations_composite(const Tensor& input,
                                     const c10::optional<Tensor>& other,
                                     c10::string_view metric, bool kernel,
                                     c10::string_view operation,
                                     const PairwiseOperationOptions& options) {
  return execute_prepared(prepare_pairwise_operator(input, other, metric,
                                                    kernel, operation, options),
                          operation, options, operator_autocast_enabled(input));
}

Tensor pairwise_operations_impl(
    const Tensor& input, const c10::optional<Tensor>& other,
    c10::string_view metric, bool kernel, c10::string_view operation,
    const c10::optional<Tensor>& left, const c10::optional<Tensor>& right,
    const c10::optional<Tensor>& rows, const c10::optional<Tensor>& columns,
    c10::string_view reduction, c10::optional<int64_t> dim, bool keepdim,
    bool exclude_self, bool transposed, c10::optional<int64_t> chunk_size,
    const c10::optional<Tensor>& w, const c10::optional<Tensor>& V,
    const c10::optional<Tensor>& VI, const c10::optional<Tensor>& bin_positions,
    const c10::optional<Tensor>& projections,
    const c10::optional<Tensor>& x1_lengths,
    const c10::optional<Tensor>& x2_lengths, c10::optional<double> p,
    c10::optional<double> base, c10::optional<double> alpha,
    c10::optional<double> length_scale, c10::optional<double> nu,
    c10::optional<double> gamma, c10::optional<double> coef0,
    c10::optional<double> regularization, c10::optional<int64_t> degree,
    c10::optional<int64_t> max_iter, c10::optional<int64_t> metric_chunk_size,
    c10::optional<int64_t> feature_dim, c10::optional<bool> squared,
    c10::optional<bool> shuffle, c10::optional<bool> batched,
    c10::optional<c10::string_view> family,
    const c10::optional<at::Generator>& generator) {
  PairwiseOperationOptions o;
  o.left = left;
  o.right = right;
  o.rows = rows;
  o.columns = columns;
  o.reduction = std::string(reduction);
  o.dim = dim;
  o.keepdim = keepdim;
  o.exclude_self = exclude_self;
  o.transposed = transposed;
  o.chunk_size = chunk_size;
  o.comparison.w = w;
  o.comparison.V = V;
  o.comparison.VI = VI;
  o.comparison.bin_positions = bin_positions;
  o.comparison.projections = projections;
  o.comparison.x1_lengths = x1_lengths;
  o.comparison.x2_lengths = x2_lengths;
  o.comparison.p = p;
  o.comparison.base = base;
  o.comparison.alpha = alpha;
  o.comparison.length_scale = length_scale;
  o.comparison.nu = nu;
  o.comparison.gamma = gamma;
  o.comparison.coef0 = coef0;
  o.comparison.regularization = regularization;
  o.comparison.degree = degree;
  o.comparison.max_iter = max_iter;
  o.comparison.chunk_size = metric_chunk_size;
  o.comparison.feature_dim = feature_dim;
  o.comparison.squared = squared;
  o.comparison.shuffle = shuffle;
  o.comparison.batched = batched;
  if (family)
    o.comparison.kernel = std::string(*family);
  o.comparison.generator = generator;
  return pairwise_operations_composite(input, other, metric, kernel, operation,
                                       o);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "pairwise_operations(Tensor x1, Tensor? x2=None, str metric='euclidean', "
      "bool is_kernel=False, str operation='reduce', Tensor? left=None, "
      "Tensor? right=None, Tensor? rows=None, Tensor? columns=None, str "
      "reduction='sum', int? dim=None, bool keepdim=False, bool "
      "exclude_self=False, bool transposed=False, int? chunk_size=None, "
      "Tensor? w=None, Tensor? V=None, Tensor? VI=None, Tensor? "
      "bin_positions=None, Tensor? projections=None, Tensor? x1_lengths=None, "
      "Tensor? x2_lengths=None, float? p=None, float? base=None, float? "
      "alpha=None, float? length_scale=None, float? nu=None, float? "
      "gamma=None, float? coef0=None, float? regularization=None, int? "
      "degree=None, int? max_iter=None, int? metric_chunk_size=None, int? "
      "feature_dim=None, bool? squared=None, bool? shuffle=None, bool? "
      "batched=None, str? family=None, Generator? generator=None) -> Tensor",
      TORCH_FN(pairwise_operations_impl));
  m.def(
      "_pairwise_form(Tensor[] operands, Tensor left, Tensor right, str "
      "metric, str operation, bool kernel, int rank, int k, bool largest, bool "
      "exclusion, bool self_cosine, bool nan_first, int size, int dtype, int "
      "mask, float?[] reals, int?[] integers, bool?[] flags, str? family) -> "
      "Tensor");
  m.def(
      "_pairwise_form_backward(Tensor grad, Tensor[] operands, Tensor left, "
      "Tensor right, str metric, str operation, bool kernel, int rank, int k, "
      "bool largest, bool exclusion, bool self_cosine, bool nan_first, int "
      "size, int dtype, int mask, float?[] reals, int?[] integers, bool?[] "
      "flags, str? family) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_pairwise_form", TORCH_FN(pairwise_form_kernel));
  m.impl("_pairwise_form_backward", TORCH_FN(pairwise_form_backward_kernel));
}
}  // namespace
Tensor pairwise_operations(const Tensor& input,
                           const c10::optional<Tensor>& other,
                           c10::string_view metric, bool kernel,
                           c10::string_view operation,
                           const PairwiseOperationOptions& o) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::pairwise_operations", "")
          .typed<decltype(pairwise_operations_impl)>();
  return op.call(
      input, other, metric, kernel, operation, o.left, o.right, o.rows,
      o.columns, o.reduction, o.dim, o.keepdim, o.exclude_self, o.transposed,
      o.chunk_size, o.comparison.w, o.comparison.V, o.comparison.VI,
      o.comparison.bin_positions, o.comparison.projections,
      o.comparison.x1_lengths, o.comparison.x2_lengths, o.comparison.p,
      o.comparison.base, o.comparison.alpha, o.comparison.length_scale,
      o.comparison.nu, o.comparison.gamma, o.comparison.coef0,
      o.comparison.regularization, o.comparison.degree, o.comparison.max_iter,
      o.comparison.chunk_size, o.comparison.feature_dim, o.comparison.squared,
      o.comparison.shuffle, o.comparison.batched,
      o.comparison.kernel
          ? c10::optional<c10::string_view>(*o.comparison.kernel)
          : c10::nullopt,
      o.comparison.generator);
}
}  // namespace torchpairwise::ops
