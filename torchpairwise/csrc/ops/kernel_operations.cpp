#include "kernel_operations.h"

#include <ATen/ExpandUtils.h>
#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include "kernel_operations_detail.h"
#include "pairwise_operations.h"
#include "streaming_reductions.h"
#include "utils/comparison_dispatch.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops {
namespace {
using at::Tensor;
using detail::kernel_operations::comparison_options;
using detail::kernel_operations::Reals;
std::vector<int64_t> batch_shape(const Tensor& x, const Tensor& y,
                                 const Tensor& left, const Tensor& right) {
  auto shape = at::infer_size(x.sizes().slice(0, x.dim() - 2),
                              y.sizes().slice(0, y.dim() - 2));
  for (const auto& value : {left, right})
    shape = at::infer_size(shape, value.sizes().slice(0, value.dim() - 2));
  return shape;
}
std::vector<int64_t> coordinates(int64_t flat,
                                 const std::vector<int64_t>& batch) {
  std::vector<int64_t> index(batch.size());
  for (int64_t i = int64_t(batch.size()) - 1; i >= 0; --i) {
    index[i] = flat % batch[i];
    flat /= batch[i];
  }
  return index;
}
Tensor lookup(Tensor value, const std::vector<int64_t>& index) {
  auto rank = value.dim() - 2, offset = int64_t(index.size()) - rank;
  for (int64_t i = 0; i < rank; ++i)
    value = value.select(0, value.size(0) == 1 ? 0 : index[offset + i]);
  return value;
}
void add_compensated(Tensor total, Tensor correction, const Tensor& term) {
  auto updated = total + term;
  auto valid = total.isfinite() & term.isfinite() & updated.isfinite();
  auto residual = at::where(total.abs().ge(term.abs()),
                            (total - updated) + term, (term - updated) + total);
  correction.copy_(at::where(valid, correction + residual, 0.));
  total.copy_(updated);
}
Tensor kernel_form_kernel(const Tensor& x, const Tensor& y, const Tensor& left,
                          const Tensor& right, c10::string_view name,
                          int64_t size, Reals reals,
                          c10::optional<int64_t> degree) {
  utils::DeferredValueChecks checks;
  auto batch = batch_shape(x, y, left, right), shape = batch;
  shape.insert(shape.end(), {left.size(-1), right.size(-1)});
  auto result = at::zeros(shape, x.options()),
       correction = at::zeros_like(result);
  int64_t batches = 1;
  for (auto dim : batch)
    batches *= dim;
  auto q = comparison_options(reals, degree);
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    auto a = lookup(x, index), b = lookup(y, index);
    auto l = lookup(left, index), r = lookup(right, index);
    for (int64_t i = 0; i < a.size(0); i += size)
      for (int64_t j = 0; j < b.size(0); j += size) {
        auto n = std::min(size, a.size(0) - i),
             m = std::min(size, b.size(0) - j);
        auto tile = builtin_comparisons(a.narrow(0, i, n), b.narrow(0, j, m),
                                        name, "all_pairs", true, q);
        for (int64_t u = 0; u < l.size(-1); u += size) {
          auto p = std::min(size, l.size(-1) - u);
          auto lp =
              l.narrow(0, i, n).narrow(-1, u, p).transpose(0, 1).matmul(tile);
          for (int64_t v = 0; v < r.size(-1); v += size) {
            auto t = std::min(size, r.size(-1) - v);
            auto term = lp.matmul(r.narrow(0, j, m).narrow(-1, v, t));
            add_compensated(
                lookup(result, index).narrow(0, u, p).narrow(1, v, t),
                lookup(correction, index).narrow(0, u, p).narrow(1, v, t),
                term);
          }
        }
      }
  }
  checks.finish();
  return result + correction;
}
std::tuple<Tensor, Tensor, Tensor, Tensor> kernel_form_backward_kernel(
    const Tensor& gradient, const Tensor& x, const Tensor& y,
    const Tensor& left, const Tensor& right, c10::string_view name,
    int64_t size, Reals reals, c10::optional<int64_t> degree) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  utils::DeferredValueChecks checks;
  auto gx = at::zeros(x.sizes(), x.options().dtype(at::kDouble));
  auto gy = at::zeros(y.sizes(), y.options().dtype(at::kDouble));
  auto gl = at::zeros(left.sizes(), left.options().dtype(at::kDouble));
  auto gr = at::zeros(right.sizes(), right.options().dtype(at::kDouble));
  auto batch = batch_shape(x, y, left, right);
  int64_t batches = 1;
  for (auto dim : batch)
    batches *= dim;
  auto q = comparison_options(reals, degree);
  at::AutoGradMode enable(true);
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    auto a = lookup(x, index), b = lookup(y, index);
    auto l = lookup(left, index).detach(), r = lookup(right, index).detach();
    auto g = lookup(gradient, index).detach();
    for (int64_t i = 0; i < a.size(0); i += size)
      for (int64_t j = 0; j < b.size(0); j += size) {
        auto n = std::min(size, a.size(0) - i),
             m = std::min(size, b.size(0) - j);
        auto aa = a.narrow(0, i, n).detach().set_requires_grad(true);
        auto bb = b.narrow(0, j, m).detach().set_requires_grad(true);
        auto tile = builtin_comparisons(aa, bb, name, "all_pairs", true, q);
        auto cotangent = at::zeros({n, m}, x.options().dtype(at::kDouble));
        {
          at::NoGradGuard guard;
          auto k = tile.detach().to(at::kDouble);
          for (int64_t u = 0; u < l.size(-1); u += size) {
            auto p = std::min(size, l.size(-1) - u);
            auto lp = l.narrow(0, i, n).narrow(-1, u, p).to(at::kDouble);
            for (int64_t v = 0; v < r.size(-1); v += size) {
              auto t = std::min(size, r.size(-1) - v);
              auto rp = r.narrow(0, j, m).narrow(-1, v, t).to(at::kDouble);
              auto gp = g.narrow(0, u, p).narrow(1, v, t).to(at::kDouble);
              cotangent.add_(lp.matmul(gp).matmul(rp.transpose(0, 1)));
              lookup(gl, index).narrow(0, i, n).narrow(1, u, p).add_(
                  k.matmul(rp).matmul(gp.transpose(0, 1)));
              lookup(gr, index).narrow(0, j, m).narrow(1, v, t).add_(
                  k.transpose(0, 1).matmul(lp).matmul(gp));
            }
          }
        }
        auto gradients = torch::autograd::grad(
            {tile}, {aa, bb}, {cotangent.to(tile.scalar_type())}, false, false,
            true);
        at::NoGradGuard guard;
        if (gradients[0].defined())
          lookup(gx, index).narrow(0, i, n).add_(gradients[0].to(at::kDouble));
        if (gradients[1].defined())
          lookup(gy, index).narrow(0, j, m).add_(gradients[1].to(at::kDouble));
      }
  }
  checks.finish();
  return {gx.to(x.scalar_type()), gy.to(y.scalar_type()),
          gl.to(left.scalar_type()), gr.to(right.scalar_type())};
}
Tensor kernel_operations_impl(
    const Tensor& input, const c10::optional<Tensor>& other,
    c10::string_view raw, c10::string_view operation,
    const c10::optional<Tensor>& left, const c10::optional<Tensor>& right,
    const c10::optional<Tensor>& rows, const c10::optional<Tensor>& columns,
    c10::string_view reduction, c10::optional<int64_t> dim, bool keepdim,
    bool exclude_self, c10::optional<int64_t> chunk_size,
    c10::optional<int64_t> degree, c10::optional<double> gamma,
    c10::optional<double> coef0, c10::optional<double> length_scale,
    c10::optional<double> alpha, c10::optional<double> nu) {
  PairwiseOperationOptions options;
  auto& q = options.comparison;
  q.degree = degree;
  q.gamma = gamma;
  q.coef0 = coef0;
  q.length_scale = length_scale;
  q.alpha = alpha;
  q.nu = nu;
  options.chunk_size = chunk_size;
  if (operation == "displacement" || operation == "displacement_normalized") {
    TORCH_CHECK(!left && !right && !rows && !columns && !dim && !keepdim &&
                    !exclude_self && reduction == "sum",
                "Unused displacement operation option");
    auto y = other.value_or(input);
    auto ones = at::ones_like(y.narrow(-1, 0, 1));
    options.right = at::cat({y, ones}, -1);
    auto combined =
        pairwise_operations(input, other, raw, true, "matmul", options);
    auto mass = combined.narrow_symint(-1, y.sym_size(-1), 1);
    auto value = combined.narrow_symint(-1, 0, y.sym_size(-1)) - input * mass;
    if (operation == "displacement_normalized") {
      auto nonzero = mass.ne(0);
      value = at::where(nonzero, value / at::where(nonzero, mass, 1.), 0.);
    }
    return value;
  }
  PairwiseOperationOptions o;
  o.comparison = q;
  o.left = left;
  o.right = right;
  o.rows = rows;
  o.columns = columns;
  o.reduction = std::string(reduction);
  o.dim = dim;
  o.keepdim = keepdim;
  o.exclude_self = exclude_self;
  o.chunk_size = chunk_size;
  return pairwise_operations(input, other, raw, true, operation, o);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "kernel_operations(Tensor x1, Tensor? x2=None, str kernel='rbf', str "
      "operation='reduce', Tensor? left=None, Tensor? right=None, Tensor? "
      "rows=None, Tensor? columns=None, str reduction='sum', int? dim=None, "
      "bool keepdim=False, bool exclude_self=False, int? chunk_size=None, int? "
      "degree=None, float? gamma=None, float? coef0=None, float? "
      "length_scale=None, float? alpha=None, float? nu=None) -> Tensor",
      TORCH_FN(kernel_operations_impl));
  m.def(
      "_kernel_form(Tensor x, Tensor y, Tensor left, Tensor right, str kernel, "
      "int size, float?[] reals, int? degree) -> Tensor");
  m.def(
      "_kernel_form_backward(Tensor grad, Tensor x, Tensor y, Tensor left, "
      "Tensor right, str kernel, int size, float?[] reals, int? degree) -> "
      "(Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_kernel_form", TORCH_FN(kernel_form_kernel));
  m.impl("_kernel_form_backward", TORCH_FN(kernel_form_backward_kernel));
}
}  // namespace
Tensor kernel_operations(const Tensor& x, const c10::optional<Tensor>& y,
                         c10::string_view name, c10::string_view operation,
                         const KernelOperationOptions& o) {
  // The dispatcher exposes kernel scalar options only. Reject other union
  // fields explicitly rather than silently dropping a C++ caller's options.
  const auto& q = o.comparison;
  TORCH_CHECK(!q.w && !q.V && !q.VI && !q.bin_positions && !q.projections &&
                  !q.x1_lengths && !q.x2_lengths && !q.p && !q.base &&
                  !q.regularization && !q.max_iter && !q.chunk_size &&
                  !q.feature_dim && !q.squared && !q.shuffle && !q.batched &&
                  !q.kernel && !q.generator,
              "Unused non-kernel comparison option");
  using Signature = Tensor(
      const Tensor&, const c10::optional<Tensor>&, c10::string_view,
      c10::string_view, const c10::optional<Tensor>&,
      const c10::optional<Tensor>&, const c10::optional<Tensor>&,
      const c10::optional<Tensor>&, c10::string_view, c10::optional<int64_t>,
      bool, bool, c10::optional<int64_t>, c10::optional<int64_t>,
      c10::optional<double>, c10::optional<double>, c10::optional<double>,
      c10::optional<double>, c10::optional<double>);
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::kernel_operations", "")
          .typed<Signature>();
  return op.call(x, y, name, operation, o.left, o.right, o.rows, o.columns,
                 o.reduction, o.dim, o.keepdim, o.exclude_self, o.chunk_size,
                 q.degree, q.gamma, q.coef0, q.length_scale, q.alpha, q.nu);
}
}  // namespace torchpairwise::ops
