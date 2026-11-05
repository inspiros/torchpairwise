#include "kernel_moments_detail.h"
#include "kernel_operations.h"
#include "modality_comparisons_detail.h"
#include "utils/comparison_dispatch.h"

namespace torchpairwise::ops {
namespace {
using at::Tensor;
using detail::kernel_moments::Reals;
using detail::kernel_operations::comparison_options;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
Triple kernel_moments_kernel(const Tensor& x, const Tensor& y,
                             const Tensor& origins, c10::string_view name,
                             int64_t size, Reals reals,
                             c10::optional<int64_t> degree,
                             bool compute_second) {
  utils::DeferredValueChecks checks;
  auto batch = modality_detail::batch_shape(x, y, 2);
  batch = at::infer_size_symint(
      batch, origins.sym_sizes().slice(0, origins.dim() - 2));
  auto origin = modality_detail::normalize(origins, batch, 2);
  auto a = modality_detail::normalize(x, batch, 2),
       b = modality_detail::normalize(y, batch, 2);
  auto B = a.size(0), n = a.size(1), m = b.size(1), d = a.size(2);
  auto second = compute_second
                    ? at::zeros({B, n, d, d}, x.options().dtype(at::kDouble))
                    : at::empty({0}, x.options().dtype(at::kDouble));
  auto first = at::zeros({B, n, d}, second.options()),
       mass = at::zeros({B, n}, second.options());
  auto q = comparison_options(reals, degree);
  for (int64_t z = 0; z < B; ++z)
    for (int64_t i = 0; i < n; i += size)
      for (int64_t j = 0; j < m; j += size) {
        auto nr = std::min(size, n - i), mr = std::min(size, m - j);
        auto aa = a.select(0, z).narrow(0, i, nr),
             bb = b.select(0, z).narrow(0, j, mr);
        auto k = builtin_comparisons(aa, bb, name, "all_pairs", true, q)
                     .to(at::kDouble);
        mass.select(0, z).narrow(0, i, nr).add_(k.sum(-1));
        for (int64_t u = 0; u < d; u += size) {
          auto p = std::min(size, d - u);
          auto left = bb.narrow(-1, u, p).to(at::kDouble).unsqueeze(0) -
                      origin.select(0, z)
                          .narrow(0, i, nr)
                          .narrow(-1, u, p)
                          .to(at::kDouble)
                          .unsqueeze(1);
          auto weighted = left * k.unsqueeze(-1);
          first.select(0, z).narrow(0, i, nr).narrow(-1, u, p).add_(
              weighted.sum(1));
          for (int64_t v = 0; compute_second && v < d; v += size) {
            auto t = std::min(size, d - v);
            auto right = bb.narrow(-1, v, t).to(at::kDouble).unsqueeze(0) -
                         origin.select(0, z)
                             .narrow(0, i, nr)
                             .narrow(-1, v, t)
                             .to(at::kDouble)
                             .unsqueeze(1);
            second.select(0, z)
                .narrow(0, i, nr)
                .narrow(1, u, p)
                .narrow(2, v, t)
                .add_(weighted.transpose(1, 2).matmul(right));
          }
        }
      }
  auto shape = batch;
  shape.insert(shape.end(), {x.sym_size(-2), x.sym_size(-1)});
  first = first.reshape_symint(shape);
  shape.pop_back();
  mass = mass.reshape_symint(shape);
  shape.push_back(x.sym_size(-1));
  shape.push_back(x.sym_size(-1));
  if (compute_second)
    second = second.reshape_symint(shape);
  checks.finish();
  return {second, first, mass};
}
std::tuple<Tensor, Tensor, Tensor> kernel_moments_backward_kernel(
    const Tensor& gsecond, const Tensor& gfirst, const Tensor& gmass,
    const Tensor& x, const Tensor& y, const Tensor& origins,
    c10::string_view name, int64_t size, Reals reals,
    c10::optional<int64_t> degree, bool compute_second) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  utils::DeferredValueChecks checks;
  auto batch = modality_detail::batch_shape(x, y, 2);
  batch = at::infer_size_symint(
      batch, origins.sym_sizes().slice(0, origins.dim() - 2));
  auto origin = modality_detail::normalize(origins, batch, 2);
  auto a = modality_detail::normalize(x, batch, 2),
       b = modality_detail::normalize(y, batch, 2);
  auto B = a.size(0), n = a.size(1), m = b.size(1), d = a.size(2);
  auto gx = at::zeros(a.sizes(), a.options().dtype(at::kDouble));
  auto gy = at::zeros(b.sizes(), b.options().dtype(at::kDouble));
  auto go = at::zeros(origin.sizes(), origin.options().dtype(at::kDouble));
  auto gs = compute_second ? gsecond.reshape({B, n, d, d}) : gsecond,
       gf = gfirst.reshape({B, n, d}), gm = gmass.reshape({B, n});
  auto q = comparison_options(reals, degree);
  for (int64_t z = 0; z < B; ++z)
    for (int64_t i = 0; i < n; i += size)
      for (int64_t j = 0; j < m; j += size) {
        auto nr = std::min(size, n - i), mr = std::min(size, m - j);
        auto aa =
            a.select(0, z).narrow(0, i, nr).detach().set_requires_grad(true);
        auto bb =
            b.select(0, z).narrow(0, j, mr).detach().set_requires_grad(true);
        Tensor tile;
        {
          at::AutoGradMode enable(true);
          tile = builtin_comparisons(aa, bb, name, "all_pairs", true, q);
        }
        auto k = tile.detach().to(at::kDouble);
        auto cot = gm.select(0, z)
                       .narrow(0, i, nr)
                       .unsqueeze(-1)
                       .expand({nr, mr})
                       .clone();
        auto add_delta = [&](const Tensor& grad, int64_t u, int64_t p) {
          auto weighted = grad * k.unsqueeze(-1);
          go.select(0, z).narrow(0, i, nr).narrow(-1, u, p).sub_(
              weighted.sum(1));
          gy.select(0, z).narrow(0, j, mr).narrow(-1, u, p).add_(
              weighted.sum(0));
        };
        for (int64_t u = 0; u < d; u += size) {
          auto p = std::min(size, d - u);
          auto left = bb.narrow(-1, u, p).to(at::kDouble).unsqueeze(0) -
                      origin.select(0, z)
                          .narrow(0, i, nr)
                          .narrow(-1, u, p)
                          .to(at::kDouble)
                          .unsqueeze(1);
          auto f =
              gf.select(0, z).narrow(0, i, nr).narrow(-1, u, p).unsqueeze(1);
          cot.add_((f * left).sum(-1));
          add_delta(f, u, p);
          for (int64_t v = 0; compute_second && v < d; v += size) {
            auto t = std::min(size, d - v);
            auto right = bb.narrow(-1, v, t).to(at::kDouble).unsqueeze(0) -
                         origin.select(0, z)
                             .narrow(0, i, nr)
                             .narrow(-1, v, t)
                             .to(at::kDouble)
                             .unsqueeze(1);
            auto g = gs.select(0, z).narrow(0, i, nr).narrow(1, u, p).narrow(
                2, v, t);
            auto dl = right.matmul(g.transpose(1, 2));
            auto dr = left.matmul(g);
            cot.add_((dr * right).sum(-1));
            add_delta(dl, u, p);
            add_delta(dr, v, t);
          }
        }
        std::vector<Tensor> grads;
        {
          at::AutoGradMode enable(true);
          grads = torch::autograd::grad({tile}, {aa, bb},
                                        {cot.to(tile.scalar_type())}, false,
                                        false, true);
        }
        if (grads[0].defined())
          gx.select(0, z).narrow(0, i, nr).add_(grads[0].to(at::kDouble));
        if (grads[1].defined())
          gy.select(0, z).narrow(0, j, mr).add_(grads[1].to(at::kDouble));
      }
  auto xs = batch, ys = batch;
  xs.insert(xs.end(), {x.sym_size(-2), x.sym_size(-1)});
  ys.insert(ys.end(), {y.sym_size(-2), y.sym_size(-1)});
  gx = at::sum_to(gx.reshape_symint(xs), x.sizes())
           .to(x.scalar_type())
           .contiguous();
  gy = at::sum_to(gy.reshape_symint(ys), y.sizes())
           .to(y.scalar_type())
           .contiguous();
  go = at::sum_to(go.reshape_symint(xs), origins.sizes())
           .to(origins.scalar_type())
           .contiguous();
  checks.finish();
  return {gx, gy, go};
}
Tensor kernel_residual_outer_product_impl(
    const Tensor& input, const c10::optional<Tensor>& other,
    c10::string_view raw, bool normalize, bool center,
    c10::optional<int64_t> chunk_size, c10::optional<int64_t> degree,
    c10::optional<double> gamma, c10::optional<double> coef0,
    c10::optional<double> length_scale, c10::optional<double> alpha,
    c10::optional<double> nu) {
  auto name = detail::canonical_comparison_name(raw);
  detail::comparison_family(name, true);
  auto size = chunk_size.value_or(128);
  TORCH_CHECK(size > 0, "chunk_size must be positive");
  auto x = input, y = other.value_or(input);
  TORCH_CHECK(x.device() == y.device() && x.scalar_type() == y.scalar_type(),
              "Inputs must share dtype and device");
  bool amp = (x.device().is_cpu() || x.device().is_cuda()) &&
             at::autocast::is_autocast_enabled(x.device().type());
  if (amp &&
      (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16)) {
    x = x.to(at::kFloat);
    y = y.to(at::kFloat);
  }
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  Reals reals{gamma, coef0, length_scale, alpha, nu};
  auto q = comparison_options(reals, degree);
  TORCH_CHECK(x.dim() >= 2 && y.dim() >= 2, "Kernel moments require rank >= 2");
  // Reuse original-domain/catalog validation, even when one side is empty.
  builtin_comparisons(x, y.narrow(-2, 0, 0), name, "all_pairs", true, q);
  builtin_comparisons(x.narrow(-2, 0, 0), y, name, "all_pairs", true, q);
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble,
              "Kernel moments require float32 or float64 outside autocast");
  // Centering uses a first pass for the mean and a second pass over actual
  // centered residuals; avoid subtracting two large raw second moments.
  auto out = detail::kernel_moments::moments(x, y, x, name, size, reals, degree,
                                             !center);
  auto second = std::get<0>(out), first = std::get<1>(out),
       mass = std::get<2>(out);
  if (center || normalize) {
    auto nonzero = mass.ne(0), safe = at::where(nonzero, mass, 1.);
    if (center) {
      auto origins = x.to(at::kDouble) + first / safe.unsqueeze(-1);
      second = std::get<0>(detail::kernel_moments::moments(
          x, y, origins, name, size, reals, degree));
    }
    if (normalize)
      second = second / safe.unsqueeze(-1).unsqueeze(-1);
    second = at::where(nonzero.unsqueeze(-1).unsqueeze(-1), second, 0.);
  }
  return second.to(x.scalar_type());
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "kernel_residual_outer_product(Tensor x1, Tensor? x2=None, str "
      "kernel='rbf', bool normalize=False, bool center=False, int? "
      "chunk_size=None, int? degree=None, float? gamma=None, float? "
      "coef0=None, float? length_scale=None, float? alpha=None, float? "
      "nu=None) -> Tensor",
      TORCH_FN(kernel_residual_outer_product_impl));
  m.def(
      "_kernel_moments(Tensor x, Tensor y, Tensor origins, str kernel, int "
      "size, float?[] "
      "reals, int? degree, bool second=True) -> (Tensor, Tensor, Tensor)");
  m.def(
      "_kernel_moments_backward(Tensor second_grad, Tensor first_grad, Tensor "
      "mass_grad, Tensor x, Tensor y, Tensor origins, str kernel, int size, "
      "float?[] reals, "
      "int? degree, bool second) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_kernel_moments", TORCH_FN(kernel_moments_kernel));
  m.impl("_kernel_moments_backward", TORCH_FN(kernel_moments_backward_kernel));
}
}  // namespace
Tensor kernel_residual_outer_product(const Tensor& x,
                                     const c10::optional<Tensor>& y,
                                     c10::string_view name, bool normalize,
                                     bool center,
                                     const KernelOperationOptions& o) {
  const auto& q = o.comparison;
  TORCH_CHECK(!o.left && !o.right && !o.rows && !o.columns && !o.dim &&
                  !o.keepdim && !o.exclude_self && o.reduction == "sum",
              "Unused kernel moment operation option");
  TORCH_CHECK(!q.w && !q.V && !q.VI && !q.bin_positions && !q.projections &&
                  !q.x1_lengths && !q.x2_lengths && !q.p && !q.base &&
                  !q.regularization && !q.max_iter && !q.chunk_size &&
                  !q.feature_dim && !q.squared && !q.shuffle && !q.batched &&
                  !q.kernel && !q.generator,
              "Unused non-kernel comparison option");
  using Signature = Tensor(const Tensor&, const c10::optional<Tensor>&,
                           c10::string_view, bool, bool, c10::optional<int64_t>,
                           c10::optional<int64_t>, c10::optional<double>,
                           c10::optional<double>, c10::optional<double>,
                           c10::optional<double>, c10::optional<double>);
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::kernel_residual_outer_product", "")
          .typed<Signature>();
  return op.call(x, y, name, normalize, center, o.chunk_size, q.degree, q.gamma,
                 q.coef0, q.length_scale, q.alpha, q.nu);
}
}  // namespace torchpairwise::ops
