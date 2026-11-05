#include "linear_eigh.h"

#include <ATen/ExpandUtils.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

#include "linear_plan_detail.h"
#include "utils/eigh.h"

namespace torchpairwise::ops {
namespace {
using namespace detail::linear;
using Result = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
Tensor batch_mask(const Tensor& mask, const Tensor& yes, const Tensor& no) {
  return at::where(mask.unsqueeze(-1).unsqueeze(-1), yes, no);
}
Tensor vector_norms(const Tensor& values) {
  return at::linalg_vector_norm(values.to(at::kDouble), 2., {-2}, false);
}
Tensor finite_block(const Tensor& values) {
  return values.isfinite().all(-1).all(-1);
}
Result iterate(const Plan& a, const Tensor& anchor, int64_t k,
               c10::optional<int64_t> width_option, int64_t max_iter,
               double rtol, double atol, const c10::optional<Tensor>& initial) {
  const auto& matrix_shape = a.shapes.back();
  TORCH_CHECK(matrix_shape.back() == matrix_shape[matrix_shape.size() - 2],
              "eigh requires a square operator");
  auto n = matrix_shape.back().expect_int();
  TORCH_CHECK(k >= 1 && k <= n, "eigh requires 1 <= k <= N");
  if (initial)
    TORCH_CHECK(initial->dim() >= 2,
                "eigh initial requires shape [...,N,subspace_size]");
  auto width = width_option.value_or(
      initial ? initial->size(-1) : std::min(n, k + std::min(n - k, k + 8)));
  TORCH_CHECK(width >= k && width <= n,
              "eigh requires k <= subspace_size <= N");
  auto batch = std::vector<int64_t>();
  for (size_t i = 0; i + 2 < matrix_shape.size(); ++i)
    batch.push_back(matrix_shape[i].expect_int());
  auto shape = batch;
  shape.insert(shape.end(), {n, width});
  Tensor seed;
  if (initial) {
    TORCH_CHECK(initial->layout() == at::kStrided && initial->dim() >= 2 &&
                    initial->scalar_type() == anchor.scalar_type() &&
                    initial->device() == anchor.device(),
                "eigh initial must be strided and match operator dtype/device");
    TORCH_CHECK(initial->size(-2) == n && initial->size(-1) == width,
                "eigh initial requires shape [...,N,subspace_size]");
    TORCH_CHECK(at::infer_size(initial->sizes(), shape) == shape,
                "eigh initial may broadcast to operator batches but cannot add "
                "batch dimensions");
    seed = initial->expand(shape);
  } else {
    seed = at::randn(shape, anchor.options());
  }
  auto pair_shape = batch;
  pair_shape.push_back(k);
  auto vector_shape = batch;
  vector_shape.insert(vector_shape.end(), {n, k});
  auto values = at::full(pair_shape, std::numeric_limits<double>::quiet_NaN(),
                         anchor.options());
  auto vectors = at::full(
      vector_shape, std::numeric_limits<double>::quiet_NaN(), anchor.options());
  auto residual = at::full(pair_shape, std::numeric_limits<double>::quiet_NaN(),
                           anchor.options().dtype(at::kDouble));
  auto iterations = at::zeros(pair_shape, anchor.options().dtype(at::kLong));
  auto converged = at::zeros(pair_shape, anchor.options().dtype(at::kBool));
  auto status = at::ones(pair_shape, anchor.options().dtype(at::kLong));
  // Empty leading batches have no QR/eigh work; N=0 is rejected by k's bounds.
  if (seed.numel() == 0)
    return {values, vectors, residual, iterations, converged, status};
  auto finite = finite_block(seed);
  auto [q, triangular] =
      at::linalg_qr(batch_mask(finite, seed, at::zeros_like(seed)), "reduced");
  double eps = anchor.scalar_type() == at::kDouble
                   ? std::numeric_limits<double>::epsilon()
                   : std::numeric_limits<float>::epsilon();
  auto diagonal = triangular.diagonal(0, -2, -1).abs().to(at::kDouble);
  auto rank_cutoff = std::get<0>(diagonal.max(-1)) * (eps * std::max(n, width));
  auto full_rank = diagonal.gt(rank_cutoff.unsqueeze(-1)).all(-1);
  finite = finite & finite_block(q) & finite_block(triangular);
  auto batch_status = at::where(
      ~finite, 3,
      at::where(~full_rank, 2,
                at::ones_like(finite, finite.options().dtype(at::kLong))));
  auto active = batch_status.eq(1);
  auto identity = at::eye(k, anchor.options().dtype(at::kDouble));
  const double orthogonal_tolerance = 64 * eps * width;
  for (int64_t step = 0; active.any().item<bool>(); ++step) {
    // Preserve fixed-width full QR bases even after low-rank A collapses AQ.
    // QR completion supplies nullspace directions; no division by Ritz values.
    auto aq = a(q);
    finite = finite_block(aq);
    batch_status = at::where(active & ~finite, 3, batch_status);
    auto safe_aq = batch_mask(finite, aq, at::zeros_like(aq));
    auto projected =
        q.to(at::kDouble).transpose(-1, -2).matmul(safe_aq.to(at::kDouble));
    finite = finite_block(projected);
    batch_status = at::where(active & ~finite, 3, batch_status);
    projected = batch_mask(finite, projected, at::zeros_like(projected));
    // Only remove projected roundoff asymmetry. The source itself is unchanged;
    // symmetry/PSD are caller assumptions, not established by this projection.
    auto [spectrum, basis] =
        utils::eigh(projected * .5 + projected.transpose(-1, -2) * .5, "L");
    auto scale = std::get<0>(spectrum.abs().max(-1));
    auto negative =
        spectrum.select(-1, 0).lt(-(32 * eps * width * scale).clamp_min(atol));
    batch_status =
        at::where(active & batch_status.eq(1) & negative, 4, batch_status);
    auto theta = spectrum.slice(-1, width - k, width)
                     .flip({-1})
                     .to(anchor.scalar_type());
    auto u =
        basis.slice(-1, width - k, width).flip({-1}).to(anchor.scalar_type());
    auto candidate = q.matmul(u);
    auto product = a(candidate);
    auto norms = vector_norms(product - candidate * theta.unsqueeze(-2));
    auto orthogonality = vector_norms(candidate.to(at::kDouble)
                                          .transpose(-1, -2)
                                          .matmul(candidate.to(at::kDouble)) -
                                      identity);
    auto finite_pairs =
        norms.isfinite() & theta.isfinite() & orthogonality.isfinite();
    batch_status = at::where(
        active & batch_status.eq(1) & ~finite_pairs.all(-1), 3, batch_status);
    batch_status = at::where(active & batch_status.eq(1) &
                                 orthogonality.gt(orthogonal_tolerance).any(-1),
                             5, batch_status);
    values = at::where(active.unsqueeze(-1), theta, values);
    vectors = batch_mask(active, candidate, vectors);
    residual = at::where(active.unsqueeze(-1), norms, residual);
    iterations = at::where(active.unsqueeze(-1), step, iterations);
    auto meets =
        finite_pairs &
        norms.le((theta.abs().to(at::kDouble) * rtol).clamp_min(atol)) &
        orthogonality.le(orthogonal_tolerance);
    converged = at::where(active.unsqueeze(-1),
                          meets & batch_status.eq(1).unsqueeze(-1), converged);
    auto done = active & batch_status.eq(1) & converged.all(-1);
    batch_status = at::where(done, 0, batch_status);
    active = batch_status.eq(1);
    if (step >= max_iter || !active.any().item<bool>())
      break;
    // Finite zero/rank-deficient AQ is valid for PSD maps. Reduced QR returns
    // a complete orthonormal width-column basis even for a zero map.
    auto next_q = std::get<0>(at::linalg_qr(safe_aq, "reduced"));
    q = batch_mask(active, next_q, q);
  }
  auto failure = batch_status.ge(2);
  auto nan_values =
      at::full_like(values, std::numeric_limits<double>::quiet_NaN());
  values = at::where(failure.unsqueeze(-1), nan_values, values);
  vectors = batch_mask(
      failure, at::full_like(vectors, std::numeric_limits<double>::quiet_NaN()),
      vectors);
  residual = at::where(
      failure.unsqueeze(-1),
      at::full_like(residual, std::numeric_limits<double>::quiet_NaN()),
      residual);
  converged = converged & ~failure.unsqueeze(-1);
  status = at::where(failure.unsqueeze(-1), batch_status.unsqueeze(-1),
                     at::where(converged, 0, status));
  return {values, vectors, residual, iterations, converged, status};
}
Result linear_eigh_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                        c10::SymIntArrayRef shape, const Tensor& anchor,
                        int64_t k, c10::optional<int64_t> subspace_size,
                        int64_t max_iter, double rtol, double atol,
                        const c10::optional<Tensor>& initial) {
  TORCH_CHECK(anchor.layout() == at::kStrided && anchor.dim() == 0 &&
                  (anchor.scalar_type() == at::kFloat ||
                   anchor.scalar_type() == at::kDouble),
              "eigh requires a scalar float32/float64 metadata anchor");
  TORCH_CHECK(k >= 1 && max_iter >= 0 && std::isfinite(rtol) && rtol >= 0 &&
                  std::isfinite(atol) && atol >= 0,
              "eigh requires positive k, nonnegative max_iter and "
              "finite/nonnegative tolerances");
  if (anchor.is_cuda())
    TORCH_CHECK(!c10::impl::VirtualGuardImpl(anchor.device().type())
                     .getStream(anchor.device())
                     .is_capturing(),
                "eigh uses native early stopping and does not support CUDA "
                "graph capture");
  if (at::GradMode::is_enabled()) {
    auto check = [](const Tensor& t) {
      TORCH_CHECK(!t.requires_grad(),
                  "eigh is inference-only; use torch.no_grad() or detach "
                  "inputs explicitly");
    };
    for (const auto& t : inputs)
      check(t);
    for (const auto& t : factors)
      check(t);
    for (const auto& t : linear_factors)
      check(t);
    for (const auto& item : tensors)
      check(item.value());
    check(anchor);
    if (initial)
      check(*initial);
  }
  at::NoGradGuard no_grad;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto graph = unpack_graph(TORCHPAIRWISE_LINEAR_PACKET_ARGS);
  Plan plan(std::move(graph), anchor);
  TORCH_CHECK(c10::SymIntArrayRef(plan.shapes.back()).equals(shape),
              "Bound eigh source shape changed; reconstruct the operator");
  return iterate(plan, anchor, k, subspace_size, max_iter, rtol, atol, initial);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "linear_eigh(Tensor[] inputs, int[] counts, str[] metrics, str[] "
      "families, "
      "Dict(str,Tensor) tensors, Dict(str,float) reals, Dict(str,int) "
      "integers, "
      "Dict(str,bool) flags, Dict(str,str) strings, Dict(str,float[]) "
      "real_lists, "
      "Dict(str,int[]) integer_lists, bool[] transposed, int[] source_counts, "
      "Dict(str,int[]) programs, "
      "Tensor[] factors, int[] factor_counts, Dict(str,int[]) kinds, "
      "Dict(str,int[]) integer_numbers, "
      "Dict(str,float[]) real_numbers, int[] linear_program, Tensor[] "
      "linear_factors, SymInt[] shape, "
      "Tensor anchor, int k, int? subspace_size=None, int max_iter=100, float "
      "rtol=1e-5, float atol=0., "
      "Tensor? initial=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor, "
      "Tensor)",
      {at::Tag::nondeterministic_seeded});
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("linear_eigh", TORCH_FN(linear_eigh_impl));
}
}  // namespace
EigenResult eigh(const LinearOperator& op, const EighOptions& options) {
  auto p = pack_graph(op.graph());
  auto anchor =
      at::empty({}, at::TensorOptions().dtype(op.dtype()).device(op.device()));
  static auto handle = c10::Dispatcher::singleton()
                           .findSchemaOrThrow("torchpairwise::linear_eigh", "")
                           .typed<decltype(linear_eigh_impl)>();
  auto result =
      handle.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(), anchor,
                  options.k, options.subspace_size, options.max_iter,
                  options.rtol, options.atol, options.initial);
  return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
          std::get<3>(result), std::get<4>(result), std::get<5>(result)};
}
}  // namespace torchpairwise::ops
