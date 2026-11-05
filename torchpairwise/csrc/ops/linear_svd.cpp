#include "linear_svd.h"

#include <ATen/ExpandUtils.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

#include "linear_plan_detail.h"

namespace torchpairwise::ops {
namespace {
using namespace detail::linear;
using Result =
    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
Tensor select_batch(const Tensor& mask, const Tensor& yes, const Tensor& no) {
  return at::where(mask.unsqueeze(-1).unsqueeze(-1), yes, no);
}
Tensor finite_block(const Tensor& value) {
  return value.isfinite().all(-1).all(-1);
}
Tensor column_norm(const Tensor& value) {
  return at::linalg_vector_norm(value.to(at::kDouble), 2., {-2}, false);
}
Result iterate(const Plan& a, const Tensor& anchor, int64_t k,
               c10::optional<int64_t> width_option, int64_t max_iter,
               double rtol, double atol, const c10::optional<Tensor>& initial) {
  const auto& shape = a.shapes.back();
  auto n = shape[shape.size() - 2].expect_int(), m = shape.back().expect_int();
  auto rank = std::min(n, m);
  TORCH_CHECK(k >= 1 && k <= rank, "svd requires 1 <= k <= min(N,M)");
  if (initial)
    TORCH_CHECK(initial->dim() >= 2, "svd initial must be [...,M,r]");
  auto width = width_option.value_or(
      initial ? initial->size(-1)
              : std::min(rank, k + std::min(rank - k, k + 8)));
  TORCH_CHECK(width >= k && width <= rank,
              "svd requires k <= subspace_size <= min(N,M)");
  std::vector<int64_t> batch;
  for (size_t i = 0; i + 2 < shape.size(); ++i)
    batch.push_back(shape[i].expect_int());
  auto right_shape = batch;
  right_shape.insert(right_shape.end(), {m, width});
  Tensor seed;
  if (initial) {
    TORCH_CHECK(initial->layout() == at::kStrided && initial->dim() >= 2 &&
                    initial->scalar_type() == anchor.scalar_type() &&
                    initial->device() == anchor.device(),
                "svd initial must be strided and match source dtype/device");
    TORCH_CHECK(
        initial->size(-2) == m && initial->size(-1) == width &&
            at::infer_size(initial->sizes(), right_shape) == right_shape,
        "svd initial must match [...,M,r] and cannot add batch dimensions");
    seed = initial->expand(right_shape);
  } else
    seed = at::randn(right_shape, anchor.options());
  auto pair_shape = batch;
  pair_shape.push_back(k);
  auto left_shape = batch;
  left_shape.insert(left_shape.end(), {n, k});
  auto vector_shape = batch;
  vector_shape.insert(vector_shape.end(), {m, k});
  double nan = std::numeric_limits<double>::quiet_NaN();
  auto values = at::full(pair_shape, nan, anchor.options());
  auto left = at::full(left_shape, nan, anchor.options());
  auto right = at::full(vector_shape, nan, anchor.options());
  auto residual =
      at::full(pair_shape, nan, anchor.options().dtype(at::kDouble));
  auto iterations = at::zeros(pair_shape, anchor.options().dtype(at::kLong));
  auto converged = at::zeros(pair_shape, anchor.options().dtype(at::kBool));
  auto status = at::ones_like(iterations);
  if (seed.numel() == 0)
    return {values, left, right, residual, iterations, converged, status};
  double eps = anchor.scalar_type() == at::kDouble
                   ? std::numeric_limits<double>::epsilon()
                   : std::numeric_limits<float>::epsilon();
  auto finite = finite_block(seed);
  auto [q, r] = at::linalg_qr(
      select_batch(finite, seed, at::zeros_like(seed)).to(at::kDouble),
      "reduced");
  auto diagonal = r.diagonal(0, -2, -1).abs();
  auto cutoff = std::get<0>(diagonal.max(-1)) * (eps * std::max(m, width));
  auto full_rank = diagonal.gt(cutoff.unsqueeze(-1)).all(-1);
  finite = finite & finite_block(q) & finite_block(r);
  auto batch_status = at::where(
      ~finite, 3,
      at::where(~full_rank, 2,
                at::ones_like(finite, finite.options().dtype(at::kLong))));
  auto active = batch_status.eq(1);
  auto identity = at::eye(k, anchor.options().dtype(at::kDouble));
  double orthogonal_tolerance = 64 * eps * width;
  auto action = [&](const Tensor& block, bool transpose = false) {
    return a
        .apply(a.graph.nodes.size() - 1, block.to(anchor.scalar_type()),
               transpose)
        .to(at::kDouble);
  };
  for (int64_t step = 0; active.any().item<bool>(); ++step) {
    // Two-sided QR stabilization, followed by an r-by-r projected SVD.
    // Finite rank collapse is valid; QR completes null-space basis columns.
    auto aq = action(q);
    finite = finite_block(aq);
    auto l = std::get<0>(
        at::linalg_qr(select_batch(finite, aq, at::zeros_like(aq)), "reduced"));
    auto atl = action(l, true);
    finite = finite & finite_block(atl);
    auto [next_q, triangular] = at::linalg_qr(
        select_batch(finite, atl, at::zeros_like(atl)), "reduced");
    finite = finite & finite_block(l) & finite_block(next_q) &
             finite_block(triangular);
    batch_status = at::where(active & ~finite, 3, batch_status);
    auto projected = select_batch(finite, triangular.transpose(-1, -2),
                                  at::zeros_like(triangular));
    auto [small_left, spectrum, small_vh] = at::linalg_svd(projected, false);
    auto theta = spectrum.slice(-1, 0, k).to(anchor.scalar_type());
    auto u = l.matmul(small_left.slice(-1, 0, k)).to(anchor.scalar_type());
    auto v = next_q.matmul(small_vh.slice(-2, 0, k).transpose(-1, -2))
                 .to(anchor.scalar_type());
    auto forward_residual = column_norm(
        action(v) - u.to(at::kDouble) * theta.to(at::kDouble).unsqueeze(-2));
    auto transpose_residual =
        column_norm(action(u, true) -
                    v.to(at::kDouble) * theta.to(at::kDouble).unsqueeze(-2));
    auto norms = at::maximum(forward_residual, transpose_residual);
    auto orthogonality = at::maximum(
        column_norm(
            u.to(at::kDouble).transpose(-1, -2).matmul(u.to(at::kDouble)) -
            identity),
        column_norm(
            v.to(at::kDouble).transpose(-1, -2).matmul(v.to(at::kDouble)) -
            identity));
    auto finite_pairs =
        norms.isfinite() & theta.isfinite() & orthogonality.isfinite();
    batch_status = at::where(
        active & batch_status.eq(1) & ~finite_pairs.all(-1), 3, batch_status);
    batch_status = at::where(active & batch_status.eq(1) &
                                 orthogonality.gt(orthogonal_tolerance).any(-1),
                             4, batch_status);
    values = at::where(active.unsqueeze(-1), theta, values);
    left = select_batch(active, u, left);
    right = select_batch(active, v, right);
    residual = at::where(active.unsqueeze(-1), norms, residual);
    iterations = at::where(active.unsqueeze(-1), step, iterations);
    auto meets = finite_pairs &
                 norms.le((theta.to(at::kDouble) * rtol).clamp_min(atol)) &
                 orthogonality.le(orthogonal_tolerance);
    converged = at::where(active.unsqueeze(-1),
                          meets & batch_status.eq(1).unsqueeze(-1), converged);
    batch_status = at::where(active & batch_status.eq(1) & converged.all(-1), 0,
                             batch_status);
    active = batch_status.eq(1);
    if (step >= max_iter || !active.any().item<bool>())
      break;
    q = select_batch(active, next_q, q);
  }
  auto failure = batch_status.ge(2);
  values = at::where(failure.unsqueeze(-1), at::full_like(values, nan), values);
  left = select_batch(failure, at::full_like(left, nan), left);
  right = select_batch(failure, at::full_like(right, nan), right);
  residual =
      at::where(failure.unsqueeze(-1), at::full_like(residual, nan), residual);
  converged = converged & ~failure.unsqueeze(-1);
  status = at::where(failure.unsqueeze(-1), batch_status.unsqueeze(-1),
                     at::where(converged, 0, status));
  return {values, left, right, residual, iterations, converged, status};
}
Result linear_svd_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                       c10::SymIntArrayRef shape, const Tensor& anchor,
                       int64_t k, c10::optional<int64_t> subspace_size,
                       int64_t max_iter, double rtol, double atol,
                       const c10::optional<Tensor>& initial) {
  TORCH_CHECK(anchor.layout() == at::kStrided && anchor.dim() == 0 &&
                  (anchor.scalar_type() == at::kFloat ||
                   anchor.scalar_type() == at::kDouble),
              "svd requires a scalar float32/float64 metadata anchor");
  TORCH_CHECK(k >= 1 && max_iter >= 0 && std::isfinite(rtol) && rtol >= 0 &&
                  std::isfinite(atol) && atol >= 0,
              "svd requires positive k, nonnegative max_iter and "
              "finite/nonnegative tolerances");
  if (anchor.is_cuda())
    TORCH_CHECK(
        !c10::impl::VirtualGuardImpl(anchor.device().type())
             .getStream(anchor.device())
             .is_capturing(),
        "svd uses native stopping and does not support CUDA graph capture");
  if (at::GradMode::is_enabled()) {
    auto check = [](const Tensor& value) {
      TORCH_CHECK(!value.requires_grad(),
                  "svd is inference-only; use torch.no_grad() or detach inputs "
                  "explicitly");
    };
    for (const auto& value : inputs)
      check(value);
    for (const auto& value : factors)
      check(value);
    for (const auto& value : linear_factors)
      check(value);
    for (const auto& item : tensors)
      check(item.value());
    check(anchor);
    if (initial)
      check(*initial);
  }
  at::NoGradGuard no_grad;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  Plan plan(unpack_graph(TORCHPAIRWISE_LINEAR_PACKET_ARGS), anchor);
  TORCH_CHECK(c10::SymIntArrayRef(plan.shapes.back()).equals(shape),
              "Bound svd source shape changed; reconstruct the operator");
  return iterate(plan, anchor, k, subspace_size, max_iter, rtol, atol, initial);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "linear_svd(Tensor[] inputs, int[] counts, str[] metrics, str[] "
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
      "Tensor, Tensor)",
      {at::Tag::nondeterministic_seeded});
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("linear_svd", TORCH_FN(linear_svd_impl));
}
}  // namespace
SVDResult svd(const LinearOperator& op, const SVDOptions& options) {
  auto p = detail::linear::pack_graph(op.graph());
  auto anchor =
      at::empty({}, at::TensorOptions().dtype(op.dtype()).device(op.device()));
  static auto handle = c10::Dispatcher::singleton()
                           .findSchemaOrThrow("torchpairwise::linear_svd", "")
                           .typed<decltype(linear_svd_impl)>();
  auto result =
      handle.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(), anchor,
                  options.k, options.subspace_size, options.max_iter,
                  options.rtol, options.atol, options.initial);
  return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
          std::get<3>(result), std::get<4>(result), std::get<5>(result),
          std::get<6>(result)};
}
}  // namespace torchpairwise::ops
