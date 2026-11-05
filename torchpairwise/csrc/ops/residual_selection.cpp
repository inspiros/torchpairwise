#include <ATen/ExpandUtils.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include "residual_operations.h"
#include "residual_operations_detail.h"

namespace torchpairwise::ops {
namespace {
using at::Tensor;
using detail::residual::FeatureOutputs;
using namespace detail::streaming;
void validate_topk(const Tensor& x, const Tensor& y, int64_t k,
                   c10::string_view transform, int64_t sample_size,
                   int64_t feature_size) {
  TORCH_CHECK(
      x.dim() >= 2 && y.dim() >= 2 && x.layout() == at::kStrided &&
          y.layout() == at::kStrided && x.device() == y.device() &&
          x.scalar_type() == y.scalar_type() &&
          (x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble),
      "Residual topk requires dense matching float32/float64 tensors");
  TORCH_CHECK(x.sym_size(-1) == y.sym_size(-1),
              "Residual feature counts differ");
  TORCH_CHECK(k >= 0 && k <= x.size(-1),
              "k exceeds the residual feature count");
  TORCH_CHECK(
      transform == "identity" || transform == "abs" || transform == "square",
      "Residual transform must be identity, abs or square");
  TORCH_CHECK(sample_size > 0 && feature_size > 0,
              "Tile sizes must be positive");
}
Tensor transform_value(const Tensor& value, c10::string_view transform) {
  return transform == "abs"      ? value.abs()
         : transform == "square" ? value.square()
                                 : value;
}
Tensor pair_tile(Tensor value, int64_t i, int64_t ni, int64_t j, int64_t nj) {
  return value.narrow(0, i, ni).narrow(1, j, nj);
}
Tensor residual_tile(const Tensor& a, const Tensor& b, int64_t i, int64_t ni,
                     int64_t j, int64_t nj, int64_t d, int64_t nd) {
  return a.narrow(0, i, ni).narrow(1, d, nd).unsqueeze(1) -
         b.narrow(0, j, nj).narrow(1, d, nd).unsqueeze(0);
}
FeatureOutputs topk_kernel(const Tensor& x, const Tensor& y, int64_t k,
                           bool largest, c10::string_view transform,
                           int64_t sample_size, int64_t feature_size) {
  validate_topk(x, y, k, transform, sample_size, feature_size);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::NoGradGuard guard;
  auto batch = at::infer_size(x.sizes().slice(0, x.dim() - 2),
                              y.sizes().slice(0, y.dim() - 2));
  auto shape = batch;
  auto n = x.size(-2), m = y.size(-2), d = x.size(-1);
  shape.insert(shape.end(), {n, m, k});
  auto output = at::empty(shape, x.options());
  auto indices = at::empty(shape, x.options().dtype(at::kLong));
  if (output.numel() == 0)
    return {output, indices};
  int64_t batches = 1;
  for (auto width : batch)
    batches *= width;
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    auto a = lookup(x, index, 2), b = lookup(y, index, 2);
    auto out = lookup(output, index, 3), ids = lookup(indices, index, 3);
    for (int64_t i = 0; i < n; i += sample_size) {
      auto ni = std::min(sample_size, n - i);
      for (int64_t j = 0; j < m; j += sample_size) {
        auto nj = std::min(sample_size, m - j);
        auto best = at::empty({ni, nj, 0}, x.options());
        auto selected = at::empty({ni, nj, 0}, indices.options());
        for (int64_t start = 0; start < d; start += feature_size) {
          auto width = std::min(feature_size, d - start);
          auto values = transform_value(
              residual_tile(a, b, i, ni, j, nj, start, width), transform);
          auto positions = at::arange(start, start + width, indices.options())
                               .expand({ni, nj, width});
          values = at::cat({best, values}, -1);
          positions = at::cat({selected, positions}, -1);
          // Earlier chunks contain lower IDs; their tie order is already
          // stable. Newly appended features are in increasing ID order.
          auto order = at::argsort(values, true, -1, largest)
                           .narrow(-1, 0, std::min(k, values.size(-1)));
          best = values.gather(-1, order);
          selected = positions.gather(-1, order);
        }
        pair_tile(out, i, ni, j, nj).copy_(best);
        pair_tile(ids, i, ni, j, nj).copy_(selected);
      }
    }
  }
  return {output, indices};
}
FeatureOutputs topk_backward_kernel(const Tensor& gradient, const Tensor& x,
                                    const Tensor& y, const Tensor& indices,
                                    c10::string_view transform,
                                    int64_t sample_size, int64_t feature_size) {
  TORCH_CHECK(indices.dim() >= 3, "Invalid residual topk indices");
  auto k = indices.size(-1);
  validate_topk(x, y, k, transform, sample_size, feature_size);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::NoGradGuard guard;
  auto batch = at::infer_size(x.sizes().slice(0, x.dim() - 2),
                              y.sizes().slice(0, y.dim() - 2));
  auto n = x.size(-2), m = y.size(-2), d = x.size(-1);
  auto shape = batch;
  shape.insert(shape.end(), {n, m, k});
  TORCH_CHECK(indices.sizes() == at::IntArrayRef(shape) &&
                  gradient.sizes() == indices.sizes() &&
                  indices.scalar_type() == at::kLong &&
                  indices.device() == x.device() &&
                  gradient.scalar_type() == x.scalar_type() &&
                  gradient.device() == x.device(),
              "Invalid residual topk gradient or indices");
  auto gx = at::zeros(x.sizes(), x.options().dtype(at::kDouble));
  auto gy = at::zeros(y.sizes(), y.options().dtype(at::kDouble));
  int64_t batches = 1;
  for (auto width : batch)
    batches *= width;
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    auto a = lookup(x, index, 2), b = lookup(y, index, 2);
    auto da = lookup(gx, index, 2), db = lookup(gy, index, 2);
    auto g = lookup(gradient, index, 3), ids = lookup(indices, index, 3);
    for (int64_t i = 0; i < n; i += sample_size) {
      auto ni = std::min(sample_size, n - i);
      for (int64_t j = 0; j < m; j += sample_size) {
        auto nj = std::min(sample_size, m - j);
        auto witnesses = pair_tile(ids, i, ni, j, nj);
        auto cotangents = pair_tile(g, i, ni, j, nj);
        if (transform == "identity" || transform == "abs") {
          // Signed/absolute derivatives at unselected coordinates are zero,
          // even for nonfinite residuals (ATen sign(NaN) is zero).
          // Gather/scatter only witnesses; expanded input views do not allocate
          // N*M*D.
          for (int64_t rank = 0; rank < k; rank += feature_size) {
            auto count = std::min(feature_size, k - rank);
            auto selected = witnesses.narrow(-1, rank, count);
            auto values = cotangents.narrow(-1, rank, count).to(at::kDouble);
            if (transform == "abs") {
              auto residual = a.narrow(0, i, ni)
                                  .unsqueeze(1)
                                  .expand({ni, nj, d})
                                  .gather(-1, selected) -
                              b.narrow(0, j, nj)
                                  .unsqueeze(0)
                                  .expand({ni, nj, d})
                                  .gather(-1, selected);
              values = values * residual.sign();
            }
            da.narrow(0, i, ni).scatter_add_(1,
                                             selected.reshape({ni, nj * count}),
                                             values.reshape({ni, nj * count}));
            db.narrow(0, j, nj).scatter_add_(
                1, selected.transpose(0, 1).reshape({nj, ni * count}),
                -values.transpose(0, 1).reshape({nj, ni * count}));
          }
          continue;
        }
        for (int64_t start = 0; start < d; start += feature_size) {
          auto width = std::min(feature_size, d - start);
          auto cot = at::zeros({ni, nj, width}, gx.options());
          for (int64_t rank = 0; rank < k; rank += feature_size) {
            auto count = std::min(feature_size, k - rank);
            auto local = witnesses.narrow(-1, rank, count) - start;
            auto eligible = local.ge(0) & local.lt(width);
            auto values = at::where(
                eligible, cotangents.narrow(-1, rank, count).to(at::kDouble),
                0.);
            cot.scatter_add_(-1, local.clamp(0, width - 1), values);
          }
          // Replay every feature for nonlinear transforms: even an unselected
          // infinite residual has the dense square derivative 0*Inf = NaN.
          if (transform != "identity") {
            auto residual =
                residual_tile(a, b, i, ni, j, nj, start, width).to(at::kDouble);
            cot = cot * (transform == "abs" ? residual.sign() : 2. * residual);
          }
          da.narrow(0, i, ni).narrow(1, start, width).add_(cot.sum(1));
          db.narrow(0, j, nj).narrow(1, start, width).sub_(cot.sum(0));
        }
      }
    }
  }
  return {gx.to(x.scalar_type()), gy.to(y.scalar_type())};
}
FeatureOutputs residual_topk_impl(const Tensor& x, int64_t k,
                                  const c10::optional<Tensor>& y, int64_t dim,
                                  bool largest, bool sorted,
                                  c10::string_view transform,
                                  int64_t sample_size, int64_t feature_size) {
  PairwiseOperationOptions options;
  options.chunk_size = sample_size;
  auto prepared = prepare_residual_operator(x, y, "sum", transform, 1.,
                                            feature_size, options);
  auto rank = int64_t(prepared.batch.size()) + 3;
  TORCH_CHECK(dim == -1 || dim == rank - 1,
              "Residual topk currently supports only the feature axis");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_residual_topk", "")
                       .typed<detail::residual::TopKSignature>();
  // sorted=False permits sorted output, as with the scalar top-k consumers.
  return op.call(prepared.values[X], prepared.values[Y], k, largest, transform,
                 sample_size, feature_size);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "residual_topk(Tensor x, int k, Tensor? y=None, int dim=-1, bool "
      "largest=True, bool sorted=True, str transform='identity', int "
      "chunk_size=128, int feature_chunk_size=128) -> (Tensor, Tensor)",
      TORCH_FN(residual_topk_impl));
  m.def(
      "_residual_topk(Tensor x, Tensor y, int k, bool largest, str transform, "
      "int sample_size, int feature_size) -> (Tensor, Tensor)");
  m.def(
      "_residual_topk_backward(Tensor grad, Tensor x, Tensor y, Tensor "
      "indices, str transform, int sample_size, int feature_size) -> (Tensor, "
      "Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_residual_topk", TORCH_FN(topk_kernel));
  m.impl("_residual_topk_backward", TORCH_FN(topk_backward_kernel));
}
}  // namespace
std::tuple<Tensor, Tensor> residual_topk(const Tensor& x, int64_t k,
                                         const c10::optional<Tensor>& y,
                                         int64_t dim, bool largest, bool sorted,
                                         c10::string_view transform,
                                         int64_t chunk_size,
                                         int64_t feature_chunk_size) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::residual_topk", "")
                       .typed<decltype(residual_topk_impl)>();
  return op.call(x, k, y, dim, largest, sorted, transform, chunk_size,
                 feature_chunk_size);
}
}  // namespace torchpairwise::ops
