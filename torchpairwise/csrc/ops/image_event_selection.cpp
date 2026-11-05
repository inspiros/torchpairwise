#include <ATen/ExpandUtils.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include "image_map_operations_detail.h"
#include "utils/operator_axes.h"

namespace torchpairwise::ops {
namespace {
using at::Tensor;
using namespace detail::media;
using namespace detail::streaming;
struct TopKLayout {
  std::vector<int64_t> batch, shape;
  EventMetadata metadata;
  int64_t batches = 1, ni = 1, nj = 1;
};
TopKLayout topk_layout(const Tensor& x, const Tensor& y, int64_t comparison,
                       at::IntArrayRef settings,
                       at::ArrayRef<double> parameters, int64_t dtype,
                       at::IntArrayRef program,
                       at::ArrayRef<double> corrections, int64_t k,
                       int64_t axis, int64_t size) {
  TORCH_CHECK(x.dim() >= 2 && y.dim() >= 2 && x.layout() == at::kStrided &&
                  y.layout() == at::kStrided && x.device() == y.device() &&
                  x.scalar_type() == at::kDouble &&
                  y.scalar_type() == at::kDouble && parameters.size() == 4 &&
                  size > 0,
              "Image event topk requires matching packed double collections "
              "and positive tile size");
  auto metadata =
      event_metadata(comparison, settings, dtype, program, corrections);
  TORCH_CHECK(axis >= 0 && axis < int64_t(metadata.shape.size()),
              "Image topk requires one current event axis");
  TORCH_CHECK(metadata.dtype != at::kBool,
              "topk requires numeric entries, not boolean");
  TORCH_CHECK(k >= 0 && k <= metadata.shape[axis],
              "k exceeds the image event axis length");
  TopKLayout l;
  l.batch = at::infer_size(x.sizes().slice(0, x.dim() - 2),
                           y.sizes().slice(0, y.dim() - 2));
  l.metadata = std::move(metadata);
  l.shape = l.batch;
  l.shape.insert(l.shape.end(), {x.size(-2), y.size(-2)});
  l.shape.insert(l.shape.end(), l.metadata.shape.begin(),
                 l.metadata.shape.end());
  l.shape[l.batch.size() + 2 + axis] = k;
  for (auto b : l.batch)
    l.batches *= b;
  int64_t pairs = int64_t(1) << 20;
  if (!program.empty()) {
    // Keep ordinary program outputs near a million entries per sample tile.
    // A single pair can still have more retained events than that budget.
    for (auto d : l.metadata.shape)
      pairs = std::max<int64_t>(1, pairs / d);
  }
  l.ni = std::min(size, pairs);
  l.nj = std::min(size, std::max<int64_t>(1, pairs / l.ni));
  return l;
}
Tensor sample_tile(Tensor value, int64_t i, int64_t ni, int64_t j, int64_t nj) {
  return value.narrow(0, i, ni).narrow(1, j, nj);
}
Tensor event_slice(Tensor value, int64_t axis, at::IntArrayRef start,
                   at::IntArrayRef width) {
  for (int64_t d = 0; d < int64_t(start.size()); ++d)
    if (d != axis)
      value = value.narrow(d + 2, start[d], width[d]);
  return value;
}
// Stable sorting plus chronological chunks gives lower axis IDs on ties.
void merge_region(Tensor output, Tensor indices, const Tensor& value,
                  at::IntArrayRef start, at::IntArrayRef width, int64_t axis,
                  int64_t k, bool largest) {
  auto dim = axis + 2;
  auto old = std::min(k, start[axis]);
  auto count = std::min(k, start[axis] + width[axis]);
  output = event_slice(output, axis, start, width);
  indices = event_slice(indices, axis, start, width);
  auto values = at::cat({output.narrow(dim, 0, old), value}, dim);
  std::vector<int64_t> shape(value.dim(), 1);
  shape[dim] = width[axis];
  auto ids =
      at::arange(start[axis], start[axis] + width[axis], indices.options())
          .reshape(shape)
          .expand_as(value);
  ids = at::cat({indices.narrow(dim, 0, old), ids}, dim);
  auto order = at::argsort(values, true, dim, largest).narrow(dim, 0, count);
  output.narrow(dim, 0, count).copy_(values.gather(dim, order));
  indices.narrow(dim, 0, count).copy_(ids.gather(dim, order));
}
Tensor selection_cotangent(const Tensor& gradient, const Tensor& indices,
                           int64_t axis, at::IntArrayRef start,
                           at::IntArrayRef width, at::IntArrayRef shape) {
  auto g = event_slice(gradient, axis, start, width);
  auto ids = event_slice(indices, axis, start, width);
  auto result = at::zeros(shape, gradient.options());
  int64_t elements = 1;
  for (int64_t d = 0; d < int64_t(shape.size()); ++d)
    if (d != axis + 2)
      elements *= shape[d];
  auto step =
      std::max<int64_t>(1, (int64_t(1) << 20) / std::max<int64_t>(1, elements));
  auto dim = axis + 2;
  for (int64_t rank = 0; rank < ids.size(dim); rank += step) {
    auto count = std::min(step, ids.size(dim) - rank);
    auto local = ids.narrow(dim, rank, count) - start[axis];
    auto eligible = local.ge(0) & local.lt(width[axis]);
    auto values = at::where(eligible, g.narrow(dim, rank, count), 0.);
    result.scatter_add_(dim, local.clamp(0, width[axis] - 1), values);
  }
  return result;
}
EventOutputs event_topk_kernel(const Tensor& x, const Tensor& y,
                               int64_t comparison, at::IntArrayRef settings,
                               at::ArrayRef<double> parameters, int64_t dtype,
                               at::IntArrayRef program,
                               at::ArrayRef<double> corrections, int64_t k,
                               int64_t axis, bool largest, int64_t size) {
  auto l = topk_layout(x, y, comparison, settings, parameters, dtype, program,
                       corrections, k, axis, size);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::NoGradGuard guard;
  auto output = at::empty(l.shape, x.options().dtype(l.metadata.dtype));
  auto indices = at::empty(l.shape, x.options().dtype(at::kLong));
  if (!output.numel())
    return {output, indices};
  auto rank = l.metadata.shape.size() + 2;
  for (int64_t flat = 0; flat < l.batches; ++flat) {
    auto index = coordinates(flat, l.batch);
    auto a = lookup(x, index, 2), b = lookup(y, index, 2);
    auto out = lookup(output, index, rank), ids = lookup(indices, index, rank);
    for (int64_t i = 0; i < x.size(-2); i += l.ni)
      for (int64_t j = 0; j < y.size(-2); j += l.nj) {
        auto ni = std::min(l.ni, x.size(-2) - i),
             nj = std::min(l.nj, y.size(-2) - j);
        auto left = a.narrow(0, i, ni), right = b.narrow(0, j, nj);
        auto target = sample_tile(out, i, ni, j, nj),
             target_ids = sample_tile(ids, i, ni, j, nj);
        if (program.empty()) {
          visit_map_regions(left, right, comparison, settings, parameters,
                            dtype, false,
                            [&](const Tensor& value, int64_t c, int64_t nc,
                                int64_t h, int64_t nh, int64_t w, int64_t nw) {
                              merge_region(target, target_ids, value, {c, h, w},
                                           {nc, nh, nw}, axis, k, largest);
                            });
        } else {
          auto value = std::get<0>(event_reduce(
              left, right, comparison, settings, parameters, dtype, program,
              corrections, "", {}, false, 1., size, false));
          auto order = at::argsort(value, true, axis + 2, largest)
                           .narrow(axis + 2, 0, k);
          target.copy_(value.gather(axis + 2, order));
          target_ids.copy_(order);
        }
      }
  }
  return {output, indices};
}
EventOutputs event_topk_backward_kernel(
    const Tensor& gradient, const Tensor& x, const Tensor& y,
    const Tensor& indices, int64_t comparison, at::IntArrayRef settings,
    at::ArrayRef<double> parameters, int64_t dtype, at::IntArrayRef program,
    at::ArrayRef<double> corrections, int64_t axis, int64_t size) {
  auto metadata =
      event_metadata(comparison, settings, dtype, program, corrections);
  TORCH_CHECK(axis >= 0 && axis < int64_t(metadata.shape.size()),
              "Image topk requires one current event axis");
  auto rank = int64_t(metadata.shape.size()) + 2;
  TORCH_CHECK(indices.dim() >= int64_t(rank), "Invalid image topk indices");
  auto k = indices.size(indices.dim() - rank + 2 + axis);
  auto l = topk_layout(x, y, comparison, settings, parameters, dtype, program,
                       corrections, k, axis, size);
  TORCH_CHECK(indices.sizes() == at::IntArrayRef(l.shape) &&
                  gradient.sizes() == indices.sizes() &&
                  indices.scalar_type() == at::kLong &&
                  indices.device() == x.device() &&
                  gradient.scalar_type() == l.metadata.dtype &&
                  gradient.device() == x.device(),
              "Invalid image topk gradient or indices");
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::NoGradGuard guard;
  auto gx = at::zeros(x.sizes(), x.options()),
       gy = at::zeros(y.sizes(), y.options());
  if (!l.metadata.differentiable || !at::isFloatingType(l.metadata.dtype))
    return {gx, gy};
  for (int64_t flat = 0; flat < l.batches; ++flat) {
    auto index = coordinates(flat, l.batch);
    auto a = lookup(x, index, 2), b = lookup(y, index, 2);
    auto da = lookup(gx, index, 2), db = lookup(gy, index, 2);
    auto g = lookup(gradient, index, rank), ids = lookup(indices, index, rank);
    for (int64_t i = 0; i < x.size(-2); i += l.ni)
      for (int64_t j = 0; j < y.size(-2); j += l.nj) {
        auto ni = std::min(l.ni, x.size(-2) - i),
             nj = std::min(l.nj, y.size(-2) - j);
        auto left = a.narrow(0, i, ni), right = b.narrow(0, j, nj);
        auto upstream = sample_tile(g, i, ni, j, nj),
             witnesses = sample_tile(ids, i, ni, j, nj);
        EventOutputs grads;
        if (program.empty()) {
          auto cotangent = [&](const Tensor&, int64_t c, int64_t nc, int64_t h,
                               int64_t nh, int64_t w, int64_t nw) {
            return selection_cotangent(upstream, witnesses, axis, {c, h, w},
                                       {nc, nh, nw}, {ni, nj, nc, nh, nw});
          };
          grads =
              map_backward_regions(cotangent, false, left, right, comparison,
                                   settings, parameters, dtype, false);
        } else {
          auto shape = std::vector<int64_t>{ni, nj};
          shape.insert(shape.end(), l.metadata.shape.begin(),
                       l.metadata.shape.end());
          auto cotangent = at::zeros(shape, gradient.options())
                               .scatter_(axis + 2, witnesses, upstream);
          static auto op =
              c10::Dispatcher::singleton()
                  .findSchemaOrThrow(
                      "torchpairwise::_image_event_reduce_backward", "")
                  .typed<EventGradSignature>();
          grads = op.call(cotangent, left, right, at::empty({0}, x.options()),
                          comparison, settings, parameters, dtype, program,
                          corrections, "", {}, false, 1., size, false);
        }
        da.narrow(0, i, ni).add_(std::get<0>(grads));
        db.narrow(0, j, nj).add_(std::get<1>(grads));
      }
  }
  return {gx, gy};
}
EventOutputs image_event_topk_impl(
    const Tensor& x, const c10::optional<Tensor>& y,
    c10::string_view comparison, at::IntArrayRef program,
    at::ArrayRef<double> corrections, int64_t k, int64_t dim, bool largest,
    bool sorted, c10::optional<double> data_range, int64_t window, double sigma,
    double k1, double k2, int64_t size) {
  ImageMapOptions q;
  q.data_range = data_range;
  q.window_size = window;
  q.sigma = sigma;
  q.k1 = k1;
  q.k2 = k2;
  q.chunk_size = size;
  PairwiseOperationOptions o;
  o.chunk_size = size;
  auto p = prepare_image_event_operator(x, y, comparison, program, corrections,
                                        o, q);
  std::vector<int64_t> settings;
  for (size_t n = 1; n < 5; ++n)
    settings.push_back(p.settings.integers[n].value());
  std::vector<double> parameters;
  for (size_t n = 0; n < 4; ++n)
    parameters.push_back(p.settings.reals[n].value());
  auto dtype = p.settings.integers[5].value(),
       kind = p.settings.integers[0].value();
  auto metadata = event_metadata(kind, settings, dtype, program, corrections);
  auto rank = metadata.shape.size() + 2;
  auto axes = detail::operator_axes({dim}, p.batch.size() + rank, rank);
  TORCH_CHECK(axes[0] >= 2,
              "Image topk requires one current event axis; contract events "
              "before sample topk");
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_image_event_topk", "")
          .typed<EventTopKSignature>();
  return op.call(p.values[X], p.values[Y], kind, settings, parameters, dtype,
                 program, corrections, k, axes[0] - 2, largest, size);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "image_event_topk(Tensor x, Tensor? y, str comparison, int[] program, "
      "float[] corrections, int k, int dim=-1, bool largest=True, bool "
      "sorted=True, float? data_range=None, int window_size=11, float "
      "sigma=1.5, float k1=0.01, float k2=0.03, int chunk_size=16) -> (Tensor, "
      "Tensor)",
      TORCH_FN(image_event_topk_impl));
  m.def(
      "_image_event_topk(Tensor x, Tensor y, int comparison, int[] settings, "
      "float[] parameters, int dtype, int[] program, float[] corrections, int "
      "k, int axis, bool largest, int chunk_size) -> (Tensor, Tensor)");
  m.def(
      "_image_event_topk_backward(Tensor grad, Tensor x, Tensor y, Tensor "
      "indices, int comparison, int[] settings, float[] parameters, int dtype, "
      "int[] program, float[] corrections, int axis, int chunk_size) -> "
      "(Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_image_event_topk", TORCH_FN(event_topk_kernel));
  m.impl("_image_event_topk_backward", TORCH_FN(event_topk_backward_kernel));
}
}  // namespace
std::tuple<Tensor, Tensor> image_event_topk(
    const Tensor& x, const c10::optional<Tensor>& y,
    c10::string_view comparison, at::IntArrayRef program,
    at::ArrayRef<double> corrections, int64_t k, int64_t dim, bool largest,
    bool sorted, const ImageMapOptions& q) {
  TORCH_CHECK(!q.keepdim,
              "topk has no keepdim; set it on individual event reductions");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::image_event_topk", "")
                       .typed<decltype(image_event_topk_impl)>();
  return op.call(x, y, comparison, program, corrections, k, dim, largest,
                 sorted, q.data_range, q.window_size, q.sigma, q.k1, q.k2,
                 q.chunk_size);
}
}  // namespace torchpairwise::ops
