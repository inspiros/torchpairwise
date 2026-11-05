#include "linear_lowrank.h"

#include <ATen/ExpandUtils.h>
#include <ATen/TensorIndexing.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "linear_plan_detail.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops {
namespace {
using namespace detail::linear;
using Four = std::tuple<Tensor, Tensor, Tensor, Tensor>;
using Five = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>;

void check_input(const Tensor& value) {
  if (at::GradMode::is_enabled())
    TORCH_CHECK(!value.requires_grad(),
                "Low-rank and greedy setup is inference-only; use "
                "torch.no_grad() or detach inputs explicitly");
}
void check_packet(TORCHPAIRWISE_LINEAR_PACKET_DECL, const Tensor& anchor) {
  TORCH_CHECK(anchor.layout() == at::kStrided && anchor.dim() == 0 &&
                  (anchor.scalar_type() == at::kFloat ||
                   anchor.scalar_type() == at::kDouble),
              "Low-rank and greedy setup requires a scalar float32/float64 "
              "metadata anchor");
  if (anchor.is_cuda())
    TORCH_CHECK(!c10::impl::VirtualGuardImpl(anchor.device().type())
                     .getStream(anchor.device())
                     .is_capturing(),
                "Low-rank and greedy setup uses native stopping and does not "
                "support CUDA graph capture");
  for (const auto& t : inputs)
    check_input(t);
  for (const auto& t : factors)
    check_input(t);
  for (const auto& t : linear_factors)
    check_input(t);
  for (const auto& item : tensors)
    check_input(item.value());
  check_input(anchor);
}
LinearOperatorGraph double_graph(LinearOperatorGraph graph) {
  auto convert = [](Tensor& t) {
    if (t.is_floating_point())
      t = t.to(at::kDouble);
  };
  for (auto& source : graph.sources) {
    for (auto& leaf : source.sources) {
      for (auto& t : leaf.inputs)
        convert(t);
      // Dict handles are shared by value; copy before changing their entries.
      auto values = leaf.tensors.copy();
      for (const auto& item : leaf.tensors) {
        auto t = item.value();
        convert(t);
        values.insert_or_assign(item.key(), t);
      }
      leaf.tensors = std::move(values);
    }
    for (auto& t : source.factors)
      convert(t);
  }
  for (auto& t : graph.factors)
    convert(t);
  return graph;
}
std::vector<int64_t> matrix_shape(const Plan& plan,
                                  c10::SymIntArrayRef expected, int64_t k) {
  TORCH_CHECK(c10::SymIntArrayRef(plan.shapes.back()).equals(expected),
              "Bound low-rank source shape changed; reconstruct the operator");
  std::vector<int64_t> shape;
  for (const auto& d : plan.shapes.back())
    shape.push_back(d.expect_int());
  TORCH_CHECK(shape.size() >= 2 && shape.back() == shape[shape.size() - 2],
              "Low-rank and greedy setup requires a square operator");
  TORCH_CHECK(k >= 1 && k <= shape.back(), "Choose 1 <= rank/k <= N");
  return shape;
}
const PreparedComparison& scalar_source(const Plan& plan) {
  TORCH_CHECK(plan.graph.nodes.size() == 1 && plan.graph.nodes.back()[0] == 0,
              "This algorithm requires a scalar entrywise operator; pass a "
              "kernel/distance source or as_lazy(matrix), not a linear-only "
              "composition");
  return plan.sources[plan.graph.nodes.back()[1]];
}
// Evaluate paired diagonal entries or one row/column block, with source
// preparation retained across steps. Batched pivots never become a union of
// columns: each batch evaluates exactly its own N-by-1 block in bounded tiles.
Tensor entries(const PreparedComparison& p, const std::vector<int64_t>& batch,
               const Tensor& rows, const Tensor& columns, bool paired) {
  const auto& s = p.settings;
  if (s.family == c10::optional<std::string>("_matrix")) {
    auto matrix = lookup(p.values[X], batch, 2);
    auto r = s.transposed() ? columns : rows;
    auto c = s.transposed() ? rows : columns;
    auto result = matrix.index(
        {paired ? r : r.unsqueeze(-1), paired ? c : c.unsqueeze(0)});
    return !paired && s.transposed() ? result.transpose(-1, -2) : result;
  }
  auto shape = std::vector<int64_t>{rows.numel()};
  if (!paired)
    shape.push_back(columns.numel());
  auto result = at::empty(shape, p.values[X].options().dtype(at::kDouble));
  auto size = s.size;
  for (int64_t i = 0; i < rows.numel(); i += size) {
    auto ni = std::min(size, rows.numel() - i);
    auto r = rows.narrow(0, i, ni);
    for (int64_t j = 0; j < (paired ? 1 : columns.numel()); j += size) {
      auto nj = paired ? ni : std::min(size, columns.numel() - j);
      auto c = columns.narrow(0, paired ? i : j, nj);
      auto local = local_inputs(p.values, s, batch, 0, 0, 0, 0, r, c);
      auto value = evaluate(local, s, paired, 0, 0, r, c);
      if (paired)
        result.narrow(0, i, ni).copy_(value);
      else
        result.narrow(0, i, ni).narrow(1, j, nj).copy_(value);
    }
  }
  return result;
}
Four cholesky(const Plan& plan, const Tensor& anchor,
              const std::vector<int64_t>& shape, int64_t width, double rtol,
              double atol) {
  const auto& source = scalar_source(plan);
  auto n = shape.back();
  auto batch = std::vector<int64_t>(shape.begin(), shape.end() - 2);
  auto fshape = batch, pshape = batch;
  fshape.insert(fshape.end(), {n, width});
  pshape.push_back(width);
  auto options = anchor.options().dtype(at::kDouble);
  auto factor = at::zeros(fshape, options);
  auto pivots = at::full(pshape, -1, options.dtype(at::kLong));
  auto error = at::zeros(batch, options);
  auto ranks = at::zeros(batch, options.dtype(at::kLong));
  auto all = at::arange(n, options.dtype(at::kLong));
  int64_t batches = error.numel();
  for (int64_t z = 0; z < batches; ++z) {
    auto index = coordinates(z, batch);
    auto f = lookup(factor, index, 2), p = lookup(pivots, index, 1);
    auto residual = entries(source, index, all, all, true);
    utils::check_values((residual.isfinite() & residual.ge(0)).all(),
                        "Pivoted Cholesky requires a finite nonnegative PSD "
                        "diagonal");
    auto trace = residual.sum().item<double>();
    TORCH_CHECK(std::isfinite(trace),
                "Pivoted Cholesky requires a finite initial trace; rescale "
                "the operator");
    auto threshold = std::max(atol, rtol * trace);
    int64_t effective = 0;
    for (; effective < width; ++effective) {
      if (residual.sum().item<double>() <= threshold)
        break;
      auto pivot = residual.argmax(-1, true);
      auto diagonal = residual.gather(0, pivot);
      if (diagonal.item<double>() <= 0.)
        break;
      auto column = entries(source, index, all, pivot, false).squeeze(-1);
      if (effective) {
        auto previous = f.narrow(-1, 0, effective);
        column =
            column -
            previous
                .matmul(
                    previous.index_select(0, pivot).squeeze(0).unsqueeze(-1))
                .squeeze(-1);
      }
      column = column / diagonal.sqrt();
      utils::check_values(column.isfinite().all(),
                          "Pivoted Cholesky encountered nonfinite entries");
      f.select(-1, effective).copy_(column);
      p.select(-1, effective).copy_(pivot.squeeze(0));
      residual = (residual - column.square()).clamp_min(0.);
      // Exact exclusion avoids reselecting a pivot through roundoff.
      residual.index_fill_(0, p.narrow(0, 0, effective + 1), 0.);
    }
    lookup(error, index, 0).copy_(residual.sum());
    lookup(ranks, index, 0).fill_(effective);
  }
  return {factor.to(anchor.scalar_type()), pivots,
          error.to(anchor.scalar_type()), ranks};
}
Tensor vector_length(const Tensor& value) {
  return value.square().sum(-1).sqrt();
}
Five recurrence(const Plan& plan, const Tensor& anchor,
                const std::vector<int64_t>& shape, int64_t k,
                const Tensor& initial, bool reorthogonalize) {
  TORCH_CHECK(initial.layout() == at::kStrided && initial.dim() >= 1 &&
                  initial.scalar_type() == anchor.scalar_type() &&
                  initial.device() == anchor.device(),
              "Lanczos initial must be strided [...,N] matching operator "
              "dtype/device");
  auto batch = std::vector<int64_t>(shape.begin(), shape.end() - 2);
  auto vshape = batch;
  vshape.push_back(shape.back());
  TORCH_CHECK(
      initial.size(-1) == shape.back() &&
          at::infer_size(initial.sizes(), vshape) == vshape,
      "Lanczos initial must broadcast to [...,N] without adding batches");
  auto qshape = vshape, ashape = batch, bshape = batch;
  qshape.push_back(k);
  ashape.push_back(k);
  bshape.push_back(k - 1);
  auto options = anchor.options().dtype(at::kDouble);
  auto basis = at::zeros(qshape, options), alpha = at::zeros(ashape, options),
       beta = at::zeros(bshape, options);
  auto iterations = at::zeros(batch, options.dtype(at::kLong));
  auto status = at::zeros_like(iterations);
  auto seed = initial.to(at::kDouble).expand(vshape);
  // Scale before normalizing, avoiding overflow/underflow for finite seeds.
  auto scale = seed.abs().amax(-1);
  auto valid = seed.isfinite().all(-1) & scale.gt(0.);
  auto scaled = at::where(valid.unsqueeze(-1), seed, 0.) /
                at::where(valid, scale, 1.).unsqueeze(-1);
  auto q = scaled / vector_length(scaled).clamp_min(1e-300).unsqueeze(-1);
  status = at::where(valid, status, 2);
  auto active = valid;
  auto previous = at::zeros_like(q), previous_beta = at::zeros(batch, options);
  const double eps = 64 * std::numeric_limits<double>::epsilon() * shape.back();
  for (int64_t j = 0; j < k && active.any().item<bool>(); ++j) {
    auto aq = plan(q.unsqueeze(-1)).squeeze(-1);
    auto w = aq - previous * previous_beta.unsqueeze(-1);
    auto a = (q * w).sum(-1);
    w = w - q * a.unsqueeze(-1);
    auto candidate = at::where(active.unsqueeze(-1), q, 0.);
    basis.select(-1, j).copy_(candidate);
    if (reorthogonalize) {
      auto history = basis.narrow(-1, 0, j + 1);
      // Two passes retain orthogonality near invariant-subspace breakdown.
      for (int pass = 0; pass < 2; ++pass)
        w = w -
            history.matmul(history.transpose(-1, -2).matmul(w.unsqueeze(-1)))
                .squeeze(-1);
    }
    auto b = vector_length(w);
    auto finite = aq.isfinite().all(-1) & a.isfinite() & b.isfinite();
    auto accepted = active & finite;
    basis.select(-1, j).copy_(at::where(accepted.unsqueeze(-1), q, 0.));
    alpha.select(-1, j).copy_(at::where(accepted, a, 0.));
    iterations = at::where(accepted, j + 1, iterations);
    auto breakdown =
        accepted & b.le(eps * (vector_length(aq) + a.abs() + previous_beta));
    status = at::where(active & ~finite, 3, status);
    if (j + 1 < k)
      status = at::where(breakdown, 1, status);
    active = accepted & ~breakdown;
    if (j + 1 < k) {
      beta.select(-1, j).copy_(at::where(active, b, 0.));
      previous = q;
      previous_beta = at::where(active, b, 0.);
      q = at::where(active.unsqueeze(-1),
                    w / at::where(active, b, 1.).unsqueeze(-1), 0.);
    }
  }
  if (k > 1) {
    auto used = at::arange(k - 1, iterations.options())
                    .lt((iterations - 1).unsqueeze(-1));
    beta = at::where(used, beta, 0.);
  }
  return {basis, alpha, beta, iterations, status};
}

Four linear_pivoted_cholesky_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                                  c10::SymIntArrayRef shape,
                                  const Tensor& anchor, int64_t rank,
                                  double rtol, double atol) {
  check_packet(TORCHPAIRWISE_LINEAR_PACKET_ARGS, anchor);
  TORCH_CHECK(
      std::isfinite(rtol) && rtol >= 0 && std::isfinite(atol) && atol >= 0,
      "Pivoted Cholesky tolerances must be finite and nonnegative");
  at::NoGradGuard guard;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  Plan plan(double_graph(unpack_graph(TORCHPAIRWISE_LINEAR_PACKET_ARGS)),
            anchor.to(at::kDouble));
  return cholesky(plan, anchor, matrix_shape(plan, shape, rank), rank, rtol,
                  atol);
}
Five linear_lanczos_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                         c10::SymIntArrayRef shape, const Tensor& anchor,
                         int64_t k, const Tensor& initial,
                         bool reorthogonalize) {
  check_packet(TORCHPAIRWISE_LINEAR_PACKET_ARGS, anchor);
  check_input(initial);
  at::NoGradGuard guard;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  Plan plan(double_graph(unpack_graph(TORCHPAIRWISE_LINEAR_PACKET_ARGS)),
            anchor.to(at::kDouble));
  auto [q, a, b, iterations, status] = recurrence(
      plan, anchor, matrix_shape(plan, shape, k), k, initial, reorthogonalize);
  return {q.to(anchor.scalar_type()), a.to(anchor.scalar_type()),
          b.to(anchor.scalar_type()), iterations, status};
}
Tensor linear_love_cache_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                              c10::SymIntArrayRef shape, const Tensor& anchor,
                              int64_t rank, const Tensor& initial) {
  check_packet(TORCHPAIRWISE_LINEAR_PACKET_ARGS, anchor);
  check_input(initial);
  at::NoGradGuard guard;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  Plan plan(double_graph(unpack_graph(TORCHPAIRWISE_LINEAR_PACKET_ARGS)),
            anchor.to(at::kDouble));
  auto [q, a, b, iterations, status] = recurrence(
      plan, anchor, matrix_shape(plan, shape, rank), rank, initial, true);
  utils::check_values(status.le(1).all(),
                      "LOVE requires finite Lanczos actions and a finite "
                      "nonzero initial vector; inspect lanczos diagnostics");
  auto used =
      at::arange(rank, iterations.options()).lt(iterations.unsqueeze(-1));
  // Padding is identity so variable-sized batch tridiagonals can share a
  // Cholesky call; padded Q columns and hence returned root columns are zero.
  auto t = at::diag_embed(at::where(used, a, 1.));
  if (rank > 1) {
    t.diagonal(1, -2, -1).copy_(b);
    t.diagonal(-1, -2, -1).copy_(b);
  }
  auto [l, info] = at::linalg_cholesky_ex(t);
  utils::check_values(info.eq(0).all(),
                      "LOVE requires positive definite A; add positive noise "
                      "or inspect the Lanczos projection");
  auto root = at::linalg_solve_triangular(l, q.transpose(-1, -2), false)
                  .transpose(-1, -2);
  utils::check_values(root.isfinite().all(), "LOVE root must be finite");
  return root.to(anchor.scalar_type()).clone(at::MemoryFormat::Contiguous);
}

std::tuple<Tensor, Tensor> selection(
    const Plan& plan, const Tensor& anchor, const std::vector<int64_t>& shape,
    int64_t k, int64_t start, bool random,
    const c10::optional<at::Generator>& generator) {
  const auto& source = scalar_source(plan);
  auto n = shape.back();
  TORCH_CHECK(random || (start >= 0 && start < n),
              "FPS start must satisfy 0 <= start < N");
  auto batch = std::vector<int64_t>(shape.begin(), shape.end() - 2);
  auto outshape = batch;
  outshape.push_back(k);
  auto options = anchor.options().dtype(at::kDouble);
  auto indices = at::empty(outshape, options.dtype(at::kLong));
  auto distances = at::zeros(outshape, options);
  auto all = at::arange(n, indices.options());
  auto batches = indices.numel() / k;
  for (int64_t z = 0; z < batches; ++z) {
    auto index = coordinates(z, batch);
    auto ids = lookup(indices, index, 1), ds = lookup(distances, index, 1);
    auto selected = at::zeros({n}, options.dtype(at::kBool));
    auto minimum =
        at::full({n}, std::numeric_limits<double>::infinity(), options);
    auto pivot = random ? at::randint(n, {1}, generator, indices.options())
                        : at::full({1}, start, indices.options());
    if (!random)
      ds.select(0, 0).fill_(std::numeric_limits<double>::infinity());
    for (int64_t j = 0; j < k; ++j) {
      ids.select(0, j).copy_(pivot.squeeze(0));
      if (j)
        ds.select(0, j).copy_(minimum.gather(0, pivot).squeeze(0));
      selected.index_fill_(0, pivot, true);
      auto row = entries(source, index, pivot, all, false).squeeze(0);
      utils::check_values((row.isfinite() & row.ge(0)).all(),
                          "Greedy selection requires finite nonnegative "
                          "self-distances; use squared distances for kmeans++");
      minimum = at::minimum(minimum, row);
      minimum.masked_fill_(selected, 0.);
      if (j + 1 == k)
        break;
      if (random) {
        // Rescale before multinomial to keep large finite squared distances
        // from overflowing their total. Degenerate mass uses the first unused.
        auto maximum = minimum.max();
        pivot = maximum.item<double>() > 0.
                    ? at::multinomial(minimum / maximum, 1, false, generator)
                    : (~selected).to(at::kLong).argmax(-1, true);
      } else {
        pivot = minimum.masked_fill(selected, -1.).argmax(-1, true);
      }
    }
  }
  return {indices, distances.to(anchor.scalar_type())};
}
std::tuple<Tensor, Tensor> linear_fps_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                                           c10::SymIntArrayRef shape,
                                           const Tensor& anchor, int64_t k,
                                           int64_t start) {
  check_packet(TORCHPAIRWISE_LINEAR_PACKET_ARGS, anchor);
  at::NoGradGuard guard;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  Plan plan(double_graph(unpack_graph(TORCHPAIRWISE_LINEAR_PACKET_ARGS)),
            anchor.to(at::kDouble));
  return selection(plan, anchor, matrix_shape(plan, shape, k), k, start, false,
                   c10::nullopt);
}
Tensor linear_kmeans_plusplus_impl(
    TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef shape,
    const Tensor& anchor, int64_t k,
    const c10::optional<at::Generator>& generator) {
  check_packet(TORCHPAIRWISE_LINEAR_PACKET_ARGS, anchor);
  if (generator)
    TORCH_CHECK(generator->device().type() == anchor.device().type() &&
                    (!generator->device().has_index() ||
                     generator->device().index() == anchor.device().index()),
                "kmeans++ generator must match the operator device");
  at::NoGradGuard guard;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  Plan plan(double_graph(unpack_graph(TORCHPAIRWISE_LINEAR_PACKET_ARGS)),
            anchor.to(at::kDouble));
  return std::get<0>(selection(plan, anchor, matrix_shape(plan, shape, k), k, 0,
                               true, generator));
}

#define LOWRANK_SCHEMA_PACKET                                                 \
  "(Tensor[] inputs, int[] counts, str[] metrics, str[] families, "           \
  "Dict(str,Tensor) tensors, Dict(str,float) reals, Dict(str,int) integers, " \
  "Dict(str,bool) flags, Dict(str,str) strings, Dict(str,float[]) "           \
  "real_lists, "                                                              \
  "Dict(str,int[]) integer_lists, bool[] transposed, int[] source_counts, "   \
  "Dict(str,int[]) programs, Tensor[] factors, int[] factor_counts, "         \
  "Dict(str,int[]) kinds, Dict(str,int[]) integer_numbers, "                  \
  "Dict(str,float[]) real_numbers, int[] linear_program, "                    \
  "Tensor[] linear_factors, SymInt[] shape, Tensor anchor, "
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def("linear_pivoted_cholesky" LOWRANK_SCHEMA_PACKET
        "int rank, float rtol=0., float atol=0.) -> (Tensor, Tensor, Tensor, "
        "Tensor)");
  m.def("linear_lanczos" LOWRANK_SCHEMA_PACKET
        "int k, Tensor initial, bool reorthogonalize=True) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("linear_love_cache" LOWRANK_SCHEMA_PACKET
        "int rank, Tensor initial) -> Tensor");
  m.def("linear_farthest_point_sampling" LOWRANK_SCHEMA_PACKET
        "int k, int start=0) -> (Tensor, Tensor)");
  m.def("linear_kmeans_plusplus" LOWRANK_SCHEMA_PACKET
        "int k, Generator? generator=None) -> Tensor",
        {at::Tag::nondeterministic_seeded});
}
#undef LOWRANK_SCHEMA_PACKET
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("linear_pivoted_cholesky", TORCH_FN(linear_pivoted_cholesky_impl));
  m.impl("linear_lanczos", TORCH_FN(linear_lanczos_impl));
  m.impl("linear_love_cache", TORCH_FN(linear_love_cache_impl));
  m.impl("linear_farthest_point_sampling", TORCH_FN(linear_fps_impl));
  m.impl("linear_kmeans_plusplus", TORCH_FN(linear_kmeans_plusplus_impl));
}
// All numerical paths reject gradient-bearing setup inputs before NoGradGuard.
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("linear_pivoted_cholesky", torch::CppFunction::makeFallthrough());
  m.impl("linear_lanczos", torch::CppFunction::makeFallthrough());
  m.impl("linear_love_cache", torch::CppFunction::makeFallthrough());
  m.impl("linear_farthest_point_sampling",
         torch::CppFunction::makeFallthrough());
  m.impl("linear_kmeans_plusplus", torch::CppFunction::makeFallthrough());
}
Tensor metadata_anchor(const LinearOperator& op) {
  return at::empty({},
                   at::TensorOptions().dtype(op.dtype()).device(op.device()));
}
}  // namespace

PivotedCholeskyResult pivoted_cholesky(const LinearOperator& op, int64_t rank,
                                       double rtol, double atol) {
  auto p = pack_graph(op.graph());
  static auto handle =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::linear_pivoted_cholesky", "")
          .typed<decltype(linear_pivoted_cholesky_impl)>();
  auto [f, pivots, error, ranks] =
      handle.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(),
                  metadata_anchor(op), rank, rtol, atol);
  return {f, pivots, error, ranks};
}
LanczosResult lanczos(const LinearOperator& op, int64_t k,
                      const Tensor& initial, bool reorthogonalize) {
  auto p = pack_graph(op.graph());
  static auto handle =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::linear_lanczos", "")
          .typed<decltype(linear_lanczos_impl)>();
  auto [q, a, b, iterations, status] =
      handle.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(),
                  metadata_anchor(op), k, initial, reorthogonalize);
  return {q, a, b, iterations, status};
}
LOVECache love_cache(const LinearOperator& op, int64_t rank,
                     const Tensor& initial) {
  auto p = pack_graph(op.graph());
  static auto handle =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::linear_love_cache", "")
          .typed<decltype(linear_love_cache_impl)>();
  return {handle.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(),
                      metadata_anchor(op), rank, initial)};
}
Tensor predictive_variance(const LOVECache& cache, const LinearOperator& cross,
                           const Tensor& prior_variance) {
  TORCH_CHECK(cache.root.dim() >= 2 && prior_variance.dim() >= 1 &&
                  cache.root.scalar_type() == cross.dtype() &&
                  cache.root.device() == cross.device() &&
                  prior_variance.scalar_type() == cross.dtype() &&
                  prior_variance.device() == cross.device() &&
                  cache.root.sym_size(-2) == cross.shape().back() &&
                  prior_variance.sym_size(-1) ==
                      cross.shape()[cross.shape().size() - 2],
              "Predictive variance requires matching root [...,N,r], cross "
              "[...,Ntest,N] and prior [...,Ntest] dtype/device");
  return (prior_variance - cross.matmul(cache.root.detach()).square().sum(-1))
      .clamp_min(0.);
}
std::tuple<Tensor, Tensor> farthest_point_sampling(const LinearOperator& op,
                                                   int64_t k, int64_t start) {
  auto p = pack_graph(op.graph());
  static auto handle =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::linear_farthest_point_sampling",
                             "")
          .typed<decltype(linear_fps_impl)>();
  return handle.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(),
                     metadata_anchor(op), k, start);
}
Tensor kmeans_plusplus(const LinearOperator& op, int64_t k,
                       const c10::optional<at::Generator>& generator) {
  auto p = pack_graph(op.graph());
  static auto handle =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::linear_kmeans_plusplus", "")
          .typed<decltype(linear_kmeans_plusplus_impl)>();
  return handle.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(),
                     metadata_anchor(op), k, generator);
}
}  // namespace torchpairwise::ops
