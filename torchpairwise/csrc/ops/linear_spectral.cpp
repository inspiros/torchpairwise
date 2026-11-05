#include "linear_spectral.h"

#include <ATen/Context.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

#include "linear_plan_detail.h"
#include "linear_spectral_detail.h"
#include "utils/eigh.h"

namespace torchpairwise::ops {
namespace {
using namespace detail::linear;
using Estimate = std::tuple<Tensor, Tensor, Tensor, Tensor>;
using Action = std::tuple<Tensor, Tensor, Tensor>;
void require_finite(const Tensor& value, const char* message) {
  TORCH_CHECK(value.isfinite().all().item<bool>(), message);
}
void require_inference(TORCHPAIRWISE_LINEAR_PACKET_DECL, const Tensor& anchor,
                       const c10::optional<Tensor>& extra) {
  TORCH_CHECK(anchor.layout() == at::kStrided &&
                  (anchor.scalar_type() == at::kFloat ||
                   anchor.scalar_type() == at::kDouble) &&
                  (anchor.device().is_cpu() || anchor.device().is_cuda()),
              "Spectral consumers require strided CPU/CUDA float32/float64");
  if (anchor.is_cuda())
    TORCH_CHECK(!c10::impl::VirtualGuardImpl(anchor.device().type())
                     .getStream(anchor.device())
                     .is_capturing(),
                "Spectral consumers do not support CUDA graph capture");
  if (!at::GradMode::is_enabled())
    return;
  auto check = [](const Tensor& t) {
    TORCH_CHECK(!t.requires_grad(),
                "Spectral consumers are inference-only; use torch.no_grad() "
                "or detach inputs explicitly");
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
  if (extra)
    check(*extra);
}
std::vector<int64_t> checked_shape(const Plan& plan,
                                   c10::SymIntArrayRef expected) {
  TORCH_CHECK(c10::SymIntArrayRef(plan.shapes.back()).equals(expected),
              "Bound spectral source shape changed; reconstruct the operator");
  std::vector<int64_t> shape;
  for (const auto& dim : plan.shapes.back())
    shape.push_back(dim.expect_int());
  TORCH_CHECK(shape.back() == shape[shape.size() - 2],
              "Spectral consumers require a square operator");
  return shape;
}
Tensor basis_block(int64_t n, int64_t start, int64_t width,
                   const at::TensorOptions& options) {
  auto block = at::zeros({n, width}, options);
  block.narrow(0, start, width).diagonal().fill_(1.);
  return block;
}
Tensor exact_trace(const Plan& plan, const Tensor& anchor,
                   at::IntArrayRef shape, int64_t block_size) {
  auto node = plan.graph.nodes.back();
  if (node[0] == 0) {
    auto p = plan.sources[node[1]];
    p.settings.operation = "diagonal";
    return std::get<0>(_streaming_reduce(p.values, p.settings))
        .to(at::kDouble)
        .sum(-1);
  }
  if (node[0] == 11)
    return plan.graph.factors[node[1]].to(at::kDouble).sum(-1);
  auto result = at::zeros(shape.slice(0, shape.size() - 2),
                          anchor.options().dtype(at::kDouble));
  auto n = shape.back();
  for (int64_t start = 0; start < n; start += block_size) {
    auto count = std::min(block_size, n - start);
    auto block = plan(basis_block(n, start, count, anchor.options()));
    result.add_(block.narrow(-2, start, count)
                    .diagonal(0, -2, -1)
                    .to(at::kDouble)
                    .sum(-1));
  }
  return result;
}
Tensor exact_logdet(const Plan& plan, const Tensor& anchor,
                    at::IntArrayRef shape, int64_t block_size) {
  auto n = shape.back();
  if (!n)
    return at::zeros(shape.slice(0, shape.size() - 2),
                     anchor.options().dtype(at::kDouble));
  auto matrix = at::empty(shape, anchor.options().dtype(at::kDouble));
  for (int64_t start = 0; start < n; start += block_size) {
    auto count = std::min(block_size, n - start);
    matrix.narrow(-1, start, count)
        .copy_(plan(basis_block(n, start, count, anchor.options())));
  }
  require_finite(matrix, "logdet requires finite operator entries");
  auto eps = anchor.scalar_type() == at::kDouble
                 ? std::numeric_limits<double>::epsilon()
                 : std::numeric_limits<float>::epsilon();
  auto scale = std::get<0>(std::get<0>(matrix.abs().max(-1)).max(-1));
  TORCH_CHECK((matrix - matrix.transpose(-1, -2))
                  .abs()
                  .le((scale * (32 * eps)).unsqueeze(-1).unsqueeze(-1))
                  .all()
                  .item<bool>(),
              "Exact logdet requires a symmetric positive definite operator");
  auto [factor, info] = at::linalg_cholesky_ex(matrix, false, false);
  TORCH_CHECK(
      info.eq(0).all().item<bool>(),
      "logdet requires a positive definite operator; no jitter is added");
  return factor.diagonal(0, -2, -1).log().sum(-1) * 2.;
}
// Independent Lanczos recurrences per RHS/probe. Full two-pass
// reorthogonalization bounds loss of orthogonality at the cost of O(N*P*k)
// stored basis entries. P is bounded by the caller's block_size.
struct Krylov {
  std::vector<Tensor> basis;
  Tensor alpha, beta, norm, iterations, terminated;
};
Krylov lanczos(const Plan& plan, const Tensor& values, int64_t max_iter) {
  auto norm = at::linalg_vector_norm(values.to(at::kDouble), 2., {-2}, false);
  require_finite(norm, "Spectral probes/RHS must be finite");
  auto active = norm.gt(0);
  auto iterations = at::zeros_like(norm, norm.options().dtype(at::kLong));
  auto q = values.to(at::kDouble) / at::where(active, norm, 1.).unsqueeze(-2);
  auto previous = at::zeros_like(q), previous_beta = at::zeros_like(norm);
  std::vector<Tensor> basis, alpha, beta;
  auto eps = values.scalar_type() == at::kDouble
                 ? std::numeric_limits<double>::epsilon()
                 : std::numeric_limits<float>::epsilon();
  auto limit = std::min(max_iter, values.size(-2));
  for (int64_t step = 0; step < limit && active.any().item<bool>(); ++step) {
    basis.push_back(q);
    auto aq = plan(q.to(values.scalar_type())).to(at::kDouble);
    require_finite(aq, "Nonfinite operator product in Lanczos");
    auto diagonal = (q * aq).sum(-2);
    auto w = aq - q * diagonal.unsqueeze(-2) -
             previous * previous_beta.unsqueeze(-2);
    for (int pass = 0; pass < 2; ++pass)
      for (const auto& v : basis)
        w = w - v * (v * w).sum(-2).unsqueeze(-2);
    auto next_beta = at::linalg_vector_norm(w, 2., {-2}, false);
    auto scale = at::linalg_vector_norm(aq, 2., {-2}, false) + diagonal.abs() +
                 previous_beta;
    auto continuing = active & next_beta.gt(scale * (16 * eps));
    iterations = iterations + active.to(at::kLong);
    alpha.push_back(at::where(active, diagonal, 1.));
    beta.push_back(at::where(continuing, next_beta, 0.));
    previous = q;
    q = at::where(continuing.unsqueeze(-2),
                  w / at::where(continuing, next_beta, 1.).unsqueeze(-2), 0.);
    previous_beta = beta.back();
    active = continuing;
  }
  auto empty_shape = norm.sizes().vec();
  empty_shape.push_back(0);
  return {std::move(basis),
          alpha.empty() ? at::empty(empty_shape, norm.options())
                        : at::stack(alpha, -1),
          beta.empty() ? at::empty(empty_shape, norm.options())
                       : at::stack(beta, -1),
          norm,
          iterations,
          ~active};
}
std::pair<Tensor, Tensor> projected_function(const Krylov& k,
                                             c10::string_view function,
                                             double scale, int64_t width) {
  auto alpha = k.alpha.narrow(-1, 0, width);
  auto matrix = at::diag_embed(alpha);
  if (width > 1) {
    auto beta = k.beta.narrow(-1, 0, width - 1);
    matrix = matrix + at::diag_embed(beta, 1) + at::diag_embed(beta, -1);
  }
  auto [eigenvalues, vectors] = utils::eigh(matrix, "L");
  if (function != "exp")
    TORCH_CHECK(eigenvalues.gt(0).all().item<bool>(),
                "Observed a nonpositive Ritz value; logdet/inverse square root "
                "require an SPD operator");
  auto transformed = function == "log"            ? eigenvalues.log()
                     : function == "inverse_sqrt" ? eigenvalues.rsqrt()
                                                  : (eigenvalues * scale).exp();
  require_finite(transformed, "Nonfinite projected matrix function");
  auto first = vectors.select(-2, 0);
  auto quadratic = (first.square() * transformed).sum(-1) * k.norm.square();
  auto coefficients =
      vectors.matmul((transformed * first).unsqueeze(-1)).squeeze(-1);
  return {quadratic, coefficients};
}
Tensor combine(const Krylov& k, const Tensor& coefficients) {
  auto result = at::zeros_like(k.basis.front());
  for (int64_t i = 0; i < coefficients.size(-1); ++i)
    result.add_(k.basis[i] *
                (coefficients.select(-1, i) * k.norm).unsqueeze(-2));
  return result;
}
std::tuple<Tensor, Tensor, Tensor> summarize(const Tensor& samples) {
  auto mean = samples.mean(-1);
  auto error =
      samples.size(-1) > 1
          ? samples.std(-1, 1, false) / std::sqrt(double(samples.size(-1)))
          : at::full_like(mean, std::numeric_limits<double>::quiet_NaN());
  return {mean, error, samples};
}
Estimate linear_trace_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                           c10::SymIntArrayRef shape, const Tensor& anchor,
                           c10::string_view function, c10::string_view method,
                           c10::optional<int64_t> num_probes,
                           const c10::optional<Tensor>& probes,
                           c10::optional<int64_t> seed,
                           c10::optional<int64_t> max_iter,
                           int64_t block_size) {
  TORCH_CHECK(anchor.dim() == 0,
              "Spectral estimate requires a scalar metadata anchor");
  require_inference(TORCHPAIRWISE_LINEAR_PACKET_ARGS, anchor, probes);
  TORCH_CHECK(block_size > 0, "block_size must be a positive integer");
  TORCH_CHECK(function == "trace" || function == "logdet",
              "Unknown spectral estimate");
  TORCH_CHECK(method == "exact" || (function == "trace" ? method == "hutchinson"
                                                        : method == "slq"),
              "Select exact/hutchinson for trace or exact/slq for SPD logdet");
  TORCH_CHECK(
      !max_iter || (function == "logdet" && method == "slq" && *max_iter > 0),
      "max_iter is positive and only used by SLQ logdet");
  at::NoGradGuard no_grad;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  Plan plan(unpack_graph(TORCHPAIRWISE_LINEAR_PACKET_ARGS), anchor);
  auto dimensions = checked_shape(plan, shape);
  auto batch = c10::IntArrayRef(dimensions).slice(0, dimensions.size() - 2);
  auto sample_shape = batch.vec();
  sample_shape.push_back(0);
  if (method == "exact") {
    TORCH_CHECK(!num_probes && !probes && !seed,
                "Exact methods do not accept probes, num_probes or seed");
    auto value = function == "trace"
                     ? exact_trace(plan, anchor, dimensions, block_size)
                     : exact_logdet(plan, anchor, dimensions, block_size);
    require_finite(value, "Nonfinite exact spectral estimate");
    return {value, at::zeros_like(value),
            at::empty(sample_shape, value.options()),
            at::empty(sample_shape, value.options().dtype(at::kLong))};
  }
  int64_t count;
  if (probes) {
    TORCH_CHECK(!num_probes && !seed,
                "Supplied probes cannot be combined with num_probes or seed");
    TORCH_CHECK(
        probes->layout() == at::kStrided && probes->dim() >= 2 &&
            probes->scalar_type() == anchor.scalar_type() &&
            probes->device() == anchor.device(),
        "Probes must be strided [...,N,P] matching operator dtype/device");
    TORCH_CHECK(probes->size(-2) == dimensions.back() && probes->size(-1) > 0,
                "Probes require shape [...,N,P] with positive P");
    TORCH_CHECK(
        at::infer_size(probes->sizes().slice(0, probes->dim() - 2), batch) ==
            batch,
        "Probes may broadcast to operator batches but cannot add batches");
    require_finite(*probes, "Spectral probes must be finite");
    count = probes->size(-1);
  } else {
    TORCH_CHECK(
        num_probes && *num_probes > 0,
        "Stochastic methods require positive num_probes or explicit probes");
    TORCH_CHECK(!seed || *seed >= 0, "seed must be a nonnegative integer");
    count = *num_probes;
  }
  c10::optional<at::Generator> generator;
  if (seed) {
    generator = at::globalContext().defaultGenerator(anchor.device()).clone();
    generator->set_current_seed(uint64_t(*seed));
  }
  sample_shape.back() = count;
  auto samples = at::zeros(sample_shape, anchor.options().dtype(at::kDouble));
  auto iterations = at::zeros(sample_shape, anchor.options().dtype(at::kLong));
  for (int64_t start = 0; start < count; start += block_size) {
    auto width = std::min(block_size, count - start);
    auto probe_shape = batch.vec();
    probe_shape.insert(probe_shape.end(), {dimensions.back(), width});
    auto z = probes ? probes->narrow(-1, start, width).expand(probe_shape)
                    : at::randint(0, 2, probe_shape, generator,
                                  anchor.options().dtype(at::kLong))
                          .to(anchor.scalar_type())
                          .mul_(2)
                          .sub_(1);
    Tensor terms;
    if (function == "trace") {
      auto az = plan(z).to(at::kDouble);
      require_finite(az, "Nonfinite operator product in trace estimation");
      terms = (z.to(at::kDouble) * az).sum(-2);
    } else {
      auto krylov = lanczos(plan, z, max_iter.value_or(32));
      terms = krylov.basis.empty()
                  ? at::zeros_like(krylov.norm)
                  : projected_function(krylov, "log", 1., krylov.basis.size())
                        .first;
      iterations.narrow(-1, start, width).copy_(krylov.iterations);
    }
    samples.narrow(-1, start, width).copy_(terms);
  }
  auto [estimate, error, values] = summarize(samples);
  require_finite(estimate, "Nonfinite stochastic spectral estimate");
  return {estimate, error, values, iterations};
}
Action linear_function_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                            c10::SymIntArrayRef shape, const Tensor& values,
                            c10::string_view function, double scale,
                            int64_t max_iter, int64_t block_size) {
  require_inference(TORCHPAIRWISE_LINEAR_PACKET_ARGS, values, c10::nullopt);
  TORCH_CHECK(values.dim() >= 1,
              "Matrix function RHS requires [N] or [...,N,R]");
  TORCH_CHECK(function == "exp" || function == "inverse_sqrt",
              "Unknown matrix function action");
  TORCH_CHECK(
      std::isfinite(scale) && max_iter > 0 && block_size > 0,
      "Matrix function scale must be finite; max_iter/block_size positive");
  at::NoGradGuard no_grad;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  Plan plan(unpack_graph(TORCHPAIRWISE_LINEAR_PACKET_ARGS), values);
  auto dimensions = checked_shape(plan, shape);
  auto rhs = values.dim() == 1 ? values.unsqueeze(-1) : values;
  TORCH_CHECK(rhs.size(-2) == dimensions.back() && rhs.size(-1) > 0,
              "Matrix function RHS must have shape [N] or [...,N,R], R>0");
  auto batch = at::infer_size(
      c10::IntArrayRef(dimensions).slice(0, dimensions.size() - 2),
      rhs.sizes().slice(0, rhs.dim() - 2));
  auto target = batch;
  target.insert(target.end(), {dimensions.back(), rhs.size(-1)});
  rhs = rhs.expand(target);
  require_finite(rhs, "Matrix function RHS must be finite");
  auto output = at::zeros(target, values.options());
  target.erase(target.end() - 2);
  auto error = at::zeros(target, values.options().dtype(at::kDouble));
  auto iterations = at::zeros(target, values.options().dtype(at::kLong));
  for (int64_t start = 0; start < rhs.size(-1); start += block_size) {
    auto width = std::min(block_size, rhs.size(-1) - start);
    auto block = rhs.narrow(-1, start, width);
    auto krylov = lanczos(plan, block, max_iter);
    if (krylov.basis.empty())
      continue;
    auto depth = int64_t(krylov.basis.size());
    auto coefficients =
        projected_function(krylov, function, scale, depth).second;
    auto action = combine(krylov, coefficients);
    auto difference =
        at::full_like(krylov.norm, std::numeric_limits<double>::infinity());
    if (depth > 1) {
      auto previous = combine(
          krylov,
          projected_function(krylov, function, scale, depth - 1).second);
      difference = at::linalg_vector_norm(action - previous, 2., {-2}, false);
    }
    difference = at::where(krylov.terminated, 0., difference);
    output.narrow(-1, start, width).copy_(action);
    error.narrow(-1, start, width).copy_(difference);
    iterations.narrow(-1, start, width).copy_(krylov.iterations);
  }
  require_finite(output,
                 "Matrix function action is not finite in the output dtype");
  if (values.dim() == 1)
    return {output.squeeze(-1), error.squeeze(-1), iterations.squeeze(-1)};
  return {output, error, iterations};
}
#define TORCHPAIRWISE_SPECTRAL_SCHEMA_PACKET                                  \
  "Tensor[] inputs, int[] counts, str[] metrics, str[] families, "            \
  "Dict(str,Tensor) tensors, Dict(str,float) reals, Dict(str,int) integers, " \
  "Dict(str,bool) flags, Dict(str,str) strings, Dict(str,float[]) "           \
  "real_lists, "                                                              \
  "Dict(str,int[]) integer_lists, bool[] transposed, int[] source_counts, "   \
  "Dict(str,int[]) programs, Tensor[] factors, int[] factor_counts, "         \
  "Dict(str,int[]) kinds, Dict(str,int[]) integer_numbers, "                  \
  "Dict(str,float[]) real_numbers, int[] linear_program, Tensor[] "           \
  "linear_factors, "                                                          \
  "SymInt[] shape, "
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def("linear_trace(" TORCHPAIRWISE_SPECTRAL_SCHEMA_PACKET
        "Tensor anchor, str function, str method, int? num_probes=None, "
        "Tensor? probes=None, int? seed=None, int? max_iter=None, int "
        "block_size=16) "
        "-> (Tensor, Tensor, Tensor, Tensor)",
        {at::Tag::nondeterministic_seeded});
  m.def("linear_function(" TORCHPAIRWISE_SPECTRAL_SCHEMA_PACKET
        "Tensor values, str function, float scale=1., int max_iter=32, "
        "int block_size=16) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("linear_trace", TORCH_FN(linear_trace_impl));
  m.impl("linear_function", TORCH_FN(linear_function_impl));
}
}  // namespace
namespace detail::spectral {
Estimate estimate(TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef shape,
                  const Tensor& anchor, c10::string_view function,
                  c10::string_view method, c10::optional<int64_t> num_probes,
                  const c10::optional<Tensor>& probes,
                  c10::optional<int64_t> seed, c10::optional<int64_t> max_iter,
                  int64_t block_size) {
  return linear_trace_impl(TORCHPAIRWISE_LINEAR_PACKET_ARGS, shape, anchor,
                           function, method, num_probes, probes, seed, max_iter,
                           block_size);
}
}  // namespace detail::spectral
TraceResult trace(const LinearOperator& op, const TraceOptions& options) {
  auto p = pack_graph(op.graph());
  auto anchor =
      at::empty({}, at::TensorOptions().dtype(op.dtype()).device(op.device()));
  if (options.differentiable) {
    static auto differentiable =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::linear_trace_differentiable", "")
            .typed<detail::spectral::ForwardSignature>();
    auto result = differentiable.call(
        TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(), anchor, "trace",
        options.method, options.num_probes, options.probes, options.seed,
        c10::nullopt, options.block_size, "exact", 1e-5, 0., 1000);
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
  static auto handle = c10::Dispatcher::singleton()
                           .findSchemaOrThrow("torchpairwise::linear_trace", "")
                           .typed<decltype(linear_trace_impl)>();
  auto result =
      handle.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(), anchor,
                  "trace", options.method, options.num_probes, options.probes,
                  options.seed, c10::nullopt, options.block_size);
  return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
}
LogdetResult logdet(const LinearOperator& op, const LogdetOptions& options) {
  bool slq_gradient = options.differentiable && options.method == "slq";
  TORCH_CHECK(!options.gradient || slq_gradient,
              "gradient is only used by differentiable SLQ logdet");
  TORCH_CHECK(
      slq_gradient || (!options.solve_rtol && !options.solve_atol &&
                       !options.solve_max_iter),
      "Backward solve options are only used by differentiable SLQ logdet");
  TORCH_CHECK(!slq_gradient || options.gradient.value_or("") == "solve",
              "Differentiable SLQ requires explicit gradient='solve'");
  auto p = pack_graph(op.graph());
  auto anchor =
      at::empty({}, at::TensorOptions().dtype(op.dtype()).device(op.device()));
  if (options.differentiable) {
    static auto differentiable =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::linear_trace_differentiable", "")
            .typed<detail::spectral::ForwardSignature>();
    auto result = differentiable.call(
        TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(), anchor, "logdet",
        options.method, options.num_probes, options.probes, options.seed,
        options.max_iter, options.block_size, slq_gradient ? "solve" : "exact",
        options.solve_rtol.value_or(1e-5), options.solve_atol.value_or(0.),
        options.solve_max_iter.value_or(1000));
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
            std::get<3>(result)};
  }
  static auto handle = c10::Dispatcher::singleton()
                           .findSchemaOrThrow("torchpairwise::linear_trace", "")
                           .typed<decltype(linear_trace_impl)>();
  auto result =
      handle.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(), anchor,
                  "logdet", options.method, options.num_probes, options.probes,
                  options.seed, options.max_iter, options.block_size);
  return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
          std::get<3>(result)};
}
MatrixFunctionResult function_action(const LinearOperator& op,
                                     const Tensor& values,
                                     c10::string_view function, double scale,
                                     const MatrixFunctionOptions& options) {
  auto p = pack_graph(op.graph());
  static auto handle =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::linear_function", "")
          .typed<decltype(linear_function_impl)>();
  auto result =
      handle.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(), values,
                  function, scale, options.max_iter, options.block_size);
  return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
}
MatrixFunctionResult expm_multiply(const LinearOperator& op,
                                   const Tensor& values, double scale,
                                   const MatrixFunctionOptions& options) {
  return function_action(op, values, "exp", scale, options);
}
MatrixFunctionResult inv_sqrt_multiply(const LinearOperator& op,
                                       const Tensor& values,
                                       const MatrixFunctionOptions& options) {
  return function_action(op, values, "inverse_sqrt", 1., options);
}
}  // namespace torchpairwise::ops
