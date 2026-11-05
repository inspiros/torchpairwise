#include <ATen/Context.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <cmath>

#include "linear_spectral_detail.h"

namespace torchpairwise::ops::detail::spectral {
namespace {
void validate(c10::string_view function, c10::string_view method,
              c10::string_view gradient, double rtol, double atol,
              int64_t solve_max_iter) {
  TORCH_CHECK(function == "trace" || function == "logdet",
              "Unknown spectral gradient");
  TORCH_CHECK(
      (function == "trace" && gradient == "exact") ||
          (function == "logdet" && method == "exact" && gradient == "exact") ||
          (function == "logdet" && method == "slq" && gradient == "solve"),
      "Trace/exact logdet use gradient='exact'; SLQ requires explicit "
      "gradient='solve', a trace/solve estimator, not the derivative "
      "of finite Lanczos quadrature");
  TORCH_CHECK(std::isfinite(rtol) && rtol >= 0 && std::isfinite(atol) &&
                  atol >= 0 && solve_max_iter >= 0,
              "Invalid logdet backward solve tolerances/budget");
}
Forward differentiable_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                            c10::SymIntArrayRef shape, const Tensor& anchor,
                            c10::string_view function, c10::string_view method,
                            c10::optional<int64_t> num_probes,
                            const c10::optional<Tensor>& probes,
                            c10::optional<int64_t> seed,
                            c10::optional<int64_t> max_iter, int64_t block_size,
                            c10::string_view gradient, double solve_rtol,
                            double solve_atol, int64_t solve_max_iter) {
  validate(function, method, gradient, solve_rtol, solve_atol, solve_max_iter);
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(
      anchor.dim() == 0 && anchor.layout() == at::kStrided &&
          (anchor.scalar_type() == at::kFloat ||
           anchor.scalar_type() == at::kDouble),
      "Spectral estimates require a scalar float32/float64 metadata anchor");
  TORCH_CHECK(function == "trace"
                  ? (method == "exact" || method == "hutchinson")
                  : (method == "exact" || method == "slq"),
              "Unknown spectral estimate method");
  TORCH_CHECK(
      !max_iter || (function == "logdet" && method == "slq" && *max_iter > 0),
      "max_iter is positive and only used by SLQ logdet");
  if (anchor.is_cuda())
    TORCH_CHECK(!c10::impl::VirtualGuardImpl(anchor.device().type())
                     .getStream(anchor.device())
                     .is_capturing(),
                "Spectral consumers do not support CUDA graph capture");
  // Snapshot fixed probes, not Krylov history. Draw blocks in inference order.
  at::NoGradGuard no_grad;
  Tensor saved;
  if (method == "exact") {
    auto result =
        estimate(TORCHPAIRWISE_LINEAR_PACKET_ARGS, shape, anchor, function,
                 method, num_probes, probes, seed, max_iter, block_size);
    saved = at::empty({0}, anchor.options());
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
            std::get<3>(result), saved};
  }
  TORCH_CHECK(!probes || (!num_probes && !seed),
              "Supplied probes cannot be combined with num_probes or seed");
  TORCH_CHECK(
      probes || (num_probes && *num_probes > 0),
      "Stochastic methods require positive num_probes or explicit probes");
  TORCH_CHECK(!seed || *seed >= 0, "seed must be nonnegative");
  if (probes) {
    saved = probes->detach().clone(at::MemoryFormat::Contiguous);
  } else {
    TORCH_CHECK(shape.size() >= 2 && shape.back() == shape[shape.size() - 2],
                "Spectral consumers require square maps");
    std::vector<int64_t> sizes;
    for (size_t i = 0; i + 2 < shape.size(); ++i)
      sizes.push_back(shape[i].expect_int());
    sizes.insert(sizes.end(), {shape.back().expect_int(), *num_probes});
    saved = at::empty(sizes, anchor.options());
    c10::optional<at::Generator> generator;
    if (seed) {
      generator = at::globalContext().defaultGenerator(anchor.device()).clone();
      generator->set_current_seed(uint64_t(*seed));
    }
    for (int64_t start = 0; start < *num_probes; start += block_size) {
      auto width = std::min(block_size, *num_probes - start);
      sizes.back() = width;
      saved.narrow(-1, start, width)
          .copy_(at::randint(0, 2, sizes, generator,
                             anchor.options().dtype(at::kLong))
                     .to(anchor.scalar_type())
                     .mul_(2)
                     .sub_(1));
    }
  }
  auto result =
      estimate(TORCHPAIRWISE_LINEAR_PACKET_ARGS, shape, anchor, function,
               method, c10::nullopt, saved, c10::nullopt, max_iter, block_size);
  return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
          std::get<3>(result), saved};
}
Tensor coordinate_block(int64_t n, int64_t start, int64_t count,
                        const at::TensorOptions& options) {
  auto result = at::zeros({n, count}, options);
  result.narrow(0, start, count).diagonal().fill_(1.);
  return result;
}
// Retain only prepared source state, not product/iteration history. Each VJP
// destroys its block graph after accumulation; source statistics shared by Plan
// remain alive for subsequent blocks through retain_graph=true.
class ProductVJP {
 public:
  Tensors leaves, active, buffers;
  std::vector<int64_t> indices;
  std::unique_ptr<Plan> plan;
  ProductVJP(const Packet& p, const Tensor& anchor) {
    auto values = packet_values(p);
    for (const auto& value : values) {
      auto leaf = value.detach();
      if (value.is_floating_point()) {
        leaf.set_requires_grad(true);
        indices.push_back(leaves.size());
        active.push_back(leaf);
      }
      leaves.push_back(leaf);
      // The backward schema/fake kernel promise contiguous output gradients,
      // independently of the bound operands' strides.
      buffers.push_back(at::zeros(value.sizes(), value.options()));
    }
    auto replay = replace_values(p, leaves);
    plan = std::make_unique<Plan>(
        unpack_graph(TORCHPAIRWISE_LINEAR_PACKED_ARGS(replay)), anchor);
  }
  void add(const Tensor& block, const Tensor& cotangent) {
    if (active.empty())
      return;
    auto value = (*plan)(block);
    if (!value.requires_grad())
      return;
    auto derivatives =
        torch::autograd::grad({value}, active, {cotangent}, true, false, true);
    at::NoGradGuard no_grad;
    for (size_t i = 0; i < indices.size(); ++i)
      if (derivatives[i].defined())
        buffers[indices[i]].add_(derivatives[i]);
  }
};
Tensors backward_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                      c10::SymIntArrayRef shape, const Tensor& anchor,
                      const Tensor& probes, const Tensor& upstream,
                      c10::string_view function, c10::string_view method,
                      c10::string_view gradient, int64_t block_size,
                      double solve_rtol, double solve_atol,
                      int64_t solve_max_iter) {
  TORCH_CHECK(!at::GradMode::is_enabled(),
              "Higher derivatives of spectral estimates are unsupported");
  validate(function, method, gradient, solve_rtol, solve_atol, solve_max_iter);
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto p = packet_from_args(TORCHPAIRWISE_LINEAR_PACKET_ARGS);
  auto n = shape.back().expect_int();
  Tensor cholesky;
  if (function == "logdet" && method == "exact" && n > 0) {
    at::NoGradGuard no_grad;
    Plan plan(unpack_graph(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p)), anchor);
    auto matrix = at::empty_symint(shape, anchor.options().dtype(at::kDouble));
    for (int64_t start = 0; start < n; start += block_size) {
      auto count = std::min(block_size, n - start);
      matrix.narrow(-1, start, count)
          .copy_(plan(coordinate_block(n, start, count, anchor.options())));
    }
    auto [factor, info] = at::linalg_cholesky_ex(matrix, false, false);
    TORCH_CHECK(info.eq(0).all().item<bool>(),
                "Exact logdet backward requires SPD operator");
    cholesky = factor;
  }
  at::AutoGradMode enable(true);
  ProductVJP replay(p, anchor);
  if (method == "exact") {
    for (int64_t start = 0; start < n; start += block_size) {
      auto count = std::min(block_size, n - start);
      auto block = coordinate_block(n, start, count, anchor.options());
      Tensor cotangent = block;
      if (function == "logdet") {
        at::NoGradGuard no_grad;
        cotangent = at::cholesky_solve(block.to(at::kDouble), cholesky, false)
                        .to(anchor.scalar_type());
      }
      replay.add(
          block,
          cotangent *
              upstream.to(anchor.scalar_type()).unsqueeze(-1).unsqueeze(-1));
    }
  } else {
    auto count = probes.size(-1);
    for (int64_t start = 0; start < count; start += block_size) {
      auto width = std::min(block_size, count - start);
      auto block = probes.narrow(-1, start, width).detach();
      Tensor cotangent = block;
      if (function == "logdet") {
        at::NoGradGuard no_grad;
        cotangent = std::get<0>(
            implicit_solve(p, shape, block, "cg", solve_rtol, solve_atol,
                           solve_max_iter, 20, c10::nullopt));
      }
      replay.add(block,
                 cotangent * (upstream.to(anchor.scalar_type()) / double(count))
                                 .unsqueeze(-1)
                                 .unsqueeze(-1));
    }
  }
  return replay.buffers;
}
#define TP_SPECTRAL_PACKET                                                    \
  "Tensor[] inputs, int[] counts, str[] metrics, str[] families, "            \
  "Dict(str,Tensor) tensors, Dict(str,float) reals, Dict(str,int) integers, " \
  "Dict(str,bool) flags, Dict(str,str) strings, Dict(str,float[]) "           \
  "real_lists, "                                                              \
  "Dict(str,int[]) integer_lists, bool[] transposed, int[] source_counts, "   \
  "Dict(str,int[]) programs, Tensor[] factors, int[] factor_counts, "         \
  "Dict(str,int[]) kinds, Dict(str,int[]) integer_numbers, "                  \
  "Dict(str,float[]) real_numbers, int[] linear_program, Tensor[] "           \
  "linear_factors, SymInt[] shape, "
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def("linear_trace_differentiable(" TP_SPECTRAL_PACKET
        "Tensor anchor, str function, str method, int? num_probes, Tensor? "
        "probes, "
        "int? seed, int? max_iter, int block_size, str gradient, float "
        "solve_rtol, "
        "float solve_atol, int solve_max_iter) -> (Tensor, Tensor, Tensor, "
        "Tensor, Tensor)",
        {at::Tag::nondeterministic_seeded});
  m.def("linear_trace_backward(" TP_SPECTRAL_PACKET
        "Tensor anchor, Tensor probes, Tensor upstream, str function, str "
        "method, "
        "str gradient, int block_size, float solve_rtol, float solve_atol, int "
        "solve_max_iter) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("linear_trace_differentiable", TORCH_FN(differentiable_impl));
  m.impl("linear_trace_backward", TORCH_FN(backward_impl));
}
}  // namespace
}  // namespace torchpairwise::ops::detail::spectral
