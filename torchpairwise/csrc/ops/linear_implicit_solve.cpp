#include <torch/autograd.h>
#include <torch/library.h>

#include <cmath>

#include "linear_solve_detail.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops::detail::linear {
Packet packet_from_args(TORCHPAIRWISE_LINEAR_PACKET_DECL) {
  Packet p;
  p.inputs = inputs.vec();
  p.counts = counts.vec();
  p.metrics = metrics;
  p.families = families;
  p.tensors = tensors.copy();
  p.reals = reals;
  p.integers = integers;
  p.flags = flags;
  p.strings = strings;
  p.real_lists = real_lists;
  p.integer_lists = integer_lists;
  p.transposed = transposed;
  p.source_counts = source_counts.vec();
  p.programs = programs;
  p.factors = factors.vec();
  p.factor_counts = factor_counts.vec();
  p.kinds = kinds;
  p.integer_numbers = integer_numbers;
  p.real_numbers = real_numbers;
  p.linear_program = linear_program.vec();
  p.linear_factors = linear_factors.vec();
  return p;
}
Tensors packet_values(const Packet& p) {
  Tensors v = p.inputs;
  v.insert(v.end(), p.factors.begin(), p.factors.end());
  v.insert(v.end(), p.linear_factors.begin(), p.linear_factors.end());
  for (const auto& item : p.tensors)
    v.push_back(item.value());
  return v;
}
Packet replace_values(const Packet& metadata, at::TensorList values) {
  Packet p = metadata;
  int64_t offset = 0;
  auto take = [&](Tensors& out) {
    auto size = out.size();
    out = values.slice(offset, size).vec();
    offset += size;
  };
  take(p.inputs);
  take(p.factors);
  take(p.linear_factors);
  p.tensors = TensorDict();
  for (const auto& item : metadata.tensors)
    p.tensors.insert(item.key(), values[offset++]);
  TORCH_CHECK(offset == int64_t(values.size()),
              "Implicit solve tensor packet size mismatch");
  return p;
}
SolveTuple implicit_solve(const Packet& p, c10::SymIntArrayRef shape,
                          const Tensor& rhs, c10::string_view method,
                          double rtol, double atol, int64_t max_iter,
                          int64_t restart, const c10::optional<Tensor>& initial,
                          bool preconditioned) {
  TORCH_CHECK(!preconditioned || method == "cg",
              "Implicit preconditioning requires method=cg");
  TORCH_CHECK(method == "cg" || method == "minres" || method == "gmres",
              "Implicit solve method must be cg, minres or gmres");
  TORCH_CHECK(!initial || !initial->requires_grad(),
              "Implicit solve initial guess is an algorithmic aid; detach it "
              "explicitly");
  at::NoGradGuard no_grad;
  using Signature = SolveTuple(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                               c10::SymIntArrayRef, const Tensor&, double,
                               double, int64_t, const c10::optional<Tensor>&);
  using GMRESSignature = SolveTuple(
      TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef, const Tensor&,
      double, double, int64_t, int64_t, const c10::optional<Tensor>&);
  static auto cg = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::linear_cg", "")
                       .typed<Signature>();
  static auto minres =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::linear_minres", "")
          .typed<Signature>();
  static auto gmres = c10::Dispatcher::singleton()
                          .findSchemaOrThrow("torchpairwise::linear_gmres", "")
                          .typed<GMRESSignature>();
  static auto pcg = c10::Dispatcher::singleton()
                        .findSchemaOrThrow("torchpairwise::linear_pcg", "")
                        .typed<Signature>();
  auto fit = preconditioned
                 ? pcg.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), shape, rhs,
                            rtol, atol, max_iter, initial)
             : method == "gmres"
                 ? gmres.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), shape, rhs,
                              rtol, atol, restart, max_iter, initial)
                 : (method == "cg" ? cg : minres)
                       .call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), shape, rhs,
                             rtol, atol, max_iter, initial);
  TORCH_CHECK(
      std::get<3>(fit).all().item<bool>(),
      "Implicit solve requires a converged forward/adjoint solve; inspect the "
      "inference solver diagnostics or increase max_iter/tighten conditioning");
  return fit;
}
namespace {
SolveTuple linear_implicit_solve_impl(
    TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef shape,
    const Tensor& rhs, c10::string_view method, double rtol, double atol,
    int64_t max_iter, int64_t restart, const c10::optional<Tensor>& initial) {
  return implicit_solve(packet_from_args(TORCHPAIRWISE_LINEAR_PACKET_ARGS),
                        shape, rhs, method, rtol, atol, max_iter, restart,
                        initial);
}
Tensors product_vjp(const Packet& p, const Tensor& rhs, int64_t root,
                    const Tensors& blocks, const Tensors& cotangents) {
  auto values = packet_values(p);
  Tensors leaves, active;
  std::vector<int64_t> indices;
  for (const auto& value : values) {
    auto leaf = value.detach();
    if (value.is_floating_point()) {
      leaf.set_requires_grad(true);
      indices.push_back(leaves.size());
      active.push_back(leaf);
    }
    leaves.push_back(leaf);
  }
  Tensors result(values.size() + 1);
  if (!active.empty()) {
    at::AutoGradMode enable_grad(true);
    auto replay = replace_values(p, leaves);
    Plan plan(unpack_graph(TORCHPAIRWISE_LINEAR_PACKED_ARGS(replay)), rhs);
    Tensors products;
    for (const auto& block : blocks)
      products.push_back(plan.apply(root, block));
    if (products[0].requires_grad()) {
      auto derivatives = torch::autograd::grad(products, active, cotangents,
                                               false, false, true);
      for (size_t i = 0; i < indices.size(); ++i)
        if (derivatives[i].defined())
          result[indices[i]] = derivatives[i];
    }
  }
  // Materialize only unused active gradients, after computing the real VJPs.
  // This avoids retaining a redundant zero matrix for caller-owned dense
  // inputs.
  for (size_t index = 0; index < values.size(); ++index)
    if (!result[index].defined())
      result[index] = at::zeros_like(values[index]);
  return result;
}
Tensors square_backward(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                        c10::SymIntArrayRef shape, const Tensor& rhs,
                        const Tensor& solution, const Tensor& gradient,
                        c10::string_view method, double rtol, double atol,
                        int64_t max_iter, int64_t restart,
                        bool preconditioned) {
  TORCH_CHECK(!at::GradMode::is_enabled(),
              "Higher derivatives of implicit solves are unsupported");
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto p = packet_from_args(TORCHPAIRWISE_LINEAR_PACKET_ARGS);
  // Reverse the map, never form A^T. Adjoint uses the same tolerances and
  // method.
  auto adjoint = p;
  auto root = int64_t(p.linear_program.size() / 3) - 1;
  if (preconditioned) {
    TORCH_CHECK(p.linear_program[3 * root] == 3, "Invalid implicit PCG graph");
    root = p.linear_program[3 * root + 1];
  }
  auto transposed_root = int64_t(adjoint.linear_program.size() / 3);
  adjoint.linear_program.insert(adjoint.linear_program.end(), {7, root, -1});
  if (preconditioned)
    adjoint.linear_program.insert(
        adjoint.linear_program.end(),
        {3, transposed_root, p.linear_program.back()});
  bool vector = rhs.dim() == 1;
  auto lambda = std::get<0>(implicit_solve(
      adjoint, shape, vector ? gradient.unsqueeze(-1) : gradient, method, rtol,
      atol, max_iter, restart, c10::nullopt, preconditioned));
  auto block = vector ? solution.detach().unsqueeze(-1) : solution.detach();
  auto result = product_vjp(p, rhs, root, {block}, {-lambda});
  result.back() =
      (vector ? lambda.squeeze(-1) : lambda).sum_to_size(rhs.sizes());
  return result;
}
Tensors linear_implicit_solve_backward_impl(
    TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef shape,
    const Tensor& rhs, const Tensor& solution, const Tensor& gradient,
    c10::string_view method, double rtol, double atol, int64_t max_iter,
    int64_t restart) {
  return square_backward(TORCHPAIRWISE_LINEAR_PACKET_ARGS, shape, rhs, solution,
                         gradient, method, rtol, atol, max_iter, restart,
                         false);
}
Tensors linear_implicit_pcg_backward_impl(
    TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef shape,
    const Tensor& rhs, const Tensor& solution, const Tensor& gradient,
    c10::string_view method, double rtol, double atol, int64_t max_iter,
    int64_t restart) {
  return square_backward(TORCHPAIRWISE_LINEAR_PACKET_ARGS, shape, rhs, solution,
                         gradient, method, rtol, atol, max_iter, restart, true);
}
SolveTuple linear_implicit_pcg_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                                    c10::SymIntArrayRef shape,
                                    const Tensor& rhs, c10::string_view method,
                                    double rtol, double atol, int64_t max_iter,
                                    int64_t restart,
                                    const c10::optional<Tensor>& initial) {
  return implicit_solve(packet_from_args(TORCHPAIRWISE_LINEAR_PACKET_ARGS),
                        shape, rhs, method, rtol, atol, max_iter, restart,
                        initial, true);
}
Tensor damping_square(const Tensor& rhs, double damp) {
  TORCH_CHECK(std::isfinite(damp) && damp > 0.,
              "Differentiable lstsq requires finite positive damp");
  auto square = at::scalar_tensor(damp * damp, rhs.options());
  utils::check_values((square.isfinite() & square.gt(0)).all(),
                      "Differentiable lstsq requires damp squared finite and "
                      "positive in the source dtype");
  return square;
}
LeastSquaresTuple linear_implicit_lstsq_impl(
    TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef shape,
    const Tensor& rhs, double damp, double rtol, double atol, int64_t max_iter,
    const c10::optional<Tensor>& initial) {
  damping_square(rhs, damp);
  TORCH_CHECK(
      !initial || !initial->requires_grad(),
      "Implicit lstsq initial is an algorithmic aid; detach it explicitly");
  at::NoGradGuard no_grad;
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::linear_lsqr", "")
                       .typed<ImplicitLstsqSignature>();
  auto fit = op.call(TORCHPAIRWISE_LINEAR_PACKET_ARGS, shape, rhs, damp, rtol,
                     atol, max_iter, initial);
  TORCH_CHECK(std::get<4>(fit).all().item<bool>(),
              "Implicit lstsq requires a converged forward solve; inspect LSQR "
              "diagnostics or improve stopping options/conditioning");
  return fit;
}
Tensors linear_implicit_lstsq_backward_impl(
    TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef shape,
    const Tensor& rhs, const Tensor& solution, const Tensor& gradient,
    double damp, double rtol, double atol, int64_t max_iter) {
  TORCH_CHECK(!at::GradMode::is_enabled(),
              "Higher derivatives of implicit lstsq are unsupported");
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto p = packet_from_args(TORCHPAIRWISE_LINEAR_PACKET_ARGS), normal = p;
  auto root = int64_t(p.linear_program.size() / 3) - 1, transpose = root + 1;
  auto factor = int64_t(normal.linear_factors.size());
  normal.linear_factors.push_back(damping_square(rhs, damp));
  normal.linear_program.insert(
      normal.linear_program.end(),
      {7, root, -1, 3, transpose, root, 5, transpose + 1, factor});
  auto normal_shape = shape.vec();
  normal_shape[normal_shape.size() - 2] = normal_shape.back();
  bool vector = rhs.dim() == 1;
  auto x = vector ? solution.detach().unsqueeze(-1) : solution.detach();
  auto g = vector ? gradient.unsqueeze(-1) : gradient;
  auto lambda = std::get<0>(implicit_solve(normal, normal_shape, g, "cg", rtol,
                                           atol, max_iter, 20, c10::nullopt));
  Tensor residual, projected;
  {
    at::NoGradGuard no_grad;
    Plan plan(unpack_graph(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p)), rhs);
    projected = plan(lambda);
    residual = (vector ? rhs.unsqueeze(-1) : rhs) - plan(x);
  }
  auto result = product_vjp(p, rhs, root, {lambda, x}, {residual, -projected});
  result.back() =
      (vector ? projected.squeeze(-1) : projected).sum_to_size(rhs.sizes());
  return result;
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "linear_implicit_solve(Tensor[] inputs, int[] counts, str[] metrics, "
      "str[] families, Dict(str,Tensor) tensors, Dict(str,float) reals, "
      "Dict(str,int) integers, Dict(str,bool) flags, Dict(str,str) strings, "
      "Dict(str,float[]) real_lists, Dict(str,int[]) integer_lists, bool[] "
      "transposed, int[] source_counts, Dict(str,int[]) programs, Tensor[] "
      "factors, int[] factor_counts, Dict(str,int[]) kinds, Dict(str,int[]) "
      "integer_numbers, Dict(str,float[]) real_numbers, int[] linear_program, "
      "Tensor[] linear_factors, SymInt[] shape, Tensor rhs, str method, float "
      "rtol, float atol, int max_iter, int restart, Tensor? initial=None) -> "
      "(Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "linear_implicit_solve_backward(Tensor[] inputs, int[] counts, str[] "
      "metrics, str[] families, Dict(str,Tensor) tensors, Dict(str,float) "
      "reals, Dict(str,int) integers, Dict(str,bool) flags, Dict(str,str) "
      "strings, Dict(str,float[]) real_lists, Dict(str,int[]) integer_lists, "
      "bool[] transposed, int[] source_counts, Dict(str,int[]) programs, "
      "Tensor[] factors, int[] factor_counts, Dict(str,int[]) kinds, "
      "Dict(str,int[]) integer_numbers, Dict(str,float[]) real_numbers, int[] "
      "linear_program, Tensor[] linear_factors, SymInt[] shape, Tensor rhs, "
      "Tensor solution, Tensor gradient, str method, float rtol, float atol, "
      "int max_iter, int restart) -> Tensor[]");
  m.def(
      "linear_implicit_pcg(Tensor[] inputs, int[] counts, str[] metrics, "
      "str[] families, Dict(str,Tensor) tensors, Dict(str,float) reals, "
      "Dict(str,int) integers, Dict(str,bool) flags, Dict(str,str) strings, "
      "Dict(str,float[]) real_lists, Dict(str,int[]) integer_lists, bool[] "
      "transposed, int[] source_counts, Dict(str,int[]) programs, Tensor[] "
      "factors, int[] factor_counts, Dict(str,int[]) kinds, Dict(str,int[]) "
      "integer_numbers, Dict(str,float[]) real_numbers, int[] linear_program, "
      "Tensor[] linear_factors, SymInt[] shape, Tensor rhs, str method, float "
      "rtol, float atol, int max_iter, int restart, Tensor? initial=None) -> "
      "(Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "linear_implicit_pcg_backward(Tensor[] inputs, int[] counts, str[] "
      "metrics, str[] families, Dict(str,Tensor) tensors, Dict(str,float) "
      "reals, Dict(str,int) integers, Dict(str,bool) flags, Dict(str,str) "
      "strings, Dict(str,float[]) real_lists, Dict(str,int[]) integer_lists, "
      "bool[] transposed, int[] source_counts, Dict(str,int[]) programs, "
      "Tensor[] factors, int[] factor_counts, Dict(str,int[]) kinds, "
      "Dict(str,int[]) integer_numbers, Dict(str,float[]) real_numbers, int[] "
      "linear_program, Tensor[] linear_factors, SymInt[] shape, Tensor rhs, "
      "Tensor solution, Tensor gradient, str method, float rtol, float atol, "
      "int max_iter, int restart) -> Tensor[]");
  m.def(
      "linear_implicit_lstsq(Tensor[] inputs, int[] counts, str[] metrics, "
      "str[] families, Dict(str,Tensor) tensors, Dict(str,float) reals, "
      "Dict(str,int) integers, Dict(str,bool) flags, Dict(str,str) strings, "
      "Dict(str,float[]) real_lists, Dict(str,int[]) integer_lists, bool[] "
      "transposed, int[] source_counts, Dict(str,int[]) programs, Tensor[] "
      "factors, int[] factor_counts, Dict(str,int[]) kinds, Dict(str,int[]) "
      "integer_numbers, Dict(str,float[]) real_numbers, int[] linear_program, "
      "Tensor[] linear_factors, SymInt[] shape, Tensor rhs, float damp, float "
      "rtol, float atol, int max_iter, Tensor? initial=None) -> (Tensor, "
      "Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "linear_implicit_lstsq_backward(Tensor[] inputs, int[] counts, str[] "
      "metrics, str[] families, Dict(str,Tensor) tensors, Dict(str,float) "
      "reals, Dict(str,int) integers, Dict(str,bool) flags, Dict(str,str) "
      "strings, Dict(str,float[]) real_lists, Dict(str,int[]) integer_lists, "
      "bool[] transposed, int[] source_counts, Dict(str,int[]) programs, "
      "Tensor[] factors, int[] factor_counts, Dict(str,int[]) kinds, "
      "Dict(str,int[]) integer_numbers, Dict(str,float[]) real_numbers, int[] "
      "linear_program, Tensor[] linear_factors, SymInt[] shape, Tensor rhs, "
      "Tensor solution, Tensor gradient, float damp, float rtol, float atol, "
      "int max_iter) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("linear_implicit_solve", TORCH_FN(linear_implicit_solve_impl));
  m.impl("linear_implicit_pcg", TORCH_FN(linear_implicit_pcg_impl));
  m.impl("linear_implicit_pcg_backward",
         TORCH_FN(linear_implicit_pcg_backward_impl));
  m.impl("linear_implicit_lstsq", TORCH_FN(linear_implicit_lstsq_impl));
  m.impl("linear_implicit_lstsq_backward",
         TORCH_FN(linear_implicit_lstsq_backward_impl));
  m.impl("linear_implicit_solve_backward",
         TORCH_FN(linear_implicit_solve_backward_impl));
}
}  // namespace
}  // namespace torchpairwise::ops::detail::linear
