#pragma once
#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>

#include "utils/value_checks.h"

namespace torchpairwise::ops::modality_detail {
using at::Tensor;
using Features = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Values = torch::autograd::variable_list;
inline int64_t mode_for(c10::string_view mode) {
  if (mode == "all_pairs" || mode == "allpairs")
    return 0;
  if (mode == "paired")
    return 1;
  if (mode == "condensed")
    return 2;
  TORCH_CHECK(false, "mode must be all_pairs, paired or condensed");
}
inline void positive(double value, const char* name) {
  TORCH_CHECK(std::isfinite(value) && value > 0, name,
              " must be positive and finite");
}
inline void validate(const Tensor& x, const Tensor& y, int64_t rank,
                     int64_t mode, int64_t size) {
  TORCH_CHECK(x.layout() == at::kStrided && y.layout() == at::kStrided,
              "Inputs must be dense strided tensors");
  TORCH_CHECK(x.dim() >= rank && y.dim() >= rank, "Insufficient input rank");
  TORCH_CHECK(x.device() == y.device() && x.scalar_type() == y.scalar_type(),
              "Inputs must have the same dtype and device");
  TORCH_CHECK(
      x.device().is_cpu() || x.device().is_cuda() || x.device().is_meta(),
      "Inputs require CPU or CUDA");
  for (int64_t i = 1; i < rank; ++i)
    TORCH_CHECK(x.sym_size(-i) > 0 && x.sym_size(-i) == y.sym_size(-i),
                "Sample dimensions must match and be non-empty");
  TORCH_CHECK(mode != 1 || x.sym_size(-rank) == y.sym_size(-rank),
              "Paired inputs require equal sample counts");
  TORCH_CHECK(size > 0, "chunk_size must be positive");
  bool autocast = !x.device().is_meta() &&
                  at::autocast::is_autocast_enabled(x.device().type());
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble ||
                  (autocast && (x.scalar_type() == at::kHalf ||
                                x.scalar_type() == at::kBFloat16)),
              "Inputs require float32 or float64 outside autocast");
  if (!x.device().is_meta())
    utils::check_values(at::isfinite(x).all() & at::isfinite(y).all(),
                        "Inputs must be finite");
}
inline std::vector<c10::SymInt> batch_shape(const Tensor& x, const Tensor& y,
                                            int64_t rank) {
  return at::infer_size_symint(x.sym_sizes().slice(0, x.dim() - rank),
                               y.sym_sizes().slice(0, y.dim() - rank));
}
inline Tensor normalize(const Tensor& x, const std::vector<c10::SymInt>& batch,
                        int64_t rank) {
  c10::SymInt B = 1;
  for (const auto& d : batch)
    B *= d;
  auto expanded = batch;
  auto flat = std::vector<c10::SymInt>{B};
  for (int64_t i = -rank; i < 0; ++i) {
    expanded.push_back(x.sym_size(i));
    flat.push_back(x.sym_size(i));
  }
  return x.expand_symint(expanded).reshape_symint(flat);
}
inline std::vector<c10::SymInt> output_shape(const Tensor& x, const Tensor& y,
                                             int64_t rank, int64_t mode) {
  auto shape = batch_shape(x, y, rank);
  auto n = x.sym_size(-rank);
  shape.push_back(mode == 2 ? n * (n - 1) / 2 : n);
  if (mode == 0)
    shape.push_back(y.sym_size(-rank));
  return shape;
}
inline Features slice(const Features& inputs, int64_t start, int64_t count) {
  Features result;
  for (const auto& t : inputs)
    result.push_back(t.narrow(1, start, count));
  return result;
}
template <class F>
void tiles(int64_t n, int64_t m, int64_t mode, int64_t size, F&& function) {
  int64_t offset = 0;
  for (int64_t i = 0; i < n; i += mode == 1 ? std::min(size, n - i) : 1) {
    auto count = mode == 1 ? std::min(size, n - i) : int64_t(1);
    for (int64_t j = mode == 2   ? i + 1
                     : mode == 1 ? i
                                 : 0;
         j < (mode == 1 ? i + 1 : m);
         j += mode == 1 ? count : std::min(size, m - j)) {
      auto other = mode == 1 ? count : std::min(size, m - j);
      function(i, j, count, other, offset + j - i - 1);
    }
    if (mode == 2)
      offset += n - i - 1;
  }
}
template <class Feature, class Score>
Tensor assemble(const Tensor& x, const Tensor& y, int64_t mode, int64_t size,
                int64_t rank, Feature&& features, Score&& score) {
  auto n = x.size(-rank), m = y.size(-rank);
  auto batch = batch_shape(x, y, rank);
  int64_t B = 1;
  for (const auto& dimension : batch)
    B *= dimension.expect_int();
  auto shape = output_shape(x, y, rank, mode);
  if (B == 0 || n == 0 || m == 0 || (mode == 2 && n < 2))
    return at::empty_symint(shape, x.options());
  auto a = features(x), b = x.is_same(y) ? a : features(y);
  for (auto& feature : a)
    feature = normalize(feature, batch, rank);
  for (auto& feature : b)
    feature = normalize(feature, batch, rank);
  std::vector<Tensor> rows, columns;
  int64_t previous = -1;
  tiles(n, m, mode, size,
        [&](int64_t i, int64_t j, int64_t count, int64_t other, int64_t) {
          if (previous != i && !columns.empty()) {
            rows.push_back(at::cat(columns, -1));
            columns.clear();
          }
          previous = i;
          columns.push_back(score(slice(a, i, count), slice(b, j, other)));
        });
  if (!columns.empty())
    rows.push_back(at::cat(columns, -1));
  auto result = mode == 0 ? at::stack(rows, -2) : at::cat(rows, -1);
  return result.reshape_symint(shape);
}
// Replay cross-feature maps one tile at a time. Collection features are built
// once per side, their gradients accumulated, then differentiated once. This
// bounds retained pair state without repeating STFTs or pyramids per partner.
template <class Feature, class Score>
Pair backward(const Tensor& gradient, const Tensor& x, const Tensor& y,
              int64_t mode, int64_t size, int64_t rank, Feature&& features,
              Score&& score) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU);
  c10::impl::ExcludeDispatchKeyGuard no_cuda(c10::DispatchKey::AutocastCUDA);
  if (gradient.numel() == 0)
    return {at::zeros_like(x), at::zeros_like(y)};
  at::AutoGradMode enable(true);
  auto ax = x.detach().set_requires_grad(true),
       by = y.detach().set_requires_grad(true);
  auto a = features(ax), b = features(by);
  auto batch = batch_shape(x, y, rank);
  int64_t B = 1;
  for (const auto& dimension : batch)
    B *= dimension.expect_int();
  auto n = x.size(-rank), m = y.size(-rank);
  std::vector<int64_t> gradient_shape{B, mode == 2 ? n * (n - 1) / 2 : n};
  if (mode == 0)
    gradient_shape.push_back(m);
  auto flat_gradient = gradient.reshape(gradient_shape);
  for (auto& feature : a)
    feature = normalize(feature, batch, rank);
  for (auto& feature : b)
    feature = normalize(feature, batch, rank);
  Features ga, gb;
  for (const auto& t : a)
    ga.push_back(at::zeros(t.sizes(), t.options().dtype(at::kDouble)));
  for (const auto& t : b)
    gb.push_back(at::zeros(t.sizes(), t.options().dtype(at::kDouble)));
  tiles(
      n, m, mode, size,
      [&](int64_t i, int64_t j, int64_t count, int64_t other, int64_t offset) {
        Features al, bl, leaves;
        for (const auto& t : a) {
          al.push_back(t.narrow(1, i, count).detach().set_requires_grad(true));
          leaves.push_back(al.back());
        }
        for (const auto& t : b) {
          bl.push_back(t.narrow(1, j, other).detach().set_requires_grad(true));
          leaves.push_back(bl.back());
        }
        auto value = score(al, bl);
        auto g = mode == 0   ? flat_gradient.select(-2, i).narrow(-1, j, other)
                 : mode == 1 ? flat_gradient.narrow(-1, i, count)
                             : flat_gradient.narrow(-1, offset, other);
        auto grads =
            torch::autograd::grad({value}, leaves, {g}, false, false, true);
        at::NoGradGuard no_grad;
        for (size_t k = 0; k < a.size(); ++k)
          if (grads[k].defined())
            ga[k].narrow(1, i, count).add_(grads[k].to(at::kDouble));
        for (size_t k = 0; k < b.size(); ++k)
          if (grads[a.size() + k].defined())
            gb[k].narrow(1, j, other).add_(grads[a.size() + k].to(at::kDouble));
      });
  Features outputs, grads;
  for (size_t k = 0; k < a.size(); ++k)
    if (a[k].requires_grad()) {
      outputs.push_back(a[k]);
      grads.push_back(ga[k].to(a[k].scalar_type()));
    }
  for (size_t k = 0; k < b.size(); ++k)
    if (b[k].requires_grad()) {
      outputs.push_back(b[k]);
      grads.push_back(gb[k].to(b[k].scalar_type()));
    }
  auto result =
      torch::autograd::grad(outputs, {ax, by}, grads, false, false, true);
  return {result[0].defined() ? result[0] : at::zeros_like(x),
          result[1].defined() ? result[1] : at::zeros_like(y)};
}
}  // namespace torchpairwise::ops::modality_detail
