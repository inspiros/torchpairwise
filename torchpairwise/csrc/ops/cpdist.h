#pragma once

#include <ATen/ATen.h>

#include "../macros.h"
#include "comparison_selectors.h"

#define TORCHPAIRWISE_CPDIST_EXTRA_ARGS_SCHEMA_STR                          \
  "Tensor? w=None, Tensor? V=None, Tensor? VI=None, float? p=None, float? " \
  "base=None, bool? shuffle=None, Generator? generator=None"

namespace torchpairwise::ops {
TORCHPAIRWISE_API at::Tensor cdist(
    const at::Tensor& x1, const at::Tensor& x2,
    c10::string_view metric = "minkowski",
    const c10::optional<at::Tensor>& w = c10::nullopt,
    const c10::optional<at::Tensor>& V = c10::nullopt,
    const c10::optional<at::Tensor>& VI = c10::nullopt,
    c10::optional<double> p = c10::nullopt,
    c10::optional<double> base = c10::nullopt,
    c10::optional<bool> shuffle = c10::nullopt,
    c10::optional<at::Generator> generator = c10::nullopt);

TORCHPAIRWISE_API at::Tensor pdist(
    const at::Tensor& input, c10::string_view metric = "minkowski",
    const c10::optional<at::Tensor>& w = c10::nullopt,
    const c10::optional<at::Tensor>& V = c10::nullopt,
    const c10::optional<at::Tensor>& VI = c10::nullopt,
    c10::optional<double> p = c10::nullopt,
    c10::optional<double> base = c10::nullopt,
    c10::optional<bool> shuffle = c10::nullopt,
    c10::optional<at::Generator> generator = c10::nullopt);

// Broad selectors with the same names, roles, options and defaults as Python.
TORCHPAIRWISE_API at::Tensor cdist(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::string_view metric = "minkowski",
    const ComparisonOptions& options = {});
TORCHPAIRWISE_API at::Tensor cdist(
    at::TensorList x1,
    const c10::optional<ComparisonCollection>& x2 = c10::nullopt,
    c10::string_view metric = "minkowski",
    const ComparisonOptions& options = {});
TORCHPAIRWISE_API at::Tensor pdist(const at::Tensor& x1,
                                   c10::string_view metric,
                                   const ComparisonOptions& options);
TORCHPAIRWISE_API at::Tensor pdist(at::TensorList x1,
                                   c10::string_view metric = "minkowski",
                                   const ComparisonOptions& options = {});
TORCHPAIRWISE_API at::Tensor pdist_kernel(
    const at::Tensor& input, c10::string_view kernel = "rbf",
    int64_t degree = 3, c10::optional<double> gamma = c10::nullopt,
    double coef0 = 1.);
}  // namespace torchpairwise::ops
