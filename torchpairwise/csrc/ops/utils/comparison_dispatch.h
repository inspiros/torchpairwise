#pragma once
#include <ATen/ATen.h>
namespace torchpairwise::ops::detail {
enum class ComparisonFamily { Legacy, Vector, Pointset, Transport };
int64_t ordered_comparison_kind(c10::string_view raw);
int64_t media_comparison_kind(c10::string_view raw);
int64_t structured_comparison_kind(c10::string_view raw);
std::string canonical_comparison_name(c10::string_view raw);
ComparisonFamily comparison_family(c10::string_view name, bool kernel);
at::Tensor call_named_comparison(const at::Tensor& x,
                                 const c10::optional<at::Tensor>& y,
                                 c10::string_view name, bool kernel, double p,
                                 double base, int64_t degree, double gamma,
                                 double coef0, const at::Tensor& variance,
                                 const at::Tensor& inverse);
}  // namespace torchpairwise::ops::detail
