#pragma once
#include "expression_operations.h"
#include "image_comparisons.h"

namespace torchpairwise::ops {
// Terminal top-k along one current event axis; sample and batch axes persist.
TORCHPAIRWISE_API std::tuple<at::Tensor, at::Tensor> image_event_topk(
    const at::Tensor& x, const c10::optional<at::Tensor>& y,
    c10::string_view comparison, at::IntArrayRef program,
    at::ArrayRef<double> corrections, int64_t k, int64_t dim = -1,
    bool largest = true, bool sorted = true,
    const ImageMapOptions& options = {});
// Event-only reduction chains stay lazy. C++ uses an explicit scalar()
// conversion after contracting all events, keeping return types static.
TORCHPAIRWISE_API at::Tensor image_event_reduce(
    const at::Tensor& x, const c10::optional<at::Tensor>& y,
    c10::string_view comparison, at::IntArrayRef program,
    at::ArrayRef<double> corrections, c10::string_view reduction,
    at::IntArrayRef dims, bool keepdim = false,
    c10::optional<double> correction = c10::nullopt,
    const ImageMapOptions& options = {});
class TORCHPAIRWISE_API ImageMapExpression {
 public:
  ImageMapExpression(at::Tensor x, c10::optional<at::Tensor> y,
                     std::string comparison, ImageMapOptions options,
                     std::vector<int64_t> program = {},
                     std::vector<double> corrections = {});
  ImageMapExpression reduce(
      c10::string_view reduction, at::IntArrayRef dims, bool keepdim = false,
      c10::optional<double> correction = c10::nullopt) const;
  ImageMapExpression sum(at::IntArrayRef dims, bool keepdim = false) const {
    return reduce("sum", dims, keepdim);
  }
  ImageMapExpression mean(at::IntArrayRef dims, bool keepdim = false) const {
    return reduce("mean", dims, keepdim);
  }
  ImageMapExpression min(at::IntArrayRef dims, bool keepdim = false) const {
    return reduce("min", dims, keepdim);
  }
  ImageMapExpression max(at::IntArrayRef dims, bool keepdim = false) const {
    return reduce("max", dims, keepdim);
  }
  ImageMapExpression prod(at::IntArrayRef dims, bool keepdim = false) const {
    return reduce("prod", dims, keepdim);
  }
  ImageMapExpression all(at::IntArrayRef dims, bool keepdim = false) const {
    return reduce("all", dims, keepdim);
  }
  ImageMapExpression any(at::IntArrayRef dims, bool keepdim = false) const {
    return reduce("any", dims, keepdim);
  }
  ImageMapExpression var(at::IntArrayRef dims, double correction = 1.,
                         bool keepdim = false) const {
    return reduce("var", dims, keepdim, correction);
  }
  ImageMapExpression std(at::IntArrayRef dims, double correction = 1.,
                         bool keepdim = false) const {
    return reduce("std", dims, keepdim, correction);
  }
  ScalarExpression scalar() const;
  std::tuple<at::Tensor, at::Tensor> topk(int64_t k, int64_t dim = -1,
                                          bool largest = true,
                                          bool sorted = true) const {
    return image_event_topk(x_, y_, comparison_, program_, corrections_, k, dim,
                            largest, sorted, options_);
  }
  at::Tensor reduce_samples(
      c10::string_view reduction, at::IntArrayRef dims, bool keepdim = false,
      c10::optional<double> correction = c10::nullopt) const;

 private:
  at::Tensor x_;
  c10::optional<at::Tensor> y_;
  std::string comparison_;
  ImageMapOptions options_;
  std::vector<int64_t> program_;
  std::vector<double> corrections_;
};
// Full-event convenience methods retain their existing ScalarExpression API.
class TORCHPAIRWISE_API ImageMapOperator {
 public:
  ImageMapOperator(at::Tensor x, c10::optional<at::Tensor> y = c10::nullopt,
                   std::string comparison = "squared_error",
                   ImageMapOptions options = {});
  ImageMapExpression events() const {
    return ImageMapExpression(x_, y_, comparison_, options_);
  }
  std::tuple<at::Tensor, at::Tensor> topk(int64_t k, int64_t dim = -1,
                                          bool largest = true,
                                          bool sorted = true) const {
    return events().topk(k, dim, largest, sorted);
  }
  ScalarExpression reduce(
      c10::string_view reduction = "mean", at::IntArrayRef dims = {-3, -2, -1},
      c10::optional<double> correction = c10::nullopt) const;
  ScalarExpression sum(at::IntArrayRef dims = {-3, -2, -1}) const {
    return reduce("sum", dims);
  }
  ScalarExpression mean(at::IntArrayRef dims = {-3, -2, -1}) const {
    return reduce("mean", dims);
  }
  ScalarExpression min(at::IntArrayRef dims = {-3, -2, -1}) const {
    return reduce("min", dims);
  }
  ScalarExpression max(at::IntArrayRef dims = {-3, -2, -1}) const {
    return reduce("max", dims);
  }
  ScalarExpression prod(at::IntArrayRef dims = {-3, -2, -1}) const {
    return reduce("prod", dims);
  }
  ScalarExpression all(at::IntArrayRef dims = {-3, -2, -1}) const {
    return reduce("all", dims);
  }
  ScalarExpression any(at::IntArrayRef dims = {-3, -2, -1}) const {
    return reduce("any", dims);
  }
  ScalarExpression var(at::IntArrayRef dims = {-3, -2, -1},
                       double correction = 1.) const {
    return reduce("var", dims, correction);
  }
  ScalarExpression std(at::IntArrayRef dims = {-3, -2, -1},
                       double correction = 1.) const {
    return reduce("std", dims, correction);
  }

 private:
  at::Tensor x_;
  c10::optional<at::Tensor> y_;
  std::string comparison_;
  ImageMapOptions options_;
};
TORCHPAIRWISE_API ImageMapOperator image_map_operator(
    const at::Tensor& x, const c10::optional<at::Tensor>& y = c10::nullopt,
    c10::string_view comparison = "squared_error",
    const ImageMapOptions& options = {});
}  // namespace torchpairwise::ops
