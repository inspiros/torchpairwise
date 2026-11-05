#pragma once
#include <functional>

#include "image_map_operations.h"
#include "operator_sources_detail.h"

namespace torchpairwise::ops::detail::media {
using at::Tensor;
// Shape-only identity Y-X adjoint. Reduced dimensions in gradient stay
// singleton until sample cotangents have been summed at public precision.
std::tuple<Tensor, Tensor> identity_linear_backward(
    const Tensor& gradient, const Tensor& a, const Tensor& b,
    at::IntArrayRef settings, int64_t dtype, double divisor, bool paired);
// Scatter identity Y-X extrema adjoints from saved logical witnesses. Return
// false if double witness IDs cannot represent the complete index domain.
bool identity_extrema_backward(Tensor dx, Tensor dy, const Tensor& gradient,
                               const Tensor& witnesses, int64_t dtype,
                               bool sample_ids, bool paired,
                               int64_t sample_size);
// Programs contain (reducer opcode, current event-axis mask, keepdim) triples.
// Corrections contain one finite value per stage. No sample axes occur here.
using EventOutputs = std::tuple<Tensor, Tensor>;
using EventTopKSignature = EventOutputs(const Tensor&, const Tensor&, int64_t,
                                        at::IntArrayRef, at::ArrayRef<double>,
                                        int64_t, at::IntArrayRef,
                                        at::ArrayRef<double>, int64_t, int64_t,
                                        bool, int64_t);
using EventTopKGradSignature =
    EventOutputs(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                 int64_t, at::IntArrayRef, at::ArrayRef<double>, int64_t,
                 at::IntArrayRef, at::ArrayRef<double>, int64_t, int64_t);
using EventSignature = EventOutputs(const Tensor&, const Tensor&, int64_t,
                                    at::IntArrayRef, at::ArrayRef<double>,
                                    int64_t, at::IntArrayRef,
                                    at::ArrayRef<double>, c10::string_view,
                                    at::IntArrayRef, bool, double, int64_t,
                                    bool);
using EventGradSignature = std::tuple<Tensor, Tensor>(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, int64_t,
    at::IntArrayRef, at::ArrayRef<double>, int64_t, at::IntArrayRef,
    at::ArrayRef<double>, c10::string_view, at::IntArrayRef, bool, double,
    int64_t, bool);
// Native-only visitors; no Python callback participates in built-in execution.
using MapRegionConsumer = std::function<void(
    const Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)>;
using MapRegionCotangent = std::function<Tensor(
    const Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)>;
void visit_map_regions(const Tensor& a, const Tensor& b, int64_t comparison,
                       at::IntArrayRef settings,
                       at::ArrayRef<double> parameters, int64_t dtype,
                       bool paired, const MapRegionConsumer& consume,
                       bool promote = false);
std::tuple<Tensor, Tensor> map_backward_regions(
    const MapRegionCotangent& cotangent_for, bool needs_value, const Tensor& a,
    const Tensor& b, int64_t comparison, at::IntArrayRef settings,
    at::ArrayRef<double> parameters, int64_t dtype, bool paired,
    int64_t element_budget = int64_t(1) << 23);
#ifdef WITH_CUDA
// Generate rounded pixel maps, optionally promoting the rounded entry in
// registers for double reduction state without a public-dtype temporary.
Tensor pixel_map_region_cuda(const Tensor& x, const Tensor& y,
                             int64_t comparison, int64_t dtype, bool promote);
// Internal optimized product state/adjoint; CPU keeps the ATen algorithms.
void image_product_update_cuda(Tensor state, const Tensor& value,
                               at::IntArrayRef axes);
Tensor image_product_gradient_cuda(const Tensor& state, const Tensor& value,
                                   const Tensor& gradient,
                                   const Tensor& any_zero);
#endif
void update_map_extrema(Tensor state, const Tensor& value, const Tensor& ids,
                        at::IntArrayRef axes, bool smallest);
using MapValuesSignature = Tensor(const Tensor&, const Tensor&, int64_t,
                                  at::IntArrayRef, at::ArrayRef<double>,
                                  int64_t, bool);
using MapValuesGradSignature = std::tuple<Tensor, Tensor>(
    const Tensor&, const Tensor&, const Tensor&, int64_t, at::IntArrayRef,
    at::ArrayRef<double>, int64_t, bool);
Tensor map_values(const Tensor& a, const Tensor& b, int64_t comparison,
                  at::IntArrayRef settings, at::ArrayRef<double> parameters,
                  int64_t dtype, bool paired);
// Keep small replay graphs local; large maps use the opaque regional core.
Tensor map_values_for_replay(const Tensor& a, const Tensor& b,
                             int64_t comparison, at::IntArrayRef settings,
                             at::ArrayRef<double> parameters, int64_t dtype,
                             bool paired);
using MapContractSignature = EventOutputs(const Tensor&, const Tensor&, int64_t,
                                          at::IntArrayRef, at::ArrayRef<double>,
                                          int64_t, int64_t, int64_t, bool,
                                          double, bool);
using MapContractGradSignature = std::tuple<Tensor, Tensor>(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, int64_t,
    at::IntArrayRef, at::ArrayRef<double>, int64_t, int64_t, int64_t, bool,
    double, bool);
Tensor map_contract(const Tensor& a, const Tensor& b, int64_t comparison,
                    at::IntArrayRef settings, at::ArrayRef<double> parameters,
                    int64_t dtype, int64_t code, int64_t mask, bool keepdim,
                    double correction, bool paired);
bool use_map_contract(const Tensor& a, const Tensor& b, int64_t comparison,
                      at::IntArrayRef settings, bool paired);
struct EventMetadata {
  std::vector<int64_t> shape;
  at::ScalarType dtype;
  bool differentiable;
};
EventMetadata event_metadata(int64_t comparison, at::IntArrayRef settings,
                             int64_t dtype, at::IntArrayRef program,
                             at::ArrayRef<double> corrections);
EventOutputs event_reduce(const Tensor& x, const Tensor& y, int64_t comparison,
                          at::IntArrayRef settings,
                          at::ArrayRef<double> parameters, int64_t dtype,
                          at::IntArrayRef program,
                          at::ArrayRef<double> corrections,
                          c10::string_view reduction, at::IntArrayRef axes,
                          bool keepdim, double correction, int64_t size,
                          bool paired);
using MapReductionSignature = Tensor(const Tensor&, const Tensor&, int64_t,
                                     c10::string_view, at::IntArrayRef,
                                     at::ArrayRef<double>, int64_t, double,
                                     bool);
using MapReductionGradSignature = std::tuple<Tensor, Tensor>(
    const Tensor&, const Tensor&, const Tensor&, int64_t, c10::string_view,
    at::IntArrayRef, at::ArrayRef<double>, int64_t, double, bool);
Tensor map_reduction(const Tensor& a, const Tensor& b, int64_t comparison,
                     c10::string_view reduction, at::IntArrayRef settings,
                     at::ArrayRef<double> parameters, int64_t dtype,
                     double correction, bool paired);
}  // namespace torchpairwise::ops::detail::media
namespace torchpairwise::ops::detail::streaming {
PreparedComparison prepare_image_event_operator(
    const Tensor& x, const c10::optional<Tensor>& y,
    c10::string_view comparison, at::IntArrayRef program,
    at::ArrayRef<double> corrections, const PairwiseOperationOptions& matrix,
    const ImageMapOptions& options);
PreparedComparison prepare_image_map_operator(
    const Tensor& x, const c10::optional<Tensor>& y,
    c10::string_view comparison, c10::string_view reduction, double correction,
    const PairwiseOperationOptions& matrix, const ImageMapOptions& options);
}  // namespace torchpairwise::ops::detail::streaming
