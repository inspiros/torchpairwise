#pragma once
#include "linear_solve_detail.h"

namespace torchpairwise::ops::detail::spectral {
using namespace linear;
using Estimate = std::tuple<Tensor, Tensor, Tensor, Tensor>;
using Forward = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>;
using ForwardSignature = Forward(
    TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef shape,
    const Tensor& anchor, c10::string_view function, c10::string_view method,
    c10::optional<int64_t> num_probes, const c10::optional<Tensor>& probes,
    c10::optional<int64_t> seed, c10::optional<int64_t> max_iter,
    int64_t block_size, c10::string_view gradient, double solve_rtol,
    double solve_atol, int64_t solve_max_iter);
using BackwardSignature =
    Tensors(TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef shape,
            const Tensor& anchor, const Tensor& probes, const Tensor& upstream,
            c10::string_view function, c10::string_view method,
            c10::string_view gradient, int64_t block_size, double solve_rtol,
            double solve_atol, int64_t solve_max_iter);
// Call the inference numerical implementation under an explicit no-grad scope.
Estimate estimate(TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef shape,
                  const Tensor& anchor, c10::string_view function,
                  c10::string_view method, c10::optional<int64_t> num_probes,
                  const c10::optional<Tensor>& probes,
                  c10::optional<int64_t> seed, c10::optional<int64_t> max_iter,
                  int64_t block_size);
}  // namespace torchpairwise::ops::detail::spectral
