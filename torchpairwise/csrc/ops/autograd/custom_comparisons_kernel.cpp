#include <torch/autograd.h>
#include <torch/library.h>

#include "../custom_comparisons_detail.h"

namespace torchpairwise::ops::detail::custom {
namespace {
struct PairState {
  std::string names;
  std::vector<int64_t> shapes;
  int64_t dtype;
};
class CustomPairFunction
    : public torch::autograd::Function<CustomPairFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, at::TensorList v,
                        PairState state) {
    ctx->save_for_backward(v.vec());
    ctx->saved_data["names"] = state.names;
    ctx->saved_data["shapes"] = state.shapes;
    return custom::forward(v[0], v[1], v[2], state.names, state.shapes,
                           state.dtype);
  }
  static Tensors backward(torch::autograd::AutogradContext* ctx,
                          const Tensors& g) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher custom VJP derivatives are unsupported");
    auto v = ctx->get_saved_variables();
    auto result = custom::backward(g[0], v[0], v[1], v[2],
                                   ctx->saved_data["names"].toStringRef(),
                                   ctx->saved_data["shapes"].toIntVector());
    result.push_back(Tensor());
    return result;
  }
};
struct DenseState {
  std::string names;
  std::vector<int64_t> shapes;
  int64_t mode, rank, dtype, size;
};
class CustomComparisonFunction
    : public torch::autograd::Function<CustomComparisonFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, at::TensorList v,
                        DenseState state) {
    ctx->save_for_backward(v.vec());
    ctx->saved_data["names"] = state.names;
    ctx->saved_data["shapes"] = state.shapes;
    ctx->saved_data["control"] =
        std::vector<int64_t>{state.mode, state.rank, state.dtype, state.size};
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_custom_compare", "")
            .typed<DenseSignature>();
    return op.call(v, state.names, state.shapes, state.mode, state.rank,
                   state.dtype, state.size);
  }
  static Tensors backward(torch::autograd::AutogradContext* ctx,
                          const Tensors& g) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher custom comparison derivatives are unsupported");
    auto v = ctx->get_saved_variables();
    auto c = ctx->saved_data["control"].toIntVector();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_custom_compare_backward", "")
            .typed<DenseGradSignature>();
    auto result = op.call(g[0], v, ctx->saved_data["names"].toStringRef(),
                          ctx->saved_data["shapes"].toIntVector(), c[0], c[1],
                          c[2], c[3]);
    result.push_back(Tensor());
    return result;
  }
};
Tensor dense_autograd(at::TensorList v, c10::string_view names,
                      at::IntArrayRef shapes, int64_t mode, int64_t rank,
                      int64_t dtype, int64_t size) {
  return CustomComparisonFunction::apply(
      v, DenseState{std::string(names), shapes.vec(), mode, rank, dtype, size});
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_custom_compare", TORCH_FN(dense_autograd));
  m.impl("_custom_compare_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
Tensor paired_with_vjp(const Tensor& x, const Tensor& y, const Tensor& packed,
                       c10::string_view names, at::IntArrayRef shapes,
                       int64_t dtype) {
  return CustomPairFunction::apply(
      at::TensorList({x, y, packed}),
      PairState{std::string(names), shapes.vec(), dtype});
}
}  // namespace torchpairwise::ops::detail::custom
