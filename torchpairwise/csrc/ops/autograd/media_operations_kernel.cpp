#include <torch/autograd.h>
#include <torch/library.h>

#include "../media_operations_detail.h"
namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = std::vector<Tensor>;
class MediaFeaturesFunction
    : public torch::autograd::Function<MediaFeaturesFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        int64_t kind, std::vector<int64_t> settings,
                        std::vector<double> parameters) {
    ctx->save_for_backward({x});
    ctx->saved_data["kind"] = kind;
    ctx->saved_data["settings"] = settings;
    ctx->saved_data["parameters"] = parameters;
    at::AutoDispatchBelowADInplaceOrView guard;
    return detail::media::features(x, kind, settings, parameters);
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of media features are unsupported");
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_media_features_backward", "")
            .typed<detail::media::FeatureGradSignature>();
    auto result = op.call(gradients[0], ctx->get_saved_variables()[0],
                          ctx->saved_data["kind"].toInt(),
                          ctx->saved_data["settings"].toIntVector(),
                          ctx->saved_data["parameters"].toDoubleVector());
    return {result, Tensor(), Tensor(), Tensor()};
  }
};
Tensor media_features_autograd(const Tensor& x, int64_t kind,
                               at::IntArrayRef settings,
                               at::ArrayRef<double> parameters) {
  return MediaFeaturesFunction::apply(x, kind, settings.vec(),
                                      parameters.vec());
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_media_features", TORCH_FN(media_features_autograd));
  m.impl("_media_features_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
}  // namespace torchpairwise::ops
