#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <limits>

#include "../audio_comparisons_detail.h"
namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;

using Grad = Pair(const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t,
                  int64_t, at::IntArrayRef, at::ArrayRef<double>);
class AudioComparisonFunction
    : public torch::autograd::Function<AudioComparisonFunction> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x,
                        Tensor y, int64_t kind, int64_t mode, int64_t size,
                        std::vector<int64_t> settings, std::vector<double> p) {
    ctx->save_for_backward({x, y});
    ctx->saved_data["kind"] = kind;
    ctx->saved_data["mode"] = mode;
    ctx->saved_data["size"] = size;
    ctx->saved_data["settings"] = settings;
    ctx->saved_data["p"] = p;
    at::AutoDispatchBelowADInplaceOrView guard;
    return detail::audio::_audio_compare(x, y, kind, mode, size, settings, p);
  }
  static Values backward(torch::autograd::AutogradContext* ctx,
                         const Values& gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "Higher derivatives of native audio replay are unsupported");
    auto saved = ctx->get_saved_variables();
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_audio_compare_backward", "")
            .typed<Grad>();
    auto result = op.call(
        gradients[0], saved[0], saved[1], ctx->saved_data["kind"].toInt(),
        ctx->saved_data["mode"].toInt(), ctx->saved_data["size"].toInt(),
        ctx->saved_data["settings"].toIntVector(),
        ctx->saved_data["p"].toDoubleVector());
    return {std::get<0>(result),
            std::get<1>(result),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor(),
            Tensor()};
  }
};
Tensor _audio_compare_autograd(const Tensor& x, const Tensor& y, int64_t kind,
                               int64_t mode, int64_t size,
                               at::IntArrayRef settings,
                               at::ArrayRef<double> p) {
  return AudioComparisonFunction::apply(x, y, kind, mode, size, settings.vec(),
                                        p.vec());
}
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_audio_compare", TORCH_FN(_audio_compare_autograd));
  m.impl("_audio_compare_backward", torch::CppFunction::makeFallthrough());
}
}  // namespace
namespace detail::audio {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;

}  // namespace detail::audio
}  // namespace torchpairwise::ops
