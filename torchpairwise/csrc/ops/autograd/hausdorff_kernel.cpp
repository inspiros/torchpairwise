#include <torch/autograd.h>
#include <torch/types.h>

#include "../hausdorff.h"

namespace torchpairwise::ops {
namespace {
class DirectedHausdorffDistancesFunction
    : public torch::autograd::Function<DirectedHausdorffDistancesFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& x1, const torch::autograd::Variable& x2,
      bool shuffle, c10::optional<at::Generator> generator) {
    at::AutoDispatchBelowADInplaceOrView g;

    at::Tensor output, x1_indices, x2_indices;
    std::tie(output, x1_indices, x2_indices) =
        _directed_hausdorff(x1, x2, shuffle, generator);
    ctx->save_for_backward({x1, x2, output, x1_indices, x2_indices});
    ctx->mark_non_differentiable({x1_indices, x2_indices});

    return {
        output,
        x1_indices,
        x2_indices,
    };
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    auto saved = ctx->get_saved_variables();
    auto x1 = saved[0];
    auto x2 = saved[1];
    auto output = saved[2];
    auto i1 = saved[3];
    auto i2 = saved[4];
    auto grad = grad_output[0];
    const bool unbatched = x1.dim() == 3;
    if (unbatched) {
      x1 = x1.unsqueeze(0);
      x2 = x2.unsqueeze(0);
      output = output.unsqueeze(0);
      i1 = i1.unsqueeze(0);
      i2 = i2.unsqueeze(0);
      grad = grad.unsqueeze(0);
    }
    const auto B = x1.size(0), N = x1.size(1), M = x2.size(1), D = x1.size(3);
    auto m = i1.unsqueeze(-1).expand({B, N, M, D});
    auto n = i2.transpose(1, 2).unsqueeze(-1).expand({B, M, N, D});
    auto delta = x1.gather(2, m) - x2.gather(2, n).transpose(1, 2);
    auto denominator = at::where(output.ne(0), output, at::ones_like(output));
    auto contribution = delta * (grad / denominator).unsqueeze(-1);
    // Reuse the exact forward witnesses, including shuffled ties.
    // No RNG replay or pairwise distance recomputation in backward.
    auto grad_x1 = at::zeros_like(x1).scatter_add(2, m, contribution);
    auto grad_x2 =
        at::zeros_like(x2).scatter_add(2, n, -contribution.transpose(1, 2));
    if (unbatched) {
      grad_x1 = grad_x1.squeeze(0);
      grad_x2 = grad_x2.squeeze(0);
    }

    return {
        grad_x1,
        grad_x2,
        torch::autograd::Variable(),
        torch::autograd::Variable(),
    };
  }
};

std::tuple<at::Tensor, at::Tensor, at::Tensor> _directed_hausdorff_autograd(
    const at::Tensor& x1, const at::Tensor& x2, bool shuffle,
    c10::optional<at::Generator> generator) {
  auto result =
      DirectedHausdorffDistancesFunction::apply(x1, x2, shuffle, generator);
  return std::make_tuple(result[0], result[1], result[2]);
}
}  // namespace

TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchpairwise::_directed_hausdorff"),
         TORCH_FN(_directed_hausdorff_autograd));
}
}  // namespace torchpairwise::ops
