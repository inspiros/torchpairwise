#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>

#include <cmath>
#include <iostream>

#include "torchpairwise.h"

int main(int argc, char** argv) {
  at::set_num_threads(1);
  auto device = argc > 1 ? at::Device(argv[1]) : at::Device(at::kCPU);
  auto options = at::TensorOptions().device(device).dtype(at::kDouble);
  auto x = at::rand({3, 2}, options).set_requires_grad(true);
  auto y = at::rand({4, 2}, options).set_requires_grad(true);
  auto values = at::rand({4, 2}, options).set_requires_grad(true);
  using namespace torchpairwise::ops;
  auto expected = (x.unsqueeze(1) - y.unsqueeze(0)).square().sum(-1).sqrt();
  TORCH_CHECK(at::allclose(euclidean_distances(x, y), expected));
  TORCH_CHECK(at::allclose(cdist(x, y, "euclidean"), expected));
  TORCH_CHECK(at::allclose(builtin_comparisons(x, y, "euclidean"), expected));
  TORCH_CHECK(pdist(x, "euclidean").numel() == 3);
  TORCH_CHECK(paired_angular_distances(x, y.narrow(0, 0, 3)).numel() == 3);

  StreamingReductionOptions reduction;
  reduction.values = values;
  reduction.chunk_size = 2;
  reduction.comparison.gamma = .3;
  auto result = std::get<0>(
      streaming_reductions(x, y, "rbf", "kernel_sum", true, reduction));
  expected = (-.3 * expected.square()).exp().matmul(values);
  TORCH_CHECK(at::allclose(result, expected));
  auto gradient = torch::autograd::grad({result.sum()}, {x, y, values});
  auto reference = torch::autograd::grad({expected.sum()}, {x, y, values});
  for (size_t i = 0; i < gradient.size(); ++i)
    TORCH_CHECK(at::allclose(gradient[i], reference[i]));
  auto js = jensenshannon_distances(x, y).square();
  auto p = (x / x.sum(-1, true)).unsqueeze(1);
  auto q = (y / y.sum(-1, true)).unsqueeze(0);
  auto midpoint = (p + q) * .5;
  expected = (p * (p / midpoint).log() + q * (q / midpoint).log()).sum(-1) * .5;
  TORCH_CHECK(at::allclose(js, expected));
  gradient = torch::autograd::grad({js.sum()}, {x, y});
  reference = torch::autograd::grad({expected.sum()}, {x, y});
  for (size_t i = 0; i < gradient.size(); ++i)
    TORCH_CHECK(at::allclose(gradient[i], reference[i]));
  auto clouds_x = x.unsqueeze(0), clouds_y = y.unsqueeze(0);
  TransportComparisonOptions mmd;
  mmd.kernel = "polynomial";
  mmd.gamma = .4;
  mmd.degree = 3;
  mmd.coef0 = 1.;
  mmd.squared = true;
  auto poly = [](const at::Tensor& a, const at::Tensor& b) {
    return (.4 * a.matmul(b.transpose(-1, -2)) + 1.).pow(3).mean();
  };
  expected = poly(x, x) + poly(y, y) - 2 * poly(x, y);
  result = mmd_comparisons(clouds_x, clouds_y, "all_pairs", mmd);
  TORCH_CHECK(at::allclose(result.squeeze(), expected));
  TORCH_CHECK(at::allclose(
      transport_comparisons(clouds_x, clouds_y, "mmd", "all_pairs", mmd),
      result));
  BuiltinComparisonOptions selector;
  selector.kernel = mmd.kernel;
  selector.gamma = mmd.gamma;
  selector.degree = mmd.degree;
  selector.coef0 = mmd.coef0;
  selector.squared = mmd.squared;
  TORCH_CHECK(at::allclose(
      builtin_comparisons(clouds_x, clouds_y, "mmd", "paired", false, selector),
      result.squeeze(-1)));
  gradient = torch::autograd::grad({result.sum()}, {x, y});
  reference = torch::autograd::grad({expected}, {x, y});
  for (size_t i = 0; i < gradient.size(); ++i)
    TORCH_CHECK(at::allclose(gradient[i], reference[i]));
  auto xx = (.4 * x.matmul(x.transpose(-1, -2)) + 1.).pow(3);
  auto yy = (.4 * y.matmul(y.transpose(-1, -2)) + 1.).pow(3);
  auto xy = (.4 * x.matmul(y.transpose(-1, -2)) + 1.).pow(3);
  expected = (xx.sum() - xx.diagonal().sum()) / (x.size(0) * (x.size(0) - 1)) +
             (yy.sum() - yy.diagonal().sum()) / (y.size(0) * (y.size(0) - 1)) -
             2 * xy.mean();
  mmd.squared = c10::nullopt;
  result = mmd_statistics(clouds_x, clouds_y, "all_pairs", "unbiased", mmd);
  TORCH_CHECK(at::allclose(result.squeeze(), expected));
  gradient = torch::autograd::grad({result.sum()}, {x, y});
  reference = torch::autograd::grad({expected}, {x, y});
  for (size_t i = 0; i < gradient.size(); ++i)
    TORCH_CHECK(at::allclose(gradient[i], reference[i]));
  auto identical =
      mmd_statistics(clouds_x, clouds_x, "paired", "unbiased", mmd);
  TORCH_CHECK(identical.lt(0).all().item<bool>(),
              "Signed estimate was clamped");
  KernelOperationOptions kernel_options;
  kernel_options.comparison.gamma = .3;
  kernel_options.chunk_size = 2;
  KernelOperator kernel(x, y, "rbf", kernel_options);
  auto displacement = kernel.displacement(true);
  auto weights = rbf_kernel(x, y, .3);
  TORCH_CHECK(at::allclose(displacement,
                           weights.matmul(y) / weights.sum(-1, true) - x));
  SinkhornMapOptions map_options;
  map_options.chunk_size = 2;
  map_options.max_iter = 4;
  map_options.regularization = .7;
  auto mapped = sinkhorn_map(x, y, map_options);
  auto cost = at::cdist(x, y, 2, 2).square() / .7;
  auto loga = at::full({3}, -std::log(3.), options);
  auto logb = at::full({4}, -std::log(4.), options);
  auto dual = at::zeros_like(logb);
  for (int t = 0; t < 4; ++t) {
    auto u = loga - at::logsumexp(dual.unsqueeze(0) - cost, -1);
    dual = logb - at::logsumexp(u.unsqueeze(-1) - cost, -2);
  }
  auto expected_map = at::softmax(dual.unsqueeze(0) - cost, -1).matmul(y);
  TORCH_CHECK(at::allclose(mapped, expected_map));
  gradient = torch::autograd::grad({mapped.sum()}, {x, y});
  reference = torch::autograd::grad({expected_map.sum()}, {x, y});
  for (size_t i = 0; i < gradient.size(); ++i)
    TORCH_CHECK(at::allclose(gradient[i], reference[i]));
  auto mk = rbf_kernel(x, y, .3);
  auto delta = y.unsqueeze(0) - x.unsqueeze(1);
  auto mean_delta = (mk.unsqueeze(-1) * delta).sum(1) / mk.sum(1, true);
  auto centered_delta = delta - mean_delta.unsqueeze(1);
  expected = (mk.unsqueeze(-1).unsqueeze(-1) * centered_delta.unsqueeze(-1) *
              centered_delta.unsqueeze(-2))
                 .sum(1) /
             mk.sum(1).unsqueeze(-1).unsqueeze(-1);
  result = kernel.residual_outer_product(true, true);
  TORCH_CHECK(at::allclose(result, expected));
  auto moment_gradient = at::randn_like(result);
  gradient = torch::autograd::grad({result}, {x, y}, {moment_gradient});
  reference = torch::autograd::grad({expected}, {x, y}, {moment_gradient});
  for (size_t i = 0; i < gradient.size(); ++i)
    TORCH_CHECK(at::allclose(gradient[i], reference[i]));
  auto ix = at::rand({2, 1, 5, 6}, options).set_requires_grad(true);
  auto iy = at::rand({3, 1, 5, 6}, options).set_requires_grad(true);
  ImageMapOptions im;
  im.chunk_size = 2;
  result = image_maps(ix, iy, "squared_error", "all_pairs", "mean", im);
  expected = (iy.unsqueeze(0) - ix.unsqueeze(1)).square().mean(1);
  TORCH_CHECK(at::allclose(result, expected));
  gradient = torch::autograd::grad({result.sum()}, {ix, iy});
  reference = torch::autograd::grad({expected.sum()}, {ix, iy});
  for (size_t i = 0; i < gradient.size(); ++i)
    TORCH_CHECK(at::allclose(gradient[i], reference[i]));
  im.data_range = 1.;
  im.window_size = 3;
  auto maps = image_maps(ix, iy, "ssim", "all_pairs", "none", im);
  ImageComparisonOptions score_options;
  score_options.data_range = 1.;
  score_options.window_size = 3;
  TORCH_CHECK(at::allclose(
      maps.mean({-3, -2, -1}),
      image_comparisons(ix, iy, "ssim", "all_pairs", score_options)));
  auto left = at::rand({3, 2}, options).set_requires_grad(true);
  auto gram = rbf_kernel(x, y, .3);
  result = kernel.bilinear(left, values);
  expected = left.transpose(0, 1).matmul(gram).matmul(values);
  TORCH_CHECK(at::allclose(result, expected));
  gradient = torch::autograd::grad({result.sum()}, {x, y, left, values});
  reference = torch::autograd::grad({expected.sum()}, {x, y, left, values});
  for (size_t i = 0; i < gradient.size(); ++i)
    TORCH_CHECK(at::allclose(gradient[i], reference[i]));
  gram = rbf_kernel(x, y, .3);
  TORCH_CHECK(at::allclose(kernel.reduce("sum"), gram.sum()));
  TORCH_CHECK(at::allclose(kernel.reduce("mean"), gram.mean()));
  TORCH_CHECK(at::allclose(kernel.diagonal(), gram.diagonal()));
  TORCH_CHECK(at::allclose(kernel.matmul(values), gram.matmul(values)));
  auto row_ids = at::tensor({2, 0}, options.dtype(at::kLong));
  auto column_ids = at::tensor({3, 1}, options.dtype(at::kLong));
  TORCH_CHECK(
      at::allclose(kernel.block(row_ids, column_ids),
                   gram.index_select(0, row_ids).index_select(1, column_ids)));
  KernelOperator self_kernel(x, c10::nullopt, "rbf", kernel_options);
  TORCH_CHECK(at::allclose(
      self_kernel.quadratic(left),
      left.transpose(0, 1).matmul(rbf_kernel(x, x, .3)).matmul(left)));
  {
    auto ex = at::rand({3, 2}, options).set_requires_grad(true);
    auto ey = at::rand({3, 2}, options);
    PairwiseOperationOptions expression_options;
    expression_options.chunk_size = 2;
    auto a =
        pairwise_expression(ex, ey, "sqeuclidean", false, expression_options);
    auto dense = (ex.unsqueeze(1) - ey.unsqueeze(0)).square().sum(-1);
    auto reduced = ((a + 1).sqrt() / 2).clamp_min(.6).reduce("sum", -1);
    auto dense_reduced = ((dense + 1).sqrt() / 2).clamp_min(.6).sum(-1);
    TORCH_CHECK(at::allclose(reduced, dense_reduced));
    auto actual_grad = torch::autograd::grad({reduced.sum()}, {ex});
    auto dense_grad = torch::autograd::grad({dense_reduced.sum()}, {ex});
    TORCH_CHECK(at::allclose(actual_grad[0], dense_grad[0]));

    auto labels = at::tensor({0, 0, 1}, options.dtype(at::kLong));
    auto ids = at::arange(3, options.dtype(at::kLong));
    auto mask =
        same_group_mask(a, labels).logical_and(diagonal_mask(a).logical_not());
    TORCH_CHECK(mask.metadata().dtype == at::kBool);
    TORCH_CHECK(a.row_index().metadata().dtype == at::kLong);
    auto dense_mask = labels.unsqueeze(1).eq(labels.unsqueeze(0)) &
                      ids.unsqueeze(1).ne(ids.unsqueeze(0));
    TORCH_CHECK(at::allclose(a.masked(mask).reduce("sum", -1),
                             at::where(dense_mask, dense, 0).sum(-1)));
    auto causal = ids.unsqueeze(0).le(ids.unsqueeze(1));
    TORCH_CHECK(at::equal(causal_mask(a).reduce("sum", -1), causal.sum(-1)));
    TORCH_CHECK(
        at::allclose(a.where(causal_mask(a), -INFINITY).logsumexp(-1),
                     at::where(causal, dense, -INFINITY).logsumexp(-1)));

    auto t = at::tensor({.2, .5, .9}, options);
    auto rows = a.transpose().row_values(t);
    auto dense_rows = t.unsqueeze(-1).expand({3, 3});
    TORCH_CHECK(at::allclose(rows.reduce("sum", -1), dense_rows.sum(-1)));
    TORCH_CHECK(at::allclose(rows.transpose().reduce("sum", -1),
                             dense_rows.transpose(-1, -2).sum(-1)));
  }
  {
    auto cost = pairwise_expression(x, y, "sqeuclidean");
    auto a = at::full({3}, 1. / 3., options);
    auto b = at::full({4}, 1. / 4., options);
    SinkhornOptions ot_options;
    ot_options.epsilon = .5;
    ot_options.rtol = 1e-9;
    auto fit = sinkhorn(cost, a, b, ot_options);
    auto plan = sinkhorn_plan(cost, fit, a, b, ot_options.epsilon);
    TORCH_CHECK(fit.converged.all().item<bool>());
    TORCH_CHECK(at::allclose(plan.reduce("sum", -1), a, 1e-7, 1e-9));
    auto ot_gradient = torch::autograd::grad({fit.objective}, {x});
    TORCH_CHECK(ot_gradient[0].isfinite().all().item<bool>());
  }
  TORCH_CHECK(torchpairwise::cuda_version() >= -1);
  {
    at::NoGradGuard no_grad;
    auto diagonal = at::arange(1., 5., options);
    auto matrix = at::diag(diagonal);
    auto op = LinearOperator(matrix);
    auto fit = pivoted_cholesky(op, 4);
    TORCH_CHECK(at::allclose(fit.factor.matmul(fit.factor.transpose(-1, -2)),
                             matrix));
    auto initial = at::ones({4}, options);
    auto projection = lanczos(op, 4, initial);
    TORCH_CHECK(projection.iterations.item<int64_t>() == 4);
    auto cache = love_cache(op, 4, initial);
    TORCH_CHECK(at::allclose(cache.root.matmul(cache.root.transpose(-1, -2)),
                             at::diag(diagonal.reciprocal())));
    auto points = at::arange(4, options).unsqueeze(-1);
    auto distances = LinearOperator(pairwise_expression(points));
    auto [selected, insertion] = farthest_point_sampling(distances, 3);
    TORCH_CHECK(selected.select(0, 0).item<int64_t>() == 0);
    TORCH_CHECK(selected.select(0, 1).item<int64_t>() == 3);
    auto seeds = kmeans_plusplus(
        LinearOperator(pairwise_expression(points, c10::nullopt,
                                            "sqeuclidean")), 3);
    TORCH_CHECK(seeds.numel() == 3);
    auto cross = LinearOperator(matrix);
    auto variance = predictive_variance(cache, cross, diagonal);
    TORCH_CHECK(variance.abs().max().item<double>() < 1e-10);
  }
  std::cout << device << ": standalone native dispatch and gradients passed\n";
}
