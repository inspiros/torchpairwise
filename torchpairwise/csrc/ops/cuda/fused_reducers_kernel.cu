
#include <algorithm>
#include <limits>
#include <type_traits>
#include <tuple>
#include <utility>

#include "../fused_reducers.h"
#include "cuda_helpers.h"
#include "pair_functors.cuh"

namespace torchpairwise::ops::detail::fused {
namespace {
using pairs::Chain;
using pairs::ChainOp;
using pairs::Feature;
using pairs::Outer;
using pairs::OuterParams;
static_assert(pairs::max_chain == max_chain);
constexpr int threads = 128;
constexpr int64_t max_grid_z = 65535;

// Columns staged per shared-memory tile; (MaxD + Ct) * tile values stay
// within 48 KiB.
template <int MaxD>
constexpr int column_tile() {
  return MaxD <= 16 ? 128 : 64;
}

// Scalars cast to the input dtype exactly as the native tile path casts
// them (ATen divides by a wrapped scalar via its in-dtype reciprocal).
template <typename T>
OuterParams<T> outer_params(const Plan& p) {
  OuterParams<T> o{Outer(p.outer), 0, T(0), T(0), T(0), T(0)};
  switch (o.kind) {
    case Outer::Root:
      o.a = T(p.a);
      o.code = p.a == .5 ? 1 : p.a == 2 ? 2 : p.a == 3 ? 3 : 0;
      break;
    case Outer::ScaledExp:
      o.a = T(p.a);
      break;
    case Outer::Matern05:
    case Outer::Matern15:
    case Outer::Matern25:
      o.a = T(1) / T(p.a);
      o.b = T(p.b);
      o.c = T(-4) * std::log(std::numeric_limits<T>::min());
      o.d = T(1) / T(3);
      break;
    case Outer::RationalQuadratic:
      o.a = T(1) / T(p.a);
      o.b = T(1) / T(p.b);
      o.c = T(-p.b);
      break;
    case Outer::Polynomial:
      o.a = T(p.a);  // gamma
      o.b = T(p.b);  // coef0
      o.c = T(p.c);  // integer degree
      o.code = p.c == 2 ? 2 : p.c == 3 ? 3 : 0;
      break;
    case Outer::Sigmoid:
      o.a = T(p.a);
      o.b = T(p.b);
      break;
    default:
      break;
  }
  return o;
}

// Everything a kernel needs besides the sample geometry and channels.
// shift == nullptr disables the softmax weighting (see Probability).
template <typename T>
struct Context {
  OuterParams<T> outer;
  Chain<T> chain;
  const T* constants[max_chain];  // chain numbers on device, or nullptr
  const T* values[max_chain];     // row/column values [B or 1, samples]
  int64_t value_stride[max_chain];
  // scale: the chain derivative, set by resolve() unless dynamic (row and
  // column operands, exp, division or masks make it per pair).
  T power, scale;
  bool dynamic, column_values;
  const T *shift, *mass, *center;
  int64_t state_stride;
  bool on_columns, exclusion;
};

template <typename T>
Context<T> context(const Plan& p, bool exclusion, const Probability& w) {
  Context<T> c{};
  c.outer = outer_params<T>(p);
  c.chain.size = p.chain_size;
  for (int k = 0; k < p.chain_size; ++k) {
    const auto op = ChainOp(p.chain_op[k]);
    c.chain.op[k] = op;
    c.chain.axis[k] = p.chain_axis[k];
    const auto& constant = p.chain_constant[k];
    c.constants[k] =
        constant.defined() ? constant.template const_data_ptr<T>() : nullptr;
    const auto& value = p.chain_values[k];
    if (value.defined()) {
      c.values[k] = value.template const_data_ptr<T>();
      c.value_stride[k] = value.size(0) == 1 ? 0 : value.size(1);
    }
    c.dynamic |= p.chain_axis[k] != 0 || op == ChainOp::Exp ||
                 op == ChainOp::Div || op == ChainOp::Keep ||
                 op == ChainOp::Drop;
    c.column_values |= p.chain_axis[k] == 2;
  }
  c.power = T(p.power);
  c.exclusion = exclusion;
  if (w.shift.defined()) {
    c.shift = w.shift.const_data_ptr<T>();
    c.mass = w.mass.const_data_ptr<T>();
    c.center = w.center.defined() ? w.center.const_data_ptr<T>() : nullptr;
    c.state_stride = w.shift.size(-1);
    c.on_columns = w.on_columns;
  }
  return c;
}

// Loads the chain constants and derives the chain's derivative of a static
// chain, the product of its multiplicative constants and signs (divisions
// too, for the fast kernels; the per-pair slope applies them in this order).
template <typename T>
__device__ __forceinline__ void resolve(Context<T>& c) {
  c.scale = T(1);
  for (int k = 0; k < c.chain.size; ++k) {
    if (c.constants[k] != nullptr)
      c.chain.constant[k] = *c.constants[k];
    if (c.chain.op[k] == ChainOp::Mul)
      c.scale *= c.chain.constant[k];
    else if (c.chain.op[k] == ChainOp::Div)
      c.scale /= c.chain.constant[k];
    else if (c.chain.op[k] == ChainOp::Neg || c.chain.op[k] == ChainOp::SubFrom)
      c.scale = -c.scale;
  }
}

// Fast kernels serve float32 plans of up to 16 features whose chain applies
// numbers only (Path::Numbers), or no chain and no softmax weighting
// (Path::Plain). Their outer kind is a template argument, so its switch folds
// away; features are zero-padded to MaxD, which adds exact zeros, instead of
// a per-feature predicate; and the chain runs as a rolled loop of Steps.
// Values are bit-identical to the general kernels.
// O is an Outer code, or runtime_outer (template arguments stay integers:
// nvcc's host stubs cannot name a cast enum value).
enum class Path { General, Numbers, Plain };
constexpr int runtime_outer = -1;
template <int O, typename T>
__device__ __forceinline__ OuterParams<T> fixed(OuterParams<T> o) {
  if constexpr (O != runtime_outer)
    o.kind = Outer(O);
  return o;
}

// A number chain in shared memory. Add, Sub, SubFrom, Mul and Neg are each
// add_rn(mul_rn(value, scale), offset) with one exact operation: a scale of
// 1 or -1, or an offset of -0, which adds exactly. Div and Exp keep their op.
enum StepKind : int { MulAdd, Divide, Exponential };
template <typename T>
struct Steps {
  int size, kind[max_chain];
  T scale[max_chain], offset[max_chain];
};
// Thread k writes step k after resolve(); readers wait for a barrier.
template <typename T>
__device__ __forceinline__ void share(const Context<T>& c, Steps<T>& steps) {
  const int k = threadIdx.x;
  if (k == 0)
    steps.size = c.chain.size;
  if (k >= c.chain.size)
    return;
  const auto op = c.chain.op[k];
  const T t = c.chain.constant[k];
  steps.kind[k] = op == ChainOp::Div   ? Divide
                  : op == ChainOp::Exp ? Exponential
                                       : MulAdd;
  steps.scale[k] = op == ChainOp::Mul || op == ChainOp::Div ? t
                   : op == ChainOp::SubFrom || op == ChainOp::Neg ? T(-1)
                                                                  : T(1);
  steps.offset[k] = op == ChainOp::Add || op == ChainOp::SubFrom ? t
                    : op == ChainOp::Sub                         ? -t
                                                                 : -T(0);
}
template <typename T>
__device__ __forceinline__ T run_steps(const Steps<T>& steps, T value) {
#pragma unroll 1
  for (int k = 0; k < steps.size; ++k) {
    const int kind = steps.kind[k];
    if (kind == MulAdd)
      value = pairs::add_rn(pairs::mul_rn(value, steps.scale[k]),
                            steps.offset[k]);
    else if (kind == Divide)
      value = value / steps.scale[k];
    else
      value = ::exp(value);
  }
  return value;
}

// Per-pair chain operands: numbers and the values of row i, loaded once per
// thread, then each column's values (a warp reads the same column, so the
// loads broadcast).
template <typename T>
__device__ __forceinline__ void row_operands(const Context<T>& c, int64_t b,
                                             int64_t i, bool valid,
                                             T (&operand)[max_chain]) {
#pragma unroll
  for (int k = 0; k < max_chain; ++k)
    operand[k] = k < c.chain.size && c.chain.axis[k] == 1
                     ? (valid ? c.values[k][b * c.value_stride[k] + i] : T(0))
                     : c.chain.constant[k];
}
template <typename T>
__device__ __forceinline__ void column_operands(const Context<T>& c, int64_t b,
                                                int64_t j,
                                                T (&operand)[max_chain]) {
  if (!c.column_values)
    return;
#pragma unroll
  for (int k = 0; k < max_chain; ++k)
    if (k < c.chain.size && c.chain.axis[k] == 2)
      operand[k] = __ldg(c.values[k] + b * c.value_stride[k] + j);
}

struct Geometry {
  int64_t batch, batch_offset, n, m, chunk;
  int d;
};
// Per-sample channel operands: [samples, channels] rows at a batch stride
// (0 when shared), restricted to [channel, channel + width).
template <typename T>
struct Channels {
  const T *left, *right;
  int64_t left_stride, right_stride, channels, channel;
  int width;
};

// Pairs summed plainly before each compensated update. One plain sum per
// 128-pair tile lost accuracy against native's blocked GEMM (near-equal
// terms); groups of 8 cost under one extra flop per pair and channel.
constexpr int pair_group = 8;

// The streaming path's compensated() rule: corrections reset to zero once a
// total, term or update is non-finite.
template <typename T>
__device__ __forceinline__ void compensate(T& total, T& error, T term) {
  T updated = total + term;
  bool valid = ::isfinite(total) && ::isfinite(term) && ::isfinite(updated);
  T residual = ::fabs(total) >= ::fabs(term) ? (total - updated) + term
                                             : (term - updated) + total;
  error = valid ? error + residual : T(0);
  total = updated;
}

// Pad: x and y are zero beyond d (fast kernels).
template <typename T, Feature F, int MaxD, bool Pad = false>
__device__ __forceinline__ T reduce_features(const T (&x)[MaxD], const T* y,
                                             int d, T power) {
  T s = T(0);
#pragma unroll
  for (int k = 0; k < MaxD; ++k)
    if (Pad || k < d)
      pairs::step<F>(s, x[k], y[k], power, k == 0);
  return s;
}

template <typename T>
__device__ __forceinline__ int64_t state_index(const Context<T>& c, int64_t b,
                                               int64_t i, int64_t j) {
  return b * c.state_stride + (c.on_columns ? j : i);
}

// Stages the column samples [start, start + count) and their channel rows;
// Pad zero-fills features d to MaxD.
template <typename T, int MaxD, int Ct, bool Pad = false>
__device__ __forceinline__ void stage(T (&columns)[column_tile<MaxD>()][MaxD],
                                      T (&channels)[column_tile<MaxD>()][Ct],
                                      const T* source, const T* right,
                                      int64_t start, int count, int d,
                                      int64_t stride, int width) {
  __syncthreads();
  if constexpr (Pad)
    for (int t = threadIdx.x; t < count * MaxD; t += threads) {
      const int j = t / MaxD, k = t % MaxD;
      columns[j][k] = k < d ? source[(start + j) * d + k] : T(0);
    }
  else
    for (int t = threadIdx.x; t < count * d; t += threads)
      columns[t / d][t % d] = source[start * d + t];
  for (int t = threadIdx.x; t < count * Ct; t += threads) {
    const int j = t / Ct, c = t % Ct;
    channels[j][c] = c < width ? right[(start + j) * stride + c] : T(0);
  }
  __syncthreads();
}

// One thread per output row: the row stays in registers while column tiles
// and their payload rows are staged in shared memory. Each column tile is
// summed plainly and merged into the running total with compensation.
// Blocks along y split the columns into chunks; partial is
// [chunks, batch, n, width]. Plain kernels unroll full tiles by pair group.
template <typename T, Feature F, int MaxD, int Ct, int O = runtime_outer,
          Path P = Path::General>
C10_LAUNCH_BOUNDS_1(threads)
__global__ void matmul_kernel(const T* __restrict__ rows,
                              const T* __restrict__ columns, Geometry g,
                              Channels<T> h, Context<T> c,
                              T* __restrict__ partial) {
  constexpr int tile = column_tile<MaxD>();
  constexpr bool fast = P != Path::General;
  // Padded features skip the k < d test; pow costs more than the test.
  constexpr bool pad = fast && F != Feature::Power;
  resolve(c);
  const auto outer = fixed<O>(c.outer);
  __shared__ T staged[tile][MaxD];
  __shared__ T payload[tile][Ct];
  __shared__ Steps<T> steps;
  if constexpr (P == Path::Numbers)
    share(c, steps);
  const int64_t b = g.batch_offset + blockIdx.z, part = blockIdx.y;
  const int64_t i = int64_t(blockIdx.x) * threads + threadIdx.x;
  rows += b * g.n * g.d;
  columns += b * g.m * g.d;
  const T* right = h.right + b * h.right_stride + h.channel;
  T x[MaxD], total[Ct], error[Ct], operand[max_chain];
#pragma unroll
  for (int k = 0; k < MaxD; ++k)
    x[k] = i < g.n && k < g.d ? rows[i * g.d + k] : T(0);
  if constexpr (!fast)
    row_operands(c, b, i, i < g.n, operand);
#pragma unroll
  for (int k = 0; k < Ct; ++k)
    total[k] = error[k] = T(0);
  // The pair's value before payload weighting.
  auto pair = [&](int64_t j, const T* y) {
    const T s = reduce_features<T, F, MaxD, pad>(x, y, g.d, c.power);
    T value = pairs::outer(outer, s);
    if constexpr (P == Path::Plain)
      return value;
    if constexpr (P == Path::Numbers)
      value = run_steps(steps, value);
    else {
      column_operands(c, b, j, operand);
      value = pairs::apply(c.chain, operand, value);
    }
    if (c.shift != nullptr) {
      const int64_t k = state_index(c, b, i, j);
      value = ::exp(value - c.shift[k]) / c.mass[k];
    }
    return value;
  };
  const int64_t begin = part * g.chunk,
                end = begin + g.chunk < g.m ? begin + g.chunk : g.m;
  for (int64_t start = begin; start < end; start += tile) {
    const int count = end - start < tile ? int(end - start) : tile;
    stage<T, MaxD, Ct, fast>(staged, payload, columns, right, start, count,
                             g.d, h.channels, h.width);
    if (i >= g.n)
      continue;
    // An excluded pair weighs 0, as native's zeroed probability, so a
    // non-finite payload still propagates.
    const int64_t skip = c.exclusion ? i - start : -1;
    T term[Ct];
#pragma unroll
    for (int k = 0; k < Ct; ++k)
      term[k] = T(0);
    if (P == Path::Plain && count == tile) {
      for (int group = 0; group < tile; group += pair_group) {
#pragma unroll
        for (int jj = 0; jj < pair_group; ++jj) {
          const int j = group + jj;
          const T value = j == skip ? T(0) : pair(start + j, staged[j]);
#pragma unroll
          for (int k = 0; k < Ct; ++k)
            term[k] += value * payload[j][k];
        }
#pragma unroll
        for (int k = 0; k < Ct; ++k) {
          compensate(total[k], error[k], term[k]);
          term[k] = T(0);
        }
      }
      continue;
    }
    for (int j = 0; j < count; ++j) {
      const T value = j == skip ? T(0) : pair(start + j, staged[j]);
#pragma unroll
      for (int k = 0; k < Ct; ++k)
        term[k] += value * payload[j][k];
      if ((j + 1) % pair_group == 0 || j + 1 == count)
#pragma unroll
        for (int k = 0; k < Ct; ++k) {
          compensate(total[k], error[k], term[k]);
          term[k] = T(0);
        }
    }
  }
  if (i < g.n)
#pragma unroll
    for (int k = 0; k < Ct; ++k)
      if (k < h.width)
        partial[((part * g.batch + b) * g.n + i) * h.width + k] =
            total[k] + error[k];
}

// Row gradients sum_j w_ij * d value(rows_i, columns_j) / d rows_i, laid out
// like matmul_kernel with d outputs per row; w_ij = left_i . right_j, then
// the optional softmax weighting p_ij * (w_ij - center), with the center
// subtracted in the first channel block only (blocks add up). exclusion
// zeroes w_ii, as the tile path zeroes that cotangent, so non-finite partials
// still propagate. Abs and Power terms are zero at delta == 0 whatever the
// coefficient (native PowerTerm); Square partials are doubled at the end.
// Fast kernels (Path::Numbers) take chains without Exp, whose slope is the
// constant c.scale.
template <typename T, Feature F, int MaxD, int Ct, int O = runtime_outer,
          Path P = Path::General>
C10_LAUNCH_BOUNDS_1(threads)
__global__ void gradient_kernel(const T* __restrict__ rows,
                                const T* __restrict__ columns, Geometry g,
                                Channels<T> h, Context<T> c,
                                T* __restrict__ partial) {
  constexpr int tile = column_tile<MaxD>();
  constexpr bool fast = P != Path::General;
  // Padded features skip the k < d test; pow costs more than the test.
  constexpr bool pad = fast && F != Feature::Power;
  resolve(c);
  const auto outer = fixed<O>(c.outer);
  __shared__ T staged[tile][MaxD];
  __shared__ T weights[tile][Ct];
  __shared__ Steps<T> steps;
  if constexpr (fast)
    share(c, steps);
  const int64_t b = g.batch_offset + blockIdx.z, part = blockIdx.y;
  const int64_t i = int64_t(blockIdx.x) * threads + threadIdx.x;
  rows += b * g.n * g.d;
  columns += b * g.m * g.d;
  const T* left = h.left + b * h.left_stride + h.channel;
  const T* right = h.right + b * h.right_stride + h.channel;
  T x[MaxD], own[Ct], total[MaxD], error[MaxD], operand[max_chain];
#pragma unroll
  for (int k = 0; k < MaxD; ++k) {
    x[k] = i < g.n && k < g.d ? rows[i * g.d + k] : T(0);
    total[k] = error[k] = T(0);
  }
  if constexpr (!fast)
    row_operands(c, b, i, i < g.n, operand);
#pragma unroll
  for (int k = 0; k < Ct; ++k)
    own[k] = i < g.n && k < h.width ? left[i * h.channels + k] : T(0);
  const int64_t begin = part * g.chunk,
                end = begin + g.chunk < g.m ? begin + g.chunk : g.m;
  for (int64_t start = begin; start < end; start += tile) {
    const int count = end - start < tile ? int(end - start) : tile;
    stage<T, MaxD, Ct, fast>(staged, weights, columns, right, start, count,
                             g.d, h.channels, h.width);
    if (i >= g.n)
      continue;
    T term[MaxD];
#pragma unroll
    for (int k = 0; k < MaxD; ++k)
      term[k] = T(0);
    for (int j = 0; j < count; ++j) {
      const T s =
          reduce_features<T, F, MaxD, pad>(x, staged[j], g.d, c.power);
      const T value = pairs::outer(outer, s);
      T slope = c.scale, score = value;
      if constexpr (fast) {
        if (c.shift != nullptr)
          score = run_steps(steps, value);
      } else {
        column_operands(c, b, start + j, operand);
        if (c.dynamic)
          score = pairs::apply(c.chain, operand, value, slope);
        else if (c.shift != nullptr)
          score = pairs::apply(c.chain, operand, value);
      }
      T w = T(0);
#pragma unroll
      for (int k = 0; k < Ct; ++k)
        w += own[k] * weights[j][k];
      if (c.shift != nullptr) {
        const int64_t k = state_index(c, b, i, start + j);
        const T p = ::exp(score - c.shift[k]) / c.mass[k];
        w = p * (c.center != nullptr && h.channel == 0 ? w - c.center[k] : w);
      }
      if (c.exclusion && start + j == i)
        w = T(0);
      const T coefficient = w * slope * pairs::douter(outer, s, value);
#pragma unroll
      for (int k = 0; k < MaxD; ++k)
        if (pad || k < g.d) {
          const T d = pairs::dstep<F>(x[k], staged[j][k], c.power);
          if constexpr (F == Feature::Square || F == Feature::Product)
            term[k] += coefficient * d;
          else
            term[k] += x[k] - staged[j][k] != T(0) ? coefficient * d : T(0);
        }
      if ((j + 1) % pair_group == 0 || j + 1 == count)
#pragma unroll
        for (int k = 0; k < MaxD; ++k) {
          compensate(total[k], error[k], term[k]);
          term[k] = T(0);
        }
    }
  }
  if (i < g.n)
#pragma unroll
    for (int k = 0; k < MaxD; ++k)
      if (k < g.d)
        partial[((part * g.batch + b) * g.n + i) * g.d + k] =
            F == Feature::Square ? T(2) * (total[k] + error[k])
                                 : total[k] + error[k];
}

// Gradients of the chain's row values, sum_j w_ij * d score_ij / d operand,
// with w_ij as in gradient_kernel; partial is [chunks, batch, n, chain size]
// (zero for ops without a row operand).
template <typename T, Feature F, int MaxD, int Ct>
C10_LAUNCH_BOUNDS_1(threads)
__global__ void value_gradient_kernel(const T* __restrict__ rows,
                                      const T* __restrict__ columns,
                                      Geometry g, Channels<T> h, Context<T> c,
                                      T* __restrict__ partial) {
  constexpr int tile = column_tile<MaxD>();
  resolve(c);
  __shared__ T staged[tile][MaxD];
  __shared__ T weights[tile][Ct];
  const int64_t b = g.batch_offset + blockIdx.z, part = blockIdx.y;
  const int64_t i = int64_t(blockIdx.x) * threads + threadIdx.x;
  rows += b * g.n * g.d;
  columns += b * g.m * g.d;
  const T* left = h.left + b * h.left_stride + h.channel;
  const T* right = h.right + b * h.right_stride + h.channel;
  T x[MaxD], own[Ct], operand[max_chain], total[max_chain], error[max_chain];
#pragma unroll
  for (int k = 0; k < MaxD; ++k)
    x[k] = i < g.n && k < g.d ? rows[i * g.d + k] : T(0);
#pragma unroll
  for (int k = 0; k < Ct; ++k)
    own[k] = i < g.n && k < h.width ? left[i * h.channels + k] : T(0);
#pragma unroll
  for (int k = 0; k < max_chain; ++k)
    total[k] = error[k] = T(0);
  row_operands(c, b, i, i < g.n, operand);
  const int64_t begin = part * g.chunk,
                end = begin + g.chunk < g.m ? begin + g.chunk : g.m;
  for (int64_t start = begin; start < end; start += tile) {
    const int count = end - start < tile ? int(end - start) : tile;
    stage<T, MaxD, Ct>(staged, weights, columns, right, start, count, g.d,
                       h.channels, h.width);
    if (i >= g.n)
      continue;
    T term[max_chain];
#pragma unroll
    for (int k = 0; k < max_chain; ++k)
      term[k] = T(0);
    for (int j = 0; j < count; ++j) {
      const T s = reduce_features<T, F, MaxD>(x, staged[j], g.d, c.power);
      const T value = pairs::outer(c.outer, s);
      column_operands(c, b, start + j, operand);
      T w = T(0);
#pragma unroll
      for (int k = 0; k < Ct; ++k)
        w += own[k] * weights[j][k];
      if (c.shift != nullptr) {
        const int64_t k = state_index(c, b, i, start + j);
        const T p =
            ::exp(pairs::apply(c.chain, operand, value) - c.shift[k]) /
            c.mass[k];
        w = p * (c.center != nullptr && h.channel == 0 ? w - c.center[k] : w);
      }
      if (c.exclusion && start + j == i)
        w = T(0);
      T slope[max_chain];
      pairs::operand_slopes(c.chain, operand, value, slope);
#pragma unroll
      for (int k = 0; k < max_chain; ++k)
        if (k < c.chain.size && c.chain.axis[k] == 1)
          term[k] += w * slope[k];
      if ((j + 1) % pair_group == 0 || j + 1 == count)
#pragma unroll
        for (int k = 0; k < max_chain; ++k) {
          compensate(total[k], error[k], term[k]);
          term[k] = T(0);
        }
    }
  }
  if (i < g.n)
    for (int k = 0; k < c.chain.size; ++k)
      partial[((part * g.batch + b) * g.n + i) * c.chain.size + k] =
          total[k] + error[k];
}

// Row (maximum, mass, NaN seen) of the scores, partial [chunks, batch, n, 3]
// in float64. Each column tile keeps an online maximum and rescaled mass in
// the score dtype (one exp per pair, Neumaier-compensated), merged into
// float64 per tile.
// Excluded and -inf scores carry no mass; equal infinite maxima add mass 1.
template <typename T, Feature F, int MaxD, int O = runtime_outer,
          Path P = Path::General>
C10_LAUNCH_BOUNDS_1(threads)
__global__ void state_kernel(const T* __restrict__ rows,
                             const T* __restrict__ columns, Geometry g,
                             Context<T> c, double* __restrict__ partial) {
  constexpr int tile = column_tile<MaxD>();
  constexpr bool fast = P != Path::General;
  // Padded features skip the k < d test; pow costs more than the test.
  constexpr bool pad = fast && F != Feature::Power;
  resolve(c);
  const auto outer = fixed<O>(c.outer);
  __shared__ T staged[tile][MaxD];
  __shared__ Steps<T> steps;
  if constexpr (fast)
    share(c, steps);
  const int64_t b = g.batch_offset + blockIdx.z, part = blockIdx.y;
  const int64_t i = int64_t(blockIdx.x) * threads + threadIdx.x;
  rows += b * g.n * g.d;
  columns += b * g.m * g.d;
  T x[MaxD], operand[max_chain];
#pragma unroll
  for (int k = 0; k < MaxD; ++k)
    x[k] = i < g.n && k < g.d ? rows[i * g.d + k] : T(0);
  if constexpr (!fast)
    row_operands(c, b, i, i < g.n, operand);
  const T low = -std::numeric_limits<T>::infinity();
  double maximum = -std::numeric_limits<double>::infinity(), mass = 0;
  bool nan = false;
  const int64_t begin = part * g.chunk,
                end = begin + g.chunk < g.m ? begin + g.chunk : g.m;
  for (int64_t start = begin; start < end; start += tile) {
    const int count = end - start < tile ? int(end - start) : tile;
    __syncthreads();
    if constexpr (fast)
      for (int t = threadIdx.x; t < count * MaxD; t += threads) {
        const int j = t / MaxD, k = t % MaxD;
        staged[j][k] = k < g.d ? columns[(start + j) * g.d + k] : T(0);
      }
    else
      for (int t = threadIdx.x; t < count * g.d; t += threads)
        staged[t / g.d][t % g.d] = columns[start * g.d + t];
    __syncthreads();
    if (i >= g.n)
      continue;
    T top = low, sum = T(0), carry = T(0);
    for (int j = 0; j < count; ++j) {
      if (c.exclusion && start + j == i)
        continue;
      const T s =
          reduce_features<T, F, MaxD, pad>(x, staged[j], g.d, c.power);
      T score = pairs::outer(outer, s);
      if constexpr (fast)
        score = run_steps(steps, score);
      else {
        column_operands(c, b, start + j, operand);
        score = pairs::apply(c.chain, operand, score);
      }
      if (::isnan(score)) {
        nan = true;
        continue;
      }
      if (score == low)
        continue;
      if (score > top) {
        const T scale = ::exp(top - score);
        sum *= scale;
        carry *= scale;
        top = score;
        compensate(sum, carry, T(1));
      } else
        compensate(sum, carry, score == top ? T(1) : ::exp(score - top));
    }
    if (sum > T(0)) {
      const double tile_max = double(top),
                   next = tile_max > maximum ? tile_max : maximum;
      mass = mass * (maximum == next ? 1. : ::exp(maximum - next)) +
             (double(sum) + double(carry)) *
                 (tile_max == next ? 1. : ::exp(tile_max - next));
      maximum = next;
    }
  }
  if (i < g.n) {
    double* out = partial + ((part * g.batch + b) * g.n + i) * 3;
    out[0] = maximum;
    out[1] = mass;
    out[2] = nan ? 1. : 0.;
  }
}

// Splits a float64 value into float hi + lo (lo is 0 for infinite values).
template <typename T>
__device__ __forceinline__ void split(double value, T& hi, T& lo) {
  hi = T(value);
  lo = ::isfinite(hi) ? T(value - double(hi)) : T(0);
}

// state_kernel for the scores scale * value + bias[j], reported relative to
// reference[i] (the result adds it back). Sinkhorn potentials make the bias
// and the cost term large while their sum stays near the row maximum, so the
// gap to the reference is formed with error-free float transforms of the
// float64 operands: the large parts cancel in one correctly rounded addition
// and the gap keeps float accuracy relative to itself, without per-pair
// float64 work. bias_stride is 0 when the bias is shared across the batch.
template <typename T, Feature F, int MaxD>
C10_LAUNCH_BOUNDS_1(threads)
__global__ void biased_state_kernel(const T* __restrict__ rows,
                                    const T* __restrict__ columns, Geometry g,
                                    Context<T> c,
                                    const double* __restrict__ scale,
                                    const double* __restrict__ bias,
                                    int64_t bias_stride,
                                    const double* __restrict__ reference,
                                    double* __restrict__ partial) {
  constexpr int tile = column_tile<MaxD>();
  resolve(c);
  __shared__ T staged[tile][MaxD];
  __shared__ T bias_hi[tile], bias_lo[tile];
  const int64_t b = g.batch_offset + blockIdx.z, part = blockIdx.y;
  const int64_t i = int64_t(blockIdx.x) * threads + threadIdx.x;
  rows += b * g.n * g.d;
  columns += b * g.m * g.d;
  bias += b * bias_stride;
  T factor_hi, factor_lo, reference_hi = T(0), reference_lo = T(0);
  split(*scale, factor_hi, factor_lo);
  if (i < g.n)
    split(reference[b * g.n + i], reference_hi, reference_lo);
  T x[MaxD], operand[max_chain];
#pragma unroll
  for (int k = 0; k < MaxD; ++k)
    x[k] = i < g.n && k < g.d ? rows[i * g.d + k] : T(0);
  row_operands(c, b, i, i < g.n, operand);
  const T low = -std::numeric_limits<T>::infinity();
  double maximum = -std::numeric_limits<double>::infinity(), mass = 0;
  bool nan = false;
  const int64_t begin = part * g.chunk,
                end = begin + g.chunk < g.m ? begin + g.chunk : g.m;
  for (int64_t start = begin; start < end; start += tile) {
    const int count = end - start < tile ? int(end - start) : tile;
    __syncthreads();
    for (int t = threadIdx.x; t < count * g.d; t += threads)
      staged[t / g.d][t % g.d] = columns[start * g.d + t];
    for (int t = threadIdx.x; t < count; t += threads)
      split(bias[start + t], bias_hi[t], bias_lo[t]);
    __syncthreads();
    if (i >= g.n)
      continue;
    T top = low, sum = T(0), carry = T(0);
    for (int j = 0; j < count; ++j) {
      const T bh = bias_hi[j];
      if (bh == low)
        continue;
      const T s = reduce_features<T, F, MaxD>(x, staged[j], g.d, c.power);
      column_operands(c, b, start + j, operand);
      const T value =
          pairs::apply(c.chain, operand, pairs::outer(c.outer, s));
      const T product = value * factor_hi;
      T score;
      if (!::isfinite(product) || !::isfinite(bh))
        score = (bh - reference_hi) + product;
      else {
        // two_sum(bh, -reference_hi) and two_product(value, factor_hi).
        const T head = bh - reference_hi, back = head - bh;
        const T tail = (bh - (head - back)) + (-reference_hi - back);
        const T error = std::fma(value, factor_hi, -product);
        // add_rn: a contracted fma would count the product error twice.
        score = pairs::add_rn(head, product) +
                (tail + (bias_lo[j] - reference_lo) + error +
                 value * factor_lo);
      }
      if (::isnan(score)) {
        nan = true;
        continue;
      }
      if (score == low)
        continue;
      if (score > top) {
        const T rescale = ::exp(top - score);
        sum *= rescale;
        carry *= rescale;
        top = score;
        compensate(sum, carry, T(1));
      } else
        compensate(sum, carry, score == top ? T(1) : ::exp(score - top));
    }
    if (sum > T(0)) {
      const double tile_max = double(top),
                   next = tile_max > maximum ? tile_max : maximum;
      mass = mass * (maximum == next ? 1. : ::exp(maximum - next)) +
             (double(sum) + double(carry)) *
                 (tile_max == next ? 1. : ::exp(tile_max - next));
      maximum = next;
    }
  }
  if (i < g.n) {
    double* out = partial + ((part * g.batch + b) * g.n + i) * 3;
    out[0] = maximum;
    out[1] = mass;
    out[2] = nan ? 1. : 0.;
  }
}

// The tile path's selection order (select() in streaming_reductions.cpp):
// category (non-NaN 0 and NaN 1, swapped by nan_first), then key
// (largest ? -v : v, NaN keyed 0, -0 == 0), then column. It is a strict total
// order, so chunk lists merge to the native values and indices in any
// grouping.
template <typename T>
struct Order {
  bool largest, nan_first;
  __device__ __forceinline__ bool before(T v, T w) const {
    const bool a = ::isnan(v) != nan_first, b = ::isnan(w) != nan_first;
    if (a != b)
      return b;
    const T x = ::isnan(v) ? T(0) : largest ? -v : v,
            y = ::isnan(w) ? T(0) : largest ? -w : w;
    return x < y;
  }
};

// Each thread keeps its row's first k entries of a column chunk, sorted, in
// [chunks, B, n, k]; unfilled slots get column -1. Columns arrive in
// increasing order, so an entry tied with listed ones goes after them. One
// list size serves every k (a k = 1 instantiation cost binary size).
template <typename T, Feature F, int MaxD>
C10_LAUNCH_BOUNDS_1(threads)
__global__ void topk_kernel(const T* __restrict__ rows,
                            const T* __restrict__ columns, Geometry g,
                            Context<T> c, Order<T> o, int k,
                            T* __restrict__ values,
                            int64_t* __restrict__ indices) {
  constexpr int tile = column_tile<MaxD>();
  resolve(c);
  __shared__ T staged[tile][MaxD];
  const int64_t b = g.batch_offset + blockIdx.z, part = blockIdx.y;
  const int64_t i = int64_t(blockIdx.x) * threads + threadIdx.x;
  rows += b * g.n * g.d;
  columns += b * g.m * g.d;
  T x[MaxD], operand[max_chain];
#pragma unroll
  for (int t = 0; t < MaxD; ++t)
    x[t] = i < g.n && t < g.d ? rows[i * g.d + t] : T(0);
  row_operands(c, b, i, i < g.n, operand);
  T best[max_topk];
  int64_t column[max_topk];
  int size = 0;
  const int64_t begin = part * g.chunk,
                end = begin + g.chunk < g.m ? begin + g.chunk : g.m;
  for (int64_t start = begin; start < end; start += tile) {
    const int count = end - start < tile ? int(end - start) : tile;
    __syncthreads();
    for (int t = threadIdx.x; t < count * g.d; t += threads)
      staged[t / g.d][t % g.d] = columns[start * g.d + t];
    __syncthreads();
    if (i >= g.n)
      continue;
    for (int j = 0; j < count; ++j) {
      if (c.exclusion && start + j == i)
        continue;
      const T s = reduce_features<T, F, MaxD>(x, staged[j], g.d, c.power);
      column_operands(c, b, start + j, operand);
      const T value =
          pairs::apply(c.chain, operand, pairs::outer(c.outer, s));
      if (size == k && !o.before(value, best[k - 1]))
        continue;
      int slot = size < k ? size++ : k - 1;
      for (; slot > 0 && o.before(value, best[slot - 1]); --slot) {
        best[slot] = best[slot - 1];
        column[slot] = column[slot - 1];
      }
      best[slot] = value;
      column[slot] = start + j;
    }
  }
  if (i < g.n) {
    const int64_t base = ((part * g.batch + b) * g.n + i) * k;
    for (int t = 0; t < k; ++t) {
      values[base + t] = t < size ? best[t] : T(0);
      indices[base + t] = t < size ? column[t] : -1;
    }
  }
}

// Columns per chunk so that rows x chunks fill the device: split only while
// the rows alone underfill it, keeping at least 8 column tiles per chunk.
Geometry geometry(const Plan& p, int tile) {
  Geometry g{p.rows.size(0), 0, p.rows.size(1), p.columns.size(1), 0,
             int(p.rows.size(2))};
  const int64_t row_blocks = (g.n + threads - 1) / threads;
  const int64_t target =
      4 * int64_t(at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
  const int64_t tiles = (g.m + tile - 1) / tile;
  const int64_t chunks = std::clamp<int64_t>(
      (target + row_blocks * g.batch - 1) / (row_blocks * g.batch), 1,
      std::max<int64_t>(1, tiles / 8));
  g.chunk = (tiles + chunks - 1) / chunks * tile;
  return g;
}

// Runs launch(grid, g) over (row blocks, column chunks, batch slices).
template <typename Launch>
void for_grid(Geometry g, Launch&& launch) {
  const int64_t row_blocks = (g.n + threads - 1) / threads,
                chunks = (g.m + g.chunk - 1) / g.chunk;
  for (int64_t offset = 0; offset < g.batch; offset += max_grid_z) {
    g.batch_offset = offset;
    launch(dim3(unsigned(row_blocks), unsigned(chunks),
                unsigned(std::min(max_grid_z, g.batch - offset))),
           g);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

// Kernel selection of run(): products, feature gradients or row value
// gradients.
enum class Mode { Matmul, Gradient, Values };

template <typename T, Feature F, int MaxD, int Ct, Mode M, int O, Path P>
void run_channels(const Plan& p, Channels<T> h, const Context<T>& c,
                  Tensor& result) {
  constexpr bool Gradient = M != Mode::Matmul;
  const auto geometry_ = geometry(p, column_tile<MaxD>());
  const int64_t chunks = (geometry_.m + geometry_.chunk - 1) / geometry_.chunk;
  const int64_t width = M == Mode::Gradient ? geometry_.d
                        : M == Mode::Values ? p.chain_size
                                            : h.width;
  auto partial =
      at::empty({chunks, geometry_.batch, geometry_.n, width}, p.rows.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const T *rows = p.rows.const_data_ptr<T>(),
          *columns = p.columns.const_data_ptr<T>();
  T* out = partial.template mutable_data_ptr<T>();
  for_grid(geometry_, [&](dim3 grid, Geometry g) {
    if constexpr (M == Mode::Gradient)
      gradient_kernel<T, F, MaxD, Ct, O, P>
          <<<grid, threads, 0, stream>>>(rows, columns, g, h, c, out);
    else if constexpr (M == Mode::Values)
      value_gradient_kernel<T, F, MaxD, Ct>
          <<<grid, threads, 0, stream>>>(rows, columns, g, h, c, out);
    else
      matmul_kernel<T, F, MaxD, Ct, O, P>
          <<<grid, threads, 0, stream>>>(rows, columns, g, h, c, out);
  });
  auto total = chunks == 1 ? partial.select(0, 0) : partial.sum(0);
  if (Gradient)
    result.add_(total);
  else
    result.narrow(-1, h.channel, h.width).copy_(total);
}

// Channel blocks beyond 16 re-evaluate the pairs; gradients add up across
// blocks because they are linear in the channel weights.
template <typename T, Feature F, int MaxD, Mode M, int O = runtime_outer,
          Path P = Path::General>
void run(const Plan& p, Channels<T> h, const Context<T>& c, Tensor& result) {
  if (h.channels == 1) {
    h.channel = 0;
    h.width = 1;
    run_channels<T, F, MaxD, 1, M, O, P>(p, h, c, result);
    return;
  }
  for (int64_t channel = 0; channel < h.channels; channel += 16) {
    h.channel = channel;
    h.width = int(std::min<int64_t>(16, h.channels - channel));
    run_channels<T, F, MaxD, 16, M, O, P>(p, h, c, result);
  }
}

// Calls visit(T* tag, feature, max_features) for the plan's dtype (float32
// or float64), feature and register bucket; Limit caps the bucket (gradients
// use 16). AllowProduct is false for selections, which keep the tile path
// for dot products (see plan()), so their kernels are not instantiated.
template <int Limit, bool AllowMax, bool AllowProduct = true, typename Visit>
void visit_plan(const Plan& p, Visit&& visit) {
  const auto d = p.rows.size(2);
  auto bucket = [&](auto scalar, auto feature) {
    if (d <= 4)
      visit(scalar, feature, std::integral_constant<int, 4>());
    else if (d <= 16 || Limit <= 16)
      visit(scalar, feature, std::integral_constant<int, 16>());
    else if constexpr (Limit > 16)
      visit(scalar, feature, std::integral_constant<int, 64>());
  };
  auto features = [&](auto tag) {
    switch (Feature(p.feature)) {
      case Feature::Square:
        bucket(tag, std::integral_constant<Feature, Feature::Square>());
        break;
      case Feature::Abs:
        bucket(tag, std::integral_constant<Feature, Feature::Abs>());
        break;
      case Feature::Power:
        bucket(tag, std::integral_constant<Feature, Feature::Power>());
        break;
      case Feature::Max:
        if constexpr (AllowMax)
          bucket(tag, std::integral_constant<Feature, Feature::Max>());
        else
          TORCH_CHECK(false, "Chebyshev has no fused gradient");
        break;
      case Feature::Product:
        if constexpr (AllowProduct)
          bucket(tag, std::integral_constant<Feature, Feature::Product>());
        else
          TORCH_CHECK(false, "Dot products have no fused selection");
        break;
    }
  };
  if (p.rows.scalar_type() == at::kFloat)
    features(static_cast<float*>(nullptr));
  else if (p.rows.scalar_type() == at::kDouble)
    features(static_cast<double*>(nullptr));
  else
    TORCH_CHECK(false, "Fused reducers support float32 and float64 only");
}

// The fast kernels' path for a plan (see Path): General unless the plan is
// float32 with at most 16 features and its chain applies numbers only
// (gradients: no Exp, whose slope varies per pair). Plain needs no chain and
// no softmax weighting.
Path path(const Plan& p, bool weighted, bool gradient) {
  if (p.rows.scalar_type() != at::kFloat || p.rows.size(2) > 16)
    return Path::General;
  for (int k = 0; k < p.chain_size; ++k) {
    const auto op = ChainOp(p.chain_op[k]);
    if (p.chain_axis[k] != 0 || op == ChainOp::Keep || op == ChainOp::Drop ||
        (gradient && op == ChainOp::Exp))
      return Path::General;
  }
  return p.chain_size == 0 && !weighted && !gradient ? Path::Plain
                                                     : Path::Numbers;
}

template <Feature V>
using feature_constant = std::integral_constant<Feature, V>;
template <Outer V>
using outer_constant = std::integral_constant<int, int(V)>;

// Calls visit(feature, outer, max_features) for a float32 plan of at most 16
// features, over the (feature, outer) pairs plan() produces; AllowMax is
// false for gradients.
template <bool AllowMax, typename Visit>
void visit_fast(const Plan& p, Visit&& visit) {
  const auto d = p.rows.size(2);
  bool found = false;
  auto pair = [&](auto feature, auto outer) {
    if (found || Feature(p.feature) != decltype(feature)::value ||
        p.outer != decltype(outer)::value)
      return;
    found = true;
    if (d <= 4)
      visit(feature, outer, std::integral_constant<int, 4>());
    else if (d <= 8)
      visit(feature, outer, std::integral_constant<int, 8>());
    else
      visit(feature, outer, std::integral_constant<int, 16>());
  };
  using F = Feature;
  using O = Outer;
  pair(feature_constant<F::Square>(), outer_constant<O::Identity>());
  pair(feature_constant<F::Square>(), outer_constant<O::ScaledExp>());
  pair(feature_constant<F::Square>(), outer_constant<O::Matern05>());
  pair(feature_constant<F::Square>(), outer_constant<O::Matern15>());
  pair(feature_constant<F::Square>(), outer_constant<O::Matern25>());
  pair(feature_constant<F::Square>(), outer_constant<O::RationalQuadratic>());
  pair(feature_constant<F::Abs>(), outer_constant<O::Identity>());
  pair(feature_constant<F::Abs>(), outer_constant<O::ScaledExp>());
  pair(feature_constant<F::Power>(), outer_constant<O::Root>());
  pair(feature_constant<F::Product>(), outer_constant<O::Identity>());
  pair(feature_constant<F::Product>(), outer_constant<O::Polynomial>());
  pair(feature_constant<F::Product>(), outer_constant<O::Sigmoid>());
  if constexpr (AllowMax)
    pair(feature_constant<F::Max>(), outer_constant<O::Identity>());
  TORCH_CHECK(found, "Unexpected fused reducer plan");
}

// Per-sample channel operands as [samples, C] (shared, stride 0) or
// [B, samples, C] contiguous.
std::pair<Tensor, int64_t> batched_operand(const Tensor& raw, const Plan& p,
                                           int64_t samples) {
  const auto channels = raw.size(-1), batch = p.rows.size(0);
  int64_t batches = 1;
  for (int64_t axis = 0; axis + 2 < raw.dim(); ++axis)
    batches *= raw.size(axis);
  if (batches == 1)
    return {raw.reshape({samples, channels}).contiguous(), 0};
  auto shape = p.batch;
  shape.insert(shape.end(), {samples, channels});
  return {raw.expand(shape).reshape({batch, samples, channels}).contiguous(),
          samples * channels};
}

// Probability state as contiguous [B, samples] in the input dtype.
Probability prepared(const Probability& w, const Plan& p) {
  if (!w.shift.defined())
    return w;
  const auto batch = p.rows.size(0);
  auto flat = [&](const Tensor& value) {
    return value.defined() ? value.to(p.rows.scalar_type())
                                 .reshape({batch, -1})
                                 .contiguous()
                           : value;
  };
  return {flat(w.shift), flat(w.mass), flat(w.center), w.on_columns};
}

// Merges state partials [chunks, B, n, 3] into maximum, mass, NaN seen. A
// defined scale selects biased_state_kernel (bias [B, M] or [M], reference
// [B, n]); its maxima are then relative to the reference.
std::tuple<Tensor, Tensor, Tensor> row_state(
    const Plan& p, bool exclusion, const Tensor& scale = Tensor(),
    const Tensor& bias = Tensor(), const Tensor& reference = Tensor()) {
  const auto batch = p.rows.size(0), n = p.rows.size(1);
  auto options = p.rows.options().dtype(at::kDouble);
  if (batch == 0 || n == 0 || p.columns.size(1) == 0) {
    auto empty = at::zeros({batch, n}, options);
    return {empty - std::numeric_limits<double>::infinity(), empty,
            empty.to(at::kBool)};
  }
  Tensor partial;
  auto launch = [&](auto scalar, auto feature, auto features, auto outer,
                    auto route) {
    using T = std::remove_pointer_t<decltype(scalar)>;
    constexpr Feature F = decltype(feature)::value;
    constexpr int MaxD = decltype(features)::value;
    constexpr int O = decltype(outer)::value;
    constexpr Path P = decltype(route)::value;
    const auto geometry_ = geometry(p, column_tile<MaxD>());
    const int64_t chunks =
        (geometry_.m + geometry_.chunk - 1) / geometry_.chunk;
    partial = at::empty({chunks, batch, n, 3}, options);
    const auto c = context<T>(p, exclusion, {});
    auto stream = at::cuda::getCurrentCUDAStream();
    const T *rows = p.rows.const_data_ptr<T>(),
            *columns = p.columns.const_data_ptr<T>();
    double* out = partial.mutable_data_ptr<double>();
    for_grid(geometry_, [&](dim3 grid, Geometry g) {
      if constexpr (P != Path::General)
        state_kernel<T, F, MaxD, O, P>
            <<<grid, threads, 0, stream>>>(rows, columns, g, c, out);
      else if (scale.defined())
        biased_state_kernel<T, F, MaxD><<<grid, threads, 0, stream>>>(
            rows, columns, g, c, scale.const_data_ptr<double>(),
            bias.const_data_ptr<double>(), bias.dim() == 1 ? 0 : bias.size(-1),
            reference.const_data_ptr<double>(), out);
      else
        state_kernel<T, F, MaxD>
            <<<grid, threads, 0, stream>>>(rows, columns, g, c, out);
    });
  };
  if (!scale.defined() && path(p, false, false) != Path::General)
    visit_fast<true>(p, [&](auto feature, auto outer, auto features) {
      launch(static_cast<float*>(nullptr), feature, features, outer,
             std::integral_constant<Path, Path::Numbers>());
    });
  else
    visit_plan<64, true>(p, [&](auto scalar, auto feature, auto features) {
      launch(scalar, feature, features,
             std::integral_constant<int, runtime_outer>(),
             std::integral_constant<Path, Path::General>());
    });
  auto maxima = partial.select(-1, 0), masses = partial.select(-1, 1);
  auto maximum = std::get<0>(maxima.max(0));
  auto factor = at::where(maxima.eq(maximum), 1., (maxima - maximum).exp());
  auto mass = (masses * factor).sum(0);
  auto nan = partial.select(-1, 2).amax(0).gt(0);
  return {maximum, mass, nan};
}
}  // namespace

Tensor matmul_cuda(const Plan& p, const Tensor& raw_payload, bool exclusion,
                   const Probability& probability) {
  const at::cuda::CUDAGuard guard(p.rows.device());
  const auto batch = p.rows.size(0), n = p.rows.size(1),
             m = p.columns.size(1), channels = raw_payload.size(-1);
  TORCH_CHECK(p.rows.size(2) <= 64,
              "Fused reducers support at most 64 features");
  auto result = at::zeros({batch, n, channels}, p.rows.options());
  if (batch == 0 || n == 0 || m == 0 || channels == 0)
    return result;
  // A payload without batch dimensions is shared across the batch.
  Tensor payload;
  int64_t payload_stride;
  std::tie(payload, payload_stride) = batched_operand(raw_payload, p, m);
  auto w = prepared(probability, p);
  const auto route = path(p, w.shift.defined(), false);
  if (route != Path::General) {
    Channels<float> h{nullptr, payload.const_data_ptr<float>(), 0,
                      payload_stride, channels, 0, 0};
    const auto c = context<float>(p, exclusion, w);
    visit_fast<true>(p, [&](auto feature, auto outer, auto features) {
      constexpr Feature F = decltype(feature)::value;
      constexpr int O = decltype(outer)::value;
      constexpr int MaxD = decltype(features)::value;
      if (route == Path::Plain)
        run<float, F, MaxD, Mode::Matmul, O, Path::Plain>(p, h, c, result);
      else
        run<float, F, MaxD, Mode::Matmul, O, Path::Numbers>(p, h, c, result);
    });
    return result;
  }
  visit_plan<64, true>(p, [&](auto scalar, auto feature, auto features) {
    using T = std::remove_pointer_t<decltype(scalar)>;
    constexpr Feature F = decltype(feature)::value;
    constexpr int MaxD = decltype(features)::value;
    Channels<T> h{nullptr, payload.const_data_ptr<T>(), 0, payload_stride,
                  channels, 0, 0};
    run<T, F, MaxD, Mode::Matmul>(p, h, context<T>(p, exclusion, w), result);
  });
  return result;
}

Tensor gradient_cuda(const Plan& p, const Tensor& raw_left,
                     const Tensor& raw_right, bool exclusion,
                     const Probability& probability) {
  const at::cuda::CUDAGuard guard(p.rows.device());
  const auto batch = p.rows.size(0), n = p.rows.size(1),
             m = p.columns.size(1), d = p.rows.size(2),
             channels = raw_left.size(-1);
  TORCH_CHECK(p.differentiable && d <= max_gradient_features &&
                  Feature(p.feature) != Feature::Max,
              "Unsupported fused reducer gradient");
  TORCH_CHECK(channels == raw_right.size(-1),
              "Fused gradient channel counts differ");
  auto result = at::zeros({batch, n, d}, p.rows.options());
  if (batch == 0 || n == 0 || m == 0 || channels == 0)
    return result;
  Tensor left, right;
  int64_t left_stride, right_stride;
  std::tie(left, left_stride) = batched_operand(raw_left, p, n);
  std::tie(right, right_stride) = batched_operand(raw_right, p, m);
  auto w = prepared(probability, p);
  if (path(p, w.shift.defined(), true) != Path::General) {
    Channels<float> h{left.const_data_ptr<float>(),
                      right.const_data_ptr<float>(), left_stride, right_stride,
                      channels, 0, 0};
    const auto c = context<float>(p, exclusion, w);
    visit_fast<false>(p, [&](auto feature, auto outer, auto features) {
      run<float, decltype(feature)::value, decltype(features)::value,
          Mode::Gradient, decltype(outer)::value, Path::Numbers>(p, h, c,
                                                                 result);
    });
    return result;
  }
  visit_plan<16, false>(p, [&](auto scalar, auto feature, auto features) {
    using T = std::remove_pointer_t<decltype(scalar)>;
    constexpr Feature F = decltype(feature)::value;
    constexpr int MaxD = decltype(features)::value;
    Channels<T> h{left.const_data_ptr<T>(), right.const_data_ptr<T>(),
                  left_stride, right_stride, channels, 0, 0};
    run<T, F, MaxD, Mode::Gradient>(p, h, context<T>(p, exclusion, w),
                                    result);
  });
  return result;
}

namespace {
// [B, n, chain size] gradients of the plan's row values, laid out as
// gradient_cuda.
Tensor value_gradient_cuda(const Plan& p, const Tensor& raw_left,
                           const Tensor& raw_right, bool exclusion,
                           const Probability& probability) {
  const auto batch = p.rows.size(0), n = p.rows.size(1), m = p.columns.size(1);
  auto result = at::zeros({batch, n, p.chain_size}, p.rows.options());
  if (batch == 0 || n == 0 || m == 0 || raw_left.size(-1) == 0)
    return result;
  Tensor left, right;
  int64_t left_stride, right_stride;
  std::tie(left, left_stride) = batched_operand(raw_left, p, n);
  std::tie(right, right_stride) = batched_operand(raw_right, p, m);
  auto w = prepared(probability, p);
  visit_plan<16, false>(p, [&](auto scalar, auto feature, auto features) {
    using T = std::remove_pointer_t<decltype(scalar)>;
    constexpr Feature F = decltype(feature)::value;
    constexpr int MaxD = decltype(features)::value;
    Channels<T> h{left.const_data_ptr<T>(), right.const_data_ptr<T>(),
                  left_stride, right_stride, raw_left.size(-1), 0, 0};
    run<T, F, MaxD, Mode::Values>(p, h, context<T>(p, exclusion, w), result);
  });
  return result;
}
}  // namespace

void operand_gradients(const Plan& p, const Tensor& left, const Tensor& right,
                       bool exclusion, const Probability& rows,
                       const Probability& columns, std::vector<Tensor>& result,
                       at::TensorList v) {
  const at::cuda::CUDAGuard guard(p.rows.device());
  auto transposed = swapped(p);
  result[p.row_slot] =
      to_operand(p, gradient_cuda(p, left, right, exclusion, rows), true, v);
  result[p.column_slot] = to_operand(
      p, gradient_cuda(transposed, right, left, exclusion, columns), false, v);
  // Row values from the plan, column values as the rows of its transpose.
  auto values = [&](const Plan& side, const Tensor& own, const Tensor& other,
                    const Probability& weighting) {
    bool any = false;
    for (int k = 0; k < side.chain_size; ++k)
      any |= side.chain_axis[k] == 1 && side.chain_slot[k] >= 0;
    if (!any)
      return;
    auto gradient = value_gradient_cuda(side, own, other, exclusion, weighting);
    for (int k = 0; k < side.chain_size; ++k) {
      if (side.chain_axis[k] != 1 || side.chain_slot[k] < 0)
        continue;
      const auto& target = v[side.chain_slot[k]];
      auto value = gradient.select(2, k);
      if (side.row_map.defined())
        value = at::zeros({value.size(0), target.size(-1)}, value.options())
                    .index_add_(1, side.row_map, value);
      auto shape = side.batch;
      shape.push_back(value.size(1));
      auto& slot = result[side.chain_slot[k]];
      slot = (slot + at::sum_to(value.reshape(shape), target.sizes())
                         .to(target.scalar_type()))
                 .contiguous();
    }
  };
  values(p, left, right, rows);
  values(transposed, right, left, columns);
}

Tensor logsumexp_cuda(const Plan& p, bool exclusion) {
  const at::cuda::CUDAGuard guard(p.rows.device());
  Tensor maximum, mass, nan;
  std::tie(maximum, mass, nan) = row_state(p, exclusion);
  // torch.logsumexp: an infinite maximum is returned as is.
  auto value = at::where(maximum.isinf(), maximum, maximum + mass.log());
  return at::where(nan, std::numeric_limits<double>::quiet_NaN(), value);
}

Tensor biased_logsumexp_cuda(const Plan& p, const Tensor& scale,
                             const Tensor& raw_bias,
                             const Tensor& raw_reference) {
  const at::cuda::CUDAGuard guard(p.rows.device());
  const auto batch = p.rows.size(0), n = p.rows.size(1), m = p.columns.size(1);
  auto valid = [&](const Tensor& value, int64_t samples) {
    return value.scalar_type() == at::kDouble &&
           value.device() == p.rows.device() && value.dim() >= 1 &&
           value.size(-1) == samples &&
           (value.dim() == 1 || value.numel() == batch * samples);
  };
  TORCH_CHECK(scale.dim() == 0 && scale.scalar_type() == at::kDouble &&
                  scale.device() == p.rows.device() && valid(raw_bias, m) &&
                  valid(raw_reference, n),
              "Fused biased log-sum-exp needs a float64 scale, [B, M] bias "
              "and [B, N] reference on the plan device");
  auto bias = raw_bias.dim() == 1 ? raw_bias.contiguous()
                                  : raw_bias.reshape({batch, m}).contiguous();
  // Any finite reference is exact; non-finite guesses fall back to zero.
  auto reference = raw_reference.expand({batch, n});
  reference = at::where(reference.isfinite(), reference, 0.).contiguous();
  Tensor maximum, mass, nan;
  std::tie(maximum, mass, nan) =
      row_state(p, false, scale.contiguous(), bias, reference);
  auto value = at::where(maximum.isinf(), maximum, maximum + mass.log());
  return at::where(nan, std::numeric_limits<double>::quiet_NaN(),
                   reference + value);
}

Tensor softmax_state_cuda(const Plan& p, bool exclusion) {
  const at::cuda::CUDAGuard guard(p.rows.device());
  Tensor maximum, mass, nan;
  std::tie(maximum, mass, nan) = row_state(p, exclusion);
  // As softmax pass 1: NaN scores poison both entries, a +inf maximum the
  // mass (exp(inf - inf)).
  constexpr double not_a_number = std::numeric_limits<double>::quiet_NaN();
  mass = at::where(nan | maximum.eq(std::numeric_limits<double>::infinity()),
                   not_a_number, mass);
  maximum = at::where(nan, not_a_number, maximum);
  return at::stack({maximum, mass}, -1);
}

std::pair<Tensor, Tensor> topk_cuda(const Plan& p, int64_t k, bool largest,
                                    bool nan_first, bool exclusion) {
  const at::cuda::CUDAGuard guard(p.rows.device());
  const auto batch = p.rows.size(0), n = p.rows.size(1);
  TORCH_CHECK(k >= 1 && k <= max_topk && k <= p.columns.size(1) - exclusion,
              "Unsupported fused top-k size");
  if (batch == 0 || n == 0)
    return {at::empty({batch, n, k}, p.rows.options()),
            at::empty({batch, n, k}, p.rows.options().dtype(at::kLong))};
  Tensor values, indices;
  visit_plan<64, true, false>(p, [&](auto scalar, auto feature, auto features) {
    using T = std::remove_pointer_t<decltype(scalar)>;
    constexpr Feature F = decltype(feature)::value;
    constexpr int MaxD = decltype(features)::value;
    const auto geometry_ = geometry(p, column_tile<MaxD>());
    const int64_t chunks =
        (geometry_.m + geometry_.chunk - 1) / geometry_.chunk;
    values = at::empty({chunks, batch, n, k}, p.rows.options());
    indices = at::empty({chunks, batch, n, k},
                        p.rows.options().dtype(at::kLong));
    const auto c = context<T>(p, exclusion, {});
    const Order<T> o{largest, nan_first};
    auto stream = at::cuda::getCurrentCUDAStream();
    const T *rows = p.rows.const_data_ptr<T>(),
            *columns = p.columns.const_data_ptr<T>();
    T* value_out = values.mutable_data_ptr<T>();
    int64_t* index_out = indices.mutable_data_ptr<int64_t>();
    for_grid(geometry_, [&](dim3 grid, Geometry g) {
      topk_kernel<T, F, MaxD><<<grid, threads, 0, stream>>>(
          rows, columns, g, c, o, int(k), value_out, index_out);
    });
  });
  if (values.size(0) == 1)
    return {values.select(0, 0), indices.select(0, 0)};
  // Chunk lists concatenated in chunk order are in column order among equal
  // (category, key) pairs, so two stable sorts finish the merge as select()
  // does; unfilled slots sort last as excluded entries.
  auto flat = [&](const Tensor& value) {
    return value.permute({1, 2, 0, 3}).reshape({batch, n, -1});
  };
  values = flat(values);
  indices = flat(indices);
  auto nan = values.isnan();
  auto order = at::argsort(at::where(nan, 0., largest ? -values : values),
                           true, -1, false);
  values = values.gather(-1, order);
  indices = indices.gather(-1, order);
  auto category =
      at::where(indices.lt(0), 2,
                (nan_first ? ~nan.gather(-1, order) : nan.gather(-1, order))
                    .to(at::kChar));
  order = at::argsort(category, true, -1, false).narrow(-1, 0, k);
  return {values.gather(-1, order), indices.gather(-1, order)};
}
}  // namespace torchpairwise::ops::detail::fused
