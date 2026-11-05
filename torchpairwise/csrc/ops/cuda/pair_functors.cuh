#pragma once
// Pair functors for fused native reducers (no ATen dependency).
//
// A metric is value = outer(R_k feature(x_k, y_k)): R is a sum (or a max
// for Chebyshev) over features in k order, and feature is a function of the
// difference x_k - y_k or, for the dot-product kernels, the product x_k y_k.
// Each formula follows the native tile path's operation sequence
// (_ppminkowski's loop, then the named kernel's scalar steps), so values
// agree with native to rounding. Dot products are the exception: native
// forms them with a GEMM, whose summation order differs. Scalar
// steps that ATen runs as separate kernels use explicitly rounded operations,
// so nvcc cannot contract them into FMAs that native does not perform.
//
// Derivatives are explicit (`dstep`, `douter`) so backward needs no autograd
// replay; they keep native's subgradient conventions at zero.
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define TP_PAIR_INLINE __host__ __device__ __forceinline__
#else
#define TP_PAIR_INLINE inline
#endif

namespace torchpairwise::pairs {
enum class Feature : int { Square, Abs, Power, Max, Product };
enum class Outer : int {
  Identity,
  Sqrt,      // distance_sqrt: zero (and zero subgradient) at zero
  Root,      // s^a, zero at zero (masked Minkowski root)
  ScaledExp, // exp(s * a): RBF and Laplacian with a = -gamma
  Matern05,
  Matern15,
  Matern25,
  RationalQuadratic,
  Polynomial,  // (s * a + b)^c, ATen pow special case code
  Sigmoid      // tanh(s * a + b)
};

template <typename T>
TP_PAIR_INLINE T mul_rn(T a, T b) {
#if defined(__CUDA_ARCH__)
  if constexpr (sizeof(T) == 4)
    return __fmul_rn(a, b);
  else
    return __dmul_rn(a, b);
#else
  return a * b;
#endif
}
template <typename T>
TP_PAIR_INLINE T add_rn(T a, T b) {
#if defined(__CUDA_ARCH__)
  if constexpr (sizeof(T) == 4)
    return __fadd_rn(a, b);
  else
    return __dadd_rn(a, b);
#else
  return a + b;
#endif
}

// Outer scalar parameters, cast once to T on the host as ATen casts wrapped
// scalars. a/b/c/d meanings depend on Outer (see make_outer in the host gate).
template <typename T>
struct OuterParams {
  Outer kind;
  int code;  // Root: ATen pow special case (0 general, 1 sqrt, 2 square, 3 cube)
  T a, b, c, d;
};

// Expression chain on the metric value, applied node by node in the value
// dtype as the tile path evaluates them. Each op has one operand: a number,
// or a row or column value of the pair (Keep/Drop read it as the condition
// of where(condition, value, fill) and where(condition, fill, value), with
// the number as fill).
enum class ChainOp : int { Add, Sub, SubFrom, Mul, Neg, Exp, Div, Keep, Drop };
constexpr int max_chain = 8;
template <typename T>
struct Chain {
  int size;
  ChainOp op[max_chain];
  T constant[max_chain];
  int axis[max_chain];  // operand: 0 number, 1 row value, 2 column value
};
template <typename T>
TP_PAIR_INLINE T chain_step(ChainOp op, T value, T operand, T constant) {
  switch (op) {
    case ChainOp::Add:
      return add_rn(value, operand);
    case ChainOp::Sub:
      return add_rn(value, -operand);
    case ChainOp::SubFrom:
      return add_rn(operand, -value);
    case ChainOp::Mul:
      return mul_rn(value, operand);
    case ChainOp::Neg:
      return -value;
    case ChainOp::Exp:
      return ::exp(value);
    case ChainOp::Div:
      return value / operand;
    case ChainOp::Keep:
      return operand != T(0) ? value : constant;
    case ChainOp::Drop:
      return operand != T(0) ? constant : value;
  }
  return value;
}
// operand[k] is op k's operand for this pair (its number when axis is 0).
template <typename T>
TP_PAIR_INLINE T apply(const Chain<T>& chain, const T (&operand)[max_chain],
                       T value) {
#pragma unroll
  for (int k = 0; k < max_chain; ++k)
    if (k < chain.size)
      value = chain_step(chain.op[k], value, operand[k], chain.constant[k]);
  return value;
}
// apply, with slope = d result / d value in forward mode. Unselected where
// branches get an exact zero, as autograd's where backward.
template <typename T>
TP_PAIR_INLINE T apply(const Chain<T>& chain, const T (&operand)[max_chain],
                       T value, T& slope) {
  slope = T(1);
#pragma unroll
  for (int k = 0; k < max_chain; ++k)
    if (k < chain.size) {
      const T t = operand[k];
      value = chain_step(chain.op[k], value, t, chain.constant[k]);
      switch (chain.op[k]) {
        case ChainOp::SubFrom:
        case ChainOp::Neg:
          slope = -slope;
          break;
        case ChainOp::Mul:
          slope = slope * t;
          break;
        case ChainOp::Exp:
          slope = slope * value;
          break;
        case ChainOp::Div:
          slope = slope / t;
          break;
        case ChainOp::Keep:
          slope = t != T(0) ? slope : T(0);
          break;
        case ChainOp::Drop:
          slope = t != T(0) ? T(0) : slope;
          break;
        default:
          break;
      }
    }
  return value;
}
// d result / d operand[k] for every op (zero for conditions and for ops
// without an operand), by a reverse sweep over the stored op inputs.
template <typename T>
TP_PAIR_INLINE void operand_slopes(const Chain<T>& chain,
                                   const T (&operand)[max_chain], T value,
                                   T (&slope)[max_chain]) {
  T input[max_chain];
#pragma unroll
  for (int k = 0; k < max_chain; ++k)
    if (k < chain.size) {
      input[k] = value;
      value = chain_step(chain.op[k], value, operand[k], chain.constant[k]);
    }
  T adjoint = T(1);
#pragma unroll
  for (int k = max_chain - 1; k >= 0; --k) {
    slope[k] = T(0);
    if (k >= chain.size)
      continue;
    const T u = input[k], t = operand[k];
    switch (chain.op[k]) {
      case ChainOp::Add:
        slope[k] = adjoint;
        break;
      case ChainOp::Sub:
        slope[k] = -adjoint;
        break;
      case ChainOp::SubFrom:
        slope[k] = adjoint;
        adjoint = -adjoint;
        break;
      case ChainOp::Mul:
        slope[k] = adjoint * u;
        adjoint = adjoint * t;
        break;
      case ChainOp::Neg:
        adjoint = -adjoint;
        break;
      case ChainOp::Exp:
        adjoint = adjoint * ::exp(u);  // the op's output, recomputed
        break;
      case ChainOp::Div:
        // ATen's div backward: -grad * self / (other * other).
        slope[k] = -adjoint * u / (t * t);
        adjoint = adjoint / t;
        break;
      case ChainOp::Keep:
        adjoint = t != T(0) ? adjoint : T(0);
        break;
      case ChainOp::Drop:
        adjoint = t != T(0) ? T(0) : adjoint;
        break;
    }
  }
}

template <Feature F, typename T>
TP_PAIR_INLINE void step(T& s, T x, T y, T p, bool first) {
  if constexpr (F == Feature::Product) {
    s += x * y;
    return;
  }
  T delta = x - y;
  if constexpr (F == Feature::Square)
    s += delta * delta;
  else if constexpr (F == Feature::Abs)
    s += std::fabs(delta);
  else if constexpr (F == Feature::Power)
    s += std::pow(std::fabs(delta), p);
  else {
    T value = std::fabs(delta);
    // _ppminkowski's infinite branch: every feature propagates NaN.
    if (first || std::isnan(value) || value > s)
      s = value;
  }
}

// d feature / d x; kernels take d / d y from the swapped plan. Difference
// features are zero at delta == 0, as native. Square returns delta: native
// doubles the reduced sum (DifferenceTerm, then mul_(2)), so a zero cotangent
// never meets an overflowed 2 * delta.
template <Feature F, typename T>
TP_PAIR_INLINE T dstep(T x, T y, T p) {
  if constexpr (F == Feature::Product)
    return y;
  T delta = x - y;
  if constexpr (F == Feature::Square)
    return delta;
  else if constexpr (F == Feature::Abs)
    return T((delta > 0) - (delta < 0));
  else {
    if (delta == 0)
      return T(0);
    T magnitude = std::fabs(delta);
    T sign = T((delta > 0) - (delta < 0));
    return sign * p * std::pow(magnitude, p - 1);
  }
}

// ATen pow(tensor, scalar) special cases (CUDA pow_tensor_scalar), chosen
// on the host from the double exponent as ATen does.
template <typename T>
TP_PAIR_INLINE T scalar_pow(T base, T exponent, int code) {
  if (code == 1)
    return std::sqrt(base);
  if (code == 2)
    return base * base;
  if (code == 3)
    return base * base * base;
  return std::pow(base, exponent);
}

template <typename T>
TP_PAIR_INLINE T masked_sqrt(T s) {
  return s != T(0) ? std::sqrt(s) : T(0);
}

// Matern radius r = min(sqrt(s) * (1/l) * sqrt(2 nu), clamp).
template <typename T>
TP_PAIR_INLINE T matern_radius(const OuterParams<T>& o, T s) {
  T r = mul_rn(mul_rn(masked_sqrt(s), o.a), o.b);
  return r > o.c ? o.c : r;  // clamp_max keeps NaN
}

template <typename T>
TP_PAIR_INLINE T outer(const OuterParams<T>& o, T s) {
  switch (o.kind) {
    case Outer::Identity:
      return s;
    case Outer::Sqrt:
      return masked_sqrt(s);
    case Outer::Root:
      return s != T(0) ? scalar_pow(s, o.a, o.code) : T(0);
    case Outer::ScaledExp:
      return std::exp(mul_rn(s, o.a));
    case Outer::Matern05:
      return std::exp(-matern_radius(o, s));
    case Outer::Matern15: {
      T r = matern_radius(o, s);
      return mul_rn(add_rn(r, T(1)), std::exp(-r));
    }
    case Outer::Matern25: {
      T r = matern_radius(o, s);
      T polynomial = add_rn(add_rn(r, T(1)), mul_rn(mul_rn(r, r), o.d));
      return mul_rn(polynomial, std::exp(-r));
    }
    case Outer::RationalQuadratic: {
      // exp(-alpha * log1p(s * (1/l) * (1/l) * .5 * (1/alpha)))
      T z = mul_rn(mul_rn(mul_rn(mul_rn(s, o.a), o.a), T(.5)), o.b);
      return std::exp(mul_rn(std::log1p(z), o.c));
    }
    case Outer::Polynomial:
      return scalar_pow(add_rn(mul_rn(s, o.a), o.b), o.c, o.code);
    case Outer::Sigmoid:
      return std::tanh(add_rn(mul_rn(s, o.a), o.b));
  }
  return s;
}

// d outer / d s at (s, value = outer(s)). Zero where native's masked forms
// have a zero subgradient (sqrt and root at s == 0, clamped Matern radius).
template <typename T>
TP_PAIR_INLINE T douter(const OuterParams<T>& o, T s, T value) {
  switch (o.kind) {
    case Outer::Identity:
      return T(1);
    case Outer::Sqrt:
      return s != T(0) ? T(.5) / value : T(0);
    case Outer::Root:
      // value / s first: a * value can overflow where a * s^(a-1) does not.
      return s != T(0) ? o.a * (value / s) : T(0);
    case Outer::ScaledExp:
      return o.a * value;
    case Outer::Matern05:
    case Outer::Matern15:
    case Outer::Matern25: {
      if (s == T(0))
        return T(0);
      T root = std::sqrt(s);
      T r = root * o.a * o.b;
      if (r > o.c)
        return T(0);
      // dr/ds = a b / (2 sqrt(s)); dk/dr per nu.
      T e = std::exp(-r), dk;
      if (o.kind == Outer::Matern05)
        dk = -e;
      else if (o.kind == Outer::Matern15)
        dk = -r * e;
      else
        dk = -r * (T(1) + r) * e * o.d;
      return dk * o.a * o.b / (T(2) * root);
    }
    case Outer::RationalQuadratic: {
      T z = s * o.a * o.a * T(.5) * o.b;
      // d/ds exp(c log1p(z)) = value * c / (1 + z) * dz/ds, with dz/ds
      // applied factor by factor as autograd does: a * a alone can overflow.
      return value * o.c / (T(1) + z) * o.b * T(.5) * o.a * o.a;
    }
    case Outer::Polynomial: {
      // ATen's pow backward, exponent * z^(exponent - 1), then the gamma mul.
      T z = add_rn(mul_rn(s, o.a), o.b);
      T power = o.c == T(1)   ? T(1)
                : o.c == T(2) ? z
                : o.c == T(3) ? z * z
                              : std::pow(z, o.c - T(1));
      return o.c * power * o.a;
    }
    case Outer::Sigmoid:
      return (T(1) - value * value) * o.a;  // ATen's tanh backward
  }
  return T(1);
}
}  // namespace torchpairwise::pairs
