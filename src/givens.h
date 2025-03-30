#ifndef GIVENS_H
#define GIVENS_H

#include <cassert>
#include <complex>
#include <concepts>

#include "matrix_types.h"
#include "scalar_types.h"
#include "scalar_utils.h"

namespace linalg {

template <detail::FloatingOrComplexType Scalar>
struct GivensRotationParams {
  Scalar c;  // cos
  Scalar s;  // sin
};

template <std::floating_point Scalar>
GivensRotationParams<Scalar> GetZeroingGivensRotationParams(Scalar a, Scalar b) {
  // TODO ?: assert a != 0 || b != 0 (rotation is useless if both are zero)

  if (detail::ApproxZero(b)) {
    return {Scalar{1}, Scalar{0}};
  }

  if (detail::ApproxZero(a)) {
    return {Scalar{0}, Scalar{1}};
  }

  if (std::abs(b) >= std::abs(a)) {
    auto t = -a / b;
    auto s = Scalar{1} / std::sqrt(Scalar{1} + t * t);
    return {s * t, s};
  }

  auto t = -b / a;
  auto c = Scalar{1} / std::sqrt(Scalar{1} + t * t);
  return {c, c * t};
}

// We will use Givens rotations more with real numbers, since output of our reduction to tridiagonal form, after
// postprocessing, yields a real-valued tridiagonal symmatric matrix.
template <std::floating_point UnderlyingScalar>
GivensRotationParams<std::complex<UnderlyingScalar>> GetZeroingGivensRotationParams(std::complex<UnderlyingScalar> a,
                                                                                    std::complex<UnderlyingScalar> b) {
  auto sqrt_norm = std::sqrt(std::norm(a) + std::norm(b));
  if (detail::ApproxZero(sqrt_norm)) {
    return {std::complex<UnderlyingScalar>{1}, std::complex<UnderlyingScalar>{0}};
  }

  return {std::conj(a) / sqrt_norm, -std::conj(b) / sqrt_norm};
}

template <detail::MutableMatrixType MatrixT>
void ApplyGivensRotationLeft(MatrixT& matrix, GivensRotationParams<typename MatrixT::value_type> params, Index row_i,
                             Index row_j) {
  using Scalar = typename MatrixT::value_type;

  assert(row_i != row_j && "Rows should be different");
  assert(row_i < matrix.Rows() && row_j < matrix.Rows() && "Rows should be in range");

  for (Index j = 0; j < matrix.Cols(); ++j) {
    Scalar tmp_i = matrix(row_i, j);
    Scalar tmp_j = matrix(row_j, j);

    if constexpr (detail::kIsComplexV<Scalar>) {
      matrix(row_i, j) = std::conj(params.c) * tmp_i - params.s * tmp_j;
      matrix(row_j, j) = std::conj(params.s) * tmp_i + std::conj(params.c) * tmp_j;
    } else {
      matrix(row_i, j) = params.c * tmp_i - params.s * tmp_j;
      matrix(row_j, j) = params.s * tmp_i + params.c * tmp_j;
    }
  }
}

}  // namespace linalg

#endif  // GIVENS_H
