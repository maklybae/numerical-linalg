#ifndef GIVENS_H
#define GIVENS_H

#include <cassert>
#include <complex>

#include "matrix_types.h"
#include "scalar_types.h"
#include "scalar_utils.h"

namespace linalg {

template <detail::FloatingOrComplexType Scalar>
struct GivensRotationParams {
  Scalar c;  // cos
  Scalar s;  // sin
};

// We will use Givens rotations more with real numbers, since output of our reduction to tridiagonal form, after
// postprocessing, yields a real-valued tridiagonal symmatric matrix.
template <detail::FloatingOrComplexType Scalar>
GivensRotationParams<Scalar> GetZeroingFirstGivensRotationParams(Scalar a, Scalar b) {
  auto sqrt_norm = std::sqrt(std::norm(a) + std::norm(b));
  if (detail::ApproxZero(sqrt_norm)) {
    return {Scalar{1}, Scalar{0}};
  }

  return {a / sqrt_norm, -b / sqrt_norm};
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
      matrix(row_i, j) = std::conj(params.c) * tmp_i - std::conj(params.s) * tmp_j;
      matrix(row_j, j) = params.s * tmp_i + params.c * tmp_j;
    } else {
      matrix(row_i, j) = params.c * tmp_i - params.s * tmp_j;
      matrix(row_j, j) = params.s * tmp_i + params.c * tmp_j;
    }
  }
}

template <detail::MutableMatrixType MatrixT>
void ApplyGivensRotationRight(MatrixT& matrix, GivensRotationParams<typename MatrixT::value_type> params, Index col_i,
                              Index col_j) {
  using Scalar = typename MatrixT::value_type;

  assert(col_i != col_j && "Columns should be different");
  assert(col_i < matrix.Cols() && col_j < matrix.Cols() && "Columns should be in range");

  for (Index i = 0; i < matrix.Rows(); ++i) {
    Scalar tmp_i = matrix(i, col_i);
    Scalar tmp_j = matrix(i, col_j);

    if constexpr (detail::kIsComplexV<Scalar>) {
      matrix(i, col_i) = std::conj(params.c) * tmp_i - std::conj(params.s) * tmp_j;
      matrix(i, col_j) = params.s * tmp_i + params.c * tmp_j;
    } else {
      matrix(i, col_i) = params.c * tmp_i - params.s * tmp_j;
      matrix(i, col_j) = params.s * tmp_i + params.c * tmp_j;
    }
  }
}
}  // namespace linalg

#endif  // GIVENS_H
