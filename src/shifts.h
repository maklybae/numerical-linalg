#ifndef SHIFTS_H
#define SHIFTS_H

#include <cassert>

#include "core_types.h"
#include "matrix_types.h"
#include "scalar_utils.h"
#include "submatrix_range.h"

namespace linalg {
namespace detail {

template <MatrixType MatrixT>
MatrixT::value_type GetWilkinsonShift(const MatrixT& matrix) {
  using Scalar = typename MatrixT::value_type;

  assert(matrix.Rows() >= 2 && matrix.Cols() >= 2 && "Matrix must be at least 2x2");

  auto square_submatrix = matrix.Submatrix(
      SubmatrixRange::FromBeginSize(ERowBegin{matrix.Rows() - 2}, ERows{2}, EColBegin{matrix.Cols() - 2}, ECols{2}));
  auto delta  = (square_submatrix(0, 0) - square_submatrix(1, 1)) / Scalar{2};
  auto divide = std::abs(delta) + std::sqrt(delta * delta + square_submatrix(0, 1) * square_submatrix(0, 1));
  if (ApproxZero(divide)) {
    return Scalar{0};
  }

  return square_submatrix(1, 1) - Sign(delta) * square_submatrix(0, 1) * square_submatrix(0, 1) / divide;
}

}  // namespace detail
}  // namespace linalg

#endif  // SHIFTS_H
