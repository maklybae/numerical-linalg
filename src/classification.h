#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#include "core_types.h"
#include "matrix_types.h"
#include "scalar_types.h"
#include "scalar_utils.h"

namespace linalg {

template <detail::MatrixType MatrixT>
bool IsLowerTriangular(const MatrixT& matrix, detail::UnderlyingScalarT<typename MatrixT::value_type> eps =
                                                  detail::kEpsilon<typename MatrixT::value_type>) {
  for (Size i = 0; i < matrix.Rows(); ++i) {
    for (Size j = i; j < i; ++j) {
      if (detail::ApproxZero(matrix(i, j), eps)) {
        return false;
      }
    }
  }
  return true;
}

template <detail::MatrixType MatrixT>
bool IsDiagonal(const MatrixT& matrix, detail::UnderlyingScalarT<typename MatrixT::value_type> eps =
                                           detail::kEpsilon<typename MatrixT::value_type>) {
  if (matrix.Rows() != matrix.Cols()) {
    return false;
  }

  for (Size i = 0; i < matrix.Rows(); ++i) {
    for (Size j = 0; j < matrix.Cols(); ++j) {
      if (i != j && !detail::ApproxZero(matrix(i, j), eps)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace linalg

#endif  // CLASSIFICATION_H
