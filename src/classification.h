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
bool IsUpperTriangular(const MatrixT& matrix, detail::UnderlyingScalarT<typename MatrixT::value_type> eps =
                                                  detail::kEpsilon<typename MatrixT::value_type>) {
  for (Size i = 0; i < matrix.Rows(); ++i) {
    for (Size j = i + 1; j < matrix.Cols(); ++j) {
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

template <detail::MatrixType MatrixT>
bool IsBidiagonal(const MatrixT& matrix, detail::UnderlyingScalarT<typename MatrixT::value_type> eps =
                                             detail::kEpsilon<typename MatrixT::value_type>) {
  for (Size i = 0; i < matrix.Rows(); ++i) {
    for (Size j = 0; j < matrix.Cols(); ++j) {
      if (i != j && i + 1 != j && !detail::ApproxZero(matrix(i, j), eps)) {
        return false;
      }
    }
  }
  return true;
}

template <detail::MatrixType MatrixT>
bool IsSquare(const MatrixT& matrix) {
  return matrix.Rows() == matrix.Cols();
}

template <detail::MatrixType MatrixT>
bool IsIdentity(const MatrixT& matrix, detail::UnderlyingScalarT<typename MatrixT::value_type> eps =
                                           detail::kEpsilon<typename MatrixT::value_type>) {

  using Scalar = typename MatrixT::value_type;
  Scalar one   = Scalar{1};

  if (!IsSquare(matrix)) {
    return false;
  }

  for (Size i = 0; i < matrix.Rows(); ++i) {
    for (Size j = 0; j < matrix.Cols(); ++j) {
      if (i == j && !detail::ApproxEqual(matrix(i, j), one, eps)) {
        return false;
      }
      if (i != j && !detail::ApproxZero(matrix(i, j), eps)) {
        return false;
      }
    }
  }
  return true;
}

template <detail::MatrixType MatrixT>
bool IsUnitary(const MatrixT& matrix, detail::UnderlyingScalarT<typename MatrixT::value_type> eps =
                                          detail::kEpsilon<typename MatrixT::value_type>) {
  auto conj  = Conjugated(matrix);
  auto prod1 = conj * matrix;
  auto prod2 = matrix * conj;
  return IsIdentity(prod1, eps) && IsIdentity(prod2, eps);
}

template <detail::MatrixType MatrixT>
bool IsHermitian(const MatrixT& matrix, detail::UnderlyingScalarT<typename MatrixT::value_type> eps =
                                            detail::kEpsilon<typename MatrixT::value_type>) {
  if (!IsSquare(matrix)) {
    return false;
  }

  auto conj = Conjugated(matrix);
  return detail::ApproxEqual(matrix, conj, eps);
}

template <detail::MatrixType MatrixT>
bool IsHessenberg(const MatrixT& matrix, detail::UnderlyingScalarT<typename MatrixT::value_type> eps =
                                             detail::kEpsilon<typename MatrixT::value_type>) {
  if (!IsSquare(matrix)) {
    return false;
  }

  for (Size i = 0; i < matrix.Rows(); ++i) {
    for (Size j = 0; j < matrix.Cols(); ++j) {
      if (i - j >= 2 && !detail::ApproxZero(matrix(i, j), eps)) {
        return false;
      }
    }
  }
  return true;
}

template <detail::MatrixType MatrixT>
bool IsZero(const MatrixT& matrix, detail::UnderlyingScalarT<typename MatrixT::value_type> eps =
                                       detail::kEpsilon<typename MatrixT::value_type>) {
  for (Size i = 0; i < matrix.Rows(); ++i) {
    for (Size j = 0; j < matrix.Cols(); ++j) {
      if (!detail::ApproxZero(matrix(i, j), eps)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace linalg

#endif  // CLASSIFICATION_H
