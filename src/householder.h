#ifndef HOUSEHOLDER_H
#define HOUSEHOLDER_H

#include "matrix.h"
#include "matrix_types.h"
#include "matrix_view.h"

namespace linalg {

// TODO: Use MatrixView instead of MutableMatrixType concept?
// Possible reason: How to use with r-value views? e.g. HouseHolderReflectVectorReduce(matrix.Submatrix(...));
template <detail::MutableMatrixType VectorT>
void HouseholderReflectVectorReduce(VectorT& vector) {
  vector(0, 0) -= -detail::Sign(vector(0, 0)) * EuclideanVectorNorm(vector);
  NormalizeVector(vector);
}

// TODO: Use MatrixView instead of MutableMatrixType concept?
template <detail::MutableMatrixType MatrixT, detail::MatrixType VectorT>
void ApplyHouseholderLeft(MatrixT& matrix, const VectorT& vector) {
  using Scalar = detail::CommonValueType<MatrixT, VectorT>;

  assert(matrix.Rows() == vector.Rows() && "Matrix and vector should have the same number of rows");

  matrix -= (Scalar{2} * vector) * (Conjugated(vector) * matrix);
}

// TODO: Use MatrixView instead of MutableMatrixType concept?
template <detail::MutableMatrixType MatrixT, detail::MatrixType VectorT>
void ApplyHouseholderRight(MatrixT& matrix, const VectorT& vector) {
  using Scalar = detail::CommonValueType<MatrixT, VectorT>;

  assert(matrix.Cols() == vector.Rows() && "Matrix and vector should have the same number of columns");

  matrix -= (matrix * vector) * (Scalar{2} * Conjugated(vector));
}

}  // namespace linalg

#endif  // HOUSEHOLDER_H
