#ifndef QR_DECOMPOSITION_H
#define QR_DECOMPOSITION_H

#include "core_types.h"
#include "householder.h"
#include "matrix.h"
#include "matrix_types.h"
#include "matrix_view.h"
#include "scalar_types.h"
#include "submatrix_range.h"

namespace linalg {

template <detail::FloatingOrComplexType Scalar>
struct QRDecompositionResult {
  Matrix<Scalar> q;
  Matrix<Scalar> r;
};

// TODO: Use MatrixView instead of MutableMatrixType concept.
template <detail::MatrixType MatrixT>
QRDecompositionResult<typename MatrixT::value_type> QRDecomposition(const MatrixT& matrix) {
  using Scalar = typename MatrixT::value_type;

  auto q = Matrix<Scalar>::Identity(matrix.Rows());
  auto r = Matrix<Scalar>{matrix};

  // TODO: +- 1
  for (Index i = 0; i < std::min(matrix.Rows(), matrix.Cols()) - 1; ++i) {
    auto first_vector_subrange =
        SubmatrixRange::FromBeginSize(ERowBegin{i}, ERows{r.Rows() - i}, EColBegin{i}, ECols{1});

    auto reflect_vector = Matrix<Scalar>{r.Submatrix(first_vector_subrange)};
    HouseholderReflectVectorReduce(reflect_vector);

    auto r_submatrix =
        r.Submatrix(SubmatrixRange::FromBeginEnd(ERowBegin{i}, ERowEnd{r.Rows()}, EColBegin{i}, EColEnd{r.Cols()}));
    ApplyHouseholderLeft(r_submatrix, reflect_vector);
    auto q_submatrix =
        q.Submatrix(SubmatrixRange::FromBeginEnd(ERowBegin{i}, ERowEnd{q.Rows()}, EColBegin{0}, EColEnd{q.Cols()}));
    ApplyHouseholderLeft(q_submatrix, reflect_vector);
  }

  q = Matrix<Scalar>{Conjugated(q)};  // such operator= because Conjugated can be MatrixView or Matrix

  return QRDecompositionResult<Scalar>{std::move(q), std::move(r)};
}

// Givens QR decomposition rarely used as it's complexity is worse than Householder QR decomposition.
// Use GivensQRDecomposition only for small or sparse matrices.
template <detail::MatrixType MatrixT>
QRDecompositionResult<typename MatrixT::value_type> GivensQRDecomposition(const MatrixT& matrix) {
  using Scalar = typename MatrixT::value_type;

  auto q = Matrix<Scalar>::Identity(matrix.Rows());
  auto r = Matrix<Scalar>{matrix};

  for (Index j = 0; j < std::min(matrix.Rows(), matrix.Cols()); ++j) {
    for (Index i = matrix.Rows() - 1; i > j; --i) {
      auto params = GetZeroingGivensRotationParams(r(i - 1, j), r(i, j));
      ApplyGivensRotationLeft(r, params, i - 1, i);
      ApplyGivensRotationLeft(q, params, i - 1, i);
    }
  }

  q = Matrix<Scalar>{Conjugated(q)};  // such operator= because Conjugated can be MatrixView or Matrix
  return QRDecompositionResult<Scalar>{std::move(q), std::move(r)};
}

}  // namespace linalg

#endif  // QR_DECOMPOSITION_H
