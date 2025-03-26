#ifndef HESSENBERG_H
#define HESSENBERG_H

#include "householder.h"
#include "matrix.h"
#include "matrix_types.h"

namespace linalg {

// Q*_n * ... * Q*_1 * A * Q_1 * ... * Q_n = H
struct HessenbergFormResult {
  Matrix<long double> q;  // Unitary matrix (Q_1 * ... * Q_n)
  Matrix<long double> h;  // Hessenberg matrix
};

template <detail::MatrixType MatrixT>
HessenbergFormResult GetHeisenbergForm(const MatrixT& matrix) {
  using Scalar = typename MatrixT::value_type;

  // TODO: square matrix assert

  auto q = Matrix<Scalar>::Identity(matrix.Rows());
  auto h = Matrix<Scalar>{matrix};

  for (Index i = 0; i < matrix.Rows() - 2; ++i) {
    auto first_vector_subrange =
        SubmatrixRange::FromBeginSize(ERowBegin{i + 1}, ERows{h.Rows() - i - 1}, EColBegin{i}, ECols{1});

    auto reflect_vector = Matrix<Scalar>{h.Submatrix(first_vector_subrange)};
    HouseholderReflectVectorReduce(reflect_vector);

    // Apply reflector for unitary matrix (left)
    auto q_submatrix =
        q.Submatrix(SubmatrixRange::FromBeginEnd(ERowBegin{i + 1}, ERowEnd{q.Rows()}, EColBegin{0}, EColEnd{q.Cols()}));
    ApplyHouseholderLeft(q_submatrix, reflect_vector);

    // Apply reflector for Hessenberg matrix (left + right)
    auto h_left_submatrix =
        h.Submatrix(SubmatrixRange::FromBeginEnd(ERowBegin{i + 1}, ERowEnd{q.Rows()}, EColBegin{i}, EColEnd{q.Cols()}));
    ApplyHouseholderLeft(h_left_submatrix, reflect_vector);

    auto h_right_submatrix =
        h.Submatrix(SubmatrixRange::FromBeginEnd(ERowBegin{0}, ERowEnd{q.Rows()}, EColBegin{i + 1}, EColEnd{q.Cols()}));
    ApplyHouseholderRight(h_right_submatrix, reflect_vector);
  }

  q = Matrix<Scalar>{Conjugated(q)};  // such operator= because Conjugated can be MatrixView or Matrix

  return HessenbergFormResult{std::move(q), std::move(h)};
}

}  // namespace linalg

#endif  // HESSENBERG_H
