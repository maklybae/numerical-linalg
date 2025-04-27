#ifndef BIDIAGONALIZATION_H
#define BIDIAGONALIZATION_H

#include "core_types.h"
#include "householder.h"
#include "matrix.h"
#include "matrix_types.h"
#include "scalar_types.h"
#include "scalar_utils.h"

namespace linalg {

// Invariant: A(m*n) = U(m*m) * B(m*n) * V^T(n*n)
template <detail::FloatingOrComplexType Scalar>
struct BidiagonalizationResult {
  Matrix<Scalar> u;                             // U_1 * ... * U_m - unitary matrix
  Matrix<detail::UnderlyingScalarT<Scalar>> b;  // Upper bidiagonal matrix. Purpose: real SVD Sigma matrix.
  Matrix<Scalar> vt;                            // V_{m-2}^T * ... * V_{1}^T - unitary matrix
};

template <detail::MatrixType MatrixT>
BidiagonalizationResult<typename MatrixT::value_type> GetBidiagonalization(const MatrixT& matrix) {
  using Scalar = typename MatrixT::value_type;

  Matrix<Scalar> ut        = Matrix<Scalar>::Identity(matrix.Rows());
  Matrix<Scalar> v         = Matrix<Scalar>::Identity(matrix.Cols());
  Matrix<Scalar> b_noncast = Matrix<Scalar>{matrix};

  for (Index i = 0; i < std::min(b_noncast.Rows(), b_noncast.Cols()); ++i) {
    // Column reduction (same as QR decomposition)
    auto reflect_colvector = Matrix<Scalar>{b_noncast.Submatrix(
        SubmatrixRange::FromBeginSize(ERowBegin{i}, ERows{b_noncast.Rows() - i}, EColBegin{i}, ECols{1}))};
    HouseholderReflectVectorReduce(reflect_colvector);

    auto b_noncast_submatrix = b_noncast.Submatrix(
        SubmatrixRange::FromBeginEnd(ERowBegin{i}, ERowEnd{b_noncast.Rows()}, EColBegin{i}, EColEnd{b_noncast.Cols()}));
    ApplyHouseholderLeft(b_noncast_submatrix, reflect_colvector);

    auto ut_submatrix =
        ut.Submatrix(SubmatrixRange::FromBeginEnd(ERowBegin{i}, ERowEnd{ut.Rows()}, EColBegin{0}, EColEnd{ut.Cols()}));
    ApplyHouseholderLeft(ut_submatrix, reflect_colvector);

    if constexpr (detail::kIsComplexV<Scalar>) {
      if (!detail::ApproxZero(b_noncast(i, i))) {
        auto ut_row        = ut.Row(i);
        auto b_noncast_row = b_noncast.Row(i);
        auto mult          = std::conj(b_noncast(i, i)) / std::abs(b_noncast(i, i));
        ut_row *= mult;
        b_noncast_row *= mult;
      }
    }

    // Row reduction only if not the last column
    if (i < b_noncast.Cols() - 1) {
      auto reflect_rowvector            = Matrix<Scalar>{b_noncast.Submatrix(
          SubmatrixRange::FromBeginSize(ERowBegin{i}, ERows{1}, EColBegin{i + 1}, ECols{b_noncast.Cols() - i - 1}))};
      auto reflect_rowvector_transposed = Transposed(reflect_rowvector);
      HouseholderReflectVectorReduce(reflect_rowvector_transposed);

      b_noncast_submatrix = b_noncast.Submatrix(SubmatrixRange::FromBeginEnd(
          ERowBegin{i}, ERowEnd{b_noncast.Rows()}, EColBegin{i + 1}, EColEnd{b_noncast.Cols()}));
      ApplyHouseholderRight(b_noncast_submatrix, reflect_rowvector_transposed);

      auto v_submatrix = v.Submatrix(
          SubmatrixRange::FromBeginEnd(ERowBegin{0}, ERowEnd{v.Rows()}, EColBegin{i + 1}, EColEnd{v.Cols()}));
      ApplyHouseholderRight(v_submatrix, reflect_rowvector_transposed);

      if constexpr (detail::kIsComplexV<Scalar>) {
        if (!detail::ApproxZero(b_noncast(i, i + 1))) {
          auto v_col         = v.Col(i + 1);
          auto b_noncast_col = b_noncast.Col(i + 1);
          auto mult          = std::conj(b_noncast(i, i + 1)) / std::abs(b_noncast(i, i + 1));
          v_col *= mult;
          b_noncast_col *= mult;
        }
      }
    }
  }

  // Do not use move as construct result from rvalues
  return {Matrix<Scalar>{Conjugated(ut)}, detail::CastToUnderlyingScalarMatrix(b_noncast),
          Matrix<Scalar>{Conjugated(v)}};
}

}  // namespace linalg

#endif  // BIDIAGONALIZATION_H
