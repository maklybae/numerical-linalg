#ifndef QR_ALGORITHM_H
#define QR_ALGORITHM_H

#include "classification.h"
#include "core_types.h"
#include "matrix.h"
#include "matrix_types.h"
#include "scalar_types.h"
#include "shifts.h"

namespace linalg {

template <detail::FloatingOrComplexType Scalar>
struct EigenDecompositionResult {
  Matrix<Scalar> q;  // Eigenvectors
  Matrix<Scalar> d;  // Diagonal matrix of eigenvalues
};

// Use QR algorithm to find eigenvalues and eigenvectors of a symmetric matrix
// such that A = Q * D * Q^T, where Q is orthogonal (unitray) (because A is symmetric)
// and D is diagonal matrix of eigenvalues.
// TODO: add shift function
template <detail::MatrixType MatrixT>
EigenDecompositionResult<typename MatrixT::value_type> SymEigenDecomposition(const MatrixT& matrix,
                                                                             Size it_per_vec = 100) {

  assert(IsHermitian(matrix) && "Matrix should be symmetric or hermitian");

  auto [q, d] = GetHessenbergForm(matrix);  // q = Q_1 * ... * Q_n
  for (Size i = 0; i < it_per_vec * matrix.Cols(); ++i) {

    if constexpr (detail::kIsComplexV<typename MatrixT::value_type>) {
      if (IsUpperTriangular(d)) {
        break;
      }
    } else {
      if (IsDiagonal(d)) {
        break;
      }
    }

    auto qr = QRDecomposition(d);
    d       = qr.r * qr.q;
    q *= qr.q;
  }

  detail::FixZeros(q);
  detail::FixZeros(d);
  return {std::move(q), std::move(d)};
}

}  // namespace linalg

#endif  // QR_ALGORITHM_H
