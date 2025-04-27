#ifndef SVD_H
#define SVD_H

#include "classification.h"
#include "core_types.h"
#include "givens.h"
#include "matrix.h"
#include "matrix_types.h"
#include "matrix_view.h"
#include "scalar_types.h"
#include "scalar_utils.h"
#include "shifts.h"

namespace linalg {

template <detail::FloatingOrComplexType Scalar>
struct SVDResult {
  Matrix<Scalar> u;                             // Unitary matrix
  Matrix<detail::UnderlyingScalarT<Scalar>> s;  // Diagonal matrix of singular values
  Matrix<Scalar> vt;                            // Unitary matrix (transposed)
};

namespace detail {

template <FloatingOrComplexType Scalar>
struct ImplicitShiftedQRAlgorithmResult {
  Matrix<Scalar> u;                     // Unitary matrix
  Matrix<UnderlyingScalarT<Scalar>> d;  // Diagonal matrix of eigenvalues (not sorted, intermediate result)
  Matrix<Scalar> vt;                    // Unitary matrix (transposed)
};

// Find the closest EV (Wilkinson shift) of B^TB by bidiagonalized matrix B
template <MatrixType MatrixT>
typename MatrixT::value_type GetWilkinsonImplicitShift(const MatrixT& matrix) {
  assert(matrix.Rows() >= 2 && matrix.Cols() >= 2 && "Matrix must be at least 2x2");

  auto bidiag_square_submatrix = matrix.Submatrix(
      SubmatrixRange::FromBeginSize(ERowBegin{matrix.Rows() - 2}, ERows{2}, EColBegin{matrix.Cols() - 2}, ECols{2}));
  auto btb_square_submatrix = Transposed(bidiag_square_submatrix) * bidiag_square_submatrix;

  if (matrix.Cols() >= 3) {
    btb_square_submatrix(0, 0) +=
        matrix(matrix.Rows() - 3, matrix.Cols() - 2) * matrix(matrix.Rows() - 3, matrix.Cols() - 2);
  }

  return GetWilkinsonShift(btb_square_submatrix);
}

template <MatrixType MatrixT>
UnderlyingScalarT<typename MatrixT::value_type> GetImplicitShiftedQRAlgorithmEps(const MatrixT& matrix) {
  using Scalar = UnderlyingScalarT<typename MatrixT::value_type>;

  Scalar eps = 0;
  for (Size i = 0; i < matrix.Cols() - 1; ++i) {
    eps = std::max(eps, std::abs(matrix(i, i)) + std::abs(matrix(i, i + 1)));
  }
  return eps * kEpsilon<Scalar>;
}

// Use only for SVD
template <MatrixType MatrixT>
ImplicitShiftedQRAlgorithmResult<typename MatrixT::value_type> ImplicitShiftedQRAlgorithm(const MatrixT& matrix,
                                                                                          Size it_per_vec = 1000) {
  using Scalar = typename MatrixT::value_type;

  Matrix<Scalar> d  = Matrix<Scalar>{matrix};
  Matrix<Scalar> u  = Matrix<Scalar>::Identity(d.Rows());
  Matrix<Scalar> vt = Matrix<Scalar>::Identity(d.Cols());

  if (d.Cols() == 1) {
    return {std::move(u), std::move(d), std::move(vt)};
  }

  for (Size it = 0; it < it_per_vec * matrix.Cols(); ++it) {
    // Convergence epsilon check
    auto eps = GetImplicitShiftedQRAlgorithmEps(d);

    if (IsDiagonal(d, eps)) {
      return {std::move(u), std::move(d), std::move(vt)};
    }

    // Splitting and cancellation
    for (Size j = 0; j < d.Cols() - 1; ++j) {
      // Check for cancellation
      if (ApproxZero(d(j, j))) {
        for (Size i = j + 1; i < std::min(d.Rows(), d.Cols()); ++i) {
          auto params = GetZeroingGivensRotationParams(d(i, i), d(j, i));
          ApplyGivensRotationLeft(d, params, i, j);
          ApplyGivensRotationRight(u, params, i, j);
        }
      }

      // Check for splitting (also splits after cancellation because cancellation zeros d(j, j + 1))
      if (ApproxZero(d(j, j + 1))) {
        auto submatrix1 =
            d.Submatrix(SubmatrixRange::FromBeginEnd(ERowBegin{0}, ERowEnd{j + 1}, EColBegin{0}, EColEnd{j + 1}));
        auto submatrix2 = d.Submatrix(
            SubmatrixRange::FromBeginEnd(ERowBegin{j + 1}, ERowEnd{d.Rows()}, EColBegin{j + 1}, EColEnd{d.Cols()}));
        auto res1 = ImplicitShiftedQRAlgorithm(submatrix1, it_per_vec);  // TODO: How to count iterations?
        auto res2 = ImplicitShiftedQRAlgorithm(submatrix2, it_per_vec);

        // Form U matrix from two parts
        Matrix<Scalar> common_u  = Matrix<Scalar>::Identity(d.Rows());
        auto common_u_submatrix1 = common_u.Submatrix(
            SubmatrixRange::FromBeginEnd(ERowBegin{0}, ERowEnd{j + 1}, EColBegin{0}, EColEnd{j + 1}));
        CopyMatrix(res1.u, common_u_submatrix1);
        auto common_u_submatrix2 = common_u.Submatrix(
            SubmatrixRange::FromBeginEnd(ERowBegin{j + 1}, ERowEnd{u.Rows()}, EColBegin{j + 1}, EColEnd{u.Cols()}));
        CopyMatrix(res2.u, common_u_submatrix2);

        // Form V^T matrix from two parts
        Matrix<Scalar> common_vt  = Matrix<Scalar>::Identity(d.Cols());
        auto common_vt_submatrix1 = common_vt.Submatrix(
            SubmatrixRange::FromBeginEnd(ERowBegin{0}, ERowEnd{j + 1}, EColBegin{0}, EColEnd{j + 1}));
        CopyMatrix(res1.vt, common_vt_submatrix1);
        auto common_vt_submatrix2 = common_vt.Submatrix(
            SubmatrixRange::FromBeginEnd(ERowBegin{j + 1}, ERowEnd{vt.Rows()}, EColBegin{j + 1}, EColEnd{vt.Cols()}));
        CopyMatrix(res2.vt, common_vt_submatrix2);

        // Form D matrix from two parts
        Matrix<Scalar> common_d  = Matrix<Scalar>::Zero(ERows{d.Rows()}, ECols{d.Cols()});
        auto common_d_submatrix1 = common_d.Submatrix(
            SubmatrixRange::FromBeginEnd(ERowBegin{0}, ERowEnd{j + 1}, EColBegin{0}, EColEnd{j + 1}));
        CopyMatrix(res1.d, common_d_submatrix1);
        auto common_d_submatrix2 = common_d.Submatrix(
            SubmatrixRange::FromBeginEnd(ERowBegin{j + 1}, ERowEnd{d.Rows()}, EColBegin{j + 1}, EColEnd{d.Cols()}));
        CopyMatrix(res2.d, common_d_submatrix2);

        return {std::move(u * common_u), std::move(common_d), std::move(common_vt * vt)};
      }
    }

    // Algorithm step: Implicitly change B^TB
    auto shift = GetWilkinsonImplicitShift(d);
    for (Size j = 0; j < d.Cols() - 1; ++j) {
      auto first_rotation = j == 0 ? GetZeroingGivensRotationParams(d(0, 0) * d(0, 0) - shift, d(0, 0) * d(0, 1))
                                   : GetZeroingGivensRotationParams(d(j - 1, j), d(j - 1, j + 1));

      ApplyGivensRotationRight(d, first_rotation, j, j + 1);
      ApplyGivensRotationLeft(vt, first_rotation, j, j + 1);

      auto second_rotation = GetZeroingGivensRotationParams(d(j, j), d(j + 1, j));
      ApplyGivensRotationLeft(d, second_rotation, j, j + 1);
      ApplyGivensRotationRight(u, second_rotation, j, j + 1);
    }
  }

  return {std::move(u), std::move(d), std::move(vt)};
}

template <FloatingOrComplexType Scalar>
void FixNegativeDiagonal(SVDResult<Scalar>& svd_res) {
  for (Size i = 0; i < std::min(svd_res.s.Rows(), svd_res.s.Cols()); ++i) {
    if (svd_res.s(i, i) < 0) {
      svd_res.s(i, i) = -svd_res.s(i, i);
      auto row        = svd_res.vt.Row(i);
      row *= -1;
    }
  }
}

template <FloatingOrComplexType Scalar>
void FixOrder(SVDResult<Scalar>& svd_res) {
  for (Size i = 1; i < std::min(svd_res.s.Rows(), svd_res.s.Cols()); ++i) {
    for (Size j = i; j > 0 && svd_res.s(j, j) > svd_res.s(j - 1, j - 1); --j) {
      std::swap(svd_res.s(j, j), svd_res.s(j - 1, j - 1));

      auto temp_u = svd_res.u.Col(j);
      svd_res.u.Col(j - 1).SwapElements(temp_u);

      auto temp_vt = svd_res.vt.Row(j);
      svd_res.vt.Row(j - 1).SwapElements(temp_vt);
    }
  }
}

}  // namespace detail

template <detail::MatrixType MatrixT>
SVDResult<typename MatrixT::value_type> SVD(const MatrixT& matrix, Size it_per_vec = 1000) {
  if (matrix.Rows() < matrix.Cols()) {
    auto res = SVD(Conjugated(matrix), it_per_vec);

    return {Matrix{Conjugated(res.vt)}, std::move(res.s), Matrix{Conjugated(res.u)}};
  }

  auto bidiag_res = GetBidiagonalization(matrix);
  auto qr_res     = detail::ImplicitShiftedQRAlgorithm(bidiag_res.b, it_per_vec);

  auto vt = qr_res.vt * bidiag_res.vt;
  auto u  = bidiag_res.u * qr_res.u;

  auto svd_res = SVDResult{std::move(u), std::move(qr_res.d), std::move(vt)};
  detail::FixNegativeDiagonal(svd_res);
  detail::FixOrder(svd_res);

  return svd_res;
}

}  // namespace linalg

#endif  // SVD_H
